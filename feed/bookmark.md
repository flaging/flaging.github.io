
## 2021-10-26

### [[2110.12025] Interaction and Conflict Management in AI-assisted Operational Control Loops in 6G](http://arxiv.org/abs/2110.12025)


  This paper studies autonomous and AI-assisted control loops (ACLs) in the
next generation of wireless networks in the lens of multi-agent environments.
We will study the diverse interactions and conflict management among these
loops. We propose "interaction and conflict management" (ICM) modules to
achieve coherent, consistent and interactions among these ACLs. We introduce
three categories of ACLs based on their sizes, their cooperative and
competitive behaviors, and their sharing of datasets and models. These
categories help to introduce conflict resolution and interaction management
mechanisms for ICM. Using Kubernetes, we present an implementation of ICM to
remove the conflicts in the scheduling and rescheduling of Pods for different
ACLs in networks.

    

### [[2110.12038] Characterizing Performance Inequity Across U.S. Ookla Speedtest Users](http://arxiv.org/abs/2110.12038)


  The Internet has become indispensable to daily activities, such as work,
education and health care. Many of these activities require Internet access
data rates that support real-time video conferencing. However, digital
inequality persists across the United States, not only in who has access but in
the quality of that access. Speedtest by Ookla allows users to run network
diagnostic tests to better understand the current performance of their network.
In this work, we leverage an Internet performance dataset from Ookla, together
with an ESRI demographic dataset, to conduct a comprehensive analysis that
characterizes performance differences between Speedtest users across the U.S.
Our analysis shows that median download speeds for Speedtest users can differ
by over 150Mbps between states. Further, there are important distinctions
between user categories. For instance, all but one state showed statistically
significant differences in performance between Speedtest users in urban and
rural areas. The difference also exists in urban areas between high and low
income users in 27 states. Our analysis reveals that states that demonstrate
this disparity in Speedtest results are geographically bigger, more populous
and have a wider dispersion of median household income. We conclude by
highlighting several challenges to the complex problem space of digital
inequality characterization and provide recommendations for furthering research
on this topic.

    

### [[2110.12245] Knowledge Transfer based Radio and Computation Resource Allocation for 5G RAN Slicing](http://arxiv.org/abs/2110.12245)


  To implement network slicing in 5G, resource allocation is a key function to
allocate limited network resources such as radio and computation resources to
multiple slices. However, the joint resource allocation also leads to a higher
complexity in the network management. In this work, we propose a knowledge
transfer based resource allocation (KTRA) method to jointly allocate radio and
computation resources for 5G RAN slicing. Compared with existing works, the
main difference is that the proposed KTRA method has a knowledge transfer
capability. It is designed to use the prior knowledge of similar tasks to
improve performance of the target task, e.g., faster convergence speed or
higher average reward. The proposed KTRA is compared with Qlearning based
resource allocation (QLRA), and KTRA method presents a 18.4% lower URLLC delay
and a 30.1% higher eMBB throughput as well as a faster convergence speed.

    

### [[2102.04196] Challenges in Net Neutrality Violation Detection: A Case Study of Wehe Tool and Improvements](http://arxiv.org/abs/2102.04196)


  We consider the problem of detecting deliberate traffic discrimination on the
Internet. Given the complex nature of the Internet, detection of deliberate
discrimination is not easy to detect, and tools developed so far suffer from
various limitations. We study challenges in detecting the violations (focusing
on the HTTPS traffic) and discuss possible mitigation approaches. We focus on
`Wehe,' the most recent tool developed to detect net-neutrality violations.
Wehe hosts traffic from all services of interest in a common server and replays
them to mimic the behavior of the traffic from original servers. Despite Wehe's
vast utility and possible influences over policy decisions, its mechanisms are
not yet validated by others. In this work, we highlight critical weaknesses in
Wehe where its replay traffic is not being correctly classified as intended
services by the network middleboxes. We validate this observation using a
commercial traffic shaper. We propose a new method in which the SNI parameter
is set appropriately in the initial TLS handshake to overcome this weakness.
Using commercial traffic shapers, we validate that SNI makes the replay traffic
gets correctly classified as the intended traffic by the middleboxes. Our new
approach thus provides a more realistic method for detecting neutrality
violations of HTTPS traffic.

    

### [[2105.09463] Stochastic Buffer-Aided Relay-Assisted MEC in Time-Slotted Systems](http://arxiv.org/abs/2105.09463)


  Mobile Edge Computing (MEC) has attracted significant research efforts in the
recent years. However, these works consider mostly the computation resources
located at the cloud centers and wireless access nodes, ignoring the
possibility of utilizing server-empowered relays to improve the performance. In
this paper, we study stochastic relay-assisted MEC in systems with discrete
transmission time-line and block fading wireless channels. In order to clearly
identify and inspect the fundamental affecting factors, we investigate the
building block of this architecture, namely a hierarchical network consisting
of a source, a buffer-and-server-aided relay and another higher-level computing
node. We provide a framework to take into account the effects of the fading
channels, the task arrival dynamics as well as the queuing delays in both the
transmission and computation buffers, which facilitates the derivation of the
expression for the Average Response Time (ART). Based on that and the system
average power consumption in each slot, we introduce the concept of Average
Response Energy (ARE) as a novel metric to capture the energy efficiency in MEC
while considering the stochastic nature of the system parameters. Accordingly,
we propose two offloading schemes with their respective problem formulations,
namely the Minimum ART (MART) and the Minimum ARE (MARE) schemes, to optimize
the transmission power and task assignment probability while keeping the system
queues stable. We demonstrate the difference of the formulated problems with
the relevant problem in a recent work, analyze the properties of the problems
and noting them, we propose effective solution methods. Using extensive
simulations, we validate the presented analysis and show the effectiveness of
the proposed schemes in comparison with various baseline methods adapting
existing approaches.

    

### [[2110.11960] ReLACE: Reinforcement Learning Agent for Counterfactual Explanations of Arbitrary Predictive Models](http://arxiv.org/abs/2110.11960)


  The demand for explainable machine learning (ML) models has been growing
rapidly in recent years. Amongst the methods proposed to associate ML model
predictions with human-understandable rationale, counterfactual explanations
are one of the most popular. They consist of post-hoc rules derived from
counterfactual examples (CFs), i.e., modified versions of input samples that
result in alternative output responses from the predictive model to be
explained. However, existing CF generation strategies either exploit the
internals of specific models (e.g., random forests or neural networks), or
depend on each sample's neighborhood, which makes them hard to be generalized
for more complex models and inefficient for larger datasets. In this work, we
aim to overcome these limitations and introduce a model-agnostic algorithm to
generate optimal counterfactual explanations. Specifically, we formulate the
problem of crafting CFs as a sequential decision-making task and then find the
optimal CFs via deep reinforcement learning (DRL) with discrete-continuous
hybrid action space. Differently from other techniques, our method is easily
applied to any black-box model, as this resembles the environment that the DRL
agent interacts with. In addition, we develop an algorithm to extract
explainable decision rules from the DRL agent's policy, so as to make the
process of generating CFs itself transparent. Extensive experiments conducted
on several datasets have shown that our method outperforms existing CF
generation baselines.

    

### [[2110.11987] Improving Robustness of Malware Classifiers using Adversarial Strings Generated from Perturbed Latent Representations](http://arxiv.org/abs/2110.11987)


  In malware behavioral analysis, the list of accessed and created files very
often indicates whether the examined file is malicious or benign. However,
malware authors are trying to avoid detection by generating random filenames
and/or modifying used filenames with new versions of the malware. These changes
represent real-world adversarial examples. The goal of this work is to generate
realistic adversarial examples and improve the classifier's robustness against
these attacks. Our approach learns latent representations of input strings in
an unsupervised fashion and uses gradient-based adversarial attack methods in
the latent domain to generate adversarial examples in the input domain. We use
these examples to improve the classifier's robustness by training on the
generated adversarial set of strings. Compared to classifiers trained only on
perturbed latent vectors, our approach produces classifiers that are
significantly more robust without a large trade-off in standard accuracy.

    

### [[2110.11991] A Reinforcement Learning Approach to Parameter Selection for Distributed Optimization in Power Systems](http://arxiv.org/abs/2110.11991)


  With the increasing penetration of distributed energy resources, distributed
optimization algorithms have attracted significant attention for power systems
applications due to their potential for superior scalability, privacy, and
robustness to a single point-of-failure. The Alternating Direction Method of
Multipliers (ADMM) is a popular distributed optimization algorithm; however,
its convergence performance is highly dependent on the selection of penalty
parameters, which are usually chosen heuristically. In this work, we use
reinforcement learning (RL) to develop an adaptive penalty parameter selection
policy for the AC optimal power flow (ACOPF) problem solved via ADMM with the
goal of minimizing the number of iterations until convergence. We train our RL
policy using deep Q-learning, and show that this policy can result in
significantly accelerated convergence (up to a 59% reduction in the number of
iterations compared to existing, curvature-informed penalty parameter selection
methods). Furthermore, we show that our RL policy demonstrates promise for
generalizability, performing well under unseen loading schemes as well as under
unseen losses of lines and generators (up to a 50% reduction in iterations).
This work thus provides a proof-of-concept for using RL for parameter selection
in ADMM for power systems applications.

    

### [[2110.11999] Machine Learning in Finance-Emerging Trends and Challenges](http://arxiv.org/abs/2110.11999)


  The paradigm of machine learning and artificial intelligence has pervaded our
everyday life in such a way that it is no longer an area for esoteric academics
and scientists putting their effort to solve a challenging research problem.
The evolution is quite natural rather than accidental. With the exponential
growth in processing speed and with the emergence of smarter algorithms for
solving complex and challenging problems, organizations have found it possible
to harness a humongous volume of data in realizing solutions that have
far-reaching business values. This introductory chapter highlights some of the
challenges and barriers that organizations in the financial services sector at
the present encounter in adopting machine learning and artificial
intelligence-based models and applications in their day-to-day operations.

    

### [[2110.12000] Bank transactions embeddings help to uncover current macroeconomics](http://arxiv.org/abs/2110.12000)


  Macroeconomic indexes are of high importance for banks: many risk-control
decisions utilize these indexes. A typical workflow of these indexes evaluation
is costly and protracted, with a lag between the actual date and available
index being a couple of months. Banks predict such indexes now using
autoregressive models to make decisions in a rapidly changing environment.
However, autoregressive models fail in complex scenarios related to appearances
of crises.
We propose to use clients' financial transactions data from a large Russian
bank to get such indexes. Financial transactions are long, and a number of
clients is huge, so we develop an efficient approach that allows fast and
accurate estimation of macroeconomic indexes based on a stream of transactions
consisting of millions of transactions. The approach uses a neural networks
paradigm and a smart sampling scheme.
The results show that our neural network approach outperforms the baseline
method on hand-crafted features based on transactions. Calculated embeddings
show the correlation between the client's transaction activity and bank
macroeconomic indexes over time.

    

### [[2110.12002] Fairness in Missing Data Imputation](http://arxiv.org/abs/2110.12002)


  Missing data are ubiquitous in the era of big data and, if inadequately
handled, are known to lead to biased findings and have deleterious impact on
data-driven decision makings. To mitigate its impact, many missing value
imputation methods have been developed. However, the fairness of these
imputation methods across sensitive groups has not been studied. In this paper,
we conduct the first known research on fairness of missing data imputation. By
studying the performance of imputation methods in three commonly used datasets,
we demonstrate that unfairness of missing value imputation widely exists and
may be associated with multiple factors. Our results suggest that, in practice,
a careful investigation of related factors can provide valuable insights on
mitigating unfairness associated with missing data imputation.

    

### [[2110.12003] Embracing advanced AI/ML to help investors achieve success: Vanguard Reinforcement Learning for Financial Goal Planning](http://arxiv.org/abs/2110.12003)


  In the world of advice and financial planning, there is seldom one right
answer. While traditional algorithms have been successful in solving linear
problems, its success often depends on choosing the right features from a
dataset, which can be a challenge for nuanced financial planning scenarios.
Reinforcement learning is a machine learning approach that can be employed with
complex data sets where picking the right features can be nearly impossible. In
this paper, we will explore the use of machine learning for financial
forecasting, predicting economic indicators, and creating a savings strategy.
Vanguard ML algorithm for goals-based financial planning is based on deep
reinforcement learning that identifies optimal savings rates across multiple
goals and sources of income to help clients achieve financial success. Vanguard
learning algorithms are trained to identify market indicators and behaviors too
complex to capture with formulas and rules, instead, it works to model the
financial success trajectory of investors and their investment outcomes as a
Markov decision process. We believe that reinforcement learning can be used to
create value for advisors and end-investors, creating efficiency, more
personalized plans, and data to enable customized solutions.

    

### [[2110.12006] Uncertainty aware anomaly detection to predict errant beam pulses in the SNS accelerator](http://arxiv.org/abs/2110.12006)


  High-power particle accelerators are complex machines with thousands of
pieces of equipmentthat are frequently running at the cutting edge of
technology. In order to improve the day-to-dayoperations and maximize the
delivery of the science, new analytical techniques are being exploredfor
anomaly detection, classification, and prognostications. As such, we describe
the applicationof an uncertainty aware Machine Learning method, the Siamese
neural network model, to predictupcoming errant beam pulses using the data from
a single monitoring device. By predicting theupcoming failure, we can stop the
accelerator before damage occurs. We describe the acceleratoroperation, related
Machine Learning research, the prediction performance required to abort
beamwhile maintaining operations, the monitoring device and its data, and the
Siamese method andits results. These results show that the researched method
can be applied to improve acceleratoroperations.

    

### [[2110.12007] When to Prune? A Policy towards Early Structural Pruning](http://arxiv.org/abs/2110.12007)


  Pruning enables appealing reductions in network memory footprint and time
complexity. Conventional post-training pruning techniques lean towards
efficient inference while overlooking the heavy computation for training.
Recent exploration of pre-training pruning at initialization hints on training
cost reduction via pruning, but suffers noticeable performance degradation. We
attempt to combine the benefits of both directions and propose a policy that
prunes as early as possible during training without hurting performance.
Instead of pruning at initialization, our method exploits initial dense
training for few epochs to quickly guide the architecture, while constantly
evaluating dominant sub-networks via neuron importance ranking. This unveils
dominant sub-networks whose structures turn stable, allowing conventional
pruning to be pushed earlier into the training. To do this early, we further
introduce an Early Pruning Indicator (EPI) that relies on sub-network
architectural similarity and quickly triggers pruning when the sub-network's
architecture stabilizes. Through extensive experiments on ImageNet, we show
that EPI empowers a quick tracking of early training epochs suitable for
pruning, offering same efficacy as an otherwise ``oracle'' grid-search that
scans through epochs and requires orders of magnitude more compute. Our method
yields $1.4\%$ top-1 accuracy boost over state-of-the-art pruning counterparts,
cuts down training cost on GPU by $2.4\times$, hence offers a new
efficiency-accuracy boundary for network pruning during training.

    

### [[2110.12012] RDD-Eclat: Approaches to Parallelize Eclat Algorithm on Spark RDD Framework (Extended Version)](http://arxiv.org/abs/2110.12012)


  Frequent itemset mining (FIM) is a highly computational and data intensive
algorithm. Therefore, parallel and distributed FIM algorithms have been
designed to process large volume of data in a reduced time. Recently, a number
of FIM algorithms have been designed on Hadoop MapReduce, a distributed big
data processing framework. But, due to heavy disk I/O, MapReduce is found to be
inefficient for the highly iterative FIM algorithms. Therefore, Spark, a more
efficient distributed data processing framework, has been developed with
in-memory computation and resilient distributed dataset (RDD) features to
support the iterative algorithms. On this framework, Apriori and FP-Growth
based FIM algorithms have been designed on the Spark RDD framework, but
Eclat-based algorithm has not been explored yet. In this paper, RDD-Eclat, a
parallel Eclat algorithm on the Spark RDD framework is proposed with its five
variants. The proposed algorithms are evaluated on the various benchmark
datasets, and the experimental results show that RDD-Eclat outperforms the
Spark-based Apriori by many times. Also, the experimental results show the
scalability of the proposed algorithms on increasing the number of cores and
size of the dataset.

    

### [[2110.12020] Fairness Degrading Adversarial Attacks Against Clustering Algorithms](http://arxiv.org/abs/2110.12020)


  Clustering algorithms are ubiquitous in modern data science pipelines, and
are utilized in numerous fields ranging from biology to facility location. Due
to their widespread use, especially in societal resource allocation problems,
recent research has aimed at making clustering algorithms fair, with great
success. Furthermore, it has also been shown that clustering algorithms, much
like other machine learning algorithms, are susceptible to adversarial attacks
where a malicious entity seeks to subvert the performance of the learning
algorithm. However, despite these known vulnerabilities, there has been no
research undertaken that investigates fairness degrading adversarial attacks
for clustering. We seek to bridge this gap by formulating a generalized attack
optimization problem aimed at worsening the group-level fairness of
centroid-based clustering algorithms. As a first step, we propose a fairness
degrading attack algorithm for k-median clustering that operates under a
whitebox threat model -- where the clustering algorithm, fairness notion, and
the input dataset are known to the adversary. We provide empirical results as
well as theoretical analysis for our simple attack algorithm, and find that the
addition of the generated adversarial samples can lead to significantly lower
fairness values. In this manner, we aim to motivate fairness degrading
adversarial attacks as a direction for future research in fair clustering.

    

### [[2110.12024] A Prototype-Oriented Framework for Unsupervised Domain Adaptation](http://arxiv.org/abs/2110.12024)


  Existing methods for unsupervised domain adaptation often rely on minimizing
some statistical distance between the source and target samples in the latent
space. To avoid the sampling variability, class imbalance, and data-privacy
concerns that often plague these methods, we instead provide a memory and
computation-efficient probabilistic framework to extract class prototypes and
align the target features with them. We demonstrate the general applicability
of our method on a wide range of scenarios, including single-source,
multi-source, class-imbalance, and source-private domain adaptation. Requiring
no additional model parameters and having a moderate increase in computation
over the source model alone, the proposed method achieves competitive
performance with state-of-the-art methods.

    

### [[2110.12033] A Simple Baseline for Low-Budget Active Learning](http://arxiv.org/abs/2110.12033)


  Active learning focuses on choosing a subset of unlabeled data to be labeled.
However, most such methods assume that a large subset of the data can be
annotated. We are interested in low-budget active learning where only a small
subset (e.g., 0.2% of ImageNet) can be annotated. Instead of proposing a new
query strategy to iteratively sample batches of unlabeled data given an initial
pool, we learn rich features by an off-the-shelf self-supervised learning
method only once and then study the effectiveness of different sampling
strategies given a low budget on a variety of datasets as well as ImageNet
dataset. We show that although the state-of-the-art active learning methods
work well given a large budget of data labeling, a simple k-means clustering
algorithm can outperform them on low budgets. We believe this method can be
used as a simple baseline for low-budget active learning on image
classification. Code is available at:
this https URL


### [[2110.12035] Distance-wise Prototypical Graph Neural Network in Node Imbalance Classification](http://arxiv.org/abs/2110.12035)


  Recent years have witnessed the significant success of applying graph neural
networks (GNNs) in learning effective node representations for classification.
However, current GNNs are mostly built under the balanced data-splitting, which
is inconsistent with many real-world networks where the number of training
nodes can be extremely imbalanced among the classes. Thus, directly utilizing
current GNNs on imbalanced data would generate coarse representations of nodes
in minority classes and ultimately compromise the classification performance.
This therefore portends the importance of developing effective GNNs for
handling imbalanced graph data. In this work, we propose a novel Distance-wise
Prototypical Graph Neural Network (DPGNN), which proposes a class
prototype-driven training to balance the training loss between majority and
minority classes and then leverages distance metric learning to differentiate
the contributions of different dimensions of representations and fully encode
the relative position of each node to each class prototype. Moreover, we design
a new imbalanced label propagation mechanism to derive extra supervision from
unlabeled nodes and employ self-supervised learning to smooth representations
of adjacent nodes while separating inter-class prototypes. Comprehensive node
classification experiments and parameter analysis on multiple networks are
conducted and the proposed DPGNN almost always significantly outperforms all
other baselines, which demonstrates its effectiveness in imbalanced node
classification. The implementation of DPGNN is available at
\url{this https URL}.

    

### [[2110.12036] Recursive Causal Structure Learning in the Presence of Latent Variables and Selection Bias](http://arxiv.org/abs/2110.12036)


  We consider the problem of learning the causal MAG of a system from
observational data in the presence of latent variables and selection bias.
Constraint-based methods are one of the main approaches for solving this
problem, but the existing methods are either computationally impractical when
dealing with large graphs or lacking completeness guarantees. We propose a
novel computationally efficient recursive constraint-based method that is sound
and complete. The key idea of our approach is that at each iteration a specific
type of variable is identified and removed. This allows us to learn the
structure efficiently and recursively, as this technique reduces both the
number of required conditional independence (CI) tests and the size of the
conditioning sets. The former substantially reduces the computational
complexity, while the latter results in more reliable CI tests. We provide an
upper bound on the number of required CI tests in the worst case. To the best
of our knowledge, this is the tightest bound in the literature. We further
provide a lower bound on the number of CI tests required by any
constraint-based method. The upper bound of our proposed approach and the lower
bound at most differ by a factor equal to the number of variables in the worst
case. We provide experimental results to compare the proposed approach with the
state of the art on both synthetic and real-world structures.

    

### [[2110.12046] Uncertainty Quantification For Low-Rank Matrix Completion With Heterogeneous and Sub-Exponential Noise](http://arxiv.org/abs/2110.12046)


  The problem of low-rank matrix completion with heterogeneous and
sub-exponential (as opposed to homogeneous and Gaussian) noise is particularly
relevant to a number of applications in modern commerce. Examples include panel
sales data and data collected from web-commerce systems such as recommendation
engines. An important unresolved question for this problem is characterizing
the distribution of estimated matrix entries under common low-rank estimators.
Such a characterization is essential to any application that requires
quantification of uncertainty in these estimates and has heretofore only been
available under the assumption of homogenous Gaussian noise. Here we
characterize the distribution of estimated matrix entries when the observation
noise is heterogeneous sub-exponential and provide, as an application, explicit
formulas for this distribution when observed entries are Poisson or Binary
distributed.

    

### [[2110.12052] On the Tractability of Neural Causal Inference](http://arxiv.org/abs/2110.12052)


  Roth (1996) proved that any form of marginal inference with probabilistic
graphical models (e.g. Bayesian Networks) will at least be NP-hard. Introduced
and extensively investigated in the past decade, the neural probabilistic
circuits known as sum-product network (SPN) offers linear time complexity. On
another note, research around neural causal models (NCM) recently gained
traction, demanding a tighter integration of causality for machine learning. To
this end, we present a theoretical investigation of if, when, how and under
what cost tractability occurs for different NCM. We prove that SPN-based causal
inference is generally tractable, opposed to standard MLP-based NCM. We further
introduce a new tractable NCM-class that is efficient in inference and fully
expressive in terms of Pearl's Causal Hierarchy. Our comparative empirical
illustration on simulations and standard benchmarks validates our theoretical
proofs.

    

### [[2110.12059] Two-Timescale End-to-End Learning for Channel Acquisition and Hybrid Precoding](http://arxiv.org/abs/2110.12059)


  In this paper, we propose an end-to-end deep learning-based joint transceiver
design algorithm for millimeter wave (mmWave) massive multiple-input
multiple-output (MIMO) systems, which consists of deep neural network
(DNN)-aided pilot training, channel feedback, and hybrid analog-digital (HAD)
precoding. Specifically, we develop a DNN architecture that maps the received
pilots into feedback bits at the receiver, and then further maps the feedback
bits into the hybrid precoder at the transmitter. To reduce the signaling
overhead and channel state information (CSI) mismatch caused by the
transmission delay, a two-timescale DNN composed of a long-term DNN and a
short-term DNN is developed. The analog precoders are designed by the long-term
DNN based on the CSI statistics and updated once in a frame consisting of a
number of time slots. In contrast, the digital precoders are optimized by the
short-term DNN at each time slot based on the estimated low-dimensional
equivalent CSI matrices. A two-timescale training method is also developed for
the proposed DNN with a binary layer. We then analyze the generalization
ability and signaling overhead for the proposed DNN based algorithm. Simulation
results show that our proposed technique significantly outperforms conventional
schemes in terms of bit-error rate performance with reduced signaling overhead
and shorter pilot sequences.

    

### [[2110.12062] DeepAg: Deep Learning Approach for Measuring the Effects of Outlier Events on Agricultural Production and Policy](http://arxiv.org/abs/2110.12062)


  Quantitative metrics that measure the global economy's equilibrium have
strong and interdependent relationships with the agricultural supply chain and
international trade flows. Sudden shocks in these processes caused by outlier
events such as trade wars, pandemics, or weather can have complex effects on
the global economy. In this paper, we propose a novel framework, namely:
DeepAg, that employs econometrics and measures the effects of outlier events
detection using Deep Learning (DL) to determine relationships between
commonplace financial indices (such as the DowJones), and the production values
of agricultural commodities (such as Cheese and Milk). We employed a DL
technique called Long Short-Term Memory (LSTM) networks successfully to predict
commodity production with high accuracy and also present five popular models
(regression and boosting) as baselines to measure the effects of outlier
events. The results indicate that DeepAg with outliers' considerations (using
Isolation Forests) outperforms baseline models, as well as the same model
without outliers detection. Outlier events make a considerable impact when
predicting commodity production with respect to financial indices. Moreover, we
present the implications of DeepAg on public policy, provide insights for
policymakers and farmers, and for operational decisions in the agricultural
ecosystem. Data are collected, models developed, and the results are recorded
and presented.

    

### [[2110.12064] Causal Effect Identification with Context-specific Independence Relations of Control Variables](http://arxiv.org/abs/2110.12064)


  We study the problem of causal effect identification from observational
distribution given the causal graph and some context-specific independence
(CSI) relations. It was recently shown that this problem is NP-hard, and while
a sound algorithm to learn the causal effects is proposed in Tikka et al.
(2019), no complete algorithm for the task exists. In this work, we propose a
sound and complete algorithm for the setting when the CSI relations are limited
to observed nodes with no parents in the causal graph. One limitation of the
state of the art in terms of its applicability is that the CSI relations among
all variables, even unobserved ones, must be given (as opposed to learned).
Instead, We introduce a set of graphical constraints under which the CSI
relations can be learned from mere observational distribution. This expands the
set of identifiable causal effects beyond the state of the art.

    

### [[2110.12065] Multiplication-Avoiding Variant of Power Iteration with Applications](http://arxiv.org/abs/2110.12065)


  Power iteration is a fundamental algorithm in data analysis. It extracts the
eigenvector corresponding to the largest eigenvalue of a given matrix.
Applications include ranking algorithms, recommendation systems, principal
component analysis (PCA), among many others. In this paper, We introduce
multiplication-avoiding power iteration (MAPI), which replaces the standard
$\ell_2$-inner products that appear at the regular power iteration (RPI) with
multiplication-free vector products which are Mercer-type kernel operations
related with the $\ell_1$ norm. Precisely, for an $n\times n$ matrix, MAPI
requires $n$ multiplications, while RPI needs $n^2$ multiplications per
iteration. Therefore, MAPI provides a significant reduction of the number of
multiplication operations, which are known to be costly in terms of energy
consumption. We provide applications of MAPI to PCA-based image reconstruction
as well as to graph-based ranking algorithms. When compared to RPI, MAPI not
only typically converges much faster, but also provides superior performance.

    

### [[2110.12066] The Causal Loss: Driving Correlation to Imply Causation](http://arxiv.org/abs/2110.12066)


  Most algorithms in classical and contemporary machine learning focus on
correlation-based dependence between features to drive performance. Although
success has been observed in many relevant problems, these algorithms fail when
the underlying causality is inconsistent with the assumed relations. We propose
a novel model-agnostic loss function called Causal Loss that improves the
interventional quality of the prediction using an intervened neural-causal
regularizer. In support of our theoretical results, our experimental
illustration shows how causal loss bestows a non-causal associative model (like
a standard neural net or decision tree) with interventional capabilities.

    

### [[2110.12067] Gaussian Graphical Model Selection for Huge Data via Minipatch Learning](http://arxiv.org/abs/2110.12067)


  Gaussian graphical models are essential unsupervised learning techniques to
estimate conditional dependence relationships between sets of nodes. While
graphical model selection is a well-studied problem with many popular
techniques, there are typically three key practical challenges: i) many
existing methods become computationally intractable in huge-data settings with
tens of thousands of nodes; ii) the need for separate data-driven tuning
hyperparameter selection procedures considerably adds to the computational
burden; iii) the statistical accuracy of selected edges often deteriorates as
the dimension and/or the complexity of the underlying graph structures
increase. We tackle these problems by proposing the Minipatch Graph (MPGraph)
estimator. Our approach builds upon insights from the latent variable graphical
model problem and utilizes ensembles of thresholded graph estimators fit to
tiny, random subsets of both the observations and the nodes, termed
minipatches. As estimates are fit on small problems, our approach is
computationally fast with integrated stability-based hyperparameter tuning.
Additionally, we prove that under certain conditions our MPGraph algorithm
achieves finite-sample graph selection consistency. We compare our approach to
state-of-the-art computational approaches to Gaussian graphical model selection
including the BigQUIC algorithm, and empirically demonstrate that our approach
is not only more accurate but also extensively faster for huge graph selection
problems.

    

### [[2110.12072] How and When Adversarial Robustness Transfers in Knowledge Distillation?](http://arxiv.org/abs/2110.12072)


  Knowledge distillation (KD) has been widely used in teacher-student training,
with applications to model compression in resource-constrained deep learning.
Current works mainly focus on preserving the accuracy of the teacher model.
However, other important model properties, such as adversarial robustness, can
be lost during distillation. This paper studies how and when the adversarial
robustness can be transferred from a teacher model to a student model in KD. We
show that standard KD training fails to preserve adversarial robustness, and we
propose KD with input gradient alignment (KDIGA) for remedy. Under certain
assumptions, we prove that the student model using our proposed KDIGA can
achieve at least the same certified robustness as the teacher model. Our
experiments of KD contain a diverse set of teacher and student models with
varying network architectures and sizes evaluated on ImageNet and CIFAR-10
datasets, including residual neural networks (ResNets) and vision transformers
(ViTs). Our comprehensive analysis shows several novel insights that (1) With
KDIGA, students can preserve or even exceed the adversarial robustness of the
teacher model, even when their models have fundamentally different
architectures; (2) KDIGA enables robustness to transfer to pre-trained
students, such as KD from an adversarially trained ResNet to a pre-trained ViT,
without loss of clean accuracy; and (3) Our derived local linearity bounds for
characterizing adversarial robustness in KD are consistent with the empirical
results.

    

### [[2110.12076] Applications of Generative Adversarial Networks in Anomaly Detection: A Systematic Literature Review](http://arxiv.org/abs/2110.12076)


  Anomaly detection has become an indispensable tool for modern society,
applied in a wide range of applications, from detecting fraudulent transactions
to malignant brain tumours. Over time, many anomaly detection techniques have
been introduced. However, in general, they all suffer from the same problem: a
lack of data that represents anomalous behaviour. As anomalous behaviour is
usually costly (or dangerous) for a system, it is difficult to gather enough
data that represents such behaviour. This, in turn, makes it difficult to
develop and evaluate anomaly detection techniques. Recently, generative
adversarial networks (GANs) have attracted a great deal of attention in anomaly
detection research, due to their unique ability to generate new data. In this
paper, we present a systematic literature review of the applications of GANs in
anomaly detection, covering 128 papers on the subject. The goal of this review
paper is to analyze and summarize: (1) which anomaly detection techniques can
benefit from certain types of GANs, and how, (2) in which application domains
GAN-assisted anomaly detection techniques have been applied, and (3) which
datasets and performance metrics have been used to evaluate these techniques.
Our study helps researchers and practitioners to find the most suitable
GAN-assisted anomaly detection technique for their application. In addition, we
present a research roadmap for future studies in this area.

    

### [[2110.12080] C-Planning: An Automatic Curriculum for Learning Goal-Reaching Tasks](http://arxiv.org/abs/2110.12080)


  Goal-conditioned reinforcement learning (RL) can solve tasks in a wide range
of domains, including navigation and manipulation, but learning to reach
distant goals remains a central challenge to the field. Learning to reach such
goals is particularly hard without any offline data, expert demonstrations, and
reward shaping. In this paper, we propose an algorithm to solve the distant
goal-reaching task by using search at training time to automatically generate a
curriculum of intermediate states. Our algorithm, Classifier-Planning
(C-Planning), frames the learning of the goal-conditioned policies as
expectation maximization: the E-step corresponds to planning an optimal
sequence of waypoints using graph search, while the M-step aims to learn a
goal-conditioned policy to reach those waypoints. Unlike prior methods that
combine goal-conditioned RL with graph search, ours performs search only during
training and not testing, significantly decreasing the compute costs of
deploying the learned policy. Empirically, we demonstrate that our method is
more sample efficient than prior methods. Moreover, it is able to solve very
long horizons manipulation and navigation tasks, tasks that prior
goal-conditioned methods and methods based on graph search fail to solve.

    

### [[2110.12081] Off-policy Reinforcement Learning with Optimistic Exploration and Distribution Correction](http://arxiv.org/abs/2110.12081)


  Improving sample efficiency of reinforcement learning algorithms requires
effective exploration. Following the principle of $\textit{optimism in the face
of uncertainty}$, we train a separate exploration policy to maximize an
approximate upper confidence bound of the critics in an off-policy actor-critic
framework. However, this introduces extra differences between the replay buffer
and the target policy in terms of their stationary state-action distributions.
To mitigate the off-policy-ness, we adapt the recently introduced DICE
framework to learn a distribution correction ratio for off-policy actor-critic
training. In particular, we correct the training distribution for both policies
and critics. Empirically, we evaluate our proposed method in several
challenging continuous control tasks and show superior performance compared to
state-of-the-art methods. We also conduct extensive ablation studies to
demonstrate the effectiveness and the rationality of the proposed method.

    

### [[2110.12087] Gaussian Process Sampling and Optimization with Approximate Upper and Lower Bounds](http://arxiv.org/abs/2110.12087)


  Many functions have approximately-known upper and/or lower bounds,
potentially aiding the modeling of such functions. In this paper, we introduce
Gaussian process models for functions where such bounds are (approximately)
known. More specifically, we propose the first use of such bounds to improve
Gaussian process (GP) posterior sampling and Bayesian optimization (BO). That
is, we transform a GP model satisfying the given bounds, and then sample and
weight functions from its posterior. To further exploit these bounds in BO
settings, we present bounded entropy search (BES) to select the point gaining
the most information about the underlying function, estimated by the GP
samples, while satisfying the output constraints. We characterize the sample
variance bounds and show that the decision made by BES is explainable. Our
proposed approach is conceptually straightforward and can be used as a plug in
extension to existing methods for GP posterior sampling and Bayesian
optimization.

    

### [[2110.12088] Learning with Noisy Labels Revisited: A Study Using Real-World Human Annotations](http://arxiv.org/abs/2110.12088)


  Existing research on learning with noisy labels mainly focuses on synthetic
label noise. Synthetic label noise, though has clean structures which greatly
enable statistical analyses, often fails to model the real-world noise
patterns. The recent literature has observed several efforts to offer
real-world noisy datasets, yet the existing efforts suffer from two caveats:
firstly, the lack of ground-truth verification makes it hard to theoretically
study the property and treatment of real-world label noise. Secondly, these
efforts are often of large scales, which may lead to unfair comparisons of
robust methods within reasonable and accessible computation power. To better
understand real-world label noise, it is important to establish controllable
and moderate-sized real-world noisy datasets with both ground-truth and noisy
labels. This work presents two new benchmark datasets (CIFAR-10N, CIFAR-100N),
equipping the train dataset of CIFAR-10 and CIFAR-100 with human-annotated
real-world noisy labels that we collect from Amazon Mechanical Turk. We
quantitatively and qualitatively show that real-world noisy labels follow an
instance-dependent pattern rather than the classically adopted class-dependent
ones. We then initiate an effort to benchmark a subset of existing solutions
using CIFAR-10N, CIFAR-100N. We next proceed to study the memorization of model
predictions, which further illustrates the difference between human noise and
class-dependent synthetic noise. We show indeed the real-world noise patterns
impose new and outstanding challenges as compared to synthetic ones. These
observations require us to rethink the treatment of noisy labels, and we hope
the availability of these two datasets would facilitate the development and
evaluation of future learning with noisy label solutions. The corresponding
datasets and the leaderboard are publicly available at
\url{this http URL}.

    

### [[2110.12091] Contrastively Disentangled Sequential Variational Autoencoder](http://arxiv.org/abs/2110.12091)


  Self-supervised disentangled representation learning is a critical task in
sequence modeling. The learnt representations contribute to better model
interpretability as well as the data generation, and improve the sample
efficiency for downstream tasks. We propose a novel sequence representation
learning method, named Contrastively Disentangled Sequential Variational
Autoencoder (C-DSVAE), to extract and separate the static (time-invariant) and
dynamic (time-variant) factors in the latent space. Different from previous
sequential variational autoencoder methods, we use a novel evidence lower bound
which maximizes the mutual information between the input and the latent
factors, while penalizes the mutual information between the static and dynamic
factors. We leverage contrastive estimations of the mutual information terms in
training, together with simple yet effective augmentation techniques, to
introduce additional inductive biases. Our experiments show that C-DSVAE
significantly outperforms the previous state-of-the-art methods on multiple
metrics.

    

### [[2110.12093] Circle Representation for Medical Object Detection](http://arxiv.org/abs/2110.12093)


  Box representation has been extensively used for object detection in computer
vision. Such representation is efficacious but not necessarily optimized for
biomedical objects (e.g., glomeruli), which play an essential role in renal
pathology. In this paper, we propose a simple circle representation for medical
object detection and introduce CircleNet, an anchor-free detection framework.
Compared with the conventional bounding box representation, the proposed
bounding circle representation innovates in three-fold: (1) it is optimized for
ball-shaped biomedical objects; (2) The circle representation reduced the
degree of freedom compared with box representation; (3) It is naturally more
rotation invariant. When detecting glomeruli and nuclei on pathological images,
the proposed circle representation achieved superior detection performance and
be more rotation-invariant, compared with the bounding box. The code has been
made publicly available: this https URL


### [[2110.12108] ConformalLayers: A non-linear sequential neural network with associative layers](http://arxiv.org/abs/2110.12108)


  Convolutional Neural Networks (CNNs) have been widely applied. But as the
CNNs grow, the number of arithmetic operations and memory footprint also
increase. Furthermore, typical non-linear activation functions do not allow
associativity of the operations encoded by consecutive layers, preventing the
simplification of intermediate steps by combining them. We present a new
activation function that allows associativity between sequential layers of
CNNs. Even though our activation function is non-linear, it can be represented
by a sequence of linear operations in the conformal model for Euclidean
geometry. In this domain, operations like, but not limited to, convolution,
average pooling, and dropout remain linear. We take advantage of associativity
to combine all the "conformal layers" and make the cost of inference constant
regardless of the depth of the network.

    

### [[2110.12111] Improve High Level Classification with a More Sensitive metric and Optimization approach for Complex Network Building](http://arxiv.org/abs/2110.12111)


  Complex Networks are a good approach to find internal relationships and
represent the structure of classes in a dataset then they are used for High
Level Classification. Previous works use K-Nearest Neighbors to build each
Complex Network considering all the available samples. This paper introduces a
different creation of Complex Networks, considering only sample which belongs
to each class. And metric is used to analyze the structure of Complex Networks,
besides an optimization approach to improve the performance is presented.
Experiments are executed considering a cross validation process, the
optimization approach is performed using grid search and Genetic Algorithm,
this process can improve the results up to 10%.

    

### [[2110.12112] Why Machine Learning Cannot Ignore Maximum Likelihood Estimation](http://arxiv.org/abs/2110.12112)


  The growth of machine learning as a field has been accelerating with
increasing interest and publications across fields, including statistics, but
predominantly in computer science. How can we parse this vast literature for
developments that exemplify the necessary rigor? How many of these manuscripts
incorporate foundational theory to allow for statistical inference? Which
advances have the greatest potential for impact in practice? One could posit
many answers to these queries. Here, we assert that one essential idea is for
machine learning to integrate maximum likelihood for estimation of functional
parameters, such as prediction functions and conditional densities.

    

### [[2110.12113] Multi-task Recurrent Neural Networks to Simultaneously Infer Mode and Purpose in GPS Trajectories](http://arxiv.org/abs/2110.12113)


  Multi-task learning is assumed as a powerful inference method, specifically,
where there is a considerable correlation between multiple tasks, predicting
them in an unique framework may enhance prediction results. This research
challenges this assumption by developing several single-task models to compare
their results against multi-task learners to infer mode and purpose of trip
from smartphone travel survey data collected as part of a smartphone-based
travel survey. GPS trajectory data along with socio-demographics and
destination-related characteristics are fed into a multi-input neural network
framework to predict two outputs; mode and purpose. We deployed Recurrent
Neural Networks (RNN) that are fed by sequential GPS trajectories. To process
the socio-demographics and destination-related characteristics, another neural
network, with different embedding and dense layers is used in parallel with RNN
layers in a multi-input multi-output framework. The results are compared
against the single-task learners that classify mode and purpose independently.
We also investigate different RNN approaches such as Long-Short Term Memory
(LSTM), Gated Recurrent Units (GRU) and Bi-directional Gated Recurrent Units
(Bi-GRU). The best multi-task learner was a Bi-GRU model able to classify mode
and purpose with an F1-measures of 84.33% and 78.28%, while the best
single-task learner to infer mode of transport was a GRU model that achieved an
F1-measure of 86.50%, and the best single-task Bi-GRU purpose detection model
that reached an F1-measure of 77.38%. While there's an assumption of higher
performance of multi-task over sing-task learners, the results of this study
does not hold such an assumption and shows, in the context of mode and trip
purpose inference from GPS trajectory data, a multi-task learning approach does
not bring any considerable advantage over single-task learners.

    

### [[2110.12118] The Countable-armed Bandit with Vanishing Arms](http://arxiv.org/abs/2110.12118)


  We consider a bandit problem with countably many arms, partitioned into
finitely many "types," each characterized by a unique mean reward. A
"non-stationary" distribution governs the relative abundance of each arm-type
in the population of arms, aka the "arm-reservoir." This non-stationarity is
attributable to a probabilistic leakage of "optimal" arms from the reservoir
over time, which we refer to as the "vanishing arms" phenomenon; this induces a
time-varying (potentially "endogenous," policy-dependent) distribution over the
reservoir. The objective is minimization of the expected cumulative regret. We
characterize necessary and sufficient conditions for achievability of
sub-linear regret in terms of a critical vanishing rate of optimal arms. We
also discuss two reservoir distribution-oblivious algorithms that are
long-run-average optimal whenever sub-linear regret is statistically
achievable. Numerical experiments highlight a distinctive characteristic of
this problem related to ex ante knowledge of the "gap" parameter (the
difference between the top two mean rewards): in contrast to the stationary
bandit formulation, regret in our setting may suffer substantial inflation
under adaptive exploration-based (gap-oblivious) algorithms such as UCB
vis-`a-vis their non-adaptive forced exploration-based (gap-aware) counterparts
like ETC.

    

### [[2110.12121] On Geometric Connections of Embedded and Quotient Geometries in Riemannian Fixed-rank Matrix Optimization](http://arxiv.org/abs/2110.12121)


  In this paper, we propose a general procedure for establishing the landscape
connections of a Riemannian optimization problem under the embedded and
quotient geometries. By applying the general procedure to the fixed-rank
positive semidefinite (PSD) and general matrix optimization, we establish an
exact Riemannian gradient connection under two geometries at every point on the
manifold and sandwich inequalities between the spectra of Riemannian Hessians
at Riemannian first-order stationary points (FOSPs). These results immediately
imply an equivalence on the sets of Riemannian FOSPs, Riemannian second-order
stationary points (SOSPs) and strict saddles of fixed-rank matrix optimization
under the embedded and the quotient geometries. To the best of our knowledge,
this is the first geometric landscape connection between the embedded and the
quotient geometries for fixed-rank matrix optimization and it provides a
concrete example on how these two geometries are connected in Riemannian
optimization. In addition, the effects of the Riemannian metric and quotient
structure on the landscape connection are discussed. We also observe an
algorithmic connection for fixed-rank matrix optimization under two geometries
with some specific Riemannian metrics. A number of novel ideas and technical
ingredients including a unified treatment for different Riemannian metrics and
new horizontal space representations under quotient geometries are developed to
obtain our results. The results in this paper deepen our understanding of
geometric connections of Riemannian optimization under different Riemannian
geometries and provide a few new theoretical insights to unanswered questions
in the literature.

    

### [[2110.12122] Quantifying Epistemic Uncertainty in Deep Learning](http://arxiv.org/abs/2110.12122)


  Uncertainty quantification is at the core of the reliability and robustness
of machine learning. It is well-known that uncertainty consists of two
different types, often referred to as aleatoric and epistemic uncertainties. In
this paper, we provide a systematic study on the epistemic uncertainty in deep
supervised learning. We rigorously distinguish different sources of epistemic
uncertainty, including in particular procedural variability (from the training
procedure) and data variability (from the training data). We use our framework
to explain how deep ensemble enhances prediction by reducing procedural
variability. We also propose two approaches to estimate epistemic uncertainty
for a well-trained neural network in practice. One uses influence function
derived from the theory of neural tangent kernel that bypasses the convexity
assumption violated by modern neural networks. Another uses batching that
bypasses the time-consuming Gram matrix inversion in the influence function
calculation, while expending minimal re-training effort. We discuss how both
approaches overcome some difficulties in applying classical statistical methods
to the inference on deep learning.

    

### [[2110.12132] Towards the D-Optimal Online Experiment Design for Recommender Selection](http://arxiv.org/abs/2110.12132)


  Selecting the optimal recommender via online exploration-exploitation is
catching increasing attention where the traditional A/B testing can be slow and
costly, and offline evaluations are prone to the bias of history data. Finding
the optimal online experiment is nontrivial since both the users and displayed
recommendations carry contextual features that are informative to the reward.
While the problem can be formalized via the lens of multi-armed bandits, the
existing solutions are found less satisfactorily because the general
methodologies do not account for the case-specific structures, particularly for
the e-commerce recommendation we study. To fill in the gap, we leverage the
\emph{D-optimal design} from the classical statistics literature to achieve the
maximum information gain during exploration, and reveal how it fits seamlessly
with the modern infrastructure of online inference. To demonstrate the
effectiveness of the optimal designs, we provide semi-synthetic simulation
studies with published code and data for reproducibility purposes. We then use
our deployment example on this http URL to fully illustrate the practical
insights and effectiveness of the proposed methods.

    

### [[2110.12141] Rethinking Neural vs. Matrix-Factorization Collaborative Filtering: the Theoretical Perspectives](http://arxiv.org/abs/2110.12141)


  The recent work by Rendle et al. (2020), based on empirical observations,
argues that matrix-factorization collaborative filtering (MCF) compares
favorably to neural collaborative filtering (NCF), and conjectures the dot
product's superiority over the feed-forward neural network as similarity
function. In this paper, we address the comparison rigorously by answering the
following questions: 1. what is the limiting expressivity of each model; 2.
under the practical gradient descent, to which solution does each optimization
path converge; 3. how would the models generalize under the inductive and
transductive learning setting. Our results highlight the similar expressivity
for the overparameterized NCF and MCF as kernelized predictors, and reveal the
relation between their optimization paths. We further show their different
generalization behaviors, where MCF and NCF experience specific tradeoff and
comparison in the transductive and inductive collaborative filtering setting.
Lastly, by showing a novel generalization result, we reveal the critical role
of correcting exposure bias for model evaluation in the inductive setting. Our
results explain some of the previously observed conflicts, and we provide
synthetic and real-data experiments to shed further insights to this topic.

    

### [[2110.12144] Foresight of Graph Reinforcement Learning Latent Permutations Learnt by Gumbel Sinkhorn Network](http://arxiv.org/abs/2110.12144)


  Vital importance has necessity to be attached to cooperation in multi-agent
environments, as a result of which some reinforcement learning algorithms
combined with graph neural networks have been proposed to understand the mutual
interplay between agents. However, highly complicated and dynamic multi-agent
environments require more ingenious graph neural networks, which can
comprehensively represent not only the graph topology structure but also
evolution process of the structure due to agents emerging, disappearing and
moving. To tackle these difficulties, we propose Gumbel Sinkhorn graph
attention reinforcement learning, where a graph attention network highly
represents the underlying graph topology structure of the multi-agent
environment, and can adapt to the dynamic topology structure of graph better
with the help of Gumbel Sinkhorn network by learning latent permutations.
Empirically, simulation results show how our proposed graph reinforcement
learning methodology outperforms existing methods in the PettingZoo multi-agent
environment by learning latent permutations.

    

### [[2110.12148] Event Detection on Dynamic Graphs](http://arxiv.org/abs/2110.12148)


  Event detection is a critical task for timely decision-making in graph
analytics applications. Despite the recent progress towards deep learning on
graphs, event detection on dynamic graphs presents particular challenges to
existing architectures. Real-life events are often associated with sudden
deviations of the normal behavior of the graph. However, existing approaches
for dynamic node embedding are unable to capture the graph-level dynamics
related to events.
In this paper, we propose DyGED, a simple yet novel deep learning model for
event detection on dynamic graphs. DyGED learns correlations between the graph
macro dynamics -- i.e. a sequence of graph-level representations -- and labeled
events. Moreover, our approach combines structural and temporal self-attention
mechanisms to account for application-specific node and time importances
effectively. Our experimental evaluation, using a representative set of
datasets, demonstrates that DyGED outperforms competing solutions in terms of
event detection accuracy by up to 8.5% while being more scalable than the top
alternatives. We also present case studies illustrating key features of our
model.

    

### [[2110.12149] On Parameter Estimation in Unobserved Components Models subject to Linear Inequality Constraints](http://arxiv.org/abs/2110.12149)


  We propose a new quadratic-programming-based method of approximating a
nonstandard density using a multivariate Gaussian density. Such nonstandard
densities usually arise while developing posterior samplers for unobserved
components models involving inequality constraints on the parameters. For
instance, Chat et al. (2016) propose a new model of trend inflation with linear
inequality constraints on the stochastic trend. We implement the proposed new
method for this model and compare it to the existing approximation. We observe
that the proposed new method works as good as the existing approximation in
terms of the final trend estimates while achieving greater gains in terms of
sample efficiency.

    

### [[2110.12160] Multi-armed Bandit Algorithm against Strategic Replication](http://arxiv.org/abs/2110.12160)


  We consider a multi-armed bandit problem in which a set of arms is registered
by each agent, and the agent receives reward when its arm is selected. An agent
might strategically submit more arms with replications, which can bring more
reward by abusing the bandit algorithm's exploration-exploitation balance. Our
analysis reveals that a standard algorithm indeed fails at preventing
replication and suffers from linear regret in time $T$. We aim to design a
bandit algorithm which demotivates replications and also achieves a small
cumulative regret. We devise Hierarchical UCB (H-UCB) of replication-proof,
which has $O(\ln T)$-regret under any equilibrium. We further propose Robust
Hierarchical UCB (RH-UCB) which has a sublinear regret even in a realistic
scenario with irrational agents replicating careless. We verify our theoretical
findings through numerical experiments.

    

### [[2110.12163] Adversarial Deep Feature Extraction Network for User Independent Human Activity Recognition](http://arxiv.org/abs/2110.12163)


  User dependence remains one of the most difficult general problems in Human
Activity Recognition (HAR), in particular when using wearable sensors. This is
due to the huge variability of the way different people execute even the
simplest actions. In addition, detailed sensor fixtures and placement will be
different for different people or even at different times for the same users.
In theory, the problem can be solved by a large enough data set. However,
recording data sets that capture the entire diversity of complex activity sets
is seldom practicable. Instead, models are needed that focus on features that
are invariant across users. To this end, we present an adversarial
subject-independent feature extraction method with the maximum mean discrepancy
(MMD) regularization for human activity recognition. The proposed model is
capable of learning a subject-independent embedding feature representation from
multiple subjects datasets and generalizing it to unseen target subjects. The
proposed network is based on the adversarial encoder-decoder structure with the
MMD realign the data distribution over multiple subjects. Experimental results
show that the proposed method not only outperforms state-of-the-art methods
over the four real-world datasets but also improves the subject generalization
effectively. We evaluate the method on well-known public data sets showing that
it significantly improves user-independent performance and reduces variance in
results.

    

### [[2110.12172] Scalable Smartphone Cluster for Deep Learning](http://arxiv.org/abs/2110.12172)


  Various deep learning applications on smartphones have been rapidly rising,
but training deep neural networks (DNNs) has too large computational burden to
be executed on a single smartphone. A portable cluster, which connects
smartphones with a wireless network and supports parallel computation using
them, can be a potential approach to resolve the issue. However, by our
findings, the limitations of wireless communication restrict the cluster size
to up to 30 smartphones. Such small-scale clusters have insufficient
computational power to train DNNs from scratch. In this paper, we propose a
scalable smartphone cluster enabling deep learning training by removing the
portability to increase its computational efficiency. The cluster connects 138
Galaxy S10+ devices with a wired network using Ethernet. We implemented
large-batch synchronous training of DNNs based on Caffe, a deep learning
library. The smartphone cluster yielded 90% of the speed of a P100 when
training ResNet-50, and approximately 43x speed-up of a V100 when training
MobileNet-v1.

    

### [[2110.12175] Analysis of Thompson Sampling for Partially Observable Contextual Multi-Armed Bandits](http://arxiv.org/abs/2110.12175)


  Contextual multi-armed bandits are classical models in reinforcement learning
for sequential decision-making associated with individual information. A
widely-used policy for bandits is Thompson Sampling, where samples from a
data-driven probabilistic belief about unknown parameters are used to select
the control actions. For this computationally fast algorithm, performance
analyses are available under full context-observations. However, little is
known for problems that contexts are not fully observed. We propose a Thompson
Sampling algorithm for partially observable contextual multi-armed bandits, and
establish theoretical performance guarantees. Technically, we show that the
regret of the presented policy scales logarithmically with time and the number
of arms, and linearly with the dimension. Further, we establish rates of
learning unknown parameters, and provide illustrative numerical analyses.

    

### [[2110.12178] An attention-driven hierarchical multi-scale representation for visual recognition](http://arxiv.org/abs/2110.12178)


  Convolutional Neural Networks (CNNs) have revolutionized the understanding of
visual content. This is mainly due to their ability to break down an image into
smaller pieces, extract multi-scale localized features and compose them to
construct highly expressive representations for decision making. However, the
convolution operation is unable to capture long-range dependencies such as
arbitrary relations between pixels since it operates on a fixed-size window.
Therefore, it may not be suitable for discriminating subtle changes (e.g.
fine-grained visual recognition). To this end, our proposed method captures the
high-level long-range dependencies by exploring Graph Convolutional Networks
(GCNs), which aggregate information by establishing relationships among
multi-scale hierarchical regions. These regions consist of smaller (closer
look) to larger (far look), and the dependency between regions is modeled by an
innovative attention-driven message propagation, guided by the graph structure
to emphasize the neighborhoods of a given region. Our approach is simple yet
extremely effective in solving both the fine-grained and generic visual
classification problems. It outperforms the state-of-the-arts with a
significant margin on three and is very competitive on other two datasets.

    

### [[2110.12179] MisMatch: Learning to Change Predictive Confidences with Attention for Consistency-Based, Semi-Supervised Medical Image Segmentation](http://arxiv.org/abs/2110.12179)


  The lack of labels is one of the fundamental constraints in deep learning
based methods for image classification and segmentation, especially in
applications such as medical imaging. Semi-supervised learning (SSL) is a
promising method to address the challenge of labels carcity. The
state-of-the-art SSL methods utilise consistency regularisation to learn
unlabelled predictions which are invariant to perturbations on the prediction
confidence. However, such SSL approaches rely on hand-crafted augmentation
techniques which could be sub-optimal. In this paper, we propose MisMatch, a
novel consistency based semi-supervised segmentation method. MisMatch
automatically learns to produce paired predictions with increasedand decreased
confidences. MisMatch consists of an encoder and two decoders. One decoder
learns positive attention for regions of interest (RoI) on unlabelled data
thereby generating higher confidence predictions of RoI. The other decoder
learns negative attention for RoI on the same unlabelled data thereby
generating lower confidence predictions. We then apply a consistency
regularisation between the paired predictions of the decoders. For evaluation,
we first perform extensive cross-validation on a CT-based pulmonary vessel
segmentation task and show that MisMatch statistically outperforms
state-of-the-art semi-supervised methods when only 6.25% of the total labels
are used. Furthermore MisMatch performance using 6.25% ofthe total labels is
comparable to state-of-the-art methodsthat utilise all available labels. In a
second experiment, MisMatch outperforms state-of-the-art methods on an
MRI-based brain tumour segmentation task.

    

### [[2110.12183] Attend and Guide (AG-Net): A Keypoints-driven Attention-based Deep Network for Image Recognition](http://arxiv.org/abs/2110.12183)


  This paper presents a novel keypoints-based attention mechanism for visual
recognition in still images. Deep Convolutional Neural Networks (CNNs) for
recognizing images with distinctive classes have shown great success, but their
performance in discriminating fine-grained changes is not at the same level. We
address this by proposing an end-to-end CNN model, which learns meaningful
features linking fine-grained changes using our novel attention mechanism. It
captures the spatial structures in images by identifying semantic regions (SRs)
and their spatial distributions, and is proved to be the key to modelling
subtle changes in images. We automatically identify these SRs by grouping the
detected keypoints in a given image. The ``usefulness'' of these SRs for image
recognition is measured using our innovative attentional mechanism focusing on
parts of the image that are most relevant to a given task. This framework
applies to traditional and fine-grained image recognition tasks and does not
require manually annotated regions (e.g. bounding-box of body parts, objects,
etc.) for learning and prediction. Moreover, the proposed keypoints-driven
attention mechanism can be easily integrated into the existing CNN models. The
framework is evaluated on six diverse benchmark datasets. The model outperforms
the state-of-the-art approaches by a considerable margin using Distracted
Driver V1 (Acc: 3.39%), Distracted Driver V2 (Acc: 6.58%), Stanford-40 Actions
(mAP: 2.15%), People Playing Musical Instruments (mAP: 16.05%), Food-101 (Acc:
6.30%) and Caltech-256 (Acc: 2.59%) datasets.

    

### [[2110.12184] Domain Adaptation via Maximizing Surrogate Mutual Information](http://arxiv.org/abs/2110.12184)


  Unsupervised domain adaptation (UDA), which is an important topic in transfer
learning, aims to predict unlabeled data from target domain with access to
labeled data from the source domain. In this work, we propose a novel framework
called SIDA (Surrogate Mutual Information Maximization Domain Adaptation) with
strong theoretical guarantees. To be specific, SIDA implements adaptation by
maximizing mutual information (MI) between features. In the framework, a
surrogate joint distribution models the underlying joint distribution of the
unlabeled target domain. Our theoretical analysis validates SIDA by bounding
the expected risk on target domain with MI and surrogate distribution bias.
Experiments show that our approach is comparable with state-of-the-art
unsupervised adaptation methods on standard UDA tasks.

    

### [[2110.12185] Group-disentangled Representation Learning with Weakly-Supervised Regularization](http://arxiv.org/abs/2110.12185)


  Learning interpretable and human-controllable representations that uncover
factors of variation in data remains an ongoing key challenge in representation
learning. We investigate learning group-disentangled representations for groups
of factors with weak supervision. Existing techniques to address this challenge
merely constrain the approximate posterior by averaging over observations of a
shared group. As a result, observations with a common set of variations are
encoded to distinct latent representations, reducing their capacity to
disentangle and generalize to downstream tasks. In contrast to previous works,
we propose GroupVAE, a simple yet effective Kullback-Leibler (KL)
divergence-based regularization across shared latent representations to enforce
consistent and disentangled representations. We conduct a thorough evaluation
and demonstrate that our GroupVAE significantly improves group disentanglement.
Further, we demonstrate that learning group-disentangled representations
improve upon downstream tasks, including fair classification and 3D
shape-related tasks such as reconstruction, classification, and transfer
learning, and is competitive to supervised methods.

    

### [[2110.12187] AFEC: Active Forgetting of Negative Transfer in Continual Learning](http://arxiv.org/abs/2110.12187)


  Continual learning aims to learn a sequence of tasks from dynamic data
distributions. Without accessing to the old training samples, knowledge
transfer from the old tasks to each new task is difficult to determine, which
might be either positive or negative. If the old knowledge interferes with the
learning of a new task, i.e., the forward knowledge transfer is negative, then
precisely remembering the old tasks will further aggravate the interference,
thus decreasing the performance of continual learning. By contrast, biological
neural networks can actively forget the old knowledge that conflicts with the
learning of a new experience, through regulating the learning-triggered
synaptic expansion and synaptic convergence. Inspired by the biological active
forgetting, we propose to actively forget the old knowledge that limits the
learning of new tasks to benefit continual learning. Under the framework of
Bayesian continual learning, we develop a novel approach named Active
Forgetting with synaptic Expansion-Convergence (AFEC). Our method dynamically
expands parameters to learn each new task and then selectively combines them,
which is formally consistent with the underlying mechanism of biological active
forgetting. We extensively evaluate AFEC on a variety of continual learning
benchmarks, including CIFAR-10 regression tasks, visual classification tasks
and Atari reinforcement tasks, where AFEC effectively improves the learning of
new tasks and achieves the state-of-the-art performance in a plug-and-play way.

    

### [[2110.12190] PROMPT: Parallel Iterative Algorithm for $\ell_{p}$ norm linear regression via Majorization Minimization with an application to semi-supervised graph learning](http://arxiv.org/abs/2110.12190)


  In this paper, we consider the problem of $\ell_{p}$ norm linear regression,
which has several applications such as in sparse recovery, data clustering, and
semi-supervised learning. The problem, even though convex, does not enjoy a
closed-form solution. The state-of-the-art algorithms are iterative but suffer
from convergence issues, i.e., they either diverge for p>3 or the convergence
to the optimal solution is sensitive to the initialization of the algorithm.
Also, these algorithms are not generalizable to every possible value of $p$. In
this paper, we propose an iterative algorithm : Parallel IteRative AlgOrithM
for $\ell_{P}$ norm regression via MajorizaTion Minimization (PROMPT) based on
the principle of Majorization Minimization and prove that the proposed
algorithm is monotonic and converges to the optimal solution of the problem for
any value of $p$. The proposed algorithm can also parallelly update each
element of the regression variable, which helps to handle large scale data
efficiently, a common scenario in this era of data explosion. Subsequently, we
show that the proposed algorithm can also be applied for the graph based
semi-supervised learning problem. We show through numerical simulations that
the proposed algorithm converges to the optimal solution for any random
initialization and also performs better than the state-of-the-art algorithms in
terms of speed of convergence. We also evaluate the performance of the proposed
algorithm using simulated and real data for the graph based semi-supervised
learning problem.

    

### [[2110.12192] Dual Shape Guided Segmentation Network for Organs-at-Risk in Head and Neck CT Images](http://arxiv.org/abs/2110.12192)


  The accurate segmentation of organs-at-risk (OARs) in head and neck CT images
is a critical step for radiation therapy of head and neck cancer patients.
However, manual delineation for numerous OARs is time-consuming and laborious,
even for expert oncologists. Moreover, manual delineation results are
susceptible to high intra- and inter-variability. To this end, we propose a
novel dual shape guided network (DSGnet) to automatically delineate nine
important OARs in head and neck CT images. To deal with the large shape
variation and unclear boundary of OARs in CT images, we represent the organ
shape using an organ-specific unilateral inverse-distance map (UIDM) and guide
the segmentation task from two different perspectives: direct shape guidance by
following the segmentation prediction and across shape guidance by sharing the
segmentation feature. In the direct shape guidance, the segmentation prediction
is not only supervised by the true label mask, but also by the true UIDM, which
is implemented through a simple yet effective encoder-decoder mapping from the
label space to the distance space. In the across shape guidance, UIDM is used
to facilitate the segmentation by optimizing the shared feature maps. For the
experiments, we build a large head and neck CT dataset with a total of 699
images from different volunteers, and conduct comprehensive experiments and
comparisons with other state-of-the-art methods to justify the effectiveness
and efficiency of our proposed method. The overall Dice Similarity Coefficient
(DSC) value of 0.842 across the nine important OARs demonstrates great
potential applications in improving the delineation quality and reducing the
time cost.

    

### [[2110.12197] Towards a Robust Differentiable Architecture Search under Label Noise](http://arxiv.org/abs/2110.12197)


  Neural Architecture Search (NAS) is the game changer in designing robust
neural architectures. Architectures designed by NAS outperform or compete with
the best manual network designs in terms of accuracy, size, memory footprint
and FLOPs. That said, previous studies focus on developing NAS algorithms for
clean high quality data, a restrictive and somewhat unrealistic assumption. In
this paper, focusing on the differentiable NAS algorithms, we show that vanilla
NAS algorithms suffer from a performance loss if class labels are noisy. To
combat this issue, we make use of the principle of information bottleneck as a
regularizer. This leads us to develop a noise injecting operation that is
included during the learning process, preventing the network from learning from
noisy samples. Our empirical evaluations show that the noise injecting
operation does not degrade the performance of the NAS algorithm if the data is
indeed clean. In contrast, if the data is noisy, the architecture learned by
our algorithm comfortably outperforms algorithms specifically equipped with
sophisticated mechanisms to learn in the presence of label noise. In contrast
to many algorithms designed to work in the presence of noisy labels, prior
knowledge about the properties of the noise and its characteristics are not
required for our algorithm.

    

### [[2110.12200] Hate and Offensive Speech Detection in Hindi and Marathi](http://arxiv.org/abs/2110.12200)


  Sentiment analysis is the most basic NLP task to determine the polarity of
text data. There has been a significant amount of work in the area of
multilingual text as well. Still hate and offensive speech detection faces a
challenge due to inadequate availability of data, especially for Indian
languages like Hindi and Marathi. In this work, we consider hate and offensive
speech detection in Hindi and Marathi texts. The problem is formulated as a
text classification task using the state of the art deep learning approaches.
We explore different deep learning architectures like CNN, LSTM, and variations
of BERT like multilingual BERT, IndicBERT, and monolingual RoBERTa. The basic
models based on CNN and LSTM are augmented with fast text word embeddings. We
use the HASOC 2021 Hindi and Marathi hate speech datasets to compare these
algorithms. The Marathi dataset consists of binary labels and the Hindi dataset
consists of binary as well as more-fine grained labels. We show that the
transformer-based models perform the best and even the basic models along with
FastText embeddings give a competitive performance. Moreover, with normal
hyper-parameter tuning, the basic models perform better than BERT-based models
on the fine-grained Hindi dataset.

    

### [[2110.12216] Domain Adaptation for Rare Classes Augmented with Synthetic Samples](http://arxiv.org/abs/2110.12216)


  To alleviate lower classification performance on rare classes in imbalanced
datasets, a possible solution is to augment the underrepresented classes with
synthetic samples. Domain adaptation can be incorporated in a classifier to
decrease the domain discrepancy between real and synthetic samples. While
domain adaptation is generally applied on completely synthetic source domains
and real target domains, we explore how domain adaptation can be applied when
only a single rare class is augmented with simulated samples. As a testbed, we
use a camera trap animal dataset with a rare deer class, which is augmented
with synthetic deer samples. We adapt existing domain adaptation methods to two
new methods for the single rare class setting: DeerDANN, based on the
Domain-Adversarial Neural Network (DANN), and DeerCORAL, based on deep
correlation alignment (Deep CORAL) architectures. Experiments show that
DeerDANN has the highest improvement in deer classification accuracy of 24.0%
versus 22.4% improvement of DeerCORAL when compared to the baseline. Further,
both methods require fewer than 10k synthetic samples, as used by the baseline,
to achieve these higher accuracies. DeerCORAL requires the least number of
synthetic samples (2k deer), followed by DeerDANN (8k deer).

    

### [[2110.12231] Learning curves for Gaussian process regression with power-law priors and targets](http://arxiv.org/abs/2110.12231)


  We study the power-law asymptotics of learning curves for Gaussian process
regression (GPR). When the eigenspectrum of the prior decays with rate $\alpha$
and the eigenexpansion coefficients of the target function decay with rate
$\beta$, we show that the generalization error behaves as $\tilde
O(n^{\max\{\frac{1}{\alpha}-1, \frac{1-2\beta}{\alpha}\}})$ with high
probability over the draw of $n$ input samples. Under similar assumptions, we
show that the generalization error of kernel ridge regression (KRR) has the
same asymptotics. Infinitely wide neural networks can be related to KRR with
respect to the neural tangent kernel (NTK), which in several cases is known to
have a power-law spectrum. Hence our methods can be applied to study the
generalization error of infinitely wide neural networks. We present toy
experiments demonstrating the theory.

    

### [[2110.12239] Policy Search using Dynamic Mirror Descent MPC for Model Free Off Policy RL](http://arxiv.org/abs/2110.12239)


  Recent works in Reinforcement Learning (RL) combine model-free (Mf)-RL
algorithms with model-based (Mb)-RL approaches to get the best from both:
asymptotic performance of Mf-RL and high sample-efficiency of Mb-RL. Inspired
by these works, we propose a hierarchical framework that integrates online
learning for the Mb-trajectory optimization with off-policy methods for the
Mf-RL. In particular, two loops are proposed, where the Dynamic Mirror Descent
based Model Predictive Control (DMD-MPC) is used as the inner loop to obtain an
optimal sequence of actions. These actions are in turn used to significantly
accelerate the outer loop Mf-RL. We show that our formulation is generic for a
broad class of MPC based policies and objectives, and includes some of the
well-known Mb-Mf approaches. Based on the framework we define two algorithms to
increase sample efficiency of Off Policy RL and to guide end to end RL
algorithms for online adaption respectively. Thus we finally introduce two
novel algorithms: Dynamic-Mirror Descent Model Predictive RL(DeMoRL), which
uses the method of elite fractions for the inner loop and Soft Actor-Critic
(SAC) as the off-policy RL for the outer loop and Dynamic-Mirror Descent Model
Predictive Layer(DeMo Layer), a special case of the hierarchical framework
which guides linear policies trained using Augmented Random Search(ARS). Our
experiments show faster convergence of the proposed DeMo RL, and better or
equal performance compared to other Mf-Mb approaches on benchmark MuJoCo
control tasks. The DeMo Layer was tested on classical Cartpole and custom-built
Quadruped trained using Linear Policy.

    

### [[2110.12246] Parametric Variational Linear Units (PVLUs) in Deep Convolutional Networks](http://arxiv.org/abs/2110.12246)


  The Rectified Linear Unit is currently a state-of-the-art activation function
in deep convolutional neural networks. To combat ReLU's dying neuron problem,
we propose the Parametric Variational Linear Unit (PVLU), which adds a
sinusoidal function with trainable coefficients to ReLU. Along with introducing
nonlinearity and non-zero gradients across the entire real domain, PVLU allows
for increased model generalization and robustness when implemented in the
context of transfer learning. On a simple, non-transfer sequential CNN, PVLU
led to relative error decrease of 16.3% and 11.3% without and with data
augmentation, relative to ReLU. PVLU is also tested on transfer learning
problems. The VGG-16 and VGG-19 models experience relative error reductions of
9.5% and 10.7% on CIFAR-10, respectively, after the substitution of ReLU with
PVLU. When training on Gaussian-filtered CIFAR-10 images, similar improvements
are noted for the VGG models. Most notably, PVLU fine tuning allows for
relative error reductions up to and exceeding 10% on near state-of-the-art
ResNet models for both CIFAR-10 and CIFAR-100.

    

### [[2110.12257] Game of Gradients: Mitigating Irrelevant Clients in Federated Learning](http://arxiv.org/abs/2110.12257)


  The paradigm of Federated learning (FL) deals with multiple clients
participating in collaborative training of a machine learning model under the
orchestration of a central server. In this setup, each client's data is private
to itself and is not transferable to other clients or the server. Though FL
paradigm has received significant interest recently from the research
community, the problem of selecting the relevant clients w.r.t. the central
server's learning objective is under-explored. We refer to these problems as
Federated Relevant Client Selection (FRCS). Because the server doesn't have
explicit control over the nature of data possessed by each client, the problem
of selecting relevant clients is significantly complex in FL settings. In this
paper, we resolve important and related FRCS problems viz., selecting clients
with relevant data, detecting clients that possess data relevant to a
particular target label, and rectifying corrupted data samples of individual
clients. We follow a principled approach to address the above FRCS problems and
develop a new federated learning method using the Shapley value concept from
cooperative game theory. Towards this end, we propose a cooperative game
involving the gradients shared by the clients. Using this game, we compute
Shapley values of clients and then present Shapley value based Federated
Averaging (S-FedAvg) algorithm that empowers the server to select relevant
clients with high probability. S-FedAvg turns out to be critical in designing
specific algorithms to address the FRCS problems. We finally conduct a thorough
empirical analysis on image classification and speech recognition tasks to show
the superior performance of S-FedAvg than the baselines in the context of
supervised federated learning settings.

    

### [[2110.12259] In Search of Probeable Generalization Measures](http://arxiv.org/abs/2110.12259)


  Understanding the generalization behaviour of deep neural networks is a topic
of recent interest that has driven the production of many studies, notably the
development and evaluation of generalization "explainability" measures that
quantify model generalization ability. Generalization measures have also proven
useful in the development of powerful layer-wise model tuning and optimization
algorithms, though these algorithms require specific kinds of generalization
measures which can probe individual layers. The purpose of this paper is to
explore the neglected subtopic of probeable generalization measures; to
establish firm ground for further investigations, and to inspire and guide the
development of novel model tuning and optimization algorithms. We evaluate and
compare measures, demonstrating effectiveness and robustness across model
variations, dataset complexities, training hyperparameters, and training
stages. We also introduce a new dataset of trained models and performance
metrics, GenProb, for testing generalization measures, model tuning algorithms
and optimization algorithms.

    

### [[2110.12270] Benchmarking of Lightweight Deep Learning Architectures for Skin Cancer Classification using ISIC 2017 Dataset](http://arxiv.org/abs/2110.12270)


  Skin cancer is one of the deadly types of cancer and is common in the world.
Recently, there has been a huge jump in the rate of people getting skin cancer.
For this reason, the number of studies on skin cancer classification with deep
learning are increasing day by day. For the growth of work in this area, the
International Skin Imaging Collaboration (ISIC) organization was established
and they created an open dataset archive. In this study, images were taken from
ISIC 2017 Challenge. The skin cancer images taken were preprocessed and data
augmented. Later, these images were trained with transfer learning and
fine-tuning approach and deep learning models were created in this way. 3
different mobile deep learning models and 3 different batch size values were
determined for each, and a total of 9 models were created. Among these models,
the NASNetMobile model with 16 batch size got the best result. The accuracy
value of this model is 82.00%, the precision value is 81.77% and the F1 score
value is 0.8038. Our method is to benchmark mobile deep learning models which
have few parameters and compare the results of the models.

    

### [[2110.12271] Self-Validation: Early Stopping for Single-Instance Deep Generative Priors](http://arxiv.org/abs/2110.12271)


  Recent works have shown the surprising effectiveness of deep generative
models in solving numerous image reconstruction (IR) tasks, even without
training data. We call these models, such as deep image prior and deep decoder,
collectively as single-instance deep generative priors (SIDGPs). The successes,
however, often hinge on appropriate early stopping (ES), which by far has
largely been handled in an ad-hoc manner. In this paper, we propose the first
principled method for ES when applying SIDGPs to IR, taking advantage of the
typical bell trend of the reconstruction quality. In particular, our method is
based on collaborative training and self-validation: the primal reconstruction
process is monitored by a deep autoencoder, which is trained online with the
historic reconstructed images and used to validate the reconstruction quality
constantly. Experimentally, on several IR problems and different SIDGPs, our
self-validation method is able to reliably detect near-peak performance and
signal good ES points. Our code is available at
this https URL.

    

### [[2110.12275] Signal to Noise Ratio Loss Function](http://arxiv.org/abs/2110.12275)


  This work proposes a new loss function targeting classification problems,
utilizing a source of information overlooked by cross entropy loss. First, we
derive a series of the tightest upper and lower bounds for the probability of a
random variable in a given interval. Second, a lower bound is proposed for the
probability of a true positive for a parametric classification problem, where
the form of probability density function (pdf) of data is given. A closed form
for finding the optimal function of unknowns is derived to maximize the
probability of true positives. Finally, for the case that the pdf of data is
unknown, we apply the proposed boundaries to find the lower bound of the
probability of true positives and upper bound of the probability of false
positives and optimize them using a loss function which is given by combining
the boundaries. We demonstrate that the resultant loss function is a function
of the signal to noise ratio both within and across logits. We empirically
evaluate our proposals to show their benefit for classification problems.

    

### [[2110.12276] Coarse-Grained Smoothness for RL in Metric Spaces](http://arxiv.org/abs/2110.12276)


  Principled decision-making in continuous state--action spaces is impossible
without some assumptions. A common approach is to assume Lipschitz continuity
of the Q-function. We show that, unfortunately, this property fails to hold in
many typical domains. We propose a new coarse-grained smoothness definition
that generalizes the notion of Lipschitz continuity, is more widely applicable,
and allows us to compute significantly tighter bounds on Q-functions, leading
to improved learning. We provide a theoretical analysis of our new smoothness
definition, and discuss its implications and impact on control and exploration
in continuous domains.

    

### [[2110.12279] Hierarchical Few-Shot Generative Models](http://arxiv.org/abs/2110.12279)


  A few-shot generative model should be able to generate data from a
distribution by only observing a limited set of examples. In few-shot learning
the model is trained on data from many sets from different distributions
sharing some underlying properties such as sets of characters from different
alphabets or sets of images of different type objects. We study a latent
variables approach that extends the Neural Statistician to a fully hierarchical
approach with an attention-based point to set-level aggregation. We extend the
previous work to iterative data sampling, likelihood-based model comparison,
and adaptation-free out of distribution generalization. Our results show that
the hierarchical formulation better captures the intrinsic variability within
the sets in the small data regime. With this work we generalize deep latent
variable approaches to few-shot learning, taking a step towards large-scale
few-shot generation with a formulation that readily can work with current
state-of-the-art deep generative models.

    

### [[2110.12285] Generalized Resubstitution for Classification Error Estimation](http://arxiv.org/abs/2110.12285)


  We propose the family of generalized resubstitution classifier error
estimators based on empirical measures. These error estimators are
computationally efficient and do not require re-training of classifiers. The
plain resubstitution error estimator corresponds to choosing the standard
empirical measure. Other choices of empirical measure lead to bolstered,
posterior-probability, Gaussian-process, and Bayesian error estimators; in
addition, we propose bolstered posterior-probability error estimators as a new
family of generalized resubstitution estimators. In the two-class case, we show
that a generalized resubstitution estimator is consistent and asymptotically
unbiased, regardless of the distribution of the features and label, if the
corresponding generalized empirical measure converges uniformly to the standard
empirical measure and the classification rule has a finite VC dimension. A
generalized resubstitution estimator typically has hyperparameters that can be
tuned to control its bias and variance, which adds flexibility. Numerical
experiments with various classification rules trained on synthetic data assess
the thefinite-sample performance of several representative generalized
resubstitution error estimators. In addition, results of an image
classification experiment using the LeNet-5 convolutional neural network and
the MNIST data set demonstrate the potential of this class of error estimators
in deep learning for computer vision.

    

### [[2110.12288] Path Signature Area-Based Causal Discovery in Coupled Time Series](http://arxiv.org/abs/2110.12288)


  Coupled dynamical systems are frequently observed in nature, but often not
well understood in terms of their causal structure without additional domain
knowledge about the system. Especially when analyzing observational time series
data of dynamical systems where it is not possible to conduct controlled
experiments, for example time series of climate variables, it can be
challenging to determine how features causally influence each other. There are
many techniques available to recover causal relationships from data, such as
Granger causality, convergent cross mapping, and causal graph structure
learning approaches such as PCMCI. Path signatures and their associated signed
areas provide a new way to approach the analysis of causally linked dynamical
systems, particularly in informing a model-free, data-driven approach to
algorithmic causal discovery. With this paper, we explore the use of path
signatures in causal discovery and propose the application of confidence
sequences to analyze the significance of the magnitude of the signed area
between two variables. These confidence sequence regions converge with greater
sampling length, and in conjunction with analyzing pairwise signed areas across
time-shifted versions of the time series, can help identify the presence of
lag/lead causal relationships. This approach provides a new way to define the
confidence of a causal link existing between two time series, and ultimately
may provide a framework for hypothesis testing to define whether one time
series causes another

    

### [[2110.12292] Federated Multiple Label Hashing (FedMLH): Communication Efficient Federated Learning on Extreme Classification Tasks](http://arxiv.org/abs/2110.12292)


  Federated learning enables many local devices to train a deep learning model
jointly without sharing the local data. Currently, most of federated training
schemes learns a global model by averaging the parameters of local models.
However, most of these training schemes suffer from high communication cost
resulted from transmitting full local model parameters. Moreover, directly
averaging model parameters leads to a significant performance degradation, due
to the class-imbalanced non-iid data on different devices. Especially for the
real life federated learning tasks involving extreme classification, (1)
communication becomes the main bottleneck since the model size increases
proportionally to the number of output classes; (2) extreme classification
(such as user recommendation) normally have extremely imbalanced classes and
heterogeneous data on different devices. To overcome this problem, we propose
federated multiple label hashing (FedMLH), which leverages label hashing to
simultaneously reduce the model size (up to 3.40X decrease) with communication
cost (up to 18.75X decrease) and achieves significant better accuracy (up to
35.5%} relative accuracy improvement) and faster convergence rate (up to 5.5X
increase) for free on the federated extreme classification tasks compared to
federated average algorithm.

    

### [[2110.12301] Map Induction: Compositional spatial submap learning for efficient exploration in novel environments](http://arxiv.org/abs/2110.12301)


  Humans are expert explorers. Understanding the computational cognitive
mechanisms that support this efficiency can advance the study of the human mind
and enable more efficient exploration algorithms. We hypothesize that humans
explore new environments efficiently by inferring the structure of unobserved
spaces using spatial information collected from previously explored spaces.
This cognitive process can be modeled computationally using program induction
in a Hierarchical Bayesian framework that explicitly reasons about uncertainty
with strong spatial priors. Using a new behavioral Map Induction Task, we
demonstrate that this computational framework explains human exploration
behavior better than non-inductive models and outperforms state-of-the-art
planning algorithms when applied to a realistic spatial navigation domain.

    

### [[2110.12306] Fully Distributed Actor-Critic Architecture for Multitask Deep Reinforcement Learning](http://arxiv.org/abs/2110.12306)


  We propose a fully distributed actor-critic architecture, named Diff-DAC,
with application to multitask reinforcement learning (MRL). During the learning
process, agents communicate their value and policy parameters to their
neighbours, diffusing the information across a network of agents with no need
for a central station. Each agent can only access data from its local task, but
aims to learn a common policy that performs well for the whole set of tasks.
The architecture is scalable, since the computational and communication cost
per agent depends on the number of neighbours rather than the overall number of
agents. We derive Diff-DAC from duality theory and provide novel insights into
the actor-critic framework, showing that it is actually an instance of the dual
ascent method. We prove almost sure convergence of Diff-DAC to a common policy
under general assumptions that hold even for deep-neural network
approximations. For more restrictive assumptions, we also prove that this
common policy is a stationary point of an approximation of the original
problem. Numerical results on multitask extensions of common continuous control
benchmarks demonstrate that Diff-DAC stabilises learning and has a regularising
effect that induces higher performance and better generalisation properties
than previous architectures.

    

### [[2110.12308] A Layer-wise Adversarial-aware Quantization Optimization for Improving Robustness](http://arxiv.org/abs/2110.12308)


  Neural networks are getting better accuracy with higher energy and
computational cost. After quantization, the cost can be greatly saved, and the
quantized models are more hardware friendly with acceptable accuracy loss. On
the other hand, recent research has found that neural networks are vulnerable
to adversarial attacks, and the robustness of a neural network model can only
be improved with defense methods, such as adversarial training. In this work,
we find that adversarially-trained neural networks are more vulnerable to
quantization loss than plain models. To minimize both the adversarial and the
quantization losses simultaneously and to make the quantized model robust, we
propose a layer-wise adversarial-aware quantization method, using the Lipschitz
constant to choose the best quantization parameter settings for a neural
network. We theoretically derive the losses and prove the consistency of our
metric selection. The experiment results show that our method can effectively
and efficiently improve the robustness of quantized adversarially-trained
neural networks.

    

### [[2110.12311] Vector Optimization with Stochastic Bandit Feedback](http://arxiv.org/abs/2110.12311)


  We introduce vector optimization problems with stochastic bandit feedback,
which extends the best arm identification problem to vector-valued rewards. We
consider $K$ designs, with multi-dimensional mean reward vectors, which are
partially ordered according to a polyhedral ordering cone $C$. This generalizes
the concept of Pareto set in multi-objective optimization and allows different
sets of preferences of decision-makers to be encoded by $C$. Different than
prior work, we define approximations of the Pareto set based on direction-free
covering and gap notions. We study the setting where an evaluation of each
design yields a noisy observation of the mean reward vector. Under subgaussian
noise assumption, we investigate the sample complexity of the naïve
elimination algorithm in an ($\epsilon,\delta$)-PAC setting, where the goal is
to identify an ($\epsilon,\delta$)-PAC Pareto set with the minimum number of
design evaluations. In particular, we identify cone-dependent geometric
conditions on the deviations of empirical reward vectors from their mean under
which the Pareto front can be approximated accurately. We run experiments to
verify our theoretical results and illustrate how $C$ and sampling budget
affect the Pareto set, returned ($\epsilon,\delta$)-PAC Pareto set and the
success of identification.

    

### [[2110.12319] Non-Asymptotic Error Bounds for Bidirectional GANs](http://arxiv.org/abs/2110.12319)


  We derive nearly sharp bounds for the bidirectional GAN (BiGAN) estimation
error under the Dudley distance between the latent joint distribution and the
data joint distribution with appropriately specified architecture of the neural
networks used in the model. To the best of our knowledge, this is the first
theoretical guarantee for the bidirectional GAN learning approach. An appealing
feature of our results is that they do not assume the reference and the data
distributions to have the same dimensions or these distributions to have
bounded support. These assumptions are commonly assumed in the existing
convergence analysis of the unidirectional GANs but may not be satisfied in
practice. Our results are also applicable to the Wasserstein bidirectional GAN
if the target distribution is assumed to have a bounded support. To prove these
results, we construct neural network functions that push forward an empirical
distribution to another arbitrary empirical distribution on a possibly
different-dimensional space. We also develop a novel decomposition of the
integral probability metric for the error analysis of bidirectional GANs. These
basic theoretical results are of independent interest and can be applied to
other related learning problems.

    

### [[2110.12321] ADC: Adversarial attacks against object Detection that evade Context consistency checks](http://arxiv.org/abs/2110.12321)


  Deep Neural Networks (DNNs) have been shown to be vulnerable to adversarial
examples, which are slightly perturbed input images which lead DNNs to make
wrong predictions. To protect from such examples, various defense strategies
have been proposed. A very recent defense strategy for detecting adversarial
examples, that has been shown to be robust to current attacks, is to check for
intrinsic context consistencies in the input data, where context refers to
various relationships (e.g., object-to-object co-occurrence relationships) in
images. In this paper, we show that even context consistency checks can be
brittle to properly crafted adversarial examples and to the best of our
knowledge, we are the first to do so. Specifically, we propose an adaptive
framework to generate examples that subvert such defenses, namely, Adversarial
attacks against object Detection that evade Context consistency checks (ADC).
In ADC, we formulate a joint optimization problem which has two attack goals,
viz., (i) fooling the object detector and (ii) evading the context consistency
check system, at the same time. Experiments on both PASCAL VOC and MS COCO
datasets show that examples generated with ADC fool the object detector with a
success rate of over 85% in most cases, and at the same time evade the recently
proposed context consistency checks, with a bypassing rate of over 80% in most
cases. Our results suggest that how to robustly model context and check its
consistency, is still an open problem.

    

### [[2110.12328] Improving Spectral Clustering Using Spectrum-Preserving Node Reduction](http://arxiv.org/abs/2110.12328)


  Spectral clustering is one of the most popular clustering methods. However,
the high computational cost due to the involved eigen-decomposition procedure
can immediately hinder its applications in large-scale tasks. In this paper we
use spectrum-preserving node reduction to accelerate eigen-decomposition and
generate concise representations of data sets. Specifically, we create a small
number of pseudonodes based on spectral similarity. Then, standard spectral
clustering algorithm is performed on the smaller node set. Finally, each data
point in the original data set is assigned to the cluster as its representative
pseudo-node. The proposed framework run in nearly-linear time. Meanwhile, the
clustering accuracy can be significantly improved by mining concise
representations. The experimental results show dramatically improved clustering
performance when compared with state-of-the-art methods.

    

### [[2110.12329] The network signature of constellation line figures](http://arxiv.org/abs/2110.12329)


  In traditional astronomies across the world, groups of stars in the night sky
were linked into constellations -- symbolic representations on the celestial
sphere, rich in meaning and with practical roles. In cultures where line or
connect-the-dot figures were documented, these visual representations are
constrained to the fixed background of stars, but are free in their choice of
stars and lines to draw. Over a dataset of 1591 constellation line figures from
50 astronomical cultures, we define metrics to measure the visual signature (or
complexity) of a constellation, and answer two questions: (1) does the type of
culture associate with the visual signature of constellations? 2) does the sky
region associate with the visual signature of constellations? We find that (1)
individual cultures are only rarely and weakly thus associated, but the type of
culture (by practical use, level of development, and ancestry) show an
association. We find clear clusters of cross-culture and cross-type similarity
in visual signatures, with SE Asian traditions far apart from Mesopotamian, N
and S American, Austronesian and Polynesian traditions, which are similar. We
also find (2) more diversity of constellation signature per sky region than
expected, with diverse designs around the majority of popular stars.

    

### [[2110.12343] Off-Policy Evaluation in Partially Observed Markov Decision Processes](http://arxiv.org/abs/2110.12343)


  We consider off-policy evaluation of dynamic treatment rules under the
assumption that the underlying system can be modeled as a partially observed
Markov decision process (POMDP). We propose an estimator, partial history
importance weighting, and show that it can consistently estimate the stationary
mean rewards of a target policy given long enough draws from the behavior
policy. Furthermore, we establish an upper bound on its error that decays
polynomially in the number of observations (i.e., the number of trajectories
times their length), with an exponent that depends on the overlap of the target
and behavior policies, and on the mixing time of the underlying system. We also
establish a polynomial minimax lower bound for off-policy evaluation under the
POMDP assumption, and show that its exponent has the same qualitative
dependence on overlap and mixing time as obtained in our upper bound. Together,
our upper and lower bounds imply that off-policy evaluation in POMDPs is
strictly harder than off-policy evaluation in (fully observed) Markov decision
processes, but strictly easier than model-free off-policy evaluation.

    

### [[2110.12344] A Broader Picture of Random-walk Based Graph Embedding](http://arxiv.org/abs/2110.12344)


  Graph embedding based on random-walks supports effective solutions for many
graph-related downstream tasks. However, the abundance of embedding literature
has made it increasingly difficult to compare existing methods and to identify
opportunities to advance the state-of-the-art. Meanwhile, existing work has
left several fundamental questions -- such as how embeddings capture different
structural scales and how they should be applied for effective link prediction
-- unanswered. This paper addresses these challenges with an analytical
framework for random-walk based graph embedding that consists of three
components: a random-walk process, a similarity function, and an embedding
algorithm. Our framework not only categorizes many existing approaches but
naturally motivates new ones. With it, we illustrate novel ways to incorporate
embeddings at multiple scales to improve downstream task performance. We also
show that embeddings based on autocovariance similarity, when paired with dot
product ranking for link prediction, outperform state-of-the-art methods based
on Pointwise Mutual Information similarity by up to 100%.

    

### [[2110.12347] Acceleration in Distributed Optimization Under Similarity](http://arxiv.org/abs/2110.12347)


  We study distributed (strongly convex) optimization problems over a network
of agents, with no centralized nodes. The loss functions of the agents are
assumed to be similar, due to statistical data similarity or otherwise. In
order to reduce the number of communications to reach a solution accuracy, we
proposed a preconditioned, accelerated distributed method. An
$\varepsilon$-solution is achieved in
$\tilde{\mathcal{O}}\big(\sqrt{\frac{\beta/\mu}{(1-\rho)}}\log1/\varepsilon\big)$
number of communications steps, where $\beta/\mu$ is the relative condition
number between the global and local loss functions, and $\rho$ characterizes
the connectivity of the network. This rate matches (up to poly-log factors) for
the first time lower complexity communication bounds of distributed
gossip-algorithms applied to the class of problems of interest. Numerical
results show significant communication savings with respect to existing
accelerated distributed schemes, especially when solving ill-conditioned
problems.

    

### [[2110.12351] Integrated Conditional Estimation-Optimization](http://arxiv.org/abs/2110.12351)


  Many real-world optimization problems involve uncertain parameters with
probability distributions that can be estimated using contextual feature
information. In contrast to the standard approach of first estimating the
distribution of uncertain parameters and then optimizing the objective based on
the estimation, we propose an integrated conditional estimation-optimization
(ICEO) framework that estimates the underlying conditional distribution of the
random parameter while considering the structure of the optimization problem.
We directly model the relationship between the conditional distribution of the
random parameter and the contextual features, and then estimate the
probabilistic model with an objective that aligns with the downstream
optimization problem. We show that our ICEO approach is asymptotically
consistent under moderate regularity conditions and further provide finite
performance guarantees in the form of generalization bounds. Computationally,
performing estimation with the ICEO approach is a non-convex and often
non-differentiable optimization problem. We propose a general methodology for
approximating the potentially non-differentiable mapping from estimated
conditional distribution to the optimal decision by a differentiable function,
which greatly improves the performance of gradient-based algorithms applied to
the non-convex problem. We also provide a polynomial optimization solution
approach in the semi-algebraic case. Numerical experiments are also conducted
to show the empirical success of our approach in different situations including
with limited data samples and model mismatches.

    

### [[2110.12357] Towards A Conceptually Simple Defensive Approach for Few-shot classifiers Against Adversarial Support Samples](http://arxiv.org/abs/2110.12357)


  Few-shot classifiers have been shown to exhibit promising results in use
cases where user-provided labels are scarce. These models are able to learn to
predict novel classes simply by training on a non-overlapping set of classes.
This can be largely attributed to the differences in their mechanisms as
compared to conventional deep networks. However, this also offers new
opportunities for novel attackers to induce integrity attacks against such
models, which are not present in other machine learning setups. In this work,
we aim to close this gap by studying a conceptually simple approach to defend
few-shot classifiers against adversarial attacks. More specifically, we propose
a simple attack-agnostic detection method, using the concept of self-similarity
and filtering, to flag out adversarial support sets which destroy the
understanding of a victim classifier for a certain class. Our extended
evaluation on the miniImagenet (MI) and CUB datasets exhibit good attack
detection performance, across three different few-shot classifiers and across
different attack strengths, beating baselines. Our observed results allow our
approach to establishing itself as a strong detection method for support set
poisoning attacks. We also show that our approach constitutes a generalizable
concept, as it can be paired with other filtering functions. Finally, we
provide an analysis of our results when we vary two components found in our
detection approach.

    

### [[2110.12359] Encoding Integrated Decision and Control for Autonomous Driving with Mixed Traffic Flow](http://arxiv.org/abs/2110.12359)


  Reinforcement learning (RL) has been widely adopted to make intelligent
driving policy in autonomous driving due to the self-evolution ability and
humanoid learning paradigm. Despite many elegant demonstrations of RL-enabled
decision-making, current research mainly focuses on the pure vehicle driving
environment while ignoring other traffic participants like bicycles and
pedestrians. For urban roads, the interaction of mixed traffic flows leads to a
quite dynamic and complex relationship, which poses great difficulty to learn a
safe and intelligent policy. This paper proposes the encoding integrated
decision and control (E-IDC) to handle complicated driving tasks with mixed
traffic flows, which composes of an encoding function to construct driving
states, a value function to choose the optimal path as well as a policy
function to output the control command of ego vehicle. Specially, the encoding
function is capable of dealing with different types and variant number of
traffic participants and extracting features from original driving observation.
Next, we design the training principle for the functions of E-IDC with RL
algorithms by adding the gradient-based update rules and refine the safety
constraints concerning the otherness of different participants. The
verification is conducted on the intersection scenario with mixed traffic flows
and result shows that E-IDC can enhance the driving performance, including the
tracking performance and safety constraint requirements with a large margin.
The online application indicates that E-IDC can realize efficient and smooth
driving in the complex intersection, guaranteeing the intelligence and safety
simultaneously.

    

### [[2110.12364] CvT-ASSD: Convolutional vision-Transformer Based Attentive Single Shot MultiBox Detector](http://arxiv.org/abs/2110.12364)


  Due to the success of Bidirectional Encoder Representations from Transformers
(BERT) in natural language process (NLP), the multi-head attention transformer
has been more and more prevalent in computer-vision researches (CV). However,
it still remains a challenge for researchers to put forward complex tasks such
as vision detection and semantic segmentation. Although multiple
Transformer-Based architectures like DETR and ViT-FRCNN have been proposed to
complete object detection task, they inevitably decreases discrimination
accuracy and brings down computational efficiency caused by the enormous
learning parameters and heavy computational complexity incurred by the
traditional self-attention operation. In order to alleviate these issues, we
present a novel object detection architecture, named Convolutional vision
Transformer Based Attentive Single Shot MultiBox Detector (CvT-ASSD), that
built on the top of Convolutional vision Transormer (CvT) with the efficient
Attentive Single Shot MultiBox Detector (ASSD). We provide comprehensive
empirical evidence showing that our model CvT-ASSD can leads to good system
efficiency and performance while being pretrained on large-scale detection
datasets such as PASCAL VOC and MS COCO. Code has been released on public
github repository at this https URL.

    

### [[2110.12365] Conditional Generation of Periodic Signals with Fourier-Based Decoder](http://arxiv.org/abs/2110.12365)


  Periodic signals play an important role in daily lives. Although conventional
sequential models have shown remarkable success in various fields, they still
come short in modeling periodicity; they either collapse, diverge or ignore
details. In this paper, we introduce a novel framework inspired by Fourier
series to generate periodic signals. We first decompose the given signals into
multiple sines and cosines and then conditionally generate periodic signals
with the output components. We have shown our model efficacy on three tasks:
reconstruction, imputation and conditional generation. Our model outperforms
baselines in all tasks and shows more stable and refined results.

    

### [[2110.12367] Deep Learning for Simultaneous Inference of Hydraulic and Transport Properties](http://arxiv.org/abs/2110.12367)


  Identifying the heterogeneous conductivity field and reconstructing the
contaminant release history are key aspects of subsurface remediation.
Achieving these two goals with limited and noisy hydraulic head and
concentration measurements is challenging. The obstacles include solving an
inverse problem for high-dimensional parameters, and the high-computational
cost needed for the repeated forward modeling. We use a convolutional
adversarial autoencoder (CAAE) for the parameterization of the heterogeneous
non-Gaussian conductivity field with a low-dimensional latent representation.
Additionally, we trained a three-dimensional dense convolutional
encoder-decoder (DenseED) network to serve as the forward surrogate for the
flow and transport processes. Combining the CAAE and DenseED forward surrogate
models, the ensemble smoother with multiple data assimilation (ESMDA) algorithm
is used to sample from the Bayesian posterior distribution of the unknown
parameters, forming a CAAE-DenseED-ESMDA inversion framework. We applied this
CAAE-DenseED-ESMDA inversion framework in a three-dimensional contaminant
source and conductivity field identification problem. A comparison of the
inversion results from CAAE-ESMDA with physical flow and transport simulator
and CAAE-DenseED-ESMDA is provided, showing that accurate reconstruction
results were achieved with a much higher computational efficiency.

    

### [[2110.12369] AuxAdapt: Stable and Efficient Test-Time Adaptation for Temporally Consistent Video Semantic Segmentation](http://arxiv.org/abs/2110.12369)


  In video segmentation, generating temporally consistent results across frames
is as important as achieving frame-wise accuracy. Existing methods rely either
on optical flow regularization or fine-tuning with test data to attain temporal
consistency. However, optical flow is not always avail-able and reliable.
Besides, it is expensive to compute. Fine-tuning the original model in test
time is cost sensitive.
This paper presents an efficient, intuitive, and unsupervised online
adaptation method, AuxAdapt, for improving the temporal consistency of most
neural network models. It does not require optical flow and only takes one pass
of the video. Since inconsistency mainly arises from the model's uncertainty in
its output, we propose an adaptation scheme where the model learns from its own
segmentation decisions as it streams a video, which allows producing more
confident and temporally consistent labeling for similarly-looking pixels
across frames. For stability and efficiency, we leverage a small auxiliary
segmentation network (AuxNet) to assist with this adaptation. More
specifically, AuxNet readjusts the decision of the original segmentation
network (Main-Net) by adding its own estimations to that of MainNet. At every
frame, only AuxNet is updated via back-propagation while keeping MainNet fixed.
We extensively evaluate our test-time adaptation approach on standard video
benchmarks, including Cityscapes, CamVid, and KITTI. The results demonstrate
that our approach provides label-wise accurate, temporally consistent, and
computationally efficient adaptation (5+ folds overhead reduction comparing to
state-of-the-art test-time adaptation methods).

    

### [[2110.12377] SenseMag: Enabling Low-Cost Traffic Monitoring using Non-invasive Magnetic Sensing](http://arxiv.org/abs/2110.12377)


  The operation and management of intelligent transportation systems (ITS),
such as traffic monitoring, relies on real-time data aggregation of vehicular
traffic information, including vehicular types (e.g., cars, trucks, and buses),
in the critical roads and highways. While traditional approaches based on
vehicular-embedded GPS sensors or camera networks would either invade drivers'
privacy or require high deployment cost, this paper introduces a low-cost
method, namely SenseMag, to recognize the vehicular type using a pair of
non-invasive magnetic sensors deployed on the straight road section. SenseMag
filters out noises and segments received magnetic signals by the exact time
points that the vehicle arrives or departs from every sensor node. Further,
SenseMag adopts a hierarchical recognition model to first estimate the
speed/velocity, then identify the length of vehicle using the predicted speed,
sampling cycles, and the distance between the sensor nodes. With the vehicle
length identified and the temporal/spectral features extracted from the
magnetic signals, SenseMag classify the types of vehicles accordingly. Some
semi-automated learning techniques have been adopted for the design of filters,
features, and the choice of hyper-parameters. Extensive experiment based on
real-word field deployment (on the highways in Shenzhen, China) shows that
SenseMag significantly outperforms the existing methods in both classification
accuracy and the granularity of vehicle types (i.e., 7 types by SenseMag versus
4 types by the existing work in comparisons). To be specific, our field
experiment results validate that SenseMag is with at least $90\%$ vehicle type
classification accuracy and less than 5\% vehicle length classification error.

    

### [[2110.12379] Boson sampling discrete solitons by quantum machine learning](http://arxiv.org/abs/2110.12379)


  We use a neural network variational ansatz to compute Gaussian quantum
discrete solitons in an array of waveguides described by the quantum discrete
nonlinear Schroedinger equation. By training the quantum machine learning model
in the phase space, we find different quantum soliton solutions varying the
number of particles and interaction strength. The use of Gaussian states
enables measuring the degree of entanglement and the boson sampling patterns.
We compute the probability of generating different particle pairs when varying
the soliton features and unveil that bound states of discrete solitons emit
correlated pairs of photons. These results may have a role in boson sampling
experiments with nonlinear systems and in developing quantum processors to
generate entangled many-photon nonlinear states.

    

### [[2110.12381] Regularizing Variational Autoencoder with Diversity and Uncertainty Awareness](http://arxiv.org/abs/2110.12381)


  As one of the most popular generative models, Variational Autoencoder (VAE)
approximates the posterior of latent variables based on amortized variational
inference. However, when the decoder network is sufficiently expressive, VAE
may lead to posterior collapse; that is, uninformative latent representations
may be learned. To this end, in this paper, we propose an alternative model,
DU-VAE, for learning a more Diverse and less Uncertain latent space, and thus
the representation can be learned in a meaningful and compact manner.
Specifically, we first theoretically demonstrate that it will result in better
latent space with high diversity and low uncertainty awareness by controlling
the distribution of posterior's parameters across the whole data accordingly.
Then, without the introduction of new loss terms or modifying training
strategies, we propose to exploit Dropout on the variances and
Batch-Normalization on the means simultaneously to regularize their
distributions implicitly. Furthermore, to evaluate the generalization effect,
we also exploit DU-VAE for inverse autoregressive flow based-VAE (VAE-IAF)
empirically. Finally, extensive experiments on three benchmark datasets clearly
show that our approach can outperform state-of-the-art baselines on both
likelihood estimation and underlying classification tasks.

    

### [[2110.12385] Perceptual Consistency in Video Segmentation](http://arxiv.org/abs/2110.12385)


  In this paper, we present a novel perceptual consistency perspective on video
semantic segmentation, which can capture both temporal consistency and
pixel-wise correctness. Given two nearby video frames, perceptual consistency
measures how much the segmentation decisions agree with the pixel
correspondences obtained via matching general perceptual features. More
specifically, for each pixel in one frame, we find the most perceptually
correlated pixel in the other frame. Our intuition is that such a pair of
pixels are highly likely to belong to the same class. Next, we assess how much
the segmentation agrees with such perceptual correspondences, based on which we
derive the perceptual consistency of the segmentation maps across these two
frames. Utilizing perceptual consistency, we can evaluate the temporal
consistency of video segmentation by measuring the perceptual consistency over
consecutive pairs of segmentation maps in a video. Furthermore, given a
sparsely labeled test video, perceptual consistency can be utilized to aid with
predicting the pixel-wise correctness of the segmentation on an unlabeled
frame. More specifically, by measuring the perceptual consistency between the
predicted segmentation and the available ground truth on a nearby frame and
combining it with the segmentation confidence, we can accurately assess the
classification correctness on each pixel. Our experiments show that the
proposed perceptual consistency can more accurately evaluate the temporal
consistency of video segmentation as compared to flow-based measures.
Furthermore, it can help more confidently predict segmentation accuracy on
unlabeled test frames, as compared to using classification confidence alone.
Finally, our proposed measure can be used as a regularizer during the training
of segmentation models, which leads to more temporally consistent video
segmentation while maintaining accuracy.

    

### [[2110.12392] Variation is the Norm: Brain State Dynamics Evoked By Emotional Video Clips](http://arxiv.org/abs/2110.12392)


  For the last several decades, emotion research has attempted to identify a
"biomarker" or consistent pattern of brain activity to characterize a single
category of emotion (e.g., fear) that will remain consistent across all
instances of that category, regardless of individual and context. In this
study, we investigated variation rather than consistency during emotional
experiences while people watched video clips chosen to evoke instances of
specific emotion categories. Specifically, we developed a sequential
probabilistic approach to model the temporal dynamics in a participant's brain
activity during video viewing. We characterized brain states during these clips
as distinct state occupancy periods between state transitions in blood oxygen
level dependent (BOLD) signal patterns. We found substantial variation in the
state occupancy probability distributions across individuals watching the same
video, supporting the hypothesis that when it comes to the brain correlates of
emotional experience, variation may indeed be the norm.

    

### [[2110.12393] Deep Learning Approximation of Diffeomorphisms via Linear-Control Systems](http://arxiv.org/abs/2110.12393)


  In this paper we propose a Deep Learning architecture to approximate
diffeomorphisms isotopic to the identity. We consider a control system of the
form $\dot x = \sum_{i=1}^lF_i(x)u_i$, with linear dependence in the controls,
and we use the corresponding flow to approximate the action of a diffeomorphism
on a compact ensemble of points. Despite the simplicity of the control system,
it has been recently shown that a Universal Approximation Property holds. The
problem of minimizing the sum of the training error and of a regularizing term
induces a gradient flow in the space of admissible controls. A possible
training procedure for the discrete-time neural network consists in projecting
the gradient flow onto a finite-dimensional subspace of the admissible
controls. An alternative approach relies on an iterative method based on
Pontryagin Maximum Principle for the numerical resolution of Optimal Control
problems. Here the maximization of the Hamiltonian can be carried out with an
extremely low computational effort, owing to the linear dependence of the
system in the control variables.

    

### [[2110.12399] IQNAS: Interpretable Integer Quadratic Programming Neural Architecture Search](http://arxiv.org/abs/2110.12399)


  Realistic use of neural networks often requires adhering to multiple
constraints on latency, energy and memory among others. A popular approach to
find fitting networks is through constrained Neural Architecture Search (NAS).
However, previous methods use complicated predictors for the accuracy of the
network. Those predictors are hard to interpret and sensitive to many
hyperparameters to be tuned, hence, the resulting accuracy of the generated
models is often harmed. In this work we resolve this by introducing
Interpretable Integer Quadratic programming Neural Architecture Search (IQNAS),
that is based on an accurate and simple quadratic formulation of both the
accuracy predictor and the expected resource requirement, together with a
scalable search method with theoretical guarantees. The simplicity of our
proposed predictor together with the intuitive way it is constructed bring
interpretability through many insights about the contribution of different
design choices. For example, we find that in the examined search space, adding
depth and width is more effective at deeper stages of the network and at the
beginning of each resolution stage. Our experiments show that IQNAS generates
comparable to or better architectures than other state-of-the-art NAS methods
within a reduced search cost for each additional generated network, while
strictly satisfying the resource constraints.

    

### [[2110.12403] Learning to Estimate Without Bias](http://arxiv.org/abs/2110.12403)


  We consider the use of deep learning for parameter estimation. We propose
Bias Constrained Estimators (BCE) that add a squared bias term to the standard
mean squared error (MSE) loss. The main motivation to BCE is learning to
estimate deterministic unknown parameters with no Bayesian prior. Unlike
standard learning based estimators that are optimal on average, we prove that
BCEs converge to Minimum Variance Unbiased Estimators (MVUEs). We derive closed
form solutions to linear BCEs. These provide a flexible bridge between linear
regrssion and the least squares method. In non-linear settings, we demonstrate
that BCEs perform similarly to MVUEs even when the latter are computationally
intractable. A second motivation to BCE is in applications where multiple
estimates of the same unknown are averaged for improved performance. Examples
include distributed sensor networks and data augmentation in test-time. In such
applications, unbiasedness is a necessary condition for asymptotic consistency.

    

### [[2110.12412] Improved Goal Oriented Dialogue via Utterance Generation and Look Ahead](http://arxiv.org/abs/2110.12412)


  Goal oriented dialogue systems have become a prominent customer-care
interaction channel for most businesses. However, not all interactions are
smooth, and customer intent misunderstanding is a major cause of dialogue
failure. We show that intent prediction can be improved by training a deep
text-to-text neural model to generate successive user utterances from unlabeled
dialogue data. For that, we define a multi-task training regime that utilizes
successive user-utterance generation to improve the intent prediction. Our
approach achieves the reported improvement due to two complementary factors:
First, it uses a large amount of unlabeled dialogue data for an auxiliary
generation task. Second, it uses the generated user utterance as an additional
signal for the intent prediction model. Lastly, we present a novel look-ahead
approach that uses user utterance generation to improve intent prediction in
inference time. Specifically, we generate counterfactual successive user
utterances for conversations with ambiguous predicted intents, and disambiguate
the prediction by reassessing the concatenated sequence of available and
generated utterances.

    

### [[2110.12415] A Distributed Deep Reinforcement Learning Technique for Application Placement in Edge and Fog Computing Environments](http://arxiv.org/abs/2110.12415)


  Fog/Edge computing is a novel computing paradigm supporting
resource-constrained Internet of Things (IoT) devices by the placement of their
tasks on the edge and/or cloud servers. Recently, several Deep Reinforcement
Learning (DRL)-based placement techniques have been proposed in fog/edge
computing environments, which are only suitable for centralized setups. The
training of well-performed DRL agents requires manifold training data while
obtaining training data is costly. Hence, these centralized DRL-based
techniques lack generalizability and quick adaptability, thus failing to
efficiently tackle application placement problems. Moreover, many IoT
applications are modeled as Directed Acyclic Graphs (DAGs) with diverse
topologies. Satisfying dependencies of DAG-based IoT applications incur
additional constraints and increase the complexity of placement problems. To
overcome these challenges, we propose an actor-critic-based distributed
application placement technique, working based on the IMPortance weighted
Actor-Learner Architectures (IMPALA). IMPALA is known for efficient distributed
experience trajectory generation that significantly reduces the exploration
costs of agents. Besides, it uses an adaptive off-policy correction method for
faster convergence to optimal solutions. Our technique uses recurrent layers to
capture temporal behaviors of input data and a replay buffer to improve the
sample efficiency. The performance results, obtained from simulation and
testbed experiments, demonstrate that our technique significantly improves the
execution cost of IoT applications up to 30\% compared to its counterparts.

    

### [[2110.12425] Kernelized Heterogeneous Risk Minimization](http://arxiv.org/abs/2110.12425)


  The ability to generalize under distributional shifts is essential to
reliable machine learning, while models optimized with empirical risk
minimization usually fail on non-$i.i.d$ testing data. Recently, invariant
learning methods for out-of-distribution (OOD) generalization propose to find
causally invariant relationships with multi-environments. However, modern
datasets are frequently multi-sourced without explicit source labels, rendering
many invariant learning methods inapplicable. In this paper, we propose
Kernelized Heterogeneous Risk Minimization (KerHRM) algorithm, which achieves
both the latent heterogeneity exploration and invariant learning in kernel
space, and then gives feedback to the original neural network by appointing
invariant gradient direction. We theoretically justify our algorithm and
empirically validate the effectiveness of our algorithm with extensive
experiments.

    

### [[1807.09647] Variational Bayesian Reinforcement Learning with Regret Bounds](http://arxiv.org/abs/1807.09647)


  In reinforcement learning the Q-values summarize the expected future rewards
that the agent will attain. However, they cannot capture the epistemic
uncertainty about those rewards. In this work we derive a new Bellman operator
with associated fixed point we call the `knowledge values'. These K-values
compress both the expected future rewards and the epistemic uncertainty into a
single value, so that high uncertainty, high reward, or both, can yield high
K-values. The key principle is to endow the agent with a risk-seeking utility
function that is carefully tuned to balance exploration and exploitation. When
the agent follows a Boltzmann policy over the K-values it yields a Bayes regret
bound of $\tilde O(L^{3/2} \sqrt{S A T})$, where $L$ is the time horizon, $S$
is the number of states, $A$ is the number of actions, and $T$ is the total
number of elapsed timesteps. We show deep connections of this approach to the
soft-max and maximum-entropy strands of research in reinforcement learning.

    

### [[1903.11688] Rallying Adversarial Techniques against Deep Learning for Network Security](http://arxiv.org/abs/1903.11688)


  Recent advances in artificial intelligence and the increasing need for
powerful defensive measures in the domain of network security, have led to the
adoption of deep learning approaches for use in network intrusion detection
systems. These methods have achieved superior performance against conventional
network attacks, which enable the deployment of practical security systems to
unique and dynamic sectors. Adversarial machine learning, unfortunately, has
recently shown that deep learning models are inherently vulnerable to
adversarial modifications on their input data. Because of this susceptibility,
the deep learning models deployed to power a network defense could in fact be
the weakest entry point for compromising a network system. In this paper, we
show that by modifying on average as little as 1.38 of the input features, an
adversary can generate malicious inputs which effectively fool a deep learning
based NIDS. Therefore, when designing such systems, it is crucial to consider
the performance from not only the conventional network security perspective but
also the adversarial machine learning domain.

    

### [[1906.10509] Zero-Shot Image Classification Using Coupled Dictionary Embedding](http://arxiv.org/abs/1906.10509)


  Zero-shot learning (ZSL) is a framework to classify images belonging to
unseen classes based on solely semantic information about these unseen classes.
In this paper, we propose a new ZSL algorithm using coupled dictionary
learning. The core idea is that the visual features and the semantic attributes
of an image can share the same sparse representation in an intermediate space.
We use images from seen classes and semantic attributes from seen and unseen
classes to learn two dictionaries that can represent sparsely the visual and
semantic feature vectors of an image. In the ZSL testing stage and in the
absence of labeled data, images from unseen classes can be mapped into the
attribute space by finding the joint sparse representation using solely the
visual data. The image is then classified in the attribute space given semantic
descriptions of unseen classes. We also provide an attribute-aware formulation
to tackle domain shift and hubness problems in ZSL. Extensive experiments are
provided to demonstrate the superior performance of our approach against the
state of the art ZSL algorithms on benchmark ZSL datasets.

    

### [[1906.12304] Statistical Learning from Biased Training Samples](http://arxiv.org/abs/1906.12304)


  With the deluge of digitized information in the Big Data era, massive
datasets are becoming increasingly available for learning predictive models.
However, in many practical situations, the poor control of the data acquisition
processes may naturally jeopardize the outputs of machine learning algorithms,
and selection bias issues are now the subject of much attention in the
literature. The present article investigates how to extend Empirical Risk
Minimization, the principal paradigm in statistical learning, when training
observations are generated from biased models, i.e., from distributions that
are different from that in the test/prediction stage, and absolutely continuous
with respect to the latter. Precisely, we show how to build a "nearly debiased"
training statistical population from biased samples and the related biasing
functions, following in the footsteps of the approach originally proposed in
Vardi (1985). Furthermore, we study from a nonasymptotic perspective the
performance of minimizers of an empirical version of the risk computed from the
statistical population thus created. Remarkably, the learning rate achieved by
this procedure is of the same order as that attained in absence of selection
bias. Beyond the theoretical guarantees, we also present experimental results
supporting the relevance of the algorithmic approach promoted in this paper.

    

### [[1909.05244] Automatic Debiased Machine Learning for Instrumental Variable Models of Complier Treatment Effects](http://arxiv.org/abs/1909.05244)


  We propose debiased machine learning estimators for complier parameters, such
as local average treatment effect, with high dimensional covariates. To do so,
we characterize the doubly robust moment function for the entire class of
complier parameters as the combination of Wald and $\kappa$ weight
formulations. We directly estimate the $\kappa$ weights, rather than their
components, in order to eliminate the numerically unstable step of inverting
propensity scores of high dimensional covariates. We prove our estimator is
balanced, consistent, asymptotically normal, and semiparametrically efficient,
and use it to estimate the effect of 401(k) participation on the distribution
of net financial assets.

    

### [[1911.12641] PhIT-Net: Photo-consistent Image Transform for Robust Illumination Invariant Matching](http://arxiv.org/abs/1911.12641)


  We propose a new and completely data-driven approach for generating a
photo-consistent image transform. We show that simple classical algorithms
which operate in the transform domain become extremely resilient to
illumination changes. This considerably improves matching accuracy,
outperforming the use of state-of-the-art invariant representations as well as
new matching methods based on deep features. The transform is obtained by
training a neural network with a specialized triplet loss, designed to
emphasize actual scene changes while attenuating illumination changes. The
transform yields an illumination invariant representation, structured as an
image map, which is highly flexible and can be easily used for various tasks.

    

### [[2001.07697] Stochastic Approximation versus Sample Average Approximation for population Wasserstein barycenters](http://arxiv.org/abs/2001.07697)


  In the machine learning and optimization community, there are two main
approaches for the convex risk minimization problem, namely, the Stochastic
Approximation (SA) and the Sample Average Approximation (SAA). In terms of
oracle complexity (required number of stochastic gradient evaluations), both
approaches are considered equivalent on average (up to a logarithmic factor).
The total complexity depends on the specific problem, however, starting from
work \cite{nemirovski2009robust} it was generally accepted that the SA is
better than the SAA. % Nevertheless, in case of large-scale problems SA may run
out of memory as storing all data on one machine and organizing online access
to it can be impossible without communications with other machines. SAA in
contradistinction to SA allows parallel/distributed calculations. We show that
for the Wasserstein barycenter problem this superiority can be inverted. We
provide a detailed comparison by stating the complexity bounds for the SA and
the SAA implementations calculating barycenters defined with respect to optimal
transport distances and entropy-regularized optimal transport distances. As a
byproduct, we also construct confidence intervals for the barycenter defined
with respect to entropy-regularized optimal transport distances in the
$\ell_2$-norm. The preliminary results are derived for a general convex
optimization problem given by the expectation in order to have other
applications besides the Wasserstein barycenter problem.

    

### [[2001.11845] Learn to Predict Sets Using Feed-Forward Neural Networks](http://arxiv.org/abs/2001.11845)


  This paper addresses the task of set prediction using deep feed-forward
neural networks. A set is a collection of elements which is invariant under
permutation and the size of a set is not fixed in advance. Many real-world
problems, such as image tagging and object detection, have outputs that are
naturally expressed as sets of entities. This creates a challenge for
traditional deep neural networks which naturally deal with structured outputs
such as vectors, matrices or tensors. We present a novel approach for learning
to predict sets with unknown permutation and cardinality using deep neural
networks. In our formulation we define a likelihood for a set distribution
represented by a) two discrete distributions defining the set cardinally and
permutation variables, and b) a joint distribution over set elements with a
fixed cardinality. Depending on the problem under consideration, we define
different training models for set prediction using deep neural networks. We
demonstrate the validity of our set formulations on relevant vision problems
such as: 1) multi-label image classification where we outperform the other
competing methods on the PASCAL VOC and MS COCO datasets, 2) object detection,
for which our formulation outperforms popular state-of-the-art detectors, and
3) a complex CAPTCHA test, where we observe that, surprisingly, our set-based
network acquired the ability of mimicking arithmetics without any rules being
coded.

    

### [[2002.05038] Robustness analytics to data heterogeneity in edge computing](http://arxiv.org/abs/2002.05038)


  Federated Learning is a framework that jointly trains a model \textit{with}
complete knowledge on a remotely placed centralized server, but
\textit{without} the requirement of accessing the data stored in distributed
machines. Some work assumes that the data generated from edge devices are
identically and independently sampled from a common population distribution.
However, such ideal sampling may not be realistic in many contexts. Also,
models based on intrinsic agency, such as active sampling schemes, may lead to
highly biased sampling. So an imminent question is how robust Federated
Learning is to biased sampling? In this
work\footnote{\url{this https URL}}, we
experimentally investigate two such scenarios. First, we study a centralized
classifier aggregated from a collection of local classifiers trained with data
having categorical heterogeneity. Second, we study a classifier aggregated from
a collection of local classifiers trained by data through active sampling at
the edge. We present evidence in both scenarios that Federated Learning is
robust to data heterogeneity when local training iterations and communication
frequency are appropriately chosen.

    

### [[2004.11262] Supervised Domain Adaptation: A Graph Embedding Perspective and a Rectified Experimental Protocol](http://arxiv.org/abs/2004.11262)


  Domain Adaptation is the process of alleviating distribution gaps between
data from different domains. In this paper, we show that Domain Adaptation
methods using pair-wise relationships between source and target domain data can
be formulated as a Graph Embedding in which the domain labels are incorporated
into the structure of the intrinsic and penalty graphs. Specifically, we
analyse the loss functions of three existing state-of-the-art Supervised Domain
Adaptation methods and demonstrate that they perform Graph Embedding. Moreover,
we highlight some generalisation and reproducibility issues related to the
experimental setup commonly used to demonstrate the few-shot learning
capabilities of these methods. To assess and compare Supervised Domain
Adaptation methods accurately, we propose a rectified evaluation protocol, and
report updated benchmarks on the standard datasets Office31 (Amazon, DSLR, and
Webcam), Digits (MNIST, USPS, SVHN, and MNIST-M) and VisDA (Synthetic, Real).

    

### [[2004.11627] Convolution-Weight-Distribution Assumption: Rethinking the Criteria of Channel Pruning](http://arxiv.org/abs/2004.11627)


  Channel pruning is a popular technique for compressing convolutional neural
networks (CNNs), where various pruning criteria have been proposed to remove
the redundant filters. From our comprehensive experiments, we found two blind
spots in the study of pruning criteria: (1) Similarity: There are some strong
similarities among several primary pruning criteria that are widely cited and
compared. According to these criteria, the ranks of filters'Importance Score
are almost identical, resulting in similar pruned structures. (2)
Applicability: The filters'Importance Score measured by some pruning criteria
are too close to distinguish the network redundancy well. In this paper, we
analyze these two blind spots on different types of pruning criteria with
layer-wise pruning or global pruning. The analyses are based on the empirical
experiments and our assumption (Convolutional Weight Distribution Assumption)
that the well-trained convolutional filters each layer approximately follow a
Gaussian-alike distribution. This assumption has been verified through
systematic and extensive statistical tests.

    

### [[2004.13912] Neural Additive Models: Interpretable Machine Learning with Neural Nets](http://arxiv.org/abs/2004.13912)


  Deep neural networks (DNNs) are powerful black-box predictors that have
achieved impressive performance on a wide variety of tasks. However, their
accuracy comes at the cost of intelligibility: it is usually unclear how they
make their decisions. This hinders their applicability to high stakes
decision-making domains such as healthcare. We propose Neural Additive Models
(NAMs) which combine some of the expressivity of DNNs with the inherent
intelligibility of generalized additive models. NAMs learn a linear combination
of neural networks that each attend to a single input feature. These networks
are trained jointly and can learn arbitrarily complex relationships between
their input feature and the output. Our experiments on regression and
classification datasets show that NAMs are more accurate than widely used
intelligible models such as logistic regression and shallow decision trees.
They perform similarly to existing state-of-the-art generalized additive models
in accuracy, but are more flexible because they are based on neural nets
instead of boosted trees. To demonstrate this, we show how NAMs can be used for
multitask learning on synthetic data and on the COMPAS recidivism data due to
their composability, and demonstrate that the differentiability of NAMs allows
them to train more complex interpretable models for COVID-19.

    

### [[2005.14117] Multimodal Feature Fusion and Knowledge-Driven Learning via Experts Consult for Thyroid Nodule Classification](http://arxiv.org/abs/2005.14117)


  Computer-aided diagnosis (CAD) is becoming a prominent approach to assist
clinicians spanning across multiple fields. These automated systems take
advantage of various computer vision (CV) procedures, as well as artificial
intelligence (AI) techniques, to formulate a diagnosis of a given image, e.g.,
computed tomography and ultrasound. Advances in both areas (CV and AI) are
enabling ever increasing performances of CAD systems, which can ultimately
avoid performing invasive procedures such as fine-needle aspiration. In this
study, a novel end-to-end knowledge-driven classification framework is
presented. The system focuses on multimodal data generated by thyroid
ultrasonography, and acts as a CAD system by providing a thyroid nodule
classification into the benign and malignant categories. Specifically, the
proposed system leverages cues provided by an ensemble of experts to guide the
learning phase of a densely connected convolutional network (DenseNet). The
ensemble is composed by various networks pretrained on ImageNet, including
AlexNet, ResNet, VGG, and others. The previously computed multimodal feature
parameters are used to create ultrasonography domain experts via transfer
learning, decreasing, moreover, the number of samples required for training. To
validate the proposed method, extensive experiments were performed, providing
detailed performances for both the experts ensemble and the knowledge-driven
DenseNet. As demonstrated by the results, the proposed system achieves relevant
performances in terms of qualitative metrics for the thyroid nodule
classification task, thus resulting in a great asset when formulating a
diagnosis.

    

### [[2006.07322] Evaluation of Neural Architectures Trained with Square Loss vs Cross-Entropy in Classification Tasks](http://arxiv.org/abs/2006.07322)


  Modern neural architectures for classification tasks are trained using the
cross-entropy loss, which is widely believed to be empirically superior to the
square loss. In this work we provide evidence indicating that this belief may
not be well-founded. We explore several major neural architectures and a range
of standard benchmark datasets for NLP, automatic speech recognition (ASR) and
computer vision tasks to show that these architectures, with the same
hyper-parameter settings as reported in the literature, perform comparably or
better when trained with the square loss, even after equalizing computational
resources. Indeed, we observe that the square loss produces better results in
the dominant majority of NLP and ASR experiments. Cross-entropy appears to have
a slight edge on computer vision tasks.
We argue that there is little compelling empirical or theoretical evidence
indicating a clear-cut advantage to the cross-entropy loss. Indeed, in our
experiments, performance on nearly all non-vision tasks can be improved,
sometimes significantly, by switching to the square loss. Furthermore, training
with square loss appears to be less sensitive to the randomness in
initialization. We posit that training using the square loss for classification
needs to be a part of best practices of modern deep learning on equal footing
with cross-entropy.

    

### [[2007.10976] Interactive Inference under Information Constraints](http://arxiv.org/abs/2007.10976)


  We study the role of interactivity in distributed statistical inference under
information constraints, e.g., communication constraints and local differential
privacy. We focus on the tasks of goodness-of-fit testing and estimation of
discrete distributions. From prior work, these tasks are well understood under
noninteractive protocols. Extending these approaches directly for interactive
protocols is difficult due to correlations that can build due to interactivity;
in fact, gaps can be found in prior claims of tight bounds of distribution
estimation using interactive protocols.
We propose a new approach to handle this correlation and establish a unified
method to establish lower bounds for both tasks. As an application, we obtain
optimal bounds for both estimation and testing under local differential privacy
and communication constraints. We also provide an example of a natural testing
problem where interactivity helps.

    

### [[2009.03892] Neural-PDE: A RNN based neural network for solving time dependent PDEs](http://arxiv.org/abs/2009.03892)


  Partial differential equations (PDEs) play a crucial role in studying a vast
number of problems in science and engineering. Numerically solving nonlinear
and/or high-dimensional PDEs is often a challenging task. Inspired by the
traditional finite difference and finite elements methods and emerging
advancements in machine learning, we propose a sequence deep learning framework
called Neural-PDE, which allows to automatically learn governing rules of any
time-dependent PDE system from existing data by using a bidirectional LSTM
encoder, and predict the next n time steps data. One critical feature of our
proposed framework is that the Neural-PDE is able to simultaneously learn and
simulate the multiscale variables.We test the Neural-PDE by a range of examples
from one-dimensional PDEs to a high-dimensional and nonlinear complex fluids
model. The results show that the Neural-PDE is capable of learning the initial
conditions, boundary conditions and differential operators without the
knowledge of the specific form of a PDE this http URL our experiments the
Neural-PDE can efficiently extract the dynamics within 20 epochs training, and
produces accurate predictions. Furthermore, unlike the traditional machine
learning approaches in learning PDE such as CNN and MLP which require vast
parameters for model precision, Neural-PDE shares parameters across all time
steps, thus considerably reduces the computational complexity and leads to a
fast learning algorithm.

    

### [[2010.00163] Bayesian Meta-reinforcement Learning for Traffic Signal Control](http://arxiv.org/abs/2010.00163)


  In recent years, there has been increasing amount of interest around meta
reinforcement learning methods for traffic signal control, which have achieved
better performance compared with traditional control methods. However, previous
methods lack robustness in adaptation and stability in training process in
complex situations, which largely limits its application in real-world traffic
signal control. In this paper, we propose a novel value-based Bayesian
meta-reinforcement learning framework BM-DQN to robustly speed up the learning
process in new scenarios by utilizing well-trained prior knowledge learned from
existing scenarios. This framework is based on our proposed fast-adaptation
variation to Gradient-EM Bayesian Meta-learning and the fast-update advantage
of DQN, which allows for fast adaptation to new scenarios with continual
learning ability and robustness to uncertainty. The experiments on restricted
2D navigation and traffic signal control show that our proposed framework
adapts more quickly and robustly in new scenarios than previous methods, and
specifically, much better continual learning ability in heterogeneous
scenarios.

    

### [[2010.02002] Enhancing Haptic Distinguishability of Surface Materials with Boosting Technique](http://arxiv.org/abs/2010.02002)


  Discriminative features are crucial for various learning applications such as
object detection and classification. Neural network models have shown enormous
potential in extracting discriminative features in the vision and speech
domain. However, the lack of large datasets in the haptics domain often limits
the applicability of such techniques. This paper presents a general framework
for the analysis of the discriminative properties of haptic signals. We
demonstrate the effectiveness of metric-based feature transformation techniques
in enhancing the distinguishability of haptic signals. Experiments indicate our
framework needs less training data, generalizes well for different predictors,
and outperforms the related state-of-the-art.

    

### [[2010.02610] Robust priors for regularized regression](http://arxiv.org/abs/2010.02610)


  Induction benefits from useful priors. Penalized regression approaches, like
ridge regression, shrink weights toward zero but zero association is usually
not a sensible prior. Inspired by simple and robust decision heuristics humans
use, we constructed non-zero priors for penalized regression models that
provide robust and interpretable solutions across several tasks. Our approach
enables estimates from a constrained model to serve as a prior for a more
general model, yielding a principled way to interpolate between models of
differing complexity. We successfully applied this approach to a number of
decision and classification problems, as well as analyzing simulated brain
imaging data. Models with robust priors had excellent worst-case performance.
Solutions followed from the form of the heuristic that was used to derive the
prior. These new algorithms can serve applications in data analysis and machine
learning, as well as help in understanding how people transition from novice to
expert performance.

    

### [[2010.07853] Selective Classification via One-Sided Prediction](http://arxiv.org/abs/2010.07853)


  We propose a novel method for selective classification (SC), a problem which
allows a classifier to abstain from predicting some instances, thus trading off
accuracy against coverage (the fraction of instances predicted). In contrast to
prior gating or confidence-set based work, our proposed method optimises a
collection of class-wise decoupled one-sided empirical risks, and is in essence
a method for explicitly finding the largest decision sets for each class that
have few false positives. This one-sided prediction (OSP) based relaxation
yields an SC scheme that attains near-optimal coverage in the practically
relevant high target accuracy regime, and further admits efficient
implementation, leading to a flexible and principled method for SC. We
theoretically derive generalization bounds for SC and OSP, and empirically we
show that our scheme strongly outperforms state of the art methods in coverage
at small error levels.

    

### [[2010.10670] Iterative Amortized Policy Optimization](http://arxiv.org/abs/2010.10670)


  Policy networks are a central feature of deep reinforcement learning (RL)
algorithms for continuous control, enabling the estimation and sampling of
high-value actions. From the variational inference perspective on RL, policy
networks, when used with entropy or KL regularization, are a form of
\textit{amortized optimization}, optimizing network parameters rather than the
policy distributions directly. However, \textit{direct} amortized mappings can
yield suboptimal policy estimates and restricted distributions, limiting
performance and exploration. Given this perspective, we consider the more
flexible class of \textit{iterative} amortized optimizers. We demonstrate that
the resulting technique, iterative amortized policy optimization, yields
performance improvements over direct amortization on benchmark continuous
control tasks.

    

### [[2010.10682] VenoMave: Targeted Poisoning Against Speech Recognition](http://arxiv.org/abs/2010.10682)


  The wide adoption of Automatic Speech Recognition (ASR) remarkably enhanced
human-machine interaction. Prior research has demonstrated that modern ASR
systems are susceptible to adversarial examples, i.e., malicious audio inputs
that lead to misclassification by the victim's model at run time. The research
question of whether ASR systems are also vulnerable to data-poisoning attacks
is still unanswered. In such an attack, a manipulation happens during the
training phase: an adversary injects malicious inputs into the training set to
compromise the neural network's integrity and performance. Prior work in the
image domain demonstrated several types of data-poisoning attacks, but these
results cannot directly be applied to the audio domain. In this paper, we
present the first data-poisoning attack against ASR, called VenoMave. We
evaluate our attack on an ASR system that detects sequences of digits. When
poisoning only 0.17% of the dataset on average, we achieve an attack success
rate of 86.67%. To demonstrate the practical feasibility of our attack, we also
evaluate if the target audio waveform can be played over the air via simulated
room transmissions. In this more realistic threat model, VenoMave still
maintains a success rate up to 73.33%. We further extend our evaluation to the
Speech Commands corpus and demonstrate the scalability of VenoMave to a larger
vocabulary. During a transcription test with human listeners, we verify that
more than 85% of the original text of poisons can be correctly transcribed. We
conclude that data-poisoning attacks against ASR represent a real threat, and
we are able to perform poisoning for arbitrary target input files while the
crafted poison samples remain inconspicuous.

    

### [[2010.13723] Optimal Client Sampling for Federated Learning](http://arxiv.org/abs/2010.13723)


  It is well understood that client-master communication can be a primary
bottleneck in Federated Learning. In this work, we address this issue with a
novel client subsampling scheme, where we restrict the number of clients
allowed to communicate their updates back to the master node. In each
communication round, all participating clients compute their updates, but only
the ones with "important" updates communicate back to the master. We show that
importance can be measured using only the norm of the update and give a formula
for optimal client participation. This formula minimizes the distance between
the full update, where all clients participate, and our limited update, where
the number of participating clients is restricted. In addition, we provide a
simple algorithm that approximates the optimal formula for client
participation, which only requires secure aggregation and thus does not
compromise client privacy. We show both theoretically and empirically that for
Distributed SGD (DSGD) and Federated Averaging (FedAvg), the performance of our
approach can be close to full participation and superior to the baseline where
participating clients are sampled uniformly. Moreover, our approach is
orthogonal to and compatible with existing methods for reducing communication
overhead, such as local methods and communication compression methods.

    

### [[2010.14498] Implicit Under-Parameterization Inhibits Data-Efficient Deep Reinforcement Learning](http://arxiv.org/abs/2010.14498)


  We identify an implicit under-parameterization phenomenon in value-based deep
RL methods that use bootstrapping: when value functions, approximated using
deep neural networks, are trained with gradient descent using iterated
regression onto target values generated by previous instances of the value
network, more gradient updates decrease the expressivity of the current value
network. We characterize this loss of expressivity via a drop in the rank of
the learned value network features, and show that this typically corresponds to
a performance drop. We demonstrate this phenomenon on Atari and Gym benchmarks,
in both offline and online RL settings. We formally analyze this phenomenon and
show that it results from a pathological interaction between bootstrapping and
gradient-based optimization. We further show that mitigating implicit
under-parameterization by controlling rank collapse can improve performance.

    

### [[2010.16212] Efficient constrained sampling via the mirror-Langevin algorithm](http://arxiv.org/abs/2010.16212)


  We propose a new discretization of the mirror-Langevin diffusion and give a
crisp proof of its convergence. Our analysis uses relative convexity/smoothness
and self-concordance, ideas which originated in convex optimization, together
with a new result in optimal transport that generalizes the displacement
convexity of the entropy. Unlike prior works, our result both (1) requires much
weaker assumptions on the mirror map and the target distribution, and (2) has
vanishing bias as the step size tends to zero. In particular, for the task of
sampling from a log-concave distribution supported on a compact set, our
theoretical results are significantly better than the existing guarantees.

    

### [[2011.02803] Intriguing Properties of Contrastive Losses](http://arxiv.org/abs/2011.02803)


  We study three intriguing properties of contrastive learning. First, we
generalize the standard contrastive loss to a broader family of losses, and we
find that various instantiations of the generalized loss perform similarly
under the presence of a multi-layer non-linear projection head. Second, we
study if instance-based contrastive learning (with a global image
representation) can learn well on images with multiple objects present. We find
that meaningful hierarchical local features can be learned despite the fact
that these objectives operate on global instance-level features. Finally, we
study the phenomenon of feature suppression among competing features shared
across augmented views, such as "color distribution" vs "object class". We
construct datasets with explicit and controllable competing features, and show
that, for contrastive learning, a few bits of easy-to-learn shared features can
suppress, and even fully prevent, the learning of other sets of competing
features. In scenarios where there are multiple objects in an image, the
dominant object would suppress the learning of smaller objects. Existing
contrastive learning methods critically rely on data augmentation to favor
certain sets of features over others, and could suffer from learning saturation
for scenarios where existing augmentations cannot fully address the feature
suppression. This poses open challenges to existing contrastive learning
techniques.

    

### [[2011.10530] On barren plateaus and cost function locality in variational quantum algorithms](http://arxiv.org/abs/2011.10530)


  Variational quantum algorithms rely on gradient based optimization to
iteratively minimize a cost function evaluated by measuring output(s) of a
quantum processor. A barren plateau is the phenomenon of exponentially
vanishing gradients in sufficiently expressive parametrized quantum circuits.
It has been established that the onset of a barren plateau regime depends on
the cost function, although the particular behavior has been demonstrated only
for certain classes of cost functions. Here we derive a lower bound on the
variance of the gradient, which depends mainly on the width of the circuit
causal cone of each term in the Pauli decomposition of the cost function. Our
result further clarifies the conditions under which barren plateaus can occur.

    

### [[2011.10577] Deep learning insights into cosmological structure formation](http://arxiv.org/abs/2011.10577)


  While the evolution of linear initial conditions present in the early
universe into extended halos of dark matter at late times can be computed using
cosmological simulations, a theoretical understanding of this complex process
remains elusive. Here, we build a deep learning framework to learn this
non-linear relationship, and develop techniques to physically interpret the
learnt mapping. A three-dimensional convolutional neural network (CNN) is
trained to predict the mass of dark matter halos from the initial conditions.
N-body simulations follow the microphysical laws of gravity, whereas the CNN
model provides a simplified description of halo collapse where features are
extracted from the initial conditions through convolutions and combined in a
non-linear way to provide a halo mass prediction. We find no significant change
in the predictive accuracy of the model if we retrain it removing anisotropic
information from the inputs, suggesting that the features learnt by the CNN are
equivalent to spherical averages over the initial conditions. Despite including
all possible feature combinations that can be extracted by convolutions in the
model, the final halo mass predictions do not depend on anisotropic aspects of
the initial conditions. Our results indicate that deep learning frameworks can
provide a powerful tool for extracting physical insight into cosmological
structure formation.

    

### [[2012.02178] Steady-State Planning in Expected Reward Multichain MDPs](http://arxiv.org/abs/2012.02178)


  The planning domain has experienced increased interest in the formal
synthesis of decision-making policies. This formal synthesis typically entails
finding a policy which satisfies formal specifications in the form of some
well-defined logic. While many such logics have been proposed with varying
degrees of expressiveness and complexity in their capacity to capture desirable
agent behavior, their value is limited when deriving decision-making policies
which satisfy certain types of asymptotic behavior in general system models. In
particular, we are interested in specifying constraints on the steady-state
behavior of an agent, which captures the proportion of time an agent spends in
each state as it interacts for an indefinite period of time with its
environment. This is sometimes called the average or expected behavior of the
agent and the associated planning problem is faced with significant challenges
unless strong restrictions are imposed on the underlying model in terms of the
connectivity of its graph structure. In this paper, we explore this
steady-state planning problem that consists of deriving a decision-making
policy for an agent such that constraints on its steady-state behavior are
satisfied. A linear programming solution for the general case of multichain
Markov Decision Processes (MDPs) is proposed and we prove that optimal
solutions to the proposed programs yield stationary policies with rigorous
guarantees of behavior.

    

### [[2012.04061] Faster Non-Convex Federated Learning via Global and Local Momentum](http://arxiv.org/abs/2012.04061)


  We propose \texttt{FedGLOMO}, a novel federated learning (FL) algorithm with
an iteration complexity of $\mathcal{O}(\epsilon^{-1.5})$ to converge to an
$\epsilon$-stationary point (i.e., $\mathbb{E}[\|\nabla f(\bm{x})\|^2] \leq
\epsilon$) for smooth non-convex functions -- under arbitrary client
heterogeneity and compressed communication -- compared to the
$\mathcal{O}(\epsilon^{-2})$ complexity of most prior works. Our key
algorithmic idea that enables achieving this improved complexity is based on
the observation that the convergence in FL is hampered by two sources of high
variance: (i) the global server aggregation step with multiple local updates,
exacerbated by client heterogeneity, and (ii) the noise of the local
client-level stochastic gradients. By modeling the server aggregation step as a
generalized gradient-type update, we propose a variance-reducing momentum-based
global update at the server, which when applied in conjunction with
variance-reduced local updates at the clients, enables \texttt{FedGLOMO} to
enjoy an improved convergence rate. Moreover, we derive our results under a
novel and more realistic client-heterogeneity assumption which we verify
empirically -- unlike prior assumptions that are hard to verify. Our
experiments illustrate the intrinsic variance reduction effect of
\texttt{FedGLOMO}, which implicitly suppresses client-drift in heterogeneous
data distribution settings and promotes communication efficiency.

    

### [[2012.14100] Exploiting Chain Rule and Bayes' Theorem to Compare Probability Distributions](http://arxiv.org/abs/2012.14100)


  To measure the difference between two probability distributions, referred to
as the source and target, respectively, we exploit both the chain rule and
Bayes' theorem to construct conditional transport (CT), which is constituted by
both a forward component and a backward one. The forward CT is the expected
cost of moving a source data point to a target one, with their joint
distribution defined by the product of the source probability density function
(PDF) and a source-dependent conditional distribution, which is related to the
target PDF via Bayes' theorem. The backward CT is defined by reversing the
direction. The CT cost can be approximated by replacing the source and target
PDFs with their discrete empirical distributions supported on mini-batches,
making it amenable to implicit distributions and stochastic gradient
descent-based optimization. When applied to train a generative model, CT is
shown to strike a good balance between mode-covering and mode-seeking behaviors
and strongly resist mode collapse. On a wide variety of benchmark datasets for
generative modeling, substituting the default statistical distance of an
existing generative adversarial network with CT is shown to consistently
improve the performance. PyTorch code is provided.

    

### [[2101.12727] Surprisingly Simple Semi-Supervised Domain Adaptation with Pretraining and Consistency](http://arxiv.org/abs/2101.12727)


  Most modern unsupervised domain adaptation (UDA) approaches are rooted in
domain alignment, i.e., learning to align source and target features to learn a
target domain classifier using source labels. In semi-supervised domain
adaptation (SSDA), when the learner can access few target domain labels, prior
approaches have followed UDA theory to use domain alignment for learning. We
show that the case of SSDA is different and a good target classifier can be
learned without needing alignment. We use self-supervised pretraining (via
rotation prediction) and consistency regularization to achieve well separated
target clusters, aiding in learning a low error target classifier. With our
Pretraining and Consistency (PAC) approach, we achieve state of the art target
accuracy on this semi-supervised domain adaptation task, surpassing multiple
adversarial domain alignment methods, across multiple datasets. PAC, while
using simple techniques, performs remarkably well on large and challenging SSDA
benchmarks like DomainNet and Visda-17, often outperforming recent state of the
art by sizeable margins. Code for our experiments can be found at
this https URL


### [[2102.02694] Invertible DenseNets with Concatenated LipSwish](http://arxiv.org/abs/2102.02694)


  We introduce Invertible Dense Networks (i-DenseNets), a more parameter
efficient extension of Residual Flows. The method relies on an analysis of the
Lipschitz continuity of the concatenation in DenseNets, where we enforce
invertibility of the network by satisfying the Lipschitz constant. Furthermore,
we propose a learnable weighted concatenation, which not only improves the
model performance but also indicates the importance of the concatenated
weighted representation. Additionally, we introduce the Concatenated LipSwish
as activation function, for which we show how to enforce the Lipschitz
condition and which boosts performance. The new architecture, i-DenseNet,
out-performs Residual Flow and other flow-based models on density estimation
evaluated in bits per dimension, where we utilize an equal parameter budget.
Moreover, we show that the proposed model out-performs Residual Flows when
trained as a hybrid model where the model is both a generative and a
discriminative model.

    

### [[2102.03147] Learning Conjoint Attentions for Graph Neural Nets](http://arxiv.org/abs/2102.03147)


  In this paper, we present Conjoint Attentions (CAs), a class of novel
learning-to-attend strategies for graph neural networks (GNNs). Besides
considering the layer-wise node features propagated within the GNN, CAs can
additionally incorporate various structural interventions, such as node cluster
embedding, and higher-order structural correlations that can be learned outside
of GNN, when computing attention scores. The node features that are regarded as
significant by the conjoint criteria are therefore more likely to be propagated
in the GNN. Given the novel Conjoint Attention strategies, we then propose
Graph conjoint attention networks (CATs) that can learn representations
embedded with significant latent features deemed by the Conjoint Attentions.
Besides, we theoretically validate the discriminative capacity of CATs. CATs
utilizing the proposed Conjoint Attention strategies have been extensively
tested in well-established benchmarking datasets and comprehensively compared
with state-of-the-art baselines. The obtained notable performance demonstrates
the effectiveness of the proposed Conjoint Attentions.

    

### [[2102.06866] Understanding Negative Samples in Instance Discriminative Self-supervised Representation Learning](http://arxiv.org/abs/2102.06866)


  Instance discriminative self-supervised representation learning has been
attracted attention thanks to its unsupervised nature and informative feature
representation for downstream tasks. In practice, it commonly uses a larger
number of negative samples than the number of supervised classes. However,
there is an inconsistency in the existing analysis; theoretically, a large
number of negative samples degrade classification performance on a downstream
supervised task, while empirically, they improve the performance. We provide a
novel framework to analyze this empirical result regarding negative samples
using the coupon collector's problem. Our bound can implicitly incorporate the
supervised loss of the downstream task in the self-supervised loss by
increasing the number of negative samples. We confirm that our proposed
analysis holds on real-world benchmark datasets.

    

### [[2102.07521] Distributed Online Learning for Joint Regret with Communication Constraints](http://arxiv.org/abs/2102.07521)


  We consider distributed online learning for joint regret with communication
constraints. In this setting, there are multiple agents that are connected in a
graph. Each round, an adversary first activates one of the agents to issue a
prediction and provides a corresponding gradient, and then the agents are
allowed to send a $b$-bit message to their neighbors in the graph. All agents
cooperate to control the joint regret, which is the sum of the losses of the
activated agents minus the losses evaluated at the best fixed common comparator
parameters $u$. We observe that it is suboptimal for agents to wait for
gradients that take too long to arrive. Instead, the graph should be
partitioned into local clusters that communicate among themselves. Our main
result is a new method that can adapt to the optimal graph partition for the
adversarial activations and gradients, where the graph partition is selected
from a set of candidate partitions. A crucial building block along the way is a
new algorithm for online convex optimization with delayed gradient information
that is comparator-adaptive, meaning that its joint regret scales with the norm
of the comparator $||u||$. We further provide near-optimal gradient compression
schemes depending on the ratio of $b$ and the dimension times the diameter of
the graph.

    

### [[2102.08504] Label Leakage and Protection in Two-party Split Learning](http://arxiv.org/abs/2102.08504)


  Two-party split learning is a popular technique for learning a model across
feature-partitioned data. In this work, we explore whether it is possible for
one party to steal the private label information from the other party during
split training, and whether there are methods that can protect against such
attacks. Specifically, we first formulate a realistic threat model and propose
a privacy loss metric to quantify label leakage in split learning. We then show
that there exist two simple yet effective methods within the threat model that
can allow one party to accurately recover private ground-truth labels owned by
the other party. To combat these attacks, we propose several random
perturbation techniques, including $\texttt{Marvell}$, an approach that
strategically finds the structure of the noise perturbation by minimizing the
amount of label leakage (measured through our quantification metric) of a
worst-case adversary. We empirically demonstrate the effectiveness of our
protection techniques against the identified attacks, and show that
$\texttt{Marvell}$ in particular has improved privacy-utility tradeoffs
relative to baseline approaches.

    

### [[2102.10221] Logarithmic Regret in Feature-based Dynamic Pricing](http://arxiv.org/abs/2102.10221)


  Feature-based dynamic pricing is an increasingly popular model of setting
prices for highly differentiated products with applications in digital
marketing, online sales, real estate and so on. The problem was formally
studied as an online learning problem [Javanmard & Nazerzadeh, 2019] where a
seller needs to propose prices on the fly for a sequence of $T$ products based
on their features $x$ while having a small regret relative to the best --
"omniscient" -- pricing strategy she could have come up with in hindsight. We
revisit this problem and provide two algorithms (EMLP and ONSP) for stochastic
and adversarial feature settings, respectively, and prove the optimal
$O(d\log{T})$ regret bounds for both. In comparison, the best existing results
are $O\left(\min\left\{\frac{1}{\lambda_{\min}^2}\log{T},
\sqrt{T}\right\}\right)$ and $O(T^{2/3})$ respectively, with $\lambda_{\min}$
being the smallest eigenvalue of $\mathbb{E}[xx^T]$ that could be arbitrarily
close to $0$. We also prove an $\Omega(\sqrt{T})$ information-theoretic lower
bound for a slightly more general setting, which demonstrates that
"knowing-the-demand-curve" leads to an exponential improvement in feature-based
dynamic pricing.

    

### [[2102.11976] Learner-Private Convex Optimization](http://arxiv.org/abs/2102.11976)


  Convex optimization with feedback is a framework where a learner relies on
iterative queries and feedback to arrive at the minimizer of a convex function.
It has gained considerable popularity thanks to its scalability in large-scale
optimization and machine learning. The repeated interactions, however, expose
the learner to privacy risks from eavesdropping adversaries that observe the
submitted queries. In this paper, we study how to optimally obfuscate the
learner's queries in convex optimization with first-order feedback, so that
their learned optimal value is provably difficult to estimate for an
eavesdropping adversary. We consider two formulations of learner privacy: a
Bayesian formulation in which the convex function is drawn randomly, and a
minimax formulation in which the function is fixed and the adversary's
probability of error is measured with respect to a minimax criterion.
Suppose that the learner wishes to ensure the adversary cannot estimate
accurately with probability greater than $1/L$ for some $L>0$. Our main results
show that the query complexity overhead is additive in $L$ in the minimax
formulation, but multiplicative in $L$ in the Bayesian formulation. Compared to
existing learner-private sequential learning models with binary feedback, our
results apply to the significantly richer family of general convex functions
with full-gradient feedback. Our proofs learn on tools from the theory of
Dirichlet processes, as well as a novel strategy designed for measuring
information leakage under a full-gradient oracle.

    

### [[2103.00397] Data-Efficient GAN Training Beyond (Just) Augmentations: A Lottery Ticket Perspective](http://arxiv.org/abs/2103.00397)


  Training generative adversarial networks (GANs) with limited real image data
generally results in deteriorated performance and collapsed models. To conquer
this challenge, we are inspired by the latest observation, that one can
discover independently trainable and highly sparse subnetworks (a.k.a., lottery
tickets) from GANs. Treating this as an inductive prior, we suggest a brand-new
angle towards data-efficient GAN training: by first identifying the lottery
ticket from the original GAN using the small training set of real images; and
then focusing on training that sparse subnetwork by re-using the same set. We
find our coordinated framework to offer orthogonal gains to existing real image
data augmentation methods, and we additionally present a new feature-level
augmentation that can be applied together with them. Comprehensive experiments
endorse the effectiveness of our proposed framework, across various GAN
architectures (SNGAN, BigGAN, and StyleGAN-V2) and diverse datasets (CIFAR-10,
CIFAR-100, Tiny-ImageNet, ImageNet, and multiple few-shot generation datasets).
Codes are available at:
this https URL.

    

### [[2103.02898] Fast Tucker Rank Reduction for Non-Negative Tensors Using Mean-Field Approximation](http://arxiv.org/abs/2103.02898)


  We present an efficient low-rank approximation algorithm for non-negative
tensors. The algorithm is derived from our two findings: First, we show that
rank-1 approximation for tensors can be viewed as a mean-field approximation by
treating each tensor as a probability distribution. Second, we theoretically
provide a sufficient condition for distribution parameters to reduce Tucker
ranks of tensors; interestingly, this sufficient condition can be achieved by
iterative application of the mean-field approximation. Since the mean-field
approximation is always given as a closed formula, our findings lead to a fast
low-rank approximation algorithm without using a gradient method. We
empirically demonstrate that our algorithm is faster than the existing
non-negative Tucker rank reduction methods and achieves competitive or better
approximation of given tensors.

    

### [[2103.03568] Can Pretext-Based Self-Supervised Learning Be Boosted by Downstream Data? A Theoretical Analysis](http://arxiv.org/abs/2103.03568)


  Pretext-based self-supervised learning learns the semantic representation via
a handcrafted pretext task over unlabeled data and then uses the learned
representation for downstream tasks, which effectively reduces the sample
complexity of downstream tasks under Conditional Independence (CI) condition.
However, the downstream sample complexity gets much worse if the CI condition
does not hold. One interesting question is whether we can make the CI condition
hold by using downstream data to refine the unlabeled data to boost
self-supervised learning. At first glance, one might think that seeing
downstream data in advance would always boost the downstream performance.
However, we show that it is not intuitively true and point out that in some
cases, it hurts the final performance instead. In particular, we prove both
model-free and model-dependent lower bounds of the number of downstream samples
used for data refinement. Moreover, we conduct several experiments on both
synthetic and real-world datasets to verify our theoretical results.

    

### [[2103.14974] Automatic differentiation for Riemannian optimization on low-rank matrix and tensor-train manifolds](http://arxiv.org/abs/2103.14974)


  In scientific computing and machine learning applications, matrices and more
general multidimensional arrays (tensors) can often be approximated with the
help of low-rank decompositions. Since matrices and tensors of fixed rank form
smooth Riemannian manifolds, one of the popular tools for finding low-rank
approximations is to use Riemannian optimization. Nevertheless, efficient
implementation of Riemannian gradients and Hessians, required in Riemannian
optimization algorithms, can be a nontrivial task in practice. Moreover, in
some cases, analytic formulas are not even available. In this paper, we build
upon automatic differentiation and propose a method that, given an
implementation of the function to be minimized, efficiently computes Riemannian
gradients and matrix-by-vector products between an approximate Riemannian
Hessian and a given vector.

    

### [[2103.17258] Co-Adaptation of Algorithmic and Implementational Innovations in Inference-based Deep Reinforcement Learning](http://arxiv.org/abs/2103.17258)


  Recently many algorithms were devised for reinforcement learning (RL) with
function approximation. While they have clear algorithmic distinctions, they
also have many implementation differences that are algorithm-independent and
sometimes under-emphasized. Such mixing of algorithmic novelty and
implementation craftsmanship makes rigorous analyses of the sources of
performance improvements across algorithms difficult. In this work, we focus on
a series of off-policy inference-based actor-critic algorithms -- MPO, AWR, and
SAC -- to decouple their algorithmic innovations and implementation decisions.
We present unified derivations through a single control-as-inference objective,
where we can categorize each algorithm as based on either
Expectation-Maximization (EM) or direct Kullback-Leibler (KL) divergence
minimization and treat the rest of specifications as implementation details. We
performed extensive ablation studies, and identified substantial performance
drops whenever implementation details are mismatched for algorithmic choices.
These results show which implementation or code details are co-adapted and
co-evolved with algorithms, and which are transferable across algorithms: as
examples, we identified that tanh Gaussian policy and network sizes are highly
adapted to algorithmic types, while layer normalization and ELU are critical
for MPO's performances but also transfer to noticeable gains in SAC. We hope
our work can inspire future work to further demystify sources of performance
improvements across multiple algorithms and allow researchers to build on one
another's both algorithmic and implementational innovations.

    

### [[2104.00170] An Investigation of Critical Issues in Bias Mitigation Techniques](http://arxiv.org/abs/2104.00170)


  A critical problem in deep learning is that systems learn inappropriate
biases, resulting in their inability to perform well on minority groups. This
has led to the creation of multiple algorithms that endeavor to mitigate bias.
However, it is not clear how effective these methods are. This is because study
protocols differ among papers, systems are tested on datasets that fail to test
many forms of bias, and systems have access to hidden knowledge or are tuned
specifically to the test set. To address this, we introduce an improved
evaluation protocol, sensible metrics, and a new dataset, which enables us to
ask and answer critical questions about bias mitigation algorithms. We evaluate
seven state-of-the-art algorithms using the same network architecture and
hyperparameter selection policy across three benchmark datasets. We introduce a
new dataset called Biased MNIST that enables assessment of robustness to
multiple bias sources. We use Biased MNIST and a visual question answering
(VQA) benchmark to assess robustness to hidden biases. Rather than only tuning
to the test set distribution, we study robustness across different tuning
distributions, which is critical because for many applications the test
distribution may not be known during development. We find that algorithms
exploit hidden biases, are unable to scale to multiple forms of bias, and are
highly sensitive to the choice of tuning set. Based on our findings, we implore
the community to adopt more rigorous assessment of future bias mitigation
methods. All data, code, and results are publicly available at:
this https URL.

    

### [[2104.01194] Physics Informed Convex Artificial Neural Networks (PICANNs) for Optimal Transport based Density Estimation](http://arxiv.org/abs/2104.01194)


  Optimal Mass Transport (OMT) is a well studied problem with a variety of
applications in a diverse set of fields ranging from Physics to Computer Vision
and in particular Statistics and Data Science. Since the original formulation
of Monge in 1781 significant theoretical progress been made on the existence,
uniqueness and properties of the optimal transport maps. The actual numerical
computation of the transport maps, particularly in high dimensions, remains a
challenging problem. By Brenier's theorem, the continuous OMT problem can be
reduced to that of solving a non-linear PDE of Monge-Ampere type whose solution
is a convex function. In this paper, building on recent developments of input
convex neural networks and physics informed neural networks for solving PDE's,
we propose a Deep Learning approach to solve the continuous OMT problem.
To demonstrate the versatility of our framework we focus on the ubiquitous
density estimation and generative modeling tasks in statistics and machine
learning. Finally as an example we show how our framework can be incorporated
with an autoencoder to estimate an effective probabilistic generative model.

    

### [[2104.09967] Multi-target prediction for dummies using two-branch neural networks](http://arxiv.org/abs/2104.09967)


  Multi-target prediction (MTP) serves as an umbrella term for machine learning
tasks that concern the simultaneous prediction of multiple target variables.
Classical instantiations are multi-label classification, multivariate
regression, multi-task learning, dyadic prediction, zero-shot learning, network
inference, and matrix completion. Despite the significant similarities, all
these domains have evolved separately into distinct research areas over the
last two decades. This led to the development of a plethora of
highly-engineered methods, and created a substantially-high entrance barrier
for machine learning practitioners that are not experts in the field. In this
work we present a generic deep learning methodology that can be used for a wide
range of multi-target prediction problems. We introduce a flexible multi-branch
neural network architecture, partially configured via a questionnaire that
helps end-users to select a suitable MTP problem setting for their needs.
Experimental results for a wide range of domains illustrate that the proposed
methodology manifests a competitive performance compared to methods from
specific MTP domains.

    

### [[2104.09994] Federated Learning for Malware Detection in IoT Devices](http://arxiv.org/abs/2104.09994)


  This work investigates the possibilities enabled by federated learning
concerning IoT malware detection and studies security issues inherent to this
new learning paradigm. In this context, a framework that uses federated
learning to detect malware affecting IoT devices is presented. N-BaIoT, a
dataset modeling network traffic of several real IoT devices while affected by
malware, has been used to evaluate the proposed framework. Both supervised and
unsupervised federated models (multi-layer perceptron and autoencoder) able to
detect malware affecting seen and unseen IoT devices of N-BaIoT have been
trained and evaluated. Furthermore, their performance has been compared to two
traditional approaches. The first one lets each participant locally train a
model using only its own data, while the second consists of making the
participants share their data with a central entity in charge of training a
global model. This comparison has shown that the use of more diverse and large
data, as done in the federated and centralized methods, has a considerable
positive impact on the model performance. Besides, the federated models, while
preserving the participant's privacy, show similar results as the centralized
ones. As an additional contribution and to measure the robustness of the
federated approach, an adversarial setup with several malicious participants
poisoning the federated model has been considered. The baseline model
aggregation averaging step used in most federated learning algorithms appears
highly vulnerable to different attacks, even with a single adversary. The
performance of other model aggregation functions acting as countermeasures is
thus evaluated under the same attack scenarios. These functions provide a
significant improvement against malicious participants, but more efforts are
still needed to make federated approaches robust.

    

### [[2104.10410] Principal Component Density Estimation for Scenario Generation Using Normalizing Flows](http://arxiv.org/abs/2104.10410)


  Neural networks-based learning of the distribution of non-dispatchable
renewable electricity generation from sources such as photovoltaics (PV) and
wind as well as load demands has recently gained attention. Normalizing flow
density models are particularly well suited for this task due to the training
through direct log-likelihood maximization. However, research from the field of
image generation has shown that standard normalizing flows can only learn
smeared-out versions of manifold distributions. Previous works on normalizing
flow-based scenario generation do not address this issue, and the smeared-out
distributions result in the sampling of noisy time series. In this paper, we
propose reducing the dimensionality through principal component analysis (PCA),
which sets up the normalizing flow in a lower-dimensional space while
maintaining the direct and computationally efficient likelihood maximization.
We train the resulting principal component flow (PCF) on data of PV and wind
power generation as well as load demand in Germany in the years 2013 to 2015.
The results of this investigation show that the PCF preserves critical features
of the original distributions, such as the probability density and frequency
behavior of the time series. The application of the PCF is, however, not
limited to renewable power generation but rather extends to any data set, time
series, or otherwise, which can be efficiently reduced using PCA.

    

### [[2105.01747] Information Complexity and Generalization Bounds](http://arxiv.org/abs/2105.01747)


  We present a unifying picture of PAC-Bayesian and mutual information-based
upper bounds on the generalization error of randomized learning algorithms. As
we show, Tong Zhang's information exponential inequality (IEI) gives a general
recipe for constructing bounds of both flavors. We show that several important
results in the literature can be obtained as simple corollaries of the IEI
under different assumptions on the loss function. Moreover, we obtain new
bounds for data-dependent priors and unbounded loss functions. Optimizing the
bounds gives rise to variants of the Gibbs algorithm, for which we discuss two
practical examples for learning with neural networks, namely, Entropy- and
PAC-Bayes- SGD. Further, we use an Occam's factor argument to show a
PAC-Bayesian bound that incorporates second-order curvature information of the
training loss.

    

### [[2105.04037] Graph Attention Networks with Positional Embeddings](http://arxiv.org/abs/2105.04037)


  Graph Neural Networks (GNNs) are deep learning methods which provide the
current state of the art performance in node classification tasks. GNNs often
assume homophily -- neighboring nodes having similar features and labels--, and
therefore may not be at their full potential when dealing with non-homophilic
graphs. In this work, we focus on addressing this limitation and enable Graph
Attention Networks (GAT), a commonly used variant of GNNs, to explore the
structural information within each graph locality. Inspired by the positional
encoding in the Transformers, we propose a framework, termed Graph Attentional
Networks with Positional Embeddings (GAT-POS), to enhance GATs with positional
embeddings which capture structural and positional information of the nodes in
the graph. In this framework, the positional embeddings are learned by a model
predictive of the graph context, plugged into an enhanced GAT architecture,
which is able to leverage both the positional and content information of each
node. The model is trained jointly to optimize for the task of node
classification as well as the task of predicting graph context. Experimental
results show that GAT-POS reaches remarkable improvement compared to strong GNN
baselines and recent structural embedding enhanced GNNs on non-homophilic
graphs.

    

### [[2105.05686] Yes, BM25 is a Strong Baseline for Legal Case Retrieval](http://arxiv.org/abs/2105.05686)


  We describe our single submission to task 1 of COLIEE 2021. Our vanilla BM25
got second place, well above the median of submissions. Code is available at
this https URL.

    

### [[2105.06813] A cost-benefit analysis of cross-lingual transfer methods](http://arxiv.org/abs/2105.06813)


  An effective method for cross-lingual transfer is to fine-tune a bilingual or
multilingual model on a supervised dataset in one language and evaluating it on
another language in a zero-shot manner. Translating examples at training time
or inference time are also viable alternatives. However, there are costs
associated with these methods that are rarely addressed in the literature. In
this work, we analyze cross-lingual methods in terms of their effectiveness
(e.g., accuracy), development and deployment costs, as well as their latencies
at inference time. Our experiments on three tasks indicate that the best
cross-lingual method is highly task-dependent. Finally, by combining zero-shot
and translation methods, we achieve the state-of-the-art in two of the three
datasets used in this work. Based on these results, we question the need for
manually labeled training data in a target language. Code, models and
translated datasets are available at
this https URL


### [[2105.14490] Ladder-GNN: Hop-Aware Representation Learning for Graph Neural Networks](http://arxiv.org/abs/2105.14490)


  In the representation learning of Graph Neural Networks (GNNs), as the
messages passed among nodes contain both information and noise, it is critical
to retrieve information effectively while suppressing noise. Generally
speaking, interactions with distant nodes introduce more noise for a particular
node than those with close neighbours. However, in most existing works, the
messages being passed among nodes are mingled together, which is inefficient
from a communication perspective. Motivated by the above, we propose a simple
yet effective hop-aware aggregation scheme, resulting in a ladder-style GNN
architecture, namely Ladder-GNN. Specifically, we separate messages from
different hops, assign different dimensions for them, and then concatenate them
to obtain the node representation. Such disentangled representations facilitate
improving the information-to-noise ratio of messages passed from different
hops. To explore an effective hop-dimension relationship, we propose a
conditionally progressive neural architecture search strategy. The resulting
hop-aware representations generally contain more dimensions for low-order
neighbours and fewer dimensions for high-order neighbours, leading to a
ladder-style architecture. This observation motivates us to introduce an
efficient approximate hop-dimension relation function used in Ladder-GNN
design. We verify the proposed method on seven semi-supervised node
classification datasets, including both homogeneous and heterogeneous graphs.
Experimental results show that the proposed simple hop-aware representation
learning solution outperforms existing techniques.

    

### [[2106.00600] Fair Clustering Using Antidote Data](http://arxiv.org/abs/2106.00600)


  Clustering algorithms are widely utilized for many modern data science
applications. This motivates the need to make outputs of clustering algorithms
fair. Traditionally, new fair algorithmic variants to clustering algorithms are
developed for specific notions of fairness. However, depending on the
application context, different definitions of fairness might need to be
employed. As a result, new algorithms and analysis need to be proposed for each
combination of clustering algorithm and fairness definition. Additionally, each
new algorithm would need to be reimplemented for deployment in a real-world
system. Hence, we propose an alternate approach to group-level fairness in
center-based clustering inspired by research on data poisoning attacks. We seek
to augment the original dataset with a small number of data points, called
antidote data. When clustering is undertaken on this new dataset, the output is
fair, for the chosen clustering algorithm and fairness definition. We formulate
this as a general bi-level optimization problem which can accommodate any
center-based clustering algorithms and fairness notions. We then categorize
approaches for solving this bi-level optimization for two different problem
settings. Extensive experiments on different clustering algorithms and fairness
notions show that our algorithms can achieve desired levels of fairness on many
real-world datasets with a very small percentage of antidote data added. We
also find that our algorithms achieve lower fairness costs and competitive
clustering performance compared to other state-of-the-art fair clustering
algorithms.

    

### [[2106.00736] Large-Scale Wasserstein Gradient Flows](http://arxiv.org/abs/2106.00736)


  Wasserstein gradient flows provide a powerful means of understanding and
solving many diffusion equations. Specifically, Fokker-Planck equations, which
model the diffusion of probability measures, can be understood as gradient
descent over entropy functionals in Wasserstein space. This equivalence,
introduced by Jordan, Kinderlehrer and Otto, inspired the so-called JKO scheme
to approximate these diffusion processes via an implicit discretization of the
gradient flow in Wasserstein space. Solving the optimization problem associated
to each JKO step, however, presents serious computational challenges. We
introduce a scalable method to approximate Wasserstein gradient flows, targeted
to machine learning applications. Our approach relies on input-convex neural
networks (ICNNs) to discretize the JKO steps, which can be optimized by
stochastic gradient descent. Unlike previous work, our method does not require
domain discretization or particle simulation. As a result, we can sample from
the measure at each time step of the diffusion and compute its probability
density. We demonstrate our algorithm's performance by computing diffusions
following the Fokker-Planck equation and apply it to unnormalized density
sampling as well as nonlinear filtering.

    

### [[2106.00885] Robustifying Algorithms of Learning Latent Trees with Vector Variables](http://arxiv.org/abs/2106.00885)


  We consider learning the structures of Gaussian latent tree models with
vector observations when a subset of them are arbitrarily corrupted. First, we
present the sample complexities of Recursive Grouping (RG) and Chow-Liu
Recursive Grouping (CLRG) without the assumption that the effective depth is
bounded in the number of observed nodes, significantly generalizing the results
in Choi et al. (2011). We show that Chow-Liu initialization in CLRG greatly
reduces the sample complexity of RG from being exponential in the diameter of
the tree to only logarithmic in the diameter for the hidden Markov model (HMM).
Second, we robustify RG, CLRG, Neighbor Joining (NJ) and Spectral NJ (SNJ) by
using the truncated inner product. These robustified algorithms can tolerate a
number of corruptions up to the square root of the number of clean samples.
Finally, we derive the first known instance-dependent impossibility result for
structure learning of latent trees. The optimalities of the robust version of
CLRG and NJ are verified by comparing their sample complexities and the
impossibility result.

    

### [[2106.01577] A Provably-Efficient Model-Free Algorithm for Constrained Markov Decision Processes](http://arxiv.org/abs/2106.01577)


  This paper presents the first model-free, simulator-free reinforcement
learning algorithm for Constrained Markov Decision Processes (CMDPs) with
sublinear regret and zero constraint violation. The algorithm is named Triple-Q
because it includes three key components: a Q-function (also called
action-value function) for the cumulative reward, a Q-function for the
cumulative utility for the constraint, and a virtual-Queue that
(over)-estimates the cumulative constraint violation. Under Triple-Q, at each
step, an action is chosen based on the pseudo-Q-value that is a combination of
the three "Q" values. The algorithm updates the reward and utility Q-values
with learning rates that depend on the visit counts to the corresponding
(state, action) pairs and are periodically reset. In the episodic CMDP setting,
Triple-Q achieves $\tilde{\cal O}\left(\frac{1 }{\delta}H^4
S^{\frac{1}{2}}A^{\frac{1}{2}}K^{\frac{4}{5}} \right)$ regret, where $K$ is the
total number of episodes, $H$ is the number of steps in each episode, $S$ is
the number of states, $A$ is the number of actions, and $\delta$ is Slater's
constant. Furthermore, Triple-Q guarantees zero constraint violation, both on
expectation and with a high probability, when $K$ is sufficiently large.
Finally, the computational complexity of Triple-Q is similar to SARSA for
unconstrained MDPs and is computationally efficient.

    

### [[2106.01954] Do Neural Optimal Transport Solvers Work? A Continuous Wasserstein-2 Benchmark](http://arxiv.org/abs/2106.01954)


  Despite the recent popularity of neural network-based solvers for optimal
transport (OT), there is no standard quantitative way to evaluate their
performance. In this paper, we address this issue for quadratic-cost transport
-- specifically, computation of the Wasserstein-2 distance, a commonly-used
formulation of optimal transport in machine learning. To overcome the challenge
of computing ground truth transport maps between continuous measures needed to
assess these solvers, we use input-convex neural networks (ICNN) to construct
pairs of measures whose ground truth OT maps can be obtained analytically. This
strategy yields pairs of continuous benchmark measures in high-dimensional
spaces such as spaces of images. We thoroughly evaluate existing optimal
transport solvers using these benchmark measures. Even though these solvers
perform well in downstream tasks, many do not faithfully recover optimal
transport maps. To investigate the cause of this discrepancy, we further test
the solvers in a setting of image generation. Our study reveals crucial
limitations of existing solvers and shows that increased OT accuracy does not
necessarily correlate to better results downstream.

    

### [[2106.02684] Learning Policies with Zero or Bounded Constraint Violation for Constrained MDPs](http://arxiv.org/abs/2106.02684)


  We address the issue of safety in reinforcement learning. We pose the problem
in an episodic framework of a constrained Markov decision process. Existing
results have shown that it is possible to achieve a reward regret of
$\tilde{\mathcal{O}}(\sqrt{K})$ while allowing an
$\tilde{\mathcal{O}}(\sqrt{K})$ constraint violation in $K$ episodes. A
critical question that arises is whether it is possible to keep the constraint
violation even smaller. We show that when a strictly safe policy is known, then
one can confine the system to zero constraint violation with arbitrarily high
probability while keeping the reward regret of order
$\tilde{\mathcal{O}}(\sqrt{K})$. The algorithm which does so employs the
principle of optimistic pessimism in the face of uncertainty to achieve safe
exploration. When no strictly safe policy is known, though one is known to
exist, then it is possible to restrict the system to bounded constraint
violation with arbitrarily high probability. This is shown to be realized by a
primal-dual algorithm with an optimistic primal estimate and a pessimistic dual
update.

    

### [[2106.02847] Navigating to the Best Policy in Markov Decision Processes](http://arxiv.org/abs/2106.02847)


  We investigate the classical active pure exploration problem in Markov
Decision Processes, where the agent sequentially selects actions and, from the
resulting system trajectory, aims at identifying the best policy as fast as
possible. We propose a problem-dependent lower bound on the average number of
steps required before a correct answer can be given with probability at least
$1-\delta$. We further provide the first algorithm with an instance-specific
sample complexity in this setting. This algorithm addresses the general case of
communicating MDPs; we also propose a variant with a reduced exploration rate
(and hence faster convergence) under an additional ergodicity assumption. This
work extends previous results relative to the \emph{generative
setting}~\cite{pmlr-v139-marjani21a}, where the agent could at each step query
the random outcome of any (state, action) pair. In contrast, we show here how
to deal with the \emph{navigation constraints}, induced by the \emph{online
setting}. Our analysis relies on an ergodic theorem for non-homogeneous Markov
chains which we consider of wide interest in the analysis of Markov Decision
Processes.

    

### [[2106.03216] On Memorization in Probabilistic Deep Generative Models](http://arxiv.org/abs/2106.03216)


  Recent advances in deep generative models have led to impressive results in a
variety of application domains. Motivated by the possibility that deep learning
models might memorize part of the input data, there have been increased efforts
to understand how memorization arises. In this work, we extend a recently
proposed measure of memorization for supervised learning (Feldman, 2019) to the
unsupervised density estimation problem and adapt it to be more computationally
efficient. Next, we present a study that demonstrates how memorization can
occur in probabilistic deep generative models such as variational autoencoders.
This reveals that the form of memorization to which these models are
susceptible differs fundamentally from mode collapse and overfitting.
Furthermore, we show that the proposed memorization score measures a phenomenon
that is not captured by commonly-used nearest neighbor tests. Finally, we
discuss several strategies that can be used to limit memorization in practice.
Our work thus provides a framework for understanding problematic memorization
in probabilistic generative models.

    

### [[2106.03765] On Inductive Biases for Heterogeneous Treatment Effect Estimation](http://arxiv.org/abs/2106.03765)


  We investigate how to exploit structural similarities of an individual's
potential outcomes (POs) under different treatments to obtain better estimates
of conditional average treatment effects in finite samples. Especially when it
is unknown whether a treatment has an effect at all, it is natural to
hypothesize that the POs are similar - yet, some existing strategies for
treatment effect estimation employ regularization schemes that implicitly
encourage heterogeneity even when it does not exist and fail to fully make use
of shared structure. In this paper, we investigate and compare three end-to-end
learning strategies to overcome this problem - based on regularization,
reparametrization and a flexible multi-task architecture - each encoding
inductive bias favoring shared behavior across POs. To build understanding of
their relative strengths, we implement all strategies using neural networks and
conduct a wide range of semi-synthetic experiments. We observe that all three
approaches can lead to substantial improvements upon numerous baselines and
gain insight into performance differences across various experimental settings.

    

### [[2106.04188] Stability and Generalization of Bilevel Programming in Hyperparameter Optimization](http://arxiv.org/abs/2106.04188)


  The (gradient-based) bilevel programming framework is widely used in
hyperparameter optimization and has achieved excellent performance empirically.
Previous theoretical work mainly focuses on its optimization properties, while
leaving the analysis on generalization largely open. This paper attempts to
address the issue by presenting an expectation bound w.r.t. the validation set
based on uniform stability. Our results can explain some mysterious behaviours
of the bilevel programming in practice, for instance, overfitting to the
validation set. We also present an expectation bound for the classical
cross-validation algorithm. Our results suggest that gradient-based algorithms
can be better than cross-validation under certain conditions in a theoretical
perspective. Furthermore, we prove that regularization terms in both the outer
and inner levels can relieve the overfitting problem in gradient-based
algorithms. In experiments on feature learning and data reweighting for noisy
labels, we corroborate our theoretical findings.

    

### [[2106.05455] Effective Graph Learning with Adaptive Knowledge Exchange](http://arxiv.org/abs/2106.05455)


  Graph Neural Networks (GNNs), due to their capability to learn complex
relations (edges) among attributed objects (nodes) within graph datasets, have
already been widely used in various graph mining tasks. Considerable efforts
have been devoted to improving GNN learning through designing new architectures
and/or loss objectives. In this paper, we introduce a novel GNN learning
framework, called AKE-GNN (Adaptive-Knowledge-Exchange GNN), which adaptively
exchanges diverse knowledge learned from multiple graph views generated by
graph augmentations. Specifically, AKE-GNN iteratively exchanges redundant
channels in the weight matrix of one GNN by informative channels of another GNN
in a layer-wise manner. Furthermore, existing GNN models can be seamlessly
incorporated into our framework. Extensive experiments on node classification,
graph classification, and edge prediction demonstrate the effectiveness of
AKE-GNN. In particular, we conduct a series of experiments on 15 public
benchmarks, 8 popular GNN models, and 3 graph tasks -- node classification,
graph classification, and edge prediction -- and show that AKE-GNN consistently
outperforms existing popular GNN models and even their ensembles. On the Cora
semi-supervised node classification dataset, our framework achieves new
state-of-the-art results. Extensive ablation studies and analyses on knowledge
exchange methods also verify the effectiveness of AKE-GNN.

    

### [[2106.05960] Compositional Modeling of Nonlinear Dynamical Systems with ODE-based Random Features](http://arxiv.org/abs/2106.05960)


  Effectively modeling phenomena present in highly nonlinear dynamical systems
whilst also accurately quantifying uncertainty is a challenging task, which
often requires problem-specific techniques. We present a novel, domain-agnostic
approach to tackling this problem, using compositions of physics-informed
random features, derived from ordinary differential equations. The architecture
of our model leverages recent advances in approximate inference for deep
Gaussian processes, such as layer-wise weight-space approximations which allow
us to incorporate random Fourier features, and stochastic variational inference
for approximate Bayesian inference. We provide evidence that our model is
capable of capturing highly nonlinear behaviour in real-world multivariate time
series data. In addition, we find that our approach achieves comparable
performance to a number of other probabilistic models on benchmark regression
tasks.

    

### [[2106.06573] Understanding Deflation Process in Over-parametrized Tensor Decomposition](http://arxiv.org/abs/2106.06573)


  In this paper we study the training dynamics for gradient flow on
over-parametrized tensor decomposition problems. Empirically, such training
process often first fits larger components and then discovers smaller
components, which is similar to a tensor deflation process that is commonly
used in tensor decomposition algorithms. We prove that for orthogonally
decomposable tensor, a slightly modified version of gradient flow would follow
a tensor deflation process and recover all the tensor components. Our proof
suggests that for orthogonal tensors, gradient flow dynamics works similarly as
greedy low-rank learning in the matrix setting, which is a first step towards
understanding the implicit regularization effect of over-parametrized models
for low-rank tensors.

    

### [[2106.07094] Privacy-Preserving Federated Learning via Normalized (instead of Clipped) Updates](http://arxiv.org/abs/2106.07094)


  Differentially private federated learning (FL) entails bounding the
sensitivity to each client's update. The customary approach used in practice
for bounding sensitivity is to \textit{clip} the client updates, which is just
projection onto an $\ell_2$ ball of some radius (called the clipping threshold)
centered at the origin. However, clipping introduces bias depending on the
clipping threshold and its impact on convergence has not been properly analyzed
in the FL literature. In this work, we propose a simpler alternative for
bounding sensitivity which is \textit{normalization}, i.e. use only the
\textit{unit vector} along the client updates, completely discarding the
magnitude information. We call this algorithm \texttt{DP-NormFedAvg} and show
that it has the same order-wise convergence rate as \texttt{FedAvg} on smooth
quasar-convex functions (an important class of non-convex functions for
modeling optimization of deep neural networks) modulo the noise variance term
(due to privacy). Further, assuming that the per-sample client losses obey a
strong-growth type of condition, we show that with high probability, the
sensitivity reduces by a factor of $\mathcal{O}(\frac{1}{m})$, where $m$ is the
minimum number of samples within a client, compared to its worst-case value.
Using this high probability sensitivity value enables us to reduce the
iteration complexity of \texttt{DP-NormFedAvg} by a factor of
$\mathcal{O}(\frac{1}{m^2})$, at the expense of an exponentially small
degradation in the privacy guarantee. We also corroborate our theory with
experiments on neural networks.

    

### [[2106.07098] Security Analysis of Camera-LiDAR Fusion Against Black-Box Attacks on Autonomous Vehicles](http://arxiv.org/abs/2106.07098)


  To enable safe and reliable decision-making, autonomous vehicles (AVs) feed
sensor data to perception algorithms to understand the environment. Sensor
fusion with multi-frame tracking is becoming increasingly popular for detecting
3D objects. Thus, in this work, we perform an analysis of camera-LiDAR fusion,
in the AV context, under LiDAR spoofing attacks. Recently, LiDAR-only
perception was shown vulnerable to LiDAR spoofing attacks; however, we
demonstrate these attacks are not capable of disrupting camera-LiDAR fusion. We
then define a novel, context-aware attack: frustum attack, and show that out of
8 widely used perception algorithms - across 3 architectures of LiDAR-only and
3 architectures of camera-LiDAR fusion - all are significantly vulnerable to
the frustum attack. In addition, we demonstrate that the frustum attack is
stealthy to existing defenses against LiDAR spoofing as it preserves
consistencies between camera and LiDAR semantics. Finally, we show that the
frustum attack can be exercised consistently over time to form stealthy
longitudinal attack sequences, compromising the tracking module and creating
adverse outcomes on end-to-end AV control.

    

### [[2106.07289] Decentralized Personalized Federated Min-Max Problems](http://arxiv.org/abs/2106.07289)


  Personalized Federated Learning (PFL) has recently seen tremendous progress,
allowing the design of novel machine learning applications to preserve the
privacy of the training data. Existing theoretical results in this field mainly
focus on distributed optimization for minimization problems. This paper is the
first to study PFL for saddle point problems (which cover a broader class of
optimization problems), allowing for a more rich class of applications
requiring more than just solving minimization problems. In this work, we
consider a recently proposed PFL setting with the mixing objective function, an
approach combining the learning of a global model together with locally
distributed learners. Unlike most previous work, which considered only the
centralized setting, we work in a more general and decentralized setup that
allows us to design and analyze more practical and federated ways to connect
devices to the network. We proposed new algorithms to address this problem and
provide a theoretical analysis of the smooth
(strongly-)convex-(strongly-)concave saddle point problems in stochastic and
deterministic cases. Numerical experiments for bilinear problems and neural
networks with adversarial noise demonstrate the effectiveness of the proposed
methods.

    

### [[2106.07411] Partial success in closing the gap between human and machine vision](http://arxiv.org/abs/2106.07411)


  A few years ago, the first CNN surpassed human performance on ImageNet.
However, it soon became clear that machines lack robustness on more challenging
test cases, a major obstacle towards deploying machines "in the wild" and
towards obtaining better computational models of human visual perception. Here
we ask: Are we making progress in closing the gap between human and machine
vision? To answer this question, we tested human observers on a broad range of
out-of-distribution (OOD) datasets, recording 85,120 psychophysical trials
across 90 participants. We then investigated a range of promising machine
learning developments that crucially deviate from standard supervised CNNs
along three axes: objective function (self-supervised, adversarially trained,
CLIP language-image training), architecture (e.g. vision transformers), and
dataset size (ranging from 1M to 1B).
Our findings are threefold. (1.) The longstanding distortion robustness gap
between humans and CNNs is closing, with the best models now exceeding human
feedforward performance on most of the investigated OOD datasets. (2.) There is
still a substantial image-level consistency gap, meaning that humans make
different errors than models. In contrast, most models systematically agree in
their categorisation errors, even substantially different ones like contrastive
self-supervised vs. standard supervised models. (3.) In many cases,
human-to-model consistency improves when training dataset size is increased by
one to three orders of magnitude. Our results give reason for cautious
optimism: While there is still much room for improvement, the behavioural
difference between human and machine vision is narrowing. In order to measure
future progress, 17 OOD datasets with image-level human behavioural data and
evaluation code are provided as a toolbox and benchmark at:
this https URL


### [[2106.08630] HELP: Hardware-Adaptive Efficient Latency Prediction for NAS via Meta-Learning](http://arxiv.org/abs/2106.08630)


  For deployment, neural architecture search should be hardware-aware, in order
to satisfy the device-specific constraints (e.g., memory usage, latency and
energy consumption) and enhance the model efficiency. Existing methods on
hardware-aware NAS collect a large number of samples (e.g., accuracy and
latency) from a target device, either builds a lookup table or a latency
estimator. However, such approach is impractical in real-world scenarios as
there exist numerous devices with different hardware specifications, and
collecting samples from such a large number of devices will require prohibitive
computational and monetary cost. To overcome such limitations, we propose
Hardware-adaptive Efficient Latency Predictor (HELP), which formulates the
device-specific latency estimation problem as a meta-learning problem, such
that we can estimate the latency of a model's performance for a given task on
an unseen device with a few samples. To this end, we introduce novel hardware
embeddings to embed any devices considering them as black-box functions that
output latencies, and meta-learn the hardware-adaptive latency predictor in a
device-dependent manner, using the hardware embeddings. We validate the
proposed HELP for its latency estimation performance on unseen platforms, on
which it achieves high estimation performance with as few as 10 measurement
samples, outperforming all relevant baselines. We also validate end-to-end NAS
frameworks using HELP against ones without it, and show that it largely reduces
the total time cost of the base NAS method, in latency-constrained settings.
Code is available at this https URL.

    

### [[2106.08903] GemNet: Universal Directional Graph Neural Networks for Molecules](http://arxiv.org/abs/2106.08903)


  Effectively predicting molecular interactions has the potential to accelerate
molecular dynamics by multiple orders of magnitude and thus revolutionize
chemical simulations. Graph neural networks (GNNs) have recently shown great
successes for this task, overtaking classical methods based on fixed molecular
kernels. However, they still appear very limited from a theoretical
perspective, since regular GNNs cannot distinguish certain types of graphs. In
this work we close this gap between theory and practice. We show that GNNs with
directed edge embeddings and two-hop message passing are indeed universal
approximators for predictions that are invariant to translation, and
equivariant to permutation and rotation. We then leverage these insights and
multiple structural improvements to propose the geometric message passing
neural network (GemNet). We demonstrate the benefits of the proposed changes in
multiple ablation studies. GemNet outperforms previous models on the COLL,
MD17, and OC20 datasets by 34%, 41%, and 20%, respectively, and performs
especially well on the most challenging molecules. Our implementation is
available online.

    

### [[2106.08918] Towards Automatic Actor-Critic Solutions to Continuous Control](http://arxiv.org/abs/2106.08918)


  Model-free off-policy actor-critic methods are an efficient solution to
complex continuous control tasks. However, these algorithms rely on a number of
design tricks and hyperparameters, making their application to new domains
difficult and computationally expensive. This paper creates an evolutionary
approach that automatically tunes these design decisions and eliminates the
RL-specific hyperparameters from the Soft Actor-Critic algorithm. Our design is
sample efficient and provides practical advantages over baseline approaches,
including improved exploration, generalization over multiple control
frequencies, and a robust ensemble of high-performance policies. Empirically,
we show that our agent outperforms well-tuned hyperparameter settings in
popular benchmarks from the DeepMind Control Suite. We then apply it to less
common control tasks outside of simulated robotics to find high-performance
solutions with minimal compute and research effort.

    

### [[2106.09563] On Anytime Learning at Macroscale](http://arxiv.org/abs/2106.09563)


  Classical machine learning frameworks assume access to a possibly large
dataset in order to train a predictive model. In many practical applications
however, data does not arrive all at once, but in batches over time. This
creates a natural trade-off between accuracy of a model and time to obtain such
a model. A greedy predictor could produce non-trivial predictions by
immediately training on batches as soon as these become available but, it may
also make suboptimal use of future data. On the other hand, a tardy predictor
could wait for a long time to aggregate several batches into a larger dataset,
but ultimately deliver a much better performance. In this work, we consider
such a streaming learning setting, which we dub anytime learning at macroscale}
(ALMA). It is an instance of anytime learning applied not at the level of a
single chunk of data, but at the level of the entire sequence of large batches.
We first formalize this learning setting, we then introduce metrics to assess
how well learners perform on the given task for a given memory and compute
budget, and finally we test about thirty baseline approaches on three standard
benchmarks repurposed for anytime learning at macroscale. Our findings indicate
that no model strikes the best trade-off across the board. While replay-based
methods attain the lowest error rate, they also incur in a 5 to 10 times
increase of compute. Approaches that grow capacity over time do offer better
scaling in terms of training flops, but they also underperform simpler
ensembling methods in terms of error rate. Overall, ALMA offers both a good
abstraction of the typical learning setting faced everyday by practitioners,
and a set of unsolved modeling problems for those interested in efficient
learning of dynamic models.

    

### [[2106.10517] A Max-Min Entropy Framework for Reinforcement Learning](http://arxiv.org/abs/2106.10517)


  In this paper, we propose a max-min entropy framework for reinforcement
learning (RL) to overcome the limitation of the soft actor-critic (SAC)
algorithm implementing the maximum entropy RL in model-free sample-based
learning. Whereas the maximum entropy RL guides learning for policies to reach
states with high entropy in the future, the proposed max-min entropy framework
aims to learn to visit states with low entropy and maximize the entropy of
these low-entropy states to promote better exploration. For general Markov
decision processes (MDPs), an efficient algorithm is constructed under the
proposed max-min entropy framework based on disentanglement of exploration and
exploitation. Numerical results show that the proposed algorithm yields drastic
performance improvement over the current state-of-the-art RL algorithms.

    

### [[2106.10543] Signal Processing Based Deep Learning for Blind Symbol Decoding and Modulation Classification](http://arxiv.org/abs/2106.10543)


  Blindly decoding a signal requires estimating its unknown transmit
parameters, compensating for the wireless channel impairments, and identifying
the modulation type. While deep learning can solve complex problems, digital
signal processing (DSP) is interpretable and can be more computationally
efficient. To combine both, we propose the dual path network (DPN). It consists
of a signal path of DSP operations that recover the signal, and a feature path
of neural networks that estimate the unknown transmit parameters. By
interconnecting the paths over several recovery stages, later stages benefit
from the recovered signals and reuse all the previously extracted features. The
proposed design is demonstrated to provide 5% improvement in modulation
classification compared to alternative designs lacking either feature sharing
or access to recovered signals. The estimation results of DPN along with its
blind decoding performance are shown to outperform a blind signal processing
algorithm for BPSK and QPSK on a simulated dataset. An over-the-air
software-defined-radio capture was used to verify DPN results at high SNRs. DPN
design can process variable length inputs and is shown to outperform relying on
fixed length inputs with prediction averaging on longer signals by up to 15% in
modulation classification.

    

### [[2106.10773] Neural Spectral Marked Point Processes](http://arxiv.org/abs/2106.10773)


  Self- and mutually-exciting point processes are popular models in machine
learning and statistics for dependent discrete event data. To date, most
existing models assume stationary kernels (including the classical Hawkes
processes) and simple parametric models. Modern applications with complex event
data require more general point process models that can incorporate contextual
information of the events, called marks, besides the temporal and location
information. Moreover, such applications often require non-stationary models to
capture more complex spatio-temporal dependence. To tackle these challenges, a
key question is to devise a versatile influence kernel in the point process
model. In this paper, we introduce a novel and general neural network-based
non-stationary influence kernel with high expressiveness for handling complex
discrete events data while providing theoretical performance guarantees. We
demonstrate the superior performance of our proposed method compared with the
state-of-the-art on synthetic and real data.

    

### [[2106.14702] Scalable Optimal Classifiers for Adversarial Settings under Uncertainty](http://arxiv.org/abs/2106.14702)


  We consider the problem of finding optimal classifiers in an adversarial
setting where the class-1 data is generated by an attacker whose objective is
not known to the defender -- an aspect that is key to realistic applications
but has so far been overlooked in the literature. To model this situation, we
propose a Bayesian game framework where the defender chooses a classifier with
no a priori restriction on the set of possible classifiers. The key difficulty
in the proposed framework is that the set of possible classifiers is
exponential in the set of possible data, which is itself exponential in the
number of features used for classification. To counter this, we first show that
Bayesian Nash equilibria can be characterized completely via functional
threshold classifiers with a small number of parameters. We then show that this
low-dimensional characterization enables to develop a training method to
compute provably approximately optimal classifiers in a scalable manner; and to
develop a learning algorithm for the online setting with low regret (both
independent of the dimension of the set of possible data). We illustrate our
results through simulations.

    

### [[2106.14979] On component interactions in two-stage recommender systems](http://arxiv.org/abs/2106.14979)


  Thanks to their scalability, two-stage recommenders are used by many of
today's largest online platforms, including YouTube, LinkedIn, and Pinterest.
These systems produce recommendations in two steps: (i) multiple nominators,
tuned for low prediction latency, preselect a small subset of candidates from
the whole item pool; (ii) a slower but more accurate ranker further narrows
down the nominated items, and serves to the user. Despite their popularity, the
literature on two-stage recommenders is relatively scarce, and the algorithms
are often treated as mere sums of their parts. Such treatment presupposes that
the two-stage performance is explained by the behavior of the individual
components in isolation. This is not the case: using synthetic and real-world
data, we demonstrate that interactions between the ranker and the nominators
substantially affect the overall performance. Motivated by these findings, we
derive a generalization lower bound which shows that independent nominator
training can lead to performance on par with uniformly random recommendations.
We find that careful design of item pools, each assigned to a different
nominator, alleviates these issues. As manual search for a good pool allocation
is difficult, we propose to learn one instead using a Mixture-of-Experts based
approach. This significantly improves both precision and recall at K.

    

### [[2106.15416] High-dimensional separability for one- and few-shot learning](http://arxiv.org/abs/2106.15416)


  This work is driven by a practical question: corrections of Artificial
Intelligence (AI) errors. These corrections should be quick and non-iterative.
To solve this problem without modification of a legacy AI system, we propose
special `external' devices, correctors. Elementary correctors consist of two
parts, a classifier that separates the situations with high risk of error from
the situations in which the legacy AI system works well and a new decision for
situations with potential errors. Input signals for the correctors can be the
inputs of the legacy AI system, its internal signals, and outputs. If the
intrinsic dimensionality of data is high enough then the classifiers for
correction of small number of errors can be very simple. According to the
blessing of dimensionality effects, even simple and robust Fisher's
discriminants can be used for one-shot learning of AI correctors. Stochastic
separation theorems provide the mathematical basis for this one-short learning.
However, as the number of correctors needed grows, the cluster structure of
data becomes important and a new family of stochastic separation theorems is
required. We refuse the classical hypothesis of the regularity of the data
distribution and assume that the data can have a fine-grained structure with
many clusters and peaks in the probability density. New stochastic separation
theorems for data with fine-grained structure are formulated and proved. The
multi-correctors for granular data are proposed. The advantages of the
multi-corrector technology were demonstrated by examples of correcting errors
and learning new classes of objects by a deep convolutional neural network on
the CIFAR-10 dataset. The key problems of the non-classical high-dimensional
data analysis are reviewed together with the basic preprocessing steps
including supervised, semi-supervised and domain adaptation Principal Component
Analysis.

    

### [[2106.15610] An Image is Worth More Than a Thousand Words: Towards Disentanglement in the Wild](http://arxiv.org/abs/2106.15610)


  Unsupervised disentanglement has been shown to be theoretically impossible
without inductive biases on the models and the data. As an alternative
approach, recent methods rely on limited supervision to disentangle the factors
of variation and allow their identifiability. While annotating the true
generative factors is only required for a limited number of observations, we
argue that it is infeasible to enumerate all the factors of variation that
describe a real-world image distribution. To this end, we propose a method for
disentangling a set of factors which are only partially labeled, as well as
separating the complementary set of residual factors that are never explicitly
specified. Our success in this challenging setting, demonstrated on synthetic
benchmarks, gives rise to leveraging off-the-shelf image descriptors to
partially annotate a subset of attributes in real image domains (e.g. of human
faces) with minimal manual effort. Specifically, we use a recent language-image
embedding model (CLIP) to annotate a set of attributes of interest in a
zero-shot manner and demonstrate state-of-the-art disentangled image
manipulation results.

    

### [[2106.15728] Detecting Errors and Estimating Accuracy on Unlabeled Data with Self-training Ensembles](http://arxiv.org/abs/2106.15728)


  When a deep learning model is deployed in the wild, it can encounter test
data drawn from distributions different from the training data distribution and
suffer drop in performance. For safe deployment, it is essential to estimate
the accuracy of the pre-trained model on the test data. However, the labels for
the test inputs are usually not immediately available in practice, and
obtaining them can be expensive. This observation leads to two challenging
tasks: (1) unsupervised accuracy estimation, which aims to estimate the
accuracy of a pre-trained classifier on a set of unlabeled test inputs; (2)
error detection, which aims to identify mis-classified test inputs. In this
paper, we propose a principled and practically effective framework that
simultaneously addresses the two tasks. The proposed framework iteratively
learns an ensemble of models to identify mis-classified data points and
performs self-training to improve the ensemble with the identified points.
Theoretical analysis demonstrates that our framework enjoys provable guarantees
for both accuracy estimation and error detection under mild conditions readily
satisfied by practical deep learning models. Along with the framework, we
proposed and experimented with two instantiations and achieved state-of-the-art
results on 59 tasks. For example, on iWildCam, one instantiation reduces the
estimation error for unsupervised accuracy estimation by at least 70% and
improves the F1 score for error detection by at least 4.7% compared to existing
methods.

    

### [[2106.16147] Nearly-Tight and Oblivious Algorithms for Explainable Clustering](http://arxiv.org/abs/2106.16147)


  We study the problem of explainable clustering in the setting first
formalized by Dasgupta, Frost, Moshkovitz, and Rashtchian (ICML 2020). A
$k$-clustering is said to be explainable if it is given by a decision tree
where each internal node splits data points with a threshold cut in a single
dimension (feature), and each of the $k$ leaves corresponds to a cluster. We
give an algorithm that outputs an explainable clustering that loses at most a
factor of $O(\log^2 k)$ compared to an optimal (not necessarily explainable)
clustering for the $k$-medians objective, and a factor of $O(k \log^2 k)$ for
the $k$-means objective. This improves over the previous best upper bounds of
$O(k)$ and $O(k^2)$, respectively, and nearly matches the previous $\Omega(\log
k)$ lower bound for $k$-medians and our new $\Omega(k)$ lower bound for
$k$-means. The algorithm is remarkably simple. In particular, given an initial
not necessarily explainable clustering in $\mathbb{R}^d$, it is oblivious to
the data points and runs in time $O(dk \log^2 k)$, independent of the number of
data points $n$. Our upper and lower bounds also generalize to objectives given
by higher $\ell_p$-norms.

    

### [[2107.00488] Differentiable Particle Filters through Conditional Normalizing Flow](http://arxiv.org/abs/2107.00488)


  Differentiable particle filters provide a flexible mechanism to adaptively
train dynamic and measurement models by learning from observed data. However,
most existing differentiable particle filters are within the bootstrap particle
filtering framework and fail to incorporate the information from latest
observations to construct better proposals. In this paper, we utilize
conditional normalizing flows to construct proposal distributions for
differentiable particle filters, enriching the distribution families that the
proposal distributions can represent. In addition, normalizing flows are
incorporated in the construction of the dynamic model, resulting in a more
expressive dynamic model. We demonstrate the performance of the proposed
conditional normalizing flow-based differentiable particle filters in a visual
tracking task.

    

### [[2107.01076] DUKweb: Diachronic word representations from the UK Web Archive corpus](http://arxiv.org/abs/2107.01076)


  Lexical semantic change (detecting shifts in the meaning and usage of words)
is an important task for social and cultural studies as well as for Natural
Language Processing applications. Diachronic word embeddings (time-sensitive
vector representations of words that preserve their meaning) have become the
standard resource for this task. However, given the significant computational
resources needed for their generation, very few resources exist that make
diachronic word embeddings available to the scientific community.
In this paper we present DUKweb, a set of large-scale resources designed for
the diachronic analysis of contemporary English. DUKweb was created from the
JISC UK Web Domain Dataset (1996-2013), a very large archive which collects
resources from the Internet Archive that were hosted on domains ending in
`.uk'. DUKweb consists of a series word co-occurrence matrices and two types of
word embeddings for each year in the JISC UK Web Domain dataset. We show the
reuse potential of DUKweb and its quality standards via a case study on word
meaning change detection.

    

### [[2109.02485] Severity and Mortality Prediction Models to Triage Indian COVID-19 Patients](http://arxiv.org/abs/2109.02485)


  As the second wave in India mitigates, COVID-19 has now infected about 29
million patients countrywide, leading to more than 350 thousand people dead. As
the infections surged, the strain on the medical infrastructure in the country
became apparent. While the country vaccinates its population, opening up the
economy may lead to an increase in infection rates. In this scenario, it is
essential to effectively utilize the limited hospital resources by an informed
patient triaging system based on clinical parameters. Here, we present two
interpretable machine learning models predicting the clinical outcomes,
severity, and mortality, of the patients based on routine non-invasive
surveillance of blood parameters from one of the largest cohorts of Indian
patients at the day of admission. Patient severity and mortality prediction
models achieved 86.3% and 88.06% accuracy, respectively, with an AUC-ROC of
0.91 and 0.92. We have integrated both the models in a user-friendly web app
calculator, this https URL, to showcase the potential
deployment of such efforts at scale.

    

### [[2109.09506] Decoupling Long- and Short-Term Patterns in Spatiotemporal Inference](http://arxiv.org/abs/2109.09506)


  Sensors are the key to sensing the environment and imparting benefits to
smart cities in many aspects, such as providing real-time air quality
information throughout an urban area. However, a prerequisite is to obtain
fine-grained knowledge of the environment. There is a limit to how many sensors
can be installed in the physical world due to non-negligible expenses. In this
paper, we propose to infer real-time information of any given location in a
city based on historical and current observations from the available sensors
(termed spatiotemporal inference). Our approach decouples the modeling of
short-term and long-term patterns, relying on two major components. Firstly,
unlike previous studies that separated the spatial and temporal relation
learning, we introduce a joint spatiotemporal graph attention network that
learns the short-term dependencies across both the spatial and temporal
dimensions. Secondly, we propose an adaptive graph recurrent network with a
time skip for capturing long-term patterns. The adaptive adjacency matrices are
learned inductively first as the inputs of a recurrent network to learn dynamic
dependencies. Experimental results on four public real-world datasets show that
our method reduces state-of-the-art baseline mean absolute errors by 5%~12%.

    

### [[2110.04831] Feature Imitating Networks](http://arxiv.org/abs/2110.04831)


  In this paper, we introduce a novel approach to neural learning: the
Feature-Imitating-Network (FIN). A FIN is a neural network with weights that
are initialized to reliably approximate one or more closed-form statistical
features, such as Shannon's entropy. In this paper, we demonstrate that FINs
(and FIN ensembles) provide best-in-class performance for a variety of
downstream signal processing and inference tasks, while using less data and
requiring less fine-tuning compared to other networks of similar (or even
greater) representational power. We conclude that FINs can help bridge the gap
between domain experts and machine learning practitioners by enabling
researchers to harness insights from feature-engineering to enhance the
performance of contemporary representation learning approaches.

    

### [[2110.04844] Frequency-aware SGD for Efficient Embedding Learning with Provable Benefits](http://arxiv.org/abs/2110.04844)


  Embedding learning has found widespread applications in recommendation
systems and natural language modeling, among other domains. To learn quality
embeddings efficiently, adaptive learning rate algorithms have demonstrated
superior empirical performance over SGD, largely accredited to their
token-dependent learning rate. However, the underlying mechanism for the
efficiency of token-dependent learning rate remains underexplored. We show that
incorporating frequency information of tokens in the embedding learning
problems leads to provably efficient algorithms, and demonstrate that common
adaptive algorithms implicitly exploit the frequency information to a large
extent. Specifically, we propose (Counter-based) Frequency-aware Stochastic
Gradient Descent, which applies a frequency-dependent learning rate for each
token, and exhibits provable speed-up compared to SGD when the token
distribution is imbalanced. Empirically, we show the proposed algorithms are
able to improve or match adaptive algorithms on benchmark recommendation tasks
and a large-scale industrial recommendation system, closing the performance gap
between SGD and adaptive algorithms. Our results are the first to show
token-dependent learning rate provably improves convergence for non-convex
embedding learning problems.

    

### [[2110.08996] Finding Everything within Random Binary Networks](http://arxiv.org/abs/2110.08996)


  A recent work by Ramanujan et al. (2020) provides significant empirical
evidence that sufficiently overparameterized, random neural networks contain
untrained subnetworks that achieve state-of-the-art accuracy on several
predictive tasks. A follow-up line of theoretical work provides justification
of these findings by proving that slightly overparameterized neural networks,
with commonly used continuous-valued random initializations can indeed be
pruned to approximate any target network. In this work, we show that the
amplitude of those random weights does not even matter. We prove that any
target network can be approximated up to arbitrary accuracy by simply pruning a
random network of binary $\{\pm1\}$ weights that is only a polylogarithmic
factor wider and deeper than the target network.

    

### [[2110.09295] Fair Tree Learning](http://arxiv.org/abs/2110.09295)


  When dealing with sensitive data in automated data-driven decision-making, an
important concern is to learn predictors with high performance towards a class
label, whilst minimising for the discrimination towards some sensitive
attribute, like gender or race, induced from biased data. Various hybrid
optimisation criteria exist which combine classification performance with a
fairness metric. However, while the threshold-free ROC-AUC is the standard for
measuring traditional classification model performance, current fair decision
tree methods only optimise for a fixed threshold on both the classification
task as well as the fairness metric. Moreover, current tree learning frameworks
do not allow for fair treatment with respect to multiple categories or multiple
sensitive attributes. Lastly, the end-users of a fair model should be able to
balance fairness and classification performance according to their specific
ethical, legal, and societal needs. In this paper we address these shortcomings
by proposing a threshold-independent fairness metric termed uniform demographic
parity, and a derived splitting criterion entitled SCAFF -- Splitting Criterion
AUC for Fairness -- towards fair decision tree learning, which extends to
bagged and boosted frameworks. Compared to the state-of-the-art, our method
provides three main advantages: (1) classifier performance and fairness are
defined continuously instead of relying upon an, often arbitrary, decision
threshold; (2) it leverages multiple sensitive attributes simultaneously, of
which the values may be multicategorical; and (3) the unavoidable
performance-fairness trade-off is tunable during learning. In our experiments,
we demonstrate how SCAFF attains high predictive performance towards the class
label and low discrimination with respect to binary, multicategorical, and
multiple sensitive attributes, further substantiating our claims.

    

### [[2110.11072] Sequential Modeling with Multiple Attributes for Watchlist Recommendation in E-Commerce](http://arxiv.org/abs/2110.11072)


  In e-commerce, the watchlist enables users to track items over time and has
emerged as a primary feature, playing an important role in users' shopping
journey. Watchlist items typically have multiple attributes whose values may
change over time (e.g., price, quantity). Since many users accumulate dozens of
items on their watchlist, and since shopping intents change over time,
recommending the top watchlist items in a given context can be valuable. In
this work, we study the watchlist functionality in e-commerce and introduce a
novel watchlist recommendation task. Our goal is to prioritize which watchlist
items the user should pay attention to next by predicting the next items the
user will click. We cast this task as a specialized sequential recommendation
task and discuss its characteristics. Our proposed recommendation model,
Trans2D, is built on top of the Transformer architecture, where we further
suggest a novel extended attention mechanism (Attention2D) that allows to learn
complex item-item, attribute-attribute and item-attribute patterns from
sequential-data with multiple item attributes. Using a large-scale watchlist
dataset from eBay, we evaluate our proposed model, where we demonstrate its
superiority compared to multiple state-of-the-art baselines, many of which are
adapted for this task.

    

### [[2110.11265] Deep Reinforcement Learning for Online Control of Stochastic Partial Differential Equations](http://arxiv.org/abs/2110.11265)


  In many areas, such as the physical sciences, life sciences, and finance,
control approaches are used to achieve a desired goal in complex dynamical
systems governed by differential equations. In this work we formulate the
problem of controlling stochastic partial differential equations (SPDE) as a
reinforcement learning problem. We present a learning-based, distributed
control approach for online control of a system of SPDEs with high dimensional
state-action space using deep deterministic policy gradient method. We tested
the performance of our method on the problem of controlling the stochastic
Burgers' equation, describing a turbulent fluid flow in an infinitely large
domain.

    

### [[2110.11291] Likelihood Training of Schrödinger Bridge using Forward-Backward SDEs Theory](http://arxiv.org/abs/2110.11291)


  Schrödinger Bridge (SB) is an optimal transport problem that has received
increasing attention in deep generative modeling for its mathematical
flexibility compared to the Scored-based Generative Model (SGM). However, it
remains unclear whether the optimization principle of SB relates to the modern
training of deep generative models, which often rely on constructing
parameterized log-likelihood objectives.This raises questions on the
suitability of SB models as a principled alternative for generative
applications. In this work, we present a novel computational framework for
likelihood training of SB models grounded on Forward-Backward Stochastic
Differential Equations Theory -- a mathematical methodology appeared in
stochastic optimal control that transforms the optimality condition of SB into
a set of SDEs. Crucially, these SDEs can be used to construct the likelihood
objectives for SB that, surprisingly, generalizes the ones for SGM as special
cases. This leads to a new optimization principle that inherits the same SB
optimality yet without losing applications of modern generative training
techniques, and we show that the resulting training algorithm achieves
comparable results on generating realistic images on MNIST, CelebA, and
CIFAR10.

    

### [[2110.09127] SpecTNT: a Time-Frequency Transformer for Music Audio](http://arxiv.org/abs/2110.09127)


  Transformers have drawn attention in the MIR field for their remarkable
performance shown in natural language processing and computer vision. However,
prior works in the audio processing domain mostly use Transformer as a temporal
feature aggregator that acts similar to RNNs. In this paper, we propose
SpecTNT, a Transformer-based architecture to model both spectral and temporal
sequences of an input time-frequency representation. Specifically, we introduce
a novel variant of the Transformer-in-Transformer (TNT) architecture. In each
SpecTNT block, a spectral Transformer extracts frequency-related features into
the frequency class token (FCT) for each frame. Later, the FCTs are linearly
projected and added to the temporal embeddings (TEs), which aggregate useful
information from the FCTs. Then, a temporal Transformer processes the TEs to
exchange information across the time axis. By stacking the SpecTNT blocks, we
build the SpecTNT model to learn the representation for music signals. In
experiments, SpecTNT demonstrates state-of-the-art performance in music tagging
and vocal melody extraction, and shows competitive performance for chord
recognition. The effectiveness of SpecTNT and other design choices are further
examined through ablation studies.

    

### [[2110.12127] Low-Latency VLSI Architectures for Modular Polynomial Multiplication via Fast Filtering and Applications to Lattice-Based Cryptography](http://arxiv.org/abs/2110.12127)


  This paper presents a low-latency hardware accelerator for modular polynomial
multiplication for lattice-based post-quantum cryptography and homomorphic
encryption applications. The proposed novel modular polynomial multiplier
exploits the fast finite impulse response (FIR) filter architecture to reduce
the computational complexity for the schoolbook modular polynomial
multiplication. We also extend this structure to fast M-parallel architectures
while achieving low-latency, high-speed, and full hardware utilization. We
comprehensively evaluate the performance of the proposed architectures under
various polynomial settings as well as in the Saber scheme for post-quantum
cryptography as a case study. The experimental results show that our design
reduces the computational time and area-time product by 61% and 32%,
respectively, compared to the state-of-the-art designs.

    

### [[2104.07582] SISA: Set-Centric Instruction Set Architecture for Graph Mining on Processing-in-Memory Systems](http://arxiv.org/abs/2104.07582)


  Simple graph algorithms such as PageRank have been the target of numerous
hardware accelerators. Yet, there also exist much more complex graph mining
algorithms for problems such as clustering or maximal clique listing. These
algorithms are memory-bound and thus could be accelerated by hardware
techniques such as Processing-in-Memory (PIM). However, they also come with
nonstraightforward parallelism and complicated memory access patterns. In this
work, we address this problem with a simple yet surprisingly powerful
observation: operations on sets of vertices, such as intersection or union,
form a large part of many complex graph mining algorithms, and can offer rich
and simple parallelism at multiple levels. This observation drives our
cross-layer design, in which we (1) expose set operations using a novel
programming paradigm, (2) express and execute these operations efficiently with
carefully designed set-centric ISA extensions called SISA, and (3) use PIM to
accelerate SISA instructions. The key design idea is to alleviate the bandwidth
needs of SISA instructions by mapping set operations to two types of PIM:
in-DRAM bulk bitwise computing for bitvectors representing high-degree
vertices, and near-memory logic layers for integer arrays representing
low-degree vertices. Set-centric SISA-enhanced algorithms are efficient and
outperform hand-tuned baselines, offering more than 10x speedup over the
established Bron-Kerbosch algorithm for listing maximal cliques. We deliver
more than 10 SISA set-centric algorithm formulations, illustrating SISA's wide
applicability.

    

### [[2110.12090] Towards Demystifying Intra-Function Parallelism in Serverless Computing](http://arxiv.org/abs/2110.12090)


  Serverless computing offers a pay-per-use model with high elasticity and
automatic scaling for a wide range of applications. Since cloud providers
abstract most of the underlying infrastructure, these services work similarly
to black-boxes. As a result, users can influence the resources allocated to
their functions, but might not be aware that they have to parallelize them to
profit from the additionally allocated virtual CPUs (vCPUs). In this paper, we
analyze the impact of parallelization within a single function and container
instance for AWS Lambda, Google Cloud Functions (GCF), and Google Cloud Run
(GCR). We focus on compute-intensive workloads since they benefit greatly from
parallelization. Furthermore, we investigate the correlation between the number
of allocated CPU cores and vCPUs in serverless environments. Our results show
that the number of available cores to a function/container instance does not
always equal the number of allocated vCPUs. By parallelizing serverless
workloads, we observed cost savings up to 81% for AWS Lambda, 49% for GCF, and
69.8% for GCR.

    

### [[2110.12106] HWTool: Fully Automatic Mapping of an Extensible C++ Image Processing Language to Hardware](http://arxiv.org/abs/2110.12106)


  Implementing image processing algorithms using FPGAs or ASICs can improve
energy efficiency by orders of magnitude over optimized CPU, DSP, or GPU code.
These efficiency improvements are crucial for enabling new applications on
mobile power-constrained devices, such as cell phones or AR/VR headsets.
Unfortunately, custom hardware is commonly implemented using a waterfall
process with time-intensive manual mapping and optimization phases. Thus, it
can take years for a new algorithm to make it all the way from an algorithm
design to shipping silicon. Recent improvements in hardware design tools, such
as C-to-gates High-Level Synthesis (HLS), can reduce design time, but still
require manual tuning from hardware experts.
In this paper, we present HWTool, a novel system for automatically mapping
image processing and computer vision algorithms to hardware. Our system maps
between two domains: HWImg, an extensible C++ image processing library
containing common image processing and parallel computing operators, and
Rigel2, a library of optimized hardware implementations of HWImg's operators
and backend Verilog compiler. We show how to automatically compile HWImg to
Rigel2, by solving for interfaces, hardware sizing, and FIFO buffer allocation.
Finally, we map full-scale image processing applications like convolution,
optical flow, depth from stereo, and feature descriptors to FPGA using our
system. On these examples, HWTool requires on average only 11% more FPGA area
than hand-optimized designs (with manual FIFO allocation), and 33% more FPGA
area than hand-optimized designs with automatic FIFO allocation, and performs
similarly to HLS.

    

### [[2110.12237] Characterizing User and Provider Reported Cloud Failures](http://arxiv.org/abs/2110.12237)


  Cloud computing is the backbone of the digital society. Digital banking,
media, communication, gaming, and many others depend on cloud services.
Unfortunately, cloud services may fail, leading to damaged services, unhappy
users, and perhaps millions of dollars lost for companies. Understanding a
cloud service failure requires a detailed report on why and how the service
failed. Previous work studies how cloud services fail using logs published by
cloud operators. However, information is lacking on how users perceive and
experience cloud failures. Therefore, we collect and characterize the data for
user-reported cloud failures from Down Detector for three cloud service
providers over three years. We count and analyze time patterns in the user
reports, and derive failures from those user reports and characterize their
duration and interarrival time. We characterize provider-reported cloud
failures and compare the results with the characterization of user-reported
failures. The comparison reveals the information of how users perceive failures
and how much of the failures are reported by cloud service providers. Overall,
this work provides a characterization of user- and provider-reported cloud
failures and compares them with each other.

    

### [[2110.12244] Operational Characterization of a Public Scientific Datacenter During and Beyond the COVID-19 Period](http://arxiv.org/abs/2110.12244)


  Datacenters are imperative for the digital society. They offer services such
as computing, telecommunication, media, and entertainment. Datacenters,
however, consume a lot of power. Thus, Improving datacenter operations is
important and may result in better services, reduced energy consumption and
reduced costs. To improve datacenters, we must understand what is going on
inside them. Therefore, we use operational traces from a scientific cluster in
the Netherlands to investigate and understand how that cluster operates. Due to
work-from-home circumstance, the covid period might have changed our daily
usage of online applications, such as zoom and google meet. In this research,
we focus on the operations of a scientific cluster (LISA) inside the SURF
datacenter. The global pandemic might have changed how the LISA cluster
operates. To understand the change, we collect, combine, and analyze
operational logs from the LISA cluster. The tool to collect the data that
belongs to the non-covid period was accomplished in previous research.
Nonetheless, both the tool and instrument to combine and analyze the traces are
lacking. This research focuses on designing an instrument that can combine and
analyze the traces during and before the coronavirus period. The instrument can
also produce graphs for customarily selected rack, nodes and periods. Moreover,
we characterize the traces that belong to the coronavirus period using the
scientific instrument and additional tools. The outcome of this research helps
us understand how the operations for a scientific cluster (LISA) in the
Netherlands has changed after the global pandemic.

    

### [[2005.00942] Alignment-free Genomic Analysis via a Big Data Spark Platform](http://arxiv.org/abs/2005.00942)


  Motivation: Alignment-free distance and similarity functions (AF functions,
for short) are a well established alternative to two and multiple sequence
alignments for many genomic, metagenomic and epigenomic tasks. Due to
data-intensive applications, the computation of AF functions is a Big Data
problem, with the recent Literature indicating that the development of fast and
scalable algorithms computing AF functions is a high-priority task. Somewhat
surprisingly, despite the increasing popularity of Big Data technologies in
Computational Biology, the development of a Big Data platform for those tasks
has not been pursued, possibly due to its complexity. Results: We fill this
important gap by introducing FADE, the first extensible, efficient and scalable
Spark platform for Alignment-free genomic analysis. It supports natively
eighteen of the best performing AF functions coming out of a recent hallmark
benchmarking study. FADE development and potential impact comprises novel
aspects of interest. Namely, (a) a considerable effort of distributed
algorithms, the most tangible result being a much faster execution time of
reference methods like MASH and FSWM; (b) a software design that makes FADE
user-friendly and easily extendable by Spark non-specialists; (c) its ability
to support data- and compute-intensive tasks. About this, we provide a novel
and much needed analysis of how informative and robust AF functions are, in
terms of the statistical significance of their output. Our findings naturally
extend the ones of the highly regarded benchmarking study, since the functions
that can really be used are reduced to a handful of the eighteen included in
FADE.

    

### [[2010.15444] Advanced Python Performance Monitoring with Score-P](http://arxiv.org/abs/2010.15444)


  Within the last years, Python became more prominent in the scientific
community and is now used for simulations, machine learning, and data analysis.
All these tasks profit from additional compute power offered by parallelism and
offloading. In the domain of High Performance Computing (HPC), we can look back
to decades of experience exploiting different levels of parallelism on the
core, node or inter-node level, as well as utilising accelerators. By using
performance analysis tools to investigate all these levels of parallelism, we
can tune applications for unprecedented performance. Unfortunately, standard
Python performance analysis tools cannot cope with highly parallel programs.
Since the development of such software is complex and error-prone, we
demonstrate an easy-to-use solution based on an existing tool infrastructure
for performance analysis. In this paper, we describe how to apply the
established instrumentation framework \scorep to trace Python applications. We
finish with a study of the overhead that users can expect for instrumenting
their applications.

    

### [[2105.06068] On Sparsity Awareness in Distributed Computations](http://arxiv.org/abs/2105.06068)


  We extract a core principle underlying seemingly different fundamental
distributed settings, showing sparsity awareness may induce faster algorithms
for problems in these settings. To leverage this, we establish a new framework
by developing an intermediate auxiliary model weak enough to be simulated in
the CONGEST model given low mixing time, as well as in the recently introduced
HYBRID model. We prove that despite imposing harsh restrictions, this
artificial model allows balancing massive data transfers with high bandwidth
utilization. We exemplify the power of our methods, by deriving shortest-paths
algorithms improving upon the state-of-the-art.
Specifically, we show the following for graphs of $n$ nodes:
A $(3+\epsilon)$ approximation for weighted APSP in
$(n/\delta)\tau_{mix}\cdot 2^{O(\sqrt\log n)}$ rounds in the CONGEST model,
where $\delta$ is the minimum degree of the graph and $\tau_{mix}$ is its
mixing time. For graphs with $\delta=\tau_{mix}\cdot 2^{\omega(\sqrt\log n)}$,
this takes $o(n)$ rounds, despite the $\Omega(n)$ lower bound for general
graphs [Nanongkai, STOC'14].
An $(n^{7/6}/m^{1/2}+n^2/m)\cdot\tau_{mix}\cdot 2^{O(\sqrt\log n)}$-round
exact SSSP algorithm in the CONGNEST model, for graphs with $m$ edges and a
mixing time of $\tau_{mix}$. This improves upon the algorithm of [Chechik and
Mukhtar, PODC'20] for significant ranges of values of $m$ and $ \tau_{mix}$.
A CONGESTED CLIQUE simulation in the CONGEST model improving upon the
state-of-the-art simulation of [Ghaffari, Kuhn, and SU, PODC'17] by a factor
proportional to the average degree in the graph.
An $\tilde O(n^{5/17}/\epsilon^9)$-round algorithm for a $(1+\epsilon)$
approximation for SSSP in the HYBRID model. The only previous $o(n^{1/3})$
round algorithm for distance approximations in this model is for a much larger
factor [Augustine, Hinnenthal, Kuhn, Scheideler, Schneider, SODA'20].

    

### [[2106.15911] A parallel fast multipole method for a space-time boundary element method for the heat equation](http://arxiv.org/abs/2106.15911)


  We present a novel approach to the parallelization of the parabolic fast
multipole method for a space-time boundary element method for the heat
equation. We exploit the special temporal structure of the involved operators
to provide an efficient distributed parallelization with respect to time and
with a one-directional communication pattern. On top, we apply a task-based
shared memory parallelization and SIMD vectorization. In the numerical tests we
observe high efficiencies of our parallelization approach.

    

### [[2110.12053] Towards Dynamic Consistency Checking in Goal-directed Predicate Answer Set Programming](http://arxiv.org/abs/2110.12053)


  Goal-directed evaluation of Answer Set Programs is gaining traction thanks to
its amenability to create AI systems that can, due to the evaluation mechanism
used, generate explanations and justifications. s(CASP) is one of these systems
and has been already used to write reasoning systems in several fields. It
provides enhanced expressiveness w.r.t. other ASP systems due to its ability to
use constraints, data structures, and unbound variables natively. However, the
performance of existing s(CASP) implementations is not on par with other ASP
systems: model consistency is checked once models have been generated, in
keeping with the generate-and-test paradigm. In this work, we present a
variation of the top-down evaluation strategy, termed Dynamic Consistency
Checking, which interleaves model generation and consistency checking. This
makes it possible to determine when a literal is not compatible with the
denials associated to the global constraints in the program, prune the current
execution branch, and choose a different alternative. This strategy is
specially (but not exclusively) relevant in problems with a high combinatorial
component. We have experimentally observed speedups of up to 90x w.r.t. the
standard versions of s(CASP).

    

### [[2110.12058] Development of Semantic Web-based Imaging Database for Biological Morphome](http://arxiv.org/abs/2110.12058)


  We introduce the RIKEN Microstructural Imaging Metadatabase, a semantic
web-based imaging database in which image metadata are described using the
Resource Description Framework (RDF) and detailed biological properties
observed in the images can be represented as Linked Open Data. The metadata are
used to develop a large-scale imaging viewer that provides a straightforward
graphical user interface to visualise a large microstructural tiling image at
the gigabyte level. We applied the database to accumulate comprehensive
microstructural imaging data produced by automated scanning electron
microscopy. As a result, we have successfully managed vast numbers of images
and their metadata, including the interpretation of morphological phenotypes
occurring in sub-cellular components and biosamples captured in the images. We
also discuss advanced utilisation of morphological imaging data that can be
promoted by this database.

    

### [[2110.12201] Spanish Legalese Language Model and Corpora](http://arxiv.org/abs/2110.12201)


  There are many Language Models for the English language according to its
worldwide relevance. However, for the Spanish language, even if it is a widely
spoken language, there are very few Spanish Language Models which result to be
small and too general. Legal slang could be think of a Spanish variant on its
own as it is very complicated in vocabulary, semantics and phrase
understanding. For this work we gathered legal-domain corpora from different
sources, generated a model and evaluated against Spanish general domain tasks.
The model provides reasonable results in those tasks.

    

### [[2110.12320] CoVA: Context-aware Visual Attention for Webpage Information Extraction](http://arxiv.org/abs/2110.12320)


  Webpage information extraction (WIE) is an important step to create knowledge
bases. For this, classical WIE methods leverage the Document Object Model (DOM)
tree of a website. However, use of the DOM tree poses significant challenges as
context and appearance are encoded in an abstract manner. To address this
challenge we propose to reformulate WIE as a context-aware Webpage Object
Detection task. Specifically, we develop a Context-aware Visual Attention-based
(CoVA) detection pipeline which combines appearance features with syntactical
structure from the DOM tree. To study the approach we collect a new large-scale
dataset of e-commerce websites for which we manually annotate every web element
with four labels: product price, product title, product image and background.
On this dataset we show that the proposed CoVA approach is a new challenging
baseline which improves upon prior state-of-the-art methods.

    

### [[2110.12334] SOLVER: Scene-Object Interrelated Visual Emotion Reasoning Network](http://arxiv.org/abs/2110.12334)


  Visual Emotion Analysis (VEA) aims at finding out how people feel emotionally
towards different visual stimuli, which has attracted great attention recently
with the prevalence of sharing images on social networks. Since human emotion
involves a highly complex and abstract cognitive process, it is difficult to
infer visual emotions directly from holistic or regional features in affective
images. It has been demonstrated in psychology that visual emotions are evoked
by the interactions between objects as well as the interactions between objects
and scenes within an image. Inspired by this, we propose a novel Scene-Object
interreLated Visual Emotion Reasoning network (SOLVER) to predict emotions from
images. To mine the emotional relationships between distinct objects, we first
build up an Emotion Graph based on semantic concepts and visual features. Then,
we conduct reasoning on the Emotion Graph using Graph Convolutional Network
(GCN), yielding emotion-enhanced object features. We also design a Scene-Object
Fusion Module to integrate scenes and objects, which exploits scene features to
guide the fusion process of object features with the proposed scene-based
attention mechanism. Extensive experiments and comparisons are conducted on
eight public visual emotion datasets, and the results demonstrate that the
proposed SOLVER consistently outperforms the state-of-the-art methods by a
large margin. Ablation studies verify the effectiveness of our method and
visualizations prove its interpretability, which also bring new insight to
explore the mysteries in VEA. Notably, we further discuss SOLVER on three other
potential datasets with extended experiments, where we validate the robustness
of our method and notice some limitations of it.

    

### [[2110.12335] Chinese Traditional Poetry Generating System Based on Deep Learning](http://arxiv.org/abs/2110.12335)


  Chinese traditional poetry is an important intangible cultural heritage of
China and an artistic carrier of thought, culture, spirit and emotion. However,
due to the strict rules of ancient poetry, it is very difficult to write poetry
by machine. This paper proposes an automatic generation method of Chinese
traditional poetry based on deep learning technology, which extracts keywords
from each poem and matches them with the previous text to make the poem conform
to the theme, and when a user inputs a paragraph of text, the machine obtains
the theme and generates poem sentence by sentence. Using the classic word2vec
model as the preprocessing model, the Chinese characters which are not
understood by the computer are transformed into matrix for processing.
Bi-directional Long Short-Term Memory is used as the neural network model to
generate Chinese characters one by one and make the meaning of Chinese
characters as accurate as possible. At the same time, TF-IDF and TextRank are
used to extract keywords. Using the attention mechanism based encoding-decoding
model, we can solve practical problems by transforming the model, and
strengthen the important information of long-distance information, so as to
grasp the key points without losing important information. In the aspect of
emotion judgment, Long Short-Term Memory network is used. The final result
shows that it can get good poetry outputs according to the user input text.

    

### [[2110.12349] Think about it! Improving defeasible reasoning by first modeling the question scenario](http://arxiv.org/abs/2110.12349)


  Defeasible reasoning is the mode of reasoning where conclusions can be
overturned by taking into account new evidence. Existing cognitive science
literature on defeasible reasoning suggests that a person forms a mental model
of the problem scenario before answering questions. Our research goal asks
whether neural models can similarly benefit from envisioning the question
scenario before answering a defeasible query. Our approach is, given a
question, to have a model first create a graph of relevant influences, and then
leverage that graph as an additional input when answering the question. Our
system, CURIOUS, achieves a new state-of-the-art on three different defeasible
reasoning datasets. This result is significant as it illustrates that
performance can be improved by guiding a system to "think about" a question and
explicitly model the scenario, rather than answering reflexively. Code, data,
and pre-trained models are located at this https URL.

    

### [[2110.12352] DiffSRL: Learning Dynamic-aware State Representation for Deformable Object Control with Differentiable Simulator](http://arxiv.org/abs/2110.12352)


  Dynamic state representation learning is an important task in robot learning.
Latent space that can capture dynamics related information has wide application
in areas such as accelerating model free reinforcement learning, closing the
simulation to reality gap, as well as reducing the motion planning complexity.
However, current dynamic state representation learning methods scale poorly on
complex dynamic systems such as deformable objects, and cannot directly embed
well defined simulation function into the training pipeline. We propose
DiffSRL, a dynamic state representation learning pipeline utilizing
differentiable simulation that can embed complex dynamics models as part of the
end-to-end training. We also integrate differentiable dynamic constraints as
part of the pipeline which provide incentives for the latent state to be aware
of dynamical constraints. We further establish a state representation learning
benchmark on a soft-body simulation system, PlasticineLab, and our model
demonstrates superior performance in terms of capturing long-term dynamics as
well as reward prediction.

    

### [[2110.12371] Neural Embeddings of Urban Big Data Reveal Emergent Structures in Cities](http://arxiv.org/abs/2110.12371)


  In this study, we propose using a neural embedding model-graph neural network
(GNN)- that leverages the heterogeneous features of urban areas and their
interactions captured by human mobility network to obtain vector
representations of these areas. Using large-scale high-resolution mobility data
sets from millions of aggregated and anonymized mobile phone users in 16
metropolitan counties in the United States, we demonstrate that our embeddings
encode complex relationships among features related to urban components (such
as distribution of facilities) and population attributes and activities. The
spatial gradient in each direction from city center to suburbs is measured
using clustered representations and the shared characteristics among urban
areas in the same cluster. Furthermore, we show that embeddings generated by a
model trained on a different county can capture 50% to 60% of the emergent
spatial structure in another county, allowing us to make cross-county
comparisons in a quantitative way. Our GNN-based framework overcomes the
limitations of previous methods used for examining spatial structures and is
highly scalable. The findings reveal non-linear relationships among urban
components and anisotropic spatial gradients in cities. Since the identified
spatial structures and gradients capture the combined effects of various
mechanisms, such as segregation, disparate facility distribution, and human
mobility, the findings could help identify the limitations of the current city
structure to inform planning decisions and policies. Also, the model and
findings set the stage for a variety of research in urban planning, engineering
and social science through integrated understanding of how the complex
interactions between urban components and population activities and attributes
shape the spatial structures in cities.

    

### [[2110.12442] Bangla Image Caption Generation through CNN-Transformer based Encoder-Decoder Network](http://arxiv.org/abs/2110.12442)


  Automatic Image Captioning is the never-ending effort of creating
syntactically and validating the accuracy of textual descriptions of an image
in natural language with context. The encoder-decoder structure used throughout
existing Bengali Image Captioning (BIC) research utilized abstract image
feature vectors as the encoder's input. We propose a novel transformer-based
architecture with an attention mechanism with a pre-trained ResNet-101 model
image encoder for feature extraction from images. Experiments demonstrate that
the language decoder in our technique captures fine-grained information in the
caption and, then paired with image features, produces accurate and diverse
captions on the BanglaLekhaImageCaptions dataset. Our approach outperforms all
existing Bengali Image Captioning work and sets a new benchmark by scoring
0.694 on BLEU-1, 0.630 on BLEU-2, 0.582 on BLEU-3, and 0.337 on METEOR.

    

### [[1912.00747] The Transformative Potential of Artificial Intelligence](http://arxiv.org/abs/1912.00747)


  The terms 'human-level artificial intelligence' and 'artificial general
intelligence' are widely used to refer to the possibility of advanced
artificial intelligence (AI) with potentially extreme impacts on society. These
terms are poorly defined and do not necessarily indicate what is most important
with respect to future societal impacts. We suggest that the term
'transformative AI' is a helpful alternative, reflecting the possibility that
advanced AI systems could have very large impacts on society without reaching
human-level cognitive abilities. To be most useful, however, more analysis of
what it means for AI to be 'transformative' is needed. In this paper, we
propose three different levels on which AI might be said to be transformative,
associated with different levels of societal change. We suggest that these
distinctions would improve conversations between policy makers and decision
makers concerning the mid- to long-term impacts of advances in AI. Further, we
feel this would have a positive effect on strategic foresight efforts involving
advanced AI, which we expect to illuminate paths to alternative futures. We
conclude with a discussion of the benefits of our new framework and by
highlighting directions for future work in this area.

    

### [[1912.01683] Optimal Policies Tend to Seek Power](http://arxiv.org/abs/1912.01683)


  Some researchers speculate that intelligent reinforcement learning (RL)
agents would be incentivized to seek resources and power in pursuit of their
objectives. Other researchers are skeptical, because RL agents need not have
human-like power-seeking instincts. To clarify this debate, we develop the
first formal theory of the statistical tendencies of optimal policies. In the
context of Markov decision processes, we prove that certain environmental
symmetries are sufficient for optimal policies to tend to seek power over the
environment. These symmetries exist in many environments in which the agent can
be shut down or destroyed. We prove that in these environments, most reward
functions make it optimal to seek power by keeping a range of options available
and, when maximizing average reward, by navigating towards larger sets of
potential terminal states.

    

### [[2012.01244] General Characterization of Agents by States they Visit](http://arxiv.org/abs/2012.01244)


  Behavioural characterizations (BCs) of decision-making agents, or their
policies, are used to study outcomes of training algorithms and as part of the
algorithms themselves to encourage unique policies, match expert policy or
restrict changes to policy per update. However, previously presented solutions
are not applicable in general, either due to lack of expressive power,
computational constraint or constraints on the policy or environment.
Furthermore, many BCs rely on the actions of policies. We discuss and
demonstrate how these BCs can be misleading, especially in stochastic
environments, and propose a novel solution based on what states policies visit.
We run experiments to evaluate the quality of the proposed BC against baselines
and evaluate their use in studying training algorithms, novelty search and
trust-region policy optimization. The code is available at
this https URL.

    

### [[2103.09382] SPICE: Semantic Pseudo-labeling for Image Clustering](http://arxiv.org/abs/2103.09382)


  The similarity among samples and the discrepancy between clusters are two
crucial aspects of image clustering. However, current deep clustering methods
suffer from the inaccurate estimation of either feature similarity or semantic
discrepancy. In this paper, we present a Semantic Pseudo-labeling-based Image
ClustEring (SPICE) framework, which divides the clustering network into a
feature model for measuring the instance-level similarity and a clustering head
for identifying the cluster-level discrepancy. We design two semantics-aware
pseudo-labeling algorithms, prototype pseudo-labeling, and reliable
pseudo-labeling, which enable accurate and reliable self-supervision over
clustering. Without using any ground-truth label, we optimize the clustering
network in three stages: 1) train the feature model through contrastive
learning to measure the instance similarity, 2) train the clustering head with
the prototype pseudo-labeling algorithm to identify cluster semantics, and 3)
jointly train the feature model and clustering head with the reliable
pseudo-labeling algorithm to improve the clustering performance. Extensive
experimental results demonstrate that SPICE achieves significant improvements
(~10%) over existing methods and establishes the new state-of-the-art
clustering results on six image benchmark datasets in terms of three popular
metrics. Importantly, SPICE significantly reduces the gap between unsupervised
and fully-supervised classification; e.g., there is only a 2% (91.8% vs 93.8%)
accuracy difference on CIFAR-10. Our code has been made publically available at
this https URL.

    

### [[2103.14930] Hyperbolic Geometry is Not Necessary: Lightweight Euclidean-Based Models for Low-Dimensional Knowledge Graph Embeddings](http://arxiv.org/abs/2103.14930)


  Recent knowledge graph embedding (KGE) models based on hyperbolic geometry
have shown great potential in a low-dimensional embedding space. However, the
necessity of hyperbolic space in KGE is still questionable, because the
calculation based on hyperbolic geometry is much more complicated than
Euclidean operations. In this paper, based on the state-of-the-art
hyperbolic-based model RotH, we develop two lightweight Euclidean-based models,
called RotL and Rot2L. The RotL model simplifies the hyperbolic operations
while keeping the flexible normalization effect. Utilizing a novel two-layer
stacked transformation and based on RotL, the Rot2L model obtains an improved
representation capability, yet costs fewer parameters and calculations than
RotH. The experiments on link prediction show that Rot2L achieves the
state-of-the-art performance on two widely-used datasets in low-dimensional
knowledge graph embeddings. Furthermore, RotL achieves similar performance as
RotH but only requires half of the training time.

    

### [[2104.07225] Text Guide: Improving the quality of long text classification by a text selection method based on feature importance](http://arxiv.org/abs/2104.07225)


  The performance of text classification methods has improved greatly over the
last decade for text instances of less than 512 tokens. This limit has been
adopted by most state-of-the-research transformer models due to the high
computational cost of analyzing longer text instances. To mitigate this problem
and to improve classification for longer texts, researchers have sought to
resolve the underlying causes of the computational cost and have proposed
optimizations for the attention mechanism, which is the key element of every
transformer model. In our study, we are not pursuing the ultimate goal of long
text classification, i.e., the ability to analyze entire text instances at one
time while preserving high performance at a reasonable computational cost.
Instead, we propose a text truncation method called Text Guide, in which the
original text length is reduced to a predefined limit in a manner that improves
performance over naive and semi-naive approaches while preserving low
computational costs. Text Guide benefits from the concept of feature
importance, a notion from the explainable artificial intelligence domain. We
demonstrate that Text Guide can be used to improve the performance of recent
language models specifically designed for long text classification, such as
Longformer. Moreover, we discovered that parameter optimization is the key to
Text Guide performance and must be conducted before the method is deployed.
Future experiments may reveal additional benefits provided by this new method.

    

### [[2105.09605] Learning Robust Recommenders through Cross-Model Agreement](http://arxiv.org/abs/2105.09605)


  Learning from implicit feedback is one of the most common cases in the
application of recommender systems. Generally speaking, interacted examples are
considered as positive while negative examples are sampled from uninteracted
ones. However, noisy examples are prevalent in real-world implicit feedback. A
noisy positive example could be interacted but it actually leads to negative
user preference. A noisy negative example which is uninteracted because of
unawareness of the user could also denote potential positive user preference.
Conventional training methods overlook these noisy examples, leading to
sub-optimal recommendations. In this work, we propose a novel framework to
learn robust recommenders from implicit feedback. Through an empirical study,
we find that different models make relatively similar predictions on clean
examples which denote the real user preference, while the predictions on noisy
examples vary much more across different models. Motivated by this observation,
we propose denoising with cross-model agreement(DeCA) which aims to minimize
the KL-divergence between the real user preference distributions parameterized
by two recommendation models while maximizing the likelihood of data
observation. We employ the proposed DeCA on four state-of-the-art
recommendation models and conduct experiments on four datasets. Experimental
results demonstrate that DeCA significantly improves recommendation performance
compared with normal training and other denoising methods. Codes will be
open-sourced.

    

### [[2106.04533] Chasing Sparsity in Vision Transformers: An End-to-End Exploration](http://arxiv.org/abs/2106.04533)


  Vision transformers (ViTs) have recently received explosive popularity, but
their enormous model sizes and training costs remain daunting. Conventional
post-training pruning often incurs higher training budgets. In contrast, this
paper aims to trim down both the training memory overhead and the inference
complexity, without sacrificing the achievable accuracy. We carry out the
first-of-its-kind comprehensive exploration, on taking a unified approach of
integrating sparsity in ViTs "from end to end". Specifically, instead of
training full ViTs, we dynamically extract and train sparse subnetworks, while
sticking to a fixed small parameter budget. Our approach jointly optimizes
model parameters and explores connectivity throughout training, ending up with
one sparse network as the final output. The approach is seamlessly extended
from unstructured to structured sparsity, the latter by considering to guide
the prune-and-grow of self-attention heads inside ViTs. We further co-explore
data and architecture sparsity for additional efficiency gains by plugging in a
novel learnable token selector to adaptively determine the currently most vital
patches. Extensive results on ImageNet with diverse ViT backbones validate the
effectiveness of our proposals which obtain significantly reduced computational
cost and almost unimpaired generalization. Perhaps most surprisingly, we find
that the proposed sparse (co-)training can sometimes improve the ViT accuracy
rather than compromising it, making sparsity a tantalizing "free lunch". For
example, our sparsified DeiT-Small at (5%, 50%) sparsity for (data,
architecture), improves 0.28% top-1 accuracy, and meanwhile enjoys 49.32% FLOPs
and 4.40% running time savings. Our codes are available at
this https URL.

    

### [[2109.06123] Knowledge Graph-based Neurodegenerative Diseases and Diet Relationship Discovery](http://arxiv.org/abs/2109.06123)


  To date, there are no effective treatments for most neurodegenerative
diseases. However, certain foods may be associated with these diseases and
bring an opportunity to prevent or delay neurodegenerative progression. Our
objective is to construct a knowledge graph for neurodegenerative diseases
using literature mining to study their relations with diet. We collected
biomedical annotations (Disease, Chemical, Gene, Species, SNP&Mutation) in the
abstracts from 4,300 publications relevant to both neurodegenerative diseases
and diet using PubTator, an NIH-supported tool that can extract biomedical
concepts from literature. A knowledge graph was created from these annotations.
Graph embeddings were then trained with the node2vec algorithm to support
potential concept clustering and similar concept identification. We found
several food-related species and chemicals that might come from diet and have
an impact on neurodegenerative diseases.

    

### [[2109.12907] Expressing High-Level Scientific Claims with Formal Semantics](http://arxiv.org/abs/2109.12907)


  The use of semantic technologies is gaining significant traction in science
communication with a wide array of applications in disciplines including the
Life Sciences, Computer Science, and the Social Sciences. Languages like RDF,
OWL, and other formalisms based on formal logic are applied to make scientific
knowledge accessible not only to human readers but also to automated systems.
These approaches have mostly focused on the structure of scientific
publications themselves, on the used scientific methods and equipment, or on
the structure of the used datasets. The core claims or hypotheses of scientific
work have only been covered in a shallow manner, such as by linking mentioned
entities to established identifiers. In this research, we therefore want to
find out whether we can use existing semantic formalisms to fully express the
content of high-level scientific claims using formal semantics in a systematic
way. Analyzing the main claims from a sample of scientific articles from all
disciplines, we find that their semantics are more complex than what a
straight-forward application of formalisms like RDF or OWL account for, but we
managed to elicit a clear semantic pattern which we call the 'super-pattern'.
We show here how the instantiation of the five slots of this super-pattern
leads to a strictly defined statement in higher-order logic. We successfully
applied this super-pattern to an enlarged sample of scientific claims. We show
that knowledge representation experts, when instructed to independently
instantiate the super-pattern with given scientific claims, show a high degree
of consistency and convergence given the complexity of the task and the
subject. These results therefore open the door for expressing high-level
scientific findings in a manner they can be automatically interpreted, which on
the longer run can allow us to do automated consistency checking, and much
more.

    

### [[2110.05327] Compositionality as we see it, everywhere around us](http://arxiv.org/abs/2110.05327)


  There are different meanings of the term "compositionality" within science:
what one researcher would call compositional, is not at all compositional for
another researcher. The most established conception is usually attributed to
Frege, and is characterised by a bottom-up flow of meanings: the meaning of the
whole can be derived from the meanings of the parts, and how these parts are
structured together.
Inspired by work on compositionality in quantum theory, and categorical
quantum mechanics in particular, we propose the notions of Schrodinger,
Whitehead, and complete compositionality. Accounting for recent important
developments in quantum technology and artificial intelligence, these do not
have the bottom-up meaning flow as part of their definitions.
Schrodinger compositionality accommodates quantum theory, and also
meaning-as-context. Complete compositionality further strengthens Schrodinger
compositionality in order to single out theories like ZX-calculus, that are
complete with regard to the intended model. All together, our new notions aim
to capture the fact that compositionality is at its best when it is `real',
`non-trivial', and even more when it also is `complete'.
At this point we only put forward the intuitive and/or restricted formal
definitions, and leave a fully comprehensive definition to future collaborative
work.

    

### [[2110.08975] Deep Transfer Learning & Beyond: Transformer Language Models in Information Systems Research](http://arxiv.org/abs/2110.08975)


  AI is widely thought to be poised to transform business, yet current
perceptions of the scope of this transformation may be myopic. Recent progress
in natural language processing involving transformer language models (TLMs)
offers a potential avenue for AI-driven business and societal transformation
that is beyond the scope of what most currently foresee. We review this recent
progress as well as recent literature utilizing text mining in top IS journals
to develop an outline for how future IS research can benefit from these new
techniques. Our review of existing IS literature reveals that suboptimal text
mining techniques are prevalent and that the more advanced TLMs could be
applied to enhance and increase IS research involving text data, and to enable
new IS research topics, thus creating more value for the research community.
This is possible because these techniques make it easier to develop very
powerful custom systems and their performance is superior to existing methods
for a wide range of tasks and applications. Further, multilingual language
models make possible higher quality text analytics for research in multiple
languages. We also identify new avenues for IS research, like language user
interfaces, that may offer even greater potential for future IS research.

    

### [[2110.09397] Using Psychological Characteristics of Situations for Social Situation Comprehension in Support Agents](http://arxiv.org/abs/2110.09397)


  Support agents that help users in their daily lives need to take into account
not only the user's characteristics, but also the social situation of the user.
Existing work on including social context uses some type of situation cue as an
input to information processing techniques in order to assess the expected
behavior of the user. However, research shows that it is important to also
determine the meaning of a situation, a step which we refer to as social
situation comprehension. We propose using psychological characteristics of
situations, which have been proposed in social science for ascribing meaning to
situations, as the basis for social situation comprehension. Using data from
user studies, we evaluate this proposal from two perspectives. First, from a
technical perspective, we show that psychological characteristics of situations
can be used as input to predict the priority of social situations, and that
psychological characteristics of situations can be predicted from the features
of a social situation. Second, we investigate the role of the comprehension
step in human-machine meaning making. We show that psychological
characteristics can be successfully used as a basis for explanations given to
users about the decisions of an agenda management personal assistant agent.

    

### [[1805.08359] DRESS: Dynamic RESource-reservation Scheme for Congested Data-intensive Computing Platforms](http://arxiv.org/abs/1805.08359)


  In the past few years, we have envisioned an increasing number of businesses
start driving by big data analytics, such as Amazon recommendations and Google
Advertisements. At the back-end side, the businesses are powered by big data
processing platforms to quickly extract information and make decisions. Running
on top of a computing cluster, those platforms utilize scheduling algorithms to
allocate resources. An efficient scheduler is crucial to the system performance
due to limited resources, e.g. CPU and Memory, and a large number of user
demands. However, besides requests from clients and current status of the
system, it has limited knowledge about execution length of the running jobs,
and incoming jobs' resource demands, which make assigning resources a
challenging task. If most of the resources are occupied by a long-running job,
other jobs will have to keep waiting until it releases them. This paper
presents a new scheduling strategy, named DRESS that particularly aims to
optimize the allocation among jobs with various demands. Specifically, it
classifies the jobs into two categories based on their requests, reserves a
portion of resources for each of category, and dynamically adjusts the reserved
ratio by monitoring the pending requests and estimating release patterns of
running jobs. The results demonstrate DRESS significantly reduces the
completion time for one category, up to 76.1% in our experiments, and in the
meanwhile, maintains a stable overall system performance.

    

### [[2104.05348] Quotients of Bounded Natural Functors](http://arxiv.org/abs/2104.05348)


  The functorial structure of type constructors is the foundation for many
definition and proof principles in higher-order logic (HOL). For example,
inductive and coinductive datatypes can be built modularly from bounded natural
functors (BNFs), a class of well-behaved type constructors. Composition,
fixpoints, and, under certain conditions, subtypes are known to preserve the
BNF structure. In this article, we tackle the preservation question for
quotients, the last important principle for introducing new types in HOL. We
identify sufficient conditions under which a quotient inherits the BNF
structure from its underlying type. Surprisingly, lifting the structure in the
obvious manner fails for some quotients, a problem that also affects the
quotients of polynomial functors used in the Lean proof assistant. We provide a
strictly more general lifting scheme that supports such problematic quotients.
We extend the Isabelle/HOL proof assistant with a command that automates the
registration of a quotient type as a BNF, reducing the proof burden on the user
from the full set of BNF axioms to our inheritance conditions. We demonstrate
the command's usefulness through several case studies.

    

### [<title data-react-helmet="true">NeurIPS 2021 | 图上不均衡表示学习新视野：基于拓扑结构的不均衡学习 - 知乎</title>](https://zhuanlan.zhihu.com/p/425818142)