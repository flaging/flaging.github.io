
## 2021-8-24

### [[2108.09343] Early-exit deep neural networks for distorted images: providing an efficient edge offloading](http://arxiv.org/abs/2108.09343)


  Edge offloading for deep neural networks (DNNs) can be adaptive to the
input's complexity by using early-exit DNNs. These DNNs have side branches
throughout their architecture, allowing the inference to end earlier in the
edge. The branches estimate the accuracy for a given input. If this estimated
accuracy reaches a threshold, the inference ends on the edge. Otherwise, the
edge offloads the inference to the cloud to process the remaining DNN layers.
However, DNNs for image classification deals with distorted images, which
negatively impact the branches' estimated accuracy. Consequently, the edge
offloads more inferences to the cloud. This work introduces expert side
branches trained on a particular distortion type to improve robustness against
image distortion. The edge detects the distortion type and selects appropriate
expert branches to perform the inference. This approach increases the estimated
accuracy on the edge, improving the offloading decisions. We validate our
proposal in a realistic scenario, in which the edge offloads DNN inference to
Amazon EC2 instances.

    

### [[2108.09358] Crown Jewels Analysis using Reinforcement Learning with Attack Graphs](http://arxiv.org/abs/2108.09358)


  Cyber attacks pose existential threats to nations and enterprises. Current
practice favors piece-wise analysis using threat-models in the stead of
rigorous cyber terrain analysis and intelligence preparation of the
battlefield. Automated penetration testing using reinforcement learning offers
a new and promising approach for developing methodologies that are driven by
network structure and cyber terrain, that can be later interpreted in terms of
threat-models, but that are principally network-driven analyses. This paper
presents a novel method for crown jewel analysis termed CJA-RL that uses
reinforcement learning to identify key terrain and avenues of approach for
exploiting crown jewels. In our experiment, CJA-RL identified ideal entry
points, choke points, and pivots for exploiting a network with multiple crown
jewels, exemplifying how CJA-RL and reinforcement learning for penetration
testing generally can benefit computer network operations workflows.

    

### [[2108.09509] Hop-by-hop Accounting and Rewards for Packet dIspAtching](http://arxiv.org/abs/2108.09509)


  Community networks are prone to free-riders, i.e., participants who take
advantage of cooperation from others' routers but do not contribute
reciprocally. In this paper, we present HARPIA, a system for credit-based
incentive mechanisms for data forwarding in community networks aimed to prevent
selfish behavior. HARPIA does not require a trusted third-party or
tamper-resistant security modules as in other incentive mechanisms. Instead, it
uses a distributed accounting scheme (DPIFA) to estimate the balance of data
forwarding contribution and consumption of each network router and settle
correspondent cryptocurrency debts on an Ethereum smart contract. On-chain
settlement transactions are performed every HARPIA cycle (e.g., daily, weekly,
monthly) and must be validated by at least m-of-n network routers using a
multi-signature scheme (MuSig). We also realized a performance evaluation,
security threat assessment, and cryptocurrency costs estimation. Results show
that our proposal is suitable for community networks with up to 64
infrastructure routers under specific m-of-n MuSig thresholds.

    

### [[2108.09555] Reliable Firmware Updates for the Information-Centric Internet of Things](http://arxiv.org/abs/2108.09555)


  Security in the Internet of Things (IoT) requires ways to regularly update
firmware in the field. These demands ever increase with new, agile concepts
such as security as code and should be considered a regular operation. Hosting
massive firmware roll-outs present a crucial challenge for the constrained
wireless environment. In this paper, we explore how information-centric
networking can ease reliable firmware updates. We start from the recent
standards developed by the IETF SUIT working group and contribute a system that
allows for a timely discovery of new firmware versions by using
cryptographically protected manifest files. Our design enables a cascading
firmware roll-out from a gateway towards leaf nodes in a low-power multi-hop
network. While a chunking mechanism prepares firmware images for typically
low-sized maximum transmission units (MTUs), an early Denial-of-Service (DoS)
detection prevents the distribution of tampered or malformed chunks. In
experimental evaluations on a real-world IoT testbed, we demonstrate feasible
strategies with adaptive bandwidth consumption and a high resilience to
connectivity loss when replicating firmware images into the IoT edge.

    

### [[2108.09569] Wireless Sensor Networks for Optimisation of Search and Rescue Management in Floods](http://arxiv.org/abs/2108.09569)


  We propose a novel search-and-rescue management method that relies on the
aerial deployment of Wireless Sensor Network (WSN) for locating victims after
floods. The sensor nodes will collect vital information such as heat signatures
for detecting human presence and location, the flow of flood. The sensor
modules are packed in a portable floating buoy with a user interface to convey
emergency messages to the base station. Sensor nodes are designed based on
disaster conditions, cost-effectiveness and deployed in the affected region by
a centrifugal dispersion system from a helicopter.
A mobile ad-hoc network is set up by modifying the Low Energy Adaptive
Cluster Hierarchy (LEACH) protocol for greater efficiency and adoption of
multi-hop of Cluster Heads for long-distance communication to Base Station. The
model metrics have been defined considering previous rural floods in India. The
efficiency and power characteristics of the network are compared to other
protocols via simulations. The sensor data from the network makes resource
management, rescue planning and emergency priority more efficient, thus saving
more lives from floods.

    

### [[2108.09728] Joint Link Rate Selection and Channel State Change Detection in Block-Fading Channels](http://arxiv.org/abs/2108.09728)


  In this work, we consider the problem of transmission rate selection for a
discrete time point-to-point block fading wireless communication link. The
wireless channel remains constant within the channel coherence time but can
change rapidly across blocks. The goal is to design a link rate selection
strategy that can identify the best transmission rate quickly and adaptively in
quasi-static channels. This problem can be cast into the stochastic bandit
framework, and the unawareness of time-stamps where channel changes
necessitates running change-point detection simultaneously with stochastic
bandit algorithms to improve adaptivity. We present a joint channel
change-point detection and link rate selection algorithm based on Thompson
Sampling (CD-TS) and show it can achieve a sublinear regret with respect to the
number of time steps $T$ when the channel coherence time is larger than a
threshold. We then improve the CD-TS algorithm by considering the fact that
higher transmission rate has higher packet-loss probability. Finally, we
validate the performance of the proposed algorithms through numerical
simulations.

    

### [[2108.09864] A Round-Robin Packet Scheduler for Hierarchical Max-Min Fairness](http://arxiv.org/abs/2108.09864)


  Hierarchical link sharing addresses the demand for fine-grain traffic control
at multiple levels of aggregation. At present, packet schedulers that can
support hierarchical link sharing are not suitable for an implementation at
line rates, and deployed schedulers perform poorly when distributing excess
capacity to classes that need additional bandwidth. We present HLS, a packet
scheduler that ensures a hierarchical max-min fair allocation of the link
bandwidth. HLS supports minimum rate guarantees and isolation between classes.
Since it is realized as a non-hierarchical round robin scheduler, it is
suitable to operate at high rates. We implement HLS in the Linux kernel and
evaluate it with respect to achieved rate allocations and overhead. We compare
the results with those obtained for CBQ and HTB, the existing scheduling
algorithms in Linux for hierarchical link sharing. We show that the overhead of
HLS is comparable to that of other classful packet schedulers.

    

### [[2108.10056] Composite Time-Frequency Analysis and Siamese Neural Network based Compound Interference Identification for Hopping Frequency System](http://arxiv.org/abs/2108.10056)


  In a hostile environment, interference identification plays an important role
in protecting the authorized communication system and avoiding its performance
degradation. In this paper, the interference identification problem for the
frequency hopping communication system is discussed. Considering presence of
multiple and compound interference in the frequency hopping system, in order to
fully extracted effective features of the interferences from the received
signals, a composite time-frequency analysis method based on both the linear
and bilinear transform is proposed. The time-frequency spectrograms obtained
from the time-frequency analysis are constructed as matching pairs and input
into the deep neural network for identification. In particular, the Siamese
neural network is adopted as the classifier to perform the interference
identification. That is, the paired spectrograms are input into the two
sub-networks of the Siamese neural network to extract the features of the
paired spectrograms. The Siamese neural network is trained and tested by
calculating the gap between the generated features, and the interference type
identification is realized by the trained Siamese neural network. The
simulation results confirm that the proposed algorithm can obtain higher
identification accuracy than both traditional single time-frequency
representation based approach and the AlexNet transfer learning or
convolutional neural network based methods.

    

### [[2105.13500] An Analysis of Amazon Echo's Network Behavior](http://arxiv.org/abs/2105.13500)


  With over 20 million units sold since 2015, Amazon Echo, the Alexa-enabled
smart speaker developed by Amazon, is probably one of the most widely deployed
Internet of Things consumer devices. Despite the very large installed base,
surprisingly little is known about the device's network behavior. We modify a
first generation Echo device, decrypt its communication with Amazon cloud, and
analyze the device pairing, Alexa Voice Service, and drop-in calling protocols.
We also describe our methodology and the experimental setup. We find a minor
shortcoming in the device pairing protocol and learn that drop-in calls are
end-to-end encrypted and based on modern open standards. Overall, we find the
Echo to be a well-designed device from the network communication perspective.

    

### [[2108.09305] Data-driven Smart Ponzi Scheme Detection](http://arxiv.org/abs/2108.09305)


  A smart Ponzi scheme is a new form of economic crime that uses Ethereum smart
contract account and cryptocurrency to implement Ponzi scheme. The smart Ponzi
scheme has harmed the interests of many investors, but researches on smart
Ponzi scheme detection is still very limited. The existing smart Ponzi scheme
detection methods have the problems of requiring many human resources in
feature engineering and poor model portability. To solve these problems, we
propose a data-driven smart Ponzi scheme detection system in this paper. The
system uses dynamic graph embedding technology to automatically learn the
representation of an account based on multi-source and multi-modal data related
to account transactions. Compared with traditional methods, the proposed system
requires very limited human-computer interaction. To the best of our knowledge,
this is the first work to implement smart Ponzi scheme detection through
dynamic graph embedding. Experimental results show that this method is
significantly better than the existing smart Ponzi scheme detection methods.

    

### [[2108.09306] D-DARTS: Distributed Differentiable Architecture Search](http://arxiv.org/abs/2108.09306)


  Differentiable ARchiTecture Search (DARTS) is one of the most trending Neural
Architecture Search (NAS) methods, drastically reducing search cost by
resorting to Stochastic Gradient Descent (SGD) and weight-sharing. However, it
also greatly reduces the search space, thus excluding potential promising
architectures from being discovered. In this paper, we propose D-DARTS, a novel
solution that addresses this problem by nesting several neural networks at
cell-level instead of using weight-sharing to produce more diversified and
specialized architectures. Moreover, we introduce a novel algorithm which can
derive deeper architectures from a few trained cells, increasing performance
and saving computation time. Our solution is able to provide state-of-the-art
results on CIFAR-10, CIFAR-100 and ImageNet while using significantly less
parameters than previous baselines, resulting in more hardware-efficient neural
networks.

    

### [[2108.09331] Influence Selection for Active Learning](http://arxiv.org/abs/2108.09331)


  The existing active learning methods select the samples by evaluating the
sample's uncertainty or its effect on the diversity of labeled datasets based
on different task-specific or model-specific criteria. In this paper, we
propose the Influence Selection for Active Learning(ISAL) which selects the
unlabeled samples that can provide the most positive Influence on model
performance. To obtain the Influence of the unlabeled sample in the active
learning scenario, we design the Untrained Unlabeled sample Influence
Calculation(UUIC) to estimate the unlabeled sample's expected gradient with
which we calculate its Influence. To prove the effectiveness of UUIC, we
provide both theoretical and experimental analyses. Since the UUIC just depends
on the model gradients, which can be obtained easily from any neural network,
our active learning algorithm is task-agnostic and model-agnostic. ISAL
achieves state-of-the-art performance in different active learning settings for
different tasks with different datasets. Compared with previous methods, our
method decreases the annotation cost at least by 12%, 13% and 16% on CIFAR10,
VOC2012 and COCO, respectively.

    

### [[2108.09335] LoOp: Looking for Optimal Hard Negative Embeddings for Deep Metric Learning](http://arxiv.org/abs/2108.09335)


  Deep metric learning has been effectively used to learn distance metrics for
different visual tasks like image retrieval, clustering, etc. In order to aid
the training process, existing methods either use a hard mining strategy to
extract the most informative samples or seek to generate hard synthetics using
an additional network. Such approaches face different challenges and can lead
to biased embeddings in the former case, and (i) harder optimization (ii)
slower training speed (iii) higher model complexity in the latter case. In
order to overcome these challenges, we propose a novel approach that looks for
optimal hard negatives (LoOp) in the embedding space, taking full advantage of
each tuple by calculating the minimum distance between a pair of positives and
a pair of negatives. Unlike mining-based methods, our approach considers the
entire space between pairs of embeddings to calculate the optimal hard
negatives. Extensive experiments combining our approach and representative
metric learning losses reveal a significant boost in performance on three
benchmark datasets.

    

### [[2108.09373] Understanding and Co-designing the Data Ingestion Pipeline for Industry-Scale RecSys Training](http://arxiv.org/abs/2108.09373)


  The data ingestion pipeline, responsible for storing and preprocessing
training data, is an important component of any machine learning training job.
At Facebook, we use recommendation models extensively across our services. The
data ingestion requirements to train these models are substantial. In this
paper, we present an extensive characterization of the data ingestion
challenges for industry-scale recommendation model training. First, dataset
storage requirements are massive and variable; exceeding local storage
capacities. Secondly, reading and preprocessing data is computationally
expensive, requiring substantially more compute, memory, and network resources
than are available on trainers themselves. These demands result in drastically
reduced training throughput, and thus wasted GPU resources, when current
on-trainer preprocessing solutions are used. To address these challenges, we
present a disaggregated data ingestion pipeline. It includes a central data
warehouse built on distributed storage nodes. We introduce Data PreProcessing
Service (DPP), a fully disaggregated preprocessing service that scales to
hundreds of nodes, eliminating data stalls that can reduce training throughput
by 56%. We implement important optimizations across storage and DPP, increasing
storage and preprocessing throughput by 1.9x and 2.3x, respectively, addressing
the substantial power requirements of data ingestion. We close with lessons
learned and cover the important remaining challenges and opportunities
surrounding data ingestion at scale.

    

### [[2108.09375] Cascade Watchdog: A Multi-tiered Adversarial Guard for Outlier Detection](http://arxiv.org/abs/2108.09375)


  The identification of out-of-distribution content is critical to the
successful implementation of neural networks. Watchdog techniques have been
developed to support the detection of these inputs, but the performance can be
limited by the amount of available data. Generative adversarial networks have
displayed numerous capabilities, including the ability to generate facsimiles
with excellent accuracy. This paper presents and empirically evaluates a
multi-tiered watchdog, which is developed using GAN generated data, for
improved out-of-distribution detection. The cascade watchdog uses adversarial
training to increase the amount of available data similar to the
out-of-distribution elements that are more difficult to detect. Then, a
specialized second guard is added in sequential order. The results show a solid
and significant improvement on the detection of the most challenging
out-of-distribution inputs while preserving an extremely low false positive
rate.

    

### [[2108.09402] A Multi-Task Learning Framework for COVID-19 Monitoring and Prediction of PPE Demand in Community Health Centres](http://arxiv.org/abs/2108.09402)


  Currently, the world seeks to find appropriate mitigation techniques to
control and prevent the spread of the new SARS-CoV-2. In our paper herein, we
present a peculiar Multi-Task Learning framework that jointly predicts the
effect of SARS-CoV-2 as well as Personal-Protective-Equipment consumption in
Community Health Centres for a given populace. Predicting the effect of the
virus (SARS-CoV-2), via studies and analyses, enables us to understand the
nature of SARS-CoV- 2 with reference to factors that promote its growth and
spread. Therefore, these foster widespread awareness; and the populace can
become more proactive and cautious so as to mitigate the spread of Corona Virus
Disease 2019 (COVID- 19). Furthermore, understanding and predicting the demand
for Personal Protective Equipment promotes the efficiency and safety of
healthcare workers in Community Health Centres. Owing to the novel nature and
strains of SARS-CoV-2, relatively few literature and research exist in this
regard. These existing literature have attempted to solve the problem
statement(s) using either Agent-based Models, Machine Learning Models, or
Mathematical Models. In view of this, our work herein adds to existing
literature via modeling our problem statements as Multi- Task Learning
problems. Results from our research indicate that government actions and human
factors are the most significant determinants that influence the spread of
SARS-CoV-2.

    

### [[2108.09412] SemiFed: Semi-supervised Federated Learning with Consistency and Pseudo-Labeling](http://arxiv.org/abs/2108.09412)


  Federated learning enables multiple clients, such as mobile phones and
organizations, to collaboratively learn a shared model for prediction while
protecting local data privacy. However, most recent research and applications
of federated learning assume that all clients have fully labeled data, which is
impractical in real-world settings. In this work, we focus on a new scenario
for cross-silo federated learning, where data samples of each client are
partially labeled. We borrow ideas from semi-supervised learning methods where
a large amount of unlabeled data is utilized to improve the model's accuracy
despite limited access to labeled examples. We propose a new framework dubbed
SemiFed that unifies two dominant approaches for semi-supervised learning:
consistency regularization and pseudo-labeling. SemiFed first applies advanced
data augmentation techniques to enforce consistency regularization and then
generates pseudo-labels using the model's predictions during training. SemiFed
takes advantage of the federation so that for a given image, the pseudo-label
holds only if multiple models from different clients produce a high-confidence
prediction and agree on the same label. Extensive experiments on two image
benchmarks demonstrate the effectiveness of our approach under both homogeneous
and heterogeneous data distribution settings

    

### [[2108.09413] Integer-arithmetic-only Certified Robustness for Quantized Neural Networks](http://arxiv.org/abs/2108.09413)


  Adversarial data examples have drawn significant attention from the machine
learning and security communities. A line of work on tackling adversarial
examples is certified robustness via randomized smoothing that can provide a
theoretical robustness guarantee. However, such a mechanism usually uses
floating-point arithmetic for calculations in inference and requires large
memory footprints and daunting computational costs. These defensive models
cannot run efficiently on edge devices nor be deployed on integer-only logical
units such as Turing Tensor Cores or integer-only ARM processors. To overcome
these challenges, we propose an integer randomized smoothing approach with
quantization to convert any classifier into a new smoothed classifier, which
uses integer-only arithmetic for certified robustness against adversarial
perturbations. We prove a tight robustness guarantee under L2-norm for the
proposed approach. We show our approach can obtain a comparable accuracy and
4x~5x speedup over floating-point arithmetic certified robust methods on
general-purpose CPUs and mobile devices on two distinct datasets (CIFAR-10 and
Caltech-101).

    

### [[2108.09420] Fast Sketching of Polynomial Kernels of Polynomial Degree](http://arxiv.org/abs/2108.09420)


  Kernel methods are fundamental in machine learning, and faster algorithms for
kernel approximation provide direct speedups for many core tasks in machine
learning. The polynomial kernel is especially important as other kernels can
often be approximated by the polynomial kernel via a Taylor series expansion.
Recent techniques in oblivious sketching reduce the dependence in the running
time on the degree $q$ of the polynomial kernel from exponential to polynomial,
which is useful for the Gaussian kernel, for which $q$ can be chosen to be
polylogarithmic. However, for more slowly growing kernels, such as the neural
tangent and arc-cosine kernels, $q$ needs to be polynomial, and previous work
incurs a polynomial factor slowdown in the running time. We give a new
oblivious sketch which greatly improves upon this running time, by removing the
dependence on $q$ in the leading order term. Combined with a novel sampling
scheme, we give the fastest algorithms for approximating a large family of
slow-growing kernels.

    

### [[2108.09423] Adaptive unsupervised learning with enhanced feature representation for intra-tumor partitioning and survival prediction for glioblastoma](http://arxiv.org/abs/2108.09423)


  Glioblastoma is profoundly heterogeneous in regional microstructure and
vasculature. Characterizing the spatial heterogeneity of glioblastoma could
lead to more precise treatment. With unsupervised learning techniques,
glioblastoma MRI-derived radiomic features have been widely utilized for tumor
sub-region segmentation and survival prediction. However, the reliability of
algorithm outcomes is often challenged by both ambiguous intermediate process
and instability introduced by the randomness of clustering algorithms,
especially for data from heterogeneous patients.
In this paper, we propose an adaptive unsupervised learning approach for
efficient MRI intra-tumor partitioning and glioblastoma survival prediction. A
novel and problem-specific Feature-enhanced Auto-Encoder (FAE) is developed to
enhance the representation of pairwise clinical modalities and therefore
improve clustering stability of unsupervised learning algorithms such as
K-means. Moreover, the entire process is modelled by the Bayesian optimization
(BO) technique with a custom loss function that the hyper-parameters can be
adaptively optimized in a reasonably few steps. The results demonstrate that
the proposed approach can produce robust and clinically relevant MRI
sub-regions and statistically significant survival predictions.

    

### [[2108.09435] Fairness-Aware Online Meta-learning](http://arxiv.org/abs/2108.09435)


  In contrast to offline working fashions, two research paradigms are devised
for online learning: (1) Online Meta Learning (OML) learns good priors over
model parameters (or learning to learn) in a sequential setting where tasks are
revealed one after another. Although it provides a sub-linear regret bound,
such techniques completely ignore the importance of learning with fairness
which is a significant hallmark of human intelligence. (2) Online
Fairness-Aware Learning. This setting captures many classification problems for
which fairness is a concern. But it aims to attain zero-shot generalization
without any task-specific adaptation. This therefore limits the capability of a
model to adapt onto newly arrived data. To overcome such issues and bridge the
gap, in this paper for the first time we proposed a novel online meta-learning
algorithm, namely FFML, which is under the setting of unfairness prevention.
The key part of FFML is to learn good priors of an online fair classification
model's primal and dual parameters that are associated with the model's
accuracy and fairness, respectively. The problem is formulated in the form of a
bi-level convex-concave optimization. Theoretic analysis provides sub-linear
upper bounds for loss regret and for violation of cumulative fairness
constraints. Our experiments demonstrate the versatility of FFML by applying it
to classification on three real-world datasets and show substantial
improvements over the best prior work on the tradeoff between fairness and
classification accuracy

    

### [[2108.09444] Temporal Induced Self-Play for Stochastic Bayesian Games](http://arxiv.org/abs/2108.09444)


  One practical requirement in solving dynamic games is to ensure that the
players play well from any decision point onward. To satisfy this requirement,
existing efforts focus on equilibrium refinement, but the scalability and
applicability of existing techniques are limited. In this paper, we propose
Temporal-Induced Self-Play (TISP), a novel reinforcement learning-based
framework to find strategies with decent performances from any decision point
onward. TISP uses belief-space representation, backward induction, policy
learning, and non-parametric approximation. Building upon TISP, we design a
policy-gradient-based algorithm TISP-PG. We prove that TISP-based algorithms
can find approximate Perfect Bayesian Equilibrium in zero-sum one-sided
stochastic Bayesian games with finite horizon. We test TISP-based algorithms in
various games, including finitely repeated security games and a grid-world
game. The results show that TISP-PG is more scalable than existing mathematical
programming-based methods and significantly outperforms other learning-based
methods.

    

### [[2108.09446] Reservoir Computing with Diverse Timescales for Prediction of Multiscale Dynamics](http://arxiv.org/abs/2108.09446)


  Machine learning approaches have recently been leveraged as a substitute or
an aid for physical/mathematical modeling approaches to dynamical systems. To
develop an efficient machine learning method dedicated to modeling and
prediction of multiscale dynamics, we propose a reservoir computing model with
diverse timescales by using a recurrent network of heterogeneous leaky
integrator neurons. In prediction tasks with fast-slow chaotic dynamical
systems including a large gap in timescales of their subsystems dynamics, we
demonstrate that the proposed model has a higher potential than the existing
standard model and yields a performance comparable to the best one of the
standard model even without an optimization of the leak rate parameter. Our
analysis reveals that the timescales required for producing each component of
target dynamics are appropriately and flexibly selected from the reservoir
dynamics by model training.

    

### [[2108.09454] "Adversarial Examples" for Proof-of-Learning](http://arxiv.org/abs/2108.09454)


  In S&P '21, Jia et al. proposed a new concept/mechanism named
proof-of-learning (PoL), which allows a prover to demonstrate ownership of a
machine learning model by proving integrity of the training procedure. It
guarantees that an adversary cannot construct a valid proof with less cost (in
both computation and storage) than that made by the prover in generating the
proof. A PoL proof includes a set of intermediate models recorded during
training, together with the corresponding data points used to obtain each
recorded model. Jia et al. claimed that an adversary merely knowing the final
model and training dataset cannot efficiently find a set of intermediate models
with correct data points. In this paper, however, we show that PoL is
vulnerable to "adversarial examples"! Specifically, in a similar way as
optimizing an adversarial example, we could make an arbitrarily-chosen data
point "generate" a given model, hence efficiently generating intermediate
models with correct data points. We demonstrate, both theoretically and
empirically, that we are able to generate a valid proof with significantly less
cost than generating a proof by the prover, thereby we successfully break PoL.

    

### [[2108.09478] MimicBot: Combining Imitation and Reinforcement Learning to win in Bot Bowl](http://arxiv.org/abs/2108.09478)


  This paper describe an hybrid agent trained to play in Fantasy Football AI
which participated in the Bot Bowl III competition. The agent, MimicBot, is
implemented using a specifically designed deep policy network and trained using
a combination of imitation and reinforcement learning. Previous attempts in
using a reinforcement learning approach in such context failed for a number of
reasons, e.g. due to the intrinsic randomness in the environment and the large
and uneven number of actions available, with a curriculum learning approach
failing to consistently beat a randomly paying agent. Currently no machine
learning approach can beat a scripted bot which makes use of the domain
knowledge on the game. Our solution, thanks to an imitation learning and a
hybrid decision-making process, consistently beat such scripted agents.
Moreover we shed lights on how to more efficiently train in a reinforcement
learning setting while drastically increasing sample efficiency. MimicBot is
the winner of the Bot Bowl III competition, and it is currently the
state-of-the-art solution.

    

### [[2108.09484] CushLEPOR: Customised hLEPOR Metric Using LABSE Distilled Knowledge Model to Improve Agreement with Human Judgements](http://arxiv.org/abs/2108.09484)


  Human evaluation has always been expensive while researchers struggle to
trust the automatic metrics. To address this, we propose to customise
traditional metrics by taking advantages of the pre-trained language models
(PLMs) and the limited available human labelled scores. We first re-introduce
the hLEPOR metric factors, followed by the Python portable version we developed
which achieved the automatic tuning of the weighting parameters in hLEPOR
metric. Then we present the customised hLEPOR (cushLEPOR) which uses LABSE
distilled knowledge model to improve the metric agreement with human judgements
by automatically optimised factor weights regarding the exact MT language pairs
that cushLEPOR is deployed to. We also optimise cushLEPOR towards human
evaluation data based on MQM and pSQM framework on English-German and
Chinese-English language pairs. The experimental investigations show cushLEPOR
boosts hLEPOR performances towards better agreements to PLMs like LABSE with
much lower cost, and better agreements to human evaluations including MQM and
pSQM scores, and yields much better performances than BLEU (data available at
\url{this https URL}).

    

### [[2108.09500] A computational study on imputation methods for missing environmental data](http://arxiv.org/abs/2108.09500)


  Data acquisition and recording in the form of databases are routine
operations. The process of collecting data, however, may experience
irregularities, resulting in databases with missing data. Missing entries might
alter analysis efficiency and, consequently, the associated decision-making
process. This paper focuses on databases collecting information related to the
natural environment. Given the broad spectrum of recorded activities, these
databases typically are of mixed nature. It is therefore relevant to evaluate
the performance of missing data processing methods considering this
characteristic. In this paper we investigate the performances of several
missing data imputation methods and their application to the problem of missing
data in environment. A computational study was performed to compare the method
missForest (MF) with two other imputation methods, namely Multivariate
Imputation by Chained Equations (MICE) and K-Nearest Neighbors (KNN). Tests
were made on 10 pretreated datasets of various types. Results revealed that MF
generally outperformed MICE and KNN in terms of imputation errors, with a more
pronounced performance gap for mixed typed databases where MF reduced the
imputation error up to 150%, when compared to the other methods. KNN was
usually the fastest method. MF was then successfully applied to a case study on
Quebec wastewater treatment plants performance monitoring. We believe that the
present study demonstrates the pertinence of using MF as imputation method when
dealing with missing environmental data.

    

### [[2108.09501] A Sparse Structure Learning Algorithm for Bayesian Network Identification from Discrete High-Dimensional Data](http://arxiv.org/abs/2108.09501)


  This paper addresses the problem of learning a sparse structure Bayesian
network from high-dimensional discrete data. Compared to continuous Bayesian
networks, learning a discrete Bayesian network is a challenging problem due to
the large parameter space. Although many approaches have been developed for
learning continuous Bayesian networks, few approaches have been proposed for
the discrete ones. In this paper, we address learning Bayesian networks as an
optimization problem and propose a score function that satisfies the sparsity
and the DAG property simultaneously. Besides, we implement a block-wised
stochastic coordinate descent algorithm to optimize the score function.
Specifically, we use a variance reducing method in our optimization algorithm
to make the algorithm work efficiently in high-dimensional data. The proposed
approach is applied to synthetic data from well-known benchmark networks. The
quality, scalability, and robustness of the constructed network are measured.
Compared to some competitive approaches, the results reveal that our algorithm
outperforms the others in evaluation metrics.

    

### [[2108.09506] Deep Representation of Imbalanced Spatio-temporal Traffic Flow Data for Traffic Accident Detection](http://arxiv.org/abs/2108.09506)


  Automatic detection of traffic accidents has a crucial effect on improving
transportation, public safety, and path planning. Many lives can be saved by
the consequent decrease in the time between when the accidents occur and when
rescue teams are dispatched, and much travelling time can be saved by notifying
drivers to select alternative routes. This problem is challenging mainly
because of the rareness of accidents and spatial heterogeneity of the
environment. This paper studies deep representation of loop detector data using
Long-Short Term Memory (LSTM) network for automatic detection of freeway
accidents. The LSTM-based framework increases class separability in the encoded
feature space while reducing the dimension of data. Our experiments on real
accident and loop detector data collected from the Twin Cities Metro freeways
of Minnesota demonstrate that deep representation of traffic flow data using
LSTM network has the potential to detect freeway accidents in less than 18
minutes with a true positive rate of 0.71 and a false positive rate of 0.25
which outperforms other competing methods in the same arrangement.

    

### [[2108.09507] How Can Increased Randomness in Stochastic Gradient Descent Improve Generalization?](http://arxiv.org/abs/2108.09507)


  Recent works report that increasing the learning rate or decreasing the
minibatch size in stochastic gradient descent (SGD) can improve test set
performance. We argue this is expected under some conditions in models with a
loss function with multiple local minima. Our main contribution is an
approximate but analytical approach inspired by methods in Physics to study the
role of the SGD learning rate and batch size in generalization. We characterize
test set performance under a shift between the training and test data
distributions for loss functions with multiple minima. The shift can simply be
due to sampling, and is therefore typically present in practical applications.
We show that the resulting shift in local minima worsens test performance by
picking up curvature, implying that generalization improves by selecting wide
and/or little-shifted local minima. We then specialize to SGD, and study its
test performance under stationarity. Because obtaining the exact stationary
distribution of SGD is intractable, we derive a Fokker-Planck approximation of
SGD and obtain its stationary distribution instead. This process shows that the
learning rate divided by the minibatch size plays a role analogous to
temperature in statistical mechanics, and implies that SGD, including its
stationary distribution, is largely invariant to changes in learning rate or
batch size that leave its temperature constant. We show that increasing SGD
temperature encourages the selection of local minima with lower curvature, and
can enable better generalization. We provide experiments on CIFAR10
demonstrating the temperature invariance of SGD, improvement of the test loss
as SGD temperature increases, and quantifying the impact of sampling versus
domain shift in driving this effect. Finally, we present synthetic experiments
showing how our theory applies in a simplified loss with two local minima.

    

### [[2108.09513] A Hard Label Black-box Adversarial Attack Against Graph Neural Networks](http://arxiv.org/abs/2108.09513)


  Graph Neural Networks (GNNs) have achieved state-of-the-art performance in
various graph structure related tasks such as node classification and graph
classification. However, GNNs are vulnerable to adversarial attacks. Existing
works mainly focus on attacking GNNs for node classification; nevertheless, the
attacks against GNNs for graph classification have not been well explored.
In this work, we conduct a systematic study on adversarial attacks against
GNNs for graph classification via perturbing the graph structure. In
particular, we focus on the most challenging attack, i.e., hard label black-box
attack, where an attacker has no knowledge about the target GNN model and can
only obtain predicted labels through querying the target this http URL achieve this
goal, we formulate our attack as an optimization problem, whose objective is to
minimize the number of edges to be perturbed in a graph while maintaining the
high attack success rate. The original optimization problem is intractable to
solve, and we relax the optimization problem to be a tractable one, which is
solved with theoretical convergence guarantee. We also design a coarse-grained
searching algorithm and a query-efficient gradient computation algorithm to
decrease the number of queries to the target GNN model. Our experimental
results on three real-world datasets demonstrate that our attack can
effectively attack representative GNNs for graph classification with less
queries and perturbations. We also evaluate the effectiveness of our attack
under two defenses: one is well-designed adversarial graph detector and the
other is that the target GNN model itself is equipped with a defense to prevent
adversarial graph generation. Our experimental results show that such defenses
are not effective enough, which highlights more advanced defenses.

    

### [[2108.09523] Automating Crystal-Structure Phase Mapping: Combining Deep Learning with Constraint Reasoning](http://arxiv.org/abs/2108.09523)


  Crystal-structure phase mapping is a core, long-standing challenge in
materials science that requires identifying crystal structures, or mixtures
thereof, in synthesized materials. Materials science experts excel at solving
simple systems but cannot solve complex systems, creating a major bottleneck in
high-throughput materials discovery. Herein we show how to automate
crystal-structure phase mapping. We formulate phase mapping as an unsupervised
pattern demixing problem and describe how to solve it using Deep Reasoning
Networks (DRNets). DRNets combine deep learning with constraint reasoning for
incorporating scientific prior knowledge and consequently require only a modest
amount of (unlabeled) data. DRNets compensate for the limited data by
exploiting and magnifying the rich prior knowledge about the thermodynamic
rules governing the mixtures of crystals with constraint reasoning seamlessly
integrated into neural network optimization. DRNets are designed with an
interpretable latent space for encoding prior-knowledge domain constraints and
seamlessly integrate constraint reasoning into neural network optimization.
DRNets surpass previous approaches on crystal-structure phase mapping,
unraveling the Bi-Cu-V oxide phase diagram, and aiding the discovery of
solar-fuels materials.

    

### [[2108.09529] Term Interrelations and Trends in Software Engineering](http://arxiv.org/abs/2108.09529)


  The Software Engineering (SE) community is prolific, making it challenging
for experts to keep up with the flood of new papers and for neophytes to enter
the field. Therefore, we posit that the community may benefit from a tool
extracting terms and their interrelations from the SE community's text corpus
and showing terms' trends. In this paper, we build a prototyping tool using the
word embedding technique. We train the embeddings on the SE Body of Knowledge
handbook and 15,233 research papers' titles and abstracts. We also create test
cases necessary for validation of the training of the embeddings. We provide
representative examples showing that the embeddings may aid in summarizing
terms and uncovering trends in the knowledge base.

    

### [[2108.09537] Using growth transform dynamical systems for spatio-temporal data sonification](http://arxiv.org/abs/2108.09537)


  Sonification, or encoding information in meaningful audio signatures, has
several advantages in augmenting or replacing traditional visualization methods
for human-in-the-loop decision-making. Standard sonification methods reported
in the literature involve either (i) using only a subset of the variables, or
(ii) first solving a learning task on the data and then mapping the output to
an audio waveform, which is utilized by the end-user to make a decision. This
paper presents a novel framework for sonifying high-dimensional data using a
complex growth transform dynamical system model where both the learning (or,
more generally, optimization) and the sonification processes are integrated
together. Our algorithm takes as input the data and optimization parameters
underlying the learning or prediction task and combines it with the
psychoacoustic parameters defined by the user. As a result, the proposed
framework outputs binaural audio signatures that not only encode some
statistical properties of the high-dimensional data but also reveal the
underlying complexity of the optimization/learning process. Along with
extensive experiments using synthetic datasets, we demonstrate the framework on
sonifying Electro-encephalogram (EEG) data with the potential for detecting
epileptic seizures in pediatric patients.

    

### [[2108.09541] Rotationally Equivariant Neural Operators for Learning Transformations on Tensor Fields (eg 3D Images and Vector Fields)](http://arxiv.org/abs/2108.09541)


  We introduce equivariant neural operators for learning resolution invariant
as well as translation and rotation equivariant transformations between sets of
tensor fields. Input and output may contain arbitrary mixes of scalar fields,
vector fields, second order tensor fields and higher order fields. Our tensor
field convolution layers emulate any linear operator by learning its impulse
response or Green's function as the convolution kernel. Our tensor field
attention layers emulate pairwise field coupling via local tensor products.
Convolutions and associated adjoints can be in real or Fourier space allowing
for linear scaling. By unifying concepts from E3NN, TBNN and FNO, we achieve
good predictive performance on a wide range of PDEs and dynamical systems in
engineering and quantum chemistry. Code is in Julia and available upon request
from authors.

    

### [[2108.09545] Joint Characterization of Spatiotemporal Data Manifolds](http://arxiv.org/abs/2108.09545)


  Spatiotemporal (ST) image data are increasingly common and often
high-dimensional (high-D). Modeling ST data can be a challenge due to the
plethora of independent and interacting processes which may or may not
contribute to the measurements. Characterization can be considered the
complement to modeling by helping guide assumptions about generative processes
and their representation in the data. Dimensionality reduction (DR) is a
frequently implemented type of characterization designed to mitigate the "curse
of dimensionality" on high-D signals. For decades, Principal Component (PC) and
Empirical Orthogonal Function (EOF) analysis has been used as a linear,
invertible approach to DR and ST analysis. Recent years have seen the
additional development of a suite of nonlinear DR algorithms, frequently
categorized as "manifold learning". Here, we explore the idea of joint
characterization of ST data manifolds using PCs/EOFs alongside two nonlinear DR
approaches: Laplacian Eigenmaps (LE) and t-distributed stochastic neighbor
embedding (t-SNE). Starting with a synthetic example and progressing to global,
regional, and field scale ST datasets spanning roughly 5 orders of magnitude in
space and 2 in time, we show these three DR approaches can yield complementary
information about ST manifold topology. Compared to the relatively diffuse TFS
produced by PCs/EOFs, the nonlinear approaches yield more compact manifolds
with decreased ambiguity in temporal endmembers (LE) and/or in spatiotemporal
clustering (t-SNE). These properties are compensated by the greater
interpretability, significantly lower computational demand and diminished
sensitivity to spatial aliasing for PCs/EOFs than LE or t-SNE. Taken together,
we find joint characterization using the three complementary DR approaches
capable of greater insight into generative ST processes than possible using any
single approach alone.

    

### [[2108.09551] Variable-Rate Deep Image Compression through Spatially-Adaptive Feature Transform](http://arxiv.org/abs/2108.09551)


  We propose a versatile deep image compression network based on Spatial
Feature Transform (SFT arXiv:1804.02815), which takes a source image and a
corresponding quality map as inputs and produce a compressed image with
variable rates. Our model covers a wide range of compression rates using a
single model, which is controlled by arbitrary pixel-wise quality maps. In
addition, the proposed framework allows us to perform task-aware image
compressions for various tasks, e.g., classification, by efficiently estimating
optimized quality maps specific to target tasks for our encoding network. This
is even possible with a pretrained network without learning separate models for
individual tasks. Our algorithm achieves outstanding rate-distortion trade-off
compared to the approaches based on multiple models that are optimized
separately for several different target rates. At the same level of
compression, the proposed approach successfully improves performance on image
classification and text region quality preservation via task-aware quality map
estimation without additional model training. The code is available at the
project website: this https URL


### [[2108.09585] Sequential Stochastic Optimization in Separable Learning Environments](http://arxiv.org/abs/2108.09585)


  We consider a class of sequential decision-making problems under uncertainty
that can encompass various types of supervised learning concepts. These
problems have a completely observed state process and a partially observed
modulation process, where the state process is affected by the modulation
process only through an observation process, the observation process only
observes the modulation process, and the modulation process is exogenous to
control. We model this broad class of problems as a partially observed Markov
decision process (POMDP). The belief function for the modulation process is
control invariant, thus separating the estimation of the modulation process
from the control of the state process. We call this specially structured POMDP
the separable POMDP, or SEP-POMDP, and show it (i) can serve as a model for a
broad class of application areas, e.g., inventory control, finance, healthcare
systems, (ii) inherits value function and optimal policy structure from a set
of completely observed MDPs, (iii) can serve as a bridge between classical
models of sequential decision making under uncertainty having fully specified
model artifacts and such models that are not fully specified and require the
use of predictive methods from statistics and machine learning, and (iv) allows
for specialized approximate solution procedures.

    

### [[2108.09592] Principal Gradient Direction and Confidence Reservoir Sampling for Continual Learning](http://arxiv.org/abs/2108.09592)


  Task-free online continual learning aims to alleviate catastrophic forgetting
of the learner on a non-iid data stream. Experience Replay (ER) is a SOTA
continual learning method, which is broadly used as the backbone algorithm for
other replay-based methods. However, the training strategy of ER is too simple
to take full advantage of replayed examples and its reservoir sampling strategy
is also suboptimal. In this work, we propose a general proximal gradient
framework so that ER can be viewed as a special case. We further propose two
improvements accordingly: Principal Gradient Direction (PGD) and Confidence
Reservoir Sampling (CRS). In Principal Gradient Direction, we optimize a target
gradient that not only represents the major contribution of past gradients, but
also retains the new knowledge of the current gradient. We then present
Confidence Reservoir Sampling for maintaining a more informative memory buffer
based on a margin-based metric that measures the value of stored examples.
Experiments substantiate the effectiveness of both our improvements and our new
algorithm consistently boosts the performance of MIR-replay, a SOTA ER-based
method: our algorithm increases the average accuracy up to 7.9% and reduces
forgetting up to 15.4% on four datasets.

    

### [[2108.09598] SERF: Towards better training of deep neural networks using log-Softplus ERror activation Function](http://arxiv.org/abs/2108.09598)


  Activation functions play a pivotal role in determining the training dynamics
and neural network performance. The widely adopted activation function ReLU
despite being simple and effective has few disadvantages including the Dying
ReLU problem. In order to tackle such problems, we propose a novel activation
function called Serf which is self-regularized and nonmonotonic in nature. Like
Mish, Serf also belongs to the Swish family of functions. Based on several
experiments on computer vision (image classification and object detection) and
natural language processing (machine translation, sentiment classification and
multimodal entailment) tasks with different state-of-the-art architectures, it
is observed that Serf vastly outperforms ReLU (baseline) and other activation
functions including both Swish and Mish, with a markedly bigger margin on
deeper architectures. Ablation studies further demonstrate that Serf based
architectures perform better than those of Swish and Mish in varying scenarios,
validating the effectiveness and compatibility of Serf with varying depth,
complexity, optimizers, learning rates, batch sizes, initializers and dropout
rates. Finally, we investigate the mathematical relation between Swish and
Serf, thereby showing the impact of preconditioner function ingrained in the
first derivative of Serf which provides a regularization effect making
gradients smoother and optimization faster.

    

### [[2108.09605] Self-Supervised Delineation of Geological Structures using Orthogonal Latent Space Projection](http://arxiv.org/abs/2108.09605)


  We developed two machine learning frameworks that could assist in automated
litho-stratigraphic interpretation of seismic volumes without any manual hand
labeling from an experienced seismic interpreter. The first framework is an
unsupervised hierarchical clustering model to divide seismic images from a
volume into certain number of clusters determined by the algorithm. The
clustering framework uses a combination of density and hierarchical techniques
to determine the size and homogeneity of the clusters. The second framework
consists of a self-supervised deep learning framework to label regions of
geological interest in seismic images. It projects the latent-space of an
encoder-decoder architecture unto two orthogonal subspaces, from which it
learns to delineate regions of interest in the seismic images. To demonstrate
an application of both frameworks, a seismic volume was clustered into various
contiguous clusters, from which four clusters were selected based on distinct
seismic patterns: horizons, faults, salt domes and chaotic structures. Images
from the selected clusters are used to train the encoder-decoder network. The
output of the encoder-decoder network is a probability map of the possibility
an amplitude reflection event belongs to an interesting geological structure.
The structures are delineated using the probability map. The delineated images
are further used to post-train a segmentation model to extend our results to
full-vertical sections. The results on vertical sections show that we can
factorize a seismic volume into its corresponding structural components.
Lastly, we showed that our deep learning framework could be modeled as an
attribute extractor and we compared our attribute result with various existing
attributes in literature and demonstrate competitive performance with them.

    

### [[2108.09618] Personalised Federated Learning: A Combinational Approach](http://arxiv.org/abs/2108.09618)


  Federated learning (FL) is a distributed machine learning approach involving
multiple clients collaboratively training a shared model. Such a system has the
advantage of more training data from multiple clients, but data can be
non-identically and independently distributed (non-i.i.d.). Privacy and
integrity preserving features such as differential privacy (DP) and robust
aggregation (RA) are commonly used in FL. In this work, we show that on common
deep learning tasks, the performance of FL models differs amongst clients and
situations, and FL models can sometimes perform worse than local models due to
non-i.i.d. data. Secondly, we show that incorporating DP and RA degrades
performance further. Then, we conduct an ablation study on the performance
impact of different combinations of common personalization approaches for FL,
such as finetuning, mixture-of-experts ensemble, multi-task learning, and
knowledge distillation. It is observed that certain combinations of
personalization approaches are more impactful in certain scenarios while others
always improve performance, and combination approaches are better than
individual ones. Most clients obtained better performance with combined
personalized FL and recover from performance degradation caused by non-i.i.d.
data, DP, and RA.

    

### [[2108.09619] Evaluation Methodologies for Code Learning Tasks](http://arxiv.org/abs/2108.09619)


  There has been a growing interest in developing machine learning (ML) models
for code learning tasks, e.g., comment generation and method naming. Despite
substantial increase in the effectiveness of ML models, the evaluation
methodologies, i.e., the way people split datasets into training, validation,
and testing sets, were not well designed. Specifically, no prior work on the
aforementioned topics considered the timestamps of code and comments during
evaluation (e.g., examples in the testing set might be from 2010 and examples
from the training set might be from 2020). This may lead to evaluations that
are inconsistent with the intended use cases of the ML models. In this paper,
we formalize a novel time-segmented evaluation methodology, as well as the two
methodologies commonly used in the literature: mixed-project and cross-project.
We argue that time-segmented methodology is the most realistic. We also
describe various use cases of ML models and provide a guideline for using
methodologies to evaluate each use case. To assess the impact of methodologies,
we collect a dataset of code-comment pairs with timestamps to train and
evaluate several recent code learning ML models for the comment generation and
method naming tasks. Our results show that different methodologies can lead to
conflicting and inconsistent results. We invite the community to adopt the
time-segmented evaluation methodology.

    

### [[2108.09637] Graph-Convolutional Deep Learning to Identify Optimized Molecular Configurations](http://arxiv.org/abs/2108.09637)


  Tackling molecular optimization problems using conventional computational
methods is challenging, because the determination of the optimized
configuration is known to be an NP-hard problem. Recently, there has been
increasing interest in applying different deep-learning techniques to benchmark
molecular optimization tasks. In this work, we implement a graph-convolutional
method to classify molecular structures using the equilibrium and
non-equilibrium configurations provided in the QM7-X data set. Atomic forces
are encoded in graph vertices and the substantial suppression in the total
force magnitude on the atoms in the optimized structure is learned for the
graph classification task. We demonstrate the results using two different graph
pooling layers and compare their respective performances.

    

### [[2108.09645] An Efficient Mini-batch Method via Partial Transportation](http://arxiv.org/abs/2108.09645)


  Mini-batch optimal transport (m-OT) has been widely used recently to deal
with the memory issue of OT in large-scale applications. Despite their
practicality, m-OT suffers from misspecified mappings, namely, mappings that
are optimal on the mini-batch level but do not exist in the optimal
transportation plan between the original measures. To address the misspecified
mappings issue, we propose a novel mini-batch method by using partial optimal
transport (POT) between mini-batch empirical measures, which we refer to as
mini-batch partial optimal transport (m-POT). Leveraging the insight from the
partial transportation, we explain the source of misspecified mappings from the
m-OT and motivate why limiting the amount of transported masses among
mini-batches via POT can alleviate the incorrect mappings. Finally, we carry
out extensive experiments on various applications to compare m-POT with m-OT
and recently proposed mini-batch method, mini-batch unbalanced optimal
transport (m-UOT). We observe that m-POT is better than m-OT deep domain
adaptation applications while having comparable performance with m-UOT. On
other applications, such as deep generative model, gradient flow, and color
transfer, m-POT yields more favorable performance than both m-OT and m-UOT.

    

### [[2108.09646] A Systematic Literature Review of Automated Query Reformulations in Source Code Search](http://arxiv.org/abs/2108.09646)


  Software developers often fix critical bugs to ensure the reliability of
their software. They might also need to add new features to their software at a
regular interval to stay competitive in the market. These bugs and features are
reported as change requests (i.e., technical documents written by software
users). Developers consult these documents to implement the required changes in
the software code. As a part of change implementation, they often choose a few
important keywords from a change request as an ad hoc query. Then they execute
the query with a code search engine (e.g., Lucene) and attempt to find out the
exact locations within the software code that need to be changed.
Unfortunately, even experienced developers often fail to choose the right
queries. As a consequence, the developers often experience difficulties in
detecting the appropriate locations within the code and spend the majority of
their time in numerous trials and errors. There have been many studies that
attempt to support developers in constructing queries by automatically
reformulating their ad hoc queries. In this systematic literature review, we
carefully select 70 primary studies on query reformulations from 2,970
candidate studies, perform an in-depth qualitative analysis using the Grounded
Theory approach, and then answer six important research questions. Our
investigation has reported several major findings. First, to date, eight major
methodologies (e.g., term weighting, query-term co-occurrence analysis,
thesaurus lookup) have been adopted in query reformulation. Second, the
existing studies suffer from several major limitations (e.g., lack of
generalizability, vocabulary mismatch problem, weak evaluation, the extra
burden on the developers) that might prevent their wide adoption. Finally, we
discuss several open issues in search query reformulations and suggest multiple
future research opportunities.

    

### [[2108.09649] The Exploitation of Distance Distributions for Clustering](http://arxiv.org/abs/2108.09649)


  Although distance measures are used in many machine learning algorithms, the
literature on the context-independent selection and evaluation of distance
measures is limited in the sense that prior knowledge is used. In cluster
analysis, current studies evaluate the choice of distance measure after
applying unsupervised methods based on error probabilities, implicitly setting
the goal of reproducing predefined partitions in data. Such studies use
clusters of data that are often based on the context of the data as well as the
custom goal of the specific study. Depending on the data context, different
properties for distance distributions are judged to be relevant for appropriate
distance selection. However, if cluster analysis is based on the task of
finding similar partitions of data, then the intrapartition distances should be
smaller than the interpartition distances. By systematically investigating this
specification using distribution analysis through a mirrored-density plot, it
is shown that multimodal distance distributions are preferable in cluster
analysis. As a consequence, it is advantageous to model distance distributions
with Gaussian mixtures prior to the evaluation phase of unsupervised methods.
Experiments are performed on several artificial datasets and natural datasets
for the task of clustering.

    

### [[2108.09656] ExamGAN and Twin-ExamGAN for Exam Script Generation](http://arxiv.org/abs/2108.09656)


  Nowadays, the learning management system (LMS) has been widely used in
different educational stages from primary to tertiary education for student
administration, documentation, tracking, reporting, and delivery of educational
courses, training programs, or learning and development programs. Towards
effective learning outcome assessment, the exam script generation problem has
attracted many attentions and been investigated recently. But the research in
this field is still in its early stage. There are opportunities to further
improve the quality of generated exam scripts in various aspects. In
particular, two essential issues have been ignored largely by existing
solutions. First, given a course, it is unknown yet how to generate an exam
script which can result in a desirable distribution of student scores in a
class (or across different classes). Second, while it is frequently encountered
in practice, it is unknown so far how to generate a pair of high quality exam
scripts which are equivalent in assessment (i.e., the student scores are
comparable by taking either of them) but have significantly different sets of
questions. To fill the gap, this paper proposes ExamGAN (Exam Script Generative
Adversarial Network) to generate high quality exam scripts, and then extends
ExamGAN to T-ExamGAN (Twin-ExamGAN) to generate a pair of high quality exam
scripts. Based on extensive experiments on three benchmark datasets, it has
verified the superiority of proposed solutions in various aspects against the
state-of-the-art. Moreover, we have conducted a case study which demonstrated
the effectiveness of proposed solution in a real teaching scenario.

    

### [[2108.09659] Evolutionary Ensemble Learning for Multivariate Time Series Prediction](http://arxiv.org/abs/2108.09659)


  Multivariate time series (MTS) prediction plays a key role in many fields
such as finance, energy and transport, where each individual time series
corresponds to the data collected from a certain data source, so-called
channel. A typical pipeline of building an MTS prediction model (PM) consists
of selecting a subset of channels among all available ones, extracting features
from the selected channels, and building a PM based on the extracted features,
where each component involves certain optimization tasks, i.e., selection of
channels, feature extraction (FE) methods, and PMs as well as configuration of
the selected FE method and PM. Accordingly, pursuing the best prediction
performance corresponds to optimizing the pipeline by solving all of its
involved optimization problems. This is a non-trivial task due to the vastness
of the solution space. Different from most of the existing works which target
at optimizing certain components of the pipeline, we propose a novel
evolutionary ensemble learning framework to optimize the entire pipeline in a
holistic manner. In this framework, a specific pipeline is encoded as a
candidate solution and a multi-objective evolutionary algorithm is applied
under different population sizes to produce multiple Pareto optimal sets
(POSs). Finally, selective ensemble learning is designed to choose the optimal
subset of solutions from the POSs and combine them to yield final prediction by
using greedy sequential selection and least square methods. We implement the
proposed framework and evaluate our implementation on two real-world
applications, i.e., electricity consumption prediction and air quality
prediction. The performance comparison with state-of-the-art techniques
demonstrates the superiority of the proposed approach.

    

### [[2108.09664] New Trends in Quantum Machine Learning](http://arxiv.org/abs/2108.09664)


  Here we will give a perspective on new possible interplays between Machine
Learning and Quantum Physics, including also practical cases and applications.
We will explore the ways in which machine learning could benefit from new
quantum technologies and algorithms to find new ways to speed up their
computations by breakthroughs in physical hardware, as well as to improve
existing models or devise new learning schemes in the quantum domain. Moreover,
there are lots of experiments in quantum physics that do generate incredible
amounts of data and machine learning would be a great tool to analyze those and
make predictions, or even control the experiment itself. On top of that, data
visualization techniques and other schemes borrowed from machine learning can
be of great use to theoreticians to have better intuition on the structure of
complex manifolds or to make predictions on theoretical models. This new
research field, named as Quantum Machine Learning, is very rapidly growing
since it is expected to provide huge advantages over its classical counterpart
and deeper investigations are timely needed since they can be already tested on
the already commercially available quantum machines.

    

### [[2108.09671] Pi-NAS: Improving Neural Architecture Search by Reducing Supernet Training Consistency Shift](http://arxiv.org/abs/2108.09671)


  Recently proposed neural architecture search (NAS) methods co-train billions
of architectures in a supernet and estimate their potential accuracy using the
network weights detached from the supernet. However, the ranking correlation
between the architectures' predicted accuracy and their actual capability is
incorrect, which causes the existing NAS methods' dilemma. We attribute this
ranking correlation problem to the supernet training consistency shift,
including feature shift and parameter shift. Feature shift is identified as
dynamic input distributions of a hidden layer due to random path sampling. The
input distribution dynamic affects the loss descent and finally affects
architecture ranking. Parameter shift is identified as contradictory parameter
updates for a shared layer lay in different paths in different training steps.
The rapidly-changing parameter could not preserve architecture ranking. We
address these two shifts simultaneously using a nontrivial supernet-Pi model,
called Pi-NAS. Specifically, we employ a supernet-Pi model that contains
cross-path learning to reduce the feature consistency shift between different
paths. Meanwhile, we adopt a novel nontrivial mean teacher containing negative
samples to overcome parameter shift and model collision. Furthermore, our
Pi-NAS runs in an unsupervised manner, which can search for more transferable
architectures. Extensive experiments on ImageNet and a wide range of downstream
tasks (e.g., COCO 2017, ADE20K, and Cityscapes) demonstrate the effectiveness
and universality of our Pi-NAS compared to supervised NAS. See Codes:
this https URL.

    

### [[2108.09676] Efficient Gaussian Neural Processes for Regression](http://arxiv.org/abs/2108.09676)


  Conditional Neural Processes (CNP; Garnelo et al., 2018) are an attractive
family of meta-learning models which produce well-calibrated predictions,
enable fast inference at test time, and are trainable via a simple maximum
likelihood procedure. A limitation of CNPs is their inability to model
dependencies in the outputs. This significantly hurts predictive performance
and renders it impossible to draw coherent function samples, which limits the
applicability of CNPs in down-stream applications and decision making.
NeuralProcesses (NPs; Garnelo et al., 2018) attempt to alleviate this issue by
using latent variables, rely-ing on these to model output dependencies, but
introduces difficulties stemming from approximate inference. One recent
alternative (Bruinsma et al.,2021), which we refer to as the FullConvGNP,
models dependencies in the predictions while still being trainable via exact
maximum-likelihood.Unfortunately, the FullConvGNP relies on expensive
2D-dimensional convolutions, which limit its applicability to only
one-dimensional this http URL this work, we present an alternative way to model
output dependencies which also lends it-self maximum likelihood training but,
unlike the FullConvGNP, can be scaled to two- and three-dimensional data. The
proposed models exhibit good performance in synthetic experiments

    

### [[2108.09684] Rainfall-runoff prediction using a Gustafson-Kessel clustering based Takagi-Sugeno Fuzzy model](http://arxiv.org/abs/2108.09684)


  A rainfall-runoff model predicts surface runoff either using a
physically-based approach or using a systems-based approach. Takagi-Sugeno (TS)
Fuzzy models are systems-based approaches and a popular modeling choice for
hydrologists in recent decades due to several advantages and improved accuracy
in prediction over other existing models. In this paper, we propose a new
rainfall-runoff model developed using Gustafson-Kessel (GK) clustering-based TS
Fuzzy model. We present comparative performance measures of GK algorithms with
two other clustering algorithms: (i) Fuzzy C-Means (FCM), and (ii)Subtractive
Clustering (SC). Our proposed TS Fuzzy model predicts surface runoff using: (i)
observed rainfall in a drainage basin and (ii) previously observed
precipitation flow in the basin outlet. The proposed model is validated using
the rainfall-runoff data collected from the sensors installed on the campus of
the Indian Institute of Technology, Kharagpur. The optimal number of rules of
the proposed model is obtained by different validation indices. A comparative
study of four performance criteria: RootMean Square Error (RMSE), Coefficient
of Efficiency (CE), Volumetric Error (VE), and Correlation Coefficient of
Determination(R) have been quantitatively demonstrated for each clustering
algorithm.

    

### [[2108.09711] FEDI: Few-shot learning based on Earth Mover's Distance algorithm combined with deep residual network to identify diabetic retinopathy](http://arxiv.org/abs/2108.09711)


  Diabetic retinopathy(DR) is the main cause of blindness in diabetic patients.
However, DR can easily delay the occurrence of blindness through the diagnosis
of the fundus. In view of the reality, it is difficult to collect a large
amount of diabetic retina data in clinical practice. This paper proposes a
few-shot learning model of a deep residual network based on Earth Mover's
Distance algorithm to assist in diagnosing DR. We build training and validation
classification tasks for few-shot learning based on 39 categories of 1000
sample data, train deep residual networks, and obtain experience maximization
pre-training models. Based on the weights of the pre-trained model, the Earth
Mover's Distance algorithm calculates the distance between the images, obtains
the similarity between the images, and changes the model's parameters to
improve the accuracy of the training model. Finally, the experimental
construction of the small sample classification task of the test set to
optimize the model further, and finally, an accuracy of 93.5667% on the
3way10shot task of the diabetic retina test set. For the experimental code and
results, please refer to:
this https URL.

    

### [[2108.09733] A universally consistent learning rule with a universally monotone error](http://arxiv.org/abs/2108.09733)


  We present a universally consistent learning rule whose expected error is
monotone non-increasing with the sample size under every data distribution. The
question of existence of such rules was brought up in 1996 by Devroye, Györfi
and Lugosi (who called them "smart"). Our rule is fully deterministic, a
data-dependent partitioning rule constructed in an arbitrary domain (a standard
Borel space) using a cyclic order. The central idea is to only partition at
each step those cyclic intervals that exhibit a sufficient empirical diversity
of labels, thus avoiding a region where the error function is convex.

    

### [[2108.09737] A Transformer Architecture for Stress Detection from ECG](http://arxiv.org/abs/2108.09737)


  Electrocardiogram (ECG) has been widely used for emotion recognition. This
paper presents a deep neural network based on convolutional layers and a
transformer mechanism to detect stress using ECG signals. We perform
leave-one-subject-out experiments on two publicly available datasets, WESAD and
SWELL-KW, to evaluate our method. Our experiments show that the proposed model
achieves strong results, comparable or better than the state-of-the-art models
for ECG-based stress detection on these two datasets. Moreover, our method is
end-to-end, does not require handcrafted features, and can learn robust
representations with only a few convolutional blocks and the transformer
component.

    

### [[2108.09749] Flexible Clustered Federated Learning for Client-Level Data Distribution Shift](http://arxiv.org/abs/2108.09749)


  Federated Learning (FL) enables the multiple participating devices to
collaboratively contribute to a global neural network model while keeping the
training data locally. Unlike the centralized training setting, the non-IID,
imbalanced (statistical heterogeneity) and distribution shifted training data
of FL is distributed in the federated network, which will increase the
divergences between the local models and the global model, further degrading
performance. In this paper, we propose a flexible clustered federated learning
(CFL) framework named FlexCFL, in which we 1) group the training of clients
based on the similarities between the clients' optimization directions for
lower training divergence; 2) implement an efficient newcomer device cold start
mechanism for framework scalability and practicality; 3) flexibly migrate
clients to meet the challenge of client-level data distribution shift. FlexCFL
can achieve improvements by dividing joint optimization into groups of
sub-optimization and can strike a balance between accuracy and communication
efficiency in the distribution shift environment. The convergence and
complexity are analyzed to demonstrate the efficiency of FlexCFL. We also
evaluate FlexCFL on several open datasets and made comparisons with related CFL
frameworks. The results show that FlexCFL can significantly improve absolute
test accuracy by +10.6% on FEMNIST compared to FedAvg, +3.5% on FashionMNIST
compared to FedProx, +8.4% on MNIST compared to FeSEM. The experiment results
show that FlexCFL is also communication efficient in the distribution shift
environment.

    

### [[2108.09767] A Boosting Approach to Reinforcement Learning](http://arxiv.org/abs/2108.09767)


  We study efficient algorithms for reinforcement learning in Markov decision
processes whose complexity is independent of the number of states. This
formulation succinctly captures large scale problems, but is also known to be
computationally hard in its general form. Previous approaches attempt to
circumvent the computational hardness by assuming structure in either
transition function or the value function, or by relaxing the solution
guarantee to a local optimality condition.
We consider the methodology of boosting, borrowed from supervised learning,
for converting weak learners into an accurate policy. The notion of weak
learning we study is that of sampled-based approximate optimization of linear
functions over policies. Under this assumption of weak learnability, we give an
efficient algorithm that is capable of improving the accuracy of such weak
learning methods, till global optimality is reached. We prove sample complexity
and running time bounds on our method, that are polynomial in the natural
parameters of the problem: approximation guarantee, discount factor,
distribution mismatch and number of actions. In particular, our bound does not
depend on the number of states.
A technical difficulty in applying previous boosting results, is that the
value function over policy space is not convex. We show how to use a non-convex
variant of the Frank-Wolfe method, coupled with recent advances in gradient
boosting that allow incorporating a weak learner with multiplicative
approximation guarantee, to overcome the non-convexity and attain global
convergence.

    

### [[2108.09779] Transferring Dexterous Manipulation from GPU Simulation to a Remote Real-World TriFinger](http://arxiv.org/abs/2108.09779)


  We present a system for learning a challenging dexterous manipulation task
involving moving a cube to an arbitrary 6-DoF pose with only 3-fingers trained
with NVIDIA's IsaacGym simulator. We show empirical benefits, both in
simulation and sim-to-real transfer, of using keypoints as opposed to
position+quaternion representations for the object pose in 6-DoF for policy
observations and in reward calculation to train a model-free reinforcement
learning agent. By utilizing domain randomization strategies along with the
keypoint representation of the pose of the manipulated object, we achieve a
high success rate of 83% on a remote TriFinger system maintained by the
organizers of the Real Robot Challenge. With the aim of assisting further
research in learning in-hand manipulation, we make the codebase of our system,
along with trained checkpoints that come with billions of steps of experience
available, at this https URL


### [[2108.09797] Wind Power Projection using Weather Forecasts by Novel Deep Neural Networks](http://arxiv.org/abs/2108.09797)


  The transition from conventional methods of energy production to renewable
energy production necessitates better prediction models of the upcoming supply
of renewable energy. In wind power production, error in forecasting production
is impossible to negate owing to the intermittence of wind. For successful
power grid integration, it is crucial to understand the uncertainties that
arise in predicting wind power production and use this information to build an
accurate and reliable forecast. This can be achieved by observing the
fluctuations in wind power production with changes in different parameters such
as wind speed, temperature, and wind direction, and deriving functional
dependencies for the same. Using optimized machine learning algorithms, it is
possible to find obscured patterns in the observations and obtain meaningful
data, which can then be used to accurately predict wind power requirements .
Utilizing the required data provided by the Gamesa's wind farm at Bableshwar,
the paper explores the use of both parametric and the non-parametric models for
calculating wind power prediction using power curves. The obtained results are
subject to comparison to better understand the accuracy of the utilized models
and to determine the most suitable model for predicting wind power production
based on the given data set.

    

### [[2108.09805] Efficient Algorithms for Learning from Coarse Labels](http://arxiv.org/abs/2108.09805)


  For many learning problems one may not have access to fine grained label
information; e.g., an image can be labeled as husky, dog, or even animal
depending on the expertise of the annotator. In this work, we formalize these
settings and study the problem of learning from such coarse data. Instead of
observing the actual labels from a set $\mathcal{Z}$, we observe coarse labels
corresponding to a partition of $\mathcal{Z}$ (or a mixture of partitions).
Our main algorithmic result is that essentially any problem learnable from
fine grained labels can also be learned efficiently when the coarse data are
sufficiently informative. We obtain our result through a generic reduction for
answering Statistical Queries (SQ) over fine grained labels given only coarse
labels. The number of coarse labels required depends polynomially on the
information distortion due to coarsening and the number of fine labels
$|\mathcal{Z}|$.
We also investigate the case of (infinitely many) real valued labels focusing
on a central problem in censored and truncated statistics: Gaussian mean
estimation from coarse data. We provide an efficient algorithm when the sets in
the partition are convex and establish that the problem is NP-hard even for
very simple non-convex sets.

    

### [[2108.09817] Electroencephalogram Signal Processing with Independent Component Analysis and Cognitive Stress Classification using Convolutional Neural Networks](http://arxiv.org/abs/2108.09817)


  Electroencephalogram (EEG) is the recording which is the result due to the
activity of bio-electrical signals that is acquired from electrodes placed on
the scalp. In Electroencephalogram signal(EEG) recordings, the signals obtained
are contaminated predominantly by the Electrooculogram(EOG) signal. Since this
artifact has higher magnitude compared to EEG signals, these noise signals have
to be removed in order to have a better understanding regarding the functioning
of a human brain for applications such as medical diagnosis. This paper
proposes an idea of using Independent Component Analysis(ICA) along with
cross-correlation to de-noise EEG signal. This is done by selecting the
component based on the cross-correlation coefficient with a threshold value and
reducing its effect instead of zeroing it out completely, thus reducing the
information loss. The results of the recorded data show that this algorithm can
eliminate the EOG signal artifact with little loss in EEG data. The denoising
is verified by an increase in SNR value and the decrease in cross-correlation
coefficient value. The denoised signals are used to train an Artificial Neural
Network(ANN) which would examine the features of the input EEG signal and
predict the stress levels of the individual.

    

### [[2108.09837] Temporal Network Embedding via Tensor Factorization](http://arxiv.org/abs/2108.09837)


  Representation learning on static graph-structured data has shown a
significant impact on many real-world applications. However, less attention has
been paid to the evolving nature of temporal networks, in which the edges are
often changing over time. The embeddings of such temporal networks should
encode both graph-structured information and the temporally evolving pattern.
Existing approaches in learning temporally evolving network representations
fail to capture the temporal interdependence. In this paper, we propose Toffee,
a novel approach for temporal network representation learning based on tensor
decomposition. Our method exploits the tensor-tensor product operator to encode
the cross-time information, so that the periodic changes in the evolving
networks can be captured. Experimental results demonstrate that Toffee
outperforms existing methods on multiple real-world temporal networks in
generating effective embeddings for the link prediction tasks.

    

### [[2108.09847] FRUGAL: Unlocking SSL for Software Analytics](http://arxiv.org/abs/2108.09847)


  Standard software analytics often involves having a large amount of data with
labels in order to commission models with acceptable performance. However,
prior work has shown that such requirements can be expensive, taking several
weeks to label thousands of commits, and not always available when traversing
new research problems and domains. Unsupervised Learning is a promising
direction to learn hidden patterns within unlabelled data, which has only been
extensively studied in defect prediction. Nevertheless, unsupervised learning
can be ineffective by itself and has not been explored in other domains (e.g.,
static analysis and issue close time).
Motivated by this literature gap and technical limitations, we present
FRUGAL, a tuned semi-supervised method that builds on a simple optimization
scheme that does not require sophisticated (e.g., deep learners) and expensive
(e.g., 100% manually labelled data) methods. FRUGAL optimizes the unsupervised
learner's configurations (via a simple grid search) while validating our design
decision of labelling just 2.5% of the data before prediction.
As shown by the experiments of this paper FRUGAL outperforms the
state-of-the-art adoptable static code warning recognizer and issue closed time
predictor, while reducing the cost of labelling by a factor of 40 (from 100% to
2.5%). Hence we assert that FRUGAL can save considerable effort in data
labelling especially in validating prior work or researching new problems.
Based on this work, we suggest that proponents of complex and expensive
methods should always baseline such methods against simpler and cheaper
alternatives. For instance, a semi-supervised learner like FRUGAL can serve as
a baseline to the state-of-the-art software analytics.

    

### [[2108.09858] Data Augmentation Using Many-To-Many RNNs for Session-Aware Recommender Systems](http://arxiv.org/abs/2108.09858)


  The ACM WSDM WebTour 2021 Challenge organized by this http URL focuses on
applying Session-Aware recommender systems in the travel domain. Given a
sequence of travel bookings in a user trip, we look to recommend the user's
next destination. To handle the large dimensionality of the output's space, we
propose a many-to-many RNN model, predicting the next destination chosen by the
user at every sequence step as opposed to only the final one. We show how this
is a computationally efficient alternative to doing data augmentation in a
many-to-one RNN, where we consider every subsequence of a session starting from
the first element. Our solution achieved 4th place in the final leaderboard,
with an accuracy@4 of 0.5566.

    

### [[2108.09859] Convex Latent Effect Logit Model via Sparse and Low-rank Decomposition](http://arxiv.org/abs/2108.09859)


  In this paper, we propose a convex formulation for learning logistic
regression model (logit) with latent heterogeneous effect on sub-population. In
transportation, logistic regression and its variants are often interpreted as
discrete choice models under utility theory (McFadden, 2001). Two prominent
applications of logit models in the transportation domain are traffic accident
analysis and choice modeling. In these applications, researchers often want to
understand and capture the individual variation under the same accident or
choice scenario. The mixed effect logistic regression (mixed logit) is a
popular model employed by transportation researchers. To estimate the
distribution of mixed logit parameters, a non-convex optimization problem with
nested high-dimensional integrals needs to be solved. Simulation-based
optimization is typically applied to solve the mixed logit parameter estimation
problem. Despite its popularity, the mixed logit approach for learning
individual heterogeneity has several downsides. First, the parametric form of
the distribution requires domain knowledge and assumptions imposed by users,
although this issue can be addressed to some extent by using a non-parametric
approach. Second, the optimization problems arise from parameter estimation for
mixed logit and the non-parametric extensions are non-convex, which leads to
unstable model interpretation. Third, the simulation size in
simulation-assisted estimation lacks finite-sample theoretical guarantees and
is chosen somewhat arbitrarily in practice. To address these issues, we are
motivated to develop a formulation that models the latent individual
heterogeneity while preserving convexity, and avoids the need for
simulation-based approximation. Our setup is based on decomposing the
parameters into a sparse homogeneous component in the population and low-rank
heterogeneous parts for each individual.

    

### [[2108.09862] Explainable Machine Learning using Real, Synthetic and Augmented Fire Tests to Predict Fire Resistance and Spalling of RC Columns](http://arxiv.org/abs/2108.09862)


  This paper presents the development of systematic machine learning (ML)
approach to enable explainable and rapid assessment of fire resistance and
fire-induced spalling of reinforced concrete (RC) columns. The developed
approach comprises of an ensemble of three novel ML algorithms namely; random
forest (RF), extreme gradient boosted trees (ExGBT), and deep learning (DL).
These algorithms are trained to account for a wide collection of geometric
characteristics and material properties, as well as loading conditions to
examine fire performance of normal and high strength RC columns by analyzing a
comprehensive database of fire tests comprising of over 494 observations. The
developed ensemble is also capable of presenting quantifiable insights to ML
predictions; thus, breaking free from the notion of 'blackbox' ML and
establishing a solid step towards transparent and explainable ML. Most
importantly, this work tackles the scarcity of available fire tests by
proposing new techniques to leverage the use of real, synthetic and augmented
fire test observations. The developed ML ensemble has been calibrated and
validated for standard and design fire exposures and for one, two, three and
four-sided fire exposures thus; covering a wide range of practical scenarios
present during fire incidents. When fully deployed, the developed ensemble can
analyze over 5,000 RC columns in under 60 seconds thus, providing an attractive
solution for researchers and practitioners. The presented approach can also be
easily extended for evaluating fire resistance and spalling of other structural
members and under varying fire scenarios and loading conditions and hence paves
the way to modernize the state of this research area and practice.

    

### [[2108.09873] An Adversarial Learning Based Approach for Unknown View Tomographic Reconstruction](http://arxiv.org/abs/2108.09873)


  The goal of 2D tomographic reconstruction is to recover an image given its
projection lines from various views. It is often presumed that projection
angles associated with the projection lines are known in advance. Under certain
situations, however, these angles are known only approximately or are
completely unknown. It becomes more challenging to reconstruct the image from a
collection of random projection lines. We propose an adversarial learning based
approach to recover the image and the projection angle distribution by matching
the empirical distribution of the measurements with the generated data. Fitting
the distributions is achieved through solving a min-max game between a
generator and a critic based on Wasserstein generative adversarial network
structure. To accommodate the update of the projection angle distribution
through gradient back propagation, we approximate the loss using the
Gumbel-Softmax reparameterization of samples from discrete distributions. Our
theoretical analysis verifies the unique recovery of the image and the
projection distribution up to a rotation and reflection upon convergence. Our
extensive numerical experiments showcase the potential of our method to
accurately recover the image and the projection angle distribution under noise
contamination.

    

### [[2108.09875] Anarchic Federated Learning](http://arxiv.org/abs/2108.09875)


  Present-day federated learning (FL) systems deployed over edge networks have
to consistently deal with a large number of workers with high degrees of
heterogeneity in data and/or computing capabilities. This diverse set of
workers necessitates the development of FL algorithms that allow: (1) flexible
worker participation that grants the workers' capability to engage in training
at will, (2) varying number of local updates (based on computational resources)
at each worker along with asynchronous communication with the server, and (3)
heterogeneous data across workers. To address these challenges, in this work,
we propose a new paradigm in FL called ``Anarchic Federated Learning'' (AFL).
In stark contrast to conventional FL models, each worker in AFL has complete
freedom to choose i) when to participate in FL, and ii) the number of local
steps to perform in each round based on its current situation (e.g., battery
level, communication channels, privacy concerns). However, AFL also introduces
significant challenges in algorithmic design because the server needs to handle
the chaotic worker behaviors. Toward this end, we propose two Anarchic
FedAvg-like algorithms with two-sided learning rates for both cross-device and
cross-silo settings, which are named AFedAvg-TSLR-CD and AFedAvg-TSLR-CS,
respectively. For general worker information arrival processes, we show that
both algorithms retain the highly desirable linear speedup effect in the new
AFL paradigm. Moreover, we show that our AFedAvg-TSLR algorithmic framework can
be viewed as a {\em meta-algorithm} for AFL in the sense that they can utilize
advanced FL algorithms as worker- and/or server-side optimizers to achieve
enhanced performance under AFL. We validate the proposed algorithms with
extensive experiments on real-world datasets.

    

### [[2108.09876] On Quantifying Literals in Boolean Logic and Its Applications to Explainable AI](http://arxiv.org/abs/2108.09876)


  Quantified Boolean logic results from adding operators to Boolean logic for
existentially and universally quantifying variables. This extends the reach of
Boolean logic by enabling a variety of applications that have been explored
over the decades. The existential quantification of literals (variable states)
and its applications have also been studied in the literature. In this paper,
we complement this by studying universal literal quantification and its
applications, particularly to explainable AI. We also provide a novel semantics
for quantification, discuss the interplay between variable/literal and
existential/universal quantification. We further identify some classes of
Boolean formulas and circuits on which quantification can be done efficiently.
Literal quantification is more fine-grained than variable quantification as the
latter can be defined in terms of the former. This leads to a refinement of
quantified Boolean logic with literal quantification as its primitive.

    

### [[2108.09885] DTWSSE: Data Augmentation with a Siamese Encoder for Time Series](http://arxiv.org/abs/2108.09885)


  Access to labeled time series data is often limited in the real world, which
constrains the performance of deep learning models in the field of time series
analysis. Data augmentation is an effective way to solve the problem of small
sample size and imbalance in time series datasets. The two key factors of data
augmentation are the distance metric and the choice of interpolation method.
SMOTE does not perform well on time series data because it uses a Euclidean
distance metric and interpolates directly on the object. Therefore, we propose
a DTW-based synthetic minority oversampling technique using siamese encoder for
interpolation named DTWSSE. In order to reasonably measure the distance of the
time series, DTW, which has been verified to be an effective method forts, is
employed as the distance metric. To adapt the DTW metric, we use an autoencoder
trained in an unsupervised self-training manner for interpolation. The encoder
is a Siamese Neural Network for mapping the time series data from the DTW
hidden space to the Euclidean deep feature space, and the decoder is used to
map the deep feature space back to the DTW hidden space. We validate the
proposed methods on a number of different balanced or unbalanced time series
datasets. Experimental results show that the proposed method can lead to better
performance of the downstream deep learning model.

    

### [[2108.09896] Generative and Contrastive Self-Supervised Learning for Graph Anomaly Detection](http://arxiv.org/abs/2108.09896)


  Anomaly detection from graph data has drawn much attention due to its
practical significance in many critical applications including cybersecurity,
finance, and social networks. Existing data mining and machine learning methods
are either shallow methods that could not effectively capture the complex
interdependency of graph data or graph autoencoder methods that could not fully
exploit the contextual information as supervision signals for effective anomaly
detection. To overcome these challenges, in this paper, we propose a novel
method, Self-Supervised Learning for Graph Anomaly Detection (SL-GAD). Our
method constructs different contextual subgraphs (views) based on a target node
and employs two modules, generative attribute regression and multi-view
contrastive learning for anomaly detection. While the generative attribute
regression module allows us to capture the anomalies in the attribute space,
the multi-view contrastive learning module can exploit richer structure
information from multiple subgraphs, thus abling to capture the anomalies in
the structure space, mixing of structure, and attribute information. We conduct
extensive experiments on six benchmark datasets and the results demonstrate
that our method outperforms state-of-the-art methods by a large margin.

    

### [[2108.09898] Face Photo-Sketch Recognition Using Bidirectional Collaborative Synthesis Network](http://arxiv.org/abs/2108.09898)


  This research features a deep-learning based framework to address the problem
of matching a given face sketch image against a face photo database. The
problem of photo-sketch matching is challenging because 1) there is large
modality gap between photo and sketch, and 2) the number of paired training
samples is insufficient to train deep learning based networks. To circumvent
the problem of large modality gap, our approach is to use an intermediate
latent space between the two modalities. We effectively align the distributions
of the two modalities in this latent space by employing a bidirectional (photo
-> sketch and sketch -> photo) collaborative synthesis network. A StyleGAN-like
architecture is utilized to make the intermediate latent space be equipped with
rich representation power. To resolve the problem of insufficient training
samples, we introduce a three-step training scheme. Extensive evaluation on
public composite face sketch database confirms superior performance of our
method compared to existing state-of-the-art methods. The proposed methodology
can be employed in matching other modality pairs.

    

### [[2108.09914] Genetic Programming for Manifold Learning: Preserving Local Topology](http://arxiv.org/abs/2108.09914)


  Manifold learning methods are an invaluable tool in today's world of
increasingly huge datasets. Manifold learning algorithms can discover a much
lower-dimensional representation (embedding) of a high-dimensional dataset
through non-linear transformations that preserve the most important structure
of the original data. State-of-the-art manifold learning methods directly
optimise an embedding without mapping between the original space and the
discovered embedded space. This makes interpretability - a key requirement in
exploratory data analysis - nearly impossible. Recently, genetic programming
has emerged as a very promising approach to manifold learning by evolving
functional mappings from the original space to an embedding. However, genetic
programming-based manifold learning has struggled to match the performance of
other approaches. In this work, we propose a new approach to using genetic
programming for manifold learning, which preserves local topology. This is
expected to significantly improve performance on tasks where local
neighbourhood structure (topology) is paramount. We compare our proposed
approach with various baseline manifold learning methods and find that it often
outperforms other methods, including a clear improvement over previous genetic
programming approaches. These results are particularly promising, given the
potential interpretability and reusability of the evolved mappings.

    

### [[2108.09918] Fluent: An AI Augmented Writing Tool for People who Stutter](http://arxiv.org/abs/2108.09918)


  Stuttering is a speech disorder which impacts the personal and professional
lives of millions of people worldwide. To save themselves from stigma and
discrimination, people who stutter (PWS) may adopt different strategies to
conceal their stuttering. One of the common strategies is word substitution
where an individual avoids saying a word they might stutter on and use an
alternative instead. This process itself can cause stress and add more burden.
In this work, we present Fluent, an AI augmented writing tool which assists PWS
in writing scripts which they can speak more fluently. Fluent embodies a novel
active learning based method of identifying words an individual might struggle
pronouncing. Such words are highlighted in the interface. On hovering over any
such word, Fluent presents a set of alternative words which have similar
meaning but are easier to speak. The user is free to accept or ignore these
suggestions. Based on such user interaction (feedback), Fluent continuously
evolves its classifier to better suit the personalized needs of each user. We
evaluated our tool by measuring its ability to identify difficult words for 10
simulated users. We found that our tool can identify difficult words with a
mean accuracy of over 80% in under 20 interactions and it keeps improving with
more feedback. Our tool can be beneficial for certain important life situations
like giving a talk, presentation, etc. The source code for this tool has been
made publicly accessible at this http URL.

    

### [[2108.09922] Subject Envelope based Multitype Reconstruction Algorithm of Speech Samples of Parkinson's Disease](http://arxiv.org/abs/2108.09922)


  The risk of Parkinson's disease (PD) is extremely serious, and PD speech
recognition is an effective method of diagnosis nowadays. However, due to the
influence of the disease stage, corpus, and other factors on data collection,
the ability of every samples within one subject to reflect the status of PD
vary. No samples are useless totally, and not samples are 100% perfect. This
characteristic means that it is not suitable just to remove some samples or
keep some samples. It is necessary to consider the sample transformation for
obtaining high quality new samples. Unfortunately, existing PD speech
recognition methods focus mainly on feature learning and classifier design
rather than sample learning, and few methods consider the sample
transformation. To solve the problem above, a PD speech sample transformation
algorithm based on multitype reconstruction operators is proposed in this
paper. The algorithm is divided into four major steps. Three types of
reconstruction operators are designed in the algorithm: types A, B and C.
Concerning the type A operator, the original dataset is directly reconstructed
by designing a linear transformation to obtain the first dataset. The type B
operator is designed for clustering and linear transformation of the dataset to
obtain the second new dataset. The third operator, namely, the type C operator,
reconstructs the dataset by clustering and convolution to obtain the third
dataset. Finally, the base classifier is trained based on the three new
datasets, and then the classification results are fused by decision weighting.
In the experimental section, two representative PD speech datasets are used for
verification. The results show that the proposed algorithm is effective.
Compared with other algorithms, the proposed algorithm achieves apparent
improvements in terms of classification accuracy.

    

### [[2108.09923] Convolutional Filtering and Neural Networks with Non Commutative Algebras](http://arxiv.org/abs/2108.09923)


  In this paper we provide stability results for algebraic neural networks
(AlgNNs) based on non commutative algebras. AlgNNs are stacked layered
structures with each layer associated to an algebraic signal model (ASM)
determined by an algebra, a vector space, and a homomorphism. Signals are
modeled as elements of the vector space, filters are elements in the algebra,
while the homomorphism provides a realization of the filters as concrete
operators. We study the stability of the algebraic filters in non commutative
algebras to perturbations on the homomorphisms, and we provide conditions under
which stability is guaranteed. We show that the commutativity between shift
operators and between shifts and perturbations does not affect the property of
an architecture of being stable. This provides an answer to the question of
whether shift invariance was a necessary attribute of convolutional
architectures to guarantee stability. Additionally, we show that although the
frequency responses of filters in non commutative algebras exhibit substantial
differences with respect to filters in commutative algebras, their derivatives
for stable filters have a similar behavior.

    

### [[2108.09924] Sarcasm Detection in Twitter -- Performance Impact when using Data Augmentation: Word Embeddings](http://arxiv.org/abs/2108.09924)


  Sarcasm is the use of words usually used to either mock or annoy someone, or
for humorous purposes. Sarcasm is largely used in social networks and
microblogging websites, where people mock or censure in a way that makes it
difficult even for humans to tell if what is said is what is meant. Failure to
identify sarcastic utterances in Natural Language Processing applications such
as sentiment analysis and opinion mining will confuse classification algorithms
and generate false results. Several studies on sarcasm detection have utilized
different learning algorithms. However, most of these learning models have
always focused on the contents of expression only, leaving the contextual
information in isolation. As a result, they failed to capture the contextual
information in the sarcastic expression. Moreover, some datasets used in
several studies have an unbalanced dataset which impacting the model result. In
this paper, we propose a contextual model for sarcasm identification in twitter
using RoBERTa, and augmenting the dataset by applying Global Vector
representation (GloVe) for the construction of word embedding and context
learning to generate more data and balancing the dataset. The effectiveness of
this technique is tested with various datasets and data augmentation settings.
In particular, we achieve performance gain by 3.2% in the iSarcasm dataset when
using data augmentation to increase 20% of data labeled as sarcastic, resulting
F-score of 40.4% compared to 37.2% without data augmentation.

    

### [[2108.09926] APObind: A Dataset of Ligand Unbound Protein Conformations for Machine Learning Applications in De Novo Drug Design](http://arxiv.org/abs/2108.09926)


  Protein-ligand complex structures have been utilised to design benchmark
machine learning methods that perform important tasks related to drug design
such as receptor binding site detection, small molecule docking and binding
affinity prediction. However, these methods are usually trained on only ligand
bound (or holo) conformations of the protein and therefore are not guaranteed
to perform well when the protein structure is in its native unbound
conformation (or apo), which is usually the conformation available for a newly
identified receptor. A primary reason for this is that the local structure of
the binding site usually changes upon ligand binding. To facilitate solutions
for this problem, we propose a dataset called APObind that aims to provide apo
conformations of proteins present in the PDBbind dataset, a popular dataset
used in drug design. Furthermore, we explore the performance of methods
specific to three use cases on this dataset, through which, the importance of
validating them on the APObind dataset is demonstrated.

    

### [[2108.09932] Federated Learning Meets Fairness and Differential Privacy](http://arxiv.org/abs/2108.09932)


  Deep learning's unprecedented success raises several ethical concerns ranging
from biased predictions to data privacy. Researchers tackle these issues by
introducing fairness metrics, or federated learning, or differential privacy. A
first, this work presents an ethical federated learning model, incorporating
all three measures simultaneously. Experiments on the Adult, Bank and Dutch
datasets highlight the resulting ``empirical interplay" between accuracy,
fairness, and privacy.

    

### [[2108.09976] Revealing Distributional Vulnerability of Explicit Discriminators by Implicit Generators](http://arxiv.org/abs/2108.09976)


  An explicit discriminator trained on observable in-distribution (ID) samples
can make high-confidence prediction on out-of-distribution (OOD) samples due to
its distributional vulnerability. This is primarily caused by the limited ID
samples observable for training discriminators when OOD samples are
unavailable. To address this issue, the state-of-the-art methods train the
discriminator with OOD samples generated by general assumptions without
considering the data and network characteristics. However, different network
architectures and training ID datasets may cause diverse vulnerabilities, and
the generated OOD samples thus usually misaddress the specific distributional
vulnerability of the explicit discriminator. To reveal and patch the
distributional vulnerabilities, we propose a novel method of
\textit{fine-tuning explicit discriminators by implicit generators} (FIG).
According to the Shannon entropy, an explicit discriminator can construct its
corresponding implicit generator to generate specific OOD samples without extra
training costs. A Langevin Dynamic sampler then draws high-quality OOD samples
from the generator to reveal the vulnerability. Finally, a regularizer,
constructed according to the design principle of the implicit generator,
patches the distributional vulnerability by encouraging those generated OOD
samples with high entropy. Our experiments on four networks, four ID datasets
and seven OOD datasets demonstrate that FIG achieves state-of-the-art OOD
detection performance and maintains a competitive classification capability.

    

### [[2108.09980] TACo: Token-aware Cascade Contrastive Learning for Video-Text Alignment](http://arxiv.org/abs/2108.09980)


  Contrastive learning has been widely used to train transformer-based
vision-language models for video-text alignment and multi-modal representation
learning. This paper presents a new algorithm called Token-Aware Cascade
contrastive learning (TACo) that improves contrastive learning using two novel
techniques. The first is the token-aware contrastive loss which is computed by
taking into account the syntactic classes of words. This is motivated by the
observation that for a video-text pair, the content words in the text, such as
nouns and verbs, are more likely to be aligned with the visual contents in the
video than the function words. Second, a cascade sampling method is applied to
generate a small set of hard negative examples for efficient loss estimation
for multi-modal fusion layers. To validate the effectiveness of TACo, in our
experiments we finetune pretrained models for a set of downstream tasks
including text-video retrieval (YouCook2, MSR-VTT and ActivityNet), video
action step localization (CrossTask), video action segmentation (COIN). The
results show that our models attain consistent improvements across different
experimental settings over previous methods, setting new state-of-the-art on
three public text-video retrieval benchmarks of YouCook2, MSR-VTT and
ActivityNet.

    

### [[2108.09992] Learned Image Coding for Machines: A Content-Adaptive Approach](http://arxiv.org/abs/2108.09992)


  Today, according to the Cisco Annual Internet Report (2018-2023), the
fastest-growing category of Internet traffic is machine-to-machine
communication. In particular, machine-to-machine communication of images and
videos represents a new challenge and opens up new perspectives in the context
of data compression. One possible solution approach consists of adapting
current human-targeted image and video coding standards to the use case of
machine consumption. Another approach consists of developing completely new
compression paradigms and architectures for machine-to-machine communications.
In this paper, we focus on image compression and present an inference-time
content-adaptive finetuning scheme that optimizes the latent representation of
an end-to-end learned image codec, aimed at improving the compression
efficiency for machine-consumption. The conducted experiments show that our
online finetuning brings an average bitrate saving (BD-rate) of -3.66% with
respect to our pretrained image codec. In particular, at low bitrate points,
our proposed method results in a significant bitrate saving of -9.85%. Overall,
our pretrained-and-then-finetuned system achieves -30.54% BD-rate over the
state-of-the-art image/video codec Versatile Video Coding (VVC).

    

### [[2108.09993] Image coding for machines: an end-to-end learned approach](http://arxiv.org/abs/2108.09993)


  Over recent years, deep learning-based computer vision systems have been
applied to images at an ever-increasing pace, oftentimes representing the only
type of consumption for those images. Given the dramatic explosion in the
number of images generated per day, a question arises: how much better would an
image codec targeting machine-consumption perform against state-of-the-art
codecs targeting human-consumption? In this paper, we propose an image codec
for machines which is neural network (NN) based and end-to-end learned. In
particular, we propose a set of training strategies that address the delicate
problem of balancing competing loss functions, such as computer vision task
losses, image distortion losses, and rate loss. Our experimental results show
that our NN-based codec outperforms the state-of-the-art Versa-tile Video
Coding (VVC) standard on the object detection and instance segmentation tasks,
achieving -37.87% and -32.90% of BD-rate gain, respectively, while being fast
thanks to its compact size. To the best of our knowledge, this is the first
end-to-end learned machine-targeted image codec.

    

### [[2108.10004] Relative Entropy-Regularized Optimal Transport on a Graph: a new algorithm and an experimental comparison](http://arxiv.org/abs/2108.10004)


  Following [21, 23], the present work investigates a new relative
entropy-regularized algorithm for solving the optimal transport on a graph
problem within the randomized shortest paths formalism. More precisely, a unit
flow is injected into a set of input nodes and collected from a set of output
nodes while minimizing the expected transportation cost together with a paths
relative entropy regularization term, providing a randomized routing policy.
The main advantage of this new formulation is the fact that it can easily
accommodate edge flow capacity constraints which commonly occur in real-world
problems. The resulting optimal routing policy, i.e., the probability
distribution of following an edge in each node, is Markovian and is computed by
constraining the input and output flows to the prescribed marginal
probabilities thanks to a variant of the algorithm developed in [8]. Besides,
experimental comparisons with other recently developed techniques show that the
distance measure between nodes derived from the introduced model provides
competitive results on semi-supervised classification tasks.

    

### [[2108.10026] Deep Relational Metric Learning](http://arxiv.org/abs/2108.10026)


  This paper presents a deep relational metric learning (DRML) framework for
image clustering and retrieval. Most existing deep metric learning methods
learn an embedding space with a general objective of increasing interclass
distances and decreasing intraclass distances. However, the conventional losses
of metric learning usually suppress intraclass variations which might be
helpful to identify samples of unseen classes. To address this problem, we
propose to adaptively learn an ensemble of features that characterizes an image
from different aspects to model both interclass and intraclass distributions.
We further employ a relational module to capture the correlations among each
feature in the ensemble and construct a graph to represent an image. We then
perform relational inference on the graph to integrate the ensemble and obtain
a relation-aware embedding to measure the similarities. Extensive experiments
on the widely-used CUB-200-2011, Cars196, and Stanford Online Products datasets
demonstrate that our framework improves existing deep metric learning methods
and achieves very competitive results.

    

### [[2108.10029] Modeling COVID-19 uncertainties evolving over time and density-dependent social reinforcement and asymptomatic infections](http://arxiv.org/abs/2108.10029)


  The novel coronavirus disease 2019 (COVID-19) presents unique and unknown
problem complexities and modeling challenges, where an imperative task is to
model both its process and data uncertainties, represented in implicit and
high-proportional undocumented infections, asymptomatic contagion, social
reinforcement of infections, and various quality issues in the reported data.
These uncertainties become even more phenomenal in the overwhelming
mutation-dominated resurgences with vaccinated but still susceptible
populations. Here we introduce a novel hybrid approach to (1) characterizing
and distinguishing Undocumented (U) and Documented (D) infections commonly seen
during COVID-19 incubation periods and asymptomatic infections by expanding the
foundational compartmental epidemic Susceptible-Infected-Recovered (SIR) model
with two compartments, resulting in a new Susceptible-Undocumented
infected-Documented infected-Recovered (SUDR) model; (2) characterizing the
probabilistic density of infections by empowering SUDR to capture exogenous
processes like clustering contagion interactions, superspreading and social
reinforcement; and (3) approximating the density likelihood of COVID-19
prevalence over time by incorporating Bayesian inference into SUDR. Different
from existing COVID-19 models, SUDR characterizes the undocumented infections
during unknown transmission processes. To capture the uncertainties of temporal
transmission and social reinforcement during the COVID-19 contagion, the
transmission rate is modeled by a time-varying density function of undocumented
infectious cases. We solve the modeling by sampling from the mean-field
posterior distribution with reasonable priors, making SUDR suitable to handle
the randomness, noise and sparsity of COVID-19 observations widely seen in the
public COVID-19 case data.

    

### [[2108.10037] Primal and Dual Combinatorial Dimensions](http://arxiv.org/abs/2108.10037)


  We give tight bounds on the relation between the primal and dual of various
combinatorial dimensions, such as the pseudo-dimension and fat-shattering
dimension, for multi-valued function classes. These dimensional notions play an
important role in the area of learning theory. We first review some (folklore)
results that bound the dual dimension of a function class in terms of its
primal, and after that give (almost) matching lower bounds. In particular, we
give an appropriate generalization to multi-valued function classes of a
well-known bound due to Assouad (1983), that relates the primal and dual
VC-dimension of a binary function class.

    

### [[2108.10052] Integrating LSTMs and GNNs for COVID-19 Forecasting](http://arxiv.org/abs/2108.10052)


  The spread of COVID-19 has coincided with the rise of Graph Neural Networks
(GNNs), leading to several studies proposing their use to better forecast the
evolution of the pandemic. Many such models also include Long Short Term Memory
(LSTM) networks, a common tool for time series forecasting. In this work, we
further investigate the integration of these two methods by implementing GNNs
within the gates of an LSTM and exploiting spatial information. In addition, we
introduce a skip connection which proves critical to jointly capture the
spatial and temporal patterns in the data. We validate our daily COVID-19 new
cases forecast model on data of 37 European nations for the last 472 days and
show superior performance compared to state-of-the-art graph time series models
based on mean absolute scaled error (MASE). This area of research has important
applications to policy-making and we analyze its potential for pandemic
resource control.

    

### [[2108.10054] Remote Sensing and Machine Learning for Food Crop Production Data in Africa Post-COVID-19](http://arxiv.org/abs/2108.10054)


  In the agricultural sector, the COVID-19 threatens to lead to a severe food
security crisis in the region, with disruptions in the food supply chain and
agricultural production expected to contract between 2.6% and 7%. From the food
crop production side, the travel bans and border closures, the late reception
and the use of agricultural inputs such as imported seeds, fertilizers, and
pesticides could lead to poor food crop production performances. Another layer
of disruption introduced by the mobility restriction measures is the scarcity
of agricultural workers, mainly seasonal workers. The lockdown measures and
border closures limit seasonal workers' availability to get to the farm on time
for planting and harvesting activities. Moreover, most of the imported
agricultural inputs travel by air, which the pandemic has heavily impacted.
Such transportation disruptions can also negatively affect the food crop
production system.
This chapter assesses food crop production levels in 2020 -- before the
harvesting period -- in all African regions and four staples such as maize,
cassava, rice, and wheat. The production levels are predicted using the
combination of biogeophysical remote sensing data retrieved from satellite
images and machine learning artificial neural networks (ANNs) technique. The
remote sensing products are used as input variables and the ANNs as the
predictive modeling framework. The input remote sensing products are the
Normalized Difference Vegetation Index (NDVI), the daytime Land Surface
Temperature (LST), rainfall data, and agricultural lands' Evapotranspiration
(ET). The output maps and data are made publicly available on a web-based
platform, AAgWa (Africa Agriculture Watch, this http URL), to facilitate access
to such information to policymakers, deciders, and other stakeholders.

    

### [[2108.10061] An Extensible and Modular Design and Implementation of Monte Carlo Tree Search for the JVM](http://arxiv.org/abs/2108.10061)


  Flexible implementations of Monte Carlo Tree Search (MCTS), combined with
domain specific knowledge and hybridization with other search algorithms, can
be powerful for finding the solutions to problems in complex planning. We
introduce mctreesearch4j, an MCTS implementation written as a standard JVM
library following key design principles of object oriented programming. We
define key class abstractions allowing the MCTS library to flexibly adapt to
any well defined Markov Decision Process or turn-based adversarial game.
Furthermore, our library is designed to be modular and extensible, utilizing
class inheritance and generic typing to standardize custom algorithm
definitions. We demonstrate that the design of the MCTS implementation provides
ease of adaptation for unique heuristics and customization across varying
Markov Decision Process (MDP) domains. In addition, the implementation is
reasonably performant and accurate for standard MDP's. In addition, via the
implementation of mctreesearch4j, the nuances of different types of MCTS
algorithms are discussed.

    

### [[2108.10064] Effective and Privacy preserving Tabular Data Synthesizing](http://arxiv.org/abs/2108.10064)


  While data sharing is crucial for knowledge development, privacy concerns and
strict regulation (e.g., European General Data Protection Regulation (GDPR))
unfortunately limits its full effectiveness. Synthetic tabular data emerges as
an alternative to enable data sharing while fulfilling regulatory and privacy
constraints. The state-of-the-art tabular data synthesizers draw methodologies
from Generative Adversarial Networks (GAN). In this thesis, we develop
CTAB-GAN, a novel conditional table GAN architecture that can effectively model
diverse data types with complex distributions. CTAB-GAN is extensively
evaluated with the state of the art GANs that generate synthetic tables, in
terms of data similarity and analysis utility. The results on five datasets
show that the synthetic data of CTAB-GAN remarkably resembles the real data for
all three types of variables and results in higher accuracy for five machine
learning algorithms, by up to 17%.
Additionally, to ensure greater security for training tabular GANs against
malicious privacy attacks, differential privacy (DP) is studied and used to
train CTAB-GAN with strict privacy guarantees. DP-CTAB-GAN is rigorously
evaluated using state-of-the-art DP-tabular GANs in terms of data utility and
privacy robustness against membership and attribute inference attacks. Our
results on three datasets indicate that strict theoretical differential privacy
guarantees come only after severely affecting data utility. However, it is
shown empirically that these guarantees help provide a stronger defence against
privacy attacks. Overall, it is found that DP-CTABGAN is capable of being
robust to privacy attacks while maintaining the highest data utility as
compared to prior work, by up to 18% in terms of the average precision score.

    

### [[2108.10066] Dynamic Neural Network Architectural and Topological Adaptation and Related Methods -- A Survey](http://arxiv.org/abs/2108.10066)


  Training and inference in deep neural networks (DNNs) has, due to a steady
increase in architectural complexity and data set size, lead to the development
of strategies for reducing time and space requirements of DNN training and
inference, which is of particular importance in scenarios where training takes
place in resource constrained computation environments or inference is part of
a time critical application. In this survey, we aim to provide a general
overview and categorization of state-of-the-art (SOTA) of techniques to reduced
DNN training and inference time and space complexities with a particular focus
on architectural adaptions.

    

### [[1803.11521] A Novel Framework for Online Supervised Learning with Feature Selection](http://arxiv.org/abs/1803.11521)


  Current online learning methods suffer issues such as lower convergence rates
and limited capability to recover the support of the true features compared to
their offline counterparts. In this paper, we present a novel framework for
online learning based on running averages and introduce a series of online
versions of some popular existing offline methods such as Elastic Net, Minimax
Concave Penalty and Feature Selection with Annealing. The framework can handle
an arbitrarily large number of observations with the restriction that the data
dimension is not too large, e.g. $p<50,000$. We prove the equivalence between
our online methods and their offline counterparts and give theoretical true
feature recovery and convergence guarantees for some of them. In contrast to
the existing online methods, the proposed methods can extract models with any
desired sparsity level at any time. Numerical experiments indicate that our new
methods enjoy high accuracy of true feature recovery and a fast convergence
rate, compared with standard online and offline algorithms. We also show how
the running averages framework can be used for model adaptation in the presence
of model drift. Finally, we present some applications to large datasets where
again the proposed framework shows competitive results compared to popular
online and offline algorithms.

    

### [[1902.00440] Spatial-Temporal-Textual Point Processes for Crime Linkage Detection](http://arxiv.org/abs/1902.00440)


  Crimes emerge out of complex interactions of human behaviors and situations.
Linkages between crime incidents are highly complex. Detecting crime linkage
given a set of incidents is a highly challenging task since we only have
limited information, including text descriptions, incident times, and
locations. In practice, there are very few labels. We propose a new statistical
modeling framework for {\it spatio-temporal-textual} data and demonstrate its
usage on crime linkage detection. We capture linkages of crime incidents via
multivariate marked spatio-temporal Hawkes processes and treat embedding
vectors of the free-text as {\it marks} of the incident, inspired by the notion
of {\it modus operandi} (M.O.) in crime analysis. Numerical results using real
data demonstrate the good performance of our method as well as reveals
interesting patterns in the crime data: the joint modeling of space, time, and
text information enhances crime linkage detection compared with the
state-of-the-art, and the learned spatial dependence from data can be useful
for police operations.

    

### [[1903.02152] Safeguarded Dynamic Label Regression for Generalized Noisy Supervision](http://arxiv.org/abs/1903.02152)


  Learning with noisy labels, which aims to reduce expensive labors on accurate
annotations, has become imperative in the Big Data era. Previous noise
transition based method has achieved promising results and presented a
theoretical guarantee on performance in the case of class-conditional noise.
However, this type of approaches critically depend on an accurate
pre-estimation of the noise transition, which is usually impractical.
Subsequent improvement adapts the pre-estimation along with the training
progress via a Softmax layer. However, the parameters in the Softmax layer are
highly tweaked for the fragile performance due to the ill-posed stochastic
approximation. To address these issues, we propose a Latent Class-Conditional
Noise model (LCCN) that naturally embeds the noise transition under a Bayesian
framework. By projecting the noise transition into a Dirichlet-distributed
space, the learning is constrained on a simplex based on the whole dataset,
instead of some ad-hoc parametric space. We then deduce a dynamic label
regression method for LCCN to iteratively infer the latent labels, to
stochastically train the classifier and to model the noise. Our approach
safeguards the bounded update of the noise transition, which avoids previous
arbitrarily tuning via a batch of samples. We further generalize LCCN for
open-set noisy labels and the semi-supervised setting. We perform extensive
experiments with the controllable noise data sets, CIFAR-10 and CIFAR-100, and
the agnostic noise data sets, Clothing1M and WebVision17. The experimental
results have demonstrated that the proposed model outperforms several
state-of-the-art methods.

    

### [[1908.01672] Imbalance-XGBoost: Leveraging Weighted and Focal Losses for Binary Label-Imbalanced Classification with XGBoost](http://arxiv.org/abs/1908.01672)


  The paper presents Imbalance-XGBoost, a Python package that combines the
powerful XGBoost software with weighted and focal losses to tackle binary
label-imbalanced classification tasks. Though a small-scale program in terms of
size, the package is, to the best of the authors' knowledge, the first of its
kind which provides an integrated implementation for the two losses on XGBoost
and brings a general-purpose extension on XGBoost for label-imbalanced
scenarios. In this paper, the design and usage of the package are described
with exemplar code listings, and its convenience to be integrated into
Python-driven Machine Learning projects is illustrated. Furthermore, as the
first- and second-order derivatives of the loss functions are essential for the
implementations, the algebraic derivation is discussed and it can be deemed as
a separate algorithmic contribution. The performances of the algorithms
implemented in the package are empirically evaluated on Parkinson's disease
classification data set, and multiple state-of-the-art performances have been
observed. Given the scalable nature of XGBoost, the package has great
potentials to be applied to real-life binary classification tasks, which are
usually of large-scale and label-imbalanced.

    

### [[1911.10454] Regularized and Smooth Double Core Tensor Factorization for Heterogeneous Data](http://arxiv.org/abs/1911.10454)


  We introduce a general tensor model suitable for data analytic tasks for {\em
heterogeneous} datasets, wherein there are joint low-rank structures within
groups of observations, but also discriminative structures across different
groups. To capture such complex structures, a double core tensor (DCOT)
factorization model is introduced together with a family of smoothing loss
functions. By leveraging the proposed smoothing function, the model accurately
estimates the model factors, even in the presence of missing entries. A
linearized ADMM method is employed to solve regularized versions of DCOT
factorizations, that avoid large tensor operations and large memory storage
requirements. Further, we establish theoretically its global convergence,
together with consistency of the estimates of the model parameters. The
effectiveness of the DCOT model is illustrated on several real-world examples
including image completion, recommender systems, subspace clustering and
detecting modules in heterogeneous Omics multi-modal data, since it provides
more insightful decompositions than conventional tensor methods.

    

### [[1912.10325] Online Reinforcement Learning of Optimal Threshold Policies for Markov Decision Processes](http://arxiv.org/abs/1912.10325)


  To overcome the curses of dimensionality and modeling of Dynamic Programming
(DP) methods to solve Markov Decision Process (MDP) problems, Reinforcement
Learning (RL) methods are adopted in practice. Contrary to traditional RL
algorithms which do not consider the structural properties of the optimal
policy, we propose a structure-aware learning algorithm to exploit the ordered
multi-threshold structure of the optimal policy, if any. We prove the
asymptotic convergence of the proposed algorithm to the optimal policy. Due to
the reduction in the policy space, the proposed algorithm provides remarkable
improvements in storage and computational complexities over classical RL
algorithms. Simulation results establish that the proposed algorithm converges
faster than other RL algorithms.

    

### [[2002.01927] Self-Directed Online Machine Learning for Topology Optimization](http://arxiv.org/abs/2002.01927)


  Topology optimization by optimally distributing materials in a given domain
requires gradient-free optimizers to solve highly complicated problems.
However, with hundreds of design variables or more involved, solving such
problems would require millions of Finite Element Method (FEM) calculations
whose computational cost is huge and impractical. Here we report Self-directed
Online Learning Optimization (SOLO) which integrates Deep Neural Network (DNN)
with FEM calculations. A DNN learns and substitutes the objective as a function
of design variables. A small number of training data is generated dynamically
based on the DNN's prediction of the global optimum. The DNN adapts to the new
training data and gives better prediction in the region of interest until
convergence. Our algorithm was tested by four types of problems including
compliance minimization, fluid-structure optimization, heat transfer
enhancement and truss optimization. It reduced the computational time by 2 ~ 5
orders of magnitude compared with directly using heuristic methods, and
outperformed all state-of-the-art algorithms tested in our experiments. This
approach enables solving large multi-dimensional optimization problems.

    

### [[2003.04549] Slice Tuner: A Selective Data Acquisition Framework for Accurate and Fair Machine Learning Models](http://arxiv.org/abs/2003.04549)


  As machine learning becomes democratized in the era of Software 2.0, a
serious bottleneck is acquiring enough data to ensure accurate and fair models.
Recent techniques including crowdsourcing provide cost-effective ways to gather
such data. However, simply acquiring data as much as possible is not
necessarily an effective strategy for optimizing accuracy and fairness. For
example, if an online app store has enough training data for certain slices of
data (say American customers), but not for others, obtaining more American
customer data will only bias the model training. Instead, we contend that one
needs to selectively acquire data and propose Slice Tuner, which acquires
possibly-different amounts of data per slice such that the model accuracy and
fairness on all slices are optimized. This problem is different than labeling
existing data (as in active learning or weak supervision) because the goal is
obtaining the right amounts of new data. At its core, Slice Tuner maintains
learning curves of slices that estimate the model accuracies given more data
and uses convex optimization to find the best data acquisition strategy. The
key challenges of estimating learning curves are that they may be inaccurate if
there is not enough data, and there may be dependencies among slices where
acquiring data for one slice influences the learning curves of others. We solve
these issues by iteratively and efficiently updating the learning curves as
more data is acquired. We evaluate Slice Tuner on real datasets using
crowdsourcing for data acquisition and show that Slice Tuner significantly
outperforms baselines in terms of model accuracy and fairness, even when the
learning curves cannot be reliably estimated.

    

### [[2004.12908] Omnidirectional Transfer for Quasilinear Lifelong Learning](http://arxiv.org/abs/2004.12908)


  In biological learning, data are used to improve performance not only on the
current task, but also on previously encountered and as yet unencountered
tasks. In contrast, classical machine learning starts from a blank slate, or
tabula rasa, using data only for the single task at hand. While typical
transfer learning algorithms can improve performance on future tasks, their
performance on prior tasks degrades upon learning new tasks (called
catastrophic forgetting). Many recent approaches for continual or lifelong
learning have attempted to maintain performance given new tasks. But striving
to avoid forgetting sets the goal unnecessarily low: the goal of lifelong
learning, whether biological or artificial, should be to improve performance on
all tasks (including past and future) with any new data. We propose
omnidirectional transfer learning algorithms, which includes two special cases
of interest: decision forests and deep networks. Our key insight is the
development of the omni-voter layer, which ensembles representations learned
independently on all tasks to jointly decide how to proceed on any given new
data point, thereby improving performance on both past and future tasks. Our
algorithms demonstrate omnidirectional transfer in a variety of simulated and
real data scenarios, including tabular data, image data, spoken data, and
adversarial tasks. Moreover, they do so with quasilinear space and time
complexity.

    

### [[2005.08088] Learning and Optimization with Seasonal Patterns](http://arxiv.org/abs/2005.08088)


  A standard assumption adopted in the multi-armed bandit (MAB) framework is
that the mean rewards are constant over time. This assumption can be
restrictive in the business world as decision-makers often face an evolving
environment where the mean rewards are time-varying. In this paper, we consider
a non-stationary MAB model with $K$ arms whose mean rewards vary over time in a
periodic manner. The unknown periods can be different across arms and scale
with the length of the horizon $T$ polynomially. We propose a two-stage policy
that combines the Fourier analysis with a confidence-bound-based learning
procedure to learn the periods and minimize the regret. In stage one, the
policy correctly estimates the periods of all arms with high probability. In
stage two, the policy explores the periodic mean rewards of arms using the
periods estimated in stage one and exploits the optimal arm in the long run. We
show that our learning policy incurs a regret upper bound
$\tilde{O}(\sqrt{T\sum_{k=1}^K T_k})$ where $T_k$ is the period of arm $k$.
Moreover, we establish a general lower bound $\Omega(\sqrt{T\max_{k}\{ T_k\}})$
for any policy. Therefore, our policy is near-optimal up to a factor of
$\sqrt{K}$.

    

### [[2006.03184] Pick-Object-Attack: Type-Specific Adversarial Attack for Object Detection](http://arxiv.org/abs/2006.03184)


  Many recent studies have shown that deep neural models are vulnerable to
adversarial samples: images with imperceptible perturbations, for example, can
fool image classifiers. In this paper, we present the first type-specific
approach to generating adversarial examples for object detection, which entails
detecting bounding boxes around multiple objects present in the image and
classifying them at the same time, making it a harder task than against image
classification. We specifically aim to attack the widely used Faster R-CNN by
changing the predicted label for a particular object in an image: where prior
work has targeted one specific object (a stop sign), we generalise to arbitrary
objects, with the key challenge being the need to change the labels of all
bounding boxes for all instances of that object type. To do so, we propose a
novel method, named Pick-Object-Attack. Pick-Object-Attack successfully adds
perturbations only to bounding boxes for the targeted object, preserving the
labels of other detected objects in the image. In terms of perceptibility, the
perturbations induced by the method are very small. Furthermore, for the first
time, we examine the effect of adversarial attacks on object detection in terms
of a downstream task, image captioning; we show that where a method that can
modify all object types leads to very obvious changes in captions, the changes
from our constrained attack are much less apparent.

    

### [[2006.03869] Learning Mixtures of Plackett-Luce Models with Features from Top-$l$ Orders](http://arxiv.org/abs/2006.03869)


  Plackett-Luce model (PL) is one of the most popular models for preference
learning. In this paper, we consider PL with features and its mixture models,
where each alternative has a vector of features, possibly different across
agents. Such models significantly generalize the standard PL, but are not as
well investigated in the literature. We extend mixtures of PLs with features to
models that generate top-$l$ and characterize their identifiability. We further
prove that when PL with features is identifiable, its MLE is consistent with a
strictly concave objective function under mild assumptions, by characterizing a
bound on root-mean-square-error (RMSE), which naturally leads to a sample
complexity bound. Our experiments on synthetic data demonstrate the
effectiveness of MLE on PL with features with tradeoffs between statistical
efficiency and computational efficiency when $l$ takes different values. Our
experiments on real-world data show the prediction power of PL with features
and its mixtures.

    

### [[2006.04270] EDropout: Energy-Based Dropout and Pruning of Deep Neural Networks](http://arxiv.org/abs/2006.04270)


  Dropout is a well-known regularization method by sampling a sub-network from
a larger deep neural network and training different sub-networks on different
subsets of the data. Inspired by the dropout concept, we propose EDropout as an
energy-based framework for pruning neural networks in classification tasks. In
this approach, a set of binary pruning state vectors (population) represents a
set of corresponding sub-networks from an arbitrary provided original neural
network. An energy loss function assigns a scalar energy loss value to each
pruning state. The energy-based model stochastically evolves the population to
find states with lower energy loss. The best pruning state is then selected and
applied to the original network. Similar to dropout, the kept weights are
updated using backpropagation in a probabilistic model. The energy-based model
again searches for better pruning states and the cycle continuous. Indeed, this
procedure is in fact switching between the energy model, which manages the
pruning states, and the probabilistic model, which updates the temporarily
unpruned weights, in each iteration. The population can dynamically converge to
a pruning state. This can be interpreted as dropout leading to pruning the
network. From an implementation perspective, EDropout can prune typical neural
networks without modification of the network architecture. We evaluated the
proposed method on different flavours of ResNets, AlexNet, and SqueezeNet on
the Kuzushiji, Fashion, CIFAR-10, CIFAR-100, and Flowers datasets, and compared
the pruning rate and classification performance of the models. On average the
networks trained with EDropout achieved a pruning rate of more than $50\%$ of
the trainable parameters with approximately $<5\%$ and $<1\%$ drop of Top-1 and
Top-5 classification accuracy, respectively.

    

### [[2006.04920] Survival regression with accelerated failure time model in XGBoost](http://arxiv.org/abs/2006.04920)


  Survival regression is used to estimate the relation between time-to-event
and feature variables, and is important in application domains such as
medicine, marketing, risk management and sales management. Nonlinear tree based
machine learning algorithms as implemented in libraries such as XGBoost,
scikit-learn, LightGBM, and CatBoost are often more accurate in practice than
linear models. However, existing state-of-the-art implementations of tree-based
models have offered limited support for survival regression. In this work, we
implement loss functions for learning accelerated failure time (AFT) models in
XGBoost, to increase the support for survival modeling for different kinds of
label censoring. We demonstrate with real and simulated experiments the
effectiveness of AFT in XGBoost with respect to a number of baselines, in two
respects: generalization performance and training speed. Furthermore, we take
advantage of the support for NVIDIA GPUs in XGBoost to achieve substantial
speedup over multi-core CPUs. To our knowledge, our work is the first
implementation of AFT that utilizes the processing power of NVIDIA GPUs.
Starting from the 1.2.0 release, the XGBoost package natively supports the AFT
model. The addition of AFT in XGBoost has had significant impact in the open
source community, and a few statistics packages now utilize the XGBoost AFT
model.

    

### [[2006.15669] Geometry-Inspired Top-k Adversarial Perturbations](http://arxiv.org/abs/2006.15669)


  The brittleness of deep image classifiers to small adversarial input
perturbations have been extensively studied in the last several years. However,
the main objective of existing perturbations is primarily limited to change the
correctly predicted Top-$1$ class by an incorrect one, which does not intend
changing the Top-$k$ prediction. In many digital real-world scenarios Top-$k$
prediction is more relevant. In this work, we propose a fast and accurate
method of computing Top-$k$ adversarial examples as a simple multi-objective
optimization. We demonstrate its efficacy and performance by comparing it to
other adversarial example crafting techniques. Moreover, based on this method,
we propose Top-$k$ Universal Adversarial Perturbations, image-agnostic tiny
perturbations that cause the true class to be absent among the Top-$k$
prediction for the majority of natural images. We experimentally show that our
approach outperforms baseline methods and even improves existing techniques of
generating Universal Adversarial Perturbations.

    

### [[2007.12769] A unified survey of treatment effect heterogeneity modeling and uplift modeling](http://arxiv.org/abs/2007.12769)


  A central question in many fields of scientific research is to determine how
an outcome would be affected by an action, or to measure the effect of an
action (a.k.a treatment effect). In recent years, a need for estimating the
heterogeneous treatment effects conditioning on the different characteristics
of individuals has emerged from research fields such as personalized
healthcare, social science, and online marketing. To meet the need, researchers
and practitioners from different communities have developed algorithms by
taking the treatment effect heterogeneity modeling approach and the uplift
modeling approach, respectively. In this paper, we provide a unified survey of
these two seemingly disconnected yet closely related approaches under the
potential outcome framework. We then provide a structured survey of existing
methods by emphasizing on their inherent connections with a set of unified
notations to make comparisons of the different methods easy. We then review the
main applications of the surveyed methods in personalized marketing,
personalized medicine, and social studies. Finally, we summarize the existing
software packages and present discussions based on the use of methods on
synthetic, semi-synthetic and real world data sets and provide some general
guidelines for choosing methods.

    

### [[2009.05236] An Efficient Quantitative Approach for Optimizing Convolutional Neural Networks](http://arxiv.org/abs/2009.05236)


  With the increasing popularity of deep learning, Convolutional Neural
Networks (CNNs) have been widely applied in various domains, such as image
classification and object detection, and achieve stunning success in terms of
their high accuracy over the traditional statistical methods. To exploit the
potential of CNN models, a huge amount of research and industry efforts have
been devoted to optimizing CNNs. Among these endeavors, CNN architecture design
has attracted tremendous attention because of its great potential of improving
model accuracy or reducing model complexity. However, existing work either
introduces repeated training overhead in the search process or lacks an
interpretable metric to guide the design. To clear these hurdles, we propose
Information Field (IF), an explainable and easy-to-compute metric, to estimate
the quality of a CNN architecture and guide the search process of designs. To
validate the effectiveness of IF, we build a static optimizer to improve the
CNN architectures at both the stage level and the kernel level. Our optimizer
not only provides a clear and reproducible procedure but also mitigates
unnecessary training efforts in the architecture search process. Extensive
experiments and studies show that the models generated by our optimizer can
achieve up to 5.47% accuracy improvement and up to 65.38% parameters deduction,
compared with state-of-the-art CNN structures like MobileNet and ResNet.

    

### [[2010.11750] Analysis of Information Transfer from Heterogeneous Sources via Precise High-dimensional Asymptotics](http://arxiv.org/abs/2010.11750)


  We consider the problem of transfer learning -- gaining knowledge from one
source task and applying it to a different but related target task. A
fundamental question in transfer learning is whether combining the data of both
tasks works better than using only the target task's data (equivalently,
whether a "positive information transfer" happens). We study this question
formally in a linear regression setting where a two-layer linear neural network
estimator combines both tasks' data. The estimator uses a shared parameter
vector for both tasks and exhibits positive or negative information transfer by
varying dataset characteristics.
We characterize the precise asymptotic limit of the prediction risk of the
above estimator when the sample sizes increase with the feature dimension
proportionally at fixed ratios. We also show that the asymptotic limit is
sufficiently accurate for finite dimensions. Then, we provide the exact
condition to determine positive (and negative) information transfer in a
random-effect model, leading to several theoretical insights. For example, the
risk curve is non-monotone under model shift, thus motivating a transfer
learning procedure that progressively adds data from the source task. We
validate this procedure's efficiency on text classification tasks with a neural
network that applies a shared feature space for both tasks, similar to the
above estimator. The main ingredient of the analysis is finding the
high-dimensional asymptotic limits of various functions involving the sum of
two independent sample covariance matrices with different population covariance
matrices, which may be of independent interest.

    

### [[2011.00958] Collection and Validation of Psychophysiological Data from Professional and Amateur Players: a Multimodal eSports Dataset](http://arxiv.org/abs/2011.00958)


  Proper training and analytics in eSports require accurately collected and
annotated data. Most eSports research focuses exclusively on in-game data
analysis, and there is a lack of prior work involving eSports athletes'
psychophysiological data. In this paper, we present a dataset collected from
professional and amateur teams in 22 matches in League of Legends video game
with more than 40 hours of recordings. Recorded data include the players'
physiological activity, e.g. movements, pulse, saccades, obtained from various
sensors, self-reported aftermatch survey, and in-game data. An important
feature of the dataset is simultaneous data collection from five players, which
facilitates the analysis of sensor data on a team level. Upon the collection of
dataset we carried out its validation. In particular, we demonstrate that
stress and concentration levels for professional players are less correlated,
meaning more independent playstyle. Also, we show that the absence of team
communication does not affect the professional players as much as amateur ones.
To investigate other possible use cases of the dataset, we have trained
classical machine learning algorithms for skill prediction and player
re-identification using 3-minute sessions of sensor data. Best models achieved
0.856 and 0.521 (0.10 for a chance level) accuracy scores on a validation set
for skill prediction and player re-id problems, respectively. The dataset is
available at this https URL Sensors Dataset.

    

### [[2012.02299] Detecting Video Game Player Burnout with the Use of Sensor Data and Machine Learning](http://arxiv.org/abs/2012.02299)


  Current research in eSports lacks the tools for proper game practising and
performance analytics. The majority of prior work relied only on in-game data
for advising the players on how to perform better. However, in-game mechanics
and trends are frequently changed by new patches limiting the lifespan of the
models trained exclusively on the in-game logs. In this article, we propose the
methods based on the sensor data analysis for predicting whether a player will
win the future encounter. The sensor data were collected from 10 participants
in 22 matches in League of Legends video game. We have trained machine learning
models including Transformer and Gated Recurrent Unit to predict whether the
player wins the encounter taking place after some fixed time in the future. For
10 seconds forecasting horizon Transformer neural network architecture achieves
ROC AUC score 0.706. This model is further developed into the detector capable
of predicting that a player will lose the encounter occurring in 10 seconds in
88.3% of cases with 73.5% accuracy. This might be used as a players' burnout or
fatigue detector, advising players to retreat. We have also investigated which
physiological features affect the chance to win or lose the next in-game
encounter.

    

### [[2012.02670] Unleashing the Tiger: Inference Attacks on Split Learning](http://arxiv.org/abs/2012.02670)


  We investigate the security of Split Learning -- a novel collaborative
machine learning framework that enables peak performance by requiring minimal
resources consumption. In the present paper, we expose vulnerabilities of the
protocol and demonstrate its inherent insecurity by introducing general attack
strategies targeting the reconstruction of clients' private training sets. More
prominently, we show that a malicious server can actively hijack the learning
process of the distributed model and bring it into an insecure state that
enables inference attacks on clients' data. We implement different adaptations
of the attack and test them on various datasets as well as within realistic
threat scenarios. We demonstrate that our attack is able to overcome recently
proposed defensive techniques aimed at enhancing the security of the split
learning protocol. Finally, we also illustrate the protocol's insecurity
against malicious clients by extending previously devised attacks for Federated
Learning. To make our results reproducible, we made our code available at
this https URL.

    

### [[2012.07654] Session-Aware Query Auto-completion using Extreme Multi-label Ranking](http://arxiv.org/abs/2012.07654)


  Query auto-completion (QAC) is a fundamental feature in search engines where
the task is to suggest plausible completions of a prefix typed in the search
bar. Previous queries in the user session can provide useful context for the
user's intent and can be leveraged to suggest auto-completions that are more
relevant while adhering to the user's prefix. Such session-aware QACs can be
generated by recent sequence-to-sequence deep learning models; however, these
generative approaches often do not meet the stringent latency requirements of
responding to each user keystroke. Moreover, these generative approaches pose
the risk of showing nonsensical queries.
In this paper, we provide a solution to this problem: we take the novel
approach of modeling session-aware QAC as an eXtreme Multi-Label Ranking (XMR)
problem where the input is the previous query in the session and the user's
current prefix, while the output space is the set of tens of millions of
queries entered by users in the recent past. We adapt a popular XMR algorithm
for this purpose by proposing several modifications to the key steps in the
algorithm. The proposed modifications yield a 3.9x improvement in terms of Mean
Reciprocal Rank (MRR) over the baseline XMR approach on a public search logs
dataset. We are able to maintain an inference latency of less than 10 ms while
still using session context. When compared against baseline models of
acceptable latency, we observed a 33% improvement in MRR for short prefixes of
up to 3 characters. Moreover, our model yielded a statistically significant
improvement of 2.81% over a production QAC system in terms of suggestion
acceptance rate, when deployed on the search bar of an online shopping store as
part of an A/B test.

    

### [[2012.09720] Hardness of Learning Halfspaces with Massart Noise](http://arxiv.org/abs/2012.09720)


  We study the complexity of PAC learning halfspaces in the presence of Massart
(bounded) noise. Specifically, given labeled examples $(x, y)$ from a
distribution $D$ on $\mathbb{R}^{n} \times \{ \pm 1\}$ such that the marginal
distribution on $x$ is arbitrary and the labels are generated by an unknown
halfspace corrupted with Massart noise at rate $\eta<1/2$, we want to compute a
hypothesis with small misclassification error. Characterizing the efficient
learnability of halfspaces in the Massart model has remained a longstanding
open problem in learning theory.
Recent work gave a polynomial-time learning algorithm for this problem with
error $\eta+\epsilon$. This error upper bound can be far from the
information-theoretically optimal bound of $\mathrm{OPT}+\epsilon$. More recent
work showed that {\em exact learning}, i.e., achieving error
$\mathrm{OPT}+\epsilon$, is hard in the Statistical Query (SQ) model. In this
work, we show that there is an exponential gap between the
information-theoretically optimal error and the best error that can be achieved
by a polynomial-time SQ algorithm. In particular, our lower bound implies that
no efficient SQ algorithm can approximate the optimal error within any
polynomial factor.

    

### [[2012.12821] Focal Frequency Loss for Image Reconstruction and Synthesis](http://arxiv.org/abs/2012.12821)


  Image reconstruction and synthesis have witnessed remarkable progress thanks
to the development of generative models. Nonetheless, gaps could still exist
between the real and generated images, especially in the frequency domain. In
this study, we show that narrowing gaps in the frequency domain can ameliorate
image reconstruction and synthesis quality further. We propose a novel focal
frequency loss, which allows a model to adaptively focus on frequency
components that are hard to synthesize by down-weighting the easy ones. This
objective function is complementary to existing spatial losses, offering great
impedance against the loss of important frequency information due to the
inherent bias of neural networks. We demonstrate the versatility and
effectiveness of focal frequency loss to improve popular models, such as VAE,
pix2pix, and SPADE, in both perceptual quality and quantitative performance. We
further show its potential on StyleGAN2.

    

### [[2102.01063] Zen-NAS: A Zero-Shot NAS for High-Performance Deep Image Recognition](http://arxiv.org/abs/2102.01063)


  Accuracy predictor is a key component in Neural Architecture Search (NAS) for
ranking architectures. Building a high-quality accuracy predictor usually costs
enormous computation. To address this issue, instead of using an accuracy
predictor, we propose a novel zero-shot index dubbed Zen-Score to rank the
architectures. The Zen-Score represents the network expressivity and positively
correlates with the model accuracy. The calculation of Zen-Score only takes a
few forward inferences through a randomly initialized network, without training
network parameters. Built upon the Zen-Score, we further propose a new NAS
algorithm, termed as Zen-NAS, by maximizing the Zen-Score of the target network
under given inference budgets. Within less than half GPU day, Zen-NAS is able
to directly search high performance architectures in a data-free style.
Comparing with previous NAS methods, the proposed Zen-NAS is magnitude times
faster on multiple server-side and mobile-side GPU platforms with
state-of-the-art accuracy on ImageNet. Our source code and pre-trained models
are released on this https URL.

    

### [[2102.04969] Semantic Borrowing for Generalized Zero-Shot Learning](http://arxiv.org/abs/2102.04969)


  Generalized zero-shot learning (GZSL) is one of the most realistic but
challenging problems due to the partiality of the classifier to supervised
classes, especially under the class-inductive instance-inductive (CIII)
training setting, where testing data are not available. Instance-borrowing
methods and synthesizing methods solve it to some extent with the help of
testing semantics, but therefore neither can be used under CIII. Besides, the
latter require the training process of a classifier after generating examples.
In contrast, a novel non-transductive regularization under CIII called Semantic
Borrowing (SB) for improving GZSL methods with compatibility metric learning is
proposed in this paper, which not only can be used for training linear models,
but also nonlinear ones such as artificial neural networks. This regularization
item in the loss function borrows similar semantics in the training set, so
that the classifier can model the relationship between the semantics of
zero-shot and supervised classes more accurately during training. In practice,
the information of semantics of unknown classes would not be available for
training while this approach does NOT need it. Extensive experiments on GZSL
benchmark datasets show that SB can reduce the partiality of the classifier to
supervised classes and improve the performance of generalized zero-shot
classification, surpassing inductive GZSL state of the arts.

    

### [[2102.06197] Estimating a Latent Tree for Extremes](http://arxiv.org/abs/2102.06197)


  The Latent River Problem has emerged as a flagship problem for causal
discovery in extreme value statistics. This paper gives QTree, a simple and
efficient algorithm to solve the Latent River Problem that outperforms existing
methods. QTree returns a directed graph and achieves almost perfect recovery on
the Upper Danube, the existing benchmark dataset, as well as on new data from
the Lower Colorado River in Texas. It can handle missing data, has an automated
parameter tuning procedure, and runs in time $O(n |V|^2)$, where $n$ is the
number of observations and $|V|$ the number of nodes in the graph. In addition,
under a Bayesian network model for extreme values with propagating noise, we
show that the QTree estimator returns for $n\to\infty$ a.s. the correct tree.

    

### [[2102.07078] Exploiting Shared Representations for Personalized Federated Learning](http://arxiv.org/abs/2102.07078)


  Deep neural networks have shown the ability to extract universal feature
representations from data such as images and text that have been useful for a
variety of learning tasks. However, the fruits of representation learning have
yet to be fully-realized in federated settings. Although data in federated
settings is often non-i.i.d. across clients, the success of centralized deep
learning suggests that data often shares a global feature representation, while
the statistical heterogeneity across clients or tasks is concentrated in the
labels. Based on this intuition, we propose a novel federated learning
framework and algorithm for learning a shared data representation across
clients and unique local heads for each client. Our algorithm harnesses the
distributed computational power across clients to perform many local-updates
with respect to the low-dimensional local parameters for every update of the
representation. We prove that this method obtains linear convergence to the
ground-truth representation with near-optimal sample complexity in a linear
setting, demonstrating that it can efficiently reduce the problem dimension for
each client. This result is of interest beyond federated learning to a broad
class of problems in which we aim to learn a shared low-dimensional
representation among data distributions, for example in meta-learning and
multi-task learning. Further, extensive experimental results show the empirical
improvement of our method over alternative personalized federated learning
approaches in federated environments with heterogeneous data.

    

### [[2102.08248] Hierarchical VAEs Know What They Don't Know](http://arxiv.org/abs/2102.08248)


  Deep generative models have been demonstrated as state-of-the-art density
estimators. Yet, recent work has found that they often assign a higher
likelihood to data from outside the training distribution. This seemingly
paradoxical behavior has caused concerns over the quality of the attained
density estimates. In the context of hierarchical variational autoencoders, we
provide evidence to explain this behavior by out-of-distribution data having
in-distribution low-level features. We argue that this is both expected and
desirable behavior. With this insight in hand, we develop a fast, scalable and
fully unsupervised likelihood-ratio score for OOD detection that requires data
to be in-distribution across all feature-levels. We benchmark the method on a
vast set of data and model combinations and achieve state-of-the-art results on
out-of-distribution detection.

    

### [[2102.11855] Deep Unitary Convolutional Neural Networks](http://arxiv.org/abs/2102.11855)


  Deep neural networks can suffer from the exploding and vanishing activation
problem, in which the networks fail to train properly because the neural
signals either amplify or attenuate across the layers and become saturated.
While other normalization methods aim to fix the stated problem, most of them
have inference speed penalties in those applications that require running
averages of the neural activations. Here we extend the unitary framework based
on Lie algebra to neural networks of any dimensionalities, overcoming the major
constraints of the prior arts that limit synaptic weights to be square
matrices. Our proposed unitary convolutional neural networks deliver up to 32%
faster inference speeds and up to 50% reduction in permanent hard disk space
while maintaining competitive prediction accuracy.

    

### [[2103.00422] Alignment Knowledge Distillation for Online Streaming Attention-based Speech Recognition](http://arxiv.org/abs/2103.00422)


  This article describes an efficient training method for online streaming
attention-based encoder-decoder (AED) automatic speech recognition (ASR)
systems. AED models have achieved competitive performance in offline scenarios
by jointly optimizing all components. They have recently been extended to an
online streaming framework via models such as monotonic chunkwise attention
(MoChA). However, the elaborate attention calculation process is not robust for
long-form speech utterances. Moreover, the sequence-level training objective
and time-restricted streaming encoder cause a nonnegligible delay in token
emission during inference. To address these problems, we propose CTC
synchronous training (CTC-ST), in which CTC alignments are leveraged as a
reference for token boundaries to enable a MoChA model to learn optimal
monotonic input-output alignments. We formulate a purely end-to-end training
objective to synchronize the boundaries of MoChA to those of CTC. The CTC model
shares an encoder with the MoChA model to enhance the encoder representation.
Moreover, the proposed method provides alignment information learned in the CTC
branch to the attention-based decoder. Therefore, CTC-ST can be regarded as
self-distillation of alignment knowledge from CTC to MoChA. Experimental
evaluations on a variety of benchmark datasets show that the proposed method
significantly reduces recognition errors and emission latency simultaneously.
The robustness to long-form and noisy speech is also demonstrated. We compare
CTC-ST with several methods that distill alignment knowledge from a hybrid ASR
system and show that the CTC-ST can achieve a comparable tradeoff of accuracy
and latency without relying on external alignment information. The best MoChA
system shows recognition accuracy comparable to that of RNN-transducer (RNN-T)
while achieving lower emission latency.

    

### [[2103.00550] A Survey on Deep Semi-supervised Learning](http://arxiv.org/abs/2103.00550)


  Deep semi-supervised learning is a fast-growing field with a range of
practical applications. This paper provides a comprehensive survey on both
fundamentals and recent advances in deep semi-supervised learning methods from
perspectives of model design and unsupervised loss functions. We first present
a taxonomy for deep semi-supervised learning that categorizes existing methods,
including deep generative methods, consistency regularization methods,
graph-based methods, pseudo-labeling methods, and hybrid methods. Then we
provide a comprehensive review of 52 representative methods and offer a
detailed comparison of these methods in terms of the type of losses,
contributions, and architecture differences. In addition to the progress in the
past few years, we further discuss some shortcomings of existing methods and
provide some tentative heuristic solutions for solving these open problems.

    

### [[2103.01119] DTW-Merge: A Novel Data Augmentation Technique for Time Series Classification](http://arxiv.org/abs/2103.01119)


  In recent years, neural networks achieved much success in various
applications. The main challenge in training deep neural networks is the lack
of sufficient data to improve the model's generalization and avoid overfitting.
One of the solutions is to generate new training samples. This paper proposes a
novel data augmentation method for time series based on Dynamic Time Warping.
This method is inspired by the concept that warped parts of two time series
have similar temporal properties and therefore, exchanging them between the two
series generates a new training sample. The proposed method selects an element
of the optimal warping path randomly and then exchanges the segments that are
aligned together. Exploiting the proposed approach with recently introduced
ResNet reveals improved results on the 2018 UCR Time Series Classification
Archive. By employing Gradient-weighted Class Activation Mapping (Grad-CAM) and
Multidimensional Scaling (MDS), we manifest that our method extract more
discriminant features out of time series.

    

### [[2103.01447] ZeroSARAH: Efficient Nonconvex Finite-Sum Optimization with Zero Full Gradient Computation](http://arxiv.org/abs/2103.01447)


  We propose ZeroSARAH -- a novel variant of the variance-reduced method SARAH
(Nguyen et al., 2017) -- for minimizing the average of a large number of
nonconvex functions $\frac{1}{n}\sum_{i=1}^{n}f_i(x)$. To the best of our
knowledge, in this nonconvex finite-sum regime, all existing variance-reduced
methods, including SARAH, SVRG, SAGA and their variants, need to compute the
full gradient over all $n$ data samples at the initial point $x^0$, and then
periodically compute the full gradient once every few iterations (for SVRG,
SARAH and their variants). Note that SVRG, SAGA and their variants typically
achieve weaker convergence results than variants of SARAH: $n^{2/3}/\epsilon^2$
vs. $n^{1/2}/\epsilon^2$. Thus we focus on the variant of SARAH. The proposed
ZeroSARAH and its distributed variant D-ZeroSARAH are the \emph{first}
variance-reduced algorithms which \emph{do not require any full gradient
computations}, not even for the initial point. Moreover, for both standard and
distributed settings, we show that ZeroSARAH and D-ZeroSARAH obtain new
state-of-the-art convergence results, which can improve the previous best-known
result (given by e.g., SPIDER, SARAH, and PAGE) in certain regimes. Avoiding
any full gradient computations (which are time-consuming steps) is important in
many applications as the number of data samples $n$ usually is very large.
Especially in the distributed setting, periodic computation of full gradient
over all data samples needs to periodically synchronize all
clients/devices/machines, which may be impossible or unaffordable. Thus, we
expect that ZeroSARAH/D-ZeroSARAH will have a practical impact in distributed
and federated learning where full device participation is impractical.

    

### [[2103.03191] Generalization Bounds for Sparse Random Feature Expansions](http://arxiv.org/abs/2103.03191)


  Random feature methods have been successful in various machine learning
tasks, are easy to compute, and come with theoretical accuracy bounds. They
serve as an alternative approach to standard neural networks since they can
represent similar function spaces without a costly training phase. However, for
accuracy, random feature methods require more measurements than trainable
parameters, limiting their use for data-scarce applications or problems in
scientific machine learning. This paper introduces the sparse random feature
expansion to obtain parsimonious random feature models. Specifically, we
leverage ideas from compressive sensing to generate random feature expansions
with theoretical guarantees even in the data-scarce setting. In particular, we
provide generalization bounds for functions in a certain class (that is dense
in a reproducing kernel Hilbert space) depending on the number of samples and
the distribution of features. The generalization bounds improve with additional
structural conditions, such as coordinate sparsity, compact clusters of the
spectrum, or rapid spectral decay. In particular, by introducing sparse
features, i.e. features with random sparse weights, we provide improved bounds
for low order functions. We show that the sparse random feature expansions
outperforms shallow networks in several scientific machine learning tasks.

    

### [[2103.04047] Reinforcement Learning, Bit by Bit](http://arxiv.org/abs/2103.04047)


  Reinforcement learning agents have demonstrated remarkable achievements in
simulated environments. Data efficiency poses an impediment to carrying this
success over to real environments. The design of data-efficient agents calls
for a deeper understanding of information acquisition and representation. We
develop concepts and establish a regret bound that together offer principled
guidance. The bound sheds light on questions of what information to seek, how
to seek that information, and it what information to retain. To illustrate
concepts, we design simple agents that build on them and present computational
results that demonstrate improvements in data efficiency.

    

### [[2103.04850] Quantifying Ignorance in Individual-Level Causal-Effect Estimates under Hidden Confounding](http://arxiv.org/abs/2103.04850)


  We study the problem of learning conditional average treatment effects (CATE)
from high-dimensional, observational data with unobserved confounders.
Unobserved confounders introduce ignorance -- a level of unidentifiability --
about an individual's response to treatment by inducing bias in CATE estimates.
We present a new parametric interval estimator suited for high-dimensional
data, that estimates a range of possible CATE values when given a predefined
bound on the level of hidden confounding. Further, previous interval estimators
do not account for ignorance about the CATE associated with samples that may be
underrepresented in the original study, or samples that violate the overlap
assumption. Our interval estimator also incorporates model uncertainty so that
practitioners can be made aware of out-of-distribution data. We prove that our
estimator converges to tight bounds on CATE when there may be unobserved
confounding, and assess it using semi-synthetic, high-dimensional datasets.

    

### [[2103.05236] GAN Vocoder: Multi-Resolution Discriminator Is All You Need](http://arxiv.org/abs/2103.05236)


  Several of the latest GAN-based vocoders show remarkable achievements,
outperforming autoregressive and flow-based competitors in both qualitative and
quantitative measures while synthesizing orders of magnitude faster. In this
work, we hypothesize that the common factor underlying their success is the
multi-resolution discriminating framework, not the minute details in
architecture, loss function, or training strategy. We experimentally test the
hypothesis by evaluating six different generators paired with one shared
multi-resolution discriminating framework. For all evaluative measures with
respect to text-to-speech syntheses and for all perceptual metrics, their
performances are not distinguishable from one another, which supports our
hypothesis.

    

### [[2103.10094] KoDF: A Large-scale Korean DeepFake Detection Dataset](http://arxiv.org/abs/2103.10094)


  A variety of effective face-swap and face-reenactment methods have been
publicized in recent years, democratizing the face synthesis technology to a
great extent. Videos generated as such have come to be called deepfakes with a
negative connotation, for various social problems they have caused. Facing the
emerging threat of deepfakes, we have built the Korean DeepFake Detection
Dataset (KoDF), a large-scale collection of synthesized and real videos focused
on Korean subjects. In this paper, we provide a detailed description of methods
used to construct the dataset, experimentally show the discrepancy between the
distributions of KoDF and existing deepfake detection datasets, and underline
the importance of using multiple datasets for real-world generalization. KoDF
is publicly available at this https URL in its
entirety (i.e. real clips, synthesized clips, clips with adversarial attack,
and metadata).

    

### [[2103.16652] Robustness Certification for Point Cloud Models](http://arxiv.org/abs/2103.16652)


  The use of deep 3D point cloud models in safety-critical applications, such
as autonomous driving, dictates the need to certify the robustness of these
models to real-world transformations. This is technically challenging, as it
requires a scalable verifier tailored to point cloud models that handles a wide
range of semantic 3D transformations. In this work, we address this challenge
and introduce 3DCertify, the first verifier able to certify the robustness of
point cloud models. 3DCertify is based on two key insights: (i) a generic
relaxation based on first-order Taylor approximations, applicable to any
differentiable transformation, and (ii) a precise relaxation for global feature
pooling, which is more complex than pointwise activations (e.g., ReLU or
sigmoid) but commonly employed in point cloud models. We demonstrate the
effectiveness of 3DCertify by performing an extensive evaluation on a wide
range of 3D transformations (e.g., rotation, twisting) for both classification
and part segmentation tasks. For example, we can certify robustness against
rotations by $\pm$60° for 95.7% of point clouds, and our max pool
relaxation increases certification by up to 15.6%.

    

### [[2103.17150] Federated Learning: A Signal Processing Perspective](http://arxiv.org/abs/2103.17150)


  The dramatic success of deep learning is largely due to the availability of
data. Data samples are often acquired on edge devices, such as smart phones,
vehicles and sensors, and in some cases cannot be shared due to privacy
considerations. Federated learning is an emerging machine learning paradigm for
training models across multiple edge devices holding local datasets, without
explicitly exchanging the data. Learning in a federated manner differs from
conventional centralized machine learning, and poses several core unique
challenges and requirements, which are closely related to classical problems
studied in the areas of signal processing and communications. Consequently,
dedicated schemes derived from these areas are expected to play an important
role in the success of federated learning and the transition of deep learning
from the domain of centralized servers to mobile edge devices. In this article,
we provide a unified systematic framework for federated learning in a manner
that encapsulates and highlights the main challenges that are natural to treat
using signal processing tools. We present a formulation for the federated
learning paradigm from a signal processing perspective, and survey a set of
candidate approaches for tackling its unique challenges. We further provide
guidelines for the design and adaptation of signal processing and communication
methods to facilitate federated learning at large scale.

    

### [[2104.01061] Information Geometry and Classical Cramér-Rao Type Inequalities](http://arxiv.org/abs/2104.01061)


  We examine the role of information geometry in the context of classical
Cramér-Rao (CR) type inequalities. In particular, we focus on Eguchi's theory
of obtaining dualistic geometric structures from a divergence function and then
applying Amari-Nagoaka's theory to obtain a CR type inequality. The classical
deterministic CR inequality is derived from Kullback-Leibler (KL)-divergence.
We show that this framework could be generalized to other CR type inequalities
through four examples: $\alpha$-version of CR inequality, generalized CR
inequality, Bayesian CR inequality, and Bayesian $\alpha$-CR inequality. These
are obtained from, respectively, $I_\alpha$-divergence (or relative
$\alpha$-entropy), generalized Csiszár divergence, Bayesian KL divergence,
and Bayesian $I_\alpha$-divergence.

    

### [[2104.01503] STL Robustness Risk over Discrete-Time Stochastic Processes](http://arxiv.org/abs/2104.01503)


  We present a framework to interpret signal temporal logic (STL) formulas over
discrete-time stochastic processes in terms of the induced risk. Each
realization of a stochastic process either satisfies or violates an STL
formula. In fact, we can assign a robustness value to each realization that
indicates how robustly this realization satisfies an STL formula. We then
define the risk of a stochastic process not satisfying an STL formula robustly,
referred to as the STL robustness risk. In our definition, we permit general
classes of risk measures such as, but not limited to, the conditional
value-at-risk. While in general hard to compute, we propose an approximation of
the STL robustness risk. This approximation has the desirable property of being
an upper bound of the STL robustness risk when the chosen risk measure is
monotone, a property satisfied by most risk measures. Motivated by the interest
in data-driven approaches, we present a sampling-based method for estimating
the approximate STL robustness risk from data for the value-at-risk. While we
consider the value-at-risk, we highlight that such sampling-based methods are
viable for other risk measures.

    

### [[2104.03642] CLIMAT: Clinically-Inspired Multi-Agent Transformers for Disease Trajectory Forecasting from Multi-modal Data](http://arxiv.org/abs/2104.03642)


  In medical applications, deep learning methods are built to automate
diagnostic tasks, often formulated as single-target classification problems.
However, a clinically relevant question that practitioners usually face, is how
to predict the future trajectory of a disease (prognosis). Current methods for
such a problem often require domain knowledge, and are complicated to apply. In
this paper, we formulate the prognosis prediction problem as a one-to-many
forecasting problem. Inspired by a clinical decision-making process with two
agents -- a radiologist and a general practitioner, we model a prognosis
prediction problem with two transformer-based components that share information
between each other. The first transformer in this model aims to analyze the
imaging data, and the second one leverages its internal states as inputs, also
fusing them with auxiliary patient data. We show the effectiveness of our
method in predicting the development of structural knee osteoarthritis changes,
and forecasting Alzheimer's disease clinical status. Our results show that the
proposed method outperforms the state-of-the-art baselines in terms of various
performance metrics, including calibration, which is desired from a medical
decision support system. An open source implementation of our method is made
publicly available at this https URL.

    

### [[2104.08938] On the approximation of functions by tanh neural networks](http://arxiv.org/abs/2104.08938)


  We derive bounds on the error, in high-order Sobolev norms, incurred in the
approximation of Sobolev-regular as well as analytic functions by neural
networks with the hyperbolic tangent activation function. These bounds provide
explicit estimates on the approximation error with respect to the size of the
neural networks. We show that tanh neural networks with only two hidden layers
suffice to approximate functions at comparable or better rates than much deeper
ReLU neural networks.

    

### [[2104.09248] LSPnet: A 2D Localization-oriented Spacecraft Pose Estimation Neural Network](http://arxiv.org/abs/2104.09248)


  Being capable of estimating the pose of uncooperative objects in space has
been proposed as a key asset for enabling safe close-proximity operations such
as space rendezvous, in-orbit servicing and active debris removal. Usual
approaches for pose estimation involve classical computer vision-based
solutions or the application of Deep Learning (DL) techniques. This work
explores a novel DL-based methodology, using Convolutional Neural Networks
(CNNs), for estimating the pose of uncooperative spacecrafts. Contrary to other
approaches, the proposed CNN directly regresses poses without needing any prior
3D information. Moreover, bounding boxes of the spacecraft in the image are
predicted in a simple, yet efficient manner. The performed experiments show how
this work competes with the state-of-the-art in uncooperative spacecraft pose
estimation, including works which require 3D information as well as works which
predict bounding boxes through sophisticated CNNs.

    

### [[2104.10425] Sparse-shot Learning with Exclusive Cross-Entropy for Extremely Many Localisations](http://arxiv.org/abs/2104.10425)


  Object localisation, in the context of regular images, often depicts objects
like people or cars. In these images, there is typically a relatively small
number of objects per class, which usually is manageable to annotate. However,
outside the setting of regular images, we are often confronted with a different
situation. In computational pathology, digitised tissue sections are extremely
large images, whose dimensions quickly exceed 250'000x250'000 pixels, where
relevant objects, such as tumour cells or lymphocytes can quickly number in the
millions. Annotating them all is practically impossible and annotating sparsely
a few, out of many more, is the only possibility. Unfortunately, learning from
sparse annotations, or sparse-shot learning, clashes with standard supervised
learning because what is not annotated is treated as a negative. However,
assigning negative labels to what are true positives leads to confusion in the
gradients and biased learning. To this end, we present exclusive cross-entropy,
which slows down the biased learning by examining the second-order loss
derivatives in order to drop the loss terms corresponding to likely biased
terms. Experiments on nine datasets and two different localisation tasks,
detection with YOLLO and segmentation with Unet, show that we obtain
considerable improvements compared to cross-entropy or focal loss, while often
reaching the best possible performance for the model with only 10-40% of
annotations.

    

### [[2104.12276] Learning to Better Segment Objects from Unseen Classes with Unlabeled Videos](http://arxiv.org/abs/2104.12276)


  The ability to localize and segment objects from unseen classes would open
the door to new applications, such as autonomous object learning in active
vision. Nonetheless, improving the performance on unseen classes requires
additional training data, while manually annotating the objects of the unseen
classes can be labor-extensive and expensive. In this paper, we explore the use
of unlabeled video sequences to automatically generate training data for
objects of unseen classes. It is in principle possible to apply existing video
segmentation methods to unlabeled videos and automatically obtain object masks,
which can then be used as a training set even for classes with no manual labels
available. However, our experiments show that these methods do not perform well
enough for this purpose. We therefore introduce a Bayesian method that is
specifically designed to automatically create such a training set: Our method
starts from a set of object proposals and relies on (non-realistic)
analysis-by-synthesis to select the correct ones by performing an efficient
optimization over all the frames simultaneously. Through extensive experiments,
we show that our method can generate a high-quality training set which
significantly boosts the performance of segmenting objects of unseen classes.
We thus believe that our method could open the door for open-world instance
segmentation using abundant Internet videos.

    

### [[2105.04854] Improving Molecular Graph Neural Network Explainability with Orthonormalization and Induced Sparsity](http://arxiv.org/abs/2105.04854)


  Rationalizing which parts of a molecule drive the predictions of a molecular
graph convolutional neural network (GCNN) can be difficult. To help, we propose
two simple regularization techniques to apply during the training of GCNNs:
Batch Representation Orthonormalization (BRO) and Gini regularization. BRO,
inspired by molecular orbital theory, encourages graph convolution operations
to generate orthonormal node embeddings. Gini regularization is applied to the
weights of the output layer and constrains the number of dimensions the model
can use to make predictions. We show that Gini and BRO regularization can
improve the accuracy of state-of-the-art GCNN attribution methods on artificial
benchmark datasets. In a real-world setting, we demonstrate that medicinal
chemists significantly prefer explanations extracted from regularized models.
While we only study these regularizers in the context of GCNNs, both can be
applied to other types of neural networks

    

### [[2105.12049] Honest-but-Curious Nets: Sensitive Attributes of Private Inputs Can Be Secretly Coded into the Classifiers' Outputs](http://arxiv.org/abs/2105.12049)


  It is known that deep neural networks, trained for the classification of a
non-sensitive target attribute, can reveal some sensitive attributes of their
input data; through features of different granularity extracted by the
classifier. We take a step forward and show that deep classifiers can be
trained to secretly encode a sensitive attribute of users' input data into the
classifier's outputs for the target attribute, at inference time. This results
in an attack that works even if users have a full white-box view of the
classifier, and can keep all internal representations hidden except for the
classifier's outputs for the target attribute. We introduce an
information-theoretical formulation of such attacks and present efficient
empirical implementations for training honest-but-curious (HBC) classifiers
based on this formulation: classifiers that can be accurate in predicting their
target attribute, but can also exploit their outputs to secretly encode a
sensitive attribute. Our evaluations on several tasks in real-world datasets
show that a semi-trusted server can build a classifier that is not only
perfectly honest but also accurately curious. Our work highlights a
vulnerability that can be exploited by malicious machine learning service
providers to attack their user's privacy in several seemingly safe scenarios;
such as encrypted inferences, computations at the edge, or private knowledge
distillation. We conclude by showing the difficulties in distinguishing between
standard and HBC classifiers, discussing challenges in defending against this
vulnerability of deep classifiers, and enumerating related open directions for
future studies.

    

### [[2106.06508] Preferential Temporal Difference Learning](http://arxiv.org/abs/2106.06508)


  Temporal-Difference (TD) learning is a general and very useful tool for
estimating the value function of a given policy, which in turn is required to
find good policies. Generally speaking, TD learning updates states whenever
they are visited. When the agent lands in a state, its value can be used to
compute the TD-error, which is then propagated to other states. However, it may
be interesting, when computing updates, to take into account other information
than whether a state is visited or not. For example, some states might be more
important than others (such as states which are frequently seen in a successful
trajectory). Or, some states might have unreliable value estimates (for
example, due to partial observability or lack of data), making their values
less desirable as targets. We propose an approach to re-weighting states used
in TD updates, both when they are the input and when they provide the target
for the update. We prove that our approach converges with linear function
approximation and illustrate its desirable empirical behaviour compared to
other TD-style methods.

    

### [[2106.06969] SoundDet: Polyphonic Moving Sound Event Detection and Localization from Raw Waveform](http://arxiv.org/abs/2106.06969)


  We present a new framework SoundDet, which is an end-to-end trainable and
light-weight framework, for polyphonic moving sound event detection and
localization. Prior methods typically approach this problem by preprocessing
raw waveform into time-frequency representations, which is more amenable to
process with well-established image processing pipelines. Prior methods also
detect in segment-wise manner, leading to incomplete and partial detections.
SoundDet takes a novel approach and directly consumes the raw, multichannel
waveform and treats the spatio-temporal sound event as a complete
"sound-object" to be detected. Specifically, SoundDet consists of a backbone
neural network and two parallel heads for temporal detection and spatial
localization, respectively. Given the large sampling rate of raw waveform, the
backbone network first learns a set of phase-sensitive and frequency-selective
bank of filters to explicitly retain direction-of-arrival information, whilst
being highly computationally and parametrically efficient than standard 1D/2D
convolution. A dense sound event proposal map is then constructed to handle the
challenges of predicting events with large varying temporal duration.
Accompanying the dense proposal map are a temporal overlapness map and a motion
smoothness map that measure a proposal's confidence to be an event from
temporal detection accuracy and movement consistency perspective. Involving the
two maps guarantees SoundDet to be trained in a spatio-temporally unified
manner. Experimental results on the public DCASE dataset show the advantage of
SoundDet on both segment-based and our newly proposed event-based evaluation
system.

    

### [[2106.07178] A Comprehensive Survey on Graph Anomaly Detection with Deep Learning](http://arxiv.org/abs/2106.07178)


  Anomalies represent rare observations (e.g., data records or events) that are
deviating significantly from others. Over the last forty years, researches on
anomalies have received great interests because of their significance in many
disciplines (e.g., computer science, chemistry, and biology). Anomaly
detection, which aims to identify these rare observations, is among the most
vital tasks and has shown its power in preventing detrimental events, such as
financial fraud and network intrusion, from happening. The detection task is
typically solved by detecting outlying data points in the features space and
inherently overlooks the structural information in real-world data. Graphs have
been prevalently used to preserve the structural information, and this raises
the graph anomaly detection problem - identifying anomalous graph objects
(i.e., nodes, edges and sub-graphs). However, conventional anomaly detection
techniques cannot well solve this problem because of the complexity of graph
data (e.g., irregular structures, non-independent and large-scale). For the
aptitudes of deep learning in breaking these limitations, graph anomaly
detection with deep learning has received intensified studies recently. In this
survey, we aim to provide a systematic and comprehensive review of the
contemporary deep learning techniques for graph anomaly detection.
Specifically, our categorization follows a task-driven strategy and classifies
existing works according to the anomalous graph objects they can detect. We
especially focus on the motivations, key intuitions and technical details of
existing works. We also summarize open-sourced implementations, public
datasets, and commonly-used evaluation metrics for future studies. Finally, we
highlight twelve future research directions according to our survey results
covering emerging problems introduced by graph data, anomaly detection and real
applications.

    

### [[2106.08462] Multi-Resolution Continuous Normalizing Flows](http://arxiv.org/abs/2106.08462)


  Recent work has shown that Neural Ordinary Differential Equations (ODEs) can
serve as generative models of images using the perspective of Continuous
Normalizing Flows (CNFs). Such models offer exact likelihood calculation, and
invertible generation/density estimation. In this work we introduce a
Multi-Resolution variant of such models (MRCNF), by characterizing the
conditional distribution over the additional information required to generate a
fine image that is consistent with the coarse image. We introduce a
transformation between resolutions that allows for no change in the log
likelihood. We show that this approach yields comparable likelihood values for
various image datasets, with improved performance at higher resolutions, with
fewer parameters, using only 1 GPU. Further, we examine the out-of-distribution
properties of (Multi-Resolution) Continuous Normalizing Flows, and find that
they are similar to those of other likelihood-based generative models.

    

### [[2107.00595] Fast Margin Maximization via Dual Acceleration](http://arxiv.org/abs/2107.00595)


  We present and analyze a momentum-based gradient method for training linear
classifiers with an exponentially-tailed loss (e.g., the exponential or
logistic loss), which maximizes the classification margin on separable data at
a rate of $\widetilde{\mathcal{O}}(1/t^2)$. This contrasts with a rate of
$\mathcal{O}(1/\log(t))$ for standard gradient descent, and $\mathcal{O}(1/t)$
for normalized gradient descent. This momentum-based method is derived via the
convex dual of the maximum-margin problem, and specifically by applying
Nesterov acceleration to this dual, which manages to result in a simple and
intuitive method in the primal. This dual view can also be used to derive a
stochastic variant, which performs adaptive non-uniform sampling via the dual
variables.

    

### [[2108.00893] Static analysis of ReLU neural networks with tropical polyhedra](http://arxiv.org/abs/2108.00893)


  This paper studies the problem of range analysis for feedforward neural
networks, which is a basic primitive for applications such as robustness of
neural networks, compliance to specifications and reachability analysis of
neural-network feedback systems. Our approach focuses on ReLU (rectified linear
unit) feedforward neural nets that present specific difficulties: approaches
that exploit derivatives do not apply in general, the number of patterns of
neuron activations can be quite large even for small networks, and convex
approximations are generally too coarse. In this paper, we employ set-based
methods and abstract interpretation that have been very successful in coping
with similar difficulties in classical program verification. We present an
approach that abstracts ReLU feedforward neural networks using tropical
polyhedra. We show that tropical polyhedra can efficiently abstract ReLU
activation function, while being able to control the loss of precision due to
linear computations. We show how the connection between ReLU networks and
tropical rational functions can provide approaches for range analysis of ReLU
neural networks.

    

### [[2108.09601] Programmable FPGA-based Memory Controller](http://arxiv.org/abs/2108.09601)


  Even with generational improvements in DRAM technology, memory access latency
still remains the major bottleneck for application accelerators, primarily due
to limitations in memory interface IPs which cannot fully account for
variations in target applications, the algorithms used, and accelerator
architectures. Since developing memory controllers for different applications
is time-consuming, this paper introduces a modular and programmable memory
controller that can be configured for different target applications on
available hardware resources. The proposed memory controller efficiently
supports cache-line accesses along with bulk memory transfers. The user can
configure the controller depending on the available logic resources on the
FPGA, memory access pattern, and external memory specifications. The modular
design supports various memory access optimization techniques including,
request scheduling, internal caching, and direct memory access. These
techniques contribute to reducing the overall latency while maintaining high
sustained bandwidth. We implement the system on a state-of-the-art FPGA and
evaluate its performance using two widely studied domains: graph analytics and
deep learning workloads. We show improved overall memory access time up to 58%
on CNN and GCN workloads compared with commercial memory controller IPs.

    

### [[2108.09337] On the Parallel I/O Optimality of Linear Algebra Kernels: Near-Optimal Matrix Factorizations](http://arxiv.org/abs/2108.09337)


  Matrix factorizations are among the most important building blocks of
scientific computing. State-of-the-art libraries, however, are not
communication-optimal, underutilizing current parallel architectures. We
present novel algorithms for Cholesky and LU factorizations that utilize an
asymptotically communication-optimal 2.5D decomposition. We first establish a
theoretical framework for deriving parallel I/O lower bounds for linear algebra
kernels, and then utilize its insights to derive Cholesky and LU schedules,
both communicating N^3/(P*sqrt(M)) elements per processor, where M is the local
memory size. The empirical results match our theoretical analysis: our
implementations communicate significantly less than Intel MKL, SLATE, and the
asymptotically communication-optimal CANDMC and CAPITAL libraries. Our code
outperforms these state-of-the-art libraries in almost all tested scenarios,
with matrix sizes ranging from 2,048 to 262,144 on up to 512 CPU nodes of the
Piz Daint supercomputer, decreasing the time-to-solution by up to three times.
Our code is ScaLAPACK-compatible and available as an open-source library.

    

### [[2108.09351] Power Reduction of Automatic Heterogeneous Device Offloading](http://arxiv.org/abs/2108.09351)


  In recent years, utilization of heterogeneous hardware other than small core
CPU such as GPU, FPGA or many core CPU is increasing. However, when using
heterogeneous hardware, barriers of technical skills such as CUDA are high.
Based on that, I have proposed environment-adaptive software that enables
automatic conversion, configuration, and high performance and low power
operation of once written code, according to the hardware to be placed. I also
have verified performance improvement of automatic GPU and FPGA offloading so
far. In this paper, I verify low power operation with environment adaptation by
confirming power utilization after automatic offloading. I compare Watt*seconds
of existing applications after automatic offloading with the case of CPU only
processing.

    

### [[2108.09365] L-DQN: An Asynchronous Limited-Memory Distributed Quasi-Newton Method](http://arxiv.org/abs/2108.09365)


  This work proposes a distributed algorithm for solving empirical risk
minimization problems, called L-DQN, under the master/worker communication
model. L-DQN is a distributed limited-memory quasi-Newton method that supports
asynchronous computations among the worker nodes. Our method is efficient both
in terms of storage and communication costs, i.e., in every iteration the
master node and workers communicate vectors of size $O(d)$, where $d$ is the
dimension of the decision variable, and the amount of memory required on each
node is $O(md)$, where $m$ is an adjustable parameter. To our knowledge, this
is the first distributed quasi-Newton method with provable global linear
convergence guarantees in the asynchronous setting where delays between nodes
are present. Numerical experiments are provided to illustrate the theory and
the practical performance of our method.

    

### [[2108.09403] Deadlock and Noise in Self-Organized Aggregation Without Computation](http://arxiv.org/abs/2108.09403)


  Aggregation is a fundamental behavior for swarm robotics that requires a
system to gather together in a compact, connected cluster. In 2014, Gauci et
al. proposed a surprising algorithm that reliably achieves swarm aggregation
using only a binary line-of-sight sensor and no arithmetic computation or
persistent memory. It has been rigorously proven that this algorithm will
aggregate one robot to another, but it remained open whether it would always
aggregate a system of $n > 2$ robots as was observed in experiments and
simulations. We prove that there exist deadlocked configurations from which
this algorithm cannot achieve aggregation for $n > 3$ robots when the robots'
motion is uniform and deterministic. On the positive side, we show that the
algorithm (i) is robust to small amounts of error, enabling deadlock avoidance,
and (ii) provably achieves a linear runtime speedup for the $n = 2$ case when
using a cone-of-sight sensor. Finally, we introduce a noisy, discrete
adaptation of this algorithm that is more amenable to rigorous analysis of
noise and whose simulation results align qualitatively with the original,
continuous algorithm.

    

### [[2108.09418] Analyzing the Effect of Consistency Violation Faults in Self-Stabilizing Programs](http://arxiv.org/abs/2108.09418)


  Consistency violation faults \cvf{s} refer to faults that occur due to
inconsistent reads in a shared memory program. In the execution of
shared-memory, interleaving, self-stabilizing programs, \cvf{s} offer a
trade-off. Specifically, preventing \cvf{s} requires processes to coordinate
with each other thereby slowing the execution of each action. On the other
hand, permitting \cvf{s} requires less coordination and faster execution of
each action. However, when a \cvf occurs, it can disrupt the convergence of
self-stabilizing programs. Thus, a computation of a program in the presence of
\cvf{s} can be thought of as a contest where program actions attempt to take
the program to a legitimate state whereas \cvf{s} could potentially this
convergence. We analyze three self-stabilizing programs (token ring, coloring
and maximal matching) to evaluate this contest between program transitions and
\cvf{s}. We find that the relative cost of \cvf{s} is generally small, i.e., as
long as a few program transitions can execute between \cvf{s}, the program
(probabilistically) converges. We also find that the cost-distribution of
\cvf{s} is exponential in nature in that the fraction of \cvf{s} with cost $c$
exponentially decreases with $c$. We validate these results via simulation
where we control the rate of \cvf{s}.

    

### [[2108.09457] DeepEdgeBench: Benchmarking Deep Neural Networks on Edge Devices](http://arxiv.org/abs/2108.09457)


  EdgeAI (Edge computing based Artificial Intelligence) has been most actively
researched for the last few years to handle variety of massively distributed AI
applications to meet up the strict latency requirements. Meanwhile, many
companies have released edge devices with smaller form factors (low power
consumption and limited resources) like the popular Raspberry Pi and Nvidia's
Jetson Nano for acting as compute nodes at the edge computing environments.
Although the edge devices are limited in terms of computing power and hardware
resources, they are powered by accelerators to enhance their performance
behavior. Therefore, it is interesting to see how AI-based Deep Neural Networks
perform on such devices with limited resources.
In this work, we present and compare the performance in terms of inference
time and power consumption of the four Systems on a Chip (SoCs): Asus Tinker
Edge R, Raspberry Pi 4, Google Coral Dev Board, Nvidia Jetson Nano, and one
microcontroller: Arduino Nano 33 BLE, on different deep learning models and
frameworks. We also provide a method for measuring power consumption, inference
time and accuracy for the devices, which can be easily extended to other
devices. Our results showcase that, for Tensorflow based quantized model, the
Google Coral Dev Board delivers the best performance, both for inference time
and power consumption. For a low fraction of inference computation time, i.e.
less than 29.3% of the time for MobileNetV2, the Jetson Nano performs faster
than the other devices.

    

### [[2108.09604] The Power of Random Symmetry-Breaking in Nakamoto Consensus](http://arxiv.org/abs/2108.09604)


  Nakamoto consensus underlies the security of many of the world's largest
cryptocurrencies, such as Bitcoin and Ethereum. Common lore is that Nakamoto
consensus only achieves consistency and liveness under a regime where the
difficulty of its underlying mining puzzle is very high, negatively impacting
overall throughput and latency. In this work, we study Nakamoto consensus under
a wide range of puzzle difficulties, including very easy puzzles. We first
analyze an adversary-free setting and show that, surprisingly, the common
prefix of the blockchain grows quickly even with easy puzzles. In a setting
with adversaries, we provide a small backwards-compatible change to Nakamoto
consensus to achieve consistency and liveness with easy puzzles. Our insight
relies on a careful choice of \emph{symmetry-breaking strategy}, which was
significantly underestimated in prior work. We introduce a new method --
\emph{coalescing random walks} -- to analyzing the correctness of Nakamoto
consensus under the uniformly-at-random symmetry-breaking strategy. This method
is more powerful than existing analysis methods that focus on bounding the
number of {\it convergence opportunities}.

    

### [[2108.09615] Apache Submarine: A Unified Machine Learning Platform Made Simple](http://arxiv.org/abs/2108.09615)


  As machine learning is applied more widely, it is necessary to have a machine
learning platform for both infrastructure administrators and users including
expert data scientists and citizen data scientists to improve their
productivity. However, existing machine learning platforms are ill-equipped to
address the "Machine Learning tech debts" such as glue code, reproducibility,
and portability. Furthermore, existing platforms only take expert data
scientists into consideration, and thus they are inflexible for infrastructure
administrators and non-user-friendly for citizen data scientists. We propose
Submarine, a unified machine learning platform, to address the challenges.

    

### [[2108.09899] Rate distortion comparison of a few gradient quantizers](http://arxiv.org/abs/2108.09899)


  This article is in the context of gradient compression. Gradient compression
is a popular technique for mitigating the communication bottleneck observed
when training large machine learning models in a distributed manner using
gradient-based methods such as stochastic gradient descent. In this article,
assuming a Gaussian distribution for the components in gradient, we find the
rate distortion trade-off of gradient quantization schemes such as Scaled-sign
and Top-K, and compare with the Shannon rate distortion limit. A similar
comparison with vector quantizers also is presented.

    

### [[2010.04424] Gathering a Euclidean Closed Chain of Robots in Linear Time](http://arxiv.org/abs/2010.04424)


  This work focuses on the following question related to the Gathering problem
of $n$ autonomous, mobile robots in the Euclidean plane: Is it possible to
solve Gathering of robots that do not agree on any axis of their coordinate
systems (disoriented robots) and see other robots only up to a constant
distance (limited visibility) in $o(n^2)$ fully synchronous rounds? The best
known algorithm that solves Gathering of disoriented robots with limited
visibility assuming oblivious robots needs $\Theta(n^2)$ rounds [SPAA'11]. The
lower bound for this algorithm even holds in a simplified closed chain model,
where each robot has exactly two neighbors and the chain connections form a
cycle. The only existing algorithms achieving a linear number of rounds for
disoriented robots assume robots that are located on a two dimensional grid
[IPDPS'16] and [SPAA'16]. Both algorithms make use of locally visible lights
(the LUMINOUS model).
In this work, we show for the closed chain model, that $n$ disoriented robots
with limited visibility in the Euclidean plane can be gathered in
$\Theta\left(n\right)$ rounds assuming the LUMINOUS model. The lights are used
to initiate and perform so-called runs along the chain. For the start of such
runs, locally unique robots need to be determined. In contrast to the grid
[IPDPS'16], this is not possible in every configuration in the Euclidean plane.
Based on the theory of isogonal polygons by Grünbaum, we identify the class
of isogonal configurations in which no such locally unique robots can be
identified. Our solution combines two algorithms: The first one gathers
isogonal configurations; it works without any lights. The second one works for
non-isogonal configurations; it identifies locally unique robots to start runs,
using a constant number of lights. Interleaving these algorithms solves the
Gathering problem in $\mathcal{O}(n)$ rounds.

    

### [[2012.15443] Automatic Synthesis of Parallel Unix Commands and Pipelines with KumQuat](http://arxiv.org/abs/2012.15443)


  We present KumQuat, a system for automatically generating data parallel
implementations of Unix shell commands and pipelines. The generated parallel
versions split input streams, execute multiple instantiations of the original
pipeline commands to process the splits in parallel, then combine the resulting
parallel outputs to produce the final output stream. KumQuat automatically
synthesizes the combine operators, with a domain-specific combiner language
acting as a strong regularizer that promotes efficient inference of correct
combiners.
We evaluate KumQuat on 70 benchmark scripts that together have a total of 427
stages. KumQuat synthesizes a correct combiner for 113 of the 121 unique
commands that appear in these benchmark scripts. The synthesis times vary
between 39 seconds and 331 seconds with a median of 60 seconds. We present
experimental results that show that these combiners enable the effective
parallelization of our benchmark scripts.

    

### [[2102.12058] A Survey on Consortium Blockchain Consensus Mechanisms](http://arxiv.org/abs/2102.12058)


  Blockchain is a distributed ledger that is decentralized, immutable, and
transparent, which maintains a continuously growing list of transaction records
ordered into blocks. As the core of blockchain, the consensus algorithm is an
agreement to validate the correctness of blockchain transactions. For example,
Bitcoin is a public blockchain where each node in Bitcoin uses the Proof of
Work (PoW) algorithm to reach a consensus by competing to solve a puzzle.
Unlike a public blockchain, a consortium blockchain is an enterprise-level
blockchain that does not contend with the issues of creating a resource-saving
global consensus protocol. This paper highilights several state-of-the art
solutions in consensus algorithms for enterprise blockchain. For example, the
HyperLedger by Linux Foundation includes implementing Practical Byzantine Fault
Tolerance (PBFT) as the consensus algorithm. PBFT can tolerate a range of
malicious nodes and reach consensus with quadratic complexity. Another
consensus algorithm, HotStuff, implemented by Facebook's Libra project, has
achieved linear complexity of the authenticator. This paper presents the
operational mechanisms of these and other consensus protocols, and analyzes and
compares their advantages and drawbacks.

    

### [[2107.00555] Productivity, Portability, Performance: Data-Centric Python](http://arxiv.org/abs/2107.00555)


  Python has become the de facto language for scientific computing. Programming
in Python is highly productive, mainly due to its rich science-oriented
software ecosystem built around the NumPy module. As a result, the demand for
Python support in High Performance Computing (HPC) has skyrocketed. However,
the Python language itself does not necessarily offer high performance. In this
work, we present a workflow that retains Python's high productivity while
achieving portable performance across different architectures. The workflow's
key features are HPC-oriented language extensions and a set of automatic
optimizations powered by a data-centric intermediate representation. We show
performance results and scaling across CPU, GPU, FPGA, and the Piz Daint
supercomputer (up to 23,328 cores), with 2.47x and 3.75x speedups over
previous-best solutions, first-ever Xilinx and Intel FPGA results of annotated
Python, and up to 93.16% scaling efficiency on 512 nodes.

    

### [[2108.09353] Temporally Nonstationary Component Analysis; Application to Noninvasive Fetal Electrocardiogram Extraction](http://arxiv.org/abs/2108.09353)


  Objective: Mixtures of temporally nonstationary signals are very common in
biomedical applications. The nonstationarity of the source signals can be used
as a discriminative property for signal separation. Herein, a semi-blind source
separation algorithm is proposed for the extraction of temporally nonstationary
components from linear multichannel mixtures of signals and noises. Methods: A
hypothesis test is proposed for the detection and fusion of temporally
nonstationary events, by using ad hoc indexes for monitoring the first and
second order statistics of the innovation process. As proof of concept, the
general framework is customized and tested over noninvasive fetal cardiac
recordings acquired from the maternal abdomen, over publicly available
datasets, using two types of nonstationarity detectors: 1) a local power
variations detector, and 2) a model-deviations detector using the innovation
process properties of an extended Kalman filter. Results: The performance of
the proposed method is assessed in presence of white and colored noise, in
different signal-to-noise ratios. Conclusion and Significance: The proposed
scheme is general and it can be used for the extraction of nonstationary events
and sample deviations from a presumed model in multivariate data, which is a
recurrent problem in many machine learning applications.

    

### [[2108.09355] One Chatbot Per Person: Creating Personalized Chatbots based on Implicit User Profiles](http://arxiv.org/abs/2108.09355)


  Personalized chatbots focus on endowing chatbots with a consistent
personality to behave like real users, give more informative responses, and
further act as personal assistants. Existing personalized approaches tried to
incorporate several text descriptions as explicit user profiles. However, the
acquisition of such explicit profiles is expensive and time-consuming, thus
being impractical for large-scale real-world applications. Moreover, the
restricted predefined profile neglects the language behavior of a real user and
cannot be automatically updated together with the change of user interests. In
this paper, we propose to learn implicit user profiles automatically from
large-scale user dialogue history for building personalized chatbots.
Specifically, leveraging the benefits of Transformer on language understanding,
we train a personalized language model to construct a general user profile from
the user's historical responses. To highlight the relevant historical responses
to the input post, we further establish a key-value memory network of
historical post-response pairs, and build a dynamic post-aware user profile.
The dynamic profile mainly describes what and how the user has responded to
similar posts in history. To explicitly utilize users' frequently used words,
we design a personalized decoder to fuse two decoding strategies, including
generating a word from the generic vocabulary and copying one word from the
user's personalized vocabulary. Experiments on two real-world datasets show the
significant improvement of our model compared with existing methods.

    

### [[2108.09372] InBiodiv-O: An Ontology for Indian Biodiversity Knowledge Management](http://arxiv.org/abs/2108.09372)


  To present the biodiversity information, a semantic model is required that
connects all kinds of data about living creatures and their habitats. The model
must be able to encode human knowledge for machines to be understood. Ontology
offers the richest machine-interpretable (rather than just machine-processable)
and explicit semantics that are being extensively used in the biodiversity
domain. Various ontologies are developed for the biodiversity domain however a
review of the current landscape shows that these ontologies are not capable to
define the Indian biodiversity information though India is one of the
megadiverse countries. To semantically analyze the Indian biodiversity
information, it is crucial to build an ontology that describes all the
essential terms of this domain from the unstructured format of the data
available on the web. Since, the curation of the ontologies heavily depends on
the domain where these are implemented hence there is no ideal methodology is
defined yet to be ready for universal use. The aim of this article is to
develop an ontology that semantically encodes all the terms of Indian
biodiversity information in all its dimensions based on the proposed
methodology. The comprehensive evaluation of the proposed ontology depicts that
ontology is well built in the specified domain.

    

### [[2108.09404] Safe Transformative AI via a Windfall Clause](http://arxiv.org/abs/2108.09404)


  Society could soon see transformative artificial intelligence (TAI). Models
of competition for TAI show firms face strong competitive pressure to deploy
TAI systems before they are safe. This paper explores a proposed solution to
this problem, a Windfall Clause, where developers commit to donating a
significant portion of any eventual extremely large profits to good causes.
However, a key challenge for a Windfall Clause is that firms must have reason
to join one. Firms must also believe these commitments are credible. We extend
a model of TAI competition with a Windfall Clause to show how firms and
policymakers can design a Windfall Clause which overcomes these challenges.
Encouragingly, firms benefit from joining a Windfall Clause under a wide range
of scenarios. We also find that firms join the Windfall Clause more often when
the competition is more dangerous. Even when firms learn each other's
capabilities, firms rarely wish to withdraw their support for the Windfall
Clause. These three findings strengthen the case for using a Windfall Clause to
promote the safe development of TAI.

    

### [[2108.09432] ARAPReg: An As-Rigid-As Possible Regularization Loss for Learning Deformable Shape Generators](http://arxiv.org/abs/2108.09432)


  This paper introduces an unsupervised loss for training parametric
deformation shape generators. The key idea is to enforce the preservation of
local rigidity among the generated shapes. Our approach builds on an
approximation of the as-rigid-as possible (or ARAP) deformation energy. We show
how to develop the unsupervised loss via a spectral decomposition of the
Hessian of the ARAP energy. Our loss nicely decouples pose and shape variations
through a robust norm. The loss admits simple closed-form expressions. It is
easy to train and can be plugged into any standard generation models, e.g.,
variational auto-encoder (VAE) and auto-decoder (AD). Experimental results show
that our approach outperforms existing shape generation approaches considerably
on public benchmark datasets of various shape categories such as human, animal
and bone.

    

### [[2108.09443] Towards Personalized and Human-in-the-Loop Document Summarization](http://arxiv.org/abs/2108.09443)


  The ubiquitous availability of computing devices and the widespread use of
the internet have generated a large amount of data continuously. Therefore, the
amount of available information on any given topic is far beyond humans'
processing capacity to properly process, causing what is known as information
overload. To efficiently cope with large amounts of information and generate
content with significant value to users, we require identifying, merging and
summarising information. Data summaries can help gather related information and
collect it into a shorter format that enables answering complicated questions,
gaining new insight and discovering conceptual boundaries.
This thesis focuses on three main challenges to alleviate information
overload using novel summarisation techniques. It further intends to facilitate
the analysis of documents to support personalised information extraction. This
thesis separates the research issues into four areas, covering (i) feature
engineering in document summarisation, (ii) traditional static and inflexible
summaries, (iii) traditional generic summarisation approaches, and (iv) the
need for reference summaries. We propose novel approaches to tackle these
challenges, by: i)enabling automatic intelligent feature engineering, ii)
enabling flexible and interactive summarisation, iii) utilising intelligent and
personalised summarisation approaches. The experimental results prove the
efficiency of the proposed approaches compared to other state-of-the-art
models. We further propose solutions to the information overload problem in
different domains through summarisation, covering network traffic data, health
data and business process data.

    

### [[2108.09451] Learn-Explain-Reinforce: Counterfactual Reasoning and Its Guidance to Reinforce an Alzheimer's Disease Diagnosis Model](http://arxiv.org/abs/2108.09451)


  Existing studies on disease diagnostic models focus either on diagnostic
model learning for performance improvement or on the visual explanation of a
trained diagnostic model. We propose a novel learn-explain-reinforce (LEAR)
framework that unifies diagnostic model learning, visual explanation generation
(explanation unit), and trained diagnostic model reinforcement (reinforcement
unit) guided by the visual explanation. For the visual explanation, we generate
a counterfactual map that transforms an input sample to be identified as an
intended target label. For example, a counterfactual map can localize
hypothetical abnormalities within a normal brain image that may cause it to be
diagnosed with Alzheimer's disease (AD). We believe that the generated
counterfactual maps represent data-driven and model-induced knowledge about a
target task, i.e., AD diagnosis using structural MRI, which can be a vital
source of information to reinforce the generalization of the trained diagnostic
model. To this end, we devise an attention-based feature refinement module with
the guidance of the counterfactual maps. The explanation and reinforcement
units are reciprocal and can be operated iteratively. Our proposed approach was
validated via qualitative and quantitative analysis on the ADNI dataset. Its
comprehensibility and fidelity were demonstrated through ablation studies and
comparisons with existing methods.

    

### [[2108.09473] Robust Ensembling Network for Unsupervised Domain Adaptation](http://arxiv.org/abs/2108.09473)


  Recently, in order to address the unsupervised domain adaptation (UDA)
problem, extensive studies have been proposed to achieve transferrable models.
Among them, the most prevalent method is adversarial domain adaptation, which
can shorten the distance between the source domain and the target domain.
Although adversarial learning is very effective, it still leads to the
instability of the network and the drawbacks of confusing category information.
In this paper, we propose a Robust Ensembling Network (REN) for UDA, which
applies a robust time ensembling teacher network to learn global information
for domain transfer. Specifically, REN mainly includes a teacher network and a
student network, which performs standard domain adaptation training and updates
weights of the teacher network. In addition, we also propose a dual-network
conditional adversarial loss to improve the ability of the discriminator.
Finally, for the purpose of improving the basic ability of the student network,
we utilize the consistency constraint to balance the error between the student
network and the teacher network. Extensive experimental results on several UDA
datasets have demonstrated the effectiveness of our model by comparing with
other state-of-the-art UDA algorithms.

    

### [[2108.09491] Flikcer -- A Chrome Extension to Resolve Online Epileptogenic Visual Content with Real-Time Luminance Frequency Analysis](http://arxiv.org/abs/2108.09491)


  Video content with fast luminance variations, or with spatial patterns of
high contrast - referred to as epileptogenic visual content - may induce
seizures on viewers with photosensitive epilepsy, and even cause discomfort in
users not affected by this disease. Flikcer is a web app in the form of a
website and chrome extension which aims to resolve epileptic content in videos.
It provides the number of possible triggers for a seizure. It also provides the
timestamps for these triggers along with a safer version of the video, free to
download. The algorithm is written in Python and uses machine learning and
computer vision. A key aspect of the algorithm is its computational efficiency,
allowing real time implementation for public users.

    

### [[2108.09556] A generalized forecasting solution to enable future insights of COVID-19 at sub-national level resolutions](http://arxiv.org/abs/2108.09556)


  COVID-19 continues to cause a significant impact on public health. To
minimize this impact, policy makers undertake containment measures that
however, when carried out disproportionately to the actual threat, as a result
if errorneous threat assessment, cause undesirable long-term socio-economic
complications. In addition, macro-level or national level decision making fails
to consider the localized sensitivities in small regions. Hence, the need
arises for region-wise threat assessments that provide insights on the
behaviour of COVID-19 through time, enabled through accurate forecasts. In this
study, a forecasting solution is proposed, to predict daily new cases of
COVID-19 in regions small enough where containment measures could be locally
implemented, by targeting three main shortcomings that exist in literature; the
unreliability of existing data caused by inconsistent testing patterns in
smaller regions, weak deploy-ability of forecasting models towards predicting
cases in previously unseen regions, and model training biases caused by the
imbalanced nature of data in COVID-19 epi-curves. Hence, the contributions of
this study are three-fold; an optimized smoothing technique to smoothen less
deterministic epi-curves based on epidemiological dynamics of that region, a
Long-Short-Term-Memory (LSTM) based forecasting model trained using data from
select regions to create a representative and diverse training set that
maximizes deploy-ability in regions with lack of historical data, and an
adaptive loss function whilst training to mitigate the data imbalances seen in
epi-curves. The proposed smoothing technique, the generalized training strategy
and the adaptive loss function largely increased the overall accuracy of the
forecast, which enables efficient containment measures at a more localized
micro-level.

    

### [[2108.09586] Learning Causal Models of Autonomous Agents using Interventions](http://arxiv.org/abs/2108.09586)


  One of the several obstacles in the widespread use of AI systems is the lack
of requirements of interpretability that can enable a layperson to ensure the
safe and reliable behavior of such systems. We extend the analysis of an agent
assessment module that lets an AI system execute high-level instruction
sequences in simulators and answer the user queries about its execution of
sequences of actions. We show that such a primitive query-response capability
is sufficient to efficiently derive a user-interpretable causal model of the
system in stationary, fully observable, and deterministic settings. We also
introduce dynamic causal decision networks (DCDNs) that capture the causal
structure of STRIPS-like domains. A comparative analysis of different classes
of queries is also presented in terms of the computational requirements needed
to answer them and the efforts required to evaluate their responses to learn
the correct model.

    

### [[2108.09597] Hierarchical Summarization for Longform Spoken Dialog](http://arxiv.org/abs/2108.09597)


  Every day we are surrounded by spoken dialog. This medium delivers rich
diverse streams of information auditorily; however, systematically
understanding dialog can often be non-trivial. Despite the pervasiveness of
spoken dialog, automated speech understanding and quality information
extraction remains markedly poor, especially when compared to written prose.
Furthermore, compared to understanding text, auditory communication poses many
additional challenges such as speaker disfluencies, informal prose styles, and
lack of structure. These concerns all demonstrate the need for a distinctly
speech tailored interactive system to help users understand and navigate the
spoken language domain. While individual automatic speech recognition (ASR) and
text summarization methods already exist, they are imperfect technologies;
neither consider user purpose and intent nor address spoken language induced
complications. Consequently, we design a two stage ASR and text summarization
pipeline and propose a set of semantic segmentation and merging algorithms to
resolve these speech modeling challenges. Our system enables users to easily
browse and navigate content as well as recover from errors in these underlying
technologies. Finally, we present an evaluation of the system which highlights
user preference for hierarchical summarization as a tool to quickly skim audio
and identify content of interest to the user.

    

### [[2108.09628] DisenKGAT: Knowledge Graph Embedding with Disentangled Graph Attention Network](http://arxiv.org/abs/2108.09628)


  Knowledge graph completion (KGC) has become a focus of attention across deep
learning community owing to its excellent contribution to numerous downstream
tasks. Although recently have witnessed a surge of work on KGC, they are still
insufficient to accurately capture complex relations, since they adopt the
single and static representations. In this work, we propose a novel
Disentangled Knowledge Graph Attention Network (DisenKGAT) for KGC, which
leverages both micro-disentanglement and macro-disentanglement to exploit
representations behind Knowledge graphs (KGs). To achieve
micro-disentanglement, we put forward a novel relation-aware aggregation to
learn diverse component representation. For macro-disentanglement, we leverage
mutual information as a regularization to enhance independence. With the
assistance of disentanglement, our model is able to generate adaptive
representations in terms of the given scenario. Besides, our work has strong
robustness and flexibility to adapt to various score functions. Extensive
experiments on public benchmark datasets have been conducted to validate the
superiority of DisenKGAT over existing methods in terms of both accuracy and
explainability.

    

### [[2108.09638] Signed Bipartite Graph Neural Networks](http://arxiv.org/abs/2108.09638)


  Signed networks are such social networks having both positive and negative
links. A lot of theories and algorithms have been developed to model such
networks (e.g., balance theory). However, previous work mainly focuses on the
unipartite signed networks where the nodes have the same type. Signed bipartite
networks are different from classical signed networks, which contain two
different node sets and signed links between two node sets. Signed bipartite
networks can be commonly found in many fields including business, politics, and
academics, but have been less studied. In this work, we firstly define the
signed relationship of the same set of nodes and provide a new perspective for
analyzing signed bipartite networks. Then we do some comprehensive analysis of
balance theory from two perspectives on several real-world datasets.
Specifically, in the peer review dataset, we find that the ratio of balanced
isomorphism in signed bipartite networks increased after rebuttal phases.
Guided by these two perspectives, we propose a novel Signed Bipartite Graph
Neural Networks (SBGNNs) to learn node embeddings for signed bipartite
networks. SBGNNs follow most GNNs message-passing scheme, but we design new
message functions, aggregation functions, and update functions for signed
bipartite networks. We validate the effectiveness of our model on four
real-world datasets on Link Sign Prediction task, which is the main machine
learning task for signed networks. Experimental results show that our SBGNN
model achieves significant improvement compared with strong baseline methods,
including feature-based methods and network embedding methods.

    

### [[2108.09823] Embodied AI-Driven Operation of Smart Cities: A Concise Review](http://arxiv.org/abs/2108.09823)


  A smart city can be seen as a framework, comprised of Information and
Communication Technologies (ICT). An intelligent network of connected devices
that collect data with their sensors and transmit them using cloud technologies
in order to communicate with other assets in the ecosystem plays a pivotal role
in this framework. Maximizing the quality of life of citizens, making better
use of resources, cutting costs, and improving sustainability are the ultimate
goals that a smart city is after. Hence, data collected from connected devices
will continuously get thoroughly analyzed to gain better insights into the
services that are being offered across the city; with this goal in mind that
they can be used to make the whole system more efficient. Robots and physical
machines are inseparable parts of a smart city. Embodied AI is the field of
study that takes a deeper look into these and explores how they can fit into
real-world environments. It focuses on learning through interaction with the
surrounding environment, as opposed to Internet AI which tries to learn from
static datasets. Embodied AI aims to train an agent that can See (Computer
Vision), Talk (NLP), Navigate and Interact with its environment (Reinforcement
Learning), and Reason (General Intelligence), all at the same time. Autonomous
driving cars and personal companions are some of the examples that benefit from
Embodied AI nowadays. In this paper, we attempt to do a concise review of this
field. We will go through its definitions, its characteristics, and its current
achievements along with different algorithms, approaches, and solutions that
are being used in different components of it (e.g. Vision, NLP, RL). We will
then explore all the available simulators and 3D interactable databases that
will make the research in this area feasible. Finally, we will address its
challenges and identify its potentials for future research.

    

### [[2108.09936] Voxel-based Network for Shape Completion by Leveraging Edge Generation](http://arxiv.org/abs/2108.09936)


  Deep learning technique has yielded significant improvements in point cloud
completion with the aim of completing missing object shapes from partial
inputs. However, most existing methods fail to recover realistic structures due
to over-smoothing of fine-grained details. In this paper, we develop a
voxel-based network for point cloud completion by leveraging edge generation
(VE-PCN). We first embed point clouds into regular voxel grids, and then
generate complete objects with the help of the hallucinated shape edges. This
decoupled architecture together with a multi-scale grid feature learning is
able to generate more realistic on-surface details. We evaluate our model on
the publicly available completion datasets and show that it outperforms
existing state-of-the-art approaches quantitatively and qualitatively. Our
source code is available at this https URL.

    

### [[2108.09954] Pulse-Width Modulation Neuron Implemented by Single Positive-Feedback Device](http://arxiv.org/abs/2108.09954)


  Positive-feedback (PF) device and its operation scheme to implement pulse
width modulation (PWM) function was proposed and demonstrated, and the device
operation mechanism for implementing PWM function was analyzed. By adjusting
the amount of the charge stored in the n- floating body (Qn), the potential of
the floating body linearly changes with time. When Qn reaches to a threshold
value (Qth), the PF device turns on abruptly. From the linear time-varying
property of Qn and the gate bias dependency of Qth, fully functionable PWM
neuron properties including voltage to pulse width conversion and hard-sigmoid
activation function were successfully obtained from a single PF device. A PWM
neuron can be implemented by using a single PF device, thus it is beneficial to
extremely reduce the area of a PWM neuron circuit than the previously reported
one.

    

### [[2108.09988] Farsighted Probabilistic Sampling based Local Search for (Weighted) Partial MaxSAT](http://arxiv.org/abs/2108.09988)


  Partial MaxSAT (PMS) and Weighted Partial MaxSAT (WPMS) are both practical
generalizations to the typical combinatorial problem of MaxSAT. In this work,
we propose an effective farsighted probabilistic sampling based local search
algorithm called FPS for solving these two problems, denoted as (W)PMS. The FPS
algorithm replaces the mechanism of flipping a single variable per iteration
step, that is widely used in existing (W)PMS local search algorithms, with the
proposed farsighted local search strategy, and provides higher-quality local
optimal solutions. The farsighted strategy employs the probabilistic sampling
technique that allows the algorithm to look-ahead widely and efficiently. In
this way, FPS can provide more and better search directions and improve the
performance without reducing the efficiency. Extensive experiments on all the
benchmarks of (W)PMS problems from the incomplete track of recent four years of
MaxSAT Evaluations demonstrate that our method significantly outperforms
SATLike3.0, the state-of-the-art local search algorithm, for solving both the
PMS and WPMS problems. We furthermore do comparison with the extended solver of
SATLike, SATLike-c, which is the champion of three categories among the total
four (PMS and WPMS categories, each associated with two time limits) of the
incomplete track in the recent MaxSAT Evaluation (MSE2021). We replace the
local search component in SATLike-c with the proposed farsighted sampling local
search approach, and the resulting solver FPS-c also outperforms SATLike-c for
solving both the PMS and WPMS problems.

    

### [[2108.09996] MS-DARTS: Mean-Shift Based Differentiable Architecture Search](http://arxiv.org/abs/2108.09996)


  Differentiable Architecture Search (DARTS) is an effective continuous
relaxation-based network architecture search (NAS) method with low search cost.
It has attracted significant attentions in Auto-ML research and becomes one of
the most useful paradigms in NAS. Although DARTS can produce superior
efficiency over traditional NAS approaches with better control of complex
parameters, oftentimes it suffers from stabilization issues in producing
deteriorating architectures when discretizing the continuous architecture. We
observed considerable loss of validity causing dramatic decline in performance
at this final discretization step of DARTS. To address this issue, we propose a
Mean-Shift based DARTS (MS-DARTS) to improve stability based on sampling and
perturbation. Our approach can improve bot the stability and accuracy of DARTS,
by smoothing the loss landscape and sampling architecture parameters within a
suitable bandwidth. We investigate the convergence of our mean-shift approach,
together with the effects of bandwidth selection that affects stability and
accuracy. Evaluations performed on CIFAR-10, CIFAR-100, and ImageNet show that
MS-DARTS archives higher performance over other state-of-the-art NAS methods
with reduced search cost.

    

### [[2108.10005] Credit Card Fraud Detection using Machine Learning: A Study](http://arxiv.org/abs/2108.10005)


  As the world is rapidly moving towards digitization and money transactions
are becoming cashless, the use of credit cards has rapidly increased. The fraud
activities associated with it have also been increasing which leads to a huge
loss to the financial institutions. Therefore, we need to analyze and detect
the fraudulent transaction from the non-fraudulent ones. In this paper, we
present a comprehensive review of various methods used to detect credit card
fraud. These methodologies include Hidden Markov Model, Decision Trees,
Logistic Regression, Support Vector Machines (SVM), Genetic algorithm, Neural
Networks, Random Forests, Bayesian Belief Network. A comprehensive analysis of
various techniques is presented. We conclude the paper with the pros and cons
of the same as stated in the respective papers.

    

### [[2108.10008] BiaSwap: Removing dataset bias with bias-tailored swapping augmentation](http://arxiv.org/abs/2108.10008)


  Deep neural networks often make decisions based on the spurious correlations
inherent in the dataset, failing to generalize in an unbiased data
distribution. Although previous approaches pre-define the type of dataset bias
to prevent the network from learning it, recognizing the bias type in the real
dataset is often prohibitive. This paper proposes a novel bias-tailored
augmentation-based approach, BiaSwap, for learning debiased representation
without requiring supervision on the bias type. Assuming that the bias
corresponds to the easy-to-learn attributes, we sort the training images based
on how much a biased classifier can exploits them as shortcut and divide them
into bias-guiding and bias-contrary samples in an unsupervised manner.
Afterwards, we integrate the style-transferring module of the image translation
model with the class activation maps of such biased classifier, which enables
to primarily transfer the bias attributes learned by the classifier. Therefore,
given the pair of bias-guiding and bias-contrary, BiaSwap generates the
bias-swapped image which contains the bias attributes from the bias-contrary
images, while preserving bias-irrelevant ones in the bias-guiding images. Given
such augmented images, BiaSwap demonstrates the superiority in debiasing
against the existing baselines over both synthetic and real-world datasets.
Even without careful supervision on the bias, BiaSwap achieves a remarkable
performance on both unbiased and bias-guiding samples, implying the improved
generalization capability of the model.

    

### [[2108.10021] QDEF and Its Approximations in OBDM](http://arxiv.org/abs/2108.10021)


  Given an input dataset (i.e., a set of tuples), query definability in
Ontology-based Data Management (OBDM) amounts to find a query over the ontology
whose certain answers coincide with the tuples in the given dataset. We refer
to such a query as a characterization of the dataset with respect to the OBDM
system. Our first contribution is to propose approximations of perfect
characterizations in terms of recall (complete characterizations) and precision
(sound characterizations). A second contribution is to present a thorough
complexity analysis of three computational problems, namely verification (check
whether a given query is a perfect, or an approximated characterization of a
given dataset), existence (check whether a perfect, or a best approximated
characterization of a given dataset exists), and computation (compute a
perfect, or best approximated characterization of a given dataset).

    

### [[2108.10046] Discovering Spatial Relationships by Transformers for Domain Generalization](http://arxiv.org/abs/2108.10046)


  Due to the rapid increase in the diversity of image data, the problem of
domain generalization has received increased attention recently. While domain
generalization is a challenging problem, it has achieved great development
thanks to the fast development of AI techniques in computer vision. Most of
these advanced algorithms are proposed with deep architectures based on
convolution neural nets (CNN). However, though CNNs have a strong ability to
find the discriminative features, they do a poor job of modeling the relations
between different locations in the image due to the response to CNN filters are
mostly local. Since these local and global spatial relationships are
characterized to distinguish an object under consideration, they play a
critical role in improving the generalization ability against the domain gap.
In order to get the object parts relationships to gain better domain
generalization, this work proposes to use the self attention model. However,
the attention models are proposed for sequence, which are not expert in
discriminate feature extraction for 2D images. Considering this, we proposed a
hybrid architecture to discover the spatial relationships between these local
features, and derive a composite representation that encodes both the
discriminative features and their relationships to improve the domain
generalization. Evaluation on three well-known benchmarks demonstrates the
benefits of modeling relationships between the features of an image using the
proposed method and achieves state-of-the-art domain generalization
performance. More specifically, the proposed algorithm outperforms the
state-of-the-art by $2.2\%$ and $3.4\%$ on PACS and Office-Home databases,
respectively.

    

### [[2108.10062] EEG-based Classification of Drivers Attention using Convolutional Neural Network](http://arxiv.org/abs/2108.10062)


  Accurate detection of a drivers attention state can help develop assistive
technologies that respond to unexpected hazards in real time and therefore
improve road safety. This study compares the performance of several attention
classifiers trained on participants brain activity. Participants performed a
driving task in an immersive simulator where the car randomly deviated from the
cruising lane. They had to correct the deviation and their response time was
considered as an indicator of attention level. Participants repeated the task
in two sessions; in one session they received kinesthetic feedback and in
another session no feedback. Using their EEG signals, we trained three
attention classifiers; a support vector machine (SVM) using EEG spectral band
powers, and a Convolutional Neural Network (CNN) using either spectral features
or the raw EEG data. Our results indicated that the CNN model trained on raw
EEG data obtained under kinesthetic feedback achieved the highest accuracy
(89%). While using a participants own brain activity to train the model
resulted in the best performances, inter-subject transfer learning still
performed high (75%), showing promise for calibration-free Brain-Computer
Interface (BCI) systems. Our findings show that CNN and raw EEG signals can be
employed for effective training of a passive BCI for real-time attention
classification.

    

### [[2001.07578] Adequate and fair explanations](http://arxiv.org/abs/2001.07578)


  Explaining sophisticated machine-learning based systems is an important issue
at the foundations of AI. Recent efforts have shown various methods for
providing explanations. These approaches can be broadly divided into two
schools: those that provide a local and human interpreatable approximation of a
machine learning algorithm, and logical approaches that exactly characterise
one aspect of the decision. In this paper we focus upon the second school of
exact explanations with a rigorous logical foundation. There is an
epistemological problem with these exact methods. While they can furnish
complete explanations, such explanations may be too complex for humans to
understand or even to write down in human readable form. Interpretability
requires epistemically accessible explanations, explanations humans can grasp.
Yet what is a sufficiently complete epistemically accessible explanation still
needs clarification. We do this here in terms of counterfactuals, following
[Wachter et al., 2017]. With counterfactual explanations, many of the
assumptions needed to provide a complete explanation are left implicit. To do
so, counterfactual explanations exploit the properties of a particular data
point or sample, and as such are also local as well as partial explanations. We
explore how to move from local partial explanations to what we call complete
local explanations and then to global ones. But to preserve accessibility we
argue for the need for partiality. This partiality makes it possible to hide
explicit biases present in the algorithm that may be injurious or unfair.We
investigate how easy it is to uncover these biases in providing complete and
fair explanations by exploiting the structure of the set of counterfactuals
providing a complete local explanation.

    

### [[2005.05538] Dynamic Cognition Applied to Value Learning in Artificial Intelligence](http://arxiv.org/abs/2005.05538)


  Experts in Artificial Intelligence (AI) development predict that advances in
the development of intelligent systems and agents will reshape vital areas in
our society. Nevertheless, if such an advance isn't done with prudence, it can
result in negative outcomes for humanity. For this reason, several researchers
in the area are trying to develop a robust, beneficial, and safe concept of
artificial intelligence. Currently, several of the open problems in the field
of AI research arise from the difficulty of avoiding unwanted behaviors of
intelligent agents, and at the same time specifying what we want such systems
to do. It is of utmost importance that artificial intelligent agents have their
values aligned with human values, given the fact that we cannot expect an AI to
develop our moral preferences simply because of its intelligence, as discussed
in the Orthogonality Thesis. Perhaps this difficulty comes from the way we are
addressing the problem of expressing objectives, values, and ends, using
representational cognitive methods. A solution to this problem would be the
dynamic cognitive approach proposed by Dreyfus, whose phenomenological
philosophy defends that the human experience of being-in-the-world cannot be
represented by the symbolic or connectionist cognitive methods. A possible
approach to this problem would be to use theoretical models such as SED
(situated embodied dynamics) to address the values learning problem in AI.

    

### [[2009.06141] Composing Answer from Multi-spans for Reading Comprehension](http://arxiv.org/abs/2009.06141)


  This paper presents a novel method to generate answers for non-extraction
machine reading comprehension (MRC) tasks whose answers cannot be simply
extracted as one span from the given passages. Using a pointer network-style
extractive decoder for such type of MRC may result in unsatisfactory
performance when the ground-truth answers are given by human annotators or
highly re-paraphrased from parts of the passages. On the other hand, using
generative decoder cannot well guarantee the resulted answers with well-formed
syntax and semantics when encountering long sentences. Therefore, to alleviate
the obvious drawbacks of both sides, we propose an answer making-up method from
extracted multi-spans that are learned by our model as highly confident
$n$-gram candidates in the given passage. That is, the returned answers are
composed of discontinuous multi-spans but not just one consecutive span in the
given passages anymore. The proposed method is simple but effective: empirical
experiments on MS MARCO show that the proposed method has a better performance
on accurately generating long answers, and substantially outperforms two
competitive typical one-span and Seq2Seq baseline decoders.

    

### [[2101.02032] Socially Responsible AI Algorithms: Issues, Purposes, and Challenges](http://arxiv.org/abs/2101.02032)


  In the current era, people and society have grown increasingly reliant on
artificial intelligence (AI) technologies. AI has the potential to drive us
towards a future in which all of humanity flourishes. It also comes with
substantial risks for oppression and calamity. Discussions about whether we
should (re)trust AI have repeatedly emerged in recent years and in many
quarters, including industry, academia, healthcare, services, and so on.
Technologists and AI researchers have a responsibility to develop trustworthy
AI systems. They have responded with great effort to design more responsible AI
algorithms. However, existing technical solutions are narrow in scope and have
been primarily directed towards algorithms for scoring or classification tasks,
with an emphasis on fairness and unwanted bias. To build long-lasting trust
between AI and human beings, we argue that the key is to think beyond
algorithmic fairness and connect major aspects of AI that potentially cause
AI's indifferent behavior. In this survey, we provide a systematic framework of
Socially Responsible AI Algorithms that aims to examine the subjects of AI
indifference and the need for socially responsible AI algorithms, define the
objectives, and introduce the means by which we may achieve these objectives.
We further discuss how to leverage this framework to improve societal
well-being through protection, information, and prevention/mitigation.

    

### [[2103.10455] 3D Human Pose Estimation with Spatial and Temporal Transformers](http://arxiv.org/abs/2103.10455)


  Transformer architectures have become the model of choice in natural language
processing and are now being introduced into computer vision tasks such as
image classification, object detection, and semantic segmentation. However, in
the field of human pose estimation, convolutional architectures still remain
dominant. In this work, we present PoseFormer, a purely transformer-based
approach for 3D human pose estimation in videos without convolutional
architectures involved. Inspired by recent developments in vision transformers,
we design a spatial-temporal transformer structure to comprehensively model the
human joint relations within each frame as well as the temporal correlations
across frames, then output an accurate 3D human pose of the center frame. We
quantitatively and qualitatively evaluate our method on two popular and
standard benchmark datasets: Human3.6M and MPI-INF-3DHP. Extensive experiments
show that PoseFormer achieves state-of-the-art performance on both datasets.
Code is available at \url{this https URL}

    

### [[2104.00312] Normal vs. Adversarial: Salience-based Analysis of Adversarial Samples for Relation Extraction](http://arxiv.org/abs/2104.00312)


  Recent neural-based relation extraction approaches, though achieving
promising improvement on benchmark datasets, have reported their vulnerability
towards adversarial attacks. Thus far, efforts mostly focused on generating
adversarial samples or defending adversarial attacks, but little is known about
the difference between normal and adversarial samples. In this work, we take
the first step to leverage the salience-based method to analyze those
adversarial samples. We observe that salience tokens have a direct correlation
with adversarial perturbations. We further find the adversarial perturbations
are either those tokens not existing in the training set or superficial cues
associated with relation labels. To some extent, our approach unveils the
characters against adversarial samples. We release an open-source testbed,
"DiagnoseAdv".

    

### [[2104.04907] Disentangled Contrastive Learning for Learning Robust Textual Representations](http://arxiv.org/abs/2104.04907)


  Although the self-supervised pre-training of transformer models has resulted
in the revolutionizing of natural language processing (NLP) applications and
the achievement of state-of-the-art results with regard to various benchmarks,
this process is still vulnerable to small and imperceptible permutations
originating from legitimate inputs. Intuitively, the representations should be
similar in the feature space with subtle input permutations, while large
variations occur with different meanings. This motivates us to investigate the
learning of robust textual representation in a contrastive manner. However, it
is non-trivial to obtain opposing semantic instances for textual samples. In
this study, we propose a disentangled contrastive learning method that
separately optimizes the uniformity and alignment of representations without
negative sampling. Specifically, we introduce the concept of momentum
representation consistency to align features and leverage power normalization
while conforming the uniformity. Our experimental results for the NLP
benchmarks demonstrate that our approach can obtain better results compared
with the baselines, as well as achieve promising improvements with invariance
tests and adversarial attacks. The code is available in
this https URL.

    

### [[2108.09534] Theoretical Analysis and Evaluation of NoCs with Weighted Round-Robin Arbitration](http://arxiv.org/abs/2108.09534)


  Fast and accurate performance analysis techniques are essential in early
design space exploration and pre-silicon evaluations, including software
eco-system development. In particular, on-chip communication continues to play
an increasingly important role as the many-core processors scale up. This paper
presents the first performance analysis technique that targets networks-on-chip
(NoCs) that employ weighted round-robin (WRR) arbitration. Besides fairness,
WRR arbitration provides flexibility in allocating bandwidth proportionally to
the importance of the traffic classes, unlike basic round-robin and
priority-based arbitration. The proposed approach first estimates the effective
service time of the packets in the queue due to WRR arbitration. Then, it uses
the effective service time to compute the average waiting time of the packets.
Next, we incorporate a decomposition technique to extend the analytical model
to handle NoC of any size. The proposed approach achieves less than 5% error
while executing real applications and 10% error under challenging synthetic
traffic with different burstiness levels.

    

### [[1912.11661] Large fork-join queues with nearly deterministic arrival and service times](http://arxiv.org/abs/1912.11661)


  In this paper, we study an $N$ server fork-join queue with nearly
deterministic arrival and service times. Specifically, we present a fluid limit
for the maximum queue length as $N\to\infty$. This fluid limit depends on the
initial number of tasks. In order to prove these results, we develop extreme
value theory and diffusion approximations for the queue lengths.

    

### [[2108.09624] Proceedings Combined 28th International Workshop on Expressiveness in Concurrency and 18th Workshop on Structural Operational Semantics](http://arxiv.org/abs/2108.09624)


  This volume contains the proceedings of EXPRESS/SOS 2021: the Combined 28th
International Workshop on Expressiveness in Concurrency and the 18th Workshop
on Structural Operational Semantics, which was held online, as an affiliated
workshop of CONCUR 2021, the 32nd International Conference on Concurrency
Theory. The EXPRESS/SOS workshop series aims at bringing together researchers
interested in the formal semantics of systems and programming concepts, and in
the expressiveness of computational models.

    

### [[2108.09744] Bugs4Q: A Benchmark of Real Bugs for Quantum Programs](http://arxiv.org/abs/2108.09744)


  Realistic benchmarks of reproducible bugs and fixes are vital to good
experimental evaluation of debugging and testing approaches. However, there is
no suitable benchmark suite that can systematically evaluate the debugging and
testing methods of quantum programs until now. This paper proposes Bugs4Q, a
benchmark of thirty-six real, manually validated Qiskit bugs from four popular
Qiskit elements (Terra, Aer, Ignis, and Aqua), supplemented with the test cases
for reproducing buggy behaviors. Bugs4Q also provides interfaces for accessing
the buggy and fixed versions of the Qiskit programs and executing the
corresponding test cases, facilitating the reproducible empirical studies and
comparisons of Qiskit program debugging and testing tools. Bugs4Q is publicly
available at this https URL.

    

### [[2108.09753] Custom-Tailored Clone Detection for IEC 61131-3 Programming Languages](http://arxiv.org/abs/2108.09753)


  Automated production systems (aPS) are highly customized systems that consist
of hardware and software. Such aPS are controlled by a programmable logic
controller (PLC), often in accordance with the IEC 61131-3 standard that
divides system implementation into so-called program organization units (POUs)
as the smallest software unit and is comprised of multiple textual and
graphical programming languages that can be arbitrarily nested. A common
practice during the development of such systems is reusing implementation
artifacts by copying, pasting, and then modifying code. This approach is
referred to as code cloning. It is used on a fine-granular level where a POU is
cloned within a system variant. It is also applied on the coarse-granular
system level, where the entire system is cloned and adapted to create a system
variant, for example for another customer. This ad hoc practice for the
development of variants is commonly referred to as clone-and-own. It allows the
fast development of variants to meet varying customer requirements or altered
regulatory guidelines. However, clone-and-own is a non-sustainable approach and
does not scale with an increasing number of variants. It has a detrimental
effect on the overall quality of a software system, such as the propagation of
bugs to other variants, which harms maintenance. In order to support the
effective development and maintenance of such systems, a detailed code clone
analysis is required. On the one hand, an analysis of code clones within a
variant (i.e., clone detection in the classical sense) supports experts in
refactoring respective code into library components. On the other hand, an
analysis of commonalities and differences between cloned variants (i.e.,
variability analysis) supports the maintenance and further reuse and
facilitates the migration of variants into a software product line (SPL).

    

### [[1902.04836] Differentials and distances in probabilistic coherence spaces](http://arxiv.org/abs/1902.04836)


  In probabilistic coherence spaces, a denotational model of probabilistic
functional languages, mor-phisms are analytic and therefore smooth. We explore
two related applications of the corresponding derivatives. First we show how
derivatives allow to compute the expectation of execution time in the weak head
reduction of probabilistic PCF (pPCF). Next we apply a general notion of
"local" differential of morphisms to the proof of a Lipschitz property of these
morphisms allowing in turn to relate the observational distance on pPCF terms
to a distance the model is naturally equipped with. This suggests that
extending probabilistic programming languages with derivatives, in the spirit
of the differential lambda-calculus, could be quite meaningful.

    

### [[2101.09699] Longest segment of balanced parentheses -- an exercise in program inversion in a segment problem (Functional Pearl)](http://arxiv.org/abs/2101.09699)


  Given a string of parentheses, the task is to find the longest consecutive
segment that is balanced, in linear time. We find this problem interesting
because it involves a combination of techniques: the usual approach for solving
segment problems, and a theorem for constructing the inverse of a function --
through which we derive an instance of shift-reduce parsing.

    

### [[2103.02976] Contextual Modal Types for Algebraic Effects and Handlers](http://arxiv.org/abs/2103.02976)


  Programming languages with algebraic effects often track the computations'
effects using type-and-effect systems. In this paper, we propose to view an
algebraic effect theory of a computation as a variable context; consequently,
we propose to track algebraic effects of a computation with \emph{contextual
modal types}. We develop ECMTT, a novel calculus which tracks algebraic effects
by a contextualized variant of the modal $\Box$ (necessity) operator, that it
inherits from Contextual Modal Type Theory (CMTT).
Whereas type-and-effect systems add effect annotations on top of a prior
programming language, the effect annotations in ECMTT are inherent to the
language, as they are managed by programming constructs corresponding to the
logical introduction and elimination forms for the $\Box$ modality. Thus, the
type-and-effect system of ECMTT is actually just a type system.
Our design obtains the properties of local soundness and completeness, and
determines the operational semantics solely by $\beta$-reduction, as customary
in other logic-based calculi. In this view, effect handlers arise naturally as
a witness that one context (i.e., algebraic theory) can be reached from
another, generalizing explicit substitutions from CMTT.
To the best of our knowledge, ECMTT is the first system to relate algebraic
effects to modal types. We also see it as a step towards providing a
correspondence in the style of Curry and Howard that may transfer a number of
results from the fields of modal logic and modal type theory to that of
algebraic effects.

    