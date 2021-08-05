
## 2021-8-5

### [[2108.01885] Intelligent Sensing Scheduling for Mobile Target Tracking Wireless Sensor Networks](http://arxiv.org/abs/2108.01885)


  Edge computing has emerged as a prospective paradigm to meet ever-increasing
computation demands in Mobile Target Tracking Wireless Sensor Networks
(MTT-WSN). This paradigm can offload time-sensitive tasks to sink nodes to
improve computing efficiency. Nevertheless, it is difficult to execute dynamic
and critical tasks in the MTT-WSN network. Besides, the network cannot ensure
consecutive tracking due to the limited energy. To address the problems, this
paper proposes a new hierarchical target tracking structure based on Edge
Intelligence (EI) technology. The structure integrates the computing resource
of both mobile nodes and edge servers to provide efficient computation
capability for real-time target tracking. Based on the proposed structure, we
formulate an energy optimization model with the constrains of system execution
latency and trajectory prediction accuracy. Moreover, we propose a long-term
dynamic resource allocation algorithm to obtain the optimal resource allocation
solution for the ac- curate and consecutive tracking. Simulation results
demonstrate that our algorithm outperforms the deep Q-learning over 14.5% in
terms of system energy consumption. It can also obtain a significant
enhancement in tracking accuracy compared with the non-cooperative scheme.

    

### [[2108.01919] When to Preprocess? Keeping Information Fresh for Computing Enable Internet of Things](http://arxiv.org/abs/2108.01919)


  Age of information (AoI), a notion that measures the information freshness,
is an essential performance measure for time-critical applications in Internet
of Things (IoT). With the surge of computing resources at the IoT devices, it
is possible to preprocess the information packets that contain the status
update before sending them to the destination so as to alleviate the
transmission burden. However, the additional time and energy expenditure
induced by computing also make the optimal updating a non-trivial problem. In
this paper, we consider a time-critical IoT system, where the IoT device is
capable of preprocessing the status update before the transmission.
Particularly, we aim to jointly design the preprocessing and transmission so
that the weighted sum of the average AoI of the destination and the energy
consumption of the IoT device is minimized. Due to the heterogeneity in
transmission and computation capacities, the durations of distinct actions of
the IoT device are non-uniform. Therefore, we formulate the status updating
problem as an infinite horizon average cost semi-Markov decision process (SMDP)
and then transform it into a discrete-time Markov decision process. We
demonstrate that the optimal policy is of threshold type with respect to the
AoI. Equipped with this, a structure-aware relative policy iteration algorithm
is proposed to obtain the optimal policy of the SMDP. Our analysis shows that
preprocessing is more beneficial in regimes of high AoIs, given it can reduce
the time required for updates. We further prove the switching structure of the
optimal policy in a special scenario, where the status updates are transmitted
over a reliable channel, and derive the optimal threshold. Finally, simulation
results demonstrate the efficacy of preprocessing and show that the proposed
policy outperforms two baseline policies.

    

### [[2108.02008] Personal Devices for Contact Tracing: Smartphones and Wearables to Fight Covid-19](http://arxiv.org/abs/2108.02008)


  Digital contact tracing has emerged as a viable tool supplementing manual
contact tracing. To date, more than 100 contact tracing applications have been
published to slow down the spread of highly contagious Covid-19. Despite subtle
variabilities among these applications, all of them achieve contact tracing by
manipulating the following three components: a) use a personal device to
identify the user while designing a secure protocol to anonymize the user's
identity; b) leverage networking technologies to analyze and store the data; c)
exploit rich sensing features on the user device to detect the interaction
among users and thus estimate the exposure risk. This paper reviews the current
digital contact tracing based on these three components. We focus on two
personal devices that are intimate to the user: smartphones and wearables. We
discuss the centralized and decentralized networking approaches that use to
facilitate the data flow. Lastly, we investigate the sensing feature available
on smartphones and wearables to detect the proximity between any two users and
present experiments comparing the proximity sensing performance between these
two personal devices.

    

### [[2108.02136] Randomized Local Fast Rerouting for Datacenter Networks with Almost Optimal Congestion](http://arxiv.org/abs/2108.02136)


  To ensure high availability, datacenter networks must rely on local fast
rerouting mechanisms that allow routers to quickly react to link failures, in a
fully decentralized manner. However, configuring these mechanisms to provide a
high resilience against multiple failures while avoiding congestion along
failover routes is algorithmically challenging, as the rerouting rules can only
depend on local failure information and must be defined ahead of time. This
paper presents a randomized local fast rerouting algorithm for Clos networks,
the predominant datacenter topologies. Given a graph $G=(V,E)$ describing a
Clos topology, our algorithm defines local routing rules for each node $v\in
V$, which only depend on the packet's destination and are conditioned on the
incident link failures. We prove that as long as number of failures at each
node does not exceed a certain bound, our algorithm achieves an asymptotically
minimal congestion up to polyloglog factors along failover paths. Our lower
bounds are developed under some natural routing assumptions.

    

### [[2101.10632] A Survey on Data Plane Programming with P4: Fundamentals, Advances, and Applied Research](http://arxiv.org/abs/2101.10632)


  Programmable data planes allow users to define their own data plane
algorithms for network devices including appropriate data plane application
programming interfaces (APIs) which may be leveraged by user-defined
software-defined networking (SDN) control. This offers great flexibility for
network customization, be it for specialized, commercial appliances, e.g., in
5G or data center networks, or for rapid prototyping in industrial and academic
research. Programming protocol-independent packet processors (P4) has emerged
as the currently most widespread abstraction, programming language, and concept
for data plane programming. It is developed and standardized by an open
community, and it is supported by various software and hardware platforms. In
the first part of this paper we give a tutorial of data plane programming
models, the P4 programming language, architectures, compilers, targets, and
data plane APIs. We also consider research efforts to advance P4 technology. In
the second part, we categorize a large body of literature of P4-based applied
research into different research domains, summarize the contributions of these
papers, and extract prototypes, target platforms, and source code availability.
For each research domain, we analyze how the reviewed works benefit from P4's
core features. Finally, we discuss potential next steps based on our findings.

    

### [[2108.01699] Predicting Zip Code-Level Vaccine Hesitancy in US Metropolitan Areas Using Machine Learning Models on Public Tweets](http://arxiv.org/abs/2108.01699)


  Although the recent rise and uptake of COVID-19 vaccines in the United States
has been encouraging, there continues to be significant vaccine hesitancy in
various geographic and demographic clusters of the adult population. Surveys,
such as the one conducted by Gallup over the past year, can be useful in
determining vaccine hesitancy, but can be expensive to conduct and do not
provide real-time data. At the same time, the advent of social media suggests
that it may be possible to get vaccine hesitancy signals at an aggregate level
(such as at the level of zip codes) by using machine learning models and
socioeconomic (and other) features from publicly available sources. It is an
open question at present whether such an endeavor is feasible, and how it
compares to baselines that only use constant priors. To our knowledge, a proper
methodology and evaluation results using real data has also not been presented.
In this article, we present such a methodology and experimental study, using
publicly available Twitter data collected over the last year. Our goal is not
to devise novel machine learning algorithms, but to evaluate existing and
established models in a comparative framework. We show that the best models
significantly outperform constant priors, and can be set up using open-source
tools.

    

### [[2108.01701] Categorical EHR Imputation with Generative Adversarial Nets](http://arxiv.org/abs/2108.01701)


  Electronic Health Records often suffer from missing data, which poses a major
problem in clinical practice and clinical studies. A novel approach for dealing
with missing data are Generative Adversarial Nets (GANs), which have been
generating huge research interest in image generation and transformation.
Recently, researchers have attempted to apply GANs to missing data generation
and imputation for EHR data: a major challenge here is the categorical nature
of the data. State-of-the-art solutions to the GAN-based generation of
categorical data involve either reinforcement learning, or learning a
bidirectional mapping between the categorical and the real latent feature
space, so that the GANs only need to generate real-valued features. However,
these methods are designed to generate complete feature vectors instead of
imputing only the subsets of missing features. In this paper we propose a
simple and yet effective approach that is based on previous work on GANs for
data imputation. We first motivate our solution by discussing the reason why
adversarial training often fails in case of categorical features. Then we
derive a novel way to re-code the categorical features to stabilize the
adversarial training. Based on experiments on two real-world EHR data with
multiple settings, we show that our imputation approach largely improves the
prediction accuracy, compared to more traditional data imputation approaches.

    

### [[2108.01711] Improving Music Performance Assessment with Contrastive Learning](http://arxiv.org/abs/2108.01711)


  Several automatic approaches for objective music performance assessment (MPA)
have been proposed in the past, however, existing systems are not yet capable
of reliably predicting ratings with the same accuracy as professional judges.
This study investigates contrastive learning as a potential method to improve
existing MPA systems. Contrastive learning is a widely used technique in
representation learning to learn a structured latent space capable of
separately clustering multiple classes. It has been shown to produce state of
the art results for image-based classification problems. We introduce a
weighted contrastive loss suitable for regression tasks applied to a
convolutional neural network and show that contrastive loss results in
performance gains in regression tasks for MPA. Our results show that
contrastive-based methods are able to match and exceed SoTA performance for MPA
regression tasks by creating better class clusters within the latent space of
the neural networks.

    

### [[2108.01724] Approximating Attributed Incentive Salience In Large Scale Scenarios. A Representation Learning Approach Based on Artificial Neural Networks](http://arxiv.org/abs/2108.01724)


  Incentive salience attribution can be understood as a psychobiological
process ascribing relevance to potentially rewarding objects and actions.
Despite being an important component of the motivational process guiding our
everyday behaviour its study in naturalistic contexts is not straightforward.
Here we propose a methodology based on artificial neural networks (ANNs) for
approximating latent states produced by this process in situations where large
volumes of behavioural data are available but no strict experimental control is
possible. Leveraging knowledge derived from theoretical and computational
accounts of incentive salience attribution we designed an ANN for estimating
duration and intensity of future interactions between individuals and a series
of video games in a large-scale ($N> 3 \times 10^6$) longitudinal dataset.
Through model comparison and inspection we show that our approach outperforms
competing ones while also generating a representation that well approximate
some of the functions of attributed incentive salience. We discuss our findings
with reference to the adopted theoretical and computational frameworks and
suggest how our methodology could be an initial step for estimating attributed
incentive salience in large scale behavioural studies.

    

### [[2108.01728] A Study on Herd Behavior Using Sentiment Analysis in Online Social Network](http://arxiv.org/abs/2108.01728)


  Social media platforms are thriving nowadays, so a huge volume of data is
produced. As it includes brief and clear statements, millions of people post
their thoughts on microblogging sites every day. This paper represents and
analyze the capacity of diverse strategies to volumetric, delicate, and social
networks to predict critical opinions from online social networking sites. In
the exploration of certain searching for relevant, the thoughts of people play
a crucial role. Social media becomes a good outlet since the last decades to
share the opinions globally. Sentiment analysis as well as opinion mining is a
tool that is used to extract the opinions or thoughts of the common public. An
occurrence in one place, be it economic, political, or social, may trigger
large-scale chain public reaction across many other sites in an increasingly
interconnected world. This study demonstrates the evaluation of sentiment
analysis techniques using social media contents and creating the association
between subjectivity with herd behavior and clustering coefficient as well as
tries to predict the election result (2021 election in West Bengal). This is an
implementation of sentiment analysis targeted at estimating the results of an
upcoming election by assessing the public's opinion across social media. This
paper also has a short discussion section on the usefulness of the idea in
other fields.

    

### [[2108.01731] Scalable Community Detection via Parallel Correlation Clustering](http://arxiv.org/abs/2108.01731)


  Graph clustering and community detection are central problems in modern data
mining. The increasing need for analyzing billion-scale data calls for faster
and more scalable algorithms for these problems. There are certain trade-offs
between the quality and speed of such clustering algorithms. In this paper, we
design scalable algorithms that achieve high quality when evaluated based on
ground truth. We develop a generalized sequential and shared-memory parallel
framework based on the LambdaCC objective (introduced by Veldt et al.), which
encompasses modularity and correlation clustering. Our framework consists of
highly-optimized implementations that scale to large data sets of billions of
edges and that obtain high-quality clusters compared to ground-truth data, on
both unweighted and weighted graphs. Our empirical evaluation shows that this
framework improves the state-of-the-art trade-offs between speed and quality of
scalable community detection. For example, on a 30-core machine with two-way
hyper-threading, our implementations achieve orders of magnitude speedups over
other correlation clustering baselines, and up to 28.44x speedups over our own
sequential baselines while maintaining or improving quality.

    

### [[2108.01758] Factor Representation and Decision Making in Stock Markets Using Deep Reinforcement Learning](http://arxiv.org/abs/2108.01758)


  Deep Reinforcement learning is a branch of unsupervised learning in which an
agent learns to act based on environment state in order to maximize its total
reward. Deep reinforcement learning provides good opportunity to model the
complexity of portfolio choice in high-dimensional and data-driven environment
by leveraging the powerful representation of deep neural networks. In this
paper, we build a portfolio management system using direct deep reinforcement
learning to make optimal portfolio choice periodically among S\&P500 underlying
stocks by learning a good factor representation (as input). The result shows
that an effective learning of market conditions and optimal portfolio
allocations can significantly outperform the average market.

    

### [[2108.01763] HTTP2vec: Embedding of HTTP Requests for Detection of Anomalous Traffic](http://arxiv.org/abs/2108.01763)


  Hypertext transfer protocol (HTTP) is one of the most widely used protocols
on the Internet. As a consequence, most attacks (i.e., SQL injection, XSS) use
HTTP as the transport mechanism. Therefore, it is crucial to develop an
intelligent solution that would allow to effectively detect and filter out
anomalies in HTTP traffic. Currently, most of the anomaly detection systems are
either rule-based or trained using manually selected features. We propose
utilizing modern unsupervised language representation model for embedding HTTP
requests and then using it to classify anomalies in the traffic. The solution
is motivated by methods used in Natural Language Processing (NLP) such as
Doc2Vec which could potentially capture the true understanding of HTTP
messages, and therefore improve the efficiency of Intrusion Detection System.
In our work, we not only aim at generating a suitable embedding space, but also
at the interpretability of the proposed model. We decided to use the current
state-of-the-art RoBERTa, which, as far as we know, has never been used in a
similar problem. To verify how the solution would work in real word conditions,
we train the model using only legitimate traffic. We also try to explain the
results based on clusters that occur in the vectorized requests space and a
simple logistic regression classifier. We compared our approach with the
similar, previously proposed methods. We evaluate the feasibility of our method
on three different datasets: CSIC2010, CSE-CIC-IDS2018 and one that we prepared
ourselves. The results we show are comparable to others or better, and most
importantly - interpretable.

    

### [[2108.01769] An Empirical Evaluation of End-to-End Polyphonic Optical Music Recognition](http://arxiv.org/abs/2108.01769)


  Previous work has shown that neural architectures are able to perform optical
music recognition (OMR) on monophonic and homophonic music with high accuracy.
However, piano and orchestral scores frequently exhibit polyphonic passages,
which add a second dimension to the task. Monophonic and homophonic music can
be described as homorhythmic, or having a single musical rhythm. Polyphonic
music, on the other hand, can be seen as having multiple rhythmic sequences, or
voices, concurrently. We first introduce a workflow for creating large-scale
polyphonic datasets suitable for end-to-end recognition from sheet music
publicly available on the MuseScore forum. We then propose two novel
formulations for end-to-end polyphonic OMR -- one treating the problem as a
type of multi-task binary classification, and the other treating it as
multi-sequence detection. Building upon the encoder-decoder architecture and an
image encoder proposed in past work on end-to-end OMR, we propose two novel
decoder models -- FlagDecoder and RNNDecoder -- that correspond to the two
formulations. Finally, we compare the empirical performance of these end-to-end
approaches to polyphonic OMR and observe a new state-of-the-art performance
with our multi-sequence detection decoder, RNNDecoder.

    

### [[2108.01772] Nonconvex Factorization and Manifold Formulations are Almost Equivalent in Low-rank Matrix Optimization](http://arxiv.org/abs/2108.01772)


  In this paper, we consider the geometric landscape connection of the widely
studied manifold and factorization formulations in low-rank positive
semidefinite (PSD) and general matrix optimization. We establish an equivalence
on the set of first-order stationary points (FOSPs) and second-order stationary
points (SOSPs) between the manifold and the factorization formulations. We
further give a sandwich inequality on the spectrum of Riemannian and Euclidean
Hessians at FOSPs, which can be used to transfer more geometric properties from
one formulation to another. Similarities and differences on the landscape
connection under the PSD case and the general case are discussed. To the best
of our knowledge, this is the first geometric landscape connection between the
manifold and the factorization formulations for handling rank constraints. In
the general low-rank matrix optimization, the landscape connection of two
factorization formulations (unregularized and regularized ones) is also
provided. By applying these geometric landscape connections, we are able to
solve unanswered questions in literature and establish stronger results in the
applications on geometric analysis of phase retrieval, well-conditioned
low-rank matrix optimization, and the role of regularization in factorization
arising from machine learning and signal processing.

    

### [[2108.01789] Monte Carlo Tree Search for high precision manufacturing](http://arxiv.org/abs/2108.01789)


  Monte Carlo Tree Search (MCTS) has shown its strength for a lot of
deterministic and stochastic examples, but literature lacks reports of
applications to real world industrial processes. Common reasons for this are
that there is no efficient simulator of the process available or there exist
problems in applying MCTS to the complex rules of the process. In this paper,
we apply MCTS for optimizing a high-precision manufacturing process that has
stochastic and partially observable outcomes. We make use of an
expert-knowledge-based simulator and adapt the MCTS default policy to deal with
the manufacturing process.

    

### [[2108.01808] An Effective Leaf Recognition Using Convolutional Neural Networks Based Features](http://arxiv.org/abs/2108.01808)


  There is a warning light for the loss of plant habitats worldwide that
entails concerted efforts to conserve plant biodiversity. Thus, plant species
classification is of crucial importance to address this environmental
challenge. In recent years, there is a considerable increase in the number of
studies related to plant taxonomy. While some researchers try to improve their
recognition performance using novel approaches, others concentrate on
computational optimization for their framework. In addition, a few studies are
diving into feature extraction to gain significantly in terms of accuracy. In
this paper, we propose an effective method for the leaf recognition problem. In
our proposed approach, a leaf goes through some pre-processing to extract its
refined color image, vein image, xy-projection histogram, handcrafted shape,
texture features, and Fourier descriptors. These attributes are then
transformed into a better representation by neural network-based encoders
before a support vector machine (SVM) model is utilized to classify different
leaves. Overall, our approach performs a state-of-the-art result on the Flavia
leaf dataset, achieving the accuracy of 99.58\% on test sets under random
10-fold cross-validation and bypassing the previous methods. We also release
our codes\footnote{Scripts are available at
\url{this https URL}} for contributing to
the research community in the leaf classification problem.

    

### [[2108.01810] Deep Learning Chromatic and Clique Numbers of Graphs](http://arxiv.org/abs/2108.01810)


  Deep neural networks have been applied to a wide range of problems across
different application domains with great success. Recently, research into
combinatorial optimization problems in particular has generated much interest
in the machine learning community. In this work, we develop deep learning
models to predict the chromatic number and maximum clique size of graphs, both
of which represent classical NP-complete combinatorial optimization problems
encountered in graph theory. The neural networks are trained using the most
basic representation of the graph, the adjacency matrix, as opposed to
undergoing complex domain-specific feature engineering. The experimental
results show that deep neural networks, and in particular convolutional neural
networks, obtain strong performance on this problem.

    

### [[2108.01828] Emergent Discrete Communication in SemanticSpaces](http://arxiv.org/abs/2108.01828)


  Neural agents trained in reinforcement learning settings can learn to
communicate among themselves via discrete tokens, accomplishing as a team what
agents would be unable to do alone. However, the current standard of using
one-hot vectors as discrete communication tokens prevents agents from acquiring
more desirable aspects of communication such as zero-shot understanding.
Inspired by word embedding techniques from natural language processing, we
propose neural agent architectures that enables them to communicate via
discrete tokens derived from a learned, continuous space. We show in a decision
theoretic framework that our technique optimizes communication over a wide
range of scenarios, whereas one-hot tokens are only optimal under restrictive
assumptions. In self-play experiments, we validate that our trained agents
learn to cluster tokens in semantically-meaningful ways, allowing them
communicate in noisy environments where other techniques fail. Lastly, we
demonstrate both that agents using our method can effectively respond to novel
human communication and that humans can understand unlabeled emergent agent
communication, outperforming the use of one-hot communication.

    

### [[2108.01832] Offline Decentralized Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2108.01832)


  In many real-world multi-agent cooperative tasks, due to high cost and risk,
agents cannot interact with the environment and collect experiences during
learning, but have to learn from offline datasets. However, the transition
probabilities calculated from the dataset can be much different from the
transition probabilities induced by the learned policies of other agents,
creating large errors in value estimates. Moreover, the experience
distributions of agents' datasets may vary wildly due to diverse behavior
policies, causing large difference in value estimates between agents.
Consequently, agents will learn uncoordinated suboptimal policies. In this
paper, we propose MABCQ, which exploits value deviation and transition
normalization to modify the transition probabilities. Value deviation
optimistically increases the transition probabilities of high-value next
states, and transition normalization normalizes the biased transition
probabilities of next states. They together encourage agents to discover
potential optimal and coordinated policies. Mathematically, we prove the
convergence of Q-learning under the non-stationary transition probabilities
after modification. Empirically, we show that MABCQ greatly outperforms
baselines and reduces the difference in value estimates between agents.

    

### [[2108.01843] Model-Based Opponent Modeling](http://arxiv.org/abs/2108.01843)


  When one agent interacts with a multi-agent environment, it is challenging to
deal with various opponents unseen before. Modeling the behaviors, goals, or
beliefs of opponents could help the agent adjust its policy to adapt to
different opponents. In addition, it is also important to consider opponents
who are learning simultaneously or capable of reasoning. However, existing work
usually tackles only one of the aforementioned types of opponent. In this
paper, we propose model-based opponent modeling (MBOM), which employs the
environment model to adapt to all kinds of opponent. MBOM simulates the
recursive reasoning process in the environment model and imagines a set of
improving opponent policies. To effectively and accurately represent the
opponent policy, MBOM further mixes the imagined opponent policies according to
the similarity with the real behaviors of opponents. Empirically, we show that
MBOM achieves more effective adaptation than existing methods in competitive
and cooperative environments, respectively with different types of opponent,
i.e., fixed policy, naïve learner, and reasoning learner.

    

### [[2108.01846] Learning Barrier Certificates: Towards Safe Reinforcement Learning with Zero Training-time Violations](http://arxiv.org/abs/2108.01846)


  Training-time safety violations have been a major concern when we deploy
reinforcement learning algorithms in the real world. This paper explores the
possibility of safe RL algorithms with zero training-time safety violations in
the challenging setting where we are only given a safe but trivial-reward
initial policy without any prior knowledge of the dynamics model and additional
offline data. We propose an algorithm, Co-trained Barrier Certificate for Safe
RL (CRABS), which iteratively learns barrier certificates, dynamics models, and
policies. The barrier certificates, learned via adversarial training, ensure
the policy's safety assuming calibrated learned dynamics model. We also add a
regularization term to encourage larger certified regions to enable better
exploration. Empirical simulations show that zero safety violations are already
challenging for a suite of simple environments with only 2-4 dimensional state
space, especially if high-reward policies have to visit regions near the safety
boundary. Prior methods require hundreds of violations to achieve decent
rewards on these tasks, whereas our proposed algorithms incur zero violations.

    

### [[2108.01851] Risk Conditioned Neural Motion Planning](http://arxiv.org/abs/2108.01851)


  Risk-bounded motion planning is an important yet difficult problem for
safety-critical tasks. While existing mathematical programming methods offer
theoretical guarantees in the context of constrained Markov decision processes,
they either lack scalability in solving larger problems or produce conservative
plans. Recent advances in deep reinforcement learning improve scalability by
learning policy networks as function approximators. In this paper, we propose
an extension of soft actor critic model to estimate the execution risk of a
plan through a risk critic and produce risk-bounded policies efficiently by
adding an extra risk term in the loss function of the policy network. We define
the execution risk in an accurate form, as opposed to approximating it through
a summation of immediate risks at each time step that leads to conservative
plans. Our proposed model is conditioned on a continuous spectrum of risk
bounds, allowing the user to adjust the risk-averse level of the agent on the
fly. Through a set of experiments, we show the advantage of our model in terms
of both computational time and plan quality, compared to a state-of-the-art
mathematical programming baseline, and validate its performance in more
complicated scenarios, including nonlinear dynamics and larger state space.

    

### [[2108.01854] Efficient Neural Architecture Search with Performance Prediction](http://arxiv.org/abs/2108.01854)


  Neural networks are powerful models that have a remarkable ability to extract
patterns that are too complex to be noticed by humans or other machine learning
models. Neural networks are the first class of models that can train end-to-end
systems with large learning capacities. However, we still have the difficult
challenge of designing the neural network, which requires human experience and
a long process of trial and error. As a solution, we can use a neural
architecture search to find the best network architecture for the task at hand.
Existing NAS algorithms generally evaluate the fitness of a new architecture by
fully training from scratch, resulting in the prohibitive computational cost,
even if operated on high-performance computers. In this paper, an end-to-end
offline performance predictor is proposed to accelerate the evaluation of
sampled architectures.
Index Terms- Learning Curve Prediction, Neural Architecture Search,
Reinforcement Learning.

    

### [[2108.01862] Reconstructing a dynamical system and forecasting time series by self-consistent deep learning](http://arxiv.org/abs/2108.01862)


  We introduce a self-consistent deep-learning framework which, for a noisy
deterministic time series, provides unsupervised filtering, state-space
reconstruction, identification of the underlying differential equations and
forecasting. Without a priori information on the signal, we embed the time
series in a state space, where deterministic structures, i.e. attractors, are
revealed. Under the assumption that the evolution of solution trajectories is
described by an unknown dynamical system, we filter out stochastic outliers.
The embedding function, the solution trajectories and the dynamical systems are
constructed using deep neural networks, respectively. By exploiting the
differentiability of the neural solution trajectory, the neural dynamical
system is defined locally at each time, mitigating the need for propagating
gradients through numerical solvers. On a chaotic time series masked by
additive Gaussian noise, we demonstrate the filtering ability and the
predictive power of the proposed framework.

    

### [[2108.01867] A Pragmatic Look at Deep Imitation Learning](http://arxiv.org/abs/2108.01867)


  The introduction of the generative adversarial imitation learning (GAIL)
algorithm has spurred the development of scalable imitation learning approaches
using deep neural networks. The GAIL objective can be thought of as 1) matching
the expert policy's state distribution; 2) penalising the learned policy's
state distribution; and 3) maximising entropy. While theoretically motivated,
in practice GAIL can be difficult to apply, not least due to the instabilities
of adversarial training. In this paper, we take a pragmatic look at GAIL and
related imitation learning algorithms. We implement and automatically tune a
range of algorithms in a unified experimental setup, presenting a fair
evaluation between the competing methods. From our results, our primary
recommendation is to consider non-adversarial methods. Furthermore, we discuss
the common components of imitation learning objectives, and present promising
avenues for future research.

    

### [[2108.01869] Learning Task Agnostic Skills with Data-driven Guidance](http://arxiv.org/abs/2108.01869)


  To increase autonomy in reinforcement learning, agents need to learn useful
behaviours without reliance on manually designed reward functions. To that end,
skill discovery methods have been used to learn the intrinsic options available
to an agent using task-agnostic objectives. However, without the guidance of
task-specific rewards, emergent behaviours are generally useless due to the
under-constrained problem of skill discovery in complex and high-dimensional
spaces. This paper proposes a framework for guiding the skill discovery towards
the subset of expert-visited states using a learned state projection. We apply
our method in various reinforcement learning (RL) tasks and show that such a
projection results in more useful behaviours.

    

### [[2108.01887] PARADISE: Exploiting Parallel Data for Multilingual Sequence-to-Sequence Pretraining](http://arxiv.org/abs/2108.01887)


  Despite the success of multilingual sequence-to-sequence pretraining, most
existing approaches rely on monolingual corpora, and do not make use of the
strong cross-lingual signal contained in parallel data. In this paper, we
present PARADISE (PARAllel & Denoising Integration in SEquence-to-sequence
models), which extends the conventional denoising objective used to train these
models by (i) replacing words in the noised sequence according to a
multilingual dictionary, and (ii) predicting the reference translation
according to a parallel corpus instead of recovering the original sequence. Our
experiments on machine translation and cross-lingual natural language inference
show an average improvement of 2.0 BLEU points and 6.7 accuracy points from
integrating parallel data into pretraining, respectively, obtaining results
that are competitive with several popular models at a fraction of their
computational cost.

    

### [[2108.01895] High Performance Across Two Atari Paddle Games Using the Same Perceptual Control Architecture Without Training](http://arxiv.org/abs/2108.01895)


  Deep reinforcement learning (DRL) requires large samples and a long training
time to operate optimally. Yet humans rarely require long periods training to
perform well on novel tasks, such as computer games, once they are provided
with an accurate program of instructions. We used perceptual control theory
(PCT) to construct a simple closed-loop model which requires no training
samples and training time within a video game study using the Arcade Learning
Environment (ALE). The model was programmed to parse inputs from the
environment into hierarchically organised perceptual signals, and it computed a
dynamic error signal by subtracting the incoming signal for each perceptual
variable from a reference signal to drive output signals to reduce this error.
We tested the same model across two different Atari paddle games Breakout and
Pong to achieve performance at least as high as DRL paradigms, and close to
good human performance. Our study shows that perceptual control models, based
on simple assumptions, can perform well without learning. We conclude by
specifying a parsimonious role of learning that may be more similar to
psychological functioning.

    

### [[2108.01899] Generic Neural Architecture Search via Regression](http://arxiv.org/abs/2108.01899)


  Most existing neural architecture search (NAS) algorithms are dedicated to
the downstream tasks, e.g., image classification in computer vision. However,
extensive experiments have shown that, prominent neural architectures, such as
ResNet in computer vision and LSTM in natural language processing, are
generally good at extracting patterns from the input data and perform well on
different downstream tasks. These observations inspire us to ask: Is it
necessary to use the performance of specific downstream tasks to evaluate and
search for good neural architectures? Can we perform NAS effectively and
efficiently while being agnostic to the downstream task? In this work, we
attempt to affirmatively answer the above two questions and improve the
state-of-the-art NAS solution by proposing a novel and generic NAS framework,
termed Generic NAS (GenNAS). GenNAS does not use task-specific labels but
instead adopts \textit{regression} on a set of manually designed synthetic
signal bases for architecture evaluation. Such a self-supervised regression
task can effectively evaluate the intrinsic power of an architecture to capture
and transform the input signal patterns, and allow more sufficient usage of
training samples. We then propose an automatic task search to optimize the
combination of synthetic signals using limited downstream-task-specific labels,
further improving the performance of GenNAS. We also thoroughly evaluate
GenNAS's generality and end-to-end NAS performance on all search spaces, which
outperforms almost all existing works with significant speedup.

    

### [[2108.01903] Personalized Federated Learning with Clustering: Non-IID Heart Rate Variability Data Application](http://arxiv.org/abs/2108.01903)


  While machine learning techniques are being applied to various fields for
their exceptional ability to find complex relations in large datasets, the
strengthening of regulations on data ownership and privacy is causing
increasing difficulty in its application to medical data. In light of this,
Federated Learning has recently been proposed as a solution to train on private
data without breach of confidentiality. This conservation of privacy is
particularly appealing in the field of healthcare, where patient data is highly
confidential. However, many studies have shown that its assumption of
Independent and Identically Distributed data is unrealistic for medical data.
In this paper, we propose Personalized Federated Cluster Models, a hierarchical
clustering-based FL process, to predict Major Depressive Disorder severity from
Heart Rate Variability. By allowing clients to receive more personalized model,
we address problems caused by non-IID data, showing an accuracy increase in
severity prediction. This increase in performance may be sufficient to use
Personalized Federated Cluster Models in many existing Federated Learning
scenarios.

    

### [[2108.01913] Secure and Privacy-Preserving Federated Learning via Co-Utility](http://arxiv.org/abs/2108.01913)


  The decentralized nature of federated learning, that often leverages the
power of edge devices, makes it vulnerable to attacks against privacy and
security. The privacy risk for a peer is that the model update she computes on
her private data may, when sent to the model manager, leak information on those
private data. Even more obvious are security attacks, whereby one or several
malicious peers return wrong model updates in order to disrupt the learning
process and lead to a wrong model being learned. In this paper we build a
federated learning framework that offers privacy to the participating peers as
well as security against Byzantine and poisoning attacks. Our framework
consists of several protocols that provide strong privacy to the participating
peers via unlinkable anonymity and that are rationally sustainable based on the
co-utility property. In other words, no rational party is interested in
deviating from the proposed protocols. We leverage the notion of co-utility to
build a decentralized co-utile reputation management system that provides
incentives for parties to adhere to the protocols. Unlike privacy protection
via differential privacy, our approach preserves the values of model updates
and hence the accuracy of plain federated learning; unlike privacy protection
via update aggregation, our approach preserves the ability to detect bad model
updates while substantially reducing the computational overhead compared to
methods based on homomorphic encryption.

    

### [[2108.01928] How to Query Language Models?](http://arxiv.org/abs/2108.01928)


  Large pre-trained language models (LMs) are capable of not only recovering
linguistic but also factual and commonsense knowledge. To access the knowledge
stored in mask-based LMs, we can use cloze-style questions and let the model
fill in the blank. The flexibility advantage over structured knowledge bases
comes with the drawback of finding the right query for a certain information
need. Inspired by human behavior to disambiguate a question, we propose to
query LMs by example. To clarify the ambivalent question "Who does Neuer play
for?", a successful strategy is to demonstrate the relation using another
subject, e.g., "Ronaldo plays for Portugal. Who does Neuer play for?". We apply
this approach of querying by example to the LAMA probe and obtain substantial
improvements of up to 37.8% for BERT-large on the T-REx data when providing
only 10 demonstrations--even outperforming a baseline that queries the model
with up to 40 paraphrases of the question. The examples are provided through
the model's context and thus require neither fine-tuning nor an additional
forward pass. This suggests that LMs contain more factual and commonsense
knowledge than previously assumed--if we query the model in the right way.

    

### [[2108.01938] PDE-GCN: Novel Architectures for Graph Neural Networks Motivated by Partial Differential Equations](http://arxiv.org/abs/2108.01938)


  Graph neural networks are increasingly becoming the go-to approach in various
fields such as computer vision, computational biology and chemistry, where data
are naturally explained by graphs. However, unlike traditional convolutional
neural networks, deep graph networks do not necessarily yield better
performance than shallow graph networks. This behavior usually stems from the
over-smoothing phenomenon. In this work, we propose a family of architectures
to control this behavior by design. Our networks are motivated by numerical
methods for solving Partial Differential Equations (PDEs) on manifolds, and as
such, their behavior can be explained by similar analysis. Moreover, as we
demonstrate using an extensive set of experiments, our PDE-motivated networks
can generalize and be effective for various types of problems from different
fields. Our architectures obtain better or on par with the current
state-of-the-art results for problems that are typically approached using
different architectures.

    

### [[2108.01949] Using Interaction Data to Predict Engagement with Interactive Media](http://arxiv.org/abs/2108.01949)


  Media is evolving from traditional linear narratives to personalised
experiences, where control over information (or how it is presented) is given
to individual audience members. Measuring and understanding audience engagement
with this media is important in at least two ways: (1) a post-hoc understanding
of how engaged audiences are with the content will help production teams learn
from experience and improve future productions; (2), this type of media has
potential for real-time measures of engagement to be used to enhance the user
experience by adapting content on-the-fly. Engagement is typically measured by
asking samples of users to self-report, which is time consuming and expensive.
In some domains, however, interaction data have been used to infer engagement.
Fortuitously, the nature of interactive media facilitates a much richer set of
interaction data than traditional media; our research aims to understand if
these data can be used to infer audience engagement. In this paper, we report a
study using data captured from audience interactions with an interactive TV
show to model and predict engagement. We find that temporal metrics, including
overall time spent on the experience and the interval between events, are
predictive of engagement. The results demonstrate that interaction data can be
used to infer users' engagement during and after an experience, and the
proposed techniques are relevant to better understand audience preference and
responses.

    

### [[2108.01952] MRCpy: A Library for Minimax Risk Classifiers](http://arxiv.org/abs/2108.01952)


  Existing libraries for supervised classification implement techniques that
are based on empirical risk minimization and utilize surrogate losses. We
present MRCpy library that implements minimax risk classifiers (MRCs) that are
based on robust risk minimization and can utilize 0-1-loss. Such techniques
give rise to a manifold of classification methods that can provide tight bounds
on the expected loss. MRCpy provides a unified interface for different variants
of MRCs and follows the standards of popular Python libraries. The presented
library also provides implementation for popular techniques that can be seen as
MRCs such as L1-regularized logistic regression, zero-one adversarial, and
maximum entropy machines. In addition, MRCpy implements recent feature mappings
such as Fourier, ReLU, and threshold features. The library is designed with an
object-oriented approach that facilitates collaborators and users.

    

### [[2108.01965] Graph Attention Network For Microwave Imaging of Brain Anomaly](http://arxiv.org/abs/2108.01965)


  So far, numerous learned models have been pressed to use in microwave imaging
problems. These models however, are oblivious to the imaging geometry. It has
always been hard to bake the physical setup of the imaging array into the
structure of the network, resulting in a data-intensive models that are not
practical. This work put forward a graph formulation of the microwave imaging
array. The architectures proposed is made cognizant of the physical setup,
allowing it to incorporate the symmetries, resulting in a less data
requirements. Graph convolution and attention mechanism is deployed to handle
the cases of fully-connected graphs corresponding to multi-static arrays. The
graph-treatment of the problem is evaluated on experimental setup in context of
brain anomaly localization with microwave imaging.

    

### [[2108.01988] Sparse Continuous Distributions and Fenchel-Young Losses](http://arxiv.org/abs/2108.01988)


  Exponential families are widely used in machine learning; they include many
distributions in continuous and discrete domains (e.g., Gaussian, Dirichlet,
Poisson, and categorical distributions via the softmax transformation).
Distributions in each of these families have fixed support. In contrast, for
finite domains, there has been recent works on sparse alternatives to softmax
(e.g. sparsemax, $\alpha$-entmax, and fusedmax) and corresponding losses, which
have varying support.
This paper expands that line of work in several directions: first, it extends
$\Omega$-regularized prediction maps and Fenchel-Young losses to arbitrary
domains (possibly countably infinite or continuous). For linearly parametrized
families, we show that minimization of Fenchel-Young losses is equivalent to
moment matching of the statistics, generalizing a fundamental property of
exponential families. When $\Omega$ is a Tsallis negentropy with parameter
$\alpha$, we obtain "deformed exponential families," which include
$\alpha$-entmax and sparsemax ($\alpha$ = 2) as particular cases. For quadratic
energy functions in continuous domains, the resulting densities are
$\beta$-Gaussians, an instance of elliptical distributions that contain as
particular cases the Gaussian, biweight, triweight and Epanechnikov densities,
and for which we derive closed-form expressions for the variance, Tsallis
entropy, and Fenchel-Young loss. When $\Omega$ is a total variation or Sobolev
regularizer, we obtain a continuous version of the fusedmax. Finally, we
introduce continuous-domain attention mechanisms, deriving efficient gradient
backpropagation algorithms for $\alpha \in \{1, 4/3, 3/2, 2\}$. Using them, we
demonstrate our sparse continuous distributions for attention-based audio
classification and visual question answering, showing that they allow attending
to time intervals and compact regions.

    

### [[2108.01991] Lung Sound Classification Using Co-tuning and Stochastic Normalization](http://arxiv.org/abs/2108.01991)


  In this paper, we use pre-trained ResNet models as backbone architectures for
classification of adventitious lung sounds and respiratory diseases. The
knowledge of the pre-trained model is transferred by using vanilla fine-tuning,
co-tuning, stochastic normalization and the combination of the co-tuning and
stochastic normalization techniques. Furthermore, data augmentation in both
time domain and time-frequency domain is used to account for the class
imbalance of the ICBHI and our multi-channel lung sound dataset. Additionally,
we apply spectrum correction to consider the variations of the recording device
properties on the ICBHI dataset. Empirically, our proposed systems mostly
outperform all state-of-the-art lung sound classification systems for the
adventitious lung sounds and respiratory diseases of both datasets.

    

### [[2108.01994] Staged trees and asymmetry-labeled DAGs](http://arxiv.org/abs/2108.01994)


  Bayesian networks are a widely-used class of probabilistic graphical models
capable of representing symmetric conditional independence between variables of
interest using the topology of the underlying graph. They can be seen as a
special case of the much more general class of models called staged trees,
which can represent any type of non-symmetric conditional independence. Here we
formalize the relationship between these two models and introduce a minimal
Bayesian network representation of the staged tree, which can be used to read
conditional independences in an intuitive way. Furthermore, we define a new
labeled graph, termed asymmetry-labeled directed acyclic graph, whose edges are
labeled to denote the type of dependence existing between any two random
variables. Various datasets are used to illustrate the methodology,
highlighting the need to construct models which more flexibly encode and
represent non-symmetric structures.

    

### [[2108.01995] Robustness of convolutional neural networks to physiological ECG noise](http://arxiv.org/abs/2108.01995)


  The electrocardiogram (ECG) is one of the most widespread diagnostic tools in
healthcare and supports the diagnosis of cardiovascular disorders. Deep
learning methods are a successful and popular technique to detect indications
of disorders from an ECG signal. However, there are open questions around the
robustness of these methods to various factors, including physiological ECG
noise. In this study we generate clean and noisy versions of an ECG dataset
before applying Symmetric Projection Attractor Reconstruction (SPAR) and
scalogram image transformations. A pretrained convolutional neural network is
trained using transfer learning to classify these image transforms. For the
clean ECG dataset, F1 scores for SPAR attractor and scalogram transforms were
0.70 and 0.79, respectively, and the scores decreased by less than 0.05 for the
noisy ECG datasets. Notably, when the network trained on clean data was used to
classify the noisy datasets, performance decreases of up to 0.18 in F1 scores
were seen. However, when the network trained on the noisy data was used to
classify the clean dataset, the performance decrease was less than 0.05. We
conclude that physiological ECG noise impacts classification using deep
learning methods and careful consideration should be given to the inclusion of
noisy ECG signals in the training data when developing supervised networks for
ECG classification.

    

### [[2108.01997] DuCN: Dual-children Network for Medical Diagnosis and Similar Case Recommendation towards COVID-19](http://arxiv.org/abs/2108.01997)


  Early detection of the coronavirus disease 2019 (COVID-19) helps to treat
patients timely and increase the cure rate, thus further suppressing the spread
of the disease. In this study, we propose a novel deep learning based detection
and similar case recommendation network to help control the epidemic. Our
proposed network contains two stages: the first one is a lung region
segmentation step and is used to exclude irrelevant factors, and the second is
a detection and recommendation stage. Under this framework, in the second
stage, we develop a dual-children network (DuCN) based on a pre-trained
ResNet-18 to simultaneously realize the disease diagnosis and similar case
recommendation. Besides, we employ triplet loss and intrapulmonary distance
maps to assist the detection, which helps incorporate tiny differences between
two images and is conducive to improving the diagnostic accuracy. For each
confirmed COVID-19 case, we give similar cases to provide radiologists with
diagnosis and treatment references. We conduct experiments on a large publicly
available dataset (CC-CCII) and compare the proposed model with
state-of-the-art COVID-19 detection methods. The results show that our proposed
model achieves a promising clinical performance.

    

### [[2108.01998] Adversarial Energy Disaggregation for Non-intrusive Load Monitoring](http://arxiv.org/abs/2108.01998)


  Energy disaggregation, also known as non-intrusive load monitoring (NILM),
challenges the problem of separating the whole-home electricity usage into
appliance-specific individual consumptions, which is a typical application of
data analysis. {NILM aims to help households understand how the energy is used
and consequently tell them how to effectively manage the energy, thus allowing
energy efficiency which is considered as one of the twin pillars of sustainable
energy policy (i.e., energy efficiency and renewable energy).} Although NILM is
unidentifiable, it is widely believed that the NILM problem can be addressed by
data science. Most of the existing approaches address the energy disaggregation
problem by conventional techniques such as sparse coding, non-negative matrix
factorization, and hidden Markov model. Recent advances reveal that deep neural
networks (DNNs) can get favorable performance for NILM since DNNs can
inherently learn the discriminative signatures of the different appliances. In
this paper, we propose a novel method named adversarial energy disaggregation
(AED) based on DNNs. We introduce the idea of adversarial learning into NILM,
which is new for the energy disaggregation task. Our method trains a generator
and multiple discriminators via an adversarial fashion. The proposed method not
only learns shard representations for different appliances, but captures the
specific multimode structures of each appliance. Extensive experiments on
real-world datasets verify that our method can achieve new state-of-the-art
performance.

    

### [[2108.02001] Deep Neural Network Approach to Estimate Early Worst-Case Execution Time](http://arxiv.org/abs/2108.02001)


  Estimating Worst-Case Execution Time (WCET) is of utmost importance for
developing Cyber-Physical and Safety-Critical Systems. The system's scheduler
uses the estimated WCET to schedule each task of these systems, and failure may
lead to catastrophic events. It is thus imperative to build provably reliable
systems. WCET is available to us in the last stage of systems development when
the hardware is available and the application code is compiled on it. Different
methodologies measure the WCET, but none of them give early insights on WCET,
which is crucial for system development. If the system designers overestimate
WCET in the early stage, then it would lead to the overqualified system, which
will increase the cost of the final product, and if they underestimate WCET in
the early stage, then it would lead to financial loss as the system would not
perform as expected. This paper estimates early WCET using Deep Neural Networks
as an approximate predictor model for hardware architecture and compiler. This
model predicts the WCET based on the source code without compiling and running
on the hardware architecture. Our WCET prediction model is created using the
Pytorch framework. The resulting WCET is too erroneous to be used as an upper
bound on the WCET. However, getting these results in the early stages of system
development is an essential prerequisite for the system's dimensioning and
configuration of the hardware setup.

    

### [[2108.02005] A purely data-driven framework for prediction, optimization, and control of networked processes: application to networked SIS epidemic model](http://arxiv.org/abs/2108.02005)


  Networks are landmarks of many complex phenomena where interweaving
interactions between different agents transform simple local rule-sets into
nonlinear emergent behaviors. While some recent studies unveil associations
between the network structure and the underlying dynamical process, identifying
stochastic nonlinear dynamical processes continues to be an outstanding
problem. Here we develop a simple data-driven framework based on
operator-theoretic techniques to identify and control stochastic nonlinear
dynamics taking place over large-scale networks. The proposed approach requires
no prior knowledge of the network structure and identifies the underlying
dynamics solely using a collection of two-step snapshots of the states. This
data-driven system identification is achieved by using the Koopman operator to
find a low dimensional representation of the dynamical patterns that evolve
linearly. Further, we use the global linear Koopman model to solve critical
control problems by applying to model predictive control (MPC)--typically, a
challenging proposition when applied to large networks. We show that our
proposed approach tackles this by converting the original nonlinear programming
into a more tractable optimization problem that is both convex and with far
fewer variables.

    

### [[2108.02010] On the Exploitability of Audio Machine Learning Pipelines to Surreptitious Adversarial Examples](http://arxiv.org/abs/2108.02010)


  Machine learning (ML) models are known to be vulnerable to adversarial
examples. Applications of ML to voice biometrics authentication are no
exception. Yet, the implications of audio adversarial examples on these
real-world systems remain poorly understood given that most research targets
limited defenders who can only listen to the audio samples. Conflating
detectability of an attack with human perceptibility, research has focused on
methods that aim to produce imperceptible adversarial examples which humans
cannot distinguish from the corresponding benign samples. We argue that this
perspective is coarse for two reasons: 1. Imperceptibility is impossible to
verify; it would require an experimental process that encompasses variations in
listener training, equipment, volume, ear sensitivity, types of background
noise etc, and 2. It disregards pipeline-based detection clues that realistic
defenders leverage. This results in adversarial examples that are ineffective
in the presence of knowledgeable defenders. Thus, an adversary only needs an
audio sample to be plausible to a human. We thus introduce surreptitious
adversarial examples, a new class of attacks that evades both human and
pipeline controls. In the white-box setting, we instantiate this class with a
joint, multi-stage optimization attack. Using an Amazon Mechanical Turk user
study, we show that this attack produces audio samples that are more
surreptitious than previous attacks that aim solely for imperceptibility.
Lastly we show that surreptitious adversarial examples are challenging to
develop in the black-box setting.

    

### [[2108.02032] Multi-Label Gold Asymmetric Loss Correction with Single-Label Regulators](http://arxiv.org/abs/2108.02032)


  Multi-label learning is an emerging extension of the multi-class
classification where an image contains multiple labels. Not only acquiring a
clean and fully labeled dataset in multi-label learning is extremely expensive,
but also many of the actual labels are corrupted or missing due to the
automated or non-expert annotation techniques. Noisy label data decrease the
prediction performance drastically. In this paper, we propose a novel Gold
Asymmetric Loss Correction with Single-Label Regulators (GALC-SLR) that
operates robust against noisy labels. GALC-SLR estimates the noise confusion
matrix using single-label samples, then constructs an asymmetric loss
correction via estimated confusion matrix to avoid overfitting to the noisy
labels. Empirical results show that our method outperforms the state-of-the-art
original asymmetric loss multi-label classifier under all corruption levels,
showing mean average precision improvement up to 28.67% on a real world dataset
of MS-COCO, yielding a better generalization of the unseen data and increased
prediction performance.

    

### [[2108.02037] The MIT Supercloud Dataset](http://arxiv.org/abs/2108.02037)


  Artificial intelligence (AI) and Machine learning (ML) workloads are an
increasingly larger share of the compute workloads in traditional
High-Performance Computing (HPC) centers and commercial cloud systems. This has
led to changes in deployment approaches of HPC clusters and the commercial
cloud, as well as a new focus on approaches to optimized resource usage,
allocations and deployment of new AI frame- works, and capabilities such as
Jupyter notebooks to enable rapid prototyping and deployment. With these
changes, there is a need to better understand cluster/datacenter operations
with the goal of developing improved scheduling policies, identifying
inefficiencies in resource utilization, energy/power consumption, failure
prediction, and identifying policy violations. In this paper we introduce the
MIT Supercloud Dataset which aims to foster innovative AI/ML approaches to the
analysis of large scale HPC and datacenter/cloud operations. We provide
detailed monitoring logs from the MIT Supercloud system, which include CPU and
GPU usage by jobs, memory usage, file system logs, and physical monitoring
data. This paper discusses the details of the dataset, collection methodology,
data availability, and discusses potential challenge problems being developed
using this data. Datasets and future challenge announcements will be available
via this https URL.

    

### [[2108.02039] Random Convolution Kernels with Multi-Scale Decomposition for Preterm EEG Inter-burst Detection](http://arxiv.org/abs/2108.02039)


  Linear classifiers with random convolution kernels are computationally
efficient methods that need no design or domain knowledge. Unlike deep neural
networks, there is no need to hand-craft a network architecture; the kernels
are randomly generated and only the linear classifier needs training. A
recently proposed method, RandOm Convolutional KErnel Transforms (ROCKETs), has
shown high accuracy across a range of time-series data sets. Here we propose a
multi-scale version of this method, using both high- and low-frequency
components. We apply our methods to inter-burst detection in a cohort of
preterm EEG recorded from 36 neonates <30 weeks gestational age. Two features
from the convolution of 10,000 random kernels are combined using ridge
regression. The proposed multi-scale ROCKET method out-performs the method
without scale: median (interquartile range, IQR) Matthews correlation
coefficient (MCC) of 0.859 (0.815 to 0.874) for multi-scale versus 0.841 (0.807
to 0.865) without scale, p<0.001. The proposed method lags behind an existing
feature-based machine learning method developed with deep domain knowledge, but
is fast to train and can quickly set an initial baseline threshold of
performance for generic and biomedical time-series classification.

    

### [[2108.02040] Convergence of gradient descent for learning linear neural networks](http://arxiv.org/abs/2108.02040)


  We study the convergence properties of gradient descent for training deep
linear neural networks, i.e., deep matrix factorizations, by extending a
previous analysis for the related gradient flow. We show that under suitable
conditions on the step sizes gradient descent converges to a critical point of
the loss function, i.e., the square loss in this article. Furthermore, we
demonstrate that for almost all initializations gradient descent converges to a
global minimum in the case of two layers. In the case of three or more layers
we show that gradient descent converges to a global minimum on the manifold
matrices of some fixed rank, where the rank cannot be determined a priori.

    

### [[2108.02067] Discovering outliers in the Mars Express thermal power consumption patterns](http://arxiv.org/abs/2108.02067)


  The Mars Express (MEX) spacecraft has been orbiting Mars since 2004. The
operators need to constantly monitor its behavior and handle sporadic
deviations (outliers) from the expected patterns of measurements of quantities
that the satellite is sending to Earth. In this paper, we analyze the patterns
of the electrical power consumption of MEX's thermal subsystem, that maintains
the spacecraft's temperature at the desired level. The consumption is not
constant, but should be roughly periodic in the short term, with the period
that corresponds to one orbit around Mars. By using long short-term memory
neural networks, we show that the consumption pattern is more irregular than
expected, and successfully detect such irregularities, opening possibility for
automatic outlier detection on MEX in the future.

    

### [[2108.02076] The Potential of Using Vision Videos for CrowdRE: Video Comments as a Source of Feedback](http://arxiv.org/abs/2108.02076)


  Vision videos are established for soliciting feedback and stimulating
discussions in requirements engineering (RE) practices, such as focus groups.
Different researchers motivated the transfer of these benefits into crowd-based
RE (CrowdRE) by using vision videos on social media platforms. So far, however,
little research explored the potential of using vision videos for CrowdRE in
detail. In this paper, we analyze and assess this potential, in particular,
focusing on video comments as a source of feedback. In a case study, we
analyzed 4505 comments on a vision video from YouTube. We found that the video
solicited 2770 comments from 2660 viewers in four days. This is more than 50%
of all comments the video received in four years. Even though only a certain
fraction of these comments are relevant to RE, the relevant comments address
typical intentions and topics of user feedback, such as feature request or
problem report. Besides the typical user feedback categories, we found more
than 300 comments that address the topic safety, which has not appeared in
previous analyses of user feedback. In an automated analysis, we compared the
performance of three machine learning algorithms on classifying the video
comments. Despite certain differences, the algorithms classified the video
comments well. Based on these findings, we conclude that the use of vision
videos for CrowdRE has a large potential. Despite the preliminary nature of the
case study, we are optimistic that vision videos can motivate stakeholders to
actively participate in a crowd and solicit numerous of video comments as a
valuable source of feedback.

    

### [[2108.02083] Auto-encoder based Model for High-dimensional Imbalanced Industrial Data](http://arxiv.org/abs/2108.02083)


  With the proliferation of IoT devices, the distributed control systems are
now capturing and processing more sensors at higher frequency than ever before.
These new data, due to their volume and novelty, cannot be effectively consumed
without the help of data-driven techniques. Deep learning is emerging as a
promising technique to analyze these data, particularly in soft sensor
modeling. The strong representational capabilities of complex data and the
flexibility it offers from an architectural perspective make it a topic of
active applied research in industrial settings. However, the successful
applications of deep learning in soft sensing are still not widely integrated
in factory control systems, because most of the research on soft sensing do not
have access to large scale industrial data which are varied, noisy and
incomplete. The results published in most research papers are therefore not
easily reproduced when applied to the variety of data in industrial settings.
Here we provide manufacturing data sets that are much larger and more complex
than public open soft sensor data. Moreover, the data sets are from Seagate
factories on active service with only necessary anonymization, so that they
reflect the complex and noisy nature of real-world data. We introduce a
variance weighted multi-headed auto-encoder classification model that fits well
into the high-dimensional and highly imbalanced data. Besides the use of
weighting or sampling methods to handle the highly imbalanced data, the model
also simultaneously predicts multiple outputs by exploiting output-supervised
representation learning and multi-task weighting.

    

### [[2108.02096] Policy Gradients Incorporating the Future](http://arxiv.org/abs/2108.02096)


  Reasoning about the future -- understanding how decisions in the present time
affect outcomes in the future -- is one of the central challenges for
reinforcement learning (RL), especially in highly-stochastic or partially
observable environments. While predicting the future directly is hard, in this
work we introduce a method that allows an agent to "look into the future"
without explicitly predicting it. Namely, we propose to allow an agent, during
its training on past experience, to observe what \emph{actually} happened in
the future at that time, while enforcing an information bottleneck to avoid the
agent overly relying on this privileged information. This gives our agent the
opportunity to utilize rich and useful information about the future trajectory
dynamics in addition to the present. Our method, Policy Gradients Incorporating
the Future (PGIF), is easy to implement and versatile, being applicable to
virtually any policy gradient algorithm. We apply our proposed method to a
number of off-the-shelf RL algorithms and show that PGIF is able to achieve
higher reward faster in a variety of online and offline RL domains, as well as
sparse-reward and partially observable environments.

    

### [[2108.02113] Hyperparameter-free and Explainable Whole Graph Embedding](http://arxiv.org/abs/2108.02113)


  Many real-world complex systems can be described as graphs. For a large-scale
graph with low sparsity, a node's adjacency vector is a long and sparse
representation, limiting the practical utilization of existing machine learning
methods on nodal features. In practice, graph embedding (graph representation
learning) attempts to learn a lower-dimensional representation vector for each
node or the whole graph while maintaining the most basic information of graph.
Since various machine learning methods can efficiently process
lower-dimensional vectors, graph embedding has recently attracted a lot of
attention. However, most node embedding or whole graph embedding methods suffer
from the problem of having more sophisticated methodology, hyperparameter
optimization, and low explainability. This paper proposes a
hyperparameter-free, extensible, and explainable whole graph embedding method,
combining the DHC (Degree, H-index and Coreness) theorem and Shannon Entropy
(E), abbreviated as DHC-E. The new whole graph embedding scheme can obtain a
trade-off between the simplicity and the quality under some supervised
classification learning tasks, using molecular, social, and brain networks. In
addition, the proposed approach has a good performance in lower-dimensional
graph visualization. The new methodology is overall simple,
hyperparameter-free, extensible, and explainable for whole graph embedding with
promising potential for exploring graph classification, prediction, and
lower-dimensional graph visualization.

    

### [[2108.02117] FedJAX: Federated learning simulation with JAX](http://arxiv.org/abs/2108.02117)


  Federated learning is a machine learning technique that enables training
across decentralized data. Recently, federated learning has become an active
area of research due to the increased concerns over privacy and security. In
light of this, a variety of open source federated learning libraries have been
developed and released. We introduce FedJAX, a JAX-based open source library
for federated learning simulations that emphasizes ease-of-use in research.
With its simple primitives for implementing federated learning algorithms,
prepackaged datasets, models and algorithms, and fast simulation speed, FedJAX
aims to make developing and evaluating federated algorithms faster and easier
for researchers. Our benchmark results show that FedJAX can be used to train
models with federated averaging on the EMNIST dataset in a few minutes and the
Stack Overflow dataset in roughly an hour with standard hyperparmeters using
TPUs.

    

### [[2108.02120] Statistical Analysis of Wasserstein Distributionally Robust Estimators](http://arxiv.org/abs/2108.02120)


  We consider statistical methods which invoke a min-max distributionally
robust formulation to extract good out-of-sample performance in data-driven
optimization and learning problems. Acknowledging the distributional
uncertainty in learning from limited samples, the min-max formulations
introduce an adversarial inner player to explore unseen covariate data. The
resulting Distributionally Robust Optimization (DRO) formulations, which
include Wasserstein DRO formulations (our main focus), are specified using
optimal transportation phenomena. Upon describing how these
infinite-dimensional min-max problems can be approached via a
finite-dimensional dual reformulation, the tutorial moves into its main
component, namely, explaining a generic recipe for optimally selecting the size
of the adversary's budget. This is achieved by studying the limit behavior of
an optimal transport projection formulation arising from an inquiry on the
smallest confidence region that includes the unknown population risk minimizer.
Incidentally, this systematic prescription coincides with those in specific
examples in high-dimensional statistics and results in error bounds that are
free from the curse of dimensions. Equipped with this prescription, we present
a central limit theorem for the DRO estimator and provide a recipe for
constructing compatible confidence regions that are useful for uncertainty
quantification. The rest of the tutorial is devoted to insights into the nature
of the optimizers selected by the min-max formulations and additional
applications of optimal transport projections.

    

### [[2108.02122] Semi-weakly Supervised Contrastive Representation Learning for Retinal Fundus Images](http://arxiv.org/abs/2108.02122)


  We explore the value of weak labels in learning transferable representations
for medical images. Compared to hand-labeled datasets, weak or inexact labels
can be acquired in large quantities at significantly lower cost and can provide
useful training signals for data-hungry models such as deep neural networks. We
consider weak labels in the form of pseudo-labels and propose a semi-weakly
supervised contrastive learning (SWCL) framework for representation learning
using semi-weakly annotated images. Specifically, we train a semi-supervised
model to propagate labels from a small dataset consisting of diverse
image-level annotations to a large unlabeled dataset. Using the propagated
labels, we generate a patch-level dataset for pretraining and formulate a
multi-label contrastive learning objective to capture position-specific
features encoded in each patch. We empirically validate the transfer learning
performance of SWCL on seven public retinal fundus datasets, covering three
disease classification tasks and two anatomical structure segmentation tasks.
Our experiment results suggest that, under very low data regime, large-scale
ImageNet pretraining on improved architecture remains a very strong baseline,
and recently proposed self-supervised methods falter in segmentation tasks,
possibly due to the strong invariant constraint imposed. Our method surpasses
all prior self-supervised methods and standard cross-entropy training, while
closing the gaps with ImageNet pretraining.

    

### [[2108.02128] Parallelized Reverse Curriculum Generation](http://arxiv.org/abs/2108.02128)


  For reinforcement learning (RL), it is challenging for an agent to master a
task that requires a specific series of actions due to sparse rewards. To solve
this problem, reverse curriculum generation (RCG) provides a reverse expansion
approach that automatically generates a curriculum for the agent to learn. More
specifically, RCG adapts the initial state distribution from the neighborhood
of a goal to a distance as training proceeds. However, the initial state
distribution generated for each iteration might be biased, thus making the
policy overfit or slowing down the reverse expansion rate. While training RCG
for actor-critic (AC) based RL algorithms, this poor generalization and slow
convergence might be induced by the tight coupling between an AC pair.
Therefore, we propose a parallelized approach that simultaneously trains
multiple AC pairs and periodically exchanges their critics. We empirically
demonstrate that this proposed approach can improve RCG in performance and
convergence, and it can also be applied to other AC based RL algorithms with
adapted initial state distribution.

    

### [[2108.02137] Under the Radar -- Auditing Fairness in ML for Humanitarian Mapping](http://arxiv.org/abs/2108.02137)


  Humanitarian mapping from space with machine learning helps policy-makers to
timely and accurately identify people in need. However, recent concerns around
fairness and transparency of algorithmic decision-making are a significant
obstacle for applying these methods in practice. In this paper, we study if
humanitarian mapping approaches from space are prone to bias in their
predictions. We map village-level poverty and electricity rates in India based
on nighttime lights (NTLs) with linear regression and random forest and analyze
if the predictions systematically show prejudice against scheduled caste or
tribe communities. To achieve this, we design a causal approach to measure
counterfactual fairness based on propensity score matching. This allows to
compare villages within a community of interest to synthetic counterfactuals.
Our findings indicate that poverty is systematically overestimated and
electricity systematically underestimated for scheduled tribes in comparison to
a synthetic counterfactual group of villages. The effects have the opposite
direction for scheduled castes where poverty is underestimated and
electrification overestimated. These results are a warning sign for a variety
of applications in humanitarian mapping where fairness issues would compromise
policy goals.

    

### [[2108.02148] Pervasive Hand Gesture Recognition for Smartphones using Non-audible Sound and Deep Learning](http://arxiv.org/abs/2108.02148)


  Due to the mass advancement in ubiquitous technologies nowadays, new
pervasive methods have come into the practice to provide new innovative
features and stimulate the research on new human-computer interactions. This
paper presents a hand gesture recognition method that utilizes the smartphone's
built-in speakers and microphones. The proposed system emits an ultrasonic
sonar-based signal (inaudible sound) from the smartphone's stereo speakers,
which is then received by the smartphone's microphone and processed via a
Convolutional Neural Network (CNN) for Hand Gesture Recognition. Data
augmentation techniques are proposed to improve the detection accuracy and
three dual-channel input fusion methods are compared. The first method merges
the dual-channel audio as a single input spectrogram image. The second method
adopts early fusion by concatenating the dual-channel spectrograms. The third
method adopts late fusion by having two convectional input branches processing
each of the dual-channel spectrograms and then the outputs are merged by the
last layers. Our experimental results demonstrate a promising detection
accuracy for the six gestures presented in our publicly available dataset with
an accuracy of 93.58\% as a baseline.

    

### [[2108.02155] Improving Aleatoric Uncertainty Quantification in Multi-Annotated Medical ImageSegmentation with Normalizing Flows](http://arxiv.org/abs/2108.02155)


  Quantifying uncertainty in medical image segmentation applications is
essential, as it is often connected to vital decision-making. Compelling
attempts have been made in quantifying the uncertainty in image segmentation
architectures, e.g. to learn a density segmentation model conditioned on the
input image. Typical work in this field restricts these learnt densities to be
strictly Gaussian. In this paper, we propose to use a more flexible approach by
introducing Normalizing Flows (NFs), which enables the learnt densities to be
more complex and facilitate more accurate modeling for uncertainty. We prove
this hypothesis by adopting the Probabilistic U-Net and augmenting the
posterior density with an NF, allowing it to be more expressive. Our
qualitative as well as quantitative (GED and IoU) evaluations on the
multi-annotated and single-annotated LIDC-IDRI and Kvasir-SEG segmentation
datasets, respectively, show a clear improvement. This is mostly apparent in
the quantification of aleatoric uncertainty and the increased predictive
performance of up to 14 percent. This result strongly indicates that a more
flexible density model should be seriously considered in architectures that
attempt to capture segmentation ambiguity through density modeling. The benefit
of this improved modeling will increase human confidence in annotation and
segmentation, and enable eager adoption of the technology in practice.

    

### [[2108.02191] Random Offset Block Embedding Array (ROBE) for CriteoTB Benchmark MLPerf DLRM Model : 1000$\times$ Compression and 2.7$\times$ Faster Inference](http://arxiv.org/abs/2108.02191)


  Deep learning for recommendation data is the one of the most pervasive and
challenging AI workload in recent times. State-of-the-art recommendation models
are one of the largest models rivalling the likes of GPT-3 and Switch
Transformer. Challenges in deep learning recommendation models (DLRM) stem from
learning dense embeddings for each of the categorical values. These embedding
tables in industrial scale models can be as large as hundreds of terabytes.
Such large models lead to a plethora of engineering challenges, not to mention
prohibitive communication overheads, and slower training and inference times.
Of these, slower inference time directly impacts user experience. Model
compression for DLRM is gaining traction and the community has recently shown
impressive compression results. In this paper, we present Random Offset Block
Embedding Array (ROBE) as a low memory alternative to embedding tables which
provide orders of magnitude reduction in memory usage while maintaining
accuracy and boosting execution speed. ROBE is a simple fundamental approach in
improving both cache performance and the variance of randomized hashing, which
could be of independent interest in itself. We demonstrate that we can
successfully train DLRM models with same accuracy while using $1000 \times$
less memory. A $1000\times$ compressed model directly results in faster
inference without any engineering. In particular, we show that we can train
DLRM model using ROBE Array of size 100MB on a single GPU to achieve AUC of
0.8025 or higher as required by official MLPerf CriteoTB benchmark DLRM model
of 100GB while achieving about $2.7\times$ (170\%) improvement in inference
throughput.

    

### [[2108.02200] Spacetime Neural Network for High Dimensional Quantum Dynamics](http://arxiv.org/abs/2108.02200)


  We develop a spacetime neural network method with second order optimization
for solving quantum dynamics from the high dimensional Schrödinger
equation. In contrast to the standard iterative first order optimization and
the time-dependent variational principle, our approach utilizes the implicit
mid-point method and generates the solution for all spatial and temporal values
simultaneously after optimization. We demonstrate the method in the
Schrödinger equation with a self-normalized autoregressive spacetime neural
network construction. Future explorations for solving different high
dimensional differential equations are discussed.

    

### [[1809.08771] Modeling longitudinal data using matrix completion](http://arxiv.org/abs/1809.08771)


  In clinical practice and biomedical research, measurements are often
collected sparsely and irregularly in time while the data acquisition is
expensive and inconvenient. Examples include measurements of spine bone mineral
density, cancer growth through mammography or biopsy, a progression of
defective vision, or assessment of gait in patients with neurological
disorders. Since the data collection is often costly and inconvenient,
estimation of progression from sparse observations is of great interest for
practitioners.
From the statistical standpoint, such data is often analyzed in the context
of a mixed-effect model where time is treated as both a fixed-effect
(population progression curve) and a random-effect (individual variability).
Alternatively, researchers analyze Gaussian processes or functional data where
observations are assumed to be drawn from a certain distribution of processes.
These models are flexible but rely on probabilistic assumptions, require very
careful implementation, specific to the given problem, and tend to be slow in
practice.
In this study, we propose an alternative elementary framework for analyzing
longitudinal data, relying on matrix completion. Our method yields estimates of
progression curves by iterative application of the Singular Value
Decomposition. Our framework covers multivariate longitudinal data, regression,
and can be easily extended to other settings. As it relies on existing tools
for matrix algebra it is efficient and easy to implement.
We apply our methods to understand trends of progression of motor impairment
in children with Cerebral Palsy. Our model approximates individual progression
curves and explains 30% of the variability. Low-rank representation of
progression trends enables identification of different progression trends in
subtypes of Cerebral Palsy.

    

### [[1810.01248] A Lightweight Music Texture Transfer System](http://arxiv.org/abs/1810.01248)


  Deep learning researches on the transformation problems for image and text
have raised great attention. However, present methods for music feature
transfer using neural networks are far from practical application. In this
paper, we initiate a novel system for transferring the texture of music, and
release it as an open source project. Its core algorithm is composed of a
converter which represents sounds as texture spectra, a corresponding
reconstructor and a feed-forward transfer network. We evaluate this system from
multiple perspectives, and experimental results reveal that it achieves
convincing results in both sound effects and computational performance.

    

### [[2002.06212] Ensemble Slice Sampling: Parallel, black-box and gradient-free inference for correlated & multimodal distributions](http://arxiv.org/abs/2002.06212)


  Slice Sampling has emerged as a powerful Markov Chain Monte Carlo algorithm
that adapts to the characteristics of the target distribution with minimal
hand-tuning. However, Slice Sampling's performance is highly sensitive to the
user-specified initial length scale hyperparameter and the method generally
struggles with poorly scaled or strongly correlated distributions. This paper
introduces Ensemble Slice Sampling (ESS), a new class of algorithms that
bypasses such difficulties by adaptively tuning the initial length scale and
utilising an ensemble of parallel walkers in order to efficiently handle strong
correlations between parameters. These affine-invariant algorithms are trivial
to construct, require no hand-tuning, and can easily be implemented in parallel
computing environments. Empirical tests show that Ensemble Slice Sampling can
improve efficiency by more than an order of magnitude compared to conventional
MCMC methods on a broad range of highly correlated target distributions. In
cases of strongly multimodal target distributions, Ensemble Slice Sampling can
sample efficiently even in high dimensions. We argue that the parallel,
black-box and gradient-free nature of the method renders it ideal for use in
scientific fields such as physics, astrophysics and cosmology which are
dominated by a wide variety of computationally expensive and non-differentiable
models.

    

### [[2002.07088] Robust Physical Hard-Label Attacks on Deep Learning Visual Classification](http://arxiv.org/abs/2002.07088)


  The physical, black-box hard-label setting is arguably the most realistic
threat model for cyber-physical vision systems. In this setting, the attacker
only has query access to the model and only receives the top-1 class label
without confidence information. Creating small physical stickers that are
robust to environmental variation is difficult in the discrete and
discontinuous hard-label space because the attack must both design a small
shape to perturb within and find robust noise to fill it with. Unfortunately,
we find that existing $\ell_2$ or $\ell_\infty$ minimizing hard-label attacks
do not easily extend to finding such robust physical perturbation attacks.
Thus, we propose GRAPHITE, the first algorithm for hard-label physical attacks
on computer vision models. We show that "survivability", an estimate of
physical variation robustness, can be used in new ways to generate small masks
and is a sufficiently smooth function to optimize with gradient-free
optimization. We use GRAPHITE to attack a traffic sign classifier and a
publicly-available Automatic License Plate Recognition (ALPR) tool using only
query access. We evaluate both tools in real-world field tests to measure its
physical-world robustness. We successfully cause a Stop sign to be
misclassified as a Speed Limit 30 km/hr sign in 95.7% of physical images and
cause errors in 75% of physical images for the ALPR tool.

    

### [[2006.14091] Learning Reward Functions from Diverse Sources of Human Feedback: Optimally Integrating Demonstrations and Preferences](http://arxiv.org/abs/2006.14091)


  Reward functions are a common way to specify the objective of a robot. As
designing reward functions can be extremely challenging, a more promising
approach is to directly learn reward functions from human teachers.
Importantly, data from human teachers can be collected either passively or
actively in a variety of forms: passive data sources include demonstrations,
(e.g., kinesthetic guidance), whereas preferences (e.g., comparative rankings)
are actively elicited. Prior research has independently applied reward learning
to these different data sources. However, there exist many domains where
multiple sources are complementary and expressive. Motivated by this general
problem, we present a framework to integrate multiple sources of information,
which are either passively or actively collected from human users. In
particular, we present an algorithm that first utilizes user demonstrations to
initialize a belief about the reward function, and then actively probes the
user with preference queries to zero-in on their true reward. This algorithm
not only enables us combine multiple data sources, but it also informs the
robot when it should leverage each type of information. Further, our approach
accounts for the human's ability to provide data: yielding user-friendly
preference queries which are also theoretically optimal. Our extensive
simulated experiments and user studies on a Fetch mobile manipulator
demonstrate the superiority and the usability of our integrated framework.

    

### [[2007.06028] TERA: Self-Supervised Learning of Transformer Encoder Representation for Speech](http://arxiv.org/abs/2007.06028)


  We introduce a self-supervised speech pre-training method called TERA, which
stands for Transformer Encoder Representations from Alteration. Recent
approaches often learn by using a single auxiliary task like contrastive
prediction, autoregressive prediction, or masked reconstruction. Unlike
previous methods, we use alteration along three orthogonal axes to pre-train
Transformer Encoders on a large amount of unlabeled speech. The model learns
through the reconstruction of acoustic frames from their altered counterpart,
where we use a stochastic policy to alter along various dimensions: time,
frequency, and magnitude. TERA can be used for speech representations
extraction or fine-tuning with downstream models. We evaluate TERA on several
downstream tasks, including phoneme classification, keyword spotting, speaker
recognition, and speech recognition. We present a large-scale comparison of
various self-supervised models. TERA achieves strong performance in the
comparison by improving upon surface features and outperforming previous
models. In our experiments, we study the effect of applying different
alteration techniques, pre-training on more data, and pre-training on various
features. We analyze different model sizes and find that smaller models are
strong representation learners than larger models, while larger models are more
effective for downstream fine-tuning than smaller models. Furthermore, we show
the proposed method is transferable to downstream datasets not used in
pre-training.

    

### [[2008.05228] Optimal to-do list gamification](http://arxiv.org/abs/2008.05228)


  What should I work on first? What can wait until later? Which projects should
I prioritize and which tasks are not worth my time? These are challenging
questions that many people face every day. People's intuitive strategy is to
prioritize their immediate experience over the long-term consequences. This
leads to procrastination and the neglect of important long-term projects in
favor of seemingly urgent tasks that are less important. Optimal gamification
strives to help people overcome these problems by incentivizing each task by a
number of points that communicates how valuable it is in the long-run.
Unfortunately, computing the optimal number of points with standard dynamic
programming methods quickly becomes intractable as the number of a person's
projects and the number of tasks required by each project increase. Here, we
introduce and evaluate a scalable method for identifying which tasks are most
important in the long run and incentivizing each task according to its
long-term value. Our method makes it possible to create to-do list gamification
apps that can handle the size and complexity of people's to-do lists in the
real world.

    

### [[2008.06822] Relevance Attack on Detectors](http://arxiv.org/abs/2008.06822)


  This paper focuses on high-transferable adversarial attacks on detectors,
which are hard to attack in a black-box manner, because of their
multiple-output characteristics and the diversity across architectures. To
pursue a high attack transferability, one plausible way is to find a common
property across detectors, which facilitates the discovery of common
weaknesses. We are the first to suggest that the relevance map from
interpreters for detectors is such a property. Based on it, we design a
Relevance Attack on Detectors (RAD), which achieves a state-of-the-art
transferability, exceeding existing results by above 20%. On MS COCO, the
detection mAPs for all 8 black-box architectures are more than halved and the
segmentation mAPs are also significantly influenced. Given the great
transferability of RAD, we generate the first adversarial dataset for object
detection and instance segmentation, i.e., Adversarial Objects in COntext
(AOCO), which helps to quickly evaluate and improve the robustness of
detectors.

    

### [[2009.08978] Modeling Online Behavior in Recommender Systems: The Importance of Temporal Context](http://arxiv.org/abs/2009.08978)


  Recommender systems research tends to evaluate model performance offline and
on randomly sampled targets, yet the same systems are later used to predict
user behavior sequentially from a fixed point in time. Simulating online
recommender system performance is notoriously difficult and the discrepancy
between online and offline behaviors is typically not accounted for in offline
evaluations. This disparity permits weaknesses to go unnoticed until the model
is deployed in a production setting. In this paper, we first demonstrate how
omitting temporal context when evaluating recommender system performance leads
to false confidence. To overcome this, we postulate that offline evaluation
protocols can only model real-life use-cases if they account for temporal
context. Next, we propose a training procedure to further embed the temporal
context in existing models: we introduce it in a multi-objective approach to
traditionally time-unaware recommender systems and confirm its advantage via
the proposed evaluation protocol. Finally, we validate that the Pareto Fronts
obtained with the added objective dominate those produced by state-of-the-art
models that are only optimized for accuracy on three real-world publicly
available datasets. The results show that including our temporal objective can
improve recall@20 by up to 20%.

    

### [[2010.02469] Generalized Matrix Factorization](http://arxiv.org/abs/2010.02469)


  Unmeasured or latent variables are often the cause of correlations between
multivariate measurements and are studied in a variety of fields such as
psychology, ecology, and medicine. For Gaussian measurements, there are
classical tools such as factor analysis or principal component analysis with a
well-established theory and fast algorithms. Generalized Linear Latent Variable
models (GLLVM) generalize such factor models to non-Gaussian responses.
However, current algorithms for estimating model parameters in GLLVMs require
intensive computation and do not scale to large datasets with thousands of
observational units or responses. In this article, we propose a new approach
for fitting GLLVMs to such high-volume, high-dimensional datasets. We
approximate the likelihood using penalized quasi-likelihood and use a Newton
method and Fisher scoring to learn the model parameters. Our method greatly
reduces the computation time and can be easily parallelized, enabling
factorization at unprecedented scale using commodity hardware. We illustrate
application of our method on a dataset of 48,000 observational units with over
2,000 observed species in each unit, finding that most of the variability can
be explained with a handful of factors.

    

### [[2010.04331] Targeted Attention Attack on Deep Learning Models in Road Sign Recognition](http://arxiv.org/abs/2010.04331)


  Real world traffic sign recognition is an important step towards building
autonomous vehicles, most of which highly dependent on Deep Neural Networks
(DNNs). Recent studies demonstrated that DNNs are surprisingly susceptible to
adversarial examples. Many attack methods have been proposed to understand and
generate adversarial examples, such as gradient based attack, score based
attack, decision based attack, and transfer based attacks. However, most of
these algorithms are ineffective in real-world road sign attack, because (1)
iteratively learning perturbations for each frame is not realistic for a fast
moving car and (2) most optimization algorithms traverse all pixels equally
without considering their diverse contribution. To alleviate these problems,
this paper proposes the targeted attention attack (TAA) method for real world
road sign attack. Specifically, we have made the following contributions: (1)
we leverage the soft attention map to highlight those important pixels and skip
those zero-contributed areas - this also helps to generate natural
perturbations, (2) we design an efficient universal attack that optimizes a
single perturbation/noise based on a set of training images under the guidance
of the pre-trained attention map, (3) we design a simple objective function
that can be easily optimized, (4) we evaluate the effectiveness of TAA on real
world data sets. Experimental results validate that the TAA method improves the
attack successful rate (nearly 10%) and reduces the perturbation loss (about a
quarter) compared with the popular RP2 method. Additionally, our TAA also
provides good properties, e.g., transferability and generalization capability.
We provide code and data to ensure the reproducibility:
this https URL.

    

### [[2010.05150] Safe Reinforcement Learning with Natural Language Constraints](http://arxiv.org/abs/2010.05150)


  While safe reinforcement learning (RL) holds great promise for many practical
applications like robotics or autonomous cars, current approaches require
specifying constraints in mathematical form. Such specifications demand domain
expertise, limiting the adoption of safe RL. In this paper, we propose learning
to interpret natural language constraints for safe RL. To this end, we first
introduce HazardWorld, a new multi-task benchmark that requires an agent to
optimize reward while not violating constraints specified in free-form text. We
then develop an agent with a modular architecture that can interpret and adhere
to such textual constraints while learning new tasks. Our model consists of (1)
a constraint interpreter that encodes textual constraints into spatial and
temporal representations of forbidden states, and (2) a policy network that
uses these representations to produce a policy achieving minimal constraint
violations during training. Across different domains in HazardWorld, we show
that our method achieves higher rewards (up to11x) and fewer constraint
violations (by 1.8x) compared to existing approaches. However, in terms of
absolute performance, HazardWorld still poses significant challenges for agents
to learn efficiently, motivating the need for future work.

    

### [[2010.14265] A Weaker Faithfulness Assumption based on Triple Interactions](http://arxiv.org/abs/2010.14265)


  One of the core assumptions in causal discovery is the faithfulness
assumption, i.e., assuming that independencies found in the data are due to
separations in the true causal graph. This assumption can, however, be violated
in many ways, including xor connections, deterministic functions or cancelling
paths. In this work, we propose a weaker assumption that we call $2$-adjacency
faithfulness. In contrast to adjacency faithfulness, which assumes that there
is no conditional independence between each pair of variables that are
connected in the causal graph, we only require no conditional independence
between a node and a subset of its Markov blanket that can contain up to two
nodes. Equivalently, we adapt orientation faithfulness to this setting. We
further propose a sound orientation rule for causal discovery that applies
under weaker assumptions. As a proof of concept, we derive a modified Grow and
Shrink algorithm that recovers the Markov blanket of a target node and prove
its correctness under strictly weaker assumptions than the standard
faithfulness assumption.

    

### [[2011.04457] Binary Matrix Factorisation via Column Generation](http://arxiv.org/abs/2011.04457)


  Identifying discrete patterns in binary data is an important dimensionality
reduction tool in machine learning and data mining. In this paper, we consider
the problem of low-rank binary matrix factorisation (BMF) under Boolean
arithmetic. Due to the hardness of this problem, most previous attempts rely on
heuristic techniques. We formulate the problem as a mixed integer linear
program and use a large scale optimisation technique of column generation to
solve it without the need of heuristic pattern mining. Our approach focuses on
accuracy and on the provision of optimality guarantees. Experimental results on
real world datasets demonstrate that our proposed method is effective at
producing highly accurate factorisations and improves on the previously
available best known results for 15 out of 24 problem instances.

    

### [[2012.01699] Essential Features: Content-Adaptive Pixel Discretization to Improve Model Robustness to Adaptive Adversarial Attacks](http://arxiv.org/abs/2012.01699)


  To remove the effects of adversarial perturbations, preprocessing defenses
such as pixel discretization are appealing due to their simplicity but have so
far been shown to be ineffective except on simple datasets such as MNIST,
leading to the belief that pixel discretization approaches are doomed to
failure as a defense technique. This paper revisits the pixel discretization
approaches. We hypothesize that the reason why existing approaches have failed
is that they have used a fixed codebook for the entire dataset. In particular,
we find that can lead to situations where images become more susceptible to
adversarial perturbations and also suffer significant loss of accuracy after
discretization. We propose a novel image preprocessing technique called
Essential Features that uses an adaptive codebook that is based on per-image
content and threat model. Essential Features adaptively selects a separable set
of color clusters for each image to reduce the color space while preserving the
pertinent features of the original image, maximizing both separability and
representation of colors. Additionally, to limit the adversary's ability to
influence the chosen color clusters, Essential Features takes advantage of
spatial correlation with an adaptive blur that moves pixels closer to their
original value without destroying original edge information. We design several
adaptive attacks and find that our approach is more robust than previous
baselines on $L_\infty$ and $L_2$ bounded attacks for several challenging
datasets including CIFAR-10, GTSRB, RESISC45, and ImageNet.

    

### [[2012.09962] Making Contrastive Learning Robust to Shortcuts](http://arxiv.org/abs/2012.09962)


  Contrastive learning is effective at learning useful representations without
supervision. Yet contrastive learning is susceptible to shortcuts -- i.e., it
may learn shortcut features irrelevant to the downstream task and discard
relevant information. Past work has addressed this limitation via handcrafted
data augmentations that eliminate the shortcut. However, handcrafted
augmentations are infeasible for data modalities that are not interpretable by
humans (e.g., radio signals). Further, even when the modality is interpretable
(e.g., RGB), sometimes eliminating the shortcut information may be undesirable.
For example, in multi-attribute classification, information related to one
attribute may act as a shortcut around other attributes. This paper presents
reconstructive contrastive learning (RCL), a framework for learning
unsupervised representations that are robust to shortcuts. The key idea is to
force the learned representation to reconstruct the input, which naturally
counters potential shortcuts. Extensive experiments verify that RCL is highly
robust to shortcuts and outperforms state-of-the-art contrastive learning
methods on both RGB and RF datasets for a variety of tasks.

    

### [[2012.10582] Learning by Fixing: Solving Math Word Problems with Weak Supervision](http://arxiv.org/abs/2012.10582)


  Previous neural solvers of math word problems (MWPs) are learned with full
supervision and fail to generate diverse solutions. In this paper, we address
this issue by introducing a \textit{weakly-supervised} paradigm for learning
MWPs. Our method only requires the annotations of the final answers and can
generate various solutions for a single problem. To boost weakly-supervised
learning, we propose a novel \textit{learning-by-fixing} (LBF) framework, which
corrects the misperceptions of the neural network via symbolic reasoning.
Specifically, for an incorrect solution tree generated by the neural network,
the \textit{fixing} mechanism propagates the error from the root node to the
leaf nodes and infers the most probable fix that can be executed to get the
desired answer. To generate more diverse solutions, \textit{tree
regularization} is applied to guide the efficient shrinkage and exploration of
the solution space, and a \textit{memory buffer} is designed to track and save
the discovered various fixes for each problem. Experimental results on the
Math23K dataset show the proposed LBF framework significantly outperforms
reinforcement learning baselines in weakly-supervised learning. Furthermore, it
achieves comparable top-1 and much better top-3/5 answer accuracies than
fully-supervised methods, demonstrating its strength in producing diverse
solutions.

    

### [[2012.11905] GANterfactual -- Counterfactual Explanations for Medical Non-Experts using Generative Adversarial Learning](http://arxiv.org/abs/2012.11905)


  With the ongoing rise of machine learning, the need for methods for
explaining decisions made by artificial intelligence systems is becoming a more
and more important topic. Especially for image classification tasks, many
state-of-the-art tools to explain such classifiers rely on visual highlighting
of important areas of the input data. Contrary, counterfactual explanation
systems try to enable a counterfactual reasoning by modifying the input image
in a way such that the classifier would have made a different prediction. By
doing so, the users of counterfactual explanation systems are equipped with a
completely different kind of explanatory information. However, methods for
generating realistic counterfactual explanations for image classifiers are
still rare. Especially in medical contexts, where relevant information often
consists of textural and structural information, high-quality counterfactual
images have the potential to give meaningful insights into decision processes.
In this work, we present GANterfactual, an approach to generate such
counterfactual image explanations based on adversarial image-to-image
translation techniques. Additionally, we conduct a user study to evaluate our
approach in an exemplary medical use case. Our results show that, in the chosen
medical use-case, counterfactual explanations lead to significantly better
results regarding mental models, explanation satisfaction, trust, emotions, and
self-efficacy than two state-of-the-art systems that work with saliency maps,
namely LIME and LRP.

    

### [[2012.15715] Beyond Offline Mapping: Learning Cross Lingual Word Embeddings through Context Anchoring](http://arxiv.org/abs/2012.15715)


  Recent research on cross-lingual word embeddings has been dominated by
unsupervised mapping approaches that align monolingual embeddings. Such methods
critically rely on those embeddings having a similar structure, but it was
recently shown that the separate training in different languages causes
departures from this assumption. In this paper, we propose an alternative
approach that does not have this limitation, while requiring a weak seed
dictionary (e.g., a list of identical words) as the only form of supervision.
Rather than aligning two fixed embedding spaces, our method works by fixing the
target language embeddings, and learning a new set of embeddings for the source
language that are aligned with them. To that end, we use an extension of
skip-gram that leverages translated context words as anchor points, and
incorporates self-learning and iterative restarts to reduce the dependency on
the initial dictionary. Our approach outperforms conventional mapping methods
on bilingual lexicon induction, and obtains competitive results in the
downstream XNLI task.

    

### [[2103.04436] Insta-RS: Instance-wise Randomized Smoothing for Improved Robustness and Accuracy](http://arxiv.org/abs/2103.04436)


  Randomized smoothing (RS) is an effective and scalable technique for
constructing neural network classifiers that are certifiably robust to
adversarial perturbations. Most RS works focus on training a good base model
that boosts the certified robustness of the smoothed model. However, existing
RS techniques treat every data point the same, i.e., the variance of the
Gaussian noise used to form the smoothed model is preset and universal for all
training and test data. This preset and universal Gaussian noise variance is
suboptimal since different data points have different margins and the local
properties of the base model vary across the input examples. In this paper, we
examine the impact of customized handling of examples and propose Instance-wise
Randomized Smoothing (Insta-RS) -- a multiple-start search algorithm that
assigns customized Gaussian variances to test examples. We also design Insta-RS
Train -- a novel two-stage training algorithm that adaptively adjusts and
customizes the noise level of each training example for training a base model
that boosts the certified robustness of the instance-wise Gaussian smoothed
model. Through extensive experiments on CIFAR-10 and ImageNet, we show that our
method significantly enhances the average certified radius (ACR) as well as the
clean data accuracy compared to existing state-of-the-art provably robust
classifiers.

    

### [[2104.00411] Explaining COVID-19 and Thoracic Pathology Model Predictions by Identifying Informative Input Features](http://arxiv.org/abs/2104.00411)


  Neural networks have demonstrated remarkable performance in classification
and regression tasks on chest X-rays. In order to establish trust in the
clinical routine, the networks' prediction mechanism needs to be interpretable.
One principal approach to interpretation is feature attribution. Feature
attribution methods identify the importance of input features for the output
prediction. Building on Information Bottleneck Attribution (IBA) method, for
each prediction we identify the chest X-ray regions that have high mutual
information with the network's output. Original IBA identifies input regions
that have sufficient predictive information. We propose Inverse IBA to identify
all informative regions. Thus all predictive cues for pathologies are
highlighted on the X-rays, a desirable property for chest X-ray diagnosis.
Moreover, we propose Regression IBA for explaining regression models. Using
Regression IBA we observe that a model trained on cumulative severity score
labels implicitly learns the severity of different X-ray regions. Finally, we
propose Multi-layer IBA to generate higher resolution and more detailed
attribution/saliency maps. We evaluate our methods using both human-centric
(ground-truth-based) interpretability metrics, and human-independent feature
importance metrics on NIH Chest X-ray8 and BrixIA datasets. The Code is
publicly available.

    

### [[2104.02481] Towards Semantic Interpretation of Thoracic Disease and COVID-19 Diagnosis Models](http://arxiv.org/abs/2104.02481)


  Convolutional neural networks are showing promise in the automatic diagnosis
of thoracic pathologies on chest x-rays. Their black-box nature has sparked
many recent works to explain the prediction via input feature attribution
methods (aka saliency methods). However, input feature attribution methods
merely identify the importance of input regions for the prediction and lack
semantic interpretation of model behavior. In this work, we first identify the
semantics associated with internal units (feature maps) of the network. We
proceed to investigate the following questions; Does a regression model that is
only trained with COVID-19 severity scores implicitly learn visual patterns
associated with thoracic pathologies? Does a network that is trained on weakly
labeled data (e.g. healthy, unhealthy) implicitly learn pathologies? Moreover,
we investigate the effect of pretraining and data imbalance on the
interpretability of learned features. In addition to the analysis, we propose
semantic attribution to semantically explain each prediction. We present our
findings using publicly available chest pathologies (CheXpert, NIH ChestX-ray8)
and COVID-19 datasets (BrixIA, and COVID-19 chest X-ray segmentation dataset).
The Code is publicly available.

    

### [[2104.07511] Ensemble of MRR and NDCG models for Visual Dialog](http://arxiv.org/abs/2104.07511)


  Assessing an AI agent that can converse in human language and understand
visual content is challenging. Generation metrics, such as BLEU scores favor
correct syntax over semantics. Hence a discriminative approach is often used,
where an agent ranks a set of candidate options. The mean reciprocal rank (MRR)
metric evaluates the model performance by taking into account the rank of a
single human-derived answer. This approach, however, raises a new challenge:
the ambiguity and synonymy of answers, for instance, semantic equivalence
(e.g., `yeah' and `yes'). To address this, the normalized discounted cumulative
gain (NDCG) metric has been used to capture the relevance of all the correct
answers via dense annotations. However, the NDCG metric favors the usually
applicable uncertain answers such as `I don't know. Crafting a model that
excels on both MRR and NDCG metrics is challenging. Ideally, an AI agent should
answer a human-like reply and validate the correctness of any answer. To
address this issue, we describe a two-step non-parametric ranking approach that
can merge strong MRR and NDCG models. Using our approach, we manage to keep
most MRR state-of-the-art performance (70.41% vs. 71.24%) and the NDCG
state-of-the-art performance (72.16% vs. 75.35%). Moreover, our approach won
the recent Visual Dialog 2020 challenge. Source code is available at
this https URL.

    

### [[2104.12822] Recommending Burgers based on Pizza Preferences: Addressing Data Sparsity with a Product of Experts](http://arxiv.org/abs/2104.12822)


  In this paper, we describe a method to tackle data sparsity and create
recommendations in domains with limited knowledge about user preferences. We
expand the variational autoencoder collaborative filtering from a single-domain
to a multi-domain setting. The intuition is that user-item interactions in a
source domain can augment the recommendation quality in a target domain. The
intuition can be taken to its extreme, where, in a cross-domain setup, the user
history in a source domain is enough to generate high-quality recommendations
in a target one. We thus create a Product-of-Experts (POE) architecture for
recommendations that jointly models user-item interactions across multiple
domains. The method is resilient to missing data for one or more of the
domains, which is a situation often found in real life. We present results on
two widely-used datasets - Amazon and Yelp, which support the claim that
holistic user preference knowledge leads to better recommendations.
Surprisingly, we find that in some cases, a POE recommender that does not
access the target domain user representation can surpass a strong VAE
recommender baseline trained on the target domain.

    

### [[2106.07597] MLPerf Tiny Benchmark](http://arxiv.org/abs/2106.07597)


  Advancements in ultra-low-power tiny machine learning (TinyML) systems
promise to unlock an entirely new class of smart applications. However,
continued progress is limited by the lack of a widely accepted and easily
reproducible benchmark for these systems. To meet this need, we present MLPerf
Tiny, the first industry-standard benchmark suite for ultra-low-power tiny
machine learning systems. The benchmark suite is the collaborative effort of
more than 50 organizations from industry and academia and reflects the needs of
the community. MLPerf Tiny measures the accuracy, latency, and energy of
machine learning inference to properly evaluate the tradeoffs between systems.
Additionally, MLPerf Tiny implements a modular design that enables benchmark
submitters to show the benefits of their product, regardless of where it falls
on the ML deployment stack, in a fair and reproducible manner. The suite
features four benchmarks: keyword spotting, visual wake words, image
classification, and anomaly detection.

    

### [[2106.12447] How Well do Feature Visualizations Support Causal Understanding of CNN Activations?](http://arxiv.org/abs/2106.12447)


  One widely used approach towards understanding the inner workings of deep
convolutional neural networks is to visualize unit responses via activation
maximization. Feature visualizations via activation maximization are thought to
provide humans with precise information about the image features that cause a
unit to be activated. If this is indeed true, these synthetic images should
enable humans to predict the effect of an intervention, such as whether
occluding a certain patch of the image (say, a dog's head) changes a unit's
activation. Here, we test this hypothesis by asking humans to predict which of
two square occlusions causes a larger change to a unit's activation. Both a
large-scale crowdsourced experiment and measurements with experts show that on
average, the extremely activating feature visualizations by Olah et al. (2017)
indeed help humans on this task ($67 \pm 4\%$ accuracy; baseline performance
without any visualizations is $60 \pm 3\%$). However, they do not provide any
significant advantage over other visualizations (such as e.g. dataset samples),
which yield similar performance ($66 \pm 3\%$ to $67 \pm 3\%$ accuracy). Taken
together, we propose an objective psychophysical task to quantify the benefit
of unit-level interpretability methods for humans, and find no evidence that
feature visualizations provide humans with better "causal understanding" than
simple alternative visualizations.

    

### [[2106.13434] Binary Matrix Factorisation and Completion via Integer Programming](http://arxiv.org/abs/2106.13434)


  Binary matrix factorisation is an essential tool for identifying discrete
patterns in binary data. In this paper we consider the rank-k binary matrix
factorisation problem (k-BMF) under Boolean arithmetic: we are given an n x m
binary matrix X with possibly missing entries and need to find two binary
matrices A and B of dimension n x k and k x m respectively, which minimise the
distance between X and the Boolean product of A and B in the squared Frobenius
distance. We present a compact and two exponential size integer programs (IPs)
for k-BMF and show that the compact IP has a weak LP relaxation, while the
exponential size IPs have a stronger equivalent LP relaxation. We introduce a
new objective function, which differs from the traditional squared Frobenius
objective in attributing a weight to zero entries of the input matrix that is
proportional to the number of times the zero is erroneously covered in a rank-k
factorisation. For one of the exponential size IPs we describe a computational
approach based on column generation. Experimental results on synthetic and real
word datasets suggest that our integer programming approach is competitive
against available methods for k-BMF and provides accurate low-error
factorisations.

    

### [[2106.13861] Auto-Pipeline: Synthesizing Complex Data Pipelines By-Target Using Reinforcement Learning and Search](http://arxiv.org/abs/2106.13861)


  Recent work has made significant progress in helping users to automate single
data preparation steps, such as string-transformations and table-manipulation
operators (e.g., Join, GroupBy, Pivot, etc.). We in this work propose to
automate multiple such steps end-to-end, by synthesizing complex data pipelines
with both string transformations and table-manipulation operators. We propose a
novel "by-target" paradigm that allows users to easily specify the desired
pipeline, which is a significant departure from the traditional by-example
paradigm. Using by-target, users would provide input tables (e.g., csv or json
files), and point us to a "target table" (e.g., an existing database table or
BI dashboard) to demonstrate how the output from the desired pipeline would
schematically "look like". While the problem is seemingly underspecified, our
unique insight is that implicit table constraints such as FDs and keys can be
exploited to significantly constrain the space to make the problem tractable.
We develop an Auto-Pipeline system that learns to synthesize pipelines using
reinforcement learning and search. Experiments on large numbers of real
pipelines crawled from GitHub suggest that Auto-Pipeline can successfully
synthesize 60-70% of these complex pipelines with up to 10 steps.

    

### [[2108.01290] Linking Sap Flow Measurements with Earth Observations](http://arxiv.org/abs/2108.01290)


  While single-tree transpiration is challenging to compare with earth
observation, canopy scale data are suitable for this purpose. To test the
potentialities of the second approach, we equipped the trees at two measurement
sites with sap flow sensors in spruce forests. The sites have contrasting
topography. The measurement period covered the months between June 2020 and
January 2021. To link plot scale transpiration with earth observations, we
utilized Sentinel-2 and local meteorological data. Within a machine learning
framework, we have tested the suitability of earth observations for modelling
canopy transpiration. The R2 of the cross-validated trained models at the
measurement sites was between 0.57 and 0.80. These results demonstrate the
relevance of Sentinel-2 data for the data-driven upscaling of ecosystem fluxes
from plot scale sap flow data. If applied to a broader network of sites and
climatic conditions, such an approach could offer unprecedented possibilities
for investigating our forests' resilience and resistance capacity to an
intensified hydrological cycle in the contest of a changing climate.

    

### [[2108.02023] DFSynthesizer: Dataflow-based Synthesis of Spiking Neural Networks to Neuromorphic Hardware](http://arxiv.org/abs/2108.02023)


  Spiking Neural Networks (SNN) are an emerging computation model, which uses
event-driven activation and bio-inspired learning algorithms. SNN-based
machine-learning programs are typically executed on tile- based neuromorphic
hardware platforms, where each tile consists of a computation unit called
crossbar, which maps neurons and synapses of the program. However, synthesizing
such programs on an off-the-shelf neuromorphic hardware is challenging. This is
because of the inherent resource and latency limitations of the hardware, which
impact both model performance, e.g., accuracy, and hardware performance, e.g.,
throughput. We propose DFSynthesizer, an end-to-end framework for synthesizing
SNN-based machine learning programs to neuromorphic hardware. The proposed
framework works in four steps. First, it analyzes a machine-learning program
and generates SNN workload using representative data. Second, it partitions the
SNN workload and generates clusters that fit on crossbars of the target
neuromorphic hardware. Third, it exploits the rich semantics of Synchronous
Dataflow Graph (SDFG) to represent a clustered SNN program, allowing for
performance analysis in terms of key hardware constraints such as number of
crossbars, dimension of each crossbar, buffer space on tiles, and tile
communication bandwidth. Finally, it uses a novel scheduling algorithm to
execute clusters on crossbars of the hardware, guaranteeing hardware
performance. We evaluate DFSynthesizer with 10 commonly used machine-learning
programs. Our results demonstrate that DFSynthesizer provides much tighter
performance guarantee compared to current mapping approaches.

    

### [[2108.02073] Efficient Hardware Realizations of Feedforward Artificial Neural Networks](http://arxiv.org/abs/2108.02073)


  This article presents design techniques proposed for efficient hardware
implementation of feedforward artificial neural networks (ANNs) under parallel
and time-multiplexed architectures. To reduce their design complexity, after
the weights of ANN are determined in a training phase, we introduce a technique
to find the minimum quantization value used to convert the floating-point
weight values to integers. For each design architecture, we also propose an
algorithm that tunes the integer weights to reduce the hardware complexity
avoiding a loss in the hardware accuracy. Furthermore, the multiplications of
constant weights by input variables are implemented under the shift-adds
architecture using the fewest number of addition/subtraction operations found
by prominent previously proposed algorithms. Finally, we introduce a
computer-aided design (CAD) tool, called SIMURG, that can describe an ANN
design in hardware automatically based on the ANN structure and the solutions
of proposed design techniques and algorithms. Experimental results indicate
that the tuning techniques can significantly reduce the ANN hardware complexity
under a design architecture and the multiplierless design of ANN can lead to a
significant reduction in area and energy consumption, increasing the latency
slightly.

    

### [[2108.02119] Low-complexity Scaling Methods for DCT-II Approximations](http://arxiv.org/abs/2108.02119)


  This paper introduces a collection of scaling methods for generating
$2N$-point DCT-II approximations based on $N$-point low-complexity
transformations. Such scaling is based on the Hou recursive matrix
factorization of the exact $2N$-point DCT-II matrix. Encompassing the widely
employed Jridi-Alfalou-Meher scaling method, the proposed techniques are shown
to produce DCT-II approximations that outperform the transforms resulting from
the JAM scaling method according to total error energy and mean squared error.
Orthogonality conditions are derived and an extensive error analysis based on
statistical simulation demonstrates the good performance of the introduced
scaling methods. A hardware implementation is also provided demonstrating the
competitiveness of the proposed methods when compared to the JAM scaling
method.

    

### [[2108.02156] STBPU: A Reasonably Safe Branch Predictor Unit](http://arxiv.org/abs/2108.02156)


  Modern processors have suffered a deluge of danger- ous side channel and
speculative execution attacks that exploit vulnerabilities rooted in branch
predictor units (BPU). Many such attacks exploit the shared use of the BPU
between un- related processes, which allows malicious processes to retrieve
sensitive data or enable speculative execution attacks. Attacks that exploit
collisions between different branch instructions inside the BPU are among the
most dangerous. Various protections and mitigations are proposed such as CPU
microcode updates, secured cache designs, fencing mechanisms, invisible
speculations. While some effectively mitigate speculative execution attacks,
they overlook BPU as an attack vector, leaving BPU prone to malicious
collisions and resulting critical penalty such as advanced micro-op cache
attacks. Furthermore, some mitigations severely hamper the accuracy of the BPU
resulting in increased CPU performance overhead. To address these, we present
the secret token branch predictor unit (STBPU), a branch predictor design that
mitigates collision-based speculative execution attacks and BPU side channel
whilst incurring little to no performance overhead. STBPU achieves this by
customizing inside data representations for each software entity requiring
isolation. To prevent more advanced attacks, STBPU monitors hardware events and
preemptively changes how STBPU data is stored and interpreted.

    

### [[2108.01740] Deterministic Distributed Algorithms and Lower Bounds in the Hybrid Model](http://arxiv.org/abs/2108.01740)


  The $\hybrid$ model was recently introduced by Augustine et al.
\cite{DBLP:conf/soda/AugustineHKSS20} in order to characterize from an
algorithmic standpoint the capabilities of networks which combine multiple
communication modes. Concretely, it is assumed that the standard $\local$ model
of distributed computing is enhanced with the feature of all-to-all
communication, but with very limited bandwidth, captured by the
node-capacitated clique ($\ncc$). In this work we provide several new insights
on the power of hybrid networks for fundamental problems in distributed
algorithms.
First, we present a deterministic algorithm which solves any problem on a
sparse $n$-node graph in $\widetilde{\mathcal{O}}(\sqrt{n})$ rounds of
$\hybrid$. We combine this primitive with several sparsification techniques to
obtain efficient distributed algorithms for general graphs. Most notably, for
the all-pairs shortest paths problem we give deterministic $(1 + \epsilon)$-
and $\log n/\log \log n$-approximate algorithms for unweighted and weighted
graphs respectively with round complexity $\widetilde{\mathcal{O}}(\sqrt{n})$
in $\hybrid$, closely matching the performance of the state of the art
randomized algorithm of Kuhn and Schneider \cite{10.1145/3382734.3405719}.
Moreover, we make a connection with the Ghaffari-Haeupler framework of
low-congestion shortcuts \cite{DBLP:conf/soda/GhaffariH16}, leading -- among
others -- to a $(1 + \epsilon)$-approximate algorithm for Min-Cut after
$\log^{\mathcal{O}(1)}n$ rounds, with high probability, even if we restrict
local edges to transfer $\mathcal{O}(\log n)$-bits per round. Finally, we prove
via a reduction from the set disjointness problem that
$\widetilde{\Omega}(n^{1/3})$ rounds are required to determine the radius of an
unweighted graph, as well as a $(3/2 - \epsilon)$-approximation for weighted
graphs.

    

### [[2108.01776] How Can Datacenters Join the Smart Grid to Address the Climate Crisis? Using Simulation to Explore Power and Cost Effects of Direct Participation in the Energy Market](http://arxiv.org/abs/2108.01776)


  Amidst the climate crisis, the massive introduction of renewable energy
sources has brought tremendous challenges to both the power grid and its
surrounding markets. As datacenters have become ever-larger and more powerful,
they play an increasingly significant role in the energy arena. With their
unique characteristics, datacenters have been proved to be well-suited for
regulating the power grid yet currently provide little, if any, such active
response. This problem is due to issues such as unsuitability of the market
design, high complexity of the currently proposed solutions, as well as the
potential risks thereof. This work aims to provide individual datacenters with
insights on the feasibility and profitability of directly participating in the
energy market. By modelling the power system of datacenters, and by conducting
simulations on real-world datacenter traces, we demonstrate the substantial
financial incentive for individual datacenters to directly participate in both
the day-ahead and the balancing markets. In turn, we suggest a new short-term,
direct scheme of market participation for individual datacenters in place of
the current long-term, inactive participation. Furthermore, we develop a novel
proactive DVFS scheduling algorithm that can both reduce energy consumption and
save energy costs during the market participation of datacenters. Also, in
developing this scheduler, we propose an innovative combination of machine
learning methods and the DVFS technology that can provide the power grid with
indirect demand response (DR). Our experimental results strongly support that
individual datacenters can and should directly participate in the energy market
both to save their energy costs and to curb their energy consumption, whilst
providing the power grid with indirect DR.

    

### [[2108.01900] Lachesis: Scalable Asynchronous BFT on DAG Streams](http://arxiv.org/abs/2108.01900)


  This paper consolidates the core technologies and key concepts of our novel
Lachesis consensus protocol and Fantom Opera platform, which is permissionless,
leaderless and EVM compatible.
We introduce our new protocol, so-called Lachesis, for distributed networks
achieving Byzantine fault tolerance (BFT)~\cite{lachesis01}. Each node in
Lachesis protocol operates on a local block DAG, namely \emph{OPERA DAG}.
Aiming for a low time to finality (TTF) for transactions, our general model
considers DAG streams of high speed but asynchronous events. We integrate
Proof-of-Stake (PoS) into a DAG model in Lachesis protocol to improve
performance and security. Our general model of trustless system leverages
participants' stake as their validating power~\cite{stakedag}. Lachesis's
consensus algorithm uses Lamport timestamps, graph layering and concurrent
common knowledge to guarantee a consistent total ordering of event blocks and
transactions. In addition, Lachesis protocol allows dynamic participation of
new nodes into Opera network. Lachesis optimizes DAG storage and processing
time by splitting local history into checkpoints (so-called epochs). We also
propose a model to improve stake decentralization, and network safety and
liveness ~\cite{stairdag}.
Built on our novel Lachesis protocol, Fantom's Opera platform is a public,
leaderless, asynchronous BFT, layer-1 blockchain, with guaranteed deterministic
finality. Hence, Lachesis protocol is suitable for distributed ledgers by
leveraging asynchronous partially ordered sets with logical time ordering
instead of blockchains. We also present our proofs into a model that can be
applied to abstract asynchronous distributed system.

    

### [[2108.01922] UniGPS: A Unified Programming Framework for Distributed Graph Processing](http://arxiv.org/abs/2108.01922)


  The industry and academia have proposed many distributed graph processing
systems. However, the existing systems are not friendly enough for users like
data analysts and algorithm engineers. On the one hand, the programing models
and interfaces differ a lot in the existing systems, leading to high learning
costs and program migration costs. On the other hand, these graph processing
systems are tightly bound to the underlying distributed computing platforms,
requiring users to be familiar with distributed computing. To improve the
usability of distributed graph processing, we propose a unified distributed
graph programming framework UniGPS. Firstly, we propose a unified
cross-platform graph programming model VCProg for UniGPS. VCProg hides details
of distributed computing from users. It is compatible with the popular graph
programming models Pregel, GAS, and Push-Pull. VCProg programs can be executed
by compatible distributed graph processing systems without modification,
reducing the learning overheads of users. Secondly, UniGPS supports Python as
the programming language. We propose an interprocess-communication-based
execution environment isolation mechanism to enable Java/C++-based systems to
call user-defined methods written in Python. The experimental results show that
UniGPS enables users to process big graphs beyond the memory capacity of a
single machine without sacrificing usability. UniGPS shows near-linear data
scalability and machine scalability.

    

### [[2108.01963] Deterministic Logarithmic Completeness in the Distributed Sleeping Model](http://arxiv.org/abs/2108.01963)


  We provide a deterministic scheme for solving any decidable problem in the
distributed {sleeping model}. The sleeping model is a generalization of the
standard message-passing model, with an additional capability of network nodes
to enter a sleeping state occasionally. As long as a vertex is in the awake
state, it is similar to the standard message-passing setting. However, when a
vertex is asleep it cannot receive or send messages in the network nor can it
perform internal computations. On the other hand, sleeping rounds do not count
towards {\awake complexity.} Awake complexity is the main complexity
measurement in this setting, which is the number of awake rounds a vertex
spends during an execution. In this paper we devise algorithms with worst-case
guarantees on the awake complexity.
We devise a deterministic scheme with awake complexity of $O(\log n)$ for
solving any decidable problem in this model by constructing a structure we call
{ Distributed Layered Tree}. This structure turns out to be very powerful in
the sleeping model, since it allows one to collect the entire graph information
within a constant number of awake rounds. Moreover, we prove that our general
technique cannot be improved in this model, by showing that the construction of
distributed layered trees itself requires $\Omega(\log n)$ awake rounds.
Another result we obtain in this work is a deterministic scheme for solving any
problem from a class of problems, denoted O-LOCAL, in $O(\log \Delta +
\log^*n)$ awake rounds. This class contains various well-studied problems, such
as MIS and $(\Delta+1)$-vertex-coloring.

    

### [[2108.01989] On Extending Brandt's Speedup Theorem from LOCAL to Round-Based Full-Information Models](http://arxiv.org/abs/2108.01989)


  Given any task $\Pi$, Brandt's speedup theorem (PODC 2019) provides a
mechanical way to design another task~$\Pi'$ on the same input-set as $\Pi$
such that, for any $t\geq 1$, $\Pi$ is solvable in $t$ rounds if and only if
$\Pi'$ is solvable in $t-1$ rounds. The theorem applies to the anonymous
variant of the LOCAL model, in graphs with sufficiently large girth, and to
locally checkable labeling (LCL) tasks. In this paper, using combinatorial
topology applied to distributed computing, we dissect the construction in
Brandt's speedup theorem for expressing it in the broader framework of
round-based models supporting full information protocols, which includes models
as different as wait-free shared-memory computing with iterated immediate
snapshots, and synchronous failure-free network computing. In particular, we
provide general definitions for notions such as local checkability and local
independence, in our broader framework. In this way, we are able to identify
the hypotheses on the computing model, and on the tasks, that are sufficient
for Brandt's speedup theorem to apply. More precisely, we identify which
hypotheses are sufficient for the each direction of the if-and-only-if
condition. Interestingly, these hypotheses are of different natures. Our
general approach enables to extend Brandt's speedup theorem from LOCAL to
directed networks, to hypergraphs, to dynamic networks, and even to graphs
including short cyclic dependencies between processes (i.e., the large girth
condition is, to some extend, not necessary). The theorem can even be extended
to shared-memory wait-free computing. In particular, we provide new
impossibility proofs for consensus and perfect renaming in 2-process systems.

    

### [[2108.02102] ErrorCompensatedX: error compensation for variance reduced algorithms](http://arxiv.org/abs/2108.02102)


  Communication cost is one major bottleneck for the scalability for
distributed learning. One approach to reduce the communication cost is to
compress the gradient during communication. However, directly compressing the
gradient decelerates the convergence speed, and the resulting algorithm may
diverge for biased compression. Recent work addressed this problem for
stochastic gradient descent by adding back the compression error from the
previous step. This idea was further extended to one class of variance reduced
algorithms, where the variance of the stochastic gradient is reduced by taking
a moving average over all history gradients. However, our analysis shows that
just adding the previous step's compression error, as done in existing work,
does not fully compensate the compression error. So, we propose
ErrorCompensatedX, which uses the compression error from the previous two
steps. We show that ErrorCompensatedX can achieve the same asymptotic
convergence rate with the training without compression. Moreover, we provide a
unified theoretical analysis framework for this class of variance reduced
algorithms, with or without error compensation.

    

### [[2108.02197] Singularly Near Optimal Leader Election in Asynchronous Networks](http://arxiv.org/abs/2108.02197)


  This paper concerns designing distributed algorithms that are {\em singularly
optimal}, i.e., algorithms that are {\em simultaneously} time and message {\em
optimal}, for the fundamental leader election problem in {\em asynchronous}
networks.
Kutten et al. (JACM 2015) presented a singularly near optimal randomized
leader election algorithm for general {\em synchronous} networks that ran in
$O(D)$ time and used $O(m \log n)$ messages (where $D$, $m$, and $n$ are the
network's diameter, number of edges and number of nodes, respectively) with
high probability.\footnote{Throughout, "with high probability" means "with
probability at least $1-1/n^c$, for constant $c$."} Both bounds are near
optimal (up to a logarithmic factor), since $\Omega(D)$ and $\Omega(m)$ are the
respective lower bounds for time and messages for leader election even for
synchronous networks and even for (Monte-Carlo) randomized algorithms. On the
other hand, for general asynchronous networks, leader election algorithms are
only known that are either time or message optimal, but not both. Kutten et al.
(DISC 2020) presented a randomized asynchronous leader election algorithm that
is singularly near optimal for \emph{complete networks}, but left open the
problem for general networks.
This paper shows that singularly near optimal (up to polylogarithmic factors)
bounds can be achieved for general {\em asynchronous} networks. We present a
randomized singularly near optimal leader election algorithm that runs in $O(D
+ \log^2n)$ time and $O(m\log^2 n)$ messages with high probability. Our result
is the first known distributed leader election algorithm for asynchronous
networks that is near optimal with respect to both time and message complexity
and improves over a long line of results including the classical results of
Gallager et al. (ACM TOPLAS, 1983), Peleg (JPDC, 1989), and Awerbuch (STOC 89).

    

### [[2012.06646] A fine-grained parallelization of the immersed boundary method](http://arxiv.org/abs/2012.06646)


  We present new algorithms for the parallelization of Eulerian-Lagrangian
interaction operations in the immersed boundary method. Our algorithms rely on
two well-studied parallel primitives: key-value sort and segmented reduce. The
use of these parallel primitives allows us to implement our algorithms on both
graphics processing units (GPUs) and on other shared memory architectures. We
present strong and weak scaling tests on problems involving scattered points
and elastic structures. Our tests show that our algorithms exhibit near-ideal
scaling on both multicore CPUs and GPUs.

    

### [[2105.07028] Methods Included: Standardizing Computational Reuse and Portability with the Common Workflow Language](http://arxiv.org/abs/2105.07028)


  Computational Workflows are widely used in data analysis, enabling innovation
and decision-making. In many domains (bioinformatics, image analysis, & radio
astronomy) the analysis components are numerous and written in multiple
different computer languages by third parties. However, many competing workflow
systems exist, severely limiting portability of such workflows, thereby
hindering the transfer of workflows between different systems, between
different projects and different settings, leading to vendor lock-ins and
limiting their generic re-usability. Here we present the Common Workflow
Language (CWL) project which produces free and open standards for describing
command-line tool based workflows. The CWL standards provide a common but
reduced set of abstractions that are both used in practice and implemented in
many popular workflow systems. The CWL language is declarative, which allows
expressing computational workflows constructed from diverse software tools,
executed each through their command-line interface. Being explicit about the
runtime environment and any use of software containers enables portability and
reuse. Workflows written according to the CWL standards are a reusable
description of that analysis that are runnable on a diverse set of computing
environments. These descriptions contain enough information for advanced
optimization without additional input from workflow authors. The CWL standards
support polylingual workflows, enabling portability and reuse of such
workflows, easing for example scholarly publication, fulfilling regulatory
requirements, collaboration in/between academic research and industry, while
reducing implementation costs. CWL has been taken up by a wide variety of
domains, and industries and support has been implemented in many major workflow
systems.

    

### [[2105.09756] Fully Adaptive Self-Stabilizing Transformer for LCL Problems](http://arxiv.org/abs/2105.09756)


  The first generic self-stabilizing transformer for local problems in a
constrained bandwidth model is introduced. This transformer can be applied to a
wide class of locally checkable labeling (LCL) problems, converting a given
fault free synchronous algorithm that satisfies certain conditions into a
self-stabilizing synchronous algorithm for the same problem. The resulting
self-stabilizing algorithms are anonymous, size-uniform, and \emph{fully
adaptive} in the sense that their time complexity is bounded as a function of
the number $k$ of nodes that suffered faults (possibly at different times)
since the last legal configuration. Specifically, for graphs whose degrees are
up-bounded by $\Delta$, the algorithms produced by the transformer stabilize in
time proportional to $\log (k + \Delta)$ in expectation, independently of the
number of nodes in the graph. As such, the transformer is applicable also for
infinite graphs (with degree bound $\Delta$). Another appealing feature of the
transformer is its small message size overhead. The transformer is applied to
known algorithms (or simple variants thereof) for some classic LCL problems,
producing the first anonymous size-uniform self-stabilizing algorithms for
these problems that are provably fully adaptive. From a technical point of
view, the transformer's key design feature is a novel probabilistic tool that
allows different nodes to act in synchrony even though their clocks may have
been adversarially manipulated.

    

### [[2108.01764] Q-Pain: A Question Answering Dataset to Measure Social Bias in Pain Management](http://arxiv.org/abs/2108.01764)


  Recent advances in Natural Language Processing (NLP), and specifically
automated Question Answering (QA) systems, have demonstrated both impressive
linguistic fluency and a pernicious tendency to reflect social biases. In this
study, we introduce Q-Pain, a dataset for assessing bias in medical QA in the
context of pain management, one of the most challenging forms of clinical
decision-making. Along with the dataset, we propose a new, rigorous framework,
including a sample experimental design, to measure the potential biases present
when making treatment decisions. We demonstrate its use by assessing two
reference Question-Answering systems, GPT-2 and GPT-3, and find statistically
significant differences in treatment between intersectional race-gender
subgroups, thus reaffirming the risks posed by AI in medical settings, and the
need for datasets like ours to ensure safety before medical AI applications are
deployed.

    

### [[2108.01884] Adaptive Path Planning for UAV-based Multi-Resolution Semantic Segmentation](http://arxiv.org/abs/2108.01884)


  In this paper, we address the problem of adaptive path planning for accurate
semantic segmentation of terrain using unmanned aerial vehicles (UAVs). The
usage of UAVs for terrain monitoring and remote sensing is rapidly gaining
momentum due to their high mobility, low cost, and flexible deployment.
However, a key challenge is planning missions to maximize the value of acquired
data in large environments given flight time limitations. To address this, we
propose an online planning algorithm which adapts the UAV paths to obtain
high-resolution semantic segmentations necessary in areas on the terrain with
fine details as they are detected in incoming images. This enables us to
perform close inspections at low altitudes only where required, without wasting
energy on exhaustive mapping at maximum resolution. A key feature of our
approach is a new accuracy model for deep learning-based architectures that
captures the relationship between UAV altitude and semantic segmentation
accuracy. We evaluate our approach on the application of crop/weed segmentation
in precision agriculture using real-world field data.

    

### [[2108.01955] Log-based Anomaly Detection Without Log Parsing](http://arxiv.org/abs/2108.01955)


  Software systems often record important runtime information in system logs
for troubleshooting purposes. There have been many studies that use log data to
construct machine learning models for detecting system anomalies. Through our
empirical study, we find that existing log-based anomaly detection approaches
are significantly affected by log parsing errors that are introduced by 1) OOV
(out-of-vocabulary) words, and 2) semantic misunderstandings. The log parsing
errors could cause the loss of important information for anomaly detection. To
address the limitations of existing methods, we propose NeuralLog, a novel
log-based anomaly detection approach that does not require log parsing.
NeuralLog extracts the semantic meaning of raw log messages and represents them
as semantic vectors. These representation vectors are then used to detect
anomalies through a Transformer-based classification model, which can capture
the contextual information from log sequences. Our experimental results show
that the proposed approach can effectively understand the semantic meaning of
log messages and achieve accurate anomaly detection results. Overall, NeuralLog
achieves F1-scores greater than 0.95 on four public datasets, outperforming the
existing approaches.

    

### [[2108.01987] Core-Stable Committees under Restricted Domains](http://arxiv.org/abs/2108.01987)


  We study the setting of committee elections, where a group of individuals
needs to collectively select a given size subset of available objects. This
model is relevant for a number of real-life scenarios including political
elections, participatory budgeting, and facility-location. We focus on the core
-- the classic notion of proportionality, stability and fairness. We show that
for a number of restricted domains including voter-interval,
candidate-interval, single-peaked, and single-crossing preferences the core is
non-empty and can be found in polynomial time. We show that the core might be
empty for strict top-monotonic preferences, yet we introduce a relaxation of
this class, which guarantees non-emptiness of the core. Our algorithms work
both in the randomized and discrete models. We also show that the classic known
proportional rules do not return committees from the core even for the most
restrictive domains among those we consider (in particular for 1D-Euclidean
preferences). We additionally prove a number of structural results that give
better insights into the nature of some of the restricted domains, and which in
particular give a better intuitive understanding of the class of top-monotonic
preferences.

    

### [[2108.01996] Exact and Heuristic Approaches to Drone Delivery Problems](http://arxiv.org/abs/2108.01996)


  The Flying Sidekick Traveling Salesman Problem (FSTSP) considers a delivery
system composed by a truck and a drone. The drone launches from the truck with
a single package to deliver to a customer. Each drone must return to the truck
to recharge batteries, pick up another package, and launch again to a new
customer location. This work proposes a novel Mixed Integer Programming (MIP)
formulation and a heuristic approach to address the problem. The proposedMIP
formulation yields better linear relaxation bounds than previously proposed
formulations for all instances, and was capable of optimally solving several
unsolved instances from the literature. A hybrid heuristic based on the General
Variable Neighborhood Search metaheuristic combining Tabu Search concepts is
employed to obtain high-quality solutions for large-size instances. The
efficiency of the algorithm was evaluated on 1415 benchmark instances from the
literature, and over 80% of the best known solutions were improved.

    

### [[2108.02000] Do What You Know: Coupling Knowledge with Action in Discrete-Event Systems](http://arxiv.org/abs/2108.02000)


  An epistemic model for decentralized discrete-event systems with non-binary
control is presented. This framework combines existing work on conditional
control decisions with existing work on formal reasoning about knowledge in
discrete-event systems. The novelty in the model presented is that the
necessary and sufficient conditions for problem solvability encapsulate the
actions that supervisors must take. This direct coupling between knowledge and
action -- in a formalism that mimics natural language -- makes it easier, when
the problem conditions fail, to determine how the problem requirements should
be revised.

    

### [[2108.02006] On the Importance of Domain-specific Explanations in AI-based Cybersecurity Systems (Technical Report)](http://arxiv.org/abs/2108.02006)


  With the availability of large datasets and ever-increasing computing power,
there has been a growing use of data-driven artificial intelligence systems,
which have shown their potential for successful application in diverse areas.
However, many of these systems are not able to provide information about the
rationale behind their decisions to their users. Lack of understanding of such
decisions can be a major drawback, especially in critical domains such as those
related to cybersecurity. In light of this problem, in this paper we make three
contributions: (i) proposal and discussion of desiderata for the explanation of
outputs generated by AI-based cybersecurity systems; (ii) a comparative
analysis of approaches in the literature on Explainable Artificial Intelligence
(XAI) under the lens of both our desiderata and further dimensions that are
typically used for examining XAI approaches; and (iii) a general architecture
that can serve as a roadmap for guiding research efforts towards the
development of explainable AI-based cybersecurity systems -- at its core, this
roadmap proposes combinations of several research lines in a novel way towards
tackling the unique challenges that arise in this context.

    

### [[2108.02029] Signature Verification using Geometrical Features and Artificial Neural Network Classifier](http://arxiv.org/abs/2108.02029)


  Signature verification has been one of the major researched areas in the
field of computer vision. Many financial and legal organizations use signature
verification as access control and authentication. Signature images are not
rich in texture; however, they have much vital geometrical information. Through
this work, we have proposed a signature verification methodology that is simple
yet effective. The technique presented in this paper harnesses the geometrical
features of a signature image like center, isolated points, connected
components, etc., and with the power of Artificial Neural Network (ANN)
classifier, classifies the signature image based on their geometrical features.
Publicly available dataset MCYT, BHSig260 (contains the image of two regional
languages Bengali and Hindi) has been used in this paper to test the
effectiveness of the proposed method. We have received a lower Equal Error Rate
(EER) on MCYT 100 dataset and higher accuracy on the BHSig260 dataset.

    

### [[2108.02170] Curriculum learning for language modeling](http://arxiv.org/abs/2108.02170)


  Language Models like ELMo and BERT have provided robust representations of
natural language, which serve as the language understanding component for a
diverse range of downstream tasks.Curriculum learning is a method that employs
a structured training regime instead, which has been leveraged in computer
vision and machine translation to improve model training speed and model
performance. While language models have proven transformational for the natural
language processing community, these models have proven expensive,
energy-intensive, and challenging to train. In this work, we explore the effect
of curriculum learning on language model pretraining using various
linguistically motivated curricula and evaluate transfer performance on the
GLUE Benchmark. Despite a broad variety of training methodologies and
experiments we do not find compelling evidence that curriculum learning methods
improve language model training.

    

### [[2010.12854] ReadOnce Transformers: Reusable Representations of Text for Transformers](http://arxiv.org/abs/2010.12854)


  We present ReadOnce Transformers, an approach to convert a transformer-based
model into one that can build an information-capturing, task-independent, and
compressed representation of text. The resulting representation is reusable
across different examples and tasks, thereby requiring a document shared across
many examples or tasks to only be \emph{read once}. This leads to faster
training and evaluation of models. Additionally, we extend standard
text-to-text transformer models to Representation+Text-to-text models, and
evaluate on multiple downstream tasks: multi-hop QA, abstractive QA, and
long-document summarization. Our one-time computed representation results in a
2x-5x speedup compared to standard text-to-text models, while the compression
also allows existing language models to handle longer documents without the
need for designing new pre-trained models.

    

### [[2011.12077] CLAWS: Clustering Assisted Weakly Supervised Learning with Normalcy Suppression for Anomalous Event Detection](http://arxiv.org/abs/2011.12077)


  Learning to detect real-world anomalous events through video-level labels is
a challenging task due to the rare occurrence of anomalies as well as noise in
the labels. In this work, we propose a weakly supervised anomaly detection
method which has manifold contributions including1) a random batch based
training procedure to reduce inter-batch correlation, 2) a normalcy suppression
mechanism to minimize anomaly scores of the normal regions of a video by taking
into account the overall information available in one training batch, and 3) a
clustering distance based loss to contribute towards mitigating the label noise
and to produce better anomaly representations by encouraging our model to
generate distinct normal and anomalous clusters. The proposed method
obtains83.03% and 89.67% frame-level AUC performance on the UCF Crime and
ShanghaiTech datasets respectively, demonstrating its superiority over the
existing state-of-the-art algorithms.

    

### [[2104.10535] Exploiting Learned Policies in Focal Search](http://arxiv.org/abs/2104.10535)


  Recent machine-learning approaches to deterministic search and
domain-independent planning employ policy learning to speed up search.
Unfortunately, when attempting to solve a search problem by successively
applying a policy, no guarantees can be given on solution quality. The problem
of how to effectively use a learned policy within a bounded-suboptimal search
algorithm remains largely as an open question. In this paper, we propose
various ways in which such policies can be integrated into Focal Search,
assuming that the policy is a neural network classifier. Furthermore, we
provide mathematical foundations for some of the resulting algorithms. To
evaluate the resulting algorithms over a number of policies with varying
accuracy, we use synthetic policies which can be generated for a target
accuracy for problems where the search space can be held in memory. We evaluate
our focal search variants over three benchmark domains using our synthetic
approach, and on the 15-puzzle using a neural network learned using 1.5 million
examples. We observe that Discrepancy Focal Search, which we show expands the
node which maximizes an approximation of the probability that its corresponding
path is a prefix of an optimal path, obtains, in general, the best results in
terms of runtime and solution quality.

    

### [[2106.05654] Visual scoping operations for physical assembly](http://arxiv.org/abs/2106.05654)


  Planning is hard. The use of subgoals can make planning more tractable, but
selecting these subgoals is computationally costly. What algorithms might
enable us to reap the benefits of planning using subgoals while minimizing the
computational overhead of selecting them? We propose visual scoping, a strategy
that interleaves planning and acting by alternately defining a spatial region
as the next subgoal and selecting actions to achieve it. We evaluated our
visual scoping algorithm on a variety of physical assembly problems against two
baselines: planning all subgoals in advance and planning without subgoals. We
found that visual scoping achieves comparable task performance to the subgoal
planner while requiring only a fraction of the total computational cost.
Together, these results contribute to our understanding of how humans might
make efficient use of cognitive resources to solve complex planning problems.

    

### [[2106.14625] AMU-EURANOVA at CASE 2021 Task 1: Assessing the stability of multilingual BERT](http://arxiv.org/abs/2106.14625)


  This paper explains our participation in task 1 of the CASE 2021 shared task.
This task is about multilingual event extraction from news. We focused on
sub-task 4, event information extraction. This sub-task has a small training
dataset and we fine-tuned a multilingual BERT to solve this sub-task. We
studied the instability problem on the dataset and tried to mitigate it.

    

### [[2108.02011] High-performance Passive Eigen-model-based Detectors of Single Emitter Using Massive MIMO Receivers](http://arxiv.org/abs/2108.02011)


  For a passive direction of arrival (DoA) measurement system using massive
multiple input multiple output (MIMO), it is mandatory to infer whether the
emitter exists or not before performing DOA estimation operation. Inspired by
the detection idea from radio detection and ranging (radar), three
high-performance detectors are proposed to infer the existence of single
passive emitter from the eigen-space of sample covariance matrix of receive
signal vector. The test statistic (TS) of the first method is defined as the
ratio of maximum eigen-value (Max-EV) to minimum eigen-value (R-MaxEV-MinEV)
while that of the second one is defined as the ratio of Max-EV to noise
variance (R-MaxEV-NV). The TS of the third method is the mean of maximum
eigen-value (EV) and minimum EV(M-MaxEV-MinEV). Their closed-form expressions
are presented and the corresponding detection performance is given. Simulation
results show that the proposed M-MaxEV-MinEV and R-MaxEV-NV methods can
approximately achieve the same detection performance that is better than the
traditional generalized likelihood ratio test method with false alarm
probability being less than 0.3.

    

### [[2108.01883] Reasoning about Iteration and Recursion Uniformly based on Big-step Semantics](http://arxiv.org/abs/2108.01883)


  A reliable technique for deductive program verification should be proven
sound with respect to the semantics of the programming language. For each
different language, the construction of a separate soundness proof is often a
laborious undertaking. In language-independent program verification, common
aspects of computer programs are addressed to enable sound reasoning for all
languages. In this work, we propose a solution for the sound reasoning about
iteration and recursion based on the big-step operational semantics of any
programming language. We give inductive proofs on the soundness and relative
completeness of our reasoning technique. We illustrate the technique at
simplified programming languages of the imperative and functional paradigms,
with diverse features. We also mechanism all formal results in the Coq proof
assistant.

    

### [[2108.02188] On Lexicographic Proof Rules for Probabilistic Termination](http://arxiv.org/abs/2108.02188)


  We consider the almost-sure (a.s.) termination problem for probabilistic
programs, which are a stochastic extension of classical imperative programs.
Lexicographic ranking functions provide a sound and practical approach for
termination of non-probabilistic programs, and their extension to probabilistic
programs is achieved via lexicographic ranking supermartingales (LexRSMs).
However, LexRSMs introduced in the previous work have a limitation that impedes
their automation: all of their components have to be non-negative in all
reachable states. This might result in LexRSM not existing even for simple
terminating programs. Our contributions are twofold: First, we introduce a
generalization of LexRSMs which allows for some components to be negative. This
standard feature of non-probabilistic termination proofs was hitherto not known
to be sound in the probabilistic setting, as the soundness proof requires a
careful analysis of the underlying stochastic process. Second, we present
polynomial-time algorithms using our generalized LexRSMs for proving a.s.
termination in broad classes of linear-arithmetic programs.

    

### [[2105.12819] Implementation of Live Reverse Debugging in LLDB](http://arxiv.org/abs/2105.12819)


  Debugging is an essential process with a large share of the development
effort, being a relentless quest for offensive code through tracing, inspection
and iterative running sessions. Probably every developer has been in a
situation with a clear wish to rewind time just for a while, only to retry some
actions alternatively, instead of restarting the entire session. Well, the
genie to fulfill such a wish is known as a reverse debugger. Their inherent
technical complexity makes them very hard to implement, while the imposed
execution overhead turns them to less preferable for adoption. There are only a
few available, most being off-line tools, working on recorded, previously run,
sessions. We consider live reverse debuggers both challenging and promising,
since they can fit into existing forward debuggers, and we developed the first
live reverse debugger on top of LLDB, discussing in detail our implementation
approach.

    