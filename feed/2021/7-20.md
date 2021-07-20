
## 2021-7-20

### [[2107.08194] Towards autonomic orchestration of machine learning pipelines in future networks](http://arxiv.org/abs/2107.08194)


  Machine learning (ML) techniques are being increasingly used in mobile
networks for network planning, operation, management, optimisation and much
more. These techniques are realised using a set of logical nodes known as ML
pipeline. A single network operator might have thousands of such ML pipelines
distributed across its network. These pipelines need to be managed and
orchestrated across network domains. Thus it is essential to have autonomic
multi-domain orchestration of ML pipelines in mobile networks. International
Telecommunications Union (ITU) has provided an architectural framework for
management and orchestration of ML pipelines in future networks. We extend this
framework to enable autonomic orchestration of ML pipelines across multiple
network domains. We present our system architecture and describe its
application using a smart factory use case. Our work allows autonomic
orchestration of multi-domain ML pipelines in a standardised, technology
agnostic, privacy preserving fashion.

    

### [[2107.08300] Priority-Queue based Dynamic Scaling for Efficient Resource Allocation in Fog Computing](http://arxiv.org/abs/2107.08300)


  In this emerging world of connected devices, the need for more computing
devices with a focus on delay-sensitive application is critical. In this paper,
we propose a priority-queue based Fog computing architecture combined with
dynamic scalability of fog devices, which not only reduces the delay
experienced by delay-sensitive tasks by categorizing the delay-sensitive and
delay-insensitive tasks, but also dynamically allocates the fog devices within
the network depending upon the computation load for reducing the power
consumption. The results show that the proposed algorithm is able to achieve a
significantly lower delay for both delay-sensitive and -insensitive tasks when
compared with other related schemes with a 14.5% lower power consumption.

    

### [[2107.08334] Predictable Bandwidth Slicing with Open vSwitch](http://arxiv.org/abs/2107.08334)


  Software switching, a.k.a virtual switching, plays a vital role in network
virtualization and network function virtualization, enhances configurability,
and reduces deployment and operational costs. Software switching also
facilitates the development of edge and fog computing networks by allowing the
use of commodity hardware for both data processing and packet switching.
Despite these benefits, characterizing and ensuring deterministic performance
with software switches is harder, compared to physical switching appliances. In
particular, achieving deterministic performance is essential to adopt software
switching in mission-critical applications, especially those deployed in edge
and fog computing architectures. In this paper, we study the impact of switch
configurations on bandwidth slicing and predictable packet latency. We
demonstrate that latency and predictability are dependent on the implementation
of the bandwidth slicing mechanism and that the packet schedulers used in OVS
Kernel-Path and OVS-DPDK each focus on different aspects of switching
performance.

    

### [[2107.08335] Silent Tracker: In-band Beam Management for Soft Handover for mm-Wave Networks](http://arxiv.org/abs/2107.08335)


  In mm-wave networks, cell sizes are small due to high path and penetration
losses. Mobiles need to frequently switch softly from one cell to another to
preserve network connections and context. Each soft handover involves the
mobile performing directional neighbor cell search, tracking cell beam,
completing cell access request, and finally, context switching. The mobile must
independently discover cell beams, derive timing information, and maintain beam
alignment throughout the process to avoid packet loss and hard handover. We
propose Silent tracker which enables a mobile to reliably manage handover
events by maintaining an aligned beam until the successful handover completion.
It is entirely in-band beam mechanism that does not need any side information.
Experimental evaluations show that Silent Tracker maintains the mobile's
receive beam aligned to the potential target base station's transmit beam till
the successful conclusion of handover in three mobility scenarios: human walk,
device rotation, and 20 mph vehicular speed.

    

### [[2107.08336] QuicSDN: Transitioning from TCP to QUIC for Southbound Communication in SDNs](http://arxiv.org/abs/2107.08336)


  Transport and security layer protocols make up the backbone of communication
between end point devices. In Software Defined Networking (SDN), these
protocols play a crucial role in both control-plane and data-plane
communications. However, the current transport and security layer protocols:
TCP and TLS, are unable to keep up with the pace of SDN application
development. For these applications, the TCP/TLS protocol suite generates
excessive network overhead. After identifying the main overhead origins, we
demonstrate that using QUIC as the SDN transport layer protocol significantly
reduces the overhead and improves the efficiency of the network. In this paper,
we introduce quicSDN to enable robust, low-overhead communication between the
controller and switches. We ran a variety of experiments to highlight quicSDN's
benefits, and compared experimental results with transport-layer overhead
prediction models. quicSDN's performance is evaluated in terms of network
overhead reduction and we also demonstrated quicSDN's connection migration
capabilities. First, we compare the differences in controller-switch
communication overhead between tcpSDN(SDN over TCP) and quicSDN. Overhead
reduction was measured in three scenarios: flow rule installation, queue
configuration, and flow rule statistics polling. Second, we compare the
connection migration of quicSDN and tcpSDN; QUIC's ability to quickly migrate
connections allows for reduced total traffic in comparison to TCP. Overall, our
results show that quicSDN outperformed tcpSDN in our SDN scenarios, and as
such, QUIC is a promising candidate as an SDN transport layer protocol in the
future.

    

### [[2107.08490] System-Wide Security for Offline Payment Terminals](http://arxiv.org/abs/2107.08490)


  Most self-service payment terminals require network connectivity for
processing electronic payments. The necessity to maintain network connectivity
increases costs, introduces cybersecurity risks, and significantly limits the
number of places where the terminals can be installed. Leading payment service
providers have proposed offline payment solutions that rely on algorithmically
generated payment tokens. Existing payment token solutions, however, require
complex mechanisms for authentication, transaction management, and most
importantly, security risk management. In this paper, we present VolgaPay, a
blockchain-based system that allows merchants to deploy secure offline payment
terminal infrastructure that does not require collection and storage of any
sensitive data. We design a novel payment protocol which mitigates security
threats for all the participants of VolgaPay, such that the maximum loss from
gaining full access to any component by an adversary incurs only a limited
scope of harm. We achieve significant enhancements in security, operation
efficiency, and cost reduction via a combination of polynomial multi-hash chain
micropayment channels and blockchain grafting for off-chain channel state
transition. We implement the VolgaPay payment system, and with thorough
evaluation and security analysis, we demonstrate that VolgaPay is capable of
delivering a fast, secure, and cost-efficient solution for offline payment
terminals.

    

### [[2107.08617] DeepCC: Bridging the Gap Between Congestion Control and Applications via Multi-Objective Optimization](http://arxiv.org/abs/2107.08617)


  The increasingly complicated and diverse applications have distinct network
performance demands, e.g., some desire high throughput while others require low
latency. Traditional congestion controls (CC) have no perception of these
demands. Consequently, literatures have explored the objective-specific
algorithms, which are based on either offline training or online learning, to
adapt to certain application demands. However, once generated, such algorithms
are tailored to a specific performance objective function. Newly emerged
performance demands in a changeable network environment require either
expensive retraining (in the case of offline training), or manually redesigning
a new objective function (in the case of online learning). To address this
problem, we propose a novel architecture, DeepCC. It generates a CC agent that
is generically applicable to a wide range of application requirements and
network conditions. The key idea of DeepCC is to leverage both offline deep
reinforcement learning and online fine-tuning. In the offline phase, instead of
training towards a specific objective function, DeepCC trains its deep neural
network model using multi-objective optimization. With the trained model,
DeepCC offers near Pareto optimal policies w.r.t different user-specified
trade-offs between throughput, delay, and loss rate without any redesigning or
retraining. In addition, a quick online fine-tuning phase further helps DeepCC
achieve the application-specific demands under dynamic network conditions. The
simulation and real-world experiments show that DeepCC outperforms
state-of-the-art schemes in a wide range of settings. DeepCC gains a higher
target completion ratio of application requirements up to 67.4% than that of
other schemes, even in an untrained environment.

    

### [[2107.08619] Trends in LEO Satellite Handover Algorithms](http://arxiv.org/abs/2107.08619)


  In this paper, we review well-known handovers algorithms in satellite
environment. The modern research trends and contributions are proposed and
summarized in order to overcome their considering problems in
satellite-air-ground integrated network environment caused by the fast movement
of Low Earth Orbit (LEO) satellite and related frequent handover occurrences.

    

### [[2107.08671] Zero Trust Service Function Chaining](http://arxiv.org/abs/2107.08671)


  In this paper, we address the inefficient handling of traditional security
functions in Zero Trust (ZT) networks. For this reason, we propose a novel
network security concept that combines the ideas of ZT and Service Function
Chaining (SFC). This allows us to efficiently decide which security functions
to apply to which packets and when.

    

### [[2107.08681] A New Distributed Method for Training Generative Adversarial Networks](http://arxiv.org/abs/2107.08681)


  Generative adversarial networks (GANs) are emerging machine learning models
for generating synthesized data similar to real data by jointly training a
generator and a discriminator. In many applications, data and computational
resources are distributed over many devices, so centralized computation with
all data in one location is infeasible due to privacy and/or communication
constraints. This paper proposes a new framework for training GANs in a
distributed fashion: Each device computes a local discriminator using local
data; a single server aggregates their results and computes a global GAN.
Specifically, in each iteration, the server sends the global GAN to the
devices, which then update their local discriminators; the devices send their
results to the server, which then computes their average as the global
discriminator and updates the global generator accordingly. Two different
update schedules are designed with different levels of parallelism between the
devices and the server. Numerical results obtained using three popular datasets
demonstrate that the proposed framework can outperform a state-of-the-art
framework in terms of convergence speed.

    

### [[2106.07536] Throughput Maximization Leveraging Just-Enough SNR Margin and Channel Spacing Optimization](http://arxiv.org/abs/2106.07536)


  Flexible optical network is a promising technology to accommodate
high-capacity demands in next-generation networks. To ensure uninterrupted
communication, existing lightpath provisioning schemes are mainly done with the
assumption of worst-case resource under-provisioning and fixed channel spacing,
which preserves an excessive signal-to-noise ratio (SNR) margin. However, under
a resource over-provisioning scenario, the excessive SNR margin restricts the
transmission bit-rate or transmission reach, leading to physical layer resource
waste and stranded transmission capacity. To tackle this challenging problem,
we leverage an iterative feedback tuning algorithm to provide a just-enough SNR
margin, so as to maximize the network throughput. Specifically, the proposed
algorithm is implemented in three steps. First, starting from the high SNR
margin setup, we establish an integer linear programming model as well as a
heuristic algorithm to maximize the network throughput by solving the problem
of routing, modulation format, forward error correction, baud-rate selection,
and spectrum assignment. Second, we optimize the channel spacing of the
lightpaths obtained from the previous step, thereby increasing the available
physical layer resources. Finally, we iteratively reduce the SNR margin of each
lightpath until the network throughput cannot be increased. Through numerical
simulations, we confirm the throughput improvement in different networks and
with different baud-rates. In particular, we find that our algorithm enables
over 20\% relative gain when network resource is over-provisioned, compared to
the traditional method preserving an excessive SNR margin.

    

### [[2107.08045] Desiderata for Explainable AI in statistical production systems of the European Central Bank](http://arxiv.org/abs/2107.08045)


  Explainable AI constitutes a fundamental step towards establishing fairness
and addressing bias in algorithmic decision-making. Despite the large body of
work on the topic, the benefit of solutions is mostly evaluated from a
conceptual or theoretical point of view and the usefulness for real-world use
cases remains uncertain. In this work, we aim to state clear user-centric
desiderata for explainable AI reflecting common explainability needs
experienced in statistical production systems of the European Central Bank. We
link the desiderata to archetypical user roles and give examples of techniques
and methods which can be used to address the user's needs. To this end, we
provide two concrete use cases from the domain of statistical data production
in central banks: the detection of outliers in the Centralised Securities
Database and the data-driven identification of data quality checks for the
Supervisory Banking data system.

    

### [[2107.08066] LeanML: A Design Pattern To Slash Avoidable Wastes in Machine Learning Projects](http://arxiv.org/abs/2107.08066)


  We introduce the first application of the lean methodology to machine
learning projects. Similar to lean startups and lean manufacturing, we argue
that lean machine learning (LeanML) can drastically slash avoidable wastes in
commercial machine learning projects, reduce the business risk in investing in
machine learning capabilities and, in so doing, further democratize access to
machine learning. The lean design pattern we propose in this paper is based on
two realizations. First, it is possible to estimate the best performance one
may achieve when predicting an outcome $y \in \mathcal{Y}$ using a given set of
explanatory variables $x \in \mathcal{X}$, for a wide range of performance
metrics, and without training any predictive model. Second, doing so is
considerably easier, faster, and cheaper than learning the best predictive
model. We derive formulae expressing the best $R^2$, MSE, classification
accuracy, and log-likelihood per observation achievable when using $x$ to
predict $y$ as a function of the mutual information $I\left(y; x\right)$, and
possibly a measure of the variability of $y$ (e.g. its Shannon entropy in the
case of classification accuracy, and its variance in the case regression MSE).
We illustrate the efficacy of the LeanML design pattern on a wide range of
regression and classification problems, synthetic and real-life.

    

### [[2107.08068] Refined Policy Improvement Bounds for MDPs](http://arxiv.org/abs/2107.08068)


  The policy improvement bound on the difference of the discounted returns
plays a crucial role in the theoretical justification of the trust-region
policy optimization (TRPO) algorithm. The existing bound leads to a degenerate
bound when the discount factor approaches one, making the applicability of TRPO
and related algorithms questionable when the discount factor is close to one.
We refine the results in \cite{Schulman2015, Achiam2017} and propose a novel
bound that is "continuous" in the discount factor. In particular, our bound is
applicable for MDPs with the long-run average rewards as well.

    

### [[2107.08074] A comparative study of stochastic and deep generative models for multisite precipitation synthesis](http://arxiv.org/abs/2107.08074)


  Future climate change scenarios are usually hypothesized using simulations
from weather generators. However, there only a few works comparing and
evaluating promising deep learning models for weather generation against
classical approaches. This study shows preliminary results making such
evaluations for the multisite precipitation synthesis task. We compared two
open-source weather generators: IBMWeathergen (an extension of the Weathergen
library) and RGeneratePrec, and two deep generative models: GAN and VAE, on a
variety of metrics. Our preliminary results can serve as a guide for improving
the design of deep learning architectures and algorithms for the multisite
precipitation synthesis task.

    

### [[2107.08083] Robust Risk-Sensitive Reinforcement Learning Agents for Trading Markets](http://arxiv.org/abs/2107.08083)


  Trading markets represent a real-world financial application to deploy
reinforcement learning agents, however, they carry hard fundamental challenges
such as high variance and costly exploration. Moreover, markets are inherently
a multiagent domain composed of many actors taking actions and changing the
environment. To tackle these type of scenarios agents need to exhibit certain
characteristics such as risk-awareness, robustness to perturbations and low
learning variance. We take those as building blocks and propose a family of
four algorithms. First, we contribute with two algorithms that use risk-averse
objective functions and variance reduction techniques. Then, we augment the
framework to multi-agent learning and assume an adversary which can take over
and perturb the learning process. Our third and fourth algorithms perform well
under this setting and balance theoretical guarantees with practical use.
Additionally, we consider the multi-agent nature of the environment and our
work is the first one extending empirical game theory analysis for multi-agent
learning by considering risk-sensitive payoffs.

    

### [[2107.08090] Near-Optimal Algorithms for Linear Algebra in the Current Matrix Multiplication Time](http://arxiv.org/abs/2107.08090)


  Currently, in the numerical linear algebra community, it is thought that to
obtain nearly-optimal bounds for various problems such as rank computation,
finding a maximal linearly independent subset of columns, regression, low rank
approximation, maximum matching on general graphs and linear matroid union, one
would need to resolve the main open question of Nelson and Nguyen (FOCS, 2013)
regarding the logarithmic factors in the sketching dimension for existing
constant factor approximation oblivious subspace embeddings. We show how to
bypass this question using a refined sketching technique, and obtain optimal or
nearly optimal bounds for these problems. A key technique we use is an explicit
mapping of Indyk based on uncertainty principles and extractors, which after
first applying known oblivious subspace embeddings, allows us to quickly spread
out the mass of the vector so that sampling is now effective, and we avoid a
logarithmic factor that is standard in the sketching dimension resulting from
matrix Chernoff bounds. For the fundamental problems of rank computation and
finding a linearly independent subset of columns, our algorithms improve
Cheung, Kwok, and Lau (JACM, 2013) and are optimal to within a constant factor
and a $\log\log(n)$-factor, respectively. Further, for constant factor
regression and low rank approximation we give the first optimal algorithms, for
the current matrix multiplication exponent.

    

### [[2107.08096] Learning to Limit Data Collection via Scaling Laws: Data Minimization Compliance in Practice](http://arxiv.org/abs/2107.08096)


  Data minimization is a legal obligation defined in the European Union's
General Data Protection Regulation (GDPR) as the responsibility to process an
adequate, relevant, and limited amount of personal data in relation to a
processing purpose. However, unlike fairness or transparency, the principle has
not seen wide adoption for machine learning systems due to a lack of
computational interpretation. In this paper, we build on literature in machine
learning and law to propose the first learning framework for limiting data
collection based on an interpretation that ties the data collection purpose to
system performance. We formalize a data minimization criterion based on
performance curve derivatives and provide an effective and interpretable
piecewise power law technique that models distinct stages of an algorithm's
performance throughout data collection. Results from our empirical
investigation offer deeper insights into the relevant considerations when
designing a data minimization framework, including the choice of feature
acquisition algorithm, initialization conditions, as well as impacts on
individuals that hint at tensions between data minimization and fairness.

    

### [[2107.08114] Decentralized Multi-Agent Reinforcement Learning for Task Offloading Under Uncertainty](http://arxiv.org/abs/2107.08114)


  Multi-Agent Reinforcement Learning (MARL) is a challenging subarea of
Reinforcement Learning due to the non-stationarity of the environments and the
large dimensionality of the combined action space. Deep MARL algorithms have
been applied to solve different task offloading problems. However, in
real-world applications, information required by the agents (i.e. rewards and
states) are subject to noise and alterations. The stability and the robustness
of deep MARL to practical challenges is still an open research problem. In this
work, we apply state-of-the art MARL algorithms to solve task offloading with
reward uncertainty. We show that perturbations in the reward signal can induce
decrease in the performance compared to learning with perfect rewards. We
expect this paper to stimulate more research in studying and addressing the
practical challenges of deploying deep MARL solutions in wireless
communications systems.

    

### [[2107.08135] Mediated Uncoupled Learning: Learning Functions without Direct Input-output Correspondences](http://arxiv.org/abs/2107.08135)


  Ordinary supervised learning is useful when we have paired training data of
input $X$ and output $Y$. However, such paired data can be difficult to collect
in practice. In this paper, we consider the task of predicting $Y$ from $X$
when we have no paired data of them, but we have two separate, independent
datasets of $X$ and $Y$ each observed with some mediating variable $U$, that
is, we have two datasets $S_X = \{(X_i, U_i)\}$ and $S_Y = \{(U'_j, Y'_j)\}$. A
naive approach is to predict $U$ from $X$ using $S_X$ and then $Y$ from $U$
using $S_Y$, but we show that this is not statistically consistent. Moreover,
predicting $U$ can be more difficult than predicting $Y$ in practice, e.g.,
when $U$ has higher dimensionality. To circumvent the difficulty, we propose a
new method that avoids predicting $U$ but directly learns $Y = f(X)$ by
training $f(X)$ with $S_{X}$ to predict $h(U)$ which is trained with $S_{Y}$ to
approximate $Y$. We prove statistical consistency and error bounds of our
method and experimentally confirm its practical usefulness.

    

### [[2107.08140] Markov Blanket Discovery using Minimum Message Length](http://arxiv.org/abs/2107.08140)


  Causal discovery automates the learning of causal Bayesian networks from data
and has been of active interest from their beginning. With the sourcing of
large data sets off the internet, interest in scaling up to very large data
sets has grown. One approach to this is to parallelize search using Markov
Blanket (MB) discovery as a first step, followed by a process of combining MBs
in a global causal model. We develop and explore three new methods of MB
discovery using Minimum Message Length (MML) and compare them empirically to
the best existing methods, whether developed specifically as MB discovery or as
feature selection. Our best MML method is consistently competitive and has some
advantageous features.

    

### [[2107.08142] Autonomy 2.0: Why is self-driving always 5 years away?](http://arxiv.org/abs/2107.08142)


  Despite the numerous successes of machine learning over the past decade
(image recognition, decision-making, NLP, image synthesis), self-driving
technology has not yet followed the same trend. In this paper, we study the
history, composition, and development bottlenecks of the modern self-driving
stack. We argue that the slow progress is caused by approaches that require too
much hand-engineering, an over-reliance on road testing, and high fleet
deployment costs. We observe that the classical stack has several bottlenecks
that preclude the necessary scale needed to capture the long tail of rare
events. To resolve these problems, we outline the principles of Autonomy 2.0,
an ML-first approach to self-driving, as a viable alternative to the currently
adopted state-of-the-art. This approach is based on (i) a fully differentiable
AV stack trainable from human demonstrations, (ii) closed-loop data-driven
reactive simulation, and (iii) large-scale, low-cost data collections as
critical solutions towards scalability issues. We outline the general
architecture, survey promising works in this direction and propose key
challenges to be addressed by the community in the future.

    

### [[2107.08147] AutoFL: Enabling Heterogeneity-Aware Energy Efficient Federated Learning](http://arxiv.org/abs/2107.08147)


  Federated learning enables a cluster of decentralized mobile devices at the
edge to collaboratively train a shared machine learning model, while keeping
all the raw training samples on device. This decentralized training approach is
demonstrated as a practical solution to mitigate the risk of privacy leakage.
However, enabling efficient FL deployment at the edge is challenging because of
non-IID training data distribution, wide system heterogeneity and
stochastic-varying runtime effects in the field. This paper jointly optimizes
time-to-convergence and energy efficiency of state-of-the-art FL use cases by
taking into account the stochastic nature of edge execution. We propose AutoFL
by tailor-designing a reinforcement learning algorithm that learns and
determines which K participant devices and per-device execution targets for
each FL model aggregation round in the presence of stochastic runtime variance,
system and data heterogeneity. By considering the unique characteristics of FL
edge deployment judiciously, AutoFL achieves 3.6 times faster model convergence
time and 4.7 and 5.2 times higher energy efficiency for local clients and
globally over the cluster of K participants, respectively.

    

### [[2107.08148] Declarative Machine Learning Systems](http://arxiv.org/abs/2107.08148)


  In the last years machine learning (ML) has moved from a academic endeavor to
a pervasive technology adopted in almost every aspect of computing. ML-powered
products are now embedded in our digital lives: from recommendations of what to
watch, to divining our search intent, to powering virtual assistants in
consumer and enterprise settings. Recent successes in applying ML in natural
sciences revealed that ML can be used to tackle some of the hardest real-world
problems humanity faces today. For these reasons ML has become central in the
strategy of tech companies and has gathered even more attention from academia
than ever before. Despite these successes, what we have witnessed so far is
just the beginning. Right now the people training and using ML models are
expert developers working within large organizations, but we believe the next
wave of ML systems will allow a larger amount of people, potentially without
coding skills, to perform the same tasks. These new ML systems will not require
users to fully understand all the details of how models are trained and
utilized for obtaining predictions. Declarative interfaces are well suited for
this goal, by hiding complexity and favouring separation of interests, and can
lead to increased productivity. We worked on such abstract interfaces by
developing two declarative ML systems, Overton and Ludwig, that require users
to declare only their data schema (names and types of inputs) and tasks rather
then writing low level ML code. In this article we will describe how ML systems
are currently structured, highlight important factors for their success and
adoption, what are the issues current ML systems are facing and how the systems
we developed addressed them. Finally we will talk about learnings from the
development of ML systems throughout the years and how we believe the next
generation of ML systems will look like.

    

### [[2107.08170] Megaverse: Simulating Embodied Agents at One Million Experiences per Second](http://arxiv.org/abs/2107.08170)


  We present Megaverse, a new 3D simulation platform for reinforcement learning
and embodied AI research. The efficient design of our engine enables
physics-based simulation with high-dimensional egocentric observations at more
than 1,000,000 actions per second on a single 8-GPU node. Megaverse is up to
70x faster than DeepMind Lab in fully-shaded 3D scenes with interactive
objects. We achieve this high simulation performance by leveraging batched
simulation, thereby taking full advantage of the massive parallelism of modern
GPUs. We use Megaverse to build a new benchmark that consists of several
single-agent and multi-agent tasks covering a variety of cognitive challenges.
We evaluate model-free RL on this benchmark to provide baselines and facilitate
future research. The source code is available at this https URL


### [[2107.08176] Automatic Fairness Testing of Neural Classifiers through Adversarial Sampling](http://arxiv.org/abs/2107.08176)


  Although deep learning has demonstrated astonishing performance in many
applications, there are still concerns on their dependability. One desirable
property of deep learning applications with societal impact is fairness (i.e.,
non-discrimination). Unfortunately, discrimination might be intrinsically
embedded into the models due to discrimination in the training data. As a
countermeasure, fairness testing systemically identifies discriminative
samples, which can be used to retrain the model and improve its fairness.
Existing fairness testing approaches however have two major limitations. First,
they only work well on traditional machine learning models and have poor
performance (e.g., effectiveness and efficiency) on deep learning models.
Second, they only work on simple tabular data and are not applicable for
domains such as text. In this work, we bridge the gap by proposing a scalable
and effective approach for systematically searching for discriminative samples
while extending fairness testing to address a challenging domain, i.e., text
classification. Compared with state-of-the-art methods, our approach only
employs lightweight procedures like gradient computation and clustering, which
makes it significantly more scalable. Experimental results show that on
average, our approach explores the search space more effectively (9.62 and 2.38
times more than the state-of-art methods respectively on tabular and text
datasets) and generates much more individual discriminatory instances (24.95
and 2.68 times) within reasonable time. The retrained models reduce
discrimination by 57.2% and 60.2% respectively on average.

    

### [[2107.08179] Model Uncertainty and Correctability for Directed Graphical Models](http://arxiv.org/abs/2107.08179)


  Probabilistic graphical models are a fundamental tool in probabilistic
modeling, machine learning and artificial intelligence. They allow us to
integrate in a natural way expert knowledge, physical modeling, heterogeneous
and correlated data and quantities of interest. For exactly this reason,
multiple sources of model uncertainty are inherent within the modular structure
of the graphical model. In this paper we develop information-theoretic, robust
uncertainty quantification methods and non-parametric stress tests for directed
graphical models to assess the effect and the propagation through the graph of
multi-sourced model uncertainties to quantities of interest. These methods
allow us to rank the different sources of uncertainty and correct the graphical
model by targeting its most impactful components with respect to the quantities
of interest. Thus, from a machine learning perspective, we provide a
mathematically rigorous approach to correctability that guarantees a systematic
selection for improvement of components of a graphical model while controlling
potential new errors created in the process in other parts of the model. We
demonstrate our methods in two physico-chemical examples, namely quantum
scale-informed chemical kinetics and materials screening to improve the
efficiency of fuel cells.

    

### [[2107.08183] Hierarchical Reinforcement Learning with Optimal Level Synchronization based on a Deep Generative Model](http://arxiv.org/abs/2107.08183)


  The high-dimensional or sparse reward task of a reinforcement learning (RL)
environment requires a superior potential controller such as hierarchical
reinforcement learning (HRL) rather than an atomic RL because it absorbs the
complexity of commands to achieve the purpose of the task in its hierarchical
structure. One of the HRL issues is how to train each level policy with the
optimal data collection from its experience. That is to say, how to synchronize
adjacent level policies optimally. Our research finds that a HRL model through
the off-policy correction technique of HRL, which trains a higher-level policy
with the goal of reflecting a lower-level policy which is newly trained using
the off-policy method, takes the critical role of synchronizing both level
policies at all times while they are being trained. We propose a novel HRL
model supporting the optimal level synchronization using the off-policy
correction technique with a deep generative model. This uses the advantage of
the inverse operation of a flow-based deep generative model (FDGM) to achieve
the goal corresponding to the current state of the lower-level policy. The
proposed model also considers the freedom of the goal dimension between HRL
policies which makes it the generalized inverse model of the model-free RL in
HRL with the optimal synchronization method. The comparative experiment results
show the performance of our proposed model.

    

### [[2107.08189] BEDS-Bench: Behavior of EHR-models under Distributional Shift--A Benchmark](http://arxiv.org/abs/2107.08189)


  Machine learning has recently demonstrated impressive progress in predictive
accuracy across a wide array of tasks. Most ML approaches focus on
generalization performance on unseen data that are similar to the training data
(In-Distribution, or IND). However, real world applications and deployments of
ML rarely enjoy the comfort of encountering examples that are always IND. In
such situations, most ML models commonly display erratic behavior on
Out-of-Distribution (OOD) examples, such as assigning high confidence to wrong
predictions, or vice-versa. Implications of such unusual model behavior are
further exacerbated in the healthcare setting, where patient health can
potentially be put at risk. It is crucial to study the behavior and robustness
properties of models under distributional shift, understand common failure
modes, and take mitigation steps before the model is deployed. Having a
benchmark that shines light upon these aspects of a model is a first and
necessary step in addressing the issue. Recent work and interest in increasing
model robustness in OOD settings have focused more on image modality, while the
Electronic Health Record (EHR) modality is still largely under-explored. We aim
to bridge this gap by releasing BEDS-Bench, a benchmark for quantifying the
behavior of ML models over EHR data under OOD settings. We use two open access,
de-identified EHR datasets to construct several OOD data settings to run tests
on, and measure relevant metrics that characterize crucial aspects of a model's
OOD behavior. We evaluate several learning algorithms under BEDS-Bench and find
that all of them show poor generalization performance under distributional
shift in general. Our results highlight the need and the potential to improve
robustness of EHR models under distributional shift, and BEDS-Bench provides
one way to measure progress towards that goal.

    

### [[2107.08190] COVID-19 Multidimensional Kaggle Literature Organization](http://arxiv.org/abs/2107.08190)


  The unprecedented outbreak of Severe Acute Respiratory Syndrome Coronavirus-2
(SARS-CoV-2), or COVID-19, continues to be a significant worldwide problem. As
a result, a surge of new COVID-19 related research has followed suit. The
growing number of publications requires document organization methods to
identify relevant information. In this paper, we expand upon our previous work
with clustering the CORD-19 dataset by applying multi-dimensional analysis
methods. Tensor factorization is a powerful unsupervised learning method
capable of discovering hidden patterns in a document corpus. We show that a
higher-order representation of the corpus allows for the simultaneous grouping
of similar articles, relevant journals, authors with similar research
interests, and topic keywords. These groupings are identified within and among
the latent components extracted via tensor decomposition. We further
demonstrate the application of this method with a publicly available
interactive visualization of the dataset.

    

### [[2107.08195] Sparse Bayesian Learning with Diagonal Quasi-Newton Method For Large Scale Classification](http://arxiv.org/abs/2107.08195)


  Sparse Bayesian Learning (SBL) constructs an extremely sparse probabilistic
model with very competitive generalization. However, SBL needs to invert a big
covariance matrix with complexity O(M^3 ) (M: feature size) for updating the
regularization priors, making it difficult for practical use. There are three
issues in SBL: 1) Inverting the covariance matrix may obtain singular solutions
in some cases, which hinders SBL from convergence; 2) Poor scalability to
problems with high dimensional feature space or large data size; 3) SBL easily
suffers from memory overflow for large-scale data. This paper addresses these
issues with a newly proposed diagonal Quasi-Newton (DQN) method for SBL called
DQN-SBL where the inversion of big covariance matrix is ignored so that the
complexity and memory storage are reduced to O(M). The DQN-SBL is thoroughly
evaluated on non-linear classifiers and linear feature selection using various
benchmark datasets of different sizes. Experimental results verify that DQN-SBL
receives competitive generalization with a very sparse model and scales well to
large-scale problems.

    

### [[2107.08199] Dynamic Transformer for Efficient Machine Translation on Embedded Devices](http://arxiv.org/abs/2107.08199)


  The Transformer architecture is widely used for machine translation tasks.
However, its resource-intensive nature makes it challenging to implement on
constrained embedded devices, particularly where available hardware resources
can vary at run-time. We propose a dynamic machine translation model that
scales the Transformer architecture based on the available resources at any
particular time. The proposed approach, 'Dynamic-HAT', uses a HAT
SuperTransformer as the backbone to search for SubTransformers with different
accuracy-latency trade-offs at design time. The optimal SubTransformers are
sampled from the SuperTransformer at run-time, depending on latency
constraints. The Dynamic-HAT is tested on the Jetson Nano and the approach uses
inherited SubTransformers sampled directly from the SuperTransformer with a
switching time of <1s. Using inherited SubTransformers results in a BLEU score
loss of <1.5% because the SubTransformer configuration is not retrained from
scratch after sampling. However, to recover this loss in performance, the
dimensions of the design space can be reduced to tailor it to a family of
target hardware. The new reduced design space results in a BLEU score increase
of approximately 1% for sub-optimal models from the original design space, with
a wide range for performance scaling between 0.356s - 1.526s for the GPU and
2.9s - 7.31s for the CPU.

    

### [[2107.08209] Minimising quantifier variance under prior probability shift](http://arxiv.org/abs/2107.08209)


  For the binary prevalence quantification problem under prior probability
shift, we determine the asymptotic variance of the maximum likelihood
estimator. We find that it is a function of the Brier score for the regression
of the class label against the features under the test data set distribution.
This observation suggests that optimising the accuracy of a base classifier on
the training data set helps to reduce the variance of the related quantifier on
the test data set. Therefore, we also point out training criteria for the base
classifier that imply optimisation of both of the Brier scores on the training
and the test data sets.

    

### [[2107.08211] Self Training with Ensemble of Teacher Models](http://arxiv.org/abs/2107.08211)


  In order to train robust deep learning models, large amounts of labelled data
is required. However, in the absence of such large repositories of labelled
data, unlabeled data can be exploited for the same. Semi-Supervised learning
aims to utilize such unlabeled data for training classification models. Recent
progress of self-training based approaches have shown promise in this area,
which leads to this study where we utilize an ensemble approach for the same. A
by-product of any semi-supervised approach may be loss of calibration of the
trained model especially in scenarios where unlabeled data may contain
out-of-distribution samples, which leads to this investigation on how to adapt
to such effects. Our proposed algorithm carefully avoids common pitfalls in
utilizing unlabeled data and leads to a more accurate and calibrated supervised
model compared to vanilla self-training based student-teacher algorithms. We
perform several experiments on the popular STL-10 database followed by an
extensive analysis of our approach and study its effects on model accuracy and
calibration.

    

### [[2107.08212] On the Copying Behaviors of Pre-Training for Neural Machine Translation](http://arxiv.org/abs/2107.08212)


  Previous studies have shown that initializing neural machine translation
(NMT) models with the pre-trained language models (LM) can speed up the model
training and boost the model performance. In this work, we identify a critical
side-effect of pre-training for NMT, which is due to the discrepancy between
the training objectives of LM-based pre-training and NMT. Since the LM
objective learns to reconstruct a few source tokens and copy most of them, the
pre-training initialization would affect the copying behaviors of NMT models.
We provide a quantitative analysis of copying behaviors by introducing a metric
called copying ratio, which empirically shows that pre-training based NMT
models have a larger copying ratio than the standard one. In response to this
problem, we propose a simple and effective method named copying penalty to
control the copying behaviors in decoding. Extensive experiments on both
in-domain and out-of-domain benchmarks show that the copying penalty method
consistently improves translation performance by controlling copying behaviors
for pre-training based NMT models. Source code is freely available at
this https URL.

    

### [[2107.08221] Visual Representation Learning Does Not Generalize Strongly Within the Same Domain](http://arxiv.org/abs/2107.08221)


  An important component for generalization in machine learning is to uncover
underlying latent factors of variation as well as the mechanism through which
each factor acts in the world. In this paper, we test whether 17 unsupervised,
weakly supervised, and fully supervised representation learning approaches
correctly infer the generative factors of variation in simple datasets
(dSprites, Shapes3D, MPI3D). In contrast to prior robustness work that
introduces novel factors of variation during test time, such as blur or other
(un)structured noise, we here recompose, interpolate, or extrapolate only
existing factors of variation from the training data set (e.g., small and
medium-sized objects during training and large objects during testing). Models
that learn the correct mechanism should be able to generalize to this
benchmark. In total, we train and test 2000+ models and observe that all of
them struggle to learn the underlying mechanism regardless of supervision
signal and architectural bias. Moreover, the generalization capabilities of all
tested models drop significantly as we move from artificial datasets towards
more realistic real-world datasets. Despite their inability to identify the
correct mechanism, the models are quite modular as their ability to infer other
in-distribution factors remains fairly stable, providing only a single factor
is out-of-distribution. These results point to an important yet understudied
problem of learning mechanistic models of observations that can facilitate
generalization.

    

### [[2107.08225] On Constraints in First-Order Optimization: A View from Non-Smooth Dynamical Systems](http://arxiv.org/abs/2107.08225)


  We introduce a class of first-order methods for smooth constrained
optimization that are based on an analogy to non-smooth dynamical systems. Two
distinctive features of our approach are that (i) projections or optimizations
over the entire feasible set are avoided, in stark contrast to projected
gradient methods or the Frank-Wolfe method, and (ii) iterates are allowed to
become infeasible, which differs from active set or feasible direction methods,
where the descent motion stops as soon as a new constraint is encountered. The
resulting algorithmic procedure is simple to implement even when constraints
are nonlinear, and is suitable for large-scale constrained optimization
problems in which the feasible set fails to have a simple structure. The key
underlying idea is that constraints are expressed in terms of velocities
instead of positions, which has the algorithmic consequence that optimizations
over feasible sets at each iteration are replaced with optimizations over
local, sparse convex approximations. The result is a simplified suite of
algorithms and an expanded range of possible applications in machine learning.

    

### [[2107.08241] High-Accuracy Model-Based Reinforcement Learning, a Survey](http://arxiv.org/abs/2107.08241)


  Deep reinforcement learning has shown remarkable success in the past few
years. Highly complex sequential decision making problems from game playing and
robotics have been solved with deep model-free methods. Unfortunately, the
sample complexity of model-free methods is often high. To reduce the number of
environment samples, model-based reinforcement learning creates an explicit
model of the environment dynamics. Achieving high model accuracy is a challenge
in high-dimensional problems. In recent years, a diverse landscape of
model-based methods has been introduced to improve model accuracy, using
methods such as uncertainty modeling, model-predictive control, latent models,
and end-to-end learning and planning. Some of these methods succeed in
achieving high accuracy at low sample complexity, most do so either in a
robotics or in a games context. In this paper, we survey these methods; we
explain in detail how they work and what their strengths and weaknesses are. We
conclude with a research agenda for future work to make the methods more robust
and more widely applicable to other applications.

    

### [[2107.08248] Learning De-identified Representations of Prosody from Raw Audio](http://arxiv.org/abs/2107.08248)


  We propose a method for learning de-identified prosody representations from
raw audio using a contrastive self-supervised signal. Whereas prior work has
relied on conditioning models on bottlenecks, we introduce a set of inductive
biases that exploit the natural structure of prosody to minimize timbral
information and decouple prosody from speaker representations. Despite
aggressive downsampling of the input and having no access to linguistic
information, our model performs comparably to state-of-the-art speech
representations on DAMMP, a new benchmark we introduce for spoken language
understanding. We use minimum description length probing to show that our
representations have selectively learned the subcomponents of non-timbral
prosody, and that the product quantizer naturally disentangles them without
using bottlenecks. We derive an information-theoretic definition of speech
de-identifiability and use it to demonstrate that our prosody representations
are less identifiable than other speech representations.

    

### [[2107.08249] The Effects of Learning in Morphologically Evolving Robot Systems](http://arxiv.org/abs/2107.08249)


  When controllers (brains) and morphologies (bodies) of robots simultaneously
evolve, this can lead to a problem, namely the brain & body mismatch problem.
In this research, we propose a solution of lifetime learning. We set up a
system where modular robots can create offspring that inherit the bodies of
parents by recombination and mutation. With regards to the brains of the
offspring, we use two methods to create them. The first one entails solely
evolution which means the brain of a robot child is inherited from its parents.
The second approach is evolution plus learning which means the brain of a child
is inherited as well, but additionally is developed by a learning algorithm -
RevDEknn. We compare these two methods by running experiments in a simulator
called Revolve and use efficiency, efficacy, and the morphology intelligence of
the robots for the comparison. The experiments show that the evolution plus
learning method does not only lead to a higher fitness level, but also to more
morphologically evolving robots. This constitutes a quantitative demonstration
that changes in the brain can induce changes in the body, leading to the
concept of morphological intelligence, which is quantified by the learning
delta, meaning the ability of a morphology to facilitate learning.

    

### [[2107.08251] Generative Pretraining for Paraphrase Evaluation](http://arxiv.org/abs/2107.08251)


  We introduce ParaBLEU, a paraphrase representation learning model and
evaluation metric for text generation. Unlike previous approaches, ParaBLEU
learns to understand paraphrasis using generative conditioning as a pretraining
objective. ParaBLEU correlates more strongly with human judgements than
existing metrics, obtaining new state-of-the-art results on the 2017 WMT
Metrics Shared Task. We show that our model is robust to data scarcity,
exceeding previous state-of-the-art performance using only $50\%$ of the
available training data and surpassing BLEU, ROUGE and METEOR with only $40$
labelled examples. Finally, we demonstrate that ParaBLEU can be used to
conditionally generate novel paraphrases from a single demonstration, which we
use to confirm our hypothesis that it learns abstract, generalized paraphrase
representations.

    

### [[2107.08264] M2Lens: Visualizing and Explaining Multimodal Models for Sentiment Analysis](http://arxiv.org/abs/2107.08264)


  Multimodal sentiment analysis aims to recognize people's attitudes from
multiple communication channels such as verbal content (i.e., text), voice, and
facial expressions. It has become a vibrant and important research topic in
natural language processing. Much research focuses on modeling the complex
intra- and inter-modal interactions between different communication channels.
However, current multimodal models with strong performance are often
deep-learning-based techniques and work like black boxes. It is not clear how
models utilize multimodal information for sentiment predictions. Despite recent
advances in techniques for enhancing the explainability of machine learning
models, they often target unimodal scenarios (e.g., images, sentences), and
little research has been done on explaining multimodal models. In this paper,
we present an interactive visual analytics system, M2Lens, to visualize and
explain multimodal models for sentiment analysis. M2Lens provides explanations
on intra- and inter-modal interactions at the global, subset, and local levels.
Specifically, it summarizes the influence of three typical interaction types
(i.e., dominance, complement, and conflict) on the model predictions. Moreover,
M2Lens identifies frequent and influential multimodal features and supports the
multi-faceted exploration of model behaviors from language, acoustic, and
visual modalities. Through two case studies and expert interviews, we
demonstrate our system can help users gain deep insights into the multimodal
models for sentiment analysis.

    

### [[2107.08265] Subset-of-Data Variational Inference for Deep Gaussian-Processes Regression](http://arxiv.org/abs/2107.08265)


  Deep Gaussian Processes (DGPs) are multi-layer, flexible extensions of
Gaussian processes but their training remains challenging. Sparse
approximations simplify the training but often require optimization over a
large number of inducing inputs and their locations across layers. In this
paper, we simplify the training by setting the locations to a fixed subset of
data and sampling the inducing inputs from a variational distribution. This
reduces the trainable parameters and computation cost without significant
performance degradations, as demonstrated by our empirical results on
regression problems. Our modifications simplify and stabilize DGP training
while making it amenable to sampling schemes for setting the inducing inputs.

    

### [[2107.08273] STRODE: Stochastic Boundary Ordinary Differential Equation](http://arxiv.org/abs/2107.08273)


  Perception of time from sequentially acquired sensory inputs is rooted in
everyday behaviors of individual organisms. Yet, most algorithms for
time-series modeling fail to learn dynamics of random event timings directly
from visual or audio inputs, requiring timing annotations during training that
are usually unavailable for real-world applications. For instance, neuroscience
perspectives on postdiction imply that there exist variable temporal ranges
within which the incoming sensory inputs can affect the earlier perception, but
such temporal ranges are mostly unannotated for real applications such as
automatic speech recognition (ASR). In this paper, we present a probabilistic
ordinary differential equation (ODE), called STochastic boundaRy ODE (STRODE),
that learns both the timings and the dynamics of time series data without
requiring any timing annotations during training. STRODE allows the usage of
differential equations to sample from the posterior point processes,
efficiently and analytically. We further provide theoretical guarantees on the
learning of STRODE. Our empirical results show that our approach successfully
infers event timings of time series data. Our method achieves competitive or
superior performances compared to existing state-of-the-art methods for both
synthetic and real-world datasets.

    

### [[2107.08277] Learning Augmented Online Facility Location](http://arxiv.org/abs/2107.08277)


  Following the research agenda initiated by Munoz & Vassilvitskii [1] and
Lykouris & Vassilvitskii [2] on learning-augmented online algorithms for
classical online optimization problems, in this work, we consider the Online
Facility Location problem under this framework. In Online Facility Location
(OFL), demands arrive one-by-one in a metric space and must be (irrevocably)
assigned to an open facility upon arrival, without any knowledge about future
demands.
We present an online algorithm for OFL that exploits potentially imperfect
predictions on the locations of the optimal facilities. We prove that the
competitive ratio decreases smoothly from sublogarithmic in the number of
demands to constant, as the error, i.e., the total distance of the predicted
locations to the optimal facility locations, decreases towards zero. We
complement our analysis with a matching lower bound establishing that the
dependence of the algorithm's competitive ratio on the error is optimal, up to
constant factors. Finally, we evaluate our algorithm on real world data and
compare our learning augmented approach with the current best online algorithm
for the problem.

    

### [[2107.08285] Greedification Operators for Policy Optimization: Investigating Forward and Reverse KL Divergences](http://arxiv.org/abs/2107.08285)


  Approximate Policy Iteration (API) algorithms alternate between (approximate)
policy evaluation and (approximate) greedification. Many different approaches
have been explored for approximate policy evaluation, but less is understood
about approximate greedification and what choices guarantee policy improvement.
In this work, we investigate approximate greedification when reducing the KL
divergence between the parameterized policy and the Boltzmann distribution over
action values. In particular, we investigate the difference between the forward
and reverse KL divergences, with varying degrees of entropy regularization. We
show that the reverse KL has stronger policy improvement guarantees, but that
reducing the forward KL can result in a worse policy. We also demonstrate,
however, that a large enough reduction of the forward KL can induce improvement
under additional assumptions. Empirically, we show on simple continuous-action
environments that the forward KL can induce more exploration, but at the cost
of a more suboptimal policy. No significant differences were observed in the
discrete-action setting or on a suite of benchmark problems. Throughout, we
highlight that many policy gradient methods can be seen as an instance of API,
with either the forward or reverse KL for the policy update, and discuss next
steps for understanding and improving our policy optimization algorithms.

    

### [[2107.08293] On the Robustness of Deep Reinforcement Learning in IRS-Aided Wireless Communications Systems](http://arxiv.org/abs/2107.08293)


  We consider an Intelligent Reflecting Surface (IRS)-aided multiple-input
single-output (MISO) system for downlink transmission. We compare the
performance of Deep Reinforcement Learning (DRL) and conventional optimization
methods in finding optimal phase shifts of the IRS elements to maximize the
user signal-to-noise (SNR) ratio. Furthermore, we evaluate the robustness of
these methods to channel impairments and changes in the system. We demonstrate
numerically that DRL solutions show more robustness to noisy channels and user
mobility.

    

### [[2107.08310] Fair Balance: Mitigating Machine Learning Bias Against Multiple Protected Attributes With Data Balancing](http://arxiv.org/abs/2107.08310)


  This paper aims to improve machine learning fairness on multiple protected
at-tributes. Machine learning fairness has attracted increasing attention since
machine learning models are increasingly used for high-stakes and high-risk
decisions. Most existing solutions for machine learning fairness only target
one protected attribute(e.g. sex) at a time. These solutions cannot generate a
machine learning model which is fair against every protected attribute (e.g.
both sex and race) at the same time. To solve this problem, we propose
FairBalance in this paper to balance the distribution of training data across
every protected attribute before training the machine learning models. Our
results show that, under the assumption of unbiased ground truth labels,
FairBalance can significantly reduce bias metrics (AOD, EOD, and SPD) on every
known protected attribute without much, if not any damage to the prediction
performance.

    

### [[2107.08319] Characterizing Online Engagement with Disinformation and Conspiracies in the 2020 U.S. Presidential Election](http://arxiv.org/abs/2107.08319)


  Identifying and characterizing disinformation in political discourse on
social media is critical to ensure the integrity of elections and democratic
processes around the world. Persistent manipulation of social media has
resulted in increased concerns regarding the 2020 U.S. Presidential Election,
due to its potential to influence individual opinions and social dynamics. In
this work, we focus on the identification of distorted facts, in the form of
unreliable and conspiratorial narratives in election-related tweets, to
characterize discourse manipulation prior to the election. We apply a detection
model to separate factual from unreliable (or conspiratorial) claims analyzing
a dataset of 242 million election-related tweets. The identified claims are
used to investigate targeted topics of disinformation, and conspiracy groups,
most notably the far-right QAnon conspiracy group. Further, we characterize
account engagements with unreliable and conspiracy tweets, and with the QAnon
conspiracy group, by political leaning and tweet types. Finally, using a
regression discontinuity design, we investigate whether Twitter's actions to
curb QAnon activity on the platform were effective, and how QAnon accounts
adapt to Twitter's restrictions.

    

### [[2107.08326] Otimizacao de Redes Neurais atraves de Algoritmos Geneticos Celulares](http://arxiv.org/abs/2107.08326)


  This works proposes a methodology to searching for automatically Artificial
Neural Networks (ANN) by using Cellular Genetic Algorithm (CGA). The goal of
this methodology is to find compact networks whit good performance for
classification problems. The main reason for developing this work is centered
at the difficulties of configuring compact ANNs with good performance rating.
The use of CGAs aims at seeking the components of the RNA in the same way that
a common Genetic Algorithm (GA), but it has the differential of incorporating a
Cellular Automaton (CA) to give location for the GA individuals. The location
imposed by the CA aims to control the spread of solutions in the populations to
maintain the genetic diversity for longer time. This genetic diversity is
important for obtain good results with the GAs.

    

### [[2107.08346] Policy Optimization in Adversarial MDPs: Improved Exploration via Dilated Bonuses](http://arxiv.org/abs/2107.08346)


  Policy optimization is a widely-used method in reinforcement learning. Due to
its local-search nature, however, theoretical guarantees on global optimality
often rely on extra assumptions on the Markov Decision Processes (MDPs) that
bypass the challenge of global exploration. To eliminate the need of such
assumptions, in this work, we develop a general solution that adds dilated
bonuses to the policy update to facilitate global exploration. To showcase the
power and generality of this technique, we apply it to several episodic MDP
settings with adversarial losses and bandit feedback, improving and
generalizing the state-of-the-art. Specifically, in the tabular case, we obtain
$\widetilde{\mathcal{O}}(\sqrt{T})$ regret where $T$ is the number of episodes,
improving the $\widetilde{\mathcal{O}}({T}^{2/3})$ regret bound by Shani et al.
(2020). When the number of states is infinite, under the assumption that the
state-action values are linear in some low-dimensional features, we obtain
$\widetilde{\mathcal{O}}({T}^{2/3})$ regret with the help of a simulator,
matching the result of Neu and Olkhovskaya (2020) while importantly removing
the need of an exploratory policy that their algorithm requires. When a
simulator is unavailable, we further consider a linear MDP setting and obtain
$\widetilde{\mathcal{O}}({T}^{14/15})$ regret, which is the first result for
linear MDPs with adversarial losses and bandit feedback.

    

### [[2107.08353] Top-label calibration](http://arxiv.org/abs/2107.08353)


  We study the problem of post-hoc calibration for multiclass classification,
with an emphasis on histogram binning. Multiple works have focused on
calibration with respect to the confidence of just the predicted class (or
'top-label'). We find that the popular notion of confidence calibration [Guo et
al., 2017] is not sufficiently strong -- there exist predictors that are not
calibrated in any meaningful way but are perfectly confidence calibrated. We
propose a closely related (but subtly different) notion, top-label calibration,
that accurately captures the intuition and simplicity of confidence
calibration, but addresses its drawbacks. We formalize a histogram binning (HB)
algorithm that reduces top-label multiclass calibration to the binary case,
prove that it has clean theoretical guarantees without distributional
assumptions, and perform a methodical study of its practical performance. Some
prediction tasks require stricter notions of multiclass calibration such as
class-wise or canonical calibration. We formalize appropriate HB algorithms
corresponding to each of these goals. In experiments with deep neural nets, we
find that our principled versions of HB are often better than temperature
scaling, for both top-label and class-wise calibration. Code for this work will
be made publicly available at this https URL.

    

### [[2107.08356] DeHumor: Visual Analytics for Decomposing Humor](http://arxiv.org/abs/2107.08356)


  Despite being a critical communication skill, grasping humor is challenging
-- a successful use of humor requires a mixture of both engaging content
build-up and an appropriate vocal delivery (e.g., pause). Prior studies on
computational humor emphasize the textual and audio features immediately next
to the punchline, yet overlooking longer-term context setup. Moreover, the
theories are usually too abstract for understanding each concrete humor
snippet. To fill in the gap, we develop DeHumor, a visual analytical system for
analyzing humorous behaviors in public speaking. To intuitively reveal the
building blocks of each concrete example, DeHumor decomposes each humorous
video into multimodal features and provides inline annotations of them on the
video script. In particular, to better capture the build-ups, we introduce
content repetition as a complement to features introduced in theories of
computational humor and visualize them in a context linking graph. To help
users locate the punchlines that have the desired features to learn, we
summarize the content (with keywords) and humor feature statistics on an
augmented time matrix. With case studies on stand-up comedy shows and TED
talks, we show that DeHumor is able to highlight various building blocks of
humor examples. In addition, expert interviews with communication coaches and
humor researchers demonstrate the effectiveness of DeHumor for multimodal humor
analysis of speech content and vocal delivery.

    

### [[2107.08362] Probabilistic Verification of Neural Networks Against Group Fairness](http://arxiv.org/abs/2107.08362)


  Fairness is crucial for neural networks which are used in applications with
important societal implication. Recently, there have been multiple attempts on
improving fairness of neural networks, with a focus on fairness testing (e.g.,
generating individual discriminatory instances) and fairness training (e.g.,
enhancing fairness through augmented training). In this work, we propose an
approach to formally verify neural networks against fairness, with a focus on
independence-based fairness such as group fairness. Our method is built upon an
approach for learning Markov Chains from a user-provided neural network (i.e.,
a feed-forward neural network or a recurrent neural network) which is
guaranteed to facilitate sound analysis. The learned Markov Chain not only
allows us to verify (with Probably Approximate Correctness guarantee) whether
the neural network is fair or not, but also facilities sensitivity analysis
which helps to understand why fairness is violated. We demonstrate that with
our analysis results, the neural weights can be optimized to improve fairness.
Our approach has been evaluated with multiple models trained on benchmark
datasets and the experiment results show that our approach is effective and
efficient.

    

### [[2107.08364] A Survey on Data-driven Software Vulnerability Assessment and Prioritization](http://arxiv.org/abs/2107.08364)


  Software Vulnerabilities (SVs) are increasing in complexity and scale, posing
great security risks to many software systems. Given the limited resources in
practice, SV assessment and prioritization help practitioners devise optimal SV
mitigation plans based on various SV characteristics. The surge in SV data
sources and data-driven techniques such as Machine Learning and Deep Learning
have taken SV assessment and prioritization to the next level. Our survey
provides a taxonomy of the past research efforts and highlights the best
practices for data-driven SV assessment and prioritization. We also discuss the
current limitations and propose potential solutions to address such issues.

    

### [[2107.08369] Flood Segmentation on Sentinel-1 SAR Imagery with Semi-Supervised Learning](http://arxiv.org/abs/2107.08369)


  Floods wreak havoc throughout the world, causing billions of dollars in
damages, and uprooting communities, ecosystems and economies. Accurate and
robust flood detection including delineating open water flood areas and
identifying flood levels can aid in disaster response and mitigation. However,
estimating flood levels remotely is of essence as physical access to flooded
areas is limited and the ability to deploy instruments in potential flood zones
can be dangerous. Aligning flood extent mapping with local topography can
provide a plan-of-action that the disaster response team can consider. Thus,
remote flood level estimation via satellites like Sentinel-1 can prove to be
remedial. The Emerging Techniques in Computational Intelligence (ETCI)
competition on Flood Detection tasked participants with predicting flooded
pixels after training with synthetic aperture radar (SAR) images in a
supervised setting. We use a cyclical approach involving two stages (1)
training an ensemble model of multiple UNet architectures with available high
and low confidence labeled data and, (2) generating pseudo labels or low
confidence labels on the unlabeled test dataset, and then, combining the
generated labels with the previously available high confidence labeled dataset.
This assimilated dataset is used for the next round of training ensemble
models. This cyclical process is repeated until the performance improvement
plateaus. Additionally, we post process our results with Conditional Random
Fields. Our approach sets a high score on the public leaderboard for the ETCI
competition with 0.7654 IoU. Our method, which we release with all the code
including trained models, can also be used as an open science benchmark for the
Sentinel-1 released dataset on GitHub.

    

### [[2107.08371] An Experimental Study of Data Heterogeneity in Federated Learning Methods for Medical Imaging](http://arxiv.org/abs/2107.08371)


  Federated learning enables multiple institutions to collaboratively train
machine learning models on their local data in a privacy-preserving way.
However, its distributed nature often leads to significant heterogeneity in
data distributions across institutions. In this paper, we investigate the
deleterious impact of a taxonomy of data heterogeneity regimes on federated
learning methods, including quantity skew, label distribution skew, and imaging
acquisition skew. We show that the performance degrades with the increasing
degrees of data heterogeneity. We present several mitigation strategies to
overcome performance drops from data heterogeneity, including weighted average
for data quantity skew, weighted loss and batch normalization averaging for
label distribution skew. The proposed optimizations to federated learning
methods improve their capability of handling heterogeneity across institutions,
which provides valuable guidance for the deployment of federated learning in
real clinical applications.

    

### [[2107.08377] A New Adaptive Gradient Method with Gradient Decomposition](http://arxiv.org/abs/2107.08377)


  Adaptive gradient methods, especially Adam-type methods (such as Adam,
AMSGrad, and AdaBound), have been proposed to speed up the training process
with an element-wise scaling term on learning rates. However, they often
generalize poorly compared with stochastic gradient descent (SGD) and its
accelerated schemes such as SGD with momentum (SGDM). In this paper, we propose
a new adaptive method called DecGD, which simultaneously achieves good
generalization like SGDM and obtain rapid convergence like Adam-type methods.
In particular, DecGD decomposes the current gradient into the product of two
terms including a surrogate gradient and a loss based vector. Our method
adjusts the learning rates adaptively according to the current loss based
vector instead of the squared gradients used in Adam-type methods. The
intuition for adaptive learning rates of DecGD is that a good optimizer, in
general cases, needs to decrease the learning rates as the loss decreases,
which is similar to the learning rates decay scheduling technique. Therefore,
DecGD gets a rapid convergence in the early phases of training and controls the
effective learning rates according to the loss based vectors which help lead to
a better generalization. Convergence analysis is discussed in both convex and
non-convex situations. Finally, empirical results on widely-used tasks and
models demonstrate that DecGD shows better generalization performance than SGDM
and rapid convergence like Adam-type methods.

    

### [[2107.08382] A High-Performance Adaptive Quantization Approach for Edge CNN Applications](http://arxiv.org/abs/2107.08382)


  Recent convolutional neural network (CNN) development continues to advance
the state-of-the-art model accuracy for various applications. However, the
enhanced accuracy comes at the cost of substantial memory bandwidth and storage
requirements and demanding computational resources. Although in the past the
quantization methods have effectively reduced the deployment cost for edge
devices, it suffers from significant information loss when processing the
biased activations of contemporary CNNs. In this paper, we hence introduce an
adaptive high-performance quantization method to resolve the issue of biased
activation by dynamically adjusting the scaling and shifting factors based on
the task loss. Our proposed method has been extensively evaluated on image
classification models (ResNet-18/34/50, MobileNet-V2, EfficientNet-B0) with
ImageNet dataset, object detection model (YOLO-V4) with COCO dataset, and
language models with PTB dataset. The results show that our 4-bit integer
(INT4) quantization models achieve better accuracy than the state-of-the-art
4-bit models, and in some cases, even surpass the golden full-precision models.
The final designs have been successfully deployed onto extremely
resource-constrained edge devices for many practical applications.

    

### [[2107.08383] GuideBoot: Guided Bootstrap for Deep Contextual Bandits](http://arxiv.org/abs/2107.08383)


  The exploration/exploitation (E&E) dilemma lies at the core of interactive
systems such as online advertising, for which contextual bandit algorithms have
been proposed. Bayesian approaches provide guided exploration with principled
uncertainty estimation, but the applicability is often limited due to
over-simplified assumptions. Non-Bayesian bootstrap methods, on the other hand,
can apply to complex problems by using deep reward models, but lacks clear
guidance to the exploration behavior. It still remains largely unsolved to
develop a practical method for complex deep contextual bandits.
In this paper, we introduce Guided Bootstrap (GuideBoot for short), combining
the best of both worlds. GuideBoot provides explicit guidance to the
exploration behavior by training multiple models over both real samples and
noisy samples with fake labels, where the noise is added according to the
predictive uncertainty. The proposed method is efficient as it can make
decisions on-the-fly by utilizing only one randomly chosen model, but is also
effective as we show that it can be viewed as a non-Bayesian approximation of
Thompson sampling. Moreover, we extend it to an online version that can learn
solely from streaming data, which is favored in real applications. Extensive
experiments on both synthetic task and large-scale advertising environments
show that GuideBoot achieves significant improvements against previous
state-of-the-art methods.

    

### [[2107.08387] Train on Small, Play the Large: Scaling Up Board Games with AlphaZero and GNN](http://arxiv.org/abs/2107.08387)


  Playing board games is considered a major challenge for both humans and AI
researchers. Because some complicated board games are quite hard to learn,
humans usually begin with playing on smaller boards and incrementally advance
to master larger board strategies. Most neural network frameworks that are
currently tasked with playing board games neither perform such incremental
learning nor possess capabilities to automatically scale up. In this work, we
look at the board as a graph and combine a graph neural network architecture
inside the AlphaZero framework, along with some other innovative improvements.
Our ScalableAlphaZero is capable of learning to play incrementally on small
boards, and advancing to play on large ones. Our model can be trained quickly
to play different challenging board games on multiple board sizes, without
using any domain knowledge. We demonstrate the effectiveness of
ScalableAlphaZero and show, for example, that by training it for only three
days on small Othello boards, it can defeat the AlphaZero model on a large
board, which was trained to play the large board for $30$ days.

    

### [[2107.08396] GraphGen-Redux: a Fast and Lightweight Recurrent Model for labeled Graph Generation](http://arxiv.org/abs/2107.08396)


  The problem of labeled graph generation is gaining attention in the Deep
Learning community. The task is challenging due to the sparse and discrete
nature of graph spaces. Several approaches have been proposed in the
literature, most of which require to transform the graphs into sequences that
encode their structure and labels and to learn the distribution of such
sequences through an auto-regressive generative model. Among this family of
approaches, we focus on the GraphGen model. The preprocessing phase of GraphGen
transforms graphs into unique edge sequences called Depth-First Search (DFS)
codes, such that two isomorphic graphs are assigned the same DFS code. Each
element of a DFS code is associated with a graph edge: specifically, it is a
quintuple comprising one node identifier for each of the two endpoints, their
node labels, and the edge label. GraphGen learns to generate such sequences
auto-regressively and models the probability of each component of the quintuple
independently. While effective, the independence assumption made by the model
is too loose to capture the complex label dependencies of real-world graphs
precisely. By introducing a novel graph preprocessing approach, we are able to
process the labeling information of both nodes and edges jointly. The
corresponding model, which we term GraphGen-Redux, improves upon the generative
performances of GraphGen in a wide range of datasets of chemical and social
graphs. In addition, it uses approximately 78% fewer parameters than the
vanilla variant and requires 50% fewer epochs of training on average.

    

### [[2107.08398] Unsupervised Skill-Discovery and Skill-Learning in Minecraft](http://arxiv.org/abs/2107.08398)


  Pre-training Reinforcement Learning agents in a task-agnostic manner has
shown promising results. However, previous works still struggle in learning and
discovering meaningful skills in high-dimensional state-spaces, such as
pixel-spaces. We approach the problem by leveraging unsupervised skill
discovery and self-supervised learning of state representations. In our work,
we learn a compact latent representation by making use of variational and
contrastive techniques. We demonstrate that both enable RL agents to learn a
set of basic navigation skills by maximizing an information theoretic
objective. We assess our method in Minecraft 3D pixel maps with different
complexities. Our results show that representations and conditioned policies
learned from pixels are enough for toy examples, but do not scale to realistic
and complex maps. To overcome these limitations, we explore alternative input
observations such as the relative position of the agent along with the raw
pixels.

    

### [[2107.08399] A method for estimating the entropy of time series using artificial neural network](http://arxiv.org/abs/2107.08399)


  Measuring the predictability and complexity of time series is an essential
tool in designing and controlling the nonlinear system. There exist different
entropy measures in the literature to analyze the predictability and complexity
of time series. However, these measures have some drawbacks especially in short
time series. To overcome the difficulties, this paper proposes a new method for
estimating the entropy of a time series using the LogNNet 784:25:10 neural
network model. The LogNNet reservoir matrix consists of 19625 elements which is
filled with the time series elements. After that, the network is trained on
MNIST-10 dataset and the classification accuracy is calculated. The accuracy is
considered as the entropy measure and denoted by NNetEn. A more complex
transformation of the input information by the time series in the reservoir
leads to higher NNetEn values. Many practical time series data have less than
19625 elements. Some duplicating or stretching methods are investigated to
overcome this difficulty and the most successful method is identified for
practical applications. The epochs number in the training process of LogNNet is
considered as the input parameter. A new time series characteristic called time
series learning inertia is introduced to investigate the effect of epochs
number in the efficiency of neural network. To show the robustness and
efficiency of the proposed method, it is applied on some chaotic, periodic,
random, binary and constant time series. The NNetEn is compared with some
existing entropy measures. The results show that the proposed method is more
robust and accurate than existing methods.

    

### [[2107.08402] RobustFed: A Truth Inference Approach for Robust Federated Learning](http://arxiv.org/abs/2107.08402)


  Federated learning is a prominent framework that enables clients (e.g.,
mobile devices or organizations) to train a collaboratively global model under
a central server's orchestration while keeping local training datasets'
privacy. However, the aggregation step in federated learning is vulnerable to
adversarial attacks as the central server cannot manage clients' behavior.
Therefore, the global model's performance and convergence of the training
process will be affected under such this http URL mitigate this vulnerability
issue, we propose a novel robust aggregation algorithm inspired by the truth
inference methods in crowdsourcing via incorporating the worker's reliability
into aggregation. We evaluate our solution on three real-world datasets with a
variety of machine learning models. Experimental results show that our solution
ensures robust federated learning and is resilient to various types of attacks,
including noisy data attacks, Byzantine attacks, and label flipping attacks.

    

### [[2107.08426] A New Representation of Successor Features for Transfer across Dissimilar Environments](http://arxiv.org/abs/2107.08426)


  Transfer in reinforcement learning is usually achieved through generalisation
across tasks. Whilst many studies have investigated transferring knowledge when
the reward function changes, they have assumed that the dynamics of the
environments remain consistent. Many real-world RL problems require transfer
among environments with different dynamics. To address this problem, we propose
an approach based on successor features in which we model successor feature
functions with Gaussian Processes permitting the source successor features to
be treated as noisy measurements of the target successor feature function. Our
theoretical analysis proves the convergence of this approach as well as the
bounded error on modelling successor feature functions with Gaussian Processes
in environments with both different dynamics and rewards. We demonstrate our
method on benchmark datasets and show that it outperforms current baselines.

    

### [[2107.08429] Support vector machines for learning reactive islands](http://arxiv.org/abs/2107.08429)


  We develop a machine learning framework that can be applied to data sets
derived from the trajectories of Hamilton's equations. The goal is to learn the
phase space structures that play the governing role for phase space transport
relevant to particular applications. Our focus is on learning reactive islands
in two degrees-of-freedom Hamiltonian systems. Reactive islands are constructed
from the stable and unstable manifolds of unstable periodic orbits and play the
role of quantifying transition dynamics. We show that support vector machines
(SVM) is an appropriate machine learning framework for this purpose as it
provides an approach for finding the boundaries between qualitatively distinct
dynamical behaviors, which is in the spirit of the phase space transport
framework. We show how our method allows us to find reactive islands directly
in the sense that we do not have to first compute unstable periodic orbits and
their stable and unstable manifolds. We apply our approach to the
Hénon-Heiles Hamiltonian system, which is a benchmark system in the dynamical
systems community. We discuss different sampling and learning approaches and
their advantages and disadvantages.

    

### [[2107.08442] Sleep Staging Based on Serialized Dual Attention Network](http://arxiv.org/abs/2107.08442)


  Sleep staging assumes an important role in the diagnosis of sleep disorders.
In general, experts classify sleep stages manually based on polysomnography
(PSG), which is quite time-consuming. Meanwhile, the acquisition of multiple
signals is complex, which can affect the subject's sleep. Therefore, the use of
single-channel electroencephalogram (EEG) for automatic sleep staging has
become mainstream. In the literature, a large number of sleep staging methods
based on single-channel EEG have been proposed with good results and realize
the preliminary automation of sleep staging. However, the performance for most
of these methods in the N1 stage is generally not high. In this paper, we
propose a deep learning model SDAN based on raw EEG. The method utilises a
one-dimensional convolutional neural network (CNN) to automatically extract
features from raw EEG. It serially combines the channel attention and spatial
attention mechanisms to filter and highlight key information and then uses soft
threshold to eliminate redundant information. Additionally, we introduce a
residual network to avoid degradation problems caused by network deepening.
Experiments were conducted using two datasets with 5-fold cross-validation and
hold-out validation method. The final average accuracy, overall accuracy, macro
F1 score and Cohen's Kappa coefficient of the model reach 96.74%, 91.86%,
82.64% and 0.8742 on the Sleep-EDF dataset, and 95.98%, 89.96%, 79.08% and
0.8216 on the Sleep-EDFx dataset. Significantly, our model performed superiorly
in the N1 stage, with F1 scores of 54.08% and 52.49% on the two datasets
respectively. The results show the superiority of our network over the best
existing methods, reaching a new state-of-the-art. In particular, the present
method achieves excellent results in the N1 sleep stage compared to other
methods.

    

### [[2107.08444] A Theory of PAC Learnability of Partial Concept Classes](http://arxiv.org/abs/2107.08444)


  We extend the theory of PAC learning in a way which allows to model a rich
variety of learning tasks where the data satisfy special properties that ease
the learning process. For example, tasks where the distance of the data from
the decision boundary is bounded away from zero. The basic and simple idea is
to consider partial concepts: these are functions that can be undefined on
certain parts of the space. When learning a partial concept, we assume that the
source distribution is supported only on points where the partial concept is
defined.
This way, one can naturally express assumptions on the data such as lying on
a lower dimensional surface or margin conditions. In contrast, it is not at all
clear that such assumptions can be expressed by the traditional PAC theory. In
fact we exhibit easy-to-learn partial concept classes which provably cannot be
captured by the traditional PAC theory. This also resolves a question posed by
Attias, Kontorovich, and Mansour 2019.
We characterize PAC learnability of partial concept classes and reveal an
algorithmic landscape which is fundamentally different than the classical one.
For example, in the classical PAC model, learning boils down to Empirical Risk
Minimization (ERM). In stark contrast, we show that the ERM principle fails in
explaining learnability of partial concept classes. In fact, we demonstrate
classes that are incredibly easy to learn, but such that any algorithm that
learns them must use an hypothesis space with unbounded VC dimension. We also
find that the sample compression conjecture fails in this setting.
Thus, this theory features problems that cannot be represented nor solved in
the traditional way. We view this as evidence that it might provide insights on
the nature of learnability in realistic scenarios which the classical theory
fails to explain.

    

### [[2107.08461] Differentially Private Bayesian Neural Networks on Accuracy, Privacy and Reliability](http://arxiv.org/abs/2107.08461)


  Bayesian neural network (BNN) allows for uncertainty quantification in
prediction, offering an advantage over regular neural networks that has not
been explored in the differential privacy (DP) framework. We fill this
important gap by leveraging recent development in Bayesian deep learning and
privacy accounting to offer a more precise analysis of the trade-off between
privacy and accuracy in BNN. We propose three DP-BNNs that characterize the
weight uncertainty for the same network architecture in distinct ways, namely
DP-SGLD (via the noisy gradient method), DP-BBP (via changing the parameters of
interest) and DP-MC Dropout (via the model architecture). Interestingly, we
show a new equivalence between DP-SGD and DP-SGLD, implying that some
non-Bayesian DP training naturally allows for uncertainty quantification.
However, the hyperparameters such as learning rate and batch size, can have
different or even opposite effects in DP-SGD and DP-SGLD.
Extensive experiments are conducted to compare DP-BNNs, in terms of privacy
guarantee, prediction accuracy, uncertainty quantification, calibration,
computation speed, and generalizability to network architecture. As a result,
we observe a new tradeoff between the privacy and the reliability. When
compared to non-DP and non-Bayesian approaches, DP-SGLD is remarkably accurate
under strong privacy guarantee, demonstrating the great potential of DP-BNN in
real-world tasks.

    

### [[2107.08467] GoTube: Scalable Stochastic Verification of Continuous-Depth Models](http://arxiv.org/abs/2107.08467)


  We introduce a new stochastic verification algorithm that formally quantifies
the behavioral robustness of any time-continuous process formulated as a
continuous-depth model. The algorithm solves a set of global optimization (Go)
problems over a given time horizon to construct a tight enclosure (Tube) of the
set of all process executions starting from a ball of initial states. We call
our algorithm GoTube. Through its construction, GoTube ensures that the
bounding tube is conservative up to a desired probability. GoTube is
implemented in JAX and optimized to scale to complex continuous-depth models.
Compared to advanced reachability analysis tools for time-continuous neural
networks, GoTube provably does not accumulate over-approximation errors between
time steps and avoids the infamous wrapping effect inherent in symbolic
techniques. We show that GoTube substantially outperforms state-of-the-art
verification tools in terms of the size of the initial ball, speed,
time-horizon, task completion, and scalability, on a large set of experiments.
GoTube is stable and sets the state-of-the-art for its ability to scale up to
time horizons well beyond what has been possible before.

    

### [[2107.08470] ANFIC: Image Compression Using Augmented Normalizing Flows](http://arxiv.org/abs/2107.08470)


  This paper introduces an end-to-end learned image compression system, termed
ANFIC, based on Augmented Normalizing Flows (ANF). ANF is a new type of flow
model, which stacks multiple variational autoencoders (VAE) for greater model
expressiveness. The VAE-based image compression has gone mainstream, showing
promising compression performance. Our work presents the first attempt to
leverage VAE-based compression in a flow-based framework. ANFIC advances
further compression efficiency by stacking and extending hierarchically
multiple VAE's. The invertibility of ANF, together with our training
strategies, enables ANFIC to support a wide range of quality levels without
changing the encoding and decoding networks. Extensive experimental results
show that in terms of PSNR-RGB, ANFIC performs comparably to or better than the
state-of-the-art learned image compression. Moreover, it performs close to VVC
intra coding, from low-rate compression up to nearly-lossless compression. In
particular, ANFIC achieves the state-of-the-art performance, when extended with
conditional convolution for variable rate compression with a single model.

    

### [[2107.08471] A stepped sampling method for video detection using LSTM](http://arxiv.org/abs/2107.08471)


  Artificial neural networks that simulate human achieves great successes. From
the perspective of simulating human memory method, we propose a stepped sampler
based on the "repeated input". We repeatedly inputted data to the LSTM model
stepwise in a batch. The stepped sampler is used to strengthen the ability of
fusing the temporal information in LSTM. We tested the stepped sampler on the
LSTM built-in in PyTorch. Compared with the traditional sampler of PyTorch,
such as sequential sampler, batch sampler, the training loss of the proposed
stepped sampler converges faster in the training of the model, and the training
loss after convergence is more stable. Meanwhile, it can maintain a higher test
accuracy. We quantified the algorithm of the stepped sampler.

    

### [[2107.08484] A Novel Evolutionary Algorithm for Hierarchical Neural Architecture Search](http://arxiv.org/abs/2107.08484)


  In this work, we propose a novel evolutionary algorithm for neural
architecture search, applicable to global search spaces. The algorithm's
architectural representation organizes the topology in multiple hierarchical
modules, while the design process exploits this representation, in order to
explore the search space. We also employ a curation system, which promotes the
utilization of well performing sub-structures to subsequent generations. We
apply our method to Fashion-MNIST and NAS-Bench101, achieving accuracies of
$93.2\%$ and $94.8\%$ respectively in a relatively small number of generations.

    

### [[2107.08488] A note on the article "On Exploiting Spectral Properties for Solving MDP with Large State Space"](http://arxiv.org/abs/2107.08488)


  We improve a theoretical result of the article "On Exploiting Spectral
Properties for Solving MDP with Large State Space" showing that their
algorithm, which was proved to converge under some unrealistic assumptions, is
actually guaranteed to converge always.

    

### [[2107.08514] Classification of Upper Arm Movements from EEG signals using Machine Learning with ICA Analysis](http://arxiv.org/abs/2107.08514)


  The Brain-Computer Interface system is a profoundly developing area of
experimentation for Motor activities which plays vital role in decoding
cognitive activities. Classification of Cognitive-Motor Imagery activities from
EEG signals is a critical task. Hence proposed a unique algorithm for
classifying left/right-hand movements by utilizing Multi-layer Perceptron
Neural Network. Handcrafted statistical Time domain and Power spectral density
frequency domain features were extracted and obtained a combined accuracy of
96.02%. Results were compared with the deep learning framework. In addition to
accuracy, Precision, F1-Score, and recall was considered as the performance
metrics. The intervention of unwanted signals contaminates the EEG signals
which influence the performance of the algorithm. Therefore, a novel approach
was approached to remove the artifacts using Independent Components Analysis
which boosted the performance. Following the selection of appropriate feature
vectors that provided acceptable accuracy. The same method was used on all nine
subjects. As a result, intra-subject accuracy was obtained for 9 subjects
94.72%. The results show that the proposed approach would be useful to classify
the upper limb movements accurately.

    

### [[2107.08517] Decentralized federated learning of deep neural networks on non-iid data](http://arxiv.org/abs/2107.08517)


  We tackle the non-convex problem of learning a personalized deep learning
model in a decentralized setting. More specifically, we study decentralized
federated learning, a peer-to-peer setting where data is distributed among many
clients and where there is no central server to orchestrate the training. In
real world scenarios, the data distributions are often heterogeneous between
clients. Therefore, in this work we study the problem of how to efficiently
learn a model in a peer-to-peer system with non-iid client data. We propose a
method named Performance-Based Neighbor Selection (PENS) where clients with
similar data distributions detect each other and cooperate by evaluating their
training losses on each other's data to learn a model suitable for the local
data distribution. Our experiments on benchmark datasets show that our proposed
method is able to achieve higher accuracies as compared to strong baselines.

    

### [[2107.08558] A Topological Perspective on Causal Inference](http://arxiv.org/abs/2107.08558)


  This paper presents a topological learning-theoretic perspective on causal
inference by introducing a series of topologies defined on general spaces of
structural causal models (SCMs). As an illustration of the framework we prove a
topological causal hierarchy theorem, showing that substantive assumption-free
causal inference is possible only in a meager set of SCMs. Thanks to a known
correspondence between open sets in the weak topology and statistically
verifiable hypotheses, our results show that inductive assumptions sufficient
to license valid causal inferences are statistically unverifiable in principle.
Similar to no-free-lunch theorems for statistical inference, the present
results clarify the inevitability of substantial assumptions for causal
inference. An additional benefit of our topological approach is that it easily
accommodates SCMs with infinitely many variables. We finally suggest that the
framework may be helpful for the positive project of exploring and assessing
alternative causal-inductive assumptions.

    

### [[2107.08562] Rethinking Graph Autoencoder Models for Attributed Graph Clustering](http://arxiv.org/abs/2107.08562)


  Most recent graph clustering methods have resorted to Graph Auto-Encoders
(GAEs) to perform joint clustering and embedding learning. However, two
critical issues have been overlooked. First, the accumulative error, inflicted
by learning with noisy clustering assignments, degrades the effectiveness and
robustness of the clustering model. This problem is called Feature Randomness.
Second, reconstructing the adjacency matrix sets the model to learn irrelevant
similarities for the clustering task. This problem is called Feature Drift.
Interestingly, the theoretical relation between the aforementioned problems has
not yet been investigated. We study these issues from two aspects: (1) the
existence of a trade-off between Feature Randomness and Feature Drift when
clustering and reconstruction are performed at the same level, and (2) the
problem of Feature Drift is more pronounced for GAE models, compared with
vanilla auto-encoder models, due to the graph convolutional operation and the
graph decoding design. Motivated by these findings, we reformulate the
GAE-based clustering methodology. Our solution is two-fold. First, we propose a
sampling operator $\Xi$ that triggers a protection mechanism against the noisy
clustering assignments. Second, we propose an operator $\Upsilon$ that triggers
a correction mechanism against Feature Drift by gradually transforming the
reconstructed graph into a clustering-oriented one. As principal advantages,
our solution grants a considerable improvement in clustering effectiveness and
robustness and can be easily tailored to existing GAE models.

    

### [[2107.08564] Wave-based extreme deep learning based on non-linear time-Floquet entanglement](http://arxiv.org/abs/2107.08564)


  Wave-based analog signal processing holds the promise of extremely fast,
on-the-fly, power-efficient data processing, occurring as a wave propagates
through an artificially engineered medium. Yet, due to the fundamentally weak
non-linearities of traditional wave materials, such analog processors have been
so far largely confined to simple linear projections such as image edge
detection or matrix multiplications. Complex neuromorphic computing tasks,
which inherently require strong non-linearities, have so far remained
out-of-reach of wave-based solutions, with a few attempts that implemented
non-linearities on the digital front, or used weak and inflexible non-linear
sensors, restraining the learning performance. Here, we tackle this issue by
demonstrating the relevance of Time-Floquet physics to induce a strong
non-linear entanglement between signal inputs at different frequencies,
enabling a power-efficient and versatile wave platform for analog extreme deep
learning involving a single, uniformly modulated dielectric layer and a
scattering medium. We prove the efficiency of the method for extreme learning
machines and reservoir computing to solve a range of challenging learning
tasks, from forecasting chaotic time series to the simultaneous classification
of distinct datasets. Our results open the way for wave-based machine learning
with high energy efficiency, speed, and scalability.

    

### [[2107.08567] Structural Design Recommendations in the Early Design Phase using Machine Learning](http://arxiv.org/abs/2107.08567)


  Structural engineering knowledge can be of significant importance to the
architectural design team during the early design phase. However, architects
and engineers do not typically work together during the conceptual phase; in
fact, structural engineers are often called late into the process. As a result,
updates in the design are more difficult and time-consuming to complete. At the
same time, there is a lost opportunity for better design exploration guided by
structural feedback. In general, the earlier in the design process the
iteration happens, the greater the benefits in cost efficiency and informed
de-sign exploration, which can lead to higher-quality creative results. In
order to facilitate an informed exploration in the early design stage, we
suggest the automation of fundamental structural engineering tasks and
introduce ApproxiFramer, a Machine Learning-based system for the automatic
generation of structural layouts from building plan sketches in real-time. The
system aims to assist architects by presenting them with feasible structural
solutions during the conceptual phase so that they proceed with their design
with adequate knowledge of its structural implications. In this paper, we
describe the system and evaluate the performance of a proof-of-concept
implementation in the domain of orthogonal, metal, rigid structures. We trained
a Convolutional Neural Net to iteratively generate structural design solutions
for sketch-level building plans using a synthetic dataset and achieved an
average error of 2.2% in the predicted positions of the columns.

    

### [[2107.08572] Early-Phase Performance-Driven Design using Generative Models](http://arxiv.org/abs/2107.08572)


  Current performance-driven building design methods are not widely adopted
outside the research field for several reasons that make them difficult to
integrate into a typical design process. In the early design phase, in
particular, the time-intensity and the cognitive load associated with
optimization and form parametrization are incompatible with design exploration,
which requires quick iteration. This research introduces a novel method for
performance-driven geometry generation that can afford interaction directly in
the 3d modeling environment, eliminating the need for explicit parametrization,
and is multiple orders faster than the equivalent form optimization. The method
uses Machine Learning techniques to train a generative model offline. The
generative model learns a distribution of optimal performing geometries and
their simulation contexts based on a dataset that addresses the performance(s)
of interest. By navigating the generative model's latent space, geometries with
the desired characteristics can be quickly generated. A case study is
presented, demonstrating the generation of a synthetic dataset and the use of a
Variational Autoencoder (VAE) as a generative model for geometries with optimal
solar gain. The results show that the VAE-generated geometries perform on
average at least as well as the optimized ones, suggesting that the introduced
method shows a feasible path towards more intuitive and interactive early-phase
performance-driven design assistance.

    

### [[2107.08574] A Modulation Layer to Increase Neural Network Robustness Against Data Quality Issues](http://arxiv.org/abs/2107.08574)


  Data quality is a common problem in machine learning, especially in
high-stakes settings such as healthcare. Missing data affects accuracy,
calibration, and feature attribution in complex patterns. Developers often
train models on carefully curated datasets to minimize missing data bias;
however, this reduces the usability of such models in production environments,
such as real-time healthcare records. Making machine learning models robust to
missing data is therefore crucial for practical application. While some
classifiers naturally handle missing data, others, such as deep neural
networks, are not designed for unknown values. We propose a novel neural
network modification to mitigate the impacts of missing data. The approach is
inspired by neuromodulation that is performed by biological neural networks.
Our proposal replaces the fixed weights of a fully-connected layer with a
function of an additional input (reliability score) at each input, mimicking
the ability of cortex to up- and down-weight inputs based on the presence of
other data. The modulation function is jointly learned with the main task using
a multi-layer perceptron. We tested our modulating fully connected layer on
multiple classification, regression, and imputation problems, and it either
improved performance or generated comparable performance to conventional neural
network architectures concatenating reliability to the inputs. Models with
modulating layers were more robust against degradation of data quality by
introducing additional missingness at evaluation time. These results suggest
that explicitly accounting for reduced information quality with a modulating
fully connected layer can enable the deployment of artificial intelligence
systems in real-time settings.

    

### [[2107.08577] Structured World Belief for Reinforcement Learning in POMDP](http://arxiv.org/abs/2107.08577)


  Object-centric world models provide structured representation of the scene
and can be an important backbone in reinforcement learning and planning.
However, existing approaches suffer in partially-observable environments due to
the lack of belief states. In this paper, we propose Structured World Belief, a
model for learning and inference of object-centric belief states. Inferred by
Sequential Monte Carlo (SMC), our belief states provide multiple object-centric
scene hypotheses. To synergize the benefits of SMC particles with object
representations, we also propose a new object-centric dynamics model that
considers the inductive bias of object permanence. This enables tracking of
object states even when they are invisible for a long time. To further
facilitate object tracking in this regime, we allow our model to attend
flexibly to any spatial location in the image which was restricted in previous
models. In experiments, we show that object-centric belief provides a more
accurate and robust performance for filtering and generation. Furthermore, we
show the efficacy of structured world belief in improving the performance of
reinforcement learning, planning and supervised reasoning.

    

### [[2107.08588] Chef: a cheap and fast pipeline for iteratively cleaning label uncertainties](http://arxiv.org/abs/2107.08588)


  High-quality labels are expensive to obtain for many machine learning tasks,
such as medical image classification tasks. Therefore, probabilistic (weak)
labels produced by weak supervision tools are used to seed a process in which
influential samples with weak labels are identified and cleaned by several
human annotators to improve the model performance. To lower the overall cost
and computational overhead of this process, we propose a solution called
Chef(CHEap and Fast label cleaning), which consists of the following three
components. First, to reduce the cost of human annotators, we use Infl, which
prioritizes the most influential training samples for cleaning and provides
cleaned labels to save the cost of one human annotator. Second, to accelerate
the sample selector phase and the model constructor phase, we use Increm-Infl
to incrementally produce influential samples, and DeltaGrad-L to incrementally
update the model. Third, we redesign the typical label cleaning pipeline so
that human annotators iteratively clean smaller batch of samples rather than
one big batch of samples. This yields better over all model performance and
enables possible early termination when the expected model performance has been
achieved. Extensive experiments show that our approach gives good model
prediction performance while achieving significant speed-ups.

    

### [[2107.08593] Inverse Problem of Nonlinear Schrödinger Equation as Learning of Convolutional Neural Network](http://arxiv.org/abs/2107.08593)


  In this work, we use an explainable convolutional neural network (NLS-Net) to
solve an inverse problem of the nonlinear Schrödinger equation, which is
widely used in fiber-optic communications. The landscape and minimizers of the
non-convex loss function of the learning problem are studied empirically. It
provides a guidance for choosing hyper-parameters of the method. The estimation
error of the optimal solution is discussed in terms of expressive power of the
NLS-Net and data. Besides, we compare the performance of several training
algorithms that are popular in deep learning. It is shown that one can obtain a
relatively accurate estimate of the considered parameters using the proposed
method. The study provides a natural framework of solving inverse problems of
nonlinear partial differential equations with deep learning.

    

### [[2107.08594] Optimal Resource Allocation for Serverless Queries](http://arxiv.org/abs/2107.08594)


  Optimizing resource allocation for analytical workloads is vital for reducing
costs of cloud-data services. At the same time, it is incredibly hard for users
to allocate resources per query in serverless processing systems, and they
frequently misallocate by orders of magnitude. Unfortunately, prior work
focused on predicting peak allocation while ignoring aggressive trade-offs
between resource allocation and run-time. Additionally, these methods fail to
predict allocation for queries that have not been observed in the past. In this
paper, we tackle both these problems. We introduce a system for optimal
resource allocation that can predict performance with aggressive trade-offs,
for both new and past observed queries. We introduce the notion of a
performance characteristic curve (PCC) as a parameterized representation that
can compactly capture the relationship between resources and performance. To
tackle training data sparsity, we introduce a novel data augmentation technique
to efficiently synthesize the entire PCC using a single run of the query.
Lastly, we demonstrate the advantages of a constrained loss function coupled
with GNNs, over traditional ML methods, for capturing the domain specific
behavior through an extensive experimental evaluation over SCOPE big data
workloads at Microsoft.

    

### [[2107.08595] High-Dimensional Simulation Optimization via Brownian Fields and Sparse Grids](http://arxiv.org/abs/2107.08595)


  High-dimensional simulation optimization is notoriously challenging. We
propose a new sampling algorithm that converges to a global optimal solution
and suffers minimally from the curse of dimensionality. The algorithm consists
of two stages. First, we take samples following a sparse grid experimental
design and approximate the response surface via kernel ridge regression with a
Brownian field kernel. Second, we follow the expected improvement strategy --
with critical modifications that boost the algorithm's sample efficiency -- to
iteratively sample from the next level of the sparse grid. Under mild
conditions on the smoothness of the response surface and the simulation noise,
we establish upper bounds on the convergence rate for both noise-free and noisy
simulation samples. These upper rates deteriorate only slightly in the
dimension of the feasible set, and they can be improved if the objective
function is known be of a higher-order smoothness. Extensive numerical
experiments demonstrate that the proposed algorithm dramatically outperforms
typical alternatives in practice.

    

### [[2107.08596] Equivariant Manifold Flows](http://arxiv.org/abs/2107.08596)


  Tractably modelling distributions over manifolds has long been an important
goal in the natural sciences. Recent work has focused on developing general
machine learning models to learn such distributions. However, for many
applications these distributions must respect manifold symmetries -- a trait
which most previous models disregard. In this paper, we lay the theoretical
foundations for learning symmetry-invariant distributions on arbitrary
manifolds via equivariant manifold flows. We demonstrate the utility of our
approach by using it to learn gauge invariant densities over $SU(n)$ in the
context of quantum field theory.

    

### [[2107.08598] Learning-To-Ensemble by Contextual Rank Aggregation in E-Commerce](http://arxiv.org/abs/2107.08598)


  Ensemble models in E-commerce combine predictions from multiple sub-models
for ranking and revenue improvement. Industrial ensemble models are typically
deep neural networks, following the supervised learning paradigm to infer
conversion rate given inputs from sub-models. However, this process has the
following two problems. Firstly, the point-wise scoring approach disregards the
relationships between items and leads to homogeneous displayed results, while
diversified display benefits user experience and revenue. Secondly, the
learning paradigm focuses on the ranking metrics and does not directly optimize
the revenue. In our work, we propose a new Learning-To-Ensemble (LTE) framework
RAEGO, which replaces the ensemble model with a contextual Rank Aggregator (RA)
and explores the best weights of sub-models by the Evaluator-Generator
Optimization (EGO). To achieve the best online performance, we propose a new
rank aggregation algorithm TournamentGreedy as a refinement of classic rank
aggregators, which also produces the best average weighted Kendall Tau Distance
(KTD) amongst all the considered algorithms with quadratic time complexity.
Under the assumption that the best output list should be Pareto Optimal on the
KTD metric for sub-models, we show that our RA algorithm has higher efficiency
and coverage in exploring the optimal weights. Combined with the idea of
Bayesian Optimization and gradient descent, we solve the online contextual
Black-Box Optimization task that finds the optimal weights for sub-models given
a chosen RA model. RA-EGO has been deployed in our online system and has
improved the revenue significantly.

    

### [[2107.08622] Provably Efficient Multi-Task Reinforcement Learning with Model Transfer](http://arxiv.org/abs/2107.08622)


  We study multi-task reinforcement learning (RL) in tabular episodic Markov
decision processes (MDPs). We formulate a heterogeneous multi-player RL
problem, in which a group of players concurrently face similar but not
necessarily identical MDPs, with a goal of improving their collective
performance through inter-player information sharing. We design and analyze an
algorithm based on the idea of model transfer, and provide gap-dependent and
gap-independent upper and lower bounds that characterize the intrinsic
complexity of the problem.

    

### [[2107.08640] Facial Expressions Recognition with Convolutional Neural Networks](http://arxiv.org/abs/2107.08640)


  Over the centuries, humans have developed and acquired a number of ways to
communicate. But hardly any of them can be as natural and instinctive as facial
expressions. On the other hand, neural networks have taken the world by storm.
And no surprises, that the area of Computer Vision and the problem of facial
expressions recognitions hasn't remained untouched. Although a wide range of
techniques have been applied, achieving extremely high accuracies and preparing
highly robust FER systems still remains a challenge due to heterogeneous
details in human faces. In this paper, we will be deep diving into implementing
a system for recognition of facial expressions (FER) by leveraging neural
networks, and more specifically, Convolutional Neural Networks (CNNs). We adopt
the fundamental concepts of deep learning and computer vision with various
architectures, fine-tune it's hyperparameters and experiment with various
optimization methods and demonstrate a state-of-the-art single-network-accuracy
of 70.10% on the FER2013 dataset without using any additional training data.

    

### [[2107.08649] Non-asymptotic estimates for TUSLA algorithm for non-convex learning with applications to neural networks with ReLU activation function](http://arxiv.org/abs/2107.08649)


  We consider non-convex stochastic optimization problems where the objective
functions have super-linearly growing and discontinuous stochastic gradients.
In such a setting, we provide a non-asymptotic analysis for the tamed
unadjusted stochastic Langevin algorithm (TUSLA) introduced in Lovas et al.
(2021). In particular, we establish non-asymptotic error bounds for the TUSLA
algorithm in Wasserstein-1 and Wasserstein-2 distances. The latter result
enables us to further derive non-asymptotic estimates for the expected excess
risk. To illustrate the applicability of the main results, we consider an
example from transfer learning with ReLU neural networks, which represents a
key paradigm in machine learning. Numerical experiments are presented for the
aforementioned example which supports our theoretical findings. Hence, in this
setting, we demonstrate both theoretically and numerically that the TUSLA
algorithm can solve the optimization problem involving neural networks with
ReLU activation function. Besides, we provide simulation results for synthetic
examples where popular algorithms, e.g. ADAM, AMSGrad, RMSProp, and (vanilla)
SGD, may fail to find the minimizer of the objective functions due to the
super-linear growth and the discontinuity of the corresponding stochastic
gradient, while the TUSLA algorithm converges rapidly to the optimal solution.

    

### [[2107.08661] Translatotron 2: Robust direct speech-to-speech translation](http://arxiv.org/abs/2107.08661)


  We present Translatotron 2, a neural direct speech-to-speech translation
model that can be trained end-to-end. Translatotron 2 consists of a speech
encoder, a phoneme decoder, a mel-spectrogram synthesizer, and an attention
module that connects all the previous three components. Experimental results
suggest that Translatotron 2 outperforms the original Translatotron by a large
margin in terms of translation quality and predicted speech naturalness, and
drastically improves the robustness of the predicted speech by mitigating
over-generation, such as babbling or long pause. We also propose a new method
for retaining the source speaker's voice in the translated speech. The trained
model is restricted to retain the source speaker's voice, and unlike the
original Translatotron, it is not able to generate speech in a different
speaker's voice, making the model more robust for production deployment, by
mitigating potential misuse for creating spoofing audio artifacts. When the new
method is used together with a simple concatenation-based data augmentation,
the trained Translatotron 2 model is able to retain each speaker's voice for
input with speaker turns.

    

### [[2107.08662] A Queueing-Theoretic Framework for Vehicle Dispatching in Dynamic Car-Hailing [technical report]](http://arxiv.org/abs/2107.08662)


  With the rapid development of smart mobile devices, the car-hailing platforms
(e.g., Uber or Lyft) have attracted much attention from both the academia and
the industry. In this paper, we consider an important dynamic car-hailing
problem, namely \textit{maximum revenue vehicle dispatching} (MRVD), in which
rider requests dynamically arrive and drivers need to serve as many riders as
possible such that the entire revenue of the platform is maximized. We prove
that the MRVD problem is NP-hard and intractable. In addition, the dynamic
car-hailing platforms have no information of the future riders, which makes the
problem even harder. To handle the MRVD problem, we propose a queueing-based
vehicle dispatching framework, which first uses existing machine learning
algorithms to predict the future vehicle demand of each region, then estimates
the idle time periods of drivers through a queueing model for each region. With
the information of the predicted vehicle demands and estimated idle time
periods of drivers, we propose two batch-based vehicle dispatching algorithms
to efficiently assign suitable drivers to riders such that the expected overall
revenue of the platform is maximized during each batch processing. Through
extensive experiments, we demonstrate the efficiency and effectiveness of our
proposed approaches over both real and synthetic datasets.

    

### [[2107.08673] Input Agnostic Deep Learning for Alzheimer's Disease Classification Using Multimodal MRI Images](http://arxiv.org/abs/2107.08673)


  Alzheimer's disease (AD) is a progressive brain disorder that causes memory
and functional impairments. The advances in machine learning and publicly
available medical datasets initiated multiple studies in AD diagnosis. In this
work, we utilize a multi-modal deep learning approach in classifying normal
cognition, mild cognitive impairment and AD classes on the basis of structural
MRI and diffusion tensor imaging (DTI) scans from the OASIS-3 dataset. In
addition to a conventional multi-modal network, we also present an input
agnostic architecture that allows diagnosis with either sMRI or DTI scan, which
distinguishes our method from previous multi-modal machine learning-based
methods. The results show that the input agnostic model achieves 0.96 accuracy
when both structural MRI and DTI scans are provided as inputs.

    

### [[2107.08686] Improved Learning Rates for Stochastic Optimization: Two Theoretical Viewpoints](http://arxiv.org/abs/2107.08686)


  Generalization performance of stochastic optimization stands a central place
in machine learning. In this paper, we investigate the excess risk performance
and towards improved learning rates for two popular approaches of stochastic
optimization: empirical risk minimization (ERM) and stochastic gradient descent
(SGD). Although there exists plentiful generalization analysis of ERM and SGD
for supervised learning, current theoretical understandings of ERM and SGD are
either have stronger assumptions in convex learning, e.g., strong convexity
condition, or show slow rates and less studied in nonconvex learning. Motivated
by these problems, we aim to provide improved rates under milder assumptions in
convex learning and derive faster rates in nonconvex learning. It is notable
that our analysis span two popular theoretical viewpoints: stability and
uniform convergence. To be specific, in stability regime, we present high
probability rates of order $\mathcal{O} (1/n)$ w.r.t. the sample size $n$ for
ERM and SGD with milder assumptions in convex learning and similar high
probability rates of order $\mathcal{O} (1/n)$ in nonconvex learning, rather
than in expectation. Furthermore, this type of learning rate is improved to
faster order $\mathcal{O} (1/n^2)$ in uniform convergence regime. To the best
of our knowledge, for ERM and SGD, the learning rates presented in this paper
are all state-of-the-art.

    

### [[2107.08687] Long-term series forecasting with Query Selector -- efficient model of sparse attention](http://arxiv.org/abs/2107.08687)


  Various modifications of TRANSFORMER were recently used to solve time-series
forecasting problem. We propose Query Selector - an efficient, deterministic
algorithm for sparse attention matrix. Experiments show it achieves
state-of-the art results on ETT data set.

    

### [[2107.08695] Deceptive Logic Locking for Hardware Integrity Protection against Machine Learning Attacks](http://arxiv.org/abs/2107.08695)


  Logic locking has emerged as a prominent key-driven technique to protect the
integrity of integrated circuits. However, novel machine-learning-based attacks
have recently been introduced to challenge the security foundations of locking
schemes. These attacks are able to recover a significant percentage of the key
without having access to an activated circuit. This paper address this issue
through two focal points. First, we present a theoretical model to test locking
schemes for key-related structural leakage that can be exploited by machine
learning. Second, based on the theoretical model, we introduce D-MUX: a
deceptive multiplexer-based logic-locking scheme that is resilient against
structure-exploiting machine learning attacks. Through the design of D-MUX, we
uncover a major fallacy in existing multiplexer-based locking schemes in the
form of a structural-analysis attack. Finally, an extensive cost evaluation of
D-MUX is presented. To the best of our knowledge, D-MUX is the first
machine-learning-resilient locking scheme capable of protecting against all
known learning-based attacks. Hereby, the presented work offers a starting
point for the design and evaluation of future-generation logic locking in the
era of machine learning.

    

### [[2107.08697] Interpreting Process Predictions using a Milestone-Aware Counterfactual Approach](http://arxiv.org/abs/2107.08697)


  Predictive process analytics often apply machine learning to predict the
future states of a running business process. However, the internal mechanisms
of many existing predictive algorithms are opaque and a human decision-maker is
unable to understand \emph{why} a certain activity was predicted. Recently,
counterfactuals have been proposed in the literature to derive
human-understandable explanations from predictive models. Current
counterfactual approaches consist of finding the minimum feature change that
can make a certain prediction flip its outcome. Although many algorithms have
been proposed, their application to the sequence and multi-dimensional data
like event logs has not been explored in the literature.
In this paper, we explore the use of a recent, popular model-agnostic
counterfactual algorithm, DiCE, in the context of predictive process analytics.
The analysis reveals that the algorithm is limited when being applied to derive
explanations of process predictions, due to (1) process domain knowledge not
being taken into account, (2) long traces that often tend to be less
understandable, and (3) difficulties in optimising the counterfactual search
with categorical variables. We design an extension of DiCE that can generate
counterfactuals for process predictions, and propose an approach that supports
deriving milestone-aware counterfactuals at different stages of a trace to
promote interpretability. We apply our approach to BPIC2012 event log and the
analysis results demonstrate the effectiveness of the proposed approach.

    

### [[2107.08710] Quantum Deep Learning: Sampling Neural Nets with a Quantum Annealer](http://arxiv.org/abs/2107.08710)


  We demonstrate the feasibility of framing a classically learned deep neural
network as an energy based model that can be processed on a one-step quantum
annealer in order to exploit fast sampling times. We propose approaches to
overcome two hurdles for high resolution image classification on a quantum
processing unit (QPU): the required number and binary nature of the model
states. With this novel method we successfully transfer a convolutional neural
network to the QPU and show the potential for classification speedup of at
least one order of magnitude.

    

### [[2107.08714] CETransformer: Casual Effect Estimation via Transformer Based Representation Learning](http://arxiv.org/abs/2107.08714)


  Treatment effect estimation, which refers to the estimation of causal effects
and aims to measure the strength of the causal relationship, is of great
importance in many fields but is a challenging problem in practice. As present,
data-driven causal effect estimation faces two main challenges, i.e., selection
bias and the missing of counterfactual. To address these two issues, most of
the existing approaches tend to reduce the selection bias by learning a
balanced representation, and then to estimate the counterfactual through the
representation. However, they heavily rely on the finely hand-crafted metric
functions when learning balanced representations, which generally doesn't work
well for the situations where the original distribution is complicated. In this
paper, we propose a CETransformer model for casual effect estimation via
transformer based representation learning. To learn the representation of
covariates(features) robustly, a self-supervised transformer is proposed, by
which the correlation between covariates can be well exploited through
self-attention mechanism. In addition, an adversarial network is adopted to
balance the distribution of the treated and control groups in the
representation space. Experimental results on three real-world datasets
demonstrate the advantages of the proposed CETransformer, compared with the
state-of-the-art treatment effect estimation methods.

    

### [[2107.08721] Stock Movement Prediction with Financial News using Contextualized Embedding from BERT](http://arxiv.org/abs/2107.08721)


  News events can greatly influence equity markets. In this paper, we are
interested in predicting the short-term movement of stock prices after
financial news events using only the headlines of the news. To achieve this
goal, we introduce a new text mining method called Fine-Tuned
Contextualized-Embedding Recurrent Neural Network (FT-CE-RNN). Compared with
previous approaches which use static vector representations of the news (static
embedding), our model uses contextualized vector representations of the
headlines (contextualized embeddings) generated from Bidirectional Encoder
Representations from Transformers (BERT). Our model obtains the
state-of-the-art result on this stock movement prediction task. It shows
significant improvement compared with other baseline models, in both accuracy
and trading simulations. Through various trading simulations based on millions
of headlines from Bloomberg News, we demonstrate the ability of this model in
real scenarios.

    

### [[2107.08751] Adversarial Continual Learning for Multi-Domain Hippocampal Segmentation](http://arxiv.org/abs/2107.08751)


  Deep learning for medical imaging suffers from temporal and privacy-related
restrictions on data availability. To still obtain viable models, continual
learning aims to train in sequential order, as and when data is available. The
main challenge that continual learning methods face is to prevent catastrophic
forgetting, i.e., a decrease in performance on the data encountered earlier.
This issue makes continuous training of segmentation models for medical
applications extremely difficult. Yet, often, data from at least two different
domains is available which we can exploit to train the model in a way that it
disregards domain-specific information. We propose an architecture that
leverages the simultaneous availability of two or more datasets to learn a
disentanglement between the content and domain in an adversarial fashion. The
domain-invariant content representation then lays the base for continual
semantic segmentation. Our approach takes inspiration from domain adaptation
and combines it with continual learning for hippocampal segmentation in brain
MRI. We showcase that our method reduces catastrophic forgetting and
outperforms state-of-the-art continual learning methods.

    

### [[2107.08756] Path Integrals for the Attribution of Model Uncertainties](http://arxiv.org/abs/2107.08756)


  Enabling interpretations of model uncertainties is of key importance in
Bayesian machine learning applications. Often, this requires to meaningfully
attribute predictive uncertainties to source features in an image, text or
categorical array. However, popular attribution methods are particularly
designed for classification and regression scores. In order to explain
uncertainties, state of the art alternatives commonly procure counterfactual
feature vectors, and proceed by making direct comparisons. In this paper, we
leverage path integrals to attribute uncertainties in Bayesian differentiable
models. We present a novel algorithm that relies on in-distribution curves
connecting a feature vector to some counterfactual counterpart, and we retain
desirable properties of interpretability methods. We validate our approach on
benchmark image data sets with varying resolution, and show that it
significantly simplifies interpretability over the existing alternatives.

    

### [[2107.08761] Experimental Investigation and Evaluation of Model-based Hyperparameter Optimization](http://arxiv.org/abs/2107.08761)


  Machine learning algorithms such as random forests or xgboost are gaining
more importance and are increasingly incorporated into production processes in
order to enable comprehensive digitization and, if possible, automation of
processes. Hyperparameters of these algorithms used have to be set
appropriately, which can be referred to as hyperparameter tuning or
optimization. Based on the concept of tunability, this article presents an
overview of theoretical and practical results for popular machine learning
algorithms. This overview is accompanied by an experimental analysis of 30
hyperparameters from six relevant machine learning algorithms. In particular,
it provides (i) a survey of important hyperparameters, (ii) two parameter
tuning studies, and (iii) one extensive global parameter tuning study, as well
as (iv) a new way, based on consensus ranking, to analyze results from multiple
algorithms. The R package mlr is used as a uniform interface to the machine
learning models. The R package SPOT is used to perform the actual tuning
(optimization). All additional code is provided together with this paper.

    

### [[2107.08763] Renyi Differential Privacy of the Subsampled Shuffle Model in Distributed Learning](http://arxiv.org/abs/2107.08763)


  We study privacy in a distributed learning framework, where clients
collaboratively build a learning model iteratively through interactions with a
server from whom we need privacy. Motivated by stochastic optimization and the
federated learning (FL) paradigm, we focus on the case where a small fraction
of data samples are randomly sub-sampled in each round to participate in the
learning process, which also enables privacy amplification. To obtain even
stronger local privacy guarantees, we study this in the shuffle privacy model,
where each client randomizes its response using a local differentially private
(LDP) mechanism and the server only receives a random permutation (shuffle) of
the clients' responses without their association to each client. The principal
result of this paper is a privacy-optimization performance trade-off for
discrete randomization mechanisms in this sub-sampled shuffle privacy model.
This is enabled through a new theoretical technique to analyze the Renyi
Differential Privacy (RDP) of the sub-sampled shuffle model. We numerically
demonstrate that, for important regimes, with composition our bound yields
significant improvement in privacy guarantee over the state-of-the-art
approximate Differential Privacy (DP) guarantee (with strong composition) for
sub-sampled shuffled models. We also demonstrate numerically significant
improvement in privacy-learning performance operating point using real data
sets.

    

### [[2107.08765] Adaptive Transfer Learning on Graph Neural Networks](http://arxiv.org/abs/2107.08765)


  Graph neural networks (GNNs) is widely used to learn a powerful
representation of graph-structured data. Recent work demonstrates that
transferring knowledge from self-supervised tasks to downstream tasks could
further improve graph representation. However, there is an inherent gap between
self-supervised tasks and downstream tasks in terms of optimization objective
and training data. Conventional pre-training methods may be not effective
enough on knowledge transfer since they do not make any adaptation for
downstream tasks. To solve such problems, we propose a new transfer learning
paradigm on GNNs which could effectively leverage self-supervised tasks as
auxiliary tasks to help the target task. Our methods would adaptively select
and combine different auxiliary tasks with the target task in the fine-tuning
stage. We design an adaptive auxiliary loss weighting model to learn the
weights of auxiliary tasks by quantifying the consistency between auxiliary
tasks and the target task. In addition, we learn the weighting model through
meta-learning. Our methods can be applied to various transfer learning
approaches, it performs well not only in multi-task learning but also in
pre-training and fine-tuning. Comprehensive experiments on multiple downstream
tasks demonstrate that the proposed methods can effectively combine auxiliary
tasks with the target task and significantly improve the performance compared
to state-of-the-art methods.

    

### [[2107.08773] Learning Attributed Graph Representations with Communicative Message Passing Transformer](http://arxiv.org/abs/2107.08773)


  Constructing appropriate representations of molecules lies at the core of
numerous tasks such as material science, chemistry and drug designs. Recent
researches abstract molecules as attributed graphs and employ graph neural
networks (GNN) for molecular representation learning, which have made
remarkable achievements in molecular graph modeling. Albeit powerful, current
models either are based on local aggregation operations and thus miss
higher-order graph properties or focus on only node information without fully
using the edge information. For this sake, we propose a Communicative Message
Passing Transformer (CoMPT) neural network to improve the molecular graph
representation by reinforcing message interactions between nodes and edges
based on the Transformer architecture. Unlike the previous transformer-style
GNNs that treat molecules as fully connected graphs, we introduce a message
diffusion mechanism to leverage the graph connectivity inductive bias and
reduce the message enrichment explosion. Extensive experiments demonstrated
that the proposed model obtained superior performances (around 4$\%$ on
average) against state-of-the-art baselines on seven chemical property datasets
(graph-level tasks) and two chemical shift datasets (node-level tasks). Further
visualization studies also indicated a better representation capacity achieved
by our model.

    

### [[2107.08784] Boost-R: Gradient Boosted Trees for Recurrence Data](http://arxiv.org/abs/2107.08784)


  Recurrence data arise from multi-disciplinary domains spanning reliability,
cyber security, healthcare, online retailing, etc. This paper investigates an
additive-tree-based approach, known as Boost-R (Boosting for Recurrence Data),
for recurrent event data with both static and dynamic features. Boost-R
constructs an ensemble of gradient boosted additive trees to estimate the
cumulative intensity function of the recurrent event process, where a new tree
is added to the ensemble by minimizing the regularized L2 distance between the
observed and predicted cumulative intensity. Unlike conventional regression
trees, a time-dependent function is constructed by Boost-R on each tree leaf.
The sum of these functions, from multiple trees, yields the ensemble estimator
of the cumulative intensity. The divide-and-conquer nature of tree-based
methods is appealing when hidden sub-populations exist within a heterogeneous
population. The non-parametric nature of regression trees helps to avoid
parametric assumptions on the complex interactions between event processes and
features. Critical insights and advantages of Boost-R are investigated through
comprehensive numerical examples. Datasets and computer code of Boost-R are
made available on GitHub. To our best knowledge, Boost-R is the first gradient
boosted additive-tree-based approach for modeling large-scale recurrent event
data with both static and dynamic feature information.

    

### [[2107.08785] On Out-of-distribution Detection with Energy-based Models](http://arxiv.org/abs/2107.08785)


  Several density estimation methods have shown to fail to detect
out-of-distribution (OOD) samples by assigning higher likelihoods to anomalous
data. Energy-based models (EBMs) are flexible, unnormalized density models
which seem to be able to improve upon this failure mode. In this work, we
provide an extensive study investigating OOD detection with EBMs trained with
different approaches on tabular and image data and find that EBMs do not
provide consistent advantages. We hypothesize that EBMs do not learn semantic
features despite their discriminative structure similar to Normalizing Flows.
To verify this hypotheses, we show that supervision and architectural
restrictions improve the OOD detection of EBMs independent of the training
approach.

    

### [[2107.08787] The Future will be Different than Today: Model Evaluation Considerations when Developing Translational Clinical Biomarker](http://arxiv.org/abs/2107.08787)


  Finding translational biomarkers stands center stage of the future of
personalized medicine in healthcare. We observed notable challenges in
identifying robust biomarkers as some with great performance in one scenario
often fail to perform well in new trials (e.g. different population,
indications). With rapid development in the clinical trial world (e.g. assay,
disease definition), new trials very likely differ from legacy ones in many
perspectives and in development of biomarkers this heterogeneity should be
considered. In response, we recommend considering building in the heterogeneity
when evaluating biomarkers. In this paper, we present one evaluation strategy
by using leave-one-study-out (LOSO) in place of conventional cross-validation
(cv) methods to account for the potential heterogeneity across trials used for
building and testing the biomarkers. To demonstrate the performance of K-fold
vs LOSO cv in estimating the effect size of biomarkers, we leveraged data from
clinical trials and simulation studies. In our assessment, LOSO cv provided a
more objective estimate of the future performance. This conclusion remained
true across different evaluation metrics and different statistical methods.

    

### [[2107.08790] Anomaly Detection Based on Multiple-Hypothesis Autoencoder](http://arxiv.org/abs/2107.08790)


  Recently Autoencoder(AE) based models are widely used in the field of anomaly
detection. A model trained with normal data generates a larger restoration
error for abnormal data. Whether or not abnormal data is determined by
observing the restoration error. It takes a lot of cost and time to obtain
abnormal data in the industrial field. Therefore the model trains only normal
data and detects abnormal data in the inference phase. However, the restoration
area for the input data of AE is limited in the latent space. To solve this
problem, we propose Multiple-hypothesis Autoencoder(MH-AE) model composed of
several decoders. MH-AE model increases the restoration area through contention
between decoders. The proposed method shows that the anomaly detection
performance is improved compared to the traditional AE for various input
datasets.

    

### [[2107.08792] Selective Focusing Learning in Conditional GANs](http://arxiv.org/abs/2107.08792)


  Conditional generative adversarial networks (cGANs) have demonstrated
remarkable success due to their class-wise controllability and superior quality
for complex generation tasks. Typical cGANs solve the joint distribution
matching problem by decomposing two easier sub-problems: marginal matching and
conditional matching. From our toy experiments, we found that it is the best to
apply only conditional matching to certain samples due to the content-aware
optimization of the discriminator. This paper proposes a simple (a few lines of
code) but effective training methodology, selective focusing learning, which
enforces the discriminator and generator to learn easy samples of each class
rapidly while maintaining diversity. Our key idea is to selectively apply
conditional and joint matching for the data in each mini-batch. We conducted
experiments on recent cGAN variants in ImageNet (64x64 and 128x128), CIFAR-10,
and CIFAR-100 datasets, and improved the performance significantly (up to
35.18% in terms of FID) without sacrificing diversity.

    

### [[2107.08795] Federated Learning with Dynamic Transformer for Text to Speech](http://arxiv.org/abs/2107.08795)


  Text to speech (TTS) is a crucial task for user interaction, but TTS model
training relies on a sizable set of high-quality original datasets. Due to
privacy and security issues, the original datasets are usually unavailable
directly. Recently, federated learning proposes a popular distributed machine
learning paradigm with an enhanced privacy protection mechanism. It offers a
practical and secure framework for data owners to collaborate with others, thus
obtaining a better global model trained on the larger dataset. However, due to
the high complexity of transformer models, the convergence process becomes slow
and unstable in the federated learning setting. Besides, the transformer model
trained in federated learning is costly communication and limited computational
speed on clients, impeding its popularity. To deal with these challenges, we
propose the federated dynamic transformer. On the one hand, the performance is
greatly improved comparing with the federated transformer, approaching
centralize-trained Transformer-TTS when increasing clients number. On the other
hand, it achieves faster and more stable convergence in the training phase and
significantly reduces communication time. Experiments on the LJSpeech dataset
also strongly prove our method's advantage.

    

### [[2107.08800] Deep Learning with Nonsmooth Objectives](http://arxiv.org/abs/2107.08800)


  We explore the potential for using a nonsmooth loss function based on the
max-norm in the training of an artificial neural network. We hypothesise that
this may lead to superior classification results in some special cases where
the training data is either very small or unbalanced.
Our numerical experiments performed on a simple artificial neural network
with no hidden layers (a setting immediately amenable to standard nonsmooth
optimisation techniques) appear to confirm our hypothesis that uniform
approximation based approaches may be more suitable for the datasets with
reliable training data that either is limited size or biased in terms of
relative cluster sizes.

    

### [[2107.08814] MARC: Mining Association Rules from datasets by using Clustering models](http://arxiv.org/abs/2107.08814)


  Association rules are useful to discover relationships, which are mostly
hidden, between the different items in large datasets. Symbolic models are the
principal tools to extract association rules. This basic technique is
time-consuming, and it generates a big number of associated rules. To overcome
this drawback, we suggest a new method, called MARC, to extract the more
important association rules of two important levels: Type I, and Type II. This
approach relies on a multi-topographic unsupervised neural network model as
well as clustering quality measures that evaluate the success of a given
numerical classification model to behave as a natural symbolic model.

    

### [[2107.08815] Boosting the Convergence of Reinforcement Learning-based Auto-pruning Using Historical Data](http://arxiv.org/abs/2107.08815)


  Recently, neural network compression schemes like channel pruning have been
widely used to reduce the model size and computational complexity of deep
neural network (DNN) for applications in power-constrained scenarios such as
embedded systems. Reinforcement learning (RL)-based auto-pruning has been
further proposed to automate the DNN pruning process to avoid expensive
hand-crafted work. However, the RL-based pruner involves a time-consuming
training process and the high expense of each sample further exacerbates this
problem. These impediments have greatly restricted the real-world application
of RL-based auto-pruning. Thus, in this paper, we propose an efficient
auto-pruning framework which solves this problem by taking advantage of the
historical data from the previous auto-pruning process. In our framework, we
first boost the convergence of the RL-pruner by transfer learning. Then, an
augmented transfer learning scheme is proposed to further speed up the training
process by improving the transferability. Finally, an assistant learning
process is proposed to improve the sample efficiency of the RL agent. The
experiments have shown that our framework can accelerate the auto-pruning
process by 1.5-2.5 times for ResNet20, and 1.81-2.375 times for other neural
networks like ResNet56, ResNet18, and MobileNet v1.

    

### [[2107.08819] Model-free prediction of emergence of extreme events in a parametrically driven nonlinear dynamical system by Deep Learning](http://arxiv.org/abs/2107.08819)


  We predict the emergence of extreme events in a parametrically driven
nonlinear dynamical system using three Deep Learning models, namely Multi-Layer
Perceptron, Convolutional Neural Network and Long Short-Term Memory. The Deep
Learning models are trained using the training set and are allowed to predict
the test set data. After prediction, the time series of the actual and the
predicted values are plotted one over the other in order to visualize the
performance of the models. Upon evaluating the Root Mean Square Error value
between predicted and the actual values of all three models, we find that the
Long Short-Term Memory model can serve as the best model to forecast the
chaotic time series and to predict the emergence of extreme events for the
considered system.

    

### [[2107.08821] Proceedings of ICML 2021 Workshop on Theoretic Foundation, Criticism, and Application Trend of Explainable AI](http://arxiv.org/abs/2107.08821)


  This is the Proceedings of ICML 2021 Workshop on Theoretic Foundation,
Criticism, and Application Trend of Explainable AI. Deep neural networks (DNNs)
have undoubtedly brought great success to a wide range of applications in
computer vision, computational linguistics, and AI. However, foundational
principles underlying the DNNs' success and their resilience to adversarial
attacks are still largely missing. Interpreting and theorizing the internal
mechanisms of DNNs becomes a compelling yet controversial topic. This workshop
pays a special interest in theoretic foundations, limitations, and new
application trends in the scope of XAI. These issues reflect new bottlenecks in
the future development of XAI.

    

### [[2107.08823] One-Class Classification for Wafer Map using Adversarial Autoencoder with DSVDD Prior](http://arxiv.org/abs/2107.08823)


  Recently, semiconductors' demand has exploded in virtual reality,
smartphones, wearable devices, the internet of things, robotics, and
automobiles. Semiconductor manufacturers want to make semiconductors with high
yields. To do this, manufacturers conduct many quality assurance activities.
Wafer map pattern classification is a typical way of quality assurance. The
defect pattern on the wafer map can tell us which process has a problem. Most
of the existing wafer map classification methods are based on supervised
methods. The supervised methods tend to have high performance, but they require
extensive labor and expert knowledge to produce labeled datasets with a
balanced distribution in mind. In the semiconductor manufacturing process, it
is challenging to get defect data with balanced distribution. In this paper, we
propose a one-class classification method using an Adversarial Autoencoder
(AAE) with Deep Support Vector Data Description (DSVDD) prior, which generates
random vectors within the hypersphere of DSVDD. We use the WM-811k dataset,
which consists of a real-world wafer map. We compare the F1 score performance
of our model with DSVDD and AAE.

    

### [[2107.08828] Reinforcement Learning for Education: Opportunities and Challenges](http://arxiv.org/abs/2107.08828)


  This survey article has grown out of the RL4ED workshop organized by the
authors at the Educational Data Mining (EDM) 2021 conference. We organized this
workshop as part of a community-building effort to bring together researchers
and practitioners interested in the broad areas of reinforcement learning (RL)
and education (ED). This article aims to provide an overview of the workshop
activities and summarize the main research directions in the area of RL for ED.

    

### [[2107.08829] Visual Adversarial Imitation Learning using Variational Models](http://arxiv.org/abs/2107.08829)


  Reward function specification, which requires considerable human effort and
iteration, remains a major impediment for learning behaviors through deep
reinforcement learning. In contrast, providing visual demonstrations of desired
behaviors often presents an easier and more natural way to teach agents. We
consider a setting where an agent is provided a fixed dataset of visual
demonstrations illustrating how to perform a task, and must learn to solve the
task using the provided demonstrations and unsupervised environment
interactions. This setting presents a number of challenges including
representation learning for visual observations, sample complexity due to high
dimensional spaces, and learning instability due to the lack of a fixed reward
or learning signal. Towards addressing these challenges, we develop a
variational model-based adversarial imitation learning (V-MAIL) algorithm. The
model-based approach provides a strong signal for representation learning,
enables sample efficiency, and improves the stability of adversarial training
by enabling on-policy learning. Through experiments involving several
vision-based locomotion and manipulation tasks, we find that V-MAIL learns
successful visuomotor policies in a sample-efficient manner, has better
stability compared to prior work, and also achieves higher asymptotic
performance. We further find that by transferring the learned models, V-MAIL
can learn new tasks from visual demonstrations without any additional
environment interactions. All results including videos can be found online at
\url{this https URL}.

    

### [[2107.08834] A Multi-UAV System for Exploration and Target Finding in Cluttered and GPS-Denied Environments](http://arxiv.org/abs/2107.08834)


  The use of multi-rotor Unmanned Aerial Vehicles (UAVs) for search and rescue
as well as remote sensing is rapidly increasing. Multi-rotor UAVs, however,
have limited endurance. The range of UAV applications can be widened if teams
of multiple UAVs are used. We propose a framework for a team of UAVs to
cooperatively explore and find a target in complex GPS-denied environments with
obstacles. The team of UAVs autonomously navigates, explores, detects, and
finds the target in a cluttered environment with a known map. Examples of such
environments include indoor scenarios, urban or natural canyons, caves, and
tunnels, where the GPS signal is limited or blocked. The framework is based on
a probabilistic decentralised Partially Observable Markov Decision Process
which accounts for the uncertainties in sensing and the environment. The team
can cooperate efficiently, with each UAV sharing only limited processed
observations and their locations during the mission. The system is simulated
using the Robotic Operating System and Gazebo. Performance of the system with
an increasing number of UAVs in several indoor scenarios with obstacles is
tested. Results indicate that the proposed multi-UAV system has improvements in
terms of time-cost, the proportion of search area surveyed, as well as
successful rates for search and rescue missions.

    

### [[2107.08835] Time Series Anomaly Detection for Smart Grids: A Survey](http://arxiv.org/abs/2107.08835)


  With the rapid increase in the integration of renewable energy generation and
the wide adoption of various electric appliances, power grids are now faced
with more and more challenges. One prominent challenge is to implement
efficient anomaly detection for different types of anomalous behaviors within
power grids. These anomalous behaviors might be induced by unusual consumption
patterns of the users, faulty grid infrastructures, outages, external
cyberattacks, or energy fraud. Identifying such anomalies is of critical
importance for the reliable and efficient operation of modern power grids.
Various methods have been proposed for anomaly detection on power grid
time-series data. This paper presents a short survey of the recent advances in
anomaly detection for power grid time-series data. Specifically, we first
outline current research challenges in the power grid anomaly detection domain
and further review the major anomaly detection approaches. Finally, we conclude
the survey by identifying the potential directions for future research.

    

### [[2107.08849] Exploring the efficacy of neural networks for trajectory compression and the inverse problem](http://arxiv.org/abs/2107.08849)


  In this document, a neural network is employed in order to estimate the
solution of the initial value problem in the context of non linear
trajectories. Such trajectories can be subject to gravity, thrust, drag,
centrifugal force, temperature, ambient air density and pressure. First, we
generate a grid of trajectory points given a specified uniform density as a
design parameter and then we investigate the performance of a neural network in
a compression and inverse problem task: the network is trained to predict the
initial conditions of the dynamics model we used in the simulation, given a
target point in space. We investigate this as a regression task, with error
propagation in consideration. For target points, up to a radius of 2
kilometers, the model is able to accurately predict the initial conditions of
the trajectories, with sub-meter deviation. This simulation-based training
process and novel real-world evaluation method is capable of computing
trajectories of arbitrary dimensions.

    

### [[2107.08850] Automatic and explainable grading of meningiomas from histopathology images](http://arxiv.org/abs/2107.08850)


  Meningioma is one of the most prevalent brain tumors in adults. To determine
its malignancy, it is graded by a pathologist into three grades according to
WHO standards. This grade plays a decisive role in treatment, and yet may be
subject to inter-rater discordance. In this work, we present and compare three
approaches towards fully automatic meningioma grading from histology whole
slide images. All approaches are following a two-stage paradigm, where we first
identify a region of interest based on the detection of mitotic figures in the
slide using a state-of-the-art object detection deep learning network. This
region of highest mitotic rate is considered characteristic for biological
tumor behavior. In the second stage, we calculate a score corresponding to
tumor malignancy based on information contained in this region using three
different settings. In a first approach, image patches are sampled from this
region and regression is based on morphological features encoded by a
ResNet-based network. We compare this to learning a logistic regression from
the determined mitotic count, an approach which is easily traceable and
explainable. Lastly, we combine both approaches in a single network. We trained
the pipeline on 951 slides from 341 patients and evaluated them on a separate
set of 141 slides from 43 patients. All approaches yield a high correlation to
the WHO grade. The logistic regression and the combined approach had the best
results in our experiments, yielding correct predictions in 32 and 33 of all
cases, respectively, with the image-based approach only predicting 25 cases
correctly. Spearman's correlation was 0.716, 0.792 and 0.790 respectively. It
may seem counterintuitive at first that morphological features provided by
image patches do not improve model performance. Yet, this mirrors the criteria
of the grading scheme, where mitotic count is the only unequivocal parameter.

    

### [[2107.08861] VolcanoML: Speeding up End-to-End AutoML via Scalable Search Space Decomposition](http://arxiv.org/abs/2107.08861)


  End-to-end AutoML has attracted intensive interests from both academia and
industry, which automatically searches for ML pipelines in a space induced by
feature engineering, algorithm/model selection, and hyper-parameter tuning.
Existing AutoML systems, however, suffer from scalability issues when applying
to application domains with large, high-dimensional search spaces. We present
VolcanoML, a scalable and extensible framework that facilitates systematic
exploration of large AutoML search spaces. VolcanoML introduces and implements
basic building blocks that decompose a large search space into smaller ones,
and allows users to utilize these building blocks to compose an execution plan
for the AutoML problem at hand. VolcanoML further supports a Volcano-style
execution model - akin to the one supported by modern database systems - to
execute the plan constructed. Our evaluation demonstrates that, not only does
VolcanoML raise the level of expressiveness for search space decomposition in
AutoML, it also leads to actual findings of decomposition strategies that are
significantly more efficient than the ones employed by state-of-the-art AutoML
systems such as auto-sklearn.

    

### [[2107.08865] Ab Initio Particle-based Object Manipulation](http://arxiv.org/abs/2107.08865)


  This paper presents Particle-based Object Manipulation (Prompt), a new
approach to robot manipulation of novel objects ab initio, without prior object
models or pre-training on a large object data set. The key element of Prompt is
a particle-based object representation, in which each particle represents a
point in the object, the local geometric, physical, and other features of the
point, and also its relation with other particles. Like the model-based
analytic approaches to manipulation, the particle representation enables the
robot to reason about the object's geometry and dynamics in order to choose
suitable manipulation actions. Like the data-driven approaches, the particle
representation is learned online in real-time from visual sensor input,
specifically, multi-view RGB images. The particle representation thus connects
visual perception with robot control. Prompt combines the benefits of both
model-based reasoning and data-driven learning. We show empirically that Prompt
successfully handles a variety of everyday objects, some of which are
transparent. It handles various manipulation tasks, including grasping,
pushing, etc,. Our experiments also show that Prompt outperforms a
state-of-the-art data-driven grasping method on the daily objects, even though
it does not use any offline training data.

    

### [[2107.08873] RingFed: Reducing Communication Costs in Federated Learning on Non-IID Data](http://arxiv.org/abs/2107.08873)


  Federated learning is a widely used distributed deep learning framework that
protects the privacy of each client by exchanging model parameters rather than
raw data. However, federated learning suffers from high communication costs, as
a considerable number of model parameters need to be transmitted many times
during the training process, making the approach inefficient, especially when
the communication network bandwidth is limited. This article proposes RingFed,
a novel framework to reduce communication overhead during the training process
of federated learning. Rather than transmitting parameters between the center
server and each client, as in original federated learning, in the proposed
RingFed, the updated parameters are transmitted between each client in turn,
and only the final result is transmitted to the central server, thereby
reducing the communication overhead substantially. After several local updates,
clients first send their parameters to another proximal client, not to the
center server directly, to preaggregate. Experiments on two different public
datasets show that RingFed has fast convergence, high model accuracy, and low
communication cost.

    

### [[2107.08881] Reasoning-Modulated Representations](http://arxiv.org/abs/2107.08881)


  Neural networks leverage robust internal representations in order to
generalise. Learning them is difficult, and often requires a large training set
that covers the data distribution densely. We study a common setting where our
task is not purely opaque. Indeed, very often we may have access to information
about the underlying system (e.g. that observations must obey certain laws of
physics) that any "tabula rasa" neural network would need to re-learn from
scratch, penalising data efficiency. We incorporate this information into a
pre-trained reasoning module, and investigate its role in shaping the
discovered representations in diverse self-supervised learning settings from
pixels. Our approach paves the way for a new class of data-efficient
representation learning.

    

### [[2107.08888] Multimodal Reward Shaping for Efficient Exploration in Reinforcement Learning](http://arxiv.org/abs/2107.08888)


  Maintaining long-term exploration ability remains one of the challenges of
deep reinforcement learning (DRL). In practice, the reward shaping-based
approaches are leveraged to provide intrinsic rewards for the agent to
incentivize motivation. However, most existing IRS modules rely on attendant
models or additional memory to record and analyze learning procedures, which
leads to high computational complexity and low robustness. Moreover, they
overemphasize the influence of a single state on exploration, which cannot
evaluate the exploration performance from a global perspective. To tackle the
problem, state entropy-based methods are proposed to encourage the agent to
visit the state space more equitably. However, the estimation error and sample
complexity are prohibitive when handling environments with high-dimensional
observation. In this paper, we introduce a novel metric entitled Jain's
fairness index (JFI) to replace the entropy regularizer, which requires no
additional models or memory. In particular, JFI overcomes the vanishing
intrinsic rewards problem and can be generalized into arbitrary tasks.
Furthermore, we use a variational auto-encoder (VAE) model to capture the
life-long novelty of states. Finally, the global JFI score and local state
novelty are combined to form a multimodal intrinsic reward, controlling the
exploration extent more precisely. Finally, extensive simulation results
demonstrate that our multimodal reward shaping (MMRS) method can achieve higher
performance in contrast to other benchmark schemes.

    

### [[2107.08902] Analysing Cyberbullying using Natural Language Processing by Understanding Jargon in Social Media](http://arxiv.org/abs/2107.08902)


  Cyberbullying is of extreme prevalence today. Online-hate comments, toxicity,
cyberbullying amongst children and other vulnerable groups are only growing
over online classes, and increased access to social platforms, especially post
COVID-19. It is paramount to detect and ensure minors' safety across social
platforms so that any violence or hate-crime is automatically detected and
strict action is taken against it. In our work, we explore binary
classification by using a combination of datasets from various social media
platforms that cover a wide range of cyberbullying such as sexism, racism,
abusive, and hate-speech. We experiment through multiple models such as
Bi-LSTM, GloVe, state-of-the-art models like BERT, and apply a unique
preprocessing technique by introducing a slang-abusive corpus, achieving a
higher precision in comparison to models without slang preprocessing.

    

### [[2107.08909] MEGEX: Data-Free Model Extraction Attack against Gradient-Based Explainable AI](http://arxiv.org/abs/2107.08909)


  The advance of explainable artificial intelligence, which provides reasons
for its predictions, is expected to accelerate the use of deep neural networks
in the real world like Machine Learning as a Service (MLaaS) that returns
predictions on queried data with the trained model. Deep neural networks
deployed in MLaaS face the threat of model extraction attacks. A model
extraction attack is an attack to violate intellectual property and privacy in
which an adversary steals trained models in a cloud using only their
predictions. In particular, a data-free model extraction attack has been
proposed recently and is more critical. In this attack, an adversary uses a
generative model instead of preparing input data. The feasibility of this
attack, however, needs to be studied since it requires more queries than that
with surrogate datasets. In this paper, we propose MEGEX, a data-free model
extraction attack against a gradient-based explainable AI. In this method, an
adversary uses the explanations to train the generative model and reduces the
number of queries to steal the model. Our experiments show that our proposed
method reconstructs high-accuracy models -- 0.97$\times$ and 0.98$\times$ the
victim model accuracy on SVHN and CIFAR-10 datasets given 2M and 20M queries,
respectively. This implies that there is a trade-off between the
interpretability of models and the difficulty of stealing them.

    

### [[2107.08924] Epistemic Neural Networks](http://arxiv.org/abs/2107.08924)


  We introduce the \textit{epistemic neural network} (ENN) as an interface for
uncertainty modeling in deep learning. All existing approaches to uncertainty
modeling can be expressed as ENNs, and any ENN can be identified with a
Bayesian neural network. However, this new perspective provides several
promising directions for future research. Where prior work has developed
probabilistic inference tools for neural networks; we ask instead, `which
neural networks are suitable as tools for probabilistic inference?'. We propose
a clear and simple metric for progress in ENNs: the KL-divergence with respect
to a target distribution. We develop a computational testbed based on inference
in a neural network Gaussian process and release our code as a benchmark at
\url{this https URL}. We evaluate several canonical approaches
to uncertainty modeling in deep learning, and find they vary greatly in their
performance. We provide insight to the sensitivity of these results and show
that our metric is highly correlated with performance in sequential decision
problems. Finally, we provide indications that new ENN architectures can
improve performance in both the statistical quality and computational cost.

    

### [[2107.08925] Estimating covariant Lyapunov vectors from data](http://arxiv.org/abs/2107.08925)


  Covariant Lyapunov vectors (CLVs) characterize the directions along which
perturbations in dynamical systems grow. They have also been studied as
potential predictors of critical transitions and extreme events. For many
applications, it is, however, necessary to estimate the vectors from data since
model equations are unknown for many interesting phenomena. We propose a novel
method for estimating CLVs based on data records without knowing the underlying
equations of the system which is suitable also for high-dimensional data and
computationally inexpensive. We demonstrate that this purely data-driven
approach can accurately estimate CLVs from data records generated by chaotic
dynamical systems of dimension 128 and multiple lower-dimensional systems and
thus provides the foundation for numerous future applications in data-analysis
and data-based predictions.

    

### [[2107.08928] Introducing a Family of Synthetic Datasets for Research on Bias in Machine Learning](http://arxiv.org/abs/2107.08928)


  A significant impediment to progress in research on bias in machine learning
(ML) is the availability of relevant datasets. This situation is unlikely to
change much given the sensitivity of such data. For this reason, there is a
role for synthetic data in this research. In this short paper, we present one
such family of synthetic data sets. We provide an overview of the data,
describe how the level of bias can be varied, and present a simple example of
an experiment on the data.

    

### [[2107.08933] Over-Parameterization and Generalization in Audio Classification](http://arxiv.org/abs/2107.08933)


  Convolutional Neural Networks (CNNs) have been dominating classification
tasks in various domains, such as machine vision, machine listening, and
natural language processing. In machine listening, while generally exhibiting
very good generalization capabilities, CNNs are sensitive to the specific audio
recording device used, which has been recognized as a substantial problem in
the acoustic scene classification (DCASE) community. In this study, we
investigate the relationship between over-parameterization of acoustic scene
classification models, and their resulting generalization abilities.
Specifically, we test scaling CNNs in width and depth, under different
conditions. Our results indicate that increasing width improves generalization
to unseen devices, even without an increase in the number of parameters.

    

### [[1705.07164] Relaxed Wasserstein with Applications to GANs](http://arxiv.org/abs/1705.07164)


  Wasserstein Generative Adversarial Networks (WGANs) provide a versatile class
of models, which have attracted great attention in various applications.
However, this framework has two main drawbacks: (i) Wasserstein-1 (or
Earth-Mover) distance is restrictive such that WGANs cannot always fit data
geometry well; (ii) It is difficult to achieve fast training of WGANs. In this
paper, we propose a new class of \textit{Relaxed Wasserstein} (RW) distances by
generalizing Wasserstein-1 distance with Bregman cost functions. We show that
RW distances achieve nice statistical properties while not sacrificing the
computational tractability. Combined with the GANs framework, we develop
Relaxed WGANs (RWGANs) which are not only statistically flexible but can be
approximated efficiently using heuristic approaches. Experiments on real images
demonstrate that the RWGAN with Kullback-Leibler (KL) cost function outperforms
other competing approaches, e.g., WGANs, even with gradient penalty.

    

### [[1810.03024] Learning to Optimize under Non-Stationarity](http://arxiv.org/abs/1810.03024)


  We introduce algorithms that achieve state-of-the-art \emph{dynamic regret}
bounds for non-stationary linear stochastic bandit setting. It captures natural
applications such as dynamic pricing and ads allocation in a changing
environment. We show how the difficulty posed by the non-stationarity can be
overcome by a novel marriage between stochastic and adversarial bandits
learning algorithms. Defining $d,B_T,$ and $T$ as the problem dimension, the
\emph{variation budget}, and the total time horizon, respectively, our main
contributions are the tuned Sliding Window UCB (\texttt{SW-UCB}) algorithm with
optimal $\widetilde{O}(d^{2/3}(B_T+1)^{1/3}T^{2/3})$ dynamic regret, and the
tuning free bandit-over-bandit (\texttt{BOB}) framework built on top of the
\texttt{SW-UCB} algorithm with best
$\widetilde{O}(d^{2/3}(B_T+1)^{1/4}T^{3/4})$ dynamic regret.

    

### [[1910.06070] Review of Learning-based Longitudinal Motion Planning for Autonomous Vehicles: Research Gaps between Self-driving and Traffic Congestion](http://arxiv.org/abs/1910.06070)


  Self-driving technology companies and the research community are accelerating
their pace to use machine learning longitudinal motion planning (mMP) for
autonomous vehicles (AVs). This paper reviews the current state of the art in
mMP, with an exclusive focus on its impact on traffic congestion. We identify
the availability of congestion scenarios in current datasets, and summarize the
required features for training mMP. For learning methods, we survey the major
methods in both imitation learning and non-imitation learning. We also
highlight the emerging technologies adopted by some leading AV companies, e.g.
Tesla, Waymo, and this http URL. We find that: i) the AV industry has been mostly
focusing on the long tail problem related to safety and overlooked the impact
on traffic congestion, ii) the current public self-driving datasets have not
included enough congestion scenarios, and mostly lack the necessary input
features/output labels to train mMP, and iii) albeit reinforcement learning
(RL) approach can integrate congestion mitigation into the learning goal, the
major mMP method adopted by industry is still behavior cloning (BC), whose
capability to learn a congestion-mitigating mMP remains to be seen. Based on
the review, the study identifies the research gaps in current mMP development.
Some suggestions towards congestion mitigation for future mMP studies are
proposed: i) enrich data collection to facilitate the congestion learning, ii)
incorporate non-imitation learning methods to combine traffic efficiency into a
safety-oriented technical route, and iii) integrate domain knowledge from the
traditional car following (CF) theory to improve the string stability of mMP.

    

### [[1910.09739] Composite Neural Network: Theory and Application to PM2.5 Prediction](http://arxiv.org/abs/1910.09739)


  This work investigates the framework and performance issues of the composite
neural network, which is composed of a collection of pre-trained and
non-instantiated neural network models connected as a rooted directed acyclic
graph for solving complicated applications. A pre-trained neural network model
is generally well trained, targeted to approximate a specific function. Despite
a general belief that a composite neural network may perform better than a
single component, the overall performance characteristics are not clear. In
this work, we construct the framework of a composite network, and prove that a
composite neural network performs better than any of its pre-trained components
with a high probability bound. In addition, if an extra pre-trained component
is added to a composite network, with high probability, the overall performance
will not be degraded. In the study, we explore a complicated application --
PM2.5 prediction -- to illustrate the correctness of the proposed composite
network theory. In the empirical evaluations of PM2.5 prediction, the
constructed composite neural network models support the proposed theory and
perform better than other machine learning models, demonstrate the advantages
of the proposed framework.

    

### [[1911.05461] On the Complexity of Labeled Datasets](http://arxiv.org/abs/1911.05461)


  The Statistical Learning Theory (SLT) provides the foundation to ensure that
a supervised algorithm generalizes the mapping $f: \mathcal{X} \to \mathcal{Y}$
given $f$ is selected from its search space bias $\mathcal{F}$. SLT depends on
the Shattering coefficient function $\mathcal{N}(\mathcal{F},n)$ to upper bound
the empirical risk minimization principle, from which one can estimate the
necessary training sample size to ensure the probabilistic learning convergence
and, most importantly, the characterization of the capacity of $\mathcal{F}$,
including its underfitting and overfitting abilities while addressing specific
target problems. However, the analytical solution of the Shattering coefficient
is still an open problem since the first studies by Vapnik and Chervonenkis in
$1962$, which we address on specific datasets, in this paper, by employing
equivalence relations from Topology, data separability results by Har-Peled and
Jones, and combinatorics. Our approach computes the Shattering coefficient for
both binary and multi-class datasets, leading to the following additional
contributions: (i) the estimation of the required number of hyperplanes in the
worst and best-case classification scenarios and the respective $\Omega$ and
$O$ complexities; (ii) the estimation of the training sample sizes required to
ensure supervised learning; and (iii) the comparison of dataset embeddings,
once they (re)organize samples into some new space configuration. All results
introduced and discussed along this paper are supported by the R package
shattering (this https URL).

    

### [[2002.06470] Pitfalls of In-Domain Uncertainty Estimation and Ensembling in Deep Learning](http://arxiv.org/abs/2002.06470)


  Uncertainty estimation and ensembling methods go hand-in-hand. Uncertainty
estimation is one of the main benchmarks for assessment of ensembling
performance. At the same time, deep learning ensembles have provided
state-of-the-art results in uncertainty estimation. In this work, we focus on
in-domain uncertainty for image classification. We explore the standards for
its quantification and point out pitfalls of existing metrics. Avoiding these
pitfalls, we perform a broad study of different ensembling techniques. To
provide more insight in this study, we introduce the deep ensemble equivalent
score (DEE) and show that many sophisticated ensembling techniques are
equivalent to an ensemble of only few independently trained networks in terms
of test performance.

    

### [[2002.09478] On the Search for Feedback in Reinforcement Learning](http://arxiv.org/abs/2002.09478)


  The problem of Reinforcement Learning (RL) in an unknown nonlinear dynamical
system is equivalent to the search for an optimal feedback law utilizing the
simulations/ rollouts of the unknown dynamical system. Most RL techniques
search over a complex global nonlinear feedback parametrization making them
suffer from high training times as well as variance. Instead, we advocate
searching over a local feedback representation consisting of an open-loop
sequence, and an associated optimal linear feedback law completely determined
by the open-loop. We show that this alternate approach results in highly
efficient training, the answers obtained are repeatable and hence reliable, and
the resulting closed performance is superior to global state-of-the-art RL
techniques. Finally, if we replan, whenever required, which is feasible due to
the fast and reliable local solution, allows us to recover global optimality of
the resulting feedback law.

    

### [[2003.13955] Differentially Private Naive Bayes Classifier using Smooth Sensitivity](http://arxiv.org/abs/2003.13955)


  With the increasing collection of users' data, protecting individual privacy
has gained more interest. Differential Privacy is a strong concept of
protecting individuals. Naive Bayes is one of the popular machine learning
algorithm, used as a baseline for many tasks. In this work, we have provided a
differentially private Naive Bayes classifier that adds noise proportional to
the Smooth Sensitivity of its parameters. We have compared our result to
Vaidya, Shafiq, Basu, and Hong in which they have scaled the noise to the
global sensitivity of the parameters. Our experiment results on the real-world
datasets show that the accuracy of our method has improved significantly while
still preserving $\varepsilon$-differential privacy.

    

### [[2004.05094] Encoder blind combinatorial compressed sensing](http://arxiv.org/abs/2004.05094)


  In its most elementary form, compressed sensing studies the design of
decoding algorithms to recover a sufficiently sparse vector or code from a
lower dimensional linear measurement vector. Typically it is assumed that the
decoder has access to the encoder matrix, which in the combinatorial case is
sparse and binary. In this paper we consider the problem of designing a decoder
to recover a set of sparse codes from their linear measurements alone, that is
without access to encoder matrix. To this end we study the matrix factorisation
task of recovering both the encoder and sparse coding matrices from the
associated linear measurement matrix. The contribution of this paper is a
computationally efficient decoding algorithm, Decoder-Expander Based
Factorisation, with strong performance guarantees. In particular, under mild
assumptions on the sparse coding matrix and by deploying a novel random encoder
matrix, we prove that Decoder-Expander Based Factorisation recovers both the
encoder and sparse coding matrix at the optimal measurement rate with high
probability and from a near optimal number of measurement vectors. In addition,
our experiments demonstrate the efficacy and computational efficiency of our
algorithm in practice. Beyond compressed sensing our results may be of interest
for researchers working in areas such as linear sketching, coding theory and
matrix compression.

    

### [[2005.02463] Spatio-Temporal Event Segmentation and Localization for Wildlife Extended Videos](http://arxiv.org/abs/2005.02463)


  Using offline training schemes, researchers have tackled the event
segmentation problem by providing full or weak-supervision through manually
annotated labels or self-supervised epoch-based training. Most works consider
videos that are at most 10's of minutes long. We present a self-supervised
perceptual prediction framework capable of temporal event segmentation by
building stable representations of objects over time and demonstrate it on long
videos, spanning several days. The approach is deceptively simple but quite
effective. We rely on predictions of high-level features computed by a standard
deep learning backbone. For prediction, we use an LSTM, augmented with an
attention mechanism, trained in a self-supervised manner using the prediction
error. The self-learned attention maps effectively localize and track the
event-related objects in each frame. The proposed approach does not require
labels. It requires only a single pass through the video, with no separate
training set. Given the lack of datasets of very long videos, we demonstrate
our method on video from 10 days (254 hours) of continuous wildlife monitoring
data that we had collected with required permissions. We find that the approach
is robust to various environmental conditions such as day/night conditions,
rain, sharp shadows, and windy conditions. For the task of temporally locating
events, we had an 80% recall rate at 20% false-positive rate for frame-level
segmentation. At the activity level, we had an 80% activity recall rate for one
false activity detection every 50 minutes. We will make the dataset, which is
the first of its kind, and the code available to the research community.

    

### [[2006.06979] Non-Negative Bregman Divergence Minimization for Deep Direct Density Ratio Estimation](http://arxiv.org/abs/2006.06979)


  Density ratio estimation (DRE) is at the core of various machine learning
tasks such as anomaly detection and domain adaptation. In existing studies on
DRE, methods based on Bregman divergence (BD) minimization have been
extensively studied. However, BD minimization when applied with highly flexible
models, such as deep neural networks, tends to suffer from what we call
train-loss hacking, which is a source of overfitting caused by a typical
characteristic of empirical BD estimators. In this paper, to mitigate
train-loss hacking, we propose a non-negative correction for empirical BD
estimators. Theoretically, we confirm the soundness of the proposed method
through a generalization error bound. Through our experiments, the proposed
methods show a favorable performance in inlier-based outlier detection.

    

### [[2006.07458] Projection Robust Wasserstein Distance and Riemannian Optimization](http://arxiv.org/abs/2006.07458)


  Projection robust Wasserstein (PRW) distance, or Wasserstein projection
pursuit (WPP), is a robust variant of the Wasserstein distance. Recent work
suggests that this quantity is more robust than the standard Wasserstein
distance, in particular when comparing probability measures in high-dimensions.
However, it is ruled out for practical application because the optimization
model is essentially non-convex and non-smooth which makes the computation
intractable. Our contribution in this paper is to revisit the original
motivation behind WPP/PRW, but take the hard route of showing that, despite its
non-convexity and lack of nonsmoothness, and even despite some hardness results
proved by~\citet{Niles-2019-Estimation} in a minimax sense, the original
formulation for PRW/WPP \textit{can} be efficiently computed in practice using
Riemannian optimization, yielding in relevant cases better behavior than its
convex relaxation. More specifically, we provide three simple algorithms with
solid theoretical guarantee on their complexity bound (one in the appendix),
and demonstrate their effectiveness and efficiency by conducing extensive
experiments on synthetic and real data. This paper provides a first step into a
computational theory of the PRW distance and provides the links between optimal
transport and Riemannian optimization.

    

### [[2006.08170] MetaCURE: Meta Reinforcement Learning with Empowerment-Driven Exploration](http://arxiv.org/abs/2006.08170)


  Meta reinforcement learning (meta-RL) extracts knowledge from previous tasks
and achieves fast adaptation to new tasks. Despite recent progress, efficient
exploration in meta-RL remains a key challenge in sparse-reward tasks, as it
requires quickly finding informative task-relevant experiences in both
meta-training and adaptation. To address this challenge, we explicitly model an
exploration policy learning problem for meta-RL, which is separated from
exploitation policy learning, and introduce a novel empowerment-driven
exploration objective, which aims to maximize information gain for task
identification. We derive a corresponding intrinsic reward and develop a new
off-policy meta-RL framework, which efficiently learns separate context-aware
exploration and exploitation policies by sharing the knowledge of task
inference. Experimental evaluation shows that our meta-RL method significantly
outperforms state-of-the-art baselines on various sparse-reward MuJoCo
locomotion tasks and more complex sparse-reward Meta-World tasks.

    

### [[2006.10483] Algorithmic Decision Making with Conditional Fairness](http://arxiv.org/abs/2006.10483)


  Nowadays fairness issues have raised great concerns in decision-making
systems. Various fairness notions have been proposed to measure the degree to
which an algorithm is unfair. In practice, there frequently exist a certain set
of variables we term as fair variables, which are pre-decision covariates such
as users' choices. The effects of fair variables are irrelevant in assessing
the fairness of the decision support algorithm. We thus define conditional
fairness as a more sound fairness metric by conditioning on the fairness
variables. Given different prior knowledge of fair variables, we demonstrate
that traditional fairness notations, such as demographic parity and equalized
odds, are special cases of our conditional fairness notations. Moreover, we
propose a Derivable Conditional Fairness Regularizer (DCFR), which can be
integrated into any decision-making model, to track the trade-off between
precision and fairness of algorithmic decision making. Specifically, an
adversarial representation based conditional independence loss is proposed in
our DCFR to measure the degree of unfairness. With extensive experiments on
three real-world datasets, we demonstrate the advantages of our conditional
fairness notation and DCFR.

    

### [[2006.12301] On Projection Robust Optimal Transport: Sample Complexity and Model Misspecification](http://arxiv.org/abs/2006.12301)


  Optimal transport (OT) distances are increasingly used as loss functions for
statistical inference, notably in the learning of generative models or
supervised learning. Yet, the behavior of minimum Wasserstein estimators is
poorly understood, notably in high-dimensional regimes or under model
misspecification. In this work we adopt the viewpoint of projection robust (PR)
OT, which seeks to maximize the OT cost between two measures by choosing a
$k$-dimensional subspace onto which they can be projected. Our first
contribution is to establish several fundamental statistical properties of PR
Wasserstein distances, complementing and improving previous literature that has
been restricted to one-dimensional and well-specified cases. Next, we propose
the integral PR Wasserstein (IPRW) distance as an alternative to the PRW
distance, by averaging rather than optimizing on subspaces. Our complexity
bounds can help explain why both PRW and IPRW distances outperform Wasserstein
distances empirically in high-dimensional inference tasks. Finally, we consider
parametric inference using the PRW distance. We provide an asymptotic guarantee
of two types of minimum PRW estimators and formulate a central limit theorem
for max-sliced Wasserstein estimator under model misspecification. To enable
our analysis on PRW with projection dimension larger than one, we devise a
novel combination of variational analysis and statistical theory.

    

### [[2007.02443] Pseudo-Rehearsal for Continual Learning with Normalizing Flows](http://arxiv.org/abs/2007.02443)


  Catastrophic forgetting (CF) happens whenever a neural network overwrites
past knowledge while being trained on new tasks. Common techniques to handle CF
include regularization of the weights (using, e.g., their importance on past
tasks), and rehearsal strategies, where the network is constantly re-trained on
past data. Generative models have also been applied for the latter, in order to
have endless sources of data. In this paper, we propose a novel method that
combines the strengths of regularization and generative-based rehearsal
approaches. Our generative model consists of a normalizing flow (NF), a
probabilistic and invertible neural network, trained on the internal embeddings
of the network. By keeping a single NF conditioned on the task, we show that
our memory overhead remains constant. In addition, exploiting the invertibility
of the NF, we propose a simple approach to regularize the network's embeddings
with respect to past tasks. We show that our method performs favorably with
respect to state-of-the-art approaches in the literature, with bounded
computational power and memory overheads.

    

### [[2008.01683] A Bayesian Hierarchical Score for Structure Learning from Related Data Sets](http://arxiv.org/abs/2008.01683)


  Score functions for learning the structure of Bayesian networks in the
literature assume that data are a homogeneous set of observations; whereas it
is often the case that they comprise different related, but not homogeneous,
data sets collected in different ways. In this paper we propose a new Bayesian
Dirichlet score, which we call Bayesian Hierarchical Dirichlet (BHD). The
proposed score is based on a hierarchical model that pools information across
data sets to learn a single encompassing network structure, while taking into
account the differences in their probabilistic structures. We derive a
closed-form expression for BHD using a variational approximation of the
marginal likelihood, we study the associated computational cost and we evaluate
its performance using simulated data. We find that, when data comprise multiple
related data sets, BHD outperforms the Bayesian Dirichlet equivalent uniform
(BDeu) score in terms of reconstruction accuracy as measured by the Structural
Hamming distance, and that it is as accurate as BDeu when data are homogeneous.
This improvement is particularly clear when either the number of variables in
the network or the number of observations is large. Moreover, the estimated
networks are sparser and therefore more interpretable than those obtained with
BDeu thanks to a lower number of false positive arcs.

    

### [[2008.07545] Whitening and second order optimization both make information in the dataset unusable during training, and can reduce or prevent generalization](http://arxiv.org/abs/2008.07545)


  Machine learning is predicated on the concept of generalization: a model
achieving low error on a sufficiently large training set should also perform
well on novel samples from the same distribution. We show that both data
whitening and second order optimization can harm or entirely prevent
generalization. In general, model training harnesses information contained in
the sample-sample second moment matrix of a dataset. For a general class of
models, namely models with a fully connected first layer, we prove that the
information contained in this matrix is the only information which can be used
to generalize. Models trained using whitened data, or with certain second order
optimization schemes, have less access to this information, resulting in
reduced or nonexistent generalization ability. We experimentally verify these
predictions for several architectures, and further demonstrate that
generalization continues to be harmed even when theoretical requirements are
relaxed. However, we also show experimentally that regularized second order
optimization can provide a practical tradeoff, where training is accelerated
but less information is lost, and generalization can in some circumstances even
improve.

    

### [[2008.12623] From Optimizing Engagement to Measuring Value](http://arxiv.org/abs/2008.12623)


  Most recommendation engines today are based on predicting user engagement,
e.g. predicting whether a user will click on an item or not. However, there is
potentially a large gap between engagement signals and a desired notion of
"value" that is worth optimizing for. We use the framework of measurement
theory to (a) confront the designer with a normative question about what the
designer values, (b) provide a general latent variable model approach that can
be used to operationalize the target construct and directly optimize for it,
and (c) guide the designer in evaluating and revising their operationalization.
We implement our approach on the Twitter platform on millions of users. In line
with established approaches to assessing the validity of measurements, we
perform a qualitative evaluation of how well our model captures a desired
notion of "value".

    

### [[2008.12804] Rethinking the Objectives of Extractive Question Answering](http://arxiv.org/abs/2008.12804)


  This work demonstrates that, contrary to a common belief, using the objective
with independence assumption for modelling the span probability $P(a_s,a_e) =
P(a_s)P(a_e)$ of span starting at position $a_s$ and ending at position $a_e$
has adverse effects. Therefore we propose multiple approaches to modelling
joint probability $P(a_s,a_e)$ directly. Among those, we propose a compound
objective, composed from the joint probability while still keeping the
objective with independence assumption as an auxiliary objective. We find that
the compound objective is consistently superior or equal to other assumptions
in exact match. Additionally, we identified common errors caused by the
assumption of independence and manually checked the counterpart predictions,
demonstrating the impact of the compound objective on the real examples. Our
findings are supported via experiments with three extractive QA models (BIDAF,
BERT, ALBERT) over six datasets and our code, individual results and manual
analysis are available online.

    

### [[2009.00038] Uncertainty quantification for Markov Random Fields](http://arxiv.org/abs/2009.00038)


  We present an information-based uncertainty quantification method for general
Markov Random Fields. Markov Random Fields (MRF) are structured, probabilistic
graphical models over undirected graphs, and provide a fundamental unifying
modeling tool for statistical mechanics, probabilistic machine learning, and
artificial intelligence. Typically MRFs are complex and high-dimensional with
nodes and edges (connections) built in a modular fashion from simpler,
low-dimensional probabilistic models and their local connections; in turn, this
modularity allows to incorporate available data to MRFs and efficiently
simulate them by leveraging their graph-theoretic structure. Learning graphical
models from data and/or constructing them from physical modeling and
constraints necessarily involves uncertainties inherited from data, modeling
choices, or numerical approximations. These uncertainties in the MRF can be
manifested either in the graph structure or the probability distribution
functions, and necessarily will propagate in predictions for quantities of
interest. Here we quantify such uncertainties using tight, information based
bounds on the predictions of quantities of interest; these bounds take
advantage of the graphical structure of MRFs and are capable of handling the
inherent high-dimensionality of such graphical models. We demonstrate our
methods in MRFs for medical diagnostics and statistical mechanics models. In
the latter, we develop uncertainty quantification bounds for finite size
effects and phase diagrams, which constitute two of the typical predictions
goals of statistical mechanics modeling.

    

### [[2009.06342] Reservoir Memory Machines as Neural Computers](http://arxiv.org/abs/2009.06342)


  Differentiable neural computers extend artificial neural networks with an
explicit memory without interference, thus enabling the model to perform
classic computation tasks such as graph traversal. However, such models are
difficult to train, requiring long training times and large datasets. In this
work, we achieve some of the computational capabilities of differentiable
neural computers with a model that can be trained very efficiently, namely an
echo state network with an explicit memory without interference. This extension
enables echo state networks to recognize all regular languages, including those
that contractive echo state networks provably can not recognize. Further, we
demonstrate experimentally that our model performs comparably to its
fully-trained deep version on several typical benchmark tasks for
differentiable neural computers.

    

### [[2010.15728] Explainable Automated Coding of Clinical Notes using Hierarchical Label-wise Attention Networks and Label Embedding Initialisation](http://arxiv.org/abs/2010.15728)


  Diagnostic or procedural coding of clinical notes aims to derive a coded
summary of disease-related information about patients. Such coding is usually
done manually in hospitals but could potentially be automated to improve the
efficiency and accuracy of medical coding. Recent studies on deep learning for
automated medical coding achieved promising performances. However, the
explainability of these models is usually poor, preventing them to be used
confidently in supporting clinical practice. Another limitation is that these
models mostly assume independence among labels, ignoring the complex
correlation among medical codes which can potentially be exploited to improve
the performance. We propose a Hierarchical Label-wise Attention Network (HLAN),
which aimed to interpret the model by quantifying importance (as attention
weights) of words and sentences related to each of the labels. Secondly, we
propose to enhance the major deep learning models with a label embedding (LE)
initialisation approach, which learns a dense, continuous vector representation
and then injects the representation into the final layers and the label-wise
attention layers in the models. We evaluated the methods using three settings
on the MIMIC-III discharge summaries: full codes, top-50 codes, and the UK NHS
COVID-19 shielding codes. Experiments were conducted to compare HLAN and LE
initialisation to the state-of-the-art neural network based methods. HLAN
achieved the best Micro-level AUC and $F_1$ on the top-50 code prediction and
comparable results on the NHS COVID-19 shielding code prediction to other
models. By highlighting the most salient words and sentences for each label,
HLAN showed more meaningful and comprehensive model interpretation compared to
its downgraded baselines and the CNN-based models. LE initialisation
consistently boosted most deep learning models for automated medical coding.

    

### [[2011.05601] A Nonconvex Framework for Structured Dynamic Covariance Recovery](http://arxiv.org/abs/2011.05601)


  We propose a flexible yet interpretable model for high-dimensional data with
time-varying second order statistics, motivated and applied to functional
neuroimaging data. Motivated by the neuroscience literature, we factorize the
covariances into sparse spatial and smooth temporal components. While this
factorization results in both parsimony and domain interpretability, the
resulting estimation problem is nonconvex. To this end, we design a two-stage
optimization scheme with a carefully tailored spectral initialization, combined
with iteratively refined alternating projected gradient descent. We prove a
linear convergence rate up to a nontrivial statistical error for the proposed
descent scheme and establish sample complexity guarantees for the estimator. We
further quantify the statistical error for the multivariate Gaussian case.
Empirical results using simulated and real brain imaging data illustrate that
our approach outperforms existing baselines.

    

### [[2011.12916] Equivariant Learning of Stochastic Fields: Gaussian Processes and Steerable Conditional Neural Processes](http://arxiv.org/abs/2011.12916)


  Motivated by objects such as electric fields or fluid streams, we study the
problem of learning stochastic fields, i.e. stochastic processes whose samples
are fields like those occurring in physics and engineering. Considering general
transformations such as rotations and reflections, we show that spatial
invariance of stochastic fields requires an inference model to be equivariant.
Leveraging recent advances from the equivariance literature, we study
equivariance in two classes of models. Firstly, we fully characterise
equivariant Gaussian processes. Secondly, we introduce Steerable Conditional
Neural Processes (SteerCNPs), a new, fully equivariant member of the Neural
Process family. In experiments with Gaussian process vector fields, images, and
real-world weather data, we observe that SteerCNPs significantly improve the
performance of previous models and equivariance leads to improvements in
transfer learning tasks.

    

### [[2012.09830] Intrinsically Motivated Goal-Conditioned Reinforcement Learning: a Short Survey](http://arxiv.org/abs/2012.09830)


  Building autonomous machines that can explore open-ended environments,
discover possible interactions and autonomously build repertoires of skills is
a general objective of artificial intelligence. Developmental approaches argue
that this can only be achieved by autonomous and intrinsically motivated
learning agents that can generate, select and learn to solve their own
problems. In recent years, we have seen a convergence of developmental
approaches, and developmental robotics in particular, with deep reinforcement
learning (RL) methods, forming the new domain of developmental machine
learning. Within this new domain, we review here a set of methods where deep RL
algorithms are trained to tackle the developmental robotics problem of the
autonomous acquisition of open-ended repertoires of skills. Intrinsically
motivated goal-conditioned RL algorithms train agents to learn to represent,
generate and pursue their own goals. The self-generation of goals requires the
learning of compact goal encodings as well as their associated goal-achievement
functions, which results in new challenges compared to traditional RL
algorithms designed to tackle pre-defined sets of goals using external reward
signals. This paper proposes a typology of these methods at the intersection of
deep RL and developmental approaches, surveys recent approaches and discusses
future avenues.

    

### [[2012.11589] Making transport more robust and interpretable by moving data through a small number of anchor points](http://arxiv.org/abs/2012.11589)


  Optimal transport (OT) is a widely used technique for distribution alignment,
with applications throughout the machine learning, graphics, and vision
communities. Without any additional structural assumptions on trans-port,
however, OT can be fragile to outliers or noise, especially in high dimensions.
Here, we introduce a new form of structured OT that simultaneously learns
low-dimensional structure in data while leveraging this structure to solve the
alignment task. Compared with OT, the resulting transport plan has better
structural interpretability, highlighting the connections between individual
data points and local geometry, and is more robust to noise and sampling. We
apply the method to synthetic as well as real datasets, where we show that our
method can facilitate alignment in noisy settings and can be used to both
correct and interpret domain shift.

    

### [[2012.11860] Efficient and Visualizable Convolutional Neural Networks for COVID-19 Classification Using Chest CT](http://arxiv.org/abs/2012.11860)


  With COVID-19 cases rising rapidly, deep learning has emerged as a promising
diagnosis technique. However, identifying the most accurate models to
characterize COVID-19 patients is challenging because comparing results
obtained with different types of data and acquisition processes is non-trivial.
In this paper we designed, evaluated, and compared the performance of 20
convolutional neutral networks in classifying patients as COVID-19 positive,
healthy, or suffering from other pulmonary lung infections based on Chest CT
scans, serving as the first to consider the EfficientNet family for COVID-19
diagnosis and employ intermediate activation maps for visualizing model
performance. All models are trained and evaluated in Python using 4173 Chest CT
images from the dataset entitled "A COVID multiclass dataset of CT scans," with
2168, 758, and 1247 images of patients that are COVID-19 positive, healthy, or
suffering from other pulmonary infections, respectively. EfficientNet-B5 was
identified as the best model with an F1 score of 0.9769+/-0.0046, accuracy of
0.9759+/-0.0048, sensitivity of 0.9788+/-0.0055, specificity of
0.9730+/-0.0057, and precision of 0.9751 +/- 0.0051. On an alternate 2-class
dataset, EfficientNetB5 obtained an accuracy of 0.9845+/-0.0109, F1 score of
0.9599+/-0.0251, sensitivity of 0.9682+/-0.0099, specificity of
0.9883+/-0.0150, and precision of 0.9526 +/- 0.0523. Intermediate activation
maps and Gradient-weighted Class Activation Mappings offered
human-interpretable evidence of the model's perception of ground-class
opacities and consolidations, hinting towards a promising use-case of
artificial intelligence-assisted radiology tools. With a prediction speed of
under 0.1 seconds on GPUs and 0.5 seconds on CPUs, our proposed model offers a
rapid, scalable, and accurate diagnostic for COVID-19.

    

### [[2012.13134] Sensitivity -- Local Index to Control Chaoticity or Gradient Globally](http://arxiv.org/abs/2012.13134)


  Here, we introduce a fully local index named "sensitivity" for each neuron to
control chaoticity or gradient globally in a neural network (NN). We also
propose a learning method to adjust it named "sensitivity adjustment learning
(SAL)". The index is the gradient magnitude of its output with respect to its
inputs. By adjusting its time average to 1.0 in each neuron, information
transmission in the neuron changes to be moderate without shrinking or
expanding for both forward and backward computations. That results in moderate
information transmission through a layer of neurons when the weights and inputs
are random. Therefore, SAL can control the chaoticity of the network dynamics
in a recurrent NN (RNN). It can also solve the vanishing gradient problem in
error backpropagation (BP) learning in a deep feedforward NN or an RNN. We
demonstrate that when applying SAL to an RNN with small and random initial
weights, log-sensitivity, which is the logarithm of RMS (root mean square)
sensitivity over all the neurons, is equivalent to the maximum Lyapunov
exponent until it reaches 0.0. We also show that SAL works with BP or BPTT (BP
through time) to avoid the vanishing gradient problem in a 300-layer NN or an
RNN that learns a problem with a lag of 300 steps between the first input and
the output. Compared with manually fine-tuning the spectral radius of the
weight matrix before learning, SAL's continuous nonlinear learning nature
prevents loss of sensitivities during learning, resulting in a significant
improvement in learning performance.

    

### [[2101.11186] Evolutionary Generative Adversarial Networks with Crossover Based Knowledge Distillation](http://arxiv.org/abs/2101.11186)


  Generative Adversarial Networks (GAN) is an adversarial model, and it has
been demonstrated to be effective for various generative tasks. However, GAN
and its variants also suffer from many training problems, such as mode collapse
and gradient vanish. In this paper, we firstly propose a general crossover
operator, which can be widely applied to GANs using evolutionary strategies.
Then we design an evolutionary GAN framework C-GAN based on it. And we combine
the crossover operator with evolutionary generative adversarial networks (EGAN)
to implement the evolutionary generative adversarial networks with crossover
(CE-GAN). Under the premise that a variety of loss functions are used as
mutation operators to generate mutation individuals, we evaluate the generated
samples and allow the mutation individuals to learn experiences from the output
in a knowledge distillation manner, imitating the best output outcome,
resulting in better offspring. Then, we greedily selected the best offspring as
parents for subsequent training using discriminator as evaluator. Experiments
on real datasets demonstrate the effectiveness of CE-GAN and show that our
method is competitive in terms of generated images quality and time efficiency.

    

### [[2101.11296] FedH2L: Federated Learning with Model and Statistical Heterogeneity](http://arxiv.org/abs/2101.11296)


  Federated learning (FL) enables distributed participants to collectively
learn a strong global model without sacrificing their individual data privacy.
Mainstream FL approaches require each participant to share a common network
architecture and further assume that data are are sampled IID across
participants. However, in real-world deployments participants may require
heterogeneous network architectures; and the data distribution is almost
certainly non-uniform across participants. To address these issues we introduce
FedH2L, which is agnostic to both the model architecture and robust to
different data distributions across participants. In contrast to approaches
sharing parameters or gradients, FedH2L relies on mutual distillation,
exchanging only posteriors on a shared seed set between participants in a
decentralized manner. This makes it extremely bandwidth efficient, model
agnostic, and crucially produces models capable of performing well on the whole
data distribution when learning from heterogeneous silos.

    

### [[2102.00434] The Connection Between Approximation, Depth Separation and Learnability in Neural Networks](http://arxiv.org/abs/2102.00434)


  Several recent works have shown separation results between deep neural
networks, and hypothesis classes with inferior approximation capacity such as
shallow networks or kernel classes. On the other hand, the fact that deep
networks can efficiently express a target function does not mean that this
target function can be learned efficiently by deep neural networks. In this
work we study the intricate connection between learnability and approximation
capacity. We show that learnability with deep networks of a target function
depends on the ability of simpler classes to approximate the target.
Specifically, we show that a necessary condition for a function to be learnable
by gradient descent on deep neural networks is to be able to approximate the
function, at least in a weak sense, with shallow neural networks. We also show
that a class of functions can be learned by an efficient statistical query
algorithm if and only if it can be approximated in a weak sense by some kernel
class. We give several examples of functions which demonstrate depth
separation, and conclude that they cannot be efficiently learned, even by a
hypothesis class that can efficiently approximate them.

    

### [[2102.03088] Boost AI Power: Data Augmentation Strategies with unlabelled Data and Conformal Prediction, a Case in Alternative Herbal Medicine Discrimination with Electronic Nose](http://arxiv.org/abs/2102.03088)


  Electronic nose has been proven to be effective in alternative herbal
medicine classification, but due to the nature of supervised learning, previous
research heavily relies on the labelled training data, which are time-costly
and labor-intensive to collect. To alleviate the critical dependency on the
training data in real-world applications, this study aims to improve
classification accuracy via data augmentation strategies. The effectiveness of
five data augmentation strategies under different training data inadequacy are
investigated in two scenarios: the noise-free scenario where different
availabilities of unlabelled data were considered, and the noisy scenario where
different levels of Gaussian noises and translational shifts were added to
represent sensor drifts. The five augmentation strategies, namely noise-adding
data augmentation, semi-supervised learning, classifier-based online learning,
Inductive Conformal Prediction (ICP) online learning and our novel ensemble ICP
online learning proposed in this study, are experimented and compared against
supervised learning baseline, with Linear Discriminant Analysis (LDA) and
Support Vector Machine (SVM) as the classifiers. Our novel strategy, ensemble
ICP online learning, outperforms the others by showing non-decreasing
classification accuracy on all tasks and a significant improvement on most
simulated tasks (25out of 36 tasks,p<=0.05). Furthermore, this study provides a
systematic analysis of different augmentation strategies. It shows at least one
strategy significantly improved the classification accuracy with LDA (p<=0.05)
and non-decreasing classification accuracy with SVM in each task. In
particular, our proposed strategy demonstrated both effectiveness and
robustness in boosting the classification model generalizability, which can be
employed in other machine learning applications.

    

### [[2102.03980] Grab the Reins of Crowds: Estimating the Effects of Crowd Movement Guidance Using Causal Inference](http://arxiv.org/abs/2102.03980)


  Crowd movement guidance has been a fascinating problem in various fields,
such as easing traffic congestion in unusual events and evacuating people from
an emergency-affected area. To grab the reins of crowds, there has been
considerable demand for a decision support system that can answer a typical
question: ``what will be the outcomes of each of the possible options in the
current situation. In this paper, we consider the problem of estimating the
effects of crowd movement guidance from past data. To cope with limited amount
of available data biased by past decision-makers, we leverage two recent
techniques in deep representation learning for spatial data analysis and causal
inference. We use a spatial convolutional operator to extract effective spatial
features of crowds from a small amount of data and use balanced representation
learning based on the integral probability metrics to mitigate the selection
bias and missing counterfactual outcomes. To evaluate the performance on
estimating the treatment effects of possible guidance, we use a multi-agent
simulator to generate realistic data on evacuation scenarios in a crowded
theater, since there are no available datasets recording outcomes of all
possible crowd movement guidance. The results of three experiments demonstrate
that our proposed method reduces the estimation error by at most 56% from
state-of-the-art methods.

    

### [[2102.05791] Differentiable Implicit Soft-Body Physics](http://arxiv.org/abs/2102.05791)


  We present a differentiable soft-body physics simulator that can be composed
with neural networks as a differentiable layer. In contrast to other
differentiable physics approaches that use explicit forward models to define
state transitions, we focus on implicit state transitions defined via function
minimization. Implicit state transitions appear in implicit numerical
integration methods, which offer the benefits of large time steps and excellent
numerical stability, but require a special treatment to achieve
differentiability due to the absence of an explicit differentiable forward
pass. In contrast to other implicit differentiation approaches that require
explicit formulas for the force function and the force Jacobian matrix, we
present an energy-based approach that allows us to compute these derivatives
automatically and in a matrix-free fashion via reverse-mode automatic
differentiation. This allows for more flexibility and productivity when
defining physical models and is particularly important in the context of neural
network training, which often relies on reverse-mode automatic differentiation
(backpropagation). We demonstrate the effectiveness of our differentiable
simulator in policy optimization for locomotion tasks and show that it achieves
better sample efficiency than model-free reinforcement learning.

    

### [[2102.09663] Smart Feasibility Pump: Reinforcement Learning for (Mixed) Integer Programming](http://arxiv.org/abs/2102.09663)


  In this work, we propose a deep reinforcement learning (DRL) model for
finding a feasible solution for (mixed) integer programming (MIP) problems.
Finding a feasible solution for MIP problems is critical because many
successful heuristics rely on a known initial feasible solution. However, it is
in general NP-hard. Inspired by the feasibility pump (FP), a well-known
heuristic for searching feasible MIP solutions, we develop a smart feasibility
pump (SFP) method using DRL. In addition to multi-layer perception (MLP), we
propose a novel convolution neural network (CNN) structure for the policy
network to capture the hidden information of the constraint matrix of the MIP
problem. Numerical experiments on various problem instances show that SFP
significantly outperforms the classic FP in terms of the number of steps
required to reach the first feasible solution. Moreover, the CNN structure
works without the projection of the current solution as the input, which saves
the computational effort at each step of the FP algorithms to find projections.
This highlights the representational power of the CNN structure.

    

### [[2102.10440] Interventional Sum-Product Networks: Causal Inference with Tractable Probabilistic Models](http://arxiv.org/abs/2102.10440)


  While probabilistic models are an important tool for studying causality,
doing so suffers from the intractability of inference. As a step towards
tractable causal models, we consider the problem of learning interventional
distributions using sum-product networks (SPNs) that are over-parameterized by
gate functions, e.g., neural networks. Providing an arbitrarily intervened
causal graph as input, effectively subsuming Pearl's do-operator, the gate
function predicts the parameters of the SPN. The resulting interventional SPNs
are motivated and illustrated by a structural causal model themed around
personal health. Our empirical evaluation on three benchmark data sets as well
as a synthetic health data set clearly demonstrates that interventional SPNs
indeed are both expressive in modelling and flexible in adapting to the
interventions.

    

### [[2102.10618] Towards the Unification and Robustness of Perturbation and Gradient Based Explanations](http://arxiv.org/abs/2102.10618)


  As machine learning black boxes are increasingly being deployed in critical
domains such as healthcare and criminal justice, there has been a growing
emphasis on developing techniques for explaining these black boxes in a post
hoc manner. In this work, we analyze two popular post hoc interpretation
techniques: SmoothGrad which is a gradient based method, and a variant of LIME
which is a perturbation based method. More specifically, we derive explicit
closed form expressions for the explanations output by these two methods and
show that they both converge to the same explanation in expectation, i.e., when
the number of perturbed samples used by these methods is large. We then
leverage this connection to establish other desirable properties, such as
robustness, for these techniques. We also derive finite sample complexity
bounds for the number of perturbations required for these methods to converge
to their expected explanation. Finally, we empirically validate our theory
using extensive experimentation on both synthetic and real world datasets.

    

### [[2102.11068] Lottery Ticket Preserves Weight Correlation: Is It Desirable or Not?](http://arxiv.org/abs/2102.11068)


  In deep model compression, the recent finding "Lottery Ticket Hypothesis"
(LTH) (Frankle & Carbin, 2018) pointed out that there could exist a winning
ticket (i.e., a properly pruned sub-network together with original weight
initialization) that can achieve competitive performance than the original
dense network. However, it is not easy to observe such winning property in many
scenarios, where for example, a relatively large learning rate is used even if
it benefits training the original dense model. In this work, we investigate the
underlying condition and rationale behind the winning property, and find that
the underlying reason is largely attributed to the correlation between
initialized weights and final-trained weights when the learning rate is not
sufficiently large. Thus, the existence of winning property is correlated with
an insufficient DNN pretraining, and is unlikely to occur for a well-trained
DNN. To overcome this limitation, we propose the "pruning & fine-tuning" method
that consistently outperforms lottery ticket sparse training under the same
pruning algorithm and the same total training epochs. Extensive experiments
over multiple deep models (VGG, ResNet, MobileNet-v2) on different datasets
have been conducted to justify our proposals.

    

### [[2102.11830] Solving high-dimensional parabolic PDEs using the tensor train format](http://arxiv.org/abs/2102.11830)


  High-dimensional partial differential equations (PDEs) are ubiquitous in
economics, science and engineering. However, their numerical treatment poses
formidable challenges since traditional grid-based methods tend to be
frustrated by the curse of dimensionality. In this paper, we argue that tensor
trains provide an appealing approximation framework for parabolic PDEs: the
combination of reformulations in terms of backward stochastic differential
equations and regression-type methods in the tensor format holds the promise of
leveraging latent low-rank structures enabling both compression and efficient
computation. Following this paradigm, we develop novel iterative schemes,
involving either explicit and fast or implicit and accurate updates. We
demonstrate in a number of examples that our methods achieve a favorable
trade-off between accuracy and computational efficiency in comparison with
state-of-the-art neural network based approaches.

    

### [[2102.11866] Doubly Robust Off-Policy Actor-Critic: Convergence and Optimality](http://arxiv.org/abs/2102.11866)


  Designing off-policy reinforcement learning algorithms is typically a very
challenging task, because a desirable iteration update often involves an
expectation over an on-policy distribution. Prior off-policy actor-critic (AC)
algorithms have introduced a new critic that uses the density ratio for
adjusting the distribution mismatch in order to stabilize the convergence, but
at the cost of potentially introducing high biases due to the estimation errors
of both the density ratio and value function. In this paper, we develop a
doubly robust off-policy AC (DR-Off-PAC) for discounted MDP, which can take
advantage of learned nuisance functions to reduce estimation errors. Moreover,
DR-Off-PAC adopts a single timescale structure, in which both actor and critics
are updated simultaneously with constant stepsize, and is thus more sample
efficient than prior algorithms that adopt either two timescale or nested-loop
structure. We study the finite-time convergence rate and characterize the
sample complexity for DR-Off-PAC to attain an $\epsilon$-accurate optimal
policy. We also show that the overall convergence of DR-Off-PAC is doubly
robust to the approximation errors that depend only on the expressive power of
approximation functions. To the best of our knowledge, our study establishes
the first overall sample complexity analysis for a single time-scale off-policy
AC algorithm.

    

### [[2102.12056] Multi-Slice Low-Rank Tensor Decomposition Based Multi-Atlas Segmentation: Application to Automatic Pathological Liver CT Segmentation](http://arxiv.org/abs/2102.12056)


  Liver segmentation from abdominal CT images is an essential step for liver
cancer computer-aided diagnosis and surgical planning. However, both the
accuracy and robustness of existing liver segmentation methods cannot meet the
requirements of clinical applications. In particular, for the common clinical
cases where the liver tissue contains major pathology, current segmentation
methods show poor performance. In this paper, we propose a novel low-rank
tensor decomposition (LRTD) based multi-atlas segmentation (MAS) framework that
achieves accurate and robust pathological liver segmentation of CT images.
Firstly, we propose a multi-slice LRTD scheme to recover the underlying
low-rank structure embedded in 3D medical images. It performs the LRTD on small
image segments consisting of multiple consecutive image slices. Then, we
present an LRTD-based atlas construction method to generate tumor-free liver
atlases that mitigates the performance degradation of liver segmentation due to
the presence of tumors. Finally, we introduce an LRTD-based MAS algorithm to
derive patient-specific liver atlases for each test image, and to achieve
accurate pairwise image registration and label propagation. Extensive
experiments on three public databases of pathological liver cases validate the
effectiveness of the proposed method. Both qualitative and quantitative results
demonstrate that, in the presence of major pathology, the proposed method is
more accurate and robust than state-of-the-art methods.

    

### [[2102.12534] Entanglement Diagnostics for Efficient Quantum Computation](http://arxiv.org/abs/2102.12534)


  We consider information spreading measures in randomly initialized
variational quantum circuits and introduce entanglement diagnostics for
efficient variational quantum/classical computations. We establish a robust
connection between entanglement measures and optimization accuracy by solving
two eigensolver problems for Ising Hamiltonians with nearest-neighbor and
long-range spin interactions. As the circuit depth affects the average
entanglement of random circuit states, the entanglement diagnostics can
identify a high-performing depth range for optimization tasks encoded in local
Hamiltonians. We argue, based on an eigensolver problem for the
Sachdev-Ye-Kitaev model, that entanglement alone is insufficient as a
diagnostic to the approximation of volume-law entangled target states and that
a large number of circuit parameters is needed for such an optimization task.

    

### [[2103.00383] Brain Signals to Rescue Aphasia, Apraxia and Dysarthria Speech Recognition](http://arxiv.org/abs/2103.00383)


  In this paper, we propose a deep learning-based algorithm to improve the
performance of automatic speech recognition (ASR) systems for aphasia, apraxia,
and dysarthria speech by utilizing electroencephalography (EEG) features
recorded synchronously with aphasia, apraxia, and dysarthria speech. We
demonstrate a significant decoding performance improvement by more than 50\%
during test time for isolated speech recognition task and we also provide
preliminary results indicating performance improvement for the more challenging
continuous speech recognition task by utilizing EEG features. The results
presented in this paper show the first step towards demonstrating the
possibility of utilizing non-invasive neural signals to design a real-time
robust speech prosthetic for stroke survivors recovering from aphasia, apraxia,
and dysarthria. Our aphasia, apraxia, and dysarthria speech-EEG data set will
be released to the public to help further advance this interesting and crucial
research.

    

### [[2103.00502] Optimal Approximation Rate of ReLU Networks in terms of Width and Depth](http://arxiv.org/abs/2103.00502)


  This paper concentrates on the approximation power of deep feed-forward
neural networks in terms of width and depth. It is proved by construction that
ReLU networks with width $\mathcal{O}\big(\max\{d\lfloor N^{1/d}\rfloor,\,
N+2\}\big)$ and depth $\mathcal{O}(L)$ can approximate a Hölder continuous
function on $[0,1]^d$ with an approximation rate
$\mathcal{O}\big(\lambda\sqrt{d} (N^2L^2\ln N)^{-\alpha/d}\big)$, where
$\alpha\in (0,1]$ and $\lambda>0$ are Hölder order and constant,
respectively. Such a rate is optimal up to a constant in terms of width and
depth separately, while existing results are only nearly optimal without the
logarithmic factor in the approximation rate. More generally, for an arbitrary
continuous function $f$ on $[0,1]^d$, the approximation rate becomes
$\mathcal{O}\big(\,\sqrt{d}\,\omega_f\big( (N^2L^2\ln N)^{-1/d}\big)\,\big)$,
where $\omega_f(\cdot)$ is the modulus of continuity. We also extend our
analysis to any continuous function $f$ on a bounded set. Particularly, if ReLU
networks with depth $31$ and width $\mathcal{O}(N)$ are used to approximate
one-dimensional Lipschitz continuous functions on $[0,1]$ with a Lipschitz
constant $\lambda>0$, the approximation rate in terms of the total number of
parameters, $W=\mathcal{O}(N^2)$, becomes $\mathcal{O}(\tfrac{\lambda}{W\ln
W})$, which has not been discovered in the literature for fixed-depth ReLU
networks.

    

### [[2103.01134] Domain Generalization via Inference-time Label-Preserving Target Projections](http://arxiv.org/abs/2103.01134)


  Generalization of machine learning models trained on a set of source domains
on unseen target domains with different statistics, is a challenging problem.
While many approaches have been proposed to solve this problem, they only
utilize source data during training but do not take advantage of the fact that
a single target example is available at the time of inference. Motivated by
this, we propose a method that effectively uses the target sample during
inference beyond mere classification. Our method has three components - (i) A
label-preserving feature or metric transformation on source data such that the
source samples are clustered in accordance with their class irrespective of
their domain (ii) A generative model trained on the these features (iii) A
label-preserving projection of the target point on the source-feature manifold
during inference via solving an optimization problem on the input space of the
generative model using the learned metric. Finally, the projected target is
used in the classifier. Since the projected target feature comes from the
source manifold and has the same label as the real target by design, the
classifier is expected to perform better on it than the true target. We
demonstrate that our method outperforms the state-of-the-art Domain
Generalization methods on multiple datasets and tasks.

    

### [[2103.01488] Multi-Level Attention Pooling for Graph Neural Networks: Unifying Graph Representations with Multiple Localities](http://arxiv.org/abs/2103.01488)


  Graph neural networks (GNNs) have been widely used to learn vector
representation of graph-structured data and achieved better task performance
than conventional methods. The foundation of GNNs is the message passing
procedure, which propagates the information in a node to its neighbors. Since
this procedure proceeds one step per layer, the range of the information
propagation among nodes is small in the lower layers, and it expands toward the
higher layers. Therefore, a GNN model has to be deep enough to capture global
structural information in a graph. On the other hand, it is known that deep GNN
models suffer from performance degradation because they lose nodes' local
information, which would be essential for good model performance, through many
message passing steps. In this study, we propose a multi-level attention
pooling (MLAP) for graph-level classification tasks, which can adapt to both
local and global structural information in a graph. It has an attention pooling
layer for each message passing step and computes the final graph representation
by unifying the layer-wise graph representations. The MLAP architecture allows
models to utilize the structural information of graphs with multiple levels of
localities because it preserves layer-wise information before losing them due
to oversmoothing. Results of our experiments show that the MLAP architecture
improves deeper models' performance in graph classification tasks compared to
the baseline architectures. In addition, analyses on the layer-wise graph
representations suggest that aggregating information from multiple levels of
localities indeed has the potential to improve the discriminability of learned
graph representations.

    

### [[2103.02503] Domain Generalization in Vision: A Survey](http://arxiv.org/abs/2103.02503)


  Generalization to out-of-distribution (OOD) data is a capability natural to
humans yet challenging for machines to reproduce. This is because most learning
algorithms strongly rely on the i.i.d.~assumption on source/target data, which
is often violated in practice due to domain shift. Domain generalization (DG)
aims to achieve OOD generalization by using only source data for model
learning. Since first introduced in 2011, research in DG has made great
progresses. In particular, intensive research in this topic has led to a broad
spectrum of methodologies, e.g., those based on domain alignment,
meta-learning, data augmentation, or ensemble learning, just to name a few; and
has covered various vision applications such as object recognition,
segmentation, action recognition, and person re-identification. In this paper,
for the first time a comprehensive literature review is provided to summarize
the developments in DG for computer vision over the past decade. Specifically,
we first cover the background by formally defining DG and relating it to other
research fields like domain adaptation and transfer learning. Second, we
conduct a thorough review into existing methods and present a categorization
based on their methodologies and motivations. Finally, we conclude this survey
with insights and discussions on future research directions.

    

### [[2103.04075] Domain Adaptive Robotic Gesture Recognition with Unsupervised Kinematic-Visual Data Alignment](http://arxiv.org/abs/2103.04075)


  Automated surgical gesture recognition is of great importance in
robot-assisted minimally invasive surgery. However, existing methods assume
that training and testing data are from the same domain, which suffers from
severe performance degradation when a domain gap exists, such as the simulator
and real robot. In this paper, we propose a novel unsupervised domain
adaptation framework which can simultaneously transfer multi-modality
knowledge, i.e., both kinematic and visual data, from simulator to real robot.
It remedies the domain gap with enhanced transferable features by using
temporal cues in videos, and inherent correlations in multi-modal towards
recognizing gesture. Specifically, we first propose an MDO-K to align
kinematics, which exploits temporal continuity to transfer motion directions
with smaller gap rather than position values, relieving the adaptation burden.
Moreover, we propose a KV-Relation-ATT to transfer the co-occurrence signals of
kinematics and vision. Such features attended by correlation similarity are
more informative for enhancing domain-invariance of the model. Two feature
alignment strategies benefit the model mutually during the end-to-end learning
process. We extensively evaluate our method for gesture recognition using DESK
dataset with peg transfer procedure. Results show that our approach recovers
the performance with great improvement gains, up to 12.91% in ACC and 20.16% in
F1score without using any annotations in real robot.

    

### [[2103.06419] SAR-U-Net: squeeze-and-excitation block and atrous spatial pyramid pooling based residual U-Net for automatic liver segmentation in Computed Tomography](http://arxiv.org/abs/2103.06419)


  Background and objective: In this paper, a modified U-Net based framework is
presented, which leverages techniques from Squeeze-and-Excitation (SE) block,
Atrous Spatial Pyramid Pooling (ASPP) and residual learning for accurate and
robust liver CT segmentation, and the effectiveness of the proposed method was
tested on two public datasets LiTS17 and SLiver07.
Methods: A new network architecture called SAR-U-Net was designed. Firstly,
the SE block is introduced to adaptively extract image features after each
convolution in the U-Net encoder, while suppressing irrelevant regions, and
highlighting features of specific segmentation task; Secondly, ASPP was
employed to replace the transition layer and the output layer, and acquire
multi-scale image information via different receptive fields. Thirdly, to
alleviate the degradation problem, the traditional convolution block was
replaced with the residual block and thus prompt the network to gain accuracy
from considerably increased depth.
Results: In the LiTS17 experiment, the mean values of Dice, VOE, RVD, ASD and
MSD were 95.71, 9.52, -0.84, 1.54 and 29.14, respectively. Compared with other
closely related 2D-based models, the proposed method achieved the highest
accuracy. In the experiment of the SLiver07, the mean values of Dice, VOE, RVD,
ASD and MSD were 97.31, 5.37, -1.08, 1.85 and 27.45, respectively. Compared
with other closely related models, the proposed method achieved the highest
segmentation accuracy except for the RVD.
Conclusion: The proposed model enables a great improvement on the accuracy
compared to 2D-based models, and its robustness in circumvent challenging
problems, such as small liver regions, discontinuous liver regions, and fuzzy
liver boundaries, is also well demonstrated and validated.

    

### [[2103.09022] Missing Cone Artifacts Removal in ODT using Unsupervised Deep Learning in Projection Domain](http://arxiv.org/abs/2103.09022)


  Optical diffraction tomography (ODT) produces three dimensional distribution
of refractive index (RI) by measuring scattering fields at various angles.
Although the distribution of RI index is highly informative, due to the missing
cone problem stemming from the limited-angle acquisition of holograms,
reconstructions have very poor resolution along axial direction compared to the
horizontal imaging plane. To solve this issue, here we present a novel
unsupervised deep learning framework, which learns the probability distribution
of missing projection views through optimal transport driven cycleGAN.
Experimental results show that missing cone artifact in ODT can be
significantly resolved by the proposed method.

    

### [[2103.11793] DeepOPF-V: Solving AC-OPF Problems Efficiently](http://arxiv.org/abs/2103.11793)


  AC optimal power flow (AC-OPF) problems need to be solved more frequently in
the future to maintain stable and economic power system operation. To tackle
this challenge, a deep neural network-based voltage-constrained approach
(DeepOPF-V) is proposed to solve AC-OPF problems with high computational
efficiency. Its unique design predicts voltages of all buses and then uses them
to reconstruct the remaining variables without solving non-linear AC power flow
equations. A fast post-processing process is developed to enforce the box
constraints. The effectiveness of DeepOPF-V is validated by simulations on IEEE
118/300-bus systems and a 2000-bus test system. Compared with existing studies,
DeepOPF-V achieves decent computation speedup up to four orders of magnitude
and comparable performance in optimality gap and preserving the feasibility of
the solution.

    

### [[2103.16336] Model-based clustering of partial records](http://arxiv.org/abs/2103.16336)


  Partially recorded data are frequently encountered in many applications and
usually clustered by first removing incomplete cases or features with missing
values, or by imputing missing values, followed by application of a clustering
algorithm to the resulting altered dataset. Here, we develop clustering
methodology through a model-based approach using the marginal density for the
observed values, assuming a finite mixture model of multivariate $t$
distributions. We compare our approximate algorithm to the corresponding full
expectation-maximization (EM) approach that considers the missing values in the
incomplete data set and makes a missing at random (MAR) assumption, as well as
case deletion and imputation methods. Since only the observed values are
utilized, our approach is computationally more efficient than imputation or
full EM. Simulation studies demonstrate that our approach has favorable
recovery of the true cluster partition compared to case deletion and imputation
under various missingness mechanisms, and is at least competitive with the full
EM approach, even when MAR assumptions are violated. Our methodology is
demonstrated on a problem of clustering gamma-ray bursts and is implemented at
this https URL.

    

### [[2104.04046] Semi-Supervised Learning of Classifiers from a Statistical Perspective: A Brief Review](http://arxiv.org/abs/2104.04046)


  There has been increasing attention to semi-supervised learning (SSL)
approaches in machine learning to forming a classifier in situations where the
training data for a classifier consists of a limited number of classified
observations but a much larger number of unclassified observations. This is
because the procurement of classified data can be quite costly due to high
acquisition costs and subsequent financial, time, and ethical issues that can
arise in attempts to provide the true class labels for the unclassified data
that have been acquired. We provide here a review of statistical SSL approaches
to this problem, focussing on the recent result that a classifier formed from a
partially classified sample can actually have smaller expected error rate than
that if the sample were completely classified.

    

### [[2104.04184] Robust Training of Social Media Image Classification Models for Rapid Disaster Response](http://arxiv.org/abs/2104.04184)


  Images shared on social media help crisis managers gain situational awareness
and assess incurred damages, among other response tasks. As the volume and
velocity of such content are typically high, real-time image classification has
become an urgent need for a faster disaster response. Recent advances in
computer vision and deep neural networks have enabled the development of models
for real-time image classification for a number of tasks, including detecting
crisis incidents, filtering irrelevant images, classifying images into specific
humanitarian categories, and assessing the severity of the damage. To develop
robust real-time models, it is necessary to understand the capability of the
publicly available pre-trained models for these tasks, which remains to be
under-explored in the crisis informatics literature. In this study, we address
such limitations by investigating ten different network architectures for four
different tasks using the largest publicly available datasets for these tasks.
We also explore various data augmentation strategies, semi-supervised
techniques, and a multitask learning setup. In our extensive experiments, we
achieve promising results.

    

### [[2104.05802] Efficient Optimal Transport Algorithm by Accelerated Gradient descent](http://arxiv.org/abs/2104.05802)


  Optimal transport (OT) plays an essential role in various areas like machine
learning and deep learning. However, computing discrete optimal transport plan
for large scale problems with adequate accuracy and efficiency is still highly
challenging. Recently, methods based on the Sinkhorn algorithm add an entropy
regularizer to the prime problem and get a trade off between efficiency and
accuracy. In this paper, we propose a novel algorithm to further improve the
efficiency and accuracy based on Nesterov's smoothing technique. Basically, the
non-smooth c-transform of the Kantorovich potential is approximated by the
smooth Log-Sum-Exp function, which finally smooths the original non-smooth
Kantorovich dual functional (energy). The smooth Kantorovich functional can be
optimized by the fast proximal gradient algorithm (FISTA) efficiently.
Theoretically, the computational complexity of the proposed method is given by
$O(n^{\frac{5}{2}} \sqrt{\log n} /\epsilon)$, which is lower than that of the
Sinkhorn algorithm. Empirically, compared with the Sinkhorn algorithm, our
experimental results demonstrate that the proposed method achieves faster
convergence and better accuracy with the same parameter.

    

### [[2104.10818] XAI-N: Sensor-based Robot Navigation using Expert Policies and Decision Trees](http://arxiv.org/abs/2104.10818)


  We present a novel sensor-based learning navigation algorithm to compute a
collision-free trajectory for a robot in dense and dynamic environments with
moving obstacles or targets. Our approach uses deep reinforcement
learning-based expert policy that is trained using a sim2real paradigm. In
order to increase the reliability and handle the failure cases of the expert
policy, we combine with a policy extraction technique to transform the
resulting policy into a decision tree format. The resulting decision tree has
properties which we use to analyze and modify the policy and improve
performance on navigation metrics including smoothness, frequency of
oscillation, frequency of immobilization, and obstruction of target. We are
able to modify the policy to address these imperfections without retraining,
combining the learning power of deep learning with the control of
domain-specific algorithms. We highlight the benefits of our algorithm in
simulated environments and navigating a Clearpath Jackal robot among moving
pedestrians. (Videos at this url:
this https URL)

    

### [[2105.04180] Rate-Distortion Analysis of Minimum Excess Risk in Bayesian Learning](http://arxiv.org/abs/2105.04180)


  In parametric Bayesian learning, a prior is assumed on the parameter $W$
which determines the distribution of samples. In this setting, Minimum Excess
Risk (MER) is defined as the difference between the minimum expected loss
achievable when learning from data and the minimum expected loss that could be
achieved if $W$ was observed. In this paper, we build upon and extend the
recent results of (Xu & Raginsky, 2020) to analyze the MER in Bayesian learning
and derive information-theoretic bounds on it. We formulate the problem as a
(constrained) rate-distortion optimization and show how the solution can be
bounded above and below by two other rate-distortion functions that are easier
to study. The lower bound represents the minimum possible excess risk
achievable by any process using $R$ bits of information from the parameter $W$.
For the upper bound, the optimization is further constrained to use $R$ bits
from the training set, a setting which relates MER to information-theoretic
bounds on the generalization gap in frequentist learning. We derive
information-theoretic bounds on the difference between these upper and lower
bounds and show that they can provide order-wise tight rates for MER under
certain conditions. This analysis gives more insight into the
information-theoretic nature of Bayesian learning as well as providing novel
bounds.

    

### [[2105.05318] GANs for Medical Image Synthesis: An Empirical Study](http://arxiv.org/abs/2105.05318)


  Generative Adversarial Networks (GANs) have become increasingly powerful,
generating mind-blowing photorealistic images that mimic the content of
datasets they were trained to replicate. One recurrent theme in medical imaging
is whether GANs can also be effective at generating workable medical data as
they are for generating realistic RGB images. In this paper, we perform a
multi-GAN and multi-application study to gauge the benefits of GANs in medical
imaging. We tested various GAN architectures from basic DCGAN to more
sophisticated style-based GANs on three medical imaging modalities and organs
namely : cardiac cine-MRI, liver CT and RGB retina images. GANs were trained on
well-known and widely utilized datasets from which their FID score were
computed to measure the visual acuity of their generated images. We further
tested their usefulness by measuring the segmentation accuracy of a U-Net
trained on these generated images.
Results reveal that GANs are far from being equal as some are ill-suited for
medical imaging applications while others are much better off. The
top-performing GANs are capable of generating realistic-looking medical images
by FID standards that can fool trained experts in a visual Turing test and
comply to some metrics. However, segmentation results suggests that no GAN is
capable of reproducing the full richness of a medical datasets.

    

### [[2105.14399] Improving Entropic Out-of-Distribution Detection using Isometric Distances and the Minimum Distance Score](http://arxiv.org/abs/2105.14399)


  Current out-of-distribution detection approaches usually present special
requirements (e.g., collecting outlier data and hyperparameter validation) and
produce side effects (classification accuracy drop and slow/inefficient
inferences). Recently, entropic out-of-distribution detection has been proposed
as a seamless approach (i.e., a solution that avoids all the previously
mentioned drawbacks). The entropic out-of-distribution detection solution
comprises the IsoMax loss for training and the entropic score for
out-of-distribution detection. The IsoMax loss works as a SoftMax loss drop-in
replacement because swapping the SoftMax loss with the IsoMax loss requires no
changes in the model's architecture or training procedures/hyperparameters. In
this paper, we propose to perform what we call an isometrization of the
distances used in the IsoMax loss. Additionally, we propose to replace the
entropic score with the minimum distance score. Our experiments showed that
these simple modifications increase out-of-distribution detection performance
while keeping the solution seamless. Code available at
$\href{this https URL}{\text{entropic
out-of-distribution detection}}$.

    

### [[2106.02246] Deep Contextual Learners for Protein Networks](http://arxiv.org/abs/2106.02246)


  Spatial context is central to understanding health and disease. Yet reference
protein interaction networks lack such contextualization, thereby limiting the
study of where protein interactions likely occur in the human body and how they
may be altered in disease. Contextualized protein interactions could better
characterize genes with disease-specific interactions and elucidate diseases'
manifestation in specific cell types. Here, we introduce AWARE, a graph neural
message passing approach to inject cellular and tissue context into protein
embeddings. AWARE optimizes for a multi-scale embedding space, whose structure
reflects network topology at a single-cell resolution. We construct a
multi-scale network of the Human Cell Atlas and apply AWARE to learn protein,
cell type, and tissue embeddings that uphold cell type and tissue hierarchies.
We demonstrate AWARE's utility on the novel task of predicting whether a
protein is altered in disease and where that association most likely manifests
in the human body. To this end, AWARE outperforms generic embeddings without
contextual information by at least 12.5%, showing AWARE's potential to reveal
context-dependent roles of proteins in disease.

    

### [[2106.03723] Self-Supervised Graph Learning with Proximity-based Views and Channel Contrast](http://arxiv.org/abs/2106.03723)


  We consider graph representation learning in a self-supervised manner. Graph
neural networks (GNNs) use neighborhood aggregation as a core component that
results in feature smoothing among nodes in proximity. While successful in
various prediction tasks, such a paradigm falls short of capturing nodes'
similarities over a long distance, which proves to be important for
high-quality learning. To tackle this problem, we strengthen the graph with two
additional graph views, in which nodes are directly linked to those with the
most similar features or local structures. Not restricted by connectivity in
the original graph, the generated views allow the model to enhance its
expressive power with new and complementary perspectives from which to look at
the relationship between nodes. Following a contrastive learning approach, we
propose a method that aims to maximize the agreement between representations
across generated views and the original graph. We also propose a channel-level
contrast approach that greatly reduces computation cost, compared to the
commonly used node level contrast, which requires computation cost quadratic in
the number of nodes. Extensive experiments on seven assortative graphs and four
disassortative graphs demonstrate the effectiveness of our approach.

    

### [[2106.06981] Thinking Like Transformers](http://arxiv.org/abs/2106.06981)


  What is the computational model behind a Transformer? Where recurrent neural
networks have direct parallels in finite state machines, allowing clear
discussion and thought around architecture variants or trained models,
Transformers have no such familiar parallel. In this paper we aim to change
that, proposing a computational model for the transformer-encoder in the form
of a programming language. We map the basic components of a transformer-encoder
-- attention and feed-forward computation -- into simple primitives, around
which we form a programming language: the Restricted Access Sequence Processing
Language (RASP). We show how RASP can be used to program solutions to tasks
that could conceivably be learned by a Transformer, and how a Transformer can
be trained to mimic a RASP solution. In particular, we provide RASP programs
for histograms, sorting, and Dyck-languages. We further use our model to relate
their difficulty in terms of the number of required layers and attention heads:
analyzing a RASP program implies a maximum number of heads and layers necessary
to encode a task in a transformer. Finally, we see how insights gained from our
abstraction might be used to explain phenomena seen in recent works.

    

### [[2106.07830] On the Convergence of Deep Learning with Differential Privacy](http://arxiv.org/abs/2106.07830)


  In deep learning with differential privacy (DP), the neural network achieves
the privacy usually at the cost of slower convergence (and thus lower
performance) than its non-private counterpart. This work gives the first
convergence analysis of the DP deep learning, through the lens of training
dynamics and the neural tangent kernel (NTK). Our convergence theory
successfully characterizes the effects of two key components in the DP
training: the per-sample clipping (flat or layerwise) and the noise addition.
Our analysis not only initiates a general principled framework to understand
the DP deep learning with any network architecture and loss function, but also
motivates a new clipping method -- the global clipping, that significantly
improves the convergence while preserving the same privacy guarantee as the
existing local clipping.
In terms of theoretical results, we establish the precise connection between
the per-sample clipping and NTK matrix. We show that in the gradient flow,
i.e., with infinitesimal learning rate, the noise level of DP optimizers does
not affect the convergence. We prove that DP gradient descent (GD) with global
clipping guarantees the monotone convergence to zero loss, which can be
violated by the existing DP-GD with local clipping. Notably, our analysis
framework easily extends to other optimizers, e.g., DP-Adam. Empirically
speaking, DP optimizers equipped with global clipping perform strongly on a
wide range of classification and regression tasks. In particular, our global
clipping is surprisingly effective at learning calibrated classifiers, in
contrast to the existing DP classifiers which are oftentimes over-confident and
unreliable. Implementation-wise, the new clipping can be realized by adding one
line of code into the Opacus library.

    

### [[2106.07851] Code Integrity Attestation for PLCs using Black Box Neural Network Predictions](http://arxiv.org/abs/2106.07851)


  Cyber-physical systems (CPSs) are widespread in critical domains, and
significant damage can be caused if an attacker is able to modify the code of
their programmable logic controllers (PLCs). Unfortunately, traditional
techniques for attesting code integrity (i.e. verifying that it has not been
modified) rely on firmware access or roots-of-trust, neither of which
proprietary or legacy PLCs are likely to provide. In this paper, we propose a
practical code integrity checking solution based on privacy-preserving black
box models that instead attest the input/output behaviour of PLC programs.
Using faithful offline copies of the PLC programs, we identify their most
important inputs through an information flow analysis, execute them on multiple
combinations to collect data, then train neural networks able to predict PLC
outputs (i.e. actuator commands) from their inputs. By exploiting the black box
nature of the model, our solution maintains the privacy of the original PLC
code and does not assume that attackers are unaware of its presence. The trust
instead comes from the fact that it is extremely hard to attack the PLC code
and neural networks at the same time and with consistent outcomes. We evaluated
our approach on a modern six-stage water treatment plant testbed, finding that
it could predict actuator states from PLC inputs with near-100% accuracy, and
thus could detect all 120 effective code mutations that we subjected the PLCs
to. Finally, we found that it is not practically possible to simultaneously
modify the PLC code and apply discreet adversarial noise to our attesters in a
way that leads to consistent (mis-)predictions.

    

### [[2106.07868] Voting for the right answer: Adversarial defense for speaker verification](http://arxiv.org/abs/2106.07868)


  Automatic speaker verification (ASV) is a well developed technology for
biometric identification, and has been ubiquitous implemented in
security-critic applications, such as banking and access control. However,
previous works have shown that ASV is under the radar of adversarial attacks,
which are very similar to their original counterparts from human's perception,
yet will manipulate the ASV render wrong prediction. Due to the very late
emergence of adversarial attacks for ASV, effective countermeasures against
them are limited. Given that the security of ASV is of high priority, in this
work, we propose the idea of "voting for the right answer" to prevent risky
decisions of ASV in blind spot areas, by employing random sampling and voting.
Experimental results show that our proposed method improves the robustness
against both the limited-knowledge attackers by pulling the adversarial samples
out of the blind spots, and the perfect-knowledge attackers by introducing
randomness and increasing the attackers' budgets.

    

### [[2106.09110] Safe Reinforcement Learning Using Advantage-Based Intervention](http://arxiv.org/abs/2106.09110)


  Many sequential decision problems involve finding a policy that maximizes
total reward while obeying safety constraints. Although much recent research
has focused on the development of safe reinforcement learning (RL) algorithms
that produce a safe policy after training, ensuring safety during training as
well remains an open problem. A fundamental challenge is performing exploration
while still satisfying constraints in an unknown Markov decision process (MDP).
In this work, we address this problem for the chance-constrained setting. We
propose a new algorithm, SAILR, that uses an intervention mechanism based on
advantage functions to keep the agent safe throughout training and optimizes
the agent's policy using off-the-shelf RL algorithms designed for unconstrained
MDPs. Our method comes with strong guarantees on safety during both training
and deployment (i.e., after training and without the intervention mechanism)
and policy performance compared to the optimal safety-constrained policy. In
our experiments, we show that SAILR violates constraints far less during
training than standard safe RL and constrained MDP approaches and converges to
a well-performing policy that can be deployed safely without intervention. Our
code is available at this https URL.

    

### [[2106.10360] Prediction-Free, Real-Time Flexible Control of Tidal Lagoons through Proximal Policy Optimisation: A Case Study for the Swansea Lagoon](http://arxiv.org/abs/2106.10360)


  Tidal range structures have been considered for large scale electricity
generation for their potential ability to produce reasonable predictable energy
without the emission of greenhouse gases. Once the main forcing components for
driving the tides have deterministic dynamics, the available energy in a given
tidal power plant has been estimated, through analytical and numerical
optimisation routines, as a mostly predictable event. This constraint imposes
state-of-art flexible operation methods to rely on tidal predictions
(concurrent with measured data and up to a multiple of half-tidal cycles into
the future) to infer best operational strategies for tidal lagoons, with the
additional cost of requiring to run optimisation routines for every new tide.
In this paper, we propose a novel optimised operation of tidal lagoons with
proximal policy optimisation through Unity ML-Agents. We compare this technique
with 6 different operation optimisation approaches (baselines) devised from the
literature, utilising the Swansea Bay Tidal Lagoon as a case study. We show
that our approach is successful in maximising energy generation through an
optimised operational policy of turbines and sluices, yielding competitive
results with state-of-the-art methods of optimisation, regardless of test data
used, requiring training once and performing real-time flexible control with
measured ocean data only.

    

### [[2106.12627] Provably efficient machine learning for quantum many-body problems](http://arxiv.org/abs/2106.12627)


  Classical machine learning (ML) provides a potentially powerful approach to
solving challenging quantum many-body problems in physics and chemistry.
However, the advantages of ML over more traditional methods have not been
firmly established. In this work, we prove that classical ML algorithms can
efficiently predict ground state properties of gapped Hamiltonians in finite
spatial dimensions, after learning from data obtained by measuring other
Hamiltonians in the same quantum phase of matter. In contrast, under widely
accepted complexity theory assumptions, classical algorithms that do not learn
from data cannot achieve the same guarantee. We also prove that classical ML
algorithms can efficiently classify a wide range of quantum phases of matter.
Our arguments are based on the concept of a classical shadow, a succinct
classical description of a many-body quantum state that can be constructed in
feasible quantum experiments and be used to predict many properties of the
state. Extensive numerical experiments corroborate our theoretical results in a
variety of scenarios, including Rydberg atom systems, 2D random Heisenberg
models, symmetry-protected topological phases, and topologically ordered
phases.

    

### [[2106.13061] Fea2Fea: Exploring Structural Feature Correlations via Graph Neural Networks](http://arxiv.org/abs/2106.13061)


  Structural features are important features in graph datasets. However,
although there are some correlation analysis of features based on covariance,
there is no relevant research on exploring structural feature correlation on
graphs with graph neural network based models. In this paper, we introduce
graph feature to feature (Fea2Fea) prediction pipelines in a low dimensional
space to explore some preliminary results on structural feature correlation,
which is based on graph neural network. The results show that there exists high
correlation between some of the structural features. A non-redundant feature
combination with initial node features, which is filtered by graph neural
network has improved its classification accuracy in some graph datasets. We
compare the difference between concatenation methods on connecting embeddings
between features and show that the simplest is the best. We generalize on the
synthetic geometric graphs and certify the results on prediction difficulty
between two structural features.

    

### [[2107.00088] Inverse Design of Grating Couplers Using the Policy Gradient Method from Reinforcement Learning](http://arxiv.org/abs/2107.00088)


  We present a proof-of-concept technique for the inverse design of
electromagnetic devices motivated by the policy gradient method in
reinforcement learning, named PHORCED (PHotonic Optimization using REINFORCE
Criteria for Enhanced Design). This technique uses a probabilistic generative
neural network interfaced with an electromagnetic solver to assist in the
design of photonic devices, such as grating couplers. We show that PHORCED
obtains better performing grating coupler designs than local gradient-based
inverse design via the adjoint method, while potentially providing faster
convergence over competing state-of-the-art generative methods. Furthermore, we
implement transfer learning with PHORCED, demonstrating that a neural network
trained to optimize 8$^\circ$ grating couplers can then be re-trained on
grating couplers with alternate scattering angles while requiring >$10\times$
fewer simulations than control cases.

    

### [[2107.05556] DebiasedDTA: Model Debiasing to Boost Drug-Target Affinity Prediction](http://arxiv.org/abs/2107.05556)


  Motivation: Computational models that accurately identify high-affinity
protein-compound pairs can accelerate drug discovery pipelines. These models
aim to learn binding mechanics through drug-target interaction datasets and use
the learned knowledge for predicting the affinity of an input protein-compound
pair. However, the datasets they rely on bear misleading patterns that bias
models towards memorizing dataset-specific biomolecule properties, instead of
learning binding mechanics. This results in models that struggle while
predicting drug-target affinities (DTA), especially between de novo
biomolecules. Here we present DebiasedDTA, the first DTA model debiasing
approach that avoids dataset biases in order to boost affinity prediction for
novel biomolecules. DebiasedDTA uses ensemble learning and sample weight
adaptation for bias identification and avoidance and is applicable to almost
all existing DTA prediction models. Results: The results show that DebiasedDTA
can boost models while predicting the interactions between novel biomolecules.
Known biomolecules also benefit from the performance improvement, especially
when the test biomolecules are dissimilar to the training set. The experiments
also show that DebiasedDTA can augment DTA prediction models of different input
and model structures and is able to avoid biases of different sources.
Availability and Implementation: The source code, the models, and the datasets
are freely available for download at
this https URL, implementation in Python3,
and supported for Linux, MacOS and MS Windows. Contact:
arzucan.ozgur@boun.this http URL, elif.ozkirimli@roche.com

    

### [[1908.01089] Path Length Bounds for Gradient Descent and Flow](http://arxiv.org/abs/1908.01089)


  We derive bounds on the path length $\zeta$ of gradient descent (GD) and
gradient flow (GF) curves for various classes of smooth convex and nonconvex
functions. Among other results, we prove that: (a) if the iterates are linearly
convergent with factor $(1-c)$, then $\zeta$ is at most $\mathcal{O}(1/c)$; (b)
under the Polyak-Kurdyka-Lojasiewicz (PKL) condition, $\zeta$ is at most
$\mathcal{O}(\sqrt{\kappa})$, where $\kappa$ is the condition number, and at
least $\widetilde\Omega(\sqrt{d} \wedge \kappa^{1/4})$; (c) for quadratics,
$\zeta$ is $\Theta(\min\{\sqrt{d},\sqrt{\log \kappa}\})$ and in some cases can
be independent of $\kappa$; (d) assuming just convexity, $\zeta$ can be at most
$2^{4d\log d}$; (e) for separable quasiconvex functions, $\zeta$ is
${\Theta}(\sqrt{d})$. Thus, we advance current understanding of the properties
of GD and GF curves beyond rates of convergence. We expect our techniques to
facilitate future studies for other algorithms.

    

### [[2006.10564] Distribution-free binary classification: prediction sets, confidence intervals and calibration](http://arxiv.org/abs/2006.10564)


  We study three notions of uncertainty quantification -- calibration,
confidence intervals and prediction sets -- for binary classification in the
distribution-free setting, that is without making any distributional
assumptions on the data. With a focus towards calibration, we establish a
'tripod' of theorems that connect these three notions for score-based
classifiers. A direct implication is that distribution-free calibration is only
possible, even asymptotically, using a scoring function whose level sets
partition the feature space into at most countably many sets. Parametric
calibration schemes such as variants of Platt scaling do not satisfy this
requirement, while nonparametric schemes based on binning do. To close the
loop, we derive distribution-free confidence intervals for binned probabilities
for both fixed-width and uniform-mass binning. As a consequence of our 'tripod'
theorems, these confidence intervals for binned probabilities lead to
distribution-free calibration. We also derive extensions to settings with
streaming data and covariate shift.

    

### [[2107.08367] SpecBox: A Label-Based Transparent Speculation Scheme Against Transient Execution Attacks](http://arxiv.org/abs/2107.08367)


  Speculative execution techniques have been a cornerstone of modern processors
to improve instruction-level parallelism. However, recent studies showed that
this kind of techniques could be exploited by attackers to leak secret data via
transient execution attacks, such as Spectre. Many defenses are proposed to
address this problem, but they all face various challenges: (1) Tracking data
flow in the instruction pipeline could comprehensively address this problem,
but it could cause pipeline stalls and incur high performance overhead; (2)
Making side effect of speculative execution imperceptible to attackers, but it
often needs additional storage components and complicated data movement
operations. In this paper, we propose a label-based transparent speculation
scheme called SpecBox. It dynamically partitions the cache system to isolate
speculative data and non-speculative data, which can prevent transient
execution from being observed by subsequent execution. Moreover, it uses thread
ownership semaphores to prevent speculative data from being accessed across
cores. In addition, SpecBox also enhances the auxiliary components in the cache
system against transient execution attacks, such as hardware prefetcher. Our
security analysis shows that SpecBox is secure and the performance evaluation
shows that the performance overhead on SPEC CPU 2006 and PARSEC-3.0 benchmarks
is small.

    

### [[2107.08600] Fast polar codes for terabits-per-second throughput communications](http://arxiv.org/abs/2107.08600)


  Targeting high-throughput and low-power communications, we implement two
successive cancellation (SC) decoders for polar codes. With $16nm$ ASIC
technology, the area efficiency and energy efficiency are $4Tbps/mm^2$ and
$0.63pJ/bit$, respectively, for the unrolled decoder, and $561Gbps/mm^2$ and
$1.21pJ/bit$, respectively, for the recursive decoder. To achieve such a high
throughput, a novel code construction, coined as fast polar codes, is proposed
and jointly optimized with a highly-parallel SC decoding architecture. First,
we reuse existing modules to fast decode more outer code blocks, and then
modify code construction to facilitate faster decoding for all outer code
blocks up to a degree of parallelism of $16$. Furthermore, parallel comparison
circuits and bit quantization schemes are customized for hardware
implementation. Collectively, they contribute to an $2.66\times$ area
efficiency improvement and $33\%$ energy saving over the state of the art.

    

### [[2107.08607] A unified polar decoder platform for low-power and low-cost devices](http://arxiv.org/abs/2107.08607)


  In this paper, we design a polar decoding platform for diverse application
scenarios that require low-cost and low-power communications. Specifically,
prevalent polar decoders such as successive cancellation (SC), SC-list (SCL)
and Fano decoders are all supported under the same architecture. Unlike
high-throughput or low-latency decoders that promote parallelism, this
architecture promotes serialization by repeatedly calling a ``sub-process''
that is executed by a core module. The resulting serial SCL-8 decoder is only 3
times as big as an SC decoder. Cost and power are minimized through resource
sharing and adaptive decoding techniques, etc. We carried out performance
simulation and hardware implementation to evaluate the actual chip area and
energy consumption.

    

### [[2107.08709] ZIPPER: Exploiting Tile- and Operator-level Parallelism for General and Scalable Graph Neural Network Acceleration](http://arxiv.org/abs/2107.08709)


  Graph neural networks (GNNs) start to gain momentum after showing significant
performance improvement in a variety of domains including molecular science,
recommendation, and transportation. Turning such performance improvement of
GNNs into practical applications relies on effective and efficient execution,
especially for inference. However, neither CPU nor GPU can meet these needs if
considering both performance and energy efficiency. That's because accelerating
GNNs is challenging due to their excessive memory usage and arbitrary
interleaving of diverse operations. Besides, the semantics gap between the
high-level GNN programming model and efficient hardware makes it difficult in
accelerating general-domain GNNs.
To address the challenge, we propose Zipper, an efficient yet general
acceleration system for GNNs. The keys to Zipper include a graph-native
intermediate representation (IR) and the associated compiler. By capturing GNN
primitive operations and representing with GNN IR, Zipper is able to fit GNN
semantics into hardware structure for efficient execution. The IR also enables
GNN-specific optimizations including sparse graph tiling and redundant
operation elimination. We further present an hardware architecture design
consisting of dedicated blocks for different primitive operations, along with a
run-time scheduler to map a IR program to the hardware blocks. Our evaluation
shows that Zipper achieves 93.6x speedup and 147x energy reduction over Intel
Xeon CPU, and 1.56x speedup and 4.85x energy reduction over NVIDIA V100 GPU on
averages.

    

### [[2107.08716] NERO: Accelerating Weather Prediction using Near-Memory Reconfigurable Fabric](http://arxiv.org/abs/2107.08716)


  Ongoing climate change calls for fast and accurate weather and climate
modeling. However, when solving large-scale weather prediction simulations,
state-of-the-art CPU and GPU implementations suffer from limited performance
and high energy consumption. These implementations are dominated by complex
irregular memory access patterns and low arithmetic intensity that pose
fundamental challenges to acceleration. To overcome these challenges, we
propose and evaluate the use of near-memory acceleration using a reconfigurable
fabric with high-bandwidth memory (HBM). We focus on compound stencils that are
fundamental kernels in weather prediction models. By using high-level synthesis
techniques, we develop NERO, an FPGA+HBM-based accelerator connected through
IBM OCAPI (Open Coherent Accelerator Processor Interface) to an IBM POWER9 host
system. Our experimental results show that NERO outperforms a 16-core POWER9
system by 5.3x and 12.7x when running two different compound stencil kernels.
NERO reduces the energy consumption by 12x and 35x for the same two kernels
over the POWER9 system with an energy efficiency of 1.61 GFLOPS/Watt and 21.01
GFLOPS/Watt. We conclude that employing near-memory acceleration solutions for
weather prediction modeling is promising as a means to achieve both high
performance and high energy efficiency.

    

### [[2107.08267] Throughput Maximization of UAV Networks](http://arxiv.org/abs/2107.08267)


  In this paper we study the deployment of multiple unmanned aerial vehicles
(UAVs) to form a temporal UAV network for the provisioning of emergent
communications to affected people in a disaster zone, where each UAV is
equipped with a lightweight base station device and thus can act as an aerial
base station for users. Unlike most existing studies that assumed that a UAV
can serve all users in its communication range, we observe that both
computation and communication capabilities of a single lightweight UAV are very
limited, due to various constraints on its size, weight, and power supply.
Thus, a single UAV can only provide communication services to a limited number
of users. We study a novel problem of deploying $K$ UAVs in the top of a
disaster area such that the sum of the data rates of users served by the UAVs
is maximized, subject to that (i) the number of users served by each UAV is no
greater than its service capacity; and (ii) the communication network induced
by the $K$ UAVs is connected. We then propose a $\frac{1-1/e}{\lfloor \sqrt{K}
\rfloor}$-approximation algorithm for the problem, improving the current best
result of the problem by five times (the best approximation ratio so far is
$\frac{1-1/e}{5( \sqrt{K} +1)}$), where $e$ is the base of the natural
logarithm. We finally evaluate the algorithm performance via simulation
experiments. Experimental results show that the proposed algorithm is very
promising. Especially, the solution delivered by the proposed algorithm is up
to 12% better than those by existing algorithms.

    

### [[2107.08348] Adaptive Priority-based Conflict Resolution of IoT Services](http://arxiv.org/abs/2107.08348)


  We propose a novel conflict resolution framework for IoT services in
multi-resident smart homes. An adaptive priority model is developed considering
the residents' contextual factors (e.g., age, illness, impairment). The
proposed priority model is designed using the concept of the analytic hierarchy
process. A set of experiments on real-world datasets are conducted to show the
efficiency of the proposed approach.

    

### [[2107.08386] A Bilevel Programming Framework for Joint Edge Resource Management and Pricing](http://arxiv.org/abs/2107.08386)


  The emerging edge computing paradigm promises to provide low latency and
ubiquitous computation to numerous mobile and Internet of Things (IoT) devices
at the network edge. How to efficiently allocate geographically distributed
heterogeneous edge resources to a variety of services is a challenging task.
While this problem has been studied extensively in recent years, most of the
previous work has largely ignored the preferences of the services when making
edge resource allocation decisions. To this end, this paper introduces a novel
bilevel optimization model, which explicitly takes the service preferences into
consideration, to study the interaction between an EC platform and multiple
services. The platform manages a set of edge nodes (ENs) and acts as the leader
while the services are the followers. Given the service placement and resource
pricing decisions of the leader, each service decides how to optimally divide
its workload to different ENs. The proposed framework not only maximizes the
profit of the platform but also minimizes the cost of every service. When there
is a single EN, we derive a simple analytic solution for the underlying
problem. For the general case with multiple ENs and multiple services, we
present a Karush Kuhn Tucker based solution and a duality based solution,
combining with a series of linearizations, to solve the bilevel problem.
Extensive numerical results are shown to illustrate the efficacy of the
proposed model.

    

### [[2107.08450] Robust Composition of Drone Delivery Services under Uncertainty](http://arxiv.org/abs/2107.08450)


  We propose a novel robust composition framework for drone delivery services
considering changes in the wind patterns in urban areas. The proposed framework
incorporates the dynamic arrival of drone services at the recharging stations.
We propose a Probabilistic Forward Search (PFS) algorithm to select and compose
the best drone delivery services under uncertainty. A set of experiments with a
real drone dataset is conducted to illustrate the effectiveness and efficiency
of the proposed approach.

    

### [[2107.08496] A Practical Algorithm Design and Evaluation for Heterogeneous Elastic Computing with Stragglers](http://arxiv.org/abs/2107.08496)


  Our extensive real measurements over Amazon EC2 show that the virtual
instances often have different computing speeds even if they share the same
configurations. This motivates us to study heterogeneous Coded Storage Elastic
Computing (CSEC) systems where machines, with different computing speeds, join
and leave the network arbitrarily over different computing steps. In CSEC
systems, a Maximum Distance Separable (MDS) code is used for coded storage such
that the file placement does not have to be redefined with each elastic event.
Computation assignment algorithms are used to minimize the computation time
given computation speeds of different machines. While previous studies of
heterogeneous CSEC do not include stragglers-the slow machines during the
computation, we develop a new framework in heterogeneous CSEC that introduces
straggler tolerance. Based on this framework, we design a novel algorithm using
our previously proposed approach for heterogeneous CSEC such that the system
can handle any subset of stragglers of a specified size while minimizing the
computation time. Furthermore, we establish a trade-off in computation time and
straggler tolerance. Another major limitation of existing CSEC designs is the
lack of practical evaluations using real applications. In this paper, we
evaluate the performance of our designs on Amazon EC2 for applications of the
power iteration and linear regression. Evaluation results show that the
proposed heterogeneous CSEC algorithms outperform the state-of-the-art designs
by more than 30%.

    

### [[2107.08538] Effective GPU Sharing Under Compiler Guidance](http://arxiv.org/abs/2107.08538)


  Modern computing platforms tend to deploy multiple GPUs (2, 4, or more) on a
single node to boost system performance, with each GPU having a large capacity
of global memory and streaming multiprocessors (SMs). GPUs are an expensive
resource, and boosting utilization of GPUs without causing performance
degradation of individual workloads is an important and challenging problem.
Although services like MPS support simultaneous execution of multiple
co-operative kernels on a single device, they do not solve the above problem
for uncooperative kernels, MPS being oblivious to the resource needs of each
kernel.
We propose a fully automated compiler-assisted scheduling framework. The
compiler constructs GPU tasks by identifying kernel launches and their related
GPU operations (e.g. memory allocations). For each GPU task, a probe is
instrumented in the host-side code right before its launch point. At runtime,
the probe conveys the information about the task's resource requirements (e.g.
memory and compute cores) to a scheduler, such that the scheduler can place the
task on an appropriate device based on the task's resource requirements and
devices' load in a memory-safe, resource-aware manner. To demonstrate its
advantages, we prototyped a throughput-oriented scheduler based on the
framework, and evaluated it with the Rodinia benchmark suite and the Darknet
neural network framework on NVIDIA GPUs. The results show that the proposed
solution outperforms existing state-of-the-art solutions by leveraging its
knowledge about applications' multiple resource requirements, which include
memory as well as SMs. It improves throughput by up to 2.5x for Rodinia
benchmarks, and up to 2.7x for Darknet neural networks. In addition, it
improves job turnaround time by up to 4.9x, and limits individual kernel
performance degradation to at most 2.5%.

    

### [[2107.08628] Secure Aerial Surveillance using Split Learning](http://arxiv.org/abs/2107.08628)


  Personal monitoring devices such as cyclist helmet cameras to record
accidents or dash cams to catch collisions have proliferated, with more
companies producing smaller and compact recording gadgets. As these devices are
becoming a part of citizens' everyday arsenal, concerns over the residents'
privacy are progressing. Therefore, this paper presents SASSL, a secure aerial
surveillance drone using split learning to classify whether there is a presence
of a fire on the streets. This innovative split learning method transfers CCTV
footage captured with a drone to a nearby server to run a deep neural network
to detect a fire's presence in real-time without exposing the original data. We
devise a scenario where surveillance UAVs roam around the suburb, recording any
unnatural behavior. The UAV can process the recordings through its on-mobile
deep neural network system or transfer the information to a server. Due to the
resource limitations of mobile UAVs, the UAV does not have the capacity to run
an entire deep neural network on its own. This is where the split learning
method comes in handy. The UAV runs the deep neural network only up to the
first hidden layer and sends only the feature map to the cloud server, where
the rest of the deep neural network is processed. By ensuring that the learning
process is divided between the UAV and the server, the privacy of raw data is
secured while the UAV does not overexert its minimal resources.

    

### [[2107.08809] Revisiting the Primal-Dual Method of Multipliers for Optimisation over Centralised Networks](http://arxiv.org/abs/2107.08809)


  The primal-dual method of multipliers (PDMM) was originally designed for
solving a decomposable optimisation problem over a general network. In this
paper, we revisit PDMM for optimisation over a centralized network. We first
note that the recently proposed method FedSplit [1] implements PDMM for a
centralized network. In [1], Inexact FedSplit (i.e., gradient based FedSplit)
was also studied both empirically and theoretically. We identify the cause for
the poor reported performance of Inexact FedSplit, which is due to the improper
initialisation in the gradient operations at the client side. To fix the issue
of Inexact FedSplit, we propose two versions of Inexact PDMM, which are
referred to as gradient-based PDMM (GPDMM) and accelerated GPDMM (AGPDMM),
respectively. AGPDMM accelerates GPDMM at the cost of transmitting two times
the number of parameters from the server to each client per iteration compared
to GPDMM. We provide a new convergence bound for GPDMM for a class of convex
optimisation problems. Our new bounds are tighter than those derived for
Inexact FedSplit. We also investigate the update expressions of AGPDMM and
SCAFFOLD to find their similarities. It is found that when the number K of
gradient steps at the client side per iteration is K=1, both AGPDMM and
SCAFFOLD reduce to vanilla gradient descent with proper parameter setup.
Experimental results indicate that AGPDMM converges faster than SCAFFOLD when
K>1 while GPDMM converges slightly worse than SCAFFOLD.

    

### [[1903.00227] Parallel Weighted Random Sampling](http://arxiv.org/abs/1903.00227)


  Data structures for efficient sampling from a set of weighted items are an
important building block of many applications. However, few parallel solutions
are known. We close many of these gaps both for shared-memory and
distributed-memory machines. We give efficient, fast, and practicable parallel
algorithms for building data structures that support sampling single items
(alias tables, compressed data structures). This also yields a simplified and
more space-efficient sequential algorithm for alias table construction. Our
approaches to sampling $k$ out of $n$ items with/without replacement and to
subset (Poisson) sampling are output-sensitive, i.e., the sampling algorithms
use work linear in the number of different samples. This is also interesting in
the sequential case. Weighted random permutation can be done by sorting
appropriate random deviates. We show that this is possible with linear work
using a nonlinear transformation of these deviates. Finally, we give a
communication-efficient, highly scalable approach to (weighted and unweighted)
reservoir sampling. This algorithm is based on a fully distributed model of
streaming algorithms that might be of independent interest. Experiments for
alias tables and sampling with replacement show near linear speedups both for
construction and queries using up to 158 threads of shared-memory machines. An
experimental evaluation of distributed weighted reservoir sampling on up to 256
nodes (5120 cores) also shows good speedups.

    

### [[1903.04205] Incentives in Ethereum's Hybrid Casper Protocol](http://arxiv.org/abs/1903.04205)


  We present an overview of hybrid Casper the Friendly Finality Gadget (FFG): a
Proof-of-Stake checkpointing protocol overlaid onto Ethereum's Proof-of-Work
blockchain. We describe its core functionalities and reward scheme, and explore
its properties. Our findings indicate that Casper's implemented incentives
mechanism ensures liveness, while providing safety guarantees that improve over
standard Proof-of-Work protocols. Based on a minimal-impact implementation of
the protocol as a smart contract on the blockchain, we discuss additional
issues related to parametrisation, funding, throughput and network overhead and
detect potential limitations.

    

### [[2010.10378] Modeling Data Movement Performance on Heterogeneous Architectures](http://arxiv.org/abs/2010.10378)


  The cost of data movement on parallel systems varies greatly with machine
architecture, job partition, and nearby jobs. Performance models that
accurately capture the cost of data movement provide a tool for analysis,
allowing for communication bottlenecks to be pinpointed. Modern heterogeneous
architectures yield increased variance in data movement as there are a number
of viable paths for inter-GPU communication. In this paper, we present
performance models for the various paths of inter-node communication on modern
heterogeneous architectures, including the trade-off between GPUDirect
communication and copying to CPUs. Furthermore, we present a novel optimization
for inter-node communication based on these models, utilizing all available CPU
cores per node. Finally, we show associated performance improvements for MPI
collective operations.

    

### [[2012.01686] Dynamic Asynchronous Iterations](http://arxiv.org/abs/2012.01686)


  Many problems can be solved by iteration by multiple participants
(processors, servers, routers etc.). Previous mathematical models for such
asynchronous iterations assume a single function being iterated by a fixed set
of participants. We will call such iterations static since the system's
configuration does not change. However in several real-world examples, such as
inter-domain routing, both the function being iterated and the set of
participants change frequently while the system continues to function. In this
paper we extend Uresin & Dubois's work on static iterations to develop a model
for this class of "dynamic" or "always on" asynchronous iterations. We explore
what it means for such an iteration to be implemented correctly, and then prove
two different conditions on the set of iterated functions that guarantee the
full asynchronous iteration satisfies this new definition of correctness. These
results have been formalised in Agda and the resulting library is publicly
available.

    

### [[2103.04234] Bottlenecks in Blockchain Consensus Protocols](http://arxiv.org/abs/2103.04234)


  Most of the Blockchain permissioned systems employ Byzantine fault-tolerance
(BFT) consensus protocols to ensure that honest validators agree on the order
for appending entries to their ledgers. In this paper, we study the performance
and the scalability of prominent consensus protocols, namely PBFT, Tendermint,
HotStuff, and Streamlet, both analytically via load formulas and practically
via implementation and evaluation. Under identical conditions, we identify the
bottlenecks of these consensus protocols and show that these protocols do not
scale well as the number of validators increases. Our investigation points to
the communication complexity as the culprit. Even when there is enough network
bandwidth, the CPU cost of serialization and deserialization of the messages
limits the throughput and increases the latency of the protocols. To alleviate
the bottlenecks, the most useful techniques include reducing the communication
complexity, rotating the hotspot of communications, and pipelining across
consensus instances.

    

### [[2107.08067] DeformerNet: A Deep Learning Approach to 3D Deformable Object Manipulation](http://arxiv.org/abs/2107.08067)


  In this paper, we propose a novel approach to 3D deformable object
manipulation leveraging a deep neural network called DeformerNet. Controlling
the shape of a 3D object requires an effective state representation that can
capture the full 3D geometry of the object. Current methods work around this
problem by defining a set of feature points on the object or only deforming the
object in 2D image space, which does not truly address the 3D shape control
problem. Instead, we explicitly use 3D point clouds as the state representation
and apply Convolutional Neural Network on point clouds to learn the 3D
features. These features are then mapped to the robot end-effector's position
using a fully-connected neural network. Once trained in an end-to-end fashion,
DeformerNet directly maps the current point cloud of a deformable object, as
well as a target point cloud shape, to the desired displacement in robot
gripper position. In addition, we investigate the problem of predicting the
manipulation point location given the initial and goal shape of the object.

    

### [[2107.08122] Future Intelligent Autonomous Robots, Ethical by Design. Learning from Autonomous Cars Ethics](http://arxiv.org/abs/2107.08122)


  Development of the intelligent autonomous robot technology presupposes its
anticipated beneficial effect on the individuals and societies. In the case of
such disruptive emergent technology, not only questions of how to build, but
also why to build and with what consequences are important. The field of ethics
of intelligent autonomous robotic cars is a good example of research with
actionable practical value, where a variety of stakeholders, including the
legal system and other societal and governmental actors, as well as companies
and businesses, collaborate bringing about shared view of ethics and societal
aspects of technology. It could be used as a starting platform for the
approaches to the development of intelligent autonomous robots in general,
considering human-machine interfaces in different phases of the life cycle of
technology - the development, implementation, testing, use and disposal.
Drawing from our work on ethics of autonomous intelligent robocars, and the
existing literature on ethics of robotics, our contribution consists of a set
of values and ethical principles with identified challenges and proposed
approaches for meeting them. This may help stakeholders in the field of
intelligent autonomous robotics to connect ethical principles with their
applications. Our recommendations of ethical requirements for autonomous cars
can be used for other types of intelligent autonomous robots, with the caveat
for social robots that require more research regarding interactions with the
users. We emphasize that existing ethical frameworks need to be applied in a
context-sensitive way, by assessments in interdisciplinary, multi-competent
teams through multi-criteria analysis. Furthermore, we argue for the need of a
continuous development of ethical principles, guidelines, and regulations,
informed by the progress of technologies and involving relevant stakeholders.

    

### [[2107.08124] Architectures of Meaning, A Systematic Corpus Analysis of NLP Systems](http://arxiv.org/abs/2107.08124)


  This paper proposes a novel statistical corpus analysis framework targeted
towards the interpretation of Natural Language Processing (NLP) architectural
patterns at scale. The proposed approach combines saturation-based lexicon
construction, statistical corpus analysis methods and graph collocations to
induce a synthesis representation of NLP architectural patterns from corpora.
The framework is validated in the full corpus of Semeval tasks and demonstrated
coherent architectural patterns which can be used to answer architectural
questions on a data-driven fashion, providing a systematic mechanism to
interpret a largely dynamic and exponentially growing field.

    

### [[2107.08166] Data-informed Deep Optimization](http://arxiv.org/abs/2107.08166)


  Complex design problems are common in the scientific and industrial fields.
In practice, objective functions or constraints of these problems often do not
have explicit formulas, and can be estimated only at a set of sampling points
through experiments or simulations. Such optimization problems are especially
challenging when design parameters are high-dimensional due to the curse of
dimensionality. In this work, we propose a data-informed deep optimization
(DiDo) approach as follows: first, we use a deep neural network (DNN)
classifier to learn the feasible region; second, we sample feasible points
based on the DNN classifier for fitting of the objective function; finally, we
find optimal points of the DNN-surrogate optimization problem by gradient
descent. To demonstrate the effectiveness of our DiDo approach, we consider a
practical design case in industry, in which our approach yields good solutions
using limited size of training data. We further use a 100-dimension toy example
to show the effectiveness of our model for higher dimensional problems. Our
results indicate that the DiDo approach empowered by DNN is flexible and
promising for solving general high-dimensional design problems in practice.

    

### [[2107.08252] Constraint Answer Set Programming: Integrational and Translational (or SMT-based) Approaches](http://arxiv.org/abs/2107.08252)


  Constraint answer set programming or CASP, for short, is a hybrid approach in
automated reasoning putting together the advances of distinct research areas
such as answer set programming, constraint processing, and satisfiability
modulo theories. Constraint answer set programming demonstrates promising
results, including the development of a multitude of solvers: acsolver,
clingcon, ezcsp, idp, inca, dingo, mingo, aspmt, clingo[l,dl], and ezsmt. It
opens new horizons for declarative programming applications such as solving
complex train scheduling problems. Systems designed to find solutions to
constraint answer set programs can be grouped according to their construction
into, what we call, integrational or translational approaches. The focus of
this paper is an overview of the key ingredients of the design of constraint
answer set solvers drawing distinctions and parallels between integrational and
translational approaches. The paper also provides a glimpse at the kind of
programs its users develop by utilizing a CASP encoding of Travelling Salesman
problem for illustration. In addition, we place the CASP technology on the map
among its automated reasoning peers as well as discuss future possibilities for
the development of CASP.

    

### [[2107.08262] Tea: Program Repair Using Neural Network Based on Program Information Attention Matrix](http://arxiv.org/abs/2107.08262)


  The advance in machine learning (ML)-driven natural language process (NLP)
points a promising direction for automatic bug fixing for software programs, as
fixing a buggy program can be transformed to a translation task. While software
programs contain much richer information than one-dimensional natural language
documents, pioneering work on using ML-driven NLP techniques for automatic
program repair only considered a limited set of such information. We
hypothesize that more comprehensive information of software programs, if
appropriately utilized, can improve the effectiveness of ML-driven NLP
approaches in repairing software programs. As the first step towards proving
this hypothesis, we propose a unified representation to capture the syntax,
data flow, and control flow aspects of software programs, and devise a method
to use such a representation to guide the transformer model from NLP in better
understanding and fixing buggy programs. Our preliminary experiment confirms
that the more comprehensive information of software programs used, the better
ML-driven NLP techniques can perform in fixing bugs in these programs.

    

### [[2107.08295] Implicit Communication as Minimum Entropy Coupling](http://arxiv.org/abs/2107.08295)


  In many common-payoff games, achieving good performance requires players to
develop protocols for communicating their private information implicitly --
i.e., using actions that have non-communicative effects on the environment.
Multi-agent reinforcement learning practitioners typically approach this
problem using independent learning methods in the hope that agents will learn
implicit communication as a byproduct of expected return maximization.
Unfortunately, independent learning methods are incapable of doing this in many
settings. In this work, we isolate the implicit communication problem by
identifying a class of partially observable common-payoff games, which we call
implicit referential games, whose difficulty can be attributed to implicit
communication. Next, we introduce a principled method based on minimum entropy
coupling that leverages the structure of implicit referential games, yielding a
new perspective on implicit communication. Lastly, we show that this method can
discover performant implicit communication protocols in settings with very
large spaces of messages.

    

### [[2107.08325] Vision-Based Autonomous Car Racing Using Deep Imitative Reinforcement Learning](http://arxiv.org/abs/2107.08325)


  Autonomous car racing is a challenging task in the robotic control area.
Traditional modular methods require accurate mapping, localization and
planning, which makes them computationally inefficient and sensitive to
environmental changes. Recently, deep-learning-based end-to-end systems have
shown promising results for autonomous driving/racing. However, they are
commonly implemented by supervised imitation learning (IL), which suffers from
the distribution mismatch problem, or by reinforcement learning (RL), which
requires a huge amount of risky interaction data. In this work, we present a
general deep imitative reinforcement learning approach (DIRL), which
successfully achieves agile autonomous racing using visual inputs. The driving
knowledge is acquired from both IL and model-based RL, where the agent can
learn from human teachers as well as perform self-improvement by safely
interacting with an offline world model. We validate our algorithm both in a
high-fidelity driving simulation and on a real-world 1/20-scale RC-car with
limited onboard computation. The evaluation results demonstrate that our method
outperforms previous IL and RL methods in terms of sample efficiency and task
performance. Demonstration videos are available at
this https URL


### [[2107.08361] An Improved StarGAN for Emotional Voice Conversion: Enhancing Voice Quality and Data Augmentation](http://arxiv.org/abs/2107.08361)


  Emotional Voice Conversion (EVC) aims to convert the emotional style of a
source speech signal to a target style while preserving its content and speaker
identity information. Previous emotional conversion studies do not disentangle
emotional information from emotion-independent information that should be
preserved, thus transforming it all in a monolithic manner and generating audio
of low quality, with linguistic distortions. To address this distortion
problem, we propose a novel StarGAN framework along with a two-stage training
process that separates emotional features from those independent of emotion by
using an autoencoder with two encoders as the generator of the Generative
Adversarial Network (GAN). The proposed model achieves favourable results in
both the objective evaluation and the subjective evaluation in terms of
distortion, which reveals that the proposed model can effectively reduce
distortion. Furthermore, in data augmentation experiments for end-to-end speech
emotion recognition, the proposed StarGAN model achieves an increase of 2% in
Micro-F1 and 5% in Macro-F1 compared to the baseline StarGAN model, which
indicates that the proposed model is more valuable for data augmentation.

    

### [[2107.08378] Co-designing Intelligent Control of Building HVACs and Microgrids](http://arxiv.org/abs/2107.08378)


  Building loads consume roughly 40% of the energy produced in developed
countries, a significant part of which is invested towards building
temperature-control infrastructure. Therein, renewable resource-based
microgrids offer a greener and cheaper alternative. This communication explores
the possible co-design of microgrid power dispatch and building HVAC (heating,
ventilation and air conditioning system) actuations with the objective of
effective temperature control under minimised operating cost. For this, we
attempt control designs with various levels of abstractions based on
information available about microgrid and HVAC system models using the Deep
Reinforcement Learning (DRL) technique. We provide control architectures that
consider model information ranging from completely determined system models to
systems with fully unknown parameter settings and illustrate the advantages of
DRL for the design prescriptions.

    

### [[2107.08379] A Survey on Role-Oriented Network Embedding](http://arxiv.org/abs/2107.08379)


  Recently, Network Embedding (NE) has become one of the most attractive
research topics in machine learning and data mining. NE approaches have
achieved promising performance in various of graph mining tasks including link
prediction and node clustering and classification. A wide variety of NE methods
focus on the proximity of networks. They learn community-oriented embedding for
each node, where the corresponding representations are similar if two nodes are
closer to each other in the network. Meanwhile, there is another type of
structural similarity, i.e., role-based similarity, which is usually
complementary and completely different from the proximity. In order to preserve
the role-based structural similarity, the problem of role-oriented NE is
raised. However, compared to community-oriented NE problem, there are only a
few role-oriented embedding approaches proposed recently. Although less
explored, considering the importance of roles in analyzing networks and many
applications that role-oriented NE can shed light on, it is necessary and
timely to provide a comprehensive overview of existing role-oriented NE
methods. In this review, we first clarify the differences between
community-oriented and role-oriented network embedding. Afterwards, we propose
a general framework for understanding role-oriented NE and a two-level
categorization to better classify existing methods. Then, we select some
representative methods according to the proposed categorization and briefly
introduce them by discussing their motivation, development and differences.
Moreover, we conduct comprehensive experiments to empirically evaluate these
methods on a variety of role-related tasks including node classification and
clustering (role discovery), top-k similarity search and visualization using
some widely used synthetic and real-world datasets...

    

### [[2107.08403] Reasoning about actions with EL ontologies with temporal answer sets](http://arxiv.org/abs/2107.08403)


  We propose an approach based on Answer Set Programming for reasoning about
actions with domain descriptions including ontological knowledge, expressed in
the lightweight description logic EL^\bot. We consider a temporal action
theory, which allows for non-deterministic actions and causal rules to deal
with ramifications, and whose extensions are defined by temporal answer sets.
We provide conditions under which action consistency can be guaranteed with
respect to an ontology, by a polynomial encoding of an action theory extended
with an EL^\bot knowledge base (in normal form) into a temporal action theory.

    

### [[2107.08408] Pre-trained Language Models as Prior Knowledge for Playing Text-based Games](http://arxiv.org/abs/2107.08408)


  Recently, text world games have been proposed to enable artificial agents to
understand and reason about real-world scenarios. These text-based games are
challenging for artificial agents, as it requires understanding and interaction
using natural language in a partially observable environment. In this paper, we
improve the semantic understanding of the agent by proposing a simple RL with
LM framework where we use transformer-based language models with Deep RL
models. We perform a detailed study of our framework to demonstrate how our
model outperforms all existing agents on the popular game, Zork1, to achieve a
score of 44.7, which is 1.6 higher than the state-of-the-art model. Our
proposed approach also performs comparably to the state-of-the-art models on
the other set of text games.

    

### [[2107.08465] Compressed particle methods for expensive models with application in Astronomy and Remote Sensing](http://arxiv.org/abs/2107.08465)


  In many inference problems, the evaluation of complex and costly models is
often required. In this context, Bayesian methods have become very popular in
several fields over the last years, in order to obtain parameter inversion,
model selection or uncertainty quantification. Bayesian inference requires the
approximation of complicated integrals involving (often costly) posterior
distributions. Generally, this approximation is obtained by means of Monte
Carlo (MC) methods. In order to reduce the computational cost of the
corresponding technique, surrogate models (also called emulators) are often
employed. Another alternative approach is the so-called Approximate Bayesian
Computation (ABC) scheme. ABC does not require the evaluation of the costly
model but the ability to simulate artificial data according to that model.
Moreover, in ABC, the choice of a suitable distance between real and artificial
data is also required. In this work, we introduce a novel approach where the
expensive model is evaluated only in some well-chosen samples. The selection of
these nodes is based on the so-called compressed Monte Carlo (CMC) scheme. We
provide theoretical results supporting the novel algorithms and give empirical
evidence of the performance of the proposed method in several numerical
experiments. Two of them are real-world applications in astronomy and satellite
remote sensing.

    

### [[2107.08581] A Systematical Solution for Face De-identification](http://arxiv.org/abs/2107.08581)


  With the identity information in face data more closely related to personal
credit and property security, people pay increasing attention to the protection
of face data privacy. In different tasks, people have various requirements for
face de-identification (De-ID), so we propose a systematical solution
compatible for these De-ID operations. Firstly, an attribute disentanglement
and generative network is constructed to encode two parts of the face, which
are the identity (facial features like mouth, nose and eyes) and expression
(including expression, pose and illumination). Through face swapping, we can
remove the original ID completely. Secondly, we add an adversarial vector
mapping network to perturb the latent code of the face image, different from
previous traditional adversarial methods. Through this, we can construct
unrestricted adversarial image to decrease ID similarity recognized by model.
Our method can flexibly de-identify the face data in various ways and the
processed images have high image quality.

    

### [[2107.08590] EvilModel: Hiding Malware Inside of Neural Network Models](http://arxiv.org/abs/2107.08590)


  Delivering malware covertly and detection-evadingly is critical to advanced
malware campaigns. In this paper, we present a method that delivers malware
covertly and detection-evadingly through neural network models. Neural network
models are poorly explainable and have a good generalization ability. By
embedding malware into the neurons, malware can be delivered covertly with
minor or even no impact on the performance of neural networks. Meanwhile, since
the structure of the neural network models remains unchanged, they can pass the
security scan of antivirus engines. Experiments show that 36.9MB of malware can
be embedded into a 178MB-AlexNet model within 1% accuracy loss, and no
suspicious are raised by antivirus engines in VirusTotal, which verifies the
feasibility of this method. With the widespread application of artificial
intelligence, utilizing neural networks becomes a forwarding trend of malware.
We hope this work could provide a referenceable scenario for the defense on
neural network-assisted attacks.

    

### [[2107.08621] Face.evoLVe: A High-Performance Face Recognition Library](http://arxiv.org/abs/2107.08621)


  In this paper, we develop face.evoLVe -- a comprehensive library that
collects and implements a wide range of popular deep learning-based methods for
face recognition. First of all, face.evoLVe is composed of key components that
cover the full process of face analytics, including face alignment, data
processing, various backbones, losses, and alternatives with bags of tricks for
improving performance. Later, face.evoLVe supports multi-GPU training on top of
different deep learning platforms, such as PyTorch and PaddlePaddle, which
facilitates researchers to work on both large-scale datasets with millions of
images and low-shot counterparts with limited well-annotated data. More
importantly, along with face.evoLVe, images before & after alignment in the
common benchmark datasets are released with source codes and trained models
provided. All these efforts lower the technical burdens in reproducing the
existing methods for comparison, while users of our library could focus on
developing advanced approaches more efficiently. Last but not least,
face.evoLVe is well designed and vibrantly evolving, so that new face
recognition approaches can be easily plugged into our framework. Note that we
have used face.evoLVe to participate in a number of face recognition
competitions and secured the first place. The version that supports PyTorch is
publicly available at this https URL and the
PaddlePaddle version is available at
this https URL.
Face.evoLVe has been widely used for face analytics, receiving 2.4K stars and
622 forks.

    

### [[2107.08630] Data Sharing Markets](http://arxiv.org/abs/2107.08630)


  With the growing use of distributed machine learning techniques, there is a
growing need for data markets that allows agents to share data with each other.
Nevertheless data has unique features that separates it from other commodities
including replicability, cost of sharing, and ability to distort. We study a
setup where each agent can be both buyer and seller of data. For this setup, we
consider two cases: bilateral data exchange (trading data with data) and
unilateral data exchange (trading data with money). We model bilateral sharing
as a network formation game and show the existence of strongly stable outcome
under the top agents property by allowing limited complementarity. We propose
ordered match algorithm which can find the stable outcome in O(N^2) (N is the
number of agents). For the unilateral sharing, under the assumption of additive
cost structure, we construct competitive prices that can implement any social
welfare maximizing outcome. Finally for this setup when agents have private
information, we propose mixed-VCG mechanism which uses zero cost data
distortion of data sharing with its isolated impact to achieve budget balance
while truthfully implementing socially optimal outcomes to the exact level of
budget imbalance of standard VCG mechanisms. Mixed-VCG uses data distortions as
data money for this purpose. We further relax zero cost data distortion
assumption by proposing distorted-mixed-VCG. We also extend our model and
results to data sharing via incremental inquiries and differential privacy
costs.

    

### [[2107.08739] E-PDDL: A Standardized Way of Defining Epistemic Planning Problems](http://arxiv.org/abs/2107.08739)


  Epistemic Planning (EP) refers to an automated planning setting where the
agent reasons in the space of knowledge states and tries to find a plan to
reach a desirable state from the current state. Its general form, the
Multi-agent Epistemic Planning (MEP) problem involves multiple agents who need
to reason about both the state of the world and the information flow between
agents. In a MEP problem, multiple approaches have been developed recently with
varying restrictions, such as considering only the concept of knowledge while
not allowing the idea of belief, or not allowing for ``complex" modal operators
such as those needed to handle dynamic common knowledge. While the diversity of
approaches has led to a deeper understanding of the problem space, the lack of
a standardized way to specify MEP problems independently of solution approaches
has created difficulties in comparing performance of planners, identifying
promising techniques, exploring new strategies like ensemble methods, and
making it easy for new researchers to contribute to this research area. To
address the situation, we propose a unified way of specifying EP problems - the
Epistemic Planning Domain Definition Language, E-PDDL. We show that E-PPDL can
be supported by leading MEP planners and provide corresponding parser code that
translates EP problems specified in E-PDDL into (M)EP problems that can be
handled by several planners. This work is also useful in building more general
epistemic planning environments where we envision a meta-cognitive module that
takes a planning problem in E-PDDL, identifies and assesses some of its
features, and autonomously decides which planner is the best one to solve it.

    

### [[2107.08760] CVEfixes: Automated Collection of Vulnerabilities and Their Fixes from Open-Source Software](http://arxiv.org/abs/2107.08760)


  Data-driven research on the automated discovery and repair of security
vulnerabilities in source code requires comprehensive datasets of real-life
vulnerable code and their fixes. To assist in such research, we propose a
method to automatically collect and curate a comprehensive vulnerability
dataset from Common Vulnerabilities and Exposures (CVE) records in the public
National Vulnerability Database (NVD). We implement our approach in a fully
automated dataset collection tool and share an initial release of the resulting
vulnerability dataset named CVEfixes.
The CVEfixes collection tool automatically fetches all available CVE records
from the NVD, gathers the vulnerable code and corresponding fixes from
associated open-source repositories, and organizes the collected information in
a relational database. Moreover, the dataset is enriched with meta-data such as
programming language, and detailed code and security metrics at five levels of
abstraction. The collection can easily be repeated to keep up-to-date with
newly discovered or patched vulnerabilities. The initial release of CVEfixes
spans all published CVEs up to 9 June 2021, covering 5365 CVE records for 1754
open-source projects that were addressed in a total of 5495 vulnerability
fixing commits.
CVEfixes supports various types of data-driven software security research,
such as vulnerability prediction, vulnerability classification, vulnerability
severity prediction, analysis of vulnerability-related code changes, and
automated vulnerability repair.

    

### [[2107.08803] Channel-wise Gated Res2Net: Towards Robust Detection of Synthetic Speech Attacks](http://arxiv.org/abs/2107.08803)


  Existing approaches for anti-spoofing in automatic speaker verification (ASV)
still lack generalizability to unseen attacks. The Res2Net approach designs a
residual-like connection between feature groups within one block, which
increases the possible receptive fields and improves the system's detection
generalizability. However, such a residual-like connection is performed by a
direct addition between feature groups without channel-wise priority. We argue
that the information across channels may not contribute to spoofing cues
equally, and the less relevant channels are expected to be suppressed before
adding onto the next feature group, so that the system can generalize better to
unseen attacks. This argument motivates the current work that presents a novel,
channel-wise gated Res2Net (CG-Res2Net), which modifies Res2Net to enable a
channel-wise gating mechanism in the connection between feature groups. This
gating mechanism dynamically selects channel-wise features based on the input,
to suppress the less relevant channels and enhance the detection
generalizability. Three gating mechanisms with different structures are
proposed and integrated into Res2Net. Experimental results conducted on
ASVspoof 2019 logical access (LA) demonstrate that the proposed CG-Res2Net
significantly outperforms Res2Net on both the overall LA evaluation set and
individual difficult unseen attacks, which also outperforms other
state-of-the-art single systems, depicting the effectiveness of our method.

    

### [[2107.08842] Relative Localization of Mobile Robots with Multiple Ultra-WideBand Ranging Measurements](http://arxiv.org/abs/2107.08842)


  Relative localization between autonomous robots without infrastructure is
crucial to achieve their navigation, path planning, and formation in many
applications, such as emergency response, where acquiring a prior knowledge of
the environment is not possible. The traditional Ultra-WideBand (UWB)-based
approach provides a good estimation of the distance between the robots, but
obtaining the relative pose (including the displacement and orientation)
remains challenging. We propose an approach to estimate the relative pose
between a group of robots by equipping each robot with multiple UWB ranging
nodes. We determine the pose between two robots by minimizing the residual
error of the ranging measurements from all UWB nodes. To improve the
localization accuracy, we propose to utilize the odometry constraints through a
sliding window-based optimization. The optimized pose is then fused with the
odometry in a particle filtering for pose tracking among a group of mobile
robots. We have conducted extensive experiments to validate the effectiveness
of the proposed approach.

    

### [[2107.08908] Dynamic Cat Swarm Optimization Algorithm for Backboard Wiring Problem](http://arxiv.org/abs/2107.08908)


  This paper presents a powerful swarm intelligence meta-heuristic optimization
algorithm called Dynamic Cat Swarm Optimization. The formulation is through
modifying the existing Cat Swarm Optimization. The original Cat Swarm
Optimization suffers from the shortcoming of 'premature convergence', which is
the possibility of entrapment in local optima which usually happens due to the
off-balance between exploration and exploitation phases. Therefore, the
proposed algorithm suggests a new method to provide a proper balance between
these phases by modifying the selection scheme and the seeking mode of the
algorithm. To evaluate the performance of the proposed algorithm, 23 classical
test functions, 10 modern test functions (CEC 2019) and a real world scenario
are used. In addition, the Dimension-wise diversity metric is used to measure
the percentage of the exploration and exploitation phases. The optimization
results show the effectiveness of the proposed algorithm, which ranks first
compared to several well-known algorithms available in the literature.
Furthermore, statistical methods and graphs are also used to further confirm
the outperformance of the algorithm. Finally, the conclusion as well as future
directions to further improve the algorithm are discussed.

    

### [[1906.00908] Phase-based Minimalist Parsing and complexity in non-local dependencies](http://arxiv.org/abs/1906.00908)


  A cognitively plausible parsing algorithm should perform like the human
parser in critical contexts. Here I propose an adaptation of Earley's parsing
algorithm, suitable for Phase-based Minimalist Grammars (PMG, Chesi 2012), that
is able to predict complexity effects in performance. Focusing on self-paced
reading experiments of object clefts sentences (Warren & Gibson 2005) I will
associate to parsing a complexity metric based on cued features to be retrieved
at the verb segment (Feature Retrieval & Encoding Cost, FREC). FREC is
crucially based on the usage of memory predicted by the discussed parsing
algorithm and it correctly fits with the reading time revealed.

    

### [[2004.01697] Designer Modeling through Design Style Clustering](http://arxiv.org/abs/2004.01697)


  We propose modeling designer style in mixed-initiative game content creation
tools as archetypical design traces. These design traces are formulated as
transitions between design styles; these design styles are in turn found
through clustering all intermediate designs along the way to making a complete
design. This method is implemented in the Evolutionary Dungeon Designer, a
research platform for mixed-initiative systems to create roguelike games. We
present results both in the form of design styles for rooms, which can be
analyzed to better understand the kind of rooms designed by users, and in the
form of archetypical sequences between these rooms. We further discuss how the
results here can be used to create style-sensitive suggestions. Such
suggestions would allow the system to be one step ahead of the designer,
offering suggestions for the next cluster, assuming that the designer will
follow one of the archetypical design traces.

    

### [[2007.04477] Good AI for the Present of Humanity Democratizing AI Governance](http://arxiv.org/abs/2007.04477)


  What do Cyberpunk and AI Ethics have to do with each other? Cyberpunk is a
sub-genre of science fiction that explores the post-human relationships between
human experience and technology. One similarity between AI Ethics and Cyberpunk
literature is that both seek to explore future social and ethical problems that
our technological advances may bring upon society. In recent years, an
increasing number of ethical matters involving AI have been pointed and
debated, and several ethical principles and guides have been suggested as
governance policies for the tech industry. However, would this be the role of
AI Ethics? To serve as a soft and ambiguous version of the law? We would like
to advocate in this article for a more Cyberpunk way of doing AI Ethics, with a
more democratic way of governance. In this study, we will seek to expose some
of the deficits of the underlying power structures of the AI industry, and
suggest that AI governance be subject to public opinion, so that good AI can
become good AI for all.

    

### [[2009.06110] Identity-Based Patterns in Deep Convolutional Networks: Generative Adversarial Phonology and Reduplication](http://arxiv.org/abs/2009.06110)


  This paper models unsupervised learning of an identity-based pattern (or
copying) in speech called reduplication from raw continuous data with deep
convolutional neural networks. We use the ciwGAN architecture Beguš (2021a;
arXiv:2006.02951) in which learning of meaningful representations in speech
emerges from a requirement that the CNNs generate informative data. We propose
a technique to wug-test CNNs trained on speech and, based on four generative
tests, argue that the network learns to represent an identity-based pattern in
its latent space. By manipulating only two categorical variables in the latent
space, we can actively turn an unreduplicated form into a reduplicated form
with no other substantial changes to the output in the majority of cases. We
also argue that the network extends the identity-based pattern to unobserved
data. Exploration of how meaningful representations of identity-based patterns
emerge in CNNs and how the latent space variables outside of the training range
correlate with identity-based patterns in the output has general implications
for neural network interpretability.

    

### [[2010.05357] A Knowledge-Driven Approach to Classifying Object and Attribute Coreferences in Opinion Mining](http://arxiv.org/abs/2010.05357)


  Classifying and resolving coreferences of objects (e.g., product names) and
attributes (e.g., product aspects) in opinionated reviews is crucial for
improving the opinion mining performance. However, the task is challenging as
one often needs to consider domain-specific knowledge (e.g., iPad is a tablet
and has aspect resolution) to identify coreferences in opinionated reviews.
Also, compiling a handcrafted and curated domain-specific knowledge base for
each domain is very time consuming and arduous. This paper proposes an approach
to automatically mine and leverage domain-specific knowledge for classifying
objects and attribute coreferences. The approach extracts domain-specific
knowledge from unlabeled review data and trains a knowledgeaware neural
coreference classification model to leverage (useful) domain knowledge together
with general commonsense knowledge for the task. Experimental evaluation on
realworld datasets involving five domains (product types) shows the
effectiveness of the approach.

    

### [[2011.09410] A Definition and a Test for Human-Level Artificial Intelligence](http://arxiv.org/abs/2011.09410)


  Despite recent advances in many application-specific domains, we do not know
how to build a human-level artificial intelligence (HLAI). We conjecture that
learning from others' experience with the language is the essential
characteristic that distinguishes human intelligence from the rest. Humans can
update the action-value function with the verbal description as if they
experience states, actions, and corresponding rewards sequences firsthand. In
this paper, we present a classification of intelligence according to how
individual agents learn and propose a definition and a test for HLAI. The main
idea is that language acquisition without explicit rewards can be a sufficient
test for HLAI.

    

### [[2012.12060] Information Leakage Games: Exploring Information as a Utility Function](http://arxiv.org/abs/2012.12060)


  A common goal in the areas of secure information flow and privacy is to build
effective defenses against unwanted leakage of information. To this end, one
must be able to reason about potential attacks and their interplay with
possible defenses. In this paper, we propose a game-theoretic framework to
formalize strategies of attacker and defender in the context of information
leakage, and provide a basis for developing optimal defense methods. A novelty
of our games is that their utility is given by information leakage, which in
some cases may behave in a non-linear way. This causes a significant deviation
from classic game theory, in which utility functions are linear with respect to
players' strategies. Hence, a key contribution of this paper is the
establishment of the foundations of information leakage games. We consider two
kinds of games, depending on the notion of leakage considered. The first kind,
the QIF-games, is tailored for the theory of quantitative information flow
(QIF). The second one, the DP-games, corresponds to differential privacy (DP).

    

### [[2102.07537] CHARET: Character-centered Approach to Emotion Tracking in Stories](http://arxiv.org/abs/2102.07537)


  Autonomous agents that can engage in social interactions witha human is the
ultimate goal of a myriad of applications. A keychallenge in the design of
these applications is to define the socialbehavior of the agent, which requires
extensive content this http URL this research, we explore how we can leverage
current state-of-the-art tools to make inferences about the emotional state ofa
character in a story as events unfold, in a coherent way. Wepropose a character
role-labelling approach to emotion tracking thataccounts for the semantics of
emotions. We show that by identifyingactors and objects of events and
considering the emotional stateof the characters, we can achieve better
performance in this task,when compared to end-to-end approaches.

    

### [[2103.13870] A novel weighted approach for time series forecasting based on visibility graph](http://arxiv.org/abs/2103.13870)


  Time series have attracted widespread attention in many fields today. Based
on the analysis of complex networks and visibility graph theory, a new time
series forecasting method is proposed. In time series analysis, visibility
graph theory transforms time series data into a network model. In the network
model, the node similarity index is an important factor. On the basis of
directly using the node prediction method with the largest similarity, the node
similarity index is used as the weight coefficient to optimize the prediction
algorithm. Compared with the single-node sampling node prediction algorithm,
the multi-node sampling prediction algorithm can provide more accurate
prediction values when the data set is sufficient. According to results of
experiments on four real-world representative datasets, the method has more
accurate forecasting ability and can provide more accurate forecasts in the
field of time series and actual scenes.

    

### [[2104.05204] A Fast Evidential Approach for Stock Forecasting](http://arxiv.org/abs/2104.05204)


  Within the framework of evidence theory, the confidence functions of
different information can be combined into a combined confidence function to
solve uncertain problems. The Dempster combination rule is a classic method of
fusing different information. This paper proposes a similar confidence function
for the time point in the time series. The Dempster combination rule can be
used to fuse the growth rate of the last time point, and finally a relatively
accurate forecast data can be obtained. Stock price forecasting is a concern of
economics. The stock price data is large in volume, and more accurate forecasts
are required at the same time. The classic methods of time series, such as
ARIMA, cannot balance forecasting efficiency and forecasting accuracy at the
same time. In this paper, the fusion method of evidence theory is applied to
stock price prediction. Evidence theory deals with the uncertainty of stock
price prediction and improves the accuracy of prediction. At the same time, the
fusion method of evidence theory has low time complexity and fast prediction
processing speed.

    

### [[2105.02331] Safety Enhancement for Deep Reinforcement Learning in Autonomous Separation Assurance](http://arxiv.org/abs/2105.02331)


  The separation assurance task will be extremely challenging for air traffic
controllers in a complex and high density airspace environment. Deep
reinforcement learning (DRL) was used to develop an autonomous separation
assurance framework in our previous work where the learned model advised speed
maneuvers. In order to improve the safety of this model in unseen environments
with uncertainties, in this work we propose a safety module for DRL in
autonomous separation assurance applications. The proposed module directly
addresses both model uncertainty and state uncertainty to improve safety. Our
safety module consists of two sub-modules: (1) the state safety sub-module is
based on the execution-time data augmentation method to introduce state
disturbances in the model input state; (2) the model safety sub-module is a
Monte-Carlo dropout extension that learns the posterior distribution of the DRL
model policy. We demonstrate the effectiveness of the two sub-modules in an
open-source air traffic simulator with challenging environment settings.
Through extensive numerical experiments, our results show that the proposed
sub-safety modules help the DRL agent significantly improve its safety
performance in an autonomous separation assurance task.

    

### [[2105.02704] AI Risk Skepticism](http://arxiv.org/abs/2105.02704)


  In this work, we survey skepticism regarding AI risk and show parallels with
other types of scientific skepticism. We start by classifying different types
of AI Risk skepticism and analyze their root causes. We conclude by suggesting
some intervention approaches, which may be successful in reducing AI risk
skepticism, at least amongst artificial intelligence researchers.

    

### [[2105.08715] Human Motion Prediction Using Manifold-Aware Wasserstein GAN](http://arxiv.org/abs/2105.08715)


  Human motion prediction aims to forecast future human poses given a prior
pose sequence. The discontinuity of the predicted motion and the performance
deterioration in long-term horizons are still the main challenges encountered
in current literature. In this work, we tackle these issues by using a compact
manifold-valued representation of human motion. Specifically, we model the
temporal evolution of the 3D human poses as trajectory, what allows us to map
human motions to single points on a sphere manifold. To learn these
non-Euclidean representations, we build a manifold-aware Wasserstein generative
adversarial model that captures the temporal and spatial dependencies of human
motion through different losses. Extensive experiments show that our approach
outperforms the state-of-the-art on CMU MoCap and Human 3.6M datasets. Our
qualitative results show the smoothness of the predicted motions.

    

### [[2105.09464] Content-Augmented Feature Pyramid Network with Light Linear Spatial Transformers for Object Detection](http://arxiv.org/abs/2105.09464)


  As one of the prevalent components, Feature Pyramid Network (FPN) is widely
used in the current object detection models to improve the performance of
multi-scale detection. However, its interaction is still in a local and lossy
manner, thus limiting the representation power. In this paper, to simulate a
global view of human vision in object detection and address the inherent
defects of interaction mode in FPN, we construct a novel architecture termed
Content-Augmented Feature Pyramid Network (CA-FPN). Unlike the vanilla FPN,
which fuses features within a local receptive field, CA-FPN can adaptively
aggregate similar features from a global view. It is equipped with a global
content extraction module and light linear spatial transformers. The former
allows to extract multi-scale context information and the latter can deeply
combine the global content extraction module with the vanilla FPN using the
linearized attention function, which is designed to reduce model complexity.
Furthermore, CA-FPN can be readily plugged into existing FPN-based models.
Extensive experiments on the challenging COCO and PASCAL VOC object detection
datasets demonstrated that our CA-FPN significantly outperforms competitive
FPN-based detectors without bells and whistles. When plugging CA-FPN into
Cascade R-CNN framework built upon a standard ResNet-50 backbone, our method
can achieve 44.8 AP on COCO mini-val. Its performance surpasses the previous
state-of-the-art by 1.5 AP, demonstrating the potentiality of application.

    

### [[2106.04316] Exploration and preference satisfaction trade-off in reward-free learning](http://arxiv.org/abs/2106.04316)


  Biological agents have meaningful interactions with their environment despite
the absence of immediate reward signals. In such instances, the agent can learn
preferred modes of behaviour that lead to predictable states -- necessary for
survival. In this paper, we pursue the notion that this learnt behaviour can be
a consequence of reward-free preference learning that ensures an appropriate
trade-off between exploration and preference satisfaction. For this, we
introduce a model-based Bayesian agent equipped with a preference learning
mechanism (pepper) using conjugate priors. These conjugate priors are used to
augment the expected free energy planner for learning preferences over states
(or outcomes) across time. Importantly, our approach enables the agent to learn
preferences that encourage adaptive behaviour at test time. We illustrate this
in the OpenAI Gym FrozenLake and the 3D mini-world environments -- with and
without volatility. Given a constant environment, these agents learn confident
(i.e., precise) preferences and act to satisfy them. Conversely, in a volatile
setting, perpetual preference uncertainty maintains exploratory behaviour. Our
experiments suggest that learnable (reward-free) preferences entail a trade-off
between exploration and preference satisfaction. Pepper offers a
straightforward framework suitable for designing adaptive agents when reward
functions cannot be predefined as in real environments.

    

### [[1809.00970] Iterative multi-path tracking for video and volume segmentation with sparse point supervision](http://arxiv.org/abs/1809.00970)


  Recent machine learning strategies for segmentation tasks have shown great
ability when trained on large pixel-wise annotated image datasets. It remains a
major challenge however to aggregate such datasets, as the time and monetary
cost associated with collecting extensive annotations is extremely high. This
is particularly the case for generating precise pixel-wise annotations in video
and volumetric image data. To this end, this work presents a novel framework to
produce pixel-wise segmentations using minimal supervision. Our method relies
on 2D point supervision, whereby a single 2D location within an object of
interest is provided on each image of the data. Our method then estimates the
object appearance in a semi-supervised fashion by learning
object-image-specific features and by using these in a semi-supervised learning
framework. Our object model is then used in a graph-based optimization problem
that takes into account all provided locations and the image data in order to
infer the complete pixel-wise segmentation. In practice, we solve this
optimally as a tracking problem using a K-shortest path approach. Both the
object model and segmentation are then refined iteratively to further improve
the final segmentation. We show that by collecting 2D locations using a gaze
tracker, our approach can provide state-of-the-art segmentations on a range of
objects and image modalities (video and 3D volumes), and that these can then be
used to train supervised machine learning classifiers.

    

### [[1910.10562] Nested conformal prediction and quantile out-of-bag ensemble methods](http://arxiv.org/abs/1910.10562)


  Conformal prediction is a popular tool for providing valid prediction sets
for classification and regression problems, without relying on any
distributional assumptions on the data. While the traditional description of
conformal prediction starts with a nonconformity score, we provide an alternate
(but equivalent) view that starts with a sequence of nested sets and calibrates
them to find a valid prediction set. The nested framework subsumes all
nonconformity scores, including recent proposals based on quantile regression
and density estimation. While these ideas were originally derived based on
sample splitting, our framework seamlessly extends them to other aggregation
schemes like cross-conformal, jackknife+ and out-of-bag methods. We use the
framework to derive a new algorithm (QOOB, pronounced cube) that combines four
ideas: quantile regression, cross-conformalization, ensemble methods and
out-of-bag predictions. We develop a computationally efficient implementation
of cross-conformal, that is also used by QOOB. In a detailed numerical
investigation, QOOB performs either the best or close to the best on all
simulated and real datasets.

    

### [[2107.08132] Loop Transformations using Clang's Abstract Syntax Tree](http://arxiv.org/abs/2107.08132)


  OpenMP 5.1 introduced the first loop nest transformation directives unroll
and tile, and more are expected to be included in OpenMP 6.0. We discuss the
two Abstract Syntax Tree (AST) representations used by Clang's implementation
that is currently under development. The first representation is designed for
compatibility with the existing implementation and stores the transformed loop
nest in a shadow AST next to the syntactical AST. The second representation
introduces a new meta AST-node OMPCanonicalLoop that guarantees that the
semantic requirements of an OpenMP loop are met, and a CanonicalLoopInfo type
that the OpenMPIRBuilder uses to represent literal and transformed loops. This
second approach provides a better abstraction of loop semantics, removes the
need for shadow AST nodes that are only relevant for code generation, allows
sharing the implementation with other front-ends such as flang, but depends on
the OpenMPIRBuilder which is currently under development.

    

### [[2107.08729] Towards Probabilistic Session-Type Monitoring](http://arxiv.org/abs/2107.08729)


  We present a tool-based approach for the runtime analysis of communicating
processes grounded on probabilistic binary session types. We synthesise a
monitor out of a probabilistic session type where each choice point is
decorated by a probability distribution. The monitor observes the execution of
a process, infers its probabilistic behaviour and issues warnings when the
observed behaviour deviates from the one specified by the probabilistic session
type.

    

### [[2107.08824] Verified Mutable Data Structures](http://arxiv.org/abs/2107.08824)


  Malfunctions in software like airplane control systems or nuclear plant
control systems can have catastrophic consequences. Formal verification is the
only form of sofware testing that can guarantee the absence of bugs. Formally
verified software gives a mathematical proof that the specification is
correctly implemented and that no bugs would induce unwanted behaviour. This
has a high development cost and having an entirely verified program takes time
and effort. However, having verified components already has great benefits. We
implement in Scala and formally verify with Stainless a hash map that can then
be reused and act as a basis on which to rely. The implementation we propose is
based on the LongMap of the Scala standard library with some minor adaptations.
This map is implemented with mutable arrays. We give the specification with
respect to an implementation of a map based on a list of tuples, that we
implement and formally verify as well.

    

### [[2107.08852] Structured Proofs for Adversarial Cyber-Physical Systems](http://arxiv.org/abs/2107.08852)


  Many cyber-physical systems (CPS) are safety-critical, so it is important to
formally verify them, e.g. in formal logics that show a model's correctness
specification always holds. Constructive Differential Game Logic (CdGL) is such
a logic for (constructive) hybrid games, including hybrid systems. To overcome
undecidability, the user first writes a proof, for which we present a
proof-checking tool.
We introduce Kaisar, the first language and tool for CdGL proofs, which until
now could only be written by hand with a low-level proof calculus. Kaisar's
structured proofs simplify challenging CPS proof tasks, especially by using
programming language principles and high-level stateful reasoning. Kaisar
exploits CdGL's constructivity and refinement relations to build proofs around
models of game strategies. The evaluation reproduces and extends existing case
studies on 1D and 2D driving. Proof metrics are compared and reported
experiences are discussed for the original studies and their reproductions.

    

### [[2103.09518] Sliceable Monolith: Monolith First, Microservices Later](http://arxiv.org/abs/2103.09518)


  We propose Sliceable Monolith, a new methodology for developing microservice
architectures and perform their integration testing by leveraging most of the
simplicity of a monolith: a single codebase and a local execution environment
that simulates distribution. Then, a tool compiles a codebase for each
microservice and a cloud deployment configuration. The key enabler of our
approach is the technology-agnostic service definition language offered by
Jolie.

    