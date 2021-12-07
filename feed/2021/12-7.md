
## 2021-12-7

### [[2112.02161] MAC address randomization tolerant crowd monitoring system using Wi-Fi packets](http://arxiv.org/abs/2112.02161)


  Media access control (MAC) addresses inside Wi-Fi packets can be used for
beneficial activities such as crowdedness estimation, marketing, and hazard
maps. However, the MAC address randomization systems introduced around 2014
make all conventional MAC-address-based crowd monitoring systems count the same
device more than once. Therefore, there is a need to create a new crowd
monitoring system tolerant to MAC address randomization to estimate the number
of devices accurately. In this paper, Vision and TrueSight, two new crowd
monitoring algorithms that estimate the number of devices, are proposed to
prove that MAC-address-based crowd monitoring is still possible. In addition to
probe requests, Vision uses data packets and beacon packets to mitigate the
influence of randomization. Moreover, TrueSight uses sequence numbers and
hierarchical clustering to estimate the number of devices. The experimental
results of this study show that even without installing any special software,
Vision can gather 440 randomly generated MAC addresses into one group and count
only once, and TrueSight can estimate the number of devices with an accuracy of
more than 75% with an acceptable error range of 1.

    

### [[2112.02163] WebRTC-based measurement tool for peer-to-peer applications and preliminary findings with real users](http://arxiv.org/abs/2112.02163)


  Direct peer-to-peer (P2P) communication is often used to minimize the
end-to-end latency for real-time applications that require accurate
synchronization, such as remote musical ensembles. However, there are few
studies on the performance of P2P communication between home network
environments, thus hindering the deployment of services that require
synchronization. In this study, we developed a P2P performance measurement tool
using the Web Real-Time Communication (WebRTC) statistics application
programming interface. Using this tool, we can easily measure P2P performance
between home network environments on a web browser without downloading client
applications. We also verified the reliability of round-trip time (RTT)
measurements using WebRTC and confirmed that our system could provide the
necessary measurement accuracy for RTT and jitter measurements for real-time
applications. In addition, we measured the performance of a full mesh topology
connection with 10 users in an actual environment in Japan. Consequently, we
found that only 66% of the peer connections had a latency of 30 ms or less,
which is the minimum requirement for high synchronization applications, such as
musical ensembles.

    

### [[2112.02417] Predicting Bandwidth Utilization on Network Links Using Machine Learning](http://arxiv.org/abs/2112.02417)


  Predicting the bandwidth utilization on network links can be extremely useful
for detecting congestion in order to correct them before they occur. In this
paper, we present a solution to predict the bandwidth utilization between
different network links with a very high accuracy. A simulated network is
created to collect data related to the performance of the network links on
every interface. These data are processed and expanded with feature engineering
in order to create a training set. We evaluate and compare three types of
machine learning algorithms, namely ARIMA (AutoRegressive Integrated Moving
Average), MLP (Multi Layer Perceptron) and LSTM (Long Short-Term Memory), in
order to predict the future bandwidth consumption. The LSTM outperforms ARIMA
and MLP with very accurate predictions, rarely exceeding a 3\% error (40\% for
ARIMA and 20\% for the MLP). We then show that the proposed solution can be
used in real time with a reaction managed by a Software-Defined Networking
(SDN) platform.

    

### [[2112.02439] Deep Learning on Mobile Devices Through Neural Processing Units and Edge Computing](http://arxiv.org/abs/2112.02439)


  Deep Neural Network (DNN) is becoming adopted for video analytics on mobile
devices. To reduce the delay of running DNNs, many mobile devices are equipped
with Neural Processing Units (NPU). However, due to the resource limitations of
NPU, these DNNs have to be compressed to increase the processing speed at the
cost of accuracy. To address the low accuracy problem, we propose a Confidence
Based Offloading (CBO) framework for deep learning video analytics. The major
challenge is to determine when to return the NPU classification result based on
the confidence level of running the DNN, and when to offload the video frames
to the server for further processing to increase the accuracy. We first
identify the problem of using existing confidence scores to make offloading
decisions, and propose confidence score calibration techniques to improve the
performance. Then, we formulate the CBO problem where the goal is to maximize
accuracy under some time constraint, and propose an adaptive solution that
determines which frames to offload at what resolution based on the confidence
score and the network condition. Through real implementations and extensive
evaluations, we demonstrate that the proposed solution can significantly
outperform other approaches.

    

### [[2112.02461] Multilateral Micro-Monitoring for Internet Streaming](http://arxiv.org/abs/2112.02461)


  Video streaming is dominating the Internet. To compete with the performance
of traditional cable and satellite options, content providers outsource the
content delivery to third-party content distribution networks and brokers.
However, no existing monitoring mechanism offers a multilateral view of a
streaming service's performance. In other words, no auditing mechanism reflects
the mutual agreement of content providers, content distributors and end-users
alike about how well, or not, a service performs.
In this paper, we present UgoVor, a system for monitoring multilateral
streaming contracts, that is enforceable descriptions of mutual agreements
among content providers, content distributors and end-users. Our key insight is
that real-time multilateral micro-monitoring -- capable of accounting for every
re-buffering event and the resolution of every video chunk in a stream -- is
not only feasible, but an Internet-scalable task. To demonstrate this claim we
evaluate UgoVor in the context of a 10-month long experiment, corresponding to
over 25 years of streaming data, including over 430,000 streaming sessions with
clients from over 1,300 unique ASes. Our measurements confirm that UgoVor can
provide an accurate distributed performance consensus for Internet streaming,
and can help radically advance existing performance-agnostic pricing model
towards novel and transparent pay-what-you-experience ones.

    

### [[2112.02476] Provisioning Fog Services to 3GPP Subscribers: Authentication and Application Mobility](http://arxiv.org/abs/2112.02476)


  Multi-Access Edge computing (MEC) and Fog computing provide services to
subscribers at low latency. There is a need to form a federation among 3GPP MEC
and fog to provide better coverage to 3GPP subscribers. This federation gives
rise to two issues - third-party authentication and application mobility - for
continuous service during handover from 3GPP MEC to fog without
re-authentication. In this paper, we propose: 1) a proxy-based state transfer
and third-party authentication (PS3A) that uses a transparent proxy to transfer
the authentication and application state information, and 2) a token-based
state transfer and proxy-based third-party authentication (TSP3A) that uses the
proxy to transfer the authentication information and tokens to transfer the
application state from 3GPP MEC to the fog. The proxy is kept transparent with
virtual counterparts, to avoid any changes to the existing 3GPP MEC and fog
architectures. We implemented these solutions on a testbed and results show
that PS3A and TSP3A provide authentication within 0.345-2.858s for a 0-100 Mbps
proxy load. The results further show that TSP3A provides application mobility
while taking 40-52% less time than PS3A using state tokens. TSP3A and PS3A also
reduce the service interruption latency by 82.4% and 84.6%, compared to the
cloud-based service via tokens and prefetching.

    

### [[2112.02637] Modeling Live Video Streaming: Real-Time Classification, QoE Inference, and Field Evaluation](http://arxiv.org/abs/2112.02637)


  Social media, professional sports, and video games are driving rapid growth
in live video streaming, on platforms such as Twitch and YouTube Live. Live
streaming experience is very susceptible to short-time-scale network congestion
since client playback buffers are often no more than a few seconds.
Unfortunately, identifying such streams and measuring their QoE for network
management is challenging, since content providers largely use the same
delivery infrastructure for live and video-on-demand (VoD) streaming, and
packet inspection techniques (including SNI/DNS query monitoring) cannot always
distinguish between the two.
In this paper, we design, build, and deploy ReCLive: a machine learning
method for live video detection and QoE measurement based on network-level
behavioral characteristics. Our contributions are four-fold: (1) We analyze
about 23,000 video streams from Twitch and YouTube, and identify key features
in their traffic profile that differentiate live and on-demand streaming. We
release our traffic traces as open data to the public; (2) We develop an
LSTM-based binary classifier model that distinguishes live from on-demand
streams in real-time with over 95% accuracy across providers; (3) We develop a
method that estimates QoE metrics of live streaming flows in terms of
resolution and buffer stall events with overall accuracies of 93% and 90%,
respectively; and (4) Finally, we prototype our solution, train it in the lab,
and deploy it in a live ISP network serving more than 7,000 subscribers. Our
method provides ISPs with fine-grained visibility into live video streams,
enabling them to measure and improve user experience.

    

### [[2112.02692] Faster Content Delivery using RSU Caching and Vehicular Pre-caching in Vehicular Networks](http://arxiv.org/abs/2112.02692)


  Most non-safety applications deployed in Vehicular Ad-hoc Network (VANET) use
vehicle-to-infrastructure (V2I) and I2V communications to receive various forms
of content such as periodic traffic updates, advertisements from adjacent
road-side units (RSUs). In case of heavy traffic on highways and urban areas,
content delivery time (CDT) can be significantly affected. Increase in CDT can
be attributed to high load on the RSU or high volume of broadcasted content
which can flood the network. Therefore, this paper suggests a novel caching
strategy to improve CDT in high traffic areas and three major contributions
have been made: (1) Design and simulation of a caching strategy to decrease the
average content delivery time; (2) Evaluation and comparison of caching
performance in both urban scenario and highway scenario; (3) Evaluation and
comparison of caching performance in single RSU and multiple RSUs. The
simulation results show that caching effectively reduces the CDT by 50% in
urban scenario and 60-70% in highway scenario.

    

### [[2112.02775] Sensor as a Company: On Self-Sustaining IoT Commons](http://arxiv.org/abs/2112.02775)


  Beyond the "smart home" and "smart enterprise", the Internet of Things (IoT)
revolution is creating "smart communities", where shared IoT devices
collectively benefit a large number of residents, for transportation,
healthcare, safety, and more. However, large-scale deployments of IoT-powered
neighborhoods face two key socio-technical challenges: the significant upfront
investment and the lack of information on local IoT needs. In this paper, we
present SensorInc, a new IoT deployment paradigm that incentivizes residents to
design and manage sensor deployment through sensor liquefaction. By turning
shared sensors into liquid (i.e. tradeable) assets akin to company stock or
bond, users can design and invest in promising IoT deployments and receive
monetary rewards afterward. We present the detailed design of SensorInc and
conduct two case studies (parking occupancy sensors and air pollution sensors)
to study the self-sustainability and deployment challenges of such a paradigm.

    

### [[1907.13612] MSNM-Sensor: An Applied Network Monitoring Tool for Anomaly Detection in Complex Networks and Systems](http://arxiv.org/abs/1907.13612)


  Technology evolves quickly. Low-cost and ready-to-connect devices are
designed to provide new services and applications. Smart grids or smart
healthcare systems are some examples of these applications, all of which are in
the context of smart cities. In this total-connectivity scenario, some security
issues arise since the larger the number of connected devices is, the greater
the surface attack dimension. In this way, new solutions for monitoring and
detecting security events are needed to address new challenges brought about by
this scenario, among others, the large number of devices to monitor, the large
amount of data to manage and the real-time requirement to provide quick
security event detection and, consequently, quick response to attacks. In this
work, a practical and ready-to-use tool for monitoring and detecting security
events in these environments is developed and introduced. The tool is based on
the Multivariate Statistical Network Monitoring (MSNM) methodology for
monitoring and anomaly detection and we call it MSNM-Sensor. Although it is in
its early development stages, experimental results based on the detection of
well-known attacks in hierarchical network systems prove the suitability of
this tool for more complex scenarios, such as those found in smart cities or
IoT ecosystems.

    

### [[2004.04346] Massive MIMO Channels with Inter-User Angle Correlation: Open-Access Dataset, Analysis and Measurement-Based Validation](http://arxiv.org/abs/2004.04346)


  In practical propagation environments, different massive MIMO users can have
correlated angles in spatial paths. In this paper, we study the effect of angle
correlation on inter-user channel correlation via a combination of measurement
and analysis. We show three key results. First, we collect a massive MIMO
channel dataset for examining the inter-user channel correlation in a
real-world propagation environment; the dataset is now open-access. We observed
channel correlation higher than $0.48$ for all close-by users. Additionally,
over $30$ % of far-away users, even when they are tens of wavelengths apart,
have inter-user channel correlation that is at least twice higher than the
correlation in the i.i.d. Rayleigh fading channel. Second, we compute the
inter-user channel correlation in closed-form as a function of inter-user angle
correlation, the number of base-station antennas, and base-station
inter-antenna spacing. Our analysis shows that inter-user angle correlation
increases the inter-user channel correlation. Inter-user channel correlation
reduces with a larger base-station array aperture, i.e., more antennas and
larger inter-antenna spacing. Third, we explain the measurements with numerical
experiments to show that inter-user angle correlation can result in significant
inter-user channel correlation in practical massive MIMO channels.

    

### [[2012.03719] Dimmer: Self-Adaptive Network-Wide Flooding with Reinforcement Learning](http://arxiv.org/abs/2012.03719)


  The last decade saw an emergence of Synchronous Transmissions (ST) as an
effective communication paradigm in low-power wireless networks. Numerous ST
protocols provide high reliability and energy efficiency in normal wireless
conditions, for a large variety of traffic requirements. Recently, with the
EWSN dependability competitions, the community pushed ST to harsher and
highly-interfered environments, improving upon classical ST protocols through
the use of custom rules, hand-tailored parameters, and additional
retransmissions. The results are sophisticated protocols, that require prior
expert knowledge and extensive testing, often tuned for a specific deployment
and envisioned scenario. In this paper, we explore how ST protocols can benefit
from self-adaptivity; a self-adaptive ST protocol selects itself its best
parameters to (1) tackle external environment dynamics and (2) adapt to its
topology over time. We introduce Dimmer as a self-adaptive ST protocol. Dimmer
builds on LWB and uses Reinforcement Learning to tune its parameters and match
the current properties of the wireless medium. By learning how to behave from
an unlabeled dataset, Dimmer adapts to different interference types and
patterns, and is able to tackle previously unseen interference. With Dimmer, we
explore how to efficiently design AI-based systems for constrained devices, and
outline the benefits and downfalls of AI-based low-power networking. We
evaluate our protocol on two deployments of resource-constrained nodes
achieving 95.8% reliability against strong, unknown WiFi interference. Our
results outperform baselines such as non-adaptive ST protocols (27%) and PID
controllers, and show a performance close to hand-crafted and more
sophisticated solutions, such as Crystal (99%).

    

### [[2105.11620] A Comparative Synthesis Approach to Optimal Network Designs with Indeterminate Objectives](http://arxiv.org/abs/2105.11620)


  When managing wide-area networks, network architects must decide how to
balance multiple conflicting metrics, and ensure fair allocations to competing
traffic while prioritizing critical traffic. The state of practice poses
challenges since architects must precisely encode their (somewhat fuzzy) intent
into formal optimization models using abstract notions such as utility
functions, and ad-hoc manually tuned knobs. In this paper, we present the first
effort to synthesize network designs with indeterminate objectives using an
interactive program-synthesis-based approach. We make three contributions.
First, we present a novel framework in which a user's design objective, and the
synthesis of a program (network design) that optimizes that objective are done
in tandem. Second, we develop a novel algorithm for our framework in which a
voting-guided learner makes two kinds of queries (Propose and Compare) to the
user, with the aim of minimizing the number of queries. We present theoretical
analysis of the convergence rate of the algorithm. Third, we implemented
Net10Q, a system based on our approach, and demonstrate its effectiveness on
four real-world network case studies using black-box oracles and simulation
experiments, as well as a pilot user study comprising network researchers and
practitioners. Both theoretical and experimental results show the promise of
our approach.

    

### [[2112.02095] Intelligent Trading Systems: A Sentiment-Aware Reinforcement Learning Approach](http://arxiv.org/abs/2112.02095)


  The feasibility of making profitable trades on a single asset on stock
exchanges based on patterns identification has long attracted researchers.
Reinforcement Learning (RL) and Natural Language Processing have gained
notoriety in these single-asset trading tasks, but only a few works have
explored their combination. Moreover, some issues are still not addressed, such
as extracting market sentiment momentum through the explicit capture of
sentiment features that reflect the market condition over time and assessing
the consistency and stability of RL results in different situations. Filling
this gap, we propose the Sentiment-Aware RL (SentARL) intelligent trading
system that improves profit stability by leveraging market mood through an
adaptive amount of past sentiment features drawn from textual news. We
evaluated SentARL across twenty assets, two transaction costs, and five
different periods and initializations to show its consistent effectiveness
against baselines. Subsequently, this thorough assessment allowed us to
identify the boundary between news coverage and market sentiment regarding the
correlation of price-time series above which SentARL's effectiveness is
outstanding.

    

### [[2112.02097] Global alignment for relation extraction in Microbiology](http://arxiv.org/abs/2112.02097)


  We investigate a method to extract relations from texts based on global
alignment and syntactic information. Combined with SVM, this method is shown to
have a performance comparable or even better than LSTM on two RE tasks.

    

### [[2112.02100] ProbNum: Probabilistic Numerics in Python](http://arxiv.org/abs/2112.02100)


  Probabilistic numerical methods (PNMs) solve numerical problems via
probabilistic inference. They have been developed for linear algebra,
optimization, integration and differential equation simulation. PNMs naturally
incorporate prior information about a problem and quantify uncertainty due to
finite computational resources as well as stochastic input. In this paper, we
present ProbNum: a Python library providing state-of-the-art probabilistic
numerical solvers. ProbNum enables custom composition of PNMs for specific
problem classes via a modular design as well as wrappers for off-the-shelf use.
Tutorials, documentation, developer guides and benchmarks are available online
at this http URL.

    

### [[2112.02102] Echocardiography Segmentation with Enforced Temporal Consistency](http://arxiv.org/abs/2112.02102)


  Convolutional neural networks (CNN) have demonstrated their ability to
segment 2D cardiac ultrasound images. However, despite recent successes
according to which the intra-observer variability on end-diastole and
end-systole images has been reached, CNNs still struggle to leverage temporal
information to provide accurate and temporally consistent segmentation maps
across the whole cycle. Such consistency is required to accurately describe the
cardiac function, a necessary step in diagnosing many cardiovascular diseases.
In this paper, we propose a framework to learn the 2D+time long-axis cardiac
shape such that the segmented sequences can benefit from temporal and
anatomical consistency constraints. Our method is a post-processing that takes
as input segmented echocardiographic sequences produced by any state-of-the-art
method and processes it in two steps to (i) identify spatio-temporal
inconsistencies according to the overall dynamics of the cardiac sequence and
(ii) correct the inconsistencies. The identification and correction of cardiac
inconsistencies relies on a constrained autoencoder trained to learn a
physiologically interpretable embedding of cardiac shapes, where we can both
detect and fix anomalies. We tested our framework on 98 full-cycle sequences
from the CAMUS dataset, which will be rendered public alongside this paper. Our
temporal regularization method not only improves the accuracy of the
segmentation across the whole sequences, but also enforces temporal and
anatomical consistency.

    

### [[2112.02120] Prediction and compression of lattice QCD data using machine learning algorithms on quantum annealer](http://arxiv.org/abs/2112.02120)


  We present regression and compression algorithms for lattice QCD data
utilizing the efficient binary optimization ability of quantum annealers. In
the regression algorithm, we encode the correlation between the input and
output variables into a sparse coding machine learning algorithm. The trained
correlation pattern is used to predict lattice QCD observables of unseen
lattice configurations from other observables measured on the lattice. In the
compression algorithm, we define a mapping from lattice QCD data of
floating-point numbers to the binary coefficients that closely reconstruct the
input data from a set of basis vectors. Since the reconstruction is not exact,
the mapping defines a lossy compression, but, a reasonably small number of
binary coefficients are able to reconstruct the input vector of lattice QCD
data with the reconstruction error much smaller than the statistical
fluctuation. In both applications, we use D-Wave quantum annealers to solve the
NP-hard binary optimization problems of the machine learning algorithms.

    

### [[2112.02129] Distributed Adaptive Learning Under Communication Constraints](http://arxiv.org/abs/2112.02129)


  This work examines adaptive distributed learning strategies designed to
operate under communication constraints. We consider a network of agents that
must solve an online optimization problem from continual observation of
streaming data. The agents implement a distributed cooperative strategy where
each agent is allowed to perform local exchange of information with its
neighbors. In order to cope with communication constraints, the exchanged
information must be unavoidably compressed. We propose a diffusion strategy
nicknamed as ACTC (Adapt-Compress-Then-Combine), which relies on the following
steps: i) an adaptation step where each agent performs an individual
stochastic-gradient update with constant step-size; ii) a compression step that
leverages a recently introduced class of stochastic compression operators; and
iii) a combination step where each agent combines the compressed updates
received from its neighbors. The distinguishing elements of this work are as
follows. First, we focus on adaptive strategies, where constant (as opposed to
diminishing) step-sizes are critical to respond in real time to nonstationary
variations. Second, we consider the general class of directed graphs and
left-stochastic combination policies, which allow us to enhance the interplay
between topology and learning. Third, in contrast with related works that
assume strong convexity for all individual agents' cost functions, we require
strong convexity only at a network level, a condition satisfied even if a
single agent has a strongly-convex cost and the remaining agents have
non-convex costs. Fourth, we focus on a diffusion (as opposed to consensus)
strategy. Under the demanding setting of compressed information, we establish
that the ACTC iterates fluctuate around the desired optimizer, achieving
remarkable savings in terms of bits exchanged between neighboring agents.

    

### [[2112.02139] Face Reconstruction with Variational Autoencoder and Face Masks](http://arxiv.org/abs/2112.02139)


  Variational AutoEncoders (VAE) employ deep learning models to learn a
continuous latent z-space that is subjacent to a high-dimensional observed
dataset. With that, many tasks are made possible, including face reconstruction
and face synthesis. In this work, we investigated how face masks can help the
training of VAEs for face reconstruction, by restricting the learning to the
pixels selected by the face mask. An evaluation of the proposal using the
celebA dataset shows that the reconstructed images are enhanced with the face
masks, especially when SSIM loss is used either with l1 or l2 loss functions.
We noticed that the inclusion of a decoder for face mask prediction in the
architecture affected the performance for l1 or l2 loss functions, while this
was not the case for the SSIM loss. Besides, SSIM perceptual loss yielded the
crispest samples between all hypotheses tested, although it shifts the original
color of the image, making the usage of the l1 or l2 losses together with SSIM
helpful to solve this issue.

    

### [[2112.02140] Combining Embeddings and Fuzzy Time Series for High-Dimensional Time Series Forecasting in Internet of Energy Applications](http://arxiv.org/abs/2112.02140)


  The prediction of residential power usage is essential in assisting a smart
grid to manage and preserve energy to ensure efficient use. An accurate energy
forecasting at the customer level will reflect directly into efficiency
improvements across the power grid system, however forecasting building energy
use is a complex task due to many influencing factors, such as meteorological
and occupancy patterns. In addiction, high-dimensional time series increasingly
arise in the Internet of Energy (IoE), given the emergence of multi-sensor
environments and the two way communication between energy consumers and the
smart grid. Therefore, methods that are capable of computing high-dimensional
time series are of great value in smart building and IoE applications. Fuzzy
Time Series (FTS) models stand out as data-driven non-parametric models of easy
implementation and high accuracy. Unfortunately, the existing FTS models can be
unfeasible if all features were used to train the model. We present a new
methodology for handling high-dimensional time series, by projecting the
original high-dimensional data into a low dimensional embedding space and using
multivariate FTS approach in this low dimensional representation. Combining
these techniques enables a better representation of the complex content of
multivariate time series and more accurate forecasts.

    

### [[2112.02165] On Submodular Contextual Bandits](http://arxiv.org/abs/2112.02165)


  We consider the problem of contextual bandits where actions are subsets of a
ground set and mean rewards are modeled by an unknown monotone submodular
function that belongs to a class $\mathcal{F}$. We allow time-varying matroid
constraints to be placed on the feasible sets. Assuming access to an online
regression oracle with regret $\mathsf{Reg}(\mathcal{F})$, our algorithm
efficiently randomizes around local optima of estimated functions according to
the Inverse Gap Weighting strategy. We show that cumulative regret of this
procedure with time horizon $n$ scales as $O(\sqrt{n
\mathsf{Reg}(\mathcal{F})})$ against a benchmark with a multiplicative factor
$1/2$. On the other hand, using the techniques of (Filmus and Ward 2014), we
show that an $\epsilon$-Greedy procedure with local randomization attains
regret of $O(n^{2/3} \mathsf{Reg}(\mathcal{F})^{1/3})$ against a stronger
$(1-e^{-1})$ benchmark.

    

### [[2112.02170] Counterfactual Fairness in Mortgage Lending via Matching and Randomization](http://arxiv.org/abs/2112.02170)


  Unfairness in mortgage lending has created generational inequality among
racial and ethnic groups in the US. Many studies address this problem, but most
existing work focuses on correlation-based techniques. In our work, we use the
framework of counterfactual fairness to train fair machine learning models. We
propose a new causal graph for the variables available in the Home Mortgage
Disclosure Act (HMDA) data. We use a matching-based approach instead of the
latent variable modeling approach, because the former approach does not rely on
any modeling assumptions. Furthermore, matching provides us with counterfactual
pairs in which the race variable is isolated. We first demonstrate the
unfairness in mortgage approval and interest rates between African-American and
non-Hispanic White sub-populations. Then, we show that having balanced data
using matching does not guarantee perfect counterfactual fairness of the
machine learning models.

    

### [[2112.02185] Neural Pseudo-Label Optimism for the Bank Loan Problem](http://arxiv.org/abs/2112.02185)


  We study a class of classification problems best exemplified by the
\emph{bank loan} problem, where a lender decides whether or not to issue a
loan. The lender only observes whether a customer will repay a loan if the loan
is issued to begin with, and thus modeled decisions affect what data is
available to the lender for future decisions. As a result, it is possible for
the lender's algorithm to ``get stuck'' with a self-fulfilling model. This
model never corrects its false negatives, since it never sees the true label
for rejected data, thus accumulating infinite regret. In the case of linear
models, this issue can be addressed by adding optimism directly into the model
predictions. However, there are few methods that extend to the function
approximation case using Deep Neural Networks. We present Pseudo-Label Optimism
(PLOT), a conceptually and computationally simple method for this setting
applicable to DNNs. \PLOT{} adds an optimistic label to the subset of decision
points the current model is deciding on, trains the model on all data so far
(including these points along with their optimistic labels), and finally uses
the resulting \emph{optimistic} model for decision making. \PLOT{} achieves
competitive performance on a set of three challenging benchmark problems,
requiring minimal hyperparameter tuning. We also show that \PLOT{} satisfies a
logarithmic regret guarantee, under a Lipschitz and logistic mean label model,
and under a separability condition on the data.

    

### [[2112.02191] NN-LUT: Neural Approximation of Non-Linear Operations for Efficient Transformer Inference](http://arxiv.org/abs/2112.02191)


  Non-linear operations such as GELU, Layer normalization, and Softmax are
essential yet costly building blocks of Transformer models. Several prior works
simplified these operations with look-up tables or integer computations, but
such approximations suffer inferior accuracy or considerable hardware cost with
long latency. This paper proposes an accurate and hardware-friendly
approximation framework for efficient Transformer inference. Our framework
employs a simple neural network as a universal approximator with its structure
equivalently transformed into a LUT. The proposed framework called NN-LUT can
accurately replace all the non-linear operations in popular BERT models with
significant reductions in area, power consumption, and latency.

    

### [[2112.02194] ALX: Large Scale Matrix Factorization on TPUs](http://arxiv.org/abs/2112.02194)


  We present ALX, an open-source library for distributed matrix factorization
using Alternating Least Squares, written in JAX. Our design allows for
efficient use of the TPU architecture and scales well to matrix factorization
problems of O(B) rows/columns by scaling the number of available TPU cores. In
order to spur future research on large scale matrix factorization methods and
to illustrate the scalability properties of our own implementation, we also
built a real world web link prediction dataset called WebGraph. This dataset
can be easily modeled as a matrix factorization problem. We created several
variants of this dataset based on locality and sparsity properties of
sub-graphs. The largest variant of WebGraph has around 365M nodes and training
a single epoch finishes in about 20 minutes with 256 TPU cores. We include
speed and performance numbers of ALX on all variants of WebGraph. Both the
framework code and the dataset is open-sourced.

    

### [[2112.02195] Learning to Search in Local Branching](http://arxiv.org/abs/2112.02195)


  Finding high-quality solutions to mixed-integer linear programming problems
(MILPs) is of great importance for many practical applications. In this
respect, the refinement heuristic local branching (LB) has been proposed to
produce improving solutions and has been highly influential for the development
of local search methods in MILP. The algorithm iteratively explores a sequence
of solution neighborhoods defined by the so-called local branching constraint,
namely, a linear inequality limiting the distance from a reference solution.
For a LB algorithm, the choice of the neighborhood size is critical to
performance. Although it was initialized by a conservative value in the
original LB scheme, our new observation is that the best size is strongly
dependent on the particular MILP instance. In this work, we investigate the
relation between the size of the search neighborhood and the behavior of the
underlying LB algorithm, and we devise a leaning based framework for guiding
the neighborhood search of the LB heuristic. The framework consists of a
two-phase strategy. For the first phase, a scaled regression model is trained
to predict the size of the LB neighborhood at the first iteration through a
regression task. In the second phase, we leverage reinforcement learning and
devise a reinforced neighborhood search strategy to dynamically adapt the size
at the subsequent iterations. We computationally show that the neighborhood
size can indeed be learned, leading to improved performances and that the
overall algorithm generalizes well both with respect to the instance size and,
remarkably, across instances.

    

### [[2112.02205] Behind the Curtain: Learning Occluded Shapes for 3D Object Detection](http://arxiv.org/abs/2112.02205)


  Advances in LiDAR sensors provide rich 3D data that supports 3D scene
understanding. However, due to occlusion and signal miss, LiDAR point clouds
are in practice 2.5D as they cover only partial underlying shapes, which poses
a fundamental challenge to 3D perception. To tackle the challenge, we present a
novel LiDAR-based 3D object detection model, dubbed Behind the Curtain Detector
(BtcDet), which learns the object shape priors and estimates the complete
object shapes that are partially occluded (curtained) in point clouds. BtcDet
first identifies the regions that are affected by occlusion and signal miss. In
these regions, our model predicts the probability of occupancy that indicates
if a region contains object shapes. Integrated with this probability map,
BtcDet can generate high-quality 3D proposals. Finally, the probability of
occupancy is also integrated into a proposal refinement module to generate the
final bounding boxes. Extensive experiments on the KITTI Dataset and the Waymo
Open Dataset demonstrate the effectiveness of BtcDet. Particularly, for the 3D
detection of both cars and cyclists on the KITTI benchmark, BtcDet surpasses
all of the published state-of-the-art methods by remarkable margins. Code is
released
(this https URL}{this https URL).

    

### [[2112.02206] Data Fusion with Latent Map Gaussian Processes](http://arxiv.org/abs/2112.02206)


  Multi-fidelity modeling and calibration are data fusion tasks that
ubiquitously arise in engineering design. In this paper, we introduce a novel
approach based on latent-map Gaussian processes (LMGPs) that enables efficient
and accurate data fusion. In our approach, we convert data fusion into a latent
space learning problem where the relations among different data sources are
automatically learned. This conversion endows our approach with attractive
advantages such as increased accuracy, reduced costs, flexibility to jointly
fuse any number of data sources, and ability to visualize correlations between
data sources. This visualization allows the user to detect model form errors or
determine the optimum strategy for high-fidelity emulation by fitting LMGP only
to the subset of the data sources that are well-correlated. We also develop a
new kernel function that enables LMGPs to not only build a probabilistic
multi-fidelity surrogate but also estimate calibration parameters with high
accuracy and consistency. The implementation and use of our approach are
considerably simpler and less prone to numerical issues compared to existing
technologies. We demonstrate the benefits of LMGP-based data fusion by
comparing its performance against competing methods on a wide range of
examples.

    

### [[2112.02209] Generalized Likelihood Ratio Test for Adversarially Robust Hypothesis Testing](http://arxiv.org/abs/2112.02209)


  Machine learning models are known to be susceptible to adversarial attacks
which can cause misclassification by introducing small but well designed
perturbations. In this paper, we consider a classical hypothesis testing
problem in order to develop fundamental insight into defending against such
adversarial perturbations. We interpret an adversarial perturbation as a
nuisance parameter, and propose a defense based on applying the generalized
likelihood ratio test (GLRT) to the resulting composite hypothesis testing
problem, jointly estimating the class of interest and the adversarial
perturbation. While the GLRT approach is applicable to general multi-class
hypothesis testing, we first evaluate it for binary hypothesis testing in white
Gaussian noise under $\ell_{\infty}$ norm-bounded adversarial perturbations,
for which a known minimax defense optimizing for the worst-case attack provides
a benchmark. We derive the worst-case attack for the GLRT defense, and show
that its asymptotic performance (as the dimension of the data increases)
approaches that of the minimax defense. For non-asymptotic regimes, we show via
simulations that the GLRT defense is competitive with the minimax approach
under the worst-case attack, while yielding a better robustness-accuracy
tradeoff under weaker attacks. We also illustrate the GLRT approach for a
multi-class hypothesis testing problem, for which a minimax strategy is not
known, evaluating its performance under both noise-agnostic and noise-aware
adversarial settings, by providing a method to find optimal noise-aware
attacks, and heuristics to find noise-agnostic attacks that are close to
optimal in the high SNR regime.

    

### [[2112.02215] Math Programming based Reinforcement Learning for Multi-Echelon Inventory Management](http://arxiv.org/abs/2112.02215)


  Reinforcement learning has lead to considerable break-throughs in diverse
areas such as robotics, games and many others. But the application to RL in
complex real-world decision making problems remains limited. Many problems in
operations management (inventory and revenue management, for example) are
characterized by large action spaces and stochastic system dynamics. These
characteristics make the problem considerably harder to solve for existing RL
methods that rely on enumeration techniques to solve per step action problems.
To resolve these issues, we develop Programmable Actor Reinforcement Learning
(PARL), a policy iteration method that uses techniques from integer programming
and sample average approximation. Analytically, we show that the for a given
critic, the learned policy in each iteration converges to the optimal policy as
the underlying samples of the uncertainty go to infinity. Practically, we show
that a properly selected discretization of the underlying uncertain
distribution can yield near optimal actor policy even with very few samples
from the underlying uncertainty. We then apply our algorithm to real-world
inventory management problems with complex supply chain structures and show
that PARL outperforms state-of-the-art RL and inventory optimization methods in
these settings. We find that PARL outperforms commonly used base stock
heuristic by 44.7% and the best performing RL method by up to 12.1% on average
across different supply chain environments.

    

### [[2112.02226] PhishMatch: A Layered Approach for Effective Detection of Phishing URLs](http://arxiv.org/abs/2112.02226)


  Phishing attacks continue to be a significant threat on the Internet. Prior
studies show that it is possible to determine whether a website is phishing or
not just by analyzing its URL more carefully. A major advantage of the URL
based approach is that it can identify a phishing website even before the web
page is rendered in the browser, thus avoiding other potential problems such as
cryptojacking and drive-by downloads. However, traditional URL based approaches
have their limitations. Blacklist based approaches are prone to zero-hour
phishing attacks, advanced machine learning based approaches consume high
resources, and other approaches send the URL to a remote server which
compromises user's privacy. In this paper, we present a layered anti-phishing
defense, PhishMatch, which is robust, accurate, inexpensive, and client-side.
We design a space-time efficient Aho-Corasick algorithm for exact string
matching and n-gram based indexing technique for approximate string matching to
detect various cybersquatting techniques in the phishing URL. To reduce false
positives, we use a global whitelist and personalized user whitelists. We also
determine the context in which the URL is visited and use that information to
classify the input URL more accurately. The last component of PhishMatch
involves a machine learning model and controlled search engine queries to
classify the URL. A prototype plugin of PhishMatch, developed for the Chrome
browser, was found to be fast and lightweight. Our evaluation shows that
PhishMatch is both efficient and effective.

    

### [[2112.02230] SHAPr: An Efficient and Versatile Membership Privacy Risk Metric for Machine Learning](http://arxiv.org/abs/2112.02230)


  Data used to train machine learning (ML) models can be sensitive. Membership
inference attacks (MIAs), attempting to determine whether a particular data
record was used to train an ML model, risk violating membership privacy. ML
model builders need a principled definition of a metric that enables them to
quantify the privacy risk of (a) individual training data records, (b)
independently of specific MIAs, (c) efficiently. None of the prior work on
membership privacy risk metrics simultaneously meets all of these criteria.
We propose such a metric, SHAPr, which uses Shapley values to quantify a
model's memorization of an individual training data record by measuring its
influence on the model's utility. This memorization is a measure of the
likelihood of a successful MIA.
Using ten benchmark datasets, we show that SHAPr is effective (precision:
0.94$\pm 0.06$, recall: 0.88$\pm 0.06$) in estimating susceptibility of a
training data record for MIAs, and is efficient (computable within minutes for
smaller datasets and in ~90 minutes for the largest dataset).
SHAPr is also versatile in that it can be used for other purposes like
assessing fairness or assigning valuation for subsets of a dataset. For
example, we show that SHAPr correctly captures the disproportionate
vulnerability of different subgroups to MIAs.
Using SHAPr, we show that the membership privacy risk of a dataset is not
necessarily improved by removing high risk training data records, thereby
confirming an observation from prior work in a significantly extended setting
(in ten datasets, removing up to 50% of data).

    

### [[2112.02250] Dense Extreme Inception Network for Edge Detection](http://arxiv.org/abs/2112.02250)


  Edge detection is the basis of many computer vision applications. State of
the art predominantly relies on deep learning with two decisive factors:
dataset content and network's architecture. Most of the publicly available
datasets are not curated for edge detection tasks. Here, we offer a solution to
this constraint. First, we argue that edges, contours and boundaries, despite
their overlaps, are three distinct visual features requiring separate benchmark
datasets. To this end, we present a new dataset of edges. Second, we propose a
novel architecture, termed Dense Extreme Inception Network for Edge Detection
(DexiNed), that can be trained from scratch without any pre-trained weights.
DexiNed outperforms other algorithms in the presented dataset. It also
generalizes well to other datasets without any fine-tuning. The higher quality
of DexiNed is also perceptually evident thanks to the sharper and finer edges
it outputs.

    

### [[2112.02256] Towards the One Learning Algorithm Hypothesis: A System-theoretic Approach](http://arxiv.org/abs/2112.02256)


  The existence of a universal learning architecture in human cognition is a
widely spread conjecture supported by experimental findings from neuroscience.
While no low-level implementation can be specified yet, an abstract outline of
human perception and learning is believed to entail three basic properties: (a)
hierarchical attention and processing, (b) memory-based knowledge
representation, and (c) progressive learning and knowledge compaction. We
approach the design of such a learning architecture from a system-theoretic
viewpoint, developing a closed-loop system with three main components: (i) a
multi-resolution analysis pre-processor, (ii) a group-invariant feature
extractor, and (iii) a progressive knowledge-based learning module.
Multi-resolution feedback loops are used for learning, i.e., for adapting the
system parameters to online observations. To design (i) and (ii), we build upon
the established theory of wavelet-based multi-resolution analysis and the
properties of group convolution operators. Regarding (iii), we introduce a
novel learning algorithm that constructs progressively growing knowledge
representations in multiple resolutions. The proposed algorithm is an extension
of the Online Deterministic Annealing (ODA) algorithm based on annealing
optimization, solved using gradient-free stochastic approximation. ODA has
inherent robustness and regularization properties and provides a means to
progressively increase the complexity of the learning model i.e. the number of
the neurons, as needed, through an intuitive bifurcation phenomenon. The
proposed multi-resolution approach is hierarchical, progressive,
knowledge-based, and interpretable. We illustrate the properties of the
proposed architecture in the context of the state-of-the-art learning
algorithms and deep learning methods.

    

### [[2112.02262] STJLA: A Multi-Context Aware Spatio-Temporal Joint Linear Attention Network for Traffic Forecasting](http://arxiv.org/abs/2112.02262)


  Traffic prediction has gradually attracted the attention of researchers
because of the increase in traffic big data. Therefore, how to mine the complex
spatio-temporal correlations in traffic data to predict traffic conditions more
accurately become a difficult problem. Previous works combined graph
convolution networks (GCNs) and self-attention mechanism with deep time series
models (e.g. recurrent neural networks) to capture the spatio-temporal
correlations separately, ignoring the relationships across time and space.
Besides, GCNs are limited by over-smoothing issue and self-attention is limited
by quadratic problem, result in GCNs lack global representation capabilities,
and self-attention inefficiently capture the global spatial dependence. In this
paper, we propose a novel deep learning model for traffic forecasting, named
Multi-Context Aware Spatio-Temporal Joint Linear Attention (STJLA), which
applies linear attention to the spatio-temporal joint graph to capture global
dependence between all spatio-temporal nodes efficiently. More specifically,
STJLA utilizes static structural context and dynamic semantic context to
improve model performance. The static structure context based on node2vec and
one-hot encoding enriches the spatio-temporal position information.
Furthermore, the multi-head diffusion convolution network based dynamic spatial
context enhances the local spatial perception ability, and the GRU based
dynamic temporal context stabilizes sequence position information of the linear
attention, respectively. Experiments on two real-world traffic datasets,
England and PEMSD7, demonstrate that our STJLA can achieve up to 9.83% and
3.08% accuracy improvement in MAE measure over state-of-the-art baselines.

    

### [[2112.02264] DMGCRN: Dynamic Multi-Graph Convolution Recurrent Network for Traffic Forecasting](http://arxiv.org/abs/2112.02264)


  Traffic forecasting is a problem of intelligent transportation systems (ITS)
and crucial for individuals and public agencies. Therefore, researches pay
great attention to deal with the complex spatio-temporal dependencies of
traffic system for accurate forecasting. However, there are two challenges: 1)
Most traffic forecasting studies mainly focus on modeling correlations of
neighboring sensors and ignore correlations of remote sensors, e.g., business
districts with similar spatio-temporal patterns; 2) Prior methods which use
static adjacency matrix in graph convolutional networks (GCNs) are not enough
to reflect the dynamic spatial dependence in traffic system. Moreover,
fine-grained methods which use self-attention to model dynamic correlations of
all sensors ignore hierarchical information in road networks and have quadratic
computational complexity. In this paper, we propose a novel dynamic multi-graph
convolution recurrent network (DMGCRN) to tackle above issues, which can model
the spatial correlations of distance, the spatial correlations of structure,
and the temporal correlations simultaneously. We not only use the
distance-based graph to capture spatial information from nodes are close in
distance but also construct a novel latent graph which encoded the structure
correlations among roads to capture spatial information from nodes are similar
in structure. Furthermore, we divide the neighbors of each sensor into
coarse-grained regions, and dynamically assign different weights to each region
at different times. Meanwhile, we integrate the dynamic multi-graph convolution
network into the gated recurrent unit (GRU) to capture temporal dependence.
Extensive experiments on three real-world traffic datasets demonstrate that our
proposed algorithm outperforms state-of-the-art baselines.

    

### [[2112.02287] BenchML: an extensible pipelining framework for benchmarking representations of materials and molecules at scale](http://arxiv.org/abs/2112.02287)


  We introduce a machine-learning (ML) framework for high-throughput
benchmarking of diverse representations of chemical systems against datasets of
materials and molecules. The guiding principle underlying the benchmarking
approach is to evaluate raw descriptor performance by limiting model complexity
to simple regression schemes while enforcing best ML practices, allowing for
unbiased hyperparameter optimization, and assessing learning progress through
learning curves along series of synchronized train-test splits. The resulting
models are intended as baselines that can inform future method development,
next to indicating how easily a given dataset can be learnt. Through a
comparative analysis of the training outcome across a diverse set of
physicochemical, topological and geometric representations, we glean insight
into the relative merits of these representations as well as their
interrelatedness.

    

### [[2112.02290] Interactive Disentanglement: Learning Concepts by Interacting with their Prototype Representations](http://arxiv.org/abs/2112.02290)


  Learning visual concepts from raw images without strong supervision is a
challenging task. In this work, we show the advantages of prototype
representations for understanding and revising the latent space of neural
concept learners. For this purpose, we introduce interactive Concept Swapping
Networks (iCSNs), a novel framework for learning concept-grounded
representations via weak supervision and implicit prototype representations.
iCSNs learn to bind conceptual information to specific prototype slots by
swapping the latent representations of paired images. This semantically
grounded and discrete latent space facilitates human understanding and
human-machine interaction. We support this claim by conducting experiments on
our novel data set "Elementary Concept Reasoning" (ECR), focusing on visual
concepts shared by geometric objects.

    

### [[2112.02291] KDCTime: Knowledge Distillation with Calibration on InceptionTime for Time-series Classification](http://arxiv.org/abs/2112.02291)


  Time-series classification approaches based on deep neural networks are easy
to be overfitting on UCR datasets, which is caused by the few-shot problem of
those datasets. Therefore, in order to alleviate the overfitting phenomenon for
further improving the accuracy, we first propose Label Smoothing for
InceptionTime (LSTime), which adopts the information of soft labels compared to
just hard labels. Next, instead of manually adjusting soft labels by LSTime,
Knowledge Distillation for InceptionTime (KDTime) is proposed in order to
automatically generate soft labels by the teacher model. At last, in order to
rectify the incorrect predicted soft labels from the teacher model, Knowledge
Distillation with Calibration for InceptionTime (KDCTime) is proposed, where it
contains two optional calibrating strategies, i.e. KDC by Translating (KDCT)
and KDC by Reordering (KDCR). The experimental results show that the accuracy
of KDCTime is promising, while its inference time is two orders of magnitude
faster than ROCKET with an acceptable training time overhead.

    

### [[2112.02292] PreGAN: Preemptive Migration Prediction Network for Proactive Fault-Tolerant Edge Computing](http://arxiv.org/abs/2112.02292)


  Building a fault-tolerant edge system that can quickly react to node
overloads or failures is challenging due to the unreliability of edge devices
and the strict service deadlines of modern applications. Moreover, unnecessary
task migrations can stress the system network, giving rise to the need for a
smart and parsimonious failure recovery scheme. Prior approaches often fail to
adapt to highly volatile workloads or accurately detect and diagnose faults for
optimal remediation. There is thus a need for a robust and proactive
fault-tolerance mechanism to meet service level objectives. In this work, we
propose PreGAN, a composite AI model using a Generative Adversarial Network
(GAN) to predict preemptive migration decisions for proactive fault-tolerance
in containerized edge deployments. PreGAN uses co-simulations in tandem with a
GAN to learn a few-shot anomaly classifier and proactively predict migration
decisions for reliable computing. Extensive experiments on a Raspberry-Pi based
edge environment show that PreGAN can outperform state-of-the-art baseline
methods in fault-detection, diagnosis and classification, thus achieving high
quality of service. PreGAN accomplishes this by 5.1% more accurate fault
detection, higher diagnosis scores and 23.8% lower overheads compared to the
best method among the considered baselines.

    

### [[2112.02301] Adaptive label thresholding methods for online multi-label classification](http://arxiv.org/abs/2112.02301)


  Existing online multi-label classification works cannot well handle the
online label thresholding problem and lack the regret analysis for their online
algorithms. This paper proposes a novel framework of adaptive label
thresholding algorithms for online multi-label classification, with the aim to
overcome the drawbacks of existing methods. The key feature of our framework is
that both scoring and thresholding models are included as important components
of the online multi-label classifier and are incorporated into one online
optimization problem. Further, in order to establish the relationship between
scoring and thresholding models, a novel multi-label classification loss
function is derived, which measures to what an extent the multi-label
classifier can distinguish between relevant labels and irrelevant ones for an
incoming instance. Based on this new framework and loss function, we present a
first-order linear algorithm and a second-order one, which both enjoy closed
form update, but rely on different techniques for updating the multi-label
classifier. Both algorithms are proved to achieve a sub-linear regret. Using
Mercer kernels, our first-order algorithm has been extended to deal with
nonlinear multi-label prediction tasks. Experiments show the advantage of our
linear and nonlinear algorithms, in terms of various multi-label performance
metrics.

    

### [[2112.02309] Artificial Intelligence and Machine Learning in Nuclear Physics](http://arxiv.org/abs/2112.02309)


  Advances in artificial intelligence/machine learning methods provide tools
that have broad applicability in scientific research. These techniques are
being applied across the diversity of nuclear physics research topics, leading
to advances that will facilitate scientific discoveries and societal
applications.
This Review gives a snapshot of nuclear physics research which has been
transformed by artificial intelligence and machine learning techniques.

    

### [[2112.02336] Efficient Pressure: Improving efficiency for signalized intersections](http://arxiv.org/abs/2112.02336)


  Since conventional approaches could not adapt to dynamic traffic conditions,
reinforcement learning (RL) has attracted more attention to help solve the
traffic signal control (TSC) problem. However, existing RL-based methods are
rarely deployed considering that they are neither cost-effective in terms of
computing resources nor more robust than traditional approaches, which raises a
critical research question: how to construct an adaptive controller for TSC
with less training and reduced complexity based on RL-based approach? To
address this question, in this paper, we (1) innovatively specify the traffic
movement representation as a simple but efficient pressure of vehicle queues in
a traffic network, namely efficient pressure (EP); (2) build a traffic signal
settings protocol, including phase duration, signal phase number and EP for
TSC; (3) design a TSC approach based on the traditional max pressure (MP)
approach, namely efficient max pressure (Efficient-MP) using the EP to capture
the traffic state; and (4) develop a general RL-based TSC algorithm template:
efficient Xlight (Efficient-XLight) under EP. Through comprehensive experiments
on multiple real-world datasets in our traffic signal settings' protocol for
TSC, we demonstrate that efficient pressure is complementary to traditional and
RL-based modeling to design better TSC methods. Our code is released on Github.

    

### [[2112.02342] Overcome Anterograde Forgetting with Cycled Memory Networks](http://arxiv.org/abs/2112.02342)


  Learning from a sequence of tasks for a lifetime is essential for an agent
towards artificial general intelligence. This requires the agent to
continuously learn and memorize new knowledge without interference. This paper
first demonstrates a fundamental issue of lifelong learning using neural
networks, named anterograde forgetting, i.e., preserving and transferring
memory may inhibit the learning of new knowledge. This is attributed to the
fact that the learning capacity of a neural network will be reduced as it keeps
memorizing historical knowledge, and the fact that conceptual confusion may
occur as it transfers irrelevant old knowledge to the current task. This work
proposes a general framework named Cycled Memory Networks (CMN) to address the
anterograde forgetting in neural networks for lifelong learning. The CMN
consists of two individual memory networks to store short-term and long-term
memories to avoid capacity shrinkage. A transfer cell is designed to connect
these two memory networks, enabling knowledge transfer from the long-term
memory network to the short-term memory network to mitigate the conceptual
confusion, and a memory consolidation mechanism is developed to integrate
short-term knowledge into the long-term memory network for knowledge
accumulation. Experimental results demonstrate that the CMN can effectively
address the anterograde forgetting on several task-related, task-conflict,
class-incremental and cross-domain benchmarks.

    

### [[2112.02346] Logic Shrinkage: Learned FPGA Netlist Sparsity for Efficient Neural Network Inference](http://arxiv.org/abs/2112.02346)


  FPGA-specific DNN architectures using the native LUTs as independently
trainable inference operators have been shown to achieve favorable
area-accuracy and energy-accuracy tradeoffs. The first work in this area,
LUTNet, exhibited state-of-the-art performance for standard DNN benchmarks. In
this paper, we propose the learned optimization of such LUT-based topologies,
resulting in higher-efficiency designs than via the direct use of
off-the-shelf, hand-designed networks. Existing implementations of this class
of architecture require the manual specification of the number of inputs per
LUT, K. Choosing appropriate K a priori is challenging, and doing so at even
high granularity, e.g. per layer, is a time-consuming and error-prone process
that leaves FPGAs' spatial flexibility underexploited. Furthermore, prior works
see LUT inputs connected randomly, which does not guarantee a good choice of
network topology. To address these issues, we propose logic shrinkage, a
fine-grained netlist pruning methodology enabling K to be automatically learned
for every LUT in a neural network targeted for FPGA inference. By removing LUT
inputs determined to be of low importance, our method increases the efficiency
of the resultant accelerators. Our GPU-friendly solution to LUT input removal
is capable of processing large topologies during their training with negligible
slowdown. With logic shrinkage, we better the area and energy efficiency of the
best-performing LUTNet implementation of the CNV network classifying CIFAR-10
by 1.54x and 1.31x, respectively, while matching its accuracy. This
implementation also reaches 2.71x the area efficiency of an equally accurate,
heavily pruned BNN. On ImageNet with the Bi-Real Net architecture, employment
of logic shrinkage results in a post-synthesis area reduction of 2.67x vs
LUTNet, allowing for implementation that was previously impossible on today's
largest FPGAs.

    

### [[2112.02353] Label Hierarchy Transition: Modeling Class Hierarchies to Enhance Deep Classifiers](http://arxiv.org/abs/2112.02353)


  Hierarchical classification aims to sort the object into a hierarchy of
categories. For example, a bird can be categorized according to a three-level
hierarchy of order, family, and species. Existing methods commonly address
hierarchical classification by decoupling it into several multi-class
classification tasks. However, such a multi-task learning strategy fails to
fully exploit the correlation among various categories across different
hierarchies. In this paper, we propose Label Hierarchy Transition, a unified
probabilistic framework based on deep learning, to address hierarchical
classification. Specifically, we explicitly learn the label hierarchy
transition matrices, whose column vectors represent the conditional label
distributions of classes between two adjacent hierarchies and could be capable
of encoding the correlation embedded in class hierarchies. We further propose a
confusion loss, which encourages the classification network to learn the
correlation across different label hierarchies during training. The proposed
framework can be adapted to any existing deep network with only minor
modifications. We experiment with three public benchmark datasets with various
class hierarchies, and the results demonstrate the superiority of our approach
beyond the prior arts. Source code will be made publicly available.

    

### [[2112.02365] TransBoost: A Boosting-Tree Kernel Transfer Learning Algorithm for Improving Financial Inclusion](http://arxiv.org/abs/2112.02365)


  The prosperity of mobile and financial technologies has bred and expanded
various kinds of financial products to a broader scope of people, which
contributes to advocating financial inclusion. It has non-trivial social
benefits of diminishing financial inequality. However, the technical challenges
in individual financial risk evaluation caused by the distinct characteristic
distribution and limited credit history of new users, as well as the
inexperience of newly-entered companies in handling complex data and obtaining
accurate labels, impede further promoting financial inclusion. To tackle these
challenges, this paper develops a novel transfer learning algorithm (i.e.,
TransBoost) that combines the merits of tree-based models and kernel methods.
The TransBoost is designed with a parallel tree structure and efficient weights
updating mechanism with theoretical guarantee, which enables it to excel in
tackling real-world data with high dimensional features and sparsity in $O(n)$
time complexity. We conduct extensive experiments on two public datasets and a
unique large-scale dataset from Tencent Mobile Payment. The results show that
the TransBoost outperforms other state-of-the-art benchmark transfer learning
algorithms in terms of prediction accuracy with superior efficiency, shows
stronger robustness to data sparsity, and provides meaningful model
interpretation. Besides, given a financial risk level, the TransBoost enables
financial service providers to serve the largest number of users including
those who would otherwise be excluded by other algorithms. That is, the
TransBoost improves financial inclusion.

    

### [[2112.02382] My(o) Armband Leaks Passwords: An EMG and IMU Based Keylogging Side-Channel Attack](http://arxiv.org/abs/2112.02382)


  Wearables that constantly collect various sensor data of their users increase
the chances for inferences of unintentional and sensitive information such as
passwords typed on a physical keyboard. We take a thorough look at the
potential of using electromyographic (EMG) data, a sensor modality which is new
to the market but has lately gained attention in the context of wearables for
augmented reality (AR), for a keylogging side-channel attack. Our approach is
based on neural networks for a between-subject attack in a realistic scenario
using the Myo Armband to collect the sensor data. In our approach, the EMG data
has proven to be the most prominent source of information compared to the
accelerometer and gyroscope, increasing the keystroke detection performance.
For our end-to-end approach on raw data, we report a mean balanced accuracy of
about 76 % for the keystroke detection and a mean top-3 key accuracy of about
32 % on 52 classes for the key identification on passwords of varying
strengths. We have created an extensive dataset including more than 310 000
keystrokes recorded from 37 volunteers, which is available as open access along
with the source code used to create the given results.

    

### [[2112.02393] Optimization-Based Separations for Neural Networks](http://arxiv.org/abs/2112.02393)


  Depth separation results propose a possible theoretical explanation for the
benefits of deep neural networks over shallower architectures, establishing
that the former possess superior approximation capabilities. However, there are
no known results in which the deeper architecture leverages this advantage into
a provable optimization guarantee. We prove that when the data are generated by
a distribution with radial symmetry which satisfies some mild assumptions,
gradient descent can efficiently learn ball indicator functions using a depth 2
neural network with two layers of sigmoidal activations, and where the hidden
layer is held fixed throughout training. Since it is known that ball indicators
are hard to approximate with respect to a certain heavy-tailed distribution
when using depth 2 networks with a single layer of non-linearities (Safran and
Shamir, 2017), this establishes what is to the best of our knowledge, the first
optimization-based separation result where the approximation benefits of the
stronger architecture provably manifest in practice. Our proof technique relies
on a random features approach which reduces the problem to learning with a
single neuron, where new tools are required to show the convergence of gradient
descent when the distribution of the data is heavy-tailed.

    

### [[2112.02409] Understanding Dynamic Spatio-Temporal Contexts in Long Short-Term Memory for Road Traffic Speed Prediction](http://arxiv.org/abs/2112.02409)


  Reliable traffic flow prediction is crucial to creating intelligent
transportation systems. Many big-data-based prediction approaches have been
developed but they do not reflect complicated dynamic interactions between
roads considering time and location. In this study, we propose a dynamically
localised long short-term memory (LSTM) model that involves both spatial and
temporal dependence between roads. To do so, we use a localised dynamic spatial
weight matrix along with its dynamic variation. Moreover, the LSTM model can
deal with sequential data with long dependency as well as complex non-linear
features. Empirical results indicated superior prediction performances of the
proposed model compared to two different baseline methods.

    

### [[2112.02421] Nonparametric mixture MLEs under Gaussian-smoothed optimal transport distance](http://arxiv.org/abs/2112.02421)


  The Gaussian-smoothed optimal transport (GOT) framework, pioneered in
Goldfeld et al. (2020) and followed up by a series of subsequent papers, has
quickly caught attention among researchers in statistics, machine learning,
information theory, and related fields. One key observation made therein is
that, by adapting to the GOT framework instead of its unsmoothed counterpart,
the curse of dimensionality for using the empirical measure to approximate the
true data generating distribution can be lifted. The current paper shows that a
related observation applies to the estimation of nonparametric mixing
distributions in discrete exponential family models, where under the GOT cost
the estimation accuracy of the nonparametric MLE can be accelerated to a
polynomial rate. This is in sharp contrast to the classical sub-polynomial
rates based on unsmoothed metrics, which cannot be improved from an
information-theoretical perspective. A key step in our analysis is the
establishment of a new Jackson-type approximation bound of Gaussian-convoluted
Lipschitz functions. This insight bridges existing techniques of analyzing the
nonparametric MLEs and the new GOT framework.

    

### [[2112.02424] Variational Wasserstein gradient flow](http://arxiv.org/abs/2112.02424)


  The gradient flow of a function over the space of probability densities with
respect to the Wasserstein metric often exhibits nice properties and has been
utilized in several machine learning applications. The standard approach to
compute the Wasserstein gradient flow is the finite difference which
discretizes the underlying space over a grid, and is not scalable. In this
work, we propose a scalable proximal gradient type algorithm for Wasserstein
gradient flow. The key of our method is a variational formulation of the
objective function, which makes it possible to realize the JKO proximal map
through a primal-dual optimization. This primal-dual problem can be efficiently
solved by alternatively updating the parameters in the inner and outer loops.
Our framework covers all the classical Wasserstein gradient flows including the
heat equation and the porous medium equation. We demonstrate the performance
and scalability of our algorithm with several numerical examples.

    

### [[2112.02446] Fast Graph Neural Tangent Kernel via Kronecker Sketching](http://arxiv.org/abs/2112.02446)


  Many deep learning tasks have to deal with graphs (e.g., protein structures,
social networks, source code abstract syntax trees). Due to the importance of
these tasks, people turned to Graph Neural Networks (GNNs) as the de facto
method for learning on graphs. GNNs have become widely applied due to their
convincing performance. Unfortunately, one major barrier to using GNNs is that
GNNs require substantial time and resources to train. Recently, a new method
for learning on graph data is Graph Neural Tangent Kernel (GNTK) [Du, Hou,
Salakhutdinov, Poczos, Wang and Xu 19]. GNTK is an application of Neural
Tangent Kernel (NTK) [Jacot, Gabriel and Hongler 18] (a kernel method) on graph
data, and solving NTK regression is equivalent to using gradient descent to
train an infinite-wide neural network. The key benefit of using GNTK is that,
similar to any kernel method, GNTK's parameters can be solved directly in a
single step. This can avoid time-consuming gradient descent. Meanwhile,
sketching has become increasingly used in speeding up various optimization
problems, including solving kernel regression. Given a kernel matrix of $n$
graphs, using sketching in solving kernel regression can reduce the running
time to $o(n^3)$. But unfortunately such methods usually require extensive
knowledge about the kernel matrix beforehand, while in the case of GNTK we find
that the construction of the kernel matrix is already $O(n^2N^4)$, assuming
each graph has $N$ nodes. The kernel matrix construction time can be a major
performance bottleneck when the size of graphs $N$ increases. A natural
question to ask is thus whether we can speed up the kernel matrix construction
to improve GNTK regression's end-to-end running time. This paper provides the
first algorithm to construct the kernel matrix in $o(n^2N^3)$ running time.

    

### [[2112.02448] Emojich -- zero-shot emoji generation using Russian language: a technical report](http://arxiv.org/abs/2112.02448)


  This technical report presents a text-to-image neural network "Emojich" that
generates emojis using captions in Russian language as a condition. We aim to
keep the generalization ability of a pretrained big model ruDALL-E Malevich
(XL) 1.3B parameters at the fine-tuning stage, while giving special style to
the images generated. Here are presented some engineering methods, code
realization, all hyper-parameters for reproducing results and a Telegram bot
where everyone can create their own customized sets of stickers. Also, some
newly generated emojis obtained by "Emojich" model are demonstrated.

    

### [[2112.02468] Anomaly Detection of Wind Turbine Time Series using Variational Recurrent Autoencoders](http://arxiv.org/abs/2112.02468)


  Ice accumulation in the blades of wind turbines can cause them to describe
anomalous rotations or no rotations at all, thus affecting the generation of
electricity and power output. In this work, we investigate the problem of ice
accumulation in wind turbines by framing it as anomaly detection of
multi-variate time series. Our approach focuses on two main parts: first,
learning low-dimensional representations of time series using a Variational
Recurrent Autoencoder (VRAE), and second, using unsupervised clustering
algorithms to classify the learned representations as normal (no ice
accumulated) or abnormal (ice accumulated). We have evaluated our approach on a
custom wind turbine time series dataset, for the two-classes problem (one
normal versus one abnormal class), we obtained a classification accuracy of up
to 96$\%$ on test data. For the multiple-class problem (one normal versus
multiple abnormal classes), we present a qualitative analysis of the
low-dimensional learned latent space, providing insights into the capacities of
our approach to tackle such problem. The code to reproduce this work can be
found here this https URL.

    

### [[2112.02472] Augmentation-Free Self-Supervised Learning on Graphs](http://arxiv.org/abs/2112.02472)


  Inspired by the recent success of self-supervised methods applied on images,
self-supervised learning on graph structured data has seen rapid growth
especially centered on augmentation-based contrastive methods. However, we
argue that without carefully designed augmentation techniques, augmentations on
graphs may behave arbitrarily in that the underlying semantics of graphs can
drastically change. As a consequence, the performance of existing
augmentation-based methods is highly dependent on the choice of augmentation
scheme, i.e., hyperparameters associated with augmentations. In this paper, we
propose a novel augmentation-free self-supervised learning framework for
graphs, named AFGRL. Specifically, we generate an alternative view of a graph
by discovering nodes that share the local structural information and the global
semantics with the graph. Extensive experiments towards various node-level
tasks, i.e., node classification, clustering, and similarity search on various
real-world datasets demonstrate the superiority of AFGRL. The source code for
AFGRL is available at this https URL.

    

### [[2112.02478] Classification of COVID-19 on chest X-Ray images using Deep Learning model with Histogram Equalization and Lungs Segmentation](http://arxiv.org/abs/2112.02478)


  Background and Objective: Artificial intelligence (AI) methods coupled with
biomedical analysis has a critical role during pandemics as it helps to release
the overwhelming pressure from healthcare systems and physicians. As the
ongoing COVID-19 crisis worsens in countries having dense populations and
inadequate testing kits like Brazil and India, radiological imaging can act as
an important diagnostic tool to accurately classify covid-19 patients and
prescribe the necessary treatment in due time. With this motivation, we present
our study based on deep learning architecture for detecting covid-19 infected
lungs using chest X-rays. Dataset: We collected a total of 2470 images for
three different class labels, namely, healthy lungs, ordinary pneumonia, and
covid-19 infected pneumonia, out of which 470 X-ray images belong to the
covid-19 category. Methods: We first pre-process all the images using histogram
equalization techniques and segment them using U-net architecture. VGG-16
network is then used for feature extraction from the pre-processed images which
is further sampled by SMOTE oversampling technique to achieve a balanced
dataset. Finally, the class-balanced features are classified using a support
vector machine (SVM) classifier with 10-fold cross-validation and the accuracy
is evaluated. Result and Conclusion: Our novel approach combining well-known
pre-processing techniques, feature extraction methods, and dataset balancing
method, lead us to an outstanding rate of recognition of 98% for COVID-19
images over a dataset of 2470 X-ray images. Our model is therefore fit to be
utilized in healthcare facilities for screening purposes.

    

### [[2112.02487] Face Trees for Expression Recognition](http://arxiv.org/abs/2112.02487)


  We propose an end-to-end architecture for facial expression recognition. Our
model learns an optimal tree topology for facial landmarks, whose traversal
generates a sequence from which we obtain an embedding to feed a sequential
learner. The proposed architecture incorporates two main streams, one focusing
on landmark positions to learn the structure of the face, while the other
focuses on patches around the landmarks to learn texture information. Each
stream is followed by an attention mechanism and the outputs are fed to a
two-stream fusion component to perform the final classification. We conduct
extensive experiments on two large-scale publicly available facial expression
datasets, AffectNet and FER2013, to evaluate the efficacy of our approach. Our
method outperforms other solutions in the area and sets new state-of-the-art
expression recognition rates on these datasets.

    

### [[2112.02488] Exploring Complicated Search Spaces with Interleaving-Free Sampling](http://arxiv.org/abs/2112.02488)


  The existing neural architecture search algorithms are mostly working on
search spaces with short-distance connections. We argue that such designs,
though safe and stable, obstacles the search algorithms from exploring more
complicated scenarios. In this paper, we build the search algorithm upon a
complicated search space with long-distance connections, and show that existing
weight-sharing search algorithms mostly fail due to the existence of
\textbf{interleaved connections}. Based on the observation, we present a simple
yet effective algorithm named \textbf{IF-NAS}, where we perform a periodic
sampling strategy to construct different sub-networks during the search
procedure, avoiding the interleaved connections to emerge in any of them. In
the proposed search space, IF-NAS outperform both random sampling and previous
weight-sharing search algorithms by a significant margin. IF-NAS also
generalizes to the micro cell-based spaces which are much easier. Our research
emphasizes the importance of macro structure and we look forward to further
efforts along this direction.

    

### [[2112.02499] Radial Basis Function Approximation with Distributively Stored Data on Spheres](http://arxiv.org/abs/2112.02499)


  This paper proposes a distributed weighted regularized least squares
algorithm (DWRLS) based on spherical radial basis functions and spherical
quadrature rules to tackle spherical data that are stored across numerous local
servers and cannot be shared with each other. Via developing a novel integral
operator approach, we succeed in deriving optimal approximation rates for DWRLS
and theoretically demonstrate that DWRLS performs similarly as running a
weighted regularized least squares algorithm with the whole data on a large
enough machine. This interesting finding implies that distributed learning is
capable of sufficiently exploiting potential values of distributively stored
spherical data, even though every local server cannot access all the data.

    

### [[2112.02504] A Novel Sequential Coreset Method for Gradient Descent Algorithms](http://arxiv.org/abs/2112.02504)


  A wide range of optimization problems arising in machine learning can be
solved by gradient descent algorithms, and a central question in this area is
how to efficiently compress a large-scale dataset so as to reduce the
computational complexity. {\em Coreset} is a popular data compression technique
that has been extensively studied before. However, most of existing coreset
methods are problem-dependent and cannot be used as a general tool for a
broader range of applications. A key obstacle is that they often rely on the
pseudo-dimension and total sensitivity bound that can be very high or hard to
obtain. In this paper, based on the ''locality'' property of gradient descent
algorithms, we propose a new framework, termed ''sequential coreset'', which
effectively avoids these obstacles. Moreover, our method is particularly
suitable for sparse optimization whence the coreset size can be further reduced
to be only poly-logarithmically dependent on the dimension. In practice, the
experimental results suggest that our method can save a large amount of running
time compared with the baseline algorithms.

    

### [[2112.02505] Causal Distillation for Language Models](http://arxiv.org/abs/2112.02505)


  Distillation efforts have led to language models that are more compact and
efficient without serious drops in performance. The standard approach to
distillation trains a student model against two objectives: a task-specific
objective (e.g., language modeling) and an imitation objective that encourages
the hidden states of the student model to be similar to those of the larger
teacher model. In this paper, we show that it is beneficial to augment
distillation with a third objective that encourages the student to imitate the
causal computation process of the teacher through interchange intervention
training(IIT). IIT pushes the student model to become a causal abstraction of
the teacher model - a simpler model with the same causal structure. IIT is
fully differentiable, easily implemented, and combines flexibly with other
objectives. Compared with standard distillation of BERT, distillation via IIT
results in lower perplexity on Wikipedia (masked language modeling) and marked
improvements on the GLUE benchmark (natural language understanding), SQuAD
(question answering), and CoNLL-2003 (named entity recognition).

    

### [[2112.02520] Neural Photometry-guided Visual Attribute Transfer](http://arxiv.org/abs/2112.02520)


  We present a deep learning-based method for propagating spatially-varying
visual material attributes (e.g. texture maps or image stylizations) to larger
samples of the same or similar materials. For training, we leverage images of
the material taken under multiple illuminations and a dedicated data
augmentation policy, making the transfer robust to novel illumination
conditions and affine deformations. Our model relies on a supervised
image-to-image translation framework and is agnostic to the transferred domain;
we showcase a semantic segmentation, a normal map, and a stylization. Following
an image analogies approach, the method only requires the training data to
contain the same visual structures as the input guidance. Our approach works at
interactive rates, making it suitable for material edit applications. We
thoroughly evaluate our learning methodology in a controlled setup providing
quantitative measures of performance. Last, we demonstrate that training the
model on a single material is enough to generalize to materials of the same
type without the need for massive datasets.

    

### [[2112.02521] Inf-CP: A Reliable Channel Pruning based on Channel Influence](http://arxiv.org/abs/2112.02521)


  One of the most effective methods of channel pruning is to trim on the basis
of the importance of each neuron. However, measuring the importance of each
neuron is an NP-hard problem. Previous works have proposed to trim by
considering the statistics of a single layer or a plurality of successive
layers of neurons. These works cannot eliminate the influence of different data
on the model in the reconstruction error, and currently, there is no work to
prove that the absolute values of the parameters can be directly used as the
basis for judging the importance of the weights. A more reasonable approach is
to eliminate the difference between batch data that accurately measures the
weight of influence. In this paper, we propose to use ensemble learning to
train a model for different batches of data and use the influence function (a
classic technique from robust statistics) to learn the algorithm to track the
model's prediction and return its training parameter gradient, so that we can
determine the responsibility for each parameter, which we call "influence", in
the prediction process. In addition, we theoretically prove that the
back-propagation of the deep network is a first-order Taylor approximation of
the influence function of the weights. We perform extensive experiments to
prove that pruning based on the influence function using the idea of ensemble
learning will be much more effective than just focusing on error
reconstruction. Experiments on CIFAR shows that the influence pruning achieves
the state-of-the-art result.

    

### [[2112.02531] Trivial bundle embeddings for learning graph representations](http://arxiv.org/abs/2112.02531)


  Embedding real-world networks presents challenges because it is not clear how
to identify their latent geometries. Embedding some disassortative networks,
such as scale-free networks, to the Euclidean space has been shown to incur
distortions. Embedding scale-free networks to hyperbolic spaces offer an
exciting alternative but incurs distortions when embedding assortative networks
with latent geometries not hyperbolic. We propose an inductive model that
leverages both the expressiveness of GCNs and trivial bundle to learn inductive
node representations for networks with or without node features. A trivial
bundle is a simple case of fiber bundles,a space that is globally a product
space of its base space and fiber. The coordinates of base space and those of
fiber can be used to express the assortative and disassortative factors in
generating edges. Therefore, the model has the ability to learn embeddings that
can express those factors. In practice, it reduces errors for link prediction
and node classification when compared to the Euclidean and hyperbolic GCNs.

    

### [[2112.02542] Robust Active Learning: Sample-Efficient Training of Robust Deep Learning Models](http://arxiv.org/abs/2112.02542)


  Active learning is an established technique to reduce the labeling cost to
build high-quality machine learning models. A core component of active learning
is the acquisition function that determines which data should be selected to
annotate. State-of-the-art acquisition functions -- and more largely, active
learning techniques -- have been designed to maximize the clean performance
(e.g. accuracy) and have disregarded robustness, an important quality property
that has received increasing attention. Active learning, therefore, produces
models that are accurate but not robust.
In this paper, we propose \emph{robust active learning}, an active learning
process that integrates adversarial training -- the most established method to
produce robust models. Via an empirical study on 11 acquisition functions, 4
datasets, 6 DNN architectures, and 15105 trained DNNs, we show that robust
active learning can produce models with the robustness (accuracy on adversarial
examples) ranging from 2.35\% to 63.85\%, whereas standard active learning
systematically achieves negligible robustness (less than 0.20\%). Our study
also reveals, however, that the acquisition functions that perform well on
accuracy are worse than random sampling when it comes to robustness. We,
therefore, examine the reasons behind this and devise a new acquisition
function that targets both clean performance and robustness. Our acquisition
function -- named density-based robust sampling with entropy (DRE) --
outperforms the other acquisition functions (including random) in terms of
robustness by up to 24.40\% (3.84\% than random particularly), while remaining
competitive on accuracy. Additionally, we prove that DRE is applicable as a
test selection metric for model retraining and stands out from all compared
functions by up to 8.21\% robustness.

    

### [[2112.02543] Joint Superposition Coding and Training for Federated Learning over Multi-Width Neural Networks](http://arxiv.org/abs/2112.02543)


  This paper aims to integrate two synergetic technologies, federated learning
(FL) and width-adjustable slimmable neural network (SNN) architectures. FL
preserves data privacy by exchanging the locally trained models of mobile
devices. By adopting SNNs as local models, FL can flexibly cope with the
time-varying energy capacities of mobile devices. Combining FL and SNNs is
however non-trivial, particularly under wireless connections with time-varying
channel conditions. Furthermore, existing multi-width SNN training algorithms
are sensitive to the data distributions across devices, so are ill-suited to
FL. Motivated by this, we propose a communication and energy-efficient
SNN-based FL (named SlimFL) that jointly utilizes superposition coding (SC) for
global model aggregation and superposition training (ST) for updating local
models. By applying SC, SlimFL exchanges the superposition of multiple width
configurations that are decoded as many as possible for a given communication
throughput. Leveraging ST, SlimFL aligns the forward propagation of different
width configurations, while avoiding the inter-width interference during
backpropagation. We formally prove the convergence of SlimFL. The result
reveals that SlimFL is not only communication-efficient but also can counteract
non-IID data distributions and poor channel conditions, which is also
corroborated by simulations.

    

### [[2112.02563] A Novel Approach to Solving Goal-Achieving Problems for Board Games](http://arxiv.org/abs/2112.02563)


  Goal-achieving problems are puzzles that set up a specific situation with a
clear objective. An example that is well-studied is the category of
life-and-death (L&D) problems for Go, which helps players hone their skill of
identifying region safety. Many previous methods like lambda search try null
moves first, then derive so-called relevance zones (RZs), outside of which the
opponent does not need to search. This paper first proposes a novel RZ-based
approach, called the RZ-Based Search (RZS), to solving L&D problems for Go. RZS
tries moves before determining whether they are null moves post-hoc. This means
we do not need to rely on null move heuristics, resulting in a more elegant
algorithm, so that it can also be seamlessly incorporated into AlphaZero's
super-human level play in our solver. To repurpose AlphaZero for solving, we
also propose a new training method called Faster to Life (FTL), which modifies
AlphaZero to entice it to win more quickly. We use RZS and FTL to solve L&D
problems on Go, namely solving 68 among 106 problems from a professional L&D
book while a previous program solves 11 only. Finally, we discuss that the
approach is generic in the sense that RZS is applicable to solving many other
goal-achieving problems for board games.

    

### [[2112.02577] Smart IoT-Biofloc water management system using Decision regression tree](http://arxiv.org/abs/2112.02577)


  The conventional fishing industry has several difficulties: water
contamination, temperature instability, nutrition, area, expense, etc. In fish
farming, Biofloc technology turns traditional farming into a sophisticated
infrastructure that enables the utilization of leftover food by turning it into
bacterial biomass. The purpose of our study is to propose an intelligent IoT
Biofloc system that improves efficiency and production. This article introduced
a system that gathers data from sensors, store data in the cloud, analyses it
using a machine learning model such as a Decision regression tree model to
predict the water condition, and provides real-time monitoring through an
android app. The proposed system has achieved a satisfactory accuracy of 79%
during the experiment.

    

### [[2112.02589] Local Adaptivity of Gradient Boosting in Histogram Transform Ensemble Learning](http://arxiv.org/abs/2112.02589)


  In this paper, we propose a gradient boosting algorithm called
\textit{adaptive boosting histogram transform} (\textit{ABHT}) for regression
to illustrate the local adaptivity of gradient boosting algorithms in histogram
transform ensemble learning. From the theoretical perspective, when the target
function lies in a locally Hölder continuous space, we show that our ABHT can
filter out the regions with different orders of smoothness. Consequently, we
are able to prove that the upper bound of the convergence rates of ABHT is
strictly smaller than the lower bound of \textit{parallel ensemble histogram
transform} (\textit{PEHT}). In the experiments, both synthetic and real-world
data experiments empirically validate the theoretical results, which
demonstrates the advantageous performance and local adaptivity of our ABHT.

    

### [[2112.02591] Multiple Interest and Fine Granularity Network for User Modeling](http://arxiv.org/abs/2112.02591)


  User modeling plays a fundamental role in industrial recommender systems,
either in the matching stage and the ranking stage, in terms of both the
customer experience and business revenue. How to extract users' multiple
interests effectively from their historical behavior sequences to improve the
relevance and personalization of the recommend results remains an open problem
for user modeling.Most existing deep-learning based approaches exploit item-ids
and category-ids but neglect fine-grained features like color and mate-rial,
which hinders modeling the fine granularity of users' this http URL the paper,
we present Multiple interest and Fine granularity Net-work (MFN), which tackle
users' multiple and fine-grained interests and construct the model from both
the similarity relationship and the combination relationship among the users'
multiple interests.Specifically, for modeling the similarity relationship, we
leverage two sets of embeddings, where one is the fixed embedding from
pre-trained models (e.g. Glove) to give the attention weights and the other is
trainable embedding to be trained with MFN together.For modeling the
combination relationship, self-attentive layers are exploited to build the
higher order combinations of different interest representations. In the
construction of network, we design an interest-extract module using attention
mechanism to capture multiple interest representations from user historical
behavior sequences and leverage an auxiliary loss to boost the distinction of
the interest representations. Then a hierarchical network is applied to model
the attention relation between the multiple interest vectors of different
granularities and the target item. We evaluate MFNon both public and industrial
datasets. The experimental results demonstrate that the proposed MFN achieves
superior performance than other existed representing methods.

    

### [[2112.02598] Real-time Informative Surgical Skill Assessment with Gaussian Process Learning](http://arxiv.org/abs/2112.02598)


  Endoscopic Sinus and Skull Base Surgeries (ESSBSs) is a challenging and
potentially dangerous surgical procedure, and objective skill assessment is the
key components to improve the effectiveness of surgical training, to
re-validate surgeons' skills, and to decrease surgical trauma and the
complication rate in operating rooms. Because of the complexity of surgical
procedures, the variation of operation styles, and the fast development of new
surgical skills, the surgical skill assessment remains a challenging problem.
This work presents a novel Gaussian Process Learning-based heuristic automatic
objective surgical skill assessment method for ESSBSs. Different with classical
surgical skill assessment algorithms, the proposed method 1) utilizes the
kinematic features in surgical instrument relative movements, instead of using
specific surgical tasks or the statistics to assess skills in real-time; 2)
provide informative feedback, instead of a summative scores; 3) has the ability
to incrementally learn from new data, instead of depending on a fixed dataset.
The proposed method projects the instrument movements into the endoscope
coordinate to reduce the data dimensionality. It then extracts the kinematic
features of the projected data and learns the relationship between surgical
skill levels and the features with the Gaussian Process learning technique. The
proposed method was verified in full endoscopic skull base and sinus surgeries
on cadavers. These surgeries have different pathology, requires different
treatment and has different complexities. The experimental results show that
the proposed method reaches 100\% prediction precision for complete surgical
procedures and 90\% precision for real-time prediction assessment.

    

### [[2112.02608] Real-time Virtual Intraoperative CT for Image Guided Surgery](http://arxiv.org/abs/2112.02608)


  Abstract. Purpose: This paper presents a scheme for generating virtual
intraoperative CT scans in order to improve surgical completeness in Endoscopic
Sinus Surgeries (ESS). Approach: The work presents three methods, the tip
motion-based, the tip trajectory-based, and the instrument based, along with
non-parametric smoothing and Gaussian Process Regression, for virtual
intraoperative CT generation. Results: The proposed methods studied and
compared on ESS performed on cadavers. Surgical results show all three methods
improve the Dice Similarity Coefficients > 86%, with F-score > 92% and
precision > 89.91%. The tip trajectory-based method was found to have best
performance and reached 96.87% precision in surgical completeness evaluation.
Conclusions: This work demonstrated that virtual intraoperative CT scans
improves the consistency between the actual surgical scene and the reference
model, and improves surgical completeness in ESS. Comparing with actual
intraoperative CT scans, the proposed scheme has no impact on existing surgical
protocols, does not require extra hardware other than the one is already
available in most ESS overcome the high costs, the repeated radiation, and the
elongated anesthesia caused by actual intraoperative CTs, and is practical in
ESS.

    

### [[2112.02611] Contextual Multi-View Query Learning for Short Text Classification in User-Generated Data](http://arxiv.org/abs/2112.02611)


  Mining user-generated content--e.g., for the early detection of outbreaks or
for extracting personal observations--often suffers from the lack of enough
training data, short document length, and informal language model. We propose a
novel multi-view active learning model, called Context-aware Co-testing with
Bagging (COCOBA), to address these issues in the classification tasks tailored
for a query word--e.g., detecting illness reports given the disease name.
COCOBA employs the context of user postings to construct two views. Then it
uses the distribution of the representations in each view to detect the regions
that are assigned to the opposite classes. This effectively leads to detecting
the contexts that the two base learners disagree on. Our model also employs a
query-by-committee model to address the usually noisy language of user
postings. The experiments testify that our model is applicable to multiple
important representative Twitter tasks and also significantly outperforms the
existing baselines.

    

### [[2112.02612] Training Structured Neural Networks Through Manifold Identification and Variance Reduction](http://arxiv.org/abs/2112.02612)


  This paper proposes an algorithm (RMDA) for training neural networks (NNs)
with a regularization term for promoting desired structures. RMDA does not
incur computation additional to proximal SGD with momentum, and achieves
variance reduction without requiring the objective function to be of the
finite-sum form. Through the tool of manifold identification from nonlinear
optimization, we prove that after a finite number of iterations, all iterates
of RMDA possess a desired structure identical to that induced by the
regularizer at the stationary point of asymptotic convergence, even in the
presence of engineering tricks like data augmentation and dropout that
complicate the training process. Experiments on training NNs with structured
sparsity confirm that variance reduction is necessary for such an
identification, and show that RMDA thus significantly outperforms existing
methods for this task. For unstructured sparsity, RMDA also outperforms a
state-of-the-art pruning method, validating the benefits of training structured
NNs through regularization.

    

### [[2112.02622] Probabilistic Deep Learning to Quantify Uncertainty in Air Quality Forecasting](http://arxiv.org/abs/2112.02622)


  Data-driven forecasts of air quality have recently achieved more accurate
short-term predictions. Despite their success, most of the current data-driven
solutions lack proper quantifications of model uncertainty that communicate how
much to trust the forecasts. Recently, several practical tools to estimate
uncertainty have been developed in probabilistic deep learning. However, there
have not been empirical applications and extensive comparisons of these tools
in the domain of air quality forecasts. Therefore, this work applies
state-of-the-art techniques of uncertainty quantification in a real-world
setting of air quality forecasts. Through extensive experiments, we describe
training probabilistic models and evaluate their predictive uncertainties based
on empirical performance, reliability of confidence estimate, and practical
applicability. We also propose improving these models using "free" adversarial
training and exploiting temporal and spatial correlation inherent in air
quality data. Our experiments demonstrate that the proposed models perform
better than previous works in quantifying uncertainty in data-driven air
quality forecasts. Overall, Bayesian neural networks provide a more reliable
uncertainty estimate but can be challenging to implement and scale. Other
scalable methods, such as deep ensemble, Monte Carlo (MC) dropout, and
stochastic weight averaging-Gaussian (SWAG), can perform well if applied
correctly but with different tradeoffs and slight variations in performance
metrics. Finally, our results show the practical impact of uncertainty
estimation and demonstrate that, indeed, probabilistic models are more suitable
for making informed decisions. Code and dataset are available at
\url{this https URL}

    

### [[2112.02625] Explainable Deep Learning in Healthcare: A Methodological Survey from an Attribution View](http://arxiv.org/abs/2112.02625)


  The increasing availability of large collections of electronic health record
(EHR) data and unprecedented technical advances in deep learning (DL) have
sparked a surge of research interest in developing DL based clinical decision
support systems for diagnosis, prognosis, and treatment. Despite the
recognition of the value of deep learning in healthcare, impediments to further
adoption in real healthcare settings remain due to the black-box nature of DL.
Therefore, there is an emerging need for interpretable DL, which allows end
users to evaluate the model decision making to know whether to accept or reject
predictions and recommendations before an action is taken. In this review, we
focus on the interpretability of the DL models in healthcare. We start by
introducing the methods for interpretability in depth and comprehensively as a
methodological reference for future researchers or clinical practitioners in
this field. Besides the methods' details, we also include a discussion of
advantages and disadvantages of these methods and which scenarios each of them
is suitable for, so that interested readers can know how to compare and choose
among them for use. Moreover, we discuss how these methods, originally
developed for solving general-domain problems, have been adapted and applied to
healthcare problems and how they can help physicians better understand these
data-driven technologies. Overall, we hope this survey can help researchers and
practitioners in both artificial intelligence (AI) and clinical fields
understand what methods we have for enhancing the interpretability of their DL
models and choose the optimal one accordingly.

    

### [[2112.02627] Ensemble and Mixed Learning Techniques for Credit Card Fraud Detection](http://arxiv.org/abs/2112.02627)


  Spurious credit card transactions are a significant source of financial
losses and urge the development of accurate fraud detection algorithms. In this
paper, we use machine learning strategies for such an aim. First, we apply a
mixed learning technique that uses K-means preprocessing before trained
classification to the problem at hand. Next, we introduce an adapted detector
ensemble technique that uses OR-logic algorithm aggregation to enhance the
detection rate. Then, both strategies are deployed in tandem in numerical
simulations using real-world transactions data. We observed from simulation
results that the proposed methods diminished computational cost and enhanced
performance concerning state-of-the-art techniques.

    

### [[2112.02639] Using Static and Dynamic Malware features to perform Malware Ascription](http://arxiv.org/abs/2112.02639)


  Malware ascription is a relatively unexplored area, and it is rather
difficult to attribute malware and detect authorship. In this paper, we employ
various Static and Dynamic features of malicious executables to classify
malware based on their family. We leverage Cuckoo Sandbox and machine learning
to make progress in this research. Post analysis, classification is performed
using various deep learning and machine learning algorithms. Using the features
gathered from VirusTotal (static) and Cuckoo (dynamic) reports, we ran the
vectorized data against Multinomial Naive Bayes, Support Vector Machine, and
Bagging using Decision Trees as the base estimator. For each classifier, we
tuned the hyper-parameters using exhaustive search methods. Our reports can be
extremely useful in malware ascription.

    

### [[2112.02646] Diverse, Global and Amortised Counterfactual Explanations for Uncertainty Estimates](http://arxiv.org/abs/2112.02646)


  To interpret uncertainty estimates from differentiable probabilistic models,
recent work has proposed generating a single Counterfactual Latent Uncertainty
Explanation (CLUE) for a given data point where the model is uncertain,
identifying a single, on-manifold change to the input such that the model
becomes more certain in its prediction. We broaden the exploration to examine
{\delta}-CLUE, the set of potential CLUEs within a {\delta} ball of the
original input in latent space. We study the diversity of such sets and find
that many CLUEs are redundant; as such, we propose DIVerse CLUE
({\nabla}-CLUE), a set of CLUEs which each propose a distinct explanation as to
how one can decrease the uncertainty associated with an input. We then further
propose GLobal AMortised CLUE (GLAM-CLUE), a distinct and novel method which
learns amortised mappings on specific groups of uncertain inputs, taking them
and efficiently transforming them in a single function call into inputs for
which a model will be certain. Our experiments show that {\delta}-CLUE,
{\nabla}-CLUE, and GLAM-CLUE all address shortcomings of CLUE and provide
beneficial explanations of uncertainty estimates to practitioners.

    

### [[2112.02650] VarCLR: Variable Semantic Representation Pre-training via Contrastive Learning](http://arxiv.org/abs/2112.02650)


  Variable names are critical for conveying intended program behavior. Machine
learning-based program analysis methods use variable name representations for a
wide range of tasks, such as suggesting new variable names and bug detection.
Ideally, such methods could capture semantic relationships between names beyond
syntactic similarity, e.g., the fact that the names average and mean are
similar. Unfortunately, previous work has found that even the best of previous
representation approaches primarily capture relatedness (whether two variables
are linked at all), rather than similarity (whether they actually have the same
meaning).
We propose VarCLR, a new approach for learning semantic representations of
variable names that effectively captures variable similarity in this stricter
sense. We observe that this problem is an excellent fit for contrastive
learning, which aims to minimize the distance between explicitly similar
inputs, while maximizing the distance between dissimilar inputs. This requires
labeled training data, and thus we construct a novel, weakly-supervised
variable renaming dataset mined from GitHub edits. We show that VarCLR enables
the effective application of sophisticated, general-purpose language models
like BERT, to variable name representation and thus also to related downstream
tasks like variable name similarity search or spelling correction. VarCLR
produces models that significantly outperform the state-of-the-art on IdBench,
an existing benchmark that explicitly captures variable similarity (as distinct
from relatedness). Finally, we contribute a release of all data, code, and
pre-trained models, aiming to provide a drop-in replacement for variable
representations used in either existing or future program analyses that rely on
variable names.

    

### [[2112.02656] Intrinisic Gradient Compression for Federated Learning](http://arxiv.org/abs/2112.02656)


  Federated learning is a rapidly-growing area of research which enables a
large number of clients to jointly train a machine learning model on
privately-held data. One of the largest barriers to wider adoption of federated
learning is the communication cost of sending model updates from and to the
clients, which is accentuated by the fact that many of these devices are
bandwidth-constrained. In this paper, we aim to address this issue by
optimizing networks within a subspace of their full parameter space, an idea
known as intrinsic dimension in the machine learning theory community. We use a
correspondence between the notion of intrinsic dimension and gradient
compressibility to derive a family of low-bandwidth optimization algorithms,
which we call intrinsic gradient compression algorithms. Specifically, we
present three algorithms in this family with different levels of upload and
download bandwidth for use in various federated settings, along with
theoretical guarantees on their performance. Finally, in large-scale federated
learning experiments with models containing up to 100M parameters, we show that
our algorithms perform extremely well compared to current state-of-the-art
gradient compression methods.

    

### [[2112.02657] Using Convolutional Neural Networks for fault analysis and alleviation in accelerator systems](http://arxiv.org/abs/2112.02657)


  Today, Neural Networks are the basis of breakthroughs in virtually every
technical domain. Their application to accelerators has recently resulted in
better performance and efficiency in these systems. At the same time, the
increasing hardware failures due to the latest (shrinked) semiconductor
technology needs to be addressed. Since accelerator systems are often used to
back time-critical applications such as self-driving cars or medical diagnosis
applications, these hardware failures must be eliminated. Our research
evaluates these failures from a systemic point of view. Based on our results,
we find critical results for the system reliability enhancement and we further
put forth an efficient method to avoid these failures with minimal hardware
overhead.

    

### [[2112.02663] ES-dRNN: A Hybrid Exponential Smoothing and Dilated Recurrent Neural Network Model for Short-Term Load Forecasting](http://arxiv.org/abs/2112.02663)


  Short-term load forecasting (STLF) is challenging due to complex time series
(TS) which express three seasonal patterns and a nonlinear trend. This paper
proposes a novel hybrid hierarchical deep learning model that deals with
multiple seasonality and produces both point forecasts and predictive intervals
(PIs). It combines exponential smoothing (ES) and a recurrent neural network
(RNN). ES extracts dynamically the main components of each individual TS and
enables on-the-fly deseasonalization, which is particularly useful when
operating on a relatively small data set. A multi-layer RNN is equipped with a
new type of dilated recurrent cell designed to efficiently model both short and
long-term dependencies in TS. To improve the internal TS representation and
thus the model's performance, RNN learns simultaneously both the ES parameters
and the main mapping function transforming inputs into forecasts. We compare
our approach against several baseline methods, including classical statistical
methods and machine learning (ML) approaches, on STLF problems for 35 European
countries. The empirical study clearly shows that the proposed model has high
expressive power to solve nonlinear stochastic forecasting problems with TS
including multiple seasonality and significant random fluctuations. In fact, it
outperforms both statistical and state-of-the-art ML models in terms of
accuracy.

    

### [[2112.02668] On the Convergence of Shallow Neural Network Training with Randomly Masked Neurons](http://arxiv.org/abs/2112.02668)


  Given a dense shallow neural network, we focus on iteratively creating,
training, and combining randomly selected subnetworks (surrogate functions),
towards training the full model. By carefully analyzing $i)$ the subnetworks'
neural tangent kernel, $ii)$ the surrogate functions' gradient, and $iii)$ how
we sample and combine the surrogate functions, we prove linear convergence rate
of the training error -- within an error region -- for an overparameterized
single-hidden layer perceptron with ReLU activations for a regression task. Our
result implies that, for fixed neuron selection probability, the error term
decreases as we increase the number of surrogate models, and increases as we
increase the number of local training steps for each selected subnetwork. The
considered framework generalizes and provides new insights on dropout training,
multi-sample dropout training, as well as Independent Subnet Training; for each
case, we provide corresponding convergence results, as corollaries of our main
theorem.

    

### [[2112.02671] Stochastic Local Winner-Takes-All Networks Enable Profound Adversarial Robustness](http://arxiv.org/abs/2112.02671)


  This work explores the potency of stochastic competition-based activations,
namely Stochastic Local Winner-Takes-All (LWTA), against powerful
(gradient-based) white-box and black-box adversarial attacks; we especially
focus on Adversarial Training settings. In our work, we replace the
conventional ReLU-based nonlinearities with blocks comprising locally and
stochastically competing linear units. The output of each network layer now
yields a sparse output, depending on the outcome of winner sampling in each
block. We rely on the Variational Bayesian framework for training and
inference; we incorporate conventional PGD-based adversarial training arguments
to increase the overall adversarial robustness. As we experimentally show, the
arising networks yield state-of-the-art robustness against powerful adversarial
attacks while retaining very high classification rate in the benign case.

    

### [[2112.02675] Learning Swarm Interaction Dynamics from Density Evolution](http://arxiv.org/abs/2112.02675)


  We consider the problem of understanding the coordinated movements of
biological or artificial swarms. In this regard, we propose a learning scheme
to estimate the coordination laws of the interacting agents from observations
of the swarm's density over time. We describe the dynamics of the swarm based
on pairwise interactions according to a Cucker-Smale flocking model, and
express the swarm's density evolution as the solution to a system of mean-field
hydrodynamic equations. We propose a new family of parametric functions to
model the pairwise interactions, which allows for the mean-field macroscopic
system of integro-differential equations to be efficiently solved as an
augmented system of PDEs. Finally, we incorporate the augmented system in an
iterative optimization scheme to learn the dynamics of the interacting agents
from observations of the swarm's density evolution over time. The results of
this work can offer an alternative approach to study how animal flocks
coordinate, create new control schemes for large networked systems, and serve
as a central part of defense mechanisms against adversarial drone attacks.

    

### [[2112.02682] BERTMap: A BERT-based Ontology Alignment System](http://arxiv.org/abs/2112.02682)


  Ontology alignment (a.k.a ontology matching (OM)) plays a critical role in
knowledge integration. Owing to the success of machine learning in many
domains, it has been applied in OM. However, the existing methods, which often
adopt ad-hoc feature engineering or non-contextual word embeddings, have not
yet outperformed rule-based systems especially in an unsupervised setting. In
this paper, we propose a novel OM system named BERTMap which can support both
unsupervised and semi-supervised settings. It first predicts mappings using a
classifier based on fine-tuning the contextual embedding model BERT on text
semantics corpora extracted from ontologies, and then refines the mappings
through extension and repair by utilizing the ontology structure and logic. Our
evaluation with three alignment tasks on biomedical ontologies demonstrates
that BERTMap can often perform better than the leading OM systems LogMap and
AML.

    

### [[2112.02694] Benchmark for Out-of-Distribution Detection in Deep Reinforcement Learning](http://arxiv.org/abs/2112.02694)


  Reinforcement Learning (RL) based solutions are being adopted in a variety of
domains including robotics, health care and industrial automation. Most focus
is given to when these solutions work well, but they fail when presented with
out of distribution inputs. RL policies share the same faults as most machine
learning models. Out of distribution detection for RL is generally not well
covered in the literature, and there is a lack of benchmarks for this task. In
this work we propose a benchmark to evaluate OOD detection methods in a
Reinforcement Learning setting, by modifying the physical parameters of
non-visual standard environments or corrupting the state observation for visual
environments. We discuss ways to generate custom RL environments that can
produce OOD data, and evaluate three uncertainty methods for the OOD detection
task. Our results show that ensemble methods have the best OOD detection
performance with a lower standard deviation across multiple environments.

    

### [[2112.02705] Beyond Robustness: Resilience Verification of Tree-Based Classifiers](http://arxiv.org/abs/2112.02705)


  In this paper we criticize the robustness measure traditionally employed to
assess the performance of machine learning models deployed in adversarial
settings. To mitigate the limitations of robustness, we introduce a new measure
called resilience and we focus on its verification. In particular, we discuss
how resilience can be verified by combining a traditional robustness
verification technique with a data-independent stability analysis, which
identifies a subset of the feature space where the model does not change its
predictions despite adversarial manipulations. We then introduce a formally
sound data-independent stability analysis for decision trees and decision tree
ensembles, which we experimentally assess on public datasets and we leverage
for resilience verification. Our results show that resilience verification is
useful and feasible in practice, yielding a more reliable security assessment
of both standard and robust decision tree models.

    

### [[2112.02713] Joint Symmetry Detection and Shape Matching for Non-Rigid Point Cloud](http://arxiv.org/abs/2112.02713)


  Despite the success of deep functional maps in non-rigid 3D shape matching,
there exists no learning framework that models both self-symmetry and shape
matching simultaneously. This is despite the fact that errors due to symmetry
mismatch are a major challenge in non-rigid shape matching. In this paper, we
propose a novel framework that simultaneously learns both self symmetry as well
as a pairwise map between a pair of shapes. Our key idea is to couple a self
symmetry map and a pairwise map through a regularization term that provides a
joint constraint on both of them, thereby, leading to more accurate maps. We
validate our method on several benchmarks where it outperforms many competitive
baselines on both tasks.

    

### [[2112.02719] A Survey on Deep learning based Document Image Enhancement](http://arxiv.org/abs/2112.02719)


  Digitized documents such as scientific articles, tax forms, invoices,
contract papers, and historic texts, are widely used nowadays. These images
could be degraded or damaged due to various reasons including poor lighting
conditions when capturing the image, shadow while scanning them, distortion
like noise and blur, aging, ink stain, bleed through, watermark, stamp, etc.
Document image enhancement and restoration play a crucial role in many
automated document analysis and recognition tasks, such as content extraction
using optical character recognition (OCR). With recent advances in deep
learning, many methods are proposed to enhance the quality of these document
images. In this paper, we review deep learning-based methods, datasets, and
metrics for different document image enhancement problems. We provide a
comprehensive overview of deep learning-based methods for six different
document image enhancement tasks, including binarization, debluring, denoising,
defading, watermark removal, and shadow removal. We summarize the main
state-of-the-art works for each task and discuss their features, challenges,
and limitations. We introduce multiple document image enhancement tasks that
have received no to little attention, including over and under exposure
correction and bleed-through removal, and identify several other promising
research directions and opportunities for future research.

    

### [[2112.02721] NL-Augmenter: A Framework for Task-Sensitive Natural Language Augmentation](http://arxiv.org/abs/2112.02721)


  Data augmentation is an important component in the robustness evaluation of
models in natural language processing (NLP) and in enhancing the diversity of
the data they are trained on. In this paper, we present NL-Augmenter, a new
participatory Python-based natural language augmentation framework which
supports the creation of both transformations (modifications to the data) and
filters (data splits according to specific features). We describe the framework
and an initial set of 117 transformations and 23 filters for a variety of
natural language tasks. We demonstrate the efficacy of NL-Augmenter by using
several of its transformations to analyze the robustness of popular natural
language models. The infrastructure, datacards and robustness analysis results
are available publicly on the NL-Augmenter repository
(\url{this https URL}).

    

### [[2112.02729] Facial Emotion Characterization and Detection using Fourier Transform and Machine Learning](http://arxiv.org/abs/2112.02729)


  We present a Fourier-based machine learning technique that characterizes and
detects facial emotions. The main challenging task in the development of
machine learning (ML) models for classifying facial emotions is the detection
of accurate emotional features from a set of training samples, and the
generation of feature vectors for constructing a meaningful feature space and
building ML models. In this paper, we hypothesis that the emotional features
are hidden in the frequency domain; hence, they can be captured by leveraging
the frequency domain and masking techniques. We also make use of the conjecture
that a facial emotions are convoluted with the normal facial features and the
other emotional features; however, they carry linearly separable spatial
frequencies (we call computational emotional frequencies). Hence, we propose a
technique by leveraging fast Fourier transform (FFT) and rectangular
narrow-band frequency kernels, and the widely used Yale-Faces image dataset. We
test the hypothesis using the performance scores of the random forest (RF) and
the artificial neural network (ANN) classifiers as the measures to validate the
effectiveness of the captured emotional frequencies. Our finding is that the
computational emotional frequencies discovered by the proposed approach
provides meaningful emotional features that help RF and ANN achieve a high
precision scores above 93%, on average.

    

### [[2112.02731] Detecting DeFi Securities Violations from Token Smart Contract Code with Random Forest Classification](http://arxiv.org/abs/2112.02731)


  Decentralized Finance (DeFi) is a system of financial products and services
built and delivered through smart contracts on various blockchains. In the past
year, DeFi has gained popularity and market capitalization. However, it has
also become an epicenter of cryptocurrency-related crime, in particular,
various types of securities violations. The lack of Know Your Customer
requirements in DeFi has left governments unsure of how to handle the magnitude
of offending in this space. This study aims to address this problem with a
machine learning approach to identify DeFi projects potentially engaging in
securities violations based on their tokens' smart contract code. We adapt
prior work on detecting specific types of securities violations across Ethereum
more broadly, building a random forest classifier based on features extracted
from DeFi projects' tokens' smart contract code. The final classifier achieves
a 99.1% F1-score. Such high performance is surprising for any classification
problem, however, from further feature-level, we find a single feature makes
this a highly detectable problem. Another contribution of our study is a new
dataset, comprised of (a) a verified ground truth dataset for tokens involved
in securities violations and (b) a set of valid tokens from a DeFi aggregator
which conducts due diligence on the projects it lists. This paper further
discusses the use of our model by prosecutors in enforcement efforts and
connects its potential use to the wider legal context.

    

### [[2112.02736] CDGNet: A Cross-Time Dynamic Graph-based Deep Learning Model for Traffic Forecasting](http://arxiv.org/abs/2112.02736)


  Traffic forecasting is important in intelligent transportation systems of
webs and beneficial to traffic safety, yet is very challenging because of the
complex and dynamic spatio-temporal dependencies in real-world traffic systems.
Prior methods use the pre-defined or learnable static graph to extract spatial
correlations. However, the static graph-based methods fail to mine the
evolution of the traffic network. Researchers subsequently generate the dynamic
graph for each time slice to reflect the changes of spatial correlations, but
they follow the paradigm of independently modeling spatio-temporal
dependencies, ignoring the cross-time spatial influence. In this paper, we
propose a novel cross-time dynamic graph-based deep learning model, named
CDGNet, for traffic forecasting. The model is able to effectively capture the
cross-time spatial dependence between each time slice and its historical time
slices by utilizing the cross-time dynamic graph. Meanwhile, we design a gating
mechanism to sparse the cross-time dynamic graph, which conforms to the sparse
spatial correlations in the real world. Besides, we propose a novel
encoder-decoder architecture to incorporate the cross-time dynamic graph-based
GCN for multi-step traffic forecasting. Experimental results on three
real-world public traffic datasets demonstrate that CDGNet outperforms the
state-of-the-art baselines. We additionally provide a qualitative study to
analyze the effectiveness of our architecture.

    

### [[2112.02740] STformer: A Noise-Aware Efficient Spatio-Temporal Transformer Architecture for Traffic Forecasting](http://arxiv.org/abs/2112.02740)


  Traffic forecasting plays an indispensable role in the intelligent
transportation system, which makes daily travel more convenient and safer.
However, the dynamic evolution of spatio-temporal correlations makes accurate
traffic forecasting very difficult. Existing work mainly employs graph neural
netwroks (GNNs) and deep time series models (e.g., recurrent neural networks)
to capture complex spatio-temporal patterns in the dynamic traffic system. For
the spatial patterns, it is difficult for GNNs to extract the global spatial
information, i.e., remote sensors information in road networks. Although we can
use the self-attention to extract global spatial information as in the previous
work, it is also accompanied by huge resource consumption. For the temporal
patterns, traffic data have not only easy-to-recognize daily and weekly trends
but also difficult-to-recognize short-term noise caused by accidents (e.g., car
accidents and thunderstorms). Prior traffic models are difficult to distinguish
intricate temporal patterns in time series and thus hard to get accurate
temporal dependence. To address above issues, we propose a novel noise-aware
efficient spatio-temporal Transformer architecture for accurate traffic
forecasting, named STformer. STformer consists of two components, which are the
noise-aware temporal self-attention (NATSA) and the graph-based sparse spatial
self-attention (GBS3A). NATSA separates the high-frequency component and the
low-frequency component from the time series to remove noise and capture stable
temporal dependence by the learnable filter and the temporal self-attention,
respectively. GBS3A replaces the full query in vanilla self-attention with the
graph-based sparse query to decrease the time and memory usage. Experiments on
four real-world traffic datasets show that STformer outperforms
state-of-the-art baselines with lower computational cost.

    

### [[2112.02741] Team Hitachi @ AutoMin 2021: Reference-free Automatic Minuting Pipeline with Argument Structure Construction over Topic-based Summarization](http://arxiv.org/abs/2112.02741)


  This paper introduces the proposed automatic minuting system of the Hitachi
team for the First Shared Task on Automatic Minuting (AutoMin-2021). We utilize
a reference-free approach (i.e., without using training minutes) for automatic
minuting (Task A), which first splits a transcript into blocks on the basis of
topics and subsequently summarizes those blocks with a pre-trained BART model
fine-tuned on a summarization corpus of chat dialogue. In addition, we apply a
technique of argument mining to the generated minutes, reorganizing them in a
well-structured and coherent way. We utilize multiple relevance scores to
determine whether or not a minute is derived from the same meeting when either
a transcript or another minute is given (Task B and C). On top of those scores,
we train a conventional machine learning model to bind them and to make final
decisions. Consequently, our approach for Task A achieve the best adequacy
score among all submissions and close performance to the best system in terms
of grammatical correctness and fluency. For Task B and C, the proposed model
successfully outperformed a majority vote baseline.

    

### [[2112.02743] Separated Contrastive Learning for Organ-at-Risk and Gross-Tumor-Volume Segmentation with Limited Annotation](http://arxiv.org/abs/2112.02743)


  Automatic delineation of organ-at-risk (OAR) and gross-tumor-volume (GTV) is
of great significance for radiotherapy planning. However, it is a challenging
task to learn powerful representations for accurate delineation under limited
pixel (voxel)-wise annotations. Contrastive learning at pixel-level can
alleviate the dependency on annotations by learning dense representations from
unlabeled data. Recent studies in this direction design various contrastive
losses on the feature maps, to yield discriminative features for each pixel in
the map. However, pixels in the same map inevitably share semantics to be
closer than they actually are, which may affect the discrimination of pixels in
the same map and lead to the unfair comparison to pixels in other maps. To
address these issues, we propose a separated region-level contrastive learning
scheme, namely SepaReg, the core of which is to separate each image into
regions and encode each region separately. Specifically, SepaReg comprises two
components: a structure-aware image separation (SIS) module and an intra- and
inter-organ distillation (IID) module. The SIS is proposed to operate on the
image set to rebuild a region set under the guidance of structural information.
The inter-organ representation will be learned from this set via typical
contrastive losses cross regions. On the other hand, the IID is proposed to
tackle the quantity imbalance in the region set as tiny organs may produce
fewer regions, by exploiting intra-organ representations. We conducted
extensive experiments to evaluate the proposed model on a public dataset and
two private datasets. The experimental results demonstrate the effectiveness of
the proposed model, consistently achieving better performance than
state-of-the-art approaches. Code is available at
this https URL.

    

### [[2112.02746] Unfairness Despite Awareness: Group-Fair Classification with Strategic Agents](http://arxiv.org/abs/2112.02746)


  The use of algorithmic decision making systems in domains which impact the
financial, social, and political well-being of people has created a demand for
these decision making systems to be "fair" under some accepted notion of
equity. This demand has in turn inspired a large body of work focused on the
development of fair learning algorithms which are then used in lieu of their
conventional counterparts. Most analysis of such fair algorithms proceeds from
the assumption that the people affected by the algorithmic decisions are
represented as immutable feature vectors. However, strategic agents may possess
both the ability and the incentive to manipulate this observed feature vector
in order to attain a more favorable outcome. We explore the impact that
strategic agent behavior could have on fair classifiers and derive conditions
under which this behavior leads to fair classifiers becoming less fair than
their conventional counterparts under the same measure of fairness that the
fair classifier takes into account. These conditions are related to the the way
in which the fair classifier remedies unfairness on the original unmanipulated
data: fair classifiers which remedy unfairness by becoming more selective than
their conventional counterparts are the ones that become less fair than their
counterparts when agents are strategic. We further demonstrate that both the
increased selectiveness of the fair classifier, and consequently the loss of
fairness, arises when performing fair learning on domains in which the
advantaged group is overrepresented in the region near (and on the beneficial
side of) the decision boundary of conventional classifiers. Finally, we observe
experimentally, using several datasets and learning methods, that this fairness
reversal is common, and that our theoretical characterization of the fairness
reversal conditions indeed holds in most such cases.

    

### [[2112.02752] End-to-end Adaptive Distributed Training on PaddlePaddle](http://arxiv.org/abs/2112.02752)


  Distributed training has become a pervasive and effective approach for
training a large neural network (NN) model with processing massive data.
However, it is very challenging to satisfy requirements from various NN models,
diverse computing resources, and their dynamic changes during a training job.
In this study, we design our distributed training framework in a systematic
end-to-end view to provide the built-in adaptive ability for different
scenarios, especially for industrial applications and production environments,
by fully considering resource allocation, model partition, task placement, and
distributed execution. Based on the unified distributed graph and the unified
cluster object, our adaptive framework is equipped with a global cost model and
a global planner, which can enable arbitrary parallelism, resource-aware
placement, multi-mode execution, fault-tolerant, and elastic distributed
training. The experiments demonstrate that our framework can satisfy various
requirements from the diversity of applications and the heterogeneity of
resources with highly competitive performance. The ERNIE language model with
260 billion parameters is efficiently trained on thousands of AI processors
with 91.7% weak scalability. The throughput of the model from the recommender
system by employing the heterogeneous pipeline asynchronous execution can be
increased up to 2.1 times and 3.3 times that of the GPU-only and CPU-only
training respectively. Moreover, the fault-tolerant and elastic distributed
training have been successfully applied to the online industrial applications,
which give a reduction of 34.49% in the number of failed long-term training
jobs and an increase of 33.91% for the global scheduling efficiency in the
production environment.

    

### [[2112.02761] BCD Nets: Scalable Variational Approaches for Bayesian Causal Discovery](http://arxiv.org/abs/2112.02761)


  A structural equation model (SEM) is an effective framework to reason over
causal relationships represented via a directed acyclic graph (DAG). Recent
advances have enabled effective maximum-likelihood point estimation of DAGs
from observational data. However, a point estimate may not accurately capture
the uncertainty in inferring the underlying graph in practical scenarios,
wherein the true DAG is non-identifiable and/or the observed dataset is
limited. We propose Bayesian Causal Discovery Nets (BCD Nets), a variational
inference framework for estimating a distribution over DAGs characterizing a
linear-Gaussian SEM. Developing a full Bayesian posterior over DAGs is
challenging due to the the discrete and combinatorial nature of graphs. We
analyse key design choices for scalable VI over DAGs, such as 1) the
parametrization of DAGs via an expressive variational family, 2) a continuous
relaxation that enables low-variance stochastic optimization, and 3) suitable
priors over the latent variables. We provide a series of experiments on real
and synthetic data showing that BCD Nets outperform maximum-likelihood methods
on standard causal discovery metrics such as structural Hamming distance in low
data regimes.

    

### [[2112.02792] Incentive Compatible Pareto Alignment for Multi-Source Large Graphs](http://arxiv.org/abs/2112.02792)


  In this paper, we focus on learning effective entity matching models over
multi-source large-scale data. For real applications, we relax typical
assumptions that data distributions/spaces, or entity identities are shared
between sources, and propose a Relaxed Multi-source Large-scale Entity-matching
(RMLE) problem. Challenges of the problem include 1) how to align large-scale
entities between sources to share information and 2) how to mitigate negative
transfer from joint learning multi-source data. What's worse, one practical
issue is the entanglement between both challenges. Specifically, incorrect
alignments may increase negative transfer; while mitigating negative transfer
for one source may result in poorly learned representations for other sources
and then decrease alignment accuracy. To handle the entangled challenges, we
point out that the key is to optimize information sharing first based on Pareto
front optimization, by showing that information sharing significantly
influences the Pareto front which depicts lower bounds of negative transfer.
Consequently, we proposed an Incentive Compatible Pareto Alignment (ICPA)
method to first optimize cross-source alignments based on Pareto front
optimization, then mitigate negative transfer constrained on the optimized
alignments. This mechanism renders each source can learn based on its true
preference without worrying about deteriorating representations of other
sources. Specifically, the Pareto front optimization encourages minimizing
lower bounds of negative transfer, which optimizes whether and which to align.
Comprehensive empirical evaluation results on four large-scale datasets are
provided to demonstrate the effectiveness and superiority of ICPA. Online A/B
test results at a search advertising platform also demonstrate the
effectiveness of ICPA in production environments.

    

### [[2112.02796] Conditional Deep Hierarchical Variational Autoencoder for Voice Conversion](http://arxiv.org/abs/2112.02796)


  Variational autoencoder-based voice conversion (VAE-VC) has the advantage of
requiring only pairs of speeches and speaker labels for training. Unlike the
majority of the research in VAE-VC which focuses on utilizing auxiliary losses
or discretizing latent variables, this paper investigates how an increasing
model expressiveness has benefits and impacts on the VAE-VC. Specifically, we
first analyze VAE-VC from a rate-distortion perspective, and point out that
model expressiveness is significant for VAE-VC because rate and distortion
reflect similarity and naturalness of converted speeches. Based on the
analysis, we propose a novel VC method using a deep hierarchical VAE, which has
high model expressiveness as well as having fast conversion speed thanks to its
non-autoregressive decoder. Also, our analysis reveals another problem that
similarity can be degraded when the latent variable of VAEs has redundant
information. We address the problem by controlling the information contained in
the latent variable using $\beta$-VAE objective. In the experiment using VCTK
corpus, the proposed method achieved mean opinion scores higher than 3.5 on
both naturalness and similarity in inter-gender settings, which are higher than
the scores of existing autoencoder-based VC methods.

    

### [[2112.02797] ML Attack Models: Adversarial Attacks and Data Poisoning Attacks](http://arxiv.org/abs/2112.02797)


  Many state-of-the-art ML models have outperformed humans in various tasks
such as image classification. With such outstanding performance, ML models are
widely used today. However, the existence of adversarial attacks and data
poisoning attacks really questions the robustness of ML models. For instance,
Engstrom et al. demonstrated that state-of-the-art image classifiers could be
easily fooled by a small rotation on an arbitrary image. As ML systems are
being increasingly integrated into safety and security-sensitive applications,
adversarial attacks and data poisoning attacks pose a considerable threat. This
chapter focuses on the two broad and important areas of ML security:
adversarial attacks and data poisoning attacks.

    

### [[1901.11311] New Tricks for Estimating Gradients of Expectations](http://arxiv.org/abs/1901.11311)


  We introduce a family of pairwise stochastic gradient estimators for
gradients of expectations, which are related to the log-derivative trick, but
involve pairwise interactions between samples. The simplest example of our new
estimator, dubbed the fundamental trick estimator, is shown to arise from
either a) introducing and approximating an integral representation based on the
fundamental theorem of calculus, or b) applying the reparameterisation trick to
an implicit parameterisation under infinitesimal perturbation of the
parameters. From the former perspective we generalise to a reproducing kernel
Hilbert space representation, giving rise to a locality parameter in the
pairwise interactions mentioned above, yielding our representer trick
estimator. The resulting estimators are unbiased and shown to offer an
independent component of useful information in comparison with the
log-derivative estimator. We provide a further novel theoretical analysis which
further characterises the variance reduction afforded by the new techniques.
Promising analytical and numerical examples confirm the theory and intuitions
behind the new estimators.

    

### [[1905.11374] The Stability and Accuracy Tradeoff Under Dataset Shift: A Causal Graphical Analysis](http://arxiv.org/abs/1905.11374)


  Recent interest in dataset shift has produced many methods for finding
invariant distributions for prediction in new, unseen environments. However,
these methods consider different types of shifts and have been developed under
disparate frameworks, making it difficult to theoretically analyze how
solutions differ with respect to stability and accuracy. Taking a causal
graphical view, we use a flexible graphical representation to express various
types of dataset shifts. We show that all invariant distributions correspond to
a causal hierarchy of graphical operators which disable the edges in the graph
that are responsible for the shifts. The hierarchy provides a common
theoretical underpinning for understanding when and how stability to shifts can
be achieved, and in what ways stable distributions can differ. We use it to
establish conditions for minimax optimal performance across environments, and
derive new algorithms that find optimal stable distributions. Using this new
perspective, we empirically demonstrate that that there is a tradeoff between
minimax and average performance.

    

### [[1906.08530] Bounding the error of discretized Langevin algorithms for non-strongly log-concave targets](http://arxiv.org/abs/1906.08530)


  In this paper, we provide non-asymptotic upper bounds on the error of
sampling from a target density using three schemes of discretized Langevin
diffusions. The first scheme is the Langevin Monte Carlo (LMC) algorithm, the
Euler discretization of the Langevin diffusion. The second and the third
schemes are, respectively, the kinetic Langevin Monte Carlo (KLMC) for
differentiable potentials and the kinetic Langevin Monte Carlo for
twice-differentiable potentials (KLMC2). The main focus is on the target
densities that are smooth and log-concave on $\mathbb R^p$, but not necessarily
strongly log-concave. Bounds on the computational complexity are obtained under
two types of smoothness assumption: the potential has a Lipschitz-continuous
gradient and the potential has a Lipschitz-continuous Hessian matrix. The error
of sampling is measured by Wasserstein-$q$ distances. We advocate for the use
of a new dimension-adapted scaling in the definition of the computational
complexity, when Wasserstein-$q$ distances are considered. The obtained results
show that the number of iterations to achieve a scaled-error smaller than a
prescribed value depends only polynomially in the dimension.

    

### [[1907.09693] A Survey on Federated Learning Systems: Vision, Hype and Reality for Data Privacy and Protection](http://arxiv.org/abs/1907.09693)


  Federated learning has been a hot research topic in enabling the
collaborative training of machine learning models among different organizations
under the privacy restrictions. As researchers try to support more machine
learning models with different privacy-preserving approaches, there is a
requirement in developing systems and infrastructures to ease the development
of various federated learning algorithms. Similar to deep learning systems such
as PyTorch and TensorFlow that boost the development of deep learning,
federated learning systems (FLSs) are equivalently important, and face
challenges from various aspects such as effectiveness, efficiency, and privacy.
In this survey, we conduct a comprehensive review on federated learning
systems. To achieve smooth flow and guide future research, we introduce the
definition of federated learning systems and analyze the system components.
Moreover, we provide a thorough categorization for federated learning systems
according to six different aspects, including data distribution, machine
learning model, privacy mechanism, communication architecture, scale of
federation and motivation of federation. The categorization can help the design
of federated learning systems as shown in our case studies. By systematically
summarizing the existing federated learning systems, we present the design
factors, case studies, and future research opportunities.

    

### [[1909.03245] Regularized Anderson Acceleration for Off-Policy Deep Reinforcement Learning](http://arxiv.org/abs/1909.03245)


  Model-free deep reinforcement learning (RL) algorithms have been widely used
for a range of complex control tasks. However, slow convergence and sample
inefficiency remain challenging problems in RL, especially when handling
continuous and high-dimensional state spaces. To tackle this problem, we
propose a general acceleration method for model-free, off-policy deep RL
algorithms by drawing the idea underlying regularized Anderson acceleration
(RAA), which is an effective approach to accelerating the solving of fixed
point problems with perturbations. Specifically, we first explain how policy
iteration can be applied directly with Anderson acceleration. Then we extend
RAA to the case of deep RL by introducing a regularization term to control the
impact of perturbation induced by function approximation errors. We further
propose two strategies, i.e., progressive update and adaptive restart, to
enhance the performance. The effectiveness of our method is evaluated on a
variety of benchmark tasks, including Atari 2600 and MuJoCo. Experimental
results show that our approach substantially improves both the learning speed
and final performance of state-of-the-art deep RL algorithms.

    

### [[1911.04415] Revisiting the Approximate Carathéodory Problem via the Frank-Wolfe Algorithm](http://arxiv.org/abs/1911.04415)


  The approximate Carathéodory theorem states that given a compact convex set
$\mathcal{C}\subset\mathbb{R}^n$ and $p\in\left[2,+\infty\right[$, each point
$x^*\in\mathcal{C}$ can be approximated to $\epsilon$-accuracy in the
$\ell_p$-norm as the convex combination of $\mathcal{O}(pD_p^2/\epsilon^2)$
vertices of $\mathcal{C}$, where $D_p$ is the diameter of $\mathcal{C}$ in the
$\ell_p$-norm. A solution satisfying these properties can be built using
probabilistic arguments or by applying mirror descent to the dual problem. We
revisit the approximate Carathéodory problem by solving the primal problem
via the Frank-Wolfe algorithm, providing a simplified analysis and leading to
an efficient practical method. Furthermore, improved cardinality bounds are
derived naturally using existing convergence rates of the Frank-Wolfe algorithm
in different scenarios, when $x^*$ is in the interior of $\mathcal{C}$, when
$x^*$ is the convex combination of a subset of vertices with small diameter, or
when $\mathcal{C}$ is uniformly convex. We also propose cardinality bounds when
$p\in\left[1,2\right[\cup\{+\infty\}$ via a nonsmooth variant of the algorithm.
Lastly, we address the problem of finding sparse approximate projections onto
$\mathcal{C}$ in the $\ell_p$-norm, $p\in\left[1,+\infty\right]$.

    

### [[2005.01117] Multi-agent Reinforcement Learning for Decentralized Stable Matching](http://arxiv.org/abs/2005.01117)


  In the real world, people/entities usually find matches independently and
autonomously, such as finding jobs, partners, roommates, etc. It is possible
that this search for matches starts with no initial knowledge of the
environment. We propose the use of a multi-agent reinforcement learning (MARL)
paradigm for a spatially formulated decentralized two-sided matching market
with independent and autonomous agents. Having autonomous agents acting
independently makes our environment very dynamic and uncertain. Moreover,
agents lack the knowledge of preferences of other agents and have to explore
the environment and interact with other agents to discover their own
preferences through noisy rewards. We think such a setting better approximates
the real world and we study the usefulness of our MARL approach for it. Along
with conventional stable matching case where agents have strictly ordered
preferences, we check the applicability of our approach for stable matching
with incomplete lists and ties. We investigate our results for stability, level
of instability (for unstable results), and fairness. Our MARL approach mostly
yields stable and fair outcomes.

    

### [[2005.07385] Enhancing Lattice-based Motion Planning with Introspective Learning and Reasoning](http://arxiv.org/abs/2005.07385)


  Lattice-based motion planning is a hybrid planning method where a plan made
up of discrete actions simultaneously is a physically feasible trajectory. The
planning takes both discrete and continuous aspects into account, for example
action pre-conditions and collision-free action-duration in the configuration
space. Safe motion planing rely on well-calibrated safety-margins for collision
checking. The trajectory tracking controller must further be able to reliably
execute the motions within this safety margin for the execution to be safe. In
this work we are concerned with introspective learning and reasoning about
controller performance over time. Normal controller execution of the different
actions is learned using reliable and uncertainty-aware machine learning
techniques. By correcting for execution bias we manage to substantially reduce
the safety margin of motion actions. Reasoning takes place to both verify that
the learned models stays safe and to improve collision checking effectiveness
in the motion planner by the use of more accurate execution predictions with a
smaller safety margin. The presented approach allows for explicit awareness of
controller performance under normal circumstances, and timely detection of
incorrect performance in abnormal circumstances. Evaluation is made on the
nonlinear dynamics of a quadcopter in 3D using simulation. Video:
this https URL


### [[2006.07540] MetaPerturb: Transferable Regularizer for Heterogeneous Tasks and Architectures](http://arxiv.org/abs/2006.07540)


  Regularization and transfer learning are two popular techniques to enhance
generalization on unseen data, which is a fundamental problem of machine
learning. Regularization techniques are versatile, as they are task- and
architecture-agnostic, but they do not exploit a large amount of data
available. Transfer learning methods learn to transfer knowledge from one
domain to another, but may not generalize across tasks and architectures, and
may introduce new training cost for adapting to the target task. To bridge the
gap between the two, we propose a transferable perturbation, MetaPerturb, which
is meta-learned to improve generalization performance on unseen data.
MetaPerturb is implemented as a set-based lightweight network that is agnostic
to the size and the order of the input, which is shared across the layers.
Then, we propose a meta-learning framework, to jointly train the perturbation
function over heterogeneous tasks in parallel. As MetaPerturb is a set-function
trained over diverse distributions across layers and tasks, it can generalize
to heterogeneous tasks and architectures. We validate the efficacy and
generality of MetaPerturb trained on a specific source domain and architecture,
by applying it to the training of diverse neural architectures on heterogeneous
target datasets against various regularizers and fine-tuning. The results show
that the networks trained with MetaPerturb significantly outperform the
baselines on most of the tasks and architectures, with a negligible increase in
the parameter size and no hyperparameters to tune.

    

### [[2006.11911] Towards Tractable Optimism in Model-Based Reinforcement Learning](http://arxiv.org/abs/2006.11911)


  The principle of optimism in the face of uncertainty is prevalent throughout
sequential decision making problems such as multi-armed bandits and
reinforcement learning (RL). To be successful, an optimistic RL algorithm must
over-estimate the true value function (optimism) but not by so much that it is
inaccurate (estimation error). In the tabular setting, many state-of-the-art
methods produce the required optimism through approaches which are intractable
when scaling to deep RL. We re-interpret these scalable optimistic model-based
algorithms as solving a tractable noise augmented MDP. This formulation
achieves a competitive regret bound: $\tilde{\mathcal{O}}(
|\mathcal{S}|H\sqrt{|\mathcal{A}| T } )$ when augmenting using Gaussian noise,
where $T$ is the total number of environment steps. We also explore how this
trade-off changes in the deep RL setting, where we show empirically that
estimation error is significantly more troublesome. However, we also show that
if this error is reduced, optimistic model-based RL algorithms can match
state-of-the-art performance in continuous control problems.

    

### [[2007.02191] Coded Distributed Computing with Partial Recovery](http://arxiv.org/abs/2007.02191)


  Coded computation techniques provide robustness against straggling workers in
distributed computing. However, most of the existing schemes require exact
provisioning of the straggling behaviour and ignore the computations carried
out by straggling workers. Moreover, these schemes are typically designed to
recover the desired computation results accurately, while in many machine
learning and iterative optimization algorithms, faster approximate solutions
are known to result in an improvement in the overall convergence time. In this
paper, we first introduce a novel coded matrix-vector multiplication scheme,
called coded computation with partial recovery (CCPR), which benefits from the
advantages of both coded and uncoded computation schemes, and reduces both the
computation time and the decoding complexity by allowing a trade-off between
the accuracy and the speed of computation. We then extend this approach to
distributed implementation of more general computation tasks by proposing a
coded communication scheme with partial recovery, where the results of subtasks
computed by the workers are coded before being communicated. Numerical
simulations on a large linear regression task confirm the benefits of the
proposed distributed computation scheme with partial recovery in terms of the
trade-off between the computation accuracy and latency.

    

### [[2007.09654] Distribution-Balanced Loss for Multi-Label Classification in Long-Tailed Datasets](http://arxiv.org/abs/2007.09654)


  We present a new loss function called Distribution-Balanced Loss for the
multi-label recognition problems that exhibit long-tailed class distributions.
Compared to conventional single-label classification problem, multi-label
recognition problems are often more challenging due to two significant issues,
namely the co-occurrence of labels and the dominance of negative labels (when
treated as multiple binary classification problems). The Distribution-Balanced
Loss tackles these issues through two key modifications to the standard binary
cross-entropy loss: 1) a new way to re-balance the weights that takes into
account the impact caused by label co-occurrence, and 2) a negative tolerant
regularization to mitigate the over-suppression of negative labels. Experiments
on both Pascal VOC and COCO show that the models trained with this new loss
function achieve significant performance gains over existing methods. Code and
models are available at: this https URL .

    

### [[2007.14321] Label-Only Membership Inference Attacks](http://arxiv.org/abs/2007.14321)


  Membership inference attacks are one of the simplest forms of privacy leakage
for machine learning models: given a data point and model, determine whether
the point was used to train the model. Existing membership inference attacks
exploit models' abnormal confidence when queried on their training data. These
attacks do not apply if the adversary only gets access to models' predicted
labels, without a confidence measure. In this paper, we introduce label-only
membership inference attacks. Instead of relying on confidence scores, our
attacks evaluate the robustness of a model's predicted labels under
perturbations to obtain a fine-grained membership signal. These perturbations
include common data augmentations or adversarial examples. We empirically show
that our label-only membership inference attacks perform on par with prior
attacks that required access to model confidences. We further demonstrate that
label-only attacks break multiple defenses against membership inference attacks
that (implicitly or explicitly) rely on a phenomenon we call confidence
masking. These defenses modify a model's confidence scores in order to thwart
attacks, but leave the model's predicted labels unchanged. Our label-only
attacks demonstrate that confidence-masking is not a viable defense strategy
against membership inference. Finally, we investigate worst-case label-only
attacks, that infer membership for a small number of outlier data points. We
show that label-only attacks also match confidence-based attacks in this
setting. We find that training models with differential privacy and (strong) L2
regularization are the only known defense strategies that successfully prevents
all attacks. This remains true even when the differential privacy budget is too
high to offer meaningful provable guarantees.

    

### [[2008.01935] GloDyNE: Global Topology Preserving Dynamic Network Embedding](http://arxiv.org/abs/2008.01935)


  Learning low-dimensional topological representation of a network in dynamic
environments is attracting much attention due to the time-evolving nature of
many real-world networks. The main and common objective of Dynamic Network
Embedding (DNE) is to efficiently update node embeddings while preserving
network topology at each time step. The idea of most existing DNE methods is to
capture the topological changes at or around the most affected nodes (instead
of all nodes) and accordingly update node embeddings. Unfortunately, this kind
of approximation, although can improve efficiency, cannot effectively preserve
the global topology of a dynamic network at each time step, due to not
considering the inactive sub-networks that receive accumulated topological
changes propagated via the high-order proximity. To tackle this challenge, we
propose a novel node selecting strategy to diversely select the representative
nodes over a network, which is coordinated with a new incremental learning
paradigm of Skip-Gram based embedding approach. The extensive experiments show
GloDyNE, with a small fraction of nodes being selected, can already achieve the
superior or comparable performance w.r.t. the state-of-the-art DNE methods in
three typical downstream tasks. Particularly, GloDyNE significantly outperforms
other methods in the graph reconstruction task, which demonstrates its ability
of global topology preservation. The source code is available at
this https URL


### [[2008.08637] SODEN: A Scalable Continuous-Time Survival Model through Ordinary Differential Equation Networks](http://arxiv.org/abs/2008.08637)


  In this paper, we propose a flexible model for survival analysis using neural
networks along with scalable optimization algorithms. One key technical
challenge for directly applying maximum likelihood estimation (MLE) to censored
data is that evaluating the objective function and its gradients with respect
to model parameters requires the calculation of integrals. To address this
challenge, we recognize that the MLE for censored data can be viewed as a
differential-equation constrained optimization problem, a novel perspective.
Following this connection, we model the distribution of event time through an
ordinary differential equation and utilize efficient ODE solvers and adjoint
sensitivity analysis to numerically evaluate the likelihood and the gradients.
Using this approach, we are able to 1) provide a broad family of
continuous-time survival distributions without strong structural assumptions,
2) obtain powerful feature representations using neural networks, and 3) allow
efficient estimation of the model in large-scale applications using stochastic
gradient descent. Through both simulation studies and real-world data examples,
we demonstrate the effectiveness of the proposed method in comparison to
existing state-of-the-art deep learning survival analysis models. The
implementation of the proposed SODEN approach has been made publicly available
at this https URL.

    

### [[2008.10498] Noise-induced degeneration in online learning](http://arxiv.org/abs/2008.10498)


  In order to elucidate the plateau phenomena caused by vanishing gradient, we
herein analyse stability of stochastic gradient descent near degenerated
subspaces in a multi-layer perceptron. In stochastic gradient descent for
Fukumizu-Amari model, which is the minimal multi-layer perceptron showing
non-trivial plateau phenomena, we show that (1) attracting regions exist in
multiply degenerated subspaces, (2) a strong plateau phenomenon emerges as a
noise-induced synchronisation, which is not observed in deterministic gradient
descent, (3) an optimal fluctuation exists to minimise the escape time from the
degenerated subspace. The noise-induced degeneration observed herein is
expected to be found in a broad class of machine learning via neural networks.

    

### [[2009.00236] A Survey of Deep Active Learning](http://arxiv.org/abs/2009.00236)


  Active learning (AL) attempts to maximize the performance gain of the model
by marking the fewest samples. Deep learning (DL) is greedy for data and
requires a large amount of data supply to optimize massive parameters, so that
the model learns how to extract high-quality features. In recent years, due to
the rapid development of internet technology, we are in an era of information
torrents and we have massive amounts of data. In this way, DL has aroused
strong interest of researchers and has been rapidly developed. Compared with
DL, researchers have relatively low interest in AL. This is mainly because
before the rise of DL, traditional machine learning requires relatively few
labeled samples. Therefore, early AL is difficult to reflect the value it
deserves. Although DL has made breakthroughs in various fields, most of this
success is due to the publicity of the large number of existing annotation
datasets. However, the acquisition of a large number of high-quality annotated
datasets consumes a lot of manpower, which is not allowed in some fields that
require high expertise, especially in the fields of speech recognition,
information extraction, medical images, etc. Therefore, AL has gradually
received due attention. A natural idea is whether AL can be used to reduce the
cost of sample annotations, while retaining the powerful learning capabilities
of DL. Therefore, deep active learning (DAL) has emerged. Although the related
research has been quite abundant, it lacks a comprehensive survey of DAL. This
article is to fill this gap, we provide a formal classification method for the
existing work, and a comprehensive and systematic overview. In addition, we
also analyzed and summarized the development of DAL from the perspective of
application. Finally, we discussed the confusion and problems in DAL, and gave
some possible development directions for DAL.

    

### [[2009.04131] SoK: Certified Robustness for Deep Neural Networks](http://arxiv.org/abs/2009.04131)


  Great advances in deep neural networks (DNNs) have led to state-of-the-art
performance on a wide range of tasks. However, recent studies have shown that
DNNs are vulnerable to adversarial attacks, which have brought great concerns
when deploying these models to safety-critical applications such as autonomous
driving. Different defense approaches have been proposed against adversarial
attacks, including: a) empirical defenses, which usually can be adaptively
attacked again without providing robustness certification; and b) certifiably
robust approaches which consist of robustness verification providing the lower
bound of robust accuracy against any attacks under certain conditions and
corresponding robust training approaches. In this paper, we systematize the
certifiably robust approaches and related practical and theoretical
implications and findings. We also provide the first comprehensive benchmark on
existing robustness verification and training approaches on different datasets.
In particular, we 1) provide a taxonomy for the robustness verification and
training approaches, as well as summarize the methodologies for representative
algorithms, 2) reveal the characteristics, strengths, limitations, and
fundamental connections among these approaches, 3) discuss current research
progresses, theoretical barriers, main challenges, and future directions for
certifiably robust approaches for DNNs, and 4) provide an open-sourced unified
platform to evaluate over 20 representative certifiably robust approaches for a
wide range of DNNs.

    

### [[2009.06602] VacSIM: Learning Effective Strategies for COVID-19 Vaccine Distribution using Reinforcement Learning](http://arxiv.org/abs/2009.06602)


  A COVID-19 vaccine is our best bet for mitigating the ongoing onslaught of
the pandemic. However, vaccine is also expected to be a limited resource. An
optimal allocation strategy, especially in countries with access inequities and
temporal separation of hot-spots, might be an effective way of halting the
disease spread. We approach this problem by proposing a novel pipeline VacSIM
that dovetails Deep Reinforcement Learning models into a Contextual Bandits
approach for optimizing the distribution of COVID-19 vaccine. Whereas the
Reinforcement Learning models suggest better actions and rewards, Contextual
Bandits allow online modifications that may need to be implemented on a
day-to-day basis in the real world scenario. We evaluate this framework against
a naive allocation approach of distributing vaccine proportional to the
incidence of COVID-19 cases in five different States across India (Assam,
Delhi, Jharkhand, Maharashtra and Nagaland) and demonstrate up to 9039
potential infections prevented and a significant increase in the efficacy of
limiting the spread over a period of 45 days through the VacSIM approach. Our
models and the platform are extensible to all states of India and potentially
across the globe. We also propose novel evaluation strategies including
standard compartmental model-based projections and a causality-preserving
evaluation of our model. Since all models carry assumptions that may need to be
tested in various contexts, we open source our model VacSIM and contribute a
new reinforcement learning environment compatible with OpenAI gym to make it
extensible for real-world applications across the globe.
(this http URL).

    

### [[2009.07961] Decomposition and Adaptive Sampling for Data-Driven Inverse Linear Optimization](http://arxiv.org/abs/2009.07961)


  This work addresses inverse linear optimization where the goal is to infer
the unknown cost vector of a linear program. Specifically, we consider the
data-driven setting in which the available data are noisy observations of
optimal solutions that correspond to different instances of the linear program.
We introduce a new formulation of the problem that, compared to other existing
methods, allows the recovery of a less restrictive and generally more
appropriate admissible set of cost estimates. It can be shown that this inverse
optimization problem yields a finite number of solutions, and we develop an
exact two-phase algorithm to determine all such solutions. Moreover, we propose
an efficient decomposition algorithm to solve large instances of the problem.
The algorithm extends naturally to an online learning environment where it can
be used to provide quick updates of the cost estimate as new data becomes
available over time. For the online setting, we further develop an effective
adaptive sampling strategy that guides the selection of the next samples. The
efficacy of the proposed methods is demonstrated in computational experiments
involving two applications, customer preference learning and cost estimation
for production planning. The results show significant reductions in computation
and sampling efforts.

    

### [[2012.01737] Automatic Routability Predictor Development Using Neural Architecture Search](http://arxiv.org/abs/2012.01737)


  The rise of machine learning technology inspires a boom of its applications
in electronic design automation (EDA) and helps improve the degree of
automation in chip designs. However, manually crafted machine learning models
require extensive human expertise and tremendous engineering efforts. In this
work, we leverage neural architecture search (NAS) to automate the development
of high-quality neural architectures for routability prediction, which can help
to guide cell placement toward routable solutions. Our search method supports
various operations and highly flexible connections, leading to architectures
significantly different from all previous human-crafted models. Experimental
results on a large dataset demonstrate that our automatically generated neural
architectures clearly outperform multiple representative manually crafted
solutions. Compared to the best case of manually crafted models, NAS-generated
models achieve 5.85% higher Kendall's $\tau$ in predicting the number of nets
with DRC violations and 2.12% better area under ROC curve (ROC-AUC) in DRC
hotspot detection. Moreover, compared with human-crafted models, which easily
take weeks to develop, our efficient NAS approach finishes the whole automatic
search process with only 0.3 days.

    

### [[2012.07297] Source Data-absent Unsupervised Domain Adaptation through Hypothesis Transfer and Labeling Transfer](http://arxiv.org/abs/2012.07297)


  Unsupervised domain adaptation (UDA) aims to transfer knowledge from a
related but different well-labeled source domain to a new unlabeled target
domain. Most existing UDA methods require access to the source data, and thus
are not applicable when the data are confidential and not shareable due to
privacy concerns. This paper aims to tackle a realistic setting with only a
classification model available trained over, instead of accessing to, the
source data. To effectively utilize the source model for adaptation, we propose
a novel approach called Source HypOthesis Transfer (SHOT), which learns the
feature extraction module for the target domain by fitting the target data
features to the frozen source classification module (representing
classification hypothesis). Specifically, SHOT exploits both information
maximization and self-supervised learning for the feature extraction module
learning to ensure the target features are implicitly aligned with the features
of unseen source data via the same hypothesis. Furthermore, we propose a new
labeling transfer strategy, which separates the target data into two splits
based on the confidence of predictions (labeling information), and then employ
semi-supervised learning to improve the accuracy of less-confident predictions
in the target domain. We denote labeling transfer as SHOT++ if the predictions
are obtained by SHOT. Extensive experiments on both digit classification and
object recognition tasks show that SHOT and SHOT++ achieve results surpassing
or comparable to the state-of-the-arts, demonstrating the effectiveness of our
approaches for various visual domain adaptation problems. Code is available at
\url{this https URL}.

    

### [[2101.01100] Wasserstein barycenters are NP-hard to compute](http://arxiv.org/abs/2101.01100)


  Computing Wasserstein barycenters (a.k.a. Optimal Transport barycenters) is a
fundamental problem in geometry which has recently attracted considerable
attention due to many applications in data science. While there exist
polynomial-time algorithms in any fixed dimension, all known running times
suffer exponentially in the dimension. It is an open question whether this
exponential dependence is improvable to a polynomial dependence. This paper
proves that unless P=NP, the answer is no. This uncovers a "curse of
dimensionality" for Wasserstein barycenter computation which does not occur for
Optimal Transport computation. Moreover, our hardness results for computing
Wasserstein barycenters extend to approximate computation, to seemingly simple
cases of the problem, and to averaging probability distributions in other
Optimal Transport metrics.

    

### [[2102.07215] Large-Scale Meta-Learning with Continual Trajectory Shifting](http://arxiv.org/abs/2102.07215)


  Meta-learning of shared initialization parameters has shown to be highly
effective in solving few-shot learning tasks. However, extending the framework
to many-shot scenarios, which may further enhance its practicality, has been
relatively overlooked due to the technical difficulties of meta-learning over
long chains of inner-gradient steps. In this paper, we first show that allowing
the meta-learners to take a larger number of inner gradient steps better
captures the structure of heterogeneous and large-scale task distributions,
thus results in obtaining better initialization points. Further, in order to
increase the frequency of meta-updates even with the excessively long
inner-optimization trajectories, we propose to estimate the required shift of
the task-specific parameters with respect to the change of the initialization
parameters. By doing so, we can arbitrarily increase the frequency of
meta-updates and thus greatly improve the meta-level convergence as well as the
quality of the learned initializations. We validate our method on a
heterogeneous set of large-scale tasks and show that the algorithm largely
outperforms the previous first-order meta-learning methods in terms of both
generalization performance and convergence, as well as multi-task learning and
fine-tuning baselines.

    

### [[2102.09225] Continuous Doubly Constrained Batch Reinforcement Learning](http://arxiv.org/abs/2102.09225)


  Reliant on too many experiments to learn good actions, current Reinforcement
Learning (RL) algorithms have limited applicability in real-world settings,
which can be too expensive to allow exploration. We propose an algorithm for
batch RL, where effective policies are learned using only a fixed offline
dataset instead of online interactions with the environment. The limited data
in batch RL produces inherent uncertainty in value estimates of states/actions
that were insufficiently represented in the training data. This leads to
particularly severe extrapolation when our candidate policies diverge from one
that generated the data. We propose to mitigate this issue via two
straightforward penalties: a policy-constraint to reduce this divergence and a
value-constraint that discourages overly optimistic estimates. Over a
comprehensive set of 32 continuous-action batch RL benchmarks, our approach
compares favorably to state-of-the-art methods, regardless of how the offline
data were collected.

    

### [[2103.00139] Scalable Causal Domain Adaptation](http://arxiv.org/abs/2103.00139)


  One of the most critical problems in transfer learning is the task of domain
adaptation, where the goal is to apply an algorithm trained in one or more
source domains to a different (but related) target domain. This paper deals
with domain adaptation in the presence of covariate shift while invariances
exist across domains. One of the main limitations of existing causal inference
methods for solving this problem is scalability. To overcome this difficulty,
we propose SCTL, an algorithm that avoids an exhaustive search and identifies
invariant causal features across source and target domains based on Markov
blanket discovery. SCTL does not require having prior knowledge of the causal
structure, the type of interventions, or the intervention targets. There is an
intrinsic locality associated with SCTL that makes it practically scalable and
robust because local causal discovery increases the power of computational
independence tests and makes the task of domain adaptation computationally
tractable. We show the scalability and robustness of SCTL for domain adaptation
using synthetic and real data sets in low-dimensional and high-dimensional
settings.

    

### [[2103.07678] A review of machine learning in processing remote sensing data for mineral exploration](http://arxiv.org/abs/2103.07678)


  The decline of the number of newly discovered mineral deposits and increase
in demand for different minerals in recent years has led exploration geologists
to look for more efficient and innovative methods for processing different data
types at each stage of mineral exploration. As a primary step, various
features, such as lithological units, alteration types, structures, and
indicator minerals, are mapped to aid decision-making in targeting ore
deposits. Different types of remote sensing datasets, such as satellite and
airborne data, make it possible to overcome common problems associated with
mapping geological features. The rapid increase in the volume of remote sensing
data obtained from different platforms has encouraged scientists to develop
advanced, innovative, and robust data processing methodologies. Machine
learning methods can help process a wide range of remote sensing datasets and
determine the relationship between components such as the reflectance continuum
and features of interest. These methods are robust in processing spectral and
ground truth measurements against noise and uncertainties. In recent years,
many studies have been carried out by supplementing geological surveys with
remote sensing datasets, which is now prominent in geoscience research. This
paper provides a comprehensive review of the implementation and adaptation of
some popular and recently established machine learning methods for processing
different types of remote sensing data and investigates their applications for
detecting various ore deposit types. We demonstrate the high capability of
combining remote sensing data and machine learning methods for mapping
different geological features that are critical for providing potential maps.
Moreover, we find there is scope for advanced methods to process the new
generation of remote sensing data for creating improved mineral prospectivity
maps.

    

### [[2104.00871] A Comparative Analysis of Machine Learning and Grey Models](http://arxiv.org/abs/2104.00871)


  Artificial Intelligence (AI) has recently shown its capabilities for almost
every field of life. Machine Learning, which is a subset of AI, is a `HOT'
topic for researchers. Machine Learning outperforms other classical forecasting
techniques in almost all-natural applications. It is a crucial part of modern
research. As per this statement, Modern Machine Learning algorithms are hungry
for big data. Due to the small datasets, the researchers may not prefer to use
Machine Learning algorithms. To tackle this issue, the main purpose of this
survey is to illustrate, demonstrate related studies for significance of a
semi-parametric Machine Learning framework called Grey Machine Learning (GML).
This kind of framework is capable of handling large datasets as well as small
datasets for time series forecasting likely outcomes. This survey presents a
comprehensive overview of the existing semi-parametric machine learning
techniques for time series forecasting. In this paper, a primer survey on the
GML framework is provided for researchers. To allow an in-depth understanding
for the readers, a brief description of Machine Learning, as well as various
forms of conventional grey forecasting models are discussed. Moreover, a brief
description on the importance of GML framework is presented.

    

### [[2104.11706] Safe Chance Constrained Reinforcement Learning for Batch Process Control](http://arxiv.org/abs/2104.11706)


  Reinforcement Learning (RL) controllers have generated excitement within the
control community. The primary advantage of RL controllers relative to existing
methods is their ability to optimize uncertain systems independently of
explicit assumption of process uncertainty. Recent focus on engineering
applications has been directed towards the development of safe RL controllers.
Previous works have proposed approaches to account for constraint satisfaction
through constraint tightening from the domain of stochastic model predictive
control. Here, we extend these approaches to account for plant-model mismatch.
Specifically, we propose a data-driven approach that utilizes Gaussian
processes for the offline simulation model and use the associated posterior
uncertainty prediction to account for joint chance constraints and plant-model
mismatch. The method is benchmarked against nonlinear model predictive control
via case studies. The results demonstrate the ability of the methodology to
account for process uncertainty, enabling satisfaction of joint chance
constraints even in the presence of plant-model mismatch.

    

### [[2105.01937] FLEX: Parameter-free Multi-view 3D Human Motion Reconstruction](http://arxiv.org/abs/2105.01937)


  The increasing availability of video recordings made by multiple cameras has
offered new means for mitigatingocclusion and depth ambiguities in pose and
motion reconstruction methods. Yet, multi-view algorithms strongly depend on
camera parameters; particularly, the relativepositions between the cameras.
Such a dependency becomes a hurdle once shifting to dynamic capture in
uncontrolled settings. We introduce FLEX (Free muLti-view rEconstruXion), an
end-to-end parameter-free multi-viewmodel. FLEX is parameter-free in the sense
that it does not require any camera parameters, neither intrinsic nor
extrinsic. Our key idea is that the 3D angles between skeletal parts, as well
as bone lengths, are invariant to the camera position. Hence, learning 3D
rotations and bone lengths rather than locations allows predicting common
values for all camera views. Our network takes multiple video streams, learns
fused deep features through a novel multi-view fusion layer, and reconstructs a
single consistent skeleton with temporally coherent joint rotations. We
demonstrate quantitative and qualitative results on the Human3.6M and KTH
Multi-view Football II datasets, and on synthetic multi-person video streams
captured by dynamic cameras. We compare our model to state-of-the-art methods
that are not parameter-free and show that in the absence of camera parameters,
we outperform them by a large margin while obtaining comparable results when
camera parameters are available. Code, trained models, video examples, and more
material will be available on our project page.

    

### [[2105.07581] Vision Transformers are Robust Learners](http://arxiv.org/abs/2105.07581)


  Transformers, composed of multiple self-attention layers, hold strong
promises toward a generic learning primitive applicable to different data
modalities, including the recent breakthroughs in computer vision achieving
state-of-the-art (SOTA) standard accuracy. What remains largely unexplored is
their robustness evaluation and attribution. In this work, we study the
robustness of the Vision Transformer (ViT) against common corruptions and
perturbations, distribution shifts, and natural adversarial examples. We use
six different diverse ImageNet datasets concerning robust classification to
conduct a comprehensive performance comparison of ViT models and SOTA
convolutional neural networks (CNNs), Big-Transfer. Through a series of six
systematically designed experiments, we then present analyses that provide both
quantitative and qualitative indications to explain why ViTs are indeed more
robust learners. For example, with fewer parameters and similar dataset and
pre-training combinations, ViT gives a top-1 accuracy of 28.10% on ImageNet-A
which is 4.3x higher than a comparable variant of BiT. Our analyses on image
masking, Fourier spectrum sensitivity, and spread on discrete cosine energy
spectrum reveal intriguing properties of ViT attributing to improved
robustness. Code for reproducing our experiments is available at
this https URL.

    

### [[2105.07754] A Fusion-Denoising Attack on InstaHide with Data Augmentation](http://arxiv.org/abs/2105.07754)


  InstaHide is a state-of-the-art mechanism for protecting private training
images, by mixing multiple private images and modifying them such that their
visual features are indistinguishable to the naked eye. In recent work,
however, Carlini et al. show that it is possible to reconstruct private images
from the encrypted dataset generated by InstaHide. Nevertheless, we demonstrate
that Carlini et al.'s attack can be easily defeated by incorporating data
augmentation into InstaHide. This leads to a natural question: is InstaHide
with data augmentation secure? In this paper, we provide a negative answer to
this question, by devising an attack for recovering private images from the
outputs of InstaHide even when data augmentation is present. The basic idea is
to use a comparative network to identify encrypted images that are likely to
correspond to the same private image, and then employ a fusion-denoising
network for restoring the private image from the encrypted ones, taking into
account the effects of data augmentation. Extensive experiments demonstrate the
effectiveness of the proposed attack in comparison to Carlini et al.'s attack.

    

### [[2106.00273] Improving the Adversarial Robustness for Speaker Verification by Self-Supervised Learning](http://arxiv.org/abs/2106.00273)


  Previous works have shown that automatic speaker verification (ASV) is
seriously vulnerable to malicious spoofing attacks, such as replay, synthetic
speech, and recently emerged adversarial attacks. Great efforts have been
dedicated to defending ASV against replay and synthetic speech; however, only a
few approaches have been explored to deal with adversarial attacks. All the
existing approaches to tackle adversarial attacks for ASV require the knowledge
for adversarial samples generation, but it is impractical for defenders to know
the exact attack algorithms that are applied by the in-the-wild attackers. This
work is among the first to perform adversarial defense for ASV without knowing
the specific attack algorithms. Inspired by self-supervised learning models
(SSLMs) that possess the merits of alleviating the superficial noise in the
inputs and reconstructing clean samples from the interrupted ones, this work
regards adversarial perturbations as one kind of noise and conducts adversarial
defense for ASV by SSLMs. Specifically, we propose to perform adversarial
defense from two perspectives: 1) adversarial perturbation purification and 2)
adversarial perturbation detection. Experimental results show that our
detection module effectively shields the ASV by detecting adversarial samples
with an accuracy of around 80%. Moreover, since there is no common metric for
evaluating the adversarial defense performance for ASV, this work also
formalizes evaluation metrics for adversarial defense considering both
purification and detection based approaches into account. We sincerely
encourage future works to benchmark their approaches based on the proposed
evaluation framework.

    

### [[2106.02112] Finding and Fixing Spurious Patterns with Explanations](http://arxiv.org/abs/2106.02112)


  Machine learning models often use spurious patterns such as "relying on the
presence of a person to detect a tennis racket," which do not generalize. In
this work, we present an end-to-end pipeline for identifying and mitigating
spurious patterns for image classifiers. We start by finding patterns such as
"the model's prediction for tennis racket changes 63% of the time if we hide
the people." Then, if a pattern is spurious, we mitigate it via a novel form of
data augmentation. We demonstrate that this approach identifies a diverse set
of spurious patterns and that it mitigates them by producing a model that is
both more accurate on a distribution where the spurious pattern is not helpful
and more robust to distribution shift.

    

### [[2106.02694] Efficient Classification of Very Large Images with Tiny Objects](http://arxiv.org/abs/2106.02694)


  An increasing number of applications in computer vision, specially, in
medical imaging and remote sensing, become challenging when the goal is to
classify very large images with tiny informative objects. Specifically, these
classification tasks face two key challenges: $i$) the size of the input image
is usually in the order of mega- or giga-pixels, however, existing deep
architectures do not easily operate on such big images due to memory
constraints, consequently, we seek a memory-efficient method to process these
images; and $ii$) only a very small fraction of the input images are
informative of the label of interest, resulting in low region of interest (ROI)
to image ratio. However, most of the current convolutional neural networks
(CNNs) are designed for image classification datasets that have relatively
large ROIs and small image sizes (sub-megapixel). Existing approaches have
addressed these two challenges in isolation. We present an end-to-end CNN model
termed Zoom-In network that leverages hierarchical attention sampling for
classification of large images with tiny objects using a single GPU. We
evaluate our method on four large-image histopathology, road-scene and
satellite imaging datasets, and one gigapixel pathology dataset. Experimental
results show that our model achieves higher accuracy than existing methods
while requiring less memory resources.

    

### [[2106.03849] SIMONe: View-Invariant, Temporally-Abstracted Object Representations via Unsupervised Video Decomposition](http://arxiv.org/abs/2106.03849)


  To help agents reason about scenes in terms of their building blocks, we wish
to extract the compositional structure of any given scene (in particular, the
configuration and characteristics of objects comprising the scene). This
problem is especially difficult when scene structure needs to be inferred while
also estimating the agent's location/viewpoint, as the two variables jointly
give rise to the agent's observations. We present an unsupervised variational
approach to this problem. Leveraging the shared structure that exists across
different scenes, our model learns to infer two sets of latent representations
from RGB video input alone: a set of "object" latents, corresponding to the
time-invariant, object-level contents of the scene, as well as a set of "frame"
latents, corresponding to global time-varying elements such as viewpoint. This
factorization of latents allows our model, SIMONe, to represent object
attributes in an allocentric manner which does not depend on viewpoint.
Moreover, it allows us to disentangle object dynamics and summarize their
trajectories as time-abstracted, view-invariant, per-object properties. We
demonstrate these capabilities, as well as the model's performance in terms of
view synthesis and instance segmentation, across three procedurally generated
video datasets.

    

### [[2106.04751] Self-Supervised Graph Learning with Hyperbolic Embedding for Temporal Health Event Prediction](http://arxiv.org/abs/2106.04751)


  Electronic Health Records (EHR) have been heavily used in modern healthcare
systems for recording patients' admission information to hospitals. Many
data-driven approaches employ temporal features in EHR for predicting specific
diseases, readmission times, or diagnoses of patients. However, most existing
predictive models cannot fully utilize EHR data, due to an inherent lack of
labels in supervised training for some temporal events. Moreover, it is hard
for existing works to simultaneously provide generic and personalized
interpretability. To address these challenges, we first propose a hyperbolic
embedding method with information flow to pre-train medical code
representations in a hierarchical structure. We incorporate these pre-trained
representations into a graph neural network to detect disease complications,
and design a multi-level attention method to compute the contributions of
particular diseases and admissions, thus enhancing personalized
interpretability. We present a new hierarchy-enhanced historical prediction
proxy task in our self-supervised learning framework to fully utilize EHR data
and exploit medical domain knowledge. We conduct a comprehensive set of
experiments and case studies on widely used publicly available EHR datasets to
verify the effectiveness of our model. The results demonstrate our model's
strengths in both predictive tasks and interpretable abilities.

    

### [[2106.05409] Zero Time Waste: Recycling Predictions in Early Exit Neural Networks](http://arxiv.org/abs/2106.05409)


  The problem of reducing processing time of large deep learning models is a
fundamental challenge in many real-world applications. Early exit methods
strive towards this goal by attaching additional Internal Classifiers (ICs) to
intermediate layers of a neural network. ICs can quickly return predictions for
easy examples and, as a result, reduce the average inference time of the whole
model. However, if a particular IC does not decide to return an answer early,
its predictions are discarded, with its computations effectively being wasted.
To solve this issue, we introduce Zero Time Waste (ZTW), a novel approach in
which each IC reuses predictions returned by its predecessors by (1) adding
direct connections between ICs and (2) combining previous outputs in an
ensemble-like manner. We conduct extensive experiments across various datasets
and architectures to demonstrate that ZTW achieves a significantly better
accuracy vs. inference time trade-off than other recently proposed early exit
methods.

    

### [[2106.05945] Does Knowledge Distillation Really Work?](http://arxiv.org/abs/2106.05945)


  Knowledge distillation is a popular technique for training a small student
network to emulate a larger teacher model, such as an ensemble of networks. We
show that while knowledge distillation can improve student generalization, it
does not typically work as it is commonly understood: there often remains a
surprisingly large discrepancy between the predictive distributions of the
teacher and the student, even in cases when the student has the capacity to
perfectly match the teacher. We identify difficulties in optimization as a key
reason for why the student is unable to match the teacher. We also show how the
details of the dataset used for distillation play a role in how closely the
student matches the teacher -- and that more closely matching the teacher
paradoxically does not always lead to better student generalization.

    

### [[2106.06530] Label Noise SGD Provably Prefers Flat Global Minimizers](http://arxiv.org/abs/2106.06530)


  In overparametrized models, the noise in stochastic gradient descent (SGD)
implicitly regularizes the optimization trajectory and determines which local
minimum SGD converges to. Motivated by empirical studies that demonstrate that
training with noisy labels improves generalization, we study the implicit
regularization effect of SGD with label noise. We show that SGD with label
noise converges to a stationary point of a regularized loss $L(\theta) +\lambda
R(\theta)$, where $L(\theta)$ is the training loss, $\lambda$ is an effective
regularization parameter depending on the step size, strength of the label
noise, and the batch size, and $R(\theta)$ is an explicit regularizer that
penalizes sharp minimizers. Our analysis uncovers an additional regularization
effect of large learning rates beyond the linear scaling rule that penalizes
large eigenvalues of the Hessian more than small ones. We also prove extensions
to classification with general loss functions, SGD with momentum, and SGD with
general noise covariance, significantly strengthening the prior work of Blanc
et al. to global convergence and large learning rates and of HaoChen et al. to
general models.

    

### [[2106.06880] Random Shuffling Beats SGD Only After Many Epochs on Ill-Conditioned Problems](http://arxiv.org/abs/2106.06880)


  Recently, there has been much interest in studying the convergence rates of
without-replacement SGD, and proving that it is faster than with-replacement
SGD in the worst case. However, known lower bounds ignore the problem's
geometry, including its condition number, whereas the upper bounds explicitly
depend on it. Perhaps surprisingly, we prove that when the condition number is
taken into account, without-replacement SGD \emph{does not} significantly
improve on with-replacement SGD in terms of worst-case bounds, unless the
number of epochs (passes over the data) is larger than the condition number.
Since many problems in machine learning and other areas are both
ill-conditioned and involve large datasets, this indicates that
without-replacement does not necessarily improve over with-replacement sampling
for realistic iteration budgets. We show this by providing new lower and upper
bounds which are tight (up to log factors), for quadratic problems with
commuting quadratic terms, precisely quantifying the dependence on the problem
parameters.

    

### [[2106.08970] Sleeper Agent: Scalable Hidden Trigger Backdoors for Neural Networks Trained from Scratch](http://arxiv.org/abs/2106.08970)


  As the curation of data for machine learning becomes increasingly automated,
dataset tampering is a mounting threat. Backdoor attackers tamper with training
data to embed a vulnerability in models that are trained on that data. This
vulnerability is then activated at inference time by placing a "trigger" into
the model's input. Typical backdoor attacks insert the trigger directly into
the training data, although the presence of such an attack may be visible upon
inspection. In contrast, the Hidden Trigger Backdoor Attack achieves poisoning
without placing a trigger into the training data at all. However, this hidden
trigger attack is ineffective at poisoning neural networks trained from
scratch. We develop a new hidden trigger attack, Sleeper Agent, which employs
gradient matching, data selection, and target model re-training during the
crafting process. Sleeper Agent is the first hidden trigger backdoor attack to
be effective against neural networks trained from scratch. We demonstrate its
effectiveness on ImageNet and in black-box settings. Our implementation code
can be found at this https URL.

    

### [[2106.11905] Dangers of Bayesian Model Averaging under Covariate Shift](http://arxiv.org/abs/2106.11905)


  Approximate Bayesian inference for neural networks is considered a robust
alternative to standard training, often providing good performance on
out-of-distribution data. However, Bayesian neural networks (BNNs) with
high-fidelity approximate inference via full-batch Hamiltonian Monte Carlo
achieve poor generalization under covariate shift, even underperforming
classical estimation. We explain this surprising result, showing how a Bayesian
model average can in fact be problematic under covariate shift, particularly in
cases where linear dependencies in the input features cause a lack of posterior
contraction. We additionally show why the same issue does not affect many
approximate inference procedures, or classical maximum a-posteriori (MAP)
training. Finally, we propose novel priors that improve the robustness of BNNs
to many sources of covariate shift.

    

### [[2106.12766] Factors affecting the COVID-19 risk in the US counties: an innovative approach by combining unsupervised and supervised learning](http://arxiv.org/abs/2106.12766)


  The COVID-19 disease spreads swiftly, and nearly three months after the first
positive case was confirmed in China, Coronavirus started to spread all over
the United States. Some states and counties reported high number of positive
cases and deaths, while some reported lower COVID-19 related cases and
mortality. In this paper, the factors that could affect the risk of COVID-19
infection and mortality were analyzed in county level. An innovative method by
using K-means clustering and several classification models is utilized to
determine the most critical factors. Results showed that mean temperature,
percent of people below poverty, percent of adults with obesity, air pressure,
population density, wind speed, longitude, and percent of uninsured people were
the most significant attributes

    

### [[2106.15739] On the Periodic Behavior of Neural Network Training with Batch Normalization and Weight Decay](http://arxiv.org/abs/2106.15739)


  Training neural networks with batch normalization and weight decay has become
a common practice in recent years. In this work, we show that their combined
use may result in a surprising periodic behavior of optimization dynamics: the
training process regularly exhibits destabilizations that, however, do not lead
to complete divergence but cause a new period of training. We rigorously
investigate the mechanism underlying the discovered periodic behavior from both
empirical and theoretical points of view and analyze the conditions in which it
occurs in practice. We also demonstrate that periodic behavior can be regarded
as a generalization of two previously opposing perspectives on training with
batch normalization and weight decay, namely the equilibrium presumption and
the instability presumption.

    

### [[2110.10873] Controllable and Compositional Generation with Latent-Space Energy-Based Models](http://arxiv.org/abs/2110.10873)


  Controllable generation is one of the key requirements for successful
adoption of deep generative models in real-world applications, but it still
remains as a great challenge. In particular, the compositional ability to
generate novel concept combinations is out of reach for most current models. In
this work, we use energy-based models (EBMs) to handle compositional generation
over a set of attributes. To make them scalable to high-resolution image
generation, we introduce an EBM in the latent space of a pre-trained generative
model such as StyleGAN. We propose a novel EBM formulation representing the
joint distribution of data and attributes together, and we show how sampling
from it is formulated as solving an ordinary differential equation (ODE). Given
a pre-trained generator, all we need for controllable generation is to train an
attribute classifier. Sampling with ODEs is done efficiently in the latent
space and is robust to hyperparameters. Thus, our method is simple, fast to
train, and efficient to sample. Experimental results show that our method
outperforms the state-of-the-art in both conditional sampling and sequential
editing. In compositional generation, our method excels at zero-shot generation
of unseen attribute combinations. Also, by composing energy functions with
logical operators, this work is the first to achieve such compositionality in
generating photo-realistic images of resolution 1024x1024. Code is available at
this https URL.

    

### [[2111.14454] TsFeX: Contact Tracing Model using Time Series Feature Extraction and Gradient Boosting](http://arxiv.org/abs/2111.14454)


  With the outbreak of COVID-19 pandemic, a dire need to effectively identify
the individuals who may have come in close-contact to others who have been
infected with COVID-19 has risen. This process of identifying individuals, also
termed as 'Contact tracing', has significant implications for the containment
and control of the spread of this virus. However, manual tracing has proven to
be ineffective calling for automated contact tracing approaches. As such, this
research presents an automated machine learning system for identifying
individuals who may have come in contact with others infected with COVID-19
using sensor data transmitted through handheld devices. This paper describes
the different approaches followed in arriving at an optimal solution model that
effectually predicts whether a person has been in close proximity to an
infected individual using a gradient boosting algorithm and time series feature
extraction.

    

### [[2112.01363] Breaking the Convergence Barrier: Optimization via Fixed-Time Convergent Flows](http://arxiv.org/abs/2112.01363)


  Accelerated gradient methods are the cornerstones of large-scale, data-driven
optimization problems that arise naturally in machine learning and other fields
concerning data analysis. We introduce a gradient-based optimization framework
for achieving acceleration, based on the recently introduced notion of
fixed-time stability of dynamical systems. The method presents itself as a
generalization of simple gradient-based methods suitably scaled to achieve
convergence to the optimizer in a fixed-time, independent of the
initialization. We achieve this by first leveraging a continuous-time framework
for designing fixed-time stable dynamical systems, and later providing a
consistent discretization strategy, such that the equivalent discrete-time
algorithm tracks the optimizer in a practically fixed number of iterations. We
also provide a theoretical analysis of the convergence behavior of the proposed
gradient flows, and their robustness to additive disturbances for a range of
functions obeying strong convexity, strict convexity, and possibly nonconvexity
but satisfying the Polyak-Łojasiewicz inequality. We also show that the
regret bound on the convergence rate is constant by virtue of the fixed-time
convergence. The hyperparameters have intuitive interpretations and can be
tuned to fit the requirements on the desired convergence rates. We validate the
accelerated convergence properties of the proposed schemes on a range of
numerical examples against the state-of-the-art optimization algorithms. Our
work provides insights on developing novel optimization algorithms via
discretization of continuous-time flows.

    

### [[2112.02204] Understanding the Limits of Conventional Hardware Architectures for Deep-Learning](http://arxiv.org/abs/2112.02204)


  Deep learning and hardware for it has garnered immense academic and industry
interest in the past 5 years -- including almost 100 startups, more than $5B of
VC investment -- and a re-relevance of the role of architecture. However, the
state-of-art remains NVIDIA's TensorCore-based systems that provide i)
top-of-line performance, ii) turnkey software stack, and iii) coverage across a
wide-spectrum of DL network styles (DL-architecture in AI parlance). Other
academic and industry efforts have included novel approaches like spatial
dataflow, CGRAs, systolic arrays, blended FPGA LUTs with fixed function units
and more. These have all necessitated their own innovations in architecture,
compiler, and software stack integration. However, none of these have yet
satisfied all the 3 metrics that NVIDIA's TensorCore and software stack
provides, and generally seem to perform worse. In this paper, we systematically
investigate the behavior of DL workloads and imputed needs on
hardware/compiler/software. We show that SIMD/short-vector, caching, and
synchronization in a fairly well-understood multicore chip organization we call
UPCYCLE can achieve day-zero software maturity, and provide big integer factor
speedups over the state-of-art NVIDIA solutions. Compared to an A100, UPCYCLE
at small-batch size is geo-mean 3.8X faster for inference, geo-mean 4.2X faster
at training, while consuming only half the power. Second, the UPCYCLE
architecture requires no new compiler or software stack innovation. Third, it
provides full DL-architecture coverage, and can be instantiated to provide
training-optimized, inference-optimized, or balanced training and inference
systems. Overall, this paper motivates the treatment of software maturity as a
first class design constraint in developing new architectures for DL. This is
achieved by revisiting well understood ideas, upcycling them for future DL
architectures...

    

### [[2112.02229] Efficient FPGA-based ECDSA Verification Engine for Permissioned Blockchains](http://arxiv.org/abs/2112.02229)


  As enterprises embrace blockchain technology, many real-world applications
have been developed and deployed using permissioned blockchain platforms
(access to network is controlled and given to only nodes with known
identities). Such blockchain platforms heavily depend on cryptography to
provide a layer of trust within the network, thus verification of cryptographic
signatures often becomes the bottleneck. The Elliptic Curve Digital Signature
Algorithm (ECDSA) is the most commonly used cryptographic scheme in
permissioned blockchains. In this paper, we propose an efficient implementation
of ECDSA signature verification on an FPGA, in order to improve the performance
of permissioned blockchains that aim to use FPGA-based hardware accelerators.
We propose several optimizations for modular arithmetic (e.g., custom
multipliers and fast modular reduction) and point arithmetic (e.g., reduced
number of point double and addition operations, and optimal width NAF
representation). Based on these optimized modular and point arithmetic modules,
we propose an ECDSA verification engine that can be used by any application for
fast verification of ECDSA signatures. We further optimize our ECDSA
verification engine for Hyperledger Fabric (one of the most widely used
permissioned blockchain platforms) by moving carefully selected operations to a
precomputation block, thus simplifying the critical path of ECDSA signature
verification. From our implementation on Xilinx Alveo U250 accelerator board
with target frequency of 250MHz, our ECDSA verification engine can perform a
single verification in $760\mu s$ resulting in a throughput of 1,315
verifications per second, which is ~2.5x faster than state-of-the-art
FPGA-based implementations. Our Hyperledger Fabric-specific ECDSA engine can
perform a single verification in $368\mu s$ with a throughput of 2,717
verifications per second.

    

### [[2112.02231] IMCRYPTO: An In-Memory Computing Fabric for AES Encryption and Decryption](http://arxiv.org/abs/2112.02231)


  This paper proposes IMCRYPTO, an in-memory computing (IMC) fabric for
accelerating AES encryption and decryption. IMCRYPTO employs a unified
structure to implement encryption and decryption in a single hardware
architecture, with combined (Inv)SubBytes and (Inv)MixColumns steps. Because of
this step-combination, as well as the high parallelism achieved by multiple
units of random-access memory (RAM) and random-access/content-addressable
memory (RA/CAM) arrays, IMCRYPTO achieves high throughput encryption and
decryption without sacrificing area and power consumption. Additionally, due to
the integration of a RISC-V core, IMCRYPTO offers programmability and
flexibility. IMCRYPTO improves the throughput per area by a minimum (maximum)
of 3.3x (223.1x) when compared to previous ASICs/IMC architectures for AES-128
encryption. Projections show added benefit from emerging technologies of up to
5.3x to the area-delay-power product of IMCRYPTO.

    

### [[2112.02263] On the Implementation of Fixed-point Exponential Function for Machine Learning and Signal Processing Accelerators](http://arxiv.org/abs/2112.02263)


  The natural exponential function is widely used in modeling many engineering
and scientific systems. It is also an integral part of many neural network
activation function such as sigmoid, tanh, ELU, RBF etc. Dedicated hardware
accelerator and processors are designed for faster execution of such
applications. Such accelerators can immensely benefit from an optimal
implementation of exponential function. This can be achieved for most
applications with the knowledge that the exponential function for a negative
domain is more widely used than the positive domain. This paper presents an
optimized implementation of exponential function for variable precision fixed
point negative input. The implementation presented here significantly reduces
the number of multipliers and adders. This is further optimized using mixed
world-length implementation for the series expansion. The reduction in area and
power consumption is more than 30% and 50% respectively over previous
equivalent method.

    

### [[2112.02516] Energy-Efficient Deflection-based On-chip Networks: Topology, Routing, Flow Control](http://arxiv.org/abs/2112.02516)


  As the number of cores scale to tens and hundreds, the energy consumption of
routers across various types of on-chip networks in chip muiltiprocessors
(CMPs) increases significantly. A major source of this energy consumption comes
from the input buffers inside Network-on-Chip (NoC) routers, which are
traditionally designed to maximize performance. To mitigate this high energy
cost, many works propose bufferless router designs that utilize deflection
routing to resolve port contention. While this approach is able to maintain
high performance relative to its buffered counterparts at low network traffic,
the bufferless router design suffers performance degradation under high network
load.
In order to maintain high performance and energy efficiency under both low
and high network loads, this chapter discusses critical drawbacks of
traditional bufferless designs and describes recent research works focusing on
two major modifications to improve the overall performance of the traditional
bufferless network-on-chip design. The first modification is a
minimally-buffered design that introduces limited buffering inside critical
parts of the on-chip network in order to reduce the number of deflections. The
second modification is a hierarchical bufferless interconnect design that aims
to further improve performance by limiting the number of hops each packet needs
to travel while in the network. In both approaches, we discuss design tradeoffs
and provide evaluation results based on common CMP configurations with various
network topologies to show the effectiveness of each proposal.

    

### [[2112.02793] Kraken: An Efficient Engine with a Uniform Dataflow for Deep Neural Networks](http://arxiv.org/abs/2112.02793)


  Deep neural networks (DNNs) have been successfully employed in a multitude of
applications with remarkable performance. As such performance is achieved at a
significant computational cost, several embedded applications demand fast and
efficient hardware accelerators for DNNs. Previously proposed application
specific integrated circuit (ASIC) architectures strive to utilize arrays of
hundreds of processing elements (PEs) and reduce power-hungry DRAM accesses
using multiple dataflows requiring complex PE architectures. These consume
significant area and reduce the maximum clock frequency. This paper introduces
the Kraken architecture, which optimally processes the convolutional layers,
fully-connected layers, and matrix products of any DNN through a
hardware-friendly uniform dataflow. This enables maximal data reuse of weights,
inputs, and outputs, with a bare-bones PE design and on-the-fly dynamic
reconfiguration. Kraken, implemented in 65-nm CMOS technology at 400 MHz, packs
672 PEs in 7.3 mm2, with a peak performance of 537.6 Gops. Kraken processes the
convolutional layers of AlexNet, VGG-16, and ResNet-50 at 336.6, 17.5, and 64.2
frames/s, respectively, hence outperforming the state-of-the-art ASIC
architectures in terms of overall performance efficiency, DRAM accesses,
arithmetic intensity, and throughput, with 5.8x more Gops/mm2 and 1.6x more
Gops/W.

    

### [[2112.02197] A Divide-and-Conquer Algorithm for Distributed Optimization on Networks](http://arxiv.org/abs/2112.02197)


  In this paper, we consider networks with topologies described by some
connected undirected graph ${\mathcal{G}}=(V, E)$ and with some agents (fusion
centers) equipped with processing power and local peer-to-peer communication,
and optimization problem $\min_{\boldsymbol x}\big\{F({\boldsymbol
x})=\sum_{i\in V}f_i({\boldsymbol x})\big\}$ with local objective functions
$f_i$ depending only on neighboring variables of the vertex $i\in V$. We
introduce a divide-and-conquer algorithm to solve the above optimization
problem in a distributed and decentralized manner. The proposed
divide-and-conquer algorithm has exponential convergence, its computational
cost is almost linear with respect to the size of the network, and it can be
fully implemented at fusion centers of the network. Our numerical
demonstrations also indicate that the proposed divide-and-conquer algorithm has
superior performance than popular decentralized optimization methods do for the
least squares problem with/without $\ell^1$ penalty.

    

### [[2112.02211] An iterative solver for the HPS discretization applied to three dimensional Helmholtz problems](http://arxiv.org/abs/2112.02211)


  This manuscript presents an efficient solver for the linear system that
arises from the Hierarchical Poincaré-Steklov (HPS) discretization of three
dimensional variable coefficient Helmholtz problems. Previous work on the HPS
method has tied it with a direct solver. This work is the first efficient
iterative solver for the linear system that results from the HPS
discretization. The solution technique utilizes GMRES coupled with an exact
block-Jacobi preconditioner. The construction of the block-Jacobi
preconditioner involves two nested local solves that are accelerated by local
homogenization. The local nature of the discretization and preconditioner
naturally yield matrix-free application of the linear system. A distributed
memory implementation allows the solution technique to tackle problems
approximately $50$ wavelengths in each direction requiring more than a billion
unknowns to get approximately 7 digits of accuracy in less than an hour.
Additional numerical results illustrate the performance of the solution
technique.

    

### [[2112.02267] Container Orchestration Techniques in Cloud and Edge/Fog Computing Environments](http://arxiv.org/abs/2112.02267)


  Currently, due to the advantages of light weight, simple deployment,
multi-environment support, short startup time, scalability, and easy migration,
container technology has been widely used in both cloud and edge/fog computing,
and addresses the problem of device heterogeneity in different computing
environments. On this basis, as one of the most popular container orchestration
and management systems, Kubernetes almost dominates the cloud environment.
However, since it is primarily designed for centralized resource management
scenarios where computing resources are sufficient, the system is unstable in
edge environments due to hardware limitations. Therefore, in order to realize
container orchestration in the cloud and edge/fog hybrid computing environment,
we propose a feasible approach to build a hybrid clustering based on K3s, which
solves the problem that virtual instances in different environments cannot be
connected due to IP addresses. We also propose three design patterns for
deploying the FogBus2 framework into hybrid environments, including 1) Host
Network Mode, 2) Proxy Server, and 3) Environment Variable.

    

### [[2112.02289] Towards Aggregated Asynchronous Checkpointing](http://arxiv.org/abs/2112.02289)


  High-Performance Computing (HPC) applications need to checkpoint massive
amounts of data at scale. Multi-level asynchronous checkpoint runtimes like
VELOC (Very Low Overhead Checkpoint Strategy) are gaining popularity among
application scientists for their ability to leverage fast node-local storage
and flush independently to stable, external storage (e.g., parallel file
systems) in the background. Currently, VELOC adopts a one-file-per-process
flush strategy, which results in a large number of files being written to
external storage, thereby overwhelming metadata servers and making it difficult
to transfer and access checkpoints as a whole. This paper discusses the
viability and challenges of designing aggregation techniques for asynchronous
multi-level checkpointing. To this end we implement and study two aggregation
strategies, their limitations, and propose a new aggregation strategy
specifically for asynchronous multi-level checkpointing.

    

### [[2112.02377] On the stability and performance of the solution of sparse linear systems by partitioned procedures](http://arxiv.org/abs/2112.02377)


  In this paper, we present, evaluate and analyse the performance of parallel
synchronous Jacobi algorithms by different partitioned procedures including
band-row splitting, band-row sparsity pattern splitting and substructuring
splitting, when solving sparse large linear systems. Numerical experiments
performed on a set of academic 3D Laplace equation and on a real gravity
matrices arising from the Chicxulub crater are exhibited, and show the impact
of splitting on parallel synchronous iterations when solving sparse large
linear systems. The numerical results clearly show the interest of
substructuring methods compared to band-row splitting strategies.

    

### [[2112.02405] Invalidation-Based Protocols for Replicated Datastores](http://arxiv.org/abs/2112.02405)


  Distributed in-memory datastores underpin cloud applications that run within
a datacenter and demand high performance, strong consistency, and availability.
A key feature of datastores is data replication. The data are replicated across
servers because a single server often cannot handle the request load.
Replication is also necessary to guarantee that a server or link failure does
not render a portion of the dataset inaccessible. A replication protocol is
responsible for ensuring strong consistency between the replicas of a
datastore, even when faults occur, by determining the actions necessary to
access and manipulate the data. Consequently, a replication protocol also
drives the datastore's performance.
Existing strongly consistent replication protocols deliver fault tolerance
but fall short in terms of performance. Meanwhile, the opposite occurs in the
world of multiprocessors, where data are replicated across the private caches
of different cores. The multiprocessor regime uses invalidations to afford
strongly consistent replication with high performance but neglects fault
tolerance.
Although handling failures in the datacenter is critical for data
availability, we observe that the common operation is fault-free and far
exceeds the operation during faults. In other words, the common operating
environment inside a datacenter closely resembles that of a multiprocessor.
Based on this insight, we draw inspiration from the multiprocessor for
high-performance, strongly consistent replication in the datacenter. The
primary contribution of this thesis is in adapting invalidating protocols to
the nuances of replicated datastores, which include skewed data accesses, fault
tolerance, and distributed transactions.

    

### [[2112.02456] Online Social Welfare Maximization with Spatio-Temporal Resource Mesh for Serverless](http://arxiv.org/abs/2112.02456)


  Serverless computing is leading the way to a simplified and general purpose
programming model for the cloud. A key enabler behind serverless is efficient
load balancing, which routes continuous workloads to appropriate backend
resources. However, current load balancing algorithms implemented in Kubernetes
native serverless platforms are simple heuristics without performance
guarantee. Although policies such as Pod or JFIQ yield asymptotically optimal
mean response time, the information they depend on are usually unavailable. In
addition, dispatching jobs with strict deadlines, fractional workloads, and
maximum parallelism bound to limited resources online is difficult because the
resource allocation decisions for jobs are intertwined. To design an online
load balancing algorithm without assumptions on distributions while maximizing
the social welfare, we construct several pseudo-social welfare functions and
cost functions, where the latter is to estimate the marginal cost for
provisioning services to every newly arrived job based on present resource
surplus. The proposed algorithm, named OnSocMax, works by following the
solutions of several convex pseudo-social welfare maximization problems. It is
proved to be $\alpha$-competitive for some $\alpha$ at least 2. We also
validate OnSocMax with simulations and the results show that it distinctly
outperforms several handcrafted benchmarks.

    

### [[2112.02593] A Taxonomy of Live Migration Management in Cloud Computing](http://arxiv.org/abs/2112.02593)


  Cloud Data Centers have become the backbone infrastructure to provide
services. With the emerging edge computing paradigm, computation and networking
capabilities have been pushed from clouds to the edge to provide computation,
intelligence, networking management with low end-to-end latency. Service
migration across different computing nodes in edge and cloud computing becomes
essential to guarantee the quality of service in the dynamic environment. Many
studies have been conducted on the dynamic resource management involving
migrating Virtual Machines to achieve various objectives, such as load
balancing, consolidation, performance, energy-saving, and disaster recovery.
Some have investigated to improve and predict the performance of single live
migration of VM and container. Recently, several studies service migration in
the edge-centric computing paradigms. However, there is a lack of surveys to
focus on the live migration management in edge and cloud computing
environments. We examine the characteristics of each field and conduct a
migration management-centric taxonomy and survey. We also identify the gap and
research opportunities to guarantee the performance of resource management with
live migrations.

    

### [[1906.01211] Raising the Performance of the Tinker-HP Molecular Modeling Package [Article v1.0]](http://arxiv.org/abs/1906.01211)


  This living paper reviews the present High Performance Computing (HPC)
capabilities of the Tinker-HP molecular modeling package. We focus here on the
reference, double precision, massively parallel molecular dynamics engine
present in Tinker-HP and dedicated to perform large scale simulations. We show
how it can be adapted to recent Intel Central Processing Unit (CPU) petascale
architectures. First, we discuss the new set of Intel Advanced Vector
Extensions 512 (Intel AVX-512) instructions present in recent Intel processors
(e.g., the Intel Xeon Scalable and Intel Xeon Phi 2nd generation processors)
allowing for larger vectorization enhancements. These instructions constitute
the central source of potential computational gains when using the latest
processors, justifying important vectorization efforts for developers. We then
briefly review the organization of the Tinker-HP code and identify the
computational hotspots which require Intel AVX-512 optimization and we propose
a general and optimal strategy to vectorize those particular parts of the code.
We intended to present our optimization strategy in a pedagogical way so it
could benefit to other researchers and students interested in gaining
performances in their own software. Finally we present the performance
enhancements obtained compared to the unoptimized code both sequentially and at
the scaling limit in parallel for classical non-polarizable (CHARMM) and
polarizable force fields (AMOEBA). This paper never ceases to be updated as we
accumulate new data on the associated Github repository between new versions of
this living paper.

    

### [[2112.02125] Can OpenAI Codex and Other Large Language Models Help Us Fix Security Bugs?](http://arxiv.org/abs/2112.02125)


  Human developers can produce code with cybersecurity weaknesses. Can emerging
'smart' code completion tools help repair those weaknesses? In this work, we
examine the use of large language models (LLMs) for code (such as OpenAI's
Codex and AI21's Jurassic J-1) for zero-shot vulnerability repair. We
investigate challenges in the design of prompts that coax LLMs into generating
repaired versions of insecure code. This is difficult due to the numerous ways
to phrase key information -- both semantically and syntactically -- with
natural languages. By performing a large scale study of four commercially
available, black-box, "off-the-shelf" LLMs, as well as a locally-trained model,
on a mix of synthetic, hand-crafted, and real-world security bug scenarios, our
experiments show that LLMs could collectively repair 100% of our synthetically
generated and hand-crafted scenarios, as well as 58% of vulnerabilities in a
selection of historical bugs in real-world open-source projects.

    

### [[2112.02143] CTIN: Robust Contextual Transformer Network for Inertial Navigation](http://arxiv.org/abs/2112.02143)


  Recently, data-driven inertial navigation approaches have demonstrated their
capability of using well-trained neural networks to obtain accurate position
estimates from inertial measurement units (IMU) measurements. In this paper, we
propose a novel robust Contextual Transformer-based network for Inertial
Navigation~(CTIN) to accurately predict velocity and trajectory. To this end,
we first design a ResNet-based encoder enhanced by local and global multi-head
self-attention to capture spatial contextual information from IMU measurements.
Then we fuse these spatial representations with temporal knowledge by
leveraging multi-head attention in the Transformer decoder. Finally, multi-task
learning with uncertainty reduction is leveraged to improve learning efficiency
and prediction accuracy of velocity and trajectory. Through extensive
experiments over a wide range of inertial datasets~(e.g. RIDI, OxIOD, RoNIN,
IDOL, and our own), CTIN is very robust and outperforms state-of-the-art
models.

    

### [[2112.02223] A Game-Theoretic Approach for AI-based Botnet Attack Defence](http://arxiv.org/abs/2112.02223)


  The new generation of botnets leverages Artificial Intelligent (AI)
techniques to conceal the identity of botmasters and the attack intention to
avoid detection. Unfortunately, there has not been an existing assessment tool
capable of evaluating the effectiveness of existing defense strategies against
this kind of AI-based botnet attack. In this paper, we propose a sequential
game theory model that is capable to analyse the details of the potential
strategies botnet attackers and defenders could use to reach Nash Equilibrium
(NE). The utility function is computed under the assumption when the attacker
launches the maximum number of DDoS attacks with the minimum attack cost while
the defender utilises the maximum number of defense strategies with the minimum
defense cost. We conduct a numerical analysis based on a various number of
defense strategies involved on different (simulated) cloud-band sizes in
relation to different attack success rate values. Our experimental results
confirm that the success of defense highly depends on the number of defense
strategies used according to careful evaluation of attack rates.

    

### [[2112.02255] In Search of Ambiguity: A Three-Stage Workflow Design to Clarify Annotation Guidelines for Crowd Workers](http://arxiv.org/abs/2112.02255)


  We propose a novel three-stage FIND-RESOLVE-LABEL workflow for crowdsourced
annotation to reduce ambiguity in task instructions and thus improve annotation
quality. Stage 1 (FIND) asks the crowd to find examples whose correct label
seems ambiguous given task instructions. Workers are also asked to provide a
short tag which describes the ambiguous concept embodied by the specific
instance found. We compare collaborative vs. non-collaborative designs for this
stage. In Stage 2 (RESOLVE), the requester selects one or more of these
ambiguous examples to label (resolving ambiguity). The new label(s) are
automatically injected back into task instructions in order to improve clarity.
Finally, in Stage 3 (LABEL), workers perform the actual annotation using the
revised guidelines with clarifying examples. We compare three designs for using
these examples: examples only, tags only, or both. We report image labeling
experiments over six task designs using Amazon's Mechanical Turk. Results show
improved annotation accuracy and further insights regarding effective design
for crowdsourced annotation tasks.

    

### [[2112.02268] Bridging Pre-trained Models and Downstream Tasks for Source Code Understanding](http://arxiv.org/abs/2112.02268)


  With the great success of pre-trained models, the pretrain-then-finetune
paradigm has been widely adopted on downstream tasks for source code
understanding. However, compared to costly training a large-scale model from
scratch, how to effectively adapt pre-trained models to a new task has not been
fully explored. In this paper, we propose an approach to bridge pre-trained
models and code-related tasks. We exploit semantic-preserving transformation to
enrich downstream data diversity, and help pre-trained models learn semantic
features invariant to these semantically equivalent transformations. Further,
we introduce curriculum learning to organize the transformed data in an
easy-to-hard manner to fine-tune existing pre-trained models.
We apply our approach to a range of pre-trained models, and they
significantly outperform the state-of-the-art models on tasks for source code
understanding, such as algorithm classification, code clone detection, and code
search. Our experiments even show that without heavy pre-training on code data,
natural language pre-trained model RoBERTa fine-tuned with our lightweight
approach could outperform or rival existing code pre-trained models fine-tuned
on the above tasks, such as CodeBERT and GraphCodeBERT. This finding suggests
that there is still much room for improvement in code pre-trained models.

    

### [[2112.02274] Self-supervised Graph Learning for Occasional Group Recommendation](http://arxiv.org/abs/2112.02274)


  We study the problem of recommending items to occasional groups (a.k.a.
cold-start groups), where the occasional groups are formed ad-hoc and
have few or no historical interacted items. Due to the extreme sparsity issue
of the occasional groups' interactions with items, it is difficult to learn
high-quality embeddings for these occasional groups. Despite the recent
advances on Graph Neural Networks (GNNs) incorporate high-order collaborative
signals to alleviate the problem, the high-order cold-start neighbors are not
explicitly considered during the graph convolution in GNNs. This paper proposes
a self-supervised graph learning paradigm, which jointly trains the backbone
GNN model to reconstruct the group/user/item embeddings under the meta-learning
setting, such that it can directly improve the embedding quality and can be
easily adapted to the new occasional groups. To further reduce the impact from
the cold-start neighbors, we incorporate a self-attention-based meta aggregator
to enhance the aggregation ability of each graph convolution step. Besides, we
add a contrastive learning (CL) adapter to explicitly consider the correlations
between the group and non-group members. Experimental results on three public
recommendation datasets show the superiority of our proposed model against the
state-of-the-art group recommendation methods.

    

### [[2112.02275] A Multi-Strategy based Pre-Training Method for Cold-Start Recommendation](http://arxiv.org/abs/2112.02275)


  Cold-start problem is a fundamental challenge for recommendation tasks. The
recent self-supervised learning (SSL) on Graph Neural Networks (GNNs) model,
PT-GNN, pre-trains the GNN model to reconstruct the cold-start embeddings and
has shown great potential for cold-start recommendation. However, due to the
over-smoothing problem, PT-GNN can only capture up to 3-order relation, which
can not provide much useful auxiliary information to depict the target
cold-start user or item. Besides, the embedding reconstruction task only
considers the intra-correlations within the subgraph of users and items, while
ignoring the inter-correlations across different subgraphs. To solve the above
challenges, we propose a multi-strategy based pre-training method for
cold-start recommendation (MPT), which extends PT-GNN from the perspective of
model architecture and pretext tasks to improve the cold-start recommendation
performance. Specifically, in terms of the model architecture, in addition to
the short-range dependencies of users and items captured by the GNN encoder, we
introduce a Transformer encoder to capture long-range dependencies. In terms of
the pretext task, in addition to considering the intra-correlations of users
and items by the embedding reconstruction task, we add embedding contrastive
learning task to capture inter-correlations of users and items. We train the
GNN and Transformer encoders on these pretext tasks under the meta-learning
setting to simulate the real cold-start scenario, making the model easily and
rapidly being adapted to new cold-start users and items. Experiments on three
public recommendation datasets show the superiority of the proposed MPT model
against the vanilla GNN models, the pre-training GNN model on user/item
embedding inference and the recommendation task.

    

### [[2112.02278] Stage Conscious Attention Network (SCAN) : A Demonstration-Conditioned Policy for Few-Shot Imitation](http://arxiv.org/abs/2112.02278)


  In few-shot imitation learning (FSIL), using behavioral cloning (BC) to solve
unseen tasks with few expert demonstrations becomes a popular research
direction. The following capabilities are essential in robotics applications:
(1) Behaving in compound tasks that contain multiple stages. (2) Retrieving
knowledge from few length-variant and misalignment demonstrations. (3) Learning
from a different expert. No previous work can achieve these abilities at the
same time. In this work, we conduct FSIL problem under the union of above
settings and introduce a novel stage conscious attention network (SCAN) to
retrieve knowledge from few demonstrations simultaneously. SCAN uses an
attention module to identify each stage in length-variant demonstrations.
Moreover, it is designed under demonstration-conditioned policy that learns the
relationship between experts and agents. Experiment results show that SCAN can
learn from different experts without fine-tuning and outperform baselines in
complicated compound tasks with explainable visualization.

    

### [[2112.02303] An Annotated Video Dataset for Computing Video Memorability](http://arxiv.org/abs/2112.02303)


  Using a collection of publicly available links to short form video clips of
an average of 6 seconds duration each, 1,275 users manually annotated each
video multiple times to indicate both long-term and short-term memorability of
the videos. The annotations were gathered as part of an online memory game and
measured a participant's ability to recall having seen the video previously
when shown a collection of videos. The recognition tasks were performed on
videos seen within the previous few minutes for short-term memorability and
within the previous 24 to 72 hours for long-term memorability. Data includes
the reaction times for each recognition of each video. Associated with each
video are text descriptions (captions) as well as a collection of image-level
features applied to 3 frames extracted from each video (start, middle and end).
Video-level features are also provided. The dataset was used in the Video
Memorability task as part of the MediaEval benchmark in 2020.

    

### [[2112.02333] LoNLI: An Extensible Framework for Testing Diverse Logical Reasoning Capabilities for NLI](http://arxiv.org/abs/2112.02333)


  Natural Language Inference (NLI) is considered a representative task to test
natural language understanding (NLU). In this work, we propose an extensible
framework to collectively yet categorically test diverse Logical reasoning
capabilities required for NLI (and by extension, NLU). Motivated by behavioral
testing, we create a semi-synthetic large test-bench (363 templates, 363k
examples) and an associated framework that offers following utilities: 1)
individually test and analyze reasoning capabilities along 17 reasoning
dimensions (including pragmatic reasoning), 2) design experiments to study
cross-capability information content (leave one out or bring one in); and 3)
the synthetic nature enable us to control for artifacts and biases. The
inherited power of automated test case instantiation from free-form natural
language templates (using CheckList), and a well-defined taxonomy of
capabilities enable us to extend to (cognitively) harder test cases while
varying the complexity of natural language. Through our analysis of
state-of-the-art NLI systems, we observe that our benchmark is indeed hard (and
non-trivial even with training on additional resources). Some capabilities
stand out as harder. Further fine-grained analysis and fine-tuning experiments
reveal more insights about these capabilities and the models -- supporting and
extending previous observations. Towards the end we also perform an user-study,
to investigate whether behavioral information can be utilised to generalize
much better for some models compared to others.

    

### [[2112.02397] Towards automated verification of multi-party consensus protocols](http://arxiv.org/abs/2112.02397)


  Blockchain technology and related frameworks have recently received extensive
attention. Blockchain systems use multi-party consensus protocols to reach
agreements on transactions. Hyperledger Fabric framework exposes a multi-party
consensus, based on endorsement policy protocol, to reach a consensus on a
transaction. In this paper, we define a problem of verification of a blockchain
multi-party consensus with probabilistic properties. Further, we propose a
verification technique of endorsement policies using statistical model checking
and hypothesis testing. We analyze several aspects of the policies, including
the ability to assign weights to organizations and the refusal probabilities of
organizations. We demonstrate on experiments the work of our verification
technique and how one can use experimental results to make the model
satisfiable the specification. One can use our technique to design enterprise
applications with the Hyperledger Fabric framework.

    

### [[2112.02413] PointCLIP: Point Cloud Understanding by CLIP](http://arxiv.org/abs/2112.02413)


  Recently, zero-shot and few-shot learning via Contrastive Vision-Language
Pre-training (CLIP) have shown inspirational performance on 2D visual
recognition, which learns to match images with their corresponding texts in
open-vocabulary settings. However, it remains under explored that whether CLIP,
pre-trained by large-scale image-text pairs in 2D, can be generalized to 3D
recognition. In this paper, we identify such a setting is feasible by proposing
PointCLIP, which conducts alignment between CLIP-encoded point cloud and 3D
category texts. Specifically, we encode a point cloud by projecting it into
multi-view depth maps without rendering, and aggregate the view-wise zero-shot
prediction to achieve knowledge transfer from 2D to 3D. On top of that, we
design an inter-view adapter to better extract the global feature and
adaptively fuse the few-shot knowledge learned from 3D into CLIP pre-trained in
2D. By just fine-tuning the lightweight adapter in the few-shot settings, the
performance of PointCLIP could be largely improved. In addition, we observe the
complementary property between PointCLIP and classical 3D-supervised networks.
By simple ensembling, PointCLIP boosts baseline's performance and even
surpasses state-of-the-art models. Therefore, PointCLIP is a promising
alternative for effective 3D point cloud understanding via CLIP under low
resource cost and data regime. We conduct thorough experiments on
widely-adopted ModelNet10, ModelNet40 and the challenging ScanObjectNN to
demonstrate the effectiveness of PointCLIP. The code is released at
this https URL.

    

### [[2112.02433] Functional Task Tree Generation from a Knowledge Graph to Solve Unseen Problems](http://arxiv.org/abs/2112.02433)


  A major component for developing intelligent and autonomous robots is a
suitable knowledge representation, from which a robot can acquire knowledge
about its actions or world. However, unlike humans, robots cannot creatively
adapt to novel scenarios, as their knowledge and environment are rigidly
defined. To address the problem of producing novel and flexible task plans
called task trees, we explore how we can derive plans with concepts not
originally in the robot's knowledge base. Existing knowledge in the form of a
knowledge graph is used as a base of reference to create task trees that are
modified with new object or state combinations. To demonstrate the flexibility
of our method, we randomly selected recipes from the Recipe1M+ dataset and
generated their task trees. The task trees were then thoroughly checked with a
visualization tool that portrays how each ingredient changes with each action
to produce the desired meal. Our results indicate that the proposed method can
produce task plans with high accuracy even for never-before-seen ingredient
combinations.

    

### [[2112.02457] Artificial Cognitively-inspired Generation of the Notion of Topological Group in the Context of Artificial Mathematical Intelligence](http://arxiv.org/abs/2112.02457)


  The new computational paradigm of conceptual computation has been introduced
in the research program of Artificial Mathematical Intelligence. We provide the
explicit artificial generation (or conceptual computation) for the fundamental
mathematical notion of topological groups. Specifically, we start with two
basic notions belonging to topology and abstract algebra, and we describe
recursively formal specifications in the Common Algebraic Specification
Language (CASL). The notion of conceptual blending between such conceptual
spaces can be materialized computationally in the Heterogeneous Tool Set
(HETS). The fundamental notion of topological groups is explicitly generated
through three different artificial specifications based on conceptual blending
and conceptual identification, starting with the concepts of continuous
functions and mathematical groups (described with minimal set-theoretical
conditions). This constitutes in additional heuristic evidence for the third
pillar of Artificial Mathematical Intelligence.

    

### [[2112.02498] Consistent Training and Decoding For End-to-end Speech Recognition Using Lattice-free MMI](http://arxiv.org/abs/2112.02498)


  Recently, End-to-End (E2E) frameworks have achieved remarkable results on
various Automatic Speech Recognition (ASR) tasks. However, Lattice-Free Maximum
Mutual Information (LF-MMI), as one of the discriminative training criteria
that show superior performance in hybrid ASR systems, is rarely adopted in E2E
ASR frameworks. In this work, we propose a novel approach to integrate LF-MMI
criterion into E2E ASR frameworks in both training and decoding stages. The
proposed approach shows its effectiveness on two of the most widely used E2E
frameworks including Attention-Based Encoder-Decoders (AEDs) and Neural
Transducers (NTs). Experiments suggest that the introduction of the LF-MMI
criterion consistently leads to significant performance improvements on various
datasets and different E2E ASR frameworks. The best of our models achieves
competitive CER of 4.1\% / 4.4\% on Aishell-1 dev/test set; we also achieve
significant error reduction on Aishell-2 and Librispeech datasets over strong
baselines.

    

### [[2112.02513] Intention Recognition for Multiple Agents](http://arxiv.org/abs/2112.02513)


  Intention recognition is an important step to facilitate collaboration in
multi-agent systems. Existing work mainly focuses on intention recognition in a
single-agent setting and uses a descriptive model, e.g. Bayesian networks, in
the recognition process. In this paper, we resort to a prescriptive approach to
model agents' behaviour where which their intentions are hidden in implementing
their plans. We introduce landmarks into the behavioural model therefore
enhancing informative features for identifying common intentions for multiple
agents. We further refine the model by focusing only action sequences in their
plan and provide a light model for identifying and comparing their intentions.
The new models provide a simple approach of grouping agents' common intentions
upon partial plans observed in agents' interactions. We provide experimental
results in support.

    

### [[2112.02557] Interpretable Privacy Preservation of Text Representations Using Vector Steganography](http://arxiv.org/abs/2112.02557)


  Contextual word representations generated by language models (LMs) learn
spurious associations present in the training corpora. Recent findings reveal
that adversaries can exploit these associations to reverse-engineer the private
attributes of entities mentioned within the corpora. These findings have led to
efforts towards minimizing the privacy risks of language models. However,
existing approaches lack interpretability, compromise on data utility and fail
to provide privacy guarantees. Thus, the goal of my doctoral research is to
develop interpretable approaches towards privacy preservation of text
representations that retain data utility while guaranteeing privacy. To this
end, I aim to study and develop methods to incorporate steganographic
modifications within the vector geometry to obfuscate underlying spurious
associations and preserve the distributional semantic properties learnt during
training.

    

### [[2112.02604] PSI: A Pedestrian Behavior Dataset for Socially Intelligent Autonomous Car](http://arxiv.org/abs/2112.02604)


  Prediction of pedestrian behavior is critical for fully autonomous vehicles
to drive in busy city streets safely and efficiently. The future autonomous
cars need to fit into mixed conditions with not only technical but also social
capabilities. As more algorithms and datasets have been developed to predict
pedestrian behaviors, these efforts lack the benchmark labels and the
capability to estimate the temporal-dynamic intent changes of the pedestrians,
provide explanations of the interaction scenes, and support algorithms with
social intelligence. This paper proposes and shares another benchmark dataset
called the IUPUI-CSRC Pedestrian Situated Intent (PSI) data with two innovative
labels besides comprehensive computer vision labels. The first novel label is
the dynamic intent changes for the pedestrians to cross in front of the
ego-vehicle, achieved from 24 drivers with diverse backgrounds. The second one
is the text-based explanations of the driver reasoning process when estimating
pedestrian intents and predicting their behaviors during the interaction
period. These innovative labels can enable several computer vision tasks,
including pedestrian intent/behavior prediction, vehicle-pedestrian interaction
segmentation, and video-to-language mapping for explainable algorithms. The
released dataset can fundamentally improve the development of pedestrian
behavior prediction models and develop socially intelligent autonomous cars to
interact with pedestrians efficiently. The dataset has been evaluated with
different tasks and is released to the public to access.

    

### [[2112.02624] Dynamic Token Normalization Improves Vision Transformer](http://arxiv.org/abs/2112.02624)


  Vision Transformer (ViT) and its variants (e.g., Swin, PVT) have achieved
great success in various computer vision tasks, owing to their capability to
learn long-range contextual information. Layer Normalization (LN) is an
essential ingredient in these models. However, we found that the ordinary LN
makes tokens at different positions similar in magnitude because it normalizes
embeddings within each token. It is difficult for Transformers to capture
inductive bias such as the positional context in an image with LN. We tackle
this problem by proposing a new normalizer, termed Dynamic Token Normalization
(DTN), where normalization is performed both within each token (intra-token)
and across different tokens (inter-token). DTN has several merits. Firstly, it
is built on a unified formulation and thus can represent various existing
normalization methods. Secondly, DTN learns to normalize tokens in both
intra-token and inter-token manners, enabling Transformers to capture both the
global contextual information and the local positional context. {Thirdly, by
simply replacing LN layers, DTN can be readily plugged into various vision
transformers, such as ViT, Swin, PVT, LeViT, T2T-ViT, BigBird and Reformer.
Extensive experiments show that the transformer equipped with DTN consistently
outperforms baseline model with minimal extra parameters and computational
overhead. For example, DTN outperforms LN by $0.5\%$ - $1.2\%$ top-1 accuracy
on ImageNet, by $1.2$ - $1.4$ box AP in object detection on COCO benchmark, by
$2.3\%$ - $3.9\%$ mCE in robustness experiments on ImageNet-C, and by $0.5\%$ -
$0.8\%$ accuracy in Long ListOps on Long-Range Arena.} Codes will be made
public at \url{this https URL}

    

### [[2112.02626] The Complexity of Data-Driven Norm Synthesis and Revision](http://arxiv.org/abs/2112.02626)


  Norms have been widely proposed as a way of coordinating and controlling the
activities of agents in a multi-agent system (MAS). A norm specifies the
behaviour an agent should follow in order to achieve the objective of the MAS.
However, designing norms to achieve a particular system objective can be
difficult, particularly when there is no direct link between the language in
which the system objective is stated and the language in which the norms can be
expressed. In this paper, we consider the problem of synthesising a norm from
traces of agent behaviour, where each trace is labelled with whether the
behaviour satisfies the system objective. We show that the norm synthesis
problem is NP-complete.

    

### [[2112.02690] Open Vocabulary Electroencephalography-To-Text Decoding and Zero-shot Sentiment Classification](http://arxiv.org/abs/2112.02690)


  State-of-the-art brain-to-text systems have achieved great success in
decoding language directly from brain signals using neural networks. However,
current approaches are limited to small closed vocabularies which are far from
enough for natural communication. In addition, most of the high-performing
approaches require data from invasive devices (e.g., ECoG). In this paper, we
extend the problem to open vocabulary Electroencephalography(EEG)-To-Text
Sequence-To-Sequence decoding and zero-shot sentence sentiment classification
on natural reading tasks. We hypothesis that the human brain functions as a
special text encoder and propose a novel framework leveraging pre-trained
language models (e.g., BART). Our model achieves a 40.1% BLEU-1 score on
EEG-To-Text decoding and a 55.6% F1 score on zero-shot EEG-based ternary
sentiment classification, which significantly outperforms supervised baselines.
Furthermore, we show that our proposed model can handle data from various
subjects and sources, showing great potential for a high-performance open
vocabulary brain-to-text system once sufficient data is available

    

### [[2112.02732] JointLK: Joint Reasoning with Language Models and Knowledge Graphs for Commonsense Question Answering](http://arxiv.org/abs/2112.02732)


  Existing KG-augmented models for question answering primarily focus on
designing elaborate Graph Neural Networks (GNNs) to model knowledge graphs
(KGs). However, they ignore (i) the effectively fusing and reasoning over
question context representations and the KG representations, and (ii)
automatically selecting relevant nodes from the noisy KGs during reasoning. In
this paper, we propose a novel model, JointLK, which solves the above
limitations through the joint reasoning of LMs and GNNs and the dynamic KGs
pruning mechanism. Specifically, JointLK performs joint reasoning between the
LMs and the GNNs through a novel dense bidirectional attention module, in which
each question token attends on KG nodes and each KG node attends on question
tokens, and the two modal representations fuse and update mutually by
multi-step interactions. Then, the dynamic pruning module uses the attention
weights generated by joint reasoning to recursively prune irrelevant KG nodes.
Our results on the CommonsenseQA and OpenBookQA datasets demonstrate that our
modal fusion and knowledge pruning methods can make better use of relevant
knowledge for reasoning.

    

### [[2112.02767] A General Framework for Debiasing in CTR Prediction](http://arxiv.org/abs/2112.02767)


  Most of the existing methods for debaising in click-through rate (CTR)
prediction depend on an oversimplified assumption, i.e., the click probability
is the product of observation probability and relevance probability. However,
since there is a complicated interplay between these two probabilities, these
methods cannot be applied to other scenarios, e.g. query auto completion (QAC)
and route recommendation. We propose a general debiasing framework without
simplifying the relationships between variables, which can handle all scenarios
in CTR prediction. Simulation experiments show that: under the simplest
scenario, our method maintains a similar AUC with the state-of-the-art methods;
in other scenarios, our method achieves considerable improvements compared with
existing methods. Meanwhile, in online experiments, the framework also gains
significant improvements consistently.

    

### [[2112.02788] Texture Reformer: Towards Fast and Universal Interactive Texture Transfer](http://arxiv.org/abs/2112.02788)


  In this paper, we present the texture reformer, a fast and universal
neural-based framework for interactive texture transfer with user-specified
guidance. The challenges lie in three aspects: 1) the diversity of tasks, 2)
the simplicity of guidance maps, and 3) the execution efficiency. To address
these challenges, our key idea is to use a novel feed-forward multi-view and
multi-stage synthesis procedure consisting of I) a global view structure
alignment stage, II) a local view texture refinement stage, and III) a holistic
effect enhancement stage to synthesize high-quality results with coherent
structures and fine texture details in a coarse-to-fine fashion. In addition,
we also introduce a novel learning-free view-specific texture reformation
(VSTR) operation with a new semantic map guidance strategy to achieve more
accurate semantic-guided and structure-preserved texture transfer. The
experimental results on a variety of application scenarios demonstrate the
effectiveness and superiority of our framework. And compared with the
state-of-the-art interactive texture transfer algorithms, it not only achieves
higher quality results but, more remarkably, also is 2-5 orders of magnitude
faster. Code is available at this https URL.

    

### [[2012.09110] Developing Future Human-Centered Smart Cities: Critical Analysis of Smart City Security, Interpretability, and Ethical Challenges](http://arxiv.org/abs/2012.09110)


  As the globally increasing population drives rapid urbanisation in various
parts of the world, there is a great need to deliberate on the future of the
cities worth living. In particular, as modern smart cities embrace more and
more data-driven artificial intelligence services, it is worth remembering that
technology can facilitate prosperity, wellbeing, urban livability, or social
justice, but only when it has the right analog complements (such as
well-thought out policies, mature institutions, responsible governance); and
the ultimate objective of these smart cities is to facilitate and enhance human
welfare and social flourishing. Researchers have shown that various
technological business models and features can in fact contribute to social
problems such as extremism, polarization, misinformation, and Internet
addiction. In the light of these observations, addressing the philosophical and
ethical questions involved in ensuring the security, safety, and
interpretability of such AI algorithms that will form the technological bedrock
of future cities assumes paramount importance. Globally there are calls for
technology to be made more humane and human-centered. In this paper, we analyze
and explore key challenges including security, robustness, interpretability,
and ethical (data and algorithmic) challenges to a successful deployment of AI
in human-centric applications, with a particular emphasis on the convergence of
these concepts/challenges. We provide a detailed review of existing literature
on these key challenges and analyze how one of these challenges may lead to
others or help in solving other challenges. The paper also advises on the
current limitations, pitfalls, and future directions of research in these
domains, and how it can fill the current gaps and lead to better solutions. We
believe such rigorous analysis will provide a baseline for future research in
the domain.

    

### [[2104.08219] Flexible Instance-Specific Rationalization of NLP Models](http://arxiv.org/abs/2104.08219)


  Recent research on model interpretability in natural language processing
extensively uses feature scoring methods for identifying which parts of the
input are the most important for a model to make a prediction (i.e. explanation
or rationale). However, previous research has shown that there is no clear best
scoring method across various text classification tasks while practitioners
typically have to make several other ad-hoc choices regarding the length and
the type of the rationale (e.g. short or long, contiguous or not). Inspired by
this, we propose a simple yet effective and flexible method that allows
selecting optimally for each data instance: (1) a feature scoring method; (2)
the length; and (3) the type of the rationale. Our method is inspired by input
erasure approaches to interpretability which assume that the most faithful
rationale for a prediction should be the one with the highest difference
between the model's output distribution using the full text and the text after
removing the rationale as input respectively. Evaluation on four standard text
classification datasets shows that our proposed method provides more faithful,
comprehensive and highly sufficient explanations compared to using a fixed
feature scoring method, rationale length and type. More importantly, we
demonstrate that a practitioner is not required to make any ad-hoc choices in
order to extract faithful rationales using our approach.

    

### [[2104.10726] Learning Fine-grained Fact-Article Correspondence in Legal Cases](http://arxiv.org/abs/2104.10726)


  Automatically recommending relevant law articles to a given legal case has
attracted much attention as it can greatly release human labor from searching
over the large database of laws. However, current researches only support
coarse-grained recommendation where all relevant articles are predicted as a
whole without explaining which specific fact each article is relevant with.
Since one case can be formed of many supporting facts, traversing over them to
verify the correctness of recommendation results can be time-consuming. We
believe that learning fine-grained correspondence between each single fact and
law articles is crucial for an accurate and trustworthy AI system. With this
motivation, we perform a pioneering study and create a corpus with manually
annotated fact-article correspondences. We treat the learning as a text
matching task and propose a multi-level matching network to address it. To help
the model better digest the content of law articles, we parse articles in form
of premise-conclusion pairs with random forest. Experiments show that the
parsed form yielded better performance and the resulting model surpassed other
popular text matching baselines. Furthermore, we compare with previous
researches and find that establishing the fine-grained fact-article
correspondences can improve the recommendation accuracy by a large margin. Our
best system reaches an F1 score of 96.3%, making it of great potential for
practical use. It can also significantly boost the downstream task of legal
decision prediction, increasing the F1 score by up to 12.7%.

    

### [[2106.05346] End-to-End Training of Multi-Document Reader and Retriever for Open-Domain Question Answering](http://arxiv.org/abs/2106.05346)


  We present an end-to-end differentiable training method for
retrieval-augmented open-domain question answering systems that combine
information from multiple retrieved documents when generating answers. We model
retrieval decisions as latent variables over sets of relevant documents. Since
marginalizing over sets of retrieved documents is computationally hard, we
approximate this using an expectation-maximization algorithm. We iteratively
estimate the value of our latent variable (the set of relevant documents for a
given question) and then use this estimate to update the retriever and reader
parameters. We hypothesize that such end-to-end training allows training
signals to flow to the reader and then to the retriever better than staged-wise
training. This results in a retriever that is able to select more relevant
documents for a question and a reader that is trained on more accurate
documents to generate an answer. Experiments on three benchmark datasets
demonstrate that our proposed method outperforms all existing approaches of
comparable size by 2-3% absolute exact match points, achieving new
state-of-the-art results. Our results also demonstrate the feasibility of
learning to retrieve to improve answer generation without explicit supervision
of retrieval decisions.

    

### [[2106.09259] A Random CNN Sees Objects: One Inductive Bias of CNN and Its Applications](http://arxiv.org/abs/2106.09259)


  This paper starts by revealing a surprising finding: without any learning, a
randomly initialized CNN can localize objects surprisingly well. That is, a CNN
has an inductive bias to naturally focus on objects, named as Tobias ("The
object is at sight") in this paper. This empirical inductive bias is further
analyzed and successfully applied to self-supervised learning (SSL). A CNN is
encouraged to learn representations that focus on the foreground object, by
transforming every image into various versions with different backgrounds,
where the foreground and background separation is guided by Tobias.
Experimental results show that the proposed Tobias significantly improves
downstream tasks, especially for object detection. This paper also shows that
Tobias has consistent improvements on training sets of different sizes, and is
more resilient to changes in image augmentations. Code is available at
this https URL.

    

### [[2109.05739] CEM: Commonsense-aware Empathetic Response Generation](http://arxiv.org/abs/2109.05739)


  A key trait of daily conversations between individuals is the ability to
express empathy towards others, and exploring ways to implement empathy is a
crucial step towards human-like dialogue systems. Previous approaches on this
topic mainly focus on detecting and utilizing the user's emotion for generating
empathetic responses. However, since empathy includes both aspects of affection
and cognition, we argue that in addition to identifying the user's emotion,
cognitive understanding of the user's situation should also be considered. To
this end, we propose a novel approach for empathetic response generation, which
leverages commonsense to draw more information about the user's situation and
uses this additional information to further enhance the empathy expression in
generated responses. We evaluate our approach on EmpatheticDialogues, which is
a widely-used benchmark dataset for empathetic response generation. Empirical
results demonstrate that our approach outperforms the baseline models in both
automatic and human evaluations and can generate more informative and
empathetic responses.

    

### [[2110.01509] DeepA2: A Modular Framework for Deep Argument Analysis with Pretrained Neural Text2Text Language Models](http://arxiv.org/abs/2110.01509)


  In this paper, we present and implement a multi-dimensional, modular
framework for performing deep argument analysis (DeepA2) using current
pre-trained language models (PTLMs). ArgumentAnalyst -- a T5 model (Raffel et
al. 2020) set up and trained within DeepA2 -- reconstructs argumentative texts,
which advance an informal argumentation, as valid arguments: It inserts, e.g.,
missing premises and conclusions, formalizes inferences, and coherently links
the logical reconstruction to the source text. We create a synthetic corpus for
deep argument analysis, and evaluate ArgumentAnalyst on this new dataset as
well as on existing data, specifically EntailmentBank (Dalvi et al. 2021). Our
empirical findings vindicate the overall framework and highlight the advantages
of a modular design, in particular its ability to emulate established
heuristics (such as hermeneutic cycles), to explore the model's uncertainty, to
cope with the plurality of correct solutions (underdetermination), and to
exploit higher-order evidence.

    

### [[2111.07970] Triggerless Backdoor Attack for NLP Tasks with Clean Labels](http://arxiv.org/abs/2111.07970)


  Backdoor attacks pose a new threat to NLP models. A standard strategy to
construct poisoned data in backdoor attacks is to insert triggers (e.g., rare
words) into selected sentences and alter the original label to a target label.
This strategy comes with a severe flaw of being easily detected from both the
trigger and the label perspectives: the trigger injected, which is usually a
rare word, leads to an abnormal natural language expression, and thus can be
easily detected by a defense model; the changed target label leads the example
to be mistakenly labeled and thus can be easily detected by manual inspections.
To deal with this issue, in this paper, we propose a new strategy to perform
textual backdoor attacks which do not require an external trigger, and the
poisoned samples are correctly labeled. The core idea of the proposed strategy
is to construct clean-labeled examples, whose labels are correct but can lead
to test label changes when fused with the training set. To generate poisoned
clean-labeled examples, we propose a sentence generation model based on the
genetic algorithm to cater to the non-differentiable characteristic of text
data. Extensive experiments demonstrate that the proposed attacking strategy is
not only effective, but more importantly, hard to defend due to its triggerless
and clean-labeled nature. Our work marks the first step towards developing
triggerless attacking strategies in NLP.

    

### [[2102.02710] Matching Impatient and Heterogeneous Demand and Supply](http://arxiv.org/abs/2102.02710)


  Service platforms must determine rules for matching heterogeneous demand
(customers) and supply (workers) that arrive randomly over time and may be lost
if forced to wait too long for a match. Our objective is to maximize the
cumulative value of matches, minus costs incurred when demand and supply wait.
We develop a fluid model, that approximates the evolution of the stochastic
model, and captures explicitly the nonlinear dependence between the amount of
demand and supply waiting and the distribution of their patience times. The
fluid model invariant states approximate the steady-state mean queue-lengths in
the stochastic system, and, therefore, can be used to develop an optimization
problem whose optimal solution provides matching rates between demand and
supply types that are asymptotically optimal (on fluid scale, as demand and
supply rates grow large). We propose a discrete review matching policy that
asymptotically achieves the optimal matching rates. We further show that when
the aforementioned matching optimization problem has an optimal extreme point
solution, which occurs when the patience time distributions have increasing
hazard rate functions, a state-independent priority policy, that ranks the
edges on the bipartite graph connecting demand and supply, is asymptotically
optimal. A key insight from this analysis is that the ranking critically
depends on the patience time distributions, and may be different for different
distributions even if they have the same mean, demonstrating that models
assuming, e.g., exponential patience times for tractability, may lack
robustness. Finally, we observe that when holding costs are zero, a discrete
review policy, that does not require knowledge of inter-arrival and patience
time distributions, is asymptotically optimal.

    

### [[2011.13451] Strongly-Normalizing Higher-Order Relational Queries](http://arxiv.org/abs/2011.13451)


  Language-integrated query is a popular and powerful programming construct
allowing database queries and ordinary program code to interoperate seamlessly
and safely. Language-integrated query techniques rely on classical results
about the nested relational calculus stating that its queries can be
algorithmically translated to SQL, as long as their result type is a flat
relation. Cooper and others advocated higher-order nested relational calculi as
a basis for language-integrated queries in functional languages such as Links
and F#. However, the translation of higher-order relational queries to SQL
relies on a rewrite system for which no strong normalization proof has been
published: a previous proof attempt does not deal correctly with rewrite rules
that duplicate subterms. This paper fills the gap in the literature, explaining
the difficulty with the previous attempt, and showing how to extend the
$\top\top$-lifting approach of Lindley and Stark to accommodate duplicating
rewrites. We also show how to extend the proof to a recently-introduced
calculus for heterogeneous queries mixing set and multiset semantics.

    

### [<title>「央视新闻」江山开医院疾病诊断证明「代开具医院病历《手机搜狐网》 - DockOne.io</title>](http://dockone.io/question/1661183)

### [<title>【央视新闻】上海如何代开具专用住宿费发票【手机搜狐网】 - DockOne.io</title>](http://dockone.io/question/1661182)

### [<title>【央视新闻】杭州如何代开具住宿费发票【手机搜狐网】 - DockOne.io</title>](http://dockone.io/question/1661181)

### [<title>【央视新闻】杭州如何代开具住宿费增值税发票【手机搜狐网】 - DockOne.io</title>](http://dockone.io/question/1661180)

### [<title>「今日爆料」平湖开三甲医院诊断证明「代开具医院病历[长津湖」 - DockOne.io</title>](http://dockone.io/question/1661179)

### [<title>【央视新闻】北京如何代开具住宿费发票【手机搜狐网】 - DockOne.io</title>](http://dockone.io/question/1661178)

### [<title>【央视新闻】北京如何代开具住宿费增值税发票【手机搜狐网】 - DockOne.io</title>](http://dockone.io/question/1661177)

### [<title>【央视新闻】上海如何代开具住宿费发票【手机搜狐网】 - DockOne.io</title>](http://dockone.io/question/1661176)

### [<title>【央视新闻】上海如何代开具住宿费增值税发票【手机搜狐网】 - DockOne.io</title>](http://dockone.io/question/1661175)

### [<title>【央视新闻】杭州如何代开具专用住宿费发票【手机搜狐网】 - DockOne.io</title>](http://dockone.io/question/1661174)

### [<title>「今日爆料」奉化开三甲医院诊断证明「代开具医院病历[长津湖」 - DockOne.io</title>](http://dockone.io/question/1661173)

### [<title>「央视新闻」嵊州开医院疾病诊断证明「代开具医院病历《手机搜狐网》 - DockOne.io</title>](http://dockone.io/question/1661172)

### [<title>「央视新闻」诸暨开医院疾病诊断证明「代开具医院病历《手机搜狐网》 - DockOne.io</title>](http://dockone.io/question/1661171)

### [<title>【央视新闻】太原如何开具餐饮费发票_太原【手机搜狐网】 - DockOne.io</title>](http://dockone.io/question/1661170)

### [<title>【央视新闻】北京如何代开具专用住宿费发票【手机搜狐网】 - DockOne.io</title>](http://dockone.io/question/1661169)

### [<title>【央视新闻】合肥如何开具餐饮费发票_合肥【手机搜狐网】 - DockOne.io</title>](http://dockone.io/question/1661168)

### [<title>【央视新闻】上海如何代开具专用住宿费发票【手机搜狐网】 - DockOne.io</title>](http://dockone.io/question/1661167)

### [<title>「今日爆料」余姚开三甲医院诊断证明「代开具医院病历[长津湖」 - DockOne.io</title>](http://dockone.io/question/1661166)

### [<title>「央视新闻」桐乡开医院疾病诊断证明「代开具医院病历《手机搜狐网》 - DockOne.io</title>](http://dockone.io/question/1661165)

### [<title>【央视新闻】西安如何开具餐饮费发票_西安【手机搜狐网】 - DockOne.io</title>](http://dockone.io/question/1661164)

### [<title>【央视新闻】广州如何可以开具税务局查询发票【手机搜狐网】 - DockOne.io</title>](http://dockone.io/question/1715007)

### [<title>「今日爆料」淮北开三甲医院诊断证明「休假病例单[长津湖」 - DockOne.io</title>](http://dockone.io/question/1715006)

### [<title>「今日发布」南宁如何可以开具税务局查询发票「手机腾讯网」 - DockOne.io</title>](http://dockone.io/question/1715005)

### [<title>「央视新闻」张掖开医院疾病诊断证明「医院休假单《手机搜狐网》 - DockOne.io</title>](http://dockone.io/question/1715004)

### [<title>「今日爆料」宿州开三甲医院诊断证明「休假病例单[长津湖」 - DockOne.io</title>](http://dockone.io/question/1715003)

### [<title>【央视新闻】深圳如何可以开具税务局查询发票【手机搜狐网】 - DockOne.io</title>](http://dockone.io/question/1715002)

### [<title>「今日发布」广西如何可以开具税务局查询发票「手机腾讯网」 - DockOne.io</title>](http://dockone.io/question/1715001)

### [<title>「央视新闻」酒泉开医院疾病诊断证明「医院休假单《手机搜狐网》 - DockOne.io</title>](http://dockone.io/question/1715000)

### [<title>【央视新闻】广东如何可以开具税务局查询发票【手机搜狐网】 - DockOne.io</title>](http://dockone.io/question/1714999)

### [<title>「今日爆料」
合肥开三甲医院诊断证明「休假病例单[长津湖」 - DockOne.io</title>](http://dockone.io/question/1714998)

### [<title>「今日爆料」重庆开三甲医院诊断证明「休假病例单[长津湖」 - DockOne.io</title>](http://dockone.io/question/1714997)

### [<title>「央视新闻」白银开医院疾病诊断证明「医院休假单《手机搜狐网》 - DockOne.io</title>](http://dockone.io/question/1714996)

### [<title>「今日发布」福州如何可以开具税务局查询发票「手机腾讯网」 - DockOne.io</title>](http://dockone.io/question/1714995)

### [<title>「央视新闻」金昌开医院疾病诊断证明「医院休假单《手机搜狐网》 - DockOne.io</title>](http://dockone.io/question/1714994)

### [<title>【央视新闻】北京如何可以开具税务局查询发票【手机搜狐网】 - DockOne.io</title>](http://dockone.io/question/1714993)

### [<title>「今日爆料」天津开三甲医院诊断证明「休假病例单[长津湖」 - DockOne.io</title>](http://dockone.io/question/1714992)

### [<title>「今日发布」厦门如何可以开具税务局查询发票「手机腾讯网」 - DockOne.io</title>](http://dockone.io/question/1714991)

### [<title>【央视新闻】上海如何可以开具税务局查询发票【手机搜狐网】 - DockOne.io</title>](http://dockone.io/question/1714990)

### [<title>「央视新闻」嘉峪关开医院疾病诊断证明「医院休假单《手机搜狐网》 - DockOne.io</title>](http://dockone.io/question/1714989)

### [<title>「今日爆料」上海开三甲医院诊断证明「休假病例单[长津湖」 - DockOne.io</title>](http://dockone.io/question/1714988)