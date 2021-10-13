
## 2021-10-13

### [<title>Lack of reproducibility despite using the same seed - XGBoost</title>](https://discuss.xgboost.ai/t/lack-of-reproducibility-despite-using-the-same-seed/2485/3)

### [[2110.05554] Towards a Cost vs. Quality Sweet Spot for Monitoring Networks](http://arxiv.org/abs/2110.05554)


  Continuously monitoring a wide variety of performance and fault metrics has
become a crucial part of operating large-scale datacenter networks. In this
work, we ask whether we can reduce the costs to monitor -- in terms of
collection, storage and analysis -- by judiciously controlling how much and
which measurements we collect. By positing that we can treat almost all
measured signals as sampled time-series, we show that we can use signal
processing techniques such as the Nyquist-Shannon theorem to avoid wasteful
data collection. We show that large savings appear possible by analyzing tens
of popular measurements from a production datacenter network. We also discuss
the technical challenges that must be solved when applying these techniques in
practice.

    

### [[2110.05772] Quantifying Nations Exposure to Traffic Observation and Selective Tampering](http://arxiv.org/abs/2110.05772)


  Almost all popular Internet services are hosted in a select set of countries,
forcing other nations to rely on international connectivity to access them. We
infer instances where traffic towards a large portion of a country is serviced
by a small number of Autonomous Systems, and, therefore, may be exposed to
observation or selective tampering. We introduce the Country-level Transit
Influence (CTI) metric to quantify the significance of a given AS on the
international transit service of a particular country. By studying the CTI
values for the top ASes in each country, we find that 32 nations have transit
ecosystems that render them particularly exposed, with traffic destined to over
40% of their IP addresses privy to a single AS. In the nations where we are
able to validate our findings with in-country operators, we obtain 83% accuracy
on average. In the countries we examine, CTI reveals two classes of networks
that play a particularly prominent role: submarine cable operators and
state-owned ASes.

    

### [[2110.05808] Worst-case Delay Bounds in Time-Sensitive Networks with Packet Replication and Elimination](http://arxiv.org/abs/2110.05808)


  Packet replication and elimination functions are used by time-sensitive
networks (as in the context of IEEE TSN and IETF DetNet) to increase the
reliability of the network. Packets are replicated onto redundant paths by a
replication function. Later the paths merge again and an elimination function
removes the duplicates. This redundancy scheme has an effect on the timing
behavior of time-sensitive networks and many challenges arise from conducting
timing analyses. The replication can induce a burstiness increase along the
paths of replicates, as well as packet mis-ordering that could increase the
delays in the crossed bridges or routers. The induced packet mis-ordering could
also negatively affect the interactions between the redundancy and scheduling
mechanisms such as traffic regulators (as with per-flow regulators and
interleaved regulators, implemented by TSN asynchronous traffic shaping). Using
the network calculus framework, we provide a method of worst-case timing
analysis for time-sensitive networks that implement redundancy mechanisms in
the general use case, i.e., at end-devices and/or intermediate nodes. We first
provide a network calculus toolbox for bounding the burstiness increase and the
amount of reordering caused by the elimination function of duplicate packets.
We then analyze the interactions with traffic regulators and show that their
shaping-for-free property does not hold when placed after a packet elimination
function. We provide a bound for the delay penalty when using per-flow
regulators and prove that the penalty is not bounded with interleaved
regulators. Finally, we use an industrial use-case to show the applicability
and the benefits of our findings.

    

### [[2110.05906] Energy-cost aware off-grid base stations with IoT devices for developing a green heterogeneous network](http://arxiv.org/abs/2110.05906)


  Heterogeneous network (HetNet) is a specified cellular platform to tackle the
rapidly growing anticipated data traffic. From communications perspective, data
loads can be mapped to energy loads that are generally placed on the operator
networks. Meanwhile, renewable energy aided networks offer to curtail fossil
fuel consumption, so to reduce environmental pollution. This paper proposes a
renewable energy based power supply architecture for off-grid HetNet using a
novel energy sharing model. Solar photovoltaic (PV) along with sufficient
energy storage devices are used for each macro, micro, pico, or femto base
station (BS). Additionally, biomass generator (BG) is used for macro and micro
BSs. The collocated macro and micro BSs are connected through end-to-end
resistive lines. A novel weighted proportional-fair resource-scheduling
algorithm with sleep mechanisms is proposed for non-real time (NRT)
applications by trading-off the power consumption and communication delays.
Furthermore, the proposed algorithm with extended discontinuous reception
(eDRX) and power saving mode (PSM) for narrowband internet of things (IoT)
applications extends battery lifetime for IoT devices. HOMER optimization
software is used to perform optimal system architecture, economic, and carbon
footprint analyses while Monte-Carlo simulation tool is used for evaluating the
throughput and energy efficiency performances. The proposed algorithms are
valid for the practical data of the rural areas. We demonstrate the proposed
power supply architecture is energy-efficient, cost-effective, reliable, and
eco-friendly.

    

### [[2110.06060] Downtime-Aware O-RAN VNF Deployment Strategy for Optimized Self-Healing in the O-Cloud](http://arxiv.org/abs/2110.06060)


  Due to the huge surge in the traffic of IoT devices and applications, mobile
networks require a new paradigm shift to handle such demand roll out. With the
5G economics, those networks should provide virtualized multi-vendor and
intelligent systems that can scale and efficiently optimize the investment of
the underlying infrastructure. Therefore, the market stakeholders have proposed
the Open Radio Access Network (O-RAN) as one of the solutions to improve the
network performance, agility, and time-to-market of new applications. O-RAN
harnesses the power of artificial intelligence, cloud computing, and new
network technologies (NFV and SDN) to allow operators to manage their
infrastructure in a cost-efficient manner. Therefore, it is necessary to
address the O-RAN performance and availability challenges autonomously while
maintaining the quality of service. In this work, we propose an optimized
deployment strategy for the virtualized O-RAN units in the O-Cloud to minimize
the network's outage while complying with the performance and operational
requirements. The model's evaluation provides an optimal deployment strategy
that maximizes the network's overall availability and adheres to the
O-RAN-specific requirements.

    

### [[2012.06433] Advanced Algorithms in Heterogeneous and Uncertain Networking Environments](http://arxiv.org/abs/2012.06433)


  Communication networks are used today everywhere and on every scale: starting
from small Internet of Things (IoT) networks at home, via campus and enterprise
networks, and up to tier-one networks of Internet providers. Accordingly,
network devices should support a plethora of tasks with highly heterogeneous
characteristics in terms of processing time, bandwidth energy consumption,
deadlines and so on. Evaluating these characteristics and the amount of
currently available resources for handling them requires analyzing all the
arriving inputs, gathering information from numerous remote devices, and
integrating all this information. Performing all these tasks in real time is
very challenging in today's networking environments, which are characterized by
tight bounds on the latency, and always-increasing data rates. Hence, network
algorithms should typically make decisions under uncertainty.
This work addresses optimizing performance in heterogeneous and uncertain
networking environments. We begin by detailing the sources of heterogeneity and
uncertainty and show that uncertainty appears in all layers of network design,
including the time required to perform a task; the amount of available
resources; and the expected gain from successfully completing a task. Next, we
survey current solutions and show their limitations. Based on these insights we
develop general design concepts to tackle heterogeneity and uncertainty, and
then use these concepts to design practical algorithms. For each of our
algorithms, we provide rigorous mathematical analysis, thus showing worst-case
performance guarantees. Finally, we implement and run the suggested algorithms
on various input traces, thus obtaining further insights as to our algorithmic
design principles.

    

### [[2110.05476] Image Compression and Classification Using Qubits and Quantum Deep Learning](http://arxiv.org/abs/2110.05476)


  Recent work suggests that quantum machine learning techniques can be used for
classical image classification by encoding the images in quantum states and
using a quantum neural network for inference. However, such work has been
restricted to very small input images, at most 4 x 4, that are unrealistic and
cannot even be accurately labeled by humans. The primary difficulties in using
larger input images is that hitherto-proposed encoding schemes necessitate more
qubits than are physically realizable. We propose a framework to classify
larger, realistic images using quantum systems. Our approach relies on a novel
encoding mechanism that embeds images in quantum states while necessitating
fewer qubits than prior work. Our framework is able to classify images that are
larger than previously possible, up to 16 x 16 for the MNIST dataset on a
personal laptop, and obtains accuracy comparable to classical neural networks
with the same number of learnable parameters. We also propose a technique for
further reducing the number of qubits needed to represent images that may
result in an easier physical implementation at the expense of final
performance. Our work enables quantum machine learning and classification on
classical datasets of dimensions that were previously intractable by physically
realizable quantum computers or classical simulation

    

### [[2110.05477] Predicting the spread of COVID-19 in Delhi, India using Deep Residual Recurrent Neural Networks](http://arxiv.org/abs/2110.05477)


  Detecting the spread of coronavirus will go a long way toward reducing human
and economic loss. Unfortunately, existing Epidemiological models used for
COVID 19 prediction models are too slow and fail to capture the COVID-19
development in detail. This research uses Partial Differential Equations to
improve the processing speed and accuracy of forecasting of COVID 19 governed
by SEIRD model equations. The dynamics of COVID 19 were extracted using
Convolutional Neural Networks and Deep Residual Recurrent Neural Networks from
data simulated using PDEs. The DRRNNs accuracy is measured using Mean Squared
Error. The DRRNNs COVID-19 prediction model has been shown to have accurate
COVID-19 predictions. In addition, we concluded that DR-RNNs can significantly
advance the ability to support decision-making in real time COVID-19
prediction.

    

### [[2110.05478] An In-depth Summary of Recent Artificial Intelligence Applications in Drug Design](http://arxiv.org/abs/2110.05478)


  As a promising tool to navigate in the vast chemical space, artificial
intelligence (AI) is leveraged for drug design. From the year 2017 to 2021, the
number of applications of several recent AI models (i.e. graph neural network
(GNN), recurrent neural network (RNN), variation autoencoder (VAE), generative
adversarial network (GAN), flow and reinforcement learning (RL)) in drug design
increases significantly. Many relevant literature reviews exist. However, none
of them provides an in-depth summary of many applications of the recent AI
models in drug design. To complement the existing literature, this survey
includes the theoretical development of the previously mentioned AI models and
detailed summaries of 42 recent applications of AI in drug design. Concretely,
13 of them leverage GNN for molecular property prediction and 29 of them use RL
and/or deep generative models for molecule generation and optimization. In most
cases, the focus of the summary is the models, their variants, and
modifications for specific tasks in drug design. Moreover, 60 additional
applications of AI in molecule generation and optimization are briefly
summarized in a table. Finally, this survey provides a holistic discussion of
the abundant applications so that the tasks, potential solutions, and
challenges in AI-based drug design become evident.

    

### [[2110.05481] Which Samples Should be Learned First: Easy or Hard?](http://arxiv.org/abs/2110.05481)


  An effective weighting scheme for training samples is essential for learning
tasks. Numerous weighting schemes have been proposed. Some schemes take the
easy-first mode on samples, whereas some others take the hard-first mode.
Naturally, an interesting yet realistic question is raised. Which samples
should be learned first given a new learning task, easy or hard? To answer this
question, three aspects of research are carried out. First, a high-level
unified weighted loss is proposed, providing a more comprehensive view for
existing schemes. Theoretical analysis is subsequently conducted and
preliminary conclusions are obtained. Second, a flexible weighting scheme is
proposed to overcome the defects of existing schemes. The three modes, namely,
easy/medium/hard-first, can be flexibly switched in the proposed scheme. Third,
a wide range of experiments are conducted to further compare the weighting
schemes in different modes. On the basis of these works, reasonable answers are
obtained. Factors including prior knowledge and data characteristics determine
which samples should be learned first in a learning task.

    

### [[2110.05498] Satellite galaxy abundance dependency on cosmology in Magneticum simulations](http://arxiv.org/abs/2110.05498)


  Context: Modelling satellite galaxy abundance $N_s$ in Galaxy Clusters (GCs)
is a key element in modelling the Halo Occupation Distribution (HOD), which
itself is a powerful tool to connect observational studies with numerical
simulations. Aims: To study the impact of cosmological parameters on satellite
abundance both in cosmological simulations and in mock observations. Methods:
We build an emulator (HODEmu, \url{this https URL}) of
satellite abundance based on cosmological parameters $\Omega_m, \Omega_b,
\sigma_8, h_0$ and redshift $z.$ We train our emulator using \magneticum
hydrodynamic simulations that span 15 different cosmologies, each over $4$
redshift slices between $0<z<0.5,$ and for each setup we fit normalisation $A$,
log-slope $\beta$ and Gaussian fractional-scatter $\sigma$ of the $N_s-M$
relation. The emulator is based on multi-variate output Gaussian Process
Regression (GPR). Results: We find that $A$ and $\beta$ depend on cosmological
parameters, even if weakly, especially on $\Omega_m,$ $\Omega_b.$ This
dependency can explain some discrepancies found in literature between satellite
HOD of different cosmological simulations (Magneticum, Illustris, BAHAMAS). We
also show that satellite abundance cosmology dependency differs between
full-physics (FP) simulations, dark-matter only (DMO), and non-radiative
simulations. Conclusions: This work provides a preliminary calibration of the
cosmological dependency of the satellite abundance of high mass halos, and we
showed that modelling HOD with cosmological parameters is necessary to
interpret satellite abundance, and we showed the importance of using FP
simulations in modelling this dependency.

    

### [[2110.05517] Learnability of the output distributions of local quantum circuits](http://arxiv.org/abs/2110.05517)


  There is currently a large interest in understanding the potential advantages
quantum devices can offer for probabilistic modelling. In this work we
investigate, within two different oracle models, the probably approximately
correct (PAC) learnability of quantum circuit Born machines, i.e., the output
distributions of local quantum circuits. We first show a negative result,
namely, that the output distributions of super-logarithmic depth Clifford
circuits are not sample-efficiently learnable in the statistical query model,
i.e., when given query access to empirical expectation values of bounded
functions over the sample space. This immediately implies the hardness, for
both quantum and classical algorithms, of learning from statistical queries the
output distributions of local quantum circuits using any gate set which
includes the Clifford group. As many practical generative modelling algorithms
use statistical queries -- including those for training quantum circuit Born
machines -- our result is broadly applicable and strongly limits the
possibility of a meaningful quantum advantage for learning the output
distributions of local quantum circuits. As a positive result, we show that in
a more powerful oracle model, namely when directly given access to samples, the
output distributions of local Clifford circuits are computationally efficiently
PAC learnable by a classical learner. Our results are equally applicable to the
problems of learning an algorithm for generating samples from the target
distribution (generative modelling) and learning an algorithm for evaluating
its probabilities (density modelling). They provide the first rigorous insights
into the learnability of output distributions of local quantum circuits from
the probabilistic modelling perspective.

    

### [[2110.05518] Global Optimality Beyond Two Layers: Training Deep ReLU Networks via Convex Programs](http://arxiv.org/abs/2110.05518)


  Understanding the fundamental mechanism behind the success of deep neural
networks is one of the key challenges in the modern machine learning
literature. Despite numerous attempts, a solid theoretical analysis is yet to
be developed. In this paper, we develop a novel unified framework to reveal a
hidden regularization mechanism through the lens of convex optimization. We
first show that the training of multiple three-layer ReLU sub-networks with
weight decay regularization can be equivalently cast as a convex optimization
problem in a higher dimensional space, where sparsity is enforced via a group
$\ell_1$-norm regularization. Consequently, ReLU networks can be interpreted as
high dimensional feature selection methods. More importantly, we then prove
that the equivalent convex problem can be globally optimized by a standard
convex optimization solver with a polynomial-time complexity with respect to
the number of samples and data dimension when the width of the network is
fixed. Finally, we numerically validate our theoretical results via experiments
involving both synthetic and real datasets.

    

### [[2110.05528] Smoothed Separable Nonnegative Matrix Factorization](http://arxiv.org/abs/2110.05528)


  Given a set of data points belonging to the convex hull of a set of vertices,
a key problem in data analysis and machine learning is to estimate these
vertices in the presence of noise. Many algorithms have been developed under
the assumption that there is at least one nearby data point to each vertex; two
of the most widely used ones are vertex component analysis (VCA) and the
successive projection algorithm (SPA). This assumption is known as the
pure-pixel assumption in blind hyperspectral unmixing, and as the separability
assumption in nonnegative matrix factorization. More recently, Bhattacharyya
and Kannan (ACM-SIAM Symposium on Discrete Algorithms, 2020) proposed an
algorithm for learning a latent simplex (ALLS) that relies on the assumption
that there is more than one nearby data point for each vertex. In that
scenario, ALLS is probalistically more robust to noise than algorithms based on
the separability assumption. In this paper, inspired by ALLS, we propose
smoothed VCA (SVCA) and smoothed SPA (SSPA) that generalize VCA and SPA by
assuming the presence of several nearby data points to each vertex. We
illustrate the effectiveness of SVCA and SSPA over VCA, SPA and ALLS on
synthetic data sets, and on the unmixing of hyperspectral images.

    

### [[2110.05529] HUNTER: AI based Holistic Resource Management for Sustainable Cloud Computing](http://arxiv.org/abs/2110.05529)


  The worldwide adoption of cloud data centers (CDCs) has given rise to the
ubiquitous demand for hosting application services on the cloud. Further,
contemporary data-intensive industries have seen a sharp upsurge in the
resource requirements of modern applications. This has led to the provisioning
of an increased number of cloud servers, giving rise to higher energy
consumption and, consequently, sustainability concerns. Traditional heuristics
and reinforcement learning based algorithms for energy-efficient cloud resource
management address the scalability and adaptability related challenges to a
limited extent. Existing work often fails to capture dependencies across
thermal characteristics of hosts, resource consumption of tasks and the
corresponding scheduling decisions. This leads to poor scalability and an
increase in the compute resource requirements, particularly in environments
with non-stationary resource demands. To address these limitations, we propose
an artificial intelligence (AI) based holistic resource management technique
for sustainable cloud computing called HUNTER. The proposed model formulates
the goal of optimizing energy efficiency in data centers as a multi-objective
scheduling problem, considering three important models: energy, thermal and
cooling. HUNTER utilizes a Gated Graph Convolution Network as a surrogate model
for approximating the Quality of Service (QoS) for a system state and
generating optimal scheduling decisions. Experiments on simulated and physical
cloud environments using the CloudSim toolkit and the COSCO framework show that
HUNTER outperforms state-of-the-art baselines in terms of energy consumption,
SLA violation, scheduling time, cost and temperature by up to 12, 35, 43, 54
and 3 percent respectively.

    

### [[2110.05531] Study of Drug Assimilation in Human System using Physics Informed Neural Networks](http://arxiv.org/abs/2110.05531)


  Differential equations play a pivotal role in modern world ranging from
science, engineering, ecology, economics and finance where these can be used to
model many physical systems and processes. In this paper, we study two
mathematical models of a drug assimilation in the human system using Physics
Informed Neural Networks (PINNs). In the first model, we consider the case of
single dose of drug in the human system and in the second case, we consider the
course of this drug taken at regular intervals. We have used the compartment
diagram to model these cases. The resulting differential equations are solved
using PINN, where we employ a feed forward multilayer perceptron as function
approximator and the network parameters are tuned for minimum error. Further,
the network is trained by finding the gradient of the error function with
respect to the network parameters. We have employed DeepXDE, a python library
for PINNs, to solve the simultaneous first order differential equations
describing the two models of drug assimilation. The results show high degree of
accuracy between the exact solution and the predicted solution as much as the
resulting error reaches10^(-11) for the first model and 10^(-8) for the second
model. This validates the use of PINN in solving any dynamical system.

    

### [[2110.05532] Urban traffic dynamic rerouting framework: A DRL-based model with fog-cloud architecture](http://arxiv.org/abs/2110.05532)


  Past research and practice have demonstrated that dynamic rerouting framework
is effective in mitigating urban traffic congestion and thereby improve urban
travel efficiency. It has been suggested that dynamic rerouting could be
facilitated using emerging technologies such as fog-computing which offer
advantages of low-latency capabilities and information exchange between
vehicles and roadway infrastructure. To address this question, this study
proposes a two-stage model that combines GAQ (Graph Attention Network - Deep Q
Learning) and EBkSP (Entropy Based k Shortest Path) using a fog-cloud
architecture, to reroute vehicles in a dynamic urban environment and therefore
to improve travel efficiency in terms of travel speed. First, GAQ analyzes the
traffic conditions on each road and for each fog area, and then assigns a road
index based on the information attention from both local and neighboring areas.
Second, EBkSP assigns the route for each vehicle based on the vehicle priority
and route popularity. A case study experiment is carried out to investigate the
efficacy of the proposed model. At the model training stage, different methods
are used to establish the vehicle priorities, and their impact on the results
is assessed. Also, the proposed model is tested under various scenarios with
different ratios of rerouting and background (non-rerouting) vehicles. The
results demonstrate that vehicle rerouting using the proposed model can help
attain higher speed and reduces possibility of severe congestion. This result
suggests that the proposed model can be deployed by urban transportation
agencies for dynamic rerouting and ultimately, to reduce urban traffic
congestion.

    

### [[2110.05564] Scalable Traffic Signal Controls using Fog-Cloud Based Multiagent Reinforcement Learning](http://arxiv.org/abs/2110.05564)


  Optimizing traffic signal control (TSC) at intersections continues to pose a
challenging problem, particularly for large-scale traffic networks. It has been
shown in past research that it is feasible to optimize the operations of
individual TSC systems or a small number of such systems. However, it has been
computationally difficult to scale these solution approaches to large networks
partly due to the curse of dimensionality that is encountered as the number of
intersections increases. Fortunately, recent studies have recognized the
potential of exploiting advancements in deep and reinforcement learning to
address this problem, and some preliminary successes have been achieved in this
regard. However, facilitating such intelligent solution approaches may require
large amounts of infrastructural investments such as roadside units (RSUs) and
drones in order to ensure thorough connectivity across all intersections in
large networks, an investment that may be burdensome for agencies to undertake.
As such, this study builds on recent work to present a scalable TSC model that
may reduce the number of required enabling infrastructure. This is achieved
using graph attention networks (GATs) to serve as the neural network for deep
reinforcement learning, which aids in maintaining the graph topology of the
traffic network while disregarding any irrelevant or unnecessary information. A
case study is carried out to demonstrate the effectiveness of the proposed
model, and the results show much promise. The overall research outcome suggests
that by decomposing large networks using fog-nodes, the proposed fog-based
graphic RL (FG-RL) model can be easily applied to scale into larger traffic
networks.

    

### [[2110.05572] EchoVPR: Echo State Networks for Visual Place Recognition](http://arxiv.org/abs/2110.05572)


  Recognising previously visited locations is an important, but unsolved, task
in autonomous navigation. Current visual place recognition (VPR) benchmarks
typically challenge models to recover the position of a query image (or images)
from sequential datasets that include both spatial and temporal components.
Recently, Echo State Network (ESN) varieties have proven particularly powerful
at solving machine learning tasks that require spatio-temporal modelling. These
networks are simple, yet powerful neural architectures that -- exhibiting
memory over multiple time-scales and non-linear high-dimensional
representations -- can discover temporal relations in the data while still
maintaining linearity in the learning. In this paper, we present a series of
ESNs and analyse their applicability to the VPR problem. We report that the
addition of ESNs to pre-processed convolutional neural networks led to a
dramatic boost in performance in comparison to non-recurrent networks in four
standard benchmarks (GardensPoint, SPEDTest, ESSEX3IN1, Nordland) demonstrating
that ESNs are able to capture the temporal structure inherent in VPR problems.
Moreover, we show that ESNs can outperform class-leading VPR models which also
exploit the sequential dynamics of the data. Finally, our results demonstrate
that ESNs also improve generalisation abilities, robustness, and accuracy
further supporting their suitability to VPR applications.

    

### [[2110.05573] Spatial Data Mining of Public Transport Incidents reported in Social Media](http://arxiv.org/abs/2110.05573)


  Public transport agencies use social media as an essential tool for
communicating mobility incidents to passengers. However, while the short term,
day-to-day information about transport phenomena is usually posted in social
media with low latency, its availability is short term as the content is rarely
made an aggregated form. Social media communication of transport phenomena
usually lacks GIS annotations as most social media platforms do not allow
attaching non-POI GPS coordinates to posts. As a result, the analysis of
transport phenomena information is minimal. We collected three years of social
media posts of a polish public transport company with user comments. Through
exploration, we infer a six-class transport information typology. We
successfully build an information type classifier for social media posts,
detect stop names in posts, and relate them to GPS coordinates, obtaining a
spatial understanding of long-term aggregated phenomena. We show that our
approach enables citizen science and use it to analyze the impact of three
years of infrastructure incidents on passenger mobility, and the sentiment and
reaction scale towards each of the events. All these results are achieved for
Polish, an under-resourced language when it comes to spatial language
understanding, especially in social media contexts. To improve the situation,
we released two of our annotated data sets: social media posts with incident
type labels and matched stop names and social media comments with the annotated
sentiment. We also opensource the experimental codebase.

    

### [[2110.05587] Evaluation of Latent Space Disentanglement in the Presence of Interdependent Attributes](http://arxiv.org/abs/2110.05587)


  Controllable music generation with deep generative models has become
increasingly reliant on disentanglement learning techniques. However, current
disentanglement metrics, such as mutual information gap (MIG), are often
inadequate and misleading when used for evaluating latent representations in
the presence of interdependent semantic attributes often encountered in
real-world music datasets. In this work, we propose a dependency-aware
information metric as a drop-in replacement for MIG that accounts for the
inherent relationship between semantic attributes.

    

### [[2110.05588] DeepFilterNet: A Low Complexity Speech Enhancement Framework for Full-Band Audio based on Deep Filtering](http://arxiv.org/abs/2110.05588)


  Complex-valued processing has brought deep learning-based speech enhancement
and signal extraction to a new level. Typically, the process is based on a
time-frequency (TF) mask which is applied to a noisy spectrogram, while complex
masks (CM) are usually preferred over real-valued masks due to their ability to
modify the phase. Recent work proposed to use a complex filter instead of a
point-wise multiplication with a mask. This allows to incorporate information
from previous and future time steps exploiting local correlations within each
frequency band. In this work, we propose DeepFilterNet, a two stage speech
enhancement framework utilizing deep filtering. First, we enhance the spectral
envelope using ERB-scaled gains modeling the human frequency perception. The
second stage employs deep filtering to enhance the periodic components of
speech. Additionally to taking advantage of perceptual properties of speech, we
enforce network sparsity via separable convolutions and extensive grouping in
linear and recurrent layers to design a low complexity architecture. We further
show that our two stage deep filtering approach outperforms complex masks over
a variety of frequency resolutions and latencies and demonstrate convincing
performance compared to other state-of-the-art models.

    

### [[2110.05589] TTRS: Tinkoff Transactions Recommender System benchmark](http://arxiv.org/abs/2110.05589)


  Over the past decade, tremendous progress has been made in inventing new
RecSys methods. However, one of the fundamental problems of the RecSys research
community remains the lack of applied datasets and benchmarks with well-defined
evaluation rules and metrics to test these novel approaches. In this article,
we present the TTRS - Tinkoff Transactions Recommender System benchmark. This
financial transaction benchmark contains over 2 million interactions between
almost 10,000 users and more than 1,000 merchant brands over 14 months. To the
best of our knowledge, this is the first publicly available financial
transactions dataset. To make it more suitable for possible applications, we
provide a complete description of the data collection pipeline, its
preprocessing, and the resulting dataset statistics. We also present a
comprehensive comparison of the current popular RecSys methods on the
next-period recommendation task and conduct a detailed analysis of their
performance against various metrics and recommendation goals. Last but not
least, we also introduce Personalized Item-Frequencies-based Model (Re)Ranker -
PIFMR, a simple yet powerful approach that has proven to be the most effective
for the benchmarked tasks.

    

### [[2110.05597] Learning to Coordinate in Multi-Agent Systems: A Coordinated Actor-Critic Algorithm and Finite-Time Guarantees](http://arxiv.org/abs/2110.05597)


  Multi-agent reinforcement learning (MARL) has attracted much research
attention recently. However, unlike its single-agent counterpart, many
theoretical and algorithmic aspects of MARL have not been well-understood. In
this paper, we study the emergence of coordinated behavior by autonomous agents
using an actor-critic (AC) algorithm. Specifically, we propose and analyze a
class of coordinated actor-critic algorithms (CAC) in which individually
parametrized policies have a {\it shared} part (which is jointly optimized
among all agents) and a {\it personalized} part (which is only locally
optimized). Such kind of {\it partially personalized} policy allows agents to
learn to coordinate by leveraging peers' past experience and adapt to
individual tasks. The flexibility in our design allows the proposed MARL-CAC
algorithm to be used in a {\it fully decentralized} setting, where the agents
can only communicate with their neighbors, as well as a {\it federated}
setting, where the agents occasionally communicate with a server while
optimizing their (partially personalized) local models. Theoretically, we show
that under some standard regularity assumptions, the proposed MARL-CAC
algorithm requires $\mathcal{O}(\epsilon^{-\frac{5}{2}})$ samples to achieve an
$\epsilon$-stationary solution (defined as the solution whose squared norm of
the gradient of the objective function is less than $\epsilon$). To the best of
our knowledge, this work provides the first finite-sample guarantee for
decentralized AC algorithm with partially personalized policies.

    

### [[2110.05598] GCN-SE: Attention as Explainability for Node Classification in Dynamic Graphs](http://arxiv.org/abs/2110.05598)


  Graph Convolutional Networks (GCNs) are a popular method from graph
representation learning that have proved effective for tasks like node
classification tasks. Although typical GCN models focus on classifying nodes
within a static graph, several recent variants propose node classification in
dynamic graphs whose topologies and node attributes change over time, e.g.,
social networks with dynamic relationships, or literature citation networks
with changing co-authorships. These works, however, do not fully address the
challenge of flexibly assigning different importance to snapshots of the graph
at different times, which depending on the graph dynamics may have more or less
predictive power on the labels. We address this challenge by proposing a new
method, GCN-SE, that attaches a set of learnable attention weights to graph
snapshots at different times, inspired by Squeeze and Excitation Net (SE-Net).
We show that GCN-SE outperforms previously proposed node classification methods
on a variety of graph datasets. To verify the effectiveness of the attention
weight in determining the importance of different graph snapshots, we adapt
perturbation-based methods from the field of explainable machine learning to
graphical settings and evaluate the correlation between the attention weights
learned by GCN-SE and the importance of different snapshots over time. These
experiments demonstrate that GCN-SE can in fact identify different snapshots'
predictive power for dynamic node classification.

    

### [[2110.05607] Partial Variable Training for Efficient On-Device Federated Learning](http://arxiv.org/abs/2110.05607)


  This paper aims to address the major challenges of Federated Learning (FL) on
edge devices: limited memory and expensive communication. We propose a novel
method, called Partial Variable Training (PVT), that only trains a small subset
of variables on edge devices to reduce memory usage and communication cost.
With PVT, we show that network accuracy can be maintained by utilizing more
local training steps and devices, which is favorable for FL involving a large
population of devices. According to our experiments on two state-of-the-art
neural networks for speech recognition and two different datasets, PVT can
reduce memory usage by up to 1.9$\times$ and communication cost by up to
593$\times$ while attaining comparable accuracy when compared with full network
training.

    

### [[2110.05610] TSK Fuzzy System Towards Few Labeled Incomplete Multi-View Data Classification](http://arxiv.org/abs/2110.05610)


  Data collected by multiple methods or from multiple sources is called
multi-view data. To make full use of the multi-view data, multi-view learning
plays an increasingly important role. Traditional multi-view learning methods
rely on a large number of labeled and completed multi-view data. However, it is
expensive and time-consuming to obtain a large number of labeled multi-view
data in real-world applications. Moreover, multi-view data is often incomplete
because of data collection failures, self-deficiency, or other reasons.
Therefore, we may have to face the problem of fewer labeled and incomplete
multi-view data in real application scenarios. In this paper, a transductive
semi-supervised incomplete multi-view TSK fuzzy system modeling method
(SSIMV_TSK) is proposed to address these challenges. First, in order to
alleviate the dependency on labeled data and keep the model interpretable, the
proposed method integrates missing view imputation, pseudo label learning of
unlabeled data, and fuzzy system modeling into a single process to yield a
model with interpretable fuzzy rules. Then, two new mechanisms, i.e. the
bidirectional structural preservation of instance and label, as well as the
adaptive multiple alignment collaborative learning, are proposed to improve the
robustness of the model. The proposed method has the following distinctive
characteristics: 1) it can deal with the incomplete and few labeled multi-view
data simultaneously; 2) it integrates the missing view imputation and model
learning as a single process, which is more efficient than the traditional
two-step strategy; 3) attributed to the interpretable fuzzy inference rules,
this method is more interpretable. Experimental results on real datasets show
that the proposed method significantly outperforms the state-of-the-art
methods.

    

### [[2110.05614] Signal Processing on Cell Complexes](http://arxiv.org/abs/2110.05614)


  The processing of signals supported on non-Euclidean domains has attracted
large interest in the last years. Thus far, such non-Euclidean domains have
been abstracted primarily as graphs with signals supported on the nodes, though
recently the processing of signals on more general structures such as
simplicial complexes has also been considered. In this paper, we give an
introduction to signal processing on (abstract) regular cell complexes, which
provide a unifying framework encompassing graphs, simplicial complexes, cubical
complexes and various meshes as special cases. We discuss how appropriate Hodge
Laplacians for these cell complexes can be derived. These Hodge Laplacians
enable the construction of convolutional filters, which can be employed in
linear filtering and non-linear filtering via neural networks defined on cell
complexes.

    

### [[2110.05622] Review of Kernel Learning for Intra-Hour Solar Forecasting with Infrared Sky Images and Cloud Dynamic Feature Extraction](http://arxiv.org/abs/2110.05622)


  The uncertainty of the energy generated by photovoltaic systems incurs an
additional cost for a guaranteed, reliable supply of energy (i.e., energy
storage). This investigation aims to decrease the additional cost by
introducing probabilistic multi-task intra-hour solar forecasting (feasible in
real time applications) to increase the penetration of photovoltaic systems in
power grids. The direction of moving clouds is estimated in consecutive
sequences of sky images by extracting features of cloud dynamics with the
objective of forecasting the global solar irradiance that reaches photovoltaic
systems. The sky images are acquired using a low-cost infrared sky imager
mounted on a solar tracker. The solar forecasting algorithm is based on kernel
learning methods, and uses the clear sky index as predictor and features
extracted from clouds as feature vectors. The proposed solar forecasting
algorithm achieved 16.45\% forecasting skill 8 minutes ahead with a resolution
of 15 seconds. In contrast, previous work reached 15.4\% forecasting skill with
the resolution of 1 minute. Therefore, this solar forecasting algorithm
increases the performances with respect to the state-of-the-art, providing grid
operators with the capability of managing the inherent uncertainties of power
grids with a high penetration of photovoltaic systems.

    

### [[2110.05626] Parameterizing Activation Functions for Adversarial Robustness](http://arxiv.org/abs/2110.05626)


  Deep neural networks are known to be vulnerable to adversarially perturbed
inputs. A commonly used defense is adversarial training, whose performance is
influenced by model capacity. While previous works have studied the impact of
varying model width and depth on robustness, the impact of increasing capacity
by using learnable parametric activation functions (PAFs) has not been studied.
We study how using learnable PAFs can improve robustness in conjunction with
adversarial training. We first ask the question: how should we incorporate
parameters into activation functions to improve robustness? To address this, we
analyze the direct impact of activation shape on robustness through PAFs and
observe that activation shapes with positive outputs on negative inputs and
with high finite curvature can increase robustness. We combine these properties
to create a new PAF, which we call Parametric Shifted Sigmoidal Linear Unit
(PSSiLU). We then combine PAFs (including PReLU, PSoftplus and PSSiLU) with
adversarial training and analyze robust performance. We find that PAFs optimize
towards activation shape properties found to directly affect robustness.
Additionally, we find that while introducing only 1-2 learnable parameters into
the network, smooth PAFs can significantly increase robustness over ReLU. For
instance, when trained on CIFAR-10 with additional synthetic data, PSSiLU
improves robust accuracy by 4.54% over ReLU on ResNet-18 and 2.69% over ReLU on
WRN-28-10 in the $\ell_{\infty}$ threat model while adding only 2 additional
parameters into the network architecture. The PSSiLU WRN-28-10 model achieves
61.96% AutoAttack accuracy, improving over the state-of-the-art robust accuracy
on RobustBench (Croce et al., 2020).

    

### [[2110.05635] Real-time EEG-based Emotion Recognition using Discrete Wavelet Transforms on Full and Reduced Channel Signals](http://arxiv.org/abs/2110.05635)


  Real-time EEG-based Emotion Recognition (EEG-ER) with consumer-grade EEG
devices involves classification of emotions using a reduced number of channels.
These devices typically provide only four or five channels, unlike the high
number of channels (32 or more) typically used in most current state-of-the-art
research. In this work we propose to use Discrete Wavelet Transforms (DWT) to
extract time-frequency domain features, and we use time-windows of a few
seconds to perform EEG-ER classification. This technique can be used in
real-time, as opposed to post-hoc on the full session data. We also apply
baseline removal preprocessing, developed in prior research, to our proposed
DWT Entropy and Energy features, which improves classification accuracy
significantly. We consider two different classifier architectures, a 3D
Convolutional Neural Network (3D CNN) and a Support Vector Machine (SVM). We
evaluate both models on subject-independent and subject dependent setups to
classify the Valence and Arousal dimensions of an individual's emotional state.
We test them on both the full 32-channel data provided by the DEAP dataset, and
also a reduced 5-channel extract of the same dataset. The SVM model performs
best on all the presented scenarios, achieving an accuracy of 95.32% on Valence
and 95.68% on Arousal for the full 32-channel subject-dependent case, beating
prior real-time EEG-ER subject-dependent benchmarks. On the subject-independent
case an accuracy of 80.70% on Valence and 81.41% on Arousal was also obtained.
Reducing the input data to 5 channels only degrades the accuracy by an average
of 3.54% across all scenarios, making this model appropriate for use with more
accessible low-end EEG devices.

    

### [[2110.05636] CAPITAL: Optimal Subgroup Identification via Constrained Policy Tree Search](http://arxiv.org/abs/2110.05636)


  Personalized medicine, a paradigm of medicine tailored to a patient's
characteristics, is an increasingly attractive field in health care. An
important goal of personalized medicine is to identify a subgroup of patients,
based on baseline covariates, that benefits more from the targeted treatment
than other comparative treatments. Most of the current subgroup identification
methods only focus on obtaining a subgroup with an enhanced treatment effect
without paying attention to subgroup size. Yet, a clinically meaningful
subgroup learning approach should identify the maximum number of patients who
can benefit from the better treatment. In this paper, we present an optimal
subgroup selection rule (SSR) that maximizes the number of selected patients,
and in the meantime, achieves the pre-specified clinically meaningful mean
outcome, such as the average treatment effect. We derive two equivalent
theoretical forms of the optimal SSR based on the contrast function that
describes the treatment-covariates interaction in the outcome. We further
propose a ConstrAined PolIcy Tree seArch aLgorithm (CAPITAL) to find the
optimal SSR within the interpretable decision tree class. The proposed method
is flexible to handle multiple constraints that penalize the inclusion of
patients with negative treatment effects, and to address time to event data
using the restricted mean survival time as the clinically interesting mean
outcome. Extensive simulations, comparison studies, and real data applications
are conducted to demonstrate the validity and utility of our method.

    

### [[2110.05645] A global convergence theory for deep ReLU implicit networks via over-parameterization](http://arxiv.org/abs/2110.05645)


  Implicit deep learning has received increasing attention recently due to the
fact that it generalizes the recursive prediction rules of many commonly used
neural network architectures. Its prediction rule is provided implicitly based
on the solution of an equilibrium equation. Although a line of recent empirical
studies has demonstrated its superior performances, the theoretical
understanding of implicit neural networks is limited. In general, the
equilibrium equation may not be well-posed during the training. As a result,
there is no guarantee that a vanilla (stochastic) gradient descent (SGD)
training nonlinear implicit neural networks can converge. This paper fills the
gap by analyzing the gradient flow of Rectified Linear Unit (ReLU) activated
implicit neural networks. For an $m$-width implicit neural network with ReLU
activation and $n$ training samples, we show that a randomly initialized
gradient descent converges to a global minimum at a linear rate for the square
loss function if the implicit neural network is \textit{over-parameterized}. It
is worth noting that, unlike existing works on the convergence of (S)GD on
finite-layer over-parameterized neural networks, our convergence results hold
for implicit neural networks, where the number of layers is \textit{infinite}.

    

### [[2110.05649] Learned Robust PCA: A Scalable Deep Unfolding Approach for High-Dimensional Outlier Detection](http://arxiv.org/abs/2110.05649)


  Robust principal component analysis (RPCA) is a critical tool in modern
machine learning, which detects outliers in the task of low-rank matrix
reconstruction. In this paper, we propose a scalable and learnable non-convex
approach for high-dimensional RPCA problems, which we call Learned Robust PCA
(LRPCA). LRPCA is highly efficient, and its free parameters can be effectively
learned to optimize via deep unfolding. Moreover, we extend deep unfolding from
finite iterations to infinite iterations via a novel
feedforward-recurrent-mixed neural network model. We establish the recovery
guarantee of LRPCA under mild assumptions for RPCA. Numerical experiments show
that LRPCA outperforms the state-of-the-art RPCA algorithms, such as ScaledGD
and AltProj, on both synthetic datasets and real-world applications.

    

### [[2110.05651] Learning with Algorithmic Supervision via Continuous Relaxations](http://arxiv.org/abs/2110.05651)


  The integration of algorithmic components into neural architectures has
gained increased attention recently, as it allows training neural networks with
new forms of supervision such as ordering constraints or silhouettes instead of
using ground truth labels. Many approaches in the field focus on the continuous
relaxation of a specific task and show promising results in this context. But
the focus on single tasks also limits the applicability of the proposed
concepts to a narrow range of applications. In this work, we build on those
ideas to propose an approach that allows to integrate algorithms into
end-to-end trainable neural network architectures based on a general
approximation of discrete conditions. To this end, we relax these conditions in
control structures such as conditional statements, loops, and indexing, so that
resulting algorithms are smoothly differentiable. To obtain meaningful
gradients, each relevant variable is perturbed via logistic distributions and
the expectation value under this perturbation is approximated. We evaluate the
proposed continuous relaxation model on four challenging tasks and show that it
can keep up with relaxations specifically designed for each individual task.

    

### [[2110.05661] BotNet Detection On Social Media](http://arxiv.org/abs/2110.05661)


  Given the popularity of social media and the notion of it being a platform
encouraging free speech, it has become an open playground for user (bot)
accounts trying to manipulate other users using these platforms. Social bots
not only learn human conversations, manners, and presence but also manipulate
public opinion, act as scammers, manipulate stock markets, etc. There has been
evidence of bots manipulating the election results which can be a great threat
to the whole nation and hence the whole world. So identification and prevention
of such campaigns that release or create the bots have become critical to
tackling it at its source of origin. Our goal is to leverage semantic web
mining techniques to identify fake bots or accounts involved in these
activities.

    

### [[2110.05667] Why Lottery Ticket Wins? A Theoretical Perspective of Sample Complexity on Pruned Neural Networks](http://arxiv.org/abs/2110.05667)


  The \textit{lottery ticket hypothesis} (LTH) states that learning on a
properly pruned network (the \textit{winning ticket}) improves test accuracy
over the original unpruned network. Although LTH has been justified empirically
in a broad range of deep neural network (DNN) involved applications like
computer vision and natural language processing, the theoretical validation of
the improved generalization of a winning ticket remains elusive. To the best of
our knowledge, our work, for the first time, characterizes the performance of
training a pruned neural network by analyzing the geometric structure of the
objective function and the sample complexity to achieve zero generalization
error. We show that the convex region near a desirable model with guaranteed
generalization enlarges as the neural network model is pruned, indicating the
structural importance of a winning ticket. Moreover, when the algorithm for
training a pruned neural network is specified as an (accelerated) stochastic
gradient descent algorithm, we theoretically show that the number of samples
required for achieving zero generalization error is proportional to the number
of the non-pruned weights in the hidden layer. With a fixed number of samples,
training a pruned neural network enjoys a faster convergence rate to the
desired model than training the original unpruned one, providing a formal
justification of the improved generalization of the winning ticket. Our
theoretical results are acquired from learning a pruned neural network of one
hidden layer, while experimental results are further provided to justify the
implications in pruning multi-layer neural networks.

    

### [[2110.05668] NAS-Bench-360: Benchmarking Diverse Tasks for Neural Architecture Search](http://arxiv.org/abs/2110.05668)


  Most existing neural architecture search (NAS) benchmarks and algorithms
prioritize performance on well-studied tasks, e.g., image classification on
CIFAR and ImageNet. This makes the applicability of NAS approaches in more
diverse areas inadequately understood. In this paper, we present NAS-Bench-360,
a benchmark suite for evaluating state-of-the-art NAS methods for convolutional
neural networks (CNNs). To construct it, we curate a collection of ten tasks
spanning a diverse array of application domains, dataset sizes, problem
dimensionalities, and learning objectives. By carefully selecting tasks that
can both interoperate with modern CNN-based search methods but that are also
far-afield from their original development domain, we can use NAS-Bench-360 to
investigate the following central question: do existing state-of-the-art NAS
methods perform well on diverse tasks? Our experiments show that a modern NAS
procedure designed for image classification can indeed find good architectures
for tasks with other dimensionalities and learning objectives; however, the
same method struggles against more task-specific methods and performs
catastrophically poorly on classification in non-vision domains. The case for
NAS robustness becomes even more dire in a resource-constrained setting, where
a recent NAS method provides little-to-no benefit over much simpler baselines.
These results demonstrate the need for a benchmark such as NAS-Bench-360 to
help develop NAS approaches that work well on a variety of tasks, a crucial
component of a truly robust and automated pipeline. We conclude with a
demonstration of the kind of future research our suite of tasks will enable.
All data and code is made publicly available.

    

### [[2110.05671] Predicting the Stereoselectivity of Chemical Transformations by Machine Learning](http://arxiv.org/abs/2110.05671)


  Stereoselective reactions (both chemical and enzymatic reactions) have been
essential for origin of life, evolution, human biology and medicine. Since late
1960s, there have been numerous successes in the exciting new frontier of
asymmetric catalysis. However, most industrial and academic asymmetric
catalysis nowadays do follow the trial-and-error model, since the energetic
difference for success or failure in asymmetric catalysis is incredibly small.
Our current understanding about stereoselective reactions is mostly qualitative
that stereoselectivity arises from differences in steric effects and electronic
effects in multiple competing mechanistic pathways. Quantitatively
understanding and modulating the stereoselectivity of for a given chemical
reaction still remains extremely difficult. As a proof of principle, we herein
present a novel machine learning technique, which combines a LASSO model and
two Random Forest model via two Gaussian Mixture models, for quantitatively
predicting stereoselectivity of chemical reactions. Compared to the recent
ground-breaking approach [1], our approach is able to capture interactions
between features and exploit complex data distributions, which are important
for predicting stereoselectivity. Experimental results on a recently published
dataset demonstrate that our approach significantly outperform [1]. The insight
obtained from our results provide a solid foundation for further exploration of
other synthetically valuable yet mechanistically intriguing stereoselective
reactions.

    

### [[2110.05674] Deviance Matrix Factorization](http://arxiv.org/abs/2110.05674)


  We investigate a general matrix factorization for deviance-based losses,
extending the ubiquitous singular value decomposition beyond squared error
loss. While similar approaches have been explored before, here we propose an
efficient algorithm that is flexible enough to allow for structural zeros and
entry weights. Moreover, we provide theoretical support for these
decompositions by (i) showing strong consistency under a generalized linear
model setup, (ii) checking the adequacy of a chosen exponential family via a
generalized Hosmer-Lemeshow test, and (iii) determining the rank of the
decomposition via a maximum eigenvalue gap method. To further support our
findings, we conduct simulation studies to assess robustness to decomposition
assumptions and extensive case studies using benchmark datasets from image face
recognition, natural language processing, network analysis, and biomedical
studies. Our theoretical and empirical results indicate that the proposed
decomposition is more flexible, general, and can provide improved performance
when compared to traditional methods.

    

### [[2110.05679] Large Language Models Can Be Strong Differentially Private Learners](http://arxiv.org/abs/2110.05679)


  Differentially Private (DP) learning has seen limited success for building
large deep learning models of text, and attempts at straightforwardly applying
Differentially Private Stochastic Gradient Descent (DP-SGD) to NLP tasks have
resulted in large performance drops and high computational overhead. We show
that this performance drop can be mitigated with (1) the use of large
pretrained models; (2) hyperparameters that suit DP optimization; and (3)
fine-tuning objectives aligned with the pretraining procedure. With these
factors set right, we obtain private NLP models that outperform
state-of-the-art private training approaches and strong non-private baselines
-- by directly fine-tuning pretrained models with DP optimization on
moderately-sized corpora. To address the computational challenge of running
DP-SGD with large Transformers, we propose a memory saving technique that
allows clipping in DP-SGD to run without instantiating per-example gradients
for any layer in the model. The technique enables privately training
Transformers with almost the same memory cost as non-private training at a
modest run-time overhead. Contrary to conventional wisdom that DP optimization
fails at learning high-dimensional models (due to noise that scales with
dimension) empirical results reveal that private learning with pretrained
models tends to not suffer from dimension-dependent performance degradation.

    

### [[2110.05682] Provably Efficient Reinforcement Learning in Decentralized General-Sum Markov Games](http://arxiv.org/abs/2110.05682)


  This paper addresses the problem of learning an equilibrium efficiently in
general-sum Markov games through decentralized multi-agent reinforcement
learning. Given the fundamental difficulty of calculating a Nash equilibrium
(NE), we instead aim at finding a coarse correlated equilibrium (CCE), a
solution concept that generalizes NE by allowing possible correlations among
the agents' strategies. We propose an algorithm in which each agent
independently runs optimistic V-learning (a variant of Q-learning) to
efficiently explore the unknown environment, while using a stabilized online
mirror descent (OMD) subroutine for policy updates. We show that the agents can
find an $\epsilon$-approximate CCE in at most $\widetilde{O}( H^6S A
/\epsilon^2)$ episodes, where $S$ is the number of states, $A$ is the size of
the largest individual action space, and $H$ is the length of an episode. This
appears to be the first sample complexity result for learning in generic
general-sum Markov games. Our results rely on a novel investigation of an
anytime high-probability regret bound for OMD with a dynamic learning rate and
weighted regret, which would be of independent interest. One key feature of our
algorithm is that it is fully \emph{decentralized}, in the sense that each
agent has access to only its local information, and is completely oblivious to
the presence of others. This way, our algorithm can readily scale up to an
arbitrary number of agents, without suffering from the exponential dependence
on the number of agents.

    

### [[2110.05688] Inclusive Design: Accessibility Settings for People with Cognitive Disabilities](http://arxiv.org/abs/2110.05688)


  The advancement of technology has progressed faster than any other field in
the world and with the development of these new technologies, it is important
to make sure that these tools can be used by everyone, including people with
disabilities. Accessibility options in computing devices help ensure that
everyone has the same access to advanced technologies. Unfortunately, for those
who require more unique and sometimes challenging accommodations, such as
people with Amyotrophic lateral sclerosis ( ALS), the most commonly used
accessibility features are simply not enough. While assistive technology for
those with ALS does exist, it requires multiple peripheral devices that can
become quite expensive collectively. The purpose of this paper is to suggest a
more affordable and readily available option for ALS assistive technology that
can be implemented on a smartphone or tablet.

    

### [[2110.05695] The Mirrornet : Learning Audio Synthesizer Controls Inspired by Sensorimotor Interaction](http://arxiv.org/abs/2110.05695)


  Experiments to understand the sensorimotor neural interactions in the human
cortical speech system support the existence of a bidirectional flow of
interactions between the auditory and motor regions. Their key function is to
enable the brain to 'learn' how to control the vocal tract for speech
production. This idea is the impetus for the recently proposed "MirrorNet", a
constrained autoencoder architecture. In this paper, the MirrorNet is applied
to learn, in an unsupervised manner, the controls of a specific audio
synthesizer (DIVA) to produce melodies only from their auditory spectrograms.
The results demonstrate how the MirrorNet discovers the synthesizer parameters
to generate the melodies that closely resemble the original and those of unseen
melodies, and even determine the best set parameters to approximate renditions
of complex piano melodies generated by a different synthesizer. This
generalizability of the MirrorNet illustrates its potential to discover from
sensory data the controls of arbitrary motor-plants such as autonomous
vehicles.

    

### [[2110.05702] Auditing Robot Learning for Safety and Compliance during Deployment](http://arxiv.org/abs/2110.05702)


  Robots of the future are going to exhibit increasingly human-like and
super-human intelligence in a myriad of different tasks. They are also likely
going to fail and be incompliant with human preferences in increasingly subtle
ways. Towards the goal of achieving autonomous robots, the robot learning
community has made rapid strides in applying machine learning techniques to
train robots through data and interaction. This makes the study of how best to
audit these algorithms for checking their compatibility with humans, pertinent
and urgent. In this paper, we draw inspiration from the AI Safety and Alignment
communities and make the case that we need to urgently consider ways in which
we can best audit our robot learning algorithms to check for failure modes, and
ensure that when operating autonomously, they are indeed behaving in ways that
the human algorithm designers intend them to. We believe that this is a
challenging problem that will require efforts from the entire robot learning
community, and do not attempt to provide a concrete framework for auditing.
Instead, we outline high-level guidance and a possible approach towards
formulating this framework which we hope will serve as a useful starting point
for thinking about auditing in the context of robot learning.

    

### [[2110.05706] Deep Fusion Prior for Multi-Focus Image Super Resolution Fusion](http://arxiv.org/abs/2110.05706)


  This paper unifies the multi-focus images fusion (MFIF) and blind super
resolution (SR) problems as the multi-focus image super resolution fusion
(MFISRF) task, and proposes a novel unified dataset-free unsupervised framework
named deep fusion prior (DFP) to address such MFISRF task. DFP consists of
SKIPnet network, DoubleReblur focus measurement tactic, decision embedding
module and loss functions. In particular, DFP can obtain MFISRF only from two
low-resolution inputs without any extent dataset; SKIPnet implementing
unsupervised learning via deep image prior is an end-to-end generated network
acting as the engine of DFP; DoubleReblur is used to determine the primary
decision map without learning but based on estimated PSF and Gaussian kernels
convolution; decision embedding module optimizes the decision map via learning;
and DFP losses composed of content loss, joint gradient loss and gradient limit
loss can obtain high-quality MFISRF results robustly. Experiments have proved
that our proposed DFP approaches and even outperforms those state-of-art MFIF
and SR method combinations. Additionally, DFP is a general framework, thus its
networks and focus measurement tactics can be continuously updated to further
improve the MFISRF performance. DFP codes are open source and will be available
soon at this http URL.

    

### [[2110.05707] Decentralized Cooperative Multi-Agent Reinforcement Learning with Exploration](http://arxiv.org/abs/2110.05707)


  Many real-world applications of multi-agent reinforcement learning (RL), such
as multi-robot navigation and decentralized control of cyber-physical systems,
involve the cooperation of agents as a team with aligned objectives. We study
multi-agent RL in the most basic cooperative setting -- Markov teams -- a class
of Markov games where the cooperating agents share a common reward. We propose
an algorithm in which each agent independently runs stage-based V-learning (a
Q-learning style algorithm) to efficiently explore the unknown environment,
while using a stochastic gradient descent (SGD) subroutine for policy updates.
We show that the agents can learn an $\epsilon$-approximate Nash equilibrium
policy in at most $\propto\widetilde{O}(1/\epsilon^4)$ episodes. Our results
advocate the use of a novel \emph{stage-based} V-learning approach to create a
stage-wise stationary environment. We also show that under certain smoothness
assumptions of the team, our algorithm can achieve a nearly \emph{team-optimal}
Nash equilibrium. Simulation results corroborate our theoretical findings. One
key feature of our algorithm is being \emph{decentralized}, in the sense that
each agent has access to only the state and its local actions, and is even
\emph{oblivious} to the presence of the other agents. Neither communication
among teammates nor coordination by a central controller is required during
learning. Hence, our algorithm can readily generalize to an arbitrary number of
agents, without suffering from the exponential dependence on the number of
agents.

    

### [[2110.05712] DecGAN: Decoupling Generative Adversarial Network detecting abnormal neural circuits for Alzheimer's disease](http://arxiv.org/abs/2110.05712)


  One of the main reasons for Alzheimer's disease (AD) is the disorder of some
neural circuits. Existing methods for AD prediction have achieved great
success, however, detecting abnormal neural circuits from the perspective of
brain networks is still a big challenge. In this work, a novel decoupling
generative adversarial network (DecGAN) is proposed to detect abnormal neural
circuits for AD. Concretely, a decoupling module is designed to decompose a
brain network into two parts: one part is composed of a few sparse graphs which
represent the neural circuits largely determining the development of AD; the
other part is a supplement graph, whose influence on AD can be ignored.
Furthermore, the adversarial strategy is utilized to guide the decoupling
module to extract the feature more related to AD. Meanwhile, by encoding the
detected neural circuits to hypergraph data, an analytic module associated with
the hyperedge neurons algorithm is designed to identify the neural circuits.
More importantly, a novel sparse capacity loss based on the spatial-spectral
hypergraph similarity is developed to minimize the intrinsic topological
distribution of neural circuits, which can significantly improve the accuracy
and robustness of the proposed model. Experimental results demonstrate that the
proposed model can effectively detect the abnormal neural circuits at different
stages of AD, which is helpful for pathological study and early treatment.

    

### [[2110.05721] Action-Sufficient State Representation Learning for Control with Structural Constraints](http://arxiv.org/abs/2110.05721)


  Perceived signals in real-world scenarios are usually high-dimensional and
noisy, and finding and using their representation that contains essential and
sufficient information required by downstream decision-making tasks will help
improve computational efficiency and generalization ability in the tasks. In
this paper, we focus on partially observable environments and propose to learn
a minimal set of state representations that capture sufficient information for
decision-making, termed \textit{Action-Sufficient state Representations}
(ASRs). We build a generative environment model for the structural
relationships among variables in the system and present a principled way to
characterize ASRs based on structural constraints and the goal of maximizing
cumulative reward in policy learning. We then develop a structured sequential
Variational Auto-Encoder to estimate the environment model and extract ASRs.
Our empirical results on CarRacing and VizDoom demonstrate a clear advantage of
learning and using ASRs for policy learning. Moreover, the estimated
environment model and ASRs allow learning behaviors from imagined outcomes in
the compact latent space to improve sample efficiency.

    

### [[2110.05724] Dare not to Ask: Problem-Dependent Guarantees for Budgeted Bandits](http://arxiv.org/abs/2110.05724)


  We consider a stochastic multi-armed bandit setting where feedback is limited
by a (possibly time-dependent) budget, and reward must be actively inquired for
it to be observed. Previous works on this setting assumed a strict feedback
budget and focused on not violating this constraint while providing
problem-independent regret guarantees. In this work, we provide
problem-dependent guarantees on both the regret and the asked feedback. In
particular, we derive problem-dependent lower bounds on the required feedback
and show that there is a fundamental difference between problems with a unique
and multiple optimal arms. Furthermore, we present a new algorithm called
BuFALU for which we derive problem-dependent regret and cumulative feedback
bounds. Notably, we show that BuFALU naturally adapts to the number of optimal
arms.

    

### [[2110.05728] Rethinking the Spatial Route Prior in Vision-and-Language Navigation](http://arxiv.org/abs/2110.05728)


  Vision-and-language navigation (VLN) is a trending topic which aims to
navigate an intelligent agent to an expected position through natural language
instructions. This work addresses the task of VLN from a previously-ignored
aspect, namely the spatial route prior of the navigation scenes. A critically
enabling innovation of this work is explicitly considering the spatial route
prior under several different VLN settings. In a most information-rich case of
knowing environment maps and admitting shortest-path prior, we observe that
given an origin-destination node pair, the internal route can be uniquely
determined. Thus, VLN can be effectively formulated as an ordinary
classification problem over all possible destination nodes in the scenes.
Furthermore, we relax it to other more general VLN settings, proposing a
sequential-decision variant (by abandoning the shortest-path route prior) and
an explore-and-exploit scheme (for addressing the case of not knowing the
environment maps) that curates a compact and informative sub-graph to exploit.
As reported by [34], the performance of VLN methods has been stuck at a plateau
in past two years. Even with increased model complexity, the state-of-the-art
success rate on R2R validation-unseen set has stayed around 62% for single-run
and 73% for beam-search with model-ensemble. We have conducted comprehensive
evaluations on both R2R and R4R, and surprisingly found that utilizing the
spatial route priors may be the key of breaking above-mentioned performance
ceiling. For example, on R2R validation-unseen set, when the number of discrete
nodes explored is about 40, our single-model success rate reaches 73%, and
increases to 78% if a Speaker model is ensembled, which significantly outstrips
previous state-of-the-art VLN-BERT with 3 models ensembled.

    

### [[2110.05732] Guided-GAN: Adversarial Representation Learning for Activity Recognition with Wearables](http://arxiv.org/abs/2110.05732)


  Human activity recognition (HAR) is an important research field in ubiquitous
computing where the acquisition of large-scale labeled sensor data is tedious,
labor-intensive and time consuming. State-of-the-art unsupervised remedies
investigated to alleviate the burdens of data annotations in HAR mainly explore
training autoencoder frameworks. In this paper: we explore generative
adversarial network (GAN) paradigms to learn unsupervised feature
representations from wearable sensor data; and design a new GAN
framework-Geometrically-Guided GAN or Guided-GAN-for the task. To demonstrate
the effectiveness of our formulation, we evaluate the features learned by
Guided-GAN in an unsupervised manner on three downstream classification
benchmarks. Our results demonstrate Guided-GAN to outperform existing
unsupervised approaches whilst closely approaching the performance with fully
supervised learned representations. The proposed approach paves the way to
bridge the gap between unsupervised and supervised human activity recognition
whilst helping to reduce the cost of human data annotation tasks.

    

### [[2110.05734] Learning Efficient Multi-Agent Cooperative Visual Exploration](http://arxiv.org/abs/2110.05734)


  We consider the task of visual indoor exploration with multiple agents, where
the agents need to cooperatively explore the entire indoor region using as few
steps as possible. Classical planning-based methods often suffer from
particularly expensive computation at each inference step and a limited
expressiveness of cooperation strategy. By contrast, reinforcement learning
(RL) has become a trending paradigm for tackling this challenge due to its
modeling capability of arbitrarily complex strategies and minimal inference
overhead. We extend the state-of-the-art single-agent RL solution, Active
Neural SLAM (ANS), to the multi-agent setting by introducing a novel RL-based
global-goal planner, Spatial Coordination Planner (SCP), which leverages
spatial information from each individual agent in an end-to-end manner and
effectively guides the agents to navigate towards different spatial goals with
high exploration efficiency. SCP consists of a transformer-based relation
encoder to capture intra-agent interactions and a spatial action decoder to
produce accurate goals. In addition, we also implement a few multi-agent
enhancements to process local information from each agent for an aligned
spatial representation and more precise planning. Our final solution,
Multi-Agent Active Neural SLAM (MAANS), combines all these techniques and
substantially outperforms 4 different planning-based methods and various RL
baselines in the photo-realistic physical testbed, Habitat.

    

### [[2110.05740] Temporal Abstraction in Reinforcement Learning with the Successor Representation](http://arxiv.org/abs/2110.05740)


  Reasoning at multiple levels of temporal abstraction is one of the key
attributes of intelligence. In reinforcement learning, this is often modeled
through temporally extended courses of actions called options. Options allow
agents to make predictions and to operate at different levels of abstraction
within an environment. Nevertheless, approaches based on the options framework
often start with the assumption that a reasonable set of options is known
beforehand. When this is not the case, there are no definitive answers for
which options one should consider. In this paper, we argue that the successor
representation (SR), which encodes states based on the pattern of state
visitation that follows them, can be seen as a natural substrate for the
discovery and use of temporal abstractions. To support our claim, we take a big
picture view of recent results, showing how the SR can be used to discover
options that facilitate either temporally-extended exploration or planning. We
cast these results as instantiations of a general framework for option
discovery in which the agent's representation is used to identify useful
options, which are then used to further improve its representation. This
results in a virtuous, never-ending, cycle in which both the representation and
the options are constantly refined based on each other. Beyond option discovery
itself, we discuss how the SR allows us to augment a set of options into a
combinatorially large counterpart without additional learning. This is achieved
through the combination of previously learned options. Our empirical evaluation
focuses on options discovered for temporally-extended exploration and on the
use of the SR to combine them. The results of our experiments shed light on
design decisions involved in the definition of options and demonstrate the
synergy of different methods based on the SR, such as eigenoptions and the
option keyboard.

    

### [[2110.05745] VarArray: Array-Geometry-Agnostic Continuous Speech Separation](http://arxiv.org/abs/2110.05745)


  Continuous speech separation using a microphone array was shown to be
promising in dealing with the speech overlap problem in natural conversation
transcription. This paper proposes VarArray, an array-geometry-agnostic speech
separation neural network model. The proposed model is applicable to any number
of microphones without retraining while leveraging the nonlinear correlation
between the input channels. The proposed method adapts different elements that
were proposed before separately, including transform-average-concatenate,
conformer speech separation, and inter-channel phase differences, and combines
them in an efficient and cohesive way. Large-scale evaluation was performed
with two real meeting transcription tasks by using a fully developed
transcription system requiring no prior knowledge such as reference
segmentations, which allowed us to measure the impact that the continuous
speech separation system could have in realistic settings. The proposed model
outperformed a previous approach to array-geometry-agnostic modeling for all of
the geometry configurations considered, achieving asclite-based
speaker-agnostic word error rates of 17.5% and 20.4% for the AMI development
and evaluation sets, respectively, in the end-to-end setting using no
ground-truth segmentations.

    

### [[2110.05753] Predicting the Efficiency of CO$_2$ Sequestering by Metal Organic Frameworks Through Machine Learning Analysis of Structural and Electronic Properties](http://arxiv.org/abs/2110.05753)


  Due the alarming rate of climate change, the implementation of efficient
CO$_2$ capture has become crucial. This project aims to create an algorithm
that predicts the uptake of CO$_2$ adsorbing Metal-Organic Frameworks (MOFs) by
using Machine Learning. These values will in turn gauge the efficiency of these
MOFs and provide scientists who are looking to maximize the uptake a way to
know whether or not the MOF is worth synthesizing. This algorithm will save
resources such as time and equipment as scientists will be able to disregard
hypothetical MOFs with low efficiencies. In addition, this paper will also
highlight the most important features within the data set. This research will
contribute to enable the rapid synthesis of CO$_2$ adsorbing MOFs.

    

### [[2110.05754] Deep Federated Learning for Autonomous Driving](http://arxiv.org/abs/2110.05754)


  Autonomous driving is an active research topic in both academia and industry.
However, most of the existing solutions focus on improving the accuracy by
training learnable models with centralized large-scale data. Therefore, these
methods do not take into account the user's privacy. In this paper, we present
a new approach to learn autonomous driving policy while respecting privacy
concerns. We propose a peer-to-peer Deep Federated Learning (DFL) approach to
train deep architectures in a fully decentralized manner and remove the need
for central orchestration. We design a new Federated Autonomous Driving network
(FADNet) that can improve the model stability, ensure convergence, and handle
imbalanced data distribution problems while is being trained with federated
learning methods. Intensively experimental results on three datasets show that
our approach with FADNet and DFL achieves superior accuracy compared with other
recent methods. Furthermore, our approach can maintain privacy by not
collecting user data to a central server.

    

### [[2110.05765] Music Sentiment Transfer](http://arxiv.org/abs/2110.05765)


  Music sentiment transfer is a completely novel task. Sentiment transfer is a
natural evolution of the heavily-studied style transfer task, as sentiment
transfer is rooted in applying the sentiment of a source to be the new
sentiment for a target piece of media; yet compared to style transfer,
sentiment transfer has been only scantily studied on images. Music sentiment
transfer attempts to apply the high level objective of sentiment transfer to
the domain of music. We propose CycleGAN to bridge disparate domains. In order
to use the network, we choose to use symbolic, MIDI, data as the music format.
Through the use of a cycle consistency loss, we are able to create one-to-one
mappings that preserve the content and realism of the source data. Results and
literature suggest that the task of music sentiment transfer is more difficult
than image sentiment transfer because of the temporal characteristics of music
and lack of existing datasets.

    

### [[2110.05769] Interpretation of Emergent Communication in Heterogeneous Collaborative Embodied Agents](http://arxiv.org/abs/2110.05769)


  Communication between embodied AI agents has received increasing attention in
recent years. Despite its use, it is still unclear whether the learned
communication is interpretable and grounded in perception. To study the
grounding of emergent forms of communication, we first introduce the
collaborative multi-object navigation task CoMON. In this task, an oracle agent
has detailed environment information in the form of a map. It communicates with
a navigator agent that perceives the environment visually and is tasked to find
a sequence of goals. To succeed at the task, effective communication is
essential. CoMON hence serves as a basis to study different communication
mechanisms between heterogeneous agents, that is, agents with different
capabilities and roles. We study two common communication mechanisms and
analyze their communication patterns through an egocentric and spatial lens. We
show that the emergent communication can be grounded to the agent observations
and the spatial structure of the 3D environment. Video summary:
this https URL


### [[2110.05781] BERTraffic: A Robust BERT-Based Approach for Speaker Change Detection and Role Identification of Air-Traffic Communications](http://arxiv.org/abs/2110.05781)


  Automatic Speech Recognition (ASR) is gaining special interest in Air Traffic
Control (ATC). ASR allows transcribing the communications between air traffic
controllers (ATCOs) and pilots. These transcriptions are used to extract ATC
command types and named entities such as aircraft callsigns. One common problem
is when the Speech Activity Detection (SAD) or diarization system fails and
then two or more single speaker segments are in the same recording,
jeopardizing the overall system's performance. We developed a system that
combines the segmentation of a SAD module with a BERT-based model that performs
Speaker Change Detection (SCD) and Speaker Role Identification (SRI) based on
ASR transcripts (i.e., diarization + SRI). This research demonstrates on a
real-life ATC test set that performing diarization directly on textual data
surpass acoustic level diarization. The proposed model reaches up to
~0.90/~0.95 F1-score on ATCO/pilot for SRI on several test sets. The text-based
diarization system brings a 27% relative improvement on Diarization Error Rate
(DER) compared to standard acoustic-based diarization. These results were on
ASR transcripts of a challenging ATC test set with an estimated ~13% word error
rate, validating the approach's robustness even on noisy ASR transcripts.

    

### [[2110.05794] Information Theoretic Structured Generative Modeling](http://arxiv.org/abs/2110.05794)


  Rényi's information provides a theoretical foundation for tractable and
data-efficient non-parametric density estimation, based on pair-wise
evaluations in a reproducing kernel Hilbert space (RKHS). This paper extends
this framework to parametric probabilistic modeling, motivated by the fact that
Rényi's information can be estimated in closed-form for Gaussian mixtures.
Based on this special connection, a novel generative model framework called the
structured generative model (SGM) is proposed that makes straightforward
optimization possible, because costs are scale-invariant, avoiding high
gradient variance while imposing less restrictions on absolute continuity,
which is a huge advantage in parametric information theoretic optimization. The
implementation employs a single neural network driven by an orthonormal input
appended to a single white noise source adapted to learn an infinite Gaussian
mixture model (IMoG), which provides an empirically tractable model
distribution in low dimensions. To train SGM, we provide three novel
variational cost functions, based on Rényi's second-order entropy and
divergence, to implement minimization of cross-entropy, minimization of
variational representations of $f$-divergence, and maximization of the evidence
lower bound (conditional probability). We test the framework for estimation of
mutual information and compare the results with the mutual information neural
estimation (MINE), for density estimation, for conditional probability
estimation in Markov models as well as for training adversarial networks. Our
preliminary results show that SGM significantly improves MINE estimation in
terms of data efficiency and variance, conventional and variational Gaussian
mixture models, as well as the performance of generative adversarial networks.

    

### [[2110.05797] Zero-bias Deep Neural Network for Quickest RF Signal Surveillance](http://arxiv.org/abs/2110.05797)


  The Internet of Things (IoT) is reshaping modern society by allowing a decent
number of RF devices to connect and share information through RF channels.
However, such an open nature also brings obstacles to surveillance. For
alleviation, a surveillance oracle, or a cognitive communication entity needs
to identify and confirm the appearance of known or unknown signal sources in
real-time. In this paper, we provide a deep learning framework for RF signal
surveillance. Specifically, we jointly integrate the Deep Neural Networks
(DNNs) and Quickest Detection (QD) to form a sequential signal surveillance
scheme. We first analyze the latent space characteristic of neural network
classification models, and then we leverage the response characteristics of DNN
classifiers and propose a novel method to transform existing DNN classifiers
into performance-assured binary abnormality detectors. In this way, we
seamlessly integrate the DNNs with the parametric quickest detection. Finally,
we propose an enhanced Elastic Weight Consolidation (EWC) algorithm with better
numerical stability for DNNs in signal surveillance systems to evolve
incrementally, we demonstrate that the zero-bias DNN is superior to regular DNN
models considering incremental learning and decision fairness. We evaluated the
proposed framework using real signal datasets and we believe this framework is
helpful in developing a trustworthy IoT ecosystem.

    

### [[2110.05802] Codabench: Flexible, Easy-to-Use and Reproducible Benchmarking for Everyone](http://arxiv.org/abs/2110.05802)


  Obtaining standardized crowdsourced benchmark of computational methods is a
major issue in scientific communities. Dedicated frameworks enabling fair
continuous benchmarking in a unified environment are yet to be developed. Here
we introduce Codabench, an open-sourced, community-driven platform for
benchmarking algorithms or software agents versus datasets or tasks. A public
instance of Codabench is open to everyone, free of charge, and allows benchmark
organizers to compare fairly submissions, under the same setting (software,
hardware, data, algorithms), with custom protocols and data formats. Codabench
has unique features facilitating the organization of benchmarks flexibly,
easily and reproducibly. Firstly, it supports code submission and data
submission for testing on dedicated compute workers, which can be supplied by
the benchmark organizers. This makes the system scalable, at low cost for the
platform providers. Secondly, Codabench benchmarks are created from
self-contained bundles, which are zip files containing a full description of
the benchmark in a configuration file (following a well-defined schema),
documentation pages, data, ingestion and scoring programs, making benchmarks
reusable and portable. The Codabench documentation includes many examples of
bundles that can serve as templates. Thirdly, Codabench uses dockers for each
task's running environment to make results reproducible. Codabench has been
used internally and externally with more than 10 applications during the past 6
months. As illustrative use cases, we introduce 4 diverse benchmarks covering
Graph Machine Learning, Cancer Heterogeneity, Clinical Diagnosis and
Reinforcement Learning.

    

### [[2110.05807] Optimizing Ranking Systems Online as Bandits](http://arxiv.org/abs/2110.05807)


  Ranking system is the core part of modern retrieval and recommender systems,
where the goal is to rank candidate items given user contexts. Optimizing
ranking systems online means that the deployed system can serve user requests,
e.g., queries in the web search, and optimize the ranking policy by learning
from user interactions, e.g., clicks. Bandit is a general online learning
framework and can be used in our optimization task. However, due to the unique
features of ranking, there are several challenges in designing bandit
algorithms for ranking system optimization. In this dissertation, we study and
propose solutions for four challenges in optimizing ranking systems online:
effectiveness, safety, nonstationarity, and diversification. First, the
effectiveness is related to how fast the algorithm learns from interactions. We
study the effective online ranker evaluation task and propose the MergeDTS
algorithm to solve the problem effectively. Second, the deployed algorithm
should be safe, which means the algorithm only displays reasonable content to
user requests. To solve the safe online learning to rank problem, we propose
the BubbleRank algorithm. Third, as users change their preferences constantly,
the algorithm should handle the nonstationarity. We formulate this
nonstationary online learning to rank problem as cascade non-stationary bandits
and propose CascadeDUCB and CascadeSWUCB algorithms to solve the problem.
Finally, the contents in ranked lists should be diverse. We consider the
results diversification task and propose the CascadeHybird algorithm that
considers both the item relevance and results diversification when learning
from user interactions.

    

### [[2110.05809] Couple Learning: Mean Teacher method with pseudo-labels improves semi-supervised deep learning results](http://arxiv.org/abs/2110.05809)


  The recently proposed Mean Teacher has achieved state-of-the-art results in
several semi-supervised learning benchmarks. The Mean Teacher method can
exploit large-scale unlabeled data in a self-ensembling manner. In this paper,
an effective Couple Learning method based on a well-trained model and a Mean
Teacher model is proposed. The proposed pseudo-labels generated model (PLG) can
increase strongly-labeled data and weakly-labeled data to improve performance
of the Mean Teacher method. The Mean Teacher method can suppress noise in
pseudo-labels data. The Couple Learning method can extract more information in
the compound training data. These experimental results on Task 4 of the
DCASE2020 challenge demonstrate the superiority of the proposed method,
achieving about 39.18% F1-score on public eval set, outperforming 37.12% of the
baseline system by a significant margin.

    

### [[2110.05820] CoarSAS2hvec: Heterogeneous Information Network Embedding with Balanced Network Sampling](http://arxiv.org/abs/2110.05820)


  Heterogeneous information network (HIN) embedding aims to find the
representations of nodes that preserve the proximity between entities of
different nature. A family of approaches that are wildly adopted applies random
walk to generate a sequence of heterogeneous context, from which the embedding
is learned. However, due to the multipartite graph structure of HIN, hub nodes
tend to be over-represented in the sampled sequence, giving rise to imbalanced
samples of the network. Here we propose a new embedding method CoarSAS2hvec.
The self-avoid short sequence sampling with the HIN coarsening procedure
(CoarSAS) is utilized to better collect the rich information in HIN. An
optimized loss function is used to improve the performance of the HIN structure
embedding. CoarSAS2hvec outperforms nine other methods in two different tasks
on four real-world data sets. The ablation study confirms that the samples
collected by CoarSAS contain richer information of the network compared with
those by other methods, which is characterized by a higher information entropy.
Hence, the traditional loss function applied to samples by CoarSAS can also
yield improved results. Our work addresses a limitation of the
random-walk-based HIN embedding that has not been emphasized before, which can
shed light on a range of problems in HIN analyses.

    

### [[2110.05838] Balancing Average and Worst-case Accuracy in Multitask Learning](http://arxiv.org/abs/2110.05838)


  When training and evaluating machine learning models on a large number of
tasks, it is important to not only look at average task accuracy -- which may
be biased by easy or redundant tasks -- but also worst-case accuracy (i.e. the
performance on the task with the lowest accuracy). In this work, we show how to
use techniques from the distributionally robust optimization (DRO) literature
to improve worst-case performance in multitask learning. We highlight several
failure cases of DRO when applied off-the-shelf and present an improved method,
Lookahead-DRO (L-DRO), which mitigates these issues. The core idea of L-DRO is
to anticipate the interaction between tasks during training in order to choose
a dynamic re-weighting of the various task losses, which will (i) lead to
minimal worst-case loss and (ii) train on as many tasks as possible. After
demonstrating the efficacy of L-DRO on a small controlled synthetic setting, we
evaluate it on two realistic benchmarks: a multitask version of the CIFAR-100
image classification dataset and a large-scale multilingual language modeling
experiment. Our empirical results show that L-DRO achieves a better trade-off
between average and worst-case accuracy with little computational overhead
compared to several strong baselines.

    

### [[2110.05841] Relative Molecule Self-Attention Transformer](http://arxiv.org/abs/2110.05841)


  Self-supervised learning holds promise to revolutionize molecule property
prediction - a central task to drug discovery and many more industries - by
enabling data efficient learning from scarce experimental data. Despite
significant progress, non-pretrained methods can be still competitive in
certain settings. We reason that architecture might be a key bottleneck. In
particular, enriching the backbone architecture with domain-specific inductive
biases has been key for the success of self-supervised learning in other
domains. In this spirit, we methodologically explore the design space of the
self-attention mechanism tailored to molecular data. We identify a novel
variant of self-attention adapted to processing molecules, inspired by the
relative self-attention layer, which involves fusing embedded graph and
distance relationships between atoms. Our main contribution is Relative
Molecule Attention Transformer (R-MAT): a novel Transformer-based model based
on the developed self-attention layer that achieves state-of-the-art or very
competitive results across a~wide range of molecule property prediction tasks.

    

### [[2110.05842] Across-Task Neural Architecture Search via Meta Learning](http://arxiv.org/abs/2110.05842)


  Adequate labeled data and expensive compute resources are the prerequisites
for the success of neural architecture search(NAS). It is challenging to apply
NAS in meta-learning scenarios with limited compute resources and data. In this
paper, an across-task neural architecture search (AT-NAS) is proposed to
address the problem through combining gradient-based meta-learning with
EA-based NAS to learn over the distribution of tasks. The supernet is learned
over an entire set of tasks by meta-learning its weights. Architecture encodes
of subnets sampled from the supernet are iteratively adapted by evolutionary
algorithms while simultaneously searching for a task-sensitive meta-network.
Searched meta-network can be adapted to a novel task via a few learning steps
and only costs a little search time. Empirical results show that AT-NAS
surpasses the related approaches on few-shot classification accuracy. The
performance of AT-NAS on classification benchmarks is comparable to that of
models searched from scratch, by adapting the architecture in less than an hour
from a 5-GPU-day pretrained meta-network.

    

### [[2110.05843] Fast Block Linear System Solver Using Q-Learning Schduling for Unified Dynamic Power System Simulations](http://arxiv.org/abs/2110.05843)


  We present a fast block direct solver for the unified dynamic simulations of
power systems. This solver uses a novel Q-learning based method for task
scheduling. Unified dynamic simulations of power systems represent a method in
which the electric-mechanical transient, medium-term and long-term dynamic
phenomena are organically united. Due to the high rank and large numbers in
solving, fast solution of these equations is the key to speeding up the
simulation. The sparse systems of simulation contain complex nested block
structure, which could be used by the solver to speed up. For the scheduling of
blocks and frontals in the solver, we use a learning based task-tree scheduling
technique in the framework of Markov Decision Process. That is, we could learn
optimal scheduling strategies by offline training on many sample matrices. Then
for any systems, the solver would get optimal task partition and scheduling on
the learned model. Our learning-based algorithm could help improve the
performance of sparse solver, which has been verified in some numerical
experiments. The simulation on some large power systems shows that our solver
is 2-6 times faster than KLU, which is the state-of-the-art sparse solver for
circuit simulation problems.

    

### [[2110.05847] Evaluation of Abstractive Summarisation Models with Machine Translation in Deliberative Processes](http://arxiv.org/abs/2110.05847)


  We present work on summarising deliberative processes for non-English
languages. Unlike commonly studied datasets, such as news articles, this
deliberation dataset reflects difficulties of combining multiple narratives,
mostly of poor grammatical quality, in a single text. We report an extensive
evaluation of a wide range of abstractive summarisation models in combination
with an off-the-shelf machine translation model. Texts are translated into
English, summarised, and translated back to the original language. We obtain
promising results regarding the fluency, consistency and relevance of the
summaries produced. Our approach is easy to implement for many languages for
production purposes by simply changing the translation model.

    

### [[2110.05849] Sharing FANCI Features: A Privacy Analysis of Feature Extraction for DGA Detection](http://arxiv.org/abs/2110.05849)


  The goal of Domain Generation Algorithm (DGA) detection is to recognize
infections with bot malware and is often done with help of Machine Learning
approaches that classify non-resolving Domain Name System (DNS) traffic and are
trained on possibly sensitive data. In parallel, the rise of privacy research
in the Machine Learning world leads to privacy-preserving measures that are
tightly coupled with a deep learning model's architecture or training routine,
while non deep learning approaches are commonly better suited for the
application of privacy-enhancing methods outside the actual classification
module. In this work, we aim to measure the privacy capability of the feature
extractor of feature-based DGA detector FANCI (Feature-based Automated Nxdomain
Classification and Intelligence). Our goal is to assess whether a data-rich
adversary can learn an inverse mapping of FANCI's feature extractor and thereby
reconstruct domain names from feature vectors. Attack success would pose a
privacy threat to sharing FANCI's feature representation, while the opposite
would enable this representation to be shared without privacy concerns. Using
three real-world data sets, we train a recurrent Machine Learning model on the
reconstruction task. Our approaches result in poor reconstruction performance
and we attempt to back our findings with a mathematical review of the feature
extraction process. We thus reckon that sharing FANCI's feature representation
does not constitute a considerable privacy leakage.

    

### [[2110.05852] On the Self-Penalization Phenomenon in Feature Selection](http://arxiv.org/abs/2110.05852)


  We describe an implicit sparsity-inducing mechanism based on minimization
over a family of kernels: \begin{equation*}
\min_{\beta, f}~\widehat{\mathbb{E}}[L(Y, f(\beta^{1/q} \odot X)] + \lambda_n
\|f\|_{\mathcal{H}_q}^2~~\text{subject to}~~\beta \ge 0, \end{equation*} where
$L$ is the loss, $\odot$ is coordinate-wise multiplication and $\mathcal{H}_q$
is the reproducing kernel Hilbert space based on the kernel $k_q(x, x') =
h(\|x-x'\|_q^q)$, where $\|\cdot\|_q$ is the $\ell_q$ norm. Using gradient
descent to optimize this objective with respect to $\beta$ leads to exactly
sparse stationary points with high probability. The sparsity is achieved
without using any of the well-known explicit sparsification techniques such as
penalization (e.g., $\ell_1$), early stopping or post-processing (e.g.,
clipping).
As an application, we use this sparsity-inducing mechanism to build
algorithms consistent for feature selection.

    

### [[2110.05854] A scalable and fast artificial neural network syndrome decoder for surface codes](http://arxiv.org/abs/2110.05854)


  Surface code error correction offers a highly promising pathway to achieve
scalable fault-tolerant quantum computing. When operated as stabilizer codes,
surface code computations consist of a syndrome decoding step where measured
stabilizer operators are used to determine appropriate corrections for errors
in physical qubits. Decoding algorithms have undergone substantial development,
with recent work incorporating machine learning (ML) techniques. Despite
promising initial results, the ML-based syndrome decoders are still limited to
small scale demonstrations with low latency and are incapable of handling
surface codes with boundary conditions and various shapes needed for lattice
surgery and braiding. Here, we report the development of an artificial neural
network (ANN) based scalable and fast syndrome decoder capable of decoding
surface codes of arbitrary shape and size with data qubits suffering from the
depolarizing error model. Based on rigorous training over 50 million random
quantum error instances, our ANN decoder is shown to work with code distances
exceeding 1000 (more than 4 million physical qubits), which is the largest
ML-based decoder demonstration to-date. The established ANN decoder
demonstrates an execution time in principle independent of code distance,
implying that its implementation on dedicated hardware could potentially offer
surface code decoding times of O($\mu$sec), commensurate with the
experimentally realisable qubit coherence times. With the anticipated scale-up
of quantum processors within the next decade, their augmentation with a fast
and scalable syndrome decoder such as developed in our work is expected to play
a decisive role towards experimental implementation of fault-tolerant quantum
information processing.

    

### [[2110.05864] Observing a group to infer individual characteristics](http://arxiv.org/abs/2110.05864)


  In the study of collective motion, it is common practice to collect movement
information at the level of the group to infer the characteristics of the
individual agents and their interactions. However, it is not clear whether one
can always correctly infer individual characteristics from movement data of the
collective. We investigate this question in the context of a composite crowd
with two groups of agents, each with its own desired direction of motion. A
simple observer attempts to classify an agent into its group based on its
movement information. However, collective effects such as collisions,
entrainment of agents, formation of lanes and clusters, etc. render the
classification problem non-trivial, and lead to misclassifications. Based on
our understanding of these effects, we propose a new observer algorithm that
infers, based only on observed movement information, how the local neighborhood
aids or hinders agent movement. Unlike a traditional supervised learning
approach, this algorithm is based on physical insights and scaling arguments,
and does not rely on training-data. This new observer improves classification
performance and is able to differentiate agents belonging to different groups
even when their motion is identical. Data-agnostic approaches like this have
relevance to a large class of real-world problems where clean, labeled data is
difficult to obtain, and is a step towards hybrid approaches that integrate
both data and domain knowledge.

    

### [[2110.05876] Label-Aware Ranked Loss for robust People Counting using Automotive in-cabin Radar](http://arxiv.org/abs/2110.05876)


  In this paper, we introduce the Label-Aware Ranked loss, a novel metric loss
function. Compared to the state-of-the-art Deep Metric Learning losses, this
function takes advantage of the ranked ordering of the labels in regression
problems. To this end, we first show that the loss minimises when datapoints of
different labels are ranked and laid at uniform angles between each other in
the embedding space. Then, to measure its performance, we apply the proposed
loss on a regression task of people counting with a short-range radar in a
challenging scenario, namely a vehicle cabin. The introduced approach improves
the accuracy as well as the neighboring labels accuracy up to 83.0% and 99.9%:
An increase of 6.7%and 2.1% on state-of-the-art methods, respectively.

    

### [[2110.05877] OpenHands: Making Sign Language Recognition Accessible with Pose-based Pretrained Models across Languages](http://arxiv.org/abs/2110.05877)


  AI technologies for Natural Languages have made tremendous progress recently.
However, commensurate progress has not been made on Sign Languages, in
particular, in recognizing signs as individual words or as complete sentences.
We introduce OpenHands, a library where we take four key ideas from the NLP
community for low-resource languages and apply them to sign languages for
word-level recognition. First, we propose using pose extracted through
pretrained models as the standard modality of data to reduce training time and
enable efficient inference, and we release standardized pose datasets for 6
different sign languages - American, Argentinian, Chinese, Greek, Indian, and
Turkish. Second, we train and release checkpoints of 4 pose-based isolated sign
language recognition models across all 6 languages, providing baselines and
ready checkpoints for deployment. Third, to address the lack of labelled data,
we propose self-supervised pretraining on unlabelled data. We curate and
release the largest pose-based pretraining dataset on Indian Sign Language
(Indian-SL). Fourth, we compare different pretraining strategies and for the
first time establish that pretraining is effective for sign language
recognition by demonstrating (a) improved fine-tuning performance especially in
low-resource settings, and (b) high crosslingual transfer from Indian-SL to few
other sign languages. We open-source all models and datasets in OpenHands with
a hope that it makes research in sign languages more accessible, available here
at this https URL .

    

### [[2110.05887] Single Independent Component Recovery and Applications](http://arxiv.org/abs/2110.05887)


  Latent variable discovery is a central problem in data analysis with a broad
range of applications in applied science. In this work, we consider data given
as an invertible mixture of two statistically independent components, and
assume that one of the components is observed while the other is hidden. Our
goal is to recover the hidden component. For this purpose, we propose an
autoencoder equipped with a discriminator. Unlike the standard nonlinear ICA
problem, which was shown to be non-identifiable, in the special case of ICA we
consider here, we show that our approach can recover the component of interest
up to entropy-preserving transformation. We demonstrate the performance of the
proposed approach on several datasets, including image synthesis, voice
cloning, and fetal ECG extraction.

    

### [[2110.05892] Investigation on Data Adaptation Techniques for Neural Named Entity Recognition](http://arxiv.org/abs/2110.05892)


  Data processing is an important step in various natural language processing
tasks. As the commonly used datasets in named entity recognition contain only a
limited number of samples, it is important to obtain additional labeled data in
an efficient and reliable manner. A common practice is to utilize large
monolingual unlabeled corpora. Another popular technique is to create synthetic
data from the original labeled data (data augmentation). In this work, we
investigate the impact of these two methods on the performance of three
different named entity recognition tasks.

    

### [[2110.05922] Trivial or impossible -- dichotomous data difficulty masks model differences (on ImageNet and beyond)](http://arxiv.org/abs/2110.05922)


  "The power of a generalization system follows directly from its biases"
(Mitchell 1980). Today, CNNs are incredibly powerful generalisation systems --
but to what degree have we understood how their inductive bias influences model
decisions? We here attempt to disentangle the various aspects that determine
how a model decides. In particular, we ask: what makes one model decide
differently from another? In a meticulously controlled setting, we find that
(1.) irrespective of the network architecture or objective (e.g.
self-supervised, semi-supervised, vision transformers, recurrent models) all
models end up with a similar decision boundary. (2.) To understand these
findings, we analysed model decisions on the ImageNet validation set from epoch
to epoch and image by image. We find that the ImageNet validation set, among
others, suffers from dichotomous data difficulty (DDD): For the range of
investigated models and their accuracies, it is dominated by 46.0% "trivial"
and 11.5% "impossible" images (beyond label errors). Only 42.5% of the images
could possibly be responsible for the differences between two models' decision
boundaries. (3.) Only removing the "impossible" and "trivial" images allows us
to see pronounced differences between models. (4.) Humans are highly accurate
at predicting which images are "trivial" and "impossible" for CNNs (81.4%).
This implies that in future comparisons of brains, machines and behaviour, much
may be gained from investigating the decisive role of images and the
distribution of their difficulties.

    

### [[2110.05941] Rank-based loss for learning hierarchical representations](http://arxiv.org/abs/2110.05941)


  Hierarchical taxonomies are common in many contexts, and they are a very
natural structure humans use to organise information. In machine learning, the
family of methods that use the 'extra' information is called hierarchical
classification. However, applied to audio classification, this remains
relatively unexplored. Here we focus on how to integrate the hierarchical
information of a problem to learn embeddings representative of the hierarchical
relationships. Previously, triplet loss has been proposed to address this
problem, however it presents some issues like requiring the careful
construction of the triplets, and being limited in the extent of hierarchical
information it uses at each iteration. In this work we propose a rank based
loss function that uses hierarchical information and translates this into a
rank ordering of target distances between the examples. We show that rank based
loss is suitable to learn hierarchical representations of the data. By testing
on unseen fine level classes we show that this method is also capable of
learning hierarchically correct representations of the new classes. Rank based
loss has two promising aspects, it is generalisable to hierarchies with any
number of levels, and is capable of dealing with data with incomplete
hierarchical labels.

    

### [[2110.05945] Multi-condition multi-objective optimization using deep reinforcement learning](http://arxiv.org/abs/2110.05945)


  A multi-condition multi-objective optimization method that can find Pareto
front over a defined condition space is developed for the first time using deep
reinforcement learning. Unlike the conventional methods which perform
optimization at a single condition, the present method learns the correlations
between conditions and optimal solutions. The exclusive capability of the
developed method is examined in the solutions of a novel modified Kursawe
benchmark problem and an airfoil shape optimization problem which include
nonlinear characteristics which are difficult to resolve using conventional
optimization methods. Pareto front with high resolution over a defined
condition space is successfully determined in each problem. Compared with
multiple operations of a single-condition optimization method for multiple
conditions, the present multi-condition optimization method based on deep
reinforcement learning shows a greatly accelerated search of Pareto front by
reducing the number of required function evaluations. An analysis of
aerodynamics performance of airfoils with optimally designed shapes confirms
that multi-condition optimization is indispensable to avoid significant
degradation of target performance for varying flow conditions.

    

### [[2110.05947] C3PU: Cross-Coupling Capacitor Processing Unit Using Analog-Mixed Signal In-Memory Computing for AI Inference](http://arxiv.org/abs/2110.05947)


  This paper presents a novel cross-coupling capacitor processing unit (C3PU)
that supports analog-mixed signal in memory computing to perform
multiply-and-accumulate (MAC) operations. The C3PU consists of a capacitive
unit, a CMOS transistor, and a voltage-to-time converter (VTC). The capacitive
unit serves as a computational element that holds the multiplier operand and
performs multiplication once the multiplicand is applied at the terminal. The
multiplicand is the input voltage that is converted to a pulse width signal
using a low power VTC. The transistor transfers this multiplication where a
voltage level is generated. A demonstrator of 5x4 C3PU array that is capable of
implementing 4 MAC units is presented. The design has been verified using Monte
Carlo simulation in 65 nm technology. The 5x4 C3PU consumed energy of 66.4
fJ/MAC at 0.3 V voltage supply with an error of 5.7%. The proposed unit
achieves lower energy and occupies a smaller area by 3.4x and 3.6x,
respectively, with similar error value when compared to a digital-based 8x4-bit
fixed point MAC unit. The C3PU has been utilized through an iris fower
classification utilizing an artificial neural network which achieved a 90%
classification accuracy compared to ideal accuracy of 96.67% using MATLAB.

    

### [[2110.05948] Denoising Diffusion Gamma Models](http://arxiv.org/abs/2110.05948)


  Generative diffusion processes are an emerging and effective tool for image
and speech generation. In the existing methods, the underlying noise
distribution of the diffusion process is Gaussian noise. However, fitting
distributions with more degrees of freedom could improve the performance of
such generative models. In this work, we investigate other types of noise
distribution for the diffusion process. Specifically, we introduce the
Denoising Diffusion Gamma Model (DDGM) and show that noise from Gamma
distribution provides improved results for image and speech generation. Our
approach preserves the ability to efficiently sample state in the training
diffusion process while using Gamma noise.

    

### [[2110.05954] Mining the Weights Knowledge for Optimizing Neural Network Structures](http://arxiv.org/abs/2110.05954)


  Knowledge embedded in the weights of the artificial neural network can be
used to improve the network structure, such as in network compression. However,
the knowledge is set up by hand, which may not be very accurate, and relevant
information may be overlooked. Inspired by how learning works in the mammalian
brain, we mine the knowledge contained in the weights of the neural network
toward automatic architecture learning in this paper. We introduce a switcher
neural network (SNN) that uses as inputs the weights of a task-specific neural
network (called TNN for short). By mining the knowledge contained in the
weights, the SNN outputs scaling factors for turning off and weighting neurons
in the TNN. To optimize the structure and the parameters of TNN simultaneously,
the SNN and TNN are learned alternately under the same performance evaluation
of TNN using stochastic gradient descent. We test our method on widely used
datasets and popular networks in classification applications. In terms of
accuracy, we outperform baseline networks and other structure learning methods
stably and significantly. At the same time, we compress the baseline networks
without introducing any sparse induction mechanism, and our method, in
particular, leads to a lower compression rate when dealing with simpler
baselines or more difficult tasks. These results demonstrate that our method
can produce a more reasonable structure.

    

### [[2110.05960] Imitating Deep Learning Dynamics via Locally Elastic Stochastic Differential Equations](http://arxiv.org/abs/2110.05960)


  Understanding the training dynamics of deep learning models is perhaps a
necessary step toward demystifying the effectiveness of these models. In
particular, how do data from different classes gradually become separable in
their feature spaces when training neural networks using stochastic gradient
descent? In this study, we model the evolution of features during deep learning
training using a set of stochastic differential equations (SDEs) that each
corresponds to a training sample. As a crucial ingredient in our modeling
strategy, each SDE contains a drift term that reflects the impact of
backpropagation at an input on the features of all samples. Our main finding
uncovers a sharp phase transition phenomenon regarding the {intra-class impact:
if the SDEs are locally elastic in the sense that the impact is more
significant on samples from the same class as the input, the features of the
training data become linearly separable, meaning vanishing training loss;
otherwise, the features are not separable, regardless of how long the training
time is. Moreover, in the presence of local elasticity, an analysis of our SDEs
shows that the emergence of a simple geometric structure called the neural
collapse of the features. Taken together, our results shed light on the
decisive role of local elasticity in the training dynamics of neural networks.
We corroborate our theoretical analysis with experiments on a synthesized
dataset of geometric shapes and CIFAR-10.

    

### [[2110.05973] Can machines learn to see without visual databases?](http://arxiv.org/abs/2110.05973)


  This paper sustains the position that the time has come for thinking of
learning machines that conquer visual skills in a truly human-like context,
where a few human-like object supervisions are given by vocal interactions and
pointing aids only. This likely requires new foundations on computational
processes of vision with the final purpose of involving machines in tasks of
visual description by living in their own visual environment under simple
man-machine linguistic interactions. The challenge consists of developing
machines that learn to see without needing to handle visual databases. This
might open the doors to a truly orthogonal competitive track concerning deep
learning technologies for vision which does not rely on the accumulation of
huge visual databases.

    

### [[2110.05976] Early Melanoma Diagnosis with Sequential Dermoscopic Images](http://arxiv.org/abs/2110.05976)


  Dermatologists often diagnose or rule out early melanoma by evaluating the
follow-up dermoscopic images of skin lesions. However, existing algorithms for
early melanoma diagnosis are developed using single time-point images of
lesions. Ignoring the temporal, morphological changes of lesions can lead to
misdiagnosis in borderline cases. In this study, we propose a framework for
automated early melanoma diagnosis using sequential dermoscopic images. To this
end, we construct our method in three steps. First, we align sequential
dermoscopic images of skin lesions using estimated Euclidean transformations,
extract the lesion growth region by computing image differences among the
consecutive images, and then propose a spatio-temporal network to capture the
dermoscopic changes from aligned lesion images and the corresponding difference
images. Finally, we develop an early diagnosis module to compute probability
scores of malignancy for lesion images over time. We collected 179 serial
dermoscopic imaging data from 122 patients to verify our method. Extensive
experiments show that the proposed model outperforms other commonly used
sequence models. We also compared the diagnostic results of our model with
those of seven experienced dermatologists and five registrars. Our model
achieved higher diagnostic accuracy than clinicians (63.69% vs. 54.33%,
respectively) and provided an earlier diagnosis of melanoma (60.7% vs. 32.7% of
melanoma correctly diagnosed on the first follow-up images). These results
demonstrate that our model can be used to identify melanocytic lesions that are
at high-risk of malignant transformation earlier in the disease process and
thereby redefine what is possible in the early detection of melanoma.

    

### [[2110.06014] Rethinking supervised pre-training for better downstream transferring](http://arxiv.org/abs/2110.06014)


  The pretrain-finetune paradigm has shown outstanding performance on many
applications of deep learning, where a model is pre-trained on a upstream large
dataset (e.g. ImageNet), and is then fine-tuned to different downstream tasks.
Though for most cases, the pre-training stage is conducted based on supervised
methods, recent works on self-supervised pre-training have shown powerful
transferability and even outperform supervised pre-training on multiple
downstream tasks. It thus remains an open question how to better generalize
supervised pre-training model to downstream tasks. In this paper, we argue that
the worse transferability of existing supervised pre-training methods arise
from the negligence of valuable intra-class semantic difference. This is
because these methods tend to push images from the same class close to each
other despite of the large diversity in their visual contents, a problem to
which referred as "overfit of upstream tasks". To alleviate this problem, we
propose a new supervised pre-training method based on Leave-One-Out
K-Nearest-Neighbor, or LOOK for short. It relieves the problem of overfitting
upstream tasks by only requiring each image to share its class label with most
of its k nearest neighbors, thus allowing each class to exhibit a multi-mode
distribution and consequentially preserving part of intra-class difference for
better transferring to downstream tasks. We developed efficient implementation
of the proposed method that scales well to large datasets. Experimental studies
on multiple downstream tasks show that LOOK outperforms other state-of-the-art
methods for supervised and self-supervised pre-training.

    

### [[2110.06018] On the Security Risks of AutoML](http://arxiv.org/abs/2110.06018)


  Neural Architecture Search (NAS) represents an emerging machine learning (ML)
paradigm that automatically searches for models tailored to given tasks, which
greatly simplifies the development of ML systems and propels the trend of ML
democratization. Yet, little is known about the potential security risks
incurred by NAS, which is concerning given the increasing use of NAS-generated
models in critical domains.
This work represents a solid initial step towards bridging the gap. Through
an extensive empirical study of 10 popular NAS methods, we show that compared
with their manually designed counterparts, NAS-generated models tend to suffer
greater vulnerability to various malicious attacks (e.g., adversarial evasion,
model poisoning, and functionality stealing). Further, with both empirical and
analytical evidence, we provide possible explanations for such phenomena: given
the prohibitive search space and training cost, most NAS methods favor models
that converge fast at early training stages; this preference results in
architectural properties associated with attack vulnerability (e.g., high loss
smoothness and low gradient variance). Our findings not only reveal the
relationships between model characteristics and attack vulnerability but also
suggest the inherent connections underlying different attacks. Finally, we
discuss potential remedies to mitigate such drawbacks, including increasing
cell depth and suppressing skip connects, which lead to several promising
research directions.

    

### [[2110.06020] Uncertainty-based out-of-distribution detection requires suitable function space priors](http://arxiv.org/abs/2110.06020)


  The need to avoid confident predictions on unfamiliar data has sparked
interest in out-of-distribution (OOD) detection. It is widely assumed that
Bayesian neural networks (BNNs) are well suited for this task, as the endowed
epistemic uncertainty should lead to disagreement in predictions on outliers.
In this paper, we question this assumption and show that proper Bayesian
inference with function space priors induced by neural networks does not
necessarily lead to good OOD detection. To circumvent the use of approximate
inference, we start by studying the infinite-width case, where Bayesian
inference can be exact due to the correspondence with Gaussian processes.
Strikingly, the kernels induced under common architectural choices lead to
uncertainties that do not reflect the underlying data generating process and
are therefore unsuited for OOD detection. Importantly, we find this OOD
behavior to be consistent with the corresponding finite-width networks.
Desirable function space properties can be encoded in the prior in weight
space, however, this currently only applies to a specified subset of the domain
and thus does not inherently extend to OOD data. Finally, we argue that a
trade-off between generalization and OOD capabilities might render the
application of BNNs for OOD detection undesirable in practice. Overall, our
study discloses fundamental problems when naively using BNNs for OOD detection
and opens interesting avenues for future research.

    

### [[2110.06021] Embedded-model flows: Combining the inductive biases of model-free deep learning and explicit probabilistic modeling](http://arxiv.org/abs/2110.06021)


  Normalizing flows have shown great success as general-purpose density
estimators. However, many real world applications require the use of
domain-specific knowledge, which normalizing flows cannot readily incorporate.
We propose embedded-model flows(EMF), which alternate general-purpose
transformations with structured layers that embed domain-specific inductive
biases. These layers are automatically constructed by converting user-specified
differentiable probabilistic models into equivalent bijective transformations.
We also introduce gated structured layers, which allow bypassing the parts of
the models that fail to capture the statistics of the data. We demonstrate that
EMFs can be used to induce desirable properties such as multimodality,
hierarchical coupling and continuity. Furthermore, we show that EMFs enable a
high performance form of variational inference where the structure of the prior
model is embedded in the variational architecture. In our experiments, we show
that this approach outperforms state-of-the-art methods in common structured
inference problems.

    

### [[2110.06022] Smart Crawling: A New Approach toward Focus Crawling from Twitter](http://arxiv.org/abs/2110.06022)


  Twitter is a social network that offers a rich and interesting source of
information challenging to retrieve and analyze. Twitter data can be accessed
using a REST API. The available operations allow retrieving tweets on the basis
of a set of keywords but with limitations such as the number of calls per
minute and the size of results. Besides, there is no control on retrieved
results and finding tweets which are relevant to a specific topic is a big
issue. Given these limitations, it is important that the query keywords cover
unambiguously the topic of interest in order to both reach the relevant answers
and decrease the number of API calls. In this paper, we introduce a new
crawling algorithm called "SmartTwitter Crawling" (STiC) that retrieves a set
of tweets related to a target topic. In this algorithm, we take an initial
keyword query and enrich it using a set of additional keywords that come from
different data sources. STiC algorithm relies on a DFS search in Twittergraph
where each reached tweet is considered if it is relevant with the query
keywords using a scoring, updated throughout the whole crawling process. This
scoring takes into account the tweet text, hashtags and the users who have
posted the tweet, replied to the tweet, been mentioned in the tweet or
retweeted the tweet. Given this score, STiC is able to select relevant tweets
in each iteration and continue by adding the related valuable tweets. Several
experiments have been achieved for different kinds of queries, the results
showedthat the precision increases compared to a simple BFS search.

    

### [[2110.06025] Privacy-Preserving Phishing Email Detection Based on Federated Learning and LSTM](http://arxiv.org/abs/2110.06025)


  Phishing emails that appear legitimate lure people into clicking on the
attached malicious links or documents. Increasingly more sophisticated phishing
campaigns in recent years necessitate a more adaptive detection system other
than traditional signature-based methods. In this regard, natural language
processing (NLP) with deep neural networks (DNNs) is adopted for knowledge
acquisition from a large number of emails. However, such sensitive daily
communications containing personal information are difficult to collect on a
server for centralized learning in real life due to escalating privacy
concerns. To this end, we propose a decentralized phishing email detection
method called the Federated Phish Bowl (FPB) leveraging federated learning and
long short-term memory (LSTM). FPB allows common knowledge representation and
sharing among different clients through the aggregation of trained models to
safeguard the email security and privacy. A recent phishing email dataset was
collected from an intergovernmental organization to train the model. Moreover,
we evaluated the model performance based on various assumptions regarding the
total client number and the level of data heterogeneity. The comprehensive
experimental results suggest that FPB is robust to a continually increasing
client number and various data heterogeneity levels, retaining a detection
accuracy of 0.83 and protecting the privacy of sensitive email communications.

    

### [[2110.06037] SoftNeuro: Fast Deep Inference using Multi-platform Optimization](http://arxiv.org/abs/2110.06037)


  Faster inference of deep learning models is highly demanded on edge devices
and even servers, for both financial and environmental reasons. To address this
issue, we propose SoftNeuro, a novel, high-performance inference framework with
efficient performance tuning. The key idea is to separate algorithmic routines
from network layers. Our framework maximizes the inference performance by
profiling various routines for each layer and selecting the fastest path. To
efficiently find the best path, we propose a routine-selection algorithm based
on dynamic programming. Experiments show that the proposed framework achieves
both fast inference and efficient tuning.

    

### [[2110.06042] SlideGraph+: Whole Slide Image Level Graphs to Predict HER2Status in Breast Cancer](http://arxiv.org/abs/2110.06042)


  Human epidermal growth factor receptor 2 (HER2) is an important prognostic
and predictive factor which is overexpressed in 15-20% of breast cancer (BCa).
The determination of its status is a key clinical decision making step for
selection of treatment regimen and prognostication. HER2 status is evaluated
using transcroptomics or immunohistochemistry (IHC) through situ hybridisation
(ISH) which require additional costs and tissue burden in addition to
analytical variabilities in terms of manual observational biases in scoring. In
this study, we propose a novel graph neural network (GNN) based model (termed
SlideGraph+) to predict HER2 status directly from whole-slide images of routine
Haematoxylin and Eosin (H&E) slides. The network was trained and tested on
slides from The Cancer Genome Atlas (TCGA) in addition to two independent test
datasets. We demonstrate that the proposed model outperforms the
state-of-the-art methods with area under the ROC curve (AUC) values > 0.75 on
TCGA and 0.8 on independent test sets. Our experiments show that the proposed
approach can be utilised for case triaging as well as pre-ordering diagnostic
tests in a diagnostic setting. It can also be used for other weakly supervised
prediction problems in computational pathology. The SlideGraph+ code is
available at this https URL.

    

### [[2110.06057] Gated Information Bottleneck for Generalization in Sequential Environments](http://arxiv.org/abs/2110.06057)


  Deep neural networks suffer from poor generalization to unseen environments
when the underlying data distribution is different from that in the training
set. By learning minimum sufficient representations from training data, the
information bottleneck (IB) approach has demonstrated its effectiveness to
improve generalization in different AI applications. In this work, we propose a
new neural network-based IB approach, termed gated information bottleneck
(GIB), that dynamically drops spurious correlations and progressively selects
the most task-relevant features across different environments by a trainable
soft mask (on raw features). GIB enjoys a simple and tractable objective,
without any variational approximation or distributional assumption. We
empirically demonstrate the superiority of GIB over other popular neural
network-based IB approaches in adversarial robustness and out-of-distribution
(OOD) detection. Meanwhile, we also establish the connection between IB theory
and invariant causal representation learning, and observed that GIB
demonstrates appealing performance when different environments arrive
sequentially, a more practical scenario where invariant risk minimization (IRM)
fails. Code of GIB is available at this https URL


### [[2110.06059] Development of Deep Transformer-Based Models for Long-Term Prediction of Transient Production of Oil Wells](http://arxiv.org/abs/2110.06059)


  We propose a novel approach to data-driven modeling of a transient production
of oil wells. We apply the transformer-based neural networks trained on the
multivariate time series composed of various parameters of oil wells measured
during their exploitation. By tuning the machine learning models for a single
well (ignoring the effect of neighboring wells) on the open-source field
datasets, we demonstrate that transformer outperforms recurrent neural networks
with LSTM/GRU cells in the forecasting of the bottomhole pressure dynamics. We
apply the transfer learning procedure to the transformer-based surrogate model,
which includes the initial training on the dataset from a certain well and
additional tuning of the model's weights on the dataset from a target well.
Transfer learning approach helps to improve the prediction capability of the
model. Next, we generalize the single-well model based on the transformer
architecture for multiple wells to simulate complex transient oilfield-level
patterns. In other words, we create the global model which deals with the
dataset, comprised of the production history from multiple wells, and allows
for capturing the well interference resulting in more accurate prediction of
the bottomhole pressure or flow rate evolutions for each well under
consideration. The developed instruments for a single-well and oilfield-scale
modelling can be used to optimize the production process by selecting the
operating regime and submersible equipment to increase the hydrocarbon
recovery. In addition, the models can be helpful to perform well-testing
avoiding costly shut-in operations.

    

### [[2110.06073] Synergy: Resource Sensitive DNN Scheduling in Multi-Tenant Clusters](http://arxiv.org/abs/2110.06073)


  Training Deep Neural Networks (DNNs) is a widely popular workload in both
enterprises and cloud data centers. Existing schedulers for DNN training
consider GPU as the dominant resource, and allocate other resources such as CPU
and memory proportional to the number of GPUs requested by the job.
Unfortunately, these schedulers do not consider the impact of a job's
sensitivity to allocation of CPU, memory, and storage resources. In this work,
we propose Synergy, a resource-sensitive scheduler for shared GPU clusters.
Synergy infers the sensitivity of DNNs to different resources using optimistic
profiling; some jobs might benefit from more than the GPU-proportional
allocation and some jobs might not be affected by less than GPU-proportional
allocation. Synergy performs such multi-resource workload-aware assignments
across a set of jobs scheduled on shared multi-tenant clusters using a new
near-optimal online algorithm. Our experiments show that workload-aware CPU and
memory allocations can improve average JCT up to 3.4x when compared to
traditional GPU-proportional scheduling.

    

### [[2110.06078] Model-based analysis of brain activity reveals the hierarchy of language in 305 subjects](http://arxiv.org/abs/2110.06078)


  A popular approach to decompose the neural bases of language consists in
correlating, across individuals, the brain responses to different stimuli (e.g.
regular speech versus scrambled words, sentences, or paragraphs). Although
successful, this `model-free' approach necessitates the acquisition of a large
and costly set of neuroimaging data. Here, we show that a model-based approach
can reach equivalent results within subjects exposed to natural stimuli. We
capitalize on the recently-discovered similarities between deep language models
and the human brain to compute the mapping between i) the brain responses to
regular speech and ii) the activations of deep language models elicited by
modified stimuli (e.g. scrambled words, sentences, or paragraphs). Our
model-based approach successfully replicates the seminal study of Lerner et al.
(2011), which revealed the hierarchy of language areas by comparing the
functional-magnetic resonance imaging (fMRI) of seven subjects listening to
7min of both regular and scrambled narratives. We further extend and precise
these results to the brain signals of 305 individuals listening to 4.1 hours of
narrated stories. Overall, this study paves the way for efficient and flexible
analyses of the brain bases of language.

    

### [[2110.06081] Expressivity and Trainability of Quadratic Networks](http://arxiv.org/abs/2110.06081)


  Inspired by diversity of biological neurons, quadratic artificial neurons can
play an important role in deep learning models. The type of quadratic neurons
of our interest replaces the inner-product operation in the conventional neuron
with a quadratic function. Despite promising results so far achieved by
networks of quadratic neurons, there are important issues not well addressed.
Theoretically, the superior expressivity of a quadratic network over either a
conventional network or a conventional network via quadratic activation is not
fully elucidated, which makes the use of quadratic networks not well grounded.
Practically, although a quadratic network can be trained via generic
backpropagation, it can be subject to a higher risk of collapse than the
conventional counterpart. To address these issues, we first apply the spline
theory and a measure from algebraic geometry to give two theorems that
demonstrate better model expressivity of a quadratic network than the
conventional counterpart with or without quadratic activation. Then, we propose
an effective and efficient training strategy referred to as ReLinear to
stabilize the training process of a quadratic network, thereby unleashing the
full potential in its associated machine learning tasks. Comprehensive
experiments on popular datasets are performed to support our findings and
evaluate the performance of quadratic deep learning.

    

### [[2110.06082] Efficient Bayesian network structure learning via local Markov boundary search](http://arxiv.org/abs/2110.06082)


  We analyze the complexity of learning directed acyclic graphical models from
observational data in general settings without specific distributional
assumptions. Our approach is information-theoretic and uses a local Markov
boundary search procedure in order to recursively construct ancestral sets in
the underlying graphical model. Perhaps surprisingly, we show that for certain
graph ensembles, a simple forward greedy search algorithm (i.e. without a
backward pruning phase) suffices to learn the Markov boundary of each node.
This substantially improves the sample complexity, which we show is at most
polynomial in the number of nodes. This is then applied to learn the entire
graph under a novel identifiability condition that generalizes existing
conditions from the literature. As a matter of independent interest, we
establish finite-sample guarantees for the problem of recovering Markov
boundaries from data. Moreover, we apply our results to the special case of
polytrees, for which the assumptions simplify, and provide explicit conditions
under which polytrees are identifiable and learnable in polynomial time. We
further illustrate the performance of the algorithm, which is easy to
implement, in a simulation study. Our approach is general, works for discrete
or continuous distributions without distributional assumptions, and as such
sheds light on the minimal assumptions required to efficiently learn the
structure of directed graphical models from data.

    

### [[2110.06084] Implicit Bias of Linear Equivariant Networks](http://arxiv.org/abs/2110.06084)


  Group equivariant convolutional neural networks (G-CNNs) are generalizations
of convolutional neural networks (CNNs) which excel in a wide range of
scientific and technical applications by explicitly encoding group symmetries,
such as rotations and permutations, in their architectures. Although the
success of G-CNNs is driven by the explicit symmetry bias of their
convolutional architecture, a recent line of work has proposed that the
implicit bias of training algorithms on a particular parameterization (or
architecture) is key to understanding generalization for overparameterized
neural nets. In this context, we show that $L$-layer full-width linear G-CNNs
trained via gradient descent in a binary classification task converge to
solutions with low-rank Fourier matrix coefficients, regularized by the
$2/L$-Schatten matrix norm. Our work strictly generalizes previous analysis on
the implicit bias of linear CNNs to linear G-CNNs over all finite groups,
including the challenging setting of non-commutative symmetry groups (such as
permutations). We validate our theorems via experiments on a variety of groups
and empirically explore more realistic nonlinear networks, which locally
capture similar regularization patterns. Finally, we provide intuitive
interpretations of our Fourier space implicit regularization results in real
space via uncertainty principles.

    

### [[2110.06088] ConTIG: Continuous Representation Learning on Temporal Interaction Graphs](http://arxiv.org/abs/2110.06088)


  Representation learning on temporal interaction graphs (TIG) is to model
complex networks with the dynamic evolution of interactions arising in a broad
spectrum of problems. Existing dynamic embedding methods on TIG discretely
update node embeddings merely when an interaction occurs. They fail to capture
the continuous dynamic evolution of embedding trajectories of nodes. In this
paper, we propose a two-module framework named ConTIG, a continuous
representation method that captures the continuous dynamic evolution of node
embedding trajectories. With two essential modules, our model exploit
three-fold factors in dynamic networks which include latest interaction,
neighbor features and inherent characteristics. In the first update module, we
employ a continuous inference block to learn the nodes' state trajectories by
learning from time-adjacent interaction patterns between node pairs using
ordinary differential equations. In the second transform module, we introduce a
self-attention mechanism to predict future node embeddings by aggregating
historical temporal interaction information. Experiments results demonstrate
the superiority of ConTIG on temporal link prediction, temporal node
recommendation and dynamic node classification tasks compared with a range of
state-of-the-art baselines, especially for long-interval interactions
prediction.

    

### [[2110.06089] Cubature Kalman Filter Based Training of Hybrid Differential Equation Recurrent Neural Network Physiological Dynamic Models](http://arxiv.org/abs/2110.06089)


  Modeling biological dynamical systems is challenging due to the
interdependence of different system components, some of which are not fully
understood. To fill existing gaps in our ability to mechanistically model
physiological systems, we propose to combine neural networks with physics-based
models. Specifically, we demonstrate how we can approximate missing ordinary
differential equations (ODEs) coupled with known ODEs using Bayesian filtering
techniques to train the model parameters and simultaneously estimate dynamic
state variables. As a study case we leverage a well-understood model for blood
circulation in the human retina and replace one of its core ODEs with a neural
network approximation, representing the case where we have incomplete knowledge
of the physiological state dynamics. Results demonstrate that state dynamics
corresponding to the missing ODEs can be approximated well using a neural
network trained using a recursive Bayesian filtering approach in a fashion
coupled with the known state dynamic differential equations. This demonstrates
that dynamics and impact of missing state variables can be captured through
joint state estimation and model parameter estimation within a recursive
Bayesian state estimation (RBSE) framework. Results also indicate that this
RBSE approach to training the NN parameters yields better outcomes
(measurement/state estimation accuracy) than training the neural network with
backpropagation through time in the same setting.

    

### [[2110.06116] Two-level monotonic multistage recommender systems](http://arxiv.org/abs/2110.06116)


  A recommender system learns to predict the user-specific preference or
intention over many items simultaneously for all users, making personalized
recommendations based on a relatively small number of observations. One central
issue is how to leverage three-way interactions, referred to as user-item-stage
dependencies on a monotonic chain of events, to enhance the prediction
accuracy. A monotonic chain of events occurs, for instance, in an article
sharing dataset, where a ``follow'' action implies a ``like'' action, which in
turn implies a ``view'' action. In this article, we develop a multistage
recommender system utilizing a two-level monotonic property characterizing a
monotonic chain of events for personalized prediction. Particularly, we derive
a large-margin classifier based on a nonnegative additive latent factor model
in the presence of a high percentage of missing observations, particularly
between stages, reducing the number of model parameters for personalized
prediction while guaranteeing prediction consistency. On this ground, we derive
a regularized cost function to learn user-specific behaviors at different
stages, linking decision functions to numerical and categorical covariates to
model user-item-stage interactions. Computationally, we derive an algorithm
based on blockwise coordinate descent. Theoretically, we show that the
two-level monotonic property enhances the accuracy of learning as compared to a
standard method treating each stage individually and an ordinal method
utilizing only one-level monotonicity. Finally, the proposed method compares
favorably with existing methods in simulations and an article sharing dataset.

    

### [[2110.06117] Live Multi-Streaming and Donation Recommendations via Coupled Donation-Response Tensor Factorization](http://arxiv.org/abs/2110.06117)


  In contrast to traditional online videos, live multi-streaming supports
real-time social interactions between multiple streamers and viewers, such as
donations. However, donation and multi-streaming channel recommendations are
challenging due to complicated streamer and viewer relations, asymmetric
communications, and the tradeoff between personal interests and group
interactions. In this paper, we introduce Multi-Stream Party (MSP) and
formulate a new multi-streaming recommendation problem, called Donation and MSP
Recommendation (DAMRec). We propose Multi-stream Party Recommender System
(MARS) to extract latent features via socio-temporal coupled donation-response
tensor factorization for donation and MSP recommendations. Experimental results
on Twitch and Douyu manifest that MARS significantly outperforms existing
recommenders by at least 38.8% in terms of hit ratio and mean average
precision.

    

### [[2110.06122] Nonnegative spatial factorization](http://arxiv.org/abs/2110.06122)


  Gaussian processes are widely used for the analysis of spatial data due to
their nonparametric flexibility and ability to quantify uncertainty, and
recently developed scalable approximations have facilitated application to
massive datasets. For multivariate outcomes, linear models of coregionalization
combine dimension reduction with spatial correlation. However, their
real-valued latent factors and loadings are difficult to interpret because,
unlike nonnegative models, they do not recover a parts-based representation. We
present nonnegative spatial factorization (NSF), a spatially-aware
probabilistic dimension reduction model that naturally encourages sparsity. We
compare NSF to real-valued spatial factorizations such as MEFISTO and
nonspatial dimension reduction methods using simulations and high-dimensional
spatial transcriptomics data. NSF identifies generalizable spatial patterns of
gene expression. Since not all patterns of gene expression are spatial, we also
propose a hybrid extension of NSF that combines spatial and nonspatial
components, enabling quantification of spatial importance for both observations
and features. A TensorFlow implementation of NSF is available from
this https URL .

    

### [[2110.06125] Embracing Structure in Data for Billion-Scale Semantic Product Search](http://arxiv.org/abs/2110.06125)


  We present principled approaches to train and deploy dyadic neural embedding
models at the billion scale, focusing our investigation on the application of
semantic product search. When training a dyadic model, one seeks to embed two
different types of entities (e.g., queries and documents or users and movies)
in a common vector space such that pairs with high relevance are positioned
nearby. During inference, given an embedding of one type (e.g., a query or a
user), one seeks to retrieve the entities of the other type (e.g., documents or
movies, respectively) that are highly relevant. In this work, we show that
exploiting the natural structure of real-world datasets helps address both
challenges efficiently. Specifically, we model dyadic data as a bipartite graph
with edges between pairs with positive associations. We then propose to
partition this network into semantically coherent clusters and thus reduce our
search space by focusing on a small subset of these partitions for a given
input. During training, this technique enables us to efficiently mine hard
negative examples while, at inference, we can quickly find the nearest
neighbors for a given embedding. We provide offline experimental results that
demonstrate the efficacy of our techniques for both training and inference on a
billion-scale this http URL product search dataset.

    

### [[2110.06126] Spatial mixup: Directional loudness modification as data augmentation for sound event localization and detection](http://arxiv.org/abs/2110.06126)


  Data augmentation methods have shown great importance in diverse supervised
learning problems where labeled data is scarce or costly to obtain. For sound
event localization and detection (SELD) tasks several augmentation methods have
been proposed, with most borrowing ideas from other domains such as images,
speech, or monophonic audio. However, only a few exploit the spatial properties
of a full 3D audio scene. We propose Spatial Mixup, as an application of
parametric spatial audio effects for data augmentation, which modifies the
directional properties of a multi-channel spatial audio signal encoded in the
ambisonics domain. Similarly to beamforming, these modifications enhance or
suppress signals arriving from certain directions, although the effect is less
pronounced. Therefore enabling deep learning models to achieve invariance to
small spatial perturbations. The method is evaluated with experiments in the
DCASE 2021 Task 3 dataset, where spatial mixup increases performance over a
non-augmented baseline, and compares to other well known augmentation methods.
Furthermore, combining spatial mixup with other methods greatly improves
performance.

    

### [[2110.06131] Fetal Gender Identification using Machine and Deep Learning Algorithms on Phonocardiogram Signals](http://arxiv.org/abs/2110.06131)


  Phonocardiogram (PCG) signal analysis is a critical, widely-studied
technology to noninvasively analyze the heart's mechanical activity. Through
evaluating heart sounds, this technology has been chiefly leveraged as a
preliminary solution to automatically diagnose Cardiovascular diseases among
adults; however, prenatal tasks such as fetal gender identification have been
relatively less studied using fetal Phonocardiography (FPCG). In this work, we
apply common PCG signal processing techniques on the gender-tagged Shiraz
University Fetal Heart Sounds Database and study the applicability of
previously proposed features in classifying fetal gender using both Machine
Learning and Deep Learning models. Even though PCG data acquisition's
cost-effectiveness and feasibility make it a convenient method of Fetal Heart
Rate (FHR) monitoring, the contaminated nature of PCG signals with the noise of
various types makes it a challenging modality. To address this problem, we
experimented with both static and adaptive noise reduction techniques such as
Low-pass filtering, Denoising Autoencoders, and Source Separators. We apply a
wide range of previously proposed classifiers to our dataset and propose a
novel ensemble method of Fetal Gender Identification (FGI). Our method
substantially outperformed the baseline and reached up to 91% accuracy in
classifying fetal gender of unseen subjects.

    

### [[2110.06135] Label scarcity in biomedicine: Data-rich latent factor discovery enhances phenotype prediction](http://arxiv.org/abs/2110.06135)


  High-quality data accumulation is now becoming ubiquitous in the health
domain. There is increasing opportunity to exploit rich data from normal
subjects to improve supervised estimators in specific diseases with notorious
data scarcity. We demonstrate that low-dimensional embedding spaces can be
derived from the UK Biobank population dataset and used to enhance data-scarce
prediction of health indicators, lifestyle and demographic characteristics.
Phenotype predictions facilitated by Variational Autoencoder manifolds
typically scaled better with increasing unlabeled data than dimensionality
reduction by PCA or Isomap. Performances gains from semisupervison approaches
will probably become an important ingredient for various medical data science
applications.

    

### [[2110.06137] An Activity Recognition Framework for Continuous Monitoring of Non-Steady-State Locomotion of Individuals with Parkinson's Disease](http://arxiv.org/abs/2110.06137)


  Fundamental knowledge in activity recognition of individuals with motor
disorders such as Parkinson's disease (PD) has been primarily limited to
detection of steady-state/static tasks (sitting, standing, walking). To date,
identification of non-steady-state locomotion on uneven terrains (stairs,
ramps) has not received much attention. Furthermore, previous research has
mainly relied on data from a large number of body locations which could
adversely affect user convenience and system performance. Here, individuals
with mild stages of PD and healthy subjects performed non-steady-state circuit
trials comprising stairs, ramp, and changes of direction. An offline analysis
using a linear discriminant analysis (LDA) classifier and a Long-Short Term
Memory (LSTM) neural network was performed for task recognition. The
performance of accelerographic and gyroscopic information from varied
lower/upper-body segments were tested across a set of user-independent and
user-dependent training paradigms. Comparing the F1 score of a given signal
across classifiers showed improved performance using LSTM compared to LDA.
Using LSTM, even a subset of information (e.g., feet data) in
subject-independent training appeared to provide F1 score > 0.8. However,
employing LDA was shown to be at the expense of being limited to using a
subject-dependent training and/or biomechanical data from multiple body
locations. The findings could inform a number of applications in the field of
healthcare monitoring and developing advanced lower-limb assistive devices by
providing insights into classification schemes capable of handling
non-steady-state and unstructured locomotion in individuals with mild
Parkinson's disease.

    

### [[2110.06139] Classification of anomalous gait using Machine Learning techniques and embedded sensors](http://arxiv.org/abs/2110.06139)


  Human gait can be a predictive factor for detecting pathologies that affect
human locomotion according to studies. In addition, it is known that a high
investment is demanded in order to raise a traditional clinical infrastructure
able to provide human gait examinations, making them unaffordable for
economically vulnerable patients. In face of this scenario, this work proposes
an accessible and modern solution composed of a wearable device, to acquire
3D-accelerometer and 3D-gyroscope measurements, and machine learning techniques
to classify between distinct categories of induced gait disorders. In order to
develop the proposed research, it was created a dataset with the target label
being 4 distinct and balanced categories of anomalous gait. The machine
learning techniques that achieved the best performances (in terms of accuracy)
in this dataset were through the application of Principal Component Analysis
algorithm following of a Support Vector Machines classifier (94 \%). Further,
an architecture based on a Feedforward Neural Network yielded even better
results (96 \%). Finally, it is also presented computational performance
comparison between the models implemented.

    

### [[2110.06140] EEG functional connectivity and deep learning for automatic diagnosis of brain disorders: Alzheimer's disease and schizophrenia](http://arxiv.org/abs/2110.06140)


  Mental disorders are among the leading causes of disability worldwide. The
first step in treating these conditions is to obtain an accurate diagnosis, but
the absence of established clinical tests makes this task challenging. Machine
learning algorithms can provide a possible solution to this problem, as we
describe in this work. We present a method for the automatic diagnosis of
mental disorders based on the matrix of connections obtained from EEG time
series and deep learning. We show that our approach can classify patients with
Alzheimer's disease and schizophrenia with a high level of accuracy. The
comparison with the traditional cases, that use raw EEG time series, shows that
our method provides the highest precision. Therefore, the application of deep
neural networks on data from brain connections is a very promising method to
the diagnosis of neurological disorders.

    

### [[2110.06149] Planning from Pixels in Environments with Combinatorially Hard Search Spaces](http://arxiv.org/abs/2110.06149)


  The ability to form complex plans based on raw visual input is a litmus test
for current capabilities of artificial intelligence, as it requires a seamless
combination of visual processing and abstract algorithmic execution, two
traditionally separate areas of computer science. A recent surge of interest in
this field brought advances that yield good performance in tasks ranging from
arcade games to continuous control; these methods however do not come without
significant issues, such as limited generalization capabilities and
difficulties when dealing with combinatorially hard planning instances. Our
contribution is two-fold: (i) we present a method that learns to represent its
environment as a latent graph and leverages state reidentification to reduce
the complexity of finding a good policy from exponential to linear (ii) we
introduce a set of lightweight environments with an underlying discrete
combinatorial structure in which planning is challenging even for humans.
Moreover, we show that our methods achieves strong empirical generalization to
variations in the environment, even across highly disadvantaged regimes, such
as "one-shot" planning, or in an offline RL paradigm which only provides
low-quality trajectories.

    

### [[2110.06150] Sparsity in Partially Controllable Linear Systems](http://arxiv.org/abs/2110.06150)


  A fundamental concept in control theory is that of controllability, where any
system state can be reached through an appropriate choice of control inputs.
Indeed, a large body of classical and modern approaches are designed for
controllable linear dynamical systems. However, in practice, we often encounter
systems in which a large set of state variables evolve exogenously and
independently of the control inputs; such systems are only \emph{partially
controllable}. The focus of this work is on a large class of partially
controllable linear dynamical systems, specified by an underlying sparsity
pattern. Our main results establish structural conditions and finite-sample
guarantees for learning to control such systems. In particular, our structural
results characterize those state variables which are irrelevant for optimal
control, an analysis which departs from classical control techniques. Our
algorithmic results adapt techniques from high-dimensional statistics --
specifically soft-thresholding and semiparametric least-squares -- to exploit
the underlying sparsity pattern in order to obtain finite-sample guarantees
that significantly improve over those based on certainty-equivalence. We also
corroborate these theoretical improvements over certainty-equivalent control
through a simulation study.

    

### [[2110.06163] Finding Relevant Points for Nearest-Neighbor Classification](http://arxiv.org/abs/2110.06163)


  In nearest-neighbor classification problems, a set of $d$-dimensional
training points are given, each with a known classification, and are used to
infer unknown classifications of other points by using the same classification
as the nearest training point. A training point is relevant if its omission
from the training set would change the outcome of some of these inferences. We
provide a simple algorithm for thinning a training set down to its subset of
relevant points, using as subroutines algorithms for finding the minimum
spanning tree of a set of points and for finding the extreme points (convex
hull vertices) of a set of points. The time bounds for our algorithm, in any
constant dimension $d\ge 3$, improve on a previous algorithm for the same
problem by Clarkson (FOCS 1994).

    

### [[2110.06166] Game Theory for Adversarial Attacks and Defenses](http://arxiv.org/abs/2110.06166)


  Adversarial attacks can generate adversarial inputs by applying small but
intentionally worst-case perturbations to samples from the dataset, which leads
to even state-of-the-art deep neural networks outputting incorrect answers with
high confidence. Hence, some adversarial defense techniques are developed to
improve the security and robustness of the models and avoid them being
attacked. Gradually, a game-like competition between attackers and defenders
formed, in which both players would attempt to play their best strategies
against each other while maximizing their own payoffs. To solve the game, each
player would choose an optimal strategy against the opponent based on the
prediction of the opponent's strategy choice. In this work, we are on the
defensive side to apply game-theoretic approaches on defending against attacks.
We use two randomization methods, random initialization and stochastic
activation pruning, to create diversity of networks. Furthermore, we use one
denoising technique, super resolution, to improve models' robustness by
preprocessing images before attacks. Our experimental results indicate that
those three methods can effectively improve the robustness of deep-learning
neural networks.

    

### [[2110.06169] Offline Reinforcement Learning with Implicit Q-Learning](http://arxiv.org/abs/2110.06169)


  Offline reinforcement learning requires reconciling two conflicting aims:
learning a policy that improves over the behavior policy that collected the
dataset, while at the same time minimizing the deviation from the behavior
policy so as to avoid errors due to distributional shift. This trade-off is
critical, because most current offline reinforcement learning methods need to
query the value of unseen actions during training to improve the policy, and
therefore need to either constrain these actions to be in-distribution, or else
regularize their values. We propose an offline RL method that never needs to
evaluate actions outside of the dataset, but still enables the learned policy
to improve substantially over the best behavior in the data through
generalization. The main insight in our work is that, instead of evaluating
unseen actions from the latest policy, we can approximate the policy
improvement step implicitly by treating the state value function as a random
variable, with randomness determined by the action (while still integrating
over the dynamics to avoid excessive optimism), and then taking a state
conditional upper expectile of this random variable to estimate the value of
the best actions in that state. This leverages the generalization capacity of
the function approximator to estimate the value of the best available action at
a given state without ever directly querying a Q-function with this unseen
action. Our algorithm alternates between fitting this upper expectile value
function and backing it up into a Q-function. Then, we extract the policy via
advantage-weighted behavioral cloning. We dub our method implicit Q-learning
(IQL). IQL demonstrates the state-of-the-art performance on D4RL, a standard
benchmark for offline reinforcement learning. We also demonstrate that IQL
achieves strong performance fine-tuning using online interaction after offline
initialization.

    

### [[2110.06176] Mention Memory: incorporating textual knowledge into Transformers through entity mention attention](http://arxiv.org/abs/2110.06176)


  Natural language understanding tasks such as open-domain question answering
often require retrieving and assimilating factual information from multiple
sources. We propose to address this problem by integrating a semi-parametric
representation of a large text corpus into a Transformer model as a source of
factual knowledge. Specifically, our method represents knowledge with `mention
memory', a table of dense vector representations of every entity mention in a
corpus. The proposed model - TOME - is a Transformer that accesses the
information through internal memory layers in which each entity mention in the
input passage attends to the mention memory. This approach enables synthesis of
and reasoning over many disparate sources of information within a single
Transformer model. In experiments using a memory of 150 million Wikipedia
mentions, TOME achieves strong performance on several open-domain
knowledge-intensive tasks, including the claim verification benchmarks HoVer
and FEVER and several entity-based QA benchmarks. We also show that the model
learns to attend to informative mentions without any direct supervision.
Finally we demonstrate that the model can generalize to new unseen entities by
updating the memory without retraining.

    

### [[2110.06177] Tracking the risk of a deployed model and detecting harmful distribution shifts](http://arxiv.org/abs/2110.06177)


  When deployed in the real world, machine learning models inevitably encounter
changes in the data distribution, and certain -- but not all -- distribution
shifts could result in significant performance degradation. In practice, it may
make sense to ignore benign shifts, under which the performance of a deployed
model does not degrade substantially, making interventions by a human expert
(or model retraining) unnecessary. While several works have developed tests for
distribution shifts, these typically either use non-sequential methods, or
detect arbitrary shifts (benign or harmful), or both. We argue that a sensible
method for firing off a warning has to both (a) detect harmful shifts while
ignoring benign ones, and (b) allow continuous monitoring of model performance
without increasing the false alarm rate. In this work, we design simple
sequential tools for testing if the difference between source (training) and
target (test) distributions leads to a significant drop in a risk function of
interest, like accuracy or calibration. Recent advances in constructing
time-uniform confidence sequences allow efficient aggregation of statistical
evidence accumulated during the tracking process. The designed framework is
applicable in settings where (some) true labels are revealed after the
prediction is performed, or when batches of labels become available in a
delayed fashion. We demonstrate the efficacy of the proposed framework through
an extensive empirical study on a collection of simulated and real datasets.

    

### [[2110.06192] Beyond Pick-and-Place: Tackling Robotic Stacking of Diverse Shapes](http://arxiv.org/abs/2110.06192)


  We study the problem of robotic stacking with objects of complex geometry. We
propose a challenging and diverse set of such objects that was carefully
designed to require strategies beyond a simple "pick-and-place" solution. Our
method is a reinforcement learning (RL) approach combined with vision-based
interactive policy distillation and simulation-to-reality transfer. Our learned
policies can efficiently handle multiple object combinations in the real world
and exhibit a large variety of stacking skills. In a large experimental study,
we investigate what choices matter for learning such general vision-based
agents in simulation, and what affects optimal transfer to the real robot. We
then leverage data collected by such policies and improve upon them with
offline RL. A video and a blog post of our work are provided as supplementary
material.

    

### [[2110.06196] GraPE: fast and scalable Graph Processing and Embedding](http://arxiv.org/abs/2110.06196)


  Graph Representation Learning methods have enabled a wide range of learning
problems to be addressed for data that can be represented in graph form.
Nevertheless, several real world problems in economy, biology, medicine and
other fields raised relevant scaling problems with existing methods and their
software implementation, due to the size of real world graphs characterized by
millions of nodes and billions of edges. We present GraPE, a software resource
for graph processing and random walk based embedding, that can scale with large
and high-degree graphs and significantly speed up-computation. GraPE comprises
specialized data structures, algorithms, and a fast parallel implementation
that displays everal orders of magnitude improvement in empirical space and
time complexity compared to state of the art software resources, with a
corresponding boost in the performance of machine learning methods for edge and
node label prediction and for the unsupervised analysis of graphs.GraPE is
designed to run on laptop and desktop computers, as well as on high performance
computing clusters

    

### [[2110.06197] Crystal Diffusion Variational Autoencoder for Periodic Material Generation](http://arxiv.org/abs/2110.06197)


  Generating the periodic structure of stable materials is a long-standing
challenge for the material design community. This task is difficult because
stable materials only exist in a low-dimensional subspace of all possible
periodic arrangements of atoms: 1) the coordinates must lie in the local energy
minimum defined by quantum mechanics, and 2) global stability also requires the
structure to follow the complex, yet specific bonding preferences between
different atom types. Existing methods fail to incorporate these factors and
often lack proper invariances. We propose a Crystal Diffusion Variational
Autoencoder (CDVAE) that captures the physical inductive bias of material
stability. By learning from the data distribution of stable materials, the
decoder generates materials in a diffusion process that moves atomic
coordinates towards a lower energy state and updates atom types to satisfy
bonding preferences between neighbors. Our model also explicitly encodes
interactions across periodic boundaries and respects permutation, translation,
rotation, and periodic invariances. We significantly outperform past methods in
three tasks: 1) reconstructing the input structure, 2) generating valid,
diverse, and realistic materials, and 3) generating materials that optimize a
specific property. We also provide several standard datasets and evaluation
metrics for the broader machine learning community.

    

### [[2110.06198] Last Iterate Risk Bounds of SGD with Decaying Stepsize for Overparameterized Linear Regression](http://arxiv.org/abs/2110.06198)


  Stochastic gradient descent (SGD) has been demonstrated to generalize well in
many deep learning applications. In practice, one often runs SGD with a
geometrically decaying stepsize, i.e., a constant initial stepsize followed by
multiple geometric stepsize decay, and uses the last iterate as the output.
This kind of SGD is known to be nearly minimax optimal for classical
finite-dimensional linear regression problems (Ge et al., 2019), and provably
outperforms SGD with polynomially decaying stepsize in terms of the statistical
minimax rates. However, a sharp analysis for the last iterate of SGD with
decaying step size in the overparameterized setting is still open. In this
paper, we provide problem-dependent analysis on the last iterate risk bounds of
SGD with decaying stepsize, for (overparameterized) linear regression problems.
In particular, for SGD with geometrically decaying stepsize (or tail
geometrically decaying stepsize), we prove nearly matching upper and lower
bounds on the excess risk. Our results demonstrate the generalization ability
of SGD for a wide class of overparameterized problems, and can recover the
minimax optimal results up to logarithmic factors in the classical regime.
Moreover, we provide an excess risk lower bound for SGD with polynomially
decaying stepsize and illustrate the advantage of geometrically decaying
stepsize in an instance-wise manner, which complements the minimax rate
comparison made in previous work.

    

### [[2110.06206] StARformer: Transformer with State-Action-Reward Representations](http://arxiv.org/abs/2110.06206)


  Reinforcement Learning (RL) can be considered as a sequence modeling task,
i.e., given a sequence of past state-action-reward experiences, a model
autoregressively predicts a sequence of future actions. Recently, Transformers
have been successfully adopted to model this problem. In this work, we propose
State-Action-Reward Transformer (StARformer), which explicitly models local
causal relations to help improve action prediction in long sequences.
StARformer first extracts local representations (i.e., StAR-representations)
from each group of state-action-reward tokens within a very short time span. A
sequence of such local representations combined with state representations, is
then used to make action predictions over a long time span. Our experiments
show that StARformer outperforms the state-of-the-art Transformer-based method
on Atari (image) and Gym (state vector) benchmarks, in both offline-RL and
imitation learning settings. StARformer is also more compliant with longer
sequences of inputs compared to the baseline. Our code is available at
this https URL.

    

### [[1703.01014] Active Learning for Cost-Sensitive Classification](http://arxiv.org/abs/1703.01014)


  We design an active learning algorithm for cost-sensitive multiclass
classification: problems where different errors have different costs. Our
algorithm, COAL, makes predictions by regressing to each label's cost and
predicting the smallest. On a new example, it uses a set of regressors that
perform well on past data to estimate possible costs for each label. It queries
only the labels that could be the best, ignoring the sure losers. We prove COAL
can be efficiently implemented for any regression family that admits squared
loss optimization; it also enjoys strong guarantees with respect to predictive
performance and labeling effort. We empirically compare COAL to passive
learning and several active learning baselines, showing significant
improvements in labeling effort and test cost on real-world datasets.

    

### [[1906.09501] Learning partial correlation graphs and graphical models by covariance queries](http://arxiv.org/abs/1906.09501)


  We study the problem of recovering the structure underlying large Gaussian
graphical models or, more generally, partial correlation graphs. In
high-dimensional problems it is often too costly to store the entire sample
covariance matrix. We propose a new input model in which one can query single
entries of the covariance matrix. We prove that it is possible to recover the
support of the inverse covariance matrix with low query and computational
complexity. Our algorithms work in a regime when this support is represented by
tree-like graphs and, more generally, for graphs of small treewidth. Our
results demonstrate that for large classes of graphs, the structure of the
corresponding partial correlation graphs can be determined much faster than
even computing the empirical covariance matrix.

    

### [[1912.05695] Randomized Exploration for Non-Stationary Stochastic Linear Bandits](http://arxiv.org/abs/1912.05695)


  We investigate two perturbation approaches to overcome conservatism that
optimism based algorithms chronically suffer from in practice. The first
approach replaces optimism with a simple randomization when using confidence
sets. The second one adds random perturbations to its current estimate before
maximizing the expected reward. For non-stationary linear bandits, where each
action is associated with a $d$-dimensional feature and the unknown parameter
is time-varying with total variation $B_T$, we propose two randomized
algorithms, Discounted Randomized LinUCB (D-RandLinUCB) and Discounted Linear
Thompson Sampling (D-LinTS) via the two perturbation approaches. We highlight
the statistical optimality versus computational efficiency trade-off between
them in that the former asymptotically achieves the optimal dynamic regret
$\tilde{O}(d^{7/8} B_T^{1/4}T^{3/4})$, but the latter is oracle-efficient with
an extra logarithmic factor in the number of arms compared to minimax-optimal
dynamic regret. In a simulation study, both algorithms show outstanding
performance in tackling conservatism issue that Discounted LinUCB struggles
with.

    

### [[2001.02798] Self-guided Approximate Linear Programs](http://arxiv.org/abs/2001.02798)


  Approximate linear programs (ALPs) are well-known models based on value
function approximations (VFAs) to obtain policies and lower bounds on the
optimal policy cost of discounted-cost Markov decision processes (MDPs).
Formulating an ALP requires (i) basis functions, the linear combination of
which defines the VFA, and (ii) a state-relevance distribution, which
determines the relative importance of different states in the ALP objective for
the purpose of minimizing VFA error. Both these choices are typically
heuristic: basis function selection relies on domain knowledge while the
state-relevance distribution is specified using the frequency of states visited
by a heuristic policy. We propose a self-guided sequence of ALPs that embeds
random basis functions obtained via inexpensive sampling and uses the known VFA
from the previous iteration to guide VFA computation in the current iteration.
Self-guided ALPs mitigate the need for domain knowledge during basis function
selection as well as the impact of the initial choice of the state-relevance
distribution, thus significantly reducing the ALP implementation burden. We
establish high probability error bounds on the VFAs from this sequence and show
that a worst-case measure of policy performance is improved. We find that these
favorable implementation and theoretical properties translate to encouraging
numerical results on perishable inventory control and options pricing
applications, where self-guided ALP policies improve upon policies from
problem-specific methods. More broadly, our research takes a meaningful step
toward application-agnostic policies and bounds for MDPs.

    

### [[2001.08922] RePAD: Real-time Proactive Anomaly Detection for Time Series](http://arxiv.org/abs/2001.08922)


  During the past decade, many anomaly detection approaches have been
introduced in different fields such as network monitoring, fraud detection, and
intrusion detection. However, they require understanding of data pattern and
often need a long off-line period to build a model or network for the target
data. Providing real-time and proactive anomaly detection for streaming time
series without human intervention and domain knowledge is highly valuable since
it greatly reduces human effort and enables appropriate countermeasures to be
undertaken before a disastrous damage, failure, or other harmful event occurs.
However, this issue has not been well studied yet. To address it, this paper
proposes RePAD, which is a Real-time Proactive Anomaly Detection algorithm for
streaming time series based on Long Short-Term Memory (LSTM). RePAD utilizes
short-term historic data points to predict and determine whether or not the
upcoming data point is a sign that an anomaly is likely to happen in the near
future. By dynamically adjusting the detection threshold over time, RePAD is
able to tolerate minor pattern change in time series and detect anomalies
either proactively or on time. Experiments based on two time series datasets
collected from the Numenta Anomaly Benchmark demonstrate that RePAD is able to
proactively detect anomalies and provide early warnings in real time without
human intervention and domain knowledge.

    

### [[2004.02319] ReRe: A Lightweight Real-time Ready-to-Go Anomaly Detection Approach for Time Series](http://arxiv.org/abs/2004.02319)


  Anomaly detection is an active research topic in many different fields such
as intrusion detection, network monitoring, system health monitoring, IoT
healthcare, etc. However, many existing anomaly detection approaches require
either human intervention or domain knowledge, and may suffer from high
computation complexity, consequently hindering their applicability in
real-world scenarios. Therefore, a lightweight and ready-to-go approach that is
able to detect anomalies in real-time is highly sought-after. Such an approach
could be easily and immediately applied to perform time series anomaly
detection on any commodity machine. The approach could provide timely anomaly
alerts and by that enable appropriate countermeasures to be undertaken as early
as possible. With these goals in mind, this paper introduces ReRe, which is a
Real-time Ready-to-go proactive Anomaly Detection algorithm for streaming time
series. ReRe employs two lightweight Long Short-Term Memory (LSTM) models to
predict and jointly determine whether or not an upcoming data point is
anomalous based on short-term historical data points and two long-term
self-adaptive thresholds. Experiments based on real-world time-series datasets
demonstrate the good performance of ReRe in real-time anomaly detection without
requiring human intervention or domain knowledge.

    

### [[2006.02624] Bayesian optimization for modular black-box systems with switching costs](http://arxiv.org/abs/2006.02624)


  Most existing black-box optimization methods assume that all variables in the
system being optimized have equal cost and can change freely at each iteration.
However, in many real world systems, inputs are passed through a sequence of
different operations or modules, making variables in earlier stages of
processing more costly to update. Such structure imposes a cost on switching
variables in early parts of a data processing pipeline. In this work, we
propose a new algorithm for switch cost-aware optimization called Lazy Modular
Bayesian Optimization (LaMBO). This method efficiently identifies the global
optimum while minimizing cost through a passive change of variables in early
modules. The method is theoretical grounded and achieves vanishing regret when
augmented with switching cost. We apply LaMBO to multiple synthetic functions
and a three-stage image segmentation pipeline used in a neuroscience
application, where we obtain promising improvements over prevailing cost-aware
Bayesian optimization algorithms. Our results demonstrate that LaMBO is an
effective strategy for black-box optimization that is capable of minimizing
switching costs in modular systems.

    

### [[2006.07040] Learning Decomposed Representation for Counterfactual Inference](http://arxiv.org/abs/2006.07040)


  The fundamental problem in treatment effect estimation from observational
data is confounder identification and balancing. Most of the previous methods
realized confounder balancing by treating all observed pre-treatment variables
as confounders, ignoring further identifying confounders and non-confounders.
In general, not all the observed pre-treatment variables are confounders that
refer to the common causes of the treatment and the outcome, some variables
only contribute to the treatment and some only contribute to the outcome.
Balancing those non-confounders, including instrumental variables and
adjustment variables, would generate additional bias for treatment effect
estimation. By modeling the different causal relations among observed
pre-treatment variables, treatment and outcome, we propose a synergistic
learning framework to 1) identify confounders by learning decomposed
representations of both confounders and non-confounders, 2) balance confounder
with sample re-weighting technique, and simultaneously 3) estimate the
treatment effect in observational studies via counterfactual inference.
Empirical results on synthetic and real-world datasets demonstrate that the
proposed method can precisely decompose confounders and achieve a more precise
estimation of treatment effect than baselines.

    

### [[2006.07749] Parametric Bootstrap for Differentially Private Confidence Intervals](http://arxiv.org/abs/2006.07749)


  The goal of this paper is to develop a practical and general-purpose approach
to construct confidence intervals for differentially private parametric
estimation. We find that the parametric bootstrap is a simple and effective
solution. It cleanly reasons about variability of both the data sample and the
randomized privacy mechanism and applies "out of the box" to a wide class of
private estimation routines. It can also help correct bias caused by clipping
data to limit sensitivity. We prove that the parametric bootstrap gives
consistent confidence intervals in two broadly relevant settings, including a
novel adaptation to linear regression that avoids accessing the covariate data
multiple times. We demonstrate its effectiveness for a variety of estimators,
and find that it provides confidence intervals with good coverage even at
modest sample sizes and performs better than alternative approaches.

    

### [[2006.08812] Augmented Sliced Wasserstein Distances](http://arxiv.org/abs/2006.08812)


  While theoretically appealing, the application of the Wasserstein distance to
large-scale machine learning problems has been hampered by its prohibitive
computational cost. The sliced Wasserstein distance and its variants improve
the computational efficiency through the random projection, yet they suffer
from low accuracy if the number of projections is not sufficiently large,
because the majority of projections result in trivially small values. In this
work, we propose a new family of distance metrics, called augmented sliced
Wasserstein distances (ASWDs), constructed by first mapping samples to
higher-dimensional hypersurfaces parameterized by neural networks. It is
derived from a key observation that (random) linear projections of samples
residing on these hypersurfaces would translate to much more flexible nonlinear
projections in the original sample space, so they can capture complex
structures of the data distribution. We show that the hypersurfaces can be
optimized by gradient ascent efficiently. We provide the condition under which
the ASWD is a valid metric and show that this can be obtained by an injective
neural network architecture. Numerical results demonstrate that the ASWD
significantly outperforms other Wasserstein variants for both synthetic and
real-world problems.

    

### [[2008.00047] Towards Class-Oriented Poisoning Attacks Against Neural Networks](http://arxiv.org/abs/2008.00047)


  Poisoning attacks on machine learning systems compromise the model
performance by deliberately injecting malicious samples in the training dataset
to influence the training process. Prior works focus on either availability
attacks (i.e., lowering the overall model accuracy) or integrity attacks (i.e.,
enabling specific instance-based backdoor). In this paper, we advance the
adversarial objectives of the availability attacks to a per-class basis, which
we refer to as class-oriented poisoning attacks. We demonstrate that the
proposed attack is capable of forcing the corrupted model to predict in two
specific ways: (i) classify unseen new images to a targeted "supplanter" class,
and (ii) misclassify images from a "victim" class while maintaining the
classification accuracy on other non-victim classes. To maximize the
adversarial effect as well as reduce the computational complexity of poisoned
data generation, we propose a gradient-based framework that crafts poisoning
images with carefully manipulated feature information for each scenario. Using
newly defined metrics at the class level, we demonstrate the effectiveness of
the proposed class-oriented poisoning attacks on various models (e.g., LeNet-5,
Vgg-9, and ResNet-50) over a wide range of datasets (e.g., MNIST, CIFAR-10, and
ImageNet-ILSVRC2012) in an end-to-end training setting.

    

### [[2009.10627] Forecasting elections results via the voter model with stubborn nodes](http://arxiv.org/abs/2009.10627)


  In this paper we propose a novel method to forecast the result of elections
using only official results of previous ones. It is based on the voter model
with stubborn nodes and uses theoretical results developed in a previous work
of ours. We look at popular vote shares for the Conservative and Labour parties
in the UK and the Republican and Democrat parties in the US. We are able to
perform time-evolving estimates of the model parameters and use these to
forecast the vote shares for each party in any election. We obtain a mean
absolute error of 4.74\%. As a side product, our parameters estimates provide
meaningful insight on the political landscape, informing us on the proportion
of voters that are strong supporters of each of the considered parties.

    

### [[2009.12919] Benchmarking deep inverse models over time, and the neural-adjoint method](http://arxiv.org/abs/2009.12919)


  We consider the task of solving generic inverse problems, where one wishes to
determine the hidden parameters of a natural system that will give rise to a
particular set of measurements. Recently many new approaches based upon deep
learning have arisen generating impressive results. We conceptualize these
models as different schemes for efficiently, but randomly, exploring the space
of possible inverse solutions. As a result, the accuracy of each approach
should be evaluated as a function of time rather than a single estimated
solution, as is often done now. Using this metric, we compare several
state-of-the-art inverse modeling approaches on four benchmark tasks: two
existing tasks, one simple task for visualization and one new task from
metamaterial design. Finally, inspired by our conception of the inverse
problem, we explore a solution that uses a deep learning model to approximate
the forward model, and then uses backpropagation to search for good inverse
solutions. This approach, termed the neural-adjoint, achieves the best
performance in many scenarios.

    

### [[2010.01356] Expectigrad: Fast Stochastic Optimization with Robust Convergence Properties](http://arxiv.org/abs/2010.01356)


  Many popular adaptive gradient methods such as Adam and RMSProp rely on an
exponential moving average (EMA) to normalize their stepsizes. While the EMA
makes these methods highly responsive to new gradient information, recent
research has shown that it also causes divergence on at least one convex
optimization problem. We propose a novel method called Expectigrad, which
adjusts stepsizes according to a per-component unweighted mean of all
historical gradients and computes a bias-corrected momentum term jointly
between the numerator and denominator. We prove that Expectigrad cannot diverge
on every instance of the optimization problem known to cause Adam to diverge.
We also establish a regret bound in the general stochastic nonconvex setting
that suggests Expectigrad is less susceptible to gradient variance than
existing methods are. Testing Expectigrad on several high-dimensional machine
learning tasks, we find it often performs favorably to state-of-the-art methods
with little hyperparameter tuning.

    

### [[2010.02990] First-Order Optimization Inspired from Finite-Time Convergent Flows](http://arxiv.org/abs/2010.02990)


  In this paper, we investigate the performance of two first-order optimization
algorithms, obtained from forward Euler discretization of finite-time
optimization flows. These flows are the rescaled-gradient flow (RGF) and the
signed-gradient flow (SGF), and consist of non-Lipscthiz or discontinuous
dynamical systems that converge locally in finite time to the minima of
gradient-dominated functions. We propose an Euler discretization for these
first-order finite-time flows, and provide convergence guarantees, in the
deterministic and the stochastic setting. We then apply the proposed algorithms
to academic examples, as well as deep neural networks training, where we
empirically test their performances on the SVHN dataset. Our results show that
our schemes demonstrate faster convergences against standard optimization
alternatives.

    

### [[2011.05074] Efficient and Transferable Adversarial Examples from Bayesian Neural Networks](http://arxiv.org/abs/2011.05074)


  An established way to improve the transferability of black-box evasion
attacks is to craft the adversarial examples on a surrogate ensemble model to
increase diversity. We argue that transferability is fundamentally related to
epistemic uncertainty. Based on a state-of-the-art Bayesian Deep Learning
technique, we propose a new method to efficiently build a surrogate by sampling
approximately from the posterior distribution of neural network weights, which
represents the belief about the value of each parameter. Our extensive
experiments on ImageNet and CIFAR-10 show that our approach improves the
transfer rates of four state-of-the-art attacks significantly (up to 62.1
percentage points), in both intra-architecture and inter-architecture cases. On
ImageNet, our approach can reach 94% of transfer rate while reducing training
computations from 11.6 to 2.4 exaflops, compared to an ensemble of
independently trained DNNs. Our vanilla surrogate achieves 87.5% of the time
higher transferability than 3 test-time techniques designed for this purpose.
Our work demonstrates that the way to train a surrogate has been overlooked
although it is an important element of transfer-based attacks. We are,
therefore, the first to review the effectiveness of several training methods in
increasing transferability. We provide new directions to better understand the
transferability phenomenon and offer a simple but strong baseline for future
work.

    

### [[2011.07989] Corrupted Contextual Bandits with Action Order Constraints](http://arxiv.org/abs/2011.07989)


  We consider a variant of the novel contextual bandit problem with corrupted
context, which we call the contextual bandit problem with corrupted context and
action correlation, where actions exhibit a relationship structure that can be
exploited to guide the exploration of viable next decisions. Our setting is
primarily motivated by adaptive mobile health interventions and related
applications, where users might transitions through different stages requiring
more targeted action selection approaches. In such settings, keeping user
engagement is paramount for the success of interventions and therefore it is
vital to provide relevant recommendations in a timely manner. The context
provided by users might not always be informative at every decision point and
standard contextual approaches to action selection will incur high regret. We
propose a meta-algorithm using a referee that dynamically combines the policies
of a contextual bandit and multi-armed bandit, similar to previous work, as
wells as a simple correlation mechanism that captures action to action
transition probabilities allowing for more efficient exploration of
time-correlated actions. We evaluate empirically the performance of said
algorithm on a simulation where the sequence of best actions is determined by a
hidden state that evolves in a Markovian manner. We show that the proposed
meta-algorithm improves upon regret in situations where the performance of both
policies varies such that one is strictly superior to the other for a given
time period. To demonstrate that our setting has relevant practical
applicability, we evaluate our method on several real world data sets, clearly
showing better empirical performance compared to a set of simple algorithms.

    

### [[2011.10881] Rethinking Transformer-based Set Prediction for Object Detection](http://arxiv.org/abs/2011.10881)


  DETR is a recently proposed Transformer-based method which views object
detection as a set prediction problem and achieves state-of-the-art performance
but demands extra-long training time to converge. In this paper, we investigate
the causes of the optimization difficulty in the training of DETR. Our
examinations reveal several factors contributing to the slow convergence of
DETR, primarily the issues with the Hungarian loss and the Transformer
cross-attention mechanism. To overcome these issues we propose two solutions,
namely, TSP-FCOS (Transformer-based Set Prediction with FCOS) and TSP-RCNN
(Transformer-based Set Prediction with RCNN). Experimental results show that
the proposed methods not only converge much faster than the original DETR, but
also significantly outperform DETR and other baselines in terms of detection
accuracy.

    

### [[2011.14229] Deep Learning for Regularization Prediction in Diffeomorphic Image Registration](http://arxiv.org/abs/2011.14229)


  This paper presents a predictive model for estimating regularization
parameters of diffeomorphic image registration. We introduce a novel framework
that automatically determines the parameters controlling the smoothness of
diffeomorphic transformations. Our method significantly reduces the effort of
parameter tuning, which is time and labor-consuming. To achieve the goal, we
develop a predictive model based on deep convolutional neural networks (CNN)
that learns the mapping between pairwise images and the regularization
parameter of image registration. In contrast to previous methods that estimate
such parameters in a high-dimensional image space, our model is built in an
efficient bandlimited space with much lower dimensions. We demonstrate the
effectiveness of our model on both 2D synthetic data and 3D real brain images.
Experimental results show that our model not only predicts appropriate
regularization parameters for image registration, but also improving the
network training in terms of time and memory efficiency.

    

### [[2012.05895] Few-Shot Attribute Learning](http://arxiv.org/abs/2012.05895)


  Semantic concepts are frequently defined by combinations of underlying
attributes. As mappings from attributes to classes are often simple,
attribute-based representations facilitate novel concept learning with zero or
few examples. A significant limitation of existing attribute-based learning
paradigms, such as zero-shot learning, is that the attributes are assumed to be
known and fixed. In this work we study the rapid learning of attributes that
were not previously labeled. Compared to standard few-shot learning of semantic
classes, in which novel classes may be defined by attributes that were relevant
at training time, learning new attributes imposes a stiffer challenge. We found
that supervised learning with training attributes does not generalize well to
new test attributes, whereas self-supervised pre-training brings significant
improvement. We further experimented with random splits of the attribute space
and found that predictability of test attributes provides an informative
estimate of a model's generalization ability.

    

### [[2101.04948] Deep State Inference: Toward Behavioral Model Inference of Black-box Software Systems](http://arxiv.org/abs/2101.04948)


  Many software engineering tasks, such as testing, and anomaly detection can
benefit from the ability to infer a behavioral model of the software.Most
existing inference approaches assume access to code to collect execution
sequences. In this paper, we investigate a black-box scenario, where the system
under analysis cannot be instrumented, in this granular fashion.This scenario
is particularly prevalent with control systems' log analysis in the form of
continuous signals. In this situation, an execution trace amounts to a
multivariate time-series of input and output signals, where different states of
the system correspond to different `phases` in the time-series. The main
challenge is to detect when these phase changes take place. Unfortunately, most
existing solutions are either univariate, make assumptions on the data
distribution, or have limited learning power.Therefore, we propose a hybrid
deep neural network that accepts as input a multivariate time series and
applies a set of convolutional and recurrent layers to learn the non-linear
correlations between signals and the patterns over time.We show how this
approach can be used to accurately detect state changes, and how the inferred
models can be successfully applied to transfer-learning scenarios, to
accurately process traces from different products with similar execution
characteristics. Our experimental results on two UAV autopilot case studies
indicate that our approach is highly accurate (over 90% F1 score for state
classification) and significantly improves baselines (by up to 102% for change
point detection).Using transfer learning we also show that up to 90% of the
maximum achievable F1 scores in the open-source case study can be achieved by
reusing the trained models from the industrial case and only fine tuning them
using as low as 5 labeled samples, which reduces the manual labeling effort by
98%.

    

### [[2101.06553] Self-Supervised Representation Learning from Flow Equivariance](http://arxiv.org/abs/2101.06553)


  Self-supervised representation learning is able to learn semantically
meaningful features; however, much of its recent success relies on multiple
crops of an image with very few objects. Instead of learning view-invariant
representation from simple images, humans learn representations in a complex
world with changing scenes by observing object movement, deformation, pose
variation, and ego motion. Motivated by this ability, we present a new
self-supervised learning representation framework that can be directly deployed
on a video stream of complex scenes with many moving objects. Our framework
features a simple flow equivariance objective that encourages the network to
predict the features of another frame by applying a flow transformation to the
features of the current frame. Our representations, learned from
high-resolution raw video, can be readily used for downstream tasks on static
images. Readout experiments on challenging semantic segmentation, instance
segmentation, and object detection benchmarks show that we are able to
outperform representations obtained from previous state-of-the-art methods
including SimCLR and BYOL.

    

### [[2101.06560] Adversarial Attacks On Multi-Agent Communication](http://arxiv.org/abs/2101.06560)


  Growing at a fast pace, modern autonomous systems will soon be deployed at
scale, opening up the possibility for cooperative multi-agent systems. Sharing
information and distributing workloads allow autonomous agents to better
perform tasks and increase computation efficiency. However, shared information
can be modified to execute adversarial attacks on deep learning models that are
widely employed in modern systems. Thus, we aim to study the robustness of such
systems and focus on exploring adversarial attacks in a novel multi-agent
setting where communication is done through sharing learned intermediate
representations of neural networks. We observe that an indistinguishable
adversarial message can severely degrade performance, but becomes weaker as the
number of benign agents increases. Furthermore, we show that black-box transfer
attacks are more difficult in this setting when compared to directly perturbing
the inputs, as it is necessary to align the distribution of learned
representations with domain adaptation. Our work studies robustness at the
neural network level to contribute an additional layer of fault tolerance to
modern security protocols for more secure multi-agent systems.

    

### [[2102.06560] How Far Should We Look Back to Achieve Effective Real-Time Time-Series Anomaly Detection?](http://arxiv.org/abs/2102.06560)


  Anomaly detection is the process of identifying unexpected events or
ab-normalities in data, and it has been applied in many different areas such as
system monitoring, fraud detection, healthcare, intrusion detection, etc.
Providing real-time, lightweight, and proactive anomaly detection for time
series with neither human intervention nor domain knowledge could be highly
valuable since it reduces human effort and enables appropriate countermeasures
to be undertaken before a disastrous event occurs. To our knowledge, RePAD
(Real-time Proactive Anomaly Detection algorithm) is a generic approach with
all above-mentioned features. To achieve real-time and lightweight detection,
RePAD utilizes Long Short-Term Memory (LSTM) to detect whether or not each
upcoming data point is anomalous based on short-term historical data points.
However, it is unclear that how different amounts of historical data points
affect the performance of RePAD. Therefore, in this paper, we investigate the
impact of different amounts of historical data on RePAD by introducing a set of
performance metrics that cover novel detection accuracy measures, time
efficiency, readiness, and resource consumption, etc. Empirical experiments
based on real-world time series datasets are conducted to evaluate RePAD in
different scenarios, and the experimental results are presented and discussed.

    

### [[2102.12677] Do Not Let Privacy Overbill Utility: Gradient Embedding Perturbation for Private Learning](http://arxiv.org/abs/2102.12677)


  The privacy leakage of the model about the training data can be bounded in
the differential privacy mechanism. However, for meaningful privacy parameters,
a differentially private model degrades the utility drastically when the model
comprises a large number of trainable parameters. In this paper, we propose an
algorithm \emph{Gradient Embedding Perturbation (GEP)} towards training
differentially private deep models with decent accuracy. Specifically, in each
gradient descent step, GEP first projects individual private gradient into a
non-sensitive anchor subspace, producing a low-dimensional gradient embedding
and a small-norm residual gradient. Then, GEP perturbs the low-dimensional
embedding and the residual gradient separately according to the privacy budget.
Such a decomposition permits a small perturbation variance, which greatly helps
to break the dimensional barrier of private learning. With GEP, we achieve
decent accuracy with reasonable computational cost and modest privacy guarantee
for deep models. Especially, with privacy bound $\epsilon=8$, we achieve
$74.9\%$ test accuracy on CIFAR10 and $95.1\%$ test accuracy on SVHN,
significantly improving over existing results.

    

### [[2103.05524] On the interplay between data structure and loss function in classification problems](http://arxiv.org/abs/2103.05524)


  One of the central puzzles in modern machine learning is the ability of
heavily overparametrized models to generalize well. Although the
low-dimensional structure of typical datasets is key to this behavior, most
theoretical studies of overparametrization focus on isotropic inputs. In this
work, we instead consider an analytically tractable model of structured data,
where the input covariance is built from independent blocks allowing us to tune
the saliency of low-dimensional structures and their alignment with respect to
the target function. Using methods from statistical physics, we derive a
precise asymptotic expression for the train and test error achieved by random
feature models trained to classify such data, which is valid for any convex
loss function. We study in detail how the data structure affects the double
descent curve, and show that in the over-parametrized regime, its impact is
greater for logistic loss than for mean-squared loss: the easier the task, the
wider the gap in performance at the advantage of the logistic loss. Our
insights are confirmed by numerical experiments on MNIST and CIFAR10.

    

### [[2103.10427] The Low-Rank Simplicity Bias in Deep Networks](http://arxiv.org/abs/2103.10427)


  Modern deep neural networks are highly over-parameterized compared to the
data on which they are trained, yet they often generalize remarkably well. A
flurry of recent work has asked: why do deep networks not overfit to their
training data? In this work, we make a series of empirical observations that
investigate the hypothesis that deeper networks are inductively biased to find
solutions with lower rank embeddings. We conjecture that this bias exists
because the volume of functions that maps to low-rank embedding increases with
depth. We show empirically that our claim holds true on finite width linear and
non-linear models and show that these are the solutions that generalize well.
We then show that the low-rank simplicity bias exists even after training,
using a wide variety of commonly used optimizers. We found this phenomenon to
be resilient to initialization, hyper-parameters, and learning methods. We
further demonstrate how linear over-parameterization of deep non-linear models
can be used to induce low-rank bias, improving generalization performance
without changing the effective model capacity. Practically, we demonstrate that
simply linearly over-parameterizing standard models at training time can
improve performance on image classification tasks, including ImageNet.

    

### [[2103.17182] Positive-Negative Momentum: Manipulating Stochastic Gradient Noise to Improve Generalization](http://arxiv.org/abs/2103.17182)


  It is well-known that stochastic gradient noise (SGN) acts as implicit
regularization for deep learning and is essentially important for both
optimization and generalization of deep networks. Some works attempted to
artificially simulate SGN by injecting random noise to improve deep learning.
However, it turned out that the injected simple random noise cannot work as
well as SGN, which is anisotropic and parameter-dependent. For simulating SGN
at low computational costs and without changing the learning rate or batch
size, we propose the Positive-Negative Momentum (PNM) approach that is a
powerful alternative to conventional Momentum in classic optimizers. The
introduced PNM method maintains two approximate independent momentum terms.
Then, we can control the magnitude of SGN explicitly by adjusting the momentum
difference. We theoretically prove the convergence guarantee and the
generalization advantage of PNM over Stochastic Gradient Descent (SGD). By
incorporating PNM into the two conventional optimizers, SGD with Momentum and
Adam, our extensive experiments empirically verified the significant advantage
of the PNM-based variants over the corresponding conventional Momentum-based
optimizers.

    

### [[2104.00987] Bayesian Structural Learning for an Improved Diagnosis of Cyber-Physical Systems](http://arxiv.org/abs/2104.00987)


  The diagnosis of cyber-physical systems aims to detect faulty behaviour, its
root cause and a mitigation or even prevention policy. Therefore, diagnosis
relies on a representation of the system's functional and faulty behaviour
combined with observations of the system taken at runtime. The main challenges
are the time-intensive building of a model, possible state-explosion while
searching for the root cause and interpretability of the results. In this paper
we propose a scalable algorithm tackling these challenges. We use a Bayesian
network to learn a structured model automatically and optimise the model by a
genetic algorithm. Our approach differs from existing work in two aspects:
instead of selecting features prior to the analysis we learn a global
representation using all available information which is then transformed to a
smaller, label-specific one and we focus on interpretability to facilitate
repairs. The evaluation shows that our approach is able to learn a model with
equal performance to state-of-the-art algorithms while giving better
interpretability and having a reduced size.

    

### [[2104.04282] Direct Differentiable Augmentation Search](http://arxiv.org/abs/2104.04282)


  Data augmentation has been an indispensable tool to improve the performance
of deep neural networks, however the augmentation can hardly transfer among
different tasks and datasets. Consequently, a recent trend is to adopt AutoML
technique to learn proper augmentation policy without extensive hand-crafted
tuning. In this paper, we propose an efficient differentiable search algorithm
called Direct Differentiable Augmentation Search (DDAS). It exploits
meta-learning with one-step gradient update and continuous relaxation to the
expected training loss for efficient search. Our DDAS can achieve efficient
augmentation search without relying on approximations such as Gumbel Softmax or
second order gradient approximation. To further reduce the adverse effect of
improper augmentations, we organize the search space into a two level
hierarchy, in which we first decide whether to apply augmentation, and then
determine the specific augmentation policy. On standard image classification
benchmarks, our DDAS achieves state-of-the-art performance and efficiency
tradeoff while reducing the search cost dramatically, e.g. 0.15 GPU hours for
CIFAR-10. In addition, we also use DDAS to search augmentation for object
detection task and achieve comparable performance with AutoAugment, while being
1000x faster.

    

### [[2104.06521] TAAC: Temporally Abstract Actor-Critic for Continuous Control](http://arxiv.org/abs/2104.06521)


  We present temporally abstract actor-critic (TAAC), a simple but effective
off-policy RL algorithm that incorporates closed-loop temporal abstraction into
the actor-critic framework. TAAC adds a second-stage binary policy to choose
between the previous action and a new action output by an actor. Crucially, its
"act-or-repeat" decision hinges on the actually sampled action instead of the
expected behavior of the actor. This post-acting switching scheme let the
overall policy make more informed decisions. TAAC has two important features:
a) persistent exploration, and b) a new compare-through Q operator for
multi-step TD backup, specially tailored to the action repetition scenario. We
demonstrate TAAC's advantages over several strong baselines across 14
continuous control tasks. Our surprising finding reveals that while achieving
top performance, TAAC is able to "mine" a significant number of repeated
actions with the trained policy even on continuous tasks whose problem
structures on the surface seem to repel action repetition. This suggests that
aside from encouraging persistent exploration, action repetition can find its
place in a good policy behavior. Code is available at
this https URL.

    

### [[2104.12763] MDETR -- Modulated Detection for End-to-End Multi-Modal Understanding](http://arxiv.org/abs/2104.12763)


  Multi-modal reasoning systems rely on a pre-trained object detector to
extract regions of interest from the image. However, this crucial module is
typically used as a black box, trained independently of the downstream task and
on a fixed vocabulary of objects and attributes. This makes it challenging for
such systems to capture the long tail of visual concepts expressed in free form
text. In this paper we propose MDETR, an end-to-end modulated detector that
detects objects in an image conditioned on a raw text query, like a caption or
a question. We use a transformer-based architecture to reason jointly over text
and image by fusing the two modalities at an early stage of the model. We
pre-train the network on 1.3M text-image pairs, mined from pre-existing
multi-modal datasets having explicit alignment between phrases in text and
objects in the image. We then fine-tune on several downstream tasks such as
phrase grounding, referring expression comprehension and segmentation,
achieving state-of-the-art results on popular benchmarks. We also investigate
the utility of our model as an object detector on a given label set when
fine-tuned in a few-shot setting. We show that our pre-training approach
provides a way to handle the long tail of object categories which have very few
labelled instances. Our approach can be easily extended for visual question
answering, achieving competitive performance on GQA and CLEVR. The code and
models are available at this https URL.

    

### [[2105.08318] Zero-Shot Recommender Systems](http://arxiv.org/abs/2105.08318)


  Performance of recommender systems (RS) relies heavily on the amount of
training data available. This poses a chicken-and-egg problem for early-stage
products, whose amount of data, in turn, relies on the performance of their RS.
On the other hand, zero-shot learning promises some degree of generalization
from an old dataset to an entirely new dataset. In this paper, we explore the
possibility of zero-shot learning in RS. We develop an algorithm, dubbed
ZEro-Shot Recommenders (ZESRec), that is trained on an old dataset and
generalize to a new one where there are neither overlapping users nor
overlapping items, a setting that contrasts typical cross-domain RS that has
either overlapping users or items. Different from categorical item indices,
i.e., item ID, in previous methods, ZESRec uses items' natural-language
descriptions (or description embeddings) as their continuous indices, and
therefore naturally generalize to any unseen items. In terms of users, ZESRec
builds upon recent advances on sequential RS to represent users using their
interactions with items, thereby generalizing to unseen users as well. We study
three pairs of real-world RS datasets and demonstrate that ZESRec can
successfully enable recommendations in such a zero-shot setting, opening up new
opportunities for resolving the chicken-and-egg problem for data-scarce
startups or early-stage products.

    

### [[2106.01548] When Vision Transformers Outperform ResNets without Pre-training or Strong Data Augmentations](http://arxiv.org/abs/2106.01548)


  Vision Transformers (ViTs) and MLPs signal further efforts on replacing
hand-wired features or inductive biases with general-purpose neural
architectures. Existing works empower the models by massive data, such as
large-scale pre-training and/or repeated strong data augmentations, and still
report optimization-related problems (e.g., sensitivity to initialization and
learning rates). Hence, this paper investigates ViTs and MLP-Mixers from the
lens of loss geometry, intending to improve the models' data efficiency at
training and generalization at inference. Visualization and Hessian reveal
extremely sharp local minima of converged models. By promoting smoothness with
a recently proposed sharpness-aware optimizer, we substantially improve the
accuracy and robustness of ViTs and MLP-Mixers on various tasks spanning
supervised, adversarial, contrastive, and transfer learning (e.g., +5.3\% and
+11.0\% top-1 accuracy on ImageNet for ViT-B/16 and Mixer-B/16, respectively,
with the simple Inception-style preprocessing). We show that the improved
smoothness attributes to sparser active neurons in the first few layers. The
resultant ViTs outperform ResNets of similar size and throughput when trained
from scratch on ImageNet without large-scale pre-training or strong data
augmentations. They also possess more perceptive attention maps. Our model
checkpoints are released at
\url{this https URL}.

    

### [[2106.02888] Bandwidth-based Step-Sizes for Non-Convex Stochastic Optimization](http://arxiv.org/abs/2106.02888)


  Many popular learning-rate schedules for deep neural networks combine a
decaying trend with local perturbations that attempt to escape saddle points
and bad local minima. We derive convergence guarantees for bandwidth-based
step-sizes, a general class of learning rates that are allowed to vary in a
banded region. This framework includes many popular cyclic and non-monotonic
step-sizes for which no theoretical guarantees were previously known. We
provide worst-case guarantees for SGD on smooth non-convex problems under
several bandwidth-based step sizes, including stagewise $1/\sqrt{t}$ and the
popular step-decay (constant and then drop by a constant), which is also shown
to be optimal. Moreover, we show that its momentum variant converges as fast as
SGD with the bandwidth-based step-decay step-size. Finally, we propose novel
step-size schemes in the bandwidth-based family and verify their efficiency on
several deep neural network training tasks.

    

### [[2106.07306] Constraining Linear-chain CRFs to Regular Languages](http://arxiv.org/abs/2106.07306)


  A major challenge in structured prediction is to represent the
interdependencies within output structures. When outputs are structured as
sequences, linear-chain conditional random fields (CRFs) are a widely used
model class which can learn \textit{local} dependencies in the output. However,
the CRF's Markov assumption makes it impossible for CRFs to represent
distributions with \textit{nonlocal} dependencies, and standard CRFs are unable
to respect nonlocal constraints of the data (such as global arity constraints
on output labels). We present a generalization of CRFs that can enforce a broad
class of constraints, including nonlocal ones, by specifying the space of
possible output structures as a regular language $\mathcal{L}$. The resulting
regular-constrained CRF (RegCCRF) has the same formal properties as a standard
CRF, but assigns zero probability to all label sequences not in $\mathcal{L}$.
Notably, RegCCRFs can incorporate their constraints during training, while
related models only enforce constraints during decoding. We prove that
constrained training is never worse than constrained decoding, and show
empirically that it can be substantially better in practice. Additionally, we
demonstrate a practical benefit on downstream tasks by incorporating a RegCCRF
into a deep neural model for semantic role labeling, exceeding state-of-the-art
results on a standard dataset.

    

### [[2106.07630] Hierarchically Regularized Deep Forecasting](http://arxiv.org/abs/2106.07630)


  Hierarchical forecasting is a key problem in many practical multivariate
forecasting applications - the goal is to simultaneously predict a large number
of correlated time series that are arranged in a pre-specified aggregation
hierarchy. The main challenge is to exploit the hierarchical correlations to
simultaneously obtain good prediction accuracy for time series at different
levels of the hierarchy. In this paper, we propose a new approach for
hierarchical forecasting which consists of two components. First, decomposing
the time series along a global set of basis time series and modeling
hierarchical constraints using the coefficients of the basis decomposition. And
second, using a linear autoregressive model with coefficients that vary with
time. Unlike past methods, our approach is scalable (inference for a specific
time series only needs access to its own history) while also modeling the
hierarchical structure via (approximate) coherence constraints among the time
series forecasts. We experiment on several public datasets and demonstrate
significantly improved overall performance on forecasts at different levels of
the hierarchy, compared to existing state-of-the-art hierarchical models.

    

### [[2106.09779] Private Federated Learning Without a Trusted Server: Optimal Algorithms for Convex Losses](http://arxiv.org/abs/2106.09779)


  This paper studies the problem of federated learning (FL) in the absence of a
trustworthy server/clients. In this setting, each client needs to ensure the
privacy of its own data without relying on the server or other clients. We
study local differential privacy (LDP) and provide tight upper and lower bounds
that establish the minimax optimal rates (up to logarithms) for LDP
convex/strongly convex federated stochastic optimization. Our rates match the
optimal statistical rates in certain practical parameter regimes ("privacy for
free"). Second, we develop a novel time-varying noisy SGD algorithm, leading to
the first non-trivial LDP risk bounds for FL with non-i.i.d. clients. Third, we
consider the special case where each client's loss function is empirical and
develop an accelerated LDP FL algorithm to improve communication complexity
compared to existing works. We also provide matching lower bounds, establishing
the optimality of our algorithm for convex/strongly convex settings. Fourth,
with a secure shuffler to anonymize client reports (but without a trusted
server), our algorithm attains the optimal central DP rates for stochastic
convex/strongly convex optimization, thereby achieving optimality in the local
and central models simultaneously. Our upper bounds quantify the role of
network communication reliability in performance.

    

### [[2106.10185] NoiseGrad: enhancing explanations by introducing stochasticity to model weights](http://arxiv.org/abs/2106.10185)


  Many efforts have been made for revealing the decision-making process of
black-box learning machines such as deep neural networks, resulting in useful
local and global explanation methods. For local explanation, stochasticity is
known to help: a simple method, called SmoothGrad, has improved the visual
quality of gradient-based attribution by adding noise in the input space and
taking the average over the noise. In this paper, we extend this idea and
propose NoiseGrad that enhances both local and global explanation methods.
Specifically, NoiseGrad introduces stochasticity in the weight parameter space,
such that the decision boundary is perturbed. NoiseGrad is expected to enhance
the local explanation, similarly to SmoothGrad, due to the dual relationship
between the input perturbation and the decision boundary perturbation.
Furthermore, NoiseGrad can be used to enhance global explanations. We evaluate
NoiseGrad and its fusion with SmoothGrad -- FusionGrad -- qualitatively and
quantitatively with several evaluation criteria, and show that our novel
approach significantly outperforms the baseline methods. Both NoiseGrad and
FusionGrad are method-agnostic and as handy as SmoothGrad using simple
heuristics for the choice of hyperparameter setting without the need of
fine-tuning.

    

### [[2106.11420] Policy Smoothing for Provably Robust Reinforcement Learning](http://arxiv.org/abs/2106.11420)


  The study of provable adversarial robustness for deep neural networks (DNNs)
has mainly focused on static supervised learning tasks such as image
classification. However, DNNs have been used extensively in real-world adaptive
tasks such as reinforcement learning (RL), making such systems vulnerable to
adversarial attacks as well. Prior works in provable robustness in RL seek to
certify the behaviour of the victim policy at every time-step against a
non-adaptive adversary using methods developed for the static setting. But in
the real world, an RL adversary can infer the defense strategy used by the
victim agent by observing the states, actions, etc. from previous time-steps
and adapt itself to produce stronger attacks in future steps. We present an
efficient procedure, designed specifically to defend against an adaptive RL
adversary, that can directly certify the total reward without requiring the
policy to be robust at each time-step. Our main theoretical contribution is to
prove an adaptive version of the Neyman-Pearson Lemma -- a key lemma for
smoothing-based certificates -- where the adversarial perturbation at a
particular time can be a stochastic function of current and previous
observations and states as well as previous actions. Building on this result,
we propose policy smoothing where the agent adds a Gaussian noise to its
observation at each time-step before passing it through the policy function.
Our robustness certificates guarantee that the final total reward obtained by
policy smoothing remains above a certain threshold, even though the actions at
intermediate time-steps may change under the attack. Our experiments on various
environments like Cartpole, Pong, Freeway and Mountain Car show that our method
can yield meaningful robustness guarantees in practice.

    

### [[2106.12723] Meaningfully Explaining Model Mistakes Using Conceptual Counterfactuals](http://arxiv.org/abs/2106.12723)


  Understanding and explaining the mistakes made by trained models is critical
to many machine learning objectives, such as improving robustness, addressing
concept drift, and mitigating biases. However, this is often an ad hoc process
that involves manually looking at the model's mistakes on many test samples and
guessing at the underlying reasons for those incorrect predictions. In this
paper, we propose a systematic approach, conceptual counterfactual
explanations(CCE), that explains why a classifier makes a mistake on a
particular test sample(s) in terms of human-understandable concepts (e.g. this
zebra is misclassified as a dog because of faint stripes). We base CCE on two
prior ideas: counterfactual explanations and concept activation vectors, and
validate our approach on well-known pretrained models, showing that it explains
the models' mistakes meaningfully. In addition, for new models trained on data
with spurious correlations, CCE accurately identifies the spurious correlation
as the cause of model mistakes from a single misclassified test sample. On two
challenging medical applications, CCE generated useful insights, confirmed by
clinicians, into biases and mistakes the model makes in real-world settings.
The code for CCE is publicly available and can easily be applied to explain
mistakes in new models.

    

### [[2106.12936] Fundamental limits for learning hidden Markov model parameters](http://arxiv.org/abs/2106.12936)


  We study the frontier between learnable and unlearnable hidden Markov models
(HMMs). HMMs are flexible tools for clustering dependent data coming from
unknown populations. The model parameters are known to be fully identifiable
(up to label-switching) without any modeling assumption on the distributions of
the populations as soon as the clusters are distinct and the hidden chain is
ergodic with a full rank transition matrix. In the limit as any one of these
conditions fails, it becomes impossible in general to identify parameters. For
a chain with two hidden states we prove nonasymptotic minimax upper and lower
bounds, matching up to constants, which exhibit thresholds at which the
parameters become learnable. We also provide an upper bound on the relative
entropy rate for parameters in a neighbourhood of the unlearnable region which
may have interest in itself.

    

### [[2107.09028] Structured Stochastic Gradient MCMC](http://arxiv.org/abs/2107.09028)


  Stochastic gradient Markov chain Monte Carlo (SGMCMC) is considered the gold
standard for Bayesian inference in large-scale models, such as Bayesian neural
networks. Since practitioners face speed versus accuracy tradeoffs in these
models, variational inference (VI) is often the preferable option.
Unfortunately, VI makes strong assumptions on both the factorization and
functional form of the posterior. In this work, we propose a new non-parametric
variational approximation that makes no assumptions about the approximate
posterior's functional form and allows practitioners to specify the exact
dependencies the algorithm should respect or break. The approach relies on a
new Langevin-type algorithm that operates on a modified energy function, where
parts of the latent variables are averaged over samples from earlier iterations
of the Markov chain. This way, statistical dependencies can be broken in a
controlled way, allowing the chain to mix faster. This scheme can be further
modified in a "dropout" manner, leading to even more scalability. By
implementing the scheme on a ResNet-20 architecture, we obtain better
predictive likelihoods and larger effective sample sizes than full SGMCMC.

    

### [[2109.05675] Online Unsupervised Learning of Visual Representations and Categories](http://arxiv.org/abs/2109.05675)


  Real world learning scenarios involve a nonstationary distribution of classes
with sequential dependencies among the samples, in contrast to the standard
machine learning formulation of drawing samples independently from a fixed,
typically uniform distribution. Furthermore, real world interactions demand
learning on-the-fly from few or no class labels. In this work, we propose an
unsupervised model that simultaneously performs online visual representation
learning and few-shot learning of new categories without relying on any class
labels. Our model is a prototype-based memory network with a control component
that determines when to form a new class prototype. We formulate it as an
online Gaussian mixture model, where components are created online with only a
single new example, and assignments do not have to be balanced, which permits
an approximation to natural imbalanced distributions from uncurated raw data.
Learning includes a contrastive loss that encourages different views of the
same image to be assigned to the same prototype. The result is a mechanism that
forms categorical representations of objects in nonstationary environments.
Experiments show that our method can learn from an online stream of visual
input data and is significantly better at category recognition compared to
state-of-the-art self-supervised learning methods.

    

### [[2109.09444] When Do Extended Physics-Informed Neural Networks (XPINNs) Improve Generalization?](http://arxiv.org/abs/2109.09444)


  Physics-informed neural networks (PINNs) have become a popular choice for
solving high-dimensional partial differential equations (PDEs) due to their
excellent approximation power and generalization ability. Recently, Extended
PINNs (XPINNs) based on domain decomposition methods have attracted
considerable attention due to their effectiveness in modeling multiscale and
multiphysics problems and their parallelization. However, theoretical
understanding on their convergence and generalization properties remains
unexplored. In this study, we take an initial step towards understanding how
and when XPINNs outperform PINNs. Specifically, for general multi-layer PINNs
and XPINNs, we first provide a prior generalization bound via the complexity of
the target functions in the PDE problem, and a posterior generalization bound
via the posterior matrix norms of the networks after optimization. Moreover,
based on our bounds, we analyze the conditions under which XPINNs improve
generalization. Concretely, our theory shows that the key building block of
XPINN, namely the domain decomposition, introduces a tradeoff for
generalization. On the one hand, XPINNs decompose the complex PDE solution into
several simple parts, which decreases the complexity needed to learn each part
and boosts generalization. On the other hand, decomposition leads to less
training data being available in each subdomain, and hence such model is
typically prone to overfitting and may become less generalizable. Empirically,
we choose five PDEs to show when XPINNs perform better than, similar to, or
worse than PINNs, hence demonstrating and justifying our new theory.

    

### [[2109.13090] Oscillatory Fourier Neural Network: A Compact and Efficient Architecture for Sequential Processing](http://arxiv.org/abs/2109.13090)


  Tremendous progress has been made in sequential processing with the recent
advances in recurrent neural networks. However, recurrent architectures face
the challenge of exploding/vanishing gradients during training, and require
significant computational resources to execute back-propagation through time.
Moreover, large models are typically needed for executing complex sequential
tasks. To address these challenges, we propose a novel neuron model that has
cosine activation with a time varying component for sequential processing. The
proposed neuron provides an efficient building block for projecting sequential
inputs into spectral domain, which helps to retain long-term dependencies with
minimal extra model parameters and computation. A new type of recurrent network
architecture, named Oscillatory Fourier Neural Network, based on the proposed
neuron is presented and applied to various types of sequential tasks. We
demonstrate that recurrent neural network with the proposed neuron model is
mathematically equivalent to a simplified form of discrete Fourier transform
applied onto periodical activation. In particular, the computationally
intensive back-propagation through time in training is eliminated, leading to
faster training while achieving the state of the art inference accuracy in a
diverse group of sequential tasks. For instance, applying the proposed model to
sentiment analysis on IMDB review dataset reaches 89.4% test accuracy within 5
epochs, accompanied by over 35x reduction in the model size compared to LSTM.
The proposed novel RNN architecture is well poised for intelligent sequential
processing in resource constrained hardware.

    

### [[2110.05057] Can Stochastic Gradient Langevin Dynamics Provide Differential Privacy for Deep Learning?](http://arxiv.org/abs/2110.05057)


  Bayesian learning via Stochastic Gradient Langevin Dynamics (SGLD) has been
suggested for differentially private learning. While previous research provides
differential privacy bounds for SGLD when close to convergence or at the
initial steps of the algorithm, the question of what differential privacy
guarantees can be made in between remains unanswered. This interim region is
essential, especially for Bayesian neural networks, as it is hard to guarantee
convergence to the posterior. This paper will show that using SGLD might result
in unbounded privacy loss for this interim region, even when sampling from the
posterior is as differentially private as desired.

    

### [[2110.05076] A Closer Look at Prototype Classifier for Few-shot Image Classification](http://arxiv.org/abs/2110.05076)


  The prototypical network is a prototype classifier based on meta-learning and
is widely used for few-shot learning because it classifies unseen examples by
constructing class-specific prototypes without adjusting hyper-parameters
during meta-testing. Interestingly, recent research has attracted a lot of
attention, showing that a linear classifier with fine-tuning, which does not
use a meta-learning algorithm, performs comparably with the prototypical
network. However, fine-tuning requires additional hyper-parameters when
adapting a model to a new environment. In addition, although the purpose of
few-shot learning is to enable the model to quickly adapt to a new environment,
fine-tuning needs to be applied every time a new class appears, making fast
adaptation difficult. In this paper, we analyze how a prototype classifier
works equally well without fine-tuning and meta-learning. We experimentally
found that directly using the feature vector extracted using standard
pre-trained models to construct a prototype classifier in meta-testing does not
perform as well as the prototypical network and linear classifiers with
fine-tuning and feature vectors of pre-trained models. Thus, we derive a novel
generalization bound for the prototypical network and show that focusing on the
variance of the norm of a feature vector can improve performance. We
experimentally investigated several normalization methods for minimizing the
variance of the norm and found that the same performance can be obtained by
using the L2 normalization and embedding space transformation without
fine-tuning or meta-learning.

    

### [[2110.05177] Learning Division with Neural Arithmetic Logic Modules](http://arxiv.org/abs/2110.05177)


  To achieve systematic generalisation, it first makes sense to master simple
tasks such as arithmetic. Of the four fundamental arithmetic operations
(+,-,$\times$,$÷$), division is considered the most difficult for both
humans and computers. In this paper we show that robustly learning division in
a systematic manner remains a challenge even at the simplest level of dividing
two numbers. We propose two novel approaches for division which we call the
Neural Reciprocal Unit (NRU) and the Neural Multiplicative Reciprocal Unit
(NMRU), and present improvements for an existing division module, the Real
Neural Power Unit (Real NPU). Experiments in learning division with input
redundancy on 225 different training sets, find that our proposed modifications
to the Real NPU obtains an average success of 85.3$\%$ improving over the
original by 15.1$\%$. In light of the suggestion above, our NMRU approach can
further improve the success to 91.6$\%$.

    

### [[2110.05188] A Theory of Tournament Representations](http://arxiv.org/abs/2110.05188)


  Real world tournaments are almost always intransitive. Recent works have
noted that parametric models which assume $d$ dimensional node representations
can effectively model intransitive tournaments. However, nothing is known about
the structure of the class of tournaments that arise out of any fixed $d$
dimensional representations. In this work, we develop a novel theory for
understanding parametric tournament representations. Our first contribution is
to structurally characterize the class of tournaments that arise out of $d$
dimensional representations. We do this by showing that these tournament
classes have forbidden configurations which must necessarily be union of flip
classes, a novel way to partition the set of all tournaments. We further
characterise rank $2$ tournaments completely by showing that the associated
forbidden flip class contains just $2$ tournaments. Specifically, we show that
the rank $2$ tournaments are equivalent to locally-transitive tournaments. This
insight allows us to show that the minimum feedback arc set problem on this
tournament class can be solved using the standard Quicksort procedure. For a
general rank $d$ tournament class, we show that the flip class associated with
a coned-doubly regular tournament of size $\mathcal{O}(\sqrt{d})$ must be a
forbidden configuration. To answer a dual question, using a celebrated result
of \cite{forster}, we show a lower bound of $\mathcal{O}(\sqrt{n})$ on the
minimum dimension needed to represent all tournaments on $n$ nodes. For any
given tournament, we show a novel upper bound on the smallest representation
dimension that depends on the least size of the number of unique nodes in any
feedback arc set of the flip class associated with a tournament. We show how
our results also shed light on upper bound of sign-rank of matrices.

    

### [[2110.05291] Graph Neural Network Guided Local Search for the Traveling Salesperson Problem](http://arxiv.org/abs/2110.05291)


  Solutions to the Traveling Salesperson Problem (TSP) have practical
applications to processes in transportation, logistics, and automation, yet
must be computed with minimal delay to satisfy the real-time nature of the
underlying tasks. However, solving large TSP instances quickly without
sacrificing solution quality remains challenging for current approximate
algorithms. To close this gap, we present a hybrid data-driven approach for
solving the TSP based on Graph Neural Networks (GNNs) and Guided Local Search
(GLS). Our model predicts the regret of including each edge of the problem
graph in the solution; GLS uses these predictions in conjunction with the
original problem graph to find solutions. Our experiments demonstrate that this
approach converges to optimal solutions at a faster rate than state-of-the-art
learning-based approaches and non-learning GLS algorithms for the TSP, notably
finding optimal solutions to 96% of the 50-node problem set, 7% more than the
next best benchmark, and to 20% of the 100-node problem set, 4.5x more than the
next best benchmark. When generalizing from 20-node problems to the 100-node
problem set, our approach finds solutions with an average optimality gap of
2.5%, a 10x improvement over the next best learning-based benchmark.

    

### [[2109.13391] Curvature-Aware Derivative-Free Optimization](http://arxiv.org/abs/2109.13391)


  We propose a new line-search method, coined Curvature-Aware Random Search
(CARS), for derivative-free optimization. CARS exploits approximate curvature
information to estimate the optimal step-size given a search direction. We
prove that for strongly convex objective functions, CARS converges linearly if
the search direction is drawn from a distribution satisfying very mild
conditions. We also explore a variant, CARS-NQ, which uses Numerical Quadrature
instead of a Monte Carlo method when approximating curvature along the search
direction. We show CARS-NQ is effective on highly non-convex problems of the
form $f = f_{\mathrm{cvx}} + f_{\mathrm{osc}}$ where $f_{\mathrm{cvx}}$ is
strongly convex and $f_{\mathrm{osc}}$ is rapidly oscillating. Experimental
results show that CARS and CARS-NQ match or exceed the state-of-the-arts on
benchmark problem sets.

    

### [[2110.05855] MoRS: An Approximate Fault Modelling Framework for Reduced-Voltage SRAMs](http://arxiv.org/abs/2110.05855)


  On-chip memory (usually based on Static RAMs-SRAMs) are crucial components
for various computing devices including heterogeneous devices, e.g., GPUs,
FPGAs, ASICs to achieve high performance. Modern workloads such as Deep Neural
Networks (DNNs) running on these heterogeneous fabrics are highly dependent on
the on-chip memory architecture for efficient acceleration. Hence, improving
the energy-efficiency of such memories directly leads to an efficient system.
One of the common methods to save energy is undervolting i.e., supply voltage
underscaling below the nominal level. Such systems can be safely undervolted
without incurring faults down to a certain voltage limit. This safe range is
also called voltage guardband. However, reducing voltage below the guardband
level without decreasing frequency causes timing-based faults.
In this paper, we propose MoRS, a framework that generates the first
approximate undervolting fault model using real faults extracted from
experimental undervolting studies on SRAMs to build the model. We inject the
faults generated by MoRS into the on-chip memory of the DNN accelerator to
evaluate the resilience of the system under the test. MoRS has the advantage of
simplicity without any need for high-time overhead experiments while being
accurate enough in comparison to a fully randomly-generated fault injection
approach. We evaluate our experiment in popular DNN workloads by mapping
weights to SRAMs and measure the accuracy difference between the output of the
MoRS and the real data. Our results show that the maximum difference between
real fault data and the output fault model of MoRS is 6.21%, whereas the
maximum difference between real data and random fault injection model is 23.2%.
In terms of average proximity to the real data, the output of MoRS outperforms
the random fault injection approach by 3.21x.

    

### [[2110.06155] Memory-Efficient CNN Accelerator Based on Interlayer Feature Map Compression](http://arxiv.org/abs/2110.06155)


  Existing deep convolutional neural networks (CNNs) generate massive
interlayer feature data during network inference. To maintain real-time
processing in embedded systems, large on-chip memory is required to buffer the
interlayer feature maps. In this paper, we propose an efficient hardware
accelerator with an interlayer feature compression technique to significantly
reduce the required on-chip memory size and off-chip memory access bandwidth.
The accelerator compresses interlayer feature maps through transforming the
stored data into frequency domain using hardware-implemented 8x8 discrete
cosine transform (DCT). The high-frequency components are removed after the DCT
through quantization. Sparse matrix compression is utilized to further compress
the interlayer feature maps. The on-chip memory allocation scheme is designed
to support dynamic configuration of the feature map buffer size and scratch pad
size according to different network-layer requirements. The hardware
accelerator combines compression, decompression, and CNN acceleration into one
computing stream, achieving minimal compressing and processing delay. A
prototype accelerator is implemented on an FPGA platform and also synthesized
in TSMC 28-nm COMS technology. It achieves 403GOPS peak throughput and
1.4x~3.3x interlayer feature map reduction by adding light hardware area
overhead, making it a promising hardware accelerator for intelligent IoT
devices.

    

### [[2105.11754] ScalaBFS: A Scalable BFS Accelerator on HBM-Enhanced FPGAs](http://arxiv.org/abs/2105.11754)


  High Bandwidth Memory (HBM) provides massive aggregated memory bandwidth by
exposing multiple memory channels to the processing units. To achieve high
performance, an accelerator built on top of an FPGA configured with HBM (i.e.,
FPGA-HBM platform) needs to scale its performance according to the available
memory channels. In this paper, we propose an accelerator for BFS
(Breadth-First Search) algorithm, named as ScalaBFS, that builds multiple
processing elements to sufficiently exploit the high bandwidth of HBM to
improve efficiency. We implement the prototype system of ScalaBFS and conduct
BFS in both real-world and synthetic scale-free graphs on Xilinx Alveo U280
FPGA card real hardware. The experimental results show that ScalaBFS scales its
performance almost linearly according to the available memory pseudo channels
(PCs) from the HBM2 subsystem of U280. By fully using the 32 PCs and building
64 processing elements (PEs) on U280, ScalaBFS achieves a performance up to
19.7 GTEPS (Giga Traversed Edges Per Second). When conducting BFS in sparse
real-world graphs, ScalaBFS achieves equivalent GTEPS to Gunrock running on the
state-of-art Nvidia V100 GPU that features 64-PC HBM2 (twice memory bandwidth
than U280).

    

### [[2110.05335] From FPGAs to Obfuscated eASICs: Design and Security Trade-offs](http://arxiv.org/abs/2110.05335)


  Threats associated with the untrusted fabrication of integrated circuits
(ICs) are numerous: piracy, overproduction, reverse engineering, hardware
trojans, etc. The use of reconfigurable elements (i.e., look-up tables as in
FPGAs) is a known obfuscation technique. In the extreme case, when the circuit
is entirely implemented as an FPGA, no information is revealed to the adversary
but at a high cost in area, power, and performance. In the opposite extreme,
when the same circuit is implemented as an ASIC, best-in-class performance is
obtained but security is compromised. This paper investigates an intermediate
solution between these two. Our results are supported by a custom CAD tool that
explores this FPGA-ASIC design space and enables a standard-cell based physical
synthesis flow that is flexible and compatible with current design practices.
Layouts are presented for obfuscated circuits in a 65nm commercial technology,
demonstrating the attained obfuscation both graphically and quantitatively.
Furthermore, our security analysis revealed that for truly hiding the circuit's
intent (not only portions of its structure), the obfuscated design also has to
chiefly resemble an FPGA: only some small amount of logic can be made static
for an adversary to remain unaware of what the circuit does.

    

### [[2110.05540] Parallel Batched Interpolation Search Tree](http://arxiv.org/abs/2110.05540)


  Ordered set (and map) is one of the most used data type. In addition to
standard set operations, like insert, delete and contains, it can provide
set-set operations such as union, intersection, and difference. Each of these
set-set operations is equivalent to batched operations: the data structure
should process a set of operations insert, delete, and contains. It is obvious
that we want these "large" operations to be parallelized. Typically, these sets
are implemented with the trees of logarithmic height, such as 2-3 tree, Treap,
AVL tree, Red-Black tree, etc. Until now, little attention was devoted to data
structures that work better but under several restrictions on the data. In this
work, we parallelize Interpolation Search Tree which serves each request from a
smooth distribution in doubly-logarithmic time. Our data structure of size $n$
performs a batch of $m$ operations in $O(m \log\log n)$ work and poly-log span.

    

### [[2110.05543] Fallout: Distributed Systems Testing as a Service](http://arxiv.org/abs/2110.05543)


  All modern distributed systems list performance and scalability as their core
strengths. Given that optimal performance requires carefully selecting
configuration options, and typical cluster sizes can range anywhere from 2 to
300 nodes, it is rare for any two clusters to be exactly the same. Validating
the behavior and performance of distributed systems in this large configuration
space is challenging without automation that stretches across the software
stack. In this paper we present Fallout, an open-source distributed systems
testing service that automatically provisions and configures distributed
systems and clients, supports running a variety of workloads and benchmarks,
and generates performance reports based on collected metrics for visual
analysis. We have been running the Fallout service internally at DataStax for
over 5 years and have recently open sourced it to support our work with Apache
Cassandra, Pulsar, and other open source projects. We describe the architecture
of Fallout along with the evolution of its design and the lessons we learned
operating this service in a dynamic environment where teams work on different
products and favor different benchmarking tools.

    

### [[2110.05545] Peformance Prediction for Coarse-Grained Locking: MCS Case](http://arxiv.org/abs/2110.05545)


  A standard design pattern found in many concurrent data structures, such as
hash tables or ordered containers, is alternation of parallelizable sections
that incur no data conflicts and critical sections that must run sequentially
and are protected with locks. It was already shown that simple stochastic
analysis can predict the throughput of coarse-grained lock-based algorithms
using CLH lock. In this short paper, we extend this analysis to algorithms
based on the popular MCS lock.

    

### [[2110.05783] Delay-Sensitive and Power-Efficient Quality Control of Dynamic Video Streaming using Adaptive Super-Resolution](http://arxiv.org/abs/2110.05783)


  In a decade, the adaptive quality control of video streaming and the
super-resolution (SR) technique have been deeply explored. As edge devices
improved to have exceptional processing capability than ever before, streaming
users can enhance the received image quality to allow the transmitter to
compress the images to save its power or pursue network efficiency. In this
sense, this paper proposes a novel dynamic video streaming algorithm that
adaptively compresses video chunks at the transmitter and separately enhances
the quality at the receiver using SR. In order to allow transmission of video
chunks with different compression levels and control of the computation burden,
we present the adaptive SR network which is optimized by minimizing the
weighted sum of losses extracted from different layer outputs. for dynamic
video streaming. In addition, we jointly orchestrate video delivery and
resource usage, and the proposed video delivery scheme balances the tradeoff
well among the average video quality, the queuing delay, buffering time,
transmit power, and computation power. Simulation results show that the
proposed scheme pursues the quality-of-services (QoS) of the video streaming
better than the adaptive quality control without the cooperation of the
transmitter and the receiver and the non-adaptive SR network.

    

### [[2110.06003] Impact of delay classes on the data structure in IOTA](http://arxiv.org/abs/2110.06003)


  In distributed ledger technologies (DLTs) with a directed acyclic graph (DAG)
data structure, a message-issuing node can decide where to append that message
and, consequently, how to grow the DAG. This DAG data structure can typically
be decomposed into two pools of messages: referenced messages and unreferenced
messages (tips). The selection of the parent messages to which a node appends
the messages it issues, depends on which messages it considers as tips.
However, the exact time that a message enters the tip pool of a node depends on
the delay of that message. In previous works, it was considered that messages
have the same or similar delay; however, this generally may not be the case. We
introduce the concept of classes of delays, where messages belonging to a
certain class have a specific delay, and where these classes coexist in the
DAG. We provide a general model that predicts the tip pool size for any finite
number of different classes.
This categorisation and model is applied to the first iteration of the IOTA
2.0 protocol (a.k.a. Coordicide), where two distinct classes, namely value and
data messages, coexist. We show that the tip pool size depends strongly on the
dominating class that is present. Finally, we provide a methodology for
controlling the tip pool size by dynamically adjusting the number of references
a message creates.

    

### [[2110.06013] On Wave-Based Majority Gates with Cellular Automata](http://arxiv.org/abs/2110.06013)


  We demonstrate a discrete implementation of a wave-based majority gate in a
chaotic Life-like cellular automaton. The gate functions via controlling of
patterns' propagation into stationary channels. The gate presented is
realisable in many living and non-living substrates that show wave-like
activity of its space-time dynamics or pattern propagation. In the gate a
symmetric pattern represents a binary value 0 while a non-symmetric pattern
represents a binary value 1. Origination of the patterns and their symmetry
type are encoded by the particle reactions at the beginning of computation. The
patterns propagate in channels of the gate and compete for the space at the
intersection of the channels. We implement 3-inputs majority gates using a W
topology showing additional implementations of 5-inputs majority gates and one
tree (cascade) majority gate.

    

### [[2011.04719] Probabilistic Indistinguishability and the Quality of Validity in Byzantine Agreement](http://arxiv.org/abs/2011.04719)


  Lower bounds and impossibility results in distributed computing are both
intellectually challenging and practically important. Hundreds if not thousands
of proofs appear in the literature, but surprisingly, the vast majority of them
apply to deterministic algorithms only. Probabilistic protocols have been
around for at least four decades and are receiving a lot of attention with the
emergence of blockchain systems. Nonetheless, we are aware of only a handful of
randomized lower bounds.
In this paper we provide a formal framework for reasoning about randomized
distributed algorithms. We generalize the notion of indistinguishability, the
most useful tool in deterministic lower bounds, to apply to a probabilistic
setting. We apply this framework to prove a result of independent interest.
Namely, we completely characterize the quality of decisions that protocols for
a randomized multi-valued Consensus problem can guarantee in an asynchronous
environment with Byzantine faults. We use the new notion to prove a lower bound
on the probability at which it can be guaranteed that honest parties will not
decide on a possibly bogus value. Finally, we show that the bound is tight by
providing a protocol that matches it.

    

### [[2110.05009] Long-term balanced allocation via thinning](http://arxiv.org/abs/2110.05009)


  We study the long-term behavior of the two-thinning variant of the classical
balls-and-bins model. In this model, an overseer is provided with uniform
random allocation of $m$ balls into $n$ bins in an on-line fashion. For each
ball, the overseer could reject its allocation and place the ball into a new
bin drawn independently at random. The purpose of the overseer is to reduce the
maximum load of the bins, which is defined as the difference between the
maximum number of balls in a single bin and $m/n$, i.e., the average number of
balls among all bins.
We provide tight estimates for three quantities: the lowest maximum load that
could be achieved at time $m$, the lowest maximum load that could be achieved
uniformly over the entire time interval $[m]:=\{1, 2, \cdots, m\}$, and the
lowest \emph{typical} maximum load that could be achieved over the interval
$[m]$, where the typicality means that the maximum load holds for $1-o(1)$
portion of the times in $[m]$.
We show that when $m$ and $n$ are sufficiently large, a typical maximum load
of $(\log n)^{1/2+o(1)}$ can be achieved with high probability, asymptotically
the same as the optimal maximum load that could be achieved at time $m$.
However, for any strategy, the maximal load among all times in the interval
$[m]$ is $\Omega\big(\frac{\log n}{\log\log n}\big)$ with high probability. A
strategy achieving this bound is provided.
An explanation for this gap is provided by our optimal strategies as follows.
To control the typical load, we restrain the maximum load for some time, during
which we accumulate more and more bins with relatively high load. After a
while, we have to employ for a short time a different strategy to reduce the
number of relatively heavily loaded bins, at the expanse of temporarily
inducing high load in a few bins.

    

### [[2110.05561] UrbanNet: Leveraging Urban Maps for Long Range 3D Object Detection](http://arxiv.org/abs/2110.05561)


  Relying on monocular image data for precise 3D object detection remains an
open problem, whose solution has broad implications for cost-sensitive
applications such as traffic monitoring. We present UrbanNet, a modular
architecture for long range monocular 3D object detection with static cameras.
Our proposed system combines commonly available urban maps along with a mature
2D object detector and an efficient 3D object descriptor to accomplish accurate
detection at long range even when objects are rotated along any of their three
axes. We evaluate UrbanNet on a novel challenging synthetic dataset and
highlight the advantages of its design for traffic detection in roads with
changing slope, where the flat ground approximation does not hold. Data and
code are available at this https URL


### [[2110.05633] TCube: Domain-Agnostic Neural Time-series Narration](http://arxiv.org/abs/2110.05633)


  The task of generating rich and fluent narratives that aptly describe the
characteristics, trends, and anomalies of time-series data is invaluable to the
sciences (geology, meteorology, epidemiology) or finance (trades, stocks, or
sales and inventory). The efforts for time-series narration hitherto are
domain-specific and use predefined templates that offer consistency but lead to
mechanical narratives. We present TCube (Time-series-to-text), a
domain-agnostic neural framework for time-series narration, that couples the
representation of essential time-series elements in the form of a dense
knowledge graph and the translation of said knowledge graph into rich and
fluent narratives through the transfer-learning capabilities of PLMs
(Pre-trained Language Models). TCube's design primarily addresses the challenge
that lies in building a neural framework in the complete paucity of annotated
training data for time-series. The design incorporates knowledge graphs as an
intermediary for the representation of essential time-series elements which can
be linearized for textual translation. To the best of our knowledge, TCube is
the first investigation of the use of neural strategies for time-series
narration. Through extensive evaluations, we show that TCube can improve the
lexical diversity of the generated narratives by up to 65.38% while still
maintaining grammatical integrity. The practicality and deployability of TCube
is further validated through an expert review (n=21) where 76.2% of
participating experts wary of auto-generated narratives favored TCube as a
deployable system for time-series narration due to its richer narratives. Our
code-base, models, and datasets, with detailed instructions for reproducibility
is publicly hosted at this https URL.

    

### [[2110.05664] Accurate and Generalizable Quantitative Scoring of Liver Steatosis from Ultrasound Images via Scalable Deep Learning](http://arxiv.org/abs/2110.05664)


  Background & Aims: Hepatic steatosis is a major cause of chronic liver
disease. 2D ultrasound is the most widely used non-invasive tool for screening
and monitoring, but associated diagnoses are highly subjective. We developed a
scalable deep learning (DL) algorithm for quantitative scoring of liver
steatosis from 2D ultrasound images.
Approach & Results: Using retrospectively collected multi-view ultrasound
data from 3,310 patients, 19,513 studies, and 228,075 images, we trained a DL
algorithm to diagnose steatosis stages (healthy, mild, moderate, or severe)
from ultrasound diagnoses. Performance was validated on two multi-scanner
unblinded and blinded (initially to DL developer) histology-proven cohorts (147
and 112 patients) with histopathology fatty cell percentage diagnoses, and a
subset with FibroScan diagnoses. We also quantified reliability across scanners
and viewpoints. Results were evaluated using Bland-Altman and receiver
operating characteristic (ROC) analysis. The DL algorithm demonstrates
repeatable measurements with a moderate number of images (3 for each viewpoint)
and high agreement across 3 premium ultrasound scanners. High diagnostic
performance was observed across all viewpoints: area under the curves of the
ROC to classify >=mild, >=moderate, =severe steatosis grades were 0.85, 0.90,
and 0.93, respectively. The DL algorithm outperformed or performed at least
comparably to FibroScan with statistically significant improvements for all
levels on the unblinded histology-proven cohort, and for =severe steatosis on
the blinded histology-proven cohort.
Conclusions: The DL algorithm provides a reliable quantitative steatosis
assessment across view and scanners on two multi-scanner cohorts. Diagnostic
performance was high with comparable or better performance than FibroScan.

    

### [[2110.05687] No way to crop: On robust image crop localization](http://arxiv.org/abs/2110.05687)


  Previous image forensics schemes for crop detection are only limited on
predicting whether an image has been cropped. This paper presents a novel
scheme for image crop localization using robust watermarking. We further extend
our scheme to detect tampering attack on the attacked image. We demonstrate
that our scheme is the first to provide high-accuracy and robust image crop
localization. Besides, the accuracy of tamper detection is comparable to many
state-of-the-art methods.

    

### [[2110.05689] Hiding Images into Images with Real-world Robustness](http://arxiv.org/abs/2110.05689)


  The existing image embedding networks are basically vulnerable to malicious
attacks such as JPEG compression and noise adding, not applicable for
real-world copyright protection tasks. To solve this problem, we introduce a
generative deep network based method for hiding images into images while
assuring high-quality extraction from the destructive synthesized images. An
embedding network is sequentially concatenated with an attack layer, a
decoupling network and an image extraction network. The addition of decoupling
network learns to extract the embedded watermark from the attacked image. We
also pinpoint the weaknesses of the adversarial training for robustness in
previous works and build our improved real-world attack simulator. Experimental
results demonstrate the superiority of the proposed method against typical
digital attacks by a large margin, as well as the performance boost of the
recovered images with the aid of progressive recovery strategy. Besides, we are
the first to robustly hide three secret images.

    

### [[2110.05690] Partial Counterfactual Identification from Observational and Experimental Data](http://arxiv.org/abs/2110.05690)


  This paper investigates the problem of bounding counterfactual queries from
an arbitrary collection of observational and experimental distributions and
qualitative knowledge about the underlying data-generating model represented in
the form of a causal diagram. We show that all counterfactual distributions in
an arbitrary structural causal model (SCM) could be generated by a canonical
family of SCMs with the same causal diagram where unobserved (exogenous)
variables are discrete with a finite domain. Utilizing the canonical SCMs, we
translate the problem of bounding counterfactuals into that of polynomial
programming whose solution provides optimal bounds for the counterfactual
query. Solving such polynomial programs is in general computationally
expensive. We therefore develop effective Monte Carlo algorithms to approximate
the optimal bounds from an arbitrary combination of observational and
experimental data. Our algorithms are validated extensively on synthetic and
real-world datasets.

    

### [[2110.05723] Prediction of Political Leanings of Chinese Speaking Twitter Users](http://arxiv.org/abs/2110.05723)


  This work presents a supervised method for generating a classifier model of
the stances held by Chinese-speaking politicians and other Twitter users. Many
previous works of political tweets prediction exist on English tweets, but to
the best of our knowledge, this is the first work that builds prediction model
on Chinese political tweets. It firstly collects data by scraping tweets of
famous political figure and their related users. It secondly defines the
political spectrum in two groups: the group that shows approvals to the Chinese
Communist Party and the group that does not. Since there are not space between
words in Chinese to identify the independent words, it then completes
segmentation and vectorization by Jieba, a Chinese segmentation tool. Finally,
it trains the data collected from political tweets and produce a classification
model with high accuracy for understanding users' political stances from their
tweets on Twitter.

    

### [[2110.05743] Program Transfer and Ontology Awareness for Semantic Parsing in KBQA](http://arxiv.org/abs/2110.05743)


  Semantic parsing in KBQA aims to parse natural language questions into
logical forms, whose execution against a knowledge base produces answers.
Learning semantic parsers from question-answer pairs requires searching over a
huge space of logical forms for ones consistent with answers. Current methods
utilize various prior knowlege or entity-level KB constraints to reduce the
search space. In this paper, we investigate for the first time prior knowledge
from external logical form annotations and ontology-level constraints. We
design a hierarchical architecture for program transfer, and propose an
ontology-guided pruning algorithm to reduce the search space. The experiments
on ComplexWebQuestions show that our method improves the state-of-the-art F1
score from 44.0% to 58.7%, with an absolute gain of 14.7%, which demonstrates
the effectiveness of program transfer and ontology awareness.

    

### [[2110.05792] Aspect-driven User Preference and News Representation Learning for News Recommendation](http://arxiv.org/abs/2110.05792)


  News recommender systems are essential for helping users to efficiently and
effectively find out those interesting news from a large amount of news. Most
of existing news recommender systems usually learn topic-level representations
of users and news for recommendation, and neglect to learn more informative
aspect-level features of users and news for more accurate recommendation. As a
result, they achieve limited recommendation performance. Aiming at addressing
this deficiency, we propose a novel Aspect-driven News Recommender System
(ANRS) built on aspect-level user preference and news representation learning.
Here, \textit{news aspect} is fine-grained semantic information expressed by a
set of related words, which indicates specific aspects described by the news.
In ANRS, \textit{news aspect-level encoder} and \textit{user aspect-level
encoder} are devised to learn the fine-grained aspect-level representations of
user's preferences and news characteristics respectively, which are fed into
\textit{click predictor} to judge the probability of the user clicking the
candidate news. Extensive experiments are done on the commonly used real-world
dataset MIND, which demonstrate the superiority of our method compared with
representative and state-of-the-art methods.

    

### [[2110.05810] Open Player Modeling: Empowering Players through Data Transparency](http://arxiv.org/abs/2110.05810)


  Data is becoming an important central point for making design decisions for
most software. Game development is not an exception. As data-driven methods and
systems start to populate these environments, a good question is: can we make
models developed from this data transparent to users? In this paper, we
synthesize existing work from the Intelligent User Interface and Learning
Science research communities, where they started to investigate the potential
of making such data and models available to users. We then present a new area
exploring this question, which we call Open Player Modeling, as an emerging
research area. We define the design space of Open Player Models and present
exciting open problems that the games research community can explore. We
conclude the paper with a case study and discuss the potential value of this
approach.

    

### [[2110.05836] AVoE: A Synthetic 3D Dataset on Understanding Violation of Expectation for Artificial Cognition](http://arxiv.org/abs/2110.05836)


  Recent work in cognitive reasoning and computer vision has engendered an
increasing popularity for the Violation-of-Expectation (VoE) paradigm in
synthetic datasets. Inspired by work in infant psychology, researchers have
started evaluating a model's ability to discriminate between expected and
surprising scenes as a sign of its reasoning ability. Existing VoE-based 3D
datasets in physical reasoning only provide vision data. However, current
cognitive models of physical reasoning by psychologists reveal infants create
high-level abstract representations of objects and interactions. Capitalizing
on this knowledge, we propose AVoE: a synthetic 3D VoE-based dataset that
presents stimuli from multiple novel sub-categories for five event categories
of physical reasoning. Compared to existing work, AVoE is armed with
ground-truth labels of abstract features and rules augmented to vision data,
paving the way for high-level symbolic predictions in physical reasoning tasks.

    

### [[2110.05861] Convolutional Neural Networks Are Not Invariant to Translation, but They Can Learn to Be](http://arxiv.org/abs/2110.05861)


  When seeing a new object, humans can immediately recognize it across
different retinal locations: the internal object representation is invariant to
translation. It is commonly believed that Convolutional Neural Networks (CNNs)
are architecturally invariant to translation thanks to the convolution and/or
pooling operations they are endowed with. In fact, several studies have found
that these networks systematically fail to recognise new objects on untrained
locations. In this work, we test a wide variety of CNNs architectures showing
how, apart from DenseNet-121, none of the models tested was architecturally
invariant to translation. Nevertheless, all of them could learn to be invariant
to translation. We show how this can be achieved by pretraining on ImageNet,
and it is sometimes possible with much simpler data sets when all the items are
fully translated across the input canvas. At the same time, this invariance can
be disrupted by further training due to catastrophic forgetting/interference.
These experiments show how pretraining a network on an environment with the
right `latent' characteristics (a more naturalistic environment) can result in
the network learning deep perceptual rules which would dramatically improve
subsequent generalization.

    

### [[2110.05886] MGH: Metadata Guided Hypergraph Modeling for Unsupervised Person Re-identification](http://arxiv.org/abs/2110.05886)


  As a challenging task, unsupervised person ReID aims to match the same
identity with query images which does not require any labeled information. In
general, most existing approaches focus on the visual cues only, leaving
potentially valuable auxiliary metadata information (e.g., spatio-temporal
context) unexplored. In the real world, such metadata is normally available
alongside captured images, and thus plays an important role in separating
several hard ReID matches. With this motivation in mind, we
propose~\textbf{MGH}, a novel unsupervised person ReID approach that uses meta
information to construct a hypergraph for feature learning and label
refinement. In principle, the hypergraph is composed of camera-topology-aware
hyperedges, which can model the heterogeneous data correlations across cameras.
Taking advantage of label propagation on the hypergraph, the proposed approach
is able to effectively refine the ReID results, such as correcting the wrong
labels or smoothing the noisy labels. Given the refined results, We further
present a memory-based listwise loss to directly optimize the average precision
in an approximate manner. Extensive experiments on three benchmarks demonstrate
the effectiveness of the proposed approach against the state-of-the-art.

    

### [[2110.05929] One Timestep is All You Need: Training Spiking Neural Networks with Ultra Low Latency](http://arxiv.org/abs/2110.05929)


  Spiking Neural Networks (SNNs) are energy efficient alternatives to commonly
used deep neural networks (DNNs). Through event-driven information processing,
SNNs can reduce the expensive compute requirements of DNNs considerably, while
achieving comparable performance. However, high inference latency is a
significant hindrance to the edge deployment of deep SNNs. Computation over
multiple timesteps not only increases latency as well as overall energy budget
due to higher number of operations, but also incurs memory access overhead of
fetching membrane potentials, both of which lessen the energy benefits of SNNs.
To overcome this bottleneck and leverage the full potential of SNNs, we propose
an Iterative Initialization and Retraining method for SNNs (IIR-SNN) to perform
single shot inference in the temporal axis. The method starts with an SNN
trained with T timesteps (T>1). Then at each stage of latency reduction, the
network trained at previous stage with higher timestep is utilized as
initialization for subsequent training with lower timestep. This acts as a
compression method, as the network is gradually shrunk in the temporal domain.
In this paper, we use direct input encoding and choose T=5, since as per
literature, it is the minimum required latency to achieve satisfactory
performance on ImageNet. The proposed scheme allows us to obtain SNNs with up
to unit latency, requiring a single forward pass during inference. We achieve
top-1 accuracy of 93.05%, 70.15% and 67.71% on CIFAR-10, CIFAR-100 and
ImageNet, respectively using VGG16, with just 1 timestep. In addition, IIR-SNNs
perform inference with 5-2500X reduced latency compared to other
state-of-the-art SNNs, maintaining comparable or even better accuracy.
Furthermore, in comparison with standard DNNs, the proposed IIR-SNNs
provide25-33X higher energy efficiency, while being comparable to them in
classification performance.

    

### [[2110.05951] Evolving Evolutionary Algorithms with Patterns](http://arxiv.org/abs/2110.05951)


  A new model for evolving Evolutionary Algorithms (EAs) is proposed in this
paper. The model is based on the Multi Expression Programming (MEP) technique.
Each MEP chromosome encodes an evolutionary pattern that is repeatedly used for
generating the individuals of a new generation. The evolved pattern is embedded
into a standard evolutionary scheme that is used for solving a particular
problem. Several evolutionary algorithms for function optimization are evolved
by using the considered model. The evolved evolutionary algorithms are compared
with a human-designed Genetic Algorithm. Numerical experiments show that the
evolved evolutionary algorithms can compete with standard approaches for
several well-known benchmarking problems.

    

### [[2110.05968] Improving Character Error Rate Is Not Equal to Having Clean Speech: Speech Enhancement for ASR Systems with Black-box Acoustic Models](http://arxiv.org/abs/2110.05968)


  A deep neural network (DNN)-based speech enhancement (SE) aiming to maximize
the performance of an automatic speech recognition (ASR) system is proposed in
this paper. In order to optimize the DNN-based SE model in terms of the
character error rate (CER), which is one of the metric to evaluate the ASR
system and generally non-differentiable, our method uses two DNNs: one for
speech processing and one for mimicking the output CERs derived through an
acoustic model (AM). Then both of DNNs are alternately optimized in the
training phase. Even if the AM is a black-box, e.g., like one provided by a
third-party, the proposed method enables the DNN-based SE model to be optimized
in terms of the CER since the DNN mimicking the AM is differentiable.
Consequently, it becomes feasible to build CER-centric SE model that has no
negative effect, e.g., additional calculation cost and changing network
architecture, on the inference phase since our method is merely a training
scheme for the existing DNN-based methods. Experimental results show that our
method improved CER by 7.3% relative derived through a black-box AM although
certain noise levels are kept.

    

### [[2110.05985] A Categorical Semantics of Fuzzy Concepts in Conceptual Spaces](http://arxiv.org/abs/2110.05985)


  We define a symmetric monoidal category modelling fuzzy concepts and fuzzy
conceptual reasoning within Gärdenfors' framework of conceptual (convex)
spaces. We propose log-concave functions as models of fuzzy concepts, showing
that these are the most general choice satisfying a criterion due to
Gärdenfors and which are well-behaved compositionally. We then generalise
these to define the category of log-concave probabilistic channels between
convex spaces, which allows one to model fuzzy reasoning with noisy inputs, and
provides a novel example of a Markov category.

    

### [[2110.05992] Weighted Model Counting in FO2 with Cardinality Constraints and Counting Quantifiers: A Closed Form Formula](http://arxiv.org/abs/2110.05992)


  Weighted First-Order Model Counting (WFOMC) computes the weighted sum of the
models of a first-order logic theory on a given finite domain. First-Order
Logic theories that admit polynomial-time WFOMC w.r.t domain cardinality are
called domain liftable. We introduce the concept of lifted interpretations as a
tool for formulating closed-forms for WFOMC. Using lifted interpretations, we
reconstruct the closed-form formula for polynomial-time FOMC in the universally
quantified fragment of FO2, earlier proposed by Beame et al. We then expand
this closed-form to incorporate cardinality constraints, existential
quantifiers, and counting quantifiers (a.k.a C2) without losing
domain-liftability. Finally, we show that the obtained closed-form motivates a
natural definition of a family of weight functions strictly larger than
symmetric weight functions.

    

### [[2110.06006] Robust Glare Detection: Review, Analysis, and Dataset Release](http://arxiv.org/abs/2110.06006)


  Sun Glare widely exists in the images captured by unmanned ground and aerial
vehicles performing in outdoor environments. The existence of such artifacts in
images will result in wrong feature extraction and failure of autonomous
systems. Humans will try to adapt their view once they observe a glare
(especially when driving), and this behavior is an essential requirement for
the next generation of autonomous vehicles. The source of glare is not limited
to the sun, and glare can be seen in the images captured during the nighttime
and in indoor environments, which is due to the presence of different light
sources; reflective surfaces also influence the generation of such artifacts.
The glare's visual characteristics are different on images captured by various
cameras and depend on several factors such as the camera's shutter speed and
exposure level. Hence, it is challenging to introduce a general - robust and
accurate - algorithm for glare detection that can perform well in various
captured images. This research aims to introduce the first dataset for glare
detection, which includes images captured by different cameras. Besides, the
effect of multiple image representations and their combination in glare
detection is examined using the proposed deep network architecture. The
released dataset is available at this https URL


### [[2110.06161] Sign Language Recognition via Skeleton-Aware Multi-Model Ensemble](http://arxiv.org/abs/2110.06161)


  Sign language is commonly used by deaf or mute people to communicate but
requires extensive effort to master. It is usually performed with the fast yet
delicate movement of hand gestures, body posture, and even facial expressions.
Current Sign Language Recognition (SLR) methods usually extract features via
deep neural networks and suffer overfitting due to limited and noisy data.
Recently, skeleton-based action recognition has attracted increasing attention
due to its subject-invariant and background-invariant nature, whereas
skeleton-based SLR is still under exploration due to the lack of hand
annotations. Some researchers have tried to use off-line hand pose trackers to
obtain hand keypoints and aid in recognizing sign language via recurrent neural
networks. Nevertheless, none of them outperforms RGB-based approaches yet. To
this end, we propose a novel Skeleton Aware Multi-modal Framework with a Global
Ensemble Model (GEM) for isolated SLR (SAM-SLR-v2) to learn and fuse
multi-modal feature representations towards a higher recognition rate.
Specifically, we propose a Sign Language Graph Convolution Network (SL-GCN) to
model the embedded dynamics of skeleton keypoints and a Separable
Spatial-Temporal Convolution Network (SSTCN) to exploit skeleton features. The
skeleton-based predictions are fused with other RGB and depth based modalities
by the proposed late-fusion GEM to provide global information and make a
faithful SLR prediction. Experiments on three isolated SLR datasets demonstrate
that our proposed SAM-SLR-v2 framework is exceedingly effective and achieves
state-of-the-art performance with significant margins. Our code will be
available at this https URL


### [[2110.06199] ABO: Dataset and Benchmarks for Real-World 3D Object Understanding](http://arxiv.org/abs/2110.06199)


  We introduce Amazon-Berkeley Objects (ABO), a new large-scale dataset of
product images and 3D models corresponding to real household objects. We use
this realistic, object-centric 3D dataset to measure the domain gap for
single-view 3D reconstruction networks trained on synthetic objects. We also
use multi-view images from ABO to measure the robustness of state-of-the-art
metric learning approaches to different camera viewpoints. Finally, leveraging
the physically-based rendering materials in ABO, we perform single- and
multi-view material estimation for a variety of complex, real-world geometries.
The full dataset is available for download at
this https URL.

    

### [[2003.06649] Partial Queries for Constraint Acquisition](http://arxiv.org/abs/2003.06649)


  Learning constraint networks is known to require a number of membership
queries exponential in the number of variables. In this paper, we learn
constraint networks by asking the user partial queries. That is, we ask the
user to classify assignments to subsets of the variables as positive or
negative. We provide an algorithm, called QUACQ, that, given a negative
example, focuses onto a constraint of the target network in a number of queries
logarithmic in the size of the example. The whole constraint network can then
be learned with a polynomial number of partial queries. We give information
theoretic lower bounds for learning some simple classes of constraint networks
and show that our generic algorithm is optimal in some cases.

    

### [[2008.01188] Learning to Play Two-Player Perfect-Information Games without Knowledge](http://arxiv.org/abs/2008.01188)


  In this paper, several techniques for learning game state evaluation
functions by reinforcement are proposed. The first is a generalization of tree
bootstrapping (tree learning): it is adapted to the context of reinforcement
learning without knowledge based on non-linear functions. With this technique,
no information is lost during the reinforcement learning process. The second is
a modification of minimax with unbounded depth extending the best sequences of
actions to the terminal states. This modified search is intended to be used
during the learning process. The third is to replace the classic gain of a game
(+1 / -1) with a reinforcement heuristic. We study particular reinforcement
heuristics such as: quick wins and slow defeats ; scoring ; mobility or
presence. The four is another variant of unbounded minimax, which plays the
safest action instead of playing the best action. This modified search is
intended to be used after the learning process. The five is a new action
selection distribution. The conducted experiments suggest that these techniques
improve the level of play. Finally, we apply these different techniques to
design program-players to the game of Hex (size 11 and 13) surpassing the level
of Mohex 3HNN with reinforcement learning from self-play without knowledge.

    

### [[2104.05570] Learning from Subjective Ratings Using Auto-Decoded Deep Latent Embeddings](http://arxiv.org/abs/2104.05570)


  Depending on the application, radiological diagnoses can be associated with
high inter- and intra-rater variabilities. Most computer-aided diagnosis (CAD)
solutions treat such data as incontrovertible, exposing learning algorithms to
considerable and possibly contradictory label noise and biases. Thus, managing
subjectivity in labels is a fundamental problem in medical imaging analysis. To
address this challenge, we introduce auto-decoded deep latent embeddings
(ADDLE), which explicitly models the tendencies of each rater using an
auto-decoder framework. After a simple linear transformation, the latent
variables can be injected into any backbone at any and multiple points,
allowing the model to account for rater-specific effects on the diagnosis.
Importantly, ADDLE does not expect multiple raters per image in training,
meaning it can readily learn from data mined from hospital archives. Moreover,
the complexity of training ADDLE does not increase as more raters are added.
During inference each rater can be simulated and a 'mean' or 'greedy' virtual
rating can be produced. We test ADDLE on the problem of liver steatosis
diagnosis from 2D ultrasound (US) by collecting 46 084 studies along with
clinical US diagnoses originating from 65 different raters. We evaluated
diagnostic performance using a separate dataset with gold-standard biopsy
diagnoses. ADDLE can improve the partial areas under the curve (AUCs) for
diagnosing severe steatosis by 10.5% over standard classifiers while
outperforming other annotator-noise approaches, including those requiring 65
times the parameters.

    

### [[2106.15844] Bounded rationality for relaxing best response and mutual consistency: The Quantal Hierarchy model of decision-making](http://arxiv.org/abs/2106.15844)


  While game theory has been transformative for decision-making, the
assumptions made can be overly restrictive in certain instances. In this work,
we focus on some of the assumptions underlying rationality such as mutual
consistency and best response, and consider ways to relax these assumptions
using concepts from level-$k$ reasoning and quantal response equilibrium (QRE)
respectively. Specifically, we provide an information-theoretic two-parameter
model that can relax both mutual consistency and best response, but can recover
approximations of level-$k$, QRE, or typical Nash equilibrium behaviour in the
limiting cases. The proposed Quantal Hierarchy model is based on a recursive
form of the variational free energy principle, representing self-referential
games as (pseudo) sequential decisions. Bounds in player processing abilities
are captured as information costs, where future chains of reasoning are
discounted, implying a hierarchy of players where lower-level players have
fewer processing resources. We demonstrate the applicability of the proposed
model to several canonical economic games.

    

### [[2110.04725] Yuan 1.0: Large-Scale Pre-trained Language Model in Zero-Shot and Few-Shot Learning](http://arxiv.org/abs/2110.04725)


  Recent work like GPT-3 has demonstrated excellent performance of Zero-Shot
and Few-Shot learning on many natural language processing (NLP) tasks by
scaling up model size, dataset size and the amount of computation. However,
training a model like GPT-3 requires huge amount of computational resources
which makes it challengeable to researchers. In this work, we propose a method
that incorporates large-scale distributed training performance into model
architecture design. With this method, Yuan 1.0, the current largest singleton
language model with 245B parameters, achieves excellent performance on
thousands GPUs during training, and the state-of-the-art results on NLP tasks.
A data processing method is designed to efficiently filter massive amount of
raw data. The current largest high-quality Chinese corpus with 5TB high quality
texts is built based on this method. In addition, a calibration and label
expansion method is proposed to improve the Zero-Shot and Few-Shot performance,
and steady improvement is observed on the accuracy of various tasks. Yuan 1.0
presents strong capacity of natural language generation, and the generated
articles are difficult to distinguish from the human-written ones.

    

### [[2110.04984] Advances in Multi-turn Dialogue Comprehension: A Survey](http://arxiv.org/abs/2110.04984)


  Training machines to understand natural language and interact with humans is
an elusive and essential task of artificial intelligence. A diversity of
dialogue systems has been designed with the rapid development of deep learning
techniques, especially the recent pre-trained language models (PrLMs). Among
these studies, the fundamental yet challenging type of task is dialogue
comprehension whose role is to teach the machines to read and comprehend the
dialogue context before responding. In this paper, we review the previous
methods from the technical perspective of dialogue modeling for the dialogue
comprehension task. We summarize the characteristics and challenges of dialogue
comprehension in contrast to plain-text reading comprehension. Then, we discuss
three typical patterns of dialogue modeling. In addition, we categorize
dialogue-related pre-training techniques which are employed to enhance PrLMs in
dialogue scenarios. Finally, we highlight the technical advances in recent
years and point out the lessons from the empirical analysis and the prospects
towards a new frontier of researches.

    

### [[2110.05033] Pitch Preservation In Singing Voice Synthesis](http://arxiv.org/abs/2110.05033)


  Suffering from limited singing voice corpus, existing singing voice synthesis
(SVS) methods that build encoder-decoder neural networks to directly generate
spectrogram could lead to out-of-tune issues during the inference phase. To
attenuate these issues, this paper presents a novel acoustic model with
independent pitch encoder and phoneme encoder, which disentangles the phoneme
and pitch information from music score to fully utilize the corpus.
Specifically, according to equal temperament theory, the pitch encoder is
constrained by a pitch metric loss that maps distances between adjacent input
pitches into corresponding frequency multiples between the encoder outputs. For
the phoneme encoder, based on the analysis that same phonemes corresponding to
varying pitches can produce similar pronunciations, this encoder is followed by
an adversarially trained pitch classifier to enforce the identical phonemes
with different pitches mapping into the same phoneme feature space. By these
means, the sparse phonemes and pitches in original input spaces can be
transformed into more compact feature spaces respectively, where the same
elements cluster closely and cooperate mutually to enhance synthesis quality.
Then, the outputs of the two encoders are summed together to pass through the
following decoder in the acoustic model. Experimental results indicate that the
proposed approaches can characterize intrinsic structure between pitch inputs
to obtain better pitch synthesis accuracy and achieve superior singing
synthesis performance against the advanced baseline system.

    

### [[2110.05146] ViSeRet: A simple yet effective approach to moment retrieval via fine-grained video segmentation](http://arxiv.org/abs/2110.05146)


  Video-text retrieval has many real-world applications such as media
analytics, surveillance, and robotics. This paper presents the 1st place
solution to the video retrieval track of the ICCV VALUE Challenge 2021. We
present a simple yet effective approach to jointly tackle two video-text
retrieval tasks (video retrieval and video corpus moment retrieval) by
leveraging the model trained only on the video retrieval task. In addition, we
create an ensemble model that achieves the new state-of-the-art performance on
all four datasets (TVr, How2r, YouCook2r, and VATEXr) presented in the VALUE
Challenge.

    

### [[2110.05638] Searching for Replacement Classes](http://arxiv.org/abs/2110.05638)


  Software developers must often replace existing components in their systems
to adapt to evolving environments or tooling. While traditional code search
systems are effective at retrieving components with related functionality, it
is much more challenging to retrieve components that can be used to directly
replace existing functionality, as replacements must account for more
fundamental program properties such as type compatibility. To address this
problem, we introduce ClassFinder, a system which given a query class Q, and a
search corpus S, returns a ranked subset of classes that can replace Q and its
functionality. ClassFinder produces afield and method mapping between the
classes that can provide useful hints to a developer and can be used to
effectively refine the ranking of candidate replacement classes. Our technique
leverages the complementary strengths of a distributed embeddings-based search
and type-based analysis, using the former to prune down candidates for an
optimization-based approach based on the latter. ClassFinder retrieves
replacement classes, along with a type-aware field/method mapping between
classes. We evaluate ClassFinder on a search space of ~600thousand open
sourceJava classes. Querying ClassFinder with 24 known Java classes provided
meaningful replacement classes and mappings, in many cases producing complete
mappings with functionally identical replacement classes.

    

### [[2110.05771] Toward SMT-Based Refinement Types in Agda](http://arxiv.org/abs/2110.05771)


  Dependent types offer great versatility and power, but developing proofs with
them can be tedious and requires considerable human guidance. We propose to
integrate Satisfiability Modulo Theories (SMT)-based refinement types into the
dependently-typed language Agda in an effort to ease some of the burden of
programming with dependent types and combine the strengths of the two
approaches to mechanized theorem proving.

    

### [[2110.06107] Generic Level Polymorphic N-ary Functions](http://arxiv.org/abs/2110.06107)


  Agda's standard library struggles in various places with n-ary functions and
relations. It introduces congruence and substitution operators for functions of
arities one and two, and provides users with convenient combinators for
manipulating indexed families of arity exactly one.
After a careful analysis of the kinds of problems the unifier can easily
solve, we design a unifier-friendly representation of n-ary functions. This
allows us to write generic programs acting on n-ary functions which
automatically reconstruct the representation of their inputs' types by
unification. In particular, we can define fully level polymorphic n-ary
versions of congruence, substitution and the combinators for indexed families,
all requiring minimal user input.

    

### [[2001.11001] A Type and Scope Safe Universe of Syntaxes with Binding: Their Semantics and Proofs](http://arxiv.org/abs/2001.11001)


  Almost every programming language's syntax includes a notion of binder and
corresponding bound occurrences, along with the accompanying notions of
$\alpha$-equivalence, capture-avoiding substitution, typing contexts, runtime
environments, and so on. In the past, implementing and reasoning about
programming languages required careful handling to maintain the correct
behaviour of bound variables. Modern programming languages include features
that enable constraints like scope safety to be expressed in types.
Nevertheless, the programmer is still forced to write the same boilerplate over
again for each new implementation of a scope safe operation (e.g., renaming,
substitution, desugaring, printing, etc.), and then again for correctness
proofs.
We present an expressive universe of syntaxes with binding and demonstrate
how to (1) implement scope safe traversals once and for all by generic
programming; and (2) how to derive properties of these traversals by generic
proving. Our universe description, generic traversals and proofs, and our
examples have all been formalised in Agda and are available in the accompanying
material available online at this https URL.

    

### [[2010.05812] A Complete Approach to Loop Verification with Invariants and Summaries](http://arxiv.org/abs/2010.05812)


  Invariants are the predominant approach to verify the correctness of loops.
As an alternative, loop contracts, which make explicit the premise and
conclusion of the underlying induction proof, can sometimes capture correctness
conditions more naturally. But despite this advantage, the second approach
receives little attention overall, and the goal of this paper is to lift it out
of its niche. We give the first comprehensive exposition of the theory of loop
contracts, including a characterization of its completeness. We show concrete
examples on standard algorithms that showcase their relative merits. Moreover,
we demonstrate a novel constructive translation between the two approaches,
which decouples the chosen specification approach from the verification
backend.

    

### [<title>Lack of reproducibility despite using the same seed - XGBoost</title>](https://discuss.xgboost.ai/t/lack-of-reproducibility-despite-using-the-same-seed/2485/4)

### [<title>NaN values with early stopping - XGBoost</title>](https://discuss.xgboost.ai/t/nan-values-with-early-stopping/2494/1)

### [<title>NaN values with early stopping - XGBoost</title>](https://discuss.xgboost.ai/t/nan-values-with-early-stopping/2494/2)

### [<title>Lack of reproducibility despite using the same seed - XGBoost</title>](https://discuss.xgboost.ai/t/lack-of-reproducibility-despite-using-the-same-seed/2485/5)