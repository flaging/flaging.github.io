
## 2021-11-16

### [<title>XGBoost4j - sparse vector prediction - XGBoost</title>](https://discuss.xgboost.ai/t/xgboost4j-sparse-vector-prediction/2534/3)

### [<title>Some notes on using esbuild</title>](https://jvns.ca/blog/2021/11/15/esbuild-vue/)

### [[2111.06978] RLOps: Development Life-cycle of Reinforcement Learning Aided Open RAN](http://arxiv.org/abs/2111.06978)


  Radio access network (RAN) technologies continue to witness massive growth,
with Open RAN gaining the most recent momentum. In the O-RAN specifications,
the RAN intelligent controller (RIC) serves as an automation host. This article
introduces principles for machine learning (ML), in particular, reinforcement
learning (RL) relevant for the O-RAN stack. Furthermore, we review
state-of-the-art research in wireless networks and cast it onto the RAN
framework and the hierarchy of the O-RAN architecture. We provide a taxonomy of
the challenges faced by ML/RL models throughout the development life-cycle:
from the system specification to production deployment (data acquisition, model
design, testing and management, etc.). To address the challenges, we integrate
a set of existing MLOps principles with unique characteristics when RL agents
are considered. This paper discusses a systematic life-cycle model development,
testing and validation pipeline, termed: RLOps. We discuss all fundamental
parts of RLOps, which include: model specification, development and
distillation, production environment serving, operations monitoring,
safety/security and data engineering platform. Based on these principles, we
propose the best practices for RLOps to achieve an automated and reproducible
model development process.

    

### [[2111.07006] A Framework for Routing DNN Inference Jobs over Distributed Computing Networks](http://arxiv.org/abs/2111.07006)


  Ubiquitous artificial intelligence (AI) is considered one of the key services
in 6G systems. AI services typically rely on deep neural network (DNN)
requiring heavy computation. Hence, in order to support ubiquitous AI, it is
crucial to provide a solution for offloading or distributing computational
burden due to DNN, especially at end devices with limited resources. We develop
a framework for assigning the computation tasks of DNN inference jobs to the
nodes with computing resources in the network, so as to reduce the inference
latency in the presence of limited computing power at end devices. To this end,
we propose a layered graph model that enables to solve the problem of assigning
computation tasks of a single DNN inference job via simple conventional
routing. Using this model, we develop algorithms for routing DNN inference jobs
over the distributed computing network. We show through numerical evaluations
that our algorithms can select nodes and paths adaptively to the computational
attributes of given DNN inference jobs in order to reduce the end-to-end
latency.

    

### [[2111.07038] Commodity Wi-Fi Sensing in 10 Years: Current Status, Challenges, and Opportunities](http://arxiv.org/abs/2111.07038)


  The prevalence of WiFi devices and ubiquitous coverage of WiFi networks
provide us the opportunity to extend WiFi capabilities beyond communication,
particularly in sensing the physical environment. In this paper, we survey the
evolution of WiFi sensing systems utilizing commodity devices over the past
decade. It groups WiFi sensing systems into three main categories: activity
recognition (large-scale and small-scale), object sensing, and localization. We
highlight the milestone work in each category and the underline techniques they
adopted. Next, this work presents the challenges faced by existing WiFi sensing
systems. Lastly, we comprehensively discuss the future trending of commodity
WiFi sensing.

    

### [[2111.07078] Networking of Internet of UAVs: Challenges and Intelligent Approaches](http://arxiv.org/abs/2111.07078)


  Internet of unmanned aerial vehicle (I-UAV) networks promise to accomplish
sensing and transmission tasks quickly, robustly, and cost-efficiently via
effective cooperation among UAVs. To achieve the promising benefits, the
crucial I-UAV networking issue should be tackled. This article argues that
I-UAV networking can be classified into three categories, quality-of-service
(QoS) driven networking, quality-of-experience (QoE) driven networking, and
situation aware networking. Each category of networking poses emerging
challenges which have severe effects on the safe and efficient accomplishment
of I-UAV missions. This article elaborately analyzes these challenges and
expounds on the corresponding intelligent approaches to tackle the I-UAV
networking issue. Besides, considering the uplifting effect of extending the
scalability of I-UAV networks through cooperating with high altitude platforms
(HAPs), this article gives an overview of the integrated HAP and I-UAV networks
and presents the corresponding networking challenges and intelligent
approaches.

    

### [[2111.07229] Video Streaming in Cooperative Vehicular Networks](http://arxiv.org/abs/2111.07229)


  Video services in vehicular networks play a significant role in our daily
traveling. In this paper, we propose a cooperative communication scheme to
facilitate video data transmission, utilizing the mobility of vehicles and the
cooperation among infrastructure and vehicles. To improve the video quality of
experience (QoE), i.e., reduce the interruption ratio, quality variation and
improve the playback quality, we design a Back Compensation (BC) video
transmission strategy with the knowledge of vehicle status information. In
addition, we analyze the throughput with one-hop and target-cluster-based
cooperation schemes and obtain their closed-form expressions, respectively,
which is useful for video encoding design in the central server. Simulation
results demonstrate that the proposed approach can improve the video
performance significantly and verify the accuracy of our analytical results.

    

### [[2111.07392] Edge-Native Intelligence for 6G Communications Driven by Federated Learning: A Survey of Trends and Challenges](http://arxiv.org/abs/2111.07392)


  The unprecedented surge of data volume in wireless networks empowered with
artificial intelligence (AI) opens up new horizons for providing ubiquitous
data-driven intelligent services. Traditional cloud-centric machine learning
(ML)-based services are implemented by collecting datasets and training models
centrally. However, this conventional training technique encompasses two
challenges: (i) high communication and energy cost due to increased data
communication, (ii) threatened data privacy by allowing untrusted parties to
utilise this information. Recently, in light of these limitations, a new
emerging technique, coined as federated learning (FL), arose to bring ML to the
edge of wireless networks. FL can extract the benefits of data silos by
training a global model in a distributed manner, orchestrated by the FL server.
FL exploits both decentralised datasets and computing resources of
participating clients to develop a generalised ML model without compromising
data privacy. In this article, we introduce a comprehensive survey of the
fundamentals and enabling technologies of FL. Moreover, an extensive study is
presented detailing various applications of FL in wireless networks and
highlighting their challenges and limitations. The efficacy of FL is further
explored with emerging prospective beyond fifth generation (B5G) and sixth
generation (6G) communication systems. The purpose of this survey is to provide
an overview of the state-of-the-art of FL applications in key wireless
technologies that will serve as a foundation to establish a firm understanding
of the topic. Lastly, we offer a road forward for future research directions.

    

### [[2111.07457] Attentive Federated Learning for Concept Drift in Distributed 5G Edge Networks](http://arxiv.org/abs/2111.07457)


  Machine learning (ML) is expected to play a major role in 5G edge computing.
Various studies have demonstrated that ML is highly suitable for optimizing
edge computing systems as rapid mobility and application-induced changes occur
at the edge. For ML to provide the best solutions, it is important to
continually train the ML models to include the changing scenarios. The sudden
changes in data distributions caused by changing scenarios (e.g., 5G base
station failures) is referred to as concept drift and is a major challenge to
continual learning. The ML models can present high error rates while the drifts
take place and the errors decrease only after the model learns the
distributions. This problem is more pronounced in a distributed setting where
multiple ML models are being used for different heterogeneous datasets and the
final model needs to capture all concept drifts. In this paper, we show that
using Attention in Federated Learning (FL) is an efficient way of handling
concept drifts. We use a 5G network traffic dataset to simulate concept drift
and test various scenarios. The results indicate that Attention can
significantly improve the concept drift handling capability of FL.

    

### [[2111.07480] Power Allocation for Wireless Federated Learning using Graph Neural Networks](http://arxiv.org/abs/2111.07480)


  We propose a data-driven approach for power allocation in the context of
federated learning (FL) over interference-limited wireless networks. The power
policy is designed to maximize the transmitted information during the FL
process under communication constraints, with the ultimate objective of
improving the accuracy and efficiency of the global FL model being trained. The
proposed power allocation policy is parameterized using a graph convolutional
network and the associated constrained optimization problem is solved through a
primal-dual algorithm. Numerical experiments show that the proposed method
outperforms three baseline methods in both transmission success rate and FL
global performance.

    

### [[2111.07583] Optimizing Unlicensed Coexistence Network Performance Through Data Learning](http://arxiv.org/abs/2111.07583)


  Unlicensed LTE-WiFi coexistence networks are undergoing consistent
densification to meet the rising mobile data demands. With the increase in
coexistence network complexity, it is important to study network feature
relationships (NFRs) and utilize them to optimize dense coexistence network
performance. This work studies NFRs in unlicensed LTE-WiFi (LTE-U and LTE-LAA)
networks through supervised learning of network data collected from real-world
experiments. Different 802.11 standards and varying channel bandwidths are
considered in the experiments and the learning model selection policy is
precisely outlined. Thereafter, a comparative analysis of different LTE-WiFi
network configurations is performed through learning model parameters such as
R-sq, residual error, outliers, choice of predictor, etc. Further, a Network
Feature Relationship based Optimization (NeFRO) framework is proposed. NeFRO
improves upon the conventional optimization formulations by utilizing the
feature-relationship equations learned from network data. It is demonstrated to
be highly suitable for time-critical dense coexistence networks through two
optimization objectives, viz., network capacity and signal strength. NeFRO is
validated against four recent works on network optimization. NeFRO is
successfully able to reduce optimization convergence time by as much as 24%
while maintaining accuracy as high as 97.16%, on average.

    

### [[2111.07637] Drone delivery: Reliable Cellular UAV Communication Using Multi-Operator Diversity](http://arxiv.org/abs/2111.07637)


  The market size of Unmanned Aerial Vehicles (UAVs, a.k.a drones) can reach up
to 10\% of the global market value. In particular, drone delivery is one of the
most attractive applications. The growing number of drones requires appropriate
traffic management systems that will rely on cellular networks. However, it has
been shown in the literature that these networks cannot provide reliable
communication due to low coverage probability and frequent handovers. This
article presents a potential solution targeting these problems while requiring
no modifications of the existing infrastructure. Namely, equipping the UAV with
multiple cellular modems to connect to different providers' networks introduces
network diversity resulting in 98\% coverage probability at the flight altitude
of 100 meters. In contrast, one network ensures only 80\% coverage. At the same
time, the size of the outage zones becomes up to ten times smaller and the
frequency of harmful handovers is reduced to zero. The results are obtained
with a physical-layer simulator utilizing a real urban 3D environment, cellular
network parameters (e.g., site locations, antenna orientation and gains), and
specific aerial channel models.

    

### [[1912.11146] A Replication Strategy for Mobile Opportunistic Networks based on Utility Clustering](http://arxiv.org/abs/1912.11146)


  Dynamic replication is a wide-spread multi-copy routing approach for
efficiently coping with the intermittent connectivity in mobile opportunistic
networks. According to it, a node forwards a message replica to an encountered
node based on a utility value that captures the latter's fitness for delivering
the message to the destination. The popularity of the approach stems from its
flexibility to effectively operate in networks with diverse characteristics
without requiring special customization. Nonetheless, its drawback is the
tendency to produce a high number of replicas that consume limited resources
such as energy and storage. To tackle the problem we make the observation that
network nodes can be grouped, based on their utility values, into clusters that
portray different delivery capabilities. We exploit this finding to transform
the basic forwarding strategy, which is to move a packet using nodes of
increasing utility, and actually forward it through clusters of increasing
delivery capability. The new strategy works in synergy with the basic dynamic
replication algorithms and is fully configurable, in the sense that it can be
used with virtually any utility function. We also extend our approach to work
with two utility functions at the same time, a feature that is especially
efficient in mobile networks that exhibit social characteristics. By conducting
experiments in a wide set of real-life networks, we empirically show that our
method is robust in reducing the overall number of replicas in networks with
diverse connectivity characteristics without at the same time hindering
delivery efficiency.

    

### [[2101.11871] Website fingerprinting on early QUIC traffic](http://arxiv.org/abs/2101.11871)


  Cryptographic protocols have been widely used to protect the user's privacy
and avoid exposing private information. QUIC (Quick UDP Internet Connections),
including the version originally designed by Google (GQUIC) and the version
standardized by IETF (IQUIC), as alternatives to the traditional HTTP,
demonstrate their unique transmission characteristics: based on UDP for
encrypted resource transmitting, accelerating web page rendering. However,
existing encrypted transmission schemes based on TCP are vulnerable to website
fingerprinting (WFP) attacks, allowing adversaries to infer the users' visited
websites by eavesdropping on the transmission channel. Whether GQUIC and IQUIC
can effectively resist such attacks is worth investigating. In this paper, we
study the vulnerabilities of GQUIC, IQUIC, and HTTPS to WFP attacks from the
perspective of traffic analysis. Extensive experiments show that, in the early
traffic scenario, GQUIC is the most vulnerable to WFP attacks among GQUIC,
IQUIC, and HTTPS, while IQUIC is more vulnerable than HTTPS, but the
vulnerability of the three protocols is similar in the normal full traffic
scenario. Features transferring analysis shows that most features are
transferable between protocols when on normal full traffic scenario. However,
combining with the qualitative analysis of latent feature representation, we
find that the transferring is inefficient when on early traffic, as GQUIC,
IQUIC, and HTTPS show the significantly different magnitude of variation in the
traffic distribution on early traffic. By upgrading the one-time WFP attacks to
multiple WFP Top-a attacks, we find that the attack accuracy on GQUIC and IQUIC
reach 95.4% and 95.5%, respectively, with only 40 packets and just using simple
features, whereas reach only 60.7% when on HTTPS. We also demonstrate that the
vulnerability of IQUIC is only slightly dependent on the network environment.

    

### [[2102.12169] Digital-Twin-Enabled 6G: Vision, Architectural Trends, and Future Directions](http://arxiv.org/abs/2102.12169)


  Internet of Everything (IoE) applications such as haptics, human-computer
interaction, and extended reality, using the sixth-generation (6G) of wireless
systems have diverse requirements in terms of latency, reliability, data rate,
and user-defined performance metrics. Therefore, enabling IoE applications over
6G requires a new framework that can be used to manage, operate, and optimize
the 6G wireless system and its underlying IoE services. Such a new framework
for 6G can be based on digital twins. Digital twins use a virtual
representation of the 6G physical system along with the associated algorithms
(e.g., machine learning, optimization), communication technologies (e.g.,
millimeter-wave and terahertz communication), computing systems (e.g., edge
computing and cloud computing), as well as privacy and security-related
technologists (e.g., blockchain). First, we present the key design requirements
for enabling 6G through the use of a digital twin. Next, the architectural
components and trends such as edge-based twins, cloud-based-twins, and
edge-cloud-based twins are presented. Furthermore, we provide a comparative
description of various twins. Finally, we outline and recommend guidelines for
several future research directions.

    

### [[2111.06888] Learning Generalized Gumbel-max Causal Mechanisms](http://arxiv.org/abs/2111.06888)


  To perform counterfactual reasoning in Structural Causal Models (SCMs), one
needs to know the causal mechanisms, which provide factorizations of
conditional distributions into noise sources and deterministic functions
mapping realizations of noise to samples. Unfortunately, the causal mechanism
is not uniquely identified by data that can be gathered by observing and
interacting with the world, so there remains the question of how to choose
causal mechanisms. In recent work, Oberst & Sontag (2019) propose Gumbel-max
SCMs, which use Gumbel-max reparameterizations as the causal mechanism due to
an intuitively appealing counterfactual stability property. In this work, we
instead argue for choosing a causal mechanism that is best under a quantitative
criteria such as minimizing variance when estimating counterfactual treatment
effects. We propose a parameterized family of causal mechanisms that generalize
Gumbel-max. We show that they can be trained to minimize counterfactual effect
variance and other losses on a distribution of queries of interest, yielding
lower variance estimates of counterfactual treatment effect than fixed
alternatives, also generalizing to queries not seen at training time.

    

### [[2111.06889] DriverGym: Democratising Reinforcement Learning for Autonomous Driving](http://arxiv.org/abs/2111.06889)


  Despite promising progress in reinforcement learning (RL), developing
algorithms for autonomous driving (AD) remains challenging: one of the critical
issues being the absence of an open-source platform capable of training and
effectively validating the RL policies on real-world data. We propose
DriverGym, an open-source OpenAI Gym-compatible environment specifically
tailored for developing RL algorithms for autonomous driving. DriverGym
provides access to more than 1000 hours of expert logged data and also supports
reactive and data-driven agent behavior. The performance of an RL policy can be
easily validated on real-world data using our extensive and flexible
closed-loop evaluation protocol. In this work, we also provide behavior cloning
baselines using supervised learning and RL, trained in DriverGym. We make
DriverGym code, as well as all the baselines publicly available to further
stimulate development from the community.

    

### [[2111.06907] Improving Experience Replay through Modeling of Similar Transitions' Sets](http://arxiv.org/abs/2111.06907)


  In this work, we propose and evaluate a new reinforcement learning method,
COMPact Experience Replay (COMPER), which uses temporal difference learning
with predicted target values based on recurrence over sets of similar
transitions, and a new approach for experience replay based on two transitions
memories. Our objective is to reduce the required number of experiences to
agent training regarding the total accumulated rewarding in the long run. Its
relevance to reinforcement learning is related to the small number of
observations that it needs to achieve results similar to that obtained by
relevant methods in the literature, that generally demand millions of video
frames to train an agent on the Atari 2600 games. We report detailed results
from five training trials of COMPER for just 100,000 frames and about 25,000
iterations with a small experiences memory on eight challenging games of Arcade
Learning Environment (ALE). We also present results for a DQN agent with the
same experimental protocol on the same games set as the baseline. To verify the
performance of COMPER on approximating a good policy from a smaller number of
observations, we also compare its results with that obtained from millions of
frames presented on the benchmark of ALE.

    

### [[2111.06916] Offense Detection in Dravidian Languages using Code-Mixing Index based Focal Loss](http://arxiv.org/abs/2111.06916)


  Over the past decade, we have seen exponential growth in online content
fueled by social media platforms. Data generation of this scale comes with the
caveat of insurmountable offensive content in it. The complexity of identifying
offensive content is exacerbated by the usage of multiple modalities (image,
language, etc.), code mixed language and more. Moreover, even if we carefully
sample and annotate offensive content, there will always exist significant
class imbalance in offensive vs non offensive content. In this paper, we
introduce a novel Code-Mixing Index (CMI) based focal loss which circumvents
two challenges (1) code mixing in languages (2) class imbalance problem for
Dravidian language offense detection. We also replace the conventional dot
product-based classifier with the cosine-based classifier which results in a
boost in performance. Further, we use multilingual models that help transfer
characteristics learnt across languages to work effectively with low resourced
languages. It is also important to note that our model handles instances of
mixed script (say usage of Latin and Dravidian - Tamil script) as well. Our
model can handle offensive language detection in a low-resource, class
imbalanced, multilingual and code mixed setting.

    

### [[2111.06924] A Simple and Fast Baseline for Tuning Large XGBoost Models](http://arxiv.org/abs/2111.06924)


  XGBoost, a scalable tree boosting algorithm, has proven effective for many
prediction tasks of practical interest, especially using tabular datasets.
Hyperparameter tuning can further improve the predictive performance, but
unlike neural networks, full-batch training of many models on large datasets
can be time consuming. Owing to the discovery that (i) there is a strong linear
relation between dataset size & training time, (ii) XGBoost models satisfy the
ranking hypothesis, and (iii) lower-fidelity models can discover promising
hyperparameter configurations, we show that uniform subsampling makes for a
simple yet fast baseline to speed up the tuning of large XGBoost models using
multi-fidelity hyperparameter optimization with data subsets as the fidelity
dimension. We demonstrate the effectiveness of this baseline on large-scale
tabular datasets ranging from $15-70\mathrm{GB}$ in size.

    

### [[2111.06929] Hierarchical Bayesian Bandits](http://arxiv.org/abs/2111.06929)


  Meta-, multi-task, and federated learning can be all viewed as solving
similar tasks, drawn from an unknown distribution that reflects task
similarities. In this work, we provide a unified view of all these problems, as
learning to act in a hierarchical Bayesian bandit. We analyze a natural
hierarchical Thompson sampling algorithm (hierTS) that can be applied to any
problem in this class. Our regret bounds hold under many instances of such
problems, including when the tasks are solved sequentially or in parallel; and
capture the structure of the problems, such that the regret decreases with the
width of the task prior. Our proofs rely on novel total variance
decompositions, which can be applied to other graphical model structures.
Finally, our theory is complemented by experiments, which show that the
hierarchical structure helps with knowledge sharing among the tasks. This
confirms that hierarchical Bayesian bandits are a universal and
statistically-efficient tool for learning to act with similar bandit tasks.

    

### [[2111.06934] Contrastive Feature Loss for Image Prediction](http://arxiv.org/abs/2111.06934)


  Training supervised image synthesis models requires a critic to compare two
images: the ground truth to the result. Yet, this basic functionality remains
an open problem. A popular line of approaches uses the L1 (mean absolute error)
loss, either in the pixel or the feature space of pretrained deep networks.
However, we observe that these losses tend to produce overly blurry and grey
images, and other techniques such as GANs need to be employed to fight these
artifacts. In this work, we introduce an information theory based approach to
measuring similarity between two images. We argue that a good reconstruction
should have high mutual information with the ground truth. This view enables
learning a lightweight critic to "calibrate" a feature space in a contrastive
manner, such that reconstructions of corresponding spatial patches are brought
together, while other patches are repulsed. We show that our formulation
immediately boosts the perceptual realism of output images when used as a
drop-in replacement for the L1 loss, with or without an additional GAN loss.

    

### [[2111.06942] Predictive coding, precision and natural gradients](http://arxiv.org/abs/2111.06942)


  There is an increasing convergence between biologically plausible
computational models of inference and learning with local update rules and the
global gradient-based optimization of neural network models employed in machine
learning. One particularly exciting connection is the correspondence between
the locally informed optimization in predictive coding networks and the error
backpropagation algorithm that is used to train state-of-the-art deep
artificial neural networks. Here we focus on the related, but still largely
under-explored connection between precision weighting in predictive coding
networks and the Natural Gradient Descent algorithm for deep neural networks.
Precision-weighted predictive coding is an interesting candidate for scaling up
uncertainty-aware optimization -- particularly for models with large parameter
spaces -- due to its distributed nature of the optimization process and the
underlying local approximation of the Fisher information metric, the adaptive
learning rate that is central to Natural Gradient Descent. Here, we show that
hierarchical predictive coding networks with learnable precision indeed are
able to solve various supervised and unsupervised learning tasks with
performance comparable to global backpropagation with natural gradients and
outperform their classical gradient descent counterpart on tasks where high
amounts of noise are embedded in data or label inputs. When applied to
unsupervised auto-encoding of image inputs, the deterministic network produces
hierarchically organized and disentangled embeddings, hinting at the close
connections between predictive coding and hierarchical variational inference.

    

### [[2111.06945] Learning Interpretation with Explainable Knowledge Distillation](http://arxiv.org/abs/2111.06945)


  Knowledge Distillation (KD) has been considered as a key solution in model
compression and acceleration in recent years. In KD, a small student model is
generally trained from a large teacher model by minimizing the divergence
between the probabilistic outputs of the two. However, as demonstrated in our
experiments, existing KD methods might not transfer critical explainable
knowledge of the teacher to the student, i.e. the explanations of predictions
made by the two models are not consistent. In this paper, we propose a novel
explainable knowledge distillation model, called XDistillation, through which
both the performance the explanations' information are transferred from the
teacher model to the student model. The XDistillation model leverages the idea
of convolutional autoencoders to approximate the teacher explanations. Our
experiments shows that models trained by XDistillation outperform those trained
by conventional KD methods not only in term of predictive accuracy but also
faithfulness to the teacher models.

    

### [[2111.06956] Human irrationality: both bad and good for reward inference](http://arxiv.org/abs/2111.06956)


  Assuming humans are (approximately) rational enables robots to infer reward
functions by observing human behavior. But people exhibit a wide array of
irrationalities, and our goal with this work is to better understand the effect
they can have on reward inference. The challenge with studying this effect is
that there are many types of irrationality, with varying degrees of
mathematical formalization. We thus operationalize irrationality in the
language of MDPs, by altering the Bellman optimality equation, and use this
framework to study how these alterations would affect inference.
We find that wrongly modeling a systematically irrational human as
noisy-rational performs a lot worse than correctly capturing these biases -- so
much so that it can be better to skip inference altogether and stick to the
prior! More importantly, we show that an irrational human, when correctly
modelled, can communicate more information about the reward than a perfectly
rational human can. That is, if a robot has the correct model of a human's
irrationality, it can make an even stronger inference than it ever could if the
human were rational. Irrationality fundamentally helps rather than hinder
reward inference, but it needs to be correctly accounted for.

    

### [[2111.06961] Adversarially Robust Learning for Security-Constrained Optimal Power Flow](http://arxiv.org/abs/2111.06961)


  In recent years, the ML community has seen surges of interest in both
adversarially robust learning and implicit layers, but connections between
these two areas have seldom been explored. In this work, we combine innovations
from these areas to tackle the problem of N-k security-constrained optimal
power flow (SCOPF). N-k SCOPF is a core problem for the operation of electrical
grids, and aims to schedule power generation in a manner that is robust to
potentially k simultaneous equipment outages. Inspired by methods in
adversarially robust training, we frame N-k SCOPF as a minimax optimization
problem - viewing power generation settings as adjustable parameters and
equipment outages as (adversarial) attacks - and solve this problem via
gradient-based techniques. The loss function of this minimax problem involves
resolving implicit equations representing grid physics and operational
decisions, which we differentiate through via the implicit function theorem. We
demonstrate the efficacy of our framework in solving N-3 SCOPF, which has
traditionally been considered as prohibitively expensive to solve given that
the problem size depends combinatorially on the number of potential outages.

    

### [[2111.06968] Hierarchical clustering by aggregating representatives in sub-minimum-spanning-trees](http://arxiv.org/abs/2111.06968)


  One of the main challenges for hierarchical clustering is how to
appropriately identify the representative points in the lower level of the
cluster tree, which are going to be utilized as the roots in the higher level
of the cluster tree for further aggregation. However, conventional hierarchical
clustering approaches have adopted some simple tricks to select the
"representative" points which might not be as representative as enough. Thus,
the constructed cluster tree is less attractive in terms of its poor robustness
and weak reliability. Aiming at this issue, we propose a novel hierarchical
clustering algorithm, in which, while building the clustering dendrogram, we
can effectively detect the representative point based on scoring the reciprocal
nearest data points in each sub-minimum-spanning-tree. Extensive experiments on
UCI datasets show that the proposed algorithm is more accurate than other
benchmarks. Meanwhile, under our analysis, the proposed algorithm has O(nlogn)
time-complexity and O(logn) space-complexity, indicating that it has the
scalability in handling massive data with less time and storage consumptions.

    

### [[2111.06977] Scalable Diverse Model Selection for Accessible Transfer Learning](http://arxiv.org/abs/2111.06977)


  With the preponderance of pretrained deep learning models available
off-the-shelf from model banks today, finding the best weights to fine-tune to
your use-case can be a daunting task. Several methods have recently been
proposed to find good models for transfer learning, but they either don't scale
well to large model banks or don't perform well on the diversity of
off-the-shelf models. Ideally the question we want to answer is, "given some
data and a source model, can you quickly predict the model's accuracy after
fine-tuning?" In this paper, we formalize this setting as "Scalable Diverse
Model Selection" and propose several benchmarks for evaluating on this task. We
find that existing model selection and transferability estimation methods
perform poorly here and analyze why this is the case. We then introduce simple
techniques to improve the performance and speed of these algorithms. Finally,
we iterate on existing methods to create PARC, which outperforms all other
methods on diverse model selection. We have released the benchmarks and method
code in hope to inspire future work in model selection for accessible transfer
learning.

    

### [[2111.06979] Neural Population Geometry Reveals the Role of Stochasticity in Robust Perception](http://arxiv.org/abs/2111.06979)


  Adversarial examples are often cited by neuroscientists and machine learning
researchers as an example of how computational models diverge from biological
sensory systems. Recent work has proposed adding biologically-inspired
components to visual neural networks as a way to improve their adversarial
robustness. One surprisingly effective component for reducing adversarial
vulnerability is response stochasticity, like that exhibited by biological
neurons. Here, using recently developed geometrical techniques from
computational neuroscience, we investigate how adversarial perturbations
influence the internal representations of standard, adversarially trained, and
biologically-inspired stochastic networks. We find distinct geometric
signatures for each type of network, revealing different mechanisms for
achieving robust representations. Next, we generalize these results to the
auditory domain, showing that neural stochasticity also makes auditory models
more robust to adversarial perturbations. Geometric analysis of the stochastic
networks reveals overlap between representations of clean and adversarially
perturbed stimuli, and quantitatively demonstrates that competing geometric
effects of stochasticity mediate a tradeoff between adversarial and clean
performance. Our results shed light on the strategies of robust perception
utilized by adversarially trained and stochastic networks, and help explain how
stochasticity may be beneficial to machine and biological computation.

    

### [[2111.06980] GraSSNet: Graph Soft Sensing Neural Networks](http://arxiv.org/abs/2111.06980)


  In the era of big data, data-driven based classification has become an
essential method in smart manufacturing to guide production and optimize
inspection. The industrial data obtained in practice is usually time-series
data collected by soft sensors, which are highly nonlinear, nonstationary,
imbalanced, and noisy. Most existing soft-sensing machine learning models focus
on capturing either intra-series temporal dependencies or pre-defined
inter-series correlations, while ignoring the correlation between labels as
each instance is associated with multiple labels simultaneously. In this paper,
we propose a novel graph based soft-sensing neural network (GraSSNet) for
multivariate time-series classification of noisy and highly-imbalanced
soft-sensing data. The proposed GraSSNet is able to 1) capture the inter-series
and intra-series dependencies jointly in the spectral domain; 2) exploit the
label correlations by superimposing label graph that built from statistical
co-occurrence information; 3) learn features with attention mechanism from both
textual and numerical domain; and 4) leverage unlabeled data and mitigate data
imbalance by semi-supervised learning. Comparative studies with other commonly
used classifiers are carried out on Seagate soft sensing data, and the
experimental results validate the competitive performance of our proposed
method.

    

### [[2111.06981] Soft-Sensing ConFormer: A Curriculum Learning-based Convolutional Transformer](http://arxiv.org/abs/2111.06981)


  Over the last few decades, modern industrial processes have investigated
several cost-effective methodologies to improve the productivity and yield of
semiconductor manufacturing. While playing an essential role in facilitating
real-time monitoring and control, the data-driven soft-sensors in industries
have provided a competitive edge when augmented with deep learning approaches
for wafer fault-diagnostics. Despite the success of deep learning methods
across various domains, they tend to suffer from bad performance on
multi-variate soft-sensing data domains. To mitigate this, we propose a
soft-sensing ConFormer (CONvolutional transFORMER) for wafer fault-diagnostic
classification task which primarily consists of multi-head convolution modules
that reap the benefits of fast and light-weight operations of convolutions, and
also the ability to learn the robust representations through multi-head design
alike transformers. Another key issue is that traditional learning paradigms
tend to suffer from low performance on noisy and highly-imbalanced soft-sensing
data. To address this, we augment our soft-sensing ConFormer model with a
curriculum learning-based loss function, which effectively learns easy samples
in the early phase of training and difficult ones later. To further demonstrate
the utility of our proposed architecture, we performed extensive experiments on
various toolsets of Seagate Technology's wafer manufacturing process which are
shared openly along with this work. To the best of our knowledge, this is the
first time that curriculum learning-based soft-sensing ConFormer architecture
has been proposed for soft-sensing data and our results show strong promise for
future use in soft-sensing research domain.

    

### [[2111.06982] Soft Sensing Model Visualization: Fine-tuning Neural Network from What Model Learned](http://arxiv.org/abs/2111.06982)


  The growing availability of the data collected from smart manufacturing is
changing the paradigms of production monitoring and control. The increasing
complexity and content of the wafer manufacturing process in addition to the
time-varying unexpected disturbances and uncertainties, make it infeasible to
do the control process with model-based approaches. As a result, data-driven
soft-sensing modeling has become more prevalent in wafer process diagnostics.
Recently, deep learning has been utilized in soft sensing system with promising
performance on highly nonlinear and dynamic time-series data. Despite its
successes in soft-sensing systems, however, the underlying logic of the deep
learning framework is hard to understand. In this paper, we propose a deep
learning-based model for defective wafer detection using a highly imbalanced
dataset. To understand how the proposed model works, the deep visualization
approach is applied. Additionally, the model is then fine-tuned guided by the
deep visualization. Extensive experiments are performed to validate the
effectiveness of the proposed system. The results provide an interpretation of
how the model works and an instructive fine-tuning method based on the
interpretation.

    

### [[2111.06994] Learning Online for Unified Segmentation and Tracking Models](http://arxiv.org/abs/2111.06994)


  Tracking requires building a discriminative model for the target in the
inference stage. An effective way to achieve this is online learning, which can
comfortably outperform models that are only trained offline. Recent research
shows that visual tracking benefits significantly from the unification of
visual tracking and segmentation due to its pixel-level discrimination.
However, it imposes a great challenge to perform online learning for such a
unified model. A segmentation model cannot easily learn from prior information
given in the visual tracking scenario. In this paper, we propose TrackMLP: a
novel meta-learning method optimized to learn from only partial information to
resolve the imposed challenge. Our model is capable of extensively exploiting
limited prior information hence possesses much stronger target-background
discriminability than other online learning methods. Empirically, we show that
our model achieves state-of-the-art performance and tangible improvement over
competing models. Our model achieves improved average overlaps of66.0%,67.1%,
and68.5% in VOT2019, VOT2018, and VOT2016 datasets, which are 6.4%,7.3%,
and6.4% higher than our baseline. Code will be made publicly available.

    

### [[2111.07001] LoMEF: A Framework to Produce Local Explanations for Global Model Time Series Forecasts](http://arxiv.org/abs/2111.07001)


  Global Forecasting Models (GFM) that are trained across a set of multiple
time series have shown superior results in many forecasting competitions and
real-world applications compared with univariate forecasting approaches. One
aspect of the popularity of statistical forecasting models such as ETS and
ARIMA is their relative simplicity and interpretability (in terms of relevant
lags, trend, seasonality, and others), while GFMs typically lack
interpretability, especially towards particular time series. This reduces the
trust and confidence of the stakeholders when making decisions based on the
forecasts without being able to understand the predictions. To mitigate this
problem, in this work, we propose a novel local model-agnostic interpretability
approach to explain the forecasts from GFMs. We train simpler univariate
surrogate models that are considered interpretable (e.g., ETS) on the
predictions of the GFM on samples within a neighbourhood that we obtain through
bootstrapping or straightforwardly as the one-step-ahead global black-box model
forecasts of the time series which needs to be explained. After, we evaluate
the explanations for the forecasts of the global models in both qualitative and
quantitative aspects such as accuracy, fidelity, stability and
comprehensibility, and are able to show the benefits of our approach.

    

### [[2111.07009] Leveraging Unsupervised Image Registration for Discovery of Landmark Shape Descriptor](http://arxiv.org/abs/2111.07009)


  In current biological and medical research, statistical shape modeling (SSM)
provides an essential framework for the characterization of anatomy/morphology.
Such analysis is often driven by the identification of a relatively small
number of geometrically consistent features found across the samples of a
population. These features can subsequently provide information about the
population shape variation. Dense correspondence models can provide ease of
computation and yield an interpretable low-dimensional shape descriptor when
followed by dimensionality reduction. However, automatic methods for obtaining
such correspondences usually require image segmentation followed by significant
preprocessing, which is taxing in terms of both computation as well as human
resources. In many cases, the segmentation and subsequent processing require
manual guidance and anatomy specific domain expertise. This paper proposes a
self-supervised deep learning approach for discovering landmarks from images
that can directly be used as a shape descriptor for subsequent analysis. We use
landmark-driven image registration as the primary task to force the neural
network to discover landmarks that register the images well. We also propose a
regularization term that allows for robust optimization of the neural network
and ensures that the landmarks uniformly span the image domain. The proposed
method circumvents segmentation and preprocessing and directly produces a
usable shape descriptor using just 2D or 3D images. In addition, we also
propose two variants on the training loss function that allows for prior shape
information to be integrated into the model. We apply this framework on several
2D and 3D datasets to obtain their shape descriptors, and analyze their utility
for various applications.

    

### [[2111.07015] HydraGAN A Multi-head, Multi-objective Approach to Synthetic Data Generation](http://arxiv.org/abs/2111.07015)


  Synthetic data generation overcomes limitations of real-world machine
learning. Traditional methods are valuable for augmenting costly datasets but
only optimize one criterion: realism. In this paper, we tackle the problem of
generating synthetic data that optimize multiple criteria. This goal is
necessary when real data are replaced by synthetic for privacy preservation. We
introduce HydraGAN, a new approach to synthetic data generation that introduces
multiple generator and discriminator agents into the system. The multi-agent
GAN optimizes the goal of privacy-preservation as well as data realism. To
facilitate multi-agent training, we adapt game-theoretic principles to offer
equilibrium guarantees. We observe that HydraGAN outperforms baseline methods
for three datasets for multiple criteria of maximizing data realism, maximizing
model accuracy, and minimizing re-identification risk.

    

### [[2111.07018] Identification and Adaptive Control of Markov Jump Systems: Sample Complexity and Regret Bounds](http://arxiv.org/abs/2111.07018)


  Learning how to effectively control unknown dynamical systems is crucial for
intelligent autonomous systems. This task becomes a significant challenge when
the underlying dynamics are changing with time. Motivated by this challenge,
this paper considers the problem of controlling an unknown Markov jump linear
system (MJS) to optimize a quadratic objective. By taking a model-based
perspective, we consider identification-based adaptive control for MJSs. We
first provide a system identification algorithm for MJS to learn the dynamics
in each mode as well as the Markov transition matrix, underlying the evolution
of the mode switches, from a single trajectory of the system states, inputs,
and modes. Through mixing-time arguments, sample complexity of this algorithm
is shown to be $\mathcal{O}(1/\sqrt{T})$. We then propose an adaptive control
scheme that performs system identification together with certainty equivalent
control to adapt the controllers in an episodic fashion. Combining our sample
complexity results with recent perturbation results for certainty equivalent
control, we prove that when the episode lengths are appropriately chosen, the
proposed adaptive control scheme achieves $\mathcal{O}(\sqrt{T})$ regret, which
can be improved to $\mathcal{O}(polylog(T))$ with partial knowledge of the
system. Our proof strategy introduces innovations to handle Markovian jumps and
a weaker notion of stability common in MJSs. Our analysis provides insights
into system theoretic quantities that affect learning accuracy and control
performance. Numerical simulations are presented to further reinforce these
insights.

    

### [[2111.07032] Learning to Evolve on Dynamic Graphs](http://arxiv.org/abs/2111.07032)


  Representation learning in dynamic graphs is a challenging problem because
the topology of graph and node features vary at different time. This requires
the model to be able to effectively capture both graph topology information and
temporal information. Most existing works are built on recurrent neural
networks (RNNs), which are used to exact temporal information of dynamic
graphs, and thus they inherit the same drawbacks of RNNs. In this paper, we
propose Learning to Evolve on Dynamic Graphs (LEDG) - a novel algorithm that
jointly learns graph information and time information. Specifically, our
approach utilizes gradient-based meta-learning to learn updating strategies
that have better generalization ability than RNN on snapshots. It is
model-agnostic and thus can train any message passing based graph neural
network (GNN) on dynamic graphs. To enhance the representation power, we
disentangle the embeddings into time embeddings and graph intrinsic embeddings.
We conduct experiments on various datasets and down-stream tasks, and the
experimental results validate the effectiveness of our method.

    

### [[2111.07035] Measuring the Contribution of Multiple Model Representations in Detecting Adversarial Instances](http://arxiv.org/abs/2111.07035)


  Deep learning models have been used for a wide variety of tasks. They are
prevalent in computer vision, natural language processing, speech recognition,
and other areas. While these models have worked well under many scenarios, it
has been shown that they are vulnerable to adversarial attacks. This has led to
a proliferation of research into ways that such attacks could be identified
and/or defended against. Our goal is to explore the contribution that can be
attributed to using multiple underlying models for the purpose of adversarial
instance detection. Our paper describes two approaches that incorporate
representations from multiple models for detecting adversarial examples. We
devise controlled experiments for measuring the detection impact of
incrementally utilizing additional models. For many of the scenarios we
consider, the results show that performance increases with the number of
underlying models used for extracting representations.

    

### [[2111.07046] Iterative Training: Finding Binary Weight Deep Neural Networks with Layer Binarization](http://arxiv.org/abs/2111.07046)


  In low-latency or mobile applications, lower computation complexity, lower
memory footprint and better energy efficiency are desired. Many prior works
address this need by removing redundant parameters. Parameter quantization
replaces floating-point arithmetic with lower precision fixed-point arithmetic,
further reducing complexity.
Typical training of quantized weight neural networks starts from fully
quantized weights. Quantization creates random noise. As a way to compensate
for this noise, during training, we propose to quantize some weights while
keeping others in floating-point precision. A deep neural network has many
layers. To arrive at a fully quantized weight network, we start from one
quantized layer and then quantize more and more layers. We show that the order
of layer quantization affects accuracies. Order count is large for deep neural
networks. A sensitivity pre-training is proposed to guide the layer
quantization order.
Recent work in weight binarization replaces weight-input matrix
multiplication with additions. We apply the proposed iterative training to
weight binarization. Our experiments cover fully connected and convolutional
networks on MNIST, CIFAR-10 and ImageNet datasets. We show empirically that,
starting from partial binary weights instead of from fully binary ones,
training reaches fully binary weight networks with better accuracies for larger
and deeper networks. Layer binarization in the forward order results in better
accuracies. Guided layer binarization can further improve that. The
improvements come at a cost of longer training time.

    

### [[2111.07058] Bolstering Stochastic Gradient Descent with Model Building](http://arxiv.org/abs/2111.07058)


  Stochastic gradient descent method and its variants constitute the core
optimization algorithms that achieve good convergence rates for solving machine
learning problems. These rates are obtained especially when these algorithms
are fine-tuned for the application at hand. Although this tuning process can
require large computational costs, recent work has shown that these costs can
be reduced by line search methods that iteratively adjust the stepsize. We
propose an alternative approach to stochastic line search by using a new
algorithm based on forward step model building. This model building step
incorporates a second-order information that allows adjusting not only the
stepsize but also the search direction. Noting that deep learning model
parameters come in groups (layers of tensors), our method builds its model and
calculates a new step for each parameter group. This novel diagonalization
approach makes the selected step lengths adaptive. We provide convergence rate
analysis, and experimentally show that the proposed algorithm achieves faster
convergence and better generalization in most problems. Moreover, our
experiments show that the proposed method is quite robust as it converges for a
wide range of initial stepsizes.

    

### [[2111.07060] PAMMELA: Policy Administration Methodology using Machine Learning](http://arxiv.org/abs/2111.07060)


  In recent years, Attribute-Based Access Control (ABAC) has become quite
popular and effective for enforcing access control in dynamic and collaborative
environments. Implementation of ABAC requires the creation of a set of
attribute-based rules which cumulatively form a policy. Designing an ABAC
policy ab initio demands a substantial amount of effort from the system
administrator. Moreover, organizational changes may necessitate the inclusion
of new rules in an already deployed policy. In such a case, re-mining the
entire ABAC policy will require a considerable amount of time and
administrative effort. Instead, it is better to incrementally augment the
policy. Keeping these aspects of reducing administrative overhead in mind, in
this paper, we propose PAMMELA, a Policy Administration Methodology using
Machine Learning to help system administrators in creating new ABAC policies as
well as augmenting existing ones. PAMMELA can generate a new policy for an
organization by learning the rules of a policy currently enforced in a similar
organization. For policy augmentation, PAMMELA can infer new rules based on the
knowledge gathered from the existing rules. Experimental results show that our
proposed approach provides a reasonably good performance in terms of the
various machine learning evaluation metrics as well as execution time.

    

### [[2111.07074] Memotion Analysis through the Lens of Joint Embedding](http://arxiv.org/abs/2111.07074)


  Joint embedding (JE) is a way to encode multi-modal data into a vector space
where text remains as the grounding key and other modalities like image are to
be anchored with such keys. Meme is typically an image with embedded text onto
it. Although, memes are commonly used for fun, they could also be used to
spread hate and fake information. That along with its growing ubiquity over
several social platforms has caused automatic analysis of memes to become a
widespread topic of research. In this paper, we report our initial experiments
on Memotion Analysis problem through joint embeddings. Results are marginally
yielding SOTA.

    

### [[2111.07083] Learning Data Teaching Strategies Via Knowledge Tracing](http://arxiv.org/abs/2111.07083)


  Teaching plays a fundamental role in human learning. Typically, a human
teaching strategy would involve assessing a student's knowledge progress for
tailoring the teaching materials in a way that enhances the learning progress.
A human teacher would achieve this by tracing a student's knowledge over
important learning concepts in a task. Albeit, such teaching strategy is not
well exploited yet in machine learning as current machine teaching methods tend
to directly assess the progress on individual training samples without paying
attention to the underlying learning concepts in a learning task. In this
paper, we propose a novel method, called Knowledge Augmented Data Teaching
(KADT), which can optimize a data teaching strategy for a student model by
tracing its knowledge progress over multiple learning concepts in a learning
task. Specifically, the KADT method incorporates a knowledge tracing model to
dynamically capture the knowledge progress of a student model in terms of
latent learning concepts. Then we develop an attention pooling mechanism to
distill knowledge representations of a student model with respect to class
labels, which enables to develop a data teaching strategy on critical training
samples. We have evaluated the performance of the KADT method on four different
machine learning tasks including knowledge tracing, sentiment analysis, movie
recommendation, and image classification. The results comparing to the
state-of-the-art methods empirically validate that KADT consistently
outperforms others on all tasks.

    

### [[2111.07089] Evaluating Contrastive Learning on Wearable Timeseries for Downstream Clinical Outcomes](http://arxiv.org/abs/2111.07089)


  Vast quantities of person-generated health data (wearables) are collected but
the process of annotating to feed to machine learning models is impractical.
This paper discusses ways in which self-supervised approaches that use
contrastive losses, such as SimCLR and BYOL, previously applied to the vision
domain, can be applied to high-dimensional health signals for downstream
classification tasks of various diseases spanning sleep, heart, and metabolic
conditions. To this end, we adapt the data augmentation step and the overall
architecture to suit the temporal nature of the data (wearable traces) and
evaluate on 5 downstream tasks by comparing other state-of-the-art methods
including supervised learning and an adversarial unsupervised representation
learning method. We show that SimCLR outperforms the adversarial method and a
fully-supervised method in the majority of the downstream evaluation tasks, and
that all self-supervised methods outperform the fully-supervised methods. This
work provides a comprehensive benchmark for contrastive methods applied to the
wearable time-series domain, showing the promise of task-agnostic
representations for downstream clinical outcomes.

    

### [[2111.07094] Speech Emotion Recognition Using Deep Sparse Auto-Encoder Extreme Learning Machine with a New Weighting Scheme and Spectro-Temporal Features Along with Classical Feature Selection and A New Quantum-Inspired Dimension Reduction Method](http://arxiv.org/abs/2111.07094)


  Affective computing is very important in the relationship between man and
machine. In this paper, a system for speech emotion recognition (SER) based on
speech signal is proposed, which uses new techniques in different stages of
processing. The system consists of three stages: feature extraction, feature
selection, and finally feature classification. In the first stage, a complex
set of long-term statistics features is extracted from both the speech signal
and the glottal-waveform signal using a combination of new and diverse features
such as prosodic, spectral, and spectro-temporal features. One of the
challenges of the SER systems is to distinguish correlated emotions. These
features are good discriminators for speech emotions and increase the SER's
ability to recognize similar and different emotions. This feature vector with a
large number of dimensions naturally has redundancy. In the second stage, using
classical feature selection techniques as well as a new quantum-inspired
technique to reduce the feature vector dimensionality, the number of feature
vector dimensions is reduced. In the third stage, the optimized feature vector
is classified by a weighted deep sparse extreme learning machine (ELM)
classifier. The classifier performs classification in three steps: sparse
random feature learning, orthogonal random projection using the singular value
decomposition (SVD) technique, and discriminative classification in the last
step using the generalized Tikhonov regularization technique. Also, many
existing emotional datasets suffer from the problem of data imbalanced
distribution, which in turn increases the classification error and decreases
system performance. In this paper, a new weighting method has also been
proposed to deal with class imbalance, which is more efficient than existing
weighting methods. The proposed method is evaluated on three standard emotional
databases.

    

### [[2111.07104] A strong baseline for image and video quality assessment](http://arxiv.org/abs/2111.07104)


  In this work, we present a simple yet effective unified model for perceptual
quality assessment of image and video. In contrast to existing models which
usually consist of complex network architecture, or rely on the concatenation
of multiple branches of features, our model achieves a comparable performance
by applying only one global feature derived from a backbone network (i.e.
resnet18 in the presented work). Combined with some training tricks, the
proposed model surpasses the current baselines of SOTA models on public and
private datasets. Based on the architecture proposed, we release the models
well trained for three common real-world scenarios: UGC videos in the wild, PGC
videos with compression, Game videos with compression. These three pre-trained
models can be directly applied for quality assessment, or be further fine-tuned
for more customized usages. All the code, SDK, and the pre-trained weights of
the proposed models are publicly available at
this https URL.

    

### [[2111.07109] Nyström Regularization for Time Series Forecasting](http://arxiv.org/abs/2111.07109)


  This paper focuses on learning rate analysis of Nyström regularization
with sequential sub-sampling for $\tau$-mixing time series. Using a recently
developed Banach-valued Bernstein inequality for $\tau$-mixing sequences and an
integral operator approach based on second-order decomposition, we succeed in
deriving almost optimal learning rates of Nyström regularization with
sequential sub-sampling for $\tau$-mixing time series. A series of numerical
experiments are carried out to verify our theoretical results, showing the
excellent learning performance of Nyström regularization with sequential
sub-sampling in learning massive time series data. All these results extend the
applicable range of Nyström regularization from i.i.d. samples to
non-i.i.d. sequences.

    

### [[2111.07117] Learning Object-Centric Representations of Multi-Object Scenes from Multiple Views](http://arxiv.org/abs/2111.07117)


  Learning object-centric representations of multi-object scenes is a promising
approach towards machine intelligence, facilitating high-level reasoning and
control from visual sensory data. However, current approaches for unsupervised
object-centric scene representation are incapable of aggregating information
from multiple observations of a scene. As a result, these "single-view" methods
form their representations of a 3D scene based only on a single 2D observation
(view). Naturally, this leads to several inaccuracies, with these methods
falling victim to single-view spatial ambiguities. To address this, we propose
The Multi-View and Multi-Object Network (MulMON) -- a method for learning
accurate, object-centric representations of multi-object scenes by leveraging
multiple views. In order to sidestep the main technical difficulty of the
multi-object-multi-view scenario -- maintaining object correspondences across
views -- MulMON iteratively updates the latent object representations for a
scene over multiple views. To ensure that these iterative updates do indeed
aggregate spatial information to form a complete 3D scene understanding, MulMON
is asked to predict the appearance of the scene from novel viewpoints during
training. Through experiments, we show that MulMON better-resolves spatial
ambiguities than single-view methods -- learning more accurate and disentangled
object representations -- and also achieves new functionality in predicting
object segmentations for novel viewpoints.

    

### [[2111.07125] MC-CIM: Compute-in-Memory with Monte-Carlo Dropouts for Bayesian Edge Intelligence](http://arxiv.org/abs/2111.07125)


  We propose MC-CIM, a compute-in-memory (CIM) framework for robust, yet low
power, Bayesian edge intelligence. Deep neural networks (DNN) with
deterministic weights cannot express their prediction uncertainties, thereby
pose critical risks for applications where the consequences of mispredictions
are fatal such as surgical robotics. To address this limitation, Bayesian
inference of a DNN has gained attention. Using Bayesian inference, not only the
prediction itself, but the prediction confidence can also be extracted for
planning risk-aware actions. However, Bayesian inference of a DNN is
computationally expensive, ill-suited for real-time and/or edge deployment. An
approximation to Bayesian DNN using Monte Carlo Dropout (MC-Dropout) has shown
high robustness along with low computational complexity. Enhancing the
computational efficiency of the method, we discuss a novel CIM module that can
perform in-memory probabilistic dropout in addition to in-memory weight-input
scalar product to support the method. We also propose a compute-reuse
reformulation of MC-Dropout where each successive instance can utilize the
product-sum computations from the previous iteration. Even more, we discuss how
the random instances can be optimally ordered to minimize the overall
MC-Dropout workload by exploiting combinatorial optimization methods.
Application of the proposed CIM-based MC-Dropout execution is discussed for
MNIST character recognition and visual odometry (VO) of autonomous drones. The
framework reliably gives prediction confidence amidst non-idealities imposed by
MC-CIM to a good extent. Proposed MC-CIM with 16x31 SRAM array, 0.85 V supply,
16nm low-standby power (LSTP) technology consumes 27.8 pJ for 30 MC-Dropout
instances of probabilistic inference in its most optimal computing and
peripheral configuration, saving 43% energy compared to typical execution.

    

### [[2111.07126] On the Statistical Benefits of Curriculum Learning](http://arxiv.org/abs/2111.07126)


  Curriculum learning (CL) is a commonly used machine learning training
strategy. However, we still lack a clear theoretical understanding of CL's
benefits. In this paper, we study the benefits of CL in the multitask linear
regression problem under both structured and unstructured settings. For both
settings, we derive the minimax rates for CL with the oracle that provides the
optimal curriculum and without the oracle, where the agent has to adaptively
learn a good curriculum. Our results reveal that adaptive learning can be
fundamentally harder than the oracle learning in the unstructured setting, but
it merely introduces a small extra term in the structured setting. To connect
theory with practice, we provide justification for a popular empirical method
that selects tasks with highest local prediction gain by comparing its
guarantees with the minimax rates mentioned above.

    

### [[2111.07138] Towards One Shot Search Space Poisoning in Neural Architecture Search](http://arxiv.org/abs/2111.07138)


  We evaluate the robustness of a Neural Architecture Search (NAS) algorithm
known as Efficient NAS (ENAS) against data agnostic poisoning attacks on the
original search space with carefully designed ineffective operations. We
empirically demonstrate how our one shot search space poisoning approach
exploits design flaws in the ENAS controller to degrade predictive performance
on classification tasks. With just two poisoning operations injected into the
search space, we inflate prediction error rates for child networks upto 90% on
the CIFAR-10 dataset.

    

### [[2111.07139] Full-attention based Neural Architecture Search using Context Auto-regression](http://arxiv.org/abs/2111.07139)


  Self-attention architectures have emerged as a recent advancement for
improving the performance of vision tasks. Manual determination of the
architecture for self-attention networks relies on the experience of experts
and cannot automatically adapt to various scenarios. Meanwhile, neural
architecture search (NAS) has significantly advanced the automatic design of
neural architectures. Thus, it is appropriate to consider using NAS methods to
discover a better self-attention architecture automatically. However, it is
challenging to directly use existing NAS methods to search attention networks
because of the uniform cell-based search space and the lack of long-term
content dependencies. To address this issue, we propose a full-attention based
NAS method. More specifically, a stage-wise search space is constructed that
allows various attention operations to be adopted for different layers of a
network. To extract global features, a self-supervised search algorithm is
proposed that uses context auto-regression to discover the full-attention
architecture. To verify the efficacy of the proposed methods, we conducted
extensive experiments on various learning tasks, including image
classification, fine-grained image recognition, and zero-shot image retrieval.
The empirical results show strong evidence that our method is capable of
discovering high-performance, full-attention architectures while guaranteeing
the required search efficiency.

    

### [[2111.07140] The Pseudo Projection Operator: Applications of Deep Learning to Projection Based Filtering in Non-Trivial Frequency Regimes](http://arxiv.org/abs/2111.07140)


  Traditional frequency based projection filters, or projection operators (PO),
separate signal and noise through a series of transformations which remove
frequencies where noise is present. However, this technique relies on a priori
knowledge of what frequencies contain signal and noise and that these
frequencies do not overlap, which is difficult to achieve in practice. To
address these issues, we introduce a PO-neural network hybrid model, the Pseudo
Projection Operator (PPO), which leverages a neural network to perform
frequency selection. We compare the filtering capabilities of a PPO, PO, and
denoising autoencoder (DAE) on the University of Rochester Multi-Modal Music
Performance Dataset with a variety of added noise types. In the majority of
experiments, the PPO outperforms both the PO and DAE. Based upon these results,
we suggest future application of the PPO to filtering problems in the physical
and biological sciences.

    

### [[2111.07156] Developing a Novel Approach for Periapical Dental Radiographs Segmentation](http://arxiv.org/abs/2111.07156)


  Image processing techniques has been widely used in dental researches such as
human identification and forensic dentistry, teeth numbering, dental carries
detection and periodontal disease analysis. One of the most challenging parts
in dental imaging is teeth segmentation and how to separate them from each
other. In this paper, an automated method for teeth segmentation of Periapical
dental x-ray images which contain at least one root-canalled tooth is proposed.
The result of this approach can be used as an initial step in bone lesion
detection. The proposed algorithm is made of two stages. The first stage is
pre-processing. The second and main part of this algorithm calculated rotation
degree and uses the integral projection method for tooth isolation.
Experimental results show that this algorithm is robust and achieves high
accuracy.

    

### [[2111.07163] Efficient Binary Embedding of Categorical Data using BinSketch](http://arxiv.org/abs/2111.07163)


  In this work, we present a dimensionality reduction algorithm, aka.
sketching, for categorical datasets. Our proposed sketching algorithm Cabin
constructs low-dimensional binary sketches from high-dimensional categorical
vectors, and our distance estimation algorithm Cham computes a close
approximation of the Hamming distance between any two original vectors only
from their sketches. The minimum dimension of the sketches required by Cham to
ensure a good estimation theoretically depends only on the sparsity of the data
points - making it useful for many real-life scenarios involving sparse
datasets. We present a rigorous theoretical analysis of our approach and
supplement it with extensive experiments on several high-dimensional real-world
data sets, including one with over a million dimensions. We show that the Cabin
and Cham duo is a significantly fast and accurate approach for tasks such as
RMSE, all-pairs similarity, and clustering when compared to working with the
full dataset and other dimensionality reduction techniques.

    

### [[2111.07167] The Three Stages of Learning Dynamics in High-Dimensional Kernel Methods](http://arxiv.org/abs/2111.07167)


  To understand how deep learning works, it is crucial to understand the
training dynamics of neural networks. Several interesting hypotheses about
these dynamics have been made based on empirically observed phenomena, but
there exists a limited theoretical understanding of when and why such phenomena
occur.
In this paper, we consider the training dynamics of gradient flow on kernel
least-squares objectives, which is a limiting dynamics of SGD trained neural
networks. Using precise high-dimensional asymptotics, we characterize the
dynamics of the fitted model in two "worlds": in the Oracle World the model is
trained on the population distribution and in the Empirical World the model is
trained on a sampled dataset. We show that under mild conditions on the kernel
and $L^2$ target regression function the training dynamics undergo three stages
characterized by the behaviors of the models in the two worlds. Our theoretical
results also mathematically formalize some interesting deep learning phenomena.
Specifically, in our setting we show that SGD progressively learns more complex
functions and that there is a "deep bootstrap" phenomenon: during the second
stage, the test error of both worlds remain close despite the empirical
training error being much smaller. Finally, we give a concrete example
comparing the dynamics of two different kernels which shows that faster
training is not necessary for better generalization.

    

### [[2111.07171] Deep Reinforcement Learning with Shallow Controllers: An Experimental Application to PID Tuning](http://arxiv.org/abs/2111.07171)


  Deep reinforcement learning (RL) is an optimization-driven framework for
producing control strategies for general dynamical systems without explicit
reliance on process models. Good results have been reported in simulation. Here
we demonstrate the challenges in implementing a state of the art deep RL
algorithm on a real physical system. Aspects include the interplay between
software and existing hardware; experiment design and sample efficiency;
training subject to input constraints; and interpretability of the algorithm
and control law. At the core of our approach is the use of a PID controller as
the trainable RL policy. In addition to its simplicity, this approach has
several appealing features: No additional hardware needs to be added to the
control system, since a PID controller can easily be implemented through a
standard programmable logic controller; the control law can easily be
initialized in a "safe'' region of the parameter space; and the final product
-- a well-tuned PID controller -- has a form that practitioners can reason
about and deploy with confidence.

    

### [[2111.07180] Explainable Semantic Space by Grounding Language to Vision with Cross-Modal Contrastive Learning](http://arxiv.org/abs/2111.07180)


  In natural language processing, most models try to learn semantic
representations merely from texts. The learned representations encode the
distributional semantics but fail to connect to any knowledge about the
physical world. In contrast, humans learn language by grounding concepts in
perception and action and the brain encodes grounded semantics for cognition.
Inspired by this notion and recent work in vision-language learning, we design
a two-stream model for grounding language learning in vision. The model
includes a VGG-based visual stream and a Bert-based language stream. The two
streams merge into a joint representational space. Through cross-modal
contrastive learning, the model first learns to align visual and language
representations with the MS COCO dataset. The model further learns to retrieve
visual objects with language queries through a cross-modal attention module and
to infer the visual relations between the retrieved objects through a bilinear
operator with the Visual Genome dataset. After training, the language stream of
this model is a stand-alone language model capable of embedding concepts in a
visually grounded semantic space. This semantic space manifests principal
dimensions explainable with human intuition and neurobiological knowledge. Word
embeddings in this semantic space are predictive of human-defined norms of
semantic features and are segregated into perceptually distinctive clusters.
Furthermore, the visually grounded language model also enables compositional
language understanding based on visual knowledge and multimodal image search
with queries based on images, texts, or their combinations.

    

### [[2111.07183] Reliably-stabilizing piecewise-affine neural network controllers](http://arxiv.org/abs/2111.07183)


  A common problem affecting neural network (NN) approximations of model
predictive control (MPC) policies is the lack of analytical tools to assess the
stability of the closed-loop system under the action of the NN-based
controller. We present a general procedure to quantify the performance of such
a controller, or to design minimum complexity NNs with rectified linear units
(ReLUs) that preserve the desirable properties of a given MPC scheme. By
quantifying the approximation error between NN-based and MPC-based
state-to-input mappings, we first establish suitable conditions involving two
key quantities, the worst-case error and the Lipschitz constant, guaranteeing
the stability of the closed-loop system. We then develop an offline,
mixed-integer optimization-based method to compute those quantities exactly.
Together these techniques provide conditions sufficient to certify the
stability and performance of a ReLU-based approximation of an MPC control law.

    

### [[2111.07189] Learning Neural Models for Continuous-Time Sequences](http://arxiv.org/abs/2111.07189)


  The large volumes of data generated by human activities such as online
purchases, health records, spatial mobility etc. are stored as a sequence of
events over a continuous time. Learning deep learning methods over such
sequences is a non-trivial task as it involves modeling the ever-increasing
event timestamps, inter-event time gaps, event types, and the influences
between events -- within and across different sequences. This situation is
further exacerbated by the constraints associated with data collection e.g.
limited data, incomplete sequences, privacy restrictions etc. With the research
direction described in this work, we aim to study the properties of
continuous-time event sequences (CTES) and design robust yet scalable neural
network-based models to overcome the aforementioned problems. In this work, we
model the underlying generative distribution of events using marked temporal
point processes (MTPP) to address a wide range of real-world problems.
Moreover, we highlight the efficacy of the proposed approaches over the
state-of-the-art baselines and later report the ongoing research problems.

    

### [[2111.07198] Keyphrase Extraction Using Neighborhood Knowledge Based on Word Embeddings](http://arxiv.org/abs/2111.07198)


  Keyphrase extraction is the task of finding several interesting phrases in a
text document, which provide a list of the main topics within the document.
Most existing graph-based models use co-occurrence links as cohesion indicators
to model the relationship of syntactic elements. However, a word may have
different forms of expression within the document, and may have several
synonyms as well. Simply using co-occurrence information cannot capture this
information. In this paper, we enhance the graph-based ranking model by
leveraging word embeddings as background knowledge to add semantic information
to the inter-word graph. Our approach is evaluated on established benchmark
datasets and empirical results show that the word embedding neighborhood
information improves the model performance.

    

### [[2111.07220] SDnDTI: Self-supervised deep learning-based denoising for diffusion tensor MRI](http://arxiv.org/abs/2111.07220)


  The noise in diffusion-weighted images (DWIs) decreases the accuracy and
precision of diffusion tensor magnetic resonance imaging (DTI) derived
microstructural parameters and leads to prolonged acquisition time for
achieving improved signal-to-noise ratio (SNR). Deep learning-based image
denoising using convolutional neural networks (CNNs) has superior performance
but often requires additional high-SNR data for supervising the training of
CNNs, which reduces the practical feasibility. We develop a self-supervised
deep learning-based method entitled "SDnDTI" for denoising DTI data, which does
not require additional high-SNR data for training. Specifically, SDnDTI divides
multi-directional DTI data into many subsets, each consisting of six DWI
volumes along optimally chosen diffusion-encoding directions that are robust to
noise for the tensor fitting, and then synthesizes DWI volumes along all
acquired directions from the diffusion tensors fitted using each subset of the
data as the input data of CNNs. On the other hand, SDnDTI synthesizes DWI
volumes along acquired diffusion-encoding directions with higher SNR from the
diffusion tensors fitted using all acquired data as the training target. SDnDTI
removes noise from each subset of synthesized DWI volumes using a deep
3-dimensional CNN to match the quality of the cleaner target DWI volumes and
achieves even higher SNR by averaging all subsets of denoised data. The
denoising efficacy of SDnDTI is demonstrated on two datasets provided by the
Human Connectome Project (HCP) and the Lifespan HCP in Aging. The SDnDTI
results preserve image sharpness and textural details and substantially improve
upon those from the raw data. The results of SDnDTI are comparable to those
from supervised learning-based denoising and outperform those from
state-of-the-art conventional denoising algorithms including BM4D, AONLM and
MPPCA.

    

### [[2111.07228] Curriculum Learning for Vision-and-Language Navigation](http://arxiv.org/abs/2111.07228)


  Vision-and-Language Navigation (VLN) is a task where an agent navigates in an
embodied indoor environment under human instructions. Previous works ignore the
distribution of sample difficulty and we argue that this potentially degrade
their agent performance. To tackle this issue, we propose a novel
curriculum-based training paradigm for VLN tasks that can balance human prior
knowledge and agent learning progress about training samples. We develop the
principle of curriculum design and re-arrange the benchmark Room-to-Room (R2R)
dataset to make it suitable for curriculum training. Experiments show that our
method is model-agnostic and can significantly improve the performance, the
generalizability, and the training efficiency of current state-of-the-art
navigation agents without increasing model complexity.

    

### [[2111.07243] Simulating Diffusion Bridges with Score Matching](http://arxiv.org/abs/2111.07243)


  We consider the problem of simulating diffusion bridges, i.e. diffusion
processes that are conditioned to initialize and terminate at two given states.
Diffusion bridge simulation has applications in diverse scientific fields and
plays a crucial role for statistical inference of discretely-observed
diffusions. This is known to be a challenging problem that has received much
attention in the last two decades. In this work, we first show that the
time-reversed diffusion bridge process can be simulated if one can time-reverse
the unconditioned diffusion process. We introduce a variational formulation to
learn this time-reversal that relies on a score matching method to circumvent
intractability. We then consider another iteration of our proposed methodology
to approximate the Doob's $h$-transform defining the diffusion bridge process.
As our approach is generally applicable under mild assumptions on the
underlying diffusion process, it can easily be used to improve the proposal
bridge process within existing methods and frameworks. We discuss algorithmic
considerations and extensions, and present some numerical results.

    

### [[2111.07265] Linear, or Non-Linear, That is the Question!](http://arxiv.org/abs/2111.07265)


  There were fierce debates on whether the non-linear embedding propagation of
GCNs is appropriate to GCN-based recommender systems. It was recently found
that the linear embedding propagation shows better accuracy than the non-linear
embedding propagation. Since this phenomenon was discovered especially in
recommender systems, it is required that we carefully analyze the linearity and
non-linearity issue. In this work, therefore, we revisit the issues of i) which
of the linear or non-linear propagation is better and ii) which factors of
users/items decide the linearity/non-linearity of the embedding propagation. We
propose a novel Hybrid Method of Linear and non-linEar collaborative filTering
method (HMLET, pronounced as Hamlet). In our design, there exist both linear
and non-linear propagation steps, when processing each user or item node, and
our gating module chooses one of them, which results in a hybrid model of the
linear and non-linear GCN-based collaborative filtering (CF). The proposed
model yields the best accuracy in three public benchmark datasets. Moreover, we
classify users/items into the following three classes depending on our gating
modules' selections: Full-Non-Linearity (FNL), Partial-Non-Linearity (PNL), and
Full-Linearity (FL). We found that there exist strong correlations between
nodes' centrality and their class membership, i.e., important user/item nodes
exhibit more preferences towards the non-linearity during the propagation
steps. To our knowledge, we are the first who designs a hybrid method and
reports the correlation between the graph centrality and the
linearity/non-linearity of nodes. All HMLET codes and datasets are available
at: this https URL.

    

### [[2111.07284] Energy Efficient Learning with Low Resolution Stochastic Domain Wall Synapse Based Deep Neural Networks](http://arxiv.org/abs/2111.07284)


  We demonstrate that extremely low resolution quantized (nominally 5-state)
synapses with large stochastic variations in Domain Wall (DW) position can be
both energy efficient and achieve reasonably high testing accuracies compared
to Deep Neural Networks (DNNs) of similar sizes using floating precision
synaptic weights. Specifically, voltage controlled DW devices demonstrate
stochastic behavior as modeled rigorously with micromagnetic simulations and
can only encode limited states; however, they can be extremely energy efficient
during both training and inference. We show that by implementing suitable
modifications to the learning algorithms, we can address the stochastic
behavior as well as mitigate the effect of their low-resolution to achieve high
testing accuracies. In this study, we propose both in-situ and ex-situ training
algorithms, based on modification of the algorithm proposed by Hubara et al.
[1] which works well with quantization of synaptic weights. We train several
5-layer DNNs on MNIST dataset using 2-, 3- and 5-state DW device as synapse.
For in-situ training, a separate high precision memory unit is adopted to
preserve and accumulate the weight gradients, which are then quantized to
program the low precision DW devices. Moreover, a sizeable noise tolerance
margin is used during the training to address the intrinsic programming noise.
For ex-situ training, a precursor DNN is first trained based on the
characterized DW device model and a noise tolerance margin, which is similar to
the in-situ training. Remarkably, for in-situ inference the energy dissipation
to program the devices is only 13 pJ per inference given that the training is
performed over the entire MNIST dataset for 10 epochs.

    

### [[2111.07307] Improving usual Naive Bayes classifier performances with Neural Naive Bayes based models](http://arxiv.org/abs/2111.07307)


  Naive Bayes is a popular probabilistic model appreciated for its simplicity
and interpretability. However, the usual form of the related classifier suffers
from two major problems. First, as caring about the observations' law, it
cannot consider complex features. Moreover, it considers the conditional
independence of the observations given the hidden variable. This paper
introduces the original Neural Naive Bayes, modeling the parameters of the
classifier induced from the Naive Bayes with neural network functions. This
allows to correct the first problem. We also introduce new Neural Pooled Markov
Chain models, alleviating the independence condition. We empirically study the
benefits of these models for Sentiment Analysis, dividing the error rate of the
usual classifier by 4.5 on the IMDB dataset with the FastText embedding.

    

### [[2111.07334] Relative Distributed Formation and Obstacle Avoidance with Multi-agent Reinforcement Learning](http://arxiv.org/abs/2111.07334)


  Multi-agent formation as well as obstacle avoidance is one of the most
actively studied topics in the field of multi-agent systems. Although some
classic controllers like model predictive control (MPC) and fuzzy control
achieve a certain measure of success, most of them require precise global
information which is not accessible in harsh environments. On the other hand,
some reinforcement learning (RL) based approaches adopt the leader-follower
structure to organize different agents' behaviors, which sacrifices the
collaboration between agents thus suffering from bottlenecks in maneuverability
and robustness. In this paper, we propose a distributed formation and obstacle
avoidance method based on multi-agent reinforcement learning (MARL). Agents in
our system only utilize local and relative information to make decisions and
control themselves distributively. Agent in the multi-agent system will
reorganize themselves into a new topology quickly in case that any of them is
disconnected. Our method achieves better performance regarding formation error,
formation convergence rate and on-par success rate of obstacle avoidance
compared with baselines (both classic control methods and another RL-based
method). The feasibility of our method is verified by both simulation and
hardware implementation with Ackermann-steering vehicles.

    

### [[2111.07337] $p$-Laplacian Based Graph Neural Networks](http://arxiv.org/abs/2111.07337)


  Graph neural networks (GNNs) have demonstrated superior performance for
semi-supervised node classification on graphs, as a result of their ability to
exploit node features and topological information simultaneously. However, most
GNNs implicitly assume that the labels of nodes and their neighbors in a graph
are the same or consistent, which does not hold in heterophilic graphs, where
the labels of linked nodes are likely to differ. Hence, when the topology is
non-informative for label prediction, ordinary GNNs may work significantly
worse than simply applying multi-layer perceptrons (MLPs) on each node. To
tackle the above problem, we propose a new $p$-Laplacian based GNN model,
termed as $^p$GNN, whose message passing mechanism is derived from a discrete
regularization framework and could be theoretically explained as an
approximation of a polynomial graph filter defined on the spectral domain of
$p$-Laplacians. The spectral analysis shows that the new message passing
mechanism works simultaneously as low-pass and high-pass filters, thus making
$^p$GNNs are effective on both homophilic and heterophilic graphs. Empirical
studies on real-world and synthetic datasets validate our findings and
demonstrate that $^p$GNNs significantly outperform several state-of-the-art GNN
architectures on heterophilic benchmarks while achieving competitive
performance on homophilic benchmarks. Moreover, $^p$GNNs can adaptively learn
aggregation weights and are robust to noisy edges.

    

### [[2111.07344] Towards Privacy-Preserving Affect Recognition: A Two-Level Deep Learning Architecture](http://arxiv.org/abs/2111.07344)


  Automatically understanding and recognising human affective states using
images and computer vision can improve human-computer and human-robot
interaction. However, privacy has become an issue of great concern, as the
identities of people used to train affective models can be exposed in the
process. For instance, malicious individuals could exploit images from users
and assume their identities. In addition, affect recognition using images can
lead to discriminatory and algorithmic bias, as certain information such as
race, gender, and age could be assumed based on facial features. Possible
solutions to protect the privacy of users and avoid misuse of their identities
are to: (1) extract anonymised facial features, namely action units (AU) from a
database of images, discard the images and use AUs for processing and training,
and (2) federated learning (FL) i.e. process raw images in users' local
machines (local processing) and send the locally trained models to the main
processing machine for aggregation (central processing). In this paper, we
propose a two-level deep learning architecture for affect recognition that uses
AUs in level 1 and FL in level 2 to protect users' identities. The architecture
consists of recurrent neural networks to capture the temporal relationships
amongst the features and predict valence and arousal affective states. In our
experiments, we evaluate the performance of our privacy-preserving architecture
using different variations of recurrent neural networks on RECOLA, a
comprehensive multimodal affective database. Our results show state-of-the-art
performance of $0.426$ for valence and $0.401$ for arousal using the
Concordance Correlation Coefficient evaluation metric, demonstrating the
feasibility of developing models for affect recognition that are both accurate
and ensure privacy.

    

### [[2111.07348] Invariant Risk Minimisation for Cross-Organism Inference: Substituting Mouse Data for Human Data in Human Risk Factor Discovery](http://arxiv.org/abs/2111.07348)


  Human medical data can be challenging to obtain due to data privacy concerns,
difficulties conducting certain types of experiments, or prohibitive associated
costs. In many settings, data from animal models or in-vitro cell lines are
available to help augment our understanding of human data. However, this data
is known for having low etiological validity in comparison to human data. In
this work, we augment small human medical datasets with in-vitro data and
animal models. We use Invariant Risk Minimisation (IRM) to elucidate invariant
features by considering cross-organism data as belonging to different
data-generating environments. Our models identify genes of relevance to human
cancer development. We observe a degree of consistency between varying the
amounts of human and mouse data used, however, further work is required to
obtain conclusive insights. As a secondary contribution, we enhance existing
open source datasets and provide two uniformly processed, cross-organism,
homologue gene-matched datasets to the community.

    

### [[2111.07355] Fracture Detection in Wrist X-ray Images Using Deep Learning-Based Object Detection Models](http://arxiv.org/abs/2111.07355)


  Wrist fractures are common cases in hospitals, particularly in emergency
services. Physicians need images from various medical devices, and patients
medical history and physical examination to diagnose these fractures correctly
and apply proper treatment. This study aims to perform fracture detection using
deep learning on wrist Xray images to assist physicians not specialized in the
field, working in emergency services in particular, in diagnosis of fractures.
For this purpose, 20 different detection procedures were performed using deep
learning based object detection models on dataset of wrist Xray images obtained
from Gazi University Hospital. DCN, Dynamic R_CNN, Faster R_CNN, FSAF, Libra
R_CNN, PAA, RetinaNet, RegNet and SABL deep learning based object detection
models with various backbones were used herein. To further improve detection
procedures in the study, 5 different ensemble models were developed, which were
later used to reform an ensemble model to develop a detection model unique to
our study, titled wrist fracture detection combo (WFD_C). Based on detection of
26 different fractures in total, the highest result of detection was 0.8639
average precision (AP50) in WFD_C model developed. This study is supported by
Huawei Turkey R&D Center within the scope of the ongoing cooperation project
coded 071813 among Gazi University, Huawei and Medskor.

    

### [[2111.07369] Estimation of Acetabular Version from Anteroposterior Pelvic Radiograph Employing Deep Learning](http://arxiv.org/abs/2111.07369)


  Background and Objective: The Acetabular version, an essential factor in
total hip arthroplasty, is measured by CT scan as the gold standard. The dose
of radiation and expensiveness of CT make anterior-posterior pelvic radiograph
an appropriate alternative procedure. In this study, we applied a deep learning
approach on anteroposterior pelvic X-rays to measure anatomical version,
eliminating the necessity of using Computed tomography scan. Methods: The right
and left acetabular version angles of the hips of 300 patients are computed
using their CT images. The proposed deep learning model, Attention on
Pretrained-VGG16 for Bone Age, is applied to the AP images of the included
population. The age and gender of these people are added as two other inputs to
the last fully connected layer of attention mechanism. As the output, the
angles of both hips are predicted. Results: The angles of hips computed on CT
increase as people get older with the mean values of 16.54 and 16.11 (right and
left angles) for men and 20.61 and 19.55 for women in our dataset. The
predicted errors in the estimation of right and left angles using the proposed
method of deep learning are in the accurate region of error (<=3 degrees) which
shows the ability of the proposed method in measuring anatomical version based
on AP images. Conclusion: The suggested algorithm, applying pre-trained vgg16
on the AP images of the pelvis of patients followed by an attention model
considering age and gender of patients, can assess version accurately using
only AP radiographs while obviating the need for CT scan. The applied technique
of estimation of anatomical acetabular version based on AP pelvic images using
DL approaches, to the best of authors' knowledge, has not been published yet.

    

### [[2111.07376] On equivalence between linear-chain conditional random fields and hidden Markov chains](http://arxiv.org/abs/2111.07376)


  Practitioners successfully use hidden Markov chains (HMCs) in different
problems for about sixty years. HMCs belong to the family of generative models
and they are often compared to discriminative models, like conditional random
fields (CRFs). Authors usually consider CRFs as quite different from HMCs, and
CRFs are often presented as interesting alternative to HMCs. In some areas,
like natural language processing (NLP), discriminative models have completely
supplanted generative models. However, some recent results show that both
families of models are not so different, and both of them can lead to identical
processing power. In this paper we compare the simple linear-chain CRFs to the
basic HMCs. We show that HMCs are identical to CRFs in that for each CRF we
explicitly construct an HMC having the same posterior distribution. Therefore,
HMCs and linear-chain CRFs are not different but just differently parametrized
models.

    

### [[2111.07378] TEA: A Sequential Recommendation Framework via Temporally Evolving Aggregations](http://arxiv.org/abs/2111.07378)


  Sequential recommendation aims to choose the most suitable items for a user
at a specific timestamp given historical behaviors. Existing methods usually
model the user behavior sequence based on the transition-based methods like
Markov Chain. However, these methods also implicitly assume that the users are
independent of each other without considering the influence between users. In
fact, this influence plays an important role in sequence recommendation since
the behavior of a user is easily affected by others. Therefore, it is desirable
to aggregate both user behaviors and the influence between users, which are
evolved temporally and involved in the heterogeneous graph of users and items.
In this paper, we incorporate dynamic user-item heterogeneous graphs to propose
a novel sequential recommendation framework. As a result, the historical
behaviors as well as the influence between users can be taken into
consideration. To achieve this, we firstly formalize sequential recommendation
as a problem to estimate conditional probability given temporal dynamic
heterogeneous graphs and user behavior sequences. After that, we exploit the
conditional random field to aggregate the heterogeneous graphs and user
behaviors for probability estimation, and employ the pseudo-likelihood approach
to derive a tractable objective function. Finally, we provide scalable and
flexible implementations of the proposed framework. Experimental results on
three real-world datasets not only demonstrate the effectiveness of our
proposed method but also provide some insightful discoveries on sequential
recommendation.

    

### [[2111.07379] A Robust Unsupervised Ensemble of Feature-Based Explanations using Restricted Boltzmann Machines](http://arxiv.org/abs/2111.07379)


  Understanding the results of deep neural networks is an essential step
towards wider acceptance of deep learning algorithms. Many approaches address
the issue of interpreting artificial neural networks, but often provide
divergent explanations. Moreover, different hyperparameters of an explanatory
method can lead to conflicting interpretations. In this paper, we propose a
technique for aggregating the feature attributions of different explanatory
algorithms using Restricted Boltzmann Machines (RBMs) to achieve a more
reliable and robust interpretation of deep neural networks. Several challenging
experiments on real-world datasets show that the proposed RBM method
outperforms popular feature attribution methods and basic ensemble techniques.

    

### [[2111.07380] Eluding Secure Aggregation in Federated Learning via Model Inconsistency](http://arxiv.org/abs/2111.07380)


  Federated learning allows a set of users to train a deep neural network over
their private training datasets. During the protocol, datasets never leave the
devices of the respective users. This is achieved by requiring each user to
send "only" model updates to a central server that, in turn, aggregates them to
update the parameters of the deep neural network. However, it has been shown
that each model update carries sensitive information about the user's dataset
(e.g., gradient inversion attacks).
The state-of-the-art implementations of federated learning protect these
model updates by leveraging secure aggregation: A cryptographic protocol that
securely computes the aggregation of the model updates of the users. Secure
aggregation is pivotal to protect users' privacy since it hinders the server
from learning the value and the source of the individual model updates provided
by the users, preventing inference and data attribution attacks.
In this work, we show that a malicious server can easily elude secure
aggregation as if the latter were not in place. We devise two different attacks
capable of inferring information on individual private training datasets,
independently of the number of users participating in the secure aggregation.
This makes them concrete threats in large-scale, real-world federated learning
applications.
The attacks are generic and do not target any specific secure aggregation
protocol. They are equally effective even if the secure aggregation protocol is
replaced by its ideal functionality that provides the perfect level of
security. Our work demonstrates that secure aggregation has been incorrectly
combined with federated learning and that current implementations offer only a
"false sense of security".

    

### [[2111.07382] Adaptive Cost-Sensitive Learning in Neural Networks for Misclassification Cost Problems](http://arxiv.org/abs/2111.07382)


  We design a new adaptive learning algorithm for misclassification cost
problems that attempt to reduce the cost of misclassified instances derived
from the consequences of various errors. Our algorithm (adaptive cost sensitive
learning - AdaCSL) adaptively adjusts the loss function such that the
classifier bridges the difference between the class distributions between
subgroups of samples in the training and test data sets with similar predicted
probabilities (i.e., local training-test class distribution mismatch). We
provide some theoretical performance guarantees on the proposed algorithm and
present empirical evidence that a deep neural network used with the proposed
AdaCSL algorithm yields better cost results on several binary classification
data sets that have class-imbalanced and class-balanced distributions compared
to other alternative approaches.

    

### [[2111.07386] Interpretable ECG classification via a query-based latent space traversal (qLST)](http://arxiv.org/abs/2111.07386)


  Electrocardiography (ECG) is an effective and non-invasive diagnostic tool
that measures the electrical activity of the heart. Interpretation of ECG
signals to detect various abnormalities is a challenging task that requires
expertise. Recently, the use of deep neural networks for ECG classification to
aid medical practitioners has become popular, but their black box nature
hampers clinical implementation. Several saliency-based interpretability
techniques have been proposed, but they only indicate the location of important
features and not the actual features. We present a novel interpretability
technique called qLST, a query-based latent space traversal technique that is
able to provide explanations for any ECG classification model. With qLST, we
train a neural network that learns to traverse in the latent space of a
variational autoencoder trained on a large university hospital dataset with
over 800,000 ECGs annotated for 28 diseases. We demonstrate through experiments
that we can explain different black box classifiers by generating ECGs through
these traversals.

    

### [[2111.07395] Explicit Explore, Exploit, or Escape ($E^4$): near-optimal safety-constrained reinforcement learning in polynomial time](http://arxiv.org/abs/2111.07395)


  In reinforcement learning (RL), an agent must explore an initially unknown
environment in order to learn a desired behaviour. When RL agents are deployed
in real world environments, safety is of primary concern. Constrained Markov
decision processes (CMDPs) can provide long-term safety constraints; however,
the agent may violate the constraints in an effort to explore its environment.
This paper proposes a model-based RL algorithm called Explicit Explore,
Exploit, or Escape ($E^{4}$), which extends the Explicit Explore or Exploit
($E^{3}$) algorithm to a robust CMDP setting. $E^4$ explicitly separates
exploitation, exploration, and escape CMDPs, allowing targeted policies for
policy improvement across known states, discovery of unknown states, as well as
safe return to known states. $E^4$ robustly optimises these policies on the
worst-case CMDP from a set of CMDP models consistent with the empirical
observations of the deployment environment. Theoretical results show that $E^4$
finds a near-optimal constraint-satisfying policy in polynomial time whilst
satisfying safety constraints throughout the learning process. We discuss
robust-constrained offline optimisation algorithms as well as how to
incorporate uncertainty in transition dynamics of unknown states based on
empirical inference and prior knowledge.

    

### [[2111.07401] Neural Capacity Estimators: How Reliable Are They?](http://arxiv.org/abs/2111.07401)


  Recently, several methods have been proposed for estimating the mutual
information from sample data using deep neural networks and without the knowing
closed form distribution of the data. This class of estimators is referred to
as neural mutual information estimators. Although very promising, such
techniques have yet to be rigorously bench-marked so as to establish their
efficacy, ease of implementation, and stability for capacity estimation which
is joint maximization frame-work. In this paper, we compare the different
techniques proposed in the literature for estimating capacity and provide a
practitioner perspective on their effectiveness. In particular, we study the
performance of mutual information neural estimator (MINE), smoothed mutual
information lower-bound estimator (SMILE), and directed information neural
estimator (DINE) and provide insights on InfoNCE. We evaluated these algorithms
in terms of their ability to learn the input distributions that are capacity
approaching for the AWGN channel, the optical intensity channel, and peak
power-constrained AWGN channel. For both scenarios, we provide insightful
comments on various aspects of the training process, such as stability,
sensitivity to initialization.

    

### [[2111.07402] Textless Speech Emotion Conversion using Decomposed and Discrete Representations](http://arxiv.org/abs/2111.07402)


  Speech emotion conversion is the task of modifying the perceived emotion of a
speech utterance while preserving the lexical content and speaker identity. In
this study, we cast the problem of emotion conversion as a spoken language
translation task. We decompose speech into discrete and disentangled learned
representations, consisting of content units, F0, speaker, and emotion. First,
we modify the speech content by translating the content units to a target
emotion, and then predict the prosodic features based on these units. Finally,
the speech waveform is generated by feeding the predicted representations into
a neural vocoder. Such a paradigm allows us to go beyond spectral and
parametric changes of the signal, and model non-verbal vocalizations, such as
laughter insertion, yawning removal, etc. We demonstrate objectively and
subjectively that the proposed method is superior to the baselines in terms of
perceived emotion and audio quality. We rigorously evaluate all components of
such a complex system and conclude with an extensive model analysis and
ablation study to better emphasize the architectural choices, strengths and
weaknesses of the proposed method. Samples and code will be publicly available
under the following link: this https URL.

    

### [[2111.07407] A Machine Learning Approach for Recruitment Prediction in Clinical Trial Design](http://arxiv.org/abs/2111.07407)


  Significant advancements have been made in recent years to optimize patient
recruitment for clinical trials, however, improved methods for patient
recruitment prediction are needed to support trial site selection and to
estimate appropriate enrollment timelines in the trial design stage. In this
paper, using data from thousands of historical clinical trials, we explore
machine learning methods to predict the number of patients enrolled per month
at a clinical trial site over the course of a trial's enrollment duration. We
show that these methods can reduce the error that is observed with current
industry standards and propose opportunities for further improvement.

    

### [[2111.07419] Learning a Shared Model for Motorized Prosthetic Joints to Predict Ankle-Joint Motion](http://arxiv.org/abs/2111.07419)


  Control strategies for active prostheses or orthoses use sensor inputs to
recognize the user's locomotive intention and generate corresponding control
commands for producing the desired locomotion. In this paper, we propose a
learning-based shared model for predicting ankle-joint motion for different
locomotion modes like level-ground walking, stair ascent, stair descent, slope
ascent, and slope descent without the need to classify between them. Features
extracted from hip and knee joint angular motion are used to continuously
predict the ankle angles and moments using a Feed-Forward Neural Network-based
shared model. We show that the shared model is adequate for predicting the
ankle angles and moments for different locomotion modes without explicitly
classifying between the modes. The proposed strategy shows the potential for
devising a high-level controller for an intelligent prosthetic ankle that can
adapt to different locomotion modes.

    

### [[2111.07424] Generating Band-Limited Adversarial Surfaces Using Neural Networks](http://arxiv.org/abs/2111.07424)


  Generating adversarial examples is the art of creating a noise that is added
to an input signal of a classifying neural network, and thus changing the
network's classification, while keeping the noise as tenuous as possible. While
the subject is well-researched in the 2D regime, it is lagging behind in the 3D
regime, i.e. attacking a classifying network that works on 3D point-clouds or
meshes and, for example, classifies the pose of people's 3D scans. As of now,
the vast majority of papers that describe adversarial attacks in this regime
work by methods of optimization. In this technical report we suggest a neural
network that generates the attacks. This network utilizes PointNet's
architecture with some alterations. While the previous articles on which we
based our work on have to optimize each shape separately, i.e. tailor an attack
from scratch for each individual input without any learning, we attempt to
create a unified model that can deduce the needed adversarial example with a
single forward run.

    

### [[2111.07426] Unsupervised Action Localization Crop in Video Retargeting for 3D ConvNets](http://arxiv.org/abs/2111.07426)


  Untrimmed videos on social media or those captured by robots and surveillance
cameras are of varied aspect ratios. However, 3D CNNs require a square-shaped
video whose spatial dimension is smaller than the original one. Random or
center-cropping techniques in use may leave out the video's subject altogether.
To address this, we propose an unsupervised video cropping approach by shaping
this as a retargeting and video-to-video synthesis problem. The synthesized
video maintains 1:1 aspect ratio, smaller in size and is targeted at the
video-subject throughout the whole duration. First, action localization on the
individual frames is performed by identifying patches with homogeneous motion
patterns and a single salient patch is pin-pointed. To avoid viewpoint jitters
and flickering artifacts, any inter-frame scale or position changes among the
patches is performed gradually over time. This issue is addressed with a
poly-Bezier fitting in 3D space that passes through some chosen pivot
timestamps and its shape is influenced by in-between control timestamps. To
corroborate the effectiveness of the proposed method, we evaluate the video
classification task by comparing our dynamic cropping with static random on
three benchmark datasets: UCF-101, HMDB-51 and ActivityNet v1.3. The clip
accuracy and top-1 accuracy for video classification after our cropping,
outperform 3D CNN performances for same-sized inputs with random crop;
sometimes even surpassing larger random crop sizes.

    

### [[2111.07430] Safe Online Convex Optimization with Unknown Linear Safety Constraints](http://arxiv.org/abs/2111.07430)


  We study the problem of safe online convex optimization, where the action at
each time step must satisfy a set of linear safety constraints. The goal is to
select a sequence of actions to minimize the regret without violating the
safety constraints at any time step (with high probability). The parameters
that specify the linear safety constraints are unknown to the algorithm. The
algorithm has access to only the noisy observations of constraints for the
chosen actions. We propose an algorithm, called the {Safe Online Projected
Gradient Descent} (SO-PGD) algorithm, to address this problem. We show that,
under the assumption of the availability of a safe baseline action, the SO-PGD
algorithm achieves a regret $O(T^{2/3})$. While there are many algorithms for
online convex optimization (OCO) problems with safety constraints available in
the literature, they allow constraint violations during learning/optimization,
and the focus has been on characterizing the cumulative constraint violations.
To the best of our knowledge, ours is the first work that provides an algorithm
with provable guarantees on the regret, without violating the linear safety
constraints (with high probability) at any time step.

    

### [[2111.07439] Improving Compound Activity Classification via Deep Transfer and Representation Learning](http://arxiv.org/abs/2111.07439)


  Recent advances in molecular machine learning, especially deep neural
networks such as Graph Neural Networks (GNNs) for predicting structure activity
relationships (SAR) have shown tremendous potential in computer-aided drug
discovery. However, the applicability of such deep neural networks are limited
by the requirement of large amounts of training data. In order to cope with
limited training data for a target task, transfer learning for SAR modeling has
been recently adopted to leverage information from data of related tasks. In
this work, in contrast to the popular parameter-based transfer learning such as
pretraining, we develop novel deep transfer learning methods TAc and TAc-fc to
leverage source domain data and transfer useful information to the target
domain. TAc learns to generate effective molecular features that can generalize
well from one domain to another, and increase the classification performance in
the target domain. Additionally, TAc-fc extends TAc by incorporating novel
components to selectively learn feature-wise and compound-wise transferability.
We used the bioassay screening data from PubChem, and identified 120 pairs of
bioassays such that the active compounds in each pair are more similar to each
other compared to its inactive compounds. Overall, TAc achieves the best
performance with average ROC-AUC of 0.801; it significantly improves ROC-AUC of
83% target tasks with average task-wise performance improvement of 7.102%,
compared to the best baseline FCN-dmpna (DT). Our experiments clearly
demonstrate that TAc achieves significant improvement over all baselines across
a large number of target tasks. Furthermore, although TAc-fc achieves slightly
worse ROC-AUC on average compared to TAc (0.798 vs 0.801), TAc-fc still
achieves the best performance on more tasks in terms of PR-AUC and F1 compared
to other methods.

    

### [[2111.07447] Learning Multi-Stage Tasks with One Demonstration via Self-Replay](http://arxiv.org/abs/2111.07447)


  In this work, we introduce a novel method to learn everyday-like multi-stage
tasks from a single human demonstration, without requiring any prior object
knowledge. Inspired by the recent Coarse-to-Fine Imitation Learning method, we
model imitation learning as a learned object reaching phase followed by an
open-loop replay of the demonstrator's actions. We build upon this for
multi-stage tasks where, following the human demonstration, the robot can
autonomously collect image data for the entire multi-stage task, by reaching
the next object in the sequence and then replaying the demonstration, and then
repeating in a loop for all stages of the task. We evaluate with real-world
experiments on a set of everyday-like multi-stage tasks, which we show that our
method can solve from a single demonstration. Videos and supplementary material
can be found at this https URL.

    

### [[2111.07455] HAD-Net: Hybrid Attention-based Diffusion Network for Glucose Level Forecast](http://arxiv.org/abs/2111.07455)


  Data-driven models for glucose level forecast often do not provide meaningful
insights despite accurate predictions. Yet, context understanding in medicine
is crucial, in particular for diabetes management. In this paper, we introduce
HAD-Net: a hybrid model that distills knowledge into a deep neural network from
physiological models. It models glucose, insulin and carbohydrates diffusion
through a biologically inspired deep learning architecture tailored with a
recurrent attention network constrained by ODE expert models. We apply HAD-Net
for glucose level forecast of patients with type-2 diabetes. It achieves
competitive performances while providing plausible measurements of insulin and
carbohydrates diffusion over time.

    

### [[2111.07458] Mean-based Best Arm Identification in Stochastic Bandits under Reward Contamination](http://arxiv.org/abs/2111.07458)


  This paper investigates the problem of best arm identification in
$\textit{contaminated}$ stochastic multi-arm bandits. In this setting, the
rewards obtained from any arm are replaced by samples from an adversarial model
with probability $\varepsilon$. A fixed confidence (infinite-horizon) setting
is considered, where the goal of the learner is to identify the arm with the
largest mean. Owing to the adversarial contamination of the rewards, each arm's
mean is only partially identifiable. This paper proposes two algorithms, a
gap-based algorithm and one based on the successive elimination, for best arm
identification in sub-Gaussian bandits. These algorithms involve mean estimates
that achieve the optimal error guarantee on the deviation of the true mean from
the estimate asymptotically. Furthermore, these algorithms asymptotically
achieve the optimal sample complexity. Specifically, for the gap-based
algorithm, the sample complexity is asymptotically optimal up to constant
factors, while for the successive elimination-based algorithm, it is optimal up
to logarithmic factors. Finally, numerical experiments are provided to
illustrate the gains of the algorithms compared to the existing baselines.

    

### [[2111.07462] Federated Learning with Hyperparameter-based Clustering for Electrical Load Forecasting](http://arxiv.org/abs/2111.07462)


  Electrical load prediction has become an integral part of power system
operation. Deep learning models have found popularity for this purpose.
However, to achieve a desired prediction accuracy, they require huge amounts of
data for training. Sharing electricity consumption data of individual
households for load prediction may compromise user privacy and can be expensive
in terms of communication resources. Therefore, edge computing methods, such as
federated learning, are gaining more importance for this purpose. These methods
can take advantage of the data without centrally storing it. This paper
evaluates the performance of federated learning for short-term forecasting of
individual house loads as well as the aggregate load. It discusses the
advantages and disadvantages of this method by comparing it to centralized and
local learning schemes. Moreover, a new client clustering method is proposed to
reduce the convergence time of federated learning. The results show that
federated learning has a good performance with a minimum root mean squared
error (RMSE) of 0.117kWh for individual load forecasting.

    

### [[2111.07465] Decoding Causality by Fictitious VAR Modeling](http://arxiv.org/abs/2111.07465)


  In modeling multivariate time series for either forecast or policy analysis,
it would be beneficial to have figured out the cause-effect relations within
the data. Regression analysis, however, is generally for correlation relation,
and very few researches have focused on variance analysis for causality
discovery. We first set up an equilibrium for the cause-effect relations using
a fictitious vector autoregressive model. In the equilibrium, long-run
relations are identified from noise, and spurious ones are negligibly close to
zero. The solution, called causality distribution, measures the relative
strength causing the movement of all series or specific affected ones. If a
group of exogenous data affects the others but not vice versa, then, in theory,
the causality distribution for other variables is necessarily zero. The
hypothesis test of zero causality is the rule to decide a variable is
endogenous or not. Our new approach has high accuracy in identifying the true
cause-effect relations among the data in the simulation studies. We also apply
the approach to estimating the causal factors' contribution to climate change.

    

### [[2111.07470] Skillful Twelve Hour Precipitation Forecasts using Large Context Neural Networks](http://arxiv.org/abs/2111.07470)


  The problem of forecasting weather has been scientifically studied for
centuries due to its high impact on human lives, transportation, food
production and energy management, among others. Current operational forecasting
models are based on physics and use supercomputers to simulate the atmosphere
to make forecasts hours and days in advance. Better physics-based forecasts
require improvements in the models themselves, which can be a substantial
scientific challenge, as well as improvements in the underlying resolution,
which can be computationally prohibitive. An emerging class of weather models
based on neural networks represents a paradigm shift in weather forecasting:
the models learn the required transformations from data instead of relying on
hand-coded physics and are computationally efficient. For neural models,
however, each additional hour of lead time poses a substantial challenge as it
requires capturing ever larger spatial contexts and increases the uncertainty
of the prediction. In this work, we present a neural network that is capable of
large-scale precipitation forecasting up to twelve hours ahead and, starting
from the same atmospheric state, the model achieves greater skill than the
state-of-the-art physics-based models HRRR and HREF that currently operate in
the Continental United States. Interpretability analyses reinforce the
observation that the model learns to emulate advanced physics principles. These
results represent a substantial step towards establishing a new paradigm of
efficient forecasting with neural networks.

    

### [[2111.07473] Scrutinizing XAI using linear ground-truth data with suppressor variables](http://arxiv.org/abs/2111.07473)


  Machine learning (ML) is increasingly often used to inform high-stakes
decisions. As complex ML models (e.g., deep neural networks) are often
considered black boxes, a wealth of procedures has been developed to shed light
on their inner workings and the ways in which their predictions come about,
defining the field of 'explainable AI' (XAI). Saliency methods rank input
features according to some measure of 'importance'. Such methods are difficult
to validate since a formal definition of feature importance is, thus far,
lacking. It has been demonstrated that some saliency methods can highlight
features that have no statistical association with the prediction target
(suppressor variables). To avoid misinterpretations due to such behavior, we
propose the actual presence of such an association as a necessary condition and
objective preliminary definition for feature importance. We carefully crafted a
ground-truth dataset in which all statistical dependencies are well-defined and
linear, serving as a benchmark to study the problem of suppressor variables. We
evaluate common explanation methods including LRP, DTD, PatternNet,
PatternAttribution, LIME, Anchors, SHAP, and permutation-based methods with
respect to our objective definition. We show that most of these methods are
unable to distinguish important features from suppressors in this setting.

    

### [[2111.07478] Physics in the Machine: Integrating Physical Knowledge in Autonomous Phase-Mapping](http://arxiv.org/abs/2111.07478)


  Application of artificial intelligence (AI), and more specifically machine
learning, to the physical sciences has expanded significantly over the past
decades. In particular, science-informed AI or scientific AI has grown from a
focus on data analysis to now controlling experiment design, simulation,
execution and analysis in closed-loop autonomous systems. The CAMEO
(closed-loop autonomous materials exploration and optimization) algorithm
employs scientific AI to address two tasks: learning a material system's
composition-structure relationship and identifying materials compositions with
optimal functional properties. By integrating these, accelerated materials
screening across compositional phase diagrams was demonstrated, resulting in
the discovery of a best-in-class phase change memory material. Key to this
success is the ability to guide subsequent measurements to maximize knowledge
of the composition-structure relationship, or phase map. In this work we
investigate the benefits of incorporating varying levels of prior physical
knowledge into CAMEO's autonomous phase-mapping. This includes the use of
ab-initio phase boundary data from the AFLOW repositories, which has been shown
to optimize CAMEO's search when used as a prior.

    

### [[2111.07489] Deep Learning based Urban Vehicle Trajectory Analytics](http://arxiv.org/abs/2111.07489)


  A `trajectory' refers to a trace generated by a moving object in geographical
spaces, usually represented by of a series of chronologically ordered points,
where each point consists of a geo-spatial coordinate set and a timestamp.
Rapid advancements in location sensing and wireless communication technology
enabled us to collect and store a massive amount of trajectory data. As a
result, many researchers use trajectory data to analyze mobility of various
moving objects. In this dissertation, we focus on the `urban vehicle
trajectory,' which refers to trajectories of vehicles in urban traffic
networks, and we focus on `urban vehicle trajectory analytics.' The urban
vehicle trajectory analytics offers unprecedented opportunities to understand
vehicle movement patterns in urban traffic networks including both user-centric
travel experiences and system-wide spatiotemporal patterns. The spatiotemporal
features of urban vehicle trajectory data are structurally correlated with each
other, and consequently, many previous researchers used various methods to
understand this structure. Especially, deep-learning models are getting
attentions of many researchers due to its powerful function approximation and
feature representation abilities. As a result, the objective of this
dissertation is to develop deep-learning based models for urban vehicle
trajectory analytics to better understand the mobility patterns of urban
traffic networks. Particularly, this dissertation focuses on two research
topics, which has high necessity, importance and applicability: Next Location
Prediction, and Synthetic Trajectory Generation. In this study, we propose
various novel models for urban vehicle trajectory analytics using deep
learning.

    

### [[2111.07490] Deep-Learning Inversion Method for the Interpretation of Noisy Logging-While-Drilling Resistivity Measurements](http://arxiv.org/abs/2111.07490)


  Deep Learning (DL) inversion is a promising method for real time
interpretation of logging while drilling (LWD) resistivity measurements for
well navigation applications. In this context, measurement noise may
significantly affect inversion results. Existing publications examining the
effects of measurement noise on DL inversion results are scarce. We develop a
method to generate training data sets and construct DL architectures that
enhance the robustness of DL inversion methods in the presence of noisy LWD
resistivity measurements. We use two synthetic resistivity models to test three
approaches that explicitly consider the presence of noise: (1) adding noise to
the measurements in the training set, (2) augmenting the training set by
replicating it and adding varying noise realizations, and (3) adding a noise
layer in the DL architecture. Numerical results confirm that the three
approaches produce a denoising effect, yielding better inversion results in
both predicted earth model and measurements compared not only to the basic DL
inversion but also to traditional gradient based inversion results. A
combination of the second and third approaches delivers the best results. The
proposed methods can be readily generalized to multi dimensional DL inversion.

    

### [[2111.07494] Federated Learning for Internet of Things: Applications, Challenges, and Opportunities](http://arxiv.org/abs/2111.07494)


  Billions of IoT devices will be deployed in the near future, taking advantage
of the faster Internet speed and the possibility of orders of magnitude more
endpoints brought by 5G/6G. With the blooming of IoT devices, vast quantities
of data that may contain private information of users will be generated. The
high communication and storage costs, mixed with privacy concerns, will
increasingly be challenging the traditional ecosystem of centralized
over-the-cloud learning and processing for IoT platforms. Federated Learning
(FL) has emerged as the most promising alternative approach to this problem. In
FL, training of data-driven machine learning models is an act of collaboration
between multiple clients without requiring the data to be brought to a central
point, hence alleviating communication and storage costs and providing a great
degree of user-level privacy. We discuss the opportunities and challenges of FL
for IoT platforms, as well as how it can enable future IoT applications.

    

### [[2111.07495] Distribution-Free Models for Community Detection](http://arxiv.org/abs/2111.07495)


  Community detection for un-weighted networks has been widely studied in
network analysis, but the case of weighted networks remains a challenge. In
this paper, a Distribution-Free Models (DFM) is proposed for networks in which
nodes are partitioned into different communities. DFM is a general,
interpretable and identifiable model for both un-weighted networks and weighted
networks. The proposed model does not require prior knowledge on a specific
distribution for elements of adjacency matrix but only the expected value. The
distribution-free property of DFM even allows adjacency matrix to have negative
elements. We develop an efficient spectral algorithm to fit DFM. By introducing
a noise matrix, we build a theoretic framework on perturbation analysis to show
that the proposed algorithm stably yields consistent community detection under
DFM. Numerical experiments on both synthetic networks and two social networks
from literature are used to illustrate the algorithm.

    

### [[2111.07503] Measuring Outcomes in Healthcare Economics using Artificial Intelligence: with Application to Resource Management](http://arxiv.org/abs/2111.07503)


  The quality of service in healthcare is constantly challenged by outlier
events such as pandemics (i.e. Covid-19) and natural disasters (such as
hurricanes and earthquakes). In most cases, such events lead to critical
uncertainties in decision making, as well as in multiple medical and economic
aspects at a hospital. External (geographic) or internal factors (medical and
managerial), lead to shifts in planning and budgeting, but most importantly,
reduces confidence in conventional processes. In some cases, support from other
hospitals proves necessary, which exacerbates the planning aspect. This
manuscript presents three data-driven methods that provide data-driven
indicators to help healthcare managers organize their economics and identify
the most optimum plan for resources allocation and sharing. Conventional
decision-making methods fall short in recommending validated policies for
managers. Using reinforcement learning, genetic algorithms, traveling salesman,
and clustering, we experimented with different healthcare variables and
presented tools and outcomes that could be applied at health institutes.
Experiments are performed; the results are recorded, evaluated, and presented.

    

### [[2111.07508] Public Policymaking for International Agricultural Trade using Association Rules and Ensemble Machine Learning](http://arxiv.org/abs/2111.07508)


  International economics has a long history of improving our understanding of
factors causing trade, and the consequences of free flow of goods and services
across countries. The recent shocks to the free trade regime, especially trade
disputes among major economies, as well as black swan events, such as trade
wars and pandemics, raise the need for improved predictions to inform policy
decisions. AI methods are allowing economists to solve such prediction problems
in new ways. In this manuscript, we present novel methods that predict and
associate food and agricultural commodities traded internationally. Association
Rules (AR) analysis has been deployed successfully for economic scenarios at
the consumer or store level, such as for market basket analysis. In our work
however, we present analysis of imports and exports associations and their
effects on commodity trade flows. Moreover, Ensemble Machine Learning methods
are developed to provide improved agricultural trade predictions, outlier
events' implications, and quantitative pointers to policy makers.

    

### [[2111.07512] Scalable Intervention Target Estimation in Linear Models](http://arxiv.org/abs/2111.07512)


  This paper considers the problem of estimating the unknown intervention
targets in a causal directed acyclic graph from observational and
interventional data. The focus is on soft interventions in linear structural
equation models (SEMs). Current approaches to causal structure learning either
work with known intervention targets or use hypothesis testing to discover the
unknown intervention targets even for linear SEMs. This severely limits their
scalability and sample complexity. This paper proposes a scalable and efficient
algorithm that consistently identifies all intervention targets. The pivotal
idea is to estimate the intervention sites from the difference between the
precision matrices associated with the observational and interventional
datasets. It involves repeatedly estimating such sites in different subsets of
variables. The proposed algorithm can be used to also update a given
observational Markov equivalence class into the interventional Markov
equivalence class. Consistency, Markov equivalency, and sample complexity are
established analytically. Finally, simulation results on both real and
synthetic data demonstrate the gains of the proposed approach for scalable
causal structure recovery. Implementation of the algorithm and the code to
reproduce the simulation results are available at
\url{this https URL}.

    

### [[2111.07513] A Comparative Study on Basic Elements of Deep Learning Models for Spatial-Temporal Traffic Forecasting](http://arxiv.org/abs/2111.07513)


  Traffic forecasting plays a crucial role in intelligent transportation
systems. The spatial-temporal complexities in transportation networks make the
problem especially challenging. The recently suggested deep learning models
share basic elements such as graph convolution, graph attention, recurrent
units, and/or attention mechanism. In this study, we designed an in-depth
comparative study for four deep neural network models utilizing different basic
elements. For base models, one RNN-based model and one attention-based model
were chosen from previous literature. Then, the spatial feature extraction
layers in the models were substituted with graph convolution and graph
attention. To analyze the performance of each element in various environments,
we conducted experiments on four real-world datasets - highway speed, highway
flow, urban speed from a homogeneous road link network, and urban speed from a
heterogeneous road link network. The results demonstrate that the RNN-based
model and the attention-based model show a similar level of performance for
short-term prediction, and the attention-based model outperforms the RNN in
longer-term predictions. The choice of graph convolution and graph attention
makes a larger difference in the RNN-based models. Also, our modified version
of GMAN shows comparable performance with the original with less memory
consumption.

    

### [[2111.07535] T-AutoML: Automated Machine Learning for Lesion Segmentation using Transformers in 3D Medical Imaging](http://arxiv.org/abs/2111.07535)


  Lesion segmentation in medical imaging has been an important topic in
clinical research. Researchers have proposed various detection and segmentation
algorithms to address this task. Recently, deep learning-based approaches have
significantly improved the performance over conventional methods. However, most
state-of-the-art deep learning methods require the manual design of multiple
network components and training strategies. In this paper, we propose a new
automated machine learning algorithm, T-AutoML, which not only searches for the
best neural architecture, but also finds the best combination of
hyper-parameters and data augmentation strategies simultaneously. The proposed
method utilizes the modern transformer model, which is introduced to adapt to
the dynamic length of the search space embedding and can significantly improve
the ability of the search. We validate T-AutoML on several large-scale public
lesion segmentation data-sets and achieve state-of-the-art performance.

    

### [[2111.07545] Randomized Classifiers vs Human Decision-Makers: Trustworthy AI May Have to Act Randomly and Society Seems to Accept This](http://arxiv.org/abs/2111.07545)


  As \emph{artificial intelligence} (AI) systems are increasingly involved in
decisions affecting our lives, ensuring that automated decision-making is fair
and ethical has become a top priority. Intuitively, we feel that akin to human
decisions, judgments of artificial agents should necessarily be grounded in
some moral principles. Yet a decision-maker (whether human or artificial) can
only make truly ethical (based on any ethical theory) and fair (according to
any notion of fairness) decisions if full information on all the relevant
factors on which the decision is based are available at the time of
decision-making. This raises two problems: (1) In settings, where we rely on AI
systems that are using classifiers obtained with supervised learning, some
induction/generalization is present and some relevant attributes may not be
present even during learning. (2) Modeling such decisions as games reveals that
any -- however ethical -- pure strategy is inevitably susceptible to
exploitation.
Moreover, in many games, a Nash Equilibrium can only be obtained by using
mixed strategies, i.e., to achieve mathematically optimal outcomes, decisions
must be randomized. In this paper, we argue that in supervised learning
settings, there exist random classifiers that perform at least as well as
deterministic classifiers, and may hence be the optimal choice in many
circumstances. We support our theoretical results with an empirical study
indicating a positive societal attitude towards randomized artificial
decision-makers, and discuss some policy and implementation issues related to
the use of random classifiers that relate to and are relevant for current AI
policy and standardization initiatives.

    

### [[2111.07564] Adding more data does not always help: A study in medical conversation summarization with PEGASUS](http://arxiv.org/abs/2111.07564)


  Medical conversation summarization is integral in capturing information
gathered during interactions between patients and physicians. Summarized
conversations are used to facilitate patient hand-offs between physicians, and
as part of providing care in the future. Summaries, however, can be
time-consuming to produce and require domain expertise. Modern pre-trained NLP
models such as PEGASUS have emerged as capable alternatives to human
summarization, reaching state-of-the-art performance on many summarization
benchmarks. However, many downstream tasks still require at least moderately
sized datasets to achieve satisfactory performance. In this work we (1) explore
the effect of dataset size on transfer learning medical conversation
summarization using PEGASUS and (2) evaluate various iterative labeling
strategies in the low-data regime, following their success in the
classification setting. We find that model performance saturates with increase
in dataset size and that the various active-learning strategies evaluated all
show equivalent performance consistent with simple dataset size increase. We
also find that naive iterative pseudo-labeling is on-par or slightly worse than
no pseudo-labeling. Our work sheds light on the successes and challenges of
translating low-data regime techniques in classification to medical
conversation summarization and helps guides future work in this space. Relevant
code available at
\url{this https URL}.

    

### [[2111.07592] Say What? Collaborative Pop Lyric Generation Using Multitask Transfer Learning](http://arxiv.org/abs/2111.07592)


  Lyric generation is a popular sub-field of natural language generation that
has seen growth in recent years. Pop lyrics are of unique interest due to the
genre's unique style and content, in addition to the high level of
collaboration that goes on behind the scenes in the professional pop
songwriting process. In this paper, we present a collaborative line-level lyric
generation system that utilizes transfer-learning via the T5 transformer model,
which, till date, has not been used to generate pop lyrics. By working and
communicating directly with professional songwriters, we develop a model that
is able to learn lyrical and stylistic tasks like rhyming, matching line beat
requirements, and ending lines with specific target words. Our approach
compares favorably to existing methods for multiple datasets and yields
positive results from our online studies and interviews with industry
songwriters.

    

### [[2111.07599] DNN gradient lossless compression: Can GenNorm be the answer?](http://arxiv.org/abs/2111.07599)


  In this paper, the problem of optimal gradient lossless compression in Deep
Neural Network (DNN) training is considered. Gradient compression is relevant
in many distributed DNN training scenarios, including the recently popular
federated learning (FL) scenario in which each remote users are connected to
the parameter server (PS) through a noiseless but rate limited channel. In
distributed DNN training, if the underlying gradient distribution is available,
classical lossless compression approaches can be used to reduce the number of
bits required for communicating the gradient entries. Mean field analysis has
suggested that gradient updates can be considered as independent random
variables, while Laplace approximation can be used to argue that gradient has a
distribution approximating the normal (Norm) distribution in some regimes. In
this paper we argue that, for some networks of practical interest, the gradient
entries can be well modelled as having a generalized normal (GenNorm)
distribution. We provide numerical evaluations to validate that the hypothesis
GenNorm modelling provides a more accurate prediction of the DNN gradient tail
distribution. Additionally, this modeling choice provides concrete improvement
in terms of lossless compression of the gradients when applying classical
fix-to-variable lossless coding algorithms, such as Huffman coding, to the
quantized gradient updates. This latter results indeed provides an effective
compression strategy with low memory and computational complexity that has
great practical relevance in distributed DNN training scenarios.

    

### [[2111.07602] Spectral Transform Forms Scalable Transformer](http://arxiv.org/abs/2111.07602)


  Many real-world relational systems, such as social networks and biological
systems, contain dynamic interactions. When learning dynamic graph
representation, it is essential to employ sequential temporal information and
geometric structure. Mainstream work achieves topological embedding via message
passing networks (e.g., GCN, GAT). The temporal evolution, on the other hand,
is conventionally expressed via memory units (e.g., LSTM or GRU) that possess
convenient information filtration in a gate mechanism. Though, such a design
prevents large-scale input sequence due to the over-complicated encoding. This
work learns from the philosophy of self-attention and proposes an efficient
spectral-based neural unit that employs informative long-range temporal
interaction. The developed spectral window unit (SWINIT) model predicts
scalable dynamic graphs with assured efficiency. The architecture is assembled
with a few simple effective computational blocks that constitute randomized
SVD, MLP, and graph Framelet convolution. The SVD plus MLP module encodes the
long-short-term feature evolution of the dynamic graph events. A fast framelet
graph transform in the framelet convolution embeds the structural dynamics.
Both strategies enhance the model's ability on scalable analysis. In
particular, the iterative SVD approximation shrinks the computational
complexity of attention to O(Nd\log(d)) for the dynamic graph with N edges and
d edge features, and the multiscale transform of framelet convolution allows
sufficient scalability in the network training. Our SWINIT achieves
state-of-the-art performance on a variety of online continuous-time dynamic
graph learning tasks, while compared to baseline methods, the number of its
learnable parameters reduces by up to seven times.

    

### [[2111.07603] Counterfactual Temporal Point Processes](http://arxiv.org/abs/2111.07603)


  Machine learning models based on temporal point processes are the state of
the art in a wide variety of applications involving discrete events in
continuous time. However, these models lack the ability to answer
counterfactual questions, which are increasingly relevant as these models are
being used to inform targeted interventions. In this work, our goal is to fill
this gap. To this end, we first develop a causal model of thinning for temporal
point processes that builds upon the Gumbel-Max structural causal model. This
model satisfies a desirable counterfactual monotonicity condition, which is
sufficient to identify counterfactual dynamics in the process of thinning.
Then, given an observed realization of a temporal point process with a given
intensity function, we develop a sampling algorithm that uses the above causal
model of thinning and the superposition theorem to simulate counterfactual
realizations of the temporal point process under a given alternative intensity
function. Simulation experiments using synthetic and real epidemiological data
show that the counterfactual realizations provided by our algorithm may give
valuable insights to enhance targeted interventions.

    

### [[2111.07608] Property Inference Attacks Against GANs](http://arxiv.org/abs/2111.07608)


  While machine learning (ML) has made tremendous progress during the past
decade, recent research has shown that ML models are vulnerable to various
security and privacy attacks. So far, most of the attacks in this field focus
on discriminative models, represented by classifiers. Meanwhile, little
attention has been paid to the security and privacy risks of generative models,
such as generative adversarial networks (GANs). In this paper, we propose the
first set of training dataset property inference attacks against GANs.
Concretely, the adversary aims to infer the macro-level training dataset
property, i.e., the proportion of samples used to train a target GAN with
respect to a certain attribute. A successful property inference attack can
allow the adversary to gain extra knowledge of the target GAN's training
dataset, thereby directly violating the intellectual property of the target
model owner. Also, it can be used as a fairness auditor to check whether the
target GAN is trained with a biased dataset. Besides, property inference can
serve as a building block for other advanced attacks, such as membership
inference. We propose a general attack pipeline that can be tailored to two
attack scenarios, including the full black-box setting and partial black-box
setting. For the latter, we introduce a novel optimization framework to
increase the attack efficacy. Extensive experiments over four representative
GAN models on five property inference tasks show that our attacks achieve
strong performance. In addition, we show that our attacks can be used to
enhance the performance of membership inference against GANs.

    

### [[2111.07613] Generate plane quad mesh with neural networks and tree search](http://arxiv.org/abs/2111.07613)


  The quality of mesh generation has long been considered a vital aspect in
providing engineers with reliable simulation results throughout the history of
the Finite Element Method (FEM). The element extraction method, which is
currently the most robust method, is used in business software. However, in
order to speed up extraction, the approach is done by finding the next element
that optimizes a target function, which can result in local mesh of bad quality
after many time steps. We provide TreeMesh, a method that uses this method in
conjunction with reinforcement learning (also possible with supervised
learning) and a novel Monte-Carlo tree search (MCTS) (Coulom(2006), Kocsis and
Szepesvári(2006), Browne et~al.(2012)). The algorithm is based on a
previously proposed approach (Pan et~al.(2021)). After making many improvements
on DRL (algorithm, state-action-reward setting) and adding a MCTS, it
outperforms the former work on the same boundary. Furthermore, using tree
search, our program reveals much preponderance on seed-density-changing
boundaries, which is common on thin-film materials.

    

### [[2111.07615] Delayed Feedback in Episodic Reinforcement Learning](http://arxiv.org/abs/2111.07615)


  There are many provably efficient algorithms for episodic reinforcement
learning. However, these algorithms are built under the assumption that the
sequences of states, actions and rewards associated with each episode arrive
immediately, allowing policy updates after every interaction with the
environment. This assumption is often unrealistic in practice, particularly in
areas such as healthcare and online recommendation. In this paper, we study the
impact of delayed feedback on several provably efficient algorithms for regret
minimisation in episodic reinforcement learning. Firstly, we consider updating
the policy as soon as new feedback becomes available. Using this updating
scheme, we show that the regret increases by an additive term involving the
number of states, actions, episode length and the expected delay. This additive
term changes depending on the optimistic algorithm of choice. We also show that
updating the policy less frequently can lead to an improved dependency of the
regret on the delays.

    

### [[2111.07632] CoReS: Compatible Representations via Stationarity](http://arxiv.org/abs/2111.07632)


  In this paper, we propose a novel method to learn internal feature
representation models that are \textit{compatible} with previously learned
ones. Compatible features enable for direct comparison of old and new learned
features, allowing them to be used interchangeably over time. This eliminates
the need for visual search systems to extract new features for all previously
seen images in the gallery-set when sequentially upgrading the representation
model. Extracting new features is typically quite expensive or infeasible in
the case of very large gallery-sets and/or real time systems (i.e.,
face-recognition systems, social networks, life-long learning systems, robotics
and surveillance systems). Our approach, called Compatible Representations via
Stationarity (CoReS), achieves compatibility by encouraging stationarity to the
learned representation model without relying on previously learned models.
Stationarity allows features' statistical properties not to change under time
shift so that the current learned features are inter-operable with the old
ones. We evaluate single and sequential multi-model upgrading in growing
large-scale training datasets and we show that our method improves the
state-of-the-art in achieving compatible features by a large margin. In
particular, upgrading ten times with training data taken from CASIA-WebFace and
evaluating in Labeled Face in the Wild (LFW), we obtain a 49\% increase in
measuring the average number of times compatibility is achieved, which is a
544\% relative improvement over previous state-of-the-art.

    

### [[2111.07634] Pseudo-domains in imaging data improve prediction of future disease status in multi-center studies](http://arxiv.org/abs/2111.07634)


  In multi-center randomized clinical trials imaging data can be diverse due to
acquisition technology or scanning protocols. Models predicting future outcome
of patients are impaired by this data heterogeneity. Here, we propose a
prediction method that can cope with a high number of different scanning sites
and a low number of samples per site. We cluster sites into pseudo-domains
based on visual appearance of scans, and train pseudo-domain specific models.
Results show that they improve the prediction accuracy for steatosis after 48
weeks from imaging data acquired at an initial visit and 12-weeks follow-up in
liver disease

    

### [[2111.07646] Multimodal Generalized Zero Shot Learning for Gleason Grading using Self-Supervised Learning](http://arxiv.org/abs/2111.07646)


  Gleason grading from histopathology images is essential for accurate prostate
cancer (PCa) diagnosis. Since such images are obtained after invasive tissue
resection quick diagnosis is challenging under the existing paradigm. We
propose a method to predict Gleason grades from magnetic resonance (MR) images
which are non-interventional and easily acquired. We solve the problem in a
generalized zero-shot learning (GZSL) setting since we may not access training
images of every disease grade. Synthetic MRI feature vectors of unseen grades
(classes) are generated by exploiting Gleason grades' ordered nature through a
conditional variational autoencoder (CVAE) incorporating self-supervised
learning. Corresponding histopathology features are generated using cycle GANs,
and combined with MR features to predict Gleason grades of test images.
Experimental results show our method outperforms competing feature generating
approaches for GZSL, and comes close to performance of fully supervised
methods.

    

### [[2111.07667] Versatile Inverse Reinforcement Learning via Cumulative Rewards](http://arxiv.org/abs/2111.07667)


  Inverse Reinforcement Learning infers a reward function from expert
demonstrations, aiming to encode the behavior and intentions of the expert.
Current approaches usually do this with generative and uni-modal models,
meaning that they encode a single behavior. In the common setting, where there
are various solutions to a problem and the experts show versatile behavior this
severely limits the generalization capabilities of these methods. We propose a
novel method for Inverse Reinforcement Learning that overcomes these problems
by formulating the recovered reward as a sum of iteratively trained
discriminators. We show on simulated tasks that our approach is able to recover
general, high-quality reward functions and produces policies of the same
quality as behavioral cloning approaches designed for versatile behavior.

    

### [[2111.07668] Fast Axiomatic Attribution for Neural Networks](http://arxiv.org/abs/2111.07668)


  Mitigating the dependence on spurious correlations present in the training
dataset is a quickly emerging and important topic of deep learning. Recent
approaches include priors on the feature attribution of a deep neural network
(DNN) into the training process to reduce the dependence on unwanted features.
However, until now one needed to trade off high-quality attributions,
satisfying desirable axioms, against the time required to compute them. This in
turn either led to long training times or ineffective attribution priors. In
this work, we break this trade-off by considering a special class of
efficiently axiomatically attributable DNNs for which an axiomatic feature
attribution can be computed with only a single forward/backward pass. We
formally prove that nonnegatively homogeneous DNNs, here termed
$\mathcal{X}$-DNNs, are efficiently axiomatically attributable and show that
they can be effortlessly constructed from a wide range of regular DNNs by
simply removing the bias term of each layer. Various experiments demonstrate
the advantages of $\mathcal{X}$-DNNs, beating state-of-the-art generic
attribution methods on regular DNNs for training with attribution priors.

    

### [[2111.07671] NeuralPDE: Modelling Dynamical Systems from Data](http://arxiv.org/abs/2111.07671)


  Many physical processes such as weather phenomena or fluid mechanics are
governed by partial differential equations (PDEs). Modelling such dynamical
systems using Neural Networks is an emerging research field. However, current
methods are restricted in various ways: they require prior knowledge about the
governing equations, and are limited to linear or first-order equations. In
this work we propose NeuralPDE, a model which combines convolutional neural
networks (CNNs) with differentiable ODE solvers to model dynamical systems. We
show that the Method of Lines used in standard PDE solvers can be represented
using convolutions which makes CNNs the natural choice to parametrize arbitrary
PDE dynamics. Our model can be applied to any data without requiring any prior
knowledge about the governing PDE. We evaluate NeuralPDE on datasets generated
by solving a wide variety of PDEs, covering higher orders, non-linear equations
and multiple spatial dimensions.

    

### [[2111.07679] Contrastive Representation Learning with Trainable Augmentation Channel](http://arxiv.org/abs/2111.07679)


  In contrastive representation learning, data representation is trained so
that it can classify the image instances even when the images are altered by
augmentations. However, depending on the datasets, some augmentations can
damage the information of the images beyond recognition, and such augmentations
can result in collapsed representations. We present a partial solution to this
problem by formalizing a stochastic encoding process in which there exist a
tug-of-war between the data corruption introduced by the augmentations and the
information preserved by the encoder. We show that, with the infoMax objective
based on this framework, we can learn a data-dependent distribution of
augmentations to avoid the collapse of the representation.

    

### [[1910.05858] Deep Kernels with Probabilistic Embeddings for Small-Data Learning](http://arxiv.org/abs/1910.05858)


  Gaussian Processes (GPs) are known to provide accurate predictions and
uncertainty estimates even with small amounts of labeled data by capturing
similarity between data points through their kernel function. However
traditional GP kernels are not very effective at capturing similarity between
high dimensional data points. Neural networks can be used to learn good
representations that encode intricate structures in high dimensional data, and
can be used as inputs to the GP kernel. However the huge data requirement of
neural networks makes this approach ineffective in small data settings. To
solves the conflicting problems of representation learning and data efficiency,
we propose to learn deep kernels on probabilistic embeddings by using a
probabilistic neural network. Our approach maps high-dimensional data to a
probability distribution in a low dimensional subspace and then computes a
kernel between these distributions to capture similarity. To enable end-to-end
learning, we derive a functional gradient descent procedure for training the
model. Experiments on a variety of datasets show that our approach outperforms
the state-of-the-art in GP kernel learning in both supervised and
semi-supervised settings. We also extend our approach to other small-data
paradigms such as few-shot classification where it outperforms previous
approaches on mini-Imagenet and CUB datasets.

    

### [[1910.07115] HiGitClass: Keyword-Driven Hierarchical Classification of GitHub Repositories](http://arxiv.org/abs/1910.07115)


  GitHub has become an important platform for code sharing and scientific
exchange. With the massive number of repositories available, there is a
pressing need for topic-based search. Even though the topic label functionality
has been introduced, the majority of GitHub repositories do not have any
labels, impeding the utility of search and topic-based analysis. This work
targets the automatic repository classification problem as keyword-driven
hierarchical classification. Specifically, users only need to provide a label
hierarchy with keywords to supply as supervision. This setting is flexible,
adaptive to the users' needs, accounts for the different granularity of topic
labels and requires minimal human effort. We identify three key challenges of
this problem, namely (1) the presence of multi-modal signals; (2) supervision
scarcity and bias; (3) supervision format mismatch. In recognition of these
challenges, we propose the HiGitClass framework, comprising of three modules:
heterogeneous information network embedding; keyword enrichment; topic modeling
and pseudo document generation. Experimental results on two GitHub repository
collections confirm that HiGitClass is superior to existing weakly-supervised
and dataless hierarchical classification methods, especially in its ability to
integrate both structured and unstructured data for repository classification.

    

### [[1911.07391] Justification-Based Reliability in Machine Learning](http://arxiv.org/abs/1911.07391)


  With the advent of Deep Learning, the field of machine learning (ML) has
surpassed human-level performance on diverse classification tasks. At the same
time, there is a stark need to characterize and quantify reliability of a
model's prediction on individual samples. This is especially true in
application of such models in safety-critical domains of industrial control and
healthcare. To address this need, we link the question of reliability of a
model's individual prediction to the epistemic uncertainty of the model's
prediction. More specifically, we extend the theory of Justified True Belief
(JTB) in epistemology, created to study the validity and limits of
human-acquired knowledge, towards characterizing the validity and limits of
knowledge in supervised classifiers. We present an analysis of neural network
classifiers linking the reliability of its prediction on an input to
characteristics of the support gathered from the input and latent spaces of the
network. We hypothesize that the JTB analysis exposes the epistemic uncertainty
(or ignorance) of a model with respect to its inference, thereby allowing for
the inference to be only as strong as the justification permits. We explore
various forms of support (for e.g., k-nearest neighbors (k-NN) and l_p-norm
based) generated for an input, using the training data to construct a
justification for the prediction with that input. Through experiments conducted
on simulated and real datasets, we demonstrate that our approach can provide
reliability for individual predictions and characterize regions where such
reliability cannot be ascertained.

    

### [[2001.11419] Grassmannian Optimization for Online Tensor Completion and Tracking with the t-SVD](http://arxiv.org/abs/2001.11419)


  We propose a new fast streaming algorithm for the tensor completion problem
of imputing missing entries of a low-tubal-rank tensor using the tensor
singular value decomposition (t-SVD) algebraic framework. We show the t-SVD is
a specialization of the well-studied block-term decomposition for third-order
tensors, and we present an algorithm under this model that can track changing
free submodules from incomplete streaming 2-D data. The proposed algorithm uses
principles from incremental gradient descent on the Grassmann manifold of
subspaces to solve the tensor completion problem with linear complexity and
constant memory in the number of time samples. We provide a local expected
linear convergence result for our algorithm. Our empirical results are
competitive in accuracy but much faster in compute time than state-of-the-art
tensor completion algorithms on real applications to recover temporal
chemo-sensing and MRI data under limited sampling.

    

### [[2002.05212] Estimating Uncertainty Intervals from Collaborating Networks](http://arxiv.org/abs/2002.05212)


  Effective decision making requires understanding the uncertainty inherent in
a prediction. In regression, this uncertainty can be estimated by a variety of
methods; however, many of these methods are laborious to tune, generate
overconfident uncertainty intervals, or lack sharpness (give imprecise
intervals). We address these challenges by proposing a novel method to capture
predictive distributions in regression by defining two neural networks with two
distinct loss functions. Specifically, one network approximates the cumulative
distribution function, and the second network approximates its inverse. We
refer to this method as Collaborating Networks (CN). Theoretical analysis
demonstrates that a fixed point of the optimization is at the idealized
solution, and that the method is asymptotically consistent to the ground truth
distribution. Empirically, learning is straightforward and robust. We benchmark
CN against several common approaches on two synthetic and six real-world
datasets, including forecasting A1c values in diabetic patients from electronic
health records, where uncertainty is critical. In the synthetic data, the
proposed approach essentially matches ground truth. In the real-world datasets,
CN improves results on many performance metrics, including log-likelihood
estimates, mean absolute errors, coverage estimates, and prediction interval
widths.

    

### [[2002.08313] NNoculation: Catching BadNets in the Wild](http://arxiv.org/abs/2002.08313)


  This paper proposes a novel two-stage defense (NNoculation) against
backdoored neural networks (BadNets) that, repairs a BadNet both pre-deployment
and online in response to backdoored test inputs encountered in the field. In
the pre-deployment stage, NNoculation retrains the BadNet with random
perturbations of clean validation inputs to partially reduce the adversarial
impact of a backdoor. Post-deployment, NNoculation detects and quarantines
backdoored test inputs by recording disagreements between the original and
pre-deployment patched networks. A CycleGAN is then trained to learn
transformations between clean validation and quarantined inputs; i.e., it
learns to add triggers to clean validation images. Backdoored validation images
along with their correct labels are used to further retrain the pre-deployment
patched network, yielding our final defense. Empirical evaluation on a
comprehensive suite of backdoor attacks show that NNoculation outperforms all
state-of-the-art defenses that make restrictive assumptions and only work on
specific backdoor attacks, or fail on adaptive attacks. In contrast,
NNoculation makes minimal assumptions and provides an effective defense, even
under settings where existing defenses are ineffective due to attackers
circumventing their restrictive assumptions.

    

### [[2005.08480] Non-Clicks Mean Irrelevant? Propensity Ratio Scoring As a Correction](http://arxiv.org/abs/2005.08480)


  Recent advances in unbiased learning to rank (LTR) count on Inverse
Propensity Scoring (IPS) to eliminate bias in implicit feedback. Though
theoretically sound in correcting the bias introduced by treating clicked
documents as relevant, IPS ignores the bias caused by (implicitly) treating
non-clicked ones as irrelevant. In this work, we first rigorously prove that
such use of click data leads to unnecessary pairwise comparisons between
relevant documents, which prevent unbiased ranker optimization. Based on the
proof, we derive a simple yet well justified new weighting scheme, called
Propensity Ratio Scoring (PRS), which provides treatments on both clicks and
non-clicks. Besides correcting the bias in clicks, PRS avoids relevant-relevant
document comparisons in LTR training and enjoys a lower variability. Our
extensive empirical evaluations confirm that PRS ensures a more effective use
of click data and improved performance in both synthetic data from a set of LTR
benchmarks, as well as in the real-world large-scale data from GMail search.

    

### [[2007.01174] Robust Inverse Reinforcement Learning under Transition Dynamics Mismatch](http://arxiv.org/abs/2007.01174)


  We study the inverse reinforcement learning (IRL) problem under a transition
dynamics mismatch between the expert and the learner. Specifically, we consider
the Maximum Causal Entropy (MCE) IRL learner model and provide a tight upper
bound on the learner's performance degradation based on the $\ell_1$-distance
between the transition dynamics of the expert and the learner. Leveraging
insights from the Robust RL literature, we propose a robust MCE IRL algorithm,
which is a principled approach to help with this mismatch. Finally, we
empirically demonstrate the stable performance of our algorithm compared to the
standard MCE IRL algorithm under transition dynamics mismatches in both finite
and continuous MDP problems.

    

### [[2007.08322] Understanding Implicit Regularization in Over-Parameterized Single Index Model](http://arxiv.org/abs/2007.08322)


  In this paper, we leverage over-parameterization to design
regularization-free algorithms for the high-dimensional single index model and
provide theoretical guarantees for the induced implicit regularization
phenomenon. Specifically, we study both vector and matrix single index models
where the link function is nonlinear and unknown, the signal parameter is
either a sparse vector or a low-rank symmetric matrix, and the response
variable can be heavy-tailed. To gain a better understanding of the role played
by implicit regularization without excess technicality, we assume that the
distribution of the covariates is known a priori. For both the vector and
matrix settings, we construct an over-parameterized least-squares loss function
by employing the score function transform and a robust truncation step designed
specifically for heavy-tailed data. We propose to estimate the true parameter
by applying regularization-free gradient descent to the loss function. When the
initialization is close to the origin and the stepsize is sufficiently small,
we prove that the obtained solution achieves minimax optimal statistical rates
of convergence in both the vector and matrix cases. In addition, our
experimental results support our theoretical findings and also demonstrate that
our methods empirically outperform classical methods with explicit
regularization in terms of both $\ell_2$-statistical rate and variable
selection consistency.

    

### [[2008.01064] Predicting What You Already Know Helps: Provable Self-Supervised Learning](http://arxiv.org/abs/2008.01064)


  Self-supervised representation learning solves auxiliary prediction tasks
(known as pretext tasks) without requiring labeled data to learn useful
semantic representations. These pretext tasks are created solely using the
input features, such as predicting a missing image patch, recovering the color
channels of an image from context, or predicting missing words in text; yet
predicting this \textit{known} information helps in learning representations
effective for downstream prediction tasks. We posit a mechanism exploiting the
statistical connections between certain {\em reconstruction-based} pretext
tasks that guarantee to learn a good representation. Formally, we quantify how
the approximate independence between the components of the pretext task
(conditional on the label and latent variables) allows us to learn
representations that can solve the downstream task by just training a linear
layer on top of the learned representation. We prove the linear layer yields
small approximation error even for complex ground truth function class and will
drastically reduce labeled sample complexity. Next, we show a simple
modification of our method leads to nonlinear CCA, analogous to the popular
SimSiam algorithm, and show similar guarantees for nonlinear CCA.

    

### [[2008.06595] Decision-making at Unsignalized Intersection for Autonomous Vehicles: Left-turn Maneuver with Deep Reinforcement Learning](http://arxiv.org/abs/2008.06595)


  Decision-making module enables autonomous vehicles to reach appropriate
maneuvers in the complex urban environments, especially the intersection
situations. This work proposes a deep reinforcement learning (DRL) based
left-turn decision-making framework at unsignalized intersection for autonomous
vehicles. The objective of the studied automated vehicle is to make an
efficient and safe left-turn maneuver at a four-way unsignalized intersection.
The exploited DRL methods include deep Q-learning (DQL) and double DQL.
Simulation results indicate that the presented decision-making strategy could
efficaciously reduce the collision rate and improve transport efficiency. This
work also reveals that the constructed left-turn control structure has a great
potential to be applied in real-time.

    

### [[2008.10846] Federated Learning for Channel Estimation in Conventional and RIS-Assisted Massive MIMO](http://arxiv.org/abs/2008.10846)


  Machine learning (ML) has attracted a great research interest for physical
layer design problems, such as channel estimation, thanks to its low complexity
and robustness. Channel estimation via ML requires model training on a dataset,
which usually includes the received pilot signals as input and channel data as
output. In previous works, model training is mostly done via centralized
learning (CL), where the whole training dataset is collected from the users at
the base station (BS). This approach introduces huge communication overhead for
data collection. In this paper, to address this challenge, we propose a
federated learning (FL) framework for channel estimation. We design a
convolutional neural network (CNN) trained on the local datasets of the users
without sending them to the BS. We develop FL-based channel estimation schemes
for both conventional and RIS (intelligent reflecting surface) assisted massive
MIMO (multiple-input multiple-output) systems, where a single CNN is trained
for two different datasets for both scenarios. We evaluate the performance for
noisy and quantized model transmission and show that the proposed approach
provides approximately 16 times lower overhead than CL, while maintaining
satisfactory performance close to CL. Furthermore, the proposed architecture
exhibits lower estimation error than the state-of-the-art ML-based schemes.

    

### [[2009.05949] Advanced Graph-Based Deep Learning for Probabilistic Type Inference](http://arxiv.org/abs/2009.05949)


  Dynamically typed languages such as JavaScript and Python have emerged as the
most popular programming languages in use. Important benefits can accrue from
including type annotations in dynamically typed programs. This approach to
gradual typing is exemplified by the TypeScript programming system which allows
programmers to specify partially typed programs, and then uses static analysis
to infer the remaining types. However, in general, the effectiveness of static
type inference is limited and depends on the complexity of the program's
structure and the initial type annotations. As a result, there is a strong
motivation for new approaches that can advance the state of the art in
statically predicting types in dynamically typed programs, and that do so with
acceptable performance for use in interactive programming environments.
Previous work has demonstrated the promise of probabilistic type inference
using deep learning. In this paper, we advance past work by introducing a range
of graph neural network (GNN) models that operate on a novel type flow graph
(TFG) representation. The TFG represents an input program's elements as graph
nodes connected with syntax edges and data flow edges, and our GNN models are
trained to predict the type labels in the TFG for a given input program. We
study different design choices for our GNN models for the 100 most common types
in our evaluation dataset, and show that our best two GNN configurations for
accuracy achieve a top-1 accuracy of 87.76% and 86.89% respectively,
outperforming the two most closely related deep learning type inference
approaches from past work -- DeepTyper with a top-1 accuracy of 84.62% and
LambdaNet with a top-1 accuracy of 79.45%. Further, the average inference
throughputs of those two configurations are 353.8 and 1,303.9 files/second,
compared to 186.7 files/second for DeepTyper and 1,050.3 files/second for
LambdaNet.

    

### [[2009.08687] Chemical Property Prediction Under Experimental Biases](http://arxiv.org/abs/2009.08687)


  Predicting the chemical properties of compounds is crucial in discovering
novel materials and drugs with specific desired characteristics. Recent
significant advances in machine learning technologies have enabled automatic
predictive modeling from past experimental data reported in the literature.
However, these datasets are often biased because of various reasons, such as
experimental plans and publication decisions, and the prediction models trained
using such biased datasets often suffer from over-fitting to the biased
distributions and perform poorly on subsequent uses. Hence, this study focused
on mitigating bias in the experimental datasets. We adopted two techniques from
causal inference and domain adaptation combined with graph neural networks that
can represent molecular structures. The experimental results in four possible
bias scenarios indicated that the inverse propensity scoring-based method made
solid improvements, but the domain-invariant representation learning approach
failed.

    

### [[2009.11128] Using Under-trained Deep Ensembles to Learn Under Extreme Label Noise](http://arxiv.org/abs/2009.11128)


  Improper or erroneous labelling can pose a hindrance to reliable
generalization for supervised learning. This can have negative consequences,
especially for critical fields such as healthcare. We propose an effective new
approach for learning under extreme label noise, based on under-trained deep
ensembles. Each ensemble member is trained with a subset of the training data,
to acquire a general overview of the decision boundary separation, without
focusing on potentially erroneous details. The accumulated knowledge of the
ensemble is combined to form new labels, that determine a better class
separation than the original labels. A new model is trained with these labels
to generalize reliably despite the label noise. We focus on a healthcare
setting and extensively evaluate our approach on the task of sleep apnea
detection. For comparison with related work, we additionally evaluate on the
task of digit recognition. In our experiments, we observed performance
improvement in accuracy from 6.7\% up-to 49.3\% for the task of digit
classification and in kappa from 0.02 up-to 0.55 for the task of sleep apnea
detection.

    

### [[2010.11413] Predicting human decision making in psychological tasks with recurrent neural networks](http://arxiv.org/abs/2010.11413)


  Unlike traditional time series, the action sequences of human decision making
usually involve many cognitive processes such as beliefs, desires, intentions
and theory of mind, i.e. what others are thinking. This makes predicting human
decision making challenging to be treated agnostically to the underlying
psychological mechanisms. We propose to use a recurrent neural network
architecture based on long short-term memory networks (LSTM) to predict the
time series of the actions taken by the human subjects at each step of their
decision making, the first application of such methods in this research domain.
In this study, we collate the human data from 8 published literature of the
Iterated Prisoner's Dilemma comprising 168,386 individual decisions and
postprocess them into 8,257 behavioral trajectories of 9 actions each for both
players. Similarly, we collate 617 trajectories of 95 actions from 10 different
published studies of Iowa Gambling Task experiments with healthy human
subjects. We train our prediction networks on the behavioral data from these
published psychological experiments of human decision making, and demonstrate a
clear advantage over the state-of-the-art methods in predicting human decision
making trajectories in both single-agent scenarios such as the Iowa Gambling
Task and multi-agent scenarios such as the Iterated Prisoner's Dilemma. In the
prediction, we observe that the weights of the top performers tends to have a
wider distribution, and a bigger bias in the LSTM networks, which suggests
possible interpretations for the distribution of strategies adopted by each
group.

    

### [[2010.15382] Learning to Actively Learn: A Robust Approach](http://arxiv.org/abs/2010.15382)


  This work proposes a procedure for designing algorithms for specific adaptive
data collection tasks like active learning and pure-exploration multi-armed
bandits. Unlike the design of traditional adaptive algorithms that rely on
concentration of measure and careful analysis to justify the correctness and
sample complexity of the procedure, our adaptive algorithm is learned via
adversarial training over equivalence classes of problems derived from
information theoretic lower bounds. In particular, a single adaptive learning
algorithm is learned that competes with the best adaptive algorithm learned for
each equivalence class. Our procedure takes as input just the available
queries, set of hypotheses, loss function, and total query budget. This is in
contrast to existing meta-learning work that learns an adaptive algorithm
relative to an explicit, user-defined subset or prior distribution over
problems which can be challenging to define and be mismatched to the instance
encountered at test time. This work is particularly focused on the regime when
the total query budget is very small, such as a few dozen, which is much
smaller than those budgets typically considered by theoretically derived
algorithms. We perform synthetic experiments to justify the stability and
effectiveness of the training procedure, and then evaluate the method on tasks
derived from real data including a noisy 20 Questions game and a joke
recommendation task.

    

### [[2011.11236] Learning Hidden Markov Models from Aggregate Observations](http://arxiv.org/abs/2011.11236)


  In this paper, we propose an algorithm for estimating the parameters of a
time-homogeneous hidden Markov model from aggregate observations. This problem
arises when only the population level counts of the number of individuals at
each time step are available, from which one seeks to learn the individual
hidden Markov model. Our algorithm is built upon expectation-maximization and
the recently proposed aggregate inference algorithm, the Sinkhorn belief
propagation. As compared with existing methods such as expectation-maximization
with non-linear belief propagation, our algorithm exhibits convergence
guarantees. Moreover, our learning framework naturally reduces to the standard
Baum-Welch learning algorithm when observations corresponding to a single
individual are recorded. We further extend our learning algorithm to handle
HMMs with continuous observations. The efficacy of our algorithm is
demonstrated on a variety of datasets.

    

### [[2011.14371] Predicting Regional Locust Swarm Distribution with Recurrent Neural Networks](http://arxiv.org/abs/2011.14371)


  Locust infestation of some regions in the world, including Africa, Asia and
Middle East has become a concerning issue that can affect the health and the
lives of millions of people. In this respect, there have been attempts to
resolve or reduce the severity of this problem via detection and monitoring of
locust breeding areas using satellites and sensors, or the use of chemicals to
prevent the formation of swarms. However, such methods have not been able to
suppress the emergence and the collective behaviour of locusts. The ability to
predict the location of the locust swarms prior to their formation, on the
other hand, can help people get prepared and tackle the infestation issue more
effectively. Here, we use machine learning to predict the location of locust
swarms using the available data published by the Food and Agriculture
Organization of the United Nations. The data includes the location of the
observed swarms as well as environmental information, including soil moisture
and the density of vegetation. The obtained results show that our proposed
model can successfully, and with reasonable precision, predict the location of
locust swarms, as well as their likely level of damage using a notion of
density.

    

### [[2012.05549] Categorical Perception: A Groundwork for Deep Learning](http://arxiv.org/abs/2012.05549)


  A well-known perceptual consequence of categorization in humans and other
animals, called categorical perception, is notably characterized by a
within-category compression and a between-category separation: two items, close
in input space, are perceived closer if they belong to the same category than
if they belong to different categories. Elaborating on experimental and
theoretical results in cognitive science, here we study categorical effects in
artificial neural networks. We combine a theoretical analysis that makes use of
mutual and Fisher information quantities, and a series of numerical simulations
on networks of increasing complexity. These formal and numerical analyses
provide insights into the geometry of the neural representation in deep layers,
with expansion of space near category boundaries and contraction far from
category boundaries. We investigate categorical representation by using two
complementary approaches: one mimics experiments in psychophysics and cognitive
neuroscience by means of morphed continua between stimuli of different
categories, while the other introduces a categoricality index that, for each
layer in the network, quantifies the separability of the categories at the
neural population level. We show on both shallow and deep neural networks that
category learning automatically induces categorical perception. We further show
that the deeper a layer, the stronger the categorical effects. As an outcome of
our study, we propose a coherent view of the efficacy of different heuristic
practices of the dropout regularization technique. More generally, our view,
which finds echoes in the neuroscience literature, insists on the differential
impact of noise in any given layer depending on the geometry of the neural
representation that is being learned, i.e. on how this geometry reflects the
structure of the categories.

    

### [[2101.00124] Discourse-level Relation Extraction via Graph Pooling](http://arxiv.org/abs/2101.00124)


  The ability to capture complex linguistic structures and long-term
dependencies among words in the passage is essential for discourse-level
relation extraction (DRE) tasks. Graph neural networks (GNNs), one of the
methods to encode dependency graphs, have been shown effective in prior works
for DRE. However, relatively little attention has been paid to receptive fields
of GNNs, which can be crucial for cases with extremely long text that requires
discourse understanding. In this work, we leverage the idea of graph pooling
and propose to use pooling-unpooling framework on DRE tasks. The pooling branch
reduces the graph size and enables the GNNs to obtain larger receptive fields
within fewer layers; the unpooling branch restores the pooled graph to its
original resolution so that representations for entity mention can be
extracted. We propose Clause Matching (CM), a novel linguistically inspired
graph pooling method for NLP tasks. Experiments on two DRE datasets demonstrate
that our models significantly improve over baselines when modeling long-term
dependencies is required, which shows the effectiveness of the
pooling-unpooling framework and our CM pooling method.

    

### [[2101.00169] Generative Deep Learning for Virtuosic Classical Music: Generative Adversarial Networks as Renowned Composers](http://arxiv.org/abs/2101.00169)


  Current AI-generated music lacks fundamental principles of good compositional
techniques. By narrowing down implementation issues both programmatically and
musically, we can create a better understanding of what parameters are
necessary for a generated composition nearly indistinguishable from that of a
master composer.

    

### [[2101.07241] Learning by Watching: Physical Imitation of Manipulation Skills from Human Videos](http://arxiv.org/abs/2101.07241)


  Learning from visual data opens the potential to accrue a large range of
manipulation behaviors by leveraging human demonstrations without specifying
each of them mathematically, but rather through natural task specification. In
this paper, we present Learning by Watching (LbW), an algorithmic framework for
policy learning through imitation from a single video specifying the task. The
key insights of our method are two-fold. First, since the human arms may not
have the same morphology as robot arms, our framework learns unsupervised human
to robot translation to overcome the morphology mismatch issue. Second, to
capture the details in salient regions that are crucial for learning state
representations, our model performs unsupervised keypoint detection on the
translated robot videos. The detected keypoints form a structured
representation that contains semantically meaningful information and can be
used directly for computing reward and policy learning. We evaluate the
effectiveness of our LbW framework on five robot manipulation tasks, including
reaching, pushing, sliding, coffee making, and drawer closing. Extensive
experimental evaluations demonstrate that our method performs favorably against
the state-of-the-art approaches.

    

### [[2101.08367] Influence Estimation for Generative Adversarial Networks](http://arxiv.org/abs/2101.08367)


  Identifying harmful instances, whose absence in a training dataset improves
model performance, is important for building better machine learning models.
Although previous studies have succeeded in estimating harmful instances under
supervised settings, they cannot be trivially extended to generative
adversarial networks (GANs). This is because previous approaches require that
(1) the absence of a training instance directly affects the loss value and that
(2) the change in the loss directly measures the harmfulness of the instance
for the performance of a model. In GAN training, however, neither of the
requirements is satisfied. This is because, (1) the generator's loss is not
directly affected by the training instances as they are not part of the
generator's training steps, and (2) the values of GAN's losses normally do not
capture the generative performance of a model. To this end, (1) we propose an
influence estimation method that uses the Jacobian of the gradient of the
generator's loss with respect to the discriminator's parameters (and vice
versa) to trace how the absence of an instance in the discriminator's training
affects the generator's parameters, and (2) we propose a novel evaluation
scheme, in which we assess harmfulness of each training instance on the basis
of how GAN evaluation metric (e.g., inception score) is expect to change due to
the removal of the instance. We experimentally verified that our influence
estimation method correctly inferred the changes in GAN evaluation metrics.
Further, we demonstrated that the removal of the identified harmful instances
effectively improved the model's generative performance with respect to various
GAN evaluation metrics.

    

### [[2102.05198] Statistical Inference for Polyak-Ruppert Averaged Zeroth-order Stochastic Gradient Algorithm](http://arxiv.org/abs/2102.05198)


  Statistical machine learning models trained with stochastic gradient
algorithms are increasingly being deployed in critical scientific applications.
However, computing the stochastic gradient in several such applications is
highly expensive or even impossible at times. In such cases, derivative-free or
zeroth-order algorithms are used. An important question which has thus far not
been addressed sufficiently in the statistical machine learning literature is
that of equipping stochastic zeroth-order algorithms with practical yet
rigorous inferential capabilities so that we not only have point estimates or
predictions but also quantify the associated uncertainty via confidence
intervals or sets. Towards this, in this work, we first establish a central
limit theorem for Polyak-Ruppert averaged stochastic zeroth-order gradient
algorithm. We then provide online estimators of the asymptotic covariance
matrix appearing in the central limit theorem, thereby providing a practical
procedure for constructing asymptotically valid confidence sets (or intervals)
for parameter estimation (or prediction) in the zeroth-order setting.

    

### [[2102.07969] Machine Learning Based Cyber Attacks Targeting on Controlled Information: A Survey](http://arxiv.org/abs/2102.07969)


  Stealing attack against controlled information, along with the increasing
number of information leakage incidents, has become an emerging cyber security
threat in recent years. Due to the booming development and deployment of
advanced analytics solutions, novel stealing attacks utilize machine learning
(ML) algorithms to achieve high success rate and cause a lot of damage.
Detecting and defending against such attacks is challenging and urgent so that
governments, organizations, and individuals should attach great importance to
the ML-based stealing attacks. This survey presents the recent advances in this
new type of attack and corresponding countermeasures. The ML-based stealing
attack is reviewed in perspectives of three categories of targeted controlled
information, including controlled user activities, controlled ML model-related
information, and controlled authentication information. Recent publications are
summarized to generalize an overarching attack methodology and to derive the
limitations and future directions of ML-based stealing attacks. Furthermore,
countermeasures are proposed towards developing effective protections from
three aspects -- detection, disruption, and isolation.

    

### [[2102.08026] EDITH :ECG biometrics aided by Deep learning for reliable Individual auTHentication](http://arxiv.org/abs/2102.08026)


  In recent years, physiological signal based authentication has shown great
promises,for its inherent robustness against forgery. Electrocardiogram (ECG)
signal, being the most widely studied biosignal, has also received the highest
level of attention in this regard. It has been proven with numerous studies
that by analyzing ECG signals from different persons, it is possible to
identify them, with acceptable accuracy. In this work, we present, EDITH, a
deep learning-based framework for ECG biometrics authentication system.
Moreover, we hypothesize and demonstrate that Siamese architectures can be used
over typical distance metrics for improved performance. We have evaluated EDITH
using 4 commonly used datasets and outperformed the prior works using less
number of beats. EDITH performs competitively using just a single heartbeat
(96-99.75% accuracy) and can be further enhanced by fusing multiple beats (100%
accuracy from 3 to 6 beats). Furthermore, the proposed Siamese architecture
manages to reduce the identity verification Equal Error Rate (EER) to 1.29%. A
limited case study of EDITH with real-world experimental data also suggests its
potential as a practical authentication system.

    

### [[2103.01089] A Biased Graph Neural Network Sampler with Near-Optimal Regret](http://arxiv.org/abs/2103.01089)


  Graph neural networks (GNN) have recently emerged as a vehicle for applying
deep network architectures to graph and relational data. However, given the
increasing size of industrial datasets, in many practical situations the
message passing computations required for sharing information across GNN layers
are no longer scalable. Although various sampling methods have been introduced
to approximate full-graph training within a tractable budget, there remain
unresolved complications such as high variances and limited theoretical
guarantees. To address these issues, we build upon existing work and treat GNN
neighbor sampling as a multi-armed bandit problem but with a newly-designed
reward function that introduces some degree of bias designed to reduce variance
and avoid unstable, possibly-unbounded pay outs. And unlike prior bandit-GNN
use cases, the resulting policy leads to near-optimal regret while accounting
for the GNN training dynamics introduced by SGD. From a practical standpoint,
this translates into lower variance estimates and competitive or superior test
accuracy across several benchmarks.

    

### [[2103.06859] Understanding the Origin of Information-Seeking Exploration in Probabilistic Objectives for Control](http://arxiv.org/abs/2103.06859)


  The exploration-exploitation trade-off is central to the description of
adaptive behaviour in fields ranging from machine learning, to biology, to
economics. While many approaches have been taken, one approach to solving this
trade-off has been to equip or propose that agents possess an intrinsic
'exploratory drive' which is often implemented in terms of maximizing the
agents information gain about the world -- an approach which has been widely
studied in machine learning and cognitive science. In this paper we
mathematically investigate the nature and meaning of such approaches and
demonstrate that this combination of utility maximizing and information-seeking
behaviour arises from the minimization of an entirely difference class of
objectives we call divergence objectives. We propose a dichotomy in the
objective functions underlying adaptive behaviour between \emph{evidence}
objectives, which correspond to well-known reward or utility maximizing
objectives in the literature, and \emph{divergence} objectives which instead
seek to minimize the divergence between the agent's expected and desired
futures, and argue that this new class of divergence objectives could form the
mathematical foundation for a much richer understanding of the exploratory
components of adaptive and intelligent action, beyond simply greedy utility
maximization.

    

### [[2103.11166] Efficient Density Ratio-Guided Subsampling of Conditional GANs, With Conditioning on a Class or a Continuous Variable](http://arxiv.org/abs/2103.11166)


  Recently, subsampling or refining images generated from unconditional
generative adversarial networks (GANs) has been actively studied to improve the
overall image quality. Unfortunately, these methods are often observed less
effective or inefficient in handling conditional GANs (cGANs) -- conditioning
on a class (aka class-conditional GANs) or a continuous variable (aka
continuous cGANs or CcGANs). In this work, we introduce an effective and
efficient subsampling scheme, named conditional density ratio-guided rejection
sampling (cDR-RS), to sample high-quality images from cGANs. Specifically, we
first develop a novel conditional density ratio estimation method, termed
cDRE-F-cSP, by proposing the conditional Softplus (cSP) loss and an improved
feature extraction mechanism. We then derive the error bound of a density ratio
model trained with the cSP loss. Finally, we accept or reject a fake image in
terms of its estimated conditional density ratio. A filtering scheme is also
developed to increase fake images' label consistency without losing diversity
when sampling from CcGANs. We extensively test the effectiveness and efficiency
of cDR-RS in sampling from both class-conditional GANs and CcGANs on five
benchmark datasets. When sampling from class-conditional GANs, cDR-RS
outperforms modern state-of-the-art methods by a large margin (except
DRE-F-SP+RS) in terms of effectiveness. Although the effectiveness of cDR-RS is
often comparable to that of DRE-F-SP+RS, cDR-RS is substantially more
efficient. When sampling from CcGANs, the superiority of cDR-RS is even more
noticeable in terms of both effectiveness and efficiency. Notably, with the
consumption of reasonable computational resources, cDR-RS can substantially
reduce Label Score without decreasing the diversity of CcGAN-generated images,
while other methods often need to trade much diversity for slightly improved
Label Score.

    

### [[2104.03602] SiT: Self-supervised vIsion Transformer](http://arxiv.org/abs/2104.03602)


  Self-supervised learning methods are gaining increasing traction in computer
vision due to their recent success in reducing the gap with supervised
learning. In natural language processing (NLP) self-supervised learning and
transformers are already the methods of choice. The recent literature suggests
that the transformers are becoming increasingly popular also in computer
vision. So far, the vision transformers have been shown to work well when
pretrained either using a large scale supervised data or with some kind of
co-supervision, e.g. in terms of teacher network. These supervised pretrained
vision transformers achieve very good results in downstream tasks with minimal
changes. In this work we investigate the merits of self-supervised learning for
pretraining image/vision transformers and then using them for downstream
classification tasks. We propose Self-supervised vIsion Transformers (SiT) and
discuss several self-supervised training mechanisms to obtain a pretext model.
The architectural flexibility of SiT allows us to use it as an autoencoder and
work with multiple self-supervised tasks seamlessly. We show that a pretrained
SiT can be finetuned for a downstream classification task on small scale
datasets, consisting of a few thousand images rather than several millions. The
proposed approach is evaluated on standard datasets using common protocols. The
results demonstrate the strength of the transformers and their suitability for
self-supervised learning. We outperformed existing self-supervised learning
methods by large margin. We also observed that SiT is good for few shot
learning and also showed that it is learning useful representation by simply
training a linear classifier on top of the learned features from SiT.
Pretraining, finetuning, and evaluation codes will be available under:
this https URL.

    

### [[2104.09027] Decentralized Inference with Graph Neural Networks in Wireless Communication Systems](http://arxiv.org/abs/2104.09027)


  Graph neural network (GNN) is an efficient neural network model for graph
data and is widely used in different fields, including wireless communications.
Different from other neural network models, GNN can be implemented in a
decentralized manner with information exchanges among neighbors, making it a
potentially powerful tool for decentralized control in wireless communication
systems. The main bottleneck, however, is wireless channel impairments that
deteriorate the prediction robustness of GNN. To overcome this obstacle, we
analyze and enhance the robustness of the decentralized GNN in different
wireless communication systems in this paper. Specifically, using a GNN binary
classifier as an example, we first develop a methodology to verify whether the
predictions are robust. Then, we analyze the performance of the decentralized
GNN binary classifier in both uncoded and coded wireless communication systems.
To remedy imperfect wireless transmission and enhance the prediction
robustness, we further propose novel retransmission mechanisms for the above
two communication systems, respectively. Through simulations on the synthetic
graph data, we validate our analysis, verify the effectiveness of the proposed
retransmission mechanisms, and provide some insights for practical
implementation.

    

### [[2104.09665] Learning GMMs with Nearly Optimal Robustness Guarantees](http://arxiv.org/abs/2104.09665)


  In this work we solve the problem of robustly learning a high-dimensional
Gaussian mixture model with $k$ components from $\epsilon$-corrupted samples up
to accuracy $\widetilde{O}(\epsilon)$ in total variation distance for any
constant $k$ and with mild assumptions on the mixture. This robustness
guarantee is optimal up to polylogarithmic factors. The main challenge is that
most earlier works rely on learning individual components in the mixture, but
this is impossible in our setting, at least for the types of strong robustness
guarantees we are aiming for. Instead we introduce a new framework which we
call {\em strong observability} that gives us a route to circumvent this
obstacle.

    

### [[2104.13039] Deep Learning of the Eddington Tensor in the Core-collapse Supernova Simulation](http://arxiv.org/abs/2104.13039)


  We trained deep neural networks (DNNs) as a function of the neutrino energy
density, flux, and the fluid velocity to reproduce the Eddington tensor for
neutrinos obtained in our first-principles core-collapse supernova (CCSN)
simulations. Although the moment method, which is one of the most popular
approximations for neutrino transport, requires a closure relation, none of the
analytical closure relations commonly employed in the literature captures all
aspects of the neutrino angular distribution in momentum space. In this paper,
we developed a closure relation by using the DNN that takes the neutrino energy
density, flux, and the fluid velocity as the input and the Eddington tensor as
the output. We consider two kinds of DNNs: a conventional DNN named a
component-wise neural network (CWNN) and a tensor-basis neural network (TBNN).
We found that the diagonal component of the Eddington tensor is reproduced
better by the DNNs than the M1-closure relation especially for low to
intermediate energies. For the off-diagonal component, the DNNs agree better
with the Boltzmann solver than the M1 closure at large radii. In the comparison
between the two DNNs, the TBNN has slightly better performance than the CWNN.
With the new closure relations at hand based on the DNNs that well reproduce
the Eddington tensor with much smaller costs, we opened up a new possibility
for the moment method.

    

### [[2105.14980] Crowdsourcing Learning as Domain Adaptation: A Case Study on Named Entity Recognition](http://arxiv.org/abs/2105.14980)


  Crowdsourcing is regarded as one prospective solution for effective
supervised learning, aiming to build large-scale annotated training data by
crowd workers. Previous studies focus on reducing the influences from the
noises of the crowdsourced annotations for supervised models. We take a
different point in this work, regarding all crowdsourced annotations as
gold-standard with respect to the individual annotators. In this way, we find
that crowdsourcing could be highly similar to domain adaptation, and then the
recent advances of cross-domain methods can be almost directly applied to
crowdsourcing. Here we take named entity recognition (NER) as a study case,
suggesting an annotator-aware representation learning model that inspired by
the domain adaptation methods which attempt to capture effective domain-aware
features. We investigate both unsupervised and supervised crowdsourcing
learning, assuming that no or only small-scale expert annotations are
available. Experimental results on a benchmark crowdsourced NER dataset show
that our method is highly effective, leading to a new state-of-the-art
performance. In addition, under the supervised setting, we can achieve
impressive performance gains with only a very small scale of expert
annotations.

    

### [[2106.02680] Smoothed Analysis for Orbit Recovery over $SO(3)$](http://arxiv.org/abs/2106.02680)


  In this work we study the orbit recovery problem over $SO(3)$, where the goal
is to recover a band-limited function on the sphere from noisy measurements of
randomly rotated copies of it. This is a natural abstraction for the problem of
recovering the three-dimensional structure of a molecule through cryo-electron
tomography. Symmetries play an important role: Recovering the function up to
rotation is equivalent to solving a system of polynomial equations that comes
from the invariant ring associated with the group action. Prior work
investigated this system through computational algebra tools up to a certain
size. However many statistical and algorithmic questions remain: How many
moments suffice for recovery, or equivalently at what degree do the invariant
polynomials generate the full invariant ring? And is it possible to
algorithmically solve this system of polynomial equations?
We revisit these problems from the perspective of smoothed analysis whereby
we perturb the coefficients of the function in the basis of spherical
harmonics. Our main result is a quasi-polynomial time algorithm for orbit
recovery over $SO(3)$ in this model. We analyze a popular heuristic called
frequency marching that exploits the layered structure of the system of
polynomial equations by setting up a system of {\em linear} equations to solve
for the higher-order frequencies assuming the lower-order ones have already
been found. The main questions are: Do these systems have a unique solution?
And how fast can the errors compound? Our main technical contribution is in
bounding the condition number of these algebraically-structured linear systems.
Thus smoothed analysis provides a compelling model in which we can expand the
types of group actions we can handle in orbit recovery, beyond the finite
and/or abelian case.

    

### [[2106.03034] Minibatch and Momentum Model-based Methods for Stochastic Weakly Convex Optimization](http://arxiv.org/abs/2106.03034)


  Stochastic model-based methods have received increasing attention lately due
to their appealing robustness to the stepsize selection and provable efficiency
guarantee. We make two important extensions for improving model-based methods
on stochastic weakly convex optimization. First, we propose new minibatch
model-based methods by involving a set of samples to approximate the model
function in each iteration. For the first time, we show that stochastic
algorithms achieve linear speedup over the batch size even for non-smooth and
non-convex (particularly, weakly convex) problems. To this end, we develop a
novel sensitivity analysis of the proximal mapping involved in each algorithm
iteration. Our analysis appears to be of independent interests in more general
settings. Second, motivated by the success of momentum stochastic gradient
descent, we propose a new stochastic extrapolated model-based method, greatly
extending the classic Polyak momentum technique to a wider class of stochastic
algorithms for weakly convex optimization. The rate of convergence to some
natural stationarity condition is established over a fairly flexible range of
extrapolation terms.
While mainly focusing on weakly convex optimization, we also extend our work
to convex optimization. We apply the minibatch and extrapolated model-based
methods to stochastic convex optimization, for which we provide a new
complexity bound and promising linear speedup in batch size. Moreover, an
accelerated model-based method based on Nesterov's momentum is presented, for
which we establish an optimal complexity bound for reaching optimality.

    

### [[2106.03746] Efficient Training of Visual Transformers with Small Datasets](http://arxiv.org/abs/2106.03746)


  Visual Transformers (VTs) are emerging as an architectural paradigm
alternative to Convolutional networks (CNNs). Differently from CNNs, VTs can
capture global relations between image elements and they potentially have a
larger representation capacity. However, the lack of the typical convolutional
inductive bias makes these models more data-hungry than common CNNs. In fact,
some local properties of the visual domain which are embedded in the CNN
architectural design, in VTs should be learned from samples. In this paper, we
empirically analyse different VTs, comparing their robustness in a small
training-set regime, and we show that, despite having a comparable accuracy
when trained on ImageNet, their performance on smaller datasets can be largely
different. Moreover, we propose a self-supervised task which can extract
additional information from images with only a negligible computational
overhead. This task encourages the VTs to learn spatial relations within an
image and makes the VT training much more robust when training data are scarce.
Our task is used jointly with the standard (supervised) training and it does
not depend on specific architectural choices, thus it can be easily plugged in
the existing VTs. Using an extensive evaluation with different VTs and
datasets, we show that our method can improve (sometimes dramatically) the
final accuracy of the VTs. Our code is available at:
this https URL.

    

### [[2106.03812] Scalable Computation of Monge Maps with General Costs](http://arxiv.org/abs/2106.03812)


  Monge map refers to the optimal transport map between two probability
distributions and provides a principled approach to transform one distribution
to another. In spite of the rapid developments of the numerical methods for
optimal transport problems, computing the Monge maps remains challenging,
especially for high dimensional problems. In this paper, we present a scalable
algorithm for computing the Monge map between two probability distributions.
Our algorithm is based on a weak form of the optimal transport problem, thus it
only requires samples from the marginals instead of their analytic expressions,
and can accommodate optimal transport between two distributions with different
dimensions. Our algorithm is suitable for general cost functions, compared with
other existing methods for estimating Monge maps using samples, which are
usually for quadratic costs. The performance of our algorithms is demonstrated
through a series of experiments with both synthetic and realistic data.

    

### [[2106.04986] Multistep Electric Vehicle Charging Station Occupancy Prediction using Hybrid LSTM Neural Networks](http://arxiv.org/abs/2106.04986)


  Public charging station occupancy prediction plays key importance in
developing a smart charging strategy to reduce electric vehicle (EV) operator
and user inconvenience. However, existing studies are mainly based on
conventional econometric or time series methodologies with limited accuracy. We
propose a new mixed long short-term memory neural network incorporating both
historical charging state sequences and time-related features for multistep
discrete charging occupancy state prediction. Unlike the existing LSTM
networks, the proposed model separates different types of features and handles
them differently with mixed neural network architecture. The model is compared
to a number of state-of-the-art machine learning and deep learning approaches
based on the EV charging data obtained from the open data portal of the city of
Dundee, UK. The results show that the proposed method produces very accurate
predictions (99.99% and 81.87% for 1 step (10 minutes) and 6 steps (1 hour)
ahead, respectively, and outperforms the benchmark approaches significantly
(+22.4% for one-step-ahead prediction and +6.2% for 6 steps ahead). A
sensitivity analysis is conducted to evaluate the impact of the model
parameters on prediction accuracy.

    

### [[2106.08598] Ada-BKB: Scalable Gaussian Process Optimization on Continuous Domain by Adaptive Discretization](http://arxiv.org/abs/2106.08598)


  Gaussian process optimization is a successful class of algorithms(e.g.
GP-UCB) to optimize a black-box function through sequential evaluations.
However, for functions with continuous domains, Gaussian process optimization
has to rely on either a fixed discretization of the space, or the solution of a
non-convex optimization subproblem at each evaluation. The first approach can
negatively affect performance, while the second approach puts requires a heavy
computational burden. A third option, only recently theoretically studied, is
to adaptively discretize the function domain. Even though this approach avoids
the extra non-convex optimization costs, the overall computational complexity
is still prohibitive. An algorithm such as GP-UCB has a runtime of $O(T^4)$,
where $T$ is the number of iterations. In this paper, we introduce Ada-BKB
(Adaptive Budgeted Kernelized Bandit), a no-regret Gaussian process
optimization algorithm for functions on continuous domains, that provably runs
in $O(T^2 d_\text{eff}^2)$, where $d_\text{eff}$ is the effective dimension of
the explored space, and which is typically much smaller than $T$. We
corroborate our theoretical findings with experiments on synthetic non-convex
functions and on the real-world problem of hyper-parameter optimization,
confirming the good practical performances of the proposed approach.

    

### [[2106.10064] Fitting summary statistics of neural data with a differentiable spiking network simulator](http://arxiv.org/abs/2106.10064)


  Fitting network models to neural activity is an important tool in
neuroscience. A popular approach is to model a brain area with a probabilistic
recurrent spiking network whose parameters maximize the likelihood of the
recorded activity. Although this is widely used, we show that the resulting
model does not produce realistic neural activity. To correct for this, we
suggest to augment the log-likelihood with terms that measure the dissimilarity
between simulated and recorded activity. This dissimilarity is defined via
summary statistics commonly used in neuroscience and the optimization is
efficient because it relies on back-propagation through the stochastically
simulated spike trains. We analyze this method theoretically and show
empirically that it generates more realistic activity statistics. We find that
it improves upon other fitting algorithms for spiking network models like GLMs
(Generalized Linear Models) which do not usually rely on back-propagation. This
new fitting algorithm also enables the consideration of hidden neurons which is
otherwise notoriously hard, and we show that it can be crucial when trying to
infer the network connectivity from spike recordings.

    

### [[2106.11879] Asynchronous Stochastic Optimization Robust to Arbitrary Delays](http://arxiv.org/abs/2106.11879)


  We consider stochastic optimization with delayed gradients where, at each
time step $t$, the algorithm makes an update using a stale stochastic gradient
from step $t - d_t$ for some arbitrary delay $d_t$. This setting abstracts
asynchronous distributed optimization where a central server receives gradient
updates computed by worker machines. These machines can experience computation
and communication loads that might vary significantly over time. In the general
non-convex smooth optimization setting, we give a simple and efficient
algorithm that requires $O( \sigma^2/\epsilon^4 + \tau/\epsilon^2 )$ steps for
finding an $\epsilon$-stationary point $x$, where $\tau$ is the \emph{average}
delay $\smash{\frac{1}{T}\sum_{t=1}^T d_t}$ and $\sigma^2$ is the variance of
the stochastic gradients. This improves over previous work, which showed that
stochastic gradient decent achieves the same rate but with respect to the
\emph{maximal} delay $\max_{t} d_t$, that can be significantly larger than the
average delay especially in heterogeneous distributed systems. Our experiments
demonstrate the efficacy and robustness of our algorithm in cases where the
delay distribution is skewed or heavy-tailed.

    

### [[2106.13638] Transient Stability Analysis with Physics-Informed Neural Networks](http://arxiv.org/abs/2106.13638)


  We explore the possibility to use physics-informed neural networks to
drastically accelerate the solution of ordinary differential-algebraic
equations that govern the power system dynamics. When it comes to transient
stability assessment, the traditionally applied methods either carry a
significant computational burden, require model simplifications, or use overly
conservative surrogate models. Conventional neural networks can circumvent
these limitations but are faced with high demand of high-quality training
datasets, while they ignore the underlying governing equations.
Physics-informed neural networks are different: they incorporate the power
system differential algebraic equations directly into the neural network
training and drastically reduce the need for training data. This paper takes a
deep dive into the performance of physics-informed neural networks for power
system transient stability assessment. Introducing a new neural network
training procedure to facilitate a thorough comparison, we explore how
physics-informed neural networks compare with conventional
differential-algebraic solvers and classical neural networks in terms of
computation time, requirements in data, and prediction accuracy. We illustrate
the findings on the Kundur two-area system, and assess the opportunities and
challenges of physics-informed neural networks to serve as a transient
stability analysis tool, highlighting possible pathways to further develop this
method.

    

### [[2106.14483] Cheating Detection Pipeline for Online Interviews and Exams](http://arxiv.org/abs/2106.14483)


  Remote examination and job interviews have gained popularity and become
indispensable because of both pandemics and the advantage of remote working
circumstances. Most companies and academic institutions utilize these systems
for their recruitment processes and also for online exams. However, one of the
critical problems of the remote examination systems is conducting the exams in
a reliable environment. In this work, we present a cheating analysis pipeline
for online interviews and exams. The system only requires a video of the
candidate, which is recorded during the exam. Then cheating detection pipeline
is employed to detect another person, electronic device usage, and candidate
absence status. The pipeline consists of face detection, face recognition,
object detection, and face tracking algorithms. To evaluate the performance of
the pipeline we collected a private video dataset. The video dataset includes
both cheating activities and clean videos. Ultimately, our pipeline presents an
efficient and fast guideline to detect and analyze cheating activities in an
online interview and exam video.

    

### [[2109.09463] Predicting Visual Improvement after Macular Hole Surgery: a Cautionary Tale on Deep Learning with Very Limited Data](http://arxiv.org/abs/2109.09463)


  We investigate the potential of machine learning models for the prediction of
visual improvement after macular hole surgery from preoperative data (retinal
images and clinical features). Collecting our own data for the task, we end up
with only 121 total samples, putting our work in the very limited data regime.
We explore a variety of deep learning methods for limited data to train deep
computer vision models, finding that all tested deep vision models are
outperformed by a simple regression model on the clinical features. We believe
this is compelling evidence of the extreme difficulty of using deep learning on
very limited data.

    

### [[2110.13101] Latent-Insensitive autoencoders for Anomaly Detection](http://arxiv.org/abs/2110.13101)


  Reconstruction-based approaches to anomaly detection tend to fall short when
applied to complex datasets with target classes that possess high inter-class
variance. Similar to the idea of self-taught learning used in transfer
learning, many domains are rich with similar unlabelled datasets that could be
leveraged as a proxy for out-of-distribution samples. In this paper we
introduce Latent-Insensitive autoencoder (LIS-AE) where unlabeled data from a
similar domain is utilized as negative examples to shape the latent layer
(bottleneck) of a regular autoencoder such that it is only capable of
reconstructing one task. We provide theoretical justification for the proposed
training process and loss functions along with an extensive ablation study
highlighting important aspects of our model. We test our model in multiple
anomaly detection settings presenting quantitative and qualitative analysis
showcasing the significant performance improvement of our model for anomaly
detection tasks.

    

### [[1903.09688] Symbolic Regression Methods for Reinforcement Learning](http://arxiv.org/abs/1903.09688)


  Reinforcement learning algorithms can solve dynamic decision-making and
optimal control problems. With continuous-valued state and input variables,
reinforcement learning algorithms must rely on function approximators to
represent the value function and policy mappings. Commonly used numerical
approximators, such as neural networks or basis function expansions, have two
main drawbacks: they are black-box models offering little insight into the
mappings learned, and they require extensive trial and error tuning of their
hyper-parameters. In this paper, we propose a new approach to constructing
smooth value functions in the form of analytic expressions by using symbolic
regression. We introduce three off-line methods for finding value functions
based on a state-transition model: symbolic value iteration, symbolic policy
iteration, and a direct solution of the Bellman equation. The methods are
illustrated on four nonlinear control problems: velocity control under
friction, one-link and two-link pendulum swing-up, and magnetic manipulation.
The results show that the value functions yield well-performing policies and
are compact, mathematically tractable, and easy to plug into other algorithms.
This makes them potentially suitable for further analysis of the closed-loop
system. A comparison with an alternative approach using neural networks shows
that our method outperforms the neural network-based one.

    

### [[2111.06818] Dynamic treatment effects: high-dimensional inference under model misspecification](http://arxiv.org/abs/2111.06818)


  This paper considers the inference for heterogeneous treatment effects in
dynamic settings that covariates and treatments are longitudinal. We focus on
high-dimensional cases that the sample size, $N$, is potentially much larger
than the covariate vector's dimension, $d$. The marginal structural mean models
are considered. We propose a "sequential model doubly robust" estimator
constructed based on "moment targeted" nuisance estimators. Such nuisance
estimators are carefully designed through non-standard loss functions, reducing
the bias resulting from potential model misspecifications. We achieve $\sqrt
N$-inference even when model misspecification occurs. We only require one
nuisance model to be correctly specified at each time spot. Such model
correctness conditions are weaker than all the existing work, even containing
the literature on low dimensions.

    

### [[2111.07584] Design and Evaluation Frameworks for Advanced RISC-based Ternary Processor](http://arxiv.org/abs/2111.07584)


  In this paper, we introduce the design and verification frameworks for
developing a fully-functional emerging ternary processor. Based on the existing
compiling environments for binary processors, for the given ternary
instructions, the software-level framework provides an efficient way to convert
the given programs to the ternary assembly codes. We also present a
hardware-level framework to rapidly evaluate the performance of a ternary
processor implemented in arbitrary design technology. As a case study, the
fully-functional 9-trit advanced RISC-based ternary (ART-9) core is newly
developed by using the proposed frameworks. Utilizing 24 custom ternary
instructions, the 5-stage ART-9 prototype architecture is successfully verified
by a number of test programs including dhrystone benchmark in a ternary domain,
achieving the processing efficiency of 57.8 DMIPS/W and 3.06 x 10^6 DMIPS/W in
the FPGA-level ternary-logic emulations and the emerging CNTFET ternary gates,
respectively.

    

### [[2010.10416] Composite Enclaves: Towards Disaggregated Trusted Execution](http://arxiv.org/abs/2010.10416)


  The ever-rising computation demand is forcing the move from the CPU to
heterogeneous specialized hardware, which is readily available across modern
datacenters through disaggregated infrastructure. On the other hand, trusted
execution environments (TEEs), one of the most promising recent developments in
hardware security, can only protect code confined in the CPU, limiting TEEs'
potential and applicability to a handful of applications. We observe that the
TEEs' hardware trusted computing base (TCB) is fixed at design time, which in
practice leads to using untrusted software to employ peripherals in TEEs. Based
on this observation, we propose \emph{composite enclaves} with a configurable
hardware and software TCB, allowing enclaves access to multiple computing and
IO resources. Finally, we present two case studies of composite enclaves: i) an
FPGA platform based on RISC-V Keystone connected to emulated peripherals and
sensors, and ii) a large-scale accelerator. These case studies showcase a
flexible but small TCB (2.5 KLoC for IO peripherals and drivers), with a
low-performance overhead (only around 220 additional cycles for a context
switch), thus demonstrating the feasibility of our approach and showing that it
can work with a wide range of specialized hardware.

    

### [[2111.07226] Practical Scheduling for Real-World Serverless Computing](http://arxiv.org/abs/2111.07226)


  Serverless computing has seen rapid growth due to the ease-of-use and
cost-efficiency it provides. However, function scheduling, a critical component
of serverless systems, has been overlooked. In this paper, we take a
first-principles approach toward designing a scheduler that caters to the
unique characteristics of serverless functions as seen in real-world
deployments. We first create a taxonomy of scheduling policies along three
dimensions. Next, we use simulation to explore the scheduling policy space for
the function characteristics in a 14-day trace of Azure functions and conclude
that frequently used features such as late binding and random load balancing
are sub-optimal for common execution time distributions and load ranges. We use
these insights to design Hermes, a scheduler for serverless functions with
three key characteristics. First, to avoid head-of-line blocking due to high
function execution time variability, Hermes uses a combination of early binding
and processor sharing for scheduling at individual worker machines. Second,
Hermes uses a hybrid load balancing approach that improves consolidation at low
load while employing least-loaded balancing at high load to retain high
performance. Third, Hermes is both load and locality-aware, reducing the number
of cold starts compared to pure load-based policies. We implement Hermes for
Apache OpenWhisk and demonstrate that, for the case of the function patterns
observed both in the Azure and in other real-world traces, it achieves up to
85% lower function slowdown and 60% higher throughput compared to existing
policies.

    

### [[2111.07528] Composing Energy Services in a Crowdsourced IoT Environment](http://arxiv.org/abs/2111.07528)


  We propose a novel framework for composing crowdsourced wireless energy
services to satisfy users' energy requirements in a crowdsourced Internet of
Things (IoT) environment. A new energy service model is designed to transform
the harvested energy from IoT devices into crowdsourced services. We propose a
new energy service composability model that considers the spatio-temporal
aspects and the usage patterns of the IoT devices. A multiple local
knapsack-based approach is developed to select an optimal set of partial energy
services based on the deliverable energy capacity of IoT devices. We propose a
heuristic-based composition approach using the temporal and energy capacity
distributions of services. Experimental results demonstrate the effectiveness
and efficiency of the proposed approach.

    

### [[2111.07672] A Data Quarantine Model to Secure Data in Edge Computing](http://arxiv.org/abs/2111.07672)


  Edge computing provides an agile data processing platform for
latency-sensitive and communication-intensive applications through a
decentralized cloud and geographically distributed edge nodes. Gaining
centralized control over the edge nodes can be challenging due to security
issues and threats. Among several security issues, data integrity attacks can
lead to inconsistent data and intrude edge data analytics. Further
intensification of the attack makes it challenging to mitigate and identify the
root cause. Therefore, this paper proposes a new concept of data quarantine
model to mitigate data integrity attacks by quarantining intruders. The
efficient security solutions in cloud, ad-hoc networks, and computer systems
using quarantine have motivated adopting it in edge computing. The data
acquisition edge nodes identify the intruders and quarantine all the suspected
devices through dimensionality reduction. During quarantine, the proposed
concept builds the reputation scores to determine the falsely identified
legitimate devices and sanitize their affected data to regain data integrity.
As a preliminary investigation, this work identifies an appropriate machine
learning method, Linear Discriminant Analysis (LDA), for dimensionality
reduction. The LDA results in 72.83% quarantine accuracy and 0.9 seconds
training time, which is efficient than other state-of-the-art methods. In
future, this would be implemented and validated with ground truth data.

    

### [[2007.03972] Secure Distributed Matrix Computation with Discrete Fourier Transform](http://arxiv.org/abs/2007.03972)


  We consider the problem of secure distributed matrix computation (SDMC),
where a \textit{user} queries a function of data matrices generated at
distributed \textit{source} nodes. We assume the availability of $N$ honest but
curious computation servers, which are connected to the sources, the user, and
each other through orthogonal and reliable communication links. Our goal is to
minimize the amount of data that must be transmitted from the sources to the
servers, called the \textit{upload cost}, while guaranteeing that no $T$
colluding servers can learn any information about the source matrices, and the
user cannot learn any information beyond the computation result. We first focus
on secure distributed matrix multiplication (SDMM), considering two matrices,
and propose a novel polynomial coding scheme using the properties of finite
field discrete Fourier transform, which achieves an upload cost significantly
lower than the existing results in the literature. We then generalize the
proposed scheme to include straggler mitigation, and to the multiplication of
multiple matrices while keeping the input matrices, the intermediate
computation results, as well as the final result secure against any $T$
colluding servers. We also consider a special case, called computation with own
data, where the data matrices used for computation belong to the user. In this
case, we drop the security requirement against the user, and show that the
proposed scheme achieves the minimal upload cost. We then propose methods for
performing other common matrix computations securely on distributed servers,
including changing the parameters of secret sharing, matrix transpose, matrix
exponentiation, solving a linear system, and matrix inversion, which are then
used to show how arbitrary matrix polynomials can be computed securely on
distributed servers using the proposed procedure.

    

### [[2106.06022] IoT Virtualization with ML-based Information Extraction](http://arxiv.org/abs/2106.06022)


  For IoT to reach its full potential, the sharing and reuse of information in
different applications and across verticals is of paramount importance.
However, there are a plethora of IoT platforms using different representations,
protocols and interaction patterns. To address this issue, the Fed4IoT project
has developed an IoT virtualization platform that, on the one hand, integrates
information from many different source platforms and, on the other hand, makes
the information required by the respective users available in the target
platform of choice. To enable this, information is translated into a common,
neutral exchange format. The format of choice is NGSI-LD, which is being
standardized by the ETSI Industry Specification Group on Context Information
Management (ETSI ISG CIM). Thing Visors are the components that translate the
source information to NGSI-LD, which is then delivered to the target platform
and translated into the target format. ThingVisors can be implemented by hand,
but this requires significant human effort, especially considering the
heterogeneity of low level information produced by a multitude of sensors.
Thus, supporting the human developer and, ideally, fully automating the process
of extracting and enriching data and translating it to NGSI-LD is a crucial
step. Machine learning is a promising approach for this, but it typically
requires large amounts of hand-labelled data for training, an effort that makes
it unrealistic in many IoT scenarios. A programmatic labelling approach called
knowledge infusion that encodes expert knowledge is used for matching a schema
or ontology extracted from the data with a target schema or ontology, providing
the basis for annotating the data and facilitating the translation to NGSI-LD.

    

### [[2106.16002] Distributed Nash Equilibrium Seeking under Quantization Communication](http://arxiv.org/abs/2106.16002)


  This paper investigates Nash equilibrium (NE) seeking problems for
noncooperative games over multi-players networks with finite bandwidth
communication. A distributed quantized algorithm is presented, which consists
of local gradient play, distributed decision estimating, and adaptive
quantization. Exponential convergence of the algorithm is established, and a
relationship between the convergence rate and the bandwidth is quantitatively
analyzed. Finally, a simulation of an energy consumption game is presented to
validate the proposed results.

    

### [[2111.06902] MS-LaTTE: A Dataset of Where and When To-do Tasks are Completed](http://arxiv.org/abs/2111.06902)


  Tasks are a fundamental unit of work in the daily lives of people, who are
increasingly using digital means to keep track of, organize, triage and act on
them. These digital tools -- such as task management applications -- provide a
unique opportunity to study and understand tasks and their connection to the
real world, and through intelligent assistance, help people be more productive.
By logging signals such as text, timestamp information, and social connectivity
graphs, an increasingly rich and detailed picture of how tasks are created and
organized, what makes them important, and who acts on them, can be
progressively developed. Yet the context around actual task completion remains
fuzzy, due to the basic disconnect between actions taken in the real world and
telemetry recorded in the digital world. Thus, in this paper we compile and
release a novel, real-life, large-scale dataset called MS-LaTTE that captures
two core aspects of the context surrounding task completion: location and time.
We describe our annotation framework and conduct a number of analyses on the
data that were collected, demonstrating that it captures intuitive contextual
properties for common tasks. Finally, we test the dataset on the two problems
of predicting spatial and temporal task co-occurrence, concluding that
predictors for co-location and co-time are both learnable, with a BERT
fine-tuned model outperforming several other baselines. The MS-LaTTE dataset
provides an opportunity to tackle many new modeling challenges in contextual
task understanding and we hope that its release will spur future research in
task intelligence more broadly.

    

### [[2111.06908] Explainable AI for Psychological Profiling from Digital Footprints: A Case Study of Big Five Personality Predictions from Spending Data](http://arxiv.org/abs/2111.06908)


  Every step we take in the digital world leaves behind a record of our
behavior; a digital footprint. Research has suggested that algorithms can
translate these digital footprints into accurate estimates of psychological
characteristics, including personality traits, mental health or intelligence.
The mechanisms by which AI generates these insights, however, often remain
opaque. In this paper, we show how Explainable AI (XAI) can help domain experts
and data subjects validate, question, and improve models that classify
psychological traits from digital footprints. We elaborate on two popular XAI
methods (rule extraction and counterfactual explanations) in the context of Big
Five personality predictions (traits and facets) from financial transactions
data (N = 6,408). First, we demonstrate how global rule extraction sheds light
on the spending patterns identified by the model as most predictive for
personality, and discuss how these rules can be used to explain, validate, and
improve the model. Second, we implement local rule extraction to show that
individuals are assigned to personality classes because of their unique
financial behavior, and that there exists a positive link between the model's
prediction confidence and the number of features that contributed to the
prediction. Our experiments highlight the importance of both global and local
XAI methods. By better understanding how predictive models work in general as
well as how they derive an outcome for a particular person, XAI promotes
accountability in a world in which AI impacts the lives of billions of people
around the world.

    

### [[2111.06928] Generalized Nested Rollout Policy Adaptation with Dynamic Bias for Vehicle Routing](http://arxiv.org/abs/2111.06928)


  In this paper we present an extension of the Nested Rollout Policy Adaptation
algorithm (NRPA), namely the Generalized Nested Rollout Policy Adaptation
(GNRPA), as well as its use for solving some instances of the Vehicle Routing
Problem. We detail some results obtained on the Solomon instances set which is
a conventional benchmark for the Vehicle Routing Problem (VRP). We show that on
all instances, GNRPA performs better than NRPA. On some instances, it performs
better than the Google OR Tool module dedicated to VRP.

    

### [[2111.06958] Computational Argumentation and Cognition](http://arxiv.org/abs/2111.06958)


  This paper examines the interdisciplinary research question of how to
integrate Computational Argumentation, as studied in AI, with Cognition, as can
be found in Cognitive Science, Linguistics, and Philosophy. It stems from the
work of the 1st Workshop on Computational Argumentation and Cognition
(COGNITAR), which was organized as part of the 24th European Conference on
Artificial Intelligence (ECAI), and took place virtually on September 8th,
2020. The paper begins with a brief presentation of the scientific motivation
for the integration of Computational Argumentation and Cognition, arguing that
within the context of Human-Centric AI the use of theory and methods from
Computational Argumentation for the study of Cognition can be a promising
avenue to pursue. A short summary of each of the workshop presentations is
given showing the wide spectrum of problems where the synthesis of the theory
and methods of Computational Argumentation with other approaches that study
Cognition can be applied. The paper presents the main problems and challenges
in the area that would need to be addressed, both at the scientific level but
also at the epistemological level, particularly in relation to the synthesis of
ideas and approaches from the various disciplines involved.

    

### [[2111.07036] Introducing Variational Autoencoders to High School Students](http://arxiv.org/abs/2111.07036)


  Generative Artificial Intelligence (AI) models are a compelling way to
introduce K-12 students to AI education using an artistic medium, and hence
have drawn attention from K-12 AI educators. Previous Creative AI curricula
mainly focus on Generative Adversarial Networks (GANs) while paying less
attention to Autoregressive Models, Variational Autoencoders (VAEs), or other
generative models, which have since become common in the field of generative
AI. VAEs' latent-space structure and interpolation ability could effectively
ground the interdisciplinary learning of AI, creative arts, and philosophy.
Thus, we designed a lesson to teach high school students about VAEs. We
developed a web-based game and used Plato's cave, a philosophical metaphor, to
introduce how VAEs work. We used a Google Colab notebook for students to
re-train VAEs with their hand-written digits to consolidate their
understandings. Finally, we guided the exploration of creative VAE tools such
as SketchRNN and MusicVAE to draw the connection between what they learned and
real-world applications. This paper describes the lesson design and shares
insights from the pilot studies with 22 students. We found that our approach
was effective in teaching students about a novel AI concept.

    

### [[2111.07037] Obstacle Avoidance for UAS in Continuous Action Space Using Deep Reinforcement Learning](http://arxiv.org/abs/2111.07037)


  Obstacle avoidance for small unmanned aircraft is vital for the safety of
future urban air mobility (UAM) and Unmanned Aircraft System (UAS) Traffic
Management (UTM). There are many techniques for real-time robust drone
guidance, but many of them solve in discretized airspace and control, which
would require an additional path smoothing step to provide flexible commands
for UAS. To provide a safe and efficient computational guidance of operations
for unmanned aircraft, we explore the use of a deep reinforcement learning
algorithm based on Proximal Policy Optimization (PPO) to guide autonomous UAS
to their destinations while avoiding obstacles through continuous control. The
proposed scenario state representation and reward function can map the
continuous state space to continuous control for both heading angle and speed.
To verify the performance of the proposed learning framework, we conducted
numerical experiments with static and moving obstacles. Uncertainties
associated with the environments and safety operation bounds are investigated
in detail. Results show that the proposed model can provide accurate and robust
guidance and resolve conflict with a success rate of over 99%.

    

### [[2111.07039] UET-Headpose: A sensor-based top-view head pose dataset](http://arxiv.org/abs/2111.07039)


  Head pose estimation is a challenging task that aims to solve problems
related to predicting three dimensions vector, that serves for many
applications in human-robot interaction or customer behavior. Previous
researches have proposed some precise methods for collecting head pose data.
But those methods require either expensive devices like depth cameras or
complex laboratory environment setup. In this research, we introduce a new
approach with efficient cost and easy setup to collecting head pose images,
namely UET-Headpose dataset, with top-view head pose data. This method uses an
absolute orientation sensor instead of Depth cameras to be set up quickly and
small cost but still ensure good results. Through experiments, our dataset has
been shown the difference between its distribution and available dataset like
CMU Panoptic Dataset \cite{CMU}. Besides using the UET-Headpose dataset and
other head pose datasets, we also introduce the full-range model called
FSANet-Wide, which significantly outperforms head pose estimation results by
the UET-Headpose dataset, especially on top-view images. Also, this model is
very lightweight and takes small size images.

    

### [[2111.07102] Deep Neural Networks for Automatic Grain-matrix Segmentation in Plane and Cross-polarized Sandstone Photomicrographs](http://arxiv.org/abs/2111.07102)


  Grain segmentation of sandstone that is partitioning the grain from its
surrounding matrix/cement in the thin section is the primary step for
computer-aided mineral identification and sandstone classification. The
microscopic images of sandstone contain many mineral grains and their
surrounding matrix/cement. The distinction between adjacent grains and the
matrix is often ambiguous, making grain segmentation difficult. Various
solutions exist in literature to handle these problems; however, they are not
robust against sandstone petrography's varied pattern. In this paper, we
formulate grain segmentation as a pixel-wise two-class (i.e., grain and
background) semantic segmentation task. We develop a deep learning-based
end-to-end trainable framework named Deep Semantic Grain Segmentation network
(DSGSN), a data-driven method, and provide a generic solution. As per the
authors' knowledge, this is the first work where the deep neural network is
explored to solve the grain segmentation problem. Extensive experiments on
microscopic images highlight that our method obtains better segmentation
accuracy than various segmentation architectures with more parameters.

    

### [[2111.07129] Visual Understanding of Complex Table Structures from Document Images](http://arxiv.org/abs/2111.07129)


  Table structure recognition is necessary for a comprehensive understanding of
documents. Tables in unstructured business documents are tough to parse due to
the high diversity of layouts, varying alignments of contents, and the presence
of empty cells. The problem is particularly difficult because of challenges in
identifying individual cells using visual or linguistic contexts or both.
Accurate detection of table cells (including empty cells) simplifies structure
extraction and hence, it becomes the prime focus of our work. We propose a
novel object-detection-based deep model that captures the inherent alignments
of cells within tables and is fine-tuned for fast optimization. Despite
accurate detection of cells, recognizing structures for dense tables may still
be challenging because of difficulties in capturing long-range row/column
dependencies in presence of multi-row/column spanning cells. Therefore, we also
aim to improve structure recognition by deducing a novel rectilinear
graph-based formulation. From a semantics perspective, we highlight the
significance of empty cells in a table. To take these cells into account, we
suggest an enhancement to a popular evaluation criterion. Finally, we introduce
a modestly sized evaluation dataset with an annotation style inspired by human
cognition to encourage new approaches to the problem. Our framework improves
the previous state-of-the-art performance by a 2.7% average F1-score on
benchmark datasets.

    

### [[2111.07145] New Performance Measures for Object Tracking under Complex Environments](http://arxiv.org/abs/2111.07145)


  Various performance measures based on the ground truth and without ground
truth exist to evaluate the quality of a developed tracking algorithm. The
existing popular measures - average center location error (ACLE) and average
tracking accuracy (ATA) based on ground truth, may sometimes create confusion
to quantify the quality of a developed algorithm for tracking an object under
some complex environments (e.g., scaled or oriented or both scaled and oriented
object). In this article, we propose three new auxiliary performance measures
based on ground truth information to evaluate the quality of a developed
tracking algorithm under such complex environments. Moreover, one performance
measure is developed by combining both two existing measures ACLE and ATA and
three new proposed measures for better quantifying the developed tracking
algorithm under such complex conditions. Some examples and experimental results
conclude that the proposed measure is better than existing measures to quantify
one developed algorithm for tracking objects under such complex environments.

    

### [[2111.07148] SocialBERT -- Transformers for Online SocialNetwork Language Modelling](http://arxiv.org/abs/2111.07148)


  The ubiquity of the contemporary language understanding tasks gives relevance
to the development of generalized, yet highly efficient models that utilize all
knowledge, provided by the data source. In this work, we present SocialBERT -
the first model that uses knowledge about the author's position in the network
during text analysis. We investigate possible models for learning social
network information and successfully inject it into the baseline BERT model.
The evaluation shows that embedding this information maintains a good
generalization, with an increase in the quality of the probabilistic model for
the given author up to 7.5%. The proposed model has been trained on the
majority of groups for the chosen social network, and still able to work with
previously unknown groups. The obtained model, as well as the code of our
experiments, is available for download and use in applied tasks.

    

### [[2111.07154] Session-aware Item-combination Recommendation with Transformer Network](http://arxiv.org/abs/2111.07154)


  In this paper, we detailedly describe our solution for the IEEE BigData Cup
2021: RL-based RecSys (Track 1: Item Combination Prediction). We first conduct
an exploratory data analysis on the dataset and then utilize the findings to
design our framework. Specifically, we use a two-headed transformer-based
network to predict user feedback and unlocked sessions, along with the proposed
session-aware reweighted loss, multi-tasking with click behavior prediction,
and randomness-in-session augmentation. In the final private leaderboard on
Kaggle, our method ranked 2nd with a categorization accuracy of 0.39224.

    

### [[2111.07158] Robust Deep Reinforcement Learning for Extractive Legal Summarization](http://arxiv.org/abs/2111.07158)


  Automatic summarization of legal texts is an important and still a
challenging task since legal documents are often long and complicated with
unusual structures and styles. Recent advances of deep models trained
end-to-end with differentiable losses can well-summarize natural text, yet when
applied to legal domain, they show limited results. In this paper, we propose
to use reinforcement learning to train current deep summarization models to
improve their performance on the legal domain. To this end, we adopt proximal
policy optimization methods and introduce novel reward functions that encourage
the generation of candidate summaries satisfying both lexical and semantic
criteria. We apply our method to training different summarization backbones and
observe a consistent and significant performance gain across 3 public legal
datasets.

    

### [[2111.07224] Local Multi-Head Channel Self-Attention for Facial Expression Recognition](http://arxiv.org/abs/2111.07224)


  Since the Transformer architecture was introduced in 2017 there has been many
attempts to bring the self-attention paradigm in the field of computer vision.
In this paper we propose a novel self-attention module that can be easily
integrated in virtually every convolutional neural network and that is
specifically designed for computer vision, the LHC: Local (multi) Head Channel
(self-attention). LHC is based on two main ideas: first, we think that in
computer vision the best way to leverage the self-attention paradigm is the
channel-wise application instead of the more explored spatial attention and
that convolution will not be replaced by attention modules like recurrent
networks were in NLP; second, a local approach has the potential to better
overcome the limitations of convolution than global attention. With LHC-Net we
managed to achieve a new state of the art in the famous FER2013 dataset with a
significantly lower complexity and impact on the ``host'' architecture in terms
of computational cost when compared with the previous SOTA.

    

### [[2111.07238] FACOS: Finding API Relevant Contents on Stack Overflow with Semantic and Syntactic Analysis](http://arxiv.org/abs/2111.07238)


  Collecting API examples, usages, and mentions relevant to a specific API
method over discussions on venues such as Stack Overflow is not a trivial
problem. It requires efforts to correctly recognize whether the discussion
refers to the API method that developers/tools are searching for. The content
of the thread, which consists of both text paragraphs describing the
involvement of the API method in the discussion and the code snippets
containing the API invocation, may refer to the given API method. Leveraging
this observation, we develop FACOS, a context-specific algorithm to capture the
semantic and syntactic information of the paragraphs and code snippets in a
discussion. FACOS combines a syntactic word-based score with a score from a
predictive model fine-tuned from CodeBERT. FACOS beats the state-of-the-art
approach by 13.9% in terms of F1-score.

    

### [[2111.07263] Code Representation Learning with Prüfer Sequences](http://arxiv.org/abs/2111.07263)


  An effective and efficient encoding of the source code of a computer program
is critical to the success of sequence-to-sequence deep neural network models
for tasks in computer program comprehension, such as automated code
summarization and documentation. A significant challenge is to find a
sequential representation that captures the structural/syntactic information in
a computer program and facilitates the training of the learning models.
In this paper, we propose to use the Prüfer sequence of the Abstract Syntax
Tree (AST) of a computer program to design a sequential representation scheme
that preserves the structural information in an AST. Our representation makes
it possible to develop deep-learning models in which signals carried by lexical
tokens in the training examples can be exploited automatically and selectively
based on their syntactic role and importance. Unlike other recently-proposed
approaches, our representation is concise and lossless in terms of the
structural information of the AST. Empirical studies on real-world benchmark
datasets, using a sequence-to-sequence learning model we designed for code
summarization, show that our Prüfer-sequence-based representation is indeed
highly effective and efficient, outperforming significantly all the
recently-proposed deep-learning models we used as the baseline models.

    

### [[2111.07267] CDM: Combining Extraction and Generation for Definition Modeling](http://arxiv.org/abs/2111.07267)


  Definitions are essential for term understanding. Recently, there is an
increasing interest in extracting and generating definitions of terms
automatically. However, existing approaches for this task are either extractive
or abstractive - definitions are either extracted from a corpus or generated by
a language generation model. In this paper, we propose to combine extraction
and generation for definition modeling: first extract self- and correlative
definitional information of target terms from the Web and then generate the
final definitions by incorporating the extracted definitional information.
Experiments demonstrate our framework can generate high-quality definitions for
technical terms and outperform state-of-the-art models for definition modeling
significantly.

    

### [[2111.07308] What Should We Optimize in Participatory Budgeting? An Experimental Study](http://arxiv.org/abs/2111.07308)


  Participatory Budgeting (PB) is a process in which voters decide how to
allocate a common budget; most commonly it is done by ordinary people -- in
particular, residents of some municipality -- to decide on a fraction of the
municipal budget. From a social choice perspective, existing research on PB
focuses almost exclusively on designing computationally-efficient aggregation
methods that satisfy certain axiomatic properties deemed "desirable" by the
research community. Our work complements this line of research through a user
study (N = 215) involving several experiments aimed at identifying what
potential voters (i.e., non-experts) deem fair or desirable in simple PB
settings. Our results show that some modern PB aggregation techniques greatly
differ from users' expectations, while other, more standard approaches, provide
more aligned results. We also identify a few possible discrepancies between
what non-experts consider \say{desirable} and how they perceive the notion of
"fairness" in the PB context. Taken jointly, our results can be used to help
the research community identify appropriate PB aggregation methods to use in
practice.

    

### [[2111.07393] DEEP: DEnoising Entity Pre-training for Neural Machine Translation](http://arxiv.org/abs/2111.07393)


  It has been shown that machine translation models usually generate poor
translations for named entities that are infrequent in the training corpus.
Earlier named entity translation methods mainly focus on phonetic
transliteration, which ignores the sentence context for translation and is
limited in domain and language coverage. To address this limitation, we propose
DEEP, a DEnoising Entity Pre-training method that leverages large amounts of
monolingual data and a knowledge base to improve named entity translation
accuracy within sentences. Besides, we investigate a multi-task learning
strategy that finetunes a pre-trained neural machine translation model on both
entity-augmented monolingual data and parallel data to further improve entity
translation. Experimental results on three language pairs demonstrate that
\method results in significant improvements over strong denoising auto-encoding
baselines, with a gain of up to 1.3 BLEU and up to 9.2 entity accuracy points
for English-Russian translation.

    

### [[2111.07441] A distributed, plug-n-play algorithm for multi-robot applications with a priori non-computable objective functions](http://arxiv.org/abs/2111.07441)


  This paper presents a distributed algorithm applicable to a wide range of
practical multi-robot applications. In such multi-robot applications, the
user-defined objectives of the mission can be cast as a general optimization
problem, without explicit guidelines of the subtasks per different robot. Owing
to the unknown environment, unknown robot dynamics, sensor nonlinearities,
etc., the analytic form of the optimization cost function is not available a
priori. Therefore, standard gradient-descent-like algorithms are not applicable
to these problems. To tackle this, we introduce a new algorithm that carefully
designs each robot's subcost function, the optimization of which can accomplish
the overall team objective. Upon this transformation, we propose a distributed
methodology based on the cognitive-based adaptive optimization (CAO) algorithm,
that is able to approximate the evolution of each robot's cost function and to
adequately optimize its decision variables (robot actions). The latter can be
achieved by online learning only the problem-specific characteristics that
affect the accomplishment of mission objectives. The overall, low-complexity
algorithm can straightforwardly incorporate any kind of operational constraint,
is fault tolerant, and can appropriately tackle time-varying cost functions. A
cornerstone of this approach is that it shares the same convergence
characteristics as those of block coordinate descent algorithms. The proposed
algorithm is evaluated in three heterogeneous simulation set-ups under multiple
scenarios, against both general-purpose and problem-specific algorithms. Source
code is available at
\url{this https URL}.

    

### [[2111.07448] Contrastive Clustering: Toward Unsupervised Bias Reduction for Emotion and Sentiment Classification](http://arxiv.org/abs/2111.07448)


  Background: When neural network emotion and sentiment classifiers are used in
public health informatics studies, biases present in the classifiers could
produce inadvertently misleading results.
Objective: This study assesses the impact of bias on COVID-19 topics, and
demonstrates an automatic algorithm for reducing bias when applied to COVID-19
social media texts. This could help public health informatics studies produce
more timely results during crises, with a reduced risk of misleading results.
Methods: Emotion and sentiment classifiers were applied to COVID-19 data
before and after debiasing the classifiers using unsupervised contrastive
clustering. Contrastive clustering approximates the degree to which tokens
exhibit a causal versus correlational relationship with emotion or sentiment,
by contrasting the tokens' relative salience to topics versus emotions or
sentiments.
Results: Contrastive clustering distinguishes correlation from causation for
tokens with an F1 score of 0.753. Masking bias prone tokens from the classifier
input decreases the classifier's overall F1 score by 0.02 (anger) and 0.033
(negative sentiment), but improves the F1 score for sentences annotated as bias
prone by 0.155 (anger) and 0.103 (negative sentiment). Averaging across topics,
debiasing reduces anger estimates by 14.4% and negative sentiment estimates by
8.0%.
Conclusions: Contrastive clustering reduces algorithmic bias in emotion and
sentiment classification for social media text pertaining to the COVID-19
pandemic. Public health informatics studies should account for bias, due to its
prevalence across a range of topics. Further research is needed to improve bias
reduction techniques and to explore the adverse impact of bias on public health
informatics analyses.

    

### [[2111.07505] A Survey on AI Assurance](http://arxiv.org/abs/2111.07505)


  Artificial Intelligence (AI) algorithms are increasingly providing decision
making and operational support across multiple domains. AI includes a wide
library of algorithms for different problems. One important notion for the
adoption of AI algorithms into operational decision process is the concept of
assurance. The literature on assurance, unfortunately, conceals its outcomes
within a tangled landscape of conflicting approaches, driven by contradicting
motivations, assumptions, and intuitions. Accordingly, albeit a rising and
novel area, this manuscript provides a systematic review of research works that
are relevant to AI assurance, between years 1985 - 2021, and aims to provide a
structured alternative to the landscape. A new AI assurance definition is
adopted and presented and assurance methods are contrasted and tabulated.
Additionally, a ten-metric scoring system is developed and introduced to
evaluate and compare existing methods. Lastly, in this manuscript, we provide
foundational insights, discussions, future directions, a roadmap, and
applicable recommendations for the development and deployment of AI assurance.

    

### [[2111.07533] Automated scholarly paper review: Possibility and challenges](http://arxiv.org/abs/2111.07533)


  Peer review is a widely accepted mechanism for research evaluation, playing a
pivotal role in scholarly publishing. However, criticisms have long been
leveled on this mechanism, mostly because of its inefficiency and subjectivity.
Recent years have seen the application of artificial intelligence (AI) in
assisting the peer review process. Nonetheless, with the involvement of humans,
such limitations remain inevitable. In this review paper, we propose the
concept of automated scholarly paper review (ASPR) and review the relevant
literature and technologies to discuss the possibility of achieving a
full-scale computerized review process. We further look into the challenges in
ASPR with the existing technologies. On the basis of the review and discussion,
we conclude that there are already corresponding research and technologies at
each stage of ASPR. This verifies that ASPR can be realized in the long term as
the relevant technologies continue to develop. The major difficulties in its
realization lie in imperfect document parsing and representation, inadequate
data, defected human-computer interaction and flawed deep logical reasoning. In
the foreseeable future, ASPR and peer review will coexist in a reinforcing
manner before ASPR is able to fully undertake the reviewing workload from
humans.

    

### [[2111.07555] Confucius, Cyberpunk and Mr. Science: Comparing AI ethics between China and the EU](http://arxiv.org/abs/2111.07555)


  The exponential development and application of artificial intelligence
triggered an unprecedented global concern for potential social and ethical
issues. Stakeholders from different industries, international foundations,
governmental organisations and standards institutions quickly improvised and
created various codes of ethics attempting to regulate AI. A major concern is
the large homogeneity and presumed consensualism around these principles. While
it is true that some ethical doctrines, such as the famous Kantian deontology,
aspire to universalism, they are however not universal in practice. In fact,
ethical pluralism is more about differences in which relevant questions to ask
rather than different answers to a common question. When people abide by
different moral doctrines, they tend to disagree on the very approach to an
issue. Even when people from different cultures happen to agree on a set of
common principles, it does not necessarily mean that they share the same
understanding of these concepts and what they entail. In order to better
understand the philosophical roots and cultural context underlying ethical
principles in AI, we propose to analyse and compare the ethical principles
endorsed by the Chinese National New Generation Artificial Intelligence
Governance Professional Committee (CNNGAIGPC) and those elaborated by the
European High-level Expert Group on AI (HLEGAI). China and the EU have very
different political systems and diverge in their cultural heritages. In our
analysis, we wish to highlight that principles that seem similar a priori may
actually have different meanings, derived from different approaches and reflect
distinct goals.

    

### [[2111.07556] High-Quality Real Time Facial Capture Based on Single Camera](http://arxiv.org/abs/2111.07556)


  We propose a real time deep learning framework for video-based facial
expression capture. Our process uses a high-end facial capture pipeline based
on FACEGOOD to capture facial expression. We train a convolutional neural
network to produce high-quality continuous blendshape weight output from video
training. Since this facial capture is fully automated, our system can
drastically reduce the amount of labor involved in the development of modern
narrative-driven video games or films involving realistic digital doubles of
actors and potentially hours of animated dialogue per character. We demonstrate
compelling animation inference in challenging areas such as eyes and lips.

    

### [[2111.07568] Can Graph Neural Networks Learn to Solve MaxSAT Problem?](http://arxiv.org/abs/2111.07568)


  With the rapid development of deep learning techniques, various recent work
has tried to apply graph neural networks (GNNs) to solve NP-hard problems such
as Boolean Satisfiability (SAT), which shows the potential in bridging the gap
between machine learning and symbolic reasoning. However, the quality of
solutions predicted by GNNs has not been well investigated in the literature.
In this paper, we study the capability of GNNs in learning to solve Maximum
Satisfiability (MaxSAT) problem, both from theoretical and practical
perspectives. We build two kinds of GNN models to learn the solution of MaxSAT
instances from benchmarks, and show that GNNs have attractive potential to
solve MaxSAT problem through experimental evaluation. We also present a
theoretical explanation of the effect that GNNs can learn to solve MaxSAT
problem to some extent for the first time, based on the algorithmic alignment
theory.

    

### [[2111.07611] Rationale production to support clinical decision-making](http://arxiv.org/abs/2111.07611)


  The development of neural networks for clinical artificial intelligence (AI)
is reliant on interpretability, transparency, and performance. The need to
delve into the black-box neural network and derive interpretable explanations
of model output is paramount. A task of high clinical importance is predicting
the likelihood of a patient being readmitted to hospital in the near future to
enable efficient triage. With the increasing adoption of electronic health
records (EHRs), there is great interest in applications of natural language
processing (NLP) to clinical free-text contained within EHRs. In this work, we
apply InfoCal, the current state-of-the-art model that produces extractive
rationales for its predictions, to the task of predicting hospital readmission
using hospital discharge notes. We compare extractive rationales produced by
InfoCal to competitive transformer-based models pretrained on clinical text
data and for which the attention mechanism can be used for interpretation. We
find each presented model with selected interpretability or feature importance
methods yield varying results, with clinical language domain expertise and
pretraining critical to performance and subsequent interpretability.

    

### [[2111.07631] AI in Games: Techniques, Challenges and Opportunities](http://arxiv.org/abs/2111.07631)


  With breakthrough of AlphaGo, AI in human-computer game has become a very hot
topic attracting researchers all around the world, which usually serves as an
effective standard for testing artificial intelligence. Various game AI systems
(AIs) have been developed such as Libratus, OpenAI Five and AlphaStar, beating
professional human players. In this paper, we survey recent successful game
AIs, covering board game AIs, card game AIs, first-person shooting game AIs and
real time strategy game AIs. Through this survey, we 1) compare the main
difficulties among different kinds of games for the intelligent decision making
field ; 2) illustrate the mainstream frameworks and techniques for developing
professional level AIs; 3) raise the challenges or drawbacks in the current AIs
for intelligent decision making; and 4) try to propose future trends in the
games and intelligent decision making techniques. Finally, we hope this brief
review can provide an introduction for beginners, inspire insights for
researchers in the filed of AI in games.

    

### [[2111.07640] AnimeCeleb: Large-Scale Animation CelebFaces Dataset via Controllable 3D Synthetic Models](http://arxiv.org/abs/2111.07640)


  Despite remarkable success in deep learning-based face-related models, these
models are still limited to the domain of real human faces. On the other hand,
the domain of animation faces has been studied less intensively due to the
absence of a well-organized dataset. In this paper, we present a large-scale
animation celebfaces dataset (AnimeCeleb) via controllable synthetic animation
models to boost research on the animation face domain. To facilitate the data
generation process, we build a semi-automatic pipeline based on an open 3D
software and a developed annotation system. This leads to constructing a
large-scale animation face dataset that includes multi-pose and multi-style
animation faces with rich annotations. Experiments suggest that our dataset is
applicable to various animation-related tasks such as head reenactment and
colorization.

    

### [[2111.07648] The Possibilistic Horn Non-Clausal Knowledge Bases](http://arxiv.org/abs/2111.07648)


  Posibilistic logic is the most extended approach to handle uncertain and
partially inconsistent information. Regarding normal forms, advances in
possibilistic reasoning are mostly focused on clausal form. Yet, the encoding
of real-world problems usually results in a non-clausal (NC) formula and
NC-to-clausal translators produce severe drawbacks that heavily limit the
practical performance of clausal reasoning. Thus, by computing formulas in its
original NC form, we propose several contributions showing that notable
advances are also possible in possibilistic non-clausal reasoning.
{\em Firstly,} we define the class of {\em Possibilistic Horn Non-Clausal
Knowledge Bases,} or $\mathcal{\overline{H}}_\Sigma$, which subsumes the
classes: possibilistic Horn and propositional Horn-NC.
$\mathcal{\overline{H}}_\Sigma $ is shown to be a kind of NC analogous of the
standard Horn class.
{\em Secondly}, we define {\em Possibilistic Non-Clausal Unit-Resolution,} or
$ \mathcal{UR}_\Sigma $, and prove that $ \mathcal{UR}_\Sigma $ correctly
computes the inconsistency degree of $\mathcal{\overline{H}}_\Sigma $members.
$\mathcal{UR}_\Sigma $ had not been proposed before and is formulated in a
clausal-like manner, which eases its understanding, formal proofs and future
extension towards non-clausal resolution.
{\em Thirdly}, we prove that computing the inconsistency degree of
$\mathcal{\overline{H}}_\Sigma $ members takes polynomial time. Although there
already exist tractable classes in possibilistic logic, all of them are
clausal, and thus, $\mathcal{\overline{H}}_\Sigma $ turns out to be the first
characterized polynomial non-clausal class within possibilistic reasoning.

    

### [[2111.07658] Calculating Question Similarity is Enough:A New Method for KBQA Tasks](http://arxiv.org/abs/2111.07658)


  Knowledge Base Question Answering (KBQA) aims to answer natural language
questions with the help of an external knowledge base. The core idea is to find
the link between the internal knowledge behind questions and known triples of
the knowledge base. The KBQA task pipeline contains several steps, including
entity recognition, relationship extraction, and entity linking. This kind of
pipeline method means that errors in any procedure will inevitably propagate to
the final prediction. In order to solve the above problem, this paper proposes
a Corpus Generation - Retrieve Method (CGRM) with Pre-training Language Model
(PLM) and Knowledge Graph (KG). Firstly, based on the mT5 model, we designed
two new pre-training tasks: knowledge masked language modeling and question
generation based on the paragraph to obtain the knowledge enhanced T5 (kT5)
model. Secondly, after preprocessing triples of knowledge graph with a series
of heuristic rules, the kT5 model generates natural language QA pairs based on
processed triples. Finally, we directly solve the QA by retrieving the
synthetic dataset. We test our method on NLPCC-ICCPOL 2016 KBQA dataset, and
the results show that our framework improves the performance of KBQA and the
out straight-forward method is competitive with the state-of-the-art.

    

### [[2103.11909] Identifying Machine-Paraphrased Plagiarism](http://arxiv.org/abs/2103.11909)


  Employing paraphrasing tools to conceal plagiarized text is a severe threat
to academic integrity. To enable the detection of machine-paraphrased text, we
evaluate the effectiveness of five pre-trained word embedding models combined
with machine learning classifiers and state-of-the-art neural language models.
We analyze preprints of research papers, graduation theses, and Wikipedia
articles, which we paraphrased using different configurations of the tools
SpinBot and SpinnerChief. The best performing technique, Longformer, achieved
an average F1 score of 80.99% (F1=99.68% for SpinBot and F1=71.64% for
SpinnerChief cases), while human evaluators achieved F1=78.4% for SpinBot and
F1=65.6% for SpinnerChief cases. We show that the automated classification
alleviates shortcomings of widely-used text-matching systems, such as Turnitin
and PlagScan. To facilitate future research, all data, code, and two web
applications showcasing our contributions are openly available.

    

### [[2103.12450] Are Neural Language Models Good Plagiarists? A Benchmark for Neural Paraphrase Detection](http://arxiv.org/abs/2103.12450)


  The rise of language models such as BERT allows for high-quality text
paraphrasing. This is a problem to academic integrity, as it is difficult to
differentiate between original and machine-generated content. We propose a
benchmark consisting of paraphrased articles using recent language models
relying on the Transformer architecture. Our contribution fosters future
research of paraphrase detection systems as it offers a large collection of
aligned original and paraphrased documents, a study regarding its structure,
classification experiments with state-of-the-art systems, and we make our
findings publicly available.

    

### [[2104.00835] Cursed yet Satisfied Agents](http://arxiv.org/abs/2104.00835)


  In real life auctions, a widely observed phenomenon is the winner's curse --
the winner's high bid implies that the winner often over-estimates the value of
the good for sale, resulting in an incurred negative utility. The seminal work
of Eyster and Rabin [Econometrica'05] introduced a behavioral model aimed to
explain this observed anomaly. We term agents who display this bias "cursed
agents". We adopt their model in the interdependent value setting, and aim to
devise mechanisms that prevent the cursed agents from obtaining negative
utility. We design mechanisms that are cursed ex-post IC, that is, incentivize
agents to bid their true signal even though they are cursed, while ensuring
that the outcome is individually rational -- the price the agents pay is no
more than the agents' true value.
Since the agents might over-estimate the good's value, such mechanisms might
require the seller to make positive transfers to the agents to prevent agents
from over-paying. For revenue maximization, we give the optimal deterministic
and anonymous mechanism. For welfare maximization, we require ex-post budget
balance (EPBB), as positive transfers might lead to negative revenue. We
propose a masking operation that takes any deterministic mechanism, and imposes
that the seller would not make positive transfers, enforcing EPBB. We show that
in typical settings, EPBB implies that the mechanism cannot make any positive
transfers, implying that applying the masking operation on the fully efficient
mechanism results in a socially optimal EPBB mechanism. This further implies
that if the valuation function is the maximum of agents' signals, the optimal
EPBB mechanism obtains zero welfare. In contrast, we show that for sum-concave
valuations, which include weighted-sum valuations and l_p-norms, the welfare
optimal EPBB mechanism obtains half of the optimal welfare as the number of
agents grows large.

    

### [[2104.12643] Exploring Bayesian Deep Learning for Urgent Instructor Intervention Need in MOOC Forums](http://arxiv.org/abs/2104.12643)


  Massive Open Online Courses (MOOCs) have become a popular choice for
e-learning thanks to their great flexibility. However, due to large numbers of
learners and their diverse backgrounds, it is taxing to offer real-time
support. Learners may post their feelings of confusion and struggle in the
respective MOOC forums, but with the large volume of posts and high workloads
for MOOC instructors, it is unlikely that the instructors can identify all
learners requiring intervention. This problem has been studied as a Natural
Language Processing (NLP) problem recently, and is known to be challenging, due
to the imbalance of the data and the complex nature of the task. In this paper,
we explore for the first time Bayesian deep learning on learner-based text
posts with two methods: Monte Carlo Dropout and Variational Inference, as a new
solution to assessing the need of instructor interventions for a learner's
post. We compare models based on our proposed methods with probabilistic
modelling to its baseline non-Bayesian models under similar circumstances, for
different cases of applying prediction. The results suggest that Bayesian deep
learning offers a critical uncertainty measure that is not supplied by
traditional neural networks. This adds more explainability, trust and
robustness to AI, which is crucial in education-based applications.
Additionally, it can achieve similar or better performance compared to
non-probabilistic neural networks, as well as grant lower variance.

    

### [[2107.00048] Uncertainty-Aware Learning for Improvements in Image Quality of the Canada-France-Hawaii Telescope](http://arxiv.org/abs/2107.00048)


  We leverage state-of-the-art machine learning methods and a decade's worth of
archival data from CFHT to predict observatory image quality (IQ) from
environmental conditions and observatory operating parameters. Specifically, we
develop accurate and interpretable models of the complex dependence between
data features and observed IQ for CFHT's wide-field camera, MegaCam. Our
contributions are several-fold. First, we collect, collate and reprocess
several disparate data sets gathered by CFHT scientists. Second, we predict
probability distribution functions (PDFs) of IQ and achieve a mean absolute
error of $\sim0.07''$ for the predicted medians. Third, we explore the
data-driven actuation of the 12 dome "vents" installed in 2013-14 to accelerate
the flushing of hot air from the dome. We leverage epistemic and aleatoric
uncertainties in conjunction with probabilistic generative modeling to identify
candidate vent adjustments that are in-distribution (ID); for the optimal
configuration for each ID sample, we predict the reduction in required
observing time to achieve a fixed SNR. On average, the reduction is $\sim12\%$.
Finally, we rank input features by their Shapley values to identify the most
predictive variables for each observation. Our long-term goal is to construct
reliable and real-time models that can forecast optimal observatory operating
parameters to optimize IQ. We can then feed such forecasts into scheduling
protocols and predictive maintenance routines. We anticipate that such
approaches will become standard in automating observatory operations and
maintenance by the time CFHT's successor, the Maunakea Spectroscopic Explorer,
is installed in the next decade.

    

### [[2010.11342] Contextual Linear Types for Differential Privacy](http://arxiv.org/abs/2010.11342)


  Language support for differentially-private programming is both crucial and
delicate. While elaborate program logics can be very expressive, type-system
based approaches using linear types tend to be more lightweight and amenable to
automatic checking and inference, and in particular in the presence of
higher-order programming. Since the seminal design of Fuzz, which is restricted
to $\epsilon$-differential privacy, a lot of effort has been made to support
more advanced variants of differential privacy, like
($\epsilon$,$\delta$)-differential privacy. However, supporting these advanced
privacy variants while also supporting higher-order programming in full has
been proven to be challenging. We present Jazz, a language and type system
which uses linear types and latent contextual effects to support both advanced
variants of differential privacy and higher-order programming. Even when
avoiding advanced variants and higher-order programming, our system achieves
higher precision than prior work for a large class of programming patterns. We
formalize the core of the Jazz language, prove it sound for privacy via a
logical relation for metric preservation, and illustrate its expressive power
through a number of case studies drawn from the recent differential privacy
literature.

    

### [[2012.02154] Quantum Hoare Type Theory](http://arxiv.org/abs/2012.02154)


  As quantum computers become real, it is high time we come up with effective
techniques that help programmers write correct quantum programs. Inspired by
Hoare Type Theory in classical computing, we propose Quantum Hoare Type Theory
(QHTT), in which precise specifications about the modification to the quantum
state can be provided within the type of computation. These specifications
within a Hoare type are given in the form of Hoare-logic style pre- and
postconditions following the propositions-as-types principle. The type-checking
process verifies that the implementation conforms to the provided
specification. QHTT has the potential to be a unified system for programming,
specifying, and reasoning about quantum programs.

    

### [[2105.00417] Formalizing Stack Safety as a Security Property](http://arxiv.org/abs/2105.00417)


  What does "stack safety" mean? The phrase is associated with a variety of
compiler, run-time, and hardware mechanisms for protecting stack memory, but
these mechanisms typically lack precise specifications, relying instead on
informal descriptions and examples of the bad behaviors that they prevent.
We propose a generic, formal characterization of stack safety based on
concepts from language-based security: a combination of an integrity property
("the private state in each caller's stack frame is held invariant by the
callee"), and a confidentiality property ("the callee's behavior is insensitive
to the caller's private state"), plus an optional control-flow property.
We use these properties to validate the stack safety micro-policies proposed
by Roessler and DeHon [2018]. Specifically, we check (with property-based
random testing) that their "eager" micro-policy, which catches violations
early, enforces a simple "stepwise" variant of our properties, and that (a
repaired version of) their more performant "lazy" micro-policy enforces a
slightly weaker and more extensional observational property. Meanwhile our
testing successfully detects violations in several broken variants, including
Roessler and DeHon's original lazy policy.

    