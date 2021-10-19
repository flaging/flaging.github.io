
## 2021-10-19

### [[2110.08374] Gemini: Practical Reconfigurable Datacenter Networks with Topology and Traffic Engineering](http://arxiv.org/abs/2110.08374)


  To reduce cost, datacenter network operators are exploring blocking network
designs. An example of such a design is a "spine-free" form of a Fat-Tree, in
which pods directly connect to each other, rather than via spine blocks. To
maintain application-perceived performance in the face of dynamic workloads,
these new designs must be able to reconfigure routing and the inter-pod
topology. Gemini is a system designed to achieve these goals on commodity
hardware while reconfiguring the network infrequently, rendering these blocking
designs practical enough for deployment in the near future. The key to Gemini
is the joint optimization of topology and routing, using as input a robust
estimation of future traffic derived from multiple historical traffic matrices.
Gemini "hedges" against unpredicted bursts, by spreading these bursts across
multiple paths, to minimize packet loss in exchange for a small increase in
path lengths. It incorporates a robust decision algorithm to determine when to
reconfigure, and whether to use hedging. Data from tens of production fabrics
allows us to categorize these as either low-or high-volatility; these
categories seem stable. For the former, Gemini finds topologies and routing
with near-optimal performance and cost. For the latter, Gemini's use of
multi-traffic-matrix optimization and hedging avoids the need for frequent
topology reconfiguration, with only marginal increases in path length. As a
result, Gemini can support existing workloads on these production fabrics using
a spine-free topology that is half the cost of the existing topology on these
fabrics.

    

### [[2110.08481] On the randomness analysis of link quality prediction: limitations and benefits](http://arxiv.org/abs/2110.08481)


  In wireless multi-hop networks, such as wireless sensor networks, link
quality (LQ) is one of the most important metrics and is widely used in
higher-layer applications such as routing protocols. An accurate link quality
prediction may greatly help to improve the performance of wireless multi-hop
networks. Researchers have proposed a lot of link quality prediction models in
recent years. However, due to the dynamic and stochastic nature of wireless
transmission, the performance of link quality prediction remains challenging.
In this article, we mainly analyze the influence of the stochastic nature of
wireless transmission on the link quality prediction model and discuss the
benefits in the application of wireless multi-hop networks with the
performance-limited link quality prediction models.

    

### [[2110.08485] Design of Link-Quality-prediction-based Software-Defined Wireless Sensor Networks](http://arxiv.org/abs/2110.08485)


  In wireless multi-hop networks, the instability of the wireless links leads
to unstable networking. Even in the newly designed Software-Defined Wireless
Sensor Networks (SDWSN), similar problems exist. To further improve the
stability of SDWSN, we introduce a Link Quality (LQ) prediction model into the
SDWSN architecture. The prediction model is used to improve the stability
between neighbor nodes, and thus the stability of wireless multi-hop routes.
Simulation results show that the LQ prediction model can make reasonable
corrections to the reality wireless link model, which can well address the
restrictive nature of unstable links. As a result, by reducing the use of
unstable wireless links, the stability of the SDWSN network improved.

    

### [[2110.08588] Preproduction Deploys: Cloud-Native Integration Testing](http://arxiv.org/abs/2110.08588)


  The microservice architecture for cloud-based systems is extended to not only
require each loosely coupled component to be independently deployable, but also
to provide independent routing for each component. This supports canary
deployments, green/blue deployments and roll-back. Both ad hoc and system
integration test traffic can be directed to components before they are released
to production traffic. Front-end code is included in this architecture by using
server-side rendering of JS bundles. Environments for integration testing are
created with preproduction deploys side by side with production deploys using
appropriate levels of isolation. After a successful integration test run,
preproduction components are known to work with production precisely as it is.
For isolation, test traffic uses staging databases that are copied daily from
the production databases, omitting sensitive data. Safety and security concerns
are dealt with in a targeted fashion, not monolithically. This architecture
scales well with organization size; is more effective for integration testing;
and is better aligned with agile business practices than traditional
approaches.

    

### [[2110.08654] LEO Satellites in 5G and Beyond Networks: A Review from a Standardization Perspective](http://arxiv.org/abs/2110.08654)


  Low Earth Orbit (LEO) Satellite Network (SatNet) with their
mega-constellations are expected to play a key role in providing ubiquitous
Internet and communications services in the future. LEO SatNets will provide
wide-area coverage and support service availability, continuity, and
scalability. To support the integration of SatNets and terrestrial Fifth
Generation (5G)networks and beyond, the satellite communication industry has
become increasingly involved with the 3rd Generation Partnership Project (3GPP)
standardization activities for 5G. In this work, we review the 3GPP
standardization activities for the integration of SatNets in 5G and beyond. The
3GPP use cases of SatNets are highlighted and potential requirements to realize
them are summarized as well. The impacted areas of New Radio(NR) are discussed
with some potential solutions. The foreseen requirements for the management and
orchestration of SatNets within 5G are described. Future standardization
directions are discussed to support the full integration of SatNets in
SixthGeneration (6G) with the goal of ubiquitous global connectivity.

    

### [[2101.10961] The Wireless Control Bus: Enabling Efficient Multi-hop Event-Triggered Control with Concurrent Transmissions](http://arxiv.org/abs/2101.10961)


  Event-triggered control (ETC) holds the potential to significantly improve
the efficiency of wireless networked control systems. Unfortunately, its
real-world impact has hitherto been hampered by the lack of a network stack
able to transfer its benefits from theory to practice specifically by
supporting the latency and reliability requirements of the aperiodic
communication ETC induces. This is precisely the contribution of this paper.
Our Wireless Control Bus (WCB) exploits carefully orchestrated network-wide
floods of concurrent transmissions to minimize overhead during quiescent,
steady-state periods, and ensures timely and reliable collection of sensor
readings and dissemination of actuation commands when an ETC triggering
condition is violated. Using a cyber-physical testbed emulating a water
distribution system controlled over a real-world multi-hop wireless network, we
show that ETC over WCB achieves the same quality of periodic control at a
fraction of the energy costs, therefore unleashing and concretely demonstrating
its full potential for the first time.

    

### [[2106.14229] Over-the-Air Federated Multi-Task Learning](http://arxiv.org/abs/2106.14229)


  In this letter, we introduce over-the-air computation into the communication
design of federated multi-task learning (FMTL), and propose an over-the-air
federated multi-task learning (OA-FMTL) framework, where multiple learning
tasks deployed on edge devices share a non-orthogonal fading channel under the
coordination of an edge server (ES). Specifically, the model updates for all
the tasks are transmitted and superimposed concurrently over a non-orthogonal
uplink fading channel, and the model aggregations of all the tasks are
reconstructed at the ES through a modified version of the turbo compressed
sensing algorithm (Turbo-CS) that overcomes inter-task interference. Both
convergence analysis and numerical results show that the OA-FMTL framework can
significantly improve the system efficiency in terms of reducing the number of
channel uses without causing substantial learning performance degradation.

    

### [[2110.08252] A Rate-Distortion Framework for Explaining Black-box Model Decisions](http://arxiv.org/abs/2110.08252)


  We present the Rate-Distortion Explanation (RDE) framework, a mathematically
well-founded method for explaining black-box model decisions. The framework is
based on perturbations of the target input signal and applies to any
differentiable pre-trained model such as neural networks. Our experiments
demonstrate the framework's adaptability to diverse data modalities,
particularly images, audio, and physical simulations of urban environments.

    

### [[2110.08253] A Field Guide to Scientific XAI: Transparent and Interpretable Deep Learning for Bioinformatics Research](http://arxiv.org/abs/2110.08253)


  Deep learning has become popular because of its potential to achieve high
accuracy in prediction tasks. However, accuracy is not always the only goal of
statistical modelling, especially for models developed as part of scientific
research. Rather, many scientific models are developed to facilitate scientific
discovery, by which we mean to abstract a human-understandable representation
of the natural world. Unfortunately, the opacity of deep neural networks limit
their role in scientific discovery, creating a new demand for models that are
transparently interpretable. This article is a field guide to transparent model
design. It provides a taxonomy of transparent model design concepts, a
practical workflow for putting design concepts into practice, and a general
template for reporting design choices. We hope this field guide will help
researchers more effectively design transparently interpretable models, and
thus enable them to use deep learning for scientific discovery.

    

### [[2110.08254] Inconsistent Few-Shot Relation Classification via Cross-Attentional Prototype Networks with Contrastive Learning](http://arxiv.org/abs/2110.08254)


  Standard few-shot relation classification (RC) is designed to learn a robust
classifier with only few labeled data for each class. However, previous works
rarely investigate the effects of a different number of classes (i.e., $N$-way)
and number of labeled data per class (i.e., $K$-shot) during training vs.
testing. In this work, we define a new task, \textit{inconsistent few-shot RC},
where the model needs to handle the inconsistency of $N$ and $K$ between
training and testing. To address this new task, we propose Prototype
Network-based cross-attention contrastive learning (ProtoCACL) to capture the
rich mutual interactions between the support set and query set. Experimental
results demonstrate that our ProtoCACL can outperform the state-of-the-art
baseline model under both inconsistent $K$ and inconsistent $N$ settings, owing
to its more robust and discriminate representations. Moreover, we identify that
in the inconsistent few-shot learning setting, models can achieve better
performance with \textit{less data} than the standard few-shot setting with
carefully-selected $N$ and $K$. In the end of the paper, we provide further
analyses and suggestions to systematically guide the selection of $N$ and $K$
under different scenarios.

    

### [[2110.08255] Yformer: U-Net Inspired Transformer Architecture for Far Horizon Time Series Forecasting](http://arxiv.org/abs/2110.08255)


  Time series data is ubiquitous in research as well as in a wide variety of
industrial applications. Effectively analyzing the available historical data
and providing insights into the far future allows us to make effective
decisions. Recent research has witnessed the superior performance of
transformer-based architectures, especially in the regime of far horizon time
series forecasting. However, the current state of the art sparse Transformer
architectures fail to couple down- and upsampling procedures to produce outputs
in a similar resolution as the input. We propose the Yformer model, based on a
novel Y-shaped encoder-decoder architecture that (1) uses direct connection
from the downscaled encoder layer to the corresponding upsampled decoder layer
in a U-Net inspired architecture, (2) Combines the downscaling/upsampling with
sparse attention to capture long-range effects, and (3) stabilizes the
encoder-decoder stacks with the addition of an auxiliary reconstruction loss.
Extensive experiments have been conducted with relevant baselines on four
benchmark datasets, demonstrating an average improvement of 19.82, 18.41
percentage MSE and 13.62, 11.85 percentage MAE in comparison to the current
state of the art for the univariate and the multivariate settings respectively.

    

### [[2110.08256] Model-Agnostic Meta-Attack: Towards Reliable Evaluation of Adversarial Robustness](http://arxiv.org/abs/2110.08256)


  The vulnerability of deep neural networks to adversarial examples has
motivated an increasing number of defense strategies for promoting model
robustness. However, the progress is usually hampered by insufficient
robustness evaluations. As the de facto standard to evaluate adversarial
robustness, adversarial attacks typically solve an optimization problem of
crafting adversarial examples with an iterative process. In this work, we
propose a Model-Agnostic Meta-Attack (MAMA) approach to discover stronger
attack algorithms automatically. Our method learns the optimizer in adversarial
attacks parameterized by a recurrent neural network, which is trained over a
class of data samples and defenses to produce effective update directions
during adversarial example generation. Furthermore, we develop a model-agnostic
training algorithm to improve the generalization ability of the learned
optimizer when attacking unseen defenses. Our approach can be flexibly
incorporated with various attacks and consistently improves the performance
with little extra computational cost. Extensive experiments demonstrate the
effectiveness of the learned attacks by MAMA compared to the state-of-the-art
attacks on different defenses, leading to a more reliable evaluation of
adversarial robustness.

    

### [[2110.08257] C-AllOut: Catching & Calling Outliers by Type](http://arxiv.org/abs/2110.08257)


  Given an unlabeled dataset, wherein we have access only to pairwise
similarities (or distances), how can we effectively (1) detect outliers, and
(2) annotate/tag the outliers by type? Outlier detection has a large
literature, yet we find a key gap in the field: to our knowledge, no existing
work addresses the outlier annotation problem. Outliers are broadly classified
into 3 types, representing distinct patterns that could be valuable to
analysts: (a) global outliers are severe yet isolate cases that do not repeat,
e.g., a data collection error; (b) local outliers diverge from their peers
within a context, e.g., a particularly short basketball player; and (c)
collective outliers are isolated micro-clusters that may indicate coalition or
repetitions, e.g., frauds that exploit the same loophole. This paper presents
C-AllOut: a novel and effective outlier detector that annotates outliers by
type. It is parameter-free and scalable, besides working only with pairwise
similarities (or distances) when it is needed. We show that C-AllOut achieves
on par or significantly better performance than state-of-the-art detectors when
spotting outliers regardless of their type. It is also highly effective in
annotating outliers of particular types, a task that none of the baselines can
perform.

    

### [[2110.08258] Learning When and What to Ask: a Hierarchical Reinforcement Learning Framework](http://arxiv.org/abs/2110.08258)


  Reliable AI agents should be mindful of the limits of their knowledge and
consult humans when sensing that they do not have sufficient knowledge to make
sound decisions. We formulate a hierarchical reinforcement learning framework
for learning to decide when to request additional information from humans and
what type of information would be helpful to request. Our framework extends
partially-observed Markov decision processes (POMDPs) by allowing an agent to
interact with an assistant to leverage their knowledge in accomplishing tasks.
Results on a simulated human-assisted navigation problem demonstrate the
effectiveness of our framework: aided with an interaction policy learned by our
method, a navigation policy achieves up to a 7x improvement in task success
rate compared to performing tasks only by itself. The interaction policy is
also efficient: on average, only a quarter of all actions taken during a task
execution are requests for information. We analyze benefits and challenges of
learning with a hierarchical policy structure and suggest directions for future
work.

    

### [[2110.08259] Training Neural Networks for Solving 1-D Optimal Piecewise Linear Approximation](http://arxiv.org/abs/2110.08259)


  Recently, the interpretability of deep learning has attracted a lot of
attention. A plethora of methods have attempted to explain neural networks by
feature visualization, saliency maps, model distillation, and so on. However,
it is hard for these methods to reveal the intrinsic properties of neural
networks. In this work, we studied the 1-D optimal piecewise linear
approximation (PWLA) problem, and associated it with a designed neural network,
named lattice neural network (LNN). We asked four essential questions as
following: (1) What are the characters of the optimal solution of the PWLA
problem? (2) Can an LNN converge to the global optimum? (3) Can an LNN converge
to the local optimum? (4) Can an LNN solve the PWLA problem? Our main
contributions are that we propose the theorems to characterize the optimal
solution of the PWLA problem and present the LNN method for solving it. We
evaluated the proposed LNNs on approximation tasks, forged an empirical method
to improve the performance of LNNs. The experiments verified that our LNN
method is competitive with the start-of-the-art method.

    

### [[2110.08260] Effective Certification of Monotone Deep Equilibrium Models](http://arxiv.org/abs/2110.08260)


  Monotone Operator Equilibrium Models (monDEQs) represent a class of models
combining the powerful deep equilibrium paradigm with convergence guarantees.
Further, their inherent robustness to adversarial perturbations makes
investigating their certifiability a promising research direction.
Unfortunately, existing approaches are either imprecise or severely limited in
scalability. In this work, we propose the first scalable and precise monDEQ
verifier, based on two key ideas: (i) a novel convex relaxation enabling
efficient inclusion checks, and (ii) non-trivial mathematical insights
characterizing the fixpoint operations at the heart of monDEQs on sets rather
than concrete inputs. An extensive evaluation of our verifier on the
challenging $\ell_\infty$ perturbations demonstrates that it exceeds
state-of-the-art performance in terms of speed (two orders of magnitude) and
scalability (an order of magnitude) while yielding 25% higher certified
accuracies on the same networks.

    

### [[2110.08263] FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling](http://arxiv.org/abs/2110.08263)


  The recently proposed FixMatch achieved state-of-the-art results on most
semi-supervised learning (SSL) benchmarks. However, like other modern SSL
algorithms, FixMatch uses a pre-defined constant threshold for all classes to
select unlabeled data that contribute to the training, thus failing to consider
different learning status and learning difficulties of different classes. To
address this issue, we propose Curriculum Pseudo Labeling (CPL), a curriculum
learning approach to leverage unlabeled data according to the model's learning
status. The core of CPL is to flexibly adjust thresholds for different classes
at each time step to let pass informative unlabeled data and their pseudo
labels. CPL does not introduce additional parameters or computations (forward
or backward propagation). We apply CPL to FixMatch and call our improved
algorithm FlexMatch. FlexMatch achieves state-of-the-art performance on a
variety of SSL benchmarks, with especially strong performances when the labeled
data are extremely limited or when the task is challenging. For example,
FlexMatch outperforms FixMatch by 14.32% and 24.55% on CIFAR-100 and STL-10
datasets respectively, when there are only 4 labels per class. CPL also
significantly boosts the convergence speed, e.g., FlexMatch can use only 1/5
training time of FixMatch to achieve even better performance. Furthermore, we
show that CPL can be easily adapted to other SSL algorithms and remarkably
improve their performances. We open source our code at
this https URL.

    

### [[2110.08264] Self-supervised Contrastive Attributed Graph Clustering](http://arxiv.org/abs/2110.08264)


  Attributed graph clustering, which learns node representation from node
attribute and topological graph for clustering, is a fundamental but
challenging task for graph analysis. Recently, methods based on graph
contrastive learning (GCL) have obtained impressive clustering performance on
this task. Yet, we observe that existing GCL-based methods 1) fail to benefit
from imprecise clustering labels; 2) require a post-processing operation to get
clustering labels; 3) cannot solve out-of-sample (OOS) problem. To address
these issues, we propose a novel attributed graph clustering network, namely
Self-supervised Contrastive Attributed Graph Clustering (SCAGC). In SCAGC, by
leveraging inaccurate clustering labels, a self-supervised contrastive loss,
which aims to maximize the similarities of intra-cluster nodes while minimizing
the similarities of inter-cluster nodes, are designed for node representation
learning. Meanwhile, a clustering module is built to directly output clustering
labels by contrasting the representation of different clusters. Thus, for the
OOS nodes, SCAGC can directly calculate their clustering labels. Extensive
experimental results on four benchmark datasets have shown that SCAGC
consistently outperforms 11 competitive clustering methods.

    

### [[2110.08265] Knowledge-driven Active Learning](http://arxiv.org/abs/2110.08265)


  In the last few years, Deep Learning models have become increasingly popular.
However, their deployment is still precluded in those contexts where the amount
of supervised data is limited and manual labelling expensive. Active learning
strategies aim at solving this problem by requiring supervision only on few
unlabelled samples, which improve the most model performances after adding them
to the training set. Most strategies are based on uncertain sample selection,
and even often restricted to samples lying close to the decision boundary. Here
we propose a very different approach, taking into consideration domain
knowledge. Indeed, in the case of multi-label classification, the relationships
among classes offer a way to spot incoherent predictions, i.e., predictions
where the model may most likely need supervision. We have developed a framework
where first-order-logic knowledge is converted into constraints and their
violation is checked as a natural guide for sample selection. We empirically
demonstrate that knowledge-driven strategy outperforms standard strategies,
particularly on those datasets where domain knowledge is complete. Furthermore,
we show how the proposed approach enables discovering data distributions lying
far from training data. Finally, the proposed knowledge-driven strategy can be
also easily used in object-detection problems where standard uncertainty-based
techniques are difficult to apply.

    

### [[2110.08266] PG$^2$Net: Personalized and Group Preferences Guided Network for Next Place Prediction](http://arxiv.org/abs/2110.08266)


  Predicting the next place to visit is a key in human mobility behavior
modeling, which plays a significant role in various fields, such as epidemic
control, urban planning, traffic management, and travel recommendation. To
achieve this, one typical solution is designing modules based on RNN to capture
their preferences to various locations. Although these RNN-based methods can
effectively learn individual's hidden personalized preferences to her visited
places, the interactions among users can only be weakly learned through the
representations of locations. Targeting this, we propose an end-to-end
framework named personalized and group preference guided network (PG$^2$Net),
considering the users' preferences to various places at both individual and
collective levels. Specifically, PG$^2$Net concatenates Bi-LSTM and attention
mechanism to capture each user's long-term mobility tendency. To learn
population's group preferences, we utilize spatial and temporal information of
the visitations to construct a spatio-temporal dependency module. We adopt a
graph embedding method to map users' trajectory into a hidden space, capturing
their sequential relation. In addition, we devise an auxiliary loss to learn
the vectorial representation of her next location. Experiment results on two
Foursquare check-in datasets and one mobile phone dataset indicate the
advantages of our model compared to the state-of-the-art baselines. Source
codes are available at this https URL.

    

### [[2110.08268] Explainable Student Performance Prediction With Personalized Attention for Explaining Why A Student Fails](http://arxiv.org/abs/2110.08268)


  As student failure rates continue to increase in higher education, predicting
student performance in the following semester has become a significant demand.
Personalized student performance prediction helps educators gain a
comprehensive view of student status and effectively intervene in advance.
However, existing works scarcely consider the explainability of student
performance prediction, which educators are most concerned about. In this
paper, we propose a novel Explainable Student performance prediction method
with Personalized Attention (ESPA) by utilizing relationships in student
profiles and prior knowledge of related courses. The designed Bidirectional
Long Short-Term Memory (BiLSTM) architecture extracts the semantic information
in the paths with specific patterns. As for leveraging similar paths' internal
relations, a local and global-level attention mechanism is proposed to
distinguish the influence of different students or courses for making
predictions. Hence, valid reasoning on paths can be applied to predict the
performance of students. The ESPA consistently outperforms the other
state-of-the-art models for student performance prediction, and the results are
intuitively explainable. This work can help educators better understand the
different impacts of behavior on students' studies.

    

### [[2110.08270] From Multimodal to Unimodal Attention in Transformers using Knowledge Distillation](http://arxiv.org/abs/2110.08270)


  Multimodal Deep Learning has garnered much interest, and transformers have
triggered novel approaches, thanks to the cross-attention mechanism. Here we
propose an approach to deal with two key existing challenges: the high
computational resource demanded and the issue of missing modalities. We
introduce for the first time the concept of knowledge distillation in
transformers to use only one modality at inference time. We report a full study
analyzing multiple student-teacher configurations, levels at which distillation
is applied, and different methodologies. With the best configuration, we
improved the state-of-the-art accuracy by 3%, we reduced the number of
parameters by 2.5 times and the inference time by 22%. Such
performance-computation tradeoff can be exploited in many applications and we
aim at opening a new research area where the deployment of complex models with
limited resources is demanded.

    

### [[2110.08271] Training Deep Neural Networks with Joint Quantization and Pruning of Weights and Activations](http://arxiv.org/abs/2110.08271)


  Quantization and pruning are core techniques used to reduce the inference
costs of deep neural networks. State-of-the-art quantization techniques are
currently applied to both the weights and activations; however, pruning is most
often applied to only the weights of the network. In this work, we jointly
apply novel uniform quantization and unstructured pruning methods to both the
weights and activations of deep neural networks during training. Using our
methods, we empirically evaluate the currently accepted prune-then-quantize
paradigm across a wide range of computer vision tasks and observe a
non-commutative nature when applied to both the weights and activations of deep
neural networks. Informed by these observations, we articulate the
non-commutativity hypothesis: for a given deep neural network being trained for
a specific task, there exists an exact training schedule in which quantization
and pruning can be introduced to optimize network performance. We identify that
this optimal ordering not only exists, but also varies across discriminative
and generative tasks. Using the optimal training schedule within our training
framework, we demonstrate increased performance per memory footprint over
existing solutions.

    

### [[2110.08272] Tree-based local explanations of machine learning model predictions, AraucanaXAI](http://arxiv.org/abs/2110.08272)


  Increasingly complex learning methods such as boosting, bagging and deep
learning have made ML models more accurate, but harder to understand and
interpret. A tradeoff between performance and intelligibility is often to be
faced, especially in high-stakes applications like medicine. In the present
article we propose a novel methodological approach for generating explanations
of the predictions of a generic ML model, given a specific instance for which
the prediction has been made, that can tackle both classification and
regression tasks. Advantages of the proposed XAI approach include improved
fidelity to the original model, the ability to deal with non-linear decision
boundaries, and native support to both classification and regression problems

    

### [[2110.08295] Nonlinear proper orthogonal decomposition for convection-dominated flows](http://arxiv.org/abs/2110.08295)


  Autoencoder techniques find increasingly common use in reduced order modeling
as a means to create a latent space. This reduced order representation offers a
modular data-driven modeling approach for nonlinear dynamical systems when
integrated with a time series predictive model. In this letter, we put forth a
nonlinear proper orthogonal decomposition (POD) framework, which is an
end-to-end Galerkin-free model combining autoencoders with long short-term
memory networks for dynamics. By eliminating the projection error due to the
truncation of Galerkin models, a key enabler of the proposed nonintrusive
approach is the kinematic construction of a nonlinear mapping between the
full-rank expansion of the POD coefficients and the latent space where the
dynamics evolve. We test our framework for model reduction of a
convection-dominated system, which is generally challenging for reduced order
models. Our approach not only improves the accuracy, but also significantly
reduces the computational cost of training and testing.

    

### [[2110.08306] Memory-augmented Adversarial Autoencoders for Multivariate Time-series Anomaly Detection with Deep Reconstruction and Prediction](http://arxiv.org/abs/2110.08306)


  Detecting anomalies for multivariate time-series without manual supervision
continues a challenging problem due to the increased scale of dimensions and
complexity of today's IT monitoring systems. Recent progress of unsupervised
time-series anomaly detection mainly use deep autoencoders to solve this
problem, i.e. training on normal samples and producing significant
reconstruction error on abnormal inputs. However, in practice, autoencoders can
reconstruct anomalies so well, due to powerful capabilites of neural networks.
Besides, these approaches can be ineffective for identifying non-point
anomalies, e.g. contextual anomalies and collective anomalies, since they
solely utilze a point-wise reconstruction objective. To tackle the above
issues, we propose MemAAE (\textit{Memory-augmented Adversarial Autoencoders
with Deep Reconstruction and Prediction}), a novel unsupervised anomaly
detection method for time-series. By jointly training two complementary proxy
tasks, reconstruction and prediction, with a shared network architecture, we
show that detecting anomalies via multiple tasks obtains superior performance
rather than single-task training. Additionally, a compressive memory module is
introduced to preserve normal patterns, avoiding unexpected generalization on
abnormal inputs. Through extensive experiments, MemAAE achieves an overall F1
score of 0.90 on four public datasets, significantly outperforming the best
baseline by 0.02.

    

### [[2110.08307] GrowSpace: Learning How to Shape Plants](http://arxiv.org/abs/2110.08307)


  Plants are dynamic systems that are integral to our existence and survival.
Plants face environment changes and adapt over time to their surrounding
conditions. We argue that plant responses to an environmental stimulus are a
good example of a real-world problem that can be approached within a
reinforcement learning (RL)framework. With the objective of controlling a plant
by moving the light source, we propose GrowSpace, as a new RL benchmark. The
back-end of the simulator is implemented using the Space Colonisation
Algorithm, a plant growing model based on competition for space. Compared to
video game RL environments, this simulator addresses a real-world problem and
serves as a test bed to visualize plant growth and movement in a faster way
than physical experiments. GrowSpace is composed of a suite of challenges that
tackle several problems such as control, multi-stage learning,fairness and
multi-objective learning. We provide agent baselines alongside case studies to
demonstrate the difficulty of the proposed benchmark.

    

### [[2110.08313] Reduced Order Dynamical Models For Complex Dynamics in Manufacturing and Natural Systems Using Machine Learning](http://arxiv.org/abs/2110.08313)


  Dynamical analysis of manufacturing and natural systems provides critical
information about production of manufactured and natural resources
respectively, thus playing an important role in assessing sustainability of
these systems. However, current dynamic models for these systems exist as
mechanistic models, simulation of which is computationally intensive and does
not provide a simplified understanding of the mechanisms driving the overall
dynamics. For such systems, lower-order models can prove useful to enable
sustainability analysis through coupled dynamical analysis. There have been few
attempts at finding low-order models of manufacturing and natural systems, with
existing work focused on model development of individual mechanism level. This
work seeks to fill this current gap in the literature of developing simplified
dynamical models for these systems by developing reduced-order models using a
machine learning (ML) approach. The approach is demonstrated on an entire
soybean-oil to soybean-diesel process plant and a lake system. We use a
grey-box ML method with a standard nonlinear optimization approach to identify
relevant models of governing dynamics as ODEs using the data simulated from
mechanistic models. Results show that the method identifies a high accuracy
linear ODE models for the process plant, reflective of underlying linear
stoichiometric mechanisms and mass balance driving the dynamics. For the
natural systems, we modify the ML approach to include the effect of past
dynamics, which gives non-linear ODE. While the modified approach provides a
better match to dynamics of stream flow, it falls short of completely
recreating the dynamics. We conclude that the proposed ML approach work well
for systems where dynamics is smooth, such as in manufacturing plant whereas
does not work perfectly well in case of chaotic dynamics such as water stream
flow.

    

### [[2110.08321] Efficient Representations for Privacy-Preserving Inference](http://arxiv.org/abs/2110.08321)


  Deep neural networks have a wide range of applications across multiple
domains such as computer vision and medicine. In many cases, the input of a
model at inference time can consist of sensitive user data, which raises
questions concerning the levels of privacy and trust guaranteed by such
services. Much existing work has leveraged homomorphic encryption (HE) schemes
that enable computation on encrypted data to achieve private inference for
multi-layer perceptrons and CNNs. An early work along this direction was
CryptoNets, which takes 250 seconds for one MNIST inference. The main
limitation of such approaches is that of compute, which is due to the costly
nature of the NTT (number theoretic transform)operations that constitute HE
operations. Others have proposed the use of model pruning and efficient data
representations to reduce the number of HE operations required. In this paper,
we focus on improving upon existing work by proposing changes to the
representations of intermediate tensors during CNN inference. We construct and
evaluate private CNNs on the MNIST and CIFAR-10 datasets, and achieve over a
two-fold reduction in the number of operations used for inferences of the
CryptoNets architecture.

    

### [[2110.08322] Robustness of different loss functions and their impact on networks learning capability](http://arxiv.org/abs/2110.08322)


  Recent developments in AI have made it ubiquitous, every industry is trying
to adopt some form of intelligent processing of their data. Despite so many
advances in the field, AIs full capability is yet to be exploited by the
industry. Industries that involve some risk factors still remain cautious about
the usage of AI due to the lack of trust in such autonomous systems.
Present-day AI might be very good in a lot of things but it is very bad in
reasoning and this behavior of AI can lead to catastrophic results. Autonomous
cars crashing into a person or a drone getting stuck in a tree are a few
examples where AI decisions lead to catastrophic results. To develop insight
and generate an explanation about the learning capability of AI, we will try to
analyze the working of loss functions. For our case, we will use two sets of
loss functions, generalized loss functions like Binary cross-entropy or BCE and
specialized loss functions like Dice loss or focal loss. Through a series of
experiments, we will establish whether combining different loss functions is
better than using a single loss function and if yes, then what is the reason
behind it. In order to establish the difference between generalized loss and
specialized losses, we will train several models using the above-mentioned
losses and then compare their robustness on adversarial examples. In
particular, we will look at how fast the accuracy of different models decreases
when we change the pixels corresponding to the most salient gradients.

    

### [[2110.08323] On Learning the Transformer Kernel](http://arxiv.org/abs/2110.08323)


  In this work we introduce KERNELIZED TRANSFORMER, a generic, scalable, data
driven framework for learning the kernel function in Transformers. Our
framework approximates the Transformer kernel as a dot product between spectral
feature maps and learns the kernel by learning the spectral distribution. This
not only helps in learning a generic kernel end-to-end, but also reduces the
time and space complexity of Transformers from quadratic to linear. We show
that KERNELIZED TRANSFORMERS achieve performance comparable to existing
efficient Transformer architectures, both in terms of accuracy as well as
computational efficiency. Our study also demonstrates that the choice of the
kernel has a substantial impact on performance, and kernel learning variants
are competitive alternatives to fixed kernel Transformers, both in long as well
as short sequence tasks.

    

### [[2110.08324] Mitigating Membership Inference Attacks by Self-Distillation Through a Novel Ensemble Architecture](http://arxiv.org/abs/2110.08324)


  Membership inference attacks are a key measure to evaluate privacy leakage in
machine learning (ML) models. These attacks aim to distinguish training members
from non-members by exploiting differential behavior of the models on member
and non-member inputs. The goal of this work is to train ML models that have
high membership privacy while largely preserving their utility; we therefore
aim for an empirical membership privacy guarantee as opposed to the provable
privacy guarantees provided by techniques like differential privacy, as such
techniques are shown to deteriorate model utility. Specifically, we propose a
new framework to train privacy-preserving models that induces similar behavior
on member and non-member inputs to mitigate membership inference attacks. Our
framework, called SELENA, has two major components. The first component and the
core of our defense is a novel ensemble architecture for training. This
architecture, which we call Split-AI, splits the training data into random
subsets, and trains a model on each subset of the data. We use an adaptive
inference strategy at test time: our ensemble architecture aggregates the
outputs of only those models that did not contain the input sample in their
training data. We prove that our Split-AI architecture defends against a large
family of membership inference attacks, however, it is susceptible to new
adaptive attacks. Therefore, we use a second component in our framework called
Self-Distillation to protect against such stronger attacks. The
Self-Distillation component (self-)distills the training dataset through our
Split-AI ensemble, without using any external public datasets. Through
extensive experiments on major benchmark datasets we show that SELENA presents
a superior trade-off between membership privacy and utility compared to the
state of the art.

    

### [[2110.08327] Solving Image PDEs with a Shallow Network](http://arxiv.org/abs/2110.08327)


  Partial differential equations (PDEs) are typically used as models of
physical processes but are also of great interest in PDE-based image
processing. However, when it comes to their use in imaging, conventional
numerical methods for solving PDEs tend to require very fine grid resolution
for stability, and as a result have impractically high computational cost. This
work applies BLADE (Best Linear Adaptive Enhancement), a shallow learnable
filtering framework, to PDE solving, and shows that the resulting approach is
efficient and accurate, operating more reliably at coarse grid resolutions than
classical methods. As such, the model can be flexibly used for a wide variety
of problems in imaging.

    

### [[2110.08329] Control Prefixes for Text Generation](http://arxiv.org/abs/2110.08329)


  Prompt learning methods adapt pre-trained language models to downstream
applications by using a task-specific prompt together with the input. Most of
the current work on prompt learning in text generation relies on a shared
dataset-level prompt for all examples in the dataset. We extend this approach
and propose a dynamic method, Control Prefixes, which allows for the inclusion
of conditional input-dependent information in each prompt. Control Prefixes is
at the intersection of prompt learning and controlled generation, empowering
the model to have finer-grained control during text generation. The method
incorporates attribute-level learnable representations into different layers of
a pre-trained transformer, allowing for the generated text to be guided in a
particular direction. We provide a systematic evaluation of the technique and
apply it to five datasets from the GEM benchmark for natural language
generation (NLG). We present state-of-the-art results on several data-to-text
datasets, including WebNLG.

    

### [[2110.08330] Nothing Wasted: Full Contribution Enforcement in Federated Edge Learning](http://arxiv.org/abs/2110.08330)


  The explosive amount of data generated at the network edge makes mobile edge
computing an essential technology to support real-time applications, calling
for powerful data processing and analysis provided by machine learning (ML)
techniques. In particular, federated edge learning (FEL) becomes prominent in
securing the privacy of data owners by keeping the data locally used to train
ML models. Existing studies on FEL either utilize in-process optimization or
remove unqualified participants in advance. In this paper, we enhance the
collaboration from all edge devices in FEL to guarantee that the ML model is
trained using all available local data to accelerate the learning process. To
that aim, we propose a collective extortion (CE) strategy under the
imperfect-information multi-player FEL game, which is proved to be effective in
helping the server efficiently elicit the full contribution of all devices
without worrying about suffering from any economic loss. Technically, our
proposed CE strategy extends the classical extortion strategy in controlling
the proportionate share of expected utilities for a single opponent to the
swiftly homogeneous control over a group of players, which further presents an
attractive trait of being impartial for all participants. Moreover, the CE
strategy enriches the game theory hierarchy, facilitating a wider application
scope of the extortion strategy. Both theoretical analysis and experimental
evaluations validate the effectiveness and fairness of our proposed scheme.

    

### [[2110.08331] A New Approach for Interpretability and Reliability in Clinical Risk Prediction: Acute Coronary Syndrome Scenario](http://arxiv.org/abs/2110.08331)


  We intend to create a new risk assessment methodology that combines the best
characteristics of both risk score and machine learning models. More
specifically, we aim to develop a method that, besides having a good
performance, offers a personalized model and outcome for each patient, presents
high interpretability, and incorporates an estimation of the prediction
reliability which is not usually available. By combining these features in the
same approach we expect that it can boost the confidence of physicians to use
such a tool in their daily activity. In order to achieve the mentioned goals, a
three-step methodology was developed: several rules were created by
dichotomizing risk factors; such rules were trained with a machine learning
classifier to predict the acceptance degree of each rule (the probability that
the rule is correct) for each patient; that information was combined and used
to compute the risk of mortality and the reliability of such prediction. The
methodology was applied to a dataset of patients admitted with any type of
acute coronary syndromes (ACS), to assess the 30-days all-cause mortality risk.
The performance was compared with state-of-the-art approaches: logistic
regression (LR), artificial neural network (ANN), and clinical risk score model
(Global Registry of Acute Coronary Events - GRACE). The proposed approach
achieved testing results identical to the standard LR, but offers superior
interpretability and personalization; it also significantly outperforms the
GRACE risk model and the standard ANN model. The calibration curve also
suggests a very good generalization ability of the obtained model as it
approaches the ideal curve. Finally, the reliability estimation of individual
predictions presented a great correlation with the misclassifications rate.
Those properties may have a beneficial application in other clinical scenarios
as well. [abridged]

    

### [[2110.08338] Exploratory Lagrangian-Based Particle Tracing Using Deep Learning](http://arxiv.org/abs/2110.08338)


  Time-varying vector fields produced by computational fluid dynamics
simulations are often prohibitively large and pose challenges for accurate
interactive analysis and exploration. To address these challenges, reduced
Lagrangian representations have been increasingly researched as a means to
improve scientific time-varying vector field exploration capabilities. This
paper presents a novel deep neural network-based particle tracing method to
explore time-varying vector fields represented by Lagrangian flow maps. In our
workflow, in situ processing is first utilized to extract Lagrangian flow maps,
and deep neural networks then use the extracted data to learn flow field
behavior. Using a trained model to predict new particle trajectories offers a
fixed small memory footprint and fast inference. To demonstrate and evaluate
the proposed method, we perform an in-depth study of performance using a
well-known analytical data set, the Double Gyre. Our study considers two flow
map extraction strategies as well as the impact of the number of training
samples and integration durations on efficacy, evaluates multiple sampling
options for training and testing and informs hyperparameter settings. Overall,
we find our method requires a fixed memory footprint of 10.5 MB to encode a
Lagrangian representation of a time-varying vector field while maintaining
accuracy. For post hoc analysis, loading the trained model costs only two
seconds, significantly reducing the burden of I/O when reading data for
visualization. Moreover, our parallel implementation can infer one hundred
locations for each of two thousand new pathlines across the entire temporal
resolution in 1.3 seconds using one NVIDIA Titan RTX GPU.

    

### [[2110.08340] Return migration of German-affiliated researchers: Analyzing departure and return by gender, cohort, and discipline using Scopus bibliometric data 1996-2020](http://arxiv.org/abs/2110.08340)


  The international migration of researchers is a highly prized dimension of
scientific mobility and motivates considerable policy debate. However, tracking
migration life courses of researchers is challenging due to data limitations.
In this study, we use Scopus bibliometric data on 8 million publications from
1.1 million researchers who have published at least once with an affiliation
address from Germany in 1996-2020. We describe several key steps and algorithms
we develop that enable us to construct the partial life histories of published
researchers in this period. These tools allow us to explore both the
out-migration of researchers with German affiliations as well as the subsequent
return of a share of this group - the returnees. Our analyses shed light on
important career stages and gender disparities between researchers who remain
in Germany and those who both migrate out and those who eventually return.
Return migration streams are even more gender imbalanced and point to the
importance of additional efforts to attract female researchers back to Germany.
We document a slightly declining trend in return migration with cohorts which,
for most disciplines, is associated with decreasing German collaboration ties
among cohorts of researchers who leave Germany. Also, gender disparities for
the most gender imbalanced disciplines are unlikely to be mitigated by return
migration given the gender compositions in cohorts of researchers who leave
Germany and those who return. This analysis reveals new dimensions of scholarly
migration by investigating the return migration of published researchers which
is critical for science policy development.

    

### [[2110.08350] Differentiable Network Pruning for Microcontrollers](http://arxiv.org/abs/2110.08350)


  Embedded and personal IoT devices are powered by microcontroller units
(MCUs), whose extreme resource scarcity is a major obstacle for applications
relying on on-device deep learning inference. Orders of magnitude less storage,
memory and computational capacity, compared to what is typically required to
execute neural networks, impose strict structural constraints on the network
architecture and call for specialist model compression methodology. In this
work, we present a differentiable structured network pruning method for
convolutional neural networks, which integrates a model's MCU-specific resource
usage and parameter importance feedback to obtain highly compressed yet
accurate classification models. Our methodology (a) improves key resource usage
of models up to 80x; (b) prunes iteratively while a model is trained, resulting
in little to no overhead or even improved training time; (c) produces
compressed models with matching or improved resource usage up to 1.7x in less
time compared to prior MCU-specific methods. Compressed models are available
for download.

    

### [[2110.08353] Revisiting Popularity and Demographic Biases in Recommender Evaluation and Effectiveness](http://arxiv.org/abs/2110.08353)


  Recommendation algorithms are susceptible to popularity bias: a tendency to
recommend popular items even when they fail to meet user needs. A related issue
is that the recommendation quality can vary by demographic groups. Marginalized
groups or groups that are under-represented in the training data may receive
less relevant recommendations from these algorithms compared to others. In a
recent study, Ekstrand et al. investigate how recommender performance varies
according to popularity and demographics, and find statistically significant
differences in recommendation utility between binary genders on two datasets,
and significant effects based on age on one dataset. Here we reproduce those
results and extend them with additional analyses. We find statistically
significant differences in recommender performance by both age and gender. We
observe that recommendation utility steadily degrades for older users, and is
lower for women than men. We also find that the utility is higher for users
from countries with more representation in the dataset. In addition, we find
that total usage and the popularity of consumed content are strong predictors
of recommender performance and also vary significantly across demographic
groups.

    

### [[2110.08367] Dropping diversity of products of large US firms: Models and measures](http://arxiv.org/abs/2110.08367)


  It is widely assumed that in our lifetimes the products available in the
global economy have become more diverse. This assumption is difficult to
investigate directly, however, because it is difficult to collect the necessary
data about every product in an economy each year. We solve this problem by
mining publicly available textual descriptions of the products of every large
US firms each year from 1997 to 2017. Although many aspects of economic
productivity have been steadily rising during this period, our text-based
measurements show that the diversity of the products of at least large US firms
has steadily declined. This downward trend is visible using a variety of
product diversity metrics, including some that depend on a measurement of the
similarity of the products of every single pair of firms. The current state of
the art in comprehensive and detailed firm-similarity measurements is a Boolean
word vector model due to Hoberg and Phillips. We measure diversity using
firm-similarities from this Boolean model and two more sophisticated variants,
and we consistently observe a significant dropping trend in product diversity.
These results make it possible to frame and start to test specific hypotheses
for explaining the dropping product diversity trend.

    

### [[2110.08378] FedSLD: Federated Learning with Shared Label Distribution for Medical Image Classification](http://arxiv.org/abs/2110.08378)


  Machine learning in medical research, by nature, needs careful attention on
obeying the regulations of data privacy, making it difficult to train a machine
learning model over gathered data from different medical centers. Failure of
leveraging data of the same kind may result in poor generalizability for the
trained model. Federated learning (FL) enables collaboratively training a joint
model while keeping the data decentralized for multiple medical centers.
However, federated optimizations often suffer from the heterogeneity of the
data distribution across medical centers. In this work, we propose Federated
Learning with Shared Label Distribution (FedSLD) for classification tasks, a
method that assumes knowledge of the label distributions for all the
participating clients in the federation. FedSLD adjusts the contribution of
each data sample to the local objective during optimization given knowledge of
the distribution, mitigating the instability brought by data heterogeneity
across all clients. We conduct extensive experiments on four publicly available
image datasets with different types of non-IID data distributions. Our results
show that FedSLD achieves better convergence performance than the compared
leading FL optimization algorithms, increasing the test accuracy by up to 5.50
percentage points.

    

### [[2110.08382] A Neural Network Ensemble Approach to System Identification](http://arxiv.org/abs/2110.08382)


  We present a new algorithm for learning unknown governing equations from
trajectory data, using and ensemble of neural networks. Given samples of
solutions $x(t)$ to an unknown dynamical system $\dot{x}(t)=f(t,x(t))$, we
approximate the function $f$ using an ensemble of neural networks. We express
the equation in integral form and use Euler method to predict the solution at
every successive time step using at each iteration a different neural network
as a prior for $f$. This procedure yields M-1 time-independent networks, where
M is the number of time steps at which $x(t)$ is observed. Finally, we obtain a
single function $f(t,x(t))$ by neural network interpolation. Unlike our earlier
work, where we numerically computed the derivatives of data, and used them as
target in a Lipschitz regularized neural network to approximate $f$, our new
method avoids numerical differentiations, which are unstable in presence of
noise. We test the new algorithm on multiple examples both with and without
noise in the data. We empirically show that generalization and recovery of the
governing equation improve by adding a Lipschitz regularization term in our
loss function and that this method improves our previous one especially in
presence of noise, when numerical differentiation provides low quality target
data. Finally, we compare our results with the method proposed by Raissi, et
al. arXiv:1801.01236 (2018) and with SINDy.

    

### [[2110.08385] Robust Correlation Clustering with Asymmetric Noise](http://arxiv.org/abs/2110.08385)


  Graph clustering problems typically aim to partition the graph nodes such
that two nodes belong to the same partition set if and only if they are
similar. Correlation Clustering is a graph clustering formulation which: (1)
takes as input a signed graph with edge weights representing a
similarity/dissimilarity measure between the nodes, and (2) requires no prior
estimate of the number of clusters in the input graph. However, the
combinatorial optimization problem underlying Correlation Clustering is
NP-hard. In this work, we propose a novel graph generative model, called the
Node Factors Model (NFM), which is based on generating feature
vectors/embeddings for the graph nodes. The graphs generated by the NFM contain
asymmetric noise in the sense that there may exist pairs of nodes in the same
cluster which are negatively correlated. We propose a novel Correlation
Clustering algorithm, called \anormd, using techniques from semidefinite
programming. Using a combination of theoretical and computational results, we
demonstrate that $\texttt{$\ell_2$-norm-diag}$ recovers nodes with sufficiently
strong cluster membership in graph instances generated by the NFM, thereby
making progress towards establishing the provable robustness of our proposed
algorithm.

    

### [[2110.08393] A Bayesian Approach for Medical Inquiry and Disease Inference in Automated Differential Diagnosis](http://arxiv.org/abs/2110.08393)


  We propose a Bayesian approach for both medical inquiry and disease
inference, the two major phases in differential diagnosis. Unlike previous work
that simulates data from given probabilities and uses ML algorithms on them, we
directly use the Quick Medical Reference (QMR) belief network, and apply
Bayesian inference in the inference phase and Bayesian experimental design in
the inquiry phase. Moreover, we improve the inquiry phase by extending the
Bayesian experimental design framework from one-step search to multi-step
search. Our approach has some practical advantages as it is interpretable, free
of costly training, and able to adapt to new changes without any additional
effort. Our experiments show that our approach achieves new state-of-the-art
results on two simulated datasets, SymCAT and HPO, and competitive results on
two diagnosis dialogue datasets, Muzhi and Dxy.

    

### [[2110.08394] Adapt to Adaptation: Learning Personalization for Cross-Silo Federated Learning](http://arxiv.org/abs/2110.08394)


  The goal of conventional federated learning (FL) is to train a global model
for a federation of clients with decentralized data, reducing the systemic
privacy risk of centralized training. The distribution shift across non-IID
datasets, also known as the data heterogeneity, often poses a challenge for
this one-global-model-fits-all solution. In this work, we propose APPLE, a
personalized cross-silo FL framework that adaptively learns how much each
client can benefit from other clients' models. We also introduce a method to
flexibly control the focus of training APPLE between global and local
objectives. We empirically evaluate our method's convergence and generalization
behavior and performed extensive experiments on two benchmark datasets and two
medical imaging datasets under two non-IID settings. The results show that the
proposed personalized FL framework, APPLE, achieves state-of-the-art
performance compared to several other personalized FL approaches in the
literature.

    

### [[2110.08396] Comparing Human and Machine Bias in Face Recognition](http://arxiv.org/abs/2110.08396)


  Much recent research has uncovered and discussed serious concerns of bias in
facial analysis technologies, finding performance disparities between groups of
people based on perceived gender, skin type, lighting condition, etc. These
audits are immensely important and successful at measuring algorithmic bias but
have two major challenges: the audits (1) use facial recognition datasets which
lack quality metadata, like LFW and CelebA, and (2) do not compare their
observed algorithmic bias to the biases of their human alternatives. In this
paper, we release improvements to the LFW and CelebA datasets which will enable
future researchers to obtain measurements of algorithmic bias that are not
tainted by major flaws in the dataset (e.g. identical images appearing in both
the gallery and test set). We also use these new data to develop a series of
challenging facial identification and verification questions that we
administered to various algorithms and a large, balanced sample of human
reviewers. We find that both computer models and human survey participants
perform significantly better at the verification task, generally obtain lower
accuracy rates on dark-skinned or female subjects for both tasks, and obtain
higher accuracy rates when their demographics match that of the question.
Computer models are observed to achieve a higher level of accuracy than the
survey participants on both tasks and exhibit bias to similar degrees as the
human survey participants.

    

### [[2110.08406] Surrogate- and invariance-boosted contrastive learning for data-scarce applications in science](http://arxiv.org/abs/2110.08406)


  Deep learning techniques have been increasingly applied to the natural
sciences, e.g., for property prediction and optimization or material discovery.
A fundamental ingredient of such approaches is the vast quantity of labelled
data needed to train the model; this poses severe challenges in data-scarce
settings where obtaining labels requires substantial computational or labor
resources. Here, we introduce surrogate- and invariance-boosted contrastive
learning (SIB-CL), a deep learning framework which incorporates three
``inexpensive'' and easily obtainable auxiliary information sources to overcome
data scarcity. Specifically, these are: 1)~abundant unlabeled data, 2)~prior
knowledge of symmetries or invariances and 3)~surrogate data obtained at
near-zero cost. We demonstrate SIB-CL's effectiveness and generality on various
scientific problems, e.g., predicting the density-of-states of 2D photonic
crystals and solving the 3D time-independent Schrodinger equation. SIB-CL
consistently results in orders of magnitude reduction in the number of labels
needed to achieve the same network accuracies.

    

### [[2110.08412] Evaluating the Faithfulness of Importance Measures in NLP by Recursively Masking Allegedly Important Tokens and Retraining](http://arxiv.org/abs/2110.08412)


  To explain NLP models, many methods inform which inputs tokens are important
for a prediction. However, an open question is if these methods accurately
reflect the model's logic, a property often called faithfulness. In this work,
we adapt and improve a recently proposed faithfulness benchmark from computer
vision called ROAR (RemOve And Retrain), by Hooker et al. (2019).
We improve ROAR by recursively removing dataset redundancies, which otherwise
interfere with ROAR. We adapt and apply ROAR, to popular NLP importance
measures, namely attention, gradient, and integrated gradients. Additionally,
we use mutual information as an additional baseline. Evaluation is done on a
suite of classification tasks often used in the faithfulness of attention
literature. Finally, we propose a scalar faithfulness metric, which makes it
easy to compare results across papers.
We find that, importance measures considered to be unfaithful for computer
vision tasks perform favorably for NLP tasks, the faithfulness of an importance
measure is task-dependent, and the computational overhead of integrated
gradient is rarely justified.

    

### [[2110.08413] Invariant Language Modeling](http://arxiv.org/abs/2110.08413)


  Modern pretrained language models are critical components of NLP pipelines.
Yet, they suffer from spurious correlations, poor out-of-domain generalization,
and biases. Inspired by recent progress in causal machine learning, in
particular the invariant risk minimization (IRM) paradigm, we propose invariant
language modeling, a framework for learning invariant representations that
generalize better across multiple environments. In particular, we adapt a
game-theoretic implementation of IRM (IRM-games) to language models, where the
invariance emerges from a specific training schedule in which all the
environments compete to optimize their own environment-specific loss by
updating subsets of the model in a round-robin fashion. In a series of
controlled experiments, we demonstrate the ability of our method to (i) remove
structured noise, (ii) ignore specific spurious correlations without affecting
global performance, and (iii) achieve better out-of-domain generalization.
These benefits come with a negligible computational overhead compared to
standard training, do not require changing the local loss, and can be applied
to any language model architecture. We believe this framework is promising to
help mitigate spurious correlations and biases in language models.

    

### [[2110.08418] Nuances in Margin Conditions Determine Gains in Active Learning](http://arxiv.org/abs/2110.08418)


  We consider nonparametric classification with smooth regression functions,
where it is well known that notions of margin in $E[Y|X]$ determine fast or
slow rates in both active and passive learning. Here we elucidate a striking
distinction between the two settings. Namely, we show that some seemingly
benign nuances in notions of margin -- involving the uniqueness of the Bayes
classifier, and which have no apparent effect on rates in passive learning --
determine whether or not any active learner can outperform passive learning
rates. In particular, for Audibert-Tsybakov's margin condition (allowing
general situations with non-unique Bayes classifiers), no active learner can
gain over passive learning in commonly studied settings where the marginal on
$X$ is near uniform. Our results thus negate the usual intuition from past
literature that active rates should improve over passive rates in nonparametric
settings.

    

### [[2110.08419] What do Compressed Large Language Models Forget? Robustness Challenges in Model Compression](http://arxiv.org/abs/2110.08419)


  Recent works have focused on compressing pre-trained language models (PLMs)
like BERT where the major focus has been to improve the compressed model
performance for downstream tasks. However, there has been no study in analyzing
the impact of compression on the generalizability and robustness of these
models. Towards this end, we study two popular model compression techniques
including knowledge distillation and pruning and show that compressed models
are significantly less robust than their PLM counterparts on adversarial test
sets although they obtain similar performance on in-distribution development
sets for a task. Further analysis indicates that the compressed models overfit
on the easy samples and generalize poorly on the hard ones. We further leverage
this observation to develop a regularization strategy for model compression
based on sample uncertainty. Experimental results on several natural language
understanding tasks demonstrate our mitigation framework to improve both the
adversarial generalization as well as in-distribution task performance of the
compressed models.

    

### [[2110.08420] Information-Theoretic Measures of Dataset Difficulty](http://arxiv.org/abs/2110.08420)


  Estimating the difficulty of a dataset typically involves comparing
state-of-the-art models to humans; the bigger the performance gap, the harder
the dataset is said to be. Not only is this framework informal, but it also
provides little understanding of how difficult each instance is, or what
attributes make it difficult for a given model. To address these problems, we
propose an information-theoretic perspective, framing dataset difficulty as the
absence of $\textit{usable information}$. Measuring usable information is as
easy as measuring performance, but has certain theoretical advantages. While
the latter only allows us to compare different models w.r.t the same dataset,
the former also allows us to compare different datasets w.r.t the same model.
We then introduce $\textit{pointwise}$ $\mathcal{V}-$$\textit{information}$
(PVI) for measuring the difficulty of individual instances, where instances
with higher PVI are easier for model $\mathcal{V}$. By manipulating the input
before measuring usable information, we can understand $\textit{why}$ a dataset
is easy or difficult for a given model, which we use to discover annotation
artefacts in widely-used benchmarks.

    

### [[2110.08421] Dataset Knowledge Transfer for Class-Incremental Learning without Memory](http://arxiv.org/abs/2110.08421)


  Incremental learning enables artificial agents to learn from sequential data.
While important progress was made by exploiting deep neural networks,
incremental learning remains very challenging. This is particularly the case
when no memory of past data is allowed and catastrophic forgetting has a strong
negative effect. We tackle class-incremental learning without memory by
adapting prediction bias correction, a method which makes predictions of past
and new classes more comparable. It was proposed when a memory is allowed and
cannot be directly used without memory, since samples of past classes are
required. We introduce a two-step learning process which allows the transfer of
bias correction parameters between reference and target datasets. Bias
correction is first optimized offline on reference datasets which have an
associated validation memory. The obtained correction parameters are then
transferred to target datasets, for which no memory is available. The second
contribution is to introduce a finer modeling of bias correction by learning
its parameters per incremental state instead of the usual past vs. new class
modeling. The proposed dataset knowledge transfer is applicable to any
incremental method which works without memory. We test its effectiveness by
applying it to four existing methods. Evaluation with four target datasets and
different configurations shows consistent improvement, with practically no
computational and memory overhead.

    

### [[2110.08424] Deep learning-based detection of intravenous contrast in computed tomography scans](http://arxiv.org/abs/2110.08424)


  Purpose: Identifying intravenous (IV) contrast use within CT scans is a key
component of data curation for model development and testing. Currently, IV
contrast is poorly documented in imaging metadata and necessitates manual
correction and annotation by clinician experts, presenting a major barrier to
imaging analyses and algorithm deployment. We sought to develop and validate a
convolutional neural network (CNN)-based deep learning (DL) platform to
identify IV contrast within CT scans. Methods: For model development and
evaluation, we used independent datasets of CT scans of head, neck (HN) and
lung cancer patients, totaling 133,480 axial 2D scan slices from 1,979 CT scans
manually annotated for contrast presence by clinical experts. Five different DL
models were adopted and trained in HN training datasets for slice-level
contrast detection. Model performances were evaluated on a hold-out set and on
an independent validation set from another institution. DL models was then
fine-tuned on chest CT data and externally validated on a separate chest CT
dataset. Results: Initial DICOM metadata tags for IV contrast were missing or
erroneous in 1,496 scans (75.6%). The EfficientNetB4-based model showed the
best overall detection performance. For HN scans, AUC was 0.996 in the internal
validation set (n = 216) and 1.0 in the external validation set (n = 595). The
fine-tuned model on chest CTs yielded an AUC: 1.0 for the internal validation
set (n = 53), and AUC: 0.980 for the external validation set (n = 402).
Conclusion: The DL model could accurately detect IV contrast in both HN and
chest CT scans with near-perfect performance.

    

### [[2110.08430] Metadata Shaping: Natural Language Annotations for the Tail](http://arxiv.org/abs/2110.08430)


  Language models (LMs) have made remarkable progress, but still struggle to
generalize beyond the training data to rare linguistic patterns. Since rare
entities and facts are prevalent in the queries users submit to popular
applications such as search and personal assistant systems, improving the
ability of LMs to reliably capture knowledge over rare entities is a pressing
challenge studied in significant prior work. Noticing that existing approaches
primarily modify the LM architecture or introduce auxiliary objectives to
inject useful entity knowledge, we ask to what extent we could match the
quality of these architectures using a base LM architecture, and only changing
the data? We propose metadata shaping, a method in which readily available
metadata, such as entity descriptions and categorical tags, are appended to
examples based on information theoretic metrics. Intuitively, if metadata
corresponding to popular entities overlap with metadata for rare entities, the
LM may be able to better reason about the rare entities using patterns learned
from similar popular entities. On standard entity-rich tasks (TACRED, FewRel,
OpenEntity), with no changes to the LM whatsoever, metadata shaping exceeds the
BERT-baseline by up to 5.3 F1 points, and achieves or competes with
state-of-the-art results. We further show the improvements are up to 10x larger
on examples containing tail versus popular entities.

    

### [[2110.08432] Meta-Learning with Adjoint Methods](http://arxiv.org/abs/2110.08432)


  Model Agnostic Meta-Learning (MAML) is widely used to find a good
initialization for a family of tasks. Despite its success, a critical challenge
in MAML is to calculate the gradient w.r.t the initialization of a long
training trajectory for the sampled tasks, because the computation graph can
rapidly explode and the computational cost is very expensive. To address this
problem, we propose Adjoint MAML (A-MAML). We view gradient descent in the
inner optimization as the evolution of an Ordinary Differential Equation (ODE).
To efficiently compute the gradient of the validation loss w.r.t the
initialization, we use the adjoint method to construct a companion, backward
ODE. To obtain the gradient w.r.t the initialization, we only need to run the
standard ODE solver twice -- one is forward in time that evolves a long
trajectory of gradient flow for the sampled task; the other is backward and
solves the adjoint ODE. We need not create or expand any intermediate
computational graphs, adopt aggressive approximations, or impose proximal
regularizers in the training loss. Our approach is cheap, accurate, and
adaptable to different trajectory lengths. We demonstrate the advantage of our
approach in both synthetic and real-world meta-learning tasks.

    

### [[2110.08438] Unsupervised Natural Language Inference Using PHL Triplet Generation](http://arxiv.org/abs/2110.08438)


  Transformer-based models have achieved impressive performance on various
Natural Language Inference (NLI) benchmarks, when trained on respective
training datasets. However, in certain cases, training samples may not be
available or collecting them could be time-consuming and resource-intensive. In
this work, we address this challenge and present an explorative study on
unsupervised NLI, a paradigm in which no human-annotated training samples are
available. We investigate NLI under three challenging settings: PH, P, and NPH
that differ in the extent of unlabeled data available for learning. As a
solution, we propose a procedural data generation approach that leverages a set
of sentence transformations to collect PHL (Premise, Hypothesis, Label)
triplets for training NLI models, bypassing the need for human-annotated
training datasets. Comprehensive experiments show that this approach results in
accuracies of 66.75%, 65.9%, 65.39% in PH, P, NPH settings respectively,
outperforming all existing baselines. Furthermore, fine-tuning our models with
as little as ~0.1% of the training dataset (500 samples) leads to 12.2% higher
accuracy than the model trained from scratch on the same 500 instances.

    

### [[2110.08440] Online Target Q-learning with Reverse Experience Replay: Efficiently finding the Optimal Policy for Linear MDPs](http://arxiv.org/abs/2110.08440)


  Q-learning is a popular Reinforcement Learning (RL) algorithm which is widely
used in practice with function approximation \citep{mnih2015human}. In
contrast, existing theoretical results are pessimistic about Q-learning. For
example, \citep{baird1995residual} shows that Q-learning does not converge even
with linear function approximation for linear MDPs. Furthermore, even for
tabular MDPs with synchronous updates, Q-learning was shown to have sub-optimal
sample complexity \citep{li2021q,azar2013minimax}. The goal of this work is to
bridge the gap between practical success of Q-learning and the relatively
pessimistic theoretical results. The starting point of our work is the
observation that in practice, Q-learning is used with two important
modifications: (i) training with two networks, called online network and target
network simultaneously (online target learning, or OTL) , and (ii) experience
replay (ER) \citep{mnih2015human}. While they have been observed to play a
significant role in the practical success of Q-learning, a thorough theoretical
understanding of how these two modifications improve the convergence behavior
of Q-learning has been missing in literature. By carefully combining Q-learning
with OTL and \emph{reverse} experience replay (RER) (a form of experience
replay), we present novel methods Q-Rex and Q-RexDaRe (Q-Rex + data reuse). We
show that Q-Rex efficiently finds the optimal policy for linear MDPs (or more
generally for MDPs with zero inherent Bellman error with linear approximation
(ZIBEL)) and provide non-asymptotic bounds on sample complexity -- the first
such result for a Q-learning method for this class of MDPs under standard
assumptions. Furthermore, we demonstrate that Q-RexDaRe in fact achieves near
optimal sample complexity in the tabular setting, improving upon the existing
results for vanilla Q-learning.

    

### [[2110.08447] TESDA: Transform Enabled Statistical Detection of Attacks in Deep Neural Networks](http://arxiv.org/abs/2110.08447)


  Deep neural networks (DNNs) are now the de facto choice for computer vision
tasks such as image classification. However, their complexity and "black box"
nature often renders the systems they're deployed in vulnerable to a range of
security threats. Successfully identifying such threats, especially in
safety-critical real-world applications is thus of utmost importance, but still
very much an open problem. We present TESDA, a low-overhead, flexible, and
statistically grounded method for {online detection} of attacks by exploiting
the discrepancies they cause in the distributions of intermediate layer
features of DNNs. Unlike most prior work, we require neither dedicated hardware
to run in real-time, nor the presence of a Trojan trigger to detect
discrepancies in behavior. We empirically establish our method's usefulness and
practicality across multiple architectures, datasets and diverse attacks,
consistently achieving detection coverages of above 95% with operation count
overheads as low as 1-2%.

    

### [[2110.08449] Adversarial Attacks on Gaussian Process Bandits](http://arxiv.org/abs/2110.08449)


  Gaussian processes (GP) are a widely-adopted tool used to sequentially
optimize black-box functions, where evaluations are costly and potentially
noisy. Recent works on GP bandits have proposed to move beyond random noise and
devise algorithms robust to adversarial attacks. In this paper, we study this
problem from the attacker's perspective, proposing various adversarial attack
methods with differing assumptions on the attacker's strength and prior
information. Our goal is to understand adversarial attacks on GP bandits from
both a theoretical and practical perspective. We focus primarily on targeted
attacks on the popular GP-UCB algorithm and a related elimination-based
algorithm, based on adversarially perturbing the function $f$ to produce
another function $\tilde{f}$ whose optima are in some region $\mathcal{R}_{\rm
target}$. Based on our theoretical analysis, we devise both white-box attacks
(known $f$) and black-box attacks (unknown $f$), with the former including a
Subtraction attack and Clipping attack, and the latter including an Aggressive
subtraction attack. We demonstrate that adversarial attacks on GP bandits can
succeed in forcing the algorithm towards $\mathcal{R}_{\rm target}$ even with a
low attack budget, and we compare our attacks' performance and efficiency on
several real and synthetic functions.

    

### [[2110.08450] Accelerating Training and Inference of Graph Neural Networks with Fast Sampling and Pipelining](http://arxiv.org/abs/2110.08450)


  Improving the training and inference performance of graph neural networks
(GNNs) is faced with a challenge uncommon in general neural networks: creating
mini-batches requires a lot of computation and data movement due to the
exponential growth of multi-hop graph neighborhoods along network layers. Such
a unique challenge gives rise to a diverse set of system design choices. We
argue in favor of performing mini-batch training with neighborhood sampling in
a distributed multi-GPU environment, under which we identify major performance
bottlenecks hitherto under-explored by developers: mini-batch preparation and
transfer. We present a sequence of improvements to mitigate these bottlenecks,
including a performance-engineered neighborhood sampler, a shared-memory
parallelization strategy, and the pipelining of batch transfer with GPU
computation. We also conduct an empirical analysis that supports the use of
sampling for inference, showing that test accuracies are not materially
compromised. Such an observation unifies training and inference, simplifying
model implementation. We report comprehensive experimental results with several
benchmark data sets and GNN architectures, including a demonstration that, for
the ogbn-papers100M data set, our system SALIENT achieves a speedup of 3x over
a standard PyTorch-Geometric implementation with a single GPU and a further 8x
parallel speedup with 16 GPUs. Therein, training a 3-layer GraphSAGE model with
sampling fanout (15, 10, 5) takes 2.0 seconds per epoch and inference with
fanout (20, 20, 20) takes 2.4 seconds, attaining test accuracy 64.58%.

    

### [[2110.08465] A Heterogeneous Graph Based Framework for Multimodal Neuroimaging Fusion Learning](http://arxiv.org/abs/2110.08465)


  Here, we present a Heterogeneous Graph neural network for Multimodal
neuroimaging fusion learning (HGM). Traditional GNN-based models usually assume
the brain network is a homogeneous graph with single type of nodes and edges.
However, vast literatures have shown the heterogeneity of the human brain
especially between the two hemispheres. Homogeneous brain network is
insufficient to model the complicated brain state. Therefore, in this work we
firstly model the brain network as heterogeneous graph with multi-type nodes
(i.e., left and right hemispheric nodes) and multi-type edges (i.e., intra- and
inter-hemispheric edges). Besides, we also propose a self-supervised
pre-training strategy based on heterogeneou brain network to address the
overfitting problem due to the complex model and small sample size. Our results
on two datasets show the superiority of proposed model over other multimodal
methods for disease prediction task. Besides, ablation experiments show that
our model with pre-training strategy can alleviate the problem of limited
training sample size.

    

### [[2110.08470] Case-based Reasoning for Better Generalization in Text-Adventure Games](http://arxiv.org/abs/2110.08470)


  Text-based games (TBG) have emerged as promising environments for driving
research in grounded language understanding and studying problems like
generalization and sample efficiency. Several deep reinforcement learning (RL)
methods with varying architectures and learning schemes have been proposed for
TBGs. However, these methods fail to generalize efficiently, especially under
distributional shifts. In a departure from deep RL approaches, in this paper,
we propose a general method inspired by case-based reasoning to train agents
and generalize out of the training distribution. The case-based reasoner
collects instances of positive experiences from the agent's interaction with
the world in the past and later reuses the collected experiences to act
efficiently. The method can be applied in conjunction with any existing
on-policy neural agent in the literature for TBGs. Our experiments show that
the proposed approach consistently improves existing methods, obtains good
out-of-distribution generalization, and achieves new state-of-the-art results
on widely used environments.

    

### [[2110.08471] Fast Projection onto the Capped Simplex withApplications to Sparse Regression in Bioinformatics](http://arxiv.org/abs/2110.08471)


  We consider the problem of projecting a vector onto the so-called k-capped
simplex, which is a hyper-cube cut by a hyperplane. For an n-dimensional input
vector with bounded elements, we found that a simple algorithm based on
Newton's method is able to solve the projection problem to high precision with
a complexity roughly about O(n), which has a much lower computational cost
compared with the existing sorting-based methods proposed in the literature. We
provide a theory for partial explanation and justification of the method.
We demonstrate that the proposed algorithm can produce a solution of the
projection problem with high precision on large scale datasets, and the
algorithm is able to significantly outperform the state-of-the-art methods in
terms of runtime (about 6-8 times faster than a commercial software with
respect to CPU time for input vector with 1 million variables or more).
We further illustrate the effectiveness of the proposed algorithm on solving
sparse regression in a bioinformatics problem. Empirical results on the GWAS
dataset (with 1,500,000 single-nucleotide polymorphisms) show that, when using
the proposed method to accelerate the Projected Quasi-Newton (PQN) method, the
accelerated PQN algorithm is able to handle huge-scale regression problem and
it is more efficient (about 3-6 times faster) than the current state-of-the-art
methods.

    

### [[2110.08477] FedMM: Saddle Point Optimization for Federated Adversarial Domain Adaptation](http://arxiv.org/abs/2110.08477)


  Federated adversary domain adaptation is a unique distributed minimax
training task due to the prevalence of label imbalance among clients, with each
client only seeing a subset of the classes of labels required to train a global
model. To tackle this problem, we propose a distributed minimax optimizer
referred to as FedMM, designed specifically for the federated adversary domain
adaptation problem. It works well even in the extreme case where each client
has different label classes and some clients only have unsupervised tasks. We
prove that FedMM ensures convergence to a stationary point with domain-shifted
unsupervised data. On a variety of benchmark datasets, extensive experiments
show that FedMM consistently achieves either significant communication savings
or significant accuracy improvements over federated optimizers based on the
gradient descent ascent (GDA) algorithm. When training from scratch, for
example, it outperforms other GDA based federated average methods by around
$20\%$ in accuracy over the same communication rounds; and it consistently
outperforms when training from pre-trained models with an accuracy improvement
from $5.4\%$ to $9\%$ for different networks.

    

### [[2110.08483] Streaming Decision Trees and Forests](http://arxiv.org/abs/2110.08483)


  Machine learning has successfully leveraged modern data and provided
computational solutions to innumerable real-world problems, including physical
and biomedical discoveries. Currently, estimators could handle both scenarios
with all samples available and situations requiring continuous updates.
However, there is still room for improvement on streaming algorithms based on
batch decision trees and random forests, which are the leading methods in batch
data tasks. In this paper, we explore the simplest partial fitting algorithm to
extend batch trees and test our models: stream decision tree (SDT) and stream
decision forest (SDF) on three classification tasks of varying complexities.
For reference, both existing streaming trees (Hoeffding trees and Mondrian
forests) and batch estimators are included in the experiments. In all three
tasks, SDF consistently produces high accuracy, whereas existing estimators
encounter space restraints and accuracy fluctuations. Thus, our streaming trees
and forests show great potential for further improvements, which are good
candidates for solving problems like distribution drift and transfer learning.

    

### [[2110.08488] Lifelong Topological Visual Navigation](http://arxiv.org/abs/2110.08488)


  The ability for a robot to navigate with only the use of vision is appealing
due to its simplicity. Traditional vision-based navigation approaches required
a prior map-building step that was arduous and prone to failure, or could only
exactly follow previously executed trajectories. Newer learning-based visual
navigation techniques reduce the reliance on a map and instead directly learn
policies from image inputs for navigation. There are currently two prevalent
paradigms: end-to-end approaches forego the explicit map representation
entirely, and topological approaches which still preserve some loose
connectivity of the space. However, while end-to-end methods tend to struggle
in long-distance navigation tasks, topological map-based solutions are prone to
failure due to spurious edges in the graph. In this work, we propose a
learning-based topological visual navigation method with graph update
strategies that improve lifelong navigation performance over time. We take
inspiration from sampling-based planning algorithms to build image-based
topological graphs, resulting in sparser graphs yet with higher navigation
performance compared to baseline methods. Also, unlike controllers that learn
from fixed training environments, we show that our model can be finetuned using
a relatively small dataset from the real-world environment where the robot is
deployed. We further assess performance of our system in real-world
deployments.

    

### [[2110.08500] On Model Selection Consistency of Lasso for High-Dimensional Ising Models on Tree-like Graphs](http://arxiv.org/abs/2110.08500)


  We consider the problem of high-dimensional Ising model selection using
neighborhood-based least absolute shrinkage and selection operator (Lasso). It
is rigorously proved that under some mild coherence conditions on the
population covariance matrix of the Ising model, consistent model selection can
be achieved with sample sizes $n=\Omega{(d^3\log{p})}$ for any tree-like graph
in the paramagnetic phase, where $p$ is the number of variables and $d$ is the
maximum node degree. When the same conditions are imposed directly on the
sample covariance matrices, it is shown that a reduced sample size
$n=\Omega{(d^2\log{p})}$ suffices. The obtained sufficient conditions for
consistent model selection with Lasso are the same in the scaling of the sample
complexity as that of $\ell_1$-regularized logistic regression. Given the
popularity and efficiency of Lasso, our rigorous analysis provides a
theoretical backing for its practical use in Ising model selection.

    

### [[2110.08505] Mode and Ridge Estimation in Euclidean and Directional Product Spaces: A Mean Shift Approach](http://arxiv.org/abs/2110.08505)


  The set of local modes and the ridge lines estimated from a dataset are
important summary characteristics of the data-generating distribution. In this
work, we consider estimating the local modes and ridges from point cloud data
in a product space with two or more Euclidean/directional metric spaces.
Specifically, we generalize the well-known (subspace constrained) mean shift
algorithm to the product space setting and illuminate some pitfalls in such
generalization. We derive the algorithmic convergence of the proposed method,
provide practical guidelines on the implementation, and demonstrate its
effectiveness on both simulated and real datasets.

    

### [[2110.08509] BAPGAN: GAN-based Bone Age Progression of Femur and Phalange X-ray Images](http://arxiv.org/abs/2110.08509)


  Convolutional Neural Networks play a key role in bone age assessment for
investigating endocrinology, genetic, and growth disorders under various
modalities and body regions. However, no researcher has tackled bone age
progression/regression despite its valuable potential applications:
bone-related disease diagnosis, clinical knowledge acquisition, and museum
education. Therefore, we propose Bone Age Progression Generative Adversarial
Network (BAPGAN) to progress/regress both femur/phalange X-ray images while
preserving identity and realism. We exhaustively confirm the BAPGAN's clinical
potential via Frechet Inception Distance, Visual Turing Test by two expert
orthopedists, and t-Distributed Stochastic Neighbor Embedding.

    

### [[2110.08510] DFW-PP: Dynamic Feature Weighting based Popularity Prediction for Social Media Content](http://arxiv.org/abs/2110.08510)


  The increasing popularity of social media platforms makes it important to
study user engagement, which is a crucial aspect of any marketing strategy or
business model. The over-saturation of content on social media platforms has
persuaded us to identify the important factors that affect content popularity.
This comes from the fact that only an iota of the humongous content available
online receives the attention of the target audience. Comprehensive research
has been done in the area of popularity prediction using several Machine
Learning techniques. However, we observe that there is still significant scope
for improvement in analyzing the social importance of media content. We propose
the DFW-PP framework, to learn the importance of different features that vary
over time. Further, the proposed method controls the skewness of the
distribution of the features by applying a log-log normalization. The proposed
method is experimented with a benchmark dataset, to show promising results. The
code will be made publicly available at
this https URL.

    

### [[2110.08514] Analyzing Dynamic Adversarial Training Data in the Limit](http://arxiv.org/abs/2110.08514)


  To create models that are robust across a wide range of test inputs, training
datasets should include diverse examples that span numerous phenomena. Dynamic
adversarial data collection (DADC), where annotators craft examples that
challenge continually improving models, holds promise as an approach for
generating such diverse training sets. Prior work has shown that running DADC
over 1-3 rounds can help models fix some error types, but it does not
necessarily lead to better generalization beyond adversarial test data. We
argue that running DADC over many rounds maximizes its training-time benefits,
as the different rounds can together cover many of the task-relevant phenomena.
We present the first study of longer-term DADC, where we collect 20 rounds of
NLI examples for a small set of premise paragraphs, with both adversarial and
non-adversarial approaches. Models trained on DADC examples make 26% fewer
errors on our expert-curated test set compared to models trained on
non-adversarial data. Our analysis shows that DADC yields examples that are
more difficult, more lexically and syntactically diverse, and contain fewer
annotation artifacts compared to non-adversarial examples.

    

### [[2110.08515] Multimodal Dialogue Response Generation](http://arxiv.org/abs/2110.08515)


  Responsing with image has been recognized as an important capability for an
intelligent conversational agent. Yet existing works only focus on exploring
the multimodal dialogue models which depend on retrieval-based methods, but
neglecting generation methods. To fill in the gaps, we first present a
multimodal dialogue generation model, which takes the dialogue history as
input, then generates a textual sequence or an image as response. Learning such
a model often requires multimodal dialogues containing both texts and images
which are difficult to obtain. Motivated by the challenge in practice, we
consider multimodal dialogue generation under a natural assumption that only
limited training examples are available. In such a low-resource setting, we
devise a novel conversational agent, Divter, in order to isolate parameters
that depend on multimodal dialogues from the entire generation model. By this
means, the major part of the model can be learned from a large number of
text-only dialogues and text-image pairs respectively, then the whole
parameters can be well fitted using the limited training examples. Extensive
experiments demonstrate our method achieves state-of-the-art results in both
automatic and human evaluation, and can generate informative text and
high-resolution image responses.

    

### [[2110.08527] An Empirical Survey of the Effectiveness of Debiasing Techniques for Pre-Trained Language Models](http://arxiv.org/abs/2110.08527)


  Recent work has shown that pre-trained language models capture social biases
from the text corpora they are trained on. This has attracted attention to
developing techniques that mitigate such biases. In this work, we perform a
empirical survey of five recently proposed debiasing techniques: Counterfactual
Data Augmentation (CDA), Dropout, Iterative Nullspace Projection, Self-Debias,
and SentenceDebias. We quantify the effectiveness of each technique using three
different bias benchmarks while also measuring the impact of these techniques
on a model's language modeling ability, as well as its performance on
downstream NLU tasks. We experimentally find that: (1) CDA and Self-Debias are
the strongest of the debiasing techniques, obtaining improved scores on most of
the bias benchmarks (2) Current debiasing techniques do not generalize well
beyond gender bias; And (3) improvements on bias benchmarks such as StereoSet
and CrowS-Pairs by using debiasing strategies are usually accompanied by a
decrease in language modeling ability, making it difficult to determine whether
the bias mitigation is effective.

    

### [[2110.08529] Sharpness-Aware Minimization Improves Language Model Generalization](http://arxiv.org/abs/2110.08529)


  The allure of superhuman-level capabilities has led to considerable interest
in language models like GPT-3 and T5, wherein the research has, by and large,
revolved around new model architectures, training tasks, and loss objectives,
along with substantial engineering efforts to scale up model capacity and
dataset size. Comparatively little work has been done to improve the
generalization of these models through better optimization. In this work, we
show that Sharpness-Aware Minimization (SAM), a recently proposed optimization
procedure that encourages convergence to flatter minima, can substantially
improve the generalization of language models without much computational
overhead. We show that SAM is able to boost performance on SuperGLUE, GLUE, Web
Questions, Natural Questions, Trivia QA, and TyDiQA, with particularly large
gains when training data for these tasks is limited.

    

### [[2110.08531] A theoretical and empirical study of new adaptive algorithms with additional momentum steps and shifted updates for stochastic non-convex optimization](http://arxiv.org/abs/2110.08531)


  In the following paper we introduce new adaptive algorithms endowed with
momentum terms for stochastic non-convex optimization problems. We investigate
the almost sure convergence to stationary points, along with a finite-time
horizon analysis with respect to a chosen final iteration, and we also inspect
the worst-case iteration complexity. An estimate for the expectation of the
squared Euclidean norm of the gradient is given and the theoretical analysis
that we perform is assisted by various computational simulations for neural
network training.

    

### [[2110.08536] Sparse Distillation: Speeding Up Text Classification by Using Bigger Models](http://arxiv.org/abs/2110.08536)


  Distilling state-of-the-art transformer models into lightweight student
models is an effective way to reduce computation cost at inference time.
However, the improved inference speed may be still unsatisfactory for certain
time-sensitive applications. In this paper, we aim to further push the limit of
inference speed by exploring a new area in the design space of the student
model. More specifically, we consider distilling a transformer-based text
classifier into a billion-parameter, sparsely-activated student model with a
embedding-averaging architecture. Our experiments show that the student models
retain 97% of the RoBERTa-Large teacher performance on a collection of six text
classification tasks. Meanwhile, the student model achieves up to 600x speed-up
on both GPUs and CPUs, compared to the teacher models. Further investigation
shows that our pipeline is also effective in privacy-preserving and domain
generalization settings.

    

### [[2110.08545] A Unified Speaker Adaptation Approach for ASR](http://arxiv.org/abs/2110.08545)


  Transformer models have been used in automatic speech recognition (ASR)
successfully and yields state-of-the-art results. However, its performance is
still affected by speaker mismatch between training and test data. Further
finetuning a trained model with target speaker data is the most natural
approach for adaptation, but it takes a lot of compute and may cause
catastrophic forgetting to the existing speakers. In this work, we propose a
unified speaker adaptation approach consisting of feature adaptation and model
adaptation. For feature adaptation, we employ a speaker-aware persistent memory
model which generalizes better to unseen test speakers by making use of speaker
i-vectors to form a persistent memory. For model adaptation, we use a novel
gradual pruning method to adapt to target speakers without changing the model
architecture, which to the best of our knowledge, has never been explored in
ASR. Specifically, we gradually prune less contributing parameters on model
encoder to a certain sparsity level, and use the pruned parameters for
adaptation, while freezing the unpruned parameters to keep the original model
performance. We conduct experiments on the Librispeech dataset. Our proposed
approach brings relative 2.74-6.52% word error rate (WER) reduction on general
speaker adaptation. On target speaker adaptation, our method outperforms the
baseline with up to 20.58% relative WER reduction, and surpasses the finetuning
method by up to relative 2.54%. Besides, with extremely low-resource adaptation
data (e.g., 1 utterance), our method could improve the WER by relative 6.53%
with only a few epochs of training.

    

### [[2110.08551] HRKD: Hierarchical Relational Knowledge Distillation for Cross-domain Language Model Compression](http://arxiv.org/abs/2110.08551)


  On many natural language processing tasks, large pre-trained language models
(PLMs) have shown overwhelming performances compared with traditional neural
network methods. Nevertheless, their huge model size and low inference speed
have hindered the deployment on resource-limited devices in practice. In this
paper, we target to compress PLMs with knowledge distillation, and propose a
hierarchical relational knowledge distillation (HRKD) method to capture both
hierarchical and domain relational information. Specifically, to enhance the
model capability and transferability, we leverage the idea of meta-learning and
set up domain-relational graphs to capture the relational information across
different domains. And to dynamically select the most representative prototypes
for each domain, we propose a hierarchical compare-aggregate mechanism to
capture hierarchical relationships. Extensive experiments on public
multi-domain datasets demonstrate the superior performance of our HRKD method
as well as its strong few-shot learning ability. For reproducibility, we
release the code at this https URL.

    

### [[2110.08552] Virtual Augmentation Supported Contrastive Learning of Sentence Representations](http://arxiv.org/abs/2110.08552)


  Despite profound successes, contrastive representation learning relies on
carefully designed data augmentations using domain specific knowledge. This
challenge is magnified in natural language processing where no general rules
exist for data augmentation due to the discrete nature of natural language. We
tackle this challenge by presenting a Virtual augmentation Supported
Contrastive Learning of sentence representations (VaSCL). Originating from the
interpretation that data augmentation essentially constructs the neighborhoods
of each training instance, we in turn utilize the neighborhood to generate
effective data augmentations. Leveraging the large training batch size of
contrastive learning, we approximate the neighborhood of an instance via its
K-nearest in-batch neighbors in the representation space. We then define an
instance discrimination task within this neighborhood, and generate the virtual
augmentation in an adversarial training manner. We access the performance of
VaSCL on a wide range of downstream tasks, and set a new state-of-the-art for
unsupervised sentence representation learning.

    

### [[2110.08557] DPNAS: Neural Architecture Search for Deep Learningwith Differential Privacy](http://arxiv.org/abs/2110.08557)


  Training deep neural networks (DNNs) for meaningful differential privacy (DP)
guarantees severely degrades model utility. In this paper, we demonstrate that
the architecture of DNNs has a significant impact on model utility in the
context of private deep learning, whereas its effect is largely unexplored in
previous studies. In light of this missing, we propose the very first framework
that employs neural architecture search to automatic model design for private
deep learning, dubbed as DPNAS. To integrate private learning with architecture
search, we delicately design a novel search space and propose a DP-aware method
for training candidate models. We empirically certify the effectiveness of the
proposed framework. The searched model DPNASNet achieves state-of-the-art
privacy/utility trade-offs, e.g., for the privacy budget of $(\epsilon,
\delta)=(3, 1\times10^{-5})$, our model obtains test accuracy of $98.57\%$ on
MNIST, $88.09\%$ on FashionMNIST, and $68.33\%$ on CIFAR-10. Furthermore, by
studying the generated architectures, we provide several intriguing findings of
designing private-learning-friendly DNNs, which can shed new light on model
design for deep learning with differential privacy.

    

### [[2110.08562] BNAS v2: Learning Architectures for Binary Networks with Empirical Improvements](http://arxiv.org/abs/2110.08562)


  Backbone architectures of most binary networks are well-known floating point
(FP) architectures such as the ResNet family. Questioning that the
architectures designed for FP networks might not be the best for binary
networks, we propose to search architectures for binary networks (BNAS) by
defining a new search space for binary architectures and a novel search
objective. Specifically, based on the cell based search method, we define the
new search space of binary layer types, design a new cell template, and
rediscover the utility of and propose to use the Zeroise layer instead of using
it as a placeholder. The novel search objective diversifies early search to
learn better performing binary architectures. We show that our method searches
architectures with stable training curves despite the quantization error
inherent in binary networks. Quantitative analyses demonstrate that our
searched architectures outperform the architectures used in state-of-the-art
binary networks and outperform or perform on par with state-of-the-art binary
networks that employ various techniques other than architectural changes. In
addition, we further propose improvements to the training scheme of our
searched architectures. With the new training scheme for our searched
architectures, we achieve the state-of-the-art performance by binary networks
by outperforming all previous methods by non-trivial margins.

    

### [[2110.08565] Dynamic Graph Echo State Networks](http://arxiv.org/abs/2110.08565)


  Dynamic temporal graphs represent evolving relations between entities, e.g.
interactions between social network users or infection spreading. We propose an
extension of graph echo state networks for the efficient processing of dynamic
temporal graphs, with a sufficient condition for their echo state property, and
an experimental analysis of reservoir layout impact. Compared to temporal graph
kernels that need to hold the entire history of vertex interactions, our model
provides a vector encoding for the dynamic graph that is updated at each
time-step without requiring training. Experiments show accuracy comparable to
approximate temporal graph kernels on twelve dissemination process
classification tasks.

    

### [[2110.08577] Nys-Curve: Nyström-Approximated Curvature for Stochastic Optimization](http://arxiv.org/abs/2110.08577)


  The quasi-Newton methods generally provide curvature information by
approximating the Hessian using the secant equation. However, the secant
equation becomes insipid in approximating the Newton step owing to its use of
the first-order derivatives. In this study, we propose an approximate Newton
step-based stochastic optimization algorithm for large-scale empirical risk
minimization of convex functions with linear convergence rates. Specifically,
we compute a partial column Hessian of size ($d\times k$) with $k\ll d$
randomly selected variables, then use the \textit{Nyström method} to better
approximate the full Hessian matrix. To further reduce the computational
complexity per iteration, we directly compute the update step
($\Delta\boldsymbol{w}$) without computing and storing the full Hessian or its
inverse. Furthermore, to address large-scale scenarios in which even computing
a partial Hessian may require significant time, we used distribution-preserving
(DP) sub-sampling to compute a partial Hessian. The DP sub-sampling generates
$p$ sub-samples with similar first and second-order distribution statistics and
selects a single sub-sample at each epoch in a round-robin manner to compute
the partial Hessian. We integrate our approximated Hessian with stochastic
gradient descent and stochastic variance-reduced gradients to solve the
logistic regression problem. The numerical experiments show that the proposed
approach was able to obtain a better approximation of Newton\textquotesingle s
method with performance competitive with the state-of-the-art first-order and
the stochastic quasi-Newton methods.

    

### [[2110.08583] ASR4REAL: An extended benchmark for speech models](http://arxiv.org/abs/2110.08583)


  Popular ASR benchmarks such as Librispeech and Switchboard are limited in the
diversity of settings and speakers they represent. We introduce a set of
benchmarks matching real-life conditions, aimed at spotting possible biases and
weaknesses in models. We have found out that even though recent models do not
seem to exhibit a gender bias, they usually show important performance
discrepancies by accent, and even more important ones depending on the
socio-economic status of the speakers. Finally, all tested models show a strong
performance drop when tested on conversational speech, and in this precise
context even a language model trained on a dataset as big as Common Crawl does
not seem to have significant positive effect which reiterates the importance of
developing conversational language models

    

### [[2110.08586] Generative Adversarial Imitation Learning for End-to-End Autonomous Driving on Urban Environments](http://arxiv.org/abs/2110.08586)


  Autonomous driving is a complex task, which has been tackled since the first
self-driving car ALVINN in 1989, with a supervised learning approach, or
behavioral cloning (BC). In BC, a neural network is trained with state-action
pairs that constitute the training set made by an expert, i.e., a human driver.
However, this type of imitation learning does not take into account the
temporal dependencies that might exist between actions taken in different
moments of a navigation trajectory. These type of tasks are better handled by
reinforcement learning (RL) algorithms, which need to define a reward function.
On the other hand, more recent approaches to imitation learning, such as
Generative Adversarial Imitation Learning (GAIL), can train policies without
explicitly requiring to define a reward function, allowing an agent to learn by
trial and error directly on a training set of expert trajectories. In this
work, we propose two variations of GAIL for autonomous navigation of a vehicle
in the realistic CARLA simulation environment for urban scenarios. Both of them
use the same network architecture, which process high dimensional image input
from three frontal cameras, and other nine continuous inputs representing the
velocity, the next point from the sparse trajectory and a high-level driving
command. We show that both of them are capable of imitating the expert
trajectory from start to end after training ends, but the GAIL loss function
that is augmented with BC outperforms the former in terms of convergence time
and training stability.

    

### [[2110.08598] A Variational Bayesian Approach to Learning Latent Variables for Acoustic Knowledge Transfer](http://arxiv.org/abs/2110.08598)


  We propose a variational Bayesian (VB) approach to learning distributions of
latent variables in deep neural network (DNN) models for cross-domain knowledge
transfer, to address acoustic mismatches between training and testing
conditions. Instead of carrying out point estimation in conventional maximum a
posteriori estimation with a risk of having a curse of dimensionality in
estimating a huge number of model parameters, we focus our attention on
estimating a manageable number of latent variables of DNNs via a VB inference
framework. To accomplish model transfer, knowledge learnt from a source domain
is encoded in prior distributions of latent variables and optimally combined,
in a Bayesian sense, with a small set of adaptation data from a target domain
to approximate the corresponding posterior distributions. Experimental results
on device adaptation in acoustic scene classification show that our proposed VB
approach can obtain good improvements on target devices, and consistently
outperforms 13 state-of-the-art knowledge transfer algorithms.

    

### [[2110.08599] Mapping illegal waste dumping sites with neural-network classification of satellite imagery](http://arxiv.org/abs/2110.08599)


  Public health and habitat quality are crucial goals of urban planning. In
recent years, the severe social and environmental impact of illegal waste
dumping sites has made them one of the most serious problems faced by cities in
the Global South, in a context of scarce information available for decision
making. To help identify the location of dumping sites and track their
evolution over time we adopt a data-driven model from the machine learning
domain, analyzing satellite images. This allows us to take advantage of the
increasing availability of geo-spatial open-data, high-resolution satellite
imagery, and open source tools to train machine learning algorithms with a
small set of known waste dumping sites in Buenos Aires, and then predict the
location of other sites over vast areas at high speed and low cost. This case
study shows the results of a collaboration between Dymaxion Labs and
Fundación Bunge y Born to harness this technique in order to create a
comprehensive map of potential locations of illegal waste dumping sites in the
region.

    

### [[2110.08607] Physics-guided Deep Markov Models for Learning Nonlinear Dynamical Systems with Uncertainty](http://arxiv.org/abs/2110.08607)


  In this paper, we propose a probabilistic physics-guided framework, termed
Physics-guided Deep Markov Model (PgDMM). The framework is especially targeted
to the inference of the characteristics and latent structure of nonlinear
dynamical systems from measurement data, where it is typically intractable to
perform exact inference of latent variables. A recently surfaced option
pertains to leveraging variational inference to perform approximate inference.
In such a scheme, transition and emission functions of the system are
parameterized via feed-forward neural networks (deep generative models).
However, due to the generalized and highly versatile formulation of neural
network functions, the learned latent space is often prone to lack physical
interpretation and structured representation. To address this, we bridge
physics-based state space models with Deep Markov Models, thus delivering a
hybrid modeling framework for unsupervised learning and identification for
nonlinear dynamical systems. Specifically, the transition process can be
modeled as a physics-based model enhanced with an additive neural network
component, which aims to learn the discrepancy between the physics-based model
and the actual dynamical system being monitored. The proposed framework takes
advantage of the expressive power of deep learning, while retaining the driving
physics of the dynamical system by imposing physics-driven restrictions on the
side of the latent space. We demonstrate the benefits of such a fusion in terms
of achieving improved performance on illustrative simulation examples and
experimental case studies of nonlinear systems. Our results indicate that the
physics-based models involved in the employed transition and emission functions
essentially enforce a more structured and physically interpretable latent
space, which is essential to generalization and prediction capabilities.

    

### [[2110.08610] MAAD: A Model and Dataset for "Attended Awareness" in Driving](http://arxiv.org/abs/2110.08610)


  We propose a computational model to estimate a person's attended awareness of
their environment. We define attended awareness to be those parts of a
potentially dynamic scene which a person has attended to in recent history and
which they are still likely to be physically aware of. Our model takes as input
scene information in the form of a video and noisy gaze estimates, and outputs
visual saliency, a refined gaze estimate, and an estimate of the person's
attended awareness. In order to test our model, we capture a new dataset with a
high-precision gaze tracker including 24.5 hours of gaze sequences from 23
subjects attending to videos of driving scenes. The dataset also contains
third-party annotations of the subjects' attended awareness based on
observations of their scan path. Our results show that our model is able to
reasonably estimate attended awareness in a controlled setting, and in the
future could potentially be extended to real egocentric driving data to help
enable more effective ahead-of-time warnings in safety systems and thereby
augment driver performance. We also demonstrate our model's effectiveness on
the tasks of saliency, gaze calibration, and denoising, using both our dataset
and an existing saliency dataset. We make our model and dataset available at
this https URL.

    

### [[2110.08611] Deep Active Learning by Leveraging Training Dynamics](http://arxiv.org/abs/2110.08611)


  Active learning theories and methods have been extensively studied in
classical statistical learning settings. However, deep active learning, i.e.,
active learning with deep learning models, is usually based on empirical
criteria without solid theoretical justification, thus suffering from heavy
doubts when some of those fail to provide benefits in applications. In this
paper, by exploring the connection between the generalization performance and
the training dynamics, we propose a theory-driven deep active learning method
(dynamicAL) which selects samples to maximize training dynamics. In particular,
we prove that convergence speed of training and the generalization performance
is positively correlated under the ultra-wide condition and show that
maximizing the training dynamics leads to a better generalization performance.
Further on, to scale up to large deep neural networks and data sets, we
introduce two relaxations for the subset selection problem and reduce the time
complexity from polynomial to constant. Empirical results show that dynamicAL
not only outperforms the other baselines consistently but also scales well on
large deep learning models. We hope our work inspires more attempts in bridging
the theoretical findings of deep networks and practical impacts in deep active
learning applications.

    

### [[2110.08614] Deep Learning and Spectral Embedding for Graph Partitioning](http://arxiv.org/abs/2110.08614)


  We present a graph bisection and partitioning algorithm based on graph neural
networks. For each node in the graph, the network outputs probabilities for
each of the partitions. The graph neural network consists of two modules: an
embedding phase and a partitioning phase. The embedding phase is trained first
by minimizing a loss function inspired by spectral graph theory. The
partitioning module is trained through a loss function that corresponds to the
expected value of the normalized cut. Both parts of the neural network rely on
SAGE convolutional layers and graph coarsening using heavy edge matching. The
multilevel structure of the neural network is inspired by the multigrid
algorithm. Our approach generalizes very well to bigger graphs and has
partition quality comparable to METIS, Scotch and spectral partitioning, with
shorter runtime compared to METIS and spectral partitioning.

    

### [[2110.08616] GradSign: Model Performance Inference with Theoretical Insights](http://arxiv.org/abs/2110.08616)


  A key challenge in neural architecture search (NAS) is quickly inferring the
predictive performance of a broad spectrum of networks to discover
statistically accurate and computationally efficient ones. We refer to this
task as model performance inference (MPI). The current practice for efficient
MPI is gradient-based methods that leverage the gradients of a network at
initialization to infer its performance. However, existing gradient-based
methods rely only on heuristic metrics and lack the necessary theoretical
foundations to consolidate their designs. We propose GradSign, an accurate,
simple, and flexible metric for model performance inference with theoretical
insights. The key idea behind GradSign is a quantity {\Psi} to analyze the
optimization landscape of different networks at the granularity of individual
training samples. Theoretically, we show that both the network's training and
true population losses are proportionally upper-bounded by {\Psi} under
reasonable assumptions. In addition, we design GradSign, an accurate and simple
approximation of {\Psi} using the gradients of a network evaluated at a random
initialization state. Evaluation on seven NAS benchmarks across three training
datasets shows that GradSign generalizes well to real-world networks and
consistently outperforms state-of-the-art gradient-based methods for MPI
evaluated by Spearman's {\rho} and Kendall's Tau. Additionally, we integrate
GradSign into four existing NAS algorithms and show that the GradSign-assisted
NAS algorithms outperform their vanilla counterparts by improving the
accuracies of best-discovered networks by up to 0.3%, 1.1%, and 1.0% on three
real-world tasks.

    

### [[2110.08618] Convolutional Deep Denoising Autoencoders for Radio Astronomical Images](http://arxiv.org/abs/2110.08618)


  We apply a Machine Learning technique known as Convolutional Denoising
Autoencoder to denoise synthetic images of state-of-the-art radio telescopes,
with the goal of detecting the faint, diffused radio sources predicted to
characterise the radio cosmic web. In our application, denoising is intended to
address both the reduction of random instrumental noise and the minimisation of
additional spurious artefacts like the sidelobes, resulting from the aperture
synthesis technique. The effectiveness and the accuracy of the method are
analysed for different kinds of corrupted input images, together with its
computational performance. Specific attention has been devoted to create
realistic mock observations for the training, exploiting the outcomes of
cosmological numerical simulations, to generate images corresponding to LOFAR
HBA 8 hours observations at 150 MHz. Our autoencoder can effectively denoise
complex images identifying and extracting faint objects at the limits of the
instrumental sensitivity. The method can efficiently scale on large datasets,
exploiting high performance computing solutions, in a fully automated way (i.e.
no human supervision is required after training). It can accurately perform
image segmentation, identifying low brightness outskirts of diffused sources,
proving to be a viable solution for detecting challenging extended objects
hidden in noisy radio observations.

    

### [[2110.08626] Learning velocity model for complex media with deep convolutional neural networks](http://arxiv.org/abs/2110.08626)


  The paper considers the problem of velocity model acquisition for a complex
media based on boundary measurements. The acoustic model is used to describe
the media. We used an open-source dataset of velocity distributions to compare
the presented results with the previous works directly. Forward modeling is
performed using the grid-characteristic numerical method. The inverse problem
is solved using deep convolutional neural networks. Modifications for a
baseline UNet architecture are proposed to improve both structural similarity
index measure quantitative correspondence of the velocity profiles with the
ground truth. We evaluate our enhancements and demonstrate the statistical
significance of the results.

    

### [[2110.08627] On the Pareto Frontier of Regret Minimization and Best Arm Identification in Stochastic Bandits](http://arxiv.org/abs/2110.08627)


  We study the Pareto frontier of two archetypal objectives in stochastic
bandits, namely, regret minimization (RM) and best arm identification (BAI)
with a fixed horizon. It is folklore that the balance between exploitation and
exploration is crucial for both RM and BAI, but exploration is more critical in
achieving the optimal performance for the latter objective. To make this
precise, we first design and analyze the BoBW-lil'UCB$({\gamma})$ algorithm,
which achieves order-wise optimal performance for RM or BAI under different
values of ${\gamma}$. Complementarily, we show that no algorithm can
simultaneously perform optimally for both the RM and BAI objectives. More
precisely, we establish non-trivial lower bounds on the regret achievable by
any algorithm with a given BAI failure probability. This analysis shows that in
some regimes BoBW-lil'UCB$({\gamma})$ achieves Pareto-optimality up to constant
or small terms. Numerical experiments further demonstrate that when applied to
difficult instances, BoBW-lil'UCB outperforms a close competitor UCB$_{\alpha}$
(Degenne et al., 2019), which is designed for RM and BAI with a fixed
confidence.

    

### [[2110.08633] Hydra: A System for Large Multi-Model Deep Learning](http://arxiv.org/abs/2110.08633)


  Training deep learning (DL) models that do not fit into the memory of a
single GPU is a vexed process, forcing users to procure multiple GPUs to adopt
model-parallel execution. Unfortunately, sequential dependencies in neural
architectures often block efficient multi-device training, leading to
suboptimal performance. We present 'model spilling', a technique aimed at
models such as Transformers and CNNs to move groups of layers, or shards,
between DRAM and GPU memory, thus enabling arbitrarily large models to be
trained even on just one GPU. We then present a set of novel techniques
leveraging spilling to raise efficiency for multi-model training workloads such
as model selection: a new hybrid of task- and model-parallelism, a new shard
scheduling heuristic, and 'double buffering' to hide latency. We prototype our
ideas into a system we call HYDRA to support seamless single-model and
multi-model training of large DL models. Experiments with real benchmark
workloads show that HYDRA is over 7x faster than regular model parallelism and
over 50% faster than state-of-the-art industrial tools for pipeline
parallelism.

    

### [[2110.08634] Towards Robust Waveform-Based Acoustic Models](http://arxiv.org/abs/2110.08634)


  We propose an approach for learning robust acoustic models in adverse
environments, characterized by a significant mismatch between training and test
conditions. This problem is of paramount importance for the deployment of
speech recognition systems that need to perform well in unseen environments.
Our approach is an instance of vicinal risk minimization, which aims to improve
risk estimates during training by replacing the delta functions that define the
empirical density over the input space with an approximation of the marginal
population density in the vicinity of the training samples. More specifically,
we assume that local neighborhoods centered at training samples can be
approximated using a mixture of Gaussians, and demonstrate theoretically that
this can incorporate robust inductive bias into the learning process. We
characterize the individual mixture components implicitly via data augmentation
schemes, designed to address common sources of spurious correlations in
acoustic models. To avoid potential confounding effects on robustness due to
information loss, which has been associated with standard feature extraction
techniques (e.g., FBANK and MFCC features), we focus our evaluation on the
waveform-based setting. Our empirical results show that the proposed approach
can generalize to unseen noise conditions, with 150% relative improvement in
out-of-distribution generalization compared to training using the standard risk
minimization principle. Moreover, the results demonstrate competitive
performance relative to models learned using a training sample designed to
match the acoustic conditions characteristic of test utterances (i.e., optimal
vicinal densities).

    

### [[2110.08642] Local Advantage Actor-Critic for Robust Multi-Agent Deep Reinforcement Learning](http://arxiv.org/abs/2110.08642)


  Policy gradient methods have become popular in multi-agent reinforcement
learning, but they suffer from high variance due to the presence of
environmental stochasticity and exploring agents (i.e., non-stationarity),
which is potentially worsened by the difficulty in credit assignment. As a
result, there is a need for a method that is not only capable of efficiently
solving the above two problems but also robust enough to solve a variety of
tasks. To this end, we propose a new multi-agent policy gradient method, called
Robust Local Advantage (ROLA) Actor-Critic. ROLA allows each agent to learn an
individual action-value function as a local critic as well as ameliorating
environment non-stationarity via a novel centralized training approach based on
a centralized critic. By using this local critic, each agent calculates a
baseline to reduce variance on its policy gradient estimation, which results in
an expected advantage action-value over other agents' choices that implicitly
improves credit assignment. We evaluate ROLA across diverse benchmarks and show
its robustness and effectiveness over a number of state-of-the-art multi-agent
policy gradient algorithms.

    

### [[2110.08649] Equivariant Discrete Normalizing Flows](http://arxiv.org/abs/2110.08649)


  At its core, generative modeling seeks to uncover the underlying factors that
give rise to observed data that can often be modelled as the natural symmetries
that manifest themselves through invariances and equivariances to certain
transformations laws. However, current approaches are couched in the formalism
of continuous normalizing flows that require the construction of equivariant
vector fields -- inhibiting their simple application to conventional higher
dimensional generative modelling domains like natural images. In this paper we
focus on building equivariant normalizing flows using discrete layers. We first
theoretically prove the existence of an equivariant map for compact groups
whose actions are on compact spaces. We further introduce two new equivariant
flows: $G$-coupling Flows and $G$-Residual Flows that elevate classical
Coupling and Residual Flows with equivariant maps to a prescribed group $G$.
Our construction of $G$-Residual Flows are also universal, in the sense that we
prove an $G$-equivariant diffeomorphism can be exactly mapped by a $G$-residual
flow. Finally, we complement our theoretical insights with experiments -- for
the first time -- on image datasets like CIFAR-10 and show $G$-Equivariant
Discrete Normalizing flows lead to increased data efficiency, faster
convergence, and improved likelihood estimates.

    

### [[2110.08668] Fast Strain Estimation and Frame Selection in Ultrasound Elastography using Machine Learning](http://arxiv.org/abs/2110.08668)


  Ultrasound Elastography aims to determine the mechanical properties of the
tissue by monitoring tissue deformation due to internal or external forces.
Tissue deformations are estimated from ultrasound radio frequency (RF) signals
and are often referred to as time delay estimation (TDE). Given two RF frames
I1 and I2, we can compute a displacement image which shows the change in the
position of each sample in I1 to a new position in I2. Two important challenges
in TDE include high computational complexity and the difficulty in choosing
suitable RF frames. Selecting suitable frames is of high importance because
many pairs of RF frames either do not have acceptable deformation for
extracting informative strain images or are decorrelated and deformation cannot
be reliably estimated. Herein, we introduce a method that learns 12
displacement modes in quasi-static elastography by performing Principal
Component Analysis (PCA) on displacement fields of a large training database.
In the inference stage, we use dynamic programming (DP) to compute an initial
displacement estimate of around 1% of the samples, and then decompose this
sparse displacement into a linear combination of the 12 displacement modes. Our
method assumes that the displacement of the whole image could also be described
by this linear combination of principal components. We then use the GLobal
Ultrasound Elastography (GLUE) method to fine-tune the result yielding the
exact displacement image. Our method, which we call PCA-GLUE, is more than 10
times faster than DP in calculating the initial displacement map while giving
the same result. Our second contribution in this paper is determining the
suitability of the frame pair I1 and I2 for strain estimation, which we achieve
by using the weight vector that we calculated for PCA-GLUE as an input to a
multi-layer perceptron (MLP) classifier.

    

### [[2110.08676] Noise-Augmented Privacy-Preserving Empirical Risk Minimization with Dual-purpose Regularizer and Privacy Budget Retrieval and Recycling](http://arxiv.org/abs/2110.08676)


  We propose Noise-Augmented Privacy-Preserving Empirical Risk Minimization
(NAPP-ERM) that solves ERM with differential privacy guarantees. Existing
privacy-preserving ERM approaches may be subject to over-regularization with
the employment of an l2 term to achieve strong convexity on top of the target
regularization. NAPP-ERM improves over the current approaches and mitigates
over-regularization by iteratively realizing target regularization through
appropriately designed augmented data and delivering strong convexity via a
single adaptively weighted dual-purpose l2 regularizer. When the target
regularization is for variable selection, we propose a new regularizer that
achieves both privacy and sparsity guarantees simultaneously. Finally, we
propose a strategy to retrieve privacy budget when the strong convexity
requirement is met, which can be returned to users such that the DP of ERM is
guaranteed at a lower privacy cost than originally planned, or be recycled to
the ERM optimization procedure to reduce the injected DP noise and improve the
utility of DP-ERM. From an implementation perspective, NAPP-ERM can be achieved
by optimizing a non-perturbed object function given noise-augmented data and
can thus leverage existing tools for non-private ERM optimization. We
illustrate through extensive experiments the mitigation effect of the
over-regularization and private budget retrieval by NAPP-ERM on variable
selection and prediction.

    

### [[1712.01145] Learning Fast and Slow: PROPEDEUTICA for Real-time Malware Detection](http://arxiv.org/abs/1712.01145)


  Existing malware detectors on safety-critical devices have difficulties in
runtime detection due to the performance overhead. In this paper, we introduce
PROPEDEUTICA, a framework for efficient and effective real-time malware
detection, leveraging the best of conventional machine learning (ML) and deep
learning (DL) techniques. In PROPEDEUTICA, all software start execution are
considered as benign and monitored by a conventional ML classifier for fast
detection. If the software receives a borderline classification from the ML
detector (e.g. the software is 50% likely to be benign and 50% likely to be
malicious), the software will be transferred to a more accurate, yet
performance demanding DL detector. To address spatial-temporal dynamics and
software execution heterogeneity, we introduce a novel DL architecture
(DEEPMALWARE) for PROPEDEUTICA with multi-stream inputs. We evaluated
PROPEDEUTICA with 9,115 malware samples and 1,338 benign software from various
categories for the Windows OS. With a borderline interval of [30%-70%],
PROPEDEUTICA achieves an accuracy of 94.34% and a false-positive rate of 8.75%,
with 41.45% of the samples moved for DEEPMALWARE analysis. Even using only CPU,
PROPEDEUTICA can detect malware within less than 0.1 seconds.

    

### [[1901.08057] Large dimensional analysis of general margin based classification methods](http://arxiv.org/abs/1901.08057)


  Margin-based classifiers have been popular in both machine learning and
statistics for classification problems. Since a large number of classifiers are
available, one natural question is which type of classifiers should be used
given a particular classification task. We answer this question by
investigating the asymptotic performance of a family of large-margin
classifiers under the two component mixture models in situations where the data
dimension $p$ and the sample $n$ are both large. This family covers a broad
range of classifiers including support vector machine, distance weighted
discrimination, penalized logistic regression, and large-margin unified machine
as special cases. The asymptotic results are described by a set of nonlinear
equations and we observe a close match of them with Monte Carlo simulation on
finite data samples. Our analytical studies shed new light on how to select the
best classifier among various classification methods as well as on how to
choose the optimal tuning parameters for a given method.

    

### [[1902.04742] Uniform convergence may be unable to explain generalization in deep learning](http://arxiv.org/abs/1902.04742)


  Aimed at explaining the surprisingly good generalization behavior of
overparameterized deep networks, recent works have developed a variety of
generalization bounds for deep learning, all based on the fundamental
learning-theoretic technique of uniform convergence. While it is well-known
that many of these existing bounds are numerically large, through numerous
experiments, we bring to light a more concerning aspect of these bounds: in
practice, these bounds can {\em increase} with the training dataset size.
Guided by our observations, we then present examples of overparameterized
linear classifiers and neural networks trained by gradient descent (GD) where
uniform convergence provably cannot "explain generalization" -- even if we take
into account the implicit bias of GD {\em to the fullest extent possible}. More
precisely, even if we consider only the set of classifiers output by GD, which
have test errors less than some small $\epsilon$ in our settings, we show that
applying (two-sided) uniform convergence on this set of classifiers will yield
only a vacuous generalization guarantee larger than $1-\epsilon$. Through these
findings, we cast doubt on the power of uniform convergence-based
generalization bounds to provide a complete picture of why overparameterized
deep networks generalize well.

    

### [[1908.09128] Position-Aware Self-Attention based Neural Sequence Labeling](http://arxiv.org/abs/1908.09128)


  Sequence labeling is a fundamental task in natural language processing and
has been widely studied. Recently, RNN-based sequence labeling models have
increasingly gained attentions. Despite superior performance achieved by
learning the long short-term (i.e., successive) dependencies, the way of
sequentially processing inputs might limit the ability to capture the
non-continuous relations over tokens within a sentence. To tackle the problem,
we focus on how to effectively model successive and discrete dependencies of
each token for enhancing the sequence labeling performance. Specifically, we
propose an innovative attention-based model (called position-aware
selfattention, i.e., PSA) as well as a well-designed self-attentional context
fusion layer within a neural network architecture, to explore the positional
information of an input sequence for capturing the latent relations among
tokens. Extensive experiments on three classical tasks in sequence labeling
domain, i.e., partof-speech (POS) tagging, named entity recognition (NER) and
phrase chunking, demonstrate our proposed model outperforms the
state-of-the-arts without any external knowledge, in terms of various metrics.

    

### [[1910.09714] Smoothness-Adaptive Contextual Bandits](http://arxiv.org/abs/1910.09714)


  We study a non-parametric multi-armed bandit problem with stochastic
covariates, where a key complexity driver is the smoothness of payoff functions
with respect to covariates. Previous studies have focused on deriving
minimax-optimal algorithms in cases where it is a priori known how smooth the
payoff functions are. In practice, however, the smoothness of payoff functions
is typically not known in advance, and misspecification of smoothness may
severely deteriorate the performance of existing methods. In this work, we
consider a framework where the smoothness of payoff functions is not known, and
study when and how algorithms may adapt to unknown smoothness. First, we
establish that designing algorithms that adapt to unknown smoothness of payoff
functions is, in general, impossible. However, under a self-similarity
condition (which does not reduce the minimax complexity of the dynamic
optimization problem at hand), we establish that adapting to unknown smoothness
is possible, and further devise a general policy for achieving
smoothness-adaptive performance. Our policy infers the smoothness of payoffs
throughout the decision-making process, while leveraging the structure of
off-the-shelf non-adaptive policies. We establish that for problem settings
with either differentiable or non-differentiable payoff functions, this policy
matches (up to a logarithmic scale) the regret rate that is achievable when the
smoothness of payoffs is known a priori.

    

### [[1911.02319] Improving reinforcement learning algorithms: towards optimal learning rate policies](http://arxiv.org/abs/1911.02319)


  This paper investigates to what extent one can improve reinforcement learning
algorithms. Our study is split in three parts. First, our analysis shows that
the classical asymptotic convergence rate $O(1/\sqrt{N})$ is pessimistic and
can be replaced by $O((\log(N)/N)^{\beta})$ with $\frac{1}{2}\leq \beta \leq 1$
and $N$ the number of iterations. Second, we propose a dynamic optimal policy
for the choice of the learning rate $(\gamma_k)_{k\geq 0}$ used in stochastic
approximation (SA). We decompose our policy into two interacting levels: the
inner and the outer level. In the inner level, we present the
\nameref{Alg:v_4_s} algorithm (for "PAst Sign Search") which, based on a
predefined sequence $(\gamma^o_k)_{k\geq 0}$, constructs a new sequence
$(\gamma^i_k)_{k\geq 0}$ whose error decreases faster. In the outer level, we
propose an optimal methodology for the selection of the predefined sequence
$(\gamma^o_k)_{k\geq 0}$. Third, we show empirically that our selection
methodology of the learning rate outperforms significantly standard algorithms
used in reinforcement learning (RL) in the three following applications: the
estimation of a drift, the optimal placement of limit orders and the optimal
execution of large number of shares.

    

### [[2001.04286] Nonparametric Continuous Sensor Registration](http://arxiv.org/abs/2001.04286)


  This paper develops a new mathematical framework that enables nonparametric
joint semantic and geometric representation of continuous functions using data.
The joint embedding is modeled by representing the processes in a reproducing
kernel Hilbert space. The functions can be defined on arbitrary smooth
manifolds where the action of a Lie group aligns them. The continuous functions
allow the registration to be independent of a specific signal resolution. The
framework is fully analytical with a closed-form derivation of the Riemannian
gradient and Hessian. We study a more specialized but widely used case where
the Lie group acts on functions isometrically. We solve the problem by
maximizing the inner product between two functions defined over data, while the
continuous action of the rigid body motion Lie group is captured through the
integration of the flow in the corresponding Lie algebra. Low-dimensional cases
are derived with numerical examples to show the generality of the proposed
framework. The high-dimensional derivation for the special Euclidean group
acting on the Euclidean space showcases the point cloud registration and
bird's-eye view map registration abilities. An implementation of this framework
for RGB-D cameras outperforms the state-of-the-art robust visual odometry and
performs well in texture and structure-scarce environments.

    

### [[2003.00660] Upper Confidence Primal-Dual Reinforcement Learning for CMDP with Adversarial Loss](http://arxiv.org/abs/2003.00660)


  We consider online learning for episodic stochastically constrained Markov
decision processes (CMDPs), which plays a central role in ensuring the safety
of reinforcement learning. Here the loss function can vary arbitrarily across
the episodes, and both the loss received and the budget consumption are
revealed at the end of each episode. Previous works solve this problem under
the restrictive assumption that the transition model of the Markov decision
processes (MDPs) is known a priori and establish regret bounds that depend
polynomially on the cardinalities of the state space $\mathcal{S}$ and the
action space $\mathcal{A}$. In this work, we propose a new \emph{upper
confidence primal-dual} algorithm, which only requires the trajectories sampled
from the transition model. In particular, we prove that the proposed algorithm
achieves $\widetilde{\mathcal{O}}(L|\mathcal{S}|\sqrt{|\mathcal{A}|T})$ upper
bounds of both the regret and the constraint violation, where $L$ is the length
of each episode. Our analysis incorporates a new high-probability drift
analysis of Lagrange multiplier processes into the celebrated regret analysis
of upper confidence reinforcement learning, which demonstrates the power of
"optimism in the face of uncertainty" in constrained online learning.

    

### [[2003.06566] On the benefits of defining vicinal distributions in latent space](http://arxiv.org/abs/2003.06566)


  The vicinal risk minimization (VRM) principle is an empirical risk
minimization (ERM) variant that replaces Dirac masses with vicinal functions.
There is strong numerical and theoretical evidence showing that VRM outperforms
ERM in terms of generalization if appropriate vicinal functions are chosen.
Mixup Training (MT), a popular choice of vicinal distribution, improves the
generalization performance of models by introducing globally linear behavior in
between training examples. Apart from generalization, recent works have shown
that mixup trained models are relatively robust to input
perturbations/corruptions and at the same time are calibrated better than their
non-mixup counterparts. In this work, we investigate the benefits of defining
these vicinal distributions like mixup in latent space of generative models
rather than in input space itself. We propose a new approach - \textit{VarMixup
(Variational Mixup)} - to better sample mixup images by using the latent
manifold underlying the data. Our empirical studies on CIFAR-10, CIFAR-100, and
Tiny-ImageNet demonstrate that models trained by performing mixup in the latent
manifold learned by VAEs are inherently more robust to various input
corruptions/perturbations, are significantly better calibrated, and exhibit
more local-linear loss landscapes.

    

### [[2003.12739] Modulating Bottom-Up and Top-Down Visual Processing via Language-Conditional Filters](http://arxiv.org/abs/2003.12739)


  How to best integrate linguistic and perceptual processing in multi-modal
tasks that involve language and vision is an important open problem. In this
work, we argue that the common practice of using language in a top-down manner,
to direct visual attention over high-level visual features, may not be optimal.
We hypothesize that the use of language to also condition the bottom-up
processing from pixels to high-level features can provide benefits to the
overall performance. To support our claim, we propose a model for
language-vision problems involving dense prediction, and perform experiments on
two different multi-modal tasks: image segmentation from referring expressions
and language-guided image colorization. We compare results where either one or
both of the top-down and bottom-up visual branches are conditioned on language.
Our experiments reveal that using language to control the filters for bottom-up
visual processing in addition to top-down attention leads to better results on
both tasks and achieves state-of-the-art performance. Our analysis of different
word types in input expressions suggest that the bottom-up conditioning is
especially helpful in the presence of low level visual concepts like color.

    

### [[2004.08646] Macro-Action-Based Deep Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2004.08646)


  In real-world multi-robot systems, performing high-quality, collaborative
behaviors requires robots to asynchronously reason about high-level action
selection at varying time durations. Macro-Action Decentralized Partially
Observable Markov Decision Processes (MacDec-POMDPs) provide a general
framework for asynchronous decision making under uncertainty in fully
cooperative multi-agent tasks. However, multi-agent deep reinforcement learning
methods have only been developed for (synchronous) primitive-action problems.
This paper proposes two Deep Q-Network (DQN) based methods for learning
decentralized and centralized macro-action-value functions with novel
macro-action trajectory replay buffers introduced for each case. Evaluations on
benchmark problems and a larger domain demonstrate the advantage of learning
with macro-actions over primitive-actions and the scalability of our
approaches.

    

### [[2005.03566] Noisy Differentiable Architecture Search](http://arxiv.org/abs/2005.03566)


  Simplicity is the ultimate sophistication. Differentiable Architecture Search
(DARTS) has now become one of the mainstream paradigms of neural architecture
search. However, it largely suffers from the well-known performance collapse
issue due to the aggregation of skip connections. It is thought to have overly
benefited from the residual structure which accelerates the information flow.
To weaken this impact, we propose to inject unbiased random noise to impede the
flow. We name this novel approach NoisyDARTS. In effect, a network optimizer
should perceive this difficulty at each training step and refrain from
overshooting, especially on skip connections. In the long run, since we add no
bias to the gradient in terms of expectation, it is still likely to converge to
the right solution area. We also prove that the injected noise plays a role in
smoothing the loss landscape, which makes the optimization easier. Our method
features extreme simplicity and acts as a new strong baseline. We perform
extensive experiments across various search spaces, datasets, and tasks, where
we robustly achieve state-of-the-art results. Our code is available at
this https URL.

    

### [[2006.05842] The Emergence of Individuality](http://arxiv.org/abs/2006.05842)


  Individuality is essential in human society, which induces the division of
labor and thus improves the efficiency and productivity. Similarly, it should
also be the key to multi-agent cooperation. Inspired by that individuality is
of being an individual separate from others, we propose a simple yet efficient
method for the emergence of individuality (EOI) in multi-agent reinforcement
learning (MARL). EOI learns a probabilistic classifier that predicts a
probability distribution over agents given their observation and gives each
agent an intrinsic reward of being correctly predicted by the classifier. The
intrinsic reward encourages the agents to visit their own familiar
observations, and learning the classifier by such observations makes the
intrinsic reward signals stronger and the agents more identifiable. To further
enhance the intrinsic reward and promote the emergence of individuality, two
regularizers are proposed to increase the discriminability of the classifier.
We implement EOI on top of popular MARL algorithms. Empirically, we show that
EOI significantly outperforms existing methods in a variety of multi-agent
cooperative scenarios.

    

### [[2007.00823] Dropout as a Regularizer of Interaction Effects](http://arxiv.org/abs/2007.00823)


  We examine Dropout through the perspective of interactions. This view
provides a symmetry to explain Dropout: given $N$ variables, there are ${N
\choose k}$ possible sets of $k$ variables to form an interaction (i.e.
$\mathcal{O}(N^k)$); conversely, the probability an interaction of $k$
variables survives Dropout at rate $p$ is $(1-p)^k$ (decaying with $k$). These
rates effectively cancel, and so Dropout regularizes against higher-order
interactions. We prove this perspective analytically and empirically. This
perspective of Dropout as a regularizer against interaction effects has several
practical implications: (1) higher Dropout rates should be used when we need
stronger regularization against spurious high-order interactions, (2) caution
should be exercised when interpreting Dropout-based explanations and
uncertainty measures, and (3) networks trained with Input Dropout are biased
estimators. We also compare Dropout to other regularizers and find that it is
difficult to obtain the same selective pressure against high-order
interactions.

    

### [[2007.02794] Efficient Connected and Automated Driving Systemwith Multi-agent Graph Reinforcement Learning](http://arxiv.org/abs/2007.02794)


  Connected and automated vehicles (CAVs) have attracted more and more
attention recently. The fast actuation time allows them having the potential to
promote the efficiency and safety of the whole transportation system. Due to
technical challenges, there will be a proportion of vehicles that can be
equipped with automation while other vehicles are without automation. Instead
of learning a reliable behavior for ego automated vehicle, we focus on how to
improve the outcomes of the total transportation system by allowing each
automated vehicle to learn cooperation with each other and regulate
human-driven traffic flow. One of state of the art method is using
reinforcement learning to learn intelligent decision making policy. However,
direct reinforcement learning framework cannot improve the performance of the
whole system. In this article, we demonstrate that considering the problem in
multi-agent setting with shared policy can help achieve better system
performance than non-shared policy in single-agent setting. Furthermore, we
find that utilization of attention mechanism on interaction features can
capture the interplay between each agent in order to boost cooperation. To the
best of our knowledge, while previous automated driving studies mainly focus on
enhancing individual's driving performance, this work serves as a starting
point for research on system-level multi-agent cooperation performance using
graph information sharing. We conduct extensive experiments in car-following
and unsignalized intersection settings. The results demonstrate that CAVs
controlled by our method can achieve the best performance against several state
of the art baselines.

    

### [[2007.03408] A Generative Model for Texture Synthesis based on Optimal Transport between Feature Distributions](http://arxiv.org/abs/2007.03408)


  We propose GOTEX, a general framework for texture synthesis by optimization
that constrains the statistical distribution of local features. While our model
encompasses several existing texture models, we focus on the case where the
comparison between feature distributions relies on optimal transport distances.
We show that the semi-dual formulation of optimal transport allows to control
the distribution of various possible features, even if these features live in a
high-dimensional space. We then study the resulting minimax optimization
problem, which corresponds to a Wasserstein generative model, for which the
inner concave maximization problem can be solved with standard stochastic
gradient methods. The alternate optimization algorithm is shown to be versatile
in terms of applications, features and architecture; in particular it allows to
produce high-quality synthesized textures with different sets of features. We
analyze the results obtained by constraining the distribution of patches or the
distribution of responses to a pre-learned VGG neural network. We show that the
patch representation can retrieve the desired textural aspect in a more precise
manner. We also provide a detailed comparison with state-of-the-art texture
synthesis methods. The GOTEX model based on patch features is also adapted to
texture inpainting and texture interpolation. Finally, we show how to use our
framework to learn a feed-forward neural network that can synthesize on-the-fly
new textures of arbitrary size in a very fast manner. Experimental results and
comparisons with the mainstream methods from the literature illustrate the
relevance of the generative models learned with GOTEX.

    

### [[2007.09248] Fine Timing and Frequency Synchronization for MIMO-OFDM: An Extreme Learning Approach](http://arxiv.org/abs/2007.09248)


  Multiple-input multiple-output orthogonal frequency-division multiplexing
(MIMO-OFDM) is a key technology component in the evolution towards cognitive
radio (CR) in next-generation communication in which the accuracy of timing and
frequency synchronization significantly impacts the overall system performance.
In this paper, we propose a novel scheme leveraging extreme learning machine
(ELM) to achieve high-precision synchronization. Specifically, exploiting the
preamble signals with synchronization offsets, two ELMs are incorporated into a
traditional MIMO-OFDM system to estimate both the residual symbol timing offset
(RSTO) and the residual carrier frequency offset (RCFO). The simulation results
show that the performance of the proposed ELM-based synchronization scheme is
superior to the traditional method under both additive white Gaussian noise
(AWGN) and frequency selective fading channels. Furthermore, comparing with the
existing machine learning based techniques, the proposed method shows
outstanding performance without the requirement of perfect channel state
information (CSI) and prohibitive computational complexity. Finally, the
proposed method is robust in terms of the choice of channel parameters (e.g.,
number of paths) and also in terms of "generalization ability" from a machine
learning standpoint.

    

### [[2007.14052] Multioutput Gaussian Processes with Functional Data: A Study on Coastal Flood Hazard Assessment](http://arxiv.org/abs/2007.14052)


  Surrogate models are often used to replace costly-to-evaluate complex coastal
codes to achieve substantial computational savings. In many of those models,
the hydrometeorological forcing conditions (inputs) or flood events (outputs)
are conveniently parameterized by scalar representations, neglecting that the
inputs are actually time series and that floods propagate spatially inland.
Both facts are crucial in flood prediction for complex coastal systems. Our aim
is to establish a surrogate model that accounts for time-varying inputs and
provides information on spatially varying inland flooding. We introduce a
multioutput Gaussian process model based on a separable kernel that correlates
both functional inputs and spatial locations. Efficient implementations
consider tensor-structured computations or sparse-variational approximations.
In several experiments, we demonstrate the versatility of the model for both
learning maps and inferring unobserved maps, numerically showing the
convergence of predictions as the number of learning maps increases. We assess
our framework in a coastal flood prediction application. Predictions are
obtained with small error values within computation time highly compatible with
short-term forecast requirements (on the order of minutes compared to the days
required by hydrodynamic simulators). We conclude that our framework is a
promising approach for forecast and early-warning systems.

    

### [[2007.14861] Efficient Sparse Secure Aggregation for Federated Learning](http://arxiv.org/abs/2007.14861)


  Federated Learning enables one to jointly train a machine learning model
across distributed clients holding sensitive datasets. In real-world settings,
this approach is hindered by expensive communication and privacy concerns. Both
of these challenges have already been addressed individually, resulting in
competing optimisations. In this article, we tackle them simultaneously for one
of the first times. More precisely, we adapt compression-based federated
techniques to additive secret sharing, leading to an efficient secure
aggregation protocol, with an adaptable security level. We prove its privacy
against malicious adversaries and its correctness in the semi-honest setting.
Experiments on deep convolutional networks demonstrate that our secure protocol
achieves high accuracy with low communication costs. Compared to prior works on
secure aggregation, our protocol has a lower communication and computation
costs for a similar accuracy.

    

### [[2009.00664] VeRNAl: Mining RNA Structures for Fuzzy Base Pairing Network Motifs](http://arxiv.org/abs/2009.00664)


  RNA 3D motifs are recurrent substructures, modelled as networks of base pair
interactions, which are crucial for understanding structure-function
relationships. The task of automatically identifying such motifs is
computationally hard, and remains a key challenge in the field of RNA
structural biology and network analysis. State of the art methods solve special
cases of the motif problem by constraining the structural variability in
occurrences of a motif, and narrowing the substructure search space. Here, we
relax these constraints by posing the motif finding problem as a graph
representation learning and clustering task. This framing takes advantage of
the continuous nature of graph representations to model the flexibility and
variability of RNA motifs in an efficient manner. We propose a set of node
similarity functions, clustering methods, and motif construction algorithms to
recover flexible RNA motifs. Our tool, VeRNAl can be easily customized by users
to desired levels of motif flexibility, abundance and size. We show that VeRNAl
is able to retrieve and expand known classes of motifs, as well as to propose
novel motifs.

    

### [[2009.02327] OnsagerNet: Learning Stable and Interpretable Dynamics using a Generalized Onsager Principle](http://arxiv.org/abs/2009.02327)


  We propose a systematic method for learning stable and physically
interpretable dynamical models using sampled trajectory data from physical
processes based on a generalized Onsager principle. The learned dynamics are
autonomous ordinary differential equations parameterized by neural networks
that retain clear physical structure information, such as free energy,
diffusion, conservative motion and external forces. For high dimensional
problems with a low dimensional slow manifold, an autoencoder with metric
preserving regularization is introduced to find the low dimensional generalized
coordinates on which we learn the generalized Onsager dynamics. Our method
exhibits clear advantages over existing methods on benchmark problems for
learning ordinary differential equations. We further apply this method to study
Rayleigh-Benard convection and learn Lorenz-like low dimensional autonomous
reduced order models that capture both qualitative and quantitative properties
of the underlying dynamics. This forms a general approach to building reduced
order models for forced dissipative systems.

    

### [[2009.06087] Neural Networks Enhancement with Logical Knowledge](http://arxiv.org/abs/2009.06087)


  In the recent past, there has been a growing interest in Neural-Symbolic
Integration frameworks, i.e., hybrid systems that integrate connectionist and
symbolic approaches to obtain the best of both worlds. In a previous work, we
proposed KENN (Knowledge Enhanced Neural Networks), a Neural-Symbolic
architecture that injects prior logical knowledge into a neural network by
adding a new final layer which modifies the initial predictions accordingly to
the knowledge. Among the advantages of this strategy, there is the inclusion of
clause weights, learnable parameters that represent the strength of the
clauses, meaning that the model can learn the impact of each clause on the
final predictions. As a special case, if the training data contradicts a
constraint, KENN learns to ignore it, making the system robust to the presence
of wrong knowledge. In this paper, we propose an extension of KENN for
relational data. To evaluate this new extension, we tested it with different
learning configurations on Citeseer, a standard dataset for Collective
Classification. The results show that KENN is capable of increasing the
performances of the underlying neural network even in the presence relational
data, outperforming other two notable methods that combine learning with logic.

    

### [[2009.14639] Dissected 3D CNNs: Temporal Skip Connections for Efficient Online Video Processing](http://arxiv.org/abs/2009.14639)


  Convolutional Neural Networks with 3D kernels (3D-CNNs) currently achieve
state-of-the-art results in video recognition tasks due to their supremacy in
extracting spatiotemporal features within video frames. There have been many
successful 3D-CNN architectures surpassing the state-of-the-art results
successively. However, nearly all of them are designed to operate offline
creating several serious handicaps during online operation. Firstly,
conventional 3D-CNNs are not dynamic since their output features represent the
complete input clip instead of the most recent frame in the clip. Secondly,
they are not temporal resolution-preserving due to their inherent temporal
downsampling. Lastly, 3D-CNNs are constrained to be used with fixed temporal
input size limiting their flexibility. In order to address these drawbacks, we
propose dissected 3D-CNNs, where the intermediate volumes of the network are
dissected and propagated over depth (time) dimension for future calculations,
substantially reducing the number of computations at online operation. For
action classification, the dissected version of ResNet models performs 77-90%
fewer computations at online operation while achieving ~5% better
classification accuracy on the Kinetics-600 dataset than conventional 3D-ResNet
models. Moreover, the advantages of dissected 3D-CNNs are demonstrated by
deploying our approach onto several vision tasks, which consistently improved
the performance.

    

### [[2010.00373] Task Agnostic Continual Learning Using Online Variational Bayes with Fixed-Point Updates](http://arxiv.org/abs/2010.00373)


  Background: Catastrophic forgetting is the notorious vulnerability of neural
networks to the changes in the data distribution during learning. This
phenomenon has long been considered a major obstacle for using learning agents
in realistic continual learning settings. A large body of continual learning
research assumes that task boundaries are known during training. However, only
a few works consider scenarios in which task boundaries are unknown or not well
defined -- task agnostic scenarios. The optimal Bayesian solution for this
requires an intractable online Bayes update to the weights posterior.
Contributions: We aim to approximate the online Bayes update as accurately as
possible. To do so, we derive novel fixed-point equations for the online
variational Bayes optimization problem, for multivariate Gaussian parametric
distributions. By iterating the posterior through these fixed-point equations,
we obtain an algorithm (FOO-VB) for continual learning which can handle
non-stationary data distribution using a fixed architecture and without using
external memory (i.e. without access to previous data). We demonstrate that our
method (FOO-VB) outperforms existing methods in task agnostic scenarios. FOO-VB
Pytorch implementation will be available online.

    

### [[2010.01777] A Unified View on Graph Neural Networks as Graph Signal Denoising](http://arxiv.org/abs/2010.01777)


  Graph Neural Networks (GNNs) have risen to prominence in learning
representations for graph structured data. A single GNN layer typically
consists of a feature transformation and a feature aggregation operation. The
former normally uses feed-forward networks to transform features, while the
latter aggregates the transformed features over the graph. Numerous recent
works have proposed GNN models with different designs in the aggregation
operation. In this work, we establish mathematically that the aggregation
processes in a group of representative GNN models including GCN, GAT, PPNP, and
APPNP can be regarded as (approximately) solving a graph denoising problem with
a smoothness assumption. Such a unified view across GNNs not only provides a
new perspective to understand a variety of aggregation operations but also
enables us to develop a unified graph neural network framework UGNN. To
demonstrate its promising potential, we instantiate a novel GNN model,
ADA-UGNN, derived from UGNN, to handle graphs with adaptive smoothness across
nodes. Comprehensive experiments show the effectiveness of ADA-UGNN.

    

### [[2010.09429] Neural Additive Vector Autoregression Models for Causal Discovery in Time Series](http://arxiv.org/abs/2010.09429)


  Causal structure discovery in complex dynamical systems is an important
challenge for many scientific domains. Although data from (interventional)
experiments is usually limited, large amounts of observational time series data
sets are usually available. Current methods that learn causal structure from
time series often assume linear relationships. Hence, they may fail in
realistic settings that contain nonlinear relations between the variables. We
propose Neural Additive Vector Autoregression (NAVAR) models, a neural approach
to causal structure learning that can discover nonlinear relationships. We
train deep neural networks that extract the (additive) Granger causal
influences from the time evolution in multi-variate time series. The method
achieves state-of-the-art results on various benchmark data sets for causal
discovery, while providing clear interpretations of the mapped causal
relations.

    

### [[2011.03180] FedSL: Federated Split Learning on Distributed Sequential Data in Recurrent Neural Networks](http://arxiv.org/abs/2011.03180)


  Federated Learning (FL) and Split Learning (SL) are privacy-preserving
Machine-Learning (ML) techniques that enable training ML models over data
distributed among clients without requiring direct access to their raw data.
Existing FL and SL approaches work on horizontally or vertically partitioned
data and cannot handle sequentially partitioned data where segments of
multiple-segment sequential data are distributed across clients. In this paper,
we propose a novel federated split learning framework, FedSL, to train models
on distributed sequential data. The most common ML models to train on
sequential data are Recurrent Neural Networks (RNNs). Since the proposed
framework is privacy preserving, segments of multiple-segment sequential data
cannot be shared between clients or between clients and server. To circumvent
this limitation, we propose a novel SL approach tailored for RNNs. A RNN is
split into sub-networks, and each sub-network is trained on one client
containing single segments of multiple-segment training sequences. During local
training, the sub-networks on different clients communicate with each other to
capture latent dependencies between consecutive segments of multiple-segment
sequential data on different clients, but without sharing raw data or complete
model parameters. After training local sub-networks with local sequential data
segments, all clients send their sub-networks to a federated server where
sub-networks are aggregated to generate a global model. The experimental
results on simulated and real-world datasets demonstrate that the proposed
method successfully train models on distributed sequential data, while
preserving privacy, and outperforms previous FL and centralized learning
approaches in terms of achieving higher accuracy in fewer communication rounds.

    

### [[2011.03525] SigNet: A Novel Deep Learning Framework for Radio Signal Classification](http://arxiv.org/abs/2011.03525)


  Deep learning methods achieve great success in many areas due to their
powerful feature extraction capabilities and end-to-end training mechanism, and
recently they are also introduced for radio signal modulation classification.
In this paper, we propose a novel deep learning framework called SigNet, where
a signal-to-matrix (S2M) operator is adopted to convert the original signal
into a square matrix first and is co-trained with a follow-up CNN architecture
for classification. This model is further accelerated by integrating 1D
convolution operators, leading to the upgraded model SigNet2.0. The simulations
on two signal datasets show that both SigNet and SigNet2.0 outperform a number
of well-known baselines. More interestingly, our proposed models behave
extremely well in small-sample learning when only a small training dataset is
provided. They can achieve a relatively high accuracy even when 1\% training
data are kept, while other baseline models may lose their effectiveness much
more quickly as the datasets get smaller. Such result suggests that
SigNet/SigNet2.0 could be extremely useful in the situations where labeled
signal data are difficult to obtain. The visualization of the output features
of our models demonstrates that our model can well divide different modulation
types of signals in the feature hyper-space.

    

### [[2011.05348] SALR: Sharpness-aware Learning Rate Scheduler for Improved Generalization](http://arxiv.org/abs/2011.05348)


  In an effort to improve generalization in deep learning and automate the
process of learning rate scheduling, we propose SALR: a sharpness-aware
learning rate update technique designed to recover flat minimizers. Our method
dynamically updates the learning rate of gradient-based optimizers based on the
local sharpness of the loss function. This allows optimizers to automatically
increase learning rates at sharp valleys to increase the chance of escaping
them. We demonstrate the effectiveness of SALR when adopted by various
algorithms over a broad range of networks. Our experiments indicate that SALR
improves generalization, converges faster, and drives solutions to
significantly flatter regions.

    

### [[2011.13120] Evaluation of Out-of-Distribution Detection Performance of Self-Supervised Learning in a Controllable Environment](http://arxiv.org/abs/2011.13120)


  We evaluate the out-of-distribution (OOD) detection performance of
self-supervised learning (SSL) techniques with a new evaluation framework.
Unlike the previous evaluation methods, the proposed framework adjusts the
distance of OOD samples from the in-distribution samples. We evaluate an
extensive combination of OOD detection algorithms on three different
implementations of the proposed framework using simulated samples, images, and
text. SSL methods consistently demonstrated the improved OOD detection
performance in all evaluation settings.

    

### [[2012.08418] Pedestrian Behavior Prediction for Automated Driving: Requirements, Metrics, and Relevant Features](http://arxiv.org/abs/2012.08418)


  Automated vehicles require a comprehensive understanding of traffic
situations to ensure safe and anticipatory driving. In this context, the
prediction of pedestrians is particularly challenging as pedestrian behavior
can be influenced by multiple factors. In this paper, we thoroughly analyze the
requirements on pedestrian behavior prediction for automated driving via a
system-level approach. To this end we investigate real-world pedestrian-vehicle
interactions with human drivers. Based on human driving behavior we then derive
appropriate reaction patterns of an automated vehicle and determine
requirements for the prediction of pedestrians. This includes a novel metric
tailored to measure prediction performance from a system-level perspective. The
proposed metric is evaluated on a large-scale dataset comprising thousands of
real-world pedestrian-vehicle interactions. We furthermore conduct an ablation
study to evaluate the importance of different contextual cues and compare these
results to ones obtained using established performance metrics for pedestrian
prediction. Our results highlight the importance of a system-level approach to
pedestrian behavior prediction.

    

### [[2012.11772] Power-SLIC: Fast Superpixel Segmentations by Diagrams](http://arxiv.org/abs/2012.11772)


  Superpixel algorithms grouping pixels with similar color and other low-level
properties are increasingly used for pre-processing in image segmentation. In
recent years, a focus has been placed on developing geometric superpixel
methods that facilitate the extraction and analysis of geometric image
features. Diagram-based superpixel methods are important among the geometric
methods as they generate compact and sparsely representable superpixels.
Introducing generalized balanced power diagrams to the field of superpixels, we
propose a diagram method called Power-SLIC. Power-SLIC is the first geometric
superpixel method to generate piecewise quadratic boundaries. Its speed,
competitive with fast state-of-the-art methods, is unprecedented for diagram
approaches. Extensive computational experiments show that Power-SLIC
outperforms existing diagram approaches in boundary recall, under segmentation
error, achievable segmentation accuracy, and compression quality. Moreover,
Power-SLIC is robust to Gaussian noise.

    

### [[2101.01076] Understanding Health Video Engagement: An Interpretable Deep Learning Approach](http://arxiv.org/abs/2101.01076)


  Health misinformation on social media devastates physical and mental health,
invalidates health gains, and potentially costs lives. Understanding how health
misinformation is transmitted is an urgent goal for researchers, social media
platforms, health sectors, and policymakers to mitigate those ramifications.
Deep learning methods have been deployed to predict the spread of
misinformation. While achieving the state-of-the-art predictive performance,
deep learning methods lack the interpretability due to their blackbox nature.
To remedy this gap, this study proposes a novel interpretable deep learning
approach, Generative Adversarial Network based Piecewise Wide and Attention
Deep Learning (GAN-PiWAD), to predict health misinformation transmission in
social media. Improving upon state-of-the-art interpretable methods, GAN-PiWAD
captures the interactions among multi-modal data, offers unbiased estimation of
the total effect of each feature, and models the dynamic total effect of each
feature when its value varies. We select features according to social exchange
theory and evaluate GAN-PiWAD on 4,445 misinformation videos. The proposed
approach outperformed strong benchmarks. Interpretation of GAN-PiWAD indicates
video description, negative video content, and channel credibility are key
features that drive viral transmission of misinformation. This study
contributes to IS with a novel interpretable deep learning method that is
generalizable to understand other human decision factors. Our findings provide
direct implications for social media platforms and policymakers to design
proactive interventions to identify misinformation, control transmissions, and
manage infodemics.

    

### [[2101.12353] On the capacity of deep generative networks for approximating distributions](http://arxiv.org/abs/2101.12353)


  We study the efficacy and efficiency of deep generative networks for
approximating probability distributions. We prove that neural networks can
transform a low-dimensional source distribution to a distribution that is
arbitrarily close to a high-dimensional target distribution, when the closeness
are measured by Wasserstein distances and maximum mean discrepancy. Upper
bounds of the approximation error are obtained in terms of the width and depth
of neural network. Furthermore, it is shown that the approximation error in
Wasserstein distance grows at most linearly on the ambient dimension and that
the approximation order only depends on the intrinsic dimension of the target
distribution. On the contrary, when $f$-divergences are used as metrics of
distributions, the approximation property is different. We show that in order
to approximate the target distribution in $f$-divergences, the dimension of the
source distribution cannot be smaller than the intrinsic dimension of the
target distribution.

    

### [[2101.12677] Diminishing Domain Bias by Leveraging Domain Labels in Object Detection on UAVs](http://arxiv.org/abs/2101.12677)


  Object detection from Unmanned Aerial Vehicles (UAVs) is of great importance
in many aerial vision-based applications. Despite the great success of generic
object detection methods, a significant performance drop is observed when
applied to images captured by UAVs. This is due to large variations in imaging
conditions, such as varying altitudes, dynamically changing viewing angles, and
different capture times. These variations lead to domain imbalances and, thus,
trained models suffering from domain bias. We demonstrate that domain knowledge
is a valuable source of information and thus propose domain-aware object
detectors by using freely accessible sensor data. By splitting the model into
cross-domain and domain-specific parts, substantial performance improvements
are achieved on multiple data sets across various models and metrics without
changing the architecture. In particular, we achieve a new state-of-the-art
performance on UAVDT for embedded real-time detectors. Furthermore, we create a
new airborne image data set by annotating 13,713 objects in 2,900 images
featuring precise altitude and viewing angle annotations.

    

### [[2102.02611] CKConv: Continuous Kernel Convolution For Sequential Data](http://arxiv.org/abs/2102.02611)


  Conventional neural architectures for sequential data present important
limitations. Recurrent networks suffer from exploding and vanishing gradients,
small effective memory horizons, and must be trained sequentially.
Convolutional networks are unable to handle sequences of unknown size and their
memory horizon must be defined a priori. In this work, we show that all these
problems can be solved by formulating convolutional kernels in CNNs as
continuous functions. The resulting Continuous Kernel Convolution (CKConv)
allows us to model arbitrarily long sequences in a parallel manner, within a
single operation, and without relying on any form of recurrence. We show that
Continuous Kernel Convolutional Networks (CKCNNs) obtain state-of-the-art
results in multiple datasets, e.g., permuted MNIST, and, thanks to their
continuous nature, are able to handle non-uniformly sampled datasets and
irregularly-sampled data natively. CKCNNs match or perform better than neural
ODEs designed for these purposes in a faster and simpler manner.

    

### [[2102.02926] Alchemy: A structured task distribution for meta-reinforcement learning](http://arxiv.org/abs/2102.02926)


  There has been rapidly growing interest in meta-learning as a method for
increasing the flexibility and sample efficiency of reinforcement learning. One
problem in this area of research, however, has been a scarcity of adequate
benchmark tasks. In general, the structure underlying past benchmarks has
either been too simple to be inherently interesting, or too ill-defined to
support principled analysis. In the present work, we introduce a new benchmark
for meta-RL research, which combines structural richness with structural
transparency. Alchemy is a 3D video game, implemented in Unity, which involves
a latent causal structure that is resampled procedurally from episode to
episode, affording structure learning, online inference, hypothesis testing and
action sequencing based on abstract domain knowledge. We evaluate a pair of
powerful RL agents on Alchemy and present an in-depth analysis of one of these
agents. Results clearly indicate a frank and specific failure of meta-learning,
providing validation for Alchemy as a challenging benchmark for meta-RL.
Concurrent with this report, we are releasing Alchemy as public resource,
together with a suite of analysis tools and sample agent trajectories.

    

### [[2102.03988] Ising Model Selection Using $\ell_{1}$-Regularized Linear Regression: A Statistical Mechanics Analysis](http://arxiv.org/abs/2102.03988)


  We theoretically investigate the typical learning performance of
$\ell_{1}$-regularized linear regression ($\ell_1$-LinR) for Ising model
selection using the replica method from statistical mechanics. For typical
random regular (RR) graphs in the paramagnetic phase, an accurate estimate of
the typical sample complexity of $\ell_1$-LinR is obtained, demonstrating that,
for an Ising model with $N$ variables, $\ell_1$-LinR is model selection
consistent with $M=\mathcal{O}\left(\log N\right)$ samples. Moreover, we
provide a computationally efficient method to accurately predict the
non-asymptotic behavior of $\ell_1$-LinR for moderate $M$ and $N$, such as the
precision and recall rates. Simulations show a fairly good agreement between
the theoretical predictions and experimental results, even for graphs with many
loops, which supports our findings. Although this paper focuses on
$\ell_1$-LinR, our method is readily applicable for precisely investigating the
typical learning performances of a wide class of $\ell_{1}$-regularized
M-estimators including $\ell_{1}$-regularized logistic regression and
interaction screening.

    

### [[2102.12142] Gaussian boson sampling and multi-particle event optimization by machine learning in the quantum phase space](http://arxiv.org/abs/2102.12142)


  We use neural networks to represent the characteristic function of many-body
Gaussian states in the quantum phase space. By a pullback mechanism, we model
transformations due to unitary operators as linear layers that can be cascaded
to simulate complex multi-particle processes. We use the layered neural
networks for non-classical light propagation in random interferometers, and
compute boson pattern probabilities by automatic differentiation. We also
demonstrate that multi-particle events in Gaussian boson sampling can be
optimized by a proper design and training of the neural network weights. The
results are potentially useful to the creation of new sources and complex
circuits for quantum technologies.

    

### [[2103.00075] Noisy Truncated SGD: Optimization and Generalization](http://arxiv.org/abs/2103.00075)


  Recent empirical work on stochastic gradient descent (SGD) applied to
over-parameterized deep learning has shown that most gradient components over
epochs are quite small. Inspired by such observations, we rigorously study
properties of Truncated SGD (T-SGD), that truncates the majority of small
gradient components to zeros. Considering non-convex optimization problems, we
show that the convergence rate of T-SGD matches the order of vanilla SGD. We
also establish the generalization error bound for T-SGD. Further, we propose
Noisy Truncated SGD (NT-SGD), which adds Gaussian noise to the truncated
gradients. We prove that NT-SGD has the same convergence rate as T-SGD for
non-convex optimization problems. We demonstrate that with the help of noise,
NT-SGD can provably escape from saddle points and requires less noise compared
to previous related work. We also prove that NT-SGD achieves better
generalization error bound compared to T-SGD because of the noise. Our
generalization analysis is based on uniform stability and we show that
additional noise in the gradient update can boost the stability. Our
experiments on a variety of benchmark datasets (MNIST, Fashion-MNIST, CIFAR-10,
and CIFAR-100) with various networks (VGG and ResNet) validate the theoretical
properties of NT-SGD, i.e., NT-SGD matches the speed and accuracy of vanilla
SGD while effectively working with sparse gradients, and can successfully
escape poor local minima.

    

### [[2103.01946] Fixing Data Augmentation to Improve Adversarial Robustness](http://arxiv.org/abs/2103.01946)


  Adversarial training suffers from robust overfitting, a phenomenon where the
robust test accuracy starts to decrease during training. In this paper, we
focus on both heuristics-driven and data-driven augmentations as a means to
reduce robust overfitting. First, we demonstrate that, contrary to previous
findings, when combined with model weight averaging, data augmentation can
significantly boost robust accuracy. Second, we explore how state-of-the-art
generative models can be leveraged to artificially increase the size of the
training set and further improve adversarial robustness. Finally, we evaluate
our approach on CIFAR-10 against $\ell_\infty$ and $\ell_2$ norm-bounded
perturbations of size $\epsilon = 8/255$ and $\epsilon = 128/255$,
respectively. We show large absolute improvements of +7.06% and +5.88% in
robust accuracy compared to previous state-of-the-art methods. In particular,
against $\ell_\infty$ norm-bounded perturbations of size $\epsilon = 8/255$,
our model reaches 64.20% robust accuracy without using any external data,
beating most prior works that use external data.

    

### [[2103.09052] Selective Intervention Planning using Restless Multi-Armed Bandits to Improve Maternal and Child Health Outcomes](http://arxiv.org/abs/2103.09052)


  India has a maternal mortality ratio of 113 and child mortality ratio of 2830
per 100,000 live births. Lack of access to preventive care information is a
major contributing factor for these deaths, especially in low resource
households. We partner with ARMMAN, a non-profit based in India employing a
call-based information program to disseminate health-related information to
pregnant women and women with recent child deliveries. We analyze call records
of over 300,000 women registered in the program created by ARMMAN and try to
identify women who might not engage with these call programs that are proven to
result in positive health outcomes. We built machine learning based models to
predict the long term engagement pattern from call logs and beneficiaries'
demographic information, and discuss the applicability of this method in the
real world through a pilot validation. Through a pilot service quality
improvement study, we show that using our model's predictions to make
interventions boosts engagement metrics by 61.37%. We then formulate the
intervention planning problem as restless multi-armed bandits (RMABs), and
present preliminary results using this approach.

    

### [[2103.09743] Deep Learning-based Extreme Heatwave Forecast](http://arxiv.org/abs/2103.09743)


  Because of the impact of extreme heat waves and heat domes on society and
biodiversity, their study is a key challenge. Physics driven weather forecast
systems or climate models can be used to forecast their occurrence or predict
their probability. The present work explores the use of deep learning
architectures, trained using outputs of a climate model, as an alternative
strategy to forecast extreme heatwave occurrences. This new approach will be
useful for several key scientific goals which include the study of climate
model statistics, building a quantitative proxy for resampling rare events in
climate models, study the impact of climate change, and should eventually be
useful for forecasting. Fulfilling these important goals implies addressing
issues such as class-size imbalance that is intrinsically associated with rare
event prediction, assessing the potential benefits of transfer learning to
address the nested nature of extreme events (naturally included in less extreme
ones). We train a Convolutional Neural Network, using $1000$ years of climate
model outputs, with large-class undersampling and transfer learning. From the
observed snapshots of the surface temperature and the $500$ hPa geopotential
height fields, the trained network achieves significant performance in
forecasting the occurrence of long lasting extreme heatwaves. We are able to
predict them at three different levels of intensity, and as early as $15$ days
ahead of the start of the event ($30$ days ahead of the end of the event).

    

### [[2103.10245] Building Safer Autonomous Agents by Leveraging Risky Driving Behavior Knowledge](http://arxiv.org/abs/2103.10245)


  Simulation environments are good for learning different driving tasks like
lane changing, parking or handling intersections etc. in an abstract manner.
However, these simulation environments often restrict themselves to operate
under conservative interaction behavior amongst different vehicles. But, as we
know, real driving tasks often involve very high risk scenarios where other
drivers often don't behave in the expected sense. There can be many reasons for
this behavior like being tired or inexperienced. The simulation environment
doesn't take this information into account while training the navigation agent.
Therefore, in this study we especially focus on systematically creating these
risk prone scenarios with heavy traffic and unexpected random behavior for
creating better model-free learning agents. We generate multiple autonomous
driving scenarios by creating new custom Markov Decision Process (MDP)
environment iterations in the highway-env simulation package. The behavior
policy is learnt by agents trained with the help from deep reinforcement
learning models. Our behavior policy is deliberated to handle collisions and
risky randomized driver behavior. We train model free learning agents with
supplement information of risk prone driving scenarios and compare their
performance with baseline agents. Finally, we casually measure the impact of
adding these perturbations in the training process to precisely account for the
performance improvement obtained from utilizing the learnings from these
scenarios.

    

### [[2103.11933] PatentSBERTa: A Deep NLP based Hybrid Model for Patent Distance and Classification using Augmented SBERT](http://arxiv.org/abs/2103.11933)


  This study provides an efficient approach for using text data to calculate
patent-to-patent (p2p) technological similarity, and presents a hybrid
framework for leveraging the resulting p2p similarity for applications such as
semantic search and automated patent classification. We create embeddings using
Sentence-BERT (SBERT) based on patent claims. We leverage SBERTs efficiency in
creating embedding distance measures to map p2p similarity in large sets of
patent data. We deploy our framework for classification with a simple Nearest
Neighbors (KNN) model that predicts Cooperative Patent Classification (CPC) of
a patent based on the class assignment of the K patents with the highest p2p
similarity. We thereby validate that the p2p similarity captures their
technological features in terms of CPC overlap, and at the same demonstrate the
usefulness of this approach for automatic patent classification based on text
data. Furthermore, the presented classification framework is simple and the
results easy to interpret and evaluate by end-users. In the out-of-sample model
validation, we are able to perform a multi-label prediction of all assigned CPC
classes on the subclass (663) level on 1,492,294 patents with an accuracy of
54% and F1 score > 66%, which suggests that our model outperforms the current
state-of-the-art in text-based multi-label and multi-class patent
classification. We furthermore discuss the applicability of the presented
framework for semantic IP search, patent landscaping, and technology
intelligence. We finally point towards a future research agenda for leveraging
multi-source patent embeddings, their appropriateness across applications, as
well as to improve and validate patent embeddings by creating domain-expert
curated Semantic Textual Similarity (STS) benchmark datasets.

    

### [[2103.13300] Automatic Cough Classification for Tuberculosis Screening in a Real-World Environment](http://arxiv.org/abs/2103.13300)


  Objective: The automatic discrimination between the coughing sounds produced
by patients with tuberculosis (TB) and those produced by patients with other
lung ailments.
Approach: We present experiments based on a dataset of 1358 forced cough
recordings obtained in a developing-world clinic from 16 patients with
confirmed active pulmonary TB and 35 patients suffering from respiratory
conditions suggestive of TB but confirmed to be TB negative. Using nested
cross-validation, we have trained and evaluated five machine learning
classifiers: logistic regression (LR), support vector machines (SVM), k-nearest
neighbour (KNN), multilayer perceptrons (MLP) and convolutional neural networks
(CNN).
Main Results: Although classification is possible in all cases, the best
performance is achieved using LR. In combination with feature selection by
sequential forward selection (SFS), our best LR system achieves an area under
the ROC curve (AUC) of 0.94 using 23 features selected from a set of 78
high-resolution mel-frequency cepstral coefficients (MFCCs). This system
achieves a sensitivity of 93\% at a specificity of 95\% and thus exceeds the
90\% sensitivity at 70\% specificity specification considered by the World
Health Organisation (WHO) as a minimal requirement for a community-based TB
triage test.
Significance: The automatic classification of cough audio sounds, when
applied to symptomatic patients requiring investigation for TB, can meet the
WHO triage specifications for the identification of patients who should undergo
expensive molecular downstream testing. This makes it a promising and viable
means of low cost, easily deployable frontline screening for TB, which can
benefit especially developing countries with a heavy TB burden.

    

### [[2104.05914] GSA-Forecaster: Forecasting Graph-Based Time-Dependent Data with Graph Sequence Attention](http://arxiv.org/abs/2104.05914)


  Forecasting graph-based time-dependent data has many practical applications.
This task is challenging as models need not only to capture spatial dependency
and temporal dependency within the data, but also to leverage useful auxiliary
information for accurate predictions. In this paper, we analyze limitations of
state-of-the-art models on dealing with temporal dependency. To address this
limitation, we propose GSA-Forecaster, a new deep learning model for
forecasting graph-based time-dependent data. GSA-Forecaster leverages graph
sequence attention (GSA), a new attention mechanism proposed in this paper, for
effectively capturing temporal dependency. GSA-Forecaster embeds the graph
structure of the data into its architecture to address spatial dependency.
GSA-Forecaster also accounts for auxiliary information to further improve
predictions. We evaluate GSA-Forecaster with large-scale real-world graph-based
time-dependent data and demonstrate its effectiveness over state-of-the-art
models with 6.7% RMSE and 5.8% MAPE reduction.

    

### [[2104.07084] Grouped Variable Selection with Discrete Optimization: Computational and Statistical Perspectives](http://arxiv.org/abs/2104.07084)


  We present a new algorithmic framework for grouped variable selection that is
based on discrete mathematical optimization. While there exist several
appealing approaches based on convex relaxations and nonconvex heuristics, we
focus on optimal solutions for the $\ell_0$-regularized formulation, a problem
that is relatively unexplored due to computational challenges. Our methodology
covers both high-dimensional linear regression and nonparametric sparse
additive modeling with smooth components. Our algorithmic framework consists of
approximate and exact algorithms. The approximate algorithms are based on
coordinate descent and local search, with runtimes comparable to popular sparse
learning algorithms. Our exact algorithm is based on a standalone
branch-and-bound (BnB) framework, which can solve the associated mixed integer
programming (MIP) problem to certified optimality. By exploiting the problem
structure, our custom BnB algorithm can solve to optimality problem instances
with $5 \times 10^6$ features and $10^3$ observations in minutes to hours --
over $1000$ times larger than what is currently possible using state-of-the-art
commercial MIP solvers. We also explore statistical properties of the
$\ell_0$-based estimators. We demonstrate, theoretically and empirically, that
our proposed estimators have an edge over popular group-sparse estimators in
terms of statistical performance in various regimes. We provide an open-source
implementation of our proposed framework.

    

### [[2104.11510] Time Series Forecasting via Learning Convolutionally Low-Rank Models](http://arxiv.org/abs/2104.11510)


  Recently,~\citet{liu:arxiv:2019} studied the rather challenging problem of
time series forecasting from the perspective of compressed sensing. They
proposed a no-learning method, named Convolution Nuclear Norm Minimization
(CNNM), and proved that CNNM can exactly recover the future part of a series
from its observed part, provided that the series is convolutionally low-rank.
While impressive, the convolutional low-rankness condition may not be satisfied
whenever the series is far from being seasonal, and is in fact brittle to the
presence of trends and dynamics. This paper tries to approach the issues by
integrating a learnable, orthonormal transformation into CNNM, with the purpose
for converting the series of involute structures into regular signals of
convolutionally low-rank. We prove that the resultant model, termed
Learning-Based CNNM (LbCNNM), strictly succeeds in identifying the future part
of a series, as long as the transform of the series is convolutionally
low-rank. To learn proper transformations that may meet the required success
conditions, we devise an interpretable method based on Principal Component
Purist (PCP). Equipped with this learning method and some elaborate data
argumentation skills, LbCNNM not only can handle well the major components of
time series (including trends, seasonality and dynamics), but also can make use
of the forecasts provided by some other forecasting methods; this means LbCNNM
can be used as a general tool for model combination. Extensive experiments on
100,452 real-world time series from TSDL and M4 demonstrate the superior
performance of LbCNNM.

    

### [[2104.11734] Exact marginal prior distributions of finite Bayesian neural networks](http://arxiv.org/abs/2104.11734)


  Bayesian neural networks are theoretically well-understood only in the
infinite-width limit, where Gaussian priors over network weights yield Gaussian
priors over network outputs. Recent work has suggested that finite Bayesian
networks may outperform their infinite counterparts, but their non-Gaussian
function space priors have been characterized only though perturbative
approaches. Here, we derive exact solutions for the function space priors for
individual input examples of a class of finite fully-connected feedforward
Bayesian neural networks. For deep linear networks, the prior has a simple
expression in terms of the Meijer $G$-function. The prior of a finite ReLU
network is a mixture of the priors of linear networks of smaller widths,
corresponding to different numbers of active units in each layer. Our results
unify previous descriptions of finite network priors in terms of their tail
decay and large-width behavior.

    

### [[2104.12761] Adaptive Learning in Continuous Games: Optimal Regret Bounds and Convergence to Nash Equilibrium](http://arxiv.org/abs/2104.12761)


  In game-theoretic learning, several agents are simultaneously following their
individual interests, so the environment is non-stationary from each player's
perspective. In this context, the performance of a learning algorithm is often
measured by its regret. However, no-regret algorithms are not created equal in
terms of game-theoretic guarantees: depending on how they are tuned, some of
them may drive the system to an equilibrium, while others could produce cyclic,
chaotic, or otherwise divergent trajectories. To account for this, we propose a
range of no-regret policies based on optimistic mirror descent, with the
following desirable properties: i) they do not require any prior tuning or
knowledge of the game; ii) they all achieve O(\sqrt{T}) regret against
arbitrary, adversarial opponents; and iii) they converge to the best response
against convergent opponents. Also, if employed by all players, then iv) they
guarantee O(1) social regret; while v) the induced sequence of play converges
to Nash equilibrium with O(1) individual regret in all variationally stable
games (a class of games that includes all monotone and convex-concave zero-sum
games).

    

### [[2105.01099] Reinforcement Learning for Ridesharing: A Survey](http://arxiv.org/abs/2105.01099)


  In this paper, we present a comprehensive, in-depth survey of the literature
on reinforcement learning approaches to decision optimization problems in a
typical ridesharing system. Papers on the topics of rideshare matching, vehicle
repositioning, ride-pooling, routing, and dynamic pricing are covered. Popular
data sets and open simulation environments are also introduced. Subsequently,
we discuss a number of challenges and opportunities for reinforcement learning
research on this important domain.

    

### [[2105.03425] Kernel Two-Sample Tests for Manifold Data](http://arxiv.org/abs/2105.03425)


  We present a study of kernel based two-sample test statistic, which is
related to the Maximum Mean Discrepancy (MMD), in the manifold data setting,
assuming that high-dimensional observations are close to a low-dimensional
manifold. We characterize the test level and power in relation to the kernel
bandwidth, the number of samples, and the intrinsic dimensionality of the
manifold. Specifically, we show that when data densities are supported on a
$d$-dimensional sub-manifold $\mathcal{M}$ embedded in an $m$-dimensional
space, the kernel two-sample test for data sampled from a pair of distributions
$(p, q)$ that are Hölder with order $\beta$ is consistent and powerful when
the number of samples $n$ is greater than $\delta_2(p,q)^{-2-d/\beta}$ up to
certain constant, where $\delta_2$ is the squared $\ell_2$-divergence between
two distributions on manifold. Moreover, to achieve testing consistency under
this scaling of $n$, our theory suggests that the kernel bandwidth $\gamma$
scales with $n^{-1/(d+2\beta)}$. These results indicate that the kernel
two-sample test does not have a curse-of-dimensionality when the data lie on a
low-dimensional manifold. We demonstrate the validity of our theory and the
property of the kernel test for manifold data using several numerical
experiments.

    

### [[2105.08024] Sample-Efficient Reinforcement Learning Is Feasible for Linearly Realizable MDPs with Limited Revisiting](http://arxiv.org/abs/2105.08024)


  Low-complexity models such as linear function representation play a pivotal
role in enabling sample-efficient reinforcement learning (RL). The current
paper pertains to a scenario with value-based linear representation, which
postulates the linear realizability of the optimal Q-function (also called the
"linear $Q^{\star}$ problem"). While linear realizability alone does not allow
for sample-efficient solutions in general, the presence of a large
sub-optimality gap is a potential game changer, depending on the sampling
mechanism in use. Informally, sample efficiency is achievable with a large
sub-optimality gap when a generative model is available but is unfortunately
infeasible when we turn to standard online RL settings.
In this paper, we make progress towards understanding this linear $Q^{\star}$
problem by investigating a new sampling protocol, which draws samples in an
online/exploratory fashion but allows one to backtrack and revisit previous
states in a controlled and infrequent manner. This protocol is more flexible
than the standard online RL setting, while being practically relevant and far
more restrictive than the generative model. We develop an algorithm tailored to
this setting, achieving a sample complexity that scales polynomially with the
feature dimension, the horizon, and the inverse sub-optimality gap, but not the
size of the state/action space. Our findings underscore the fundamental
interplay between sampling protocols and low-complexity structural
representation in RL.

    

### [[2105.08532] Robust Learning in Heterogeneous Contexts](http://arxiv.org/abs/2105.08532)


  We consider the problem of learning from training data obtained in different
contexts, where the underlying context distribution is unknown and is estimated
empirically. We develop a robust method that takes into account the uncertainty
of the context distribution. Unlike the conventional and overly conservative
minimax approach, we focus on excess risks and construct distribution sets with
statistical coverage to achieve an appropriate trade-off between performance
and robustness. The proposed method is computationally scalable and shown to
interpolate between empirical risk minimization and minimax regret objectives.
Using both real and synthetic data, we demonstrate its ability to provide
robustness in worst-case scenarios without harming performance in the nominal
scenario.

    

### [[2105.10266] Ensemble Quantile Networks: Uncertainty-Aware Reinforcement Learning with Applications in Autonomous Driving](http://arxiv.org/abs/2105.10266)


  Reinforcement learning (RL) can be used to create a decision-making agent for
autonomous driving. However, previous approaches provide only black-box
solutions, which do not offer information on how confident the agent is about
its decisions. An estimate of both the aleatoric and epistemic uncertainty of
the agent's decisions is fundamental for real-world applications of autonomous
driving. Therefore, this paper introduces the Ensemble Quantile Networks (EQN)
method, which combines distributional RL with an ensemble approach, to obtain a
complete uncertainty estimate. The distribution over returns is estimated by
learning its quantile function implicitly, which gives the aleatoric
uncertainty, whereas an ensemble of agents is trained on bootstrapped data to
provide a Bayesian estimation of the epistemic uncertainty. A criterion for
classifying which decisions that have an unacceptable uncertainty is also
introduced. The results show that the EQN method can balance risk and time
efficiency in different occluded intersection scenarios, by considering the
estimated aleatoric uncertainty. Furthermore, it is shown that the trained
agent can use the epistemic uncertainty information to identify situations that
the agent has not been trained for and thereby avoid making unfounded,
potentially dangerous, decisions outside of the training distribution.

    

### [[2105.13348] Optimization in Open Networks via Dual Averaging](http://arxiv.org/abs/2105.13348)


  In networks of autonomous agents (e.g., fleets of vehicles, scattered
sensors), the problem of minimizing the sum of the agents' local functions has
received a lot of interest. We tackle here this distributed optimization
problem in the case of open networks when agents can join and leave the network
at any time. Leveraging recent online optimization techniques, we propose and
analyze the convergence of a decentralized asynchronous optimization method for
open networks.

    

### [[2105.13967] Bridge Data Center AI Systems with Edge Computing for Actionable Information Retrieval](http://arxiv.org/abs/2105.13967)


  Extremely high data rates at modern synchrotron and X-ray free-electron laser
light source beamlines motivate the use of machine learning methods for data
reduction, feature detection, and other purposes. Regardless of the
application, the basic concept is the same: data collected in early stages of
an experiment, data from past similar experiments, and/or data simulated for
the upcoming experiment are used to train machine learning models that, in
effect, learn specific characteristics of those data; these models are then
used to process subsequent data more efficiently than would general-purpose
models that lack knowledge of the specific dataset or data class. Thus, a key
challenge is to be able to train models with sufficient rapidity that they can
be deployed and used within useful timescales. We describe here how specialized
data center AI (DCAI) systems can be used for this purpose through a
geographically distributed workflow. Experiments show that although there are
data movement cost and service overhead to use remote DCAI systems for DNN
training, the turnaround time is still less than 1/30 of using a locally
deploy-able GPU.

    

### [[2106.00198] Gradient play in stochastic games: stationary points, convergence, and sample complexity](http://arxiv.org/abs/2106.00198)


  We study the performance of the gradient play algorithm for stochastic games
(SGs), where each agent tries to maximize its own total discounted reward by
making decisions independently based on current state information which is
shared between agents. Policies are directly parameterized by the probability
of choosing a certain action at a given state. We show that Nash equilibria
(NEs) and first-order stationary policies are equivalent in this setting, and
give a local convergence rate around strict NEs. Further, for a subclass of SGs
called Markov potential games (which includes the cooperative setting with
identical rewards among agents as an important special case), we design a
sample-based reinforcement learning algorithm and give a non-asymptotic global
convergence rate analysis for both exact gradient play and our sample-based
learning algorithm. Our result shows that the number of iterations to reach an
$\epsilon$-NE scales linearly, instead of exponentially, with the number of
agents. Local geometry and local stability are also considered, where we prove
that strict NEs are local maxima of the total potential function and
fully-mixed NEs are saddle points.

    

### [[2106.01917] SpecAttack: Specification-Based Adversarial Training for Deep Neural Networks](http://arxiv.org/abs/2106.01917)


  Safety specification-based adversarial training aims to generate examples
violating a formal safety specification and therefore provides approaches for
repair. The need for maintaining high prediction accuracy while ensuring the
save behavior remains challenging. Thus we present SpecAttack, a
query-efficient counter-example generation and repair method for deep neural
networks. Using SpecAttack allows specifying safety constraints on the model to
find inputs that violate these constraints. These violations are then used to
repair the neural network via re-training such that it becomes provably safe.
We evaluate SpecAttack's performance on the task of counter-example generation
and repair. Our experimental evaluation demonstrates that SpecAttack is in most
cases more query-efficient than comparable attacks, yields counter-examples of
higher quality, with its repair technique being more efficient, maintaining
higher functional correctness, and provably guaranteeing safety specification
compliance.

    

### [[2106.03215] PreferenceNet: Encoding Human Preferences in Auction Design with Deep Learning](http://arxiv.org/abs/2106.03215)


  The design of optimal auctions is a problem of interest in economics, game
theory and computer science. Despite decades of effort, strategyproof,
revenue-maximizing auction designs are still not known outside of restricted
settings. However, recent methods using deep learning have shown some success
in approximating optimal auctions, recovering several known solutions and
outperforming strong baselines when optimal auctions are not known. In addition
to maximizing revenue, auction mechanisms may also seek to encourage socially
desirable constraints such as allocation fairness or diversity. However, these
philosophical notions neither have standardization nor do they have widely
accepted formal definitions. In this paper, we propose PreferenceNet, an
extension of existing neural-network-based auction mechanisms to encode
constraints using (potentially human-provided) exemplars of desirable
allocations. In addition, we introduce a new metric to evaluate an auction
allocations' adherence to such socially desirable constraints and demonstrate
that our proposed method is competitive with current state-of-the-art
neural-network based auction designs. We validate our approach through human
subject research and show that we are able to effectively capture real human
preferences. Our code is available at
this https URL


### [[2106.03227] Neural Tangent Kernel Maximum Mean Discrepancy](http://arxiv.org/abs/2106.03227)


  We present a novel neural network Maximum Mean Discrepancy (MMD) statistic by
identifying a new connection between neural tangent kernel (NTK) and MMD. This
connection enables us to develop a computationally efficient and
memory-efficient approach to compute the MMD statistic and perform NTK based
two-sample tests towards addressing the long-standing challenge of memory and
computational complexity of the MMD statistic, which is essential for online
implementation to assimilating new samples. Theoretically, such a connection
allows us to understand the NTK test statistic properties, such as the Type-I
error and testing power for performing the two-sample test, by adapting
existing theories for kernel MMD. Numerical experiments on synthetic and
real-world datasets validate the theory and demonstrate the effectiveness of
the proposed NTK-MMD statistic.

    

### [[2106.03762] Frustratingly Easy Uncertainty Estimation for Distribution Shift](http://arxiv.org/abs/2106.03762)


  Distribution shift is an important concern in deep image classification,
produced either by corruption of the source images, or a complete change, with
the solution involving domain adaptation. While the primary goal is to improve
accuracy under distribution shift, an important secondary goal is uncertainty
estimation: evaluating the probability that the prediction of a model is
correct. While improving accuracy is hard, uncertainty estimation turns out to
be frustratingly easy. Prior works have appended uncertainty estimation into
the model and training paradigm in various ways. Instead, we show that we can
estimate uncertainty by simply exposing the original model to corrupted images,
and performing simple statistical calibration on the image outputs. Our
frustratingly easy methods demonstrate superior performance on a wide range of
distribution shifts as well as on unsupervised domain adaptation tasks,
measured through extensive experimentation.

    

### [[2106.04117] The best of both worlds: stochastic and adversarial episodic MDPs with unknown transition](http://arxiv.org/abs/2106.04117)


  We consider the best-of-both-worlds problem for learning an episodic Markov
Decision Process through $T$ episodes, with the goal of achieving
$\widetilde{\mathcal{O}}(\sqrt{T})$ regret when the losses are adversarial and
simultaneously $\mathcal{O}(\text{polylog}(T))$ regret when the losses are
(almost) stochastic. Recent work by [Jin and Luo, 2020] achieves this goal when
the fixed transition is known, and leaves the case of unknown transition as a
major open question. In this work, we resolve this open problem by using the
same Follow-the-Regularized-Leader ($\text{FTRL}$) framework together with a
set of new techniques. Specifically, we first propose a loss-shifting trick in
the $\text{FTRL}$ analysis, which greatly simplifies the approach of [Jin and
Luo, 2020] and already improves their results for the known transition case.
Then, we extend this idea to the unknown transition case and develop a novel
analysis which upper bounds the transition estimation error by (a fraction of)
the regret itself in the stochastic setting, a key property to ensure
$\mathcal{O}(\text{polylog}(T))$ regret.

    

### [[2106.05232] Realizing GANs via a Tunable Loss Function](http://arxiv.org/abs/2106.05232)


  We introduce a tunable GAN, called $\alpha$-GAN, parameterized by $\alpha \in
(0,\infty]$, which interpolates between various $f$-GANs and Integral
Probability Metric based GANs (under constrained discriminator set). We
construct $\alpha$-GAN using a supervised loss function, namely, $\alpha$-loss,
which is a tunable loss function capturing several canonical losses. We show
that $\alpha$-GAN is intimately related to the Arimoto divergence, which was
first proposed by Österriecher (1996), and later studied by Liese and Vajda
(2006). We also study the convergence properties of $\alpha$-GAN. We posit that
the holistic understanding that $\alpha$-GAN introduces will have practical
benefits of addressing both the issues of vanishing gradients and mode
collapse.

    

### [[2106.05565] Identifiability of interaction kernels in mean-field equations of interacting particles](http://arxiv.org/abs/2106.05565)


  We study the identifiability of the interaction kernels in mean-field
equations for intreacting particle systems. The key is to identify function
spaces on which a probabilistic loss functional has a unique minimizer. We
prove that identifiability holds on any subspace of two reproducing kernel
Hilbert spaces (RKHS), whose reproducing kernels are intrinsic to the system
and are data-adaptive. Furthermore, identifiability holds on two ambient L2
spaces if and only if the integral operators associated with the reproducing
kernels are strictly positive. Thus, the inverse problem is ill-posed in
general. We also discuss the implications of identifiability in computational
practice.

    

### [[2106.07827] Improving the compromise between accuracy, interpretability and personalization of rule-based machine learning in medical problems](http://arxiv.org/abs/2106.07827)


  One of the key challenges when developing a predictive model is the
capability to describe the domain knowledge and the cause-effect relationships
in a simple way. Decision rules are a useful and important methodology in this
context, justifying their application in several areas, particularly in
clinical practice. Several machine-learning classifiers have exploited the
advantageous properties of decision rules to build intelligent prediction
models, namely decision trees and ensembles of trees (ETs). However, such
methodologies usually suffer from a trade-off between interpretability and
predictive performance. Some procedures consider a simplification of ETs, using
heuristic approaches to select an optimal reduced set of decision rules. In
this paper, we introduce a novel step to those methodologies. We create a new
component to predict if a given rule will be correct or not for a particular
patient, which introduces personalization into the procedure. Furthermore, the
validation results using three public clinical datasets suggest that it also
allows to increase the predictive performance of the selected set of rules,
improving the mentioned trade-off.

    

### [[2106.09215] Optimum-statistical Collaboration Towards General and Efficient Black-box Optimization](http://arxiv.org/abs/2106.09215)


  In this paper, we make the key delineation on the roles of resolution and
statistical uncertainty in black-box optimization, guiding a more general
analysis and a more efficient algorithm design. We introduce
\textit{optimum-statistical collaboration}, an algorithm framework of managing
the interaction between optimization error flux and statistical error flux
evolving in the optimization process. We provide a general analysis of the
framework without specific forms of the statistical error and the uncertainty
quantifier. Our framework and its analysis, because of their generality, can be
applied to functions and partitions that satisfy different local smoothness
assumptions and has different number of local optimums, which is much larger
than the class of functions studied in prior works. Our framework also inspires
us to propose a better measure of the statistical uncertainty and consequently
a variance-adaptive algorithm \texttt{VHCT}. In theory, we prove the algorithm
enjoys rate-optimal regret bounds under different local smoothness assumptions;
in experiments, we show the algorithm outperforms prior efforts in different
settings.

    

### [[2106.09685] LoRA: Low-Rank Adaptation of Large Language Models](http://arxiv.org/abs/2106.09685)


  An important paradigm of natural language processing consists of large-scale
pre-training on general domain data and adaptation to particular tasks or
domains. As we pre-train larger models, full fine-tuning, which retrains all
model parameters, becomes less feasible. Using GPT-3 175B as an example --
deploying independent instances of fine-tuned models, each with 175B
parameters, is prohibitively expensive. We propose Low-Rank Adaptation, or
LoRA, which freezes the pre-trained model weights and injects trainable rank
decomposition matrices into each layer of the Transformer architecture, greatly
reducing the number of trainable parameters for downstream tasks. Compared to
GPT-3 175B fine-tuned with Adam, LoRA can reduce the number of trainable
parameters by 10,000 times and the GPU memory requirement by 3 times. LoRA
performs on-par or better than fine-tuning in model quality on RoBERTa,
DeBERTa, GPT-2, and GPT-3, despite having fewer trainable parameters, a higher
training throughput, and, unlike adapters, no additional inference latency. We
also provide an empirical investigation into rank-deficiency in language model
adaptation, which sheds light on the efficacy of LoRA. We release a package
that facilitates the integration of LoRA with PyTorch models and provide our
implementations and model checkpoints for RoBERTa, DeBERTa, and GPT-2 at
this https URL.

    

### [[2106.10065] Being a Bit Frequentist Improves Bayesian Neural Networks](http://arxiv.org/abs/2106.10065)


  Despite their compelling theoretical properties, Bayesian neural networks
(BNNs) tend to perform worse than frequentist methods in classification-based
uncertainty quantification (UQ) tasks such as out-of-distribution (OOD)
detection. In this paper, based on empirical findings in prior works, we
hypothesize that this issue is because even recent Bayesian methods have never
considered OOD data in their training processes, even though this ``OOD
training'' technique is an integral part of state-of-the-art frequentist UQ
methods. To validate this, we treat OOD data as a first-class citizen in BNN
training by exploring four different ways of incorporating OOD data into
Bayesian inference. We show in extensive experiments that OOD-trained BNNs are
competitive to recent frequentist baselines. This work thus provides strong
baselines for future work in Bayesian UQ.

    

### [[2106.10404] Sparse Training via Boosting Pruning Plasticity with Neuroregeneration](http://arxiv.org/abs/2106.10404)


  Works on lottery ticket hypothesis (LTH) and single-shot network pruning
(SNIP) have raised a lot of attention currently on post-training pruning
(iterative magnitude pruning), and before-training pruning (pruning at
initialization). The former method suffers from an extremely large computation
cost and the latter usually struggles with insufficient performance. In
comparison, during-training pruning, a class of pruning methods that
simultaneously enjoys the training/inference efficiency and the comparable
performance, temporarily, has been less explored. To better understand
during-training pruning, we quantitatively study the effect of pruning
throughout training from the perspective of pruning plasticity (the ability of
the pruned networks to recover the original performance). Pruning plasticity
can help explain several other empirical observations about neural network
pruning in literature. We further find that pruning plasticity can be
substantially improved by injecting a brain-inspired mechanism called
neuroregeneration, i.e., to regenerate the same number of connections as
pruned. We design a novel gradual magnitude pruning (GMP) method, named gradual
pruning with zero-cost neuroregeneration (\textbf{GraNet}), that advances state
of the art. Perhaps most impressively, its sparse-to-sparse version for the
first time boosts the sparse-to-sparse training performance over various
dense-to-sparse methods with ResNet-50 on ImageNet without extending the
training time. We release all codes in
this https URL.

    

### [[2106.15358] Towards Sample-Optimal Compressive Phase Retrieval with Sparse and Generative Priors](http://arxiv.org/abs/2106.15358)


  Compressive phase retrieval is a popular variant of the standard compressive
sensing problem in which the measurements only contain magnitude information.
In this paper, motivated by recent advances in deep generative models, we
provide recovery guarantees with near-optimal sample complexity for phase
retrieval with generative priors. We first show that when using i.i.d. Gaussian
measurements and an $L$-Lipschitz continuous generative model with bounded
$k$-dimensional inputs, roughly $O(k \log L)$ samples suffice to guarantee that
any signal minimizing an amplitude-based empirical loss function is close to
the true signal. Attaining this sample complexity with a practical algorithm
remains a difficult challenge, and finding a good initialization for
gradient-based methods has been observed to pose a major bottleneck. To
partially address this, we further show that roughly $O(k \log L)$ samples
ensure sufficient closeness between the underlying signal and any {\em globally
optimal} solution to an optimization problem designed for spectral
initialization (though finding such a solution may still be challenging). We
also adapt this result to sparse phase retrieval, and show that $O(s \log n)$
samples are sufficient for a similar guarantee when the underlying signal is
$s$-sparse and $n$-dimensional, matching an information-theoretic lower bound.
While these guarantees do not directly correspond to a practical algorithm, we
propose a practical spectral initialization method motivated by our findings,
and experimentally observe performance gains over various existing spectral
initialization methods for sparse phase retrieval.

    

### [[2107.00753] An Investigation of the (In)effectiveness of Counterfactually Augmented Data](http://arxiv.org/abs/2107.00753)


  While pretrained language models achieve excellent performance on natural
language understanding benchmarks, they tend to rely on spurious correlations
and generalize poorly to out-of-distribution (OOD) data. Recent work has
explored using counterfactually-augmented data (CAD) -- data generated by
minimally perturbing examples to flip the ground-truth label -- to identify
robust features that are invariant under distribution shift. However, empirical
results using CAD for OOD generalization have been mixed. To explain this
discrepancy, we draw insights from a linear Gaussian model and demonstrate the
pitfalls of CAD. Specifically, we show that (a) while CAD is effective at
identifying robust features, it may prevent the model from learning unperturbed
robust features; and (b) CAD may exacerbate existing spurious correlations in
the data. On two crowdsourced CAD datasets, our results show that the lack of
perturbation diversity limits their effectiveness on OOD generalization,
calling for innovative crowdsourcing procedures to elicit diverse perturbation
of examples.

    

### [[2107.00758] The Spotlight: A General Method for Discovering Systematic Errors in Deep Learning Models](http://arxiv.org/abs/2107.00758)


  Supervised learning models often make systematic errors on rare subsets of
the data. When these subsets correspond to explicit labels in the data (e.g.,
gender, race) such poor performance can be identified straightforwardly. This
paper introduces a method for discovering systematic errors that do not
correspond to such explicitly labelled subgroups. The key idea is that similar
inputs tend to have similar representations in the final hidden layer of a
neural network. We leverage this structure by "shining a spotlight" on this
representation space to find contiguous regions where the model performs
poorly. We show that the spotlight surfaces semantically meaningful areas of
weakness in a wide variety of existing models spanning computer vision, NLP,
and recommender systems.

    

### [[2108.04228] Iterative Distillation for Better Uncertainty Estimates in Multitask Emotion Recognition](http://arxiv.org/abs/2108.04228)


  When recognizing emotions, subtle nuances in displays of emotion generate
ambiguity or uncertainty in emotion perception. Emotion uncertainty has been
previously interpreted as inter-rater disagreement among multiple annotators.
In this paper, we consider a more common and challenging scenario: modeling
emotion uncertainty when only single emotion labels are available. From a
Bayesian perspective, we propose to use deep ensembles to capture uncertainty
for multiple emotion descriptors, i.e., action units, discrete expression
labels and continuous descriptors. We further apply iterative
self-distillation. Iterative distillation over multiple generations
significantly improves performance in both emotion recognition and uncertainty
estimation. Our method generates single student models that provide accurate
estimates of uncertainty for in-domain samples and a student ensemble that can
detect out-of-domain samples. Our experiments on emotion recognition and
uncertainty estimation using the Aff-wild2 dataset demonstrate that our
algorithm gives more reliable uncertainty estimates than both Temperature
Scaling and Monte Carol Dropout.

    

### [[2109.05710] Robust Stability of Neural-Network Controlled Nonlinear Systems with Parametric Variability](http://arxiv.org/abs/2109.05710)


  Stability certification and identification of the stabilizable operating
region of a system are two important concerns to ensure its operational
safety/security and robustness. With the advent of machine-learning tools,
these issues are specially important for systems with machine-learned
components in the feedback loop. Here we develop a theory for stability and
stabilizability of a class of neural-network controlled nonlinear systems,
where the equilibria can drift when parametric changes occur. A Lyapunov based
convex stability certificate is developed and is further used to devise an
estimate for a local Lipschitz upper bound for a neural-network (NN) controller
and a corresponding operating domain on the state space, containing an
initialization set from where the closed-loop (CL) local asymptotic stability
of each system in the class is guaranteed under the same controller, while the
system trajectories remain confined to the operating domain. For computing such
a robust stabilizing NN controller, a stability guaranteed training (SGT)
algorithm is also proposed. The effectiveness of the proposed framework is
demonstrated using illustrative examples.

    

### [[2109.05941] Efficient Contrastive Learning via Novel Data Augmentation and Curriculum Learning](http://arxiv.org/abs/2109.05941)


  We introduce EfficientCL, a memory-efficient continual pretraining method
that applies contrastive learning with novel data augmentation and curriculum
learning. For data augmentation, we stack two types of operation sequentially:
cutoff and PCA jittering. While pretraining steps proceed, we apply curriculum
learning by incrementing the augmentation degree for each difficulty step.
After data augmentation is finished, contrastive learning is applied on
projected embeddings of original and augmented examples. When finetuned on GLUE
benchmark, our model outperforms baseline models, especially for sentence-level
tasks. Additionally, this improvement is capable with only 70% of computational
memory compared to the baseline model.

    

### [[2110.01303] Incremental Class Learning using Variational Autoencoders with Similarity Learning](http://arxiv.org/abs/2110.01303)


  Catastrophic forgetting in neural networks during incremental learning
remains a challenging problem. Previous research investigated catastrophic
forgetting in fully connected networks, with some earlier work exploring
activation functions and learning algorithms. Applications of neural networks
have been extended to include similarity learning. It is of significant
interest to understand how similarity learning loss functions would be affected
by catastrophic forgetting. Our research investigates catastrophic forgetting
for four well-known similarity-based loss functions during incremental class
learning. The loss functions are angular, contrastive, centre, and triplet
loss. Our results show that the rate of catastrophic forgetting is different
across loss functions on multiple datasets. The angular loss was least
affected, followed by contrastive, triplet loss, and centre loss with good
mining techniques. We implemented three existing incremental learning
techniques, iCaRL, EWC, and EBLL. We further proposed our novel technique using
VAEs to generate representation as exemplars that are passed through
intermediate layers of the network. Our method outperformed the three existing
techniques. We have shown that we do not require stored images as exemplars for
incremental learning with similarity learning. The generated representations
can help preserve regions of the embedding space used by prior knowledge so
that new knowledge will not ``overwrite'' prior knowledge.

    

### [[2110.01593] Generalized Kernel Thinning](http://arxiv.org/abs/2110.01593)


  The kernel thinning (KT) algorithm of Dwivedi and Mackey (2021) compresses a
probability distribution more effectively than independent sampling by
targeting a reproducing kernel Hilbert space (RKHS) and leveraging a less
smooth square-root kernel. Here we provide four improvements. First, we show
that KT applied directly to the target RKHS yields tighter, dimension-free
guarantees for any kernel, any distribution, and any fixed function in the
RKHS. Second, we show that, for analytic kernels like Gaussian, inverse
multiquadric, and sinc, target KT admits maximum mean discrepancy (MMD)
guarantees comparable to or better than those of square-root KT without making
explicit use of a square-root kernel. Third, we prove that KT with a fractional
power kernel yields better-than-Monte-Carlo MMD guarantees for non-smooth
kernels, like Laplace and Matérn, that do not have square-roots. Fourth, we
establish that KT applied to a sum of the target and power kernels (a procedure
we call KT+) simultaneously inherits the improved MMD guarantees of power KT
and the tighter individual function guarantees of target KT. In our experiments
with target KT and KT+, we witness significant improvements in integration
error even in $100$ dimensions and when compressing challenging differential
equation posteriors.

    

### [[2110.05430] Density-based interpretable hypercube region partitioning for mixed numeric and categorical data](http://arxiv.org/abs/2110.05430)


  Consider a structured dataset of features, such as $\{\textrm{SEX},
\textrm{INCOME}, \textrm{RACE}, \textrm{EXPERIENCE}\}$. A user may want to know
where in the feature space observations are concentrated, and where it is
sparse or empty. The existence of large sparse or empty regions can provide
domain knowledge of soft or hard feature constraints (e.g., what is the typical
income range, or that it may be unlikely to have a high income with few years
of work experience). Also, these can suggest to the user that machine learning
(ML) model predictions for data inputs in sparse or empty regions may be
unreliable.
An interpretable region is a hyper-rectangle, such as $\{\textrm{RACE}
\in\{\textrm{Black}, \textrm{White}\}\}\:\&$ $\{10 \leq \:\textrm{EXPERIENCE}
\:\leq 13\}$, containing all observations satisfying the constraints;
typically, such regions are defined by a small number of features. Our method
constructs an observation density-based partition of the observed feature space
in the dataset into such regions. It has a number of advantages over others in
that it works on features of mixed type (numeric or categorical) in the
original domain, and can separate out empty regions as well.
As can be seen from visualizations, the resulting partitions accord with
spatial groupings that a human eye might identify; the results should thus
extend to higher dimensions. We also show some applications of the partition to
other data analysis tasks, such as inferring about ML model error, measuring
high-dimensional density variability, and causal inference for treatment
effect. Many of these applications are made possible by the hyper-rectangular
form of the partition regions.

    

### [[2110.05454] Momentum Centering and Asynchronous Update for Adaptive Gradient Methods](http://arxiv.org/abs/2110.05454)


  We propose ACProp (Asynchronous-centering-Prop), an adaptive optimizer which
combines centering of second momentum and asynchronous update (e.g. for $t$-th
update, denominator uses information up to step $t-1$, while numerator uses
gradient at $t$-th step). ACProp has both strong theoretical properties and
empirical performance. With the example by Reddi et al. (2018), we show that
asynchronous optimizers (e.g. AdaShift, ACProp) have weaker convergence
condition than synchronous optimizers (e.g. Adam, RMSProp, AdaBelief); within
asynchronous optimizers, we show that centering of second momentum further
weakens the convergence condition. We demonstrate that ACProp has a convergence
rate of $O(\frac{1}{\sqrt{T}})$ for the stochastic non-convex case, which
matches the oracle rate and outperforms the $O(\frac{logT}{\sqrt{T}})$ rate of
RMSProp and Adam. We validate ACProp in extensive empirical studies: ACProp
outperforms both SGD and other adaptive optimizers in image classification with
CNN, and outperforms well-tuned adaptive optimizers in the training of various
GAN models, reinforcement learning and transformers. To sum up, ACProp has good
theoretical properties including weak convergence condition and optimal
convergence rate, and strong empirical performance including good
generalization like SGD and training stability like Adam. We provide the
implementation at this https URL.

    

### [[2102.10663] MedAug: Contrastive learning leveraging patient metadata improves representations for chest X-ray interpretation](http://arxiv.org/abs/2102.10663)


  Self-supervised contrastive learning between pairs of multiple views of the
same image has been shown to successfully leverage unlabeled data to produce
meaningful visual representations for both natural and medical images. However,
there has been limited work on determining how to select pairs for medical
images, where availability of patient metadata can be leveraged to improve
representations. In this work, we develop a method to select positive pairs
coming from views of possibly different images through the use of patient
metadata. We compare strategies for selecting positive pairs for chest X-ray
interpretation including requiring them to be from the same patient, imaging
study or laterality. We evaluate downstream task performance by fine-tuning the
linear layer on 1% of the labeled dataset for pleural effusion classification.
Our best performing positive pair selection strategy, which involves using
images from the same patient from the same study across all lateralities,
achieves a performance increase of 14.4% in mean AUC from the ImageNet
pretrained baseline. Our controlled experiments show that the keys to improving
downstream performance on disease classification are (1) using patient metadata
to appropriately create positive pairs from different images with the same
underlying pathologies, and (2) maximizing the number of different images used
in query pairing. In addition, we explore leveraging patient metadata to select
hard negative pairs for contrastive learning, but do not find improvement over
baselines that do not use metadata. Our method is broadly applicable to medical
image interpretation and allows flexibility for incorporating medical insights
in choosing pairs for contrastive learning.

    

### [[2110.08613] Enabling Large-Reach TLBs for High-Throughput Processors by Exploiting Memory Subregion Contiguity](http://arxiv.org/abs/2110.08613)


  Accelerators, like GPUs, have become a trend to deliver future performance
desire, and sharing the same virtual memory space between CPUs and GPUs is
increasingly adopted to simplify programming. However, address translation,
which is the key factor of virtual memory, is becoming the bottleneck of
performance for GPUs. In GPUs, a single TLB miss can stall hundreds of threads
due to the SIMT execute model, degrading performance dramatically. Through real
system analysis, we observe that the OS shows an advanced contiguity (e.g.,
hundreds of contiguous pages), and more large memory regions with advanced
contiguity tend to be allocated with the increase of working sets. Leveraging
the observation, we propose MESC to improve the translation efficiency for
GPUs. The key idea of MESC is to divide each large page frame (2MB size) in
virtual memory space into memory subregions with fixed size (i.e., 64 4KB
pages), and store the contiguity information of subregions and large page
frames in L2PTEs. With MESC, address translations of up to 512 pages can be
coalesced into single TLB entry, without the needs of changing memory
allocation policy (i.e., demand paging) and the support of large pages. In the
experimental results, MESC achieves 77.2% performance improvement and 76.4%
reduction in dynamic translation energy for translation-sensitive workloads.

    

### [[2011.10932] Copernicus: Characterizing the Performance Implications of Compression Formats Used in Sparse Workloads](http://arxiv.org/abs/2011.10932)


  Sparse matrices are the key ingredients of several application domains, from
scientific computation to machine learning. The primary challenge with sparse
matrices has been efficiently storing and transferring data, for which many
sparse formats have been proposed to significantly eliminate zero entries. Such
formats, essentially designed to optimize memory footprint, may not be as
successful in performing faster processing. In other words, although they allow
faster data transfer and improve memory bandwidth utilization -- the classic
challenge of sparse problems -- their decompression mechanism can potentially
create a computation bottleneck. Not only is this challenge not resolved, but
also it becomes more serious with the advent of domain-specific architectures
(DSAs), as they intend to more aggressively improve performance. The
performance implications of using various formats along with DSAs, however, has
not been extensively studied by prior work. To fill this gap of knowledge, we
characterize the impact of using seven frequently used sparse formats on
performance, based on a DSA for sparse matrix-vector multiplication (SpMV),
implemented on an FPGA using high-level synthesis (HLS) tools, a growing and
popular method for developing DSAs. Seeking a fair comparison, we tailor and
optimize the HLS implementation of decompression for each format. We thoroughly
explore diverse metrics, including decompression overhead, latency, balance
ratio, throughput, memory bandwidth utilization, resource utilization, and
power consumption, on a variety of real-world and synthetic sparse workloads.

    

### [[2110.08308] Adaptive and Fair Transformation for Recoverable Mutual Exclusion](http://arxiv.org/abs/2110.08308)


  Mutual exclusion is one of the most commonly used techniques to handle
contention in concurrent systems. Traditionally, mutual exclusion algorithms
have been designed under the assumption that a process does not fail while
acquiring/releasing a lock or while executing its critical section. However,
failures do occur in real life, potentially leaving the lock in an inconsistent
state. This gives rise to the problem of recoverable mutual exclusion (RME)
that involves designing a mutual exclusion (ME) algorithm that can tolerate
failures, while maintaining safety and liveness properties.
In this work, we present a framework that transforms any algorithm that
solves the RME problem into an algorithm that can also simultaneously adapt to
(1) the number of processes competing for the lock, as well as (2) the number
of failures that have occurred in the recent past, while maintaining the
correctness and performance properties of the underlying RME algorithm.
Additionally, the algorithm constructed as a result of this transformation adds
certain desirable properties like fairness (a variation of FCFS) and bounded
recovery. Assume that the worst-case RMR complexity of a critical section
request in the underlying RME algorithm is $R(n)$. Then, our framework yields
an RME algorithm for which the worst-case RMR complexity of a critical section
request is given by $\mathcal{O}(\min \{\ddot{c}, \sqrt{F+1}, R(n)\})$, where
$\ddot{c}$ denotes the point contention of the request and $F$ denotes the
number of failures in the recent past of the request.
We further extend our framework by presenting a novel memory reclamation
algorithm to bound the worst-case space complexity of the RME algorithm. The
memory reclamation techniques maintain the fairness, performance and
correctness properties of our transformation and is general enough to be
employed to bound the space of other RME algorithms.

    

### [[2110.08375] Least Squares on GPUs in Multiple Double Precision](http://arxiv.org/abs/2110.08375)


  This paper describes the application of the code generated by the CAMPARY
software to accelerate the solving of linear systems in the least squares sense
on Graphics Processing Units (GPUs), in double double, quad double, and octo
double precision. The goal is to use accelerators to offset the cost overhead
caused by multiple double precision arithmetic. For the blocked Householder QR
and the back substitution, of interest are those dimensions at which teraflop
performance is attained. The other interesting question is the cost overhead
factor that appears each time the precision is doubled.
Experimental results are reported on five different NVIDIA GPUs, with a
particular focus on the P100 and the V100, both capable of teraflop
performance. Thanks to the high Compute to Global Memory Access (CGMA) ratios
of multiple double arithmetic, teraflop performance is already attained running
the double double QR on 1,024-by-1,024 matrices, both on the P100 and the V100.
For the back substitution, the dimension of the upper triangular system must be
as high as 17,920 to reach one teraflops on the V100, in quad double precision,
and then taking only the times spent by the kernels into account. The lower
performance of the back substitution in small dimensions does not prevent
teraflop performance of the solver at dimension 1,024, as the time for the QR
decomposition dominates.
In doubling the precision from double double to quad double and from quad
double to octo double, the observed cost overhead factors are lower than the
factors predicted by the arithmetical operation counts. This observation
correlates with the increased performance for increased precision, which can
again be explained by the high CGMA ratios.

    

### [[2110.08592] Self-stabilizing Byzantine- and Intrusion-tolerant Consensus](http://arxiv.org/abs/2110.08592)


  One of the most celebrated problems of fault-tolerant distributed computing
is the consensus problem. It was shown to abstract a myriad of problems in
which processes have to agree on a single value. Consensus applications include
fundamental services for the environments of the Cloud or Blockchain. In such
challenging environments, malicious behavior is often modeled as adversarial
Byzantine faults. At OPODIS 2010, Mostéfaoui and Raynal, in short, MR,
presented a Byzantine- and intrusion-tolerant solution to consensus in which
the decided value cannot be a value proposed only by Byzantine processes. In
addition to this validity property, MR has optimal resilience since it can deal
with up to $t < n/3$ Byzantine processes, where $n$ is the number of processes.
We note that MR provides this multivalued consensus object (which accepts
proposals taken from a set with a finite number of values) assuming the
availability of a single Binary consensus object (which accepts proposals taken
from the set {0,1}).
This work, which focuses on multivalued consensus, aims at the design of an
even more robust solution than MR. Our proposal expands MR's fault-model with
self-stabilization, a vigorous notion of fault-tolerance. In addition to
tolerating Byzantine and communication failures, self-stabilizing systems can
automatically recover after the occurrence of \emph{arbitrary
transient-faults}. These faults represent any violation of the assumptions
according to which the system was designed to operate (provided that the
algorithm code remains intact). We propose, to the best of our knowledge, the
first self-stabilizing solution for intrusion-tolerant multivalued consensus
for asynchronous message-passing systems prone to Byzantine failures.

    

### [[2007.08596] OptChain: Optimal Transactions Placement for Scalable Blockchain Sharding](http://arxiv.org/abs/2007.08596)


  A major challenge in blockchain sharding protocols is that more than 95%
transactions are cross-shard. Not only those cross-shard transactions degrade
the system throughput but also double the confirmation time, and exhaust an
already scarce network bandwidth. Are cross-shard transactions imminent for
sharding schemes? In this paper, we propose a new sharding paradigm, called
OptChain, in which cross-shard transactions are minimized, resulting in almost
twice faster confirmation time and throughput. By treating transactions as a
stream of nodes in an online graph, OptChain utilizes a lightweight and
on-the-fly transaction placement method to group both related and soon-related
transactions into the same shards. At the same time, OptChain maintains a
temporal balance among shards to guarantee the high parallelism. Our
comprehensive and large-scale simulation using Oversim P2P library confirms a
significant boost in performance with up to 10 folds reduction in cross-shard
transactions, more than twice reduction in confirmation time, and 50% increase
in throughput. When combined with Omniledger sharding protocol, OptChain
delivers a 6000 transactions per second throughput with 10.5s confirmation
time.

    

### [[2102.01009] Infrastructure Resilience Curves: Performance Measures and Summary Metrics](http://arxiv.org/abs/2102.01009)


  Resilience curves are used to communicate quantitative and qualitative
aspects of system behavior and resilience to stakeholders of critical
infrastructure. Generally, these curves illustrate the evolution of system
performance before, during, and after a disruption. As simple as these curves
may appear, the literature contains underexplored nuance when defining
"performance" and comparing curves with summary metrics. Through a critical
review of 273 publications, this manuscript aims to define a common vocabulary
for practitioners and researchers that will improve the use of resilience
curves as a tool for assessing and designing resilient infrastructure. This
vocabulary includes a taxonomy of resilience curve performance measures as well
as a taxonomy of summary metrics. In addition, this review synthesizes a
framework for examining assumptions of resilience analysis that are often
implicit or unexamined in the practice and literature. From this vocabulary and
framework comes recommendations including broader adoption of productivity
measures; additional research on endogenous performance targets and thresholds;
deliberate consideration of curve milestones when defining summary metrics; and
cautionary fundamental flaws that may arise when condensing an ensemble of
resilience curves into an "expected" trajectory.

    

### [[2104.03187] A Preliminary Proposal for an Analytical Model for Evaluating the Impact on Performance of Data Access Patterns in Transaction Execution](http://arxiv.org/abs/2104.03187)


  We present a preliminary proposal for an analytical model for evaluating the
impact on performance of data access patterns in concurrent transaction
execution. We consider the case of concurrency control protocols that use
locking to ensure isolation in the execution of transactions. We analyse
scenarios where transactions access one or more sets of data items in the same
order or in different order.

    

### [[2105.07123] Byzantine-Resilient Population Protocols](http://arxiv.org/abs/2105.07123)


  Population protocols model information spreading in networks where pairwise
node exchanges are determined by an external random scheduler. Most of the
population protocols in the literature assume that the participating $n$ nodes
are honest. Such an assumption may not be, however, accurate for large-scale
systems of small devices. Hence, in this work, we study population protocols in
a setting where up to $f$ nodes can be Byzantine. We examine the majority
(binary) consensus problem against different levels of adversary strengths,
ranging from the Full adversary that has complete knowledge of all the node
states to the Weak adversary that has only knowledge about which exchanges take
place. We also take into account Dynamic vs Static node corruption by the
adversary. We give lower bounds that require any algorithm solving the majority
consensus to have initial difference $d = \Omega(f + 1)$ for the tally between
the two proposed values, which holds for both the Full Static and Weak Dynamic
adversaries. We then present an algorithm that solves the majority consensus
problem and tolerates $f \leq n / c$ Byzantine nodes, for some constant $c>0$,
with $d = \Omega(f + \sqrt{n \log n})$ and $O(\log^3 n)$ parallel time steps,
using $O(\log n)$ states per node. We also give an alternative algorithm with
$d = \Omega(\min\{f \log^2 n + 1,n\})$. Moreover, we combine both algorithms
into one using random coins. The only other known previous work on
Byzantine-resilient population protocols tolerates up to $f = o(\sqrt n)$
faulty nodes and was analyzed against a Static adversary; hence, our protocols
significantly improve fault-tolerance by an $\omega(\sqrt n)$ factor and all of
them work correctly against a stronger Dynamic adversary.

    

### [[2109.13216] Extending Lattice linearity for Self-Stabilizing Algorithms](http://arxiv.org/abs/2109.13216)


  In this article, we focus on extending the notion of lattice linearity to
self-stabilizing programs. Lattice linearity allows a node to execute its
actions with old information about the state of other nodes and still preserve
correctness. It increases the concurrency of the program execution by
eliminating the need for synchronization among its nodes. The extension --
denoted as eventually lattice linear algorithms -- is performed with an example
of the service-demand based minimal dominating set (SDDS) problem, which is a
generalization of the dominating set problem; it converges in $2n$ moves.
Subsequently, we also show that the same approach could be used in various
other problems including minimal vertex cover, maximal independent set and
graph coloring.

    

### [[2110.08300] When Combating Hype, Proceed with Caution](http://arxiv.org/abs/2110.08300)


  In an effort to avoid reinforcing widespread hype about the capabilities of
state-of-the-art language technology, researchers have developed practices in
framing and citation that serve to deemphasize the field's successes. Though
well-meaning, these practices often yield misleading or even false claims about
the limits of our best technology. This is a problem, and it may be more
serious than it looks: It limits our ability to mitigate short-term harms from
NLP deployments and it limits our ability to prepare for the potentially
enormous impacts of more distant future advances. This paper urges researchers
to be careful about these claims and suggests some research directions and
communication strategies that will make it easier to avoid or rebut them.

    

### [[2110.08318] Dynamic probabilistic logic models for effective abstractions in RL](http://arxiv.org/abs/2110.08318)


  State abstraction enables sample-efficient learning and better task transfer
in complex reinforcement learning environments. Recently, we proposed RePReL
(Kokel et al. 2021), a hierarchical framework that leverages a relational
planner to provide useful state abstractions for learning. We present a brief
overview of this framework and the use of a dynamic probabilistic logic model
to design these state abstractions. Our experiments show that RePReL not only
achieves better performance and efficient learning on the task at hand but also
demonstrates better generalization to unseen tasks.

    

### [[2110.08343] HyperSeed: Unsupervised Learning with Vector Symbolic Architectures](http://arxiv.org/abs/2110.08343)


  Motivated by recent innovations in biologically-inspired neuromorphic
hardware, this paper presents a novel unsupervised machine learning approach
named Hyperseed that leverages Vector Symbolic Architectures (VSA) for fast
learning a topology preserving feature map of unlabelled data. It relies on two
major capabilities of VSAs: the binding operation and computing in
superposition. In this paper, we introduce the algorithmic part of Hyperseed
expressed within Fourier Holographic Reduced Representations VSA model, which
is specifically suited for implementation on spiking neuromorphic hardware. The
two distinctive novelties of the Hyperseed algorithm are: 1) Learning from only
few input data samples and 2) A learning rule based on a single vector
operation. These properties are demonstrated on synthetic datasets as well as
on illustrative benchmark use-cases, IRIS classification and a language
identification task using n-gram statistics.

    

### [[2110.08345] Towards Transparent Interactive Semantic Parsing via Step-by-Step Correction](http://arxiv.org/abs/2110.08345)


  Existing studies on semantic parsing focus primarily on mapping a
natural-language utterance to a corresponding logical form in one turn.
However, because natural language can contain a great deal of ambiguity and
variability, this is a difficult challenge. In this work, we investigate an
interactive semantic parsing framework that explains the predicted logical form
step by step in natural language and enables the user to make corrections
through natural-language feedback for individual steps. We focus on question
answering over knowledge bases (KBQA) as an instantiation of our framework,
aiming to increase the transparency of the parsing process and help the user
appropriately trust the final answer. To do so, we construct INSPIRED, a
crowdsourced dialogue dataset derived from the ComplexWebQuestions dataset. Our
experiments show that the interactive framework with human feedback has the
potential to greatly improve overall parse accuracy. Furthermore, we develop a
pipeline for dialogue simulation to evaluate our framework w.r.t. a variety of
state-of-the-art KBQA models without involving further crowdsourcing effort.
The results demonstrate that our interactive semantic parsing framework
promises to be effective across such models.

    

### [[2110.08377] Starkit: RoboCup Humanoid KidSize 2021 Worldwide Champion Team Paper](http://arxiv.org/abs/2110.08377)


  This article is devoted to the features that were under development between
RoboCup 2019 Sydney and RoboCup 2021 Worldwide. These features include
vision-related matters, such as detection and localization, mechanical and
algorithmic novelties. Since the competition was held virtually, the
simulation-specific features are also considered in the article. We give an
overview of the approaches that were tried out along with the analysis of their
preconditions, perspectives and the evaluation of their performance.

    

### [[2110.08402] sbp-env: A Python Package for Sampling-based Motion Planner and Samplers](http://arxiv.org/abs/2110.08402)


  Sampling-based motion planners' testing environment (sbp-env) is a full
feature framework to quickly test different sampling-based algorithms for
motion planning. sbp-env focuses on the flexibility of tinkering with different
aspects of the framework, and had divided the main planning components into two
categories (i) samplers and (ii) planners.
The focus of motion planning research had been mainly on (i) improving the
sampling efficiency (with methods such as heuristic or learned distribution)
and (ii) the algorithmic aspect of the planner using different routines to
build a connected graph. Therefore, by separating the two components one can
quickly swap out different components to test novel ideas.

    

### [[2110.08417] Open Domain Question Answering over Virtual Documents: A Unified Approach for Data and Text](http://arxiv.org/abs/2110.08417)


  Due to its potential for a universal interface over both data and text,
data-to-text generation is becoming increasingly popular recently. However, few
previous work has focused on its application to downstream tasks, e.g. using
the converted data for grounding or reasoning. In this work, we aim to bridge
this gap and use the data-to-text method as a means for encoding structured
knowledge for knowledge-intensive applications, i.e. open-domain question
answering (QA). Specifically, we propose a verbalizer-retriever-reader
framework for open-domain QA over data and text where verbalized tables from
Wikipedia and triples from Wikidata are used as augmented knowledge sources. We
show that our Unified Data and Text QA, UDT-QA, can effectively benefit from
the expanded knowledge index, leading to large gains over text-only baselines.
Notably, our approach sets the single-model state-of-the-art on Natural
Questions. Furthermore, our analyses indicate that verbalized knowledge is
preferred for answer reasoning for both adapted and hot-swap settings.

    

### [[2110.08423] Finding Backdoors to Integer Programs: A Monte Carlo Tree Search Framework](http://arxiv.org/abs/2110.08423)


  In Mixed Integer Linear Programming (MIP), a (strong) backdoor is a "small"
subset of an instance's integer variables with the following property: in a
branch-and-bound procedure, the instance can be solved to global optimality by
branching only on the variables in the backdoor. Constructing datasets of
pre-computed backdoors for widely used MIP benchmark sets or particular problem
families can enable new questions around novel structural properties of a MIP,
or explain why a problem that is hard in theory can be solved efficiently in
practice. Existing algorithms for finding backdoors rely on sampling candidate
variable subsets in various ways, an approach which has demonstrated the
existence of backdoors for some instances from MIPLIB2003 and MIPLIB2010.
However, these algorithms fall short of consistently succeeding at the task due
to an imbalance between exploration and exploitation. We propose BaMCTS, a
Monte Carlo Tree Search framework for finding backdoors to MIPs. Extensive
algorithmic engineering, hybridization with traditional MIP concepts, and close
integration with the CPLEX solver have enabled our method to outperform
baselines on MIPLIB2017 instances, finding backdoors more frequently and more
efficiently.

    

### [[2110.08429] TorchEsegeta: Framework for Interpretability and Explainability of Image-based Deep Learning Models](http://arxiv.org/abs/2110.08429)


  Clinicians are often very sceptical about applying automatic image processing
approaches, especially deep learning based methods, in practice. One main
reason for this is the black-box nature of these approaches and the inherent
problem of missing insights of the automatically derived decisions. In order to
increase trust in these methods, this paper presents approaches that help to
interpret and explain the results of deep learning algorithms by depicting the
anatomical areas which influence the decision of the algorithm most. Moreover,
this research presents a unified framework, TorchEsegeta, for applying various
interpretability and explainability techniques for deep learning models and
generate visual interpretations and explanations for clinicians to corroborate
their clinical findings. In addition, this will aid in gaining confidence in
such methods. The framework builds on existing interpretability and
explainability techniques that are currently focusing on classification models,
extending them to segmentation tasks. In addition, these methods have been
adapted to 3D models for volumetric analysis. The proposed framework provides
methods to quantitatively compare visual explanations using infidelity and
sensitivity metrics. This framework can be used by data scientists to perform
post-hoc interpretations and explanations of their models, develop more
explainable tools and present the findings to clinicians to increase their
faith in such models. The proposed framework was evaluated based on a use case
scenario of vessel segmentation models trained on Time-of-fight (TOF) Magnetic
Resonance Angiogram (MRA) images of the human brain. Quantitative and
qualitative results of a comparative study of different models and
interpretability methods are presented. Furthermore, this paper provides an
extensive overview of several existing interpretability and explainability
methods.

    

### [[2110.08446] Self-Annotated Training for Controllable Image Captioning](http://arxiv.org/abs/2110.08446)


  The Controllable Image Captioning (CIC) task aims to generate captions
conditioned on designated control signals. In this paper, we improve CIC from
two aspects: 1) Existing reinforcement training methods are not applicable to
structure-related CIC models due to the fact that the accuracy-based reward
focuses mainly on contents rather than semantic structures. The lack of
reinforcement training prevents the model from generating more accurate and
controllable sentences. To solve the problem above, we propose a novel
reinforcement training method for structure-related CIC models: Self-Annotated
Training (SAT), where a recursive sampling mechanism (RSM) is designed to force
the input control signal to match the actual output sentence. Extensive
experiments conducted on MSCOCO show that our SAT method improves C-Transformer
(XE) on CIDEr-D score from 118.6 to 130.1 in the length-control task and from
132.2 to 142.7 in the tense-control task, while maintaining more than 99$\%$
matching accuracy with the control signal. 2) We introduce a new control
signal: sentence quality. Equipped with it, CIC models are able to generate
captions of different quality levels as needed. Experiments show that without
additional information of ground truth captions, models controlled by the
highest level of sentence quality perform much better in accuracy than baseline
models.

    

### [[2110.08467] Improving Compositional Generalization with Self-Training for Data-to-Text Generation](http://arxiv.org/abs/2110.08467)


  Data-to-text generation focuses on generating fluent natural language
responses from structured semantic representations. Such representations are
compositional, allowing for the combination of atomic meaning schemata in
various ways to express the rich semantics in natural language. Recently,
pretrained language models (LMs) have achieved impressive results on
data-to-text tasks, though it remains unclear the extent to which these LMs
generalize to new semantic representations. In this work, we systematically
study the compositional generalization of current state-of-the-art generation
models in data-to-text tasks. By simulating structural shifts in the
compositional Weather dataset, we show that T5 models fail to generalize to
unseen structures. Next, we show that template-based input representations
greatly improve the model performance and model scale does not trivially solve
the lack of generalization. To further improve the model's performance, we
propose an approach based on self-training using finetuned BLEURT for
pseudo-response selection. Extensive experiments on the few-shot Weather and
multi-domain SGD datasets demonstrate strong gains of our method.

    

### [[2110.08480] Learning Cooperation and Online Planning Through Simulation and Graph Convolutional Network](http://arxiv.org/abs/2110.08480)


  Multi-agent Markov Decision Process (MMDP) has been an effective way of
modelling sequential decision making algorithms for multi-agent cooperative
environments. A number of algorithms based on centralized and decentralized
planning have been developed in this domain. However, dynamically changing
environment, coupled with exponential size of the state and joint action space,
make it difficult for these algorithms to provide both efficiency and
scalability. Recently, Centralized planning algorithm FV-MCTS-MP and
decentralized planning algorithm \textit{Alternate maximization with
Behavioural Cloning} (ABC) have achieved notable performance in solving MMDPs.
However, they are not capable of adapting to dynamically changing environments
and accounting for the lack of communication among agents, respectively.
Against this background, we introduce a simulation based online planning
algorithm, that we call SiCLOP, for multi-agent cooperative environments.
Specifically, SiCLOP tailors Monte Carlo Tree Search (MCTS) and uses
Coordination Graph (CG) and Graph Neural Network (GCN) to learn cooperation and
provides real time solution of a MMDP problem. It also improves scalability
through an effective pruning of action space. Additionally, unlike FV-MCTS-MP
and ABC, SiCLOP supports transfer learning, which enables learned agents to
operate in different environments. We also provide theoretical discussion about
the convergence property of our algorithm within the context of multi-agent
settings. Finally, our extensive empirical results show that SiCLOP
significantly outperforms the state-of-the-art online planning algorithms.

    

### [[2110.08511] What can we learn from universal Turing machines?](http://arxiv.org/abs/2110.08511)


  In the present paper, we construct what we call a pedagogical universal
Turing machine. We try to understand which comparisons with biological
phenomena can be deduced from its encoding and from its working.

    

### [[2110.08512] AugmentedCode: Examining the Effects of Natural Language Resources in Code Retrieval Models](http://arxiv.org/abs/2110.08512)


  Code retrieval is allowing software engineers to search codes through a
natural language query, which relies on both natural language processing and
software engineering techniques. There have been several attempts on code
retrieval from searching snippet codes to function codes. In this paper, we
introduce Augmented Code (AugmentedCode) retrieval which takes advantage of
existing information within the code and constructs augmented programming
language to improve the code retrieval models' performance. We curated a large
corpus of Python and showcased the the framework and the results of augmented
programming language which outperforms on CodeSearchNet and CodeBERT with a
Mean Reciprocal Rank (MRR) of 0.73 and 0.96, respectively. The outperformed
fine-tuned augmented code retrieval model is published in HuggingFace at
this https URL and a demonstration video is available
at: this https URL .

    

### [[2110.08578] Visual-aware Attention Dual-stream Decoder for Video Captioning](http://arxiv.org/abs/2110.08578)


  Video captioning is a challenging task that captures different visual parts
and describes them in sentences, for it requires visual and linguistic
coherence. The attention mechanism in the current video captioning method
learns to assign weight to each frame, promoting the decoder dynamically. This
may not explicitly model the correlation and the temporal coherence of the
visual features extracted in the sequence this http URL generate semantically
coherent sentences, we propose a new Visual-aware Attention (VA) model, which
concatenates dynamic changes of temporal sequence frames with the words at the
previous moment, as the input of attention mechanism to extract sequence
this http URL addition, the prevalent approaches widely use the teacher-forcing
(TF) learning during training, where the next token is generated conditioned on
the previous ground-truth tokens. The semantic information in the previously
generated tokens is lost. Therefore, we design a self-forcing (SF) stream that
takes the semantic information in the probability distribution of the previous
token as input to enhance the current token.The Dual-stream Decoder (DD)
architecture unifies the TF and SF streams, generating sentences to promote the
annotated captioning for both streams.Meanwhile, with the Dual-stream Decoder
utilized, the exposure bias problem is alleviated, caused by the discrepancy
between the training and testing in the TF learning.The effectiveness of the
proposed Visual-aware Attention Dual-stream Decoder (VADD) is demonstrated
through the result of experimental studies on Microsoft video description
(MSVD) corpus and MSR-Video to text (MSR-VTT) datasets.

    

### [[2110.08637] Conceptual Modeling and Artificial Intelligence: Mutual Benefits from Complementary Worlds](http://arxiv.org/abs/2110.08637)


  Conceptual modeling (CM) applies abstraction to reduce the complexity of a
system under study (e.g., an excerpt of reality). As a result of the conceptual
modeling process a human interpretable, formalized representation (i.e., a
conceptual model) is derived which enables understanding and communication
among humans, and processing by machines. Artificial Intelligence (AI)
algorithms are also applied to complex realities (regularly represented by vast
amounts of data) to identify patterns or to classify entities in the data.
Aside from the commonalities of both approaches, a significant difference can
be observed by looking at the results. While conceptual models are
comprehensible, reproducible, and explicit knowledge representations, AI
techniques are capable of efficiently deriving an output from a given input
while acting as a black box. AI solutions often lack comprehensiveness and
reproducibility. Even the developers of AI systems can't explain why a certain
output is derived. In the Conceptual Modeling meets Artificial Intelligence
(CMAI) workshop, we are interested in tackling the intersection of the two,
thus far, mostly isolated approached disciplines of CM and AI. The workshop
embraces the assumption, that manifold mutual benefits can be realized by i)
investigating what Conceptual Modeling (CM) can contribute to AI, and ii) the
other way around, what Artificial Intelligence (AI) can contribute to CM.

    

### [[2110.08653] Learning UI Navigation through Demonstrations composed of Macro Actions](http://arxiv.org/abs/2110.08653)


  We have developed a framework to reliably build agents capable of UI
navigation. The state space is simplified from raw-pixels to a set of UI
elements extracted from screen understanding, such as OCR and icon detection.
The action space is restricted to the UI elements plus a few global actions.
Actions can be customized for tasks and each action is a sequence of basic
operations conditioned on status checks. With such a design, we are able to
train DQfD and BC agents with a small number of demonstration episodes. We
propose demo augmentation that significantly reduces the required number of
human demonstrations. We made a customization of DQfD to allow demos collected
on screenshots to facilitate the demo coverage of rare cases. Demos are only
collected for the failed cases during the evaluation of the previous version of
the agent. With 10s of iterations looping over evaluation, demo collection, and
training, the agent reaches a 98.7\% success rate on the search task in an
environment of 80+ apps and websites where initial states and viewing
parameters are randomized.

    

### [[2110.08664] Finding Critical Scenarios for Automated Driving Systems: A Systematic Literature Review](http://arxiv.org/abs/2110.08664)


  Scenario-based approaches have been receiving a huge amount of attention in
research and engineering of automated driving systems. Due to the complexity
and uncertainty of the driving environment, and the complexity of the driving
task itself, the number of possible driving scenarios that an ADS or ADAS may
encounter is virtually infinite. Therefore it is essential to be able to reason
about the identification of scenarios and in particular critical ones that may
impose unacceptable risk if not considered. Critical scenarios are particularly
important to support design, verification and validation efforts, and as a
basis for a safety case. In this paper, we present the results of a systematic
literature review in the context of autonomous driving. The main contributions
are: (i) introducing a comprehensive taxonomy for critical scenario
identification methods; (ii) giving an overview of the state-of-the-art
research based on the taxonomy encompassing 86 papers between 2017 and 2020;
and (iii) identifying open issues and directions for further research. The
provided taxonomy comprises three main perspectives encompassing the problem
definition (the why), the solution (the methods to derive scenarios), and the
assessment of the established scenarios. In addition, we discuss open research
issues considering the perspectives of coverage, practicability, and scenario
space explosion.

    

### [[2110.08671] Blockchain and Federated Edge Learning for Privacy-Preserving Mobile Crowdsensing](http://arxiv.org/abs/2110.08671)


  Mobile crowdsensing (MCS) counting on the mobility of massive workers helps
the requestor accomplish various sensing tasks with more flexibility and lower
cost. However, for the conventional MCS, the large consumption of communication
resources for raw data transmission and high requirements on data storage and
computing capability hinder potential requestors with limited resources from
using MCS. To facilitate the widespread application of MCS, we propose a novel
MCS learning framework leveraging on blockchain technology and the new concept
of edge intelligence based on federated learning (FL), which involves four
major entities, including requestors, blockchain, edge servers and mobile
devices as workers. Even though there exist several studies on blockchain-based
MCS and blockchain-based FL, they cannot solve the essential challenges of MCS
with respect to accommodating resource-constrained requestors or deal with the
privacy concerns brought by the involvement of requestors and workers in the
learning process. To fill the gaps, four main procedures, i.e., task
publication, data sensing and submission, learning to return final results, and
payment settlement and allocation, are designed to address major challenges
brought by both internal and external threats, such as malicious edge servers
and dishonest requestors. Specifically, a mechanism design based data
submission rule is proposed to guarantee the data privacy of mobile devices
being truthfully preserved at edge servers; consortium blockchain based FL is
elaborated to secure the distributed learning process; and a
cooperation-enforcing control strategy is devised to elicit full payment from
the requestor. Extensive simulations are carried out to evaluate the
performance of our designed schemes.

    

### [[2002.00223] Dialogue-Based Simulation For Cultural Awareness Training](http://arxiv.org/abs/2002.00223)


  Existing simulations designed for cultural and interpersonal skill training
rely on pre-defined responses with a menu option selection interface. Using a
multiple-choice interface and restricting trainees' responses may limit the
trainees' ability to apply the lessons in real life situations. This systems
also uses a simplistic evaluation model, where trainees' selected options are
marked as either correct or incorrect. This model may not capture sufficient
information that could drive an adaptive feedback mechanism to improve
trainees' cultural awareness. This paper describes the design of a
dialogue-based simulation for cultural awareness training. The simulation,
built around a disaster management scenario involving a joint coalition between
the US and the Chinese armies. Trainees were able to engage in realistic
dialogue with the Chinese agent. Their responses, at different points, get
evaluated by different multi-label classification models. Based on training on
our dataset, the models score the trainees' responses for cultural awareness in
the Chinese culture. Trainees also get feedback that informs the cultural
appropriateness of their responses. The result of this work showed the
following; i) A feature-based evaluation model improves the design, modeling
and computation of dialogue-based training simulation systems; ii) Output from
current automatic speech recognition (ASR) systems gave comparable end results
compared with the output from manual transcription; iii) A multi-label
classification model trained as a cultural expert gave results which were
comparable with scores assigned by human annotators.

    

### [[2010.10216] Simulated Chats for Building Dialog Systems: Learning to Generate Conversations from Instructions](http://arxiv.org/abs/2010.10216)


  Popular dialog datasets such as MultiWOZ are created by providing crowd
workers an instruction, expressed in natural language, that describes the task
to be accomplished. Crowd workers play the role of a user and an agent to
generate dialogs to accomplish tasks involving booking restaurant tables,
calling a taxi etc. In this paper, we present a data creation strategy that
uses the pre-trained language model, GPT2, to simulate the interaction between
crowd workers by creating a user bot and an agent bot. We train the simulators
using a smaller percentage of actual crowd-generated conversations and their
corresponding instructions. We demonstrate that by using the simulated data, we
achieve significant improvements in low-resource settings on two publicly
available datasets - the MultiWOZ dataset and the Persona chat dataset.

    

### [[2011.04569] Signal-Guided Source Separation](http://arxiv.org/abs/2011.04569)


  Informed speaker extraction aims to extract a target speech signal from a
mixture of sources given prior knowledge about the desired speaker. Recent deep
learning-based methods leverage a speaker discriminative model that maps a
reference snippet uttered by the target speaker into a single embedding vector
that encapsulates the characteristics of the target speaker. However, such
modeling deliberately neglects the time-varying properties of the reference
signal. In this work, we assume that a reference signal is available that is
temporally correlated with the target signal. To take this correlation into
account, we propose a time-varying source discriminative model that captures
the temporal dynamics of the reference signal. We also show that existing
methods and the proposed method can be generalized to non-speech sources as
well. Experimental results demonstrate that the proposed method significantly
improves the extraction performance when applied in an acoustic echo reduction
scenario.

    

### [[2011.07010] Monitoring and Diagnosability of Perception Systems](http://arxiv.org/abs/2011.07010)


  Perception is a critical component of high-integrity applications of robotics
and autonomous systems, such as self-driving vehicles. In these applications,
failure of perception systems may put human life at risk, and a broad adoption
of these technologies requires the development of methodologies to guarantee
and monitor safe operation. Despite the paramount importance of perception
systems, currently there is no formal approach for system-level monitoring. In
this work, we propose a mathematical model for runtime monitoring and fault
detection and identification in perception systems. Towards this goal, we draw
connections with the literature on diagnosability in multiprocessor systems,
and generalize it to account for modules with heterogeneous outputs that
interact over time. The resulting temporal diagnostic graphs (i) provide a
framework to reason over the consistency of perception outputs -- across
modules and over time -- thus enabling fault detection, (ii) allow us to
establish formal guarantees on the maximum number of faults that can be
uniquely identified in a given perception system, and (iii) enable the design
of efficient algorithms for fault identification. We demonstrate our monitoring
system, dubbed PerSyS, in realistic simulations using the LGSVL self-driving
simulator and the Apollo Auto autonomy software stack, and show that PerSyS is
able to detect failures in challenging scenarios (including scenarios that have
caused self-driving car accidents in recent years), and is able to correctly
identify faults while entailing a minimal computation overhead (< 5 ms on a
single-core CPU).

    

### [[2011.12979] mask-Net: Learning Context Aware Invariant Features using Adversarial Forgetting (Student Abstract)](http://arxiv.org/abs/2011.12979)


  Training a robust system, e.g.,Speech to Text (STT), requires large datasets.
Variability present in the dataset such as unwanted nuisances and biases are
the reason for the need of large datasets to learn general representations. In
this work, we propose a novel approach to induce invariance using adversarial
forgetting (AF). Our initial experiments on learning invariant features such as
accent on the STT task achieve better generalizations in terms of word error
rate (WER) compared to the traditional models. We observe an absolute
improvement of 2.2% and 1.3% on out-of-distribution and in-distribution test
sets, respectively.

    

### [[2012.08105] Schema Extraction on Semi-structured Data](http://arxiv.org/abs/2012.08105)


  With the continuous development of NoSQL databases, more and more developers
choose to use semi-structured data for development and data management, which
puts forward requirements for schema management of semi-structured data stored
in NoSQL databases. Schema extraction plays an important role in understanding
schemas, optimizing queries, and validating data consistency. Therefore, in
this survey we investigate structural methods based on tree and graph and
statistical methods based on distributed architecture and machine learning to
extract schemas. The schemas obtained by the structural methods are more
interpretable, and the statistical methods have better applicability and
generalization ability. Moreover, we also investigate tools and systems for
schemas extraction. Schema extraction tools are mainly used for spark or NoSQL
databases, and are suitable for small datasets or simple application
environments. The system mainly focuses on the extraction and management of
schemas in large data sets and complex application scenarios. Furthermore, we
also compare these techniques to facilitate data managers' choice.

    

### [[2104.07644] ExplaGraphs: An Explanation Graph Generation Task for Structured Commonsense Reasoning](http://arxiv.org/abs/2104.07644)


  Recent commonsense-reasoning tasks are typically discriminative in nature,
where a model answers a multiple-choice question for a certain context.
Discriminative tasks are limiting because they fail to adequately evaluate the
model's ability to reason and explain predictions with underlying commonsense
knowledge. They also allow such models to use reasoning shortcuts and not be
"right for the right reasons". In this work, we present ExplaGraphs, a new
generative and structured commonsense-reasoning task (and an associated
dataset) of explanation graph generation for stance prediction. Specifically,
given a belief and an argument, a model has to predict if the argument supports
or counters the belief and also generate a commonsense-augmented graph that
serves as non-trivial, complete, and unambiguous explanation for the predicted
stance. We collect explanation graphs through a novel Create-Verify-And-Refine
graph collection framework that improves the graph quality (up to 90%) via
multiple rounds of verification and refinement. A significant 79% of our graphs
contain external commonsense nodes with diverse structures and reasoning
depths. Next, we propose a multi-level evaluation framework, consisting of
automatic metrics and human evaluation, that check for the structural and
semantic correctness of the generated graphs and their degree of match with
ground-truth graphs. Finally, we present several structured,
commonsense-augmented, and text generation models as strong starting points for
this explanation graph generation task, and observe that there is a large gap
with human performance, thereby encouraging future work for this new
challenging task. ExplaGraphs will be publicly available at
this https URL.

    

### [[2109.09390] Learning to Improve Representations by Communicating About Perspectives](http://arxiv.org/abs/2109.09390)


  Despite its rise as a prominent solution to the data inefficiency of today's
machine learning models, self-supervised learning has yet to be studied from a
purely multi agent perspective. In this work, we propose that aligning internal
subjective representations, which naturally arise in a multi-agent setup where
agents receive partial observations of the same underlying environmental state,
can lead to more data-efficient representations. We propose that multi-agent
environments, where agents do not have access to the observations of others but
can communicate within a limited range, guarantees a common context that can be
leveraged in individual representation learning. The reason is that subjective
observations necessarily refer to the same subset of the underlying
environmental states and that communication about these states can freely offer
a supervised signal. To highlight the importance of communication, we refer to
our setting as socially supervised representation learning. We present a
minimal architecture comprised of a population of autoencoders, where we define
loss functions, capturing different aspects of effective communication, and
examine their effect on the learned representations. We show that our proposed
architecture allows the emergence of aligned representations. The subjectivity
introduced by presenting agents with distinct perspectives of the environment
state contributes to learning abstract representations that outperform those
learned by a single autoencoder and a population of autoencoders, presented
with identical perspectives of the environment state. Altogether, our results
demonstrate how communication from subjective perspectives can lead to the
acquisition of more abstract representations in multi-agent systems, opening
promising perspectives for future research at the intersection of
representation learning and emergent communication.

    

### [[2101.12146] Online Tensor Completion Based Prediction For Wireless Edge Caching](http://arxiv.org/abs/2101.12146)


  Wireless edge caching is a popular strategy to avoid backhaul congestion in
the next generation networks, where the content is cached in advance at base
stations to serve redundant requests during peak congestion periods. In the
edge caching data, the missing observations are inevitable due to dynamic
selective popularity. Among the completion methods, the tensor-based models
have been shown to be the most advantageous for missing data imputation. Also,
since the observations are correlated across time, files, and base stations, in
this paper, we formulate the cooperative caching with recommendations as a
fourth-order tensor completion and prediction problem. Since the content
library can be large leading to a large dimension tensor, we modify the latent
norm-based Frank-Wolfe (FW) algorithm with towards a much lower time complexity
using multi-rank updates, rather than rank-1 updates in literature. This
significantly lower time computational overhead leads in developing an online
caching algorithm. With MovieLens dataset, simulations show lower
reconstruction errors for the proposed algorithm as compared to that of the
recent FW algorithm, albeit with lower computation overhead. It is also
demonstrated that the completed tensor improves normalized cache hit rates for
linear prediction schemes.

    

### [[2104.08050] Age of information without service preemption](http://arxiv.org/abs/2104.08050)


  When designing a message transmission system, from the point of view of
making sure that the information transmitted is as fresh as possible, two rules
of thumb seem reasonable: use small buffers and adopt a last-in-first-out
policy. In this paper, we measure freshness of information using the "age of
information" performance measure. Considering it as a stochastic process
operating in a stationary regime, we compute not just the first moment but the
whole marginal distribution of the age of information (something important in
applications) for two well-performing systems. In neither case do we allow for
preemption of the message being processed because this may be difficult to
implement in practice. We assume that the arrival process is Poisson and that
the messages have independent sizes (service times) with common distribution.
We use Palm and Markov-renewal theory to derive explicit results for Laplace
transforms. In particular, this approach can be used to analyze more complex
last-in-first-out systems with larger buffer sizes.

    

### [[2104.11393] Age of information distribution under dynamic service preemption](http://arxiv.org/abs/2104.11393)


  Age of Information (AoI) has emerged as an important quality-of-service
measure for applications that prioritize delivery of the freshest information,
e.g., virtual or augmented reality over mobile devices and wireless sensor
networks used in the control of cyber-physical systems. We derive the Laplace
transform of the stationary AoI for the M/GI/1/2 system with a "dynamic"
service preemption and pushout policy depending on the existing service time of
the in-service message. Thus, our system generalizes both the static M/GI/1/2
queue-pushout system without service preemption and the M/GI/1/1 bufferless
system with service preemption - two systems considered to provide very good
AoI performance. Based on our analysis, for a service-time distribution that is
a mixture of deterministic and exponential, we numerically show that the
dynamic policy has lower mean AoI than that of these two static policies and
also that of the well studied M/GI/1/1 blocking system.

    

### [[2110.08362] Fast and Reliable Formal Verification of Smart Contracts with the Move Prover](http://arxiv.org/abs/2110.08362)


  The Move Prover (MVP) is a formal verifier for smart contracts written in the
Move programming language. MVP has an expressive specification language, and is
fast and reliable enough that it can be run routinely by developers and in
integration testing in a few minutes. Besides the simplicity of smart contracts
and the Move language, three transformations are responsible for the
practicality of MVP: (1) an alias-free memory model, (2) fine-grained invariant
checking, and (3) monomorphization. The entirety of the Move code for the Diem
blockchain has been extensively specified and can be completely verified by MVP
in a few minutes. Changes in the Diem framework must be successfully verified
before being integrated into the open source repository on GitHub.

    

### [[2110.08537] Verification of MPI programs](http://arxiv.org/abs/2110.08537)


  In this paper, we outline an approach to verifying parallel programs. A new
mathematical model of parallel programs is introduced. The introduced model is
illustrated by the verification of the matrix multiplication MPI program.

    

### [[2110.08650] Challenges Porting a C++ Template-Metaprogramming Abstraction Layer to Directive-based Offloading](http://arxiv.org/abs/2110.08650)


  HPC systems employ a growing variety of compute accelerators with different
architectures and from different vendors. Large scientific applications are
required to run efficiently across these systems but need to retain a single
code-base in order to not stifle development. Directive-based offloading
programming models set out to provide the required portability, but, to
existing codes, they themselves represent yet another API to port to. Here, we
present our approach of porting the GPU-accelerated particle-in-cell code
PIConGPU to OpenACC and OpenMP target by adding two new backends to its
existing C++-template metaprogramming-based offloading abstraction layer alpaka
and avoiding other modifications to the application code. We introduce our
approach in the face of conflicts between requirements and available features
in the standards as well as practical hurdles posed by immature compiler
support.

    

### [[2103.15776] CHAD: Combinatory Homomorphic Automatic Differentiation](http://arxiv.org/abs/2103.15776)


  We introduce Combinatory Homomorphic Automatic Differentiation (CHAD), a
principled, pure, provably correct define-then-run method for performing
forward- and reverse-mode automatic differentiation (AD) on programming
languages with expressive features. It implements AD as a compositional,
type-respecting source-code transformation that generates purely functional
code. This code transformation is principled in the sense that it is the unique
homomorphic (structure preserving) extension to expressive languages of the
well-known and unambiguous definitions of AD for a first-order functional
language. Correctness of the method follows by a (compositional) logical
relations argument that shows that the semantics of the syntactic derivative is
the usual calculus derivative of the semantics of the original program.
In their most elegant formulation, the transformations generate code with
linear types. However, the code transformations can be implemented in a
standard functional language lacking linear types: while the correctness proof
requires tracking of linearity, the actual transformations do not. In fact,
even in a standard functional language, we can get all of the type-safety that
linear types give us: we can implement all linear types used to type the
transformations as abstract types, by using a basic module system.
In this paper, we detail the method when applied to a simple higher-order
language for manipulating statically sized arrays. However, we explain how the
methodology applies, more generally, to functional languages with other
expressive features. Finally, we discuss how the scope of CHAD extends beyond
applications in AD to other dynamic program analyses that accumulate data in a
commutative monoid.

    