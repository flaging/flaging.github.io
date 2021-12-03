
## 2021-12-3

### [[2112.00952] A Discrete-event-based Simulator for Deep Learning at Edge](http://arxiv.org/abs/2112.00952)


  Novel smart environments, such as smart home, smart city, and intelligent
transportation, are driving increasing interest in deploying deep neural
networks (DNN) at edge devices. Unfortunately, deploying DNN on
resource-constrained edge devices poses a huge challenge. If a simulator can
interact with deep learning frameworks, it can facilitate researches on deep
learning at edge. The existing simulation frameworks, such as Matlab, NS-3,
etc., haven't been extended to support simulations of edge learning. To support
large-scale training simulations on edge nodes, we propose a
discrete-event-based edge learning simulator. It includes a deep learning
module and a network simulation module. Specifically, it enable simulations as
an environment for deep learning. Our framework is generic and can be used in
various deep learning problems before the deep learning model is deployed. In
this paper, we give the design and implementation details of the
discrete-event-based learning simulator and present an illustrative use case of
the proposed simulator.

    

### [[2112.00956] Personalized Federated Learning of Driver Prediction Models for Autonomous Driving](http://arxiv.org/abs/2112.00956)


  Autonomous vehicles (AVs) must interact with a diverse set of human drivers
in heterogeneous geographic areas. Ideally, fleets of AVs should share
trajectory data to continually re-train and improve trajectory forecasting
models from collective experience using cloud-based distributed learning. At
the same time, these robots should ideally avoid uploading raw driver
interaction data in order to protect proprietary policies (when sharing
insights with other companies) or protect driver privacy from insurance
companies. Federated learning (FL) is a popular mechanism to learn models in
cloud servers from diverse users without divulging private local data. However,
FL is often not robust -- it learns sub-optimal models when user data comes
from highly heterogeneous distributions, which is a key hallmark of human-robot
interactions. In this paper, we present a novel variant of personalized FL to
specialize robust robot learning models to diverse user distributions. Our
algorithm outperforms standard FL benchmarks by up to 2x in real user studies
that we conducted where human-operated vehicles must gracefully merge lanes
with simulated AVs in the standard CARLA and CARLO AV simulators.

    

### [[2112.01039] A Communication-efficient Federated learning assisted by Central data: Implementation of vertical training into Horizontal Federated learning](http://arxiv.org/abs/2112.01039)


  Federated learning (FL) has emerged to jointly train a model with distributed
data sets in IoT while avoiding the need for central data collection. Due to
limited observation range, such data sets can only reflect local information,
which limits the quality of trained models. In practical network, the global
information and local observations always coexist, which requires joint
consideration for learning to make reasonable policy. However, in horizontal FL
among distributed clients, the central agency only acts as a model aggregator
without utilizing its global features to further improve the model. This could
largely degrade the performance in some missions such as flow prediction, where
the global information could obviously enhance the accuracy. Meanwhile, such
global feature may not be directly transmitted to agents for data security.
Then how to utilize the global observation residing in the central agency while
protecting its safety rises up as an important problem in FL. In this paper, we
developed the vertical-horizontal federated learning (VHFL) process, where the
global feature is shared with the agents in a procedure similar to vertical FL
without extra communication rounds. Considering the delay and packet loss, we
analyzed its convergence in the network system and validated its performance by
experiments. The proposed VHFL could enhance the accuracy compared with the
horizontal FL while protecting the security of global data.

    

### [[2112.01068] The Packet Number Space Debate in Multipath QUIC](http://arxiv.org/abs/2112.01068)


  With a standardization process that attracted many interest, QUIC can been
seen as the next general-purpose transport protocol. Still, it does not provide
true multipath support yet, missing some use cases that MPTCP can address. To
fill that gap, the IETF recently adopted a multipath proposal merging all the
proposed designs. While it focuses on its core components, there still remains
one major design issue in the proposal: the number of packet number spaces that
should be used. This paper provides experimental results with two different
Multipath QUIC implementations based on NS3 simulations to understand the
impact of using one packet number space per path or a single packet number
space for the whole connection. Our results suggest that using one packet
number space per path makes the Multipath QUIC connection more resilient to the
receiver's acknowledgment strategy.

    

### [[2112.01154] Autonomous Vehicular Networks: Perspective and Open Issues](http://arxiv.org/abs/2112.01154)


  The vehicular ad hoc networks (VANETs) have been researched for over twenty
years. Although being a fundamental communication approach for vehicles, the
conventional VANETs are challenged by the newly emerged autonomous vehicles
(AVs) which introduce new features and challenges on communications. In the
meantime, with the recent advances of artificial intelligence and 5G cellular
networks, how should the fundamental framework of VANET evolve to utilize the
new technologies? In this article, we reconsider the problem of
vehicle-to-vehicle communications when the network is composed of AVs. We
discuss the features and specific demands of AVs and how the conventional
VANETs should adapt to fit them.

    

### [[2112.01310] An Intelligent Vice Cluster Head Election Protocol in WSN](http://arxiv.org/abs/2112.01310)


  Wireless sensor networks (WSNs) has a practical ability to link a set of
sensors to build a wireless network that can be accessed remotely; this
technology has become increasingly popular in recent years. Wi-Fi-enabled
sensor networks (WSNs) are used to gather information from the environment in
which the network operates. Many obstacles prevent wireless sensor networks
from being used in a wide range of fields. This includes maintaining network
stability and extending network life. In a wireless network, sensors are the
most essential component. Sensors are powered by a battery that has a finite
amount of power. The battery is prone to power loss, and the sensor is
therefore rendered inoperative as a result. In addition, the growing number of
sensor nodes off-site affects the network's stability. The transmission and
reception of information between the sensors and the base consumes the most
energy in the sensor. An Intelligent Vice Cluster Head Selection Protocol is
proposed in this study (IVC LEACH). In order to achieve the best performance
with the least amount of energy consumption, the proposed hierarchical protocol
relies on a fuzzy logic algorithm using four parameters to calculate the value
of each node in the network and divides them into three hierarchical levels
based on their value. This improves network efficiency and reliability while
extending network life by 50 percent more than the original Low Energy Adaptive
Clustering Hierarchy protocol

    

### [[2112.01361] Phasic Policy Gradient Based Resource Allocation for Industrial Internet of Things](http://arxiv.org/abs/2112.01361)


  Time Slotted Channel Hopping (TSCH) behavioural mode has been introduced in
IEEE 802.15.4e standard to address the ultra-high reliability and ultra-low
power communication requirements of Industrial Internet of Things (IIoT)
networks. Scheduling the packet transmissions in IIoT networks is a difficult
task owing to the limited resources and dynamic topology. In this paper, we
propose a phasic policy gradient (PPG) based TSCH schedule learning algorithm.
The proposed PPG based scheduling algorithm overcomes the drawbacks of totally
distributed and totally centralized deep reinforcement learning-based
scheduling algorithms by employing the actor-critic policy gradient method that
learns the scheduling algorithm in two phases, namely policy phase and
auxiliary phase.

    

### [[2007.14247] Downlink channel access performance of NR-U: Impact of numerology and mini-slots on coexistence with Wi-Fi in the 5 GHz band](http://arxiv.org/abs/2007.14247)


  Coexistence between cellular systems and Wi-Fi gained the attention of the
research community when LTE License Assisted Access (LAA) entered the
unlicensed band. The recent introduction of NR-U as part of 5G introduces new
coexistence opportunities because it implements scalable numerology (flexible
subcarrier spacing and OFDM symbol lengths), and non-slot based scheduling
(mini-slots), which considerably impact channel access. This paper analyzes the
impact of NR-U settings on its coexistence with Wi-Fi networks and compares it
with LAA operation using simulations and experiments. First, we propose a
downlink channel access simulation model, which addresses the problem of the
dependency and non-uniformity of transmission attempts of different nodes, as a
result of the synchronization mechanism introduced by NR-U. Second, we validate
the accuracy of the proposed model using FPGA-based LAA, NR-U, and Wi-Fi
prototypes with over-the-air transmissions. Additionally, we show that
replacing LAA with NR-U would not only allow to overcome the problem of
bandwidth wastage caused by reservation signals but also, in some cases, to
preserve fairness in channel access for both scheduled and random-access
systems. Finally, we conclude that fair coexistence of the aforementioned
systems in unlicensed bands is not guaranteed in general, and novel mechanisms
are necessary for improving the sharing of resources between scheduled and
contention-based technologies.

    

### [[2101.09163] The Next Decade of Telecommunications Artificial Intelligence](http://arxiv.org/abs/2101.09163)


  It has been an exciting journey since the mobile communications and
artificial intelligence were conceived 37 years and 64 years ago. While both
fields evolved independently and profoundly changed communications and
computing industries, the rapid convergence of 5G and deep learning is
beginning to significantly transform the core communication infrastructure,
network management and vertical applications. The paper first outlines the
individual roadmaps of mobile communications and artificial intelligence in the
early stage, with a concentration to review the era from 3G to 5G when AI and
mobile communications started to converge. With regard to telecommunications
artificial intelligence, the paper further introduces in detail the progress of
artificial intelligence in the ecosystem of mobile communications. The paper
then summarizes the classifications of AI in telecom ecosystems along with its
evolution paths specified by various international telecommunications
standardization bodies. Towards the next decade, the paper forecasts the
prospective roadmap of telecommunications artificial intelligence. In line with
3GPP and ITU-R timeline of 5G & 6G, the paper further explores the network
intelligence following 3GPP and ORAN routes respectively, experience and
intention driven network management and operation, network AI signalling
system, intelligent middle-office based BSS, intelligent customer experience
management and policy control driven by BSS and OSS convergence, evolution from
SLA to ELA, and intelligent private network for verticals. The paper is
concluded with the vision that AI will reshape the future B5G or 6G landscape
and we need pivot our R&D, standardizations, and ecosystem to fully take the
unprecedented opportunities.

    

### [[2112.00733] Efficient Symptom Inquiring and Diagnosis via Adaptive Alignment of Reinforcement Learning and Classification](http://arxiv.org/abs/2112.00733)


  The medical automatic diagnosis system aims to imitate human doctors in the
real diagnostic process. This task is formulated as a sequential
decision-making problem with symptom inquiring and disease diagnosis. In recent
years, many researchers have used reinforcement learning methods to handle this
task. However, most recent works neglected to distinguish the symptom inquiring
and disease diagnosing actions and mixed them into one action space. This
results in the unsatisfactory performance of reinforcement learning methods on
this task. Moreover, there is a lack of a public evaluation dataset that
contains various diseases and corresponding information. To address these
issues, we first propose a novel method for medical automatic diagnosis with
symptom inquiring and disease diagnosing formulated as a reinforcement learning
task and a classification task, respectively. We also propose a robust and
adaptive method to align the two tasks using distribution entropies as media.
Then, we create a new dataset extracted from the MedlinePlus knowledge base.
The dataset contains more diseases and more complete symptom information. The
simulated patients for experiments are more realistic. Experimental evaluation
results show that our method outperforms three recent state-of-the-art methods
on different datasets by achieving higher medical diagnosis accuracies with few
inquiring turns.

    

### [[2112.00734] Federated Learning with Adaptive Batchnorm for Personalized Healthcare](http://arxiv.org/abs/2112.00734)


  There is a growing interest in applying machine learning techniques for
healthcare. Recently, federated machine learning (FL) is gaining popularity
since it allows researchers to train powerful models without compromising data
privacy and security. However, the performance of existing FL approaches often
deteriorates when encountering non-iid situations where there exist
distribution gaps among clients, and few previous efforts focus on
personalization in healthcare. In this article, we propose AdaFed to tackle
domain shifts and obtain personalized models for local clients. AdaFed learns
the similarity between clients via the statistics of the batch normalization
layers while preserving the specificity of each client with different local
batch normalization. Comprehensive experiments on five healthcare benchmarks
demonstrate that AdaFed achieves better accuracy compared to state-of-the-art
methods (e.g., \textbf{10}\%+ accuracy improvement for PAMAP2) with faster
convergence speed.

    

### [[2112.00735] Reference-guided Pseudo-Label Generation for Medical Semantic Segmentation](http://arxiv.org/abs/2112.00735)


  Producing densely annotated data is a difficult and tedious task for medical
imaging applications. To address this problem, we propose a novel approach to
generate supervision for semi-supervised semantic segmentation. We argue that
visually similar regions between labeled and unlabeled images likely contain
the same semantics and therefore should share their label. Following this
thought, we use a small number of labeled images as reference material and
match pixels in an unlabeled image to the semantics of the best fitting pixel
in a reference set. This way, we avoid pitfalls such as confirmation bias,
common in purely prediction-based pseudo-labeling. Since our method does not
require any architectural changes or accompanying networks, one can easily
insert it into existing frameworks. We achieve the same performance as a
standard fully supervised model on X-ray anatomy segmentation, albeit 95% fewer
labeled images. Aside from an in-depth analysis of different aspects of our
proposed method, we further demonstrate the effectiveness of our
reference-guided learning paradigm by comparing our approach against existing
methods for retinal fluid segmentation with competitive performance as we
improve upon recent work by up to 15% mean IoU.

    

### [[2112.00737] Hardware-friendly Deep Learning by Network Quantization and Binarization](http://arxiv.org/abs/2112.00737)


  Quantization is emerging as an efficient approach to promote
hardware-friendly deep learning and run deep neural networks on
resource-limited hardware. However, it still causes a significant decrease to
the network in accuracy. We summarize challenges of quantization into two
categories: Quantization for Diverse Architectures and Quantization on Complex
Scenes. Our studies focus mainly on applying quantization on various
architectures and scenes and pushing the limit of quantization to extremely
compress and accelerate networks. The comprehensive research on quantization
will achieve more powerful, more efficient, and more flexible hardware-friendly
deep learning, and make it better suited to more real-world applications.

    

### [[2112.00738] Aiding Medical Diagnosis Through the Application of Graph Neural Networks to Functional MRI Scans](http://arxiv.org/abs/2112.00738)


  Graph Neural Networks (GNNs) have been shown to be a powerful tool for
generating predictions from biological data. Their application to neuroimaging
data such as functional magnetic resonance imaging (fMRI) scans has been
limited. However, applying GNNs to fMRI scans may substantially improve
predictive accuracy and could be used to inform clinical diagnosis in the
future. In this paper, we present a novel approach to representing
resting-state fMRI data as a graph containing nodes and edges without omitting
any of the voxels and thus reducing information loss. We compare multiple GNN
architectures and show that they can successfully predict the disease and sex
of a person. We hope to provide a basis for future work to exploit the power of
GNNs when applied to brain imaging data.

    

### [[2112.00739] Incomplete Multi-view Clustering via Cross-view Relation Transfer](http://arxiv.org/abs/2112.00739)


  In this paper, we consider the problem of multi-view clustering on incomplete
views. Compared with complete multi-view clustering, the view-missing problem
increases the difficulty of learning common representations from different
views. To address the challenge, we propose a novel incomplete multi-view
clustering framework, which incorporates cross-view relation transfer and
multi-view fusion learning. Specifically, based on the consistency existing in
multi-view data, we devise a cross-view relation transfer-based completion
module, which transfers known similar inter-instance relationships to the
missing view and recovers the missing data via graph networks based on the
transferred relationship graph. Then the view-specific encoders are designed to
extract the recovered multi-view data, and an attention-based fusion layer is
introduced to obtain the common representation. Moreover, to reduce the impact
of the error caused by the inconsistency between views and obtain a better
clustering structure, a joint clustering layer is introduced to optimize
recovery and clustering simultaneously. Extensive experiments conducted on
several real datasets demonstrate the effectiveness of the proposed method.

    

### [[2112.00778] Quantum advantage in learning from experiments](http://arxiv.org/abs/2112.00778)


  Quantum technology has the potential to revolutionize how we acquire and
process experimental data to learn about the physical world. An experimental
setup that transduces data from a physical system to a stable quantum memory,
and processes that data using a quantum computer, could have significant
advantages over conventional experiments in which the physical system is
measured and the outcomes are processed using a classical computer. We prove
that, in various tasks, quantum machines can learn from exponentially fewer
experiments than those required in conventional experiments. The exponential
advantage holds in predicting properties of physical systems, performing
quantum principal component analysis on noisy states, and learning approximate
models of physical dynamics. In some tasks, the quantum processing needed to
achieve the exponential advantage can be modest; for example, one can
simultaneously learn about many noncommuting observables by processing only two
copies of the system. Conducting experiments with up to 40 superconducting
qubits and 1300 quantum gates, we demonstrate that a substantial quantum
advantage can be realized using today's relatively noisy quantum processors.
Our results highlight how quantum technology can enable powerful new strategies
to learn about nature.

    

### [[2112.00787] Provable Guarantees for Understanding Out-of-distribution Detection](http://arxiv.org/abs/2112.00787)


  Out-of-distribution (OOD) detection is important for deploying machine
learning models in the real world, where test data from shifted distributions
can naturally arise. While a plethora of algorithmic approaches have recently
emerged for OOD detection, a critical gap remains in theoretical understanding.
In this work, we develop an analytical framework that characterizes and unifies
the theoretical understanding for OOD detection. Our analytical framework
motivates a novel OOD detection method for neural networks, GEM, which
demonstrates both theoretical and empirical superiority. In particular, on
CIFAR-100 as in-distribution data, our method outperforms a competitive
baseline by 16.57% (FPR95). Lastly, we formally provide provable guarantees and
comprehensive analysis of our method, underpinning how various properties of
data distribution affect the performance of OOD detection.

    

### [[2112.00791] Controlling Conditional Language Models with Distributional Policy Gradients](http://arxiv.org/abs/2112.00791)


  Machine learning is shifting towards general-purpose pretrained generative
models, trained in a self-supervised manner on large amounts of data, which can
then be applied to solve a large number of tasks. However, due to their generic
training methodology, these models often fail to meet some of the downstream
requirements (e.g. hallucination in abstractive summarization or wrong format
in automatic code generation). This raises an important question on how to
adapt pre-trained generative models to a new task without destroying its
capabilities. Recent work has suggested to solve this problem by representing
task-specific requirements through energy-based models (EBMs) and approximating
these EBMs using distributional policy gradients (DPG). Unfortunately, this
approach is limited to unconditional distributions, represented by
unconditional EBMs. In this paper, we extend this approach to conditional tasks
by proposing Conditional DPG (CDPG). We evaluate CDPG on three different
control objectives across two tasks: summarization with T5 and code generation
with GPT-Neo. Our results show that fine-tuning using CDPG robustly moves these
pretrained models closer towards meeting control objectives and -- in contrast
with baseline approaches -- does not result in catastrophic forgetting.

    

### [[2112.00798] How Smart Guessing Strategies Can Yield Massive Scalability Improvements for Sparse Decision Tree Optimization](http://arxiv.org/abs/2112.00798)


  Sparse decision tree optimization has been one of the most fundamental
problems in AI since its inception and is a challenge at the core of
interpretable machine learning. Sparse decision tree optimization is
computationally hard, and despite steady effort since the 1960's, breakthroughs
have only been made on the problem within the past few years, primarily on the
problem of finding optimal sparse decision trees. However, current
state-of-the-art algorithms often require impractical amounts of computation
time and memory to find optimal or near-optimal trees for some real-world
datasets, particularly those having several continuous-valued features. Given
that the search spaces of these decision tree optimization problems are
massive, can we practically hope to find a sparse decision tree that competes
in accuracy with a black box machine learning model? We address this problem
via smart guessing strategies that can be applied to any optimal
branch-and-bound-based decision tree algorithm. We show that by using these
guesses, we can reduce the run time by multiple orders of magnitude, while
providing bounds on how far the resulting trees can deviate from the black
box's accuracy and expressive power. Our approach enables guesses about how to
bin continuous features, the size of the tree, and lower bounds on the error
for the optimal decision tree. Our experiments show that in many cases we can
rapidly construct sparse decision trees that match the accuracy of black box
models. To summarize: when you are having trouble optimizing, just guess.

    

### [[2112.00806] ReIGNN: State Register Identification Using Graph Neural Networks for Circuit Reverse Engineering](http://arxiv.org/abs/2112.00806)


  Reverse engineering an integrated circuit netlist is a powerful tool to help
detect malicious logic and counteract design piracy. A critical challenge in
this domain is the correct classification of data-path and control-logic
registers in a design. We present ReIGNN, a novel learning-based register
classification methodology that combines graph neural networks (GNNs) with
structural analysis to classify the registers in a circuit with high accuracy
and generalize well across different designs. GNNs are particularly effective
in processing circuit netlists in terms of graphs and leveraging properties of
the nodes and their neighborhoods to learn to efficiently discriminate between
different types of nodes. Structural analysis can further rectify any registers
misclassified as state registers by the GNN by analyzing strongly connected
components in the netlist graph. Numerical results on a set of benchmarks show
that ReIGNN can achieve, on average, 96.5% balanced accuracy and 97.7%
sensitivity across different designs.

    

### [[2112.00811] Revisiting dequantization and quantum advantage in learning tasks](http://arxiv.org/abs/2112.00811)


  It has been shown that the apparent advantage of some quantum machine
learning algorithms may be efficiently replicated using classical algorithms
with suitable data access -- a process known as dequantization. Existing works
on dequantization compare quantum algorithms which take copies of an n-qubit
quantum state $|x\rangle = \sum_{i} x_i |i\rangle$ as input to classical
algorithms which have sample and query (SQ) access to the vector $x$. In this
note, we prove that classical algorithms with SQ access can accomplish some
learning tasks exponentially faster than quantum algorithms with quantum state
inputs. Because classical algorithms are a subset of quantum algorithms, this
demonstrates that SQ access can sometimes be significantly more powerful than
quantum state inputs. Our findings suggest that the absence of exponential
quantum advantage in some learning tasks may be due to SQ access being too
powerful relative to quantum state inputs. If we compare quantum algorithms
with quantum state inputs to classical algorithms with access to measurement
data on quantum states, the landscape of quantum advantage can be dramatically
different.

    

### [[2112.00818] Models of fairness in federated learning](http://arxiv.org/abs/2112.00818)


  In many real-world situations, data is distributed across multiple locations
and can't be combined for training. Federated learning is a novel distributed
learning approach that allows multiple federating agents to jointly learn a
model. While this approach might reduce the error each agent experiences, it
also raises questions of fairness: to what extent can the error experienced by
one agent be significantly lower than the error experienced by another agent?
In this work, we consider two notions of fairness that each may be appropriate
in different circumstances: "egalitarian fairness" (which aims to bound how
dissimilar error rates can be) and "proportional fairness" (which aims to
reward players for contributing more data). For egalitarian fairness, we obtain
a tight multiplicative bound on how widely error rates can diverge between
agents federating together. For proportional fairness, we show that
sub-proportional error (relative to the number of data points contributed) is
guaranteed for any individually rational federating coalition.

    

### [[2112.00825] Output-weighted and relative entropy loss functions for deep learning precursors of extreme events](http://arxiv.org/abs/2112.00825)


  Many scientific and engineering problems require accurate models of dynamical
systems with rare and extreme events. Such problems present a challenging task
for data-driven modelling, with many naive machine learning methods failing to
predict or accurately quantify such events. One cause for this difficulty is
that systems with extreme events, by definition, yield imbalanced datasets and
that standard loss functions easily ignore rare events. That is, metrics for
goodness of fit used to train models are not designed to ensure accuracy on
rare events. This work seeks to improve the performance of regression models
for extreme events by considering loss functions designed to highlight
outliers. We propose a novel loss function, the adjusted output weighted loss,
and extend the applicability of relative entropy based loss functions to
systems with low dimensional output. The proposed functions are tested using
several cases of dynamical systems exhibiting extreme events and shown to
significantly improve accuracy in predictions of extreme events.

    

### [[2112.00826] Inducing Causal Structure for Interpretable Neural Networks](http://arxiv.org/abs/2112.00826)


  In many areas, we have well-founded insights about causal structure that
would be useful to bring into our trained models while still allowing them to
learn in a data-driven fashion. To achieve this, we present the new method of
interchange intervention training(IIT). In IIT, we (1)align variables in the
causal model with representations in the neural model and (2) train a neural
model to match the counterfactual behavior of the causal model on a base input
when aligned representations in both models are set to be the value they would
be for a second source input. IIT is fully differentiable, flexibly combines
with other objectives, and guarantees that the target causal model is acausal
abstraction of the neural model when its loss is minimized. We evaluate IIT on
a structured vision task (MNIST-PVR) and a navigational instruction task
(ReaSCAN). We compare IIT against multi-task training objectives and data
augmentation. In all our experiments, IIT achieves the best results and
produces neural models that are more interpretable in the sense that they
realize the target causal model.

    

### [[2112.00827] Changepoint Analysis of Topic Proportions in Temporal Text Data](http://arxiv.org/abs/2112.00827)


  Changepoint analysis deals with unsupervised detection and/or estimation of
time-points in time-series data, when the distribution generating the data
changes. In this article, we consider \emph{offline} changepoint detection in
the context of large scale textual data. We build a specialised temporal topic
model with provisions for changepoints in the distribution of topic
proportions. As full likelihood based inference in this model is
computationally intractable, we develop a computationally tractable approximate
inference procedure. More specifically, we use sample splitting to estimate
topic polytopes first and then apply a likelihood ratio statistic together with
a modified version of the wild binary segmentation algorithm of Fryzlewicz et
al. (2014). Our methodology facilitates automated detection of structural
changes in large corpora without the need of manual processing by domain
experts. As changepoints under our model correspond to changes in topic
structure, the estimated changepoints are often highly interpretable as marking
the surge or decline in popularity of a fashionable topic. We apply our
procedure on two large datasets: (i) a corpus of English literature from the
period 1800-1922 (Underwoodet al., 2015); (ii) abstracts from the High Energy
Physics arXiv repository (Clementet al., 2019). We obtain some historically
well-known changepoints and discover some new ones.

    

### [[2112.00838] Convergence of batch Greenkhorn for Regularized Multimarginal Optimal Transport](http://arxiv.org/abs/2112.00838)


  In this work we propose a batch version of the Greenkhorn algorithm for
multimarginal regularized optimal transport problems. Our framework is general
enough to cover, as particular cases, some existing algorithms like Sinkhorn
and Greenkhorn algorithm for the bi-marginal setting, and (greedy)
MultiSinkhorn for multimarginal optimal transport. We provide a complete
converge analysis, which is based on the properties of the iterative Bregman
projections (IBP) method with greedy control. Global linear rate of convergence
and explicit bound on the iteration complexity are obtained. When specialized
to above mentioned algorithms, our results give new insights and/or improve
existing ones.

    

### [[2112.00845] Differentially Private SGD with Sparse Gradients](http://arxiv.org/abs/2112.00845)


  To protect sensitive training data, differentially private stochastic
gradient descent (DP-SGD) has been adopted in deep learning to provide
rigorously defined privacy. However, DP-SGD requires the injection of an amount
of noise that scales with the number of gradient dimensions, resulting in large
performance drops compared to non-private training. In this work, we propose
random freeze which randomly freezes a progressively increasing subset of
parameters and results in sparse gradient updates while maintaining or
increasing accuracy. We theoretically prove the convergence of random freeze
and find that random freeze exhibits a signal loss and perturbation moderation
trade-off in DP-SGD. Applying random freeze across various DP-SGD frameworks,
we maintain accuracy within the same number of iterations while achieving up to
70% representation sparsity, which demonstrates that the trade-off exists in a
variety of DP-SGD methods. We further note that random freeze significantly
improves accuracy, in particular for large networks. Additionally, axis-aligned
sparsity induced by random freeze leads to various advantages for projected
DP-SGD or federated learning in terms of computational cost, memory footprint
and communication overhead.

    

### [[2112.00847] CLAWS: Contrastive Learning with hard Attention and Weak Supervision](http://arxiv.org/abs/2112.00847)


  Learning effective visual representations without human supervision is a
long-standing problem in computer vision. Recent advances in self-supervised
learning algorithms have utilized contrastive learning, with methods such as
SimCLR, which applies a composition of augmentations to an image, and minimizes
a contrastive loss between the two augmented images. In this paper, we present
CLAWS, an annotation-efficient learning framework, addressing the problem of
manually labeling large-scale agricultural datasets along with potential
applications such as anomaly detection and plant growth analytics. CLAWS uses a
network backbone inspired by SimCLR and weak supervision to investigate the
effect of contrastive learning within class clusters. In addition, we inject a
hard attention mask to the cropped input image before maximizing agreement
between the image pairs using a contrastive loss function. This mask forces the
network to focus on pertinent object features and ignore background features.
We compare results between a supervised SimCLR and CLAWS using an agricultural
dataset with 227,060 samples consisting of 11 different crop classes. Our
experiments and extensive evaluations show that CLAWS achieves a competitive
NMI score of 0.7325. Furthermore, CLAWS engenders the creation of low
dimensional representations of very large datasets with minimal parameter
tuning and forming well-defined clusters, which lends themselves to using
efficient, transparent, and highly interpretable clustering methods such as
Gaussian Mixture Models.

    

### [[2112.00851] The Physics of Machine Learning: An Intuitive Introduction for the Physical Scientist](http://arxiv.org/abs/2112.00851)


  This article is intended for physical scientists who wish to gain deeper
insights into machine learning algorithms which we present via the domain they
know best, physics. We begin with a review of two energy-based machine learning
algorithms, Hopfield networks and Boltzmann machines, and their connection to
the Ising model. This serves as a foundation to understand the phenomenon of
learning more generally. Equipped with this intuition we then delve into
additional, more "practical," machine learning architectures including
feedforward neural networks, convolutional neural networks, and autoencoders.
We also provide code that explicitly demonstrates training a neural network
with gradient descent.

    

### [[2112.00856] Decomposing Representations for Deterministic Uncertainty Estimation](http://arxiv.org/abs/2112.00856)


  Uncertainty estimation is a key component in any deployed machine learning
system. One way to evaluate uncertainty estimation is using
"out-of-distribution" (OoD) detection, that is, distinguishing between the
training data distribution and an unseen different data distribution using
uncertainty. In this work, we show that current feature density based
uncertainty estimators cannot perform well consistently across different OoD
detection settings. To solve this, we propose to decompose the learned
representations and integrate the uncertainties estimated on them separately.
Through experiments, we demonstrate that we can greatly improve the performance
and the interpretability of the uncertainty estimation.

    

### [[2112.00861] A General Language Assistant as a Laboratory for Alignment](http://arxiv.org/abs/2112.00861)


  Given the broad capabilities of large language models, it should be possible
to work towards a general-purpose, text-based assistant that is aligned with
human values, meaning that it is helpful, honest, and harmless. As an initial
foray in this direction we study simple baseline techniques and evaluations,
such as prompting. We find that the benefits from modest interventions increase
with model size, generalize to a variety of alignment evaluations, and do not
compromise the performance of large models. Next we investigate scaling trends
for several training objectives relevant to alignment, comparing imitation
learning, binary discrimination, and ranked preference modeling. We find that
ranked preference modeling performs much better than imitation learning, and
often scales more favorably with model size. In contrast, binary discrimination
typically performs and scales very similarly to imitation learning. Finally we
study a `preference model pre-training' stage of training, with the goal of
improving sample efficiency when finetuning on human preferences.

    

### [[2112.00874] Neural Stochastic Dual Dynamic Programming](http://arxiv.org/abs/2112.00874)


  Stochastic dual dynamic programming (SDDP) is a state-of-the-art method for
solving multi-stage stochastic optimization, widely used for modeling
real-world process optimization tasks. Unfortunately, SDDP has a worst-case
complexity that scales exponentially in the number of decision variables, which
severely limits applicability to only low dimensional problems. To overcome
this limitation, we extend SDDP by introducing a trainable neural model that
learns to map problem instances to a piece-wise linear value function within
intrinsic low-dimension space, which is architected specifically to interact
with a base SDDP solver, so that can accelerate optimization performance on new
instances. The proposed Neural Stochastic Dual Dynamic Programming ($\nu$-SDDP)
continually self-improves by solving successive problems. An empirical
investigation demonstrates that $\nu$-SDDP can significantly reduce problem
solving cost without sacrificing solution quality over competitors such as SDDP
and reinforcement learning algorithms, across a range of synthetic and
real-world process optimization problems.

    

### [[2112.00881] Learning Invariant Representations with Missing Data](http://arxiv.org/abs/2112.00881)


  Spurious correlations allow flexible models to predict well during training
but poorly on related test populations. Recent work has shown that models that
satisfy particular independencies involving correlation-inducing
\textit{nuisance} variables have guarantees on their test performance.
Enforcing such independencies requires nuisances to be observed during
training. However, nuisances, such as demographics or image background labels,
are often missing. Enforcing independence on just the observed data does not
imply independence on the entire population. Here we derive \acrshort{mmd}
estimators used for invariance objectives under missing nuisances. On
simulations and clinical data, optimizing through these estimates achieves test
performance similar to using estimators that make use of the full data.

    

### [[2112.00882] Robust and Adaptive Temporal-Difference Learning Using An Ensemble of Gaussian Processes](http://arxiv.org/abs/2112.00882)


  Value function approximation is a crucial module for policy evaluation in
reinforcement learning when the state space is large or continuous. The present
paper takes a generative perspective on policy evaluation via
temporal-difference (TD) learning, where a Gaussian process (GP) prior is
presumed on the sought value function, and instantaneous rewards are
probabilistically generated based on value function evaluations at two
consecutive states. Capitalizing on a random feature-based approximant of the
GP prior, an online scalable (OS) approach, termed {OS-GPTD}, is developed to
estimate the value function for a given policy by observing a sequence of
state-reward pairs. To benchmark the performance of OS-GPTD even in an
adversarial setting, where the modeling assumptions are violated, complementary
worst-case analyses are performed by upper-bounding the cumulative Bellman
error as well as the long-term reward prediction error, relative to their
counterparts from a fixed value function estimator with the entire state-reward
trajectory in hindsight. Moreover, to alleviate the limited expressiveness
associated with a single fixed kernel, a weighted ensemble (E) of GP priors is
employed to yield an alternative scheme, termed OS-EGPTD, that can jointly
infer the value function, and select interactively the EGP kernel on-the-fly.
Finally, performances of the novel OS-(E)GPTD schemes are evaluated on two
benchmark problems.

    

### [[2112.00885] Safe Exploration for Constrained Reinforcement Learning with Provable Guarantees](http://arxiv.org/abs/2112.00885)


  We consider the problem of learning an episodic safe control policy that
minimizes an objective function, while satisfying necessary safety constraints
-- both during learning and deployment. We formulate this safety constrained
reinforcement learning (RL) problem using the framework of a finite-horizon
Constrained Markov Decision Process (CMDP) with an unknown transition
probability function. Here, we model the safety requirements as constraints on
the expected cumulative costs that must be satisfied during all episodes of
learning. We propose a model-based safe RL algorithm that we call the
Optimistic-Pessimistic Safe Reinforcement Learning (OPSRL) algorithm, and show
that it achieves an $\tilde{\mathcal{O}}(S^{2}\sqrt{A H^{7}K}/ (\bar{C} -
\bar{C}_{b}))$ cumulative regret without violating the safety constraints
during learning, where $S$ is the number of states, $A$ is the number of
actions, $H$ is the horizon length, $K$ is the number of learning episodes, and
$(\bar{C} - \bar{C}_{b})$ is the safety gap, i.e., the difference between the
constraint value and the cost of a known safe baseline policy. The scaling as
$\tilde{\mathcal{O}}(\sqrt{K})$ is the same as the traditional approach where
constraints may be violated during learning, which means that our algorithm
suffers no additional regret in spite of providing a safety guarantee. Our key
idea is to use an optimistic exploration approach with pessimistic constraint
enforcement for learning the policy. This approach simultaneously incentivizes
the exploration of unknown states while imposing a penalty for visiting states
that are likely to cause violation of safety constraints. We validate our
algorithm by evaluating its performance on benchmark problems against
conventional approaches.

    

### [[2112.00890] Counterfactual Explanations via Latent Space Projection and Interpolation](http://arxiv.org/abs/2112.00890)


  Counterfactual explanations represent the minimal change to a data sample
that alters its predicted classification, typically from an unfavorable initial
class to a desired target class. Counterfactuals help answer questions such as
"what needs to change for this application to get accepted for a loan?". A
number of recently proposed approaches to counterfactual generation give
varying definitions of "plausible" counterfactuals and methods to generate
them. However, many of these methods are computationally intensive and provide
unconvincing explanations. Here we introduce SharpShooter, a method for binary
classification that starts by creating a projected version of the input that
classifies as the target class. Counterfactual candidates are then generated in
latent space on the interpolation line between the input and its projection. We
then demonstrate that our framework translates core characteristics of a sample
to its counterfactual through the use of learned representations. Furthermore,
we show that SharpShooter is competitive across common quality metrics on
tabular and image datasets while being orders of magnitude faster than two
comparable methods and excels at measures of realism, making it well-suited for
high velocity machine learning applications which require timely explanations.

    

### [[2112.00901] Hindsight Task Relabelling: Experience Replay for Sparse Reward Meta-RL](http://arxiv.org/abs/2112.00901)


  Meta-reinforcement learning (meta-RL) has proven to be a successful framework
for leveraging experience from prior tasks to rapidly learn new related tasks,
however, current meta-RL approaches struggle to learn in sparse reward
environments. Although existing meta-RL algorithms can learn strategies for
adapting to new sparse reward tasks, the actual adaptation strategies are
learned using hand-shaped reward functions, or require simple environments
where random exploration is sufficient to encounter sparse reward. In this
paper, we present a formulation of hindsight relabeling for meta-RL, which
relabels experience during meta-training to enable learning to learn entirely
using sparse reward. We demonstrate the effectiveness of our approach on a
suite of challenging sparse reward goal-reaching environments that previously
required dense reward during meta-training to solve. Our approach solves these
environments using the true sparse reward function, with performance comparable
to training with a proxy dense reward function.

    

### [[2112.00905] CELLS: Cost-Effective Evolution in Latent Space for Goal-Directed Molecular Generation](http://arxiv.org/abs/2112.00905)


  Efficiently discovering molecules that meet various property requirements can
significantly benefit the drug discovery industry. Since it is infeasible to
search over the entire chemical space, recent works adopt generative models for
goal-directed molecular generation. They tend to utilize the iterative
processes, optimizing the parameters of the molecular generative models at each
iteration to produce promising molecules for further validation. Assessments
are exploited to evaluate the generated molecules at each iteration, providing
direction for model optimization. However, most previous works require a
massive number of expensive and time-consuming assessments, e.g., wet
experiments and molecular dynamic simulations, leading to the lack of
practicability. To reduce the assessments in the iterative process, we propose
a cost-effective evolution strategy in latent space, which optimizes the
molecular latent representation vectors instead. We adopt a pre-trained
molecular generative model to map the latent and observation spaces, taking
advantage of the large-scale unlabeled molecules to learn chemical knowledge.
To further reduce the number of expensive assessments, we introduce a
pre-screener as the proxy to the assessments. We conduct extensive experiments
on multiple optimization tasks comparing the proposed framework to several
advanced techniques, showing that the proposed framework achieves better
performance with fewer assessments.

    

### [[2112.00911] ProtGNN: Towards Self-Explaining Graph Neural Networks](http://arxiv.org/abs/2112.00911)


  Despite the recent progress in Graph Neural Networks (GNNs), it remains
challenging to explain the predictions made by GNNs. Existing explanation
methods mainly focus on post-hoc explanations where another explanatory model
is employed to provide explanations for a trained GNN. The fact that post-hoc
methods fail to reveal the original reasoning process of GNNs raises the need
of building GNNs with built-in interpretability. In this work, we propose
Prototype Graph Neural Network (ProtGNN), which combines prototype learning
with GNNs and provides a new perspective on the explanations of GNNs. In
ProtGNN, the explanations are naturally derived from the case-based reasoning
process and are actually used during classification. The prediction of ProtGNN
is obtained by comparing the inputs to a few learned prototypes in the latent
space. Furthermore, for better interpretability and higher efficiency, a novel
conditional subgraph sampling module is incorporated to indicate which part of
the input graph is most similar to each prototype in ProtGNN+. Finally, we
evaluate our method on a wide range of datasets and perform concrete case
studies. Extensive results show that ProtGNN and ProtGNN+ can provide inherent
interpretability while achieving accuracy on par with the non-interpretable
counterparts.

    

### [[2112.00914] HyperSPNs: Compact and Expressive Probabilistic Circuits](http://arxiv.org/abs/2112.00914)


  Probabilistic circuits (PCs) are a family of generative models which allows
for the computation of exact likelihoods and marginals of its probability
distributions. PCs are both expressive and tractable, and serve as popular
choices for discrete density estimation tasks. However, large PCs are
susceptible to overfitting, and only a few regularization strategies (e.g.,
dropout, weight-decay) have been explored. We propose HyperSPNs: a new paradigm
of generating the mixture weights of large PCs using a small-scale neural
network. Our framework can be viewed as a soft weight-sharing strategy, which
combines the greater expressiveness of large models with the better
generalization and memory-footprint properties of small models. We show the
merits of our regularization strategy on two state-of-the-art PC families
introduced in recent literature -- RAT-SPNs and EiNETs -- and demonstrate
generalization improvements in both models on a suite of density estimation
benchmarks in both discrete and continuous domains.

    

### [[2112.00925] Context-Aware Online Client Selection for Hierarchical Federated Learning](http://arxiv.org/abs/2112.00925)


  Federated Learning (FL) has been considered as an appealing framework to
tackle data privacy issues of mobile devices compared to conventional Machine
Learning (ML). Using Edge Servers (ESs) as intermediaries to perform model
aggregation in proximity can reduce the transmission overhead, and it enables
great potentials in low-latency FL, where the hierarchical architecture of FL
(HFL) has been attracted more attention. Designing a proper client selection
policy can significantly improve training performance, and it has been
extensively used in FL studies. However, to the best of our knowledge, there
are no studies focusing on HFL. In addition, client selection for HFL faces
more challenges than conventional FL, e.g., the time-varying connection of
client-ES pairs and the limited budget of the Network Operator (NO). In this
paper, we investigate a client selection problem for HFL, where the NO learns
the number of successful participating clients to improve the training
performance (i.e., select as many clients in each round) as well as under the
limited budget on each ES. An online policy, called Context-aware Online Client
Selection (COCS), is developed based on Contextual Combinatorial Multi-Armed
Bandit (CC-MAB). COCS observes the side-information (context) of local
computing and transmission of client-ES pairs and makes client selection
decisions to maximize NO's utility given a limited budget. Theoretically, COCS
achieves a sublinear regret compared to an Oracle policy on both strongly
convex and non-convex HFL. Simulation results also support the efficiency of
the proposed COCS policy on real-world datasets.

    

### [[2112.00940] Reward-Free Attacks in Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2112.00940)


  We investigate how effective an attacker can be when it only learns from its
victim's actions, without access to the victim's reward. In this work, we are
motivated by the scenario where the attacker wants to behave strategically when
the victim's motivations are unknown. We argue that one heuristic approach an
attacker can use is to maximize the entropy of the victim's policy. The policy
is generally not obfuscated, which implies it may be extracted simply by
passively observing the victim. We provide such a strategy in the form of a
reward-free exploration algorithm that maximizes the attacker's entropy during
the exploration phase, and then maximizes the victim's empirical entropy during
the planning phase. In our experiments, the victim agents are subverted through
policy entropy maximization, implying an attacker might not need access to the
victim's reward to succeed. Hence, reward-free attacks, which are based only on
observing behavior, show the feasibility of an attacker to act strategically
without knowledge of the victim's motives even if the victim's reward
information is protected.

    

### [[2112.00945] DPVI: A Dynamic-Weight Particle-Based Variational Inference Framework](http://arxiv.org/abs/2112.00945)


  The recently developed Particle-based Variational Inference (ParVI) methods
drive the empirical distribution of a set of \emph{fixed-weight} particles
towards a given target distribution $\pi$ by iteratively updating particles'
positions. However, the fixed weight restriction greatly confines the empirical
distribution's approximation ability, especially when the particle number is
limited. In this paper, we propose to dynamically adjust particles' weights
according to a Fisher-Rao reaction flow. We develop a general Dynamic-weight
Particle-based Variational Inference (DPVI) framework according to a novel
continuous composite flow, which evolves the positions and weights of particles
simultaneously. We show that the mean-field limit of our composite flow is
actually a Wasserstein-Fisher-Rao gradient flow of certain dissimilarity
functional $\mathcal{F}$, which leads to a faster decrease of $\mathcal{F}$
than the Wasserstein gradient flow underlying existing fixed-weight ParVIs. By
using different finite-particle approximations in our general framework, we
derive several efficient DPVI algorithms. The empirical results demonstrate the
superiority of our derived DPVI algorithms over their fixed-weight
counterparts.

    

### [[2112.00950] Quantile Filtered Imitation Learning](http://arxiv.org/abs/2112.00950)


  We introduce quantile filtered imitation learning (QFIL), a novel policy
improvement operator designed for offline reinforcement learning. QFIL performs
policy improvement by running imitation learning on a filtered version of the
offline dataset. The filtering process removes $ s,a $ pairs whose estimated Q
values fall below a given quantile of the pushforward distribution over values
induced by sampling actions from the behavior policy. The definitions of both
the pushforward Q distribution and resulting value function quantile are key
contributions of our method. We prove that QFIL gives us a safe policy
improvement step with function approximation and that the choice of quantile
provides a natural hyperparameter to trade off bias and variance of the
improvement step. Empirically, we perform a synthetic experiment illustrating
how QFIL effectively makes a bias-variance tradeoff and we see that QFIL
performs well on the D4RL benchmark.

    

### [[2112.00953] Maximum Consensus by Weighted Influences of Monotone Boolean Functions](http://arxiv.org/abs/2112.00953)


  Robust model fitting is a fundamental problem in computer vision: used to
pre-process raw data in the presence of outliers. Maximisation of Consensus
(MaxCon) is one of the most popular robust criteria and widely used. Recently
(Tennakoon et al. CVPR2021), a connection has been made between MaxCon and
estimation of influences of a Monotone Boolean function. Equipping the Boolean
cube with different measures and adopting different sampling strategies (two
sides of the same coin) can have differing effects: which leads to the current
study. This paper studies the concept of weighted influences for solving
MaxCon. In particular, we study endowing the Boolean cube with the Bernoulli
measure and performing biased (as opposed to uniform) sampling. Theoretically,
we prove the weighted influences, under this measure, of points belonging to
larger structures are smaller than those of points belonging to smaller
structures in general. We also consider another "natural" family of
sampling/weighting strategies, sampling with uniform measure concentrated on a
particular (Hamming) level of the cube.
Based on weighted sampling, we modify the algorithm of Tennakoon et al., and
test on both synthetic and real datasets. This paper is not promoting a new
approach per se, but rather studying the issue of weighted sampling.
Accordingly, we are not claiming to have produced a superior algorithm: rather
we show some modest gains of Bernoulli sampling, and we illuminate some of the
interactions between structure in data and weighted sampling.

    

### [[2112.00955] Source Free Unsupervised Graph Domain Adaptation](http://arxiv.org/abs/2112.00955)


  Graph Neural Networks (GNNs) have achieved great success on a variety of
tasks with graph-structural data, among which node classification is an
essential one. Unsupervised Graph Domain Adaptation (UGDA) shows its practical
value of reducing the labeling cost for node classification. It leverages
knowledge from a labeled graph (i.e., source domain) to tackle the same task on
another unlabeled graph (i.e., target domain). Most existing UGDA methods
heavily rely on the labeled graph in the source domain. They utilize labels
from the source domain as the supervision signal and are jointly trained on
both the source graph and the target graph. However, in some real-world
scenarios, the source graph is inaccessible because of either unavailability or
privacy issues. Therefore, we propose a novel scenario named Source Free
Unsupervised Graph Domain Adaptation (SFUGDA). In this scenario, the only
information we can leverage from the source domain is the well-trained source
model, without any exposure to the source graph and its labels. As a result,
existing UGDA methods are not feasible anymore. To address the non-trivial
adaptation challenges in this practical scenario, we propose a model-agnostic
algorithm for domain adaptation to fully exploit the discriminative ability of
the source model while preserving the consistency of structural proximity on
the target graph. We prove the effectiveness of the proposed algorithm both
theoretically and empirically. The experimental results on four cross-domain
tasks show consistent improvements of the Macro-F1 score up to 0.17.

    

### [[2112.00958] Hierarchical Neural Implicit Pose Network for Animation and Motion Retargeting](http://arxiv.org/abs/2112.00958)


  We present HIPNet, a neural implicit pose network trained on multiple
subjects across many poses. HIPNet can disentangle subject-specific details
from pose-specific details, effectively enabling us to retarget motion from one
subject to another or to animate between keyframes through latent space
interpolation. To this end, we employ a hierarchical skeleton-based
representation to learn a signed distance function on a canonical unposed
space. This joint-based decomposition enables us to represent subtle details
that are local to the space around the body joint. Unlike previous neural
implicit method that requires ground-truth SDF for training, our model we only
need a posed skeleton and the point cloud for training, and we have no
dependency on a traditional parametric model or traditional skinning
approaches. We achieve state-of-the-art results on various single-subject and
multi-subject benchmarks.

    

### [[2112.00962] Large-Scale Data Mining of Rapid Residue Detection Assay Data From HTML and PDF Documents: Improving Data Access and Visualization for Veterinarians](http://arxiv.org/abs/2112.00962)


  Extra-label drug use in food animal medicine is authorized by the US Animal
Medicinal Drug Use Clarification Act (AMDUCA), and estimated withdrawal
intervals are based on published scientific pharmacokinetic data. Occasionally
there is a paucity of scientific data on which to base a withdrawal interval or
a large number of animals being treated, driving the need to test for drug
residues. Rapid assay commercial farm-side tests are essential for monitoring
drug residues in animal products to protect human health. Active ingredients,
sensitivity, matrices, and species that have been evaluated for commercial
rapid assay tests are typically reported on manufacturers' websites or in PDF
documents that are available to consumers but may require a special access
request. Additionally, this information is not always correlated with
FDA-approved tolerances. Furthermore, parameter changes for these tests can be
very challenging to regularly identify, especially those listed on websites or
in documents that are not publicly available. Therefore, artificial
intelligence plays a critical role in efficiently extracting the data and
ensure current information. Extracting tables from PDF and HTML documents has
been investigated both by academia and commercial tool builders. Research in
text mining of such documents has become a widespread yet challenging arena in
implementing natural language programming. However, techniques of extracting
tables are still in their infancy and being investigated and improved by
researchers. In this study, we developed and evaluated a data-mining method for
automatically extracting rapid assay data from electronic documents. Our
automatic electronic data extraction method includes a software package module,
a developed pattern recognition tool, and a data mining engine. Assay details
were provided by several commercial entities that produce these rapid drug
residue assay

    

### [[2112.00963] Multi-Domain Transformer-Based Counterfactual Augmentation for Earnings Call Analysis](http://arxiv.org/abs/2112.00963)


  Earnings call (EC), as a periodic teleconference of a publicly-traded
company, has been extensively studied as an essential market indicator because
of its high analytical value in corporate fundamentals. The recent emergence of
deep learning techniques has shown great promise in creating automated
pipelines to benefit the EC-supported financial applications. However, these
methods presume all included contents to be informative without refining
valuable semantics from long-text transcript and suffer from EC scarcity issue.
Meanwhile, these black-box methods possess inherent difficulties in providing
human-understandable explanations. To this end, in this paper, we propose a
Multi-Domain Transformer-Based Counterfactual Augmentation, named MTCA, to
address the above problems. Specifically, we first propose a transformer-based
EC encoder to attentively quantify the task-inspired significance of critical
EC content for market inference. Then, a multi-domain counterfactual learning
framework is developed to evaluate the gradient-based variations after we
perturb limited EC informative texts with plentiful cross-domain documents,
enabling MTCA to perform unsupervised data augmentation. As a bonus, we
discover a way to use non-training data as instance-based explanations for
which we show the result with case studies. Extensive experiments on the
real-world financial datasets demonstrate the effectiveness of interpretable
MTCA for improving the volatility evaluation ability of the state-of-the-art by
14.2\% in accuracy.

    

### [[2112.00964] A Survey on Scenario-Based Testing for Automated Driving Systems in High-Fidelity Simulation](http://arxiv.org/abs/2112.00964)


  Automated Driving Systems (ADSs) have seen rapid progress in recent years. To
ensure the safety and reliability of these systems, extensive testings are
being conducted before their future mass deployment. Testing the system on the
road is the closest to real-world and desirable approach, but it is incredibly
costly. Also, it is infeasible to cover rare corner cases using such real-world
testing. Thus, a popular alternative is to evaluate an ADS's performance in
some well-designed challenging scenarios, a.k.a. scenario-based testing.
High-fidelity simulators have been widely used in this setting to maximize
flexibility and convenience in testing what-if scenarios. Although many works
have been proposed offering diverse frameworks/methods for testing specific
systems, the comparisons and connections among these works are still missing.
To bridge this gap, in this work, we provide a generic formulation of
scenario-based testing in high-fidelity simulation and conduct a literature
review on the existing works. We further compare them and present the open
challenges as well as potential future research directions.

    

### [[2112.00971] Personal Comfort Estimation in Partial Observable Environment using Reinforcement Learning](http://arxiv.org/abs/2112.00971)


  The technology used in smart homes have improved to learn the user
preferences from feedbacks in order to provide convenience to the user in the
home environment. Most smart homes learn a uniform model to represent the
thermal preference of user which generally fails when the pool of occupants
includes people having different age, gender, and location. Having different
thermal sensation for each user poses a challenge for the smart homes to learn
a personalized preference for each occupant without forgetting the policy of
others. A smart home with single optimal policy may fail to provide comfort
when a new user with different preference is integrated in the home. In this
paper, we propose POSHS, a Bayesian Reinforcement learning algorithm that can
approximate the current occupant state in a partial observable environment
using its thermal preference and then decide if its a new occupant or belongs
to the pool of previously observed users. We then compare POSHS algorithm with
an LSTM based algorithm to learn and estimate the current state of the occupant
while also taking optimal actions to reduce the timesteps required to set the
preferences. We perform these experiments with upto 5 simulated human models
each based on hierarchical reinforcement learning. The results show that POSHS
can approximate the current user state just from its temperature and humidity
preference and also reduce the number of time-steps required to set optimal
temperature and humidity by the human model in the presence of the smart home.

    

### [[2112.00973] Adversarial Robustness of Deep Reinforcement Learning based Dynamic Recommender Systems](http://arxiv.org/abs/2112.00973)


  Adversarial attacks, e.g., adversarial perturbations of the input and
adversarial samples, pose significant challenges to machine learning and deep
learning techniques, including interactive recommendation systems. The latent
embedding space of those techniques makes adversarial attacks difficult to
detect at an early stage. Recent advance in causality shows that counterfactual
can also be considered one of ways to generate the adversarial samples drawn
from different distribution as the training samples. We propose to explore
adversarial examples and attack agnostic detection on reinforcement
learning-based interactive recommendation systems. We first craft different
types of adversarial examples by adding perturbations to the input and
intervening on the casual factors. Then, we augment recommendation systems by
detecting potential attacks with a deep learning-based classifier based on the
crafted data. Finally, we study the attack strength and frequency of
adversarial examples and evaluate our model on standard datasets with multiple
crafting methods. Our extensive experiments show that most adversarial attacks
are effective, and both attack strength and attack frequency impact the attack
performance. The strategically-timed attack achieves comparative attack
performance with only 1/3 to 1/2 attack frequency. Besides, our black-box
detector trained with one crafting method has the generalization ability over
several other crafting methods.

    

### [[2112.00976] Gaussian Mixture Variational Autoencoder with Contrastive Learning for Multi-Label Classification](http://arxiv.org/abs/2112.00976)


  Multi-label classification (MLC) is a prediction task where each sample can
have more than one label. We propose a novel contrastive learning boosted
multi-label prediction model based on a Gaussian mixture variational
autoencoder (C-GMVAE), which learns a multimodal prior space and employs a
contrastive loss. Many existing methods introduce extra complex neural modules
to capture the label correlations, in addition to the prediction modules. We
found that by using contrastive learning in the supervised setting, we can
exploit label information effectively, and learn meaningful feature and label
embeddings capturing both the label correlations and predictive power, without
extra neural modules. Our method also adopts the idea of learning and aligning
latent spaces for both features and labels. C-GMVAE imposes a Gaussian mixture
structure on the latent space, to alleviate posterior collapse and
over-regularization issues, in contrast to previous works based on a unimodal
prior. C-GMVAE outperforms existing methods on multiple public datasets and can
often match other models' full performance with only 50% of the training data.
Furthermore, we show that the learnt embeddings provide insights into the
interpretation of label-label interactions.

    

### [[2112.00979] Recommending with Recommendations](http://arxiv.org/abs/2112.00979)


  Recommendation systems are a key modern application of machine learning, but
they have the downside that they often draw upon sensitive user information in
making their predictions. We show how to address this deficiency by basing a
service's recommendation engine upon recommendations from other existing
services, which contain no sensitive information by nature. Specifically, we
introduce a contextual multi-armed bandit recommendation framework where the
agent has access to recommendations for other services. In our setting, the
user's (potentially sensitive) information belongs to a high-dimensional latent
space, and the ideal recommendations for the source and target tasks (which are
non-sensitive) are given by unknown linear transformations of the user
information. So long as the tasks rely on similar segments of the user
information, we can decompose the target recommendation problem into systematic
components that can be derived from the source recommendations, and
idiosyncratic components that are user-specific and cannot be derived from the
source, but have significantly lower dimensionality. We propose an
explore-then-refine approach to learning and utilizing this decomposition; then
using ideas from perturbation theory and statistical concentration of measure,
we prove our algorithm achieves regret comparable to a strong skyline that has
full knowledge of the source and target transformations. We also consider a
generalization of our algorithm to a model with many simultaneous targets and
no source. Our methods obtain superior empirical results on synthetic
benchmarks.

    

### [[2112.00980] Trap of Feature Diversity in the Learning of MLPs](http://arxiv.org/abs/2112.00980)


  In this paper, we discover a two-phase phenomenon in the learning of
multi-layer perceptrons (MLPs). I.e., in the first phase, the training loss
does not decrease significantly, but the similarity of features between
different samples keeps increasing, which hurts the feature diversity. We
explain such a two-phase phenomenon in terms of the learning dynamics of the
MLP. Furthermore, we propose two normalization operations to eliminate the
two-phase phenomenon, which avoids the decrease of the feature diversity and
speeds up the training process.

    

### [[2112.00985] Evaluation of mathematical questioning strategies using data collected through weak supervision](http://arxiv.org/abs/2112.00985)


  A large body of research demonstrates how teachers' questioning strategies
can improve student learning outcomes. However, developing new scenarios is
challenging because of the lack of training data for a specific scenario and
the costs associated with labeling. This paper presents a high-fidelity,
AI-based classroom simulator to help teachers rehearse research-based
mathematical questioning skills. Using a human-in-the-loop approach, we
collected a high-quality training dataset for a mathematical questioning
scenario. Using recent advances in uncertainty quantification, we evaluated our
conversational agent for usability and analyzed the practicality of
incorporating a human-in-the-loop approach for data collection and system
evaluation for a mathematical questioning scenario.

    

### [[2112.00987] On Large Batch Training and Sharp Minima: A Fokker-Planck Perspective](http://arxiv.org/abs/2112.00987)


  We study the statistical properties of the dynamic trajectory of stochastic
gradient descent (SGD). We approximate the mini-batch SGD and the momentum SGD
as stochastic differential equations (SDEs). We exploit the continuous
formulation of SDE and the theory of Fokker-Planck equations to develop new
results on the escaping phenomenon and the relationship with large batch and
sharp minima. In particular, we find that the stochastic process solution tends
to converge to flatter minima regardless of the batch size in the asymptotic
regime. However, the convergence rate is rigorously proven to depend on the
batch size. These results are validated empirically with various datasets and
models.

    

### [[2112.00988] Deep Transfer Learning: A Novel Collaborative Learning Model for Cyberattack Detection Systems in IoT Networks](http://arxiv.org/abs/2112.00988)


  Federated Learning (FL) has recently become an effective approach for
cyberattack detection systems, especially in Internet-of-Things (IoT) networks.
By distributing the learning process across IoT gateways, FL can improve
learning efficiency, reduce communication overheads and enhance privacy for
cyberattack detection systems. Challenges in implementation of FL in such
systems include unavailability of labeled data and dissimilarity of data
features in different IoT networks. In this paper, we propose a novel
collaborative learning framework that leverages Transfer Learning (TL) to
overcome these challenges. Particularly, we develop a novel collaborative
learning approach that enables a target network with unlabeled data to
effectively and quickly learn knowledge from a source network that possesses
abundant labeled data. It is important that the state-of-the-art studies
require the participated datasets of networks to have the same features, thus
limiting the efficiency, flexibility as well as scalability of intrusion
detection systems. However, our proposed framework can address these problems
by exchanging the learning knowledge among various deep learning models, even
when their datasets have different features. Extensive experiments on recent
real-world cybersecurity datasets show that the proposed framework can improve
more than 40% as compared to the state-of-the-art deep learning based
approaches.

    

### [[2112.00989] Embedding Decomposition for Artifacts Removal in EEG Signals](http://arxiv.org/abs/2112.00989)


  Electroencephalogram (EEG) recordings are often contaminated with artifacts.
Various methods have been developed to eliminate or weaken the influence of
artifacts. However, most of them rely on prior experience for analysis. Here,
we propose an deep learning framework to separate neural signal and artifacts
in the embedding space and reconstruct the denoised signal, which is called
DeepSeparator. DeepSeparator employs an encoder to extract and amplify the
features in the raw EEG, a module called decomposer to extract the trend,
detect and suppress artifact and a decoder to reconstruct the denoised signal.
Besides, DeepSeparator can extract the artifact, which largely increases the
model interpretability. The proposed method is tested with a semi-synthetic EEG
dataset and a real task-related EEG dataset, suggesting that DeepSeparator
outperforms the conventional models in both EOG and EMG artifact removal.
DeepSeparator can be extended to multi-channel EEG and data of any length. It
may motivate future developments and application of deep learning-based EEG
denoising. The code for DeepSeparator is available at
this https URL.

    

### [[2112.01001] SEAL: Self-supervised Embodied Active Learning using Exploration and 3D Consistency](http://arxiv.org/abs/2112.01001)


  In this paper, we explore how we can build upon the data and models of
Internet images and use them to adapt to robot vision without requiring any
extra labels. We present a framework called Self-supervised Embodied Active
Learning (SEAL). It utilizes perception models trained on internet images to
learn an active exploration policy. The observations gathered by this
exploration policy are labelled using 3D consistency and used to improve the
perception model. We build and utilize 3D semantic maps to learn both action
and perception in a completely self-supervised manner. The semantic map is used
to compute an intrinsic motivation reward for training the exploration policy
and for labelling the agent observations using spatio-temporal 3D consistency
and label propagation. We demonstrate that the SEAL framework can be used to
close the action-perception loop: it improves object detection and instance
segmentation performance of a pretrained perception model by just moving around
in training environments and the improved perception model can be used to
improve Object Goal Navigation.

    

### [[2112.01008] Editing a classifier by rewriting its prediction rules](http://arxiv.org/abs/2112.01008)


  We present a methodology for modifying the behavior of a classifier by
directly rewriting its prediction rules. Our approach requires virtually no
additional data collection and can be applied to a variety of settings,
including adapting a model to new environments, and modifying it to ignore
spurious features. Our code is available at
this https URL .

    

### [[2112.01010] Differentiable Spatial Planning using Transformers](http://arxiv.org/abs/2112.01010)


  We consider the problem of spatial path planning. In contrast to the
classical solutions which optimize a new plan from scratch and assume access to
the full map with ground truth obstacle locations, we learn a planner from the
data in a differentiable manner that allows us to leverage statistical
regularities from past data. We propose Spatial Planning Transformers (SPT),
which given an obstacle map learns to generate actions by planning over
long-range spatial dependencies, unlike prior data-driven planners that
propagate information locally via convolutional structure in an iterative
manner. In the setting where the ground truth map is not known to the agent, we
leverage pre-trained SPTs in an end-to-end framework that has the structure of
mapper and planner built into it which allows seamless generalization to
out-of-distribution maps and goals. SPTs outperform prior state-of-the-art
differentiable planners across all the setups for both manipulation and
navigation tasks, leading to an absolute improvement of 7-19%.

    

### [[2112.01012] Improving Controllability of Educational Question Generation by Keyword Provision](http://arxiv.org/abs/2112.01012)


  Question Generation (QG) receives increasing research attention in NLP
community. One motivation for QG is that QG significantly facilitates the
preparation of educational reading practice and assessments. While the
significant advancement of QG techniques was reported, current QG results are
not ideal for educational reading practice assessment in terms of
\textit{controllability} and \textit{question difficulty}. This paper reports
our results toward the two issues. First, we report a state-of-the-art
exam-like QG model by advancing the current best model from 11.96 to 20.19 (in
terms of BLEU 4 score). Second, we propose to investigate a variant of QG
setting by allowing users to provide keywords for guiding QG direction. We also
present a simple but effective model toward the QG controllability task.
Experiments are also performed and the results demonstrate the feasibility and
potentials of improving QG diversity and controllability by the proposed
keyword provision QG model.

    

### [[2112.01019] Unconstrained Face Sketch Synthesis via Perception-Adaptive Network and A New Benchmark](http://arxiv.org/abs/2112.01019)


  Face sketch generation has attracted much attention in the field of visual
computing. However, existing methods either are limited to constrained
conditions or heavily rely on various preprocessing steps to deal with
in-the-wild cases. In this paper, we argue that accurately perceiving facial
region and facial components is crucial for unconstrained sketch synthesis. To
this end, we propose a novel Perception-Adaptive Network (PANet), which can
generate high-quality face sketches under unconstrained conditions in an
end-to-end scheme. Specifically, our PANet is composed of i) a Fully
Convolutional Encoder for hierarchical feature extraction, ii) a Face-Adaptive
Perceiving Decoder for extracting potential facial region and handling face
variations, and iii) a Component-Adaptive Perceiving Module for facial
component aware feature representation learning. To facilitate further
researches of unconstrained face sketch synthesis, we introduce a new benchmark
termed WildSketch, which contains 800 pairs of face photo-sketch with large
variations in pose, expression, ethnic origin, background, and illumination.
Extensive experiments demonstrate that the proposed method is capable of
achieving state-of-the-art performance under both constrained and unconstrained
conditions. Our source codes and the WildSketch benchmark are resealed on the
project page this http URL.

    

### [[2112.01020] Learning Optimal Predictive Checklists](http://arxiv.org/abs/2112.01020)


  Checklists are simple decision aids that are often used to promote safety and
reliability in clinical applications. In this paper, we present a method to
learn checklists for clinical decision support. We represent predictive
checklists as discrete linear classifiers with binary features and unit
weights. We then learn globally optimal predictive checklists from data by
solving an integer programming problem. Our method allows users to customize
checklists to obey complex constraints, including constraints to enforce group
fairness and to binarize real-valued features at training time. In addition, it
pairs models with an optimality gap that can inform model development and
determine the feasibility of learning sufficiently accurate checklists on a
given dataset. We pair our method with specialized techniques that speed up its
ability to train a predictive checklist that performs well and has a small
optimality gap. We benchmark the performance of our method on seven clinical
classification problems, and demonstrate its practical benefits by training a
short-form checklist for PTSD screening. Our results show that our method can
fit simple predictive checklists that perform well and that can easily be
customized to obey a rich class of custom constraints.

    

### [[2112.01021] Fighting Fire with Fire: Contrastive Debiasing without Bias-free Data via Generative Bias-transformation](http://arxiv.org/abs/2112.01021)


  Despite their remarkable ability to generalize with over-capacity networks,
deep neural networks often learn to abuse spurious biases in the data instead
of using the actual task-related information. Since such shortcuts are only
effective within the collected dataset, the resulting biased model
underperforms on real-world inputs, or cause unintended social repercussions
such as gender discrimination. To counteract the influence of bias, existing
methods either exploit auxiliary information which is rarely obtainable in
practice, or sift for bias-free samples in the training data, hoping for the
sufficient existence of clean samples. However, such presumptions about the
data are not always guaranteed. In this paper, we propose Contrastive Debiasing
via Generative Bias-transformation~(CDvG) which is capable of operating in more
general environments where existing methods break down due to unmet
presumptions such as insufficient bias-free samples. Motivated by our
observation that not only discriminative models, as previously known, but also
generative models tend to focus on the bias when possible, CDvG uses a
translation model to transform the bias in the sample to another mode of bias
while preserving task-relevant information. Through contrastive learning, we
set transformed biased views against another, learning bias-invariant
representations. Experimental results on synthetic and real-world datasets
demonstrate that our framework outperforms the current state-of-the-arts, and
effectively prevents the models from being biased even when bias-free samples
are extremely scarce.

    

### [[2112.01035] Graph4Rec: A Universal Toolkit with Graph Neural Networks for Recommender Systems](http://arxiv.org/abs/2112.01035)


  In recent years, owing to the outstanding performance in graph representation
learning, graph neural network (GNN) techniques have gained considerable
interests in many real-world scenarios, such as recommender systems and social
networks. In recommender systems, the main challenge is to learn the effective
user/item representations from their interactions. However, many recent
publications using GNNs for recommender systems cannot be directly compared,
due to their difference on datasets and evaluation metrics. Furthermore, many
of them only provide a demo to conduct experiments on small datasets, which is
far away to be applied in real-world recommender systems. To address this
problem, we introduce Graph4Rec, a universal toolkit that unifies the paradigm
to train GNN models into the following parts: graphs input, random walk
generation, ego graphs generation, pairs generation and GNNs selection. From
this training pipeline, one can easily establish his own GNN model with a few
configurations. Besides, we develop a large-scale graph engine and a parameter
server to support distributed GNN training. We conduct a systematic and
comprehensive experiment to compare the performance of different GNN models on
several scenarios in different scale. Extensive experiments are demonstrated to
identify the key components of GNNs. We also try to figure out how the sparse
and dense parameters affect the performance of GNNs. Finally, we investigate
methods including negative sampling, ego graph construction order, and warm
start strategy to find a more effective and efficient GNNs practice on
recommender systems. Our toolkit is based on PGL
this https URL and the code is opened source in
this https URL.

    

### [[2112.01044] ShuttleNet: Position-aware Fusion of Rally Progress and Player Styles for Stroke Forecasting in Badminton](http://arxiv.org/abs/2112.01044)


  The increasing demand for analyzing the insights in sports has stimulated a
line of productive studies from a variety of perspectives, e.g., health state
monitoring, outcome prediction. In this paper, we focus on objectively judging
what and where to return strokes, which is still unexplored in turn-based
sports. By formulating stroke forecasting as a sequence prediction task,
existing works can tackle the problem but fail to model information based on
the characteristics of badminton. To address these limitations, we propose a
novel Position-aware Fusion of Rally Progress and Player Styles framework
(ShuttleNet) that incorporates rally progress and information of the players by
two modified encoder-decoder extractors. Moreover, we design a fusion network
to integrate rally contexts and contexts of the players by conditioning on
information dependency and different positions. Extensive experiments on the
badminton dataset demonstrate that ShuttleNet significantly outperforms the
state-of-the-art methods and also empirically validates the feasibility of each
component in ShuttleNet. On top of that, we provide an analysis scenario for
the stroke forecasting problem.

    

### [[2112.01049] Bayesian Optimization over Permutation Spaces](http://arxiv.org/abs/2112.01049)


  Optimizing expensive to evaluate black-box functions over an input space
consisting of all permutations of d objects is an important problem with many
real-world applications. For example, placement of functional blocks in
hardware design to optimize performance via simulations. The overall goal is to
minimize the number of function evaluations to find high-performing
permutations. The key challenge in solving this problem using the Bayesian
optimization (BO) framework is to trade-off the complexity of statistical model
and tractability of acquisition function optimization. In this paper, we
propose and evaluate two algorithms for BO over Permutation Spaces (BOPS).
First, BOPS-T employs Gaussian process (GP) surrogate model with Kendall
kernels and a Tractable acquisition function optimization approach based on
Thompson sampling to select the sequence of permutations for evaluation.
Second, BOPS-H employs GP surrogate model with Mallow kernels and a Heuristic
search approach to optimize expected improvement acquisition function. We
theoretically analyze the performance of BOPS-T to show that their regret grows
sub-linearly. Our experiments on multiple synthetic and real-world benchmarks
show that both BOPS-T and BOPS-H perform better than the state-of-the-art BO
algorithm for combinatorial spaces. To drive future research on this important
problem, we make new resources and real-world benchmarks available to the
community.

    

### [[2112.01061] Data-Driven Interaction Analysis of Line Failure Cascading in Power Grid Networks](http://arxiv.org/abs/2112.01061)


  We use machine learning tools to model the line interaction of failure
cascading in power grid networks. We first collect data sets of simulated
trajectories of possible consecutive line failure following an initial random
failure and considering actual constraints in a model power network until the
system settles at a steady state. We use weighted $l_1$-regularized logistic
regression-based models to find static and dynamic models that capture pairwise
and latent higher-order lines' failure interactions using pairwise statistical
data. The static model captures the failures' interactions near the steady
states of the network, and the dynamic model captures the failure unfolding in
a time series of consecutive network states. We test models over independent
trajectories of failure unfolding in the network to evaluate their failure
predictive power. We observe asymmetric, strongly positive, and negative
interactions between different lines' states in the network. We use the static
interaction model to estimate the distribution of cascade size and identify
groups of lines that tend to fail together, and compare against the data. The
dynamic interaction model successfully predicts the network state for
long-lasting failure propagation trajectories after an initial failure.

    

### [[2112.01064] AutoGEL: An Automated Graph Neural Network with Explicit Link Information](http://arxiv.org/abs/2112.01064)


  Recently, Graph Neural Networks (GNNs) have gained popularity in a variety of
real-world scenarios. Despite the great success, the architecture design of
GNNs heavily relies on manual labor. Thus, automated graph neural network
(AutoGNN) has attracted interest and attention from the research community,
which makes significant performance improvements in recent years. However,
existing AutoGNN works mainly adopt an implicit way to model and leverage the
link information in the graphs, which is not well regularized to the link
prediction task on graphs, and limits the performance of AutoGNN for other
graph tasks. In this paper, we present a novel AutoGNN work that explicitly
models the link information, abbreviated to AutoGEL. In such a way, AutoGEL can
handle the link prediction task and improve the performance of AutoGNNs on the
node classification and graph classification task. Specifically, AutoGEL
proposes a novel search space containing various design dimensions at both
intra-layer and inter-layer designs and adopts a more robust differentiable
search algorithm to further improve efficiency and effectiveness. Experimental
results on benchmark data sets demonstrate the superiority of AutoGEL on
several tasks.

    

### [[2112.01079] Who will dropout from university? Academic risk prediction based on interpretable machine learning](http://arxiv.org/abs/2112.01079)


  In the institutional research mode, in order to explore which characteristics
are the best indicators for predicting academic risk from the student behavior
data sets that have high-dimensional, unbalanced classified small sample, it
transforms the academic risk prediction of college students into a binary
classification task. It predicts academic risk based on the LightGBM model and
the interpretable machine learning method of Shapley value. The simulation
results show that from the global perspective of the prediction model,
characteristics such as the quality of academic partners, the seating position
in classroom, the dormitory study atmosphere, the English scores of the college
entrance examination, the quantity of academic partners, the addiction level of
video games, the mobility of academic partners, and the degree of truancy are
the best 8 predictors for academic risk. It is contrary to intuition that
characteristics such as living in campus or not, work-study, lipstick
addiction, student leader or not, lover amount, and smoking have little
correlation with university academic risk in this experiment. From the local
perspective of the sample, the factors affecting academic risk vary from person
to person. It can perform personalized interpretable analysis through Shapley
values, which cannot be done by traditional mathematical statistical prediction
models. The academic contributions of this research are mainly in two aspects:
First, the learning interaction networks is proposed for the first time, so
that social behavior can be used to compensate for the one-sided individual
behavior and improve the performance of academic risk prediction. Second, the
introduction of Shapley value calculation makes machine learning that lacks a
clear reasoning process visualized, and provides intuitive decision support for
education managers.

    

### [[2112.01088] Constrained Machine Learning: The Bagel Framework](http://arxiv.org/abs/2112.01088)


  Machine learning models are widely used for real-world applications, such as
document analysis and vision. Constrained machine learning problems are
problems where learned models have to both be accurate and respect constraints.
For continuous convex constraints, many works have been proposed, but learning
under combinatorial constraints is still a hard problem. The goal of this paper
is to broaden the modeling capacity of constrained machine learning problems by
incorporating existing work from combinatorial optimization. We propose first a
general framework called BaGeL (Branch, Generate and Learn) which applies
Branch and Bound to constrained learning problems where a learning problem is
generated and trained at each node until only valid models are obtained.
Because machine learning has specific requirements, we also propose an extended
table constraint to split the space of hypotheses. We validate the approach on
two examples: a linear regression under configuration constraints and a
non-negative matrix factorization with prior knowledge for latent semantics
analysis.

    

### [[2112.01110] Contrastive Adaptive Propagation Graph Neural Networks for Efficient Graph Learning](http://arxiv.org/abs/2112.01110)


  Graph Neural Networks (GNNs) have achieved great success in processing graph
data by extracting and propagating structure-aware features. Existing GNN
research designs various propagation schemes to guide the aggregation of
neighbor information. Recently the field has advanced from local propagation
schemes that focus on local neighbors towards extended propagation schemes that
can directly deal with extended neighbors consisting of both local and
high-order neighbors. Despite the impressive performance, existing approaches
are still insufficient to build an efficient and learnable extended propagation
scheme that can adaptively adjust the influence of local and high-order
neighbors. This paper proposes an efficient yet effective end-to-end framework,
namely Contrastive Adaptive Propagation Graph Neural Networks (CAPGNN), to
address these issues by combining Personalized PageRank and attention
techniques. CAPGNN models the learnable extended propagation scheme with a
polynomial of a sparse local affinity matrix, where the polynomial relies on
Personalized PageRank to provide superior initial coefficients. In order to
adaptively adjust the influence of both local and high-order neighbors, a
coefficient-attention model is introduced to learn to adjust the coefficients
of the polynomial. In addition, we leverage self-supervised learning techniques
and design a negative-free entropy-aware contrastive loss to explicitly take
advantage of unlabeled data for training. We implement CAPGNN as two different
versions named CAPGCN and CAPGAT, which use static and dynamic sparse local
affinity matrices, respectively. Experiments on graph benchmark datasets
suggest that CAPGNN can consistently outperform or match state-of-the-art
baselines. The source code is publicly available at
this https URL.

    

### [[2112.01118] Near-Optimal Lower Bounds For Convex Optimization For All Orders of Smoothness](http://arxiv.org/abs/2112.01118)


  We study the complexity of optimizing highly smooth convex functions. For a
positive integer $p$, we want to find an $\epsilon$-approximate minimum of a
convex function $f$, given oracle access to the function and its first $p$
derivatives, assuming that the $p$th derivative of $f$ is Lipschitz.
Recently, three independent research groups (Jiang et al., PLMR 2019;
Gasnikov et al., PLMR 2019; Bubeck et al., PLMR 2019) developed a new algorithm
that solves this problem with $\tilde{O}(1/\epsilon^{\frac{2}{3p+1}})$ oracle
calls for constant $p$. This is known to be optimal (up to log factors) for
deterministic algorithms, but known lower bounds for randomized algorithms do
not match this bound. We prove a new lower bound that matches this bound (up to
log factors), and holds not only for randomized algorithms, but also for
quantum algorithms.

    

### [[2112.01131] FNR: A Similarity and Transformer-Based Approachto Detect Multi-Modal FakeNews in Social Media](http://arxiv.org/abs/2112.01131)


  The availability and interactive nature of social media have made them the
primary source of news around the globe. The popularity of social media tempts
criminals to pursue their immoral intentions by producing and disseminating
fake news using seductive text and misleading images. Therefore, verifying
social media news and spotting fakes is crucial. This work aims to analyze
multi-modal features from texts and images in social media for detecting fake
news. We propose a Fake News Revealer (FNR) method that utilizes transform
learning to extract contextual and semantic features and contrastive loss to
determine the similarity between image and text. We applied FNR on two real
social media datasets. The results show the proposed method achieves higher
accuracies in detecting fake news compared to the previous works.

    

### [[2112.01137] Deep Learning-Based Carotid Artery Vessel Wall Segmentation in Black-Blood MRI Using Anatomical Priors](http://arxiv.org/abs/2112.01137)


  Carotid artery vessel wall thickness measurement is an essential step in the
monitoring of patients with atherosclerosis. This requires accurate
segmentation of the vessel wall, i.e., the region between an artery's lumen and
outer wall, in black-blood magnetic resonance (MR) images. Commonly used
convolutional neural networks (CNNs) for semantic segmentation are suboptimal
for this task as their use does not guarantee a contiguous ring-shaped
segmentation. Instead, in this work, we cast vessel wall segmentation as a
multi-task regression problem in a polar coordinate system. For each carotid
artery in each axial image slice, we aim to simultaneously find two
non-intersecting nested contours that together delineate the vessel wall. CNNs
applied to this problem enable an inductive bias that guarantees ring-shaped
vessel walls. Moreover, we identify a problem-specific training data
augmentation technique that substantially affects segmentation performance. We
apply our method to segmentation of the internal and external carotid artery
wall, and achieve top-ranking quantitative results in a public challenge, i.e.,
a median Dice similarity coefficient of 0.813 for the vessel wall and median
Hausdorff distances of 0.552 mm and 0.776 mm for lumen and outer wall,
respectively. Moreover, we show how the method improves over a conventional
semantic segmentation approach. These results show that it is feasible to
automatically obtain anatomically plausible segmentations of the carotid vessel
wall with high accuracy.

    

### [[2112.01141] Risk-Aware Algorithms for Combinatorial Semi-Bandits](http://arxiv.org/abs/2112.01141)


  In this paper, we study the stochastic combinatorial multi-armed bandit
problem under semi-bandit feedback. While much work has been done on algorithms
that optimize the expected reward for linear as well as some general reward
functions, we study a variant of the problem, where the objective is to be
risk-aware. More specifically, we consider the problem of maximizing the
Conditional Value-at-Risk (CVaR), a risk measure that takes into account only
the worst-case rewards. We propose new algorithms that maximize the CVaR of the
rewards obtained from the super arms of the combinatorial bandit for the two
cases of Gaussian and bounded arm rewards. We further analyze these algorithms
and provide regret bounds. We believe that our results provide the first
theoretical insights into combinatorial semi-bandit problems in the risk-aware
case.

    

### [[2112.01156] A Unified Framework for Adversarial Attack and Defense in Constrained Feature Space](http://arxiv.org/abs/2112.01156)


  The generation of feasible adversarial examples is necessary for properly
assessing models that work on constrained feature space. However, it remains a
challenging task to enforce constraints into attacks that were designed for
computer vision. We propose a unified framework to generate feasible
adversarial examples that satisfy given domain constraints. Our framework
supports the use cases reported in the literature and can handle both linear
and non-linear constraints. We instantiate our framework into two algorithms: a
gradient-based attack that introduces constraints in the loss function to
maximize, and a multi-objective search algorithm that aims for
misclassification, perturbation minimization, and constraint satisfaction. We
show that our approach is effective on two datasets from different domains,
with a success rate of up to 100%, where state-of-the-art attacks fail to
generate a single feasible example. In addition to adversarial retraining, we
propose to introduce engineered non-convex constraints to improve model
adversarial robustness. We demonstrate that this new defense is as effective as
adversarial retraining. Our framework forms the starting point for research on
constrained adversarial attacks and provides relevant baselines and datasets
that future research can exploit.

    

### [[2112.01163] Robust Robotic Control from Pixels using Contrastive Recurrent State-Space Models](http://arxiv.org/abs/2112.01163)


  Modeling the world can benefit robot learning by providing a rich training
signal for shaping an agent's latent state space. However, learning world
models in unconstrained environments over high-dimensional observation spaces
such as images is challenging. One source of difficulty is the presence of
irrelevant but hard-to-model background distractions, and unimportant visual
details of task-relevant entities. We address this issue by learning a
recurrent latent dynamics model which contrastively predicts the next
observation. This simple model leads to surprisingly robust robotic control
even with simultaneous camera, background, and color distractions. We
outperform alternatives such as bisimulation methods which impose
state-similarity measures derived from divergence in future reward or future
optimal actions. We obtain state-of-the-art results on the Distracting Control
Suite, a challenging benchmark for pixel-based robotic control.

    

### [[2112.01166] Forex Trading Volatility Prediction using NeuralNetwork Models](http://arxiv.org/abs/2112.01166)


  In this paper, we investigate the problem of predicting the future volatility
of Forex currency pairs using the deep learning techniques. We show
step-by-step how to construct the deep-learning network by the guidance of the
empirical patterns of the intra-day volatility. The numerical results show that
the multiscale Long Short-Term Memory (LSTM) model with the input of
multi-currency pairs consistently achieves the state-of-the-art accuracy
compared with both the conventional baselines, i.e. autoregressive and GARCH
model, and the other deep learning models.

    

### [[2112.01174] Multi-task Self-distillation for Graph-based Semi-Supervised Learning](http://arxiv.org/abs/2112.01174)


  Graph convolutional networks have made great progress in graph-based
semi-supervised learning. Existing methods mainly assume that nodes connected
by graph edges are prone to have similar attributes and labels, so that the
features smoothed by local graph structures can reveal the class similarities.
However, there often exist mismatches between graph structures and labels in
many real-world scenarios, where the structures may propagate misleading
features or labels that eventually affect the model performance. In this paper,
we propose a multi-task self-distillation framework that injects
self-supervised learning and self-distillation into graph convolutional
networks to separately address the mismatch problem from the structure side and
the label side. First, we formulate a self-supervision pipeline based on
pre-text tasks to capture different levels of similarities in graphs. The
feature extraction process is encouraged to capture more complex proximity by
jointly optimizing the pre-text task and the target task. Consequently, the
local feature aggregations are improved from the structure side. Second,
self-distillation uses soft labels of the model itself as additional
supervision, which has similar effects as label smoothing. The knowledge from
the classification pipeline and the self-supervision pipeline is collectively
distilled to improve the generalization ability of the model from the label
side. Experiment results show that the proposed method obtains remarkable
performance gains under several classic graph convolutional architectures.

    

### [[2112.01187] Computing Class Hierarchies from Classifiers](http://arxiv.org/abs/2112.01187)


  A class or taxonomic hierarchy is often manually constructed, and part of our
knowledge about the world. In this paper, we propose a novel algorithm for
automatically acquiring a class hierarchy from a classifier which is often a
large neural network these days. The information that we need from a classifier
is its confusion matrix which contains, for each pair of base classes, the
number of errors the classifier makes by mistaking one for another. Our
algorithm produces surprisingly good hierarchies for some well-known deep
neural network models trained on the CIFAR-10 dataset, a neural network model
for predicting the native language of a non-native English speaker, a neural
network model for detecting the language of a written text, and a classifier
for identifying music genre. In the literature, such class hierarchies have
been used to provide interpretability to the neural networks. We also discuss
some other potential uses of the acquired hierarchies.

    

### [[2112.01195] Maximum Entropy Model-based Reinforcement Learning](http://arxiv.org/abs/2112.01195)


  Recent advances in reinforcement learning have demonstrated its ability to
solve hard agent-environment interaction tasks on a super-human level. However,
the application of reinforcement learning methods to practical and real-world
tasks is currently limited due to most RL state-of-art algorithms' sample
inefficiency, i.e., the need for a vast number of training episodes. For
example, OpenAI Five algorithm that has beaten human players in Dota 2 has
trained for thousands of years of game time. Several approaches exist that
tackle the issue of sample inefficiency, that either offers a more efficient
usage of already gathered experience or aim to gain a more relevant and diverse
experience via a better exploration of an environment. However, to our
knowledge, no such approach exists for model-based algorithms, that showed
their high sample efficiency in solving hard control tasks with
high-dimensional state space. This work connects exploration techniques and
model-based reinforcement learning. We have designed a novel exploration method
that takes into account features of the model-based approach. We also
demonstrate through experiments that our method significantly improves the
performance of the model-based algorithm Dreamer.

    

### [[2112.01197] Sample Prior Guided Robust Model Learning to Suppress Noisy Labels](http://arxiv.org/abs/2112.01197)


  Imperfect labels are ubiquitous in real-world datasets and seriously harm the
model performance. Several recent effective methods for handling noisy labels
have two key steps: 1) dividing samples into cleanly labeled and wrongly
labeled sets by training loss, 2) using semi-supervised methods to generate
pseudo-labels for samples in the wrongly labeled set. However, current methods
always hurt the informative hard samples due to the similar loss distribution
between the hard samples and the noisy ones. In this paper, we proposed PGDF
(Prior Guided Denoising Framework), a novel framework to learn a deep model to
suppress noise by generating the samples' prior knowledge, which is integrated
into both dividing samples step and semi-supervised step. Our framework can
save more informative hard clean samples into the cleanly labeled set. Besides,
our framework also promotes the quality of pseudo-labels during the
semi-supervised step by suppressing the noise in the current pseudo-labels
generating scheme. To further enhance the hard samples, we reweight the samples
in the cleanly labeled set during training. We evaluated our method using
synthetic datasets based on CIFAR-10 and CIFAR-100, as well as on the
real-world datasets WebVision and Clothing1M. The results demonstrate
substantial improvements over state-of-the-art methods.

    

### [[2112.01206] Local Citation Recommendation with Hierarchical-Attention Text Encoder and SciBERT-based Reranking](http://arxiv.org/abs/2112.01206)


  The goal of local citation recommendation is to recommend a missing reference
from the local citation context and optionally also from the global context. To
balance the tradeoff between speed and accuracy of citation recommendation in
the context of a large-scale paper database, a viable approach is to first
prefetch a limited number of relevant documents using efficient ranking methods
and then to perform a fine-grained reranking using more sophisticated models.
In that vein, BM25 has been found to be a tough-to-beat approach to
prefetching, which is why recent work has focused mainly on the reranking step.
Even so, we explore prefetching with nearest neighbor search among text
embeddings constructed by a hierarchical attention network. When coupled with a
SciBERT reranker fine-tuned on local citation recommendation tasks, our
hierarchical Attention encoder (HAtten) achieves high prefetch recall for a
given number of candidates to be reranked. Consequently, our reranker needs to
rerank fewer prefetch candidates, yet still achieves state-of-the-art
performance on various local citation recommendation datasets such as ACL-200,
FullTextPeerRead, RefSeer, and arXiv.

    

### [[2112.01221] Analyzing High-Resolution Clouds and Convection using Multi-Channel VAEs](http://arxiv.org/abs/2112.01221)


  Understanding the details of small-scale convection and storm formation is
crucial to accurately represent the larger-scale planetary dynamics. Presently,
atmospheric scientists run high-resolution, storm-resolving simulations to
capture these kilometer-scale weather details. However, because they contain
abundant information, these simulations can be overwhelming to analyze using
conventional approaches. This paper takes a data-driven approach and jointly
embeds spatial arrays of vertical wind velocities, temperatures, and water
vapor information as three "channels" of a VAE architecture. Our "multi-channel
VAE" results in more interpretable and robust latent structures than earlier
work analyzing vertical velocities in isolation. Analyzing and clustering the
VAE's latent space identifies weather patterns and their geographical
manifestations in a fully unsupervised fashion. Our approach shows that VAEs
can play essential roles in analyzing high-dimensional simulation data and
extracting critical weather and climate characteristics.

    

### [[2112.01224] Using word embedding for environmental violation analysis: Evidence from Pennsylvania unconventional oil and gas compliance reports](http://arxiv.org/abs/2112.01224)


  With the booming of the unconventional oil and gas industry, its inevitable
damage to the environment and human health has attracted public attention. We
applied text mining on a total 6057 the type of Environmental Health and Safety
compliance reports from 2008 to 2018 lunched by the Department of Environmental
Protection in Pennsylvania, USA, to discover the intern mechanism of
environmental violations.

    

### [[2112.01230] Early Prediction of Mortality in Critical Care Setting in Sepsis Patients Using Structured Features and Unstructured Clinical Notes](http://arxiv.org/abs/2112.01230)


  Sepsis is an important cause of mortality, especially in intensive care unit
(ICU) patients. Developing novel methods to identify early mortality is
critical for improving survival outcomes in sepsis patients. Using the
MIMIC-III database, we integrated demographic data, physiological measurements
and clinical notes. We built and applied several machine learning models to
predict the risk of hospital mortality and 30-day mortality in sepsis patients.
From the clinical notes, we generated clinically meaningful word
representations and embeddings. Supervised learning classifiers and a deep
learning architecture were used to construct prediction models. The
configurations that utilized both structured and unstructured clinical features
yielded competitive F-measure of 0.512. Our results showed that the approaches
integrating both structured and unstructured clinical features can be
effectively applied to assist clinicians in identifying the risk of mortality
in sepsis patients upon admission to the ICU.

    

### [[2112.01236] Local Justice and the Algorithmic Allocation of Societal Resources](http://arxiv.org/abs/2112.01236)


  AI is increasingly used to aid decision-making about the allocation of scarce
societal resources, for example housing for homeless people, organs for
transplantation, and food donations. Recently, there have been several
proposals for how to design objectives for these systems that attempt to
achieve some combination of fairness, efficiency, incentive compatibility, and
satisfactory aggregation of stakeholder preferences. This paper lays out
possible roles and opportunities for AI in this domain, arguing for a closer
engagement with the political philosophy literature on local justice, which
provides a framework for thinking about how societies have over time framed
objectives for such allocation problems. It also discusses how we may be able
to integrate into this framework the opportunities and risks opened up by the
ubiquity of data and the availability of algorithms that can use them to make
accurate predictions about the future.

    

### [[2112.01247] Predicting Student's Performance Through Data Mining](http://arxiv.org/abs/2112.01247)


  Predicting the performance of students early and as accurately as possible is
one of the biggest challenges of educational institutions. Analyzing the
performance of students early can help in finding the strengths and weakness of
students and help the perform better in examinations. Using machine learning
the student's performance can be predicted with the help of students' data
collected from Learning Management Systems (LMS). The data collected from LMSs
can provide insights about student's behavior that will result in good or bad
performance in examinations which then can be studied and used in helping
students performing poorly in examinations to perform better.

    

### [[2112.01253] Youla-REN: Learning Nonlinear Feedback Policies with Robust Stability Guarantees](http://arxiv.org/abs/2112.01253)


  This paper presents a parameterization of nonlinear controllers for uncertain
systems building on a recently developed neural network architecture, called
the recurrent equilibrium network (REN), and a nonlinear version of the Youla
parameterization. The proposed framework has "built-in" guarantees of
stability, i.e., all policies in the search space result in a contracting
(globally exponentially stable) closed-loop system. Thus, it requires very mild
assumptions on the choice of cost function and the stability property can be
generalized to unseen data. Another useful feature of this approach is that
policies are parameterized directly without any constraints, which simplifies
learning by a broad range of policy-learning methods based on unconstrained
optimization (e.g. stochastic gradient descent). We illustrate the proposed
approach with a variety of simulation examples.

    

### [[2112.01254] Hierarchical Learning to Solve Partial Differential Equations Using Physics-Informed Neural Networks](http://arxiv.org/abs/2112.01254)


  The Neural network-based approach to solving partial differential equations
has attracted considerable attention due to its simplicity and flexibility to
represent the solution of the partial differential equation. In training a
neural network, the network tends to learn global features corresponding to
low-frequency components while high-frequency components are approximated at a
much slower rate (F-principle). For a class of equations in which the solution
contains a wide range of scales, the network training process can suffer from
slow convergence and low accuracy due to its inability to capture the
high-frequency components. In this work, we propose a hierarchical approach to
improve the convergence rate and accuracy of the neural network solution to
partial differential equations. The proposed method comprises multi-training
levels in which a newly introduced neural network is guided to learn the
residual of the previous level approximation. By the nature of neural networks'
training process, the high-level correction is inclined to capture the
high-frequency components. We validate the efficiency and robustness of the
proposed hierarchical approach through a suite of linear and nonlinear partial
differential equations.

    

### [[2112.01259] Borrowing from Similar Code: A Deep Learning NLP-Based Approach for Log Statement Automation](http://arxiv.org/abs/2112.01259)


  Software developers embed logging statements inside the source code as an
imperative duty in modern software development as log files are necessary for
tracking down runtime system issues and troubleshooting system management
tasks. However, the current logging process is mostly manual, and thus, proper
placement and content of logging statements remain as challenges. To overcome
these challenges, methods that aim to automate log placement and predict its
content, i.e., 'where and what to log', are of high interest. Thus, we focus on
predicting the location (i.e., where) and description (i.e., what) for log
statements by utilizing source code clones and natural language processing
(NLP), as these approaches provide additional context and advantage for log
prediction. Specifically, we guide our research with three research questions
(RQs): (RQ1) how similar code snippets, i.e., code clones, can be leveraged for
log statements prediction? (RQ2) how the approach can be extended to automate
log statements' descriptions? and (RQ3) how effective the proposed methods are
for log location and description prediction? To pursue our RQs, we perform an
experimental study on seven open-source Java projects. We introduce an updated
and improved log-aware code-clone detection method to predict the location of
logging statements (RQ1). Then, we incorporate natural language processing
(NLP) and deep learning methods to automate the log statements' description
prediction (RQ2). Our analysis shows that our hybrid NLP and code-clone
detection approach (NLP CC'd) outperforms conventional clone detectors in
finding log statement locations on average by 15.60% and achieves 40.86% higher
performance on BLEU and ROUGE scores for predicting the description of logging
statements when compared to prior research (RQ3).

    

### [[2112.01270] Multi-Object Grasping -- Estimating the Number of Objects in a Robotic Grasp](http://arxiv.org/abs/2112.01270)


  A human hand can grasp a desired number of objects at once from a pile based
solely on tactile sensing. To do so, a robot needs to grasp within a pile,
sense the number of objects in the grasp before lifting, and predict the number
of objects that will remain in the grasp after lifting. It is a challenging
problem because when making the prediction, the robotic hand is still in the
pile and the objects in the grasp are not observable to vision systems.
Moreover, some objects that are grasped by the hand before lifting from the
pile may fall out of the grasp when the hand is lifted. This occurs because
they were supported by other objects in the pile instead of the fingers of the
hand. Therefore, a robotic hand should sense the number of objects in a grasp
using its tactile sensors before lifting. This paper presents novel
multi-object grasping analyzing methods for solving this problem. They include
a grasp volume calculation, tactile force analysis, and a data-driven deep
learning approach. The methods have been implemented on a Barrett hand and then
evaluated in simulations and a real setup with a robotic system. The evaluation
results conclude that once the Barrett hand grasps multiple objects in the
pile, the data-driven model can predict, before lifting, the number of objects
that will remain in the hand after lifting. The root-mean-square errors for our
approach are 0.74 for balls and 0.58 for cubes in simulations, and 1.06 for
balls, and 1.45 for cubes in the real system.

    

### [[2112.01273] RawArray: A Simple, Fast, and Extensible Archival Format for Numeric Data](http://arxiv.org/abs/2112.01273)


  Raw data sizes are growing and proliferating in scientific research, driven
by the success of data-hungry computational methods, such as machine learning.
The preponderance of proprietary and shoehorned data formats make computations
slower and make it harder to reproduce research and to port methods to new
platforms. Here we present the RawArray format: a simple, fast, and extensible
format for archival storage of multidimensional numeric arrays on disk.
The RawArray file format is a simple concatenation of a header array and a
data array. The header comprises seven or more 64-bit unsigned integers. The
array data can be anything. Arbitrary user metadata can be appended to an
RawArray file if desired, for example to store measurement details, color
palettes, or geolocation data.
We present benchmarks showing a factor of 2--3$\times$ speedup over HDF5 for
a range of array sizes and a speedup of up to 20$\times$ in reading the common
deep learning datasets MNIST and CIFAR10.

    

### [[2112.01274] The Impact of Data Distribution on Fairness and Robustness in Federated Learning](http://arxiv.org/abs/2112.01274)


  Federated Learning (FL) is a distributed machine learning protocol that
allows a set of agents to collaboratively train a model without sharing their
datasets. This makes FL particularly suitable for settings where data privacy
is desired. However, it has been observed that the performance of FL is closely
related to the similarity of the local data distributions of agents.
Particularly, as the data distributions of agents differ, the accuracy of the
trained models drop. In this work, we look at how variations in local data
distributions affect the fairness and the robustness properties of the trained
models in addition to the accuracy. Our experimental results indicate that, the
trained models exhibit higher bias, and become more susceptible to attacks as
local data distributions differ. Importantly, the degradation in the fairness,
and robustness can be much more severe than the accuracy. Therefore, we reveal
that small variations that have little impact on the accuracy could still be
important if the trained model is to be deployed in a fairness/security
critical context.

    

### [[2112.01280] Learning Graphon Mean Field Games and Approximate Nash Equilibria](http://arxiv.org/abs/2112.01280)


  Recent advances at the intersection of dense large graph limits and mean
field games have begun to enable the scalable analysis of a broad class of
dynamical sequential games with large numbers of agents. So far, results have
been largely limited to graphon mean field systems with continuous-time
diffusive or jump dynamics, typically without control and with little focus on
computational methods. We propose a novel discrete-time formulation for graphon
mean field games as the limit of non-linear dense graph Markov games with weak
interaction. On the theoretical side, we give extensive and rigorous existence
and approximation properties of the graphon mean field solution in sufficiently
large systems. On the practical side, we provide general learning schemes for
graphon mean field equilibria by either introducing agent equivalence classes
or reformulating the graphon mean field system as a classical mean field
system. By repeatedly finding a regularized optimal control solution and its
generated mean field, we successfully obtain plausible approximate Nash
equilibria in otherwise infeasible large dense graph games with many agents.
Empirically, we are able to demonstrate on a number of examples that the
finite-agent behavior comes increasingly close to the mean field behavior for
our computed equilibria as the graph or system size grows, verifying our
theory. More generally, we successfully apply policy gradient reinforcement
learning in conjunction with sequential Monte Carlo methods.

    

### [[2112.01283] Detecting Extratropical Cyclones of the Northern Hemisphere with Single Shot Detector](http://arxiv.org/abs/2112.01283)


  In this paper, we propose a deep learning-based model to detect extratropical
cyclones (ETCs) of northern hemisphere, while developing a novel workflow of
processing images and generating labels for ETCs. We first label the cyclone
center by adapting an approach from Bonfanti this http URL. [1] and set up criteria of
labeling ETCs of three categories: developing, mature, and declining stages. We
then propose a framework of labeling and preprocessing the images in our
dataset. Once the images and labels are ready to serve as inputs, we create our
object detection model named Single Shot Detector (SSD) to fit the format of
our dataset. We train and evaluate our model with our labeled dataset on two
settings (binary and multiclass classifications), while keeping a record of the
results. Finally, we achieved relatively high performance with detecting ETCs
of mature stage (mean Average Precision is 86.64%), and an acceptable result
for detecting ETCs of all three categories (mean Average Precision 79.34%). We
conclude that the single-shot detector model can succeed in detecting ETCs of
different stages, and it has demonstrated great potential in the future
applications of ETC detection in other relevant settings.

    

### [[2112.01288] How to quantify fields or textures? A guide to the scattering transform](http://arxiv.org/abs/2112.01288)


  Extracting information from stochastic fields or textures is a ubiquitous
task in science, from exploratory data analysis to classification and parameter
estimation. From physics to biology, it tends to be done either through a power
spectrum analysis, which is often too limited, or the use of convolutional
neural networks (CNNs), which require large training sets and lack
interpretability. In this paper, we advocate for the use of the scattering
transform (Mallat 2012), a powerful statistic which borrows mathematical ideas
from CNNs but does not require any training, and is interpretable. We show that
it provides a relatively compact set of summary statistics with visual
interpretation and which carries most of the relevant information in a wide
range of scientific applications. We present a non-technical introduction to
this estimator and we argue that it can benefit data analysis, comparison to
models and parameter inference in many fields of science. Interestingly,
understanding the core operations of the scattering transform allows one to
decipher many key aspects of the inner workings of CNNs.

    

### [[2112.01292] Optimal regularizations for data generation with probabilistic graphical models](http://arxiv.org/abs/2112.01292)


  Understanding the role of regularization is a central question in Statistical
Inference. Empirically, well-chosen regularization schemes often dramatically
improve the quality of the inferred models by avoiding overfitting of the
training data. We consider here the particular case of L 2 and L 1
regularizations in the Maximum A Posteriori (MAP) inference of generative
pairwise graphical models. Based on analytical calculations on Gaussian
multivariate distributions and numerical experiments on Gaussian and Potts
models we study the likelihoods of the training, test, and 'generated data'
(with the inferred models) sets as functions of the regularization strengths.
We show in particular that, at its maximum, the test likelihood and the
'generated' likelihood, which quantifies the quality of the generated samples,
have remarkably close values. The optimal value for the regularization strength
is found to be approximately equal to the inverse sum of the squared couplings
incoming on sites on the underlying network of interactions. Our results seem
largely independent of the structure of the true underlying interactions that
generated the data, of the regularization scheme considered, and are valid when
small fluctuations of the posterior distribution around the MAP estimator are
taken into account. Connections with empirical works on protein models learned
from homologous sequences are discussed.

    

### [[2112.01299] Gradient Inversion Attack: Leaking Private Labels in Two-Party Split Learning](http://arxiv.org/abs/2112.01299)


  Split learning is a popular technique used to perform vertical federated
learning, where the goal is to jointly train a model on the private input and
label data held by two parties. To preserve privacy of the input and label
data, this technique uses a split model and only requires the exchange of
intermediate representations (IR) of the inputs and gradients of the IR between
the two parties during the learning process. In this paper, we propose Gradient
Inversion Attack (GIA), a label leakage attack that allows an adversarial input
owner to learn the label owner's private labels by exploiting the gradient
information obtained during split learning. GIA frames the label leakage attack
as a supervised learning problem by developing a novel loss function using
certain key properties of the dataset and models. Our attack can uncover the
private label data on several multi-class image classification problems and a
binary conversion prediction task with near-perfect accuracy (97.01% - 99.96%),
demonstrating that split learning provides negligible privacy benefits to the
label owner. Furthermore, we evaluate the use of gradient noise to defend
against GIA. While this technique is effective for simpler datasets, it
significantly degrades utility for datasets with higher input dimensionality.
Our findings underscore the need for better privacy-preserving training
techniques for vertically split data.

    

### [[2112.01301] Machine Learning for Air Transport Planning and Management](http://arxiv.org/abs/2112.01301)


  In this work we compare the performance of several machine learning
algorithms applied to the problem of modelling air transport demand.
Forecasting in the air transport industry is an essential part of planning and
managing because of the economic and financial aspects of the industry. The
traditional approach used in airline operations as specified by the
International Civil Aviation Organization is the use of a multiple linear
regression (MLR) model, utilizing cost variables and economic factors. Here,
the performance of models utilizing an artificial neural network (ANN), an
adaptive neuro-fuzzy inference system (ANFIS), a genetic algorithm, a support
vector machine, and a regression tree are compared to MLR. The ANN and ANFIS
had the best performance in terms of the lowest mean squared error.

    

### [[2112.01319] TinyML Platforms Benchmarking](http://arxiv.org/abs/2112.01319)


  Recent advances in state-of-the-art ultra-low power embedded devices for
machine learning (ML) have permitted a new class of products whose key features
enable ML capabilities on microcontrollers with less than 1 mW power
consumption (TinyML). TinyML provides a unique solution by aggregating and
analyzing data at the edge on low-power embedded devices. However, we have only
recently been able to run ML on microcontrollers, and the field is still in its
infancy, which means that hardware, software, and research are changing
extremely rapidly. Consequently, many TinyML frameworks have been developed for
different platforms to facilitate the deployment of ML models and standardize
the process. Therefore, in this paper, we focus on bench-marking two popular
frameworks: Tensorflow Lite Micro (TFLM) on the Arduino Nano BLE and CUBE AI on
the STM32-NucleoF401RE to provide a standardized framework selection criterion
for specific applications.

    

### [[2112.01327] A modified limited memory Nesterov's accelerated quasi-Newton](http://arxiv.org/abs/2112.01327)


  The Nesterov's accelerated quasi-Newton (L)NAQ method has shown to accelerate
the conventional (L)BFGS quasi-Newton method using the Nesterov's accelerated
gradient in several neural network (NN) applications. However, the calculation
of two gradients per iteration increases the computational cost. The Momentum
accelerated Quasi-Newton (MoQ) method showed that the Nesterov's accelerated
gradient can be approximated as a linear combination of past gradients. This
abstract extends the MoQ approximation to limited memory NAQ and evaluates the
performance on a function approximation problem.

    

### [[2112.01328] Homotopy Based Reinforcement Learning with Maximum Entropy for Autonomous Air Combat](http://arxiv.org/abs/2112.01328)


  The Intelligent decision of the unmanned combat aerial vehicle (UCAV) has
long been a challenging problem. The conventional search method can hardly
satisfy the real-time demand during high dynamics air combat scenarios. The
reinforcement learning (RL) method can significantly shorten the decision time
via using neural networks. However, the sparse reward problem limits its
convergence speed and the artificial prior experience reward can easily deviate
its optimal convergent direction of the original task, which raises great
difficulties for the RL air combat application. In this paper, we propose a
homotopy-based soft actor-critic method (HSAC) which focuses on addressing
these problems via following the homotopy path between the original task with
sparse reward and the auxiliary task with artificial prior experience reward.
The convergence and the feasibility of this method are also proved in this
paper. To confirm our method feasibly, we construct a detailed 3D air combat
simulation environment for the RL-based methods training firstly, and we
implement our method in both the attack horizontal flight UCAV task and the
self-play confrontation task. Experimental results show that our method
performs better than the methods only utilizing the sparse reward or the
artificial prior experience reward. The agent trained by our method can reach
more than 98.3% win rate in the attack horizontal flight UCAV task and average
67.4% win rate when confronted with the agents trained by the other two
methods.

    

### [[2112.01330] CSAW-M: An Ordinal Classification Dataset for Benchmarking Mammographic Masking of Cancer](http://arxiv.org/abs/2112.01330)


  Interval and large invasive breast cancers, which are associated with worse
prognosis than other cancers, are usually detected at a late stage due to false
negative assessments of screening mammograms. The missed screening-time
detection is commonly caused by the tumor being obscured by its surrounding
breast tissues, a phenomenon called masking. To study and benchmark
mammographic masking of cancer, in this work we introduce CSAW-M, the largest
public mammographic dataset, collected from over 10,000 individuals and
annotated with potential masking. In contrast to the previous approaches which
measure breast image density as a proxy, our dataset directly provides
annotations of masking potential assessments from five specialists. We also
trained deep learning models on CSAW-M to estimate the masking level and showed
that the estimated masking is significantly more predictive of screening
participants diagnosed with interval and large invasive cancers -- without
being explicitly trained for these tasks -- than its breast density
counterparts.

    

### [[2112.01334] Stationary Diffusion State Neural Estimation for Multiview Clustering](http://arxiv.org/abs/2112.01334)


  Although many graph-based clustering methods attempt to model the stationary
diffusion state in their objectives, their performance limits to using a
predefined graph. We argue that the estimation of the stationary diffusion
state can be achieved by gradient descent over neural networks. We specifically
design the Stationary Diffusion State Neural Estimation (SDSNE) to exploit
multiview structural graph information for co-supervised learning. We explore
how to design a graph neural network specially for unsupervised multiview
learning and integrate multiple graphs into a unified consensus graph by a
shared self-attentional module. The view-shared self-attentional module
utilizes the graph structure to learn a view-consistent global graph.
Meanwhile, instead of using auto-encoder in most unsupervised learning graph
neural networks, SDSNE uses a co-supervised strategy with structure information
to supervise the model learning. The co-supervised strategy as the loss
function guides SDSNE in achieving the stationary state. With the help of the
loss and the self-attentional module, we learn to obtain a graph in which nodes
in each connected component fully connect by the same weight. Experiments on
several multiview datasets demonstrate effectiveness of SDSNE in terms of six
clustering evaluation metrics.

    

### [[2112.01358] Mixing Deep Learning and Multiple Criteria Optimization: An Application to Distributed Learning with Multiple Datasets](http://arxiv.org/abs/2112.01358)


  The training phase is the most important stage during the machine learning
process. In the case of labeled data and supervised learning, machine training
consists in minimizing the loss function subject to different constraints. In
an abstract setting, it can be formulated as a multiple criteria optimization
model in which each criterion measures the distance between the output
associated with a specific input and its label. Therefore, the fitting term is
a vector function and its minimization is intended in the Pareto sense. We
provide stability results of the efficient solutions with respect to
perturbations of input and output data. We then extend the same approach to the
case of learning with multiple datasets. The multiple dataset environment is
relevant when reducing the bias due to the choice of a specific training set.
We propose a scalarization approach to implement this model and numerical
experiments in digit classification using MNIST data.

    

### [[2112.01360] Probabilistic Approach for Road-Users Detection](http://arxiv.org/abs/2112.01360)


  Object detection in autonomous driving applications implies that the
detection and tracking of semantic objects are commonly native to urban driving
environments, as pedestrians and vehicles. One of the major challenges in
state-of-the-art deep-learning based object detection is false positive which
occurrences with overconfident scores. This is highly undesirable in autonomous
driving and other critical robotic-perception domains because of safety
concerns. This paper proposes an approach to alleviate the problem of
overconfident predictions by introducing a novel probabilistic layer to deep
object detection networks in testing. The suggested approach avoids the
traditional Sigmoid or Softmax prediction layer which often produces
overconfident predictions. It is demonstrated that the proposed technique
reduces overconfidence in the false positives without degrading the performance
on the true positives. The approach is validated on the 2D-KITTI objection
detection through the YOLOV4 and SECOND (Lidar-based detector). The proposed
approach enables enabling interpretable probabilistic predictions without the
requirement of re-training the network and therefore is very practical.

    

### [[2112.01368] ScaleVLAD: Improving Multimodal Sentiment Analysis via Multi-Scale Fusion of Locally Descriptors](http://arxiv.org/abs/2112.01368)


  Fusion technique is a key research topic in multimodal sentiment analysis.
The recent attention-based fusion demonstrates advances over simple
operation-based fusion. However, these fusion works adopt single-scale, i.e.,
token-level or utterance-level, unimodal representation. Such single-scale
fusion is suboptimal because that different modality should be aligned with
different granularities. This paper proposes a fusion model named ScaleVLAD to
gather multi-Scale representation from text, video, and audio with shared
Vectors of Locally Aggregated Descriptors to improve unaligned multimodal
sentiment analysis. These shared vectors can be regarded as shared topics to
align different modalities. In addition, we propose a self-supervised shifted
clustering loss to keep the fused feature differentiation among samples. The
backbones are three Transformer encoders corresponding to three modalities, and
the aggregated features generated from the fusion module are feed to a
Transformer plus a full connection to finish task predictions. Experiments on
three popular sentiment analysis benchmarks, IEMOCAP, MOSI, and MOSEI,
demonstrate significant gains over baselines.

    

### [[2112.01372] Hierarchical clustering: visualization, feature importance and model selection](http://arxiv.org/abs/2112.01372)


  We propose methods for the analysis of hierarchical clustering that fully use
the multi-resolution structure provided by a dendrogram. Specifically, we
propose a loss for choosing between clustering methods, a feature importance
score and a graphical tool for visualizing the segmentation of features in a
dendrogram. Current approaches to these tasks lead to loss of information since
they require the user to generate a single partition of the instances by
cutting the dendrogram at a specified level. Our proposed methods, instead, use
the full structure of the dendrogram. The key insight behind the proposed
methods is to view a dendrogram as a phylogeny. This analogy permits the
assignment of a feature value to each internal node of a tree through ancestral
state reconstruction. Real and simulated datasets provide evidence that our
proposed framework has desirable outcomes. We provide an R package that
implements our methods.

    

### [[2112.01375] Flood Analytics Information System (FAIS) Version 4.00 Manual](http://arxiv.org/abs/2112.01375)


  This project was the first attempt to use big data analytics approaches and
machine learning along with Natural Language Processing (NLP) of tweets for
flood risk assessment and decision making. Multiple Python packages were
developed and integrated within the Flood Analytics Information System (FAIS).
FAIS workflow includes the use of IoTs-APIs and various machine learning
approaches for transmitting, processing, and loading big data through which the
application gathers information from various data servers and replicates it to
a data warehouse (IBM database service). Users are allowed to directly stream
and download flood related images/videos from the US Geological Survey (USGS)
and Department of Transportation (DOT) and save the data on a local storage.
The outcome of the river measurement, imagery, and tabular data is displayed on
a web based remote dashboard and the information can be plotted in real-time.
FAIS proved to be a robust and user-friendly tool for flood data analysis at
regional scale that could help stakeholders for rapid assessment of flood
situation and damages. FAIS also provides flood frequency analysis (FFA) to
estimate flood quantiles including the associated uncertainties that combine
the elements of observational analysis, stochastic probability distribution and
design return periods. FAIS is publicly available and deployed on the
Clemson-IBM cloud service.

    

### [[2112.01377] Structural Sieves](http://arxiv.org/abs/2112.01377)


  This paper explores the use of deep neural networks for semiparametric
estimation of economic models of maximizing behavior in production or discrete
choice. We argue that certain deep networks are particularly well suited as a
nonparametric sieve to approximate regression functions that result from
nonlinear latent variable models of continuous or discrete optimization.
Multi-stage models of this type will typically generate rich interaction
effects between regressors ("inputs") in the regression function so that there
may be no plausible separability restrictions on the "reduced-form" mapping
form inputs to outputs to alleviate the curse of dimensionality. Rather,
economic shape, sparsity, or separability restrictions either at a global level
or intermediate stages are usually stated in terms of the latent variable
model. We show that restrictions of this kind are imposed in a more
straightforward manner if a sufficiently flexible version of the latent
variable model is in fact used to approximate the unknown regression function.

    

### [[2112.01387] Generalizing Off-Policy Learning under Sample Selection Bias](http://arxiv.org/abs/2112.01387)


  Learning personalized decision policies that generalize to the target
population is of great relevance. Since training data is often not
representative of the target population, standard policy learning methods may
yield policies that do not generalize target population. To address this
challenge, we propose a novel framework for learning policies that generalize
to the target population. For this, we characterize the difference between the
training data and the target population as a sample selection bias using a
selection variable. Over an uncertainty set around this selection variable, we
optimize the minimax value of a policy to achieve the best worst-case policy
value on the target population. In order to solve the minimax problem, we
derive an efficient algorithm based on a convex-concave procedure and prove
convergence for parametrized spaces of policies such as logistic policies. We
prove that, if the uncertainty set is well-specified, our policies generalize
to the target population as they can not do worse than on the training data.
Using simulated data and a clinical trial, we demonstrate that, compared to
standard policy learning methods, our framework improves the generalizability
of policies substantially.

    

### [[2112.01388] Residual Pathway Priors for Soft Equivariance Constraints](http://arxiv.org/abs/2112.01388)


  There is often a trade-off between building deep learning systems that are
expressive enough to capture the nuances of the reality, and having the right
inductive biases for efficient learning. We introduce Residual Pathway Priors
(RPPs) as a method for converting hard architectural constraints into soft
priors, guiding models towards structured solutions, while retaining the
ability to capture additional complexity. Using RPPs, we construct neural
network priors with inductive biases for equivariances, but without limiting
flexibility. We show that RPPs are resilient to approximate or misspecified
symmetries, and are as effective as fully constrained models even when
symmetries are exact. We showcase the broad applicability of RPPs with
dynamical systems, tabular data, and reinforcement learning. In Mujoco
locomotion tasks, where contact forces and directional rewards violate strict
equivariance assumptions, the RPP outperforms baseline model-free RL agents,
and also improves the learned transition models for model-based RL.

    

### [[2112.01401] Newton methods based convolution neural networks using parallel processing](http://arxiv.org/abs/2112.01401)


  Training of convolutional neural networks is a high dimensional and a
non-convex optimization problem. At present, it is inefficient in situations
where parametric learning rates can not be confidently set. Some past works
have introduced Newton methods for training deep neural networks. Newton
methods for convolutional neural networks involve complicated operations.
Finding the Hessian matrix in second-order methods becomes very complex as we
mainly use the finite differences method with the image data. Newton methods
for convolutional neural networks deals with this by using the sub-sampled
Hessian Newton methods. In this paper, we have used the complete data instead
of the sub-sampled methods that only handle partial data at a time. Further, we
have used parallel processing instead of serial processing in mini-batch
computations. The results obtained using parallel processing in this study,
outperform the time taken by the previous approach.

    

### [[2112.01405] FedRAD: Federated Robust Adaptive Distillation](http://arxiv.org/abs/2112.01405)


  The robustness of federated learning (FL) is vital for the distributed
training of an accurate global model that is shared among large number of
clients. The collaborative learning framework by typically aggregating model
updates is vulnerable to model poisoning attacks from adversarial clients.
Since the shared information between the global server and participants are
only limited to model parameters, it is challenging to detect bad model
updates. Moreover, real-world datasets are usually heterogeneous and not
independent and identically distributed (Non-IID) among participants, which
makes the design of such robust FL pipeline more difficult. In this work, we
propose a novel robust aggregation method, Federated Robust Adaptive
Distillation (FedRAD), to detect adversaries and robustly aggregate local
models based on properties of the median statistic, and then performing an
adapted version of ensemble Knowledge Distillation. We run extensive
experiments to evaluate the proposed method against recently published works.
The results show that FedRAD outperforms all other aggregators in the presence
of adversaries, as well as in heterogeneous data distributions.

    

### [[2112.01406] Active Learning for Domain Adaptation: An Energy-based Approach](http://arxiv.org/abs/2112.01406)


  Unsupervised domain adaptation has recently emerged as an effective paradigm
for generalizing deep neural networks to new target domains. However, there is
still enormous potential to be tapped to reach the fully supervised
performance. In this paper, we present a novel active learning strategy to
assist knowledge transfer in the target domain, dubbed active domain
adaptation. We start from an observation that energy-based models exhibit free
energy biases when training (source) and test (target) data come from different
distributions. Inspired by this inherent mechanism, we empirically reveal that
a simple yet efficient energy-based sampling strategy sheds light on selecting
the most valuable target samples than existing approaches requiring particular
architectures or computation of the distances. Our algorithm, Energy-based
Active Domain Adaptation (EADA), queries groups of targe data that incorporate
both domain characteristic and instance uncertainty into every selection round.
Meanwhile, by aligning the free energy of target data compact around the source
domain via a regularization term, domain gap can be implicitly diminished.
Through extensive experiments, we show that EADA surpasses state-of-the-art
methods on well-known challenging benchmarks with substantial improvements,
making it a useful option in the open world. Code is available at
this https URL.

    

### [[2112.01421] Deep residential representations: Using unsupervised learning to unlock elevation data for geo-demographic prediction](http://arxiv.org/abs/2112.01421)


  LiDAR (short for "Light Detection And Ranging" or "Laser Imaging, Detection,
And Ranging") technology can be used to provide detailed three-dimensional
elevation maps of urban and rural landscapes. To date, airborne LiDAR imaging
has been predominantly confined to the environmental and archaeological
domains. However, the geographically granular and open-source nature of this
data also lends itself to an array of societal, organizational and business
applications where geo-demographic type data is utilised. Arguably, the
complexity involved in processing this multi-dimensional data has thus far
restricted its broader adoption. In this paper, we propose a series of
convenient task-agnostic tile elevation embeddings to address this challenge,
using recent advances from unsupervised Deep Learning. We test the potential of
our embeddings by predicting seven English indices of deprivation (2019) for
small geographies in the Greater London area. These indices cover a range of
socio-economic outcomes and serve as a proxy for a wide variety of downstream
tasks to which the embeddings can be applied. We consider the suitability of
this data not just on its own but also as an auxiliary source of data in
combination with demographic features, thus providing a realistic use case for
the embeddings. Having trialled various model/embedding configurations, we find
that our best performing embeddings lead to Root-Mean-Squared-Error (RMSE)
improvements of up to 21% over using standard demographic features alone. We
also demonstrate how our embedding pipeline, using Deep Learning combined with
K-means clustering, produces coherent tile segments which allow the latent
embedding features to be interpreted.

    

### [[2112.01423] Training Efficiency and Robustness in Deep Learning](http://arxiv.org/abs/2112.01423)


  Deep Learning has revolutionized machine learning and artificial
intelligence, achieving superhuman performance in several standard benchmarks.
It is well-known that deep learning models are inefficient to train; they learn
by processing millions of training data multiple times and require powerful
computational resources to process large batches of data in parallel at the
same time rather than sequentially. Deep learning models also have unexpected
failure modes; they can be fooled into misbehaviour, producing unexpectedly
incorrect predictions.
In this thesis, we study approaches to improve the training efficiency and
robustness of deep learning models. In the context of learning visual-semantic
embeddings, we find that prioritizing learning on more informative training
data increases convergence speed and improves generalization performance on
test data. We formalize a simple trick called hard negative mining as a
modification to the learning objective function with no computational overhead.
Next, we seek improvements to optimization speed in general-purpose
optimization methods in deep learning. We show that a redundancy-aware
modification to the sampling of training data improves the training speed and
develops an efficient method for detecting the diversity of training signal,
namely, gradient clustering. Finally, we study adversarial robustness in deep
learning and approaches to achieve maximal adversarial robustness without
training with additional data. For linear models, we prove guaranteed maximal
robustness achieved only by appropriate choice of the optimizer,
regularization, or architecture.

    

### [[2112.01433] Loss Landscape Dependent Self-Adjusting Learning Rates in Decentralized Stochastic Gradient Descent](http://arxiv.org/abs/2112.01433)


  Distributed Deep Learning (DDL) is essential for large-scale Deep Learning
(DL) training. Synchronous Stochastic Gradient Descent (SSGD) 1 is the de facto
DDL optimization method. Using a sufficiently large batch size is critical to
achieving DDL runtime speedup. In a large batch setting, the learning rate must
be increased to compensate for the reduced number of parameter updates.
However, a large learning rate may harm convergence in SSGD and training could
easily diverge. Recently, Decentralized Parallel SGD (DPSGD) has been proposed
to improve distributed training speed. In this paper, we find that DPSGD not
only has a system-wise run-time benefit but also a significant convergence
benefit over SSGD in the large batch setting. Based on a detailed analysis of
the DPSGD learning dynamics, we find that DPSGD introduces additional
landscape-dependent noise that automatically adjusts the effective learning
rate to improve convergence. In addition, we theoretically show that this noise
smoothes the loss landscape, hence allowing a larger learning rate. We conduct
extensive studies over 18 state-of-the-art DL models/tasks and demonstrate that
DPSGD often converges in cases where SSGD diverges for large learning rates in
the large batch setting. Our findings are consistent across two different
application domains: Computer Vision (CIFAR10 and ImageNet-1K) and Automatic
Speech Recognition (SWB300 and SWB2000), and two different types of neural
network models: Convolutional Neural Networks and Long Short-Term Memory
Recurrent Neural Networks.

    

### [[2112.01438] Level set learning with pseudo-reversible neural networks for nonlinear dimension reduction in function approximation](http://arxiv.org/abs/2112.01438)


  Due to the curse of dimensionality and the limitation on training data,
approximating high-dimensional functions is a very challenging task even for
powerful deep neural networks. Inspired by the Nonlinear Level set Learning
(NLL) method that uses the reversible residual network (RevNet), in this paper
we propose a new method of Dimension Reduction via Learning Level Sets (DRiLLS)
for function approximation. Our method contains two major components: one is
the pseudo-reversible neural network (PRNN) module that effectively transforms
high-dimensional input variables to low-dimensional active variables, and the
other is the synthesized regression module for approximating function values
based on the transformed data in the low-dimensional space. The PRNN not only
relaxes the invertibility constraint of the nonlinear transformation present in
the NLL method due to the use of RevNet, but also adaptively weights the
influence of each sample and controls the sensitivity of the function to the
learned active variables. The synthesized regression uses Euclidean distance in
the input space to select neighboring samples, whose projections on the space
of active variables are used to perform local least-squares polynomial fitting.
This helps to resolve numerical oscillation issues present in traditional local
and global regressions. Extensive experimental results demonstrate that our
DRiLLS method outperforms both the NLL and Active Subspace methods, especially
when the target function possesses critical points in the interior of its input
domain.

    

### [[2112.01452] Indexed Minimum Empirical Divergence for Unimodal Bandits](http://arxiv.org/abs/2112.01452)


  We consider a multi-armed bandit problem specified by a set of
one-dimensional family exponential distributions endowed with a unimodal
structure. We introduce IMED-UB, a algorithm that optimally exploits the
unimodal-structure, by adapting to this setting the Indexed Minimum Empirical
Divergence (IMED) algorithm introduced by Honda and Takemura [2015]. Owing to
our proof technique, we are able to provide a concise finite-time analysis of
IMED-UB algorithm. Numerical experiments show that IMED-UB competes with the
state-of-the-art algorithms.

    

### [[2112.01453] Target Propagation via Regularized Inversion](http://arxiv.org/abs/2112.01453)


  Target Propagation (TP) algorithms compute targets instead of gradients along
neural networks and propagate them backward in a way that is similar yet
different than gradient back-propagation (BP). The idea was first presented as
a perturbative alternative to back-propagation that may achieve greater
accuracy in gradient evaluation when training multi-layer neural networks
(LeCun et al., 1989). However, TP has remained more of a template algorithm
with many variations than a well-identified algorithm. Revisiting insights of
LeCun et al., (1989) and more recently of Lee et al. (2015), we present a
simple version of target propagation based on regularized inversion of network
layers, easily implementable in a differentiable programming framework. We
compare its computational complexity to the one of BP and delineate the regimes
in which TP can be attractive compared to BP. We show how our TP can be used to
train recurrent neural networks with long sequences on various sequence
modeling problems. The experimental results underscore the importance of
regularization in TP in practice.

    

### [[2112.01455] Zero-Shot Text-Guided Object Generation with Dream Fields](http://arxiv.org/abs/2112.01455)


  We combine neural rendering with multi-modal image and text representations
to synthesize diverse 3D objects solely from natural language descriptions. Our
method, Dream Fields, can generate the geometry and color of a wide range of
objects without 3D supervision. Due to the scarcity of diverse, captioned 3D
data, prior methods only generate objects from a handful of categories, such as
ShapeNet. Instead, we guide generation with image-text models pre-trained on
large datasets of captioned images from the web. Our method optimizes a Neural
Radiance Field from many camera views so that rendered images score highly with
a target caption according to a pre-trained CLIP model. To improve fidelity and
visual quality, we introduce simple geometric priors, including
sparsity-inducing transmittance regularization, scene bounds, and new MLP
architectures. In experiments, Dream Fields produce realistic, multi-view
consistent object geometry and color from a variety of natural language
captions.

    

### [[2112.01475] A Hybrid Science-Guided Machine Learning Approach for Modeling and Optimizing Chemical Processes](http://arxiv.org/abs/2112.01475)


  This study presents a broad perspective of hybrid process modeling and
optimization combining the scientific knowledge and data analytics in
bioprocessing and chemical engineering with a science-guided machine learning
(SGML) approach. We divide the approach into two major categories. The first
refers to the case where a data-based ML model compliments and makes the
first-principle science-based model more accurate in prediction, and the second
corresponds to the case where scientific knowledge helps make the ML model more
scientifically consistent. We present a detailed review of scientific and
engineering literature relating to the hybrid SGML approach, and propose a
systematic classification of hybrid SGML models. For applying ML to improve
science-based models, we present expositions of the sub-categories of direct
serial and parallel hybrid modeling and their combinations, inverse modeling,
reduced-order modeling, quantifying uncertainty in the process and even
discovering governing equations of the process model. For applying scientific
principles to improve ML models, we discuss the sub-categories of
science-guided design, learning and refinement. For each sub-category, we
identify its requirements, advantages and limitations, together with their
published and potential areas of applications in bioprocessing and chemical
engineering.

    

### [[2112.01477] Why Calibration Error is Wrong Given Model Uncertainty: Using Posterior Predictive Checks with Deep Learning](http://arxiv.org/abs/2112.01477)


  Within the last few years, there has been a move towards using statistical
models in conjunction with neural networks with the end goal of being able to
better answer the question, "what do our models know?". From this trend,
classical metrics such as Prediction Interval Coverage Probability (PICP) and
new metrics such as calibration error have entered the general repertoire of
model evaluation in order to gain better insight into how the uncertainty of
our model compares to reality. One important component of uncertainty modeling
is model uncertainty (epistemic uncertainty), a measurement of what the model
does and does not know. However, current evaluation techniques tends to
conflate model uncertainty with aleatoric uncertainty (irreducible error),
leading to incorrect conclusions. In this paper, using posterior predictive
checks, we show how calibration error and its variants are almost always
incorrect to use given model uncertainty, and further show how this mistake can
lead to trust in bad models and mistrust in good models. Though posterior
predictive checks has often been used for in-sample evaluation of Bayesian
models, we show it still has an important place in the modern deep learning
world.

    

### [[2112.01484] Safe Reinforcement Learning for Grid Voltage Control](http://arxiv.org/abs/2112.01484)


  Under voltage load shedding has been considered as a standard approach to
recover the voltage stability of the electric power grid under emergency
conditions, yet this scheme usually trips a massive amount of load
inefficiently. Reinforcement learning (RL) has been adopted as a promising
approach to circumvent the issues; however, RL approach usually cannot
guarantee the safety of the systems under control. In this paper, we discuss a
couple of novel safe RL approaches, namely constrained optimization approach
and Barrier function-based approach, that can safely recover voltage under
emergency events. This method is general and can be applied to other
safety-critical control problems. Numerical simulations on the 39-bus IEEE
benchmark are performed to demonstrate the effectiveness of the proposed safe
RL emergency control.

    

### [[2112.01496] Analysis of an adaptive lead weighted ResNet for multiclass classification of 12-lead ECGs](http://arxiv.org/abs/2112.01496)


  Background: Twelve lead ECGs are a core diagnostic tool for cardiovascular
diseases. Here, we describe and analyse an ensemble deep neural network
architecture to classify 24 cardiac abnormalities from 12-lead ECGs.
Method: We proposed a squeeze and excite ResNet to automatically learn deep
features from 12-lead ECGs, in order to identify 24 cardiac conditions. The
deep features were augmented with age and gender features in the final fully
connected layers. Output thresholds for each class were set using a constrained
grid search. To determine why the model made incorrect predictions, two expert
clinicians independently interpreted a random set of 100 misclassified ECGs
concerning Left Axis Deviation.
Results: Using the bespoke weighted accuracy metric, we achieved a 5-fold
cross validation score of 0.684, and sensitivity and specificity of 0.758 and
0.969, respectively. We scored 0.520 on the full test data, and ranked 2nd out
of 41 in the official challenge rankings. On a random set of misclassified
ECGs, agreement between two clinicians and training labels was poor (clinician
1: kappa = -0.057, clinician 2: kappa = -0.159). In contrast, agreement between
the clinicians was very high (kappa = 0.92).
Discussion: The proposed prediction model performed well on the validation
and hidden test data in comparison to models trained on the same data. We also
discovered considerable inconsistency in training labels, which is likely to
hinder development of more accurate models.

    

### [[2112.01503] Machine Learning-Based Classification Algorithms for the Prediction of Coronary Heart Diseases](http://arxiv.org/abs/2112.01503)


  Coronary heart disease, which is a form of cardiovascular disease (CVD), is
the leading cause of death worldwide. The odds of survival are good if it is
found or diagnosed early. The current report discusses a comparative approach
to the classification of coronary heart disease datasets using machine learning
(ML) algorithms. The current study created and tested several
machine-learning-based classification models. The dataset was subjected to
Smote to handle unbalanced classes and feature selection technique in order to
assess the impact on two distinct performance metrics. The results show that
logistic regression produced the highest performance score on the original
dataset compared to the other algorithms employed. In conclusion, this study
suggests that LR on a well-processed and standardized dataset can predict
coronary heart disease with greater accuracy than the other algorithms.

    

### [[2112.01506] Sample Complexity of Robust Reinforcement Learning with a Generative Model](http://arxiv.org/abs/2112.01506)


  The Robust Markov Decision Process (RMDP) framework focuses on designing
control policies that are robust against the parameter uncertainties due to the
mismatches between the simulator model and real-world settings. An RMDP problem
is typically formulated as a max-min problem, where the objective is to find
the policy that maximizes the value function for the worst possible model that
lies in an uncertainty set around a nominal model. The standard robust dynamic
programming approach requires the knowledge of the nominal model for computing
the optimal robust policy. In this work, we propose a model-based reinforcement
learning (RL) algorithm for learning an $\epsilon$-optimal robust policy when
the nominal model is unknown. We consider three different forms of uncertainty
sets, characterized by the total variation distance, chi-square divergence, and
KL divergence. For each of these uncertainty sets, we give a precise
characterization of the sample complexity of our proposed algorithm. In
addition to the sample complexity results, we also present a formal analytical
argument on the benefit of using robust policies. Finally, we demonstrate the
performance of our algorithm on two benchmark problems.

    

### [[2112.01511] The Surprising Effectiveness of Representation Learning for Visual Imitation](http://arxiv.org/abs/2112.01511)


  While visual imitation learning offers one of the most effective ways of
learning from visual demonstrations, generalizing from them requires either
hundreds of diverse demonstrations, task specific priors, or large,
hard-to-train parametric models. One reason such complexities arise is because
standard visual imitation frameworks try to solve two coupled problems at once:
learning a succinct but good representation from the diverse visual data, while
simultaneously learning to associate the demonstrated actions with such
representations. Such joint learning causes an interdependence between these
two problems, which often results in needing large amounts of demonstrations
for learning. To address this challenge, we instead propose to decouple
representation learning from behavior learning for visual imitation. First, we
learn a visual representation encoder from offline data using standard
supervised and self-supervised learning methods. Once the representations are
trained, we use non-parametric Locally Weighted Regression to predict the
actions. We experimentally show that this simple decoupling improves the
performance of visual imitation models on both offline demonstration datasets
and real-robot door opening compared to prior work in visual imitation. All of
our generated data, code, and robot videos are publicly available at
this https URL.

    

### [[2112.01518] DenseCLIP: Language-Guided Dense Prediction with Context-Aware Prompting](http://arxiv.org/abs/2112.01518)


  Recent progress has shown that large-scale pre-training using contrastive
image-text pairs can be a promising alternative for high-quality visual
representation learning from natural language supervision. Benefiting from a
broader source of supervision, this new paradigm exhibits impressive
transferability to downstream classification tasks and datasets. However, the
problem of transferring the knowledge learned from image-text pairs to more
complex dense prediction tasks has barely been visited. In this work, we
present a new framework for dense prediction by implicitly and explicitly
leveraging the pre-trained knowledge from CLIP. Specifically, we convert the
original image-text matching problem in CLIP to a pixel-text matching problem
and use the pixel-text score maps to guide the learning of dense prediction
models. By further using the contextual information from the image to prompt
the language model, we are able to facilitate our model to better exploit the
pre-trained knowledge. Our method is model-agnostic, which can be applied to
arbitrary dense prediction systems and various pre-trained visual backbones
including both CLIP models and ImageNet pre-trained models. Extensive
experiments demonstrate the superior performance of our methods on semantic
segmentation, object detection, and instance segmentation tasks. Code is
available at this https URL


### [[2112.01521] Object-aware Monocular Depth Prediction with Instance Convolutions](http://arxiv.org/abs/2112.01521)


  With the advent of deep learning, estimating depth from a single RGB image
has recently received a lot of attention, being capable of empowering many
different applications ranging from path planning for robotics to computational
cinematography. Nevertheless, while the depth maps are in their entirety fairly
reliable, the estimates around object discontinuities are still far from
satisfactory. This can be contributed to the fact that the convolutional
operator naturally aggregates features across object discontinuities, resulting
in smooth transitions rather than clear boundaries. Therefore, in order to
circumvent this issue, we propose a novel convolutional operator which is
explicitly tailored to avoid feature aggregation of different object parts. In
particular, our method is based on estimating per-part depth values by means of
superpixels. The proposed convolutional operator, which we dub "Instance
Convolution", then only considers each object part individually on the basis of
the estimated superpixels. Our evaluation with respect to the NYUv2 as well as
the iBims dataset clearly demonstrates the superiority of Instance Convolutions
over the classical convolution at estimating depth around occlusion boundaries,
while producing comparable results elsewhere. Code will be made publicly
available upon acceptance.

    

### [[2112.01524] GLAMR: Global Occlusion-Aware Human Mesh Recovery with Dynamic Cameras](http://arxiv.org/abs/2112.01524)


  We present an approach for 3D global human mesh recovery from monocular
videos recorded with dynamic cameras. Our approach is robust to severe and
long-term occlusions and tracks human bodies even when they go outside the
camera's field of view. To achieve this, we first propose a deep generative
motion infiller, which autoregressively infills the body motions of occluded
humans based on visible motions. Additionally, in contrast to prior work, our
approach reconstructs human meshes in consistent global coordinates even with
dynamic cameras. Since the joint reconstruction of human motions and camera
poses is underconstrained, we propose a global trajectory predictor that
generates global human trajectories based on local body movements. Using the
predicted trajectories as anchors, we present a global optimization framework
that refines the predicted trajectories and optimizes the camera poses to match
the video evidence such as 2D keypoints. Experiments on challenging indoor and
in-the-wild datasets with dynamic cameras demonstrate that the proposed
approach outperforms prior methods significantly in terms of motion infilling
and global mesh recovery.

    

### [[2112.01525] Co-domain Symmetry for Complex-Valued Deep Learning](http://arxiv.org/abs/2112.01525)


  We study complex-valued scaling as a type of symmetry natural and unique to
complex-valued measurements and representations. Deep Complex Networks (DCN)
extends real-valued algebra to the complex domain without addressing
complex-valued scaling. SurReal takes a restrictive manifold view of complex
numbers, adopting a distance metric to achieve complex-scaling invariance while
losing rich complex-valued information. We analyze complex-valued scaling as a
co-domain transformation and design novel equivariant and invariant neural
network layer functions for this special transformation. We also propose novel
complex-valued representations of RGB images, where complex-valued scaling
indicates hue shift or correlated changes across color channels. Benchmarked on
MSTAR, CIFAR10, CIFAR100, and SVHN, our co-domain symmetric (CDS) classifiers
deliver higher accuracy, better generalization, robustness to co-domain
transformations, and lower model bias and variance than DCN and SurReal with
far fewer parameters.

    

### [[2112.01527] Masked-attention Mask Transformer for Universal Image Segmentation](http://arxiv.org/abs/2112.01527)


  Image segmentation is about grouping pixels with different semantics, e.g.,
category or instance membership, where each choice of semantics defines a task.
While only the semantics of each task differ, current research focuses on
designing specialized architectures for each task. We present Masked-attention
Mask Transformer (Mask2Former), a new architecture capable of addressing any
image segmentation task (panoptic, instance or semantic). Its key components
include masked attention, which extracts localized features by constraining
cross-attention within predicted mask regions. In addition to reducing the
research effort by at least three times, it outperforms the best specialized
architectures by a significant margin on four popular datasets. Most notably,
Mask2Former sets a new state-of-the-art for panoptic segmentation (57.8 PQ on
COCO), instance segmentation (50.1 AP on COCO) and semantic segmentation (57.7
mIoU on ADE20K).

    

### [[2112.01528] A Fast Knowledge Distillation Framework for Visual Recognition](http://arxiv.org/abs/2112.01528)


  While Knowledge Distillation (KD) has been recognized as a useful tool in
many visual tasks, such as supervised classification and self-supervised
representation learning, the main drawback of a vanilla KD framework is its
mechanism, which consumes the majority of the computational overhead on
forwarding through the giant teacher networks, making the entire learning
procedure inefficient and costly. ReLabel, a recently proposed solution,
suggests creating a label map for the entire image. During training, it
receives the cropped region-level label by RoI aligning on a pre-generated
entire label map, allowing for efficient supervision generation without having
to pass through the teachers many times. However, as the KD teachers are from
conventional multi-crop training, there are various mismatches between the
global label-map and region-level label in this technique, resulting in
performance deterioration. In this study, we present a Fast Knowledge
Distillation (FKD) framework that replicates the distillation training phase
and generates soft labels using the multi-crop KD approach, while training
faster than ReLabel since no post-processes such as RoI align and softmax
operations are used. When conducting multi-crop in the same image for data
loading, our FKD is even more efficient than the traditional image
classification framework. On ImageNet-1K, we obtain 79.8% with ResNet-50,
outperforming ReLabel by ~1.0% while being faster. On the self-supervised
learning task, we also show that FKD has an efficiency advantage. Our project
page: this http URL, source code and models
are available at: this https URL.

    

### [[2112.01529] BEVT: BERT Pretraining of Video Transformers](http://arxiv.org/abs/2112.01529)


  This paper studies the BERT pretraining of video transformers. It is a
straightforward but worth-studying extension given the recent success from BERT
pretraining of image transformers. We introduce BEVT which decouples video
representation learning into spatial representation learning and temporal
dynamics learning. In particular, BEVT first performs masked image modeling on
image data, and then conducts masked image modeling jointly with masked video
modeling on video data. This design is motivated by two observations: 1)
transformers learned on image datasets provide decent spatial priors that can
ease the learning of video transformers, which are often times
computationally-intensive if trained from scratch; 2) discriminative clues,
i.e., spatial and temporal information, needed to make correct predictions vary
among different videos due to large intra-class and inter-class variations. We
conduct extensive experiments on three challenging video benchmarks where BEVT
achieves very promising results. On Kinetics 400, for which recognition mostly
relies on discriminative spatial representations, BEVT achieves comparable
results to strong supervised baselines. On Something-Something-V2 and Diving
48, which contain videos relying on temporal dynamics, BEVT outperforms by
clear margins all alternative baselines and achieves state-of-the-art
performance with a 70.6% and 86.7% Top-1 accuracy respectively.

    

### [[2112.01531] The MAIEI Learning Community Report](http://arxiv.org/abs/2112.01531)


  This is a labor of the Learning Community cohort that was convened by MAIEI
in Winter 2021 to work through and discuss important research issues in the
field of AI ethics from a multidisciplinary lens. The community came together
supported by facilitators from the MAIEI staff to vigorously debate and explore
the nuances of issues like bias, privacy, disinformation, accountability, and
more especially examining them from the perspective of industry, civil society,
academia, and government.
The outcome of these discussions is reflected in the report that you are
reading now - an exploration of a variety of issues with deep-dive, critical
commentary on what has been done, what worked and what didn't, and what remains
to be done so that we can meaningfully move forward in addressing the societal
challenges posed by the deployment of AI systems.
The chapters titled "Design and Techno-isolationism", "Facebook and the
Digital Divide: Perspectives from Myanmar, Mexico, and India", "Future of
Work", and "Media & Communications & Ethical Foresight" will hopefully provide
with you novel lenses to explore this domain beyond the usual tropes that are
covered in the domain of AI ethics.

    

### [[1703.00443] OptNet: Differentiable Optimization as a Layer in Neural Networks](http://arxiv.org/abs/1703.00443)


  This paper presents OptNet, a network architecture that integrates
optimization problems (here, specifically in the form of quadratic programs) as
individual layers in larger end-to-end trainable deep networks. These layers
encode constraints and complex dependencies between the hidden states that
traditional convolutional and fully-connected layers often cannot capture. We
explore the foundations for such an architecture: we show how techniques from
sensitivity analysis, bilevel optimization, and implicit differentiation can be
used to exactly differentiate through these layers and with respect to layer
parameters; we develop a highly efficient solver for these layers that exploits
fast GPU-based batch solves within a primal-dual interior point method, and
which provides backpropagation gradients with virtually no additional cost on
top of the solve; and we highlight the application of these approaches in
several problems. In one notable example, the method is learns to play
mini-Sudoku (4x4) given just input and output games, with no a-priori
information about the rules of the game; this highlights the ability of OptNet
to learn hard constraints better than other neural architectures.

    

### [[1803.09010] Datasheets for Datasets](http://arxiv.org/abs/1803.09010)


  The machine learning community currently has no standardized process for
documenting datasets, which can lead to severe consequences in high-stakes
domains. To address this gap, we propose datasheets for datasets. In the
electronics industry, every component, no matter how simple or complex, is
accompanied with a datasheet that describes its operating characteristics, test
results, recommended uses, and other information. By analogy, we propose that
every dataset be accompanied with a datasheet that documents its motivation,
composition, collection process, recommended uses, and so on. Datasheets for
datasets will facilitate better communication between dataset creators and
dataset consumers, and encourage the machine learning community to prioritize
transparency and accountability.

    

### [[1901.10002] A Framework for Understanding Sources of Harm throughout the Machine Learning Life Cycle](http://arxiv.org/abs/1901.10002)


  As machine learning (ML) increasingly affects people and society, awareness
of its potential unwanted consequences has also grown. To anticipate, prevent,
and mitigate undesirable downstream consequences, it is critical that we
understand when and how harm might be introduced throughout the ML life cycle.
In this paper, we provide a framework that identifies seven distinct potential
sources of downstream harm in machine learning, spanning data collection,
development, and deployment. In doing so, we aim to facilitate more productive
and precise communication around these issues, as well as more direct,
application-grounded ways to mitigate them.

    

### [[1902.00468] Multilevel Monte Carlo Variational Inference](http://arxiv.org/abs/1902.00468)


  We propose a variance reduction framework for variational inference using the
Multilevel Monte Carlo (MLMC) method. Our framework is built on reparameterized
gradient estimators and "recycles" parameters obtained from past update history
in optimization. In addition, our framework provides a new optimization
algorithm based on stochastic gradient descent (SGD) that adaptively estimates
the sample size used for gradient estimation according to the ratio of the
gradient variance. We theoretically show that, with our method, the variance of
the gradient estimator decreases as optimization proceeds and that a learning
rate scheduler function helps improve the convergence. We also show that, in
terms of the \textit{signal-to-noise} ratio, our method can improve the quality
of gradient estimation by the learning rate scheduler function without
increasing the initial sample size. Finally, we confirm that our method
achieves faster convergence and reduces the variance of the gradient estimator
compared with other methods through experimental comparisons with baseline
methods using several benchmark datasets.

    

### [[1906.08386] Inherent Tradeoffs in Learning Fair Representations](http://arxiv.org/abs/1906.08386)


  Real-world applications of machine learning tools in high-stakes domains are
often regulated to be fair, in the sense that the predicted target should
satisfy some quantitative notion of parity with respect to a protected
attribute. However, the exact tradeoff between fairness and accuracy is not
entirely clear, even for the basic paradigm of classification problems. In this
paper, we characterize an inherent tradeoff between statistical parity and
accuracy in the classification setting by providing a lower bound on the sum of
group-wise errors of any fair classifiers. Our impossibility theorem could be
interpreted as a certain uncertainty principle in fairness: if the base rates
differ among groups, then any fair classifier satisfying statistical parity has
to incur a large error on at least one of the groups. We further extend this
result to give a lower bound on the joint error of any (approximately) fair
classifiers, from the perspective of learning fair representations. To show
that our lower bound is tight, assuming oracle access to Bayes (potentially
unfair) classifiers, we also construct an algorithm that returns a randomized
classifier which is both optimal and fair. Interestingly, when the protected
attribute can take more than two values, an extension of this lower bound does
not admit an analytic solution. Nevertheless, in this case, we show that the
lower bound can be efficiently computed by solving a linear program, which we
term as the TV-Barycenter problem, a barycenter problem under the TV-distance.
On the upside, we prove that if the group-wise Bayes optimal classifiers are
close, then learning fair representations leads to an alternative notion of
fairness, known as the accuracy parity, which states that the error rates are
close between groups. Finally, we also conduct experiments on real-world
datasets to confirm our theoretical findings.

    

### [[1910.08293] ALOHA: Artificial Learning of Human Attributes for Dialogue Agents](http://arxiv.org/abs/1910.08293)


  For conversational AI and virtual assistants to communicate with humans in a
realistic way, they must exhibit human characteristics such as expression of
emotion and personality. Current attempts toward constructing human-like
dialogue agents have presented significant difficulties. We propose Human Level
Attributes (HLAs) based on tropes as the basis of a method for learning
dialogue agents that can imitate the personalities of fictional characters.
Tropes are characteristics of fictional personalities that are observed
recurrently and determined by viewers' impressions. By combining detailed HLA
data with dialogue data for specific characters, we present a dataset,
HLA-Chat, that models character profiles and gives dialogue agents the ability
to learn characters' language styles through their HLAs. We then introduce a
three-component system, ALOHA (which stands for Artificial Learning of Human
Attributes), that combines character space mapping, character community
detection, and language style retrieval to build a character (or personality)
specific language model. Our preliminary experiments demonstrate that two
variations of ALOHA, combined with our proposed dataset, can outperform
baseline models at identifying the correct dialogue responses of chosen target
characters, and are stable regardless of the character's identity, the genre of
the show, and the context of the dialogue.

    

### [[2001.07641] Deceptive AI Explanations: Creation and Detection](http://arxiv.org/abs/2001.07641)


  Artificial intelligence (AI) comes with great opportunities but can also pose
significant risks. Automatically generated explanations for decisions can
increase transparency and foster trust, especially for systems based on
automated predictions by AI models. However, given, e.g., economic incentives
to create dishonest AI, to what extent can we trust explanations? To address
this issue, our work investigates how AI models (i.e., deep learning, and
existing instruments to increase transparency regarding AI decisions) can be
used to create and detect deceptive explanations. As an empirical evaluation,
we focus on text classification and alter the explanations generated by
GradCAM, a well-established explanation technique in neural networks. Then, we
evaluate the effect of deceptive explanations on users in an experiment with
200 participants. Our findings confirm that deceptive explanations can indeed
fool humans. However, one can deploy machine learning (ML) methods to detect
seemingly minor deception attempts with accuracy exceeding 80% given sufficient
domain knowledge. Without domain knowledge, one can still infer inconsistencies
in the explanations in an unsupervised manner, given basic knowledge of the
predictive model under scrutiny.

    

### [[2005.00777] Deep Feature Mining via Attention-based BiLSTM-GCN for Human Motor Imagery Recognition](http://arxiv.org/abs/2005.00777)


  Recognition accuracy and response time are both critically essential ahead of
building practical electroencephalography (EEG) based brain-computer interface
(BCI). Recent approaches, however, have either compromised in the
classification accuracy or responding time. This paper presents a novel deep
learning approach designed towards remarkably accurate and responsive motor
imagery (MI) recognition based on scalp EEG. Bidirectional Long Short-term
Memory (BiLSTM) with the Attention mechanism manages to derive relevant
features from raw EEG signals. The connected graph convolutional neural network
(GCN) promotes the decoding performance by cooperating with the topological
structure of features, which are estimated from the overall data. The
0.4-second detection framework has shown effective and efficient prediction
based on individual and group-wise training, with 98.81% and 94.64% accuracy,
respectively, which outperformed all the state-of-the-art studies. The
introduced deep feature mining approach can precisely recognize human motion
intents from raw EEG signals, which paves the road to translate the EEG based
MI recognition to practical BCI systems.

    

### [[2005.08334] Marginal likelihood computation for model selection and hypothesis testing: an extensive review](http://arxiv.org/abs/2005.08334)


  This is an up-to-date introduction to, and overview of, marginal likelihood
computation for model selection and hypothesis testing. Computing normalizing
constants of probability models (or ratio of constants) is a fundamental issue
in many applications in statistics, applied mathematics, signal processing and
machine learning. This article provides a comprehensive study of the
state-of-the-art of the topic. We highlight limitations, benefits, connections
and differences among the different techniques. Problems and possible solutions
with the use of improper priors are also described. Some of the most relevant
methodologies are compared through theoretical comparisons and numerical
experiments.

    

### [[2006.05371] Bayesian Probabilistic Numerical Integration with Tree-Based Models](http://arxiv.org/abs/2006.05371)


  Bayesian quadrature (BQ) is a method for solving numerical integration
problems in a Bayesian manner, which allows users to quantify their uncertainty
about the solution. The standard approach to BQ is based on a Gaussian process
(GP) approximation of the integrand. As a result, BQ is inherently limited to
cases where GP approximations can be done in an efficient manner, thus often
prohibiting very high-dimensional or non-smooth target functions. This paper
proposes to tackle this issue with a new Bayesian numerical integration
algorithm based on Bayesian Additive Regression Trees (BART) priors, which we
call BART-Int. BART priors are easy to tune and well-suited for discontinuous
functions. We demonstrate that they also lend themselves naturally to a
sequential design setting and that explicit convergence rates can be obtained
in a variety of settings. The advantages and disadvantages of this new
methodology are highlighted on a set of benchmark tests including the Genz
functions, and on a Bayesian survey design problem.

    

### [[2006.08924] GCNs-Net: A Graph Convolutional Neural Network Approach for Decoding Time-resolved EEG Motor Imagery Signals](http://arxiv.org/abs/2006.08924)


  Towards developing effective and efficient brain-computer interface (BCI)
systems, precise decoding of brain activity measured by electroencephalogram
(EEG), is highly demanded. Traditional works classify EEG signals without
considering the topological relationship among electrodes. However,
neuroscience research has increasingly emphasized network patterns of brain
dynamics. Thus, the Euclidean structure of electrodes might not adequately
reflect the interaction between signals. To fill the gap, a novel deep learning
framework based on the graph convolutional neural networks (GCNs) was presented
to enhance the decoding performance of raw EEG signals during different types
of motor imagery (MI) tasks while cooperating with the functional topological
relationship of electrodes. Based on the absolute Pearson's matrix of overall
signals, the graph Laplacian of EEG electrodes was built up. The GCNs-Net
constructed by graph convolutional layers learns the generalized features. The
followed pooling layers reduce dimensionality, and the fully-connected softmax
layer derives the final prediction. The introduced approach has been shown to
converge for both personalized and group-wise predictions. It has achieved the
highest averaged accuracy, 93.056% and 88.57% (PhysioNet Dataset), 96.24% and
80.89% (High Gamma Dataset), at the subject and group level, respectively,
compared with existing studies, which suggests adaptability and robustness to
individual variability. Moreover, the performance was stably reproducible among
repetitive experiments for cross-validation. To conclude, the GCNs-Net filters
EEG signals based on the functional topological relationship, which manages to
decode relevant features for brain motor imagery.

    

### [[2007.03800] Efficient and Parallel Separable Dictionary Learning](http://arxiv.org/abs/2007.03800)


  Separable, or Kronecker product, dictionaries provide natural decompositions
for 2D signals, such as images. In this paper, we describe a highly
parallelizable algorithm that learns such dictionaries which reaches sparse
representations competitive with the previous state of the art dictionary
learning algorithms from the literature but at a lower computational cost. We
highlight the performance of the proposed method to sparsely represent image
and hyperspectral data, and for image denoising.

    

### [[2008.09052] On transversality of bent hyperplane arrangements and the topological expressiveness of ReLU neural networks](http://arxiv.org/abs/2008.09052)


  Let F:R^n -> R be a feedforward ReLU neural network. It is well-known that
for any choice of parameters, F is continuous and piecewise (affine) linear. We
lay some foundations for a systematic investigation of how the architecture of
F impacts the geometry and topology of its possible decision regions for binary
classification tasks. Following the classical progression for smooth functions
in differential topology, we first define the notion of a generic, transversal
ReLU neural network and show that almost all ReLU networks are generic and
transversal. We then define a partially-oriented linear 1-complex in the domain
of F and identify properties of this complex that yield an obstruction to the
existence of bounded connected components of a decision region. We use this
obstruction to prove that a decision region of a generic, transversal ReLU
network F: R^n -> R with a single hidden layer of dimension (n + 1) can have no
more than one bounded connected component.

    

### [[2009.13772] Trust-Region Method with Deep Reinforcement Learning in Analog Design Space Exploration](http://arxiv.org/abs/2009.13772)


  This paper introduces new perspectives on analog design space search. To
minimize the time-to-market, this endeavor better cast as constraint
satisfaction problem than global optimization defined in prior arts. We
incorporate model-based agents, contrasted with model-free learning, to
implement a trust-region strategy. As such, simple feed-forward networks can be
trained with supervised learning, where the convergence is relatively trivial.
Experiment results demonstrate orders of magnitude improvement on search
iterations. Additionally, the unprecedented consideration of PVT conditions are
accommodated. On circuits with TSMC 5/6nm process, our method achieve
performance surpassing human designers. Furthermore, this framework is in
production in industrial settings.

    

### [[2010.03415] Knowledge-Based Learning of Nonlinear Dynamics and Chaos](http://arxiv.org/abs/2010.03415)


  Extracting predictive models from nonlinear systems is a central task in
scientific machine learning. One key problem is the reconciliation between
modern data-driven approaches and first principles. Despite rapid advances in
machine learning techniques, embedding domain knowledge into data-driven models
remains a challenge. In this work, we present a universal learning framework
for extracting predictive models from nonlinear systems based on observations.
Our framework can readily incorporate first principle knowledge because it
naturally models nonlinear systems as continuous-time systems. This both
improves the extracted models' extrapolation power and reduces the amount of
data needed for training. In addition, our framework has the advantages of
robustness to observational noise and applicability to irregularly sampled
data. We demonstrate the effectiveness of our scheme by learning predictive
models for a wide variety of systems including a stiff Van der Pol oscillator,
the Lorenz system, and the Kuramoto-Sivashinsky equation. For the Lorenz
system, different types of domain knowledge are incorporated to demonstrate the
strength of knowledge embedding in data-driven system identification.

    

### [[2010.05125] Learning Task-aware Robust Deep Learning Systems](http://arxiv.org/abs/2010.05125)


  Many works demonstrate that deep learning system is vulnerable to adversarial
attack. A deep learning system consists of two parts: the deep learning task
and the deep model. Nowadays, most existing works investigate the impact of the
deep model on robustness of deep learning systems, ignoring the impact of the
learning task. In this paper, we adopt the binary and interval label encoding
strategy to redefine the classification task and design corresponding loss to
improve robustness of the deep learning system. Our method can be viewed as
improving the robustness of deep learning systems from both the learning task
and deep model. Experimental results demonstrate that our learning task-aware
method is much more robust than traditional classification while retaining the
accuracy.

    

### [[2011.07442] Speech Enhancement Guided by Contextual Articulatory Information](http://arxiv.org/abs/2011.07442)


  Previous studies have confirmed the effectiveness of leveraging articulatory
information to attain improved speech enhancement (SE) performance. By
augmenting the original acoustic features with the place/manner of articulatory
features, the SE process can be guided to consider the articulatory properties
of the input speech when performing enhancement. Hence, we believe that the
contextual information of articulatory attributes should include useful
information and can further benefit SE in different languages. In this study,
we propose an SE system that improves its performance through optimizing the
contextual articulatory information in enhanced speech for both English and
Mandarin. We optimize the contextual articulatory information through
joint-train the SE model with an end-to-end automatic speech recognition (E2E
ASR) model, predicting the sequence of broad phone classes (BPC) instead of the
word sequences. Meanwhile, two training strategies are developed to train the
SE system based on the BPC-based ASR: multitask-learning and deep-feature
training strategies. Experimental results on the TIMIT and TMHINT dataset
confirm that the contextual articulatory information facilitates an SE system
in achieving better results than the traditional Acoustic Model(AM). Moreover,
in contrast to another SE system that is trained with monophonic ASR, the
BPC-based ASR (providing contextual articulatory information) can improve the
SE performance more effectively under different signal-to-noise ratios(SNR).

    

### [[2011.14821] Discovering Causal Structure with Reproducing-Kernel Hilbert Space $ε$-Machines](http://arxiv.org/abs/2011.14821)


  We merge computational mechanics' definition of causal states
(predictively-equivalent histories) with reproducing-kernel Hilbert space
(RKHS) representation inference. The result is a widely-applicable method that
infers causal structure directly from observations of a system's behaviors
whether they are over discrete or continuous events or time. A structural
representation -- a finite- or infinite-state kernel $\epsilon$-machine -- is
extracted by a reduced-dimension transform that gives an efficient
representation of causal states and their topology. In this way, the system
dynamics are represented by a stochastic (ordinary or partial) differential
equation that acts on causal states. We introduce an algorithm to estimate the
associated evolution operator. Paralleling the Fokker-Plank equation, it
efficiently evolves causal-state distributions and makes predictions in the
original data space via an RKHS functional mapping. We demonstrate these
techniques, together with their predictive abilities, on discrete-time,
discrete-value infinite Markov-order processes generated by finite-state hidden
Markov models with (i) finite or (ii) uncountably-infinite causal states and
(iii) continuous-time, continuous-value processes generated by thermally-driven
chaotic flows. The method robustly estimates causal structure in the presence
of varying external and measurement noise levels and for very high dimensional
data.

    

### [[2012.01920] Quantum learning algorithms imply circuit lower bounds](http://arxiv.org/abs/2012.01920)


  We establish the first general connection between the design of quantum
algorithms and circuit lower bounds. Specifically, let $\mathfrak{C}$ be a
class of polynomial-size concepts, and suppose that $\mathfrak{C}$ can be
PAC-learned with membership queries under the uniform distribution with error
$1/2 - \gamma$ by a time $T$ quantum algorithm. We prove that if $\gamma^2
\cdot T \ll 2^n/n$, then $\mathsf{BQE} \nsubseteq \mathfrak{C}$, where
$\mathsf{BQE} = \mathsf{BQTIME}[2^{O(n)}]$ is an exponential-time analogue of
$\mathsf{BQP}$. This result is optimal in both $\gamma$ and $T$, since it is
not hard to learn any class $\mathfrak{C}$ of functions in (classical) time $T
= 2^n$ (with no error), or in quantum time $T = \mathsf{poly}(n)$ with error at
most $1/2 - \Omega(2^{-n/2})$ via Fourier sampling. In other words, even a
marginal improvement on these generic learning algorithms would lead to major
consequences in complexity theory.
Our proof builds on several works in learning theory, pseudorandomness, and
computational complexity, and crucially, on a connection between non-trivial
classical learning algorithms and circuit lower bounds established by Oliveira
and Santhanam (CCC 2017). Extending their approach to quantum learning
algorithms turns out to create significant challenges. To achieve that, we show
among other results how pseudorandom generators imply learning-to-lower-bound
connections in a generic fashion, construct the first conditional pseudorandom
generator secure against uniform quantum computations, and extend the local
list-decoding algorithm of Impagliazzo, Jaiswal, Kabanets and Wigderson (SICOMP
2010) to quantum circuits via a delicate analysis. We believe that these
contributions are of independent interest and might find other applications.

    

### [[2012.02282] Creativity of Deep Learning: Conceptualization and Assessment](http://arxiv.org/abs/2012.02282)


  While the potential of deep learning(DL) for automating simple tasks is
already well explored, recent research started investigating the use of deep
learning for creative design, both for complete artifact creation and
supporting humans in the creation process. In this paper, we use insights from
computational creativity to conceptualize and assess current applications of
generative deep learning in creative domains identified in a literature review.
We highlight parallels between current systems and different models of human
creativity as well as their shortcomings. While deep learning yields results of
high value, such as high quality images, their novelity is typically limited
due to multiple reasons such a being tied to a conceptual space defined by
training data and humans. Current DL methods also do not allow for changes in
the internal problem representation and they lack the capability to identify
connections across highly different domains, both of which are seen as major
drivers of human creativity.

    

### [[2012.03765] Certified Robustness of Nearest Neighbors against Data Poisoning and Backdoor Attacks](http://arxiv.org/abs/2012.03765)


  Data poisoning attacks and backdoor attacks aim to corrupt a machine learning
classifier via modifying, adding, and/or removing some carefully selected
training examples, such that the corrupted classifier makes incorrect
predictions as the attacker desires. The key idea of state-of-the-art certified
defenses against data poisoning attacks and backdoor attacks is to create a
majority vote mechanism to predict the label of a testing example. Moreover,
each voter is a base classifier trained on a subset of the training dataset.
Classical simple learning algorithms such as k nearest neighbors (kNN) and
radius nearest neighbors (rNN) have intrinsic majority vote mechanisms. In this
work, we show that the intrinsic majority vote mechanisms in kNN and rNN
already provide certified robustness guarantees against data poisoning attacks
and backdoor attacks. Moreover, our evaluation results on MNIST and CIFAR10
show that the intrinsic certified robustness guarantees of kNN and rNN
outperform those provided by state-of-the-art certified defenses. Our results
serve as standard baselines for future certified defenses against data
poisoning attacks and backdoor attacks.

    

### [[2012.12687] Wasserstein Dropout](http://arxiv.org/abs/2012.12687)


  Despite of its importance for safe machine learning, uncertainty
quantification for neural networks is far from being solved. State-of-the-art
approaches to estimate neural uncertainties are often hybrid, combining
parametric models with explicit or implicit (dropout-based) ensembling. We take
another pathway and propose a novel approach to uncertainty quantification for
regression tasks, Wasserstein dropout, that is purely non-parametric.
Technically, it captures aleatoric uncertainty by means of dropout-based
sub-network distributions. This is accomplished by a new objective which
minimizes the Wasserstein distance between the label distribution and the model
distribution. An extensive empirical analysis shows that Wasserstein dropout
outperforms state-of-the-art methods, on vanilla test data as well as under
distributional shift, in terms of producing more accurate and stable
uncertainty estimates.

    

### [[2101.01000] Conditional Local Convolution for Spatio-temporal Meteorological Forecasting](http://arxiv.org/abs/2101.01000)


  Spatio-temporal forecasting is challenging attributing to the high
nonlinearity in temporal dynamics as well as complex location-characterized
patterns in spatial domains, especially in fields like weather forecasting.
Graph convolutions are usually used for modeling the spatial dependency in
meteorology to handle the irregular distribution of sensors' spatial location.
In this work, a novel graph-based convolution for imitating the meteorological
flows is proposed to capture the local spatial patterns. Based on the
assumption of smoothness of location-characterized patterns, we propose
conditional local convolution whose shared kernel on nodes' local space is
approximated by feedforward networks, with local representations of coordinate
obtained by horizon maps into cylindrical-tangent space as its input. The
established united standard of local coordinate system preserves the
orientation on geography. We further propose the distance and orientation
scaling terms to reduce the impacts of irregular spatial distribution. The
convolution is embedded in a Recurrent Neural Network architecture to model the
temporal dynamics, leading to the Conditional Local Convolution Recurrent
Network (CLCRN). Our model is evaluated on real-world weather benchmark
datasets, achieving state-of-the-art performance with obvious improvements. We
conduct further analysis on local pattern visualization, model's framework
choice, advantages of horizon maps and etc.

    

### [[2103.03240] Learning ABCs: Approximate Bijective Correspondence for isolating factors of variation](http://arxiv.org/abs/2103.03240)


  Representational learning forms the backbone of most deep learning
applications, and the value of a learned representation is intimately tied to
its information content regarding different factors of variation. Finding good
representations depends on the nature of supervision and the learning
algorithm. We propose a novel algorithm that utilizes a weak form of
supervision where the data is partitioned into sets according to certain
inactive (common) factors of variation which are invariant across elements of
each set. Our key insight is that by seeking correspondence between elements of
different sets, we learn strong representations that exclude the inactive
factors of variation and isolate the active (varying) factors which vary within
all sets. As a consequence of focusing on the active factors, our method can
leverage a mix of set-supervised and wholly unsupervised data, which can even
belong to a different domain. We tackle the challenging problem of
synthetic-to-real object pose transfer, by isolating from images pose
information which generalizes to the category level and across the
synthetic/real domain gap, even without pose annotations on anything. The
method can also boost performance in supervised settings, by strengthening
intermediate representations.

    

### [[2103.05896] Streaming Linear System Identification with Reverse Experience Replay](http://arxiv.org/abs/2103.05896)


  We consider the problem of estimating a linear time-invariant (LTI) dynamical
system from a single trajectory via streaming algorithms, which is encountered
in several applications including reinforcement learning (RL) and time-series
analysis. While the LTI system estimation problem is well-studied in the {\em
offline} setting, the practically important streaming/online setting has
received little attention. Standard streaming methods like stochastic gradient
descent (SGD) are unlikely to work since streaming points can be highly
correlated. In this work, we propose a novel streaming algorithm, SGD with
Reverse Experience Replay ($\mathsf{SGD}-\mathsf{RER}$), that is inspired by
the experience replay (ER) technique popular in the RL literature.
$\mathsf{SGD}-\mathsf{RER}$ divides data into small buffers and runs SGD
backwards on the data stored in the individual buffers. We show that this
algorithm exactly deconstructs the dependency structure and obtains information
theoretically optimal guarantees for both parameter error and prediction error.
Thus, we provide the first -- to the best of our knowledge -- optimal SGD-style
algorithm for the classical problem of linear system identification with a
first order oracle. Furthermore, $\mathsf{SGD}-\mathsf{RER}$ can be applied to
more general settings like sparse LTI identification with known sparsity
pattern, and non-linear dynamical systems. Our work demonstrates that the
knowledge of data dependency structure can aid us in designing statistically
and computationally efficient algorithms which can "decorrelate" streaming
samples.

    

### [[2103.06076] Designing Disaggregated Evaluations of AI Systems: Choices, Considerations, and Tradeoffs](http://arxiv.org/abs/2103.06076)


  Disaggregated evaluations of AI systems, in which system performance is
assessed and reported separately for different groups of people, are
conceptually simple. However, their design involves a variety of choices. Some
of these choices influence the results that will be obtained, and thus the
conclusions that can be drawn; others influence the impacts -- both beneficial
and harmful -- that a disaggregated evaluation will have on people, including
the people whose data is used to conduct the evaluation. We argue that a deeper
understanding of these choices will enable researchers and practitioners to
design careful and conclusive disaggregated evaluations. We also argue that
better documentation of these choices, along with the underlying considerations
and tradeoffs that have been made, will help others when interpreting an
evaluation's results and conclusions.

    

### [[2104.01539] DINE: Domain Adaptation from Single and Multiple Black-box Predictors](http://arxiv.org/abs/2104.01539)


  To ease the burden of labeling, unsupervised domain adaptation (UDA) aims to
transfer knowledge in previous and related labeled datasets (sources) to a new
unlabeled dataset (target). Despite impressive progress, prior methods always
need to access the raw source data and develop data-dependent alignment
approaches to recognize the target samples in a transductive learning manner,
which may raise privacy concerns from source individuals. Several recent
studies resort to an alternative solution by exploiting the well-trained
white-box model from the source domain, yet, it may still leak the raw data
through generative adversarial learning. This paper studies a practical and
interesting setting for UDA, where only black-box source models (i.e., only
network predictions are available) are provided during adaptation in the target
domain. To solve this problem, we propose a new two-step knowledge adaptation
framework called DIstill and fine-tuNE (DINE). Taking into consideration the
target data structure, DINE first distills the knowledge from the source
predictor to a customized target model, then fine-tunes the distilled model to
further fit the target domain. Besides, neural networks are not required to be
identical across domains in DINE, even allowing effective adaptation on a
low-resource device. Empirical results on three UDA scenarios (i.e.,
single-source, multi-source, and partial-set) confirm that DINE achieves highly
competitive performance compared to state-of-the-art data-dependent approaches.
Code is available at \url{this https URL}.

    

### [[2105.11558] Near-optimal Offline and Streaming Algorithms for Learning Non-Linear Dynamical Systems](http://arxiv.org/abs/2105.11558)


  We consider the setting of vector valued non-linear dynamical systems
$X_{t+1} = \phi(A^* X_t) + \eta_t$, where $\eta_t$ is unbiased noise and $\phi
: \mathbb{R} \to \mathbb{R}$ is a known link function that satisfies certain
{\em expansivity property}. The goal is to learn $A^*$ from a single trajectory
$X_1,\cdots,X_T$ of {\em dependent or correlated} samples. While the problem is
well-studied in the linear case, where $\phi$ is identity, with optimal error
rates even for non-mixing systems, existing results in the non-linear case hold
only for mixing systems. In this work, we improve existing results for learning
nonlinear systems in a number of ways: a) we provide the first offline
algorithm that can learn non-linear dynamical systems without the mixing
assumption, b) we significantly improve upon the sample complexity of existing
results for mixing systems, c) in the much harder one-pass, streaming setting
we study a SGD with Reverse Experience Replay ($\mathsf{SGD-RER}$) method, and
demonstrate that for mixing systems, it achieves the same sample complexity as
our offline algorithm, d) we justify the expansivity assumption by showing that
for the popular ReLU link function -- a non-expansive but easy to learn link
function with i.i.d. samples -- any method would require exponentially many
samples (with respect to dimension of $X_t$) from the dynamical system. We
validate our results via. simulations and demonstrate that a naive application
of SGD can be highly sub-optimal. Indeed, our work demonstrates that for
correlated data, specialized methods designed for the dependency structure in
data can significantly outperform standard SGD based methods.

    

### [[2105.11689] Self-Supervised Graph Representation Learning via Topology Transformations](http://arxiv.org/abs/2105.11689)


  We present the Topology Transformation Equivariant Representation learning, a
general paradigm of self-supervised learning for node representations of graph
data to enable the wide applicability of Graph Convolutional Neural Networks
(GCNNs). We formalize the proposed model from an information-theoretic
perspective, by maximizing the mutual information between topology
transformations and node representations before and after the transformations.
We derive that maximizing such mutual information can be relaxed to minimizing
the cross entropy between the applied topology transformation and its
estimation from node representations. In particular, we seek to sample a subset
of node pairs from the original graph and flip the edge connectivity between
each pair to transform the graph topology. Then, we self-train a representation
encoder to learn node representations by reconstructing the topology
transformations from the feature representations of the original and
transformed graphs. In experiments, we apply the proposed model to the
downstream node classification, graph classification and link prediction tasks,
and results show that the proposed method outperforms the state-of-the-art
unsupervised approaches.

    

### [[2105.14216] Efficient Cross-Device Federated Learning Algorithms for Minimax Problems](http://arxiv.org/abs/2105.14216)


  In many machine learning applications where massive and privacy-sensitive
data are generated on numerous mobile or IoT devices, collecting data in a
centralized location may be prohibitive. Thus, it is increasingly attractive to
estimate parameters over mobile or IoT devices while keeping data localized.
Such learning setting is known as cross-device federated learning. In this
paper, we propose the first theoretically guaranteed algorithms for general
minimax problems in the cross-device federated learning setting. Our algorithms
require only a fraction of devices in each round of training, which overcomes
the difficulty introduced by the low availability of devices. The communication
overhead is further reduced by performing multiple local update steps on
clients before communication with the server, and global gradient estimates are
leveraged to correct the bias in local update directions introduced by data
heterogeneity. By developing analyses based on novel potential functions, we
establish theoretical convergence guarantees for our algorithms. Experimental
results on AUC maximization, robust adversarial network training, and GAN
training tasks demonstrate the efficiency of our algorithms.

    

### [[2106.03443] Causal Influence Detection for Improving Efficiency in Reinforcement Learning](http://arxiv.org/abs/2106.03443)


  Many reinforcement learning (RL) environments consist of independent entities
that interact sparsely. In such environments, RL agents have only limited
influence over other entities in any particular situation. Our idea in this
work is that learning can be efficiently guided by knowing when and what the
agent can influence with its actions. To achieve this, we introduce a measure
of \emph{situation-dependent causal influence} based on conditional mutual
information and show that it can reliably detect states of influence. We then
propose several ways to integrate this measure into RL algorithms to improve
exploration and off-policy learning. All modified algorithms show strong
increases in data efficiency on robotic manipulation tasks.

    

### [[2106.09719] Machining Cycle Time Prediction: Data-driven Modelling of Machine Tool Feedrate Behavior with Neural Networks](http://arxiv.org/abs/2106.09719)


  Accurate prediction of machining cycle times is important in the
manufacturing industry. Usually, Computer Aided Manufacturing (CAM) software
estimates the machining times using the commanded feedrate from the toolpath
file using basic kinematic settings. Typically, the methods do not account for
toolpath geometry or toolpath tolerance and therefore under estimate the
machining cycle times considerably. Removing the need for machine specific
knowledge, this paper presents a data-driven feedrate and machining cycle time
prediction method by building a neural network model for each machine tool
axis. In this study, datasets composed of the commanded feedrate, nominal
acceleration, toolpath geometry and the measured feedrate were used to train a
neural network model. Validation trials using a representative industrial thin
wall structure component on a commercial machining centre showed that this
method estimated the machining time with more than 90% accuracy. This method
showed that neural network models have the capability to learn the behavior of
a complex machine tool system and predict cycle times. Further integration of
the methods will be critical in the implantation of digital twins in Industry
4.0.

    

### [[2106.12142] IQ-Learn: Inverse soft-Q Learning for Imitation](http://arxiv.org/abs/2106.12142)


  In many sequential decision-making problems (e.g., robotics control, game
playing, sequential prediction), human or expert data is available containing
useful information about the task. However, imitation learning (IL) from a
small amount of expert data can be challenging in high-dimensional environments
with complex dynamics. Behavioral cloning is a simple method that is widely
used due to its simplicity of implementation and stable convergence but doesn't
utilize any information involving the environment's dynamics. Many existing
methods that exploit dynamics information are difficult to train in practice
due to an adversarial optimization process over reward and policy approximators
or biased, high variance gradient estimators. We introduce a method for
dynamics-aware IL which avoids adversarial training by learning a single
Q-function, implicitly representing both reward and policy. On standard
benchmarks, the implicitly learned rewards show a high positive correlation
with the ground-truth rewards, illustrating our method can also be used for
inverse reinforcement learning (IRL). Our method, Inverse soft-Q learning
(IQ-Learn) obtains state-of-the-art results in offline and online imitation
learning settings, significantly outperforming existing methods both in the
number of required environment interactions and scalability in high-dimensional
spaces, often by more than 3x.

    

### [[2106.13436] A hybrid model-based and learning-based approach for classification using limited number of training samples](http://arxiv.org/abs/2106.13436)


  The fundamental task of classification given a limited number of training
data samples is considered for physical systems with known parametric
statistical models. The standalone learning-based and statistical model-based
classifiers face major challenges towards the fulfillment of the classification
task using a small training set. Specifically, classifiers that solely rely on
the physics-based statistical models usually suffer from their inability to
properly tune the underlying unobservable parameters, which leads to a
mismatched representation of the system's behaviors. Learning-based
classifiers, on the other hand, typically rely on a large number of training
data from the underlying physical process, which might not be feasible in most
practical scenarios. In this paper, a hybrid classification method -- termed
HyPhyLearn -- is proposed that exploits both the physics-based statistical
models and the learning-based classifiers. The proposed solution is based on
the conjecture that HyPhyLearn would alleviate the challenges associated with
the individual approaches of learning-based and statistical model-based
classifiers by fusing their respective strengths. The proposed hybrid approach
first estimates the unobservable model parameters using the available
(suboptimal) statistical estimation procedures, and subsequently use the
physics-based statistical models to generate synthetic data. Then, the training
data samples are incorporated with the synthetic data in a learning-based
classifier that is based on domain-adversarial training of neural networks.
Specifically, in order to address the mismatch problem, the classifier learns a
mapping from the training data and the synthetic data to a common feature
space. Simultaneously, the classifier is trained to find discriminative
features within this space in order to fulfill the classification task.

    

### [[2106.14790] PhysiNet: A Combination of Physics-based Model and Neural Network Model for Digital Twins](http://arxiv.org/abs/2106.14790)


  As the real-time digital counterpart of a physical system or process, digital
twins are utilized for system simulation and optimization. Neural networks are
one way to build a digital twins model by using data especially when a
physics-based model is not accurate or even not available. However, for a newly
designed system, it takes time to accumulate enough data for neural network
model and only an approximate physics-based model is available. To take
advantage of both models, this paper proposed a model that combines the
physics-based model and the neural network model to improve the prediction
accuracy for the whole life cycle of a system. The proposed hybrid model
(PhysiNet) was able to automatically combine the models and boost their
prediction performance. Experiments showed that the PhysiNet outperformed both
the physics-based model and the neural network model.

    

### [[2111.14427] Self-Training of Halfspaces with Generalization Guarantees under Massart Mislabeling Noise Model](http://arxiv.org/abs/2111.14427)


  We investigate the generalization properties of a self-training algorithm
with halfspaces. The approach learns a list of halfspaces iteratively from
labeled and unlabeled training data, in which each iteration consists of two
steps: exploration and pruning. In the exploration phase, the halfspace is
found sequentially by maximizing the unsigned-margin among unlabeled examples
and then assigning pseudo-labels to those that have a distance higher than the
current threshold. The pseudo-labeled examples are then added to the training
set, and a new classifier is learned. This process is repeated until no more
unlabeled examples remain for pseudo-labeling. In the pruning phase,
pseudo-labeled samples that have a distance to the last halfspace greater than
the associated unsigned-margin are then discarded. We prove that the
misclassification error of the resulting sequence of classifiers is bounded and
show that the resulting semi-supervised approach never degrades performance
compared to the classifier learned using only the initial labeled training set.
Experiments carried out on a variety of benchmarks demonstrate the efficiency
of the proposed approach compared to state-of-the-art methods.

    

### [[2106.15353] Patient-independent Schizophrenia Relapse Prediction Using Mobile Sensor based Daily Behavioral Rhythm Changes](http://arxiv.org/abs/2106.15353)


  A schizophrenia relapse has severe consequences for a patient's health, work,
and sometimes even life safety. If an oncoming relapse can be predicted on
time, for example by detecting early behavioral changes in patients, then
interventions could be provided to prevent the relapse. In this work, we
investigated a machine learning based schizophrenia relapse prediction model
using mobile sensing data to characterize behavioral features. A
patient-independent model providing sequential predictions, closely
representing the clinical deployment scenario for relapse prediction, was
evaluated. The model uses the mobile sensing data from the recent four weeks to
predict an oncoming relapse in the next week. We used the behavioral rhythm
features extracted from daily templates of mobile sensing data, self-reported
symptoms collected via EMA (Ecological Momentary Assessment), and demographics
to compare different classifiers for the relapse prediction. Naive Bayes based
model gave the best results with an F2 score of 0.083 when evaluated in a
dataset consisting of 63 schizophrenia patients, each monitored for up to a
year. The obtained F2 score, though low, is better than the baseline
performance of random classification (F2 score of 0.02 $\pm$ 0.024). Thus,
mobile sensing has predictive value for detecting an oncoming relapse and needs
further investigation to improve the current performance. Towards that end,
further feature engineering and model personalization based on the behavioral
idiosyncrasies of a patient could be helpful.

    

### [[2111.15640] Diffusion Autoencoders: Toward a Meaningful and Decodable Representation](http://arxiv.org/abs/2111.15640)


  Diffusion probabilistic models (DPMs) have achieved remarkable quality in
image generation that rivals GANs'. But unlike GANs, DPMs use a set of latent
variables that lack semantic meaning and cannot serve as a useful
representation for other tasks. This paper explores the possibility of using
DPMs for representation learning and seeks to extract a meaningful and
decodable representation of an input image via autoencoding. Our key idea is to
use a learnable encoder for discovering the high-level semantics, and a DPM as
the decoder for modeling the remaining stochastic variations. Our method can
encode any image into a two-part latent code, where the first part is
semantically meaningful and linear, and the second part captures stochastic
details, allowing near-exact reconstruction. This capability enables
challenging applications that currently foil GAN-based methods, such as
attribute manipulation on real images. We also show that this two-level
encoding improves denoising efficiency and naturally facilitates various
downstream tasks including few-shot conditional sampling. Please visit our
project page: this https URL


### [[2112.01168] MemPool-3D: Boosting Performance and Efficiency of Shared-L1 Memory Many-Core Clusters with 3D Integration](http://arxiv.org/abs/2112.01168)


  Three-dimensional integrated circuits promise power, performance, and
footprint gains compared to their 2D counterparts, thanks to drastic reductions
in the interconnects' length through their smaller form factor. We can leverage
the potential of 3D integration by enhancing MemPool, an open-source many-core
design with 256 cores and a shared pool of L1 scratchpad memory connected with
a low-latency interconnect. MemPool's baseline 2D design is severely limited by
routing congestion and wire propagation delay, making the design ideal for 3D
integration. In architectural terms, we increase MemPool's scratchpad memory
capacity beyond the sweet spot for 2D designs, improving performance in a
common digital signal processing kernel. We propose a 3D MemPool design that
leverages a smart partitioning of the memory resources across two layers to
balance the size and utilization of the stacked dies. In this paper, we explore
the architectural and the technology parameter spaces by analyzing the power,
performance, area, and energy efficiency of MemPool instances in 2D and 3D with
1 MiB, 2 MiB, 4 MiB, and 8 MiB of scratchpad memory in a commercial 28 nm
technology node. We observe a performance gain of 9.1% when running a matrix
multiplication on the MemPool-3D design with 4 MiB of scratchpad memory
compared to the MemPool 2D counterpart. In terms of energy efficiency, we can
implement the MemPool-3D instance with 4 MiB of L1 memory on an energy budget
15% smaller than its 2D counterpart, and even 3.7% smaller than the MemPool-2D
instance with one-fourth of the L1 scratchpad memory capacity.

    

### [[2112.01075] Memory-efficient array redistribution through portable collective communication](http://arxiv.org/abs/2112.01075)


  Modern large-scale deep learning workloads highlight the need for parallel
execution across many devices in order to fit model data into hardware
accelerator memories. In these settings, array redistribution may be required
during a computation, but can also become a bottleneck if not done efficiently.
In this paper we address the problem of redistributing multi-dimensional array
data in SPMD computations, the most prevalent form of parallelism in deep
learning. We present a type-directed approach to synthesizing array
redistributions as sequences of MPI-style collective operations. We prove
formally that our synthesized redistributions are memory-efficient and perform
no excessive data transfers. Array redistribution for SPMD computations using
collective operations has also been implemented in the context of the XLA SPMD
partitioner, a production-grade tool for partitioning programs across
accelerator systems. We evaluate our approach against the XLA implementation
and find that our approach delivers a geometric mean speedup of $1.22\times$,
with maximum speedups as a high as $5.7\times$, while offering provable memory
guarantees, making our system particularly appealing for large-scale models.

    

### [[2112.01082] Grafana plugin for visualising vote based consensus mechanisms, and network P2P overlay networks](http://arxiv.org/abs/2112.01082)


  In this paper, we present a plugin for visualising vote based consensus
mechanisms primarily aimed to help engineers understand and debug blockchain
and distributed ledger protocols. Both tools are built as Grafana plugins and
make no assumptions on the data storage implementation. The plugins can be
configured via Grafana plugin configuration interface to fit the specifics of
the protocol implementation.

    

### [[2112.01189] Simplifying heterogeneous migration between x86 and ARM machines](http://arxiv.org/abs/2112.01189)


  Heterogeneous computing is the strategy of deploying multiple types of
processing elements within a single workflow, and allowing each to perform the
tasks to which is best suited. To fully harness the power of heterogeneity, we
want to be able to dynamically schedule portions of the code and migrate
processes at runtime between the architectures. Also, migration includes
transforming the execution state of the process, which induces a significant
overhead that offsets the benefits of migrating in the first place. The goal of
my PhD is to work on techniques that allow applications to run on heterogeneous
cores under a shared programming model, and to tackle the generic problem of
creating a uniform address space between processes running on these highly
diverse systems. This would greatly simplify the migration process. We will
start by examining a common stack layout between x86 and ARM binaries, focusing
on these two widely spread architectures, and later we will try to generalize
to other more diverse execution environments. On top of that, the performance
and energy efficiency of the above effort compared to current approaches will
be benchmarked.

    

### [[2106.13002] Loosely-Stabilizing Phase Clocks and the Adaptive Majority Problem](http://arxiv.org/abs/2106.13002)


  We present a loosely-stabilizing phase clock for population protocols. In the
population model we are given a system of $n$ identical agents which interact
in a sequence of randomly chosen pairs. Our phase clock is leaderless and it
requires $O(\log n)$ states. It runs forever and is, at any point of time, in a
synchronous state w.h.p. When started in an arbitrary configuration, it
recovers rapidly and enters a synchronous configuration within $O(n\log n)$
interactions w.h.p. Once the clock is synchronized, it stays in a synchronous
configuration for at least poly $n$ parallel time w.h.p.
We use our clock to design a loosely-stabilizing protocol that solves the
comparison problem introduced by Alistarh et al., 2021. In this problem, a
subset of agents has at any time either $A$ or $B$ as input. The goal is to
keep track which of the two opinions is (momentarily) the majority. We show
that if the majority has a support of at least $\Omega(\log n)$ agents and a
sufficiently large bias is present, then the protocol converges to a correct
output within $O(n\log n)$ interactions and stays in a correct configuration
for poly $n$ interactions, w.h.p.

    

### [[2112.00740] Collaborative AI Needs Stronger Assurances Driven by Risks](http://arxiv.org/abs/2112.00740)


  Collaborative AI systems (CAISs) aim at working together with humans in a
shared space to achieve a common goal. This critical setting yields hazardous
circumstances that could harm human beings. Thus, building such systems with
strong assurances of compliance with requirements, domain-specific standards
and regulations is of greatest importance. Only few scale impact has been
reported so far for such systems since much work remains to manage possible
risks. We identify emerging problems in this context and then we report our
vision, as well as the progress of our multidisciplinary research team composed
of software/systems, and mechatronics engineers to develop a risk-driven
assurance process for CAISs.

    

### [[2112.00797] A Feedback Integrated Web-Based Multi-Criteria Group Decision Support Model for Contractor Selection using Fuzzy Analytic Hierarchy Process](http://arxiv.org/abs/2112.00797)


  In this paper, a feedback integrated multi-criteria group decision support
model for contractor selection was proposed.

    

### [[2112.00799] Finding, Scoring and Explaining Arguments in Bayesian Networks](http://arxiv.org/abs/2112.00799)


  We propose a new approach to explain Bayesian Networks. The approach revolves
around a new definition of a probabilistic argument and the evidence it
provides. We define a notion of independent arguments, and propose an algorithm
to extract a list of relevant, independent arguments given a Bayesian Network,
a target node and a set of observations. To demonstrate the relevance of the
arguments, we show how we can use the extracted arguments to approximate
message passing. Finally, we show a simple scheme to explain the arguments in
natural language.

    

### [[2112.00800] Iconary: A Pictionary-Based Game for Testing Multimodal Communication with Drawings and Text](http://arxiv.org/abs/2112.00800)


  Communicating with humans is challenging for AIs because it requires a shared
understanding of the world, complex semantics (e.g., metaphors or analogies),
and at times multi-modal gestures (e.g., pointing with a finger, or an arrow in
a diagram). We investigate these challenges in the context of Iconary, a
collaborative game of drawing and guessing based on Pictionary, that poses a
novel challenge for the research community. In Iconary, a Guesser tries to
identify a phrase that a Drawer is drawing by composing icons, and the Drawer
iteratively revises the drawing to help the Guesser in response. This
back-and-forth often uses canonical scenes, visual metaphor, or icon
compositions to express challenging words, making it an ideal test for mixing
language and visual/symbolic communication in AI. We propose models to play
Iconary and train them on over 55,000 games between human players. Our models
are skillful players and are able to employ world knowledge in language models
to play with words unseen during training. Elite human players outperform our
models, particularly at the drawing task, leaving an important gap for future
research to address. We release our dataset, code, and evaluation setup as a
challenge to the community at this http URL.

    

### [[2112.00812] Evolving Open Complexity](http://arxiv.org/abs/2112.00812)


  Information theoretic analysis of large evolved programs produced by running
genetic programming for up to a million generations has shown even functions as
smooth and well behaved as floating point addition and multiplication loose
entropy and consequently are robust and fail to propagate disruption to their
outputs. This means, while dependent upon fitness tests, many genetic changes
deep within trees are silent. For evolution to proceed at reasonable rate it
must be possible to measure the impact of most code changes, yet in large trees
most crossover sites are distant from the root node. We suggest to evolve very
large very complex programs, it will be necessary to adopt an open architecture
where most mutation sites are within 10 to 100 levels of the organism's
environment.

    

### [[2112.00848] First Steps of an Approach to the ARC Challenge based on Descriptive Grid Models and the Minimum Description Length Principle](http://arxiv.org/abs/2112.00848)


  The Abstraction and Reasoning Corpus (ARC) was recently introduced by
François Chollet as a tool to measure broad intelligence in both humans and
machines. It is very challenging, and the best approach in a Kaggle competition
could only solve 20% of the tasks, relying on brute-force search for chains of
hand-crafted transformations. In this paper, we present the first steps
exploring an approach based on descriptive grid models and the Minimum
Description Length (MDL) principle. The grid models describe the contents of a
grid, and support both parsing grids and generating grids. The MDL principle is
used to guide the search for good models, i.e. models that compress the grids
the most. We report on our progress over a year, improving on the general
approach and the models. Out of the 400 training tasks, our performance
increased from 5 to 29 solved tasks, only using 30s computation time per task.
Our approach not only predicts the output grids, but also outputs an
intelligible model and explanations for how the model was incrementally built.

    

### [[2112.00903] Modeling human intention inference in continuous 3D domains by inverse planning and body kinematics](http://arxiv.org/abs/2112.00903)


  How to build AI that understands human intentions, and uses this knowledge to
collaborate with people? We describe a computational framework for evaluating
models of goal inference in the domain of 3D motor actions, which receives as
input the 3D coordinates of an agent's body, and of possible targets, to
produce a continuously updated inference of the intended target. We evaluate
our framework in three behavioural experiments using a novel Target Reaching
Task, in which human observers infer intentions of actors reaching for targets
among distracts. We describe Generative Body Kinematics model, which predicts
human intention inference in this domain using Bayesian inverse planning and
inverse body kinematics. We compare our model to three heuristics, which
formalize the principle of least effort using simple assumptions about the
actor's constraints, without the use of inverse planning. Despite being more
computationally costly, the Generative Body Kinematics model outperforms the
heuristics in certain scenarios, such as environments with obstacles, and at
the beginning of reaching actions while the actor is relatively far from the
intended target. The heuristics make increasingly accurate predictions during
later stages of reaching actions, such as, when the intended target is close,
and can be inferred by extrapolating the wrist trajectory. Our results identify
contexts in which inverse body kinematics is useful for intention inference. We
show that human observers indeed rely on inverse body kinematics in such
scenarios, suggesting that modeling body kinematic can improve performance of
inference algorithms.

    

### [[2112.00967] Relational Graph Learning for Grounded Video Description Generation](http://arxiv.org/abs/2112.00967)


  Grounded video description (GVD) encourages captioning models to attend to
appropriate video regions (e.g., objects) dynamically and generate a
description. Such a setting can help explain the decisions of captioning models
and prevents the model from hallucinating object words in its description.
However, such design mainly focuses on object word generation and thus may
ignore fine-grained information and suffer from missing visual concepts.
Moreover, relational words (e.g., "jump left or right") are usual
spatio-temporal inference results, i.e., these words cannot be grounded on
certain spatial regions. To tackle the above limitations, we design a novel
relational graph learning framework for GVD, in which a language-refined scene
graph representation is designed to explore fine-grained visual concepts.
Furthermore, the refined graph can be regarded as relational inductive
knowledge to assist captioning models in selecting the relevant information it
needs to generate correct words. We validate the effectiveness of our model
through automatic metrics and human evaluation, and the results indicate that
our approach can generate more fine-grained and accurate description, and it
solves the problem of object hallucination to some extent.

    

### [[2112.00969] Object-Centric Unsupervised Image Captioning](http://arxiv.org/abs/2112.00969)


  Training an image captioning model in an unsupervised manner without
utilizing annotated image-caption pairs is an important step towards tapping
into a wider corpus of text and images. In the supervised setting,
image-caption pairs are "well-matched", where all objects mentioned in the
sentence appear in the corresponding image. These pairings are, however, not
available in the unsupervised setting. To overcome this, a main school of
research that has been shown to be effective in overcoming this is to construct
pairs from the images and texts in the training set according to their overlap
of objects. Unlike in the supervised setting, these constructed pairings are
however not guaranteed to have fully overlapping set of objects. Our work in
this paper overcomes this by harvesting objects corresponding to a given
sentence from the training set, even if they don't belong to the same image.
When used as input to a transformer, such mixture of objects enable larger if
not full object coverage, and when supervised by the corresponding sentence,
produced results that outperform current state of the art unsupervised methods
by a significant margin. Building upon this finding, we further show that (1)
additional information on relationship between objects and attributes of
objects also helps in boosting performance; and (2) our method also extends
well to non-English image captioning, which usually suffers from a scarcer
level of annotations. Our findings are supported by strong empirical results.

    

### [[2112.00970] Narrative Cartography with Knowledge Graphs](http://arxiv.org/abs/2112.00970)


  Narrative cartography is a discipline which studies the interwoven nature of
stories and maps. However, conventional geovisualization techniques of
narratives often encounter several prominent challenges, including the data
acquisition & integration challenge and the semantic challenge. To tackle these
challenges, in this paper, we propose the idea of narrative cartography with
knowledge graphs (KGs). Firstly, to tackle the data acquisition & integration
challenge, we develop a set of KG-based GeoEnrichment toolboxes to allow users
to search and retrieve relevant data from integrated cross-domain knowledge
graphs for narrative mapping from within a GISystem. With the help of this
tool, the retrieved data from KGs are directly materialized in a GIS format
which is ready for spatial analysis and mapping. Two use cases - Magellan's
expedition and World War II - are presented to show the effectiveness of this
approach. In the meantime, several limitations are identified from this
approach, such as data incompleteness, semantic incompatibility, and the
semantic challenge in geovisualization. For the later two limitations, we
propose a modular ontology for narrative cartography, which formalizes both the
map content (Map Content Module) and the geovisualization process (Cartography
Module). We demonstrate that, by representing both the map content and the
geovisualization process in KGs (an ontology), we can realize both data
reusability and map reproducibility for narrative cartography.

    

### [[2112.00974] Consensus Graph Representation Learning for Better Grounded Image Captioning](http://arxiv.org/abs/2112.00974)


  The contemporary visual captioning models frequently hallucinate objects that
are not actually in a scene, due to the visual misclassification or
over-reliance on priors that resulting in the semantic inconsistency between
the visual information and the target lexical words. The most common way is to
encourage the captioning model to dynamically link generated object words or
phrases to appropriate regions of the image, i.e., the grounded image
captioning (GIC). However, GIC utilizes an auxiliary task (grounding objects)
that has not solved the key issue of object hallucination, i.e., the semantic
inconsistency. In this paper, we take a novel perspective on the issue above -
exploiting the semantic coherency between the visual and language modalities.
Specifically, we propose the Consensus Rraph Representation Learning framework
(CGRL) for GIC that incorporates a consensus representation into the grounded
captioning pipeline. The consensus is learned by aligning the visual graph
(e.g., scene graph) to the language graph that consider both the nodes and
edges in a graph. With the aligned consensus, the captioning model can capture
both the correct linguistic characteristics and visual relevance, and then
grounding appropriate image regions further. We validate the effectiveness of
our model, with a significant decline in object hallucination (-9% CHAIRi) on
the Flickr30k Entities dataset. Besides, our CGRL also evaluated by several
automatic metrics and human evaluation, the results indicate that the proposed
approach can simultaneously improve the performance of image captioning (+2.9
Cider) and grounding (+2.3 F1LOC).

    

### [[2112.01016] On Two XAI Cultures: A Case Study of Non-technical Explanations in Deployed AI System](http://arxiv.org/abs/2112.01016)


  Explainable AI (XAI) research has been booming, but the question "$\textbf{To
whom}$ are we making AI explainable?" is yet to gain sufficient attention. Not
much of XAI is comprehensible to non-AI experts, who nonetheless, are the
primary audience and major stakeholders of deployed AI systems in practice. The
gap is glaring: what is considered "explained" to AI-experts versus non-experts
are very different in practical scenarios. Hence, this gap produced two
distinct cultures of expectations, goals, and forms of XAI in real-life AI
deployments.
We advocate that it is critical to develop XAI methods for non-technical
audiences. We then present a real-life case study, where AI experts provided
non-technical explanations of AI decisions to non-technical stakeholders, and
completed a successful deployment in a highly regulated industry. We then
synthesize lessons learned from the case, and share a list of suggestions for
AI experts to consider when explaining AI decisions to non-technical
stakeholders.

    

### [[2112.01040] EngineKGI: Closed-Loop Knowledge Graph Inference](http://arxiv.org/abs/2112.01040)


  Knowledge Graph (KG) inference is the vital technique to address the natural
incompleteness of KGs. The existing KG inference approaches can be classified
into rule learning-based and KG embedding-based models. However, these
approaches cannot well balance accuracy, generalization, interpretability and
efficiency, simultaneously. Besides, these models always rely on pure triples
and neglect additional information. Therefore, both KG embedding (KGE) and rule
learning KG inference approaches face challenges due to the sparse entities and
the limited semantics. We propose a novel and effective closed-loop KG
inference framework EngineKGI operating similarly as an engine based on these
observations. EngineKGI combines KGE and rule learning to complement each other
in a closed-loop pattern while taking advantage of semantics in paths and
concepts. KGE module exploits paths to enhance the semantic association between
entities and introduces rules for interpretability. A novel rule pruning
mechanism is proposed in the rule learning module by leveraging paths as
initial candidate rules and employing KG embeddings together with concepts for
extracting more high-quality rules. Experimental results on four real-world
datasets show that our model outperforms other baselines on link prediction
tasks, demonstrating the effectiveness and superiority of our model on KG
inference in a joint logic and data-driven fashion with a closed-loop
mechanism.

    

### [[2112.01148] FIBA: Frequency-Injection based Backdoor Attack in Medical Image Analysis](http://arxiv.org/abs/2112.01148)


  In recent years, the security of AI systems has drawn increasing research
attention, especially in the medical imaging realm. To develop a secure medical
image analysis (MIA) system, it is a must to study possible backdoor attacks
(BAs), which can embed hidden malicious behaviors into the system. However,
designing a unified BA method that can be applied to various MIA systems is
challenging due to the diversity of imaging modalities (e.g., X-Ray, CT, and
MRI) and analysis tasks (e.g., classification, detection, and segmentation).
Most existing BA methods are designed to attack natural image classification
models, which apply spatial triggers to training images and inevitably corrupt
the semantics of poisoned pixels, leading to the failures of attacking dense
prediction models. To address this issue, we propose a novel
Frequency-Injection based Backdoor Attack method (FIBA) that is capable of
delivering attacks in various MIA tasks. Specifically, FIBA leverages a trigger
function in the frequency domain that can inject the low-frequency information
of a trigger image into the poisoned image by linearly combining the spectral
amplitude of both images. Since it preserves the semantics of the poisoned
image pixels, FIBA can perform attacks on both classification and dense
prediction models. Experiments on three benchmarks in MIA (i.e., ISIC-2019 for
skin lesion classification, KiTS-19 for kidney tumor segmentation, and EAD-2019
for endoscopic artifact detection), validate the effectiveness of FIBA and its
superiority over state-of-the-art methods in attacking MIA models as well as
bypassing backdoor defense. The code will be available at
this https URL.

    

### [[2112.01155] Batch Normalization Tells You Which Filter is Important](http://arxiv.org/abs/2112.01155)


  The goal of filter pruning is to search for unimportant filters to remove in
order to make convolutional neural networks (CNNs) efficient without
sacrificing the performance in the process. The challenge lies in finding
information that can help determine how important or relevant each filter is
with respect to the final output of neural networks. In this work, we share our
observation that the batch normalization (BN) parameters of pre-trained CNNs
can be used to estimate the feature distribution of activation outputs, without
processing of training data. Upon observation, we propose a simple yet
effective filter pruning method by evaluating the importance of each filter
based on the BN parameters of pre-trained CNNs. The experimental results on
CIFAR-10 and ImageNet demonstrate that the proposed method can achieve
outstanding performance with and without fine-tuning in terms of the trade-off
between the accuracy drop and the reduction in computational complexity and
number of parameters of pruned networks.

    

### [[2112.01160] Learning Robust Recommender from Noisy Implicit Feedback](http://arxiv.org/abs/2112.01160)


  The ubiquity of implicit feedback makes it indispensable for building
recommender systems. However, it does not actually reflect the actual
satisfaction of users. For example, in E-commerce, a large portion of clicks do
not translate to purchases, and many purchases end up with negative reviews. As
such, it is of importance to account for the inevitable noises in implicit
feedback. However, little work on recommendation has taken the noisy nature of
implicit feedback into consideration. In this work, we explore the central
theme of denoising implicit feedback for recommender learning, including
training and inference. By observing the process of normal recommender
training, we find that noisy feedback typically has large loss values in the
early stages. Inspired by this observation, we propose a new training strategy
named Adaptive Denoising Training (ADT), which adaptively prunes the noisy
interactions by two paradigms (i.e., Truncated Loss and Reweighted Loss).
Furthermore, we consider extra feedback (e.g., rating) as auxiliary signal and
propose three strategies to incorporate extra feedback into ADT: finetuning,
warm-up training, and colliding inference. We instantiate the two paradigms on
the widely used binary cross-entropy loss and test them on three representative
recommender models. Extensive experiments on three benchmarks demonstrate that
ADT significantly improves the quality of recommendation over normal training
without using extra feedback. Besides, the proposed three strategies for using
extra feedback largely enhance the denoising ability of ADT.

    

### [[2112.01210] Resonating Minds -- Emergent Collaboration Through Hierarchical Active Inference](http://arxiv.org/abs/2112.01210)


  Working together on complex collaborative tasks requires agents to coordinate
their actions. Doing this explicitly or completely prior to the actual
interaction is not always possible nor sufficient. Agents also need to
continuously understand the current actions of others and quickly adapt their
own behavior appropriately. Here we investigate how efficient, automatic
coordination processes at the level of mental states (intentions, goals), which
we call belief resonance, can lead to collaborative situated problem-solving.
We present a model of hierarchical active inference for collaborative agents
(HAICA). It combines efficient Bayesian Theory of Mind processes with a
perception-action system based on predictive processing and active inference.
Belief resonance is realized by letting the inferred mental states of one agent
influence another agent's predictive beliefs about its own goals and
intentions. This way, the inferred mental states influence the agent's own task
behavior without explicit collaborative reasoning. We implement and evaluate
this model in the Overcooked domain, in which two agents with varying degrees
of belief resonance team up to fulfill meal orders. Our results demonstrate
that agents based on HAICA achieve a team performance comparable to recent
state of the art approaches, while incurring much lower computational costs. We
also show that belief resonance is especially beneficial in settings were the
agents have asymmetric knowledge about the environment. The results indicate
that belief resonance and active inference allow for quick and efficient agent
coordination, and thus can serve as a building block for collaborative
cognitive agents.

    

### [[2112.01229] An AI-based Solution for Enhancing Delivery of Digital Learning for Future Teachers](http://arxiv.org/abs/2112.01229)


  There has been a recent and rapid shift to digital learning hastened by the
pandemic but also influenced by ubiquitous availability of digital tools and
platforms now, making digital learning ever more accessible. An integral and
one of the most difficult part of scaling digital learning and teaching is to
be able to assess learner's knowledge and competency. An educator can record a
lecture or create digital content that can be delivered to thousands of
learners but assessing learners is extremely time consuming. In the paper, we
propose an Artificial Intelligence (AI)-based solution namely VidVersityQG for
generating questions automatically from pre-recorded video lectures. The
solution can automatically generate different types of assessment questions
(including short answer, multiple choice, true/false and fill in the blank
questions) based on contextual and semantic information inferred from the
videos. The proposed solution takes a human-centred approach, wherein teachers
are provided the ability to modify/edit any AI generated questions. This
approach encourages trust and engagement of teachers in the use and
implementation of AI in education. The AI-based solution was evaluated for its
accuracy in generating questions by 7 experienced teaching professionals and
117 education videos from multiple domains provided to us by our industry
partner VidVersity. VidVersityQG solution showed promising results in
generating high-quality questions automatically from video thereby
significantly reducing the time and effort for educators in manual question
generation.

    

### [[2112.01231] Internationalizing AI: Evolution and Impact of Distance Factors](http://arxiv.org/abs/2112.01231)


  International collaboration has become imperative in the field of AI.
However, few studies exist concerning how distance factors have affected the
international collaboration in AI research. In this study, we investigate this
problem by using 1,294,644 AI related collaborative papers harvested from the
Microsoft Academic Graph (MAG) dataset. A framework including 13 indicators to
quantify the distance factors between countries from 5 perspectives (i.e.,
geographic distance, economic distance, cultural distance, academic distance,
and industrial distance) is proposed. The relationships were conducted by the
methods of descriptive analysis and regression analysis. The results show that
international collaboration in the field of AI today is not prevalent (only
15.7%). All the separations in international collaborations have increased over
years, except for the cultural distance in masculinity/felinity dimension and
the industrial distance. The geographic distance, economic distance and
academic distances have shown significantly negative relationships with the
degree of international collaborations in the field of AI. The industrial
distance has a significant positive relationship with the degree of
international collaboration in the field of AI. Also, the results demonstrate
that the participation of the United States and China have promoted the
international collaboration in the field of AI. This study provides a
comprehensive understanding of internationalizing AI research in geographic,
economic, cultural, academic, and industrial aspects.

    

### [[2112.01241] Course Difficulty Estimation Based on Mapping of Bloom's Taxonomy and ABET Criteria](http://arxiv.org/abs/2112.01241)


  Current Educational system uses grades or marks to assess the performance of
the student. The marks or grades a students scores depends on different
parameters, the main parameter being the difficulty level of a course.
Computation of this difficulty level may serve as a support for both the
students and teachers to fix the level of training needed for successful
completion of course. In this paper, we proposed a methodology that estimates
the difficulty level of a course by mapping the Bloom's Taxonomy action words
along with Accreditation Board for Engineering and Technology (ABET) criteria
and learning outcomes. The estimated difficulty level is validated based on the
history of grades secured by the students.

    

### [[2112.01242] An AI-based Learning Companion Promoting Lifelong Learning Opportunities for All](http://arxiv.org/abs/2112.01242)


  Artifical Intelligence (AI) in Education has great potential for building
more personalised curricula, as well as democratising education worldwide and
creating a Renaissance of new ways of teaching and learning. We believe this is
a crucial moment for setting the foundations of AI in education in the
beginning of this Fourth Industrial Revolution. This report aims to synthesize
how AI might change (and is already changing) how we learn, as well as what
technological features are crucial for these AI systems in education, with the
end goal of starting this pressing dialogue of how the future of AI in
education should unfold, engaging policy makers, engineers, researchers and
obviously, teachers and learners. This report also presents the advances within
the X5GON project, a European H2020 project aimed at building and deploying a
cross-modal, cross-lingual, cross-cultural, cross-domain and cross-site
personalised learning platform for Open Educational Resources (OER).

    

### [[2112.01261] ViF-SD2E: A Robust Weakly-Supervised Method for Neural Decoding](http://arxiv.org/abs/2112.01261)


  Neural decoding plays a vital role in the interaction between the brain and
outside world. In this paper, we directly decode the movement track of the
finger based on the neural signals of a macaque. The supervised regression
methods may over-fit to actual labels contained with noise and require high
labeling cost, while unsupervised approaches often have unsatisfactory
accuracy. Besides, the spatial and temporal information are often ignored or
not well exploited in these works. This motivates us to propose a robust
weakly-supervised method termed ViF-SD2E for neural decoding. In particular,
ViF-SD2E consists of a space-division (SD) module and a
exploration-exploitation (2E) strategy, to effectively exploit both the spatial
information of the outside world and temporal information of neural activity,
where the SD2E output is compared with the weak 0/1 vision-feedback (ViF) label
for training. Extensive experiments demonstrate the effectiveness of our
method, which can be sometimes comparable to the supervised counterparts.

    

### [[2112.01298] Meaningful human control over AI systems: beyond talking the talk](http://arxiv.org/abs/2112.01298)


  The concept of meaningful human control has been proposed to address
responsibility gaps and mitigate them by establishing conditions that enable a
proper attribution of responsibility for humans (e.g., users, designers and
developers, manufacturers, legislators). However, the relevant discussions
around meaningful human control have so far not resulted in clear requirements
for researchers, designers, and engineers. As a result, there is no consensus
on how to assess whether a designed AI system is under meaningful human
control, making the practical development of AI-based systems that remain under
meaningful human control challenging. In this paper, we address the gap between
philosophical theory and engineering practice by identifying four actionable
properties which AI-based systems must have to be under meaningful human
control. First, a system in which humans and AI algorithms interact should have
an explicitly defined domain of morally loaded situations within which the
system ought to operate. Second, humans and AI agents within the system should
have appropriate and mutually compatible representations. Third, responsibility
attributed to a human should be commensurate with that human's ability and
authority to control the system. Fourth, there should be explicit links between
the actions of the AI agents and actions of humans who are aware of their moral
responsibility. We argue these four properties are necessary for AI systems
under meaningful human control, and provide possible directions to incorporate
them into practice. We illustrate these properties with two use cases,
automated vehicle and AI-based hiring. We believe these four properties will
support practically-minded professionals to take concrete steps toward
designing and engineering for AI systems that facilitate meaningful human
control and responsibility.

    

### [[2112.01317] Monolith to Microservices: Representing Application Software through Heterogeneous GNN](http://arxiv.org/abs/2112.01317)


  Monolith software applications encapsulate all functional capabilities into a
single deployable unit. While there is an intention to maintain clean
separation of functionalities even within the monolith, they tend to get
compromised with the growing demand for new functionalities, changing team
members, tough timelines, non-availability of skill sets, etc. As such
applications age, they become hard to understand and maintain. Therefore,
microservice architectures are increasingly used as they advocate building an
application through multiple smaller sized, loosely coupled functional
services, wherein each service owns a single functional responsibility. This
approach has made microservices architecture as the natural choice for cloud
based applications. But the challenges in the automated separation of
functional modules for the already written monolith code slows down their
migration task.
Graphs are a natural choice to represent software applications. Various
software artifacts like programs, tables and files become nodes in the graph
and the different relationships they share, such as function calls,
inheritance, resource(tables, files) access types (Create, Read, Update,
Delete) can be represented as links in the graph. We therefore deduce this
traditional application decomposition problem to a heterogeneous graph based
clustering task. Our solution is the first of its kind to leverage
heterogeneous graph neural network to learn representations of such diverse
software entities and their relationships for the clustering task. We study the
effectiveness by comparing with works from both software engineering and
existing graph representation based techniques. We experiment with applications
written in an object oriented language like Java and a procedural language like
COBOL and show that our work is applicable across different programming
paradigms.

    

### [[2112.01342] How not to Lie with a Benchmark: Rearranging NLP Leaderboards](http://arxiv.org/abs/2112.01342)


  Comparison with a human is an essential requirement for a benchmark for it to
be a reliable measurement of model capabilities. Nevertheless, the methods for
model comparison could have a fundamental flaw - the arithmetic mean of
separate metrics is used for all tasks of different complexity, different size
of test and training sets.
In this paper, we examine popular NLP benchmarks' overall scoring methods and
rearrange the models by geometric and harmonic mean (appropriate for averaging
rates) according to their reported results. We analyze several popular
benchmarks including GLUE, SuperGLUE, XGLUE, and XTREME. The analysis shows
that e.g. human level on SuperGLUE is still not reached, and there is still
room for improvement for the current models.

    

### [[2112.01404] LOGEN: Few-shot Logical Knowledge-Conditioned Text Generation with Self-training](http://arxiv.org/abs/2112.01404)


  Natural language generation from structured data mainly focuses on
surface-level descriptions, suffering from uncontrollable content selection and
low fidelity. Previous works leverage logical forms to facilitate logical
knowledge-conditioned text generation. Though achieving remarkable progress,
they are data-hungry, which makes the adoption for real-world applications
challenging with limited data. To this end, this paper proposes a unified
framework for logical knowledge-conditioned text generation in the few-shot
setting. With only a few seeds logical forms (e.g., 20/100 shot), our approach
leverages self-training and samples pseudo logical forms based on content and
structure consistency. Experimental results demonstrate that our approach can
obtain better few-shot performance than baselines.

    

### [[2112.01441] A Review of SHACL: From Data Validation to Schema Reasoning for RDF Graphs](http://arxiv.org/abs/2112.01441)


  We present an introduction and a review of the Shapes Constraint Language
(SHACL), the W3C recommendation language for validating RDF data. A SHACL
document describes a set of constraints on RDF nodes, and a graph is valid with
respect to the document if its nodes satisfy these constraints. We revisit the
basic concepts of the language, its constructs and components and their
interaction. We review the different formal frameworks used to study this
language and the different semantics proposed. We examine a number of related
problems, from containment and satisfiability to the interaction of SHACL with
inference rules, and exhibit how different modellings of the language are
useful for different problems. We also cover practical aspects of SHACL,
discussing its implementations and state of adoption, to present a holistic
review useful to practitioners and theoreticians alike.

    

### [[2112.01442] Learning Large-scale Network Embedding from Representative Subgraph](http://arxiv.org/abs/2112.01442)


  We study the problem of large-scale network embedding, which aims to learn
low-dimensional latent representations for network mining applications. Recent
research in the field of network embedding has led to significant progress such
as DeepWalk, LINE, NetMF, NetSMF. However, the huge size of many real-world
networks makes it computationally expensive to learn network embedding from the
entire network. In this work, we present a novel network embedding method
called "NES", which learns network embedding from a small representative
subgraph. NES leverages theories from graph sampling to efficiently construct
representative subgraph with smaller size which can be used to make inferences
about the full network, enabling significantly improved efficiency in embedding
learning. Then, NES computes the network embedding from this representative
subgraph, efficiently. Compared with well-known methods, extensive experiments
on networks of various scales and types demonstrate that NES achieves
comparable performance and significant efficiency superiority.

    

### [[2112.01451] Architecting and Visualizing Deep Reinforcement Learning Models](http://arxiv.org/abs/2112.01451)


  To meet the growing interest in Deep Reinforcement Learning (DRL), we sought
to construct a DRL-driven Atari Pong agent and accompanying visualization tool.
Existing approaches do not support the flexibility required to create an
interactive exhibit with easily-configurable physics and a human-controlled
player. Therefore, we constructed a new Pong game environment, discovered and
addressed a number of unique data deficiencies that arise when applying DRL to
a new environment, architected and tuned a policy gradient based DRL model,
developed a real-time network visualization, and combined these elements into
an interactive display to help build intuition and awareness of the mechanics
of DRL inference.

    

### [[2112.01515] TransFGU: A Top-down Approach to Fine-Grained Unsupervised Semantic Segmentation](http://arxiv.org/abs/2112.01515)


  Unsupervised semantic segmentation aims to obtain high-level semantic
representation on low-level visual features without manual annotations. Most
existing methods are bottom-up approaches that try to group pixels into regions
based on their visual cues or certain predefined rules. As a result, it is
difficult for these bottom-up approaches to generate fine-grained semantic
segmentation when coming to complicated scenes with multiple objects and some
objects sharing similar visual appearance. In contrast, we propose the first
top-down unsupervised semantic segmentation framework for fine-grained
segmentation in extremely complicated scenarios. Specifically, we first obtain
rich high-level structured semantic concept information from large-scale vision
data in a self-supervised learning manner, and use such information as a prior
to discover potential semantic categories presented in target datasets.
Secondly, the discovered high-level semantic categories are mapped to low-level
pixel features by calculating the class activate map (CAM) with respect to
certain discovered semantic representation. Lastly, the obtained CAMs serve as
pseudo labels to train the segmentation module and produce final semantic
segmentation. Experimental results on multiple semantic segmentation benchmarks
show that our top-down unsupervised segmentation is robust to both
object-centric and scene-centric datasets under different semantic granularity
levels, and outperforms all the current state-of-the-art bottom-up methods. Our
code is available at \url{this https URL}.

    

### [[1804.00596] TacticToe: Learning to Prove with Tactics](http://arxiv.org/abs/1804.00596)


  We implement a automated tactical prover TacticToe on top of the HOL4
interactive theorem prover. TacticToe learns from human proofs which
mathematical technique is suitable in each proof situation. This knowledge is
then used in a Monte Carlo tree search algorithm to explore promising
tactic-level proof paths. On a single CPU, with a time limit of 60 seconds,
TacticToe proves 66.4 percent of the 7164 theorems in HOL4's standard library,
whereas E prover with auto-schedule solves 34.5 percent. The success rate rises
to 69.0 percent by combining the results of TacticToe and E prover.

    

### [[2009.14539] Case-based Abductive Natural Language Inference](http://arxiv.org/abs/2009.14539)


  Existing accounts of explanation emphasise the role of prior experience in
the solution of new problems. However, most of the contemporary models for
multi-hop textual inference construct explanations considering each test case
in isolation. This paradigm is known to suffer from semantic drift, which
causes the construction of spurious explanations leading to wrong conclusions.
In contrast, we investigate an abductive framework for explainable multi-hop
inference that adopts the retrieve-reuse-revise paradigm largely studied in
case-based reasoning. Specifically, we present a novel framework that addresses
and explains unseen inference problems by retrieving and adapting prior natural
language explanations from similar training examples. We empirically evaluate
the case-based abductive framework on downstream commonsense and scientific
reasoning tasks. Our experiments demonstrate that the proposed framework can be
effectively integrated with sparse and dense pre-trained encoding mechanisms or
downstream transformers, achieving strong performance when compared to existing
explainable approaches. Moreover, we study the impact of the
retrieve-reuse-revise paradigm on explainability and semantic drift, showing
that it boosts the quality of the constructed explanations, resulting in
improved downstream inference performance.

    

### [[2010.09231] CT-CPP: Coverage Path Planning for 3D Terrain Reconstruction Using Dynamic Coverage Trees](http://arxiv.org/abs/2010.09231)


  This letter addresses the 3D coverage path planning (CPP) problem for terrain
reconstruction of unknown obstacle rich environments. Due to sensing
limitations, the proposed method, called CT-CPP, performs layered scanning of
the 3D region to collect terrain data, where the traveling sequence is
optimized using the concept of a coverage tree (CT) with a TSP-inspired tree
traversal strategy. The CT-CPP method is validated on a high-fidelity
underwater simulator and the results are compared to an existing terrain
following CPP method. The results show that CT-CPP yields significant reduction
in trajectory length, energy consumption, and reconstruction error.

    

### [[2102.03064] "I Don't Think So": Summarizing Policy Disagreements for Agent Comparison](http://arxiv.org/abs/2102.03064)


  With Artificial Intelligence on the rise, human interaction with autonomous
agents becomes more frequent. Effective human-agent collaboration requires
users to understand the agent's behavior, as failing to do so may cause reduced
productivity, misuse or frustration. Agent strategy summarization methods are
used to describe the strategy of an agent to its destined user through
demonstration. A summary's objective is to maximize the user's understanding of
the agent's aptitude by showcasing its behaviour in a selected set of world
states. While shown to be useful, we show that current methods are limited when
tasked with comparing between agents, as each summary is independently
generated for a specific agent. In this paper, we propose a novel method for
generating dependent and contrastive summaries that emphasize the differences
between agent policies by identifying states in which the agents disagree on
the best course of action. We conduct user studies to assess the usefulness of
disagreement-based summaries for identifying superior agents and conveying
agent differences. Results show disagreement-based summaries lead to improved
user performance compared to summaries generated using HIGHLIGHTS, a strategy
summarization algorithm which generates summaries for each agent independently.

    

### [[2104.03414] PrivateSNN: Privacy-Preserving Spiking Neural Networks](http://arxiv.org/abs/2104.03414)


  How can we bring both privacy and energy-efficiency to a neural system? In
this paper, we propose PrivateSNN, which aims to build low-power Spiking Neural
Networks (SNNs) from a pre-trained ANN model without leaking sensitive
information contained in a dataset. Here, we tackle two types of leakage
problems: 1) Data leakage is caused when the networks access real training data
during an ANN-SNN conversion process. 2) Class leakage is caused when
class-related features can be reconstructed from network parameters. In order
to address the data leakage issue, we generate synthetic images from the
pre-trained ANNs and convert ANNs to SNNs using the generated images. However,
converted SNNs remain vulnerable to class leakage since the weight parameters
have the same (or scaled) value with respect to ANN parameters. Therefore, we
encrypt SNN weights by training SNNs with a temporal spike-based learning rule.
Updating weight parameters with temporal data makes SNNs difficult to be
interpreted in the spatial domain. We observe that the encrypted PrivateSNN
eliminates data and class leakage issues with a slight performance drop (less
than ~2) and significant energy-efficiency gain (about 55x) compared to the
standard ANN. We conduct extensive experiments on various datasets including
CIFAR10, CIFAR100, and TinyImageNet, highlighting the importance of
privacy-preserving SNN training.

    

### [[2106.07824] Communicating Natural Programs to Humans and Machines](http://arxiv.org/abs/2106.07824)


  The Abstraction and Reasoning Corpus (ARC) is a set of procedural tasks that
tests an agent's ability to flexibly solve novel problems. While most ARC tasks
are easy for humans, they are challenging for state-of-the-art AI. What makes
building intelligent systems that can generalize to novel situations such as
ARC difficult? We posit that the answer might be found by studying the
difference of \emph{language}: While humans readily generate and interpret
instructions in a general language, computer systems are shackled to a narrow
domain-specific language that they can precisely execute. We present LARC, the
\textit{Language-complete ARC}: a collection of natural language descriptions
by a group of human participants who instruct each other on how to solve ARC
tasks using language alone, which contains successful instructions for 88\% of
the ARC tasks. We analyze the collected instructions as `natural programs',
finding that while they resemble computer programs, they are distinct in two
ways: First, they contain a wide range of primitives; Second, they frequently
leverage communicative strategies beyond directly executable codes. We
demonstrate that these two distinctions prevent current program synthesis
techniques from leveraging LARC to its full potential, and give concrete
suggestions on how to build the next-generation program synthesizers.

    

### [[2112.01058] A Foreground-Background queueing model with speed or capacity modulation](http://arxiv.org/abs/2112.01058)


  The models studied in the steady state involve two queues which are served
either by a single server whose speed depends on the number of jobs present, or
by several parallel servers whose number may be controlled dynamically. Job
service times have a two-phase Coxian distribution and the second phase is
given lower priority than the first. The trade-offs between holding costs and
energy consumption costs are examined by means of a suitable cost functions.
Two different two-dimensional Markov process are solved exactly. The solutions
are used in several numerical experiments. Some counter-intuitive results are
observed.

    

### [[2112.01394] Dynamic Sparse Tensor Algebra Compilation](http://arxiv.org/abs/2112.01394)


  This paper shows how to generate efficient tensor algebra code that compute
on dynamic sparse tensors, which have sparsity structures that evolve over
time. We propose a language for precisely specifying recursive, pointer-based
data structures, and we show how this language can express a wide range of
dynamic data structures that support efficient modification, such as linked
lists, binary search trees, and B-trees. We then describe how, given high-level
specifications of such data structures, a compiler can generate code to
efficiently iterate over and compute with dynamic sparse tensors that are
stored in the aforementioned data structures. Furthermore, we define an
abstract interface that captures how nonzeros can be inserted into dynamic data
structures, and we show how this abstraction guides a compiler to emit
efficient code that store the results of sparse tensor algebra computations in
dynamic data structures.
We evaluate our technique and find that it generates efficient dynamic sparse
tensor algebra kernels. Code that our technique emits to compute the main
kernel of the PageRank algorithm is 1.05$\times$ as fast as Aspen, a
state-of-the-art dynamic graph processing framework. Furthermore, our technique
outperforms PAM, a parallel ordered (key-value) maps library, by 7.40$\times$
when used to implement element-wise addition of a dynamic sparse matrix to a
static sparse matrix.

    

### [[2112.01397] Efficient Calling Conventions for Irregular Architectures](http://arxiv.org/abs/2112.01397)


  We empirically evaluated thousands of different C calling conventions for
irregular microcontroller architectures, and found potential for improvement
over the calling conventions previously used in the Small Device C Compiler
(SDCC). The improvements in code size and speed are substantial enough that
SDCC made changes to its default calling convention, breaking ABI
compatibility.

    

### [[2004.13472] Linear Dependent Type Theory for Quantum Programming Languages](http://arxiv.org/abs/2004.13472)


  Modern quantum programming languages integrate quantum resources and
classical control. They must, on the one hand, be linearly typed to reflect the
no-cloning property of quantum resources. On the other hand, high-level and
practical languages should also support quantum circuits as first-class
citizens, as well as families of circuits that are indexed by some classical
parameters. Quantum programming languages thus need linear dependent type
theory. This paper defines a general semantic structure for such a type theory
via certain fibrations of monoidal categories. The categorical model of the
quantum circuit description language Proto-Quipper-M by Rios and Selinger
(2017) constitutes an example of such a fibration, which means that the
language can readily be integrated with dependent types. We then devise both a
general linear dependent type system and a dependently typed extension of
Proto-Quipper-M, and provide them with operational semantics as well as a
prototype implementation.

    