
## 2021-7-16

### [[2107.06400] Using BERT Encoding to Tackle the Mad-lib Attack in SMS Spam Detection](http://arxiv.org/abs/2107.06400)


  One of the stratagems used to deceive spam filters is to substitute vocables
with synonyms or similar words that turn the message unrecognisable by the
detection algorithms. In this paper we investigate whether the recent
development of language models sensitive to the semantics and context of words,
such as Google's BERT, may be useful to overcome this adversarial attack
(called "Mad-lib" as per the word substitution game). Using a dataset of 5572
SMS spam messages, we first established a baseline of detection performance
using widely known document representation models (BoW and TFIDF) and the novel
BERT model, coupled with a variety of classification algorithms (Decision Tree,
kNN, SVM, Logistic Regression, Naive Bayes, Multilayer Perceptron). Then, we
built a thesaurus of the vocabulary contained in these messages, and set up a
Mad-lib attack experiment in which we modified each message of a held out
subset of data (not used in the baseline experiment) with different rates of
substitution of original words with synonyms from the thesaurus. Lastly, we
evaluated the detection performance of the three representation models (BoW,
TFIDF and BERT) coupled with the best classifier from the baseline experiment
(SVM). We found that the classic models achieved a 94% Balanced Accuracy (BA)
in the original dataset, whereas the BERT model obtained 96%. On the other
hand, the Mad-lib attack experiment showed that BERT encodings manage to
maintain a similar BA performance of 96% with an average substitution rate of
1.82 words per message, and 95% with 3.34 words substituted per message. In
contrast, the BA performance of the BoW and TFIDF encoders dropped to chance.
These results hint at the potential advantage of BERT models to combat these
type of ingenious attacks, offsetting to some extent for the inappropriate use
of semantic relationships in language.

    

### [[2107.07072] EICO: Energy-Harvesting Long-Range Environmental Sensor Nodes with Energy-Information Dynamic Co-Optimization](http://arxiv.org/abs/2107.07072)


  Intensive research on energy harvested sensor nodes with traditional battery
powered devices has been driven by the challenges in achieving the stringent
design goals of battery lifetime, information accuracy, transmission distance,
and cost. This challenge is further amplified by the inherent power intensive
nature of long-range communication when sensor networks are required to span
vast areas such as agricultural fields and remote terrain. Solar power is a
common energy source is wireless sensor nodes, however, it is not reliable due
to fluctuations in power stemming from the changing seasons and weather
conditions. This paper tackles these issues by presenting a
perpetually-powered, energy-harvesting sensor node which utilizes a minimally
sized solar cell and is capable of long range communication by dynamically
co-optimizing energy consumption and information transfer, termed as
Energy-Information Dynamic Co-Optimization (EICO). This energy-information
intelligence is achieved by adaptive duty cycling of information transfer based
on the total amount of energy available from the harvester and charge storage
element to optimize the energy consumption of the sensor node, while employing
in-sensor analytics (ISA) to minimize loss of information. This is the first
reported sensor node < 35cm2 in dimension, which is capable of long-range
communication over > 1Km at continuous information transfer rates of upto 1
packet/second which is enabled by EICO and ISA.

    

### [[2107.07217] Satellite Communication Digital Twin for Evaluating Novel Solutions: Dynamic Link Emulation Architecture](http://arxiv.org/abs/2107.07217)


  This paper presents the design and architecture of a network emulator whose
links' parameters (such as delay and bandwidth) vary at different time
instances. The emulator is used as a digital twin for satellite communication
systems, in order to test and evaluate novel solutions before their final
deployment. To achieve such a goal, different existing technologies are
carefully combined to emulate link dynamicity, automatic traffic generation,
and overall topology emulation. Since emulating asymmetric dynamic links (as
required in satellite communications) is far from trivial, we provide a
detailed design architecture for solving such task. Experimental results show
the precision of our dynamic assignments and the overall flexibility of the
proposed solution.

    

### [[2107.07263] Increasing Transmission Distance in THz systems with MDPC Code and Auxiliary Channel](http://arxiv.org/abs/2107.07263)


  We analyze whether multidimensional parity check (MDPC) code and an auxiliary
channel can improve the throughput and extend the THz transmission distance.
While channel quality is addressed by various coding approaches, and an
effective THz transmission system configuration is enabled by other approaches
with additional channels, their combination is new with the potential for
significant improvements in quality of the data transmission. Our specific
solution is designed to correct data bits at the physical layer by a low
complexity erasure code (MDPC), whereby original data and parity data are
simultaneously transferred over two parallel THz channels, including one main
channel and one additional channel. The results are theoretically analyzed to
see that our new solution can improve throughput, support higher modulation
levels and transfer data over the longer distances with THz communications.

    

### [[2107.07406] Design and Implementation of an IoT Based LPG and CO Gases Monitoring System](http://arxiv.org/abs/2107.07406)


  Nowadays use of liquefied petroleum gas (LPG) has increased. LPG is an
asphyxiating, volatile and highly flammable gas. In a LPG leak situation,
potential health accidents are increased either by inhalation or by combustion
of the gas. On the other hand, carbon monoxide (CO) is a toxic gas that comes
mainly from combustion in car engines. Breathing CO-polluted air can cause
dizziness, fainting, breathing problems, and sometimes death. To prevent health
accidents, including explosions, in open or closed environments, remote and
real-time monitoring of the concentration levels of CO and LPG gases has become
a necessity. The aim of this work is to demonstrate the use of Internet of
Things (IoT) techniques to design and build a telemetry system to monitor in
real-time the concentration of GLP and CO gases in the surrounding air. To
implement this work, as central hardware there is a microcontroller, CO and PLG
sensors on the electronic station. Besides, Amazon Web Services (AWS) was used
as an IoT platform and data storage in the cloud. The main result was a
telematics system to monitor in real time the concentrations of both GLP and CO
gases, whose data is accessible from any device with internet access through a
website. Field tests have been successful and have shown that the proposed
system is an efficient and low-cost option.

    

### [[2107.07441] Reliability Analysis of Slotted Aloha with Capture for an OWC-based IoT system](http://arxiv.org/abs/2107.07441)


  In this article, we consider a random access scheme for an indoor Internet of
Things (IoT) framework that uses optical wireless communication (OWC). We focus
on a Slotted ALOHA (SA)-based solution where a number of OWC IoT users contend
to send data to a central OWC receiver. In any given slot, and for a randomly
selected active user, we consider the reliability of decoding the user's data
packet at the receiver. This is done by deriving the
signal-to-noise-and-interference-ratio (SINR) statistics from a randomly chosen
user and evaluating the probability that the user's SINR is below a given
threshold. By placing our analysis in the context of an indoor OWC IoT uplink
setup, and employing the standard OWC channel model, we investigate the
trade-offs between the reliability and the OWC system parameters such as the
cell area or the transmitter's semi-angle. We obtain valuable insights into the
design of an SA-based random access solution for a typical indoor OWC cell.

    

### [[2107.07442] MXDAG: A Hybrid Abstraction for Cluster Applications](http://arxiv.org/abs/2107.07442)


  Distributed applications, such as database queries and distributed training,
consist of both compute and network tasks. DAG-based abstraction primarily
targets compute tasks and has no explicit network-level scheduling. In
contrast, Coflow abstraction collectively schedules network flows among compute
tasks but lacks the end-to-end view of the application DAG. Because of the
dependencies and interactions between these two types of tasks, it is
sub-optimal to only consider one of them. We argue that co-scheduling of both
compute and network tasks can help applications towards the globally optimal
end-to-end performance. However, none of the existing abstractions can provide
fine-grained information for co-scheduling. We propose MXDAG, an abstraction to
treat both compute and network tasks explicitly. It can capture the
dependencies and interactions of both compute and network tasks leading to
improved application performance.

    

### [[2007.11427] Formal Analysis of EDHOC Key Establishment for Constrained IoT Devices](http://arxiv.org/abs/2007.11427)


  Constrained IoT devices are becoming ubiquitous in society and there is a
need for secure communication protocols that respect the constraints under
which these devices operate. EDHOC is an authenticated key establishment
protocol for constrained IoT devices, currently being standardized by the
Internet Engineering Task Force (IETF). A rudimentary version of EDHOC with
only two key establishment methods was formally analyzed in 2018. Since then,
the protocol has evolved significantly and several new key establishment
methods have been added. In this paper, we present a formal analysis of all
EDHOC methods in an enhanced symbolic Dolev-Yao model using the Tamarin tool.
We show that not all methods satisfy the authentication notion injective of
agreement, but that they all do satisfy a notion of implicit authentication, as
well as Perfect Forward Secrecy (PFS) of the session key material. We identify
other weaknesses to which we propose improvements. For example, a party may
intend to establish a session key with a certain peer, but end up establishing
it with another, trusted but compromised, peer. We communicated our findings
and proposals to the IETF, which has incorporated some of these in newer
versions of the standard.

    

### [[2102.06607] Intelligent Software Web Agents: A Gap Analysis](http://arxiv.org/abs/2102.06607)


  Semantic web technologies have shown their effectiveness, especially when it
comes to knowledge representation, reasoning, and data integration. However,
the original semantic web vision, whereby machine readable web data could be
automatically actioned upon by intelligent software web agents, has yet to be
realised. In order to better understand the existing technological
opportunities and challenges, in this paper we examine the status quo in terms
of intelligent software web agents, guided by research with respect to
requirements and architectural components, coming from the agents community. We
use the identified requirements to both further elaborate on the semantic web
agent motivating use case scenario, and to summarise different perspectives on
the requirements from the semantic web agent literature. We subsequently
propose a hybrid semantic web agent architecture, and use the various
components and subcomponents in order to provide a focused discussion on the
role played by existing semantic web standards and community activities.
Finally, we highlight open research opportunities and challenges and take a
broader perspective of the research by discussing the potential for intelligent
software web agents as an enabling technology for emerging domains, such as
digital assistants, cloud computing, and the internet of things.

    

### [[2103.00022] Synthesizing Safe and Efficient Kernel Extensions for Packet Processing](http://arxiv.org/abs/2103.00022)


  Extended Berkeley Packet Filter (BPF) has emerged as a powerful method to
extend packet-processing functionality in the Linux operating system. BPF
allows users to write code in high-level languages (like C or Rust) and execute
them at specific hooks in the kernel, such as the network device driver. To
ensure safe execution of a user-developed BPF program in kernel context, Linux
uses an in-kernel static checker. The checker allows a program to execute only
if it can prove that the program is crash-free, always accesses memory within
safe bounds, and avoids leaking kernel data.
BPF programming is not easy. One, even modest-sized BPF programs are deemed
too large to analyze and rejected by the kernel checker. Two, the kernel
checker may incorrectly determine that a BPF program exhibits unsafe behaviors.
Three, even small performance optimizations to BPF code (e.g., 5% gains) must
be meticulously hand-crafted by expert developers. Traditional optimizing
compilers for BPF are often inadequate since the kernel checker's safety
constraints are incompatible with rule-based optimizations.
We present K2, a program-synthesis-based compiler that automatically
optimizes BPF bytecode with formal correctness and safety guarantees. K2
produces code with 6--26% reduced size, 1.36%--55.03% lower average
packet-processing latency, and 0--4.75% higher throughput (packets per second
per core) relative to the best clang-compiled program, across benchmarks drawn
from Cilium, Facebook, and the Linux kernel. K2 incorporates several
domain-specific techniques to make synthesis practical by accelerating
equivalence-checking of BPF programs by 6 orders of magnitude.

    

### [[2103.14917] Reinforcement Learning Random Access for Delay-Constrained Heterogeneous Wireless Networks: A Two-User Case](http://arxiv.org/abs/2103.14917)


  In this paper, we investigate the random access problem for a
delay-constrained heterogeneous wireless network. As a first attempt to study
this new problem, we consider a network with two users who deliver
delay-constrained traffic to an access point (AP) via a common unreliable
collision wireless channel. We assume that one user (called user 1) adopts
ALOHA and we optimize the random access scheme of the other user (called user
2). The most intriguing part of this problem is that user 2 does not know the
information of user 1 but needs to maximize the system timely throughput. Such
a paradigm of collaboratively sharing spectrum is envisioned by DARPA to better
dynamically match the supply and demand in the future [1], [2]. We first
propose a Markov Decision Process (MDP) formulation to derive a modelbased
upper bound, which can quantify the performance gap of any designed schemes. We
then utilize reinforcement learning (RL) to design an R-learning-based [3]-[5]
random access scheme, called TSRA. We finally carry out extensive simulations
to show that TSRA achieves close-to-upper-bound performance and better
performance than the existing baseline DLMA [6], which is our counterpart
scheme for delay-unconstrained heterogeneous wireless network. All source code
is publicly available in this https URL.

    

### [[2107.06898] Towards quantifying information flows: relative entropy in deep neural networks and the renormalization group](http://arxiv.org/abs/2107.06898)


  We investigate the analogy between the renormalization group (RG) and deep
neural networks, wherein subsequent layers of neurons are analogous to
successive steps along the RG. In particular, we quantify the flow of
information by explicitly computing the relative entropy or Kullback-Leibler
divergence in both the one- and two-dimensional Ising models under decimation
RG, as well as in a feedforward neural network as a function of depth. We
observe qualitatively identical behavior characterized by the monotonic
increase to a parameter-dependent asymptotic value. On the quantum field theory
side, the monotonic increase confirms the connection between the relative
entropy and the c-theorem. For the neural networks, the asymptotic behavior may
have implications for various information maximization methods in machine
learning, as well as for disentangling compactness and generalizability.
Furthermore, while both the two-dimensional Ising model and the random neural
networks we consider exhibit non-trivial critical points, the relative entropy
appears insensitive to the phase structure of either system. In this sense,
more refined probes are required in order to fully elucidate the flow of
information in these models.

    

### [[2107.06908] Understanding Failures in Out-of-Distribution Detection with Deep Generative Models](http://arxiv.org/abs/2107.06908)


  Deep generative models (DGMs) seem a natural fit for detecting
out-of-distribution (OOD) inputs, but such models have been shown to assign
higher probabilities or densities to OOD images than images from the training
distribution. In this work, we explain why this behavior should be attributed
to model misestimation. We first prove that no method can guarantee performance
beyond random chance without assumptions on which out-distributions are
relevant. We then interrogate the typical set hypothesis, the claim that
relevant out-distributions can lie in high likelihood regions of the data
distribution, and that OOD detection should be defined based on the data
distribution's typical set. We highlight the consequences implied by assuming
support overlap between in- and out-distributions, as well as the arbitrariness
of the typical set for OOD detection. Our results suggest that estimation error
is a more plausible explanation than the misalignment between likelihood-based
OOD detection and out-distributions of interest, and we illustrate how even
minimal estimation error can lead to OOD detection failures, yielding
implications for future work in deep generative modeling and OOD detection.

    

### [[2107.06917] A Field Guide to Federated Optimization](http://arxiv.org/abs/2107.06917)


  Federated learning and analytics are a distributed approach for
collaboratively learning models (or statistics) from decentralized data,
motivated by and designed for privacy protection. The distributed learning
process can be formulated as solving federated optimization problems, which
emphasize communication efficiency, data heterogeneity, compatibility with
privacy and system requirements, and other constraints that are not primary
considerations in other problem settings. This paper provides recommendations
and guidelines on formulating, designing, evaluating and analyzing federated
optimization algorithms through concrete examples and practical implementation,
with a focus on conducting effective simulations to infer real-world
performance. The goal of this work is not to survey the current literature, but
to inspire researchers and practitioners to design federated learning
algorithms that can be used in various practical applications.

    

### [[2107.06925] Chimera: Efficiently Training Large-Scale Neural Networks with Bidirectional Pipelines](http://arxiv.org/abs/2107.06925)


  Training large deep learning models at scale is very challenging. This paper
proposes Chimera, a novel pipeline parallelism scheme which combines
bidirectional pipelines for efficiently training large-scale models. Chimera is
a synchronous approach and therefore no loss of accuracy, which is more
convergence-friendly than asynchronous approaches. Compared with the latest
synchronous pipeline approach, Chimera reduces the number of bubbles by up to
50%; benefiting from the sophisticated scheduling of bidirectional pipelines,
Chimera has a more balanced activation memory consumption. Evaluations are
conducted on Transformer based language models. For a GPT-2 model with 1.3
billion parameters running on 2,048 GPU nodes of the Piz Daint supercomputer,
Chimera improves the training throughput by 1.16x-2.34x over the
state-of-the-art synchronous and asynchronous pipeline approaches.

    

### [[2107.06929] Feature Shift Detection: Localizing Which Features Have Shifted via Conditional Distribution Tests](http://arxiv.org/abs/2107.06929)


  While previous distribution shift detection approaches can identify if a
shift has occurred, these approaches cannot localize which specific features
have caused a distribution shift -- a critical step in diagnosing or fixing any
underlying issue. For example, in military sensor networks, users will want to
detect when one or more of the sensors has been compromised, and critically,
they will want to know which specific sensors might be compromised. Thus, we
first define a formalization of this problem as multiple conditional
distribution hypothesis tests and propose both non-parametric and parametric
statistical tests. For both efficiency and flexibility, we then propose to use
a test statistic based on the density model score function (i.e. gradient with
respect to the input) -- which can easily compute test statistics for all
dimensions in a single forward and backward pass. Any density model could be
used for computing the necessary statistics including deep density models such
as normalizing flows or autoregressive models. We additionally develop methods
for identifying when and where a shift occurs in multivariate time-series data
and show results for multiple scenarios using realistic attack models on both
simulated and real world data.

    

### [[2107.06936] Performance of Bayesian linear regression in a model with mismatch](http://arxiv.org/abs/2107.06936)


  For a model of high-dimensional linear regression with random design, we
analyze the performance of an estimator given by the mean of a log-concave
Bayesian posterior distribution with gaussian prior. The model is mismatched in
the following sense: like the model assumed by the statistician, the
labels-generating process is linear in the input data, but both the classifier
ground-truth prior and gaussian noise variance are unknown to her. This
inference model can be rephrased as a version of the Gardner model in spin
glasses and, using the cavity method, we provide fixed point equations for
various overlap order parameters, yielding in particular an expression for the
mean-square reconstruction error on the classifier (under an assumption of
uniqueness of solutions). As a direct corollary we obtain an expression for the
free energy. Similar models have already been studied by Shcherbina and Tirozzi
and by Talagrand, but our arguments are more straightforward and some
assumptions are relaxed. An interesting consequence of our analysis is that in
the random design setting of ridge regression, the performance of the posterior
mean is independent of the noise variance (or "temperature") assumed by the
statistician, and matches the one of the usual (zero temperature) ridge
estimator.

    

### [[2107.06943] FetalNet: Multi-task deep learning framework for fetal ultrasound biometric measurements](http://arxiv.org/abs/2107.06943)


  In this paper, we propose an end-to-end multi-task neural network called
FetalNet with an attention mechanism and stacked module for spatio-temporal
fetal ultrasound scan video analysis. Fetal biometric measurement is a standard
examination during pregnancy used for the fetus growth monitoring and
estimation of gestational age and fetal weight. The main goal in fetal
ultrasound scan video analysis is to find proper standard planes to measure the
fetal head, abdomen and femur. Due to natural high speckle noise and shadows in
ultrasound data, medical expertise and sonographic experience are required to
find the appropriate acquisition plane and perform accurate measurements of the
fetus. In addition, existing computer-aided methods for fetal US biometric
measurement address only one single image frame without considering temporal
features. To address these shortcomings, we propose an end-to-end multi-task
neural network for spatio-temporal ultrasound scan video analysis to
simultaneously localize, classify and measure the fetal body parts. We propose
a new encoder-decoder segmentation architecture that incorporates a
classification branch. Additionally, we employ an attention mechanism with a
stacked module to learn salient maps to suppress irrelevant US regions and
efficient scan plane localization. We trained on the fetal ultrasound video
comes from routine examinations of 700 different patients. Our method called
FetalNet outperforms existing state-of-the-art methods in both classification
and segmentation in fetal ultrasound video recordings.

    

### [[2107.06944] On the impossibility of non-trivial accuracy under fairness constraints](http://arxiv.org/abs/2107.06944)


  One of the main concerns about fairness in machine learning (ML) is that, in
order to achieve it, one may have to renounce to some accuracy. Having this
trade-off in mind, Hardt et al. have proposed the notion of equal opportunities
(EO), designed so as to be compatible with accuracy. In fact, it can be shown
that if the source of input data is deterministic, the two notions go well
along with each other. In the probabilistic case, however, things change.
As we show, there are probabilistic data sources for which EO can only be
achieved at the total detriment of accuracy, i.e. among the models that achieve
EO, those whose prediction does not depend on the input have the highest
accuracy.

    

### [[2107.06946] Towards Quantifying the Carbon Emissions of Differentially Private Machine Learning](http://arxiv.org/abs/2107.06946)


  In recent years, machine learning techniques utilizing large-scale datasets
have achieved remarkable performance. Differential privacy, by means of adding
noise, provides strong privacy guarantees for such learning algorithms. The
cost of differential privacy is often a reduced model accuracy and a lowered
convergence speed. This paper investigates the impact of differential privacy
on learning algorithms in terms of their carbon footprint due to either longer
run-times or failed experiments. Through extensive experiments, further
guidance is provided on choosing the noise levels which can strike a balance
between desired privacy levels and reduced carbon emissions.

    

### [[2107.06955] HTLM: Hyper-Text Pre-Training and Prompting of Language Models](http://arxiv.org/abs/2107.06955)


  We introduce HTLM, a hyper-text language model trained on a large-scale web
crawl. Modeling hyper-text has a number of advantages: (1) it is easily
gathered at scale, (2) it provides rich document-level and end-task-adjacent
supervision (e.g. class and id attributes often encode document category
information), and (3) it allows for new structured prompting that follows the
established semantics of HTML (e.g. to do zero-shot summarization by infilling
title tags for a webpage that contains the input text). We show that
pretraining with a BART-style denoising loss directly on simplified HTML
provides highly effective transfer for a wide range of end tasks and
supervision levels. HTLM matches or exceeds the performance of comparably sized
text-only LMs for zero-shot prompting and fine-tuning for classification
benchmarks, while also setting new state-of-the-art performance levels for
zero-shot summarization. We also find that hyper-text prompts provide more
value to HTLM, in terms of data efficiency, than plain text prompts do for
existing LMs, and that HTLM is highly effective at auto-prompting itself, by
simply generating the most likely hyper-text formatting for any available
training data. We will release all code and models to support future HTLM
research.

    

### [[2107.06960] Memory-Aware Fusing and Tiling of Neural Networks for Accelerated Edge Inference](http://arxiv.org/abs/2107.06960)


  A rising research challenge is running costly machine learning (ML) networks
locally on resource-constrained edge devices. ML networks with large
convolutional layers can easily exceed available memory, increasing latency due
to excessive swapping. Previous memory reduction techniques such as pruning and
quantization reduce model accuracy and often require retraining. Alternatively,
distributed methods partition the convolutions into equivalent smaller
sub-computations, but the implementations introduce communication costs and
require a network of devices. However, a distributed partitioning approach can
also be used to run in a reduced memory footprint on a single device by
subdividing the network into smaller operations.
This report extends prior work on distributed partitioning using tiling and
fusing of convolutional layers into a memory-aware execution on a single
device. Our approach extends prior fusing strategies to allow for two groups of
convolutional layers that are fused and tiled independently. This approach
reduces overhead via data reuse, and reduces the memory footprint further. We
also propose a memory usage predictor coupled with a search algorithm to
provide fusing and tiling configurations for an arbitrary set of convolutional
layers. When applied to the YOLOv2 object detection network, results show that
our approach can run in less than half the memory, and with a speedup of up to
2.78 under severe memory constraints. Additionally, our algorithm will return a
configuration with a latency that is within 6% of the best latency measured in
a manual search.

    

### [[2107.06981] Mapping Learning Algorithms on Data, a useful step for optimizing performances and their comparison](http://arxiv.org/abs/2107.06981)


  In the paper, we propose a novel methodology to map learning algorithms on
data (performance map) in order to gain more insights in the distribution of
their performances across their parameter space. This methodology provides
useful information when selecting a learner's best configuration for the data
at hand, and it also enhances the comparison of learners across learning
contexts. In order to explain the proposed methodology, the study introduces
the notions of learning context, performance map, and high performance
function. It then applies these concepts to a variety of learning contexts to
show how their use can provide more insights in a learner's behavior, and can
enhance the comparison of learners across learning contexts. The study is
completed by an extensive experimental study describing how the proposed
methodology can be applied.

    

### [[2107.06990] Annotation and Classification of Evidence and Reasoning Revisions in Argumentative Writing](http://arxiv.org/abs/2107.06990)


  Automated writing evaluation systems can improve students' writing insofar as
students attend to the feedback provided and revise their essay drafts in ways
aligned with such feedback. Existing research on revision of argumentative
writing in such systems, however, has focused on the types of revisions
students make (e.g., surface vs. content) rather than the extent to which
revisions actually respond to the feedback provided and improve the essay. We
introduce an annotation scheme to capture the nature of sentence-level
revisions of evidence use and reasoning (the `RER' scheme) and apply it to 5th-
and 6th-grade students' argumentative essays. We show that reliable manual
annotation can be achieved and that revision annotations correlate with a
holistic assessment of essay improvement in line with the feedback provided.
Furthermore, we explore the feasibility of automatically classifying revisions
according to our scheme.

    

### [[2107.06991] Physics-informed generative neural network: an application to troposphere temperature prediction](http://arxiv.org/abs/2107.06991)


  The troposphere is one of the atmospheric layers where most weather phenomena
occur. Temperature variations in the troposphere, especially at 500 hPa, a
typical level of the middle troposphere, are significant indicators of future
weather changes. Numerical weather prediction is effective for temperature
prediction, but its computational complexity hinders a timely response. This
paper proposes a novel temperature prediction approach in framework
ofphysics-informed deep learning. The new model, called PGnet, builds upon a
generative neural network with a mask matrix. The mask is designed to
distinguish the low-quality predicted regions generated by the first physical
stage. The generative neural network takes the mask as prior for the
second-stage refined predictions. A mask-loss and a jump pattern strategy are
developed to train the generative neural network without accumulating errors
during making time-series predictions. Experiments on ERA5 demonstrate that
PGnet can generate more refined temperature predictions than the
state-of-the-art.

    

### [[2107.06992] Finding Significant Features for Few-Shot Learning using Dimensionality Reduction](http://arxiv.org/abs/2107.06992)


  Few-shot learning is a relatively new technique that specializes in problems
where we have little amounts of data. The goal of these methods is to classify
categories that have not been seen before with just a handful of samples.
Recent approaches, such as metric learning, adopt the meta-learning strategy in
which we have episodic tasks conformed by support (training) data and query
(test) data. Metric learning methods have demonstrated that simple models can
achieve good performance by learning a similarity function to compare the
support and the query data. However, the feature space learned by a given
metric learning approach may not exploit the information given by a specific
few-shot task. In this work, we explore the use of dimension reduction
techniques as a way to find task-significant features helping to make better
predictions. We measure the performance of the reduced features by assigning a
score based on the intra-class and inter-class distance, and selecting a
feature reduction method in which instances of different classes are far away
and instances of the same class are close. This module helps to improve the
accuracy performance by allowing the similarity function, given by the metric
learning method, to have more discriminative features for the classification.
Our method outperforms the metric learning baselines in the miniImageNet
dataset by around 2% in accuracy performance.

    

### [[2107.06993] Confidence Conditioned Knowledge Distillation](http://arxiv.org/abs/2107.06993)


  In this paper, a novel confidence conditioned knowledge distillation (CCKD)
scheme for transferring the knowledge from a teacher model to a student model
is proposed. Existing state-of-the-art methods employ fixed loss functions for
this purpose and ignore the different levels of information that need to be
transferred for different samples. In addition to that, these methods are also
inefficient in terms of data usage. CCKD addresses these issues by leveraging
the confidence assigned by the teacher model to the correct class to devise
sample-specific loss functions (CCKD-L formulation) and targets (CCKD-T
formulation). Further, CCKD improves the data efficiency by employing
self-regulation to stop those samples from participating in the distillation
process on which the student model learns faster. Empirical evaluations on
several benchmark datasets show that CCKD methods achieve at least as much
generalization performance levels as other state-of-the-art methods while being
data efficient in the process. Student models trained through CCKD methods do
not retain most of the misclassifications commited by the teacher model on the
training set. Distillation through CCKD methods improves the resilience of the
student models against adversarial attacks compared to the conventional KD
method. Experiments show at least 3% increase in performance against
adversarial attacks for the MNIST and the Fashion MNIST datasets, and at least
6% increase for the CIFAR10 dataset.

    

### [[2107.06994] What underlies rapid learning and systematic generalization in humans](http://arxiv.org/abs/2107.06994)


  Despite the groundbreaking successes of neural networks, contemporary models
require extensive training with massive datasets and exhibit poor out-of-sample
generalization. One proposed solution is to build systematicity and
domain-specific constraints into the model, echoing the tenets of classical,
symbolic cognitive architectures. In this paper, we consider the limitations of
this approach by examining human adults' ability to learn an abstract reasoning
task from a brief instructional tutorial and explanatory feedback for incorrect
responses, demonstrating that human learning dynamics and ability to generalize
outside the range of the training examples differ drastically from those of a
representative neural network model, and that the model is brittle to changes
in features not anticipated by its authors. We present further evidence from
human data that the ability to consistently solve the puzzles was associated
with education, particularly basic mathematics education, and with the ability
to provide a reliably identifiable, valid description of the strategy used. We
propose that rapid learning and systematic generalization in humans may depend
on a gradual, experience-dependent process of learning-to-learn using
instructions and explanations to guide the construction of explicit abstract
rules that support generalizable inferences.

    

### [[2107.06995] Low-Rank Temporal Attention-Augmented Bilinear Network for financial time-series forecasting](http://arxiv.org/abs/2107.06995)


  Financial market analysis, especially the prediction of movements of stock
prices, is a challenging problem. The nature of financial time-series data,
being non-stationary and nonlinear, is the main cause of these challenges. Deep
learning models have led to significant performance improvements in many
problems coming from different domains, including prediction problems of
financial time-series data. Although the prediction performance is the main
goal of such models, dealing with ultra high-frequency data sets restrictions
in terms of the number of model parameters and its inference speed. The
Temporal Attention-Augmented Bilinear network was recently proposed as an
efficient and high-performing model for Limit Order Book time-series
forecasting. In this paper, we propose a low-rank tensor approximation of the
model to further reduce the number of trainable parameters and increase its
speed.

    

### [[2107.06996] Elastic Graph Neural Networks](http://arxiv.org/abs/2107.06996)


  While many existing graph neural networks (GNNs) have been proven to perform
$\ell_2$-based graph smoothing that enforces smoothness globally, in this work
we aim to further enhance the local smoothness adaptivity of GNNs via
$\ell_1$-based graph smoothing. As a result, we introduce a family of GNNs
(Elastic GNNs) based on $\ell_1$ and $\ell_2$-based graph smoothing. In
particular, we propose a novel and general message passing scheme into GNNs.
This message passing algorithm is not only friendly to back-propagation
training but also achieves the desired smoothing properties with a theoretical
convergence guarantee. Experiments on semi-supervised learning tasks
demonstrate that the proposed Elastic GNNs obtain better adaptivity on
benchmark datasets and are significantly robust to graph adversarial attacks.
The implementation of Elastic GNNs is available at
\url{this https URL}.

    

### [[2107.06997] DeepHyperion: Exploring the Feature Space of Deep Learning-Based Systems through Illumination Search](http://arxiv.org/abs/2107.06997)


  Deep Learning (DL) has been successfully applied to a wide range of
application domains, including safety-critical ones. Several DL testing
approaches have been recently proposed in the literature but none of them aims
to assess how different interpretable features of the generated inputs affect
the system's behaviour. In this paper, we resort to Illumination Search to find
the highest-performing test cases (i.e., misbehaving and closest to
misbehaving), spread across the cells of a map representing the feature space
of the system. We introduce a methodology that guides the users of our approach
in the tasks of identifying and quantifying the dimensions of the feature space
for a given domain. We developed DeepHyperion, a search-based tool for DL
systems that illuminates, i.e., explores at large, the feature space, by
providing developers with an interpretable feature map where automatically
generated inputs are placed along with information about the exposed
behaviours.

    

### [[2107.07002] The Benchmark Lottery](http://arxiv.org/abs/2107.07002)


  The world of empirical machine learning (ML) strongly relies on benchmarks in
order to determine the relative effectiveness of different algorithms and
methods. This paper proposes the notion of "a benchmark lottery" that describes
the overall fragility of the ML benchmarking process. The benchmark lottery
postulates that many factors, other than fundamental algorithmic superiority,
may lead to a method being perceived as superior. On multiple benchmark setups
that are prevalent in the ML community, we show that the relative performance
of algorithms may be altered significantly simply by choosing different
benchmark tasks, highlighting the fragility of the current paradigms and
potential fallacious interpretation derived from benchmarking ML methods. Given
that every benchmark makes a statement about what it perceives to be important,
we argue that this might lead to biased progress in the community. We discuss
the implications of the observed phenomena and provide recommendations on
mitigating them using multiple machine learning domains and communities as use
cases, including natural language processing, computer vision, information
retrieval, recommender systems, and reinforcement learning.

    

### [[2107.07005] WeightScale: Interpreting Weight Change in Neural Networks](http://arxiv.org/abs/2107.07005)


  Interpreting the learning dynamics of neural networks can provide useful
insights into how networks learn and the development of better training and
design approaches. We present an approach to interpret learning in neural
networks by measuring relative weight change on a per layer basis and
dynamically aggregating emerging trends through combination of dimensionality
reduction and clustering which allows us to scale to very deep networks. We use
this approach to investigate learning in the context of vision tasks across a
variety of state-of-the-art networks and provide insights into the learning
behavior of these networks, including how task complexity affects layer-wise
learning in deeper layers of networks.

    

### [[2107.07009] Free-Text Keystroke Dynamics for User Authentication](http://arxiv.org/abs/2107.07009)


  In this research, we consider the problem of verifying user identity based on
keystroke dynamics obtained from free-text. We employ a novel feature
engineering method that generates image-like transition matrices. For this
image-like feature, a convolution neural network (CNN) with cutout achieves the
best results. A hybrid model consisting of a CNN and a recurrent neural network
(RNN) is also shown to outperform previous research in this field.

    

### [[2107.07014] Hybrid Bayesian Neural Networks with Functional Probabilistic Layers](http://arxiv.org/abs/2107.07014)


  Bayesian neural networks provide a direct and natural way to extend standard
deep neural networks to support probabilistic deep learning through the use of
probabilistic layers that, traditionally, encode weight (and bias) uncertainty.
In particular, hybrid Bayesian neural networks utilize standard deterministic
layers together with few probabilistic layers judicially positioned in the
networks for uncertainty estimation. A major aspect and benefit of Bayesian
inference is that priors, in principle, provide the means to encode prior
knowledge for use in inference and prediction. However, it is difficult to
specify priors on weights since the weights have no intuitive interpretation.
Further, the relationships of priors on weights to the functions computed by
networks are difficult to characterize. In contrast, functions are intuitive to
interpret and are direct since they map inputs to outputs. Therefore, it is
natural to specify priors on functions to encode prior knowledge, and to use
them in inference and prediction based on functions. To support this, we
propose hybrid Bayesian neural networks with functional probabilistic layers
that encode function (and activation) uncertainty. We discuss their foundations
in functional Bayesian inference, functional variational inference, sparse
Gaussian processes, and sparse variational Gaussian processes. We further
perform few proof-of-concept experiments using GPflus, a new library that
provides Gaussian process layers and supports their use with deterministic
Keras layers to form hybrid neural network and Gaussian process models.

    

### [[2107.07029] Leveraging Hierarchical Structures for Few-Shot Musical Instrument Recognition](http://arxiv.org/abs/2107.07029)


  Deep learning work on musical instrument recognition has generally focused on
instrument classes for which we have abundant data. In this work, we exploit
hierarchical relationships between instruments in a few-shot learning setup to
enable classification of a wider set of musical instruments, given a few
examples at inference. We apply a hierarchical loss function to the training of
prototypical networks, combined with a method to aggregate prototypes
hierarchically, mirroring the structure of a predefined musical instrument
hierarchy. These extensions require no changes to the network architecture and
new levels can be easily added or removed. Compared to a non-hierarchical
few-shot baseline, our method leads to a significant increase in classification
accuracy and significant decrease mistake severity on instrument classes unseen
in training.

    

### [[2107.07038] Conditional Teaching Size](http://arxiv.org/abs/2107.07038)


  Recent research in machine teaching has explored the instruction of any
concept expressed in a universal language. In this compositional context, new
experimental results have shown that there exist data teaching sets
surprisingly shorter than the concept description itself. However, there exists
a bound for those remarkable experimental findings through teaching size and
concept complexity that we further explore here. As concepts are rarely taught
in isolation we investigate the best configuration of concepts to teach a given
set of concepts, where those that have been acquired first can be reused for
the description of new ones. This new notion of conditional teaching size
uncovers new insights, such as the interposition phenomenon: certain prior
knowledge generates simpler compatible concepts that increase the teaching size
of the concept that we want to teach. This does not happen for conditional
Kolmogorov complexity. Furthermore, we provide an algorithm that constructs
optimal curricula based on interposition avoidance. This paper presents a
series of theoretical results, including their proofs, and some directions for
future work. New research possibilities in curriculum teaching in compositional
scenarios are now wide open to exploration.

    

### [[2107.07039] Short-term Hourly Streamflow Prediction with Graph Convolutional GRU Networks](http://arxiv.org/abs/2107.07039)


  The frequency and impact of floods are expected to increase due to climate
change. It is crucial to predict streamflow, consequently flooding, in order to
prepare and mitigate its consequences in terms of property damage and
fatalities. This paper presents a Graph Convolutional GRUs based model to
predict the next 36 hours of streamflow for a sensor location using the
upstream river network. As shown in experiment results, the model presented in
this study provides better performance than the persistence baseline and a
Convolutional Bidirectional GRU network for the selected study area in
short-term streamflow prediction.

    

### [[2107.07040] Parsimony-Enhanced Sparse Bayesian Learning for Robust Discovery of Partial Differential Equations](http://arxiv.org/abs/2107.07040)


  Robust physics discovery is of great interest for many scientific and
engineering fields. Inspired by the principle that a representative model is
the one simplest possible, a new model selection criteria considering both
model's Parsimony and Sparsity is proposed. A Parsimony Enhanced Sparse
Bayesian Learning (PeSBL) method is developed for discovering the governing
Partial Differential Equations (PDEs) of nonlinear dynamical systems. Compared
with the conventional Sparse Bayesian Learning (SBL) method, the PeSBL method
promotes parsimony of the learned model in addition to its sparsity. In this
method, the parsimony of model terms is evaluated using their locations in the
prescribed candidate library, for the first time, considering the increased
complexity with the power of polynomials and the order of spatial derivatives.
Subsequently, the model parameters are updated through Bayesian inference with
the raw data. This procedure aims to reduce the error associated with the
possible loss of information in data preprocessing and numerical
differentiation prior to sparse regression. Results of numerical case studies
indicate that the governing PDEs of many canonical dynamical systems can be
correctly identified using the proposed PeSBL method from highly noisy data (up
to 50% in the current study). Next, the proposed methodology is extended for
stochastic PDE learning where all parameters and modeling error are considered
as random variables. Hierarchical Bayesian Inference (HBI) is integrated with
the proposed framework for stochastic PDE learning from a population of
observations. Finally, the proposed PeSBL is demonstrated for system response
prediction with uncertainties and anomaly diagnosis. Codes of all demonstrated
examples in this study are available on the website: this https URL.

    

### [[2107.07041] Mitigating Memorization in Sample Selection for Learning with Noisy Labels](http://arxiv.org/abs/2107.07041)


  Because deep learning is vulnerable to noisy labels, sample selection
techniques, which train networks with only clean labeled data, have attracted a
great attention. However, if the labels are dominantly corrupted by few
classes, these noisy samples are called dominant-noisy-labeled samples, the
network also learns dominant-noisy-labeled samples rapidly via content-aware
optimization. In this study, we propose a compelling criteria to penalize
dominant-noisy-labeled samples intensively through class-wise penalty labels.
By averaging prediction confidences for the each observed label, we obtain
suitable penalty labels that have high values if the labels are largely
corrupted by some classes. Experiments were performed using benchmarks
(CIFAR-10, CIFAR-100, Tiny-ImageNet) and real-world datasets (ANIMAL-10N,
Clothing1M) to evaluate the proposed criteria in various scenarios with
different noise rates. Using the proposed sample selection, the learning
process of the network becomes significantly robust to noisy labels compared to
existing methods in several noise types.

    

### [[2107.07042] Classifying Component Function in Product Assemblies with Graph Neural Networks](http://arxiv.org/abs/2107.07042)


  Function is defined as the ensemble of tasks that enable the product to
complete the designed purpose. Functional tools, such as functional modeling,
offer decision guidance in the early phase of product design, where explicit
design decisions are yet to be made. Function-based design data is often sparse
and grounded in individual interpretation. As such, function-based design tools
can benefit from automatic function classification to increase data fidelity
and provide function representation models that enable function-based
intelligent design agents. Function-based design data is commonly stored in
manually generated design repositories. These design repositories are a
collection of expert knowledge and interpretations of function in product
design bounded by function-flow and component taxonomies. In this work, we
represent a structured taxonomy-based design repository as assembly-flow
graphs, then leverage a graph neural network (GNN) model to perform automatic
function classification. We support automated function classification by
learning from repository data to establish the ground truth of component
function assignment. Experimental results show that our GNN model achieves a
micro-average F${_1}$-score of 0.832 for tier 1 (broad), 0.756 for tier 2, and
0.783 for tier 3 (specific) functions. Given the imbalance of data features,
the results are encouraging. Our efforts in this paper can be a starting point
for more sophisticated applications in knowledge-based CAD systems and
Design-for-X consideration in function-based design.

    

### [[2107.07043] GGT: Graph-Guided Testing for Adversarial Sample Detection of Deep Neural Network](http://arxiv.org/abs/2107.07043)


  Deep Neural Networks (DNN) are known to be vulnerable to adversarial samples,
the detection of which is crucial for the wide application of these DNN models.
Recently, a number of deep testing methods in software engineering were
proposed to find the vulnerability of DNN systems, and one of them, i.e., Model
Mutation Testing (MMT), was used to successfully detect various adversarial
samples generated by different kinds of adversarial attacks. However, the
mutated models in MMT are always huge in number (e.g., over 100 models) and
lack diversity (e.g., can be easily circumvented by high-confidence adversarial
samples), which makes it less efficient in real applications and less effective
in detecting high-confidence adversarial samples. In this study, we propose
Graph-Guided Testing (GGT) for adversarial sample detection to overcome these
aforementioned challenges. GGT generates pruned models with the guide of graph
characteristics, each of them has only about 5% parameters of the mutated model
in MMT, and graph guided models have higher diversity. The experiments on
CIFAR10 and SVHN validate that GGT performs much better than MMT with respect
to both effectiveness and efficiency.

    

### [[2107.07044] NVCell: Standard Cell Layout in Advanced Technology Nodes with Reinforcement Learning](http://arxiv.org/abs/2107.07044)


  High quality standard cell layout automation in advanced technology nodes is
still challenging in the industry today because of complex design rules. In
this paper we introduce an automatic standard cell layout generator called
NVCell that can generate layouts with equal or smaller area for over 90% of
single row cells in an industry standard cell library on an advanced technology
node. NVCell leverages reinforcement learning (RL) to fix design rule
violations during routing and to generate efficient placements.

    

### [[2107.07045] Explainable AI: current status and future directions](http://arxiv.org/abs/2107.07045)


  Explainable Artificial Intelligence (XAI) is an emerging area of research in
the field of Artificial Intelligence (AI). XAI can explain how AI obtained a
particular solution (e.g., classification or object detection) and can also
answer other "wh" questions. This explainability is not possible in traditional
AI. Explainability is essential for critical applications, such as defense,
health care, law and order, and autonomous driving vehicles, etc, where the
know-how is required for trust and transparency. A number of XAI techniques so
far have been purposed for such applications. This paper provides an overview
of these techniques from a multimedia (i.e., text, image, audio, and video)
point of view. The advantages and shortcomings of these techniques have been
discussed, and pointers to some future directions have also been provided.

    

### [[2107.07046] Backprop-Free Reinforcement Learning with Active Neural Generative Coding](http://arxiv.org/abs/2107.07046)


  In humans, perceptual awareness facilitates the fast recognition and
extraction of information from sensory input. This awareness largely depends on
how the human agent interacts with the environment. In this work, we propose
active neural generative coding, a computational framework for learning
action-driven generative models without backpropagation of errors (backprop) in
dynamic environments. Specifically, we develop an intelligent agent that
operates even with sparse rewards, drawing inspiration from the cognitive
theory of planning as inference. We demonstrate on several control problems, in
the online learning setting, that our proposed modeling framework performs
competitively with deep Q-learning models. The robust performance of our agent
offers promising evidence that a backprop-free approach for neural inference
and learning can drive goal-directed behavior.

    

### [[2107.07049] Learning-based Spectrum Sensing and Access in Cognitive Radios via Approximate POMDPs](http://arxiv.org/abs/2107.07049)


  A novel LEarning-based Spectrum Sensing and Access (LESSA) framework is
proposed, wherein a cognitive radio (CR) learns a time-frequency correlation
model underlying spectrum occupancy of licensed users (LUs) in a radio
ecosystem; concurrently, it devises an approximately optimal spectrum sensing
and access policy under sensing constraints. A Baum-Welch algorithm is proposed
to learn a parametric Markov transition model of LU spectrum occupancy based on
noisy spectrum measurements. Spectrum sensing and access are cast as a
Partially-Observable Markov Decision Process, approximately optimized via
randomized point-based value iteration. Fragmentation, Hamming-distance state
filters and Monte-Carlo methods are proposed to alleviate the inherent
computational complexity, and a weighted reward metric to regulate the
trade-off between CR throughput and LU interference. Numerical evaluations
demonstrate that LESSA performs within 5 percent of a genie-aided upper bound
with foreknowledge of LU spectrum occupancy, and outperforms state-of-the-art
algorithms across the entire trade-off region: 71 percent over
correlation-based clustering, 26 percent over Neyman-Pearson detection, 6
percent over the Viterbi algorithm, and 9 percent over an adaptive Deep
Q-Network. LESSA is then extended to a distributed Multi-Agent setting
(MA-LESSA), by proposing novel neighbor discovery and channel access rank
allocation. MA-LESSA improves CR throughput by 43 percent over cooperative
TD-SARSA, 84 percent over cooperative greedy distributed learning, and 3x over
non-cooperative learning via g-statistics and ACKs. Finally, MA-LESSA is
implemented on the DARPA SC2 platform, manifesting superior performance over
competitors in a real-world TDWR-UNII WLAN emulation; its implementation
feasibility is further validated on a testbed of ESP32 radios, exhibiting 96
percent success probability.

    

### [[2107.07054] Expert Graphs: Synthesizing New Expertise via Collaboration](http://arxiv.org/abs/2107.07054)


  Consider multiple experts with overlapping expertise working on a
classification problem under uncertain input. What constitutes a consistent set
of opinions? How can we predict the opinions of experts on missing sub-domains?
In this paper, we define a framework of to analyze this problem, termed "expert
graphs." In an expert graph, vertices represent classes and edges represent
binary opinions on the topics of their vertices. We derive necessary conditions
for expert graph validity and use them to create "synthetic experts" which
describe opinions consistent with the observed opinions of other experts. We
show this framework to be equivalent to the well-studied linear ordering
polytope. We show our conditions are not sufficient for describing all expert
graphs on cliques, but are sufficient for cycles.

    

### [[2107.07058] A Generalized Framework for Edge-preserving and Structure-preserving Image Smoothing](http://arxiv.org/abs/2107.07058)


  Image smoothing is a fundamental procedure in applications of both computer
vision and graphics. The required smoothing properties can be different or even
contradictive among different tasks. Nevertheless, the inherent smoothing
nature of one smoothing operator is usually fixed and thus cannot meet the
various requirements of different applications. In this paper, we first
introduce the truncated Huber penalty function which shows strong flexibility
under different parameter settings. A generalized framework is then proposed
with the introduced truncated Huber penalty function. When combined with its
strong flexibility, our framework is able to achieve diverse smoothing natures
where contradictive smoothing behaviors can even be achieved. It can also yield
the smoothing behavior that can seldom be achieved by previous methods, and
superior performance is thus achieved in challenging cases. These together
enable our framework capable of a range of applications and able to outperform
the state-of-the-art approaches in several tasks, such as image detail
enhancement, clip-art compression artifacts removal, guided depth map
restoration, image texture removal, etc. In addition, an efficient numerical
solution is provided and its convergence is theoretically guaranteed even the
optimization framework is non-convex and non-smooth. A simple yet effective
approach is further proposed to reduce the computational cost of our method
while maintaining its performance. The effectiveness and superior performance
of our approach are validated through comprehensive experiments in a range of
applications. Our code is available at
this https URL.

    

### [[2107.07064] DAL: Feature Learning from Overt Speech to Decode Imagined Speech-based EEG Signals with Convolutional Autoencoder](http://arxiv.org/abs/2107.07064)


  Brain-computer interface (BCI) is one of the tools which enables the
communication between humans and devices by reflecting intention and status of
humans. With the development of artificial intelligence, the interest in
communication between humans and drones using electroencephalogram (EEG) is
increased. Especially, in the case of controlling drone swarms such as
direction or formation, there are many advantages compared with controlling a
drone unit. Imagined speech is one of the endogenous BCI paradigms, which can
identify intentions of users. When conducting imagined speech, the users
imagine the pronunciation as if actually speaking. In contrast, overt speech is
a task in which the users directly pronounce the words. When controlling drone
swarms using imagined speech, complex commands can be delivered more
intuitively, but decoding performance is lower than that of other endogenous
BCI paradigms. We proposed the Deep-autoleaner (DAL) to learn EEG features of
overt speech for imagined speech-based EEG signals classification. To the best
of our knowledge, this study is the first attempt to use EEG features of overt
speech to decode imagined speech-based EEG signals with an autoencoder. A total
of eight subjects participated in the experiment. When classifying four words,
the average accuracy of the DAL was 48.41%. In addition, when comparing the
performance between w/o and w/ EEG features of overt speech, there was a
performance improvement of 7.42% when including EEG features of overt speech.
Hence, we demonstrated that EEG features of overt speech could improve the
decoding performance of imagined speech.

    

### [[2107.07075] Deep Learning on a Data Diet: Finding Important Examples Early in Training](http://arxiv.org/abs/2107.07075)


  The recent success of deep learning has partially been driven by training
increasingly overparametrized networks on ever larger datasets. It is therefore
natural to ask: how much of the data is superfluous, which examples are
important for generalization, and how do we find them? In this work, we make
the striking observation that, on standard vision benchmarks, the initial loss
gradient norm of individual training examples, averaged over several weight
initializations, can be used to identify a smaller set of training data that is
important for generalization. Furthermore, after only a few epochs of training,
the information in gradient norms is reflected in the normed error--L2 distance
between the predicted probabilities and one hot labels--which can be used to
prune a significant fraction of the dataset without sacrificing test accuracy.
Based on this, we propose data pruning methods which use only local information
early in training, and connect them to recent work that prunes data by
discarding examples that are rarely forgotten over the course of training. Our
methods also shed light on how the underlying data distribution shapes the
training dynamics: they rank examples based on their importance for
generalization, detect noisy examples and identify subspaces of the model's
data representation that are relatively stable over training.

    

### [[2107.07076] An Overview and Experimental Study of Learning-based Optimization Algorithms for Vehicle Routing Problem](http://arxiv.org/abs/2107.07076)


  Vehicle routing problem (VRP) is a typical discrete combinatorial
optimization problem, and many models and algorithms have been proposed to
solve VRP and variants. Although existing approaches has contributed a lot to
the development of this field, these approaches either are limited in problem
size or need manual intervening in choosing parameters. To tackle these
difficulties, many studies consider learning-based optimization algorithms to
solve VRP. This paper reviews recent advances in this field and divides
relevant approaches into end-to-end approaches and step-by-step approaches. We
design three part experiments to justly evaluate performance of four
representative learning-based optimization algorithms and conclude that
combining heuristic search can effectively improve learning ability and sampled
efficiency of LBO models. Finally we point out that research trend of LBO
algorithms is to solve large-scale and multiple constraints problems from real
world.

    

### [[2107.07087] Entropic Inequality Constraints from $e$-separation Relations in Directed Acyclic Graphs with Hidden Variables](http://arxiv.org/abs/2107.07087)


  Directed acyclic graphs (DAGs) with hidden variables are often used to
characterize causal relations between variables in a system. When some
variables are unobserved, DAGs imply a notoriously complicated set of
constraints on the distribution of observed variables. In this work, we present
entropic inequality constraints that are implied by $e$-separation relations in
hidden variable DAGs with discrete observed variables. The constraints can
intuitively be understood to follow from the fact that the capacity of
variables along a causal pathway to convey information is restricted by their
entropy; e.g. at the extreme case, a variable with entropy $0$ can convey no
information. We show how these constraints can be used to learn about the true
causal model from an observed data distribution. In addition, we propose a
measure of causal influence called the minimal mediary entropy, and demonstrate
that it can augment traditional measures such as the average causal effect.

    

### [[2107.07098] Hida-Matérn Kernel](http://arxiv.org/abs/2107.07098)


  We present the class of Hida-Matérn kernels, which is the canonical family
of covariance functions over the entire space of stationary Gauss-Markov
Processes. It extends upon Matérn kernels, by allowing for flexible
construction of priors over processes with oscillatory components. Any
stationary kernel, including the widely used squared-exponential and spectral
mixture kernels, are either directly within this class or are appropriate
asymptotic limits, demonstrating the generality of this class. Taking advantage
of its Markovian nature we show how to represent such processes as state space
models using only the kernel and its derivatives. In turn this allows us to
perform Gaussian Process inference more efficiently and side step the usual
computational burdens. We also show how exploiting special properties of the
state space representation enables improved numerical stability in addition to
further reductions of computational complexity.

    

### [[2107.07105] Continuous-variable neural-network quantum states and the quantum rotor model](http://arxiv.org/abs/2107.07105)


  We initiate the study of neural-network quantum state algorithms for
analyzing continuous-variable lattice quantum systems in first quantization. A
simple family of continuous-variable trial wavefunctons is introduced which
naturally generalizes the restricted Boltzmann machine (RBM) wavefunction
introduced for analyzing quantum spin systems. By virtue of its simplicity, the
same variational Monte Carlo training algorithms that have been developed for
ground state determination and time evolution of spin systems have natural
analogues in the continuum. We offer a proof of principle demonstration in the
context of ground state determination of a stoquastic quantum rotor
Hamiltonian. Results are compared against those obtained from partial
differential equation (PDE) based scalable eigensolvers. This study serves as a
benchmark against which future investigation of continuous-variable neural
quantum states can be compared, and points to the need to consider deep network
architectures and more sophisticated training algorithms.

    

### [[2107.07106] Online Learning for Recommendations at Grubhub](http://arxiv.org/abs/2107.07106)


  We propose a method to easily modify existing offline Recommender Systems to
run online using Transfer Learning. Online Learning for Recommender Systems has
two main advantages: quality and scale. Like many Machine Learning algorithms
in production if not regularly retrained will suffer from Concept Drift. A
policy that is updated frequently online can adapt to drift faster than a batch
system. This is especially true for user-interaction systems like recommenders
where the underlying distribution can shift drastically to follow user
behaviour. As a platform grows rapidly like Grubhub, the cost of running batch
training jobs becomes material. A shift from stateless batch learning offline
to stateful incremental learning online can recover, for example, at Grubhub,
up to a 45x cost savings and a +20% metrics increase. There are a few
challenges to overcome with the transition to online stateful learning, namely
convergence, non-stationary embeddings and off-policy evaluation, which we
explore from our experiences running this system in production.

    

### [[2107.07110] Recurrent Parameter Generators](http://arxiv.org/abs/2107.07110)


  We present a generic method for recurrently using the same parameters for
many different convolution layers to build a deep network. Specifically, for a
network, we create a recurrent parameter generator (RPG), from which the
parameters of each convolution layer are generated. Though using recurrent
models to build a deep convolutional neural network (CNN) is not entirely new,
our method achieves significant performance gain compared to the existing
works. We demonstrate how to build a one-layer neural network to achieve
similar performance compared to other traditional CNN models on various
applications and datasets. Such a method allows us to build an arbitrarily
complex neural network with any amount of parameters. For example, we build a
ResNet34 with model parameters reduced by more than $400$ times, which still
achieves $41.6\%$ ImageNet top-1 accuracy. Furthermore, we demonstrate the RPG
can be applied at different scales, such as layers, blocks, or even
sub-networks. Specifically, we use the RPG to build a ResNet18 network with the
number of weights equivalent to one convolutional layer of a conventional
ResNet and show this model can achieve $67.2\%$ ImageNet top-1 accuracy. The
proposed method can be viewed as an inverse approach to model compression.
Rather than removing the unused parameters from a large model, it aims to
squeeze more information into a small number of parameters. Extensive
experiment results are provided to demonstrate the power of the proposed
recurrent parameter generator.

    

### [[2107.07115] Principal component analysis for Gaussian process posteriors](http://arxiv.org/abs/2107.07115)


  This paper proposes an extension of principal component analysis for Gaussian
process posteriors denoted by GP-PCA. Since GP-PCA estimates a low-dimensional
space of GP posteriors, it can be used for meta-learning, which is a framework
for improving the precision of a new task by estimating a structure of a set of
tasks. The issue is how to define a structure of a set of GPs with an
infinite-dimensional parameter, such as coordinate system and a divergence. In
this study, we reduce the infiniteness of GP to the finite-dimensional case
under the information geometrical framework by considering a space of GP
posteriors that has the same prior. In addition, we propose an approximation
method of GP-PCA based on variational inference and demonstrate the
effectiveness of GP-PCA as meta-learning through experiments.

    

### [[2107.07116] Transformer-based Machine Learning for Fast SAT Solvers and Logic Synthesis](http://arxiv.org/abs/2107.07116)


  CNF-based SAT and MaxSAT solvers are central to logic synthesis and
verification systems. The increasing popularity of these constraint problems in
electronic design automation encourages studies on different SAT problems and
their properties for further computational efficiency. There has been both
theoretical and practical success of modern Conflict-driven clause learning SAT
solvers, which allows solving very large industrial instances in a relatively
short amount of time. Recently, machine learning approaches provide a new
dimension to solving this challenging problem. Neural symbolic models could
serve as generic solvers that can be specialized for specific domains based on
data without any changes to the structure of the model. In this work, we
propose a one-shot model derived from the Transformer architecture to solve the
MaxSAT problem, which is the optimization version of SAT where the goal is to
satisfy the maximum number of clauses. Our model has a scale-free structure
which could process varying size of instances. We use meta-path and
self-attention mechanism to capture interactions among homogeneous nodes. We
adopt cross-attention mechanisms on the bipartite graph to capture interactions
among heterogeneous nodes. We further apply an iterative algorithm to our model
to satisfy additional clauses, enabling a solution approaching that of an
exact-SAT problem. The attention mechanisms leverage the parallelism for
speedup. Our evaluation indicates improved speedup compared to heuristic
approaches and improved completion rate compared to machine learning
approaches.

    

### [[2107.07127] NeuSaver: Neural Adaptive Power Consumption Optimization for Mobile Video Streaming](http://arxiv.org/abs/2107.07127)


  Video streaming services strive to support high-quality videos at higher
resolutions and frame rates to improve the quality of experience (QoE).
However, high-quality videos consume considerable amounts of energy on mobile
devices. This paper proposes NeuSaver, which reduces the power consumption of
mobile devices when streaming videos by applying an adaptive frame rate to each
video chunk without compromising user experience. NeuSaver generates an optimal
policy that determines the appropriate frame rate for each video chunk using
reinforcement learning (RL). The RL model automatically learns the policy that
maximizes the QoE goals based on previous observations. NeuSaver also uses an
asynchronous advantage actor-critic algorithm to reinforce the RL model quickly
and robustly. Streaming servers that support NeuSaver preprocesses videos into
segments with various frame rates, which is similar to the process of creating
videos with multiple bit rates in dynamic adaptive streaming over HTTP.
NeuSaver utilizes the commonly used H.264 video codec. We evaluated NeuSaver in
various experiments and a user study through four video categories along with
the state-of-the-art model. Our experiments showed that NeuSaver effectively
reduces the power consumption of mobile devices when streaming video by an
average of 16.14% and up to 23.12% while achieving high QoE.

    

### [[2107.07148] What Image Features Boost Housing Market Predictions?](http://arxiv.org/abs/2107.07148)


  The attractiveness of a property is one of the most interesting, yet
challenging, categories to model. Image characteristics are used to describe
certain attributes, and to examine the influence of visual factors on the price
or timeframe of the listing. In this paper, we propose a set of techniques for
the extraction of visual features for efficient numerical inclusion in
modern-day predictive algorithms. We discuss techniques such as Shannon's
entropy, calculating the center of gravity, employing image segmentation, and
using Convolutional Neural Networks. After comparing these techniques as
applied to a set of property-related images (indoor, outdoor, and satellite),
we conclude the following: (i) the entropy is the most efficient single-digit
visual measure for housing price prediction; (ii) image segmentation is the
most important visual feature for the prediction of housing lifespan; and (iii)
deep image features can be used to quantify interior characteristics and
contribute to captivation modeling. The set of 40 image features selected here
carries a significant amount of predictive power and outperforms some of the
strongest metadata predictors. Without any need to replace a human expert in a
real-estate appraisal process, we conclude that the techniques presented in
this paper can efficiently describe visible characteristics, thus introducing
perceived attractiveness as a quantitative measure into the predictive modeling
of housing.

    

### [[2107.07160] Lockout: Sparse Regularization of Neural Networks](http://arxiv.org/abs/2107.07160)


  Many regression and classification procedures fit a parameterized function
$f(x;w)$ of predictor variables $x$ to data $\{x_{i},y_{i}\}_1^N$ based on some
loss criterion $L(y,f)$. Often, regularization is applied to improve accuracy
by placing a constraint $P(w)\leq t$ on the values of the parameters $w$.
Although efficient methods exist for finding solutions to these constrained
optimization problems for all values of $t\geq0$ in the special case when $f$
is a linear function, none are available when $f$ is non-linear (e.g. Neural
Networks). Here we present a fast algorithm that provides all such solutions
for any differentiable function $f$ and loss $L$, and any constraint $P$ that
is an increasing monotone function of the absolute value of each parameter.
Applications involving sparsity inducing regularization of arbitrary Neural
Networks are discussed. Empirical results indicate that these sparse solutions
are usually superior to their dense counterparts in both accuracy and
interpretability. This improvement in accuracy can often make Neural Networks
competitive with, and sometimes superior to, state-of-the-art methods in the
analysis of tabular data.

    

### [[2107.07170] FLEX: Unifying Evaluation for Few-Shot NLP](http://arxiv.org/abs/2107.07170)


  Few-shot NLP research is highly active, yet conducted in disjoint research
threads with evaluation suites that lack challenging-yet-realistic testing
setups and fail to employ careful experimental design. Consequently, the
community does not know which techniques perform best or even if they
outperform simple baselines. We formulate desiderata for an ideal few-shot NLP
benchmark and present FLEX, the first benchmark, public leaderboard, and
framework that provides unified, comprehensive measurement for few-shot NLP
techniques. FLEX incorporates and introduces new best practices for few-shot
evaluation, including measurement of four transfer settings, textual labels for
zero-shot evaluation, and a principled approach to benchmark design that
optimizes statistical accuracy while keeping evaluation costs accessible to
researchers without large compute resources. In addition, we present UniFew, a
simple yet strong prompt-based model for few-shot learning which unifies the
pretraining and finetuning prompt formats, eschewing complex machinery of
recent prompt-based approaches in adapting downstream task formats to language
model pretraining objectives. We demonstrate that despite simplicity UniFew
achieves results competitive with both popular meta-learning and prompt-based
approaches.

    

### [[2107.07171] DeFed: A Principled Decentralized and Privacy-Preserving Federated Learning Algorithm](http://arxiv.org/abs/2107.07171)


  Federated learning enables a large number of clients to participate in
learning a shared model while maintaining the training data stored in each
client, which protects data privacy and security. Till now, federated learning
frameworks are built in a centralized way, in which a central client is needed
for collecting and distributing information from every other client. This not
only leads to high communication pressure at the central client, but also
renders the central client highly vulnerable to failure and attack. Here we
propose a principled decentralized federated learning algorithm (DeFed), which
removes the central client in the classical Federated Averaging (FedAvg)
setting and only relies information transmission between clients and their
local neighbors. The proposed DeFed algorithm is proven to reach the global
minimum with a convergence rate of $O(1/T)$ when the loss function is smooth
and strongly convex, where $T$ is the number of iterations in gradient descent.
Finally, the proposed algorithm has been applied to a number of toy examples to
demonstrate its effectiveness.

    

### [[2107.07184] MURAL: Meta-Learning Uncertainty-Aware Rewards for Outcome-Driven Reinforcement Learning](http://arxiv.org/abs/2107.07184)


  Exploration in reinforcement learning is a challenging problem: in the worst
case, the agent must search for reward states that could be hidden anywhere in
the state space. Can we define a more tractable class of RL problems, where the
agent is provided with examples of successful outcomes? In this problem
setting, the reward function can be obtained automatically by training a
classifier to categorize states as successful or not. If trained properly, such
a classifier can not only afford a reward function, but actually provide a
well-shaped objective landscape that both promotes progress toward good states
and provides a calibrated exploration bonus. In this work, we we show that an
uncertainty aware classifier can solve challenging reinforcement learning
problems by both encouraging exploration and provided directed guidance towards
positive outcomes. We propose a novel mechanism for obtaining these calibrated,
uncertainty-aware classifiers based on an amortized technique for computing the
normalized maximum likelihood (NML) distribution, also showing how these
techniques can be made computationally tractable by leveraging tools from
meta-learning. We show that the resulting algorithm has a number of intriguing
connections to both count-based exploration methods and prior algorithms for
learning reward functions, while also providing more effective guidance towards
the goal. We demonstrate that our algorithm solves a number of challenging
navigation and robotic manipulation tasks which prove difficult or impossible
for prior methods.

    

### [[2107.07197] Randomized ReLU Activation for Uncertainty Estimation of Deep Neural Networks](http://arxiv.org/abs/2107.07197)


  Deep neural networks (DNNs) have successfully learned useful data
representations in various tasks, however, assessing the reliability of these
representations remains a challenge. Deep Ensemble is widely considered the
state-of-the-art method for uncertainty estimation, but it is very expensive to
train and test. MC-Dropout is another alternative method, which is less
expensive but lacks the diversity of predictions. To get more diverse
predictions in less time, we introduce Randomized ReLU Activation (RRA)
framework. Under the framework, we propose two strategies, MC-DropReLU and
MC-RReLU, to estimate uncertainty. Instead of randomly dropping some neurons of
the network as in MC-Dropout, the RRA framework adds randomness to the
activation function module, making the outputs diverse. As far as we know, this
is the first attempt to add randomness to the activation function module to
generate predictive uncertainty. We analyze and compare the output diversity of
MC-Dropout and our method from the variance perspective and obtain the
relationship between the hyperparameters and output diversity in the two
methods. Moreover, our method is simple to implement and does not need to
modify the existing model. We experimentally validate the RRA framework on
three widely used datasets, CIFAR10, CIFAR100, and TinyImageNet. The
experiments demonstrate that our method has competitive performance but is more
favorable in training time and memory requirements.

    

### [[2107.07211] Decentralized Bayesian Learning with Metropolis-Adjusted Hamiltonian Monte Carlo](http://arxiv.org/abs/2107.07211)


  Federated learning performed by a decentralized networks of agents is
becoming increasingly important with the prevalence of embedded software on
autonomous devices. Bayesian approaches to learning benefit from offering more
information as to the uncertainty of a random quantity, and Langevin and
Hamiltonian methods are effective at realizing sampling from an uncertain
distribution with large parameter dimensions. Such methods have only recently
appeared in the decentralized setting, and either exclusively use stochastic
gradient Langevin and Hamiltonian Monte Carlo approaches that require a
diminishing stepsize to asymptotically sample from the posterior and are known
in practice to characterize uncertainty less faithfully than constant step-size
methods with a Metropolis adjustment, or assume strong convexity properties of
the potential function. We present the first approach to incorporating constant
stepsize Metropolis-adjusted HMC in the decentralized sampling framework, show
theoretical guarantees for consensus and probability distance to the posterior
stationary distribution, and demonstrate their effectiveness numerically on
standard real world problems, including decentralized learning of neural
networks which is known to be highly non-convex.

    

### [[2107.07232] On the expressivity of bi-Lipschitz normalizing flows](http://arxiv.org/abs/2107.07232)


  An invertible function is bi-Lipschitz if both the function and its inverse
have bounded Lipschitz constants. Nowadays, most Normalizing Flows are
bi-Lipschitz by design or by training to limit numerical errors (among other
things). In this paper, we discuss the expressivity of bi-Lipschitz Normalizing
Flows and identify several target distributions that are difficult to
approximate using such models. Then, we characterize the expressivity of
bi-Lipschitz Normalizing Flows by giving several lower bounds on the Total
Variation distance between these particularly unfavorable distributions and
their best possible approximation. Finally, we discuss potential remedies which
include using more complex latent distributions.

    

### [[2107.07240] Subnet Replacement: Deployment-stage backdoor attack against deep neural networks in gray-box setting](http://arxiv.org/abs/2107.07240)


  We study the realistic potential of conducting backdoor attack against deep
neural networks (DNNs) during deployment stage. Specifically, our goal is to
design a deployment-stage backdoor attack algorithm that is both threatening
and realistically implementable. To this end, we propose Subnet Replacement
Attack (SRA), which is capable of embedding backdoor into DNNs by directly
modifying a limited number of model parameters. Considering the realistic
practicability, we abandon the strong white-box assumption widely adopted in
existing studies, instead, our algorithm works in a gray-box setting, where
architecture information of the victim model is available but the adversaries
do not have any knowledge of parameter values. The key philosophy underlying
our approach is -- given any neural network instance (regardless of its
specific parameter values) of a certain architecture, we can always embed a
backdoor into that model instance, by replacing a very narrow subnet of a
benign model (without backdoor) with a malicious backdoor subnet, which is
designed to be sensitive (fire large activation value) to a particular backdoor
trigger pattern.

    

### [[2107.07260] MCL-GAN: Generative Adversarial Networks with Multiple Specialized Discriminators](http://arxiv.org/abs/2107.07260)


  We propose a generative adversarial network with multiple discriminators,
where each discriminator is specialized to distinguish the subset of a real
dataset. This approach facilitates learning a generator coinciding with the
underlying data distribution and thus mitigates the chronic mode collapse
problem. From the inspiration of multiple choice learning, we guide each
discriminator to have expertise in the subset of the entire data and allow the
generator to find reasonable correspondences between the latent and real data
spaces automatically without supervision for training examples and the number
of discriminators. Despite the use of multiple discriminators, the backbone
networks are shared across the discriminators and the increase of training cost
is minimized. We demonstrate the effectiveness of our algorithm in the standard
datasets using multiple evaluation metrics.

    

### [[2107.07261] Turning Tables: Generating Examples from Semi-structured Tables for Endowing Language Models with Reasoning Skills](http://arxiv.org/abs/2107.07261)


  Models pre-trained with a language modeling objective possess ample world
knowledge and language skills, but are known to struggle in tasks that require
reasoning. In this work, we propose to leverage semi-structured tables, and
automatically generate at scale question-paragraph pairs, where answering the
question requires reasoning over multiple facts in the paragraph. We add a
pre-training step over this synthetic data, which includes examples that
require 16 different reasoning skills such as number comparison, conjunction,
and fact composition. To improve data efficiency, we propose sampling
strategies that focus training on reasoning skills the model is currently
lacking. We evaluate our approach on three reading comprehension datasets that
are focused on reasoning, and show that our model, PReasM, substantially
outperforms T5, a popular pre-trained encoder-decoder model. Moreover, sampling
examples based on current model errors leads to faster training and higher
overall performance.

    

### [[2107.07271] Multi-Channel Auto-Encoders and a Novel Dataset for Learning Domain Invariant Representations of Histopathology Images](http://arxiv.org/abs/2107.07271)


  Domain shift is a problem commonly encountered when developing automated
histopathology pipelines. The performance of machine learning models such as
convolutional neural networks within automated histopathology pipelines is
often diminished when applying them to novel data domains due to factors
arising from differing staining and scanning protocols. The Dual-Channel
Auto-Encoder (DCAE) model was previously shown to produce feature
representations that are less sensitive to appearance variation introduced by
different digital slide scanners. In this work, the Multi-Channel Auto-Encoder
(MCAE) model is presented as an extension to DCAE which learns from more than
two domains of data. Additionally, a synthetic dataset is generated using
CycleGANs that contains aligned tissue images that have had their appearance
synthetically modified. Experimental results show that the MCAE model produces
feature representations that are less sensitive to inter-domain variations than
the comparative StaNoSA method when tested on the novel synthetic data.
Additionally, the MCAE and StaNoSA models are tested on a novel tissue
classification task. The results of this experiment show the MCAE model out
performs the StaNoSA model by 5 percentage-points in the f1-score. These
results show that the MCAE model is able to generalise better to novel data and
tasks than existing approaches by actively learning normalised feature
representations.

    

### [[2107.07274] A Robust Deep Learning Workflow to Predict Multiphase Flow Behavior during Geological CO2 Sequestration Injection and Post-Injection Periods](http://arxiv.org/abs/2107.07274)


  This paper contributes to the development and evaluation of a deep learning
workflow that accurately and efficiently predicts the temporal-spatial
evolution of pressure and CO2 plumes during injection and post-injection
periods of geologic CO2 sequestration (GCS) operations. Based on a Fourier
Neuron Operator, the deep learning workflow takes input variables or features
including rock properties, well operational controls and time steps, and
predicts the state variables of pressure and CO2 saturation. To further improve
the predictive fidelity, separate deep learning models are trained for CO2
injection and post-injection periods due the difference in primary driving
force of fluid flow and transport during these two phases. We also explore
different combinations of features to predict the state variables. We use a
realistic example of CO2 injection and storage in a 3D heterogeneous saline
aquifer, and apply the deep learning workflow that is trained from
physics-based simulation data and emulate the physics process. Through this
numerical experiment, we demonstrate that using two separate deep learning
models to distinguish post-injection from injection period generates the most
accurate prediction of pressure, and a single deep learning model of the whole
GCS process including the cumulative injection volume of CO2 as a deep learning
feature, leads to the most accurate prediction of CO2 saturation. For the
post-injection period, it is key to use cumulative CO2 injection volume to
inform the deep learning models about the total carbon storage when predicting
either pressure or saturation. The deep learning workflow not only provides
high predictive fidelity across temporal and spatial scales, but also offers a
speedup of 250 times compared to full physics reservoir simulation, and thus
will be a significant predictive tool for engineers to manage the long term
process of GCS.

    

### [[2107.07281] Input Dependent Sparse Gaussian Processes](http://arxiv.org/abs/2107.07281)


  Gaussian Processes (GPs) are Bayesian models that provide uncertainty
estimates associated to the predictions made. They are also very flexible due
to their non-parametric nature. Nevertheless, GPs suffer from poor scalability
as the number of training instances N increases. More precisely, they have a
cubic cost with respect to $N$. To overcome this problem, sparse GP
approximations are often used, where a set of $M \ll N$ inducing points is
introduced during training. The location of the inducing points is learned by
considering them as parameters of an approximate posterior distribution $q$.
Sparse GPs, combined with variational inference for inferring $q$, reduce the
training cost of GPs to $\mathcal{O}(M^3)$. Critically, the inducing points
determine the flexibility of the model and they are often located in regions of
the input space where the latent function changes. A limitation is, however,
that for some learning tasks a large number of inducing points may be required
to obtain a good prediction performance. To address this limitation, we propose
here to amortize the computation of the inducing points locations, as well as
the parameters of the variational posterior approximation q. For this, we use a
neural network that receives the observed data as an input and outputs the
inducing points locations and the parameters of $q$. We evaluate our method in
several experiments, showing that it performs similar or better than other
state-of-the-art sparse variational GP approaches. However, with our method the
number of inducing points is reduced drastically due to their dependency on the
input data. This makes our method scale to larger datasets and have faster
training and prediction times.

    

### [[2107.07305] Training for temporal sparsity in deep neural networks, application in video processing](http://arxiv.org/abs/2107.07305)


  Activation sparsity improves compute efficiency and resource utilization in
sparsity-aware neural network accelerators. As the predominant operation in
DNNs is multiply-accumulate (MAC) of activations with weights to compute inner
products, skipping operations where (at least) one of the two operands is zero
can make inference more efficient in terms of latency and power. Spatial
sparsification of activations is a popular topic in DNN literature and several
methods have already been established to bias a DNN for it. On the other hand,
temporal sparsity is an inherent feature of bio-inspired spiking neural
networks (SNNs), which neuromorphic processing exploits for hardware
efficiency. Introducing and exploiting spatio-temporal sparsity, is a topic
much less explored in DNN literature, but in perfect resonance with the trend
in DNN, to shift from static signal processing to more streaming signal
processing. Towards this goal, in this paper we introduce a new DNN layer
(called Delta Activation Layer), whose sole purpose is to promote temporal
sparsity of activations during training. A Delta Activation Layer casts
temporal sparsity into spatial activation sparsity to be exploited when
performing sparse tensor multiplications in hardware. By employing delta
inference and ``the usual'' spatial sparsification heuristics during training,
the resulting model learns to exploit not only spatial but also temporal
activation sparsity (for a given input data distribution). One may use the
Delta Activation Layer either during vanilla training or during a refinement
phase. We have implemented Delta Activation Layer as an extension of the
standard Tensoflow-Keras library, and applied it to train deep neural networks
on the Human Action Recognition (UCF101) dataset. We report an almost 3x
improvement of activation sparsity, with recoverable loss of model accuracy
after longer training.

    

### [[2107.07312] FMNet: Latent Feature-wise Mapping Network for Cleaning up Noisy Micro-Doppler Spectrogram](http://arxiv.org/abs/2107.07312)


  Micro-Doppler signatures contain considerable information about target
dynamics. However, the radar sensing systems are easily affected by noisy
surroundings, resulting in uninterpretable motion patterns on the micro-Doppler
spectrogram. Meanwhile, radar returns often suffer from multipath, clutter and
interference. These issues lead to difficulty in, for example motion feature
extraction, activity classification using micro Doppler signatures ($\mu$-DS),
etc. In this paper, we propose a latent feature-wise mapping strategy, called
Feature Mapping Network (FMNet), to transform measured spectrograms so that
they more closely resemble the output from a simulation under the same
conditions. Based on measured spectrogram and the matched simulated data, our
framework contains three parts: an Encoder which is used to extract latent
representations/features, a Decoder outputs reconstructed spectrogram according
to the latent features, and a Discriminator minimizes the distance of latent
features of measured and simulated data. We demonstrate the FMNet with six
activities data and two experimental scenarios, and final results show strong
enhanced patterns and can keep actual motion information to the greatest
extent. On the other hand, we also propose a novel idea which trains a
classifier with only simulated data and predicts new measured samples after
cleaning them up with the FMNet. From final classification results, we can see
significant improvements.

    

### [[2107.07314] Variational Topic Inference for Chest X-Ray Report Generation](http://arxiv.org/abs/2107.07314)


  Automating report generation for medical imaging promises to reduce workload
and assist diagnosis in clinical practice. Recent work has shown that deep
learning models can successfully caption natural images. However, learning from
medical data is challenging due to the diversity and uncertainty inherent in
the reports written by different radiologists with discrepant expertise and
experience. To tackle these challenges, we propose variational topic inference
for automatic report generation. Specifically, we introduce a set of topics as
latent variables to guide sentence generation by aligning image and language
modalities in a latent space. The topics are inferred in a conditional
variational inference framework, with each topic governing the generation of a
sentence in the report. Further, we adopt a visual attention module that
enables the model to attend to different locations in the image and generate
more informative descriptions. We conduct extensive experiments on two
benchmarks, namely Indiana U. Chest X-rays and MIMIC-CXR. The results
demonstrate that our proposed variational topic inference method can generate
novel reports rather than mere copies of reports used in training, while still
achieving comparable performance to state-of-the-art methods in terms of
standard language generation criteria.

    

### [[2107.07322] A unified framework for bandit multiple testing](http://arxiv.org/abs/2107.07322)


  In bandit multiple hypothesis testing, each arm corresponds to a different
null hypothesis that we wish to test, and the goal is to design adaptive
algorithms that correctly identify large set of interesting arms (true
discoveries), while only mistakenly identifying a few uninteresting ones (false
discoveries). One common metric in non-bandit multiple testing is the false
discovery rate (FDR). We propose a unified, modular framework for bandit FDR
control that emphasizes the decoupling of exploration and summarization of
evidence. We utilize the powerful martingale-based concept of ``e-processes''
to ensure FDR control for arbitrary composite nulls, exploration rules and
stopping times in generic problem settings. In particular, valid FDR control
holds even if the reward distributions of the arms could be dependent, multiple
arms may be queried simultaneously, and multiple (cooperating or competing)
agents may be querying arms, covering combinatorial semi-bandit type settings
as well. Prior work has considered in great detail the setting where each arm's
reward distribution is independent and sub-Gaussian, and a single arm is
queried at each step. Our framework recovers matching sample complexity
guarantees in this special case, and performs comparably or better in practice.
For other settings, sample complexities will depend on the finer details of the
problem (composite nulls being tested, exploration algorithm, data dependence
structure, stopping rule) and we do not explore these; our contribution is to
show that the FDR guarantee is clean and entirely agnostic to these details.

    

### [[2107.07331] Modeling Accurate Human Activity Recognition for Embedded Devices Using Multi-level Distillation](http://arxiv.org/abs/2107.07331)


  Human activity recognition (HAR) based on IMU sensors is an essential domain
in ubiquitous computing. Because of the improving trend to deploy artificial
intelligence into IoT devices or smartphones, more researchers design the HAR
models for embedded devices. We propose a plug-and-play HAR modeling pipeline
with multi-level distillation to build deep convolutional HAR models with
native support of embedded devices. SMLDist consists of stage distillation,
memory distillation, and logits distillation, which covers all the information
flow of the deep models. Stage distillation constrains the learning direction
of the intermediate features. Memory distillation teaches the student models
how to explain and store the inner relationship between high-dimensional
features based on Hopfield networks. Logits distillation constructs distilled
logits by a smoothed conditional rule to keep the probable distribution and
improve the correctness of the soft target. We compare the performance of
accuracy, F1 macro score, and energy cost on the embedded platform of various
state-of-the-art HAR frameworks with a MobileNet V3 model built by SMLDist. The
produced model has well balance with robustness, efficiency, and accuracy.
SMLDist can also compress the models with minor performance loss in an equal
compression rate than other state-of-the-art knowledge distillation methods on
seven public datasets.

    

### [[2107.07334] Tournesol: A quest for a large, secure and trustworthy database of reliable human judgments](http://arxiv.org/abs/2107.07334)


  Today's large-scale algorithms have become immensely influential, as they
recommend and moderate the content that billions of humans are exposed to on a
daily basis. They are the de-facto regulators of our societies' information
diet, from shaping opinions on public health to organizing groups for social
movements. This creates serious concerns, but also great opportunities to
promote quality information. Addressing the concerns and seizing the
opportunities is a challenging, enormous and fabulous endeavor, as intuitively
appealing ideas often come with unwanted {\it side effects}, and as it requires
us to think about what we deeply prefer.
Understanding how today's large-scale algorithms are built is critical to
determine what interventions will be most effective. Given that these
algorithms rely heavily on {\it machine learning}, we make the following key
observation: \emph{any algorithm trained on uncontrolled data must not be
trusted}. Indeed, a malicious entity could take control over the data, poison
it with dangerously manipulative fabricated inputs, and thereby make the
trained algorithm extremely unsafe. We thus argue that the first step towards
safe and ethical large-scale algorithms must be the collection of a large,
secure and trustworthy dataset of reliable human judgments.
To achieve this, we introduce \emph{Tournesol}, an open source platform
available at \url{https://tournesol.app}. Tournesol aims to collect a large
database of human judgments on what algorithms ought to widely recommend (and
what they ought to stop widely recommending). We outline the structure of the
Tournesol database, the key features of the Tournesol platform and the main
hurdles that must be overcome to make it a successful project. Most
importantly, we argue that, if successful, Tournesol may then serve as the
essential foundation for any safe and ethical large-scale algorithm.

    

### [[2107.07338] An Overview of Machine Learning-aided Optical Performance Monitoring Techniques](http://arxiv.org/abs/2107.07338)


  Future communication systems are faced with increased demand for high
capacity, dynamic bandwidth, reliability and heterogeneous traffic. To meet
these requirements, networks have become more complex and thus require new
design methods and monitoring techniques, as they evolve towards becoming
autonomous. Machine learning has come to the forefront in recent years as a
promising technology to aid in this evolution. Optical fiber communications can
already provide the high capacity required for most applications, however,
there is a need for increased scalability and adaptability to changing user
demands and link conditions. Accurate performance monitoring is an integral
part of this transformation. In this paper we review optical performance
monitoring techniques where machine learning algorithms have been applied.
Moreover, since alot of OPM depends on knowledge of the signal type, we also
review work for modulation format recognition and bitrate identification. We
additionally briefly introduce a neuromorphic approach to OPM as an emerging
technique that has only recently been applied to this domain.

    

### [[2107.07341] Leveraging wisdom of the crowds to improve consensus among radiologists by real time, blinded collaborations on a digital swarm platform](http://arxiv.org/abs/2107.07341)


  Radiologists today play a key role in making diagnostic decisions and
labeling images for training A.I. algorithms. Low inter-reader reliability
(IRR) can be seen between experts when interpreting challenging cases. While
teams-based decisions are known to outperform individual decisions,
inter-personal biases often creep up in group interactions which limit
non-dominant participants from expressing true opinions. To overcome the dual
problems of low consensus and inter-personal bias, we explored a solution
modeled on biological swarms of bees. Two separate cohorts; three radiologists
and five radiology residents collaborated on a digital swarm platform in real
time and in a blinded fashion, grading meniscal lesions on knee MR exams. These
consensus votes were benchmarked against clinical (arthroscopy) and
radiological (senior-most radiologist) observations. The IRR of the consensus
votes was compared to the IRR of the majority and most confident votes of the
two cohorts.The radiologist cohort saw an improvement of 23% in IRR of swarm
votes over majority vote. Similar improvement of 23% in IRR in 3-resident swarm
votes over majority vote, was observed. The 5-resident swarm had an even higher
improvement of 32% in IRR over majority vote. Swarm consensus votes also
improved specificity by up to 50%. The swarm consensus votes outperformed
individual and majority vote decisions in both the radiologists and resident
cohorts. The 5-resident swarm had higher IRR than 3-resident swarm indicating
positive effect of increased swarm size. The attending and resident swarms also
outperformed predictions from a state-of-the-art A.I. algorithm. Utilizing a
digital swarm platform improved agreement and allows participants to express
judgement free intent, resulting in superior clinical performance and robust
A.I. training labels.

    

### [[2107.07342] Probabilistic analysis of solar cell optical performance using Gaussian processes](http://arxiv.org/abs/2107.07342)


  This work investigates application of different machine learning based
prediction methodologies to estimate the performance of silicon based textured
cells. Concept of confidence bound regions is introduced and advantages of this
concept are discussed in detail. Results show that reflection profiles and
depth dependent optical generation profiles can be accurately estimated using
Gaussian processes with exact knowledge of uncertainty in the prediction
this http URL is also shown that cell design parameters can be estimated for a
desired performance metric.

    

### [[2107.07343] Mutation is all you need](http://arxiv.org/abs/2107.07343)


  Neural architecture search (NAS) promises to make deep learning accessible to
non-experts by automating architecture engineering of deep neural networks.
BANANAS is one state-of-the-art NAS method that is embedded within the Bayesian
optimization framework. Recent experimental findings have demonstrated the
strong performance of BANANAS on the NAS-Bench-101 benchmark being determined
by its path encoding and not its choice of surrogate model. We present
experimental results suggesting that the performance of BANANAS on the
NAS-Bench-301 benchmark is determined by its acquisition function optimizer,
which minimally mutates the incumbent.

    

### [[2107.07344] Framework for A Personalized Intelligent Assistant to Elderly People for Activities of Daily Living](http://arxiv.org/abs/2107.07344)


  The increasing population of elderly people is associated with the need to
meet their increasing requirements and to provide solutions that can improve
their quality of life in a smart home. In addition to fear and anxiety towards
interfacing with systems; cognitive disabilities, weakened memory, disorganized
behavior and even physical limitations are some of the problems that elderly
people tend to face with increasing age. The essence of providing
technology-based solutions to address these needs of elderly people and to
create smart and assisted living spaces for the elderly; lies in developing
systems that can adapt by addressing their diversity and can augment their
performances in the context of their day to day goals. Therefore, this work
proposes a framework for development of a Personalized Intelligent Assistant to
help elderly people perform Activities of Daily Living (ADLs) in a smart and
connected Internet of Things (IoT) based environment. This Personalized
Intelligent Assistant can analyze different tasks performed by the user and
recommend activities by considering their daily routine, current affective
state and the underlining user experience. To uphold the efficacy of this
proposed framework, it has been tested on a couple of datasets for modelling an
average user and a specific user respectively. The results presented show that
the model achieves a performance accuracy of 73.12% when modelling a specific
user, which is considerably higher than its performance while modelling an
average user, this upholds the relevance for development and implementation of
this proposed framework.

    

### [[2107.07345] Inferring the Structure of Ordinary Differential Equations](http://arxiv.org/abs/2107.07345)


  Understanding physical phenomena oftentimes means understanding the
underlying dynamical system that governs observational measurements. While
accurate prediction can be achieved with black box systems, they often lack
interpretability and are less amenable for further expert investigation.
Alternatively, the dynamics can be analysed via symbolic regression. In this
paper, we extend the approach by (Udrescu et al., 2020) called AIFeynman to the
dynamic setting to perform symbolic regression on ODE systems based on
observations from the resulting trajectories. We compare this extension to
state-of-the-art approaches for symbolic regression empirically on several
dynamical systems for which the ground truth equations of increasing complexity
are available. Although the proposed approach performs best on this benchmark,
we observed difficulties of all the compared symbolic regression approaches on
more complex systems, such as Cart-Pole.

    

### [[2107.07346] You Do Not Need a Bigger Boat: Recommendations at Reasonable Scale in a (Mostly) Serverless and Open Stack](http://arxiv.org/abs/2107.07346)


  We argue that immature data pipelines are preventing a large portion of
industry practitioners from leveraging the latest research on recommender
systems. We propose our template data stack for machine learning at "reasonable
scale", and show how many challenges are solved by embracing a serverless
paradigm. Leveraging our experience, we detail how modern open source can
provide a pipeline processing terabytes of data with limited infrastructure
work.

    

### [[2107.07349] A multi-schematic classifier-independent oversampling approach for imbalanced datasets](http://arxiv.org/abs/2107.07349)


  Over 85 oversampling algorithms, mostly extensions of the SMOTE algorithm,
have been built over the past two decades, to solve the problem of imbalanced
datasets. However, it has been evident from previous studies that different
oversampling algorithms have different degrees of efficiency with different
classifiers. With numerous algorithms available, it is difficult to decide on
an oversampling algorithm for a chosen classifier. Here, we overcome this
problem with a multi-schematic and classifier-independent oversampling
approach: ProWRAS(Proximity Weighted Random Affine Shadowsampling). ProWRAS
integrates the Localized Random Affine Shadowsampling (LoRAS)algorithm and the
Proximity Weighted Synthetic oversampling (ProWSyn) algorithm. By controlling
the variance of the synthetic samples, as well as a proximity-weighted
clustering system of the minority classdata, the ProWRAS algorithm improves
performance, compared to algorithms that generate synthetic samples through
modelling high dimensional convex spaces of the minority class. ProWRAS has
four oversampling schemes, each of which has its unique way to model the
variance of the generated data. Most importantly, the performance of ProWRAS
with proper choice of oversampling schemes, is independent of the classifier
used. We have benchmarked our newly developed ProWRAS algorithm against five
sate-of-the-art oversampling models and four different classifiers on 20
publicly available datasets. ProWRAS outperforms other oversampling algorithms
in a statistically significant way, in terms of both F1-score and Kappa-score.
Moreover, we have introduced a novel measure for classifier independence
I-score, and showed quantitatively that ProWRAS performs better, independent of
the classifier used. In practice, ProWRAS customizes synthetic sample
generation according to a classifier of choice and thereby reduces benchmarking
efforts.

    

### [[2107.07352] Copula-Based Normalizing Flows](http://arxiv.org/abs/2107.07352)


  Normalizing flows, which learn a distribution by transforming the data to
samples from a Gaussian base distribution, have proven powerful density
approximations. But their expressive power is limited by this choice of the
base distribution. We, therefore, propose to generalize the base distribution
to a more elaborate copula distribution to capture the properties of the target
distribution more accurately. In a first empirical analysis, we demonstrate
that this replacement can dramatically improve the vanilla normalizing flows in
terms of flexibility, stability, and effectivity for heavy-tailed data. Our
results suggest that the improvements are related to an increased local
Lipschitz-stability of the learned flow.

    

### [[2107.07364] SilGAN: Generating driving maneuvers for scenario-based software-in-the-loop testing](http://arxiv.org/abs/2107.07364)


  Automotive software testing continues to rely largely upon expensive field
tests to ensure quality because alternatives like simulation-based testing are
relatively immature. As a step towards lowering reliance on field tests, we
present SilGAN, a deep generative model that eases specification, stimulus
generation, and automation of automotive software-in-the-loop testing. The
model is trained using data recorded from vehicles in the field. Upon training,
the model uses a concise specification for a driving scenario to generate
realistic vehicle state transitions that can occur during such a scenario. Such
authentic emulation of internal vehicle behavior can be used for rapid,
systematic and inexpensive testing of vehicle control software. In addition, by
presenting a targeted method for searching through the information learned by
the model, we show how a test objective like code coverage can be automated.
The data driven end-to-end testing pipeline that we present vastly expands the
scope and credibility of automotive simulation-based testing. This reduces time
to market while helping maintain required standards of quality.

    

### [[2107.07373] A Reinforcement Learning Environment for Mathematical Reasoning via Program Synthesis](http://arxiv.org/abs/2107.07373)


  We convert the DeepMind Mathematics Dataset into a reinforcement learning
environment by interpreting it as a program synthesis problem. Each action
taken in the environment adds an operator or an input into a discrete compute
graph. Graphs which compute correct answers yield positive reward, enabling the
optimization of a policy to construct compute graphs conditioned on problem
statements. Baseline models are trained using Double DQN on various subsets of
problem types, demonstrating the capability to learn to correctly construct
graphs despite the challenges of combinatorial explosion and noisy rewards.

    

### [[2107.07376] Proceedings of the Sixteenth Workshop on Logical Frameworks and Meta-Languages: Theory and Practice](http://arxiv.org/abs/2107.07376)


  Logical frameworks and meta-languages form a common substrate for
representing, implementing and reasoning about a wide variety of deductive
systems of interest in logic and computer science. Their design, implementation
and their use in reasoning tasks, ranging from the correctness of software to
the properties of formal systems, have been the focus of considerable research
over the last two decades. This workshop brings together designers,
implementors and practitioners to discuss various aspects impinging on the
structure and utility of logical frameworks, including the treatment of
variable binding, inductive and co-inductive reasoning techniques and the
expressiveness and lucidity of the reasoning process.

    

### [[2107.07382] Hybrid Ant Swarm-Based Data Clustering](http://arxiv.org/abs/2107.07382)


  Biologically inspired computing techniques are very effective and useful in
many areas of research including data clustering. Ant clustering algorithm is a
nature-inspired clustering technique which is extensively studied for over two
decades. In this study, we extend the ant clustering algorithm (ACA) to a
hybrid ant clustering algorithm (hACA). Specifically, we include a genetic
algorithm in standard ACA to extend the hybrid algorithm for better
performance. We also introduced novel pick up and drop off rules to speed up
the clustering performance. We study the performance of the hACA algorithm and
compare with standard ACA as a benchmark.

    

### [[2107.07384] A Fixed Version of Quadratic Program in Gradient Episodic Memory](http://arxiv.org/abs/2107.07384)


  Gradient Episodic Memory is indeed a novel method for continual learning,
which solves new problems quickly without forgetting previously acquired
knowledge. However, in the process of studying the paper, we found there were
some problems in the proof of the dual problem of Quadratic Program, so here we
give our fixed version for this problem.

    

### [[2107.07393] Auditing for Diversity using Representative Examples](http://arxiv.org/abs/2107.07393)


  Assessing the diversity of a dataset of information associated with people is
crucial before using such data for downstream applications. For a given
dataset, this often involves computing the imbalance or disparity in the
empirical marginal distribution of a protected attribute (e.g. gender, dialect,
etc.). However, real-world datasets, such as images from Google Search or
collections of Twitter posts, often do not have protected attributes labeled.
Consequently, to derive disparity measures for such datasets, the elements need
to hand-labeled or crowd-annotated, which are expensive processes.
We propose a cost-effective approach to approximate the disparity of a given
unlabeled dataset, with respect to a protected attribute, using a control set
of labeled representative examples. Our proposed algorithm uses the pairwise
similarity between elements in the dataset and elements in the control set to
effectively bootstrap an approximation to the disparity of the dataset.
Importantly, we show that using a control set whose size is much smaller than
the size of the dataset is sufficient to achieve a small approximation error.
Further, based on our theoretical framework, we also provide an algorithm to
construct adaptive control sets that achieve smaller approximation errors than
randomly chosen control sets. Simulations on two image datasets and one Twitter
dataset demonstrate the efficacy of our approach (using random and adaptive
control sets) in auditing the diversity of a wide variety of datasets.

    

### [[2107.07394] Explore and Control with Adversarial Surprise](http://arxiv.org/abs/2107.07394)


  Reinforcement learning (RL) provides a framework for learning goal-directed
policies given user-specified rewards. However, since designing rewards often
requires substantial engineering effort, we are interested in the problem of
learning without rewards, where agents must discover useful behaviors in the
absence of task-specific incentives. Intrinsic motivation is a family of
unsupervised RL techniques which develop general objectives for an RL agent to
optimize that lead to better exploration or the discovery of skills. In this
paper, we propose a new unsupervised RL technique based on an adversarial game
which pits two policies against each other to compete over the amount of
surprise an RL agent experiences. The policies each take turns controlling the
agent. The Explore policy maximizes entropy, putting the agent into surprising
or unfamiliar situations. Then, the Control policy takes over and seeks to
recover from those situations by minimizing entropy. The game harnesses the
power of multi-agent competition to drive the agent to seek out increasingly
surprising parts of the environment while learning to gain mastery over them.
We show empirically that our method leads to the emergence of complex skills by
exhibiting clear phase transitions. Furthermore, we show both theoretically
(via a latent state space coverage argument) and empirically that our method
has the potential to be applied to the exploration of stochastic,
partially-observed environments. We show that Adversarial Surprise learns more
complex behaviors, and explores more effectively than competitive baselines,
outperforming intrinsic motivation methods based on active inference,
novelty-seeking (Random Network Distillation (RND)), and multi-agent
unsupervised RL (Asymmetric Self-Play (ASP)) in MiniGrid, Atari and VizDoom
environments.

    

### [[2107.07402] CLSRIL-23: Cross Lingual Speech Representations for Indic Languages](http://arxiv.org/abs/2107.07402)


  We present a CLSRIL-23, a self supervised learning based audio pre-trained
model which learns cross lingual speech representations from raw audio across
23 Indic languages. It is built on top of wav2vec 2.0 which is solved by
training a contrastive task over masked latent speech representations and
jointly learns the quantization of latents shared across all languages. We
compare the language wise loss during pretraining to compare effects of
monolingual and multilingual pretraining. Performance on some downstream
fine-tuning tasks for speech recognition is also compared and our experiments
show that multilingual pretraining outperforms monolingual training, in terms
of learning speech representations which encodes phonetic similarity of
languages and also in terms of performance on down stream tasks. A decrease of
5% is observed in WER and 9.5% in CER when a multilingual pretrained model is
used for finetuning in Hindi. All the code models are also open sourced.
CLSRIL-23 is a model trained on $23$ languages and almost 10,000 hours of audio
data to facilitate research in speech recognition for Indic languages. We hope
that new state of the art systems will be created using the self supervised
approach, especially for low resources Indic languages.

    

### [[2107.07409] Machine Learning-Based Analysis of Free-Text Keystroke Dynamics](http://arxiv.org/abs/2107.07409)


  The development of active and passive biometric authentication and
identification technology plays an increasingly important role in
cybersecurity. Keystroke dynamics can be used to analyze the way that a user
types based on various keyboard input. Previous work has shown that user
authentication and classification can be achieved based on keystroke dynamics.
In this research, we consider the problem of user classification based on
keystroke dynamics features collected from free-text. We implement and analyze
a novel a deep learning model that combines a convolutional neural network
(CNN) and a gated recurrent unit (GRU). We optimize the resulting model and
consider several relevant related problems. Our model is competitive with the
best results obtained in previous comparable research.

    

### [[2107.07410] PC-MLP: Model-based Reinforcement Learning with Policy Cover Guided Exploration](http://arxiv.org/abs/2107.07410)


  Model-based Reinforcement Learning (RL) is a popular learning paradigm due to
its potential sample efficiency compared to model-free RL. However, existing
empirical model-based RL approaches lack the ability to explore. This work
studies a computationally and statistically efficient model-based algorithm for
both Kernelized Nonlinear Regulators (KNR) and linear Markov Decision Processes
(MDPs). For both models, our algorithm guarantees polynomial sample complexity
and only uses access to a planning oracle. Experimentally, we first demonstrate
the flexibility and efficacy of our algorithm on a set of exploration
challenging control tasks where existing empirical model-based RL approaches
completely fail. We then show that our approach retains excellent performance
even in common dense reward control benchmarks that do not require heavy
exploration. Finally, we demonstrate that our method can also perform
reward-free exploration efficiently. Our code can be found at
this https URL.

    

### [[2107.07412] Assign Hysteresis Parameter For Ericsson BTS Power Saving Algorithm Using Unsupervised Learning](http://arxiv.org/abs/2107.07412)


  Gaza Strip suffers from a chronic electricity deficit that affects all
industries including the telecommunication field, so there is a need to
optimize and reduce power consumption of the telecommunication equipment. In
this paper we propose a new model that helps GSM radio frequency engineers to
choose the optimal value of hysteresis parameter for Ericsson BTS power saving
algorithm which aims to switch OFF unused frequency channels, our model is
based on unsupervised machine learning clustering K-means algorithm. By using
our model with BTS power saving algorithm we reduce number of active TRX by
20.9%.

    

### [[2107.07420] Optimal Scoring Rule Design](http://arxiv.org/abs/2107.07420)


  This paper introduces an optimization problem for proper scoring rule design.
Consider a principal who wants to collect an agent's prediction about an
unknown state. The agent can either report his prior prediction or access a
costly signal and report the posterior prediction. Given a collection of
possible distributions containing the agent's posterior prediction
distribution, the principal's objective is to design a bounded scoring rule to
maximize the agent's worst-case payoff increment between reporting his
posterior prediction and reporting his prior prediction.
We study two settings of such optimization for proper scoring rules: static
and asymptotic settings. In the static setting, where the agent can access one
signal, we propose an efficient algorithm to compute an optimal scoring rule
when the collection of distributions is finite. The agent can adaptively and
indefinitely refine his prediction in the asymptotic setting. We first consider
a sequence of collections of posterior distributions with vanishing covariance,
which emulates general estimators with large samples, and show the optimality
of the quadratic scoring rule. Then, when the agent's posterior distribution is
a Beta-Bernoulli process, we find that the log scoring rule is optimal. We also
prove the optimality of the log scoring rule over a smaller set of functions
for categorical distributions with Dirichlet priors.

    

### [[2107.07423] Untrained DNN for Channel Estimation of RIS-Assisted Multi-User OFDM System with Hardware Impairments](http://arxiv.org/abs/2107.07423)


  Reconfigurable intelligent surface (RIS) is an emerging technology for
improving performance in fifth-generation (5G) and beyond networks. Practically
channel estimation of RIS-assisted systems is challenging due to the passive
nature of the RIS. The purpose of this paper is to introduce a deep
learning-based, low complexity channel estimator for the RIS-assisted
multi-user single-input-multiple-output (SIMO) orthogonal frequency division
multiplexing (OFDM) system with hardware impairments. We propose an untrained
deep neural network (DNN) based on the deep image prior (DIP) network to
denoise the effective channel of the system obtained from the conventional
pilot-based least-square (LS) estimation and acquire a more accurate
estimation. We have shown that our proposed method has high performance in
terms of accuracy and low complexity compared to conventional methods. Further,
we have shown that the proposed estimator is robust to interference caused by
the hardware impairments at the transceiver and RIS.

    

### [[2107.07431] High carbon stock mapping at large scale with optical satellite imagery and spaceborne LIDAR](http://arxiv.org/abs/2107.07431)


  The increasing demand for commodities is leading to changes in land use
worldwide. In the tropics, deforestation, which causes high carbon emissions
and threatens biodiversity, is often linked to agricultural expansion. While
the need for deforestation-free global supply chains is widely recognized,
making progress in practice remains a challenge. Here, we propose an automated
approach that aims to support conservation and sustainable land use planning
decisions by mapping tropical landscapes at large scale and high spatial
resolution following the High Carbon Stock (HCS) approach. A deep learning
approach is developed that estimates canopy height for each 10 m Sentinel-2
pixel by learning from sparse GEDI LIDAR reference data, achieving an overall
RMSE of 6.3 m. We show that these wall-to-wall maps of canopy top height are
predictive for classifying HCS forests and degraded areas with an overall
accuracy of 86 % and produce a first high carbon stock map for Indonesia,
Malaysia, and the Philippines.

    

### [[2107.07432] Hierarchical graph neural nets can capture long-range interactions](http://arxiv.org/abs/2107.07432)


  Graph neural networks (GNNs) based on message passing between neighboring
nodes are known to be insufficient for capturing long-range interactions in
graphs. In this project we study hierarchical message passing models that
leverage a multi-resolution representation of a given graph. This facilitates
learning of features that span large receptive fields without loss of local
information, an aspect not studied in preceding work on hierarchical GNNs. We
introduce Hierarchical Graph Net (HGNet), which for any two connected nodes
guarantees existence of message-passing paths of at most logarithmic length
w.r.t. the input graph size. Yet, under mild assumptions, its internal
hierarchy maintains asymptotic size equivalent to that of the input graph. We
observe that our HGNet outperforms conventional stacking of GCN layers
particularly in molecular property prediction benchmarks. Finally, we propose
two benchmarking tasks designed to elucidate capability of GNNs to leverage
long-range interactions in graphs.

    

### [[2107.07436] FastSHAP: Real-Time Shapley Value Estimation](http://arxiv.org/abs/2107.07436)


  Shapley values are widely used to explain black-box models, but they are
costly to calculate because they require many model evaluations. We introduce
FastSHAP, a method for estimating Shapley values in a single forward pass using
a learned explainer model. FastSHAP amortizes the cost of explaining many
inputs via a learning approach inspired by the Shapley value's weighted least
squares characterization, and it can be trained using standard stochastic
gradient optimization. We compare FastSHAP to existing estimation approaches,
revealing that it generates high-quality explanations with orders of magnitude
speedup.

    

### [[2107.07438] Convolutional Neural Bandit: Provable Algorithm for Visual-aware Advertising](http://arxiv.org/abs/2107.07438)


  Online advertising is ubiquitous in web business. Image displaying is
considered as one of the most commonly used formats to interact with customers.
Contextual multi-armed bandit has shown success in the application of
advertising to solve the exploration-exploitation dilemma existed in the
recommendation procedure. Inspired by the visual-aware advertising, in this
paper, we propose a contextual bandit algorithm, where the convolutional neural
network (CNN) is utilized to learn the reward function along with an upper
confidence bound (UCB) for exploration. We also prove a near-optimal regret
bound $\tilde{\mathcal{O}}(\sqrt{T})$ when the network is over-parameterized
and establish strong connections with convolutional neural tangent kernel
(CNTK). Finally, we evaluate the empirical performance of the proposed
algorithm and show that it outperforms other state-of-the-art UCB-based bandit
algorithms on real-world image data sets.

    

### [[2107.07443] Multi-label Chaining with Imprecise Probabilities](http://arxiv.org/abs/2107.07443)


  We present two different strategies to extend the classical multi-label
chaining approach to handle imprecise probability estimates. These estimates
use convex sets of distributions (or credal sets) in order to describe our
uncertainty rather than a precise one. The main reasons one could have for
using such estimations are (1) to make cautious predictions (or no decision at
all) when a high uncertainty is detected in the chaining and (2) to make better
precise predictions by avoiding biases caused in early decisions in the
chaining. Through the use of the naive credal classifier, we propose efficient
procedures with theoretical justifications to solve both strategies. Our
experimental results on missing labels, which investigate how reliable these
predictions are in both approaches, indicate that our approaches produce
relevant cautiousness on those hard-to-predict instances where the precise
models fail.

    

### [[2107.07445] AutoBERT-Zero: Evolving BERT Backbone from Scratch](http://arxiv.org/abs/2107.07445)


  Transformer-based pre-trained language models like BERT and its variants have
recently achieved promising performance in various natural language processing
(NLP) tasks. However, the conventional paradigm constructs the backbone by
purely stacking the manually designed global self-attention layers, introducing
inductive bias and thus leading to sub-optimal. In this work, we propose an
Operation-Priority Neural Architecture Search (OP-NAS) algorithm to
automatically search for promising hybrid backbone architectures. Our
well-designed search space (i) contains primitive math operations in the
intra-layer level to explore novel attention structures, and (ii) leverages
convolution blocks to be the supplementary for attention structure in the
inter-layer level to better learn local dependency. We optimize both the search
algorithm and evaluation of candidate models to boost the efficiency of our
proposed OP-NAS. Specifically, we propose Operation-Priority (OP) evolution
strategy to facilitate model search via balancing exploration and exploitation.
Furthermore, we design a Bi-branch Weight-Sharing (BIWS) training strategy for
fast model evaluation. Extensive experiments show that the searched
architecture (named AutoBERT-Zero) significantly outperforms BERT and its
variants of different model capacities in various downstream tasks, proving the
architecture's transfer and generalization abilities. Remarkably,
AutoBERT-Zero-base outperforms RoBERTa-base (using much more data) and
BERT-large (with much larger model size) by 2.4 and 1.4 higher score on GLUE
test set. Code and pre-trained models will be made publicly available.

    

### [[2107.07451] Data vs classifiers, who wins?](http://arxiv.org/abs/2107.07451)


  The classification experiments covered by machine learning (ML) are composed
by two important parts: the data and the algorithm. As they are a fundamental
part of the problem, both must be considered when evaluating a model's
performance against a benchmark. The best classifiers need robust benchmarks to
be properly evaluated. For this, gold standard benchmarks such as OpenML-CC18
are used. However, data complexity is commonly not considered along with the
model during a performance evaluation. Recent studies employ Item Response
Theory (IRT) as a new approach to evaluating datasets and algorithms, capable
of evaluating both simultaneously. This work presents a new evaluation
methodology based on IRT and Glicko-2, jointly with the decodIRT tool developed
to guide the estimation of IRT in ML. It explores the IRT as a tool to evaluate
the OpenML-CC18 benchmark for its algorithmic evaluation capability and checks
if there is a subset of datasets more efficient than the original benchmark.
Several classifiers, from classics to ensemble, are also evaluated using the
IRT models. The Glicko-2 rating system was applied together with IRT to
summarize the innate ability and classifiers performance. It was noted that not
all OpenML-CC18 datasets are really useful for evaluating algorithms, where
only 10% were rated as being really difficult. Furthermore, it was verified the
existence of a more efficient subset containing only 50% of the original size.
While Randon Forest was singled out as the algorithm with the best innate
ability.

    

### [[2107.07455] Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks](http://arxiv.org/abs/2107.07455)


  There has been significant research done on developing methods for improving
robustness to distributional shift and uncertainty estimation. In contrast,
only limited work has examined developing standard datasets and benchmarks for
assessing these approaches. Additionally, most work on uncertainty estimation
and robustness has developed new techniques based on small-scale regression or
image classification tasks. However, many tasks of practical interest have
different modalities, such as tabular data, audio, text, or sensor data, which
offer significant challenges involving regression and discrete or continuous
structured prediction. Thus, given the current state of the field, a
standardized large-scale dataset of tasks across a range of modalities affected
by distributional shifts is necessary. This will enable researchers to
meaningfully evaluate the plethora of recently developed uncertainty
quantification methods, as well as assessment criteria and state-of-the-art
baselines. In this work, we propose the \emph{Shifts Dataset} for evaluation of
uncertainty estimates and robustness to distributional shift. The dataset,
which has been collected from industrial sources and services, is composed of
three tasks, with each corresponding to a particular data modality: tabular
weather prediction, machine translation, and self-driving car (SDC) vehicle
motion prediction. All of these data modalities and tasks are affected by real,
`in-the-wild' distributional shifts and pose interesting challenges with
respect to uncertainty estimation. In this work we provide a description of the
dataset and baseline results for all tasks.

    

### [[2107.07467] Only Train Once: A One-Shot Neural Network Training And Pruning Framework](http://arxiv.org/abs/2107.07467)


  Structured pruning is a commonly used technique in deploying deep neural
networks (DNNs) onto resource-constrained devices. However, the existing
pruning methods are usually heuristic, task-specified, and require an extra
fine-tuning procedure. To overcome these limitations, we propose a framework
that compresses DNNs into slimmer architectures with competitive performances
and significant FLOPs reductions by Only-Train-Once (OTO). OTO contains two
keys: (i) we partition the parameters of DNNs into zero-invariant groups,
enabling us to prune zero groups without affecting the output; and (ii) to
promote zero groups, we then formulate a structured-sparsity optimization
problem and propose a novel optimization algorithm, Half-Space Stochastic
Projected Gradient (HSPG), to solve it, which outperforms the standard proximal
methods on group sparsity exploration and maintains comparable convergence. To
demonstrate the effectiveness of OTO, we train and compress full models
simultaneously from scratch without fine-tuning for inference speedup and
parameter reduction, and achieve state-of-the-art results on VGG16 for CIFAR10,
ResNet50 for CIFAR10/ImageNet and Bert for SQuAD.

    

### [[2107.07480] Newton-LESS: Sparsification without Trade-offs for the Sketched Newton Update](http://arxiv.org/abs/2107.07480)


  In second-order optimization, a potential bottleneck can be computing the
Hessian matrix of the optimized function at every iteration. Randomized
sketching has emerged as a powerful technique for constructing estimates of the
Hessian which can be used to perform approximate Newton steps. This involves
multiplication by a random sketching matrix, which introduces a trade-off
between the computational cost of sketching and the convergence rate of the
optimization algorithm. A theoretically desirable but practically much too
expensive choice is to use a dense Gaussian sketching matrix, which produces
unbiased estimates of the exact Newton step and which offers strong
problem-independent convergence guarantees. We show that the Gaussian sketching
matrix can be drastically sparsified, significantly reducing the computational
cost of sketching, without substantially affecting its convergence properties.
This approach, called Newton-LESS, is based on a recently introduced sketching
technique: LEverage Score Sparsified (LESS) embeddings. We prove that
Newton-LESS enjoys nearly the same problem-independent local convergence rate
as Gaussian embeddings, not just up to constant factors but even down to lower
order terms, for a large class of optimization tasks. In particular, this leads
to a new state-of-the-art convergence result for an iterative least squares
solver. Finally, we extend LESS embeddings to include uniformly sparsified
random sign matrices which can be implemented efficiently and which perform
well in numerical experiments.

    

### [[2107.07483] Personalized and Reliable Decision Sets: Enhancing Interpretability in Clinical Decision Support Systems](http://arxiv.org/abs/2107.07483)


  In this study, we present a novel clinical decision support system and
discuss its interpretability-related properties. It combines a decision set of
rules with a machine learning scheme to offer global and local
interpretability. More specifically, machine learning is used to predict the
likelihood of each of those rules to be correct for a particular patient, which
may also contribute to better predictive performances. Moreover, the
reliability analysis of individual predictions is also addressed, contributing
to further personalized interpretability. The combination of these several
elements may be crucial to obtain the clinical stakeholders' trust, leading to
a better assessment of patients' conditions and improvement of the physicians'
decision-making.

    

### [[2107.07493] Algorithmic Concept-based Explainable Reasoning](http://arxiv.org/abs/2107.07493)


  Recent research on graph neural network (GNN) models successfully applied
GNNs to classical graph algorithms and combinatorial optimisation problems.
This has numerous benefits, such as allowing applications of algorithms when
preconditions are not satisfied, or reusing learned models when sufficient
training data is not available or can't be generated. Unfortunately, a key
hindrance of these approaches is their lack of explainability, since GNNs are
black-box models that cannot be interpreted directly. In this work, we address
this limitation by applying existing work on concept-based explanations to GNN
models. We introduce concept-bottleneck GNNs, which rely on a modification to
the GNN readout mechanism. Using three case studies we demonstrate that: (i)
our proposed model is capable of accurately learning concepts and extracting
propositional formulas based on the learned concepts for each target class;
(ii) our concept-based GNN models achieve comparative performance with
state-of-the-art models; (iii) we can derive global graph concepts, without
explicitly providing any supervision on graph-level concepts.

    

### [[2107.07494] Mid-flight Forecasting for CPA Lines in Online Advertising](http://arxiv.org/abs/2107.07494)


  For Verizon MediaDemand Side Platform(DSP), forecasting of ad campaign
performance not only feeds key information to the optimization server to allow
the system to operate on a high-performance mode, but also produces actionable
insights to the advertisers. In this paper, the forecasting problem for CPA
lines in the middle of the flight is investigated by taking the bidding
mechanism into account. The proposed methodology generates relationships
between various key performance metrics and optimization signals. It can also
be used to estimate the sensitivity of ad campaign performance metrics to the
adjustments of optimization signal, which is important to the design of a
campaign management system. The relationship between advertiser spends and
effective Cost Per Action(eCPA) is also characterized, which serves as a
guidance for mid-flight line adjustment to the advertisers. Several practical
issues in implementation, such as downsampling of the dataset, are also
discussed in the paper. At last, the forecasting results are validated against
actual deliveries and demonstrates promising accuracy.

    

### [[2107.07502] MultiBench: Multiscale Benchmarks for Multimodal Representation Learning](http://arxiv.org/abs/2107.07502)


  Learning multimodal representations involves integrating information from
multiple heterogeneous sources of data. It is a challenging yet crucial area
with numerous real-world applications in multimedia, affective computing,
robotics, finance, human-computer interaction, and healthcare. Unfortunately,
multimodal research has seen limited resources to study (1) generalization
across domains and modalities, (2) complexity during training and inference,
and (3) robustness to noisy and missing modalities. In order to accelerate
progress towards understudied modalities and tasks while ensuring real-world
robustness, we release MultiBench, a systematic and unified large-scale
benchmark spanning 15 datasets, 10 modalities, 20 prediction tasks, and 6
research areas. MultiBench provides an automated end-to-end machine learning
pipeline that simplifies and standardizes data loading, experimental setup, and
model evaluation. To enable holistic evaluation, MultiBench offers a
comprehensive methodology to assess (1) generalization, (2) time and space
complexity, and (3) modality robustness. MultiBench introduces impactful
challenges for future research, including scalability to large-scale multimodal
datasets and robustness to realistic imperfections. To accompany this
benchmark, we also provide a standardized implementation of 20 core approaches
in multimodal learning. Simply applying methods proposed in different research
areas can improve the state-of-the-art performance on 9/15 datasets. Therefore,
MultiBench presents a milestone in unifying disjoint efforts in multimodal
research and paves the way towards a better understanding of the capabilities
and limitations of multimodal models, all the while ensuring ease of use,
accessibility, and reproducibility. MultiBench, our standardized code, and
leaderboards are publicly available, will be regularly updated, and welcomes
inputs from the community.

    

### [[2107.07506] Adaptable Agent Populations via a Generative Model of Policies](http://arxiv.org/abs/2107.07506)


  In the natural world, life has found innumerable ways to survive and often
thrive. Between and even within species, each individual is in some manner
unique, and this diversity lends adaptability and robustness to life. In this
work, we aim to learn a space of diverse and high-reward policies on any given
environment. To this end, we introduce a generative model of policies, which
maps a low-dimensional latent space to an agent policy space. Our method
enables learning an entire population of agent policies, without requiring the
use of separate policy parameters. Just as real world populations can adapt and
evolve via natural selection, our method is able to adapt to changes in our
environment solely by selecting for policies in latent space. We test our
generative model's capabilities in a variety of environments, including an
open-ended grid-world and a two-player soccer environment. Code,
visualizations, and additional experiments can be found at
this https URL.

    

### [[2107.07508] USCO-Solver: Solving Undetermined Stochastic Combinatorial Optimization Problems](http://arxiv.org/abs/2107.07508)


  Real-world decision-making systems are often subject to uncertainties that
have to be resolved through observational data. Therefore, we are frequently
confronted with combinatorial optimization problems of which the objective
function is unknown and thus has to be debunked using empirical evidence. In
contrast to the common practice that relies on a learning-and-optimization
strategy, we consider the regression between combinatorial spaces, aiming to
infer high-quality optimization solutions from samples of input-solution pairs
-- without the need to learn the objective function. Our main deliverable is a
universal solver that is able to handle abstract undetermined stochastic
combinatorial optimization problems. For learning foundations, we present
learning-error analysis under the PAC-Bayesian framework using a new
margin-based analysis. In empirical studies, we demonstrate our design using
proof-of-concept experiments, and compare it with other methods that are
potentially applicable. Overall, we obtain highly encouraging experimental
results for several classic combinatorial problems on both synthetic and
real-world datasets.

    

### [[2107.07511] A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification](http://arxiv.org/abs/2107.07511)


  Black-box machine learning learning methods are now routinely used in
high-risk settings, like medical diagnostics, which demand uncertainty
quantification to avoid consequential model failures. Distribution-free
uncertainty quantification (distribution-free UQ) is a user-friendly paradigm
for creating statistically rigorous confidence intervals/sets for such
predictions. Critically, the intervals/sets are valid without distributional
assumptions or model assumptions, with explicit guarantees with finitely many
datapoints. Moreover, they adapt to the difficulty of the input; when the input
example is difficult, the uncertainty intervals/sets are large, signaling that
the model might be wrong. Without much work, one can use distribution-free
methods on any underlying algorithm, such as a neural network, to produce
confidence sets guaranteed to contain the ground truth with a user-specified
probability, such as 90%. Indeed, the methods are easy-to-understand and
general, applying to many modern prediction problems arising in the fields of
computer vision, natural language processing, deep reinforcement learning, and
so on. This hands-on introduction is aimed at a reader interested in the
practical implementation of distribution-free UQ, including conformal
prediction and related methods, who is not necessarily a statistician. We will
include many explanatory illustrations, examples, and code samples in Python,
with PyTorch syntax. The goal is to provide the reader a working understanding
of distribution-free UQ, allowing them to put confidence intervals on their
algorithms, with one self-contained document.

    

### [[1911.02161] Exact Partitioning of High-order Models with a Novel Convex Tensor Cone Relaxation](http://arxiv.org/abs/1911.02161)


  In this paper we propose an algorithm for exact partitioning of high-order
models. We define a general class of $m$-degree Homogeneous Polynomial Models,
which subsumes several examples motivated from prior literature. Exact
partitioning can be formulated as a tensor optimization problem. We relax this
high-order combinatorial problem to a convex conic form problem. To this end,
we carefully define the Carathéodory symmetric tensor cone, and show its
convexity, and the convexity of its dual cone. This allows us to construct a
primal-dual certificate to show that the solution of the convex relaxation is
correct (equal to the unobserved true group assignment) and to analyze the
statistical upper bound of exact partitioning.

    

### [[2002.00865] Designing GANs: A Likelihood Ratio Approach](http://arxiv.org/abs/2002.00865)


  We are interested in the design of generative networks. The training of these
mathematical structures is mostly performed with the help of adversarial
(min-max) optimization problems. We propose a simple methodology for
constructing such problems assuring, at the same time, consistency of the
corresponding solution. We give characteristic examples developed by our
method, some of which can be recognized from other applications, and some are
introduced here for the first time. We present a new metric, the likelihood
ratio, that can be employed online to examine the convergence and stability
during the training of different Generative Adversarial Networks (GANs).
Finally, we compare various possibilities by applying them to well-known
datasets using neural networks of different configurations and sizes.

    

### [[2003.05383] xCos: An Explainable Cosine Metric for Face Verification Task](http://arxiv.org/abs/2003.05383)


  We study the XAI (explainable AI) on the face recognition task, particularly
the face verification here. Face verification is a crucial task in recent days
and it has been deployed to plenty of applications, such as access control,
surveillance, and automatic personal log-on for mobile devices. With the
increasing amount of data, deep convolutional neural networks can achieve very
high accuracy for the face verification task. Beyond exceptional performances,
deep face verification models need more interpretability so that we can trust
the results they generate. In this paper, we propose a novel similarity metric,
called explainable cosine ($xCos$), that comes with a learnable module that can
be plugged into most of the verification models to provide meaningful
explanations. With the help of $xCos$, we can see which parts of the two input
faces are similar, where the model pays its attention to, and how the local
similarities are weighted to form the output $xCos$ score. We demonstrate the
effectiveness of our proposed method on LFW and various competitive benchmarks,
resulting in not only providing novel and desiring model interpretability for
face verification but also ensuring the accuracy as plugging into existing face
recognition models.

    

### [[2003.07982] Adversarial Transferability in Wearable Sensor Systems](http://arxiv.org/abs/2003.07982)


  Machine learning is used for inference and decision making in wearable sensor
systems. However, recent studies have found that machine learning algorithms
are easily fooled by the addition of adversarial perturbations to their inputs.
What is more interesting is that adversarial examples generated for one machine
learning system is also effective against other systems. This property of
adversarial examples is called transferability. In this work, we take the first
stride in studying adversarial transferability in wearable sensor systems from
the following perspectives: 1) transferability between machine learning
systems, 2) transferability across subjects, 3) transferability across sensor
body locations, and 4) transferability across datasets. We found strong
untargeted transferability in most cases. Targeted attacks were less successful
with success scores from $0\%$ to $80\%$. The transferability of adversarial
examples depends on many factors such as the inclusion of data from all
subjects, sensor body position, number of samples in the dataset, type of
learning algorithm, and the distribution of source and target system dataset.
The transferability of adversarial examples decreases sharply when the data
distribution of the source and target system becomes more distinct. We also
provide guidelines for the community for designing robust sensor systems.

    

### [[2004.11094] Consistent Online Gaussian Process Regression Without the Sample Complexity Bottleneck](http://arxiv.org/abs/2004.11094)


  Gaussian processes provide a framework for nonlinear nonparametric Bayesian
inference widely applicable across science and engineering. Unfortunately,
their computational burden scales cubically with the training sample size,
which in the case that samples arrive in perpetuity, approaches infinity. This
issue necessitates approximations for use with streaming data, which to date
mostly lack convergence guarantees. Thus, we develop the first online Gaussian
process approximation that preserves convergence to the population posterior,
i.e., asymptotic posterior consistency, while ameliorating its intractable
complexity growth with the sample size. We propose an online compression scheme
that, following each a posteriori update, fixes an error neighborhood with
respect to the Hellinger metric centered at the current posterior, and greedily
tosses out past kernel dictionary elements until its boundary is hit. We call
the resulting method Parsimonious Online Gaussian Processes (POG). For
diminishing error radius, exact asymptotic consistency is preserved (Theorem
1(i)) at the cost of unbounded memory in the limit. On the other hand, for
constant error radius, POG converges to a neighborhood of the population
posterior (Theorem 1(ii))but with finite memory at-worst determined by the
metric entropy of the feature space (Theorem 2). Experimental results are
presented on several nonlinear regression problems which illuminates the merits
of this approach as compared with alternatives that fix the subspace dimension
defining the history of past points.

    

### [[2005.00054] APo-VAE: Text Generation in Hyperbolic Space](http://arxiv.org/abs/2005.00054)


  Natural language often exhibits inherent hierarchical structure ingrained
with complex syntax and semantics. However, most state-of-the-art deep
generative models learn embeddings only in Euclidean vector space, without
accounting for this structural property of language. In this paper, we
investigate text generation in a hyperbolic latent space to learn continuous
hierarchical representations. An Adversarial Poincare Variational Autoencoder
(APo-VAE) is presented, where both the prior and variational posterior of
latent variables are defined over a Poincare ball via wrapped normal
distributions. By adopting the primal-dual formulation of KL divergence, an
adversarial learning procedure is introduced to empower robust model training.
Extensive experiments in language modeling and dialog-response generation tasks
demonstrate the winning effectiveness of the proposed APo-VAE model over VAEs
in Euclidean latent space, thanks to its superb capabilities in capturing
latent language hierarchies in hyperbolic space.

    

### [[2006.08177] Dissimilarity Mixture Autoencoder for Deep Clustering](http://arxiv.org/abs/2006.08177)


  The dissimilarity mixture autoencoder (DMAE) is a neural network model for
feature-based clustering that incorporates a flexible dissimilarity function
and can be integrated into any kind of deep learning architecture. It
internally represents a dissimilarity mixture model (DMM) that extends
classical methods like K-Means, Gaussian mixture models, or Bregman clustering
to any convex and differentiable dissimilarity function through the
reinterpretation of probabilities as neural network representations. DMAE can
be integrated with deep learning architectures into end-to-end models, allowing
the simultaneous estimation of the clustering and neural network's parameters.
Experimental evaluation was performed on image and text clustering benchmark
datasets showing that DMAE is competitive in terms of unsupervised
classification accuracy and normalized mutual information. The source code with
the implementation of DMAE is publicly available at:
this https URL


### [[2007.10915] Wireless Image Retrieval at the Edge](http://arxiv.org/abs/2007.10915)


  We study the image retrieval problem at the wireless edge, where an edge
device captures an image, which is then used to retrieve similar images from an
edge server. These can be images of the same person or a vehicle taken from
other cameras at different times and locations. Our goal is to maximize the
accuracy of the retrieval task under power and bandwidth constraints over the
wireless link. Due to the stringent delay constraint of the underlying
application, sending the whole image at a sufficient quality is not possible.
We propose two alternative schemes based on digital and analog communications,
respectively. In the digital approach, we first propose a deep neural network
(DNN) aided retrieval-oriented image compression scheme, whose output bit
sequence is transmitted over the channel using conventional channel codes. In
the analog joint source and channel coding (JSCC) approach, the feature vectors
are directly mapped into channel symbols. We evaluate both schemes on image
based re-identification (re-ID) tasks under different channel conditions,
including both static and fading channels. We show that the JSCC scheme
significantly increases the end-to-end accuracy, speeds up the encoding
process, and provides graceful degradation with channel conditions. The
proposed architecture is evaluated through extensive simulations on different
datasets and channel conditions, as well as through ablation studies.

    

### [[2007.15847] A Functional Model for Structure Learning and Parameter Estimation in Continuous Time Bayesian Network: An Application in Identifying Patterns of Multiple Chronic Conditions](http://arxiv.org/abs/2007.15847)


  Bayesian networks are powerful statistical models to study the probabilistic
relationships among set random variables with major applications in disease
modeling and prediction. Here, we propose a continuous time Bayesian network
with conditional dependencies, represented as Poisson regression, to model the
impact of exogenous variables on the conditional dependencies of the network.
We also propose an adaptive regularization method with an intuitive early
stopping feature based on density based clustering for efficient learning of
the structure and parameters of the proposed network. Using a dataset of
patients with multiple chronic conditions extracted from electronic health
records of the Department of Veterans Affairs we compare the performance of the
proposed approach with some of the existing methods in the literature for both
short-term (one-year ahead) and long-term (multi-year ahead) predictions. The
proposed approach provides a sparse intuitive representation of the complex
functional relationships between multiple chronic conditions. It also provides
the capability of analyzing multiple disease trajectories over time given any
combination of prior conditions.

    

### [[2008.00394] Point Cloud Completion by Learning Shape Priors](http://arxiv.org/abs/2008.00394)


  In view of the difficulty in reconstructing object details in point cloud
completion, we propose a shape prior learning method for object completion. The
shape priors include geometric information in both complete and the partial
point clouds. We design a feature alignment strategy to learn the shape prior
from complete points, and a coarse to fine strategy to incorporate partial
prior in the fine stage. To learn the complete objects prior, we first train a
point cloud auto-encoder to extract the latent embeddings from complete points.
Then we learn a mapping to transfer the point features from partial points to
that of the complete points by optimizing feature alignment losses. The feature
alignment losses consist of a L2 distance and an adversarial loss obtained by
Maximum Mean Discrepancy Generative Adversarial Network (MMD-GAN). The L2
distance optimizes the partial features towards the complete ones in the
feature space, and MMD-GAN decreases the statistical distance of two point
features in a Reproducing Kernel Hilbert Space. We achieve state-of-the-art
performances on the point cloud completion task. Our code is available at
this https URL.

    

### [[2009.03887] Low-Rank Training of Deep Neural Networks for Emerging Memory Technology](http://arxiv.org/abs/2009.03887)


  The recent success of neural networks for solving difficult decision tasks
has incentivized incorporating smart decision making "at the edge." However,
this work has traditionally focused on neural network inference, rather than
training, due to memory and compute limitations, especially in emerging
non-volatile memory systems, where writes are energetically costly and reduce
lifespan. Yet, the ability to train at the edge is becoming increasingly
important as it enables real-time adaptability to device drift and
environmental variation, user customization, and federated learning across
devices. In this work, we address two key challenges for training on edge
devices with non-volatile memory: low write density and low auxiliary memory.
We present a low-rank training scheme that addresses these challenges while
maintaining computational efficiency. We then demonstrate the technique on a
representative convolutional neural network across several adaptation problems,
where it out-performs standard SGD both in accuracy and in number of weight
writes.

    

### [[2010.07249] Environment Inference for Invariant Learning](http://arxiv.org/abs/2010.07249)


  Learning models that gracefully handle distribution shifts is central to
research on domain generalization, robust optimization, and fairness. A
promising formulation is domain-invariant learning, which identifies the key
issue of learning which features are domain-specific versus domain-invariant.
An important assumption in this area is that the training examples are
partitioned into "domains" or "environments". Our focus is on the more common
setting where such partitions are not provided. We propose EIIL, a general
framework for domain-invariant learning that incorporates Environment Inference
to directly infer partitions that are maximally informative for downstream
Invariant Learning. We show that EIIL outperforms invariant learning methods on
the CMNIST benchmark without using environment labels, and significantly
outperforms ERM on worst-group performance in the Waterbirds and CivilComments
datasets. Finally, we establish connections between EIIL and algorithmic
fairness, which enables EIIL to improve accuracy and calibration in a fair
prediction problem.

    

### [[2010.10341] Learning to Learn Variational Semantic Memory](http://arxiv.org/abs/2010.10341)


  In this paper, we introduce variational semantic memory into meta-learning to
acquire long-term knowledge for few-shot learning. The variational semantic
memory accrues and stores semantic information for the probabilistic inference
of class prototypes in a hierarchical Bayesian framework. The semantic memory
is grown from scratch and gradually consolidated by absorbing information from
tasks it experiences. By doing so, it is able to accumulate long-term, general
knowledge that enables it to learn new concepts of objects. We formulate memory
recall as the variational inference of a latent memory variable from addressed
contents, which offers a principled way to adapt the knowledge to individual
tasks. Our variational semantic memory, as a new long-term memory module,
confers principled recall and update mechanisms that enable semantic
information to be efficiently accrued and adapted for few-shot learning.
Experiments demonstrate that the probabilistic modelling of prototypes achieves
a more informative representation of object classes compared to deterministic
vectors. The consistent new state-of-the-art performance on four benchmarks
shows the benefit of variational semantic memory in boosting few-shot
recognition.

    

### [[2011.06572] Relative Lipschitzness in Extragradient Methods and a Direct Recipe for Acceleration](http://arxiv.org/abs/2011.06572)


  We show that standard extragradient methods (i.e. mirror prox and dual
extrapolation) recover optimal accelerated rates for first-order minimization
of smooth convex functions. To obtain this result we provide a fine-grained
characterization of the convergence rates of extragradient methods for solving
monotone variational inequalities in terms of a natural condition we call
relative Lipschitzness. We further generalize this framework to handle local
and randomized notions of relative Lipschitzness and thereby recover rates for
box-constrained $\ell_\infty$ regression based on area convexity and complexity
bounds achieved by accelerated (randomized) coordinate descent for smooth
convex function minimization.

    

### [[2012.12368] Understanding Frank-Wolfe Adversarial Training](http://arxiv.org/abs/2012.12368)


  Deep neural networks are easily fooled by small perturbations known as
adversarial attacks. Adversarial Training (AT) is a technique that
approximately solves a robust optimization problem to minimize the worst-case
loss and is widely regarded as the most effective defense against such attacks.
We develop a theoretical framework for adversarial training with FW
optimization (FW-AT) that reveals a geometric connection between the loss
landscape and the distortion of $\ell_\infty$ FW attacks (the attack's $\ell_2$
norm). Specifically, we show that high distortion of FW attacks is equivalent
to low variation along the attack path. It is then experimentally demonstrated
on various deep neural network architectures that $\ell_\infty$ attacks against
robust models achieve near maximal $\ell_2$ distortion. This mathematical
transparency differentiates FW from the more popular Projected Gradient Descent
(PGD) optimization. To demonstrate the utility of our theoretical framework we
develop FW-Adapt, a novel adversarial training algorithm which uses simple
distortion measure to adaptively change number of attack steps during training.
FW-Adapt provides strong robustness at lower training times in comparison to
PGD-AT for a variety of white-box and black-box attacks.

    

### [[2012.14261] A Survey on Neural Network Interpretability](http://arxiv.org/abs/2012.14261)


  Along with the great success of deep neural networks, there is also growing
concern about their black-box nature. The interpretability issue affects
people's trust on deep learning systems. It is also related to many ethical
problems, e.g., algorithmic discrimination. Moreover, interpretability is a
desired property for deep networks to become powerful tools in other research
fields, e.g., drug discovery and genomics. In this survey, we conduct a
comprehensive review of the neural network interpretability research. We first
clarify the definition of interpretability as it has been used in many
different contexts. Then we elaborate on the importance of interpretability and
propose a novel taxonomy organized along three dimensions: type of engagement
(passive vs. active interpretation approaches), the type of explanation, and
the focus (from local to global interpretability). This taxonomy provides a
meaningful 3D view of distribution of papers from the relevant literature as
two of the dimensions are not simply categorical but allow ordinal
subcategories. Finally, we summarize the existing interpretability evaluation
methods and suggest possible research directions inspired by our new taxonomy.

    

### [[2101.12684] Modelling Sovereign Credit Ratings: Evaluating the Accuracy and Driving Factors using Machine Learning Techniques](http://arxiv.org/abs/2101.12684)


  Sovereign credit ratings summarize the creditworthiness of countries. These
ratings have a large influence on the economy and the yields at which
governments can issue new debt. This paper investigates the use of a Multilayer
Perceptron (MLP), Classification and Regression Trees (CART), Support Vector
Machines (SVM), Naïve Bayes (NB), and an Ordered Logit (OL) model for the
prediction of sovereign credit ratings. We show that MLP is best suited for
predicting sovereign credit ratings, with a random cross-validated accuracy of
68%, followed by CART (59%), SVM (41%), NB (38%), and OL (33%). Investigation
of the determining factors shows that there is some heterogeneity in the
important variables across the models. However, the two models with the highest
out-of-sample predictive accuracy, MLP and CART, show a lot of similarities in
the influential variables, with regulatory quality, and GDP per capita as
common important variables. Consistent with economic theory, a higher
regulatory quality and/or GDP per capita are associated with a higher credit
rating.

    

### [[2102.00760] Fast rates in structured prediction](http://arxiv.org/abs/2102.00760)


  Discrete supervised learning problems such as classification are often
tackled by introducing a continuous surrogate problem akin to regression.
Bounding the original error, between estimate and solution, by the surrogate
error endows discrete problems with convergence rates already shown for
continuous instances. Yet, current approaches do not leverage the fact that
discrete problems are essentially predicting a discrete output when continuous
problems are predicting a continuous value. In this paper, we tackle this issue
for general structured prediction problems, opening the way to "super fast"
rates, that is, convergence rates for the excess risk faster than $n^{-1}$,
where $n$ is the number of observations, with even exponential rates with the
strongest assumptions. We first illustrate it for predictors based on nearest
neighbors, generalizing rates known for binary classification to any discrete
problem within the framework of structured prediction. We then consider kernel
ridge regression where we improve known rates in $n^{-1/4}$ to arbitrarily fast
rates, depending on a parameter characterizing the hardness of the problem,
thus allowing, under smoothness assumptions, to bypass the curse of
dimensionality.

    

### [[2102.02789] Disambiguation of weak supervision with exponential convergence rates](http://arxiv.org/abs/2102.02789)


  Machine learning approached through supervised learning requires expensive
annotation of data. This motivates weakly supervised learning, where data are
annotated with incomplete yet discriminative information. In this paper, we
focus on partial labelling, an instance of weak supervision where, from a given
input, we are given a set of potential targets. We review a disambiguation
principle to recover full supervision from weak supervision, and propose an
empirical disambiguation algorithm. We prove exponential convergence rates of
our algorithm under classical learnability assumptions, and we illustrate the
usefulness of our method on practical examples.

    

### [[2102.11658] A Goodness-of-fit Test on the Number of Biclusters in a Relational Data Matrix](http://arxiv.org/abs/2102.11658)


  Biclustering is a method for detecting homogeneous submatrices in a given
observed matrix, and it is an effective tool for relational data analysis.
Although there are many studies that estimate the underlying bicluster
structure of a matrix, few have enabled us to determine the appropriate number
of biclusters in an observed matrix. Recently, a statistical test on the number
of biclusters has been proposed for a regular-grid bicluster structure, where
we assume that the latent bicluster structure can be represented by row-column
clustering. However, when the latent bicluster structure does not satisfy such
regular-grid assumption, the previous test requires a larger number of
biclusters than necessary (i.e., a finer bicluster structure than necessary)
for the null hypothesis to be accepted, which is not desirable in terms of
interpreting the accepted bicluster structure. In this study, we propose a new
statistical test on the number of biclusters that does not require the
regular-grid assumption and derive the asymptotic behavior of the proposed test
statistic in both null and alternative cases. We illustrate the effectiveness
of the proposed method by applying it to both synthetic and practical
relational data matrices.

    

### [[2103.02512] Approximation Algorithms for Socially Fair Clustering](http://arxiv.org/abs/2103.02512)


  We present an $(e^{O(p)} \frac{\log \ell}{\log\log\ell})$-approximation
algorithm for socially fair clustering with the $\ell_p$-objective. In this
problem, we are given a set of points in a metric space. Each point belongs to
one (or several) of $\ell$ groups. The goal is to find a $k$-medians,
$k$-means, or, more generally, $\ell_p$-clustering that is simultaneously good
for all of the groups. More precisely, we need to find a set of $k$ centers $C$
so as to minimize the maximum over all groups $j$ of $\sum_{u \text{ in group
}j} d(u,C)^p$. The socially fair clustering problem was independently proposed
by Ghadiri, Samadi, and Vempala [2021] and Abbasi, Bhaskara, and
Venkatasubramanian [2021]. Our algorithm improves and generalizes their
$O(\ell)$-approximation algorithms for the problem. The natural LP relaxation
for the problem has an integrality gap of $\Omega(\ell)$. In order to obtain
our result, we introduce a strengthened LP relaxation and show that it has an
integrality gap of $\Theta(\frac{\log \ell}{\log\log\ell})$ for a fixed $p$.
Additionally, we present a bicriteria approximation algorithm, which
generalizes the bicriteria approximation of Abbasi et al. [2021].

    

### [[2103.02644] Compute and memory efficient universal sound source separation](http://arxiv.org/abs/2103.02644)


  Recent progress in audio source separation lead by deep learning has enabled
many neural network models to provide robust solutions to this fundamental
estimation problem. In this study, we provide a family of efficient neural
network architectures for general purpose audio source separation while
focusing on multiple computational aspects that hinder the application of
neural networks in real-world scenarios. The backbone structure of this
convolutional network is the SUccessive DOwnsampling and Resampling of
Multi-Resolution Features (SuDoRM-RF) as well as their aggregation which is
performed through simple one-dimensional convolutions. This mechanism enables
our models to obtain high fidelity signal separation in a wide variety of
settings where variable number of sources are present and with limited
computational resources (e.g. floating point operations, memory footprint,
number of parameters and latency). Our experiments show that SuDoRM-RF models
perform comparably and even surpass several state-of-the-art benchmarks with
significantly higher computational resource requirements. The causal variation
of SuDoRM-RF is able to obtain competitive performance in real-time speech
separation of around 10dB scale-invariant signal-to-distortion ratio
improvement (SI-SDRi) while remaining up to 20 times faster than real-time on a
laptop device.

    

### [[2103.04150] Signal Processing on the Permutahedron: Tight Spectral Frames for Ranked Data Analysis](http://arxiv.org/abs/2103.04150)


  Ranked data sets, where m judges/voters specify a preference ranking of n
objects/candidates, are increasingly prevalent in contexts such as political
elections, computer vision, recommender systems, and bioinformatics. The vote
counts for each ranking can be viewed as an n! data vector lying on the
permutahedron, which is a Cayley graph of the symmetric group with vertices
labeled by permutations and an edge when two permutations differ by an adjacent
transposition. Leveraging combinatorial representation theory and recent
progress in signal processing on graphs, we investigate a novel, scalable
transform method to interpret and exploit structure in ranked data. We
represent data on the permutahedron using an overcomplete dictionary of atoms,
each of which captures both smoothness information about the data (typically
the focus of spectral graph decomposition methods in graph signal processing)
and structural information about the data (typically the focus of symmetry
decomposition methods from representation theory). These atoms have a more
naturally interpretable structure than any known basis for signals on the
permutahedron, and they form a Parseval frame, ensuring beneficial numerical
properties such as energy preservation. We develop specialized algorithms and
open software that take advantage of the symmetry and structure of the
permutahedron to improve the scalability of the proposed method, making it more
applicable to the high-dimensional ranked data found in applications.

    

### [[2103.10620] Towards a Dimension-Free Understanding of Adaptive Linear Control](http://arxiv.org/abs/2103.10620)


  We study the problem of adaptive control of the linear quadratic regulator
for systems in very high, or even infinite dimension. We demonstrate that while
sublinear regret requires finite dimensional inputs, the ambient state
dimension of the system need not be bounded in order to perform online control.
We provide the first regret bounds for LQR which hold for infinite dimensional
systems, replacing dependence on ambient dimension with more natural notions of
problem complexity. Our guarantees arise from a novel perturbation bound for
certainty equivalence which scales with the prediction error in estimating the
system parameters, without requiring consistent parameter recovery in more
stringent measures like the operator norm. When specialized to finite
dimensional settings, our bounds recover near optimal dimension and time
horizon dependence.

    

### [[2104.01708] A unified framework for non-negative matrix and tensor factorisations with a smoothed Wasserstein loss](http://arxiv.org/abs/2104.01708)


  Non-negative matrix and tensor factorisations are a classical tool for
finding low-dimensional representations of high-dimensional datasets. In
applications such as imaging, datasets can be regarded as distributions
supported on a space with metric structure. In such a setting, a loss function
based on the Wasserstein distance of optimal transportation theory is a natural
choice since it incorporates the underlying geometry of the data. We introduce
a general mathematical framework for computing non-negative factorisations of
both matrices and tensors with respect to an optimal transport loss. We derive
an efficient computational method for its solution using a convex dual
formulation, and demonstrate the applicability of this approach with several
numerical illustrations with both matrix and tensor-valued data.

    

### [[2105.02579] MCMC-driven importance samplers](http://arxiv.org/abs/2105.02579)


  Monte Carlo methods are the standard procedure for estimating complicated
integrals of multidimensional Bayesian posterior distributions. In this work,
we focus on LAIS, a class of adaptive importance samplers where Markov chain
Monte Carlo (MCMC) algorithms are employed to drive an underlying multiple
importance sampling (IS) scheme. Its power lies in the simplicity of the
layered framework: the upper layer locates proposal densities by means of MCMC
algorithms; while the lower layer handles the multiple IS scheme, in order to
compute the final estimators. The modular nature of LAIS allows for different
possible choices in the upper and lower layers, that will have different
performance and computational costs. In this work, we propose different
enhancements in order to increase the efficiency and reduce the computational
cost, of both upper and lower layers. The different variants are essential if
we aim to address computational challenges arising in real-world applications,
such as highly concentrated posterior distributions (due to large amounts of
data, etc.). Hamiltonian-driven importance samplers are presented and tested.
Furthermore, we introduce different strategies for designing cheaper schemes,
for instance, recycling samples generated in the upper layer and using them in
the final estimators in the lower layer. Numerical experiments show the
benefits of the proposed schemes as compared to the vanilla version of LAIS and
other benchmark methods.

    

### [[2105.04727] Separate but Together: Unsupervised Federated Learning for Speech Enhancement from Non-IID Data](http://arxiv.org/abs/2105.04727)


  We propose FEDENHANCE, an unsupervised federated learning (FL) approach for
speech enhancement and separation with non-IID distributed data across multiple
clients. We simulate a real-world scenario where each client only has access to
a few noisy recordings from a limited and disjoint number of speakers (hence
non-IID). Each client trains their model in isolation using mixture invariant
training while periodically providing updates to a central server. Our
experiments show that our approach achieves competitive enhancement performance
compared to IID training on a single device and that we can further facilitate
the convergence speed and the overall performance using transfer learning on
the server-side. Moreover, we show that we can effectively combine updates from
clients trained locally with supervised and unsupervised losses. We also
release a new dataset LibriFSD50K and its creation recipe in order to
facilitate FL research for source separation problems.

    

### [[2105.09232] Diffusion Approximations for Thompson Sampling](http://arxiv.org/abs/2105.09232)


  We study the behavior of Thompson sampling from the perspective of weak
convergence. In the regime where the gaps between arm means scale as
$1/\sqrt{n}$ with the time horizon $n$, we show that the dynamics of Thompson
sampling evolve according to discrete versions of SDEs and random ODEs. As $n
\to \infty$, we show that the dynamics converge weakly to solutions of the
corresponding SDEs and random ODEs. (Recently, Wager and Xu (arXiv:2101.09855)
independently proposed this regime and developed similar SDE and random ODE
approximations for Thompson sampling in the multi-armed bandit setting.) Our
weak convergence theory, which covers both multi-armed and linear bandit
settings, is developed from first principles using the Continuous Mapping
Theorem and can be directly adapted to analyze other sampling-based bandit
algorithms, for example, algorithms using the bootstrap for exploration. We
also establish an invariance principle for multi-armed bandits with gaps
scaling as $1/\sqrt{n}$ -- for Thompson sampling and related algorithms
involving posterior approximation or the bootstrap, the weak diffusion limits
are in general the same regardless of the specifics of the reward distributions
or the choice of prior. In particular, as suggested by the classical
Bernstein-von Mises normal approximation for posterior distributions, the weak
diffusion limits generally coincide with the limit for normally-distributed
rewards and priors.

    

### [[2105.11366] GMAC: A Distributional Perspective on Actor-Critic Framework](http://arxiv.org/abs/2105.11366)


  In this paper, we devise a distributional framework on actor-critic as a
solution to distributional instability, action type restriction, and conflation
between samples and statistics. We propose a new method that minimizes the
Cramér distance with the multi-step Bellman target distribution generated
from a novel Sample-Replacement algorithm denoted SR($\lambda$), which learns
the correct value distribution under multiple Bellman operations.
Parameterizing a value distribution with Gaussian Mixture Model further
improves the efficiency and the performance of the method, which we name GMAC.
We empirically show that GMAC captures the correct representation of value
distributions and improves the performance of a conventional actor-critic
method with low computational cost, in both discrete and continuous action
spaces using Arcade Learning Environment (ALE) and PyBullet environment.

    

### [[2105.12485] TreeBERT: A Tree-Based Pre-Trained Model for Programming Language](http://arxiv.org/abs/2105.12485)


  Source code can be parsed into the abstract syntax tree (AST) based on
defined syntax rules. However, in pre-training, little work has considered the
incorporation of tree structure into the learning process. In this paper, we
present TreeBERT, a tree-based pre-trained model for improving programming
language-oriented generation tasks. To utilize tree structure, TreeBERT
represents the AST corresponding to the code as a set of composition paths and
introduces node position embedding. The model is trained by tree masked
language modeling (TMLM) and node order prediction (NOP) with a hybrid
objective. TMLM uses a novel masking strategy designed according to the tree's
characteristics to help the model understand the AST and infer the missing
semantics of the AST. With NOP, TreeBERT extracts the syntactical structure by
learning the order constraints of nodes in AST. We pre-trained TreeBERT on
datasets covering multiple programming languages. On code summarization and
code documentation tasks, TreeBERT outperforms other pre-trained models and
state-of-the-art models designed for these tasks. Furthermore, TreeBERT
performs well when transferred to the pre-trained unseen programming language.

    

### [[2105.13309] Concept drift detection and adaptation for federated and continual learning](http://arxiv.org/abs/2105.13309)


  Smart devices, such as smartphones, wearables, robots, and others, can
collect vast amounts of data from their environment. This data is suitable for
training machine learning models, which can significantly improve their
behavior, and therefore, the user experience. Federated learning is a young and
popular framework that allows multiple distributed devices to train deep
learning models collaboratively while preserving data privacy. Nevertheless,
this approach may not be optimal for scenarios where data distribution is
non-identical among the participants or changes over time, causing what is
known as concept drift. Little research has yet been done in this field, but
this kind of situation is quite frequent in real life and poses new challenges
to both continual and federated learning. Therefore, in this work, we present a
new method, called Concept-Drift-Aware Federated Averaging (CDA-FedAvg). Our
proposal is an extension of the most popular federated algorithm, Federated
Averaging (FedAvg), enhancing it for continual adaptation under concept drift.
We empirically demonstrate the weaknesses of regular FedAvg and prove that
CDA-FedAvg outperforms it in this type of scenario.

    

### [[2105.13327] Encoders and Ensembles for Task-Free Continual Learning](http://arxiv.org/abs/2105.13327)


  We present an architecture that is effective for continual learning in an
especially demanding setting, where task boundaries do not exist or are
unknown. Our architecture comprises an encoder, pre-trained on a separate
dataset, and an ensemble of simple one-layer classifiers. Two main innovations
are required to make this combination work. First, the provision of suitably
generic pre-trained encoders has been made possible thanks to recent progress
in self-supervised training methods. Second, pairing each classifier in the
ensemble with a key, where the key-space is identical to the latent space of
the encoder, allows them to be used collectively, yet selectively, via
k-nearest neighbour lookup. We show that models trained with the
encoders-and-ensembles architecture are state-of-the-art for the task-free
setting on standard image classification continual learning benchmarks, and
improve on prior state-of-the-art by a large margin in the most challenging
cases. We also show that the architecture learns well in a fully incremental
setting, where one class is learned at a time, and we demonstrate its
effectiveness in this setting with up to 100 classes. Finally, we show that the
architecture works in a task-free continual learning context where the data
distribution changes gradually, and existing approaches requiring knowledge of
task boundaries cannot be applied.

    

### [[2106.05825] HASI: Hardware-Accelerated Stochastic Inference, A Defense Against Adversarial Machine Learning Attacks](http://arxiv.org/abs/2106.05825)


  Deep Neural Networks (DNNs) are employed in an increasing number of
applications, some of which are safety critical. Unfortunately, DNNs are known
to be vulnerable to so-called adversarial attacks that manipulate inputs to
cause incorrect results that can be beneficial to an attacker or damaging to
the victim. Multiple defenses have been proposed to increase the robustness of
DNNs. In general, these defenses have high overhead, some require
attack-specific re-training of the model or careful tuning to adapt to
different attacks.
This paper presents HASI, a hardware-accelerated defense that uses a process
we call stochastic inference to detect adversarial inputs. We show that by
carefully injecting noise into the model at inference time, we can
differentiate adversarial inputs from benign ones. HASI uses the output
distribution characteristics of noisy inference compared to a non-noisy
reference to detect adversarial inputs. We show an adversarial detection rate
of 86% when applied to VGG16 and 93% when applied to ResNet50, which exceeds
the detection rate of the state of the art approaches, with a much lower
overhead. We demonstrate two software/hardware-accelerated co-designs, which
reduces the performance impact of stochastic inference to 1.58X-2X relative to
the unprotected baseline, compared to 15X-20X overhead for a software-only GPU
implementation.

    

### [[2106.06345] JKOnet: Proximal Optimal Transport Modeling of Population Dynamics](http://arxiv.org/abs/2106.06345)


  Consider a heterogeneous population of points evolving with time. While the
population evolves, both in size and nature, we can observe it periodically,
through snapshots taken at different timestamps. Each of these snapshots is
formed by sampling points from the population at that time, and then creating
features to recover point clouds. While these snapshots describe the
population's evolution on aggregate, they do not provide directly insights on
individual trajectories. This scenario is encountered in several applications,
notably single-cell genomics experiments, tracking of particles, or when
studying crowd motion. In this paper, we propose to model that dynamic as
resulting from the celebrated Jordan-Kinderlehrer-Otto (JKO) proximal scheme.
The JKO scheme posits that the configuration taken by a population at time $t$
is one that trades off a decrease w.r.t. an energy (the model we seek to learn)
penalized by an optimal transport distance w.r.t. the previous configuration.
To that end, we propose JKOnet, a neural architecture that combines an energy
model on measures, with (small) optimal displacements solved with input convex
neural networks (ICNN). We demonstrate the applicability of our model to
explain and predict population dynamics.

    

### [[2106.10544] Learning Space Partitions for Path Planning](http://arxiv.org/abs/2106.10544)


  Path planning, the problem of efficiently discovering high-reward
trajectories, often requires optimizing a high-dimensional and multimodal
reward function. Popular approaches like CEM and CMA-ES greedily focus on
promising regions of the search space and may get trapped in local maxima. DOO
and VOOT balance exploration and exploitation, but use space partitioning
strategies independent of the reward function to be optimized. Recently, LaMCTS
empirically learns to partition the search space in a reward-sensitive manner
for black-box optimization. In this paper, we develop a novel formal regret
analysis for when and why such an adaptive region partitioning scheme works. We
also propose a new path planning method PlaLaM which improves the function
value estimation within each sub-region, and uses a latent representation of
the search space. Empirically, PlaLaM outperforms existing path planning
methods in 2D navigation tasks, especially in the presence of
difficult-to-escape local optima, and shows benefits when plugged into
model-based RL with planning components such as PETS. These gains transfer to
highly multimodal real-world tasks, where we outperform strong baselines in
compiler phase ordering by up to 245% and in molecular design by up to 0.4 on
properties on a 0-1 scale. Code is available at
this https URL.

    

### [[2106.12382] Innovations Autoencoder and its Application in One-class Anomalous Sequence Detection](http://arxiv.org/abs/2106.12382)


  An innovations sequence of a time series is a sequence of independent and
identically distributed random variables with which the original time series
has a causal representation. The innovation at a time is statistically
independent of the history of the time series. As such, it represents the new
information contained at present but not in the past. Because of its simple
probability structure, an innovations sequence is the most efficient signature
of the original. Unlike the principle or independent component analysis
representations, an innovations sequence preserves not only the complete
statistical properties but also the temporal order of the original time series.
An long-standing open problem is to find a computationally tractable way to
extract an innovations sequence of non-Gaussian processes. This paper presents
a deep learning approach, referred to as Innovations Autoencoder (IAE), that
extracts innovations sequences using a causal convolutional neural network. An
application of IAE to the one-class anomalous sequence detection problem with
unknown anomaly and anomaly-free models is also presented.

    

### [[2106.12423] Alias-Free Generative Adversarial Networks](http://arxiv.org/abs/2106.12423)


  We observe that despite their hierarchical convolutional nature, the
synthesis process of typical generative adversarial networks depends on
absolute pixel coordinates in an unhealthy manner. This manifests itself as,
e.g., detail appearing to be glued to image coordinates instead of the surfaces
of depicted objects. We trace the root cause to careless signal processing that
causes aliasing in the generator network. Interpreting all signals in the
network as continuous, we derive generally applicable, small architectural
changes that guarantee that unwanted information cannot leak into the
hierarchical synthesis process. The resulting networks match the FID of
StyleGAN2 but differ dramatically in their internal representations, and they
are fully equivariant to translation and rotation even at subpixel scales. Our
results pave the way for generative models better suited for video and
animation.

    

### [[2107.00652] CSWin Transformer: A General Vision Transformer Backbone with Cross-Shaped Windows](http://arxiv.org/abs/2107.00652)


  We present CSWin Transformer, an efficient and effective Transformer-based
backbone for general-purpose vision tasks. A challenging issue in Transformer
design is that global self-attention is very expensive to compute whereas local
self-attention often limits the field of interactions of each token. To address
this issue, we develop the Cross-Shaped Window self-attention mechanism for
computing self-attention in the horizontal and vertical stripes in parallel
that form a cross-shaped window, with each stripe obtained by splitting the
input feature into stripes of equal width. We provide a detailed mathematical
analysis of the effect of the stripe width and vary the stripe width for
different layers of the Transformer network which achieves strong modeling
capability while limiting the computation cost. We also introduce
Locally-enhanced Positional Encoding (LePE), which handles the local positional
information better than existing encoding schemes. LePE naturally supports
arbitrary input resolutions, and is thus especially effective and friendly for
downstream tasks. Incorporated with these designs and a hierarchical structure,
CSWin Transformer demonstrates competitive performance on common vision tasks.
Specifically, it achieves 85.4% Top-1 accuracy on ImageNet-1K without any extra
training data or label, 53.9 box AP and 46.4 mask AP on the COCO detection
task, and 51.7 mIOU on the ADE20K semantic segmentation task, surpassing
previous state-of-the-art Swin Transformer backbone by +1.2, +2.0, +1.4, and
+2.0 respectively under the similar FLOPs setting. By further pretraining on
the larger dataset ImageNet-21K, we achieve 87.5% Top-1 accuracy on ImageNet-1K
and state-of-the-art segmentation performance on ADE20K with 55.7 mIoU. The
code and models will be available at
this https URL.

    

### [[2107.00774] Almost Tight Approximation Algorithms for Explainable Clustering](http://arxiv.org/abs/2107.00774)


  Recently, due to an increasing interest for transparency in artificial
intelligence, several methods of explainable machine learning have been
developed with the simultaneous goal of accuracy and interpretability by
humans. In this paper, we study a recent framework of explainable clustering
first suggested by Dasgupta et al.~\cite{dasgupta2020explainable}.
Specifically, we focus on the $k$-means and $k$-medians problems and provide
nearly tight upper and lower bounds.
First, we provide an $O(\log k \log \log k)$-approximation algorithm for
explainable $k$-medians, improving on the best known algorithm of
$O(k)$~\cite{dasgupta2020explainable} and nearly matching the known
$\Omega(\log k)$ lower bound~\cite{dasgupta2020explainable}. In addition, in
low-dimensional spaces $d \ll \log k$, we show that our algorithm also provides
an $O(d \log^2 d)$-approximate solution for explainable $k$-medians. This
improves over the best known bound of $O(d \log k)$ for low
dimensions~\cite{laber2021explainable}, and is a constant for constant
dimensional spaces. To complement this, we show a nearly matching $\Omega(d)$
lower bound. Next, we study the $k$-means problem in this context and provide
an $O(k \log k)$-approximation algorithm for explainable $k$-means, improving
over the $O(k^2)$ bound of Dasgupta et al. and the $O(d k \log k)$ bound of
\cite{laber2021explainable}. To complement this we provide an almost tight
$\Omega(k)$ lower bound, improving over the $\Omega(\log k)$ lower bound of
Dasgupta et al. Given an approximate solution to the classic $k$-means and
$k$-medians, our algorithm for $k$-medians runs in time $O(kd \log^2 k )$ and
our algorithm for $k$-means runs in time $ O(k^2 d)$.

    

### [[2107.00793] The Causal-Neural Connection: Expressiveness, Learnability, and Inference](http://arxiv.org/abs/2107.00793)


  One of the central elements of any causal inference is an object called
structural causal model (SCM), which represents a collection of mechanisms and
exogenous sources of random variation of the system under investigation (Pearl,
2000). An important property of many kinds of neural networks is universal
approximability: the ability to approximate any function to arbitrary
precision. Given this property, one may be tempted to surmise that a collection
of neural nets is capable of learning any SCM by training on data generated by
that SCM. In this paper, we show this is not the case by disentangling the
notions of expressivity and learnability. Specifically, we show that the causal
hierarchy theorem (Thm. 1, Bareinboim et al., 2020), which describes the limits
of what can be learned from data, still holds for neural models. For instance,
an arbitrarily complex and expressive neural net is unable to predict the
effects of interventions given observational data alone. Given this result, we
introduce a special type of SCM called a neural causal model (NCM), and
formalize a new type of inductive bias to encode structural constraints
necessary for performing causal inferences. Building on this new class of
models, we focus on solving two canonical tasks found in the literature known
as causal identification and estimation. Leveraging the neural toolbox, we
develop an algorithm that is both sufficient and necessary to determine whether
a causal effect can be learned from data (i.e., causal identifiability); it
then estimates the effect whenever identifiability holds (causal estimation).
Simulations corroborate the proposed approach.

    

### [[2107.07169] Arrow: A RISC-V Vector Accelerator for Machine Learning Inference](http://arxiv.org/abs/2107.07169)


  In this paper we present Arrow, a configurable hardware accelerator
architecture that implements a subset of the RISC-V v0.9 vector ISA extension
aimed at edge machine learning inference. Our experimental results show that an
Arrow co-processor can execute a suite of vector and matrix benchmarks
fundamental to machine learning inference 2 - 78x faster than a scalar RISC
processor while consuming 20% - 99% less energy when implemented in a Xilinx
XC7A200T-1SBG484C FPGA.

    

### [[2107.06922] A Byzantine Fault-Tolerant Consensus Library for Hyperledger Fabric](http://arxiv.org/abs/2107.06922)


  Hyperledger Fabric is an enterprise grade permissioned distributed ledger
platform that offers modularity for a broad set of industry use cases. One
modular component is a pluggable ordering service that establishes consensus on
the order of transactions and batches them into blocks. However, as of the time
of this writing, there is no production grade Byzantine Fault-Tolerant (BFT)
ordering service for Fabric, with the latest version (v2.1) supporting only
Crash Fault-Tolerance (CFT). In our work, we address crucial aspects of BFT
integration into Fabric that were left unsolved in all prior works, making them
unfit for production use. In this work we describe the design and
implementation of a BFT ordering service for Fabric, employing a new BFT
consensus library. The new library, based on the BFT-Smart protocol and written
in Go, is tailored to the blockchain use-case, yet is general enough to cater
to a wide variety of other uses. We evaluate the new BFT ordering service by
comparing it with the currently supported Raft-based CFT ordering service in
Hyperledger Fabric.

    

### [[2107.06999] Reuse of Semantic Models for Emerging Smart Grids Applications](http://arxiv.org/abs/2107.06999)


  Data in the energy domain grows at unprecedented rates. Despite the great
potential that IoT platforms and other big data-driven technologies have
brought in the energy sector, data exchange and data integration are still not
wholly achieved. As a result, fragmented applications are developed against
energy data silos, and data exchange is limited to few applications. Therefore,
this paper identifies semantic models that can be reused for building
interoperable energy management services and applications. The ambition is to
innovate the Institute Mihajlo Pupin proprietary SCADA system and to enable
integration of the Institute Mihajlo Pupin services and applications in the
European Union (EU) Energy Data Space. The selection of reusable models has
been done based on a set of scenarios related to electricity balancing
services, predictive maintenance services, and services for the residential,
commercial and industrial sectors.

    

### [[2107.07063] BlockJack: Towards Improved Prevention of IP Prefix Hijacking Attacks in Inter-Domain Routing Via Blockchain](http://arxiv.org/abs/2107.07063)


  We propose BlockJack, a system based on a distributed and tamper-proof
consortium Blockchain that aims at blocking IP prefix hijacking in the Border
Gateway Protocol (BGP). In essence, BlockJack provides synchronization among
BlockChain and BGP network through interfaces ensuring operational independence
and this approach preserving the legacy system and accommodates the impact of a
race condition if the Blockchain process exceeds the BGP update interval.
BlockJack is also resilient to dynamic routing path changes during the
occurrence of the IP prefix hijacking in the routing tables. We implement
BlockJack using Hyperledger Fabric Blockchain and Quagga software package and
we perform initial sets of experiments to evaluate its efficacy. We evaluate
the performance and resilience of BlockJack in various attack scenarios
including single path attacks, multiple path attacks, and attacks from random
sources in the random network topology. The Evaluation results show that
BlockJack is able to handle multiple attacks caused by AS paths changes during
a BGP prefix hijacking. In experiment settings with 50 random routers,
BlockJack takes on average 0.08 seconds (with a standard deviation of 0.04
seconds) to block BGP prefix hijacking attacks. The test result showing that
BlockJack conservative approach feasible to handle the IP Prefix hijacking in
the Border Gateway Protocol.

    

### [[2107.07104] Scalable Biophysical Simulations of the Neuromuscular System](http://arxiv.org/abs/2107.07104)


  The human neuromuscular system consisting of skeletal muscles and neural
circuits is a complex system that is not yet fully understood. Surface
electromyography (EMG) can be used to study muscle behavior from the outside.
Computer simulations with detailed biophysical models provide a non-invasive
tool to interpret EMG signals and gain new insights into the system. The
numerical solution of such multi-scale models imposes high computational work
loads, which restricts their application to short simulation time spans or
coarse resolutions. We tackled this challenge by providing scalable software
employing instruction-level and task-level parallelism, suitable numerical
methods and efficient data handling. We implemented a comprehensive,
state-of-the-art, multi-scale multi-physics model framework that can simulate
surface EMG signals and muscle contraction as a result of neuromuscular
stimulation.
This work describes the model framework and its numerical discretization,
develops new algorithms for mesh generation and parallelization, covers the use
and implementation of our software OpenDiHu, and evaluates its computational
performance in numerous use cases. We obtain a speedup of several hundred
compared to a baseline solver from the literature and demonstrate, that our
distributed-memory parallelization and the use of High Performance Computing
resources enables us to simulate muscular surface EMG of the biceps brachii
muscle with realistic muscle fiber counts of several hundred thousands. We find
that certain model effects are only visible with such high resolution. In
conclusion, our software contributes to more realistic simulations of the
neuromuscular system and provides a tool for applied researchers to complement
in vivo experiments with in-silico studies. It can serve as a building block to
set up comprehensive models for more organs in the musculoskeletal system.

    

### [[2107.07108] Improving I/O Performance for Exascale Applications through Online Data Layout Reorganization](http://arxiv.org/abs/2107.07108)


  The applications being developed within the U.S. Exascale Computing Project
(ECP) to run on imminent Exascale computers will generate scientific results
with unprecedented fidelity and record turn-around time. Many of these codes
are based on particle-mesh methods and use advanced algorithms, especially
dynamic load-balancing and mesh-refinement, to achieve high performance on
Exascale machines. Yet, as such algorithms improve parallel application
efficiency, they raise new challenges for I/O logic due to their irregular and
dynamic data distributions. Thus, while the enormous data rates of Exascale
simulations already challenge existing file system write strategies, the need
for efficient read and processing of generated data introduces additional
constraints on the data layout strategies that can be used when writing data to
secondary storage. We review these I/O challenges and introduce two online data
layout reorganization approaches for achieving good tradeoffs between read and
write performance. We demonstrate the benefits of using these two approaches
for the ECP particle-in-cell simulation WarpX, which serves as a motif for a
large class of important Exascale applications. We show that by understanding
application I/O patterns and carefully designing data layouts we can increase
read performance by more than 80%.

    

### [[2107.07157] A64FX -- Your Compiler You Must Decide!](http://arxiv.org/abs/2107.07157)


  The current number one of the TOP500 list, Supercomputer Fugaku, has
demonstrated that CPU-only HPC systems aren't dead and CPUs can be used for
more than just being the host controller for a discrete accelerators. While the
specifications of the chip and overall system architecture, and benchmarks
submitted to various lists, like TOP500 and Green500, etc., are clearly
highlighting the potential, the proliferation of Arm into the HPC business is
rather recent and hence the software stack might not be fully matured and
tuned, yet. We test three state-of-the-art compiler suite against a broad set
of benchmarks. Our measurements show that orders of magnitudes in performance
can be gained by deviating from the recommended usage model of the A64FX
compute nodes.

    

### [[2107.07195] Efficient Resources Distribution for an Ephemeral Cloud/Edge continuum](http://arxiv.org/abs/2107.07195)


  This paper presents the idea and the concepts behind the vision of an
Ephemeral Cloud/Edge Continuum, a cloud/edge computing landscape that enables
the exploitation of a widely distributed, dynamic, and context-aware set of
resources. The Ephemeral Continuum answer to the need of combining a plethora
of heterogeneous devices, which nowadays are pervasively embedding anthropic
environments, with both federations of cloud providers and the resources
located at the Edge. The aim of the Ephemeral Continuum is to realise a
context-aware and personalised federation of computational, data and network
resources, able to manage their heterogeneity in a highly distributed
deployment.

    

### [[2107.07208] Design of Distributed Reconfigurable Robotics Systems with ReconROS](http://arxiv.org/abs/2107.07208)


  Robotics applications process large amounts of data in real-time and require
compute platforms that provide high performance and energy-efficiency. FPGAs
are well-suited for many of these applications, but there is a reluctance in
the robotics community to use hardware acceleration due to increased design
complexity and a lack of consistent programming models across the
software/hardware boundary. In this paper we present ReconROS, a framework that
integrates the widely-used robot operating system (ROS) with ReconOS, which
features multithreaded programming of hardware and software threads for
reconfigurable computers. This unique combination gives ROS2 developers the
flexibility to transparently accelerate parts of their robotics applications in
hardware. We elaborate on the architecture and the design flow for ReconROS and
report on a set of experiments that underline the feasibility and flexibility
of our approach.

    

### [[2107.07365] Szegedy Walk Unitaries for Quantum Maps](http://arxiv.org/abs/2107.07365)


  Szegedy developed a generic method for quantizing classical algorithms based
on random walks [Proceedings of FOCS, 2004, pp. 32-41]. A major contribution of
his work was the construction of a walk unitary for any reversible random walk.
Such unitary posses two crucial properties: its eigenvector with eigenphase $0$
is a quantum sample of the limiting distribution of the random walk and its
eigenphase gap is quadratically larger than the spectral gap of the random
walk. It was an open question if it is possible to generalize Szegedy's
quantization method for stochastic maps to quantum maps. We answer this in the
affirmative by presenting an explicit construction of a Szegedy walk unitary
for detailed balanced Lindbladians -- generators of quantum Markov semigroups
-- and detailed balanced quantum channels. We prove that our Szegedy walk
unitary has a purification of the fixed point of the Lindbladian as eigenvector
with eigenphase $0$ and that its eigenphase gap is quadratically larger than
the spectral gap of the Lindbladian. To construct the walk unitary we leverage
a canonical form for detailed balanced Lindbladians showing that they are
structurally related to Davies generators. We also explain how the quantization
method for Lindbladians can be applied to quantum channels. We give an
efficient quantum algorithm for quantizing Davies generators that describe many
important open-system dynamics, for instance, the relaxation of a quantum
system coupled to a bath. Our algorithm extends known techniques for simulating
quantum systems on a quantum computer.

    

### [[2010.09112] BBB-Voting: 1-out-of-k Blockchain-Based Boardroom Voting](http://arxiv.org/abs/2010.09112)


  Voting is a means to agree on a collective decision based on available
choices (e.g., candidates), where participants (voters) agree to abide by their
outcome. To improve some features of e-voting, decentralized solutions based on
a blockchain can be employed, where the blockchain represents a public bulletin
board that in contrast to a centralized bulletin board provides $100\%$
availability and censorship resistance. A blockchain ensures that all entities
in the voting system have the same view of the actions made by others due to
its immutable and append-only log. The existing blockchain-based boardroom
voting solution called Open Voting Network (OVN) provides the privacy of votes
and perfect ballot secrecy, but it supports only two candidates. We present
BBB-Voting, an equivalent blockchain-based approach for decentralized voting
than OVN, but in contrast to it, BBB-Voting supports 1-out-of-$k$ choices and
provides a fault tolerance mechanism that enables recovery from stalling
participants. We provide a cost-optimized implementation using Ethereum, which
we compare with OVN and show that our work decreases the costs for voters by
$13.5\%$ in terms of gas consumption. Next, we outline the extension of our
implementation scaling to magnitudes higher number of participants than in a
boardroom voting, while preserving the costs paid by the authority and
participants -- we made proof-of-concept experiments with up to 1000
participants.

    

### [[2105.11229] FaaSNet: Scalable and Fast Provisioning of Custom Serverless Container Runtimes at Alibaba Cloud Function Compute](http://arxiv.org/abs/2105.11229)


  Serverless computing, or Function-as-a-Service (FaaS), enables a new way of
building and scaling applications by allowing users to deploy fine-grained
functions while providing fully-managed resource provisioning and auto-scaling.
Custom FaaS container support is gaining traction as it enables better control
over OSes, versioning, and tooling for modernizing FaaS applications. However,
providing rapid container provisioning introduces non-trivial challenges for
FaaS providers, since container provisioning is costly, and real-world FaaS
workloads exhibit highly dynamic patterns. In this paper, we design FaaSNet, a
highly-scalable middleware system for accelerating FaaS container provisioning.
FaaSNet is driven by the workload and infrastructure requirements of the FaaS
platform at one of the world's largest cloud providers, Alibaba Cloud Function
Compute. FaaSNet enables scalable container provisioning via a lightweight,
adaptive function tree (FT) structure. FaaSNet uses an I/O efficient, on-demand
fetching mechanism to further reduce provisioning costs at scale. We implement
and integrate FaaSNet in Alibaba Cloud Function Compute. Evaluation results
show that FaaSNet: (1) finishes provisioning 2500 function containers on 1000
virtual machines in 8.3 seconds, (2) scales 13.4x and 16.3x faster than Alibaba
Cloud's current FaaS platform and a state-of-the-art P2P container registry
(Kraken), respectively, and (3) sustains a bursty workload using 75.2% less
time than an optimized baseline.

    

### [[2107.06916] Training Compact CNNs for Image Classification using Dynamic-coded Filter Fusion](http://arxiv.org/abs/2107.06916)


  The mainstream approach for filter pruning is usually either to force a
hard-coded importance estimation upon a computation-heavy pretrained model to
select "important" filters, or to impose a hyperparameter-sensitive sparse
constraint on the loss objective to regularize the network training. In this
paper, we present a novel filter pruning method, dubbed dynamic-coded filter
fusion (DCFF), to derive compact CNNs in a computation-economical and
regularization-free manner for efficient image classification. Each filter in
our DCFF is firstly given an inter-similarity distribution with a temperature
parameter as a filter proxy, on top of which, a fresh Kullback-Leibler
divergence based dynamic-coded criterion is proposed to evaluate the filter
importance. In contrast to simply keeping high-score filters in other methods,
we propose the concept of filter fusion, i.e., the weighted averages using the
assigned proxies, as our preserved filters. We obtain a one-hot
inter-similarity distribution as the temperature parameter approaches infinity.
Thus, the relative importance of each filter can vary along with the training
of the compact CNN, leading to dynamically changeable fused filters without
both the dependency on the pretrained model and the introduction of sparse
constraints. Extensive experiments on classification benchmarks demonstrate the
superiority of our DCFF over the compared counterparts. For example, our DCFF
derives a compact VGGNet-16 with only 72.77M FLOPs and 1.06M parameters while
reaching top-1 accuracy of 93.47% on CIFAR-10. A compact ResNet-50 is obtained
with 63.8% FLOPs and 58.6% parameter reductions, retaining 75.60% top-1
accuracy on ILSVRC-2012. Our code, narrower models and training logs are
available at this https URL.

    

### [[2107.06941] Mutually improved endoscopic image synthesis and landmark detection in unpaired image-to-image translation](http://arxiv.org/abs/2107.06941)


  The CycleGAN framework allows for unsupervised image-to-image translation of
unpaired data. In a scenario of surgical training on a physical surgical
simulator, this method can be used to transform endoscopic images of phantoms
into images which more closely resemble the intra-operative appearance of the
same surgical target structure. This can be viewed as a novel augmented reality
approach, which we coined Hyperrealism in previous work. In this use case, it
is of paramount importance to display objects like needles, sutures or
instruments consistent in both domains while altering the style to a more
tissue-like appearance. Segmentation of these objects would allow for a direct
transfer, however, contouring of these, partly tiny and thin foreground objects
is cumbersome and perhaps inaccurate. Instead, we propose to use landmark
detection on the points when sutures pass into the tissue. This objective is
directly incorporated into a CycleGAN framework by treating the performance of
pre-trained detector models as an additional optimization goal. We show that a
task defined on these sparse landmark labels improves consistency of synthesis
by the generator network in both domains. Comparing a baseline CycleGAN
architecture to our proposed extension (DetCycleGAN), mean precision (PPV)
improved by +61.32, mean sensitivity (TPR) by +37.91, and mean F1 score by
+0.4743. Furthermore, it could be shown that by dataset fusion, generated
intra-operative images can be leveraged as additional training data for the
detection network itself. The data is released within the scope of the AdaptOR
MICCAI Challenge 2021 at this https URL, and code at
this https URL.

    

### [[2107.07015] Do Humans Trust Advice More if it Comes from AI? An Analysis of Human-AI Interactions](http://arxiv.org/abs/2107.07015)


  In many applications of AI, the algorithm's output is framed as a suggestion
to a human user. The user may ignore the advice or take it into consideration
to modify his/her decisions. With the increasing prevalence of such human-AI
interactions, it is important to understand how users act (or do not act) upon
AI advice, and how users regard advice differently if they believe the advice
come from an "AI" versus another human. In this paper, we characterize how
humans use AI suggestions relative to equivalent suggestions from a group of
peer humans across several experimental settings. We find that participants'
beliefs about the human versus AI performance on a given task affects whether
or not they heed the advice. When participants decide to use the advice, they
do so similarly for human and AI suggestions. These results provide insights
into factors that affect human-AI interactions.

    

### [[2107.07016] Forgetting in Answer Set Programming -- A Survey](http://arxiv.org/abs/2107.07016)


  Forgetting - or variable elimination - is an operation that allows the
removal, from a knowledge base, of middle variables no longer deemed relevant.
In recent years, many different approaches for forgetting in Answer Set
Programming have been proposed, in the form of specific operators, or classes
of such operators, commonly following different principles and obeying
different properties. Each such approach was developed to somehow address some
particular view on forgetting, aimed at obeying a specific set of properties
deemed desirable in such view, but a comprehensive and uniform overview of all
the existing operators and properties is missing. In this paper, we thoroughly
examine existing properties and (classes of) operators for forgetting in Answer
Set Programming, drawing a complete picture of the landscape of these classes
of forgetting operators, which includes many novel results on relations between
properties and operators, including considerations on concrete operators to
compute results of forgetting and computational complexity. Our goal is to
provide guidance to help users in choosing the operator most adequate for their
application requirements.

    

### [[2107.07031] Experimental Evidence that Empowerment May Drive Exploration in Sparse-Reward Environments](http://arxiv.org/abs/2107.07031)


  Reinforcement Learning (RL) is known to be often unsuccessful in environments
with sparse extrinsic rewards. A possible countermeasure is to endow RL agents
with an intrinsic reward function, or 'intrinsic motivation', which rewards the
agent based on certain features of the current sensor state. An intrinsic
reward function based on the principle of empowerment assigns rewards
proportional to the amount of control the agent has over its own sensors. We
implemented a variation on a recently proposed intrinsically motivated agent,
which we refer to as the 'curious' agent, and an empowerment-inspired agent.
The former leverages sensor state encoding with a variational autoencoder,
while the latter predicts the next sensor state via a variational information
bottleneck. We compared the performance of both agents to that of an advantage
actor-critic baseline in four sparse reward grid worlds. Both the empowerment
agent and its curious competitor seem to benefit to similar extents from their
intrinsic rewards. This provides some experimental support to the conjecture
that empowerment can be used to drive exploration.

    

### [[2107.07066] Deep Reinforcement Learning based Dynamic Optimization of Bus Timetable](http://arxiv.org/abs/2107.07066)


  Bus timetable optimization is a key issue to reduce operational cost of bus
companies and improve the service quality. Existing methods use exact or
heuristic algorithms to optimize the timetable in an offline manner. In
practice, the passenger flow may change significantly over time. Timetables
determined in offline cannot adjust the departure interval to satisfy the
changed passenger flow. Aiming at improving the online performance of bus
timetable, we propose a Deep Reinforcement Learning based bus Timetable dynamic
Optimization method (DRL-TO). In this method, the timetable optimization is
considered as a sequential decision problem. A Deep Q-Network (DQN) is employed
as the decision model to determine whether to dispatch a bus service during
each minute of the service period. Therefore, the departure intervals of bus
services are determined in real time in accordance with passenger demand. We
identify several new and useful state features for the DQN, including the load
factor, carrying capacity utilization rate, and the number of stranding
passengers. Taking into account both the interests of the bus company and
passengers, a reward function is designed, which includes the indicators of
full load rate, empty load rate, passengers' waiting time, and the number of
stranding passengers. Building on an existing method for calculating the
carrying capacity, we develop a new technique to enhance the matching degree at
each bus station. Experiments demonstrate that compared with the timetable
generated by the state-of-the-art bus timetable optimization approach based on
a memetic algorithm (BTOA-MA), Genetic Algorithm (GA) and the manual method,
DRL-TO can dynamically determine the departure intervals based on the real-time
passenger flow, saving 8$\%$ of vehicles and reducing 17$\%$ of passengers'
waiting time on average.

    

### [[2107.07089] STAR: Sparse Transformer-based Action Recognition](http://arxiv.org/abs/2107.07089)


  The cognitive system for human action and behavior has evolved into a deep
learning regime, and especially the advent of Graph Convolution Networks has
transformed the field in recent years. However, previous works have mainly
focused on over-parameterized and complex models based on dense graph
convolution networks, resulting in low efficiency in training and inference.
Meanwhile, the Transformer architecture-based model has not yet been well
explored for cognitive application in human action and behavior estimation.
This work proposes a novel skeleton-based human action recognition model with
sparse attention on the spatial dimension and segmented linear attention on the
temporal dimension of data. Our model can also process the variable length of
video clips grouped as a single batch. Experiments show that our model can
achieve comparable performance while utilizing much less trainable parameters
and achieve high speed in training and inference. Experiments show that our
model achieves 4~18x speedup and 1/7~1/15 model size compared with the baseline
models at competitive accuracy.

    

### [[2107.07095] Applying the Case Difference Heuristic to Learn Adaptations from Deep Network Features](http://arxiv.org/abs/2107.07095)


  The case difference heuristic (CDH) approach is a knowledge-light method for
learning case adaptation knowledge from the case base of a case-based reasoning
system. Given a pair of cases, the CDH approach attributes the difference in
their solutions to the difference in the problems they solve, and generates
adaptation rules to adjust solutions accordingly when a retrieved case and new
query have similar problem differences. As an alternative to learning
adaptation rules, several researchers have applied neural networks to learn to
predict solution differences from problem differences. Previous work on such
approaches has assumed that the feature set describing problems is predefined.
This paper investigates a two-phase process combining deep learning for feature
extraction and neural network based adaptation learning from extracted
features. Its performance is demonstrated in a regression task on an image
data: predicting age given the image of a face. Results show that the combined
process can successfully learn adaptation knowledge applicable to nonsymbolic
differences in cases. The CBR system achieves slightly lower performance
overall than a baseline deep network regressor, but better performance than the
baseline on novel queries.

    

### [[2107.07112] Neural Code Summarization: How Far Are We?](http://arxiv.org/abs/2107.07112)


  Source code summaries are important for the comprehension and maintenance of
programs. However, there are plenty of programs with missing, outdated, or
mismatched summaries. Recently, deep learning techniques have been exploited to
automatically generate summaries for given code snippets. To achieve a profound
understanding of how far we are from solving this problem, in this paper, we
conduct a systematic and in-depth analysis of five state-of-the-art neural
source code summarization models on three widely used datasets. Our evaluation
results suggest that: (1) The BLEU metric, which is widely used by existing
work for evaluating the performance of the summarization models, has many
variants. Ignoring the differences among the BLEU variants could affect the
validity of the claimed results. Furthermore, we discover an important,
previously unknown bug about BLEU calculation in a commonly-used software
package. (2) Code pre-processing choices can have a large impact on the
summarization performance, therefore they should not be ignored. (3) Some
important characteristics of datasets (corpus size, data splitting method, and
duplication ratio) have a significant impact on model evaluation. Based on the
experimental results, we give some actionable guidelines on more systematic
ways for evaluating code summarization and choosing the best method in
different scenarios. We also suggest possible future research directions. We
believe that our results can be of great help for practitioners and researchers
in this interesting area.

    

### [[2107.07113] Robust Learning for Text Classification with Multi-source Noise Simulation and Hard Example Mining](http://arxiv.org/abs/2107.07113)


  Many real-world applications involve the use of Optical Character Recognition
(OCR) engines to transform handwritten images into transcripts on which
downstream Natural Language Processing (NLP) models are applied. In this
process, OCR engines may introduce errors and inputs to downstream NLP models
become noisy. Despite that pre-trained models achieve state-of-the-art
performance in many NLP benchmarks, we prove that they are not robust to noisy
texts generated by real OCR engines. This greatly limits the application of NLP
models in real-world scenarios. In order to improve model performance on noisy
OCR transcripts, it is natural to train the NLP model on labelled noisy texts.
However, in most cases there are only labelled clean texts. Since there is no
handwritten pictures corresponding to the text, it is impossible to directly
use the recognition model to obtain noisy labelled data. Human resources can be
employed to copy texts and take pictures, but it is extremely expensive
considering the size of data for model training. Consequently, we are
interested in making NLP models intrinsically robust to OCR errors in a low
resource manner. We propose a novel robust training framework which 1) employs
simple but effective methods to directly simulate natural OCR noises from clean
texts and 2) iteratively mines the hard examples from a large number of
simulated samples for optimal performance. 3) To make our model learn
noise-invariant representations, a stability loss is employed. Experiments on
three real-world datasets show that the proposed framework boosts the
robustness of pre-trained models by a large margin. We believe that this work
can greatly promote the application of NLP models in actual scenarios, although
the algorithm we use is simple and straightforward. We make our codes and three
datasets publicly
available\footnote{this https URL}.

    

### [[2107.07114] Uncertainty-Aware Reliable Text Classification](http://arxiv.org/abs/2107.07114)


  Deep neural networks have significantly contributed to the success in
predictive accuracy for classification tasks. However, they tend to make
over-confident predictions in real-world settings, where domain shifting and
out-of-distribution (OOD) examples exist. Most research on uncertainty
estimation focuses on computer vision because it provides visual validation on
uncertainty quality. However, few have been presented in the natural language
process domain. Unlike Bayesian methods that indirectly infer uncertainty
through weight uncertainties, current evidential uncertainty-based methods
explicitly model the uncertainty of class probabilities through subjective
opinions. They further consider inherent uncertainty in data with different
root causes, vacuity (i.e., uncertainty due to a lack of evidence) and
dissonance (i.e., uncertainty due to conflicting evidence). In our paper, we
firstly apply evidential uncertainty in OOD detection for text classification
tasks. We propose an inexpensive framework that adopts both auxiliary outliers
and pseudo off-manifold samples to train the model with prior knowledge of a
certain class, which has high vacuity for OOD samples. Extensive empirical
experiments demonstrate that our model based on evidential uncertainty
outperforms other counterparts for detecting OOD examples. Our approach can be
easily deployed to traditional recurrent neural networks and fine-tuned
pre-trained transformers.

    

### [[2107.07119] Multi-Task Learning based Online Dialogic Instruction Detection with Pre-trained Language Models](http://arxiv.org/abs/2107.07119)


  In this work, we study computational approaches to detect online dialogic
instructions, which are widely used to help students understand learning
materials, and build effective study habits. This task is rather challenging
due to the widely-varying quality and pedagogical styles of dialogic
instructions. To address these challenges, we utilize pre-trained language
models, and propose a multi-task paradigm which enhances the ability to
distinguish instances of different classes by enlarging the margin between
categories via contrastive loss. Furthermore, we design a strategy to fully
exploit the misclassified examples during the training stage. Extensive
experiments on a real-world online educational data set demonstrate that our
approach achieves superior performance compared to representative baselines. To
encourage reproducible results, we make our implementation online available at
\url{this https URL}.

    

### [[2107.07122] Solving ESL Sentence Completion Questions via Pre-trained Neural Language Models](http://arxiv.org/abs/2107.07122)


  Sentence completion (SC) questions present a sentence with one or more blanks
that need to be filled in, three to five possible words or phrases as options.
SC questions are widely used for students learning English as a Second Language
(ESL) and building computational approaches to automatically solve such
questions is beneficial to language learners. In this work, we propose a neural
framework to solve SC questions in English examinations by utilizing
pre-trained language models. We conduct extensive experiments on a real-world
K-12 ESL SC question dataset and the results demonstrate the superiority of our
model in terms of prediction accuracy. Furthermore, we run precision-recall
trade-off analysis to discuss the practical issues when deploying it in
real-life scenarios. To encourage reproducible results, we make our code
publicly available at \url{this https URL}.

    

### [[2107.07124] An Educational System for Personalized Teacher Recommendation in K-12 Online Classrooms](http://arxiv.org/abs/2107.07124)


  In this paper, we propose a simple yet effective solution to build practical
teacher recommender systems for online one-on-one classes. Our system consists
of (1) a pseudo matching score module that provides reliable training labels;
(2) a ranking model that scores every candidate teacher; (3) a novelty boosting
module that gives additional opportunities to new teachers; and (4) a diversity
metric that guardrails the recommended results to reduce the chance of
collision. Offline experimental results show that our approach outperforms a
wide range of baselines. Furthermore, we show that our approach is able to
reduce the number of student-teacher matching attempts from 7.22 to 3.09 in a
five-month observation on a third-party online education platform.

    

### [[2107.07136] Learning Mixed-Integer Linear Programs from Contextual Examples](http://arxiv.org/abs/2107.07136)


  Mixed-integer linear programs (MILPs) are widely used in artificial
intelligence and operations research to model complex decision problems like
scheduling and routing. Designing such programs however requires both domain
and modelling expertise. In this paper, we study the problem of acquiring MILPs
from contextual examples, a novel and realistic setting in which examples
capture solutions and non-solutions within a specific context. The resulting
learning problem involves acquiring continuous parameters -- namely, a cost
vector and a feasibility polytope -- but has a distinctly combinatorial flavor.
To solve this complex problem, we also contribute MISSLE, an algorithm for
learning MILPs from contextual examples. MISSLE uses a variant of stochastic
local search that is guided by the gradient of a continuous surrogate loss
function. Our empirical evaluation on synthetic data shows that MISSLE acquires
better MILPs faster than alternatives based on stochastic local search and
gradient descent.

    

### [[2107.07154] What and When to Look?: Temporal Span Proposal Network for Video Visual Relation Detection](http://arxiv.org/abs/2107.07154)


  Identifying relations between objects is central to understanding the scene.
While several works have been proposed for relation modeling in the image
domain, there have been many constraints in the video domain due to challenging
dynamics of spatio-temporal interactions (e.g., Between which objects are there
an interaction? When do relations occur and end?). To date, two representative
methods have been proposed to tackle Video Visual Relation Detection (VidVRD):
segment-based and window-based. We first point out the limitations these two
methods have and propose Temporal Span Proposal Network (TSPN), a novel method
with two advantages in terms of efficiency and effectiveness. 1) TSPN tells
what to look: it sparsifies relation search space by scoring relationness
(i.e., confidence score for the existence of a relation between pair of
objects) of object pair. 2) TSPN tells when to look: it leverages the full
video context to simultaneously predict the temporal span and categories of the
entire relations. TSPN demonstrates its effectiveness by achieving new
state-of-the-art by a significant margin on two VidVRD benchmarks
(ImageNet-VidVDR and VidOR) while also showing lower time complexity than
existing methods - in particular, twice as efficient as a popular segment-based
approach.

    

### [[2107.07173] Scene-adaptive Knowledge Distillation for Sequential Recommendation via Differentiable Architecture Search](http://arxiv.org/abs/2107.07173)


  Sequential recommender systems (SRS) have become a research hotspot due to
its power in modeling user dynamic interests and sequential behavioral
patterns. To maximize model expressive ability, a default choice is to apply a
larger and deeper network architecture, which, however, often brings high
network latency when generating online recommendations. Naturally, we argue
that compressing the heavy recommendation models into middle- or light- weight
neural networks is of great importance for practical production systems. To
realize such a goal, we propose AdaRec, a knowledge distillation (KD) framework
which compresses knowledge of a teacher model into a student model adaptively
according to its recommendation scene by using differentiable Neural
Architecture Search (NAS). Specifically, we introduce a target-oriented
distillation loss to guide the structure search process for finding the student
network architecture, and a cost-sensitive loss as constraints for model size,
which achieves a superior trade-off between recommendation effectiveness and
efficiency. In addition, we leverage Earth Mover's Distance (EMD) to realize
many-to-many layer mapping during knowledge distillation, which enables each
intermediate student layer to learn from other intermediate teacher layers
adaptively. Extensive experiments on real-world recommendation datasets
demonstrate that our model achieves competitive or better accuracy with notable
inference speedup comparing to strong counterparts, while discovering diverse
neural architectures for sequential recommender models under different
recommendation scenes.

    

### [[2107.07191] Deep Learning based Food Instance Segmentation using Synthetic Data](http://arxiv.org/abs/2107.07191)


  In the process of intelligently segmenting foods in images using deep neural
networks for diet management, data collection and labeling for network training
are very important but labor-intensive tasks. In order to solve the
difficulties of data collection and annotations, this paper proposes a food
segmentation method applicable to real-world through synthetic data. To perform
food segmentation on healthcare robot systems, such as meal assistance robot
arm, we generate synthetic data using the open-source 3D graphics software
Blender placing multiple objects on meal plate and train Mask R-CNN for
instance segmentation. Also, we build a data collection system and verify our
segmentation model on real-world food data. As a result, on our real-world
dataset, the model trained only synthetic data is available to segment food
instances that are not trained with 52.2% mask AP@all, and improve performance
by +6.4%p after fine-tuning comparing to the model trained from scratch. In
addition, we also confirm the possibility and performance improvement on the
public dataset for fair analysis. Our code and pre-trained weights are
avaliable online at: this https URL


### [[2107.07229] Trusting RoBERTa over BERT: Insights from CheckListing the Natural Language Inference Task](http://arxiv.org/abs/2107.07229)


  The recent state-of-the-art natural language understanding (NLU) systems
often behave unpredictably, failing on simpler reasoning examples. Despite
this, there has been limited focus on quantifying progress towards systems with
more predictable behavior. We think that reasoning capability-wise behavioral
summary is a step towards bridging this gap. We create a CheckList test-suite
(184K examples) for the Natural Language Inference (NLI) task, a representative
NLU task. We benchmark state-of-the-art NLI systems on this test-suite, which
reveals fine-grained insights into the reasoning abilities of BERT and RoBERTa.
Our analysis further reveals inconsistencies of the models on examples derived
from the same template or distinct templates but pertaining to same reasoning
capability, indicating that generalizing the models' behavior through
observations made on a CheckList is non-trivial. Through an user-study, we find
that users were able to utilize behavioral information to generalize much
better for examples predicted from RoBERTa, compared to that of BERT.

    

### [[2107.07233] Genetic CFL: Optimization of Hyper-Parameters in Clustered Federated Learning](http://arxiv.org/abs/2107.07233)


  Federated learning (FL) is a distributed model for deep learning that
integrates client-server architecture, edge computing, and real-time
intelligence. FL has the capability of revolutionizing machine learning (ML)
but lacks in the practicality of implementation due to technological
limitations, communication overhead, non-IID (independent and identically
distributed) data, and privacy concerns. Training a ML model over heterogeneous
non-IID data highly degrades the convergence rate and performance. The existing
traditional and clustered FL algorithms exhibit two main limitations, including
inefficient client training and static hyper-parameter utilization. To overcome
these limitations, we propose a novel hybrid algorithm, namely genetic
clustered FL (Genetic CFL), that clusters edge devices based on the training
hyper-parameters and genetically modifies the parameters cluster-wise. Then, we
introduce an algorithm that drastically increases the individual cluster
accuracy by integrating the density-based clustering and genetic
hyper-parameter optimization. The results are bench-marked using MNIST
handwritten digit dataset and the CIFAR-10 dataset. The proposed genetic CFL
shows significant improvements and works well with realistic cases of non-IID
and ambiguous data.

    

### [[2107.07235] Deep Automatic Natural Image Matting](http://arxiv.org/abs/2107.07235)


  Automatic image matting (AIM) refers to estimating the soft foreground from
an arbitrary natural image without any auxiliary input like trimap, which is
useful for image editing. Prior methods try to learn semantic features to aid
the matting process while being limited to images with salient opaque
foregrounds such as humans and animals. In this paper, we investigate the
difficulties when extending them to natural images with salient
transparent/meticulous foregrounds or non-salient foregrounds. To address the
problem, a novel end-to-end matting network is proposed, which can predict a
generalized trimap for any image of the above types as a unified semantic
representation. Simultaneously, the learned semantic features guide the matting
network to focus on the transition areas via an attention mechanism. We also
construct a test set AIM-500 that contains 500 diverse natural images covering
all types along with manually labeled alpha mattes, making it feasible to
benchmark the generalization ability of AIM models. Results of the experiments
demonstrate that our network trained on available composite matting datasets
outperforms existing methods both objectively and subjectively. The source code
and dataset are available at this https URL.

    

### [[2107.07253] Spanish Language Models](http://arxiv.org/abs/2107.07253)


  This paper presents the Spanish RoBERTa-base and RoBERTa-large models, as
well as the corresponding performance evaluations. Both models were pre-trained
using the largest Spanish corpus known to date, with a total of 570GB of clean
and deduplicated text processed for this work, compiled from the web crawlings
performed by the National Library of Spain from 2009 to 2019.

    

### [[2107.07316] Minimizing Safety Interference for Safe and Comfortable Automated Driving with Distributional Reinforcement Learning](http://arxiv.org/abs/2107.07316)


  Despite recent advances in reinforcement learning (RL), its application in
safety critical domains like autonomous vehicles is still challenging. Although
punishing RL agents for risky situations can help to learn safe policies, it
may also lead to highly conservative behavior. In this paper, we propose a
distributional RL framework in order to learn adaptive policies that can tune
their level of conservativity at run-time based on the desired comfort and
utility. Using a proactive safety verification approach, the proposed framework
can guarantee that actions generated from RL are fail-safe according to the
worst-case assumptions. Concurrently, the policy is encouraged to minimize
safety interference and generate more comfortable behavior. We trained and
evaluated the proposed approach and baseline policies using a high level
simulator with a variety of randomized scenarios including several corner cases
which rarely happen in reality but are very crucial. In light of our
experiments, the behavior of policies learned using distributional RL can be
adaptive at run-time and robust to the environment uncertainty. Quantitatively,
the learned distributional RL agent drives in average 8 seconds faster than the
normal DQN policy and requires 83\% less safety interference compared to the
rule-based policy with slightly increasing the average crossing time. We also
study sensitivity of the learned policy in environments with higher perception
noise and show that our algorithm learns policies that can still drive reliable
when the perception noise is two times higher than the training configuration
for automated merging and crossing at occluded intersections.

    

### [[2107.07335] Towards Natural Brain-Machine Interaction using Endogenous Potentials based on Deep Neural Networks](http://arxiv.org/abs/2107.07335)


  Human-robot collaboration has the potential to maximize the efficiency of the
operation of autonomous robots. Brain-machine interface (BMI) would be a
desirable technology to collaborate with robots since the intention or state of
users can be translated from the neural activities. However, the
electroencephalogram (EEG), which is one of the most popularly used
non-invasive BMI modalities, has low accuracy and a limited degree of freedom
(DoF) due to a low signal-to-noise ratio. Thus, improving the performance of
multi-class EEG classification is crucial to develop more flexible BMI-based
human-robot collaboration. In this study, we investigated the possibility for
inter-paradigm classification of multiple endogenous BMI paradigms, such as
motor imagery (MI), visual imagery (VI), and speech imagery (SI), to enhance
the limited DoF while maintaining robust accuracy. We conducted the statistical
and neurophysiological analyses on MI, VI, and SI and classified three
paradigms using the proposed temporal information-based neural network (TINN).
We confirmed that statistically significant features could be extracted on
different brain regions when classifying three endogenous paradigms. Moreover,
our proposed TINN showed the highest accuracy of 0.93 compared to the previous
methods for classifying three different types of mental imagery tasks (MI, VI,
and SI).

    

### [[2107.07356] DiRe Committee : Diversity and Representation Constraints in Multiwinner Elections](http://arxiv.org/abs/2107.07356)


  The study of fairness in multiwinner elections focuses on settings where
candidates have attributes. However, voters may also be divided into predefined
populations under one or more attributes (e.g., "California" and "Illinois"
populations under the "state" attribute), which may be same or different from
candidate attributes. The models that focus on candidate attributes alone may
systematically under-represent smaller voter populations. Hence, we develop a
model, DiRe Committee Winner Determination (DRCWD), which delineates candidate
and voter attributes to select a committee by specifying diversity and
representation constraints and a voting rule. We show the generalizability of
our model, and analyze its computational complexity, inapproximability, and
parameterized complexity. We develop a heuristic-based algorithm, which finds
the winning DiRe committee in under two minutes on 63% of the instances of
synthetic datasets and on 100% of instances of real-world datasets. We present
an empirical analysis of the running time, feasibility, and utility traded-off.
Overall, DRCWD motivates that a study of multiwinner elections should
consider both its actors, namely candidates and voters, as candidate-specific
"fair" models can unknowingly harm voter populations, and vice versa.
Additionally, even when the attributes of candidates and voters coincide, it is
important to treat them separately as having a female candidate on the
committee, for example, is different from having a candidate on the committee
who is preferred by the female voters, and who themselves may or may not be
female.

    

### [[2107.07404] Two-Sided Matching Meets Fair Division](http://arxiv.org/abs/2107.07404)


  We introduce a new model for two-sided matching which allows us to borrow
popular fairness notions from the fair division literature such as
envy-freeness up to one good and maximin share guarantee. In our model, each
agent is matched to multiple agents on the other side over whom she has
additive preferences. We demand fairness for each side separately, giving rise
to notions such as double envy-freeness up to one match (DEF1) and double
maximin share guarantee (DMMS). We show that (a slight strengthening of) DEF1
cannot always be achieved, but in the special case where both sides have
identical preferences, the round-robin algorithm with a carefully designed
agent ordering achieves it. In contrast, DMMS cannot be achieved even when both
sides have identical preferences.

    

### [[2107.07413] High-level Decisions from a Safe Maneuver Catalog with Reinforcement Learning for Safe and Cooperative Automated Merging](http://arxiv.org/abs/2107.07413)


  Reinforcement learning (RL) has recently been used for solving challenging
decision-making problems in the context of automated driving. However, one of
the main drawbacks of the presented RL-based policies is the lack of safety
guarantees, since they strive to reduce the expected number of collisions but
still tolerate them. In this paper, we propose an efficient RL-based
decision-making pipeline for safe and cooperative automated driving in merging
scenarios. The RL agent is able to predict the current situation and provide
high-level decisions, specifying the operation mode of the low level planner
which is responsible for safety. In order to learn a more generic policy, we
propose a scalable RL architecture for the merging scenario that is not
sensitive to changes in the environment configurations. According to our
experiments, the proposed RL agent can efficiently identify cooperative drivers
from their vehicle state history and generate interactive maneuvers, resulting
in faster and more comfortable automated driving. At the same time, thanks to
the safety constraints inside the planner, all of the maneuvers are collision
free and safe.

    

### [[2107.07425] Multiclass Permanent Magnets Superstructure for Indoor Localization using Artificial Intelligence](http://arxiv.org/abs/2107.07425)


  Smartphones have become a popular tool for indoor localization and position
estimation of users. Existing solutions mainly employ Wi-Fi, RFID, and magnetic
sensing techniques to track movements in crowded venues. These are highly
sensitive to magnetic clutters and depend on local ambient magnetic fields,
which frequently degrades their performance. Also, these techniques often
require pre-known mapping surveys of the area, or the presence of active
beacons, which are not always available. We embed small-volume and large-moment
magnets in pre-known locations and arrange them in specific geometric
constellations that create magnetic superstructure patterns of supervised
magnetic signatures. These signatures constitute an unambiguous magnetic
environment with respect to the moving sensor carrier. The localization
algorithm learns the unique patterns of the scattered magnets during training
and detects them from the ongoing streaming of data during localization. Our
contribution is twofold. First, we deploy passive permanent magnets that do not
require a power supply, in contrast to active magnetic transmitters. Second, we
perform localization based on smartphone motion rather than on static
positioning of the magnetometer. In our previous study, we considered a single
superstructure pattern. Here, we present an extended version of that algorithm
for multi-superstructure localization, which covers a broader localization area
of the user. Experimental results demonstrate localization accuracy of 95% with
a mean localization error of less than 1m using artificial intelligence.

    

### [[2107.07452] GI-NNet \& RGI-NNet: Development of Robotic Grasp Pose Models, Trainable with Large as well as Limited Labelled Training Datasets, under supervised and semi supervised paradigms](http://arxiv.org/abs/2107.07452)


  Our way of grasping objects is challenging for efficient, intelligent and
optimal grasp by COBOTs. To streamline the process, here we use deep learning
techniques to help robots learn to generate and execute appropriate grasps
quickly. We developed a Generative Inception Neural Network (GI-NNet) model,
capable of generating antipodal robotic grasps on seen as well as unseen
objects. It is trained on Cornell Grasping Dataset (CGD) and attained 98.87%
grasp pose accuracy for detecting both regular and irregular shaped objects
from RGB-Depth (RGB-D) images while requiring only one third of the network
trainable parameters as compared to the existing approaches. However, to attain
this level of performance the model requires the entire 90% of the available
labelled data of CGD keeping only 10% labelled data for testing which makes it
vulnerable to poor generalization. Furthermore, getting sufficient and quality
labelled dataset is becoming increasingly difficult keeping in pace with the
requirement of gigantic networks. To address these issues, we attach our model
as a decoder with a semi-supervised learning based architecture known as Vector
Quantized Variational Auto Encoder (VQVAE), which works efficiently when
trained both with the available labelled and unlabelled data. The proposed
model, which we name as Representation based GI-NNet (RGI-NNet), has been
trained with various splits of label data on CGD with as minimum as 10%
labelled dataset together with latent embedding generated from VQVAE up to 50%
labelled data with latent embedding obtained from VQVAE. The performance level,
in terms of grasp pose accuracy of RGI-NNet, varies between 92.13% to 95.6%
which is far better than several existing models trained with only labelled
dataset. For the performance verification of both GI-NNet and RGI-NNet models,
we use Anukul (Baxter) hardware cobot.

    

### [[2107.07498] FewCLUE: A Chinese Few-shot Learning Evaluation Benchmark](http://arxiv.org/abs/2107.07498)


  Pretrained Language Models (PLMs) have achieved tremendous success in natural
language understanding tasks. While different learning schemes -- fine-tuning,
zero-shot and few-shot learning -- have been widely explored and compared for
languages such as English, there is comparatively little work in Chinese to
fairly and comprehensively evaluate and compare these methods. This work first
introduces Chinese Few-shot Learning Evaluation Benchmark (FewCLUE), the first
comprehensive small sample evaluation benchmark in Chinese. It includes nine
tasks, ranging from single-sentence and sentence-pair classification tasks to
machine reading comprehension tasks. Given the high variance of the few-shot
learning performance, we provide multiple training/validation sets to
facilitate a more accurate and stable evaluation of few-shot modeling. An
unlabeled training set with up to 20,000 additional samples per task is
provided, allowing researchers to explore better ways of using unlabeled
samples. Next, we implement a set of state-of-the-art (SOTA) few-shot learning
methods (including PET, ADAPET, LM-BFF, P-tuning and EFL), and compare their
performance with fine-tuning and zero-shot learning schemes on the newly
constructed FewCLUE benchmark.Our results show that: 1) all five few-shot
learning methods exhibit better performance than fine-tuning or zero-shot
learning; 2) among the five methods, PET is the best performing few-shot
method; 3) few-shot learning performance is highly dependent on the specific
task. Our benchmark and code are available at
this https URL


### [[2107.07501] An End-to-End Differentiable Framework for Contact-Aware Robot Design](http://arxiv.org/abs/2107.07501)


  The current dominant paradigm for robotic manipulation involves two separate
stages: manipulator design and control. Because the robot's morphology and how
it can be controlled are intimately linked, joint optimization of design and
control can significantly improve performance. Existing methods for
co-optimization are limited and fail to explore a rich space of designs. The
primary reason is the trade-off between the complexity of designs that is
necessary for contact-rich tasks against the practical constraints of
manufacturing, optimization, contact handling, etc. We overcome several of
these challenges by building an end-to-end differentiable framework for
contact-aware robot design. The two key components of this framework are: a
novel deformation-based parameterization that allows for the design of
articulated rigid robots with arbitrary, complex geometry, and a differentiable
rigid body simulator that can handle contact-rich scenarios and computes
analytical gradients for a full spectrum of kinematic and dynamic parameters.
On multiple manipulation tasks, our framework outperforms existing methods that
either only optimize for control or for design using alternate representations
or co-optimize using gradient-free methods.

    

### [[2001.09461] The SPECIAL-K Personal Data Processing Transparency and Compliance Platform](http://arxiv.org/abs/2001.09461)


  The European General Data Protection Regulation (GDPR) brings new challenges
for companies who must ensure they have an appropriate legal basis for
processing personal data and must provide transparency with respect to personal
data processing and sharing within and between organisations. Additionally,
when it comes to consent as a legal basis, companies need to ensure that they
comply with usage constraints specified by data subjects. This paper presents
the policy language and supporting ontologies and vocabularies, developed
within the SPECIAL EU H2020 project, which can be used to represent data usage
policies and data processing and sharing events. We introduce a concrete
transparency and compliance architecture, referred to as SPECIAL-K, that can be
used to automatically verify that data processing and sharing complies with the
data subjects consent. Our evaluation, based on a new compliance benchmark,
shows the efficiency and scalability of the system with increasing number of
events and users.

    

### [[2104.07395] Robust Backdoor Attacks against Deep Neural Networks in Real Physical World](http://arxiv.org/abs/2104.07395)


  Deep neural networks (DNN) have been widely deployed in various applications.
However, many researches indicated that DNN is vulnerable to backdoor attacks.
The attacker can create a hidden backdoor in target DNN model, and trigger the
malicious behaviors by submitting specific backdoor instance. However, almost
all the existing backdoor works focused on the digital domain, while few
studies investigate the backdoor attacks in real physical world. Restricted to
a variety of physical constraints, the performance of backdoor attacks in the
real physical world will be severely degraded. In this paper, we propose a
robust physical backdoor attack method, PTB (physical transformations for
backdoors), to implement the backdoor attacks against deep learning models in
the real physical world. Specifically, in the training phase, we perform a
series of physical transformations on these injected backdoor instances at each
round of model training, so as to simulate various transformations that a
backdoor may experience in real world, thus improves its physical robustness.
Experimental results on the state-of-the-art face recognition model show that,
compared with the backdoor methods that without PTB, the proposed attack method
can significantly improve the performance of backdoor attacks in real physical
world. Under various complex physical conditions, by injecting only a very
small ratio (0.5%) of backdoor instances, the attack success rate of physical
backdoor attacks with the PTB method on VGGFace is 82%, while the attack
success rate of backdoor attacks without the proposed PTB method is lower than
11%. Meanwhile, the normal performance of the target DNN model has not been
affected.

    

### [[2107.07296] The Art of the Meta Stream Protocol: Torrents of Streams](http://arxiv.org/abs/2107.07296)


  The rise of streaming libraries such as Akka Stream, Reactive Extensions, and
LINQ popularized the declarative functional style of data processing. The
stream paradigm offers concise syntax to write down processing pipelines to
consume the vast amounts of real-time data available today. These libraries
offer the programmer a domain specific language (DSL) embedded in the host
language to describe data streams. These libraries however, all suffer from
extensibility issues. The semantics of a stream is hard-coded into the DSL
language and cannot be changed by the user of the library. We introduce an
approach to modify the semantics of a streaming library by means of
meta-programming at both run-time and compile-time, and showcase its
generality. We show that the expressiveness of the meta-facilities is strong
enough to enable push and pull semantics, error handling, parallelism, and
operator fusion. We evaluate our work by implementing the identified
shortcomings in terms of a novel stream meta-architecture and show that its
design and architecture adhere to the design principles of a meta-level
architecture. The state of the art offers plenty of choice to programmers
regarding reactive stream processing libraries. Expressing reactive systems is
otherwise difficult to do in general purpose languages. Extensibility and
fine-tuning should be possible in these libraries to ensure a broad variety of
applications can be expressed within this single DSL.

    

### [[2107.07298] An Optimised Flow for Futures: From Theory to Practice](http://arxiv.org/abs/2107.07298)


  A future is an entity representing the result of an ongoing computation. A
synchronisation with a "get" operation blocks the caller until the computation
is over, to return the corresponding value. When a computation in charge of
fulfilling a future delegates part of its processing to another task,
mainstream languages return nested futures, and several "get" operations are
needed to retrieve the computed value (we call such futures "control-flow
futures"). Several approaches were proposed to tackle this issues: the
"forward" construct, that allows the programmer to make delegation explicit and
avoid nested futures, and "data-flow explicit futures" which natively collapse
nested futures into plain futures. This paper supports the claim that data-flow
explicit futures form a powerful set of language primitives, on top of which
other approaches can be built. We prove the equivalence, in the context of
data-flow explicit futures, between the "forward" construct and classical
"return" from functions. The proof relies on a branching bisimulation between a
program using "forward" and its "return" counterpart. This result allows
language designers to consider "forward" as an optimisation directive rather
than as a language primitive. Following the principles of the Godot system, we
provide a library implementation of control-flow futures, based on data-flow
explicit futures implemented in the compiler. This small library supports the
claim that the implementation of classical futures based on data-flow ones is
easier than the opposite. Our benchmarks show the viability of the approach
from a performance point of view.

    

### [[2107.07300] Deriving Static Security Testing from Runtime Security Protection for Web Applications](http://arxiv.org/abs/2107.07300)


  Context: Static Application Security Testing (SAST) and Runtime Application
Security Protection (RASP) are important and complementary techniques used for
detecting and enforcing application-level security policies in web
applications.
Inquiry: The current state of the art, however, does not allow a safe and
efficient combination of SAST and RASP based on a shared set of security
policies, forcing developers to reimplement and maintain the same policies and
their enforcement code in both tools.
Approach: In this work, we present a novel technique for deriving SAST from
an existing RASP mechanism by using a two-phase abstract interpretation
approach in the SAST component that avoids duplicating the effort of specifying
security policies and implementing their semantics. The RASP mechanism enforces
security policies by instrumenting a base program to trap security-relevant
operations and execute the required policy enforcement code. The static
analysis of security policies is then obtained from the RASP mechanism by first
statically analyzing the base program without any traps. The results of this
first phase are used in a second phase to detect trapped operations and
abstractly execute the associated and unaltered RASP policy enforcement code.
Knowledge: Splitting the analysis into two phases enables running each phase
with a specific analysis configuration, rendering the static analysis approach
tractable while maintaining sufficient precision.
Grounding: We validate the applicability of our two-phase analysis approach
by using it to both dynamically enforce and statically detect a range of
security policies found in related work. Our experiments suggest that our
two-phase analysis can enable faster and more precise policy violation
detection compared to analyzing the full instrumented application under a
single analysis configuration.
Importance: Deriving a SAST component from a RASP mechanism enables
equivalent semantics for the security policies across the static and dynamic
contexts in which policies are verified during the software development
lifecycle. Moreover, our two-phase abstract interpretation approach does not
require RASP developers to reimplement the enforcement code for static
analysis.

    

### [[2107.07301] A Functional Programming Language with Versions](http://arxiv.org/abs/2107.07301)


  While modern software development heavily uses versioned packages,
programming languages rarely support the concept of versions in their
semantics, which makes software updates more bulky and unsafe. This paper
proposes a programming language that intrinsically supports versions. The main
goals are to design core language features to support multiple versions in one
program and establish a proper notion of type safety with those features. The
proposed core calculus, called Lambda VL, has versioned values, each containing
different values under different versions. We show the construction of the type
system as an extension of coeffect calculus by mapping versions to
computational resources. The type system guarantees the existence of a valid
combination of versions for a program. The calculus enables programming
languages to use multiple versions of a package within a program. It will serve
as a basis for designing advanced language features like module systems and
semantic versioning.

    

### [[1808.02010] Polymorphic Iterable Sequential Effect Systems](http://arxiv.org/abs/1808.02010)


  Effect systems are lightweight extensions to type systems that can verify a
wide range of important properties with modest developer burden. But our
general understanding of effect systems is limited primarily to systems where
the order of effects is irrelevant. Understanding such systems in terms of a
semilattice of effects grounds understanding of the essential issues, and
provides guidance when designing new effect systems. By contrast, sequential
effect systems -- where the order of effects is important -- lack an
established algebraic structure on effects.
We present an abstract polymorphic effect system parameterized by an effect
quantale -- an algebraic structure with well-defined properties that can model
the effects of a range of existing sequential effect systems. We define effect
quantales, derive useful properties, and show how they cleanly model a variety
of known sequential effect systems.
We show that for most effect quantales, there is an induced notion of
iterating a sequential effect; that for systems we consider the derived
iteration agrees with the manually designed iteration operators in prior work;
and that this induced notion of iteration is as precise as possible when
defined. We also position effect quantales with respect to work on categorical
semantics for sequential effect systems, clarifying the distinctions between
these systems and our own in the course of giving a thorough survey of these
frameworks. Our derived iteration construct should generalize to these semantic
structures, addressing limitations of that work. Finally, we consider the
relationship between sequential effects and Kleene Algebras, where the latter
may be used as instances of the former.

    

### [[2005.09028] Sham: A DSL for Fast DSLs](http://arxiv.org/abs/2005.09028)


  Domain-specific languages (DSLs) are touted as both easy to embed in programs
and easy to optimize. Yet these goals are often in tension. Embedded or
internal DSLs fit naturally with a host language, while inheriting the host's
performance characteristics. External DSLs can use external optimizers and
languages but sit apart from the host. We present Sham, a toolkit designed to
enable internal DSLs with high performance. Sham provides a domain-specific
language (embedded in Racket) for implementing other high-performance DSLs,
with transparent compilation to assembly code at runtime. Sham is well suited
as both a compilation target for other embedded DSLs and for transparently
replacing DSL support code with faster versions. Sham provides seamless
inter-operation with its host language without requiring any additional effort
from its users. Sham also provides a framework for defining language syntax
which implements Sham's own language interface as well. We validate Sham's
design on a series of case studies, ranging from Krishnamurthi's classic
automata DSL to a sound synthesis DSL and a probabilistic programming language.
All of these are existing DSLs where we replaced the backend using Sham,
resulting in major performance gains. We present an example-driven description
of how Sham can smoothly enhance an existing DSL into a high-performance one.
When compared to existing approaches for implementing high-performance DSLs,
Sham's design aims for both simplicity and programmer control. This makes it
easier to port our techniques to other languages and frameworks, or borrow
Sham's innovations "à la carte" without adopting the whole approach. Sham
builds a sophisticated and powerful DSL construction toolkit atop fundamental
language features including higher-order functions, data structures, and a
foreign-function interface (FFI), all readily available in other languages.
Furthermore, Sham's approach allows DSL developers to simply write functions,
either using Sham or generating Sham, without needing to work through complex
staging or partial evaluation systems.

    

### [[2105.06081] Gradual Program Analysis for Null Pointers](http://arxiv.org/abs/2105.06081)


  Static analysis tools typically address the problem of excessive false
positives by requiring programmers to explicitly annotate their code. However,
when faced with incomplete annotations, many analysis tools are either too
conservative, yielding false positives, or too optimistic, resulting in unsound
analysis results. In order to flexibly and soundly deal with
partially-annotated programs, we propose to build upon and adapt the gradual
typing approach to abstract-interpretation-based program analyses.
Specifically, we focus on null-pointer analysis and demonstrate that a gradual
null-pointer analysis hits a sweet spot, by gracefully applying static analysis
where possible and relying on dynamic checks where necessary for soundness. In
addition to formalizing a gradual null-pointer analysis for a core imperative
language, we build a prototype using the Infer static analysis framework, and
present preliminary evidence that the gradual null-pointer analysis reduces
false positives compared to two existing null-pointer checkers for Infer.
Further, we discuss ways in which the gradualization approach used to derive
the gradual analysis from its static counterpart can be extended to support
more domains. This work thus provides a basis for future analysis tools that
can smoothly navigate the tradeoff between human effort and run-time overhead
to reduce the number of reported false positives.

    

### [<title>Incremental training of xgboost with fewer classes present - RFC - XGBoost</title>](https://discuss.xgboost.ai/t/incremental-training-of-xgboost-with-fewer-classes-present/2374/1)