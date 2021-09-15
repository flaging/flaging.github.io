
## 2021-9-15

### [[2109.06168] Generatively Augmented Neural Network Watchdog for Image Classification Networks](http://arxiv.org/abs/2109.06168)


  The identification of out-of-distribution data is vital to the deployment of
classification networks. For example, a generic neural network that has been
trained to differentiate between images of dogs and cats can only classify an
input as either a dog or a cat. If a picture of a car or a kumquat were to be
supplied to this classifier, the result would still be either a dog or a cat.
In order to mitigate this, techniques such as the neural network watchdog have
been developed. The compression of the image input into the latent layer of the
autoencoder defines the region of in-distribution in the image space. This
in-distribution set of input data has a corresponding boundary in the image
space. The watchdog assesses whether inputs are in inside or outside this
boundary. This paper demonstrates how to sharpen this boundary using generative
network training data augmentation thereby bettering the discrimination and
overall performance of the watchdog.

    

### [[2109.06171] In-filter Computing For Designing Ultra-light Acoustic Pattern Recognizers](http://arxiv.org/abs/2109.06171)


  We present a novel in-filter computing framework that can be used for
designing ultra-light acoustic classifiers for use in smart internet-of-things
(IoTs). Unlike a conventional acoustic pattern recognizer, where the feature
extraction and classification are designed independently, the proposed
architecture integrates the convolution and nonlinear filtering operations
directly into the kernels of a Support Vector Machine (SVM). The result of this
integration is a template-based SVM whose memory and computational footprint
(training and inference) is light enough to be implemented on an FPGA-based IoT
platform. While the proposed in-filter computing framework is general enough,
in this paper, we demonstrate this concept using a Cascade of Asymmetric
Resonator with Inner Hair Cells (CAR-IHC) based acoustic feature extraction
algorithm. The complete system has been optimized using time-multiplexing and
parallel-pipeline techniques for a Xilinx Spartan 7 series Field Programmable
Gate Array (FPGA). We show that the system can achieve robust classification
performance on benchmark sound recognition tasks using only ~ 1.5k Look-Up
Tables (LUTs) and ~ 2.8k Flip-Flops (FFs), a significant improvement over other
approaches.

    

### [[2109.06176] TREATED:Towards Universal Defense against Textual Adversarial Attacks](http://arxiv.org/abs/2109.06176)


  Recent work shows that deep neural networks are vulnerable to adversarial
examples. Much work studies adversarial example generation, while very little
work focuses on more critical adversarial defense. Existing adversarial
detection methods usually make assumptions about the adversarial example and
attack method (e.g., the word frequency of the adversarial example, the
perturbation level of the attack method). However, this limits the
applicability of the detection method. To this end, we propose TREATED, a
universal adversarial detection method that can defend against attacks of
various perturbation levels without making any assumptions. TREATED identifies
adversarial examples through a set of well-designed reference models. Extensive
experiments on three competitive neural networks and two widely used datasets
show that our method achieves better detection performance than baselines. We
finally conduct ablation studies to verify the effectiveness of our method.

    

### [[2109.06180] Deep Generative Models to Extend Active Directory Graphs with Honeypot Users](http://arxiv.org/abs/2109.06180)


  Active Directory (AD) is a crucial element of large organizations, given its
central role in managing access to resources. Since AD is used by all users in
the organization, it is hard to detect attackers. We propose to generate and
place fake users (honeyusers) in AD structures to help detect attacks. However,
not any honeyuser will attract attackers. Our method generates honeyusers with
a Variational Autoencoder that enriches the AD structure with well-positioned
honeyusers. It first learns the embeddings of the original nodes and edges in
the AD, then it uses a modified Bidirectional DAG-RNN to encode the parameters
of the probability distribution of the latent space of node representations.
Finally, it samples nodes from this distribution and uses an MLP to decide
where the nodes are connected. The model was evaluated by the similarity of the
generated AD with the original, by the positions of the new nodes, by the
similarity with GraphRNN and finally by making real intruders attack the
generated AD structure to see if they select the honeyusers. Results show that
our machine learning model is good enough to generate well-placed honeyusers
for existing AD structures so that intruders are lured into them.

    

### [[2109.06181] Towards Better Model Understanding with Path-Sufficient Explanations](http://arxiv.org/abs/2109.06181)


  Feature based local attribution methods are amongst the most prevalent in
explainable artificial intelligence (XAI) literature. Going beyond standard
correlation, recently, methods have been proposed that highlight what should be
minimally sufficient to justify the classification of an input (viz. pertinent
positives). While minimal sufficiency is an attractive property, the resulting
explanations are often too sparse for a human to understand and evaluate the
local behavior of the model, thus making it difficult to judge its overall
quality. To overcome these limitations, we propose a novel method called
Path-Sufficient Explanations Method (PSEM) that outputs a sequence of
sufficient explanations for a given input of strictly decreasing size (or
value) -- from original input to a minimally sufficient explanation -- which
can be thought to trace the local boundary of the model in a smooth manner,
thus providing better intuition about the local model behavior for the specific
input. We validate these claims, both qualitatively and quantitatively, with
experiments that show the benefit of PSEM across all three modalities (image,
tabular and text). A user study depicts the strength of the method in
communicating the local behavior, where (many) users are able to correctly
determine the prediction made by a model.

    

### [[2109.06234] Machine Learning for Online Algorithm Selection under Censored Feedback](http://arxiv.org/abs/2109.06234)


  In online algorithm selection (OAS), instances of an algorithmic problem
class are presented to an agent one after another, and the agent has to quickly
select a presumably best algorithm from a fixed set of candidate algorithms.
For decision problems such as satisfiability (SAT), quality typically refers to
the algorithm's runtime. As the latter is known to exhibit a heavy-tail
distribution, an algorithm is normally stopped when exceeding a predefined
upper time limit. As a consequence, machine learning methods used to optimize
an algorithm selection strategy in a data-driven manner need to deal with
right-censored samples, a problem that has received little attention in the
literature so far. In this work, we revisit multi-armed bandit algorithms for
OAS and discuss their capability of dealing with the problem. Moreover, we
adapt them towards runtime-oriented losses, allowing for partially censored
data while keeping a space- and time-complexity independent of the time
horizon. In an extensive experimental evaluation on an adapted version of the
ASlib benchmark, we demonstrate that theoretically well-founded methods based
on Thompson sampling perform specifically strong and improve in comparison to
existing methods.

    

### [[2109.06253] Multi-Sentence Resampling: A Simple Approach to Alleviate Dataset Length Bias and Beam-Search Degradation](http://arxiv.org/abs/2109.06253)


  Neural Machine Translation (NMT) is known to suffer from a beam-search
problem: after a certain point, increasing beam size causes an overall drop in
translation quality. This effect is especially pronounced for long sentences.
While much work was done analyzing this phenomenon, primarily for
autoregressive NMT models, there is still no consensus on its underlying cause.
In this work, we analyze errors that cause major quality degradation with large
beams in NMT and Automatic Speech Recognition (ASR). We show that a factor that
strongly contributes to the quality degradation with large beams is
\textit{dataset length-bias} - \textit{NMT datasets are strongly biased towards
short sentences}. To mitigate this issue, we propose a new data augmentation
technique -- \textit{Multi-Sentence Resampling (MSR)}. This technique extends
the training examples by concatenating several sentences from the original
dataset to make a long training example. We demonstrate that MSR significantly
reduces degradation with growing beam size and improves final translation
quality on the IWSTL$15$ En-Vi, IWSTL$17$ En-Fr, and WMT$14$ En-De datasets.

    

### [[2109.06265] Program-to-Circuit: Exploiting GNNs for Program Representation and Circuit Translation](http://arxiv.org/abs/2109.06265)


  Circuit design is complicated and requires extensive domain-specific
expertise. One major obstacle stuck on the way to hardware agile development is
the considerably time-consuming process of accurate circuit quality evaluation.
To significantly expedite the circuit evaluation during the translation from
behavioral languages to circuit designs, we formulate it as a
Program-to-Circuit problem, aiming to exploit the representation power of graph
neural networks (GNNs) by representing C/C++ programs as graphs. The goal of
this work is four-fold. First, we build a standard benchmark containing 40k
C/C++ programs, each of which is translated to a circuit design with actual
hardware quality metrics, aiming to facilitate the development of effective
GNNs targeting this high-demand circuit design area. Second, 14
state-of-the-art GNN models are analyzed on the Program-to-Circuit problem. We
identify key design challenges of this problem, which should be carefully
handled but not yet solved by existing GNNs. The goal is to provide
domain-specific knowledge for designing GNNs with suitable inductive biases.
Third, we discuss three sets of real-world benchmarks for GNN generalization
evaluation, and analyze the performance gap between standard programs and the
real-case ones. The goal is to enable transfer learning from limited training
data to real-world large-scale circuit design problems. Fourth, the
Program-to-Circuit problem is a representative within the Program-to-X
framework, a set of program-based analysis problems with various downstream
tasks. The in-depth understanding of strength and weaknesses in applying GNNs
on Program-to-Circuit could largely benefit the entire family of Program-to-X.
Pioneering in this direction, we expect more GNN endeavors to revolutionize
this high-demand Program-to-Circuit problem and to enrich the expressiveness of
GNNs on programs.

    

### [[2109.06266] Automatic Tuning of Tensorflow's CPU Backend using Gradient-Free Optimization Algorithms](http://arxiv.org/abs/2109.06266)


  Modern deep learning (DL) applications are built using DL libraries and
frameworks such as TensorFlow and PyTorch. These frameworks have complex
parameters and tuning them to obtain good training and inference performance is
challenging for typical users, such as DL developers and data scientists.
Manual tuning requires deep knowledge of the user-controllable parameters of DL
frameworks as well as the underlying hardware. It is a slow and tedious
process, and it typically delivers sub-optimal solutions.
In this paper, we treat the problem of tuning parameters of DL frameworks to
improve training and inference performance as a black-box optimization problem.
We then investigate applicability and effectiveness of Bayesian optimization
(BO), genetic algorithm (GA), and Nelder-Mead simplex (NMS) to tune the
parameters of TensorFlow's CPU backend. While prior work has already
investigated the use of Nelder-Mead simplex for a similar problem, it does not
provide insights into the applicability of other more popular algorithms.
Towards that end, we provide a systematic comparative analysis of all three
algorithms in tuning TensorFlow's CPU backend on a variety of DL models. Our
findings reveal that Bayesian optimization performs the best on the majority of
models. There are, however, cases where it does not deliver the best results.

    

### [[2109.06275] MindCraft: Theory of Mind Modeling for Situated Dialogue in Collaborative Tasks](http://arxiv.org/abs/2109.06275)


  An ideal integration of autonomous agents in a human world implies that they
are able to collaborate on human terms. In particular, theory of mind plays an
important role in maintaining common ground during human collaboration and
communication. To enable theory of mind modeling in situated interactions, we
introduce a fine-grained dataset of collaborative tasks performed by pairs of
human subjects in the 3D virtual blocks world of Minecraft. It provides
information that captures partners' beliefs of the world and of each other as
an interaction unfolds, bringing abundant opportunities to study human
collaborative behaviors in situated language communication. As a first step
towards our goal of developing embodied AI agents able to infer belief states
of collaborative partners in situ, we build and present results on
computational models for several theory of mind tasks.

    

### [[2109.06294] On the regularized risk of distributionally robust learning over deep neural networks](http://arxiv.org/abs/2109.06294)


  In this paper we explore the relation between distributionally robust
learning and different forms of regularization to enforce robustness of deep
neural networks. In particular, starting from a concrete min-max
distributionally robust problem, and using tools from optimal transport theory,
we derive first order and second order approximations to the distributionally
robust problem in terms of appropriate regularized risk minimization problems.
In the context of deep ResNet models, we identify the structure of the
resulting regularization problems as mean-field optimal control problems where
the number and dimension of state variables is within a dimension-free factor
of the dimension of the original unrobust problem. Using the Pontryagin maximum
principles associated to these problems we motivate a family of scalable
algorithms for the training of robust neural networks. Our analysis recovers
some results and algorithms known in the literature (in settings explained
throughout the paper) and provides many other theoretical and algorithmic
insights that to our knowledge are novel. In our analysis we employ tools that
we deem useful for a future analysis of more general adversarial learning
problems.

    

### [[2109.06308] Mitigating Catastrophic Forgetting in Scheduled Sampling with Elastic Weight Consolidation in Neural Machine Translation](http://arxiv.org/abs/2109.06308)


  Despite strong performance in many sequence-to-sequence tasks, autoregressive
models trained with maximum likelihood estimation suffer from exposure bias,
i.e. a discrepancy between the ground-truth prefixes used during training and
the model-generated prefixes used at inference time. Scheduled sampling is a
simple and often empirically successful approach which addresses this issue by
incorporating model-generated prefixes into the training process. However, it
has been argued that it is an inconsistent training objective leading to models
ignoring the prefixes altogether. In this paper, we conduct systematic
experiments and find that it ameliorates exposure bias by increasing model
reliance on the input sequence. We also observe that as a side-effect, it
worsens performance when the model-generated prefix is correct, a form of
catastrophic forgetting. We propose using Elastic Weight Consolidation as
trade-off between mitigating exposure bias and retaining output quality.
Experiments on two IWSLT'14 translation tasks demonstrate that our approach
alleviates catastrophic forgetting and significantly improves BLEU compared to
standard scheduled sampling.

    

### [[2109.06310] State Relevance for Off-Policy Evaluation](http://arxiv.org/abs/2109.06310)


  Importance sampling-based estimators for off-policy evaluation (OPE) are
valued for their simplicity, unbiasedness, and reliance on relatively few
assumptions. However, the variance of these estimators is often high,
especially when trajectories are of different lengths. In this work, we
introduce Omitting-States-Irrelevant-to-Return Importance Sampling (OSIRIS), an
estimator which reduces variance by strategically omitting likelihood ratios
associated with certain states. We formalize the conditions under which OSIRIS
is unbiased and has lower variance than ordinary importance sampling, and we
demonstrate these properties empirically.

    

### [[2109.06312] Pre-emptive learning-to-defer for sequential medical decision-making under uncertainty](http://arxiv.org/abs/2109.06312)


  We propose SLTD (`Sequential Learning-to-Defer') a framework for
learning-to-defer pre-emptively to an expert in sequential decision-making
settings. SLTD measures the likelihood of improving value of deferring now
versus later based on the underlying uncertainty in dynamics. In particular, we
focus on the non-stationarity in the dynamics to accurately learn the deferral
policy. We demonstrate our pre-emptive deferral can identify regions where the
current policy has a low probability of improving outcomes. SLTD outperforms
existing non-sequential learning-to-defer baselines, whilst reducing overall
uncertainty on multiple synthetic and real-world simulators with non-stationary
dynamics. We further derive and decompose the propagated (long-term)
uncertainty for interpretation by the domain expert to provide an indication of
when the model's performance is reliable.

    

### [[2109.06321] Mitigating Sampling Bias and Improving Robustness in Active Learning](http://arxiv.org/abs/2109.06321)


  This paper presents simple and efficient methods to mitigate sampling bias in
active learning while achieving state-of-the-art accuracy and model robustness.
We introduce supervised contrastive active learning by leveraging the
contrastive loss for active learning under a supervised setting. We propose an
unbiased query strategy that selects informative data samples of diverse
feature representations with our methods: supervised contrastive active
learning (SCAL) and deep feature modeling (DFM). We empirically demonstrate our
proposed methods reduce sampling bias, achieve state-of-the-art accuracy and
model calibration in an active learning setup with the query computation 26x
faster than Bayesian active learning by disagreement and 11x faster than
CoreSet. The proposed SCAL method outperforms by a big margin in robustness to
dataset shift and out-of-distribution.

    

### [[2109.06324] A Massively Multilingual Analysis of Cross-linguality in Shared Embedding Space](http://arxiv.org/abs/2109.06324)


  In cross-lingual language models, representations for many different
languages live in the same space. Here, we investigate the linguistic and
non-linguistic factors affecting sentence-level alignment in cross-lingual
pretrained language models for 101 languages and 5,050 language pairs. Using
BERT-based LaBSE and BiLSTM-based LASER as our models, and the Bible as our
corpus, we compute a task-based measure of cross-lingual alignment in the form
of bitext retrieval performance, as well as four intrinsic measures of vector
space alignment and isomorphism. We then examine a range of linguistic,
quasi-linguistic, and training-related features as potential predictors of
these alignment metrics. The results of our analyses show that word order
agreement and agreement in morphological complexity are two of the strongest
linguistic predictors of cross-linguality. We also note in-family training data
as a stronger predictor than language-specific training data across the board.
We verify some of our linguistic findings by looking at the effect of
morphological segmentation on English-Inuktitut alignment, in addition to
examining the effect of word order agreement on isomorphism for 66 zero-shot
language pairs from a different corpus. We make the data and code for our
experiments publicly available.

    

### [[2109.06325] safe-control-gym: a Unified Benchmark Suite for Safe Learning-based Control and Reinforcement Learning](http://arxiv.org/abs/2109.06325)


  In recent years, reinforcement learning and learning-based control -- as well
as the study of their safety, crucial for deployment in real-world robots --
have gained significant traction. However, to adequately gauge the progress and
applicability of new results, we need the tools to equitably compare the
approaches proposed by the controls and reinforcement learning communities.
Here, we propose a new open-source benchmark suite, called safe-control-gym.
Our starting point is OpenAI's Gym API, which is one of the de facto standard
in reinforcement learning research. Yet, we highlight the reasons for its
limited appeal to control theory researchers -- and safe control, in
particular. E.g., the lack of analytical models and constraint specifications.
Thus, we propose to extend this API with (i) the ability to specify (and query)
symbolic models and constraints and (ii) introduce simulated disturbances in
the control inputs, measurements, and inertial properties. We provide
implementations for three dynamic systems -- the cart-pole, 1D, and 2D
quadrotor -- and two control tasks -- stabilization and trajectory tracking. To
demonstrate our proposal -- and in an attempt to bring research communities
closer together -- we show how to use safe-control-gym to quantitatively
compare the control performance, data efficiency, and safety of multiple
approaches from the areas of traditional control, learning-based control, and
reinforcement learning.

    

### [[2109.06332] Achieving Zero Constraint Violation for Constrained Reinforcement Learning via Primal-Dual Approach](http://arxiv.org/abs/2109.06332)


  Reinforcement learning is widely used in applications where one needs to
perform sequential decisions while interacting with the environment. The
problem becomes more challenging when the decision requirement includes
satisfying some safety constraints. The problem is mathematically formulated as
constrained Markov decision process (CMDP). In the literature, various
algorithms are available to solve CMDP problems in a model-free manner to
achieve $\epsilon$-optimal cumulative reward with $\epsilon$ feasible policies.
An $\epsilon$-feasible policy implies that it suffers from constraint
violation. An important question here is whether we can achieve
$\epsilon$-optimal cumulative reward with zero constraint violations or not. To
achieve that, we advocate the use of a randomized primal-dual approach to
solving the CMDP problems and propose a conservative stochastic primal-dual
algorithm (CSPDA) which is shown to exhibit $\tilde{\mathcal{O}}(1/\epsilon^2)$
sample complexity to achieve $\epsilon$-optimal cumulative reward with zero
constraint violations. In the prior works, the best available sample complexity
for the $\epsilon$-optimal policy with zero constraint violation is
$\tilde{\mathcal{O}}(1/\epsilon^5)$. Hence, the proposed algorithm provides a
significant improvement as compared to the state of the art.

    

### [[2109.06339] ML Based Lineage in Databases](http://arxiv.org/abs/2109.06339)


  In this work, we track the lineage of tuples throughout their database
lifetime. That is, we consider a scenario in which tuples (records) that are
produced by a query may affect other tuple insertions into the DB, as part of a
normal workflow. As time goes on, exact provenance explanations for such tuples
become deeply nested, increasingly consuming space, and resulting in decreased
clarity and readability. We present a novel approach for approximating lineage
tracking, using a Machine Learning (ML) and Natural Language Processing (NLP)
technique; namely, word embedding. The basic idea is summarizing (and
approximating) the lineage of each tuple via a small set of constant-size
vectors (the number of vectors per-tuple is a hyperparameter). Therefore, our
solution does not suffer from space complexity blow-up over time, and it
"naturally ranks" explanations to the existence of a tuple. We devise an
alternative and improved lineage tracking mechanism, that of keeping track of
and querying lineage at the column level; thereby, we manage to better
distinguish between the provenance features and the textual characteristics of
a tuple. We integrate our lineage computations into the PostgreSQL system via
an extension (ProvSQL) and experimentally exhibit useful results in terms of
accuracy against exact, semiring-based, justifications. In the experiments, we
focus on tuples with multiple generations of tuples in their lifelong lineage
and analyze them in terms of direct and distant lineage. The experiments
suggest a high usefulness potential for the proposed approximate lineage
methods and the further suggested enhancements. This especially holds for the
column-based vectors method which exhibits high precision and high per-level
recall.

    

### [[2109.06346] Physics Driven Domain Specific Transporter Framework with Attention Mechanism for Ultrasound Imaging](http://arxiv.org/abs/2109.06346)


  Most applications of deep learning techniques in medical imaging are
supervised and require a large number of labeled data which is expensive and
requires many hours of careful annotation by experts. In this paper, we propose
an unsupervised, physics driven domain specific transporter framework with an
attention mechanism to identify relevant key points with applications in
ultrasound imaging. The proposed framework identifies key points that provide a
concise geometric representation highlighting regions with high structural
variation in ultrasound videos. We incorporate physics driven domain specific
information as a feature probability map and use the radon transform to
highlight features in specific orientations. The proposed framework has been
trained on130 Lung ultrasound (LUS) videos and 113 Wrist ultrasound (WUS)
videos and validated on 100 Lung ultrasound (LUS) videos and 58 Wrist
ultrasound (WUS) videos acquired from multiple centers across the globe. Images
from both datasets were independently assessed by experts to identify
clinically relevant features such as A-lines, B-lines and pleura from LUS and
radial metaphysis, radial epiphysis and carpal bones from WUS videos. The key
points detected from both datasets showed high sensitivity (LUS = 99\% , WUS =
74\%) in detecting the image landmarks identified by experts. Also, on
employing for classification of the given lung image into normal and abnormal
classes, the proposed approach, even with no prior training, achieved an
average accuracy of 97\% and an average F1-score of 95\% respectively on the
task of co-classification with 3 fold cross-validation. With the purely
unsupervised nature of the proposed approach, we expect the key point detection
approach to increase the applicability of ultrasound in various examination
performed in emergency and point of care.

    

### [[2109.06349] Few-Shot Intent Detection via Contrastive Pre-Training and Fine-Tuning](http://arxiv.org/abs/2109.06349)


  In this work, we focus on a more challenging few-shot intent detection
scenario where many intents are fine-grained and semantically similar. We
present a simple yet effective few-shot intent detection schema via contrastive
pre-training and fine-tuning. Specifically, we first conduct self-supervised
contrastive pre-training on collected intent datasets, which implicitly learns
to discriminate semantically similar utterances without using any labels. We
then perform few-shot intent detection together with supervised contrastive
learning, which explicitly pulls utterances from the same intent closer and
pushes utterances across different intents farther. Experimental results show
that our proposed method achieves state-of-the-art performance on three
challenging intent detection datasets under 5-shot and 10-shot settings.

    

### [[2109.06352] Uncertainty-Aware Machine Translation Evaluation](http://arxiv.org/abs/2109.06352)


  Several neural-based metrics have been recently proposed to evaluate machine
translation quality. However, all of them resort to point estimates, which
provide limited information at segment level. This is made worse as they are
trained on noisy, biased and scarce human judgements, often resulting in
unreliable quality predictions. In this paper, we introduce uncertainty-aware
MT evaluation and analyze the trustworthiness of the predicted quality. We
combine the COMET framework with two uncertainty estimation methods, Monte
Carlo dropout and deep ensembles, to obtain quality scores along with
confidence intervals. We compare the performance of our uncertainty-aware MT
evaluation methods across multiple language pairs from the QT21 dataset and the
WMT20 metrics task, augmented with MQM annotations. We experiment with varying
numbers of references and further discuss the usefulness of uncertainty-aware
quality estimation (without references) to flag possibly critical translation
mistakes.

    

### [[2109.06358] A Practical Adversarial Attack on Contingency Detection of Smart Energy Systems](http://arxiv.org/abs/2109.06358)


  Due to the advances in computing and sensing, deep learning (DL) has widely
been applied in smart energy systems (SESs). These DL-based solutions have
proved their potentials in improving the effectiveness and adaptiveness of the
control systems. However, in recent years, increasing evidence shows that DL
techniques can be manipulated by adversarial attacks with carefully-crafted
perturbations. Adversarial attacks have been studied in computer vision and
natural language processing. However, there is very limited work focusing on
the adversarial attack deployment and mitigation in energy systems. In this
regard, to better prepare the SESs against potential adversarial attacks, we
propose an innovative adversarial attack model that can practically compromise
dynamical controls of energy system. We also optimize the deployment of the
proposed adversarial attack model by employing deep reinforcement learning (RL)
techniques. In this paper, we present our first-stage work in this direction.
In simulation section, we evaluate the performance of our proposed adversarial
attack model using standard IEEE 9-bus system.

    

### [[2109.06362] Theoretical Guarantees of Fictitious Discount Algorithms for Episodic Reinforcement Learning and Global Convergence of Policy Gradient Methods](http://arxiv.org/abs/2109.06362)


  When designing algorithms for finite-time-horizon episodic reinforcement
learning problems, a common approach is to introduce a fictitious discount
factor and use stationary policies for approximations. Empirically, it has been
shown that the fictitious discount factor helps reduce variance, and stationary
policies serve to save the per-iteration computational cost. Theoretically,
however, there is no existing work on convergence analysis for algorithms with
this fictitious discount recipe. This paper takes the first step towards
analyzing these algorithms. It focuses on two vanilla policy gradient (VPG)
variants: the first being a widely used variant with discounted advantage
estimations (DAE), the second with an additional fictitious discount factor in
the score functions of the policy gradient estimators. Non-asymptotic
convergence guarantees are established for both algorithms, and the additional
discount factor is shown to reduce the bias introduced in DAE and thus improve
the algorithm convergence asymptotically. A key ingredient of our analysis is
to connect three settings of Markov decision processes (MDPs): the
finite-time-horizon, the average reward and the discounted settings. To our
best knowledge, this is the first theoretical guarantee on fictitious discount
algorithms for the episodic reinforcement learning of finite-time-horizon MDPs,
which also leads to the (first) global convergence of policy gradient methods
for finite-time-horizon episodic reinforcement learning.

    

### [[2109.06363] Sensor Adversarial Traits: Analyzing Robustness of 3D Object Detection Sensor Fusion Models](http://arxiv.org/abs/2109.06363)


  A critical aspect of autonomous vehicles (AVs) is the object detection stage,
which is increasingly being performed with sensor fusion models: multimodal 3D
object detection models which utilize both 2D RGB image data and 3D data from a
LIDAR sensor as inputs. In this work, we perform the first study to analyze the
robustness of a high-performance, open source sensor fusion model architecture
towards adversarial attacks and challenge the popular belief that the use of
additional sensors automatically mitigate the risk of adversarial attacks. We
find that despite the use of a LIDAR sensor, the model is vulnerable to our
purposefully crafted image-based adversarial attacks including disappearance,
universal patch, and spoofing. After identifying the underlying reason, we
explore some potential defenses and provide some recommendations for improved
sensor fusion models.

    

### [[2109.06365] From Heatmaps to Structural Explanations of Image Classifiers](http://arxiv.org/abs/2109.06365)


  This paper summarizes our endeavors in the past few years in terms of
explaining image classifiers, with the aim of including negative results and
insights we have gained. The paper starts with describing the explainable
neural network (XNN), which attempts to extract and visualize several
high-level concepts purely from the deep network, without relying on human
linguistic concepts. This helps users understand network classifications that
are less intuitive and substantially improves user performance on a difficult
fine-grained classification task of discriminating among different species of
seagulls.
Realizing that an important missing piece is a reliable heatmap visualization
tool, we have developed I-GOS and iGOS++ utilizing integrated gradients to
avoid local optima in heatmap generation, which improved the performance across
all resolutions. During the development of those visualizations, we realized
that for a significant number of images, the classifier has multiple different
paths to reach a confident prediction. This has lead to our recent development
of structured attention graphs (SAGs), an approach that utilizes beam search to
locate multiple coarse heatmaps for a single image, and compactly visualizes a
set of heatmaps by capturing how different combinations of image regions impact
the confidence of a classifier.
Through the research process, we have learned much about insights in building
deep network explanations, the existence and frequency of multiple
explanations, and various tricks of the trade that make explanations work. In
this paper, we attempt to share those insights and opinions with the readers
with the hope that some of them will be informative for future researchers on
explainable deep learning.

    

### [[2109.06368] Policy Optimization Using Semiparametric Models for Dynamic Pricing](http://arxiv.org/abs/2109.06368)


  In this paper, we study the contextual dynamic pricing problem where the
market value of a product is linear in its observed features plus some market
noise. Products are sold one at a time, and only a binary response indicating
success or failure of a sale is observed. Our model setting is similar to
Javanmard and Nazerzadeh [2019] except that we expand the demand curve to a
semiparametric model and need to learn dynamically both parametric and
nonparametric components. We propose a dynamic statistical learning and
decision-making policy that combines semiparametric estimation from a
generalized linear model with an unknown link and online decision-making to
minimize regret (maximize revenue). Under mild conditions, we show that for a
market noise c.d.f. $F(\cdot)$ with $m$-th order derivative ($m\geq 2$), our
policy achieves a regret upper bound of $\tilde{O}_{d}(T^{\frac{2m+1}{4m-1}})$,
where $T$ is time horizon and $\tilde{O}_{d}$ is the order that hides
logarithmic terms and the dimensionality of feature $d$. The upper bound is
further reduced to $\tilde{O}_{d}(\sqrt{T})$ if $F$ is super smooth whose
Fourier transform decays exponentially. In terms of dependence on the horizon
$T$, these upper bounds are close to $\Omega(\sqrt{T})$, the lower bound where
$F$ belongs to a parametric class. We further generalize these results to the
case with dynamically dependent product features under the strong mixing
condition.

    

### [[2109.06379] Compression, Transduction, and Creation: A Unified Framework for Evaluating Natural Language Generation](http://arxiv.org/abs/2109.06379)


  Natural language generation (NLG) spans a broad range of tasks, each of which
serves for specific objectives and desires different properties of generated
text. The complexity makes automatic evaluation of NLG particularly
challenging. Previous work has typically focused on a single task and developed
individual evaluation metrics based on specific intuitions. In this paper, we
propose a unifying perspective based on the nature of information change in NLG
tasks, including compression (e.g., summarization), transduction (e.g., text
rewriting), and creation (e.g., dialog). Information alignment between input,
context, and output text plays a common central role in characterizing the
generation. With automatic alignment prediction models, we develop a family of
interpretable metrics that are suitable for evaluating key aspects of different
NLG tasks, often without need of gold reference data. Experiments show the
uniformly designed metrics achieve stronger or comparable correlations with
human judgement compared to state-of-the-art metrics in each of diverse tasks,
including text summarization, style transfer, and knowledge-grounded dialog.

    

### [[2109.06387] Rationales for Sequential Predictions](http://arxiv.org/abs/2109.06387)


  Sequence models are a critical component of modern NLP systems, but their
predictions are difficult to explain. We consider model explanations though
rationales, subsets of context that can explain individual model predictions.
We find sequential rationales by solving a combinatorial optimization: the best
rationale is the smallest subset of input tokens that would predict the same
output as the full sequence. Enumerating all subsets is intractable, so we
propose an efficient greedy algorithm to approximate this objective. The
algorithm, which is called greedy rationalization, applies to any model. For
this approach to be effective, the model should form compatible conditional
distributions when making predictions on incomplete subsets of the context.
This condition can be enforced with a short fine-tuning step. We study greedy
rationalization on language modeling and machine translation. Compared to
existing baselines, greedy rationalization is best at optimizing the
combinatorial objective and provides the most faithful rationales. On a new
dataset of annotated sequential rationales, greedy rationales are most similar
to human rationales.

    

### [[2109.06402] Exploring Personality and Online Social Engagement: An Investigation of MBTI Users on Twitter](http://arxiv.org/abs/2109.06402)


  Text-based personality prediction by computational models is an emerging
field with the potential to significantly improve on key weaknesses of
survey-based personality assessment. We investigate 3848 profiles from Twitter
with self-labeled Myers-Briggs personality traits (MBTI) - a framework closely
related to the Five Factor Model of personality - to better understand how
text-based digital traces from social engagement online can be used to predict
user personality traits. We leverage BERT, a state-of-the-art NLP architecture
based on deep learning, to analyze various sources of text that hold most
predictive power for our task. We find that biographies, statuses, and liked
tweets contain significant predictive power for all dimensions of the MBTI
system. We discuss our findings and their implications for the validity of the
MBTI and the lexical hypothesis, a foundational theory underlying the Five
Factor Model that links language use and behavior. Our results hold optimistic
implications for personality psychologists, computational linguists, and other
social scientists aiming to predict personality from observational text data
and explore the links between language and core behavioral traits.

    

### [[2109.06404] Detecting Safety Problems of Multi-Sensor Fusion in Autonomous Driving](http://arxiv.org/abs/2109.06404)


  Autonomous driving (AD) systems have been thriving in recent years. In
general, they receive sensor data, compute driving decisions, and output
control signals to the vehicles. To smooth out the uncertainties brought by
sensor inputs, AD systems usually leverage multi-sensor fusion (MSF) to fuse
the sensor inputs and produce a more reliable understanding of the
surroundings. However, MSF cannot completely eliminate the uncertainties since
it lacks the knowledge about which sensor provides the most accurate data. As a
result, critical consequences might happen unexpectedly. In this work, we
observed that the popular MSF methods in an industry-grade Advanced
Driver-Assistance System (ADAS) can mislead the car control and result in
serious safety hazards. Misbehavior can happen regardless of the used fusion
methods and the accurate data from at least one sensor. To attribute the safety
hazards to a MSF method, we formally define the fusion errors and propose a way
to distinguish safety violations causally induced by such errors. Further, we
develop a novel evolutionary-based domain-specific search framework,
FusionFuzz, for the efficient detection of fusion errors. We evaluate our
framework on two widely used MSF methods. %in two driving environments.
Experimental results show that FusionFuzz identifies more than 150 fusion
errors. Finally, we provide several suggestions to improve the MSF methods
under study.

    

### [[2109.06407] Neural Networks with Physics-Informed Architectures and Constraints for Dynamical Systems Modeling](http://arxiv.org/abs/2109.06407)


  Effective inclusion of physics-based knowledge into deep neural network
models of dynamical systems can greatly improve data efficiency and
generalization. Such a-priori knowledge might arise from physical principles
(e.g., conservation laws) or from the system's design (e.g., the Jacobian
matrix of a robot), even if large portions of the system dynamics remain
unknown. We develop a framework to learn dynamics models from trajectory data
while incorporating a-priori system knowledge as inductive bias. More
specifically, the proposed framework uses physics-based side information to
inform the structure of the neural network itself, and to place constraints on
the values of the outputs and the internal states of the model. It represents
the system's vector field as a composition of known and unknown functions, the
latter of which are parametrized by neural networks. The physics-informed
constraints are enforced via the augmented Lagrangian method during the model's
training. We experimentally demonstrate the benefits of the proposed approach
on a variety of dynamical systems -- including a benchmark suite of robotics
environments featuring large state spaces, non-linear dynamics, external
forces, contact forces, and control inputs. By exploiting a-priori system
knowledge during training, the proposed approach learns to predict the system
dynamics two orders of magnitude more accurately than a baseline approach that
does not include prior knowledge, given the same training dataset.

    

### [[2109.06429] Knowledge-guided Self-supervised Learning for estimating River-Basin Characteristics](http://arxiv.org/abs/2109.06429)


  Machine Learning is being extensively used in hydrology, especially
streamflow prediction of basins/watersheds. Basin characteristics are essential
for modeling the rainfall-runoff response of these watersheds and therefore
data-driven methods must take into account this ancillary characteristics data.
However there are several limitations, namely uncertainty in the measured
characteristics, partially missing characteristics for some of the basins or
unknown characteristics that may not be present in the known measured set. In
this paper we present an inverse model that uses a knowledge-guided
self-supervised learning algorithm to infer basin characteristics using the
meteorological drivers and streamflow response data. We evaluate our model on
the the CAMELS dataset and the results validate its ability to reduce
measurement uncertainty, impute missing characteristics, and identify unknown
characteristics.

    

### [[2109.06431] Exploring the Long Short-Term Dependencies to Infer Shot Influence in Badminton Matches](http://arxiv.org/abs/2109.06431)


  Identifying significant shots in a rally is important for evaluating players'
performance in badminton matches. While there are several studies that have
quantified player performance in other sports, analyzing badminton data is
remained untouched. In this paper, we introduce a badminton language to fully
describe the process of the shot and propose a deep learning model composed of
a novel short-term extractor and a long-term encoder for capturing a
shot-by-shot sequence in a badminton rally by framing the problem as predicting
a rally result. Our model incorporates an attention mechanism to enable the
transparency of the action sequence to the rally result, which is essential for
badminton experts to gain interpretable predictions. Experimental evaluation
based on a real-world dataset demonstrates that our proposed model outperforms
the strong baselines. The source code is publicly available at
this https URL.

    

### [[2109.06440] Complexity-aware Adaptive Training and Inference for Edge-Cloud Distributed AI Systems](http://arxiv.org/abs/2109.06440)


  The ubiquitous use of IoT and machine learning applications is creating large
amounts of data that require accurate and real-time processing. Although
edge-based smart data processing can be enabled by deploying pretrained models,
the energy and memory constraints of edge devices necessitate distributed deep
learning between the edge and the cloud for complex data. In this paper, we
propose a distributed AI system to exploit both the edge and the cloud for
training and inference. We propose a new architecture, MEANet, with a main
block, an extension block, and an adaptive block for the edge. The inference
process can terminate at either the main block, the extension block, or the
cloud. The MEANet is trained to categorize inputs into easy/hard/complex
classes. The main block identifies instances of easy/hard classes and
classifies easy classes with high confidence. Only data with high probabilities
of belonging to hard classes would be sent to the extension block for
prediction. Further, only if the neural network at the edge shows low
confidence in the prediction, the instance is considered complex and sent to
the cloud for further processing. The training technique lends to the majority
of inference on edge devices while going to the cloud only for a small set of
complex jobs, as determined by the edge. The performance of the proposed system
is evaluated via extensive experiments using modified models of ResNets and
MobileNetV2 on CIFAR-100 and ImageNet datasets. The results show that the
proposed distributed model has improved accuracy and energy consumption,
indicating its capacity to adapt.

    

### [[2109.06441] Structure-Enhanced Pop Music Generation via Harmony-Aware Learning](http://arxiv.org/abs/2109.06441)


  Automatically composing pop music with a satisfactory structure is an
attractive but challenging topic. Although the musical structure is easy to be
perceived by human, it is difficult to be described clearly and defined
accurately. And it is still far from being solved that how we should model the
structure in pop music generation. In this paper, we propose to leverage
harmony-aware learning for structure-enhanced pop music generation. On the one
hand, one of the participants of harmony, chord, represents the harmonic set of
multiple notes, which is integrated closely with the spatial structure of
music, texture. On the other hand, the other participant of harmony, chord
progression, usually accompanies with the development of the music, which
promotes the temporal structure of music, form. Besides, when chords evolve
into chord progression, the texture and the form can be bridged by the harmony
naturally, which contributes to the joint learning of the two structures.
Furthermore, we propose the Harmony-Aware Hierarchical Music Transformer (HAT),
which can exploit the structure adaptively from the music, and interact on the
music tokens at multiple levels to enhance the signals of the structure in
various musical elements. Results of subjective and objective evaluations
demonstrate that HAT significantly improves the quality of generated music,
especially in the structureness.

    

### [[2109.06448] Tesla-Rapture: A Lightweight Gesture Recognition System from mmWave Radar Point Clouds](http://arxiv.org/abs/2109.06448)


  We present Tesla-Rapture, a gesture recognition interface for point clouds
generated by mmWave Radars. State of the art gesture recognition models are
either too resource consuming or not sufficiently accurate for integration into
real-life scenarios using wearable or constrained equipment such as IoT devices
(e.g. Raspberry PI), XR hardware (e.g. HoloLens), or smart-phones. To tackle
this issue, we developed Tesla, a Message Passing Neural Network (MPNN) graph
convolution approach for mmWave radar point clouds. The model outperforms the
state of the art on two datasets in terms of accuracy while reducing the
computational complexity and, hence, the execution time. In particular, the
approach, is able to predict a gesture almost 8 times faster than the most
accurate competitor. Our performance evaluation in different scenarios
(environments, angles, distances) shows that Tesla generalizes well and
improves the accuracy up to 20% in challenging scenarios like a through-wall
setting and sensing at extreme angles. Utilizing Tesla, we develop
Tesla-Rapture, a real-time implementation using a mmWave Radar on a Raspberry
PI 4 and evaluate its accuracy and time-complexity. We also publish the source
code, the trained models, and the implementation of the model for embedded
devices.

    

### [[2109.06449] Deep hierarchical reinforcement agents for automated penetration testing](http://arxiv.org/abs/2109.06449)


  Penetration testing the organised attack of a computer system in order to
test existing defences has been used extensively to evaluate network security.
This is a time consuming process and requires in-depth knowledge for the
establishment of a strategy that resembles a real cyber-attack. This paper
presents a novel deep reinforcement learning architecture with hierarchically
structured agents called HA-DRL, which employs an algebraic action
decomposition strategy to address the large discrete action space of an
autonomous penetration testing simulator where the number of actions is
exponentially increased with the complexity of the designed cybersecurity
network. The proposed architecture is shown to find the optimal attacking
policy faster and more stably than a conventional deep Q-learning agent which
is commonly used as a method to apply artificial intelligence in automatic
penetration testing.

    

### [[2109.06450] A machine-learning framework for daylight and visual comfort assessment in early design stages](http://arxiv.org/abs/2109.06450)


  This research is mainly focused on the assessment of machine learning
algorithms in the prediction of daylight and visual comfort metrics in the
early design stages. A dataset was primarily developed from 2880 simulations
derived from Honeybee for Grasshopper. The simulations were done for a shoebox
space with a one side window. The alternatives emerged from different physical
features, including room dimensions, interior surfaces reflectance, window
dimensions and orientations, number of windows, and shading states. 5 metrics
were used for daylight evaluations, including UDI, sDA, mDA, ASE, and sVD.
Quality Views were analyzed for the same shoebox spaces via a grasshopper-based
algorithm, developed from the LEED v4 evaluation framework for Quality Views.
The dataset was further analyzed with an Artificial Neural Network algorithm
written in Python. The accuracy of the predictions was estimated at 97% on
average. The developed model could be used in early design stages analyses
without the need for time-consuming simulations in previously used platforms
and programs.

    

### [[2109.06458] Exploring the Connection between Knowledge Distillation and Logits Matching](http://arxiv.org/abs/2109.06458)


  Knowledge distillation is a generalized logits matching technique for model
compression. Their equivalence is previously established on the condition of
$\textit{infinity temperature}$ and $\textit{zero-mean normalization}$. In this
paper, we prove that with only $\textit{infinity temperature}$, the effect of
knowledge distillation equals to logits matching with an extra regularization.
Furthermore, we reveal that an additional weaker condition --
$\textit{equal-mean initialization}$ rather than the original
$\textit{zero-mean normalization}$ already suffices to set up the equivalence.
The key to our proof is we realize that in modern neural networks with the
cross-entropy loss and softmax activation, the mean of back-propagated gradient
on logits always keeps zero.

    

### [[2109.06459] A Machine-learning Framework for Acoustic Design Assessment in Early Design Stages](http://arxiv.org/abs/2109.06459)


  In time-cost scale model studies, predicting acoustic performance by using
simulation methods is a commonly used method that is preferred. In this field,
building acoustic simulation tools are complicated by several challenges,
including the high cost of acoustic tools, the need for acoustic expertise, and
the time-consuming process of acoustic simulation. The goal of this project is
to introduce a simple model with a short calculation time to estimate the room
acoustic condition in the early design stages of the building. This paper
presents a working prototype for a new method of machine learning (ML) to
approximate a series of typical room acoustic parameters using only geometric
data as input characteristics. A novel dataset consisting of acoustical
simulations of a single room with 2916 different configurations are used to
train and test the proposed model. In the stimulation process, features that
include room dimensions, window size, material absorption coefficient,
furniture, and shading type have been analysed by using Pachyderm acoustic
software. The mentioned dataset is used as the input of seven machine-learning
models based on fully connected Deep Neural Networks (DNN). The average error
of ML models is between 1% to 3%, and the average error of the new predicted
samples after the validation process is between 2% to 12%.

    

### [[2109.06467] Dodging Attack Using Carefully Crafted Natural Makeup](http://arxiv.org/abs/2109.06467)


  Deep learning face recognition models are used by state-of-the-art
surveillance systems to identify individuals passing through public areas
(e.g., airports). Previous studies have demonstrated the use of adversarial
machine learning (AML) attacks to successfully evade identification by such
systems, both in the digital and physical domains. Attacks in the physical
domain, however, require significant manipulation to the human participant's
face, which can raise suspicion by human observers (e.g. airport security
officers). In this study, we present a novel black-box AML attack which
carefully crafts natural makeup, which, when applied on a human participant,
prevents the participant from being identified by facial recognition models. We
evaluated our proposed attack against the ArcFace face recognition model, with
20 participants in a real-world setup that includes two cameras, different
shooting angles, and different lighting conditions. The evaluation results show
that in the digital domain, the face recognition system was unable to identify
all of the participants, while in the physical domain, the face recognition
system was able to identify the participants in only 1.22% of the frames
(compared to 47.57% without makeup and 33.73% with random natural makeup),
which is below a reasonable threshold of a realistic operational environment.

    

### [[2109.06469] Deep Denerative Models for Drug Design and Response](http://arxiv.org/abs/2109.06469)


  Designing new chemical compounds with desired pharmaceutical properties is a
challenging task and takes years of development and testing. Still, a majority
of new drugs fail to prove efficient. Recent success of deep generative
modeling holds promises of generation and optimization of new molecules. In
this review paper, we provide an overview of the current generative models, and
describe necessary biological and chemical terminology, including molecular
representations needed to understand the field of drug design and drug
response. We present commonly used chemical and biological databases, and tools
for generative modeling. Finally, we summarize the current state of generative
modeling for drug design and drug response prediction, highlighting the
state-of-art approaches and limitations the field is currently facing.

    

### [[2109.06486] Conditional Synthetic Data Generation for Robust Machine Learning Applications with Limited Pandemic Data](http://arxiv.org/abs/2109.06486)


  $\textbf{Background:}$ At the onset of a pandemic, such as COVID-19, data
with proper labeling/attributes corresponding to the new disease might be
unavailable or sparse. Machine Learning (ML) models trained with the available
data, which is limited in quantity and poor in diversity, will often be biased
and inaccurate. At the same time, ML algorithms designed to fight pandemics
must have good performance and be developed in a time-sensitive manner. To
tackle the challenges of limited data, and label scarcity in the available
data, we propose generating conditional synthetic data, to be used alongside
real data for developing robust ML models. $\textbf{Methods:}$ We present a
hybrid model consisting of a conditional generative flow and a classifier for
conditional synthetic data generation. The classifier decouples the feature
representation for the condition, which is fed to the flow to extract the local
noise. We generate synthetic data by manipulating the local noise with fixed
conditional feature representation. We also propose a semi-supervised approach
to generate synthetic samples in the absence of labels for a majority of the
available data. $\textbf{Results:}$ We performed conditional synthetic
generation for chest computed tomography (CT) scans corresponding to normal,
COVID-19, and pneumonia afflicted patients. We show that our method
significantly outperforms existing models both on qualitative and quantitative
performance, and our semi-supervised approach can efficiently synthesize
conditional samples under label scarcity. As an example of downstream use of
synthetic data, we show improvement in COVID-19 detection from CT scans with
conditional synthetic data augmentation.

    

### [[2109.06489] Instance-wise Graph-based Framework for Multivariate Time Series Forecasting](http://arxiv.org/abs/2109.06489)


  The multivariate time series forecasting has attracted more and more
attention because of its vital role in different fields in the real world, such
as finance, traffic, and weather. In recent years, many research efforts have
been proposed for forecasting multivariate time series. Although some previous
work considers the interdependencies among different variables in the same
timestamp, existing work overlooks the inter-connections between different
variables at different time stamps. In this paper, we propose a simple yet
efficient instance-wise graph-based framework to utilize the inter-dependencies
of different variables at different time stamps for multivariate time series
forecasting. The key idea of our framework is aggregating information from the
historical time series of different variables to the current time series that
we need to forecast. We conduct experiments on the Traffic, Electricity, and
Exchange-Rate multivariate time series datasets. The results show that our
proposed model outperforms the state-of-the-art baseline methods.

    

### [[2109.06514] Vision Transformer for Learning Driving Policies in Complex Multi-Agent Environments](http://arxiv.org/abs/2109.06514)


  Driving in a complex urban environment is a difficult task that requires a
complex decision policy. In order to make informed decisions, one needs to gain
an understanding of the long-range context and the importance of other
vehicles. In this work, we propose to use Vision Transformer (ViT) to learn a
driving policy in urban settings with birds-eye-view (BEV) input images. The
ViT network learns the global context of the scene more effectively than with
earlier proposed Convolutional Neural Networks (ConvNets). Furthermore, ViT's
attention mechanism helps to learn an attention map for the scene which allows
the ego car to determine which surrounding cars are important to its next
decision. We demonstrate that a DQN agent with a ViT backbone outperforms
baseline algorithms with ConvNet backbones pre-trained in various ways. In
particular, the proposed method helps reinforcement learning algorithms to
learn faster, with increased performance and less data than baselines.

    

### [[2109.06562] Anomaly Attribution of Multivariate Time Series using Counterfactual Reasoning](http://arxiv.org/abs/2109.06562)


  There are numerous methods for detecting anomalies in time series, but that
is only the first step to understanding them. We strive to exceed this by
explaining those anomalies. Thus we develop a novel attribution scheme for
multivariate time series relying on counterfactual reasoning. We aim to answer
the counterfactual question of would the anomalous event have occurred if the
subset of the involved variables had been more similarly distributed to the
data outside of the anomalous interval. Specifically, we detect anomalous
intervals using the Maximally Divergent Interval (MDI) algorithm, replace a
subset of variables with their in-distribution values within the detected
interval and observe if the interval has become less anomalous, by re-scoring
it with MDI. We evaluate our method on multivariate temporal and
spatio-temporal data and confirm the accuracy of our anomaly attribution of
multiple well-understood extreme climate events such as heatwaves and
hurricanes.

    

### [[2109.06565] Variation-Incentive Loss Re-weighting for Regression Analysis on Biased Data](http://arxiv.org/abs/2109.06565)


  Both classification and regression tasks are susceptible to the biased
distribution of training data. However, existing approaches are focused on the
class-imbalanced learning and cannot be applied to the problems of numerical
regression where the learning targets are continuous values rather than
discrete labels. In this paper, we aim to improve the accuracy of the
regression analysis by addressing the data skewness/bias during model training.
We first introduce two metrics, uniqueness and abnormality, to reflect the
localized data distribution from the perspectives of their feature (i.e.,
input) space and target (i.e., output) space. Combining these two metrics we
propose a Variation-Incentive Loss re-weighting method (VILoss) to optimize the
gradient descent-based model training for regression analysis. We have
conducted comprehensive experiments on both synthetic and real-world data sets.
The results show significant improvement in the model quality (reduction in
error by up to 11.9%) when using VILoss as the loss criterion in training.

    

### [[2109.06579] Bayesian AirComp with Sign-Alignment Precoding for Wireless Federated Learning](http://arxiv.org/abs/2109.06579)


  In this paper, we consider the problem of wireless federated learning based
on sign stochastic gradient descent (signSGD) algorithm via a multiple access
channel. When sending locally computed gradient's sign information, each mobile
device requires to apply precoding to circumvent wireless fading effects. In
practice, however, acquiring perfect knowledge of channel state information
(CSI) at all mobile devices is infeasible. In this paper, we present a simple
yet effective precoding method with limited channel knowledge, called
sign-alignment precoding. The idea of sign-alignment precoding is to protect
sign-flipping errors from wireless fadings. Under the Gaussian prior assumption
on the local gradients, we also derive the mean squared error (MSE)-optimal
aggregation function called Bayesian over-the-air computation (BayAirComp). Our
key finding is that one-bit precoding with BayAirComp aggregation can provide a
better learning performance than the existing precoding method even using
perfect CSI with AirComp aggregation.

    

### [[2109.06587] Sum-Product-Attention Networks: Leveraging Self-Attention in Probabilistic Circuits](http://arxiv.org/abs/2109.06587)


  Probabilistic circuits (PCs) have become the de-facto standard for learning
and inference in probabilistic modeling. We introduce Sum-Product-Attention
Networks (SPAN), a new generative model that integrates probabilistic circuits
with Transformers. SPAN uses self-attention to select the most relevant parts
of a probabilistic circuit, here sum-product networks, to improve the modeling
capability of the underlying sum-product network. We show that while modeling,
SPAN focuses on a specific set of independent assumptions in every product
layer of the sum-product network. Our empirical evaluations show that SPAN
outperforms state-of-the-art probabilistic generative models on various
benchmark data sets as well is an efficient generative image model.

    

### [[2109.06604] Non-Parametric Unsupervised Domain Adaptation for Neural Machine Translation](http://arxiv.org/abs/2109.06604)


  Recently, $k$NN-MT has shown the promising capability of directly
incorporating the pre-trained neural machine translation (NMT) model with
domain-specific token-level $k$-nearest-neighbor ($k$NN) retrieval to achieve
domain adaptation without retraining. Despite being conceptually attractive, it
heavily relies on high-quality in-domain parallel corpora, limiting its
capability on unsupervised domain adaptation, where in-domain parallel corpora
are scarce or nonexistent. In this paper, we propose a novel framework that
directly uses in-domain monolingual sentences in the target language to
construct an effective datastore for $k$-nearest-neighbor retrieval. To this
end, we first introduce an autoencoder task based on the target language, and
then insert lightweight adapters into the original NMT model to map the
token-level representation of this task to the ideal representation of
translation task. Experiments on multi-domain datasets demonstrate that our
proposed approach significantly improves the translation accuracy with
target-side monolingual data, while achieving comparable performance with
back-translation.

    

### [[2109.06609] DSDF: An approach to handle stochastic agents in collaborative multi-agent reinforcement learning](http://arxiv.org/abs/2109.06609)


  Multi-Agent reinforcement learning has received lot of attention in recent
years and have applications in many different areas. Existing methods involving
Centralized Training and Decentralized execution, attempts to train the agents
towards learning a pattern of coordinated actions to arrive at optimal joint
policy. However if some agents are stochastic to varying degrees of
stochasticity, the above methods often fail to converge and provides poor
coordination among agents. In this paper we show how this stochasticity of
agents, which could be a result of malfunction or aging of robots, can add to
the uncertainty in coordination and there contribute to unsatisfactory global
coordination. In this case, the deterministic agents have to understand the
behavior and limitations of the stochastic agents while arriving at optimal
joint policy. Our solution, DSDF which tunes the discounted factor for the
agents according to uncertainty and use the values to update the utility
networks of individual agents. DSDF also helps in imparting an extent of
reliability in coordination thereby granting stochastic agents tasks which are
immediate and of shorter trajectory with deterministic ones taking the tasks
which involve longer planning. Such an method enables joint co-ordinations of
agents some of which may be partially performing and thereby can reduce or
delay the investment of agent/robot replacement in many circumstances. Results
on benchmark environment for different scenarios shows the efficacy of the
proposed approach when compared with existing approaches.

    

### [[2109.06610] Statistical limits of dictionary learning: random matrix theory and the spectral replica method](http://arxiv.org/abs/2109.06610)


  We consider increasingly complex models of matrix denoising and dictionary
learning in the Bayes-optimal setting, in the challenging regime where the
matrices to infer have a rank growing linearly with the system size. This is in
contrast with most existing literature concerned with the low-rank (i.e.,
constant-rank) regime. We first consider a class of rotationally invariant
matrix denoising problems whose mutual information and minimum mean-square
error are computable using standard techniques from random matrix theory. Next,
we analyze the more challenging models of dictionary learning. To do so we
introduce a novel combination of the replica method from statistical mechanics
together with random matrix theory, coined spectral replica method. It allows
us to conjecture variational formulas for the mutual information between hidden
representations and the noisy data as well as for the overlaps quantifying the
optimal reconstruction error. The proposed methods reduce the number of degrees
of freedom from $\Theta(N^2)$ (matrix entries) to $\Theta(N)$ (eigenvalues or
singular values), and yield Coulomb gas representations of the mutual
information which are reminiscent of matrix models in physics. The main
ingredients are the use of HarishChandra-Itzykson-Zuber spherical integrals
combined with a new replica symmetric decoupling ansatz at the level of the
probability distributions of eigenvalues (or singular values) of certain
overlap matrices.

    

### [[2109.06625] Towards optimized actions in critical situations of soccer games with deep reinforcement learning](http://arxiv.org/abs/2109.06625)


  Soccer is a sparse rewarding game: any smart or careless action in critical
situations can change the result of the match. Therefore players, coaches, and
scouts are all curious about the best action to be performed in critical
situations, such as the times with a high probability of losing ball possession
or scoring a goal. This work proposes a new state representation for the soccer
game and a batch reinforcement learning to train a smart policy network. This
network gets the contextual information of the situation and proposes the
optimal action to maximize the expected goal for the team. We performed
extensive numerical experiments on the soccer logs made by InStat for 104
European soccer matches. The results show that in all 104 games, the optimized
policy obtains higher rewards than its counterpart in the behavior policy.
Besides, our framework learns policies that are close to the expected behavior
in the real world. For instance, in the optimized policy, we observe that some
actions such as foul, or ball out can be sometimes more rewarding than a shot
in specific situations.

    

### [[2109.06627] Scalable Font Reconstruction with Dual Latent Manifolds](http://arxiv.org/abs/2109.06627)


  We propose a deep generative model that performs typography analysis and font
reconstruction by learning disentangled manifolds of both font style and
character shape. Our approach enables us to massively scale up the number of
character types we can effectively model compared to previous methods.
Specifically, we infer separate latent variables representing character and
font via a pair of inference networks which take as input sets of glyphs that
either all share a character type, or belong to the same font. This design
allows our model to generalize to characters that were not observed during
training time, an important task in light of the relative sparsity of most
fonts. We also put forward a new loss, adapted from prior work that measures
likelihood using an adaptive distribution in a projected space, resulting in
more natural images without requiring a discriminator. We evaluate on the task
of font reconstruction over various datasets representing character types of
many languages, and compare favorably to modern style transfer systems
according to both automatic and manually-evaluated metrics.

    

### [[2109.06635] Deep Convolutional Generative Modeling for Artificial Microstructure Development of Aluminum-Silicon Alloy](http://arxiv.org/abs/2109.06635)


  Machine learning which is a sub-domain of an Artificial Intelligence which is
finding various applications in manufacturing and material science sectors. In
the present study, Deep Generative Modeling which a type of unsupervised
machine learning technique has been adapted for the constructing the artificial
microstructure of Aluminium-Silicon alloy. Deep Generative Adversarial Networks
has been used for developing the artificial microstructure of the given
microstructure image dataset. The results obtained showed that the developed
models had learnt to replicate the lining near the certain images of the
microstructures.

    

### [[2109.06638] Learnable Discrete Wavelet Pooling (LDW-Pooling) For Convolutional Networks](http://arxiv.org/abs/2109.06638)


  Pooling is a simple but essential layer in modern deep CNN architectures for
feature aggregation and extraction. Typical CNN design focuses on the conv
layers and activation functions, while leaving the pooling layers with fewer
options. We introduce the Learning Discrete Wavelet Pooling (LDW-Pooling) that
can be applied universally to replace standard pooling operations to better
extract features with improved accuracy and efficiency. Motivated from the
wavelet theory, we adopt the low-pass (L) and high-pass (H) filters
horizontally and vertically for pooling on a 2D feature map. Feature signals
are decomposed into four (LL, LH, HL, HH) subbands to retain features better
and avoid information dropping. The wavelet transform ensures features after
pooling can be fully preserved and recovered. We next adopt an energy-based
attention learning to fine-select crucial and representative features.
LDW-Pooling is effective and efficient when compared with other
state-of-the-art pooling techniques such as WaveletPooling and LiftPooling.
Extensive experimental validation shows that LDW-Pooling can be applied to a
wide range of standard CNN architectures and consistently outperform standard
(max, mean, mixed, and stochastic) pooling operations.

    

### [[2109.06661] Expert Knowledge-Guided Length-Variant Hierarchical Label Generation for Proposal Classification](http://arxiv.org/abs/2109.06661)


  To advance the development of science and technology, research proposals are
submitted to open-court competitive programs developed by government agencies
(e.g., NSF). Proposal classification is one of the most important tasks to
achieve effective and fair review assignments. Proposal classification aims to
classify a proposal into a length-variant sequence of labels. In this paper, we
formulate the proposal classification problem into a hierarchical multi-label
classification task. Although there are certain prior studies, proposal
classification exhibit unique features: 1) the classification result of a
proposal is in a hierarchical discipline structure with different levels of
granularity; 2) proposals contain multiple types of documents; 3) domain
experts can empirically provide partial labels that can be leveraged to improve
task performances. In this paper, we focus on developing a new deep proposal
classification framework to jointly model the three features. In particular, to
sequentially generate labels, we leverage previously-generated labels to
predict the label of next level; to integrate partial labels from experts, we
use the embedding of these empirical partial labels to initialize the state of
neural networks. Our model can automatically identify the best length of label
sequence to stop next label prediction. Finally, we present extensive results
to demonstrate that our method can jointly model partial labels, textual
information, and semantic dependencies in label sequences, and, thus, achieve
advanced performances.

    

### [[2109.06662] Identifying partial mouse brain microscopy images from Allen reference atlas using a contrastively learned semantic space](http://arxiv.org/abs/2109.06662)


  Precise identification of mouse brain microscopy images is a crucial first
step when anatomical structures in the mouse brain are to be registered to a
reference atlas. Practitioners usually rely on manual comparison of images or
tools that assume the presence of complete images. This work explores Siamese
Networks as the method for finding corresponding 2D reference atlas plates for
given partial 2D mouse brain images. Siamese networks are a class of
convolutional neural networks (CNNs) that use weight-shared paths to obtain low
dimensional embeddings of pairs of input images. The correspondence between the
partial mouse brain image and reference atlas plate is determined based on the
distance between low dimensional embeddings of brain slices and atlas plates
that are obtained from Siamese networks using contrastive learning. Experiments
showed that Siamese CNNs can precisely identify brain slices using the Allen
mouse brain atlas when training and testing images come from the same source.
They achieved TOP-1 and TOP-5 accuracy of 25% and 100%, respectively, taking
only 7.2 seconds to identify 29 images.

    

### [[2109.06663] An Insect-Inspired Randomly, Weighted Neural Network with Random Fourier Features For Neuro-Symbolic Relational Learning](http://arxiv.org/abs/2109.06663)


  Insects, such as fruit flies and honey bees, can solve simple associative
learning tasks and learn abstract concepts such as "sameness" and "difference",
which is viewed as a higher-order cognitive function and typically thought to
depend on top-down neocortical processing. Empirical research with fruit flies
strongly supports that a randomized representational architecture is used in
olfactory processing in insect brains. Based on these results, we propose a
Randomly Weighted Feature Network (RWFN) that incorporates randomly drawn,
untrained weights in an encoder that uses an adapted linear model as a decoder.
The randomized projections between input neurons and higher-order processing
centers in the input brain is mimicked in RWFN by a single-hidden-layer neural
network that specially structures latent representations in the hidden layer
using random Fourier features that better represent complex relationships
between inputs using kernel approximation. Because of this special
representation, RWFNs can effectively learn the degree of relationship among
inputs by training only a linear decoder model. We compare the performance of
RWFNs to LTNs for Semantic Image Interpretation (SII) tasks that have been used
as a representative example of how LTNs utilize reasoning over first-order
logic to surpass the performance of solely data-driven methods. We demonstrate
that compared to LTNs, RWFNs can achieve better or similar performance for both
object classification and detection of the part-of relations between objects in
SII tasks while using much far fewer learnable parameters (1:62 ratio) and a
faster learning process (1:2 ratio of running speed). Furthermore, we show that
because the randomized weights do not depend on the data, several decoders can
share a single randomized encoder, giving RWFNs a unique economy of spatial
scale for simultaneous classification tasks.

    

### [[2109.06668] Exploration in Deep Reinforcement Learning: A Comprehensive Survey](http://arxiv.org/abs/2109.06668)


  Deep Reinforcement Learning (DRL) and Deep Multi-agent Reinforcement Learning
(MARL) have achieved significant success across a wide range of domains, such
as game AI, autonomous vehicles, robotics and finance. However, DRL and deep
MARL agents are widely known to be sample-inefficient and millions of
interactions are usually needed even for relatively simple game settings, thus
preventing the wide application in real-industry scenarios. One bottleneck
challenge behind is the well-known exploration problem, i.e., how to
efficiently explore the unknown environments and collect informative
experiences that could benefit the policy learning most.
In this paper, we conduct a comprehensive survey on existing exploration
methods in DRL and deep MARL for the purpose of providing understandings and
insights on the critical problems and solutions. We first identify several key
challenges to achieve efficient exploration, which most of the exploration
methods aim at addressing. Then we provide a systematic survey of existing
approaches by classifying them into two major categories: uncertainty-oriented
exploration and intrinsic motivation-oriented exploration. The essence of
uncertainty-oriented exploration is to leverage the quantification of the
epistemic and aleatoric uncertainty to derive efficient exploration. By
contrast, intrinsic motivation-oriented exploration methods usually incorporate
different reward agnostic information for intrinsic exploration guidance.
Beyond the above two main branches, we also conclude other exploration methods
which adopt sophisticated techniques but are difficult to be classified into
the above two categories. In addition, we provide a comprehensive empirical
comparison of exploration methods for DRL on a set of commonly used benchmarks.
Finally, we summarize the open problems of exploration in DRL and deep MARL and
point out a few future directions.

    

### [[2109.06677] Specified Certainty Classification, with Application to Read Classification for Reference-Guided Metagenomic Assembly](http://arxiv.org/abs/2109.06677)


  Specified Certainty Classification (SCC) is a new paradigm for employing
classifiers whose outputs carry uncertainties, typically in the form of
Bayesian posterior probabilities. By allowing the classifier output to be less
precise than one of a set of atomic decisions, SCC allows all decisions to
achieve a specified level of certainty, as well as provides insights into
classifier behavior by examining all decisions that are possible. Our primary
illustration is read classification for reference-guided genome assembly, but
we demonstrate the breadth of SCC by also analyzing COVID-19 vaccination data.

    

### [[2109.06689] Reactive and Safe Road User Simulations using Neural Barrier Certificates](http://arxiv.org/abs/2109.06689)


  Reactive and safe agent modelings are important for nowadays traffic
simulator designs and safe planning applications. In this work, we proposed a
reactive agent model which can ensure safety without comprising the original
purposes, by learning only high-level decisions from expert data and a
low-level decentralized controller guided by the jointly learned decentralized
barrier certificates. Empirical results show that our learned road user
simulation models can achieve a significant improvement in safety comparing to
state-of-the-art imitation learning and pure control-based methods, while being
similar to human agents by having smaller errors to the expert data. Moreover,
our learned reactive agents are shown to generalize better to unseen traffic
conditions, and react better to other road users and therefore can help
understand challenging planning problems pragmatically.

    

### [[2109.06692] LRWR: Large-Scale Benchmark for Lip Reading in Russian language](http://arxiv.org/abs/2109.06692)


  Lipreading, also known as visual speech recognition, aims to identify the
speech content from videos by analyzing the visual deformations of lips and
nearby areas. One of the significant obstacles for research in this field is
the lack of proper datasets for a wide variety of languages: so far, these
methods have been focused only on English or Chinese. In this paper, we
introduce a naturally distributed large-scale benchmark for lipreading in
Russian language, named LRWR, which contains 235 classes and 135 speakers. We
provide a detailed description of the dataset collection pipeline and dataset
statistics. We also present a comprehensive comparison of the current popular
lipreading methods on LRWR and conduct a detailed analysis of their
performance. The results demonstrate the differences between the benchmarked
languages and provide several promising directions for lipreading models
finetuning. Thanks to our findings, we also achieved new state-of-the-art
results on the LRW benchmark.

    

### [[2109.06699] An Apparatus for the Simulation of Breathing Disorders: Physically Meaningful Generation of Surrogate Data](http://arxiv.org/abs/2109.06699)


  Whilst debilitating breathing disorders, such as chronic obstructive
pulmonary disease (COPD), are rapidly increasing in prevalence, we witness a
continued integration of artificial intelligence into healthcare. While this
promises improved detection and monitoring of breathing disorders, AI
techniques are "data hungry" which highlights the importance of generating
physically meaningful surrogate data. Such domain knowledge aware surrogates
would enable both an improved understanding of respiratory waveform changes
with different breathing disorders and different severities, and enhance the
training of machine learning algorithms. To this end, we introduce an apparatus
comprising of PVC tubes and 3D printed parts as a simple yet effective method
of simulating both obstructive and restrictive respiratory waveforms in healthy
subjects. Independent control over both inspiratory and expiratory resistances
allows for the simulation of obstructive breathing disorders through the whole
spectrum of FEV1/FVC spirometry ratios (used to classify COPD), ranging from
healthy values to values seen in severe chronic obstructive pulmonary disease.
Moreover, waveform characteristics of breathing disorders, such as a change in
inspiratory duty cycle or peak flow are also observed in the waveforms
resulting from use of the artificial breathing disorder simulation apparatus.
Overall, the proposed apparatus provides us with a simple, effective and
physically meaningful way to generate surrogate breathing disorder waveforms, a
prerequisite for the use of artificial intelligence in respiratory health.

    

### [[2109.06700] Neural Upscaling from Residue-level Protein Structure Networks to Atomistic Structure](http://arxiv.org/abs/2109.06700)


  Coarse-graining is a powerful tool for extending the reach of dynamic models
of proteins and other biological macromolecules. Topological coarse-graining,
in which biomolecules or sets thereof are represented via graph structures, is
a particularly useful way of obtaining highly compressed representations of
molecular structure, and simulations operating via such representations can
achieve substantial computational savings. A drawback of coarse-graining,
however, is the loss of atomistic detail - an effect that is especially acute
for topological representations such as protein structure networks (PSNs).
Here, we introduce an approach based on a combination of machine learning and
physically-guided refinement for inferring atomic coordinates from PSNs. This
"neural upscaling" procedure exploits the constraints implied by PSNs on
possible configurations, as well as differences in the likelihood of observing
different configurations with the same PSN. Using a 1 $\mu$s atomistic
molecular dynamics trajectory of A$\beta_{1-40}$, we show that neural upscaling
is able to effectively recapitulate detailed structural information for
intrinsically disordered proteins, being particularly successful in recovering
features such as transient secondary structure. These results suggest that
scalable network-based models for protein structure and dynamics may be used in
settings where atomistic detail is desired, with upscaling employed to impute
atomic coordinates from PSNs.

    

### [[2109.06707] A pragmatic approach to estimating average treatment effects from EHR data: the effect of prone positioning on mechanically ventilated COVID-19 patients](http://arxiv.org/abs/2109.06707)


  Despite the recent progress in the field of causal inference, to date there
is no agreed upon methodology to glean treatment effect estimation from
observational data. The consequence on clinical practice is that, when lacking
results from a randomized trial, medical personnel is left without guidance on
what seems to be effective in a real-world scenario. This article showcases a
pragmatic methodology to obtain preliminary estimation of treatment effect from
observational studies. Our approach was tested on the estimation of treatment
effect of the proning maneuver on oxygenation levels, on a cohort of COVID-19
Intensive Care patients. We modeled our study design on a recent RCT for
proning (the PROSEVA trial). Linear regression, propensity score models such as
blocking and DR-IPW, BART and two versions of Counterfactual Regression were
employed to provide estimates on observational data comprising first wave
COVID-19 ICU patient data from 25 Dutch hospitals. 6371 data points, from 745
mechanically ventilated patients, were included in the study. Estimates for the
early effect of proning -- P/F ratio from 2 to 8 hours after proning -- ranged
between 14.54 and 20.11 mm Hg depending on the model. Estimates for the late
effect of proning -- oxygenation from 12 to 24 hours after proning -- ranged
between 13.53 and 15.26 mm Hg. All confidence interval being strictly above
zero indicated that the effect of proning on oxygenation for COVID-19 patient
was positive and comparable in magnitude to the effect on non COVID-19
patients. These results provide further evidence on the effectiveness of
proning on the treatment of COVID-19 patients. This study, along with the
accompanying open-source code, provides a blueprint for treatment effect
estimation in scenarios where RCT data is lacking. Funding: SIDN fund,
CovidPredict consortium, Pacmed.

    

### [[2109.06710] Fast Federated Edge Learning with Overlapped Communication and Computation and Channel-Aware Fair Client Scheduling](http://arxiv.org/abs/2109.06710)


  We consider federated edge learning (FEEL) over wireless fading channels
taking into account the downlink and uplink channel latencies, and the random
computation delays at the clients. We speed up the training process by
overlapping the communication with computation. With fountain coded
transmission of the global model update, clients receive the global model
asynchronously, and start performing local computations right away. Then, we
propose a dynamic client scheduling policy, called MRTP, for uploading local
model updates to the parameter server (PS), which, at any time, schedules the
client with the minimum remaining upload time. However, MRTP can lead to biased
participation of clients in the update process, resulting in performance
degradation in non-iid data scenarios. To overcome this, we propose two
alternative schemes with fairness considerations, termed as age-aware MRTP
(A-MRTP), and opportunistically fair MRTP (OF-MRTP). In A-MRTP, the remaining
clients are scheduled according to the ratio between their remaining
transmission time and the update age, while in OF-MRTP, the selection mechanism
utilizes the long term average channel rate of the clients to further reduce
the latency while ensuring fair participation of the clients. It is shown
through numerical simulations that OF-MRTP provides significant reduction in
latency without sacrificing test accuracy.

    

### [[2109.06711] COVID-Net Clinical ICU: Enhanced Prediction of ICU Admission for COVID-19 Patients via Explainability and Trust Quantification](http://arxiv.org/abs/2109.06711)


  The COVID-19 pandemic continues to have a devastating global impact, and has
placed a tremendous burden on struggling healthcare systems around the world.
Given the limited resources, accurate patient triaging and care planning is
critical in the fight against COVID-19, and one crucial task within care
planning is determining if a patient should be admitted to a hospital's
intensive care unit (ICU). Motivated by the need for transparent and
trustworthy ICU admission clinical decision support, we introduce COVID-Net
Clinical ICU, a neural network for ICU admission prediction based on patient
clinical data. Driven by a transparent, trust-centric methodology, the proposed
COVID-Net Clinical ICU was built using a clinical dataset from Hospital
Sirio-Libanes comprising of 1,925 COVID-19 patients, and is able to predict
when a COVID-19 positive patient would require ICU admission with an accuracy
of 96.9% to facilitate better care planning for hospitals amidst the on-going
pandemic. We conducted system-level insight discovery using a quantitative
explainability strategy to study the decision-making impact of different
clinical features and gain actionable insights for enhancing predictive
performance. We further leveraged a suite of trust quantification metrics to
gain deeper insights into the trustworthiness of COVID-Net Clinical ICU. By
digging deeper into when and why clinical predictive models makes certain
decisions, we can uncover key factors in decision making for critical clinical
decision support tasks such as ICU admission prediction and identify the
situations under which clinical predictive models can be trusted for greater
accountability.

    

### [[2109.06713] Machine-Learned Prediction Equilibrium for Dynamic Traffic Assignment](http://arxiv.org/abs/2109.06713)


  We study a dynamic traffic assignment model, where agents base their
instantaneous routing decisions on real-time delay predictions. We formulate a
mathematically concise model and derive properties of the predictors that
ensure a dynamic prediction equilibrium exists. We demonstrate the versatility
of our framework by showing that it subsumes the well-known full information
and instantaneous information models, in addition to admitting further
realistic predictors as special cases. We complement our theoretical analysis
by an experimental study, in which we systematically compare the induced
average travel times of different predictors, including a machine-learning
model trained on data gained from previously computed equilibrium flows, both
on a synthetic and a real road network.

    

### [[2109.06715] IGNNITION: Bridging the Gap Between Graph Neural Networks and Networking Systems](http://arxiv.org/abs/2109.06715)


  Recent years have seen the vast potential of Graph Neural Networks (GNN) in
many fields where data is structured as graphs (e.g., chemistry, recommender
systems). In particular, GNNs are becoming increasingly popular in the field of
networking, as graphs are intrinsically present at many levels (e.g., topology,
routing). The main novelty of GNNs is their ability to generalize to other
networks unseen during training, which is an essential feature for developing
practical Machine Learning (ML) solutions for networking. However, implementing
a functional GNN prototype is currently a cumbersome task that requires strong
skills in neural network programming. This poses an important barrier to
network engineers that often do not have the necessary ML expertise. In this
article, we present IGNNITION, a novel open-source framework that enables fast
prototyping of GNNs for networking systems. IGNNITION is based on an intuitive
high-level abstraction that hides the complexity behind GNNs, while still
offering great flexibility to build custom GNN architectures. To showcase the
versatility and performance of this framework, we implement two
state-of-the-art GNN models applied to different networking use cases. Our
results show that the GNN models produced by IGNNITION are equivalent in terms
of accuracy and performance to their native implementations in TensorFlow.

    

### [[2109.06716] HPOBench: A Collection of Reproducible Multi-Fidelity Benchmark Problems for HPO](http://arxiv.org/abs/2109.06716)


  To achieve peak predictive performance, hyperparameter optimization (HPO) is
a crucial component of machine learning and its applications. Over the last
years,the number of efficient algorithms and tools for HPO grew substantially.
At the same time, the community is still lacking realistic, diverse,
computationally cheap,and standardized benchmarks. This is especially the case
for multi-fidelity HPO methods. To close this gap, we propose HPOBench, which
includes 7 existing and 5 new benchmark families, with in total more than 100
multi-fidelity benchmark problems. HPOBench allows to run this extendable set
of multi-fidelity HPO benchmarks in a reproducible way by isolating and
packaging the individual benchmarks in containers. It also provides surrogate
and tabular benchmarks for computationally affordable yet statistically sound
evaluations. To demonstrate the broad compatibility of HPOBench and its
usefulness, we conduct an exemplary large-scale study evaluating 6 well known
multi-fidelity HPO tools.

    

### [[2109.06723] Simulations in Recommender Systems: An industry perspective](http://arxiv.org/abs/2109.06723)


  The construction of effective Recommender Systems (RS) is a complex process,
mainly due to the nature of RSs which involves large scale software-systems and
human interactions. Iterative development processes require deep understanding
of a current baseline as well as the ability to estimate the impact of changes
in multiple variables of interest. Simulations are well suited to address both
challenges and potentially leading to a high velocity construction process, a
fundamental requirement in commercial contexts. Recently, there has been
significant interest in RS Simulation Platforms, which allow RS developers to
easily craft simulated environments where their systems can be analysed. In
this work we discuss how simulations help to increase velocity, we look at the
literature around RS Simulation Platforms, analyse strengths and gaps and
distill a set of guiding principles for the design of RS Simulation Platforms
that we believe will maximize the velocity of iterative RS construction
processes.

    

### [[2109.06728] Learning Density Distribution of Reachable States for Autonomous Systems](http://arxiv.org/abs/2109.06728)


  State density distribution, in contrast to worst-case reachability, can be
leveraged for safety-related problems to better quantify the likelihood of the
risk for potentially hazardous situations. In this work, we propose a
data-driven method to compute the density distribution of reachable states for
nonlinear and even black-box systems. Our semi-supervised approach learns
system dynamics and the state density jointly from trajectory data, guided by
the fact that the state density evolution follows the Liouville partial
differential equation. With the help of neural network reachability tools, our
approach can estimate the set of all possible future states as well as their
density. Moreover, we could perform online safety verification with probability
ranges for unsafe behaviors to occur. We use an extensive set of experiments to
show that our learned solution can produce a much more accurate estimate on
density distribution, and can quantify risks less conservatively and flexibly
comparing with worst-case analysis.

    

### [[2109.06732] Tuna-AI: tuna biomass estimation with Machine Learning models trained on oceanography and echosounder FAD data](http://arxiv.org/abs/2109.06732)


  Echo-sounder data registered by buoys attached to drifting FADs provide a
very valuablesource of information on populations of tuna and their behaviour.
This value increases whenthese data are supplemented with oceanographic data
coming from CMEMS. We use thesesources to develop Tuna-AI, a Machine Learning
model aimed at predicting tuna biomassunder a given buoy, which uses a 3-day
window of echo-sounder data to capture the dailyspatio-temporal patterns
characteristic of tuna schools. As the supervised signal for training,we employ
more than5000set events with their corresponding tuna catch reported by theAGAC
tuna purse seine fleet.

    

### [[2109.06736] Sequential Modelling with Applications to Music Recommendation, Fact-Checking, and Speed Reading](http://arxiv.org/abs/2109.06736)


  Sequential modelling entails making sense of sequential data, which naturally
occurs in a wide array of domains. One example is systems that interact with
users, log user actions and behaviour, and make recommendations of items of
potential interest to users on the basis of their previous interactions. In
such cases, the sequential order of user interactions is often indicative of
what the user is interested in next. Similarly, for systems that automatically
infer the semantics of text, capturing the sequential order of words in a
sentence is essential, as even a slight re-ordering could significantly alter
its original meaning. This thesis makes methodological contributions and new
investigations of sequential modelling for the specific application areas of
systems that recommend music tracks to listeners and systems that process text
semantics in order to automatically fact-check claims, or "speed read" text for
efficient further classification. (Rest of abstract omitted due to arXiv
abstract limit)

    

### [[2109.06737] Comparing Reconstruction- and Contrastive-based Models for Visual Task Planning](http://arxiv.org/abs/2109.06737)


  Learning state representations enables robotic planning directly from raw
observations such as images. Most methods learn state representations by
utilizing losses based on the reconstruction of the raw observations from a
lower-dimensional latent space. The similarity between observations in the
space of images is often assumed and used as a proxy for estimating similarity
between the underlying states of the system. However, observations commonly
contain task-irrelevant factors of variation which are nonetheless important
for reconstruction, such as varying lighting and different camera viewpoints.
In this work, we define relevant evaluation metrics and perform a thorough
study of different loss functions for state representation learning. We show
that models exploiting task priors, such as Siamese networks with a simple
contrastive loss, outperform reconstruction-based representations in visual
task planning.

    

### [[2109.06756] ImUnity: a generalizable VAE-GAN solution for multicenter MR image harmonization](http://arxiv.org/abs/2109.06756)


  ImUnity is an original deep-learning model designed for efficient and
flexible MR image harmonization. A VAE-GAN network, coupled with a confusion
module and an optional biological preservation module, uses multiple 2D-slices
taken from different anatomical locations in each subject of the training
database, as well as image contrast transformations for its self-supervised
training. It eventually generates 'corrected' MR images that can be used for
various multi-center population studies. Using 3 open source databases (ABIDE,
OASIS and SRPBS), which contain MR images from multiple acquisition scanner
types or vendors and a large range of subjects ages, we show that ImUnity: (1)
outperforms state-of-the-art methods in terms of quality of images generated
using traveling subjects; (2) removes sites or scanner biases while improving
patients classification; (3) harmonizes data coming from new sites or scanners
without the need for an additional fine-tuning and (4) allows the selection of
multiple MR reconstructed images according to the desired applications. Tested
here on T1-weighted images, ImUnity could be used to harmonize other types of
medical images.

    

### [[2109.06762] Greenformer: Factorization Toolkit for Efficient Deep Neural Networks](http://arxiv.org/abs/2109.06762)


  While the recent advances in deep neural networks (DNN) bring remarkable
success, the computational cost also increases considerably. In this paper, we
introduce Greenformer, a toolkit to accelerate the computation of neural
networks through matrix factorization while maintaining performance.
Greenformer can be easily applied with a single line of code to any DNN model.
Our experimental results show that Greenformer is effective for a wide range of
scenarios. We provide the showcase of Greenformer at
this https URL.

    

### [[2109.06777] PETGEN: Personalized Text Generation Attack on Deep Sequence Embedding-based Classification Models](http://arxiv.org/abs/2109.06777)


  \textit{What should a malicious user write next to fool a detection model?}
Identifying malicious users is critical to ensure the safety and integrity of
internet platforms. Several deep learning based detection models have been
created. However, malicious users can evade deep detection models by
manipulating their behavior, rendering these models of little use. The
vulnerability of such deep detection models against adversarial attacks is
unknown. Here we create a novel adversarial attack model against deep user
sequence embedding-based classification models, which use the sequence of user
posts to generate user embeddings and detect malicious users. In the attack,
the adversary generates a new post to fool the classifier. We propose a novel
end-to-end Personalized Text Generation Attack model, called \texttt{PETGEN},
that simultaneously reduces the efficacy of the detection model and generates
posts that have several key desirable properties. Specifically, \texttt{PETGEN}
generates posts that are personalized to the user's writing style, have
knowledge about a given target context, are aware of the user's historical
posts on the target context, and encapsulate the user's recent topical
interests. We conduct extensive experiments on two real-world datasets (Yelp
and Wikipedia, both with ground-truth of malicious users) to show that
\texttt{PETGEN} significantly reduces the performance of popular deep user
sequence embedding-based classification models. \texttt{PETGEN} outperforms
five attack baselines in terms of text quality and attack efficacy in both
white-box and black-box classifier settings. Overall, this work paves the path
towards the next generation of adversary-aware sequence classification models.

    

### [[2109.06780] Benchmarking the Spectrum of Agent Capabilities](http://arxiv.org/abs/2109.06780)


  Evaluating the general abilities of intelligent agents requires complex
simulation environments. Existing benchmarks typically evaluate only one narrow
task per environment, requiring researchers to perform expensive training runs
on many different environments. We introduce Crafter, an open world survival
game with visual inputs that evaluates a wide range of general abilities within
a single environment. Agents either learn from the provided reward signal or
through intrinsic objectives and are evaluated by semantically meaningful
achievements that can be unlocked during each episode, such as discovering
resources and crafting tools. Consistently unlocking all achievements requires
strong generalization, deep exploration, and long-term reasoning. We
experimentally verify that Crafter is of appropriate difficulty to drive future
research and provide baselines scores of reward agents and unsupervised agents.
Furthermore, we observe sophisticated behaviors emerging from maximizing the
reward signal, such as building tunnel systems, bridges, houses, and
plantations. We hope that Crafter will accelerate research progress by quickly
evaluating a wide spectrum of abilities.

    

### [[2109.06783] Learning to Navigate Intersections with Unsupervised Driver Trait Inference](http://arxiv.org/abs/2109.06783)


  Navigation through uncontrolled intersections is one of the key challenges
for autonomous vehicles. Identifying the subtle differences in hidden traits of
other drivers can bring significant benefits when navigating in such
environments. We propose an unsupervised method for inferring driver traits
such as driving styles from observed vehicle trajectories. We use a variational
autoencoder with recurrent neural networks to learn a latent representation of
traits without any ground truth trait labels. Then, we use this trait
representation to learn a policy for an autonomous vehicle to navigate through
a T-intersection with deep reinforcement learning. Our pipeline enables the
autonomous vehicle to adjust its actions when dealing with drivers of different
traits to ensure safety and efficiency. Our method demonstrates promising
performance and outperforms state-of-the-art baselines in the T-intersection
scenario.

    

### [[2109.06786] Multiple shooting with neural differential equations](http://arxiv.org/abs/2109.06786)


  Neural differential equations have recently emerged as a flexible
data-driven/hybrid approach to model time-series data. This work experimentally
demonstrates that if the data contains oscillations, then standard fitting of a
neural differential equation may give flattened out trajectory that fails to
describe the data. We then introduce the multiple shooting method and present
successful demonstrations of this method for the fitting of a neural
differential equation to two datasets (synthetic and experimental) that the
standard approach fails to fit. Constraints introduced by multiple shooting can
be satisfied using a penalty or augmented Lagrangian method.

    

### [[2109.06795] ROMAX: Certifiably Robust Deep Multiagent Reinforcement Learning via Convex Relaxation](http://arxiv.org/abs/2109.06795)


  In a multirobot system, a number of cyber-physical attacks (e.g.,
communication hijack, observation perturbations) can challenge the robustness
of agents. This robustness issue worsens in multiagent reinforcement learning
because there exists the non-stationarity of the environment caused by
simultaneously learning agents whose changing policies affect the transition
and reward functions. In this paper, we propose a minimax MARL approach to
infer the worst-case policy update of other agents. As the minimax formulation
is computationally intractable to solve, we apply the convex relaxation of
neural networks to solve the inner minimization problem. Such convex relaxation
enables robustness in interacting with peer agents that may have significantly
different behaviors and also achieves a certified bound of the original
optimization problem. We evaluate our approach on multiple mixed
cooperative-competitive tasks and show that our method outperforms the previous
state of the art approaches on this topic.

    

### [[2109.06808] What are the attackers doing now? Automating cyber threat intelligence extraction from text on pace with the changing threat landscape: A survey](http://arxiv.org/abs/2109.06808)


  Cybersecurity researchers have contributed to the automated extraction of CTI
from textual sources, such as threat reports and online articles, where
cyberattack strategies, procedures, and tools are described. The goal of this
article is to aid cybersecurity researchers understand the current techniques
used for cyberthreat intelligence extraction from text through a survey of
relevant studies in the literature. We systematically collect "CTI extraction
from text"-related studies from the literature and categorize the CTI
extraction purposes. We propose a CTI extraction pipeline abstracted from these
studies. We identify the data sources, techniques, and CTI sharing formats
utilized in the context of the proposed pipeline. Our work finds ten types of
extraction purposes, such as extraction indicators of compromise extraction,
TTPs (tactics, techniques, procedures of attack), and cybersecurity keywords.
We also identify seven types of textual sources for CTI extraction, and textual
data obtained from hacker forums, threat reports, social media posts, and
online news articles have been used by almost 90% of the studies. Natural
language processing along with both supervised and unsupervised machine
learning techniques such as named entity recognition, topic modelling,
dependency parsing, supervised classification, and clustering are used for CTI
extraction. We observe the technical challenges associated with these studies
related to obtaining available clean, labelled data which could assure
replication, validation, and further extension of the studies. As we find the
studies focusing on CTI information extraction from text, we advocate for
building upon the current CTI extraction work to help cybersecurity
practitioners with proactive decision making such as threat prioritization,
automated threat modelling to utilize knowledge from past cybersecurity
incidents.

    

### [[2109.06815] Predicting Loss Risks for B2B Tendering Processes](http://arxiv.org/abs/2109.06815)


  Sellers and executives who maintain a bidding pipeline of sales engagements
with multiple clients for many opportunities significantly benefit from
data-driven insight into the health of each of their bids. There are many
predictive models that offer likelihood insights and win prediction modeling
for these opportunities. Currently, these win prediction models are in the form
of binary classification and only make a prediction for the likelihood of a win
or loss. The binary formulation is unable to offer any insight as to why a
particular deal might be predicted as a loss. This paper offers a multi-class
classification model to predict win probability, with the three loss classes
offering specific reasons as to why a loss is predicted, including no bid,
customer did not pursue, and lost to competition. These classes offer an
indicator of how that opportunity might be handled given the nature of the
prediction. Besides offering baseline results on the multi-class
classification, this paper also offers results on the model after class
imbalance handling, with the results achieving a high accuracy of 85% and an
average AUC score of 0.94.

    

### [[2109.06817] Automatic hippocampal surface generation via 3D U-net and active shape modeling with hybrid particle swarm optimization](http://arxiv.org/abs/2109.06817)


  In this paper, we proposed and validated a fully automatic pipeline for
hippocampal surface generation via 3D U-net coupled with active shape modeling
(ASM). Principally, the proposed pipeline consisted of three steps. In the
beginning, for each magnetic resonance image, a 3D U-net was employed to obtain
the automatic hippocampus segmentation at each hemisphere. Secondly, ASM was
performed on a group of pre-obtained template surfaces to generate mean shape
and shape variation parameters through principal component analysis.
Ultimately, hybrid particle swarm optimization was utilized to search for the
optimal shape variation parameters that best match the segmentation. The
hippocampal surface was then generated from the mean shape and the shape
variation parameters. The proposed pipeline was observed to provide hippocampal
surfaces at both hemispheres with high accuracy, correct anatomical topology,
and sufficient smoothness.

    

### [[2109.06822] LM-Critic: Language Models for Unsupervised Grammatical Error Correction](http://arxiv.org/abs/2109.06822)


  Training a model for grammatical error correction (GEC) requires a set of
labeled ungrammatical / grammatical sentence pairs, but manually annotating
such pairs can be expensive. Recently, the Break-It-Fix-It (BIFI) framework has
demonstrated strong results on learning to repair a broken program without any
labeled examples, but this relies on a perfect critic (e.g., a compiler) that
returns whether an example is valid or not, which does not exist for the GEC
task. In this work, we show how to leverage a pretrained language model (LM) in
defining an LM-Critic, which judges a sentence to be grammatical if the LM
assigns it a higher probability than its local perturbations. We apply this
LM-Critic and BIFI along with a large set of unlabeled sentences to bootstrap
realistic ungrammatical / grammatical pairs for training a corrector. We
evaluate our approach on GEC datasets across multiple domains (CoNLL-2014,
BEA-2019, GMEG-wiki and GMEG-yahoo) and show that it outperforms existing
methods in both the unsupervised setting (+7.7 F0.5) and the supervised setting
(+0.5 F0.5).

    

### [[2109.06826] Few-shot Quality-Diversity Optimisation](http://arxiv.org/abs/2109.06826)


  In the past few years, a considerable amount of research has been dedicated
to the exploitation of previous learning experiences and the design of Few-shot
and Meta Learning approaches, in problem domains ranging from Computer Vision
to Reinforcement Learning based control. A notable exception, where to the best
of our knowledge, little to no effort has been made in this direction is
Quality-Diversity (QD) optimisation. QD methods have been shown to be effective
tools in dealing with deceptive minima and sparse rewards in Reinforcement
Learning. However, they remain costly due to their reliance on inherently
sample inefficient evolutionary processes. We show that, given examples from a
task distribution, information about the paths taken by optimisation in
parameter space can be leveraged to build a prior population, which when used
to initialise QD methods in unseen environments, allows for few-shot
adaptation. Our proposed method does not require backpropagation. It is simple
to implement and scale, and furthermore, it is agnostic to the underlying
models that are being trained. Experiments carried in both sparse and dense
reward settings using robotic manipulation and navigation benchmarks show that
it considerably reduces the number of generations that are required for QD
optimisation in these environments.

    

### [[2109.06827] Types of Out-of-Distribution Texts and How to Detect Them](http://arxiv.org/abs/2109.06827)


  Despite agreement on the importance of detecting out-of-distribution (OOD)
examples, there is little consensus on the formal definition of OOD examples
and how to best detect them. We categorize these examples by whether they
exhibit a background shift or a semantic shift, and find that the two major
approaches to OOD detection, model calibration and density estimation (language
modeling for text), have distinct behavior on these types of OOD data. Across
14 pairs of in-distribution and OOD English natural language understanding
datasets, we find that density estimation methods consistently beat calibration
methods in background shift settings, while performing worse in semantic shift
settings. In addition, we find that both methods generally fail to detect
examples from challenge data, highlighting a weak spot for current methods.
Since no single method works well across all settings, our results call for an
explicit definition of OOD examples when evaluating different detection
methods.

    

### [[2109.06849] A geometric perspective on functional outlier detection](http://arxiv.org/abs/2109.06849)


  We consider functional outlier detection from a geometric perspective,
specifically: for functional data sets drawn from a functional manifold which
is defined by the data's modes of variation in amplitude and phase. Based on
this manifold, we develop a conceptualization of functional outlier detection
that is more widely applicable and realistic than previously proposed. Our
theoretical and experimental analyses demonstrate several important advantages
of this perspective: It considerably improves theoretical understanding and
allows to describe and analyse complex functional outlier scenarios
consistently and in full generality, by differentiating between structurally
anomalous outlier data that are off-manifold and distributionally outlying data
that are on-manifold but at its margins. This improves practical feasibility of
functional outlier detection: We show that simple manifold learning methods can
be used to reliably infer and visualize the geometric structure of functional
data sets. We also show that standard outlier detection methods requiring
tabular data inputs can be applied to functional data very successfully by
simply using their vector-valued representations learned from manifold learning
methods as input features. Our experiments on synthetic and real data sets
demonstrate that this approach leads to outlier detection performances at least
on par with existing functional data-specific methods in a large variety of
settings, without the highly specialized, complex methodology and narrow domain
of application these methods often entail.

    

### [[2109.06856] Performance of a Markovian neural network versus dynamic programming on a fishing control problem](http://arxiv.org/abs/2109.06856)


  Fishing quotas are unpleasant but efficient to control the productivity of a
fishing site. A popular model has a stochastic differential equation for the
biomass on which a stochastic dynamic programming or a Hamilton-Jacobi-Bellman
algorithm can be used to find the stochastic control -- the fishing quota. We
compare the solutions obtained by dynamic programming against those obtained
with a neural network which preserves the Markov property of the solution. The
method is extended to a similar multi species model to check its robustness in
high dimension.

    

### [[2109.06859] One-Class Meta-Learning: Towards Generalizable Few-Shot Open-Set Classification](http://arxiv.org/abs/2109.06859)


  Real-world classification tasks are frequently required to work in an
open-set setting. This is especially challenging for few-shot learning problems
due to the small sample size for each known category, which prevents existing
open-set methods from working effectively; however, most multiclass few-shot
methods are limited to closed-set scenarios. In this work, we address the
problem of few-shot open-set classification by first proposing methods for
few-shot one-class classification and then extending them to few-shot
multiclass open-set classification. We introduce two independent few-shot
one-class classification methods: Meta Binary Cross-Entropy (Meta-BCE), which
learns a separate feature representation for one-class classification, and
One-Class Meta-Learning (OCML), which learns to generate one-class classifiers
given standard multiclass feature representation. Both methods can augment any
existing few-shot learning method without requiring retraining to work in a
few-shot multiclass open-set setting without degrading its closed-set
performance. We demonstrate the benefits and drawbacks of both methods in
different problem settings and evaluate them on three standard benchmark
datasets, miniImageNet, tieredImageNet, and Caltech-UCSD-Birds-200-2011, where
they surpass the state-of-the-art methods in the few-shot multiclass open-set
and few-shot one-class tasks.

    

### [[2109.06861] Nonlinearities in Steerable SO(2)-Equivariant CNNs](http://arxiv.org/abs/2109.06861)


  Invariance under symmetry is an important problem in machine learning. Our
paper looks specifically at equivariant neural networks where transformations
of inputs yield homomorphic transformations of outputs. Here, steerable CNNs
have emerged as the standard solution. An inherent problem of steerable
representations is that general nonlinear layers break equivariance, thus
restricting architectural choices. Our paper applies harmonic distortion
analysis to illuminate the effect of nonlinearities on Fourier representations
of SO(2). We develop a novel FFT-based algorithm for computing representations
of non-linearly transformed activations while maintaining band-limitation. It
yields exact equivariance for polynomial (approximations of) nonlinearities, as
well as approximate solutions with tunable accuracy for general functions. We
apply the approach to build a fully E(3)-equivariant network for sampled 3D
surface data. In experiments with 2D and 3D data, we obtain results that
compare favorably to the state-of-the-art in terms of accuracy while permitting
continuous symmetry and exact equivariance.

    

### [[2109.06870] Performance-Efficiency Trade-offs in Unsupervised Pre-training for Speech Recognition](http://arxiv.org/abs/2109.06870)


  This paper is a study of performance-efficiency trade-offs in pre-trained
models for automatic speech recognition (ASR). We focus on wav2vec 2.0, and
formalize several architecture designs that influence both the model
performance and its efficiency. Putting together all our observations, we
introduce SEW (Squeezed and Efficient Wav2vec), a pre-trained model
architecture with significant improvements along both performance and
efficiency dimensions across a variety of training setups. For example, under
the 100h-960h semi-supervised setup on LibriSpeech, SEW achieves a 1.9x
inference speedup compared to wav2vec 2.0, with a 13.5% relative reduction in
word error rate. With a similar inference time, SEW reduces word error rate by
25-50% across different model sizes.

    

### [[1906.11632] A Survey on GANs for Anomaly Detection](http://arxiv.org/abs/1906.11632)


  Anomaly detection is a significant problem faced in several research areas.
Detecting and correctly classifying something unseen as anomalous is a
challenging problem that has been tackled in many different manners over the
years.
Generative Adversarial Networks (GANs) and the adversarial training process
have been recently employed to face this task yielding remarkable results. In
this paper we survey the principal GAN-based anomaly detection methods,
highlighting their pros and cons. Our contributions are the empirical
validation of the main GAN models for anomaly detection, the increase of the
experimental results on different datasets and the public release of a complete
Open Source toolbox for Anomaly Detection using GANs.

    

### [[1910.06511] PRS-Net: Planar Reflective Symmetry Detection Net for 3D Models](http://arxiv.org/abs/1910.06511)


  In geometry processing, symmetry is a universal type of high-level structural
information of 3D models and benefits many geometry processing tasks including
shape segmentation, alignment, matching, and completion. Thus it is an
important problem to analyze various symmetry forms of 3D shapes. Planar
reflective symmetry is the most fundamental one. Traditional methods based on
spatial sampling can be time-consuming and may not be able to identify all the
symmetry planes. In this paper, we present a novel learning framework to
automatically discover global planar reflective symmetry of a 3D shape. Our
framework trains an unsupervised 3D convolutional neural network to extract
global model features and then outputs possible global symmetry parameters,
where input shapes are represented using voxels. We introduce a dedicated
symmetry distance loss along with a regularization loss to avoid generating
duplicated symmetry planes. Our network can also identify generalized cylinders
by predicting their rotation axes. We further provide a method to remove
invalid and duplicated planes and axes. We demonstrate that our method is able
to produce reliable and accurate results. Our neural network based method is
hundreds of times faster than the state-of-the-art methods, which are based on
sampling. Our method is also robust even with noisy or incomplete input
surfaces.

    

### [[2002.00797] Stochastic geometry to generalize the Mondrian Process](http://arxiv.org/abs/2002.00797)


  The stable under iterated tessellation (STIT) process is a stochastic process
that produces a recursive partition of space with cut directions drawn
independently from a distribution over the sphere. The case of random
axis-aligned cuts is known as the Mondrian process. Random forests and Laplace
kernel approximations built from the Mondrian process have led to efficient
online learning methods and Bayesian optimization. In this work, we utilize
tools from stochastic geometry to resolve some fundamental questions concerning
STIT processes in machine learning. First, we show that a STIT process with cut
directions drawn from a discrete distribution can be efficiently simulated by
lifting to a higher dimensional axis-aligned Mondrian process. Second, we
characterize all possible kernels that stationary STIT processes and their
mixtures can approximate. We also give a uniform convergence rate for the
approximation error of the STIT kernels to the targeted kernels, generalizing
the work of [3] for the Mondrian case. Third, we obtain consistency results for
STIT forests in density estimation and regression. Finally, we give a formula
for the density estimator arising from an infinite STIT random forest. This
allows for precise comparisons between the Mondrian forest, the Mondrian kernel
and the Laplace kernel in density estimation. Our paper calls for further
developments at the novel intersection of stochastic geometry and machine
learning.

    

### [[2005.02205] When Machine Unlearning Jeopardizes Privacy](http://arxiv.org/abs/2005.02205)


  The right to be forgotten states that a data owner has the right to erase
their data from an entity storing it. In the context of machine learning (ML),
the right to be forgotten requires an ML model owner to remove the data owner's
data from the training set used to build the ML model, a process known as
machine unlearning. While originally designed to protect the privacy of the
data owner, we argue that machine unlearning may leave some imprint of the data
in the ML model and thus create unintended privacy risks. In this paper, we
perform the first study on investigating the unintended information leakage
caused by machine unlearning. We propose a novel membership inference attack
that leverages the different outputs of an ML model's two versions to infer
whether a target sample is part of the training set of the original model but
out of the training set of the corresponding unlearned model. Our experiments
demonstrate that the proposed membership inference attack achieves strong
performance. More importantly, we show that our attack in multiple cases
outperforms the classical membership inference attack on the original ML model,
which indicates that machine unlearning can have counterproductive effects on
privacy. We notice that the privacy degradation is especially significant for
well-generalized ML models where classical membership inference does not
perform well. We further investigate four mechanisms to mitigate the newly
discovered privacy risks and show that releasing the predicted label only,
temperature scaling, and differential privacy are effective. We believe that
our results can help improve privacy protection in practical implementations of
machine unlearning. Our code is available at
this https URL.

    

### [[2006.02420] Learning to Scan: A Deep Reinforcement Learning Approach for Personalized Scanning in CT Imaging](http://arxiv.org/abs/2006.02420)


  Computed Tomography (CT) takes X-ray measurements on the subjects to
reconstruct tomographic images. As X-ray is radioactive, it is desirable to
control the total amount of dose of X-ray for safety concerns. Therefore, we
can only select a limited number of measurement angles and assign each of them
limited amount of dose. Traditional methods such as compressed sensing usually
randomly select the angles and equally distribute the allowed dose on them. In
most CT reconstruction models, the emphasize is on designing effective image
representations, while much less emphasize is on improving the scanning
strategy. The simple scanning strategy of random angle selection and equal dose
distribution performs well in general, but they may not be ideal for each
individual subject. It is more desirable to design a personalized scanning
strategy for each subject to obtain better reconstruction result. In this
paper, we propose to use Reinforcement Learning (RL) to learn a personalized
scanning policy to select the angles and the dose at each chosen angle for each
individual subject. We first formulate the CT scanning process as an MDP, and
then use modern deep RL methods to solve it. The learned personalized scanning
strategy not only leads to better reconstruction results, but also shows strong
generalization to be combined with different reconstruction algorithms.

    

### [[2006.03151] Hidden Markov models as recurrent neural networks: an application to Alzheimer's disease](http://arxiv.org/abs/2006.03151)


  Hidden Markov models (HMMs) are commonly used for disease progression
modeling when the true patient health state is not fully known. Since HMMs
typically have multiple local optima, incorporating additional patient
covariates can improve parameter estimation and predictive performance. To
allow for this, we develop hidden Markov recurrent neural networks (HMRNNs), a
special case of recurrent neural networks that combine neural networks'
flexibility with HMMs' interpretability. The HMRNN can be reduced to a standard
HMM, with an identical likelihood function and parameter interpretations, but
can also be combined with other predictive neural networks that take patient
information as input. The HMRNN estimates all parameters simultaneously via
gradient descent. Using a dataset of Alzheimer's disease patients, we
demonstrate how the HMRNN can combine an HMM with other predictive neural
networks to improve disease forecasting and to offer a novel clinical
interpretation compared with a standard HMM trained via
expectation-maximization.

    

### [[2006.05021] CLAIMED: A CLAssification-Incorporated Minimum Energy Design to explore a multivariate response surface with feasibility constraints](http://arxiv.org/abs/2006.05021)


  Motivated by the problem of optimization of force-field systems in physics
using large-scale computer simulations, we consider exploration of a
deterministic complex multivariate response surface. The objective is to find
input combinations that generate output close to some desired or "target"
vector. In spite of reducing the problem to exploration of the input space with
respect to a one-dimensional loss function, the search is nontrivial and
challenging due to infeasible input combinations, high dimensionalities of the
input and output space and multiple "desirable" regions in the input space and
the difficulty of emulating the objective function well with a surrogate model.
We propose an approach that is based on combining machine learning techniques
with smart experimental design ideas to locate multiple good regions in the
input space.

    

### [[2008.08170] Accelerated Zeroth-Order and First-Order Momentum Methods from Mini to Minimax Optimization](http://arxiv.org/abs/2008.08170)


  In the paper, we propose a class of accelerated zeroth-order and first-order
momentum methods for both nonconvex mini-optimization and minimax-optimization.
Specifically, we propose a new accelerated zeroth-order momentum (Acc-ZOM)
method to solve stochastic mini-optimization problems. We prove that the
Acc-ZOM method achieves a lower query complexity of
$\tilde{O}(d^{3/4}\epsilon^{-3})$ for finding an $\epsilon$-stationary point,
which improves the best known result by a factor of $O(d^{1/4})$ where $d$
denotes the parameter dimension. In particular, the Acc-ZOM does not need large
batches that are required in the existing zeroth-order stochastic algorithms.
At the same time, we propose an accelerated zeroth-order momentum descent
ascent (Acc-ZOMDA) method for black-box minimax-optimization. We prove that the
Acc-ZOMDA method reaches the best known query complexity of
$\tilde{O}((d_1+d_2)\kappa_y^{3}\epsilon^{-3})$ without large batches for
finding an $\epsilon$-stationary point, where $d_1$ and $d_2$ denote dimensions
of optimization parameters and $\kappa_y$ is condition number. Moreover, we
propose an accelerated first-order momentum descent ascent (Acc-MDA) method for
solving white-box minimax problems, and prove that it achieves a lower gradient
complexity of $\tilde{O}(\kappa_y^{2.5}\epsilon^{-3})$ given batch size
$b=\kappa_y^{4}$ for finding an $\epsilon$-stationary point, which improves the
best known result by a factor of $O(\kappa_y^{1/2})$. Extensive experimental
results on the black-box adversarial attack to deep neural networks (DNNs) and
poisoning attack demonstrate the efficiency of our algorithms.

    

### [[2009.04550] Biclustering with Alternating K-Means](http://arxiv.org/abs/2009.04550)


  Biclustering is the task of simultaneously clustering the rows and columns of
the data matrix into different subgroups such that the rows and columns within
a subgroup exhibit similar patterns. In this paper, we consider the case of
producing block-diagonal biclusters. We provide a new formulation of the
biclustering problem based on the idea of minimizing the empirical clustering
risk. We develop and prove a consistency result with respect to the empirical
clustering risk. Since the optimization problem is combinatorial in nature,
finding the global minimum is computationally intractable. In light of this
fact, we propose a simple and novel algorithm that finds a local minimum by
alternating the use of an adapted version of the k-means clustering algorithm
between columns and rows. We evaluate and compare the performance of our
algorithm to other related biclustering methods on both simulated data and
real-world gene expression data sets. The results demonstrate that our
algorithm is able to detect meaningful structures in the data and outperform
other competing biclustering methods in various settings and situations.

    

### [[2009.06689] Online learning-based trajectory tracking for underactuated vehicles with uncertain dynamics](http://arxiv.org/abs/2009.06689)


  Underactuated vehicles have gained much attention in the recent years due to
the increasing amount of aerial and underwater vehicles as well as
nanosatellites. Trajectory tracking control of these vehicles is a substantial
aspect for an increasing range of application domains. However, external
disturbances and parts of the internal dynamics are often unknown or very
time-consuming to model. To overcome this issue, we present a tracking control
law for underactuated rigid-body dynamics using an online learning-based oracle
for the prediction of the unknown dynamics. We show that Gaussian process
models are of particular interest for the role of the oracle. The presented
approach guarantees a bounded tracking error with high probability where the
bound is explicitly given. A numerical example highlights the effectiveness of
the proposed control law.

    

### [[2010.00952] Distributed Proximal Splitting Algorithms with Rates and Acceleration](http://arxiv.org/abs/2010.00952)


  We analyze several generic proximal splitting algorithms well suited for
large-scale convex nonsmooth optimization. We derive sublinear and linear
convergence results with new rates on the function value suboptimality or
distance to the solution, as well as new accelerated versions, using varying
stepsizes. In addition, we propose distributed variants of these algorithms,
which can be accelerated as well. While most existing results are ergodic, our
nonergodic results significantly broaden our understanding of primal-dual
optimization algorithms.

    

### [[2010.11536] Joint Use of Node Attributes and Proximity for Semi-Supervised Classification on Graphs](http://arxiv.org/abs/2010.11536)


  The task of node classification is to infer unknown node labels, given the
labels for some of the nodes along with the network structure and other node
attributes. Typically, approaches for this task assume homophily, whereby
neighboring nodes have similar attributes and a node's label can be predicted
from the labels of its neighbors or other proximate (i.e., nearby) nodes in the
network. However, such an assumption may not always hold -- in fact, there are
cases where labels are better predicted from the individual attributes of each
node rather than the labels of its proximate nodes. Ideally, node
classification methods should flexibly adapt to a range of settings wherein
unknown labels are predicted either from labels of proximate nodes, or
individual node attributes, or partly both. In this paper, we propose a
principled approach, JANE, based on a generative probabilistic model that
jointly weighs the role of attributes and node proximity via embeddings in
predicting labels. Our experiments on a variety of network datasets demonstrate
that JANE exhibits the desired combination of versatility and competitive
performance compared to standard baselines.

    

### [[2011.03424] Session-aware Recommendation: A Surprising Quest for the State-of-the-art](http://arxiv.org/abs/2011.03424)


  Recommender systems are designed to help users in situations of information
overload. In recent years, we observed increased interest in session-based
recommendation scenarios, where the problem is to make item suggestions to
users based only on interactions observed in an ongoing session. However, in
cases where interactions from previous user sessions are available, the
recommendations can be personalized according to the users' long-term
preferences, a process called session-aware recommendation. Today, research in
this area is scattered and many existing works only compare session-aware with
session-based models. This makes it challenging to understand what represents
the state-of-the-art. To close this research gap, we benchmarked recent
session-aware algorithms against each other and against a number of
session-based recommendation algorithms and trivial extensions thereof. Our
comparison, to some surprise, revealed that (i) item simple techniques based on
nearest neighbors consistently outperform recent neural techniques and that
(ii) session-aware models were mostly not better than approaches that do not
use long-term preference information. Our work therefore not only points to
potential methodological issues where new methods are compared to weak
baselines, but also indicates that there remains a huge potential for more
sophisticated session-aware recommendation algorithms.

    

### [[2011.08225] Automatic selection of clustering algorithms using supervised graph embedding](http://arxiv.org/abs/2011.08225)


  The widespread adoption of machine learning (ML) techniques and the extensive
expertise required to apply them have led to increased interest in automated ML
solutions that reduce the need for human intervention. One of the main
challenges in applying ML to previously unseen problems is algorithm selection
- the identification of high-performing algorithm(s) for a given dataset, task,
and evaluation measure. This study addresses the algorithm selection challenge
for data clustering, a fundamental task in data mining that is aimed at
grouping similar objects. We present MARCO-GE, a novel meta-learning approach
for the automated recommendation of clustering algorithms. MARCO-GE first
transforms datasets into graphs and then utilizes a graph convolutional neural
network technique to extract their latent representation. Using the embedding
representations obtained, MARCO-GE trains a ranking meta-model capable of
accurately recommending top-performing algorithms for a new dataset and
clustering evaluation measure. Extensive evaluation on 210 datasets, 13
clustering algorithms, and 10 clustering measures demonstrates the
effectiveness of our approach and its superiority in terms of predictive and
generalization performance over state-of-the-art clustering meta-learning
approaches.

    

### [[2102.01621] Depth separation beyond radial functions](http://arxiv.org/abs/2102.01621)


  High-dimensional depth separation results for neural networks show that
certain functions can be efficiently approximated by two-hidden-layer networks
but not by one-hidden-layer ones in high-dimensions $d$. Existing results of
this type mainly focus on functions with an underlying radial or
one-dimensional structure, which are usually not encountered in practice. The
first contribution of this paper is to extend such results to a more general
class of functions, namely functions with piece-wise oscillatory structure, by
building on the proof strategy of (Eldan and Shamir, 2016). We complement these
results by showing that, if the domain radius and the rate of oscillation of
the objective function are constant, then approximation by one-hidden-layer
networks holds at a $\mathrm{poly}(d)$ rate for any fixed error threshold.
A common theme in the proofs of depth-separation results is the fact that
one-hidden-layer networks fail to approximate high-energy functions whose
Fourier representation is spread in the domain. On the other hand, existing
approximation results of a function by one-hidden-layer neural networks rely on
the function having a sparse Fourier representation. The choice of the domain
also represents a source of gaps between upper and lower approximation bounds.
Focusing on a fixed approximation domain, namely the sphere $\mathbb{S}^{d-1}$
in dimension $d$, we provide a characterisation of both functions which are
efficiently approximable by one-hidden-layer networks and of functions which
are provably not, in terms of their Fourier expansion.

    

### [[2103.01306] Scalable Scene Flow from Point Clouds in the Real World](http://arxiv.org/abs/2103.01306)


  Autonomous vehicles operate in highly dynamic environments necessitating an
accurate assessment of which aspects of a scene are moving and where they are
moving to. A popular approach to 3D motion estimation, termed scene flow, is to
employ 3D point cloud data from consecutive LiDAR scans, although such
approaches have been limited by the small size of real-world, annotated LiDAR
data. In this work, we introduce a new large-scale dataset for scene flow
estimation derived from corresponding tracked 3D objects, which is
$\sim$1,000$\times$ larger than previous real-world datasets in terms of the
number of annotated frames. We demonstrate how previous works were bounded
based on the amount of real LiDAR data available, suggesting that larger
datasets are required to achieve state-of-the-art predictive performance.
Furthermore, we show how previous heuristics for operating on point clouds such
as down-sampling heavily degrade performance, motivating a new class of models
that are tractable on the full point cloud. To address this issue, we introduce
the FastFlow3D architecture which provides real time inference on the full
point cloud. Additionally, we design human-interpretable metrics that better
capture real world aspects by accounting for ego-motion and providing
breakdowns per object type. We hope that this dataset may provide new
opportunities for developing real world scene flow systems.

    

### [[2103.03809] PalmTree: Learning an Assembly Language Model for Instruction Embedding](http://arxiv.org/abs/2103.03809)


  Deep learning has demonstrated its strengths in numerous binary analysis
tasks, including function boundary detection, binary code search, function
prototype inference, value set analysis, etc. When applying deep learning to
binary analysis tasks, we need to decide what input should be fed into the
neural network model. More specifically, we need to answer how to represent an
instruction in a fixed-length vector. The idea of automatically learning
instruction representations is intriguing, however the existing schemes fail to
capture the unique characteristics of disassembly. These schemes ignore the
complex intra-instruction structures and mainly rely on control flow in which
the contextual information is noisy and can be influenced by compiler
optimizations.
In this paper, we propose to pre-train an assembly language model called
PalmTree for generating general-purpose instruction embeddings by conducting
self-supervised training on large-scale unlabeled binary corpora. PalmTree
utilizes three pre-training tasks to capture various characteristics of
assembly language. These training tasks overcome the problems in existing
schemes, thus can help to generate high-quality representations. We conduct
both intrinsic and extrinsic evaluations, and compare PalmTree with other
instruction embedding schemes. PalmTree has the best performance for intrinsic
metrics, and outperforms the other instruction embedding schemes for all
downstream tasks.

    

### [[2103.03854] A Pilot Study on Visually Stimulated Cognitive Tasks for EEG-Based Dementia Recognition](http://arxiv.org/abs/2103.03854)


  In the status quo, dementia is yet to be cured. Precise diagnosis prior to
the onset of the symptoms can prevent the rapid progression of the emerging
cognitive impairment. Recent progress has shown that Electroencephalography
(EEG) is the promising and cost-effective test to facilitate the detection of
neurocognitive disorders. However, most of the existing works have been using
only resting-state EEG. The efficiencies of EEG signals from various cognitive
tasks, for dementia classification, have yet to be thoroughly investigated. In
this study, we designed four cognitive tasks that engage different cognitive
performances: attention, working memory, and executive function. We
investigated these tasks by using statistical analysis on both time and
frequency domains of EEG signals from three classes of human subjects: Dementia
(DEM), Mild Cognitive Impairment (MCI), and Normal Control (NC). We also
further evaluated the classification performances of two features extraction
methods: Principal Component Analysis (PCA) and Filter Bank Common Spatial
Pattern (FBCSP). We found that the working memory related tasks yielded good
performances for dementia recognition in both cases using PCA and FBCSP.
Moreover, FBCSP with features combination from four tasks revealed the best
sensitivity of 0.87 and the specificity of 0.80. To our best knowledge, this is
the first work that concurrently investigated several cognitive tasks for
dementia recognition using both statistical analysis and classification scores.
Our results yielded essential information to design and aid in conducting
further experimental tasks to early diagnose dementia patients.

    

### [[2103.03923] Surface Warping Incorporating Machine Learning Assisted Domain Likelihood Estimation: A New Paradigm in Mine Geology Modelling and Automation](http://arxiv.org/abs/2103.03923)


  This paper illustrates an application of machine learning (ML) within a
complex system that performs grade estimation. In surface mining, assay
measurements taken from production drilling often provide useful information
that allows initially inaccurate surfaces created using sparse exploration data
to be revised and subsequently improved. Recently, a Bayesian warping technique
has been proposed to reshape modeled surfaces using geochemical and spatial
constraints imposed by newly acquired blasthole data. This paper focuses on
incorporating machine learning into this warping framework to make the
likelihood computation generalizable. The technique works by adjusting the
position of vertices on the surface to maximize the integrity of modeled
geological boundaries with respect to sparse geochemical observations. Its
foundation is laid by a Bayesian derivation in which the geological domain
likelihood given the chemistry, p(g|c), plays a similar role to p(y(c)|g). This
observation allows a manually calibrated process centered around the latter to
be automated since ML techniques may be used to estimate the former in a
data-driven way. Machine learning performance is evaluated for gradient
boosting, neural network, random forest and other classifiers in a binary and
multi-class context using precision and recall rates. Once ML likelihood
estimators are integrated in the surface warping framework, surface shaping
performance is evaluated using unseen data by examining the categorical
distribution of test samples located above and below the warped surface.
Large-scale validation experiments are performed to assess the overall efficacy
of ML assisted surface warping as a fully integrated component within an ore
grade estimation system where the posterior mean is obtained via Gaussian
Process inference with a Matern 3/2 kernel.

    

### [[2103.09154] Leveraging Recent Advances in Deep Learning for Audio-Visual Emotion Recognition](http://arxiv.org/abs/2103.09154)


  Emotional expressions are the behaviors that communicate our emotional state
or attitude to others. They are expressed through verbal and non-verbal
communication. Complex human behavior can be understood by studying physical
features from multiple modalities; mainly facial, vocal and physical gestures.
Recently, spontaneous multi-modal emotion recognition has been extensively
studied for human behavior analysis. In this paper, we propose a new deep
learning-based approach for audio-visual emotion recognition. Our approach
leverages recent advances in deep learning like knowledge distillation and
high-performing deep architectures. The deep feature representations of the
audio and visual modalities are fused based on a model-level fusion strategy. A
recurrent neural network is then used to capture the temporal dynamics. Our
proposed approach substantially outperforms state-of-the-art approaches in
predicting valence on the RECOLA dataset. Moreover, our proposed visual facial
expression feature extraction network outperforms state-of-the-art results on
the AffectNet and Google Facial Expression Comparison datasets.

    

### [[2103.11109] DataLens: Scalable Privacy Preserving Training via Gradient Compression and Aggregation](http://arxiv.org/abs/2103.11109)


  Recent success of deep neural networks (DNNs) hinges on the availability of
large-scale dataset; however, training on such dataset often poses privacy
risks for sensitive training information. In this paper, we aim to explore the
power of generative models and gradient sparsity, and propose a scalable
privacy-preserving generative model DATALENS. Comparing with the standard PATE
privacy-preserving framework which allows teachers to vote on one-dimensional
predictions, voting on the high dimensional gradient vectors is challenging in
terms of privacy preservation. As dimension reduction techniques are required,
we need to navigate a delicate tradeoff space between (1) the improvement of
privacy preservation and (2) the slowdown of SGD convergence. To tackle this,
we take advantage of communication efficient learning and propose a novel noise
compression and aggregation approach TOPAGG by combining top-k compression for
dimension reduction with a corresponding noise injection mechanism. We
theoretically prove that the DATALENS framework guarantees differential privacy
for its generated data, and provide analysis on its convergence. To demonstrate
the practical usage of DATALENS, we conduct extensive experiments on diverse
datasets including MNIST, Fashion-MNIST, and high dimensional CelebA, and we
show that, DATALENS significantly outperforms other baseline DP generative
models. In addition, we adapt the proposed TOPAGG approach, which is one of the
key building blocks in DATALENS, to DP SGD training, and show that it is able
to achieve higher utility than the state-of-the-art DP SGD approach in most
cases. Our code is publicly available at this https URL.

    

### [[2103.11955] Improving and Simplifying Pattern Exploiting Training](http://arxiv.org/abs/2103.11955)


  Recently, pre-trained language models (LMs) have achieved strong performance
when fine-tuned on difficult benchmarks like SuperGLUE. However, performance
can suffer when there are very few labeled examples available for fine-tuning.
Pattern Exploiting Training (PET) is a recent approach that leverages patterns
for few-shot learning. However, PET uses task-specific unlabeled data. In this
paper, we focus on few-shot learning without any unlabeled data and introduce
ADAPET, which modifies PET's objective to provide denser supervision during
fine-tuning. As a result, ADAPET outperforms PET on SuperGLUE without any
task-specific unlabeled data. Our code can be found at
this https URL.

    

### [[2103.15341] Stiff Neural Ordinary Differential Equations](http://arxiv.org/abs/2103.15341)


  Neural Ordinary Differential Equations (ODE) are a promising approach to
learn dynamic models from time-series data in science and engineering
applications. This work aims at learning Neural ODE for stiff systems, which
are usually raised from chemical kinetic modeling in chemical and biological
systems. We first show the challenges of learning neural ODE in the classical
stiff ODE systems of Robertson's problem and propose techniques to mitigate the
challenges associated with scale separations in stiff systems. We then present
successful demonstrations in stiff systems of Robertson's problem and an air
pollution problem. The demonstrations show that the usage of deep networks with
rectified activations, proper scaling of the network outputs as well as loss
functions, and stabilized gradient calculations are the key techniques enabling
the learning of stiff neural ODE. The success of learning stiff neural ODE
opens up possibilities of using neural ODEs in applications with widely varying
time-scales, like chemical dynamics in energy conversion, environmental
engineering, and the life sciences.

    

### [[2104.04103] Causal Decision Making and Causal Effect Estimation Are Not the Same... and Why It Matters](http://arxiv.org/abs/2104.04103)


  Causal decision making (CDM) based on machine learning has become a routine
part of business. Businesses algorithmically target offers, incentives, and
recommendations to affect consumer behavior. Recently, we have seen an
acceleration of research related to CDM and causal effect estimation (CEE)
using machine-learned models. This article highlights an important perspective:
CDM is not the same as CEE, and counterintuitively, accurate CEE is not
necessary for accurate CDM. Our experience is that this is not well understood
by practitioners or most researchers. Technically, the estimand of interest is
different, and this has important implications both for modeling and for the
use of statistical models for CDM. We draw on prior research to highlight three
implications. (1) We should consider carefully the objective function of the
causal machine learning, and if possible, optimize for accurate treatment
assignment rather than for accurate effect-size estimation. (2) Confounding
does not have the same effect on CDM as it does on CEE. The upshot is that for
supporting CDM it may be just as good or even better to learn with confounded
data as with unconfounded data. Finally, (3) causal statistical modeling may
not be necessary to support CDM because a proxy target for statistical modeling
might do as well or better. This third observation helps to explain at least
one broad common CDM practice that seems wrong at first blush: the widespread
use of non-causal models for targeting interventions. The last two implications
are particularly important in practice, as acquiring (unconfounded) data on all
counterfactuals can be costly and often impracticable. These observations open
substantial research ground. We hope to facilitate research in this area by
pointing to related articles from multiple contributing fields, including two
dozen articles published the last three to four years.

    

### [[2104.14547] NURBS-Diff: A differentiable programming module for NURBS](http://arxiv.org/abs/2104.14547)


  Boundary representations (B-reps) using Non-Uniform Rational B-splines
(NURBS) are the de facto standard used in CAD, but their utility in deep
learning-based approaches is not well researched. We propose a differentiable
NURBS module to integrate the NURBS representation of CAD models with deep
learning methods. We mathematically define the derivatives of the NURBS curves
or surfaces with respect to the input parameters. These derivatives are used to
define an approximate Jacobian that can be used to perform the "backward"
evaluation used while training deep learning models. We have implemented our
NURBS module using GPU-accelerated algorithms and integrated it with PyTorch, a
popular deep learning framework. We demonstrate the efficacy of our NURBS
module in performing CAD operations such as curve or surface fitting and
surface offsetting. Further, we show its utility in deep learning for
unsupervised point cloud reconstruction. These examples show that our module
performs better for certain deep learning frameworks and can be directly
integrated with any deep-learning framework requiring NURBS.

    

### [[2105.02963] Attention-augmented Spatio-Temporal Segmentation for Land Cover Mapping](http://arxiv.org/abs/2105.02963)


  The availability of massive earth observing satellite data provide huge
opportunities for land use and land cover mapping. However, such mapping effort
is challenging due to the existence of various land cover classes, noisy data,
and the lack of proper labels. Also, each land cover class typically has its
own unique temporal pattern and can be identified only during certain periods.
In this article, we introduce a novel architecture that incorporates the UNet
structure with Bidirectional LSTM and Attention mechanism to jointly exploit
the spatial and temporal nature of satellite data and to better identify the
unique temporal patterns of each land cover. We evaluate this method for
mapping crops in multiple regions over the world. We compare our method with
other state-of-the-art methods both quantitatively and qualitatively on two
real-world datasets which involve multiple land cover classes. We also
visualise the attention weights to study its effectiveness in mitigating noise
and identifying discriminative time period.

    

### [[2106.06882] Sparse PointPillars: Maintaining and Exploiting Input Sparsity to Improve Runtime on Embedded Systems](http://arxiv.org/abs/2106.06882)


  Bird's Eye View (BEV) is a popular representation for processing 3D point
clouds, and by its nature is fundamentally sparse. Motivated by the
computational limitations of mobile robot platforms, we take a fast,
high-performance BEV 3D object detector - PointPillars - and modify its
backbone to maintain and exploit this input sparsity, leading to decreased
runtimes. We present results on KITTI, a canonical 3D detection dataset, and
Matterport-Chair, a novel Matterport3D-derived chair detection dataset from
scenes in real furnished homes. We evaluate runtime characteristics using a
desktop GPU, an embedded ML accelerator, and a robot CPU, demonstrating that
our method results in significant runtime decreases (2x or more) for embedded
systems with only a modest decrease in detection quality. Our work represents a
new approach for practitioners to optimize models for embedded systems by
maintaining and exploiting input sparsity throughout their entire pipeline to
reduce runtime and resource usage while preserving detection performance. All
models, weights, experimental configurations, and datasets used are publicly
available.

    

### [[2106.09159] Automatic Curricula via Expert Demonstrations](http://arxiv.org/abs/2106.09159)


  We propose Automatic Curricula via Expert Demonstrations (ACED), a
reinforcement learning (RL) approach that combines the ideas of imitation
learning and curriculum learning in order to solve challenging robotic
manipulation tasks with sparse reward functions. Curriculum learning solves
complicated RL tasks by introducing a sequence of auxiliary tasks with
increasing difficulty, yet how to automatically design effective and
generalizable curricula remains a challenging research problem. ACED extracts
curricula from a small amount of expert demonstration trajectories by dividing
demonstrations into sections and initializing training episodes to states
sampled from different sections of demonstrations. Through moving the reset
states from the end to the beginning of demonstrations as the learning agent
improves its performance, ACED not only learns challenging manipulation tasks
with unseen initializations and goals, but also discovers novel solutions that
are distinct from the demonstrations. In addition, ACED can be naturally
combined with other imitation learning methods to utilize expert demonstrations
in a more efficient manner, and we show that a combination of ACED with
behavior cloning allows pick-and-place tasks to be learned with as few as 1
demonstration and block stacking tasks to be learned with 20 demonstrations.

    

### [[2106.11731] MIMIR: Deep Regression for Automated Analysis of UK Biobank Body MRI](http://arxiv.org/abs/2106.11731)


  UK Biobank (UKB) conducts large-scale examinations of more than half a
million volunteers, collecting health-related information on genetics,
lifestyle, blood biochemistry, and more. Medical imaging of 100,000 subjects,
with 70,000 follow-up sessions, enables measurements of organs, muscle, and
body composition. With up to 170,000 mounting MR images, various methodologies
are accordingly engaged in large-scale image analysis. This work presents an
experimental inference engine that can automatically predict a comprehensive
profile of subject metadata from UKB neck-to-knee body MRI. It was evaluated in
cross-validation for baseline characteristics such as age, height, weight, and
sex, but also measurements of body composition, organ volumes, and abstract
properties like grip strength, pulse rate, and type 2 diabetic status. It
predicted subsequently released test data covering twelve body composition
metrics with a 3% median error. The proposed system can automatically analyze
one thousand subjects within ten minutes, providing individual confidence
intervals. The underlying methodology utilizes convolutional neural networks
for image-based mean-variance regression on two-dimensional representations of
the MRI data. This work aims to make the proposed system available for free to
researchers, who can use it to obtain fast and fully-automated estimates of 72
different measurements immediately upon release of new UKB image data.

    

### [[2106.16101] AdaGDA: Faster Adaptive Gradient Descent Ascent Methods for Minimax Optimization](http://arxiv.org/abs/2106.16101)


  In the paper, we propose a class of faster adaptive Gradient Descent Ascent
(GDA) methods for solving the nonconvex-strongly-concave minimax problems based
on unified adaptive matrices, which include almost existing coordinate-wise and
global adaptive learning rates. Specifically, we propose a fast Adaptive
Gradient Decent Ascent (AdaGDA) method based on the basic momentum technique,
which reaches a lower sample complexity of $O(\kappa^4\epsilon^{-4})$ for
finding an $\epsilon$-stationary point without large batches, which improves
the results of the existing adaptive GDA methods by a factor of
$O(\sqrt{\kappa})$. At the same time, we present an accelerated version of
AdaGDA (VR-AdaGDA) method based on the momentum-based variance reduced
technique, which achieves a lower sample complexity of
$O(\kappa^{4.5}\epsilon^{-3})$ for finding an $\epsilon$-stationary point
without large batches, which improves the results of the existing adaptive GDA
methods by a factor of $O(\epsilon^{-1})$. Moreover, we prove that our
VR-AdaGDA method reaches the best known sample complexity of
$O(\kappa^{3}\epsilon^{-3})$ with the mini-batch size $O(\kappa^3)$. In
particular, we provide an effective convergence analysis framework for our
adaptive GDA methods. Some experimental results on fair classifier and policy
evaluation tasks demonstrate the efficiency of our algorithms.

    

### [[2109.05897] Question Answering over Electronic Devices: A New Benchmark Dataset and a Multi-Task Learning based QA Framework](http://arxiv.org/abs/2109.05897)


  Answering questions asked from instructional corpora such as E-manuals,
recipe books, etc., has been far less studied than open-domain factoid
context-based question answering. This can be primarily attributed to the
absence of standard benchmark datasets. In this paper we meticulously create a
large amount of data connected with E-manuals and develop suitable algorithm to
exploit it. We collect E-Manual Corpus, a huge corpus of 307,957 E-manuals and
pretrain RoBERTa on this large corpus. We create various benchmark QA datasets
which include question answer pairs curated by experts based upon two
E-manuals, real user questions from Community Question Answering Forum
pertaining to E-manuals etc. We introduce EMQAP (E-Manual Question Answering
Pipeline) that answers questions pertaining to electronics devices. Built upon
the pretrained RoBERTa, it harbors a supervised multi-task learning framework
which efficiently performs the dual tasks of identifying the section in the
E-manual where the answer can be found and the exact answer span within that
section. For E-Manual annotated question-answer pairs, we show an improvement
of about 40% in ROUGE-L F1 scores over the most competitive baseline. We
perform a detailed ablation study and establish the versatility of EMQAP across
different circumstances. The code and datasets are shared at
this https URL, and the corresponding
project website is this https URL.

    

### [[2109.06024] Formalizing and Estimating Distribution Inference Risks](http://arxiv.org/abs/2109.06024)


  Property inference attacks reveal statistical properties about a training set
but are difficult to distinguish from the intrinsic purpose of statistical
machine learning, namely to produce models that capture statistical properties
about a distribution. Motivated by Yeom et al.'s membership inference
framework, we propose a formal and general definition of property inference
attacks. The proposed notion describes attacks that can distinguish between
possible training distributions, extending beyond previous property inference
attacks that infer the ratio of a particular type of data in the training data
set such as the proportion of females. We show how our definition captures
previous property inference attacks as well as a new attack that can reveal the
average node degree or clustering coefficient of a training graph. Our
definition also enables a theorem that connects the maximum possible accuracy
of inference attacks distinguishing between distributions to the effective size
of dataset leaked by the model. To quantify and understand property inference
risks, we conduct a series of experiments across a range of different
distributions using both black-box and white-box attacks. Our results show that
inexpensive attacks are often as effective as expensive meta-classifier
attacks, and that there are surprising asymmetries in the effectiveness of
attacks. We also extend the state-of-the-art property inference attack to work
on convolutional neural networks, and propose techniques to help identify
parameters in a model that leak the most information, thus significantly
lowering resource requirements for meta-classifier attacks.

    

### [[2109.06126] Neural Network Guided Evolutionary Fuzzing for Finding Traffic Violations of Autonomous Vehicles](http://arxiv.org/abs/2109.06126)


  Self-driving cars and trucks, autonomous vehicles (AVs), should not be
accepted by regulatory bodies and the public until they have much higher
confidence in their safety and reliability -- which can most practically and
convincingly be achieved by testing. But existing testing methods are
inadequate for checking the end-to-end behaviors of AV controllers against
complex, real-world corner cases involving interactions with multiple
independent agents such as pedestrians and human-driven vehicles. While
test-driving AVs on streets and highways fails to capture many rare events,
existing simulation-based testing methods mainly focus on simple scenarios and
do not scale well for complex driving situations that require sophisticated
awareness of the surroundings. To address these limitations, we propose a new
fuzz testing technique, called AutoFuzz, which can leverage widely-used AV
simulators' API grammars. to generate semantically and temporally valid complex
driving scenarios (sequences of scenes). AutoFuzz is guided by a constrained
Neural Network (NN) evolutionary search over the API grammar to generate
scenarios seeking to find unique traffic violations. Evaluation of our
prototype on one state-of-the-art learning-based controller and two rule-based
controllers shows that AutoFuzz efficiently finds hundreds of realistic traffic
violations resembling real-world crashes. Further, fine-tuning the
learning-based controller with the traffic violations found by AutoFuzz
successfully reduced the traffic violations found in the new version of the AV
controller software.

    

### [[2109.06355] Optimizing FPGA-based Accelerator Design for Large-Scale Molecular Similarity Search](http://arxiv.org/abs/2109.06355)


  Molecular similarity search has been widely used in drug discovery to
identify structurally similar compounds from large molecular databases rapidly.
With the increasing size of chemical libraries, there is growing interest in
the efficient acceleration of large-scale similarity search. Existing works
mainly focus on CPU and GPU to accelerate the computation of the Tanimoto
coefficient in measuring the pairwise similarity between different molecular
fingerprints. In this paper, we propose and optimize an FPGA-based accelerator
design on exhaustive and approximate search algorithms. On exhaustive search
using BitBound & folding, we analyze the similarity cutoff and folding level
relationship with search speedup and accuracy, and propose a scalable
on-the-fly query engine on FPGAs to reduce the resource utilization and
pipeline interval. We achieve a 450 million compounds-per-second processing
throughput for a single query engine. On approximate search using hierarchical
navigable small world (HNSW), a popular algorithm with high recall and query
speed. We propose an FPGA-based graph traversal engine to utilize a high
throughput register array based priority queue and fine-grained distance
calculation engine to increase the processing capability. Experimental results
show that the proposed FPGA-based HNSW implementation has a 103385 query per
second (QPS) on the Chembl database with 0.92 recall and achieves a 35x speedup
than the existing CPU implementation on average. To the best of our knowledge,
our FPGA-based implementation is the first attempt to accelerate molecular
similarity search algorithms on FPGA and has the highest performance among
existing approaches.

    

### [[2109.06382] Cohmeleon: Learning-Based Orchestration of Accelerator Coherence in Heterogeneous SoCs](http://arxiv.org/abs/2109.06382)


  One of the most critical aspects of integrating loosely-coupled accelerators
in heterogeneous SoC architectures is orchestrating their interactions with the
memory hierarchy, especially in terms of navigating the various cache-coherence
options: from accelerators accessing off-chip memory directly, bypassing the
cache hierarchy, to accelerators having their own private cache. By running
real-size applications on FPGA-based prototypes of many-accelerator multi-core
SoCs, we show that the best cache-coherence mode for a given accelerator varies
at runtime, depending on the accelerator's characteristics, the workload size,
and the overall SoC status.
Cohmeleon applies reinforcement learning to select the best coherence mode
for each accelerator dynamically at runtime, as opposed to statically at design
time. It makes these selections adaptively, by continuously observing the
system and measuring its performance. Cohmeleon is accelerator-agnostic,
architecture-independent, and it requires minimal hardware support. Cohmeleon
is also transparent to application programmers and has a negligible software
overhead. FPGA-based experiments show that our runtime approach offers, on
average, a 38% speedup with a 66% reduction of off-chip memory accesses
compared to state-of-the-art design-time approaches. Moreover, it can match
runtime solutions that are manually tuned for the target architecture.

    

### [[2109.06561] Beyond Distributed Subgraph Detection: Induced Subgraphs, Multicolored Problems and Graph Parameters](http://arxiv.org/abs/2109.06561)


  Subgraph detection has recently been one of the most studied problems in the
CONGEST model of distributed computing. In this work, we study the distributed
complexity of problems closely related to subgraph detection, mainly focusing
on induced subgraph detection. The main line of this work presents lower bounds
and parameterized algorithms w.r.t structural parameters of the input graph:
-- On general graphs, we give unconditional lower bounds for induced
detection of cycles and patterns of treewidth 2 in CONGEST. Moreover, by
adapting reductions from centralized parameterized complexity, we prove lower
bounds in CONGEST for detecting patterns with a 4-clique, and for induced path
detection conditional on the hardness of triangle detection in the congested
clique.
-- On graphs of bounded degeneracy, we show that induced paths can be
detected fast in CONGEST using techniques from parameterized algorithms, while
detecting cycles and patterns of treewidth 2 is hard.
-- On graphs of bounded vertex cover number, we show that induced subgraph
detection is easy in CONGEST for any pattern graph. More specifically, we adapt
a centralized parameterized algorithm for a more general maximum common induced
subgraph detection problem to the distributed setting.
In addition to these induced subgraph detection results, we study various
related problems in the CONGEST and congested clique models, including for
multicolored versions of subgraph-detection-like problems.

    

### [[2109.06593] Online Algorithms with Lookaround](http://arxiv.org/abs/2109.06593)


  We introduce a new model of computation: the online LOCAL model (OLOCAL). In
this model, the adversary reveals the nodes of the input graph one by one, in
the same way as in classical online algorithms, but for each new node the
algorithm can also inspect its radius-$T$ neighborhood before choosing the
output; instead of looking ahead in time, we have the power of looking around
in space. It is natural to compare OLOCAL with the LOCAL model of distributed
computing, in which all nodes make decisions simultaneously in parallel based
on their radius-$T$ neighborhoods.

    

### [[2109.06601] Distributed Vertex Cover Reconfiguration](http://arxiv.org/abs/2109.06601)


  Reconfiguration schedules, i.e., sequences that gradually transform one
solution of a problem to another while always maintaining feasibility, have
been extensively studied. Most research has dealt with the decision problem of
whether a reconfiguration schedule exists, and the complexity of finding one. A
prime example is the reconfiguration of vertex covers. We initiate the study of
batched vertex cover reconfiguration, which allows to reconfigure multiple
vertices concurrently while requiring that any adversarial reconfiguration
order within a batch maintains feasibility. The latter provides robustness,
e.g., if the simultaneous reconfiguration of a batch cannot be guaranteed. The
quality of a schedule is measured by the number of batches until all nodes are
reconfigured, and its cost, i.e., the maximum size of an intermediate vertex
cover.
To set a baseline for batch reconfiguration, we show that for graphs
belonging to one of the classes $\{\mathsf{cycles, trees, forests, chordal,
cactus, even\text{-}hole\text{-}free, claw\text{-}free}\}$, there are schedules
that use $O(\varepsilon^{-1})$ batches and incur only a $1+\varepsilon$
multiplicative increase in cost over the best sequential schedules. Our main
contribution is to compute such batch schedules in $O(\varepsilon^{-1}\log^*
n)$ distributed time, which we also show to be tight. Further, we show that
once we step out of these graph classes we face a very different situation.
There are graph classes on which no efficient distributed algorithm can obtain
the best (or almost best) existing schedule. Moreover, there are classes of
bounded degree graphs which do not admit any reconfiguration schedules without
incurring a large multiplicative increase in the cost at all.

    

### [[2109.06811] Egalitarian Byzantine Fault Tolerance](http://arxiv.org/abs/2109.06811)


  Minimizing end-to-end latency in geo-replicated systems usually makes it
necessary to compromise on resilience, resource efficiency, or throughput
performance, because existing approaches either tolerate only crashes, require
additional replicas, or rely on a global leader for consensus. In this paper,
we eliminate the need for such tradeoffs by presenting Isos, a leaderless
replication protocol that tolerates up to $f$ Byzantine faults with a minimum
of $3f+1$ replicas. To reduce latency in wide-area environments, Isos relies on
an efficient consensus algorithm that allows all participating replicas to
propose new requests and thereby enables clients to avoid delays by submitting
requests to their nearest replica. In addition, Isos minimizes overhead by
limiting message ordering to requests that conflict with each other (e.g., due
to accessing the same state parts) and by already committing them after three
communication steps if at least $f+1$ replicas report each conflict. Our
experimental evaluation with a geo-replicated key-value store shows that these
properties allow Isos to provide lower end-to-end latency than existing
protocols, especially for use-case scenarios in which the clients of a system
are distributed across multiple locations.

    

### [[1910.12747] Introduction to local certification](http://arxiv.org/abs/1910.12747)


  A distributed graph algorithm is basically an algorithm where every node of a
graph can look at its neighborhood at some distance in the graph and chose its
output. As distributed environment are subject to faults, an important issue is
to be able to check that the output is correct, or in general that the network
is in proper configuration with respect to some predicate. One would like this
checking to be very local, to avoid using too much resources. Unfortunately
most predicates cannot be checked this way, and that is where certification
comes into play. Local certification (also known as proof-labeling schemes,
locally checkable proofs or distributed verification) consists in assigning
labels to the nodes, that certify that the configuration is correct. There are
several point of view on this topic: it can be seen as a part of
self-stabilizing algorithms, as labeling problem, or as a non-deterministic
distributed decision.
This paper is an introduction to the domain of local certification, giving an
overview of the history, the techniques and the current research directions.

    

### [[2010.07226] Discriminating Equivalent Algorithms via Relative Performance](http://arxiv.org/abs/2010.07226)


  In scientific computing, it is common that a mathematical expression can be
computed by many different algorithms (sometimes over hundreds), each
identifying a specific sequence of library calls. Although mathematically
equivalent, those algorithms might exhibit significant differences in terms of
performance. However in practice, due to fluctuations, there is not one
algorithm that consistently performs noticeably better than the rest. For this
reason, with this work we aim to identify not the one best algorithm, but the
subset of algorithms that are reliably faster than the rest. To this end,
instead of using the usual approach of quantifying the performance of an
algorithm in absolute terms, we present a measurement-based clustering approach
to sort the algorithms into equivalence (or performance) classes using
pair-wise comparisons. We show that this approach, based on relative
performance, leads to robust identification of the fastest algorithms even
under noisy system conditions. Furthermore, it enables the development of
practical machine learning models for automatic algorithm selection.

    

### [[2102.08703] Local Mending](http://arxiv.org/abs/2102.08703)


  In this work we introduce the graph-theoretic notion of mendability: for each
locally checkable graph problem we can define its mending radius, which
captures the idea of how far one needs to modify a partial solution in order to
"patch a hole."
We explore how mendability is connected to the existence of efficient
algorithms, especially in distributed, parallel, and fault-tolerant settings.
It is easy to see that $O(1)$-mendable problems are also solvable in $O(\log^*
n)$ rounds in the LOCAL model of distributed computing. One of the surprises is
that in paths and cycles, a converse also holds in the following sense: if a
problem $\Pi$ can be solved in $O(\log^* n)$, there is always a restriction
$\Pi' \subseteq \Pi$ that is still efficiently solvable but that is also
$O(1)$-mendable.
We also explore the structure of the landscape of mendability. For example,
we show that in trees, the mending radius of any locally checkable problem is
$O(1)$, $\Theta(\log n)$, or $\Theta(n)$, while in general graphs the structure
is much more diverse.

    

### [[2109.06243] KroneckerBERT: Learning Kronecker Decomposition for Pre-trained Language Models via Knowledge Distillation](http://arxiv.org/abs/2109.06243)


  The development of over-parameterized pre-trained language models has made a
significant contribution toward the success of natural language processing.
While over-parameterization of these models is the key to their generalization
power, it makes them unsuitable for deployment on low-capacity devices. We push
the limits of state-of-the-art Transformer-based pre-trained language model
compression using Kronecker decomposition. We use this decomposition for
compression of the embedding layer, all linear mappings in the multi-head
attention, and the feed-forward network modules in the Transformer layer. We
perform intermediate-layer knowledge distillation using the uncompressed model
as the teacher to improve the performance of the compressed model. We present
our KroneckerBERT, a compressed version of the BERT_BASE model obtained using
this framework. We evaluate the performance of KroneckerBERT on well-known NLP
benchmarks and show that for a high compression factor of 19 (5% of the size of
the BERT_BASE model), our KroneckerBERT outperforms state-of-the-art
compression methods on the GLUE. Our experiments indicate that the proposed
model has promising out-of-distribution robustness and is superior to the
state-of-the-art compression methods on SQuAD.

    

### [[2109.06401] Camera-Tracklet-Aware Contrastive Learning for Unsupervised Vehicle Re-Identification](http://arxiv.org/abs/2109.06401)


  Recently, vehicle re-identification methods based on deep learning constitute
remarkable achievement. However, this achievement requires large-scale and
well-annotated datasets. In constructing the dataset, assigning globally
available identities (Ids) to vehicles captured from a great number of cameras
is labour-intensive, because it needs to consider their subtle appearance
differences or viewpoint variations. In this paper, we propose
camera-tracklet-aware contrastive learning (CTACL) using the multi-camera
tracklet information without vehicle identity labels. The proposed CTACL
divides an unlabelled domain, i.e., entire vehicle images, into multiple
camera-level subdomains and conducts contrastive learning within and beyond the
subdomains. The positive and negative samples for contrastive learning are
defined using tracklet Ids of each camera. Additionally, the domain adaptation
across camera networks is introduced to improve the generalisation performance
of learnt representations and alleviate the performance degradation resulted
from the domain gap between the subdomains. We demonstrate the effectiveness of
our approach on video-based and image-based vehicle Re-ID datasets.
Experimental results show that the proposed method outperforms the recent
state-of-the-art unsupervised vehicle Re-ID methods. The source code for this
paper is publicly available on
`this https URL.

    

### [[2109.06409] Reinforcement Learning with Evolutionary Trajectory Generator: A General Approach for Quadrupedal Locomotion](http://arxiv.org/abs/2109.06409)


  Recently reinforcement learning (RL) has emerged as a promising approach for
quadrupedal locomotion, which can save the manual effort in conventional
approaches such as designing skill-specific controllers. However, due to the
complex nonlinear dynamics in quadrupedal robots and reward sparsity, it is
still difficult for RL to learn effective gaits from scratch, especially in
challenging tasks such as walking over the balance beam. To alleviate such
difficulty, we propose a novel RL-based approach that contains an evolutionary
foot trajectory generator. Unlike prior methods that use a fixed trajectory
generator, the generator continually optimizes the shape of the output
trajectory for the given task, providing diversified motion priors to guide the
policy learning. The policy is trained with reinforcement learning to output
residual control signals that fit different gaits. We then optimize the
trajectory generator and policy network alternatively to stabilize the training
and share the exploratory data to improve sample efficiency. As a result, our
approach can solve a range of challenging tasks in simulation by learning from
scratch, including walking on a balance beam and crawling through the cave. To
further verify the effectiveness of our approach, we deploy the controller
learned in the simulation on a 12-DoF quadrupedal robot, and it can
successfully traverse challenging scenarios with efficient gaits.

    

### [[2109.06415] Gradient Imitation Reinforcement Learning for Low Resource Relation Extraction](http://arxiv.org/abs/2109.06415)


  Low-resource Relation Extraction (LRE) aims to extract relation facts from
limited labeled corpora when human annotation is scarce. Existing works either
utilize self-training scheme to generate pseudo labels that will cause the
gradual drift problem, or leverage meta-learning scheme which does not solicit
feedback explicitly. To alleviate selection bias due to the lack of feedback
loops in existing LRE learning paradigms, we developed a Gradient Imitation
Reinforcement Learning method to encourage pseudo label data to imitate the
gradient descent direction on labeled data and bootstrap its optimization
capability through trial and error. We also propose a framework called GradLRE,
which handles two major scenarios in low-resource relation extraction. Besides
the scenario where unlabeled data is sufficient, GradLRE handles the situation
where no unlabeled data is available, by exploiting a contextualized
augmentation method to generate data. Experimental results on two public
datasets demonstrate the effectiveness of GradLRE on low resource relation
extraction when comparing with baselines.

    

### [[2109.06416] MMCoVaR: Multimodal COVID-19 Vaccine Focused Data Repository for Fake News Detection and a Baseline Architecture for Classification](http://arxiv.org/abs/2109.06416)


  The outbreak of COVID-19 has resulted in an "infodemic" that has encouraged
the propagation of misinformation about COVID-19 and cure methods which, in
turn, could negatively affect the adoption of recommended public health
measures in the larger population. In this paper, we provide a new multimodal
(consisting of images, text and temporal information) labeled dataset
containing news articles and tweets on the COVID-19 vaccine. We collected 2,593
news articles from 80 publishers for one year between Feb 16th 2020 to May 8th
2021 and 24184 Twitter posts (collected between April 17th 2021 to May 8th
2021). We combine ratings from three news media ranking sites: Medias Bias
Chart, News Guard and Media Bias/Fact Check (MBFC) to classify the news dataset
into two levels of credibility: reliable and unreliable. The combination of
three filters allows for higher precision of labeling. We also propose a stance
detection mechanism to annotate tweets into three levels of credibility:
reliable, unreliable and inconclusive. We provide several statistics as well as
other analytics like, publisher distribution, publication date distribution,
topic analysis, etc. We also provide a novel architecture that classifies the
news data into misinformation or truth to provide a baseline performance for
this dataset. We find that the proposed architecture has an F-Score of 0.919
and accuracy of 0.882 for fake news detection. Furthermore, we provide
benchmark performance for misinformation detection on tweet dataset. This new
multimodal dataset can be used in research on COVID-19 vaccine, including
misinformation detection, influence of fake COVID-19 vaccine information, etc.

    

### [[2109.06422] Cross-Region Domain Adaptation for Class-level Alignment](http://arxiv.org/abs/2109.06422)


  Semantic segmentation requires a lot of training data, which necessitates
costly annotation. There have been many studies on unsupervised domain
adaptation (UDA) from one domain to another, e.g., from computer graphics to
real images. However, there is still a gap in accuracy between UDA and
supervised training on native domain data. It is arguably attributable to
class-level misalignment between the source and target domain data. To cope
with this, we propose a method that applies adversarial training to align two
feature distributions in the target domain. It uses a self-training framework
to split the image into two regions (i.e., trusted and untrusted), which form
two distributions to align in the feature space. We term this approach
cross-region adaptation (CRA) to distinguish from the previous methods of
aligning different domain distributions, which we call cross-domain adaptation
(CDA). CRA can be applied after any CDA method. Experimental results show that
this always improves the accuracy of the combined CDA method, having updated
the state-of-the-art.

    

### [[2109.06432] Improved Few-shot Segmentation by Redifinition of the Roles of Multi-level CNN Features](http://arxiv.org/abs/2109.06432)


  This study is concerned with few-shot segmentation, i.e., segmenting the
region of an unseen object class in a query image, given support image(s) of
its instances. The current methods rely on the pretrained CNN features of the
support and query images. The key to good performance depends on the proper
fusion of their mid-level and high-level features; the former contains
shape-oriented information, while the latter has class-oriented information.
Current state-of-the-art methods follow the approach of Tian et al., which
gives the mid-level features the primary role and the high-level features the
secondary role. In this paper, we reinterpret this widely employed approach by
redifining the roles of the multi-level features; we swap the primary and
secondary roles. Specifically, we regard that the current methods improve the
initial estimate generated from the high-level features using the mid-level
features. This reinterpretation suggests a new application of the current
methods: to apply the same network multiple times to iteratively update the
estimate of the object's region, starting from its initial estimate. Our
experiments show that this method is effective and has updated the previous
state-of-the-art on COCO-20$^i$ in the 1-shot and 5-shot settings and on
PASCAL-5$^i$ in the 1-shot setting.

    

### [[2109.06474] Space Time Recurrent Memory Network](http://arxiv.org/abs/2109.06474)


  We propose a novel visual memory network architecture for the learning and
inference problem in the spatial-temporal domain. Different from the popular
transformers, we maintain a fixed set of memory slots in our memory network and
explore designs to input new information into the memory, combine the
information in different memory slots and decide when to discard old memory
slots. Finally, this architecture is benchmarked on the video object
segmentation and video prediction problems. Through the experiments, we show
that our memory architecture can achieve competitive results with
state-of-the-art while maintaining constant memory capacity.

    

### [[2109.06479] Large-scale Autonomous Flight with Real-time Semantic SLAM under Dense Forest Canopy](http://arxiv.org/abs/2109.06479)


  In this letter, we propose an integrated autonomous flight and semantic SLAM
system that can perform long-range missions and real-time semantic mapping in
highly cluttered, unstructured, and GPS-denied under-canopy environments.
First, tree trunks and ground planes are detected from LIDAR scans. We use a
neural network and an instance extraction algorithm to enable semantic
segmentation in real time onboard the UAV. Second, detected tree trunk
instances are modeled as cylinders and associated across the whole LIDAR
sequence. This semantic data association constraints both robot poses as well
as trunk landmark models. The output of semantic SLAM is used in state
estimation, planning, and control algorithms in real time. The global planner
relies on a sparse map to plan the shortest path to the global goal, and the
local trajectory planner uses a small but finely discretized robot-centric map
to plan a dynamically feasible and collision-free trajectory to the local goal.
Both the global path and local trajectory lead to drift-corrected goals, thus
helping the UAV execute its mission accurately and safely.

    

### [[2109.06480] Logic-level Evidence Retrieval and Graph-based Verification Network for Table-based Fact Verification](http://arxiv.org/abs/2109.06480)


  Table-based fact verification task aims to verify whether the given statement
is supported by the given semi-structured table. Symbolic reasoning with
logical operations plays a crucial role in this task. Existing methods leverage
programs that contain rich logical information to enhance the verification
process. However, due to the lack of fully supervised signals in the program
generation process, spurious programs can be derived and employed, which leads
to the inability of the model to catch helpful logical operations. To address
the aforementioned problems, in this work, we formulate the table-based fact
verification task as an evidence retrieval and reasoning framework, proposing
the Logic-level Evidence Retrieval and Graph-based Verification network
(LERGV). Specifically, we first retrieve logic-level program-like evidence from
the given table and statement as supplementary evidence for the table. After
that, we construct a logic-level graph to capture the logical relations between
entities and functions in the retrieved evidence, and design a graph-based
verification network to perform logic-level graph-based reasoning based on the
constructed graph to classify the final entailment relation. Experimental
results on the large-scale benchmark TABFACT show the effectiveness of the
proposed approach.

    

### [[2109.06481] AligNART: Non-autoregressive Neural Machine Translation by Jointly Learning to Estimate Alignment and Translate](http://arxiv.org/abs/2109.06481)


  Non-autoregressive neural machine translation (NART) models suffer from the
multi-modality problem which causes translation inconsistency such as token
repetition. Most recent approaches have attempted to solve this problem by
implicitly modeling dependencies between outputs. In this paper, we introduce
AligNART, which leverages full alignment information to explicitly reduce the
modality of the target distribution. AligNART divides the machine translation
task into $(i)$ alignment estimation and $(ii)$ translation with aligned
decoder inputs, guiding the decoder to focus on simplified one-to-one
translation. To alleviate the alignment estimation problem, we further propose
a novel alignment decomposition method. Our experiments show that AligNART
outperforms previous non-iterative NART models that focus on explicit modality
reduction on WMT14 En$\leftrightarrow$De and WMT16 Ro$\rightarrow$En.
Furthermore, AligNART achieves BLEU scores comparable to those of the
state-of-the-art connectionist temporal classification based models on WMT14
En$\leftrightarrow$De. We also observe that AligNART effectively addresses the
token repetition problem even without sequence-level knowledge distillation.

    

### [[2109.06505] Optimal To-Do List Gamification for Long Term Planning](http://arxiv.org/abs/2109.06505)


  Most people struggle with prioritizing work. While inexact heuristics have
been developed over time, there is still no tractable principled algorithm for
deciding which of the many possible tasks one should tackle in any given day,
month, week, or year. Additionally, some people suffer from cognitive biases
such as the present bias, leading to prioritization of their immediate
experience over long-term consequences which manifests itself as
procrastination and inefficient task prioritization. Our method utilizes
optimal gamification to help people overcome these problems by incentivizing
each task by a number of points that convey how valuable it is in the long-run.
We extend the previous version of our optimal gamification method with added
services for helping people decide which tasks should and should not be done
when there is not enough time to do everything. To improve the efficiency and
scalability of the to-do list solver, we designed a hierarchical procedure that
tackles the problem from the top-level goals to fine-grained tasks. We test the
accuracy of the incentivised to-do list by comparing the performance of the
strategy with the points computed exactly using Value Iteration for a variety
of case studies. These case studies were specifically designed to cover the
corner cases to get an accurate judge of performance. Our method yielded the
same performance as the exact method for all case studies. To demonstrate its
functionality, we released an API that makes it easy to deploy our method in
Web and app services. We assessed the scalability of our method by applying it
to to-do lists with increasingly larger numbers of goals, sub-goals per goal,
hierarchically nested levels of subgoals. We found that the method provided
through our API is able to tackle fairly large to-do lists having a 576 tasks.
This indicates that our method is suitable for real-world applications.

    

### [[2109.06515] Netmarble AI Center's WMT21 Automatic Post-Editing Shared Task Submission](http://arxiv.org/abs/2109.06515)


  This paper describes Netmarble's submission to WMT21 Automatic Post-Editing
(APE) Shared Task for the English-German language pair. First, we propose a
Curriculum Training Strategy in training stages. Facebook Fair's WMT19 news
translation model was chosen to engage the large and powerful pre-trained
neural networks. Then, we post-train the translation model with different
levels of data at each training stages. As the training stages go on, we make
the system learn to solve multiple tasks by adding extra information at
different training stages gradually. We also show a way to utilize the
additional data in large volume for APE tasks. For further improvement, we
apply Multi-Task Learning Strategy with the Dynamic Weight Average during the
fine-tuning stage. To fine-tune the APE corpus with limited data, we add some
related subtasks to learn a unified representation. Finally, for better
performance, we leverage external translations as augmented machine translation
(MT) during the post-training and fine-tuning. As experimental results show,
our APE system significantly improves the translations of provided MT results
by -2.848 and +3.74 on the development dataset in terms of TER and BLEU,
respectively. It also demonstrates its effectiveness on the test dataset with
higher quality than the development dataset.

    

### [[2109.06523] Dependability Analysis of Deep Reinforcement Learning based Robotics and Autonomous Systems](http://arxiv.org/abs/2109.06523)


  While Deep Reinforcement Learning (DRL) provides transformational
capabilities to the control of Robotics and Autonomous Systems (RAS), the
black-box nature of DRL and uncertain deployment-environments of RAS pose new
challenges on its dependability. Although there are many existing works
imposing constraints on the DRL policy to ensure a successful completion of the
mission, it is far from adequate in terms of assessing the DRL-driven RAS in a
holistic way considering all dependability properties. In this paper, we
formally define a set of dependability properties in temporal logic and
construct a Discrete-Time Markov Chain (DTMC) to model the dynamics of
risk/failures of a DRL-driven RAS interacting with the stochastic environment.
We then do Probabilistic Model Checking based on the designed DTMC to verify
those properties. Our experimental results show that the proposed method is
effective as a holistic assessment framework, while uncovers conflicts between
the properties that may need trade-offs in the training. Moreover, we find the
standard DRL training cannot improve dependability properties, thus requiring
bespoke optimisation objectives concerning them. Finally, our method offers a
novel dependability analysis to the Sim-to-Real challenge of DRL.

    

### [[2109.06543] Multi-Level Features Contrastive Networks for Unsupervised Domain Adaptation](http://arxiv.org/abs/2109.06543)


  Unsupervised domain adaptation aims to train a model from the labeled source
domain to make predictions on the unlabeled target domain when the data
distribution of the two domains is different. As a result, it needs to reduce
the data distribution difference between the two domains to improve the model's
generalization ability. Existing methods tend to align the two domains directly
at the domain-level, or perform class-level domain alignment based on deep
feature. The former ignores the relationship between the various classes in the
two domains, which may cause serious negative transfer, the latter alleviates
it by introducing pseudo-labels of the target domain, but it does not consider
the importance of performing class-level alignment on shallow feature
representations. In this paper, we develop this work on the method of
class-level alignment. The proposed method reduces the difference between two
domains dramaticlly by aligning multi-level features. In the case that the two
domains share the label space, the class-level alignment is implemented by
introducing Multi-Level Feature Contrastive Networks (MLFCNet). In practice,
since the categories of samples in target domain are unavailable, we
iteratively use clustering algorithm to obtain the pseudo-labels, and then
minimize Multi-Level Contrastive Discrepancy (MLCD) loss to achieve more
accurate class-level alignment. Experiments on three real-world benchmarks
ImageCLEF-DA, Office-31 and Office-Home demonstrate that MLFCNet compares
favorably against the existing state-of-the-art domain adaptation methods.

    

### [[2109.06554] Talking Space: inference from spatial linguistic meanings](http://arxiv.org/abs/2109.06554)


  This paper concerns the intersection of natural language and the physical
space around us in which we live, that we observe and/or imagine things within.
Many important features of language have spatial connotations, for example,
many prepositions (like in, next to, after, on, etc.) are fundamentally
spatial. Space is also a key factor of the meanings of many
words/phrases/sentences/text, and space is a, if not the key, context for
referencing (e.g. pointing) and embodiment.
We propose a mechanism for how space and linguistic structure can be made to
interact in a matching compositional fashion. Examples include Cartesian space,
subway stations, chesspieces on a chess-board, and Penrose's staircase. The
starting point for our construction is the DisCoCat model of compositional
natural language meaning, which we relax to accommodate physical space. We
address the issue of having multiple agents/objects in a space, including the
case that each agent has different capabilities with respect to that space,
e.g., the specific moves each chesspiece can make, or the different velocities
one may be able to reach.
Once our model is in place, we show how inferences drawing from the structure
of physical space can be made. We also how how linguistic model of space can
interact with other such models related to our senses and/or embodiment, such
as the conceptual spaces of colour, taste and smell, resulting in a rich
compositional model of meaning that is close to human experience and embodiment
in the world.

    

### [[2109.06580] Continuous Homeostatic Reinforcement Learning for Self-Regulated Autonomous Agents](http://arxiv.org/abs/2109.06580)


  Homeostasis is a prevalent process by which living beings maintain their
internal milieu around optimal levels. Multiple lines of evidence suggest that
living beings learn to act to predicatively ensure homeostasis (allostasis). A
classical theory for such regulation is drive reduction, where a function of
the difference between the current and the optimal internal state. The recently
introduced homeostatic regulated reinforcement learning theory (HRRL), by
defining within the framework of reinforcement learning a reward function based
on the internal state of the agent, makes the link between the theories of
drive reduction and reinforcement learning. The HRRL makes it possible to
explain multiple eating disorders. However, the lack of continuous change in
the internal state of the agent with the discrete-time modeling has been so far
a key shortcoming of the HRRL theory. Here, we propose an extension of the
homeostatic reinforcement learning theory to a continuous environment in space
and time, while maintaining the validity of the theoretical results and the
behaviors explained by the model in discrete time. Inspired by the
self-regulating mechanisms abundantly present in biology, we also introduce a
model for the dynamics of the agent internal state, requiring the agent to
continuously take actions to maintain homeostasis. Based on the
Hamilton-Jacobi-Bellman equation and function approximation with neural
networks, we derive a numerical scheme allowing the agent to learn directly how
its internal mechanism works, and to choose appropriate action policies via
reinforcement learning and an appropriate exploration of the environment. Our
numerical experiments show that the agent does indeed learn to behave in a way
that is beneficial to its survival in the environment, making our framework
promising for modeling animal dynamics and decision-making.

    

### [[2109.06584] Choosing the Right Algorithm With Hints From Complexity Theory](http://arxiv.org/abs/2109.06584)


  Choosing a suitable algorithm from the myriads of different search heuristics
is difficult when faced with a novel optimization problem. In this work, we
argue that the purely academic question of what could be the best possible
algorithm in a certain broad class of black-box optimizers can give fruitful
indications in which direction to search for good established optimization
heuristics. We demonstrate this approach on the recently proposed DLB
benchmark, for which the only known results are $O(n^3)$ runtimes for several
classic evolutionary algorithms and an $O(n^2 \log n)$ runtime for an
estimation-of-distribution algorithm. Our finding that the unary unbiased
black-box complexity is only $O(n^2)$ suggests the Metropolis algorithm as an
interesting candidate and we prove that it solves the DLB problem in quadratic
time. Since we also prove that better runtimes cannot be obtained in the class
of unary unbiased algorithms, we shift our attention to algorithms that use the
information of more parents to generate new solutions. An artificial algorithm
of this type having an $O(n \log n)$ runtime leads to the result that the
significance-based compact genetic algorithm (sig-cGA) can solve the DLB
problem also in time $O(n \log n)$. Our experiments show a remarkably good
performance of the Metropolis algorithm, clearly the best of all algorithms
regarded for reasonable problem sizes.

    

### [[2109.06598] Just What do You Think You're Doing, Dave?' A Checklist for Responsible Data Use in NLP](http://arxiv.org/abs/2109.06598)


  A key part of the NLP ethics movement is responsible use of data, but exactly
what that means or how it can be best achieved remain unclear. This position
paper discusses the core legal and ethical principles for collection and
sharing of textual data, and the tensions between them. We propose a potential
checklist for responsible data (re-)use that could both standardise the peer
review of conference submissions, as well as enable a more in-depth view of
published research across the community. Our proposal aims to contribute to the
development of a consistent standard for data (re-)use, embraced across NLP
conferences.

    

### [[2109.06652] Domain Adaptation by Maximizing Population Correlation with Neural Architecture Search](http://arxiv.org/abs/2109.06652)


  In Domain Adaptation (DA), where the feature distributions of the source and
target domains are different, various distance-based methods have been proposed
to minimize the discrepancy between the source and target domains to handle the
domain shift. In this paper, we propose a new similarity function, which is
called Population Correlation (PC), to measure the domain discrepancy for DA.
Base on the PC function, we propose a new method called Domain Adaptation by
Maximizing Population Correlation (DAMPC) to learn a domain-invariant feature
representation for DA. Moreover, most existing DA methods use hand-crafted
bottleneck networks, which may limit the capacity and flexibility of the
corresponding model. Therefore, we further propose a method called DAMPC with
Neural Architecture Search (DAMPC-NAS) to search the optimal network
architecture for DAMPC. Experiments on several benchmark datasets, including
Office-31, Office-Home, and VisDA-2017, show that the proposed DAMPC-NAS method
achieves better results than state-of-the-art DA methods.

    

### [[2109.06655] Improving Test Case Generation for REST APIs Through Hierarchical Clustering](http://arxiv.org/abs/2109.06655)


  With the ever-increasing use of web APIs in modern-day applications, it is
becoming more important to test the system as a whole. In the last decade,
tools and approaches have been proposed to automate the creation of
system-level test cases for these APIs using evolutionary algorithms (EAs). One
of the limiting factors of EAs is that the genetic operators (crossover and
mutation) are fully randomized, potentially breaking promising patterns in the
sequences of API requests discovered during the search. Breaking these patterns
has a negative impact on the effectiveness of the test case generation process.
To address this limitation, this paper proposes a new approach that uses
agglomerative hierarchical clustering (AHC) to infer a linkage tree model,
which captures, replicates, and preserves these patterns in new test cases. We
evaluate our approach, called LT-MOSA, by performing an empirical study on 7
real-world benchmark applications w.r.t. branch coverage and real-fault
detection capability. We also compare LT-MOSA with the two existing
state-of-the-art white-box techniques (MIO, MOSA) for REST API testing. Our
results show that LT-MOSA achieves a statistically significant increase in test
target coverage (i.e., lines and branches) compared to MIO and MOSA in 4 and 5
out of 7 applications, respectively. Furthermore, LT-MOSA discovers 27 and 18
unique real-faults that are left undetected by MIO and MOSA, respectively.

    

### [[2109.06705] A Novel Global Feature-Oriented Relational Triple Extraction Model based on Table Filling](http://arxiv.org/abs/2109.06705)


  Table filling based relational triple extraction methods are attracting
growing research interests due to their promising performance and their
abilities on extracting triples from complex sentences. However, this kind of
methods are far from their full potential because most of them only focus on
using local features but ignore the global associations of relations and of
token pairs, which increases the possibility of overlooking some important
information during triple extraction. To overcome this deficiency, we propose a
global feature-oriented triple extraction model that makes full use of the
mentioned two kinds of global associations. Specifically, we first generate a
table feature for each relation. Then two kinds of global associations are
mined from the generated table features. Next, the mined global associations
are integrated into the table feature of each relation. This
"generate-mine-integrate" process is performed multiple times so that the table
feature of each relation is refined step by step. Finally, each relation's
table is filled based on its refined table feature, and all triples linked to
this relation are extracted based on its filled table. We evaluate the proposed
model on three benchmark datasets. Experimental results show our model is
effective and it achieves state-of-the-art results on all of these datasets.
The source code of our work is available at: this https URL.

    

### [[2109.06714] Semantic Answer Type Prediction using BERT: IAI at the ISWC SMART Task 2020](http://arxiv.org/abs/2109.06714)


  This paper summarizes our participation in the SMART Task of the ISWC 2020
Challenge. A particular question we are interested in answering is how well
neural methods, and specifically transformer models, such as BERT, perform on
the answer type prediction task compared to traditional approaches. Our main
finding is that coarse-grained answer types can be identified effectively with
standard text classification methods, with over 95% accuracy, and BERT can
bring only marginal improvements. For fine-grained type detection, on the other
hand, BERT clearly outperforms previous retrieval-based approaches.

    

### [[2109.06717] Controllable Dialogue Generation with Disentangled Multi-grained Style Specification and Attribute Consistency Reward](http://arxiv.org/abs/2109.06717)


  Controllable text generation is an appealing but challenging task, which
allows users to specify particular attributes of the generated outputs. In this
paper, we propose a controllable dialogue generation model to steer response
generation under multi-attribute constraints. Specifically, we define and
categorize the commonly used control attributes into global and local ones,
which possess different granularities of effects on response generation. Then,
we significantly extend the conventional seq2seq framework by introducing a
novel two-stage decoder, which first uses a multi-grained style specification
layer to impose the stylistic constraints and determine word-level control
states of responses based on the attributes, and then employs a response
generation layer to generate final responses maintaining both semantic
relevancy to the contexts and fidelity to the attributes. Furthermore, we train
our model with an attribute consistency reward to promote response control with
explicit supervision signals. Extensive experiments and in-depth analyses on
two datasets indicate that our model can significantly outperform competitive
baselines in terms of response quality, content diversity and controllability.

    

### [[2109.06719] Sparse Fuzzy Attention for Structural Sentiment Analysis](http://arxiv.org/abs/2109.06719)


  Attention scorers have achieved success in parsing tasks like semantic and
syntactic dependency parsing. However, in tasks modeled into parsing, like
structural sentiment analysis, "dependency edges" are very sparse which hinders
parser performance. Thus we propose a sparse and fuzzy attention scorer with
pooling layers which improves parser performance and sets the new
state-of-the-art on structural sentiment analysis. We further explore the
parsing modeling on structural sentiment analysis with second-order parsing and
introduce a novel sparse second-order edge building procedure that leads to
significant improvement in parsing performance.

    

### [[2109.06740] Deceptive Decision-Making Under Uncertainty](http://arxiv.org/abs/2109.06740)


  We study the design of autonomous agents that are capable of deceiving
outside observers about their intentions while carrying out tasks in
stochastic, complex environments. By modeling the agent's behavior as a Markov
decision process, we consider a setting where the agent aims to reach one of
multiple potential goals while deceiving outside observers about its true goal.
We propose a novel approach to model observer predictions based on the
principle of maximum entropy and to efficiently generate deceptive strategies
via linear programming. The proposed approach enables the agent to exhibit a
variety of tunable deceptive behaviors while ensuring the satisfaction of
probabilistic constraints on the behavior. We evaluate the performance of the
proposed approach via comparative user studies and present a case study on the
streets of Manhattan, New York, using real travel time distributions.

    

### [[2109.06807] A Temporal Variational Model for Story Generation](http://arxiv.org/abs/2109.06807)


  Recent language models can generate interesting and grammatically correct
text in story generation but often lack plot development and long-term
coherence. This paper experiments with a latent vector planning approach based
on a TD-VAE (Temporal Difference Variational Autoencoder), using the model for
conditioning and reranking for text generation. The results demonstrate strong
performance in automatic cloze and swapping evaluations. The human judgments
show stories generated with TD-VAE reranking improve on a GPT-2 medium baseline
and show comparable performance to a hierarchical LSTM reranking model.
Conditioning on the latent vectors proves disappointing and deteriorates
performance in human evaluation because it reduces the diversity of generation,
and the models don't learn to progress the narrative. This highlights an
important difference between technical task performance (e.g. cloze) and
generating interesting stories.

    

### [[2109.06850] BenchIE: Open Information Extraction Evaluation Based on Facts, Not Tokens](http://arxiv.org/abs/2109.06850)


  Intrinsic evaluations of OIE systems are carried out either manually -- with
human evaluators judging the correctness of extractions -- or automatically, on
standardized benchmarks. The latter, while much more cost-effective, is less
reliable, primarily because of the incompleteness of the existing OIE
benchmarks: the ground truth extractions do not include all acceptable variants
of the same fact, leading to unreliable assessment of models' performance.
Moreover, the existing OIE benchmarks are available for English only. In this
work, we introduce BenchIE: a benchmark and evaluation framework for
comprehensive evaluation of OIE systems for English, Chinese and German. In
contrast to existing OIE benchmarks, BenchIE takes into account informational
equivalence of extractions: our gold standard consists of fact synsets,
clusters in which we exhaustively list all surface forms of the same fact. We
benchmark several state-of-the-art OIE systems using BenchIE and demonstrate
that these systems are significantly less effective than indicated by existing
OIE benchmarks. We make BenchIE (data and evaluation code) publicly available.

    

### [[2109.06860] Broaden the Vision: Geo-Diverse Visual Commonsense Reasoning](http://arxiv.org/abs/2109.06860)


  Commonsense is defined as the knowledge that is shared by everyone. However,
certain types of commonsense knowledge are correlated with culture and
geographic locations and they are only shared locally. For example, the
scenarios of wedding ceremonies vary across regions due to different customs
influenced by historical and religious factors. Such regional characteristics,
however, are generally omitted in prior work. In this paper, we construct a
Geo-Diverse Visual Commonsense Reasoning dataset (GD-VCR) to test
vision-and-language models' ability to understand cultural and
geo-location-specific commonsense. In particular, we study two state-of-the-art
Vision-and-Language models, VisualBERT and ViLBERT trained on VCR, a standard
multimodal commonsense benchmark with images primarily from Western regions. We
then evaluate how well the trained models can generalize to answering the
questions in GD-VCR. We find that the performance of both models for
non-Western regions including East Asia, South Asia, and Africa is
significantly lower than that for Western region. We analyze the reasons behind
the performance disparity and find that the performance gap is larger on QA
pairs that: 1) are concerned with culture-related scenarios, e.g., weddings,
religious activities, and festivals; 2) require high-level geo-diverse
commonsense reasoning rather than low-order perception and recognition. Dataset
and code are released at this https URL.

    

### [[2008.07682] Residual Learning from Demonstration: Adapting DMPs for Contact-rich Manipulation](http://arxiv.org/abs/2008.07682)


  Manipulation skills involving contact and friction are inherent to many
robotics tasks. Using the class of motor primitives for peg-in-hole like
insertions, we study how robots can learn such skills. Dynamic Movement
Primitives (DMP) are a popular way of extracting such policies through
behaviour cloning (BC), but can struggle in the context of insertion. Policy
adaptation strategies such as residual learning can help improve the overall
performance of policies in the context of contact-rich manipulation. However,
it is not clear how to best do this with DMPs. As result, we consider a number
of possible ways for adapting a DMP formulation and propose ``residual Learning
from Demonstration`` (rLfD), a framework that combines DMPs with Reinforcement
Learning (RL) to learn a residual correction policy. Our evaluations suggest
that applying residual learning directly in task space and operating on the
full pose of the robot can significantly improve the overall performance of
DMPs. We show that rLfD offers a gentle to the joints solution that improves
the task success and generalisation of DMPs. The proposed framework is
evaluated on a set of tasks in which a simulated robot and a real physical
robot arm have to successfully insert pegs, gears and plugs into their
respective sockets. Further material and videos accompanying this paper are
provided at this https URL.

    

### [[2011.01489] On Computing Stable Extensions of Abstract Argumentation Frameworks](http://arxiv.org/abs/2011.01489)


  An \textit{abstract argumentation framework} ({\sc af} for short) is a
directed graph $(A,R)$ where $A$ is a set of \textit{abstract arguments} and
$R\subseteq A \times A$ is the \textit{attack} relation. Let $H=(A,R)$ be an
{\sc af}, $S \subseteq A$ be a set of arguments and $S^+ = \{y \mid \exists
x\in S \text{ with }(x,y)\in R\}$. Then, $S$ is a \textit{stable extension} in
$H$ if and only if $S^+ = A\setminus S$. In this paper, we present a thorough,
formal validation of a known backtracking algorithm for listing all stable
extensions in a given {\sc af}.

    

### [[2101.04255] Quantum Mathematics in Artificial Intelligence](http://arxiv.org/abs/2101.04255)


  In the decade since 2010, successes in artificial intelligence have been at
the forefront of computer science and technology, and vector space models have
solidified a position at the forefront of artificial intelligence. At the same
time, quantum computers have become much more powerful, and announcements of
major advances are frequently in the news.
The mathematical techniques underlying both these areas have more in common
than is sometimes realized. Vector spaces took a position at the axiomatic
heart of quantum mechanics in the 1930s, and this adoption was a key motivation
for the derivation of logic and probability from the linear geometry of vector
spaces. Quantum interactions between particles are modelled using the tensor
product, which is also used to express objects and operations in artificial
neural networks.
This paper describes some of these common mathematical areas, including
examples of how they are used in artificial intelligence (AI), particularly in
automated reasoning and natural language processing (NLP). Techniques discussed
include vector spaces, scalar products, subspaces and implication, orthogonal
projection and negation, dual vectors, density matrices, positive operators,
and tensor products. Application areas include information retrieval,
categorization and implication, modelling word-senses and disambiguation,
inference in knowledge bases, and semantic composition.
Some of these approaches can potentially be implemented on quantum hardware.
Many of the practical steps in this implementation are in early stages, and
some are already realized. Explaining some of the common mathematical tools can
help researchers in both AI and quantum computing further exploit these
overlaps, recognizing and exploring new directions along the way.

    

### [[2102.12564] Triplet loss based embeddings for forensic speaker identification in Spanish](http://arxiv.org/abs/2102.12564)


  With the advent of digital technology, it is more common that committed
crimes or legal disputes involve some form of speech recording where the
identity of a speaker is questioned [1]. In face of this situation, the field
of forensic speaker identification has been looking to shed light on the
problem by quantifying how much a speech recording belongs to a particular
person in relation to a population. In this work, we explore the use of speech
embeddings obtained by training a CNN using the triplet loss. In particular, we
focus on the Spanish language which has not been extensively studies. We
propose extracting the embeddings from speech spectrograms samples, then
explore several configurations of such spectrograms, and finally, quantify the
embeddings quality. We also show some limitations of our data setting which is
predominantly composed by male speakers. At the end, we propose two approaches
to calculate the Likelihood Radio given out speech embeddings and we show that
triplet loss is a good alternative to create speech embeddings for forensic
speaker identification.

    

### [[2104.12379] Towards Visual Semantics](http://arxiv.org/abs/2104.12379)


  Lexical Semantics is concerned with how words encode mental representations
of the world, i.e., concepts . We call this type of concepts, classification
concepts . In this paper, we focus on Visual Semantics , namely on how humans
build concepts representing what they perceive visually. We call this second
type of concepts, substance concepts . As shown in the paper, these two types
of concepts are different and, furthermore, the mapping between them is
many-to-many. In this paper we provide a theory and an algorithm for how to
build substance concepts which are in a one-to-one correspondence with
classifications concepts, thus paving the way to the seamless integration
between natural language descriptions and visual perception. This work builds
upon three main intuitions: (i) substance concepts are modeled as visual
objects , namely sequences of similar frames, as perceived in multiple
encounters ; (ii) substance concepts are organized into a visual subsumption
hierarchy based on the notions of Genus and Differentia ; (iii) the human
feedback is exploited not to name objects, but, rather, to align the hierarchy
of substance concepts with that of classification concepts. The learning
algorithm is implemented for the base case of a hierarchy of depth two. The
experiments, though preliminary, show that the algorithm manages to acquire the
notions of Genus and Differentia with reasonable accuracy, this despite seeing
a small number of examples and receiving supervision on a fraction of them.

    

### [[2106.08556] Coreference-Aware Dialogue Summarization](http://arxiv.org/abs/2106.08556)


  Summarizing conversations via neural approaches has been gaining research
traction lately, yet it is still challenging to obtain practical solutions.
Examples of such challenges include unstructured information exchange in
dialogues, informal interactions between speakers, and dynamic role changes of
speakers as the dialogue evolves. Many of such challenges result in complex
coreference links. Therefore, in this work, we investigate different approaches
to explicitly incorporate coreference information in neural abstractive
dialogue summarization models to tackle the aforementioned challenges.
Experimental results show that the proposed approaches achieve state-of-the-art
performance, implying it is useful to utilize coreference information in
dialogue summarization. Evaluation results on factual correctness suggest such
coreference-aware models are better at tracing the information flow among
interlocutors and associating accurate status/actions with the corresponding
interlocutors and person mentions.

    

### [[2106.14431] Modelling Monotonic and Non-Monotonic Attribute Dependencies with Embeddings: A Theoretical Analysis](http://arxiv.org/abs/2106.14431)


  During the last decade, entity embeddings have become ubiquitous in
Artificial Intelligence. Such embeddings essentially serve as compact but
semantically meaningful representations of the entities of interest. In most
approaches, vectors are used for representing the entities themselves, as well
as for representing their associated attributes. An important advantage of
using attribute embeddings is that (some of the) semantic dependencies between
the attributes can thus be captured. However, little is known about what kinds
of semantic dependencies can be modelled in this way. The aim of this paper is
to shed light on this question, focusing on settings where the embedding of an
entity is obtained by pooling the embeddings of its known attributes. Our
particular focus is on studying the theoretical limitations of different
embedding strategies, rather than their ability to effectively learn attribute
dependencies in practice. We first show a number of negative results, revealing
that some of the most popular embedding models are not able to capture even
basic Horn rules. However, we also find that some embedding strategies are
capable, in principle, of modelling both monotonic and non-monotonic attribute
dependencies.

    

### [[2109.05704] Mitigating Language-Dependent Ethnic Bias in BERT](http://arxiv.org/abs/2109.05704)


  BERT and other large-scale language models (LMs) contain gender and racial
bias. They also exhibit other dimensions of social bias, most of which have not
been studied in depth, and some of which vary depending on the language. In
this paper, we study ethnic bias and how it varies across languages by
analyzing and mitigating ethnic bias in monolingual BERT for English, German,
Spanish, Korean, Turkish, and Chinese. To observe and quantify ethnic bias, we
develop a novel metric called Categorical Bias score. Then we propose two
methods for mitigation; first using a multilingual model, and second using
contextual word alignment of two monolingual models. We compare our proposed
methods with monolingual BERT and show that these methods effectively alleviate
the ethnic bias. Which of the two methods works better depends on the amount of
NLP resources available for that language. We additionally experiment with
Arabic and Greek to verify that our proposed methods work for a wider variety
of languages.

    

### [[2109.05877] Cardinality Estimation in DBMS: A Comprehensive Benchmark Evaluation](http://arxiv.org/abs/2109.05877)


  Cardinality estimation (CardEst) plays a significant role in generating
high-quality query plans for a query optimizer in DBMS. In the last decade, an
increasing number of advanced CardEst methods (especially ML-based) have been
proposed with outstanding estimation accuracy and inference latency. However,
there exists no study that systematically evaluates the quality of these
methods and answer the fundamental problem: to what extent can these methods
improve the performance of query optimizer in real-world settings, which is the
ultimate goal of a CardEst method. In this paper, we comprehensively and
systematically compare the effectiveness of CardEst methods in a real DBMS. We
establish a new benchmark for CardEst, which contains a new complex real-world
dataset STATS and a diverse query workload STATS-CEB. We integrate multiple
most representative CardEst methods into an open-source database system
PostgreSQL, and comprehensively evaluate their true effectiveness in improving
query plan quality, and other important aspects affecting their applicability,
ranging from inference latency, model size, and training time, to update
efficiency and accuracy. We obtain a number of key findings for the CardEst
methods, under different data and query settings. Furthermore, we find that the
widely used estimation accuracy metric(Q-Error) cannot distinguish the
importance of different sub-plan queries during query optimization and thus
cannot truly reflect the query plan quality generated by CardEst methods.
Therefore, we propose a new metric P-Error to evaluate the performance of
CardEst methods, which overcomes the limitation of Q-Error and is able to
reflect the overall end-to-end performance of CardEst methods. We have made all
of the benchmark data and evaluation code publicly available at
this https URL.

    

### [[2109.06493] Formal Methods for Quantum Programs: A Survey](http://arxiv.org/abs/2109.06493)


  While recent progress in quantum hardware open the door for significant
speedup in certain key areas (cryptography, biology, chemistry, optimization,
machine learning, etc), quantum algorithms are still hard to implement right,
and the validation of such quantum programs is achallenge. Moreover, importing
the testing and debugging practices at use in classical programming is
extremely difficult in the quantum case, due to the destructive aspect of
quantum measurement. As an alternative strategy, formal methods are prone to
play a decisive role in the emerging field of quantum software. Recent works
initiate solutions for problems occurring at every stage of the development
process: high-level program design, implementation, compilation, etc. We review
the induced challenges for an efficient use of formal methods in quantum
computing and the current most promising research directions.

    

### [[2109.06557] The concept of class invariant in object-oriented programming](http://arxiv.org/abs/2109.06557)


  Class invariants -- consistency constraints preserved by every operation on
objects of a given type -- are fundamental to building and understanding
object-oriented programs. They should also be a key help in verifying them, but
turn out instead to raise major verification challenges which have prompted a
significant literature with, until now, no widely accepted solution. The
present work introduces a general proof rule meant to address invariant-related
issues and allow verification tools benefit from invariants. It first clarifies
the notion of invariant and identify the three problems: callbacks, furtive
access and reference leak. As an example, the 2016 Ethereum DAO bug, in which
\$50 million were stolen, resulted from a callback invalidating an invariant.
The discussion starts with a "Simple Model" and an associated proof rule,
demonstrating its soundness. It then removes one by one the three assumptions
of the Simple Model, each removal bringing up one of the three issues, and
introduces the corresponding adaptation to the proof rule. The final version of
the rule can tackle tricky examples, including "challenge problems" listed in
the literature.

    

### [[1612.02547] Self-composable Programming](http://arxiv.org/abs/1612.02547)


  Many variability management techniques rely on sophisticated language
extension or tools to support it. While this can provide dedicated syntax and
operational mechanism but it struggling practical adaptation for the cost of
adapting new technology as part of development process. We present
Self-composable Programming, a language-driven, composition-based variability
implementation which takes an object-oriented approach to modeling and
composing behaviors in software. Self-composable Programming introduces
hierarchical relationship of behavior by providing concepts of abstract
function, which modularise commonalities, and specific function which inherits
from abstract function and be apply refinement to contain variabilities to
fulfill desired functionality. Various object-oriented techniques can
applicable in the refinement process including explicit method-based, and
implicit traits-based refinement. In order to evaluate the potential
independence of behavior from the object by applying object-orientation to
function, we compare it to Aspect-oriented Programming both conceptually and
empirically.

    

### [[1707.02590] Refinable Function : An Object-oriented Approach to Procedure Modularity](http://arxiv.org/abs/1707.02590)


  Modularity is the fundamental aspect of modern software engineering, however
many advanced modularity techniques requires prospective technologies as part
of development and operation process. In this paper, we present Refinable
Function, an object-oriented approach to advanced language-based, symmetric
modularity technique for the procedure. We conceptually compare Refinable
Function to existing technique to substantiate benefits of modularity can be
implemented in on well-established object-oriented language without compiler
support. We introduce concepts of inheritance, encapsulation, and polymorphism
of function for bringing object-orientation to procedure modularity and
describe the design and implementation of Refinable Function in JavaScript to
validate our approach to practical web application development. We introduce
the practical aspect of Refinable Function implementation by discussing
concerns of applying modularity on asynchronous processing. We tested and
implemented Refinable Function to substantiate its relevance to web application
development and product line implementation.

    

### [[2006.10604] Compositional theories for host-core languages](http://arxiv.org/abs/2006.10604)


  The syntax of a host-core language is spitted in two parts, representing
respectively a \emph{host language} H and a \emph{core language} C, embedded in
H.
This idea is rooted in Benton's Linear/non Linear formulation of Linear logic
and allows a flexible management of data linearity, which is particularly
useful in non-classical computational paradigms. Moreover, the host-core style
can be viewed as a simplified notion of multi-language programming, the process
of software development in a heterogeneous programming language. In this paper,
we present the typed calculus HC0, a minimal and flexible host-core system that
captures and standardizes common properties of an ideal class of host-core
languages. We provide a denotation in terms of enriched categories and we state
a strong correspondence between syntax and semantics through the notion of
\emph{internal language}. The latter result provides some interesting
characterizations of host-core style, otherwise difficult to achieve. We also
show discuss some concrete instances, extensions and specializations of HC0.

    

### [[2011.13127] Copy-and-Patch Binary Code Generation](http://arxiv.org/abs/2011.13127)


  Fast compilation is important when compilation occurs at runtime, such as
query compilers in modern database systems and WebAssembly virtual machines in
modern browsers. We present copy-and-patch, an extremely fast compilation
technique that also produces good quality code. It is capable of lowering both
high-level languages and low-level bytecode programs to binary code, by
stitching together code from a large library of binary implementation variants.
We call these binary implementations stencils because they have holes where
missing values must be inserted during code generation. We show how to
construct a stencil library and describe the copy-and-patch algorithm that
generates optimized binary code.
We demonstrate two use cases of copy-and-patch: a compiler for a high-level
C-like language intended for metaprogramming and a compiler for WebAssembly.
Our high-level language compiler has negligible compilation cost: it produces
code from an AST in less time than it takes to construct the AST. We have
implemented an SQL database query compiler on top of this metaprogramming
system and show that on TPC-H database benchmarks, copy-and-patch generates
code two orders of magnitude faster than LLVM -O0 and three orders of magnitude
faster than higher optimization levels. The generated code runs an order of
magnitude faster than interpretation and 14% faster than LLVM -O0. Our
WebAssembly compiler generates code 4.9X-6.5X faster than Liftoff, the
WebAssembly baseline compiler in Google Chrome. The generated code also
outperforms Liftoff's by 39%-63% on the Coremark and PolyBenchC WebAssembly
benchmarks.

    

### [[2106.12496] Threaded Code Generation with a Meta-tracing JIT Compiler](http://arxiv.org/abs/2106.12496)


  Language implementation frameworks such as RPython and Truffle/Graal are
effective tools for creating a high-performance language with lower effort than
implementing from scratch. The two frameworks support only a single JIT
compilation strategy, trace-based compilation and method-based compilation, but
they have its own advantages and disadvantages. We proposed a meta-hybrid JIT
compiler framework to take advantages of the two strategies as a language
implementation framework. We also implemented a proof-of-concept framework
called BacCaml. As a next step, in this position paper, we propose a new
approach to realize a method-based baseline JIT compiler along with a
trace-based JIT compilation. We aim to use it for further speed-up by
preventing the path-divergence problem, which causes serious slow-down. We also
show how to implement the baseline JIT compiler with minimal changes on top of
RPython.

    

### [[2109.06267] Are Group Acknowledgements Worth Anything in IEEE 802.15.4 DSME: A Comparative Analysis](http://arxiv.org/abs/2109.06267)


  For data collection scenarios in the Industrial Internet of Things, wireless
communication provides a cost-effective and easy-to-deploy alternative to wired
networks. The main focus lies on energy efficiency and reliability, as many
devices are battery operated. IEEE 802.15.4 DSME enhances reliability by
acknowledging each packet individually, imposing an overhead for each
transmitted packet, and increasing energy consumption. In networks with little
interference, it may be beneficial to aggregate the acknowledgments for
multiple nodes and broadcast them in a compressed format to all nodes in the
neighborhood. The IEEE 802.15.4 2012 standard describes such a group
acknowledgment scheme which, however, disappears in later iterations of the
standard. This paper compares different group acknowledgment schemes and
proposes a novel group acknowledgment scheme with the goal to examine whether
group acknowledgments constitute a viable alternative to regular
acknowledgments in reliable data-collection scenarios. Our analysis suggests
that apart from a few cases, GACKs do not constitute a valid alternative to the
direct acknowledgement of data packets.

    

### [[2109.06855] Timely Status Updating Over Erasure Channels Using an Energy Harvesting Sensor: Single and Multiple Sources](http://arxiv.org/abs/2109.06855)


  A status updating system is considered in which data from multiple sources
are sampled by an energy harvesting sensor and transmitted to a remote
destination through an erasure channel. The goal is to deliver status updates
of all sources in a timely manner, such that the cumulative long-term average
age-of-information (AoI) is minimized. The AoI for each source is defined as
the time elapsed since the generation time of the latest successful status
update received at the destination from that source. Transmissions are subject
to energy availability, which arrives in units according to a Poisson process,
with each energy unit capable of carrying out one transmission from only one
source. The sensor is equipped with a unit-sized battery to save the incoming
energy. A scheduling policy is designed in order to determine which source is
sampled using the available energy. The problem is studied in two main
settings: no erasure status feedback, and perfect instantaneous feedback.

    

### [[2109.06858] CyberBunker 2.0 -- A Domain and Traffic Perspective on a Bulletproof Hoster](http://arxiv.org/abs/2109.06858)


  In September 2019, 600 armed German cops seized the physical premise of a
Bulletproof Hoster (BPH) referred to as CyberBunker 2.0. The hoster resided in
a decommissioned NATO bunker and advertised to host everything but child porn
and anything related to terrorism while keeping servers online no matter what.
While the anatomy, economics and interconnection-level characteristics of BPHs
are studied, their traffic characteristics are unknown. In this poster, we
present the first analysis of domains, web pages, and traffic captured at a
major tier-1 ISP and a large IXP at the time when the CyberBunker was in
operation. Our study sheds light on traffic characteristics of a BPH in
operation. We show that a traditional BGP-based BPH identification approach
cannot detect the CyberBunker, but find characteristics from a domain and
traffic perspective that can add to future identification approaches.

    

### [[2109.06867] On Decentralized Multi-Transmitter Coded Caching](http://arxiv.org/abs/2109.06867)


  This paper investigates a setup consisting of multiple transmitters serving
multiple cache-enabled clients through a linear network, which covers both
wired and wireless transmission situations. We investigate decentralized coded
caching scenarios in which there is either no cooperation or limited
cooperation between the clients at the cache content placement phase. For the
fully decentralized caching case (i.e., no cooperation) we analyze the
performance of the system in terms of the Coding Delay metric. Furthermore, we
investigate a hybrid cache content placement scenario in which there are two
groups of users with different cache content placement situations (i.e.,
limited cooperation). Also, we examine the effect of finite file size in above
scenarios.

    

### [<title data-react-helmet="true">MECT——基于多元数据的中文NER涨点神器 - 知乎</title>](https://zhuanlan.zhihu.com/p/410326937)