
## 2021-8-26

### [<title>Sample_weight causes strange classifier performance issues - XGBoost</title>](https://discuss.xgboost.ai/t/sample-weight-causes-strange-classifier-performance-issues/2447/1)

### [[2108.11365] ML-Assisted UE Positioning: Performance Analysis and 5G Architecture Enhancements](http://arxiv.org/abs/2108.11365)


  Artificial intelligence and data-driven networks will be integral part of 6G
systems. In this article, we comprehensively discuss implementation challenges
and need for architectural changes in 5G radio access networks for integrating
machine learning (ML) solutions. As an example use case, we investigate user
equipment (UE) positioning assisted by deep learning (DL) in 5G and beyond
networks. As compared to state of the art positioning algorithms used in
today's networks, radio signal fingerprinting and machine learning (ML)
assisted positioning requires smaller additional feedback overhead; and the
positioning estimates are made directly inside the radio access network (RAN),
thereby assisting in radio resource management. In this regard, we study
ML-assisted positioning methods and evaluate their performance using system
level simulations for an outdoor scenario. The study is based on the use of
raytracing tool, a 3GPP 5G NR compliant system level simulator and DL framework
to estimate positioning accuracy of the UE. We evaluate and compare performance
of various DL models and show mean positioning error in the range of 1-1.5m for
a 2-hidden layer DL architecture with appropriate feature-modeling. Building on
our performance analysis, we discuss pros and cons of various architectures to
implement ML solutions for future networks and draw conclusions on the most
suitable architecture.

    

### [[2108.10904] SimVLM: Simple Visual Language Model Pretraining with Weak Supervision](http://arxiv.org/abs/2108.10904)


  With recent progress in joint modeling of visual and textual representations,
Vision-Language Pretraining (VLP) has achieved impressive performance on many
multimodal downstream tasks. However, the requirement for expensive annotations
including clean image captions and regional labels limits the scalability of
existing approaches, and complicates the pretraining procedure with the
introduction of multiple dataset-specific objectives. In this work, we relax
these constraints and present a minimalist pretraining framework, named Simple
Visual Language Model (SimVLM). Unlike prior work, SimVLM reduces the training
complexity by exploiting large-scale weak supervision, and is trained
end-to-end with a single prefix language modeling objective. Without utilizing
extra data or task-specific customization, the resulting model significantly
outperforms previous pretraining methods and achieves new state-of-the-art
results on a wide range of discriminative and generative vision-language
benchmarks, including VQA (+3.74% vqa-score), NLVR2 (+1.17% accuracy), SNLI-VE
(+1.37% accuracy) and image captioning tasks (+10.1% average CIDEr score).
Furthermore, we demonstrate that SimVLM acquires strong generalization and
transfer ability, enabling zero-shot behavior including open-ended visual
question answering and cross-modality transfer.

    

### [[2108.10908] GGNB: Graph-Based Gaussian Naive Bayes Intrusion Detection System for CAN Bus](http://arxiv.org/abs/2108.10908)


  The national highway traffic safety administration (NHTSA) identified
cybersecurity of the automobile systems are more critical than the security of
other information systems. Researchers already demonstrated remote attacks on
critical vehicular electronic control units (ECUs) using controller area
network (CAN). Besides, existing intrusion detection systems (IDSs) often
propose to tackle a specific type of attack, which may leave a system
vulnerable to numerous other types of attacks. A generalizable IDS that can
identify a wide range of attacks within the shortest possible time has more
practical value than attack-specific IDSs, which is not a trivial task to
accomplish. In this paper we propose a novel {\textbf g}raph-based {\textbf
G}aussian {\textbf n}aive {\textbf B}ayes (GGNB) intrusion detection algorithm
by leveraging graph properties and PageRank-related features. The GGNB on the
real rawCAN data set~\cite{Lee:2017} yields 99.61\%, 99.83\%, 96.79\%, and
96.20\% detection accuracy for denial of service (DoS), fuzzy, spoofing,
replay, mixed attacks, respectively. Also, using OpelAstra data
set~\cite{Guillaume:2019}, the proposed methodology has 100\%, 99.85\%,
99.92\%, 100\%, 99.92\%, 97.75\% and 99.57\% detection accuracy considering
DoS, diagnostic, fuzzing CAN ID, fuzzing payload, replay, suspension, and mixed
attacks, respectively. The GGNB-based methodology requires about $239\times$
and $135\times$ lower training and tests times, respectively, compared to the
SVM classifier used in the same application. Using Xilinx Zybo Z7
field-programmable gate array (FPGA) board, the proposed GGNB requires $5.7
\times$, $5.9 \times$, $5.1 \times$, and $3.6 \times$ fewer slices, LUTs,
flip-flops, and DSP units, respectively, than conventional NN architecture.

    

### [[2108.10921] The Word is Mightier than the Label Learning without Pointillistic Labels using Data Programming](http://arxiv.org/abs/2108.10921)


  Most advanced supervised Machine Learning (ML) models rely on vast amounts of
point-by-point labelled training examples. Hand-labelling vast amounts of data
may be tedious, expensive, and error-prone. Recently, some studies have
explored the use of diverse sources of weak supervision to produce competitive
end model classifiers. In this paper, we survey recent work on weak
supervision, and in particular, we investigate the Data Programming (DP)
framework. Taking a set of potentially noisy heuristics as input, DP assigns
denoised probabilistic labels to each data point in a dataset using a
probabilistic graphical model of heuristics. We analyze the math fundamentals
behind DP and demonstrate the power of it by applying it on two real-world text
classification tasks. Furthermore, we compare DP with pointillistic active and
semi-supervised learning techniques traditionally applied in data-sparse
settings.

    

### [[2108.10934] Bias Mitigated Learning from Differentially Private Synthetic Data: A Cautionary Tale](http://arxiv.org/abs/2108.10934)


  Increasing interest in privacy-preserving machine learning has led to new
models for synthetic private data generation from undisclosed real data.
However, mechanisms of privacy preservation introduce artifacts in the
resulting synthetic data that have a significant impact on downstream tasks
such as learning predictive models or inference. In particular, bias can affect
all analyses as the synthetic data distribution is an inconsistent estimate of
the real-data distribution. We propose several bias mitigation strategies using
privatized likelihood ratios that have general applicability to differentially
private synthetic data generative models. Through large-scale empirical
evaluation, we show that bias mitigation provides simple and effective
privacy-compliant augmentation for general applications of synthetic data.
However, the work highlights that even after bias correction significant
challenges remain on the usefulness of synthetic private data generators for
tasks such as prediction and inference.

    

### [[2108.10990] Online Dictionary Learning Based Fault and Cyber Attack Detection for Power Systems](http://arxiv.org/abs/2108.10990)


  The emerging wide area monitoring systems (WAMS) have brought significant
improvements in electric grids' situational awareness. However, the newly
introduced system can potentially increase the risk of cyber-attacks, which may
be disguised as normal physical disturbances. This paper deals with the event
and intrusion detection problem by leveraging a stream data mining classifier
(Hoeffding adaptive tree) with semi-supervised learning techniques to
distinguish cyber-attacks from regular system perturbations accurately. First,
our proposed approach builds a dictionary by learning higher-level features
from unlabeled data. Then, the labeled data are represented as sparse linear
combinations of learned dictionary atoms. We capitalize on those sparse codes
to train the online classifier along with efficient change detectors. We
conduct numerical experiments with industrial control systems cyber-attack
datasets. We consider five different scenarios: short-circuit faults, line
maintenance, remote tripping command injection, relay setting change, as well
as false data injection. The data are generated based on a modified IEEE 9-bus
system. Simulation results show that our proposed approach outperforms the
state-of-the-art method.

    

### [[2108.11000] Layer Adaptive Node Selection in Bayesian Neural Networks: Statistical Guarantees and Implementation Details](http://arxiv.org/abs/2108.11000)


  Sparse deep neural networks have proven to be efficient for predictive model
building in large-scale studies. Although several works have studied
theoretical and numerical properties of sparse neural architectures, they have
primarily focused on the edge selection. Sparsity through edge selection might
be intuitively appealing; however, it does not necessarily reduce the
structural complexity of a network. Instead pruning excessive nodes in each
layer leads to a structurally sparse network which would have lower
computational complexity and memory footprint. We propose a Bayesian sparse
solution using spike-and-slab Gaussian priors to allow for node selection
during training. The use of spike-and-slab prior alleviates the need of an
ad-hoc thresholding rule for pruning redundant nodes from a network. In
addition, we adopt a variational Bayes approach to circumvent the computational
challenges of traditional Markov Chain Monte Carlo (MCMC) implementation. In
the context of node selection, we establish the fundamental result of
variational posterior consistency together with the characterization of prior
parameters. In contrast to the previous works, our theoretical development
relaxes the assumptions of the equal number of nodes and uniform bounds on all
network weights, thereby accommodating sparse networks with layer-dependent
node structures or coefficient bounds. With a layer-wise characterization of
prior inclusion probabilities, we also discuss optimal contraction rates of the
variational posterior. Finally, we provide empirical evidence to substantiate
that our theoretical work facilitates layer-wise optimal node recovery together
with competitive predictive performance.

    

### [[2108.11005] Wanderlust: Online Continual Object Detection in the Real World](http://arxiv.org/abs/2108.11005)


  Online continual learning from data streams in dynamic environments is a
critical direction in the computer vision field. However, realistic benchmarks
and fundamental studies in this line are still missing. To bridge the gap, we
present a new online continual object detection benchmark with an egocentric
video dataset, Objects Around Krishna (OAK). OAK adopts the KrishnaCAM videos,
an ego-centric video stream collected over nine months by a graduate student.
OAK provides exhaustive bounding box annotations of 80 video snippets (~17.5
hours) for 105 object categories in outdoor scenes. The emergence of new object
categories in our benchmark follows a pattern similar to what a single person
might see in their day-to-day life. The dataset also captures the natural
distribution shifts as the person travels to different places. These egocentric
long-running videos provide a realistic playground for continual learning
algorithms, especially in online embodied settings. We also introduce new
evaluation metrics to evaluate the model performance and catastrophic
forgetting and provide baseline studies for online continual object detection.
We believe this benchmark will pose new exciting challenges for learning from
non-stationary data in continual learning. The OAK dataset and the associated
benchmark are released at this https URL.

    

### [[2108.11010] Adversary agent reinforcement learning for pursuit-evasion](http://arxiv.org/abs/2108.11010)


  A reinforcement learning environment with adversary agents is proposed in
this work for pursuit-evasion game in the presence of fog of war, which is of
both scientific significance and practical importance in aerospace
applications. One of the most popular learning environments, StarCraft, is
adopted here and the associated mini-games are analyzed to identify the current
limitation for training adversary agents. The key contribution includes the
analysis of the potential performance of an agent by incorporating control and
differential game theory into the specific reinforcement learning environment,
and the development of an adversary agents challenge (SAAC) environment by
extending the current StarCraft mini-games. The subsequent study showcases the
use of this learning environment and the effectiveness of an adversary agent
for evasion units. Overall, the proposed SAAC environment should benefit
pursuit-evasion studies with rapidly-emerging reinforcement learning
technologies. Last but not least, the corresponding tutorial code can be found
at GitHub.

    

### [[2108.11012] Responsive Regulation of Dynamic UAV Communication Networks Based on Deep Reinforcement Learning](http://arxiv.org/abs/2108.11012)


  In this chapter, the regulation of Unmanned Aerial Vehicle (UAV)
communication network is investigated in the presence of dynamic changes in the
UAV lineup and user distribution. We target an optimal UAV control policy which
is capable of identifying the upcoming change in the UAV lineup (quit or
join-in) or user distribution, and proactively relocating the UAVs ahead of the
change rather than passively dispatching the UAVs after the change.
Specifically, a deep reinforcement learning (DRL)-based UAV control framework
is developed to maximize the accumulated user satisfaction (US) score for a
given time horizon which is able to handle the change in both the UAV lineup
and user distribution. The framework accommodates the changed dimension of the
state-action space before and after the UAV lineup change by deliberate state
transition design. In addition, to handle the continuous state and action
space, deep deterministic policy gradient (DDPG) algorithm, which is an
actor-critic based DRL, is exploited. Furthermore, to promote the learning
exploration around the timing of the change, the original DDPG is adapted into
an asynchronous parallel computing (APC) structure which leads to a better
training performance in both the critic and actor networks. Finally, extensive
simulations are conducted to validate the convergence of the proposed learning
approach, and demonstrate its capability in jointly handling the dynamics in
UAV lineup and user distribution as well as its superiority over a passive
reaction method.

    

### [[2108.11018] A Scaling Law for Synthetic-to-Real Transfer: A Measure of Pre-Training](http://arxiv.org/abs/2108.11018)


  Synthetic-to-real transfer learning is a framework in which we pre-train
models with synthetically generated images and ground-truth annotations for
real tasks. Although synthetic images overcome the data scarcity issue, it
remains unclear how the fine-tuning performance scales with pre-trained models,
especially in terms of pre-training data size. In this study, we collect a
number of empirical observations and uncover the secret. Through experiments,
we observe a simple and general scaling law that consistently describes
learning curves in various tasks, models, and complexities of synthesized
pre-training data. Further, we develop a theory of transfer learning for a
simplified scenario and confirm that the derived generalization bound is
consistent with our empirical findings.

    

### [[2108.11019] Vector Transport Free Riemannian LBFGS for Optimization on Symmetric Positive Definite Matrix Manifolds](http://arxiv.org/abs/2108.11019)


  This work concentrates on optimization on Riemannian manifolds. The
Limited-memory Broyden-Fletcher-Goldfarb-Shanno (LBFGS) algorithm is a commonly
used quasi-Newton method for numerical optimization in Euclidean spaces.
Riemannian LBFGS (RLBFGS) is an extension of this method to Riemannian
manifolds. RLBFGS involves computationally expensive vector transports as well
as unfolding recursions using adjoint vector transports. In this article, we
propose two mappings in the tangent space using the inverse second root and
Cholesky decomposition. These mappings make both vector transport and adjoint
vector transport identity and therefore isometric. Identity vector transport
makes RLBFGS less computationally expensive and its isometry is also very
useful in convergence analysis of RLBFGS. Moreover, under the proposed
mappings, the Riemannian metric reduces to Euclidean inner product, which is
much less computationally expensive. We focus on the Symmetric Positive
Definite (SPD) manifolds which are beneficial in various fields such as data
science and statistics. This work opens a research opportunity for extension of
the proposed mappings to other well-known manifolds.

    

### [[2108.11022] Tree Decomposed Graph Neural Network](http://arxiv.org/abs/2108.11022)


  Graph Neural Networks (GNNs) have achieved significant success in learning
better representations by performing feature propagation and transformation
iteratively to leverage neighborhood information. Nevertheless, iterative
propagation restricts the information of higher-layer neighborhoods to be
transported through and fused with the lower-layer neighborhoods', which
unavoidably results in feature smoothing between neighborhoods in different
layers and can thus compromise the performance, especially on heterophily
networks. Furthermore, most deep GNNs only recognize the importance of
higher-layer neighborhoods while yet to fully explore the importance of
multi-hop dependency within the context of different layer neighborhoods in
learning better representations. In this work, we first theoretically analyze
the feature smoothing between neighborhoods in different layers and empirically
demonstrate the variance of the homophily level across neighborhoods at
different layers. Motivated by these analyses, we further propose a tree
decomposition method to disentangle neighborhoods in different layers to
alleviate feature smoothing among these layers. Moreover, we characterize the
multi-hop dependency via graph diffusion within our tree decomposition
formulation to construct Tree Decomposed Graph Neural Network (TDGNN), which
can flexibly incorporate information from large receptive fields and aggregate
this information utilizing the multi-hop dependency. Comprehensive experiments
demonstrate the superior performance of TDGNN on both homophily and heterophily
networks under a variety of node classification settings. Extensive parameter
analysis highlights the ability of TDGNN to prevent over-smoothing and
incorporate features from shallow layers with deeper multi-hop dependencies,
which provides new insights towards deeper graph neural networks. Code of
TDGNN: this http URL


### [[2108.11023] EncoderMI: Membership Inference against Pre-trained Encoders in Contrastive Learning](http://arxiv.org/abs/2108.11023)


  Given a set of unlabeled images or (image, text) pairs, contrastive learning
aims to pre-train an image encoder that can be used as a feature extractor for
many downstream tasks. In this work, we propose EncoderMI, the first membership
inference method against image encoders pre-trained by contrastive learning. In
particular, given an input and a black-box access to an image encoder,
EncoderMI aims to infer whether the input is in the training dataset of the
image encoder. EncoderMI can be used 1) by a data owner to audit whether its
(public) data was used to pre-train an image encoder without its authorization
or 2) by an attacker to compromise privacy of the training data when it is
private/sensitive. Our EncoderMI exploits the overfitting of the image encoder
towards its training data. In particular, an overfitted image encoder is more
likely to output more (or less) similar feature vectors for two augmented
versions of an input in (or not in) its training dataset. We evaluate EncoderMI
on image encoders pre-trained on multiple datasets by ourselves as well as the
Contrastive Language-Image Pre-training (CLIP) image encoder, which is
pre-trained on 400 million (image, text) pairs collected from the Internet and
released by OpenAI. Our results show that EncoderMI can achieve high accuracy,
precision, and recall. We also explore a countermeasure against EncoderMI via
preventing overfitting through early stopping. Our results show that it
achieves trade-offs between accuracy of EncoderMI and utility of the image
encoder, i.e., it can reduce the accuracy of EncoderMI, but it also incurs
classification accuracy loss of the downstream classifiers built based on the
image encoder.

    

### [[2108.11032] Improving Visual Quality of Unrestricted Adversarial Examples with Wavelet-VAE](http://arxiv.org/abs/2108.11032)


  Traditional adversarial examples are typically generated by adding
perturbation noise to the input image within a small matrix norm. In practice,
un-restricted adversarial attack has raised great concern and presented a new
threat to the AI safety. In this paper, we propose a wavelet-VAE structure to
reconstruct an input image and generate adversarial examples by modifying the
latent code. Different from perturbation-based attack, the modifications of the
proposed method are not limited but imperceptible to human eyes. Experiments
show that our method can generate high quality adversarial examples on ImageNet
dataset.

    

### [[2108.11033] GRIM: A General, Real-Time Deep Learning Inference Framework for Mobile Devices based on Fine-Grained Structured Weight Sparsity](http://arxiv.org/abs/2108.11033)


  It is appealing but challenging to achieve real-time deep neural network
(DNN) inference on mobile devices because even the powerful modern mobile
devices are considered as ``resource-constrained'' when executing large-scale
DNNs. It necessitates the sparse model inference via weight pruning, i.e., DNN
weight sparsity, and it is desirable to design a new DNN weight sparsity scheme
that can facilitate real-time inference on mobile devices while preserving a
high sparse model accuracy. This paper designs a novel mobile inference
acceleration framework GRIM that is General to both convolutional neural
networks (CNNs) and recurrent neural networks (RNNs) and that achieves
Real-time execution and high accuracy, leveraging fine-grained structured
sparse model Inference and compiler optimizations for Mobiles. We start by
proposing a new fine-grained structured sparsity scheme through the Block-based
Column-Row (BCR) pruning. Based on this new fine-grained structured sparsity,
our GRIM framework consists of two parts: (a) the compiler optimization and
code generation for real-time mobile inference; and (b) the BCR pruning
optimizations for determining pruning hyperparameters and performing weight
pruning. We compare GRIM with Alibaba MNN, TVM, TensorFlow-Lite, a sparse
implementation based on CSR, PatDNN, and ESE (a representative FPGA inference
acceleration framework for RNNs), and achieve up to 14.08x speedup.

    

### [[2108.11034] Natural Language Processing Accurately Categorizes Indications, Findings and Pathology Reports from Multicenter Colonoscopy](http://arxiv.org/abs/2108.11034)


  Colonoscopy is used for colorectal cancer (CRC) screening. Extracting details
of the colonoscopy findings from free text in electronic health records (EHRs)
can be used to determine patient risk for CRC and colorectal screening
strategies. We developed and evaluated the accuracy of a deep learning model
framework to extract information for the clinical decision support system to
interpret relevant free-text reports, including indications, pathology, and
findings notes. The Bio-Bi-LSTM-CRF framework was developed using Bidirectional
Long Short-term Memory (Bi-LSTM) and Conditional Random Fields (CRF) to extract
several clinical features from these free-text reports including indications
for the colonoscopy, findings during the colonoscopy, and pathology of resected
material. We trained the Bio-Bi-LSTM-CRF and existing Bi-LSTM-CRF models on 80%
of 4,000 manually annotated notes from 3,867 patients. These clinical notes
were from a group of patients over 40 years of age enrolled in four Veterans
Affairs Medical Centers. A total of 10% of the remaining annotated notes were
used to train hyperparameter and the remaining 10% were used to evaluate the
accuracy of our model Bio-Bi-LSTM-CRF and compare to Bi-LSTM-CRF.

    

### [[2108.11035] NGC: A Unified Framework for Learning with Open-World Noisy Data](http://arxiv.org/abs/2108.11035)


  The existence of noisy data is prevalent in both the training and testing
phases of machine learning systems, which inevitably leads to the degradation
of model performance. There have been plenty of works concentrated on learning
with in-distribution (IND) noisy labels in the last decade, i.e., some training
samples are assigned incorrect labels that do not correspond to their true
classes. Nonetheless, in real application scenarios, it is necessary to
consider the influence of out-of-distribution (OOD) samples, i.e., samples that
do not belong to any known classes, which has not been sufficiently explored
yet. To remedy this, we study a new problem setup, namely Learning with
Open-world Noisy Data (LOND). The goal of LOND is to simultaneously learn a
classifier and an OOD detector from datasets with mixed IND and OOD noise. In
this paper, we propose a new graph-based framework, namely Noisy Graph Cleaning
(NGC), which collects clean samples by leveraging geometric structure of data
and model predictive confidence. Without any additional training effort, NGC
can detect and reject the OOD samples based on the learned class prototypes
directly in testing phase. We conduct experiments on multiple benchmarks with
different types of noise and the results demonstrate the superior performance
of our method against state of the arts.

    

### [[2108.11053] Applying Semi-Automated Hyperparameter Tuning for Clustering Algorithms](http://arxiv.org/abs/2108.11053)


  When approaching a clustering problem, choosing the right clustering
algorithm and parameters is essential, as each clustering algorithm is
proficient at finding clusters of a particular nature. Due to the unsupervised
nature of clustering algorithms, there are no ground truth values available for
empirical evaluation, which makes automation of the parameter selection process
through hyperparameter tuning difficult. Previous approaches to hyperparameter
tuning for clustering algorithms have relied on internal metrics, which are
often biased towards certain algorithms, or having some ground truth labels
available, moving the problem into the semi-supervised space. This preliminary
study proposes a framework for semi-automated hyperparameter tuning of
clustering problems, using a grid search to develop a series of graphs and easy
to interpret metrics that can then be used for more efficient domain-specific
evaluation. Preliminary results show that internal metrics are unable to
capture the semantic quality of the clusters developed and approaches driven by
internal metrics would come to different conclusions than those driven by
manual evaluation.

    

### [[2108.11056] Social Norm Bias: Residual Harms of Fairness-Aware Algorithms](http://arxiv.org/abs/2108.11056)


  Many modern learning algorithms mitigate bias by enforcing fairness across
coarsely-defined groups related to a sensitive attribute like gender or race.
However, the same algorithms seldom account for the within-group biases that
arise due to the heterogeneity of group members. In this work, we characterize
Social Norm Bias (SNoB), a subtle but consequential type of discrimination that
may be exhibited by automated decision-making systems, even when these systems
achieve group fairness objectives. We study this issue through the lens of
gender bias in occupation classification from biographies. We quantify SNoB by
measuring how an algorithm's predictions are associated with conformity to
gender norms, which is measured using a machine learning approach. This
framework reveals that for classification tasks related to male-dominated
occupations, fairness-aware classifiers favor biographies written in ways that
align with masculine gender norms. We compare SNoB across fairness intervention
techniques and show that post-processing interventions do not mitigate this
type of bias at all.

    

### [[2108.11057] Opportunistic Emulation of Computationally Expensive Simulations via Deep Learning](http://arxiv.org/abs/2108.11057)


  With the underlying aim of increasing efficiency of computational modelling
pertinent for managing and protecting the Great Barrier Reef, we investigate
the use of deep neural networks for opportunistic model emulation of APSIM
models by repurposing an existing large dataset containing the outputs of APSIM
model runs. The dataset has not been specifically tailored for the model
emulation task. We employ two neural network architectures for the emulation
task: densely connected feed-forward neural network (FFNN), and gated recurrent
unit feeding into FFNN (GRU-FFNN), a type of a recurrent neural network.
Various configurations of the architectures are trialled. A minimum correlation
statistic is employed to identify clusters of APSIM scenarios that can be
aggregated to form training sets for model emulation. We focus on emulating
four important outputs of the APSIM model: runoff, soil_loss, DINrunoff,
Nleached. The GRU-FFNN architecture with three hidden layers and 128 units per
layer provides good emulation of runoff and DINrunoff. However, soil_loss and
Nleached were emulated relatively poorly under a wide range of the considered
architectures; the emulators failed to capture variability at higher values of
these two outputs. While the opportunistic data available from past modelling
activities provides a large and useful dataset for exploring APSIM emulation,
it may not be sufficiently rich enough for successful deep learning of more
complex model dynamics. Design of Computer Experiments may be required to
generate more informative data to emulate all output variables of interest. We
also suggest the use of synthetic meteorology settings to allow the model to be
fed a wide range of inputs. These need not all be representative of normal
conditions, but can provide a denser, more informative dataset from which
complex relationships between input and outputs can be learned.

    

### [[2108.11071] Decentralized optimization with non-identical sampling in presence of stragglers](http://arxiv.org/abs/2108.11071)


  We consider decentralized consensus optimization when workers sample data
from non-identical distributions and perform variable amounts of work due to
slow nodes known as stragglers. The problem of non-identical distributions and
the problem of variable amount of work have been previously studied separately.
In our work we analyze them together under a unified system model. We study the
convergence of the optimization algorithm when combining worker outputs under
two heuristic methods: (1) weighting equally, and (2) weighting by the amount
of work completed by each. We prove convergence of the two methods under
perfect consensus, assuming straggler statistics are independent and identical
across all workers for all iterations. Our numerical results show that under
approximate consensus the second method outperforms the first method for both
convex and non-convex objective functions. We make use of the theory on minimum
variance unbiased estimator (MVUE) to evaluate the existence of an optimal
method for combining worker outputs. While we conclude that neither of the two
heuristic methods are optimal, we also show that an optimal method does not
exist.

    

### [[2108.11072] Learning Class-level Prototypes for Few-shot Learning](http://arxiv.org/abs/2108.11072)


  Few-shot learning aims to recognize new categories using very few labeled
samples. Although few-shot learning has witnessed promising development in
recent years, most existing methods adopt an average operation to calculate
prototypes, thus limited by the outlier samples. In this work, we propose a
simple yet effective framework for few-shot classification, which can learn to
generate preferable prototypes from few support data, with the help of an
episodic prototype generator module. The generated prototype is meant to be
close to a certain \textit{\targetproto{}} and is less influenced by outlier
samples. Extensive experiments demonstrate the effectiveness of this module,
and our approach gets a significant raise over baseline models, and get a
competitive result compared to previous methods on \textit{mini}ImageNet,
\textit{tiered}ImageNet, and cross-domain (\textit{mini}ImageNet $\rightarrow$
CUB-200-2011) datasets.

    

### [[2108.11096] Learning From Long-Tailed Data With Noisy Labels](http://arxiv.org/abs/2108.11096)


  Class imbalance and noisy labels are the norm rather than the exception in
many large-scale classification datasets. Nevertheless, most works in machine
learning typically assume balanced and clean data. There have been some recent
attempts to tackle, on one side, the problem of learning from noisy labels and,
on the other side, learning from long-tailed data. Each group of methods make
simplifying assumptions about the other. Due to this separation, the proposed
solutions often underperform when both assumptions are violated. In this work,
we present a simple two-stage approach based on recent advances in
self-supervised learning to treat both challenges simultaneously. It consists
of, first, task-agnostic self-supervised pre-training, followed by
task-specific fine-tuning using an appropriate loss. Most significantly, we
find that self-supervised learning approaches are effectively able to cope with
severe class imbalance. In addition, the resulting learned representations are
also remarkably robust to label noise, when fine-tuned with an imbalance- and
noise-resistant loss function. We validate our claims with experiments on
CIFAR-10 and CIFAR-100 augmented with synthetic imbalance and noise, as well as
the large-scale inherently noisy Clothing-1M dataset.

    

### [[2108.11100] Multi-Attributed and Structured Text-to-Face Synthesis](http://arxiv.org/abs/2108.11100)


  Generative Adversarial Networks (GANs) have revolutionized image synthesis
through many applications like face generation, photograph editing, and image
super-resolution. Image synthesis using GANs has predominantly been uni-modal,
with few approaches that can synthesize images from text or other data modes.
Text-to-image synthesis, especially text-to-face synthesis, has promising use
cases of robust face-generation from eye witness accounts and augmentation of
the reading experience with visual cues. However, only a couple of datasets
provide consolidated face data and textual descriptions for text-to-face
synthesis. Moreover, these textual annotations are less extensive and
descriptive, which reduces the diversity of faces generated from it. This paper
empirically proves that increasing the number of facial attributes in each
textual description helps GANs generate more diverse and real-looking faces. To
prove this, we propose a new methodology that focuses on using structured
textual descriptions. We also consolidate a Multi-Attributed and Structured
Text-to-face (MAST) dataset consisting of high-quality images with structured
textual annotations and make it available to researchers to experiment and
build upon. Lastly, we report benchmark Frechet's Inception Distance (FID),
Facial Semantic Similarity (FSS), and Facial Semantic Distance (FSD) scores for
the MAST dataset.

    

### [[2108.11106] Dropout against Deep Leakage from Gradients](http://arxiv.org/abs/2108.11106)


  As the scale and size of the data increases significantly nowadays, federal
learning (Bonawitz et al. [2019]) for high performance computing and machine
learning has been much more important than ever beforeAbadi et al. [2016].
People used to believe that sharing gradients seems to be safe to conceal the
local training data during the training stage. However, Zhu et al. [2019]
demonstrated that it was possible to recover raw data from the model training
data by detecting gradients. They use generated random dummy data and minimise
the distance between them and real data. Zhao et al. [2020] pushes the
convergence algorithm even further. By replacing the original loss function
with cross entropy loss, they achieve better fidelity threshold. In this paper,
we propose using an additional dropout (Srivastava et al. [2014]) layer before
feeding the data to the classifier. It is very effective in preventing leakage
of raw data, as the training data cannot converge to a small RMSE even after
5,800 epochs with dropout rate set to 0.5.

    

### [[2108.11124] Inductive Matrix Completion Using Graph Autoencoder](http://arxiv.org/abs/2108.11124)


  Recently, the graph neural network (GNN) has shown great power in matrix
completion by formulating a rating matrix as a bipartite graph and then
predicting the link between the corresponding user and item nodes. The majority
of GNN-based matrix completion methods are based on Graph Autoencoder (GAE),
which considers the one-hot index as input, maps a user (or item) index to a
learnable embedding, applies a GNN to learn the node-specific representations
based on these learnable embeddings and finally aggregates the representations
of the target users and its corresponding item nodes to predict missing links.
However, without node content (i.e., side information) for training, the user
(or item) specific representation can not be learned in the inductive setting,
that is, a model trained on one group of users (or items) cannot adapt to new
users (or items). To this end, we propose an inductive matrix completion method
using GAE (IMC-GAE), which utilizes the GAE to learn both the user-specific (or
item-specific) representation for personalized recommendation and local graph
patterns for inductive matrix completion. Specifically, we design two
informative node features and employ a layer-wise node dropout scheme in GAE to
learn local graph patterns which can be generalized to unseen data. The main
contribution of our paper is the capability to efficiently learn local graph
patterns in GAE, with good scalability and superior expressiveness compared to
previous GNN-based matrix completion methods. Furthermore, extensive
experiments demonstrate that our model achieves state-of-the-art performance on
several matrix completion benchmarks. Our official code is publicly available.

    

### [[2108.11135] Bridged Adversarial Training](http://arxiv.org/abs/2108.11135)


  Adversarial robustness is considered as a required property of deep neural
networks. In this study, we discover that adversarially trained models might
have significantly different characteristics in terms of margin and smoothness,
even they show similar robustness. Inspired by the observation, we investigate
the effect of different regularizers and discover the negative effect of the
smoothness regularizer on maximizing the margin. Based on the analyses, we
propose a new method called bridged adversarial training that mitigates the
negative effect by bridging the gap between clean and adversarial examples. We
provide theoretical and empirical evidence that the proposed method provides
stable and better robustness, especially for large perturbations.

    

### [[2108.11139] Learning GraphQL Query Costs (Extended Version)](http://arxiv.org/abs/2108.11139)


  GraphQL is a query language for APIs and a runtime for executing those
queries, fetching the requested data from existing microservices, REST APIs,
databases, or other sources. Its expressiveness and its flexibility have made
it an attractive candidate for API providers in many industries, especially
through the web. A major drawback to blindly servicing a client's query in
GraphQL is that the cost of a query can be unexpectedly large, creating
computation and resource overload for the provider, and API rate-limit overages
and infrastructure overload for the client. To mitigate these drawbacks, it is
necessary to efficiently estimate the cost of a query before executing it.
Estimating query cost is challenging, because GraphQL queries have a nested
structure, GraphQL APIs follow different design conventions, and the underlying
data sources are hidden. Estimates based on worst-case static query analysis
have had limited success because they tend to grossly overestimate cost. We
propose a machine-learning approach to efficiently and accurately estimate the
query cost. We also demonstrate the power of this approach by testing it on
query-response data from publicly available commercial APIs. Our framework is
efficient and predicts query costs with high accuracy, consistently
outperforming the static analysis by a large margin.

    

### [[2108.11193] Models In a Spelling Bee: Language Models Implicitly Learn the Character Composition of Tokens](http://arxiv.org/abs/2108.11193)


  Standard pretrained language models operate on sequences of subword tokens
without direct access to the characters that compose each token's string
representation. We probe the embedding layer of pretrained language models and
show that models learn the internal character composition of whole word and
subword tokens to a surprising extent, without ever seeing the characters
coupled with the tokens. Our results show that the embedding layer of RoBERTa
holds enough information to accurately spell up to a third of the vocabulary
and reach high average character ngram overlap on all token types. We further
test whether enriching subword models with additional character information can
improve language modeling, and observe that this method has a near-identical
learning curve as training without spelling-based enrichment. Overall, our
results suggest that language modeling objectives incentivize the model to
implicitly learn some notion of spelling, and that explicitly teaching the
model how to spell does not enhance its performance on such tasks.

    

### [[2108.11195] Lizard: A Large-Scale Dataset for Colonic Nuclear Instance Segmentation and Classification](http://arxiv.org/abs/2108.11195)


  The development of deep segmentation models for computational pathology
(CPath) can help foster the investigation of interpretable morphological
biomarkers. Yet, there is a major bottleneck in the success of such approaches
because supervised deep learning models require an abundance of accurately
labelled data. This issue is exacerbated in the field of CPath because the
generation of detailed annotations usually demands the input of a pathologist
to be able to distinguish between different tissue constructs and nuclei.
Manually labelling nuclei may not be a feasible approach for collecting
large-scale annotated datasets, especially when a single image region can
contain thousands of different cells. However, solely relying on automatic
generation of annotations will limit the accuracy and reliability of ground
truth. Therefore, to help overcome the above challenges, we propose a
multi-stage annotation pipeline to enable the collection of large-scale
datasets for histology image analysis, with pathologist-in-the-loop refinement
steps. Using this pipeline, we generate the largest known nuclear instance
segmentation and classification dataset, containing nearly half a million
labelled nuclei in H&E stained colon tissue. We have released the dataset and
encourage the research community to utilise it to drive forward the development
of downstream cell-based models in CPath.

    

### [[2108.11204] Subgoal Search For Complex Reasoning Tasks](http://arxiv.org/abs/2108.11204)


  Humans excel in solving complex reasoning tasks through a mental process of
moving from one idea to a related one. Inspired by this, we propose Subgoal
Search (kSubS) method. Its key component is a learned subgoal generator that
produces a diversity of subgoals that are both achievable and closer to the
solution. Using subgoals reduces the search space and induces a high-level
search graph suitable for efficient planning. In this paper, we implement kSubS
using a transformer-based subgoal module coupled with the classical best-first
search framework. We show that a simple approach of generating $k$-th step
ahead subgoals is surprisingly efficient on three challenging domains: two
popular puzzle games, Sokoban and the Rubik's Cube, and an inequality proving
benchmark INT. kSubS achieves strong results including state-of-the-art on INT
within a modest computational budget.

    

### [[2108.11211] Clustering acoustic emission data streams with sequentially appearing clusters using mixture models](http://arxiv.org/abs/2108.11211)


  The interpretation of unlabeled acoustic emission (AE) data classically
relies on general-purpose clustering methods. While several external criteria
have been used in the past to select the hyperparameters of those algorithms,
few studies have paid attention to the development of dedicated objective
functions in clustering methods able to cope with the specificities of AE data.
We investigate how to explicitly represent clusters onsets in mixture models in
general, and in Gaussian Mixture Models (GMM) in particular. By modifying the
internal criterion of such models, we propose the first clustering method able
to provide, through parameters estimated by an expectation-maximization
procedure, information about when clusters occur (onsets), how they grow
(kinetics) and their level of activation through time. This new objective
function accommodates continuous timestamps of AE signals and, thus, their
order of occurrence. The method, called GMMSEQ, is experimentally validated to
characterize the loosening phenomenon in bolted structure under vibrations. A
comparison with three standard clustering methods on raw streaming data from
five experimental campaigns shows that GMMSEQ not only provides useful
qualitative information about the timeline of clusters, but also shows better
performance in terms of cluster characterization. In view of developing an open
acoustic emission initiative and according to the FAIR principles, the datasets
and the codes are made available to reproduce the research of this paper.

    

### [[2108.11220] Toward Formal Data Set Verification for Building Effective Machine Learning Models](http://arxiv.org/abs/2108.11220)


  In order to properly train a machine learning model, data must be properly
collected. To guarantee a proper data collection, verifying that the collected
data set holds certain properties is a possible solution. For example,
guaranteeing that the data set contains samples across the whole input space,
or that the data set is balanced w.r.t. different classes. We present a formal
approach for verifying a set of arbitrarily stated properties over a data set.
The proposed approach relies on the transformation of the data set into a first
order logic formula, which can be later verified w.r.t. the different
properties also stated in the same logic. A prototype tool, which uses the z3
solver, has been developed; the prototype can take as an input a set of
properties stated in a formal language and formally verify a given data set
w.r.t. to the given set of properties. Preliminary experimental results show
the feasibility and performance of the proposed approach, and furthermore the
flexibility for expressing properties of interest.

    

### [[2108.11249] Generalize then Adapt: Source-Free Domain Adaptive Semantic Segmentation](http://arxiv.org/abs/2108.11249)


  Unsupervised domain adaptation (DA) has gained substantial interest in
semantic segmentation. However, almost all prior arts assume concurrent access
to both labeled source and unlabeled target, making them unsuitable for
scenarios demanding source-free adaptation. In this work, we enable source-free
DA by partitioning the task into two: a) source-only domain generalization and
b) source-free target adaptation. Towards the former, we provide theoretical
insights to develop a multi-head framework trained with a virtually extended
multi-source dataset, aiming to balance generalization and specificity. Towards
the latter, we utilize the multi-head framework to extract reliable target
pseudo-labels for self-training. Additionally, we introduce a novel conditional
prior-enforcing auto-encoder that discourages spatial irregularities, thereby
enhancing the pseudo-label quality. Experiments on the standard
GTA5-to-Cityscapes and SYNTHIA-to-Cityscapes benchmarks show our superiority
even against the non-source-free prior-arts. Further, we show our compatibility
with online adaptation enabling deployment in a sequentially changing
environment.

    

### [[2108.11262] Deep few-shot learning for bi-temporal building change detection](http://arxiv.org/abs/2108.11262)


  In real-world applications (e.g., change detection), annotating images is
very expensive. To build effective deep learning models in these applications,
deep few-shot learning methods have been developed and prove to be a robust
approach in small training data. The analysis of building change detection from
high spatial resolution remote sensing observations is important research in
photogrammetry, computer vision, and remote sensing nowadays, which can be
widely used in a variety of real-world applications, such as map updating. As
manual high resolution image interpretation is expensive and time-consuming,
building change detection methods are of high interest. The interest in
developing building change detection approaches from optical remote sensing
images is rapidly increasing due to larger coverages, and lower costs of
optical images. In this study, we focus on building change detection analysis
on a small set of building change from different regions that sit in several
cities. In this paper, a new deep few-shot learning method is proposed for
building change detection using Monte Carlo dropout and remote sensing
observations. The setup is based on a small dataset, including bitemporal
optical images labeled for building change detection.

    

### [[2108.11283] Automatic Feature Highlighting in Noisy RES Data With CycleGAN](http://arxiv.org/abs/2108.11283)


  Radio echo sounding (RES) is a common technique used in subsurface glacial
imaging, which provides insight into the underlying rock and ice. However,
systematic noise is introduced into the data during collection, complicating
interpretation of the results. Researchers most often use a combination of
manual interpretation and filtering techniques to denoise data; however, these
processes are time intensive and inconsistent. Fully Convolutional Networks
have been proposed as an automated alternative to identify layer boundaries in
radargrams. However, they require high-quality manually processed training data
and struggle to interpolate data in noisy samples (Varshney et al. 2020).
Herein, the authors propose a GAN based model to interpolate layer boundaries
through noise and highlight layers in two-dimensional glacial RES data. In
real-world noisy images, filtering often results in loss of data such that
interpolating layer boundaries is nearly impossible. Furthermore, traditional
machine learning approaches are not suited to this task because of the lack of
paired data, so we employ an unpaired image-to-image translation model. For
this model, we create a synthetic dataset to represent the domain of images
with clear, highlighted layers and use an existing real-world RES dataset as
our noisy domain.
We implement a CycleGAN trained on these two domains to highlight layers in
noisy images that can interpolate effectively without significant loss of
structure or fidelity. Though the current implementation is not a perfect
solution, the model clearly highlights layers in noisy data and allows
researchers to determine layer size and position without mathematical
filtering, manual processing, or ground-truth images for training. This is
significant because clean images generated by our model enable subsurface
researchers to determine glacial layer thickness more efficiently.

    

### [[2108.11299] Backdoor Attacks on Network Certification via Data Poisoning](http://arxiv.org/abs/2108.11299)


  Certifiers for neural networks have made great progress towards provable
robustness guarantees against evasion attacks using adversarial examples.
However, introducing certifiers into deep learning systems also opens up new
attack vectors, which need to be considered before deployment. In this work, we
conduct the first systematic analysis of training time attacks against
certifiers in practical application pipelines, identifying new threat vectors
that can be exploited to degrade the overall system. Using these insights, we
design two backdoor attacks against network certifiers, which can drastically
reduce certified robustness when the backdoor is activated. For example, adding
1% poisoned data points during training is sufficient to reduce certified
robustness by up to 95 percentage points, effectively rendering the certifier
useless. We analyze how such novel attacks can compromise the overall system's
integrity or availability. Our extensive experiments across multiple datasets,
model architectures, and certifiers demonstrate the wide applicability of these
attacks. A first investigation into potential defenses shows that current
approaches only partially mitigate the issue, highlighting the need for new,
more specific solutions.

    

### [[2108.11305] CSG-Stump: A Learning Friendly CSG-Like Representation for Interpretable Shape Parsing](http://arxiv.org/abs/2108.11305)


  Generating an interpretable and compact representation of 3D shapes from
point clouds is an important and challenging problem. This paper presents
CSG-Stump Net, an unsupervised end-to-end network for learning shapes from
point clouds and discovering the underlying constituent modeling primitives and
operations as well. At the core is a three-level structure called {\em
CSG-Stump}, consisting of a complement layer at the bottom, an intersection
layer in the middle, and a union layer at the top. CSG-Stump is proven to be
equivalent to CSG in terms of representation, therefore inheriting the
interpretable, compact and editable nature of CSG while freeing from CSG's
complex tree structures. Particularly, the CSG-Stump has a simple and regular
structure, allowing neural networks to give outputs of a constant
dimensionality, which makes itself deep-learning friendly. Due to these
characteristics of CSG-Stump, CSG-Stump Net achieves superior results compared
to previous CSG-based methods and generates much more appealing shapes, as
confirmed by extensive experiments. Project page:
this https URL


### [[2108.11308] What do pre-trained code models know about code?](http://arxiv.org/abs/2108.11308)


  Pre-trained models of code built on the transformer architecture have
performed well on software engineering (SE) tasks such as predictive code
generation, code summarization, among others. However, whether the vector
representations from these pre-trained models comprehensively encode
characteristics of source code well enough to be applicable to a broad spectrum
of downstream tasks remains an open question.
One way to investigate this is with diagnostic tasks called probes. In this
paper, we construct four probing tasks (probing for surface-level, syntactic,
structural, and semantic information) for pre-trained code models. We show how
probes can be used to identify whether models are deficient in (understanding)
certain code properties, characterize different model layers, and get insight
into the model sample-efficiency.
We probe four models that vary in their expected knowledge of code
properties: BERT (pre-trained on English), CodeBERT and CodeBERTa (pre-trained
on source code, and natural language documentation), and GraphCodeBERT
(pre-trained on source code with dataflow). While GraphCodeBERT performs more
consistently overall, we find that BERT performs surprisingly well on some code
tasks, which calls for further investigation.

    

### [[2108.11318] Long-term, Short-term and Sudden Event: Trading Volume Movement Prediction with Graph-based Multi-view Modeling](http://arxiv.org/abs/2108.11318)


  Trading volume movement prediction is the key in a variety of financial
applications. Despite its importance, there is few research on this topic
because of its requirement for comprehensive understanding of information from
different sources. For instance, the relation between multiple stocks, recent
transaction data and suddenly released events are all essential for
understanding trading market. However, most of the previous methods only take
the fluctuation information of the past few weeks into consideration, thus
yielding poor performance. To handle this issue, we propose a graphbased
approach that can incorporate multi-view information, i.e., long-term stock
trend, short-term fluctuation and sudden events information jointly into a
temporal heterogeneous graph. Besides, our method is equipped with deep
canonical analysis to highlight the correlations between different perspectives
of fluctuation for better prediction. Experiment results show that our method
outperforms strong baselines by a large margin.

    

### [[2108.11320] The Effect of Noise Level on Causal Identification with Additive Noise Models](http://arxiv.org/abs/2108.11320)


  In recent years a lot of research has been conducted within the area of
causal inference and causal learning. Many methods have been developed to
identify the cause-effect pairs in models and have been successfully applied to
observational real-world data in order to determine the direction of causal
relationships. Many of these methods require simplifying assumptions, such as
absence of confounding, cycles, and selection bias. Yet in bivariate situations
causal discovery problems remain challenging. One class of such methods, that
also allows tackling the bivariate case, is based on Additive Noise Models
(ANMs). Unfortunately, one aspect of these methods has not received much
attention until now: what is the impact of different noise levels on the
ability of these methods to identify the direction of the causal relationship.
This work aims to bridge this gap with the help of an empirical study. For this
work, we considered bivariate cases, which is the most elementary form of a
causal discovery problem where one needs to decide whether X causes Y or Y
causes X, given joint distributions of two variables X, Y. Furthermore, two
specific methods have been selected, \textit{Regression with Subsequent
Independence Test} and \textit{Identification using Conditional Variances},
which have been tested with an exhaustive range of ANMs where the additive
noises' levels gradually change from 1% to 10000% of the causes' noise level
(the latter remains fixed). Additionally, the experiments in this work consider
several different types of distributions as well as linear and non-linear ANMs.
The results of the experiments show that these methods can fail to capture the
true causal direction for some levels of noise.

    

### [[2108.11328] Predicting Census Survey Response Rates via Interpretable Nonparametric Additive Models with Structured Interactions](http://arxiv.org/abs/2108.11328)


  Accurate and interpretable prediction of survey response rates is important
from an operational standpoint. The US Census Bureau's well-known ROAM
application uses principled statistical models trained on the US Census
Planning Database data to identify hard-to-survey areas. An earlier
crowdsourcing competition revealed that an ensemble of regression trees led to
the best performance in predicting survey response rates; however, the
corresponding models could not be adopted for the intended application due to
limited interpretability. In this paper, we present new interpretable
statistical methods to predict, with high accuracy, response rates in surveys.
We study sparse nonparametric additive models with pairwise interactions via
$\ell_0$-regularization, as well as hierarchically structured variants that
provide enhanced interpretability. Despite strong methodological underpinnings,
such models can be computationally challenging -- we present new scalable
algorithms for learning these models. We also establish novel non-asymptotic
error bounds for the proposed estimators. Experiments based on the US Census
Planning Database demonstrate that our methods lead to high-quality predictive
models that permit actionable interpretability for different segments of the
population. Interestingly, our methods provide significant gains in
interpretability without losing in predictive performance to state-of-the-art
black-box machine learning methods based on gradient boosting and feedforward
neural networks. Our code implementation in python is available at
this https URL.

    

### [[2108.11332] Self-optimizing adaptive optics control with Reinforcement Learning for high-contrast imaging](http://arxiv.org/abs/2108.11332)


  Current and future high-contrast imaging instruments require extreme adaptive
optics (XAO) systems to reach contrasts necessary to directly image exoplanets.
Telescope vibrations and the temporal error induced by the latency of the
control loop limit the performance of these systems. One way to reduce these
effects is to use predictive control. We describe how model-free Reinforcement
Learning can be used to optimize a Recurrent Neural Network controller for
closed-loop predictive control. First, we verify our proposed approach for
tip-tilt control in simulations and a lab setup. The results show that this
algorithm can effectively learn to mitigate vibrations and reduce the residuals
for power-law input turbulence as compared to an optimal gain integrator. We
also show that the controller can learn to minimize random vibrations without
requiring online updating of the control law. Next, we show in simulations that
our algorithm can also be applied to the control of a high-order deformable
mirror. We demonstrate that our controller can provide two orders of magnitude
improvement in contrast at small separations under stationary turbulence.
Furthermore, we show more than an order of magnitude improvement in contrast
for different wind velocities and directions without requiring online updating
of the control law.

    

### [[2108.11333] Lightweight Self-Attentive Sequential Recommendation](http://arxiv.org/abs/2108.11333)


  Modern deep neural networks (DNNs) have greatly facilitated the development
of sequential recommender systems by achieving state-of-the-art recommendation
performance on various sequential recommendation tasks. Given a sequence of
interacted items, existing DNN-based sequential recommenders commonly embed
each item into a unique vector to support subsequent computations of the user
interest. However, due to the potentially large number of items, the
over-parameterised item embedding matrix of a sequential recommender has become
a memory bottleneck for efficient deployment in resource-constrained
environments, e.g., smartphones and other edge devices. Furthermore, we observe
that the widely-used multi-head self-attention, though being effective in
modelling sequential dependencies among items, heavily relies on redundant
attention units to fully capture both global and local item-item transition
patterns within a sequence.
In this paper, we introduce a novel lightweight self-attentive network (LSAN)
for sequential recommendation. To aggressively compress the original embedding
matrix, LSAN leverages the notion of compositional embeddings, where each item
embedding is composed by merging a group of selected base embedding vectors
derived from substantially smaller embedding matrices. Meanwhile, to account
for the intrinsic dynamics of each item, we further propose a temporal
context-aware embedding composition scheme. Besides, we develop an innovative
twin-attention network that alleviates the redundancy of the traditional
multi-head self-attention while retaining full capacity for capturing long- and
short-term (i.e., global and local) item dependencies. Comprehensive
experiments demonstrate that LSAN significantly advances the accuracy and
memory efficiency of existing sequential recommenders.

    

### [[2108.11345] A Unifying Theory of Thompson Sampling for Continuous Risk-Averse Bandits](http://arxiv.org/abs/2108.11345)


  This paper unifies the design and simplifies the analysis of risk-averse
Thompson sampling algorithms for the multi-armed bandit problem for a generic
class of risk functionals \r{ho} that are continuous. Using the contraction
principle in the theory of large deviations, we prove novel concentration
bounds for these continuous risk functionals. In contrast to existing works in
which the bounds depend on the samples themselves, our bounds only depend on
the number of samples. This allows us to sidestep significant analytical
challenges and unify existing proofs of the regret bounds of existing Thompson
sampling-based algorithms. We show that a wide class of risk functionals as
well as "nice" functions of them satisfy the continuity condition. Using our
newly developed analytical toolkits, we analyse the algorithms $\rho$-MTS (for
multinomial distributions) and $\rho$-NPTS (for bounded distributions) and
prove that they admit asymptotically optimal regret bounds of risk-averse
algorithms under the mean-variance, CVaR, and other ubiquitous risk measures,
as well as a host of newly synthesized risk measures. Numerical simulations
show that our bounds are reasonably tight vis-à-vis algorithm-independent
lower bounds.

    

### [[2108.11346] Auxiliary Task Update Decomposition: The Good, The Bad and The Neutral](http://arxiv.org/abs/2108.11346)


  While deep learning has been very beneficial in data-rich settings, tasks
with smaller training set often resort to pre-training or multitask learning to
leverage data from other tasks. In this case, careful consideration is needed
to select tasks and model parameterizations such that updates from the
auxiliary tasks actually help the primary task. We seek to alleviate this
burden by formulating a model-agnostic framework that performs fine-grained
manipulation of the auxiliary task gradients. We propose to decompose auxiliary
updates into directions which help, damage or leave the primary task loss
unchanged. This allows weighting the update directions differently depending on
their impact on the problem of interest. We present a novel and efficient
algorithm for that purpose and show its advantage in practice. Our method
leverages efficient automatic differentiation procedures and randomized
singular value decomposition for scalability. We show that our framework is
generic and encompasses some prior work as particular cases. Our approach
consistently outperforms strong and widely used baselines when leveraging
out-of-distribution data for Text and Image classification tasks.

    

### [[2108.11368] CDCGen: Cross-Domain Conditional Generation via Normalizing Flows and Adversarial Training](http://arxiv.org/abs/2108.11368)


  How to generate conditional synthetic data for a domain without utilizing
information about its labels/attributes? Our work presents a solution to the
above question. We propose a transfer learning-based framework utilizing
normalizing flows, coupled with both maximum-likelihood and adversarial
training. We model a source domain (labels available) and a target domain
(labels unavailable) with individual normalizing flows, and perform domain
alignment to a common latent space using adversarial discriminators. Due to the
invertible property of flow models, the mapping has exact cycle consistency. We
also learn the joint distribution of the data samples and attributes in the
source domain by employing an encoder to map attributes to the latent space via
adversarial training. During the synthesis phase, given any combination of
attributes, our method can generate synthetic samples conditioned on them in
the target domain. Empirical studies confirm the effectiveness of our method on
benchmarked datasets. We envision our method to be particularly useful for
synthetic data generation in label-scarce systems by generating non-trivial
augmentations via attribute transformations. These synthetic samples will
introduce more entropy into the label-scarce domain than their geometric and
photometric transformation counterparts, helpful for robust downstream tasks.

    

### [[2108.11371] Understanding the Generalization of Adam in Learning Neural Networks with Proper Regularization](http://arxiv.org/abs/2108.11371)


  Adaptive gradient methods such as Adam have gained increasing popularity in
deep learning optimization. However, it has been observed that compared with
(stochastic) gradient descent, Adam can converge to a different solution with a
significantly worse test error in many deep learning applications such as image
classification, even with a fine-tuned regularization. In this paper, we
provide a theoretical explanation for this phenomenon: we show that in the
nonconvex setting of learning over-parameterized two-layer convolutional neural
networks starting from the same random initialization, for a class of data
distributions (inspired from image data), Adam and gradient descent (GD) can
converge to different global solutions of the training objective with provably
different generalization errors, even with weight decay regularization. In
contrast, we show that if the training objective is convex, and the weight
decay regularization is employed, any optimization algorithms including Adam
and GD will converge to the same solution if the training is successful. This
suggests that the inferior generalization performance of Adam is fundamentally
tied to the nonconvex landscape of deep learning optimization.

    

### [[2002.06873] $π$VAE: Encoding stochastic process priors with variational autoencoders](http://arxiv.org/abs/2002.06873)


  Stochastic processes provide a mathematically elegant way model complex data.
In theory, they provide flexible priors over function classes that can encode a
wide range of interesting assumptions. In practice, however, efficient
inference by optimisation or marginalisation is difficult, a problem further
exacerbated with big data and high dimensional input spaces. We propose a novel
variational autoencoder (VAE) called the prior encoding variational autoencoder
($\pi$VAE). The $\pi$VAE is finitely exchangeable and Kolmogorov consistent,
and thus is a continuous stochastic process. We use $\pi$VAE to learn low
dimensional embeddings of function classes. We show that our framework can
accurately learn expressive function classes such as Gaussian processes, but
also properties of functions to enable statistical inference (such as the
integral of a log Gaussian process). For popular tasks, such as spatial
interpolation, $\pi$VAE achieves state-of-the-art performance both in terms of
accuracy and computational efficiency. Perhaps most usefully, we demonstrate
that the low dimensional independently distributed latent space representation
learnt provides an elegant and scalable means of performing Bayesian inference
for stochastic processes within probabilistic programming languages such as
Stan.

    

### [[2002.12463] Certified Defense to Image Transformations via Randomized Smoothing](http://arxiv.org/abs/2002.12463)


  We extend randomized smoothing to cover parameterized transformations (e.g.,
rotations, translations) and certify robustness in the parameter space (e.g.,
rotation angle). This is particularly challenging as interpolation and rounding
effects mean that image transformations do not compose, in turn preventing
direct certification of the perturbed image (unlike certification with $\ell^p$
norms). We address this challenge by introducing three different kinds of
defenses, each with a different guarantee (heuristic, distributional and
individual) stemming from the method used to bound the interpolation error.
Importantly, we show how individual certificates can be obtained via either
statistical error bounds or efficient online inverse computation of the image
transformation. We provide an implementation of all methods at
this https URL.

    

### [[2005.02480] A Ladder of Causal Distances](http://arxiv.org/abs/2005.02480)


  Causal discovery, the task of automatically constructing a causal model from
data, is of major significance across the sciences. Evaluating the performance
of causal discovery algorithms should ideally involve comparing the inferred
models to ground-truth models available for benchmark datasets, which in turn
requires a notion of distance between causal models. While such distances have
been proposed previously, they are limited by focusing on graphical properties
of the causal models being compared. Here, we overcome this limitation by
defining distances derived from the causal distributions induced by the models,
rather than exclusively from their graphical structure. Pearl and Mackenzie
(2018) have arranged the properties of causal models in a hierarchy called the
"ladder of causation" spanning three rungs: observational, interventional, and
counterfactual. Following this organization, we introduce a hierarchy of three
distances, one for each rung of the ladder. Our definitions are intuitively
appealing as well as efficient to compute approximately. We put our causal
distances to use by benchmarking standard causal discovery systems on both
synthetic and real-world datasets for which ground-truth causal models are
available. Finally, we highlight the usefulness of our causal distances by
briefly discussing further applications beyond the evaluation of causal
discovery techniques.

    

### [[2005.05195] Solving Large-Scale Sparse PCA to Certifiable (Near) Optimality](http://arxiv.org/abs/2005.05195)


  Sparse principal component analysis (PCA) is a popular dimensionality
reduction technique for obtaining principal components which are linear
combinations of a small subset of the original features. Existing approaches
cannot supply certifiably optimal principal components with more than $p=100s$
of variables. By reformulating sparse PCA as a convex mixed-integer
semidefinite optimization problem, we design a cutting-plane method which
solves the problem to certifiable optimality at the scale of selecting k=5
covariates from p=300 variables, and provides small bound gaps at a larger
scale. We also propose a convex relaxation and greedy rounding scheme that
provides bound gaps of $1-2\%$ in practice within minutes for $p=100$s or hours
for $p=1,000$s and is therefore a viable alternative to the exact method at
scale. Using real-world financial and medical datasets, we illustrate our
approach's ability to derive interpretable principal components tractably at
scale.

    

### [[2006.04511] Classifying histograms of medical data using information geometry of beta distributions](http://arxiv.org/abs/2006.04511)


  In this paper, we use tools of information geometry to compare, average and
classify histograms. Beta distributions are fitted to the histograms and the
corresponding Fisher information geometry is used for comparison. We show that
this geometry is negatively curved, which guarantees uniqueness of the notion
of mean, and makes it suitable to classify histograms through the popular
K-means algorithm. We illustrate the use of these geometric tools in supervised
and unsupervised classification procedures of two medical data-sets, cardiac
shape deformations for the detection of pulmonary hypertension and brain
cortical thickness for the diagnosis of Alzheimer's disease.

    

### [[2006.10160] Matérn Gaussian processes on Riemannian manifolds](http://arxiv.org/abs/2006.10160)


  Gaussian processes are an effective model class for learning unknown
functions, particularly in settings where accurately representing predictive
uncertainty is of key importance. Motivated by applications in the physical
sciences, the widely-used Matérn class of Gaussian processes has recently
been generalized to model functions whose domains are Riemannian manifolds, by
re-expressing said processes as solutions of stochastic partial differential
equations. In this work, we propose techniques for computing the kernels of
these processes on compact Riemannian manifolds via spectral theory of the
Laplace-Beltrami operator in a fully constructive manner, thereby allowing them
to be trained via standard scalable techniques such as inducing point methods.
We also extend the generalization from the Matérn to the widely-used squared
exponential Gaussian process. By allowing Riemannian Matérn Gaussian
processes to be trained using well-understood techniques, our work enables
their use in mini-batch, online, and non-conjugate settings, and makes them
more accessible to machine learning practitioners.

    

### [[2007.02394] Meta-Semi: A Meta-learning Approach for Semi-supervised Learning](http://arxiv.org/abs/2007.02394)


  Deep learning based semi-supervised learning (SSL) algorithms have led to
promising results in recent years. However, they tend to introduce multiple
tunable hyper-parameters, making them less practical in real SSL scenarios
where the labeled data is scarce for extensive hyper-parameter search. In this
paper, we propose a novel meta-learning based SSL algorithm (Meta-Semi) that
requires tuning only one additional hyper-parameter, compared with a standard
supervised deep learning algorithm, to achieve competitive performance under
various conditions of SSL. We start by defining a meta optimization problem
that minimizes the loss on labeled data through dynamically reweighting the
loss on unlabeled samples, which are associated with soft pseudo labels during
training. As the meta problem is computationally intensive to solve directly,
we propose an efficient algorithm to dynamically obtain the approximate
solutions. We show theoretically that Meta-Semi converges to the stationary
point of the loss function on labeled data under mild conditions. Empirically,
Meta-Semi outperforms state-of-the-art SSL algorithms significantly on the
challenging semi-supervised CIFAR-100 and STL-10 tasks, and achieves
competitive performance on CIFAR-10 and SVHN.

    

### [[2007.06662] Predicting Sequences of Traversed Nodes in Graphs using Network Models with Multiple Higher Orders](http://arxiv.org/abs/2007.06662)


  We propose a novel sequence prediction method for sequential data capturing
node traversals in graphs. Our method builds on a statistical modelling
framework that combines multiple higher-order network models into a single
multi-order model. We develop a technique to fit such multi-order models in
empirical sequential data and to select the optimal maximum order. Our
framework facilitates both next-element and full sequence prediction given a
sequence-prefix of any length. We evaluate our model based on six empirical
data sets containing sequences from website navigation as well as public
transport systems. The results show that our method out-performs
state-of-the-art algorithms for next-element prediction. We further demonstrate
the accuracy of our method during out-of-sample sequence prediction and
validate that our method can scale to data sets with millions of sequences.

    

### [[2007.09647] Adversarial Immunization for Certifiable Robustness on Graphs](http://arxiv.org/abs/2007.09647)


  Despite achieving strong performance in semi-supervised node classification
task, graph neural networks (GNNs) are vulnerable to adversarial attacks,
similar to other deep learning models. Existing researches focus on developing
either robust GNN models or attack detection methods against adversarial
attacks on graphs. However, little research attention is paid to the potential
and practice of immunization to adversarial attacks on graphs. In this paper,
we propose and formulate the graph adversarial immunization problem, i.e.,
vaccinating an affordable fraction of node pairs, connected or unconnected, to
improve the certifiable robustness of graph against any admissible adversarial
attack. We further propose an effective algorithm, called AdvImmune, which
optimizes with meta-gradient in a discrete way to circumvent the
computationally expensive combinatorial optimization when solving the
adversarial immunization problem. Experiments are conducted on two citation
networks and one social network. Experimental results demonstrate that the
proposed AdvImmune method remarkably improves the ratio of robust nodes by 12%,
42%, 65%, with an affordable immune budget of only 5% edges.

    

### [[2008.13443] On the Quality Requirements of Demand Prediction for Dynamic Public Transport](http://arxiv.org/abs/2008.13443)


  As Public Transport (PT) becomes more dynamic and demand-responsive, it
increasingly depends on predictions of transport demand. But how accurate need
such predictions be for effective PT operation? We address this question
through an experimental case study of PT trips in Metropolitan Copenhagen,
Denmark, which we conduct independently of any specific prediction models.
First, we simulate errors in demand prediction through unbiased noise
distributions that vary considerably in shape. Using the noisy predictions, we
then simulate and optimize demand-responsive PT fleets via a linear programming
formulation and measure their performance. Our results suggest that the
optimized performance is mainly affected by the skew of the noise distribution
and the presence of infrequently large prediction errors. In particular, the
optimized performance can improve under non-Gaussian vs. Gaussian noise. We
also find that dynamic routing could reduce trip time by at least 23% vs.
static routing. This reduction is estimated at 809,000 EUR/year in terms of
Value of Travel Time Savings for the case study.

    

### [[2010.03060] Contrastive Cross-Modal Pre-Training: A General Strategy for Small Sample Medical Imaging](http://arxiv.org/abs/2010.03060)


  A key challenge in training neural networks for a given medical imaging task
is often the difficulty of obtaining a sufficient number of manually labeled
examples. In contrast, textual imaging reports, which are often readily
available in medical records, contain rich but unstructured interpretations
written by experts as part of standard clinical practice. We propose using
these textual reports as a form of weak supervision to improve the image
interpretation performance of a neural network without requiring additional
manually labeled examples. We use an image-text matching task to train a
feature extractor and then fine-tune it in a transfer learning setting for a
supervised task using a small labeled dataset. The end result is a neural
network that automatically interprets imagery without requiring textual reports
during inference. This approach can be applied to any task for which text-image
pairs are readily available. We evaluate our method on three classification
tasks and find consistent performance improvements, reducing the need for
labeled data by 67%-98%.

    

### [[2101.11712] Predicting the Mechanical Properties of Biopolymer Gels Using Neural Networks Trained on Discrete Fiber Network Data](http://arxiv.org/abs/2101.11712)


  Biopolymer gels, such as those made out of fibrin or collagen, are widely
used in tissue engineering applications and biomedical research. Moreover,
fibrin naturally assembles into gels in vivo during wound healing and thrombus
formation. Macroscale biopolymer gel mechanics are dictated by the microscale
fiber network. Hence, accurate description of biopolymer gels can be achieved
using representative volume elements (RVE) that explicitly model the discrete
fiber networks of the microscale. These RVE models, however, cannot be
efficiently used to model the macroscale due to the challenges and
computational demands of multiscale coupling. Here, we propose the use of an
artificial, fully connected neural network (FCNN) to efficiently capture the
behavior of the RVE models. The FCNN was trained on 1100 fiber networks
subjected to 121 biaxial deformations. The stress data from the RVE, together
with the total energy and the condition of incompressibility of the surrounding
matrix, were used to determine the derivatives of an unknown strain energy
function with respect to the deformation invariants. During training, the loss
function was modified to ensure convexity of the strain energy function and
symmetry of its Hessian. A general FCNN model was coded into a user material
subroutine (UMAT) in the software Abaqus. In this work, the FCNN trained on the
discrete fiber network data was used in finite element simulations of fibrin
gels using our UMAT. We anticipate that this work will enable further
integration of machine learning tools with computational mechanics. It will
also improve computational modeling of biological materials characterized by a
multiscale structure.

    

### [[2102.00193] Coupling innovation method and feasibility analysis of garbage classification](http://arxiv.org/abs/2102.00193)


  In order to solve the recent defect in garbage classification - including low
level of intelligence, low accuracy and high cost of equipment, this paper
presents a series of methods in identification and judgment in intelligent
garbage classification, including a material identification based on thermal
principle and non-destructive laser irradiation, another material
identification based on optical diffraction and phase analysis, a profile
identification which utilizes a scenery thermal image after PCA and histogram
correction, another profile identification which utilizes computer vision with
innovated data sets and algorithms. Combining AHP and Bayesian formula, the
paper innovates a coupling algorithm which helps to make a comprehensive
judgment of the garbage sort, based on the material and profile identification.
This paper also proposes a method for real-time space measurement of garbage
cans, which based on the characteristics of air as fluid, and analyses the
functions of air cleaning and particle disposing. Instead of the single use of
garbage image recognition, this paper provides a comprehensive method to judge
the garbage sort by material and profile identifications, which greatly
enhancing the accuracy and intelligence in garbage classification.

    

### [[2102.10527] Delayed Rewards Calibration via Reward Empirical Sufficiency](http://arxiv.org/abs/2102.10527)


  Appropriate credit assignment for delay rewards is a fundamental challenge
for reinforcement learning. To tackle this problem, we introduce a delay reward
calibration paradigm inspired from a classification perspective. We hypothesize
that well-represented state vectors share similarities with each other since
they contain the same or equivalent essential information. To this end, we
define an empirical sufficient distribution, where the state vectors within the
distribution will lead agents to environmental reward signals in the consequent
steps. Therefore, a purify-trained classifier is designed to obtain the
distribution and generate the calibrated rewards. We examine the correctness of
sufficient state extraction by tracking the real-time extraction and building
different reward functions in environments. The results demonstrate that the
classifier could generate timely and accurate calibrated rewards. Moreover, the
rewards are able to make the model training process more efficient. Finally, we
identify and discuss that the sufficient states extracted by our model resonate
with the observations of humans.

    

### [[2103.00372] Neural Network Approach to Construction of Classical Integrable Systems](http://arxiv.org/abs/2103.00372)


  Integrable systems have provided various insights into physical phenomena and
mathematics. The way of constructing many-body integrable systems is limited to
few ansatzes for the Lax pair, except for highly inventive findings of
conserved quantities. Machine learning techniques have recently been applied to
broad physics fields and proven powerful for building non-trivial
transformations and potential functions. We here propose a machine learning
approach to a systematic construction of classical integrable systems. Given
the Hamiltonian or samples in latent space, our neural network simultaneously
learns the corresponding natural Hamiltonian in real space and the canonical
transformation between the latent space and the real space variables. We also
propose a loss function for building integrable systems and demonstrate
successful unsupervised learning for the Toda lattice. Our approach enables
exploring new integrable systems without any prior knowledge about the
canonical transformation or any ansatz for the Lax pair.

    

### [[2103.09414] Toward Neural-Network-Guided Program Synthesis and Verification](http://arxiv.org/abs/2103.09414)


  We propose a novel framework of program and invariant synthesis called neural
network-guided synthesis. We first show that, by suitably designing and
training neural networks, we can extract logical formulas over integers from
the weights and biases of the trained neural networks. Based on the idea, we
have implemented a tool to synthesize formulas from positive/negative examples
and implication constraints, and obtained promising experimental results. We
also discuss two applications of our synthesis method. One is the use of our
tool for qualifier discovery in the framework of ICE-learning-based CHC
solving, which can in turn be applied to program verification and inductive
invariant synthesis. Another application is to a new program development
framework called oracle-based programming, which is a neural-network-guided
variation of Solar-Lezama's program synthesis by sketching.

    

### [[2103.10481] Semi-Decentralized Federated Learning with Cooperative D2D Local Model Aggregations](http://arxiv.org/abs/2103.10481)


  Federated learning has emerged as a popular technique for distributing
machine learning (ML) model training across the wireless edge. In this paper,
we propose two timescale hybrid federated learning (TT-HF), a
semi-decentralized learning architecture that combines the conventional
device-to-server communication paradigm for federated learning with
device-to-device (D2D) communications for model training. In TT-HF, during each
global aggregation interval, devices (i) perform multiple stochastic gradient
descent iterations on their individual datasets, and (ii) aperiodically engage
in consensus procedure of their model parameters through cooperative,
distributed D2D communications within local clusters. With a new general
definition of gradient diversity, we formally study the convergence behavior of
TT-HF, resulting in new convergence bounds for distributed ML. We leverage our
convergence bounds to develop an adaptive control algorithm that tunes the step
size, D2D communication rounds, and global aggregation period of TT-HF over
time to target a sublinear convergence rate of O(1/t) while minimizing network
resource utilization. Our subsequent experiments demonstrate that TT-HF
significantly outperforms the current art in federated learning in terms of
model accuracy and/or network energy consumption in different scenarios where
local device datasets exhibit statistical heterogeneity. Finally, our numerical
evaluations demonstrate robustness against outages caused by fading channels,
as well favorable performance with non-convex loss functions.

    

### [[2103.11333] ANITA: An Optimal Loopless Accelerated Variance-Reduced Gradient Method](http://arxiv.org/abs/2103.11333)


  In this paper, we propose a novel accelerated gradient method called ANITA
for solving the fundamental finite-sum optimization problems. Concretely, we
consider both general convex and strongly convex settings: i) For general
convex finite-sum problems, ANITA improves previous state-of-the-art result
given by Varag (Lan et al., 2019). In particular, for large-scale problems or
the target error is not very small, i.e., $n \geq \frac{1}{\epsilon^2}$, ANITA
obtains the \emph{first} optimal result $O(n)$, matching the lower bound
$\Omega(n)$ provided by Woodworth and Srebro (2016), while previous results are
$O(n \log \frac{1}{\epsilon})$ of Varag (Lan et al., 2019) and
$O(\frac{n}{\sqrt{\epsilon}})$ of Katyusha (Allen-Zhu, 2017). ii) For strongly
convex finite-sum problems, we also show that ANITA can achieve the optimal
convergence rate $O\big((n+\sqrt{\frac{nL}{\mu}})\log\frac{1}{\epsilon}\big)$
matching the lower bound
$\Omega\big((n+\sqrt{\frac{nL}{\mu}})\log\frac{1}{\epsilon}\big)$ provided by
Lan and Zhou (2015). Besides, ANITA enjoys a simpler loopless algorithmic
structure unlike previous accelerated algorithms such as Varag (Lan et al.,
2019) and Katyusha (Allen-Zhu, 2017) where they use an inconvenient double-loop
structure. Moreover, by exploiting the loopless structure of ANITA, we provide
a new \emph{dynamic multi-stage convergence analysis}, which is the key
technical part for improving previous results to the optimal rates. Finally,
the numerical experiments show that ANITA converges faster than the previous
state-of-the-art Varag (Lan et al., 2019), validating our theoretical results
and confirming the practical superiority of ANITA. We believe that our new
theoretical rates and convergence analysis for this fundamental finite-sum
problem will directly lead to key improvements for many other related problems,
such as distributed/federated/decentralized optimization problems.

    

### [[2104.01101] Fast and Accurate Randomized Algorithms for Low-rank Tensor Decompositions](http://arxiv.org/abs/2104.01101)


  Low-rank Tucker and CP tensor decompositions are powerful tools in data
analytics. The widely used alternating least squares (ALS) method, which solves
a sequence of over-determined least squares subproblems, is costly for large
and sparse tensors. We propose a fast and accurate sketched ALS algorithm for
Tucker decomposition, which solves a sequence of sketched rank-constrained
linear least squares subproblems. Theoretical sketch size upper bounds are
provided to achieve $O(\epsilon)$ relative error for each subproblem with two
sketching techniques, TensorSketch and leverage score sampling. Experimental
results show that this new ALS algorithm, combined with a new initialization
scheme based on randomized range finder, yields up to $22.0\%$ relative
decomposition residual improvement compared to the state-of-the-art sketched
randomized algorithm for Tucker decomposition of various synthetic and real
datasets. This Tucker-ALS algorithm is further used to accelerate CP
decomposition, by using randomized Tucker compression followed by CP
decomposition of the Tucker core tensor. Experimental results show that this
algorithm not only converges faster, but also yields more accurate CP
decompositions.

    

### [[2104.01725] Graph Generative Models for Fast Detector Simulations in High Energy Physics](http://arxiv.org/abs/2104.01725)


  Accurate and fast simulation of particle physics processes is crucial for the
high-energy physics community. Simulating particle interactions with detectors
is both time consuming and computationally expensive. With the proton-proton
collision energy of 13 TeV, the Large Hadron Collider is uniquely positioned to
detect and measure the rare phenomena that can shape our knowledge of new
interactions. The High-Luminosity Large Hadron Collider (HL-LHC) upgrade will
put a significant strain on the computing infrastructure due to increased event
rate and levels of pile-up. Simulation of high-energy physics collisions needs
to be significantly faster without sacrificing the physics accuracy. Machine
learning approaches can offer faster solutions, while maintaining a high level
of fidelity. We discuss a graph generative model that provides effective
reconstruction of LHC events, paving the way for full detector level fast
simulation for HL-LHC.

    

### [[2104.02095] Analytic function approximation by path norm regularized deep networks](http://arxiv.org/abs/2104.02095)


  We show that neural networks with absolute value activation function and with
the path norm, the depth, the width and the network weights having logarithmic
dependence on $1/\varepsilon$ can $\varepsilon$-approximate functions that are
analytic on certain regions of $\mathbb{C}^d$.

    

### [[2104.02410] Using Voice and Biofeedback to Predict User Engagement during Requirements Interviews](http://arxiv.org/abs/2104.02410)


  Capturing users' engagement is crucial for gathering feedback about the
features of a software product. In a market-driven context, current approaches
to collect and analyze users' feedback are based on techniques leveraging
information extracted from product reviews and social media. These approaches
are hardly applicable in bespoke software development, or in contexts in which
one needs to gather information from specific users. In such cases, companies
need to resort to face-to-face interviews to get feedback on their products. In
this paper, we propose to utilize biometric data, in terms of physiological and
voice features, to complement interviews with information about the engagement
of the user on the discussed product-relevant topics. We evaluate our approach
by interviewing users while gathering their physiological data (i.e.,
biofeedback) using an Empatica E4 wristband, and capturing their voice through
the default audio-recorder of a common laptop. Our results show that we can
predict users' engagement by training supervised machine learning algorithms on
biometric data (F1=0.72), and that voice features alone are sufficiently
effective (F1=0.71). Our work contributes with one the first studies in
requirements engineering in which biometrics are used to identify emotions.
This is also the first study in software engineering that considers voice
analysis. The usage of voice features could be particularly helpful for
emotion-aware requirements elicitation in remote communication, either
performed by human analysts or voice-based chatbots, and can also be exploited
to support the analysis of meetings in software engineering research.

    

### [[2104.03528] Neural Temporal Point Processes: A Review](http://arxiv.org/abs/2104.03528)


  Temporal point processes (TPP) are probabilistic generative models for
continuous-time event sequences. Neural TPPs combine the fundamental ideas from
point process literature with deep learning approaches, thus enabling
construction of flexible and efficient models. The topic of neural TPPs has
attracted significant attention in the recent years, leading to the development
of numerous new architectures and applications for this class of models. In
this review paper we aim to consolidate the existing body of knowledge on
neural TPPs. Specifically, we focus on important design choices and general
principles for defining neural TPP models. Next, we provide an overview of
application areas commonly considered in the literature. We conclude this
survey with the list of open challenges and important directions for future
work in the field of neural TPPs.

    

### [[2104.12669] Exploiting Explanations for Model Inversion Attacks](http://arxiv.org/abs/2104.12669)


  The successful deployment of artificial intelligence (AI) in many domains
from healthcare to hiring requires their responsible use, particularly in model
explanations and privacy. Explainable artificial intelligence (XAI) provides
more information to help users to understand model decisions, yet this
additional knowledge exposes additional risks for privacy attacks. Hence,
providing explanation harms privacy. We study this risk for image-based model
inversion attacks and identified several attack architectures with increasing
performance to reconstruct private image data from model explanations. We have
developed several multi-modal transposed CNN architectures that achieve
significantly higher inversion performance than using the target model
prediction only. These XAI-aware inversion models were designed to exploit the
spatial knowledge in image explanations. To understand which explanations have
higher privacy risk, we analyzed how various explanation types and factors
influence inversion performance. In spite of some models not providing
explanations, we further demonstrate increased inversion performance even for
non-explainable target models by exploiting explanations of surrogate models
through attention transfer. This method first inverts an explanation from the
target prediction, then reconstructs the target image. These threats highlight
the urgent and significant privacy risks of explanations and calls attention
for new privacy preservation techniques that balance the dual-requirement for
AI explainability and privacy.

    

### [[2105.00233] Matrix completion based on Gaussian parameterized belief propagation](http://arxiv.org/abs/2105.00233)


  We develop a message-passing algorithm for noisy matrix completion problems
based on matrix factorization. The algorithm is derived by approximating
message distributions of belief propagation with Gaussian distributions that
share the same first and second moments. We also derive a memory-friendly
version of the proposed algorithm by applying a perturbation treatment commonly
used in the literature of approximate message passing. In addition, a damping
technique, which is demonstrated to be crucial for optimal performance, is
introduced without computational strain, and the relationship to the
message-passing version of alternating least squares, a method reported to be
optimal in certain settings, is discussed. Experiments on synthetic datasets
show that while the proposed algorithm quantitatively exhibits almost the same
performance under settings where the earlier algorithm is optimal, it is
advantageous when the observed datasets are corrupted by non-Gaussian noise.
Experiments on real-world datasets also emphasize the performance differences
between the two algorithms.

    

### [[2105.00470] On Feature Decorrelation in Self-Supervised Learning](http://arxiv.org/abs/2105.00470)


  In self-supervised representation learning, a common idea behind most of the
state-of-the-art approaches is to enforce the robustness of the representations
to predefined augmentations. A potential issue of this idea is the existence of
completely collapsed solutions (i.e., constant features), which are typically
avoided implicitly by carefully chosen implementation details. In this work, we
study a relatively concise framework containing the most common components from
recent approaches. We verify the existence of complete collapse and discover
another reachable collapse pattern that is usually overlooked, namely
dimensional collapse. We connect dimensional collapse with strong correlations
between axes and consider such connection as a strong motivation for feature
decorrelation (i.e., standardizing the covariance matrix). The gains from
feature decorrelation are verified empirically to highlight the importance and
the potential of this insight.

    

### [[2105.02726] SparseConvMIL: Sparse Convolutional Context-Aware Multiple Instance Learning for Whole Slide Image Classification](http://arxiv.org/abs/2105.02726)


  Multiple instance learning (MIL) is the preferred approach for whole slide
image classification. However, most MIL approaches do not exploit the
interdependencies of tiles extracted from a whole slide image, which could
provide valuable cues for classification. This paper presents a novel MIL
approach that exploits the spatial relationship of tiles for classifying whole
slide images. To do so, a sparse map is built from tiles embeddings, and is
then classified by a sparse-input CNN. It obtained state-of-the-art performance
over popular MIL approaches on the classification of cancer subtype involving
10000 whole slide images. Our results suggest that the proposed approach might
(i) improve the representation learning of instances and (ii) exploit the
context of instance embeddings to enhance the classification performance. The
code of this work is open-source at {github censored for review}.

    

### [[2105.05720] Breaking the Computation and Communication Abstraction Barrier in Distributed Machine Learning Workloads](http://arxiv.org/abs/2105.05720)


  Recent trend towards increasing large machine learning models require both
training and inference tasks to be distributed. Considering the huge cost of
training these models, it is imperative to unlock optimizations in computation
and communication to obtain best performance. However, current logical
separation between computation and communication kernels in deep learning
frameworks misses the optimization opportunities across such barrier. Breaking
this abstraction with a holistic consideration can provide many optimizations
to provide performance improvements in distributed workloads. Manually applying
these optimizations needs modifications in underlying computation and
communication libraries for each scenario, which is time consuming and
error-prone.
Therefore, we present CoCoNeT, with a DSL to express a program with both
computation and communication. CoCoNeT contains several machine learning aware
transformations to optimize a program and a compiler to generate high
performance kernels. Providing both computation and communication as first
class constructs allows users to work on a high-level abstraction and apply
powerful optimizations, such as fusion or overlapping of communication and
computation. CoCoNeT enables us to optimize data-, model-and pipeline-parallel
workloads in large language models with only a few lines of code. Experiments
show CoCoNeT significantly outperforms state-of-the-art distributed machine
learning implementations.

    

### [[2105.11622] Connect the Dots: In Situ 4D Seismic Monitoring of CO2 Storage with Spatio-temporal CNNs](http://arxiv.org/abs/2105.11622)


  4D seismic imaging has been widely used in CO$_2$ sequestration projects to
monitor the fluid flow in the volumetric subsurface region that is not sampled
by wells. Ideally, real-time monitoring and near-future forecasting would
provide site operators with great insights to understand the dynamics of the
subsurface reservoir and assess any potential risks. However, due to obstacles
such as high deployment cost, availability of acquisition equipment, exclusion
zones around surface structures, only very sparse seismic imaging data can be
obtained during monitoring. That leads to an unavoidable and growing knowledge
gap over time. The operator needs to understand the fluid flow throughout the
project lifetime and the seismic data are only available at a limited number of
times. This is insufficient for understanding the reservoir behavior. To
overcome those challenges, we have developed spatio-temporal
neural-network-based models that can produce high-fidelity interpolated or
extrapolated images effectively and efficiently. Specifically, our models are
built on an autoencoder, and incorporate the long short-term memory (LSTM)
structure with a new loss function regularized by optical flow. We validate the
performance of our models using real 4D post-stack seismic imaging data
acquired at the Sleipner CO$_2$ sequestration field. We employ two different
strategies in evaluating our models. Numerically, we compare our models with
different baseline approaches using classic pixel-based metrics. We also
conduct a blind survey and collect a total of 20 responses from domain experts
to evaluate the quality of data generated by our models. Via both numerical and
expert evaluation, we conclude that our models can produce high-quality 2D/3D
seismic imaging data at a reasonable cost, offering the possibility of
real-time monitoring or even near-future forecasting of the CO$_2$ storage
reservoir.

    

### [[2105.11839] DiBS: Differentiable Bayesian Structure Learning](http://arxiv.org/abs/2105.11839)


  Bayesian structure learning allows inferring Bayesian network structure from
data while reasoning about the epistemic uncertainty -- a key element towards
enabling active causal discovery and designing interventions in real world
systems. In this work, we propose a general, fully differentiable framework for
Bayesian structure learning (DiBS) that operates in the continuous space of a
latent probabilistic graph representation. Contrary to existing work, DiBS is
agnostic to the form of the local conditional distributions and allows for
joint posterior inference of both the graph structure and the conditional
distribution parameters. This makes DiBS directly applicable to posterior
inference of nonstandard Bayesian network models, e.g., with nonlinear
dependencies encoded by neural networks. Building on recent advances in
variational inference, we use DiBS to devise an efficient general purpose
method for approximating posteriors over structural models. In evaluations on
simulated and real-world data, our method significantly outperforms related
approaches to joint posterior inference.

    

### [[2106.09395] A Self-supervised Method for Entity Alignment](http://arxiv.org/abs/2106.09395)


  Entity alignment, aiming to identify equivalent entities across different
knowledge graphs (KGs), is a fundamental problem for constructing large-scale
KGs. Over the course of its development, supervision has been considered
necessary for accurate alignments. Inspired by the recent progress of
self-supervised learning, we explore the extent to which we can get rid of
supervision for entity alignment. Existing supervised methods for this task
focus on pulling each pair of positive (labeled) entities close to each other.
However, our analysis suggests that the learning of entity alignment can
actually benefit more from pushing sampled (unlabeled) negatives far away than
pulling positive aligned pairs close. We present SelfKG by leveraging this
discovery to design a contrastive learning strategy across two KGs. Extensive
experiments on benchmark datasets demonstrate that SelfKG without supervision
can match or achieve comparable results with state-of-the-art supervised
baselines. The performance of SelfKG demonstrates self-supervised learning
offers great potential for entity alignment in KGs.

    

### [[2108.01039] Large-scale quantum machine learning](http://arxiv.org/abs/2108.01039)


  Quantum computers promise to enhance machine learning for practical
applications. Quantum machine learning for real-world data has to handle
extensive amounts of high-dimensional data. However, conventional methods for
measuring quantum kernels are impractical for large datasets as they scale with
the square of the dataset size. Here, we measure quantum kernels using
randomized measurements to gain a quadratic speedup in computation time and
quickly process large datasets. Further, we efficiently encode high-dimensional
data into quantum computers with the number of features scaling linearly with
the circuit depth. The encoding is characterized by the quantum Fisher
information metric and is related to the radial basis function kernel. We
demonstrate the advantages of our methods by classifying images with the IBM
quantum computer. To achieve further speedups we distribute the quantum
computational tasks between different quantum computers. Our approach is
exceptionally robust to noise via a complementary error mitigation scheme.
Using currently available quantum computers, the MNIST database can be
processed within 220 hours instead of 10 years which opens up industrial
applications of quantum machine learning.

    

### [[2011.00719] Optimizing embedding-related quantum annealing parameters for reducing hardware bias](http://arxiv.org/abs/2011.00719)


  Quantum annealers have been designed to propose near-optimal solutions to
NP-hard optimization problems. However, the accuracy of current annealers such
as the ones of D-Wave Systems, Inc., is limited by environmental noise and
hardware biases. One way to deal with these imperfections and to improve the
quality of the annealing results is to apply a variety of pre-processing
techniques such as spin reversal (SR), anneal offsets (AO), or chain weights
(CW). Maximizing the effectiveness of these techniques involves performing
optimizations over a large number of parameters, which would be too costly if
needed to be done for each new problem instance. In this work, we show that the
aforementioned parameter optimization can be done for an entire class of
problems, given each instance uses a previously chosen fixed embedding.
Specifically, in the training phase, we fix an embedding E of a complete graph
onto the hardware of the annealer, and then run an optimization algorithm to
tune the following set of parameter values: the set of bits to be flipped for
SR, the specific qubit offsets for AO, and the distribution of chain weights,
optimized over a set of training graphs randomly chosen from that class, where
the graphs are embedded onto the hardware using E. In the testing phase, we
estimate how well the parameters computed during the training phase work on a
random selection of other graphs from that class. We investigate graph
instances of varying densities for the Maximum Clique, Maximum Cut, and Graph
Partitioning problems. Our results indicate that, compared to their default
behavior, substantial improvements of the annealing results can be achieved by
using the optimized parameters for SR, AO, and CW.

    

### [[2106.00065] Using machine learning for quantum annealing accuracy prediction](http://arxiv.org/abs/2106.00065)


  Quantum annealers, such as the device built by D-Wave Systems, Inc., offer a
way to compute solutions of NP-hard problems that can be expressed in Ising or
QUBO (quadratic unconstrained binary optimization) form. Although such
solutions are typically of very high quality, problem instances are usually not
solved to optimality due to imperfections of the current generations quantum
annealers. In this contribution, we aim to understand some of the factors
contributing to the hardness of a problem instance, and to use machine learning
models to predict the accuracy of the D-Wave 2000Q annealer for solving
specific problems. We focus on the Maximum Clique problem, a classic NP-hard
problem with important applications in network analysis, bioinformatics, and
computational chemistry. By training a machine learning classification model on
basic problem characteristics such as the number of edges in the graph, or
annealing parameters such as D-Wave's chain strength, we are able to rank
certain features in the order of their contribution to the solution hardness,
and present a simple decision tree which allows to predict whether a problem
will be solvable to optimality with the D-Wave 2000Q. We extend these results
by training a machine learning regression model that predicts the clique size
found by D-Wave.

    

### [[2108.10964] EQUAL: Improving the Fidelity of Quantum Annealers by Injecting Controlled Perturbations](http://arxiv.org/abs/2108.10964)


  Quantum computing is an information processing paradigm that uses
quantum-mechanical properties to speedup computationally hard problems.
Although promising, existing gate-based quantum computers consist of only a few
dozen qubits and are not large enough for most applications. On the other hand,
existing QAs with few thousand of qubits have the potential to solve some
domain-specific optimization problems. QAs are single instruction machines and
to execute a program, the problem is cast to a Hamiltonian, embedded on the
hardware, and a single quantum machine instruction (QMI) is run. Unfortunately,
noise and imperfections in hardware result in sub-optimal solutions on QAs even
if the QMI is run for thousands of trials.
The limited programmability of QAs mean that the user executes the same QMI
for all trials. This subjects all trials to a similar noise profile throughout
the execution, resulting in a systematic bias. We observe that systematic bias
leads to sub-optimal solutions and cannot be alleviated by executing more
trials or using existing error-mitigation schemes. To address this challenge,
we propose EQUAL (Ensemble Quantum Annealing). EQUAL generates an ensemble of
QMIs by adding controlled perturbations to the program QMI. When executed on
the QA, the ensemble of QMIs steers the program away from encountering the same
bias during all trials and thus, improves the quality of solutions. Our
evaluations using the 2041-qubit D-Wave QA show that EQUAL bridges the
difference between the baseline and the ideal by an average of 14% (and up to
26%), without requiring any additional trials. EQUAL can be combined with
existing error mitigation schemes to further bridge the difference between the
baseline and ideal by an average of 55% (and up to 68%).

    

### [[2108.11115] Correlation Differential Power Analysis Attack to Midori64](http://arxiv.org/abs/2108.11115)


  Today, Internet communication security has become more complex as technology
becomes faster and more efficient, especially for resource-limited devices such
as embedded devices, wireless sensors, and radio frequency identification
(RFID) tags, and Internet of Things (IoT). Lightweight encryption algorithms
provide security for these devices to protect data against intruders. But the
limitation of using energy in lightweight block ciphers (LBCs) is one of the
major challenges for ever-expanding IoT technologies. Also, these LBC are
subject to Side-channel attacks, which are among the most cited threats to
these ciphers. In this paper, a differential power attack (DPA) to the Midori64
block cipher is designed. According to the proposed method, an attack on the
S-boxes of the first round is done to obtain half of the master key bits. Then,
the S-boxes of the second round were attacked to obtain remaining the master
key bits. The results confirmed that the key is ultimately obtained. With the
low volume of computational complexity, we obtained the Midori block cipher
key, which was considered secure, just by using 300 samples of the plaintext.
Following the running of Midori64 on the AVR microcontroller of the Atmega32
model, the master key of Midori block cipher is discovered with 300 known
texts. Furthermore, we obtained the master key with a smaller number of samples
than the electromagnetic analysis attack.

    

### [[2108.11099] Towards Informed Partitioning for Load Balancing: a Proof-of-Concept](http://arxiv.org/abs/2108.11099)


  Most parallel applications suffer from load imbalance, a crucial performance
degradation factor. In particle simulations, this is mainly due to the
migration of particles between processing elements, which eventually gather
unevenly and create workload imbalance. Dynamic load balancing is used at
various iterations to mitigate load imbalance, employing a partitioning method
to divide the computational space evenly while minimizing communications. In
this paper, we propose a novel partitioning methodology called ``informed
partitioning''. It uses information based on the evolution of the computation
to reduce the load balancing growth and the number of load balancing calls. We
illustrate informed partitioning by proposing a new geometric partitioning
technique for particles simulations. This technique is derived from the
well-known recursive coordinate bisection and employs the velocity of the
particles to guide the bisection axis. To properly compare the performance of
our new method with existing partitioning techniques during application
execution, we introduce an effort metric based on a theoretical model of load
balanced parallel application time. We propose a proof-of-concept of informed
partitioning, through a numerical study, on three N-Body simulations with
various particle dynamics, and we discuss its performance against popular
geometric partitioning techniques. Moreover, we show that our effort metric can
be used to rank partitioning techniques by their efficiency at any time point
during the simulation. Eventually, this could be used to choose the best
partitioning on the fly. In the numerical study, we report that our novel
concept increases the performance of two experiments out of three by up to 76%
and 15%, while being marginally slower by only $3\%$ in one experiment. Also,
we discuss the limitations of our implementation of informed partitioning and
our effort metric.

    

### [[2108.11157] Cob: a Multidimensional Byzantine Agreement Protocol for Asynchronous Incomplete Networks](http://arxiv.org/abs/2108.11157)


  In this paper we extend the Multidimensional Byzantine Agreement (MBA)
Protocol arXiv:2105.13487v2, a leaderless Byzantine agreement for vectors of
arbitrary values, into the \emph{Cob} protocol, that works in Asynchronous
Gossiping (AG) networks. This generalization allows the consensus process to be
run by an incomplete network of nodes provided with (non-synchronized)
same-speed clocks. Not all nodes are active in every step, so the network size
does not hamper the efficiency, as long as the gossiping broadcast delivers the
messages to every node in reasonable time. These network assumptions model more
closely real-life communication channels, so the Cob protocol may be applicable
to a variety of practical problems, such as blockchain platforms implementing
sharding. The Cob protocol has the same Bernoulli-like distribution that upper
bounds the number of steps required as the MBA protocol, and we prove its
correctness and security assuming a supermajority of honest nodes in the
network.

    

### [[2108.11240] Pagurus: Eliminating Cold Startup in Serverless Computing with Inter-Action Container Sharing](http://arxiv.org/abs/2108.11240)


  Serverless computing provides fine-grain resource sharing between Cloud
tenants through containers. Each function invocation (action) runs in an
individual container. When there is not an already started container for a user
function, a new container has to be created for it. However, the long cold
startup time of a container results in the long response latency of the action.
Our investigation shows that the containers for some user actions share most of
the software packages. If an action that requires a new container can
``borrow'' a similar warm container from other actions, the long cold startup
can be eliminated. Based on the above finding, we propose Pagurus, a runtime
container management system for eliminating the cold startup in serverless
computing. Pagurus is comprised of an inter-action container scheduler and an
intra-action container scheduler for each action. The inter-action container
scheduler schedules shared containers among actions. The intra-action container
scheduler deals with the management of the container lifecycle. Our
experimental results show that Pagurus effectively eliminates the
time-consuming container cold startup. An action may start to run in 10ms with
Pagurus, even if there is not warm container for it.

    

### [[2108.11255] A Case for Sampling Based Learning Techniques in Coflow Scheduling](http://arxiv.org/abs/2108.11255)


  Coflow scheduling improves data-intensive application performance by
improving their networking performance. State-of-the-art online coflow
schedulers in essence approximate the classic Shortest-Job-First (SJF)
scheduling by learning the coflow size online. In particular, they use multiple
priority queues to simultaneously accomplish two goals: to sieve long coflows
from short coflows, and to schedule short coflows with high priorities. Such a
mechanism pays high overhead in learning the coflow size: moving a large coflow
across the queues delays small and other large coflows, and moving
similar-sized coflows across the queues results in inadvertent round-robin
scheduling. We propose Philae, a new online coflow scheduler that exploits the
spatial dimension of coflows, i.e., a coflow has many flows, to drastically
reduce the overhead of coflow size learning. Philae pre-schedules sampled flows
of each coflow and uses their sizes to estimate the average flow size of the
coflow. It then resorts to Shortest Coflow First, where the notion of shortest
is determined using the learned coflow sizes and coflow contention. We show
that the sampling-based learning is robust to flow size skew and has the added
benefit of much improved scalability from reduced coordinator-local agent
interactions. Our evaluation using an Azure testbed, a publicly available
production cluster trace from Facebook shows that compared to the prior art
Aalo, Philae reduces the coflow completion time (CCT) in average (P90) cases by
1.50x (8.00x) on a 150-node testbed and 2.72x (9.78x) on a 900-node testbed.
Evaluation using additional traces further demonstrates Philae's robustness to
flow size skew.

    

### [[2108.11359] Node-Based Job Scheduling for Large Scale Simulations of Short Running Jobs](http://arxiv.org/abs/2108.11359)


  Diverse workloads such as interactive supercomputing, big data analysis, and
large-scale AI algorithm development, requires a high-performance scheduler.
This paper presents a novel node-based scheduling approach for large scale
simulations of short running jobs on MIT SuperCloud systems, that allows the
resources to be fully utilized for both long running batch jobs while
simultaneously providing fast launch and release of large-scale short running
jobs. The node-based scheduling approach has demonstrated up to 100 times
faster scheduler performance that other state-of-the-art systems.

    

### [[2108.11004] Reasoning about Counterfactuals and Explanations: Problems, Results and Directions](http://arxiv.org/abs/2108.11004)


  There are some recent approaches and results about the use of answer-set
programming for specifying counterfactual interventions on entities under
classification, and reasoning about them. These approaches are flexible and
modular in that they allow the seamless addition of domain knowledge. Reasoning
is enabled by query answering from the answer-set program. The programs can be
used to specify and compute responsibility-based numerical scores as
attributive explanations for classification results.

    

### [[2108.11068] Understanding Longitudinal Dynamics of Recommender Systems with Agent-Based Modeling and Simulation](http://arxiv.org/abs/2108.11068)


  Today's research in recommender systems is largely based on experimental
designs that are static in a sense that they do not consider potential
longitudinal effects of providing recommendations to users. In reality,
however, various important and interesting phenomena only emerge or become
visible over time, e.g., when a recommender system continuously reinforces the
popularity of already successful artists on a music streaming site or when
recommendations that aim at profit maximization lead to a loss of consumer
trust in the long run. In this paper, we discuss how Agent-Based Modeling and
Simulation (ABM) techniques can be used to study such important longitudinal
dynamics of recommender systems. To that purpose, we provide an overview of the
ABM principles, outline a simulation framework for recommender systems based on
the literature, and discuss various practical research questions that can be
addressed with such an ABM-based simulation framework.

    

### [[2108.11126] YANMTT: Yet Another Neural Machine Translation Toolkit](http://arxiv.org/abs/2108.11126)


  In this paper we present our open-source neural machine translation (NMT)
toolkit called "Yet Another Neural Machine Translation Toolkit" abbreviated as
YANMTT which is built on top of the Transformers library. Despite the growing
importance of sequence to sequence pre-training there surprisingly few, if not
none, well established toolkits that allow users to easily do pre-training.
Toolkits such as Fairseq which do allow pre-training, have very large codebases
and thus they are not beginner friendly. With regards to transfer learning via
fine-tuning most toolkits do not explicitly allow the user to have control over
what parts of the pre-trained models can be transferred. YANMTT aims to address
these issues via the minimum amount of code to pre-train large scale NMT
models, selectively transfer pre-trained parameters and fine-tune them, perform
translation as well as extract representations and attentions for visualization
and analyses. Apart from these core features our toolkit also provides other
advanced functionalities such as but not limited to document/multi-source NMT,
simultaneous NMT and model compression via distillation which we believe are
relevant to the purpose behind our toolkit.

    

### [[2108.11172] Superpixel-guided Discriminative Low-rank Representation of Hyperspectral Images for Classification](http://arxiv.org/abs/2108.11172)


  In this paper, we propose a novel classification scheme for the remotely
sensed hyperspectral image (HSI), namely SP-DLRR, by comprehensively exploring
its unique characteristics, including the local spatial information and
low-rankness. SP-DLRR is mainly composed of two modules, i.e., the
classification-guided superpixel segmentation and the discriminative low-rank
representation, which are iteratively conducted. Specifically, by utilizing the
local spatial information and incorporating the predictions from a typical
classifier, the first module segments pixels of an input HSI (or its
restoration generated by the second module) into superpixels. According to the
resulting superpixels, the pixels of the input HSI are then grouped into
clusters and fed into our novel discriminative low-rank representation model
with an effective numerical solution. Such a model is capable of increasing the
intra-class similarity by suppressing the spectral variations locally while
promoting the inter-class discriminability globally, leading to a restored HSI
with more discriminative pixels. Experimental results on three benchmark
datasets demonstrate the significant superiority of SP-DLRR over
state-of-the-art methods, especially for the case with an extremely limited
number of training pixels.

    

### [[2108.11244] Multiscale Spatio-Temporal Graph Neural Networks for 3D Skeleton-Based Motion Prediction](http://arxiv.org/abs/2108.11244)


  We propose a multiscale spatio-temporal graph neural network (MST-GNN) to
predict the future 3D skeleton-based human poses in an action-category-agnostic
manner. The core of MST-GNN is a multiscale spatio-temporal graph that
explicitly models the relations in motions at various spatial and temporal
scales. Different from many previous hierarchical structures, our multiscale
spatio-temporal graph is built in a data-adaptive fashion, which captures
nonphysical, yet motion-based relations. The key module of MST-GNN is a
multiscale spatio-temporal graph computational unit (MST-GCU) based on the
trainable graph structure. MST-GCU embeds underlying features at individual
scales and then fuses features across scales to obtain a comprehensive
representation. The overall architecture of MST-GNN follows an encoder-decoder
framework, where the encoder consists of a sequence of MST-GCUs to learn the
spatial and temporal features of motions, and the decoder uses a graph-based
attention gate recurrent unit (GA-GRU) to generate future poses. Extensive
experiments are conducted to show that the proposed MST-GNN outperforms
state-of-the-art methods in both short and long-term motion prediction on the
datasets of Human 3.6M, CMU Mocap and 3DPW, where MST-GNN outperforms previous
works by 5.33% and 3.67% of mean angle errors in average for short-term and
long-term prediction on Human 3.6M, and by 11.84% and 4.71% of mean angle
errors for short-term and long-term prediction on CMU Mocap, and by 1.13% of
mean angle errors on 3DPW in average, respectively. We further investigate the
learned multiscale graphs for interpretability.

    

### [[2011.12461] Deep Discriminative Feature Learning for Accent Recognition](http://arxiv.org/abs/2011.12461)


  Accent recognition with deep learning framework is a similar work to deep
speaker identification, they're both expected to give the input speech an
identifiable representation.
Compared with the individual-level features learned by speaker identification
network, the deep accent recognition work throws a more challenging point that
forging group-level accent features for speakers.
In this paper, we borrow and improve the deep speaker identification
framework to recognize accents, in detail, we adopt Convolutional Recurrent
Neural Network as front-end encoder and integrate local features using
Recurrent Neural Network to make an utterance-level accent representation.
Novelly, to address overfitting, we simply add Connectionist Temporal
Classification based speech recognition auxiliary task during training, and for
ambiguous accent discrimination, we introduce some powerful discriminative loss
functions in face recognition works to enhance the discriminative power of
accent features.
We show that our proposed network with discriminative training method
(without data-augment) is significantly ahead of the baseline system on the
accent classification track in the Accented English Speech Recognition
Challenge 2020, where the loss function Circle-Loss has achieved the best
discriminative optimization for accent representation.

    

### [[2104.07650] Knowledge-aware Prompt-tuning with Synergistic Optimization for Relation Extraction](http://arxiv.org/abs/2104.07650)


  Recently, prompt-tuning has achieved promising results on some few-class
classification tasks. The core idea of prompt-tuning is to insert text pieces,
i.e., template, to the input and transform a classification task into a masked
language modeling problem. However, as for relation extraction, determining the
appropriate prompt template requires domain expertise, and single label word
handcrafted or auto-searched is cumbersome and time-consuming to verify their
effectiveness in non-few-shot scenarios, which also fails to leverage the
abundant semantic knowledge in the entities and relation labels. To this end,
we focus on incorporating knowledge into prompt-tuning for relation extraction
and propose a knowledge-aware prompt-tuning with synergistic optimization
(KNIGHT) approach. Specifically, we inject entity and relation knowledge into
prompt construction with learnable virtual template words and answer words and
jointly optimize their representation with knowledge constraints. Extensive
experimental results on 5 datasets with standard and low-resource settings
demonstrate the effectiveness of our approach.

    

### [[2104.09119] Locate Who You Are: Matching Geo-location to Text for User Identity Linkage](http://arxiv.org/abs/2104.09119)


  Nowadays, users are encouraged to activate across multiple online social
networks simultaneously. Anchor link prediction, which aims to reveal the
correspondence among different accounts of the same user across networks, has
been regarded as a fundamental problem for user profiling, marketing,
cybersecurity, and recommendation. Existing methods mainly address the
prediction problem by utilizing profile, content, or structural features of
users in symmetric ways. However, encouraged by online services, users would
also post asymmetric information across networks, such as geo-locations and
texts. It leads to an emerged challenge in aligning users with asymmetric
information across networks. Instead of similarity evaluation applied in
previous works, we formalize correlation between geo-locations and texts and
propose a novel anchor link prediction framework for matching users across
networks. Moreover, our model can alleviate the label scarcity problem by
introducing external data. Experimental results on real-world datasets show
that our approach outperforms existing methods and achieves state-of-the-art
results.

    

### [[2105.00451] Multi-Agent Routing and Scheduling Through Coalition Formation](http://arxiv.org/abs/2105.00451)


  In task allocation for real-time domains, such as disaster response, a
limited number of agents is deployed across a large area to carry out numerous
tasks, each with its prerequisites, profit, time window and workload. To
maximize profits while minimizing time penalties, agents need to cooperate by
forming, disbanding and reforming coalitions. In this paper, we name this
problem Multi-Agent Routing and Scheduling through Coalition formation (MARSC)
and show that it generalizes the important Team Orienteering Problem with Time
Windows. We propose a binary integer program and an anytime and scalable
heuristic to solve it. Using public London Fire Brigade records, we create a
dataset with 347588 tasks and a test framework that simulates the mobilization
of firefighters. In problems with up to 150 agents and 3000 tasks, our
heuristic finds solutions up to 3.25 times better than the Earliest Deadline
First approach commonly used in real-time systems. Our results constitute the
first large-scale benchmark for the MARSC problem.

    

### [[2105.06453] Episodic Transformer for Vision-and-Language Navigation](http://arxiv.org/abs/2105.06453)


  Interaction and navigation defined by natural language instructions in
dynamic environments pose significant challenges for neural agents. This paper
focuses on addressing two challenges: handling long sequence of subtasks, and
understanding complex human instructions. We propose Episodic Transformer
(E.T.), a multimodal transformer that encodes language inputs and the full
episode history of visual observations and actions. To improve training, we
leverage synthetic instructions as an intermediate representation that
decouples understanding the visual appearance of an environment from the
variations of natural language instructions. We demonstrate that encoding the
history with a transformer is critical to solve compositional tasks, and that
pretraining and joint training with synthetic instructions further improve the
performance. Our approach sets a new state of the art on the challenging ALFRED
benchmark, achieving 38.4% and 8.5% task success rates on seen and unseen test
splits.

    

### [[2106.11576] Universal Domain Adaptation in Ordinal Regression](http://arxiv.org/abs/2106.11576)


  We address the problem of universal domain adaptation (UDA) in ordinal
regression (OR), which attempts to solve classification problems in which
labels are not independent, but follow a natural order. We show that the UDA
techniques developed for classification and based on the clustering assumption,
under-perform in OR settings. We propose a method that complements the OR
classifier with an auxiliary task of order learning, which plays the double
role of discriminating between common and private instances, and expanding
class labels to the private target images via ranking. Combined with
adversarial domain discrimination, our model is able to address the closed set,
partial and open set configurations. We evaluate our method on three face age
estimation datasets, and show that it outperforms the baseline methods.

    

### [[2108.11155] Latent Effects for Reusable Language Components: Extended Version](http://arxiv.org/abs/2108.11155)


  The development of programming languages can be quite complicated and costly.
Hence, much effort has been devoted to the modular definition of language
features that can be reused in various combinations to define new languages and
experiment with their semantics. A notable outcome of these efforts is the
algebra-based "datatypes a la carte" (DTC) approach. When combined with
algebraic effects, DTC can model a wide range of common language features.
Unfortunately, the current state of the art does not cover modular definitions
of advanced control-flow mechanisms that defer execution to an appropriate
point, such as call-by-name and call-by-need evaluation, as well as
(multi-)staging. This paper defines latent effects, a generic class of such
control-flow mechanisms. We demonstrate how function abstractions, lazy
computations and a MetaML-like staging can all be expressed in a modular
fashion using latent effects, and how they can be combined in various ways to
obtain complex semantics. We provide a full Haskell implementation of our
effects and handlers with a range of examples.

    

### [[2108.11212] The Choice Construct in the Souffle Language](http://arxiv.org/abs/2108.11212)


  Datalog has become a popular implementation language for solving large-scale,
real-world problems, including bug finders, network analysis tools, and
disassemblers. These applications express complex behaviour with hundreds of
relations and rules that often require a non-deterministic choice for tuples in
relations to express worklist algorithms. This work is an experience report
that describes the implementation of a choice construct in the Datalog engine
Souffle. With the choice construct, we can express worklist algorithms such as
spanning trees in a few lines of code. We highlight the differences between
rule-based choice as described in prior work, and relation-based choice
introduced by this work. We show that a choice construct enables certain
worklist algorithms to be computed up to 10kx faster than having no choice
construct.

    

### [[2108.11347] The Next 700 Program Transformers](http://arxiv.org/abs/2108.11347)


  In this paper, we describe a hierarchy of program transformers in which the
transformer at each level of the hierarchy builds on top of those at lower
levels. The program transformer at level 1 of the hierarchy corresponds to
positive supercompilation, and that at level 2 corresponds to distillation. We
prove that the transformers at each level terminate. We then consider the
speedups that can be obtained at each level in the hierarchy, and try to
characterise the improvements that can be made.

    

### [[2105.04385] Identifying Overly Restrictive Matching Patterns in SMT-based Program Verifiers](http://arxiv.org/abs/2105.04385)


  Universal quantifiers occur frequently in proof obligations produced by
program verifiers, for instance, to axiomatize uninterpreted functions and to
express properties of arrays. SMT-based verifiers typically reason about them
via E-matching, an SMT algorithm that requires syntactic matching patterns to
guide the quantifier instantiations. Devising good matching patterns is
challenging. In particular, overly restrictive patterns may lead to spurious
verification errors if the quantifiers needed for a proof are not instantiated;
they may also conceal unsoundness caused by inconsistent axiomatizations. In
this paper, we present the first technique that identifies and helps the users
remedy the effects of overly restrictive matching patterns. We designed a novel
algorithm to synthesize missing triggering terms required to complete a proof.
Tool developers can use this information to refine their matching patterns and
prevent similar verification errors, or to fix a detected unsoundness.

    