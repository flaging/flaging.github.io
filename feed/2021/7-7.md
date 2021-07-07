
## 2021-7-7

### [[2106.15113] An Efficient Cervical Whole Slide Image Analysis Framework Based on Multi-scale Semantic and Spatial Deep Features](http://arxiv.org/abs/2106.15113)


  Digital gigapixel whole slide image (WSI) is widely used in clinical
diagnosis, and automated WSI analysis is key for computer-aided diagnosis.
Currently, analyzing the integrated descriptor of probabilities or feature maps
from massive local patches encoded by ResNet classifier is the main manner for
WSI-level prediction. Feature representations of the sparse and tiny lesion
cells in cervical slides, however, are still challengeable for the
under-promoted upstream encoders, while the unused spatial representations of
cervical cells are the available features to supply the semantics analysis. As
well as patches sampling with overlap and repetitive processing incur the
inefficiency and the unpredictable side effect. This study designs a novel
inline connection network (InCNet) by enriching the multi-scale connectivity to
build the lightweight model named You Only Look Cytopathology Once (YOLCO) with
the additional supervision of spatial information. The proposed model allows
the input size enlarged to megapixel that can stitch the WSI without any
overlap by the average repeats decreased from $10^3\sim10^4$ to $10^1\sim10^2$
for collecting features and predictions at two scales. Based on Transformer for
classifying the integrated multi-scale multi-task features, the experimental
results appear $0.872$ AUC score better and $2.51\times$ faster than the best
conventional method in WSI classification on multicohort datasets of 2,019
slides from four scanning devices.

    

### [<title>Is there any unnecessary or debugging files in the XGBoost model to remove and make the model size smaller? - XGBoost</title>](https://discuss.xgboost.ai/t/is-there-any-unnecessary-or-debugging-files-in-the-xgboost-model-to-remove-and-make-the-model-size-smaller/2360/2)

### [<title>Is there any unnecessary or debugging files in the XGBoost model to remove and make the model size smaller? - XGBoost</title>](https://discuss.xgboost.ai/t/is-there-any-unnecessary-or-debugging-files-in-the-xgboost-model-to-remove-and-make-the-model-size-smaller/2360/1)

### [[2107.02244] Lucid: A Language for Control in the Data Plane](http://arxiv.org/abs/2107.02244)


  Programmable switch hardware makes it possible to move fine-grained control
logic inside the network data plane, improving performance for a wide range of
applications. However, applications with integrated control are inherently hard
to write in existing data-plane programming languages such as P4. This paper
presents Lucid, a language that raises the level of abstraction for putting
control functionality in the data plane. Lucid introduces abstractions that
make it easy to write sophisticated data-plane applications with interleaved
packet-handling and control logic, specialized type and syntax systems that
prevent programmer bugs related to data-plane state, and an open-sourced
compiler that translates Lucid programs into P4 optimized for the Intel Tofino.
These features make Lucid general and easy to use, as we demonstrate by writing
a suite of ten different data-plane applications in Lucid. Working prototypes
take well under an hour to write, even for a programmer without prior Tofino
experience, have around 10x fewer lines of code compared to P4, and compile
efficiently to real hardware. In a stateful firewall written in Lucid, we find
that moving control from a switch's CPU to its data-plane processor using Lucid
reduces the latency of performance-sensitive operations by over 300X.

    

### [[2107.02271] LUCID: Receiver-aware Model-based Data Communication for Low-power Wireless Networks](http://arxiv.org/abs/2107.02271)


  In the last decade, the advancement of the Internet of Things (IoT) has
caused unlicensed radio spectrum, especially the 2.4 GHz ISM band, to be
immensely crowded with smart wireless devices that are used in a wide range of
application domains. Due to their diversity in radio resource use and channel
access techniques, when collocated, these wireless devices create interference
with each other, known as Cross-Technology Interference (CTI), which can lead
to increased packet losses and energy consumption. CTI is a significant problem
for low-power wireless networks, such as IEEE 802.15.4, as it decreases the
overall dependability of the wireless network.
To improve the performance of low-power wireless networks under CTI
conditions, we propose a data-driven proactive receiver-aware MAC protocol,
LUCID, based on interference estimation and white space prediction. We leverage
statistical analysis of real-world traces from two indoor environments
characterised by varying channel conditions to develop CTI prediction methods.
The CTI models that generate accurate predictions of interference behaviour are
an intrinsic part of our solution. LUCID is thoroughly evaluated in realistic
simulations and we show that depending on the application data rate and the
network size, our solution achieves higher dependability, 1.2% increase in
packet delivery ratio and 0.02% decrease in duty-cycle under bursty indoor
interference than state of the art alternative methods.

    

### [[2107.02342] Energy and Thermal-aware Resource Management of Cloud Data Centres: A Taxonomy and Future Directions](http://arxiv.org/abs/2107.02342)


  This paper investigates the existing resource management approaches in Cloud
Data Centres for energy and thermal efficiency. It identifies the need for
integrated computing and cooling systems management and learning-based
solutions in resource management systems. A taxonomy on energy and thermal
efficient resource management in data centres is proposed based on an in-depth
analysis of the literature. Furthermore, a detailed survey on existing
approaches is conducted according to the taxonomy and recent advancements
including machine learning-based resource management approaches and cooling
management technologies are discussed.

    

### [[2107.02409] Edge-powered Assisted Driving For Connected Cars](http://arxiv.org/abs/2107.02409)


  Assisted driving for connected cars is one of the main applications that
5G-and-beyond networks shall support. In this work, we propose an assisted
driving system leveraging the synergy between connected vehicles and the edge
of the network infrastructure, in order to envision global traffic policies
that can effectively drive local decisions. Local decisions concern individual
vehicles, e.g., which vehicle should perform a lane-change manoeuvre and when;
global decisions, instead, involve whole traffic flows. Such decisions are made
at different time scales by different entities, which are integrated within an
edge-based architecture and can share information. In particular, we leverage a
queuing-based model and formulate an optimization problem to make global
decisions on traffic flows. To cope with the problem complexity, we then
develop an iterative, linear-time complexity algorithm called Bottleneck
Hunting (BH). We show the performance of our solution using a realistic
simulation framework, integrating a Python engine with ns-3 and SUMO, and
considering two relevant services, namely, lane change assistance and
navigation, in a real-world scenario. Results demonstrate that our solution
leads to a reduction of the vehicles' travel times by 66 in the case of lane
change assistance and by 20 for navigation, compared to traditional,
local-coordination approaches.

    

### [[2107.02496] Convolutional LSTM models to estimate network traffic](http://arxiv.org/abs/2107.02496)


  Network utilisation efficiency can, at least in principle, often be improved
by dynamically re-configuring routing policies to better distribute on-going
large data transfers. Unfortunately, the information necessary to decide on an
appropriate reconfiguration - details of on-going and upcoming data transfers
such as their source and destination and, most importantly, their volume and
duration - is usually lacking. Fortunately, the increased use of scheduled
transfer services, such as FTS, makes it possible to collect the necessary
information. However, the mere detection and characterisation of larger
transfers is not sufficient to predict with confidence the likelihood a network
link will become overloaded. In this paper we present the use of LSTM-based
models (CNN-LSTM and Conv-LSTM) to effectively estimate future network traffic
and so provide a solid basis for formulating a sensible network configuration
plan.

    

### [[2107.02505] A Latency-Aware Real-Time Video Surveillance Demo: Network Slicing for Improving Public Safety](http://arxiv.org/abs/2107.02505)


  We report the automated deployment of 5G services across a latency-aware,
semidisaggregated, and virtualized metro network. We summarize the key findings
in a detailed analysis of end-to-end latency, service setup time, and
soft-failure detection time.

    

### [[2107.02551] Implementation of RPL in OMNeT++](http://arxiv.org/abs/2107.02551)


  The growth and evolution of Internet of Things (IoT) is now of paramount
importance for next-generation networks, including the upcoming 6G. In
particular, there is a set of constrained IoT nodes that comprise the Low-Power
and Lossy Networks (LLNs), which have very particular requirements. The current
standard for routing in those networks is RPL, which was defined less than a
decade ago and still needs improvements in terms of scalability or integration
with other networks. Many researchers currently need an implementation of RPL
to evaluate their works and, for that reason, among others, we implemented it
in the OMNeT++ simulator. The results of this implementation show that is an
easy way to check prototypes in their very initial develop phases, and its code
is publicly available for the research community.

    

### [[2009.08228] LeadCache: Regret-Optimal Caching in Networks](http://arxiv.org/abs/2009.08228)


  We consider a set-valued online prediction problem in the context of network
caching. Assume that multiple users are connected to several caches via a
bipartite network. At any time slot, each user requests an arbitrary file
chosen from a large catalog. A user's request at a slot is met if the requested
file is cached in at least one of the caches connected to the user. Our
objective is to predict, prefetch, and optimally distribute the files on the
caches to maximize the total number of cache hits in an online setting. The
problem is non-trivial due to the non-convex and non-smooth nature of the
objective function. In this paper, we propose $\texttt{LeadCache}$ - an online
caching policy based on the Follow-the-Perturbed-Leader paradigm. We show that
the policy is regret-optimal up to a factor of $\tilde{O}(n^{3/8}),$ where $n$
is the number of users. We design two efficient implementations of the
$\texttt{LeadCache}$ policy, one based on Pipage rounding and the other based
on Madow's sampling, each of which makes precisely one call to an LP-solver per
iteration. With a Strong-Law-type assumption, we show that the total number of
file fetches under $\texttt{LeadCache}$ remains almost surely finite over an
infinite horizon. Finally, we derive a tight regret lower bound using results
from graph coloring. We conclude that the learning-based $\texttt{LeadCache}$
policy decisively outperforms the known caching policies both theoretically and
empirically.

    

### [[2107.02189] Label noise in segmentation networks : mitigation must deal with bias](http://arxiv.org/abs/2107.02189)


  Imperfect labels limit the quality of predictions learned by deep neural
networks. This is particularly relevant in medical image segmentation, where
reference annotations are difficult to collect and vary significantly even
across expert annotators. Prior work on mitigating label noise focused on
simple models of mostly uniform noise. In this work, we explore biased and
unbiased errors artificially introduced to brain tumour annotations on MRI
data. We found that supervised and semi-supervised segmentation methods are
robust or fairly robust to unbiased errors but sensitive to biased errors. It
is therefore important to identify the sorts of errors expected in medical
image labels and especially mitigate the biased errors.

    

### [[2107.02191] TransformerFusion: Monocular RGB Scene Reconstruction using Transformers](http://arxiv.org/abs/2107.02191)


  We introduce TransformerFusion, a transformer-based 3D scene reconstruction
approach. From an input monocular RGB video, the video frames are processed by
a transformer network that fuses the observations into a volumetric feature
grid representing the scene; this feature grid is then decoded into an implicit
3D scene representation. Key to our approach is the transformer architecture
that enables the network to learn to attend to the most relevant image frames
for each 3D location in the scene, supervised only by the scene reconstruction
task. Features are fused in a coarse-to-fine fashion, storing fine-level
features only where needed, requiring lower memory storage and enabling fusion
at interactive rates. The feature grid is then decoded to a higher-resolution
scene reconstruction, using an MLP-based surface occupancy prediction from
interpolated coarse-to-fine 3D features. Our approach results in an accurate
surface reconstruction, outperforming state-of-the-art multi-view stereo depth
estimation methods, fully-convolutional 3D reconstruction approaches, and
approaches using LSTM- or GRU-based recurrent networks for video sequence
fusion.

    

### [[2107.02192] Long-Short Transformer: Efficient Transformers for Language and Vision](http://arxiv.org/abs/2107.02192)


  Transformers have achieved success in both language and vision domains.
However, it is prohibitively expensive to scale them to long sequences such as
long documents or high-resolution images, because self-attention mechanism has
quadratic time and memory complexities with respect to the input sequence
length. In this paper, we propose Long-Short Transformer (Transformer-LS), an
efficient self-attention mechanism for modeling long sequences with linear
complexity for both language and vision tasks. It aggregates a novel long-range
attention with dynamic projection to model distant correlations and a
short-term attention to capture fine-grained local correlations. We propose a
dual normalization strategy to account for the scale mismatch between the two
attention mechanisms. Transformer-LS can be applied to both autoregressive and
bidirectional models without additional complexity. Our method outperforms the
state-of-the-art models on multiple tasks in language and vision domains,
including the Long Range Arena benchmark, autoregressive language modeling, and
ImageNet classification. For instance, Transformer-LS achieves 0.97 test BPC on
enwik8 using half the number of parameters than previous method, while being
faster and is able to handle 3$\times$ as long sequences compared to its
full-attention version on the same hardware. On ImageNet, it can obtain the
state-of-the-art results~(e.g., Top-1 accuracy 84.1% trained on 224$\times$224
ImageNet-1K only), while being more scalable on high-resolution images. The
models and source code will be released soon.

    

### [[2107.02195] Agents that Listen: High-Throughput Reinforcement Learning with Multiple Sensory Systems](http://arxiv.org/abs/2107.02195)


  Humans and other intelligent animals evolved highly sophisticated perception
systems that combine multiple sensory modalities. On the other hand,
state-of-the-art artificial agents rely mostly on visual inputs or structured
low-dimensional observations provided by instrumented environments. Learning to
act based on combined visual and auditory inputs is still a new topic of
research that has not been explored beyond simple scenarios. To facilitate
progress in this area we introduce a new version of VizDoom simulator to create
a highly efficient learning environment that provides raw audio observations.
We study the performance of different model architectures in a series of tasks
that require the agent to recognize sounds and execute instructions given in
natural language. Finally, we train our agent to play the full game of Doom and
find that it can consistently defeat a traditional vision-based adversary. We
are currently in the process of merging the augmented simulator with the main
ViZDoom code repository. Video demonstrations and experiment code can be found
at this https URL.

    

### [[2107.02211] Automated age-related macular degeneration area estimation -- first results](http://arxiv.org/abs/2107.02211)


  This work aims to research an automatic method for detecting Age-related
Macular Degeneration (AMD) lesions in RGB eye fundus images. For this, we align
invasively obtained eye fundus contrast images (the "golden standard"
diagnostic) to the RGB ones and use them to hand-annotate the lesions. This is
done using our custom-made tool. Using the data, we train and test five
different convolutional neural networks: a custom one to classify healthy and
AMD-affected eye fundi, and four well-known networks: ResNet50, ResNet101,
MobileNetV3, and UNet to segment (localize) the AMD lesions in the affected eye
fundus images. We achieve 93.55% accuracy or 69.71% Dice index as the
preliminary best results in segmentation with MobileNetV3.

    

### [[2107.02212] Featurized Density Ratio Estimation](http://arxiv.org/abs/2107.02212)


  Density ratio estimation serves as an important technique in the unsupervised
machine learning toolbox. However, such ratios are difficult to estimate for
complex, high-dimensional data, particularly when the densities of interest are
sufficiently different. In our work, we propose to leverage an invertible
generative model to map the two distributions into a common feature space prior
to estimation. This featurization brings the densities closer together in
latent space, sidestepping pathological scenarios where the learned density
ratios in input space can be arbitrarily inaccurate. At the same time, the
invertibility of our feature map guarantees that the ratios computed in feature
space are equivalent to those in input space. Empirically, we demonstrate the
efficacy of our approach in a variety of downstream tasks that require access
to accurate density ratios such as mutual information estimation, targeted
sampling in deep generative models, and classification with data augmentation.

    

### [[2107.02228] Meta-learning Amidst Heterogeneity and Ambiguity](http://arxiv.org/abs/2107.02228)


  Meta-learning aims to learn a model that can handle multiple tasks generated
from an unknown but shared distribution. However, typical meta-learning
algorithms have assumed the tasks to be similar such that a single meta-learner
is sufficient to aggregate the variations in all aspects. In addition, there
has been less consideration on uncertainty when limited information is given as
context. In this paper, we devise a novel meta-learning framework, called
Meta-learning Amidst Heterogeneity and Ambiguity (MAHA), that outperforms
previous works in terms of prediction based on its ability on task
identification. By extensively conducting several experiments in regression and
classification, we demonstrate the validity of our model, which turns out to be
robust to both task heterogeneity and ambiguity.

    

### [[2107.02232] A Deep Learning-Based Particle-in-Cell Method for Plasma Simulations](http://arxiv.org/abs/2107.02232)


  We design and develop a new Particle-in-Cell (PIC) method for plasma
simulations using Deep-Learning (DL) to calculate the electric field from the
electron phase space. We train a Multilayer Perceptron (MLP) and a
Convolutional Neural Network (CNN) to solve the two-stream instability test. We
verify that the DL-based MLP PIC method produces the correct results using the
two-stream instability: the DL-based PIC provides the expected growth rate of
the two-stream instability. The DL-based PIC does not conserve the total energy
and momentum. However, the DL-based PIC method is stable against the cold-beam
instability, affecting traditional PIC methods. This work shows that
integrating DL technologies into traditional computational methods is a viable
approach for developing next-generation PIC algorithms.

    

### [[2107.02233] End-to-End Weak Supervision](http://arxiv.org/abs/2107.02233)


  Aggregating multiple sources of weak supervision (WS) can ease the
data-labeling bottleneck prevalent in many machine learning applications, by
replacing the tedious manual collection of ground truth labels. Current state
of the art approaches that do not use any labeled training data, however,
require two separate modeling steps: Learning a probabilistic latent variable
model based on the WS sources -- making assumptions that rarely hold in
practice -- followed by downstream model training. Importantly, the first step
of modeling does not consider the performance of the downstream model. To
address these caveats we propose an end-to-end approach for directly learning
the downstream model by maximizing its agreement with probabilistic labels
generated by reparameterizing previous probabilistic posteriors with a neural
network. Our results show improved performance over prior work in terms of end
model performance on downstream test sets, as well as in terms of improved
robustness to dependencies among weak supervision sources.

    

### [[2107.02237] Efficient First-Order Contextual Bandits: Prediction, Allocation, and Triangular Discrimination](http://arxiv.org/abs/2107.02237)


  A recurring theme in statistical learning, online learning, and beyond is
that faster convergence rates are possible for problems with low noise, often
quantified by the performance of the best hypothesis; such results are known as
first-order or small-loss guarantees. While first-order guarantees are
relatively well understood in statistical and online learning, adapting to low
noise in contextual bandits (and more broadly, decision making) presents major
algorithmic challenges. In a COLT 2017 open problem, Agarwal, Krishnamurthy,
Langford, Luo, and Schapire asked whether first-order guarantees are even
possible for contextual bandits and -- if so -- whether they can be attained by
efficient algorithms. We give a resolution to this question by providing an
optimal and efficient reduction from contextual bandits to online regression
with the logarithmic (or, cross-entropy) loss. Our algorithm is simple and
practical, readily accommodates rich function classes, and requires no
distributional assumptions beyond realizability. In a large-scale empirical
evaluation, we find that our approach typically outperforms comparable
non-first-order methods.
On the technical side, we show that the logarithmic loss and an
information-theoretic quantity called the triangular discrimination play a
fundamental role in obtaining first-order guarantees, and we combine this
observation with new refinements to the regression oracle reduction framework
of Foster and Rakhlin. The use of triangular discrimination yields novel
results even for the classical statistical learning model, and we anticipate
that it will find broader use.

    

### [[2107.02239] Vision Xformers: Efficient Attention for Image Classification](http://arxiv.org/abs/2107.02239)


  Linear attention mechanisms provide hope for overcoming the bottleneck of
quadratic complexity which restricts application of transformer models in
vision tasks. We modify the ViT architecture to work on longer sequence data by
replacing the quadratic attention with efficient transformers like Performer,
Linformer and Nyströmformer of linear complexity creating Vision X-formers
(ViX). We show that ViX performs better than ViT in image classification
consuming lesser computing resources. We further show that replacing the
embedding linear layer by convolutional layers in ViX further increases their
performance. Our test on recent visions transformer models like LeViT and
Compact Convolutional Transformer (CCT) show that replacing the attention with
Nyströmformer or Performer saves GPU usage and memory without deteriorating
performance. Incorporating these changes can democratize transformers by making
them accessible to those with limited data and computing resources.

    

### [[2107.02248] A comparison of LSTM and GRU networks for learning symbolic sequences](http://arxiv.org/abs/2107.02248)


  We explore relations between the hyper-parameters of a recurrent neural
network (RNN) and the complexity of string sequences it is able to memorize. We
compare long short-term memory (LSTM) networks and gated recurrent units
(GRUs). We find that an increase of RNN depth does not necessarily result in
better memorization capability when the training time is constrained. Our
results also indicate that the learning rate and the number of units per layer
are among the most important hyper-parameters to be tuned. Generally, GRUs
outperform LSTM networks on low complexity sequences while on high complexity
sequences LSTMs perform better.

    

### [[2107.02253] Generalization by design: Shortcuts to Generalization in Deep Learning](http://arxiv.org/abs/2107.02253)


  We take a geometrical viewpoint and present a unifying view on supervised
deep learning with the Bregman divergence loss function - this entails frequent
classification and prediction tasks. Motivated by simulations we suggest that
there is principally no implicit bias of vanilla stochastic gradient descent
training of deep models towards "simpler" functions. Instead, we show that good
generalization may be instigated by bounded spectral products over layers
leading to a novel geometric regularizer. It is revealed that in deep enough
models such a regularizer enables both, extreme accuracy and generalization, to
be reached. We associate popular regularization techniques like weight decay,
drop out, batch normalization, and early stopping with this perspective. Backed
up by theory we further demonstrate that "generalization by design" is
practically possible and that good generalization may be encoded into the
structure of the network. We design two such easy-to-use structural
regularizers that insert an additional \textit{generalization layer} into a
model architecture, one with a skip connection and another one with drop-out.
We verify our theoretical results in experiments on various feedforward and
convolutional architectures, including ResNets, and datasets (MNIST, CIFAR10,
synthetic data). We believe this work opens up new avenues of research towards
better generalizing architectures.

    

### [[2107.02259] VolNet: Estimating Human Body Part Volumes from a Single RGB Image](http://arxiv.org/abs/2107.02259)


  Human body volume estimation from a single RGB image is a challenging problem
despite minimal attention from the research community. However VolNet, an
architecture leveraging 2D and 3D pose estimation, body part segmentation and
volume regression extracted from a single 2D RGB image combined with the
subject's body height can be used to estimate the total body volume. VolNet is
designed to predict the 2D and 3D pose as well as the body part segmentation in
intermediate tasks. We generated a synthetic, large-scale dataset of
photo-realistic images of human bodies with a wide range of body shapes and
realistic poses called SURREALvols. By using Volnet and combining multiple
stacked hourglass networks together with ResNeXt, our model correctly predicted
the volume in ~82% of cases with a 10% tolerance threshold. This is a
considerable improvement compared to state-of-the-art solutions such as BodyNet
with only a ~38% success rate.

    

### [[2107.02266] Near-optimal inference in adaptive linear regression](http://arxiv.org/abs/2107.02266)


  When data is collected in an adaptive manner, even simple methods like
ordinary least squares can exhibit non-normal asymptotic behavior. As an
undesirable consequence, hypothesis tests and confidence intervals based on
asymptotic normality can lead to erroneous results. We propose an online
debiasing estimator to correct these distributional anomalies in least squares
estimation. Our proposed method takes advantage of the covariance structure
present in the dataset and provides sharper estimates in directions for which
more information has accrued. We establish an asymptotic normality property for
our proposed online debiasing estimator under mild conditions on the data
collection process, and provide asymptotically exact confidence intervals. We
additionally prove a minimax lower bound for the adaptive linear regression
problem, thereby providing a baseline by which to compare estimators. There are
various conditions under which our proposed estimator achieves the minimax
lower bound up to logarithmic factors. We demonstrate the usefulness of our
theory via applications to multi-armed bandit, autoregressive time series
estimation, and active learning with exploration.

    

### [[2107.02268] Instant One-Shot Word-Learning for Context-Specific Neural Sequence-to-Sequence Speech Recognition](http://arxiv.org/abs/2107.02268)


  Neural sequence-to-sequence systems deliver state-of-the-art performance for
automatic speech recognition (ASR). When using appropriate modeling units,
e.g., byte-pair encoded characters, these systems are in principal open
vocabulary systems. In practice, however, they often fail to recognize words
not seen during training, e.g., named entities, numbers or technical terms. To
alleviate this problem we supplement an end-to-end ASR system with a
word/phrase memory and a mechanism to access this memory to recognize the words
and phrases correctly. After the training of the ASR system, and when it has
already been deployed, a relevant word can be added or subtracted instantly
without the need for further training. In this paper we demonstrate that
through this mechanism our system is able to recognize more than 85% of newly
added words that it previously failed to recognize compared to a strong
baseline.

    

### [[2107.02274] Dueling Bandits with Adversarial Sleeping](http://arxiv.org/abs/2107.02274)


  We introduce the problem of sleeping dueling bandits with stochastic
preferences and adversarial availabilities (DB-SPAA). In almost all dueling
bandit applications, the decision space often changes over time; eg, retail
store management, online shopping, restaurant recommendation, search engine
optimization, etc. Surprisingly, this `sleeping aspect' of dueling bandits has
never been studied in the literature. Like dueling bandits, the goal is to
compete with the best arm by sequentially querying the preference feedback of
item pairs. The non-triviality however results due to the non-stationary item
spaces that allow any arbitrary subsets items to go unavailable every round.
The goal is to find an optimal `no-regret' policy that can identify the best
available item at each round, as opposed to the standard `fixed best-arm regret
objective' of dueling bandits. We first derive an instance-specific lower bound
for DB-SPAA $\Omega( \sum_{i =1}^{K-1}\sum_{j=i+1}^K \frac{\log
T}{\Delta(i,j)})$, where $K$ is the number of items and $\Delta(i,j)$ is the
gap between items $i$ and $j$. This indicates that the sleeping problem with
preference feedback is inherently more difficult than that for classical
multi-armed bandits (MAB). We then propose two algorithms, with near optimal
regret guarantees. Our results are corroborated empirically.

    

### [[2107.02275] Physics-Informed Graph Learning for Robust Fault Location in Distribution Systems](http://arxiv.org/abs/2107.02275)


  The rapid growth of distributed energy resources potentially increases power
grid instability. One promising strategy is to employ data in power grids to
efficiently respond to abnormal events (e.g., faults) by detection and
location. Unfortunately, most existing works lack physical interpretation and
are vulnerable to the practical challenges: sparse observation, insufficient
labeled datasets, and stochastic environment. We propose a physics-informed
graph learning framework of two stages to handle these challenges when locating
faults. Stage- I focuses on informing a graph neural network (GNN) with the
geometrical structure of power grids; stage-II employs the physical similarity
of labeled and unlabeled data samples to improve the location accuracy. We
provide a random walk-based the underpinning of designing our GNNs to address
the challenge of sparse observation and augment the correct prediction
probability. We compare our approach with three baselines in the IEEE 123-node
benchmark system, showing that the proposed method outperforms the others by
significant margins, especially when label rates are low. Also, we validate the
robustness of our algorithms to out-of-distribution-data (ODD) due to topology
changes and load variations. Additionally, we adapt our graph learning
framework to the IEEE 37-node test feeder and show high location performance
with the proposed training strategy.

    

### [[2107.02276] Sarcasm Detection: A Comparative Study](http://arxiv.org/abs/2107.02276)


  Sarcasm detection is the task of identifying irony containing utterances in
sentiment-bearing text. However, the figurative and creative nature of sarcasm
poses a great challenge for affective computing systems performing sentiment
analysis. This article compiles and reviews the salient work in the literature
of automatic sarcasm detection. Thus far, three main paradigm shifts have
occurred in the way researchers have approached this task: 1) semi-supervised
pattern extraction to identify implicit sentiment, 2) use of hashtag-based
supervision, and 3) incorporation of context beyond target text. In this
article, we provide a comprehensive review of the datasets, approaches, trends,
and issues in sarcasm and irony detection.

    

### [[2107.02278] "Garbage In, Garbage Out" Revisited: What Do Machine Learning Application Papers Report About Human-Labeled Training Data?](http://arxiv.org/abs/2107.02278)


  Supervised machine learning, in which models are automatically derived from
labeled training data, is only as good as the quality of that data. This study
builds on prior work that investigated to what extent 'best practices' around
labeling training data were followed in applied ML publications within a single
domain (social media platforms). In this paper, we expand by studying
publications that apply supervised ML in a far broader spectrum of disciplines,
focusing on human-labeled data. We report to what extent a random sample of ML
application papers across disciplines give specific details about whether best
practices were followed, while acknowledging that a greater range of
application fields necessarily produces greater diversity of labeling and
annotation methods. Because much of machine learning research and education
only focuses on what is done once a "ground truth" or "gold standard" of
training data is available, it is especially relevant to discuss issues around
the equally-important aspect of whether such data is reliable in the first
place. This determination becomes increasingly complex when applied to a
variety of specialized fields, as labeling can range from a task requiring
little-to-no background knowledge to one that must be performed by someone with
career expertise.

    

### [[2107.02281] DeepCEL0 for 2D Single Molecule Localization in Fluorescence Microscopy](http://arxiv.org/abs/2107.02281)


  In fluorescence microscopy, Single Molecule Localization Microscopy (SMLM)
techniques aim at localizing with high precision high density fluorescent
molecules by stochastically activating and imaging small subsets of blinking
emitters. Super Resolution (SR) plays an important role in this field since it
allows to go beyond the intrinsic light diffraction limit. In this work, we
propose a deep learning-based algorithm for precise molecule localization of
high density frames acquired by SMLM techniques whose $\ell_{2}$-based loss
function is regularized by positivity and $\ell_{0}$-based constraints. The
$\ell_{0}$ is relaxed through its Continuous Exact $\ell_{0}$ (CEL0)
counterpart. The arising approach, named DeepCEL0, is parameter-free, more
flexible, faster and provides more precise molecule localization maps if
compared to the other state-of-the-art methods. We validate our approach on
both simulated and real fluorescence microscopy data.

    

### [[2107.02283] Clustering Structure of Microstructure Measures](http://arxiv.org/abs/2107.02283)


  This paper builds the clustering model of measures of market microstructure
features which are popular in predicting the stock returns. In a 10-second time
frequency, we study the clustering structure of different measures to find out
the best ones for predicting. In this way, we can predict more accurately with
a limited number of predictors, which removes the noise and makes the model
more interpretable.

    

### [[2107.02287] Morphological Classification of Galaxies in S-PLUS using an Ensemble of Convolutional Networks](http://arxiv.org/abs/2107.02287)


  The universe is composed of galaxies that have diverse shapes. Once the
structure of a galaxy is determined, it is possible to obtain important
information about its formation and evolution. Morphologically classifying
galaxies means cataloging them according to their visual appearance and the
classification is linked to the physical properties of the galaxy. A
morphological classification made through visual inspection is subject to
biases introduced by subjective observations made by human volunteers. For this
reason, systematic, objective and easily reproducible classification of
galaxies has been gaining importance since the astronomer Edwin Hubble created
his famous classification method. In this work, we combine accurate visual
classifications of the Galaxy Zoo project with \emph {Deep Learning} methods.
The goal is to find an efficient technique at human performance level
classification, but in a systematic and automatic way, for classification of
elliptical and spiral galaxies. For this, a neural network model was created
through an Ensemble of four other convolutional models, allowing a greater
accuracy in the classification than what would be obtained with any one
individual. Details of the individual models and improvements made are also
described. The present work is entirely based on the analysis of images (not
parameter tables) from DR1 (this http URL) of the Southern Photometric
Local Universe Survey (S-PLUS). In terms of classification, we achieved, with
the Ensemble, an accuracy of $\approx 99 \%$ in the test sample (using
pre-trained networks).

    

### [[2107.02293] Histogram of Cell Types: Deep Learning for Automated Bone Marrow Cytology](http://arxiv.org/abs/2107.02293)


  Bone marrow cytology is required to make a hematological diagnosis,
influencing critical clinical decision points in hematology. However, bone
marrow cytology is tedious, limited to experienced reference centers and
associated with high inter-observer variability. This may lead to a delayed or
incorrect diagnosis, leaving an unmet need for innovative supporting
technologies. We have developed the first ever end-to-end deep learning-based
technology for automated bone marrow cytology. Starting with a bone marrow
aspirate digital whole slide image, our technology rapidly and automatically
detects suitable regions for cytology, and subsequently identifies and
classifies all bone marrow cells in each region. This collective
cytomorphological information is captured in a novel representation called
Histogram of Cell Types (HCT) quantifying bone marrow cell class probability
distribution and acting as a cytological "patient fingerprint". The approach
achieves high accuracy in region detection (0.97 accuracy and 0.99 ROC AUC),
and cell detection and cell classification (0.75 mAP, 0.78 F1-score,
Log-average miss rate of 0.31). HCT has potential to revolutionize
hematopathology diagnostic workflows, leading to more cost-effective, accurate
diagnosis and opening the door to precision medicine.

    

### [[2107.02295] A Review of Explainable Artificial Intelligence in Manufacturing](http://arxiv.org/abs/2107.02295)


  The implementation of Artificial Intelligence (AI) systems in the
manufacturing domain enables higher production efficiency, outstanding
performance, and safer operations, leveraging powerful tools such as deep
learning and reinforcement learning techniques. Despite the high accuracy of
these models, they are mostly considered black boxes: they are unintelligible
to the human. Opaqueness affects trust in the system, a factor that is critical
in the context of decision-making. We present an overview of Explainable
Artificial Intelligence (XAI) techniques as a means of boosting the
transparency of models. We analyze different metrics to evaluate these
techniques and describe several application scenarios in the manufacturing
domain.

    

### [[2107.02306] Connectivity Matters: Neural Network Pruning Through the Lens of Effective Sparsity](http://arxiv.org/abs/2107.02306)


  Neural network pruning is a fruitful area of research with surging interest
in high sparsity regimes. Benchmarking in this domain heavily relies on
faithful representation of the sparsity of subnetworks, which has been
traditionally computed as the fraction of removed connections (direct
sparsity). This definition, however, fails to recognize unpruned parameters
that detached from input or output layers of underlying subnetworks,
potentially underestimating actual effective sparsity: the fraction of
inactivated connections. While this effect might be negligible for moderately
pruned networks (up to 10-100 compression rates), we find that it plays an
increasing role for thinner subnetworks, greatly distorting comparison between
different pruning algorithms. For example, we show that effective compression
of a randomly pruned LeNet-300-100 can be orders of magnitude larger than its
direct counterpart, while no discrepancy is ever observed when using SynFlow
for pruning [Tanaka et al., 2020]. In this work, we adopt the lens of effective
sparsity to reevaluate several recent pruning algorithms on common benchmark
architectures (e.g., LeNet-300-100, VGG-19, ResNet-18) and discover that their
absolute and relative performance changes dramatically in this new and more
appropriate framework. To aim for effective, rather than direct, sparsity, we
develop a low-cost extension to most pruning algorithms. Further, equipped with
effective sparsity as a reference frame, we partially reconfirm that random
pruning with appropriate sparsity allocation across layers performs as well or
better than more sophisticated algorithms for pruning at initialization [Su et
al., 2020]. In response to this observation, using a simple analogy of pressure
distribution in coupled cylinders from physics, we design novel layerwise
sparsity quotas that outperform all existing baselines in the context of random
pruning.

    

### [[2107.02308] A visual introduction to Gaussian Belief Propagation](http://arxiv.org/abs/2107.02308)


  In this article, we present a visual introduction to Gaussian Belief
Propagation (GBP), an approximate probabilistic inference algorithm that
operates by passing messages between the nodes of arbitrarily structured factor
graphs. A special case of loopy belief propagation, GBP updates rely only on
local information and will converge independently of the message schedule. Our
key argument is that, given recent trends in computing hardware, GBP has the
right computational properties to act as a scalable distributed probabilistic
inference framework for future machine learning systems.

    

### [[2107.02320] Memory-Sample Lower Bounds for Learning Parity with Noise](http://arxiv.org/abs/2107.02320)


  In this work, we show, for the well-studied problem of learning parity under
noise, where a learner tries to learn $x=(x_1,\ldots,x_n) \in \{0,1\}^n$ from a
stream of random linear equations over $\mathrm{F}_2$ that are correct with
probability $\frac{1}{2}+\varepsilon$ and flipped with probability
$\frac{1}{2}-\varepsilon$, that any learning algorithm requires either a memory
of size $\Omega(n^2/\varepsilon)$ or an exponential number of samples.
In fact, we study memory-sample lower bounds for a large class of learning
problems, as characterized by [GRT'18], when the samples are noisy. A matrix
$M: A \times X \rightarrow \{-1,1\}$ corresponds to the following learning
problem with error parameter $\varepsilon$: an unknown element $x \in X$ is
chosen uniformly at random. A learner tries to learn $x$ from a stream of
samples, $(a_1, b_1), (a_2, b_2) \ldots$, where for every $i$, $a_i \in A$ is
chosen uniformly at random and $b_i = M(a_i,x)$ with probability
$1/2+\varepsilon$ and $b_i = -M(a_i,x)$ with probability $1/2-\varepsilon$
($0<\varepsilon< \frac{1}{2}$). Assume that $k,\ell, r$ are such that any
submatrix of $M$ of at least $2^{-k} \cdot |A|$ rows and at least $2^{-\ell}
\cdot |X|$ columns, has a bias of at most $2^{-r}$. We show that any learning
algorithm for the learning problem corresponding to $M$, with error, requires
either a memory of size at least $\Omega\left(\frac{k \cdot \ell}{\varepsilon}
\right)$, or at least $2^{\Omega(r)}$ samples. In particular, this shows that
for a large class of learning problems, same as those in [GRT'18], any learning
algorithm requires either a memory of size at least $\Omega\left(\frac{(\log
|X|) \cdot (\log |A|)}{\varepsilon}\right)$ or an exponential number of noisy
samples.
Our proof is based on adapting the arguments in [Raz'17,GRT'18] to the noisy
case.

    

### [[2107.02331] Mind Your Outliers! Investigating the Negative Impact of Outliers on Active Learning for Visual Question Answering](http://arxiv.org/abs/2107.02331)


  Active learning promises to alleviate the massive data needs of supervised
machine learning: it has successfully improved sample efficiency by an order of
magnitude on traditional tasks like topic classification and object
recognition. However, we uncover a striking contrast to this promise: across 5
models and 4 datasets on the task of visual question answering, a wide variety
of active learning approaches fail to outperform random selection. To
understand this discrepancy, we profile 8 active learning methods on a
per-example basis, and identify the problem as collective outliers -- groups of
examples that active learning methods prefer to acquire but models fail to
learn (e.g., questions that ask about text in images or require external
knowledge). Through systematic ablation experiments and qualitative
visualizations, we verify that collective outliers are a general phenomenon
responsible for degrading pool-based active learning. Notably, we show that
active learning sample efficiency increases significantly as the number of
collective outliers in the active learning pool decreases. We conclude with a
discussion and prescriptive recommendations for mitigating the effects of these
outliers in future work.

    

### [[2107.02339] Multi-Modal Mutual Information (MuMMI) Training for Robust Self-Supervised Deep Reinforcement Learning](http://arxiv.org/abs/2107.02339)


  This work focuses on learning useful and robust deep world models using
multiple, possibly unreliable, sensors. We find that current methods do not
sufficiently encourage a shared representation between modalities; this can
cause poor performance on downstream tasks and over-reliance on specific
sensors. As a solution, we contribute a new multi-modal deep latent state-space
model, trained using a mutual information lower-bound. The key innovation is a
specially-designed density ratio estimator that encourages consistency between
the latent codes of each modality. We tasked our method to learn policies (in a
self-supervised manner) on multi-modal Natural MuJoCo benchmarks and a
challenging Table Wiping task. Experiments show our method significantly
outperforms state-of-the-art deep reinforcement learning methods, particularly
in the presence of missing observations.

    

### [[2107.02345] Domain Adaptation via CycleGAN for Retina Segmentation in Optical Coherence Tomography](http://arxiv.org/abs/2107.02345)


  With the FDA approval of Artificial Intelligence (AI) for point-of-care
clinical diagnoses, model generalizability is of the utmost importance as
clinical decision-making must be domain-agnostic. A method of tackling the
problem is to increase the dataset to include images from a multitude of
domains; while this technique is ideal, the security requirements of medical
data is a major limitation. Additionally, researchers with developed tools
benefit from the addition of open-sourced data, but are limited by the
difference in domains. Herewith, we investigated the implementation of a
Cycle-Consistent Generative Adversarial Networks (CycleGAN) for the domain
adaptation of Optical Coherence Tomography (OCT) volumes. This study was done
in collaboration with the Biomedical Optics Research Group and Functional &
Anatomical Imaging & Shape Analysis Lab at Simon Fraser University. In this
study, we investigated a learning-based approach of adapting the domain of a
publicly available dataset, UK Biobank dataset (UKB). To evaluate the
performance of domain adaptation, we utilized pre-existing retinal layer
segmentation tools developed on a different set of RETOUCH OCT data. This study
provides insight on state-of-the-art tools for domain adaptation compared to
traditional processing techniques as well as a pipeline for adapting publicly
available retinal data to the domains previously used by our collaborators.

    

### [[2107.02347] An Ensemble Noise-Robust K-fold Cross-Validation Selection Method for Noisy Labels](http://arxiv.org/abs/2107.02347)


  We consider the problem of training robust and accurate deep neural networks
(DNNs) when subject to various proportions of noisy labels. Large-scale
datasets tend to contain mislabeled samples that can be memorized by DNNs,
impeding the performance. With appropriate handling, this degradation can be
alleviated. There are two problems to consider: how to distinguish clean
samples and how to deal with noisy samples. In this paper, we present Ensemble
Noise-robust K-fold Cross-Validation Selection (E-NKCVS) to effectively select
clean samples from noisy data, solving the first problem. For the second
problem, we create a new pseudo label for any sample determined to have an
uncertain or likely corrupt label. E-NKCVS obtains multiple predicted labels
for each sample and the entropy of these labels is used to tune the weight
given to the pseudo label and the given label. Theoretical analysis and
extensive verification of the algorithms in the noisy label setting are
provided. We evaluate our approach on various image and text classification
tasks where the labels have been manually corrupted with different noise
ratios. Additionally, two large real-world noisy datasets are also used,
Clothing-1M and WebVision. E-NKCVS is empirically shown to be highly tolerant
to considerable proportions of label noise and has a consistent improvement
over state-of-the-art methods. Especially on more difficult datasets with
higher noise ratios, we can achieve a significant improvement over the
second-best model. Moreover, our proposed approach can easily be integrated
into existing DNN methods to improve their robustness against label noise.

    

### [[2107.02349] Physical Interaction as Communication: Learning Robot Objectives Online from Human Corrections](http://arxiv.org/abs/2107.02349)


  When a robot performs a task next to a human, physical interaction is
inevitable: the human might push, pull, twist, or guide the robot. The
state-of-the-art treats these interactions as disturbances that the robot
should reject or avoid. At best, these robots respond safely while the human
interacts; but after the human lets go, these robots simply return to their
original behavior. We recognize that physical human-robot interaction (pHRI) is
often intentional -- the human intervenes on purpose because the robot is not
doing the task correctly. In this paper, we argue that when pHRI is intentional
it is also informative: the robot can leverage interactions to learn how it
should complete the rest of its current task even after the person lets go. We
formalize pHRI as a dynamical system, where the human has in mind an objective
function they want the robot to optimize, but the robot does not get direct
access to the parameters of this objective -- they are internal to the human.
Within our proposed framework human interactions become observations about the
true objective. We introduce approximations to learn from and respond to pHRI
in real-time. We recognize that not all human corrections are perfect: often
users interact with the robot noisily, and so we improve the efficiency of
robot learning from pHRI by reducing unintended learning. Finally, we conduct
simulations and user studies on a robotic manipulator to compare our proposed
approach to the state-of-the-art. Our results indicate that learning from pHRI
leads to better task performance and improved human satisfaction.

    

### [[2107.02355] Total Nitrogen Estimation in Agricultural Soils via Aerial Multispectral Imaging and LIBS](http://arxiv.org/abs/2107.02355)


  Measuring soil health indicators is an important and challenging task that
affects farmers' decisions on timing, placement, and quantity of fertilizers
applied in the farms. Most existing methods to measure soil health indicators
(SHIs) are in-lab wet chemistry or spectroscopy-based methods, which require
significant human input and effort, time-consuming, costly, and are
low-throughput in nature. To address this challenge, we develop an artificial
intelligence (AI)-driven near real-time unmanned aerial vehicle (UAV)-based
multispectral sensing (UMS) solution to estimate total nitrogen (TN) of the
soil, an important macro-nutrient or SHI that directly affects the crop health.
Accurate prediction of soil TN can significantly increase crop yield through
informed decision making on the timing of seed planting, and fertilizer
quantity and timing. We train two machine learning models including multi-layer
perceptron and support vector machine to predict the soil nitrogen using a
suite of data classes including multispectral characteristics of the soil and
crops in red, near-infrared, and green spectral bands, computed vegetation
indices, and environmental variables including air temperature and relative
humidity. To generate the ground-truth data or the training data for the
machine learning models, we measure the total nitrogen of the soil samples
(collected from a farm) using laser-induced breakdown spectroscopy (LIBS).

    

### [[2107.02358] Impact of On-Chip Interconnect on In-Memory Acceleration of Deep Neural Networks](http://arxiv.org/abs/2107.02358)


  With the widespread use of Deep Neural Networks (DNNs), machine learning
algorithms have evolved in two diverse directions -- one with ever-increasing
connection density for better accuracy and the other with more compact sizing
for energy efficiency. The increase in connection density increases on-chip
data movement, which makes efficient on-chip communication a critical function
of the DNN accelerator. The contribution of this work is threefold. First, we
illustrate that the point-to-point (P2P)-based interconnect is incapable of
handling a high volume of on-chip data movement for DNNs. Second, we evaluate
P2P and network-on-chip (NoC) interconnect (with a regular topology such as a
mesh) for SRAM- and ReRAM-based in-memory computing (IMC) architectures for a
range of DNNs. This analysis shows the necessity for the optimal interconnect
choice for an IMC DNN accelerator. Finally, we perform an experimental
evaluation for different DNNs to empirically obtain the performance of the IMC
architecture with both NoC-tree and NoC-mesh. We conclude that, at the tile
level, NoC-tree is appropriate for compact DNNs employed at the edge, and
NoC-mesh is necessary to accelerate DNNs with high connection density.
Furthermore, we propose a technique to determine the optimal choice of
interconnect for any given DNN. In this technique, we use analytical models of
NoC to evaluate end-to-end communication latency of any given DNN. We
demonstrate that the interconnect optimization in the IMC architecture results
in up to 6$\times$ improvement in energy-delay-area product for VGG-19
inference compared to the state-of-the-art ReRAM-based IMC architectures.

    

### [[2107.02359] Leveraging Clinical Context for User-Centered Explainability: A Diabetes Use Case](http://arxiv.org/abs/2107.02359)


  Academic advances of AI models in high-precision domains, like healthcare,
need to be made explainable in order to enhance real-world adoption. Our past
studies and ongoing interactions indicate that medical experts can use AI
systems with greater trust if there are ways to connect the model inferences
about patients to explanations that are tied back to the context of use.
Specifically, risk prediction is a complex problem of diagnostic and
interventional importance to clinicians wherein they consult different sources
to make decisions. To enable the adoption of the ever improving AI risk
prediction models in practice, we have begun to explore techniques to
contextualize such models along three dimensions of interest: the patients'
clinical state, AI predictions about their risk of complications, and
algorithmic explanations supporting the predictions. We validate the importance
of these dimensions by implementing a proof-of-concept (POC) in type-2 diabetes
(T2DM) use case where we assess the risk of chronic kidney disease (CKD) - a
common T2DM comorbidity. Within the POC, we include risk prediction models for
CKD, post-hoc explainers of the predictions, and other natural-language modules
which operationalize domain knowledge and CPGs to provide context. With primary
care physicians (PCP) as our end-users, we present our initial results and
clinician feedback in this paper. Our POC approach covers multiple knowledge
sources and clinical scenarios, blends knowledge to explain data and
predictions to PCPs, and received an enthusiastic response from our medical
expert.

    

### [[2107.02361] Effects of Smart Traffic Signal Control on Air Quality](http://arxiv.org/abs/2107.02361)


  Adaptive traffic signal control (ATSC) in urban traffic networks poses a
challenging task due to the complicated dynamics arising in traffic systems. In
recent years, several approaches based on multi-agent deep reinforcement
learning (MARL) have been studied experimentally. These approaches propose
distributed techniques in which each signalized intersection is seen as an
agent in a stochastic game whose purpose is to optimize the flow of vehicles in
its vicinity. In this setting, the systems evolves towards an equilibrium among
the agents that shows beneficial for the whole traffic network. A recently
developed multi-agent variant of the well-established advantage actor-critic
(A2C) algorithm, called MA2C (multi-agent A2C) exploits the promising idea of
some communication among the agents. In this view,the agents share their
strategies with other neighbor agents, thereby stabilizing the learning process
even when the agents grow in number and variety. We experimented MA2C in two
traffic networks located in Bologna (Italy) and found that its action
translates into a significant decrease of the amount of pollutants released
into the environment.

    

### [[2107.02363] Asymptotics of Network Embeddings Learned via Subsampling](http://arxiv.org/abs/2107.02363)


  Network data are ubiquitous in modern machine learning, with tasks of
interest including node classification, node clustering and link prediction. A
frequent approach begins by learning an Euclidean embedding of the network, to
which algorithms developed for vector-valued data are applied. For large
networks, embeddings are learned using stochastic gradient methods where the
sub-sampling scheme can be freely chosen. Despite the strong empirical
performance of such methods, they are not well understood theoretically. Our
work encapsulates representation methods using a subsampling approach, such as
node2vec, into a single unifying framework. We prove, under the assumption that
the graph is exchangeable, that the distribution of the learned embedding
vectors asymptotically decouples. Moreover, we characterize the asymptotic
distribution and provided rates of convergence, in terms of the latent
parameters, which includes the choice of loss function and the embedding
dimension. This provides a theoretical foundation to understand what the
embedding vectors represent and how well these methods perform on downstream
tasks. Notably, we observe that typically used loss functions may lead to
shortcomings, such as a lack of Fisher consistency.

    

### [[2107.02367] Discrete-Valued Neural Communication](http://arxiv.org/abs/2107.02367)


  Deep learning has advanced from fully connected architectures to structured
models organized into components, e.g., the transformer composed of positional
elements, modular architectures divided into slots, and graph neural nets made
up of nodes. In structured models, an interesting question is how to conduct
dynamic and possibly sparse communication among the separate components. Here,
we explore the hypothesis that restricting the transmitted information among
components to discrete representations is a beneficial bottleneck. The
motivating intuition is human language in which communication occurs through
discrete symbols. Even though individuals have different understandings of what
a ``"cat" is based on their specific experiences, the shared discrete token
makes it possible for communication among individuals to be unimpeded by
individual differences in internal representation. To discretize the values of
concepts dynamically communicated among specialist components, we extend the
quantization mechanism from the Vector-Quantized Variational Autoencoder to
multi-headed discretization with shared codebooks and use it for
discrete-valued neural communication (DVNC). Our experiments show that DVNC
substantially improves systematic generalization in a variety of architectures
-- transformers, modular architectures, and graph neural networks. We also show
that the DVNC is robust to the choice of hyperparameters, making the method
very useful in practice. Moreover, we establish a theoretical justification of
our discretization process, proving that it has the ability to increase noise
robustness and reduce the underlying dimensionality of the model.

    

### [[2107.02371] Weighted Gaussian Process Bandits for Non-stationary Environments](http://arxiv.org/abs/2107.02371)


  In this paper, we consider the Gaussian process (GP) bandit optimization
problem in a non-stationary environment. To capture external changes, the
black-box function is allowed to be time-varying within a reproducing kernel
Hilbert space (RKHS). To this end, we develop WGP-UCB, a novel UCB-type
algorithm based on weighted Gaussian process regression. A key challenge is how
to cope with infinite-dimensional feature maps. To that end, we leverage kernel
approximation techniques to prove a sublinear regret bound, which is the first
(frequentist) sublinear regret guarantee on weighted time-varying bandits with
general nonlinear rewards. This result generalizes both non-stationary linear
bandits and standard GP-UCB algorithms. Further, a novel concentration
inequality is achieved for weighted Gaussian process regression with general
weights. We also provide universal upper bounds and weight-dependent upper
bounds for weighted maximum information gains. These results are potentially of
independent interest for applications such as news ranking and adaptive
pricing, where weights can be adopted to capture the importance or quality of
data. Finally, we conduct experiments to highlight the favorable gains of the
proposed algorithm in many cases when compared to existing methods.

    

### [[2107.02375] SplitAVG: A heterogeneity-aware federated deep learning method for medical imaging](http://arxiv.org/abs/2107.02375)


  Federated learning is an emerging research paradigm for enabling
collaboratively training deep learning models without sharing patient data.
However, the data from different institutions are usually heterogeneous across
institutions, which may reduce the performance of models trained using
federated learning. In this study, we propose a novel heterogeneity-aware
federated learning method, SplitAVG, to overcome the performance drops from
data heterogeneity in federated learning. Unlike previous federated methods
that require complex heuristic training or hyper parameter tuning, our SplitAVG
leverages the simple network split and feature map concatenation strategies to
encourage the federated model training an unbiased estimator of the target data
distribution. We compare SplitAVG with seven state-of-the-art federated
learning methods, using centrally hosted training data as the baseline on a
suite of both synthetic and real-world federated datasets. We find that the
performance of models trained using all the comparison federated learning
methods degraded significantly with the increasing degrees of data
heterogeneity. In contrast, SplitAVG method achieves comparable results to the
baseline method under all heterogeneous settings, that it achieves 96.2% of the
accuracy and 110.4% of the mean absolute error obtained by the baseline in a
diabetic retinopathy binary classification dataset and a bone age prediction
dataset, respectively, on highly heterogeneous data partitions. We conclude
that SplitAVG method can effectively overcome the performance drops from
variability in data distributions across institutions. Experimental results
also show that SplitAVG can be adapted to different base networks and
generalized to various types of medical imaging tasks.

    

### [[2107.02377] A Short Note on the Relationship of Information Gain and Eluder Dimension](http://arxiv.org/abs/2107.02377)


  Eluder dimension and information gain are two widely used methods of
complexity measures in bandit and reinforcement learning. Eluder dimension was
originally proposed as a general complexity measure of function classes, but
the common examples of where it is known to be small are function spaces
(vector spaces). In these cases, the primary tool to upper bound the eluder
dimension is the elliptic potential lemma. Interestingly, the elliptic
potential lemma also features prominently in the analysis of linear
bandits/reinforcement learning and their nonparametric generalization, the
information gain. We show that this is not a coincidence -- eluder dimension
and information gain are equivalent in a precise sense for reproducing kernel
Hilbert spaces.

    

### [[2107.02378] Learning an Explicit Hyperparameter Prediction Policy Conditioned on Tasks](http://arxiv.org/abs/2107.02378)


  Meta learning has attracted much attention recently in machine learning
community. Contrary to conventional machine learning aiming to learn inherent
prediction rules to predict labels for new query data, meta learning aims to
learn the learning methodology for machine learning from observed tasks, so as
to generalize to new query tasks by leveraging the meta-learned learning
methodology. In this study, we interpret such learning methodology as learning
an explicit hyperparameter prediction policy shared by all training tasks.
Specifically, this policy is represented as a parameterized function called
meta-learner, mapping from a training/test task to its suitable hyperparameter
setting, extracted from a pre-specified function set called meta learning
machine. Such setting guarantees that the meta-learned learning methodology is
able to flexibly fit diverse query tasks, instead of only obtaining fixed
hyperparameters by many current meta learning methods, with less adaptability
to query task's variations. Such understanding of meta learning also makes it
easily succeed from traditional learning theory for analyzing its
generalization bounds with general losses/tasks/models. The theory naturally
leads to some feasible controlling strategies for ameliorating the quality of
the extracted meta-learner, verified to be able to finely ameliorate its
generalization capability in some typical meta learning applications, including
few-shot regression, few-shot classification and domain generalization.

    

### [[2107.02381] An Inverse QSAR Method Based on Linear Regression and Integer Programming](http://arxiv.org/abs/2107.02381)


  Recently a novel framework has been proposed for designing the molecular
structure of chemical compounds using both artificial neural networks (ANNs)
and mixed integer linear programming (MILP). In the framework, we first define
a feature vector $f(C)$ of a chemical graph $C$ and construct an ANN that maps
$x=f(C)$ to a predicted value $\eta(x)$ of a chemical property $\pi$ to $C$.
After this, we formulate an MILP that simulates the computation process of
$f(C)$ from $C$ and that of $\eta(x)$ from $x$. Given a target value $y^*$ of
the chemical property $\pi$, we infer a chemical graph $C^\dagger$ such that
$\eta(f(C^\dagger))=y^*$ by solving the MILP. In this paper, we use linear
regression to construct a prediction function $\eta$ instead of ANNs. For this,
we derive an MILP formulation that simulates the computation process of a
prediction function by linear regression. The results of computational
experiments suggest our method can infer chemical graphs with around up to 50
non-hydrogen atoms.

    

### [[2107.02388] CAP-RAM: A Charge-Domain In-Memory Computing 6T-SRAM for Accurate and Precision-Programmable CNN Inference](http://arxiv.org/abs/2107.02388)


  A compact, accurate, and bitwidth-programmable in-memory computing (IMC)
static random-access memory (SRAM) macro, named CAP-RAM, is presented for
energy-efficient convolutional neural network (CNN) inference. It leverages a
novel charge-domain multiply-and-accumulate (MAC) mechanism and circuitry to
achieve superior linearity under process variations compared to conventional
IMC designs. The adopted semi-parallel architecture efficiently stores filters
from multiple CNN layers by sharing eight standard 6T SRAM cells with one
charge-domain MAC circuit. Moreover, up to six levels of bit-width of weights
with two encoding schemes and eight levels of input activations are supported.
A 7-bit charge-injection SAR (ciSAR) analog-to-digital converter (ADC) getting
rid of sample and hold (S&H) and input/reference buffers further improves the
overall energy efficiency and throughput. A 65-nm prototype validates the
excellent linearity and computing accuracy of CAP-RAM. A single 512x128 macro
stores a complete pruned and quantized CNN model to achieve 98.8% inference
accuracy on the MNIST data set and 89.0% on the CIFAR-10 data set, with a
573.4-giga operations per second (GOPS) peak throughput and a 49.4-tera
operations per second (TOPS)/W energy efficiency.

    

### [[2107.02392] Dirichlet Energy Constrained Learning for Deep Graph Neural Networks](http://arxiv.org/abs/2107.02392)


  Graph neural networks (GNNs) integrate deep architectures and topological
structure modeling in an effective way. However, the performance of existing
GNNs would decrease significantly when they stack many layers, because of the
over-smoothing issue. Node embeddings tend to converge to similar vectors when
GNNs keep recursively aggregating the representations of neighbors. To enable
deep GNNs, several methods have been explored recently. But they are developed
from either techniques in convolutional neural networks or heuristic
strategies. There is no generalizable and theoretical principle to guide the
design of deep GNNs. To this end, we analyze the bottleneck of deep GNNs by
leveraging the Dirichlet energy of node embeddings, and propose a generalizable
principle to guide the training of deep GNNs. Based on it, a novel deep GNN
framework -- EGNN is designed. It could provide lower and upper constraints in
terms of Dirichlet energy at each layer to avoid over-smoothing. Experimental
results demonstrate that EGNN achieves state-of-the-art performance by using
deep layers.

    

### [[2107.02397] Deep Network Approximation With Accuracy Independent of Number of Neurons](http://arxiv.org/abs/2107.02397)


  This paper develops simple feed-forward neural networks that achieve the
universal approximation property for all continuous functions with a fixed
finite number of neurons. These neural networks are simple because they are
designed with a simple and computable continuous activation function $\sigma$
leveraging a triangular-wave function and a softsign function. We prove that
$\sigma$-activated networks with width $36d(2d+1)$ and depth $11$ can
approximate any continuous function on a $d$-dimensioanl hypercube within an
arbitrarily small error. Hence, for supervised learning and its related
regression problems, the hypothesis space generated by these networks with a
size not smaller than $36d(2d+1)\times 11$ is dense in the space of continuous
functions. Furthermore, classification functions arising from image and signal
classification are in the hypothesis space generated by $\sigma$-activated
networks with width $36d(2d+1)$ and depth $12$, when there exist pairwise
disjoint closed bounded subsets of $\mathbb{R}^d$ such that the samples of the
same class are located in the same subset.

    

### [[2107.02408] CoReD: Generalizing Fake Media Detection with Continual Representation using Distillation](http://arxiv.org/abs/2107.02408)


  Over the last few decades, artificial intelligence research has made
tremendous strides, but it still heavily relies on fixed datasets in stationary
environments. Continual learning is a growing field of research that examines
how AI systems can learn sequentially from a continuous stream of linked data
in the same way that biological systems do. Simultaneously, fake media such as
deepfakes and synthetic face images have emerged as significant to current
multimedia technologies. Recently, numerous method has been proposed which can
detect deepfakes with high accuracy. However, they suffer significantly due to
their reliance on fixed datasets in limited evaluation settings. Therefore, in
this work, we apply continuous learning to neural networks' learning dynamics,
emphasizing its potential to increase data efficiency significantly. We propose
Continual Representation using Distillation (CoReD) method that employs the
concept of Continual Learning (CoL), Representation Learning (ReL), and
Knowledge Distillation (KD). We design CoReD to perform sequential domain
adaptation tasks on new deepfake and GAN-generated synthetic face datasets,
while effectively minimizing the catastrophic forgetting in a teacher-student
model setting. Our extensive experimental results demonstrate that our method
is efficient at domain adaptation to detect low-quality deepfakes videos and
GAN-generated images from several datasets, outperforming the-state-of-art
baseline methods.

    

### [[2107.02415] Deep Visual Attention-Based Transfer Clustering](http://arxiv.org/abs/2107.02415)


  In this paper, we propose a methodology to improvise the technique of deep
transfer clustering (DTC) when applied to the less variant data distribution.
Clustering can be considered as the most important unsupervised learning
problem. A simple definition of clustering can be stated as "the process of
organizing objects into groups, whose members are similar in some way". Image
clustering is a crucial but challenging task in the domain machine learning and
computer vision. We have discussed the clustering of the data collection where
the data is less variant. We have discussed the improvement by using
attention-based classifiers rather than regular classifiers as the initial
feature extractors in the deep transfer clustering. We have enforced the model
to learn only the required region of interest in the images to get the
differentiable and robust features that do not take into account the
background. This paper is the improvement of the existing deep transfer
clustering for less variant data distribution.

    

### [[2107.02416] Enhanced Universal Dependency Parsing with Automated Concatenation of Embeddings](http://arxiv.org/abs/2107.02416)


  This paper describes the system used in submission from SHANGHAITECH team to
the IWPT 2021 Shared Task. Our system is a graph-based parser with the
technique of Automated Concatenation of Embeddings (ACE). Because recent work
found that better word representations can be obtained by concatenating
different types of embeddings, we use ACE to automatically find the better
concatenation of embeddings for the task of enhanced universal dependencies.
According to official results averaged on 17 languages, our system ranks 2nd
over 9 teams.

    

### [[2107.02422] Equivariant bifurcation, quadratic equivariants, and symmetry breaking for the standard representation of $S_n$](http://arxiv.org/abs/2107.02422)


  Motivated by questions originating from the study of a class of shallow
student-teacher neural networks, methods are developed for the analysis of
spurious minima in classes of gradient equivariant dynamics related to neural
nets. In the symmetric case, methods depend on the generic equivariant
bifurcation theory of irreducible representations of the symmetric group on $n$
symbols, $S_n$; in particular, the standard representation of $S_n$. It is
shown that spurious minima do not arise from spontaneous symmetry breaking but
rather through a complex deformation of the landscape geometry that can be
encoded by a generic $S_n$-equivariant bifurcation. We describe minimal models
for forced symmetry breaking that give a lower bound on the dynamic complexity
involved in the creation of spurious minima when there is no symmetry. Results
on generic bifurcation when there are quadratic equivariants are also proved;
this work extends and clarifies results of Ihrig & Golubitsky and Chossat,
Lauterback & Melbourne on the instability of solutions when there are quadratic
equivariants.

    

### [[2107.02423] Improving Text-to-Image Synthesis Using Contrastive Learning](http://arxiv.org/abs/2107.02423)


  The goal of text-to-image synthesis is to generate a visually realistic image
that matches a given text description. In practice, the captions annotated by
humans for the same image have large variance in terms of contents and the
choice of words. The linguistic discrepancy between the captions of the
identical image leads to the synthetic images deviating from the ground truth.
To address this issue, we propose a contrastive learning approach to improve
the quality and enhance the semantic consistency of synthetic images. In the
pre-training stage, we utilize the contrastive learning approach to learn the
consistent textual representations for the captions corresponding to the same
image. Furthermore, in the following stage of GAN training, we employ the
contrastive learning method to enhance the consistency between the generated
images from the captions related to the same image. We evaluate our approach
over two popular text-to-image synthesis models, AttnGAN and DM-GAN, on
datasets CUB and COCO, respectively. Experimental results have shown that our
approach can effectively improve the quality of synthetic images in terms of
three metrics: IS, FID and R-precision. Especially, on the challenging COCO
dataset, our approach boosts the FID significantly by 29.60% over AttnGAn and
by 21.96% over DM-GAN.

    

### [[2107.02425] GradDiv: Adversarial Robustness of Randomized Neural Networks via Gradient Diversity Regularization](http://arxiv.org/abs/2107.02425)


  Deep learning is vulnerable to adversarial examples. Many defenses based on
randomized neural networks have been proposed to solve the problem, but fail to
achieve robustness against attacks using proxy gradients such as the
Expectation over Transformation (EOT) attack. We investigate the effect of the
adversarial attacks using proxy gradients on randomized neural networks and
demonstrate that it highly relies on the directional distribution of the loss
gradients of the randomized neural network. We show in particular that proxy
gradients are less effective when the gradients are more scattered. To this
end, we propose Gradient Diversity (GradDiv) regularizations that minimize the
concentration of the gradients to build a robust randomized neural network. Our
experiments on MNIST, CIFAR10, and STL10 show that our proposed GradDiv
regularizations improve the adversarial robustness of randomized neural
networks against a variety of state-of-the-art attack methods. Moreover, our
method efficiently reduces the transferability among sample models of
randomized neural networks.

    

### [[2107.02427] Dynamical System Parameter Identification using Deep Recurrent Cell Networks](http://arxiv.org/abs/2107.02427)


  In this paper, we investigate the parameter identification problem in
dynamical systems through a deep learning approach. Focusing mainly on
second-order, linear time-invariant dynamical systems, the topic of damping
factor identification is studied. By utilizing a six-layer deep neural network
with different recurrent cells, namely GRUs, LSTMs or BiLSTMs; and by feeding
input-output sequence pairs captured from a dynamical system simulator, we
search for an effective deep recurrent architecture in order to resolve damping
factor identification problem. Our study results show that, although previously
not utilized for this task in the literature, bidirectional gated recurrent
cells (BiLSTMs) provide better parameter identification results when compared
to unidirectional gated recurrent memory cells such as GRUs and LSTM. Thus,
indicating that an input-output sequence pair of finite length, collected from
a dynamical system and when observed anachronistically, may carry information
in both time directions for prediction of a dynamical systems parameter.

    

### [[2107.02431] Bayesian Nonparametric Modelling for Model-Free Reinforcement Learning in LTE-LAA and Wi-Fi Coexistence](http://arxiv.org/abs/2107.02431)


  With the arrival of next generation wireless communication, a growing number
of new applications like internet of things, autonomous driving systems, and
drone are crowding the unlicensed spectrum. Licensed network such as the
long-term evolution (LTE) also comes to the unlicensed spectrum for better
providing high-capacity contents with low cost. However, LTE was not designed
to share resources with others. Previous solutions usually work on fixed
scenarios. This work features a Nonparametric Bayesian reinforcement learning
algorithm to cope with the coexistence between Wi-Fi and LTE licensed assisted
access (LTE-LAA) agents in 5 GHz unlicensed spectrum. The coexistence problem
is modeled as a decentralized partially-observable Markov decision process
(Dec-POMDP) and Bayesian inference is adopted for policy learning with
nonparametric prior to accommodate the uncertainty of policy for different
agents. A fairness measure is introduced in the reward function to encourage
fair sharing between agents. Variational inference for posterior model
approximation is considered to make the algorithm computationally efficient.
Simulation results demonstrate that this algorithm can reach high value with
compact policy representations in few learning iterations.

    

### [[2107.02438] Shell Language Processing: Unix command parsing for Machine Learning](http://arxiv.org/abs/2107.02438)


  In this article, we present a Shell Language Preprocessing (SLP) library,
which implements tokenization and encoding directed on the parsing of Unix and
Linux shell commands. We describe the rationale behind the need for a new
approach with specific examples when conventional Natural Language Processing
(NLP) pipelines fail. Furthermore, we evaluate our methodology on a security
classification task against widely accepted information and communications
technology (ICT) tokenization techniques and achieve significant improvement of
an F1-score from 0.392 to 0.874.

    

### [[2107.02442] Early Recognition of Ball Catching Success in Clinical Trials with RNN-Based Predictive Classification](http://arxiv.org/abs/2107.02442)


  Motor disturbances can affect the interaction with dynamic objects, such as
catching a ball. A classification of clinical catching trials might give
insight into the existence of pathological alterations in the relation of arm
and ball movements. Accurate, but also early decisions are required to classify
a catching attempt before the catcher's first ball contact. To obtain
clinically valuable results, a significant decision confidence of at least 75%
is required. Hence, three competing objectives have to be optimized at the same
time: accuracy, earliness and decision-making confidence. Here we propose a
coupled classification and prediction approach for early time series
classification: a predictive, generative recurrent neural network (RNN)
forecasts the next data points of ball trajectories based on already available
observations; a discriminative RNN continuously generates classification
guesses based on the available data points and the unrolled sequence
predictions. We compare our approach, which we refer to as predictive
sequential classification (PSC), to state-of-the-art sequence learners,
including various RNN and temporal convolutional network (TCN) architectures.
On this hard real-world task we can consistently demonstrate the superiority of
PSC over all other models in terms of accuracy and confidence with respect to
earliness of recognition. Specifically, PSC is able to confidently classify the
success of catching trials as early as 123 milliseconds before the first ball
contact. We conclude that PSC is a promising approach for early time series
classification, when accurate and confident decisions are required.

    

### [[2107.02453] Neural Mixture Models with Expectation-Maximization for End-to-end Deep Clustering](http://arxiv.org/abs/2107.02453)


  Any clustering algorithm must synchronously learn to model the clusters and
allocate data to those clusters in the absence of labels. Mixture model-based
methods model clusters with pre-defined statistical distributions and allocate
data to those clusters based on the cluster likelihoods. They iteratively
refine those distribution parameters and member assignments following the
Expectation-Maximization (EM) algorithm. However, the cluster representability
of such hand-designed distributions that employ a limited amount of parameters
is not adequate for most real-world clustering tasks. In this paper, we realize
mixture model-based clustering with a neural network where the final layer
neurons, with the aid of an additional transformation, approximate cluster
distribution outputs. The network parameters pose as the parameters of those
distributions. The result is an elegant, much-generalized representation of
clusters than a restricted mixture of hand-designed distributions. We train the
network end-to-end via batch-wise EM iterations where the forward pass acts as
the E-step and the backward pass acts as the M-step. In image clustering, the
mixture-based EM objective can be used as the clustering objective along with
existing representation learning methods. In particular, we show that when
mixture-EM optimization is fused with consistency optimization, it improves the
sole consistency optimization performance in clustering. Our trained networks
outperform single-stage deep clustering methods that still depend on k-means,
with unsupervised classification accuracy of 63.8% in STL10, 58% in CIFAR10,
25.9% in CIFAR100, and 98.9% in MNIST.

    

### [[2107.02463] EVARS-GPR: EVent-triggered Augmented Refitting of Gaussian Process Regression for Seasonal Data](http://arxiv.org/abs/2107.02463)


  Time series forecasting is a growing domain with diverse applications.
However, changes of the system behavior over time due to internal or external
influences are challenging. Therefore, predictions of a previously learned
fore-casting model might not be useful anymore. In this paper, we present
EVent-triggered Augmented Refitting of Gaussian Process Regression for Seasonal
Data (EVARS-GPR), a novel online algorithm that is able to handle sudden shifts
in the target variable scale of seasonal data. For this purpose, EVARS-GPR
com-bines online change point detection with a refitting of the prediction
model using data augmentation for samples prior to a change point. Our
experiments on sim-ulated data show that EVARS-GPR is applicable for a wide
range of output scale changes. EVARS-GPR has on average a 20.8 % lower RMSE on
different real-world datasets compared to methods with a similar computational
resource con-sumption. Furthermore, we show that our algorithm leads to a
six-fold reduction of the averaged runtime in relation to all comparison
partners with a periodical refitting strategy. In summary, we present a
computationally efficient online fore-casting algorithm for seasonal time
series with changes of the target variable scale and demonstrate its
functionality on simulated as well as real-world data. All code is publicly
available on GitHub: this https URL.

    

### [[2107.02467] DeepDDS: deep graph neural network with attention mechanism to predict synergistic drug combinations](http://arxiv.org/abs/2107.02467)


  Drug combination therapy has become a increasingly promising method in the
treatment of cancer. However, the number of possible drug combinations is so
huge that it is hard to screen synergistic drug combinations through wet-lab
experiments. Therefore, computational screening has become an important way to
prioritize drug combinations. Graph neural network have recently shown
remarkable performance in the prediction of compound-protein interactions, but
it has not been applied to the screening of drug combinations. In this paper,
we proposed a deep learning model based on graph neural networks and attention
mechanism to identify drug combinations that can effectively inhibit the
viability of specific cancer cells. The feature embeddings of drug molecule
structure and gene expression profiles were taken as input to multi-layer
feedforward neural network to identify the synergistic drug combinations. We
compared DeepDDS with classical machine learning methods and other deep
learning-based methods on benchmark data set, and the leave-one-out
experimental results showed that DeepDDS achieved better performance than
competitive methods. Also, on an independent test set released by well-known
pharmaceutical enterprise AstraZeneca, DeepDDS was superior to competitive
methods by more than 16\% predictive precision. Furthermore, we explored the
interpretability of the graph attention network, and found the correlation
matrix of atomic features revealed important chemical substructures of drugs.
We believed that DeepDDS is an effective tool that prioritized synergistic drug
combinations for further wet-lab experiment validation.

    

### [[2107.02474] Implicit Variational Conditional Sampling with Normalizing Flows](http://arxiv.org/abs/2107.02474)


  We present a method for conditional sampling with normalizing flows when only
part of an observation is available. We rely on the following fact: if the
flow's domain can be partitioned in such a way that the flow restrictions to
subdomains keep the bijectivity property, a lower bound to the conditioning
variable log-probability can be derived. Simulation from the variational
conditional flow then amends to solving an equality constraint. Our
contribution is three-fold: a) we provide detailed insights on the choice of
variational distributions; b) we propose how to partition the input space of
the flow to preserve bijectivity property; c) we propose a set of methods to
optimise the variational distribution in specific cases. Through extensive
experiments, we show that our sampling method can be applied with success to
invertible residual networks for inference and classification.

    

### [[2107.02476] A new smart-cropping pipeline for prostate segmentation using deep learning networks](http://arxiv.org/abs/2107.02476)


  Prostate segmentation from magnetic resonance imaging (MRI) is a challenging
task. In recent years, several network architectures have been proposed to
automate this process and alleviate the burden of manual annotation. Although
the performance of these models has achieved promising results, there is still
room for improvement before these models can be used safely and effectively in
clinical practice. One of the major challenges in prostate MR image
segmentation is the presence of class imbalance in the image labels where the
background pixels dominate over the prostate. In the present work we propose a
DL-based pipeline for cropping the region around the prostate from MRI images
to produce a more balanced distribution of the foreground pixels (prostate) and
the background pixels and improve segmentation accuracy. The effect of
DL-cropping for improving the segmentation performance compared to standard
center-cropping is assessed using five popular DL networks for prostate
segmentation, namely U-net, U-net+, Res Unet++, Bridge U-net and Dense U-net.
The proposed smart-cropping outperformed the standard center cropping in terms
of segmentation accuracy for all the evaluated prostate segmentation networks.
In terms of Dice score, the highest improvement was achieved for the U-net+ and
ResU-net++ architectures corresponding to 8.9% and 8%, respectively.

    

### [[2107.02480] Midwifery Learning and Forecasting: Predicting Content Demand with User-Generated Logs](http://arxiv.org/abs/2107.02480)


  Every day, 800 women and 6,700 newborns die from complications related to
pregnancy or childbirth. A well-trained midwife can prevent most of these
maternal and newborn deaths. Data science models together with logs generated
by users of online learning applications for midwives can help to improve their
learning competencies. The goal is to use these rich behavioral data to push
digital learning towards personalized content and to provide an adaptive
learning journey. In this work, we evaluate various forecasting methods to
determine the interest of future users on the different kind of contents
available in the app, broken down by profession and region.

    

### [[2107.02495] InfoNCE is a variational autoencoder](http://arxiv.org/abs/2107.02495)


  We show that a popular self-supervised learning method, InfoNCE, is a special
case of a new family of unsupervised learning methods, the self-supervised
variational autoencoder (SSVAE). SSVAEs circumvent the usual VAE requirement to
reconstruct the data by using a carefully chosen implicit decoder. The InfoNCE
objective was motivated as a simplified parametric mutual information
estimator. Under one choice of prior, the SSVAE objective (i.e. the ELBO) is
exactly equal to the mutual information (up to constants). Under an alternative
choice of prior, the SSVAE objective is exactly equal to the simplified
parametric mutual information estimator used in InfoNCE (up to constants).
Importantly, the use of simplified parametric mutual information estimators is
believed to be critical to obtain good high-level representations, and the
SSVAE framework naturally provides a principled justification for using prior
information to choose these estimators.

    

### [[2107.02517] An Evaluation of Machine Learning and Deep Learning Models for Drought Prediction using Weather Data](http://arxiv.org/abs/2107.02517)


  Drought is a serious natural disaster that has a long duration and a wide
range of influence. To decrease the drought-caused losses, drought prediction
is the basis of making the corresponding drought prevention and disaster
reduction measures. While this problem has been studied in the literature, it
remains unknown whether drought can be precisely predicted or not with machine
learning models using weather data. To answer this question, a real-world
public dataset is leveraged in this study and different drought levels are
predicted using the last 90 days of 18 meteorological indicators as the
predictors. In a comprehensive approach, 16 machine learning models and 16 deep
learning models are evaluated and compared. The results show no single model
can achieve the best performance for all evaluation metrics simultaneously,
which indicates the drought prediction problem is still challenging. As
benchmarks for further studies, the code and results are publicly available in
a Github repository.

    

### [[2107.02520] Deep Learning Methods for Joint Optimization of Beamforming and Fronthaul Quantization in Cloud Radio Access Networks](http://arxiv.org/abs/2107.02520)


  Cooperative beamforming across access points (APs) and fronthaul quantization
strategies are essential for cloud radio access network (C-RAN) systems. The
nonconvexity of the C-RAN optimization problems, which is stemmed from per-AP
power and fronthaul capacity constraints, requires high computational
complexity for executing iterative algorithms. To resolve this issue, we
investigate a deep learning approach where the optimization module is replaced
with a well-trained deep neural network (DNN). An efficient learning solution
is proposed which constructs a DNN to produce a low-dimensional representation
of optimal beamforming and quantization strategies. Numerical results validate
the advantages of the proposed learning solution.

    

### [[2107.02521] DTGAN: Differential Private Training for Tabular GANs](http://arxiv.org/abs/2107.02521)


  Tabular generative adversarial networks (TGAN) have recently emerged to cater
to the need of synthesizing tabular data -- the most widely used data format.
While synthetic tabular data offers the advantage of complying with privacy
regulations, there still exists a risk of privacy leakage via inference attacks
due to interpolating the properties of real data during training. Differential
private (DP) training algorithms provide theoretical guarantees for training
machine learning models by injecting statistical noise to prevent privacy
leaks. However, the challenges of applying DP on TGAN are to determine the most
optimal framework (i.e., PATE/DP-SGD) and neural network (i.e.,
Generator/Discriminator)to inject noise such that the data utility is well
maintained under a given privacy guarantee. In this paper, we propose DTGAN, a
novel conditional Wasserstein tabular GAN that comes in two variants DTGAN_G
and DTGAN_D, for providing a detailed comparison of tabular GANs trained using
DP-SGD for the generator vs discriminator, respectively. We elicit the privacy
analysis associated with training the generator with complex loss functions
(i.e., classification and information losses) needed for high quality tabular
data synthesis. Additionally, we rigorously evaluate the theoretical privacy
guarantees offered by DP empirically against membership and attribute inference
attacks. Our results on 3 datasets show that the DP-SGD framework is superior
to PATE and that a DP discriminator is more optimal for training convergence.
Thus, we find (i) DTGAN_D is capable of maintaining the highest data utility
across 4 ML models by up to 18% in terms of the average precision score for a
strict privacy budget, epsilon = 1, as compared to the prior studies and (ii)
DP effectively prevents privacy loss against inference attacks by restricting
the success probability of membership attacks to be close to 50%.

    

### [[2107.02525] Semantic Segmentation Alternative Technique: Segmentation Domain Generation](http://arxiv.org/abs/2107.02525)


  Detecting objects of interest in images was always a compelling task to
automate. In recent years this task was more and more explored using deep
learning techniques, mostly using region-based convolutional networks. In this
project we propose an alternative semantic segmentation technique making use of
Generative Adversarial Networks. We consider semantic segmentation to be a
domain transfer problem. Thus, we train a feed forward network (FFNN) to
receive as input a seed real image and generate as output its segmentation
mask.

    

### [[2107.02526] Intrinsic uncertainties and where to find them](http://arxiv.org/abs/2107.02526)


  We introduce a framework for uncertainty estimation that both describes and
extends many existing methods. We consider typical hyperparameters involved in
classical training as random variables and marginalise them out to capture
various sources of uncertainty in the parameter space. We investigate which
forms and combinations of marginalisation are most useful from a practical
point of view on standard benchmarking data sets. Moreover, we discuss how some
marginalisations may produce reliable estimates of uncertainty without the need
for extensive hyperparameter tuning and/or large-scale ensembling.

    

### [[2107.02530] AdaSpeech 3: Adaptive Text to Speech for Spontaneous Style](http://arxiv.org/abs/2107.02530)


  While recent text to speech (TTS) models perform very well in synthesizing
reading-style (e.g., audiobook) speech, it is still challenging to synthesize
spontaneous-style speech (e.g., podcast or conversation), mainly because of two
reasons: 1) the lack of training data for spontaneous speech; 2) the difficulty
in modeling the filled pauses (um and uh) and diverse rhythms in spontaneous
speech. In this paper, we develop AdaSpeech 3, an adaptive TTS system that
fine-tunes a well-trained reading-style TTS model for spontaneous-style speech.
Specifically, 1) to insert filled pauses (FP) in the text sequence
appropriately, we introduce an FP predictor to the TTS model; 2) to model the
varying rhythms, we introduce a duration predictor based on mixture of experts
(MoE), which contains three experts responsible for the generation of fast,
medium and slow speech respectively, and fine-tune it as well as the pitch
predictor for rhythm adaptation; 3) to adapt to other speaker timbre, we
fine-tune some parameters in the decoder with few speech data. To address the
challenge of lack of training data, we mine a spontaneous speech dataset to
support our research this work and facilitate future research on spontaneous
TTS. Experiments show that AdaSpeech 3 synthesizes speech with natural FP and
rhythms in spontaneous styles, and achieves much better MOS and SMOS scores
than previous adaptive TTS systems.

    

### [[2107.02543] A deep-learning--based multimodal depth-aware dynamic hand gesture recognition system](http://arxiv.org/abs/2107.02543)


  Any spatio-temporal movement or reorientation of the hand, done with the
intention of conveying a specific meaning, can be considered as a hand gesture.
Inputs to hand gesture recognition systems can be in several forms, such as
depth images, monocular RGB, or skeleton joint points. We observe that raw
depth images possess low contrasts in the hand regions of interest (ROI). They
do not highlight important details to learn, such as finger bending information
(whether a finger is overlapping the palm, or another finger). Recently, in
deep-learning--based dynamic hand gesture recognition, researchers are tying to
fuse different input modalities (e.g. RGB or depth images and hand skeleton
joint points) to improve the recognition accuracy. In this paper, we focus on
dynamic hand gesture (DHG) recognition using depth quantized image features and
hand skeleton joint points. In particular, we explore the effect of using
depth-quantized features in Convolutional Neural Network (CNN) and Recurrent
Neural Network (RNN) based multi-modal fusion networks. We find that our method
improves existing results on the SHREC-DHG-14 dataset. Furthermore, using our
method, we show that it is possible to reduce the resolution of the input
images by more than four times and still obtain comparable or better accuracy
to that of the resolutions used in previous methods.

    

### [[2107.02550] The QR decomposition for radial neural networks](http://arxiv.org/abs/2107.02550)


  We provide a theoretical framework for neural networks in terms of the
representation theory of quivers, thus revealing symmetries of the parameter
space of neural networks. An exploitation of these symmetries leads to a model
compression algorithm for radial neural networks based on an analogue of the QR
decomposition. A projected version of backpropogation on the original model
matches usual backpropogation on the compressed model.

    

### [[2107.02561] Rethinking Positional Encoding](http://arxiv.org/abs/2107.02561)


  It is well noted that coordinate based MLPs benefit greatly -- in terms of
preserving high-frequency information -- through the encoding of coordinate
positions as an array of Fourier features. Hitherto, the rationale for the
effectiveness of these positional encodings has been solely studied through a
Fourier lens. In this paper, we strive to broaden this understanding by showing
that alternative non-Fourier embedding functions can indeed be used for
positional encoding. Moreover, we show that their performance is entirely
determined by a trade-off between the stable rank of the embedded matrix and
the distance preservation between embedded coordinates. We further establish
that the now ubiquitous Fourier feature mapping of position is a special case
that fulfills these conditions. Consequently, we present a more general theory
to analyze positional encoding in terms of shifted basis functions. To this
end, we develop the necessary theoretical formulae and empirically verify that
our theoretical claims hold in practice. Codes available at
this https URL.

    

### [[2107.02565] Prioritized training on points that are learnable, worth learning, and not yet learned](http://arxiv.org/abs/2107.02565)


  We introduce Goldilocks Selection, a technique for faster model training
which selects a sequence of training points that are "just right". We propose
an information-theoretic acquisition function -- the reducible validation loss
-- and compute it with a small proxy model -- GoldiProx -- to efficiently
choose training points that maximize information about a validation set. We
show that the "hard" (e.g. high loss) points usually selected in the
optimization literature are typically noisy, while the "easy" (e.g. low noise)
samples often prioritized for curriculum learning confer less information.
Further, points with uncertain labels, typically targeted by active learning,
tend to be less relevant to the task. In contrast, Goldilocks Selection chooses
points that are "just right" and empirically outperforms the above approaches.
Moreover, the selected sequence can transfer to other architectures;
practitioners can share and reuse it without the need to recreate it.

    

### [[2107.02569] Self-training with noisy student model and semi-supervised loss function for dcase 2021 challenge task 4](http://arxiv.org/abs/2107.02569)


  This report proposes a polyphonic sound event detection (SED) method for the
DCASE 2021 Challenge Task 4. The proposed SED model consists of two stages: a
mean-teacher model for providing target labels regarding weakly labeled or
unlabeled data and a self-training-based noisy student model for predicting
strong labels for sound events. The mean-teacher model, which is based on the
residual convolutional recurrent neural network (RCRNN) for the teacher and
student model, is first trained using all the training data from a weakly
labeled dataset, an unlabeled dataset, and a strongly labeled synthetic
dataset. Then, the trained mean-teacher model predicts the strong label to each
of the weakly labeled and unlabeled datasets, which is brought to the noisy
student model in the second stage of the proposed SED model. Here, the
structure of the noisy student model is identical to the RCRNN-based student
model of the mean-teacher model in the first stage. Then, it is self-trained by
adding feature noises, such as time-frequency shift, mixup, SpecAugment, and
dropout-based model noise. In addition, a semi-supervised loss function is
applied to train the noisy student model, which acts as label noise injection.
The performance of the proposed SED model is evaluated on the validation set of
the DCASE 2021 Challenge Task 4, and then, several ensemble models that combine
five-fold validation models with different hyperparameters of the
semi-supervised loss function are finally selected as our final models.

    

### [[2107.02586] Differentially private federated deep learning for multi-site medical image segmentation](http://arxiv.org/abs/2107.02586)


  Collaborative machine learning techniques such as federated learning (FL)
enable the training of models on effectively larger datasets without data
transfer. Recent initiatives have demonstrated that segmentation models trained
with FL can achieve performance similar to locally trained models. However, FL
is not a fully privacy-preserving technique and privacy-centred attacks can
disclose confidential patient data. Thus, supplementing FL with
privacy-enhancing technologies (PTs) such as differential privacy (DP) is a
requirement for clinical applications in a multi-institutional setting. The
application of PTs to FL in medical imaging and the trade-offs between privacy
guarantees and model utility, the ramifications on training performance and the
susceptibility of the final models to attacks have not yet been conclusively
investigated. Here we demonstrate the first application of differentially
private gradient descent-based FL on the task of semantic segmentation in
computed tomography. We find that high segmentation performance is possible
under strong privacy guarantees with an acceptable training time penalty. We
furthermore demonstrate the first successful gradient-based model inversion
attack on a semantic segmentation model and show that the application of DP
prevents it from divulging sensitive image features.

    

### [[2107.02597] Physics-informed regularization and structure preservation for learning stable reduced models from data with operator inference](http://arxiv.org/abs/2107.02597)


  Operator inference learns low-dimensional dynamical-system models with
polynomial nonlinear terms from trajectories of high-dimensional physical
systems (non-intrusive model reduction). This work focuses on the large class
of physical systems that can be well described by models with quadratic
nonlinear terms and proposes a regularizer for operator inference that induces
a stability bias onto quadratic models. The proposed regularizer is physics
informed in the sense that it penalizes quadratic terms with large norms and so
explicitly leverages the quadratic model form that is given by the underlying
physics. This means that the proposed approach judiciously learns from data and
physical insights combined, rather than from either data or physics alone.
Additionally, a formulation of operator inference is proposed that enforces
model constraints for preserving structure such as symmetry and definiteness in
the linear terms. Numerical results demonstrate that models learned with
operator inference and the proposed regularizer and structure preservation are
accurate and stable even in cases where using no regularization or Tikhonov
regularization leads to models that are unstable.

    

### [[2107.02603] Meta-Reinforcement Learning for Heuristic Planning](http://arxiv.org/abs/2107.02603)


  In Meta-Reinforcement Learning (meta-RL) an agent is trained on a set of
tasks to prepare for and learn faster in new, unseen, but related tasks. The
training tasks are usually hand-crafted to be representative of the expected
distribution of test tasks and hence all used in training. We show that given a
set of training tasks, learning can be both faster and more effective (leading
to better performance in the test tasks), if the training tasks are
appropriately selected. We propose a task selection algorithm,
Information-Theoretic Task Selection (ITTS), based on information theory, which
optimizes the set of tasks used for training in meta-RL, irrespectively of how
they are generated. The algorithm establishes which training tasks are both
sufficiently relevant for the test tasks, and different enough from one
another. We reproduce different meta-RL experiments from the literature and
show that ITTS improves the final performance in all of them.

    

### [[2107.02621] A Multi-Objective Approach for Sustainable Generative Audio Models](http://arxiv.org/abs/2107.02621)


  In recent years, the deep learning community has largely focused on the
accuracy of deep generative models, resulting in impressive improvements in
several research fields. However, this scientific race for quality comes at a
tremendous computational cost, which incurs vast energy consumption and
greenhouse gas emissions. If the current exponential growth of computational
consumption persists, Artificial Intelligence (AI) will sadly become a
considerable contributor to global warming.
At the heart of this problem are the measures that we use as a scientific
community to evaluate our work. Currently, researchers in the field of AI judge
scientific works mostly based on the improvement in accuracy, log-likelihood,
reconstruction or opinion scores, all of which entirely obliterates the actual
computational cost of generative models.
In this paper, we introduce the idea of relying on a multi-objective measure
based on Pareto optimality, which simultaneously integrates the models
accuracy, as well as the environmental impact of their training. By applying
this measure on the current state-of-the-art in generative audio models, we
show that this measure drastically changes the perceived significance of the
results in the field, encouraging optimal training techniques and resource
allocation. We hope that this type of measure will be widely adopted, in order
to help the community to better evaluate the significance of their work, while
bringing computational cost -- and in fine carbon emissions -- in the spotlight
of AI research.

    

### [[2107.02630] Hyperspectral Pansharpening Based on Improved Deep Image Prior and Residual Reconstruction](http://arxiv.org/abs/2107.02630)


  Hyperspectral pansharpening aims to synthesize a low-resolution hyperspectral
image (LR-HSI) with a registered panchromatic image (PAN) to generate an
enhanced HSI with high spectral and spatial resolution. Recently proposed HS
pansharpening methods have obtained remarkable results using deep convolutional
networks (ConvNets), which typically consist of three steps: (1) up-sampling
the LR-HSI, (2) predicting the residual image via a ConvNet, and (3) obtaining
the final fused HSI by adding the outputs from first and second steps. Recent
methods have leveraged Deep Image Prior (DIP) to up-sample the LR-HSI due to
its excellent ability to preserve both spatial and spectral information,
without learning from large data sets. However, we observed that the quality of
up-sampled HSIs can be further improved by introducing an additional
spatial-domain constraint to the conventional spectral-domain energy function.
We define our spatial-domain constraint as the $L_1$ distance between the
predicted PAN image and the actual PAN image. To estimate the PAN image of the
up-sampled HSI, we also propose a learnable spectral response function (SRF).
Moreover, we noticed that the residual image between the up-sampled HSI and the
reference HSI mainly consists of edge information and very fine structures. In
order to accurately estimate fine information, we propose a novel over-complete
network, called HyperKite, which focuses on learning high-level features by
constraining the receptive from increasing in the deep layers. We perform
experiments on three HSI datasets to demonstrate the superiority of our
DIP-HyperKite over the state-of-the-art pansharpening methods. The deployment
codes, pre-trained models, and final fusion outputs of our DIP-HyperKite and
the methods used for the comparisons will be publicly made available at
this https URL.

    

### [[2107.02639] Multi-Level Graph Contrastive Learning](http://arxiv.org/abs/2107.02639)


  Graph representation learning has attracted a surge of interest recently,
whose target at learning discriminant embedding for each node in the graph.
Most of these representation methods focus on supervised learning and heavily
depend on label information. However, annotating graphs are expensive to obtain
in the real world, especially in specialized domains (i.e. biology), as it
needs the annotator to have the domain knowledge to label the graph. To
approach this problem, self-supervised learning provides a feasible solution
for graph representation learning. In this paper, we propose a Multi-Level
Graph Contrastive Learning (MLGCL) framework for learning robust representation
of graph data by contrasting space views of graphs. Specifically, we introduce
a novel contrastive view - topological and feature space views. The original
graph is first-order approximation structure and contains uncertainty or error,
while the $k$NN graph generated by encoding features preserves high-order
proximity. Thus $k$NN graph generated by encoding features not only provide a
complementary view, but is more suitable to GNN encoder to extract discriminant
representation. Furthermore, we develop a multi-level contrastive mode to
preserve the local similarity and semantic similarity of graph-structured data
simultaneously. Extensive experiments indicate MLGCL achieves promising results
compared with the existing state-of-the-art graph representation learning
methods on seven datasets.

    

### [[2107.02643] Detecting Hypo-plastic Left Heart Syndrome in Fetal Ultrasound via Disease-specific Atlas Maps](http://arxiv.org/abs/2107.02643)


  Fetal ultrasound screening during pregnancy plays a vital role in the early
detection of fetal malformations which have potential long-term health impacts.
The level of skill required to diagnose such malformations from live ultrasound
during examination is high and resources for screening are often limited. We
present an interpretable, atlas-learning segmentation method for automatic
diagnosis of Hypo-plastic Left Heart Syndrome (HLHS) from a single `4 Chamber
Heart' view image. We propose to extend the recently introduced
Image-and-Spatial Transformer Networks (Atlas-ISTN) into a framework that
enables sensitising atlas generation to disease. In this framework we can
jointly learn image segmentation, registration, atlas construction and disease
prediction while providing a maximum level of clinical interpretability
compared to direct image classification methods. As a result our segmentation
allows diagnoses competitive with expert-derived manual diagnosis and yields an
AUC-ROC of 0.978 (1043 cases for training, 260 for validation and 325 for
testing).

    

### [[2107.02655] Automatic size and pose homogenization with spatial transformer network to improve and accelerate pediatric segmentation](http://arxiv.org/abs/2107.02655)


  Due to a high heterogeneity in pose and size and to a limited number of
available data, segmentation of pediatric images is challenging for deep
learning methods. In this work, we propose a new CNN architecture that is pose
and scale invariant thanks to the use of Spatial Transformer Network (STN). Our
architecture is composed of three sequential modules that are estimated
together during training: (i) a regression module to estimate a similarity
matrix to normalize the input image to a reference one; (ii) a differentiable
module to find the region of interest to segment; (iii) a segmentation module,
based on the popular UNet architecture, to delineate the object. Unlike the
original UNet, which strives to learn a complex mapping, including pose and
scale variations, from a finite training dataset, our segmentation module
learns a simpler mapping focusing on images with normalized pose and size.
Furthermore, the use of an automatic bounding box detection through STN allows
saving time and especially memory, while keeping similar performance. We test
the proposed method in kidney and renal tumor segmentation on abdominal
pediatric CT scanners. Results indicate that the estimated STN homogenization
of size and pose accelerates the segmentation (25h), compared to standard
data-augmentation (33h), while obtaining a similar quality for the kidney
(88.01\% of Dice score) and improving the renal tumor delineation (from 85.52\%
to 87.12\%).

    

### [[2107.02658] On Generalization of Graph Autoencoders with Adversarial Training](http://arxiv.org/abs/2107.02658)


  Adversarial training is an approach for increasing model's resilience against
adversarial perturbations. Such approaches have been demonstrated to result in
models with feature representations that generalize better. However, limited
works have been done on adversarial training of models on graph data. In this
paper, we raise such a question { does adversarial training improve the
generalization of graph representations. We formulate L2 and L1 versions of
adversarial training in two powerful node embedding methods: graph autoencoder
(GAE) and variational graph autoencoder (VGAE). We conduct extensive
experiments on three main applications, i.e. link prediction, node clustering,
graph anomaly detection of GAE and VGAE, and demonstrate that both L2 and L1
adversarial training boost the generalization of GAE and VGAE.

    

### [[2107.02661] Does Dataset Complexity Matters for Model Explainers?](http://arxiv.org/abs/2107.02661)


  Strategies based on Explainable Artificial Intelligence - XAI have emerged in
computing to promote a better understanding of predictions made by black box
models. Most XAI-based tools used today explain these types of models,
generating attribute rankings aimed at explaining the same, that is, the
analysis of Attribute Importance. There is no consensus on which XAI tool
generates a general rank of explainability, for this reason, several proposals
for tools have emerged (Ciu, Dalex, Eli5, Lofo, Shap and Skater). Here, we
present an experimental benchmark of explainable AI techniques capable of
producing model-agnostic global explainability ranks based on tabular data
related to different problems. Seeking to answer questions such as "Are the
explanations generated by the different tools the same, similar or different?"
and "How does data complexity play along model explainability?". The results
from the construction of 82 computational models and 592 ranks give us some
light on the other side of the problem of explainability: dataset complexity!

    

### [[2107.02681] VidLanKD: Improving Language Understanding via Video-Distilled Knowledge Transfer](http://arxiv.org/abs/2107.02681)


  Since visual perception can give rich information beyond text descriptions
for world understanding, there has been increasing interest in leveraging
visual grounding for language learning. Recently, vokenization has attracted
attention by using the predictions of a text-to-image retrieval model as labels
for language model supervision. Despite its success, the method suffers from
approximation error of using finite image labels and the lack of vocabulary
diversity of a small image-text dataset. To overcome these limitations, we
present VidLanKD, a video-language knowledge distillation method for improving
language understanding. We train a multi-modal teacher model on a video-text
dataset, and then transfer its knowledge to a student language model with a
text dataset. To avoid approximation error, we propose to use different
knowledge distillation objectives. In addition, the use of a large-scale
video-text dataset helps learn diverse and richer vocabularies. In our
experiments, VidLanKD achieves consistent improvements over text-only language
models and vokenization models, on several downstream language understanding
tasks including GLUE, SQuAD, and SWAG. We also demonstrate the improved world
knowledge, physical reasoning, and temporal reasoning capabilities of our model
by evaluating on the GLUE-diagnostics, PIQA, and TRACIE datasets. Lastly, we
present comprehensive ablation studies as well as visualizations of the learned
text-to-video grounding results of our teacher and student language models. Our
code and models are available at: this https URL


### [[2107.02689] A Model-Driven Engineering Approach to Machine Learning and Software Modeling](http://arxiv.org/abs/2107.02689)


  Models are used in both the Software Engineering (SE) and the Artificial
Intelligence (AI) communities. In the former case, models of software, which
may specify the software system architecture on different levels of abstraction
could be used in various stages of the Software Development Life-Cycle (SDLC),
from early conceptualization and design, to verification, implementation,
testing and evolution. However, in the latter case, i.e., AI, models may
provide smart capabilities, such as prediction and decision making support. For
instance, in Machine Learning (ML), which is the most popular sub-discipline of
AI at the present time, mathematical models may learn useful patterns in the
observed data instances and can become capable of making better predictions or
recommendations in the future. The goal of this work is to create synergy by
bringing models in the said communities together and proposing a holistic
approach. We illustrate how software models can become capable of producing or
dealing with data analytics and ML models. The main focus is on the Internet of
Things (IoT) and smart Cyber-Physical Systems (CPS) use cases, where both ML
and model-driven (model-based) SE play a key role. In particular, we implement
the proposed approach in an open source prototype and validate it using two use
cases from the IoT/CPS domain.

    

### [[2107.02690] Enabling Un-/Semi-Supervised Machine Learning for MDSE of the Real-World CPS/IoT Applications](http://arxiv.org/abs/2107.02690)


  In this paper, we propose a novel approach to support domain-specific
Model-Driven Software Engineering (MDSE) for the real-world use-case scenarios
of smart Cyber-Physical Systems (CPS) and the Internet of Things (IoT). We
argue that the majority of available data in the nature for Artificial
Intelligence (AI), specifically Machine Learning (ML) are unlabeled. Hence,
unsupervised and/or semi-supervised ML approaches are the practical choices.
However, prior work in the literature of MDSE has considered supervised ML
approaches, which only work with labeled training data. Our proposed approach
is fully implemented and integrated with an existing state-of-the-art MDSE tool
to serve the CPS/IoT domain. Moreover, we validate the proposed approach using
a portion of the open data of the REFIT reference dataset for the smart energy
systems domain. Our model-to-code transformations (code generators) provide the
full source code of the desired IoT services out of the model instances in an
automated manner. Currently, we generate the source code in Java and Python.
The Python code is responsible for the ML functionalities and uses the APIs of
several ML libraries and frameworks, namely Scikit-Learn, Keras and TensorFlow.
For unsupervised and semi-supervised learning, the APIs of Scikit-Learn are
deployed. In addition to the pure MDSE approach, where certain ML methods,
e.g., K-Means, Mini-Batch K-Means, DB-SCAN, Spectral Clustering, Gaussian
Mixture Model, Self-Training, Label Propagation and Label Spreading are
supported, a more flexible, hybrid approach is also enabled to support the
practitioner in deploying a pre-trained ML model with any arbitrary
architecture and learning algorithm.

    

### [[2107.02692] ML-Quadrat & DriotData: A Model-Driven Engineering Tool and a Low-Code Platform for Smart IoT Services](http://arxiv.org/abs/2107.02692)


  In this paper, we present the novel early tool prototype of ML-Quadrat, which
is an open source research prototype, based on the Eclipse Modeling Framework
(EMF) and the state of the art in the literature of Model-Driven Software
Engineering (MDSE) for smart Cyber-Physical Systems (CPS) and the Internet of
Things (IoT). Its envisioned users are mostly software developers, who might
not have deep knowledge and skills in the heterogeneous IoT platforms and the
diverse Artificial Intelligence (AI) technologies, specifically regarding Data
Analytics and Machine Learning (DAML). ML-Quadrat is released under the terms
of the Apache 2.0 license on Github: this https URL.
Additionally, the novel early tool prototype of DriotData, a Low-Code platform
targeting citizen data scientists and citizen/end-user software developers is
demonstrated. DriotData exploits and adopts ML-Quadrat and offers an extended
version of it as a web-based service to companies, especially Small- and
Medium-Sized Enterprises (SME). A basic web-based demo of the Minimum Viable
Product (MVP) of DriotData is already available. Finally, a short video
demonstrating the tools is available on YouTube: this https URL.

    

### [[2107.02693] Remote sensing, AI and innovative prediction methods for adapting cities to the impacts of the climate change](http://arxiv.org/abs/2107.02693)


  Urban areas are not only one of the biggest contributors to climate change,
but also they are one of the most vulnerable areas with high populations who
would together experience the negative impacts. In this paper, I address some
of the opportunities brought by satellite remote sensing imaging and artificial
intelligence (AI) in order to measure climate adaptation of cities
automatically. I propose an AI-based framework which might be useful for
extracting indicators from remote sensing images and might help with predictive
estimation of future states of these climate adaptation related indicators.
When such models become more robust and used in real-life applications, they
might help decision makers and early responders to choose the best actions to
sustain the wellbeing of society, natural resources and biodiversity. I
underline that this is an open field and an ongoing research for many
scientists, therefore I offer an in depth discussion on the challenges and
limitations of AI-based methods and the predictive estimation models in
general.

    

### [[2107.02711] A Unified Off-Policy Evaluation Approach for General Value Function](http://arxiv.org/abs/2107.02711)


  General Value Function (GVF) is a powerful tool to represent both the {\em
predictive} and {\em retrospective} knowledge in reinforcement learning (RL).
In practice, often multiple interrelated GVFs need to be evaluated jointly with
pre-collected off-policy samples. In the literature, the gradient temporal
difference (GTD) learning method has been adopted to evaluate GVFs in the
off-policy setting, but such an approach may suffer from a large estimation
error even if the function approximation class is sufficiently expressive.
Moreover, none of the previous work have formally established the convergence
guarantee to the ground truth GVFs under the function approximation settings.
In this paper, we address both issues through the lens of a class of GVFs with
causal filtering, which cover a wide range of RL applications such as reward
variance, value gradient, cost in anomaly detection, stationary distribution
gradient, etc. We propose a new algorithm called GenTD for off-policy GVFs
evaluation and show that GenTD learns multiple interrelated multi-dimensional
GVFs as efficiently as a single canonical scalar value function. We further
show that unlike GTD, the learned GVFs by GenTD are guaranteed to converge to
the ground truth GVFs as long as the function approximation power is
sufficiently large. To our best knowledge, GenTD is the first off-policy GVF
evaluation algorithm that has global optimality guarantee.

    

### [[2107.02716] Evaluating subgroup disparity using epistemic uncertainty in mammography](http://arxiv.org/abs/2107.02716)


  As machine learning (ML) continue to be integrated into healthcare systems
that affect clinical decision making, new strategies will need to be
incorporated in order to effectively detect and evaluate subgroup disparities
to ensure accountability and generalizability in clinical workflows. In this
paper, we explore how epistemic uncertainty can be used to evaluate disparity
in patient demographics (race) and data acquisition (scanner) subgroups for
breast density assessment on a dataset of 108,190 mammograms collected from 33
clinical sites. Our results show that even if aggregate performance is
comparable, the choice of uncertainty quantification metric can significantly
the subgroup level. We hope this analysis can promote further work on how
uncertainty can be leveraged to increase transparency of machine learning
applications for clinical deployment.

    

### [[2107.02729] AdaRL: What, Where, and How to Adapt in Transfer Reinforcement Learning](http://arxiv.org/abs/2107.02729)


  Most approaches in reinforcement learning (RL) are data-hungry and specific
to fixed environments. In this paper, we propose a principled framework for
adaptive RL, called AdaRL, that adapts reliably to changes across domains.
Specifically, we construct a generative environment model for the structural
relationships among variables in the system and embed the changes in a compact
way, which provides a clear and interpretable picture for locating what and
where the changes are and how to adapt. Based on the environment model, we
characterize a minimal set of representations, including both domain-specific
factors and domain-shared state representations, that suffice for reliable and
low-cost transfer. Moreover, we show that by explicitly leveraging a compact
representation to encode changes, we can adapt the policy with only a few
samples without further policy optimization in the target domain. We illustrate
the efficacy of AdaRL through a series of experiments that allow for changes in
different components of Cartpole and Atari games.

    

### [[2107.02732] Provable Lipschitz Certification for Generative Models](http://arxiv.org/abs/2107.02732)


  We present a scalable technique for upper bounding the Lipschitz constant of
generative models. We relate this quantity to the maximal norm over the set of
attainable vector-Jacobian products of a given generative model. We approximate
this set by layerwise convex approximations using zonotopes. Our approach
generalizes and improves upon prior work using zonotope transformers and we
extend to Lipschitz estimation of neural networks with large output dimension.
This provides efficient and tight bounds on small networks and can scale to
generative models on VAE and DCGAN architectures.

    

### [[2107.02736] DEANN: Speeding up Kernel-Density Estimation using Approximate Nearest Neighbor Search](http://arxiv.org/abs/2107.02736)


  Kernel Density Estimation (KDE) is a nonparametric method for estimating the
shape of a density function, given a set of samples from the distribution.
Recently, locality-sensitive hashing, originally proposed as a tool for nearest
neighbor search, has been shown to enable fast KDE data structures. However,
these approaches do not take advantage of the many other advances that have
been made in algorithms for nearest neighbor algorithms. We present an
algorithm called Density Estimation from Approximate Nearest Neighbors (DEANN)
where we apply Approximate Nearest Neighbor (ANN) algorithms as a black box
subroutine to compute an unbiased KDE. The idea is to find points that have a
large contribution to the KDE using ANN, compute their contribution exactly,
and approximate the remainder with Random Sampling (RS). We present a
theoretical argument that supports the idea that an ANN subroutine can speed up
the evaluation. Furthermore, we provide a C++ implementation with a Python
interface that can make use of an arbitrary ANN implementation as a subroutine
for KDE evaluation. We show empirically that our implementation outperforms
state of the art implementations in all high dimensional datasets we
considered, and matches the performance of RS in cases where the ANN yield no
gains in performance.

    

### [[2107.02738] Dueling Bandits with Team Comparisons](http://arxiv.org/abs/2107.02738)


  We introduce the dueling teams problem, a new online-learning setting in
which the learner observes noisy comparisons of disjoint pairs of $k$-sized
teams from a universe of $n$ players. The goal of the learner is to minimize
the number of duels required to identify, with high probability, a Condorcet
winning team, i.e., a team which wins against any other disjoint team (with
probability at least $1/2$). Noisy comparisons are linked to a total order on
the teams. We formalize our model by building upon the dueling bandits setting
(Yue et al.2012) and provide several algorithms, both for stochastic and
deterministic settings. For the stochastic setting, we provide a reduction to
the classical dueling bandits setting, yielding an algorithm that identifies a
Condorcet winning team within $\mathcal{O}((n + k \log (k)) \frac{\max(\log\log
n, \log k)}{\Delta^2})$ duels, where $\Delta$ is a gap parameter. For
deterministic feedback, we additionally present a gap-independent algorithm
that identifies a Condorcet winning team within $\mathcal{O}(nk\log(k)+k^5)$
duels.

    

### [[2107.02744] Neural Computing](http://arxiv.org/abs/2107.02744)


  This chapter aims to provide next-level understanding of the problems of the
world and the solutions available to those problems, which lie very well within
the domain of neural computing, and at the same time are intelligent in their
approach, to invoke a sense of innovation among the educationalists,
researchers, academic professionals, students and people concerned, by
highlighting the work done by major researchers and innovators in this field
and thus, encouraging the readers to develop newer and more advanced techniques
for the same. By means of this chapter, the societal problems are discussed and
various solutions are also given by means of the theories presented and
researches done so far. Different types of neural networks discovered so far
and applications of some of those neural networks are focused on, apart from
their theoretical understanding, the working and core concepts involved in the
applications.

    

### [[2107.02751] Quantum Annealing Formulation for Binary Neural Networks](http://arxiv.org/abs/2107.02751)


  Quantum annealing is a promising paradigm for building practical quantum
computers. Compared to other approaches, quantum annealing technology has been
scaled up to a larger number of qubits. On the other hand, deep learning has
been profoundly successful in pushing the boundaries of AI. It is thus natural
to investigate potentially game changing technologies such as quantum annealers
to augment the capabilities of deep learning. In this work, we explore binary
neural networks, which are lightweight yet powerful models typically intended
for resource constrained devices. Departing from current training regimes for
binary networks that smooth/approximate the activation functions to make the
network differentiable, we devise a quadratic unconstrained binary optimization
formulation for the training problem. While the problem is intractable, i.e.,
the cost to estimate the binary weights scales exponentially with network size,
we show how the problem can be optimized directly on a quantum annealer,
thereby opening up to the potential gains of quantum computing. We
experimentally validated our formulation via simulation and testing on an
actual quantum annealer (D-Wave Advantage), the latter to the extent allowable
by the capacity of current technology.

    

### [[2107.02755] FedFog: Network-Aware Optimization of Federated Learning over Wireless Fog-Cloud Systems](http://arxiv.org/abs/2107.02755)


  Federated learning (FL) is capable of performing large distributed machine
learning tasks across multiple edge users by periodically aggregating trained
local parameters. To address key challenges of enabling FL over a wireless
fog-cloud system (e.g., non-i.i.d. data, users' heterogeneity), we first
propose an efficient FL algorithm (called FedFog) to perform the local
aggregation of gradient parameters at fog servers and global training update at
the cloud. Next, we employ FedFog in wireless fog-cloud systems by
investigating a novel network-aware FL optimization problem that strikes the
balance between the global loss and completion time. An iterative algorithm is
then developed to obtain a precise measurement of the system performance, which
helps design an efficient stopping criteria to output an appropriate number of
global rounds. To mitigate the straggler effect, we propose a flexible user
aggregation strategy that trains fast users first to obtain a certain level of
accuracy before allowing slow users to join the global training updates.
Extensive numerical results using several real-world FL tasks are provided to
verify the theoretical convergence of FedFog. We also show that the proposed
co-design of FL and communication is essential to substantially improve
resource utilization while achieving comparable accuracy of the learning model.

    

### [[2107.02757] Sawtooth Factorial Topic Embeddings Guided Gamma Belief Network](http://arxiv.org/abs/2107.02757)


  Hierarchical topic models such as the gamma belief network (GBN) have
delivered promising results in mining multi-layer document representations and
discovering interpretable topic taxonomies. However, they often assume in the
prior that the topics at each layer are independently drawn from the Dirichlet
distribution, ignoring the dependencies between the topics both at the same
layer and across different layers. To relax this assumption, we propose
sawtooth factorial topic embedding guided GBN, a deep generative model of
documents that captures the dependencies and semantic similarities between the
topics in the embedding space. Specifically, both the words and topics are
represented as embedding vectors of the same dimension. The topic matrix at a
layer is factorized into the product of a factor loading matrix and a topic
embedding matrix, the transpose of which is set as the factor loading matrix of
the layer above. Repeating this particular type of factorization, which shares
components between adjacent layers, leads to a structure referred to as
sawtooth factorization. An auto-encoding variational inference network is
constructed to optimize the model parameter via stochastic gradient descent.
Experiments on big corpora show that our models outperform other neural topic
models on extracting deeper interpretable topics and deriving better document
representations.

    

### [[2107.02772] Causal Bandits on General Graphs](http://arxiv.org/abs/2107.02772)


  We study the problem of determining the best intervention in a Causal
Bayesian Network (CBN) specified only by its causal graph. We model this as a
stochastic multi-armed bandit (MAB) problem with side-information, where the
interventions correspond to the arms of the bandit instance. First, we propose
a simple regret minimization algorithm that takes as input a semi-Markovian
causal graph with atomic interventions and possibly unobservable variables, and
achieves $\tilde{O}(\sqrt{M/T})$ expected simple regret, where $M$ is dependent
on the input CBN and could be very small compared to the number of arms. We
also show that this is almost optimal for CBNs described by causal graphs
having an $n$-ary tree structure. Our simple regret minimization results, both
upper and lower bound, subsume previous results in the literature, which
assumed additional structural restrictions on the input causal graph. In
particular, our results indicate that the simple regret guarantee of our
proposed algorithm can only be improved by considering more nuanced structural
restrictions on the causal graph. Next, we propose a cumulative regret
minimization algorithm that takes as input a general causal graph with all
observable nodes and atomic interventions and performs better than the optimal
MAB algorithm that does not take causal side-information into account. We also
experimentally compare both our algorithms with the best known algorithms in
the literature. To the best of our knowledge, this work gives the first simple
and cumulative regret minimization algorithms for CBNs with general causal
graphs under atomic interventions and having unobserved confounders.

    

### [[2107.02776] Counterfactual Explanations in Sequential Decision Making Under Uncertainty](http://arxiv.org/abs/2107.02776)


  Methods to find counterfactual explanations have predominantly focused on one
step decision making processes. In this work, we initiate the development of
methods to find counterfactual explanations for decision making processes in
which multiple, dependent actions are taken sequentially over time. We start by
formally characterizing a sequence of actions and states using finite horizon
Markov decision processes and the Gumbel-Max structural causal model. Building
upon this characterization, we formally state the problem of finding
counterfactual explanations for sequential decision making processes. In our
problem formulation, the counterfactual explanation specifies an alternative
sequence of actions differing in at most k actions from the observed sequence
that could have led the observed process realization to a better outcome. Then,
we introduce a polynomial time algorithm based on dynamic programming to build
a counterfactual policy that is guaranteed to always provide the optimal
counterfactual explanation on every possible realization of the counterfactual
environment dynamics. We validate our algorithm using both synthetic and real
data from cognitive behavioral therapy and show that the counterfactual
explanations our algorithm finds can provide valuable insights to enhance
sequential decision making under uncertainty.

    

### [[2107.02780] Causal Inference with Corrupted Data: Measurement Error, Missing Values, Discretization, and Differential Privacy](http://arxiv.org/abs/2107.02780)


  Even the most carefully curated economic data sets have variables that are
noisy, missing, discretized, or privatized. The standard workflow for empirical
research involves data cleaning followed by data analysis that typically
ignores the bias and variance consequences of data cleaning. We formulate a
semiparametric model for causal inference with corrupted data to encompass both
data cleaning and data analysis. We propose a new end-to-end procedure for data
cleaning, estimation, and inference with data cleaning-adjusted confidence
intervals. We prove root-n consistency, Gaussian approximation, and
semiparametric efficiency for our estimator of the causal parameter by finite
sample arguments. Our key assumption is that the true covariates are
approximately low rank. In our analysis, we provide nonasymptotic theoretical
contributions to matrix completion, statistical learning, and semiparametric
statistics. We verify the coverage of the data cleaning-adjusted confidence
intervals in simulations.

    

### [[2107.02783] SAGE: Intrusion Alert-driven Attack Graph Extractor](http://arxiv.org/abs/2107.02783)


  Attack graphs (AG) are used to assess pathways availed by cyber adversaries
to penetrate a network. State-of-the-art approaches for AG generation focus
mostly on deriving dependencies between system vulnerabilities based on network
scans and expert knowledge. In real-world operations however, it is costly and
ineffective to rely on constant vulnerability scanning and expert-crafted AGs.
We propose to automatically learn AGs based on actions observed through
intrusion alerts, without prior expert knowledge. Specifically, we develop an
unsupervised sequence learning system, SAGE, that leverages the temporal and
probabilistic dependence between alerts in a suffix-based probabilistic
deterministic finite automaton (S-PDFA) -- a model that accentuates infrequent
severe alerts and summarizes paths leading to them. AGs are then derived from
the S-PDFA. Tested with intrusion alerts collected through Collegiate
Penetration Testing Competition, SAGE produces AGs that reflect the strategies
used by participating teams. The resulting AGs are succinct, interpretable, and
enable analysts to derive actionable insights, e.g., attackers tend to follow
shorter paths after they have discovered a longer one.

    

### [[2107.02784] Data-driven reduced order modeling of environmental hydrodynamics using deep autoencoders and neural ODEs](http://arxiv.org/abs/2107.02784)


  Model reduction for fluid flow simulation continues to be of great interest
across a number of scientific and engineering fields. In a previous work
[arXiv:2104.13962], we explored the use of Neural Ordinary Differential
Equations (NODE) as a non-intrusive method for propagating the latent-space
dynamics in reduced order models. Here, we investigate employing deep
autoencoders for discovering the reduced basis representation, the dynamics of
which are then approximated by NODE. The ability of deep autoencoders to
represent the latent-space is compared to the traditional proper orthogonal
decomposition (POD) approach, again in conjunction with NODE for capturing the
dynamics. Additionally, we compare their behavior with two classical
non-intrusive methods based on POD and radial basis function interpolation as
well as dynamic mode decomposition. The test problems we consider include
incompressible flow around a cylinder as well as a real-world application of
shallow water hydrodynamics in an estuarine system. Our findings indicate that
deep autoencoders can leverage nonlinear manifold learning to achieve a highly
efficient compression of spatial information and define a latent-space that
appears to be more suitable for capturing the temporal dynamics through the
NODE framework.

    

### [[2107.02791] Depth-supervised NeRF: Fewer Views and Faster Training for Free](http://arxiv.org/abs/2107.02791)


  One common failure mode of Neural Radiance Field (NeRF) models is fitting
incorrect geometries when given an insufficient number of input views. We
propose DS-NeRF (Depth-supervised Neural Radiance Fields), a loss for learning
neural radiance fields that takes advantage of readily-available depth
supervision. Our key insight is that sparse depth supervision can be used to
regularize the learned geometry, a crucial component for effectively rendering
novel views using NeRF. We exploit the fact that current NeRF pipelines require
images with known camera poses that are typically estimated by running
structure-from-motion (SFM). Crucially, SFM also produces sparse 3D points that
can be used as ``free" depth supervision during training: we simply add a loss
to ensure that depth rendered along rays that intersect these 3D points is
close to the observed depth. We find that DS-NeRF can render more accurate
images given fewer training views while training 2-6x faster. With only two
training views on real-world images, DS-NeRF significantly outperforms NeRF as
well as other sparse-view variants. We show that our loss is compatible with
these NeRF models, demonstrating that depth is a cheap and easily digestible
supervisory signal. Finally, we show that DS-NeRF supports other types of depth
supervision such as scanned depth sensors and RGBD reconstruction outputs.

    

### [[2107.02792] Learned Visual Navigation for Under-Canopy Agricultural Robots](http://arxiv.org/abs/2107.02792)


  We describe a system for visually guided autonomous navigation of
under-canopy farm robots. Low-cost under-canopy robots can drive between crop
rows under the plant canopy and accomplish tasks that are infeasible for
over-the-canopy drones or larger agricultural equipment. However, autonomously
navigating them under the canopy presents a number of challenges: unreliable
GPS and LiDAR, high cost of sensing, challenging farm terrain, clutter due to
leaves and weeds, and large variability in appearance over the season and
across crop types. We address these challenges by building a modular system
that leverages machine learning for robust and generalizable perception from
monocular RGB images from low-cost cameras, and model predictive control for
accurate control in challenging terrain. Our system, CropFollow, is able to
autonomously drive 485 meters per intervention on average, outperforming a
state-of-the-art LiDAR based system (286 meters per intervention) in extensive
field testing spanning over 25 km.

    

### [[2107.02794] Improving Coherence and Consistency in Neural Sequence Models with Dual-System, Neuro-Symbolic Reasoning](http://arxiv.org/abs/2107.02794)


  Human reasoning can often be understood as an interplay between two systems:
the intuitive and associative ("System 1") and the deliberative and logical
("System 2"). Neural sequence models -- which have been increasingly successful
at performing complex, structured tasks -- exhibit the advantages and failure
modes of System 1: they are fast and learn patterns from data, but are often
inconsistent and incoherent. In this work, we seek a lightweight, training-free
means of improving existing System 1-like sequence models by adding System
2-inspired logical reasoning. We explore several variations on this theme in
which candidate generations from a neural sequence model are examined for
logical consistency by a symbolic reasoning module, which can either accept or
reject the generations. Our approach uses neural inference to mediate between
the neural System 1 and the logical System 2. Results in robust story
generation and grounded instruction-following show that this approach can
increase the coherence and accuracy of neurally-based generations.

    

### [[1904.08084] General Purpose (GenP) Bioimage Ensemble of Handcrafted and Learned Features with Data Augmentation](http://arxiv.org/abs/1904.08084)


  Bioimage classification plays a crucial role in many biological problems. In
this work, we present a new General Purpose (GenP) ensemble that boosts
performance by combining local features, dense sampling features, and deep
learning approaches. First, we introduce three new methods for data
augmentation based on PCA/DCT; second, we show that different data augmentation
approaches can boost the performance of an ensemble of CNNs; and, finally, we
propose a set of handcrafted/learned descriptors that are highly generalizable.
Each handcrafted descriptor is used to train a different Support Vector Machine
(SVM), and the different SVMs are combined with the ensemble of CNNs. Our
method is evaluated on a diverse set of bioimage classification problems.
Results demonstrate that the proposed GenP bioimage ensemble obtains
state-of-the-art performance without any ad-hoc dataset tuning of parameters
(thus avoiding the risk of overfitting/overtraining).

    

### [[1905.13611] ADMM for Efficient Deep Learning with Global Convergence](http://arxiv.org/abs/1905.13611)


  Alternating Direction Method of Multipliers (ADMM) has been used successfully
in many conventional machine learning applications and is considered to be a
useful alternative to Stochastic Gradient Descent (SGD) as a deep learning
optimizer. However, as an emerging domain, several challenges remain, including
1) The lack of global convergence guarantees, 2) Slow convergence towards
solutions, and 3) Cubic time complexity with regard to feature dimensions. In
this paper, we propose a novel optimization framework for deep learning via
ADMM (dlADMM) to address these challenges simultaneously. The parameters in
each layer are updated backward and then forward so that the parameter
information in each layer is exchanged efficiently. The time complexity is
reduced from cubic to quadratic in (latent) feature dimensions via a dedicated
algorithm design for subproblems that enhances them utilizing iterative
quadratic approximations and backtracking. Finally, we provide the first proof
of global convergence for an ADMM-based method (dlADMM) in a deep neural
network problem under mild conditions. Experiments on benchmark datasets
demonstrated that our proposed dlADMM algorithm outperforms most of the
comparison methods.

    

### [[1910.10835] Large Scale Model Predictive Control with Neural Networks and Primal Active Sets](http://arxiv.org/abs/1910.10835)


  This work presents an explicit-implicit procedure to compute a model
predictive control (MPC) law with guarantees on recursive feasibility and
asymptotic stability. The approach combines an offline-trained fully-connected
neural network with an online primal active set solver. The neural network
provides a control input initialization while the primal active set method
ensures recursive feasibility and asymptotic stability. The neural network is
trained with a primal-dual loss function, aiming to generate control sequences
that are primal feasible and meet a desired level of suboptimality. Since the
neural network alone does not guarantee constraint satisfaction, its output is
used to warm start the primal active set method online. We demonstrate that
this approach scales to large problems with thousands of optimization
variables, which are challenging for current approaches. Our method achieves a
2x reduction in online inference time compared to the best method in a
benchmark suite of different solver and initialization strategies.

    

### [[1910.13645] Automatic Testing With Reusable Adversarial Agents](http://arxiv.org/abs/1910.13645)


  Autonomous systems such as self-driving cars and general-purpose robots are
safety-critical systems that operate in highly uncertain and dynamic
environments. We propose an interactive multi-agent framework where the
system-under-design is modeled as an ego agent and its environment is modeled
by a number of adversarial (ado) agents. For example, a self-driving car is an
ego agent whose behavior is influenced by ado agents such as pedestrians,
bicyclists, traffic lights, road geometry etc. Given a logical specification of
the correct behavior of the ego agent, and a set of constraints that encode
reasonable adversarial behavior, our framework reduces the adversarial testing
problem to the problem of synthesizing controllers for (constrained) ado agents
that cause the ego agent to violate its specifications. Specifically, we
explore the use of tabular and deep reinforcement learning approaches for
synthesizing adversarial agents. We show that ado agents trained in this
fashion are better than traditional falsification or testing techniques because
they can generalize to ego agents and environments that differ from the
original ego agent. We demonstrate the efficacy of our technique on two
real-world case studies from the domain of self-driving cars.

    

### [[2001.04026] Fractional order graph neural network](http://arxiv.org/abs/2001.04026)


  This paper proposes fractional order graph neural networks (FGNNs), optimized
by the approximation strategy to address the challenges of local optimum of
classic and fractional graph neural networks which are specialised at
aggregating information from the feature and adjacent matrices of connected
nodes and their neighbours to solve learning tasks on non-Euclidean data such
as graphs. Meanwhile the approximate calculation of fractional order gradients
also overcomes the high computational complexity of fractional order
derivations. We further prove that such an approximation is feasible and the
FGNN is unbiased towards global optimization solution. Extensive experiments on
citation networks show that FGNN achieves great advantage over baseline models
when selected appropriate fractional order.

    

### [[2003.00335] Differentiating through the Fréchet Mean](http://arxiv.org/abs/2003.00335)


  Recent advances in deep representation learning on Riemannian manifolds
extend classical deep learning operations to better capture the geometry of the
manifold. One possible extension is the Fréchet mean, the generalization of
the Euclidean mean; however, it has been difficult to apply because it lacks a
closed form with an easily computable derivative. In this paper, we show how to
differentiate through the Fréchet mean for arbitrary Riemannian manifolds.
Then, focusing on hyperbolic space, we derive explicit gradient expressions and
a fast, accurate, and hyperparameter-free Fréchet mean solver. This fully
integrates the Fréchet mean into the hyperbolic neural network pipeline. To
demonstrate this integration, we present two case studies. First, we apply our
Fréchet mean to the existing Hyperbolic Graph Convolutional Network,
replacing its projected aggregation to obtain state-of-the-art results on
datasets with high hyperbolicity. Second, to demonstrate the Fréchet mean's
capacity to generalize Euclidean neural network operations, we develop a
hyperbolic batch normalization method that gives an improvement parallel to the
one observed in the Euclidean setting.

    

### [[2005.00478] DriveML: An R Package for Driverless Machine Learning](http://arxiv.org/abs/2005.00478)


  In recent years, the concept of automated machine learning has become very
popular. Automated Machine Learning (AutoML) mainly refers to the automated
methods for model selection and hyper-parameter optimization of various
algorithms such as random forests, gradient boosting, neural networks, etc. In
this paper, we introduce a new package i.e. DriveML for automated machine
learning. DriveML helps in implementing some of the pillars of an automated
machine learning pipeline such as automated data preparation, feature
engineering, model building and model explanation by running the function
instead of writing lengthy R codes. The DriveML package is available in CRAN.
We compare the DriveML package with other relevant packages in CRAN/Github and
find that DriveML performs the best across different parameters. We also
provide an illustration by applying the DriveML package with default
configuration on a real world dataset. Overall, the main benefits of DriveML
are in development time savings, reduce developer's errors, optimal tuning of
machine learning models and reproducibility.

    

### [[2006.08328] ETHOS: an Online Hate Speech Detection Dataset](http://arxiv.org/abs/2006.08328)


  Online hate speech is a recent problem in our society that is rising at a
steady pace by leveraging the vulnerabilities of the corresponding regimes that
characterise most social media platforms. This phenomenon is primarily fostered
by offensive comments, either during user interaction or in the form of a
posted multimedia context. Nowadays, giant corporations own platforms where
millions of users log in every day, and protection from exposure to similar
phenomena appears to be necessary in order to comply with the corresponding
legislation and maintain a high level of service quality. A robust and reliable
system for detecting and preventing the uploading of relevant content will have
a significant impact on our digitally interconnected society. Several aspects
of our daily lives are undeniably linked to our social profiles, making us
vulnerable to abusive behaviours. As a result, the lack of accurate hate speech
detection mechanisms would severely degrade the overall user experience,
although its erroneous operation would pose many ethical concerns. In this
paper, we present 'ETHOS', a textual dataset with two variants: binary and
multi-label, based on YouTube and Reddit comments validated using the
Figure-Eight crowdsourcing platform. Furthermore, we present the annotation
protocol used to create this dataset: an active sampling procedure for
balancing our data in relation to the various aspects defined. Our key
assumption is that, even gaining a small amount of labelled data from such a
time-consuming process, we can guarantee hate speech occurrences in the
examined material.

    

### [[2006.09223] Risk bounds when learning infinitely many response functions by ordinary linear regression](http://arxiv.org/abs/2006.09223)


  Consider the problem of learning a large number of response functions
simultaneously based on the same input variables. The training data consist of
a single independent random sample of the input variables drawn from a common
distribution together with the associated responses. The input variables are
mapped into a high-dimensional linear space, called the feature space, and the
response functions are modelled as linear functionals of the mapped features,
with coefficients calibrated via ordinary least squares. We provide convergence
guarantees on the worst-case excess prediction risk by controlling the
convergence rate of the excess risk uniformly in the response function. The
dimension of the feature map is allowed to tend to infinity with the sample
size. The collection of response functions, although potentially infinite, is
supposed to have a finite Vapnik-Chervonenkis dimension. The bound derived can
be applied when building multiple surrogate models in a reasonable computing
time.

    

### [[2006.13823] Maximizing Ensemble Diversity in Deep Q-Learning](http://arxiv.org/abs/2006.13823)


  The classic DQN algorithm is limited by the overestimation bias of the
learned Q-function. Subsequent algorithms have proposed techniques to reduce
this problem, without fully eliminating it. Recently, the Maxmin and Ensemble
Q-learning algorithms have used different estimates provided by the ensembles
of learners to reduce the overestimation bias. Unfortunately, these learners
can converge to the same point in the parametric or representation space,
falling back to the classic single neural network DQN. In this paper, we
describe a regularization technique to maximize ensemble diversity in these
algorithms. We propose and compare five regularization functions inspired from
economics theory and consensus optimization. We show that the regularized
approach significantly outperforms the Maxmin and Ensemble Q-learning
algorithms as well as non-ensemble baselines.

    

### [[2006.14062] An $\ell_p$ theory of PCA and spectral clustering](http://arxiv.org/abs/2006.14062)


  Principal Component Analysis (PCA) is a powerful tool in statistics and
machine learning. While existing study of PCA focuses on the recovery of
principal components and their associated eigenvalues, there are few precise
characterizations of individual principal component scores that yield
low-dimensional embedding of samples. That hinders the analysis of various
spectral methods. In this paper, we first develop an $\ell_p$ perturbation
theory for a hollowed version of PCA in Hilbert spaces which provably improves
upon the vanilla PCA in the presence of heteroscedastic noises. Through a novel
$\ell_p$ analysis of eigenvectors, we investigate entrywise behaviors of
principal component score vectors and show that they can be approximated by
linear functionals of the Gram matrix in $\ell_p$ norm, which includes $\ell_2$
and $\ell_\infty$ as special examples. For sub-Gaussian mixture models, the
choice of $p$ giving optimal bounds depends on the signal-to-noise ratio, which
further yields optimality guarantees for spectral clustering. For contextual
community detection, the $\ell_p$ theory leads to a simple spectral algorithm
that achieves the information threshold for exact recovery. These also provide
optimal recovery results for Gaussian mixture and stochastic block models as
special cases.

    

### [[2007.02445] Overlapping Spaces for Compact Graph Representations](http://arxiv.org/abs/2007.02445)


  Various non-trivial spaces are becoming popular for embedding structured data
such as graphs, texts, or images. Following spherical and hyperbolic spaces,
more general product spaces have been proposed. However, searching for the best
configuration of product space is a resource-intensive procedure, which reduces
the practical applicability of the idea. We generalize the concept of product
space and introduce an overlapping space that does not have the configuration
search problem. The main idea is to allow subsets of coordinates to be shared
between spaces of different types (Euclidean, hyperbolic, spherical). As a
result, parameter optimization automatically learns the optimal configuration.
Additionally, overlapping spaces allow for more compact representations since
their geometry is more complex. Our experiments confirm that overlapping spaces
outperform the competitors in graph embedding tasks. Here, we consider both
distortion setup, where the aim is to preserve distances, and ranking setup,
where the relative order should be preserved. The proposed method effectively
solves the problem and outperforms the competitors in both settings. We also
perform an empirical analysis in a realistic information retrieval task, where
we compare all spaces by incorporating them into DSSM. In this case, the
proposed overlapping space consistently achieves nearly optimal results without
any configuration tuning. This allows for reducing training time, which can be
significant in large-scale applications.

    

### [[2008.04216] Using Experts' Opinions in Machine Learning Tasks](http://arxiv.org/abs/2008.04216)


  In machine learning tasks, especially in the tasks of prediction, scientists
tend to rely solely on available historical data and disregard unproven
insights, such as experts' opinions, polls, and betting odds. In this paper, we
propose a general three-step framework for utilizing experts' insights in
machine learning tasks and build four concrete models for a sports game
prediction case study. For the case study, we have chosen the task of
predicting NCAA Men's Basketball games, which has been the focus of a group of
Kaggle competitions in recent years. Results highly suggest that the good
performance and high scores of the past models are a result of chance, and not
because of a good-performing and stable model. Furthermore, our proposed models
can achieve more steady results with lower log loss average (best at 0.489)
compared to the top solutions of the 2019 competition (>0.503), and reach the
top 1%, 10% and 1% in the 2017, 2018 and 2019 leaderboards, respectively.

    

### [[2010.08346] From Talk to Action with Accountability: Monitoring the Public Discussion of Policy Makers with Deep Neural Networks and Topic Modelling](http://arxiv.org/abs/2010.08346)


  Decades of research on climate have provided a consensus that human activity
has changed the climate and we are currently heading into a climate crisis.
While public discussion and research efforts on climate change mitigation have
increased, potential solutions need to not only be discussed but also
effectively deployed. For preventing mismanagement and holding policy makers
accountable, transparency and degree of information about government processes
have been shown to be crucial. However, currently the quantity of information
about climate change discussions and the range of sources make it increasingly
difficult for the public and civil society to maintain an overview to hold
politicians accountable.
In response, we propose a multi-source topic aggregation system (MuSTAS)
which processes policy makers speech and rhetoric from several publicly
available sources into an easily digestible topic summary. MuSTAS uses novel
multi-source hybrid latent Dirichlet allocation to model topics from a variety
of documents. This topic digest will serve the general public and civil society
in assessing where, how, and when politicians talk about climate and climate
policies, enabling them to hold politicians accountable for their actions to
mitigate climate change and lack thereof.

    

### [[2010.11748] Classification with Rejection Based on Cost-sensitive Classification](http://arxiv.org/abs/2010.11748)


  The goal of classification with rejection is to avoid risky misclassification
in error-critical applications such as medical diagnosis and product
inspection. In this paper, based on the relationship between classification
with rejection and cost-sensitive classification, we propose a novel method of
classification with rejection by learning an ensemble of cost-sensitive
classifiers, which satisfies all the following properties: (i) it can avoid
estimating class-posterior probabilities, resulting in improved classification
accuracy, (ii) it allows a flexible choice of losses including non-convex ones,
(iii) it does not require complicated modifications when using different
losses, (iv) it is applicable to both binary and multiclass cases, and (v) it
is theoretically justifiable for any classification-calibrated loss.
Experimental results demonstrate the usefulness of our proposed approach in
clean-labeled, noisy-labeled, and positive-unlabeled classification.

    

### [[2011.05354] Contrastive Losses and Solution Caching for Predict-and-Optimize](http://arxiv.org/abs/2011.05354)


  Many decision-making processes involve solving a combinatorial optimization
problem with uncertain input that can be estimated from historic data.
Recently, problems in this class have been successfully addressed via
end-to-end learning approaches, which rely on solving one optimization problem
for each training instance at every epoch. In this context, we provide two
distinct contributions. First, we use a Noise Contrastive approach to motivate
a family of surrogate loss functions, based on viewing non-optimal solutions as
negative examples. Second, we address a major bottleneck of all
predict-and-optimize approaches, i.e. the need to frequently recompute optimal
solutions at training time. This is done via a solver-agnostic solution caching
scheme, and by replacing optimization calls with a lookup in the solution
cache. The method is formally based on an inner approximation of the feasible
space and, combined with a cache lookup strategy, provides a controllable
trade-off between training time and accuracy of the loss approximation. We
empirically show that even a very slow growth rate is enough to match the
quality of state-of-the-art methods, at a fraction of the computational cost.

    

### [[2011.11202] Effectiveness of MPC-friendly Softmax Replacement](http://arxiv.org/abs/2011.11202)


  Softmax is widely used in deep learning to map some representation to a
probability distribution. As it is based on exp/log functions that are
relatively expensive in multi-party computation, Mohassel and Zhang (2017)
proposed a simpler replacement based on ReLU to be used in secure computation.
However, we could not reproduce the accuracy they reported for training on
MNIST with three fully connected layers. Later works (e.g., Wagh et al., 2019
and 2021) used the softmax replacement not for computing the output probability
distribution but for approximating the gradient in back-propagation. In this
work, we analyze the two uses of the replacement and compare them to softmax,
both in terms of accuracy and cost in multi-party computation. We found that
the replacement only provides a significant speed-up for a one-layer network
while it always reduces accuracy, sometimes significantly. Thus we conclude
that its usefulness is limited and one should use the original softmax function
instead.

    

### [[2011.12598] Energy Forecasting in Smart Grid Systems: A Review of the State-of-the-art Techniques](http://arxiv.org/abs/2011.12598)


  Energy forecasting has a vital role to play in smart grid (SG) systems
involving various applications such as demand-side management, load shedding,
and optimum dispatch. Managing efficient forecasting while ensuring the least
possible prediction error is one of the main challenges posed in the grid
today, considering the uncertainty and granularity in SG data. This paper
presents a comprehensive and application-oriented review of state-of-the-art
forecasting methods for SG systems along with recent developments in
probabilistic deep learning (PDL) considering different models and
architectures. Traditional point forecasting methods including statistical,
machine learning (ML), and deep learning (DL) are extensively investigated in
terms of their applicability to energy forecasting. In addition, the
significance of hybrid and data pre-processing techniques to support
forecasting performance is also studied. A comparative case study using the
Victorian electricity consumption and American electric power (AEP) datasets is
conducted to analyze the performance of point and probabilistic forecasting
methods. The analysis demonstrates higher accuracy of the long-short term
memory (LSTM) models with appropriate hyper-parameter tuning among point
forecasting methods especially when sample sizes are larger and involve
nonlinear patterns with long sequences. Furthermore, Bayesian bidirectional
LSTM (BLSTM) as a probabilistic method exhibit the highest accuracy in terms of
least pinball score and root mean square error (RMSE).

    

### [[2012.01981] Advanced Graph and Sequence Neural Networks for Molecular Property Prediction and Drug Discovery](http://arxiv.org/abs/2012.01981)


  Properties of molecules are indicative of their functions and thus are useful
in many applications. With the advances of deep learning methods, computational
approaches for predicting molecular properties are gaining increasing momentum.
However, there lacks customized and advanced methods and comprehensive tools
for this task currently. Here we develop a suite of comprehensive machine
learning methods and tools spanning different computational models, molecular
representations, and loss functions for molecular property prediction and drug
discovery. Specifically, we represent molecules as both graphs and sequences.
Built on these representations, we develop novel deep models for learning from
molecular graphs and sequences. In order to learn effectively from highly
imbalanced datasets, we develop advanced loss functions that optimize areas
under precision-recall curves. Altogether, our work not only serves as a
comprehensive tool, but also contributes towards developing novel and advanced
graph and sequence learning methodologies. Results on both online and offline
antibiotics discovery and molecular property prediction tasks show that our
methods achieve consistent improvements over prior methods. In particular, our
methods achieve #1 ranking in terms of both ROC-AUC and PRC-AUC on the AI Cures
Open Challenge for drug discovery related to COVID-19. Our software is released
as part of the MoleculeX library under AdvProp.

    

### [[2012.08508] Attention over learned object embeddings enables complex visual reasoning](http://arxiv.org/abs/2012.08508)


  Neural networks have achieved success in a wide array of perceptual tasks but
often fail at tasks involving both perception and higher-level reasoning. On
these more challenging tasks, bespoke approaches (such as modular symbolic
components, independent dynamics models or semantic parsers) targeted towards
that specific type of task have typically performed better. The downside to
these targeted approaches, however, is that they can be more brittle than
general-purpose neural networks, requiring significant modification or even
redesign according to the particular task at hand. Here, we propose a more
general neural-network-based approach to dynamic visual reasoning problems that
obtains state-of-the-art performance on three different domains, in each case
outperforming bespoke modular approaches tailored specifically to the task. Our
method relies on learned object-centric representations, self-attention and
self-supervised dynamics learning, and all three elements together are required
for strong performance to emerge. The success of this combination suggests that
there may be no need to trade off flexibility for performance on problems
involving spatio-temporal or causal-style reasoning. With the right soft biases
and learning objectives in a neural network we may be able to attain the best
of both worlds.

    

### [[2101.12501] Learning-based vs Model-free Adaptive Control of a MAV under Wind Gust](http://arxiv.org/abs/2101.12501)


  Navigation problems under unknown varying conditions are among the most
important and well-studied problems in the control field. Classic model-based
adaptive control methods can be applied only when a convenient model of the
plant or environment is provided. Recent model-free adaptive control methods
aim at removing this dependency by learning the physical characteristics of the
plant and/or process directly from sensor feedback. Although there have been
prior attempts at improving these techniques, it remains an open question as to
whether it is possible to cope with real-world uncertainties in a control
system that is fully based on either paradigm. We propose a conceptually simple
learning-based approach composed of a full state feedback controller, tuned
robustly by a deep reinforcement learning framework based on the Soft
Actor-Critic algorithm. We compare it, in realistic simulations, to a
model-free controller that uses the same deep reinforcement learning framework
for the control of a micro aerial vehicle under wind gust. The results indicate
the great potential of learning-based adaptive control methods in modern
dynamical systems.

    

### [[2102.06858] LTL2Action: Generalizing LTL Instructions for Multi-Task RL](http://arxiv.org/abs/2102.06858)


  We address the problem of teaching a deep reinforcement learning (RL) agent
to follow instructions in multi-task environments. Instructions are expressed
in a well-known formal language -- linear temporal logic (LTL) -- and can
specify a diversity of complex, temporally extended behaviours, including
conditionals and alternative realizations. Our proposed learning approach
exploits the compositional syntax and the semantics of LTL, enabling our RL
agent to learn task-conditioned policies that generalize to new instructions,
not observed during training. To reduce the overhead of learning LTL semantics,
we introduce an environment-agnostic LTL pretraining scheme which improves
sample-efficiency in downstream environments. Experiments on discrete and
continuous domains target combinatorial task sets of up to $\sim10^{39}$ unique
tasks and demonstrate the strength of our approach in learning to solve
(unseen) tasks, given LTL instructions.

    

### [[2102.10333] Provably Strict Generalisation Benefit for Equivariant Models](http://arxiv.org/abs/2102.10333)


  It is widely believed that engineering a model to be invariant/equivariant
improves generalisation. Despite the growing popularity of this approach, a
precise characterisation of the generalisation benefit is lacking. By
considering the simplest case of linear models, this paper provides the first
provably non-zero improvement in generalisation for invariant/equivariant
models when the target distribution is invariant/equivariant with respect to a
compact group. Moreover, our work reveals an interesting relationship between
generalisation, the number of training examples and properties of the group
action. Our results rest on an observation of the structure of function spaces
under averaging operators which, along with its consequences for feature
averaging, may be of independent interest.

    

### [[2102.12071] Learning optimal multigrid smoothers via neural networks](http://arxiv.org/abs/2102.12071)


  Multigrid methods are one of the most efficient techniques for solving linear
systems arising from Partial Differential Equations (PDEs) and graph Laplacians
from machine learning applications. One of the key components of multigrid is
smoothing, which aims at reducing high-frequency errors on each grid level.
However, finding optimal smoothing algorithms is problem-dependent and can
impose challenges for many problems. In this paper, we propose an efficient
adaptive framework for learning optimized smoothers from operator stencils in
the form of convolutional neural networks (CNNs). The CNNs are trained on
small-scale problems from a given type of PDEs based on a supervised loss
function derived from multigrid convergence theories, and can be applied to
large-scale problems of the same class of PDEs. Numerical results on
anisotropic rotated Laplacian problems demonstrate improved convergence rates
and solution time compared with classical hand-crafted relaxation methods.

    

### [[2103.01955] The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games](http://arxiv.org/abs/2103.01955)


  Proximal Policy Optimization (PPO) is a popular on-policy reinforcement
learning algorithm but is significantly less utilized than off-policy learning
algorithms in multi-agent settings. This is often due the belief that on-policy
methods are significantly less sample efficient than their off-policy
counterparts in multi-agent problems. In this work, we investigate Multi-Agent
PPO (MAPPO), a variant of PPO which is specialized for multi-agent settings.
Using a 1-GPU desktop, we show that MAPPO achieves surprisingly strong
performance in three popular multi-agent testbeds: the particle-world
environments, the Starcraft multi-agent challenge, and the Hanabi challenge,
with minimal hyperparameter tuning and without any domain-specific algorithmic
modifications or architectures. In the majority of environments, we find that
compared to off-policy baselines, MAPPO achieves strong results while
exhibiting comparable sample efficiency. Finally, through ablation studies, we
present the implementation and algorithmic factors which are most influential
to MAPPO's practical performance.

    

### [[2103.02138] Parametric Complexity Bounds for Approximating PDEs with Neural Networks](http://arxiv.org/abs/2103.02138)


  Recent experiments have shown that deep networks can approximate solutions to
high-dimensional PDEs, seemingly escaping the curse of dimensionality. However,
questions regarding the theoretical basis for such approximations, including
the required network size, remain open. In this paper, we investigate the
representational power of neural networks for approximating solutions to linear
elliptic PDEs with Dirichlet boundary conditions. We prove that when a PDE's
coefficients are representable by small neural networks, the parameters
required to approximate its solution scale polynomially with the input
dimension $d$ and proportionally to the parameter counts of the coefficient
networks. To this we end, we develop a proof technique that simulates gradient
descent (in an appropriate Hilbert space) by growing a neural network
architecture whose iterates each participate as sub-networks in their (slightly
larger) successors, and converge to the solution of the PDE. We bound the size
of the solution, showing a polynomial dependence on $d$ and no dependence on
the volume of the domain.

    

### [[2103.05045] Size-Invariant Graph Representations for Graph Classification Extrapolations](http://arxiv.org/abs/2103.05045)


  In general, graph representation learning methods assume that the train and
test data come from the same distribution. In this work we consider an
underexplored area of an otherwise rapidly developing field of graph
representation learning: The task of out-of-distribution (OOD) graph
classification, where train and test data have different distributions, with
test data unavailable during training. Our work shows it is possible to use a
causal model to learn approximately invariant representations that better
extrapolate between train and test data. Finally, we conclude with synthetic
and real-world dataset experiments showcasing the benefits of representations
that are invariant to train/test distribution shifts.

    

### [[2103.07295] Adversarial Graph Disentanglement](http://arxiv.org/abs/2103.07295)


  A real-world graph has a complex topological structure, which is often formed
by the interaction of different latent factors. Disentanglement of these latent
factors can effectively improve the robustness and expressiveness of node
representation of graph. However, most existing methods lack consideration of
the intrinsic differences in relations between nodes caused by factor
entanglement. In this paper, we propose an Adversarial Disentangled Graph
Convolutional Network (ADGCN) for disentangled graph representation learning.
Specifically, a component-specific aggregation approach is proposed to achieve
micro-disentanglement by inferring latent components that caused the links
between nodes. On the basis of micro-disentanglement, we further propose a
macro-disentanglement adversarial regularizer to improve the separability among
component distributions, thus restricting the interdependence among components.
Additionally, to reveal the topological graph structure, a diversity-preserving
node sampling approach is proposed, by which the graph structure can be
progressively refined in a way of local structure awareness. The experimental
results on various real-world graph data verify that our ADGCN obtains more
favorable performance over currently available alternatives.

    

### [[2103.08290] DeepOPG: Improving Orthopantomogram Finding Summarization with Weak Supervision](http://arxiv.org/abs/2103.08290)


  Clinical finding summaries from an orthopantomogram, or a dental panoramic
radiograph, have significant potential to improve patient communication and
speed up clinical judgments. While orthopantomogram is a first-line tool for
dental examinations, no existing work has explored the summarization of
findings from it. A finding summary has to find teeth in the imaging study and
label the teeth with several types of past treatments. To tackle the problem,
we developDeepOPG that breaks the summarization process into functional
segmentation and tooth localization, the latter of which is further refined by
a novel dental coherence module. We also leverage weak supervision labels to
improve detection results in a reinforcement learning scenario. Experiments
show high efficacy of DeepOPG on finding summarization, achieving an overall
AUC of 88.2% in detecting six types of findings. The proposed dental coherence
and weak supervision both are shown to improve DeepOPG by adding 5.9% and 0.4%
to AP@IoU=0.5.

    

### [[2104.01404] New Benchmarks for Learning on Non-Homophilous Graphs](http://arxiv.org/abs/2104.01404)


  Much data with graph structures satisfy the principle of homophily, meaning
that connected nodes tend to be similar with respect to a specific attribute.
As such, ubiquitous datasets for graph machine learning tasks have generally
been highly homophilous, rewarding methods that leverage homophily as an
inductive bias. Recent work has pointed out this particular focus, as new
non-homophilous datasets have been introduced and graph representation learning
models better suited for low-homophily settings have been developed. However,
these datasets are small and poorly suited to truly testing the effectiveness
of new methods in non-homophilous settings. We present a series of improved
graph datasets with node label relationships that do not satisfy the homophily
principle. Along with this, we introduce a new measure of the presence or
absence of homophily that is better suited than existing measures in different
regimes. We benchmark a range of simple methods and graph neural networks
across our proposed datasets, drawing new insights for further research. Data
and codes can be found at this https URL.

    

### [[2104.09460] Bayesian Algorithm Execution: Estimating Computable Properties of Black-box Functions Using Mutual Information](http://arxiv.org/abs/2104.09460)


  In many real-world problems, we want to infer some property of an expensive
black-box function $f$, given a budget of $T$ function evaluations. One example
is budget constrained global optimization of $f$, for which Bayesian
optimization is a popular method. Other properties of interest include local
optima, level sets, integrals, or graph-structured information induced by $f$.
Often, we can find an algorithm $\mathcal{A}$ to compute the desired property,
but it may require far more than $T$ queries to execute. Given such an
$\mathcal{A}$, and a prior distribution over $f$, we refer to the problem of
inferring the output of $\mathcal{A}$ using $T$ evaluations as Bayesian
Algorithm Execution (BAX). To tackle this problem, we present a procedure,
InfoBAX, that sequentially chooses queries that maximize mutual information
with respect to the algorithm's output. Applying this to Dijkstra's algorithm,
for instance, we infer shortest paths in synthetic and real-world graphs with
black-box edge costs. Using evolution strategies, we yield variants of Bayesian
optimization that target local, rather than global, optima. On these problems,
InfoBAX uses up to 500 times fewer queries to $f$ than required by the original
algorithm. Our method is closely connected to other Bayesian optimal
experimental design procedures such as entropy search methods and optimal
sensor placement using Gaussian processes.

    

### [[2104.14129] ActNN: Reducing Training Memory Footprint via 2-Bit Activation Compressed Training](http://arxiv.org/abs/2104.14129)


  The increasing size of neural network models has been critical for
improvements in their accuracy, but device memory is not growing at the same
rate. This creates fundamental challenges for training neural networks within
limited memory environments. In this work, we propose ActNN, a memory-efficient
training framework that stores randomly quantized activations for back
propagation. We prove the convergence of ActNN for general network
architectures, and we characterize the impact of quantization on the
convergence via an exact expression for the gradient variance. Using our
theory, we propose novel mixed-precision quantization strategies that exploit
the activation's heterogeneity across feature dimensions, samples, and layers.
These techniques can be readily applied to existing dynamic graph frameworks,
such as PyTorch, simply by substituting the layers. We evaluate ActNN on
mainstream computer vision models for classification, detection, and
segmentation tasks. On all these tasks, ActNN compresses the activation to 2
bits on average, with negligible accuracy loss. ActNN reduces the memory
footprint of the activation by 12x, and it enables training with a 6.6x to 14x
larger batch size.

    

### [[2105.03287] Order in the Court: Explainable AI Methods Prone to Disagreement](http://arxiv.org/abs/2105.03287)


  By computing the rank correlation between attention weights and
feature-additive explanation methods, previous analyses either invalidate or
support the role of attention-based explanations as a faithful and plausible
measure of salience. To investigate whether this approach is appropriate, we
compare LIME, Integrated Gradients, DeepLIFT, Grad-SHAP, Deep-SHAP, and
attention-based explanations, applied to two neural architectures trained on
single- and pair-sequence language tasks. In most cases, we find that none of
our chosen methods agree. Based on our empirical observations and theoretical
objections, we conclude that rank correlation does not measure the quality of
feature-additive methods. Practitioners should instead use the numerous and
rigorous diagnostic methods proposed by the community.

    

### [[2105.03308] Geometric convergence of elliptical slice sampling](http://arxiv.org/abs/2105.03308)


  For Bayesian learning, given likelihood function and Gaussian prior, the
elliptical slice sampler, introduced by Murray, Adams and MacKay 2010, provides
a tool for the construction of a Markov chain for approximate sampling of the
underlying posterior distribution. Besides of its wide applicability and
simplicity its main feature is that no tuning is necessary. Under weak
regularity assumptions on the posterior density we show that the corresponding
Markov chain is geometrically ergodic and therefore yield qualitative
convergence guarantees. We illustrate our result for Gaussian posteriors as
they appear in Gaussian process regression, as well as in a setting of a
multi-modal distribution. Remarkably, our numerical experiments indicate a
dimension-independent performance of elliptical slice sampling even in
situations where our ergodicity result does not apply.

    

### [[2105.05495] LipBaB: Computing exact Lipschitz constant of ReLU networks](http://arxiv.org/abs/2105.05495)


  The Lipschitz constant of neural networks plays an important role in several
contexts of deep learning ranging from robustness certification and
regularization to stability analysis of systems with neural network
controllers. Obtaining tight bounds of the Lipschitz constant is therefore
important. We introduce LipBaB, a branch and bound framework to compute
certified bounds of the local Lipschitz constant of deep neural networks with
ReLU activation functions up to any desired precision. We achieve this by
bounding the norm of the Jacobians, corresponding to different activation
patterns of the network caused within the input domain. Our algorithm can
provide provably exact computation of the Lipschitz constant for any p-norm.

    

### [[2105.14625] Surrogate Model Based Hyperparameter Tuning for Deep Learning with SPOT](http://arxiv.org/abs/2105.14625)


  A surrogate model based hyperparameter tuning approach for deep learning is
presented. This article demonstrates how the architecture-level parameters
(hyperparameters) of deep learning models that were implemented in
Keras/tensorflow can be optimized. The implementation of the tuning procedure
is 100% accessible from R, the software environment for statistical computing.
With a few lines of code, existing R packages (tfruns and SPOT) can be combined
to perform hyperparameter tuning. An elementary hyperparameter tuning task
(neural network and the MNIST data) is used to exemplify this approach

    

### [[2106.05466] Adaptive machine learning for protein engineering](http://arxiv.org/abs/2106.05466)


  Machine-learning models that learn from data to predict how protein sequence
encodes function are emerging as a useful protein engineering tool. However,
when using these models to suggest new protein designs, one must deal with the
vast combinatorial complexity of protein sequences. Here, we review how to use
a sequence-to-function machine-learning surrogate model to select sequences for
experimental measurement. First, we discuss how to select sequences through a
single round of machine-learning optimization. Then, we discuss sequential
optimization, where the goal is to discover optimized sequences and improve the
model across multiple rounds of training, optimization, and experimental
measurement.

    

### [[2106.05876] Data Fusion for Deep Learning on Transport Mode Detection: A Case Study](http://arxiv.org/abs/2106.05876)


  In Transport Mode Detection, a great diversity of methodologies exist
according to the choice made on sensors, preprocessing, model used, etc. In
this domain, the comparisons between each option are not always complete.
Experiments on a public, real-life dataset are led here to evaluate carefully
each of the choices that were made, with a specific emphasis on data fusion
methods. Our most surprising finding is that none of the methods we implemented
from the literature is better than a simple late fusion. Two important
decisions are the choice of a sensor and the choice of a representation for the
data: we found that using 2D convolutions on spectrograms with a logarithmic
axis for the frequencies was better than 1-dimensional temporal
representations.

    

### [[2106.05907] Disentangled Attention as Intrinsic Regularization for Bimanual Multi-Object Manipulation](http://arxiv.org/abs/2106.05907)


  We address the problem of solving complex bimanual robot manipulation tasks
on multiple objects with sparse rewards. Such complex tasks can be decomposed
into sub-tasks that are accomplishable by different robots concurrently or
sequentially for better efficiency. While previous reinforcement learning
approaches primarily focus on modeling the compositionality of sub-tasks, two
fundamental issues are largely ignored particularly when learning cooperative
strategies for two robots: (i) domination, i.e., one robot may try to solve a
task by itself and leaves the other idle; (ii) conflict, i.e., one robot can
easily interrupt another's workspace when executing different sub-tasks
simultaneously. To tackle these two issues, we propose a novel technique called
disentangled attention, which provides an intrinsic regularization for two
robots to focus on separate sub-tasks and objects. We evaluate our method on
four bimanual manipulation tasks. Experimental results show that our proposed
intrinsic regularization successfully avoids domination and reduces conflicts
for the policies, which leads to significantly more effective cooperative
strategies than all the baselines. Our project page with videos is at
this https URL.

    

### [[2106.06613] A New Formalism, Method and Open Issues for Zero-Shot Coordination](http://arxiv.org/abs/2106.06613)


  In many coordination problems, independently reasoning humans are able to
discover mutually compatible policies. In contrast, independently trained
self-play policies are often mutually incompatible. Zero-shot coordination
(ZSC) has recently been proposed as a new frontier in multi-agent reinforcement
learning to address this fundamental issue. Prior work approaches the ZSC
problem by assuming players can agree on a shared learning algorithm but not on
labels for actions and observations, and proposes other-play as an optimal
solution. However, until now, this "label-free" problem has only been
informally defined. We formalize this setting as the label-free coordination
(LFC) problem by defining the label-free coordination game. We show that
other-play is not an optimal solution to the LFC problem as it fails to
consistently break ties between incompatible maximizers of the other-play
objective. We introduce an extension of the algorithm, other-play with
tie-breaking, and prove that it is optimal in the LFC problem and an
equilibrium in the LFC game. Since arbitrary tie-breaking is precisely what the
ZSC setting aims to prevent, we conclude that the LFC problem does not reflect
the aims of ZSC. To address this, we introduce an alternative informal
operationalization of ZSC as a starting point for future work.

    

### [[2106.07385] SemEval-2021 Task 11: NLPContributionGraph -- Structuring Scholarly NLP Contributions for a Research Knowledge Graph](http://arxiv.org/abs/2106.07385)


  There is currently a gap between the natural language expression of scholarly
publications and their structured semantic content modeling to enable
intelligent content search. With the volume of research growing exponentially
every year, a search feature operating over semantically structured content is
compelling. The SemEval-2021 Shared Task NLPContributionGraph (a.k.a. 'the NCG
task') tasks participants to develop automated systems that structure
contributions from NLP scholarly articles in the English language. Being the
first-of-its-kind in the SemEval series, the task released structured data from
NLP scholarly articles at three levels of information granularity, i.e. at
sentence-level, phrase-level, and phrases organized as triples toward Knowledge
Graph (KG) building. The sentence-level annotations comprised the few sentences
about the article's contribution. The phrase-level annotations were scientific
term and predicate phrases from the contribution sentences. Finally, the
triples constituted the research overview KG. For the Shared Task,
participating systems were then expected to automatically classify contribution
sentences, extract scientific terms and relations from the sentences, and
organize them as KG triples.
Overall, the task drew a strong participation demographic of seven teams and
27 participants. The best end-to-end task system classified contribution
sentences at 57.27% F1, phrases at 46.41% F1, and triples at 22.28% F1. While
the absolute performance to generate triples remains low, in the conclusion of
this article, the difficulty of producing such data and as a consequence of
modeling it is highlighted.

    

### [[2106.10419] Predicting Critical Nodes in Temporal Networks by Dynamic Graph Convolutional Networks](http://arxiv.org/abs/2106.10419)


  Many real-world systems can be expressed in temporal networks with nodes
playing far different roles in structure and function and edges representing
the relationships between nodes. Identifying critical nodes can help us control
the spread of public opinions or epidemics, predict leading figures in
academia, conduct advertisements for various commodities, and so on. However,
it is rather difficult to identify critical nodes because the network structure
changes over time in temporal networks. In this paper, considering the sequence
topological information of temporal networks, a novel and effective learning
framework based on the combination of special GCNs and RNNs is proposed to
identify nodes with the best spreading ability. The effectiveness of the
approach is evaluated by weighted Susceptible-Infected-Recovered model.
Experimental results on four real-world temporal networks demonstrate that the
proposed method outperforms both traditional and deep learning benchmark
methods in terms of the Kendall $\tau$ coefficient and top $k$ hit rate.

    

### [[2106.11702] Categorising Fine-to-Coarse Grained Misinformation: An Empirical Study of COVID-19 Infodemic](http://arxiv.org/abs/2106.11702)


  The spreading COVID-19 misinformation over social media already draws the
attention of many researchers. According to Google Scholar, about 26000
COVID-19 related misinformation studies have been published to date. Most of
these studies focusing on 1) detect and/or 2) analysing the characteristics of
COVID-19 related misinformation. However, the study of the social behaviours
related to misinformation is often neglected. In this paper, we introduce a
fine-grained annotated misinformation tweets dataset including social
behaviours annotation (e.g. comment or question to the misinformation). The
dataset not only allows social behaviours analysis but also suitable for both
evidence-based or non-evidence-based misinformation classification task. In
addition, we introduce leave claim out validation in our experiments and
demonstrate the misinformation classification performance could be
significantly different when applying to real-world unseen misinformation.

    

### [[2106.14565] Variance Reduction for Matrix Computations with Applications to Gaussian Processes](http://arxiv.org/abs/2106.14565)


  In addition to recent developments in computing speed and memory,
methodological advances have contributed to significant gains in the
performance of stochastic simulation. In this paper, we focus on variance
reduction for matrix computations via matrix factorization. We provide insights
into existing variance reduction methods for estimating the entries of large
matrices. Popular methods do not exploit the reduction in variance that is
possible when the matrix is factorized. We show how computing the square root
factorization of the matrix can achieve in some important cases arbitrarily
better stochastic performance. In addition, we propose a factorized estimator
for the trace of a product of matrices and numerically demonstrate that the
estimator can be up to 1,000 times more efficient on certain problems of
estimating the log-likelihood of a Gaussian process. Additionally, we provide a
new estimator of the log-determinant of a positive semi-definite matrix where
the log-determinant is treated as a normalizing constant of a probability
density.

    

### [[2107.00520] Predictive Modeling in the Presence of Nuisance-Induced Spurious Correlations](http://arxiv.org/abs/2107.00520)


  Deep predictive models often make use of spurious correlations between the
label and the covariates that differ between training and test distributions.
In many classification tasks, spurious correlations are induced by a changing
relationship between the label and some nuisance variables correlated with the
covariates. For example, in classifying animals in natural images, the
background, which is the nuisance, can predict the type of animal. This
nuisance-label relationship does not always hold. We formalize a family of
distributions that only differ in the nuisance-label relationship and introduce
a distribution where this relationship is broken called the nuisance-randomized
distribution. We introduce a set of predictive models built from the
nuisance-randomized distribution with representations, that when conditioned
on, do not correlate the label and the nuisance. For models in this set, we
lower bound the performance for any member of the family with the mutual
information between the representation and the label under the
nuisance-randomized distribution. To build predictive models that maximize the
performance lower bound, we develop Nuisance-Randomized Distillation (NURD). We
evaluate NURD on a synthetic example, colored-MNIST, and classifying chest
X-rays. When using non-lung patches as the nuisance in classifying chest
X-rays, NURD produces models that predict pneumonia under strong spurious
correlations.

    

### [[2107.00606] Action Transformer: A Self-Attention Model for Short-Time Human Action Recognition](http://arxiv.org/abs/2107.00606)


  Deep neural networks based purely on attention have been successful across
several domains, relying on minimal architectural priors from the designer. In
Human Action Recognition (HAR), attention mechanisms have been primarily
adopted on top of standard convolutional or recurrent layers, improving the
overall generalization capability. In this work, we introduce Action
Transformer (AcT), a simple, fully self-attentional architecture that
consistently outperforms more elaborated networks that mix convolutional,
recurrent, and attentive layers. In order to limit computational and energy
requests, building on previous human action recognition research, the proposed
approach exploits 2D pose representations over small temporal windows,
providing a low latency solution for accurate and effective real-time
performance. Moreover, we open-source MPOSE2021, a new large-scale dataset, as
an attempt to build a formal training and evaluation benchmark for real-time
short-time human action recognition. Extensive experimentation on MPOSE2021
with our proposed methodology and several previous architectural solutions
proves the effectiveness of the AcT model and poses the base for future work on
HAR.

    

### [[2107.00956] SocialAI: Benchmarking Socio-Cognitive Abilities in Deep Reinforcement Learning Agents](http://arxiv.org/abs/2107.00956)


  Building embodied autonomous agents capable of participating in social
interactions with humans is one of the main challenges in AI. Within the Deep
Reinforcement Learning (DRL) field, this objective motivated multiple works on
embodied language use. However, current approaches focus on language as a
communication tool in very simplified and non-diverse social situations: the
"naturalness" of language is reduced to the concept of high vocabulary size and
variability. In this paper, we argue that aiming towards human-level AI
requires a broader set of key social skills: 1) language use in complex and
variable social contexts; 2) beyond language, complex embodied communication in
multimodal settings within constantly evolving social worlds. We explain how
concepts from cognitive sciences could help AI to draw a roadmap towards
human-like intelligence, with a focus on its social dimensions. As a first
step, we propose to expand current research to a broader set of core social
skills. To do this, we present SocialAI, a benchmark to assess the acquisition
of social skills of DRL agents using multiple grid-world environments featuring
other (scripted) social agents. We then study the limits of a recent SOTA DRL
approach when tested on SocialAI and discuss important next steps towards
proficient social agents. Videos and code are available at
this https URL.

    

### [[2106.13435] NP-DRAW: A Non-Parametric Structured Latent Variable Model for Image Generation](http://arxiv.org/abs/2106.13435)


  In this paper, we present a non-parametric structured latent variable model
for image generation, called NP-DRAW, which sequentially draws on a latent
canvas in a part-by-part fashion and then decodes the image from the canvas.
Our key contributions are as follows. 1) We propose a non-parametric prior
distribution over the appearance of image parts so that the latent variable
``what-to-draw'' per step becomes a categorical random variable. This improves
the expressiveness and greatly eases the learning compared to Gaussians used in
the literature. 2) We model the sequential dependency structure of parts via a
Transformer, which is more powerful and easier to train compared to RNNs used
in the literature. 3) We propose an effective heuristic parsing algorithm to
pre-train the prior. Experiments on MNIST, Omniglot, CIFAR-10, and CelebA show
that our method significantly outperforms previous structured image models like
DRAW and AIR and is competitive to other generic generative models. Moreover,
we show that our model's inherent compositionality and interpretability bring
significant benefits in the low-data learning regime and latent space editing.
Code is available at this https URL.

    

### [[2107.02547] Energy-Efficient Accelerator Design for Deformable Convolution Networks](http://arxiv.org/abs/2107.02547)


  Deformable convolution networks (DCNs) proposed to address the image
recognition with geometric or photometric variations typically involve
deformable convolution that convolves on arbitrary locations of input features.
The locations change with different inputs and induce considerable dynamic and
irregular memory accesses which cannot be handled by classic neural network
accelerators (NNAs). Moreover, bilinear interpolation (BLI) operation that is
required to obtain deformed features in DCNs also cannot be deployed on
existing NNAs directly. Although a general purposed processor (GPP) seated
along with classic NNAs can process the deformable convolution, the processing
on GPP can be extremely slow due to the lack of parallel computing capability.
To address the problem, we develop a DCN accelerator on existing NNAs to
support both the standard convolution and deformable convolution. Specifically,
for the dynamic and irregular accesses in DCNs, we have both the input and
output features divided into tiles and build a tile dependency table (TDT) to
track the irregular tile dependency at runtime. With the TDT, we further
develop an on-chip tile scheduler to handle the dynamic and irregular accesses
efficiently. In addition, we propose a novel mapping strategy to enable
parallel BLI processing on NNAs and apply layer fusion techniques for more
energy-efficient DCN processing. According to our experiments, the proposed
accelerator achieves orders of magnitude higher performance and energy
efficiency compared to the typical computing architectures including ARM,
ARM+TPU, and GPU with 6.6\% chip area penalty to a classic NNA.

    

### [[2107.02762] Area-Delay-Efficeint FPGA Design of 32-bit Euclid's GCD based on Sum of Absolute Difference](http://arxiv.org/abs/2107.02762)


  Euclids algorithm is widely used in calculating of GCD (Greatest Common
Divisor) of two positive numbers. There are various fields where this division
is used such as channel coding, cryptography, and error correction codes. This
makes the GCD a fundamental algorithm in number theory, so a number of methods
have been discovered to efficiently compute it. The main contribution of this
paper is to investigate a method that computes the GCD of two 32-bit numbers
based on Euclidean algorithm which targets six different Xilinx chips. The
complexity of this method that we call Optimized_GCDSAD is achieved by
utilizing Sum of Absolute Difference (SAD) block which is based on a fast
carry-out generation function. The efficiency of the proposed architecture is
evaluated based on criteria such as time (latency), area delay product (ADP)
and space (slice number) complexity. The VHDL codes of these architectures have
been implemented and synthesized through ISE 14.7. A detailed comparative
analysis indicates that the proposed Optimized_GCDSAD method based on SAD block
outperforms previously known results.

    

### [[2105.03725] DAMOV: A New Methodology and Benchmark Suite for Evaluating Data Movement Bottlenecks](http://arxiv.org/abs/2105.03725)


  Data movement between the CPU and main memory is a first-order obstacle
against improving performance, scalability, and energy efficiency in modern
systems. Computer systems employ a range of techniques to reduce overheads tied
to data movement, spanning from traditional mechanisms (e.g., deep multi-level
cache hierarchies, aggressive hardware prefetchers) to emerging techniques such
as Near-Data Processing (NDP), where some computation is moved close to memory.
Our goal is to methodically identify potential sources of data movement over a
broad set of applications and to comprehensively compare traditional
compute-centric data movement mitigation techniques to more memory-centric
techniques, thereby developing a rigorous understanding of the best techniques
to mitigate each source of data movement.
With this goal in mind, we perform the first large-scale characterization of
a wide variety of applications, across a wide range of application domains, to
identify fundamental program properties that lead to data movement to/from main
memory. We develop the first systematic methodology to classify applications
based on the sources contributing to data movement bottlenecks. From our
large-scale characterization of 77K functions across 345 applications, we
select 144 functions to form the first open-source benchmark suite (DAMOV) for
main memory data movement studies. We select a diverse range of functions that
(1) represent different types of data movement bottlenecks, and (2) come from a
wide range of application domains. Using NDP as a case study, we identify new
insights about the different data movement bottlenecks and use these insights
to determine the most suitable data movement mitigation mechanism for a
particular application. We open-source DAMOV and the complete source code for
our new characterization methodology at this https URL.

    

### [[2105.03814] Benchmarking a New Paradigm: An Experimental Analysis of a Real Processing-in-Memory Architecture](http://arxiv.org/abs/2105.03814)


  Many modern workloads, such as neural networks, databases, and graph
processing, are fundamentally memory-bound. For such workloads, the data
movement between main memory and CPU cores imposes a significant overhead in
terms of both latency and energy. A major reason is that this communication
happens through a narrow bus with high latency and limited bandwidth, and the
low data reuse in memory-bound workloads is insufficient to amortize the cost
of main memory access. Fundamentally addressing this data movement bottleneck
requires a paradigm where the memory system assumes an active role in computing
by integrating processing capabilities. This paradigm is known as
processing-in-memory (PIM).
Recent research explores different forms of PIM architectures, motivated by
the emergence of new 3D-stacked memory technologies that integrate memory with
a logic layer where processing elements can be easily placed. Past works
evaluate these architectures in simulation or, at best, with simplified
hardware prototypes. In contrast, the UPMEM company has designed and
manufactured the first publicly-available real-world PIM architecture.
This paper provides the first comprehensive analysis of the first
publicly-available real-world PIM architecture. We make two key contributions.
First, we conduct an experimental characterization of the UPMEM-based PIM
system using microbenchmarks to assess various architecture limits such as
compute throughput and memory bandwidth, yielding new insights. Second, we
present PrIM, a benchmark suite of 16 workloads from different application
domains (e.g., linear algebra, databases, graph processing, neural networks,
bioinformatics).

    

### [[2107.02426] Sustaining Performance While Reducing Energy Consumption: A Control Theory Approach](http://arxiv.org/abs/2107.02426)


  Production high-performance computing systems continue to grow in complexity
and size. As applications struggle to make use of increasingly heterogeneous
compute nodes, maintaining high efficiency (performance per watt) for the whole
platform becomes a challenge. Alongside the growing complexity of scientific
workloads, this extreme heterogeneity is also an opportunity: as applications
dynamically undergo variations in workload, due to phases or data/compute
movement between devices, one can dynamically adjust power across compute
elements to save energy without impacting performance. With an aim toward an
autonomous and dynamic power management strategy for current and future HPC
architectures, this paper explores the use of control theory for the design of
a dynamic power regulation method. Structured as a feedback loop, our
approach-which is novel in computing resource management-consists of
periodically monitoring application progress and choosing at runtime a suitable
power cap for processors. Thanks to a preliminary offline identification
process, we derive a model of the dynamics of the system and a
proportional-integral (PI) controller. We evaluate our approach on top of an
existing resource management framework, the Argo Node Resource Manager,
deployed on several clusters of Grid'5000, using a standard memory-bound HPC
benchmark.

    

### [[2107.02466] On-edge Multi-task Transfer Learning: Model and Practice with Data-driven Task Allocation](http://arxiv.org/abs/2107.02466)


  On edge devices, data scarcity occurs as a common problem where transfer
learning serves as a widely-suggested remedy. Nevertheless, transfer learning
imposes a heavy computation burden to resource-constrained edge devices.
Existing task allocation works usually assume all submitted tasks are equally
important, leading to inefficient resource allocation at a task level when
directly applied in Multi-task Transfer Learning (MTL). To address these
issues, we first reveal that it is crucial to measure the impact of tasks on
overall decision performance improvement and quantify \emph{task importance}.
We then show that task allocation with task importance for MTL (TATIM) is a
variant of the NP-complete Knapsack problem, where the complicated computation
to solve this problem needs to be conducted repeatedly under varying contexts.
To solve TATIM with high computational efficiency, we propose a Data-driven
Cooperative Task Allocation (DCTA) approach. Finally, we evaluate the
performance of DCTA by not only a trace-driven simulation, but also a new
comprehensive real-world AIOps case study that bridges model and practice via a
new architecture and main components design within the AIOps system. Extensive
experiments show that our DCTA reduces 3.24 times of processing time, and saves
48.4\% energy consumption compared with the state-of-the-art when solving
TATIM.

    

### [[2107.02539] An MPI-based Algorithm for Mapping Complex Networks onto Hierarchical Architectures](http://arxiv.org/abs/2107.02539)


  Processing massive application graphs on distributed memory systems requires
to map the graphs onto the system's processing elements (PEs). This task
becomes all the more important when PEs have non-uniform communication costs or
the input is highly irregular. Typically, mapping is addressed using
partitioning, in a two-step approach or an integrated one. Parallel
partitioning tools do exist; yet, corresponding mapping algorithms or their
public implementations all have major sequential parts or other severe scaling
limitations. In this paper, we propose a parallel algorithm that maps graphs
onto the PEs of a hierarchical system. Our solution integrates partitioning and
mapping; it models the system hierarchy in a concise way as an implicit labeled
tree. The vertices of the application graph are labeled as well, and these
vertex labels induce the mapping. The mapping optimization follows the basic
idea of parallel label propagation, but we tailor the gain computations of
label changes to quickly account for the induced communication costs. Our
MPI-based code is the first public implementation of a parallel graph mapping
algorithm; to this end, we extend the partitioning library ParHIP. To evaluate
our algorithm's implementation, we perform comparative experiments with complex
networks in the million- and billion-scale range. In general our mapping tool
shows good scalability on up to a few thousand PEs. Compared to other MPI-based
competitors, our algorithm achieves the best speed to quality trade-off and our
quality results are even better than non-parallel mapping tools.

    

### [[2107.02769] Exploring a Dynamic Ring without Landmark](http://arxiv.org/abs/2107.02769)


  Consider a group of autonomous mobile computational entities, called agents,
arbitrarily placed at some nodes of a dynamic but always connected ring. The
agents neither have any knowledge about the size of the ring nor have a common
notion of orientation. We consider the \textsc{Exploration} problem where the
agents have to collaboratively to explore the graph and terminate, with the
requirement that each node has to be visited by at least one agent. It has been
shown by Di Luna et al. [Distrib. Comput. 2020] that the problem is solvable by
two anonymous agents if there is a single observably different node in the ring
called landmark node. The problem is unsolvable by any number of anonymous
agents in absence of a landmark node. We consider the problem with
non-anonymous agents (agents with distinct identifiers) in a ring with no
landmark node. The assumption of agents with distinct identifiers is strictly
weaker than having a landmark node as the problem is unsolvable by two agents
with distinct identifiers in absence of a landmark node. This setting has been
recently studied by Mandal et al. [ALGOSENSORS 2020]. There it is shown that
the problem is solvable in this setting by three agents assuming that they have
edge crossing detection capability. Edge crossing detection capability is a
strong assumption which enables two agents moving in opposite directions
through an edge in the same round to detect each other and also exchange
information. In this paper we give an algorithm that solves the problem with
three agents without the edge crossing detection capability.

    

### [[1902.10041] Population protocols with unreliable communication](http://arxiv.org/abs/1902.10041)


  Population protocols are a model of distributed computation intended for the
study of networks of independent computing agents with dynamic communication
structure. Each agent has a finite number of states, and communication
opportunities occur nondeterministically, allowing the agents involved to
change their states based on each other's states.
In the present paper we study unreliable models based on population protocols
and their variations from the point of view of expressive power. We model the
effects of message loss. We show that for a general definition of unreliable
protocols with constant-storage agents such protocols can only compute
predicates computable by immediate observation population protocols (sometimes
also called one-way protocols). Immediate observation population protocols are
inherently tolerant of unreliable communication and keep their expressive power
under a wide range of fairness conditions. We also prove that a large class of
message-based models that are generally more expressive than immediate
observation becomes strictly less expressive than immediate observation in the
unreliable case.

    

### [[2012.15422] An Order-Aware Dataflow Model for Parallel Unix Pipelines](http://arxiv.org/abs/2012.15422)


  We present a dataflow model for modelling parallel Unix shell pipelines. To
accurately capture the semantics of complex Unix pipelines, the dataflow model
is order-aware, i.e., the order in which a node in the dataflow graph consumes
inputs from different edges plays a central role in the semantics of the
computation and therefore in the resulting parallelization. We use this model
to capture the semantics of transformations that exploit data parallelism
available in Unix shell computations and prove their correctness. We
additionally formalize the translations from the Unix shell to the dataflow
model and from the dataflow model back to a parallel shell script. We implement
our model and transformations as the compiler and optimization passes of a
system parallelizing shell pipelines, and use it to evaluate the speedup
achieved on 47 pipelines.

    

### [[2102.01970] Efficient Byzantine Fault Tolerance using Trusted Execution Environment: Preventing Equivocation is only the Beginning](http://arxiv.org/abs/2102.01970)


  With the rapid development of blockchain, Byzantine fault-tolerant protocols
have attracted revived interest recently. To overcome the theoretical bounds of
Byzantine fault tolerance, many protocols attempt to use Trusted Execution
Environment (TEE) to prevent equivocation and improve performance and
scalability. However, due to the broken quorum intersection assumption caused
by the reduction of the replica number, the improvement is mostly at the cost
of increased communication complexity which prevents existing TEE-based
protocols to be applied to large-scale blockchain systems. In this paper, we
propose TBFT, an efficient Byzantine fault-tolerant protocol in the partial
synchrony setting, which has O(n) message complexity in both normal-case and
view-change. Compared to previous protocols, TBFT uses TEE-assisted primitives
to limit more types of malicious behaviors of replicas rather than preventing
equivocation only, thereby reducing the latency and communication complexity of
clients and replicas. Besides, we also introduce lightweight cryptographic
primitives including a novel leader election mechanism and an efficient voting
message aggregation mechanism for better security and performance. We evaluate
TBFT via systematic analysis and experiments, and the results show that TBFT
has better performance and scalability compared to other protocols.

    

### [[2103.09655] The Old and the New: Can Physics-Informed Deep-Learning Replace Traditional Linear Solvers?](http://arxiv.org/abs/2103.09655)


  Physics-Informed Neural Networks (PINN) are neural networks encoding the
problem governing equations, such as Partial Differential Equations (PDE), as a
part of the neural network. PINNs have emerged as a new essential tool to solve
various challenging problems, including computing linear systems arising from
PDEs, a task for which several traditional methods exist. In this work, we
focus first on evaluating the potential of PINNs as linear solvers in the case
of the Poisson equation, an omnipresent equation in scientific computing. We
characterize PINN linear solvers in terms of accuracy and performance under
different network configurations (depth, activation functions, input data set
distribution). We highlight the critical role of transfer learning. Our results
show that low-frequency components of the solution converge quickly as an
effect of the F-principle. In contrast, an accurate solution of the high
frequencies requires an exceedingly long time. To address this limitation, we
propose integrating PINNs into traditional linear solvers. We show that this
integration leads to the development of new solvers whose performance is on par
with other high-performance solvers, such as PETSc conjugate gradient linear
solvers, in terms of performance and accuracy. Overall, while the accuracy and
computational performance are still a limiting factor for the direct use of
PINN linear solvers, hybrid strategies combining old traditional linear solver
approaches with new emerging deep-learning techniques are among the most
promising methods for developing a new class of linear solvers.

    

### [[2107.02175] Identifying negativity factors from social media text corpus using sentiment analysis method](http://arxiv.org/abs/2107.02175)


  Automatic sentiment analysis play vital role in decision making. Many
organizations spend a lot of budget to understand their customer satisfaction
by manually going over their feedback/comments or tweets. Automatic sentiment
analysis can give overall picture of the comments received against any event,
product, or activity. Usually, the comments/tweets are classified into two main
classes that are negative or positive. However, the negative comments are too
abstract to understand the basic reason or the context. organizations are
interested to identify the exact reason for the negativity. In this research
study, we hierarchically goes down into negative comments, and link them with
more classes. Tweets are extracted from social media sites such as Twitter and
Facebook. If the sentiment analysis classifies any tweet into negative class,
then we further try to associates that negative comments with more possible
negative classes. Based on expert opinions, the negative comments/tweets are
further classified into 8 classes. Different machine learning algorithms are
evaluated and their accuracy are reported.

    

### [[2107.02202] An Evolutionary Algorithm for Task Scheduling in Crowdsourced Software Development](http://arxiv.org/abs/2107.02202)


  The complexity of software tasks and the uncertainty of crowd developer
behaviors make it challenging to plan crowdsourced software development (CSD)
projects. In a competitive crowdsourcing marketplace, competition for shared
worker resources from multiple simultaneously open tasks adds another layer of
uncertainty to the potential outcomes of software crowdsourcing. These factors
lead to the need for supporting CSD managers with automated scheduling to
improve the visibility and predictability of crowdsourcing processes and
outcomes. To that end, this paper proposes an evolutionary algorithm-based task
scheduling method for crowdsourced software development. The proposed
evolutionary scheduling method uses a multiobjective genetic algorithm to
recommend an optimal task start date. The method uses three fitness functions,
based on project duration, task similarity, and task failure prediction,
respectively. The task failure fitness function uses a neural network to
predict the probability of task failure with respect to a specific task start
date. The proposed method then recommends the best tasks start dates for the
project as a whole and each individual task so as to achieve the lowest project
failure ratio. Experimental results on 4 projects demonstrate that the proposed
method has the potential to reduce project duration by a factor of 33-78%.

    

### [[2107.02282] Weakly Supervised Named Entity Tagging with Learnable Logical Rules](http://arxiv.org/abs/2107.02282)


  We study the problem of building entity tagging systems by using a few rules
as weak supervision. Previous methods mostly focus on disambiguation entity
types based on contexts and expert-provided rules, while assuming entity spans
are given. In this work, we propose a novel method TALLOR that bootstraps
high-quality logical rules to train a neural tagger in a fully automated
manner. Specifically, we introduce compound rules that are composed from simple
rules to increase the precision of boundary detection and generate more diverse
pseudo labels. We further design a dynamic label selection strategy to ensure
pseudo label quality and therefore avoid overfitting the neural tagger.
Experiments on three datasets demonstrate that our method outperforms other
weakly supervised methods and even rivals a state-of-the-art distantly
supervised tagger with a lexicon of over 2,000 terms when starting from only 20
simple rules. Our method can serve as a tool for rapidly building taggers in
emerging domains and tasks. Case studies show that learned rules can
potentially explain the predicted entities.

    

### [[2107.02298] Knowledge Modelling and Active Learning in Manufacturing](http://arxiv.org/abs/2107.02298)


  The increasing digitalization of the manufacturing domain requires adequate
knowledge modeling to capture relevant information. Ontologies and Knowledge
Graphs provide means to model and relate a wide range of concepts, problems,
and configurations. Both can be used to generate new knowledge through
deductive inference and identify missing knowledge. While digitalization
increases the amount of data available, much data is not labeled and cannot be
directly used to train supervised machine learning models. Active learning can
be used to identify the most informative data instances for which to obtain
users' feedback, reduce friction, and maximize knowledge acquisition. By
combining semantic technologies and active learning, multiple use cases in the
manufacturing domain can be addressed taking advantage of the available
knowledge and data.

    

### [[2107.02326] Pedestrian Emergence Estimation and Occlusion-Aware Risk Assessment for Urban Autonomous Driving](http://arxiv.org/abs/2107.02326)


  Avoiding unseen or partially occluded vulnerable road users (VRUs) is a major
challenge for fully autonomous driving in urban scenes. However,
occlusion-aware risk assessment systems have not been widely studied. Here, we
propose a pedestrian emergence estimation and occlusion-aware risk assessment
system for urban autonomous driving. First, the proposed system utilizes
available contextual information, such as visible cars and pedestrians, to
estimate pedestrian emergence probabilities in occluded regions. These
probabilities are then used in a risk assessment framework, and incorporated
into a longitudinal motion controller. The proposed controller is tested
against several baseline controllers that recapitulate some commonly observed
driving styles. The simulated test scenarios include randomly placed parked
cars and pedestrians, most of whom are occluded from the ego vehicle's view and
emerges randomly. The proposed controller outperformed the baselines in terms
of safety and comfort measures.

    

### [[2107.02328] Polarized skylight orientation determination artificial neural network](http://arxiv.org/abs/2107.02328)


  This paper proposes an artificial neural network to determine orientation
using polarized skylight. This neural network has specific dilated convolution,
which can extract light intensity information of different polarization
directions. Then, the degree of polarization (DOP) and angle of polarization
(AOP) are directly extracted in the network. In addition, the exponential
function encoding of orientation is designed as the network output, which can
better reflect the insect's encoding of polarization information, and improve
the accuracy of orientation determination. Finally, training and testing were
conducted on a public polarized skylight navigation dataset, and the
experimental results proved the stability and effectiveness of the network.

    

### [[2107.02351] Proof Generation in CDSAT](http://arxiv.org/abs/2107.02351)


  The main ideas in the CDSAT (Conflict-Driven Satisfiability) framework for
SMT are summarized, leading to approaches to proof generation in CDSAT.

    

### [[2107.02385] Estimates for the Branching Factors of Atari Games](http://arxiv.org/abs/2107.02385)


  The branching factor of a game is the average number of new states reachable
from a given state. It is a widely used metric in AI research on board games,
but less often computed or discussed for videogames. This paper provides
estimates for the branching factors of 103 Atari 2600 games, as implemented in
the Arcade Learning Environment (ALE). Depending on the game, ALE exposes
between 3 and 18 available actions per frame of gameplay, which is an upper
bound on branching factor. This paper shows, based on an enumeration of the
first 1 million distinct states reachable in each game, that the average
branching factor is usually much lower, in many games barely above 1. In
addition to reporting the branching factors, this paper aims to clarify what
constitutes a distinct state in ALE.

    

### [[2107.02389] Learning Semantic Segmentation of Large-Scale Point Clouds with Random Sampling](http://arxiv.org/abs/2107.02389)


  We study the problem of efficient semantic segmentation of large-scale 3D
point clouds. By relying on expensive sampling techniques or computationally
heavy pre/post-processing steps, most existing approaches are only able to be
trained and operate over small-scale point clouds. In this paper, we introduce
RandLA-Net, an efficient and lightweight neural architecture to directly infer
per-point semantics for large-scale point clouds. The key to our approach is to
use random point sampling instead of more complex point selection approaches.
Although remarkably computation and memory efficient, random sampling can
discard key features by chance. To overcome this, we introduce a novel local
feature aggregation module to progressively increase the receptive field for
each 3D point, thereby effectively preserving geometric details. Comparative
experiments show that our RandLA-Net can process 1 million points in a single
pass up to 200x faster than existing approaches. Moreover, extensive
experiments on five large-scale point cloud datasets, including Semantic3D,
SemanticKITTI, Toronto3D, NPM3D and S3DIS, demonstrate the state-of-the-art
semantic segmentation performance of our RandLA-Net.

    

### [[2107.02451] Integrating Circle Kernels into Convolutional Neural Networks](http://arxiv.org/abs/2107.02451)


  The square kernel is a standard unit for contemporary Convolutional Neural
Networks (CNNs), as it fits well on the tensor computation for the convolution
operation. However, the receptive field in the human visual system is actually
isotropic like a circle. Motivated by this observation, we propose using circle
kernels with isotropic receptive fields for the convolution, and our training
takes approximately equivalent amount of calculation when compared with the
corresponding CNN with square kernels. Our preliminary experiments demonstrate
the rationality of circle kernels. We then propose a kernel boosting strategy
that integrates the circle kernels with square kernels for the training and
inference, and we further let the kernel size/radius be learnable during the
training. Note that we reparameterize the circle kernels or integrated kernels
before the inference, thus taking no extra computation as well as the number of
parameter overhead for the testing. Extensive experiments on several standard
datasets, ImageNet, CIFAR-10 and CIFAR-100, using the circle kernels or
integrated kernels on typical existing CNNs, show that our approach exhibits
highly competitive performance. Specifically, on ImageNet with standard data
augmentation, our approach dramatically boosts the performance of
MobileNetV3-Small by 5.20% top-1 accuracy and 3.39% top-5 accuracy, and boosts
the performance of MobileNetV3-Large by 2.16% top-1 accuracy and 1.18% top-5
accuracy.

    

### [[2107.02457] Comparing PCG metrics with Human Evaluation in Minecraft Settlement Generation](http://arxiv.org/abs/2107.02457)


  There are a range of metrics that can be applied to the artifacts produced by
procedural content generation, and several of them come with qualitative
claims. In this paper, we adapt a range of existing PCG metrics to generated
Minecraft settlements, develop a few new metrics inspired by PCG literature,
and compare the resulting measurements to existing human evaluations. The aim
is to analyze how those metrics capture human evaluation scores in different
categories, how the metrics generalize to another game domain, and how metrics
deal with more complex artifacts. We provide an exploratory look at a variety
of metrics and provide an information gain and several correlation analyses. We
found some relationships between human scores and metrics counting specific
elements, measuring the diversity of blocks and measuring the presence of
crafting materials for the present complex blocks.

    

### [[2107.02472] Empowering NGOs in Countering Online Hate Messages](http://arxiv.org/abs/2107.02472)


  Studies on online hate speech have mostly focused on the automated detection
of harmful messages. Little attention has been devoted so far to the
development of effective strategies to fight hate speech, in particular through
the creation of counter-messages. While existing manual scrutiny and
intervention strategies are time-consuming and not scalable, advances in
natural language processing have the potential to provide a systematic approach
to hatred management. In this paper, we introduce a novel ICT platform that NGO
operators can use to monitor and analyze social media data, along with a
counter-narrative suggestion tool. Our platform aims at increasing the
efficiency and effectiveness of operators' activities against islamophobia. We
test the platform with more than one hundred NGO operators in three countries
through qualitative and quantitative evaluation. Results show that NGOs favor
the platform solution with the suggestion tool, and that the time required to
produce counter-narratives significantly decreases.

    

### [[2107.02524] Depth-Aware Multi-Grid Deep Homography Estimation with Contextual Correlation](http://arxiv.org/abs/2107.02524)


  Homography estimation is an important task in computer vision, such as image
stitching, video stabilization, and camera calibration. Traditional homography
estimation methods heavily depend on the quantity and distribution of feature
points, leading to poor robustness in textureless scenes. The learning
solutions, on the contrary, try to learn robust deep features but demonstrate
unsatisfying performance in the scenes of low overlap rates. In this paper, we
address the two problems simultaneously, by designing a contextual correlation
layer, which can capture the long-range correlation on feature maps and
flexibly be bridged in a learning framework. In addition, considering that a
single homography can not represent the complex spatial transformation in
depth-varying images with parallax, we propose to predict multi-grid homography
from global to local. Moreover, we equip our network with depth perception
capability, by introducing a novel depth-aware shape-preserved loss. Extensive
experiments demonstrate the superiority of our method over other
state-of-the-art solutions in the synthetic benchmark dataset and real-world
dataset. The codes and models will be available at
this https URL.

    

### [[2107.02609] How to Discover a Semantic Web Service by Knowing Its Functionality Parameters](http://arxiv.org/abs/2107.02609)


  In this work, we show how to discover a semantic web service among a
repository of web services. A new approach for web service discovery based on
calculating the functions similarity. We define the Web service functions with
Ontology Web Language (OWL). We wrote some rules for comparing two web
services` parameters. Our algorithm compares the parameters of two web
services` inputs/outputs by making a bipartite graph. We compute the similarity
rate by using the Ford-Fulkerson algorithm. The higher the similarity, the less
are the differences between their functions. At last, our algorithm chooses the
service which has the highest similarity. As a consequence, our method is
useful when we need to find a web service suitable to replace an existing one
that has failed. Especially in autonomic systems, this situation is very common
and important since we need to ensure the availability of the application which
is based on the failed web service. We use Universal Description, Discovery and
Integration (UDDI) compliant web service registry.

    

### [[2107.02629] Embracing the Dark Knowledge: Domain Generalization Using Regularized Knowledge Distillation](http://arxiv.org/abs/2107.02629)


  Though convolutional neural networks are widely used in different tasks, lack
of generalization capability in the absence of sufficient and representative
data is one of the challenges that hinder their practical application. In this
paper, we propose a simple, effective, and plug-and-play training strategy
named Knowledge Distillation for Domain Generalization (KDDG) which is built
upon a knowledge distillation framework with the gradient filter as a novel
regularization term. We find that both the ``richer dark knowledge" from the
teacher network, as well as the gradient filter we proposed, can reduce the
difficulty of learning the mapping which further improves the generalization
ability of the model. We also conduct experiments extensively to show that our
framework can significantly improve the generalization capability of deep
neural networks in different tasks including image classification,
segmentation, reinforcement learning by comparing our method with existing
state-of-the-art domain generalization techniques. Last but not the least, we
propose to adopt two metrics to analyze our proposed method in order to better
understand how our proposed method benefits the generalization capability of
deep neural networks.

    

### [[2107.02748] MAJORITY-3SAT (and Related Problems) in Polynomial Time](http://arxiv.org/abs/2107.02748)


  Majority-SAT is the problem of determining whether an input $n$-variable
formula in conjunctive normal form (CNF) has at least $2^{n-1}$ satisfying
assignments. Majority-SAT and related problems have been studied extensively in
various AI communities interested in the complexity of probabilistic planning
and inference. Although Majority-SAT has been known to be PP-complete for over
40 years, the complexity of a natural variant has remained open:
Majority-$k$SAT, where the input CNF formula is restricted to have clause width
at most $k$.
We prove that for every $k$, Majority-$k$SAT is in P. In fact, for any
positive integer $k$ and rational $\rho \in (0,1)$ with bounded denominator, we
give an algorithm that can determine whether a given $k$-CNF has at least $\rho
\cdot 2^n$ satisfying assignments, in deterministic linear time (whereas the
previous best-known algorithm ran in exponential time). Our algorithms have
interesting positive implications for counting complexity and the complexity of
inference, significantly reducing the known complexities of related problems
such as E-MAJ-$k$SAT and MAJ-MAJ-$k$SAT. At the heart of our approach is an
efficient method for solving threshold counting problems by extracting
sunflowers found in the corresponding set system of a $k$-CNF.
We also show that the tractability of Majority-$k$SAT is somewhat fragile.
For the closely related GtMajority-SAT problem (where we ask whether a given
formula has greater than $2^{n-1}$ satisfying assignments) which is known to be
PP-complete, we show that GtMajority-$k$SAT is in P for $k\le 3$, but becomes
NP-complete for $k\geq 4$. These results are counterintuitive, because the
``natural'' classifications of these problems would have been PP-completeness,
and because there is a stark difference in the complexity of GtMajority-$k$SAT
and Majority-$k$SAT for all $k\ge 4$.

    

### [[2010.07647] Identifying Possible Rumor Spreaders on Twitter: A Weak Supervised Learning Approach](http://arxiv.org/abs/2010.07647)


  Online Social Media (OSM) platforms such as Twitter, Facebook are extensively
exploited by the users of these platforms for spreading the (mis)information to
a large audience effortlessly at a rapid pace. It has been observed that the
misinformation can cause panic, fear, and financial loss to society. Thus, it
is important to detect and control the misinformation in such platforms before
it spreads to the masses. In this work, we focus on rumors, which is one type
of misinformation (other types are fake news, hoaxes, etc). One way to control
the spread of the rumors is by identifying users who are possibly the rumor
spreaders, that is, users who are often involved in spreading the rumors. Due
to the lack of availability of rumor spreaders labeled dataset (which is an
expensive task), we use publicly available PHEME dataset, which contains rumor
and non-rumor tweets information, and then apply a weak supervised learning
approach to transform the PHEME dataset into rumor spreaders dataset. We
utilize three types of features, that is, user, text, and ego-network features,
before applying various supervised learning approaches. In particular, to
exploit the inherent network property in this dataset (user-user reply graph),
we explore Graph Convolutional Network (GCN), a type of Graph Neural Network
(GNN) technique. We compare GCN results with the other approaches: SVM, RF, and
LSTM. Extensive experiments performed on the rumor spreaders dataset, where we
achieve up to 0.864 value for F1-Score and 0.720 value for AUC-ROC, shows the
effectiveness of our methodology for identifying possible rumor spreaders using
the GCN technique.

    

### [[2101.12491] Efficient-CapsNet: Capsule Network with Self-Attention Routing](http://arxiv.org/abs/2101.12491)


  Deep convolutional neural networks, assisted by architectural design
strategies, make extensive use of data augmentation techniques and layers with
a high number of feature maps to embed object transformations. That is highly
inefficient and for large datasets implies a massive redundancy of features
detectors. Even though capsules networks are still in their infancy, they
constitute a promising solution to extend current convolutional networks and
endow artificial visual perception with a process to encode more efficiently
all feature affine transformations. Indeed, a properly working capsule network
should theoretically achieve higher results with a considerably lower number of
parameters count due to intrinsic capability to generalize to novel viewpoints.
Nevertheless, little attention has been given to this relevant aspect. In this
paper, we investigate the efficiency of capsule networks and, pushing their
capacity to the limits with an extreme architecture with barely 160K
parameters, we prove that the proposed architecture is still able to achieve
state-of-the-art results on three different datasets with only 2% of the
original CapsNet parameters. Moreover, we replace dynamic routing with a novel
non-iterative, highly parallelizable routing algorithm that can easily cope
with a reduced number of capsules. Extensive experimentation with other capsule
implementations has proved the effectiveness of our methodology and the
capability of capsule networks to efficiently embed visual representations more
prone to generalization.

    

### [[2106.00978] A Span Extraction Approach for Information Extraction on Visually-Rich Documents](http://arxiv.org/abs/2106.00978)


  Information extraction (IE) for visually-rich documents (VRDs) has achieved
SOTA performance recently thanks to the adaptation of Transformer-based
language models, which shows the great potential of pre-training methods. In
this paper, we present a new approach to improve the capability of language
model pre-training on VRDs. Firstly, we introduce a new query-based IE model
that employs span extraction instead of using the common sequence labeling
approach. Secondly, to further extend the span extraction formulation, we
propose a new training task that focuses on modelling the relationships among
semantic entities within a document. This task enables target spans to be
extracted recursively and can be used to pre-train the model or as an IE
downstream task. Evaluation on three datasets of popular business documents
(invoices, receipts) shows that our proposed method achieves significant
improvements compared to existing models. The method also provides a mechanism
for knowledge accumulation from multiple downstream IE tasks.

    

### [[2107.02346] Thread-modular Analysis of Release-Acquire Concurrency](http://arxiv.org/abs/2107.02346)


  We present a thread-modular abstract interpretation(TMAI) technique to verify
programs under the release-acquire (RA) memory model for safety property
violations. The main contributions of our work are: we capture the execution
order of program statements as an abstract domain, and propose a sound upper
approximation over this domain to efficiently reason over RA concurrency. The
proposed domain is general in its application and captures the ordering
relations as a first-class feature in the abstract interpretation theory. In
particular, the domain represents a set of sequences of modifications of a
global variable in concurrent programs as a partially ordered set. Under this
approximation, older sequenced-before stores of a global variable are forgotten
and only the latest stores per variable are preserved. We establish the
soundness of our proposed abstractions and implement them in a prototype
abstract interpreter called PRIORI. The evaluations of PRIORI on existing and
challenging RA benchmarks demonstrate that the proposed technique is not only
competitive in refutation, but also in verification. PRIORI shows significantly
fast analysis runtimes with higher precision compared to recent
state-of-the-art tools for RA concurrency.

    

### [[2103.10164] PySTACHIO: Python Single-molecule TrAcking stoiCHiometry Intensity and simulatiOn, a flexible, extensible, beginner-friendly and optimized program for analysis of single-molecule microscopy](http://arxiv.org/abs/2103.10164)


  As camera pixel arrays have grown larger and faster, and optical microscopy
techniques ever more refined, there has been an explosion in the quantity of
data acquired during routine light microcopy. At the single-molecule level,
analysis involves multiple steps and can rapidly become computationally
expensive, in some cases intractable on office workstations. Complex bespoke
software can present high activation barriers to entry for new users. Here, we
redevelop our quantitative single-molecule analysis routines into an optimized
and extensible Python program, with GUI and command-line implementations to
facilitate use on local machines and remote clusters, by beginners and advanced
users alike. We demonstrate that its performance is on par with previous MATLAB
implementations but runs an order of magnitude faster. We tested it against
challenge data and demonstrate its performance is comparable to
state-of-the-art analysis platforms. We show the code can extract fluorescence
intensity values for single reporter dye molecules and, using these, estimate
molecular stoichiometries and cellular copy numbers of fluorescently-labeled
biomolecules. It can evaluate 2D diffusion coefficients for the
characteristically short single-particle tracking data. To facilitate
benchmarking we include data simulation routines to compare different analysis
programs. Finally, we show that it works with 2-color data and enables
colocalization analysis based on overlap integration, to infer interactions
between differently labelled biomolecules. By making this freely available we
aim to make complex light microscopy single-molecule analysis more
democratized.

    

### [[2103.15408] A simpler encoding of indexed types](http://arxiv.org/abs/2103.15408)


  In functional programming languages, generalized algebraic data types (GADTs)
are very useful as the unnecessary pattern matching over them can be ruled out
by the failure of unification of type arguments. In dependent type systems,
this is usually called indexed types and it's particularly useful as the
identity type is a special case of it. However, pattern matching over indexed
types is very complicated as it requires term unification in general. We study
a simplified version of indexed types (called simpler indexed types) where we
explicitly specify the selection process of constructors, and we discuss its
expressiveness, limitations, and properties.

    

### [[2105.14840] Elegant elaboration with function invocation](http://arxiv.org/abs/2105.14840)


  We present an elegant design of the core language in a dependently-typed
lambda calculus with $\delta$-reduction and an elaboration algorithm.

    

### [<title>Is there any unnecessary or debugging files in the XGBoost model to remove and make the model size smaller? - XGBoost</title>](https://discuss.xgboost.ai/t/is-there-any-unnecessary-or-debugging-files-in-the-xgboost-model-to-remove-and-make-the-model-size-smaller/2360/3)