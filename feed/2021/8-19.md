
## 2021-8-19

### [[2104.09159] Conditional Variational Capsule Network for Open Set Recognition](http://arxiv.org/abs/2104.09159)


  In open set recognition, a classifier has to detect unknown classes that are
not known at training time. In order to recognize new categories, the
classifier has to project the input samples of known classes in very compact
and separated regions of the features space for discriminating samples of
unknown classes. Recently proposed Capsule Networks have shown to outperform
alternatives in many fields, particularly in image recognition, however they
have not been fully applied yet to open-set recognition. In capsule networks,
scalar neurons are replaced by capsule vectors or matrices, whose entries
represent different properties of objects. In our proposal, during training,
capsules features of the same known class are encouraged to match a pre-defined
gaussian, one for each class. To this end, we use the variational autoencoder
framework, with a set of gaussian priors as the approximation for the posterior
distribution. In this way, we are able to control the compactness of the
features of the same class around the center of the gaussians, thus controlling
the ability of the classifier in detecting samples from unknown classes. We
conducted several experiments and ablation of our model, obtaining state of the
art results on different datasets in the open set recognition and unknown
detection tasks.

    

### [[2108.07854] Coverage Hole Detection for mmWave Networks: An Unsupervised Learning Approach](http://arxiv.org/abs/2108.07854)


  The utilization of millimeter-wave (mmWave) bands in 5G networks poses new
challenges to network planning. Vulnerability to blockages at mmWave bands can
cause coverage holes (CHs) in the radio environment, leading to radio link
failure when a user enters these CHs. Detection of the CHs carries critical
importance so that necessary remedies can be introduced to improve coverage. In
this letter, we propose a novel approach to identify the CHs in an unsupervised
fashion using a state-of-the-art manifold learning technique: uniform manifold
approximation and projection. The key idea is to preserve the
local-connectedness structure inherent in the collected unlabelled channel
samples, such that the CHs from the service area are detectable. Our results on
the DeepMIMO dataset scenario demonstrate that the proposed method can learn
the structure within the data samples and provide visual holes in the
low-dimensional embedding while preserving the CH boundaries. Once the CH
boundary is determined in the low-dimensional embedding, channel-based
localization techniques can be applied to these samples to obtain the
geographical boundaries of the CHs.

    

### [[2108.08059] Resource Scheduling in Edge Computing: A Survey](http://arxiv.org/abs/2108.08059)


  With the proliferation of the Internet of Things (IoT) and the wide
penetration of wireless networks, the surging demand for data communications
and computing calls for the emerging edge computing paradigm. By moving the
services and functions located in the cloud to the proximity of users, edge
computing can provide powerful communication, storage, networking, and
communication capacity. The resource scheduling in edge computing, which is the
key to the success of edge computing systems, has attracted increasing research
interests. In this paper, we survey the state-of-the-art research findings to
know the research progress in this field. Specifically, we present the
architecture of edge computing, under which different collaborative manners for
resource scheduling are discussed. Particularly, we introduce a unified model
before summarizing the current works on resource scheduling from three research
issues, including computation offloading, resource allocation, and resource
provisioning. Based on two modes of operation, i.e., centralized and
distributed modes, different techniques for resource scheduling are discussed
and compared. Also, we summarize the main performance indicators based on the
surveyed literature. To shed light on the significance of resource scheduling
in real-world scenarios, we discuss several typical application scenarios
involved in the research of resource scheduling in edge computing. Finally, we
highlight some open research challenges yet to be addressed and outline several
open issues as the future research direction.

    

### [[2108.08241] Semi-Supervised Learning for Channel Charting-Aided IoT Localization in Millimeter Wave Networks](http://arxiv.org/abs/2108.08241)


  In this paper, a novel framework is proposed for channel charting (CC)-aided
localization in millimeter wave networks. In particular, a convolutional
autoencoder model is proposed to estimate the three-dimensional location of
wireless user equipment (UE), based on multipath channel state information
(CSI), received by different base stations. In order to learn the
radio-geometry map and capture the relative position of each UE, an
autoencoder-based channel chart is constructed in an unsupervised manner, such
that neighboring UEs in the physical space will remain close in the channel
chart. Next, the channel charting model is extended to a semi-supervised
framework, where the autoencoder is divided into two components: an encoder and
a decoder, and each component is optimized individually, using the labeled CSI
dataset with associated location information, to further improve positioning
accuracy. Simulation results show that the proposed CC-aided semi-supervised
localization yields a higher accuracy, compared with existing supervised
positioning and conventional unsupervised CC approaches.

    

### [[2012.08677] Inexact-ADMM Based Federated Meta-Learning for Fast and Continual Edge Learning](http://arxiv.org/abs/2012.08677)


  In order to meet the requirements for performance, safety, and latency in
many IoT applications, intelligent decisions must be made right here right now
at the network edge. However, the constrained resources and limited local data
amount pose significant challenges to the development of edge AI. To overcome
these challenges, we explore continual edge learning capable of leveraging the
knowledge transfer from previous tasks. Aiming to achieve fast and continual
edge learning, we propose a platform-aided federated meta-learning architecture
where edge nodes collaboratively learn a meta-model, aided by the knowledge
transfer from prior tasks. The edge learning problem is cast as a regularized
optimization problem, where the valuable knowledge learned from previous tasks
is extracted as regularization. Then, we devise an ADMM based federated
meta-learning algorithm, namely ADMM-FedMeta, where ADMM offers a natural
mechanism to decompose the original problem into many subproblems which can be
solved in parallel across edge nodes and the platform. Further, a variant of
inexact-ADMM method is employed where the subproblems are `solved' via linear
approximation as well as Hessian estimation to reduce the computational cost
per round to $\mathcal{O}(n)$. We provide a comprehensive analysis of
ADMM-FedMeta, in terms of the convergence properties, the rapid adaptation
performance, and the forgetting effect of prior knowledge transfer, for the
general non-convex case. Extensive experimental studies demonstrate the
effectiveness and efficiency of ADMM-FedMeta, and showcase that it
substantially outperforms the existing baselines.

    

### [[2108.07799] An Extensible Benchmark Suite for Learning to Simulate Physical Systems](http://arxiv.org/abs/2108.07799)


  Simulating physical systems is a core component of scientific computing,
encompassing a wide range of physical domains and applications. Recently, there
has been a surge in data-driven methods to complement traditional numerical
simulations methods, motivated by the opportunity to reduce computational costs
and/or learn new physical models leveraging access to large collections of
data. However, the diversity of problem settings and applications has led to a
plethora of approaches, each one evaluated on a different setup and with
different evaluation metrics. We introduce a set of benchmark problems to take
a step towards unified benchmarks and evaluation protocols. We propose four
representative physical systems, as well as a collection of both widely used
classical time integrators and representative data-driven methods
(kernel-based, MLP, CNN, nearest neighbors). Our framework allows evaluating
objectively and systematically the stability, accuracy, and computational
efficiency of data-driven methods. Additionally, it is configurable to permit
adjustments for accommodating other learning tasks and for establishing a
foundation for future developments in machine learning for scientific
computing.

    

### [[2108.07800] Bagging Supervised Autoencoder Classifier for Credit Scoring](http://arxiv.org/abs/2108.07800)


  Credit scoring models, which are among the most potent risk management tools
that banks and financial institutes rely on, have been a popular subject for
research in the past few decades. Accordingly, many approaches have been
developed to address the challenges in classifying loan applicants and improve
and facilitate decision-making. The imbalanced nature of credit scoring
datasets, as well as the heterogeneous nature of features in credit scoring
datasets, pose difficulties in developing and implementing effective credit
scoring models, targeting the generalization power of classification models on
unseen data. In this paper, we propose the Bagging Supervised Autoencoder
Classifier (BSAC) that mainly leverages the superior performance of the
Supervised Autoencoder, which learns low-dimensional embeddings of the input
data exclusively with regards to the ultimate classification task of credit
scoring, based on the principles of multi-task learning. BSAC also addresses
the data imbalance problem by employing a variant of the Bagging process based
on the undersampling of the majority class. The obtained results from our
experiments on the benchmark and real-life credit scoring datasets illustrate
the robustness and effectiveness of the Bagging Supervised Autoencoder
Classifier in the classification of loan applicants that can be regarded as a
positive development in credit scoring models.

    

### [[2108.07827] Compressing gradients by exploiting temporal correlation in momentum-SGD](http://arxiv.org/abs/2108.07827)


  An increasing bottleneck in decentralized optimization is communication.
Bigger models and growing datasets mean that decentralization of computation is
important and that the amount of information exchanged is quickly growing.
While compression techniques have been introduced to cope with the latter, none
has considered leveraging the temporal correlations that exist in consecutive
vector updates. An important example is distributed momentum-SGD where temporal
correlation is enhanced by the low-pass-filtering effect of applying momentum.
In this paper we design and analyze compression methods that exploit temporal
correlation in systems both with and without error-feedback. Experiments with
the ImageNet dataset demonstrate that our proposed methods offer significant
reduction in the rate of communication at only a negligible increase in
computation complexity. We further analyze the convergence of SGD when
compression is applied with error-feedback. In the literature, convergence
guarantees are developed only for compressors that provide error-bounds
point-wise, i.e., for each input to the compressor. In contrast, many important
codes (e.g. rate-distortion codes) provide error-bounds only in expectation and
thus provide a more general guarantee. In this paper we prove the convergence
of SGD under an expected error assumption by establishing a bound for the
minimum gradient norm.

    

### [[2108.07856] OncoPetNet: A Deep Learning based AI system for mitotic figure counting on H&E stained whole slide digital images in a large veterinary diagnostic lab setting](http://arxiv.org/abs/2108.07856)


  Background: Histopathology is an important modality for the diagnosis and
management of many diseases in modern healthcare, and plays a critical role in
cancer care. Pathology samples can be large and require multi-site sampling,
leading to upwards of 20 slides for a single tumor, and the human-expert tasks
of site selection and and quantitative assessment of mitotic figures are time
consuming and subjective. Automating these tasks in the setting of a digital
pathology service presents significant opportunities to improve workflow
efficiency and augment human experts in practice. Approach: Multiple
state-of-the-art deep learning techniques for histopathology image
classification and mitotic figure detection were used in the development of
OncoPetNet. Additionally, model-free approaches were used to increase speed and
accuracy. The robust and scalable inference engine leverages Pytorch's
performance optimizations as well as specifically developed speed up techniques
in inference. Results: The proposed system, demonstrated significantly improved
mitotic counting performance for 41 cancer cases across 14 cancer types
compared to human expert baselines. In 21.9% of cases use of OncoPetNet led to
change in tumor grading compared to human expert evaluation. In deployment, an
effective 0.27 min/slide inference was achieved in a high throughput veterinary
diagnostic pathology service across 2 centers processing 3,323 digital whole
slide images daily. Conclusion: This work represents the first successful
automated deployment of deep learning systems for real-time expert-level
performance on important histopathology tasks at scale in a high volume
clinical practice. The resulting impact outlines important considerations for
model development, deployment, clinical decision making, and informs best
practices for implementation of deep learning systems in digital histopathology
practices.

    

### [[2108.07872] Aggregated Customer Engagement Model](http://arxiv.org/abs/2108.07872)


  E-commerce websites use machine learned ranking models to serve shopping
results to customers. Typically, the websites log the customer search events,
which include the query entered and the resulting engagement with the shopping
results, such as clicks and purchases. Each customer search event serves as
input training data for the models, and the individual customer engagement
serves as a signal for customer preference. So a purchased shopping result, for
example, is perceived to be more important than one that is not. However, new
or under-impressed products do not have enough customer engagement signals and
end up at a disadvantage when being ranked alongside popular products. In this
paper, we propose a novel method for data curation that aggregates all customer
engagements within a day for the same query to use as input training data. This
aggregated customer engagement gives the models a complete picture of the
relative importance of shopping results. Training models on this aggregated
data leads to less reliance on behavioral features. This helps mitigate the
cold start problem and boosted relevant new products to top search results. In
this paper, we present the offline and online analysis and results comparing
the individual and aggregated customer engagement models trained on e-commerce
data.

    

### [[2108.07879] Edge AI without Compromise: Efficient, Versatile and Accurate Neurocomputing in Resistive Random-Access Memory](http://arxiv.org/abs/2108.07879)


  Realizing today's cloud-level artificial intelligence functionalities
directly on devices distributed at the edge of the internet calls for edge
hardware capable of processing multiple modalities of sensory data (e.g. video,
audio) at unprecedented energy-efficiency. AI hardware architectures today
cannot meet the demand due to a fundamental "memory wall": data movement
between separate compute and memory units consumes large energy and incurs long
latency. Resistive random-access memory (RRAM) based compute-in-memory (CIM)
architectures promise to bring orders of magnitude energy-efficiency
improvement by performing computation directly within memory. However,
conventional approaches to CIM hardware design limit its functional flexibility
necessary for processing diverse AI workloads, and must overcome hardware
imperfections that degrade inference accuracy. Such trade-offs between
efficiency, versatility and accuracy cannot be addressed by isolated
improvements on any single level of the design. By co-optimizing across all
hierarchies of the design from algorithms and architecture to circuits and
devices, we present NeuRRAM - the first multimodal edge AI chip using RRAM CIM
to simultaneously deliver a high degree of versatility for diverse model
architectures, record energy-efficiency $5\times$ - $8\times$ better than prior
art across various computational bit-precisions, and inference accuracy
comparable to software models with 4-bit weights on all measured standard AI
benchmarks including accuracy of 99.0% on MNIST and 85.7% on CIFAR-10 image
classification, 84.7% accuracy on Google speech command recognition, and a 70%
reduction in image reconstruction error on a Bayesian image recovery task. This
work paves a way towards building highly efficient and reconfigurable edge AI
hardware platforms for the more demanding and heterogeneous AI applications of
the future.

    

### [[2108.07880] Statistically Near-Optimal Hypothesis Selection](http://arxiv.org/abs/2108.07880)


  Hypothesis Selection is a fundamental distribution learning problem where
given a comparator-class $Q=\{q_1,\ldots, q_n\}$ of distributions, and a
sampling access to an unknown target distribution $p$, the goal is to output a
distribution $q$ such that $\mathsf{TV}(p,q)$ is close to $opt$, where $opt =
\min_i\{\mathsf{TV}(p,q_i)\}$ and $\mathsf{TV}(\cdot, \cdot)$ denotes the
total-variation distance. Despite the fact that this problem has been studied
since the 19th century, its complexity in terms of basic resources, such as
number of samples and approximation guarantees, remains unsettled (this is
discussed, e.g., in the charming book by Devroye and Lugosi `00). This is in
stark contrast with other (younger) learning settings, such as PAC learning,
for which these complexities are well understood.
We derive an optimal $2$-approximation learning strategy for the Hypothesis
Selection problem, outputting $q$ such that $\mathsf{TV}(p,q) \leq2 \cdot opt +
\eps$, with a (nearly) optimal sample complexity of~$\tilde O(\log
n/\epsilon^2)$. This is the first algorithm that simultaneously achieves the
best approximation factor and sample complexity: previously, Bousquet, Kane,
and Moran (COLT `19) gave a learner achieving the optimal $2$-approximation,
but with an exponentially worse sample complexity of $\tilde
O(\sqrt{n}/\epsilon^{2.5})$, and Yatracos~(Annals of Statistics `85) gave a
learner with optimal sample complexity of $O(\log n /\epsilon^2)$ but with a
sub-optimal approximation factor of $3$.

    

### [[2108.07886] Modulating Language Models with Emotions](http://arxiv.org/abs/2108.07886)


  Generating context-aware language that embodies diverse emotions is an
important step towards building empathetic NLP systems. In this paper, we
propose a formulation of modulated layer normalization -- a technique inspired
by computer vision -- that allows us to use large-scale language models for
emotional response generation. In automatic and human evaluation on the
MojiTalk dataset, our proposed modulated layer normalization method outperforms
prior baseline methods while maintaining diversity, fluency, and coherence. Our
method also obtains competitive performance even when using only 10% of the
available training data.

    

### [[2108.07887] Diversity-based Trajectory and Goal Selection with Hindsight Experience Replay](http://arxiv.org/abs/2108.07887)


  Hindsight experience replay (HER) is a goal relabelling technique typically
used with off-policy deep reinforcement learning algorithms to solve
goal-oriented tasks; it is well suited to robotic manipulation tasks that
deliver only sparse rewards. In HER, both trajectories and transitions are
sampled uniformly for training. However, not all of the agent's experiences
contribute equally to training, and so naive uniform sampling may lead to
inefficient learning. In this paper, we propose diversity-based trajectory and
goal selection with HER (DTGSH). Firstly, trajectories are sampled according to
the diversity of the goal states as modelled by determinantal point processes
(DPPs). Secondly, transitions with diverse goal states are selected from the
trajectories by using k-DPPs. We evaluate DTGSH on five challenging robotic
manipulation tasks in simulated robot environments, where we show that our
method can learn more quickly and reach higher performance than other
state-of-the-art approaches on all tasks.

    

### [[2108.07897] Affect-Aware Deep Belief Network Representations for Multimodal Unsupervised Deception Detection](http://arxiv.org/abs/2108.07897)


  Automated systems that detect the social behavior of deception can enhance
human well-being across medical, social work, and legal domains. Labeled
datasets to train supervised deception detection models can rarely be collected
for real-world, high-stakes contexts. To address this challenge, we propose the
first unsupervised approach for detecting real-world, high-stakes deception in
videos without requiring labels. This paper presents our novel approach for
affect-aware unsupervised Deep Belief Networks (DBN) to learn discriminative
representations of deceptive and truthful behavior. Drawing on psychology
theories that link affect and deception, we experimented with unimodal and
multimodal DBN-based approaches trained on facial valence, facial arousal,
audio, and visual features. In addition to using facial affect as a feature on
which DBN models are trained, we also introduce a DBN training procedure that
uses facial affect as an aligner of audio-visual representations. We conducted
classification experiments with unsupervised Gaussian Mixture Model clustering
to evaluate our approaches. Our best unsupervised approach (trained on facial
valence and visual features) achieved an AUC of 80%, outperforming human
ability and performing comparably to fully-supervised models. Our results
motivate future work on unsupervised, affect-aware computational approaches for
detecting deception and other social behaviors in the wild.

    

### [[2108.07901] HyperSF: Spectral Hypergraph Coarsening via Flow-based Local Clustering](http://arxiv.org/abs/2108.07901)


  Hypergraphs allow modeling problems with multi-way high-order relationships.
However, the computational cost of most existing hypergraph-based algorithms
can be heavily dependent upon the input hypergraph sizes. To address the
ever-increasing computational challenges, graph coarsening can be potentially
applied for preprocessing a given hypergraph by aggressively aggregating its
vertices (nodes). However, state-of-the-art hypergraph partitioning
(clustering) methods that incorporate heuristic graph coarsening techniques are
not optimized for preserving the structural (global) properties of hypergraphs.
In this work, we propose an efficient spectral hypergraph coarsening scheme
(HyperSF) for well preserving the original spectral (structural) properties of
hypergraphs. Our approach leverages a recent strongly-local max-flow-based
clustering algorithm for detecting the sets of hypergraph vertices that
minimize ratio cut. To further improve the algorithm efficiency, we propose a
divide-and-conquer scheme by leveraging spectral clustering of the bipartite
graphs corresponding to the original hypergraphs. Our experimental results for
a variety of hypergraphs extracted from real-world VLSI design benchmarks show
that the proposed hypergraph coarsening algorithm can significantly improve the
multi-way conductance of hypergraph clustering as well as runtime efficiency
when compared with existing state-of-the-art algorithms.

    

### [[2108.07908] M-ar-K-Fast Independent Component Analysis](http://arxiv.org/abs/2108.07908)


  This study presents the m-arcsinh Kernel ('m-ar-K') Fast Independent
Component Analysis ('FastICA') method ('m-ar-K-FastICA') for feature
extraction. The kernel trick has enabled dimensionality reduction techniques to
capture a higher extent of non-linearity in the data; however, reproducible,
open-source kernels to aid with feature extraction are still limited and may
not be reliable when projecting features from entropic data. The m-ar-K
function, freely available in Python and compatible with its open-source
library 'scikit-learn', is hereby coupled with FastICA to achieve more reliable
feature extraction in presence of a high extent of randomness in the data,
reducing the need for pre-whitening. Different classification tasks were
considered, as related to five (N = 5) open access datasets of various degrees
of information entropy, available from scikit-learn and the University
California Irvine (UCI) Machine Learning repository. Experimental results
demonstrate improvements in the classification performance brought by the
proposed feature extraction. The novel m-ar-K-FastICA dimensionality reduction
approach is compared to the 'FastICA' gold standard method, supporting its
higher reliability and computational efficiency, regardless of the underlying
uncertainty in the data.

    

### [[2108.07915] Data Pricing in Machine Learning Pipelines](http://arxiv.org/abs/2108.07915)


  Machine learning is disruptive. At the same time, machine learning can only
succeed by collaboration among many parties in multiple steps naturally as
pipelines in an eco-system, such as collecting data for possible machine
learning applications, collaboratively training models by multiple parties and
delivering machine learning services to end users. Data is critical and
penetrating in the whole machine learning pipelines. As machine learning
pipelines involve many parties and, in order to be successful, have to form a
constructive and dynamic eco-system, marketplaces and data pricing are
fundamental in connecting and facilitating those many parties. In this article,
we survey the principles and the latest research development of data pricing in
machine learning pipelines. We start with a brief review of data marketplaces
and pricing desiderata. Then, we focus on pricing in three important steps in
machine learning pipelines. To understand pricing in the step of training data
collection, we review pricing raw data sets and data labels. We also
investigate pricing in the step of collaborative training of machine learning
models, and overview pricing machine learning models for end users in the step
of machine learning deployment. We also discuss a series of possible future
directions.

    

### [[2108.07926] Learning to Collaborate](http://arxiv.org/abs/2108.07926)


  In this paper, we focus on effective learning over a collaborative research
network involving multiple clients. Each client has its own sample population
which may not be shared with other clients due to privacy concerns. The goal is
to learn a model for each client, which behaves better than the one learned
from its own data, through secure collaborations with other clients in the
network. Due to the discrepancies of the sample distributions across different
clients, it is not necessarily that collaborating with everyone will lead to
the best local models. We propose a learning to collaborate framework, where
each client can choose to collaborate with certain members in the network to
achieve a "collaboration equilibrium", where smaller collaboration coalitions
are formed within the network so that each client can obtain the model with the
best utility. We propose the concept of benefit graph which describes how each
client can benefit from collaborating with other clients and develop a Pareto
optimization approach to obtain it. Finally the collaboration coalitions can be
derived from it based on graph operations. Our framework provides a new way of
setting up collaborations in a research network. Experiments on both synthetic
and real world data sets are provided to demonstrate the effectiveness of our
method.

    

### [[2108.07927] Fed-TGAN: Federated Learning Framework for Synthesizing Tabular Data](http://arxiv.org/abs/2108.07927)


  Generative Adversarial Networks (GANs) are typically trained to synthesize
data, from images and more recently tabular data, under the assumption of
directly accessible training data. Recently, federated learning (FL) is an
emerging paradigm that features decentralized learning on client's local data
with a privacy-preserving capability. And, while learning GANs to synthesize
images on FL systems has just been demonstrated, it is unknown if GANs for
tabular data can be learned from decentralized data sources. Moreover, it
remains unclear which distributed architecture suits them best. Different from
image GANs, state-of-the-art tabular GANs require prior knowledge on the data
distribution of each (discrete and continuous) column to agree on a common
encoding -- risking privacy guarantees. In this paper, we propose Fed-TGAN, the
first Federated learning framework for Tabular GANs. To effectively learn a
complex tabular GAN on non-identical participants, Fed-TGAN designs two novel
features: (i) a privacy-preserving multi-source feature encoding for model
initialization; and (ii) table similarity aware weighting strategies to
aggregate local models for countering data skew. We extensively evaluate the
proposed Fed-TGAN against variants of decentralized learning architectures on
four widely used datasets. Results show that Fed-TGAN accelerates training time
per epoch up to 200% compared to the alternative architectures, for both IID
and Non-IID data. Overall, Fed-TGAN not only stabilizes the training loss, but
also achieves better similarity between generated and original data.

    

### [[2108.07930] A new semi-supervised inductive transfer learning framework: Co-Transfer](http://arxiv.org/abs/2108.07930)


  In many practical data mining scenarios, such as network intrusion detection,
Twitter spam detection, and computer-aided diagnosis, a source domain that is
different from but related to a target domain is very common. In addition, a
large amount of unlabeled data is available in both source and target domains,
but labeling each of them is difficult, expensive, time-consuming, and sometime
unnecessary. Therefore, it is very important and worthwhile to fully explore
the labeled and unlabeled data in source and target domains to settle the task
in target domain. In this paper, a new semi-supervised inductive transfer
learning framework, named \emph{Co-Transfer} is proposed. Co-Transfer first
generates three TrAdaBoost classifiers for transfer learning from the source
domain to the target domain, and meanwhile another three TrAdaBoost classifiers
are generated for transfer learning from the target domain to the source
domain, using bootstraped samples from the original labeled data. In each round
of co-transfer, each group of TrAdaBoost classifiers are refined using the
carefully labeled data. Finally, the group of TrAdaBoost classifiers learned to
transfer from the source domain to the target domain produce the final
hypothesis. Experiments results illustrate Co-Transfer can effectively exploit
and reuse the labeled and unlabeled data in source and target domains.

    

### [[2108.07931] Learning Federated Representations and Recommendations with Limited Negatives](http://arxiv.org/abs/2108.07931)


  Deep retrieval models are widely used for learning entity representations and
recommendations. Federated learning provides a privacy-preserving way to train
these models without requiring centralization of user data. However, federated
deep retrieval models usually perform much worse than their centralized
counterparts due to non-IID (independent and identically distributed) training
data on clients, an intrinsic property of federated learning that limits
negatives available for training. We demonstrate that this issue is distinct
from the commonly studied client drift problem. This work proposes
batch-insensitive losses as a way to alleviate the non-IID negatives issue for
federated movie recommendation. We explore a variety of techniques and identify
that batch-insensitive losses can effectively improve the performance of
federated deep retrieval models, increasing the relative recall of the
federated model by up to 93.15% and reducing the relative gap in recall between
it and a centralized model from 27.22% - 43.14% to 0.53% - 2.42%. We
open-source our code framework to accelerate further research and applications
of federated deep retrieval models.

    

### [[2108.07951] Look Before You Leap! Designing a Human-Centered AI System for Change Risk Assessment](http://arxiv.org/abs/2108.07951)


  Reducing the number of failures in a production system is one of the most
challenging problems in technology driven industries, such as, the online
retail industry. To address this challenge, change management has emerged as a
promising sub-field in operations that manages and reviews the changes to be
deployed in production in a systematic manner. However, it is practically
impossible to manually review a large number of changes on a daily basis and
assess the risk associated with them. This warrants the development of an
automated system to assess the risk associated with a large number of changes.
There are a few commercial solutions available to address this problem but
those solutions lack the ability to incorporate domain knowledge and continuous
feedback from domain experts into the risk assessment process. As part of this
work, we aim to bridge the gap between model-driven risk assessment of change
requests and the assessment of domain experts by building a continuous feedback
loop into the risk assessment process. Here we present our work to build an
end-to-end machine learning system along with the discussion of some of
practical challenges we faced related to extreme skewness in class
distribution, concept drift, estimation of the uncertainty associated with the
model's prediction and the overall scalability of the system.

    

### [[2108.07958] Semantic Perturbations with Normalizing Flows for Improved Generalization](http://arxiv.org/abs/2108.07958)


  Data augmentation is a widely adopted technique for avoiding overfitting when
training deep neural networks. However, this approach requires domain-specific
knowledge and is often limited to a fixed set of hard-coded transformations.
Recently, several works proposed to use generative models for generating
semantically meaningful perturbations to train a classifier. However, because
accurate encoding and decoding are critical, these methods, which use
architectures that approximate the latent-variable inference, remained limited
to pilot studies on small datasets.
Exploiting the exactly reversible encoder-decoder structure of normalizing
flows, we perform on-manifold perturbations in the latent space to define fully
unsupervised data augmentations. We demonstrate that such perturbations match
the performance of advanced data augmentation techniques -- reaching 96.6% test
accuracy for CIFAR-10 using ResNet-18 and outperform existing methods,
particularly in low data regimes -- yielding 10--25% relative improvement of
test accuracy from classical training. We find that our latent adversarial
perturbations adaptive to the classifier throughout its training are most
effective, yielding the first test accuracy improvement results on real-world
datasets -- CIFAR-10/100 -- via latent-space perturbations.

    

### [[2108.07961] Verifying Low-dimensional Input Neural Networks via Input Quantization](http://arxiv.org/abs/2108.07961)


  Deep neural networks are an attractive tool for compressing the control
policy lookup tables in systems such as the Airborne Collision Avoidance System
(ACAS). It is vital to ensure the safety of such neural controllers via
verification techniques. The problem of analyzing ACAS Xu networks has
motivated many successful neural network verifiers. These verifiers typically
analyze the internal computation of neural networks to decide whether a
property regarding the input/output holds. The intrinsic complexity of neural
network computation renders such verifiers slow to run and vulnerable to
floating-point error.
This paper revisits the original problem of verifying ACAS Xu networks. The
networks take low-dimensional sensory inputs with training data provided by a
precomputed lookup table. We propose to prepend an input quantization layer to
the network. Quantization allows efficient verification via input state
enumeration, whose complexity is bounded by the size of the quantization space.
Quantization is equivalent to nearest-neighbor interpolation at run time, which
has been shown to provide acceptable accuracy for ACAS in simulation. Moreover,
our technique can deliver exact verification results immune to floating-point
error if we directly enumerate the network outputs on the target inference
implementation or on an accurate simulation of the target implementation.

    

### [[2108.07969] Revisiting Adversarial Robustness Distillation: Robust Soft Labels Make Student Better](http://arxiv.org/abs/2108.07969)


  Adversarial training is one effective approach for training robust deep
neural networks against adversarial attacks. While being able to bring reliable
robustness, adversarial training (AT) methods in general favor high capacity
models, i.e., the larger the model the better the robustness. This tends to
limit their effectiveness on small models, which are more preferable in
scenarios where storage or computing resources are very limited (e.g., mobile
devices). In this paper, we leverage the concept of knowledge distillation to
improve the robustness of small models by distilling from adversarially trained
large models. We first revisit several state-of-the-art AT methods from a
distillation perspective and identify one common technique that can lead to
improved robustness: the use of robust soft labels -- predictions of a robust
model. Following this observation, we propose a novel adversarial robustness
distillation method called Robust Soft Label Adversarial Distillation (RSLAD)
to train robust small student models. RSLAD fully exploits the robust soft
labels produced by a robust (adversarially-trained) large teacher model to
guide the student's learning on both natural and adversarial examples in all
loss terms. We empirically demonstrate the effectiveness of our RSLAD approach
over existing adversarial training and distillation methods in improving the
robustness of small models against state-of-the-art attacks including the
AutoAttack. We also provide a set of understandings on our RSLAD and the
importance of robust soft labels for adversarial robustness distillation.

    

### [[2108.07971] De-identification of Unstructured Clinical Texts from Sequence to Sequence Perspective](http://arxiv.org/abs/2108.07971)


  In this work, we propose a novel problem formulation for de-identification of
unstructured clinical text. We formulate the de-identification problem as a
sequence to sequence learning problem instead of a token classification
problem. Our approach is inspired by the recent state-of -the-art performance
of sequence to sequence learning models for named entity recognition. Early
experimentation of our proposed approach achieved 98.91% recall rate on i2b2
dataset. This performance is comparable to current state-of-the-art models for
unstructured clinical text de-identification.

    

### [[2108.07976] A Unified Framework for Cross-Domain and Cross-System Recommendations](http://arxiv.org/abs/2108.07976)


  Cross-Domain Recommendation (CDR) and Cross-System Recommendation (CSR) have
been proposed to improve the recommendation accuracy in a target dataset
(domain/system) with the help of a source one with relatively richer
information. However, most existing CDR and CSR approaches are single-target,
namely, there is a single target dataset, which can only help the target
dataset and thus cannot benefit the source dataset. In this paper, we focus on
three new scenarios, i.e., Dual-Target CDR (DTCDR), Multi-Target CDR (MTCDR),
and CDR+CSR, and aim to improve the recommendation accuracy in all datasets
simultaneously for all scenarios. To do this, we propose a unified framework,
called GA (based on Graph embedding and Attention techniques), for all three
scenarios. In GA, we first construct separate heterogeneous graphs to generate
more representative user and item embeddings. Then, we propose an element-wise
attention mechanism to effectively combine the embeddings of common entities
(users/items) learned from different datasets. Moreover, to avoid negative
transfer, we further propose a Personalized training strategy to minimize the
embedding difference of common entities between a richer dataset and a sparser
dataset, deriving three new models, i.e., GA-DTCDR-P, GA-MTCDR-P, and
GA-CDR+CSR-P, for the three scenarios respectively. Extensive experiments
conducted on four real-world datasets demonstrate that our proposed GA models
significantly outperform the state-of-the-art approaches.

    

### [[2108.07992] On Multimarginal Partial Optimal Transport: Equivalent Forms and Computational Complexity](http://arxiv.org/abs/2108.07992)


  We study the multi-marginal partial optimal transport (POT) problem between
$m$ discrete (unbalanced) measures with at most $n$ supports. We first prove
that we can obtain two equivalence forms of the multimarginal POT problem in
terms of the multimarginal optimal transport problem via novel extensions of
cost tensor. The first equivalence form is derived under the assumptions that
the total masses of each measure are sufficiently close while the second
equivalence form does not require any conditions on these masses but at the
price of more sophisticated extended cost tensor. Our proof techniques for
obtaining these equivalence forms rely on novel procedures of moving mass in
graph theory to push transportation plan into appropriate regions. Finally,
based on the equivalence forms, we develop optimization algorithm, named
ApproxMPOT algorithm, that builds upon the Sinkhorn algorithm for solving the
entropic regularized multimarginal optimal transport. We demonstrate that the
ApproxMPOT algorithm can approximate the optimal value of multimarginal POT
problem with a computational complexity upper bound of the order
$\tilde{\mathcal{O}}(m^3(n+1)^{m}/ \varepsilon^2)$ where $\varepsilon > 0$
stands for the desired tolerance.

    

### [[2108.08000] Contrastive Identification of Covariate Shift in Image Data](http://arxiv.org/abs/2108.08000)


  Identifying covariate shift is crucial for making machine learning systems
robust in the real world and for detecting training data biases that are not
reflected in test data. However, detecting covariate shift is challenging,
especially when the data consists of high-dimensional images, and when multiple
types of localized covariate shift affect different subspaces of the data.
Although automated techniques can be used to detect the existence of covariate
shift, our goal is to help human users characterize the extent of covariate
shift in large image datasets with interfaces that seamlessly integrate
information obtained from the detection algorithms. In this paper, we design
and evaluate a new visual interface that facilitates the comparison of the
local distributions of training and test data. We conduct a quantitative user
study on multi-attribute facial data to compare two different learned
low-dimensional latent representations (pretrained ImageNet CNN vs. density
ratio) and two user analytic workflows (nearest-neighbor vs.
cluster-to-cluster). Our results indicate that the latent representation of our
density ratio model, combined with a nearest-neighbor comparison, is the most
effective at helping humans identify covariate shift.

    

### [[2108.08003] Stochastic Cluster Embedding](http://arxiv.org/abs/2108.08003)


  Neighbor Embedding (NE) that aims to preserve pairwise similarities between
data items has been shown to yield an effective principle for data
visualization. However, even the currently best NE methods such as Stochastic
Neighbor Embedding (SNE) may leave large-scale patterns such as clusters hidden
despite of strong signals being present in the data. To address this, we
propose a new cluster visualization method based on Neighbor Embedding. We
first present a family of Neighbor Embedding methods which generalizes SNE by
using non-normalized Kullback-Leibler divergence with a scale parameter. In
this family, much better cluster visualizations often appear with a parameter
value different from the one corresponding to SNE. We also develop an efficient
software which employs asynchronous stochastic block coordinate descent to
optimize the new family of objective functions. The experimental results
demonstrate that our method consistently and substantially improves
visualization of data clusters compared with the state-of-the-art NE
approaches.

    

### [[2108.08009] XAI Methods for Neural Time Series Classification: A Brief Review](http://arxiv.org/abs/2108.08009)


  Deep learning models have recently demonstrated remarkable results in a
variety of tasks, which is why they are being increasingly applied in
high-stake domains, such as industry, medicine, and finance. Considering that
automatic predictions in these domains might have a substantial impact on the
well-being of a person, as well as considerable financial and legal
consequences to an individual or a company, all actions and decisions that
result from applying these models have to be accountable. Given that a
substantial amount of data that is collected in high-stake domains are in the
form of time series, in this paper we examine the current state of eXplainable
AI (XAI) methods with a focus on approaches for opening up deep learning black
boxes for the task of time series classification. Finally, our contribution
also aims at deriving promising directions for future work, to advance XAI for
deep learning on time series data.

    

### [[2108.08019] RANK-NOSH: Efficient Predictor-Based Architecture Search via Non-Uniform Successive Halving](http://arxiv.org/abs/2108.08019)


  Predictor-based algorithms have achieved remarkable performance in the Neural
Architecture Search (NAS) tasks. However, these methods suffer from high
computation costs, as training the performance predictor usually requires
training and evaluating hundreds of architectures from scratch. Previous works
along this line mainly focus on reducing the number of architectures required
to fit the predictor. In this work, we tackle this challenge from a different
perspective - improve search efficiency by cutting down the computation budget
of architecture training. We propose NOn-uniform Successive Halving (NOSH), a
hierarchical scheduling algorithm that terminates the training of
underperforming architectures early to avoid wasting budget. To effectively
leverage the non-uniform supervision signals produced by NOSH, we formulate
predictor-based architecture search as learning to rank with pairwise
comparisons. The resulting method - RANK-NOSH, reduces the search budget by ~5x
while achieving competitive or even better performance than previous
state-of-the-art predictor-based methods on various spaces and datasets.

    

### [[2108.08038] Combining K-means type algorithms with Hill Climbing for Joint Stratification and Sample Allocation Designs](http://arxiv.org/abs/2108.08038)


  In this paper we combine the k-means and/or k-means type algorithms with a
hill climbing algorithm in stages to solve the joint stratification and sample
allocation problem. This is a combinatorial optimisation problem in which we
search for the optimal stratification from the set of all possible
stratifications of basic strata. Each stratification being a solution the
quality of which is measured by its cost. This problem is intractable for
larger sets. Furthermore evaluating the cost of each solution is expensive. A
number of heuristic algorithms have already been developed to solve this
problem with the aim of finding acceptable solutions in reasonable computation
times. However, the heuristics for these algorithms need to be trained in order
to optimise performance in each instance. We compare the above multi-stage
combination of algorithms with three recent algorithms and report the solution
costs, evaluation times and training times. The multi-stage combinations
generally compare well with the recent algorithms both in the case of atomic
and continuous strata and provide the survey designer with a greater choice of
algorithms to choose from.

    

### [[2108.08041] DeepCVA: Automated Commit-level Vulnerability Assessment with Deep Multi-task Learning](http://arxiv.org/abs/2108.08041)


  It is increasingly suggested to identify Software Vulnerabilities (SVs) in
code commits to give early warnings about potential security risks. However,
there is a lack of effort to assess vulnerability-contributing commits right
after they are detected to provide timely information about the exploitability,
impact and severity of SVs. Such information is important to plan and
prioritize the mitigation for the identified SVs. We propose a novel Deep
multi-task learning model, DeepCVA, to automate seven Commit-level
Vulnerability Assessment tasks simultaneously based on Common Vulnerability
Scoring System (CVSS) metrics. We conduct large-scale experiments on 1,229
vulnerability-contributing commits containing 542 different SVs in 246
real-world software projects to evaluate the effectiveness and efficiency of
our model. We show that DeepCVA is the best-performing model with 38% to 59.8%
higher Matthews Correlation Coefficient than many supervised and unsupervised
baseline models. DeepCVA also requires 6.3 times less training and validation
time than seven cumulative assessment models, leading to significantly less
model maintenance cost as well. Overall, DeepCVA presents the first effective
and efficient solution to automatically assess SVs early in software systems.

    

### [[2108.08046] Variational Graph Normalized Auto-Encoders](http://arxiv.org/abs/2108.08046)


  Link prediction is one of the key problems for graph-structured data. With
the advancement of graph neural networks, graph autoencoders (GAEs) and
variational graph autoencoders (VGAEs) have been proposed to learn graph
embeddings in an unsupervised way. It has been shown that these methods are
effective for link prediction tasks. However, they do not work well in link
predictions when a node whose degree is zero (i.g., isolated node) is involved.
We have found that GAEs/VGAEs make embeddings of isolated nodes close to zero
regardless of their content features. In this paper, we propose a novel
Variational Graph Normalized AutoEncoder (VGNAE) that utilize
$L_2$-normalization to derive better embeddings for isolated nodes. We show
that our VGNAEs outperform the existing state-of-the-art models for link
prediction tasks. The code is available at
this https URL.

    

### [[2108.08052] Moser Flow: Divergence-based Generative Modeling on Manifolds](http://arxiv.org/abs/2108.08052)


  We are interested in learning generative models for complex geometries
described via manifolds, such as spheres, tori, and other implicit surfaces.
Current extensions of existing (Euclidean) generative models are restricted to
specific geometries and typically suffer from high computational costs. We
introduce Moser Flow (MF), a new class of generative models within the family
of continuous normalizing flows (CNF). MF also produces a CNF via a solution to
the change-of-variable formula, however differently from other CNF methods, its
model (learned) density is parameterized as the source (prior) density minus
the divergence of a neural network (NN). The divergence is a local, linear
differential operator, easy to approximate and calculate on manifolds.
Therefore, unlike other CNFs, MF does not require invoking or backpropagating
through an ODE solver during training. Furthermore, representing the model
density explicitly as the divergence of a NN rather than as a solution of an
ODE facilitates learning high fidelity densities. Theoretically, we prove that
MF constitutes a universal density approximator under suitable assumptions.
Empirically, we demonstrate for the first time the use of flow models for
sampling from general curved surfaces and achieve significant improvements in
density estimation, sample quality, and training complexity over existing CNFs
on challenging synthetic geometries and real-world benchmarks from the earth
and climate sciences.

    

### [[2108.08077] Towards Interpreting Zoonotic Potential of Betacoronavirus Sequences With Attention](http://arxiv.org/abs/2108.08077)


  Current methods for viral discovery target evolutionarily conserved proteins
that accurately identify virus families but remain unable to distinguish the
zoonotic potential of newly discovered viruses. Here, we apply an
attention-enhanced long-short-term memory (LSTM) deep neural net classifier to
a highly conserved viral protein target to predict zoonotic potential across
betacoronaviruses. The classifier performs with a 94% accuracy. Analysis and
visualization of attention at the sequence and structure-level features
indicate possible association between important protein-protein interactions
governing viral replication in zoonotic betacoronaviruses and zoonotic
transmission.

    

### [[2108.08095] DRDrV3: Complete Lesion Detection in Fundus Images Using Mask R-CNN, Transfer Learning, and LSTM](http://arxiv.org/abs/2108.08095)


  Medical Imaging is one of the growing fields in the world of computer vision.
In this study, we aim to address the Diabetic Retinopathy (DR) problem as one
of the open challenges in medical imaging. In this research, we propose a new
lesion detection architecture, comprising of two sub-modules, which is an
optimal solution to detect and find not only the type of lesions caused by DR,
their corresponding bounding boxes, and their masks; but also the severity
level of the overall case. Aside from traditional accuracy, we also use two
popular evaluation criteria to evaluate the outputs of our models, which are
intersection over union (IOU) and mean average precision (mAP). We hypothesize
that this new solution enables specialists to detect lesions with high
confidence and estimate the severity of the damage with high accuracy.

    

### [[2108.08105] Deep Graph Memory Networks for Forgetting-Robust Knowledge Tracing](http://arxiv.org/abs/2108.08105)


  Tracing a student's knowledge is vital for tailoring the learning experience.
Recent knowledge tracing methods tend to respond to these challenges by
modelling knowledge state dynamics across learning concepts. However, they
still suffer from several inherent challenges including: modelling forgetting
behaviours and identifying relationships among latent concepts. To address
these challenges, in this paper, we propose a novel knowledge tracing model,
namely \emph{Deep Graph Memory Network} (DGMN). In this model, we incorporate a
forget gating mechanism into an attention memory structure in order to capture
forgetting behaviours dynamically during the knowledge tracing process.
Particularly, this forget gating mechanism is built upon attention forgetting
features over latent concepts considering their mutual dependencies. Further,
this model has the capability of learning relationships between latent concepts
from a dynamic latent concept graph in light of a student's evolving knowledge
states. A comprehensive experimental evaluation has been conducted using four
well-established benchmark datasets. The results show that DGMN consistently
outperforms the state-of-the-art KT models over all the datasets. The
effectiveness of modelling forgetting behaviours and learning latent concept
graphs has also been analyzed in our experiments.

    

### [[2108.08106] Existence, uniqueness, and convergence rates for gradient flows in the training of artificial neural networks with ReLU activation](http://arxiv.org/abs/2108.08106)


  The training of artificial neural networks (ANNs) with rectified linear unit
(ReLU) activation via gradient descent (GD) type optimization schemes is
nowadays a common industrially relevant procedure. Till this day in the
scientific literature there is in general no mathematical convergence analysis
which explains the numerical success of GD type optimization schemes in the
training of ANNs with ReLU activation. GD type optimization schemes can be
regarded as temporal discretization methods for the gradient flow (GF)
differential equations associated to the considered optimization problem and,
in view of this, it seems to be a natural direction of research to first aim to
develop a mathematical convergence theory for time-continuous GF differential
equations and, thereafter, to aim to extend such a time-continuous convergence
theory to implementable time-discrete GD type optimization methods. In this
article we establish two basic results for GF differential equations in the
training of fully-connected feedforward ANNs with one hidden layer and ReLU
activation. In the first main result of this article we establish in the
training of such ANNs under the assumption that the probability distribution of
the input data of the considered supervised learning problem is absolutely
continuous with a bounded density function that every GF differential equation
admits for every initial value a solution which is also unique among a suitable
class of solutions. In the second main result of this article we prove in the
training of such ANNs under the assumption that the target function and the
density function of the probability distribution of the input data are
piecewise polynomial that every non-divergent GF trajectory converges with an
appropriate rate of convergence to a critical point and that the risk of the
non-divergent GF trajectory converges with rate 1 to the risk of the critical
point.

    

### [[2108.08120] Stack Index Prediction Using Time-Series Analysis](http://arxiv.org/abs/2108.08120)


  The Prevalence of Community support and engagement for different domains in
the tech industry has changed and evolved throughout the years. In this study,
we aim to understand, analyze and predict the trends of technology in a
scientific manner, having collected data on numerous topics and their growth
throughout the years in the past decade. We apply machine learning models on
collected data, to understand, analyze and forecast the trends in the
advancement of different fields. We show that certain technical concepts such
as python, machine learning, and Keras have an undisputed uptrend, finally
concluding that the Stackindex model forecasts with high accuracy and can be a
viable tool for forecasting different tech domains.

    

### [[2108.08128] Single-DARTS: Towards Stable Architecture Search](http://arxiv.org/abs/2108.08128)


  Differentiable architecture search (DARTS) marks a milestone in Neural
Architecture Search (NAS), boasting simplicity and small search costs. However,
DARTS still suffers from frequent performance collapse, which happens when some
operations, such as skip connections, zeroes and poolings, dominate the
architecture. In this paper, we are the first to point out that the phenomenon
is attributed to bi-level optimization. We propose Single-DARTS which merely
uses single-level optimization, updating network weights and architecture
parameters simultaneously with the same data batch. Even single-level
optimization has been previously attempted, no literature provides a systematic
explanation on this essential point. Replacing the bi-level optimization,
Single-DARTS obviously alleviates performance collapse as well as enhances the
stability of architecture search. Experiment results show that Single-DARTS
achieves state-of-the-art performance on mainstream search spaces. For
instance, on NAS-Benchmark-201, the searched architectures are nearly optimal
ones. We also validate that the single-level optimization framework is much
more stable than the bi-level one. We hope that this simple yet effective
method will give some insights on differential architecture search. The code is
available at this https URL.

    

### [[2108.08129] Quantitative Uniform Stability of the Iterative Proportional Fitting Procedure](http://arxiv.org/abs/2108.08129)


  We establish the uniform in time stability, w.r.t. the marginals, of the
Iterative Proportional Fitting Procedure, also known as Sinkhorn algorithm,
used to solve entropy-regularised Optimal Transport problems. Our result is
quantitative and stated in terms of the 1-Wasserstein metric. As a corollary we
establish a quantitative stability result for Schrödinger bridges.

    

### [[2108.08136] Optimising Knee Injury Detection with Spatial Attention and Validating Localisation Ability](http://arxiv.org/abs/2108.08136)


  This work employs a pre-trained, multi-view Convolutional Neural Network
(CNN) with a spatial attention block to optimise knee injury detection. An
open-source Magnetic Resonance Imaging (MRI) data set with image-level labels
was leveraged for this analysis. As MRI data is acquired from three planes, we
compare our technique using data from a single-plane and multiple planes
(multi-plane). For multi-plane, we investigate various methods of fusing the
planes in the network. This analysis resulted in the novel 'MPFuseNet' network
and state-of-the-art Area Under the Curve (AUC) scores for detecting Anterior
Cruciate Ligament (ACL) tears and Abnormal MRIs, achieving AUC scores of 0.977
and 0.957 respectively. We then developed an objective metric, Penalised
Localisation Accuracy (PLA), to validate the model's localisation ability. This
metric compares binary masks generated from Grad-Cam output and the
radiologist's annotations on a sample of MRIs. We also extracted explainability
features in a model-agnostic approach that were then verified as clinically
relevant by the radiologist.

    

### [[2108.08143] Effective and scalable clustering of SARS-CoV-2 sequences](http://arxiv.org/abs/2108.08143)


  SARS-CoV-2, like any other virus, continues to mutate as it spreads,
according to an evolutionary process. Unlike any other virus, the number of
currently available sequences of SARS-CoV-2 in public databases such as GISAID
is already several million. This amount of data has the potential to uncover
the evolutionary dynamics of a virus like never before. However, a million is
already several orders of magnitude beyond what can be processed by the
traditional methods designed to reconstruct a virus's evolutionary history,
such as those that build a phylogenetic tree. Hence, new and scalable methods
will need to be devised in order to make use of the ever increasing number of
viral sequences being collected.
Since identifying variants is an important part of understanding the
evolution of a virus, in this paper, we propose an approach based on clustering
sequences to identify the current major SARS-CoV-2 variants. Using a $k$-mer
based feature vector generation and efficient feature selection methods, our
approach is effective in identifying variants, as well as being efficient and
scalable to millions of sequences. Such a clustering method allows us to show
the relative proportion of each variant over time, giving the rate of spread of
each variant in different locations -- something which is important for vaccine
development and distribution. We also compute the importance of each amino acid
position of the spike protein in identifying a given variant in terms of
information gain. Positions of high variant-specific importance tend to agree
with those reported by the USA's Centers for Disease Control and Prevention
(CDC), further demonstrating our approach.

    

### [[2108.08157] Towards Deep and Efficient: A Deep Siamese Self-Attention Fully Efficient Convolutional Network for Change Detection in VHR Images](http://arxiv.org/abs/2108.08157)


  Recently, FCNs have attracted widespread attention in the CD field. In
pursuit of better CD performance, it has become a tendency to design deeper and
more complicated FCNs, which inevitably brings about huge numbers of parameters
and an unbearable computational burden. With the goal of designing a quite deep
architecture to obtain more precise CD results while simultaneously decreasing
parameter numbers to improve efficiency, in this work, we present a very deep
and efficient CD network, entitled EffCDNet. In EffCDNet, to reduce the
numerous parameters associated with deep architecture, an efficient convolution
consisting of depth-wise convolution and group convolution with a channel
shuffle mechanism is introduced to replace standard convolutional layers. In
terms of the specific network architecture, EffCDNet does not use mainstream
UNet-like architecture, but rather adopts the architecture with a very deep
encoder and a lightweight decoder. In the very deep encoder, two very deep
siamese streams stacked by efficient convolution first extract two highly
representative and informative feature maps from input image-pairs.
Subsequently, an efficient ASPP module is designed to capture multi-scale
change information. In the lightweight decoder, a recurrent criss-cross
self-attention (RCCA) module is applied to efficiently utilize non-local
similar feature representations to enhance discriminability for each pixel,
thus effectively separating the changed and unchanged regions. Moreover, to
tackle the optimization problem in confused pixels, two novel loss functions
based on information entropy are presented. On two challenging CD datasets, our
approach outperforms other SOTA FCN-based methods, with only benchmark-level
parameter numbers and quite low computational overhead.

    

### [[2108.08170] DeepExpress: Heterogeneous and Coupled Sequence Modeling for Express Delivery Prediction](http://arxiv.org/abs/2108.08170)


  The prediction of express delivery sequence, i.e., modeling and estimating
the volumes of daily incoming and outgoing parcels for delivery, is critical
for online business, logistics, and positive customer experience, and
specifically for resource allocation optimization and promotional activity
arrangement. A precise estimate of consumer delivery requests has to involve
sequential factors such as shopping behaviors, weather conditions, events,
business campaigns, and their couplings. Besides, conventional sequence
prediction assumes a stable sequence evolution, failing to address complex
nonlinear sequences and various feature effects in the above multi-source data.
Although deep networks and attention mechanisms demonstrate the potential of
complex sequence modeling, extant networks ignore the heterogeneous and
coupling situation between features and sequences, resulting in weak prediction
accuracy. To address these issues, we propose DeepExpress - a deep-learning
based express delivery sequence prediction model, which extends the classic
seq2seq framework to learning complex coupling between sequence and features.
DeepExpress leverages an express delivery seq2seq learning, a
carefully-designed heterogeneous feature representation, and a novel joint
training attention mechanism to adaptively map heterogeneous data, and capture
sequence-feature coupling for precise estimation. Experimental results on
real-world data demonstrate that the proposed method outperforms both shallow
and deep baseline models.

    

### [[2108.08180] Structure Parameter Optimized Kernel Based Online Prediction with a Generalized Optimization Strategy for Nonstationary Time Series](http://arxiv.org/abs/2108.08180)


  In this paper, sparsification techniques aided online prediction algorithms
in a reproducing kernel Hilbert space are studied for nonstationary time
series. The online prediction algorithms as usual consist of the selection of
kernel structure parameters and the kernel weight vector updating. For
structure parameters, the kernel dictionary is selected by some sparsification
techniques with online selective modeling criteria, and moreover the kernel
covariance matrix is intermittently optimized in the light of the covariance
matrix adaptation evolution strategy (CMA-ES). Optimizing the real symmetric
covariance matrix can not only improve the kernel structure's flexibility by
the cross relatedness of the input variables, but also partly alleviate the
prediction uncertainty caused by the kernel dictionary selection for
nonstationary time series. In order to sufficiently capture the underlying
dynamic characteristics in prediction-error time series, a generalized
optimization strategy is designed to construct the kernel dictionary
sequentially in multiple kernel connection modes. The generalized optimization
strategy provides a more self-contained way to construct the entire kernel
connections, which enhances the ability to adaptively track the changing
dynamic characteristics. Numerical simulations have demonstrated that the
proposed approach has superior prediction performance for nonstationary time
series.

    

### [[2108.08186] Generalizing MLPs With Dropouts, Batch Normalization, and Skip Connections](http://arxiv.org/abs/2108.08186)


  A multilayer perceptron (MLP) is typically made of multiple fully connected
layers with nonlinear activation functions. There have been several approaches
to make them better (e.g. faster convergence, better convergence limit, etc.).
But the researches lack in more structured ways to test them. We test different
MLP architectures by carrying out the experiments on the age and gender
datasets. We empirically show that by whitening inputs before every linear
layer and adding skip connections, our proposed MLP architecture can result in
better performance. Since the whitening process includes dropouts, it can also
be used to approximate Bayesian inference. We have open sourced our code
released models and docker images at this https URL.

    

### [[2108.08189] FOX-NAS: Fast, On-device and Explainable Neural Architecture Search](http://arxiv.org/abs/2108.08189)


  Neural architecture search can discover neural networks with good
performance, and One-Shot approaches are prevalent. One-Shot approaches
typically require a supernet with weight sharing and predictors that predict
the performance of architecture. However, the previous methods take much time
to generate performance predictors thus are inefficient. To this end, we
propose FOX-NAS that consists of fast and explainable predictors based on
simulated annealing and multivariate regression. Our method is
quantization-friendly and can be efficiently deployed to the edge. The
experiments on different hardware show that FOX-NAS models outperform some
other popular neural network architectures. For example, FOX-NAS matches
MobileNetV2 and EfficientNet-Lite0 accuracy with 240% and 40% less latency on
the edge CPU. FOX-NAS is the 3rd place winner of the 2020 Low-Power Computer
Vision Challenge (LPCVC), DSP classification track. See all evaluation results
at this https URL. Search code and pre-trained models are
released at this https URL.

    

### [[2108.08195] ALLNet: A Hybrid Convolutional Neural Network to Improve Diagnosis of Acute Lymphocytic Leukemia (ALL) in White Blood Cells](http://arxiv.org/abs/2108.08195)


  Due to morphological similarity at the microscopic level, making an accurate
and time-sensitive distinction between blood cells affected by Acute
Lymphocytic Leukemia (ALL) and their healthy counterparts calls for the usage
of machine learning architectures. However, three of the most common models,
VGG, ResNet, and Inception, each come with their own set of flaws with room for
improvement which demands the need for a superior model. ALLNet, the proposed
hybrid convolutional neural network architecture, consists of a combination of
the VGG, ResNet, and Inception models. The ALL Challenge dataset of ISBI 2019
(available here) contains 10,691 images of white blood cells which were used to
train and test the models. 7,272 of the images in the dataset are of cells with
ALL and 3,419 of them are of healthy cells. Of the images, 60% were used to
train the model, 20% were used for the cross-validation set, and 20% were used
for the test set. ALLNet outperformed the VGG, ResNet, and the Inception models
across the board, achieving an accuracy of 92.6567%, a sensitivity of 95.5304%,
a specificity of 85.9155%, an AUC score of 0.966347, and an F1 score of 0.94803
in the cross-validation set. In the test set, ALLNet achieved an accuracy of
92.0991%, a sensitivity of 96.5446%, a specificity of 82.8035%, an AUC score of
0.959972, and an F1 score of 0.942963. The utilization of ALLNet in the
clinical workspace can better treat the thousands of people suffering from ALL
across the world, many of whom are children.

    

### [[2108.08197] CARE: Coherent Actionable Recourse based on Sound Counterfactual Explanations](http://arxiv.org/abs/2108.08197)


  Counterfactual explanation methods interpret the outputs of a machine
learning model in the form of "what-if scenarios" without compromising the
fidelity-interpretability trade-off. They explain how to obtain a desired
prediction from the model by recommending small changes to the input features,
aka recourse. We believe an actionable recourse should be created based on
sound counterfactual explanations originating from the distribution of the
ground-truth data and linked to the domain knowledge. Moreover, it needs to
preserve the coherency between changed/unchanged features while satisfying
user/domain-specified constraints. This paper introduces CARE, a modular
explanation framework that addresses the model- and user-level desiderata in a
consecutive and structured manner. We tackle the existing requirements by
proposing novel and efficient solutions that are formulated in a
multi-objective optimization framework. The designed framework enables
including arbitrary requirements and generating counterfactual explanations and
actionable recourse by choice. As a model-agnostic approach, CARE generates
multiple, diverse explanations for any black-box model in tabular
classification and regression settings. Several experiments on standard data
sets and black-box models demonstrate the effectiveness of our modular
framework and its superior performance compared to the baselines.

    

### [[2108.08212] Confidence Adaptive Regularization for Deep Learning with Noisy Labels](http://arxiv.org/abs/2108.08212)


  Recent studies on the memorization effects of deep neural networks on noisy
labels show that the networks first fit the correctly-labeled training samples
before memorizing the mislabeled samples. Motivated by this early-learning
phenomenon, we propose a novel method to prevent memorization of the mislabeled
samples. Unlike the existing approaches which use the model output to identify
or ignore the mislabeled samples, we introduce an indicator branch to the
original model and enable the model to produce a confidence value for each
sample. The confidence values are incorporated in our loss function which is
learned to assign large confidence values to correctly-labeled samples and
small confidence values to mislabeled samples. We also propose an auxiliary
regularization term to further improve the robustness of the model. To improve
the performance, we gradually correct the noisy labels with a well-designed
target estimation strategy. We provide the theoretical analysis and conduct the
experiments on synthetic and real-world datasets, demonstrating that our
approach achieves comparable results to the state-of-the-art methods.

    

### [[2108.08214] Distinguishing Healthy Ageing from Dementia: a Biomechanical Simulation of Brain Atrophy using Deep Networks](http://arxiv.org/abs/2108.08214)


  Biomechanical modeling of tissue deformation can be used to simulate
different scenarios of longitudinal brain evolution. In this work,we present a
deep learning framework for hyper-elastic strain modelling of brain atrophy,
during healthy ageing and in Alzheimer's Disease. The framework directly models
the effects of age, disease status, and scan interval to regress regional
patterns of atrophy, from which a strain-based model estimates deformations.
This model is trained and validated using 3D structural magnetic resonance
imaging data from the ADNI cohort. Results show that the framework can estimate
realistic deformations, following the known course of Alzheimer's disease, that
clearly differentiate between healthy and demented patterns of ageing. This
suggests the framework has potential to be incorporated into explainable models
of disease, for the exploration of interventions and counterfactual examples.

    

### [[2108.08217] X-modaler: A Versatile and High-performance Codebase for Cross-modal Analytics](http://arxiv.org/abs/2108.08217)


  With the rise and development of deep learning over the past decade, there
has been a steady momentum of innovation and breakthroughs that convincingly
push the state-of-the-art of cross-modal analytics between vision and language
in multimedia field. Nevertheless, there has not been an open-source codebase
in support of training and deploying numerous neural network models for
cross-modal analytics in a unified and modular fashion. In this work, we
propose X-modaler -- a versatile and high-performance codebase that
encapsulates the state-of-the-art cross-modal analytics into several
general-purpose stages (e.g., pre-processing, encoder, cross-modal interaction,
decoder, and decode strategy). Each stage is empowered with the functionality
that covers a series of modules widely adopted in state-of-the-arts and allows
seamless switching in between. This way naturally enables a flexible
implementation of state-of-the-art algorithms for image captioning, video
captioning, and vision-language pre-training, aiming to facilitate the rapid
development of research community. Meanwhile, since the effective modular
designs in several stages (e.g., cross-modal interaction) are shared across
different vision-language tasks, X-modaler can be simply extended to power
startup prototypes for other tasks in cross-modal analytics, including visual
question answering, visual commonsense reasoning, and cross-modal retrieval.
X-modaler is an Apache-licensed codebase, and its source codes, sample projects
and pre-trained models are available on-line:
this https URL.

    

### [[2108.08218] Out-of-Distribution Detection using Outlier Detection Methods](http://arxiv.org/abs/2108.08218)


  Out-of-distribution detection (OOD) deals with anomalous input to neural
networks. In the past, specialized methods have been proposed to reject
predictions on anomalous input. We use outlier detection algorithms to detect
anomalous input as reliable as specialized methods from the field of OOD. No
neural network adaptation is required; detection is based on the model's
softmax score. Our approach works unsupervised with an Isolation Forest or with
supervised classifiers such as a Gradient Boosting machine.

    

### [[2108.08224] Transformers predicting the future. Applying attention in next-frame and time series forecasting](http://arxiv.org/abs/2108.08224)


  Recurrent Neural Networks were, until recently, one of the best ways to
capture the timely dependencies in sequences. However, with the introduction of
the Transformer, it has been proven that an architecture with only
attention-mechanisms without any RNN can improve on the results in various
sequence processing tasks (e.g. NLP). Multiple studies since then have shown
that similar approaches can be applied for images, point clouds, video, audio
or time series forecasting. Furthermore, solutions such as the Perceiver or the
Informer have been introduced to expand on the applicability of the
Transformer. Our main objective is testing and evaluating the effectiveness of
applying Transformer-like models on time series data, tackling susceptibility
to anomalies, context awareness and space complexity by fine-tuning the
hyperparameters, preprocessing the data, applying dimensionality reduction or
convolutional encodings, etc. We are also looking at the problem of next-frame
prediction and exploring ways to modify existing solutions in order to achieve
higher performance and learn generalized knowledge.

    

### [[2108.08230] Predicting Dynamic Stability of Power Grids using Graph Neural Networks](http://arxiv.org/abs/2108.08230)


  The prediction of dynamical stability of power grids becomes more important
and challenging with increasing shares of renewable energy sources due to their
decentralized structure, reduced inertia and volatility. We investigate the
feasibility of applying graph neural networks (GNN) to predict dynamic
stability of synchronisation in complex power grids using the single-node basin
stability (SNBS) as a measure. To do so, we generate two synthetic datasets for
grids with 20 and 100 nodes respectively and estimate SNBS using Monte-Carlo
sampling. Those datasets are used to train and evaluate the performance of
eight different GNN-models. All models use the full graph without
simplifications as input and predict SNBS in a nodal-regression-setup. We show
that SNBS can be predicted in general and the performance significantly changes
using different GNN-models. Furthermore, we observe interesting transfer
capabilities of our approach: GNN-models trained on smaller grids can directly
be applied on larger grids without the need of retraining.

    

### [[2108.08236] LOKI: Long Term and Key Intentions for Trajectory Prediction](http://arxiv.org/abs/2108.08236)


  Recent advances in trajectory prediction have shown that explicit reasoning
about agents' intent is important to accurately forecast their motion. However,
the current research activities are not directly applicable to intelligent and
safety critical systems. This is mainly because very few public datasets are
available, and they only consider pedestrian-specific intents for a short
temporal horizon from a restricted egocentric view. To this end, we propose
LOKI (LOng term and Key Intentions), a novel large-scale dataset that is
designed to tackle joint trajectory and intention prediction for heterogeneous
traffic agents (pedestrians and vehicles) in an autonomous driving setting. The
LOKI dataset is created to discover several factors that may affect intention,
including i) agent's own will, ii) social interactions, iii) environmental
constraints, and iv) contextual information. We also propose a model that
jointly performs trajectory and intention prediction, showing that recurrently
reasoning about intention can assist with trajectory prediction. We show our
method outperforms state-of-the-art trajectory prediction methods by upto
$27\%$ and also provide a baseline for frame-wise intention estimation.

    

### [[2108.08253] Learned holographic light transport](http://arxiv.org/abs/2108.08253)


  Computer-Generated Holography (CGH) algorithms often fall short in matching
simulations with results from a physical holographic display. Our work
addresses this mismatch by learning the holographic light transport in
holographic displays. Using a camera and a holographic display, we capture the
image reconstructions of optimized holograms that rely on ideal simulations to
generate a dataset. Inspired by the ideal simulations, we learn a
complex-valued convolution kernel that can propagate given holograms to
captured photographs in our dataset. Our method can dramatically improve
simulation accuracy and image quality in holographic displays while paving the
way for physically informed learning approaches.

    

### [[2108.08262] SOME/IP Intrusion Detection using Deep Learning-based Sequential Models in Automotive Ethernet Networks](http://arxiv.org/abs/2108.08262)


  Intrusion Detection Systems are widely used to detect cyberattacks,
especially on protocols vulnerable to hacking attacks such as SOME/IP. In this
paper, we present a deep learning-based sequential model for offline intrusion
detection on SOME/IP application layer protocol. To assess our intrusion
detection system, we have generated and labeled a dataset with several classes
representing realistic intrusions, and a normal class - a significant
contribution due to the absence of such publicly available datasets.
Furthermore, we also propose a simple recurrent neural network (RNN), as an
instance of deep learning-based sequential model, that we apply to our
generated dataset. The numerical results show that RNN excel at predicting
in-vehicle intrusions, with F1 Scores and AUC values of 0.99 for each type of
intrusion.

    

### [[2108.08264] Fake News and Phishing Detection Using a Machine Learning Trained Expert System](http://arxiv.org/abs/2108.08264)


  Expert systems have been used to enable computers to make recommendations and
decisions. This paper presents the use of a machine learning trained expert
system (MLES) for phishing site detection and fake news detection. Both topics
share a similar goal: to design a rule-fact network that allows a computer to
make explainable decisions like domain experts in each respective area. The
phishing website detection study uses a MLES to detect potential phishing
websites by analyzing site properties (like URL length and expiration time).
The fake news detection study uses a MLES rule-fact network to gauge news story
truthfulness based on factors such as emotion, the speaker's political
affiliation status, and job. The two studies use different MLES network
implementations, which are presented and compared herein. The fake news study
utilized a more linear design while the phishing project utilized a more
complex connection structure. Both networks' inputs are based on commonly
available data sets.

    

### [[2108.08275] TB-ICT: A Trustworthy Blockchain-Enabled System for Indoor COVID-19 Contact Tracing](http://arxiv.org/abs/2108.08275)


  Recently, as a consequence of the COVID-19 pandemic, dependence on Contact
Tracing (CT) models has significantly increased to prevent spread of this
highly contagious virus and be prepared for the potential future ones. Since
the spreading probability of the novel coronavirus in indoor environments is
much higher than that of the outdoors, there is an urgent and unmet quest to
develop/design efficient, autonomous, trustworthy, and secure indoor CT
solutions. Despite such an urgency, this field is still in its infancy. The
paper addresses this gap and proposes the Trustworthy Blockchain-enabled system
for Indoor Contact Tracing (TB-ICT) framework. The TB-ICT framework is proposed
to protect privacy and integrity of the underlying CT data from unauthorized
access. More specifically, it is a fully distributed and innovative blockchain
platform exploiting the proposed dynamic Proof of Work (dPoW) credit-based
consensus algorithm coupled with Randomized Hash Window (W-Hash) and dynamic
Proof of Credit (dPoC) mechanisms to differentiate between honest and dishonest
nodes. The TB-ICT not only provides a decentralization in data replication but
also quantifies the node's behavior based on its underlying credit-based
mechanism. For achieving high localization performance, we capitalize on
availability of Internet of Things (IoT) indoor localization infrastructures,
and develop a data driven localization model based on Bluetooth Low Energy
(BLE) sensor measurements. The simulation results show that the proposed TB-ICT
prevents the COVID-19 from spreading by implementation of a highly accurate
contact tracing model while improving the users' privacy and security.

    

### [[2108.08282] OACAL: Finding Module-Consistent Solutions to Weaken User Obligations](http://arxiv.org/abs/2108.08282)


  Users interacting with a UI-embedded machine or system are typically obliged
to perform their actions in a pre-determined order, to successfully achieve
certain functional goals. However, such obligations are often not followed
strictly by users, which may lead to the violation to security properties,
especially in security-critical systems. In order to improve the security with
the awareness of unexpected user behaviors, a system can be redesigned to a
more robust one by changing the order of actions in its specification.
Meanwhile, we anticipate that the functionalities would remain consistent
following the modifications. In this paper, we propose an efficient algorithm
to automatically produce specification revisions tackling with attack scenarios
caused by the weakened user obligations. By our algorithm, all the revisions
maintain the integrity of the functionalities as the original specification,
which are generated using a novel recomposition approach. Then, the qualified
revisions that can satisfy the security requirements would be efficiently
spotted by a hybrid approach combining model checking and machine learning
techniques. We evaluate our algorithm by comparing its performance with a
state-of-the-art approach regarding their coverage and searching speed of the
desirable revisions.

    

### [[1905.11589] Learning distant cause and effect using only local and immediate credit assignment](http://arxiv.org/abs/1905.11589)


  We present a recurrent neural network memory that uses sparse coding to
create a combinatoric encoding of sequential inputs. Using several examples, we
show that the network can associate distant causes and effects in a discrete
stochastic process, predict partially-observable higher-order sequences, and
enable a DQN agent to navigate a maze by giving it memory. The network uses
only biologically-plausible, local and immediate credit assignment. Memory
requirements are typically one order of magnitude less than existing LSTM, GRU
and autoregressive feed-forward sequence learning models. The most significant
limitation of the memory is generalization to unseen input sequences. We
explore this limitation by measuring next-word prediction perplexity on the
Penn Treebank dataset.

    

### [[1907.06592] Sparsely Activated Networks](http://arxiv.org/abs/1907.06592)


  Previous literature on unsupervised learning focused on designing structural
priors with the aim of learning meaningful features. However, this was done
without considering the description length of the learned representations which
is a direct and unbiased measure of the model complexity. In this paper, first
we introduce the $\varphi$ metric that evaluates unsupervised models based on
their reconstruction accuracy and the degree of compression of their internal
representations. We then present and define two activation functions (Identity,
ReLU) as base of reference and three sparse activation functions (top-k
absolutes, Extrema-Pool indices, Extrema) as candidate structures that minimize
the previously defined $\varphi$. We lastly present Sparsely Activated Networks
(SANs) that consist of kernels with shared weights that, during encoding, are
convolved with the input and then passed through a sparse activation function.
During decoding, the same weights are convolved with the sparse activation map
and subsequently the partial reconstructions from each weight are summed to
reconstruct the input. We compare SANs using the five previously defined
activation functions on a variety of datasets (Physionet, UCI-epilepsy, MNIST,
FMNIST) and show that models that are selected using $\varphi$ have small
description representation length and consist of interpretable kernels.

    

### [[1910.09499] Supervised tensor decomposition with features on multiple modes](http://arxiv.org/abs/1910.09499)


  Higher-order tensors have received increased attention across science and
engineering. While most tensor decomposition methods are developed for a single
tensor observation, scientific studies often collect side information, in the
form of node features and interactions thereof, together with the tensor data.
Such data problems are common in neuroimaging, network analysis, and
spatial-temporal modeling. Identifying the relationship between a
high-dimensional tensor and side information is important yet challenging.
Here, we develop a tensor decomposition method that incorporates multiple
feature matrices as side information. Unlike unsupervised tensor decomposition,
our supervised decomposition captures the effective dimension reduction of the
data tensor confined to feature space of interest. An efficient alternating
optimization algorithm with provable spectral initialization is further
developed. Our proposal handles a broad range of data types, including
continuous, count, and binary observations. We apply the method to diffusion
tensor imaging data from human connectome project and multi-relational
political network data. We identify the key global connectivity pattern and
pinpoint the local regions that are associated with available features. Our
simulation code, R-package tensorregress, and datasets used in the paper are
available at this https URL.

    

### [[1911.00400] Sparsely Activated Networks: A new method for decomposing and compressing data](http://arxiv.org/abs/1911.00400)


  Recent literature on unsupervised learning focused on designing structural
priors with the aim of learning meaningful features, but without considering
the description length of the representations. In this thesis, first we
introduce the $\varphi$ metric that evaluates unsupervised models based on
their reconstruction accuracy and the degree of compression of their internal
representations. We then present and define two activation functions (Identity,
ReLU) as base of reference and three sparse activation functions (top-k
absolutes, Extrema-Pool indices, Extrema) as candidate structures that minimize
the previously defined metric $\varphi$. We lastly present Sparsely Activated
Networks (SANs) that consist of kernels with shared weights that, during
encoding, are convolved with the input and then passed through a sparse
activation function. During decoding, the same weights are convolved with the
sparse activation map and subsequently the partial reconstructions from each
weight are summed to reconstruct the input. We compare SANs using the five
previously defined activation functions on a variety of datasets (Physionet,
UCI-epilepsy, MNIST, FMNIST) and show that models that are selected using
$\varphi$ have small description representation length and consist of
interpretable kernels.

    

### [[1911.08689] Corruption-robust exploration in episodic reinforcement learning](http://arxiv.org/abs/1911.08689)


  We initiate the study of multi-stage episodic reinforcement learning under
adversarial corruptions in both the rewards and the transition probabilities of
the underlying system extending recent results for the special case of
stochastic bandits. We provide a framework which modifies the aggressive
exploration enjoyed by existing reinforcement learning approaches based on
"optimism in the face of uncertainty", by complementing them with principles
from "action elimination". Importantly, our framework circumvents the major
challenges posed by naively applying action elimination in the RL setting, as
formalized by a lower bound we demonstrate. Our framework yields efficient
algorithms which (a) attain near-optimal regret in the absence of corruptions
and (b) adapt to unknown levels corruption, enjoying regret guarantees which
degrade gracefully in the total corruption encountered. To showcase the
generality of our approach, we derive results for both tabular settings (where
states and actions are finite) as well as linear-function-approximation
settings (where the dynamics and rewards admit a linear underlying
representation). Notably, our work provides the first sublinear regret
guarantee which accommodates any deviation from purely i.i.d. transitions in
the bandit-feedback model for episodic reinforcement learning.

    

### [[2003.03021] Exploiting Verified Neural Networks via Floating Point Numerical Error](http://arxiv.org/abs/2003.03021)


  Researchers have developed neural network verification algorithms motivated
by the need to characterize the robustness of deep neural networks. The
verifiers aspire to answer whether a neural network guarantees certain
properties with respect to all inputs in a space. However, many verifiers
inaccurately model floating point arithmetic but do not thoroughly discuss the
consequences.
We show that the negligence of floating point error leads to unsound
verification that can be systematically exploited in practice. For a pretrained
neural network, we present a method that efficiently searches inputs as
witnesses for the incorrectness of robustness claims made by a complete
verifier. We also present a method to construct neural network architectures
and weights that induce wrong results of an incomplete verifier. Our results
highlight that, to achieve practically reliable verification of neural
networks, any verification system must accurately (or conservatively) model the
effects of any floating point computations in the network inference or
verification system.

    

### [[2006.04379] Schrödinger PCA: On the Duality between Principal Component Analysis and Schrödinger Equation](http://arxiv.org/abs/2006.04379)


  Principal component analysis (PCA) has achieved great success in unsupervised
learning by identifying covariance correlations among features. If the data
collection fails to capture the covariance information, PCA will not be able to
discover meaningful modes. In particular, PCA will fail the spatial Gaussian
Process (GP) model in the undersampling regime, i.e. the averaged distance of
neighboring anchor points (spatial features) is greater than the correlation
length of GP. Counterintuitively, by drawing the connection between PCA and
Schrödinger equation, we can not only attack the undersampling challenge but
also compute in an efficient and decoupled way with the proposed algorithm
called Schrödinger PCA. Our algorithm only requires variances of features and
estimated correlation length as input, constructs the corresponding
Schrödinger equation, and solves it to obtain the energy eigenstates, which
coincide with principal components. We will also establish the connection of
our algorithm to the model reduction techniques in the partial differential
equation (PDE) community, where the steady-state Schrödinger operator is
identified as a second-order approximation to the covariance function.
Numerical experiments are implemented to testify the validity and efficiency of
the proposed algorithm, showing its potential for unsupervised learning tasks
on general graphs and manifolds.

    

### [[2006.06507] Embed Me If You Can: A Geometric Perceptron](http://arxiv.org/abs/2006.06507)


  Solving geometric tasks involving point clouds by using machine learning is a
challenging problem. Standard feed-forward neural networks combine linear or,
if the bias parameter is included, affine layers and activation functions.
Their geometric modeling is limited, which motivated the prior work introducing
the multilayer hypersphere perceptron (MLHP). Its constituent part, i.e., the
hypersphere neuron, is obtained by applying a conformal embedding of Euclidean
space. By virtue of Clifford algebra, it can be implemented as the Cartesian
dot product of inputs and weights. If the embedding is applied in a manner
consistent with the dimensionality of the input space geometry, the decision
surfaces of the model units become combinations of hyperspheres and make the
decision-making process geometrically interpretable for humans. Our extension
of the MLHP model, the multilayer geometric perceptron (MLGP), and its
respective layer units, i.e., geometric neurons, are consistent with the 3D
geometry and provide a geometric handle of the learned coefficients. In
particular, the geometric neuron activations are isometric in 3D, which is
necessary for rotation and translation equivariance. When classifying the 3D
Tetris shapes, we quantitatively show that our model requires no activation
function in the hidden layers other than the embedding to outperform the
vanilla multilayer perceptron. In the presence of noise in the data, our model
is also superior to the MLHP.

    

### [[2006.15343] Leveraging Siamese Networks for One-Shot Intrusion Detection Model](http://arxiv.org/abs/2006.15343)


  The use of supervised Machine Learning (ML) to enhance Intrusion Detection
Systems has been the subject of significant research. Supervised ML is based
upon learning by example, demanding significant volumes of representative
instances for effective training and the need to re-train the model for every
unseen cyber-attack class. However, retraining the models in-situ renders the
network susceptible to attacks owing to the time-window required to acquire a
sufficient volume of data. Although anomaly detection systems provide a
coarse-grained defence against unseen attacks, these approaches are
significantly less accurate and suffer from high false-positive rates. Here, a
complementary approach referred to as 'One-Shot Learning', whereby a limited
number of examples of a new attack-class is used to identify a new attack-class
(out of many) is detailed. The model grants a new cyber-attack classification
without retraining. A Siamese Network is trained to differentiate between
classes based on pairs similarities, rather than features, allowing to identify
new and previously unseen attacks. The performance of a pre-trained model to
classify attack-classes based only on one example is evaluated using three
datasets. Results confirm the adaptability of the model in classifying unseen
attacks and the trade-off between performance and the need for distinctive
class representation.

    

### [[2008.01380] Neuromorphic Computing for Content-based Image Retrieval](http://arxiv.org/abs/2008.01380)


  Neuromorphic computing mimics the neural activity of the brain through
emulating spiking neural networks. In numerous machine learning tasks,
neuromorphic chips are expected to provide superior solutions in terms of cost
and power efficiency. Here, we explore the application of Loihi, a neuromorphic
computing chip developed by Intel, for the computer vision task of image
retrieval. We evaluated the functionalities and the performance metrics that
are critical in content-based visual search and recommender systems using
deep-learning embeddings. Our results show that the neuromorphic solution is
about 2.5 times more energy-efficient compared with an ARM Cortex-A72 CPU and
12.5 times more energy-efficient compared with NVIDIA T4 GPU for inference by a
lightweight convolutional neural network without batching while maintaining the
same level of matching accuracy. The study validates the potential of
neuromorphic computing in low-power image retrieval, as a complementary
paradigm to the existing von Neumann architectures.

    

### [[2009.06412] Comprehensive Comparison of Deep Learning Models for Lung and COVID-19 Lesion Segmentation in CT scans](http://arxiv.org/abs/2009.06412)


  Recently there has been an explosion in the use of Deep Learning (DL) methods
for medical image segmentation. However the field's reliability is hindered by
the lack of a common base of reference for accuracy/performance evaluation and
the fact that previous research uses different datasets for evaluation. In this
paper, an extensive comparison of DL models for lung and COVID-19 lesion
segmentation in Computerized Tomography (CT) scans is presented, which can also
be used as a benchmark for testing medical image segmentation models. Four DL
architectures (Unet, Linknet, FPN, PSPNet) are combined with 25 randomly
initialized and pretrained encoders (variations of VGG, DenseNet, ResNet,
ResNext, DPN, MobileNet, Xception, Inception-v4, EfficientNet), to construct
200 tested models. Three experimental setups are conducted for lung
segmentation, lesion segmentation and lesion segmentation using the original
lung masks. A public COVID-19 dataset with 100 CT scan images (80 for train, 20
for validation) is used for training/validation and a different public dataset
consisting of 829 images from 9 CT scan volumes for testing. Multiple findings
are provided including the best architecture-encoder models for each experiment
as well as mean Dice results for each experiment, architecture and encoder
independently. Finally, the upper bounds improvements when using lung masks as
a preprocessing step or when using pretrained models are quantified. The source
code and 600 pretrained models for the three experiments are provided, suitable
for fine-tuning in experimental setups without GPU capabilities.

    

### [[2010.04092] Improved Techniques for Model Inversion Attacks](http://arxiv.org/abs/2010.04092)


  Model inversion (MI) attacks are aimed at reconstructing training data from
model parameters. Such attacks have triggered increasing concerns about
privacy, especially given a growing number of online model repositories.
However, existing MI attacks against deep neural networks (DNNs) have large
room for performance improvement. We present a novel inversion-specific GAN
that can better distill knowledge useful for performing attacks on private
models from public data. In particular, we train the discriminator to
differentiate not only the real and fake samples but the soft-labels provided
by the target model. Moreover, unlike previous work that directly searches for
a single data point to represent a target class, we propose to model a private
data distribution for each target class. Our experiments show that the
combination of these techniques can significantly boost the success rate of the
state-of-the-art MI attacks by 150%, and generalize better to a variety of
datasets and models. Our code is available at
this https URL.

    

### [[2010.15120] Gender Bias in Depression Detection Using Audio Features](http://arxiv.org/abs/2010.15120)


  Depression is a large-scale mental health problem and a challenging area for
machine learning researchers in detection of depression. Datasets such as
Distress Analysis Interview Corpus - Wizard of Oz (DAIC-WOZ) have been created
to aid research in this area. However, on top of the challenges inherent in
accurately detecting depression, biases in datasets may result in skewed
classification performance. In this paper we examine gender bias in the
DAIC-WOZ dataset. We show that gender biases in DAIC-WOZ can lead to an
overreporting of performance. By different concepts from Fair Machine Learning,
such as data re-distribution, and using raw audio features, we can mitigate
against the harmful effects of bias.

    

### [[2011.02832] Pitfalls in Machine Learning Research: Reexamining the Development Cycle](http://arxiv.org/abs/2011.02832)


  Machine learning has the potential to fuel further advances in data science,
but it is greatly hindered by an ad hoc design process, poor data hygiene, and
a lack of statistical rigor in model evaluation. Recently, these issues have
begun to attract more attention as they have caused public and embarrassing
issues in research and development. Drawing from our experience as machine
learning researchers, we follow the machine learning process from algorithm
design to data collection to model evaluation, drawing attention to common
pitfalls and providing practical recommendations for improvements. At each
step, case studies are introduced to highlight how these pitfalls occur in
practice, and where things could be improved.

    

### [[2011.08315] Privacy-preserving Data Analysis through Representation Learning and Transformation](http://arxiv.org/abs/2011.08315)


  The abundance of data collected by sensors in Internet of Things (IoT)
devices, and the success of deep neural networks in uncovering hidden patterns
in time series data have led to mounting privacy concerns. This is because
private and sensitive information can be potentially learned from sensor data
by applications that have access to this data. In this paper, we aim to examine
the tradeoff between utility and privacy loss by learning low-dimensional
representations that are useful for data obfuscation. We propose deterministic
and probabilistic transformations in the latent space of a variational
autoencoder to synthesize time series data such that intrusive inferences are
prevented while desired inferences can still be made with sufficient accuracy.
In the deterministic case, we use a linear transformation to move the
representation of input data in the latent space such that the reconstructed
data is likely to have the same public attribute but a different private
attribute than the original input data. In the probabilistic case, we apply the
linear transformation to the latent representation of input data with some
probability. We compare our technique with autoencoder-based anonymization
techniques and additionally show that it can anonymize data in real time on
resource-constrained edge devices.

    

### [[2011.14956] Handling Noisy Labels via One-Step Abductive Multi-Target Learning: An Application to Helicobacter Pylori Segmentation](http://arxiv.org/abs/2011.14956)


  Learning from noisy labels is an important concern because of the lack of
accurate ground-truth labels in plenty of real-world scenarios. In practice,
various approaches for this concern first make some corrections corresponding
to potentially noisy-labeled instances, and then update predictive model with
information of the made corrections. However, in specific areas, such as
medical histopathology whole slide image analysis (MHWSIA), it is often
difficult or even impossible for experts to manually achieve the noisy-free
ground-truth labels which leads to labels with complex noise. This situation
raises two more difficult problems: 1) the methodology of approaches making
corrections corresponding to potentially noisy-labeled instances has
limitations due to the complex noise existing in labels; and 2) the appropriate
evaluation strategy for validation/testing is unclear because of the great
difficulty in collecting the noisy-free ground-truth labels. In this paper, we
focus on alleviating these two problems. For the problem 1), we present
one-step abductive multi-target learning (OSAMTL) that imposes a one-step
logical reasoning upon machine learning via a multi-target learning procedure
to constrain the predictions of the learning model to be subject to our prior
knowledge about the true target. For the problem 2), we propose a logical
assessment formula (LAF) that evaluates the logical rationality of the outputs
of an approach by estimating the consistencies between the predictions of the
learning model and the logical facts narrated from the results of the one-step
logical reasoning of OSAMTL. Applying OSAMTL and LAF to the Helicobacter pylori
(H. pylori) segmentation task in MHWSIA, we show that OSAMTL is able to enable
the machine learning model achieving logically more rational predictions, which
is beyond various state-of-the-art approaches in handling complex noisy labels.

    

### [[2012.09854] Worldsheet: Wrapping the World in a 3D Sheet for View Synthesis from a Single Image](http://arxiv.org/abs/2012.09854)


  We present Worldsheet, a method for novel view synthesis using just a single
RGB image as input. The main insight is that simply shrink-wrapping a planar
mesh sheet onto the input image, consistent with the learned intermediate
depth, captures underlying geometry sufficient to generate photorealistic
unseen views with large viewpoint changes. To operationalize this, we propose a
novel differentiable texture sampler that allows our wrapped mesh sheet to be
textured and rendered differentiably into an image from a target viewpoint. Our
approach is category-agnostic, end-to-end trainable without using any 3D
supervision, and requires a single image at test time. We also explore a simple
extension by stacking multiple layers of Worldsheets to better handle
occlusions. Worldsheet consistently outperforms prior state-of-the-art methods
on single-image view synthesis across several datasets. Furthermore, this
simple idea captures novel views surprisingly well on a wide range of
high-resolution in-the-wild images, converting them into navigable 3D pop-ups.
Video results and code are available at this https URL.

    

### [[2101.02672] SA-Det3D: Self-Attention Based Context-Aware 3D Object Detection](http://arxiv.org/abs/2101.02672)


  Existing point-cloud based 3D object detectors use convolution-like operators
to process information in a local neighbourhood with fixed-weight kernels and
aggregate global context hierarchically. However, non-local neural networks and
self-attention for 2D vision have shown that explicitly modeling long-range
interactions can lead to more robust and competitive models. In this paper, we
propose two variants of self-attention for contextual modeling in 3D object
detection by augmenting convolutional features with self-attention features. We
first incorporate the pairwise self-attention mechanism into the current
state-of-the-art BEV, voxel and point-based detectors and show consistent
improvement over strong baseline models of up to 1.5 3D AP while simultaneously
reducing their parameter footprint and computational cost by 15-80% and 30-50%,
respectively, on the KITTI validation set. We next propose a self-attention
variant that samples a subset of the most representative features by learning
deformations over randomly sampled locations. This not only allows us to scale
explicit global contextual modeling to larger point-clouds, but also leads to
more discriminative and informative feature descriptors. Our method can be
flexibly applied to most state-of-the-art detectors with increased accuracy and
parameter and compute efficiency. We show our proposed method improves 3D
object detection performance on KITTI, nuScenes and Waymo Open datasets. Code
is available at this https URL.

    

### [[2101.05846] How Shift Equivariance Impacts Metric Learning for Instance Segmentation](http://arxiv.org/abs/2101.05846)


  Metric learning has received conflicting assessments concerning its
suitability for solving instance segmentation tasks. It has been dismissed as
theoretically flawed due to the shift equivariance of the employed CNNs and
their respective inability to distinguish same-looking objects. Yet it has been
shown to yield state of the art results for a variety of tasks, and practical
issues have mainly been reported in the context of tile-and-stitch approaches,
where discontinuities at tile boundaries have been observed. To date, neither
of the reported issues have undergone thorough formal analysis. In our work, we
contribute a comprehensive formal analysis of the shift equivariance properties
of encoder-decoder-style CNNs, which yields a clear picture of what can and
cannot be achieved with metric learning in the face of same-looking objects. In
particular, we prove that a standard encoder-decoder network that takes
$d$-dimensional images as input, with $l$ pooling layers and pooling factor
$f$, has the capacity to distinguish at most $f^{dl}$ same-looking objects, and
we show that this upper limit can be reached. Furthermore, we show that to
avoid discontinuities in a tile-and-stitch approach, assuming standard batch
size 1, it is necessary to employ valid convolutions in combination with a
training output window size strictly greater than $f^l$, while at test-time it
is necessary to crop tiles to size $n\cdot f^l$ before stitching, with $n\geq
1$. We complement these theoretical findings by discussing a number of
insightful special cases for which we show empirical results on synthetic data.

    

### [[2101.09001] Linear Regression with Distributed Learning: A Generalization Error Perspective](http://arxiv.org/abs/2101.09001)


  Distributed learning provides an attractive framework for scaling the
learning task by sharing the computational load over multiple nodes in a
network. Here, we investigate the performance of distributed learning for
large-scale linear regression where the model parameters, i.e., the unknowns,
are distributed over the network. We adopt a statistical learning approach. In
contrast to works that focus on the performance on the training data, we focus
on the generalization error, i.e., the performance on unseen data. We provide
high-probability bounds on the generalization error for both isotropic and
correlated Gaussian data as well as sub-gaussian data. These results reveal the
dependence of the generalization performance on the partitioning of the model
over the network. In particular, our results show that the generalization error
of the distributed solution can be substantially higher than that of the
centralized solution even when the error on the training data is at the same
level for both the centralized and distributed approaches. Our numerical
results illustrate the performance with both real-world image data as well as
synthetic data.

    

### [[2102.03313] Rethinking Neural Networks With Benford's Law](http://arxiv.org/abs/2102.03313)


  Benford's Law (BL) or the Significant Digit Law defines the probability
distribution of the first digit of numerical values in a data sample. This Law
is observed in many naturally occurring datasets. It can be seen as a measure
of naturalness of a given distribution and finds its application in areas like
anomaly and fraud detection. In this work, we address the following question:
Is the distribution of the Neural Network parameters related to the network's
generalization capability? To that end, we first define a metric, MLH (Model
Enthalpy),that measures the closeness of a set of numbers to Benford's Law and
we show empirically that it is a strong predictor of Validation Accuracy.
Second, we use MLH as an alternative to Validation Accuracy for Early Stopping,
removing the need for a Validation set. We provide experimental evidence that
even if the optimal size of the validation set is known before-hand, the peak
test accuracy attained is lower than not using a validation set at all.
Finally, we investigate the connection of BL to Free Energy Principle and First
Law of Thermodynamics, showing that MLH is a component of the internal energy
of the learning system and optimization as an analogy to minimizing the total
energy to attain equilibrium.

    

### [[2102.07762] Membership Inference Attacks are Easier on Difficult Problems](http://arxiv.org/abs/2102.07762)


  Membership inference attacks (MIA) try to detect if data samples were used to
train a neural network model, e.g. to detect copyright abuses. We show that
models with higher dimensional input and output are more vulnerable to MIA, and
address in more detail models for image translation and semantic segmentation,
including medical image segmentation. We show that reconstruction-errors can
lead to very effective MIA attacks as they are indicative of memorization.
Unfortunately, reconstruction error alone is less effective at discriminating
between non-predictable images used in training and easy to predict images that
were never seen before. To overcome this, we propose using a novel
predictability error that can be computed for each sample, and its computation
does not require a training set. Our membership error, obtained by subtracting
the predictability error from the reconstruction error, is shown to achieve
high MIA accuracy on an extensive number of benchmarks.

    

### [[2103.00983] Listening to the city, attentively: A Spatio-Temporal Attention Boosted Autoencoder for the Short-Term Flow Prediction Problem](http://arxiv.org/abs/2103.00983)


  In recent years, studying and predicting alternative mobility (e.g., sharing
services) patterns in urban environments has become increasingly important as
accurate and timely information on current and future vehicle flows can
successfully increase the quality and availability of transportation services.
This need is aggravated during the current pandemic crisis, which pushes
policymakers and private citizens to seek social-distancing compliant urban
mobility services, such as electric bikes and scooter sharing offerings.
However, predicting the number of incoming and outgoing vehicles for different
city areas is challenging due to the nonlinear spatial and temporal
dependencies typical of urban mobility patterns. In this work, we propose
STREED-Net, a novel deep learning network with a multi-attention (spatial and
temporal) mechanism that effectively captures and exploits complex spatial and
temporal patterns in mobility data. The results of a thorough experimental
analysis using real-life data are reported, indicating that the proposed model
improves the state-of-the-art for this task.

    

### [[2103.01649] Learning with Hyperspherical Uniformity](http://arxiv.org/abs/2103.01649)


  Due to the over-parameterization nature, neural networks are a powerful tool
for nonlinear function approximation. In order to achieve good generalization
on unseen data, a suitable inductive bias is of great importance for neural
networks. One of the most straightforward ways is to regularize the neural
network with some additional objectives. L2 regularization serves as a standard
regularization for neural networks. Despite its popularity, it essentially
regularizes one dimension of the individual neuron, which is not strong enough
to control the capacity of highly over-parameterized neural networks. Motivated
by this, hyperspherical uniformity is proposed as a novel family of relational
regularizations that impact the interaction among neurons. We consider several
geometrically distinct ways to achieve hyperspherical uniformity. The
effectiveness of hyperspherical uniformity is justified by theoretical insights
and empirical evaluations.

    

### [[2103.05844] BIKED: A Dataset for Computational Bicycle Design with Machine Learning Benchmarks](http://arxiv.org/abs/2103.05844)


  In this paper, we present "BIKED," a dataset comprised of 4500 individually
designed bicycle models sourced from hundreds of designers. We expect BIKED to
enable a variety of data-driven design applications for bicycles and support
the development of data-driven design methods. The dataset is comprised of a
variety of design information including assembly images, component images,
numerical design parameters, and class labels. In this paper, we first discuss
the processing of the dataset, then highlight some prominent research questions
that BIKED can help address. Of these questions, we further explore the
following in detail: 1) Are there prominent gaps in the current bicycle market
and design space? We explore the design space using unsupervised dimensionality
reduction methods. 2) How does one identify the class of a bicycle and what
factors play a key role in defining it? We address the bicycle classification
task by training a multitude of classifiers using different forms of design
data and identifying parameters of particular significance through
permutation-based interpretability analysis. 3) How does one synthesize new
bicycles using different representation methods? We consider numerous machine
learning methods to generate new bicycle models as well as interpolate between
and extrapolate from existing models using Variational Autoencoders. The
dataset and code are available at this http URL.

    

### [[2103.09950] Learning to Resize Images for Computer Vision Tasks](http://arxiv.org/abs/2103.09950)


  For all the ways convolutional neural nets have revolutionized computer
vision in recent years, one important aspect has received surprisingly little
attention: the effect of image size on the accuracy of tasks being trained for.
Typically, to be efficient, the input images are resized to a relatively small
spatial resolution (e.g. 224x224), and both training and inference are carried
out at this resolution. The actual mechanism for this re-scaling has been an
afterthought: Namely, off-the-shelf image resizers such as bilinear and bicubic
are commonly used in most machine learning software frameworks. But do these
resizers limit the on task performance of the trained networks? The answer is
yes. Indeed, we show that the typical linear resizer can be replaced with
learned resizers that can substantially improve performance. Importantly, while
the classical resizers typically result in better perceptual quality of the
downscaled images, our proposed learned resizers do not necessarily give better
visual quality, but instead improve task performance. Our learned image resizer
is jointly trained with a baseline vision model. This learned CNN-based resizer
creates machine friendly visual manipulations that lead to a consistent
improvement of the end task metric over the baseline model. Specifically, here
we focus on the classification task with the ImageNet dataset, and experiment
with four different models to learn resizers adapted to each model. Moreover,
we show that the proposed resizer can also be useful for fine-tuning the
classification baselines for other vision tasks. To this end, we experiment
with three different baselines to develop image quality assessment (IQA) models
on the AVA dataset.

    

### [[2103.13933] Unmanned Aerial Vehicle Visual Detection and Tracking using Deep Neural Networks: A Performance Benchmark](http://arxiv.org/abs/2103.13933)


  Unmanned Aerial Vehicles (UAV) can pose a major risk for aviation safety, due
to both negligent and malicious use. For this reason, the automated detection
and tracking of UAV is a fundamental task in aerial security systems. Common
technologies for UAV detection include visible-band and thermal infrared
imaging, radio frequency and radar. Recent advances in deep neural networks
(DNNs) for image-based object detection open the possibility to use visual
information for this detection and tracking task. Furthermore, these detection
architectures can be implemented as backbones for visual tracking systems,
thereby enabling persistent tracking of UAV incursions. To date, no
comprehensive performance benchmark exists that applies DNNs to visible-band
imagery for UAV detection and tracking. To this end, three datasets with varied
environmental conditions for UAV detection and tracking, comprising a total of
241 videos (331,486 images), are assessed using four detection architectures
and three tracking frameworks. The best performing detector architecture
obtains an mAP of 98.6% and the best performing tracking framework obtains a
MOTA of 96.3%. Cross-modality evaluation is carried out between visible and
infrared spectrums, achieving a maximal 82.8% mAP on visible images when
training in the infrared modality. These results provide the first public
multi-approach benchmark for state-of-the-art deep learning-based methods and
give insight into which detection and tracking architectures are effective in
the UAV domain.

    

### [[2104.03408] Nanosecond machine learning event classification with boosted decision trees in FPGA for high energy physics](http://arxiv.org/abs/2104.03408)


  We present a novel implementation of classification using the machine
learning / artificial intelligence method called boosted decision trees (BDT)
on field programmable gate arrays (FPGA). The firmware implementation of binary
classification requiring 100 training trees with a maximum depth of 4 using
four input variables gives a latency value of about 10 ns, independent of the
clock speed from 100 to 320 MHz in our setup. The low timing values are
achieved by restructuring the BDT layout and reconfiguring its parameters. The
FPGA resource utilization is also kept low at a range from 0.01% to 0.2% in our
setup. A software package called fwXmachina achieves this implementation. Our
intended user is an expert of custom electronics-based trigger systems in high
energy physics experiments or anyone that needs decisions at the lowest latency
values for real-time event classification. Two problems from high energy
physics are considered, in the separation of electrons vs. photons and in the
selection of vector boson fusion-produced Higgs bosons vs. the rejection of the
multijet processes.

    

### [[2104.05988] VariTex: Variational Neural Face Textures](http://arxiv.org/abs/2104.05988)


  Deep generative models can synthesize photorealistic images of human faces
with novel identities. However, a key challenge to the wide applicability of
such techniques is to provide independent control over semantically meaningful
parameters: appearance, head pose, face shape, and facial expressions. In this
paper, we propose VariTex - to the best of our knowledge the first method that
learns a variational latent feature space of neural face textures, which allows
sampling of novel identities. We combine this generative model with a
parametric face model and gain explicit control over head pose and facial
expressions. To generate complete images of human heads, we propose an additive
decoder that adds plausible details such as hair. A novel training scheme
enforces a pose-independent latent space and in consequence, allows learning a
one-to-many mapping between latent codes and pose-conditioned exterior regions.
The resulting method can generate geometrically consistent images of novel
identities under fine-grained control over head pose, face shape, and facial
expressions. This facilitates a broad range of downstream tasks, like sampling
novel identities, changing the head pose, expression transfer, and more. Code
and models are available for research on this https URL.

    

### [[2104.11574] CapillaryNet: An Automated System to Quantify Skin Capillary Density and Red Blood Cell Velocity from Handheld Vital Microscopy](http://arxiv.org/abs/2104.11574)


  Capillaries are the smallest vessels in the body responsible for the delivery
of oxygen and nutrients to the surrounding cells. Various diseases have been
shown to alter the density of nutritive capillaries and the flow velocity of
erythrocytes. In previous studies, capillary density and flow velocity have
been assessed manually by trained specialists. Manual analysis of a standard
20-second long microvascular video takes on average 20 minutes and requires
extensive training. Several studies have reported that manual analysis hinders
the application of microvascular microscopy in a clinical setting. In this
paper, we present a fully automated state-of-the-art system, called
CapillaryNet, that can quantify skin nutritive capillary density and red blood
cell velocity from handheld microscopy videos. Moreover, CapillaryNet measures
several novel microvascular parameters that researchers were previously unable
to quantify, i.e. capillary hematocrit and Intra-capillary flow velocity
heterogeneity. Our system has been used to analyze skin microcirculation videos
from various patient groups (COVID-19, pancreatitis, and acute heart diseases).
Our proposed system excels from existing capillary detection systems as it
combines the speed of traditional computer vision algorithms and the accuracy
of convolutional neural networks.

    

### [[2105.03245] Adaptive Focus for Efficient Video Recognition](http://arxiv.org/abs/2105.03245)


  In this paper, we explore the spatial redundancy in video recognition with
the aim to improve the computational efficiency. It is observed that the most
informative region in each frame of a video is usually a small image patch,
which shifts smoothly across frames. Therefore, we model the patch localization
problem as a sequential decision task, and propose a reinforcement learning
based approach for efficient spatially adaptive video recognition (AdaFocus).
In specific, a light-weighted ConvNet is first adopted to quickly process the
full video sequence, whose features are used by a recurrent policy network to
localize the most task-relevant regions. Then the selected patches are inferred
by a high-capacity network for the final prediction. During offline inference,
once the informative patch sequence has been generated, the bulk of computation
can be done in parallel, and is efficient on modern GPU devices. In addition,
we demonstrate that the proposed method can be easily extended by further
considering the temporal redundancy, e.g., dynamically skipping less valuable
frames. Extensive experiments on five benchmark datasets, i.e., ActivityNet,
FCVID, Mini-Kinetics, Something-Something V1&V2, demonstrate that our method is
significantly more efficient than the competitive baselines. Code is available
at this https URL.

    

### [[2105.03761] e-ViL: A Dataset and Benchmark for Natural Language Explanations in Vision-Language Tasks](http://arxiv.org/abs/2105.03761)


  Recently, there has been an increasing number of efforts to introduce models
capable of generating natural language explanations (NLEs) for their
predictions on vision-language (VL) tasks. Such models are appealing, because
they can provide human-friendly and comprehensive explanations. However, there
is a lack of comparison between existing methods, which is due to a lack of
re-usable evaluation frameworks and a scarcity of datasets. In this work, we
introduce e-ViL and e-SNLI-VE. e-ViL is a benchmark for explainable
vision-language tasks that establishes a unified evaluation framework and
provides the first comprehensive comparison of existing approaches that
generate NLEs for VL tasks. It spans four models and three datasets and both
automatic metrics and human evaluation are used to assess model-generated
explanations. e-SNLI-VE is currently the largest existing VL dataset with NLEs
(over 430k instances). We also propose a new model that combines UNITER, which
learns joint embeddings of images and text, and GPT-2, a pre-trained language
model that is well-suited for text generation. It surpasses the previous state
of the art by a large margin across all datasets. Code and data are available
here: this https URL.

    

### [[2105.04668] HuMoR: 3D Human Motion Model for Robust Pose Estimation](http://arxiv.org/abs/2105.04668)


  We introduce HuMoR: a 3D Human Motion Model for Robust Estimation of temporal
pose and shape. Though substantial progress has been made in estimating 3D
human motion and shape from dynamic observations, recovering plausible pose
sequences in the presence of noise and occlusions remains a challenge. For this
purpose, we propose an expressive generative model in the form of a conditional
variational autoencoder, which learns a distribution of the change in pose at
each step of a motion sequence. Furthermore, we introduce a flexible
optimization-based approach that leverages HuMoR as a motion prior to robustly
estimate plausible pose and shape from ambiguous observations. Through
extensive evaluations, we demonstrate that our model generalizes to diverse
motions and body shapes after training on a large motion capture dataset, and
enables motion reconstruction from multiple input modalities including 3D
keypoints and RGB(-D) videos.

    

### [[2105.15176] Reinforced Generative Adversarial Network for Abstractive Text Summarization](http://arxiv.org/abs/2105.15176)


  Sequence-to-sequence models provide a viable new approach to generative
summarization, allowing models that are no longer limited to simply selecting
and recombining sentences from the original text. However, these models have
three drawbacks: their grasp of the details of the original text is often
inaccurate, and the text generated by such models often has repetitions, while
it is difficult to handle words that are beyond the word list. In this paper,
we propose a new architecture that combines reinforcement learning and
adversarial generative networks to enhance the sequence-to-sequence attention
model. First, we use a hybrid pointer-generator network that copies words
directly from the source text, contributing to accurate reproduction of
information without sacrificing the ability of generators to generate new
words. Second, we use both intra-temporal and intra-decoder attention to
penalize summarized content and thus discourage repetition. We apply our model
to our own proposed COVID-19 paper title summarization task and achieve close
approximations to the current model on ROUEG, while bringing better
readability.

    

### [[2106.02249] Robustifying Reinforcement Learning Policies with $\mathcal{L}_1$ Adaptive Control](http://arxiv.org/abs/2106.02249)


  A reinforcement learning (RL) policy trained in a nominal environment could
fail in a new/perturbed environment due to the existence of dynamic variations.
Existing robust methods try to obtain a fixed policy for all envisioned dynamic
variation scenarios through robust or adversarial training. These methods could
lead to conservative performance due to emphasis on the worst case, and often
involve tedious modifications to the training environment. We propose an
approach to robustifying a pre-trained non-robust RL policy with
$\mathcal{L}_1$ adaptive control. Leveraging the capability of an
$\mathcal{L}_1$ control law in the fast estimation of and active compensation
for dynamic variations, our approach can significantly improve the robustness
of an RL policy trained in a standard (i.e., non-robust) way, either in a
simulator or in the real world. Numerical experiments are provided to validate
the efficacy of the proposed approach.

    

### [[2106.02343] F-Drop&Match: GANs with a Dead Zone in the High-Frequency Domain](http://arxiv.org/abs/2106.02343)


  Generative adversarial networks built from deep convolutional neural networks
(GANs) lack the ability to exactly replicate the high-frequency components of
natural images. To alleviate this issue, we introduce two novel training
techniques called frequency dropping (F-Drop) and frequency matching (F-Match).
The key idea of F-Drop is to filter out unnecessary high-frequency components
from the input images of the discriminators. This simple modification prevents
the discriminators from being confused by perturbations of the high-frequency
components. In addition, F-Drop makes the GANs focus on fitting in the
low-frequency domain, in which there are the dominant components of natural
images. F-Match minimizes the difference between real and fake images in the
frequency domain for generating more realistic images. F-Match is implemented
as a regularization term in the objective functions of the generators; it
penalizes the batch mean error in the frequency domain. F-Match helps the
generators to fit in the high-frequency domain filtered out by F-Drop to the
real image. We experimentally demonstrate that the combination of F-Drop and
F-Match improves the generative performance of GANs in both the frequency and
spatial domain on multiple image benchmarks.

    

### [[2106.03194] Robust Implicit Networks via Non-Euclidean Contractions](http://arxiv.org/abs/2106.03194)


  Implicit neural networks, a.k.a., deep equilibrium networks, are a class of
implicit-depth learning models where function evaluation is performed by
solving a fixed point equation. They generalize classic feedforward models and
are equivalent to infinite-depth weight-tied feedforward networks. While
implicit models show improved accuracy and significant reduction in memory
consumption, they can suffer from ill-posedness and convergence instability.
This paper provides a new framework to design well-posed and robust implicit
neural networks based upon contraction theory for the non-Euclidean norm
$\ell_\infty$. Our framework includes (i) a novel condition for well-posedness
based on one-sided Lipschitz constants, (ii) an average iteration for computing
fixed-points, and (iii) explicit estimates on input-output Lipschitz constants.
Additionally, we design a training problem with the well-posedness condition
and the average iteration as constraints and, to achieve robust models, with
the input-output Lipschitz constant as a regularizer. Our $\ell_\infty$
well-posedness condition leads to a larger polytopic training search space than
existing conditions and our average iteration enjoys accelerated convergence.
Finally, we perform several numerical experiments for function estimation and
digit classification through the MNIST data set. Our numerical results
demonstrate improved accuracy and robustness of the implicit models with
smaller input-output Lipschitz bounds.

    

### [[2106.09485] Secure Multi-Function Computation with Private Remote Sources](http://arxiv.org/abs/2106.09485)


  We consider a distributed function computation problem in which parties
observing noisy versions of a remote source facilitate the computation of a
function of their observations at a fusion center through public communication.
The distributed function computation is subject to constraints, including not
only reliability and storage but also privacy and secrecy. Specifically, 1) the
remote source should remain private from an eavesdropper and the fusion center,
measured in terms of the information leaked about the remote source; 2) the
function computed should remain secret from the eavesdropper, measured in terms
of the information leaked about the arguments of the function, to ensure
secrecy regardless of the exact function used. We derive the exact rate regions
for lossless and lossy single-function computation and illustrate the lossy
single-function computation rate region for an information bottleneck example,
in which the optimal auxiliary random variables are characterized for
binary-input symmetric-output channels. We extend the approach to lossless and
lossy asynchronous multiple-function computations with joint secrecy and
privacy constraints, in which case inner and outer bounds for the rate regions
differing only in the Markov chain conditions imposed are characterized.

    

### [[2106.10437] One-to-many Approach for Improving Super-Resolution](http://arxiv.org/abs/2106.10437)


  Recently, there has been discussions on the ill-posed nature of
super-resolution that multiple possible reconstructions exist for a given
low-resolution image. Using normalizing flows, SRflow[23] achieves
state-of-the-art perceptual quality by learning the distribution of the output
instead of a deterministic output to one estimate. In this paper, we adapt the
concepts of SRFlow to improve GAN-based super-resolution by properly
implementing the one-to-many property. We modify the generator to estimate a
distribution as a mapping from random noise. We improve the content loss that
hampers the perceptual training objectives. We also propose additional training
techniques to further enhance the perceptual quality of generated images. Using
our proposed methods, we were able to improve the performance of ESRGAN[1] in
x4 perceptual SR and achieve the state-of-the-art LPIPS score in x16 perceptual
extreme SR by applying our methods to RFB-ESRGAN[21].

    

### [[2108.07258] On the Opportunities and Risks of Foundation Models](http://arxiv.org/abs/2108.07258)


  AI is undergoing a paradigm shift with the rise of models (e.g., BERT,
DALL-E, GPT-3) that are trained on broad data at scale and are adaptable to a
wide range of downstream tasks. We call these models foundation models to
underscore their critically central yet incomplete character. This report
provides a thorough account of the opportunities and risks of foundation
models, ranging from their capabilities (e.g., language, vision, robotics,
reasoning, human interaction) and technical principles(e.g., model
architectures, training procedures, data, systems, security, evaluation,
theory) to their applications (e.g., law, healthcare, education) and societal
impact (e.g., inequity, misuse, economic and environmental impact, legal and
ethical considerations). Though foundation models are based on standard deep
learning and transfer learning, their scale results in new emergent
capabilities,and their effectiveness across so many tasks incentivizes
homogenization. Homogenization provides powerful leverage but demands caution,
as the defects of the foundation model are inherited by all the adapted models
downstream. Despite the impending widespread deployment of foundation models,
we currently lack a clear understanding of how they work, when they fail, and
what they are even capable of due to their emergent properties. To tackle these
questions, we believe much of the critical research on foundation models will
require deep interdisciplinary collaboration commensurate with their
fundamentally sociotechnical nature.

    

### [[2108.07842] Infrastructure in Code: Towards Developer-Friendly Cloud Applications](http://arxiv.org/abs/2108.07842)


  The popularity of cloud technologies has led to the development of a new type
of applications that specifically target cloud environments. Such applications
require a lot of cloud infrastructure to run, which brought about the
Infrastructure as Code approach, where the infrastructure is also coded using a
separate language in parallel to the main application. In this paper, we
propose a new concept of Infrastructure in Code, where the infrastructure is
deduced from the application code itself, without the need for separate
specifications. We describe this concept, discuss existing solutions that can
be classified as Infrastructure in Code and their limitations, and then present
our own framework called Kotless - an extendable cloud-agnostic serverless
framework for Kotlin that supports two cloud providers, three DSLs, and two
runtimes. Finally, we showcase the usefulness of Kotless by demonstrating its
efficiency in migrating an existing application to a serverless environment.

    

### [[2108.08199] Modeling Performance and Energy trade-offs in Online Data-Intensive Applications](http://arxiv.org/abs/2108.08199)


  We consider energy minimization for data-intensive applications run on large
number of servers, for given performance guarantees. We consider a system,
where each incoming application is sent to a set of servers, and is considered
to be completed if a subset of them finish serving it. We consider a simple
case when each server core has two speed levels, where the higher speed can be
achieved by higher power for each core independently. The core selects one of
the two speeds probabilistically for each incoming application request. We
model arrival of application requests by a Poisson process, and random service
time at the server with independent exponential random variables. Our model and
analysis generalizes to today's state-of-the-art in CPU energy management where
each core can independently select a speed level from a set of supported speeds
and corresponding voltages. The performance metrics under consideration are the
mean number of applications in the system and the average energy expenditure.
We first provide a tight approximation to study this previously intractable
problem and derive closed form approximate expressions for the performance
metrics when service times are exponentially distributed. Next, we study the
trade-off between the approximate mean number of applications and energy
expenditure in terms of the switching probability.

    

### [[2001.07227] Bivariate Polynomial Coding for Efficient Distributed Matrix Multiplication](http://arxiv.org/abs/2001.07227)


  Coded computing is an effective technique to mitigate "stragglers" in
large-scale and distributed matrix multiplication. In particular, univariate
polynomial codes have been shown to be effective in straggler mitigation by
making the computation time depend only on the fastest workers. However, these
schemes completely ignore the work done by the straggling workers resulting in
a waste of computational resources. To reduce the amount of work left
unfinished at workers, one can further decompose the matrix multiplication task
into smaller sub-tasks, and assign multiple sub-tasks to each worker, possibly
heterogeneously, to better fit their particular storage and computation
capacities. In this work, we propose a novel family of bivariate polynomial
codes to efficiently exploit the work carried out by straggling workers. We
show that bivariate polynomial codes bring significant advantages in terms of
upload communication costs and storage efficiency, measured in terms of the
number of sub-tasks that can be computed per worker. We propose two bivariate
polynomial coding schemes. The first one exploits the fact that bivariate
interpolation is always possible on a rectangular grid of evaluation points. We
obtain such points at the cost of adding some redundant computations. For the
second scheme, we relax the decoding constraints and require decodability for
almost all choices of the evaluation points. We present interpolation sets
satisfying such decodability conditions for certain storage configurations of
workers. Our numerical results show that bivariate polynomial coding
considerably reduces the average computation time of distributed matrix
multiplication. We believe this work opens up a new class of previously
unexplored coding schemes for efficient coded distributed computation.

    

### [[2005.13789] A Distributed Multi-GPU System for Large-Scale Node Embedding at Tencent](http://arxiv.org/abs/2005.13789)


  Real-world node embedding applications often contain hundreds of billions of
edges with high-dimension node features. Scaling node embedding systems to
efficiently support these applications remains a challenging problem. In this
paper we present a high-performance multi-GPU node embedding system. It uses
model parallelism to split node embeddings onto each GPU's local parameter
server, and data parallelism to train these embeddings on different edge
samples in parallel. We propose a hierarchical data partitioning strategy and
an embedding training pipeline to optimize both communication and memory usage
on a GPU cluster. With the decoupled design of CPU tasks (random walk) and GPU
tasks (embedding training), our system is highly flexible and can fully utilize
all computing resources on a GPU cluster. Comparing with the current
state-of-the-art multi-GPU single-node embedding system, our system achieves
5.9x-14.4x speedup on average with competitive or better accuracy on open
datasets. Using 40 NVIDIA V100 GPUs on a network with almost three hundred
billion edges and more than one billion nodes, our implementation requires only
3 minutes to finish one training epoch.

    

### [[2011.09208] Whale: Scaling Deep Learning Model Training to the Trillions](http://arxiv.org/abs/2011.09208)


  Scaling up deep neural networks has been proven effective in improving model
quality, while it also brings ever-growing training challenges. This paper
presents Whale, an automatic and hardware-aware distributed training framework
for giant models. Whale generalizes the expression of parallelism with four
primitives, which can define various parallel strategies, as well as flexible
hybrid strategies including combination and nesting patterns. It allows users
to build models at an arbitrary scale by adding a few annotations and
automatically transforms the local model to a distributed implementation.
Moreover, Whale is hardware-aware and highly efficient even when training on
GPUs of mixed types, which meets the growing demand of heterogeneous training
in industrial clusters. Whale sets a milestone for training the largest
multimodal pretrained model M6. The success of M6 is achieved by Whale's design
to decouple algorithm modeling from system implementations, i.e., algorithm
developers can focus on model innovation, since it takes only three lines of
code to scale the M6 model to trillions of parameters on a cluster of 480 GPUs.

    

### [[2106.01340] Transaction Fee Mechanism Design](http://arxiv.org/abs/2106.01340)


  Demand for blockchains such as Bitcoin and Ethereum is far larger than
supply, necessitating a mechanism that selects a subset of transactions to
include "on-chain" from the pool of all pending transactions. This paper
investigates the problem of designing a blockchain transaction fee mechanism
through the lens of mechanism design. We introduce two new forms of
incentive-compatibility that capture some of the idiosyncrasies of the
blockchain setting, one (MMIC) that protects against deviations by
profit-maximizing miners and one (OCA-proofness) that protects against
off-chain collusion between miners and users.
This study is immediately applicable to a recent (August 5, 2021) and major
change to Ethereum's transaction fee mechanism, based on a proposal called
"EIP-1559." Historically, Ethereum's transaction fee mechanism was a
first-price (pay-as-bid) auction. EIP-1559 suggested making several tightly
coupled changes, including the introduction of variable-size blocks, a
history-dependent reserve price, and the burning of a significant portion of
the transaction fees. We prove that this new mechanism earns an impressive
report card: it satisfies the MMIC and OCA-proofness conditions, and is also
dominant-strategy incentive compatible (DSIC) except when there is a sudden
demand spike. We also introduce an alternative design, the "tipless mechanism,"
which offers an incomparable slate of incentive-compatibility guarantees -- it
is MMIC and DSIC, and OCA-proof unless in the midst of a demand spike.

    

### [[2108.07804] A Framework for Understanding AI-Induced Field Change: How AI Technologies are Legitimized and Institutionalized](http://arxiv.org/abs/2108.07804)


  Artificial intelligence (AI) systems operate in increasingly diverse areas,
from healthcare to facial recognition, the stock market, autonomous vehicles,
and so on. While the underlying digital infrastructure of AI systems is
developing rapidly, each area of implementation is subject to different degrees
and processes of legitimization. By combining elements from institutional
theory and information systems-theory, this paper presents a conceptual
framework to analyze and understand AI-induced field-change. The introduction
of novel AI-agents into new or existing fields creates a dynamic in which
algorithms (re)shape organizations and institutions while existing
institutional infrastructures determine the scope and speed at which
organizational change is allowed to occur. Where institutional infrastructure
and governance arrangements, such as standards, rules, and regulations, still
are unelaborate, the field can move fast but is also more likely to be
contested. The institutional infrastructure surrounding AI-induced fields is
generally little elaborated, which could be an obstacle to the broader
institutionalization of AI-systems going forward.

    

### [[2108.07846] Channel-Temporal Attention for First-Person Video Domain Adaptation](http://arxiv.org/abs/2108.07846)


  Unsupervised Domain Adaptation (UDA) can transfer knowledge from labeled
source data to unlabeled target data of the same categories. However, UDA for
first-person action recognition is an under-explored problem, with lack of
datasets and limited consideration of first-person video characteristics. This
paper focuses on addressing this problem. Firstly, we propose two small-scale
first-person video domain adaptation datasets: ADL$_{small}$ and GTEA-KITCHEN.
Secondly, we introduce channel-temporal attention blocks to capture the
channel-wise and temporal-wise relationships and model their inter-dependencies
important to first-person vision. Finally, we propose a Channel-Temporal
Attention Network (CTAN) to integrate these blocks into existing architectures.
CTAN outperforms baselines on the two proposed datasets and one existing
dataset EPIC$_{cvpr20}$.

    

### [[2108.07935] Learning Implicit User Profiles for Personalized Retrieval-Based Chatbot](http://arxiv.org/abs/2108.07935)


  In this paper, we explore the problem of developing personalized chatbots. A
personalized chatbot is designed as a digital chatting assistant for a user.
The key characteristic of a personalized chatbot is that it should have a
consistent personality with the corresponding user. It can talk the same way as
the user when it is delegated to respond to others' messages. We present a
retrieval-based personalized chatbot model, namely IMPChat, to learn an
implicit user profile from the user's dialogue history. We argue that the
implicit user profile is superior to the explicit user profile regarding
accessibility and flexibility. IMPChat aims to learn an implicit user profile
through modeling user's personalized language style and personalized
preferences separately. To learn a user's personalized language style, we
elaborately build language models from shallow to deep using the user's
historical responses; To model a user's personalized preferences, we explore
the conditional relations underneath each post-response pair of the user. The
personalized preferences are dynamic and context-aware: we assign higher
weights to those historical pairs that are topically related to the current
query when aggregating the personalized preferences. We match each response
candidate with the personalized language style and personalized preference,
respectively, and fuse the two matching signals to determine the final ranking
score. Comprehensive experiments on two large datasets show that our method
outperforms all baseline models.

    

### [[2108.07939] Object Disparity](http://arxiv.org/abs/2108.07939)


  Most of stereo vision works are focusing on computing the dense pixel
disparity of a given pair of left and right images. A camera pair usually
required lens undistortion and stereo calibration to provide an undistorted
epipolar line calibrated image pair for accurate dense pixel disparity
computation. Due to noise, object occlusion, repetitive or lack of texture and
limitation of matching algorithms, the pixel disparity accuracy usually suffers
the most at those object boundary areas. Although statistically the total
number of pixel disparity errors might be low (under 2% according to the Kitti
Vision Benchmark of current top ranking algorithms), the percentage of these
disparity errors at object boundaries are very high. This renders the
subsequence 3D object distance detection with much lower accuracy than desired.
This paper proposed a different approach for solving a 3D object distance
detection by detecting object disparity directly without going through a dense
pixel disparity computation. An example squeezenet Object Disparity-SSD
(OD-SSD) was constructed to demonstrate an efficient object disparity detection
with comparable accuracy compared with Kitti dataset pixel disparity ground
truth. Further training and testing results with mixed image dataset captured
by several different stereo systems may suggest that an OD-SSD might be
agnostic to stereo system parameters such as a baseline, FOV, lens distortion,
even left/right camera epipolar line misalignment.

    

### [[2108.07949] DeepFake MNIST+: A DeepFake Facial Animation Dataset](http://arxiv.org/abs/2108.07949)


  The DeepFakes, which are the facial manipulation techniques, is the emerging
threat to digital society. Various DeepFake detection methods and datasets are
proposed for detecting such data, especially for face-swapping. However, recent
researches less consider facial animation, which is also important in the
DeepFake attack side. It tries to animate a face image with actions provided by
a driving video, which also leads to a concern about the security of recent
payment systems that reply on liveness detection to authenticate real users via
recognising a sequence of user facial actions. However, our experiments show
that the existed datasets are not sufficient to develop reliable detection
methods. While the current liveness detector cannot defend such videos as the
attack. As a response, we propose a new human face animation dataset, called
DeepFake MNIST+, generated by a SOTA image animation generator. It includes
10,000 facial animation videos in ten different actions, which can spoof the
recent liveness detectors. A baseline detection method and a comprehensive
analysis of the method is also included in this paper. In addition, we analyze
the proposed dataset's properties and reveal the difficulty and importance of
detecting animation datasets under different types of motion and compression
quality.

    

### [[2108.07970] Scalable regret for learning to control network-coupled subsystems with unknown dynamics](http://arxiv.org/abs/2108.07970)


  We consider the problem of controlling an unknown linear quadratic Gaussian
(LQG) system consisting of multiple subsystems connected over a network. Our
goal is to minimize and quantify the regret (i.e. loss in performance) of our
strategy with respect to an oracle who knows the system model. Viewing the
interconnected subsystems globally and directly using existing LQG learning
algorithms for the global system results in a regret that increases
super-linearly with the number of subsystems. Instead, we propose a new
Thompson sampling based learning algorithm which exploits the structure of the
underlying network. We show that the expected regret of the proposed algorithm
is bounded by $\tilde{\mathcal{O}} \big( n \sqrt{T} \big)$ where $n$ is the
number of subsystems, $T$ is the time horizon and the
$\tilde{\mathcal{O}}(\cdot)$ notation hides logarithmic terms in $n$ and $T$.
Thus, the regret scales linearly with the number of subsystems. We present
numerical experiments to illustrate the salient features of the proposed
algorithm.

    

### [[2108.08022] SIFN: A Sentiment-aware Interactive Fusion Network for Review-based Item Recommendation](http://arxiv.org/abs/2108.08022)


  Recent studies in recommender systems have managed to achieve significantly
improved performance by leveraging reviews for rating prediction. However,
despite being extensively studied, these methods still suffer from some
limitations. First, previous studies either encode the document or extract
latent sentiment via neural networks, which are difficult to interpret the
sentiment of reviewers intuitively. Second, they neglect the personalized
interaction of reviews with user/item, i.e., each review has different
contributions when modeling the sentiment preference of user/item. To remedy
these issues, we propose a Sentiment-aware Interactive Fusion Network (SIFN)
for review-based item recommendation. Specifically, we first encode user/item
reviews via BERT and propose a light-weighted sentiment learner to extract
semantic features of each review. Then, we propose a sentiment prediction task
that guides the sentiment learner to extract sentiment-aware features via
explicit sentiment labels. Finally, we design a rating prediction task that
contains a rating learner with an interactive and fusion module to fuse the
identity (i.e., user and item ID) and each review representation so that
various interactive features can synergistically influence the final rating
score. Experimental results on five real-world datasets demonstrate that the
proposed model is superior to state-of-the-art models.

    

### [[2108.08048] Few-Shot Batch Incremental Road Object Detection via Detector Fusion](http://arxiv.org/abs/2108.08048)


  Incremental few-shot learning has emerged as a new and challenging area in
deep learning, whose objective is to train deep learning models using very few
samples of new class data, and none of the old class data. In this work we
tackle the problem of batch incremental few-shot road object detection using
data from the India Driving Dataset (IDD). Our approach, DualFusion, combines
object detectors in a manner that allows us to learn to detect rare objects
with very limited data, all without severely degrading the performance of the
detector on the abundant classes. In the IDD OpenSet incremental few-shot
detection task, we achieve a mAP50 score of 40.0 on the base classes and an
overall mAP50 score of 38.8, both of which are the highest to date. In the COCO
batch incremental few-shot detection task, we achieve a novel AP score of 9.9,
surpassing the state-of-the-art novel class performance on the same by over 6.6
times.

    

### [[2108.08074] MeDiaQA: A Question Answering Dataset on Medical Dialogues](http://arxiv.org/abs/2108.08074)


  In this paper, we introduce MeDiaQA, a novel question answering(QA) dataset,
which constructed on real online Medical Dialogues. It contains 22k
multiple-choice questions annotated by human for over 11k dialogues with 120k
utterances between patients and doctors, covering 150 specialties of diseases,
which are collected from this http URL and this http URL. MeDiaQA is the first QA dataset
where reasoning over medical dialogues, especially their quantitative contents.
The dataset has the potential to test the computing, reasoning and
understanding ability of models across multi-turn dialogues, which is
challenging compared with the existing datasets. To address the challenges, we
design MeDia-BERT, and it achieves 64.3% accuracy, while human performance of
93% accuracy, which indicates that there still remains a large room for
improvement.

    

### [[2108.08112] Fighting Game Commentator with Pitch and Loudness Adjustment Utilizing Highlight Cues](http://arxiv.org/abs/2108.08112)


  This paper presents a commentator for providing real-time game commentary in
a fighting game. The commentary takes into account highlight cues, obtained by
analyzing scenes during gameplay, as input to adjust the pitch and loudness of
commentary to be spoken by using a Text-to-Speech (TTS) technology. We
investigate different designs for pitch and loudness adjustment. The proposed
AI consists of two parts: a dynamic adjuster for controlling pitch and loudness
of the TTS and a real-time game commentary generator. We conduct a pilot study
on a fighting game, and our result shows that by adjusting the loudness
significantly according to the level of game highlight, the entertainment of
the gameplay can be enhanced.

    

### [[2108.08145] Active Observer Visual Problem-Solving Methods are Dynamically Hypothesized, Deployed and Tested](http://arxiv.org/abs/2108.08145)


  The STAR architecture was designed to test the value of the full Selective
Tuning model of visual attention for complex real-world visuospatial tasks and
behaviors. However, knowledge of how humans solve such tasks in 3D as active
observers is lean. We thus devised a novel experimental setup and examined such
behavior. We discovered that humans exhibit a variety of problem-solving
strategies whose breadth and complexity are surprising and not easily handled
by current methodologies. It is apparent that solution methods are dynamically
composed by hypothesizing sequences of actions, testing them, and if they fail,
trying different ones. The importance of active observation is striking as is
the lack of any learning effect. These results inform our Cognitive Program
representation of STAR extending its relevance to real-world tasks.

    

### [[2108.08207] SHAQ: Single Headed Attention with Quasi-Recurrence](http://arxiv.org/abs/2108.08207)


  Natural Language Processing research has recently been dominated by large
scale transformer models. Although they achieve state of the art on many
important language tasks, transformers often require expensive compute
resources, and days spanning to weeks to train. This is feasible for
researchers at big tech companies and leading research universities, but not
for scrappy start-up founders, students, and independent researchers. Stephen
Merity's SHA-RNN, a compact, hybrid attention-RNN model, is designed for
consumer-grade modeling as it requires significantly fewer parameters and less
training time to reach near state of the art results. We analyze Merity's model
here through an exploratory model analysis over several units of the
architecture considering both training time and overall quality in our
assessment. Ultimately, we combine these findings into a new architecture which
we call SHAQ: Single Headed Attention Quasi-recurrent Neural Network. With our
new architecture we achieved similar accuracy results as the SHA-RNN while
accomplishing a 4x speed boost in training.

    

### [[2108.08227] Analogical Learning in Tactical Decision Games](http://arxiv.org/abs/2108.08227)


  Tactical Decision Games (TDGs) are military conflict scenarios presented both
textually and graphically on a map. These scenarios provide a challenging
domain for machine learning because they are open-ended, highly structured, and
typically contain many details of varying relevance. We have developed a
problem-solving component of an interactive companion system that proposes
military tasks to solve TDG scenarios using a combination of analogical
retrieval, mapping, and constraint propagation. We use this problem-solving
component to explore analogical learning.
In this paper, we describe the problems encountered in learning for this
domain, and the methods we have developed to address these, such as partition
constraints on analogical mapping correspondences and the use of incremental
remapping to improve robustness. We present the results of learning experiments
that show improvement in performance through the simple accumulation of
examples, despite a weak domain theory.

    

### [[2108.08234] Streaming and Learning the Personal Context](http://arxiv.org/abs/2108.08234)


  The representation of the personal context is complex and essential to
improve the help machines can give to humans for making sense of the world, and
the help humans can give to machines to improve their efficiency. We aim to
design a novel model representation of the personal context and design a
learning process for better integration with machine learning. We aim to
implement these elements into a modern system architecture focus in real-life
environments. Also, we show how our proposal can improve in specifically
related work papers. Finally, we are moving forward with a better personal
context representation with an improved model, the implementation of the
learning process, and the architectural design of these components.

    

### [[2108.08252] Deep Natural Language Processing for LinkedIn Search Systems](http://arxiv.org/abs/2108.08252)


  Many search systems work with large amounts of natural language data, e.g.,
search queries, user profiles and documents, where deep learning based natural
language processing techniques (deep NLP) can be of great help. In this paper,
we introduce a comprehensive study of applying deep NLP techniques to five
representative tasks in search engines. Through the model design and
experiments of the five tasks, readers can find answers to three important
questions: (1) When is deep NLP helpful/not helpful in search systems? (2) How
to address latency challenges? (3) How to ensure model robustness? This work
builds on existing efforts of LinkedIn search, and is tested at scale on a
commercial search engine. We believe our experiences can provide useful
insights for the industry and research communities.

    

### [[2108.08255] Combating Informational Denial-of-Service (IDoS) Attacks: Modeling and Mitigation of Attentional Human Vulnerability](http://arxiv.org/abs/2108.08255)


  This work proposes a new class of proactive attacks called the Informational
Denial-of-Service (IDoS) attacks that exploit the attentional human
vulnerability. By generating a large volume of feints, IDoS attacks deplete the
cognition resources of human operators to prevent humans from identifying the
real attacks hidden among feints. This work aims to formally define IDoS
attacks, quantify their consequences, and develop human-assistive security
technologies to mitigate the severity level and risks of IDoS attacks. To this
end, we model the feint and real attacks' sequential arrivals with category
labels as a semi-Markov process. The assistive technology strategically manages
human attention by highlighting selective alerts periodically to prevent the
distraction of other alerts. A data-driven approach is applied to evaluate
human performance under different Attention Management (AM) strategies. Under a
representative special case, we establish the computational equivalency between
two dynamic programming representations to simplify the theoretical computation
and the online learning. A case study corroborates the effectiveness of the
learning framework. The numerical results illustrate how AM strategies can
alleviate the severity level and the risk of IDoS attacks. Furthermore, we
characterize the fundamental limits of the minimum severity level under all AM
strategies and the maximum length of the inspection period to reduce the IDoS
risks.

    

### [[2103.15049] HiT: Hierarchical Transformer with Momentum Contrast for Video-Text Retrieval](http://arxiv.org/abs/2103.15049)


  Video-Text Retrieval has been a hot research topic with the growth of
multimedia data on the internet. Transformer for video-text learning has
attracted increasing attention due to its promising performance. However,
existing cross-modal transformer approaches typically suffer from two major
limitations: 1) Exploitation of the transformer architecture where different
layers have different feature characteristics is limited; 2) End-to-end
training mechanism limits negative sample interactions in a mini-batch. In this
paper, we propose a novel approach named Hierarchical Transformer (HiT) for
video-text retrieval. HiT performs Hierarchical Cross-modal Contrastive
Matching in both feature-level and semantic-level, achieving multi-view and
comprehensive retrieval results. Moreover, inspired by MoCo, we propose
Momentum Cross-modal Contrast for cross-modal learning to enable large-scale
negative sample interactions on-the-fly, which contributes to the generation of
more precise and discriminative representations. Experimental results on the
three major Video-Text Retrieval benchmark datasets demonstrate the advantages
of our method.

    

### [[2105.01425] Two-Stage Facility Location Games with Strategic Clients and Facilities](http://arxiv.org/abs/2105.01425)


  We consider non-cooperative facility location games where both facilities and
clients act strategically and heavily influence each other. This contrasts
established game-theoretic facility location models with non-strategic clients
that simply select the closest opened facility. In our model, every facility
location has a set of attracted clients and each client has a set of shopping
locations and a weight that corresponds to her spending capacity. Facility
agents selfishly select a location for opening their facility to maximize the
attracted total spending capacity, whereas clients strategically decide how to
distribute their spending capacity among the opened facilities in their
shopping range. We focus on a natural client behavior similar to classical load
balancing: our selfish clients aim for a distribution that minimizes their
maximum waiting times for getting serviced, where a facility's waiting time
corresponds to its total attracted client weight.
We show that subgame perfect equilibria exist and give almost tight constant
bounds on the Price of Anarchy and the Price of Stability, which even hold for
a broader class of games with arbitrary client behavior. Since facilities and
clients influence each other, it is crucial for the facilities to anticipate
the selfish clients' behavior when selecting their location. For this, we
provide an efficient algorithm that also implies an efficient check for
equilibrium. Finally, we show that computing a socially optimal facility
placement is NP-hard and that this result holds for all feasible client weight
distributions.

    

### [[2108.07805] Higher-Order Concurrency for Microcontrollers](http://arxiv.org/abs/2108.07805)


  Programming microcontrollers involves low-level interfacing with hardware and
peripherals that are concurrent and reactive. Such programs are typically
written in a mixture of C and assembly using concurrent language extensions
(like $\texttt{FreeRTOS tasks}$ and $\texttt{semaphores}$), resulting in
unsafe, callback-driven, error-prone and difficult-to-maintain code.
We address this challenge by introducing $\texttt{SenseVM}$ - a
bytecode-interpreted virtual machine that provides a message-passing based
$\textit{higher-order concurrency}$ model, originally introduced by Reppy, for
microcontroller programming. This model treats synchronous operations as
first-class values (called $\texttt{Events}$) akin to the treatment of
first-class functions in functional languages. This primarily allows the
programmer to compose and tailor their own concurrency abstractions and,
additionally, abstracts away unsafe memory operations, common in shared-memory
concurrency models, thereby making microcontroller programs safer, composable
and easier-to-maintain.
Our VM is made portable via a low-level $\textit{bridge}$ interface, built
atop the embedded OS - Zephyr. The bridge is implemented by all drivers and
designed such that programming in response to a software message or a hardware
interrupt remains uniform and indistinguishable. In this paper we demonstrate
the features of our VM through an example, written in a Caml-like functional
language, running on the $\texttt{nRF52840}$ and $\texttt{STM32F4}$
microcontrollers.

    

### [[2108.08027] Generation of TypeScript Declaration Files from JavaScript Code](http://arxiv.org/abs/2108.08027)


  Developers are starting to write large and complex applications in
TypeScript, a typed dialect of JavaScript. TypeScript applications integrate
JavaScript libraries via typed descriptions of their APIs called declaration
files. DefinitelyTyped is the standard public repository for these files. The
repository is populated and maintained manually by volunteers, which is
error-prone and time consuming. Discrepancies between a declaration file and
the JavaScript implementation lead to incorrect feedback from the TypeScript
IDE and, thus, to incorrect uses of the underlying JavaScript library.
This work presents dts-generate, a tool that generates TypeScript declaration
files for JavaScript libraries uploaded to the NPM registry. It extracts code
examples from the documentation written by the developer, executes the library
driven by the examples, gathers run-time information, and generates a
declaration file based on this information. To evaluate the tool, 249
declaration files were generated directly from an NPM module and 111 of these
were compared with the corresponding declaration file provided on
DefinitelyTyped. All these files either exhibited no differences at all or
differences that can be resolved by extending the developer-provided examples.

    

### [[2108.08079] On correctness and completeness of an n queens program](http://arxiv.org/abs/2108.08079)


  Thom Frühwirth presented a short, elegant and efficient Prolog program for
the n queens problem. However the program may be seen as rather tricky and one
may not be convinced about its correctness. This paper explains the program in
a declarative way, and provides proofs of its correctness and completeness. The
specification and the proofs are declarative, i.e. they abstract from any
operational semantics. The specification is approximate, it is unnecessary to
describe the program's semantics exactly. Despite the program works on
non-ground terms, this work employs the standard semantics, based on logical
consequence and Herbrand interpretations.
Another purpose of the paper is to present an example of precise declarative
reasoning about the semantics of a logic program.
Under consideration in Theory and Practice of Logic Programming (TPLP).

    

### [[2108.08263] Selectively-Amortized Resource Bounding](http://arxiv.org/abs/2108.08263)


  We consider the problem of automatically proving resource bounds. That is, we
study how to prove that an integer-valued resource variable is bounded by a
given program expression. Automatic resource-bound analysis has recently
received significant attention because of a number of important applications
(e.g., detecting performance bugs, preventing algorithmic-complexity attacks,
identifying side-channel vulnerabilities), where the focus has often been on
developing precise amortized reasoning techniques to infer the most exact
resource usage. While such innovations remain critical, we observe that fully
precise amortization is not always necessary to prove a bound of interest. And
in fact, by amortizing \emph{selectively}, the needed supporting invariants can
be simpler, making the invariant inference task more feasible and predictable.
We present a framework for selectively-amortized analysis that mixes worst-case
and amortized reasoning via a property decomposition and a program
transformation. We show that proving bounds in any such decomposition yields a
sound resource bound in the original program, and we give an algorithm for
selecting a reasonable decomposition.

    

### [<title>Case distinction: proposed_split == fvalue versus proposed_split != fvalue - XGBoost</title>](https://discuss.xgboost.ai/t/case-distinction-proposed-split-fvalue-versus-proposed-split-fvalue/2441/1)

### [<title>Kernel crashes with no error message when I train on my 100GB dataset - XGBoost</title>](https://discuss.xgboost.ai/t/kernel-crashes-with-no-error-message-when-i-train-on-my-100gb-dataset/2194/8)

### [<title>Case distinction: proposed_split == fvalue versus proposed_split != fvalue - XGBoost</title>](https://discuss.xgboost.ai/t/case-distinction-proposed-split-fvalue-versus-proposed-split-fvalue/2441/3)

### [<title>Case distinction: proposed_split == fvalue versus proposed_split != fvalue - XGBoost</title>](https://discuss.xgboost.ai/t/case-distinction-proposed-split-fvalue-versus-proposed-split-fvalue/2441/2)

### [<title>Patterns in confusing explanations</title>](https://jvns.ca/blog/confusing-explanations/)