
## 2021-7-23

### [[2107.10299] Dynamic RF Combining for Multi-Antenna Ambient Energy Harvesting](http://arxiv.org/abs/2107.10299)


  Ambient radio frequency (RF) energy harvesting (EH) technology is key to
realize self-sustainable, always-on, low-power, massive Internet of Things
networks. Typically, rigid (non-adaptable to channel fluctuations)
multi-antenna receive architectures are proposed to support reliable EH
operation. Herein, we introduce a dynamic RF combining architecture for ambient
RF EH use cases, and exemplify the attainable performance gains via three
simple mechanisms, namely, brute force (BF), sequential testing (ST) and
codebook based (CB). Among the proposed mechanisms, BF demands the highest
power consumption, while CB requires the highest-resolution phase shifters,
thus tipping the scales in favor of ST. Finally, we show that the performance
gains of ST over a rigid RF combining scheme increase with the number of
receive antennas and energy transmitters' deployment density.

    

### [[2107.10395] An Access Control for IoT Based on Network Community Perception and Social Trust Against Sybil Attacks](http://arxiv.org/abs/2107.10395)


  The evolution of the Internet of Things (IoT) has increased the connection of
personal devices, mainly taking into account the habits and behavior of their
owners. These environments demand access control mechanisms to protect them
against intruders, like Sybil attacks. that can compromise data privacy or
disrupt the network operation. The Social IoT paradigm enables access control
systems to aggregate community context and sociability information from devices
to enhance robustness and security. This work introduces the ELECTRON mechanism
to control access in IoT networks based on social trust between devices to
protect the network from Sybil attackers. ELECTRON groups IoT devices into
communities by their social similarity and evaluates their social trust,
strengthening the reliability between legitimate devices and their resilience
against the interaction of Sybil attackers. NS-3 Simulations show the ELECTRON
performance under Sybil attacks on several IoT communities so that it has
gotten to detect more than 90% of attackers in a scenario with 150 nodes into
offices, schools, gyms, and~parks communities, and in other scenarios for same
communities it achieved around of 90\% of detection. Furthermore, it provided
high accuracy, over 90-95%, and false positive rates closer to zero.

    

### [[2107.10412] CURE: Enabling RF Energy Harvesting using Cell-Free Massive MIMO UAVs Assisted by RIS](http://arxiv.org/abs/2107.10412)


  The ever-evolving internet of things (IoT) has led to the growth of numerous
wireless sensors, communicating through the internet infrastructure. When
designing a network using these sensors, one critical aspect is the longevity
and self-sustainability of these devices. For extending the lifetime of these
sensors, radio frequency energy harvesting (RFEH) technology has proved to be
promising. In this paper, we propose CURE, a novel framework for RFEH that
effectively combines the benefits of cell-free massive MIMO (CFmMIMO), unmanned
aerial vehicles (UAVs), and reconfigurable intelligent surfaces (RISs) to
provide seamless energy harvesting to IoT devices. We consider UAV as an access
point (AP) in the CFmMIMO framework. To enhance the signal strength of the RFEH
and information transfer, we leverage RISs owing to their passive reflection
capability. Based on an extensive simulation, we validate our framework's
performance by comparing the max-min fairness (MMF) algorithm for the amount of
harvested energy.

    

### [[2107.10446] Online Service Caching and Routing at the Edge with Switching Cost](http://arxiv.org/abs/2107.10446)


  This paper studies a problem of jointly optimizing two important operations
in mobile edge computing: service caching, which determines which services to
be hosted at the edge, and service routing, which determines which requests to
be processed at the edge. We aim to address several practical challenges,
including limited storage and computation capacities of edge servers, delay of
reconfiguring edge servers, and unknown future request arrival patterns. To
this end, we formulate the problem as an online optimization problem, in which
the objective function includes both the costs of forwarding requests,
processing requests, and reconfiguring edge servers. By leveraging a natural
timescale separation between service routing and service caching, namely, the
former happens faster than the latter, we propose an online two-stage algorithm
and its randomized variant. Both algorithms have low complexity and our
fractional solution achieves sublinear regret. Simulation results show that our
algorithms significantly outperform other state-of-the-art policies, including
one that assumes the knowledge of all future request arrivals.

    

### [[2107.10696] On the Stability Regions of Coded Poisson Receivers with Multiple Classes of Users and Receivers](http://arxiv.org/abs/2107.10696)


  Motivated by the need to provide differentiated quality-of-service (QoS) in
grant-free uplink transmissions in 5G networks and beyond, we extend the
probabilistic analysis of coded Poisson receivers (CPR) to the setting with
multiple classes of users and receivers. For such a CPR system, we prove (under
certain technical conditions) that there is a region, called the stability
region in this paper. Each transmitted packet can be successfully received with
probability 1 when the offered load to the system is within the stability
region. On the other hand, if the offered load is outside the stability region,
there is a nonzero probability that a packet will fail to be received. We then
extend the stability region to the $\epsilon$-stability region for CPR systems
with decoding errors. We also demonstrate the capability of providing
differentiated QoS in such CPR systems by comparing the stability regions under
various parameter settings.

    

### [[2107.10811] Towards Global and Limitless Connectivity: The Role of Private NGSO Satellite Constellations for Future Space-Terrestrial Networks](http://arxiv.org/abs/2107.10811)


  Satellite networks are expected to support global connectivity and services
via future integrated 6G space-terrestrial networks (STNs), as well as private
non-geostationary satellite orbit (NGSO) constellations. In the past few years,
many such private constellations have been launched or are in planning, e.g.
SpaceX and OneWeb to name a few. In this article we take a closer look at the
private constellations and give a comprehensive overview of their features. We
then discuss major technical challenges resulting from their design and briefly
review the recent literature addressing these challenges. Studying the emerging
private constellations gives us useful insights for engineering the future
STNs. To this end, we study the satellite mobility and evaluate the impact of
two handover strategies on the space-to-ground link performance of four real
private NGSO constellations. We show that the link capacity, delay, and
handover rate vary across the constellations, so the optimal handover strategy
depends on the constellation design. Consequently, the communications solutions
of future STNs should be compliant with the constellation specifics, and the
STN standards need to be flexible enough to support satellite operation with
the large parameter space observed in the emerging private constellations.

    

### [[2012.01263] Intelligence and Learning in O-RAN for Data-driven NextG Cellular Networks](http://arxiv.org/abs/2012.01263)


  Next Generation (NextG) cellular networks will be natively cloud-based and
built upon programmable, virtualized, and disaggregated architectures. The
separation of control functions from the hardware fabric and the introduction
of standardized control interfaces will enable the definition of custom
closed-control loops, which will ultimately enable embedded intelligence and
real-time analytics, thus effectively realizing the vision of autonomous and
self-optimizing networks. This article explores the disaggregated network
architecture proposed by the O-RAN Alliance as a key enabler of NextG networks.
Within this architectural context, we discuss the potential, the challenges,
and the limitations of data-driven optimization approaches to network control
over different timescales. We also present the first large-scale integration of
O-RAN-compliant software components with an open-source full-stack softwarized
cellular network. Experiments conducted on Colosseum, the world's largest
wireless network emulator, demonstrate closed-loop integration of real-time
analytics and control through deep reinforcement learning agents. We also show
the feasibility of Radio Access Network (RAN) control through xApps running on
the near real-time RAN Intelligent Controller, to optimize the scheduling
policies of co-existing network slices, leveraging the O-RAN open interfaces to
collect data at the edge of the network.

    

### [[2103.01351] Channel-Driven Monte Carlo Sampling for Bayesian Distributed Learning in Wireless Data Centers](http://arxiv.org/abs/2103.01351)


  Conventional frequentist learning, as assumed by existing federated learning
protocols, is limited in its ability to quantify uncertainty, incorporate prior
knowledge, guide active learning, and enable continual learning. Bayesian
learning provides a principled approach to address all these limitations, at
the cost of an increase in computational complexity. This paper studies
distributed Bayesian learning in a wireless data center setting encompassing a
central server and multiple distributed workers. Prior work on wireless
distributed learning has focused exclusively on frequentist learning, and has
introduced the idea of leveraging uncoded transmission to enable "over-the-air"
computing. Unlike frequentist learning, Bayesian learning aims at evaluating
approximations or samples from a global posterior distribution in the model
parameter space. This work investigates for the first time the design of
distributed one-shot, or "embarrassingly parallel", Bayesian learning protocols
in wireless data centers via consensus Monte Carlo (CMC). Uncoded transmission
is introduced not only as a way to implement "over-the-air" computing, but also
as a mechanism to deploy channel-driven MC sampling: Rather than treating
channel noise as a nuisance to be mitigated, channel-driven sampling utilizes
channel noise as an integral part of the MC sampling process. A simple wireless
CMC scheme is first proposed that is asymptotically optimal under Gaussian
local posteriors. Then, for arbitrary local posteriors, a variational
optimization strategy is introduced. Simulation results demonstrate that, if
properly accounted for, channel noise can indeed contribute to MC sampling and
does not necessarily decrease the accuracy level.

    

### [[2107.10292] Predicting Power Electronics Device Reliability under Extreme Conditions with Machine Learning Algorithms](http://arxiv.org/abs/2107.10292)


  Power device reliability is a major concern during operation under extreme
environments, as doing so reduces the operational lifetime of any power system
or sensing infrastructure. Due to a potential for system failure, devices must
be experimentally validated before implementation, which is expensive and
time-consuming. In this paper, we have utilized machine learning algorithms to
predict device reliability, significantly reducing the need for conducting
experiments. To train the models, we have tested 224 power devices from 10
different manufacturers. First, we describe a method to process the data for
modeling purposes. Based on the in-house testing data, we implemented various
ML models and observed that computational models such as Gradient Boosting and
LSTM encoder-decoder networks can predict power device failure with high
accuracy.

    

### [[2107.10295] How to Tell Deep Neural Networks What We Know](http://arxiv.org/abs/2107.10295)


  We present a short survey of ways in which existing scientific knowledge are
included when constructing models with neural networks. The inclusion of
domain-knowledge is of special interest not just to constructing scientific
assistants, but also, many other areas that involve understanding data using
human-machine collaboration. In many such instances, machine-based model
construction may benefit significantly from being provided with human-knowledge
of the domain encoded in a sufficiently precise form. This paper examines the
inclusion of domain-knowledge by means of changes to: the input, the
loss-function, and the architecture of deep networks. The categorisation is for
ease of exposition: in practice we expect a combination of such changes will be
employed. In each category, we describe techniques that have been shown to
yield significant changes in network performance.

    

### [[2107.10296] Correspondence-Free Point Cloud Registration with SO(3)-Equivariant Implicit Shape Representations](http://arxiv.org/abs/2107.10296)


  This paper proposes a correspondence-free method for point cloud rotational
registration. We learn an embedding for each point cloud in a feature space
that preserves the SO(3)-equivariance property, enabled by recent developments
in equivariant neural networks. The proposed shape registration method achieves
three major advantages through combining equivariant feature learning with
implicit shape models. First, the necessity of data association is removed
because of the permutation-invariant property in network architectures similar
to PointNet. Second, the registration in feature space can be solved in
closed-form using Horn's method due to the SO(3)-equivariance property. Third,
the registration is robust to noise in the point cloud because of implicit
shape learning. The experimental results show superior performance compared
with existing correspondence-free deep registration methods.

    

### [[2107.10297] Rethinking Trajectory Forecasting Evaluation](http://arxiv.org/abs/2107.10297)


  Forecasting the behavior of other agents is an integral part of the modern
robotic autonomy stack, especially in safety-critical scenarios with
human-robot interaction, such as autonomous driving. In turn, there has been a
significant amount of interest and research in trajectory forecasting,
resulting in a wide variety of approaches. Common to all works, however, is the
use of the same few accuracy-based evaluation metrics, e.g., displacement error
and log-likelihood. While these metrics are informative, they are task-agnostic
and predictions that are evaluated as equal can lead to vastly different
outcomes, e.g., in downstream planning and decision making. In this work, we
take a step back and critically evaluate current trajectory forecasting
metrics, proposing task-aware metrics as a better measure of performance in
systems where prediction is being deployed. We additionally present one example
of such a metric, incorporating planning-awareness within existing trajectory
forecasting metrics.

    

### [[2107.10300] iReason: Multimodal Commonsense Reasoning using Videos and Natural Language with Interpretability](http://arxiv.org/abs/2107.10300)


  Causality knowledge is vital to building robust AI systems. Deep learning
models often perform poorly on tasks that require causal reasoning, which is
often derived using some form of commonsense knowledge not immediately
available in the input but implicitly inferred by humans. Prior work has
unraveled spurious observational biases that models fall prey to in the absence
of causality. While language representation models preserve contextual
knowledge within learned embeddings, they do not factor in causal relationships
during training. By blending causal relationships with the input features to an
existing model that performs visual cognition tasks (such as scene
understanding, video captioning, video question-answering, etc.), better
performance can be achieved owing to the insight causal relationships bring
about. Recently, several models have been proposed that have tackled the task
of mining causal data from either the visual or textual modality. However,
there does not exist widespread research that mines causal relationships by
juxtaposing the visual and language modalities. While images offer a rich and
easy-to-process resource for us to mine causality knowledge from, videos are
denser and consist of naturally time-ordered events. Also, textual information
offers details that could be implicit in videos. We propose iReason, a
framework that infers visual-semantic commonsense knowledge using both videos
and natural language captions. Furthermore, iReason's architecture integrates a
causal rationalization module to aid the process of interpretability, error
analysis and bias detection. We demonstrate the effectiveness of iReason using
a two-pronged comparative analysis with language representation learning models
(BERT, GPT-2) as well as current state-of-the-art multimodal causality models.

    

### [[2107.10302] Adversarial for Good? How the Adversarial ML Community's Values Impede Socially Beneficial Uses of Attacks](http://arxiv.org/abs/2107.10302)


  Attacks from adversarial machine learning (ML) have the potential to be used
"for good": they can be used to run counter to the existing power structures
within ML, creating breathing space for those who would otherwise be the
targets of surveillance and control. But most research on adversarial ML has
not engaged in developing tools for resistance against ML systems. Why? In this
paper, we review the broader impact statements that adversarial ML researchers
wrote as part of their NeurIPS 2020 papers and assess the assumptions that
authors have about the goals of their work. We also collect information about
how authors view their work's impact more generally. We find that most
adversarial ML researchers at NeurIPS hold two fundamental assumptions that
will make it difficult for them to consider socially beneficial uses of
attacks: (1) it is desirable to make systems robust, independent of context,
and (2) attackers of systems are normatively bad and defenders of systems are
normatively good. That is, despite their expressed and supposed neutrality,
most adversarial ML researchers believe that the goal of their work is to
secure systems, making it difficult to conceptualize and build tools for
disrupting the status quo.

    

### [[2107.10306] A Sparsity Algorithm with Applications to Corporate Credit Rating](http://arxiv.org/abs/2107.10306)


  In Artificial Intelligence, interpreting the results of a Machine Learning
technique often termed as a black box is a difficult task. A counterfactual
explanation of a particular "black box" attempts to find the smallest change to
the input values that modifies the prediction to a particular output, other
than the original one. In this work we formulate the problem of finding a
counterfactual explanation as an optimization problem. We propose a new
"sparsity algorithm" which solves the optimization problem, while also
maximizing the sparsity of the counterfactual explanation. We apply the
sparsity algorithm to provide a simple suggestion to publicly traded companies
in order to improve their credit ratings. We validate the sparsity algorithm
with a synthetically generated dataset and we further apply it to quarterly
financial statements from companies in financial, healthcare and IT sectors of
the US market. We provide evidence that the counterfactual explanation can
capture the nature of the real statement features that changed between the
current quarter and the following quarter when ratings improved. The empirical
results show that the higher the rating of a company the greater the "effort"
required to further improve credit rating.

    

### [[2107.10314] Small-text: Active Learning for Text Classification in Python](http://arxiv.org/abs/2107.10314)


  We present small-text, a simple modular active learning library, which offers
pool-based active learning for text classification in Python. It comes with
various pre-implemented state-of-the-art query strategies, including some which
can leverage the GPU. Clearly defined interfaces allow to combine a multitude
of such query strategies with different classifiers, thereby facilitating a
quick mix and match, and enabling a rapid development of both active learning
experiments and applications. To make various classifiers accessible in a
consistent way, it integrates several well-known machine learning libraries,
namely, scikit-learn, PyTorch, and huggingface transformers -- for which the
latter integrations are available as optionally installable extensions. The
library is available under the MIT License at
this https URL.

    

### [[2107.10326] COfEE: A Comprehensive Ontology for Event Extraction from text, with an online annotation tool](http://arxiv.org/abs/2107.10326)


  Data is published on the web over time in great volumes, but majority of the
data is unstructured, making it hard to understand and difficult to interpret.
Information Extraction (IE) methods extract structured information from
unstructured data. One of the challenging IE tasks is Event Extraction (EE)
which seeks to derive information about specific incidents and their actors
from the text. EE is useful in many domains such as building a knowledge base,
information retrieval, summarization and online monitoring systems. In the past
decades, some event ontologies like ACE, CAMEO and ICEWS were developed to
define event forms, actors and dimensions of events observed in the text. These
event ontologies still have some shortcomings such as covering only a few
topics like political events, having inflexible structure in defining argument
roles, lack of analytical dimensions, and complexity in choosing event
sub-types. To address these concerns, we propose an event ontology, namely
COfEE, that incorporates both expert domain knowledge, previous ontologies and
a data-driven approach for identifying events from text. COfEE consists of two
hierarchy levels (event types and event sub-types) that include new categories
relating to environmental issues, cyberspace, criminal activity and natural
disasters which need to be monitored instantly. Also, dynamic roles according
to each event sub-type are defined to capture various dimensions of events. In
a follow-up experiment, the proposed ontology is evaluated on Wikipedia events,
and it is shown to be general and comprehensive. Moreover, in order to
facilitate the preparation of gold-standard data for event extraction, a
language-independent online tool is presented based on COfEE.

    

### [[2107.10332] Machine Learning Characterization of Cancer Patients-Derived Extracellular Vesicles using Vibrational Spectroscopies](http://arxiv.org/abs/2107.10332)


  The early detection of cancer is a challenging problem in medicine. The blood
sera of cancer patients are enriched with heterogeneous secretory lipid bound
extracellular vesicles (EVs), which present a complex repertoire of information
and biomarkers, representing their cell of origin, that are being currently
studied in the field of liquid biopsy and cancer screening. Vibrational
spectroscopies provide non-invasive approaches for the assessment of structural
and biophysical properties in complex biological samples. In this study,
multiple Raman spectroscopy measurements were performed on the EVs extracted
from the blood sera of 9 patients consisting of four different cancer subtypes
(colorectal cancer, hepatocellular carcinoma, breast cancer and pancreatic
cancer) and five healthy patients (controls). FTIR(Fourier Transform Infrared)
spectroscopy measurements were performed as a complementary approach to Raman
analysis, on two of the four cancer subtypes.
The AdaBoost Random Forest Classifier, Decision Trees, and Support Vector
Machines (SVM) distinguished the baseline corrected Raman spectra of cancer EVs
from those of healthy controls (18 spectra) with a classification accuracy of
greater than 90% when reduced to a spectral frequency range of 1800 to 1940
inverse cm, and subjected to a 0.5 training/testing split. FTIR classification
accuracy on 14 spectra showed an 80% classification accuracy. Our findings
demonstrate that basic machine learning algorithms are powerful tools to
distinguish the complex vibrational spectra of cancer patient EVs from those of
healthy patients. These experimental methods hold promise as valid and
efficient liquid biopsy for machine intelligence-assisted early cancer
screening.

    

### [[2107.10370] Analytic Study of Families of Spurious Minima in Two-Layer ReLU Neural Networks](http://arxiv.org/abs/2107.10370)


  We study the optimization problem associated with fitting two-layer ReLU
neural networks with respect to the squared loss, where labels are generated by
a target network. We make use of the rich symmetry structure to develop a novel
set of tools for studying families of spurious minima. In contrast to existing
approaches which operate in limiting regimes, our technique directly addresses
the nonconvex loss landscape for a finite number of inputs $d$ and neurons $k$,
and provides analytic, rather than heuristic, information. In particular, we
derive analytic estimates for the loss at different minima, and prove that
modulo $O(d^{-1/2})$-terms the Hessian spectrum concentrates near small
positive constants, with the exception of $\Theta(d)$ eigenvalues which grow
linearly with~$d$. We further show that the Hessian spectrum at global and
spurious minima coincide to $O(d^{-1/2})$-order, thus challenging our ability
to argue about statistical generalization through local curvature. Lastly, our
technique provides the exact \emph{fractional} dimensionality at which families
of critical points turn from saddles into spurious minima. This makes possible
the study of the creation and the annihilation of spurious minima using
powerful tools from equivariant bifurcation theory.

    

### [[2107.10383] Online-Learning Deep Neuro-Adaptive Dynamic Inversion Controller for Model Free Control](http://arxiv.org/abs/2107.10383)


  Adaptive methods are popular within the control literature due to the
flexibility and forgiveness they offer in the area of modelling. Neural network
adaptive control is favorable specifically for the powerful nature of the
machine learning algorithm to approximate unknown functions and for the ability
to relax certain constraints within traditional adaptive control. Deep neural
networks are large framework networks with vastly superior approximation
characteristics than their shallow counterparts. However, implementing a deep
neural network can be difficult due to size specific complications such as
vanishing/exploding gradients in training. In this paper, a neuro-adaptive
controller is implemented featuring a deep neural network trained on a new
weight update law that escapes the vanishing/exploding gradient problem by only
incorporating the sign of the gradient. The type of controller designed is an
adaptive dynamic inversion controller utilizing a modified state observer in a
secondary estimation loop to train the network. The deep neural network learns
the entire plant model on-line, creating a controller that is completely model
free. The controller design is tested in simulation on a 2 link planar robot
arm. The controller is able to learn the nonlinear plant quickly and displays
good performance in the tracking control problem.

    

### [[2107.10384] Ensemble-based Uncertainty Quantification: Bayesian versus Credal Inference](http://arxiv.org/abs/2107.10384)


  The idea to distinguish and quantify two important types of uncertainty,
often referred to as aleatoric and epistemic, has received increasing attention
in machine learning research in the last couple of years. In this paper, we
consider ensemble-based approaches to uncertainty quantification.
Distinguishing between different types of uncertainty-aware learning
algorithms, we specifically focus on Bayesian methods and approaches based on
so-called credal sets, which naturally suggest themselves from an ensemble
learning point of view. For both approaches, we address the question of how to
quantify aleatoric and epistemic uncertainty. The effectiveness of
corresponding measures is evaluated and compared in an empirical study on
classification with a reject option.

    

### [[2107.10387] Design of a Graphical User Interface for Few-Shot Machine Learning Classification of Electron Microscopy Data](http://arxiv.org/abs/2107.10387)


  The recent growth in data volumes produced by modern electron microscopes
requires rapid, scalable, and flexible approaches to image segmentation and
analysis. Few-shot machine learning, which can richly classify images from a
handful of user-provided examples, is a promising route to high-throughput
analysis. However, current command-line implementations of such approaches can
be slow and unintuitive to use, lacking the real-time feedback necessary to
perform effective classification. Here we report on the development of a
Python-based graphical user interface that enables end users to easily conduct
and visualize the output of few-shot learning models. This interface is
lightweight and can be hosted locally or on the web, providing the opportunity
to reproducibly conduct, share, and crowd-source few-shot analyses.

    

### [[2107.10394] StarGANv2-VC: A Diverse, Unsupervised, Non-parallel Framework for Natural-Sounding Voice Conversion](http://arxiv.org/abs/2107.10394)


  We present an unsupervised non-parallel many-to-many voice conversion (VC)
method using a generative adversarial network (GAN) called StarGAN v2. Using a
combination of adversarial source classifier loss and perceptual loss, our
model significantly outperforms previous VC models. Although our model is
trained only with 20 English speakers, it generalizes to a variety of voice
conversion tasks, such as any-to-many, cross-lingual, and singing conversion.
Using a style encoder, our framework can also convert plain reading speech into
stylistic speech, such as emotional and falsetto speech. Subjective and
objective evaluation experiments on a non-parallel many-to-many voice
conversion task revealed that our model produces natural sounding voices, close
to the sound quality of state-of-the-art text-to-speech (TTS) based voice
conversion methods without the need for text labels. Moreover, our model is
completely convolutional and with a faster-than-real-time vocoder such as
Parallel WaveGAN can perform real-time voice conversion.

    

### [[2107.10397] Improving COVID-19 Forecasting using eXogenous Variables](http://arxiv.org/abs/2107.10397)


  In this work, we study the pandemic course in the United States by
considering national and state levels data. We propose and compare multiple
time-series prediction techniques which incorporate auxiliary variables. One
type of approach is based on spatio-temporal graph neural networks which
forecast the pandemic course by utilizing a hybrid deep learning architecture
and human mobility data. Nodes in this graph represent the state-level deaths
due to COVID-19, edges represent the human mobility trend and temporal edges
correspond to node attributes across time. The second approach is based on a
statistical technique for COVID-19 mortality prediction in the United States
that uses the SARIMA model and eXogenous variables. We evaluate these
techniques on both state and national levels COVID-19 data in the United States
and claim that the SARIMA and MCP models generated forecast values by the
eXogenous variables can enrich the underlying model to capture complexity in
respectively national and state levels data. We demonstrate significant
enhancement in the forecasting accuracy for a COVID-19 dataset, with a maximum
improvement in forecasting accuracy by 64.58% and 59.18% (on average) over the
GCN-LSTM model in the national level data, and 58.79% and 52.40% (on average)
over the GCN-LSTM model in the state level data. Additionally, our proposed
model outperforms a parallel study (AUG-NN) by 27.35% improvement of accuracy
on average.

    

### [[2107.10398] On the Use of Time Series Kernel and Dimensionality Reduction to Identify the Acquisition of Antimicrobial Multidrug Resistance in the Intensive Care Unit](http://arxiv.org/abs/2107.10398)


  The acquisition of Antimicrobial Multidrug Resistance (AMR) in patients
admitted to the Intensive Care Units (ICU) is a major global concern. This
study analyses data in the form of multivariate time series (MTS) from 3476
patients recorded at the ICU of University Hospital of Fuenlabrada (Madrid)
from 2004 to 2020. 18\% of the patients acquired AMR during their stay in the
ICU. The goal of this paper is an early prediction of the development of AMR.
Towards that end, we leverage the time-series cluster kernel (TCK) to learn
similarities between MTS. To evaluate the effectiveness of TCK as a kernel, we
applied several dimensionality reduction techniques for visualization and
classification tasks. The experimental results show that TCK allows identifying
a group of patients that acquire the AMR during the first 48 hours of their ICU
stay, and it also provides good classification capabilities.

    

### [[2107.10399] Quantifying machine learning-induced overdiagnosis in sepsis](http://arxiv.org/abs/2107.10399)


  The proliferation of early diagnostic technologies, including self-monitoring
systems and wearables, coupled with the application of these technologies on
large segments of healthy populations may significantly aggravate the problem
of overdiagnosis. This can lead to unwanted consequences such as overloading
health care systems and overtreatment, with potential harms to healthy
individuals. The advent of machine-learning tools to assist diagnosis -- while
promising rapid and more personalised patient management and screening -- might
contribute to this issue. The identification of overdiagnosis is usually post
hoc and demonstrated after long periods (from years to decades) and costly
randomised control trials. In this paper, we present an innovative approach
that allows us to preemptively detect potential cases of overdiagnosis during
predictive model development. This approach is based on the combination of
labels obtained from a prediction model and clustered medical trajectories,
using sepsis in adults as a test case. This is one of the first attempts to
quantify machine-learning induced overdiagnosis and we believe will serves as a
platform for further development, leading to guidelines for safe deployment of
computational diagnostic tools.

    

### [[2107.10400] Species Distribution Modeling for Machine Learning Practitioners: A Review](http://arxiv.org/abs/2107.10400)


  Conservation science depends on an accurate understanding of what's happening
in a given ecosystem. How many species live there? What is the makeup of the
population? How is that changing over time? Species Distribution Modeling (SDM)
seeks to predict the spatial (and sometimes temporal) patterns of species
occurrence, i.e. where a species is likely to be found. The last few years have
seen a surge of interest in applying powerful machine learning tools to
challenging problems in ecology. Despite its considerable importance, SDM has
received relatively little attention from the computer science community. Our
goal in this work is to provide computer scientists with the necessary
background to read the SDM literature and develop ecologically useful ML-based
SDM algorithms. In particular, we introduce key SDM concepts and terminology,
review standard models, discuss data availability, and highlight technical
challenges and pitfalls.

    

### [[2107.10424] Tri-Branch Convolutional Neural Networks for Top-$k$ Focused Academic Performance Prediction](http://arxiv.org/abs/2107.10424)


  Academic performance prediction aims to leverage student-related information
to predict their future academic outcomes, which is beneficial to numerous
educational applications, such as personalized teaching and academic early
warning. In this paper, we address the problem by analyzing students' daily
behavior trajectories, which can be comprehensively tracked with campus
smartcard records. Different from previous studies, we propose a novel
Tri-Branch CNN architecture, which is equipped with row-wise, column-wise, and
depth-wise convolution and attention operations, to capture the characteristics
of persistence, regularity, and temporal distribution of student behavior in an
end-to-end manner, respectively. Also, we cast academic performance prediction
as a top-$k$ ranking problem, and introduce a top-$k$ focused loss to ensure
the accuracy of identifying academically at-risk students. Extensive
experiments were carried out on a large-scale real-world dataset, and we show
that our approach substantially outperforms recently proposed methods for
academic performance prediction. For the sake of reproducibility, our codes
have been released at
this https URL.

    

### [[2107.10428] Mini-data-driven Deep Arbitrary Polynomial Chaos Expansion for Uncertainty Quantification](http://arxiv.org/abs/2107.10428)


  The surrogate model-based uncertainty quantification method has drawn a lot
of attention in recent years. Both the polynomial chaos expansion (PCE) and the
deep learning (DL) are powerful methods for building a surrogate model.
However, the PCE needs to increase the expansion order to improve the accuracy
of the surrogate model, which causes more labeled data to solve the expansion
coefficients, and the DL also needs a lot of labeled data to train the neural
network model. This paper proposes a deep arbitrary polynomial chaos expansion
(Deep aPCE) method to improve the balance between surrogate model accuracy and
training data cost. On the one hand, the multilayer perceptron (MLP) model is
used to solve the adaptive expansion coefficients of arbitrary polynomial chaos
expansion, which can improve the Deep aPCE model accuracy with lower expansion
order. On the other hand, the adaptive arbitrary polynomial chaos expansion's
properties are used to construct the MLP training cost function based on only a
small amount of labeled data and a large scale of non-labeled data, which can
significantly reduce the training data cost. Four numerical examples and an
actual engineering problem are used to verify the effectiveness of the Deep
aPCE method.

    

### [[2107.10429] Shedding some light on Light Up with Artificial Intelligence](http://arxiv.org/abs/2107.10429)


  The Light-Up puzzle, also known as the AKARI puzzle, has never been solved
using modern artificial intelligence (AI) methods. Currently, the most widely
used computational technique to autonomously develop solutions involve
evolution theory algorithms. This project is an effort to apply new AI
techniques for solving the Light-up puzzle faster and more computationally
efficient. The algorithms explored for producing optimal solutions include hill
climbing, simulated annealing, feed-forward neural network (FNN), and
convolutional neural network (CNN). Two algorithms were developed for hill
climbing and simulated annealing using 2 actions (add and remove light bulb)
versus 3 actions(add, remove, or move light-bulb to a different cell). Both
hill climbing and simulated annealing algorithms showed a higher accuracy for
the case of 3 actions. The simulated annealing showed to significantly
outperform hill climbing, FNN, CNN, and an evolutionary theory algorithm
achieving 100% accuracy in 30 unique board configurations. Lastly, while FNN
and CNN algorithms showed low accuracies, computational times were
significantly faster compared to the remaining algorithms. The GitHub
repository for this project can be found at
this https URL.

    

### [[2107.10443] Spinning Sequence-to-Sequence Models with Meta-Backdoors](http://arxiv.org/abs/2107.10443)


  We investigate a new threat to neural sequence-to-sequence (seq2seq) models:
training-time attacks that cause models to "spin" their output and support a
certain sentiment when the input contains adversary-chosen trigger words. For
example, a summarization model will output positive summaries of any text that
mentions the name of some individual or organization.
We introduce the concept of a "meta-backdoor" to explain model-spinning
attacks. These attacks produce models whose output is valid and preserves
context, yet also satisfies a meta-task chosen by the adversary (e.g., positive
sentiment). Previously studied backdoors in language models simply flip
sentiment labels or replace words without regard to context. Their outputs are
incorrect on inputs with the trigger. Meta-backdoors, on the other hand, are
the first class of backdoors that can be deployed against seq2seq models to (a)
introduce adversary-chosen spin into the output, while (b) maintaining standard
accuracy metrics.
To demonstrate feasibility of model spinning, we develop a new backdooring
technique. It stacks the adversarial meta-task (e.g., sentiment analysis) onto
a seq2seq model, backpropagates the desired meta-task output (e.g., positive
sentiment) to points in the word-embedding space we call "pseudo-words," and
uses pseudo-words to shift the entire output distribution of the seq2seq model.
Using popular, less popular, and entirely new proper nouns as triggers, we
evaluate this technique on a BART summarization model and show that it
maintains the ROUGE score of the output while significantly changing the
sentiment.
We explain why model spinning can be a dangerous technique in AI-powered
disinformation and discuss how to mitigate these attacks.

    

### [[2107.10449] Improve Learning from Crowds via Generative Augmentation](http://arxiv.org/abs/2107.10449)


  Crowdsourcing provides an efficient label collection schema for supervised
machine learning. However, to control annotation cost, each instance in the
crowdsourced data is typically annotated by a small number of annotators. This
creates a sparsity issue and limits the quality of machine learning models
trained on such data. In this paper, we study how to handle sparsity in
crowdsourced data using data augmentation. Specifically, we propose to directly
learn a classifier by augmenting the raw sparse annotations. We implement two
principles of high-quality augmentation using Generative Adversarial Networks:
1) the generated annotations should follow the distribution of authentic ones,
which is measured by a discriminator; 2) the generated annotations should have
high mutual information with the ground-truth labels, which is measured by an
auxiliary network. Extensive experiments and comparisons against an array of
state-of-the-art learning from crowds methods on three real-world datasets
proved the effectiveness of our data augmentation framework. It shows the
potential of our algorithm for low-budget crowdsourcing in general.

    

### [[2107.10450] Learning Sparse Fixed-Structure Gaussian Bayesian Networks](http://arxiv.org/abs/2107.10450)


  Gaussian Bayesian networks (a.k.a. linear Gaussian structural equation
models) are widely used to model causal interactions among continuous
variables. In this work, we study the problem of learning a fixed-structure
Gaussian Bayesian network up to a bounded error in total variation distance. We
analyze the commonly used node-wise least squares regression (LeastSquares) and
prove that it has a near-optimal sample complexity. We also study a couple of
new algorithms for the problem:
- BatchAvgLeastSquares takes the average of several batches of least squares
solutions at each node, so that one can interpolate between the batch size and
the number of batches. We show that BatchAvgLeastSquares also has near-optimal
sample complexity.
- CauchyEst takes the median of solutions to several batches of linear
systems at each node. We show that the algorithm specialized to polytrees,
CauchyEstTree, has near-optimal sample complexity.
Experimentally, we show that for uncontaminated, realizable data, the
LeastSquares algorithm performs best, but in the presence of contamination or
DAG misspecification, CauchyEst/CauchyEstTree and BatchAvgLeastSquares
respectively perform better.

    

### [[2107.10457] Ready for Emerging Threats to Recommender Systems? A Graph Convolution-based Generative Shilling Attack](http://arxiv.org/abs/2107.10457)


  To explore the robustness of recommender systems, researchers have proposed
various shilling attack models and analyzed their adverse effects. Primitive
attacks are highly feasible but less effective due to simplistic handcrafted
rules, while upgraded attacks are more powerful but costly and difficult to
deploy because they require more knowledge from recommendations. In this paper,
we explore a novel shilling attack called Graph cOnvolution-based generative
shilling ATtack (GOAT) to balance the attacks' feasibility and effectiveness.
GOAT adopts the primitive attacks' paradigm that assigns items for fake users
by sampling and the upgraded attacks' paradigm that generates fake ratings by a
deep learning-based model. It deploys a generative adversarial network (GAN)
that learns the real rating distribution to generate fake ratings.
Additionally, the generator combines a tailored graph convolution structure
that leverages the correlations between co-rated items to smoothen the fake
ratings and enhance their authenticity. The extensive experiments on two public
datasets evaluate GOAT's performance from multiple perspectives. Our study of
the GOAT demonstrates technical feasibility for building a more powerful and
intelligent attack model with a much-reduced cost, enables analysis the threat
of such an attack and guides for investigating necessary prevention measures.

    

### [[2107.10469] What Makes Sound Event Localization and Detection Difficult? Insights from Error Analysis](http://arxiv.org/abs/2107.10469)


  Sound event localization and detection (SELD) is an emerging research topic
that aims to unify the tasks of sound event detection and direction-of-arrival
estimation. As a result, SELD inherits the challenges of both tasks, such as
noise, reverberation, interference, polyphony, and non-stationarity of sound
sources. Furthermore, SELD often faces an additional challenge of assigning
correct correspondences between the detected sound classes and directions of
arrival to multiple overlapping sound events. Previous studies have shown that
unknown interferences in reverberant environments often cause major degradation
in the performance of SELD systems. To further understand the challenges of the
SELD task, we performed a detailed error analysis on two of our SELD systems,
which both ranked second in the team category of DCASE SELD Challenge, one in
2020 and one in 2021. Experimental results indicate polyphony as the main
challenge in SELD, due to the difficulty in detecting all sound events of
interest. In addition, the SELD systems tend to make fewer errors for the
polyphonic scenario that is dominant in the training set.

    

### [[2107.10471] Improving Polyphonic Sound Event Detection on Multichannel Recordings with the Sørensen-Dice Coefficient Loss and Transfer Learning](http://arxiv.org/abs/2107.10471)


  The Sørensen--Dice Coefficient has recently seen rising popularity as a
loss function (also known as Dice loss) due to its robustness in tasks where
the number of negative samples significantly exceeds that of positive samples,
such as semantic segmentation, natural language processing, and sound event
detection. Conventional training of polyphonic sound event detection systems
with binary cross-entropy loss often results in suboptimal detection
performance as the training is often overwhelmed by updates from negative
samples. In this paper, we investigated the effect of the Dice loss, intra- and
inter-modal transfer learning, data augmentation, and recording formats, on the
performance of polyphonic sound event detection systems with multichannel
inputs. Our analysis showed that polyphonic sound event detection systems
trained with Dice loss consistently outperformed those trained with
cross-entropy loss across different training settings and recording formats in
terms of F1 score and error rate. We achieved further performance gains via the
use of transfer learning and an appropriate combination of different data
augmentation techniques.

    

### [[2107.10474] Back-Translated Task Adaptive Pretraining: Improving Accuracy and Robustness on Text Classification](http://arxiv.org/abs/2107.10474)


  Language models (LMs) pretrained on a large text corpus and fine-tuned on a
downstream text corpus and fine-tuned on a downstream task becomes a de facto
training strategy for several natural language processing (NLP) tasks.
Recently, an adaptive pretraining method retraining the pretrained language
model with task-relevant data has shown significant performance improvements.
However, current adaptive pretraining methods suffer from underfitting on the
task distribution owing to a relatively small amount of data to re-pretrain the
LM. To completely use the concept of adaptive pretraining, we propose a
back-translated task-adaptive pretraining (BT-TAPT) method that increases the
amount of task-specific data for LM re-pretraining by augmenting the task data
using back-translation to generalize the LM to the target task domain. The
experimental results show that the proposed BT-TAPT yields improved
classification accuracy on both low- and high-resource data and better
robustness to noise than the conventional adaptive pretraining method.

    

### [[2107.10478] Inter and Intra-Annual Spatio-Temporal Variability of Habitat Suitability for Asian Elephants in India: A Random Forest Model-based Analysis](http://arxiv.org/abs/2107.10478)


  We develop a Random Forest model to estimate the species distribution of
Asian elephants in India and study the inter and intra-annual spatiotemporal
variability of habitats suitable for them. Climatic, topographic variables and
satellite-derived Land Use/Land Cover (LULC), Net Primary Productivity (NPP),
Leaf Area Index (LAI), and Normalized Difference Vegetation Index (NDVI) are
used as predictors, and the species sighting data of Asian elephants from
Global Biodiversity Information Reserve is used to develop the Random Forest
model. A careful hyper-parameter tuning and training-validation-testing cycle
are completed to identify the significant predictors and develop a final model
that gives precision and recall of 0.78 and 0.77. The model is applied to
estimate the spatial and temporal variability of suitable habitats. We observe
that seasonal reduction in the suitable habitat may explain the migration
patterns of Asian elephants and the increasing human-elephant conflict.
Further, the total available suitable habitat area is observed to have reduced,
which exacerbates the problem. This machine learning model is intended to serve
as an input to the Agent-Based Model that we are building as part of our
Artificial Intelligence-driven decision support tool to reduce human-wildlife
conflict.

    

### [[2107.10480] Unsupervised Detection of Adversarial Examples with Model Explanations](http://arxiv.org/abs/2107.10480)


  Deep Neural Networks (DNNs) have shown remarkable performance in a diverse
range of machine learning applications. However, it is widely known that DNNs
are vulnerable to simple adversarial perturbations, which causes the model to
incorrectly classify inputs. In this paper, we propose a simple yet effective
method to detect adversarial examples, using methods developed to explain the
model's behavior. Our key observation is that adding small, humanly
imperceptible perturbations can lead to drastic changes in the model
explanations, resulting in unusual or irregular forms of explanations. From
this insight, we propose an unsupervised detection of adversarial examples
using reconstructor networks trained only on model explanations of benign
examples. Our evaluations with MNIST handwritten dataset show that our method
is capable of detecting adversarial examples generated by the state-of-the-art
algorithms with high confidence. To the best of our knowledge, this work is the
first in suggesting unsupervised defense method using model explanations.

    

### [[2107.10483] Efficient Neural Causal Discovery without Acyclicity Constraints](http://arxiv.org/abs/2107.10483)


  Learning the structure of a causal graphical model using both observational
and interventional data is a fundamental problem in many scientific fields. A
promising direction is continuous optimization for score-based methods, which
efficiently learn the causal graph in a data-driven manner. However, to date,
those methods require constrained optimization to enforce acyclicity or lack
convergence guarantees. In this paper, we present ENCO, an efficient structure
learning method for directed, acyclic causal graphs leveraging observational
and interventional data. ENCO formulates the graph search as an optimization of
independent edge likelihoods, with the edge orientation being modeled as a
separate parameter. Consequently, we can provide convergence guarantees of ENCO
under mild conditions without constraining the score function with respect to
acyclicity. In experiments, we show that ENCO can efficiently recover graphs
with hundreds of nodes, an order of magnitude larger than what was previously
possible, while handling deterministic variables and latent confounders.

    

### [[2107.10484] Neural Ordinary Differential Equation Model for Evolutionary Subspace Clustering and Its Applications](http://arxiv.org/abs/2107.10484)


  The neural ordinary differential equation (neural ODE) model has attracted
increasing attention in time series analysis for its capability to process
irregular time steps, i.e., data are not observed over equally-spaced time
intervals. In multi-dimensional time series analysis, a task is to conduct
evolutionary subspace clustering, aiming at clustering temporal data according
to their evolving low-dimensional subspace structures. Many existing methods
can only process time series with regular time steps while time series are
unevenly sampled in many situations such as missing data. In this paper, we
propose a neural ODE model for evolutionary subspace clustering to overcome
this limitation and a new objective function with subspace self-expressiveness
constraint is introduced. We demonstrate that this method can not only
interpolate data at any time step for the evolutionary subspace clustering
task, but also achieve higher accuracy than other state-of-the-art evolutionary
subspace clustering methods. Both synthetic and real-world data are used to
illustrate the efficacy of our proposed method.

    

### [[2107.10492] Bandit Quickest Changepoint Detection](http://arxiv.org/abs/2107.10492)


  Detecting abrupt changes in temporal behavior patterns is of interest in many
industrial and security applications. Abrupt changes are often local and
observable primarily through a well-aligned sensing action (e.g., a camera with
a narrow field-of-view). Due to resource constraints, continuous monitoring of
all of the sensors is impractical. We propose the bandit quickest changepoint
detection framework as a means of balancing sensing cost with detection delay.
In this framework, sensing actions (or sensors) are sequentially chosen, and
only measurements corresponding to chosen actions are observed. We derive an
information-theoretic lower bound on the detection delay for a general class of
finitely parameterized probability distributions. We then propose a
computationally efficient online sensing scheme, which seamlessly balances the
need for exploration of different sensing options with exploitation of querying
informative actions. We derive expected delay bounds for the proposed scheme
and show that these bounds match our information-theoretic lower bounds at low
false alarm rates, establishing optimality of the proposed method. We then
perform a number of experiments on synthetic and real datasets demonstrating
the efficacy of our proposed method.

    

### [[2107.10493] Abstract Reasoning via Logic-guided Generation](http://arxiv.org/abs/2107.10493)


  Abstract reasoning, i.e., inferring complicated patterns from given
observations, is a central building block of artificial general intelligence.
While humans find the answer by either eliminating wrong candidates or first
constructing the answer, prior deep neural network (DNN)-based methods focus on
the former discriminative approach. This paper aims to design a framework for
the latter approach and bridge the gap between artificial and human
intelligence. To this end, we propose logic-guided generation (LoGe), a novel
generative DNN framework that reduces abstract reasoning as an optimization
problem in propositional logic. LoGe is composed of three steps: extract
propositional variables from images, reason the answer variables with a logic
layer, and reconstruct the answer image from the variables. We demonstrate that
LoGe outperforms the black box DNN frameworks for generative abstract reasoning
under the RAVEN benchmark, i.e., reconstructing answers based on capturing
correct rules of various attributes from observations.

    

### [[2107.10495] Benchmarking AutoML Frameworks for Disease Prediction Using Medical Claims](http://arxiv.org/abs/2107.10495)


  We ascertain and compare the performances of AutoML tools on large, highly
imbalanced healthcare datasets.
We generated a large dataset using historical administrative claims including
demographic information and flags for disease codes in four different time
windows prior to 2019. We then trained three AutoML tools on this dataset to
predict six different disease outcomes in 2019 and evaluated model performances
on several metrics.
The AutoML tools showed improvement from the baseline random forest model but
did not differ significantly from each other. All models recorded low area
under the precision-recall curve and failed to predict true positives while
keeping the true negative rate high. Model performance was not directly related
to prevalence. We provide a specific use-case to illustrate how to select a
threshold that gives the best balance between true and false positive rates, as
this is an important consideration in medical applications.
Healthcare datasets present several challenges for AutoML tools, including
large sample size, high imbalance, and limitations in the available features
types. Improvements in scalability, combinations of imbalance-learning
resampling and ensemble approaches, and curated feature selection are possible
next steps to achieve better performance.
Among the three explored, no AutoML tool consistently outperforms the rest in
terms of predictive performance. The performances of the models in this study
suggest that there may be room for improvement in handling medical claims data.
Finally, selection of the optimal prediction threshold should be guided by the
specific practical application.

    

### [[2107.10504] External-Memory Networks for Low-Shot Learning of Targets in Forward-Looking-Sonar Imagery](http://arxiv.org/abs/2107.10504)


  We propose a memory-based framework for real-time, data-efficient target
analysis in forward-looking-sonar (FLS) imagery. Our framework relies on first
removing non-discriminative details from the imagery using a small-scale
DenseNet-inspired network. Doing so simplifies ensuing analyses and permits
generalizing from few labeled examples. We then cascade the filtered imagery
into a novel NeuralRAM-based convolutional matching network, NRMN, for low-shot
target recognition. We employ a small-scale FlowNet, LFN to align and register
FLS imagery across local temporal scales. LFN enables target label consensus
voting across images and generally improves target detection and recognition
rates.
We evaluate our framework using real-world FLS imagery with multiple broad
target classes that have high intra-class variability and rich sub-class
structure. We show that few-shot learning, with anywhere from ten to thirty
class-specific exemplars, performs similarly to supervised deep networks
trained on hundreds of samples per class. Effective zero-shot learning is also
possible. High performance is realized from the inductive-transfer properties
of NRMNs when distractor elements are removed.

    

### [[2107.10507] Evaluating the Quality of Finite Element Meshes with Machine Learning](http://arxiv.org/abs/2107.10507)


  This paper addresses the problem of evaluating the quality of finite element
meshes for the purpose of structural mechanic simulations. It proposes the
application of a machine learning model trained on data collected from expert
evaluations. The task is characterised as a classification problem, where
quality of each individual element in a mesh is determined by its own
properties and adjacency structures. A domain-specific, yet simple
representation is proposed such that off-the-shelf machine learning methods can
be applied. Experimental data from industry practice demonstrates promising
results.

    

### [[2107.10536] Improving the Authentication with Built-in Camera ProtocolUsing Built-in Motion Sensors: A Deep Learning Solution](http://arxiv.org/abs/2107.10536)


  We propose an enhanced version of the Authentication with Built-in Camera
(ABC) protocol by employing a deep learning solution based on built-in motion
sensors. The standard ABC protocol identifies mobile devices based on the
photo-response non-uniformity (PRNU) of the camera sensor, while also
considering QR-code-based meta-information. During authentication, the user is
required to take two photos that contain two QR codes presented on a screen.
The presented QR code images also contain a unique probe signal, similar to a
camera fingerprint, generated by the protocol. During verification, the server
computes the fingerprint of the received photos and authenticates the user if
(i) the probe signal is present, (ii) the metadata embedded in the QR codes is
correct and (iii) the camera fingerprint is identified correctly. However, the
protocol is vulnerable to forgery attacks when the attacker can compute the
camera fingerprint from external photos, as shown in our preliminary work. In
this context, we propose an enhancement for the ABC protocol based on motion
sensor data, as an additional and passive authentication layer. Smartphones can
be identified through their motion sensor data, which, unlike photos, is never
posted by users on social media platforms, thus being more secure than using
photographs alone. To this end, we transform motion signals into embedding
vectors produced by deep neural networks, applying Support Vector Machines for
the smartphone identification task. Our change to the ABC protocol results in a
multi-modal protocol that lowers the false acceptance rate for the attack
proposed in our previous work to a percentage as low as 0.07%.

    

### [[2107.10554] Out of the Shadows: Analyzing Anonymous' Twitter Resurgence during the 2020 Black Lives Matter Protests](http://arxiv.org/abs/2107.10554)


  Recently, there had been little notable activity from the once prominent
hacktivist group, Anonymous. The group, responsible for activist-based cyber
attacks on major businesses and governments, appeared to have fragmented after
key members were arrested in 2013. In response to the major Black Lives Matter
(BLM) protests that occurred after the killing of George Floyd, however,
reports indicated that the group was back. To examine this apparent resurgence,
we conduct a large-scale study of Anonymous affiliates on Twitter. To this end,
we first use machine learning to identify a significant network of more than
33,000 Anonymous accounts. Through topic modelling of tweets collected from
these accounts, we find evidence of sustained interest in topics related to
BLM. We then use sentiment analysis on tweets focused on these topics, finding
evidence of a united approach amongst the group, with positive tweets typically
being used to express support towards BLM, and negative tweets typically being
used to criticize police actions. Finally, we examine the presence of
automation in the network, identifying indications of bot-like behavior across
the majority of Anonymous accounts. These findings show that whilst the group
has seen a resurgence during the protests, bot activity may be responsible for
exaggerating the extent of this resurgence.

    

### [[2107.10558] A Proactive Management Scheme for Data Synopses at the Edge](http://arxiv.org/abs/2107.10558)


  The combination of the infrastructure provided by the Internet of Things
(IoT) with numerous processing nodes present at the Edge Computing (EC)
ecosystem opens up new pathways to support intelligent applications. Such
applications can be provided upon humongous volumes of data collected by IoT
devices being transferred to the edge nodes through the network. Various
processing activities can be performed on the discussed data and multiple
collaborative opportunities between EC nodes can facilitate the execution of
the desired tasks. In order to support an effective interaction between edge
nodes, the knowledge about the geographically distributed data should be
shared. Obviously, the migration of large amounts of data will harm the
stability of the network stability and its performance. In this paper, we
recommend the exchange of data synopses than real data between EC nodes to
provide them with the necessary knowledge about peer nodes owning similar data.
This knowledge can be valuable when considering decisions such as data/service
migration and tasks offloading. We describe an continuous reasoning model that
builds a temporal similarity map of the available datasets to get nodes
understanding the evolution of data in their peers. We support the proposed
decision making mechanism through an intelligent similarity extraction scheme
based on an unsupervised machine learning model, and, at the same time, combine
it with a statistical measure that represents the trend of the so-called
discrepancy quantum. Our model can reveal the differences in the exchanged
synopses and provide a datasets similarity map which becomes the appropriate
knowledge base to support the desired processing activities. We present the
problem under consideration and suggest a solution for that, while, at the same
time, we reveal its advantages and disadvantages through a large number of
experiments.

    

### [[2107.10567] An overcome of far-distance limitation on tunnel CCTV-based accident detection in AI deep-learning frameworks](http://arxiv.org/abs/2107.10567)


  Tunnel CCTVs are installed to low height and long-distance interval. However,
because of the limitation of installation height, severe perspective effect in
distance occurs, and it is almost impossible to detect vehicles in far distance
from the CCTV in the existing tunnel CCTV-based accident detection system
(Pflugfelder 2005). To overcome the limitation, a vehicle object is detected
through an object detection algorithm based on an inverse perspective transform
by re-setting the region of interest (ROI). It can detect vehicles that are far
away from the CCTV. To verify this process, this paper creates each dataset
consisting of images and bounding boxes based on the original and warped images
of the CCTV at the same time, and then compares performance of the deep
learning object detection models trained with the two datasets. As a result,
the model that trained the warped image was able to detect vehicle objects more
accurately at the position far from the CCTV compared to the model that trained
the original image.

    

### [[2107.10585] MobileCharger: an Autonomus Mobile Robot with Inverted Delta Actuator for Robust and Safe Robot Charging](http://arxiv.org/abs/2107.10585)


  MobileCharger is a novel mobile charging robot with an Inverted Delta
actuator for safe and robust energy transfer between two mobile robots. The
RGB-D camera-based computer vision system allows to detect the electrodes on
the target mobile robot using a convolutional neural network (CNN). The
embedded high-fidelity tactile sensors are applied to estimate the misalignment
between the electrodes on the charger mechanism and the electrodes on the main
robot using CNN based on pressure data on the contact surfaces. Thus, the
developed vision-tactile perception system allows precise positioning of the
end effector of the actuator and ensures a reliable connection between the
electrodes of the two robots. The experimental results showed high average
precision (84.2%) for electrode detection using CNN. The percentage of
successful trials of the CNN-based electrode search algorithm reached 83% and
the average execution time accounted for 60 s. MobileCharger could introduce a
new level of charging systems and increase the prevalence of autonomous mobile
robots.

    

### [[2107.10599] Towards Explaining Adversarial Examples Phenomenon in Artificial Neural Networks](http://arxiv.org/abs/2107.10599)


  In this paper, we study the adversarial examples existence and adversarial
training from the standpoint of convergence and provide evidence that pointwise
convergence in ANNs can explain these observations. The main contribution of
our proposal is that it relates the objective of the evasion attacks and
adversarial training with concepts already defined in learning theory. Also, we
extend and unify some of the other proposals in the literature and provide
alternative explanations on the observations made in those proposals. Through
different experiments, we demonstrate that the framework is valuable in the
study of the phenomenon and is applicable to real-world problems.

    

### [[2107.10606] cCorrGAN: Conditional Correlation GAN for Learning Empirical Conditional Distributions in the Elliptope](http://arxiv.org/abs/2107.10606)


  We propose a methodology to approximate conditional distributions in the
elliptope of correlation matrices based on conditional generative adversarial
networks. We illustrate the methodology with an application from quantitative
finance: Monte Carlo simulations of correlated returns to compare risk-based
portfolio construction methods. Finally, we discuss about current limitations
and advocate for further exploration of the elliptope geometry to improve
results.

    

### [[2107.10607] 3D Shape Generation with Grid-based Implicit Functions](http://arxiv.org/abs/2107.10607)


  Previous approaches to generate shapes in a 3D setting train a GAN on the
latent space of an autoencoder (AE). Even though this produces convincing
results, it has two major shortcomings. As the GAN is limited to reproduce the
dataset the AE was trained on, we cannot reuse a trained AE for novel data.
Furthermore, it is difficult to add spatial supervision into the generation
process, as the AE only gives us a global representation. To remedy these
issues, we propose to train the GAN on grids (i.e. each cell covers a part of a
shape). In this representation each cell is equipped with a latent vector
provided by an AE. This localized representation enables more expressiveness
(since the cell-based latent vectors can be combined in novel ways) as well as
spatial control of the generation process (e.g. via bounding boxes). Our method
outperforms the current state of the art on all established evaluation
measures, proposed for quantitatively evaluating the generative capabilities of
GANs. We show limitations of these measures and propose the adaptation of a
robust criterion from statistical analysis as an alternative.

    

### [[2107.10609] Data Considerations in Graph Representation Learning for Supply Chain Networks](http://arxiv.org/abs/2107.10609)


  Supply chain network data is a valuable asset for businesses wishing to
understand their ethical profile, security of supply, and efficiency.
Possession of a dataset alone however is not a sufficient enabler of actionable
decisions due to incomplete information. In this paper, we present a graph
representation learning approach to uncover hidden dependency links that focal
companies may not be aware of. To the best of our knowledge, our work is the
first to represent a supply chain as a heterogeneous knowledge graph with
learnable embeddings. We demonstrate that our representation facilitates
state-of-the-art performance on link prediction of a global automotive supply
chain network using a relational graph convolutional network. It is anticipated
that our method will be directly applicable to businesses wishing to sever
links with nefarious entities and mitigate risk of supply failure. More
abstractly, it is anticipated that our method will be useful to inform
representation learning of supply chain networks for downstream tasks beyond
link prediction.

    

### [[2107.10624] HANT: Hardware-Aware Network Transformation](http://arxiv.org/abs/2107.10624)


  Given a trained network, how can we accelerate it to meet efficiency needs
for deployment on particular hardware? The commonly used hardware-aware network
compression techniques address this question with pruning, kernel fusion,
quantization and lowering precision. However, these approaches do not change
the underlying network operations. In this paper, we propose hardware-aware
network transformation (HANT), which accelerates a network by replacing
inefficient operations with more efficient alternatives using a neural
architecture search like approach. HANT tackles the problem in two phase: In
the first phase, a large number of alternative operations per every layer of
the teacher model is trained using layer-wise feature map distillation. In the
second phase, the combinatorial selection of efficient operations is relaxed to
an integer optimization problem that can be solved in a few seconds. We extend
HANT with kernel fusion and quantization to improve throughput even further.
Our experimental results on accelerating the EfficientNet family show that HANT
can accelerate them by up to 3.6x with <0.4% drop in the top-1 accuracy on the
ImageNet dataset. When comparing the same latency level, HANT can accelerate
EfficientNet-B4 to the same latency as EfficientNet-B1 while having 3% higher
accuracy. We examine a large pool of operations, up to 197 per layer, and we
provide insights into the selected operations and final architectures.

    

### [[2107.10637] A baseline model for computationally inexpensive speech recognition for Kazakh using the Coqui STT framework](http://arxiv.org/abs/2107.10637)


  Mobile devices are transforming the way people interact with computers, and
speech interfaces to applications are ever more important. Automatic Speech
Recognition systems recently published are very accurate, but often require
powerful machinery (specialised Graphical Processing Units) for inference,
which makes them impractical to run on commodity devices, especially in
streaming mode. Impressed by the accuracy of, but dissatisfied with the
inference times of the baseline Kazakh ASR model of (Khassanov et al.,2021)
when not using a GPU, we trained a new baseline acoustic model (on the same
dataset as the aforementioned paper) and three language models for use with the
Coqui STT framework. Results look promising, but further epochs of training and
parameter sweeping or, alternatively, limiting the vocabulary that the ASR
system must support, is needed to reach a production-level accuracy.

    

### [[2107.10640] Hash-Based Tree Similarity and Simplification in Genetic Programming for Symbolic Regression](http://arxiv.org/abs/2107.10640)


  We introduce in this paper a runtime-efficient tree hashing algorithm for the
identification of isomorphic subtrees, with two important applications in
genetic programming for symbolic regression: fast, online calculation of
population diversity and algebraic simplification of symbolic expression trees.
Based on this hashing approach, we propose a simple diversity-preservation
mechanism with promising results on a collection of symbolic regression
benchmark problems.

    

### [[2107.10647] Análisis de Canasta de mercado en supermercados mediante mapas auto-organizados](http://arxiv.org/abs/2107.10647)


  Introduction: An important chain of supermarkets in the western zone of the
capital of Chile, needs to obtain key information to make decisions, this
information is available in the databases but needs to be processed due to the
complexity and quantity of information which becomes difficult to visualiz,.
Method: For this purpose, an algorithm was developed using artificial neural
networks applying Kohonen's SOM method. To carry it out, certain key procedures
must be followed to develop it, such as data mining that will be responsible
for filtering and then use only the relevant data for market basket analysis.
After filtering the information, the data must be prepared. After data
preparation, we prepared the Python programming environment to adapt it to the
sample data, then proceed to train the SOM with its parameters set after test
results. Result: the result of the SOM obtains the relationship between the
products that were most purchased by positioning them topologically close, to
form promotions, packs and bundles for the retail manager to take into
consideration, because these relationships were obtained as a result of the SOM
training with the real transactions of the clients. Conclusion: Based on this,
recommendations on frequent shopping baskets have been made to the supermarket
chain that provided the data used in the research

    

### [[2107.10650] Read, Attend, and Code: Pushing the Limits of Medical Codes Prediction from Clinical Notes by Machines](http://arxiv.org/abs/2107.10650)


  Prediction of medical codes from clinical notes is both a practical and
essential need for every healthcare delivery organization within current
medical systems. Automating annotation will save significant time and excessive
effort spent by human coders today. However, the biggest challenge is directly
identifying appropriate medical codes out of several thousands of
high-dimensional codes from unstructured free-text clinical notes. In the past
three years, with Convolutional Neural Networks (CNN) and Long Short-Term
Memory (LTSM) networks, there have been vast improvements in tackling the most
challenging benchmark of the MIMIC-III-full-label inpatient clinical notes
dataset. This progress raises the fundamental question of how far automated
machine learning (ML) systems are from human coders' working performance. We
assessed the baseline of human coders' performance on the same subsampled
testing set. We also present our Read, Attend, and Code (RAC) model for
learning the medical code assignment mappings. By connecting convolved
embeddings with self-attention and code-title guided attention modules,
combined with sentence permutation-based data augmentations and stochastic
weight averaging training, RAC establishes a new state of the art (SOTA),
considerably outperforming the current best Macro-F1 by 18.7%, and reaches past
the human-level coding baseline. This new milestone marks a meaningful step
toward fully autonomous medical coding (AMC) in machines reaching parity with
human coders' performance in medical code prediction.

    

### [[2107.10651] Semiparametric Latent Topic Modeling on Consumer-Generated Corpora](http://arxiv.org/abs/2107.10651)


  Legacy procedures for topic modelling have generally suffered problems of
overfitting and a weakness towards reconstructing sparse topic structures. With
motivation from a consumer-generated corpora, this paper proposes
semiparametric topic model, a two-step approach utilizing nonnegative matrix
factorization and semiparametric regression in topic modeling. The model
enables the reconstruction of sparse topic structures in the corpus and
provides a generative model for predicting topics in new documents entering the
corpus. Assuming the presence of auxiliary information related to the topics,
this approach exhibits better performance in discovering underlying topic
structures in cases where the corpora are small and limited in vocabulary. In
an actual consumer feedback corpus, the model also demonstrably provides
interpretable and useful topic definitions comparable with those produced by
other methods.

    

### [[2107.10652] A Systematic Literature Review of Automated ICD Coding and Classification Systems using Discharge Summaries](http://arxiv.org/abs/2107.10652)


  Codification of free-text clinical narratives have long been recognised to be
beneficial for secondary uses such as funding, insurance claim processing and
research. The current scenario of assigning codes is a manual process which is
very expensive, time-consuming and error prone. In recent years, many
researchers have studied the use of Natural Language Processing (NLP), related
Machine Learning (ML) and Deep Learning (DL) methods and techniques to resolve
the problem of manual coding of clinical narratives and to assist human coders
to assign clinical codes more accurately and efficiently. This systematic
literature review provides a comprehensive overview of automated clinical
coding systems that utilises appropriate NLP, ML and DL methods and techniques
to assign ICD codes to discharge summaries. We have followed the Preferred
Reporting Items for Systematic Reviews and Meta-Analyses(PRISMA) guidelines and
conducted a comprehensive search of publications from January, 2010 to December
2020 in four academic databases- PubMed, ScienceDirect, Association for
Computing Machinery(ACM) Digital Library, and the Association for Computational
Linguistics(ACL) Anthology. We reviewed 7,556 publications; 38 met the
inclusion criteria. This review identified: datasets having discharge
summaries; NLP techniques along with some other data extraction processes,
different feature extraction and embedding techniques. To measure the
performance of classification methods, different evaluation metrics are used.
Lastly, future research directions are provided to scholars who are interested
in automated ICD code assignment. Efforts are still required to improve ICD
code prediction accuracy, availability of large-scale de-identified clinical
corpora with the latest version of the classification system. This can be a
platform to guide and share knowledge with the less experienced coders and
researchers.

    

### [[2107.10654] Fast Low-Rank Tensor Decomposition by Ridge Leverage Score Sampling](http://arxiv.org/abs/2107.10654)


  Low-rank tensor decomposition generalizes low-rank matrix approximation and
is a powerful technique for discovering low-dimensional structure in
high-dimensional data. In this paper, we study Tucker decompositions and use
tools from randomized numerical linear algebra called ridge leverage scores to
accelerate the core tensor update step in the widely-used alternating least
squares (ALS) algorithm. Updating the core tensor, a severe bottleneck in ALS,
is a highly-structured ridge regression problem where the design matrix is a
Kronecker product of the factor matrices. We show how to use approximate ridge
leverage scores to construct a sketched instance for any ridge regression
problem such that the solution vector for the sketched problem is a
$(1+\varepsilon)$-approximation to the original instance. Moreover, we show
that classical leverage scores suffice as an approximation, which then allows
us to exploit the Kronecker structure and update the core tensor in time that
depends predominantly on the rank and the sketching parameters (i.e., sublinear
in the size of the input tensor). We also give upper bounds for ridge leverage
scores as rows are removed from the design matrix (e.g., if the tensor has
missing entries), and we demonstrate the effectiveness of our approximate ridge
regressioni algorithm for large, low-rank Tucker decompositions on both
synthetic and real-world data.

    

### [[2107.10655] Lumen: A Machine Learning Framework to Expose Influence Cues in Text](http://arxiv.org/abs/2107.10655)


  Phishing and disinformation are popular social engineering attacks with
attackers invariably applying influence cues in texts to make them more
appealing to users. We introduce Lumen, a learning-based framework that exposes
influence cues in text: (i) persuasion, (ii) framing, (iii) emotion, (iv)
objectivity/subjectivity, (v) guilt/blame, and (vi) use of emphasis. Lumen was
trained with a newly developed dataset of 3K texts comprised of disinformation,
phishing, hyperpartisan news, and mainstream news. Evaluation of Lumen in
comparison to other learning models showed that Lumen and LSTM presented the
best F1-micro score, but Lumen yielded better interpretability. Our results
highlight the promise of ML to expose influence cues in text, towards the goal
of application in automatic labeling tools to improve the accuracy of
human-based detection and reduce the likelihood of users falling for deceptive
online content.

    

### [[2107.10657] Solving inverse problems with deep neural networks driven by sparse signal decomposition in a physics-based dictionary](http://arxiv.org/abs/2107.10657)


  Deep neural networks (DNN) have an impressive ability to invert very complex
models, i.e. to learn the generative parameters from a model's output. Once
trained, the forward pass of a DNN is often much faster than traditional,
optimization-based methods used to solve inverse problems. This is however done
at the cost of lower interpretability, a fundamental limitation in most medical
applications. We propose an approach for solving general inverse problems which
combines the efficiency of DNN and the interpretability of traditional
analytical methods. The measurements are first projected onto a dense
dictionary of model-based responses. The resulting sparse representation is
then fed to a DNN with an architecture driven by the problem's physics for fast
parameter learning. Our method can handle generative forward models that are
costly to evaluate and exhibits similar performance in accuracy and computation
time as a fully-learned DNN, while maintaining high interpretability and being
easier to train. Concrete results are shown on an example of model-based brain
parameter estimation from magnetic resonance imaging (MRI).

    

### [[2107.10658] Digital Einstein Experience: Fast Text-to-Speech for Conversational AI](http://arxiv.org/abs/2107.10658)


  We describe our approach to create and deliver a custom voice for a
conversational AI use-case. More specifically, we provide a voice for a Digital
Einstein character, to enable human-computer interaction within the digital
conversation experience. To create the voice which fits the context well, we
first design a voice character and we produce the recordings which correspond
to the desired speech attributes. We then model the voice. Our solution
utilizes Fastspeech 2 for log-scaled mel-spectrogram prediction from phonemes
and Parallel WaveGAN to generate the waveforms. The system supports a character
input and gives a speech waveform at the output. We use a custom dictionary for
selected words to ensure their proper pronunciation. Our proposed cloud
architecture enables for fast voice delivery, making it possible to talk to the
digital version of Albert Einstein in real-time.

    

### [[2107.10661] Robust Topology Optimization Using Variational Autoencoders](http://arxiv.org/abs/2107.10661)


  Topology Optimization is the process of finding the optimal arrangement of
materials within a design domain by minimizing a cost function, subject to some
performance constraints. Robust topology optimization (RTO) also incorporates
the effect of input uncertainties and produces a design with the best average
performance of the structure while reducing the response sensitivity to input
uncertainties. It is computationally expensive to carry out RTO using finite
element and Monte Carlo sampling. In this work, we use neural network
surrogates to enable a faster solution approach via surrogate-based
optimization and build a Variational Autoencoder (VAE) to transform the the
high dimensional design space into a low dimensional one. Furthermore, finite
element solvers will be replaced by a neural network surrogate. Also, to
further facilitate the design exploration, we limit our search to a subspace,
which consists of designs that are solutions to deterministic topology
optimization problems under different realizations of input uncertainties. With
these neural network approximations, a gradient-based optimization approach is
formed to minimize the predicted objective function over the low dimensional
design subspace. We demonstrate the effectiveness of the proposed approach on
two compliance minimization problems and show that VAE performs well on
learning the features of the design from minimal training data, and that
converting the design space into a low dimensional latent space makes the
problem computationally efficient. The resulting gradient-based optimization
algorithm produces optimal designs with lower robust compliances than those
observed in the training set.

    

### [[2107.10663] Fed-ensemble: Improving Generalization through Model Ensembling in Federated Learning](http://arxiv.org/abs/2107.10663)


  In this paper we propose Fed-ensemble: a simple approach that bringsmodel
ensembling to federated learning (FL). Instead of aggregating localmodels to
update a single global model, Fed-ensemble uses random permutations to update a
group of K models and then obtains predictions through model averaging.
Fed-ensemble can be readily utilized within established FL methods and does not
impose a computational overhead as it only requires one of the K models to be
sent to a client in each communication round. Theoretically, we show that
predictions on newdata from all K models belong to the same predictive
posterior distribution under a neural tangent kernel regime. This result in
turn sheds light onthe generalization advantages of model averaging. We also
illustrate thatFed-ensemble has an elegant Bayesian interpretation. Empirical
results show that our model has superior performance over several FL
algorithms,on a wide range of data sets, and excels in heterogeneous settings
often encountered in FL applications.

    

### [[2107.10667] $β$-Annealed Variational Autoencoder for glitches](http://arxiv.org/abs/2107.10667)


  Gravitational wave detectors such as LIGO and Virgo are susceptible to
various types of instrumental and environmental disturbances known as glitches
which can mask and mimic gravitational waves. While there are 22 classes of
non-Gaussian noise gradients currently identified, the number of classes is
likely to increase as these detectors go through commissioning between
observation runs. Since identification and labelling new noise gradients can be
arduous and time-consuming, we propose $\beta$-Annelead VAEs to learn
representations from spectograms in an unsupervised way. Using the same
formulation as \cite{alemi2017fixing}, we view
Bottleneck-VAEs~cite{burgess2018understanding} through the lens of information
theory and connect them to $\beta$-VAEs~cite{higgins2017beta}. Motivated by
this connection, we propose an annealing schedule for the hyperparameter
$\beta$ in $\beta$-VAEs which has advantages of: 1) One fewer hyperparameter to
tune, 2) Better reconstruction quality, while producing similar levels of
disentanglement.

    

### [[2107.10669] Accuracy analysis of Educational Data Mining using Feature Selection Algorithm](http://arxiv.org/abs/2107.10669)


  Abstract - Gathering relevant information to predict student academic
progress is a tedious task. Due to the large amount of irrelevant data present
in databases which provides inaccurate results. Currently, it is not possible
to accurately measure and analyze student data because there are too many
irrelevant attributes and features in the data. With the help of Educational
Data Mining (EDM), the quality of information can be improved. This research
demonstrates how EDM helps to measure the accuracy of data using relevant
attributes and machine learning algorithms performed. With EDM, irrelevant
features are removed without changing the original data. The data set used in
this study was taken from this http URL. The results compared on the basis of
recall, precision and f-measure to check the accuracy of the student data. The
importance of this research is to help improve the quality of educational
research by providing more accurate results for researchers.

    

### [[2107.10670] Structure-aware Interactive Graph Neural Networks for the Prediction of Protein-Ligand Binding Affinity](http://arxiv.org/abs/2107.10670)


  Drug discovery often relies on the successful prediction of protein-ligand
binding affinity. Recent advances have shown great promise in applying graph
neural networks (GNNs) for better affinity prediction by learning the
representations of protein-ligand complexes. However, existing solutions
usually treat protein-ligand complexes as topological graph data, thus the
biomolecular structural information is not fully utilized. The essential
long-range interactions among atoms are also neglected in GNN models. To this
end, we propose a structure-aware interactive graph neural network (SIGN) which
consists of two components: polar-inspired graph attention layers (PGAL) and
pairwise interactive pooling (PiPool). Specifically, PGAL iteratively performs
the node-edge aggregation process to update embeddings of nodes and edges while
preserving the distance and angle information among atoms. Then, PiPool is
adopted to gather interactive edges with a subsequent reconstruction loss to
reflect the global interactions. Exhaustive experimental study on two
benchmarks verifies the superiority of SIGN.

    

### [[2107.10692] Selective Pseudo-label Clustering](http://arxiv.org/abs/2107.10692)


  Deep neural networks (DNNs) offer a means of addressing the challenging task
of clustering high-dimensional data. DNNs can extract useful features, and so
produce a lower dimensional representation, which is more amenable to
clustering techniques. As clustering is typically performed in a purely
unsupervised setting, where no training labels are available, the question then
arises as to how the DNN feature extractor can be trained. The most accurate
existing approaches combine the training of the DNN with the clustering
objective, so that information from the clustering process can be used to
update the DNN to produce better features for clustering. One problem with this
approach is that these ``pseudo-labels'' produced by the clustering algorithm
are noisy, and any errors that they contain will hurt the training of the DNN.
In this paper, we propose selective pseudo-label clustering, which uses only
the most confident pseudo-labels for training the~DNN. We formally prove the
performance gains under certain conditions. Applied to the task of image
clustering, the new approach achieves a state-of-the-art performance on three
popular image datasets. Code is available at
this https URL.

    

### [[2107.10703] Typing assumptions improve identification in causal discovery](http://arxiv.org/abs/2107.10703)


  Causal discovery from observational data is a challenging task to which an
exact solution cannot always be identified. Under assumptions about the
data-generative process, the causal graph can often be identified up to an
equivalence class. Proposing new realistic assumptions to circumscribe such
equivalence classes is an active field of research. In this work, we propose a
new set of assumptions that constrain possible causal relationships based on
the nature of the variables. We thus introduce typed directed acyclic graphs,
in which variable types are used to determine the validity of causal
relationships. We demonstrate, both theoretically and empirically, that the
proposed assumptions can result in significant gains in the identification of
the causal graph.

    

### [[2107.10706] Distributed Saddle-Point Problems Under Similarity](http://arxiv.org/abs/2107.10706)


  We study solution methods for (strongly-)convex-(strongly)-concave
Saddle-Point Problems (SPPs) over networks of two type - master/workers (thus
centralized) architectures and meshed (thus decentralized) networks. The local
functions at each node are assumed to be similar, due to statistical data
similarity or otherwise. We establish lower complexity bounds for a fairly
general class of algorithms solving the SPP. We show that a given suboptimality
$\epsilon>0$ is achieved over master/workers networks in
$\Omega\big(\Delta\cdot \delta/\mu\cdot \log (1/\varepsilon)\big)$ rounds of
communications, where $\delta>0$ measures the degree of similarity of the local
functions, $\mu$ is their strong convexity constant, and $\Delta$ is the
diameter of the network. The lower communication complexity bound over meshed
networks reads $\Omega\big(1/{\sqrt{\rho}} \cdot {\delta}/{\mu}\cdot\log
(1/\varepsilon)\big)$, where $\rho$ is the (normalized) eigengap of the gossip
matrix used for the communication between neighbouring nodes. We then propose
algorithms matching the lower bounds over either types of networks (up to
log-factors). We assess the effectiveness of the proposed algorithms on a
robust logistic regression problem.

    

### [[2107.10709] A Framework for Imbalanced Time-series Forecasting](http://arxiv.org/abs/2107.10709)


  Time-series forecasting plays an important role in many domains. Boosted by
the advances in Deep Learning algorithms, it has for instance been used to
predict wind power for eolic energy production, stock market fluctuations, or
motor overheating. In some of these tasks, we are interested in predicting
accurately some particular moments which often are underrepresented in the
dataset, resulting in a problem known as imbalanced regression. In the
literature, while recognized as a challenging problem, limited attention has
been devoted on how to handle the problem in a practical setting. In this
paper, we put forward a general approach to analyze time-series forecasting
problems focusing on those underrepresented moments to reduce imbalances. Our
approach has been developed based on a case study in a large industrial
company, which we use to exemplify the approach.

    

### [[2107.10710] DeltaCharger: Charging Robot with Inverted Delta Mechanism and CNN-driven High Fidelity Tactile Perception for Precise 3D Positioning](http://arxiv.org/abs/2107.10710)


  DeltaCharger is a novel charging robot with an Inverted Delta structure for
3D positioning of electrodes to achieve robust and safe transferring energy
between two mobile robots. The embedded high-fidelity tactile sensors allow to
estimate the angular, vertical and horizontal misalignments between electrodes
on the charger mechanism and electrodes on the target robot using pressure data
on the contact surfaces. This is crucial for preventing a short circuit. In
this paper, the mechanism of the developed prototype and evaluation study of
different machine learning models for misalignment prediction are presented.
The experimental results showed that the proposed system can measure the angle,
vertical and horizontal values of misalignment from pressure data with an
accuracy of 95.46%, 98.2%, and 86.9%, respectively, using a Convolutional
Neural Network (CNN). DeltaCharger can potentially bring a new level of
charging systems and improve the prevalence of mobile autonomous robots.

    

### [[2107.10711] Physics-informed neural networks for solving Reynolds-averaged Navier$\unicode{x2013}$Stokes equations](http://arxiv.org/abs/2107.10711)


  Physics-informed neural networks (PINNs) are successful machine-learning
methods for the solution and identification of partial differential equations
(PDEs). We employ PINNs for solving the Reynolds-averaged
Navier$\unicode{x2013}$Stokes (RANS) equations for incompressible turbulent
flows without any specific model or assumption for turbulence, and by taking
only the data on the domain boundaries. We first show the applicability of
PINNs for solving the Navier$\unicode{x2013}$Stokes equations for laminar flows
by solving the Falkner$\unicode{x2013}$Skan boundary layer. We then apply PINNs
for the simulation of four turbulent-flow cases, i.e., zero-pressure-gradient
boundary layer, adverse-pressure-gradient boundary layer, and turbulent flows
over a NACA4412 airfoil and the periodic hill. Our results show the excellent
applicability of PINNs for laminar flows with strong pressure gradients, where
predictions with less than 1% error can be obtained. For turbulent flows, we
also obtain very good accuracy on simulation results even for the
Reynolds-stress components.

    

### [[2107.10716] Project Achoo: A Practical Model and Application for COVID-19 Detection from Recordings of Breath, Voice, and Cough](http://arxiv.org/abs/2107.10716)


  The COVID-19 pandemic created a significant interest and demand for infection
detection and monitoring solutions. In this paper we propose a machine learning
method to quickly triage COVID-19 using recordings made on consumer devices.
The approach combines signal processing methods with fine-tuned deep learning
networks and provides methods for signal denoising, cough detection and
classification. We have also developed and deployed a mobile application that
uses symptoms checker together with voice, breath and cough signals to detect
COVID-19 infection. The application showed robust performance on both open
sourced datasets and on the noisy data collected during beta testing by the end
users.

    

### [[2107.10718] Segmentation of Cardiac Structures via Successive Subspace Learning with Saab Transform from Cine MRI](http://arxiv.org/abs/2107.10718)


  Assessment of cardiovascular disease (CVD) with cine magnetic resonance
imaging (MRI) has been used to non-invasively evaluate detailed cardiac
structure and function. Accurate segmentation of cardiac structures from cine
MRI is a crucial step for early diagnosis and prognosis of CVD, and has been
greatly improved with convolutional neural networks (CNN). There, however, are
a number of limitations identified in CNN models, such as limited
interpretability and high complexity, thus limiting their use in clinical
practice. In this work, to address the limitations, we propose a lightweight
and interpretable machine learning model, successive subspace learning with the
subspace approximation with adjusted bias (Saab) transform, for accurate and
efficient segmentation from cine MRI. Specifically, our segmentation framework
is comprised of the following steps: (1) sequential expansion of near-to-far
neighborhood at different resolutions; (2) channel-wise subspace approximation
using the Saab transform for unsupervised dimension reduction; (3) class-wise
entropy guided feature selection for supervised dimension reduction; (4)
concatenation of features and pixel-wise classification with gradient boost;
and (5) conditional random field for post-processing. Experimental results on
the ACDC 2017 segmentation database, showed that our framework performed better
than state-of-the-art U-Net models with 200$\times$ fewer parameters in
delineating the left ventricle, right ventricle, and myocardium, thus showing
its potential to be used in clinical practice.

    

### [[2107.10731] Neural Variational Gradient Descent](http://arxiv.org/abs/2107.10731)


  Particle-based approximate Bayesian inference approaches such as Stein
Variational Gradient Descent (SVGD) combine the flexibility and convergence
guarantees of sampling methods with the computational benefits of variational
inference. In practice, SVGD relies on the choice of an appropriate kernel
function, which impacts its ability to model the target distribution -- a
challenging problem with only heuristic solutions. We propose Neural
Variational Gradient Descent (NVGD), which is based on parameterizing the
witness function of the Stein discrepancy by a deep neural network whose
parameters are learned in parallel to the inference, mitigating the necessity
to make any kernel choices whatsoever. We empirically evaluate our method on
popular synthetic inference problems, real-world Bayesian linear regression,
and Bayesian neural network inference.

    

### [[2107.10746] High Frequency EEG Artifact Detection with Uncertainty via Early Exit Paradigm](http://arxiv.org/abs/2107.10746)


  Electroencephalography (EEG) is crucial for the monitoring and diagnosis of
brain disorders. However, EEG signals suffer from perturbations caused by
non-cerebral artifacts limiting their efficacy. Current artifact detection
pipelines are resource-hungry and rely heavily on hand-crafted features.
Moreover, these pipelines are deterministic in nature, making them unable to
capture predictive uncertainty. We propose E4G, a deep learning framework for
high frequency EEG artifact detection. Our framework exploits the early exit
paradigm, building an implicit ensemble of models capable of capturing
uncertainty. We evaluate our approach on the Temple University Hospital EEG
Artifact Corpus (v2.0) achieving state-of-the-art classification results. In
addition, E4G provides well-calibrated uncertainty metrics comparable to
sampling techniques like Monte Carlo dropout in just a single forward pass. E4G
opens the door to uncertainty-aware artifact detection supporting
clinicians-in-the-loop frameworks.

    

### [[2107.10756] Semantic Text-to-Face GAN -ST^2FG](http://arxiv.org/abs/2107.10756)


  Faces generated using generative adversarial networks (GANs) have reached
unprecedented realism. These faces, also known as "Deep Fakes", appear as
realistic photographs with very little pixel-level distortions. While some work
has enabled the training of models that lead to the generation of specific
properties of the subject, generating a facial image based on a natural
language description has not been fully explored. For security and criminal
identification, the ability to provide a GAN-based system that works like a
sketch artist would be incredibly useful. In this paper, we present a novel
approach to generate facial images from semantic text descriptions. The learned
model is provided with a text description and an outline of the type of face,
which the model uses to sketch the features. Our models are trained using an
Affine Combination Module (ACM) mechanism to combine the text embedding from
BERT and the GAN latent space using a self-attention matrix. This avoids the
loss of features due to inadequate "attention", which may happen if text
embedding and latent vector are simply concatenated. Our approach is capable of
generating images that are very accurately aligned to the exhaustive textual
descriptions of faces with many fine detail features of the face and helps in
generating better images. The proposed method is also capable of making
incremental changes to a previously generated image if it is provided with
additional textual descriptions or sentences.

    

### [[2107.10763] Learning to Transfer: A Foliated Theory](http://arxiv.org/abs/2107.10763)


  Learning to transfer considers learning solutions to tasks in a such way that
relevant knowledge can be transferred from known task solutions to new, related
tasks. This is important for general learning, as well as for improving the
efficiency of the learning process. While techniques for learning to transfer
have been studied experimentally, we still lack a foundational description of
the problem that exposes what related tasks are, and how relationships between
tasks can be exploited constructively. In this work, we introduce a framework
using the differential geometric theory of foliations that provides such a
foundation.

    

### [[2107.10790] Interpretable SincNet-based Deep Learning for Emotion Recognition from EEG brain activity](http://arxiv.org/abs/2107.10790)


  Machine learning methods, such as deep learning, show promising results in
the medical domain. However, the lack of interpretability of these algorithms
may hinder their applicability to medical decision support systems. This paper
studies an interpretable deep learning technique, called SincNet. SincNet is a
convolutional neural network that efficiently learns customized band-pass
filters through trainable sinc-functions. In this study, we use SincNet to
analyze the neural activity of individuals with Autism Spectrum Disorder (ASD),
who experience characteristic differences in neural oscillatory activity. In
particular, we propose a novel SincNet-based neural network for detecting
emotions in ASD patients using EEG signals. The learned filters can be easily
inspected to detect which part of the EEG spectrum is used for predicting
emotions. We found that our system automatically learns the high-$\alpha$ (9-13
Hz) and $\beta$ (13-30 Hz) band suppression often present in individuals with
ASD. This result is consistent with recent neuroscience studies on emotion
recognition, which found an association between these band suppressions and the
behavioral deficits observed in individuals with ASD. The improved
interpretability of SincNet is achieved without sacrificing performance in
emotion recognition.

    

### [[2107.10804] Active Learning in Incomplete Label Multiple Instance Multiple Label Learning](http://arxiv.org/abs/2107.10804)


  In multiple instance multiple label learning, each sample, a bag, consists of
multiple instances. To alleviate labeling complexity, each sample is associated
with a set of bag-level labels leaving instances within the bag unlabeled. This
setting is more convenient and natural for representing complicated objects,
which have multiple semantic meanings. Compared to single instance labeling,
this approach allows for labeling larger datasets at an equivalent labeling
cost. However, for sufficiently large datasets, labeling all bags may become
prohibitively costly. Active learning uses an iterative labeling and retraining
approach aiming to provide reasonable classification performance using a small
number of labeled samples. To our knowledge, only a few works in the area of
active learning in the MIML setting are available. These approaches can provide
practical solutions to reduce labeling cost but their efficacy remains unclear.
In this paper, we propose a novel bag-class pair based approach for active
learning in the MIML setting. Due to the partial availability of bag-level
labels, we focus on the incomplete-label MIML setting for the proposed active
learning approach. Our approach is based on a discriminative graphical model
with efficient and exact inference. For the query process, we adapt active
learning criteria to the novel bag-class pair selection strategy. Additionally,
we introduce an online stochastic gradient descent algorithm to provide an
efficient model update after each query. Numerical experiments on benchmark
datasets illustrate the robustness of the proposed approach.

    

### [[2107.10845] QuantumNAS: Noise-Adaptive Search for Robust Quantum Circuits](http://arxiv.org/abs/2107.10845)


  Quantum noise is the key challenge in Noisy Intermediate-Scale Quantum (NISQ)
computers. Limited research efforts have explored a higher level of
optimization by making the quantum circuit resilient to noise. We propose and
experimentally implement QuantumNAS, the first comprehensive framework for
noise-adaptive co-search of variational circuit and qubit mapping. Variational
quantum circuits are a promising approach for constructing quantum neural
networks for machine learning and variational ansatzes for quantum simulation.
However, finding the best variational circuit and its optimal parameters is
challenging in a high-dimensional Hilbert space. We propose to decouple the
parameter training and circuit search by introducing a novel gate-sharing
SuperCircuit. The SuperCircuit is trained by sampling and updating the
SubCircuits in it and provides an accurate estimation of SubCircuit performance
trained from scratch. Then we perform an evolutionary co-search of SubCircuit
and its qubit mapping. The SubCircuit performance is estimated with parameters
inherited from SuperCircuit and simulated with real device noise models.
Finally, we perform iterative gate pruning and finetuning to further remove the
redundant gates in a fine-grained manner.
Extensively evaluated with 12 QML and VQE benchmarks on 10 quantum computers,
QuantumNAS significantly outperforms noise-unaware search, human and random
baselines. For QML tasks, QuantumNAS is the first to demonstrate over 95%
2-class, 85% 4-class, and 32% 10-class classification accuracy on real quantum
computers. It also achieves the lowest eigenvalue for VQE tasks on H2, H2O,
LiH, CH4, BeH2 compared with UCCSD baselines. We also open-source QuantumEngine
(this https URL) for fast training of
parameterized quantum circuits to facilitate future research.

    

### [[2107.10847] Accelerating Quadratic Optimization with Reinforcement Learning](http://arxiv.org/abs/2107.10847)


  First-order methods for quadratic optimization such as OSQP are widely used
for large-scale machine learning and embedded optimal control, where many
related problems must be rapidly solved. These methods face two persistent
challenges: manual hyperparameter tuning and convergence time to high-accuracy
solutions. To address these, we explore how Reinforcement Learning (RL) can
learn a policy to tune parameters to accelerate convergence. In experiments
with well-known QP benchmarks we find that our RL policy, RLQP, significantly
outperforms state-of-the-art QP solvers by up to 3x. RLQP generalizes
surprisingly well to previously unseen problems with varying dimension and
structure from different applications, including the QPLIB, Netlib LP and
Maros-Meszaros problems. Code for RLQP is available at
this https URL.

    

### [[1805.05510] Online Progressive Deep Metric Learning](http://arxiv.org/abs/1805.05510)


  Metric learning especially deep metric learning has been widely developed for
large-scale image inputs data. However, in many real-world applications, we can
only have access to vectorized inputs data. Moreover, on one hand, well-labeled
data is usually limited due to the high annotation cost. On the other hand, the
real data is commonly streaming data, which requires to be processed online. In
these scenarios, the fashionable deep metric learning is not suitable anymore.
To this end, we reconsider the traditional shallow online metric learning and
newly develop an online progressive deep metric learning (ODML) framework to
construct a metric-algorithm-based deep network. Specifically, we take an
online metric learning algorithm as a metric-algorithm-based layer (i.e.,
metric layer), followed by a nonlinear layer, and then stack these layers in a
fashion similar to deep learning. Different from the shallow online metric
learning, which can only learn one metric space (feature transformation), the
proposed ODML is able to learn multiple hierarchical metric spaces.
Furthermore, in a progressively and nonlinearly learning way, ODML has a
stronger learning ability than traditional shallow online metric learning in
the case of limited available training data. To make the learning process more
explainable and theoretically guaranteed, we also provide theoretical analysis.
The proposed ODML enjoys several nice properties and can indeed learn a metric
progressively and performs better on the benchmark datasets. Extensive
experiments with different settings have been conducted to verify these
properties of the proposed ODML.

    

### [[2001.03040] Deep Network Approximation for Smooth Functions](http://arxiv.org/abs/2001.03040)


  This paper establishes the optimal approximation error characterization of
deep ReLU networks for smooth functions in terms of both width and depth
simultaneously. To that end, we first prove that multivariate polynomials can
be approximated by deep ReLU networks of width $\mathcal{O}(N)$ and depth
$\mathcal{O}(L)$ with an approximation error $\mathcal{O}(N^{-L})$. Through
local Taylor expansions and their deep ReLU network approximations, we show
that deep ReLU networks of width $\mathcal{O}(N\ln N)$ and depth
$\mathcal{O}(L\ln L)$ can approximate $f\in C^s([0,1]^d)$ with a nearly optimal
approximation error $\mathcal{O}(\|f\|_{C^s([0,1]^d)}N^{-2s/d}L^{-2s/d})$. Our
estimate is non-asymptotic in the sense that it is valid for arbitrary width
and depth specified by $N\in\mathbb{N}^+$ and $L\in\mathbb{N}^+$, respectively.

    

### [[2004.05923] Adversarial Robustness Guarantees for Random Deep Neural Networks](http://arxiv.org/abs/2004.05923)


  The reliability of deep learning algorithms is fundamentally challenged by
the existence of adversarial examples, which are incorrectly classified inputs
that are extremely close to a correctly classified input. We explore the
properties of adversarial examples for deep neural networks with random weights
and biases, and prove that for any $p\ge1$, the $\ell^p$ distance of any given
input from the classification boundary scales as one over the square root of
the dimension of the input times the $\ell^p$ norm of the input. The results
are based on the recently proved equivalence between Gaussian processes and
deep neural networks in the limit of infinite width of the hidden layers, and
are validated with experiments on both random deep neural networks and deep
neural networks trained on the MNIST and CIFAR10 datasets. The results
constitute a fundamental advance in the theoretical understanding of
adversarial examples, and open the way to a thorough theoretical
characterization of the relation between network architecture and robustness to
adversarial perturbations.

    

### [[2006.05066] Deeply Shared Filter Bases for Parameter-Efficient Convolutional Neural Networks](http://arxiv.org/abs/2006.05066)


  Modern convolutional neural networks (CNNs) have massive identical
convolution blocks, and, hence, recursive sharing of parameters across these
blocks has been proposed to reduce the amount of parameters. However, naive
sharing of parameters poses many challenges such as limited representational
power and the vanishing/exploding gradients problem of recursively shared
parameters. In this paper, we present a recursive convolution block design and
training method, in which a recursively shareable part, or a filter basis, is
separated and learned while effectively avoiding the vanishing/exploding
gradients problem during training. We show that the unwieldy
vanishing/exploding gradients problem can be controlled by enforcing the
elements of the filter basis orthonormal, and empirically demonstrate that the
proposed orthogonality regularization improves the flow of gradients during
training. Experimental results on image classification and object detection
show that our approach, unlike previous parameter-sharing approaches, does not
trade performance to save parameters and consistently outperforms
overparameterized counterpart networks. This superior performance demonstrates
that the proposed recursive convolution block design and the orthogonality
regularization not only prevent performance degradation, but also consistently
improve the representation capability while a significant amount of parameters
are recursively shared.

    

### [[2006.05161] Provable tradeoffs in adversarially robust classification](http://arxiv.org/abs/2006.05161)


  It is well known that machine learning methods can be vulnerable to
adversarially-chosen perturbations of their inputs. Despite significant
progress in the area, foundational open problems remain. In this paper, we
address several key questions. We derive exact and approximate Bayes-optimal
robust classifiers for the important setting of two- and three-class Gaussian
classification problems with arbitrary imbalance, for $\ell_2$ and
$\ell_\infty$ adversaries. In contrast to classical Bayes-optimal classifiers,
determining the optimal decisions here cannot be made pointwise and new
theoretical approaches are needed. We develop and leverage new tools, including
recent breakthroughs from probability theory on robust isoperimetry, which, to
our knowledge, have not yet been used in the area. Our results reveal
fundamental tradeoffs between standard and robust accuracy that grow when data
is imbalanced. We also show further results, including an analysis of
classification calibration for convex losses in certain models, and finite
sample rates for the robust risk.

    

### [[2006.09503] Memory-Efficient Pipeline-Parallel DNN Training](http://arxiv.org/abs/2006.09503)


  Many state-of-the-art ML results have been obtained by scaling up the number
of parameters in existing models. However, parameters and activations for such
large models often do not fit in the memory of a single accelerator device;
this means that it is necessary to distribute training of large models over
multiple accelerators. In this work, we propose PipeDream-2BW, a system that
supports memory-efficient pipeline parallelism. PipeDream-2BW uses a novel
pipelining and weight gradient coalescing strategy, combined with the double
buffering of weights, to ensure high throughput, low memory footprint, and
weight update semantics similar to data parallelism. In addition, PipeDream-2BW
automatically partitions the model over the available hardware resources, while
respecting hardware constraints such as memory capacities of accelerators and
interconnect topologies. PipeDream-2BW can accelerate the training of large GPT
and BERT language models by up to 20$\times$ with similar final model accuracy.

    

### [[2007.00077] Similarity Search for Efficient Active Learning and Search of Rare Concepts](http://arxiv.org/abs/2007.00077)


  Many active learning and search approaches are intractable for large-scale
industrial settings with billions of unlabeled examples. Existing approaches
search globally for the optimal examples to label, scaling linearly or even
quadratically with the unlabeled data. In this paper, we improve the
computational efficiency of active learning and search methods by restricting
the candidate pool for labeling to the nearest neighbors of the currently
labeled set instead of scanning over all of the unlabeled data. We evaluate
several selection strategies in this setting on three large-scale computer
vision datasets: ImageNet, OpenImages, and a de-identified and aggregated
dataset of 10 billion images provided by a large internet company. Our approach
achieved similar mean average precision and recall as the traditional global
approach while reducing the computational cost of selection by up to three
orders of magnitude, thus enabling web-scale active learning.

    

### [[2007.00864] Hardware Acceleration of Sparse and Irregular Tensor Computations of ML Models: A Survey and Insights](http://arxiv.org/abs/2007.00864)


  Machine learning (ML) models are widely used in many important domains. For
efficiently processing these computational- and memory-intensive applications,
tensors of these over-parameterized models are compressed by leveraging
sparsity, size reduction, and quantization of tensors. Unstructured sparsity
and tensors with varying dimensions yield irregular computation, communication,
and memory access patterns; processing them on hardware accelerators in a
conventional manner does not inherently leverage acceleration opportunities.
This paper provides a comprehensive survey on the efficient execution of sparse
and irregular tensor computations of ML models on hardware accelerators. In
particular, it discusses enhancement modules in the architecture design and the
software support; categorizes different hardware designs and acceleration
techniques and analyzes them in terms of hardware and execution costs; analyzes
achievable accelerations for recent DNNs; highlights further opportunities in
terms of hardware/software/model co-design optimizations (inter/intra-module).
The takeaways from this paper include: understanding the key challenges in
accelerating sparse, irregular-shaped, and quantized tensors; understanding
enhancements in accelerator systems for supporting their efficient
computations; analyzing trade-offs in opting for a specific design choice for
encoding, storing, extracting, communicating, computing, and load-balancing the
non-zeros; understanding how structured sparsity can improve storage efficiency
and balance computations; understanding how to compile and map models with
sparse tensors on the accelerators; understanding recent design trends for
efficient accelerations and further opportunities.

    

### [[2007.03481] Necessary and Sufficient Conditions for Inverse Reinforcement Learning of Bayesian Stopping Time Problems](http://arxiv.org/abs/2007.03481)


  This paper presents an inverse reinforcement learning (IRL) framework for
Bayesian stopping time problems. By observing the actions of a Bayesian
decision maker, we provide a necessary and sufficient condition to identify if
these actions are consistent with optimizing a cost function; then we construct
set valued estimates of the cost function. To achieve this IRL objective, we
use novel ideas from Bayesian revealed preferences stemming from
microeconomics. To illustrate our IRL scheme,we consider two important examples
of stopping time problems, namely, sequential hypothesis testing and Bayesian
search. Finally, for finite datasets, we propose an IRL detection algorithm and
give finite sample bounds on its error probabilities. Also we discuss how to
identify $\epsilon$-optimal Bayesian decision makers and perform IRL.

    

### [[2007.09236] Multi-Task Federated Learning for Personalised Deep Neural Networks in Edge Computing](http://arxiv.org/abs/2007.09236)


  Federated Learning (FL) is an emerging approach for collaboratively training
Deep Neural Networks (DNNs) on mobile devices, without private user data
leaving the devices. Previous works have shown that non-Independent and
Identically Distributed (non-IID) user data harms the convergence speed of the
FL algorithms. Furthermore, most existing work on FL measures global-model
accuracy, but in many cases, such as user content-recommendation, improving
individual User model Accuracy (UA) is the real objective. To address these
issues, we propose a Multi-Task FL (MTFL) algorithm that introduces
non-federated Batch-Normalization (BN) layers into the federated DNN. MTFL
benefits UA and convergence speed by allowing users to train models
personalised to their own data. MTFL is compatible with popular iterative FL
optimisation algorithms such as Federated Averaging (FedAvg), and we show
empirically that a distributed form of Adam optimisation (FedAvg-Adam) benefits
convergence speed even further when used as the optimisation strategy within
MTFL. Experiments using MNIST and CIFAR10 demonstrate that MTFL is able to
significantly reduce the number of rounds required to reach a target UA, by up
to $5\times$ when using existing FL optimisation strategies, and with a further
$3\times$ improvement when using FedAvg-Adam. We compare MTFL to competing
personalised FL algorithms, showing that it is able to achieve the best UA for
MNIST and CIFAR10 in all considered scenarios. Finally, we evaluate MTFL with
FedAvg-Adam on an edge-computing testbed, showing that its convergence and UA
benefits outweigh its overhead.

    

### [[2008.01559] Adversarial Radar Inference: Inverse Tracking, Identifying Cognition and Designing Smart Interference](http://arxiv.org/abs/2008.01559)


  This paper considers three inter-related adversarial inference problems
involving cognitive radars. We first discuss inverse tracking of the radar to
estimate the adversary's estimate of us based on the radar's actions and
calibrate the radar's sensing accuracy. Second, using revealed preference from
microeconomics, we formulate a non-parametric test to identify if the cognitive
radar is a constrained utility maximizer with signal processing constraints. We
consider two radar functionalities, namely, beam allocation and waveform
design, with respect to which the cognitive radar is assumed to maximize its
utility and construct a set-valued estimator for the radar's utility function.
Finally, we discuss how to engineer interference at the physical layer level to
confuse the radar which forces it to change its transmit waveform. The levels
of abstraction range from smart interference design based on Wiener filters (at
the pulse/waveform level), inverse Kalman filters at the tracking level and
revealed preferences for identifying utility maximization at the systems level.

    

### [[2008.03592] Speech Driven Talking Face Generation from a Single Image and an Emotion Condition](http://arxiv.org/abs/2008.03592)


  Visual emotion expression plays an important role in audiovisual speech
communication. In this work, we propose a novel approach to rendering visual
emotion expression in speech-driven talking face generation. Specifically, we
design an end-to-end talking face generation system that takes a speech
utterance, a single face image, and a categorical emotion label as input to
render a talking face video synchronized with the speech and expressing the
conditioned emotion. Objective evaluation on image quality, audiovisual
synchronization, and visual emotion expression shows that the proposed system
outperforms a state-of-the-art baseline system. Subjective evaluation of visual
emotion expression and video realness also demonstrates the superiority of the
proposed system. Furthermore, we conduct a human emotion recognition pilot
study using generated videos with mismatched emotions among the audio and
visual modalities. Results show that humans respond to the visual modality more
significantly than the audio modality on this task.

    

### [[2008.05533] Overcoming Model Bias for Robust Offline Deep Reinforcement Learning](http://arxiv.org/abs/2008.05533)


  State-of-the-art reinforcement learning algorithms mostly rely on being
allowed to directly interact with their environment to collect millions of
observations. This makes it hard to transfer their success to industrial
control problems, where simulations are often very costly or do not exist, and
exploring in the real environment can potentially lead to catastrophic events.
Recently developed, model-free, offline RL algorithms, can learn from a single
dataset (containing limited exploration) by mitigating extrapolation error in
value functions. However, the robustness of the training process is still
comparatively low, a problem known from methods using value functions. To
improve robustness and stability of the learning process, we use dynamics
models to assess policy performance instead of value functions, resulting in
MOOSE (MOdel-based Offline policy Search with Ensembles), an algorithm which
ensures low model bias by keeping the policy within the support of the data. We
compare MOOSE with state-of-the-art model-free, offline RL algorithms { BRAC,}
BEAR and BCQ on the Industrial Benchmark and MuJoCo continuous control tasks in
terms of robust performance, and find that MOOSE outperforms its model-free
counterparts in almost all considered cases, often even by far.

    

### [[2008.07298] WAFFLE: Watermarking in Federated Learning](http://arxiv.org/abs/2008.07298)


  Federated learning is a distributed learning technique where machine learning
models are trained on client devices in which the local training data resides.
The training is coordinated via a central server which is, typically,
controlled by the intended owner of the resulting model. By avoiding the need
to transport the training data to the central server, federated learning
improves privacy and efficiency. But it raises the risk of model theft by
clients because the resulting model is available on every client device. Even
if the application software used for local training may attempt to prevent
direct access to the model, a malicious client may bypass any such restrictions
by reverse engineering the application software. Watermarking is a well-known
deterrence method against model theft by providing the means for model owners
to demonstrate ownership of their models. Several recent deep neural network
(DNN) watermarking techniques use backdooring: training the models with
additional mislabeled data. Backdooring requires full access to the training
data and control of the training process. This is feasible when a single party
trains the model in a centralized manner, but not in a federated learning
setting where the training process and training data are distributed among
several client devices. In this paper, we present WAFFLE, the first approach to
watermark DNN models trained using federated learning. It introduces a
retraining step at the server after each aggregation of local models into the
global model. We show that WAFFLE efficiently embeds a resilient watermark into
models incurring only negligible degradation in test accuracy (-0.17%), and
does not require access to training data. We also introduce a novel technique
to generate the backdoor used as a watermark. It outperforms prior techniques,
imposing no communication, and low computational (+3.2%) overhead.

    

### [[2009.07052] Demand Forecasting of Individual Probability Density Functions with Machine Learning](http://arxiv.org/abs/2009.07052)


  Demand forecasting is a central component of the replenishment process for
retailers, as it provides crucial input for subsequent decision making like
ordering processes. In contrast to point estimates, such as the conditional
mean of the underlying probability distribution, or confidence intervals,
forecasting complete probability density functions allows to investigate the
impact on operational metrics, which are important to define the business
strategy, over the full range of the expected demand. Whereas metrics
evaluating point estimates are widely used, methods for assessing the accuracy
of predicted distributions are rare, and this work proposes new techniques for
both qualitative and quantitative evaluation methods. Using the supervised
machine learning method "Cyclic Boosting", complete individual probability
density functions can be predicted such that each prediction is fully
explainable. This is of particular importance for practitioners, as it allows
to avoid "black-box" models and understand the contributing factors for each
individual prediction. Another crucial aspect in terms of both explainability
and generalizability of demand forecasting methods is the limitation of the
influence of temporal confounding, which is prevalent in most state of the art
approaches.

    

### [[2010.07093] Function Contrastive Learning of Transferable Meta-Representations](http://arxiv.org/abs/2010.07093)


  Meta-learning algorithms adapt quickly to new tasks that are drawn from the
same task distribution as the training tasks. The mechanism leading to fast
adaptation is the conditioning of a downstream predictive model on the inferred
representation of the task's underlying data generative process, or
\emph{function}. This \emph{meta-representation}, which is computed from a few
observed examples of the underlying function, is learned jointly with the
predictive model. In this work, we study the implications of this joint
training on the transferability of the meta-representations. Our goal is to
learn meta-representations that are robust to noise in the data and facilitate
solving a wide range of downstream tasks that share the same underlying
functions. To this end, we propose a decoupled encoder-decoder approach to
supervised meta-learning, where the encoder is trained with a contrastive
objective to find a good representation of the underlying function. In
particular, our training scheme is driven by the self-supervision signal
indicating whether two sets of examples stem from the same function. Our
experiments on a number of synthetic and real-world datasets show that the
representations we obtain outperform strong baselines in terms of downstream
performance and noise robustness, even when these baselines are trained in an
end-to-end manner.

    

### [[2010.08488] The Ridgelet Prior: A Covariance Function Approach to Prior Specification for Bayesian Neural Networks](http://arxiv.org/abs/2010.08488)


  Bayesian neural networks attempt to combine the strong predictive performance
of neural networks with formal quantification of uncertainty associated with
the predictive output in the Bayesian framework. However, it remains unclear
how to endow the parameters of the network with a prior distribution that is
meaningful when lifted into the output space of the network. A possible
solution is proposed that enables the user to posit an appropriate Gaussian
process covariance function for the task at hand. Our approach constructs a
prior distribution for the parameters of the network, called a ridgelet prior,
that approximates the posited Gaussian process in the output space of the
network. In contrast to existing work on the connection between neural networks
and Gaussian processes, our analysis is non-asymptotic, with finite sample-size
error bounds provided. This establishes the universality property that a
Bayesian neural network can approximate any Gaussian process whose covariance
function is sufficiently regular. Our experimental assessment is limited to a
proof-of-concept, where we demonstrate that the ridgelet prior can out-perform
an unstructured prior on regression problems for which a suitable Gaussian
process prior can be provided.

    

### [[2010.13520] Differentially Private (Gradient) Expectation Maximization Algorithm with Statistical Guarantees](http://arxiv.org/abs/2010.13520)


  (Gradient) Expectation Maximization (EM) is a widely used algorithm for
estimating the maximum likelihood of mixture models or incomplete data
problems. A major challenge facing this popular technique is how to effectively
preserve the privacy of sensitive data. Previous research on this problem has
already lead to the discovery of some Differentially Private (DP) algorithms
for (Gradient) EM. However, unlike in the non-private case, existing techniques
are not yet able to provide finite sample statistical guarantees. To address
this issue, we propose in this paper the first DP version of (Gradient) EM
algorithm with statistical guarantees. Moreover, we apply our general framework
to three canonical models: Gaussian Mixture Model (GMM), Mixture of Regressions
Model (MRM) and Linear Regression with Missing Covariates (RMC). Specifically,
for GMM in the DP model, our estimation error is near optimal in some cases.
For the other two models, we provide the first finite sample statistical
guarantees. Our theory is supported by thorough numerical experiments.

    

### [[2010.14102] Emotion recognition by fusing time synchronous and time asynchronous representations](http://arxiv.org/abs/2010.14102)


  In this paper, a novel two-branch neural network model structure is proposed
for multimodal emotion recognition, which consists of a time synchronous branch
(TSB) and a time asynchronous branch (TAB). To capture correlations between
each word and its acoustic realisation, the TSB combines speech and text
modalities at each input window frame and then does pooling across time to form
a single embedding vector. The TAB, by contrast, provides cross-utterance
information by integrating sentence text embeddings from a number of context
utterances into another embedding vector. The final emotion classification uses
both the TSB and the TAB embeddings. Experimental results on the IEMOCAP
dataset demonstrate that the two-branch structure achieves state-of-the-art
results in 4-way classification with all common test setups. When using
automatic speech recognition (ASR) output instead of manually transcribed
reference text, it is shown that the cross-utterance information considerably
improves the robustness against ASR errors. Furthermore, by incorporating an
extra class for all the other emotions, the final 5-way classification system
with ASR hypotheses can be viewed as a prototype for more realistic emotion
recognition systems.

    

### [[2010.16051] Interpretable Machine Learning Models for Predicting and Explaining Vehicle Fuel Consumption Anomalies](http://arxiv.org/abs/2010.16051)


  Identifying anomalies in the fuel consumption of the vehicles of a fleet is a
crucial aspect for optimizing consumption and reduce costs. However, this
information alone is insufficient, since fleet operators need to know the
causes behind anomalous fuel consumption. We combine unsupervised anomaly
detection techniques, domain knowledge and interpretable Machine Learning
models for explaining potential causes of abnormal fuel consumption in terms of
feature relevance. The explanations are used for generating recommendations
about fuel optimization, that are adjusted according to two different user
profiles: fleet managers and fleet operators. Results are evaluated over
real-world data from telematics devices connected to diesel and petrol vehicles
from different types of industrial fleets. We measure the proposal regarding
model performance, and using Explainable AI metrics that compare the
explanations in terms of representativeness, fidelity, stability,
contrastiveness and consistency with apriori beliefs. The potential fuel
reductions that can be achieved is round 35%.

    

### [[2011.04144] Near-Optimal Learning of Tree-Structured Distributions by Chow-Liu](http://arxiv.org/abs/2011.04144)


  We provide finite sample guarantees for the classical Chow-Liu algorithm
(IEEE Trans.~Inform.~Theory, 1968) to learn a tree-structured graphical model
of a distribution. For a distribution $P$ on $\Sigma^n$ and a tree $T$ on $n$
nodes, we say $T$ is an $\varepsilon$-approximate tree for $P$ if there is a
$T$-structured distribution $Q$ such that $D(P\;||\;Q)$ is at most
$\varepsilon$ more than the best possible tree-structured distribution for $P$.
We show that if $P$ itself is tree-structured, then the Chow-Liu algorithm with
the plug-in estimator for mutual information with $\widetilde{O}(|\Sigma|^3
n\varepsilon^{-1})$ i.i.d.~samples outputs an $\varepsilon$-approximate tree
for $P$ with constant probability. In contrast, for a general $P$ (which may
not be tree-structured), $\Omega(n^2\varepsilon^{-2})$ samples are necessary to
find an $\varepsilon$-approximate tree. Our upper bound is based on a new
conditional independence tester that addresses an open problem posed by
Canonne, Diakonikolas, Kane, and Stewart~(STOC, 2018): we prove that for three
random variables $X,Y,Z$ each over $\Sigma$, testing if $I(X; Y \mid Z)$ is $0$
or $\geq \varepsilon$ is possible with $\widetilde{O}(|\Sigma|^3/\varepsilon)$
samples. Finally, we show that for a specific tree $T$, with $\widetilde{O}
(|\Sigma|^2n\varepsilon^{-1})$ samples from a distribution $P$ over $\Sigma^n$,
one can efficiently learn the closest $T$-structured distribution in KL
divergence by applying the add-1 estimator at each node.

    

### [[2012.02177] DeepVideoMVS: Multi-View Stereo on Video with Recurrent Spatio-Temporal Fusion](http://arxiv.org/abs/2012.02177)


  We propose an online multi-view depth prediction approach on posed video
streams, where the scene geometry information computed in the previous time
steps is propagated to the current time step in an efficient and geometrically
plausible way. The backbone of our approach is a real-time capable, lightweight
encoder-decoder that relies on cost volumes computed from pairs of images. We
extend it by placing a ConvLSTM cell at the bottleneck layer, which compresses
an arbitrary amount of past information in its states. The novelty lies in
propagating the hidden state of the cell by accounting for the viewpoint
changes between time steps. At a given time step, we warp the previous hidden
state into the current camera plane using the previous depth prediction. Our
extension brings only a small overhead of computation time and memory
consumption, while improving the depth predictions significantly. As a result,
we outperform the existing state-of-the-art multi-view stereo methods on most
of the evaluated metrics in hundreds of indoor scenes while maintaining a
real-time performance. Code available:
this https URL


### [[2012.06421] When is Memorization of Irrelevant Training Data Necessary for High-Accuracy Learning?](http://arxiv.org/abs/2012.06421)


  Modern machine learning models are complex and frequently encode surprising
amounts of information about individual inputs. In extreme cases, complex
models appear to memorize entire input examples, including seemingly irrelevant
information (social security numbers from text, for example). In this paper, we
aim to understand whether this sort of memorization is necessary for accurate
learning. We describe natural prediction problems in which every sufficiently
accurate training algorithm must encode, in the prediction model, essentially
all the information about a large subset of its training examples. This remains
true even when the examples are high-dimensional and have entropy much higher
than the sample size, and even when most of that information is ultimately
irrelevant to the task at hand. Further, our results do not depend on the
training algorithm or the class of models used for learning.
Our problems are simple and fairly natural variants of the next-symbol
prediction and the cluster labeling tasks. These tasks can be seen as
abstractions of text- and image-related prediction problems. To establish our
results, we reduce from a family of one-way communication problems for which we
prove new information complexity lower bounds. Additionally, we present
synthetic-data experiments demonstrating successful attacks on logistic
regression and neural network classifiers.

    

### [[2012.09643] Automatic source localization and spectra generation from sparse beamforming maps](http://arxiv.org/abs/2012.09643)


  Beamforming is an imaging tool for the investigation of aeroacoustic
phenomena and results in high dimensional data that is broken down to spectra
by integrating spatial Regions Of Interest. This paper presents two methods
that enable the automated identification of aeroacoustic sources in sparse
beamforming maps and the extraction of their corresponding spectra to overcome
the manual definition of Regions Of Interest. The methods are evaluated on two
scaled airframe half-model wind-tunnel measurements and on a generic monopole
source. The first relies on the spatial normal distribution of aeroacoustic
broadband sources in sparse beamforming maps. The second uses hierarchical
clustering methods. Both methods are robust to statistical noise and predict
the existence, location, and spatial probability estimation for sources based
on which Regions Of Interest are automatically determined.

    

### [[2012.11330] A Bayesian multiscale CNN framework to predict local stress fields in structures with microscale features](http://arxiv.org/abs/2012.11330)


  Multiscale computational modelling is challenging due to the high
computational cost of direct numerical simulation by finite elements. To
address this issue, concurrent multiscale methods use the solution of cheaper
macroscale surrogates as boundary conditions to microscale sliding windows. The
microscale problems remain a numerically challenging operation both in terms of
implementation and cost. In this work we propose to replace the local
microscale solution by an Encoder-Decoder Convolutional Neural Network that
will generate fine-scale stress corrections to coarse predictions around
unresolved microscale features, without prior parametrisation of local
microscale problems. We deploy a Bayesian approach providing credible intervals
to evaluate the uncertainty of the predictions, which is then used to
investigate the merits of a selective learning framework. We will demonstrate
the capability of the approach to predict equivalent stress fields in porous
structures using linearised and finite strain elasticity theories.

    

### [[2101.00554] Global field reconstruction from sparse sensors with Voronoi tessellation-assisted deep learning](http://arxiv.org/abs/2101.00554)


  Achieving accurate and robust global situational awareness of a complex
time-evolving field from a limited number of sensors has been a longstanding
challenge. This reconstruction problem is especially difficult when sensors are
sparsely positioned in a seemingly random or unorganized manner, which is often
encountered in a range of scientific and engineering problems. Moreover, these
sensors can be in motion and can become online or offline over time. The key
leverage in addressing this scientific issue is the wealth of data accumulated
from the sensors. As a solution to this problem, we propose a data-driven
spatial field recovery technique founded on a structured grid-based
deep-learning approach for arbitrary positioned sensors of any numbers. It
should be noted that the naïve use of machine learning becomes prohibitively
expensive for global field reconstruction and is furthermore not adaptable to
an arbitrary number of sensors. In the present work, we consider the use of
Voronoi tessellation to obtain a structured-grid representation from sensor
locations enabling the computationally tractable use of convolutional neural
networks. One of the central features of the present method is its
compatibility with deep-learning based super-resolution reconstruction
techniques for structured sensor data that are established for image
processing. The proposed reconstruction technique is demonstrated for unsteady
wake flow, geophysical data, and three-dimensional turbulence. The current
framework is able to handle an arbitrary number of moving sensors, and thereby
overcomes a major limitation with existing reconstruction methods. The
presented technique opens a new pathway towards the practical use of neural
networks for real-time global field estimation.

    

### [[2101.01229] A Survey on Embedding Dynamic Graphs](http://arxiv.org/abs/2101.01229)


  Embedding static graphs in low-dimensional vector spaces plays a key role in
network analytics and inference, supporting applications like node
classification, link prediction, and graph visualization. However, many
real-world networks present dynamic behavior, including topological evolution,
feature evolution, and diffusion. Therefore, several methods for embedding
dynamic graphs have been proposed to learn network representations over time,
facing novel challenges, such as time-domain modeling, temporal features to be
captured, and the temporal granularity to be embedded. In this survey, we
overview dynamic graph embedding, discussing its fundamentals and the recent
advances developed so far. We introduce the formal definition of dynamic graph
embedding, focusing on the problem setting and introducing a novel taxonomy for
dynamic graph embedding input and output. We further explore different dynamic
behaviors that may be encompassed by embeddings, classifying by topological
evolution, feature evolution, and processes on networks. Afterward, we describe
existing techniques and propose a taxonomy for dynamic graph embedding
techniques based on algorithmic approaches, from matrix and tensor
factorization to deep learning, random walks, and temporal point processes. We
also elucidate main applications, including dynamic link prediction, anomaly
detection, and diffusion prediction, and we further state some promising
research directions in the area.

    

### [[2101.04470] Type4Py: Deep Similarity Learning-Based Type Inference for Python](http://arxiv.org/abs/2101.04470)


  Dynamic languages, such as Python and Javascript, trade static typing for
developer flexibility and productivity. Lack of static typing can cause
run-time exceptions and is a major factor for weak IDE support. To alleviate
these issues, PEP 484 introduced optional type annotations for Python. As
retrofitting types to existing codebases is error-prone and laborious,
learning-based approaches have been proposed to enable automatic type
annotations based on existing, partially annotated codebases. However, it is
still quite challenging for learning-based approaches to give a relevant
prediction in the first suggestion or the first few ones. In this paper, we
present Type4Py, a deep similarity learning-based hierarchical neural network
model that learns to discriminate between types of the same kind and dissimilar
types in a high-dimensional space, which results in clusters of types. Nearest
neighbor search suggests a list of likely types for arguments, variables, and
functions' return. The results of the quantitative and qualitative evaluation
indicate that Type4Py significantly outperforms state-of-the-art approaches at
the type prediction task. Considering the Top-1 prediction, Type4Py obtains a
Mean Reciprocal Rank of 72.5%, which is 10.87% and 16.45% higher than that of
Typilus and TypeWriter, respectively.

    

### [[2101.05145] Self-Supervised Vessel Enhancement Using Flow-Based Consistencies](http://arxiv.org/abs/2101.05145)


  Vessel segmentation is an essential task in many clinical applications.
Although supervised methods have achieved state-of-art performance, acquiring
expert annotation is laborious and mostly limited for two-dimensional datasets
with a small sample size. On the contrary, unsupervised methods rely on
handcrafted features to detect tube-like structures such as vessels. However,
those methods require complex pipelines involving several hyper-parameters and
design choices rendering the procedure sensitive, dataset-specific, and not
generalizable. We propose a self-supervised method with a limited number of
hyper-parameters that is generalizable across modalities. Our method uses
tube-like structure properties, such as connectivity, profile consistency, and
bifurcation, to introduce inductive bias into a learning algorithm. To model
those properties, we generate a vector field that we refer to as a flow. Our
experiments on various public datasets in 2D and 3D show that our method
performs better than unsupervised methods while learning useful transferable
features from unlabeled data. Unlike generic self-supervised methods, the
learned features learn vessel-relevant features that are transferable for
supervised approaches, which is essential when the number of annotated data is
limited.

    

### [[2101.06536] Deep Cox Mixtures for Survival Regression](http://arxiv.org/abs/2101.06536)


  Survival analysis is a challenging variation of regression modeling because
of the presence of censoring, where the outcome measurement is only partially
known, due to, for example, loss to follow up. Such problems come up frequently
in medical applications, making survival analysis a key endeavor in
biostatistics and machine learning for healthcare, with Cox regression models
being amongst the most commonly employed models. We describe a new approach for
survival analysis regression models, based on learning mixtures of Cox
regressions to model individual survival distributions. We propose an
approximation to the Expectation Maximization algorithm for this model that
does hard assignments to mixture groups to make optimization efficient. In each
group assignment, we fit the hazard ratios within each group using deep neural
networks, and the baseline hazard for each mixture component
non-parametrically.
We perform experiments on multiple real world datasets, and look at the
mortality rates of patients across ethnicity and gender. We emphasize the
importance of calibration in healthcare settings and demonstrate that our
approach outperforms classical and modern survival analysis baselines, both in
terms of discriminative performance and calibration, with large gains in
performance on the minority demographics.

    

### [[2101.12015] BERTaú: Itaú BERT for digital customer service](http://arxiv.org/abs/2101.12015)


  In the last few years, three major topics received increased interest: deep
learning, NLP and conversational agents. Bringing these three topics together
to create an amazing digital customer experience and indeed deploy in
production and solve real-world problems is something innovative and
disruptive. We introduce a new Portuguese financial domain language
representation model called BERTaú. BERTaú is an uncased BERT-base trained
from scratch with data from the Itaú virtual assistant chatbot solution. Our
novel contribution is that BERTaú pretrained language model requires less
data, reached state-of-the-art performance in three NLP tasks, and generates a
smaller and lighter model that makes the deployment feasible. We developed
three tasks to validate our model: information retrieval with Frequently Asked
Questions (FAQ) from Itaú bank, sentiment analysis from our virtual assistant
data, and a NER solution. All proposed tasks are real-world solutions in
production on our environment and the usage of a specialist model proved to be
effective when compared to Google BERT multilingual and the DPRQuestionEncoder
from Facebook, available at Hugging Face. The BERTaú improves the performance
in 22% of FAQ Retrieval MRR metric, 2.1% in Sentiment Analysis F1 score, 4.4%
in NER F1 score and can also represent the same sequence in up to 66% fewer
tokens when compared to "shelf models".

    

### [[2102.00753] Quantum Fair Machine Learning](http://arxiv.org/abs/2102.00753)


  In this paper, we inaugurate the field of quantum fair machine learning. We
undertake a comparative analysis of differences and similarities between
classical and quantum fair machine learning algorithms, specifying how the
unique features of quantum computation alter measures, metrics and remediation
strategies when quantum algorithms are subject to fairness constraints. We
present the first results in quantum fair machine learning by demonstrating the
use of Grover's search algorithm to satisfy statistical parity constraints
imposed on quantum algorithms. We provide lower-bounds on iterations needed to
achieve such statistical parity within $\epsilon$-tolerance. We extend
canonical Lipschitz-conditioned individual fairness criteria to the quantum
setting using quantum metrics. We examine the consequences for typical measures
of fairness in machine learning context when quantum information processing and
quantum data are involved. Finally, we propose open questions and research
programmes for this new field of interest to researchers in computer science,
ethics and quantum computation.

    

### [[2102.02079] Federated Learning on Non-IID Data Silos: An Experimental Study](http://arxiv.org/abs/2102.02079)


  Due to the increasing privacy concerns and data regulations, training data
have been increasingly fragmented, forming distributed databases of multiple
``data silos'' (e.g., within different organizations and countries). To develop
effective machine learning services, there is a must to exploit data from such
distributed databases without exchanging the raw data. Recently, federated
learning (FL) has been a solution with growing interests, which enables
multiple parties to collaboratively train a machine learning model without
exchanging their local data. A key and common challenge on distributed
databases is the heterogeneity of the data distribution (i.e., non-IID) among
the parties. There have been many FL algorithms to address the learning
effectiveness under non-IID data settings. However, there lacks an experimental
study on systematically understanding their advantages and disadvantages, as
previous studies have very rigid data partitioning strategies among parties,
which are hardly representative and thorough. In this paper, to help
researchers better understand and study the non-IID data setting in federated
learning, we propose comprehensive data partitioning strategies to cover the
typical non-IID data cases. Moreover, we conduct extensive experiments to
evaluate state-of-the-art FL algorithms. We find that non-IID does bring
significant challenges in learning accuracy of FL algorithms, and none of the
existing state-of-the-art FL algorithms outperforms others in all cases. Our
experiments provide insights for future studies of addressing the challenges in
``data silos''.

    

### [[2102.07870] Momentum Residual Neural Networks](http://arxiv.org/abs/2102.07870)


  The training of deep residual neural networks (ResNets) with backpropagation
has a memory cost that increases linearly with respect to the depth of the
network. A way to circumvent this issue is to use reversible architectures. In
this paper, we propose to change the forward rule of a ResNet by adding a
momentum term. The resulting networks, momentum residual neural networks
(Momentum ResNets), are invertible. Unlike previous invertible architectures,
they can be used as a drop-in replacement for any existing ResNet block. We
show that Momentum ResNets can be interpreted in the infinitesimal step size
regime as second-order ordinary differential equations (ODEs) and exactly
characterize how adding momentum progressively increases the representation
capabilities of Momentum ResNets. Our analysis reveals that Momentum ResNets
can learn any linear mapping up to a multiplicative factor, while ResNets
cannot. In a learning to optimize setting, where convergence to a fixed point
is required, we show theoretically and empirically that our method succeeds
while existing invertible architectures fail. We show on CIFAR and ImageNet
that Momentum ResNets have the same accuracy as ResNets, while having a much
smaller memory footprint, and show that pre-trained Momentum ResNets are
promising for fine-tuning models.

    

### [[2102.10562] Tractable Computation of Expected Kernels](http://arxiv.org/abs/2102.10562)


  Computing the expectation of kernel functions is a ubiquitous task in machine
learning, with applications from classical support vector machines to
exploiting kernel embeddings of distributions in probabilistic modeling,
statistical inference, causal discovery, and deep learning. In all these
scenarios, we tend to resort to Monte Carlo estimates as expectations of
kernels are intractable in general. In this work, we characterize the
conditions under which we can compute expected kernels exactly and efficiently,
by leveraging recent advances in probabilistic circuit representations. We
first construct a circuit representation for kernels and propose an approach to
such tractable computation. We then demonstrate possible advancements for
kernel embedding frameworks by exploiting tractable expected kernels to derive
new algorithms for two challenging scenarios: 1) reasoning under missing data
with kernel support vector regressors; 2) devising a collapsed black-box
importance sampling scheme. Finally, we empirically evaluate both algorithms
and show that they outperform standard baselines on a variety of datasets.

    

### [[2102.12463] Generating and Blending Game Levels via Quality-Diversity in the Latent Space of a Variational Autoencoder](http://arxiv.org/abs/2102.12463)


  Several works have demonstrated the use of variational autoencoders (VAEs)
for generating levels in the style of existing games and blending levels across
different games. Further, quality-diversity (QD) algorithms have also become
popular for generating varied game content by using evolution to explore a
search space while focusing on both variety and quality. To reap the benefits
of both these approaches, we present a level generation and game blending
approach that combines the use of VAEs and QD algorithms. Specifically, we
train VAEs on game levels and run the MAP-Elites QD algorithm using the learned
latent space of the VAE as the search space. The latent space captures the
properties of the games whose levels we want to generate and blend, while
MAP-Elites searches this latent space to find a diverse set of levels
optimizing a given objective such as playability. We test our method using
models for 5 different platformer games as well as a blended domain spanning 3
of these games. We refer to using MAP-Elites for blending as Blend-Elites. Our
results show that MAP-Elites in conjunction with VAEs enables the generation of
a diverse set of playable levels not just for each individual game but also for
the blended domain while illuminating game-specific regions of the blended
latent space.

    

### [[2103.02354] Evaluating Robustness of Counterfactual Explanations](http://arxiv.org/abs/2103.02354)


  Transparency is a fundamental requirement for decision making systems when
these should be deployed in the real world. It is usually achieved by providing
explanations of the system's behavior. A prominent and intuitive type of
explanations are counterfactual explanations. Counterfactual explanations
explain a behavior to the user by proposing actions -- as changes to the input
-- that would cause a different (specified) behavior of the system. However,
such explanation methods can be unstable with respect to small changes to the
input -- i.e. even a small change in the input can lead to huge or arbitrary
changes in the output and of the explanation. This could be problematic for
counterfactual explanations, as two similar individuals might get very
different explanations. Even worse, if the recommended actions differ
considerably in their complexity, one would consider such unstable
(counterfactual) explanations as individually unfair.
In this work, we formally and empirically study the robustness of
counterfactual explanations in general, as well as under different models and
different kinds of perturbations. Furthermore, we propose that plausible
counterfactual explanations can be used instead of closest counterfactual
explanations to improve the robustness and consequently the individual fairness
of counterfactual explanations.

    

### [[2103.03279] Remember What You Want to Forget: Algorithms for Machine Unlearning](http://arxiv.org/abs/2103.03279)


  We study the problem of unlearning datapoints from a learnt model. The
learner first receives a dataset $S$ drawn i.i.d. from an unknown distribution,
and outputs a model $\widehat{w}$ that performs well on unseen samples from the
same distribution. However, at some point in the future, any training datapoint
$z \in S$ can request to be unlearned, thus prompting the learner to modify its
output model while still ensuring the same accuracy guarantees. We initiate a
rigorous study of generalization in machine unlearning, where the goal is to
perform well on previously unseen datapoints. Our focus is on both
computational and storage complexity.
For the setting of convex losses, we provide an unlearning algorithm that can
unlearn up to $O(n/d^{1/4})$ samples, where $d$ is the problem dimension. In
comparison, in general, differentially private learning (which implies
unlearning) only guarantees deletion of $O(n/d^{1/2})$ samples. This
demonstrates a novel separation between differential privacy and machine
unlearning.

    

### [[2103.03862] Harnessing Geometric Constraints from Emotion Labels to improve Face Verification](http://arxiv.org/abs/2103.03862)


  For the task of face verification, we explore the utility of harnessing
auxiliary facial emotion labels to impose explicit geometric constraints on the
embedding space when training deep embedding models. We introduce several novel
loss functions that, in conjunction with a standard Triplet Loss [43], or
ArcFace loss [10], provide geometric constraints on the embedding space; the
labels for our loss functions can be provided using either manually annotated
or automatically detected auxiliary emotion labels. Our method is implemented
purely in terms of the loss function and does not require any changes to the
neural network backbone of the embedding function.

    

### [[2103.07470] Understanding Invariance via Feedforward Inversion of Discriminatively Trained Classifiers](http://arxiv.org/abs/2103.07470)


  A discriminatively trained neural net classifier can fit the training data
perfectly if all information about its input other than class membership has
been discarded prior to the output layer. Surprisingly, past research has
discovered that some extraneous visual detail remains in the logit vector. This
finding is based on inversion techniques that map deep embeddings back to
images. We explore this phenomenon further using a novel synthesis of methods,
yielding a feedforward inversion model that produces remarkably high fidelity
reconstructions, qualitatively superior to those of past efforts. When applied
to an adversarially robust classifier model, the reconstructions contain
sufficient local detail and global structure that they might be confused with
the original image in a quick glance, and the object category can clearly be
gleaned from the reconstruction. Our approach is based on BigGAN (Brock, 2019),
with conditioning on logits instead of one-hot class labels. We use our
reconstruction model as a tool for exploring the nature of representations,
including: the influence of model architecture and training objectives
(specifically robust losses), the forms of invariance that networks achieve,
representational differences between correctly and incorrectly classified
images, and the effects of manipulating logits and images. We believe that our
method can inspire future investigations into the nature of information flow in
a neural net and can provide diagnostics for improving discriminative models.

    

### [[2103.11580] Integrating Electrochemical Modeling with Machine Learning for Lithium-Ion Batteries](http://arxiv.org/abs/2103.11580)


  Mathematical modeling of lithium-ion batteries (LiBs) is a central challenge
in advanced battery management. This paper presents a new approach to integrate
a physics-based model with machine learning to achieve high-precision modeling
for LiBs. This approach uniquely proposes to inform the machine learning model
of the dynamic state of the physical model, enabling a deep integration between
physics and machine learning. We propose two hybrid physics-machine learning
models based on the approach, which blend a single particle model with thermal
dynamics (SPMT) with a feedforward neural network (FNN) to perform
physics-informed learning of a LiB's dynamic behavior. The proposed models are
relatively parsimonious in structure and can provide considerable predictive
accuracy even at high C-rates, as shown by extensive simulations.

    

### [[2104.05541] Optimizing the Whole-life Cost in End-to-end CNN Acceleration](http://arxiv.org/abs/2104.05541)


  The acceleration of CNNs has gained increasing atten-tion since their success
in computer vision. With the heterogeneous functional layers that cannot be
pro-cessed by the accelerators proposed for convolution layers only, modern
end-to-end CNN acceleration so-lutions either transform the diverse computation
into matrix/vector arithmetic, which loses data reuse op-portunities in
convolution, or introduce dedicated functional unit to each kind of layer,
which results in underutilization and high update expense. To enhance the
whole-life cost efficiency, we need an acceleration solution that is efficient
in processing CNN layers and has the generality to apply to all kinds of
existing and emerging layers. To this end, we pro-pose GCONV Chain, a method to
convert the entire CNN computation into a chain of standard general
convolutions (GCONV) that can be efficiently pro-cessed by the existing CNN
accelerators. This paper comprehensively analyzes the GCONV Chain model and
proposes a full-stack implementation to support GCONV Chain. On one hand, the
results on seven var-ious CNNs demonstrate that GCONV Chain improves the
performance and energy efficiency of existing CNN accelerators by an average of
3.4x and 3.2x re-spectively. On the other hand, we show that GCONV Chain
provides low whole-life costs for CNN accelera-tion, including both developer
efforts and total cost of ownership for the users.

    

### [[2104.08415] Risk score learning for COVID-19 contact tracing apps](http://arxiv.org/abs/2104.08415)


  Digital contact tracing apps for COVID, such as the one developed by Google
and Apple, need to estimate the risk that a user was infected during a
particular exposure, in order to decide whether to notify the user to take
precautions, such as entering into quarantine, or requesting a test. Such risk
score models contain numerous parameters that must be set by the public health
authority. In this paper, we show how to automatically learn these parameters
from data.
Our method needs access to exposure and outcome data. Although this data is
already being collected (in an aggregated, privacy-preserving way) by several
health authorities, in this paper we limit ourselves to simulated data, so that
we can systematically study the different factors that affect the feasibility
of the approach. In particular, we show that the parameters become harder to
estimate when there is more missing data (e.g., due to infections which were
not recorded by the app), and when there is model misspecification.
Nevertheless, the learning approach outperforms a strong manually designed
baseline. Furthermore, the learning approach can adapt even when the risk
factors of the disease change, e.g., due to the evolution of new variants, or
the adoption of vaccines.

    

### [[2105.03905] Security Concerns on Machine Learning Solutions for 6G Networks in mmWave Beam Prediction](http://arxiv.org/abs/2105.03905)


  6G -- sixth generation -- is the latest cellular technology currently under
development for wireless communication systems. In recent years, machine
learning algorithms have been applied widely in various fields, such as
healthcare, transportation, energy, autonomous car, and many more. Those
algorithms have been also using in communication technologies to improve the
system performance in terms of frequency spectrum usage, latency, and security.
With the rapid developments of machine learning techniques, especially deep
learning, it is critical to take the security concern into account when
applying the algorithms. While machine learning algorithms offer significant
advantages for 6G networks, security concerns on Artificial Intelligent (AI)
models is typically ignored by the scientific community so far. However,
security is also a vital part of the AI algorithms, this is because the AI
model itself can be poisoned by attackers. This paper proposes a mitigation
method for adversarial attacks against proposed 6G machine learning models for
the millimeter-wave (mmWave) beam prediction using adversarial learning. The
main idea behind adversarial attacks against machine learning models is to
produce faulty results by manipulating trained deep learning models for 6G
applications for mmWave beam prediction. We also present the adversarial
learning mitigation method's performance for 6G security in mmWave beam
prediction application with fast gradient sign method attack. The mean square
errors (MSE) of the defended model under attack are very close to the
undefended model without attack.

    

### [[2105.08508] A deep learning approach for inverse design of the metasurface for dual-polarized waves](http://arxiv.org/abs/2105.08508)


  Compared to the conventional metasurface design, machine learning-based
methods have recently created an inspiring platform for an inverse realization
of the metasurfaces. Here, we have used the Deep Neural Network (DNN) for the
generation of desired output unit cell structures in an ultra-wide working
frequency band for both TE and TM polarized waves. To automatically generate
metasurfaces in a wide range of working frequencies from 4 to 45 GHz, we
deliberately design an 8 ring-shaped pattern in such a way that the unit-cells
generated in the dataset can produce single or multiple notches in the desired
working frequency band. Compared to the general approach, whereby the final
metasurface structure may be formed by any randomly distributed "0" and "1", we
propose here a restricted output structure. By restricting the output, the
number of calculations will be reduced and the learning speed will be
increased. Moreover, we have shown that the accuracy of the network reaches
91\%. Obtaining the final unit cell directly without any time-consuming
optimization algorithms for both TE and TM polarized waves, and high average
accuracy, promises an effective strategy for the metasurface design; thus, the
designer is required only to focus on the design goal.

    

### [[2105.09266] Copyright in Generative Deep Learning](http://arxiv.org/abs/2105.09266)


  Machine-generated artworks are now part of the contemporary art scene: they
are attracting significant investments and they are presented in exhibitions
together with those created by human artists. These artworks are mainly based
on generative deep learning techniques, which have seen a formidable
development and remarkable refinement in the very recent years. Given the
inherent characteristics of these techniques, a series of novel legal problems
arise.
In this article, we consider a set of key questions in the area of generative
deep learning for the arts, including the following: is it possible to use
copyrighted works as training set for generative models? How do we legally
store their copies in order to perform the training process? Who (if someone)
will own the copyright on the generated data? We try to answer these questions
considering the law in force in both the United States of America and the
European Union, and potential future alternatives. Finally, we also formulate a
set of practical guidelines for artists and developers working on deep learning
generated art.

    

### [[2105.09501] Contrastive Learning for Many-to-many Multilingual Neural Machine Translation](http://arxiv.org/abs/2105.09501)


  Existing multilingual machine translation approaches mainly focus on
English-centric directions, while the non-English directions still lag behind.
In this work, we aim to build a many-to-many translation system with an
emphasis on the quality of non-English language directions. Our intuition is
based on the hypothesis that a universal cross-language representation leads to
better multilingual translation performance. To this end, we propose mRASP2, a
training method to obtain a single unified multilingual translation model.
mRASP2 is empowered by two techniques: a) a contrastive learning scheme to
close the gap among representations of different languages, and b) data
augmentation on both multiple parallel and monolingual data to further align
token representations. For English-centric directions, mRASP2 outperforms
existing best unified model and achieves competitive or even better performance
than the pre-trained and fine-tuned model mBART on tens of WMT's translation
directions. For non-English directions, mRASP2 achieves an improvement of
average 10+ BLEU compared with the multilingual Transformer baseline. Code,
data and trained models are available at this https URL.

    

### [[2105.10578] Predicting Potential Drug Targets Using Tensor Factorisation and Knowledge Graph Embeddings](http://arxiv.org/abs/2105.10578)


  The drug discovery and development process is a long and expensive one,
costing over 1 billion USD on average per drug and taking 10-15 years. To
reduce the high levels of attrition throughout the process, there has been a
growing interest in applying machine learning methodologies to various stages
of drug discovery process in the recent decade, including at the earliest stage
- identification of druggable disease genes. In this paper, we have developed a
new tensor factorisation model to predict potential drug targets (i.e.,genes or
proteins) for diseases. We created a three dimensional tensor which consists of
1,048 targets, 860 diseases and 230,011 evidence attributes and clinical
outcomes connecting them, using data extracted from the Open Targets and
PharmaProjects databases. We enriched the data with gene representations
learned from a drug discovery-oriented knowledge graph and applied our proposed
method to predict the clinical outcomes for unseen target and dis-ease pairs.
We designed three evaluation strategies to measure the prediction performance
and benchmarked several commonly used machine learning classifiers together
with matrix and tensor factorisation methods. The result shows that
incorporating knowledge graph embeddings significantly improves the prediction
accuracy and that training tensor factorisation alongside a dense neural
network outperforms other methods. In summary, our framework combines two
actively studied machine learning approaches to disease target identification,
tensor factorisation and knowledge graph representation learning, which could
be a promising avenue for further exploration in data-driven drug discovery.

    

### [[2105.13591] Spatio-Temporal Dual Graph Neural Networks for Travel Time Estimation](http://arxiv.org/abs/2105.13591)


  Travel time estimation is one of the core tasks for the development of
intelligent transportation systems. Most previous works model the road segments
or intersections separately by learning their spatio-temporal characteristics
to estimate travel time. However, due to the continuous alternations of the
road segments and intersections in a path, the dynamic features are supposed to
be coupled and interactive. Therefore, modeling one of them limits further
improvement in accuracy of estimating travel time. To address the above
problems, a novel graph-based deep learning framework for travel time
estimation is proposed in this paper, namely Spatio-Temporal Dual Graph Neural
Networks (STDGNN). Specifically, we first establish the node-wise and edge-wise
graphs to respectively characterize the adjacency relations of intersections
and that of road segments. In order to extract the joint spatio-temporal
correlations of the intersections and road segments, we adopt the
spatio-temporal dual graph learning approach that incorporates multiple
spatial-temporal dual graph learning modules with multi-scale network
architectures for capturing multi-level spatial-temporal information from the
dual graph. Finally, we employ the multi-task learning approach to estimate the
travel time of a given whole route, each road segment and intersection
simultaneously. We conduct extensive experiments to evaluate our proposed model
on three real-world trajectory datasets, and the experimental results show that
STDGNN significantly outperforms several state-of-art baselines.

    

### [[2106.02549] Detect the Interactions that Matter in Matter: Geometric Attention for Many-Body Systems](http://arxiv.org/abs/2106.02549)


  Attention mechanisms are developing into a viable alternative to
convolutional layers as elementary building block of NNs. Their main advantage
is that they are not restricted to capture local dependencies in the input, but
can draw arbitrary connections. This unprecedented capability coincides with
the long-standing problem of modeling global atomic interactions in molecular
force fields and other many-body problems. In its original formulation,
however, attention is not applicable to the continuous domains in which the
atoms live. For this purpose we propose a variant to describe geometric
relations for arbitrary atomic configurations in Euclidean space that also
respects all relevant physical symmetries. We furthermore demonstrate, how the
successive application of our learned attention matrices effectively translates
the molecular geometry into a set of individual atomic contributions
on-the-fly.

    

### [[2106.08446] Bridge Networks: Relating Inputs through Vector-Symbolic Manipulations](http://arxiv.org/abs/2106.08446)


  Despite rapid progress, current deep learning methods face a number of
critical challenges. These include high energy consumption, catastrophic
forgetting, dependance on global losses, and an inability to reason
symbolically. By combining concepts from information bottleneck theory and
vector-symbolic architectures, we propose and implement a novel information
processing architecture, the 'Bridge network.' We show this architecture
provides unique advantages which can address the problem of global losses and
catastrophic forgetting. Furthermore, we argue that it provides a further basis
for increasing energy efficiency of execution and the ability to reason
symbolically.

    

### [[2106.13329] Covariance-Aware Private Mean Estimation Without Private Covariance Estimation](http://arxiv.org/abs/2106.13329)


  We present two sample-efficient differentially private mean estimators for
$d$-dimensional (sub)Gaussian distributions with unknown covariance.
Informally, given $n \gtrsim d/\alpha^2$ samples from such a distribution with
mean $\mu$ and covariance $\Sigma$, our estimators output $\tilde\mu$ such that
$\| \tilde\mu - \mu \|_{\Sigma} \leq \alpha$, where $\| \cdot \|_{\Sigma}$ is
the Mahalanobis distance. All previous estimators with the same guarantee
either require strong a priori bounds on the covariance matrix or require
$\Omega(d^{3/2})$ samples.
Each of our estimators is based on a simple, general approach to designing
differentially private mechanisms, but with novel technical steps to make the
estimator private and sample-efficient. Our first estimator samples a point
with approximately maximum Tukey depth using the exponential mechanism, but
restricted to the set of points of large Tukey depth. Proving that this
mechanism is private requires a novel analysis. Our second estimator perturbs
the empirical mean of the data set with noise calibrated to the empirical
covariance, without releasing the covariance itself. Its sample complexity
guarantees hold more generally for subgaussian distributions, albeit with a
slightly worse dependence on the privacy parameter. For both estimators,
careful preprocessing of the data is required to satisfy differential privacy.

    

### [[2106.13430] Subgraph Federated Learning with Missing Neighbor Generation](http://arxiv.org/abs/2106.13430)


  Graphs have been widely used in data mining and machine learning due to their
unique representation of real-world objects and their interactions. As graphs
are getting bigger and bigger nowadays, it is common to see their subgraphs
separately collected and stored in multiple local systems. Therefore, it is
natural to consider the subgraph federated learning setting, where each local
system holding a small subgraph that may be biased from the distribution of the
whole graph. Hence, the subgraph federated learning aims to collaboratively
train a powerful and generalizable graph mining model without directly sharing
their graph data. In this work, towards the novel yet realistic setting of
subgraph federated learning, we propose two major techniques: (1) FedSage,
which trains a GraphSage model based on FedAvg to integrate node features, link
structures, and task labels on multiple local subgraphs; (2) FedSage+, which
trains a missing neighbor generator along FedSage to deal with missing links
across local subgraphs. Empirical results on four real-world graph datasets
with synthesized subgraph federated learning settings demonstrate the
effectiveness and efficiency of our proposed techniques. At the same time,
consistent theoretical implications are made towards their generalization
ability on the global graphs.

    

### [[2107.00623] Improving Sound Event Classification by Increasing Shift Invariance in Convolutional Neural Networks](http://arxiv.org/abs/2107.00623)


  Recent studies have put into question the commonly assumed shift invariance
property of convolutional networks, showing that small shifts in the input can
affect the output predictions substantially. In this paper, we analyze the
benefits of addressing lack of shift invariance in CNN-based sound event
classification. Specifically, we evaluate two pooling methods to improve shift
invariance in CNNs, based on low-pass filtering and adaptive sampling of
incoming feature maps. These methods are implemented via small architectural
modifications inserted into the pooling layers of CNNs. We evaluate the effect
of these architectural changes on the FSD50K dataset using models of different
capacity and in presence of strong regularization. We show that these
modifications consistently improve sound event classification in all cases
considered. We also demonstrate empirically that the proposed pooling methods
increase shift invariance in the network, making it more robust against
time/frequency shifts in input spectrograms. This is achieved by adding a
negligible amount of trainable parameters, which makes these methods an
appealing alternative to conventional pooling layers. The outcome is a new
state-of-the-art mAP of 0.541 on the FSD50K classification benchmark.

    

### [[2107.10308] The Bitlet Model: A Parameterized Analytical Model to Compare PIM and CPU Systems](http://arxiv.org/abs/2107.10308)


  Nowadays, data-intensive applications are gaining popularity and, together
with this trend, processing-in-memory (PIM)-based systems are being given more
attention and have become more relevant. This paper describes an analytical
modeling tool called Bitlet that can be used, in a parameterized fashion, to
estimate the performance and the power/energy of a PIM-based system and thereby
assess the affinity of workloads for PIM as opposed to traditional computing.
The tool uncovers interesting tradeoffs between, mainly, the PIM computation
complexity (cycles required to perform a computation through PIM), the amount
of memory used for PIM, the system memory bandwidth, and the data transfer
size. Despite its simplicity, the model reveals new insights when applied to
real-life examples. The model is demonstrated for several synthetic examples
and then applied to explore the influence of different parameters on two
systems - IMAGING and FloatPIM. Based on the demonstrations, insights about PIM
and its combination with CPU are concluded.

    

### [[2107.10448] Flexible Distributed Matrix Multiplication](http://arxiv.org/abs/2107.10448)


  The distributed matrix multiplication problem with an unknown number of
stragglers is considered, where the goal is to efficiently and flexibly obtain
the product of two massive matrices by distributing the computation across N
servers. There are up to N - R stragglers but the exact number is not known a
priori. Motivated by reducing the computation load of each server, a flexible
solution is proposed to fully utilize the computation capability of available
servers. The computing task for each server is separated into several subtasks,
constructed based on Entangled Polynomial codes by Yu et al. The final results
can be obtained from either a larger number of servers with a smaller amount of
computation completed per server or a smaller number of servers with a larger
amount of computation completed per server. The required finite field size of
the proposed solution is less than 2N. Moreover, the optimal design parameters
such as the partitioning of the input matrices is discussed. Our constructions
can also be generalized to other settings such as batch distributed matrix
multiplication and secure distributed matrix multiplication.

    

### [[2107.10467] Improving Blockchain Consistency by Assigning Weights to Random Blocks](http://arxiv.org/abs/2107.10467)


  Blockchains based on the celebrated Nakamoto consensus protocol have shown
promise in several applications, including cryptocurrencies. However, these
blockchains have inherent scalability limits caused by the protocol's consensus
properties. In particular, the consistency property demonstrates a tight
trade-off between block production speed and the system's security in terms of
resisting adversarial attacks. This paper proposes a novel method, Ironclad,
that improves blockchain consistency by assigning different weights to randomly
selected blocks. We analyze the fundamental properties of our method and show
that the combination of our method with Nakamoto consensus protocols can lead
to significant improvement in consistency. A direct result is that
Nakamoto+Ironclad can enable a much faster ($10\sim 50$ times with normal
parameter settings) block production rate than Nakamoto protocol under the same
security guarantee with the same proportion of malicious mining power.

    

### [[1802.08417] Geometric Lower Bounds for Distributed Parameter Estimation under Communication Constraints](http://arxiv.org/abs/1802.08417)


  We consider parameter estimation in distributed networks, where each sensor
in the network observes an independent sample from an underlying distribution
and has $k$ bits to communicate its sample to a centralized processor which
computes an estimate of a desired parameter. We develop lower bounds for the
minimax risk of estimating the underlying parameter for a large class of losses
and distributions. Our results show that under mild regularity conditions, the
communication constraint reduces the effective sample size by a factor of $d$
when $k$ is small, where $d$ is the dimension of the estimated parameter.
Furthermore, this penalty reduces at most exponentially with increasing $k$,
which is the case for some models, e.g., estimating high-dimensional
distributions. For other models however, we show that the sample size reduction
is re-mediated only linearly with increasing $k$, e.g. when some sub-Gaussian
structure is available. We apply our results to the distributed setting with
product Bernoulli model, multinomial model, Gaussian location models, and
logistic regression which recover or strengthen existing results.
Our approach significantly deviates from existing approaches for developing
information-theoretic lower bounds for communication-efficient estimation. We
circumvent the need for strong data processing inequalities used in prior work
and develop a geometric approach which builds on a new representation of the
communication constraint. This approach allows us to strengthen and generalize
existing results with simpler and more transparent proofs.

    

### [[2106.14479] Distributed stochastic gradient tracking algorithm with variance reduction for non-convex optimization](http://arxiv.org/abs/2106.14479)


  This paper proposes a distributed stochastic algorithm with variance
reduction for general smooth non-convex finite-sum optimization, which has wide
applications in signal processing and machine learning communities. In
distributed setting, large number of samples are allocated to multiple agents
in the network. Each agent computes local stochastic gradient and communicates
with its neighbors to seek for the global optimum. In this paper, we develop a
modified variance reduction technique to deal with the variance introduced by
stochastic gradients. Combining gradient tracking and variance reduction
techniques, this paper proposes a distributed stochastic algorithm, GT-VR, to
solve large-scale non-convex finite-sum optimization over multi-agent networks.
A complete and rigorous proof shows that the GT-VR algorithm converges to
first-order stationary points with $O(\frac{1}{k})$ convergence rate. In
addition, we provide the complexity analysis of the proposed algorithm.
Compared with some existing first-order methods, the proposed algorithm has a
lower $\mathcal{O}(PM\epsilon^{-1})$ gradient complexity under some mild
condition. By comparing state-of-the-art algorithms and GT-VR in experimental
simulations, we verify the efficiency of the proposed algorithm.

    

### [[2107.10350] Uncertainty-Aware Task Allocation for Distributed Autonomous Robots](http://arxiv.org/abs/2107.10350)


  This paper addresses task-allocation problems with uncertainty in situational
awareness for distributed autonomous robots (DARs). The uncertainty propagation
over a task-allocation process is done by using the Unscented transform that
uses the Sigma-Point sampling mechanism. It has great potential to be employed
for generic task-allocation schemes, in the sense that there is no need to
modify an existing task-allocation method that has been developed without
considering the uncertainty in the situational awareness. The proposed
framework was tested in a simulated environment where the decision-maker needs
to determine an optimal allocation of multiple locations assigned to multiple
mobile flying robots whose locations come as random variables of known mean and
covariance. The simulation result shows that the proposed stochastic task
allocation approach generates an assignment with 30% less overall cost than the
one without considering the uncertainty.

    

### [[2107.10390] Reinforcement Learning Agent Training with Goals for Real World Tasks](http://arxiv.org/abs/2107.10390)


  Reinforcement Learning (RL) is a promising approach for solving various
control, optimization, and sequential decision making tasks. However, designing
reward functions for complex tasks (e.g., with multiple objectives and safety
constraints) can be challenging for most users and usually requires multiple
expensive trials (reward function hacking). In this paper we propose a
specification language (Inkling Goal Specification) for complex control and
optimization tasks, which is very close to natural language and allows a
practitioner to focus on problem specification instead of reward function
hacking. The core elements of our framework are: (i) mapping the high level
language to a predicate temporal logic tailored to control and optimization
tasks, (ii) a novel automaton-guided dense reward generation that can be used
to drive RL algorithms, and (iii) a set of performance metrics to assess the
behavior of the system. We include a set of experiments showing that the
proposed method provides great ease of use to specify a wide range of real
world tasks; and that the reward generated is able to drive the policy training
to achieve the specified goal.

    

### [[2107.10410] Evaluation of In-Person Counseling Strategies To Develop Physical Activity Chatbot for Women](http://arxiv.org/abs/2107.10410)


  Artificial intelligence chatbots are the vanguard in technology-based
intervention to change people's behavior. To develop intervention chatbots, the
first step is to understand natural language conversation strategies in human
conversation. This work introduces an intervention conversation dataset
collected from a real-world physical activity intervention program for women.
We designed comprehensive annotation schemes in four dimensions (domain,
strategy, social exchange, and task-focused exchange) and annotated a subset of
dialogs. We built a strategy classifier with context information to detect
strategies from both trainers and participants based on the annotation. To
understand how human intervention induces effective behavior changes, we
analyzed the relationships between the intervention strategies and the
participants' changes in the barrier and social support for physical activity.
We also analyzed how participant's baseline weight correlates to the amount of
occurrence of the corresponding strategy. This work lays the foundation for
developing a personalized physical activity intervention bot. The dataset and
code are available at
this https URL


### [[2107.10433] MFGNet: Dynamic Modality-Aware Filter Generation for RGB-T Tracking](http://arxiv.org/abs/2107.10433)


  Many RGB-T trackers attempt to attain robust feature representation by
utilizing an adaptive weighting scheme (or attention mechanism). Different from
these works, we propose a new dynamic modality-aware filter generation module
(named MFGNet) to boost the message communication between visible and thermal
data by adaptively adjusting the convolutional kernels for various input images
in practical tracking. Given the image pairs as input, we first encode their
features with the backbone network. Then, we concatenate these feature maps and
generate dynamic modality-aware filters with two independent networks. The
visible and thermal filters will be used to conduct a dynamic convolutional
operation on their corresponding input feature maps respectively. Inspired by
residual connection, both the generated visible and thermal feature maps will
be summarized with input feature maps. The augmented feature maps will be fed
into the RoI align module to generate instance-level features for subsequent
classification. To address issues caused by heavy occlusion, fast motion, and
out-of-view, we propose to conduct a joint local and global search by
exploiting a new direction-aware target-driven attention mechanism. The spatial
and temporal recurrent neural network is used to capture the direction-aware
context for accurate global attention prediction. Extensive experiments on
three large-scale RGB-T tracking benchmark datasets validated the effectiveness
of our proposed algorithm. The project page of this paper is available at
this https URL.

    

### [[2107.10479] Copy and Paste method based on Pose for ReID](http://arxiv.org/abs/2107.10479)


  Re-identification(ReID) aims at matching objects in surveillance cameras with
different viewpoints. It's developing very fast, but there is no processing
method for the ReID task in multiple scenarios at this stage. However, this
dose happen all the time in real life, such as the security scenarios. This
paper explores a new scenario of Re-identification, which differs in
perspective, background, and pose(walking or cycling).
Obviously, ordinary ReID processing methods cannot handle this scenario well.
As we all konw, the best way to deal with that it is to introduce image
datasets in this scanario, But this one is very expensive. To solve this
problem, this paper proposes a simple and effective way to generate images in
some new scenario, which is named Copy and Paste method based on Pose(CPP). The
CPP is a method based on key point detection, using copy and paste, to
composite a new semantic image dataset in two different semantic image
datasets. Such as, we can use pedestrians and bicycles to generate some images
that shows the same person rides on different bicycles. The CPP is suitable for
ReID tasks in new scenarios and it outperforms state-of-the-art on the original
datasets in original ReID tasks. Specifically, it can also have better
generalization performance for third-party public datasets. Code and Datasets
which composited by the CPP will be available in the future.

    

### [[2107.10508] Multiple Query Optimization using a Hybrid Approach of Classical and Quantum Computing](http://arxiv.org/abs/2107.10508)


  Quantum computing promises to solve difficult optimization problems in
chemistry, physics and mathematics more efficiently than classical computers,
but requires fault-tolerant quantum computers with millions of qubits. To
overcome errors introduced by today's quantum computers, hybrid algorithms
combining classical and quantum computers are used. In this paper we tackle the
multiple query optimization problem (MQO) which is an important NP-hard problem
in the area of data-intensive problems. We propose a novel hybrid
classical-quantum algorithm to solve the MQO on a gate-based quantum computer.
We perform a detailed experimental evaluation of our algorithm and compare its
performance against a competing approach that employs a quantum annealer --
another type of quantum computer. Our experimental results demonstrate that our
algorithm currently can only handle small problem sizes due to the limited
number of qubits available on a gate-based quantum computer compared to a
quantum computer based on quantum annealing. However, our algorithm shows a
qubit efficiency of close to 99% which is almost a factor of 2 higher compared
to the state of the art implementation. Finally, we analyze how our algorithm
scales with larger problem sizes and conclude that our approach shows promising
results for near-term quantum computers.

    

### [[2107.10602] CNN-based Realized Covariance Matrix Forecasting](http://arxiv.org/abs/2107.10602)


  It is well known that modeling and forecasting realized covariance matrices
of asset returns play a crucial role in the field of finance. The availability
of high frequency intraday data enables the modeling of the realized covariance
matrices directly. However, most of the models available in the literature
depend on strong structural assumptions and they often suffer from the curse of
dimensionality. We propose an end-to-end trainable model built on the CNN and
Convolutional LSTM (ConvLSTM) which does not require to make any distributional
or structural assumption but could handle high-dimensional realized covariance
matrices consistently. The proposed model focuses on local structures and
spatiotemporal correlations. It learns a nonlinear mapping that connect the
historical realized covariance matrices to the future one. Our empirical
studies on synthetic and real-world datasets demonstrate its excellent
forecasting ability compared with several advanced volatility models.

    

### [[2107.10648] DEAP-FAKED: Knowledge Graph based Approach for Fake News Detection](http://arxiv.org/abs/2107.10648)


  Fake News on social media platforms has attracted a lot of attention in
recent times, primarily for events related to politics (2016 US Presidential
elections), healthcare (infodemic during COVID-19), to name a few. Various
methods have been proposed for detecting Fake News. The approaches span from
exploiting techniques related to network analysis, Natural Language Processing
(NLP), and the usage of Graph Neural Networks (GNNs). In this work, we propose
DEAP-FAKED, a knowleDgE grAPh FAKe nEws Detection framework for identifying
Fake News. Our approach is a combination of the NLP -- where we encode the news
content, and the GNN technique -- where we encode the Knowledge Graph (KG). A
variety of these encodings provides a complementary advantage to our detector.
We evaluate our framework using two publicly available datasets containing
articles from domains such as politics, business, technology, and healthcare.
As part of dataset pre-processing, we also remove the bias, such as the source
of the articles, which could impact the performance of the models. DEAP-FAKED
obtains an F1-score of 88% and 78% for the two datasets, which is an
improvement of 21%, and 3% respectively, which shows the effectiveness of the
approach.

    

### [[2107.10653] Dialogue Object Search](http://arxiv.org/abs/2107.10653)


  We envision robots that can collaborate and communicate seamlessly with
humans. It is necessary for such robots to decide both what to say and how to
act, while interacting with humans. To this end, we introduce a new task,
dialogue object search: A robot is tasked to search for a target object (e.g.
fork) in a human environment (e.g., kitchen), while engaging in a "video call"
with a remote human who has additional but inexact knowledge about the target's
location. That is, the robot conducts speech-based dialogue with the human,
while sharing the image from its mounted camera. This task is challenging at
multiple levels, from data collection, algorithm and system development,to
evaluation. Despite these challenges, we believe such a task blocks the path
towards more intelligent and collaborative robots. In this extended abstract,
we motivate and introduce the dialogue object search task and analyze examples
collected from a pilot study. We then discuss our next steps and conclude with
several challenges on which we hope to receive feedback.

    

### [[2107.10715] Philosophical Specification of Empathetic Ethical Artificial Intelligence](http://arxiv.org/abs/2107.10715)


  In order to construct an ethical artificial intelligence (AI) two complex
problems must be overcome. Firstly, humans do not consistently agree on what is
or is not ethical. Second, contemporary AI and machine learning methods tend to
be blunt instruments which either search for solutions within the bounds of
predefined rules, or mimic behaviour. An ethical AI must be capable of
inferring unspoken rules, interpreting nuance and context, possess and be able
to infer intent, and explain not just its actions but its intent. Using
enactivism, semiotics, perceptual symbol systems and symbol emergence, we
specify an agent that learns not just arbitrary relations between signs but
their meaning in terms of the perceptual states of its sensorimotor system.
Subsequently it can learn what is meant by a sentence and infer the intent of
others in terms of its own experiences. It has malleable intent because the
meaning of symbols changes as it learns, and its intent is represented
symbolically as a goal. As such it may learn a concept of what is most likely
to be considered ethical by the majority within a population of humans, which
may then be used as a goal. The meaning of abstract symbols is expressed using
perceptual symbols of raw sensorimotor stimuli as the weakest (consistent with
Ockham's Razor) necessary and sufficient concept, an intensional definition
learned from an ostensive definition, from which the extensional definition or
category of all ethical decisions may be obtained. Because these abstract
symbols are the same for both situation and response, the same symbol is used
when either performing or observing an action. This is akin to mirror neurons
in the human brain. Mirror symbols may allow the agent to empathise, because
its own experiences are associated with the symbol, which is also associated
with the observation of another agent experiencing something that symbol
represents.

    

### [[2107.10742] Multi-modal Residual Perceptron Network for Audio-Video Emotion Recognition](http://arxiv.org/abs/2107.10742)


  Emotion recognition is an important research field for Human-Computer
Interaction(HCI). Audio-Video Emotion Recognition (AVER) is now attacked with
Deep Neural Network (DNN) modeling tools. In published papers, as a rule, the
authors show only cases of the superiority of multi modalities over audio-only
or video-only modalities. However, there are cases superiority in single
modality can be found. In our research, we hypothesize that for fuzzy
categories of emotional events, the higher noise of one modality can amplify
the lower noise of the second modality represented indirectly in the parameters
of the modeling neural network. To avoid such cross-modal information
interference we define a multi-modal Residual Perceptron Network (MRPN) which
learns from multi-modal network branches creating deep feature representation
with reduced noise. For the proposed MRPN model and the novel time augmentation
for streamed digital movies, the state-of-art average recognition rate was
improved to 91.4% for The Ryerson Audio-Visual Database of Emotional Speech and
Song(RAVDESS) dataset and to 83.15% for Crowd-sourced Emotional multi-modal
Actors Dataset(Crema-d). Moreover, the MRPN concept shows its potential for
multi-modal classifiers dealing with signal sources not only of optical and
acoustical type.

    

### [[2107.10832] A Logic of Expertise](http://arxiv.org/abs/2107.10832)


  In this paper we introduce a simple modal logic framework to reason about the
expertise of an information source. In the framework, a source is an expert on
a proposition $p$ if they are able to correctly determine the truth value of
$p$ in any possible world. We also consider how information may be false, but
true after accounting for the lack of expertise of the source. This is relevant
for modelling situations in which information sources make claims beyond their
domain of expertise. We use non-standard semantics for the language based on an
expertise set with certain closure properties. It turns out there is a close
connection between our semantics and S5 epistemic logic, so that expertise can
be expressed in terms of knowledge at all possible states. We use this
connection to obtain a sound and complete axiomatisation.

    

### [[2107.10843] HARP-Net: Hyper-Autoencoded Reconstruction Propagation\\for Scalable Neural Audio Coding](http://arxiv.org/abs/2107.10843)


  An autoencoder-based codec employs quantization to turn its bottleneck layer
activation into bitstrings, a process that hinders information flow between the
encoder and decoder parts. To circumvent this issue, we employ additional skip
connections between the corresponding pair of encoder-decoder layers. The
assumption is that, in a mirrored autoencoder topology, a decoder layer
reconstructs the intermediate feature representation of its corresponding
encoder layer. Hence, any additional information directly propagated from the
corresponding encoder layer helps the reconstruction. We implement this kind of
skip connections in the form of additional autoencoders, each of which is a
small codec that compresses the massive data transfer between the paired
encoder-decoder layers. We empirically verify that the proposed
hyper-autoencoded architecture improves perceptual audio quality compared to an
ordinary autoencoder baseline.

    

### [[2004.09705] Explainable Goal-Driven Agents and Robots -- A Comprehensive Review](http://arxiv.org/abs/2004.09705)


  Recent applications of autonomous agents and robots, such as self-driving
cars, scenario-based trainers, exploration robots, and service robots have
brought attention to crucial trust-related challenges associated with the
current generation of artificial intelligence (AI) systems. AI systems based on
the connectionist deep learning neural network approach lack capabilities of
explaining their decisions and actions to others, despite their great
successes. Without symbolic interpretation capabilities, they are black boxes,
which renders their decisions or actions opaque, making it difficult to trust
them in safety-critical applications. The recent stance on the explainability
of AI systems has witnessed several approaches on eXplainable Artificial
Intelligence (XAI); however, most of the studies have focused on data-driven
XAI systems applied in computational sciences. Studies addressing the
increasingly pervasive goal-driven agents and robots are still missing. This
paper reviews approaches on explainable goal-driven intelligent agents and
robots, focusing on techniques for explaining and communicating agents
perceptual functions (example, senses, and vision) and cognitive reasoning
(example, beliefs, desires, intention, plans, and goals) with humans in the
loop. The review highlights key strategies that emphasize transparency,
understandability, and continual learning for explainability. Finally, the
paper presents requirements for explainability and suggests a roadmap for the
possible realization of effective goal-driven explainable agents and robots.

    

### [[2009.06371] SeqROCTM: A Matlab toolbox for the analysis of Sequence of Random Objects driven by Context Tree Models](http://arxiv.org/abs/2009.06371)


  In several research problems we deal with probabilistic sequences of inputs
(e.g., sequence of stimuli) from which an agent generates a corresponding
sequence of responses and it is of interest to model the relation between them.
A new class of stochastic processes, namely \textit{sequences of random objects
driven by context tree models}, has been introduced to model such relation in
the context of auditory statistical learning. This paper introduces a freely
available Matlab toolbox (SeqROCTM) that implements this new class of
stochastic processes and three model selection procedures to make inference on
it. Besides, due to the close relation of the new mathematical framework with
context tree models, the toolbox also implements several existing model
selection algorithms for context tree models.

    

### [[2010.06797] Reinforcement Learning Based Temporal Logic Control with Maximum Probabilistic Satisfaction](http://arxiv.org/abs/2010.06797)


  This paper presents a model-free reinforcement learning (RL) algorithm to
synthesize a control policy that maximizes the satisfaction probability of
linear temporal logic (LTL) specifications. Due to the consideration of
environment and motion uncertainties, we model the robot motion as a
probabilistic labeled Markov decision process with unknown transition
probabilities and unknown probabilistic label functions. The LTL task
specification is converted to a limit deterministic generalized Büchi
automaton (LDGBA) with several accepting sets to maintain dense rewards during
learning. The novelty of applying LDGBA is to construct an embedded LDGBA
(E-LDGBA) by designing a synchronous tracking-frontier function, which enables
the record of non-visited accepting sets without increasing dimensional and
computational complexity. With appropriate dependent reward and discount
functions, rigorous analysis shows that any method that optimizes the expected
discount return of the RL-based approach is guaranteed to find the optimal
policy that maximizes the satisfaction probability of the LTL specifications. A
model-free RL-based motion planning strategy is developed to generate the
optimal policy in this paper. The effectiveness of the RL-based control
synthesis is demonstrated via simulation and experimental results.

    

### [[2012.02640] A Comparison of Natural Language Understanding Platforms for Chatbots in Software Engineering](http://arxiv.org/abs/2012.02640)


  Chatbots are envisioned to dramatically change the future of Software
Engineering, allowing practitioners to chat and inquire about their software
projects and interact with different services using natural language. At the
heart of every chatbot is a Natural Language Understanding (NLU) component that
enables the chatbot to understand natural language input. Recently, many NLU
platforms were provided to serve as an off-the-shelf NLU component for
chatbots, however, selecting the best NLU for Software Engineering chatbots
remains an open challenge.
Therefore, in this paper, we evaluate four of the most commonly used NLUs,
namely IBM Watson, Google Dialogflow, Rasa, and Microsoft LUIS to shed light on
which NLU should be used in Software Engineering based chatbots. Specifically,
we examine the NLUs' performance in classifying intents, confidence scores
stability, and extracting entities. To evaluate the NLUs, we use two datasets
that reflect two common tasks performed by Software Engineering practitioners,
1) the task of chatting with the chatbot to ask questions about software
repositories 2) the task of asking development questions on Q&A forums (e.g.,
Stack Overflow). According to our findings, IBM Watson is the best performing
NLU when considering the three aspects (intents classification, confidence
scores, and entity extraction). However, the results from each individual
aspect show that, in intents classification, IBM Watson performs the best with
an F1-measure > 84%, but in confidence scores, Rasa comes on top with a median
confidence score higher than 0.91. Our results also show that all NLUs, except
for Dialogflow, generally provide trustable confidence scores. For entity
extraction, Microsoft LUIS and IBM Watson outperform other NLUs in the two SE
tasks. Our results provide guidance to software engineering practitioners when
deciding which NLU to use in their chatbots.

    

### [[2012.12305] Confronting Abusive Language Online: A Survey from the Ethical and Human Rights Perspective](http://arxiv.org/abs/2012.12305)


  The pervasiveness of abusive content on the internet can lead to severe
psychological and physical harm. Significant effort in Natural Language
Processing (NLP) research has been devoted to addressing this problem through
abusive content detection and related sub-areas, such as the detection of hate
speech, toxicity, cyberbullying, etc. Although current technologies achieve
high classification performance in research studies, it has been observed that
the real-life application of this technology can cause unintended harms, such
as the silencing of under-represented groups. We review a large body of NLP
research on automatic abuse detection with a new focus on ethical challenges,
organized around eight established ethical principles: privacy, accountability,
safety and security, transparency and explainability, fairness and
non-discrimination, human control of technology, professional responsibility,
and promotion of human values. In many cases, these principles relate not only
to situational ethical codes, which may be context-dependent, but are in fact
connected to universal human rights, such as the right to privacy, freedom from
discrimination, and freedom of expression. We highlight the need to examine the
broad social impacts of this technology, and to bring ethical and human rights
considerations to every stage of the application life-cycle, from task
formulation and dataset design, to model training and evaluation, to
application deployment. Guided by these principles, we identify several
opportunities for rights-respecting, socio-technical solutions to detect and
confront online abuse, including `nudging', `quarantining', value sensitive
design, counter-narratives, style transfer, and AI-driven public education
applications.

    

### [[2101.09491] Symbiotic System of Systems Design for Safe and Resilient Autonomous Robotics in Offshore Wind Farms](http://arxiv.org/abs/2101.09491)


  To reduce Operation and Maintenance (O&M) costs on offshore wind farms,
wherein 80% of the O&M cost relates to deploying personnel, the offshore wind
sector looks to Robotics and Artificial Intelligence (RAI) for solutions.
Barriers to Beyond Visual Line of Sight (BVLOS) robotics include operational
safety compliance and resilience, inhibiting the commercialization of
autonomous services offshore. To address safety and resilience challenges we
propose a Symbiotic System Of Systems Approach (SSOSA), reflecting the
lifecycle learning and co-evolution with knowledge sharing for mutual gain of
robotic platforms and remote human operators. Our novel methodology enables the
run-time verification of safety, reliability and resilience during autonomous
missions. To achieve this, a Symbiotic Digital Architecture (SDA) was developed
to synchronize digital models of the robot, environment, infrastructure, and
integrate front-end analytics and bidirectional communication for autonomous
adaptive mission planning and situation reporting to a remote operator. A
reliability ontology for the deployed robot, based on our holistic
hierarchical-relational model, supports computationally efficient platform data
analysis. We demonstrate an asset inspection mission within a confined space
through Cooperative, Collaborative and Corroborative (C3) governance (internal
and external symbiosis) via decision-making processes and the associated
structures. We create a hyper enabled human interaction capability to analyze
the mission status, diagnostics of critical sub-systems within the robot to
provide automatic updates to our AI-driven run-time reliability ontology. This
enables faults to be translated into failure modes for decision-making during
the mission.

    

### [[2106.00285] Shapley Counterfactual Credits for Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2106.00285)


  Centralized Training with Decentralized Execution (CTDE) has been a popular
paradigm in cooperative Multi-Agent Reinforcement Learning (MARL) settings and
is widely used in many real applications. One of the major challenges in the
training process is credit assignment, which aims to deduce the contributions
of each agent according to the global rewards. Existing credit assignment
methods focus on either decomposing the joint value function into individual
value functions or measuring the impact of local observations and actions on
the global value function. These approaches lack a thorough consideration of
the complicated interactions among multiple agents, leading to an unsuitable
assignment of credit and subsequently mediocre results on MARL. We propose
Shapley Counterfactual Credit Assignment, a novel method for explicit credit
assignment which accounts for the coalition of agents. Specifically, Shapley
Value and its desired properties are leveraged in deep MARL to credit any
combinations of agents, which grants us the capability to estimate the
individual credit for each agent. Despite this capability, the main technical
difficulty lies in the computational complexity of Shapley Value who grows
factorially as the number of agents. We instead utilize an approximation method
via Monte Carlo sampling, which reduces the sample complexity while maintaining
its effectiveness. We evaluate our method on StarCraft II benchmarks across
different scenarios. Our method outperforms existing cooperative MARL
algorithms significantly and achieves the state-of-the-art, with especially
large margins on tasks with more severe difficulties.

    

### [[2106.09885] An Improved Single Step Non-autoregressive Transformer for Automatic Speech Recognition](http://arxiv.org/abs/2106.09885)


  Non-autoregressive mechanisms can significantly decrease inference time for
speech transformers, especially when the single step variant is applied.
Previous work on CTC alignment-based single step non-autoregressive transformer
(CASS-NAT) has shown a large real time factor (RTF) improvement over
autoregressive transformers (AT). In this work, we propose several methods to
improve the accuracy of the end-to-end CASS-NAT, followed by performance
analyses. First, convolution augmented self-attention blocks are applied to
both the encoder and decoder modules. Second, we propose to expand the trigger
mask (acoustic boundary) for each token to increase the robustness of CTC
alignments. In addition, iterated loss functions are used to enhance the
gradient update of low-layer parameters. Without using an external language
model, the WERs of the improved CASS-NAT, when using the three methods, are
3.1%/7.2% on Librispeech test clean/other sets and the CER is 5.4% on the
Aishell1 test set, achieving a 7%~21% relative WER/CER improvement. For the
analyses, we plot attention weight distributions in the decoders to visualize
the relationships between token-level acoustic embeddings. When the acoustic
embeddings are visualized, we find that they have a similar behavior to word
embeddings, which explains why the improved CASS-NAT performs similarly to AT.

    

### [[1506.05903] Detecting Real-World Influence Through Twitter](http://arxiv.org/abs/1506.05903)


  In this paper, we investigate the issue of detecting the real-life influence
of people based on their Twitter account. We propose an overview of common
Twitter features used to characterize such accounts and their activity, and
show that these are inefficient in this context. In particular, retweets and
followers numbers, and Klout score are not relevant to our analysis. We thus
propose several Machine Learning approaches based on Natural Language
Processing and Social Network Analysis to label Twitter users as Influencers or
not. We also rank them according to a predicted influence level. Our proposals
are evaluated over the CLEF RepLab 2014 dataset, and outmatch state-of-the-art
ranking methods.

    

### [[2009.10679] An embedded deep learning system for augmented reality in firefighting applications](http://arxiv.org/abs/2009.10679)


  Firefighting is a dynamic activity, in which numerous operations occur
simultaneously. Maintaining situational awareness (i.e., knowledge of current
conditions and activities at the scene) is critical to the accurate
decision-making necessary for the safe and successful navigation of a fire
environment by firefighters. Conversely, the disorientation caused by hazards
such as smoke and extreme heat can lead to injury or even fatality. This
research implements recent advancements in technology such as deep learning,
point cloud and thermal imaging, and augmented reality platforms to improve a
firefighter's situational awareness and scene navigation through improved
interpretation of that scene. We have designed and built a prototype embedded
system that can leverage data streamed from cameras built into a firefighter's
personal protective equipment (PPE) to capture thermal, RGB color, and depth
imagery and then deploy already developed deep learning models to analyze the
input data in real time. The embedded system analyzes and returns the processed
images via wireless streaming, where they can be viewed remotely and relayed
back to the firefighter using an augmented reality platform that visualizes the
results of the analyzed inputs and draws the firefighter's attention to objects
of interest, such as doors and windows otherwise invisible through smoke and
flames.

    

### [[2107.10533] CGuard: Efficient Spatial Safety for C](http://arxiv.org/abs/2107.10533)


  Spatial safety violations are the root cause of many security attacks and
unexpected behavior of applications. Existing techniques to enforce spatial
safety work broadly at either object or pointer granularity. Object-based
approaches tend to incur high CPU overheads, whereas pointer-based approaches
incur both high CPU and memory overheads. SGXBounds, an object-based approach,
is so far the most efficient technique that provides complete out-of-bounds
protection for objects. However, a major drawback of this approach is that it
restricts the application address space to 4GB.
In this paper, we present CGuard, a tool that provides object-bounds
protection for C applications with comparable overheads to SGXBounds without
restricting the application address space. CGuard stores the bounds information
just before the base address of an object and encodes the relative offset of
the base address in the spare bits of the virtual address available in x86_64
architecture. For an object that can't fit in the spare bits, CGuard uses a
custom memory layout that enables it to find the base address of the object in
just one memory access. Our study revealed spatial safety violations in the gcc
and x264 benchmarks from the SPEC CPU2017 benchmark suite and the string_match
benchmark from the Phoenix benchmark suite. The execution time overheads for
the SPEC CPU2017 and Phoenix benchmark suites were 44% and 25% respectively,
whereas the reduction in the throughput for the Apache webserver when the CPUs
were fully saturated was 30%. These results indicate that CGuard can be highly
effective while maintaining a reasonable degree of efficiency.

    

### [[2107.10545] Fundamental Constructs in Programming Languages](http://arxiv.org/abs/2107.10545)


  Specifying the semantics of a programming language formally can have many
benefits. However, it can also require a huge effort. The effort can be
significantly reduced by translating language syntax to so-called fundamental
constructs (funcons). A translation to funcons is easy to update when the
language evolves, and it exposes relationships between individual language
constructs.
The PLanCompS project has developed an initial collection of funcons
(primarily for translation of functional and imperative languages). The
behaviour of each funcon is defined, once and for all, using a modular variant
of structural operational semantics. The definitions are available online.
This paper introduces and motivates funcons. It illustrates translation of
language constructs to funcons, and how funcons are defined. It also relates
funcons to notation used in previous frameworks, including monadic semantics
and action semantics.

    

### [[2107.10566] MPIs Language Bindings are Holding MPI Back](http://arxiv.org/abs/2107.10566)


  Over the past two decades, C++ has been adopted as a major HPC language
(displacing C to a large extent, andFortran to some degree as well). Idiomatic
C++ is clearly how C++ is being used nowadays. But, MPIs syntax and semantics
defined and extended with C and Fortran interfaces that align with the
capabilities and limitations of C89 and Fortran-77.Unfortunately, the
language-independent specification also clearly reflects the intersection of
what these languages could syntactically and semantically manage at the outset
in 1993, rather than being truly language this http URL this paper, we propose a
modern C++ language interface to replace the C language binding for C++
programmers with an upward-compatible architecture that leverages all the
benefits of C++11-20 for performance, productivity, and interoperability with
other popular C++ libraries and interfaces for HPC. Demand is demonstrably
strong for this second attempt at language support for C++ in MPI after the
original interface, which was added in MPI-2, then was found to lack specific
benefits over theC binding, and so was subsequently removed in MPI-3. Since C++
and its idiomatic usage have evolved since the original C++ language binding
was removed from the standard, this new effort is both timely and important for
MPI applications. Also, many C++ application programmers create their own, ad
hoc shim libraries over MPI to provide some degree of abstraction unique to
their particular project, which means many such abstraction libraries are being
devised without any specific commonality other than the demand for such.

    

### [[2107.10793] A Typed Slicing Compilation of the Polymorphic RPC Calculus](http://arxiv.org/abs/2107.10793)


  The polymorphic RPC calculus allows programmers to write succinct multitier
programs using polymorphic location constructs. However, until now it lacked an
implementation. We develop an experimental programming language based on the
polymorphic RPC calculus. We introduce a polymorphic Client-Server (CS)
calculus with the client and server parts separated. In contrast to existing
untyped CS calculi, our calculus is not only able to resolve polymorphic
locations statically, but it is also able to do so dynamically. We design a
type-based slicing compilation of the polymorphic RPC calculus into this CS
calculus, proving type and semantic correctness. We propose a method to erase
types unnecessary for execution but retaining locations at runtime by
translating the polymorphic CS calculus into an untyped CS calculus, proving
semantic correctness.

    

### [[2103.06195] Evaluating Linear Functions to Symmetric Monoidal Categories](http://arxiv.org/abs/2103.06195)


  A number of domain specific languages, such as circuits or data-science
workflows, are best expressed as diagrams of boxes connected by wires.
Unfortunately, functional languages have traditionally been ill-equipped to
embed this sort of languages. The Arrow abstraction is an approximation, but we
argue that it does not capture the right properties.
A faithful abstraction is Symmetric Monoidal Categories (SMCs), but,so far,it
hasn't been convenient to use. We show how the advent of linear typing in
Haskell lets us bridge this gap. We provide a library which lets us program in
SMCs with linear functions instead of SMC combinators. This considerably lowers
the syntactic overhead of the EDSL to be on par with that of monadic DSLs. A
remarkable feature of our library is that, contrary to previously known methods
for categories, it does not use any metaprogramming.

    