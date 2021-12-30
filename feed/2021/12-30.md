
## 2021-12-30

### [[2112.13926] Resource-Efficient and Delay-Aware Federated Learning Design under Edge Heterogeneity](http://arxiv.org/abs/2112.13926)


  Federated learning (FL) has emerged as a popular methodology for distributing
machine learning across wireless edge devices. In this work, we consider
optimizing the tradeoff between model performance and resource utilization in
FL, under device-server communication delays and device computation
heterogeneity. Our proposed StoFedDelAv algorithm incorporates a local-global
model combiner into the FL synchronization step. We theoretically characterize
the convergence behavior of StoFedDelAv and obtain the optimal combiner
weights, which consider the global model delay and expected local gradient
error at each device. We then formulate a network-aware optimization problem
which tunes the minibatch sizes of the devices to jointly minimize energy
consumption and machine learning training loss, and solve the non-convex
problem through a series of convex approximations. Our simulations reveal that
StoFedDelAv outperforms the current art in FL in terms of model convergence
speed and network resource utilization when the minibatch size and the combiner
weights are adjusted. Additionally, our method can reduce the number of uplink
communication rounds required during the model training period to reach the
same accuracy.

    

### [[2112.14006] Multi-Band Wi-Fi Sensing with Matched Feature Granularity](http://arxiv.org/abs/2112.14006)


  Complementary to the fine-grained channel state information (CSI) from the
physical layer and coarse-grained received signal strength indicator (RSSI)
measurements, the mid-grained spatial beam attributes (e.g., beam SNR) that are
available at millimeter-wave (mmWave) bands during the mandatory beam training
phase can be repurposed for Wi-Fi sensing applications. In this paper, we
propose a multi-band Wi-Fi fusion method for Wi-Fi sensing that hierarchically
fuses the features from both the fine-grained CSI at sub-6 GHz and the
mid-grained beam SNR at 60 GHz in a granularity matching framework. The
granularity matching is realized by pairing two feature maps from the CSI and
beam SNR at different granularity levels and linearly combining all paired
feature maps into a fused feature map with learnable weights.
To further address the issue of limited labeled training data, we propose an
autoencoder-based multi-band Wi-Fi fusion network that can be pre-trained in an
unsupervised fashion. Once the autoencoder-based fusion network is pre-trained,
we detach the decoders and append multi-task sensing heads to the fused feature
map by fine-tuning the fusion block and re-training the multi-task heads from
the scratch. The multi-band Wi-Fi fusion framework is thoroughly validated by
in-house experimental Wi-Fi sensing datasets spanning three tasks: 1) pose
recognition; 2) occupancy sensing; and 3) indoor localization. Comparison to
four baseline methods (i.e., CSI-only, beam SNR-only, input fusion, and feature
fusion) demonstrates the granularity matching improves the multi-task sensing
performance. Quantitative performance is evaluated as a function of the number
of labeled training data, latent space dimension, and fine-tuning learning
rates.

    

### [[2112.14137] State Compression and Quantitative Assessment Model for Assessing Security Risks in the Oil and Gas Transmission Systems](http://arxiv.org/abs/2112.14137)


  The SCADA system is the foundation of the large-scale industrial control
system. It is widely used in industries of petrochemistry, electric power,
pipeline, etc. The natural gas SCADA system is among the critical
infrastructure systems that have security issues related to trusted
communications in transactions at the control system layer, and lack
quantitative risk assessment and mitigation models. However, to guarantee the
security of the Oil and Gas Transmission SCADA systems (OGTSS), there should be
a holistic security system that considers the nature of these SCADA systems. In
this paper, we augment our Security Awareness Framework with two new
contributions, (i) a Data Quantization and State Compression Approach (DQSCA)
that improves the classification accuracy, speeds up the detection algorithm,
and reduces the computational resource consumption. DQSCA reduces the size of
processed data while preserving original key events and patterns within the
datasets. (ii) A quantitative risk assessment model that carries out regular
system information security evaluation and assessment on the SCADA system using
a deductive process. Our experiments denote that DQSCA has a low negative
impact on the reduction of the detection accuracy (2.45% and 4.45%) while it
reduces the detection time much (27.74% and 42.06%) for the Turnipseed and Gao
datasets respectively. Furthermore, the mean absolute percentage error (MAPE)
rate for the proposed risk assessment model is lower than the intrusion
response system (Suricata) for the DOS, Response Injection, and Command
Injection attacks by 59.80%, 73.72%, and 66.96% respectively.

    

### [[2112.14138] Enhanced Wi-Fi RTT Ranging: A Sensor-Aided Learning Approach](http://arxiv.org/abs/2112.14138)


  The fine timing measurement (FTM) protocol is designed to determine precise
ranging between Wi-Fi devices using round-trip time (RTT) measurements.
However, the multipath propagation of radio waves generates inaccurate timing
information, degrading the ranging performance. In this study, we use a neural
network (NN) to adaptively learn the unique measurement patterns observed at
different indoor environments and produce enhanced ranging outputs from raw FTM
measurements. Moreover, the NN is trained based on an unsupervised learning
framework, using the naturally accumulated sensor data acquired from users
accessing location services. Therefore, the effort involved in collecting
training data is significantly minimized. The experimental results verified
that the collection of unlabeled data for a short duration is sufficient to
learn the pattern in raw FTM measurements and produce improved ranging results.
The proposed method reduced the ranging errors in raw distance measurements and
well-calibrated ranging results requiring the collection of ground truth data
by 47-50% and 17-29%, respectively. Consequently, positioning error reduced by
17-30% compared to the result with well-calibrated ranging.

    

### [[2112.14240] Heterogenous Networks: From small cells to 5G NR-U](http://arxiv.org/abs/2112.14240)


  With the exponential increase in mobile users, the mobile data demand has
grown tremendously. To meet these demands, cellular operators are constantly
innovating to enhance the capacity of cellular systems. Consequently, operators
have been reusing the licensed spectrum spatially, by deploying 4G/LTE small
cells (e.g., Femto Cells) in the past. However, despite the use of small cells,
licensed spectrum will be unable to meet the consistently rising data traffic
because of data-intensive applications such as augmented reality or virtual
reality (AR/VR) and on-the-go high-definition video streaming. Applications
such AR/VR and online gaming not only place extreme data demands on the
network, but are also latency-critical. To meet the QoS guarantees, cellular
operators have begun leveraging the unlicensed spectrum by coexisting with
Wi-Fi in the 5 GHz band. The standardizing body 3GPP, has prescribed cellular
standards for fair unlicensed coexistence with Wi-Fi, namely LTE Licensed
Assisted Access (LAA), New Radio in unlicensed (NR-U), and NR in Millimeter.
The rapid roll-out of LAA deployments in developed nations like the US, offers
an opportunity to study and analyze the performance of unlicensed coexistence
networks through real-world ground truth. Thus, this paper presents a
high-level overview of past, present, and future of the research in small cell
and unlicensed coexistence communication technologies. It outlines the vision
for future research work in the recently allocated unlicensed spectrum: The 6
GHz band, where the latest Wi-Fi standard, IEEE 802.11ax, will coexist with the
latest cellular technology, 5G New Radio (NR) in unlicensed.

    

### [[2112.14309] PowerTCP: Pushing the Performance Limits of Datacenter Networks](http://arxiv.org/abs/2112.14309)


  Increasingly stringent throughput and latency requirements in datacenter
networks demand fast and accurate congestion control. We observe that the
reaction time and accuracy of existing datacenter congestion control schemes
are inherently limited. They either rely only on explicit feedback about the
network state (e.g., queue lengths in DCTCP) or only on variations of state
(e.g., RTT gradient in TIMELY). To overcome these limitations, we propose a
novel congestion control algorithm, PowerTCP, which achieves much more
fine-grained congestion control by adapting to the bandwidth-window product
(henceforth called power). PowerTCP leverages in-band network telemetry to
react to changes in the network instantaneously without loss of throughput and
while keeping queues short. Due to its fast reaction time, our algorithm is
particularly well-suited for dynamic network environments and bursty traffic
patterns. We show analytically and empirically that PowerTCP can significantly
outperform the state-of-the-art in both traditional datacenter topologies and
emerging reconfigurable datacenters where frequent bandwidth changes make
congestion control challenging. In traditional datacenter networks, PowerTCP
reduces tail flow completion times of short flows by 80% compared to DCQCN and
TIMELY, and by 33% compared to HPCC even at 60% network load. In reconfigurable
datacenters, PowerTCP achieves 85% circuit utilization without incurring
additional latency and cuts tail latency by at least 2x compared to existing
approaches.

    

### [[2112.14328] QUIC Throughput and Fairness over Dual Connectivity (extended)](http://arxiv.org/abs/2112.14328)


  Dual Connectivity (DC) is an important lower-layer feature accelerating the
transition from 4G to 5G that also is expected to play an important role in
standalone 5G. However, even though the packet reordering introduced by DC can
significantly impact the performance of upper-layer protocols, no prior work
has studied the impact of DC on QUIC. In this paper, we present the first such
performance study. Using a series of throughput and fairness experiments, we
show how QUIC is affected by different DC parameters, network conditions, and
whether the DC implementation aims to improve throughput or reliability.
Results for two QUIC implementations (aioquic, ngtcp2) and two congestion
control algorithms (NewReno, CUBIC) are presented under both static and highly
time-varying network conditions. Our findings provide insights into the impacts
of splitting QUIC traffic in a DC environment. With reasonably selected DC
parameters and increased UDP receive buffers, QUIC over DC performs similarly
to TCP over DC and achieves optimal fairness under symmetric link conditions
when DC is not used for packet duplication. The insights can help network
operators provide modern users better end-to-end service when deploying DC.

    

### [[2112.14423] Machine Learning Methods for Spectral Efficiency Prediction in Massive MIMO Systems](http://arxiv.org/abs/2112.14423)


  Channel decoding, channel detection, channel assessment, and resource
management for wireless multiple-input multiple-output (MIMO) systems are all
examples of problems where machine learning (ML) can be successfully applied.
In this paper, we study several ML approaches to solve the problem of
estimating the spectral efficiency (SE) value for a certain precoding scheme,
preferably in the shortest possible time. The best results in terms of mean
average percentage error (MAPE) are obtained with gradient boosting over sorted
features, while linear models demonstrate worse prediction quality. Neural
networks perform similarly to gradient boosting, but they are more resource-
and time-consuming because of hyperparameter tuning and frequent retraining. We
investigate the practical applicability of the proposed algorithms in a wide
range of scenarios generated by the Quadriga simulator. In almost all
scenarios, the MAPE achieved using gradient boosting and neural networks is
less than 10\%.

    

### [[2112.14457] Stochastic dynamic matching: A mixed graph-theory and linear-algebra approach](http://arxiv.org/abs/2112.14457)


  The stochastic dynamic matching problem has recently drawn attention in the
stochastic-modeling community due to its numerous applications, ranging from
supply-chain management to kidney exchange programs. In this paper, we consider
a matching problem in which items of different classes arrive according to
independent Poisson processes. Unmatched items are stored in a queue, and
compatibility constraints are described by a simple graph on the classes, so
that two items can be matched if their classes are neighbors in the graph. We
analyze the efficiency of matching policies, not only in terms of system
stability, but also in terms of matching rates between different classes. Our
results rely on the observation that, under any stable policy, the matching
rates satisfy a conservation equation that equates the arrival and departure
rates of each item class. Our main contributions are threefold. We first
introduce a mapping between the dimension of the solution set of this
conservation equation, the structure of the compatibility graph, and the
existence of a stable policy. In particular, this allows us to derive a
necessary and sufficient stability condition that is verifiable in polynomial
time. Secondly, we describe the convex polytope of non-negative solutions of
the conservation equation. When this polytope is reduced to a single point, we
give a closed-form expression of the solution; in general, we characterize the
vertices of this polytope using again the graph structure. Lastly, we show that
greedy policies cannot, in general, achieve every point in the polytope. In
contrast, non-greedy policies can reach any point of the interior of this
polytope, and we give a condition for these policies to also reach the boundary
of the polytope.

    

### [[2112.14469] Physical Layer Security Techniques for Future Wireless Networks](http://arxiv.org/abs/2112.14469)


  The broadcast nature of wireless communication systems makes wireless
transmission extremely susceptible to eavesdropping and even malicious
interference. Physical layer security technology can effectively protect the
private information sent by the transmitter from being listened to by illegal
eavesdroppers, thus ensuring the privacy and security of communication between
the transmitter and legitimate users. The development of mobile communication
presents new challenges to physical layer security research. This paper
provides a comprehensive survey of the physical layer security research on
various promising mobile technologies, including directional modulation (DM),
spatial modulation (SM), covert communication, intelligent reflecting surface
(IRS)-aided communication, and so on. Finally, future trends and the unresolved
technical challenges are summarized in physical layer security for mobile
communications.

    

### [[2112.14551] Altitude Optimization of UAV Base Stations from Satellite Images Using Deep Neural Network](http://arxiv.org/abs/2112.14551)


  It is expected that unmanned aerial vehicles (UAVs) will play a vital role in
future communication systems. Optimum positioning of UAVs, serving as base
stations, can be done through extensive field measurements or ray tracing
simulations when the 3D model of the region of interest is available. In this
paper, we present an alternative approach to optimize UAV base station altitude
for a region. The approach is based on deep learning; specifically, a 2D
satellite image of the target region is input to a deep neural network to
predict path loss distributions for different UAV altitudes. The predicted path
distributions are used to calculate the coverage in the region; and the optimum
altitude, maximizing the coverage, is determined. The neural network is
designed and trained to produce multiple path loss distributions in a single
inference; thus, it is not necessary to train a separate network for each
altitude.

    

### [[2112.14612] Managing Home Routers with NETCONF over TLS and NETCONF Call Home](http://arxiv.org/abs/2112.14612)


  The Network Configuration (NETCONF) protocol and the associated YANG data
modeling language are the foundations of contemporary network management
frameworks evolving within the Internet Engineering Task Force (IETF). netopeer
(a NETCONF server) and ncclient (a NETCONF client) are popular open-source
projects that support the latest NETCONF v1.1 protocol using the mandatory
Secure Shell (SSH) transport. We recently implemented and integrated NETCONF
over Transport Layer Security (TLS) transport and NETCONF Call Home (CH)
mechanisms using reverse TLS and SSH in both projects. The CH mechanism allows
a managed device behind a Network Address Translation (NAT) running a NETCONF
server (netopeer) to successfully establish a NETCONF session with a Network
Management System (NMS) running a NETCONF client (ncclient). In this article,
we describe how these standards allow home routers and NAT boxes (in
particular) to be managed using these latest additions to the NETCONF protocol.

    

### [[2112.14618] IoT Security Challenges and Mitigations: An Introduction](http://arxiv.org/abs/2112.14618)


  The use of IoT in society is perhaps already ubiquitous, with a vast attack
surface offering multiple opportunities for malicious actors. This short paper
first presents an introduction to IoT and its security issues, including an
overview of IoT layer models and topologies, IoT standardisation efforts and
protocols. The focus then moves to IoT vulnerabilities and specific suggestions
for mitigations. This work's intended audience are those relatively new to IoT
though with existing network-related knowledge. It is concluded that device
resource constraints and a lack of IoT standards are significant issues.
Research opportunities exist to develop efficient IoT IDS and energy-saving
cryptography techniques lightweight enough to reasonably deploy. The need for
standardised protocols and channel-based security solutions is clear,
underpinned by legislative directives to ensure high standards that prevent
cost-cutting on the device manufacturing side.

    

### [[2112.14732] Optimal Weighted Load Balancing in TCAMs](http://arxiv.org/abs/2112.14732)


  Traffic splitting is a required functionality in networks, for example for
load balancing over multiple paths or among different servers. The capacities
of the servers determine the partition by which traffic should be split. A
recent approach implements traffic splitting within the ternary content
addressable memory (TCAM), which is often available in switches. It is
important to reduce the amount of memory allocated for this task since TCAMs
are power consuming and are often also required for other tasks such as
classification and routing. Previous work showed how to compute the smallest
prefix-matching TCAM necessary to implement a given partition exactly. In this
paper we solve the more practical case, where at most $n$ prefix-matching TCAM
rules are available, restricting the ability to implement exactly the desired
partition. We give simple and efficient algorithms to find $n$ rules that
generate a partition closest in $L_\infty$ to the desired one. We do the same
for a one-sided version of $L_\infty$ which equals to the maximum overload on a
server and for a relative version of it. We use our algorithms to evaluate how
the expected error changes as a function of the number of rules, the number of
servers, and the width of the TCAM.

    

### [[1803.03914] Optimized Dynamic Cache Instantiation and Accurate LRU Approximations under Time-varying Request Volume](http://arxiv.org/abs/1803.03914)


  Content-delivery applications can achieve scalability and reduce wide-area
network traffic using geographically distributed caches. However, each deployed
cache has an associated cost, and under time-varying request rates (e.g., a
daily cycle) there may be long periods when the request rate from the local
region is not high enough to justify this cost. Cloud computing offers a
solution to problems of this kind, by supporting dynamic allocation and release
of resources. In this paper, we analyze the potential benefits from dynamically
instantiating caches using resources from cloud service providers. We develop
novel analytic caching models that accommodate time-varying request rates,
transient behavior as a cache fills following instantiation, and selective
cache insertion policies. Within the context of a simple cost model, we then
develop bounds and compare policies with optimized parameter selections to
obtain insights into key cost/performance tradeoffs. We find that dynamic cache
instantiation can provide substantial cost reductions, that potential
reductions strongly dependent on the object popularity skew, and that selective
cache insertion can be even more beneficial in this context than with
conventional edge caches. Finally, our contributions also include accurate and
easy-to-compute approximations that are shown applicable to LRU caches under
time-varying workloads.

    

### [[1906.09779] Cross-user Similarities in Viewing Behavior for 360$^{\circ}$ Video and Caching Implications](http://arxiv.org/abs/1906.09779)


  The demand and usage of 360$^{\circ}$ video services are expected to
increase. However, despite these services being highly bandwidth intensive, not
much is known about the potential value that basic bandwidth saving techniques
such as server or edge-network on-demand caching (e.g., in a CDN) could have
when used for delivery of such services. This problem is both important and
complicated as client-side solutions have been developed that split the full
360$^{\circ}$ view into multiple tiles, and adapt the quality of the downloaded
tiles based on the user's expected viewing direction and bandwidth conditions.
This paper presents new trace-based analysis methods that incorporate users'
viewports (the area of the full 360$^{\circ}$ view the user actually sees), a
first characterization of the cross-user similarities of the users' viewports,
and a trace-based analysis of the potential bandwidth savings that
caching-based techniques may offer under different conditions. Our analysis
takes into account differences in the time granularity over which viewport
overlaps can be beneficial for resource saving techniques, compares and
contrasts differences between video categories, and accounts for uncertainties
in the network conditions and the prediction of the future viewing direction
when prefetching. The results provide substantial insight into the conditions
under which overlap can be considerable and caching effective, and inform the
design of new caching system policies tailored for 360$^{\circ}$ video.

    

### [[2112.13843] BMPQ: Bit-Gradient Sensitivity Driven Mixed-Precision Quantization of DNNs from Scratch](http://arxiv.org/abs/2112.13843)


  Large DNNs with mixed-precision quantization can achieve ultra-high
compression while retaining high classification performance. However, because
of the challenges in finding an accurate metric that can guide the optimization
process, these methods either sacrifice significant performance compared to the
32-bit floating-point (FP-32) baseline or rely on a compute-expensive,
iterative training policy that requires the availability of a pre-trained
baseline. To address this issue, this paper presents BMPQ, a training method
that uses bit gradients to analyze layer sensitivities and yield
mixed-precision quantized models. BMPQ requires a single training iteration but
does not need a pre-trained baseline. It uses an integer linear program (ILP)
to dynamically adjust the precision of layers during training, subject to a
fixed hardware budget. To evaluate the efficacy of BMPQ, we conduct extensive
experiments with VGG16 and ResNet18 on CIFAR-10, CIFAR-100, and Tiny-ImageNet
datasets. Compared to the baseline FP-32 models, BMPQ can yield models that
have 15.4x fewer parameter bits with a negligible drop in accuracy. Compared to
the SOTA "during training", mixed-precision training scheme, our models are
2.1x, 2.2x, and 2.9x smaller, on CIFAR-10, CIFAR-100, and Tiny-ImageNet,
respectively, with an improved accuracy of up to 14.54%.

    

### [[2112.13845] Raw Produce Quality Detection with Shifted Window Self-Attention](http://arxiv.org/abs/2112.13845)


  Global food insecurity is expected to worsen in the coming decades with the
accelerated rate of climate change and the rapidly increasing population. In
this vein, it is important to remove inefficiencies at every level of food
production. The recent advances in deep learning can help reduce such
inefficiencies, yet their application has not yet become mainstream throughout
the industry, inducing economic costs at a massive scale. To this point, modern
techniques such as CNNs (Convolutional Neural Networks) have been applied to
RPQD (Raw Produce Quality Detection) tasks. On the other hand, Transformer's
successful debut in the vision among other modalities led us to expect a better
performance with these Transformer-based models in RPQD. In this work, we
exclusively investigate the recent state-of-the-art Swin (Shifted Windows)
Transformer which computes self-attention in both intra- and inter-window
fashion. We compare Swin Transformer against CNN models on four RPQD image
datasets, each containing different kinds of raw produce: fruits and
vegetables, fish, pork, and beef. We observe that Swin Transformer not only
achieves better or competitive performance but also is data- and
compute-efficient, making it ideal for actual deployment in real-world setting.
To the best of our knowledge, this is the first large-scale empirical study on
RPQD task, which we hope will gain more attention in future works.

    

### [[2112.13850] Using maps to predict economic activity](http://arxiv.org/abs/2112.13850)


  We introduce a novel machine learning approach to leverage historical and
contemporary maps to systematically predict economic statistics. Remote sensing
data have been used as reliable proxies for local economic activity. However,
they have only become available in recent years, thus limiting their
applicability for long-term analysis. Historical maps, on the other hand, date
back several decades. Our simple algorithm extracts meaningful features from
the maps based on their color compositions. The grid-level population
predictions by our approach outperform the conventional CNN-based predictions
using raw map images. It also predicts population better than other approaches
using night light satellite images or land cover classifications as the input
for predictions.

    

### [[2112.13865] Astronomical Image Colorization and upscaling with Generative Adversarial Networks](http://arxiv.org/abs/2112.13865)


  Automatic colorization of images without human intervention has been a
subject of interest in the machine learning community for a brief period of
time. Assigning color to an image is a highly ill-posed problem because of its
innate nature of possessing very high degrees of freedom; given an image, there
is often no single color-combination that is correct. Besides colorization,
another problem in reconstruction of images is Single Image Super Resolution,
which aims at transforming low resolution images to a higher resolution. This
research aims to provide an automated approach for the problem by focusing on a
very specific domain of images, namely astronomical images, and process them
using Generative Adversarial Networks (GANs). We explore the usage of various
models in two different color spaces, RGB and L*a*b. We use transferred
learning owing to a small data set, using pre-trained ResNet-18 as a backbone,
i.e. encoder for the U-net and fine-tune it further. The model produces
visually appealing images which hallucinate high resolution, colorized data in
these results which does not exist in the original image. We present our
results by evaluating the GANs quantitatively using distance metrics such as L1
distance and L2 distance in each of the color spaces across all channels to
provide a comparative analysis. We use Frechet inception distance (FID) to
compare the distribution of the generated images with the distribution of the
real image to assess the model's performance.

    

### [[2112.13867] Depth and Feature Learning are Provably Beneficial for Neural Network Discriminators](http://arxiv.org/abs/2112.13867)


  We construct pairs of distributions $\mu_d, \nu_d$ on $\mathbb{R}^d$ such
that the quantity $|\mathbb{E}_{x \sim \mu_d} [F(x)] - \mathbb{E}_{x \sim
\nu_d} [F(x)]|$ decreases as $\Omega(1/d^2)$ for some three-layer ReLU network
$F$ with polynomial width and weights, while declining exponentially in $d$ if
$F$ is any two-layer network with polynomial weights. This shows that deep GAN
discriminators are able to distinguish distributions that shallow
discriminators cannot. Analogously, we build pairs of distributions $\mu_d,
\nu_d$ on $\mathbb{R}^d$ such that $|\mathbb{E}_{x \sim \mu_d} [F(x)] -
\mathbb{E}_{x \sim \nu_d} [F(x)]|$ decreases as $\Omega(1/(d\log d))$ for
two-layer ReLU networks with polynomial weights, while declining exponentially
for bounded-norm functions in the associated RKHS. This confirms that feature
learning is beneficial for discriminators. Our bounds are based on Fourier
transforms.

    

### [[2112.13893] Non-Reference Quality Monitoring of Digital Images using Gradient Statistics and Feedforward Neural Networks](http://arxiv.org/abs/2112.13893)


  Digital images contain a lot of redundancies, therefore, compressions are
applied to reduce the image size without the loss of reasonable image quality.
The same become more prominent in the case of videos that contains image
sequences and higher compression ratios are achieved in low throughput
networks. Assessment of the quality of images in such scenarios becomes of
particular interest. Subjective evaluation in most of the scenarios becomes
infeasible so objective evaluation is preferred. Among the three objective
quality measures, full-reference and reduced-reference methods require an
original image in some form to calculate the quality score which is not
feasible in scenarios such as broadcasting or IP video. Therefore, a
non-reference quality metric is proposed to assess the quality of digital
images which calculates luminance and multiscale gradient statistics along with
mean subtracted contrast normalized products as features to train a Feedforward
Neural Network with Scaled Conjugate Gradient. The trained network has provided
good regression and R2 measures and further testing on LIVE Image Quality
Assessment database release-2 has shown promising results. Pearson, Kendall,
and Spearman's correlation are calculated between predicted and actual quality
scores and their results are comparable to the state-of-the-art systems.
Moreover, the proposed metric is computationally faster than its counterparts
and can be used for the quality assessment of image sequences.

    

### [[2112.13896] Two Sparsities Are Better Than One: Unlocking the Performance Benefits of Sparse-Sparse Networks](http://arxiv.org/abs/2112.13896)


  In principle, sparse neural networks should be significantly more efficient
than traditional dense networks. Neurons in the brain exhibit two types of
sparsity; they are sparsely interconnected and sparsely active. These two types
of sparsity, called weight sparsity and activation sparsity, when combined,
offer the potential to reduce the computational cost of neural networks by two
orders of magnitude. Despite this potential, today's neural networks deliver
only modest performance benefits using just weight sparsity, because
traditional computing hardware cannot efficiently process sparse networks. In
this article we introduce Complementary Sparsity, a novel technique that
significantly improves the performance of dual sparse networks on existing
hardware. We demonstrate that we can achieve high performance running
weight-sparse networks, and we can multiply those speedups by incorporating
activation sparsity. Using Complementary Sparsity, we show up to 100X
improvement in throughput and energy efficiency performing inference on FPGAs.
We analyze scalability and resource tradeoffs for a variety of kernels typical
of commercial convolutional networks such as ResNet-50 and MobileNetV2. Our
results with Complementary Sparsity suggest that weight plus activation
sparsity can be a potent combination for efficiently scaling future AI models.

    

### [[2112.13901] Expected hypervolume improvement for simultaneous multi-objective and multi-fidelity optimization](http://arxiv.org/abs/2112.13901)


  Bayesian optimization has proven to be an efficient method to optimize
expensive-to-evaluate systems. However, depending on the cost of single
observations, multi-dimensional optimizations of one or more objectives may
still be prohibitively expensive. Multi-fidelity optimization remedies this
issue by including multiple, cheaper information sources such as low-resolution
approximations in numerical simulations. Acquisition functions for
multi-fidelity optimization are typically based on exploration-heavy algorithms
that are difficult to combine with optimization towards multiple objectives.
Here we show that the expected hypervolume improvement policy can act in many
situations as a suitable substitute. We incorporate the evaluation cost either
via a two-step evaluation or within a single acquisition function with an
additional fidelity-related objective. This permits simultaneous
multi-objective and multi-fidelity optimization, which allows to accurately
establish the Pareto set and front at fractional cost. Benchmarks show a cost
reduction of an order of an order of magnitude or more. Our method thus allows
for Pareto optimization of extremely expansive black-box functions.
The presented methods are simple and straightforward to implement in
existing, optimized Bayesian optimization frameworks and can immediately be
extended to batch optimization. The techniques can also be used to combine
different continuous and/or discrete fidelity dimensions, which makes them
particularly relevant for simulation problems in plasma physics, fluid dynamics
and many other branches of scientific computing.

    

### [[2112.13906] Does CLIP Benefit Visual Question Answering in the Medical Domain as Much as it Does in the General Domain?](http://arxiv.org/abs/2112.13906)


  Contrastive Language--Image Pre-training (CLIP) has shown remarkable success
in learning with cross-modal supervision from extensive amounts of image--text
pairs collected online. Thus far, the effectiveness of CLIP has been
investigated primarily in general-domain multimodal problems. This work
evaluates the effectiveness of CLIP for the task of Medical Visual Question
Answering (MedVQA). To this end, we present PubMedCLIP, a fine-tuned version of
CLIP for the medical domain based on PubMed articles. Our experiments are
conducted on two MedVQA benchmark datasets and investigate two MedVQA methods,
MEVF (Mixture of Enhanced Visual Features) and QCR (Question answering via
Conditional Reasoning). For each of these, we assess the merits of visual
representation learning using PubMedCLIP, the original CLIP, and
state-of-the-art MAML (Model-Agnostic Meta-Learning) networks pre-trained only
on visual data. We open source the code for our MedVQA pipeline and
pre-training PubMedCLIP. CLIP and PubMedCLIP achieve improvements in comparison
to MAML's visual encoder. PubMedCLIP achieves the best results with gains in
the overall accuracy of up to 3%. Individual examples illustrate the strengths
of PubMedCLIP in comparison to the previously widely used MAML networks. Visual
representation learning with language supervision in PubMedCLIP leads to
noticeable improvements for MedVQA. Our experiments reveal distributional
differences in the two MedVQA benchmark datasets that have not been imparted in
previous work and cause different back-end visual encoders in PubMedCLIP to
exhibit different behavior on these datasets. Moreover, we witness fundamental
performance differences of VQA in general versus medical domains.

    

### [[2112.13922] Predicting Breakdown Risk Based on Historical Maintenance Data for Air Force Ground Vehicles](http://arxiv.org/abs/2112.13922)


  Unscheduled maintenance has contributed to longer downtime for vehicles and
increased costs for Logistic Readiness Squadrons (LRSs) in the Air Force. When
vehicles are in need of repair outside of their scheduled time, depending on
their priority level, the entire squadron's slated repair schedule is
transformed negatively. The repercussions of unscheduled maintenance are
specifically seen in the increase of man hours required to maintain vehicles
that should have been working well: this can include more man hours spent on
maintenance itself, waiting for parts to arrive, hours spent re-organizing the
repair schedule, and more. The dominant trend in the current maintenance system
at LRSs is that they do not have predictive maintenance infrastructure to
counteract the influx of unscheduled repairs they experience currently, and as
a result, their readiness and performance levels are lower than desired.
We use data pulled from the Defense Property and Accountability System
(DPAS), that the LRSs currently use to store their vehicle maintenance
information. Using historical vehicle maintenance data we receive from DPAS, we
apply three different algorithms independently to construct an accurate
predictive system to optimize maintenance schedules at any given time. Through
the application of Logistics Regression, Random Forest, and Gradient Boosted
Trees algorithms, we found that a Logistic Regression algorithm, fitted to our
data, produced the most accurate results. Our findings indicate that not only
would continuing the use of Logistic Regression be prudent for our research
purposes, but that there is opportunity to further tune and optimize our
Logistic Regression model for higher accuracy.

    

### [[2112.13934] RELDEC: Reinforcement Learning-Based Decoding of Moderate Length LDPC Codes](http://arxiv.org/abs/2112.13934)


  In this work we propose RELDEC, a novel approach for sequential decoding of
moderate length low-density parity-check (LDPC) codes. The main idea behind
RELDEC is that an optimized decoding policy is subsequently obtained via
reinforcement learning based on a Markov decision process (MDP). In contrast to
our previous work, where an agent learns to schedule only a single check node
(CN) within a group (cluster) of CNs per iteration, in this work we train the
agent to schedule all CNs in a cluster, and all clusters in every iteration.
That is, in each learning step of RELDEC an agent learns to schedule CN
clusters sequentially depending on a reward associated with the outcome of
scheduling a particular cluster. We also modify the state space representation
of the MDP, enabling RELDEC to be suitable for larger block length LDPC codes
than those studied in our previous work. Furthermore, to address decoding under
varying channel conditions, we propose two related schemes, namely, agile
meta-RELDEC (AM-RELDEC) and meta-RELDEC (M-RELDEC), both of which employ
meta-reinforcement learning. The proposed RELDEC scheme significantly
outperforms standard flooding and random sequential decoding for a variety of
LDPC codes, including codes designed for 5G new radio.

    

### [[2112.13935] AET-SGD: Asynchronous Event-triggered Stochastic Gradient Descent](http://arxiv.org/abs/2112.13935)


  Communication cost is the main bottleneck for the design of effective
distributed learning algorithms. Recently, event-triggered techniques have been
proposed to reduce the exchanged information among compute nodes and thus
alleviate the communication cost. However, most existing event-triggered
approaches only consider heuristic event-triggered thresholds. They also ignore
the impact of computation and network delay, which play an important role on
the training performance. In this paper, we propose an Asynchronous
Event-triggered Stochastic Gradient Descent (SGD) framework, called AET-SGD, to
i) reduce the communication cost among the compute nodes, and ii) mitigate the
impact of the delay. Compared with baseline event-triggered methods, AET-SGD
employs a linear increasing sample size event-triggered threshold, and can
significantly reduce the communication cost while keeping good convergence
performance. We implement AET-SGD and evaluate its performance on multiple
representative data sets, including MNIST, FashionMNIST, KMNIST and CIFAR10.
The experimental results validate the correctness of the design and show a
significant communication cost reduction from 44x to 120x, compared to the
state of the art. Our results also show that AET-SGD can resist large delay
from the straggler nodes while obtaining a decent performance and a desired
speedup ratio.

    

### [[2112.13939] SPIDER: Searching Personalized Neural Architecture for Federated Learning](http://arxiv.org/abs/2112.13939)


  Federated learning (FL) is an efficient learning framework that assists
distributed machine learning when data cannot be shared with a centralized
server due to privacy and regulatory restrictions. Recent advancements in FL
use predefined architecture-based learning for all the clients. However, given
that clients' data are invisible to the server and data distributions are
non-identical across clients, a predefined architecture discovered in a
centralized setting may not be an optimal solution for all the clients in FL.
Motivated by this challenge, in this work, we introduce SPIDER, an algorithmic
framework that aims to Search Personalized neural architecture for federated
learning. SPIDER is designed based on two unique features: (1) alternately
optimizing one architecture-homogeneous global model (Supernet) in a generic FL
manner and one architecture-heterogeneous local model that is connected to the
global model by weight sharing-based regularization (2) achieving
architecture-heterogeneous local model by a novel neural architecture search
(NAS) method that can select optimal subnet progressively using operation-level
perturbation on the accuracy value as the criterion. Experimental results
demonstrate that SPIDER outperforms other state-of-the-art personalization
methods, and the searched personalized architectures are more inference
efficient.

    

### [[2112.13941] Safe Reinforcement Learning with Chance-constrained Model Predictive Control](http://arxiv.org/abs/2112.13941)


  Real-world reinforcement learning (RL) problems often demand that agents
behave safely by obeying a set of designed constraints. We address the
challenge of safe RL by coupling a safety guide based on model predictive
control (MPC) with a modified policy gradient framework in a linear setting
with continuous actions. The guide enforces safe operation of the system by
embedding safety requirements as chance constraints in the MPC formulation. The
policy gradient training step then includes a safety penalty which trains the
base policy to behave safely. We show theoretically that this penalty allows
for the safety guide to be removed after training and illustrate our method
using experiments with a simulator quadrotor.

    

### [[2112.13951] Improving Nonparametric Classification via Local Radial Regression with an Application to Stock Prediction](http://arxiv.org/abs/2112.13951)


  For supervised classification problems, this paper considers estimating the
query's label probability through local regression using observed covariates.
Well-known nonparametric kernel smoother and $k$-nearest neighbor ($k$-NN)
estimator, which take label average over a ball around the query, are
consistent but asymptotically biased particularly for a large radius of the
ball. To eradicate such bias, local polynomial regression (LPoR) and multiscale
$k$-NN (MS-$k$-NN) learn the bias term by local regression around the query and
extrapolate it to the query itself. However, their theoretical optimality has
been shown for the limit of the infinite number of training samples. For
correcting the asymptotic bias with fewer observations, this paper proposes a
local radial regression (LRR) and its logistic regression variant called local
radial logistic regression (LRLR), by combining the advantages of LPoR and
MS-$k$-NN. The idea is simple: we fit the local regression to observed labels
by taking the radial distance as the explanatory variable and then extrapolate
the estimated label probability to zero distance. Our numerical experiments,
including real-world datasets of daily stock indices, demonstrate that LRLR
outperforms LPoR and MS-$k$-NN.

    

### [[2112.13964] Online Allocation with Two-sided Resource Constraints](http://arxiv.org/abs/2112.13964)


  Motivated by many interesting real-world applications in logistics and online
advertising, we consider an online allocation problem subject to lower and
upper resource constraints, where the requests arrive sequentially, sampled
i.i.d. from an unknown distribution, and we need to promptly make a decision
given limited resources and lower bounds requirements. First, with knowledge of
the measure of feasibility, i.e., $\alpha$, we propose a new algorithm that
obtains $1-O(\frac{\epsilon}{\alpha-\epsilon})$ -competitive ratio for the
offline problems that know the entire requests ahead of time. Inspired by the
previous studies, this algorithm adopts an innovative technique to dynamically
update a threshold price vector for making decisions. Moreover, an optimization
method to estimate the optimal measure of feasibility is proposed with
theoretical guarantee at the end of this paper. Based on this method, if we
tolerate slight violation of the lower bounds constraints with parameter
$\eta$, the proposed algorithm is naturally extended to the settings without
strong feasible assumption, which cover the significantly unexplored infeasible
scenarios.

    

### [[2112.13966] Online Adversarial Distillation for Graph Neural Networks](http://arxiv.org/abs/2112.13966)


  Knowledge distillation has recently become a popular technique to improve the
model generalization ability on convolutional neural networks. However, its
effect on graph neural networks is less than satisfactory since the graph
topology and node attributes are likely to change in a dynamic way and in this
case a static teacher model is insufficient in guiding student training. In
this paper, we tackle this challenge by simultaneously training a group of
graph neural networks in an online distillation fashion, where the group
knowledge plays a role as a dynamic virtual teacher and the structure changes
in graph neural networks are effectively captured. To improve the distillation
performance, two types of knowledge are transferred among the students to
enhance each other: local knowledge reflecting information in the graph
topology and node attributes, and global knowledge reflecting the prediction
over classes. We transfer the global knowledge with KL-divergence as the
vanilla knowledge distillation does, while exploiting the complicated structure
of the local knowledge with an efficient adversarial cyclic learning framework.
Extensive experiments verified the effectiveness of our proposed online
adversarial distillation approach.

    

### [[2112.13969] LINDA: Unsupervised Learning to Interpolate in Natural Language Processing](http://arxiv.org/abs/2112.13969)


  Despite the success of mixup in data augmentation, its applicability to
natural language processing (NLP) tasks has been limited due to the discrete
and variable-length nature of natural languages. Recent studies have thus
relied on domain-specific heuristics and manually crafted resources, such as
dictionaries, in order to apply mixup in NLP. In this paper, we instead propose
an unsupervised learning approach to text interpolation for the purpose of data
augmentation, to which we refer as "Learning to INterpolate for Data
Augmentation" (LINDA), that does not require any heuristics nor manually
crafted resources but learns to interpolate between any pair of natural
language sentences over a natural language manifold. After empirically
demonstrating the LINDA's interpolation capability, we show that LINDA indeed
allows us to seamlessly apply mixup in NLP and leads to better generalization
in text classification both in-domain and out-of-domain.

    

### [[2112.13974] A Moment in the Sun: Solar Nowcasting from Multispectral Satellite Data using Self-Supervised Learning](http://arxiv.org/abs/2112.13974)


  Solar energy is now the cheapest form of electricity in history.
Unfortunately, significantly increasing the grid's fraction of solar energy
remains challenging due to its variability, which makes balancing electricity's
supply and demand more difficult. While thermal generators' ramp rate -- the
maximum rate that they can change their output -- is finite, solar's ramp rate
is essentially infinite. Thus, accurate near-term solar forecasting, or
nowcasting, is important to provide advance warning to adjust thermal generator
output in response to solar variations to ensure a balanced supply and demand.
To address the problem, this paper develops a general model for solar
nowcasting from abundant and readily available multispectral satellite data
using self-supervised learning. Specifically, we develop deep auto-regressive
models using convolutional neural networks (CNN) and long short-term memory
networks (LSTM) that are globally trained across multiple locations to predict
raw future observations of the spatio-temporal data collected by the recently
launched GOES-R series of satellites. Our model estimates a location's future
solar irradiance based on satellite observations, which we feed to a regression
model trained on smaller site-specific solar data to provide near-term solar
photovoltaic (PV) forecasts that account for site-specific characteristics. We
evaluate our approach for different coverage areas and forecast horizons across
25 solar sites and show that our approach yields errors close to that of a
model using ground-truth observations.

    

### [[2112.14004] Efficient Performance Bounds for Primal-Dual Reinforcement Learning from Demonstrations](http://arxiv.org/abs/2112.14004)


  We consider large-scale Markov decision processes with an unknown cost
function and address the problem of learning a policy from a finite set of
expert demonstrations. We assume that the learner is not allowed to interact
with the expert and has no access to reinforcement signal of any kind. Existing
inverse reinforcement learning methods come with strong theoretical guarantees,
but are computationally expensive, while state-of-the-art policy optimization
algorithms achieve significant empirical success, but are hampered by limited
theoretical understanding. To bridge the gap between theory and practice, we
introduce a novel bilinear saddle-point framework using Lagrangian duality. The
proposed primal-dual viewpoint allows us to develop a model-free provably
efficient algorithm through the lens of stochastic convex optimization. The
method enjoys the advantages of simplicity of implementation, low memory
requirements, and computational and sample complexities independent of the
number of states. We further present an equivalent no-regret online-learning
interpretation.

    

### [[2112.14011] To Supervise or Not: How to Effectively Learn Wireless Interference Management Models?](http://arxiv.org/abs/2112.14011)


  Machine learning has become successful in solving wireless interference
management problems. Different kinds of deep neural networks (DNNs) have been
trained to accomplish key tasks such as power control, beamforming and
admission control. There are two popular training paradigms for such DNNs-based
interference management models: supervised learning (i.e., fitting labels
generated by an optimization algorithm) and unsupervised learning (i.e.,
directly optimizing some system performance measure). Although both of these
paradigms have been extensively applied in practice, due to the lack of any
theoretical understanding about these methods, it is not clear how to
systematically understand and compare their performance.
In this work, we conduct theoretical studies to provide some in-depth
understanding about these two training paradigms. First, we show a somewhat
surprising result, that for some special power control problem, the
unsupervised learning can perform much worse than its supervised counterpart,
because it is more likely to stuck at some low-quality local solutions. We then
provide a series of theoretical results to further understand the properties of
the two approaches. Generally speaking, we show that when high-quality labels
are available, then the supervised learning is less likely to be stuck at a
solution than its unsupervised counterpart. Additionally, we develop a
semi-supervised learning approach which properly integrates these two training
paradigms, and can effectively utilize limited number of labels to find
high-quality solutions. To our knowledge, these are the first set of
theoretical results trying to understand different training approaches in
learning-based wireless communication system design.

    

### [[2112.14012] Solving time dependent Fokker-Planck equations via temporal normalizing flow](http://arxiv.org/abs/2112.14012)


  In this work, we propose an adaptive learning approach based on temporal
normalizing flows for solving time-dependent Fokker-Planck (TFP) equations. It
is well known that solutions of such equations are probability density
functions, and thus our approach relies on modelling the target solutions with
the temporal normalizing flows. The temporal normalizing flow is then trained
based on the TFP loss function, without requiring any labeled data. Being a
machine learning scheme, the proposed approach is mesh-free and can be easily
applied to high dimensional problems. We present a variety of test problems to
show the effectiveness of the learning approach.

    

### [[2112.14021] Multilayer Graph Contrastive Clustering Network](http://arxiv.org/abs/2112.14021)


  Multilayer graph has garnered plenty of research attention in many areas due
to their high utility in modeling interdependent systems. However, clustering
of multilayer graph, which aims at dividing the graph nodes into categories or
communities, is still at a nascent stage. Existing methods are often limited to
exploiting the multiview attributes or multiple networks and ignoring more
complex and richer network frameworks. To this end, we propose a generic and
effective autoencoder framework for multilayer graph clustering named
Multilayer Graph Contrastive Clustering Network (MGCCN). MGCCN consists of
three modules: (1)Attention mechanism is applied to better capture the
relevance between nodes and neighbors for better node embeddings. (2)To better
explore the consistent information in different networks, a contrastive fusion
strategy is introduced. (3)MGCCN employs a self-supervised component that
iteratively strengthens the node embedding and clustering. Extensive
experiments on different types of real-world graph data indicate that our
proposed method outperforms state-of-the-art techniques.

    

### [[2112.14040] Deep neural networks for solving forward and inverse problems of (2+1)-dimensional nonlinear wave equations with rational solitons](http://arxiv.org/abs/2112.14040)


  In this paper, we investigate the forward problems on the data-driven
rational solitons for the (2+1)-dimensional KP-I equation and spin-nonlinear
Schrödinger (spin-NLS) equation via the deep neural networks leaning.
Moreover, the inverse problems of the (2+1)-dimensional KP-I equation and
spin-NLS equation are studied via deep learning. The main idea of the
data-driven forward and inverse problems is to use the deep neural networks
with the activation function to approximate the solutions of the considered
(2+1)-dimensional nonlinear wave equations by optimizing the chosen loss
functions related to the considered nonlinear wave equations.

    

### [[2112.14061] Investigating Shifts in GAN Output-Distributions](http://arxiv.org/abs/2112.14061)


  A fundamental and still largely unsolved question in the context of
Generative Adversarial Networks is whether they are truly able to capture the
real data distribution and, consequently, to sample from it. In particular, the
multidimensional nature of image distributions leads to a complex evaluation of
the diversity of GAN distributions. Existing approaches provide only a partial
understanding of this issue, leaving the question unanswered. In this work, we
introduce a loop-training scheme for the systematic investigation of observable
shifts between the distributions of real training data and GAN generated data.
Additionally, we introduce several bounded measures for distribution shifts,
which are both easy to compute and to interpret. Overall, the combination of
these methods allows an explorative investigation of innate limitations of
current GAN algorithms. Our experiments on different data-sets and multiple
state-of-the-art GAN architectures show large shifts between input and output
distributions, showing that existing theoretical guarantees towards the
convergence of output distributions appear not to be holding in practice.

    

### [[2112.14075] Financial Vision Based Differential Privacy Applications](http://arxiv.org/abs/2112.14075)


  The importance of deep learning data privacy has gained significant attention
in recent years. It is probably to suffer data breaches when applying deep
learning to cryptocurrency that lacks supervision of financial regulatory
agencies. However, there is little relative research in the financial area to
our best knowledge. We apply two representative deep learning privacy-privacy
frameworks proposed by Google to financial trading data. We designed the
experiments with several different parameters suggested from the original
studies. In addition, we refer the degree of privacy to Google and Apple
companies to estimate the results more reasonably. The results show that DP-SGD
performs better than the PATE framework in financial trading data. The tradeoff
between privacy and accuracy is low in DP-SGD. The degree of privacy also is in
line with the actual case. Therefore, we can obtain a strong privacy guarantee
with precision to avoid potential financial loss.

    

### [[2112.14108] Fostering the Robustness of White-Box Deep Neural Network Watermarks by Neuron Alignment](http://arxiv.org/abs/2112.14108)


  The wide application of deep learning techniques is boosting the regulation
of deep learning models, especially deep neural networks (DNN), as commercial
products. A necessary prerequisite for such regulations is identifying the
owner of deep neural networks, which is usually done through the watermark.
Current DNN watermarking schemes, particularly white-box ones, are uniformly
fragile against a family of functionality equivalence attacks, especially the
neuron permutation. This operation can effortlessly invalidate the ownership
proof and escape copyright regulations. To enhance the robustness of white-box
DNN watermarking schemes, this paper presents a procedure that aligns neurons
into the same order as when the watermark is embedded, so the watermark can be
correctly recognized. This neuron alignment process significantly facilitates
the functionality of established deep neural network watermarking schemes.

    

### [[2112.14146] Towards continual task learning in artificial neural networks: current approaches and insights from neuroscience](http://arxiv.org/abs/2112.14146)


  The innate capacity of humans and other animals to learn a diverse, and often
interfering, range of knowledge and skills throughout their lifespan is a
hallmark of natural intelligence, with obvious evolutionary motivations. In
parallel, the ability of artificial neural networks (ANNs) to learn across a
range of tasks and domains, combining and re-using learned representations
where required, is a clear goal of artificial intelligence. This capacity,
widely described as continual learning, has become a prolific subfield of
research in machine learning. Despite the numerous successes of deep learning
in recent years, across domains ranging from image recognition to machine
translation, such continual task learning has proved challenging. Neural
networks trained on multiple tasks in sequence with stochastic gradient descent
often suffer from representational interference, whereby the learned weights
for a given task effectively overwrite those of previous tasks in a process
termed catastrophic forgetting. This represents a major impediment to the
development of more generalised artificial learning systems, capable of
accumulating knowledge over time and task space, in a manner analogous to
humans. A repository of selected papers and implementations accompanying this
review can be found at this https URL.

    

### [[2112.14159] Skin feature point tracking using deep feature encodings](http://arxiv.org/abs/2112.14159)


  Facial feature tracking is a key component of imaging ballistocardiography
(BCG) where accurate quantification of the displacement of facial keypoints is
needed for good heart rate estimation. Skin feature tracking enables
video-based quantification of motor degradation in Parkinson's disease.
Traditional computer vision algorithms include Scale Invariant Feature
Transform (SIFT), Speeded-Up Robust Features (SURF), and Lucas-Kanade method
(LK). These have long represented the state-of-the-art in efficiency and
accuracy but fail when common deformations, like affine local transformations
or illumination changes, are present.
Over the past five years, deep convolutional neural networks have
outperformed traditional methods for most computer vision tasks. We propose a
pipeline for feature tracking, that applies a convolutional stacked autoencoder
to identify the most similar crop in an image to a reference crop containing
the feature of interest. The autoencoder learns to represent image crops into
deep feature encodings specific to the object category it is trained on.
We train the autoencoder on facial images and validate its ability to track
skin features in general using manually labeled face and hand videos. The
tracking errors of distinctive skin features (moles) are so small that we
cannot exclude that they stem from the manual labelling based on a
$\chi^2$-test. With a mean error of 0.6-4.2 pixels, our method outperformed the
other methods in all but one scenario. More importantly, our method was the
only one to not diverge.
We conclude that our method creates better feature descriptors for feature
tracking, feature matching, and image registration than the traditional
algorithms.

    

### [[2112.14195] Exponential Family Model-Based Reinforcement Learning via Score Matching](http://arxiv.org/abs/2112.14195)


  We propose an optimistic model-based algorithm, dubbed SMRL, for
finite-horizon episodic reinforcement learning (RL) when the transition model
is specified by exponential family distributions with $d$ parameters and the
reward is bounded and known. SMRL uses score matching, an unnormalized density
estimation technique that enables efficient estimation of the model parameter
by ridge regression. Under standard regularity assumptions, SMRL achieves
$\tilde O(d\sqrt{H^3T})$ online regret, where $H$ is the length of each episode
and $T$ is the total number of interactions (ignoring polynomial dependence on
structural scale parameters).

    

### [[2112.14204] Non-Convex Joint Community Detection and Group Synchronization via Generalized Power Method](http://arxiv.org/abs/2112.14204)


  This paper proposes a Generalized Power Method (GPM) to tackle the problem of
community detection and group synchronization simultaneously in a direct
non-convex manner. Under the stochastic group block model (SGBM), theoretical
analysis indicates that the algorithm is able to exactly recover the ground
truth in $O(n\log^2n)$ time, sharply outperforming the benchmark method of
semidefinite programming (SDP) in $O(n^{3.5})$ time. Moreover, a lower bound of
parameters is given as a necessary condition for exact recovery of GPM. The new
bound breaches the information-theoretic threshold for pure community detection
under the stochastic block model (SBM), thus demonstrating the superiority of
our simultaneous optimization algorithm over the trivial two-stage method which
performs the two tasks in succession. We also conduct numerical experiments on
GPM and SDP to evidence and complement our theoretical analysis.

    

### [[2112.14232] Constrained Gradient Descent: A Powerful and Principled Evasion Attack Against Neural Networks](http://arxiv.org/abs/2112.14232)


  Minimal adversarial perturbations added to inputs have been shown to be
effective at fooling deep neural networks. In this paper, we introduce several
innovations that make white-box targeted attacks follow the intuition of the
attacker's goal: to trick the model to assign a higher probability to the
target class than to any other, while staying within a specified distance from
the original input. First, we propose a new loss function that explicitly
captures the goal of targeted attacks, in particular, by using the logits of
all classes instead of just a subset, as is common. We show that Auto-PGD with
this loss function finds more adversarial examples than it does with other
commonly used loss functions. Second, we propose a new attack method that uses
a further developed version of our loss function capturing both the
misclassification objective and the $L_{\infty}$ distance limit $\epsilon$.
This new attack method is relatively 1.5--4.2% more successful on the CIFAR10
dataset and relatively 8.2--14.9% more successful on the ImageNet dataset, than
the next best state-of-the-art attack. We confirm using statistical tests that
our attack outperforms state-of-the-art attacks on different datasets and
values of $\epsilon$ and against different defenses.

    

### [[2112.14233] Learning Across Bandits in High Dimension via Robust Statistics](http://arxiv.org/abs/2112.14233)


  Decision-makers often face the "many bandits" problem, where one must
simultaneously learn across related but heterogeneous contextual bandit
instances. For instance, a large retailer may wish to dynamically learn product
demand across many stores to solve pricing or inventory problems, making it
desirable to learn jointly for stores serving similar customers; alternatively,
a hospital network may wish to dynamically learn patient risk across many
providers to allocate personalized interventions, making it desirable to learn
jointly for hospitals serving similar patient populations. We study the setting
where the unknown parameter in each bandit instance can be decomposed into a
global parameter plus a sparse instance-specific term. Then, we propose a novel
two-stage estimator that exploits this structure in a sample-efficient way by
using a combination of robust statistics (to learn across similar instances)
and LASSO regression (to debias the results). We embed this estimator within a
bandit algorithm, and prove that it improves asymptotic regret bounds in the
context dimension $d$; this improvement is exponential for data-poor instances.
We further demonstrate how our results depend on the underlying network
structure of bandit instances.

    

### [[2112.14238] AdaFocus V2: End-to-End Training of Spatial Dynamic Networks for Video Recognition](http://arxiv.org/abs/2112.14238)


  Recent works have shown that the computational efficiency of video
recognition can be significantly improved by reducing the spatial redundancy.
As a representative work, the adaptive focus method (AdaFocus) has achieved a
favorable trade-off between accuracy and inference speed by dynamically
identifying and attending to the informative regions in each video frame.
However, AdaFocus requires a complicated three-stage training pipeline
(involving reinforcement learning), leading to slow convergence and is
unfriendly to practitioners. This work reformulates the training of AdaFocus as
a simple one-stage algorithm by introducing a differentiable
interpolation-based patch selection operation, enabling efficient end-to-end
optimization. We further present an improved training scheme to address the
issues introduced by the one-stage formulation, including the lack of
supervision, input diversity and training stability. Moreover, a
conditional-exit technique is proposed to perform temporal adaptive computation
on top of AdaFocus without additional training. Extensive experiments on six
benchmark datasets (i.e., ActivityNet, FCVID, Mini-Kinetics,
Something-Something V1&V2, and Jester) demonstrate that our model significantly
outperforms the original AdaFocus and other competitive baselines, while being
considerably more simple and efficient to train. Code is available at
this https URL.

    

### [[2112.14244] Robust Convergence in Federated Learning through Label-wise Clustering](http://arxiv.org/abs/2112.14244)


  Non-IID dataset and heterogeneous environment of the local clients are
regarded as a major issue in Federated Learning (FL), causing a downturn in the
convergence without achieving satisfactory performance. In this paper, we
propose a novel Label-wise clustering algorithm that guarantees the
trainability among geographically dispersed heterogeneous local clients, by
selecting only the local models trained with a dataset that approximates into
uniformly distributed class labels, which is likely to obtain faster
minimization of the loss and increment the accuracy among the FL network.
Through conducting experiments on the suggested six common non-IID scenarios,
we empirically show that the vanilla FL aggregation model is incapable of
gaining robust convergence generating biased pre-trained local models and
drifting the local weights to mislead the trainability in the worst case.
Moreover, we quantitatively estimate the expected performance of the local
models before training, which offers a global server to select the optimal
clients, saving additional computational costs. Ultimately, in order to gain
resolution of the non-convergence in such non-IID situations, we design
clustering algorithms based on local input class labels, accommodating the
diversity and assorting clients that could lead the overall system to attain
the swift convergence as global training continues. Our paper shows that
proposed Label-wise clustering demonstrates prompt and robust convergence
compared to other FL algorithms when local training datasets are non-IID or
coexist with IID through multiple experiments.

    

### [[2112.14249] A Finite Sample Theorem for Longitudinal Causal Inference with Machine Learning: Long Term, Dynamic, and Mediated Effects](http://arxiv.org/abs/2112.14249)


  I construct and justify confidence intervals for longitudinal causal
parameters estimated with machine learning. Longitudinal parameters include
long term, dynamic, and mediated effects. I provide a nonasymptotic theorem for
any longitudinal causal parameter estimated with any machine learning algorithm
that satisfies a few simple, interpretable conditions. The main result
encompasses local parameters defined for specific demographics as well as
proximal parameters defined in the presence of unobserved confounding.
Formally, I prove consistency, Gaussian approximation, and semiparametric
efficiency. The rate of convergence is $n^{-1/2}$ for global parameters, and it
degrades gracefully for local parameters. I articulate a simple set of
conditions to translate mean square rates into statistical inference. A key
feature of the main result is a new multiple robustness to ill posedness for
proximal causal inference in longitudinal settings.

    

### [[2112.14278] Beta-VAE Reproducibility: Challenges and Extensions](http://arxiv.org/abs/2112.14278)


  $\beta$-VAE is a follow-up technique to variational autoencoders that
proposes special weighting of the KL divergence term in the VAE loss to obtain
disentangled representations. Unsupervised learning is known to be brittle even
on toy datasets and a meaningful, mathematically precise definition of
disentanglement remains difficult to find. Here we investigate the original
$\beta$-VAE paper and add evidence to the results previously obtained
indicating its lack of reproducibility. We also further expand the
experimentation of the models and include further more complex datasets in the
analysis. We also implement an FID scoring metric for the $\beta$-VAE model and
conclude a qualitative analysis of the results obtained. We end with a brief
discussion on possible future investigations that can be conducted to add more
robustness to the claims.

    

### [[2112.14299] DeepAdversaries: Examining the Robustness of Deep Learning Models for Galaxy Morphology Classification](http://arxiv.org/abs/2112.14299)


  Data processing and analysis pipelines in cosmological survey experiments
introduce data perturbations that can significantly degrade the performance of
deep learning-based models. Given the increased adoption of supervised deep
learning methods for processing and analysis of cosmological survey data, the
assessment of data perturbation effects and the development of methods that
increase model robustness are increasingly important. In the context of
morphological classification of galaxies, we study the effects of perturbations
in imaging data. In particular, we examine the consequences of using neural
networks when training on baseline data and testing on perturbed data. We
consider perturbations associated with two primary sources: 1) increased
observational noise as represented by higher levels of Poisson noise and 2)
data processing noise incurred by steps such as image compression or telescope
errors as represented by one-pixel adversarial attacks. We also test the
efficacy of domain adaptation techniques in mitigating the perturbation-driven
errors. We use classification accuracy, latent space visualizations, and latent
space distance to assess model robustness. Without domain adaptation, we find
that processing pixel-level errors easily flip the classification into an
incorrect class and that higher observational noise makes the model trained on
low-noise data unable to classify galaxy morphologies. On the other hand, we
show that training with domain adaptation improves model robustness and
mitigates the effects of these perturbations, improving the classification
accuracy by 23% on data with higher observational noise. Domain adaptation also
increases by a factor of ~2.3 the latent space distance between the baseline
and the incorrectly classified one-pixel perturbed image, making the model more
robust to inadvertent perturbations.

    

### [[2112.14300] Time-Incremental Learning from Data Using Temporal Logics](http://arxiv.org/abs/2112.14300)


  Real-time and human-interpretable decision-making in cyber-physical systems
is a significant but challenging task, which usually requires predictions of
possible future events from limited data. In this paper, we introduce a
time-incremental learning framework: given a dataset of labeled signal traces
with a common time horizon, we propose a method to predict the label of a
signal that is received incrementally over time, referred to as prefix signal.
Prefix signals are the signals that are being observed as they are generated,
and their time length is shorter than the common horizon of signals. We present
a novel decision-tree based approach to generate a finite number of Signal
Temporal Logic (STL) specifications from the given dataset, and construct a
predictor based on them. Each STL specification, as a binary classifier of
time-series data, captures the temporal properties of the dataset over time.
The predictor is constructed by assigning time-variant weights to the STL
formulas. The weights are learned by using neural networks, with the goal of
minimizing the misclassification rate for the prefix signals defined over the
given dataset. The learned predictor is used to predict the label of a prefix
signal, by computing the weighted sum of the robustness of the prefix signal
with respect to each STL formula. The effectiveness and classification
performance of our algorithm are evaluated on an urban-driving and a
naval-surveillance case studies.

    

### [[2112.14307] Ensemble Recognition in Reproducing Kernel Hilbert Spaces through Aggregated Measurements](http://arxiv.org/abs/2112.14307)


  In this paper, we study the problem of learning dynamical properties of
ensemble systems from their collective behaviors using statistical approaches
in reproducing kernel Hilbert space (RKHS). Specifically, we provide a
framework to identify and cluster multiple ensemble systems through computing
the maximum mean discrepancy (MMD) between their aggregated measurements in an
RKHS, without any prior knowledge of the system dynamics of ensembles. Then,
leveraging on a gradient flow of the newly proposed notion of aggregated Markov
parameters, we present a systematic framework to recognize and identify an
ensemble systems using their linear approximations. Finally, we demonstrate
that the proposed approaches can be extended to cluster multiple unknown
ensembles in RKHS using their aggregated measurements. Numerical experiments
show that our approach is reliable and robust to ensembles with different types
of system dynamics.

    

### [[2112.14314] Improving Prediction of Cognitive Performance using Deep Neural Networks in Sparse Data](http://arxiv.org/abs/2112.14314)


  Cognition in midlife is an important predictor of age-related mental decline
and statistical models that predict cognitive performance can be useful for
predicting decline. However, existing models struggle to capture complex
relationships between physical, sociodemographic, psychological and mental
health factors that effect cognition. Using data from an observational, cohort
study, Midlife in the United States (MIDUS), we modeled a large number of
variables to predict executive function and episodic memory measures. We used
cross-sectional and longitudinal outcomes with varying sparsity, or amount of
missing data. Deep neural network (DNN) models consistently ranked highest in
all of the cognitive performance prediction tasks, as assessed with root mean
squared error (RMSE) on out-of-sample data. RMSE differences between DNN and
other model types were statistically significant (T(8) = -3.70; p < 0.05). The
interaction effect between model type and sparsity was significant (F(9)=59.20;
p < 0.01), indicating the success of DNNs can partly be attributed to their
robustness and ability to model hierarchical relationships between
health-related factors. Our findings underscore the potential of neural
networks to model clinical datasets and allow better understanding of factors
that lead to cognitive decline.

    

### [[2112.14316] FRIDA -- Generative Feature Replay for Incremental Domain Adaptation](http://arxiv.org/abs/2112.14316)


  We tackle the novel problem of incremental unsupervised domain adaptation
(IDA) in this paper. We assume that a labeled source domain and different
unlabeled target domains are incrementally observed with the constraint that
data corresponding to the current domain is only available at a time. The goal
is to preserve the accuracies for all the past domains while generalizing well
for the current domain. The IDA setup suffers due to the abrupt differences
among the domains and the unavailability of past data including the source
domain. Inspired by the notion of generative feature replay, we propose a novel
framework called Feature Replay based Incremental Domain Adaptation (FRIDA)
which leverages a new incremental generative adversarial network (GAN) called
domain-generic auxiliary classification GAN (DGAC-GAN) for producing
domain-specific feature representations seamlessly. For domain alignment, we
propose a simple extension of the popular domain adversarial neural network
(DANN) called DANN-IB which encourages discriminative domain-invariant and
task-relevant feature learning. Experimental results on Office-Home,
Office-CalTech, and DomainNet datasets confirm that FRIDA maintains superior
stability-plasticity trade-off than the literature.

    

### [[2112.14327] Multi-Head Deep Metric Learning Using Global and Local Representations](http://arxiv.org/abs/2112.14327)


  Deep Metric Learning (DML) models often require strong local and global
representations, however, effective integration of local and global features in
DML model training is a challenge. DML models are often trained with specific
loss functions, including pairwise-based and proxy-based losses. The
pairwise-based loss functions leverage rich semantic relations among data
points, however, they often suffer from slow convergence during DML model
training. On the other hand, the proxy-based loss functions often lead to
significant speedups in convergence during training, while the rich relations
among data points are often not fully explored by the proxy-based losses. In
this paper, we propose a novel DML approach to address these challenges. The
proposed DML approach makes use of a hybrid loss by integrating the
pairwise-based and the proxy-based loss functions to leverage rich data-to-data
relations as well as fast convergence. Furthermore, the proposed DML approach
utilizes both global and local features to obtain rich representations in DML
model training. Finally, we also use the second-order attention for feature
enhancement to improve accurate and efficient retrieval. In our experiments, we
extensively evaluated the proposed DML approach on four public benchmarks, and
the experimental results demonstrate that the proposed method achieved
state-of-the-art performance on all benchmarks.

    

### [[2112.14332] Adaptive Client Sampling in Federated Learning via Online Learning with Bandit Feedback](http://arxiv.org/abs/2112.14332)


  In federated learning (FL) problems, client sampling plays a key role in the
convergence speed of training algorithm. However, while being an important
problem in FL, client sampling is lack of study. In this paper, we propose an
online learning with bandit feedback framework to understand the client
sampling problem in FL. By adapting an Online Stochastic Mirror Descent
algorithm to minimize the variance of gradient estimation, we propose a new
adaptive client sampling algorithm. Besides, we use online ensemble method and
doubling trick to automatically choose the tuning parameters in the algorithm.
Theoretically, we show dynamic regret bound with comparator as the
theoretically optimal sampling sequence; we also include the total variation of
this sequence in our upper bound, which is a natural measure of the intrinsic
difficulty of the problem. To the best of our knowledge, these theoretical
contributions are novel to existing literature. Moreover, by implementing both
synthetic and real data experiments, we show empirical evidence of the
advantages of our proposed algorithms over widely-used uniform sampling and
also other online learning based sampling strategies in previous studies. We
also examine its robustness to the choice of tuning parameters. Finally, we
discuss its possible extension to sampling without replacement and personalized
FL objective. While the original goal is to solve client sampling problem, this
work has more general applications on stochastic gradient descent and
stochastic coordinate descent methods.

    

### [[2112.14337] Closer Look at the Transferability of Adversarial Examples: How They Fool Different Models Differently](http://arxiv.org/abs/2112.14337)


  Deep neural networks are vulnerable to adversarial examples (AEs), which have
adversarial transferability: AEs generated for the source model can mislead
another (target) model's predictions. However, the transferability has not been
understood from the perspective of to which class target model's predictions
were misled (i.e., class-aware transferability). In this paper, we
differentiate the cases in which a target model predicts the same wrong class
as the source model ("same mistake") or a different wrong class ("different
mistake") to analyze and provide an explanation of the mechanism. First, our
analysis shows (1) that same mistakes correlate with "non-targeted
transferability" and (2) that different mistakes occur between similar models
regardless of the perturbation size. Second, we present evidence that the
difference in same and different mistakes can be explained by non-robust
features, predictive but human-uninterpretable patterns: different mistakes
occur when non-robust features in AEs are used differently by models.
Non-robust features can thus provide consistent explanations for the
class-aware transferability of AEs.

    

### [[2112.14340] Super-Efficient Super Resolution for Fast Adversarial Defense at the Edge](http://arxiv.org/abs/2112.14340)


  Autonomous systems are highly vulnerable to a variety of adversarial attacks
on Deep Neural Networks (DNNs). Training-free model-agnostic defenses have
recently gained popularity due to their speed, ease of deployment, and ability
to work across many DNNs. To this end, a new technique has emerged for
mitigating attacks on image classification DNNs, namely, preprocessing
adversarial images using super resolution -- upscaling low-quality inputs into
high-resolution images. This defense requires running both image classifiers
and super resolution models on constrained autonomous systems. However, super
resolution incurs a heavy computational cost. Therefore, in this paper, we
investigate the following question: Does the robustness of image classifiers
suffer if we use tiny super resolution models? To answer this, we first review
a recent work called Super-Efficient Super Resolution (SESR) that achieves
similar or better image quality than prior art while requiring 2x to 330x fewer
Multiply-Accumulate (MAC) operations. We demonstrate that despite being orders
of magnitude smaller than existing models, SESR achieves the same level of
robustness as significantly larger networks. Finally, we estimate end-to-end
performance of super resolution-based defenses on a commercial Arm Ethos-U55
micro-NPU. Our findings show that SESR achieves nearly 3x higher FPS than a
baseline while achieving similar robustness.

    

### [[2112.14359] Federated Learning for Cross-block Oil-water Layer Identification](http://arxiv.org/abs/2112.14359)


  Cross-block oil-water layer(OWL) identification is essential for petroleum
development. Traditional methods are greatly affected by subjective factors due
to depending mainly on the human experience. AI-based methods have promoted the
development of OWL identification. However, because of the significant
geological differences across blocks and the severe long-tailed
distribution(class imbalanced), the identification effects of existing
artificial intelligence(AI) models are limited. In this paper, we address this
limitation by proposing a dynamic fusion-based federated learning(FL) for OWL
identification. To overcome geological differences, we propose a dynamic
weighted strategy to fuse models and train a general OWL identification model.
In addition, an F1 score-based re-weighting scheme is designed and a novel loss
function is derived theoretically to solve the data long-tailed problem.
Further, a geological knowledge-based mask-attention mechanism is proposed to
enhance model feature extraction. To our best knowledge, this is the first work
to identify OWL using FL. We evaluate the proposed approach with an actual well
logging dataset from the oil field and a public 3W dataset. Experimental
results demonstrate that our approach significantly out-performs other AI
methods.

    

### [[2112.14364] Feature-context driven Federated Meta-Learning for Rare Disease Prediction](http://arxiv.org/abs/2112.14364)


  Millions of patients suffer from rare diseases around the world. However, the
samples of rare diseases are much smaller than those of common diseases. In
addition, due to the sensitivity of medical data, hospitals are usually
reluctant to share patient information for data fusion citing privacy concerns.
These challenges make it difficult for traditional AI models to extract rare
disease features for the purpose of disease prediction. In this paper, we
overcome this limitation by proposing a novel approach for rare disease
prediction based on federated meta-learning. To improve the prediction accuracy
of rare diseases, we design an attention-based meta-learning (ATML) approach
which dynamically adjusts the attention to different tasks according to the
measured training effect of base learners. Additionally, a dynamic-weight based
fusion strategy is proposed to further improve the accuracy of federated
learning, which dynamically selects clients based on the accuracy of each local
model. Experiments show that with as few as five shots, our approach
out-performs the original federated meta-learning algorithm in accuracy and
speed. Compared with each hospital's local model, the proposed model's average
prediction accuracy increased by 13.28%.

    

### [[2112.14368] Adaptivity and Non-stationarity: Problem-dependent Dynamic Regret for Online Convex Optimization](http://arxiv.org/abs/2112.14368)


  We investigate online convex optimization in non-stationary environments and
choose the \emph{dynamic regret} as the performance measure, defined as the
difference between cumulative loss incurred by the online algorithm and that of
any feasible comparator sequence. Let $T$ be the time horizon and $P_T$ be the
path-length that essentially reflects the non-stationarity of environments, the
state-of-the-art dynamic regret is $\mathcal{O}(\sqrt{T(1+P_T)})$. Although
this bound is proved to be minimax optimal for convex functions, in this paper,
we demonstrate that it is possible to further enhance the guarantee for some
easy problem instances, particularly when online functions are smooth.
Specifically, we propose novel online algorithms that can leverage smoothness
and replace the dependence on $T$ in the dynamic regret by
\emph{problem-dependent} quantities: the variation in gradients of loss
functions, the cumulative loss of the comparator sequence, and the minimum of
the previous two terms. These quantities are at most $\mathcal{O}(T)$ while
could be much smaller in benign environments. Therefore, our results are
adaptive to the intrinsic difficulty of the problem, since the bounds are
tighter than existing results for easy problems and meanwhile guarantee the
same rate in the worst case. Notably, our algorithm requires only \emph{one}
gradient per iteration, which shares the same gradient query complexity with
the methods developed for optimizing the static regret. As a further
application, we extend the results from the full-information setting to bandit
convex optimization with two-point feedback and thereby attain the first
problem-dependent dynamic regret for such bandit tasks.

    

### [[2112.14370] On the Overlooked Significance of Underutilized Contextual Features in Recent News Recommendation Models](http://arxiv.org/abs/2112.14370)


  Personalized news recommendation aims to provide attractive articles for
readers by predicting their likelihood of clicking on a certain article. To
accurately predict this probability, plenty of studies have been proposed that
actively utilize content features of articles, such as words, categories, or
entities. However, we observed that the articles' contextual features, such as
CTR (click-through-rate), popularity, or freshness, were either neglected or
underutilized recently. To prove that this is the case, we conducted an
extensive comparison between recent deep-learning models and naive contextual
models that we devised and surprisingly discovered that the latter easily
outperforms the former. Furthermore, our analysis showed that the recent
tendency to apply overly sophisticated deep-learning operations to contextual
features was actually hindering the recommendation performance. From this
knowledge, we design a purposefully simple contextual module that can boost the
previous news recommendation models by a large margin.

    

### [[2112.14375] Variational Learning for the Inverted Beta-Liouville Mixture Model and Its Application to Text Categorization](http://arxiv.org/abs/2112.14375)


  The finite invert Beta-Liouville mixture model (IBLMM) has recently gained
some attention due to its positive data modeling capability. Under the
conventional variational inference (VI) framework, the analytically tractable
solution to the optimization of the variational posterior distribution cannot
be obtained, since the variational object function involves evaluation of
intractable moments. With the recently proposed extended variational inference
(EVI) framework, a new function is proposed to replace the original variational
object function in order to avoid intractable moment computation, so that the
analytically tractable solution of the IBLMM can be derived in an elegant way.
The good performance of the proposed approach is demonstrated by experiments
with both synthesized data and a real-world application namely text
categorization.

    

### [[2112.14377] DeepHAM: A Global Solution Method for Heterogeneous Agent Models with Aggregate Shocks](http://arxiv.org/abs/2112.14377)


  We propose an efficient, reliable, and interpretable global solution method,
$\textit{Deep learning-based algorithm for Heterogeneous Agent Models,
DeepHAM}$, for solving high dimensional heterogeneous agent models with
aggregate shocks. The state distribution is approximately represented by a set
of optimal generalized moments. Deep neural networks are used to approximate
the value and policy functions, and the objective is optimized over directly
simulated paths. Besides being an accurate global solver, this method has three
additional features. First, it is computationally efficient for solving complex
heterogeneous agent models, and it does not suffer from the curse of
dimensionality. Second, it provides a general and interpretable representation
of the distribution over individual states; and this is important for
addressing the classical question of whether and how heterogeneity matters in
macroeconomics. Third, it solves the constrained efficiency problem as easily
as the competitive equilibrium, and this opens up new possibilities for
studying optimal monetary and fiscal policies in heterogeneous agent models
with aggregate shocks.

    

### [[2112.14380] Cross-Domain Empirical Risk Minimization for Unbiased Long-tailed Classification](http://arxiv.org/abs/2112.14380)


  We address the overlooked unbiasedness in existing long-tailed classification
methods: we find that their overall improvement is mostly attributed to the
biased preference of tail over head, as the test distribution is assumed to be
balanced; however, when the test is as imbalanced as the long-tailed training
data -- let the test respect Zipf's law of nature -- the tail bias is no longer
beneficial overall because it hurts the head majorities. In this paper, we
propose Cross-Domain Empirical Risk Minimization (xERM) for training an
unbiased model to achieve strong performances on both test distributions, which
empirically demonstrates that xERM fundamentally improves the classification by
learning better feature representation rather than the head vs. tail game.
Based on causality, we further theoretically explain why xERM achieves
unbiasedness: the bias caused by the domain selection is removed by adjusting
the empirical risks on the imbalanced domain and the balanced but unseen
domain. Codes are available at this https URL.

    

### [[2112.14397] Dense-to-Sparse Gate for Mixture-of-Experts](http://arxiv.org/abs/2112.14397)


  Mixture-of-experts (MoE) is becoming popular due to its success in improving
the model quality, especially in Transformers. By routing tokens with a sparse
gate to a few experts that each only contains part of the full model, MoE keeps
the model size unchanged and significantly reduces per-token computation, which
effectively scales neural networks. However, we found that the current approach
of jointly training experts and the sparse gate introduces a negative impact on
model accuracy, diminishing the efficiency of expensive large-scale model
training. In this work, we proposed Dense-To-Sparse gate (DTS-Gate) for MoE
training. Specifically, instead of using a permanent sparse gate, DTS-Gate
begins as a dense gate that routes tokens to all experts, then gradually and
adaptively becomes sparser while routes to fewer experts. MoE with DTS-Gate
naturally decouples the training of experts and the sparse gate by training all
experts at first and then learning the sparse gate. Experiments show that
compared with the state-of-the-art Switch-Gate in GPT-MoE(1.5B) model with
OpenWebText dataset(40GB), DTS-Gate can obtain 2.0x speed-up to reach the same
validation perplexity, as well as higher FLOPs-efficiency of a 1.42x speed-up.

    

### [[2112.14406] Overcoming Mode Collapse with Adaptive Multi Adversarial Training](http://arxiv.org/abs/2112.14406)


  Generative Adversarial Networks (GANs) are a class of generative models used
for various applications, but they have been known to suffer from the mode
collapse problem, in which some modes of the target distribution are ignored by
the generator. Investigative study using a new data generation procedure
indicates that the mode collapse of the generator is driven by the
discriminator's inability to maintain classification accuracy on previously
seen samples, a phenomenon called Catastrophic Forgetting in continual
learning. Motivated by this observation, we introduce a novel training
procedure that adaptively spawns additional discriminators to remember previous
modes of generation. On several datasets, we show that our training scheme can
be plugged-in to existing GAN frameworks to mitigate mode collapse and improve
standard metrics for GAN evaluation.

    

### [[2112.14407] The impact of students behaviour, their approach, emotions and problem difficulty level on the performance prediction, evaluation and overall learning process during online coding activities](http://arxiv.org/abs/2112.14407)


  Learning process while solving coding problems is quite complex to
understand. It is extremely important to understand the skills which are
required and gained during learning to code. As a first step to understand the
students behaviour and approach during learning coding, two online coding
assignments or competitions are conducted with a 1-hour time limit. A survey
has been conducted at the end of each coding test and answers to different
questions have been collected. In depth statistical analysis is done to
understand the learning process while solving the coding problems. It involves
lots of parameters including students behaviour, their approach and difficulty
level of coding problems. The inclusion of mood and emotions related questions
can improve overall prediction performance but difficulty level matters in the
submission status prediction. Two coding assignments or competitions are
analyzed through in-depth research on 229 (first coding competition dataset)
and 325 (second coding competition dataset) data points. The primary results
are promising and these results give in depth insights about how learning to
solve coding problems is affected by students behaviour, their approach,
emotions and problem difficulty level.

    

### [[2112.14415] Data-Driven Computational Methods for the Domain of Attraction and Zubov's Equation](http://arxiv.org/abs/2112.14415)


  This paper deals with a special type of Lyapunov functions, namely the
solution of Zubov's equation. Such a function can be used to characterize the
domain of attraction for systems of ordinary differential equations. We derive
and prove an integral form solution to Zubov's equation. For numerical
computation, we develop two data-driven methods. One is based on the
integration of an augmented system of differential equations; and the other one
is based on deep learning. The former is effective for systems with a
relatively low state space dimension and the latter is developed for high
dimensional problems. The deep learning method is applied to a New England
10-generator power system model. We prove that a neural network approximation
exists for the Lyapunov function of power systems such that the approximation
error is a cubic polynomial of the number of generators. The error convergence
rate as a function of n, the number of neurons, is proved.

    

### [[2112.14417] Control Theoretic Analysis of Temporal Difference Learning](http://arxiv.org/abs/2112.14417)


  The goal of this paper is to investigate a control theoretic analysis of
linear stochastic iterative algorithm and temporal difference (TD) learning.
TD-learning is a linear stochastic iterative algorithm to estimate the value
function of a given policy for a Markov decision process, which is one of the
most popular and fundamental reinforcement learning algorithms. While there has
been a series of successful works in theoretical analysis of TD-learning, it
was not until recently that researchers found some guarantees on its
statistical efficiency. In this paper, we propose a control theoretic
finite-time analysis TD-learning, which exploits standard notions in linear
system control communities. Therefore, the proposed work provides additional
insights on TD-learning and reinforcement learning with simple concepts and
analysis tools in control theory.

    

### [[2112.14430] DP-FP: Differentially Private Forward Propagation for Large Models](http://arxiv.org/abs/2112.14430)


  When applied to large-scale learning problems, the conventional wisdom on
privacy-preserving deep learning, known as Differential Private Stochastic
Gradient Descent (DP-SGD), has met with limited success due to significant
performance degradation and high memory overhead when compared to the
non-privacy counterpart. We show how to mitigate the performance drop by
replacing the DP-SGD with a novel DP Forward-Propagation (DP-FP) followed by an
off-the-shelf non-DP optimizer. Our DP-FP employs novel (1) representation
clipping followed by noise addition in the forward propagation stage, as well
as (2) micro-batch construction via subsampling to achieve DP amplification and
reduce noise power to $1/M$, where $M$ is the number of micro-batch in a step.
When training a classification model, our DP-FP with all of the
privacy-preserving operations on the representation is innately free of
gradient bias, total noise proportionally to model size, and memory issues in
DP-SGD. As a result, our DP-FP outperforms cutting-edge DP-SGD while retaining
the same level of privacy, and it approaches non-private baselines and
significantly outperforms state-of-the-art DP-SGD variants. When applied to
RoBERTa-large on four downstream tasks, for example, DP-FP achieves an average
accuracy of 91.34\% with privacy budgets less than 3, representing a 3.81\%
performance improvement over the state-of-the-art DP-SGD and only a 0.9\% loss
compared to the non-private baseline but with a significantly lower privacy
leakage risk.

    

### [[2112.14435] EiFFFeL: Enforcing Fairness in Forests by Flipping Leaves](http://arxiv.org/abs/2112.14435)


  Nowadays Machine Learning (ML) techniques are extensively adopted in many
socially sensitive systems, thus requiring to carefully study the fairness of
the decisions taken by such systems. Many approaches have been proposed to
address and to make sure there is no bias against individuals or specific
groups which might originally come from biased training datasets or algorithm
design. In this regard, we propose a fairness enforcing approach called
EiFFFeL:Enforcing Fairness in Forests by Flipping Leaves which exploits
tree-based or leaf-based post-processing strategies to relabel leaves of
selected decision trees of a given forest. Experimental results show that our
approach achieves a user defined group fairness degree without losing a
significant amount of accuracy.

    

### [[2112.14436] Monte Carlo EM for Deep Time Series Anomaly Detection](http://arxiv.org/abs/2112.14436)


  Time series data are often corrupted by outliers or other kinds of anomalies.
Identifying the anomalous points can be a goal on its own (anomaly detection),
or a means to improving performance of other time series tasks (e.g.
forecasting). Recent deep-learning-based approaches to anomaly detection and
forecasting commonly assume that the proportion of anomalies in the training
data is small enough to ignore, and treat the unlabeled data as coming from the
nominal data distribution. We present a simple yet effective technique for
augmenting existing time series models so that they explicitly account for
anomalies in the training data. By augmenting the training data with a latent
anomaly indicator variable whose distribution is inferred while training the
underlying model using Monte Carlo EM, our method simultaneously infers
anomalous points while improving model performance on nominal data. We
demonstrate the effectiveness of the approach by combining it with a simple
feed-forward forecasting model. We investigate how anomalies in the train set
affect the training of forecasting models, which are commonly used for time
series anomaly detection, and show that our method improves the training of the
model.

    

### [[2112.14438] Deformable Graph Convolutional Networks](http://arxiv.org/abs/2112.14438)


  Graph neural networks (GNNs) have significantly improved the representation
power for graph-structured data. Despite of the recent success of GNNs, the
graph convolution in most GNNs have two limitations. Since the graph
convolution is performed in a small local neighborhood on the input graph, it
is inherently incapable to capture long-range dependencies between distance
nodes. In addition, when a node has neighbors that belong to different classes,
i.e., heterophily, the aggregated messages from them often negatively affect
representation learning. To address the two common problems of graph
convolution, in this paper, we propose Deformable Graph Convolutional Networks
(Deformable GCNs) that adaptively perform convolution in multiple latent spaces
and capture short/long-range dependencies between nodes. Separated from node
representations (features), our framework simultaneously learns the node
positional embeddings (coordinates) to determine the relations between nodes in
an end-to-end fashion. Depending on node position, the convolution kernels are
deformed by deformation vectors and apply different transformations to its
neighbor nodes. Our extensive experiments demonstrate that Deformable GCNs
flexibly handles the heterophily and achieve the best performance in node
classification tasks on six heterophilic graph datasets.

    

### [[2112.14445] Differentially-Private Clustering of Easy Instances](http://arxiv.org/abs/2112.14445)


  Clustering is a fundamental problem in data analysis. In differentially
private clustering, the goal is to identify $k$ cluster centers without
disclosing information on individual data points. Despite significant research
progress, the problem had so far resisted practical solutions. In this work we
aim at providing simple implementable differentially private clustering
algorithms that provide utility when the data is "easy," e.g., when there
exists a significant separation between the clusters.
We propose a framework that allows us to apply non-private clustering
algorithms to the easy instances and privately combine the results. We are able
to get improved sample complexity bounds in some cases of Gaussian mixtures and
$k$-means. We complement our theoretical analysis with an empirical evaluation
on synthetic data.

    

### [[2112.14448] A transfer learning enhanced the physics-informed neural network model for vortex-induced vibration](http://arxiv.org/abs/2112.14448)


  Vortex-induced vibration (VIV) is a typical nonlinear fluid-structure
interaction phenomenon, which widely exists in practical engineering (the
flexible riser, the bridge and the aircraft wing, etc). The conventional finite
element model (FEM)-based and data-driven approaches for VIV analysis often
suffer from the challenges of the computational cost and acquisition of
datasets. This paper proposed a transfer learning enhanced the physics-informed
neural network (PINN) model to study the VIV (2D). The physics-informed neural
network, when used in conjunction with the transfer learning method, enhances
learning efficiency and keeps predictability in the target task by common
characteristics knowledge from the source model without requiring a huge
quantity of datasets. The datasets obtained from VIV experiment are divided
evenly two parts (source domain and target domain), to evaluate the performance
of the model. The results show that the proposed method match closely with the
results available in the literature using conventional PINN algorithms even
though the quantity of datasets acquired in training model gradually becomes
smaller. The application of the model can break the limitation of monitoring
equipment and methods in the practical projects, and promote the in-depth study
of VIV.

    

### [[2112.14466] Explainability Is in the Mind of the Beholder: Establishing the Foundations of Explainable Artificial Intelligence](http://arxiv.org/abs/2112.14466)


  Explainable artificial intelligence and interpretable machine learning are
research fields growing in importance. Yet, the underlying concepts remain
somewhat elusive and lack generally agreed definitions. While recent
inspiration from social sciences has refocused the work on needs and
expectations of human recipients, the field still misses a concrete
conceptualisation. We take steps towards addressing this challenge by reviewing
the philosophical and social foundations of human explainability, which we then
translate into the technological realm. In particular, we scrutinise the notion
of algorithmic black boxes and the spectrum of understanding determined by
explanatory processes and explainees' background knowledge. This approach
allows us to define explainability as (logical) reasoning applied to
transparent insights (into black boxes) interpreted under certain background
knowledge - a process that engenders understanding in explainees. We then
employ this conceptualisation to revisit the much disputed trade-off between
transparency and predictive power and its implications for ante-hoc and
post-hoc explainers as well as fairness and accountability engendered by
explainability. We furthermore discuss components of the machine learning
workflow that may be in need of interpretability, building on a range of ideas
from human-centred explainability, with a focus on explainees, contrastive
statements and explanatory processes. Our discussion reconciles and complements
current research to help better navigate open questions - rather than
attempting to address any individual issue - thus laying a solid foundation for
a grounded discussion and future progress of explainable artificial
intelligence and interpretable machine learning. We conclude with a summary of
our findings, revisiting the human-centred explanatory process needed to
achieve the desired level of algorithmic transparency.

    

### [[2112.14472] Temporal Attention Augmented Transformer Hawkes Process](http://arxiv.org/abs/2112.14472)


  In recent years, mining the knowledge from asynchronous sequences by Hawkes
process is a subject worthy of continued attention, and Hawkes processes based
on the neural network have gradually become the most hotly researched fields,
especially based on the recurrence neural network (RNN). However, these models
still contain some inherent shortcomings of RNN, such as vanishing and
exploding gradient and long-term dependency problems. Meanwhile, Transformer
based on self-attention has achieved great success in sequential modeling like
text processing and speech recognition. Although the Transformer Hawkes process
(THP) has gained huge performance improvement, THPs do not effectively utilize
the temporal information in the asynchronous events, for these asynchronous
sequences, the event occurrence instants are as important as the types of
events, while conventional THPs simply convert temporal information into
position encoding and add them as the input of transformer. With this in mind,
we come up with a new kind of Transformer-based Hawkes process model, Temporal
Attention Augmented Transformer Hawkes Process (TAA-THP), we modify the
traditional dot-product attention structure, and introduce the temporal
encoding into attention structure. We conduct numerous experiments on a wide
range of synthetic and real-life datasets to validate the performance of our
proposed TAA-THP model, significantly improvement compared with existing
baseline models on the different measurements is achieved, including
log-likelihood on the test dataset, and prediction accuracies of event types
and occurrence times. In addition, through the ablation studies, we vividly
demonstrate the merit of introducing additional temporal attention by comparing
the performance of the model with and without temporal attention.

    

### [[2112.14474] Bayesian Neural Hawkes Process for Event Uncertainty Prediction](http://arxiv.org/abs/2112.14474)


  Many applications comprise of sequences of event data with the time of
occurrence of the events. Models for predicting time of occurrence play a
significant role in a diverse set of applications like social networks,
financial transactions, healthcare, and human mobility. Recent works have
introduced neural network based point process for modeling event-times, and
were shown to provide state-of-the-art performance in predicting event-times.
However, neural networks are poor at quantifying predictive uncertainty and
tend to produce overconfident predictions during extrapolation. A proper
uncertainty quantification is crucial for many practical applications.
Therefore, we propose a novel point process model, Bayesian Neural Hawkes
process which leverages uncertainty modelling capability of Bayesian models and
generalization capability of the neural networks. The model is capable of
predicting epistemic uncertainty over the event occurrence time and its
effectiveness is demonstrated for on simulated and real-world datasets.

    

### [[2112.14479] Universal Transformer Hawkes Process with Adaptive Recursive Iteration](http://arxiv.org/abs/2112.14479)


  Asynchronous events sequences are widely distributed in the natural world and
human activities, such as earthquakes records, users activities in social media
and so on. How to distill the information from these seemingly disorganized
data is a persistent topic that researchers focus on. The one of the most
useful model is the point process model, and on the basis, the researchers
obtain many noticeable results. Moreover, in recent years, point process models
on the foundation of neural networks, especially recurrent neural networks
(RNN) are proposed and compare with the traditional models, their performance
are greatly improved. Enlighten by transformer model, which can learning
sequence data efficiently without recurrent and convolutional structure,
transformer Hawkes process is come out, and achieves state-of-the-art
performance. However, there is some research proving that the re-introduction
of recursive calculations in transformer can further improve transformers
performance. Thus, we come out with a new kind of transformer Hawkes process
model, universal transformer Hawkes process (UTHP), which contains both
recursive mechanism and self-attention mechanism, and to improve the local
perception ability of the model, we also introduce convolutional neural network
(CNN) in the position-wise-feed-forward part. We conduct experiments on several
datasets to validate the effectiveness of UTHP and explore the changes after
the introduction of the recursive mechanism. These experiments on multiple
datasets demonstrate that the performance of our proposed new model has a
certain improvement compared with the previous state-of-the-art models.

    

### [[2112.14482] GPS: A Policy-driven Sampling Approach for Graph Representation Learning](http://arxiv.org/abs/2112.14482)


  Graph representation learning has drawn increasing attention in recent years,
especially for learning the low dimensional embedding at both node and graph
level for classification and recommendations tasks. To enable learning the
representation on the large-scale graph data in the real world, numerous
research has focused on developing different sampling strategies to facilitate
the training process. Herein, we propose an adaptive Graph Policy-driven
Sampling model (GPS), where the influence of each node in the local
neighborhood is realized through the adaptive correlation calculation.
Specifically, the selections of the neighbors are guided by an adaptive policy
algorithm, contributing directly to the message aggregation, node embedding
updating, and graph level readout steps. We then conduct comprehensive
experiments against baseline methods on graph classification tasks from various
perspectives. Our proposed model outperforms the existing ones by 3%-8% on
several vital benchmarks, achieving state-of-the-art performance in real-world
datasets.

    

### [[2112.14491] Two-phase training mitigates class imbalance for camera trap image classification with CNNs](http://arxiv.org/abs/2112.14491)


  By leveraging deep learning to automatically classify camera trap images,
ecologists can monitor biodiversity conservation efforts and the effects of
climate change on ecosystems more efficiently. Due to the imbalanced
class-distribution of camera trap datasets, current models are biased towards
the majority classes. As a result, they obtain good performance for a few
majority classes but poor performance for many minority classes. We used
two-phase training to increase the performance for these minority classes. We
trained, next to a baseline model, four models that implemented a different
versions of two-phase training on a subset of the highly imbalanced Snapshot
Serengeti dataset. Our results suggest that two-phase training can improve
performance for many minority classes, with limited loss in performance for the
other classes. We find that two-phase training based on majority undersampling
increases class-specific F1-scores up to 3.0%. We also find that two-phase
training outperforms using only oversampling or undersampling by 6.1% in
F1-score on average. Finally, we find that a combination of over- and
undersampling leads to a better performance than using them individually.

    

### [[2112.14531] Designing the Topology of Graph Neural Networks: A Novel Feature Fusion Perspective](http://arxiv.org/abs/2112.14531)


  In recent years, Graph Neural Networks (GNNs) have shown superior performance
on diverse real-world applications. To improve the model capacity, besides
designing aggregation operations, GNN topology design is also very important.
In general, there are two mainstream GNN topology design manners. The first one
is to stack aggregation operations to obtain the higher-level features but
easily got performance drop as the network goes deeper. Secondly, the multiple
aggregation operations are utilized in each layer which provides adequate and
independent feature extraction stage on local neighbors while are costly to
obtain the higher-level information. To enjoy the benefits while alleviating
the corresponding deficiencies of these two manners, we learn to design the
topology of GNNs in a novel feature fusion perspective which is dubbed
F$^2$GNN. To be specific, we provide a feature fusion perspective in designing
GNN topology and propose a novel framework to unify the existing topology
designs with feature selection and fusion strategies. Then we develop a neural
architecture search method on top of the unified framework which contains a set
of selection and fusion operations in the search space and an improved
differentiable search algorithm. The performance gains on eight real-world
datasets demonstrate the effectiveness of F$^2$GNN. We further conduct
experiments to show that F$^2$GNN can improve the model capacity while
alleviating the deficiencies of existing GNN topology design manners,
especially alleviating the over-smoothing problem, by utilizing different
levels of features adaptively.

    

### [[2112.14553] Active Learning of Quantum System Hamiltonians yields Query Advantage](http://arxiv.org/abs/2112.14553)


  Hamiltonian learning is an important procedure in quantum system
identification, calibration, and successful operation of quantum computers.
Through queries to the quantum system, this procedure seeks to obtain the
parameters of a given Hamiltonian model and description of noise sources.
Standard techniques for Hamiltonian learning require careful design of queries
and $O(\epsilon^{-2})$ queries in achieving learning error $\epsilon$ due to
the standard quantum limit. With the goal of efficiently and accurately
estimating the Hamiltonian parameters within learning error $\epsilon$ through
minimal queries, we introduce an active learner that is given an initial set of
training examples and the ability to interactively query the quantum system to
generate new training data. We formally specify and experimentally assess the
performance of this Hamiltonian active learning (HAL) algorithm for learning
the six parameters of a two-qubit cross-resonance Hamiltonian on four different
superconducting IBM Quantum devices. Compared with standard techniques for the
same problem and a specified learning error, HAL achieves up to a $99.8\%$
reduction in queries required, and a $99.1\%$ reduction over the comparable
non-adaptive learning algorithm. Moreover, with access to prior information on
a subset of Hamiltonian parameters and given the ability to select queries with
linearly (or exponentially) longer system interaction times during learning,
HAL can exceed the standard quantum limit and achieve Heisenberg (or
super-Heisenberg) limited convergence rates during learning.

    

### [[2112.14569] Fine-Tuning Transformers: Vocabulary Transfer](http://arxiv.org/abs/2112.14569)


  Transformers are responsible for the vast majority of recent advances in
natural language processing. The majority of practical natural language
processing applications of these models is typically enabled through transfer
learning. This paper studies if corpus-specific tokenization used for
fine-tuning improves the resulting performance of the model. Through a series
of experiments, we demonstrate that such tokenization combined with the
initialization and fine-tuning strategy for the vocabulary tokens speeds up the
transfer and boosts the performance of the fine-tuned model. We call this
aspect of transfer facilitation vocabulary transfer.

    

### [[2112.14570] Lyapunov Exponents for Diversity in Differentiable Games](http://arxiv.org/abs/2112.14570)


  Ridge Rider (RR) is an algorithm for finding diverse solutions to
optimization problems by following eigenvectors of the Hessian ("ridges"). RR
is designed for conservative gradient systems (i.e., settings involving a
single loss function), where it branches at saddles - easy-to-find bifurcation
points. We generalize this idea to non-conservative, multi-agent gradient
systems by proposing a method - denoted Generalized Ridge Rider (GRR) - for
finding arbitrary bifurcation points. We give theoretical motivation for our
method by leveraging machinery from the field of dynamical systems. We
construct novel toy problems where we can visualize new phenomena while giving
insight into high-dimensional problems of interest. Finally, we empirically
evaluate our method by finding diverse solutions in the iterated prisoners'
dilemma and relevant machine learning problems including generative adversarial
networks.

    

### [[2112.14582] Polyak-Ruppert Averaged Q-Leaning is Statistically Efficient](http://arxiv.org/abs/2112.14582)


  We study synchronous Q-learning with Polyak-Ruppert averaging (a.k.a.,
averaged Q-leaning) in a $\gamma$-discounted MDP. We establish asymptotic
normality for the averaged iteration $\bar{\boldsymbol{Q}}_T$. Furthermore, we
show that $\bar{\boldsymbol{Q}}_T$ is actually a regular asymptotically linear
(RAL) estimator for the optimal Q-value function $\boldsymbol{Q}^*$ with the
most efficient influence function. It implies the averaged Q-learning iteration
has the smallest asymptotic variance among all RAL estimators. In addition, we
present a non-asymptotic analysis for the $\ell_{\infty}$ error
$\mathbb{E}\|\bar{\boldsymbol{Q}}_T-\boldsymbol{Q}^*\|_{\infty}$, showing it
matches the instance-dependent lower bound as well as the optimal minimax
complexity lower bound. As a byproduct, we find the Bellman noise has
sub-Gaussian coordinates with variance $\mathcal{O}((1-\gamma)^{-1})$ instead
of the prevailing $\mathcal{O}((1-\gamma)^{-2})$ under the standard bounded
reward assumption. The sub-Gaussian result has potential to improve the sample
complexity of many RL algorithms. In short, our theoretical analysis shows
averaged Q-Leaning is statistically efficient.

    

### [[2112.14586] Isotuning With Applications To Scale-Free Online Learning](http://arxiv.org/abs/2112.14586)


  We extend and combine several tools of the literature to design fast,
adaptive, anytime and scale-free online learning algorithms. Scale-free regret
bounds must scale linearly with the maximum loss, both toward large losses and
toward very small losses. Adaptive regret bounds demonstrate that an algorithm
can take advantage of easy data and potentially have constant regret. We seek
to develop fast algorithms that depend on as few parameters as possible, in
particular they should be anytime and thus not depend on the time horizon. Our
first and main tool, isotuning, is a generalization of the idea of balancing
the trade-off of the regret. We develop a set of tools to design and analyze
such learning rates easily and show that they adapts automatically to the rate
of the regret (whether constant, $O(\log T)$, $O(\sqrt{T})$, etc.) within a
factor 2 of the optimal learning rate in hindsight for the same observed
quantities. The second tool is an online correction, which allows us to obtain
centered bounds for many algorithms, to prevent the regret bounds from being
vacuous when the domain is overly large or only partially constrained. The last
tool, null updates, prevents the algorithm from performing overly large
updates, which could result in unbounded regret, or even invalid updates. We
develop a general theory using these tools and apply it to several standard
algorithms. In particular, we (almost entirely) restore the adaptivity to small
losses of FTRL for unbounded domains, design and prove scale-free adaptive
guarantees for a variant of Mirror Descent (at least when the Bregman
divergence is convex in its second argument), extend Adapt-ML-Prod to
scale-free guarantees, and provide several other minor contributions about
Prod, AdaHedge, BOA and Soft-Bayes.

    

### [[2112.14602] DDPG car-following model with real-world human driving experience in CARLA](http://arxiv.org/abs/2112.14602)


  In the autonomous driving field, the fusion of human knowledge into Deep
Reinforcement Learning (DRL) is often based on the human demonstration recorded
in the simulated environment. This limits the generalization and the
feasibility of application in real-world traffic. We proposed a two-stage DRL
method, that learns from real-world human driving to achieve performance that
is superior to the pure DRL agent. Training a DRL agent is done within a
framework for CARLA with Robot Operating System (ROS). For evaluation, we
designed different real-world driving scenarios to compare the proposed
two-stage DRL agent with the pure DRL agent. After extracting the 'good'
behavior from the human driver, such as anticipation in a signalized
intersection, the agent becomes more efficient and drives safer, which makes
this autonomous agent more adapt to Human-Robot Interaction (HRI) traffic.

    

### [[2112.14630] Time Series Data Mining Algorithms Towards Scalable and Real-Time Behavior Monitoring](http://arxiv.org/abs/2112.14630)


  In recent years, there have been unprecedented technological advances in
sensor technology, and sensors have become more affordable than ever. Thus,
sensor-driven data collection is increasingly becoming an attractive and
practical option for researchers around the globe. Such data is typically
extracted in the form of time series data, which can be investigated with data
mining techniques to summarize behaviors of a range of subjects including
humans and animals. While enabling cheap and mass collection of data,
continuous sensor data recording results in datasets which are big in size and
volume, which are challenging to process and analyze with traditional
techniques in a timely manner. Such collected sensor data is typically
extracted in the form of time series data. There are two main approaches in the
literature, namely, shape-based classification and feature-based
classification. Shape-based classification determines the best class according
to a distance measure. Feature-based classification, on the other hand,
measures properties of the time series and finds the best class according to
the set of features defined for the time series. In this dissertation, we
demonstrate that neither of the two techniques will dominate for some problems,
but that some combination of both might be the best. In other words, on a
single problem, it might be possible that one of the techniques is better for
one subset of the behaviors, and the other technique is better for another
subset of behaviors. We introduce a hybrid algorithm to classify behaviors,
using both shape and feature measures, in weakly labeled time series data
collected from sensors to quantify specific behaviors performed by the subject.
We demonstrate that our algorithm can robustly classify real, noisy, and
complex datasets, based on a combination of shape and features, and tested our
proposed algorithm on real-world datasets.

    

### [[2112.14638] Universal Online Learning with Bounded Loss: Reduction to Binary Classification](http://arxiv.org/abs/2112.14638)


  We study universal consistency of non-i.i.d. processes in the context of
online learning. A stochastic process is said to admit universal consistency if
there exists a learner that achieves vanishing average loss for any measurable
response function on this process. When the loss function is unbounded,
Blanchard et al. showed that the only processes admitting strong universal
consistency are those taking a finite number of values almost surely. However,
when the loss function is bounded, the class of processes admitting strong
universal consistency is much richer and its characterization could be
dependent on the response setting (Hanneke). In this paper, we show that this
class of processes is independent from the response setting thereby closing an
open question (Hanneke, Open Problem 3). Specifically, we show that the class
of processes that admit universal online learning is the same for binary
classification as for multiclass classification with countable number of
classes. Consequently, any output setting with bounded loss can be reduced to
binary classification. Our reduction is constructive and practical. Indeed, we
show that the nearest neighbor algorithm is transported by our construction.
For binary classification on a process admitting strong universal learning, we
prove that nearest neighbor successfully learns at least all finite unions of
intervals.

    

### [[2112.14644] Implementation of Convolutional Neural Network Architecture on 3D Multiparametric Magnetic Resonance Imaging for Prostate Cancer Diagnosis](http://arxiv.org/abs/2112.14644)


  Prostate cancer is one of the most common causes of cancer deaths in men.
There is a growing demand for noninvasively and accurately diagnostic methods
that facilitate the current standard prostate cancer risk assessment in
clinical practice. Still, developing computer-aided classification tools in
prostate cancer diagnostics from multiparametric magnetic resonance images
continues to be a challenge. In this work, we propose a novel deep learning
approach for automatic classification of prostate lesions in the corresponding
magnetic resonance images by constructing a two-stage multimodal multi-stream
convolutional neural network (CNN)-based architecture framework. Without
implementing sophisticated image preprocessing steps or third-party software,
our framework achieved the classification performance with the area under a
Receiver Operating Characteristic (ROC) curve value of 0.87. The result
outperformed most of the submitted methods and shared the highest value
reported by the PROSTATEx Challenge organizer. Our proposed CNN-based framework
reflects the potential of assisting medical image interpretation in prostate
cancer and reducing unnecessary biopsies.

    

### [[2112.14674] An additive graphical model for discrete data](http://arxiv.org/abs/2112.14674)


  We introduce a nonparametric graphical model for discrete node variables
based on additive conditional independence. Additive conditional independence
is a three way statistical relation that shares similar properties with
conditional independence by satisfying the semi-graphoid axioms. Based on this
relation we build an additive graphical model for discrete variables that does
not suffer from the restriction of a parametric model such as the Ising model.
We develop an estimator of the new graphical model via the penalized estimation
of the discrete version of the additive precision operator and establish the
consistency of the estimator under the ultrahigh-dimensional setting. Along
with these methodological developments, we also exploit the properties of
discrete random variables to uncover a deeper relation between additive
conditional independence and conditional independence than previously known.
The new graphical model reduces to a conditional independence graphical model
under certain sparsity conditions. We conduct simulation experiments and
analysis of an HIV antiretroviral therapy data set to compare the new method
with existing ones.

    

### [[2112.14678] Multi-Dialect Arabic Speech Recognition](http://arxiv.org/abs/2112.14678)


  This paper presents the design and development of multi-dialect automatic
speech recognition for Arabic. Deep neural networks are becoming an effective
tool to solve sequential data problems, particularly, adopting an end-to-end
training of the system. Arabic speech recognition is a complex task because of
the existence of multiple dialects, non-availability of large corpora, and
missing vocalization. Thus, the first contribution of this work is the
development of a large multi-dialectal corpus with either full or at least
partially vocalized transcription. Additionally, the open-source corpus has
been gathered from multiple sources that bring non-standard Arabic alphabets in
transcription which are normalized by defining a common character-set. The
second contribution is the development of a framework to train an acoustic
model achieving state-of-the-art performance. The network architecture
comprises of a combination of convolutional and recurrent layers. The
spectrogram features of the audio data are extracted in the frequency vs time
domain and fed in the network. The output frames, produced by the recurrent
model, are further trained to align the audio features with its corresponding
transcription sequences. The sequence alignment is performed using a beam
search decoder with a tetra-gram language model. The proposed system achieved a
14% error rate which outperforms previous systems.

    

### [[2112.14679] Profile Guided Optimization without Profiles: A Machine Learning Approach](http://arxiv.org/abs/2112.14679)


  Profile guided optimization is an effective technique for improving the
optimization ability of compilers based on dynamic behavior, but collecting
profile data is expensive, cumbersome, and requires regular updating to remain
fresh. We present a novel statistical approach to inferring branch
probabilities that improves the performance of programs that are compiled
without profile guided optimizations. We perform offline training using
information that is collected from a large corpus of binaries that have branch
probabilities information. The learned model is used by the compiler to predict
the branch probabilities of regular uninstrumented programs, which the compiler
can then use to inform optimization decisions. We integrate our technique
directly in LLVM, supplementing the existing human-engineered compiler
heuristics. We evaluate our technique on a suite of benchmarks, demonstrating
some gains over compiling without profile information. In deployment, our
technique requires no profiling runs and has negligible effect on compilation
time.

    

### [[2112.14683] StyleGAN-V: A Continuous Video Generator with the Price, Image Quality and Perks of StyleGAN2](http://arxiv.org/abs/2112.14683)


  Videos show continuous events, yet most - if not all - video synthesis
frameworks treat them discretely in time. In this work, we think of videos of
what they should be - time-continuous signals, and extend the paradigm of
neural representations to build a continuous-time video generator. For this, we
first design continuous motion representations through the lens of positional
embeddings. Then, we explore the question of training on very sparse videos and
demonstrate that a good generator can be learned by using as few as 2 frames
per clip. After that, we rethink the traditional image and video discriminators
pair and propose to use a single hypernetwork-based one. This decreases the
training cost and provides richer learning signal to the generator, making it
possible to train directly on 1024$^2$ videos for the first time. We build our
model on top of StyleGAN2 and it is just 5% more expensive to train at the same
resolution while achieving almost the same image quality. Moreover, our latent
space features similar properties, enabling spatial manipulations that our
method can propagate in time. We can generate arbitrarily long videos at
arbitrary high frame rate, while prior work struggles to generate even 64
frames at a fixed rate. Our model achieves state-of-the-art results on four
modern 256$^2$ video synthesis benchmarks and one 1024$^2$ resolution one.
Videos and the source code are available at the project website:
this https URL.

    

### [[2112.14705] Lane Change Decision-Making through Deep Reinforcement Learning](http://arxiv.org/abs/2112.14705)


  Due to the complexity and volatility of the traffic environment,
decision-making in autonomous driving is a significantly hard problem. In this
project, we use a Deep Q-Network, along with rule-based constraints to make
lane-changing decision. A safe and efficient lane change behavior may be
obtained by combining high-level lateral decision-making with low-level
rule-based trajectory monitoring. The agent is anticipated to perform
appropriate lane-change maneuvers in a real-world-like udacity simulator after
training it for a total of 100 episodes. The results shows that the rule-based
DQN performs better than the DQN method. The rule-based DQN achieves a safety
rate of 0.8 and average speed of 47 MPH

    

### [[2112.14710] Parallelized and Randomized Adversarial Imitation Learning for Safety-Critical Self-Driving Vehicles](http://arxiv.org/abs/2112.14710)


  Self-driving cars and autonomous driving research has been receiving
considerable attention as major promising prospects in modern artificial
intelligence applications. According to the evolution of advanced driver
assistance system (ADAS), the design of self-driving vehicle and autonomous
driving systems becomes complicated and safety-critical. In general, the
intelligent system simultaneously and efficiently activates ADAS functions.
Therefore, it is essential to consider reliable ADAS function coordination to
control the driving system, safely. In order to deal with this issue, this
paper proposes a randomized adversarial imitation learning (RAIL) algorithm.
The RAIL is a novel derivative-free imitation learning method for autonomous
driving with various ADAS functions coordination; and thus it imitates the
operation of decision maker that controls autonomous driving with various ADAS
functions. The proposed method is able to train the decision maker that deals
with the LIDAR data and controls the autonomous driving in multi-lane complex
highway environments. The simulation-based evaluation verifies that the
proposed method achieves desired performance.

    

### [[2112.14718] Shallow decision trees for explainable $k$-means clustering](http://arxiv.org/abs/2112.14718)


  A number of recent works have employed decision trees for the construction of
explainable partitions that aim to minimize the $k$-means cost function. These
works, however, largely ignore metrics related to the depths of the leaves in
the resulting tree, which is perhaps surprising considering how the
explainability of a decision tree depends on these depths. To fill this gap in
the literature, we propose an efficient algorithm that takes into account these
metrics. In experiments on 16 datasets, our algorithm yields better results
than decision-tree clustering algorithms such as the ones presented in
\cite{dasgupta2020explainable}, \cite{frost2020exkmc}, \cite{laber2021price}
and \cite{DBLP:conf/icml/MakarychevS21}, typically achieving lower or
equivalent costs with considerably shallower trees. We also show, through a
simple adaptation of existing techniques, that the problem of building
explainable partitions induced by binary trees for the $k$-means cost function
does not admit an $(1+\epsilon)$-approximation in polynomial time unless
$P=NP$, which justifies the quest for approximation algorithms and/or
heuristics.

    

### [[2112.14734] Sequential Episodic Control](http://arxiv.org/abs/2112.14734)


  State of the art deep reinforcement learning algorithms are sample
inefficient due to the large number of episodes they require to achieve
asymptotic performance. Episodic Reinforcement Learning (ERL) algorithms,
inspired by the mammalian hippocampus, typically use extended memory systems to
bootstrap learning from past events to overcome this sample-inefficiency
problem. However, such memory augmentations are often used as mere buffers,
from which isolated past experiences are drawn to learn from in an offline
fashion (e.g., replay). Here, we demonstrate that including a bias in the
acquired memory content derived from the order of episodic sampling improves
both the sample and memory efficiency of an episodic control algorithm. We test
our Sequential Episodic Control (SEC) model in a foraging task to show that
storing and using integrated episodes as event sequences leads to faster
learning with fewer memory requirements as opposed to a standard ERL benchmark,
Model-Free Episodic Control, that buffers isolated events only. We also study
the effect of memory constraints and forgetting on the sequential and
non-sequential version of the SEC algorithm. Furthermore, we discuss how a
hippocampal-like fast memory system could bootstrap slow cortical and
subcortical learning subserving habit formation in the mammalian brain.

    

### [[2112.14738] Nonconvex Stochastic Scaled-Gradient Descent and Generalized Eigenvector Problems](http://arxiv.org/abs/2112.14738)


  Motivated by the problem of online canonical correlation analysis, we propose
the \emph{Stochastic Scaled-Gradient Descent} (SSGD) algorithm for minimizing
the expectation of a stochastic function over a generic Riemannian manifold.
SSGD generalizes the idea of projected stochastic gradient descent and allows
the use of scaled stochastic gradients instead of stochastic gradients. In the
special case of a spherical constraint, which arises in generalized eigenvector
problems, we establish a nonasymptotic finite-sample bound of $\sqrt{1/T}$, and
show that this rate is minimax optimal, up to a polylogarithmic factor of
relevant parameters. On the asymptotic side, a novel trajectory-averaging
argument allows us to achieve local asymptotic normality with a rate that
matches that of Ruppert-Polyak-Juditsky averaging. We bring these ideas
together in an application to online canonical correlation analysis, deriving,
for the first time in the literature, an optimal one-time-scale algorithm with
an explicit rate of local asymptotic convergence to normality. Numerical
studies of canonical correlation analysis are also provided for synthetic data.

    

### [[2112.14754] Disentanglement and Generalization Under Correlation Shifts](http://arxiv.org/abs/2112.14754)


  Correlations between factors of variation are prevalent in real-world data.
Machine learning algorithms may benefit from exploiting such correlations, as
they can increase predictive performance on noisy data. However, often such
correlations are not robust (e.g., they may change between domains, datasets,
or applications) and we wish to avoid exploiting them. Disentanglement methods
aim to learn representations which capture different factors of variation in
latent subspaces. A common approach involves minimizing the mutual information
between latent subspaces, such that each encodes a single underlying attribute.
However, this fails when attributes are correlated. We solve this problem by
enforcing independence between subspaces conditioned on the available
attributes, which allows us to remove only dependencies that are not due to the
correlation structure present in the training data. We achieve this via an
adversarial approach to minimize the conditional mutual information (CMI)
between subspaces with respect to categorical variables. We first show
theoretically that CMI minimization is a good objective for robust
disentanglement on linear problems with Gaussian data. We then apply our method
on real-world datasets based on MNIST and CelebA, and show that it yields
models that are disentangled and robust under correlation shift, including in
weakly supervised settings.

    

### [[2112.14758] Multivariate Trend Filtering for Lattice Data](http://arxiv.org/abs/2112.14758)


  We study a multivariate version of trend filtering, called Kronecker trend
filtering or KTF, for the case in which the design points form a lattice in $d$
dimensions. KTF is a natural extension of univariate trend filtering (Steidl et
al., 2006; Kim et al., 2009; Tibshirani, 2014), and is defined by minimizing a
penalized least squares problem whose penalty term sums the absolute
(higher-order) differences of the parameter to be estimated along each of the
coordinate directions. The corresponding penalty operator can be written in
terms of Kronecker products of univariate trend filtering penalty operators,
hence the name Kronecker trend filtering. Equivalently, one can view KTF in
terms of an $\ell_1$-penalized basis regression problem where the basis
functions are tensor products of falling factorial functions, a piecewise
polynomial (discrete spline) basis that underlies univariate trend filtering.
This paper is a unification and extension of the results in Sadhanala et al.
(2016, 2017). We develop a complete set of theoretical results that describe
the behavior of $k^{\mathrm{th}}$ order Kronecker trend filtering in $d$
dimensions, for every $k \geq 0$ and $d \geq 1$. This reveals a number of
interesting phenomena, including the dominance of KTF over linear smoothers in
estimating heterogeneously smooth functions, and a phase transition at
$d=2(k+1)$, a boundary past which (on the high dimension-to-smoothness side)
linear smoothers fail to be consistent entirely. We also leverage recent
results on discrete splines from Tibshirani (2020), in particular, discrete
spline interpolation results that enable us to extend the KTF estimate to any
off-lattice location in constant-time (independent of the size of the lattice
$n$).

    

### [[1910.09089] Decentralized Heterogeneous Multi-Player Multi-Armed Bandits with Non-Zero Rewards on Collisions](http://arxiv.org/abs/1910.09089)


  We consider a fully decentralized multi-player stochastic multi-armed bandit
setting where the players cannot communicate with each other and can observe
only their own actions and rewards. The environment may appear differently to
different players, $\textit{i.e.}$, the reward distributions for a given arm
are heterogeneous across players. In the case of a collision (when more than
one player plays the same arm), we allow for the colliding players to receive
non-zero rewards. The time-horizon $T$ for which the arms are played is
\emph{not} known to the players. Within this setup, where the number of players
is allowed to be greater than the number of arms, we present a policy that
achieves near order-optimal expected regret of order $O(\log^{1 + \delta} T)$
for some $0 < \delta < 1$ over a time-horizon of duration $T$.
This paper is accepted at IEEE Transactions on Information Theory.

    

### [[1912.04212] Solving Bayesian Inverse Problems via Variational Autoencoders](http://arxiv.org/abs/1912.04212)


  In recent years, the field of machine learning has made phenomenal progress
in the pursuit of simulating real-world data generation processes. One notable
example of such success is the variational autoencoder (VAE). In this work,
with a small shift in perspective, we leverage and adapt VAEs for a different
purpose: uncertainty quantification in scientific inverse problems. We
introduce UQ-VAE: a flexible, adaptive, hybrid data/model-informed framework
for training neural networks capable of rapid modelling of the posterior
distribution representing the unknown parameter of interest. Specifically, from
divergence-based variational inference, our framework is derived such that most
of the information usually present in scientific inverse problems is fully
utilized in the training procedure. Additionally, this framework includes an
adjustable hyperparameter that allows selection of the notion of distance
between the posterior model and the target distribution. This introduces more
flexibility in controlling how optimization directs the learning of the
posterior model. Further, this framework possesses an inherent adaptive
optimization property that emerges through the learning of the posterior
uncertainty.

    

### [[2003.13511] Improved Gradient based Adversarial Attacks for Quantized Networks](http://arxiv.org/abs/2003.13511)


  Neural network quantization has become increasingly popular due to efficient
memory consumption and faster computation resulting from bitwise operations on
the quantized networks. Even though they exhibit excellent generalization
capabilities, their robustness properties are not well-understood. In this
work, we systematically study the robustness of quantized networks against
gradient based adversarial attacks and demonstrate that these quantized models
suffer from gradient vanishing issues and show a fake sense of robustness. By
attributing gradient vanishing to poor forward-backward signal propagation in
the trained network, we introduce a simple temperature scaling approach to
mitigate this issue while preserving the decision boundary. Despite being a
simple modification to existing gradient based adversarial attacks, experiments
on multiple image classification datasets with multiple network architectures
demonstrate that our temperature scaled attacks obtain near-perfect success
rate on quantized networks while outperforming original attacks on
adversarially trained models as well as floating-point networks. Code is
available at this https URL.

    

### [[2004.04650] State-Only Imitation Learning for Dexterous Manipulation](http://arxiv.org/abs/2004.04650)


  Modern model-free reinforcement learning methods have recently demonstrated
impressive results on a number of problems. However, complex domains like
dexterous manipulation remain a challenge due to the high sample complexity. To
address this, current approaches employ expert demonstrations in the form of
state-action pairs, which are difficult to obtain for real-world settings such
as learning from videos. In this paper, we move toward a more realistic setting
and explore state-only imitation learning. To tackle this setting, we train an
inverse dynamics model and use it to predict actions for state-only
demonstrations. The inverse dynamics model and the policy are trained jointly.
Our method performs on par with state-action approaches and considerably
outperforms RL alone. By not relying on expert actions, we are able to learn
from demonstrations with different dynamics, morphologies, and objects. Videos
available at this https URL .

    

### [[2006.03833] Domain Knowledge Alleviates Adversarial Attacks in Multi-Label Classifiers](http://arxiv.org/abs/2006.03833)


  Adversarial attacks on machine learning-based classifiers, along with defense
mechanisms, have been widely studied in the context of single-label
classification problems. In this paper, we shift the attention to multi-label
classification, where the availability of domain knowledge on the relationships
among the considered classes may offer a natural way to spot incoherent
predictions, i.e., predictions associated to adversarial examples lying outside
of the training data distribution. We explore this intuition in a framework in
which first-order logic knowledge is converted into constraints and injected
into a semi-supervised learning problem. Within this setting, the constrained
classifier learns to fulfill the domain knowledge over the marginal
distribution, and can naturally reject samples with incoherent predictions.
Even though our method does not exploit any knowledge of attacks during
training, our experimental analysis surprisingly unveils that domain-knowledge
constraints can help detect adversarial examples effectively, especially if
such constraints are not known to the attacker.

    

### [[2006.03978] Stable and Efficient Policy Evaluation](http://arxiv.org/abs/2006.03978)


  Policy evaluation algorithms are essential to reinforcement learning due to
their ability to predict the performance of a policy. However, there are two
long-standing issues lying in this prediction problem that need to be tackled:
off-policy stability and on-policy efficiency. The conventional temporal
difference (TD) algorithm is known to perform very well in the on-policy
setting, yet is not off-policy stable. On the other hand, the gradient TD and
emphatic TD algorithms are off-policy stable, but are not on-policy efficient.
This paper introduces novel algorithms that are both off-policy stable and
on-policy efficient by using the oblique projection method. The empirical
experimental results on various domains validate the effectiveness of the
proposed approach.

    

### [[2006.04981] A Framework for Neural Network Pruning Using Gibbs Distributions](http://arxiv.org/abs/2006.04981)


  Modern deep neural networks are often too large to use in many practical
scenarios. Neural network pruning is an important technique for reducing the
size of such models and accelerating inference. Gibbs pruning is a novel
framework for expressing and designing neural network pruning methods.
Combining approaches from statistical physics and stochastic regularization
methods, it can train and prune a network simultaneously in such a way that the
learned weights and pruning mask are well-adapted for each other. It can be
used for structured or unstructured pruning and we propose a number of specific
methods for each. We compare our proposed methods to a number of contemporary
neural network pruning methods and find that Gibbs pruning outperforms them. In
particular, we achieve a new state-of-the-art result for pruning ResNet-56 with
the CIFAR-10 dataset.

    

### [[2006.12800] Calibration of Neural Networks using Splines](http://arxiv.org/abs/2006.12800)


  Calibrating neural networks is of utmost importance when employing them in
safety-critical applications where the downstream decision making depends on
the predicted probabilities. Measuring calibration error amounts to comparing
two empirical distributions. In this work, we introduce a binning-free
calibration measure inspired by the classical Kolmogorov-Smirnov (KS)
statistical test in which the main idea is to compare the respective cumulative
probability distributions. From this, by approximating the empirical cumulative
distribution using a differentiable function via splines, we obtain a
recalibration function, which maps the network outputs to actual (calibrated)
class assignment probabilities. The spine-fitting is performed using a held-out
calibration set and the obtained recalibration function is evaluated on an
unseen test set. We tested our method against existing calibration approaches
on various image classification datasets and our spline-based recalibration
approach consistently outperforms existing methods on KS error as well as other
commonly used calibration measures. Our Code is available at
this https URL.

    

### [[2007.04954] ThreeDWorld: A Platform for Interactive Multi-Modal Physical Simulation](http://arxiv.org/abs/2007.04954)


  We introduce ThreeDWorld (TDW), a platform for interactive multi-modal
physical simulation. TDW enables simulation of high-fidelity sensory data and
physical interactions between mobile agents and objects in rich 3D
environments. Unique properties include: real-time near-photo-realistic image
rendering; a library of objects and environments, and routines for their
customization; generative procedures for efficiently building classes of new
environments; high-fidelity audio rendering; realistic physical interactions
for a variety of material types, including cloths, liquid, and deformable
objects; customizable agents that embody AI agents; and support for human
interactions with VR devices. TDW's API enables multiple agents to interact
within a simulation and returns a range of sensor and physics data representing
the state of the world. We present initial experiments enabled by TDW in
emerging research directions in computer vision, machine learning, and
cognitive science, including multi-modal physical scene understanding, physical
dynamics predictions, multi-agent interactions, models that learn like a child,
and attention studies in humans and neural networks.

    

### [[2008.00500] Structural Estimation of Partially Observable Markov Decision Processes](http://arxiv.org/abs/2008.00500)


  In many practical settings control decisions must be made under
partial/imperfect information about the evolution of a relevant state variable.
Partially Observable Markov Decision Processes (POMDPs) is a relatively
well-developed framework for modeling and analyzing such problems. In this
paper we consider the structural estimation of the primitives of a POMDP model
based upon the observable history of the process. We analyze the structural
properties of POMDP model with random rewards and specify conditions under
which the model is identifiable without knowledge of the state dynamics. We
consider a soft policy gradient algorithm to compute a maximum likelihood
estimator and provide a finite-time characterization of convergence to a
stationary point. We illustrate the estimation methodology with an application
to optimal equipment replacement. In this context, replacement decisions must
be made under partial/imperfect information on the true state (i.e. condition
of the equipment). We use synthetic and real data to highlight the robustness
of the proposed methodology and characterize the potential for misspecification
when partial state observability is ignored.

    

### [[2012.15465] Accelerating Neural ODE Based CNN Models on Low-Cost FPGAs](http://arxiv.org/abs/2012.15465)


  ODENet is a deep neural network architecture in which a stacking structure of
ResNet is implemented with an ordinary differential equation (ODE) solver.
ANODE is an extended approach of ODENet for stably high accuracy by introducing
more precise training algorithm using more memory in training phase. It is also
possible to improve the accuracy while keeping the same number of parameters on
resource-limited edge devices. In this paper, using Euler method as an ODE
solver, a part of ODENet and ANODE are implemented as a dedicated logic on a
low-cost FPGA (Field-Programmable Gate Array) board, such as PYNQ-Z2 board. As
ODENet variants, reduced ODENets (rODENets) each of which heavily uses a part
of ODENet layers and reduces/eliminates some layers differently are proposed
and analyzed for low-cost FPGA implementation. In addition, rANODE is proposed
as an ANODE variant in a way similar to the rODENet. They are evaluated in
terms of parameter size, accuracy, execution time, and resource utilization on
the FPGA. The results show that an overall execution time of rODENet and rANODE
variants is improved by up to 2.66 times compared to a pure software execution
while keeping a comparable accuracy to the original ODENet and ANODE.

    

### [[2101.02850] Designing Low-Correlation GPS Spreading Codes with a Natural Evolution Strategy Machine Learning Algorithm](http://arxiv.org/abs/2101.02850)


  With the birth of the next-generation GPS III constellation and the upcoming
launch of the Navigation Technology Satellite-3 (NTS-3) testing platform to
explore future technologies for GPS, we are indeed entering a new era of
satellite navigation. Correspondingly, it is time to revisit the design methods
of the GPS spreading code families. In this work, we develop a natural
evolution strategy (NES) machine learning algorithm with a Gaussian proposal
distribution which constructs high-quality families of spreading code
sequences. We minimize the maximum between the mean-squared auto-correlation
and the mean-squared cross-correlation and demonstrate the ability of our
algorithm to achieve better performance than well-chosen families of
equal-length Gold codes and Weil codes, for sequences of up to length-1023 and
length-1031 bits and family sizes of up to 31 codes. Furthermore, we compare
our algorithm with an analogous genetic algorithm implementation assigned the
same code evaluation metric. To the best of the authors' knowledge, this is the
first work to explore using a machine learning approach for designing
navigation spreading code sequences.

    

### [[2102.06924] Online Apprenticeship Learning](http://arxiv.org/abs/2102.06924)


  In Apprenticeship Learning (AL), we are given a Markov Decision Process (MDP)
without access to the cost function. Instead, we observe trajectories sampled
by an expert that acts according to some policy. The goal is to find a policy
that matches the expert's performance on some predefined set of cost functions.
We introduce an online variant of AL (Online Apprenticeship Learning; OAL),
where the agent is expected to perform comparably to the expert while
interacting with the environment. We show that the OAL problem can be
effectively solved by combining two mirror descent based no-regret algorithms:
one for policy optimization and another for learning the worst case cost. By
employing optimistic exploration, we derive a convergent algorithm with
$O(\sqrt{K})$ regret, where $K$ is the number of interactions with the MDP, and
an additional linear error term that depends on the amount of expert
trajectories available. Importantly, our algorithm avoids the need to solve an
MDP at each iteration, making it more practical compared to prior AL methods.
Finally, we implement a deep variant of our algorithm which shares some
similarities to GAIL \cite{ho2016generative}, but where the discriminator is
replaced with the costs learned by the OAL problem. Our simulations suggest
that OAL performs well in high dimensional control problems.

    

### [[2103.09424] Escaping Saddle Points in Distributed Newton's Method with Communication Efficiency and Byzantine Resilience](http://arxiv.org/abs/2103.09424)


  The problem of saddle-point avoidance for non-convex optimization is quite
challenging in large scale distributed learning frameworks, such as Federated
Learning, especially in the presence of Byzantine workers. The celebrated
cubic-regularized Newton method of \cite{nest} is one of the most elegant ways
to avoid saddle-points in the standard centralized (non-distributed) setup. In
this paper, we extend the cubic-regularized Newton method to a distributed
framework and simultaneously address several practical challenges like
communication bottleneck and Byzantine attacks. Note that the issue of
saddle-point avoidance becomes more crucial in the presence of Byzantine
machines since rogue machines may create \emph{fake local minima} near the
saddle-points of the loss function, also known as the saddle-point attack.
Being a second order algorithm, our iteration complexity is much lower than the
first order counterparts. Furthermore we use compression (or sparsification)
techniques like $\delta$-approximate compression for communication efficiency.
We obtain theoretical guarantees for our proposed scheme under several settings
including approximate (sub-sampled) gradients and Hessians. Moreover, we
validate our theoretical findings with experiments using standard datasets and
several types of Byzantine attacks, and obtain an improvement of $25\%$ with
respect to first order methods in iteration complexity.

    

### [[2103.11919] Machine Learning Emulation of 3D Cloud Radiative Effects](http://arxiv.org/abs/2103.11919)


  The treatment of cloud structure in numerical weather and climate models is
often greatly simplified to make them computationally affordable. Here we
propose to correct the European Centre for Medium-Range Weather Forecasts 1D
radiation scheme ecRad for 3D cloud effects using computationally cheap neural
networks. 3D cloud effects are learned as the difference between ecRad's fast
1D Tripleclouds solver that neglects them and its 3D SPARTACUS (SPeedy
Algorithm for Radiative TrAnsfer through CloUd Sides) solver that includes them
but is about five times more computationally expensive. With typical errors
between 20 and 30 % of the 3D signal, neural networks improve Tripleclouds'
accuracy for about 1 % increase in runtime. Thus, rather than emulating the
whole of SPARTACUS, we keep Tripleclouds unchanged for cloud-free parts of the
atmosphere and 3D-correct it elsewhere. The focus on the comparably small 3D
correction instead of the entire signal allows to improve predictions
significantly if we assume a similar signal-to-noise ratio for both.

    

### [[2103.12532] Balanced Softmax Cross-Entropy for Incremental Learning](http://arxiv.org/abs/2103.12532)


  Deep neural networks are prone to catastrophic forgetting when incrementally
trained on new classes or new tasks as adaptation to the new data leads to a
drastic decrease of the performance on the old classes and tasks. By using a
small memory for rehearsal and knowledge distillation, recent methods have
proven to be effective to mitigate catastrophic forgetting. However due to the
limited size of the memory, large imbalance between the amount of data
available for the old and new classes still remains which results in a
deterioration of the overall accuracy of the model. To address this problem, we
propose the use of the Balanced Softmax Cross-Entropy loss and show that it can
be combined with exiting methods for incremental learning to improve their
performances while also decreasing the computational cost of the training
procedure in some cases. Experiments on the competitive ImageNet, subImageNet
and CIFAR100 datasets show states-of-the-art results.

    

### [[2103.15330] Extending Multi-Sense Word Embedding to Phrases and Sentences for Unsupervised Semantic Applications](http://arxiv.org/abs/2103.15330)


  Most unsupervised NLP models represent each word with a single point or
single region in semantic space, while the existing multi-sense word embeddings
cannot represent longer word sequences like phrases or sentences. We propose a
novel embedding method for a text sequence (a phrase or a sentence) where each
sequence is represented by a distinct set of multi-mode codebook embeddings to
capture different semantic facets of its meaning. The codebook embeddings can
be viewed as the cluster centers which summarize the distribution of possibly
co-occurring words in a pre-trained word embedding space. We introduce an
end-to-end trainable neural model that directly predicts the set of cluster
centers from the input text sequence during test time. Our experiments show
that the per-sentence codebook embeddings significantly improve the
performances in unsupervised sentence similarity and extractive summarization
benchmarks. In phrase similarity experiments, we discover that the multi-facet
embeddings provide an interpretable semantic representation but do not
outperform the single-facet baseline.

    

### [[2106.02285] Subdivision-Based Mesh Convolution Networks](http://arxiv.org/abs/2106.02285)


  Convolutional neural networks (CNNs) have made great breakthroughs in 2D
computer vision. However, their irregular structure makes it hard to harness
the potential of CNNs directly on meshes. A subdivision surface provides a
hierarchical multi-resolution structure, in which each face in a closed
2-manifold triangle mesh is exactly adjacent to three faces. Motivated by these
two observations, this paper presents SubdivNet, an innovative and versatile
CNN framework for 3D triangle meshes with Loop subdivision sequence
connectivity. Making an analogy between mesh faces and pixels in a 2D image
allows us to present a mesh convolution operator to aggregate local features
from nearby faces. By exploiting face neighborhoods, this convolution can
support standard 2D convolutional network concepts, e.g. variable kernel size,
stride, and dilation. Based on the multi-resolution hierarchy, we make use of
pooling layers which uniformly merge four faces into one and an upsampling
method which splits one face into four. Thereby, many popular 2D CNN
architectures can be easily adapted to process 3D meshes. Meshes with arbitrary
connectivity can be remeshed to have Loop subdivision sequence connectivity via
self-parameterization, making SubdivNet a general approach. Extensive
evaluation and various applications demonstrate SubdivNet's effectiveness and
efficiency.

    

### [[2106.02602] InDiD: Instant Disorder Detection via Representation Learning](http://arxiv.org/abs/2106.02602)


  For sequential data, change points are moments of abrupt regime switches.
Such changes appear in different scenarios, including complex video
surveillance, and we need to detect them as fast as possible. Classic
approaches for change point detection (CPD) perform poorly for semi-structured
sequential data because of the absence of adequate data representation learning
procedure. We propose a principled loss function that approximates classic
rigorous solutions but is differentiable and makes possible representation
learning. This loss function balances change detection delay and time to false
alarm to provide a successful model for CPD. In experiments, we consider simple
series and more complex real-world image sequences and videos with change
points. For more complex problems, we show that we need more meaningful
representations tailored for the specificity of the CPD task. Taking this into
account, the proposed approach InDiD improves baseline results of CPD for
various data types. For explosion detection, F1 score for our method is $0.54$
compared to baseline scores $0.46$ and $0.30$.

    

### [[2106.05934] Flow-based sampling for fermionic lattice field theories](http://arxiv.org/abs/2106.05934)


  Algorithms based on normalizing flows are emerging as promising machine
learning approaches to sampling complicated probability distributions in a way
that can be made asymptotically exact. In the context of lattice field theory,
proof-of-principle studies have demonstrated the effectiveness of this approach
for scalar theories, gauge theories, and statistical systems. This work
develops approaches that enable flow-based sampling of theories with dynamical
fermions, which is necessary for the technique to be applied to lattice field
theory studies of the Standard Model of particle physics and many condensed
matter systems. As a practical demonstration, these methods are applied to the
sampling of field configurations for a two-dimensional theory of massless
staggered fermions coupled to a scalar field via a Yukawa interaction.

    

### [[2106.07214] Backdoor Learning Curves: Explaining Backdoor Poisoning Beyond Influence Functions](http://arxiv.org/abs/2106.07214)


  Backdoor attacks inject poisoning samples during training, with the goal of
forcing a machine learning model to output an attacker-chosen class when
presented a specific trigger at test time. Although backdoor attacks have been
demonstrated in a variety of settings and against different models, the factors
affecting their effectiveness are still not well understood. In this work, we
provide a unifying framework to study the process of backdoor learning under
the lens of incremental learning and influence functions. We show that the
effectiveness of backdoor attacks depends on: (i) the complexity of the
learning algorithm, controlled by its hyperparameters; (ii) the fraction of
backdoor samples injected into the training set; and (iii) the size and
visibility of the backdoor trigger. These factors affect how fast a model
learns to correlate the presence of the backdoor trigger with the target class.
Our analysis unveils the intriguing existence of a region in the hyperparameter
space in which the accuracy on clean test samples is still high while backdoor
attacks are ineffective, thereby suggesting novel criteria to improve existing
defenses.

    

### [[2110.01604] CertainNet: Sampling-free Uncertainty Estimation for Object Detection](http://arxiv.org/abs/2110.01604)


  Estimating the uncertainty of a neural network plays a fundamental role in
safety-critical settings. In perception for autonomous driving, measuring the
uncertainty means providing additional calibrated information to downstream
tasks, such as path planning, that can use it towards safe navigation. In this
work, we propose a novel sampling-free uncertainty estimation method for object
detection. We call it CertainNet, and it is the first to provide separate
uncertainties for each output signal: objectness, class, location and size. To
achieve this, we propose an uncertainty-aware heatmap, and exploit the
neighboring bounding boxes provided by the detector at inference time. We
evaluate the detection performance and the quality of the different uncertainty
estimates separately, also with challenging out-of-domain samples: BDD100K and
nuImages with models trained on KITTI. Additionally, we propose a new metric to
evaluate location and size uncertainties. When transferring to unseen datasets,
CertainNet generalizes substantially better than previous methods and an
ensemble, while being real-time and providing high quality and comprehensive
uncertainty estimates.

    

### [[2112.03152] Bounding Wasserstein distance with couplings](http://arxiv.org/abs/2112.03152)


  Markov chain Monte Carlo (MCMC) provides asymptotically consistent estimates
of intractable posterior expectations as the number of iterations tends to
infinity. However, in large data applications, MCMC can be computationally
expensive per iteration. This has catalyzed interest in sampling methods such
as approximate MCMC, which trade off asymptotic consistency for improved
computational speed. In this article, we propose estimators based on couplings
of Markov chains to assess the quality of such asymptotically biased sampling
methods. The estimators give empirical upper bounds of the Wassertein distance
between the limiting distribution of the asymptotically biased sampling method
and the original target distribution of interest. We establish theoretical
guarantees for our upper bounds and show that our estimators can remain
effective in high dimensions. We apply our quality measures to stochastic
gradient MCMC, variational Bayes, and Laplace approximations for tall data and
to approximate MCMC for Bayesian logistic regression in 4500 dimensions and
Bayesian linear regression in 50000 dimensions.

    

### [[2112.13705] Graph Collaborative Reasoning](http://arxiv.org/abs/2112.13705)


  Graphs can represent relational information among entities and graph
structures are widely used in many intelligent tasks such as search,
recommendation, and question answering. However, most of the graph-structured
data in practice suffers from incompleteness, and thus link prediction becomes
an important research problem. Though many models are proposed for link
prediction, the following two problems are still less explored: (1) Most
methods model each link independently without making use of the rich
information from relevant links, and (2) existing models are mostly designed
based on associative learning and do not take reasoning into consideration.
With these concerns, in this paper, we propose Graph Collaborative Reasoning
(GCR), which can use the neighbor link information for relational reasoning on
graphs from logical reasoning perspectives. We provide a simple approach to
translate a graph structure into logical expressions, so that the link
prediction task can be converted into a neural logic reasoning problem. We
apply logical constrained neural modules to build the network architecture
according to the logical expression and use back propagation to efficiently
learn the model parameters, which bridges differentiable learning and symbolic
reasoning in a unified architecture. To show the effectiveness of our work, we
conduct experiments on graph-related tasks such as link prediction and
recommendation based on commonly used benchmark datasets, and our graph
collaborative reasoning approach achieves state-of-the-art performance.

    

### [[2112.13778] Dynamic Time Warping Clustering to Discover Socio-Economic Characteristics in Smart Water Meter Data](http://arxiv.org/abs/2112.13778)


  Socio-economic characteristics are influencing the temporal and spatial
variability of water demand - the biggest source of uncertainties within water
distribution system modeling. Improving our knowledge on these influences can
be utilized to decrease demand uncertainties. This paper aims to link smart
water meter data to socio-economic user characteristics by applying a novel
clustering algorithm that uses dynamic time warping on daily demand patterns.
The approach is tested on simulated and measured single family home datasets.
We show that the novel algorithm performs better compared to commonly used
clustering methods, both, in finding the right number of clusters as well as
assigning patterns correctly. Additionally, the methodology can be used to
identify outliers within clusters of demand patterns. Furthermore, this study
investigates which socio-economic characteristics (e.g. employment status,
number of residents) are prevalent within single clusters and, consequently,
can be linked to the shape of the cluster's barycenters. In future, the
proposed methods in combination with stochastic demand models can be used to
fill data-gaps in hydraulic models.

    

### [[1712.00481] Intelligent EHRs: Predicting Procedure Codes From Diagnosis Codes](http://arxiv.org/abs/1712.00481)


  In order to submit a claim to insurance companies, a doctor needs to code a
patient encounter with both the diagnosis (ICDs) and procedures performed
(CPTs) in an Electronic Health Record (EHR). Identifying and applying relevant
procedures code is a cumbersome and time-consuming task as a doctor has to
choose from around 13,000 procedure codes with no predefined one-to-one
mapping. In this paper, we propose a state-of-the-art deep learning method for
automatic and intelligent coding of procedures (CPTs) from the diagnosis codes
(ICDs) entered by the doctor. Precisely, we cast the learning problem as a
multi-label classification problem and use distributed representation to learn
the input mapping of high-dimensional sparse ICDs codes. Our final model
trained on 2.3 million claims is able to outperform existing rule-based
probabilistic and association-rule mining based methods and has a recall of
90@3.

    

### [[2112.14013] Reducing Minor Page Fault Overheads through Enhanced Page Walker](http://arxiv.org/abs/2112.14013)


  Application virtual memory footprints are growing rapidly in all systems from
servers to smartphones. To address this growing demand, system integrators are
incorporating larger amounts of main memory, warranting rethinking of memory
management. In current systems, applications produce page faults whenever they
access virtual memory regions that are not backed by a physical page. As
application memory footprints grow, they induce more and more minor pagefaults.
Handling of each minor page fault can take few 1000's of CPU-cycles and blocks
the application till OS kernel finds a free physical frame. These page faults
can be detrimental to the performance when their frequency of occurrence is
high and spread across application run-time. Our evaluation of several
workloads indicates an overhead due to minor page faults as high as 29% of
execution time. In this paper, we propose to mitigate this problem through a
HW/SW co-design approach. Specifically, we first propose to parallelize
portions of the kernel page allocation to run ahead of fault time in a separate
thread. Then we propose the Minor Fault Offload Engine(MFOE), a per-core HW
accelerator for minor fault handling. MFOE is equipped with pre-allocated frame
table that it uses to service a page fault. On a page fault, MFOE quickly picks
a pre-allocated page frame from this table, makes an entry for it in the TLB,
and updates the page table entry to satisfy the page fault. The pre-allocation
frame tables are periodically refreshed by a background thread, which also
updates the data structures in the kernel to account for the handled page
faults. We evaluate this system in the gem5 simulator with a modified Linux
kernel running on top of simulated hardware. Our results show that MFOE
improves the critical-path fault handling latency by 37x and improves the
run-time amongst the evaluated applications, by an average of 7.5%

    

### [[2112.14216] Casper: Accelerating Stencil Computation using Near-cache Processing](http://arxiv.org/abs/2112.14216)


  Stencil computation is one of the most used kernels in a wide variety of
scientific applications, ranging from large-scale weather prediction to solving
partial differential equations. Stencil computations are characterized by three
unique properties: (1) low arithmetic intensity, (2) limited temporal data
reuse, and (3) regular and predictable data access pattern. As a result,
stencil computations are typically bandwidth-bound workloads, which only
experience limited benefits from the deep cache hierarchy of modern CPUs. In
this work, we propose Casper, a near-cache accelerator consisting of
specialized stencil compute units connected to the last-level cache (LLC) of a
traditional CPU. Casper is based on two key ideas: (1) avoiding the cost of
moving rarely reused data through the cache hierarchy, and (2) exploiting the
regularity of the data accesses and the inherent parallelism of the stencil
computation to increase the overall performance.
With minimal changes in LLC address decoding logic and data placement, Casper
performs stencil computations at the peak bandwidth of the LLC. We show that,
by tightly coupling lightweight stencil compute units near to LLC, Casper
improves the performance of stencil kernels by 1.65x on average, while reducing
the energy consumption by 35% compared to a commercial high-performance
multi-core processor. Moreover, Casper provides a 37x improvement in
performance-per-area compared to a state-of-the-art GPU.

    

### [[2112.13841] Comparative Analysis of Different Techniques of Real Time Scheduling for Multi-Core Platform](http://arxiv.org/abs/2112.13841)


  As the demand of real time computing increases day by day, there is a major
paradigm shift in processing platform of real time system from single core to
multi-core platform which provides advantages like higher throughput, linear
power consumption, efficient utilization of processor cores and high
performance per unit cost over the many single core processors unit. Currently
available most popular real time schedulers for multi-core domain are
partitioned and global scheduling and these schedulers not suitable to
efficiently use this multi-core platform efficiently. Although,
semi-partitioned algorithms increases utilization bound by using spare
capacities left by partitioning via global scheduling, it has a inherent
disadvantage of off-line task splitting. Although, semi-partitioned algorithms
increases utilization bound by using spare capacities left by partitioning via
global scheduling, it has a inherent disadvantage of off-line task splitting.
To overcome these problems of multi-core real time scheduling algorithm new
dynamic cluster based multi-core real time scheduling algorithm proposed which
is hybrid scheduling approach. This paper discuss different multi-core
scheduling techniques and comparative analysis of these techniques with the
proposed dynamic cluster based real time multi-core scheduling

    

### [[2112.13875] Design and Experimental Evaluation of Algorithms for Optimizing the Throughput of Dispersed Computing](http://arxiv.org/abs/2112.13875)


  With growing deployment of Internet of Things (IoT) and machine learning (ML)
applications, which need to leverage computation on edge and cloud resources,
it is important to develop algorithms and tools to place these distributed
computations to optimize their performance. We address the problem of optimally
placing computations (described as directed acyclic graphs (DAGs)) on a set of
machines to maximize the steady-state throughput for pipelined inputs.
Traditionally, such optimization has focused on a different metric, minimizing
single-shot makespan, and a well-known algorithm is the Heterogeneous Earliest
Finish Time (HEFT) algorithm. Maximizing throughput however, is more suitable
for many real-time, edge, cloud and IoT applications, we present a different
scheduling algorithm, namely Throughput HEFT (TPHEFT). Further, we present two
throughput-oriented enhancements which can be applied to any baseline schedule,
that we refer to as "node splitting" (SPLIT) and "task duplication" (DUP). In
order to implement and evaluate these algorithms, we built new subsystems and
plugins for an open-source dispersed computing framework called Jupiter.
Experiments with varying DAG structures indicate that: 1) TPHEFT can
significantly improve throughput performance compared to HEFT (up to 2.3 times
in our experiments), with greater gains when there is less degree of
parallelism in the DAG, 2) Node splitting can potentially improve performance
over a baseline schedule, with greater gains when there's an imbalanced
allocation of computation or inter-task communication, and 3) Task duplication
generally gives improvements only when running upon a baseline that places
communication over slow links. To our knowledge, this is the first study to
present a systematic experimental implementation and exploration of
throughput-enhancing techniques for dispersed computing on real testbeds.

    

### [[2112.13972] HiKonv: High Throughput Quantized Convolution With Novel Bit-wise Management and Computation](http://arxiv.org/abs/2112.13972)


  Quantization for Convolutional Neural Network (CNN) has shown significant
progress with the intention of reducing the cost of computation and storage
with low-bitwidth data inputs. There are, however, no systematic studies on how
an existing full-bitwidth processing unit, such as CPUs and DSPs, can be better
utilized to carry out significantly higher computation throughput for
convolution under various quantized bitwidths. In this study, we propose
HiKonv, a unified solution that maximizes the compute throughput of a given
underlying processing unit to process low-bitwidth quantized data inputs
through novel bit-wise parallel computation. We establish theoretical
performance bounds using a full-bitwidth multiplier for highly parallelized
low-bitwidth convolution, and demonstrate new breakthroughs for
high-performance computing in this critical domain. For example, a single
32-bit processing unit can deliver 128 binarized convolution operations
(multiplications and additions) under one CPU instruction, and a single 27x18
DSP core can deliver eight convolution operations with 4-bit inputs in one
cycle. We demonstrate the effectiveness of HiKonv on CPU and FPGA for both
convolutional layers or a complete DNN model. For a convolutional layer
quantized to 4-bit, HiKonv achieves a 3.17x latency improvement over the
baseline implementation using C++ on CPU. Compared to the DAC-SDC 2020 champion
model for FPGA, HiKonv achieves a 2.37x throughput improvement and 2.61x DSP
efficiency improvement, respectively.

    

### [[2112.14349] Fast Subspace Identification Method Based on Containerised Cloud Workflow Processing System](http://arxiv.org/abs/2112.14349)


  Subspace identification (SID) has been widely used in system identification
and control fields since it can estimate system models only relying on the
input and output data by reliable numerical operations such as singular value
decomposition (SVD). However, high-dimension Hankel matrices are involved to
store these data and used to obtain the system models, which increases the
computation amount of SID and leads SID not suitable for the large-scale or
real-time identification tasks. In this paper, a novel fast SID method based on
cloud workflow processing and container technology is proposed to accelerate
the traditional algorithm. First, a workflow-based structure of SID is designed
to match the distributed cloud environment, based on the computational feature
of each calculation stage. Second, a containerised cloud workflow processing
system is established to execute the logic- and data- dependent SID workflow
mission based on Kubernetes system. Finally, the experiments show that the
computation time is reduced by at most $91.6\%$ for large-scale SID mission and
decreased to within 20 ms for the real-time mission parameter.

    

### [[2112.14468] Challenges and approaches for mitigating byzantine attacks in federated learning](http://arxiv.org/abs/2112.14468)


  Recently emerged federated learning (FL) is an attractive distributed
learning framework in which numerous wireless end-user devices can train a
global model with the data remained autochthonous. Compared with the
traditional machine learning framework that collects user data for centralized
storage, which brings huge communication burden and concerns about data
privacy, this approach can not only save the network bandwidth but also protect
the data privacy. Despite the promising prospect, byzantine attack, an
intractable threat in conventional distributed network, is discovered to be
rather efficacious against FL as well. In this paper, we conduct a
comprehensive investigation of the state-of-the-art strategies for defending
against byzantine attacks in FL. We first provide a taxonomy for the existing
defense solutions according to the techniques they used, followed by an
across-the-board comparison and discussion. Then we propose a new byzantine
attack method called weight attack to defeat those defense schemes, and conduct
experiments to demonstrate its threat. The results show that existing defense
solutions, although abundant, are still far from fully protecting FL. Finally,
we indicate possible countermeasures for weight attack, and highlight several
challenges and future research directions for mitigating byzantine attacks in
FL.

    

### [[2112.14655] Broadcasting on Adversarial Multiple Access Channels](http://arxiv.org/abs/2112.14655)


  We study broadcasting on multiple-access channels under adversarial packet
injection. Leaky-bucket adversaries model packet injection. There is a fixed
set of stations attached to a channel. Additional constrains on the model
include bounds on the number of stations activated at a round, individual
injection rates, and randomness in generating and injecting packets. Broadcast
algorithms that we concentrate on are deterministic and distributed. We
demonstrate that some broadcast algorithms designed for ad-hoc channels have
bounded latency for wider ranges of injection rates when executed on channels
with a fixed number of stations against adversaries that can activate at most
one station per round. Individual injection rates are shown to impact latency,
as compared to the model of general leaky bucket adversaries. Outcomes of
experiments are given that compare the performance of broadcast algorithms
against randomized adversaries. The experiments include randomized backoff
algorithms.

    

### [[2112.14680] WaCoDiS: Automated Earth Observation Data Processing within an Event-Driven Architecture for Water Monitoring](http://arxiv.org/abs/2112.14680)


  To ensure an efficient and environmentally friendly water resource
management, water management associations need means for efficient water
monitoring as well as novel strategies to reduce the pollution of surface and
ground water. Traditionally, water management associations operate large sensor
networks to suffice their needs for hydrological and meteorological measurement
data to monitor and model physical processes within catchments. Implementing a
comprehensive monitoring system often suffers from sparse coverage of in-situ
data. Due to the evolvement of the Copernicus satellite platforms, the broader
availability of satellite data provides a great potential for deriving
complementary information from Earth Observation data. Although the number of
satellite data platforms that provide online processing environments is
growing, it is still a big challenge to integrate those platforms into
traditional workflows of users from environmental domains such as hydrology.
Thus, in this paper, we introduce a software architecture to facilitate the
generation of Earth Observation information targeted towards hydrology. The
presented WaCoDiS System comprises several microservices as well standardized
interfaces that enable a platform-independent processing of satellite data.
First, we discuss the contribution of Earth Observation data to water
monitoring and derive several challenges regarding the facilitation of
satellite data processing. We then describe our system design with a brief
overview about the different system components which form an automated
processing pipeline. The suitability of our system is proven as part of a
pre-operational deployment for a German water management association. In
addition, we demonstrate how our system is capable of integrating satellite
data platforms, using the Copernicus Data and Exploitation Platform -
Deutschland (CODE-DE) as a reference example.

    

### [[1812.07183] Optimizing Data Intensive Flows for Networks on Chips](http://arxiv.org/abs/1812.07183)


  Data flow analysis and optimization is considered for homogeneous rectangular
mesh networks. We propose a flow matrix equation which allows a closed-form
characterization of the nature of the minimal time solution, speedup and a
simple method to determine when and how much load to distribute to processors.
We also propose a rigorous mathematical proof about the flow matrix optimal
solution existence and that the solution is unique. The methodology introduced
here is applicable to many interconnection networks and switching protocols (as
an example we examine toroidal networks and hypercube networks in this paper).
An important application is improving chip area and chip scalability for
networks on chips processing divisible style loads.

    

### [[2102.05301] Parallel Minimum Cuts in $O(m \log^2(n))$ Work and Low Depth](http://arxiv.org/abs/2102.05301)


  We present a randomized $O(m \log^2 n)$ work, $O(\text{polylog } n)$ depth
parallel algorithm for minimum cut. This algorithm matches the work bounds of a
recent sequential algorithm by Gawrychowski, Mozes, and Weimann [ICALP'20], and
improves on the previously best parallel algorithm by Geissmann and Gianinazzi
[SPAA'18], which performs $O(m \log^4 n)$ work in $O(\text{polylog } n)$ depth.
Our algorithm makes use of three components that might be of independent
interest. Firstly, we design a parallel data structure that efficiently
supports batched mixed queries and updates on trees. It generalizes and
improves the work bounds of a previous data structure of Geissmann and
Gianinazzi and is work efficient with respect to the best sequential algorithm.
Secondly, we design a parallel algorithm for approximate minimum cut that
improves on previous results by Karger and Motwani. We use this algorithm to
give a work-efficient procedure to produce a tree packing, as in Karger's
sequential algorithm for minimum cuts. Lastly, we design an efficient parallel
algorithm for solving the minimum $2$-respecting cut problem.

    

### [[2103.08983] PerfSim: A Performance Simulator for Cloud Native Microservice Chains](http://arxiv.org/abs/2103.08983)


  Cloud native computing paradigm allows microservice-based applications to
take advantage of cloud infrastructure in a scalable, reusable, and
interoperable way. However, in a cloud native system, the vast number of
configuration parameters and highly granular resource allocation policies can
significantly impact the performance and deployment cost. For understanding and
analyzing these implications in an easy, quick, and cost-effective way, we
present PerfSim, a discrete-event simulator for approximating and predicting
the performance of cloud native service chains in user-defined scenarios. To
this end, we proposed a systematic approach for modeling the performance of
microservices endpoint functions by collecting and analyzing their performance
and network traces. With a combination of the extracted models and user-defined
scenarios, PerfSim can then simulate the performance behavior of all services
over a given period and provide an approximation for system KPIs, such as
requests' average response time. Using the processing power of a single laptop,
we evaluated both simulation accuracy and speed of PerfSim in 104 prevalent
scenarios and compared the simulation results with the identical deployment in
a real Kubernetes cluster. We achieved ~81-99% simulation accuracy in
approximating the average response time of incoming requests and ~16-1200 times
speed-up factor for the simulation.

    

### [[2104.00131] Distributed Banach-Picard Iteration for Locally Contractive Maps](http://arxiv.org/abs/2104.00131)


  The Banach-Picard iteration is widely used to find fixed points of locally
contractive (LC) maps. This paper extends the Banach-Picard iteration to
distributed settings; specifically, we assume the map of which the fixed point
is sought to be the average of individual (not necessarily LC) maps held by a
set of agents linked by a communication network. An additional difficulty is
that the LC map is not assumed to come from an underlying optimization problem,
which prevents exploiting strong global properties such as convexity or
Lipschitzianity. Yet, we propose a distributed algorithm and prove its
convergence, in fact showing that it maintains the linear rate of the standard
Banach-Picard iteration for the average LC map. As another contribution, our
proof imports tools from perturbation theory of linear operators, which, to the
best of our knowledge, had not been used before in the theory of distributed
computation.

    

### [[2112.13891] GPU-accelerated Faster Mean Shift with euclidean distance metrics](http://arxiv.org/abs/2112.13891)


  Handling clustering problems are important in data statistics, pattern
recognition and image processing. The mean-shift algorithm, a common
unsupervised algorithms, is widely used to solve clustering problems. However,
the mean-shift algorithm is restricted by its huge computational resource cost.
In previous research[10], we proposed a novel GPU-accelerated Faster Mean-shift
algorithm, which greatly speed up the cosine-embedding clustering problem. In
this study, we extend and improve the previous algorithm to handle Euclidean
distance metrics. Different from conventional GPU-based mean-shift algorithms,
our algorithm adopts novel Seed Selection & Early Stopping approaches, which
greatly increase computing speed and reduce GPU memory consumption. In the
simulation testing, when processing a 200K points clustering problem, our
algorithm achieved around 3 times speedup compared to the state-of-the-art
GPU-based mean-shift algorithms with optimized GPU memory consumption.
Moreover, in this study, we implemented a plug-and-play model for faster
mean-shift algorithm, which can be easily deployed. (Plug-and-play model is
available: this https URL)

    

### [[2112.13910] Visual Persuasion in COVID-19 Social Media Content: A Multi-Modal Characterization](http://arxiv.org/abs/2112.13910)


  Social media content routinely incorporates multi-modal design to covey
information and shape meanings, and sway interpretations toward desirable
implications, but the choices and outcomes of using both texts and visual
images have not been sufficiently studied. This work proposes a computational
approach to analyze the outcome of persuasive information in multi-modal
content, focusing on two aspects, popularity and reliability, in
COVID-19-related news articles shared on Twitter. The two aspects are
intertwined in the spread of misinformation: for example, an unreliable article
that aims to misinform has to attain some popularity. This work has several
contributions. First, we propose a multi-modal (image and text) approach to
effectively identify popularity and reliability of information sources
simultaneously. Second, we identify textual and visual elements that are
predictive to information popularity and reliability. Third, by modeling
cross-modal relations and similarity, we are able to uncover how unreliable
articles construct multi-modal meaning in a distorted, biased fashion. Our work
demonstrates how to use multi-modal analysis for understanding influential
content and has implications to social media literacy and engagement.

    

### [[2112.13925] Improving Depth Estimation using Location Information](http://arxiv.org/abs/2112.13925)


  The ability to accurately estimate depth information is crucial for many
autonomous applications to recognize the surrounded environment and predict the
depth of important objects. One of the most recently used techniques is
monocular depth estimation where the depth map is inferred from a single image.
This paper improves the self-supervised deep learning techniques to perform
accurate generalized monocular depth estimation. The main idea is to train the
deep model to take into account a sequence of the different frames, each frame
is geotagged with its location information. This makes the model able to
enhance depth estimation given area semantics. We demonstrate the effectiveness
of our model to improve depth estimation results. The model is trained in a
realistic environment and the results show improvements in the depth map after
adding the location data to the model training phase.

    

### [[2112.13937] Multiagent Model-based Credit Assignment for Continuous Control](http://arxiv.org/abs/2112.13937)


  Deep reinforcement learning (RL) has recently shown great promise in robotic
continuous control tasks. Nevertheless, prior research in this vein center
around the centralized learning setting that largely relies on the
communication availability among all the components of a robot. However, agents
in the real world often operate in a decentralised fashion without
communication due to latency requirements, limited power budgets and safety
concerns. By formulating robotic components as a system of decentralised
agents, this work presents a decentralised multiagent reinforcement learning
framework for continuous control. To this end, we first develop a cooperative
multiagent PPO framework that allows for centralized optimisation during
training and decentralised operation during execution. However, the system only
receives a global reward signal which is not attributed towards each agent. To
address this challenge, we further propose a generic game-theoretic credit
assignment framework which computes agent-specific reward signals. Last but not
least, we also incorporate a model-based RL module into our credit assignment
framework, which leads to significant improvement in sample efficiency. We
demonstrate the effectiveness of our framework on experimental results on
Mujoco locomotion control tasks. For a demo video please visit:
this https URL.

    

### [[2112.13984] Relative velocity-based reward functions for crowd navigation of robots](http://arxiv.org/abs/2112.13984)


  How to navigate effectively in crowd environments with socially acceptable
standards remains the key problem to be solved for the development of mobile
robots. Recent work has shown the effectiveness of deep reinforcement learning
in addressing crowd navigation, but the learning becomes progressively less
effective as the speed of pedestrians increases. To improve the effectiveness
of deep reinforcement learning, we redesigned the reward function by
introducing the penalty term of relative speed in the reward function. The
newly designed reward function is tested on three mainstream deep reinforcement
learning algorithms: deep reinforcement learning collision avoidance (CADRL),
deep learning based long and short-term memory (LSTM RL), and reinforcement
learning based on socialist riselection (SARL). The results of the experiments
show that our model navigates in a safer way, outperforming the current model
in key metrics such as success rate, collision rate, and hazard frequency.

    

### [[2112.14005] Towards Relatable Explainable AI with the Perceptual Process](http://arxiv.org/abs/2112.14005)


  Machine learning models need to provide contrastive explanations, since
people often seek to understand why a puzzling prediction occurred instead of
some expected outcome. Current contrastive explanations are rudimentary
comparisons between examples or raw features, which remain difficult to
interpret, since they lack semantic meaning. We argue that explanations must be
more relatable to other concepts, hypotheticals, and associations. Inspired by
the perceptual process from cognitive psychology, we propose the XAI Perceptual
Processing Framework and RexNet model for relatable explainable AI with
Contrastive Saliency, Counterfactual Synthetic, and Contrastive Cues
explanations. We investigated the application of vocal emotion recognition, and
implemented a modular multi-task deep neural network to predict and explain
emotions from speech. From think-aloud and controlled studies, we found that
counterfactual explanations were useful and further enhanced with semantic
cues, but not saliency explanations. This work provides insights into providing
and evaluating relatable contrastive explainable AI for perception
applications.

    

### [[2112.14072] Unsupervised Domain Adaptation for Constraining Star Formation Histories](http://arxiv.org/abs/2112.14072)


  The prevalent paradigm of machine learning today is to use past observations
to predict future ones. What if, however, we are interested in knowing the past
given the present? This situation is indeed one that astronomers must contend
with often. To understand the formation of our universe, we must derive the
time evolution of the visible mass content of galaxies. However, to observe a
complete star life, one would need to wait for one billion years! To overcome
this difficulty, astrophysicists leverage supercomputers and evolve simulated
models of galaxies till the current age of the universe, thus establishing a
mapping between observed radiation and star formation histories (SFHs). Such
ground-truth SFHs are lacking for actual galaxy observations, where they are
usually inferred -- with often poor confidence -- from spectral energy
distributions (SEDs) using Bayesian fitting methods. In this investigation, we
discuss the ability of unsupervised domain adaptation to derive accurate SFHs
for galaxies with simulated data as a necessary first step in developing a
technique that can ultimately be applied to observational data.

    

### [[2112.14243] An AGM Approach to Revising Preferences](http://arxiv.org/abs/2112.14243)


  We look at preference change arising out of an interaction between two
elements: the first is an initial preference ranking encoding a pre-existing
attitude; the second element is new preference information signaling input from
an authoritative source, which may come into conflict with the initial
preference. The aim is to adjust the initial preference and bring it in line
with the new preference, without having to give up more information than
necessary. We model this process using the formal machinery of belief change,
along the lines of the well-known AGM approach. We propose a set of fundamental
rationality postulates, and derive the main results of the paper: a set of
representation theorems showing that preference change according to these
postulates can be rationalized as a choice function guided by a ranking on the
comparisons in the initial preference order. We conclude by presenting
operators satisfying our proposed postulates. Our approach thus allows us to
situate preference revision within the larger family of belief change
operators.

    

### [[2112.14338] Socially-Optimal Mechanism Design for Incentivized Online Learning](http://arxiv.org/abs/2112.14338)


  Multi-arm bandit (MAB) is a classic online learning framework that studies
the sequential decision-making in an uncertain environment. The MAB framework,
however, overlooks the scenario where the decision-maker cannot take actions
(e.g., pulling arms) directly. It is a practically important scenario in many
applications such as spectrum sharing, crowdsensing, and edge computing. In
these applications, the decision-maker would incentivize other selfish agents
to carry out desired actions (i.e., pulling arms on the decision-maker's
behalf). This paper establishes the incentivized online learning (IOL)
framework for this scenario. The key challenge to design the IOL framework lies
in the tight coupling of the unknown environment learning and asymmetric
information revelation. To address this, we construct a special Lagrangian
function based on which we propose a socially-optimal mechanism for the IOL
framework. Our mechanism satisfies various desirable properties such as agent
fairness, incentive compatibility, and voluntary participation. It achieves the
same asymptotic performance as the state-of-art benchmark that requires extra
information. Our analysis also unveils the power of crowd in the IOL framework:
a larger agent crowd enables our mechanism to approach more closely the
theoretical upper bound of social performance. Numerical results demonstrate
the advantages of our mechanism in large-scale edge computing.

    

### [[2112.14343] Fake or Genuine? Contextualised Text Representation for Fake Review Detection](http://arxiv.org/abs/2112.14343)


  Online reviews have a significant influence on customers' purchasing
decisions for any products or services. However, fake reviews can mislead both
consumers and companies. Several models have been developed to detect fake
reviews using machine learning approaches. Many of these models have some
limitations resulting in low accuracy in distinguishing between fake and
genuine reviews. These models focused only on linguistic features to detect
fake reviews and failed to capture the semantic meaning of the reviews. To deal
with this, this paper proposes a new ensemble model that employs transformer
architecture to discover the hidden patterns in a sequence of fake reviews and
detect them precisely. The proposed approach combines three transformer models
to improve the robustness of fake and genuine behaviour profiling and modelling
to detect fake reviews. The experimental results using semi-real benchmark
datasets showed the superiority of the proposed model over state-of-the-art
models.

    

### [[2112.14382] Self-Supervised Robustifying Guidance for Monocular 3D Face Reconstruction](http://arxiv.org/abs/2112.14382)


  Despite the recent developments in 3D Face Reconstruction from occluded and
noisy face images, the performance is still unsatisfactory. One of the main
challenges is to handle moderate to heavy occlusions in the face images. In
addition, the noise in the face images inhibits the correct capture of facial
attributes, thus needing to be reliably addressed. Moreover, most existing
methods rely on additional dependencies, posing numerous constraints over the
training procedure. Therefore, we propose a Self-Supervised RObustifying
GUidancE (ROGUE) framework to obtain robustness against occlusions and noise in
the face images. The proposed network contains 1) the Guidance Pipeline to
obtain the 3D face coefficients for the clean faces, and 2) the Robustification
Pipeline to acquire the consistency between the estimated coefficients for
occluded or noisy images and the clean counterpart. The proposed image- and
feature-level loss functions aid the ROGUE learning process without posing
additional dependencies. On the three variations of the test dataset of CelebA:
rational occlusions, delusional occlusions, and noisy face images, our method
outperforms the current state-of-the-art method by large margins (e.g., for the
shape-based 3D vertex errors, a reduction from 0.146 to 0.048 for rational
occlusions, from 0.292 to 0.061 for delusional occlusions and from 0.269 to
0.053 for the noise in the face images), demonstrating the effectiveness of the
proposed approach.

    

### [[2112.14428] Efficient Belief Space Planning in High-Dimensional State Spaces using PIVOT: Predictive Incremental Variable Ordering Tactic](http://arxiv.org/abs/2112.14428)


  In this work, we examine the problem of online decision making under
uncertainty, which we formulate as planning in the belief space. Maintaining
beliefs (i.e., distributions) over high-dimensional states (e.g., entire
trajectories) was not only shown to significantly improve accuracy, but also
allows planning with information-theoretic objectives, as required for the
tasks of active SLAM and information gathering. Nonetheless, planning under
this "smoothing" paradigm holds a high computational complexity, which makes it
challenging for online solution. Thus, we suggest the following idea: before
planning, perform a standalone state variable reordering procedure on the
initial belief, and "push forwards" all the predicted loop closing variables.
Since the initial variable order determines which subset of them would be
affected by incoming updates, such reordering allows us to minimize the total
number of affected variables, and reduce the computational complexity of
candidate evaluation during planning. We call this approach PIVOT: Predictive
Incremental Variable Ordering Tactic. Applying this tactic can also improve the
state inference efficiency; if we maintain the PIVOT order after the planning
session, then we should similarly reduce the cost of loop closures, when they
actually occur. To demonstrate its effectiveness, we applied PIVOT in a
realistic active SLAM simulation, where we managed to significantly reduce the
computation time of both the planning and inference sessions. The approach is
applicable to general distributions, and induces no loss in accuracy.

    

### [[2112.14460] Baihe: SysML Framework for AI-driven Databases](http://arxiv.org/abs/2112.14460)


  We present Baihe, a SysML Framework for AI-driven Databases. Using Baihe, an
existing relational database system may be retrofitted to use learned
components for query optimization or other common tasks, such as e.g. learned
structure for indexing. To ensure the practicality and real world applicability
of Baihe, its high level architecture is based on the following requirements:
separation from the core system, minimal third party dependencies, Robustness,
stability and fault tolerance, as well as stability and configurability. Based
on the high level architecture, we then describe a concrete implementation of
Baihe for PostgreSQL and present example use cases for learned query
optimizers. To serve both practitioners, as well as researchers in the DB and
AI4DB community Baihe for PostgreSQL will be released under open source
license.

    

### [[2112.14476] ADAPQUEST: A Software for Web-Based Adaptive Questionnaires based on Bayesian Networks](http://arxiv.org/abs/2112.14476)


  We introduce ADAPQUEST, a software tool written in Java for the development
of adaptive questionnaires based on Bayesian networks. Adaptiveness is intended
here as the dynamical choice of the question sequence on the basis of an
evolving model of the skill level of the test taker. Bayesian networks offer a
flexible and highly interpretable framework to describe such testing process,
especially when coping with multiple skills. ADAPQUEST embeds dedicated
elicitation strategies to simplify the elicitation of the questionnaire
parameters. An application of this tool for the diagnosis of mental disorders
is also discussed together with some implementation details.

    

### [[2112.14480] On some Foundational Aspects of Human-Centered Artificial Intelligence](http://arxiv.org/abs/2112.14480)


  The burgeoning of AI has prompted recommendations that AI techniques should
be "human-centered". However, there is no clear definition of what is meant by
Human Centered Artificial Intelligence, or for short, HCAI. This paper aims to
improve this situation by addressing some foundational aspects of HCAI. To do
so, we introduce the term HCAI agent to refer to any physical or software
computational agent equipped with AI components and that interacts and/or
collaborates with humans. This article identifies five main conceptual
components that participate in an HCAI agent: Observations, Requirements,
Actions, Explanations and Models. We see the notion of HCAI agent, together
with its components and functions, as a way to bridge the technical and
non-technical discussions on human-centered AI. In this paper, we focus our
analysis on scenarios consisting of a single agent operating in dynamic
environments in presence of humans.

    

### [[2112.14540] Res2NetFuse: A Fusion Method for Infrared and Visible Images](http://arxiv.org/abs/2112.14540)


  This paper presents a novel Res2Net-based fusion framework for infrared and
visible images. The proposed fusion model has three parts: an encoder, a fusion
layer and a decoder, respectively. The Res2Net-based encoder is used to extract
multi-scale features of source images, the paper introducing a new training
strategy for training a Res2Net-based encoder that uses only a single image.
Then, a new fusion strategy is developed based on the attention model. Finally,
the fused image is reconstructed by the decoder. The proposed approach is also
analyzed in detail. Experiments show that our method achieves state-of-the-art
fusion performance in objective and subjective assessment by comparing with the
existing methods.

    

### [[2112.14603] Learning Higher-Order Programs without Meta-Interpretive Learning](http://arxiv.org/abs/2112.14603)


  Learning complex programs through inductive logic programming (ILP) remains a
formidable challenge. Existing higher-order enabled ILP systems show improved
accuracy and learning performance, though remain hampered by the limitations of
the underlying learning mechanism. Experimental results show that our extension
of the versatile Learning From Failures paradigm by higher-order definitions
significantly improves learning performance without the burdensome human
guidance required by existing systems. Furthermore, we provide a theoretical
framework capturing the class of higher-order definitions handled by our
extension.

    

### [[2112.14608] HPRN: Holistic Prior-embedded Relation Network for Spectral Super-Resolution](http://arxiv.org/abs/2112.14608)


  Spectral super-resolution (SSR) refers to the hyperspectral image (HSI)
recovery from an RGB counterpart. Due to the one-to-many nature of the SSR
problem, a single RGB image can be reprojected to many HSIs. The key to tackle
this illposed problem is to plug into multi-source prior information such as
the natural RGB spatial context-prior, deep feature-prior or inherent HSI
statistical-prior, etc., so as to improve the confidence and fidelity of
reconstructed spectra. However, most current approaches only consider the
general and limited priors in their designing the customized convolutional
neural networks (CNNs), which leads to the inability to effectively alleviate
the degree of ill-posedness. To address the problematic issues, we propose a
novel holistic prior-embedded relation network (HPRN) for SSR. Basically, the
core framework is delicately assembled by several multi-residual relation
blocks (MRBs) that fully facilitate the transmission and utilization of the
low-frequency content prior of RGB signals. Innovatively, the semantic prior of
RGB input is introduced to identify category attributes and a semantic-driven
spatial relation module (SSRM) is put forward to perform the feature
aggregation among the clustered similar characteristics using a
semantic-embedded relation matrix. Additionally, we develop a transformer-based
channel relation module (TCRM), which breaks the habit of employing scalars as
the descriptors of channel-wise relations in the previous deep feature-prior
and replaces them with certain vectors, together with Transformerstyle feature
interactions, supporting the representations to be more discriminative. In
order to maintain the mathematical correlation and spectral consistency between
hyperspectral bands, the second-order prior constraints (SOPC) are incorporated
into the loss function to guide the HSI reconstruction process.

    

### [[2112.14624] Towards a Shapley Value Graph Framework for Medical peer-influence](http://arxiv.org/abs/2112.14624)


  eXplainable Artificial Intelligence (XAI) is a sub-field of Artificial
Intelligence (AI) that is at the forefront of AI research. In XAI feature
attribution methods produce explanations in the form of feature importance. A
limitation of existing feature attribution methods is that there is a lack of
explanation towards the consequence of intervention. Although contribution
towards a certain prediction is highlighted, the influence between features and
the consequence of intervention is not addressed. The aim of this paper is to
introduce a new framework to look deeper into explanations using graph
representation for feature-to-feature interactions to improve the
interpretability of black-box Machine Learning (ML) models and inform
intervention.

    

### [[2112.14657] Dynamic programming with partial information to overcome navigational uncertainty in a nautical environment](http://arxiv.org/abs/2112.14657)


  Using a toy nautical navigation environment, we show that dynamic programming
can be used when only partial information about a partially observed Markov
decision process (POMDP) is known. By incorporating uncertainty into our model,
we show that navigation policies can be constructed that maintain safety.
Adding controlled sensing methods, we show that these policies can also lower
measurement costs at the same time.

    

### [[2112.14676] Learning nonlinear dynamics in synchronization of knowledge-based leader-following networks](http://arxiv.org/abs/2112.14676)


  Knowledge-based leader-following synchronization problem of heterogeneous
nonlinear multi-agent systems is challenging since the leader's dynamic
information is unknown to all follower nodes. This paper proposes a
learning-based fully distributed observer for a class of nonlinear leader
systems, which can simultaneously learn the leader's dynamics and states. The
class of leader dynamics considered here does not require a bounded Jacobian
matrix. Based on this learning-based distributed observer, we further
synthesize an adaptive distributed control law for solving the leader-following
synchronization problem of multiple Euler-Lagrange systems subject to an
uncertain nonlinear leader system. The results are illustrated by a simulation
example.

    

### [[2112.14699] Automated Urban Planning for Reimagining City Configuration via Adversarial Learning: Quantification, Generation, and Evaluation](http://arxiv.org/abs/2112.14699)


  Urban planning refers to the efforts of designing land-use configurations
given a region. However, to obtain effective urban plans, urban experts have to
spend much time and effort analyzing sophisticated planning constraints based
on domain knowledge and personal experiences. To alleviate the heavy burden of
them and produce consistent urban plans, we want to ask that can AI accelerate
the urban planning process, so that human planners only adjust generated
configurations for specific needs? The recent advance of deep generative models
provides a possible answer, which inspires us to automate urban planning from
an adversarial learning perspective. However, three major challenges arise: 1)
how to define a quantitative land-use configuration? 2) how to automate
configuration planning? 3) how to evaluate the quality of a generated
configuration? In this paper, we systematically address the three challenges.
Specifically, 1) We define a land-use configuration as a
longitude-latitude-channel tensor. 2) We formulate the automated urban planning
problem into a task of deep generative learning. The objective is to generate a
configuration tensor given the surrounding contexts of a target region. 3) We
provide quantitative evaluation metrics and conduct extensive experiments to
demonstrate the effectiveness of our framework.

    

### [[2112.14706] Intersection focused Situation Coverage-based Verification and Validation Framework for Autonomous Vehicles Implemented in CARLA](http://arxiv.org/abs/2112.14706)


  Autonomous Vehicles (AVs) i.e., self-driving cars, operate in a safety
critical domain, since errors in the autonomous driving software can lead to
huge losses. Statistically, road intersections which are a part of the AVs
operational design domain (ODD), have some of the highest accident rates.
Hence, testing AVs to the limits on road intersections and assuring their
safety on road intersections is pertinent, and thus the focus of this paper. We
present a situation coverage-based (SitCov) AV-testing framework for the
verification and validation (V&V) and safety assurance of AVs, developed in an
open-source AV simulator named CARLA. The SitCov AV-testing framework focuses
on vehicle-to-vehicle interaction on a road intersection under different
environmental and intersection configuration situations, using situation
coverage criteria for automatic test suite generation for safety assurance of
AVs. We have developed an ontology for intersection situations, and used it to
generate a situation hyperspace i.e., the space of all possible situations
arising from that ontology. For the evaluation of our SitCov AV-testing
framework, we have seeded multiple faults in our ego AV, and compared situation
coverage based and random situation generation. We have found that both
generation methodologies trigger around the same number of seeded faults, but
the situation coverage-based generation tells us a lot more about the
weaknesses of the autonomous driving algorithm of our ego AV, especially in
edge-cases. Our code is publicly available online, anyone can use our SitCov
AV-testing framework and use it or build further on top of it. This paper aims
to contribute to the domain of V&V and development of AVs, not only from a
theoretical point of view, but also from the viewpoint of an open-source
software contribution and releasing a flexible/effective tool for V&V and
development of AVs.

    

### [[1610.09077] Integrating Topic Models and Latent Factors for Recommendation](http://arxiv.org/abs/1610.09077)


  Nowadays, we have large amounts of online items in various web-based
applications, which makes it an important task to build effective personalized
recommender systems so as to save users' efforts in information seeking. One of
the most extensively and successfully used methods for personalized
recommendation is the Collaborative Filtering (CF) technique, which makes
recommendation based on users' historical choices as well as those of the
others'. The most popular CF method, like Latent Factor Model (LFM), is to
model how users evaluate items by understanding the hidden dimension or factors
of their opinions. How to model these hidden factors is key to improve the
performance of recommender system. In this work, we consider the problem of
hotel recommendation for travel planning services by integrating the location
information and the user's preference for recommendation. The intuition is that
user preferences may change dynamically over different locations, thus treating
the historical decisions of a user as static or universally applicable can be
infeasible in real-world applications. For example, users may prefer chain
brand hotels with standard configurations when traveling for business, while
they may prefer unique local hotels when traveling for entertainment. In this
paper, we aim to provide trip-level personalization for users in
recommendation.

    

### [[2101.12153] A Survey on Personality-Aware Recommendation Systems](http://arxiv.org/abs/2101.12153)


  With the emergence of personality computing as a new research field related
to artificial intelligence and personality psychology, we have witnessed an
unprecedented proliferation of personality-aware recommendation systems. Unlike
conventional recommendation systems, these new systems solve traditional
problems such as the cold start and data sparsity problems. This survey aims to
study and systematically classify personality-aware recommendation systems. To
the best of our knowledge, this survey is the first that focuses on
personality-aware recommendation systems. We explore the different design
choices of personality-aware recommendation systems, by comparing their
personality modeling methods, as well as their recommendation techniques.
Furthermore, we present the commonly used datasets and point out some of the
challenges of personality-aware recommendation systems.

    

### [[2104.14512] A General Katsuno-Mendelzon-Style Characterization of AGM Belief Base Revision for Arbitrary Monotonic Logics](http://arxiv.org/abs/2104.14512)


  The AGM postulates by Alchourrón, Gärdenfors, and Makinson continue
to represent a cornerstone in research related to belief change. We generalize
the approach of Katsuno and Mendelzon (KM) for characterizing AGM base revision
from propositional logic to the setting of (multiple) base revision in
arbitrary monotonic logics. Our core result is a representation theorem using
the assignment of total - yet not transitive - "preference" relations to belief
bases. We also provide a characterization of all logics for which our result
can be strengthened to preorder assignments (as in KM's original work).

    

### [[2112.14048] Monads for Measurable Queries in Probabilistic Databases](http://arxiv.org/abs/2112.14048)


  We consider a bag (multiset) monad on the category of standard Borel spaces,
and show that it gives a free measurable commutative monoid. Firstly, we show
that a recent measurability result for probabilistic database queries (Grohe
and Lindner, ICDT 2020) follows quickly from the fact that queries can be
expressed in monad-based terms. We also extend this measurability result to a
fuller query language. Secondly, we discuss a distributive law between
probability and bag monads, and we illustrate that this is useful for
generating probabilistic databases.

    

### [[2112.14053] From Semantics to Types: the Case of the Imperative lambda-Calculus](http://arxiv.org/abs/2112.14053)


  We propose an intersection type system for an imperative lambda-calculus
based on a state monad and equipped with algebraic operations to read and write
to the store. The system is derived by solving a suitable domain equation in
the category of omega-algebraic lattices; the solution consists of a
filter-model generalizing the well-known construction for ordinary
lambda-calculus. Then the type system is obtained out of the term
interpretations into the filter-model itself. The so obtained type system
satisfies the "type-semantics" property, and it is sound and complete by
construction.

    

### [[2112.14305] Proceedings Second Joint International Workshop on Linearity & Trends in Linear Logic and Applications](http://arxiv.org/abs/2112.14305)


  This volume contains a selection of papers presented at Linearity&TLLA 2020,
namely the Second Joint International Workshop on Linearity & Trends in Linear
Logic and Applications, held on June 29-30, 2020 online. (The workshop was
supposed to take place in Paris as part of FSCD 2020, but due to the COVID
pandemic it was decided not to hold the event live.) Linearity is a central
concept in many theoretical and practical approaches to computer science. On
the theoretical side, there is much work stemming from linear logic and dealing
with resource control, complexity classes, and more recently quantum
computation. On the practical side there is certainly work on program analysis,
operational semantics, logic programming languages, program transformations,
and efficient implementation techniques. Linear logic is not only a theoretical
tool for the analysis of resource usage in logic and computation. It is also a
corpus of distinct approaches and methodologies (e.g., proof nets, geometry of
interaction, coherent spaces, relational models) that were originally developed
for the study of linear logic's syntax and semantics have nowadays found
applications in several other fields.

    

### [[2112.14714] Automated Code Optimization with E-Graphs](http://arxiv.org/abs/2112.14714)


  This thesis proposes an advanced, generic and high-level code rewriting and
analysis system in the Julia programming language, providing applied equality
saturation in the presence of multiple dispatch and metaprogramming. We show
how our system can practically solve some challenging problems: Can programmers
implement their own high-level compiler optimizations for their domain-specific
scientific programs, without the requirement of them being compiler experts at
all? Can these optimizers be implemented by users in the same language and
inside the same programs they want to optimize, solving the two-language
problem? Can these compiler optimizers be written in a high-level fashion, as
equations, without the need to worry about the rewriting ordering? Thus, can
symbolic mathematics do high-level compiler optimizations or vice-versa?

    

### [[2112.14716] Exploring Aspects of Polyglot High-Performance Virtual Machine GraalVM](http://arxiv.org/abs/2112.14716)


  Contemporary software often becomes vastly complex, and we are required to
use a variety of technologies and different programming languages for its
development. As interoperability between programming languages could cause high
overhead resulting in a performance loss, it is important to examine how a
current polyglot virtual machine with a compiler written in a high-level
object-oriented language deals with it. OpenJDK's Project Metropolis presented
the GraalVM, an open-source, high-performance polyglot virtual machine, mostly
written in Java. This paper presents GraalVM's architecture and its features;
furthermore, examining how it resolves common interoperability and performance
problems. GraalVM makes software ecosystem productive when combining various
programming languages, for example, Java, JavaScript, C/C++, Python, Ruby, R,
and others. The vital part of GraalVM is the Graal compiler written in Java,
which allows developers to maintain and optimize code faster, simpler, and more
efficient, in comparison to traditional compilers in C/C++ languages. Graal can
be used as a just-in-time (JIT) or as static, ahead-of-time (AOT) compiler.
Graal is an aggressively optimizing compiler implementing common compiler
optimizations, with emphasis on outstanding inlining and escape analysis
algorithms. This paper compares Graal with some of the best-specialized
competitors, and presents our results tested within an academic environment.

    

### [[2105.09929] Join Inverse Rig Categories for Reversible Functional Programming, and Beyond](http://arxiv.org/abs/2105.09929)


  Reversible computing is a computational paradigm in which computations are
deterministic in both the forward and backward direction, so that programs have
well-defined forward and backward semantics. We investigate the formal
semantics of the reversible functional programming language Rfun. For this
purpose, we introduce join inverse rig categories, the natural marriage of join
inverse categories and rig categories, which we show can be used to model the
language Rfun, under reasonable assumptions. These categories turn out to be a
particularly natural fit for reversible computing as a whole, as they encompass
models for other reversible programming languages, notably Theseus and
reversible flowcharts. This suggests that join inverse rig categories really
are the categorical models of reversible computing.

    