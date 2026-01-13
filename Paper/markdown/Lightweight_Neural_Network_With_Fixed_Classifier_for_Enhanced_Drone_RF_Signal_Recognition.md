
This article has been accepted for publication in IEEE Transactions on Cognitive Communications and Networking. This is the author's version which has not been fully edited and content may change prior to final publication. Citation information: DOI 10.1109/TCCN.2025.3598245

**IEEE TRANSACTIONS ON COGNITIVE COMMUNICATIONS AND NETWORKING, VOL. XX, NO. XX, AUGUST XXXX**
**1**

# Lightweight Neural Network with Fixed Classifier for Enhanced Drone RF Signal Recognition

Feifei Zhu, Fuhui Zhou, *Senior Member, IEEE*, Rui Ding, *Student Member, IEEE*,
Qihui Wu, *Fellow, IEEE*, Kai-Kit Wong, *Fellow, IEEE*, and Chan-Byoung Chae, *Fellow, IEEE*

***Abstract*—Although small drones provide significant convenience through various applications, their widespread use poses a major threat to public safety, particularly when operated without authorization. To address this challenge, a deep learning (DL) recognition system based on drone radio frequency (RF) signals has emerged as a potentially effective method for efficiently identifying drone classes. However, traditional DL models require large computing resources, making them unsuitable for deployment on mobile devices. In this paper, a novel network model consisting of a fixed classifier architecture inspired by neural collapse (NC) is proposed. Moreover, a soft thresholding and broadcasting mechanism is integrated, an enhanced Squeeze-and-Excitation (ESE) module is introduced, and information theory principles are employed to design the loss function for the Equiangular Tight Frame (ETF) classifier. Simulation results show that the average accuracy for recognizing drone classes is one percentage point higher than that of other methods at high signal-to-noise ratio (SNR). Furthermore, experiments show that the proposed drone recognition method achieves an average accuracy of 96.64% and demonstrates strong generalization capability. Additionally, the advantages of the fixed classifier are validated by analyzing the sample distribution in the 3D semantic space. It also exhibits lower model complexity and fewer parameters compared to traditional approaches. Finally, ablation experiments confirm that our proposed model makes a trade-off between the computational complexity and the average accuracy.**

***Index Terms*—Drone recognition, lightweight network, neural collapse, feature extraction, RF signal.**

## I. INTRODUCTION

UNMANNED aerial vehicles (UAVs), also known as drones, are currently used in various industries because drones have low cost, high reliability, and easy portability characteristics for consumers,. However, the proliferation of drone technology raises significant public safety concerns, particularly regarding unauthorized commercial drone operations that lead to unidentified aerial intrusions. This has become a growing challenge in sensitive areas. For example, the problem of unidentified drone intrusions occurred at an airfield in Tianjin. It caused significant flight delays, threatened civil aviation safety, and caused serious economic losses. Therefore, it is necessary to improve the capability of anti-drone technology to deal with the threat of unidentified drone intrusions. In this case, there is an urgent need to identify unauthorized drones accurately.

Due to the limited adaptability of conventional optimization frameworks to dynamic doppler features and the poor generalization of manually designed features, new data-driven approaches are needed for drone RF signal detection and recognition. With the rapid development of machine learning (ML), especially deep learning (DL), which makes it possible to detect and recognize drones efficiently, the main methods include radar, acoustic, visual, and radio frequency (RF),. First, the authors in proposed a millimeter wave frequency modulated continuous wave (FMCW) radar antenna and employed a convolutional neural network (CNN) for drone classification. Kaya et al. used a commercial S-band horizontally polarized pulse radar to capture target echoes and identified drones using a CNN model. Additionally, a low-cost microphone for detecting drones by extracting the characteristics of drone sounds using the SVM method was introduced in. Drone sound waves exhibit distinct characteristics in the time and frequency domains. A method combining joint time-frequency analysis and local feature extraction using a CNN framework was proposed in. However, a visual to detect drones is also a relatively low-cost method. The visual captures drone flight images through target detection methods, such as the wide range of You Only Look Once (YOLO) models for drone recognition,.

In contrast to the above methods for drone detection and recognition, RF detection does not require the emission of any signals. Moreover, RF fingerprinting imposes no additional power burden on the transmitting device because it exploits inherent hardware impairments that are naturally present in the RF components,. The unique characteristics and moderate detection ranges of drone RF signals allow fine-grained classification, making them effective against unidentified drone intrusions. In, the authors proposed an RF front-end system that scans and intercepts signals in the target airspace using field-programmable gate arrays (FPGAs) to detect drone transmissions. However, the system’s performance is limited by the sampling rate of the FPGA.

In recent years, unsupervised learning has gained attention in drone RF signal recognition due to its ability to work with

---

This work was supported by the Jiangsu Provincial Key Research and Development Program under Grants BE2022068.
Feifei Zhu, Rui Ding, and Qihui Wu are with the College of Electronic and Information Engineering, Nanjing University of Aeronautics and Astronautics, Nanjing 211106, China (e-mail: feifeizhu@nuaa.edu.cn; zhoufuhui@ieee.org; rui_ding@nuaa.edu.cn; wuqihui2014@sina.com).
Fuhui Zhou is with the College of Artificial Intelligence, Nanjing University of Aeronautics and Astronautics, Nanjing 211106, China (e-mail: zhoufuhui@ieee.org).
Kai-Kit Wong is with the Department of Electronic and Electrical Engineering, University College London, WC1E 6BT London, U.K. (e-mail: kai-kit wong@ucl.ac.uk). He is also affiliated with the Yonsei Frontier Lab., Yonsei University, Seoul 03722, South Korea.
Chan-Byoung Chae is with the School of Integrated Technology, Yonsei University, Seoul 03722, South Korea (e-mail: cbchae@yonsei.ac.kr).

Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:57 UTC from IEEE Xplore. Restrictions apply.
© 2025 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

---

This article has been accepted for publication in IEEE Transactions on Cognitive Communications and Networking. This is the author's version which has not been fully edited and content may change prior to final publication. Citation information: DOI 10.1109/TCCN.2025.3598245

**IEEE TRANSACTIONS ON COGNITIVE COMMUNICATIONS AND NETWORKING, VOL. XX, NO. XX, AUGUST XXXX**
**2**

limited training samples. To reduce the spatial complexity of the algorithm, an auxiliary classifier for the Wasserstein generative adversarial networks (GANs) was proposed in. In addition, many consumer drones transmit video signals by accessing Wi-Fi signals. A Gaussian mixture model (GMM) of K-means is introduced to quickly identify drones by the characteristics of the size and interval time of drone packets in encrypted Wi-Fi traffic. The traditional K-means algorithm exhibits high sensitivity to noise and outliers and lacks robustness in complex data environments. Therefore, the authors designed a high-precision drone classification method using K-means clustering by combining spectral dimension reduction with principal component analysis (PCA) to reduce model noise and outliers. Moreover, some researchers have explored approaches for cross-domain drone detection. The authors in proposed an anchor-free detector to identify the center position of the drone spectrogram for keypoint matching classification, employing a GAN model to address cross-domain detection challenges across different drone environments. Nevertheless, the application of unsupervised learning to drone recognition remains a problem because of feature sensitivity and high computational cost.

On the other hand, traditional supervised learning methods can learn the features of drone RF signals more efficiently with better generalization. To suppress noise in drone RF signals, the authors in proposed a two-level score-based aggregation method that makes final decisions using predicted probability vectors from a combined neural network. Moreover, to address the high dimensionality of training samples, the method combining a conventional CNN model for multi-class drone classification and a support vector machine (SVM) classifier for fine-grained recognition of specific models based on impulse structure features was investigated in. The authors in classified drone RF signals into clean, interference, and noisy types, and designed three specialized loss functions for training a U-Net model. In, a random forest algorithm was used under Wi-Fi interference, where feature vector compactness was enhanced to reduce the dimensionality of the original RF features for drone classification. To reduce CNN model complexity, a method based on depthwise and pointwise convolutions for drone recognition that achieved high accuracy and efficiency was presented in. To reduce the need for training data, a framework combining a tri-residual semantic network as the backbone with an SVM classifier for drone identification was introduced in. Furthermore, the authors in investigated a transformer-based network incorporating extreme value theory (EVT) to address out-of-distribution samples and misclassification. In, structural features of signal semantics were used to reconstruct the drone spectrogram, and a CNN method was employed to classify drone types based on the reconstructed spectrograms.

To fully utilize the time domain signal information of the drone, the authors used intrinsic mode functions (IMFs) as features for drone RF signal recognition, achieving improved performance with classical ML algorithms. The authors in extracted pilot frequency slice features from drone RF signals, trained CNN models locally on individual devices, and applied federated learning to aggregate model updates at a central controller, thereby improving classification accuracy. Furthermore, carrier frequency offset compensation addresses the issue of different operating channels in drone RF signals, and fine-grained classification (FGC) of drones has been achieved using CNN models,. In addition, researchers discovered that the cyclic prefix (CP) of drone RF signals correlates with video signals and proposed a drone identification method using a CNN model with the normalized CP correlation spectrum.

Currently, neural network architectural designs are undergoing rapid transformation. To overcome the vanishing gradient problem in CNN models, researchers have focused on incorporating residual connections in neural network architectures. This design has been widely adopted in ResNet, DenseNet, and ResNeXt. The authors in proposed the squeeze-and-excitation (SE) block, which adaptively recalibrates channel-wise feature responses to enhance feature extraction. In, the authors proposed ConvNeXt, a modernized convolutional network that integrates transformer-inspired architectural elements to improve image recognition performance.

However, the above neural networks still have problems such as large numbers of parameters, slow inference speed, and difficulty running on mobile devices. To solve these problems, the researchers proposed a lightweight network architecture, which greatly reduces the number of parameters of the model and floating point operations (FLOPs). Ma et al. proposed the ShuffleNetV2 network architecture with channel rearrangement operations moved to branch merging, which reduces the amount of computation. To improve the feature extraction effect of network channels, the authors in proposed the MobileNetV3-Small network architecture, which employs neural architecture search (NAS) networks and introduced SE modules to improve the representation of characteristics between channels to achieve higher precision. Moreover, the authors in proposed the EfficientNetV2 model, which introduces a standard 3×3 convolution in the fused mobile inverted bottleneck convolution module and employs the adaptive regularization method to improve the inference efficiency of the model.

More importantly, neural networks have been considered black box models, and it is necessary to explain their principles. The authors observed that during the terminal phase of training (TPT), when neural collapse (NC) occurs, the classifier weights and learned class means form an Equiangular Tight Frame (ETF), and classification is performed by assigning the sample to the closest class mean. Inspired by, researchers have recently proposed several classification methods based on the ETF architecture, marking a growing trend in current research,,.

Hence, considering the limited computing resources of mobile devices in practical drone recognition systems, it is essential to design lightweight and efficient models. To this end, we propose a MobileNetV3-Small architecture integrated with an ETF-based classifier, termed MobileNetV3-Small-ETF. Our main contributions are summarized as follows.
(1) We introduce an enhanced Squeeze-and-Excitation

---

Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:57 UTC from IEEE Xplore. Restrictions apply.
© 2025 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

---

This article has been accepted for publication in IEEE Transactions on Cognitive Communications and Networking. This is the author's version which has not been fully edited and content may change prior to final publication. Citation information: DOI 10.1109/TCCN.2025.3598245

**IEEE TRANSACTIONS ON COGNITIVE COMMUNICATIONS AND NETWORKING, VOL. XX, NO. XX, AUGUST XXXX**
**3**

*Fig. 1. The proposed lightweight network of the drone recognition system model.*

(ESE) framework incorporating soft thresholding to mitigate noise interference in channel feature representation. Therefore, enhancing the channel feature extraction capability improves the recognition effect of the model.
(2) The proposed NC-based classifier design replaces the traditional classifier in MobileNetV3-Small by using the last hidden layer as the feature layer and setting the number of output classes according to the number of drone types to be identified. In addition, the system employs the classifier with a fixed ETF architecture. Consequently, the model can improve the accuracy of drone recognition while reducing computational complexity, including FLOPs and model parameters.
(3) We propose an additional loss function based on information theory to increase the recognition accuracy of the MobileNetV3-Small-ETF model. The assumed feature vector layer is the codeword and the classifier weights are the codebook. In the ETF framework, the loss function minimizes the distance between the learned feature codewords and the target codebook. Finally, we use samples from multiple drone classes in the public dataset to evaluate the effectiveness of the proposed scheme.
The rest of the paper is organized as follows. The system model for receiving drone RF signals and preprocessing the drone signal samples required by the drone identification system is described in Section II. The proposed MobileNetV3-Small-ETF model for drone class recognition is presented in Section III. In Section IV, the experimental results are analyzed and discussed. Finally, we conclude the paper in Section V.

## II. PROBLEM STATEMENT

In this section, we first introduce the system model for the time-domain extraction of the received drone signal based on the window function in the noise interference. Then, to reduce the high dimension of the original drone in-phase and quadrature (IQ) data, we selected the preprocessing method of using the drone spectrogram as the model’s training samples.

### A. Received Signal Model

Since the Industrial, Scientific, and Medical (ISM) frequency band is a shared frequency band, it is prone to interference from various signal sources. Thus, the video transmission signal of the general commercial drone is the direct sequence spread spectrum (DSSS) transmission mode and the control signal using frequency hopping spread spectrum (FHSS) technology. The time domain expression of the received drone RF signal, is given as

$$
r(t) = \sum_{k=1}^K \text{rect}\left(\frac{t-t_s-T_s/2}{T_s}\right) s_k(t) * h_k(t) + n(t), \quad t \in (t_s, t_s+T_s), \quad (1)
$$

where rect(·), $||t||_1 \le T_s/2$ is a rectangular function, $T_s$ is the duration time of the OFDM symbol, $t_s$ is the sampling interval time, $s_k(t)$ represents the complex transmitted signal in the k-th path of the drone, K is the total number of drone transmit antennas, $h_k(t)$ denotes the impulse response of the k-th Rayleigh fading channel, * is the convolution operation, n(t) represents the inherent noise of the device, which roughly follows additive white Gaussian noise (AWGN). This dataset can be accessed through.

---

Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:57 UTC from IEEE Xplore. Restrictions apply.
© 2025 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

---

This article has been accepted for publication in IEEE Transactions on Cognitive Communications and Networking. This is the author's version which has not been fully edited and content may change prior to final publication. Citation information: DOI 10.1109/TCCN.2025.3598245

**IEEE TRANSACTIONS ON COGNITIVE COMMUNICATIONS AND NETWORKING, VOL. XX, NO. XX, AUGUST XXXX**
**4**

### B. Drone RF Singal Preprocessing

The drone RF signal collection device uses a 100MHz sampling rate in the public dataset. It can generate a large number of IQ points even in a short time for individual samples. However, the drone RF signal is susceptible to interference from noise in the ISM band and the dimension of the raw IQ data is high, which reduces the effectiveness of training the network. On the other hand, the channel between the receiver and the drone is a time-varying impulse response. Thus, we use the short-time Fourier transform (STFT) method to train the network to recognize the drone more efficiently. This transform domain approach decouples the time domain of the original IQ data into multiple small blocks for fast Fourier transform (FFT), which results in a two-dimensional spectrogram that combines the time and frequency domain, reduces noise interference in the time domain signal, and enhances the characteristics of the network input training samples. The discrete-time STFT formula is given as

$$
\text{STFT}(m,l) = \sum_{n=0}^{N-1} r(n)w(n-mR)\exp\left(-j\frac{2\pi l}{N}n\right), \quad (2)
$$

where $w(n-mR)$ is the hanging window function for effectively controlling the spectrum leakage problem, $l$ is the time index of the STFT analysis window, $R$ is the moving step of the analysis window, exp(·) denotes the natural exponential function, and $N$ is the length of the short-time analysis window. Finally, by extracting the power spectrum of the drone RF signal as a sample for training the model, the power spectrum captured with (2) can be calculated as

$$
P(m,l) = 10 \log_{10}(|\text{STFT}(m,l)|^2), \quad (3)
$$

where P(·) is the power spectrum of each signal segment.

## III. THE PROPOSED LIGHTWEIGHT NETWORK METHOD WITH DRONE RECOGNITION

### A. Overall Drone Recognition Framework

The proposed drone recognition system consists of drone RF signal preprocessing and a lightweight network system, as described in Fig. 1. The preprocessing part of the drone signal data is shown in Section II. The received drone signal $r(t)$ needs to be converted into spectrogram features to train the network. In addition, MobileNetV3-Small is used as the backbone network to extend the model because it can reduce the demand for computing resources and maintain the model's performance. First, cropping, rotation, and normalization methods are used for the input drone spectrogram feature to enhance the robustness and generalization capability of the model. The normalization of each spectrogram $Z \in \mathbb{R}^{H \times W \times C}$ operation formula is given as

$$
\hat{Z}_i = \frac{Z_i - \mu_i}{\sigma_i}, \quad (4)
$$

where $Z_i, i \in \{1, 2, \dots, C\}$ represents the input i-th channel feature dimension, $\mu_i$ and $\sigma_i$ are the mean and variance of the i-th channel. Before the spectrogram features enter the inverted residual block, the model performs preliminary feature extraction, which mainly captures the edge information of the spectrogram. The feature extraction presented is

$$
Z_{\text{iconv}} = \sum_{i=1}^{K} \sum_{j=1}^{K} \sum_{c=1}^{C} Z_c(p+i, q+j, c) K(i, j, c, d), \quad (5)
$$

here, the output feature is $Z_{\text{conv}} \in \mathbb{R}^{P \times Q \times N}$ where $N$ is the number of output channels, where $p$ and $q$ are the spatial locations of the feature, $K \times K$ is the convolution kernel size $3 \times 3$, and input channel is $C$, the expanded dimension of the convolution kernel is $K \in \mathbb{R}^{3 \times 3 \times C \times N}$, the stride (S) length is 2, c, d is the input and output channel index. Then, the extended feature maps are batch normalized (BN) to stabilize the sample distribution. The BN function formula is denoted as

$$
Z^{(d)}_{\text{BN}} = \gamma_d \frac{Z^{(d)}_{\text{conv}} - \mu_d}{\sqrt{\sigma_d^2 + \epsilon}} + \beta_d, \quad (6)
$$

where $Z^{(d)}_{\text{BN}} \in \mathbb{R}^{P \times Q \times 1}, d \in \{1, 2, \dots, N\}$ is the d-th channel output feature, $\mu_d$ and $\sigma_d^2$ are the mean and variance of the d-th channel, the scale factors and offsets that the model learns are $\gamma_d, \beta_d, \epsilon$ is a constant used for numerical stability of models (typically $10^{-5}$). Finally, the h-swish activation function is employed in the bottleneck layers of the latter network stages to enhance nonlinearity and capture complex features, while maintaining a balance between computational cost and performance. The output feature $H \in \mathbb{R}^{P \times Q \times N}$ is generated via the h-swish activation function. It can be expressed as

$$
H^{(d)} = Z^{(d)}_{\text{BN}} \frac{\text{ReLU6}(Z^{(d)}_{\text{BN}} + 3)}{6}, \quad (7)
$$

where $R(x) = \max(x, 0)$ is rectified linear unit (ReLU) activation function, and $\text{ReLU6}(x) = \min(R(x), 6)$.

The MobileNetV3-Small-ETF model selects 11 inverted residual block structures as shown in. On the other hand, each inverted residual block requires distinct configurations of its ESE module size, activation function, and channel attention (CA) convolution kernel. If an ESE module exists, it will be described in the next section. The main steps of the inverted residual block are introduced as follows:

1) *Pointwise convolution:* The convolution operation with a convolution layer of 1×1 and occupied stride (S) size 1 expands the number of input feature channels to enhance the feature extraction capability. Then, the extended features can use BN to reduce overfitting of the model.
2) *Depthwise convolution:* The depthwise convolution of each input channel separately with kernel size 3×3 or 5×5 with S is 1 or 2 and can retain the spatial characteristics of the input feature, reducing the computation and parameters. The specific parameters for each block are specified in.
3) *ESE module:* When the ESE module is used in the block, its attention mechanism can dynamically adjust the channel weights so that the network can adaptively strengthen useful features and suppress invalid channel values, which can improve model performance.
4) *Channel compression and residual connection:* Similar to step 1), 1×1 convolution layers are used to compress the feature channels to the number of input dimension. Moreover,

---

Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:57 UTC from IEEE Xplore. Restrictions apply.
© 2025 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

---

This article has been accepted for publication in IEEE Transactions on Cognitive Communications and Networking. This is the author's version which has not been fully edited and content may change prior to final publication. Citation information: DOI 10.1109/TCCN.2025.3598245

**IEEE TRANSACTIONS ON COGNITIVE COMMUNICATIONS AND NETWORKING, VOL. XX, NO. XX, AUGUST XXXX**
**5**

a residual connection is added to mitigate the vanishing gradient problem. Finally, the BN and activation functions are applied to improve the computational efficiency of the model. The drone spectrogram through 11 inverted residual blocks to extract features and similar to 1) expands the number of output channels of the feature, which produce $H \in \mathbb{R}^{7 \times 7 \times N^{(11)}_{pw}}$ features dimension. Furthermore, to ensure robustness by avoiding weight recalibration in the classifier despite variations in input features, the global average pooling (GAP) operation is applied to compute the scalar representation for each output channel $d \in \{1, 2, \dots, N^{(11)}_{pw}\}$ of the model, which is given by

$$
x_d = \frac{1}{7 \times 7} \sum_{i=1}^7 \sum_{j=1}^7 H_d(i,j), \quad (8)
$$

where $x_d$ is the channel weight of the output feature. Then, the drone features are the BN function by (6) to obtain $\mathbf{x} \in \mathbb{R}^{N^{(11)}_{pw} \times 1}$. In the end, it can construct a multiclassifier based on the ETF architecture by ▾ to recognize the drone model, which will be introduced in the following section.

### B. The Design of ESE Module

The traditional MobileNetV3-Small model can learn the features of the drone RF signal spectrogram. However, to suppress the noise in the features and enhance the effective channel weights, we propose combining the channel attention mechanism and soft thresholding to generate the ESE module as shown in Fig. 2. Here, the input spectrogram features are processed in two stages: 1), and 2) to obtain $H^{(i)}_{dw} \in \mathbb{R}^{P^{(i)}_{dw} \times Q^{(i)}_{dw} \times N^{(i)}_{dw}}, i \in \{1, 2, \dots, 11\}$ after extended channel, $P^{(i)}_{dw}, Q^{(i)}_{dw}$ is the feature size of the i-th inverted residual block after depthwise convolution, $N^{(i)}_{pw}, N^{(i)}_{dw}$ is the number of channels of the expanded features after the pointwise convolution of the i-th inverted residual blocks.

According to, when the inverted residual blocks are employed, the ESE module can dynamically adjust the weight of the channel and suppress the noise weight component in the channel to enhance the computational efficiency of the model. Subsequently, to compress the spatial information of the feature channels, the output $H^{(j)}_{dw}, j \in I$ of the j-th inverted residual block, where $I = \{1, 4, 5, 6, 7, 8, 9, 10, 11\}$ is processed by the ESE module, which applies the GAP according to (8) to generate the channel feature description as $\mathbf{h}^{(j)}_{pw} = \{h_1, h_2, \dots, h_{N^{(j)}_{pw}}\}$. On the other hand, a two-layer FC network is used to enhance the learning of feature channel weights, which is given by

$$
\mathbf{z}^{(j)} = F_2(R(F_1(\mathbf{h}^{(j)}_{pw}))), \quad (9)
$$

where $\mathbf{z}^{(j)} \in \mathbb{R}^{1 \times 1 \times N^{(j)}_{pw}}$ is the initial channel weight of the j-th inverted residual block, and the first FC layer $F_1$ compresses the input feature channel to a reduced dimension of $1 \times 1 \times N^{(j)}_{pw}/4$ channels, where 4 is the compression factor. This compression captures complex inter-channel relationships and enhances the ability of the model to represent nonlinear patterns through the activation function. Finally, the compressed channel features from the F1 layer are then restored to dimension $1 \times 1 \times N^{(j)}_{pw}$ by the second FC layer $F_2$, which reduces the number of parameters and improves the sensitivity of the network to important channel features.

Subsequently, using the idea of soft thresholding, the noise is removed and the important channel feature weights are retained. The input features $H^{(j)}_{dw}$ to the range (0,1), which improves noise suppression for the ESE module, given as

$$
\alpha^{(j)} = \frac{1}{1 + \exp(-H^{(j)}_{dw})}, \quad (10)
$$

where $\alpha^{(j)} \in \mathbb{R}^{P^{(j)}_{dw} \times Q^{(j)}_{dw} \times N^{(j)}_{dw}}$ is the scaling parameter for the j-th inverted residual block. In addition, the noise channel component in the channel is reduced and suppressed in combination with the idea of soft thresholding, which is given as

$$
\mathbf{s}^{(j)} = \sigma(\mathbf{z}^{(j)} \otimes (1 - h_{dw}^{(j)})) (1 - \alpha^{(j)}_{dw}), \quad (11)
$$

where $\mathbf{s}^{(j)} \in \mathbb{R}^{P^{(j)}_{dw} \times Q^{(j)}_{dw} \times N^{(j)}_{dw}}$ is the scale parameter dynamically adjusted for the channel in the j-th inverted residual block, $f(x) = \max(0, \min(1, 0.2x + 0.5))$ represents the HardSigmoid activation function, $1 \in \mathbb{R}^{P^{(j)}_{dw} \times Q^{(j)}_{dw}}$ is the all-one matrix, $\otimes$ denotes the Kronecker product. Finally, the formula for weighting the output feature channels $H^{(j)}_{dw} \in \mathbb{R}^{P^{(j)}_{dw} \times Q^{(j)}_{dw} \times N^{(j)}_{dw}}$ is given as

$$
H^{(j)}_{\text{out}} = H^{(j)}_{dw} \otimes \mathbf{s}^{(j)}, \quad (12)
$$

where $H^{(j)}_{\text{out}}$ output feature is output to the next inverted residual block by 4).

### C. A Fixed Classifier with ETF Architecture

In the conventional MobileNetV3-Small model, drone classification is performed using a two-layer fully connected (FC) network, which adds to the model’s computational complexity and may hinder its efficiency in real-time applications. On the other hand, the training model weights and bias values in the entire training model network will stop updating. In addition, the training loss of the NC phenomenon will be fixed to zero, and the last layer feature vector will converge to the mean of its class in the TPT periodical. The mean and classifier within the same class are combined to form a simple ETF architecture,. Specifically, the NC phenomenon first manifests as a tendency for features of samples of the same class to collapse towards their respective class mean centers, which can be expressed as

$$
\lim_{t \to T} \frac{1}{S} \sum_{s=1}^S \frac{1}{D} \sum_{i=1}^D ||\mathbf{x}_{s,i} - \mathbf{x}_s||_2 = 0, \quad (13)
$$

where $||\cdot||_2$ is the Euclidean Norm, $t$ is the training time approaching the end of the training epoch T, S is the total number of classes, D is the number of training samples in each drone class, $\mathbf{x}_s$ is the mean vector of the s-th class, $n_s$ is the total number of sample in the s-th class, and $\mathbf{x}_{s,i}$ is the i-th samples in the s-th class. Secondly, all class centers and classifier weights forming an ETF structure can be written as

$$
\mathbf{w}^T_s \mathbf{w}_{\hat{s}} = \begin{cases} 1, & s=\hat{s}, \\ -\frac{1}{S-1}, & s \neq \hat{s}. \end{cases} \quad (14)
$$

---

Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:57 UTC from IEEE Xplore. Restrictions apply.
© 2025 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

---

This article has been accepted for publication in IEEE Transactions on Cognitive Communications and Networking. This is the author's version which has not been fully edited and content may change prior to final publication. Citation information: DOI 10.1109/TCCN.2025.3598245

**IEEE TRANSACTIONS ON COGNITIVE COMMUNICATIONS AND NETWORKING, VOL. XX, NO. XX, AUGUST XXXX**
**6**

*Fig. 2. The architecture of the ESE module.*

where $s, \hat{s} \in [1, S]$, $\mathbf{w}_s$ is the vector weight of the s-th class of the traditional classifier. It should be noted that (14) is valid exclusively in the case where $||\mathbf{x}_s - \mathbf{x}_G||_2 = ||\mathbf{x}_{\hat{s}} - \mathbf{x}_G||_2$, with $\mathbf{x}_G$ denoting the global class mean vector. Thirdly, the relationship between the class mean center and the corresponding classifier weight vector can be expressed as

$$
\frac{\mathbf{x}_s}{||\mathbf{x}_s||_2} = \frac{\mathbf{w}_s}{||\mathbf{w}_s||_2} \quad (15)
$$

Fourthly, the class mean classifier ultimately reduces the calculation of the nearest distance to each class mean, which is given as

$$
\hat{s} = \arg\min_{s \in S} ||\mathbf{x}_{\hat{s}i} - \mathbf{x}_s||_2, \quad (16)
$$

where $\hat{s}$ is the predicted class of the model, and $\mathbf{x}_{\hat{s}i}$ represents the last-layer feature of the i-th input sample.

Therefore, since the model forms the NC phenomenon through the above four processes, it is unnecessary to use the FC layer in the traditional MobileNetV3-Small as a classifier, which helps reduce parameters and related issues while maintaining classification performance. Here, the output vectors $\{\mathbf{x}_{\hat{s}i}\}_{i=1}^D$ from the penultimate layer collapse to their respective class means, and the resulting prototype matrix forms a fixed ETF matrix for classifying all classes of drones, given as

$$
C = \eta \sqrt{S-1} Q \left(I - \frac{1}{S} \mathbf{1}\right), \quad (17)
$$

where the fixed ETF architecture classifier $C = [\mathbf{c}_1, \mathbf{c}_2, \dots, \mathbf{c}_S] \in \mathbb{R}^{N^{(11)}_{pw} \times S}$ is composed of point columns, $Q \in \mathbb{R}^{S \times S}$ is a partially orthogonal matrix and satisfies the identity matrix of $Q^T Q = I \in \mathbb{R}^{S \times S}$, $1 \in \mathbb{R}^{S \times S}$ is the all-ones matrix, $\eta$ is a scaling factor of C.

In addition, the cross-entropy (CE) loss function is used to calculate the loss of different drone classes in the proposed MobilenentV3-Small-ETF model. Here, it uses the C to create a fixed linear classifier to predict the drone class. It can be expressed as

$$
\mathbf{p}_s = \eta \mathbf{c}^T_s \mathbf{x} + b_s, \quad (18)
$$

where $p_s, \hat{s} \in \{1, 2, \dots, S\}$ is the probability value of the predicted k classes drone, $b_s$ is the bias value. The cross-entropy loss function of the ETF architecture-based linear classifier employs (18) during the TPT training stage, given as

$$
L_{\text{CE}}(\mathbf{C}, \mathbf{x}; \theta) = -\log \frac{\exp(\eta \mathbf{c}^T_{\hat{s}} \mathbf{x})}{\sum_{s=1}^S \exp(\eta \mathbf{c}^T_s \mathbf{x})}, \quad (19)
$$

where $\hat{s}$ is the drone class label of $\mathbf{x}$, $\theta$ denotes the learnable parameters of the network.

In order to better fit the classifier with a simple ETF architecture, the representation of column vectors and output feature vectors in a linear classifier is given by

$$
\mathbf{c}^T_s \mathbf{x}_{s,i} = \begin{cases} \eta, & s = \hat{s}, 1 \le i \le S_D, \\ -\frac{\eta}{S-1}, & s \neq \hat{s}, 1 \le i \le S_D, \end{cases} \quad (20)
$$

where $\eta$ is the angle value of the same drone class, $\frac{\eta}{S-1}$ is the maximum separation angle of two unrelated drone classes. On the other hand, inspired by information theory, we assume that the backbone network acts as a channel. Moreover, the output of the backbone network denoted as $\mathbf{x}$ can be considered a single codeword. Additionally, the classifier C in the fixed ETF architecture represents the codebook. When the codeword is transmitted through the noisy channel, the receiver receives a distorted version of the codeword. The objective of the proposed fixed ETF architecture classifier is to optimally match the codebook information from the x feature vectors affected by the interference noise. Then, the ETF architecture classification fixed loss can be calculated as

$$
L_{\text{ETF}}(\mathbf{x}, \mathbf{C}) = \log_2(\eta ||\mathbf{c}_{\hat{s}} - \mathbf{c}^T_{\hat{s}} \mathbf{x}||^2_2 + 1). \quad (21)
$$

Furthermore, to ensure stable gradient training of the model, the total loss function of the MobileNetV3-Small-ETF model is given by

$$
\Gamma_{\text{Loss}}(\mathbf{x}, \mathbf{C}; \theta) = L_{\text{CE}}(\mathbf{x}, \mathbf{C}; \theta) + L_{\text{ETF}}(\mathbf{x}, \mathbf{C}), \quad (22)
$$

where $L_{\text{ETF}}(\mathbf{x}, \mathbf{C})$ is the normalization operation.

However, the non-convex nature of neural networks is difficult to analyze due to their highly interconnected architecture. More importantly, the model focuses only on output layer

---

Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:57 UTC from IEEE Xplore. Restrictions apply.
© 2025 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

---

This article has been accepted for publication in IEEE Transactions on Cognitive Communications and Networking. This is the author's version which has not been fully edited and content may change prior to final publication. Citation information: DOI 10.1109/TCCN.2025.3598245

**IEEE TRANSACTIONS ON COGNITIVE COMMUNICATIONS AND NETWORKING, VOL. XX, NO. XX, AUGUST XXXX**
**7**

features and the loss values among the fixed classifiers within the ETF framework. Therefore, the objective function is minimized using the adaptive moment estimation (Adam) optimizer under noise interference in the balanced drone RF signal dataset, and the corresponding optimization formulation is given by

$$
\min_{\theta} \frac{1}{SD} \sum_{s=1}^S \sum_{i=1}^D \Gamma_{\text{Loss}}(\mathbf{x}_{s,i}, \mathbf{C}; \theta). \quad (23)
$$

Based on the above parts of the MobileNetV3-Small-ETF model, the implementation process of the optimization model in the training stage is described in Algorithm 1.

## IV. SIMULATION RESULTS AND DISCUSSION

In this section, we conduct experiments on public drone detection datasets to evaluate the proposed MobileNetV3-Small-ETF model for recognizing drone classes. Then, we compare the method with existing models in terms of performance and complexity. Furthermore, the ablation experiment also demonstrates the effectiveness of the proposed method.

### A. Experimental Implementation and Evaluation Metrics

The hardware platform for the experimental setup includes a single NVIDIA GeForce RTX 4090 GPU equipped with Python 3.11.7 and PyTorch (V2.0.1). During the training process, the dataset contains DJI Air 2S, DJI Inspire 2, DJI Mavic 3 Pro, DJI Mini 2, DJI Mini 3 Pro, and DJI Matrice 200. These signals were collected outdoors at a distance of 20 meters, with a center frequency of 2.44 GHz. We divided the segmented IQ data into 1 ms segments, each containing $1 \times 10^6$ IQ points. For each drone class, 416 training samples were generated, along with 52 validation and 52 test samples, and each sample is represented as a feature map $Z \in \mathbb{R}^{224 \times 224 \times 3}$, which is normalized according to (4). In addition, the PSD estimation of drone samples employed the fast Fourier transform (NFFT), the number of points for the FFT (NFFT) is set to 512, and the overlap length of 120 is selected. Furthermore, the scale factor of the MobileNetV3-Small-ETF model $\eta=1$, the feature layer in the proposed model is set to $N^{(11)}_{pw}=576$, following the Adam optimization method proposed in. Meanwhile, to prevent gradient explosion caused by an excessively high initial learning rate, the cosine annealing strategy proposed in is adopted to gradually reduce the learning rate, thereby improving the convergence stability of the model. Additionally, the warm-up iteration count is set to 2% of the total number of iterations. The hyperparameters are set as follows: the learning rate is $l \in [0.0001, 0.001]$, the batch size is set to 64, and the number of training epochs is set to 180.

On the other hand, the experimental evaluation index can better reflect the performance of the proposed method. Here, in the multi-classification scenario, the average accuracy Avg_acc is given by

$$
\text{Avg\_acc} = \frac{\sum_{i=1}^S TP_i + TN_i}{\sum_{i=1}^S TP_i + FP_i + FN_i + TN_i} \quad (24)
$$

*Fig. 3. Comparison of the proposed model with other methods.*

In addition, the macro average method is used as the core evaluation index of the model to evaluate the impact of each drone class on the model. The computation of the macro precision $P_{\text{Macro}}$, macro recall $R_{\text{Macro}}$, and macro F1 score $F1_{\text{Macro}}$ is given by

$$
P_{\text{Macro}} = \frac{1}{S} \sum_{i=1}^S \frac{TP_i}{TP_i + FP_i}, \quad (25)
$$

$$
R_{\text{Macro}} = \frac{1}{S} \sum_{i=1}^S \frac{TP_i}{TP_i + FN_i}, \quad (26)
$$

$$
F1_{\text{Macro}} = 2 \frac{P_{\text{Macro}} \times R_{\text{Macro}}}{P_{\text{Macro}} + R_{\text{Macro}}}, \quad (27)
$$

where $TP_i, TN_i, FP_i$, and $FN_i$ denote the true positive, true negative, false positive, and false negative values of the i-th class, respectively.

### B. Comparison of Experimental Performance

First, in order to evaluate the generalization capability of the model, we add noise to the signals to create test samples at various signal-to-noise ratio (SNR) levels. As shown in Fig. 3, the ResNeXt-50 model achieves the highest accuracy at low SNR, where SNR $\in$ dB. Furthermore, this also highlights the challenge of drone identification under strong interference and low SNR conditions, where even the best models face performance degradation. However, with increasing SNR, the accuracy of drone identification for the proposed method increases rapidly. For example, the proposed method achieves an average recognition accuracy of 78.27%, which is approximately 7.5% higher than the accuracy of traditional lightweight networks such as ShuffleNetV2 and Efficient-NetV2 at 15 dB. Meanwhile, simulation results show that the proposed MobileNetV3-Small-ETF model surpasses all other models in performance and improves the drone recognition average accuracy surpass 8% compared to the lightweight EfficientNetV2 network in the SNR range of dB.

---

Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:57 UTC from IEEE Xplore. Restrictions apply.
© 2025 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

---

This article has been accepted for publication in IEEE Transactions on Cognitive Communications and Networking. This is the author's version which has not been fully edited and content may change prior to final publication. Citation information: DOI 10.1109/TCCN.2025.3598245

**IEEE TRANSACTIONS ON COGNITIVE COMMUNICATIONS AND NETWORKING, VOL. XX, NO. XX, AUGUST XXXX**
**8**

**Algorithm 1** Training Procedure of the Proposed Drone Recognition Based on MobileNetV3-Small-ETF Model.
Require:

- E: Number of training epochs;
- D: Training dataset for each drone class;
- N: Number of batch-size;
- l: Learning rate;
- Z: Size of feature map H × W × C;
- S: Number of drone classes;
- $\theta$: Parameters of MobileNetV3-Small-ETF;
  Dataset processing:
  1: Normalize the three channels of each Z using Eq. (4);
  Model Initialization:
  2: Initialize the model parameters $\theta$;
  3: Select an Adam optimizer;
  4: Select a cross-entropy loss;
  5: Define the fixed ETF classifier according to Eq. (17);
  Training Process:
  6: for e = 1; e <= E; e++ do
  7: for n = 1; n <= N; n++ do
  8: The extracted features by Eq. (7);
  9: if j $\in$ I then
  10: The $H^{(j)}_{\text{out}}$ is obtained by Eq. (12);
  11: else
  12: end if
  13: Compute the predicted classes index $\hat{s}$;
  14: Calculate loss $\Gamma_{\text{Loss}}(\mathbf{x}, \mathbf{C}; \theta)$ by Eq. (22)
  15: Updating the parameter of $\theta^{e,n+1}$:
  16: $\theta^{e,n+1} \leftarrow \text{Adam}(\nabla_{\theta} \Gamma_{\text{Loss}}, l, \theta)$;
  17: end for
  18: end for

*Fig. 4. Comparison of $P_{\text{Macro}}$, $R_{\text{Macro}}$, and $F1_{\text{Macro}}$ across different methods.*

*Fig. 5. Classification accuracy was compared across different numbers of drone classes.*

To provide a more comprehensive evaluation of the performance of the model, the proposed model was compared with four models (ResNet-18, ResNeXt-50, DenseNet-169, and ConvNeXt-Small), which are improved versions based on established neural network models. Moreover, a comparative analysis was conducted with two lightweight networks, ShuffleNetV2 and EfficientNetV2. The evaluation focused on $P_{\text{Macro}}$, $R_{\text{Macro}}$, and $F1_{\text{Macro}}$ scores as shown in Fig. 4. Furthermore, all models were evaluated on the preprocessed test set. Based on the analysis of the $P_{\text{Macro}}$ scores, the EfficientNetV2 model has suboptimal performance in correctly classifying the drone samples, and the $P_{\text{Macro}}$ is only 91%. The existing improvements based on the established models have the same level of $P_{\text{Macro}}$, reaching 92%. However, the proposed model outperforms all existing models with a $P_{\text{Macro}}$ of 96%. In addition, the $R_{\text{Macro}}$ of the proposed model reaches 96.5%, which is 7% higher than that of the EfficientNetV2 model, and an average increase of 2.63% over the established neural network models. Similarly, the proposed model achieves an $F1_{\text{Macro}}$ score of 96.5%, outperforming other models.

To evaluate the scalability and generalization performance of the proposed model, experiments were conducted using datasets with four, five, and six drone classes. Specifically, the five-class dataset incorporated the DJI Mini 3 Pro, while the six-class dataset added the DJI Matrice 200 to the existing five types, as shown in Fig. 5. The results show that while the conventional MobileNetV3-Small model achieves accuracy of 92.47%, 92.31%, and 93.27% respectively, the proposed MobileNetV3-Small-ETF model outperforms it by margins of 4.17%, 4.23%, and 3.52%. Furthermore, as the number of drone classes increases, the proposed model demonstrates even stronger performance. As demonstrated in the experiments, the model achieves a classification accuracy of 96.79% on six previously established drone classes through the integration of additional functional modules. These experiments demonstrate that our method not only maintains robust performance as the number of drone classes increases, but also exhibits superior generalization across different types.

The performance of the proposed MobileNetV3-Small-ETF method can be visualized in the semantic space of the 3D projection of the trained and test samples for datasets that do not add additional noise. As shown in Fig. 6, the proposed method employs a predefined ETF classifier with a fixed architecture, which effectively increases the inter-class

---

Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:57 UTC from IEEE Xplore. Restrictions apply.
© 2025 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

---

This article has been accepted for publication in IEEE Transactions on Cognitive Communications and Networking. This is the author's version which has not been fully edited and content may change prior to final publication. Citation information: DOI 10.1109/TCCN.2025.3598245

**IEEE TRANSACTIONS ON COGNITIVE COMMUNICATIONS AND NETWORKING, VOL. XX, NO. XX, AUGUST XXXX**
**9**

**TABLE I**
COMPARISON OF PERFORMANCE FOR DIFFERENT NETWORK MODELS

| Model          | FLOPS                | Params               | Memory consumption (MB) | Inference time (ms) | Avg_acc (%) |
| :------------- | :------------------- | :------------------- | :---------------------- | :------------------ | :---------- |
| ResNet-18      | $1.81 \times 10^9$ | $1.81 \times 10^7$ | 42.64                   | 3.50                | 94.23%      |
| DenseNet-169   | $3.38 \times 10^9$ | $1.25 \times 10^7$ | 47.65                   | 13.1                | 94.23%      |
| ResNeXt-50     | $4.24 \times 10^9$ | $2.30 \times 10^7$ | 87.69                   | 4.30                | 93.87%      |
| ConvNeXt-Small | $8.69 \times 10^9$ | $4.95 \times 10^7$ | 188.67                  | 5.90                | 95.50%      |
| ShuffleNetV2   | $1.46 \times 10^8$ | $1.26 \times 10^6$ | 4.80                    | 4.00                | 95.19%      |
| EfficientNetV2 | $5.39 \times 10^9$ | $5.29 \times 10^7$ | 201.66                  | 13.4                | 93.00%      |
| Our proposed   | $5.69 \times 10^7$ | $9.27 \times 10^5$ | 3.54                    | 4.10                | 96.64%      |

*Fig. 6. The visualization of a 3D semantic space is based on drone samples.*

distance between sample points while promoting clustering around the class means, thus enhancing intra-class cohesion. At the same time, different drone classes in the test samples can be effectively distinguished by the ETF classifier. As shown in Table I, simulation experiments show that the average accuracy of the proposed method in classifying drone classes is 96.64%, and its model performance is better than that of traditional classifier methods. Moreover, the proposed model achieves an average accuracy of more than 1% higher than that of other models. Specifically, it outperforms the traditional densely connected ResNeXt-50 by 2.77% and the lightweight EfficientNetV2 by 3.64% in average accuracy for drone class prediction. On the other hand, the ConvNeXt-Small model with large kernel depthwise convolutions achieves the performance closest to our proposed method, with its drone model prediction accuracy only 1.14% lower.

### C. Complexity Analysis

This section evaluates the complexity and efficiency of the drone identification model. Specifically, this study assess four factors in the selected model: FLOPs, parameters, memory consumption, and inference time. The computational complexity of a model is often measured in terms of its FLOPs. The FLOPs calculation method for the MobileNetV3-Small-ETF model is introduced in this paper. The MobileNetV3-Small-ETF model is composed of inverted residual blocks incorporating depthwise separable convolutions and ESE modules, along with a fixed classifier. Firstly, the FLOPs of the model based on standard convolution (SC) can be expressed as

$$
\text{SC}_{\text{FLOPS}} = HWCNK^2, \quad (28)
$$

where H and W denote the height and width of the input feature, K represents the size of the convolution kernel used by SC in (5), C denotes input channels, N denotes the number of output feature channels produced by the SC operation. The computational complexity of the pointwise and depthwise convolutions within the depthwise separable convolution (DSC) is given by

$$
\text{DSC}_{\text{FLOPS}} = \sum_{i=1}^{11} P^{(i)}_{dw} Q^{(i)}_{dw} N^{(i)}_{dw} (K^{(i)}_{dw})^2 + P^{(i)}_{pw} Q^{(i)}_{pw} (N^{(i)}_{dw} N^{(i)}_{in} + N^{(i)}_{pw} N^{(i)}_{out}), \quad (29)
$$

where $P^{(i)}_{dw}$ and $Q^{(i)}_{dw}$ denote the height and width of the input feature maps to the depthwise convolution in the i-th residual block, respectively. The kernel size of the depthwise convolution is denoted by $K^{(i)}_{dw}$, which is typically set to 3 or 5. In the pointwise convolution, $P^{(i)}_{pw}$ and $Q^{(i)}_{pw}$ represent the spatial dimensions of the feature maps in the i-th expansion and projection convolution layers, respectively. Additionally, $N^{(i)}_{in}$ denotes the number of input channels in the i-th expansion convolution, while $N^{(i)}_{out}$ represents the number of output channels in the corresponding projection convolution, which employs a 1 × 1 convolution kernel. Furthermore, within the proposed ESE module, the FLOPs produced by each use of the ESE module by (10) and (11) can be given by

$$
\text{ESE}_{\text{Total FLOPs}} = \sum_{j \in I} \frac{2(N^{(j)}_{pw})^2}{r} + 5H^{(j)}_{dw} W^{(j)}_{dw} N^{(j)}_{pw} + \frac{(N^{(j)}_{pw})^2}{r}, \quad (30)
$$

where I corresponds to the indices introduced in the ESE module, and r denotes the compression ratio. Here, the computational complexity of the traditional SE module is $2(N^{(j)}_{pw})^2/r$. On the other hand, the FLOPs calculation of the sampled ETF fixed architecture classifier is given by (18) and can be expressed as

$$
\text{ETF}_{\text{FLOPS}} = 2N^{(11)}_{pw} S - S, \quad (31)
$$

where S is the number of drone classes. To enable a direct comparison of computational complexity with conventional approaches, we provide the FLOPs calculation for the standard fully connected (FC) layer classifier in MobileNetV3-Small, which is given as

$$
\text{FC}_{\text{FLOPS}} = 2N^{(11)}_{pw} D_1 + 2D_1S, \quad (32)
$$

where $N^{(11)}_{pw}$ is the number of input characteristic channels, $D_1$ is the output dimension of the first fully connected layer.

---

Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:57 UTC from IEEE Xplore. Restrictions apply.
© 2025 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

---

This article has been accepted for publication in IEEE Transactions on Cognitive Communications and Networking. This is the author's version which has not been fully edited and content may change prior to final publication. Citation information: DOI 10.1109/TCCN.2025.3598245

**IEEE TRANSACTIONS ON COGNITIVE COMMUNICATIONS AND NETWORKING, VOL. XX, NO. XX, AUGUST XXXX**
**10**

*Fig. 7. The confusion matrices for predicting drone classes under different models: (a) Baseline model, (b) MobileNetV3-Small model using the ETF classifier, (c) the traditional model using the ETF classifier and ESE module, and (d) the proposed model.*

As shown in Table I, the proposed model replaces two fully connected layers of the MobileNetV3-Small classifier with an ETF classifier and displays lower FLOPs than other models. The proposed model applies SC to the first feature input layer. Within each inverted residual block, two 1 × 1 pointwise convolution kernels, denoted as $N_{out}$, are employed together with a single depthwise convolution. Conversely, traditional models such as ResNet, DenseNet, and ResNeXt rely on SC and dense connectivity patterns, resulting in significantly higher computational costs. Consequently, compared with the proposed method, the FLOPs of ResNet-18, DenseNet-169, and ResNeXt-50 increased by 96.86%, 98.31%, and 98.66%, respectively. Furthermore, the SC kernel size is 7 × 7, and the global attention mechanism of the Transformer architecture is integrated into the ConvNeXt-Small model. As a result, the FLOPs of the MobileNetV3-Small-ETF model are reduced by 99.34% compared to the conventional ConvNeXt-Small. Although depthwise separable convolutions are widely used in conventional lightweight networks, effective channel compression mechanisms are generally lacking compared to the proposed model. As a result, the ShuffleNetV2 and EfficientNetV2 models exhibit 60.99% and 98.95% higher computational costs in FLOPs.

Moreover, the proposed model demonstrates a substantial reduction in the number of trainable parameters. It is reduced by more than 85% compared to most conventional models, with a reduction of up to 98.13% compared to the ConvNeXt-Small model. In particular, the MobileNetV3-Small-ETF model achieves the lowest memory consumption of all comparison models, with a 95.96% reduction in memory usage compared to ResNeXt-50. The proposed method utilizes a fixed ETF architecture classifier, which reduces the number of learnable parameters and results in a 26.29% decrease compared to the traditional ShuffleNetV2 model. Although ResNet-18 and ShuffleNetV2 are 17.14% and 2.50% faster than the proposed model during inference, their other per-

---

Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:57 UTC from IEEE Xplore. Restrictions apply.
© 2025 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

---

This article has been accepted for publication in IEEE Transactions on Cognitive Communications and Networking. This is the author's version which has not been fully edited and content may change prior to final publication. Citation information: DOI 10.1109/TCCN.2025.3598245

**IEEE TRANSACTIONS ON COGNITIVE COMMUNICATIONS AND NETWORKING, VOL. XX, NO. XX, AUGUST XXXX**
**11**

formance metrics are lower. Therefore, the proposed model achieves the best overall trade-off across various indicators.

**TABLE II**
COMPARISON OF ABLATION EXPERIMENT RESULTS

| Model                        | FLOPS                | Params               | Avg_acc (%) |
| :--------------------------- | :------------------- | :------------------- | :---------- |
| MobileNetV3-Small            | $5.75 \times 10^7$ | $1.52 \times 10^6$ | 92.47%      |
| MobileNetV3-Small + ETF      | $5.66 \times 10^7$ | $9.27 \times 10^5$ | 93.63%      |
| MobileNetV3-Small + ETF+ ESE | $5.69 \times 10^7$ | $9.27 \times 10^5$ | 95.19%      |
| Our proposed                 | $5.69 \times 10^7$ | $9.27 \times 10^5$ | 96.64%      |

### D. Ablation Experiment

Ablation experiments can evaluate the effectiveness of the proposed model. We provide a traditional MobileNetV3-Small model that is used as a baseline to systematically evaluate how the integration of the ETF classifier, ESE module, and fixed ETF classifier loss affects model performance. First, as shown in Table II, the FLOPs computed by the MobileNetV3-Small method using the fully connected layer classifier are 10.33% higher than the proposed method, model parameters are increased by 39.09%, and the average recognition accuracy of drone classes is reduced by 4.17%. In addition, the FLOPs of the traditional MobileNetCV3-Small using the ETF architecture classifier are 1.53% lower than the MobileNetV3-Small model, while the average accuracy is improved by 1.16%. The proposed method increases FLOPs by 0.53% compared to the traditional MobileNetV3-Small-ETF but improves average accuracy by 3.01% through the soft thresholding and the broadcasting mechanism. Finally, by incorporating a fixed ETF architecture loss during training, the proposed model achieves a performance improvement of 1.45% over the models using ETF classification and ESE modules only at inference time. To visually illustrate the impact of different drone classes on the performance of the proposed method, the confusion matrices for identifying four types of drones under the model ablation experiments are shown in Fig. 7. The fixed classifier architecture used by the traditional MobileNetV3-Small-ETF can clearly distinguish different drone classes, and the accuracy of predicting DJI Inspire 2 and DJI Mavic Pro is improved by 3.85% and 3.85%, respectively. Moreover, compared to using a fixed ETF classifier alone, integrating the ESE module into the traditional MobileNetV3-Small-ETF model improves DJI Inspire 2 recognition accuracy by 3.84%. In addition, our proposed method incorporates the ETF architecture classifier loss, making the model capture information from the training samples more effectively, resulting in a 92.31% prediction accuracy for the DJI Mavic Pro. This demonstrates that our proposed model achieves more comprehensive performance in identifying drone classes, significantly outperforming the three models above, with over 92% accuracy for each drone type.

In addition, a well-designed loss function can accelerate model convergence during the training stage. As shown in Fig. 8, the convergence speed of the MobileNetV3-Small model is fast in the training phase, but it can be seen from Table II that the learning efficiency of the sample is not the best, and the model is more complex and requires higher computing power. On the other hand, the number of training epochs ∈, the traditional MobileNetV3-Small model with the ETF classifier alone shows a relative increase in training loss, but by introducing the ESE module, the training loss $L_{\text{ETF}}(\mathbf{x}, \mathbf{C}; \theta)$ is significantly reduced. Moreover, the incorporation of a fixed ETF architecture loss further enhances the training efficiency and convergence performance of the proposed model. Furthermore, the experiment shows that different modules affect the convergence performance of the MobileNetV3-Small model.

## V. CONCLUSION

In this paper, we introduce a novel MobileNetV3-Small model enhanced with a fixed architecture classifier and an ESE module for intelligent drone identification using RF signals. Experimental results show that our proposed method achieved superior recognition performance for four types of drone signals as the additional noise with varying SNR levels is gradually reduced. Moreover, simulation results display that our approach outperforms other models in important metrics, including computational complexity and average accuracy. This study also validates the model using different numbers of drone classes, with results confirming the scalability of the model. Additionally, 3D semantic space analysis of training and test samples reveals that the fixed ETF classifier effectively identifies drone classes. Finally, the ablation experiment confirms that replacing the fully connected layer classifier with a fixed architecture classifier reduces model parameters by 39.09% and FLOPs by 15.33%. The proposed MobileNetV3-Small-ETF model incorporates the ESE module and combines ETF classification loss with the CE loss function, surpassing other existing methods in performance. In future work, we will focus on extending drone identification in complex environments, along with the establishment of drone datasets to support the design of efficient lightweight models.

## REFERENCES

 G. Ding, Q. Wu, L. Zhang, Y. Lin, T. A. Tsiftsis, and Y. Yao, “An amateur drone surveillance system based on the cognitive Internet of Things,” *IEEE Commun. Mag.*, vol. 56, no. 1, pp. 29–35, Jan. 2018.

*Fig. 8. The epochs of TPT stage for different models.*

---

Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:57 UTC from IEEE Xplore. Restrictions apply.
© 2025 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

---

This article has been accepted for publication in IEEE Transactions on Cognitive Communications and Networking. This is the author's version which has not been fully edited and content may change prior to final publication. Citation information: DOI 10.1109/TCCN.2025.3598245

**IEEE TRANSACTIONS ON COGNITIVE COMMUNICATIONS AND NETWORKING, VOL. XX, NO. XX, AUGUST XXXX**
**12**

 S. Basak, S. Rajendran, S. Pollin and B. Scheers, “Combined RF-Based Drone Detection and Classification,” *IEEE Trans. Cogn. Commun. Netw.*, vol. 8, no. 1, pp. 111–120, March. 2022.
 China News, “Tianjin airport experiences large-scale flight delays due to drones, official report,” China News, Sep. 12, 2024. [Online]. Available: https://m.gmw.cn/2024-09/12/content_1303846331.htm
 B. He, X. Ji, G. Li and B. Cheng, “Key Technologies and Applications of UAVs in Underground Space: A Review,” *IEEE Trans. Cogn. Commun. Netw.*, vol. 10, no. 3, pp. 1026–1049, June. 2024.
 X. Shi, C. Yang, W. Xie, C. Liang, Z. Shi, and J. Chen, “Antidrone system with multiple surveillance technologies: Architecture, implementation, and challenges,” *IEEE Commun. Mag.*, vol. 56, no. 4, pp. 68–74, Apr. 2018.
 P. K. Rai et al., “Localization and activity classification of unmanned aerial vehicle using mmWave FMCW radars,” *IEEE Sensors J.*, vol. 21, no. 14, pp. 16043–16053, Jul. 2021.
 E. Kaya and G. B. Kaplan, “Neural network based drone recognition techniques with non-coherent S-band radar,” in *2021 IEEE Radar Conf. (RadarConf21)*, 2021, pp. 1–6.
 M. Z. Anwar, Z. Kaleem, and A. Jamalipour, “Machine learning inspired sound-based amateur drone detection for public safety applications,” *IEEE Trans. Veh. Technol.*, vol. 68, no. 3, pp. 2526–2534, Mar. 2019.
 H. Dong, J. Liu, C. Wang, H. Cao, C. Shen, and J. Tang, “Drone detection method based on the time-frequency complementary enhancement model,” *IEEE Trans. Instrum. Meas.*, vol. 72, pp. 1–12, 2023.
 Y. Zheng, Z. Chen, D. Lv, Z. Li, Z. Lan, and S. Zhao, “Air-to-air visual detection of micro-UAVs: An experimental evaluation of deep learning,” *IEEE Robot. Autom. Lett.*, vol. 6, no. 2, pp. 1020–1027, Apr. 2021.
 G. Wu, F. Zhou, K. Kit Wong, and X.-Y. Li, “A vehicle-mounted radar-vision system for precisely positioning clustering UAVs,” *IEEE J. Sel. Areas Commun.*, vol. 42, no. 10, pp. 2688–2703, Oct. 2024.
 T. Zhao, X. Wang, and S. Mao, “Cross-domain, scalable, and interpretable RF device fingerprinting,” in *Proc. IEEE Conf. Comput. Commun. (INFOCOM)*, 2024, pp. 2099–2108.
 J. He, S. Huang, Y. Chen, S. Chang, Y. Zhang, and Z. Feng, “Radio frequency fingerprint identification for OFDM system considering unknown multipath fading channel,” *IEEE Trans. Cogn. Commun. Netw.*, vol. 10, no. 6, pp. 2076–2087, Dec. 2024.
 Y. Xie, P. Jiang, Y. Gu, and X. Xiao, “Dual-Source Detection and Identification System Based on UAV Radio Frequency Signal,” *IEEE Trans. Instrum. Meas.*, vol. 70, pp. 1–15, 2021.
 C. Zhao, C. Chen, Z. Cai, M. Shi, X. Du, and M. Guizani, “Classification of small UAVs based on auxiliary classifier Wasserstein GANs,” in *Proc. IEEE Global Commun. Conf. (GLOBECOM)*, 2018, pp. 206–212.
 A. Alipour-Fanid, M. Dabaghchian, N. Wang, P. Wang, L. Zhao, and K. Zeng, “Machine Learning-Based Delay-Aware UAV Detection and Operation Mode Identification Over Encrypted Wi-Fi Traffic,” *IEEE Trans. Inf. Forensics Security*, vol. 15, pp. 2346–2360, 2020.
 C. J. Swinney and J. C. Woods, “K-means clustering approach to UAS classification via graphical signal representation of radio frequency signals for air traffic early warning,” *IEEE Trans. Intell. Transp. Syst.*, vol. 23, no. 12, pp. 24957–24965, Dec. 2022.
 R. Zhao, T. Li, Y. Li, Y. Ruan, and R. Zhang, “Anchor-Free Multi-UAV Detection and Classification Using Spectrogram,” *IEEE Internet Things J.*, vol. 11, no. 3, pp. 5259–5272, Feb. 2024.
 N. Soltani, G. Reus-Muns, B. Salehi, J. Dy, S. Ioannidis, and K. Chowdhury, “RF fingerprinting unmanned aerial vehicles with non-standard transmitter waveforms,” *IEEE Trans. Veh. Technol.*, vol. 69, no. 12, pp. 14785–14796, Dec. 2020.
 X. Zhao, L. Wang, Q. Wang, and J. Wang, “A hierarchical framework for drone identification based on radio frequency machine learning,” in *2022 IEEE Int. Conf. Commun. Workshops (ICC Workshops)*, 2022, pp. 391–396.
 Z. Wang, Z. Cao, J. Xie, W. Zhang, and Z. He, “RF-based drone detection enhancement via a generalized denoising and interference-removal framework,” *IEEE Signal Process. Lett.*, vol. 31, pp. 929–933, 2024.
 M. Zuo, S. Xie, X. Zhang, and M. Yang, “Recognition of UAV video signal using RF fingerprints in the presence of WiFi interference,” *IEEE Access*, vol. 9, pp. 88844–88851, 2021.
 R. Akter, V.-S. Doan, A. Zainudin and D.-S. Kim, “Sparsely Connected Low Complexity CNN for Unmanned Vehicles Detection—Sensing RF Signal,” *IEEE Trans. Veh. Technol.*, vol. 73, no. 10, pp. 14236–14251, Oct. 2024.
 H. Liang, R. Wang, M. Xu, F. Zhou, Q. Wu and O. A. Dobre, “Few-Shot Learning UAV Recognition Methods Based on the Tri-Residual Semantic Network,” *IEEE Commun. Lett.*, vol. 26, no. 9, pp. 2072–2076, Sept. 2022.
 Y. Chen, L. Zhu, Y. Jiao, C. Yao, K. Cheng, and Y. Gu, “An extreme value theory-based approach for reliable drone RF signal identification,” *IEEE Trans. Cogn. Commun. Netw.*, vol. 10, no. 2, pp. 547–555, Apr. 2024.
 N. Yu, J. Wu, C. Zhou, Z. Shi, and J. Chen, “Open set learning for RF-based drone recognition via signal semantics,” *IEEE Trans. Inf. Forensics Security*, vol. 19, pp. 9894–9909, 2024.
 C. Xu, F. He, B. Chen, Y. Jiang, and H. Song, “Adaptive RF fingerprint decomposition in micro UAV detection based on machine learning,” in *2021 IEEE Int. Conf. Acoust., Speech, Signal Process. (ICASSP)*, 2021, pp. 7968–7972.
 G. Reus-Muns and K. Chowdhury, “Classifying UAVs with proprietary waveforms via preamble feature extraction and federated learning,” *IEEE Trans. Veh. Technol.*, vol. 70, no. 7, pp. 6279–6290, Jul. 2021.
 C. Xue, T. Li, Y. Li, Y. Ruan, and R. Zhang, “Radio frequency identification for drones using spectrogram and CNN,” in *2022 IEEE Glob. Commun. Conf. (GLOBECOM)*, 2022, pp. 4564–4569.
 C. Xue, T. Li, Y. Li, Y. Ruan, R. Zhang, and O. A. Dobre, “Radio-frequency identification for drones with nonstandard waveforms using deep learning,” *IEEE Trans. Instrum. Meas.*, vol. 72, pp. 1–13, 2023.
 H. Zhang, T. Li, N. Su, D. Wei, Y. Li, and Z. Wen, “Drone identification based on normalized cyclic prefix correlation spectrum,” *IEEE Trans. Cogn. Commun. Netw.*, vol. 10, no. 4, pp. 1239–1249, Aug. 2024.
 K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2016, pp. 770–778.
 G. Huang, Z. Liu, L. Van Der Maaten, and K. Q. Weinberger, “Densely connected convolutional networks,” in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2017, pp. 2261–2269.
 S. Xie, R. Girshick, P. Dollár, Z. Tu, and K. He, “Aggregated residual transformations for deep neural networks,” in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2017, pp. 5987–5995.
 J. Hu, L. Shen, and G. Sun, “Squeeze-and-excitation networks,” in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2018, pp. 7132–7141.
 Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie., “A convnet for the 2020s,” in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2022, pp. 11976–11986.
 N. Ma, X. Zhang, H. T. Zheng, and J. Sun, “Shufflenet v2: Practical guidelines for efficient CNN architecture design,” in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, 2018, pp. 116–131.
 A. Howard et al., “Searching for MobileNetV3,” in *Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)*, 2019, pp. 1314–1324.
 M. Tan and Q. Le, “EfficientNetV2: Smaller models and faster training,” in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2021, pp. 10096–10106.
 V. Papyan, X. Y. Han, and D. L. Donoho, “Prevalence of neural collapse during the terminal phase of deep learning training,” *Proc. Nat. Acad. Sci.*, vol. 117, no. 40, pp. 24652–24663, 2020.
 Y. Yang, S. Chen, X. Li, L. Xie, Z. Lin, and D. Tao, “Inducing neural collapse in imbalanced learning: do we really need a learnable classifier at the end of deep neural network?,” in *Proc. 36th Int. Conf. Neural Inf. Process. Syst. (NeurIPS)*, 2022, pp. 1–12.
 J. Zhou, X. Li, T. Ding, C. You, Q. Qu, and Z. Zhu, “On the optimization landscape of neural collapse under MSE loss: global optimality with unconstrained features,” in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2022, pp. 27179–27202.
 Y. Chen, L. Zhu, J. Zhang, L. Yu, and Y. Gu, “Counterfactual threshold learning for drone RF signal classification under interference conditions,” *IEEE Wireless Commun. Lett.*, vol. 13, no. 7, pp. 1–5, Jul. 2024.
 C. E. Shannon, “A mathematical theory of communication,” *Bell System Technical Journal*, vol. 27, no. 3, pp. 379–423, Jul. 1948.
 N. Yu, S. Mao, C. Zhou, G. Sun, Z. Shi, and J. Chen, “DroneRFa: A large-scale dataset of drone radio frequency signals for detecting low-altitude drones,” *J. Electron. Inf. Technol.*, vol. 45, pp. 1–10, Jan. 2023.
 D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” 2017, arXiv:1412.6980.
 I. Loshchilov and F. Hutter, “SGDR: Stochastic gradient descent with warm restarts,” 2017, arXiv:1608.03983.

---

Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:57 UTC from IEEE Xplore. Restrictions apply.
© 2025 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

---

This article has been accepted for publication in IEEE Transactions on Cognitive Communications and Networking. This is the author's version which has not been fully edited and content may change prior to final publication. Citation information: DOI 10.1109/TCCN.2025.3598245

**IEEE TRANSACTIONS ON COGNITIVE COMMUNICATIONS AND NETWORKING, VOL. XX, NO. XX, AUGUST XXXX**
**13**

**Feifei Zhu** is currently pursuing the Ph.D. degree in information and communication engineering with the College of Electronic and Information Engineering, Nanjing University of Aeronautics and Astronautics. His research interests include UAV detection and identification, deep learning, and statistical signal processing.

**Fuhui Zhou** (Senior Member, IEEE) is currently a Full Professor at Nanjing University of Aeronautics and Astronautics. He is also with Key Laboratory of Dynamic Cognitive System of Electromagnetic Spectrum Space, Nanjing University of Aeronautics and Astronautics. He is an IEEE Senior Member. His research interests focus on cognitive radio, cognitive intelligence, knowledge graph, edge computing, and resource allocation. He was awarded as IEEE Com-Soc Asia-Pacific Outstanding Young Researcher and Young Elite Scientist Award of China and URSI GASS Young Scientist. He serves as an Editor of IEEE Transactions on Communications, IEEE Systems Journal, IEEE Wireless Communications Letters, IEEE Access and Physical Communications.

**Rui Ding** (Student Member, IEEE) is currently pursuing the Ph.D. degree with the School of Electronic and Information Engineering, Nanjing University of Aeronautics and Astronautics. His current research interests include deep reinforcement learning, deep learning, and deep active inference with applications in resource allocation, spectrum management for wireless communications networks including data-and-knowledge driven cognitive and decision-making, intelligent spectrum sharing, and air-ground communications.

**Qihui Wu** (Fellow, IEEE) received the B.S. degree in communications engineering, the M.S. and Ph.D. degrees in communications and information systems from the Institute of Communications Engineering, Nanjing, China, in 1994, 1997, and 2000, respectively. From 2003 to 2005, he was a Postdoctoral Research Associate with Southeast University, Nanjing, China. From 2005 to 2007, he was an Associate Professor with the College of Communications Engineering, PLA University of Science and Technology, Nanjing, China, where he was a Full Professor from 2008 to 2016. SinceMay 2016, he has been a Full Professor with the College of Electronic and Information Engineering, Nanjing University of Aeronautics and Astronautics, Nanjing, China. From March 2011 to September 2011, he was an Advanced Visiting Scholar with the Stevens Institute of Technology, Hoboken, USA. His current research interests span the areas of wireless communications and statistical signal processing, with emphasis on system design of software defined radio, cognitive radio, and smart radio.

**Kai-Kit Wong** (Fellow, IEEE) received the B.Eng., M.Phil., and Ph.D. degrees in electrical and electronic engineering from the Hong Kong University of Science and Technology, Hong Kong, in 1996, 1998, and 2001, respectively. After graduation, he took up academic and research positions with the University of Hong Kong, Lucent Technologies, Bell-Labs, Holmdel, the Smart Antennas Research Group of Stanford University, and the University of Hull, U.K. He is the Chair in wireless communications with the Department of Electronic and Electrical Engineering, University College London, London, U.K. His research focuses on 5G and beyond mobile communications. Dr.Wong was a corecipient of the 2013 IEEE Signal Processing Letters Best Paper Award and the 2000 IEEE VTS Japan Chapter Award at the IEEE Vehicular Technology Conference in Japan in 2000, and a few other international Best Paper Awards. He is a Fellow of IET and is also on the editorial board of several international journals. Since 2020, he was the Editor-in-Chief of IEEE WIRELESS COMMUNICATIONS LETTERS.

**Chan-Byoung Chae** (Fellow, IEEE) received the Ph.D. degree in electrical and computer engineering from The University of Texas at Austin (UT), USA, in 2008. Prior to joining UT, he was a Research Engineer with the Telecommunications Research and Development Center, Samsung Electronics, Suwonsi, South Korea, from 2001 to 2005. He is currently an Underwood Distinguished Professor with the School of Integrated Technology, Yonsei University, South Korea. Before joining Yonsei University, he was with Bell Labs, Alcatel-Lucent, Murray Hill, NJ, USA, from 2009 to 2011, as a Member of Technical Staff, and Harvard University, Cambridge, MA, USA, from 2008 to 2009, as a Post-Doctoral Research Fellow. Dr. Chae is a fellow of the National Academy of Engineering of Korea (NAEK). He was a recipient/co-recipient of the KICS Haedong Scholar Award in 2023, the CES Innovation Award in 2023, the IEEE ICC Best Demo Award in 2022, the IEEE WCNC Best Demo Award in 2020, the Best Young Engineer Award from NAEK in 2019, the IEEE DySPAN Best Demo Award in 2018, the IEEE/KICS JOURNAL OF COMMUNICATIONS AND NETWORKS Best Paper Award in 2018, the IEEE INFOCOM Best Demo Award in 2015, the IEIE/IEEE Joint Award for Young IT Engineer of the Year in 2014, the KICS Haedong Young Scholar Award in 2013, the IEEE Signal Processing Magazine Best Paper Award in 2013, the IEEE ComSoc AP Outstanding Young Researcher Award in 2012, and the IEEE VTS Dan. E. Noble Fellow-ship Award in 2008. He has held several editorial positions, including the Editor-in-Chief of IEEE TRANSACTIONS ON MOLECULAR, BIOLOGICAL AND MULTI-SCALE COMMUNICATIONS, a Senior Editor of IEEE WIRE-LESS COMMUNICATIONS LETTERS, and an Editor of IEEE Communications Magazine, IEEE TRANSACTIONS ON WIRELESS COMMUNICATIONS, and IEEE WIRELESS COMMUNICATIONS LETTERS. He was an IEEE ComSoc Distinguished Lecturer from 2020 to 2023.

---

Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:57 UTC from IEEE Xplore. Restrictions apply.
© 2025 IEEE. All rights reserved, including rights for text and data mining and training of artificial intelligence and similar technologies. Personal use is permitted,
but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.
