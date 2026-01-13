
IEEE SENSORS JOURNAL, VOL. 23, NO. 6, 15 MARCH 2023
6099

# RF-Based Drone Classification Under Complex Electromagnetic Environments Using Deep Learning

**Hanshuo Zhang**, **Tao Li**, **Yongzhao Li®**, *Senior Member, IEEE*, **Jinhui Li**,
**Octavia A. Dobre®**, *Fellow, IEEE*, and **Zhijin Wen**

***Abstract*—Recent studies have demonstrated that using deep-learning (DL) methods to classify drones based on the radio-frequency (RF) signal is effective. As known, the rich and diverse data is an important guarantee for good identification performance. In reality, due to the complexity and high dynamic of wireless environments, the costs of signal collection and labeling with sufficient diversity are often unacceptable. In this work, we propose a low-cost data augmentation (DA) method to improve the robustness of the signal identification neural network (NN). It generates extra training data by mixing pure drone signals with practical background signals and trains the NN to distinguish drone signals from interferences. Moreover, to further improve performance, we design a spectrogram segmentation (SS) method that directly splits the entire spectrogram into several subspectrograms to separate the interference signals working outside of the drone bandwidth. Finally, using five representative drone signals collected in multiple scenarios, we examine the identification accuracy and generalization capability of the proposed algorithm. Experimental results reveal that the proposed algorithm performs better than the NN trained without SS or DA.**

***Index Terms*—Complex electromagnetic environment, data augmentation (DA), drone detection and classification, radio-frequency (RF) signal, spectrogram.**

---

## I. INTRODUCTION

DRONES have been widely used in aerial photography, precision agriculture, rescue missions, package delivery, infrastructure inspections, and other fields. With the continuous expansion of the application scope, drones are expected to become important components of smart cities and the Internet of Things (IoT). However, some security and privacy issues caused by drones occur frequently, such as cyberattacks, drug trafficking, firearm smuggling, terrorist attacks, and invading security-sensitive infrastructures like nuclear reactors and airports,. Therefore, it is necessary to develop drone management and control technology. Accurate detection and classification of drones are critical to guarding against drones with security threats.

Existing drone detection and classification techniques can be divided into four categories, that is, radar-based, vision-based, audio-based, and communication-based technologies.

Radar has been commonly used for flying target detection based on the reflection waves from targets. Micro-Doppler features produced by propeller rotation and platform vibration can be used to detect and classify drones,. Relatively speaking, radar-based technologies have a higher cost in terms of equipment price and energy consumption. Besides, low-radar cross-section (RCS) values of consumer micro-drones limit the effectiveness of radar detection,. Benefiting from high-resolution cameras and machine vision technologies, vision-based drone recognition

*Manuscript received 4 January 2023; accepted 21 January 2023. Date of publication 10 February 2023; date of current version 14 March 2023. This work was supported in part by the National Natural Science Foundation of China under Grant 62001358; in part by the Fundamental Research Funds for the Central Universities under Grant XJS220116; in part by the Postdoctoral Science Foundation of China under Grant 2019M663630; in part by the Shaanxi Provincial Key Research and Development Program under Grant 2023-ZDLGY-33, Grant 2022ZDLGY05-04, Grant 2022ZDLGY05-03, and Grant 2021ZDLGY04-08; and in part by the State Key Laboratory of Integrated Services Network under Grant ISN090105. The work of Octavia A. Dobre was supported in part by the Natural Sciences and Engineering Research Council of Canada (NSERC) through its Discovery Program. The associate editor coordinating the review of this article and approving it for publication was Prof. Yulong Huang. (Corresponding authors: Tao Li; Yongzhao Li.)*
*Hanshuo Zhang, Tao Li, and Yongzhao Li are with the State Key Laboratory of Integrated Services Networks, Xidian University, Xi’an 710071, China (e-mail: hanshuozhang@stu.xidian.edu.cn; taoli@xidian.edu.cn; yzli@xidian.edu.cn).*
*Jinhui Li and Zhijin Wen are with the Laboratory of Electromagnetic Space Cognition and Intelligent Control, Beijing 10083, China.*
*Octavia A. Dobre is with the Faculty of Engineering and Applied Science, Memorial University, St. John’s, NL A1C 5S7, Canada (e-mail: odobre@mun.ca).*
*Digital Object Identifier 10.1109/JSEN.2023.3242985*

1558-1748 © 2023 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission.
See https://www.ieee.org/publications/rights/index.html for more information.

Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:49 UTC from IEEE Xplore. Restrictions apply.

---

6100
IEEE SENSORS JOURNAL, VOL. 23, NO. 6, 15 MARCH 2023

methods have gained increased attention. In, and, deep-learning (DL)-based object detection models were trained with artificial datasets to detect drones. However, the effectiveness of vision-based detection algorithms can only be guaranteed in the line-of-sight scenario. Drones with brushless motors produce sounds within a certain frequency range during flight. Audio-based detection methods analyze the sounds to determine the existence of drones,. Similar to the visual-based methods, the audio-based methods are easily affected by the environment. The noisy urban environment can disable the detection system.

Generally, a drone needs at least two communication links with its remote controller, including an uplink for flight control and a downlink for live video transmission. Due to the different circuit designs and modulation techniques adopted by drone manufacturers, the radio-frequency (RF) signatures of drones are distinguishable. Compared to the technologies based on radar, vision, and audio, drone detection and classification technologies based on RF signals can work in all kinds of weather and have stronger robustness. Thus, it can be regarded as an effective solution for drone detection and classification in urban environments.

Both uplink control signal and downlink video transmission signal can be utilized to detect and identify the drones, among which the schemes based on the control signal include literature,. In, a deep neural network (NN) was trained to identify the spectrums of control signals. A set of statistical features extracted from the energy transient of control signals were combined to detect and classify drones in. Zhao et al., proposed an auxiliary classifier Wasserstein generative adversarial network (AC-WGAN) to extract waveform fingerprints of control signals and classify drones. Generally, drones are operated in industrial, scientific, and medical (ISM) bands, where many other signals exist, such as Wi-Fi and Bluetooth signals. These signals are not considered in the above studies. Ezuma et al., incorporated Wi-Fi and Bluetooth into the target set and designed a hierarchical detector to identify the signals of drones, Wi-Fi devices, or Bluetooth devices. However, the algorithm works based on the assumption that only one signal exists in the received data. Besides, in urban environments, an RF-based drone surveillance system is usually deployed on the top of a building to surveil the flying drones above the buildings. On the contrary, the controller of the drone normally is located on the ground or much lower than the RF-based drone surveillance system. Thus, the control signals are normally blocked by the buildings and hardly received by the RF-based drone surveillance system.

On the other hand, the video transmission signals from hovering drones are more ready to be received. For the drones using Wi-Fi for controlling and video streaming, Alipour-Fanid et al. extracted the features of packet size and interarrival time of encrypted Wi-Fi traffic to detect drones. But many drones employ private protocols. Reus-Muns and Chowdhury and Soltani et al. aimed to classify drones by the time-domain sequences of video transmission signals. Reus-Muns and Chowdhury, leveraged the cross-correlation properties in the physical layer (PHY) preambles to classify drones. But the cross-correlation calculation increases with the number of preambles. Soltani et al. focused on subtle imperfections present in the drone waveform and designed an NN to learn them. However, co-channel interference may overwhelm these features in complex environments. In addition, using spectrum features to classify drones has aroused great interest for researchers. In, a multichannel NN was designed to learn the spectra of drone signals. However, the variability of the transmission channels of drones was not considered. Based on other transform domain analyses, Nie et al. proposed to extract fractal dimension (FD), axially integrated bispectra (AIB), and square integrated bispectra (SIB) as RF fingerprints to classify drones. These features were analyzed without introducing interference, and sufficient accuracy and robustness in practical deployments are hardly obtained.

Since drones can arbitrarily select transmission channels within ISM bands, a wideband receiver is needed to monitor peripheral signals. A wider receiving bandwidth means that the received data may contain more interference factors. The spectrogram can separate interference factors naturally and display the time-domain features and frequency-domain features simultaneously. Thus, it is a good choice for drone detection and classification. Prasad and Bhargava performed a sliding detection window across the spectrogram and extracted the histogram of ordered gradient (HOG) features to detect drones with Wi-Fi for controlling. The detection performance is difficult to guarantee in the case of spectral aliasing. Based on the you only look once (YOLO) architecture, Basak et al. proposed a combined detection and classification framework to identify drone signals. Basak et al. used a deep residual network to identify multiple drone signals based on the spectrogram. Because of the data-driven nature of DL, the abundant data collected under various scenarios is indispensable to training a robust NN. Yet, the cost of sufficient signal acquisition and labeling is unacceptable.

Data augmentation (DA) is a standard method to solve the problem of insufficient samples. Basak et al. and mixed drone signals with Gaussian noise at different scales to increase data diversity. Considering the influences of propagation channels, in, simulated channel and noise variations are utilized to process the original signal data to increase data diversity. Similarly, in, various time offsets, frequency offsets, and Gaussian noise are added to the original signal to increase data diversity. However, simply considering the signal change after passing through wireless channels cannot alleviate the adverse impact of interference signals.

To sum up, for classifying drones in complex electromagnetic environments, two major challenges remain to be solved. The first is that the electromagnetic environments are dynamic, and it is impracticable to collect complete training data. Another is that the drones and the interference signals, such as Wi-Fi devices, operate within the same ISM frequency band. Under this circumstance, the conventional NN-based classifiers degrade greatly. To address these two issues, we design a DA method to generate extra data that simulates drone

---

Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:49 UTC from IEEE Xplore. Restrictions apply.

---

**ZHANG et al.: RF-BASED DRONE CLASSIFICATION UNDER COMPLEX ELECTROMAGNETIC ENVIRONMENTS USING DL**
**6101**

signals transmitted in practical electromagnetic scenarios and a spectrogram segmentation (SS) method to reduce the adverse impact of interference signals on classification.

The main contributions of this article are as follows.

1) A low-cost DA is proposed to greatly reduce the cost of signal collection and labeling. Sufficient training data of drones in complex electromagnetic environments are generated by mixing different environment signals with pure drone signals. At a low cost of signal collection and labeling, the robustness and practicability of the proposed NN are achieved.
2) An SS method is proposed to mitigate the effect of the inferences on the NN. After the segmentation, for each subspectrogram, the number of interference signals that the NN needs to identify in one recognition greatly decreases compared with the entire spectrogram, that is, without segmentation. Therefore, the classification performance is improved. Moreover, the SS method makes the requirement of the NN relaxed, namely, lighter NNs might be applicable.
3) A modified lightweight version of the basic ResNet is proposed to further improve the classification efficiency. Specifically, we employ a smaller number of convolution kernels and layers. Besides, we supplement a Maxpooling layer after each residual block to downsample feature maps, which further reduces the subsequent calculation. Experiments verified the improvement of efficiency at a very low cost of accuracy.¹

The remainder of the article is organized as follows. A mathematical description of the received signal is introduced in Section II. The details of the drone classification algorithm are described in Section III. The experimental setup is presented in Section IV. Section V provides a detailed performance analysis. Concluding remarks are drawn in Section VI.

## II. PROBLEM DESCRIPTION

Fig. 1 illustrates a general scenario where an RF-based drone surveillance system is deployed in an urban area. The passive drone surveillance system listens for the signals in the monitored area. When a drone flies into this area, the surveillance system needs to recognize the drone timely according to the received signals with wireless interference. Affected by building occlusion, the RF surveillance system receives downlink video signals transmitted from the drone with a higher probability than uplink control signals. Thus, the received signal at time instant i can be expressed as follows:

$$
y(i) = \alpha x(i) + n(i) + \sum_{k=1}^K \omega_k(i), \quad i=1, 2, \dots, L \quad (1)
$$

where $x(i)$ is the drone signal, $\alpha$ represents the channel fading, $n(i)$ denotes environment noise, $\omega_k(i)$ corresponds to the kth interference signal, $K$ is the number of interference signals, and $L$ is the length of the received signal.

*Fig. 1. General scenario of the RF-based drone detection system in the urban area: (1) illegal drone, (2) passive RF surveillance system, (3) wireless interference sources, (4) drone remote control, (5) downlink video transmission signal, and (6) uplink control signal.*

The spectrogram is a better choice for detecting and classifying signals, because it can separate some interferences naturally and provide a more intuitive display for some signal features, such as center frequency, bandwidth, dwell time, and burst rate, than a power spectral density (PSD) or IQ representation. The received signal is reshaped to an $N \times M$ matrix Y as follows:

$$
\mathbf{Y} = \begin{bmatrix}
y(1) & y(N+1) & \dots & y(N(M-1)+1) \\
y(2) & y(N+2) & \dots & y(N(M-1)+2) \\
\vdots & \vdots & & \vdots \\
y(N) & y(2N) & \dots & y(NM)
\end{bmatrix}_{N \times M} \quad (2)
$$

where $NM=L$. The spectrum $\mathbf{s}_m$ of each column $\mathbf{y}_m$ of Y is calculated by discrete Fourier transform (DFT), that is,

$$
s_m(k) = \sum_{n=1}^N y_m(n)h(n)e^{-j\frac{2\pi}{N}nk}, \quad k=1,2,\dots,N \quad (3)
$$

where $h(\cdot)$ denotes a window function. Then, the spectrogram S of the received signal can be expressed as follows:

$$
\mathbf{S} = [|\mathbf{s}_1|^2, |\mathbf{s}_2|^2, \dots, |\mathbf{s}_M|^2]. \quad (4)
$$

Fig. 2 shows the spectrogram of the received signal including DJI Phantom 4 Pro V2.0 signal and interference signals. In the following, we aim to train an NN to extract the time-frequency features of drone signals with multiinterference and complete the signal classification tasks automatically.

## III. DRONE SIGNAL CLASSIFICATION ALGORITHM

In this section, we describe the crucial components of the proposed drone signal classification algorithm, including the architecture of the lightweight NN, the proposed DA method, and the SS method.

### A. Drone Signal Classification NN

1) *Architecture of the NN:* Without loss of generality, we adopt the basic ResNet as the signal classification NN.

¹We verify experimentally that the lightweight NN achieves a fourfold increase in identification speed at the cost of about 2% accuracy loss.

---

Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:49 UTC from IEEE Xplore. Restrictions apply.

---

6102
IEEE SENSORS JOURNAL, VOL. 23, NO. 6, 15 MARCH 2023

*Fig. 2. Spectrogram of a drone signal with interference.*

*Fig. 3. Architecture of the lightweight ResNet.*

On this basis, to improve the recognition speed, we modify it into a lightweight version. The architecture of the NN is depicted in Fig. 3.

Initially, the proposed version of ResNet contains two 2-D convolution layers using 3 × 3 kernels and the 2 × 2 stride to compress inputs. The following two ResBlocks are deployed for further feature extraction. In each ResBlock, there are two 2-D convolutional layers with N filters with size 3 × 3. The output of the ResBlock is generated by a skip-connection structure that splices the output maps of the last convolution layer and the input together. Then, a MaxPooling layer compresses the feature maps and feeds them into the next ResBlock for further feature extraction. After the last convolution layer, GlobalMaxPooling and one fully connected (FC) layer are responsible for generating the prediction vector. Finally, a softmax layer transforms the prediction result into a probability form.

2) *Loss Function and Training Program:* The NN is designed to identify the presence and type of drone in one step. It is a typical multiclass classification problem. Therefore, we use the categorical cross-entropy loss, which can be expressed as follows:

$$
\mathcal{L} = -\frac{1}{N_b} \sum_{b=1}^{N_b} \sum_{c=1}^{N_c} l_{b,c} \log p_{b,c} \quad (5)
$$

where $N_b$ is the batch size during the NN training stage, and $l_{b,c}$ and $p_{b,c}$ represent the label and prediction probability of the bth sample belonging to the cth type, respectively. The Adam optimizer is utilized for training the NN. In addition, we use the data generator in the Keras library to prepare the training batches. The data generator allows us to add various electromagnetic interference to the same drone sample in different batches according to the proposed DA.

### B. Data Augmentation

DA is commonly used to increase the diversity of samples and avoid NNs focusing on irrelevant features that could lead to overfitting. Popular DA methods for images include flipping, rotating, scaling, or adding noise, which simulates the changes in images caused by the camera angle and other factors. However, these methods cannot be directly applied to communication signals. The time-frequency distribution of communication signals follows fixed formats and does not change with the spatial position of the cooperative receiver and transmitter. Flipping, rotating, and scaling spectrograms destroy the original signal structure and make the new instance useless. Therefore, DA for communication signals should follow the wireless communication principle.

For the drone video transmission signals in different electromagnetic scenes, the additive mixing of drone signals, noise, and interference occurs. Theoretically, if the sample set covers all signal transmission scenarios, the sufficient diversity of the training data is good for the NN to focus on the more robust characteristics more easily and obtain better robustness.

In this work, there are two kinds of data used to augment the training data. One is the background signals collected outdoors, called background samples, and the other is generated by MATLAB. During the training stage, the data generator randomly selects $N_b$ samples and mixes them with additive white Gaussian noise (AWGN) and background samples to produce a training batch. In each training epoch, the same drone sample will be mixed with a different interference. To increase the diversity of the background interference, we perform random spectral shifting and fading for background samples. Besides, to ensure the validity of training data, we need to quantize the signal-to-noise ratio (SNR) or signal-to-interference-plus-noise ratio (SINR) of training samples. For the data without interference, we calculate the SNR within the bandwidth of a drone signal based on PSD, as shown in Fig. 4, and the SNR $\gamma$ can be evaluated as follows:

$$
\gamma = \frac{\int_{B_s} S(x) - N(x) dx}{\int_{B_s} N(x) dx} \quad (6)
$$

where $S(x)$ and $B_s$ are the PSD and the transmission bandwidth of the drone signal, respectively, and $N(x)$ is the PSD of the noise floor. Furthermore, we directly calculate the mean value of the spectrogram $S$ of each background sample as the

---

Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:49 UTC from IEEE Xplore. Restrictions apply.

---

**ZHANG et al.: RF-BASED DRONE CLASSIFICATION UNDER COMPLEX ELECTROMAGNETIC ENVIRONMENTS USING DL**
**6103**

*Fig. 4. PSD of the drone signal collected in the lab.*

intensity of interference. Then, the augmented data $x'(i)$ of a sample $x(i)$ can be expressed as follows:

$$
x'(i) = x(i) + \beta B(i) e^{j\frac{2\pi \Delta f i}{f_s}} + n(i), \quad i=1,2,\dots,L \quad (7)
$$

where $B(i)$ is a background sample, $\beta$ is the weight coefficient, $\Delta f$ and $f_s$ represent the random frequency bias and the system sampling rate, respectively, and $n(i)$ is AWGN.

### C. Spectrogram Segmentation

In this work, the receiving bandwidth is much larger than the bandwidth of drone signals and only a partial spectrogram contains target signals. Therefore, we design SS to divide the spectrogram S into several subspectrograms A horizontally and vertically according to scheduled rules. Specifically, the spectrogram is divided with the length $L_f$ along the frequency-domain axis according to the drone’s operation channels. Based on our preliminary investigation and signal analysis, we found that the operation channels of most small drones follow 802.11 protocols, which provides a good guideline for spectral segmentation. And details will be introduced in Section IV. In addition, we divide the spectrogram along the time-domain axis with length $L_t$ to generate all subspectrograms. Then, the (i, j)th subspectrogram can be expressed as a matrix, that is,

$$
\mathbf{A}^{i,j} = \mathbf{S}_{n \in \Psi_i, m \in \Theta_j}, \quad i=1,2,\dots,\frac{N}{L_f}, \quad j=1,2,\dots,\frac{M}{L_t} \quad (8)
$$

where $\Psi_i$ denote the index set $\{ n \in \mathbb{Z} : (i-1)L_f+1 \le n \le iL_f \}$, $\Theta_j$ is the index set $\{ m \in \mathbb{Z} : (j-1)L_t+1 \le m \le jL_t \}$, and $\mathbb{Z}$ is the set of integers.

Considering the convergence in NN training, each subspectrogram A needs to be normalized as follows:

$$
\hat{\mathbf{A}} = \frac{\mathbf{A} - \mu_A}{\sigma_A} \quad (9)
$$

where $\mu_A$ and $\sigma_A$ are the mean and standard deviation of A, respectively.

Each subspectrogram is fed into the NN, and the NN sequentially identifies the presence and type of the signal in each subspectrogram. We suppose that drones do not change video transmission channels within the observation duration. Thus, the prediction results for all A from the same frequency bin are aggregated based on soft voting, that is,

$$
\mathbf{P}_i = \frac{1}{J} \sum_j \mathbf{P}_{i,j} \quad (10)
$$

*Fig. 5. Signal collection scenarios: (a) laboratory environment without interference and (b) urban area with multiinterference.*

where $\mathbf{P}_{i,j}$ is the prediction vector corresponding to the subspectrogram $\mathbf{A}^{i,j}$, and $J$ is the number of subspectrograms in the same frequency bin and $J = (M/L_t)$. We select the max value and the corresponding index of each prediction vector $\mathbf{P}_i$ as the type identification result in this frequency bin. Furthermore, we set a threshold to screen the results, and the threshold is set to 0.5 according to the experience. If the max value exceeds the threshold and the type is a certain drone, the drone is considered to be existing in that frequency bin.

## IV. EXPERIMENTS

In this section, we first introduce the experimental setup and the RF characteristics of experimental drones. Then, we describe the process of drone signal collection in two different environments and how we develop datasets.

### A. Experimental Setup

We use the devices including a universal software radio peripheral (USRP) X310 for radio-signal acquisition and a workstation to store the raw RF signals of drones, as shown in Fig. 5. The 5.8-GHz band is set up as the target monitoring band. Certainly, it can also be directly applied to the 2.4-GHz band. According to the operating frequency ranges of drones, the sampling rate, center frequency, and receiving bandwidth of USRP X310 are set to 100 Msa/s, 5785 MHz, and 100 MHz, respectively.

Five representative drone types are selected as the monitor targets, including Parrot ANAFI, FIMI X8SE, DJI Phantom 4 Pro V2.0, DJI Mavic Air, and DJI Mavic Mini 2. The specifications and RF parameters of these drones are listed in Table I, which guides us in setting the algorithm parameters.

---

Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:49 UTC from IEEE Xplore. Restrictions apply.

---

6104
IEEE SENSORS JOURNAL, VOL. 23, NO. 6, 15 MARCH 2023

**TABLE I**
RF PARAMETER ESTIMATION OF FIVE DRONE SIGNALS

| Source             | Protocol       | Bandwidth | Burst Duration | Burst Interval |
| :----------------- | :------------- | :-------- | :------------- | :------------- |
| Parrot ANAFI       | Wi-Fi          | 20 MHz    | 0.03~0.4 ms    | 0.06~23 ms     |
| FIMI X8SE          | TDMA           | 10 MHz    | 0.18~2.1 ms    | 0.05~4 ms      |
| Phantom 4 Pro V2.0 | OcuSync 2      | 10/20 MHz | 1~5 ms         | 1~2 ms         |
| Mavic Air          | Enhanced Wi-Fi | 5 MHz     | 0.2~1.5 ms     | 0.3~18 ms      |
| Mavic Mini 2       | OcuSync 2      | 10/20 MHz | 1 ms           | 2 ms           |

### B. Signal Collection and Dataset Development

The drone signals are recorded in two different electromagnetic environments, including a laboratory environment and an urban area, as shown in Fig. 5. There is little interference around the laboratory. Hence, the signals are recorded with a high level of SNR. Nevertheless, there are multiple interference signals in the urban area due to Wi-Fi devices and other signal sources. Considering the RF characteristics of experimental drones, we set the length of each record to $L=2^{21}$, which is approximately equal to 20 ms. The length ensures that at least four bursts are present within one record,² which is appropriate for the NN model to learn signal burst characteristics.

To calculate spectrograms, the length of DFT is set to N = 1024. Then, the size of the spectrogram of each record is N × M = 1024 × 2048. Referring to the Wi-Fi channel partition criterion in IEEE 802.11ac, the frequency-domain segmentation parameter $L_f$ of SS is set as 205, that is, the spectrogram is segmented into five parts along the frequency-domain axis. Without loss of generality, we do not divide spectrograms along the time-domain axis. Thus, five subspectrograms with the size of 205 × 2048 are generated from the spectrogram.

Based on the data collected in the laboratory and pure background electromagnetic data in the urban environment, we develop datasets according to the above method to test the accuracy of detection and classification and the anti-interference capability. The background data is mixed with pure drone signals based on DA by the data generator to increase data diversity. In each training batch, the background data experiences random spectral shifting within [-40, 40] MHz. The SNR or SINR of training samples is set between dB by adjusting the weight coefficient $\beta$ and the power of AWGN. Based on the data collected in the previously mentioned environments, a series of simulations are conducted to verify the effectiveness and generalization of the proposed algorithm.

## V. PERFORMANCE ANALYSIS

### A. Effect of DA

The contributions of the proposed DA to the NN are shown in this part from the perspectives of function realization and recognition accuracy, respectively.
The proposed DA aims to help the NN focus on and identify the target signal with severe spectrum interference. We visualize the sensitive regions of input samples corresponding to predicting results with the help of a visual analysis tool, gradient-weighted class activation mapping (Grad-CAM). Grad-CAM produces a coarse localization map highlighting the crucial regions in the image for predicting the concept. Fig. 6 illustrates the distribution of heat maps corresponding to the identification results of five different samples with interference, in which the redder part means that the data in that part has a more significant contribution to the recognition result. In Fig. 6(a) and (d), it is evident that the NN is interested in all bursts of video transmission signals of the Parrot ANAFI and the DJI Mavic Air. Even if some bursts of Mavic Air signal are completely suppressed by a strong interference, other bursts are still effectively captured by the NN. Fig. 6(b) and (c) demonstrates that the NN is more sensitive to the bandwidth edges of each burst of the signals from the DJI Mavic Mini 2 and FIMI X8SE. For the signal from the DJI Phantom 4 Pro V2.0, the NN is more interested in the beginning edge of each burst, as shown in Fig. 6(e). From the above results, it can be seen that for different target signals, the NN trained with the proposed DA can always locate the target signal without being affected by co-channel interference.

To further demonstrate the gains of the proposed DA, we made the performance comparison between the proposed DA and three peer DA methods under the Gaussian noise and complex electromagnetic environments, respectively. In the Gaussian noise environment without interference, the Gaussian DA,, the DA of, and the DA of outperform the proposed DA, especially in the region of low SNR, as shown in Fig. 7. The main reason is that, except for the proposed DA, these three peer DA methods keep the distribution of the training set consistent with that of the testing set of the Gaussian environment. With an increase in SNR, the characteristics of drone signals are gradually obvious in the spectrogram, and the accuracies of the four DA methods tend to be the same. When SNR ≥ 4 dB, the accuracy of the proposed DA method can exceed 95%. Empirically, this performance is acceptable. Furthermore, false identification of the NN trained with the proposed DA at low SNRs mainly comes from missed detection, as shown in Fig. 8, which can be mitigated by performing multiple reconnaissance.

On the contrary, the proposed DA method outperforms the three peer methods under the complex electromagnetic environment, as shown in Fig. 9. Since no knowledge/information of the environment is contained in the preprocessed training

²It should be noted that, for Parrot ANAFI and Mavic Air, the largest burst intervals are 23 and 18 ms, respectively. Nevertheless, in most cases, the intervals will be 0.06 and 0.3 ms. Therefore, the value of 20 ms for $L_t$ is also applicable to these two types of drones.

---

Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:49 UTC from IEEE Xplore. Restrictions apply.

---

**ZHANG et al.: RF-BASED DRONE CLASSIFICATION UNDER COMPLEX ELECTROMAGNETIC ENVIRONMENTS USING DL**
**6105**

*Fig. 6. Grad-CAM for five drones: (a) Parrot, (b) Mavic Mini 2, (c) FIMI, (d) Mavic Air, and (e) Phantom 4 Pro, and the drone signals are marked by blue boxes.*

data for the peer DA methods, the NN is unable to reduce the adverse effects of background signals on the recognition process. Even when the SINR = 20 dB, the NN trained with Gaussian DA only reaches an accuracy of 85%, and the NNs trained with the DA methods of and can reach an accuracy of 90%. In contrast, when the SINR ≥ 10 dB, the NN with the proposed DA has already achieved an accuracy of 90%. Take the Gaussian DA, for instance. Fig. 10 presents the confusion matrix of the NN trained with Gaussian DA in the
*Fig. 7. Accuracy versus SNR under the Gaussian noise condition for DA methods.*

*Fig. 8. Confusion matrix of the NN trained with the proposed DA, testing under the Gaussian noise environment with SNR = 0 dB.*

*Fig. 9. Accuracy versus SINR under the complex electromagnetic environment for DA methods.*

complex electromagnetic environment with SINR = 14 dB. There is a high rate of error identification. It indicates that when the NN lacks information on background signals, the NN is prone to misjudgment due to the influence of background signals. The proposed DA method can train the NN to learn the difference between background and drone signals. Therefore, it achieves better performance. On the whole, the proposed DA can adapt to the Gaussian environment and meanwhile has superiority in the complex electromagnetic environment.

### B. Effect of SS

As explained in Section III, SS is designed to separate the noncochannel interference, making it easier for the NN to focus on the target signals and then improve the recognition

---

Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:49 UTC from IEEE Xplore. Restrictions apply.

---

6106
IEEE SENSORS JOURNAL, VOL. 23, NO. 6, 15 MARCH 2023

*Fig. 10. Confusion matrix of the NN trained with Gaussian DA, testing under the complex electromagnetic environment with SINR = 14 dB.*

*Fig. 11. Training accuracy comparison between the NNs trained with and without the SS.*

*Fig. 12. Accuracy versus SINR in the complex electromagnetic environment for the NNs trained with and without the SS.*

performance. Using the drone signal data mixed with background samples, we first compare the NN trained with SS and that trained by the entire spectrograms.

Fig. 11 shows the training accuracy and validation accuracy of the NNs trained with and without SS. After 50 training epochs, the NNs in both cases converge to the level of 90% accuracy. It is worth noting that the NN trained with SS has a more stable convergence and higher accuracy. The accuracy gap may come from misclassification at a low level of SINR. To express the gain of SS more clearly, we compare the recognition accuracies of the two methods versus the SINR, as shown in Fig. 12. The accuracy of the NN trained with SS is better, especially in the case of a low level of SINR. When the SINR = 0 dB, the NN trained with SS can reach 65% accuracy, whereas the NN trained by the entire spectrograms only reaches the accuracy of 55%. The accuracies of both NNs become similar as the SINR increases. When SINR ≥ 15 dB, both NNs reach a recognition accuracy of 95%.

To further clarify the identification performance, we compare the cross-misjudgment probabilities of both NNs. The confusion matrices of both NNs in the condition of SINR = 0 dB are shown in Fig. 13. At the low SINR level, the NN trained with SS has serious missing detection because of severe spectrum interference, while the cross-misjudgment probability between drones is low. However, the NN trained by the entire spectrograms has a severe cross-misjudgment probability. Regardless of the presence of a drone, any input could be recognized as Mavic Air with a high probability. It is worth noting that the condition without any drone may be identified as a Mavic Air with the probability of 51.71%, because bandwidth-similar signals in the background cause a bias during the training stage. Using SS, the misleading interferences can be separated from the target signal possibly

*Fig. 13. Confusion matrices with SINR = 0 dB with: (a) SS and (b) entire spectrogram.*

---

Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:49 UTC from IEEE Xplore. Restrictions apply.

---

**ZHANG et al.: RF-BASED DRONE CLASSIFICATION UNDER COMPLEX ELECTROMAGNETIC ENVIRONMENTS USING DL**
**6107**

and we tag them with the nondrone label to avoid training bias.

Another benefit of SS is that it helps the algorithm obtain multitarget detection ability. Since each subspectrogram is recognized independently by the NN, the proposed algorithm can recognize multiple drones operating in different channels simultaneously without additional operations.

## VI. CONCLUSION

This work has investigated the problem of detecting and classifying drones through downlink video transmission signals in complex electromagnetic environments. To improve the robustness of the signal classification NN, we have proposed two low-cost methods, including SS to eliminate noncochannel interference, and DA to guide the NN to identify target signals from interference signals without introducing extra costs of collecting signals and labeling. Through multiple experiments, the effectiveness of the method has been verified. The classification task is for different drone platforms with different protocols. In future work, we will investigate the classification of different drones with the same platform.

## REFERENCES

 H. Shakhatreh et al., “Unmanned aerial vehicles (UAVs): A survey on civil applications and key research challenges,” *IEEE Access*, vol. 7, pp. 48572–48634, 2019.
 H. Menouar, I. Guvenc, K. Akkaya, A. S. Uluagac, A. Kadri, and A. Tuncer, “UAV-enabled intelligent transportation systems for the smart city: Applications and challenges,” *IEEE Commun. Mag.*, vol. 55, no. 3, pp. 22–28, Mar. 2017.
 X. Xu, H. Zhao, H. Yao, and S. Wang, “A blockchain-enabled energy-efficient data collection system for UAV-assisted IoT,” *IEEE Internet Things J.*, vol. 8, no. 4, pp. 2431–2443, Feb. 2021.
 A. A. A. Alajmi, A. Vulpe, and O. Fratu, “UAVs for Wi-Fi receiver mapping and packet sniffing with antenna radiation pattern diversity,” *Wireless Pers. Commun.*, vol. 92, no. 1, pp. 297–313, Jan. 2017.
 U. Seidaliyeva, M. Alduraibi, L. Ilipbayeva, and N. Smailov, “Deep residual neural network-based classification of loaded and unloaded UAV images,” in *Proc. 4th IEEE Int. Conf. Robotic Comput. (IRC)*, Nov. 2020, pp. 465–469.
 M. Hutter and R. Scurek, “Possibilities of misuse of unmanned aerial vehicles (UAV) to terrorist targets,” *Prace Naukowe Akademii Im. Jana Długosza W Częstochowie. Technika, Informatyka, Inżynieria Bezpieczeństwa*, vol. 4, pp. 195–202, Jun. 2016.
 A. Levin. (2017). *FAA Warns of Drone Collision Risks With Airplanes*. [Online]. Available: https://www.bloomberg.com/news/articles/2017-11-28/faa-warns-of-drone-collision-risks-with-airplanes-as-use-grows
 P. Nguyen, M. Ravindranatha, A. Nguyen, R. Han, and T. Vu, “Investigating cost-effective RF-based detection of drones,” in *Proc. 2nd Workshop Micro Aerial Vehicle Netw., Syst., Appl. Civilian Use*, Jun. 2016, pp. 12–22.
 G. Birch, J. Griffin, and M. Erdman. (Jul. 2015). *UAS Detection Classification and Neutralization: Market Survey*. [Online]. Available: https://www.osti.gov/biblio/1222445
 M. Messina and G. Pinelli, “Classification of drones with a surveillance radar signal,” in *Proc. Int. Conf. Comput. Vis. Syst. (ICCVS)*, Sep. 2019, pp. 723–733.
 M. Ezuma, O. Ozdemir, C. K. Anjinappa, W. A. Gulzar, and I. Guvenc, “Micro-UAV detection with a low-grazing angle millimeter wave radar,” in *Proc. IEEE Radio Wireless Symp. (RWS)*, Jan. 2019, pp. 1–4.
 P. Zhang, L. Yang, G. Chen, and G. Li, “Classification of drones based on micro-Doppler signatures with dual-band radar sensors,” in *Proc. Prog. Electromagn. Res. Symp.–Fall*, Nov. 2017, pp. 638–643.
 B. K. Kim, H.-S. Kang, and S.-O. Park, “Drone classification using convolutional neural networks with merged Doppler images,” *IEEE Geosci. Remote Sens. Lett.*, vol. 14, no. 1, pp. 38–42, Jan. 2017.
 B.-S. Oh, X. Guo, F. Wan, K.-A. Toh, and Z. Lin, “Micro-Doppler mini-UAV classification using empirical-mode decomposition features,” *IEEE Geosci. Remote Sens. Lett.*, vol. 15, no. 2, pp. 227–231, Feb. 2018.
 C. J. Li and H. Ling, “An investigation on the radar signatures of small consumer drones,” *IEEE Antennas Wireless Propag. Lett.*, vol. 16, pp. 649–652, 2017.
 A. D. De Quevedo, F. I. Urzaiz, J. G. Menoyo, and A. A. Lopez, “Drone detection and RCS measurements with ubiquitous radar,” in *Proc. Int. Conf. Radar (RADAR)*, Aug. 2018, pp. 1–6.
 C. Aker and S. Kalkan, “Using deep networks for drone detection,” in *Proc. 14th IEEE Int. Conf. Adv. Video Signal Based Surveill. (AVSS)*, Aug. 2017, pp. 1–6.
 Z. Zhang, Y. Cao, M. Ding, L. Zhuang, and W. Yao, “An intruder detection algorithm for vision based sense and avoid system,” in *Proc. Int. Conf. Unmanned Aircraft Syst. (ICUAS)*, Jun. 2016, pp. 550–556.
 J. Peng, C. Zheng, T. Cui, Y. Cheng, and L. Si, “Using images rendered by PBRT to train faster R-CNN for UAV detection,” in *Proc. CSRN*, Jan. 2018, pp. 13–18.
 S. Vishwakarma and S. S. Ram, “Classification of multiple targets based on disaggregation of micro-Doppler signatures,” in *Proc. Asia–Pacific Microw. Conf. (APMC)*, Dec. 2016, pp. 1–4.
 J. Mezei, V. Fiaska, and A. Molnar, “Drone sound detection,” in *Proc. 16th IEEE Int. Symp. Comput. Intell. Informat. (CINTI)*, Nov. 2015, pp. 333–338.
 A. Bernardini, F. Mangiatordi, E. Pallotti, and L. Capodiferro, “Drone detection by acoustic signature identification,” *Electron. Imag.*, vol. 2017, no. 10, pp. 60–64, 2017.
 J. Busset et al., “Detection and tracking of drones using advanced acoustic cameras,” *Proc. SPIE*, vol. 9647, Oct. 2015, Art. no. 96470F.
 H. Liu, Z. Wei, Y. Chen, J. Pan, L. Lin, and Y. Ren, “Drone detection based on an audio-assisted camera array,” in *Proc. IEEE 3rd Int. Conf. Multimedia Big Data (BigMM)*, Apr. 2017, pp. 402–406.
 Y. Xiao and X. Zhang, “Micro-UAV detection and identification based on radio frequency signature,” in *Proc. 7th Int. Conf. Syst. Inf. (ICSAI)*, Nov. 2019, pp. 1056–1062.
 M. F. Al-Sa’d, A. Al-Ali, A. Mohamed, T. Khattab, and A. Erbad, “RF-based drone detection and identification using deep learning approaches: An initiative towards a large open source drone database,” *Future Gener. Comput. Syst.*, vol. 100, pp. 86–97, Nov. 2019.
 M. Ezuma, F. Erden, C. K. Anjinappa, O. Ozdemir, and I. Guvenc, “Micro-UAV detection and classification from RF fingerprints using machine learning techniques,” in *Proc. IEEE Aerosp. Conf.*, Mar. 2019, pp. 1–13.
 C. Zhao, C. Chen, Z. Cai, M. Shi, X. Du, and M. Guizani, “Classification of small UAVs based on auxiliary classifier Wasserstein GANs,” in *Proc. IEEE Global Commun. Conf. (GLOBECOM)*, Dec. 2018, pp. 206–212.
 M. Ezuma, F. Erden, C. Kumar Anjinappa, O. Ozdemir, and I. Guvenc, “Detection and classification of UAVs using RF fingerprints in the presence of Wi-Fi and Bluetooth interference,” *IEEE Open J. Commun. Soc.*, vol. 1, pp. 60–76, 2020.
 A. Alipour-Fanid, M. Dabaghchian, N. Wang, P. Wang, L. Zhao, and K. Zeng, “Machine learning-based delay-aware UAV detection and operation mode identification over encrypted Wi-Fi traffic,” *IEEE Trans. Inf. Forensics Security*, vol. 15, pp. 2346–2360, 2020.
 G. Reus-Muns and K. Chowdhury, “Classifying UAVs with proprietary waveforms via preamble feature extraction and federated learning,” *IEEE Trans. Veh. Technol.*, vol. 70, no. 7, pp. 6279–6290, Jul. 2021.
 N. Soltani, G. Reus-Muns, B. Salehi, J. Dy, S. Ioannidis, and K. Chowdhury, “RF fingerprinting unmanned aerial vehicles with non-standard transmitter waveforms,” *IEEE Trans. Veh. Technol.*, vol. 69, no. 12, pp. 15518–15531, Dec. 2020.
 S. Yang, Y. Luo, W. Miao, C. Ge, W. Sun, and C. Luo, “RF signal-based UAV detection and mode classification: A joint feature engineering generator and multi-channel deep neural network approach,” *Entropy*, vol. 23, no. 12, p. 1678, Dec. 2021.
 W. Nie, Z.-C. Han, M. Zhou, L.-B. Xie, and Q. Jiang, “UAV detection and identification based on WiFi signal and RF fingerprint,” *IEEE Sensors J.*, vol. 21, no. 12, pp. 13540–13550, Jun. 2021.
 K. N. R. S. V. Prasad and V. K. Bhargava, “A classification algorithm for blind UAV detection in wideband RF systems,” in *Proc. IEEE 92nd Veh. Technol. Conf. (VTC-Fall)*, Nov. 2020, pp. 1–7.
 S. Basak, S. Rajendran, S. Pollin, and B. Scheers, “Combined RF-based drone detection and classification,” *IEEE Trans. Cognit. Commun. Netw.*, vol. 8, no. 1, pp. 111–120, Mar. 2022.

---

Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:49 UTC from IEEE Xplore. Restrictions apply.

---

6108
IEEE SENSORS JOURNAL, VOL. 23, NO. 6, 15 MARCH 2023

 S. Basak, S. Rajendran, S. Pollin, and B. Scheers, “Drone classification from RF fingerprints using deep residual nets,” in *Proc. Int. Conf. Commun. Syst. Netw. (COMSNETS)*, Jan. 2021, pp. 548–555.
 N. Soltani, K. Sankhe, J. Dy, S. Ioannidis, and K. Chowdhury, “More is better: Data augmentation for channel-resilient RF fingerprinting,” *IEEE Commun. Mag.*, vol. 58, no. 10, pp. 66–72, Oct. 2020.
 R. D. Miller, S. Kokalj-Filipovic, G. Vanhoy, and J. Morman, “Policy based synthesis: Data generation and augmentation methods for RF machine learning,” in *Proc. IEEE Global Conf. Signal Inf. Process. (GlobalSIP)*, Nov. 2019, pp. 1–5.
 K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit.*, Aug. 2016, pp. 770–778.
 J. C. Duchi, E. Hazan, and Y. Singer, “Adaptive subgradient methods for online learning and stochastic optimization,” in *Proc. Conf. Learn. Theory (COLT)*, Jun. 2010, pp. 27–29.
 *IEEE Standard for Information Technology Telecommunications and Information Exchange Between Systemslocal and Metropolitan Area Networks-Specific Requirements-Part 11: Wireless LAN Medium Access Control (MAC) and Physical Layer (PHY) Specifications-Amendment 4: Enhancements for Very High Throughput for Operation in Bands Below 6 GHz, Standard 802.11ac-2013*, 2013, pp. 1–425.
 R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, “Grad-CAM: Visual explanations from deep networks via gradient-based localization,” in *Proc. IEEE Int. Conf. Comput. Vis. (ICCV)*, Oct. 2017, pp. 618–626.

**Yongzhao Li** (Senior Member, IEEE) received the B.S., M.S., and Ph.D. degrees in electronic engineering from Xidian University, Xi’an, China, in 1996, 2001, and 2005, respectively.
In 1996, he joined Xidian University. He worked as a Research Professor with the University of Delaware, Newark, DE, USA, from 2007 to 2008, and the University of Bristol, Bristol, U.K., in 2011. He is currently a Full Professor with the State Key Laboratory of Integrated Services Networks, Xidian University. He has published more than 70 journal articles and 30 conference papers. His research interests include wideband wireless communications, signal processing for communications, and spatial communications networks.
Dr. Li received the Best Paper Award of IEEE CHINACOM 2008 International Conference, in 2008. Due to his excellent contributions to education and research, in 2012, he was awarded by the Program for New Century Excellent Talents in University, Ministry of Education, China.

**Jinhui Li** received the B.S. degree from PLA Information Engineering University, Zhengzhou, China, in 2006, and the M.S. degree from PLA Electronic Engineering Institute, Hefei, China, in 2010.

**Hanshuo Zhang** received the B.S. degree in communication engineering from Xidian University, Xi’an, China, in 2017, where he is currently pursuing the Ph.D. degree in information and communication engineering. His current research interests include radio-frequency identification and blind signal processing.

**Tao Li** received the B.S. and Ph.D. degrees in communication engineering from Xidian University, Xi’an, China, in 2012 and 2018, respectively.
From September 2015 to September 2016, he worked as a Visiting Ph.D. Student, under the supervision of Prof. Leonard J. Cimini, Jr., with the University of Delaware, Newark, DE, USA. Since July 2018, he has been a Lecturer with the School of Telecommunications Engineering, Xidian University. His current research interests are in the fields of blind signal processing, random matrix theory, physical layer security, and antidrone technique.

**Octavia A. Dobre** (Fellow, IEEE) received the Dipl.Ing. and Ph.D. degrees from the Polytechnic Institute of Bucharest, Bucharest, Romania, in 1991 and 2000, respectively.
From 2002 to 2005, she was with the New Jersey Institute of Technology, Newark, NJ, USA. In 2005, she joined Memorial University, St. John’s, NL, Canada, where she is currently a Professor and the Research Chair. She was a Visiting Professor with the Massachusetts Institute of Technology, Cambridge, MA, USA, and Université de Bretagne Occidentale, Brest, France. She has coauthored over 400 refereed articles in these areas. Her research interests encompass various wireless technologies, such as nonorthogonal multiple access and full duplex, optical and underwater communications, and machine learning for communications.
Dr. Dobre is currently a Fellow of the Engineering Institute of Canada and a Fellow of the Canadian Academy of Engineering. She was a Royal Society Scholar, a Fulbright Scholar, and a Distinguished Lecturer of the IEEE Communications Society. She received the best paper awards at various conferences, including the IEEE International Conference on Communications, the IEEE Global Communications Conference, the IEEE Wireless Communications and Networking Conference, and the IEEE International Symposium on Personal, Indoor and Mobile Radio Communications. She served as the General Chair, the Technical Program Co-Chair, the Tutorial Co-Chair, and the Technical Co-Chair for symposia at numerous conferences. She also served as the Editor-in-Chief (EiC) for the IEEE OPEN JOURNAL OF THE COMMUNICATIONS SOCIETY and the EiC for the IEEE COMMUNICATIONS LETTERS, a senior editor, an editor, and a guest editor for various prestigious journals and magazines. She is currently the Director of Journals of the IEEE COMMUNICATIONS SOCIETY.

**Zhijin Wen** received the Ph.D. degree from the Beijing Institute of Technology, Beijing, China, in 2013.

---

Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:49 UTC from IEEE Xplore. Restrictions apply.
