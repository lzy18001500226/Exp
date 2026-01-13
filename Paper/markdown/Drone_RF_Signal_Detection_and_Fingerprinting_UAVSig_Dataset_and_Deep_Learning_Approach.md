MILCOM 2024 Track 3 - Cyber Security and Trusted Computing

# Drone RF Signal Detection and Fingerprinting: UAVSig Dataset and Deep Learning Approach

**Tianyi Zhao\***, **Benjamin W. Domae\***, **Connor Steigerwald\***,
**Luke B. Paradis†**, **Tim Chabuk†** and **Danijela Cabric\***
\*Electrical and Computer Engineering Department, University of California, Los Angeles, USA
†Perceptronics Solutions, Inc.
Email: zhaotianyi@ucla.edu, bdomae@ucla.edu, csteigerwald1@ucla.edu,
lukep@percsolutions.com, timc@percsolutions.com, danijela@ee.ucla.edu

---

***Abstract*—Unmanned aerial vehicles (UAVs) are useful for commercial, recreational, and military applications, but they can also be used by malicious attackers and pose security threats. Therefore, it is important to monitor their occurrences and ensure their compliance. Radio frequency signals can be leveraged for such task. However, commercial UAVs often adopt the frequency hopping spread spectrum signals and proprietary protocols, which makes them more challenging to detect. While prior works have studied the UAV detection and classification problem, most of them consider the classification between different UAV models. Classification between drones of the same model has not yet fully been investigated. In this work, we consider the tasks of detecting and localizing UAV signals in a wideband spectrum, and also the task of fingerprinting these drones of the same model simultaneously. To solve this problem, we collect the UAVSig, an over-the-air dataset of UAV RF signals. We also present a deep learning model which can solve detect and fingerprint multiple drones of the same model simultaneously based on spectrograms. Our model can achieve 99.5% precision and 99.4% recall for transmission detection, and 90.9% classification accuracy with drones of the same model.**

***Index Terms*—Radio frequency fingerprinting, deep learning, unmanned aerial vehicles, UAV, dataset, YOLO**

---

## I. INTRODUCTION

### A. Motivation

Unmanned aerial vehicles (UAVs) have been used in various scenarios including military, surveillance and monitoring. Many other potential applications, such as flying wireless base stations, are also of great potential. However, the flexibility and mobility of UAVs can also pose threats to people and property, and thus it is important to develop technologies for drone detection and classification. Existing drone detection and classification techniques are based on four types of signals: radar, acoustic, visual, and radio frequency (RF). Among these methods, RF-based techniques have many advantages, such as being able to work in all kinds of weather and operate in non-line-of-sight (NLoS) scenarios.

Using the RF signals, the monitoring system would need to detect, localize and fingerprint multiple UAVs in a wideband spectrum. The detection algorithm needs to find occurrences of the UAV transmissions in the spectrum, localize the time and frequency parameters of those transmissions, and classify their source transmitters based on RF fingerprinting. Here, fingerprinting means distinguishing between different UAVs of the same model. The fingerprinting problem is challenging because UAVs of the same model use the same protocols and many commercial UAVs transmit proprietary waveforms. As a result, the detector might have limited or no knowledge of the protocols and transmitted data, and therefore in order to differentiate between them must rely on the raw I/Q samples. Commercial UAVs usually adopt frequency hopping spread spectrum (FHSS) physical layer signals. Therefore, localization of the UAV signals in a wideband spectrum is also a major challenge for UAV detection.

To investigate this problem, we need an appropriate UAV RF dataset. We surveyed existing public UAV RF datasets–, which were collected for different purposes, such as transmission type classification, UAV operational mode classification, etc. However, those datasets have several limitations as listed in Table I. Synthetic data can be used for training but cannot always represent real-world scenarios. Consequently, a model trained on synthetic data can have degraded performance on real signals captured over-the-air (OTA). Therefore, we collect a comprehensive set of OTA RF signals from the UAVs of the same model and make, referred as UAVSig dataset, to address the problem. A detailed comparison of the datasets is presented in Table I.

Many prior works have already studied UAV detection and classification using RF signals, and achieved good performance with machine learning methods–,–. The authors in these works have considered this task under different scenarios, including low signal-to-noise ratio (SNR), the presence of interference,, out-of-distribution (OOD) and misclassified signal detection, different operational mode and preamble feature extraction. However, these works mainly consider the classification between UAVs of different models, and do not consider fingerprinting problem of interest. On the other hand, fingerprinting UAVs is discussed in, where data streams from 7 identical DJI M100 UAVs are used for classification. It only considered a single 10 MHz channel UAV transmission and thus the detection and localization of UAV signals in a wideband spectrum was not fully characterized. The authors in investigated the drone detection and classification over the entire 2.4GHz ISM band. However, this classification was also limited to differentiating between different UAV makes and models.

---
Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:57 UTC from IEEE Xplore. Restrictions apply.
979-8-3503-7423-0/24/$31.00 ©2024 IEEE
431

---
MILCOM 2024 Track 3 - Cyber Security and Trusted Computing

**TABLE I: Comparison of Datasets**
| Dataset | Sampling Rate | Bandwidth | Multiple Transmitters | Time-Frequency Labels | UAV Models |
| :--- | :--- | :--- | :--- | :--- | :--- |
| | 20 GSa/s | 6 GHz | No | No | Different |
| | 20 GSa/s | 6 GHz | No | No | Different |
| | 60 MSa/s | 28 MHz | No | No | Different |
| | 100 MSa/s | 100 MHz | Synthesized | No | Different |
| | 10 MSa/s | 10 MHz | No | No | Same |
| UAVSig (Our work) | 50 MSa/s | 50 MHz | Yes | Yes | Same |

### B. Contributions

Our contributions can be summarized as follows:
* We collect a large scale UAV RF signal dataset (UAVSig) for drone detection and fingerprinting. UAVSig consists of OTA transmission captures from 8 transmitters in the 2.4GHz ISM band. Specifically, the transmissions are labeled with both time and frequency domain parameters, as well as the transmitter identity. The dataset will be made public for future research.
* We consider the task of UAV signal detection, spectrum localization and fingerprinting and present a spectrogram-based one-stage deep learning model to address the problem. The model is able to achieve a 99.5% precision and 99.4% recall for transmission detection, and 90.9% accuracy for drone fingerprinting on test data.

### C. Organization of the paper

The rest of the paper is organized as follows. Section II discusses our approach to solve the UAV detection, spectrum localization and fingerprinting problem. Then, Section III describes the details of the collected UAVSig dataset. The evaluations and experiments are presented in Section IV. Finally, Section V concludes the paper.

## II. APPROACH

In this section, we first discuss the wideband signal representation and processing, and then explain our one-stage spectrogram-based deep learning model.

### A. Wideband Signal Representation

Due to the hopping nature of many UAV signals, we need to detect and localize them in both time and frequency domain. As a result, it is necessary to represent the signals with both time and frequency information. To obtain such representation, discrete time short time Fourier transform (STFT) is applied on the received digitized signals x(n) as shown in (1):
$$
X(m,f) = \sum_{n=-\infty}^{\infty} x(n)w(n-m)e^{-i2\pi fn} \quad (1)
$$
Here, to suppress spectral leakage, we use the Hamming window function w(k) as detailed in (2), where N = 512 is the window size.
$$
w(k) = 
\begin{cases} 
0.54 - 0.46 \cos(2\pi \frac{k}{N-1}), & 0 \le k \le N-1 \\
0, & \text{otherwise}
\end{cases} \quad (2)
$$
Then, power spectral density (PSD) is computed on the STFT results to obtain the spectrogram following (3).
$$
s(m,f) = |X(m,f)|^2 \quad (3)
$$
*Fig. 1: Neural network architecture of our model. (a) is the structure of the model and (b) is the structure of ResBlock (K). In (a), all the Conv2D and Conv2DTranspose layers have strides of 2 and are followed with a BatchNormalization layer and a LeakyReLU layer, except for the last layer. The last layer in (a) has a sigmoid activation function.*

The spectrograms can be interpreted as images, in which the transmissions are identifiable objects and the transmitter features due to RF signal distortions are converted to unique patterns. Therefore, our task is to find and localize the transmission objects on the images, as well as identify their source transmitters using those patterns. To solve this problem, we present a deep learning model as discussed below.

### B. Deep Learning Model

With the spectrogram representation of the wideband signals, we present a one-stage deep learning model based on the idea of YOLO. The idea is to evenly divide an image into (nr × nc) areas, where nr is the number of vertical division and nc is the number of horizontal division. Each area predicts the coordinates and the transmitter class of a transmission if it exists. Therefore, the model has an output shape of (nr, nc, (N + 4 + 1)), where N corresponds to the number of UAVs to classify between, 4 corresponds to (x, y,w,h), and 1 corresponds to the confidence of the current prediction. For each transmission, x ∈ is the coordinate of its center on x-axis, y ∈ is the coordinate of its center on y-axis, and w, h ∈ are the width and height of it.

Our model architecture is presented in Figure 1. The model is fully convolutional so that it can take variable input shapes and support different spectrum bandwidths and time windows. For this paper, we use a fixed spectrogram size of (512, 512, 1), where 1 represents the single channel of PSD values. Each spectrogram is normalized following (4).
$$
\bar{s}(m,f) = \frac{s(m,f) - \min(s)}{\max(s) - \min(s)} \quad (4)
$$

---
Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:57 UTC from IEEE Xplore. Restrictions apply.
432

---
MILCOM 2024 Track 3 - Cyber Security and Trusted Computing

*Fig. 2: Captured bandwidth with 4 drone channels*

The output shape of the model is (15, 15, c + 5), and the loss function of the model is detailed in (5).
$$
\begin{aligned}
L = & \sum_{i=0}^{N_c} \sum_{j=0}^{N_r} 1_{ij}^{obj} [(x_{i,j} - \hat{x}_{i,j})^2 + (y_{i,j} - \hat{y}_{i,j})^2] \\
& + \sum_{i=0}^{N_c} \sum_{j=0}^{N_r} 1_{ij}^{obj} [(\sqrt{w_{i,j}} - \sqrt{\hat{w}_{i,j}})^2 + (\sqrt{h_{i,j}} - \sqrt{\hat{h}_{i,j}})^2] \\
& + \sum_{i=0}^{N_c} \sum_{j=0}^{N_r} 1_{ij}^{obj} \sum_{n=1}^{N} (p_{i,j}(n) - \hat{p}_{i,j}(n))^2 \\
& + \sum_{i=0}^{N_c} \sum_{j=0}^{N_r} 1_{ij}^{obj} (C_{i,j} - \hat{C}_{i,j})^2 + \sum_{i=0}^{N_c} \sum_{j=0}^{N_r} 1_{ij}^{noobj} (C_{i,j} - \hat{C}_{i,j})^2
\end{aligned} \quad (5)
$$
In (5), $1_{ij}^{obj}$, $C_{i,j}$ and $p_{i,j}(n) \in \{0,1\}$ equal to 1 if a transmission exists in the area denoted by coordinates (i, j). Similarly, $1_{ij}^{nobj} \in \{0,1\}$ is 1 if no transmission exists in the area denoted by coordinates (i, j). All the predictions $\hat{x}, \hat{y}, \hat{w}, \hat{h}, \hat{p}, \hat{C} \in$, and thus the sigmoid activation function is used at the output layer in the model.

To develop and evaluate the model for this task, we collected the UAVSig dataset, which is detailed in the next section.

## III. DATASET

### A. RF and Hardware Configuration

In our UAVSig dataset, we collected over-the-air spectrum data from 8 transmitters: 4 identical DJI M100 UAVs and 4 C1 DJI remote controllers. The UAVs are wideband, fixed-channel transmitters, operating in a 10 MHz channel we select through the controller. Meanwhile, the controllers are frequency-hopped transmitters that operate over the whole 2.4 GHz ISM band. We captured 50 MHz of bandwidth centered at 2.4435 GHz using a small software-defined radio (SDR), one USRP B205mini-i. We collected data with each drone transmitting in each of 4 selected channels, as illustrated in Fig. 2. Our center frequency does not precisely center the 4 channels in the spectrum, but we do ensure that all UAV channels-of-interest are fully inside the spectrograms. Since the controllers are frequency-hopped, some transmissions are cut-off partially or entirely.

The USRP was connected to a 20 dBi panel antenna with an 18° beamwidth. While we were unable to capture data in a location devoid of 2.4 GHz ISM band devices, the directional antenna’s spatial isolation significantly reduced interference from unknown sources we cannot label or control. The receiver antenna was angled upwards towards the transmitters-of-interest and the sky, further reducing unwanted interference.

### B. Collection

The UAVSig dataset contains two capture types from two days: 1) one and two drone captures and 2) controller captures. Note that for clarity, we denote each combination of transmitters-of-interest, physical placements, and possible channel assignments as a *scenario*.

For the one and two drone captures, the drones, the transmitters-of-interest, were placed on an elevated table at a set distance away from the receiver antenna. We designated two locations for drone placement, side-by-side horizontally across the table, as shown in Fig. 3. These designated spots allowed the captured transmit power to be approximately the same for each drone. A drone was placed in each spot for two drone collection, or one drone was placed only in the leftmost spot for one drone collection. When capturing drone signals, a controller needs to be paired to each drone. Since we needed to isolate the drone signals from those of the controllers, we placed the controllers far behind and a floor below the receiver antenna. Note the data collection was conducted in a controlled, lab-like setting to make the captures from different transmitters as similar as possible. Our goal was to isolate the RF fingerprinting capability, rather than adding side-channel information that could affect the performance.

As described earlier, we assigned each drone to transmit in one of four channels. For one drone, we collected every combination of each drone transmitting on each of the four channels for a total of 16 scenarios. For two drones, we collected every combination of two drones transmitting on two different channels for a total of 72 scenarios.

For the controller captures, we placed the controllers on the same elevated table as with the drones. There were four designated locations for controller placement, two placed on the table side-by-side, and two placed additionally on top of boxes behind the first two side-by-side, as illustrated in Fig. 4. To make the transmissions more consistent and to avoid interference from the drones, we did not pair a drone to the controllers when collecting controller captures. Anecdotally,

*Fig. 3: Two drone collection setup*

---
Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:57 UTC from IEEE Xplore. Restrictions apply.
433

---
MILCOM 2024 Track 3 - Cyber Security and Trusted Computing

*Fig. 4: Controllers collection setup*

we did not visually notice any significant difference between the controller transmissions and activity with and without a drone connected. Each controller scenario does not include an RF channel to select, since, as previously stated, the controller signals are frequency-hopped with a fixed center frequency and hopping bandwidth. We instead create 16 scenarios with every combination of the four controllers being turned on or off. We also captured this for two different position setups of the controllers by flipping each controller diagonally from the original setup for a total of 32 scenarios.

For each of the 16 scenarios with one drone, 72 scenarios with two drones, and 32 scenarios with controllers in the UAVSig dataset, we captured six, one-second, 50 MSa/s captures with a receiver gain of 20 dB. The captures were taken consecutively with small time gaps between each capture. Each captured sample includes 16-bit I/Q data, saved as two 32-bit floats by our GNU-Radio-based data collection software. Synthesized signals were not used for the UAVSig dataset as addition might introduce unrealistic RF characteristics. Synthesizing multiple transmitter signals together could fail to capture more complex interactions between simultaneous transmissions.

### C. Processing

To utilize the collected data for training and testing our model, we need to label each transmission with its start time, end time, center frequency, bandwidth and source transmitter identity. To generate the time and frequency labels, we adopt a digital signal processing (DSP) procedure as depicted in Fig. 5. Then, to manually assign the transmitter identity labels, we consider each scenario separately. For a scenario where only one transmitter is on, all the detected transmissions in the 6 captures are assigned the same transmitter label. For the scenarios where two drones are on, as the channels of the drones are recorded for each scenario, we assign the drone labels by comparing the estimated and recorded true center frequencies of the transmissions. Finally, for the scenarios where multiple controllers are on, we cannot control their hopping sequence and thus can only assign non-deterministic labels for the transmissions. All active controllers are listed in these labels. It should be noted that while the DSP approach can help to label the transmissions, it cannot fingerprint the UAVs at the same time. Therefore, we present our spectrogram-based one-stage deep learning model to solve the detection and fingerprinting tasks simultaneously. After the labeling, we split each 1 second signal into 5.2 millisecond segments to generate the 512 × 512 spectrograms. Examples of labeled spectrograms are shown in Fig. 6.

*Fig. 5: DSP label generation workflow.*

*Fig. 6: Example labeled spectrograms of signals in UAVSig dataset where the transmissions are bounded with red boxes. (a) shows a spectrogram with drone 1 on channel 3 and drone 3 on channel 1, and (b) shows a spectrogram with all four controllers active.*

## IV. EVALUATION

To examine the model performance, a intersection-over-union (IoU) threshold is used, where IoU is the ratio of the intersected area over the union area of two bounding boxes. In the scope of this work, we consider a prediction as correct if it has an IoU ≥ 0.5 with the real transmission. Consequently, to evaluate the transmission detection performance of the model, we report the precision and recall of the predictions, which are calculated as shown in (6) and (7). In (6) and (7), TP, FP and FN mean true positives, false positives and false negatives, respectively. Then, to evaluate the UAV fingerprinting performance, we consider the classification accuracy, which is computed within the correctly detected transmissions. In the evaluation, we consider a baseline model from, because it is designed for a most similar problem, which is to detect UAV transmissions and classify UAV models. The baseline model adopts the YOLO-lite architecture with some adaption. To ensure fairness for the comparison, we use the same input shape of (512 × 512) for both our model and the baseline model.

The details of the experiments are explained in the rest of the section. For all the evaluations, we use the 5 captures of a scenario for training and the other capture in the same scenario for testing.
$$
\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} \quad (6)
$$
$$
\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} \quad (7)
$$

---
Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:57 UTC from IEEE Xplore. Restrictions apply.
434

---
MILCOM 2024 Track 3 - Cyber Security and Trusted Computing

First, we evaluate the model on a simple case, where only one drone is turned on each time. This case is denoted by "one-drone", which includes 16 scenarios as described in Section III. We consider both our model and the baseline model for the evaluation. The results are shown in Table II, where the performance of our model is in the "train on one-drone, test on one-drone" column. It can be observed that our model is able to correctly detect 98.2% of the drones' transmissions, with limited number of false positive predictions. Besides, within the successfully detected transmissions, we can classify between the drones with an accuracy of 81.6%. At the same time, while the baseline model is able to detect most of the transmissions, it also makes a lot of false predictions, and fail at the drone fingerprinting task. This result shows that compared to the UAV classification problem, the fingerprinting problem is indeed more challenging and warrants a more complex network architecture. At the same time, it is able to solve this problem with carefully designed deep learning model. As the baseline model cannot reliably fingerprint the drones, we only evaluate our model in the following cases.

Finally, the confusion matrix of our model in this case is presented in Figure 7, where an integer value in the confusion matrix represents the number of bounding boxes. As shown in the confusion matrix, the model tends to misclassify between the pair of drone 1 and drone 2 and also the pair of drone 3 and drone 4. This result shows that there might be some subtle hardware similarities within each pair of drones. These observations can be helpful for future security design, such as preventing possible spoofing attacks.

*Fig. 7: Confusion matrix of our model when it is trained on one-drone and also tested on one-drone data.*

Next we consider the scenario when two drones might transmit simultaneously. As introduced in Section III, this case includes 72 scenarios and is denoted by "two-drone". The drone detection results are shown in Table II in the column of "train on two-drone, test on two-drone", and the confusion matrix of the classification results is presented in Figure 8. We can observe that the precision, recall and classification accuracy are all improved upon the "one-drone” case. This result is reasonable, because we have much more training data, as well as different combinations of the transmissions in the two-drone case. Therefore, the model is able to learn more robust features for the transmissions and drones.

Then, we consider more difficult cases by evaluating the above trained models on the other test set. In other words, we evaluate the model trained on one-drone data on two-drone data, and vice versa. These results are also shown in Table II and demonstrate that the models are still able to correctly detect most transmissions. Specifically, the model trained on two-drone data can even have slightly higher precision when tested on the one-drone data than on the two-drone data. This result is not intuitive but still reasonable, because the two-drone case has different combinations of drones and thus more complicated RF environment. At the same time, both models have significant degradation in the drone fingerprinting performance, and the confusion matrices are presented in Fig. 9. It is possible that when two drones are turned on and assigned channels close to each other in the frequency domain, the drones will transmit signals in a different pattern because of the changed RF environment. However, the model trained on two-drone data still has a classification accuracy of 48.8%, which means there still exist some consistent RF features of the drones across the two cases.

Finally, we evaluate our model on the controller data. As

*Fig. 8: Confusion matrix of our model when it is trained on two-drone and also tested on two-drone data.*

*Fig. 9: Confusion matrices of the cross tests, where (a) shows the results of "train on one-drone, test on two-drone" and (b) shows the results of "train on two-drone, test on one-drone".*

---
Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:57 UTC from IEEE Xplore. Restrictions apply.
435

---
MILCOM 2024 Track 3 - Cyber Security and Trusted Computing

**TABLE II: Model Performance of Different Cases**
| Case | Baseline | Train on one-drone, Test on one-drone | Train on two-drone, Test on two-drone | Train on one-drone, Test on two-drone | Train on two-drone, Test on one-drone | Controller |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Precision | 39.3% | 98.6% | 99.5% | 94.5% | 99.5% | 97.1% |
| Recall | 74.0% | 98.2% | 99.4% | 80.8% | 98.3% | 97.6% |
| Classification Accuracy | 25.8% | 81.6% | 90.9% | 37.0% | 48.8% | |

shown in Section III, compared to drones, the controllers transmit with much smaller bandwidth, shorter time duration, and longer intervals. Therefore, the detection and spectrum localization of controller transmissions could be a harder problem. We include all 15 controller scenarios for training. Since we do not have precise ground truth controller labels for the multiple controller scenarios, we cannot fingerprinting controllers in this case. The detection and spectrum localization results are presented in the last column of Table II. Both the precision and recall slightly decrease compared to the drone transmission detection results. This degradation confirms our intuition that the detection and spectrum localization of controller transmissions is more difficult than that of drone transmissions.

## V. CONCLUSION

In this work, we considered possible security threats brought by UAVs. Specifically, to ensure UAV compliance and security, it is important to solve the UAV detection, spectrum localization and fingerprinting problem. First, we presented a one-stage deep learning model to detect and fingerprint the UAVs using spectrograms. Then, to evaluate the model, we collected UAVSig, which is an OTA captured RF dataset for the UAVs¹. Using this dataset, our model can detect the drone transmissions with 99.5% precision, 99.4% recall and fingerprint the drones with 90.9% accuracy. In the evaluations, we focused on the drone transmissions, and also observed model performance degradation in certain scenarios, which can be further analyzed for a more robust model.

In the future, we will improve the generalizablity of UAVSig and our RF fingerprinting model. We will augment the existing dataset with more OTA captures to include more UAVs and diversified scenarios, such as the varying SNRs found in UAV swarms. While increasing the training data size can improve the performance, capturing data for all possible combinations of the UAVs is unrealistic, especially with many UAVs. Therefore, future data collection must introduce enough variability within a reasonable amount of data. For the model, we will also investigate and improve its robustness in different scenarios, such as open-set scenarios due to adversarial attacks, and also domain-shift problems due to channel and receiver variations.

## REFERENCES

 M. Mozaffari, W. Saad, M. Bennis, Y.-H. Nam, and M. Debbah, “A Tutorial on UAVs for Wireless Networks: Applications, Challenges, and Open Problems,” *IEEE Communications Surveys & Tutorials*, vol. 21, no. 3, pp. 2334–2360, 2019.
 B. Taha and A. Shoufan, “Machine Learning-Based Drone Detection and Classification: State-of-the-Art in Research,” *IEEE Access*, vol. 7, pp. 138669–138682, 2019.
 H. Zhang, T. Li, Y. Li, J. Li, O. A. Dobre, and Z. Wen, “RF-Based Drone Classification Under Complex Electromagnetic Environments Using Deep Learning,” *IEEE Sensors Journal*, vol. 23, no. 6, pp. 6099–6108, 2023.
 Y. Chen, L. Zhu, Y. Jiao, C. Yao, K. Cheng, and Y. Gu, “An Extreme Value Theory-Based Approach for Reliable Drone RF Signal Identification,” *IEEE Transactions on Cognitive Communications and Networking*, vol. 10, no. 2, pp. 454–469, 2024.
 G. Reus-Muns and K. R. Chowdhury, “Classifying UAVs With Proprietary Waveforms via Preamble Feature Extraction and Federated Learning,” *IEEE Transactions on Vehicular Technology*, vol. 70, no. 7, pp. 6279–6290, 2021.
 I. Guvenc, F. Koohifar, S. Singh, M. L. Sichitiu, and D. Matolak, “Detection, Tracking, and Interdiction for Amateur Drones,” *IEEE Communications Magazine*, vol. 56, no. 4, pp. 75–81, 2018.
 M. Ezuma, F. Erden, C. K. Anjinappa, O. Ozdemir, and I. Guvenc, “Drone Remote Controller RF Signal Dataset.” IEEE Dataport, doi: 10.21227/ss99-8d56, 2020.
 C. J. Swinney and J. C. Woods, “DroneDetect Dataset: A Radio Frequency dataset of Unmanned Aerial System (UAS) Signals for Machine Learning Detection & Classification.” IEEE Dataport, doi: 10.21227/5jjj-1m32, 2021.
 O. Medaiyese, M. Ezuma, A. Lauf, and A. Adeniran, “Cardinal RF (CardRF): An Outdoor UAV/UAS/Drone RF Signals with Bluetooth and WiFi Signals Dataset.” IEEE Dataport, doi: 10.21227/1xp7-ge95, 2022.
 N. Soltani, G. Reus-Muns, B. Salehi, J. Dy, S. Ioannidis, and K. Chowdhury, “RF Fingerprinting Unmanned Aerial Vehicles With Non-Standard Transmitter Waveforms,” *IEEE Transactions on Vehicular Technology*, vol. 69, no. 12, pp. 15518–15531, 2020.
 S. Basak, S. Pollin, and B. Scheers, “Drone RF Dataset.” KU Leuven RDR, doi: 10.48804/HZRVNZ, 2024.
 D. Uvaydov, M. Zhang, C. P. Robinson, S. D’Oro, T. Melodia, and F. Restuccia, “Stitching the Spectrum: Semantic Spectrum Segmentation with Wideband Signal Stitching.” arXiv:2402.03465, 2024.
 E. Ozturk, F. Erden, and I. Guvenc, “RF-Based Low-SNR Classification of UAVs Using Convolutional Neural Networks.” arXiv:2009.05519, 2020.
 M. Ezuma, F. Erden, C. Kumar Anjinappa, O. Ozdemir, and I. Guvenc, “Detection and Classification of UAVs Using RF Fingerprints in the Presence of Wi-Fi and Bluetooth Interference,” *IEEE Open Journal of the Communications Society*, vol. 1, pp. 60–76, 2020.
 K. Raina, T. Alladi, V. Chamola, and F. R. Yu, “Detecting UAV Presence Using Convolution Feature Vectors in Light Gradient Boosting Machine,” *IEEE Transactions on Vehicular Technology*, vol. 72, no. 4, pp. 4332–4341, 2023.
 S. Basak, S. Rajendran, S. Pollin, and B. Scheers, “Combined RF-Based Drone Detection and Classification,” *IEEE Transactions on Cognitive Communications and Networking*, vol. 8, no. 1, pp. 111–120, 2022.
 Joseph Redmon and Santosh Divvala and Ross Girshick and Ali Farhadi, “You only look once: Unified, real-time object detection.” arXiv:1506.02640, 2016.
 R. Huang, J. Pedoeem, and C. Chen, “YOLO-LITE: A Real-Time Object Detection Algorithm Optimized for Non-GPU Computers,” in *2018 IEEE International Conference on Big Data (Big Data)*, (Los Alamitos, CA, USA), pp. 2503–2510, IEEE Computer Society, dec 2018.
 T. Zhao, S. Sarkar, Y. Tian, and D. Cabric, “Anomaly Transmitter Recognition and Tracking,” in *2024 IEEE International Symposium on Dynamic Spectrum Access Networks (DySPAN)*, pp. 357–364, 2024.
 G. Shen, J. Zhang, A. Marshall, and J. R. Cavallaro, “Towards scalable and channel-robust radio frequency fingerprint identification for lora,” *IEEE Transactions on Information Forensics and Security*, vol. 17, pp. 774–787, 2022.
 T. Zhao, S. Sarkar, E. Krijestorac, and D. Cabric, “GAN-RXA: A Practical Scalable Solution to Receiver-Agnostic Transmitter Fingerprinting,” *IEEE Transactions on Cognitive Communications and Networking*, vol. 10, no. 2, pp. 403–416, 2024.

---
¹Dataset available at: https://cores.ee.ucla.edu/downloads/datasets/uavsig/.

---
Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:57 UTC from IEEE Xplore. Restrictions apply.
436