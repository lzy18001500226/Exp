2024 IEEE 99th Vehicular Technology Conference (VTC2024-Spring) | 979-8-3503-8741-4/24/$31.00 ©2024 IEEE | DOI: 10.1109/VTC2024-SPRING62846.2024.10683639
# Hierarchical Deep Learning Framework for Enhanced UAV Classification Mitigating Bluetooth and WiFi Interference
Chanchal Kumari¹, Nelapati Lava Prasad¹, Udit Satija², and Barathram Ramkumar¹
¹School of Electrical Sciences, Indian Institute of Technology Bhubaneswar, Odisha-752050, INDIA
²Department of Electrical Engineering, Indian Institute of Technology Patna, Bihar-801106, INDIA
Email: 22sp06006@iitbbs.ac.in, nlp10@iitbbs.ac.in, udit@iitp.ac.in, and barathram@iitbbs.ac.in

---
***Abstract*—In recent years, there has been a widespread use of unmanned aerial vehicles (UAVs) or drones on both commercial and defense applications. There is an increasing security concern as these UAVs can also be used for illegal activities by malicious users. It is necessary to detect and classify these malicious UAVs or drones in order to neutralize them. In this paper, a hierarchical scheme is proposed for detecting and classifying UAVs based on the RF signatures obtained from the RF link (both uplink and downlink) that is used to control the UAV from the ground station or UAV controller. The work also addresses the challenge of classifying UAV RF signatures in the presence of interferences like WiFi and Bluetooth as they also operate in the same frequency band. The method involves entropy-based steady-state extraction, segmentation, and spectrogram image conversion followed by a hierarchical classification based on different variants of convolutional neural network (CNN). The proposed method is validated on signals taken from the CardRF database and achieves an average detection accuracy of 99.7% and an average precision of 99.93%. As it is a hierarchical classification scheme, performance measures at each hierarchical stages are reported in the simulation results.**

***Index Terms*—RF signatures, CNN, Deep learning, UAV classification**

---
## I. INTRODUCTION
Over the last decade, there has been an exponential in- crease in the usage of UAVs or drones due to its low cost and ease of operations. UAVs are used in both commercial and military applications. In commercial applications, UAVs are used in the field of agriculture, healthcare, disaster man- agement, traffic monitoring, coastal monitoring, and many more. In many countries, there is no proper regulation for the usage of drones. In spite of huge application potential, the unregulated use of UAVs poses a severe security challenge.
UAVs flying in populated areas or near manned aircraft can pose a collision risk and probably can lead to accidents and injuries. Criminals also use UAVs to transport illicit materials at the border, which is a threat to the order and stability of the border. UAVs, which are not authorized may enter in restricted airspace, airports, or other critical areas, at risk of public safety. UAVs, which have cameras, can trespass on people’s privacy by capturing images and videos with- out permission, probably leading to legal issues. Malignant people can use UAVs for unauthorized surveillance, or for gathering sensitive information. UAVs also can be used by terrorists for attacks which is a threat to everyone. Hence, detecting the presence of UAV and classifying the type of UAS becomes an important primary step before neutralizing a malicious UAV or drone.

There are some possible ways of detecting the presence of UAV in the airspace: Radar detection, humming sounds from UAV (audio) or sonar detection, video surveillance detection, detection using the thermal ray from UAV mo- tor or rotors (thermal sensing) and detection using radio frequency (RF) signals from UAV controller or telemetry (RF sensing). Some of the advantages of the RF detection technique are: i) it can detect UAVs from short distances to several kilometers, depending on the equipment we are using. ii) Many UAVs use RF communication for control and data transmission which can be detected by RF detection technique; iii) Passive detection: it does not emit signals that can reveal the detection system’s presence; iv) It can detect non-visual UAVs even in adverse weather conditions, at night or in situations where UAVs are difficult to be seen with naked eyes; v) it can be cost-effective compared to other methods, mainly for small to medium-sized installations; vi) it can be used for different UAV modes (like flying, hovering, videoing, etc.) identification.

### A. Related works on UAV detection and classification using RF signals
Wavelet scattering transform was used in to extract image-based features from control signals of UAV and CNN algorithm (SqueezeNet) was trained with scattergram images. An accuracy of 98.9% was reported at 10dB signal-to-noise ratio (SNR). However, only UAV’s control signals were used for UAV detection. In, the power spectral density (PSD) of UAV RF signals in the form of images are being used as signatures for identifying the flight modes of drones. For extracting the features from the PSD images, deep residual learning (ResNet50) was used. These extracted features are used to train a logistic regression algorithm for classification. However, the authors in have not considered the interferences due to other sources in the ISM band. In, an algorithm for classifying identical UAVs in hovering flight mode was proposed. The authors achieved an accuracy of 91%. They have also not considered an interference mitigation strategy for other devices operating at 2.4 GHz. In, authors used a stacked denoising autoencoder (SDAE) for signal compression. Signal compression is performed to

---
979-8-3503-8741-4/24/$31.00 ©2024 IEEE
Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:55 UTC from IEEE Xplore. Restrictions apply.

---
reduce the space complexity. Then, the compressed signal is passed into a local outlier factor (LOF) algorithm to detect unmanned aerial system (UAS) (UAV and UAV controller, refer to Fig. 1) signal in the presence of Bluetooth and WiFi signals. Then, a three-level hierarchical classification is used to determine the type of UAS (UAV or UAV controller), type of UAV and UAV controller, and type of flight mode. The Hilbert-Huang transform (HHT) and wavelet packet transform (WPT) were used for extracting unique features from steady state of the UAS signals. The extracted feature set is then used to train an machine learning (ML)-based three-level hierarchical classifier in which extreme gradient boosting classifiers (XGBoost) are used as the learning algorithm. In UAS detection, 89.49% accuracy was reported. For UAS classification, UAV classification, UAV controller classification, 91%, 73.19% and 82.49% accuracies were reported.

From the literature, it can be noted that a very few works consider the detection and classification of UAS under interference from other signals in ISM band (e.g., Bluetooth and WiFi). Further, most of the works do not consider the classification of the entire UAS. In, a hierarchical scheme to classify UAS is presented, however, the accuracy reported on standard database signals is less. In this work, a better preprocessing scheme and a hierarchical classification scheme based on different deep learning techniques are proposed, which offers better performance.

The contributions of this work are summarized as follows:
1) In this work, a simple novel method is proposed to separate the steady state of the UAV RF signal from the noise and transients unlike existing method in. The proposed method is based on finding the segment-wise entropy of the signal which discriminates the noise, transient state, and steady state.
2) The detection of UAS in the presence of Bluetooth and WiFi signals operating at 2.4 GHz frequency using spectrogram of received segmented signals and a 2D CNN-based model.
3) After UAS detection, type of UAS classification is performed to determine the type of UAS (UAV or UAV controller), classification of UAV and UAV controller, and type of flight mode of UAV.

The organization of this paper is as follows. Section II describes about the system model and the dataset used in this work. Section III explains the methodology used in this work for the detection and classification of UAVs. Section IV describes experimental results and a discussion about the results, followed by the conclusion.

## II. SYSTEM MODEL AND DESCRIPTION OF DATASET
Fig. 1 depicts the basic system model for UAS which consists of UAV, UAV controller, and the link between UAV and UAV controller. It has been noted that UAVs are operating under interference from other ISM band signals. In this paper, we have used a publicly available CardRF dataset which contains a RF signals obtained through UAS system and interference sources such as Bluetooth and WiFi. In this dataset, signals are sampled, each with five million samples. The signals are collected from different UAV models including, DJI M600, DJI MavicPro (in flying and hovering modes), Beebeerun, DJI Phantom (in flying and hovering modes), and DJI Inspire (in flying and videoing modes). Also the dataset contains UAV controller signals from different models such as, DJI M600, DJI MavicPro, Beebeerun, DJI Phantom, DJI Inspire, and Iris. Bluetooth signals are collected using APPLE Ipad3, Apple Iphone6S, Apple Iphone7, FitBit Charge3, and Motorola. For WiFi signals, Cisco Linksys E3200 and TP-Link TL WR940N are used. This dataset has both line-of-sight (LOS) and non-line-of-sight (NLOS) signals. The complete details of the dataset is summarized in Table I.

*Fig. 1. Description of the UAV operating scenario.*

## III. PROPOSED HIERARCHICAL FRAMEWORK
The block diagram of the proposed framework is illustrated in Fig. 3, which consists of three major stages: 1) preprocessing of the RF signals; 2) time-frequency representation in the form of spectrograms; 3) proposed hierarchical deep learning methodologies for following: UAS/Non-UAS signal detection, binary classification of UAV/UAV controller, multi-class classification of UAV types and their flying modes.

### A. Pre-processing
As shown in Fig. 3, preprocessing involves normalization, steady-state extraction, segmentation, and spectrogram computation.
1) *Normalization and steady-state extraction:* Let the received signal be x(n), n = 1, 2, . . . N, then the normalization process is defined as follows:
$$
x'(n) = \frac{x(n) - \frac{1}{N} \sum_{n=1}^{N} x(n)}{|\max(|x(n)|)|} \quad (1)
$$
As mentioned earlier, Card RF database has five million samples for each signal. Here, the signals are segmented with 100,000 samples per segment with 50% overlapping. In the CardRF dataset, the transient state of a few signals is only available. Hence, to extract the steady-state, we propose an

---
Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:55 UTC from IEEE Xplore. Restrictions apply.

---
**TABLE I**
CARDRF DATASET DESCRIPTION
| Class | Device | Model | Mode | Number of signals | Signal Contains | Ratio (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| UAS | UAV | Beebeerun | Flying | 350 | Noise, transient and steady state | 5.75 |
| | | DJI Inspire | Flying | 175 | steady state | 2.87 |
| | | | Videoing | 175 | Noise, transient and steady state | 2.87 |
| | | DJI M600 | Flying | 350 | Noise, transient and steady state | 5.75 |
| | | DJI Mavicpro | Flying | 175 | steady state | 2.87 |
| | | | Hovering | 175 | steady state | 2.87 |
| | | DJI Phantom | Flying | 175 | steady state | 2.87 |
| | | | Hovering | 175 | steady state | 2.87 |
| | UAV Controller | Beebeerun | | 245 | Noise, transient and steady state | 4.02 |
| | | DJI M600 | | 350 | Noise, transient and steady state | 5.75 |
| | | DJI Inspire | | 350 | Noise, transient and steady state | 5.75 |
| | | IRIS | | 350 | Noise, transient and steady state | 5.75 |
| | | DJI Mavicpro | | 350 | Noise, transient and steady state | 5.75 |
| | | Phantom | | 350 | Noise, transient and steady state | 5.75 |
| NON-UAS | Bluetooth | Apple Ipad3 | | 350 | Noise and steady state | 5.75 |
| | | Apple Iphone6S | | 350 | Noise and steady state | 5.75 |
| | | Apple Iphone7 | | 350 | Noise and steady state | 5.75 |
| | | FitBit Charge3 | | 350 | Noise and steady state | 5.75 |
| | | Motorola | | 245 | Noise and steady state | 4.02 |
| | WiFi | Cisco Linksys E3200 | | 350 | Noise and steady state | 5.75 |
| | | TP-Link TL WR940N | | 350 | Noise and steady state | 5.75 |

entropy-based detection of steady state of the RF signals. Entropy is considered due to its ability to differentiate between noise, transient and steady-state which can be computed as follows:
$$
H(X) = - \sum_i (P_X(x_i) \log(P_X(x_i))), \quad (2)
$$
where X is sample space; $P_X$ denotes the distribution of X; H(X) is the entropy of X; $x_i$ is the $i^{th}$ element of sample space.

To illustrate the extraction capability of the steady state signals, the entropy values for different RF signals are plotted in Fig. 2. In Fig. 2(b), the normalized signal plot of DJI Inspire controller, and in Fig. 2(f), its entropy plot is illustrated. It can be seen that at steady state there is a jump in the entropy values. Hence, location of the jump in entropy value can be used as a discriminant to identify the steady-state portion of the segment. Similarly, in Fig. 2(c) and 2(d), the plots of the signals of other devices and in Fig. 2(g) and 2(g), their corresponding entropy plots can be observed respectively. Hence, the steady state extraction is done by thresholding on entropy values.
2) *Time-frequency representation in the form of spectrograms:* The spectrogram shows the intensity of the short-time Fourier transform (STFT) magnitude over time. It helps us in the visualization of how the frequency content of the signal changes over time with a sequence of DTFTs of windowed data segments. The spectrogram is computed as follows:
$$
X_m(\omega) = \sum_{n=-\infty}^{\infty} x(n)w(n - mR) e^{-j\omega n}, \quad (3)
$$
where $x(n)$ denotes input signal at time $n$; $w(n)$ denotes length $m$ window function; $X_m(\omega)$ denotes DTFT of windowed data centered about time $mR$; $R$ denotes hop size, in samples, between successive DTFTs. Here, $R$ is taken as 10,000. For each signal in the databse, 100 images are generated and total number of images generated is 609,000.

### B. Hierarchical deep learning methodologies
Three-level hierarchical classification is proposed which are as follows:
1) *Proposed convolution neural network-based deep learning method for UAS Detection:* In order to detect the presence of UAS signal in the presence of other ISM band interference, CNN-based deep learning architecture is considered. The proposed CNN architecture is shown in Fig. 4 and it consists of three convolution layers and two fully connected layers. The details of the parameters in each layer are summarised in Table II. The dropout rate at the dropout layer is taken as 0.3. At the end of two fully connected layers with two activation functions ReLu, and sigmoid are used, respectively.
2) *Proposed deep learning method for binary classification:* In this CNN model, a pre-trained ResNet50 model is used. In this model, residual blocks are introduced, which alleviate the vanishing gradient problem and enable the training of significantly deeper networks. The architecture of the CNN-based binary-class classifier model is shown in Fig.

**TABLE II**
PARAMETERS FOR DIFFERENT LAYERS OF CNN FOR UAS DETECTION
| Layer | Operation | Number of feature maps | Size of feature maps | Size of window | Number of parameters |
| :--- | :--- | :--- | :--- | :--- | :--- |
| C1 | Convolution | 32 | 64 x 64 | 5 x 5 | 2432 |
| S1 | Max-Pooling | 32 | 32 x 32 | 2 x 2 | 0 |
| C2 | Convolution | 64 | 32 x 32 | 3 x 3 | 18496 |
| S2 | Max-Pooling | 64 | 16 x 16 | 2 x 2 | 0 |
| C3 | Convolution | 128 | 8 x 8 | 3 x 3 | 73856 |
| F1 | Fully connected | 512 | 1 x 1 | N/A | 4194816 |
| F2 | Fully connected | 1 | 1 x 1 | N/A | 513 |

---
Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:55 UTC from IEEE Xplore. Restrictions apply.

---
*Fig. 2. Normalized RF signals from (a) DJI Inspire UAV, (b) DJI Inspire controller, (c) Apple iPhone 6S, and (d) WiFi Cisco, from (e)-(h) their corresponding entropy plots.*

*Fig. 3. Block diagram of a proposed hierarchical framework for UAV detection and classification.*

*Fig. 4. Architecture of CNN-based UAS detection.*

*Fig. 5. Architecture of CNN-based binary-class classifier model.*

**TABLE III**
PARAMETERS FOR DIFFERENT LAYERS OF BINARY CLASS CLASSIFIER
| Layer | Operation | Number of feature maps | Size of feature maps | Number of parameters |
| :--- | :--- | :--- | :--- | :--- |
| P1 | resnet50 | 2048 | 7 x 7 | 23587712 |
| S1 | Global-average Pooling2D | 2048 | N/A | 0 |
| D1 | Dense | 256 | 1 x 1 | 524544 |
| D2 | Dense | 164 | 1 x 1 | 42148 |
| D3 | Dense | 1 | 1 x 1 | 165 |

5. The details of parameters in each layer are summarized in Table III.

### C. Architecture of CNN based multi-class classifier model
The architecture of the CNN-based multi-class classifier is shown in Fig. 6. The CNN model is similar to the previous model(Fig. 5), except that the number of dense layers here is two. Also, the activation function used here is softmax instead of the sigmoid. The details of parameters in each layer are given in Table IV.

---
Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:55 UTC from IEEE Xplore. Restrictions apply.

---
*Fig. 6. Architecture of CNN-based multi-class classifier.*

**TABLE IV**
PARAMETERS FOR DIFFERENT LAYERS OF MULTI-CLASS CLASSIFIER
| Layer | Operation of Layer | Number of feature maps | Size of feature maps | Number of parameters |
| :--- | :--- | :--- | :--- | :--- |
| P1 | resnet50 | 2048 | 7 x 7 | 23587712 |
| S1 | Global-average Pooling 2D | 2048 | N/A | 0 |
| D1 | Dense | 512 | 1 x 1 | 1049088 |
| D2 | Dense | 6 | 1 x 1 | 3078 |

## IV. EXPERIMENTAL RESULTS AND DISCUSSION

In this section, we evaluate the performance of the proposed method in terms of accuracy, precision, recall, and F1 score. The definitions of these metrics are
*Accuracy = (TP + TN) / (TP + TN + FP + FN)*
*Precision = (TP) / (TP + FP)*
*Recall = (TP) / (TP + FN)*
*F1-score = (Precision × Recall) / (Precision + Recall) (4)*
where TP, TN, FP, and FN represent true positive, true negative, false positive, and false negative, respectively.

In order to analyze the performance of the proposed methodologies, we perform the following experiments:
**Experiment 1:** In this experiment, we evaluate the UAS detection performance in the presence of WiFi and Bluetooth signals. The confusion matrix and t-SNE plot for detection are shown in Fig. 7, and the performance results are given in Table V. From the table it can be observed the proposed method offers 99.7% accuracy.
**Experiment 2:** In this experiment, the performance of the proposed method is analyzed for UAS classification, UAV classification, UAV controller classification, and UAV mode classifier.The performance of the architecture is evaluated based on training and testing of the network using a five-fold cross-validation method for all six classifiers. For all classifiers, the data is split into 80%, 10%, and 10% for training, validation, and testing sets, respectively. The average accuracy obtained for the UAS classifier, UAV classifier for five classes, UAV controller classifier for six classes, DJI inspire classifier for binary modes, Mavicpro classifier for binary modes and Phantom classifier for binary modes, are 98.61%, 91.61%, 83.48%, 92.31%, 97.48% and 87.18%, respectively. The confusion matrix for each classifier is given in Fig. 8 and t-SNE plots are given in Fig. 9. Comparison with the existing method is given in Table VI and it can be seen that the proposed method offers improved performance.

*Fig. 7. (a) 2D t-SNE visualization of UAV detection and (b) normalized confusion of the UAS detection model.*

**TABLE V**
PERFORMANCE MEASURES OF UAS DETECTION MODEL
| Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| 99.70% | 99.93% | 99.47% | 99.7 |

## V. CONCLUSION

In this paper, a hierarchical scheme was proposed to detect and classify the UAVs in the presence of other ISM band signals. As a part of the preprocessing, an entropy-based steady-state extraction method is proposed. Different types of deep learning models were used at various stages of detection and classification. The method classifies between five different classes of UAVs and six different classes of UAV controllers. The method also classifies the modes of operation of the UAVs. The performance of the method is tested using signals taken from the CardRF database. The results show improved performance over the existing methods.

## VI. ACKNOWLEDGEMENT

This work was supported by Impacting Research, Innovation and Technology (IMPRINT)-2 under Project IMP/2018/001719.

## REFERENCES

 Lygouras E., Gasteratos A., Tarchanidis K., Mitropoulos A.C. ROLFER: A fully autonomous aerial rescue support system. Microprocess. Microsyst. 2018;61:32-42. doi: 10.1016/j.micpro.2018.05.014.
 Idries A., Mohamed N., Jawhar I., Mohamed F., Al-Jaroodi J. Chal- lenges of developing UAV applications: A project management view, Proceedings of the 2015 International Conference on Industrial Engineering and Operations Management (IEOM); Dubai, United Arab Emirates. 3-5 March 2015; pp. 1-10.
 C Wang, J Tian, J Cao et al., ”Deep learning-based UAV detection in pulse-doppler radar”, IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1-12, 2021.
 I Güvenc, O Ozdemir, Y Yapici et al., ”Detection localization and tracking of unauthorized UAS and jammers”, 2017 IEEE/AIAA 36th Digital Avionics Systems Conference (DASC), pp. 1-10, 2017.
 U Seidaliyeva, D Akhmetov, L Ilipbayeva et al., ”Real-time and accurate drone detection in a video with a static background”, Sensors, vol. 20, no. 14, pp. 3856, 2020.
 P. Andrasi, T. Radišić, M. Muštra, and J. Ivošević, ”ScienceDirect Night-time Detection of UAVs using Thermal Infrared Camera”, Transp. Res. Procedia, vol. 28, pp. 0-000, 2017.

---
Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:55 UTC from IEEE Xplore. Restrictions apply.

---
*Fig. 8. Normalized confusion matrix of all level classifier, (a) UAS classifier(UAV and UAV controller class), (b) UAV classifier(for five classes), (c) UAV controller classifier(for six classes), (d) DJI Inspire classifier(for flying and videoing modes), (e) DJI Phantom classifier(for flying and hovering modes), (f) DJI Mavicpro classifier(for flying and hovering modes)*

**TABLE VI**
PERFORMANCE MEASURE OF EACH CLASSIFIER AND COMPARISON WITH
| Level | Model | Proposed Method Performance | Literature work Performance | Accuracy Difference (%) |
| :--- | :--- | :--- | :--- | :--- |
| | | Accuracy (%) | Precision (%) | Recall (%) | F1-score (%) | Accuracy (%) | Precision (%) | Recall (%) | F1-score (%) | |
| 1 | UAS classifier | 98.61 | 99.49 | 97.74 | 98.6 | 91.61 | 90.23 | 90.6 | 90.42 | 7 |
| 2 | UAV controller classifier | 83.48 | 83.68 | 83.48 | 83.49 | 73.19 | 74.01 | 74.3 | 74.06 | 10.29 |
| 2 | UAV classifier | 91.61 | 91.99 | 91.61 | 91.61 | 82.49 | 82.18 | 82.5 | 82.25 | 9.12 |
| 3 | DJI Phantom classifier | 92.31 | 92.65 | 92.31 | 92.3 | 88.58 | 89.71 | 87.4 | 88.54 | 3.73 |
| 3 | DJI Inspire classifier | 98.48 | 100.00 | 95.05 | 97.46 | 95.28 | 95.98 | 94.45 | 95.21 | 3.2 |
| 3 | DJI MavicPro classifier | 87.18 | 89.62 | 84.48 | 86.98 | 82.39 | 88.12 | 75.17 | 81.13 | 4.79 |

*Fig. 9. 2D t-SNE visualization of all level classifier, (a) UAS classifier (UAV and UAV controller class), (b) UAV classifier (for five classes), (c) UAV controller classifier(for six classes), (d) DJI Inspire classifier (for flying and videoing modes), (e) DJI Mavicpro classifier (for flying and hovering modes), (f) DJI Phantom classifier (for flying and hovering modes).*

 O.O. Medaiyese, M. Ezuma, A.P. Lauf, and I. Guvenc, 2022. Wavelet transform analytics for RF-based UAV detection and identification system using machine learning. *Pervasive and Mobile Computing, 82*, p.101569.
 J. Swinney and J. C. Woods, "Unmanned aerial vehicle operating mode classification using deep residual learning feature extraction," *Aerospace*, vol. 8, no. 3, p. 79, 2021.
 N. Soltani, G. Reus-Muns, B. Salehihikouei, J. Dy, S. Ioannidis, and K. Chowdhury, "RF fingerprinting unmanned aerial vehicles with non- standard transmitter waveforms," *IEEE Trans. Veh. Technol.*, vol. 69, no. 12, pp. 15518-15531, Dec. 2020.
 O. O. Medaiyese, M. Ezuma, A. P. Lauf and A. A. Adeniran, "Hierar- chical Learning Framework for UAV Detection and Identification," in *IEEE Journal of Radio Frequency Identification*, vol. 6, pp. 176-188, 2022, doi: 10.1109/JRFID.2022.3157653.
 Olusiji Medaiyese, Martins Ezuma, Adrian Lauf, Ayodeji Adeniran, July 13, 2022, "Cardinal RF (CardRF): An Outdoor UAV/UAS/Drone RF Signals with Bluetooth and WiFi Signals Dataset", IEEE Dataport, doi: https://dx.doi.org/10.21227/1xp7-ge95.
 K. He, X. Zhang, S. Ren and J. Sun, "Deep residual learning for image recognition", *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, pp. 770-778, Jun. 2016.
 L. Van der Maaten and G. Hinton, "Visualizing data using t-SNE," *J. Mach. Learn. Res.*, vol. 9, no. 11, pp. 2579-2605, 2008.
 O. O. Medaiyese, M. Ezuma, A. P. Lauf and I. Guvenc, "Semi- supervised Learning Framework for UAV Detection," *2021 IEEE 32nd Annual International Symposium on Personal, Indoor and Mobile Radio Communications (PIMRC)*, Helsinki, Finland, 2021, pp. 1185-1190, doi: 10.1109/PIMRC50174.2021.9569452.

---
Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:55 UTC from IEEE Xplore. Restrictions apply.