# RESEARCH ARTICLE: Signal Preprocessing Technique With Noise-Tolerant for RF-Based UAV Signal Classification

**DAE-IL NOH¹, SEON-GEUN JEONG¹, HUU-TRUNG HOANG², QUOC-VIET PHAM³, (Member, IEEE), THIEN HUYNH-THE⁴, (Member, IEEE), MIKIO HASEGAWA⁵, (Member, IEEE), HIROO SEKIYA⁶, (Senior Member, IEEE), SUN-YOUNG KWON¹,⁷, (Member, IEEE), SANG-HWA CHUNG¹, (Member, IEEE), AND WON-JOO HWANG¹,⁷, (Senior Member, IEEE)**

¹Department of Information Convergence Engineering, Pusan National University, Busan 43241, South Korea
²University of Economics, Hue University, Hue 59000, Vietnam
³Korean Southeast Center for the 4th Industrial Revolution Leader Education, Pusan National University, Busan 43241, South Korea
⁴Faculty of Electrical and Electronics Engineering, Ho Chi Minh City University of Technology and Education, Ho Chi Minh City 70000, Vietnam
⁵Department of Electrical Engineering, Tokyo University of Science, Tokyo 162-8601, Japan
⁶Graduate School of Engineering, Chiba University, Chiba 263-8522, Japan
⁷Department of Biomedical Convergence Engineering, Pusan National University, Busan 43241, South Korea

Corresponding authors: Won-Joo Hwang (wjhwang@pusan.ac.kr) and Sun-Young Kwon (sy.kwon@pusan.ac.kr)

This work was supported in part by the Institute of Information and Communications Technology Planning and Evaluation (IITP) by the Korea Government (MSIT) through the Artificial Intelligence Convergence Research Center, Pusan National University, under Grant 2020-0-01450; in part by the BK21 Four, Korean Southeast Center for the 4th Industrial Revolution Leader Education; and in part by the National Research Foundation of Korea (NRF) Grant funded by the Korean Government (MSIT) under Grant NRF-2019R1I1A3A01060518.

---

***ABSTRACT*** **Since the beginning of the COVID-19 pandemic, the demand for unmanned aerial vehicles (UAVs) has surged owing to an increasing requirement of remote, noncontact, and technologically advanced interactions. However, with the increased demand for drones across a wide range of fields, their malicious use has also increased. Therefore, an anti-UAV system is required to detect unauthorized drone use. In this study, we propose a radio frequency (RF) based solution that uses 15 drone controller signals. The proposed method can solve the problems associated with the RF based detection method, which has poor classification accuracy when the distance between the controller and antenna increases or the signal-to-noise ratio (SNR) decreases owing to the presence of a large amount of noise. For the experiment, we changed the SNR of the controller signal by adding white Gaussian noise to SNRs of −15 to 15 dB at 5 dB intervals. A power-based spectrogram image with an applied threshold value was used for convolution neural network training. The proposed model achieved 98% accuracy at an SNR of −15 dB and 99.17% accuracy in the classification of 105 classes with 15 drone controllers within 7 SNR regions. From these results, it was confirmed that the proposed method is both noise-tolerant and scalable.**

***INDEX TERMS*** **Anti-drone systems, convolutional neural networks (CNNs), noise-tolerant, spectrograms, unmanned aerial vehicles (UAVs), UAV classification.**

---

## I. INTRODUCTION

Unmanned aerial vehicles (UAVs), including drones, are used for various purposes, such as delivery, agriculture, transportation, and communication. The potential uses of such vehicles are continually increasing. Additionally, social and commercial demands for remote technology have increased owing to the recent COVID-19 outbreak, and drones are being proposed as a noncontact solution for numerous applications. The approach in shows how backup transportation systems based on existing drone infrastructure can play an important role during the COVID-19 pandemic and similar situations. Additionally, the authors in proposed the application of drones for spraying disinfectants to combat the COVID-19 pandemic. However, behind such positive applications, the illegal uses of drones, such as for spying, drug trafficking, and terrorism, are increasing. Although restricted areas and laws for drone operation have been formed, the barriers for drone purchases have decreased, which can result in numerous risks. Therefore, an anti-drone system is required to block malicious use. It is designed to protect private property and personal privacy from the use of unauthorized drones. It comprises three stages: detection, identification, and decision making,. Among the three stages, detection and classification must precede the decision step of the anti-drone system such that it can operate normally and defend successfully.

Methods for detecting and classifying drones using various data sources have been presented. Radar and audio signals, vision data, and radio frequency (RF) signals are currently used for drone detection and classification. But even well-known radar-based detection approaches struggle to detect small size drones and their low-altitude flights. In audio-based detection, although the sound generated when the brushless DC motor rotates at high speed is analyzed and detected, it has an extremely short detection distance and is sensitive to noise. Additionally, vision-based detection has the disadvantage of being limited by fog, weather, and various obstacles. RF-based detection offers the advantages of higher reliability and superior performance. In the next section, the related studies are discussed in detail.

For RF-based drone classification, various methods using feature extraction, machine learning (ML), and deep learning (DL) have been proposed, For example, the drone controller signal classification, which uses the frequency domain feature (e.g., kurtosis, entropy, and variance) and various ML classifiers have been studied. The approaches in showed the drone signal classification using the frequency spectrum and a deep neural network (DNN). The authors in proposed a method for channelizing the frequency spectrum and classifying the drone signals using a one-dimensional convolution neural network (1DCNN). Recently, CNNs with high utility as image classifiers and spectrograms have been used to obtain high accuracy in RF-based drone classification tasks. However, these studies neither considered the cases wherein the signal-to-noise ratio (SNR) is lowered nor discussed the poor classification accuracy problem in the low SNR regions.

Hence, we propose a noise-tolerant classification method that obtains high classification accuracy even at an extremely low SNR. The proposed method performs data preprocessing based on a threshold after the spectrogram expressed by the power spectral density (PSD) is converted into a power-based spectrogram¹. This data preprocessing method can achieve high classification accuracy even in the low SNR region with added noise. The dataset used was the drone remote controller RF signals published on IEEE Dataport. The superiority of the proposed method was verified by classifying 15 different drone controllers and comparing the results with other studies that used the same dataset. Focusing on the goals and preprocessing techniques mentioned above, the main contributions of this study can be summarized as follows:

- Different methods of drone detection and classification were investigated. Particularly, the preprocessing method and its related classifier were reviewed in terms of RF-based drone classification. Moreover, to stimulate future research activities, we comprehensively reviewed the open-source drone RF data that have been released thus far.
- Related studies have pointed out the problem of poor classification accuracy at low SNR regions in the RF-based drone classification with noise. To create low-SNR environments, the correlation between drone classification and various methods (e.g., Gaussian noise, Rician fading, and Rayleigh fading) was investigated. Based on this, we proposed a power-based spectrogram and preprocessing technique to achieve noise-tolerance while maintaining high classification accuracy. The proposed method achieved 98% classification accuracy even at an SNR of −15 dB.

---
The associate editor coordinating the review of this manuscript and approving it for publication was Guillermo Valencia-Palomo.
This work is licensed under a Creative Commons Attribution 4.0 License. For more information, see https://creativecommons.org/licenses/by/4.0/
VOLUME 10, 2022
134785

---
IEEE Access
D.-I. Noh et al.: Signal Preprocessing Technique With Noise-Tolerant for RF-Based UAV Signal Classification

## II. RELATED WORK

In recent years, the illegal and malicious use of UAVs has raised various security, privacy, and safety concerns. Therefore, numerous methods have been studied to detect and identify drones. Based on the type of data source used, there are four major methods: radar, audio, vision, and RF-based. In this section, a brief description of these methods and related studies are presented.

### A. RADAR-BASED METHOD

A radar emits a strong electromagnetic wave and receives an echo wave reflected from the target object to determine its position and speed. It has certain advantages compared with vision-based approaches such as being unaffected by various weather conditions including fine dust, fog, clouds, and rain, However, because the radar cross section (RCS) is optimized for aircraft operating at high altitudes between 1 and 100 m², drones with low RCSs are difficult to detect using radar. Owing to these difficulties, research on the RCS of drones is also ongoing. For example, presented an RCS analysis of DJI Phantom 2, which is a type of quadcopter, at 10 GHz and compared and analyzed the RCS and micro-Doppler signatures of drones based on the number of propellers.

¹A spectrogram is a visualization method that combines the characteristics of a waveform and a spectrum, and is generally expressed by the PSD. However, a power-based spectrogram that expresses power is proposed in this paper.

---
134786
VOLUME 10, 2022

---
D.-I. Noh et al.: Signal Preprocessing Technique With Noise-Tolerant for RF-Based UAV Signal Classification
IEEE Access

In, a passive bistatic radar (PBR) system was developed for drone detection. It uses a digital television signal with bandwidths of 685 and 738 MHz for the transmitter and receiver, respectively, at a distance of 7.5 km. Filtering and correction are applied to the signal after it is received. For the former, an extended Kalman filter was used to effectively track the trajectory of a DJI Phantom 4. However, birds similar in size to drones were also detected, and distinguishing between them in both radar and PBR systems still remains a challenge.

In, the radar signals of planes, helicopters, quad-copters, birds, and stationary rotors corresponding to 11 classes were collected using an actual 9.5 GHz radar system. Based on these data, a micro-Doppler signature was introduced and then classified by applying ML techniques. The resulting accuracies obtained by the linear support vector machine (SVM), nonlinear SVM, and naive Bayes were 94.91%, 95.39%, and 93.6%, respectively.

Despite these studies, the radar-based method has several limitations such as RF regulation and implementation cost of radar systems.

### B. AUDIO-BASED METHOD

The audio-based method collects acoustic signals generated by the rotor rotating at a high speed when the drone that is flying contains acoustic sensors (e.g., a microphone). This signal-based method directly utilizes the unique characteristics of drones. In, the authors assumed that a quadrotor vehicle has a characteristic audio power spectrum fingerprint. The plot image learning method was applied with a fast Fourier transform (FFT) of the recorded audio signal to detect the part with the highest frequency amplitude at a fixed size. This method demonstrated a detection accuracy of 83%, and to apply a different algorithm, five consecutive chunks were extracted as a wave file from the same location and converted into a csv file through the FFT. Drones can be detected with an accuracy of 61% when the k-nearest neighbor (k-NN) algorithm is applied to the processed data.

In, a CNN was applied to the matrix generated through a short-time Fourier transform (STFT) of the recorded data. The final accuracy of the proposed model was 98.97%, thereby indicating a high classification accuracy. However, when the SNR was reduced to 5 dB by adding white Gaussian noise to the signal under the same conditions as in the model, the accuracy decreased to 75.87%.

Furthermore, the audio-based method requires an appropriate microphone array, it has an extremely short detection range compared to other methods, and is very vulnerable to ambient noise.

### C. VISION-BASED METHOD

Owing to the rapid development of computer hardware, numerous computations can be performed at high speeds; thus, it is possible to use a CNN when applying a vision method, which outperforms the traditional method.

In, images and various types of drone videos were extracted and collected from the internet, and the drones were distinguished using YOLOv3. The drones were classified based on the number of rotors, and the accuracy was calculated as the mean average precision (mAP). A final mAP of 0.74 was obtained.

In, a hemispherical camera array structure was constructed by using 30 cameras. The authors collected data from this structure and from the detected drones, helicopters, and airplanes using YOLOv2, wherein the detection accuracies were 52.13%, 90.47%, and 96.03%, respectively, which are extremely low for drone classification.

In, data points were obtained by applying Harris corner detection to an 1080p image, and ConvNet was applied to remove the background from these points. Subsequently, the initialized points were traced by using the Lucas-Kanade optical flow algorithm to form a trajectory. Using the trajectory generated by the tracking module as an input, the results were derived for trajectory lengths of 30 and 60 points, wherein the classification accuracies of the aircraft and drones were 90.68%, and 92.93%, respectively. Using these results, the authors proposed not only a real-time tracking and classification model but also a precision model.

Efficient and easy-to-use vision-based methods are widely used. However, their limitations are evident in that they require a high-resolution camera and line-of-sight (LOS), and they are severely affected by weather or obstacles. Additionally, partial occlusion and illumination are also problems that must be overcome.

### D. RADIO FREQUENCY-BASED METHOD

The RF-based detection method intercepts and utilizes the RF signal between the drone and the remote controller. Compared with other methods, it is free from constraints such as weather, obstacles, and LOS. The related studies are summarized in Table 1.

The authors of, who constructed the dataset used in this study, found energy transient from spectrograms and extracted RF fingerprints. After applying a neighbor component analysis (NCA), classification was conducted through various ML algorithms, such as the k-NN, discriminant analysis (DA), SVM, and neural network (NN). The k-NN classification accuracy for the 14 classes was 96.3%, but all four algorithms demonstrated accuracies of less than 50% at an SNR of 0 dB.

In, the authors added Wi-Fi and Bluetooth signals, which are within the ISM radio band used by the drone’s remote control signals, to the dataset used in. Interfering signals, Wi-Fi and Bluetooth, can be separated based on the bandwidth and modulation function. RF fingerprints were extracted from the drone controller RF signals and the controllers were classified using ML algorithms. The proposed method obtained a classification accuracy of 98.13% by using the k-NN at an SNR of 25 dB. This model also showed a limitation in that the accuracy was less than 60% at an SNR of 0 dB.

---
VOLUME 10, 2022
134787

---
IEEE Access
D.-I. Noh et al.: Signal Preprocessing Technique With Noise-Tolerant for RF-Based UAV Signal Classification

**TABLE 1. RF-based drone classification studies.**
| Literature | Dataset | Features | Classifier | # of UAVs | Accuracy | Noise Consideration |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| | Data set is not public | PSD | Logistic regression | 4+2 (Bluetooth, WiFi) | 99.9% | |
| | Dataset is not public | Spectrogram matrix applied with PCA | Several ML algorithms | 4 | approximately 98.3% | |
| | | Spectrogram matrix | DRNN | 9+1 (WiFi) | 87.6% at −10 dB SNR | ✓ |
| | | RF spectrum | DNN | 3+1 (Background) | 84.5% | |
| | | RF signal | 1DCNN | 3+1 (Background) | 85.8% | |
| | | RF signal | Fully-connected neural network, 1DCNN | 3+1 (Background) | 92.02% | |
| | | RF signal | 1DCNN | 3+1 (Background) | 92.5% | |
| | | Channelized RF spectrum | Multichannel CNN | 3+1 (Background) | 95.6% | |
| | | RF signal | Grouped 1DCNN | 3+1 (Background) | 98.5% | |
| | | Spectrogram RGB arrays | CNN | 3+1 (Background) | 100% | |
| | | Kurtosis, entropy, variance | Several ML algorithms | 15 | less than 50% at 0 dB SNR | ✓ |
| | | Shape factor, kurtosis, variance | Several ML algorithms | 15 | 40% at 0 dB SNR | ✓ |
| Proposed method | | Spectrogram matrix | CNN | 15 | 98% at −15 dB SNR | ✓ |

In, the authors who constructed the drone dataset classified the data into four classes (3 drones and 1 background) with an accuracy of 85.4% by using the frequency spectrum and DNN. Moreover, various studies,, have used 1DCNN and fully connected neural networks to frequency data, and obtained significantly good accuracy results such as 85.8%, 92.02%, and 92.5%, respectively. Reference achieved 95.6% accuracy by using multichannel CNN as a classifier. Reference leveraged grouped convolution instead of standard convolution and obtained 98.5% accuracy. However, these studies have a limitation in that they do not consider a real noise environment because they used a dataset that was collected in a laboratory.

To obtain higher classification accuracy, studies using the spectrogram have been recently proposed,,,. The authors of shared the dataset with, and obtained 100% accuracy by using the RGB values of the spectrogram and a CNN classifier. In, 99.9% accuracy was obtained by extracting features from the PSD, which is the spectrogram value, through CNN transfer learning and classifying them using logistic regression. In, 98.3% accuracy was obtained using ML algorithms in the spectrogram RGB matrix to which principal component analysis (PCA) was applied. These studies have shown that the signal classification performed using spectrograms demonstrate a very high accuracy compared to other studies. However, these studies still did not consider noise. To contribute to the broader research community, released the dataset and obtained almost 100% accuracy using a spectrogram and deep residual neural network (DRNN). However, they obtained an accuracy of 86.7% at −10 dB SNR owing to the effects of the added noise.

As evident from related studies, RF-based drone classification can achieve higher classification accuracy compared to other studies by combining spectrograms and artificial intelligence (AI). However, RF-based classification is limited in that the accuracy is reduced if the SNR value is low. To this end, an advanced preprocessing technique is proposed in this study for denoising the RF-based signals even if the SNR decreases by adding noise.

## III. SYSTEM MODEL AND DATASET

### A. SYSTEM MODEL

The system setup of the RF-based UAV classification model is shown in Fig. 1. It consists of a drone, drone controller, antenna, high-resolution oscilloscope, and computer. Because a drone uses an RF signal to communicate with the controller, the proposed method intercepts this signal to identify the drone controller. The received signal is divided into noise and signal components. To increase the classification difficulty, white Gaussian noise is added to the signal by calculating the power of the signal component. A power-based spectrogram image is generated, and the drone controller signal is classified using a CNN model.

### B. DATASET DESCRIPTION

In this study, the drone controller signal in the IEEE Dataport was used. The collected dataset comprised 15 drone controllers made by eight manufacturers, as listed in Table 2, and the time-voltage graph of each controller is shown in Fig. 2. The RF surveillance system continuously receives RF signals and records when the drone remote control (RC) RF signals are detected. For data collection, an oscilloscope (KEYSIGHT Technologies, Infiniium S-Series) with a maximum sampling frequency of 20 GHz, 2.4 GHz 24 dBi grid parabolic antenna (Tp-link, TL-ANT2424B), and low-noise amplifier (Fairview Microwave, 0.85 dB NF Input Protected Low Noise Amplifier) operating at 2.0-2.6 GHz were used. The metadata for the collected RF signal are listed in Table 3.

### C. NOISING PROCEDURE

The signal was captured in an indoor laboratory environment wherein the distance between the drone RC and antenna varied from 1-5 m. As shown in Fig. 2, because the size of

---
134788
VOLUME 10, 2022

---
D.-I. Noh et al.: Signal Preprocessing Technique With Noise-Tolerant for RF-Based UAV Signal Classification
IEEE Access

*FIGURE 1. System setup for RF-based UAV classification diagram flowchart.*

*FIGURE 2. Signal waveform of each drone controller.*

the noise section is constant, it is assumed that the environmental noise is fixed. As the distance between the drone controller and antenna increases, SNR decreases. For successful RF-based UAV classification, it is necessary to classify the controller signal accurately even at an extremely low SNR. The authors of evaluated the classification performance in Rician and Rayleigh fading environments. No significant deviation in the classification performance owing to the channel variation was observed. Therefore, in this study, the UAV controller was classified using the dataset wherein the SNR was artificially decreased by adding white Gaussian noise to the collected signal. To change the SNR, it is necessary to generate and add noise that matches the SNR to the signal of the drone controller. Noise is generated through the following process.

To add noise with respect to the SNR to the signal, the power ratio must be calculated. Hence, the noise and signal components within the signal must be separated. For this, the starting point of the transient state is provided in. Several methods can be used to determine the starting point of the transient state, wherein the mean change point detection method is applied. This method, expressed as J(k)

---
VOLUME 10, 2022
134789

---
IEEE Access
D.-I. Noh et al.: Signal Preprocessing Technique With Noise-Tolerant for RF-Based UAV Signal Classification

**TABLE 2. 15 classes of UAV controllers.**
| UAV ID (#) | Brand & Model |
| :--- | :--- |
| 1 | DJI Inspire 1 Pro |
| 2 | DJI Matrice 100 |
| 3 | DJI Matrice 600 |
| 4 | DJI Phantom 3 |
| 5 | DJI Phantom 4 Pro |
| 6 | FlySky FST6 |
| 7 | Futaba T8FG |
| 8 | Graupner MC32 |
| 9 | HobbyKing HKT6A |
| 10 | JetiDuplex DC16 |
| 11 | Spektrum DX5e |
| 12 | Spektrum DX6e |
| 13 | Spektrum DX6i |
| 14 | Spektrum JRX9303 |
| 15 | Turnigy 9X |

**TABLE 3. Metadata of captured RF signal.**
| Description | Value |
| :--- | :--- |
| Number of drone controllers | 15 |
| Sampling frequency | 20 GHz |
| Center frequency | 2.4 GHz |
| Number of signals/drone RC | 1000 |
| Number of samples/signal | 5 million |
| Time duration/signal | 0.25 ms |
| Average data size/signal | 7 MB |
| Dataset size | 124 GB |
| Data format | .mat |

*FIGURE 3. Finding the starting point of the transient state.*

in (1), has an advantage in that there is no need to define a threshold or perform nonparametric estimation to test the hypothesis.

When the collected signal is x[n] = x₁, x₂, ..., xN, a random point k is selected from k = 2, 3, . . ., N, where N is the length of the collected signal x, and k is divided as a reference point into x₁, x₂, ... xk−1 and xk, xk+1, ..., xN. At this time, k is the starting point of the transient state that minimizes the following J(k).
$$
\arg\min_{k \in N} J(k) = \left( \frac{1}{k-1} \right) \log \left( \sum_{i=1}^{k-1} (x_i - \bar{x}_{t1})^2 \right) + \left( \frac{1}{N-k+1} \right) \log \left( \frac{1}{N-k+1} \sum_{i=k}^{N} (x_i - \bar{x}_{t2})^2 \right) = (k-1) \log(\text{var}(X_{t1})) + (N-k+1) \log(\text{var}(X_{t2})). \quad (1)
$$
In this case, the optimal solution of arg min J(k), k*, is the minimizer. $x_{t1}$ is the interval before k, $x_{t2}$ is the interval after k, $\bar{x}_{t1}$ is the average of $x_{t1}$, and $\bar{x}_{t2}$ is the average of $x_{t2}$. As shown in Fig. 3, based on the point k* at which J(k) is the minimum, the separated $x_{t1}$ is the noise component, and $x_{t2}$ is the signal component.

Using the following process, it is possible to separate the signal of $x_{t2}$ from the signal and calculate the noise based on the SNR.
$$
P_{\text{signal}}[W] = \frac{1}{N-k+1} \sum_{i=k}^{N} |x[i]|^2 - \left| \frac{1}{n} \sum_{i=1}^{n} x_{t2}[i] \right|^2, \quad (2)
$$
where n is the length of $x_{t2}$. If the desired SNR is $\gamma_{\text{req}}[\text{dB}]$ and the current power is not on the dB scale, a conversion is required. The inverse transformation from the dB scale is as follows:
$$
\gamma_{\text{req}} = 10^{\frac{\gamma_{\text{req}}[\text{dB}]}{10}}. \quad (3)
$$
Because the SNR can be calculated the power level, the noise power can be obtained as a ratio using (2) and (3) as follows:
$$
P_{\text{noise}}[W] = \frac{P_{\text{signal}}}{\gamma_{\text{req}}} \quad (4)
$$
Finally, the magnitude corresponds to the square root of the power, and n[i], which is white Gaussian noise with a Gaussian normal distribution, is generated.
$$
n[i] = \sqrt{P_{\text{noise}}N(0, 1)}, \quad (5)
$$
where i = 1, 2, . . ., n, and n is the total number of samples of x[i]. Additionally, N is the Gaussian normal distribution, 0 is the average, and 1 is the variance. A signal with the desired SNR can be generated by adding this to the original signal x[i].
$$
y[i] = x[i] + n[i]. \quad (6)
$$
The waveform of the original signal with added noise is shown in Fig. 4.

## IV. IMAGE GENERATION

Because the threshold corresponding to denoising is applied to the spectrogram transformation, the spectrogram is described first. The image generation used for CNN training from the signal with a decreased SNR comprises two steps. The first is finding a threshold value and filtering the signal based on this value, and the second is creating an image through spectrogram transformation.

---
134790
VOLUME 10, 2022

---
D.-I. Noh et al.: Signal Preprocessing Technique With Noise-Tolerant for RF-Based UAV Signal Classification
IEEE Access

*FIGURE 4. Waveform and PSD-based spectrogram of DJI Inspire 1 Pro according to SNRs: (a) original, (b) 5 dB, and (c) −5 dB.*

### A. SPECTROGRAM TRANSFORMATION

Using Fourier transform (FT), time-series data can be analyzed within the frequency domain. However, this means that the information regarding time is lost. It is often necessary to analyze information in terms of both time and frequency, such as human voices and music. The STFT was devised for this purpose.

STFT divides the time-series signal into multiple signals by applying a window function and then applying the FT to the signal. In the FT, it is assumed that the period is infinite when calculating an aperiodic signal as a periodic signal. To derive this assumption, a window function i.e., a function whose ends converge at zero, is used. Representative examples include Hanning and Kaiser windows. The discrete-time STFT to be applied to the signal captured from this equation is as follows:
$$
\text{STFT}(x[n])(m,w) = X(m,w) = \sum_{n=-\infty}^{\infty} x[n]w[n-mR]e^{-jwn}, \quad (7)
$$
where x[n] are the data obtained by preprocessing the drone controller signal collected from the antenna, w[n] is the window function, m is the discrete time, and R is the hop size of the window. The result of this STFT is represented by the following matrix:
$$
S(m,w) = |X(m,w)|^2, \quad (8)
$$
where S(m,w) denotes the spectrogram. The spectrogram is expressed in terms of time, frequency, and PSD by mapping the absolute square of the STFT to a color bar. The transformed spectrogram is shown in Fig. 4. In the spectrogram transformation used in this study, to reduce the number of computations, the hop size is set to the length of the window function; thus, there is no overlapping part.

### B. FINDING THRESHOLD AND POWER-BASED SPECTROGRAM

In this study, the threshold value was calculated from the power spectrum and filtered through the power-based and not the PSD-based spectrogram. The results of FFT is only for the frequency domain; therefore, the proposed method can not be applied to FFT. However in case of wavelet transform (WT), it is expected that the proposed method can be applied if processing for variable-sized windows in WT is added.

The CNN is a type of DL architecture and is a powerful tool used for image classification. The data used for CNN training can be interpreted as an image, and the image can be interpreted as a three-dimensional matrix with rows, columns, and RGB channels. In an actual CNN, the RGB color value, position, and other image factors affect the results. However, a spectrogram matches the corresponding values obtained from the matrix using the (8) for the color map, as shown in Fig. 4, wherein it can be observed that the same color is biased as the noise increases. These results cause a decrease in the classification accuracy of a CNN. Hence, applying a threshold to a signal to which noise has been added has the same effect as that of filtering. In other words, the application of a threshold corresponds to denoising.

The average power of a signal can be obtained using the following equation when signal x(t) is given:
$$
P = \lim_{T \to \infty} \frac{1}{T} \int_{t_0 - T/2}^{t_0 + T/2} |x(t)|^2 dt, \quad (9)
$$
where T is the period wherein $t=t_0$ at an arbitrary time. From this, a window function is applied to manage the time constraint of the signal instead of the time constraint of the integral boundary of the frequency.
$$
P = \lim_{T \to \infty} \frac{1}{T} \int_{-\infty}^{\infty} |x_T(t)|^2 dt. \quad (10)
$$
This is a window function such that $x_T(t) = x(t)w(t)$, where w(t) is 1 at any point and 0 for the remaining interval. From this, the energy of the signal is obtained through Parseval’s theorem as follows:
$$
P = \lim_{T \to \infty} \frac{1}{T} \int_{-\infty}^{\infty} |X_T(f)|^2 df, \quad (11)
$$
where $X_T(f)$ denotes the FT of $x_T(t)$. Thus, the PSD can be obtained as follows:
$$
S_{xx}(f) = \lim_{T \to \infty} \frac{1}{T} |X_T(f)|^2. \quad (12)
$$
However, because $S_{xx}(f)$ is a density function, to obtain the power spectrum, the power of the corresponding frequency can be calculated by applying an integral with respect to the frequency within a short section.
$$
P_{\text{limited},i} = 2 \int_{f_i}^{f_{i+1}} S_{xx}(f) df, \quad (13)
$$
where $P_{\text{limited},i}$ is the power corresponding to two frequency bandwidths, i.e., the i-th segments $f_i$ and $f_{i+1}$, which satisfies the condition $0 < f_i < f_{i+1}$. The same amount of power is calculated from the positive and negative frequencies, which can explain the 2 in front of the equation. Fig. 5(a) shows the power spectrum of the DJI Inspire 1 Pro, and Fig. 5(b) shows the power spectrum of a signal with noise added to an SNR of −10 dB. These two graphs show that the drone controller signal uses a bandwidth of 2.4 GHz, and the power of the drone controller signal is the largest. From (8), it is confirmed that the PSD-based spectrogram is a graph wherein the value of the absolute square of the STFT, i.e., the PSD, is mapped to the color bar. The (13) is applied to the (8) to generate a power-based spectrogram by integrating it over a limited frequency in the PSD-based spectrogram.
$$
S_p(m,w) = 2 \int_{f_i}^{f_{i+1}} S(m,w) df, \quad (14)
$$
where $S_p(m,w)$ is a power-based spectrogram obtained by converting the of mapping the time-frequency-power to the
color bar from the PSD-based spectrogram, which was originally expressed as a time-frequency PSD. The frequencies $f_i$ and $f_{i+1}$ satisfy the condition $f_i < f_{i+1}$.

As shown in Fig. 6(b), a power-based spectrogram can be obtained from the PSD-based spectrogram. However, because the image in Fig. 6(a) is color-biased, as shown in Fig. 4(c), the threshold value should be applied as follows:
$$
\gamma_t = 10 \log_{10} \left( \frac{1}{n} \sum_{i=1}^n P_{\text{limited},i} \right), \quad (15)
$$
where $\gamma_t$ denotes the threshold value.² This threshold is applied to the power-based spectrogram as follows:
$$
S_p'(m,w) = \begin{cases} S_p(m,w), & \text{if } S_p(m,w) > \gamma_t, \\ \gamma_t, & \text{otherwise,} \end{cases} \quad (16)
$$
If the threshold value used at this time is such that the drone controller signal power is larger than that of the other signal, the average signal power can be set as the threshold value for an expedited calculation. The power-based spectrogram, to which a threshold is applied, is shown in Fig. 6(b). By applying a threshold to the original and difficult-to-classify signals, the signal and noise can be separated more clearly. Finally, the complexity of the proposed method can be obtained as follows. The complexity of FFT is O(N log₂N). In case of STFT, FFT is required for each of the samples in the window function, the complexity is O(N_w N log₂N) where N_w is number of sampling points in the window function. In addition, power-based spectrogram integrates over the frequency segments from the spectrogram using STFT. Therefore, the complexity of proposed method is O(N_seg |N_w N log₂N|²) where N_seg is the number of points in the frequency segments.

## V. CNN-BASED DRONE CONTROLLER RF CLASSIFICATION

In this section, an efficient low-cost and highly-accurate CNN model is presented to classify the RF signals of 15 different drone controllers. The proposed CNN model consists of three two-dimensional convolution (conv2D) layers and three max pooling (maxpool) layers. Our goal is to propose a drone classification method with the possibility to be implemented in real-time. Therefore, we consider the basic CNN architectures because of the time complexity. This optimized model has been tested through various approaches, such as depthwise convolution and dilated convolution. The overall architecture of the CNN model is illustrated in Fig. 7, and the details of the model configuration and the number of parameters are listed in Table 4. The 356 × 452 × 3 sized power-based spectrogram image that was preprocessed in the previous section is used as the input layer of the CNN model. The first conv2D layer is a 64-channel filter with a stride

---
²The key function of a real-time spectrum analyzer is parallel sampling and FFT calculation. The data sampling continues while the calculations are performed. Furthermore, the real-time operation of STFT was studied in. Using these methods, threshold value can be calculated in a semi-adaptive manner.

---
VOLUME 10, 2022
134792

---
D.-I. Noh et al.: Signal Preprocessing Technique With Noise-Tolerant for RF-Based UAV Signal Classification
IEEE Access

*FIGURE 5. Power spectrum graph of DJI Inspire 1 Pro: (a) original level and (b) −10 dB.*

*FIGURE 6. Power-based spectrogram: (a) without and (b) with a threshold.*

*FIGURE 7. Overall CNN architecture.*

of (2, 2) and size of 2 × 2. Subsequently, the information passes through a batch normalization layer and rectified linear unit (ReLU), which is an activation layer. From the CNN architecture, the number of learnable parameters is 0.485 M and the number of floating-point operations (FLOPs) is 507.87 M.
$$
f(x) = \begin{cases}
0, & \text{if } x < 0, \\
x, & \text{if } x \ge 0.
\end{cases} \quad (17)
$$
ReLU outputs a zero if the input value is less than zero, and outputs the input value otherwise. The data then pass through the maxpool layer before being connected to a fully connected layer through 128 and 256 conv2D and maxpool layers, respectively. The last layer of the CNN model is the classification layer, for which the model uses the softmax function.
$$
\hat{y}_i(x) = \frac{\exp(a_i(x))}{\sum_{j=1}^n \exp(a_j(x))}. \quad (18)
$$

---
VOLUME 10, 2022
134793

---
IEEE Access
D.-I. Noh et al.: Signal Preprocessing Technique With Noise-Tolerant for RF-Based UAV Signal Classification

**TABLE 4. CNN model configuration.**
| Layer | Network Description | Output size |
| :--- | :--- | :--- |
| Input | Spectrogram Image | 356 × 452 × 3 |
| conv2D1 | 64 filters 2 × 2, stride 2 | 178 × 226 × 64 |
| maxpool1 | pool size 2 × 2, stride 2 | 89 × 113 × 64 |
| conv2D2 | 128 filters 3 × 3, stride 2 | 45 × 57 × 128 |
| maxpool2 | pool size 2 × 2, stride 2 | 23 × 29 × 128 |
| conv2D3 | 256 filters 3 × 3, stride 2 | 12 × 15 × 256 |
| maxpool3 | pool size 2 × 2, stride 2 | 6 × 8 × 256 |
| fc | fully connected layer | 1 × 1 × 15 |
| softmax | softmax layer | |
| # params | | 0.485 × 10⁶ |
| FLOPS | | 507.87 × 10⁶ |

where aᵢ(x) is the i-th value of the fully connected layer and n is the total number of classes. As shown in the above equation, the softmax function is a method of expressing a probability between zero and one from the values of the fully connected layer. From this, the resulting value is predicted using a one-hot vector based on one-hot encoding.
$$
v_i = \text{result}(\hat{y}_i(x)) = \begin{cases}
1, & \text{if } v_i = \max(v), \\
0, & \text{otherwise}.
\end{cases} \quad (19)
$$
In (19), v is a vector obtained through one-hot encoding of the output result of the softmax function, and i denotes the i-th element of vector v. The length of v can reach up to the total number of classes, which means that an i-th of 1 is the predicted class.

To allow for learning in the correct direction, it is necessary to ensure that the predicted values match the actual values of the cost function. Therefore, training should be directed toward minimizing the cost function. In this study, the following cross-entropy loss function is used:
$$
L(W) = -\sum_{i=1}^N y_i \log(\hat{y}_i), \quad (20)
$$
where W is the weight vector of the model; yᵢ is the true label, i.e., the previously obtained one-hot vector; ŷᵢ is the predicted label, i.e., the result of the softmax function; and N is the total number of classes. If the predicted result and actual value are the same, it converges to zero, and if they are completely opposite, it diverges toward infinity by −log. Learning proceeds such that the loss function is minimized.

## VI. SIMULATION RESULT

In this section, the performance of the proposed model is evaluated. The simulation settings for the CNN model are listed in Table 5. Particularly, the optimizer was a stochastic gradient descent with momentum, the momentum factor was 0.9, L2 regularization factor was 0.0001, maximum number of epochs for training was 100, initial learning rate was 0.01 (which decreased to 0.001 after 60 epochs for better training convergence), and the mini-batch size was 16.

For the simulations, noise was added to the drone controller signals based on the SNR. A power-based spectrogram image
**TABLE 5. CNN model option.**
| Option | Value |
| :--- | :--- |
| Optimizer | Gradient descent with momentum |
| Momentum factor | 0.9 |
| L2 regularization factor | 0.0001 |
| Maximum epoch | 100 |
| Learning rate | 0.01 |
| Learning rate drop factor | 0.1 |
| Learning rate drop epoch | 60 |
| Mini-batch size | 16 |

was generated from the signal with the added noise, with a threshold applied based on the value of the power spectrum. The signal was changed by adding noise in 5 dB steps, from an SNR of −15 to 15 dB. Two simulations were performed to confirm whether the proposed model can accurately classify the physical signals, even at a low SNR.

First, to verify the effectiveness of the proposed method in a real noise environment, white Gaussian noise was added and the accuracy of the PSD-based spectrogram was compared with that of the power-based spectrogram based on the threshold value. Additionally, the classification accuracy was compared with the results of other studies that used the same dataset.

Thereafter, the last experiment confirmed whether the proposed method is scalable as an RF-based approach for drone detection. In the first setup, data from the same drone controller in different SNR regions were added as a new class. In the second setup, 105 classes were generated by combining the 15 drone controller signals with seven different SNR regions. The result of the classification experiment indicates that the proposed method can classify the same drone controller even if it has a different SNR value.

The power-based spectrogram used in the simulation contained 300 images per class. All simulation settings were the same as that for the model and options described above. 80% of the dataset was used for training, and the remaining 20% was used as the test set for verification. Accuracy was used as an indicator to evaluate the performance.

### A. PERFORMANCE EVALUATION

In this experiment, the performance of the proposed method was evaluated. The PSD-based spectrogram was used to compare and evaluate the performance of the power-based spectrogram. The quality and number of datasets used are critical to obtain accurate outcomes. Accordingly, 300 spectrogram images were selected as the training set and testing set. The results of this experiment are shown in Fig. 8. The PSD-based spectrogram images obtained a high classification accuracy of 94.92% at an SNR of 15 dB; however, when the SNR was reduced, the classification accuracy decreased significantly. Contrastingly, in the power-based spectrogram to which the threshold value was applied, the classification accuracy did not fall below 98%, even at an SNR of −15 dB.

---
134794
VOLUME 10, 2022

---
D.-I. Noh et al.: Signal Preprocessing Technique With Noise-Tolerant for RF-Based UAV Signal Classification
IEEE Access

*FIGURE 8. Power-based spectrogram and PSD-based spectrogram classification results for various SNR regions.*

*FIGURE 9. Accuracy comparison.*

Because the drone controller and noise homogenize to their signal levels, running CNN based algorithms using PSD-based spectrogram for classification exhibits poor accuracy. Hence these results validate that the proposed method is noise-tolerant, robust, and scalable when a power-based spectrogram and threshold value are used.

To confirm whether the proposed method achieves high accuracy, the classification accuracies obtained in and were compared using the same drone controller signals. Both studies applied various ML techniques by extracting a feature called an RF fingerprint from the energy transient. In, k-NN, DA, SVM, and NN techniques were used, and in, k-NN, DA, and random forest (RandF) were applied. A comparison of these two studies is shown in Fig. 9. Although the classification accuracy is high when the SNR is high, the classification accuracy evidently decreases significantly as the SNR decreases. This indicates that there are serious difficulties in extracting the valid features owing to the effects of additive noise.

Through these experiments, it can be confirmed that even at a low SNR, the classification accuracy of the proposed method is sufficiently reliable.

**TABLE 6. Classification results.**
| SNR | #Class (Mispredicted number) | Total |
| :--- | :--- | :--- |
| 15 dB | #3 (4), #5 (1) | 5 |
| 10 dB | #5 (3) | 3 |
| 5 dB | #3 (2), #5 (4), #10 (4) | 10 |
| 0 dB | #5 (5), #10 (1) | 6 |
| −5 dB | #5 (7), #12 (2) | 9 |
| −10 dB | #3 (1), #5 (3), #10 (1) | 5 |
| −15 dB | #5 (5), #8 (6), #12 (3) | 14 |

### B. ANALYSIS OF THE RF-BASED DRONE CLASSIFIER AFTER ADDING A NEW CLASS

Through this experiment, the scalability potential of the proposed method as the RF-based drone classifier was confirmed in a real application. The classifier should be capable of maintaining the same performance even if a new model is added and classifying drone controllers in different SNR environments. Therefore, the signals of the 15 controllers were divided into 105 classes across 7 SNR regions, and the classification results of the proposed method were confirmed through preprocessing. For the training, 300 images from each class were used; therefore, 31,500 images were applied. Among the classification results, the incorrectly predicted classes are listed in Table 6, and the controller corresponding to the class is arranged in Table 2.

Out of 6,300 test set images, 52 incorrect predictions were made and a classification accuracy of 99.17% was obtained. Particularly, the classifier often incorrectly predicted DJI Phantom 4 Pro, which corresponds to #5 in Table 2, as DJI Matrice 600, which corresponds to #3, for each class. It was analyzed that this is because the two controllers are made by the same brand, and both their waveforms and spectrograms have similar values and patterns, respectively. However, the classification accuracy of 99.17% for 105 classes demonstrates that the proposed preprocessing technique that uses a power-based spectrogram is robust to denoising methods.

Thus, from the two experiments, it was demonstrated that the proposed method is noise-tolerant and scalable.

## VII. CONCLUSION

In this study, a data preprocessing technique for classifying the RF signal of a drone controller for UAV identification was proposed. The SNR of the signal was changed by adding white Gaussian noise. The proposed method used a power-based spectrogram instead of the existing PSD-based spectrogram for learning. Additionally, to reduce the effect of noise, the threshold value was calculated from the power spectrum and then applied to the power-based spectrogram to increase the classification accuracy. The proposed method can classify the drone controller signal with an accuracy exceeding 98%, even at an extremely low SNR of −15 dB.

Through the results of our proposed method, it is expected that classification performed using the spectrogram can identify drones with higher accuracy. However, classification of multiple drone controller signals remains challenging.

---
VOLUME 10, 2022
134795

---
IEEE Access
D.-I. Noh et al.: Signal Preprocessing Technique With Noise-Tolerant for RF-Based UAV Signal Classification

In future research, such classification at the physical signal level will be investigated for achieving real-time detection similar to radar.

## REFERENCES

 H. Shakhatreh, A. H. Sawalmeh, A. Al-Fuqaha, Z. Dou, E. Almaita, I. Khalil, N. S. Othman, A. Khreishah, and M. Guizani, “Unmanned aerial vehicles (UAVs): A survey on civil applications and key research challenges,” *IEEE Access*, vol. 7, pp. 48572–48634, 2019.
 N. H. Motlagh, T. Taleb, and O. Arouk, “Low-altitude unmanned aerial vehicles-based Internet of Things services: Comprehensive survey and future perspectives,” *IEEE Internet Things J.*, vol. 3, no. 6, pp. 899–922, Dec. 2016.
 M. Kunovjanek and C. Wankmüller, “Containing the COVID-19 pandemic with drones—Feasibility of a drone enabled back-up transport system,” *Transp. Policy*, vol. 106, pp. 141–152, Jun. 2021.
 Á. Restás, I. Szalkai, and G. Óvári, “Drone application for spraying disinfection liquid fighting against the COVID-19 pandemic—Examining drone-related parameters influencing effectiveness,” *Drones*, vol. 5, no. 3, p. 58, Jul. 2021.
 M. A. Siddiqi, C. Iwendi, K. Jaroslava, and N. Anumbe, “Analysis on security-related concerns of unmanned aerial vehicle: Attacks, limitations, and recommendations,” *Math. Biosci. Eng.*, vol. 19, no. 3, pp. 2641–2670, 2022.
 S.-H. Park and K.-H. Lee, “Developing criteria for invasion of privacy by personal drone,” in *Proc. Int. Conf. Platform Technol. Service (PlatCon)*, Feb. 2017, pp. 1–7.
 M. Sinclair, “Death from above: How criminal organizations’ use of drones threatens Americans,” Brookings, Washington, DC, USA, Tech. Rep., Mar. 2021.
 N. Awadalla, L. Barrington, and A. Cornwell, “Shrapnel injures 12 at Saudi Abha airport as drone intercepted,” Reuters, London, U.K., Tech. Rep., Feb. 2022.
 X. Shi, C. Yang, W. Xie, C. Liang, Z. Shi, and J. Chen, “Anti-Drone system with multiple surveillance technologies: Architecture, implementation, and challenges,” *IEEE Commun. Mag.*, vol. 56, no. 4, pp. 68–74, Apr. 2018.
 Y. N. Jurn, S. A. Mahmood, and J. A. Aldhaibani, “Anti-drone system based different technologies: Architecture, threats and challenges,” in *Proc. 11th IEEE Int. Conf. Control Syst., Comput. Eng. (ICCSCE)*, Aug. 2021, pp. 114–119.
 P. Cisar, R. Pinter, S. M. Cisar, and M. Gligorijevic, “Principles of anti-drone defense,” in *Proc. 11th IEEE Int. Conf. Cognit. Infocommunications (CogInfoCom)*, Sep. 2020, pp. 000019–000026.
 S. Al-Emadi, A. Al-Ali, and A. Al-Ali, “Audio-based drone detection and identification using deep learning techniques with dataset enhancement through generative adversarial networks,” *Sensors*, vol. 21, no. 15, p. 4953, Jul. 2021.
 G. Ding, Q. Wu, L. Zhang, Y. Lin, T. A. Tsiftsis, and Y.-D. Yao, “An amateur drone surveillance system based on the cognitive Internet of Things,” *IEEE Commun. Mag.*, vol. 56, no. 1, pp. 29–35, Jan. 2018.
 M. Ezuma, F. Erden, C. K. Anjinappa, O. Ozdemir, and I. Guvenc, “Micro-UAV detection and classification from RF fingerprints using machine learning techniques,” in *Proc. IEEE Aerosp. Conf.*, Mar. 2019, pp. 1–13.
 M. F. Al-Sa’d, A. Al-Ali, A. Mohamed, T. Khattab, and A. Erbad, “RF-based drone detection and identification using deep learning approaches: An initiative towards a large open source drone database,” *Future Gener. Comput. Syst.*, vol. 100, pp. 86–97, Nov. 2019.
 M. S. Allahham, T. Khattab, and A. Mohamed, “Deep learning for RF-based drone detection and identification: A multi-channel 1-D convolutional neural networks approach,” in *Proc. IEEE Int. Conf. Informat., IoT, Enabling Technol. (ICIoT)*, Feb. 2020, pp. 112–117.
 M. Ezuma, F. Erden, C. K. Anjinappa, O. Ozdemir, and I. Guvenc, “Drone remote controller RF signal dataset,” Tech. Rep., 2020, doi: 10.21227/ss99-8d56.
 B. Taha and A. Shoufan, “Machine learning-based drone detection and classification: State-of-the-art in research,” *IEEE Access*, vol. 7, pp. 138669–138682, 2019.
 M. Jahangir and C. Baker, “Robust detection of micro-UAS drones with L-band 3-D holographic radar,” in *Proc. Sensor Signal Process. Defence (SSPD)*, Sep. 2016, pp. 1–5.
 A. Schroder, M. Renker, U. Aulenbacher, A. Murk, U. Boniger, R. Oechslin, and P. Wellig, “Numerical and experimental radar cross section analysis of the quadrocopter DJI phantom 2,” in *Proc. IEEE Radar Conf.*, Oct. 2015, pp. 463–468.
 P. J. Speirs, A. Schroder, M. Renker, P. Wellig, and A. Murk, “Comparisons between simulated and measured X-band signatures of quad-, hexa- and octocopters,” in *Proc. 15th Eur. Radar Conf. (EuRAD)*, Sep. 2018, pp. 325–328.
 Y. Liu, X. Wan, H. Tang, J. Yi, Y. Cheng, and X. Zhang, “Digital television based passive bistatic radar system for drone detection,” in *Proc. IEEE Radar Conf. (RadarConf)*, May 2017, pp. 1493–1497.
 M. Ritchie, F. Fioranelli, H. Griffiths, and B. Torvik, “Monostatic and bistatic radar measurements of birds and micro-drone,” in *Proc. IEEE Radar Conf. (RadarConf)*, May 2016, pp. 1–5.
 P. Molchanov, K. Egiazarian, J. Astola, R. I. A. Harmanny, and J. J. M. de Wit, “Classification of small UAVs and birds by micro-Doppler signatures,” in *Proc. Eur. Radar Conf.*, 2013, pp. 172–175.
 S. Park, H. T. Kim, S. Lee, H. Joo, and H. Kim, “Survey on anti-drone systems: Components, designs, and challenges,” *IEEE Access*, vol. 9, pp. 42635–42659, 2021.
 J. Kim, C. Park, J. Ahn, Y. Ko, J. Park, and J. C. Gallagher, “Real-time UAV sound detection and analysis system,” in *Proc. IEEE Sensors Appl. Symp. (SAS)*, Mar. 2017, pp. 1–5.
 Y. Seo, B. Jang, and S. Im, “Drone detection using convolutional neural networks with acoustic STFT features,” in *Proc. 15th IEEE Int. Conf. Adv. Video Signal Based Surveill. (AVSS)*, Nov. 2018, pp. 1–6.
 C. J. Swinney and J. C. Woods, “RF detection and classification of unmanned aerial vehicles in environments with wireless interference,” in *Proc. Int. Conf. Unmanned Aircr. Syst. (ICUAS)*, Jun. 2021, pp. 1494–1498.
 C. Xu, B. Chen, Y. Liu, F. He, and H. Song, “RF fingerprint measurement for detecting multiple amateur drones based on STFT and feature reduction,” in *Proc. Integr. Commun. Navigat. Surveill. Conf. (ICNS)*, Sep. 2020, pp. 1–4.
 S. Basak, S. Rajendran, S. Pollin, and B. Scheers, “Drone classification from RF fingerprints using deep residual nets,” in *Proc. Int. Conf. Commun. Syst. Netw. (COMSNETS)*, Jan. 2021, pp. 548–555.
 S. Basak. (2021). *Drone Signals*. GitHub. [Online]. Available: https://github.com/sanjoy-basak/dronesignals
 M. S. Allahham, M. F. Al-Sa’d, A. Al-Ali, A. Mohamed, T. Khattab, and A. Erbad, “DroneRF dataset: A dataset of drones for RF-based detection, classification and identification,” *Data Brief*, vol. 26, Oct. 2019, Art. no. 104313. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S2352340919306675
 S. Al-Emadi and F. Al-Senaid, “Drone detection approach based on radio-frequency using convolutional neural network,” in *Proc. IEEE Int. Conf. Informat., IoT, Enabling Technol. (ICIoT)*, Feb. 2020, pp. 29–34.
 E. S. Basan, M. D. Tregubenko, N. N. Mudruk, and E. S. Abramov, “Analysis of artificial intelligence methods for detecting drones based on radio frequency activity,” in *Proc. 15th Int. Sci.-Tech. Conf. Actual Problems Electron. Instrum. Eng. (APEIE)*, Nov. 2021, pp. 238–242.
 R. Akter, V.-S. Doan, G. B. Tunze, J.-M. Lee, and D.-S. Kim, “RF-based UAV surveillance system: A sequential convolution neural networks approach,” in *Proc. Int. Conf. Inf. Commun. Technol. Converg. (ICTC)*, Oct. 2020, pp. 555–558.
 T. Huynh-The, Q.-V. Pham, T.-V. Nguyen, D. B. D. Costa, and D.-S. Kim, “RF-UAVNet: High-performance convolutional network for RF-based drone surveillance systems,” *IEEE Access*, vol. 10, pp. 49696–49707, 2022.
 M. Mokhtari, J. Bajcetic, B. Sazdic-Jotic, and B. Pavlovic, “RF-based drone detection and classification system using convolutional neural network,” in *Proc. 29th Telecommun. Forum (TELFOR)*, Nov. 2021, pp. 1–4.
 B. Sazdic-Jotic, I. Pokrajac, J. Bajcetic, and B. Bondzulic. (Nov. 2020). *VTI_Droneset*. Mendeley Data. [Online]. Available: https://data.mendeley.com/datasets/s6tgnnp5n2/1
 M. Ezuma, F. Erden, C. K. Anjinappa, O. Ozdemir, and I. Guvenc, “Detection and classification of UAVs using RF fingerprints in the presence of Wi-Fi and Bluetooth interference,” *IEEE Open J. Commun. Soc.*, vol. 1, pp. 60–76, 2020.
 D. K. Behera and A. Bazil Raj, “Drone detection and classification using deep learning,” in *Proc. 4th Int. Conf. Intell. Comput. Control Syst. (ICCCS)*, May 2020, pp. 1012–1016.
 J. Redmon and A. Farhadi, “YOLOv3: An incremental improvement,” 2018, arXiv:1804.02767.

---
134796
VOLUME 10, 2022

---
D.-I. Noh et al.: Signal Preprocessing Technique With Noise-Tolerant for RF-Based UAV Signal Classification
IEEE Access

 H. Liu, F. Qu, Y. Liu, W. Zhao, and Y. Chen, “A drone detection with aircraft classification based on a camera array,” *IOP Conf. Ser., Mater. Sci. Eng.*, vol. 322, Mar. 2018, Art. no. 052005.
 J. Redmon and A. Farhadi, “YOLO9000: Better, faster, stronger,” in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Jun. 2016, pp. 7263–7271.
 V.-P. Thai, W. Zhong, T. Pham, S. Alam, and V. Duong, “Detection, tracking and classification of aircraft and drones in digital towers using machine learning on motion patterns,” in *Proc. Integr. Commun., Navigat. Surveill. Conf. (ICNS)*, 2019, pp. 1–8.
 B. D. Lucas and T. Kanade, “An iterative image registration technique with an application to stereo vision,” in *Proc. 7th Int. Joint Conf. Artif. Intell.*, vol. 2. San Francisco, CA, USA: Morgan Kaufmann, 1981, pp. 674–679.
 J. Yang, L. Luo, J. Qian, Y. Tai, F. Zhang, and Y. Xu, “Nuclear norm based matrix regression with applications to face recognition with occlusion and illumination changes,” *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 39, no. 1, pp. 156–171, Jan. 2017.
 N. Soltanieh, Y. Norouzi, Y. Yang, and N. C. Karmakar, “A review of radio frequency fingerprinting techniques,” *IEEE J. Radio Freq. Identificat.*, vol. 4, no. 3, pp. 222–233, Sep. 2020.
 L. Huang, M. Gao, C. Zhao, and X. Wu, “Detection of Wi-Fi transmitter transients using statistical method,” in *Proc. IEEE Int. Conf. Signal Process., Commun. Comput. (ICSPCC)*, Aug. 2013, pp. 1–5.
 R. Klein, M. A. Temple, M. J. Mendenhall, and D. R. Reising, “Sensitivity analysis of burst detection and RF fingerprinting classification performance,” in *Proc. IEEE Int. Conf. Commun.*, Jun. 2009, pp. 1–5.
 J. B. Allen and L. Rabiner, “A unified approach to short-time Fourier analysis and synthesis,” *Proc. IEEE*, vol. 65, no. 11, pp. 1558–1564, Nov. 1977.
 P. Putranto, T. Qurrachman, W. Desvasari, P. Daud, Y. N. Wijayanto, D. Mahmudin, D. P. Kurniadi, A. N. Rahman, S. Hardiati, A. Setiawan, F. Darwis, and E. J. Pristianto, “Performance comparison of Blackman, Bartlett, Hanning, and Kaiser window for radar digital signal processing,” in *Proc. 4th Int. Conf. Inf. Technol., Inf. Syst. Electr. Eng. (ICITISEE)*, 2019, pp. 391–394.
 F. Ramian, “Implementation of real-time spectrum analysis,” Rohde & Schwarz, Munich, Germany, Tech. Rep., 2011.
 S. Zhang, D. Yu, and S. Sheng, “A discrete STFT processor for real-time spectrum analysis,” in *Proc. IEEE Asia Pacific Conf. Circuits Syst.*, Dec. 2006, pp. 1943–1946.
 A. V. Oppenheim, R. W. Schafer, and J. R. Buck, *Discrete-Time Signal Processing*, 2nd ed. Englewood Cliffs, NJ, USA: Prentice-Hall, 1999.
 J.-J. Ding and H. Hu, “Low complexity time-frequency analysis methods for efficient implementation,” in *Proc. IEEE Int. Conf. Consum. Electron.*, May 2014, pp. 195–196.

---

**DAE-IL NOH** received the double B.S. degrees in electrical engineering and biomaterials science and the M.S. degree from the Department of Information Convergence Engineering, Pusan National University, Busan, South Korea, in 2018 and 2022, respectively, where he is currently pursuing the Ph.D. degree with the Department of Information Convergence Engineering. His research interests include signal processing, machine learning, deep learning, wireless communication, and quantum-inspired evolutionary computation.

**SEON-GEUN JEONG** received the B.S. degree in electrical engineering from Pusan National University, Busan, South Korea, in 2017, where he is currently pursuing the integrated Ph.D. degree with the Department of Information Convergence Engineering. His current research interests include wireless networks, quantum-inspired evolutionary computation, quantum communication, and quantum information.

**HUU-TRUNG HOANG** received the B.S. degree in management information systems from the University of Economics, Hue University, Vietnam, in 2013, and the M.S. and Ph.D. degrees in information and communication systems from Inje University, South Korea, in 2018 and 2021, respectively. He is currently with the University of Economics–Hue University. His research interests include machine learning, deep learning, computer vision, and information systems in wireless communication, healthcare, finance, and economics.

**QUOC-VIET PHAM** (Member, IEEE) received the B.S. degree in electronics and telecommunications engineering from the Hanoi University of Science and Technology, Vietnam, in 2013, and the Ph.D. degree in telecommunications engineering from Inje University, Republic of Korea, in 2017. Since February 2020, he has been working as a Research Professor at Pusan National University, Republic of Korea. He specializes in applying convex optimization, game theory, and machine learning to analyze and optimize edge computing and future wireless communications. He was granted the Korea NRF funding for outstanding young researchers, from 2019 to 2024.
He was a recipient of the Best Ph.D. Dissertation Award in engineering from Inje University, in 2017, the Top Reviewer Award from the IEEE TRANSACTIONS ON VEHICULAR TECHNOLOGY, in 2020, and the Golden Globe Award 2021 from the Ministry of Science and Technology (Vietnam). He was a TPC/TPC Chair of leading conferences, including IEEE ICC, IEEE VTC, and EAI GameNets. He is an Editor of the *Journal of Network and Computer Applications* (Elsevier), *Scientific Reports* (Nature), and *Frontiers in Communications and Networks*; and a Lead Guest Editor of the IEEE INTERNET OF THINGS JOURNAL.

**THIEN HUYNH-THE** (Member, IEEE) received the B.S. degree in electronics and telecommunication engineering from the Ho Chi Minh City University of Technology and Education, Vietnam, in 2011, and the Ph.D. degree in computer science and engineering from Kyung Hee University (KHU), South Korea, in 2018.
From March 2018 to August 2018, he was a Postdoctoral Researcher with the Ubiquitous Computing Laboratory, KHU. From September 2018 to May 2022, he was a Postdoctoral Researcher with the ICT Convergence Research Center, Kumoh National Institute of Technology, South Korea. He is currently a Lecturer at the Department of Computer and Communication Engineering, Ho Chi Minh City University of Technology and Education (HCMUTE), Vietnam. His current research interests include digital image processing, radio signal processing, computer vision, wireless communications, the IoT applications, machine learning, and deep learning.
He was a recipient of the Superior Thesis Prize awarded by KHU. He was also a recipient of the Golden Globe Award 2020 for Vietnamese Young Scientist by the Central Ho Chi Minh Communist Youth Union associated with the Ministry of Science and Technology.

---
VOLUME 10, 2022
134797

---
IEEE Access
D.-I. Noh et al.: Signal Preprocessing Technique With Noise-Tolerant for RF-Based UAV Signal Classification

**MIKIO HASEGAWA** (Member, IEEE) received the B.Eng., M.Eng., and Dr.Eng. degrees from the Tokyo University of Science, Japan, in 1995, 1997, and 2000, respectively. From 1997 to 2000, he was a Research Fellow with the Japan Society for the Promotion of Science (JSPS). From 2000 to 2007, he was with the Communications Research Laboratory (CRL), Ministry of Posts and Telecommunications, which was reorganized as the National Institute of Information and Communications Technology (NICT), in 2004. He is currently a Professor with the Department of Electrical Engineering, Faculty of Engineering, Tokyo University of Science. His research interests include mobile networks, cognitive radio, neural networks, machine learning, and optimization techniques. He is a Senior Member of IEICE.

**HIROO SEKIYA** (Senior Member, IEEE) received the B.E., M.E., and Ph.D. degrees in electrical engineering from Keio University, Yokohama, Japan, in 1996, 1998, and 2001, respectively. Since April 2001, he has been with Chiba University, Chiba, Japan, where he is currently a Professor with the Graduate School of Engineering. His research interests include high-frequency high-efficiency tuned power amplifiers, resonant dc/dc power converters, wireless power transfer, and digital signal processing for wireless communications. He has served as a BoG Member of IEEE CASS, from 2020 to 2025. He has served as an Associate Editor for IEEE JOURNAL OF EMERGING AND SELECTED TOPICS IN POWER ELECTRONICS, IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS—II: EXPRESS BRIEFS, and IET Circuits, Devices & Systems (CDS).

**SUN-YOUNG KWON** (Member, IEEE) received the B.S. degree in computer science from Pusan National University, South Korea, in 2001, and the M.S. degree in bioinformatics and the Ph.D. degree in electrical and computer engineering from Seoul National University, South Korea, in 2014 and 2018, respectively. From 2018 to 2020, she was with Naver Corporation, South Korea. Previously, she was with Korea Communication Agency and LG Electronics Inc., South Korea. She is currently an Assistant Professor with the School of Biomedical Convergence Engineering, Pusan National University. Her research interests include AI, bioinformatics, drug discovery, graph neural networks, machine learning, and big-data analytics.

**SANG-HWA CHUNG** (Member, IEEE) was born in Busan, South Korea, in 1960. He received the B.S. degree in electrical engineering from Seoul National University, Seoul, South Korea, in 1985, the M.S. degree in computer engineering from Iowa State University, Ames, IA, USA, in 1988, and the Ph.D. degree in computer engineering from the University of Southern California, Los Angeles, CA, USA, in 1993.
From 1993 to 1994, he was an Assistant Professor with the Department of Electrical and Computer Engineering, University of Central Florida, Orlando, FL, USA. Since 2016, he has been the Director with the Dong-Nam Grand ICT Research Center. In 2017, he was selected as an Excellent Research Professor with the Computer Engineering Faculty, PNU. He is currently a Professor with the Computer Engineering Department, Pusan National University (PNU), Busan. He has authored over 240 articles and holds 70 patents. His research interests include embedded systems, wireless networks, software-defined networking, and smart factories. He received the Best Paper Award from the *ETRI Journal*, in 2010 and the Engineering Paper Award from PNU, in 2011.

**WON-JOO HWANG** (Senior Member, IEEE) received the B.S. and M.S. degrees in computer engineering from Pusan National University, Busan, South Korea, in 1998 and 2000, respectively, and the Ph.D. degree in information systems engineering from Osaka University, Osaka, Japan, in 2002. From 2002 to 2019, he was employed as a full-time Professor at Inje University, Gimhae, South Korea. He is currently working as a full-time Professor at the Biomedical Convergence Engineering Department, Pusan National University. His research interests include optimization theory, game theory, machine learning, and data science for wireless communications and networking.

...
---
134798
VOLUME 10, 2022