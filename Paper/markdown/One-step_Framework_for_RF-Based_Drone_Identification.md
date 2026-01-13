2024 5th International Conference on Computer Vision, Image and Deep Learning (CVIDL)

# One-step Framework for RF-Based Drone Identification

**Jiaxin Ge\***
Laboratory of Electromagnetic Space Cognition and Intelligent
Control Technology, Beijing, China
Corresponding author’s e-mail: jiaxinge163@163.com

**Jinhui Li**
Laboratory of Electromagnetic Space Cognition and Intelligent
Control Technology, Beijing, China
hhlee88223@yahoo.com.cn

**Hanshuo Zhang**
Xidian University
Xian, China
hanshuozhang@stu.xidian.edu.cn

**Zhijin Wen**
Laboratory of Electromagnetic Space Cognition and Intelligent
Control Technology, Beijing, China
wzj0911@gmail.com

---

***Abstract*: With the popularity of drones, incidents that threaten public safety caused by drones occur more frequently. In drone control and countermeasures scenarios, it is needed to further identify drones of the same type based on type recognition in order to determine the specific source of the signal. Existing schemes are multistep, which increases the complexity and overhead of post-maintenance. In this work, we propose a joint scheme based on RF fingerprinting where the type and individual belonging to the same type are identified in one step. Specifically, we adopt short-time Fourier transform to calculate the spectrogram of the received signal and label the types and individuals of the drone signals on the spectrogram to develop a data set. Based on the data set, we train a YOLOv5 neural network to extract detailed RF characteristics to identify which type and individual the signal belongs to. To further improve the identification performance, we introduce grayscale image colorization. Furthermore, we use median filtering to reduce noise and protect image edge information. Results on data set composed of real signals from 18 drones of 9 types show the classification accuracy of our proposed framework reaches over 98% at saturation and 3% higher than the method without grayscale image colorization.**

***Keywords*: drone type and individual detection, RF fingerprinting, spectrogram, YOLO**

## I. INTRODUCTION

In recent years, drone-related technology has developed rapidly, and the application of drones in various fields has been expanded constantly. Apart from advantages, drones also inevitably bring the problems of control and countermeasures, which need to be solved. The basis for solving these problems is to detect and classify the target drone quickly and accurately.

At present, radio frequency (RF) detection is considered to be a promising drone detection technology,. RF detection is to capture the RF signals sent by drones and controllers to identify drones. Each drone’s RF signal is not identical, has its own characteristics. These characteristics are determined by the hardware of the drone and controller, as well as the software of the communication system and protocol mode. Even if the drones are of the same type using standard protocols, there are some differences in their RF characteristics due to hardware differences. These characteristics, which can represent the individual uniqueness of the signal transmitter, are called RF fingerprinting. It represents the inherent characteristics of the equipment itself with high identification degree.

In, researchers proposed a SVM classification algorithm to detect drone with RF signals from Bluetooth and microwave interference. In, firstly, the naïve Bayes decision mechanism based on Markov model was designed to detect RF signals of drones, Wi-Fi, and Bluetooth, and then a machine learning classifier was used to classify them. However, these methods require manual feature extraction, which leads to inevitable information loss. Fortunately, deep learning provides an end-to-end processing idea, where the feature extraction and classification are completed in one step. In, researchers proposed a framework based on Faster-RCNN to detect and locate RF signals. Faster-RCNN is a two-stage target detection algorithm, which includes region proposal and classification. Under the same conditions, there is no one-stage algorithm, such as YOLO, with fast detection speed, which is very important for real-time drone detection. Researchers proposed to use YOLO for signal detection and time-frequency positioning on the spectrogram in. In, researchers used YOLO to classify 9 kinds of drones and 2 kinds of non-drone signals. The above papers focus on the recognition of drones versus non-drones, or different types of drones, while ignoring the recognition of drones of the same type.

Identifying the specific individual of a drone signal is also important for drone control and countermeasures. Specifically, with the extensive use of drones in many application fields, both one’s own side and the other side can deploy them. These drones may belong to different types or different individuals with the same type. So in order to determine the specific source of the signal, it needs to identify the individual of the drone based on type recognition. In, researchers proposed a hierarchical drone identification framework including drone type classification and individual identification of the same type, but the multi-step scheme tends to increase maintenance overhead. So one-step framework is a better choice. However, it also faces a difficulty that signal spectrograms of drones from the same type are indistinguishable by their similarity. Researchers found grayscale image colorization can improve the resolution of details,. Deep learning colorization methods, which

---
979-8-3503-7382-0/24/$31.00 ©2024 IEEE
919
Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:45 UTC from IEEE Xplore. Restrictions apply.

---
outperform other methods, uses mapping function to realize the automatic conversion of gray to color. It can add the color information to improve the resolution, which is helpful to improve the classification performance of deep learning.

In this paper, we use short-time Fourier transform (STFT) to generate the spectrogram of the received drone RF signal. The type and individual of the received signal are labeled on the spectrogram to develop a data set. The data set contains two categories of signals, including signals from different types and signals from different individuals of the same type. The CNN with YOLOv5 is trained to learn the RF fingerprinting of each drone on the spectrogram, so as to conduct one-step framework for the types and individuals of drones. We use grayscale image colorization to improve image detail resolution. So as to solve the problem that the drone spectrogram of the same type is similar and difficult to distinguish. Furthermore, we use median filtering to reduce noise and protect image edge information. The effects of noise and frequency offset are also considered for better environmental adaptability. The experiment is conducted with real drone signal. A total of 18 drones of 9 types are chosen to form the target data set. Experimental results show that the proposed framework has better detection and classification performance than the method without grayscale image colorization, i.e., the grayscale image method.

The rest of this paper is organized as follows: Section II introduces signal acquisition, signal analysis and data set generation. Section III describes one-step framework. Section IV is the specific analysis of the experimental results. Section V provides the conclusion.

## II. SIGNAL PROCESSING

### A. Signal Acquisition System

Drone signals are recorded in the anechoic chamber. We collect signals from 18 individual drones of 9 types as transmitters. The signal acquisition system is composed of a receiver and a processing terminal. A universal software radio peripheral (USRP) X310 and an omnidirectional antenna form the receiver. The processing terminal is a computer terminal installed with GNU Radio. The receiving band of the system is 2.4 GHz Industrial Scientific Medical (ISM) band, and the center frequency and sampling rate are set to 2445 MHz and 100 MHz, respectively. The drone signal acquisition system is shown in Figure 1. The drone flies at a distance of 5 meters from the receiver. For each drone individual, we collected IQ samples for 1342.18 milliseconds.

*Figure 1. Drone signal acquisition system.*

### B. Signal Analysis

Drone signals are mainly divided into two types, the remote control signal and the video transmission signal. The remote control signal has concentrated energy, but its duration is short. The video transmission signal has a large signal traffic, occupies a large amount of time-frequency resources, and displays richer information, including many subtle features of the signal, such as the specific frequency band, sequence features, pilot information and energy distribution of the signal. Under the determined time-frequency resolution, compared with the remote control signal, the video transmission signal is reflected on the time-frequency spectrogram as a larger rectangular area both in time and frequency domain, and the YOLO framework detect the large target better. Considering that the remote control is mainly below the horizontal plane of the reconnaissance receiver, the probability of being blocked by buildings is higher. Thus it is hard to be captured. On the contrary, the line of sight (LoS) between a flying drone and the reconnaissance receiver exists almost all the time. Therefore, the signal-to-noise ratio (SNR) of video transmission signal is better than that of remote signal. Considering the above points, this paper chooses the video transmission signal as the target signal and inputs it into the CNN with YOLOv5 framework for learning.

### C. Data set Generation

At present, there is no suitable open data set for drone type and individual detection and classification based on RF signals, so we have to independently develop the data set. 18 drones of 9 types are used to develop the date set, as shown in Table 1.

**Table 1. Specific drone information of the data set.**
| NO. | Individual N0.1 | Individual N0.2 | Bandwidth (MHz) | Protocol |
| :--- | :--- | :--- | :--- | :--- |
| 1 | DJI Air 2S_1 | DJI Air 2S_2 | 20 | O2(OcuSync 2.0) |
| 2 | DJI FPV_1 | DJI FPV_2 | 20 | O3 |
| 3 | DJI Mini 2_1 | DJI Mini 2_2 | 20 | O2 |
| 4 | DJI Mavic 3_1 | DJI Mavic 3_2 | 40 | O3+ |
| 5 | EVO Lite Series_1 | EVO Lite Series_2 | 10 | Autel SkyLink |
| 6 | EVO Nano_1 | EVO Nano_2 | 10 | Autel SkyLink |
| 7 | FIMI X8SE 2020_1 | FIMI X8SE 2020_2 | 10 | TDMA |
| 8 | DJI Mavic Air 2_1 | DJI Mavic Air 2_2 | 20 | O2 |

920
Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:45 UTC from IEEE Xplore. Restrictions apply.

---
| 9 | PowerEgg X_1 | PowerEgg X_2 | 10 | Wi-Fi |
| :--- | :--- | :--- | :--- | :--- |
| Signal Type | Video | | | |
| Frequency | 2.4 GHz | | | |

We divided the long signal of each individual collected by the signal acquisition system into 100 segments. Additive White Gaussian Noise (AWGN) is added to the pure drone signal segments. According to the actual situation, we positioned the SNR range from -20 to 10 dB. The interval is 2 dB. In non-cooperative reception, the oscillator between transmitter and receiver often does not match, and there is doppler frequency shift in the transmission process, which leads to classification errors in drone identification system. This influencing factor should also be taken into account. Therefore, we add random frequency bias to each sample.

According to the above methods, we develop a data set of 17.6k samples. Each sample is a spectrogram image including drone RF signal, spread randomly across a frequency span fixed to 10, 20 or 40 MHz and across a varying time span in no more time than each segment takes.

## III. ONE-STEP FRAMEWORK

This section mainly introduces the pre-processing methods such as STFT, grayscale image colorization and median filtering. In addition, we provide insights on the CNN of YOLOv5 framework in detail. Finally, we describe the specific training implementation.

### A. Pre-processing

In the real scenario, the received signal is usually the non-stationary signal. Traditional analysis methods in time domain or frequency domain have limitations in studying such signals. However, time-frequency analysis technology like STFT provides a very effective method for processing such signals. In this paper, STFT is used to analyze the received signal. STFT represents how the frequency component of a signal changes over time. After sampling by the receiver with the sampling rate f_s, the received signal is transformed to a discrete sequence r(i). With the help of STFT, the complex discrete sequence r(i) is transformed to a spectrogram matrix, i.e.
$$
S_x(n,m) = \sum_{k=0}^{N-1} w(k)r(k+m\Delta)e^{-j\frac{2\pi kn}{N}} \quad (1)
$$
Where $w()$ is a window function with length N and $\Delta$ is a hop size between consecutive windows. Then, the spectrogram of the received data is defined as:
$$
S_d = |S_x|^2 \quad (2)
$$
STFT is carried out on the signal in the data set. Specifically we set the window length N to 2048 and $\Delta$ to 0 in the FFT calculation, the Hamming window is used to analyze the signal. These parameters are chosen to achieve the best time-frequency resolution trade-off.

In order to reduce the adverse effect of noise on the signal, we filter it. Median filtering can protect the edge of signal from being blurred while filtering noise. At the same time, the algorithm is simple and easy to implement. The basic principle of median filtering is to replace the value of each pixel with the median value in the filtering window, so as to eliminate isolated noise points. The output of two-dimensional median filter is defined as:
$$
G_r(x,y) = \text{med}\{ \hat{S}_d(x-a,y-b) \}, a,b \in W \quad (3)
$$
Where $\hat{S}_d()$ is the original image, $G_r()$ is the processed image, W is two-dimensional template.

After median filtering, the signal becomes a one-dimensional gray map matrix. Compared with gray image, color image can improve the resolution of image details to achieve the purpose of image enhancement. Therefore, we introduce grayscale image colorization method. Specifically, we convert the element values in the one-dimensional gray map matrix into color values according to a mapping function, and dye them with this color at the corresponding position of the coordinate axis. Through this mapping relationship, we can get the time-frequency color map of the signal.

We show the original grayscale image after STFT, the grayscale image after median filtering, and the image after median filtering and grayscale image colorization in Figure 2. As can be seen from Figure 2 (a) to Figure 2 (c), drone signals become clearer and easier to identify.

*Figure 2. One example of drone signal time-frequency spectrogram with different pre-processing methods at -14 dB: (a) Grayscale image. (b) Grayscale image with median filtering. (c) Proposed method.*

921
Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:45 UTC from IEEE Xplore. Restrictions apply.

---
### B. YOLOv5 Architecture

YOLO architecture innovatively uses CNN to test the complete image directly to predict the boundary box and determine the classification category. For YOLO, signal recognition is a relatively simple task. At the same time, smaller networks have lower computational complexity and higher speed. YOLOv5 is the fifth version of the framework, which consists of four models, namely s, m, l and x, among which model is the smallest. Based on the above considerations, YOLOv5s is selected as the signal detection and classification framework.

YOLOv5s is mainly composed of input, backbone, neck and head. This architecture adds the focus structure and adopts slicing operation to improve the computational power without losing the information. Two kinds of CSP structures are designed, and the input is divided into two branches, which are operated respectively and then concat. This structure design enables the framework to learn more features. The architecture of YOLOv5 is shown in Figure 3.

*Figure 3. YOLOv5 architecture.*

The YOLO neural network extracts a series of features. Due to the expression of these features, the data with small absolute value is submerged in the data with large absolute value and cannot play a role. In order to ensure that every feature plays an equal role, we need to carry out normalization processing on the extracted feature vector. The normalization formula is coded as the following equation:
$$
x_{\text{norm}} = \frac{x - \mu}{\sigma} \quad (4)
$$
Where, $\mu$ is the mean, $\sigma$ is the standard deviation.

### C. Training Implementation

The main parameters of the training stage are set as shown in Table 2.

**Table 2. Main parameter Settings.**
| Keys | Values |
| :--- | :--- |
| Input shape | 640×640×3 |
| Epoch | 200 |
| Batch size | 16 |
| Initial learning rate | 0.001 |
| Optimizer type | Adam |
| Confidence | 0.5 |

We implement all steps of data set construction in MATLAB, including RF signal processing, spectral calculation and image processing. The YOLOv5 architecture is built on Tensorflow using Python language. To improve the speed of training, the computer is equipped with Intel(R) Xeon(R) Gold 6234 CPU @3.30GHz and NVIDIA Quadro GV100. The data set is divided into training data, validation data and test data in a ratio of 8:1:1.

## IV. RESULTS AND DISCUSSION

This section mainly introduces evaluation parameters. In addition, experiment results are analyzed in detail.

### A. Evaluation Parameters

We use missing alarm (MA) to evaluate the detection performance, probability of accuracy (PA) and confusion matrix to evaluate the classification performance. In general, false alarm (FA) is as important as MA when analyzing performance. But in the anti-drone scenario, the goal is to find as many non-cooperative drones as possible, so they can take the next step. In other words, the goal is to minimize MA as much as possible. In this case, MA is more important. Therefore, we use it to analyze the detection performance. And PA represents the probability of being correctly classified in the detected signal. We use it to analyze classification performance.

922
Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:45 UTC from IEEE Xplore. Restrictions apply.

---
MA reflects the proportion of positive samples predicted as negative samples in the total positive samples. The smaller the value, the better the performance. The calculation formula of MA is:
$$
MA = \frac{FN}{TP + FN} \quad (5)
$$
Where, FN denotes the number of false negatives, TP (True positive) denotes the number of true positives.

PA reflects the ability of the model to predict and classify samples correctly. The larger the value, the better the performance. The calculation formula of PA is:
$$
PA = \frac{N_{\text{correct}}}{N_{\text{all}} - N_{\text{miss}}} \quad (6)
$$
Where, $N_{\text{correct}}$ represents the number of correctly identified samples. $N_{\text{all}}$ represents the number of all samples. $N_{\text{miss}}$ represents the number of undetected samples.

To better evaluate the classification errors at low SNRs, we specifically analyze confusion matrix. Confusion matrix is used as a visual tool to compare prediction results with real labels, and to display the accuracy of classification results in the matrix.

### B. Analysis of Results

In order to analyze the influence of grayscale image colorization on detection and classification performance, we compare the grayscale image method, i.e., no grayscale image colorization, and our proposed method. We used MA parameters to evaluate the detection performance. MA of different methods under different SNR is shown in Figure 4.

*Figure 4. Curve of MA with different SNR of different methods.*

As can be seen from Figure 4, MA decrease with the increase of SNR of all methods. Compared with grayscale image method, our method has lower MA at the same SNR. At low SNR, this performance difference is more obvious. At -16 dB, the MA of our method is 21% lower than that of the grayscale image method. Even in the case of high SNR when MA reaches saturation, the MA of our method is slightly lower. MA of our method reaches saturation at 0 dB, reaching almost 0. The experimental results show that our method is significantly better. When SNR is greater than 8, there is a fluctuation of about 0.2%, which we think is very small and within a reasonable range.

The classification performance under different SNR is shown in Figure 5, and the evaluation parameter is PA which is the probability of correct recognition in the case of detected signal. With the increase of SNR, PA of all methods increased. As can be seen from the figure, PA of our method is significantly higher than grayscale image method. Especially at low SNR, the difference is more obvious. At -2 dB, PA of our method reached approximately 99.6%. Compared with grayscale image method, our method is 43% higher at -20 dB. With the increase of SNR, this gap is decreasing. Even so, PA of our method is still more than 3% higher than grayscale image method at saturation. The experimental results show that our method is superior.

*Figure 5. Curve of PA with different SNR of different methods.*

Combining the above two figures, we find that the difference in performance is due to the use of grayscale image colorization. Compared with grayscale image, grayscale image colorization improves the resolution of drone signal time-frequency spectrogram. YOLOv5 can better detect and classify signals on time-frequency spectrogram after grayscale image colorization. Thus the recognition performance is better. In addition, our approach has significantly improved performance at low SNR. It shows that this method can work well even in the case with strong noise.

To better understand classification errors at low SNR, we specifically analyze confusion matrix. At -14 dB, the confusion matrix of our method is shown in the Figure 6. As can be seen from the figure, we observe that there are two main sources of errors. The first is the confusion between types, especially between EVO Nano_2 and EVO Lite Series_1 and between DJI Mini_2_1 and DJI FPV_2. The second is individual identification errors between DJI FPV_1 and DJI FPV_2. The similarity of these signals between and within types, as well as the limited resolution of the signals in the frequency domain and time domain reduce the differences on the time-frequency spectrograms of these signals, leading to confusion. To be specific, it doesn’t distinguish between DJI FPV_1, DJI FPV_2 and DJI Mini 2_1 very well. The reason is that their signals are very similar, and YOLOv5 cannot distinguish them completely under this SNR, which leads to the decrease of classification accuracy. In addition, EVO Lite Series_1 and EVO Nano_2

923
Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:45 UTC from IEEE Xplore. Restrictions apply.

---
belong to the same brand, and the signals on the time-frequency spectrograms are similar. As a result, confusion also occurs during the classification process. However, the classification accuracy of the rest signals is very high, reaching more than 97%. Generally speaking, the signal time-frequency spectrogram has regular shape, single color and distinct bounding boxes, and there are subtle differences between the spectrograms of different individuals. For deep learning networks, these features can be learned, so the recognition effect is better even at low SNR.

*Figure 6. Confusion matrix of proposed methods at -14 dB.*

## V. CONCLUSION

In this paper, drone RF signal detection and classification of types and individuals is studied. The proposed method successfully performs type and individual recognition based on one-step framework. Experimental results show that the detection and classification of our proposed method is superior to the grayscale image method.

## REFERENCES

 Kang, H., Joung, J., Kim, J., et al, “Protect Your Sky: A Survey of Counter Unmanned Aerial Vehicle Systems,” IEEE Access, 2020, 8: 168671-168710.
 Guvenc, I., Koohifar, F., Singh, S., et al, “Detection, Tracking, and Interdiction for Amateur Drones,” IEEE Communications Magazine, 2018, 56(4): 75-81.
 Guvenc, I., Ozdemir, O., Yapici, Y., et al, “Detection, localization, and tracking of unauthorized UAS and Jammers, ”2017 IEEE/AIAA 36th Digital Avionics Systems Conference (DASC). St. Petersburg, FL: IEEE, 2017: 1-10.
 Surya Vara Prasad, K. N. R., Bhargava, V. K., “A Classification Algorithm for Blind UAV Detection in Wideband RF Systems, ”2020 IEEE 92nd Vehicular Technology Conference (VTC2020-Fall). Victoria, BC, Canada: IEEE, 2020: 1-7.
 Ezuma, M., Erden, F., Kumar Anjinappa, C., et al, “Detection and Classification of UAVs Using RF Fingerprints in the Presence of Wi-Fi and Bluetooth Interference,” IEEE Open Journal of the Communications Society, 2020, 1: 60-76.
 Prasad, K. N. R. S. V., D’souza, K. B., Bhargava, V. K., “A Downscaled Faster-RCNN Framework for Signal Detection and Time-Frequency Localization in Wideband RF Systems, ” IEEE Transactions on Wireless Communications, 2020, 19(7): 4847-4862.
 O’shea, T., Roy, T., Clancy, T. C., “Learning robust general radio signal detection using computer vision methods, ”2017 51st Asilomar Conference on Signals, Systems, and Computers. Pacific Grove, CA, USA: IEEE, 2017: 829-832.
 Basak, S., Rajendran, S., Pollin, S., et al, “Combined RF-Based Drone Detection and Classification,” IEEE Transactions on Cognitive Communications and Networking, 2022, 8(1): 111-120.
 Zhao, X, Wang, L., Wang, Q., et al, “A Hierarchical Framework for Drone Identification based on Radio Frequency Machine Learning, ”2022 IEEE International Conference on Communications Workshops (ICC Workshops). 2022: 391-396.

924
Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:45 UTC from IEEE Xplore. Restrictions apply.

---
 Žeger, I., Grgic, S., Vukovic, J., et al, “Grayscale Image Colorization Methods: Overview and Evaluation, ”IEEE Access, 2021, 9: 113326-113346.
 Yatziv, L., Sapiro, G., “Fast image and video colorization using chrominance blending,” IEEE Transactions on Image Processing, 2006, 15(5): 1120-1129.
 Žeger, I., Grgic, S., An Overview of Grayscale Image Colorization Methods[C/OL]//2020 International Symposium ELMAR. 2020: 109-112.
 Glenn, J., Alex, S., et al, “GitHub ultralytics/yolov5: YOLOv5 in PyTorch >ONNX >CoreML >TFLite, “ 2022. https://github.com/ultralytics/yolov5.
 Zhu, X., Lyu, S., Wang, X., et al, “TPH-YOLOv5: Improved YOLOv5 Based on Transformer Prediction Head for Object Detection on Drone-captured Scenarios, ”2021 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW). 2021: 2778-2788.
 Redmon, J., Divvala, S., Girshick, R., et al, “You Only Look Once: Unified, Real-Time Object Detection, ”2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). Las Vegas, NV, USA: IEEE, 2016: 779-788.

925
Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:45 UTC from IEEE Xplore. Restrictions apply.