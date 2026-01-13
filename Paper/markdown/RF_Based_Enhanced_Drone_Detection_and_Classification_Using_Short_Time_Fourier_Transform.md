# RF Based Enhanced Drone Detection and Classification Using Short Time Fourier Transform

**Tushar D Gaikwad**
SOCEMS
Defence Institute of Advanced Technology
Pune, India
tushargaikwad1118@gmail.com

**Sunita V Dhavale**
SOCEMS
Defence Institute of Advanced Technology
Pune, India
sunitadhavale@gmail.com

---

***Abstract*—The proliferation of unmanned aerial vehicles (UAVs) has led to an increased demand for effective drone detection systems. The paper proposes a novel approach for drone detection leveraging Radio Frequency (RF) signals and employing two unique feature extraction techniques based on Short-Time Fourier Transform (STFT). In Method I, we applied STFT to the time series signal segments of DroneRF dataset to generate spectrogram capturing the change in frequency signatures with time for background RF activity and different drone operations. In Method II, we analyzed the spectrograms and extracted 10 statistical features for each time column for background activity and different drone operations. We used ensemble learning classifier consisting of XGBoost and KNN for detection and classification of drones due to large similarity in samples in terms of distribution of RF signal. We compared both the methods for various evaluation metrics, number of features and processing time for classification in 2-class (drone vs no drone), 4-class (type of drone) and 10-class scenario (mode of operation). We found that Method I performed better for 10class classification while Method II used less number of features and carried out classification in less processing time as compared to Method I.**

***Index Terms*—STFT, feature extraction, spectrogram, ensemble learning**

## I. INTRODUCTION

The recent advancement in technology to include precision sensors, gyro scopes, motion sensors has resulted in growing popularity of the drones across the globe. Drones have become a transformative technology with applications that span numerous industries. It has been used for search and rescue missions, precision agriculture, intelligent transportation, smart policing, surveillance, edge computing and many other. However in recent times due to progressive drone technology, cost effective and easy availability there has been rise in various inimical activities carried out using drones. In view of the above extensive research is being carried out for development of new counter UAV techniques.

Drone detection can be carried out using active and passive methods and both methods have evolved over time. Radar drone detection is an active method; while RF, Electro optical, Infrared and Acoustic are passive detection methods. Based on analysis of various drone detection methods, it is found that RF based drone detection technology is cheaper than radar based detection. It offers a promising solution due to its capability of detection in all weather conditions, localization of operator and enhanced detection range as compared to acoustic and Electroptical (EO) methods. The advances in signal processing techniques coupled with evolving learning methods have resulted in development of robust drone detection models suitable for deployment on edge computing devices.

## II. REVIEW OF EXISTING LITERATURE

In authors describe the experiment conducted for collection of raw RF segments of three drones in various modes of operation and raw RF segments of background activity. The database formed is DroneRF dataset and is made publicly available. To test feasibility of the database the authors used DNN to detect, identify the drone and determine its mode of operation. It achieved accuracy of 99.7 for the first DNN (2classes), 84.5 for the second DNN (4-classes), and 46.8 percent for third DNN (10-classes). The authors in used the same dataset developed by and carried out drone detection and identification using three CNNs with increased accuracy for 10-class classification. The authors in proposed paper that used XGBoost algorithm for three classification tasks which are detection of drone, identification of drone type and its mode of operation. They achieved average accuracy of 99.96, 90.73 and 70.09 percent respectively using only low band RF signals.

In authors propose a multistage detector in which initially RF signals are segregated from background noise using Markov model of Naive Baye’s decision mechanism. Thereafter RF signal from drone controller are seperated from interference signals of Wi-fi and bluetooth and in the last stage the drone or controller signals are classified.

The authors in proposed an a UAV detection and identification method using an ensemble learning approach based on hierarchical classification. After preprocessing and feature extraction from raw RF data an ensemble classifier of KNN and XGBoost is evaluated on the publicly available DroneRF dataset of.

Drone classification using RF signal based spectral features is proposed by authors in. In first stage of model significant information contained in RF signal is obtained from Power Spectral Density, Mel Frequency Cepstral Coefficients and

---
Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:48 UTC from IEEE Xplore. Restrictions apply.

---
Linear Frequency Cepstral Coefficients. In second stage SVM classifier is used for the classification.

Authors in- and have used the DroneRF dataset, extracted features using Fourier transform and tried to optimize the classification algorithm to increase the accuracy of model. In our paper we have proposed two novel feature extraction techniques by applying Short Time Fourier Transform to raw RF signal that enhances drone detection and classification accuracy.

## III. ROLE OF SPECTROGRAM IN ANALYZING RF DATA

A spectrogram is a valuable tool in analyzing RF (Radio Frequency) data as it provides a visual representation of how the frequency content of a signal changes over time. In the context of RF data analysis, spectrograms offer several advantages:

### A. Time-Frequency Representation

Spectrograms display how the frequency components of a signal vary with time. In RF data, signals are often dynamic, and their characteristics may change over different time intervals. Spectrograms allow for the identification of specific frequency components and their temporal variations.

### B. Detection of Frequency Patterns

RF signals, especially those emitted by drones or electronic devices, exhibit specific frequency patterns. Spectrograms help in identifying these patterns, making it easier to distinguish between different types of RF activities. This is crucial for tasks such as drone detection, where each type of drone may emit unique RF signatures. This characteristic of Spectrogram has been used in our paper for detection and classification of drone activities.

### C. Dynamic Signal Analysis

Traditional frequency domain analysis or time-domain analysis may not capture the dynamic behavior of RF signals effectively. Spectrograms provide a dynamic perspective, offering insights into how the frequency composition evolves over time. This is essential for understanding complex RF environments.

## IV. METHODOLOGY

### A. DroneRF dataset

DroneRF database published in 2019 consist of raw RF segments recorded in lab conditions. The background RF activity is recorded for 10.25 second while RF UAV communication for each flight mode is recorded for 5.25 second. The collected RF signals are stored as segments in csv format. The dataset contains 227 segments each of low band and high band RF and each segment consist of 10 million samples making it a huge dataset. BUI is a Binary Unique Identifier for each RF activity to be used in labeling. BUI for background activities is always filled with zeros. BUI of each flight mode is as shown in Fig. 1. DroneRf database consist of three levels organized in tree manner. Level 1 corresponds to detection of drone (2-class). Level 2 corresponds to the classification

*Fig. 1. RF signatures organized in a tree manner*

of drone (4-class) and Level 3 corresponds to determine the mode of operation (10-class) classification. Although Fig 1 shows four modes of operation for Phantom drone, however dataset contains drone activity related to BUI-11000 only. The dataset has been condensed to 20GB due to limitation of online MATLAB drive. The intention of paper is to present a novel feature extraction technique. The same has been carried out using 120 segments of low band and high band RF. The segments from each BUI are taken in such a way so as to get a balanced dataset.

### B. Feature Extraction by applying STFT to Raw RF Segments

The Short Term Fourier Transform is essentially a 2D representation of how the frequency content of the RF signal changes over time. We processed raw RF segments of low band and high band corresponding to each BUI by applying STFT through Spectrogram command in MATLAB as following

[Sx, Fx, Tx] = spectrogram(x, hamming(L), L/2, M);
[Sy, Fy, Ty] = spectrogram(y, hamming(L), L/2, M);

- x and y are raw RF segments from low and high band respectively
- Hamming(L) is the window function applied to the input signals. A Hamming window is commonly used in signal processing to reduce spectral leakage. L = 1e5
- L/2 is the overlap between consecutive segments of the signal. In this case, it indicates that there is a 50 percent overlap between successive segments.
- M is the total number of frequency bins, representing the resolution in the frequency domain. For a signal length(L) of 1e5 we considered 2048 NFFT points.

---
Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:48 UTC from IEEE Xplore. Restrictions apply.

---
- Sx and Sy are the absolute magnitude for the ‘x’ and ‘y’ signals respectively.
- Fx, Fy, Tx, and Ty represent the corresponding frequency and time axes for the spectrograms.

Consider the first raw RF segment of low band (x) and high band (y) corresponding to BUI 11000 (Phantom drone switched on) as shown in two subplots in Fig. 2. To calculate STFT of first raw segment x and y (length = 1e7), we considered the window of length L (1e5). We calculate Fourier transform of 2048 NFFT points(M) in the window and shift the zero frequency component at the center. The absolute magnitude Sx and Sy derived for this window of length L are concatenated and written in column. The Hamming window L now is shifted by distance of L/2. Absolute magnitudes Sx and Sy derived from this window (2048 NFFT points) are concatenated and written in next column. The process continues till the end of signal x and y. We get a 2-D plot of 2048 frequency bins (rows) and number of columns equal to number of segments of length L processed (time columns) for signal x and y. Spectrograms of first raw RF segment of low band (x) and high band (y) corresponding to BUI 11000 are as shown in two subplots in Fig. 3. In similar manner Spectrogram function is applied to remaining raw RF segments of length (1e7) corresponding to BUI 11000. Spectrograms generated for each raw RF segment are concatenated columnwise to form a data matrix. Data matrix consist of 2048 frequency bins (rows) and number of time columns equal to the segments of length L(2048 NFFT points) processed for BUI 11000. Hence the frequency content in each time column forms a unique signature for that BUI. The data matrix corresponding to BUI 11000 is saved as .mat file in MATLAB. The other BUIs are also processed in same way to get data matrix which is stored in .mat file. Thereafter all BUIs are loaded and concatenated together. Normalisation and three-level labeling of data in data matrix is carried out using MATLAB to generate a labeled csv file. The labeled data is processed by ensemble learning classifier for 2-class, 4-class and 10-class classification respectively.

*Fig. 2. Raw RF Segment of Low and High band for BUI 11000*

### C. Feature Extraction by deriving Statistical features from Spectrogram

In this method, after applying STFT to first raw RF segment x and y of BUI 11000 we get spectrogram with 2048 rows (frequency bins) and time columns equal to the segments of

*Fig. 3. Spectrogram of Low and High band for BUI 11000*

window length (L) processed. We analyze the spectrogram generated and compute 10 statistical features for every time column of BUI. The five statistical features namely spectral centroid, mean, standard deviation, kurtosis and skewness are computed for both low band and high band of raw RF signal making a total of 10 features. Therefore the 2048 features (frequency bins) for every time column are replaced by a new set of 10 features. The remaining raw RF segments of BUI 11000 are processed in similar way and concatenated columnwise to get a data matrix consisting of 10 rows and time columns equal to segments of window length (L) processed for complete BUI 11000. This data matrix is stored in .mat format. The same procedure is repeated for every BUI. The 10 statistical features for each time column forms a unique signature for that BUI. Thereafter all BUIs are loaded and concatenated together. Normalisation and three-level labeling of data in data matrix is carried out using MATLAB to generate a labeled csv file. The labeled data is processed by ensemble learning classifier for 2-class, 4-class and 10-class classification respectively.

## V. DISCUSSION AND RESULTS

Our study focused upon the feature extraction part of the model. We proposed two novel feature extraction techniques. In our first method we applied STFT to raw RF segments and entire spectrogram was used as feature set. In the second

---
Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:48 UTC from IEEE Xplore. Restrictions apply.

---
method we analysed the spectrogram and extracted five discriminating statistical features for lower band and higher band each thereby using just ten features for classification task.

We compared effectiveness of both the methods by processing the features extracted. We chose ensemble learning classifier comprising of XGBoost and KNN due to large similarity in samples in terms of distribution of RF signal and avoid the problem of overfitting or underfitting of the model. Classification was carried out using 10 fold cross validation. We compared both the methods for their evaluation metrics as displayed in Fig. 4 and Fig. 5. From the plots displayed in Fig. 4 and Fig. 5 it is evident that the Method II has better scores in evaluation metrics as compared to Method I in case of 10 class classification.The size of labeled csv file generated and processing time for classification for both the methods are enumaerated in Table I and Table II respectively. We have used T4-GPU provided by Google Colab for processing in both the methods. From table I and II it is evident that method II has less number of features, hence very less processing time as compared to Method I.The confusion matrix for Method I for 2-class, 4-class and 10-class classification are as shown as Fig. 6,7,8 while confusion matrix for same classification for Method II are shown as Fig. 9,10,11.

*Fig. 4. Evaluation Metrics for Method I*

*Fig. 5. Evaluation Metrics for Method II*

**TABLE I**
SIZE OF LABELED CSV FILE
| | Method I | Method II |
| :--- | :--- | :--- |
| | 278.34MB | 1.32MB |

*Fig. 6. Confusion Matrix for 2-Class Classification(Method I).*

*Fig. 7. Confusion Matrix for 4-Class Classification(Method I).*

---
Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:48 UTC from IEEE Xplore. Restrictions apply.

---
*Fig. 8. Confusion Matrix for 10-Class Classification(Method I).*

*Fig. 9. Confusion Matrix for 2-Class Classification(Method II).*

*Fig. 10. Confusion Matrix for 4-Class Classification(Method II).*

*Fig. 11. Confusion Matrix for 10-Class Classification(Method II).*

**TABLE II**
PROCESSING TIME FOR CLASSIFICATION
| Processing Time | 2-Class | 4-Class | 10-Class |
| :--- | :--- | :--- | :--- |
| Method I | 2min 50second | 19min 34second | 41min 55 second |
| Method II | 3second | 4second | 23second |

## VI. CONCLUSION

The aim of this paper was to study the role of spectrogram in analysis of RF signals. We have been able to test two novel feature extraction techniques by applying STFT to raw Rf segments of DroneRF dataset. From the results of evaluation metrics we can conclude that both the methods perform almost same for drone detection and drone classification. However Method I performs better than method II for 10class classification (mode of drone operation) by 5 percent. From the results enumerated in Table I and II, Method II performs better with just 10 features, csv file of size 1.32MB, processing time of 3second, 4second and 23second for 2-class, 4-class and 10-class classification respectively. In our future work we can optimize the 10-class classification of Method II. However both the methods highlight that applying ShortTime Fourier Transform (STFT) to raw RF segments offer a robust and effective approach for drone detection providing valuable insights and features that enable accurate detection and classification of drone activities.

## REFERENCES

 Mohammad F.Al-Sa’d, Abdulla Al-Alia, Amr Mohameda, Tamer Khattab, Aiman Erbada, “A RF based drone detection and identification using deep learning approaches: An initiative towards a large open source drone database,” Futur. Gener. Comput. Syst., vol. 100, pp. 86-97, 2019, doi: 10.1016/j.future.2019.05.007.
 S. Al-Emadi and F. Al-Senaid, “Drone Detection Approach Based on Radio-Frequency Using Convolutional Neural Network,” 2020 IEEE International Conference on Informatics, IoT, and Enabling Technologies (ICIoT), Doha, Qatar, pp. 29-34,2020.

---
Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:48 UTC from IEEE Xplore. Restrictions apply.

---
 O.O.Medaiyese, A. Syed and A. P. Lauf, “Machine Learning Framework for RF-Based Drone Detection and Identification System,” 2021 2nd International Conference On Smart Cities, Automation and Intelligent Computing Systems (ICON-SONICS), Tangerang, Indonesia, pp. 58-64, 2021.
 Martins Ezuma, Fatih Erden, Chethan Kumar Anjinappa, Ozgur Ozdemir, “Detection and Classification of UAVs Using RF Fingerprints in the Presence of Wi-Fi and Bluetooth Interference”, IEEE Open J. Commun. Soc., vol. 1, pp 60-76, 2019.
 Nemer Ibrahim, Tarek Sheltami, Irfan Ahmad, Ansar Ul-Haque Yasar, and Mohammad A. R. Abdeen, “RF-Based UAV Detection and Identification Using Hierarchical Learning Approach,” Sensors, vol. 6,2021.
 Rabiye Kılıc,, Nida Kumbasar, Emin Argun Oral, Ibrahim Yucel Ozbek. “Drone classification using RF signal based spectral features.” Engineering Science and Technology, an International Journal, 2021.
 Yan, Xiaochen, Tingting Fu, Huaming Lin, Feng Xuan, Yi Huang, Yuchen Cao, Haoji Hu, and Peng Liu, “UAV Detection and Tracking in Urban Environments Using Passive Sensors: A Survey” Applied Sciences, vol. 13, 2023.

---
Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:48 UTC from IEEE Xplore. Restrictions apply.