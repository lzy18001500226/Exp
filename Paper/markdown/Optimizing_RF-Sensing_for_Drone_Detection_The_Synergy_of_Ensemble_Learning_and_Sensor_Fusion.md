# Optimizing RF-Sensing for Drone Detection: The Synergy of Ensemble Learning and Sensor Fusion

**1st Laiba Tanveer**
Electrical and Computer Engineering
COMSATS University Islamabad, Wah Campus, Pakistan
Wah, Pakistan
laibatanveer0989@gmail.com

**3rd Maham Misbah**
Electrical and Computer Engineering
COMSATS University Islamabad, Wah Campus, Pakistan
Wah, Pakistan
mahammisbah33@gmail.com

**5th Ahmed Alkhayyat**
Department of Computer Technical Engineering
College of Engineering, Islamic University
Najaf, Iraq
ahmedalkhayyat85@iunajaf.edu.iq

**2nd Muhammad Zeshan Alam**
Department of Computer Science
Brandon University, Canada
Brandon, Canada
alamz@brandonu.ca

**4th Farooq Alam Orakzai**
Electrical and Computer Engineering
COMSATS University Islamabad, Wah Campus, Pakistan
Wah, Pakistan
farooqorakzai@gmail.com

**6th Zeeshan Kaleem**
Electrical and Computer Engineering
COMSATS University Islamabad, Wah Campus, Pakistan
Wah, Pakistan
zeeshankaleem@gmail.com

---

***Abstract*—Unmanned Aerial Vehicles (UAVs) find extensive applications across various industries, surveillance, and communication services. However, concerns regarding their potential misuse have prompted the development of counter-drone measures. In this paper, we propose a counter-UAV approach centered on radio frequency (RF) signal sensing. Upon the detection of an RF signal, our system employs a Short-Time Fourier Transform (STFT)-based spectrogram (SP) generation process. This SP is further refined through adaptive windowing and logarithmic tuning to extract multi-intensity features. To classify the complex RF time-domain signals and STFT spectrograms, we utilize two deep learning classifiers: RF-Network and SP-Network, facilitating a multi-class classification process by using deep neural networks (DNN). To enhance the overall accuracy of our model, we leverage an ensemble neural network (EN-Net) by combining predictions from the RF-Network and SP-Network classifiers. Fusing data from a single sensor in both time and frequency domains enhances DNN accuracy by providing complementary information, improving robustness, and reducing overfitting, resulting in increased model performance and a deep understanding of the data. Our results demonstrate a notable improvement in accuracy-specifically, a 36% increase for multi-class models when compared to single-class models. This proves the effectiveness of our EN-Net model in addressing security threats posed by UAVs through advanced RF signal analysis and classification.**

***Index Terms*—Ensemble Neural Network, Radio Frequency, RF Spectrogram, UAVs Detection**

---

This research is supported under the Higher Education Commission (HEC) NRPU research grant 15687.

## I. INTRODUCTION

Unmanned Aerial Vehicles (UAVs) have gained widespread attention and adoption across a range of sectors, such as industry, surveillance, commerce, and public safety. Their adaptability, especially when equipped with appropriate sensors, has played a pivotal role in their increasing popularity. However, their enhanced capabilities can be misused to harm individuals by privacy breaches or terrorism activities. Therefore detection of malicious drones before any damage is of utmost importance. Number of counter-drone techniques presented in the literature to detect and classify drones including Radar, video, acoustic, RF-based, and conventional fusion-based detection,. These techniques aim to identify UAVS and employ various sensors to collect the related data for drone detection, each with its strengths and weaknesses. Most approaches rely on Computer Vision (CV) or machine learning for detection and classification tasks,. In addition to the CV, recently sensor fusion (SF) approach has gained significant attention which includes multiple sensor inputs from different sensors that in turn has the capacity to improve the classification performance.

Among CV techniques, convolutional neural network (CNN) has got popularity in image detection and classification tasks because of its ability to extract the essential and meaningful features while filtering out extraneous ones. Motivated by this, the authors in proposed an RF-based drone type and drone mode detection system using multi-channel 1D-CNN by using the open-source RF dataset. Majority of existing

---

2325-2944/24/$31.00 ©2024 IEEE
DOI 10.1109/DCOSS-IoT61029.2024.00054
308
Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:51 UTC from IEEE Xplore. Restrictions apply.

---

techniques used for drone detection such as radar-based and audio-based techniques pose limitations such as short range, night-vision, and difficulty in detecting small-scale drones. On the contrary, RF-based detection has proved significant improvements in detection and classification accuracy based on CV models and proved to be efficient in complex and small-scale targets. Considering the advantage of RF-based drone detection schemes out of several existing schemes, in this paper, we consider the RF dataset to detect and classify drones. The key steps of the drone RF signal acquisition using software-defined radios (SDR) for further processing and drone detection is shown in Fig. 1.

*Fig. 1: Key steps of UAV RF signal acquisition using SDRs.*

## II. LITERATURE REVIEW

Detecting drones is a difficult task because of the low altitude and smaller size. This is important in critical applications like airport security and for other security-sensitive institutions. Various state-of-the-art techniques are already adopted in the literature such as by using detection UAV images, sound, RF signals, and RADARS. However, using those standalone technologies has less accuracy, false positives, and low precision. Therefore, to overcome those limitations, SF has recently attracted the attention of the researchers and industries for designing a complete fool proof counter-drone detection systems.

In the existing literature, various machine learning and deep learning models are employed for RF and image-based drone detection and classification. However, the scarcity of datasets in the RF domain has hindered the development of precise detection systems. To overcome the scarcity of open-source data sets, AL-Sa'ad et al in developed a DroneRF dataset by using two RF receivers. Most of the researchers are following this dataset to test their proposed algorithm.

Rubina et al., proposed CNN with several one-dimensional layers to learn the different feature maps of RF signals recorded from multiple drones. The proposed model correctly identified drones with an average accuracy of 92.5%. The counter-drone detection system that adopted single sensors to detect drones has certain challenges like limited range, uncertainty, and less accuracy because it can only detect drones with specific areas and drones can be approached from all different directions.

Fusing data from multiple sensors can provide more precise results compared to the standalone systems which are more prone to false alarms and missed detection. Multiple sensor modules including a microphone, camera, LIDAR, RADAR, and RF detection devices were adopted for fusion-based systems. The authors in proposed a multi-modal UAV 3D trajectory tracking system based on acoustic and optical sensor fusion techniques using a commercial drone RF dataset. They achieved novelty in terms of detection range (500m), high 3D positioning accuracy (error less than 1.5% of range), and beam-forming technique. The authors in proposed multi-static radar for tracking and detecting drones using time domain and micro-Doppler signatures, and it was able to correctly detect between similar small objects and UAVs.Authors accurately detected and classified 17 types of drones using CNN and achieved an accuracy of 88.4%. Similarly, the authors in used a hybrid synthetic framework with deep features for robust UAV classification and detection using acoustic, image/video, and wireless RF signals as system inputs. Medaiyese et al., utilized RF signals, audio, and video for detecting the drones using the XGboost algorithm, and also used 10-fold cross-validation and achieved an overall accuracy of around 70.09%.

In another work, the authors introduced a novel framework termed the Hybrid Model with Feature Fusion Network for drone classification based on RF signals. In the proposed model RF signals are resized and transformed into 2D spectrograms, and transfer learning technique was applied to the spectrograms. The feature fusion yielded higher accuracy for 10-class drone classification. To deal with the data set scarcity and less accurate detection in multi-class scenarios, we propose an Ensemble Neural Network (EN-Net) to overcome the above-mentioned problems.

The main contributions of this paper are stated as follows:

- Introduction of a counter UAV approach based on radio frequency (RF) sensing, which enables the detection of RF signals using the generation of adaptive windowing Short-Time Fourier Transform (STFT)-based spectrograms and the use of logarithmic tuning for extracting multi-intensity features. Moreover, these STFT spectrograms serve as the foundational dataset for classification purposes.
- Utilization of two deep learning classifiers, RF-Net and SP-Net, to perform multi-class classification of the complex time domain RF signals and STFT spectrograms, respectively.
- Enhancement of model accuracy by combining predictions from the above two classifiers using an ensemble neural network, leading to a 36% increase in accuracy for 10-classes compared to the standalone models.

The proposed workflow of the EN-Net is shown in Fig. 2. The key steps involved in the proposed system is: initially data capturing involves gathering RF signals from UAVs, controllers, Wi-Fi, and Bluetooth devices to create a dataset.

309
Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:51 UTC from IEEE Xplore. Restrictions apply.

---

*Fig. 2: Workflow of the proposed EN-Net model.*

Then, leveraging STFT with logarithmic tuning will extract multi-intensity features for comprehensive multi-classification tasks. The resulting STFT spectrograms form a dataset for SP-Net classification model, which is a frequency classification neural network model. A separate RF-Net model is adopted for the classification of RF time-series complex signals. RF-Net extracts features from RF data to discriminate between drones and non-drones. Through pattern analysis and parameter adjustments during training, it accurately predicts associated labels. We utilized the data obtain from a single sensor and further divided into two channels for further processing (time and frequency), and in the end fused them to further optimize the system accuracy using EN-Net.

Fusing data from a single sensor processed in both time and frequency domains enhances deep neural network accuracy by leveraging complementary information, improving robustness to variability, and facilitating feature extraction. This approach reduces overfitting by providing diverse representations, leading to a more interpretable model capable of capturing intricate patterns in the data. The resulting benefits include increased model robustness, improved generalization, and a more nuanced understanding of underlying patterns in the sensor data.

## III. PROPOSED EN-NET BASED RF SENSING APPROACH

As demonstrated in Fig. 2, our proposed RF-based UAV sensing system encompasses three main steps: 1) RF signal acquisition; 2) Logarithmic tuning-enhanced STFT feature extraction; and 3) Three-tier drone classification (i.e., drone presence, drone type, and drone flying mode classification) by using the proposed EN-Net architecture. Detailed insights into these architecture are given as follows:
*Fig. 3: STFT-based spectrum of (a) Background activities (b) Phantom (c) Bebop and (d) AR drone.*

### A. Dataset Acquisition

For our proposed model, we utilized the DroneRF dataset, that comprised of three types of drones (AR, Bepop, and phantom) and background activities. The drone RF signals and background activities are stored in the form of lower $(x^L)$ and higher $(x^H)$ RF bands because they utilized two SDRs to capture the full 80 MHz band, each having the capacity to capture 40 MHz bandwidth. The dataset contains 227 segments of RF signals having a signal length of 10.25 sec of RF background activities with no drone signals and 5.25 sec in the presence of drones. The dataset spans two main classes (drones and background activities) with three types of drones including AR, Bepop, and Phantom. Moreover, the dataset is recorded for four modes i.e. on and off, hovering, flying

310
Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:51 UTC from IEEE Xplore. Restrictions apply.

---

without recording, and flying with video recording, that in turn results in total of 10 classes.

### B. Feature Extraction through STFT with Logarithmic Tuning

In RF signal processing, the conventional STFT has proven effective in capturing the spectral features of signals over time. However, in dynamic and rapidly changing RF environments, the adaptability of the analysis becomes important. Therefore, we propose a novel approach by introducing adaptive windowing techniques within the STFT framework followed by logarithmic tuning. Let $x_{\text{RF}}(n)$ denote the acquired RF signal samples. The STFT of the RF signal, $X_{\text{RF}}(m, \omega)$, is expressed as:

$$
X_{\text{RF}}(m, \omega) = \sum_{n=0}^{N-1} x_{\text{RF}}(n) w_{\text{adapt}}(n-m, \sigma_m) e^{-j\omega n}, \quad (1)
$$

The adaptive window function, denoted as $w_{\text{adapt}}(n, \sigma_m)$, incorporates a parameter $\sigma_m$ to modulate the width of the window at time index m. This adaptability allows the STFT to flexibly focus on different frequency resolutions in response to varying signal dynamics. Here, $\omega$ represents the angular frequency, N is the sample size, and $e^{-j\omega n}$ represents the complex exponential function.To enhance feature extraction, the STFT logarithmic tuning is computed as

$$
\text{STFT\_L}(X_{\text{RF}}(m, \omega)) = \log(1 + \alpha \cdot |X_{\text{RF}}(m, \omega)|). \quad (2)
$$

In this equation, $\alpha$ serves as a tuning parameter. The logarithmic tuning introduces a perceptual emphasis on lower-intensity features while maintaining a comprehensive representation across the frequency spectrum. This advanced representation, capturing both time and frequency characteristics, establishes the basis for subsequent multi-intensity feature extraction using SP-Net. The resulting spectrograms of drones are depicted in Fig. 3.

### C. Classification using EN-Net

We introduce the EN-Net as a fusion of the SP-Net and RF-Net architectures, illustrated in Fig. 4. The signal preprocessing aims to eliminate noise and enhance the focus on distinctive features within drone RF signals. The RF-Net model, an enhancement of the framework proposed in, achieves superior performance through strategic hyperparameter tuning and adjustments to the model architecture, particularly focusing on refining the model layers. Similarly, the proposed SP-Net, designed to learn the multi-intensity features of the RF spectrum using STFT uses convolutional layers with rectified linear unit (ReLU) activation, max-pooling layers, and dense layers to capture the multi-intensity features present in the RF spectrum.

EN-Net model improves accuracy and robustness by combining predictions from RF-Net and SP-Net. Leveraging averaging in a fusion technique, the EN-Net improves the predictions for a final decision. Mathematically, the predictions $P_{\text{RF-Net}}$ and $P_{\text{SP-Net}}$ are combined through averaging, and the final prediction $P_{\text{EN-Net}}$ is determined as:

$$
P_{\text{EN-Net}} = \frac{P_{\text{RF-Net}} + P_{\text{SP-Net}}}{2} \quad (3)
$$

The training process of the EN-Net model includes incorporating a sequential approach and reshaping predictions from both. Let $P_{\text{RF-Net}}^{(i)}, P_{\text{SP-Net}}^{(i)}$ represent the predictions of RF-Net and SP-Net for the i-th sample, respectively. The reshaped input $X_{\text{EN-Net}}$ is constructed as:

$$
X_{\text{EN-Net}} = 
\begin{bmatrix}
P_{\text{RF-Net}}^{(1)} \\
P_{\text{SP-Net}}^{(2)} \\
\vdots \\
P_{\text{RF-Net}}^{(N-1)} \\
P_{\text{SP-Net}}^{(N)}
\end{bmatrix} \quad (4)
$$

where N is the number of samples. This input is then utilized for the training of the final EN-Net model.

For a clear illustration of our integrated approach, refer to Fig. 4, which demonstrates how the SP-Net, RF-Net, and EN-Net components are interconnected. The ensemble methodology integrates the strengths of individual models, ensuring precise multi-classification even in complex RF environments.

## IV. SIMULATIONS AND MODEL TRAINING

The SP-Net is trained using 784 spectrograms for each drone, operating on different channels. Randomly selecting 80% of the spectrograms for training and allocating the remaining 20% for testing and validation, we maintained consistency in these settings across all models. The training process involves 100 epochs with a mini-batch size of 32, utilizing a learning rate of 0.0001, binary cross entropy loss function, and the Adam optimizer. This uniformity ensures fair and comparable results within the Google Colab environment using a Tesla T4 GPU.

To ensure the robustness of SP-Net, various scaling and augmentation techniques are employed, effectively increasing the testing data. These techniques include rotation within a 20-degree range, width, and height shifts within a 0.1 range, shear within a 0.2 range, zoom within a 0.2 range, and horizontal flip. Before augmentation, 447 images are generated for each spectrum. Following data pre-processing steps, the frequency range is normalized to (-1 to 1), and further refined to (-0.2 to 0.5) for closer inspection. This process results in a dataset comprising a total of 784 spectral images in each class. For frequency specifications, we adhere to the frequency range from 1.2 GHz to 6 GHz, a frequency step less than 1 kHz, a bandwidth of 40 MHz, and a sample rate of 200 Ms/s. These parameters contribute to the accuracy of the spectrogram data for a balanced dataset.

311
Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:51 UTC from IEEE Xplore. Restrictions apply.

---

*Fig. 4: : The proposed CNN architecture (a) SP-Net (b) RF-Net (c) EN-Net.*

*Fig. 5: 10-classes prediction confusion matrix for (a) only RF-based model, (b) only spectrogram-based model, and (c) Ensemble-based model.*

**TABLE I: Classification performance of the proposed models with multiple-classes.**

| Model          | Metric (%) | 2-Classes | 4-Classes | 10-Classes |
| :------------- | :--------- | :-------- | :-------- | :--------- |
| RF Model       | Precision  | 80        | 85.3      | 61         |
|                | Recall     | 85.7      | 85        | 61.29      |
|                | Accuracy   | 99.9      | 87.5      | 56         |
|                | F1-score   | 82.7      | 84.21     | 59         |
| Spectrum Model | Precision  | 80        | 98.9      | 30         |
|                | Recall     | 80        | 99.5      | 27         |
|                | Accuracy   | 66.7      | 99.7      | 86         |
|                | F1-score   | 80        | 93.2      | 28         |
| Ensemble Model | Precision  | 99        | 96.54     | 67.36      |
|                | Recall     | 100       | 96.58     | 67.23      |
|                | Accuracy   | 99        | 98.29     | 92.48      |
|                | F1-score   | 99.49     | 96        | 67.29      |

Additionally, in this paper, we opted to keep the pre-trained SP-Net and RF-Net models fixed while training the EN-Net, focusing solely on refining the EN-Net. Firstly, since SP-Net and RF-Net are specifically designed to extract specialized features from drone RF signals, it's crucial to preserve their unique contributions. By keeping these models frozen, we ensure that their distinct characteristics remain unchanged within the ensemble. Additionally, since the final predictions from both models have different dimensions, we ensure they are in a consistent 1D format before inputting them into the EN-Net.

Moreover, by concentrating on fine-tuning the EN-Net, we improve computational efficiency, leading to faster convergence and reducing the risk of overfitting, especially in environments with limited resources. This approach enhances overall efficiency and effectiveness in model training.

## V. RESULTS AND DISCUSSION

A comprehensive performance comparison among different models, namely the RF-Net, SP-Net, and EN-Net, across various metrics and classification scenarios is shown in Table I. The RF model in the 2-Classes scenario, achieves the highest accuracy of 99.9% because its easy to classify between the presence of drone and no-drone. However, as the classification complexity increases to 4 (i.e., to deal with 3 types of drones and background activities) and 10 classes (i.e., to further classify the drones flying modes), the accuracy continues to decrease due to increased false detection in multi-class categorization. For SP-Net based model, we achieved a remarkable accuracy of 99.7%. The extension to 2 and 10 classes showed

312
Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:51 UTC from IEEE Xplore. Restrictions apply.

---

reasonable accuracy of 66.7% and 86%, respectively, with an average accuracy of 84%. The Spectral Model effectively captures nuanced features from the dataset, showcasing its potential for complex classifications. The EN-Net model, on the other hand, demonstrates a significant enhancement in testing accuracy, outperforming existing systems. With accuracy levels of 99%, 98.29%, and 92.48% for 2, 4, and 10 classes, respectively, the ensemble model presents a robust solution for diverse classification challenges. It is also evident from the confusion matrix for three models as depicted by Fig. 5, that the ensemble model outperforms other models as the number of classes increases. The main reason behind this improvement is its ability to effectively used the fused data, which in turn improves the model prediction.

**TABLE II: Performance comparison of the proposed model with the state of the art schemes for 10-classes based on DroneRF dataset.**

| Model        | Accuracy % | F-1 Score % | Parameters |
| :----------- | :--------- | :---------- | :--------- |
| RF-Net       | 56         | 69          | 288K       |
| SP-Net       | 86         | 28          | 22K        |
| EN-Net       | 92.48      | 67.29       | 224K       |
| CNN Model    | 59.2       | 55.1        | 6.4M       |
| MC-DNN       | 90.6       | 62.2        | 34.4K      |
| XG-Boost     | 70.09      | 64          | —         |
| RF-NeuralNet | 88         | 61.5        | 32k        |

A comprehensive comparison of the proposed models with the state-of-the-art techniques is presented in Table II. The model's performance is evaluated based on accuracy, F-1 score, and number of trained parameters for 10 drone classes using the DroneRF dataset. The proposed model EN-Net achieves the highest accuracy of 92.48% and F-1 score (67.29%) leveraging STFT spectrograms and data fusion. Moreover, the proposed standalone models RF-Net and SP-Net show comparable performance in accuracy compared to the traditional methods like XGBoost and CNN utilizing fewer number of parameters,. When compared to the joint feature engineered CNN mode, MC-DNN, our fusion model EN-Net shows a 1.88% and 5.09% increase in accuracy and F-1 score, respectively. Moreover, the EN-Net model achieves superior performance with fewer parameters, when compared to the resource-intensive CNN Model. In contrast, traditional methods like XG-Boost and RF-NeuralNet lag in multi-classification tasks as they are unable to capture the complex feature inter-dependencies. These results demonstrate that the proposed EN-Net model provides an optimal balance between accuracy and efficiency for multi-class drone detection as it combines the most favorable features from the two different deep learning models.

## VI. CONCLUSION

We presented an approach that combines a few-shot learning model learned from a limited dataset with sensor fusion, aiming to optimize the system's performance in detecting drones efficiently and accurately, and proposed a novel EN-Net architecture based on two independent models: RF and STFT spectrogram for drone detection. Results highlighted the efficacy of the ensemble model, showcasing superior accuracy of 99% for 2-Classes and outperforming existing systems with a 36% boost in the challenging 10-classes scenario. This proved the effectiveness of our sensor fusion approach, emphasizing the ensemble model as the most robust solution for diverse drone detection challenges.

## REFERENCES

 I. Bisio, C. Garibotto, H. Haleem, F. Lavagetto, and A. Sciarrone, “On the localization of wireless targets: A drone surveillance perspective,” *IEEE Network*, vol. 35, no. 5, pp. 249–255, 2021.
 C. Liu, C. Hu, R. Wang, X. Nie, and F. Liu, “GNSS forward scatter radar detection: Signal processing and experiment,” in *18th International Radar Symposium (IRS)*, 2017, pp. 1–9.
 Y. Liu, X. Wan, H. Tang, J. Yi, Y. Cheng, and X. Zhang, “Digital television based passive bistatic radar system for drone detection,” in *IEEE Radar Conference (RadarConf)*, 2017, pp. 1493–1497.
 Z. Kaleem and M. H. Rehmani, “Amateur drone monitoring: State-of-the-art architectures, key enabling technologies, and future research directions,” *IEEE Wireless Communications*, vol. 25, no. 2, pp. 150–159, 2018.
 M. U. Khan, M. Dil, M. Z. Alam, F. A. Orakazi, A. M. Almasoud, Z. Kaleem, and C. Yuen, “SafeSpace MFNet: Precise and Efficient MultiFeature Drone Detection Network,” *IEEE Transactions on Vehicular Technology*, vol. 73, no. 3, pp. 3106–3118, 2024.
 M. Misbah, M. U. Khan, Z. Yang, and Z. Kaleem, “TF-NET: Deep learning empowered tiny feature network for night-time UAV detection,” in *International Conference on Wireless and Satellite Systems*. Springer, 2023, pp. 3–18.
 M. U. Khan, M. Misbah, Z. Kaleem, Y. Deng, and A. Jamalipour, “GAANet: Ghost Auto Anchor Network for Detecting Varying Size Drones in Dark,” in *IEEE 97th Vehicular Technology Conference (VTC2023-Spring)*, 2023, pp. 1–5.
 S. Samaras, E. Diamantidou, D. Ataloglou, N. Sakellariou, A. Vafeiadis, V. Magoulianitis, A. Lalas, A. Dimou, D. Zarpalas, K. Votis et al., “Deep learning on multi sensor data for counter UAV applications—A systematic review,” *Sensors*, vol. 19, no. 22, p. 4837, 2019.
 M. S. Allahham, T. Khattab, and A. Mohamed, “Deep learning for RF-based drone detection and identification: A multi-channel 1-D convolutional neural networks approach,” in *IEEE International Conference on Informatics, IoT, and Enabling Technologies (ICIoT)*, 2020, pp. 112–117.
 M. F. Al-Sa’d, A. Al-Ali, A. Mohamed, T. Khattab, and A. Erbad, “RF-based drone detection and identification using deep learning approaches: An initiative towards a large open source drone database,” *Future Generation Computer Systems*, vol. 100, pp. 86–97, 2019.
 Z. He, J. Huang, and G. Qian, “UAV Detection and Identification Based on Radio Frequency Using Transfer Learning,” in *IEEE 8th International Conference on Computer and Communications (ICCC)*, 2022, pp. 1812–1817.
 O. O. Medaiyese, A. Syed, and A. P. Lauf, “Machine learning framework for rf-based drone detection and identification system,” in *2nd International Conference On Smart Cities, Automation & Intelligent Computing Systems (ICON-SONICS)*, 2021, pp. 58–64.
 G. Lykou, D. Moustakas, and D. Gritzalis, “Defending airports from uas: A survey on cyber-attacks and counter-drone sensing technologies,” *Sensors*, vol. 20, no. 12, p. 3537, 2020.
 M. S. Allahham, M. F. Al-Sa’d, A. Al-Ali, A. Mohamed, T. Khattab, and A. Erbad, “DroneRF dataset: A dataset of drones for RF-based detection, classification and identification,” *Data in brief*, vol. 26, p. 104313, 2019.
 F. Hoffmann, M. Ritchie, F. Fioranelli, A. Charlish, and H. Griffiths, “Micro-Doppler based detection and tracking of UAVs with multistatic radar,” in *IEEE radar conference (RadarConf)*, 2016, pp. 1–6.
 R. Akter, V.-S. Doan, G. B. Tunze, J.-M. Lee, and D.-S. Kim, “RF-based UAV surveillance system: A sequential convolution neural networks approach,” in *International Conference on Information and Communication Technology Convergence (ICTC)*, 2020, pp. 555–558.
 S. Al-Emadi and F. Al-Senaid, “Drone detection approach based on radio-frequency using convolutional neural network,” in *IEEE International Conference on Informatics, IoT, and Enabling Technologies (ICIoT)*, 2020, pp. 29–34.

313
Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:51 UTC from IEEE Xplore. Restrictions apply.

---

 N. Kumbasar, R. Kılıç, E. A. Oral, and I. Y. Ozbek, “Comparison of spectrogram, persistence spectrum and percentile spectrum based image representation performances in drone detection and classification using novel HMFFNet: Hybrid Model with Feature Fusion Network,” *Expert Systems with Applications*, vol. 206, p. 117654, 2022.
 C. You, Z. Kang, Y. Zeng, and R. Zhang, “Enabling smart reflection in integrated air-ground wireless network: IRS meets UAV,” *IEEE Wireless Communications*, vol. 28, no. 6, pp. 138–144, 2021.
 S. Yang, Y. Luo, W. Miao, C. Ge, W. Sun, and C. Luo, “Rf signal-based uav detection and mode classification: A joint feature engineering generator and multi-channel deep neural network approach,” *Entropy*, vol. 23, no. 12, p. 1678, 2021.
 M. Alzenad, A. El-Keyi, and H. Yanikomeroglu, “3-D placement of an unmanned aerial vehicle base station for maximum coverage of users with different QoS requirements,” *IEEE Wireless Communications Letters*, vol. 7, no. 1, pp. 38–41, 2017.
 S. Shakoor, Z. Kaleem, D.-T. Do, O. A. Dobre, and A. Jamalipour, “Joint optimization of UAV 3-D placement and path-loss factor for energy-efficient maximal coverage,” *IEEE Internet of Things Journal*, vol. 8, no. 12, pp. 9776–9786, 2020.

314
Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:51 UTC from IEEE Xplore. Restrictions apply.
