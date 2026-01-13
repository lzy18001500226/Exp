
# Deep Learning for UAV Classification: Impact of Noise and Multipath Fading in RF Signals

**Prajoy Podder**
Dept. of ECE
Missouri S&T
Rolla, MO, USA
pp64k@mst.edu

**Maciej Zawodniok**
Dept. of ECE
Missouri S&T
Rolla, MO, USA
ORCID: https://orcid.org/0000-0003-1399-8298

**Sanjay Madria**
Dept. of CS
Missouri S&T
Rolla, MO, USA
ORCID: https://orcid.org/0000-0002-2768-3660

***Abstract*— The increasing presence of unmanned aerial vehicles (UAVs) raises serious security concerns, particularly regarding unauthorized drone operations. Recent U.S. security statistics report a sharp rise in unauthorized UAV activities, with the Federal Aviation Administration (FAA) receiving over 100 monthly reports of illegal drone operations near airports. In 2024 alone, Dedrone records 1.19 million unauthorized drone flights across major U.S. cities, highlighting the need for robust UAV detection and classification systems. In this work, a lightweight Convolutional Neural Network (CNN) model is proposed for RF-based UAV classification under noisy and multipath fading conditions. The proposed CNN consists of multiple convolutional blocks, max-pooling layers, fully connected dense layers, and dropout regularization to enhance feature extraction and prevent overfitting. We evaluate the model under four different experimental setups to assess its generalization and performance robustness. CNN is evaluated under four different experimental setups to assess its generalization and performance robustness. It achieves an overall accuracy of 99.1% on the clean original dataset and an average cross-validation accuracy of 98.7%, confirming strong generalization. For the proposed CNN model under noisy and faded conditions, the dataset is modified with Additive White Gaussian Noise (AWGN) at -2 dB SNR and Rayleigh fading, with the data split into 80% for training and 20% for testing. In this scenario, the model achieves an accuracy of 90.00%. When trained on noisy data and tested on the original dataset, the accuracy is 87.27%. When training on the original dataset and testing on the noisy, faded dataset, the accuracy drops to 79.00%. The integration of dropout layers and optimized dense configurations strengthens the model's resilience, making it a promising solution for real-time UAV classification in defense, surveillance, and airspace monitoring applications.**

***Keywords*— UAV Classification, Deep Learning, Convolutional Neural Network (CNN), Radio Frequency (RF), Spectrogram, AWGN, Rayleigh Fading, Dropout, Regularization.**

## I. INTRODUCTION

Recent advancements in drone technology have led to their widespread adoption across various industries, bringing numerous benefits. Drones, or Unmanned Aircraft Systems (UAS), are now commonly used for infrastructure inspections, search and rescue missions, precision agriculture, and even emergency medical deliveries. However, their increasing accessibility has also raised serious security concerns. Unauthorized drones can threaten critical infrastructure, compromise privacy, and pose risks to public safety [1-3]. Additionally, drones are being exploited for illegal activities such as smuggling, unauthorized surveillance, and even acts of terrorism. As a result, developing effective counter-UAS (C-UAS) solutions has become a priority for security agencies, military forces, and regulatory bodies worldwide.

The demand for drone detection solutions has surged in response to security threats in both civilian and military domains. Events such as attacks on airports, oil refineries, and power plants have demonstrated the vulnerabilities posed by unauthorized drone operations. Additionally, the ongoing conflict in Ukraine has highlighted the critical role of drone detection and neutralization on the modern battlefield. Traditional methods such as radar, LiDAR, and visual tracking have limitations in detecting small or low-altitude drones, especially in urban environments with high background noise. Given that commercial drones rely on radio frequency (RF) communication with their ground control stations (GCS), RF-based detection presents a promising approach for identifying and classifying UAVs. However, this method comes with its own set of challenges. Unlike radar and visual tracking, RF-based detection does not require a clear line of sight (LOS) and is unaffected by weather conditions or daylight limitations. RF systems detect and classify drones by monitoring the signals exchanged between a drone and its Ground Control Station (GCS). Since most drones rely on wireless communication, analyzing their unique RF signatures enables accurate identification and classification. Moreover, RF-based detection offers a cost-effective and scalable solution that can be integrated with existing detection systems, enhancing overall security measures [6-8].

While RF-based detection is highly effective, it also presents challenges. Drones operate on unlicensed Industrial, Scientific, and Medical (ISM) bands, which are shared with other wireless communication technologies like Wi-Fi, Bluetooth, and Zigbee. This overlap makes it difficult to distinguish drone signals from background noise. Furthermore, many drones use spread spectrum modulation techniques (such as frequency hopping) to avoid signal interception, making traditional detection methods less effective. Additionally, real-world conditions such as multipath fading and noise interference further complicate reliable detection [9-11]. To overcome these challenges, we propose a Convolutional Neural Network (CNN)-based RF classification model for UAV detection. Our approach extracts distinct features from RF signals to create RF fingerprints, allowing for more precise identification of drones. We introduce additive white Gaussian noise (AWGN) and multipath fading to simulate real-world conditions and evaluate the robustness of our model.

Recent efforts in RF-based UAV classification have explored various machine learning and deep learning techniques. For example, deep neural networks have been

*Research was sponsored by the Army Research Laboratory and was accomplished under Cooperative Agreement Number W911NF-22-2-0208.*
*XXX-X-XXXX-XXXX-X/XX/$XX.00 ©20XX IEEE*

Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:02:35 UTC from IEEE Xplore. Restrictions apply.

---

employed to detect drone presence and flight modes from RF emissions, and convolutional models have shown promise in classifying spectrogram-based representations of RF signals under controlled environments. However, many of these approaches have limited generalizability to real-world conditions involving fading and noise and lack validation on diverse drone types.

**Research Question:** To what extent do additive white Gaussian noise (AWGN) and multipath (Rayleigh) fading degrade radio-frequency (RF)-based UAV classifiers, and can a deep-learning architecture remain reliable under such conditions?

This is addressed by proposing a lightweight convolutional neural network (CNN) that learns RF "fingerprints" from spectrograms. The main contributions are summarized below:

a) We have proposed a deep-convolutional framework that accurately distinguishes multiple UAV types, including novel models such as the Frysky series.
b) We have quantified how different convolutional and dense-layer configurations affect detection performance, providing practical design guidance for future architectures.
c) We have demonstrated the framework’s robustness to channel impairments by systematically testing under additive white Gaussian noise (AWGN) and Rayleigh multipath fading, confirming reliability in real-world RF environments.
d) Our approach enhances RF-based drone detection, providing a scalable, accurate, and cost-effective solution for improving security and public safety.

The rest of this paper is organized as follows: Section II reviews related work, Section III presents the dataset, Section IV describes the proposed CNN model, Section V discusses experimental results, and Section VI concludes the study.

## II. LITERATURE REVIEW

Several RF sensing-based techniques [12-17] for drone identification have been proposed by many researchers. In, the authors introduced an RF-based drone detection system designed to detect and identify drones. Typically, drones communicate with their Ground Control Station (GCS) every 33 milliseconds, whereas mobile devices and access points exchange beacons every 100 milliseconds. This communication pattern was used in to identify the presence of a drone. Additionally, the study found that flying drones emit electromagnetic (EM) signals in the frequency range of 20 Hz to 100 Hz due to motor rotation, camera feed transmission, propeller rotation, and communication channels. Another method, described in, identified drones by analyzing the packet length (in bytes) of transmitted data. However, these methods present certain challenges. First, any device transmitting at a similar rate or with a comparable packet length might be mistakenly identified as a drone. Second, EM signals emitted by drones below 100 Hz have limited range, as they are not active communication signals. Therefore, accurate drone classification requires analyzing RF signatures in both the time and frequency domains. We contend that RF signatures at the physical layer are more reliable for drone classification than those at the MAC layer or other passive signal emissions.

In, a drone surveillance method based on Wi-Fi statistical fingerprinting for identification was proposed. However, many commercial drones employ Frequency-Hopping Spread Spectrum (FHSS) transmission for their RC signals, which should be considered in identification methods. The classification of drone RF signals using time and frequency domain signatures was introduced in, where the authors employed a three-layer fully connected (FC) deep neural network (DNN) to detect and classify drones. However, the study was limited to analyzing only three drone signals and did not account for the effects of noise and channel variations on detection and classification performance.

In, Al-Sa’d et al. constructed three different DNNs for UAV presence, type, and flying modes detection and classification, respectively, on their dataset published in, and validated these models using 10-fold cross-validation with several metrics used to evaluate. The results of classification indicate that the average accuracy of the DNN method in detecting the presence, types, and flying modes of a UAV reached up to 99.7%, 84.5%, and 46.8%, respectively, fully demonstrating the feasibility of UAV detection and recognition using this dataset. In, Al-Emadi et al. utilized this very dataset for solving the problem of UAV detection and classification by adding a feature extraction layer composed of a convolutional layer followed by a fully connected layer for detection and classification, which increased the accuracy of UAV detection and classification. Shi et al. provided an RF UAV dataset that includes five types of UAVs, recording two types of UAVs with yunSDR software radio devices based on the dataset in. Podder et al. proposed a CNN model in where 3 dense layers and dropout layers were added after the basic convolution layer network, and applied that CNN model to our first dataset. The proposed CNN provided the highest accuracy of 95.68% for indoor data and 78.12% for outdoor data in.

## III. DESCRIPTION OF THE DATASET

This dataset contains drone RF signals that were recorded within the semi-anechoic chamber at the CISS department of the Royal Military Academy using an Ettus Research USRP X310. A high-speed 10 Gbit Ethernet cable connected the USRP to a computer for data acquisition. We recorded IQ data at a sampling rate of 100 MSps with a center frequency of 2.44 GHz, and an OmniLOG 70600 omnidirectional antenna was employed as the receiving antenna.

Drones and remote controllers were positioned at seven meters from the receiving antenna. IQ samples from the USRP were recorded in binary format, and subsequently, we transformed these binary IQ vectors into complex signal vectors in MATLAB, storing them as .mat files. Table 1 summarizes various drones and radio controllers of the dataset. Fig. 1 shows the extracted spectrogram of two different types of UAVs.

Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:02:35 UTC from IEEE Xplore. Restrictions apply.

---

*Fig. 1. Extraction of Spectrogram from RF data*

**(a) DJI mini 2RC**

**(b) DJI inspire 2RC**

**TABLE I.**
DRONES AND RADIO CONTROLLERS OF THE DATASET

| Class        | Name                    | Protocol      |
| :----------- | :---------------------- | :------------ |
| DSM          | Spektrum DX4e RC        | DSM           |
| inspire 2RC  | DJI inspire 2 RC        | Lightbridge 2 |
| inspire 2Vid | DJI inspire 2 video     | Lightbridge 2 |
| mini 2RC     | DJI mini 2 RC           | Ocusync 2.0   |
| mini 2Vid    | DJI mini 2 video        | Ocusync 2.0   |
| matriceRC    | DJI Matrice pro RC      | Lightbridge 2 |
| matricevid   | DJI Matrice pro video   | Lightbridge 2 |
| Frysky       | RadioMaster, Taranis RC | Frysky        |

## IV. PROPOSED CNN MODEL

The proposed model illustrated in Fig. 2 is a Convolutional Neural Network (CNN) designed for UAV type classification based on RF signal spectrograms. The input to the network is a spectrogram image, which captures the time-frequency representation of RF signals.

The model begins with a series of four convolutional blocks. Each convolutional block consists of a 2D convolutional layer followed by a ReLU activation function to introduce non-linearity, enabling the network to learn complex and non-linear patterns in the data. After each convolutional layer, a max pooling layer is applied to progressively reduce the spatial dimensions of the feature maps, which decreases computational complexity while preserving important information. The first convolutional block employs 64 filters with a kernel size of (3×3), followed by max pooling. The second block uses 128 filters with (3×3) kernels, again followed by max pooling. The third convolutional block consists of 256 filters, and the fourth convolutional block employs 512 filters, each followed by a max pooling operation.

After the convolutional and pooling layers, the feature maps are flattened into a single long vector to transition from spatial feature extraction to classification. This flattened vector is passed through a series of fully connected (dense) layers. The first dense layer contains 1024 units, followed by a dropout layer with a rate of 0.4 to mitigate overfitting. A second dense layer with 512 units and a second dropout layer with a rate of 0.4 are applied. The output is passed to a dense layer with 8 units corresponding to the number of UAV classes. A softmax activation function is used at the output layer to produce probability distributions over the eight classes, enabling multi-class classification. Overall, the model architecture efficiently combines convolutional feature extraction with fully connected layers for classification, while dropout layers enhance generalization by preventing overfitting during training.

*Fig. 2. Proposed CNN model*

## V. EXPERIMENTAL RESULTS

The proposed CNN model has been evaluated in four different conditions. AWGN noise with various SNR, such as -2 dB, -5 dB, and Rayleigh fading, is introduced in the original dataset, and a faded noisy dataset is created. Then we have evaluated the model in the original and noisy faded conditions to make the model suitable for real-world conditions. Adam optimizer is used. Epoch is 50, and batch size is kept to 32. The following metrics are described in the equation. (1)- (5) have been evaluated in this experiment.

$$
\text{Precision} = \frac{TP}{TP + FP} \quad (1)
$$

$$
\text{Recall} = \frac{TP}{TP + FN} \quad (2)
$$

$$
\text{F1 score} = \frac{2 \times P \times R}{P+R} \quad (3)
$$

$$
\text{Specificty} = \frac{TN}{TN + FP} \quad (4)
$$

$$
\text{Accuracy} = \frac{\sum TP}{\sum(TP + FP + FN + TN)} \quad (5)
$$

Table II shows that the proposed CNN achieves 99.1% overall accuracy on the original (no noise and fading) dataset. Most classes—Mini2RC, Mini2Vid, MatriceRC, MatriceVid, Frysky, and DSM—reach perfect scores (1.00), confirming the network’s ability to capture their distinctive RF signatures without producing false misses. Inspire2RC gives 100% precision, but its recall drops to 92%, meaning a few Inspire2RC examples are mislabeled as another class. Specificity remains around 99% for all classes.

Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:02:35 UTC from IEEE Xplore. Restrictions apply.

---

**TABLE II.**
CR OF 8 CLASSES ON THE ORIGINAL DATASET

| Class       | Precision | Recall | F1 score | Specificity | Accuracy |
| :---------- | :-------- | :----- | :------- | :---------- | :------- |
| DSM         | 1.00      | 1.00   | 1.00     | 1.00        | 0.9909   |
| inspire2RC  | 1.00      | 0.92   | 0.96     | 1.00        |          |
| inspire2vid | 0.96      | 1.00   | 0.98     | 0.99        |          |
| mini 2RC    | 1.00      | 1.00   | 1.00     | 1.00        |          |
| mini 2Vid   | 1.00      | 1.00   | 1.00     | 1.00        |          |
| matriceRC   | 1.00      | 1.00   | 1.00     | 1.00        |          |
| matricevid  | 1.00      | 1.00   | 1.00     | 1.00        |          |
| Frysky      | 1.00      | 1.00   | 1.00     | 1.00        |          |

**TABLE III.**
CROSS-VALIDATION ON THE ORIGINAL DATASET

| Fold | Precision | Recall | F1 Score | Accuracy | Average Test Accuracy |
| :--- | :-------- | :----- | :------- | :------- | :-------------------- |
| 1    | 0.99      | 0.99   | 0.99     | 0.99     | 0.9872                |
| 2    | 1.00      | 1.00   | 1.00     | 1.00     |                       |
| 3    | 1.00      | 1.00   | 1.00     | 1.00     |                       |
| 4    | 0.96      | 0.96   | 0.96     | 0.96     |                       |
| 5    | 0.98      | 0.98   | 0.98     | 0.98     |                       |

Besides this, the 5-fold cross-validation shown in Table III gives an average test accuracy of 98.7%, with fold-wise accuracies ranging from 96% to 100%. So, the model generalizes well and is not overfitting to a particular partition of the data.

The confusion matrix shown in Figure 3 describes that 8 instances of Frysky UAVs were misclassified as DSM, likely due to overlapping feature spaces between the two classes or an insufficient number of Frysky samples.

*Fig. 3. CM of 8 classes using CNN on the original dataset*

**TABLE IV.**
CR OF 8 CLASSES ON THE NOISY FADED DATASET

| Class       | Precision | Recall | F1 score | Specificity | Accuracy |
| :---------- | :-------- | :----- | :------- | :---------- | :------- |
| DSM         | 0.60      | 1.00   | 0.75     | 0.92        | 0.9000   |
| inspire2RC  | 1.00      | 0.92   | 0.96     | 1.00        |          |
| inspire2vid | 0.95      | 1.00   | 0.98     | 0.99        |          |
| mini 2RC    | 0.88      | 1.00   | 0.94     | 0.97        |          |
| mini 2Vid   | 1.00      | 1.00   | 1.00     | 1.00        |          |
| matriceRC   | 1.00      | 1.00   | 1.00     | 1.00        |          |
| matricevid  | 1.00      | 1.00   | 1.00     | 1.00        |          |
| Frysky      | 1.00      | 0.17   | 0.29     | 1.00        |          |

Table IV presents the classification report (CR) of the proposed CNN model under noisy and faded conditions. The dataset is modified with Additive White Gaussian Noise (SNR: -2dB) and Rayleigh fading, and then split with 80% for training and 20% for testing. The accuracy of the proposed CNN is 90.00%. Some classes, such as "matriceRC", "matricevid", and "mini2Vid", achieved perfect precision, recall, and F1 scores, demonstrating the model's strong performance on these categories. However, the "DSM" class gives a low precision of 0.60, and the "Frysky" class gives a very low recall of 0.13. This highlights the challenges of classifying novel UAV types such as "Frysky" under noisy and faded conditions.

*Fig. 4. CM of 8 classes using CNN on noisy fading dataset*

Fig. 4 shows the CM on a noisy and faded dataset. It shows true labels on the y-axis and predicted labels on the x-axis, with diagonal entries representing correctly classified samples for each class. Fig. 5 illustrates the impact of varying Signal-to-Noise Ratio (SNR) levels on the training and testing accuracy of the CNN model when applied to a faded and noisy dataset. As the SNR decreases (indicating higher noise levels), both training and testing accuracies show a consistent decline, reflecting the model's reduced ability to classify RF spectrograms accurately under noisier conditions.

**High SNRs (-1 dB to -5 dB):** At higher SNR levels, the model maintains strong performance, with training accuracy between 88.98% and 94.92%, and testing accuracy between 88.45% and 94.72%, indicating effective generalization even with moderate noise.

**Moderate SNRs (-6 dB to -8 dB):** A gradual decline is observed, where training accuracies drop to 78.20%–83.87%, and testing accuracies fall to 78.18%-83.64%, suggesting that increased noise starts impacting the model's feature extraction capability.

*Fig. 5. Training vs. testing accuracy across SNR levels*

**Low SNRs (-10 dB to -14 dB):** Under severe noise conditions, the model's performance decreases further, with training accuracy ranging from 68.73% to 74.26% and testing accuracy between 68.18% and 74.08%, highlighting the significant challenges in maintaining classification accuracy at very low SNRs.

Table V shows CR on training the noisy fading and testing the original dataset at an SNR of -2 dB. The accuracy of the proposed CNN is 87.27%. inspire2RC and mini2Vid show accurate performance with precision, recall, and F1 scores near 100%, indicating strong identification capabilities even

Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:02:35 UTC from IEEE Xplore. Restrictions apply.

---

under noisy conditions. inspire2Vid achieves a high F1 score of 95%, demonstrating reliable classification accuracy. matriceRC and Frysky show significant challenges in classification, with F1 scores of 31% and 25%, respectively, indicating a high rate of misclassifications. These classes are particularly affected by noise and Rayleigh fading conditions, resulting in poor classification accuracy. The CM in Figure 6 highlights the misclassifications, mainly for MatriceRC and Frysky, suggesting substantial confusion with other classes under noisy fading conditions.

**TABLE V.**
CR ON TRAINING NOISY FADING AND TESTING THE ORIGINAL DATASET

| Class       | Precision | Recall | F1 score | Specificity | Accuracy |
| :---------- | :-------- | :----- | :------- | :---------- | :------- |
| DSM         | 0.50      | 1.00   | 0.67     | 0.88        | 0.8727   |
| inspire2RC  | 1.00      | 0.83   | 0.91     | 1.00        |          |
| inspire2Vid | 0.91      | 1.00   | 0.95     | 0.98        |          |
| mini2RC     | 1.00      | 0.93   | 0.97     | 1.00        |          |
| mini2vid    | 1.00      | 1.00   | 1.00     | 1.00        |          |
| matriceRC   | 1.00      | 1.00   | 1.00     | 1.00        |          |
| matricevid  | 1.00      | 1.00   | 1.00     | 1.00        |          |
| Frysky      | 1.00      | 0.08   | 0.15     | 1.00        |          |

**TABLE VI.**
DATA AUGMENTATION ON TRAINING THE NOISY FADED & TESTING THE ORIGINAL DATASET

| Class       | Without Data Augmentation | After Data Augmentation |
| :---------- | :------------------------ | :---------------------- |
|             | Precision                 | Recall                  |
| DSM         | 0.50                      | 1.00                    |
| inspire2RC  | 1.00                      | 0.83                    |
| inspire2Vid | 0.91                      | 1.00                    |
| mini2RC     | 1.00                      | 0.93                    |
| mini2vid    | 1.00                      | 1.00                    |
| matriceRC   | 1.00                      | 1.00                    |
| matricevid  | 1.00                      | 1.00                    |
| Frysky      | 1.00                      | 0.08                    |

Table VII presents the performance of a CNN model trained on the original (non-faded) dataset and tested on a faded dataset. matricRC, inspire2RC, and mini2RC give low recall value despite high precision, suggesting misclassification due to faded signals, leading to higher false negatives. Frysky's low recall (0.42) and precision (0.71) suggest that the proposed CNN struggles significantly with this class under noisy conditions. DSM has a high recall (0.92) but low precision (0.46), meaning many samples are misclassified into DSM. Mini2Vid, MatriceVid, and Inspire2Vid give high F1 scores (0.95 - 0.97).

**TABLE VII.**
CR OF 8 CLASSES ON ORIGINAL TRAINING AND NOISY, FADED TESTING DATA

| Class       | Precision | Recall | F1 score | Specificity | Accuracy |
| :---------- | :-------- | :----- | :------- | :---------- | :------- |
| DSM         | 0.46      | 0.92   | 0.61     | 0.87        | 0.7900   |
| inspire2RC  | 1.00      | 0.83   | 0.91     | 1.00        |          |
| inspire2vid | 0.91      | 1.00   | 0.95     | 0.98        |          |
| mini2RC     | 1.00      | 0.60   | 0.75     | 1.00        |          |
| mini2vid    | 1.00      | 0.93   | 0.97     | 1.00        |          |
| matriceRC   | 1.00      | 0.45   | 0.62     | 1.00        |          |
| matricevid  | 0.68      | 1.00   | 0.81     | 0.94        |          |
| Frysky      | 0.71      | 0.42   | 0.53     | 0.98        |          |

The proposed CNN model performs well for inspire2Vid, mini2Vid, and matriceVid, indicating these classes retain robust features even under fading and AWGN noise conditions. Fig. 7 presents the confusion matrix (CM) for the 8-class CNN classification model, evaluated on the originally trained dataset and tested on a noisy, faded dataset.

Table VI shows that applying data augmentation (with a rotation range of 30 degrees, zoom range of 0.1, and shear range of 0.1) results in a 6.54% decrease in accuracy compared to the model without data augmentation, when using CNN on the training noisy fading and testing the original dataset at an SNR of -2 dB.

*Fig. 6. CM of 8 classes using proposed CNN*

*Fig. 7. CM of 8 classes using the proposed CNN on the training original and testing noisy, faded dataset*

This highlights strong classification performance for inspire2Vid (20/20 correctly classified, 100% accuracy) and matriceVid (13/13 correctly classified, 100% accuracy). However, Mini2RC, matriceRC, and Frysky exhibit significant misclassifications. The proposed CNN model struggles to differentiate Frysky from DSM, leading to significant confusion between these classes.

In table VIII, VGG16 and VGG19 models, when tested on a highly noisy (-2 dB SNR) dataset, provide poor performance, achieving around 53-55% accuracy. They could not effectively handle the noise and fading effects present in the dataset. 80% training, 20% testing both sets are from the noisy-faded dataset. While the proposed CNN model achieves a much higher accuracy of 90%, with a strong balance between precision and recall, it demonstrates its ability to perform reliable classification even under extremely

Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:02:35 UTC from IEEE Xplore. Restrictions apply.

---

degraded signal conditions. Noise robustness often demands models to learn local features, not just global high-level abstractions this is why proposed CNNs can outperform in low-SNR environments.

**TABLE VIII.**
COMPARISON WITH VGG 16 AND VGG19

| Model        | Average precision | Average recall | Average F1 score | Average accuracy |
| :----------- | :---------------- | :------------- | :--------------- | :--------------- |
| VGG 16       | 0.58              | 0.55           | 0.56             | 0.55             |
| VGG 19       | 0.55              | 0.53           | 0.53             | 0.53             |
| Proposed CNN | 0.93              | 0.90           | 0.88             | 0.90             |

## VI. CONCLUSION

This work introduces an efficient and lightweight CNN model for RF-based UAV identification, specifically optimized to perform under noisy and faded signal conditions. The model achieves 99.1% accuracy on the clean dataset, but its performance drops under noisy conditions, with 90% accuracy at -2 dB SNR and low recall for the "Frysky" class. When trained on noisy data and tested on the original dataset, the accuracy is 87.27%, and performance declines further as the SNR decreases, with overall classification accuracy ranging from 68.73% to 74.26%. Despite these challenges, the proposed CNN remains effective, achieving 79% accuracy when trained on clean data and tested on noisy data. This research demonstrates the potential of CNNs for real-time UAV detection in contested RF environments, enhancing security in defense and surveillance. Future work focuses on several strategies to enhance UAV detection and classification robustness. First, integrating data augmentation techniques, such as synthetic noise and fading, into the training process will improve the model's resilience to real-world conditions. Research will also investigate the detection of novel UAVs not included in the training dataset, enabling adaptive learning capabilities.

## REFERENCES

 G. De Cubber, "Explosive drones: How to deal with this new threat?" in International workshop on Measurement, Prevention, Protection and Management of CBRN Risks (RISE), Belgium, 04 2019.
 M.H. Rahman et al., "A Comprehensive Survey of Unmanned Aerial Vehicles Detection and Classification Using Machine Learning Approach: Challenges, Solutions, and Future Directions. Remote Sens. 2024, 16, 879.
 Coveney, S.; Roberts, K. Lightweight UAV digital elevation models and orthoimagery for environmental applications: Data accuracy evaluation and potential for river flood risk modelling. Int. J. Remote Sens. 2017, 38, 3159-3180.
 Alsalam, B.H.Y.; Morton, K.; Campbell, D.; Gonzalez, F. Autonomous UAV with vision based on-board decision making for remote sensing and precision agriculture. In Proceedings of the 2017 IEEE Aerospace Conference, Big Sky, MT, USA, 4-11 March 2017; pp. 1-12.
 Amazon Prime Air Drone Delivery Fleet Gets FAA Approval. Available online: https://www.cnbc.com/2020/08/31/amazon-prime-now-drone-delivery-fleet-gets-faa-approval.html (accessed on 18 June 2024).
 R. Reedha et al, "Transformer neural network for weed and crop classification of high resolution UAV images", Remote Sens. 2022, 14, 592.
 Bisio, I.; Garibotto, C.; Haleem, H.; Lavagetto, F.; Sciarrone, A. On the localization of wireless targets: A drone surveillance perspective. IEEE Netw. 2021, 35, 249-255.
 Civilian. Civilian Drone Crashes into ARMY Helicopter. Available online: https://nypost.com/2017/09/22/army-helicopterhit-by-drone/ (accessed on 18 April 2025).
 Medaiyese, O.O.; Ezuma, M.; Lauf, A.P.; Guvenc, I. Wavelet transform analytics for RF-based UAV detection and identification system using machine learning. Pervasive Mob. Comput. 2022, 82, 101569.
 Birch, G.C.; Griffin, J.C.; Erdman, M.K. Uas Detection Classification and Neutralization: Market Survey 2015; Technical Report, Sandia National Lab. (SNL-NM): Albuquerque, NM, USA, 2015.
 M. Kulin, T. Kazaz, I. Moerman, and E. De Poorter, "End-to-End Learning From Spectrum Data: A Deep Learning Approach for Wireless Signal Identification in Spectrum Monitoring Applications," IEEE Access, vol. 6, pp. 18484-18501, 2018
 P. Nguyen, M. Ravindranatha, A. Nguyen, R. Han, and T. Vu, "Investigating Cost-Effective RF-Based Detection of Drones," in Proceedings of the 2nd Workshop on Micro Aerial Vehicle Networks, Systems, and Applications for Civilian Use, ser. DroNet '16. NY, USA: Association for Computing Machinery, 2016, p. 17-22.
 P. Kosolyudhthasarn, V. Visoottiviseth, D. Fall, and S. Kashihara, "Drone Detection and Identification by Using Packet Length Signature," in 2018 15th International Joint Conference on Computer Science and Software Engineering (JCSSE), 2018, pp. 1-6.
 I. Bisio, C. Garibotto, F. Lavagetto, A. Sciarrone, and S. Zappatore, "Blind Detection: Advanced Techniques for WiFi-Based Drone Surveillance," IEEE Transactions on Vehicular Technology, vol. 68, no. 1, pp. 938-946, 2019
 M. Al-Sa'd, A. Al-Ali, T. Khattab, and A. Erbad, "RF-based drone detection and identification using deep learning approaches: An initiative towards a large open source drone database," Future Generation Computer Systems, vol. 100, 05 2019.
 S. Al-Emadi and F. Al-Senaid, "Drone Detection Approach Based on Radio-Frequency Using Convolutional Neural Network," in 2020 IEEE International Conference on Informatics, IoT, and Enabling Technologies (ICIoT), 2020, pp. 29-34.
 M. M. Azari, H. Sallouha, A. Chiumento, S. Rajendran, E. Vinogradov, and S. Pollin, "Key Technologies and System Trade-offs for Detection and Localization of Amateur Drones," IEEE Communications Magazine, vol. 56, no. 1, pp. 51-57, 2018.
 T. Andre, K. A. Hummel, A. P. Schoellig, E. Yanmaz, M. Asadpour, C. Bettstetter, P. Grippa, H. Hellwagner, S. Sand, and S. Zhang, "Application-driven design of aerial communication networks," IEEE Communications Magazine, vol. 52, no. 5, pp. 129-137, 2014.
 J. Geier, "802.11 Beacons Revealed." [Online]. Available: https://www.scribd.com/document/354688905/802-11-Beacons-Revealed
 Al-Sa'D, M.; Al-Ali, A.; Mohamed, A.; Khattab, T.; Erbad, A. RF-based drone detection and identification using deep learning approaches: An initiative towards a large open source drone database. Futur. Gener. Comput. Syst. 2019,100, 86-97.
 Al-Emadi, S.; Al-Senaid, F. Drone Detection Approach Based on Radio-Frequency Using Convolutional Neural Network. In Proceedings of the 2020 IEEE International Conference on Informatics, IoT, and Enabling Technologies (ICIoT), Doha, Qatar, 2-5 February 2020.
 Shi, H.-D.; Lu, H.; Bian, Z.-A. Deep convolutional network multi-target UAV signal detection method. J. Air Force Eng. Univ.2021,22, 29-34
 Allahham, M.S.; Al-Sa'd, M.F.; Al-Ali, A.; Mohamed, A.; Khattab, T.; Erbad, A. DroneRF dataset: A dataset of drones for RF-based detection, classification, and identification. Data Brief2019,26, 1043
 P. Podder, M. Zawodniok and S. Madria, "Deep Learning for UAV Detection and Classification via Radio Frequency Signal Analysis," 2024 25th IEEE International Conference on Mobile Data Management (MDM), Brussels, Belgium, 2024, pp. 165-174.
 S. Basak et al.,"Autoencoder based framework for drone RF signal classification and novelty detection," 2023 25th International Conference on Advanced Communication Technology (ICACT), Pyeongchang, Korea, Republic of, 2023, pp. 218-225.
 S. Basak, S. Rajendran, S. Pollin and B. Scheers, "Combined RF-Based Drone Detection and Classification," in IEEE Transactions on Cognitive Communications and Networking, vol. 8, no. 1, pp. 111-120, March 2022, doi: 10.1109/TCCN.2021.3099114.
 RF-signals-of-UAVs, Distributed by Kaggle, (Version 1), Accessed on April 23, 2025 from https://www.kaggle.com/datasets/xcz74741/rf-signals-of-uavs

Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:02:35 UTC from IEEE Xplore. Restrictions apply.
