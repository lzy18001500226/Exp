
# Radio Frequency-Based Drone Detection and Classification using Deep Learning Algorithms

**Raluca Nelega**
Communications Department @ Technical University of Cluj-Napoca & Center for Advanced Research and Technologies for Alternative Energy @ National Institute for Research and Development of Isotopic and Molecular Technologies
Cluj-Napoca, Romania
nelega.si.raluca@student.utcluj.ro

**Bogdan Belean**
Center for Advanced Research and Technologies for Alternative Energy @ National Institute for Research and Development of Isotopic and Molecular Technologies
Cluj-Napoca, Romania
bogda.belean@itim-cj.ro

**Romulus Valeriu Flaviu Turcu**
Biomolecular Physics @ Babes-Bolyai University & Center for Advanced Research and Technologies for Alternative Energy @ National Institute for Research and Development of Isotopic and Molecular Technologies
Cluj-Napoca, Romania
flaviu.turcu@itim-cj.ro

**Emanuel Puschita**
Communications Department @ Technical University of Cluj-Napoca & Center for Advanced Research and Technologies for Alternative Energy @ National Institute for Research and Development of Isotopic and Molecular Technologies
Cluj-Napoca, Romania
emanuel.puschita@com.utcluj.ro

---

***Abstract*—The widespread availability of drones has determined the need for identification systems to prevent their use in illicit activities. This paper focuses on a radio-frequency (RF) detection and classification method based on the You Only Look Once (YOLO) object detection algorithm. Considering the signal spectrograms as input images, the YOLO algorithm is used for spectro-temporal localization and classification of RF signals emitted by 7 different drone types, each operating in video streaming mode. The classification algorithm involves extensive labeling and training procedures, performed using the RF signals registered from all aforementioned drone types, available within the DroneDetect dataset. Furthermore, the impact of the annotation strategy on the model performance is analyzed by comparing two different strategies, namely (1) annotation of the RF transmission bursts as multiple objects and (2) annotation of the RF transmission bursts in a sequence as a single object. The result analysis demonstrates that the information about the time distribution of video transmission bursts provided by the second annotation strategy (i.e., annotation of the RF transmission bursts in a sequence as a single object) improves the performance of the model in terms of precision, recall, and mean average precision considering an intersection over union (IoU) threshold of 50%, and a IoU threshold range of [0.5:0.95], respectively.**

***Keywords*— UAV, RF signal detection and classification, spectrograms, YOLO, image processing.**

---

## I. INTRODUCTION

Commonly known as drones, Unmanned Air Vehicles (UAVs) have grown in popularity in recent years due to the technological development of their industry. On one hand, being equipped with sensors such as GPS, radar, or video cameras, UAVs provide practical and cost-effective solutions for applications like surveillance, rescue missions, environmental monitoring, or agricultural management. On the other hand, despite the benefits they offer, the widespread availability of drones raises concerns about the potential use of this technology in illegal activities that may threaten public safety. Terrorist attacks, intrusion in security-sensitive places such as airports, prisons, or nuclear power plants, privacy violation, or drug trafficking are just some threats posed by the unsupervised use of UAVs. In this context, technologies that prevent the illicit use of drones (i.e., anti-drone systems) are of high interest. Several solutions have been proposed, a survey on anti-drone systems being presented in. Irrespective of the technology implemented, these systems must assure detection, spoofing, and jamming procedures. Therefore, drone detection and classification are key elements in such technologies.

The aim of this paper is the spectro-temporal localization and classification of RF signals emitted by drones when streaming video. Using the DroneDetect dataset detailed in, the spectrograms of 7 different drone types are computed from raw I/Q data and used as input to the YOLOv5 object detection algorithm. Furthermore, two different annotation strategies (i.e., (1) annotation of the RF transmission bursts as multiple objects and (2) annotation of the RF transmission bursts in a sequence as a single object) are compared in terms of precision, recall, and mean average precision at an intersection over union (IoU) threshold of 50%, respectively over a range of IoU thresholds [0.5:0.95]. The originality of this work consists both in the evaluation of the YOLOv5 algorithm for the detection and classification of the 7 drone types within the DroneDetect dataset, and in the comparison of the two proposed annotation strategies.

The remainder of this paper is organized as follows. Section II presents the state-of-the-art strategies for drone detection and classification. The technical aspects of RF-based drone detection and classification strategy are introduced in Section III, while Section IV describes the implementation of the proposed method on the DroneDetect dataset. The experimental results are presented in section V, whereas section VI concludes the work.

---

Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:58 UTC from IEEE Xplore. Restrictions apply.

---

## II. RELATED WORK

Taking advantage of the machine learning (ML) systems ability to recognize patterns that humans cannot perceive, several approaches have been proposed in the literature. In, a comprehensive review of the most used ML methodologies for drone detection and classification is presented, dividing the technologies into (1) radar detection, (2) visual detection, (3) acoustic detection, and (4) radio-frequency (RF) detection.

The radar technology detects objects by emitting radio waves and analyzing the reflections produced when hitting the target. The main disadvantage of this approach is due to the small size of the mini-drones, which can be missed or mistaken for birds. In are emphasized the similarities between drones and birds from the point of view of data collected by the radar, the proposed solution being the use of polarimetric parameters to distinguish between them. Other approaches would include the use of the micro-Doppler signature (MDS), or the cadence frequency spectrum (CDF) features to detect and classify UAVs.

On the other hand, visual detection exploits the progress in the field of computer vision, using images and videos as input data and extracting features from their colors, edges, and shapes,,. As expressed in, this field is still in its infancy, the lack of public datasets for training computer vision models and the conditions required for drone capture such as line of sight (LOS), daylight, or weather make this approach ineffective in real-world scenarios.

The third methodology, namely acoustic detection, uses microphones to detect the sounds produced by UAVs and classifies them based on their acoustic fingerprints. In a Support Vector Machine (SVM) classifier was trained using features such as short-time energy, temporal centroid, Zero Crossing Rate (ZCR), spectral roll-off, and Mel Frequency Cepstral Coefficients (MFCCs) extracted from web data audio files. Another approach is proposed in, where Hidden Markov Models (HMM) are used to perform phoneme analysis for UAV identification. As in the case of visual detection, acoustic detection is hard to implement in real situations, such as noisy urban or industrial environments, the maximum detection distance achieved is only 150 meters.

UAVs constantly communicate with their ground control station using RF signals, exchanging command-control, telemetry, and video information. Typically, the RF transmissions are in the 2.4GHz or 5.8GHz Industrial, Scientific, and Medical (ISM) frequency bands. Therefore, monitoring and analysis of these frequency bands represent a promising area of research. Compared to the other mentioned approaches, RF-based technology offers the possibility not only to detect the presence of a drone, but also to locate the controller used to send the signal from a wide distance.

An overview of the current literature on drone detection and classification in the RF domain is presented in. A common approach is to record raw I/Q RF signal components and to pre-process them to extract relevant features for a classifier. In, Kılıç et al. used the Power Spectral Density (PSD), MFCCs, and Linear Frequency Cepstral Coefficients (LFCCs) to train a 4-class SVM classifier on data from DroneRF dataset, obtaining an accuracy of 98.67%. The same dataset is used in to compare six different ML algorithms: XGBoost, AdaBoost, decision tree (DT), random forest (RF), k-nearest neighbors (kNN), and multilayer perceptron (MLP), considering as input the discrete Fourier transform (DFT) of the signal. With an accuracy of 98.74%, it is concluded that the XGBoost algorithm achieves the best performance.

Previous studies show promising results for drone detection and classification, but do not provide additional information about other RF characteristics (i.e., central frequency, channel bandwidth, modulation, and coding scheme). A different approach is presented in, where the signal spectrogram is computed and further processed as an image using the You Only Look Once (YOLO) deep neural network (DNN) for object detection. The advantage of this method is that it performs spectrum localization, providing information not only about the presence of a drone type but also about features such as center frequency and signal bandwidth. Its performance is evaluated on a dataset consisting of 9 signal types (of which only two are drones) in different signal-to-noise ratio (SNR) conditions. The idea of using the YOLO object detection algorithm for RF signals identification is addressed also in,, and. However, it should be noted that these studies (,,) do not specifically examine radio signals emitted by UAVs, but signals such as Wi-Fi, Bluetooth, ZigBee or other simulated RF signals.

Papers,, and use datasets of up to 3 drone types, which does not provide variety of the data. A much more diverse dataset is the DroneDetect dataset, consisting of 7 different models of drones. In, raw I/Q samples from this dataset are processed using a convolutional neural network (CNN), obtaining an accuracy of 99% for detection, and between 72% and 94% for classification.

This study evaluates an approach similar to the one presented in, using the YOLO object detection algorithm to identify drones and extract their RF characteristics. Besides the fact that the dataset used in this paper is more diverse than the one evaluated in, it also compares YOLO algorithm performances when using two different annotation strategies: (1) annotation of the RF transmission bursts as multiple objects in a spectrogram, and (2) annotation of the RF transmission bursts in a sequence as a single object in a spectrogram. The second strategy tested in this paper was used also in.

## III. RF-BASED DETECTION AND CLASSIFICATION

### A. I/Q signals

In-phase and quadrature (I/Q) signal processing is a tool commonly used in radio communication systems due to the property that any RF signal can be derived from or transformed into a complex baseband signal. I/Q representation is based on the concept of complex numbers, referring to two signals that have the same central frequency (fc) and are 90° out of phase. The in-phase (I) component represents the real part of the complex signal, while the quadrature (Q) component is the imaginary part.

A widely used application of I/Q signals is quadrature demodulation, a technique used to extract the original baseband signal from the modulated signal. As shown in Figure 1, its operation implies the splitting and mixing of the RF input signal with two local oscillator signals that are phase-shifted by 90°, allowing the separation of I and Q components. The low pass filter (LPF) is used to eliminate the high-frequency mixing products, resulting in baseband analog I(t)

---

Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:58 UTC from IEEE Xplore. Restrictions apply.

---

and Q(t) signals. Finally, the discrete values I(n) and Q(n) are obtained through an analog-to-digital converter (A/D), by sampling the analog signal at frequency f, and converting the samples to digital values.

*Fig. 1. Conventional quadrature demodulator.*

### B. Spectrogram generation

A time-frequency graphic representation of a complex signal can be obtained in the form of a spectrogram, calculated using the short-time Fourier transform (STFT), as suggested in Figure 2.

*Fig. 2. Spectrogram computation.*

The complex time-domain signal is divided into frames of N samples with some overlap between them, each frame being multiplied by a window function (e.g., Hamming, Hanning, Blackman, etc.). The Fourier Transform is then applied to each window, as expressed in equation (1), and the complex result is added to a matrix representing the spectrogram of the signal.

$$
X(m,k) = \sum_{n=0}^{N-1} x[n] w[n-m] e^{-2\pi j \frac{nk}{N}}, k = 0, \dots, N-1 \quad (1)
$$

where n is the discrete time variable, k is the frequency, m is the transformation time operator, x[n] is the analyzed time vector and w[n–m] is the window function of length N.

### C. YOLO algorithm

You Only Look Once (YOLO) is a state-of-the-art, real-time object detection algorithm that predicts bounding boxes and class probabilities from a full image, applying a single neural network. The image is divided into S × S grid cells, each cell predicting B bounding boxes along with their corresponding confidence scores. Box confidence scores reflect the probability that an object is present in that box. The network outputs a grid of the form:

$$
S \times S \times (B \times [C, x, y, w, h] + P) \quad (2)
$$

where S is the size of the grid, B is the number of the predicted bounding boxes within a grid cell, C denotes the box confidence score, x and y represent the 2D coordinates, w - width, and h - height are the dimensions of the object and P is the class probabilities vector. The class probabilities represent the likelihood that an object belongs to different predefined classes.

Afterwards, a non-maximum suppression algorithm is used to eliminate redundant and unreliable boxes from the set of predictions. Unreliable boxes are suppressed by removing those with combined confidence scores lower than a certain threshold, while for overlapping boxes, only the one with the highest confidence score is kept. Being a measure of both the localization and classification confidence of the box, the combined confidence score is calculated as the product between the box confidence and the highest-class probability associated with that box.

## IV. IMPLEMENTATION

The problem of detecting and classifying the presence of a drone using I/Q data turns into a problem of detecting objects in an image, namely the spectrogram. Using the signal spectrograms as input data, the YOLOv5 algorithm is used to detect the video streaming (i.e., downlink transmission from the drone to the remote controller) frequency components emitted by drones.

### A. Dataset description

DroneDetect dataset is an open-source dataset consisting of 7 different UAV models: DJI Mavic 2 Air S (AIR), DJI Mavic Pro (MA1), DJI Mavic Pro 2 (MAV), DJI Inspire 2 (INS), DJI Mavic Mini (MIN), DJI Phantom 4 (PHA) and the Parrot Disco (DIS). Recordings contain raw I/Q samples collected by using a Nuand BladeRF SDR with a sample rate of 60 Mbits/s at a center frequency of 2.4375 GHz, having a 28 MHz channel bandwidth.

There are 4 different scenarios in which the UAV signal is captured, depending on radio interference type: (1) clean signal, (2) signal in the presence of Bluetooth interference, (3) in the presence of Wi-Fi interference, and (4) in the presence of both Bluetooth and Wi-Fi interference. Additionally, for each interference scenario there are 5 recordings for each of the 3 drone flight modes: (1) switched on, (2) hovering and (3) flying. The recordings are saved into .dat files, consisting of 1.2 × 10⁸ complex samples that correspond to 2 seconds recording time.

### B. Data pre-processing

As also addressed in, due to the lack of data from the flight mode of the DJI Phantom 4 and from the hovering mode of the Parrot Disco, only the recordings corresponding to the switched-on flight mode are considered for evaluation. Therefore, in the present study, 140 recordings of both non-interference and interference signal are used for training and testing a classifier, providing a diverse evaluation context.

The 2 second recordings are split into smaller segments of 20 milliseconds. In the next step, from each segment is computed the corresponding spectrogram, using a Hanning window of size N=1024 and an overlap of 50%, as illustrated in Figure 3. The spectrograms are saved as grayscale images of size 640x640 pixels, and those where the video streaming signal is present are considered for annotation and further processing.

*Fig. 3. Spectrogram including DJI Mavic Mini signal in the presence of Wi-Fi and Bluetooth interference.*

---

Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:58 UTC from IEEE Xplore. Restrictions apply.

---

### C. Annotation strategies

Two annotation strategies are evaluated to detect and classify the 7 drone types based on the RF transmission bursts emitted while streaming video on the downlink channel.

1) *Annotating RF transmission bursts as multiple objects:* The purpose of this strategy is to identify each video transmission burst based on the distribution of their spectral components.
   Figure 4 illustrates the annotation of each transmission RF burst of DJI Mavic Pro in the presence of Wi-Fi interference.

*Fig. 4. Multiple object annotation of the video transmission bursts of DJI Mavic Pro.*

2) *Annotating RF transmission bursts as a single object:* Compared to the previous method, in this strategy, the model learns not only the spectral distribution of the bursts but also their distribution over time.
   As shown in Figure 5, the entire sequence of bursts emitted by DJI Mavic Pro is annotated as a single object.

*Fig. 5. Single object annotation of the video transmission bursts of DJI Mavic Pro.*

### D. Evaluation metrics

To obtain a comprehensive understanding of the model, several evaluation metrics are used to assess its performance.

1) *Precision:* Precision, expressed in equation (3), is a measure of how many of the positive predictions are truly positive.

$$
P = \frac{T_p}{T_p + F_p} \quad (3)
$$

where $T_p$ and $F_p$ represent true positive, respectively false positive predictions.
In the case of drone detection, it indicates the percentage of correctly identified drone transmission burst instances out of the total instances that the algorithm classified as belonging to drone signals.

2) *Recall:* Recall, also known as sensitivity, indicates the percentage of the correctly classified positive instances, being calculated as

$$
R = \frac{T_p}{T_p + F_N} \quad (4)
$$

where $F_N$ denotes false negative predictions.
This metric reflects the proportion of drone transmission bursts that the model succeeded in detecting out of their actual total.

3) *Mean Average Precision:* Is a metric commonly used in object detection tasks, assessing the ability of the model to correctly localize and classify objects in an image. It compares the predicted bounding box to the ground-truth one, considering a specific threshold for their IoU.

$$
\text{IoU} = \frac{\text{Area of Intersection}}{\text{Area of Union}} \quad (5)
$$

a) *Mean Average Precision at IoU threshold of 50% (mAP@0.5):* For an IoU threshold of 50%, a predicted box is considered true positive if its IoU with the ground-truth box is at least 50%. The mAP@0.5 is the mean of average precision values across all classes, the average precision being computed as the area under the precision-recall curve.

b) *Mean Average Precision over a range of IoU thresholds (mAP@[.5:.95]):* In this case, the performance of the model is evaluated across a range of IoU thresholds, the range [.5:.95] referring to a set of thresholds that take values from 0.5 to 0.95 with a step of 0.05. The mAP@[.5:.95] is calculated as the mean of mAPs obtained for each IoU threshold in the range.

### E. Training and testing the model

To obtain an unbiased result, the spectrograms corresponding to a randomly selected recording (out of 5) from each scenario are used for testing, and the rest from the other 4 recordings are used to train and validate the model. The training and validation datasets are randomly divided into 80% training and 20% validation. Table I details the splitting of spectrograms into training, validation, and test sets.

**TABLE I. TRAINING, VALIDATION AND TEST DATASETS**

| Type       | AIR | DIS | INS | MA1 | MAV | MIN | PHA |
| :--------- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| Training   | 780 | 764 | 766 | 748 | 772 | 753 | 762 |
| Validation | 169 | 198 | 193 | 212 | 191 | 209 | 165 |
| Test       | 251 | 238 | 241 | 240 | 237 | 238 | 273 |

For both annotation strategies, YOLOv5s architecture was trained on 300 epochs, using the stochastic gradient descent (SGD) optimizer with an initial learning rate of 0.01 and a batch size of 256, the other hyperparameters remaining at their default values.

To ensure the generalizing capability of the model on unseen data, the best weights recorded during the training process are used to evaluate the model on the test dataset. The best weights are considered the ones that give the highest weighted sum of the precision, recall, mAP@0.5, and mAP@[.5:.95] calculated on the validation dataset.

The training processes were distributed in three NVIDIA A100-PCIE GPUs of 40GB RAM each.

---

Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:58 UTC from IEEE Xplore. Restrictions apply.

---

## V. RESULTS

The two annotation strategies are evaluated on the test dataset in terms of precision, recall, mAP@0.5, and mAP@[.5:.95]. Furthermore, for a better visualization of the classification performance of the model in each of the two cases, the confusion matrix is also computed.

### A. Annotating the transmission bursts as multiple objects

Table II presents both the aforementioned evaluation metrics calculated for each individual class, as well as their global average for the overall model assessment when each video transmission burst is annotated as an individual object.

**TABLE II. PERFORMANCE INDICES OF THE MODEL TRAINED ON SEPARATELY ANNOTATED TRANSMISSION BURSTS**

| Class | Instances | P     | R     | mAP@0.5 | mAP@[.5-.95] |
| :---- | :-------- | :---- | :---- | :------ | :----------- |
| all   | 15865     | 0.933 | 0.919 | 0.954   | 0.743        |
| AIR   | 2328      | 0.853 | 0.933 | 0.96    | 0.713        |
| INS   | 1291      | 0.998 | 0.994 | 0.995   | 0.819        |
| DIS   | 1138      | 0.928 | 0.873 | 0.964   | 0.707        |
| MAV   | 2471      | 0.936 | 0.888 | 0.893   | 0.791        |
| MA1   | 2541      | 0.921 | 0.968 | 0.983   | 0.785        |
| PHA   | 1663      | 0.96  | 0.946 | 0.981   | 0.81         |
| MIN   | 4433      | 0.936 | 0.833 | 0.9     | 0.577        |

Evaluation parameters as precision, recall, and mAP@0.5 over 99% and mAP@[.5:.95] of 81.9% indicate that the best detection performance is obtained for DJI Inspire 2 drone model. For the other classes, the performance differs depending on the selected evaluation metric.

The ability of the model to classify the 7 types of drones is also highlighted by the confusion matrix presented in Figure 6.

*Fig. 6. Confusion matrix of the model trained on separately annotated transmission bursts.*

In addition to false detections of drone transmission bursts and to drone presence missed, it is observed that in some cases, although the model detects the presence of a burst emitted by a drone, it misclassifies the drone type.

### B. Annotating the transmission bursts as a single object

As for the previously studied annotation strategy, in Table III are presented the evaluation metrics for the model trained on entire sequences of video transmission bursts annotated as single objects.

**TABLE III. PERFORMANCE INDICES OF THE MODEL TRAINED ON COMPLETE SEQUENCE ANNOTATION OF TRANSMISSION BURSTS**

| Class | Instances | P     | R     | mAP@0.5 | mAP@[.5:.95] |
| :---- | :-------- | :---- | :---- | :------ | :----------- |
| all   | 1810      | 0.948 | 0.969 | 0.981   | 0.804        |
| AIR   | 251       | 0.911 | 1     | 0.991   | 0.92         |
| INS   | 241       | 0.999 | 0.996 | 0.995   | 0.808        |
| DIS   | 238       | 0.976 | 0.996 | 0.995   | 0.96         |
| MAV   | 237       | 0.985 | 1     | 0.995   | 0.779        |
| MA1   | 240       | 0.996 | 0.996 | 0.994   | 0.719        |
| PHA   | 365       | 0.812 | 0.885 | 0.91    | 0.669        |
| MIN   | 238       | 0.956 | 0.908 | 0.984   | 0.772        |

The detection and classification performance of the model varies depending on the evaluation metric used. In terms of recall, the best results are obtained for DJI Mavic 2 Air S and DJI Mavic Pro 2 (recall of 100%), while the best precision is obtained for DJI Inspire 2. On the other hand, regarding both mAP@0.5 (99.5%) and mAP@[.5:.95] (96%), the model provides the best detection performance for the Parrot Disco model.

Compared to the strategy in which each individual transmission burst is considered as a separate object, the current annotation strategy provides better results for all evaluation indices.

Figure 7 illustrates the confusion matrix computed for the model trained on complete sequences of transmission bursts. With one exception where a DJI Mavic Pro 2 is misclassified as a DJI Mavic Pro, this annotation strategy provides better results in terms of drone type classification than the one previously evaluated.

*Fig. 7. Confusion matrix of the model trained on complete sequences of transmission bursts*

It can be concluded that the information about the time distribution of video transmission bursts is an important element that improves the drone identification and classification performance of the YOLO algorithm.

---

Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:58 UTC from IEEE Xplore. Restrictions apply.

---

## VI. CONCLUSION

The use of the YOLOv5 algorithm for drone detection and classification showed promising results on the DroneDetect dataset, proving its effectiveness even in signal interference situations.

The results obtained after the analysis of the two annotation strategies (i.e., (1) annotation of the RF transmission bursts as multiple objects and (2) annotation of the RF transmission bursts in a sequence as a single object) demonstrate the importance of the time distribution of video transmission bursts in the process of detection and classification of drones.

Meantime, the performance evaluation process indicates that the second strategy provides overall better results in terms of precision, recall, mAP@0.5 and mAP@[.5:.95].

Up to this point, the scenarios where the drones are flying or hovering were not considered in the evaluation process, therefore future work will also include these recordings. Furthermore, in the next steps, the dataset will be expanded to also include newer drone types and the YOLOv5 algorithm will be compared to other object detection algorithms such as Faster Region-Based Convolutional Neural Network (Faster R-CNN), Single Shot Detector (SSD), and other YOLO versions.

## ACKNOWLEDGMENT

This research was funded by the Ministry of Research, Innovation and Digitalization through Programme 1 - Development of the National Research and Development System, Subprogram 1.2 - Institutional Performance - Funding Projects for Excellence in RDI, Contract No. 37PFECONSOL/30.12.2021 and by the financial support from the MCID through the “Nucleu” Programme within the National Plan for Research, Development and Innovation 2022-2027, project PN 23 24 02 01.

## REFERENCES

 S. Park, H. T. Kim, S. Lee, H. Joo, and H. Kim, “Survey on Anti-Drone Systems: Components, Designs, and Challenges,” *IEEE Access*, vol. 9, pp. 42635–42659, 2021, doi: 10.1109/ACCESS.2021.3065926.
 Swinney, Carolyn J. and Woods, John C., “DroneDetect Dataset: A Radio Frequency dataset of Unmanned Aerial System (UAS) Signals for Machine Learning Detection Classification.” IEEE DataPort, Jun. 12, 2021. doi: 10.21227/5JJJ-1M32.
 B. Taha and A. Shoufan, “Machine Learning-Based Drone Detection and Classification: State-of-the-Art in Research,” *IEEE Access*, vol. 7, pp. 138669–138682, 2019, doi: 10.1109/ACCESS.2019.2942944.
 B. Torvik, K. E. Olsen, and H. Griffiths, “Classification of Birds and UAVs Based on Radar Polarimetry,” *IEEE Geosci. Remote Sens. Lett.*, vol. 13, no. 9, pp. 1305–1309, Sep. 2016, doi: 10.1109/LGRS.2016.2582538.
 P. Molchanov, R. I. A. Harmanny, J. J. M. De Wit, K. Egiazarian, and J. Astola, “Classification of small UAVs and birds by micro-Doppler signatures,” *Int. J. Microw. Wirel. Technol.*, vol. 6, no. 3-4, pp. 435–444, Jun. 2014, doi: 10.1017/S1759078714000282.
 W. Zhang and G. Li, “Detection of multiple micro-drones via cadence velocity diagram analysis,” *Electron. Lett.*, vol. 54, no. 7, pp. 441–443, Apr. 2018, doi: 10.1049/el.2017.4317.
 A. Rozantsev, V. Lepetit, and P. Fua, “Detecting Flying Objects Using a Single Moving Camera,” *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 39, no. 5, pp. 879–892, May 2017, doi: 10.1109/TPAMI.2016.2564408.
 M. Saqib, S. Daud Khan, N. Sharma, and M. Blumenstein, “A study on detecting drones using deep convolutional neural networks,” in *2017 14th IEEE International Conference on Advanced Video and Signal Based Surveillance (AVSS)*, Lecce, Italy: IEEE, Aug. 2017, pp. 1–5. doi: 10.1109/AVSS.2017.8078541.
 C. Aker and S. Kalkan, “Using deep networks for drone detection,” in *2017 14th IEEE International Conference on Advanced Video and Signal Based Surveillance (AVSS)*, Lecce, Italy: IEEE, Aug. 2017, pp. 1–6. doi: 10.1109/AVSS.2017.8078539.
 O. Sahin and S. Ozer, “YOLODrone: Improved YOLO Architecture for Object Detection in Drone Images,” in *2021 44th International Conference on Telecommunications and Signal Processing (TSP)*, Brno, Czech Republic: IEEE, Jul. 2021, pp. 361–365. doi: 10.1109/TSP52935.2021.9522653.
 A. Bernardini, F. Mangiatordi, E. Pallotti, and L. Capodiferro, “Drone detection by acoustic signature identification,” *Electron. Imaging*, vol. 29, no. 10, pp. 60–64, Jan. 2017, doi: 10.2352/ISSN.2470-1173.2017.10.IMAWM-168.
 M. Nijim and N. Mantrawadi, “Drone classification and identification system by phenome analysis using data mining techniques,” in *2016 IEEE Symposium on Technologies for Homeland Security (HST)*, Waltham, MA, USA: IEEE, May 2016, pp. 1–5. doi: 10.1109/THS.2016.7568949.
 J. Yousaf et al., “Drone and Controller Detection and Localization: Trends and Challenges,” *Appl. Sci.*, vol. 12, no. 24, p. 12612, Dec. 2022, doi: 10.3390/app122412612.
 B. Sazdić-Jotić, “Drone Classification based on Radio-Frequency: Techniques, Datasets, and Challenges,” 2022.
 R. Kılıç, N. Kumbasar, E. A. Oral, and I. Y. Ozbek, “Drone classification using RF signal based spectral features,” *Eng. Sci. Technol. Int. J.*, vol. 28, p. 101028, Apr. 2022, doi: 10.1016/j.jestch.2021.06.008.
 M. S. Allahham, M. F. Al-Sa'd, A. Al-Ali, A. Mohamed, T. Khattab, and A. Erbad, “DroneRF dataset: A dataset of drones for RF-based detection, classification and identification,” *Data Brief*, vol. 26, p. 104313, 2019, doi: https://doi.org/10.1016/j.dib.2019.104313.
 Y. Zhang, “RF-based Drone Detection using Machine Learning,” in *2021 2nd International Conference on Computing and Data Science (CDS)*, Stanford, CA, USA: IEEE, Jan. 2021, pp. 425–428. doi: 10.1109/CDS52072.2021.00079.
 S. Basak, S. Rajendran, S. Pollin, and B. Scheers, “Combined RF-based drone detection and classification,” preprint, Jul. 2021. doi: 10.36227/techrxiv.14991999.v1.
 H. N. Nguyen, M. Vomvas, T. Vo-Huu, and G. Noubir, “Spectro-Temporal RF Identification using Deep Learning”.
 A. Vagollari, V. Schram, W. Wicke, M. Hirschbeck, and W. Gerstacker, “Joint Detection and Classification of RF Signals Using Deep Learning,” in *2021 IEEE 93rd Vehicular Technology Conference (VTC2021-Spring)*, Helsinki, Finland: IEEE, Apr. 2021, pp. 1–7. doi: 10.1109/VTC2021-Spring51267.2021.9449073.
 L. Boegner et al., “Large Scale Radio Frequency Wideband Signal Detection &amp; Recognition,” 2022, doi: 10.48550/ARXIV.2211.10335.
 S. Kunze and B. Saha, “Drone Classification with a Convolutional Neural Network Applied to Raw IQ Data,” in *2022 3rd URSI Atlantic and Asia Pacific Radio Science Meeting (AT-AP-RASC)*, Gran Canaria, Spain: IEEE, May 2022, pp. 1–4. doi: 10.23919/AT-AP-RASC54737.2022.9814170.
 C. Ziomek and P. Corredoura, “Digital I/Q demodulator,” in *Proceedings Particle Accelerator Conference*, Dallas, TX, USA: IEEE, 1995, pp. 2663–2665. doi: 10.1109/PAC.1995.505652.
 J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, “You Only Look Once: Unified, Real-Time Object Detection,” in *2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, Las Vegas, NV, USA: IEEE, Jun. 2016, pp. 779–788. doi: 10.1109/CVPR.2016.91.
 G. Jocher et al., “ultralytics/yolov5: v7.0 - YOLOV5 SOTA Realtime Instance Segmentation.” Zenodo, Nov. 22, 2022. doi: 10.5281/ZENODO.7347926.

---

Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:58 UTC from IEEE Xplore. Restrictions apply.
