
IEEE JOURNAL OF RADIO FREQUENCY IDENTIFICATION, VOL. 8, 2024
821

# Robust Low-Cost Drone Detection and Classification Using Convolutional Neural Networks in Low SNR Environments

Stefan Glüge, Matthias Nyfeler, Ahmad Aghaebrahimian, Nicola Ramagnano,
and Christof Schüpbach

***Abstract*—The proliferation of drones, or unmanned aerial vehicles (UAVs), has raised significant safety concerns due to their potential misuse in activities such as espionage, smuggling, and infrastructure disruption. This paper addresses the critical need for effective drone detection and classification systems that operate independently of UAV cooperation. We evaluate various convolutional neural networks (CNNs) for their ability to detect and classify drones using spectrogram data derived from consecutive Fourier transforms of signal components. The focus is on model robustness in low signal-to-noise ratio (SNR) environments, which is critical for real-world applications. A comprehensive dataset is provided to support future model development. In addition, we demonstrate a low-cost drone detection system using a standard computer, software-defined radio (SDR) and antenna, validated through real-world field testing. On our development dataset, all models consistently achieved an average balanced classification accuracy of ≥ 85% at SNR > -12 dB. In the field test, these models achieved an average balance accuracy of > 80%, depending on transmitter distance and antenna direction. Our contributions include: a publicly available dataset for model development, a comparative analysis of CNN for drone detection under low SNR conditions, and the deployment and field evaluation of a practical, low-cost detection system.**

***Index Terms*—Drone detection, UAV classification, low signal-to-noise ratio, robustness, real-world field test.**

---

## I. INTRODUCTION

DRONES, or civil UAVs, have evolved from hobby toys to commercial systems with many applications. In particular, mini/amateur drones have become ubiquitous. With the proliferation of these low-cost, small and easy-to-fly drones, safety issues have became more pressing (e.g., spying, transfer of illegal or dangerous goods, disruption of infrastructure, assault). Although regulations and technical solutions (such as transponder systems) are in place to safely integrate UAVs into the airspace, detection and classification systems that do not rely on the cooperation of the UAV are necessary. Various technologies such as audio, video, radar, or radio frequency (RF) scanners have been proposed for this task.

In this paper, we evaluate different CNNs for drone detection and classification using the spectrogram data computed with consecutive Fourier transforms for the real and imaginary parts of the signal. To facilitate future model development, we make the dataset publicly available. In terms of performance, we focus on the robustness of the models to low SNRs, as this is the most relevant aspect for a real-world application of the system. Furthermore, we evaluate a low-cost drone detection system consisting of a standard computer, SDR, and antenna in a real-world field test.

Our contributions can therefore be summarized as follows:

- We provide the dataset used to develop the model. Together with the code to load and transform the data, it can be easily used for future model development.
- We compare different CNNs using 2D spectrogram data for detection and classification of drones based on their RF signals under challenging conditions, i.e., low SNRs down to -20 dB.
- We visualize the model embeddings to understand how the model clusters and separates different classes, to identify potential overlaps or ambiguities, and to examine the hierarchical relationships within the learned features.
- We implement the models in a low-cost detection system and evaluate them in a field test.

### A. Related Work

A literature review on drone detection methods based on deep learning (DL) is given in and. Both works reflect the state of the art in 2024. Different DL algorithms are discussed with respect to the techniques used to detect drones based on visual, radar, acoustic, and RF signals. Given these general overviews, we briefly summaries recent work based on RF data, with a particular focus on the data side of the problem to motivate our work.

With the advent of DL-based methods, the data used to train models became the cornerstone of any detection system. Table I provides an overview of openly available datasets of RF drone signals. The DroneRF dataset is one of the first openly available datasets. It contains RF time

*Received 1 July 2024; revised 5 September 2024; accepted 25 October 2024. Date of publication 28 October 2024; date of current version 6 November 2024. This work was supported by Armasuisse Science + Technology. (Corresponding author: Stefan Glüge.)*
*Stefan Glüge, Matthias Nyfeler, and Ahmad Aghaebrahimian are with the Institute of Computational Life Sciences, Zurich University of Applied Sciences, 8820 Wädenswil, Switzerland (e-mail: stefan.gluege@zhaw.ch).*
*Nicola Ramagnano is with the Institute for Communication Systems, Eastern Switzerland University of Applied Sciences, 8640 Rapperswil-Jona, Switzerland.*
*Christof Schüpbach is with the Armasuisse Science and Technology, Section Networks and Protection, 3603 Thun, Switzerland.*
*Digital Object Identifier 10.1109/JRFID.2024.3487303*

---

© 2024 The Authors. This work is licensed under a Creative Commons Attribution 4.0 License. For more information, see https://creativecommons.org/licenses/by/4.0/

---

822
IEEE JOURNAL OF RADIO FREQUENCY IDENTIFICATION, VOL. 8, 2024

**TABLE I**
OVERVIEW ON OPENLY AVAILABLE DRONE RF DATASETS

| Dataset                                   | Year | Datatype                         | UAV                     | Noise                    | Size    |
| :---------------------------------------- | :--- | :------------------------------- | :---------------------- | :----------------------- | :------ |
| DroneRF                                   | 2019 | Raw Amplitude                    | 3 drones + 3 controller | Background RF activities | 3.75 GB |
| Drone remote controller RF signal dataset | 2020 | Raw Amplitude                    | 17 controller           | none                     | 124 GB  |
| DroneDetect dataset                       | 2020 | Raw IQ                           | 7 drones + 7 controller | Bluetooth, Wi-Fi devices | 66 GB   |
| Cardinal RF                               | 2022 | Raw Amplitude                    | 6 drones + 6 controller | Bluetooth, Wi-Fi         | 65 GB   |
| Noisy drone RF signals                    | 2023 | Pre-processed IQ and Spectrogram | 6 drones + 4 controller | Bluetooth, Wi-Fi, Gauss  | 23 GB   |

series data from three drones in four flight modes (i.e., on, hovering, flying, video recording) recorded by two universal software radio peripheral (USRP) SDR transceivers. The dataset is widely used and enabled follow-up work with different approaches to classification systems, i.e., DL-based,, focused on pre-processing and combining signals from two frequency bands, genetic algorithm-based heterogeneous integrated k-nearest neighbor, and hierarchical reinforcement learning-based. In general, the classification accuracies reported in the papers on the DroneRF dataset are close to 100%. Specifically,,, and report an average accuracy of 99.7%, 100%, and 99.98%, respectively, to detect the presence of a drone. There is therefore an obvious need for a harder, more realistic dataset.

Consequently, investigate the detection and classification of drones in the presence of Bluetooth and Wi-Fi signals. Their system used a multi-stage detector to distinguish drone signals from the background noise and interfering signals. Once a a signal was identified as a drone signal, it was classified using machine learning (ML) techniques. The detection performance of the proposed system was evaluated for different SNRs. The corresponding recordings (17 drone controls from eight different manufacturers) are openly available. Unfortunately, the Bluetooth/Wi-Fi noise is not part of the dataset. Ozturk et al. used the dataset to further investigate the classification of RF fingerprints at low SNRs by adding white Gaussian noise to the raw data. Using a CNN, they achieved classification accuracies ranging from 92% to 100% for SNR ∈ [-10, 30]dB.

The openly available DroneDetect dataset was created by Swinney and Woods. It contains raw in-phase and quadrature (IQ) data recorded with a BladeRF SDR. Seven drone models were recorded in three different flight modes (on, hovering, flying). Measurements were also repeated with different types of noise, such as interference from a Bluetooth speaker, a Wi-Fi hotspot, and simultaneous Bluetooth and Wi-Fi interference. The dataset does not include measurements without drones, which would be necessary to evaluate a drone detection system. The results in show that Bluetooth signals are more likely to interfere with detection and classification accuracy than Wi-Fi signals. Overall, frequency domain features extracted from a CNN were shown to be more robust than time domain features in the presence of interference.

In the drone signals from the DroneDetect dataset were augmented with Gaussian noise and SDR recorded background noise. Hence, the proposed approach could be evaluated regrading its capability to detect drones. They trained a CNN end-to-end on the raw IQ data and report an accuracy of 99% for detection and between 72% and 94% for classification.

The Cardinal RF dataset consists of the raw time series data from six drones + controller, two Wi-Fi and two Bluetooth devices. Based on this dataset, Medaiyese et al. proposed a semi-supervised framework for UAV detection using wavelet analysis. Accuracy between 86% and 97% was achieved at SNRs of 30 dB and 18 dB, while it dropped to chance level for SNRs below 10 dB to 6dB. In addition, investigated different wavelet transforms for the feature extraction from the RF signals. Using the wavelet scattering transform from the steady state of the RF signals at 30 dB SNR to train SqueezeNet, they achieved an accuracy of 98.9% at 10dB SNR.

In our previous work, we created the noisy drone RF signals dataset¹ from six drones and four remote controllers. It consists of non-overlapping signal vectors of 16384 samples, corresponding to ≈ 1.2 ms at 14 MHz. We added Labnoise (Bluetooth, Wi-Fi, Amplifier) and Gaussian noise to the dataset and mixed it with the drone signals with SNR ∈ [-20, 30] dB. Using IQ data and spectrogram data to train different CNNs, we found an advantage in favor of the 2D spectrogram representation of the data. There was no performance difference at SNR ≥ 0dB but a major improvement in the balanced accuracy at low SNR levels, i.e., 84.2% on the spectrogram data compared to 41.3% on the IQ data at -12 dB SNR.

Recently, proposed an anchor-free object detector based on keypoints for drone RF signal spectograms. They also proposed an adversarial learning-based data adaptation method to generate domain independent and domain aligned features. Given five different types of drones, they report a mean average precision of 97.36%, which drops to ≈ 55% when adding Gaussian noise with -25 dB SNR. The raw data used in their work is available,² but yet, unfortunately not usable without any further documentation.

### B. Motivation

As we have seen in other fields, such as computer vision, the success of DL can be attributed to: (a) high-capacity models; (b) increased computational power; and (c) the availability of large amounts of labeled data. Thus, given the large amount of available raw RF signals (cf., Table I) we promote the idea of open and reusable data, to facilitate model development and model comparison.

---

¹https://www.kaggle.com/datasets/sgluege/noisy-drone-rf-signal-classification
²https://www.kaggle.com/datasets/zhaoericry/drone-rf-dataset

---

**GLÜGE et al.: ROBUST LOW-COST DRONE DETECTION AND CLASSIFICATION USING CNNs**
**823**

**TABLE II**
TRANSMITTERS AND RECEIVERS RECORDED IN THE DEVELOPMENT DATASET AND THEIR RESPECTIVE CLASS LABELS.
ADDITIONALLY, WE SHOW THE CENTER FREQUENCY (GHZ), THE CHANNEL SPACING (MHZ), THE BURST DURATION (MS),
AND THE REPETITION PERIOD OF THE RESPECTIVE SIGNALS (MS)

| Transmitter                                                                                                                                                                | Receiver          | Label     | Center Freq. (GHz) | Spacing (MHz) | Duration (ms) | Repetition (ms) |
| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------- | :-------- | :----------------- | :------------ | :------------ | :-------------- |
| DJI Phantom GL300F                                                                                                                                                         | DJI Phantom 4 Pro | DJI       | 2.44175            | 1.7           | 2.18          | 630             |
| Futaba T7C                                                                                                                                                                 |                   | FutabaT7  | 2.44175            | 2             | 1.7           | 288             |
| Futaba T14SG                                                                                                                                                               | Futaba R7008SB    | FutabeT14 | 2.44175            | 3.1           | 1.4           | 330             |
| Graupner mx-16                                                                                                                                                             | Graupner GR-16    | Graupner  | 2.44175            | 1             | 1.9/3.7       | 750             |
| Bluetooth/Wi-Fi Noise                                                                                                                                                      |                   | Noise     | 2.44175            |               |               |                 |
| Taranis ACCST                                                                                                                                                              | X8R Receiver      | Taranis   | 2.440              | 1.5           | 3.1/4.4       | 420             |
| Turnigy 9X                                                                                                                                                                 |                   | Turnigy   | 2.445              | 2             | 1.3           | 61, 120-2900 a  |
| *a The repetition period of the Turnigy transmitter is not static. First bursts were observed after 61 ms, the following signal bursts were observed in the interval ms* |                   |           |                    |               |               |                 |

With the noisy drone RF signals dataset, we have provided a first ready-to-use dataset to enable rapid model development, without the need for any data preparation. Furthermore, the dataset contains samples that can be considered as "hard" in terms of noise, i.e., Bluetooth + Wi-Fi + Gaussian noise at very low SNRs, and allows a direct comparison with the published results.

While the models proposed in performed reasonably well in the training/lab setting, we found it difficult to transfer their performance to practical application. The reason was the choice of rather short signal vectors of 16384 samples, corresponding to ≈ 1.2 ms at 14 MHz. Since the drone signals occur in short bursts of ≈ 1.3-2 ms with a repetition period of ≈ 60–600 ms, our continuously running classifier predicts a drone whenever a burst occurs and noise during the repetition period of the signal. Therefore, in order to provide a stable and reliable classification per every second, one would need an additional "layer" to pool the classifier outputs given every 1.2 ms.

In the present work, we follow a data-centric approach and simply increase the length of the input signal to ≈ 75 ms to train a classifier in an end-to-end manner. Again, we provide the data used for model development in the hope that it will inspire others to develop better models.

In the next section, we briefly describe the data collection and preprocessing procedure. Section III describes the model architectures and their training/validation method. In addition, we describe the setup of a low-cost drone detection system and of the field test. The resulting performance metrics are presented in Section IV and are further discussed in Section V.

## II. MATERIALS

We used the raw RF signals from the drones that were collected in. Nevertheless, we briefly describe the data acquisition process again to provide a complete picture of the development from the raw RF signal to the deployment of a detection system within a single manuscript.

### A. Data Acquisition

The drone’s remote control and, if present, the drone itself were placed in an anechoic chamber to record the raw RF signal without interference for at least one minute. The signals were received by a log-periodic antenna and sampled and stored by an Ettus Research USRP B210, see Fig. 1. In *Fig. 1. Recording of drone signals in the anechoic chamber. A DJI Phantom 4 Pro drone with the DJI Phantom GL300F remote control.* the static measurement, the respective signals of the remote control (TX) alone or with the drone (RX) were measured. In the dynamic measurement, one person at a time was inside the anechoic chamber and operated the remote control (TX) to generate a signal that is as close to reality as possible. All signals were recorded at a sampling frequency of 56 MHz (highest possible real-time bandwidth). All drone models and recording parameters are listed in Table II, including both uplink and downlink signals.

We also recorded three types of noise and interference. First, Bluetooth/Wi-Fi noise was recorded using the hardware setup described above. Measurements were taken in a public and busy university building. In this open recording setup, we had no control over the exact number or types of active Bluetooth/Wi-Fi devices and the actual traffic in progress. Second, artificial white Gaussian noise was used, and third, receiver noise was recorded for 30 seconds from the USRP at various gain settings ( db in steps of 10 dB) without the antenna attached. This should prevent the final model from misclassifying quantisation noise in the absence of a signal, especially at low gain settings.

### B. Data Preparation

To reduce memory consumption and computational effort, we reduced the bandwidth of the signals by downsampling from 56 MHz to 14 MHz using the SciPy signal.decimate function with an 8th order Chebyshev type I filter.

The drone signals occur in short bursts with some low power gain or background noise in between (cf., Table II). We

---

824
IEEE JOURNAL OF RADIO FREQUENCY IDENTIFICATION, VOL. 8, 2024

divided the signals into non-overlapping vectors of 1048576 samples (74.9ms) and only vectors containing a burst, or at least a partial burst, were used for the development dataset. This was achieved by applying an energy threshold. As the recordings were made in an echo-free chamber, the signal burst is always clearly visible. Hence, we only used vectors that contained a portion of the signal whose energy was above the threshold, which was arbitrarily set at 0.001 of the average energy of the entire recording.

The selected drone signal vectors x with i ∈ {1,...k} were normalized to a carrier power of 1 per sample, i.e., only the part of the signal vector containing drone bursts was considered for the power calculation (m samples out of k). This was achieved by identifying the bursts as those samples where a smoothed energy was above a threshold.

The signal vectors x are thus normalized by

$$
\tilde{x}(i) = x(i) / \sqrt{\frac{1}{m} \sum_i |x(i)|^2}. \quad (1)
$$

Noise vectors (Bluetooth, Wi-Fi, Amplifier, Gauss) n with samples i ∈ {1,...k} were normalized to a mean power of 1 with

$$
\tilde{n}(i) = n(i) / \sqrt{\frac{1}{k} \sum_i |n(i)|^2}. \quad (2)
$$

Finally, the normalized drone signal vectors were mixed with the normalized noise vectors by

$$
\hat{y}(i) = \frac{(\sqrt{k} \cdot \tilde{x}(i) + \tilde{n}(i))}{\sqrt{k+1}}, \quad \text{with } k = 10^{\frac{\text{SNR}}{10}}. \quad (3)
$$

to generate the noisy drone signal vectors $\hat{y}$ at different SNRs.

### C. Development Dataset

To facilitate future model development, we provide our resulting dataset³ along with a code example⁴ to load and inspect the data. The dataset consists of the non-overlapping signal vectors of 2²⁰ samples, corresponding to ≈ 74.9 ms at 14 MHz.

As described in Section II-B, the drone signals were mixed with noise (cf. Eqs. (1)-(3)). More specifically, 50% of the drone signals were mixed with Labnoise (Bluetooth + Wi-Fi + Amplifier) and 50% with Gaussian noise. In the same way we created a single noise class by mixing Labnoise and Gaussian noise in all possible combinations (i.e., Labnoise + Labnoise, Labnoise + Gaussian noise, Gaussian noise + Labnoise, and Gaussian noise + Gaussian noise). This mixing was done as for the drone signals using Eq. (3). For instance, in case of a Labnoise + Gaussian noise mix, $\tilde{x}$ refers to a Labnoise vector and $\tilde{n}$ to a generated Gaussian noise vector.

For the drone signal classes, as for the noise class, the number of samples for each SNR level was evenly distributed over the interval of SNRs ∈ [-20, 30] dB in steps of 2dB, i.e., 679-685 samples per SNR level. The resulting number of samples per class is given in Table III.

**TABLE III**
NUMBER OF SAMPLES IN THE DIFFERENT CLASSES IN THE DEVELOPMENT DATASET

| Class    | DJI  | FutabaT14 | FutabaT7 | Graupner | Taranis | Turnigy | Noise |
| :------- | :--- | :-------- | :------- | :------- | :------ | :------ | :---- |
| #samples | 1280 | 3472      | 801      | 801      | 1663    | 855     | 8872  |

In our previous work we found an advantage in using the spectrogram representation of the data compared to the IQ representation, especially at low SNRs levels. Therefore, we transform the raw IQ signals by computing the spectrum of each sample with consecutive Fourier transforms with non-overlapping segments of length 1024 for the real and imaginary parts of the signal. That is, the two IQ signal vectors ([2×2²⁰]) are represented as two matrices ([2×1024×1024]). Fig. 2 shows four samples of the dataset at different SNRs. Note that we have plotted the log power spectrogram of the complex spectrum $\hat{y}_{\text{fft}}$ as

$$
\log_{10} |\hat{y}_{\text{fft}}| = \log_{10} \sqrt{\text{Re}(\hat{y}_{\text{fft}})^2 + \text{Im}(\hat{y}_{\text{fft}})^2}. \quad (4)
$$

### D. Detection System Prototype

For field use, a system based on a mobile computer was used as shown in Fig. 3 and illustrated in Fig. 4. The RF signals were received using a directional left-hand circularly polarized antenna (H&S SPA_2400/70/9/0/CP). The antenna gain of 8.5 dBi and the front-to-back ratio of 20 dB helped to increase the detection range and to attenuate the unwanted interferers in the opposite direction. Circular polarization has been chosen to eliminate the alignment problem as the transmitting antennas have a linear polarization. The USRP B210 was used to down-convert and digitize the RF signal at a sampling rate of 14 Msps. On the mobile computer, the GNU Radio program collected the baseband IQ samples in batches of one second and send one batch at a time to our PyTorch model, which classified the signal. To speed up the computations in the model we utilized an Nvidia GPU in computer. The classification results were then visualized in real time in a dedicated GUI.

## III. METHODS

### A. Model Architecture and Training

As in we chose the Visual Geometry Group (VGG) CNN architecture due to its performance history across a wide range of tasks, including medical image classification, as backbone for object detection, and audio signal classification.

The main idea of this architecture is to use multiple layers of small (3 × 3) convolutional filters instead of larger ones. This is intended to increase the depth and expressiveness of the network, while reducing the number of parameters. There are several variants of this architecture, which differ in the number of convolutional layers (11 and 19, respectively). We used a variant with a batch normalization layer after the convolutions, denoted as VGG11_BN to VGG19_BN. For the dense classification layer, we used 256 linear units followed by 7 linear units at the output (one unit per class).

---

³https://www.kaggle.com/datasets/sgluege/noisy-drone-rf-signal-classification-v2
⁴https://github.com/sgluege/noisy-drone-rf-signal-classification-v2

---

**GLÜGE et al.: ROBUST LOW-COST DRONE DETECTION AND CLASSIFICATION USING CNNs**
**825**

*Fig. 2. Log power spectrogram and IQ data samples from the development dataset at different SNRs (a-d).*

*Fig. 3. Block diagram of the mobile drone detection system.*

*Fig. 4. Detection prototype at the Zurich Lake in Rapperswil.*

A stratified 5-fold train-validation-test split was used as follows. In each fold, we trained a network using 80% and 20% of the available samples of each class for training and testing, respectively. Repeating the stratified split five times ensures that each sample was in the test set once in each experiment. Within the training set, 20% of the samples were used as the validation set during training.

Model training was performed for 200 epochs with a batch size of 8. The PyTorch implementation of the Adam algorithm was used with a learning rate of 0.005, betas (0.9, 0.999) and weight decay of 0. It was conducted on a NVIDIA A100 80GB GPU, requiring 25GB of memory and 20h of computational time for the VGG11_BN training, and 40GB of memory and 40h of computational time for the VGG19_BN.

### B. Model Evaluation

During training, the model was evaluated on the validation set after each epoch. If the balanced accuracy on the validation set increased, it was saved. After training, the model with the highest balanced accuracy on the validation set was evaluated on the withheld test data. The performance of the models on the test data was accessed in terms of classification accuracy and balanced accuracy.

As accuracy simply measures the proportion of correct predictions out of the total number of observations, it can be misleading for unbalanced datasets. In our case, the noise class is over-represented in the dataset (cf., Table III). Therefor, we also report the balanced accuracy, which is defined as the average of the recall obtained for each class, i.e., it gives equal weight to each class regardless of how frequent or rare it is.

### C. Visualization of Model Embeddings

Despite their effectiveness, CNNs are often criticized for being "black boxes". Understanding the feature representations, or embeddings, learned by the CNN helps to demystify these models and provide some understanding of their capabilities and limitations. In general, embeddings are high-dimensional vectors generated by the intermediate layers that capture essential patterns from the input data.

---

826
IEEE JOURNAL OF RADIO FREQUENCY IDENTIFICATION, VOL. 8, 2024

**TABLE IV**
DRONES AND/OR REMOTES USED IN THE FIELD TEST

| Class     | Drone/remote control               |
| :-------- | :--------------------------------- |
| DJI       | DJI Phantom Pro 4 drone and remote |
| FutabaT14 | Futaba T14 remote control          |
| FutabaT7  | Futaba T14 remote control          |
| Taranis   | FrySky Taranis Q X7 remote control |
| Turnigy   | Turnigy Evolution remote control   |

**TABLE V**
NUMBER OF SAMPLES (#SAMPLES) FOR EACH CLASS, DISTANCE AND ANTENNA DIRECTION (ANGLE) RECORDED IN THE FIELD TEST. RECORDINGS AT OM DISTANCE HAVE NO ACTIVE TRANSMITTER AND WERE THEREFORE LABELLED "NOISE"

| class     | #samples | Distance [m] | #samples | Angle [°] | #samples |
| :-------- | :------- | :----------- | :------- | :--------- | :------- |
| DJI       | 4900     | 0            | 2597     | 0          | 9110     |
| FutabaT14 | 5701     | 110          | 6208     | 90         | 9076     |
| FutabaT7  | 5086     | 340          | 6305     | 180        | 9226     |
| Noise     | 2597     | 560          | 4032     |            |          |
| Taranis   | 2597     | 670          | 6358     |            |          |
| Turnigy   | 5094     |              | 5942     |            |          |

**TABLE VI**
MEAN ± STANDARD DEVIATION OF THE ACCURACY (ACC.) AND THE BALANCED ACCURACY (BALANCED ACC.) OBTAINED IN 5-FOLD CROSS-VALIDATION OF THE DIFFERENT MODELS ON THE TEST DATA OF THE DEVELOPMENT DATASET. AN INDICATION OF THE MODEL TRAINING TIME IS GIVEN WITH THE MEAN STANDARD DEVIATION OF THE NUMBER OF TRAINING EPOCHS (#EPOCHS), I.E., WHEN THE HIGHEST BALANCED ACCURACY ON THE VALIDATION SET WAS REACHED. THE NUMBER OF TRAINABLE PARAMETERS (#PARAMS) INDICATES THE COMPLEXITY OF THE MODEL

| Model    | Acc.           | balanced Acc.  | #epochs       | #params       |
| :------- | :------------- | :------------- | :------------ | :------------ |
| VGG11_BN | 0.944 ± 0.005 | 0.932 ± 0.002 | 66.4 ± 24.4  | 9.36 · 10⁶  |
| VGG13_BN | 0.947 ± 0.003 | 0.935 ± 0.003 | 138.6 ± 46.4 | 9.54 · 10⁶  |
| VGG16_BN | 0.947 ± 0.006 | 0.937 ± 0.005 | 101.8 ± 41.5 | 14.86 · 10⁶ |
| VGG19_BN | 0.952 ± 0.006 | 0.939 ± 0.008 | 98.2 ± 45.8  | 20.17 · 10⁶ |

*Fig. 5. Experimental measurement setup at the Zurich Lake in Rapperswil. One can see the four recording positions along the wooden walkway and the detection system positioned at the lake side. Further, recordings were done at different angels of the directional antenna indicated by the arrows at the detection system.*

In our case, we chose the least complex VGG11_BN model to visualize its embeddings. When inferencing the test data, we collected the activations at the last dense classification layer, which consists of 256 units. Given 3549 test samples, this results in a 256 × 3549 matrix. Using t-distributed Stochastic Neighbor Embedding (t-SNE) and Uniform Manifold Approximation and Projection (UMAP) as dimensionality reduction techniques, we project these high-dimensional embeddings into a lower-dimensional space, creating interpretable visualizations that reveal the model’s internal data representations.

Our goals were to understand how the model clusters and separates different classes, to identify potential overlaps or ambiguities, and to examine the hierarchical relationships within the learned features.

### D. Detection System Field Test

We conducted a field test of the detection system in Rapperswil at the Zurich Lake. The drone detection prototype was placed on the shore (cf., Fig. 4) in line of sight of a wooden boardwalk across the lake, with no buildings to interfere with the signals. The transmitters were mounted on a 2.5 m long wooden pole. The signals from the transmitters were recorded (and classified in real time) at four positions along the walkway at approximately 110m, 340 m, 560 m and 670 m from the detection system. Figure 5 shows an overview of the experimental setup. At each recording position, we measured with the directional antenna at three different angles, i.e., at 0° – facing the drones and/or remote controls, at 90° perpendicular to the direction of the transmitters, and at 180° in the opposite direction. Directing the antenna in the opposite direction should result in ≈ 20 dB attenuation of the radio signals.

Table IV lists the drones and/or remote controls used in the field test. Note that the Graupner drone and remote control are part of the development dataset (cf., Table II), but were not measured in the field experiment. We assume that no other drones were present during the measurements, so recordings where none of our transmitters were used are labelled as "Noise".

For each transmitter, distance, and angle, 20 to 30s, or approximately 300 spectrograms were live classified and recorded. The resulting number of samples for each class, distance, and angle are shown in Table V.

## IV. RESULTS

### A. Classification Performance in the Development Dataset

Table VI shows the general mean ± standard deviation of accuracy and balanced accuracy on the test data of the development dataset (cf., Section II-C), obtained in the 5-fold cross-validation of the different models.

There is no meaningful difference in performance between the models, even when the model complexity increases from VGG11_BN to VGG19_BN. The number of epochs for training (#epochs) shows when the highest balanced accuracy was reached on the validation set. It can be seen that the least complex model, VGG11_BN, required the least number of epochs compared to the more complex models. However, the resulting classification performance is the same.

---

**GLÜGE et al.: ROBUST LOW-COST DRONE DETECTION AND CLASSIFICATION USING CNNs**
**827**

*Fig. 6. Mean balanced accuracy ± std obtained in the 5-fold cross-validation of the different models on the test set of the development dataset over the SNRs levels.*

*Fig. 7. Confusion matrix of the outputs of the VGG11_BN model on a single fold for the samples at −14 dB SNR from the test data. The average balanced accuracy is 0.71.*

Figure 6 shows the resulting 5-fold mean balanced accuracy over SNRs ∈ [−20, −4] dB in 2 dB steps. Note that we do not show the results for SNRs > −4dB, as those are simply saturated at 100% balanced accuracy.
In general, we observe a drastic degradation in performance from −12 dB down to near chance level at −20 dB.

The vast majority of misclassifications occurred between noise and drones and not between different types of drones. Figure 7 illustrates this fact. It shows the confusion matrix for the VGGG11_BN model for a single validation on the test data for the samples with −14dB SNR.

### B. Embedding Space Visualization

Figure 8 shows the 2D t-SNE visualization of the VGG11_BN embeddings of 3549 test samples from the development dataset. Each class forms a separate cluster. While the different drone signal clusters are rather small and dense, the noise cluster takes up most of the embedding space and even forms several sub-clusters. This is most likely due to the variety of the signals used in the noise class, i.e., Bluetooth and
*Fig. 8. 2D t-SNE visualization of the VGG11_BN embeddings of 3549 test samples from the development dataset. The hyperparameters for t-SNE were: metric "eucleidean", number of iterations 1000, perplexity 30 and method for gradient approximation "barnes_hut."*

**TABLE VII**
MEAN ± STANDARD DEVIATION OF THE BALANCED ACCURACY (BALANCED ACC.) OF THE COMPLETE FIELD TEST RECORDINGS FOR THE DIFFERENT MODELS

| Model    | balanced Acc.  |
| :------- | :------------- |
| VGG11_BN | 0.792 ± 0.022 |
| VGG13_BN | 0.807 ± 0.011 |
| VGG16_BN | 0.811 ± 0.009 |
| VGG19_BN | 0.806 ± 0.016 |

Wi-Fi signals plus Gaussian noise. It can also be seen that the DJI and FutabaT14 classes are more difficult to separate from noise than other classes (cf. Fig. 7) as they tend to overlap at the clusters’ edges.

We used t-SNE for dimensionality reduction because of its ability to preserve local structure within the high-dimensional embedding space. Furthermore, t-SNE has been widely adopted in the ML community and has a well-established track record for high-dimensional data visualization. However, it is sensitive to hyperparameters such as perplexity and requires some tuning, i.e., different parameters can lead to considerable different results.

It can be argued that UMAP would be a better choice due to its balanced preservation of local and global structure together with its robustness to hyperparameters. Therefore, we created a Web application⁵ that allows users to test and compare both approaches with different hyperparameters.

### C. Classification Performance in the Field Test

For each model architecture, we performed 5-fold cross-validation on the development dataset (cf., Section III-A), resulting in five trained models per architecture. Thus, we also evaluated all five trained models on the field test data. We report the balanced accuracy ± standard deviation for each model architecture for the complete field test dataset averaged over all directions and distances in Table VII.

As observed on the development dataset (cf., Table VI), there is no meaningful difference in performance between the model architectures. We therefore focus on VGG11_BN, the simplest model trained, in the more detailed analysis of the field test results.

---

⁵https://visvgg11bndronerfembeddings.streamlit.app

---

828
IEEE JOURNAL OF RADIO FREQUENCY IDENTIFICATION, VOL. 8, 2024

**TABLE VIII**
MEAN BALANCED ACCURACY ± STANDARD DEVIATION OF THE VGG11_BN MODELS ON THE FIELD TEST RECORDINGS FOR THE DIFFERENT CLASSES FOR EACH DIRECTION (0°, 90° AND 180°). THE UPPER PART SHOWS THE ACCURACIES FOR THE CLASSIFICATION PROBLEM (SEVEN CLASSES) AND THE LOWER PART THE ACCURACIES FOR THE DETECTION PROBLEM “DRONE” OR “NOISE”

| Class     | 0°            | 90°           | 180°          |
| :-------- | :------------- | :------------- | :------------- |
| DJI       | 0.623 ± 0.080 | 0.624 ± 0.035 | 0.540 ± 0.051 |
| FutabaT14 | 0.716 ± 0.101 | 0.984 ± 0.011 | 0.911 ± 0.042 |
| FutabaT7  | 0.724 ± 0.041 | 0.737 ± 0.034 | 0.698 ± 0.059 |
| Graupner  | 0.936 ± 0.008 | 0.924 ± 0.026 | 0.858 ± 0.002 |
| Taranis   | 0.899 ± 0.129 | 0.954 ± 0.027 | 0.962 ± 0.024 |
| Turnigy   | 0.958 ± 0.011 | 0.859 ± 0.014 | 0.847 ± 0.017 |
| Drone     | 0.554 ± 0.038 | 0.924 ± 0.026 | 0.833 ± 0.068 |
| Noise     |                |                |                |

**TABLE IX**
MEAN BALANCED ACCURACY ± STANDARD DEVIATION OF THE VGG11_BN MODELS ON THE FIELD TEST DATA WITH ACTIVE TRANSMITTERS COLLECTED AT DIFFERENT DISTANCES FOR EACH ANTENNA DIRECTION (0°, 90° AND 180°). THE UPPER PART SHOWS THE ACCURACIES FOR THE CLASSIFICATION PROBLEM (SEVEN CLASSES) AND THE LOWER PART THE ACCURACIES FOR THE DETECTION PROBLEM “DRONE” OR “NOISE”

| Classification      | Distance (m)           | 0°            | 90°           | 180°           |
| :------------------ | :--------------------- | :------------- | :------------- | :-------------- |
| 110                 | 0.852 ± 0.044         | 0.838 ± 0.024 | 0.786 ± 0.044 |                 |
| 340                 | 0.815 ± 0.078         | 0.916 ± 0.017 | 0.828 ± 0.031 |                 |
| 560                 | 0.730 ± 0.063         | 0.796 ± 0.007 | 0.764 ± 0.017 |                 |
| 670                 | 0.708 ± 0.068         | 0.805 ± 0.008 | 0.777 ± 0.007 |                 |
| **Detection** | **Distance (m)** | **0°**  | **90°** | **180°** |
| 110                 | 0.955 ± 0.007         | 0.851 ± 0.025 | 0.849 ± 0.018 |                 |
| 340                 | 0.986 ± 0.008         | 0.940 ± 0.010 | 0.913 ± 0.018 |                 |
| 560                 | 0.960 ± 0.011         | 0.820 ± 0.013 | 0.807 ± 0.022 |                 |
| 670                 | 0.925 ± 0.021         | 0.826 ± 0.013 | 0.823 ± 0.014 |                 |

*Fig. 9. Confusion matrix of the outputs of the VGG11_BN model on a single fold for the samples from the field test data. The average balanced accuracy is 0.80.*

distances for each antenna direction. There is a slight decrease in accuracy with distance. However, the longest distance of 670 m appears to be too short to be a problem for the system. Unfortunately, this was the longest distance within line-of-sight that could be recorded at this location.

Figure 9 shows the confusion matrix for the outputs of the VGG11_BN model of a single fold on the field test data. As with the development dataset (cf., Fig. 7), most of the confusion is between noise and drones rather than between different types of drones.

## V. DISCUSSION

We were able to show that a standard CNN, trained on drone RF signals recorded in a controlled laboratory environment and artificially augmented with noise, generalized well to the more challenging conditions of a real-world field test.

The drone detection system consisted of rather simple and low budget hardware (consumer grade notebook with GPU + SDR). Recording parameters such as sampling frequency, length of input vectors, etc. were set to enable real-time detection with the limited amount of memory and computing power. This means that data acquisition, pre-processing and model inference did not take longer than the signal being processed (≈ 74.9 ms per sample in our case).

Obviously, the VGG models were able to learn the relevant features for the drone classification from the complex spectrograms of the RF signal. In this respect, we did not find any advantage for the use of more complex models, such as VGG19_BN, over the least complex model, VGG11_BN (cf., Tables VI and VII).

Furthermore, we have seen that the misclassifications mainly occur between the noise class and the drones, and not between the different drones themselves (cf., Figs. 7 and 9). This is particularly relevant for the application of drone detection systems in security sensitive areas. The first priority is to detect any kind of UAV, regardless of its type.

Based on our experience and results, we see the following limitations of our work. The RF signals of drones were

A live system should trigger an alarm when a drone is present. Therefore, the question of whether the signal is from a drone at all is more important than predicting the correct type of drone. Therefore, we also evaluated the models in terms of a binary problem with two classes “Drone” (for all six classes of drones in the development dataset) and “Noise”.

Table VIII shows that the accuracies were highly depend on the class. Our models generalize well to the drones in the dataset, with the exception of the DJI. The dependence on direction is not as strong as expected. Orienting the antenna 180° away from the transmitter reduces the signal power by about 20 dB, resulting in lower SNR and lower classification accuracy. However, as the transmitters were still quite close to the antenna, the effect is not pronounced. As we have seen on the development dataset in Fig. 6, there is a clear drop in accuracy once the SNR is below −12 dB. Apparently we were still above this threshold, regardless of the direction of the antenna.

What may be surprising is the low accuracy on the signals with no active transmitter, labelled as "Noise", in the direction of the lake (0°). Given the uncontrolled nature of a field test, it could well be that there a drone was actually flying on the other side of the 2.3 km wide lake. This could explain the false positives we observed in that direction.

Table IX shows the average balanced accuracy of the VGG11_BN models on the field test data collected at different

---

**GLÜGE et al.: ROBUST LOW-COST DRONE DETECTION AND CLASSIFICATION USING CNNs**
**829**

recorded in a controlled laboratory environment and augmented with WiFi/Bluetooth/Gaussian noise (cf. Section II). While the laboratory environment was necessary for the initial development and evaluation of the model, these signals do not fully replicate the complexities encountered in real-world scenarios. None the less, the field test showed that the models can be used and work reliably (cf., Table VIII) in real world conditions. However, it is the nature of a field test that the level of interference from WiFi/Bluetooth noise and the possible presence of other drones cannot be fully controlled. Furthermore, due to the limited space/distance between the transmitter and receiver in our field test setup, we were not able to clearly demonstrate the effect of free space attenuation on detection performance (cf., Table IX).

Regarding the use of simple CNNs as classifiers, it is not possible to reliably predict whether multiple transmitters are present. In that case, an object detection approach on the spectrograms could provide a more fine-grained prediction, see for example the works,, and. Nevertheless, the current approach will still detect a drone if one or more are present.

We have only tested a limited set of VGG architectures. It remains to be seen whether more recent architectures, such as the pre-trained Vision Transformer, generalize as well or better. Further, with the deployment of the model on limited computational resources in mind, the use of more efficient and less complex models, such as GoogLeNet and ShuffleNet should be investigated. Research could aim for a balance between detection accuracy and the computational demands of real-time processing. By doing so, the deployment of a robust, low-latency drone detection system in dynamic and resource-constrained settings could become a practical reality.

Another issue to consider is the occurrence of unknown drones, i.e., drones that are not part of the train set. Examining the embedding space (cf. Section IV-B) gives a first idea of whether a signal is clearly part of a known dense drone cluster or rather falls into the larger, less dense, noise cluster. We believe that a combination of an unsupervised deep autoencoder approach, with an additional classification part (cf. ) would allow, first, to provide a stable classification of known samples and, second, to indicate whether a sample is known or rather an anomaly. This could be accompanied by additional field test to further access the model’s generalization ability to drones it was not trained on.

Finally, the current lack of standardized benchmarks makes it challenging to compare the performance of various models and systems objectively. We hope that our development dataset will inspire others to further optimize the model side of the problem and perhaps find a model architecture with better performance and/or efficiency.

## REFERENCES

 N. Al-lQubaydhi et al., “Deep learning for unmanned aerial vehicles detection: A review,” *Comput. Sci. Rev.*, vol. 51, Feb. 2024, Art. no. 100614. [Online]. Available: https://linkinghub.elsevier.com/retrieve/pii/S1574013723000813
 M. H. Rahman, M. A. S. Sejan, M. A. Aziz, R. Tabassum, J.-I. Baik, and H.-K. Song, “A comprehensive survey of unmanned aerial vehicles detection and classification using machine learning approach: Challenges, solutions, and future directions,” *Remote Sens.*, vol. 16, no. 5, p. 879, 2024. [Online]. Available: https://www.mdpi.com/2072-4292/16/5/879
 M. S. Allahham, M. F. Al-Sa'd, A. Al-Ali, A. Mohamed, T. Khattab, and A. Erbad, “DroneRF dataset: A dataset of drones for RF-based detection, classification and identification,” *Data Brief*, vol. 26, Oct. 2019, Art. no. 104313. [Online]. Available: https://linkinghub.elsevier.com/retrieve/pii/S2352340919306675
 M. F. Al-Sa'd, A. Al-Ali, A. Mohamed, T. Khattab, and A. Erbad, “RF-based drone detection and identification using deep learning approaches: An initiative towards a large open source drone database,” *Future Gener. Comput. Syst.*, vol. 100, pp. 86–97, Nov. 2019.
 C. J. Swinney and J. C. Woods, “Unmanned aerial vehicle flight mode classification using convolutional neural network and transfer learning,” in *Proc. 16th Int. Comput. Eng. Conf. (ICENCO)*, 2020, pp. 83–87.
 Y. Zhang, “RF-based drone detection using machine learning,” in *Proc. 2nd Int. Conf. Comput. Data Sci. (CDS)*, 2021, pp. 425–428.
 C. Ge, S. Yang, W. Sun, Y. Luo, and C. Luo, “For RF signal-based UAV states recognition, is pre-processing still important at the era of deep learning?” in *Proc. 7th Int. Conf. Comput. Commun. (ICCC)*, 2021, pp. 2292–2296.
 Y. Xue et al., “UAV signal recognition of heterogeneous integrated KNN based on genetic algorithm,” *Telecommun. Syst.*, vol. 85, no. 4, pp. 591–599, 2024. [Online]. Available: https://link.springer.com/10.1007/s11235-023-01099-x
 A. AlKhonaini, T. Sheltami, A. Mahmoud, and M. Imam, “UAV detection using reinforcement learning,” *Sensors*, vol. 24, no. 6, p. 1870, 2024. [Online]. Available: https://www.mdpi.com/1424-8220/24/6/1870
 M. Ezuma, F. Erden, C. K. Anjinappa, O. Ozdemir, and I. Guvenc, “Detection and classification of UAVs using RF fingerprints in the presence of Wi-Fi and Bluetooth interference,” *IEEE Open J. Commun. Soc.*, vol. 1, pp. 60–76, 2020. [Online]. Available: https://ieeexplore.ieee.org/document/8913640/
 M. Ezuma, F. Erden, C. K. Anjinappa, O. Ozdemir, and I. Guvenc, 2020, “Drone remote controller RF signal Dataset,” Dataset. [Online]. Available: https://dx.doi.org/10.21227/ss99-8d56
 E. Ozturk, F. Erden, and I. Guvenc, “RF-based low-SNR classification of UAVs using convolutional neural networks,” *ITU J. Future Evol. Technol.*, vol. 2, pp. 39–52, Jul. 2021. [Online]. Available: https://www.itu.int/pub/S-JNL-VOL2.ISSUE5-2021-A04
 C. J. Swinney and J. C. Woods, 2021, “DroneDetect dataset: A radio frequency dataset of unmanned aerial system (UAS) signals for machine learning detection & classification,” Dataset. [Online]. Available: https://dx.doi.org/10.21227/5jjj-1m32
 C. J. Swinney and J. C. Woods, “RF detection and classification of unmanned aerial vehicles in environments with wireless interference,” in *Proc. Int. Conf. Unmanned Aircr. Syst. (ICUAS)*, 2021, pp. 1494–1498.
 S. Kunze and B. Saha, “Drone classification with a convolutional neural network applied to raw IQ data,” in *Proc. 3rd URSI Atlantic Asia Pac. Radio Sci. Meeting (AT-AP-RASC)*, May 2022, pp. 1–4. [Online]. Available: https://ieeexplore.ieee.org/document/9814170/
 O. Medaiyese, M. Ezuma, A. Lauf, and A. Adeniran, 2022, “Cardinal RF (CardRF): An outdoor UAV/UAS/drone RF signals with Bluetooth and WiFi signals dataset,” Dataset. [Online]. Available: https://dx.doi.org/10.21227/1xp7-ge95
 O. O. Medaiyese, M. Ezuma, A. P. Lauf, and A. A. Adeniran, “Hierarchical learning framework for UAV detection and identification,” *IEEE J. Radio Freq. Identif.*, vol. 6, pp. 176–188, Mar. 2022, doi: 10.1109/JRFID.2022.3157653.
 O. O. Medaiyese, M. Ezuma, A. P. Lauf, and I. Guvenc, “Wavelet transform analytics for RF-based UAV detection and identification system using machine learning,” *Pervasive Mob. Comput.*, vol. 82, Jun. 2022, Art. no. 101569. [Online]. Available: https://linkinghub.elsevier.com/retrieve/pii/S1574119222000219
 F. N. Iandola, M. W. Moskewicz, K. Ashraf, S. Han, W. J. Dally, and K. Keutzer, “SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <1MB model size,” 2016, arXiv:1602.07360.
 S. Glüge, M. Nyfeler, N. Ramagnano, C. Horn, and C. Schüpbach., “Robust drone detection and classification from radio frequency signals using convolutional neural networks,” in *Proc. 15th Int. Joint Conf. Comput. Intell.*, 2023, pp. 496–504.

---

830
IEEE JOURNAL OF RADIO FREQUENCY IDENTIFICATION, VOL. 8, 2024

 R. Zhao, T. Li, Y. Li, Y. Ruan, and R. Zhang, “Anchor-free multi-UAV detection and classification using spectrogram,” *IEEE Internet Things J.*, vol. 11, no. 3, pp. 5259–5272, Feb. 2024. [Online]. Available: https://ieeexplore.ieee.org/document/10221859/
 C. Sun, A. Shrivastava, S. Singh, and A. Gupta, “Revisiting unreasonable effectiveness of data in deep learning era,” in *Proc. IEEE Int. Conf. Comput. Vis. (ICCV)*, 2017, pp. 843–852.
 P. Virtanen et al., “SciPy 1.0: Fundamental algorithms for scientific computing in python,” *Nat. Methods*, vol. 17, no. 3, pp. 261–272, 2020.
 K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” in *Proc. 3rd Int. Conf. Learn. Represent. (ICLR)*, 2015, pp. 1–14.
 H.-C. Shin et al., “Deep convolutional neural networks for computer-aided detection: CNN architectures, dataset characteristics and transfer learning,” *IEEE Trans. Med. Imag.*, vol. 35, no. 5, pp. 1285–1298, May 2016.
 S. Ren, K. He, R. Girshick, and J. Sun, “Faster R-CNN: Towards real-time object detection with region proposal networks,” in *Proc. 28th Int. Conf. Neural Inf. Process. Syst.*, 2015, pp. 91–99.
 S. Hershey et al., “CNN architectures for large-scale audio classification,” in *Proc. IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP)*, 2017, pp. 131–135.
 S. Ioffe and C. Szegedy, “Batch normalization: Accelerating deep network training by reducing internal covariate shift,” in *Proc. 32nd Int. Conf. Int. Conf. Mach. Learn.*, 2015, pp. 448–456.
 A. Paszke et al., “PyTorch: An imperative style, high-performance deep learning library,” in *Proc. 33rd Int. Conf. Neural Inf. Process. Syst.*, 2019, pp. 8026–8037.
 D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” in *Proc. 3rd Int. Conf. Learn. Represent. (ICLR)*, 2015, pp. 1–15.
 L. van der Maaten and G. Hinton, “Visualizing data using t-SNE,” *J. Mach. Learn. Res.*, vol. 9, no. 86, pp. 2579–2605, 2008. [Online]. Available: http://jmlr.org/papers/v9/vandermaaten08a.html
 L. McInnes, J. Healy, N. Saul, and L. Großberger, “UMAP: Uniform manifold approximation and projection,” *J. Open Source Softw.*, vol. 3, no. 29, p. 861, 2018. [Online]. Available: https://doi.org/10.21105/joss.00861
 K. N. R. S. V. Prasad and V. K. Bhargava, “A classification algorithm for blind UAV detection in wideband RF systems,” in *Proc. IEEE 92nd Veh. Technol. Conf.*, 2020, pp. 1–7.
 S. Basak, S. Rajendran, S. Pollin, and B. Scheers, “Combined RF-based drone detection and classification,” *IEEE Trans. Cogn. Commun. Netw.*, vol. 8, no. 1, pp. 111–120, Mar. 2022.
 A. Dosovitskiy et al., “An image is worth 16×16 words: Transformers for image recognition at scale,” in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2021, pp. 1–22.
 C. Szegedy et al., “Going deeper with convolutions,” in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2015, pp. 1–9. [Online]. Available: https://doi.ieeecomputersociety.org/10.1109/CVPR.2015.7298594
 X. Zhang, X. Zhou, M. Lin, and J. Sun, “ShuffleNet: An extremely efficient convolutional neural network for mobile devices,” in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2018, pp. 6848–6856. [Online]. Available: https://doi.ieeecomputersociety.org/10.1109/CVPR.2018.00716
 S. Lu and R. Li, “DAC-deep autoencoder-based clustering: A general deep learning framework of representation learning,” in *Intelligent Systems and Applications (Lecture Notes in Networks and Systems)*, vol. 294, K. Arai, Ed., Cham, Switzerland: Springer, 2022, pp. 205–216. [Online]. Available: https://doi.org/10.1007/978-3-030-82193-7_13
 H. Zhou, J. Bai, Y. Wang, J. Ren, X. Yang, and L. Jiao, “Deep radio signal clustering with interpretability analysis based on saliency map,” *Digit. Commun. Netw.*, Jan. 2023, to be published. [Online]. Available: https://linkinghub.elsevier.com/retrieve/pii/S2352864823000238
 E. Pintelas, I. E. Livieris, and P. E. Pintelas, “A convolutional autoencoder topology for classification in high-dimensional noisy image datasets,” *Sensors*, vol. 21, p. 7731, Nov. 2021. [Online]. Available: https://www.mdpi.com/1424-8220/21/22/7731

**Stefan Glüge** received the master’s degree in engineering with a focus on context-dependent learning for biological behavior modeling and the Ph.D. degree in implicit sequence learning in recurrent neural networks from Otto-von-Guericke University Magdeburg. He is a Senior Researcher with the Institute for Computational Life Sciences, Zurich University of Applied Sciences, Switzerland. With over a decade of experience, he specializes in machine learning, particularly in developing learning algorithms for recurrent neural networks. He has made significant contributions to the field of applied machine learning and computer vision, participating in various Innosuisse/CTI projects. Notable projects involve AI for hematological disease classification, machine learning for equine reproductive monitoring, and explainable deep learning models for medical time series data. His research includes deep learning applications in medical imaging, automated monitoring systems, and pattern recognition.

**Matthias Nyfeler** received the Ph.D. degree in theoretical physics on the numerical simulations of strongly correlated electron systems from the Albert Einstein Center for Fundamental Physics ,University of Bern, Switzerland. He is a Senior Lecturer with the Institute for Computational Life Sciences, Zurich University of Applied Sciences, Switzerland. He teaches mathematical modeling, machine learning, statistics lectures and heads the Applied Computational Life Sciences Master’s programme and has been leading various projects on deep learning, drone radio signal detection, bioacoustics, signal processing, physical computing, and education.

**Ahmad Aghaebrahimian** received the Ph.D. degree in computer science majoring in natural language processing and deep learning from Charles University, Prague. He is a Research Associate with the Institute for Computational Life Sciences, Zurich University of Applied Sciences, Switzerland. He is a Computer Scientist and Linguist by training. Throughout the last several years, he has been the PI and co-PI of several Innosuisse and DIZH projects in medical text analytics, food technology, supply chain monitoring, and bioinformatics. His areas of interest include NLP, large language models, neural-symbolic learning, deep learning, and semantic Web.

**Nicola Ramagnano** received the diploma degree in electrical engineering (dipl. Ing. FH) in Elektrotechnik from the University of Applied Sciences Eastern Switzerland, Rapperswil. After a few years in industry, he is currently a Research Assistant with the Institute for Communication Systems, Eastern Switzerland University of Applied Sciences. His main areas of interest include signal processing, wireless communications, software defined radio, and as well as RF and microwave electronics.

**Christof Schüpbach** received the Ph.D. and M.Sc. degrees in theoretical particle physics from the Albert Einstein Center for Fundamental Physics, University of Bern, Bern, Switzerland. He is currently a Scientific Project Manager with the Communications, Specialized Service Networks and Protection Group, Swiss Department of Defense, Armasuisse Science and Technology. He leads the Armasuisse Research Project on passive radar using civilian digital broadcasting transmitters, and works on projects in the fields of electronic warfare, timing and synchronization, and software-defined radio.
