# K-Means Clustering Approach to UAS Classification via Graphical Signal Representation of Radio Frequency Signals for Air Traffic Early Warning

**Carolyn J. Swinney**, *Student Member, IEEE*, and **John C. Woods**

---

***Abstract*—Small Unmanned Aerial Systems (UASs) provide significant benefits to economies across the world on a daily basis but increased usage brings a number of security challenges. For example, identifying small UASs operating with malicious intent in a restricted airspace. Supervised learning techniques applied to radio frequency (RF) signals have been considered for the classification of UAS type with high accuracy but due to labelled data assume the UAS signal is already known. Unsupervised learning algorithms such as K-means clustering provide a potential for identifying small UAS signals which have not been seen before. The use of transfer learning and CNN feature extraction (FE) with spectrogram graphical signal representations have been successfully used in a supervised manner for medical diagnosis and audio classification. This research is the first application of transfer learning and CNN FE as a pre-cursor to an unsupervised learning algorithm. This paper shows that clustering graphical representations of the signal and utilising CNN FE with transfer learning produces the highest v-measure score 0.814 but at a cost of 6s in time. Small UASs can travel at speeds of 45 mph so timely detection is essential in many use cases. A decrease in 0.2 v-measure score using PSD graphical image representations of the RF signal and PCA initialisation allows the clustering time to complete in under 0.3s even in environments with active interference in the same band. This timely result could provide effective early warning with the cueing of a secondary sensor or supervised algorithm with higher classification accuracy.**

***Index Terms*—Unmanned aerial systems, UAS, unmanned aerial vehicles, UAV, drones, detection, classification, security.**

---

## I. INTRODUCTION

THE potential limitless ways in which small UASs could benefit society and the economy vary from mail delivery to remote UK islands to the easing of road congestion. The UK is predicted to benefit from 600,000 new jobs and net cost savings of £16 billion by 2030. The Civilian Aviation Authority define small unmanned aircraft to weigh under 20kg without fuel but including any sensors or equipment. With the predicted increase of application and ease of availability to purchase, security from the malicious use of UASs must also be seriously considered. Four airports in the UK reported two small UAS breaches per day in 2018 and near miss incidents with aircraft recorded in 2019 deemed 62% to have posed significant safety risk. Aside from the physical safety concerns surrounding malicious small UASs entering restricted airspace, recent years have shown that simply the presence of a UAS is enough to cause disruption with serious financial consequences. The 2018 disruption at Gatwick Airport was reported to cost £50 million pounds with 1,000 flights cancelled and 140,000 passengers affected. The subsequent police operation which took 18 months cost £800,000 pounds and used 5 different UK police constabularies. Incidents at airports have continued across the world with 2021 seeing disruption in New Zealand and the U.S.. Concerns regarding the availability of small UASs to freely purchase were raised in 2021 by the U.S. Army, along with the requirement for countermeasures which are dependable. Before countermeasures can be employed it must first be determined that there is an unwanted UAS operating in a restricted airspace. This becomes more complicated when we consider the fact that UASs are being used in an increasing capacity for legitimate functions on airfields from security functions to detecting runway debris, building inspections and controlling wildlife. Different methods for detecting and classifying small UASs have been researched over the years and each have advantages and disadvantages.

RADAR is one method which has been researched significantly. Micro-Doppler signatures have been used to classify UASs and even distinguish between payloads. One of the issues which has been encountered when using RADAR is the ability of the method to successfully distinguish between birds and some UASs. Many researchers focus on this problem such as in where spectrogram and cepstrograms are used to address the issue. Kim et al. showed the use of a Convolutional Neural Network (CNN) with Doppler images to classify UASs and Choi and Oh continue this work in 2019 to consider micro-Doppler signatures and spectrogram images. Imagery based sensors is another method used for the detection and classification of UASs. The use of CNN with electro optical sensors has proved successful in increasing

*Manuscript received 28 March 2022; revised 12 July 2022; accepted 23 August 2022. Date of publication 8 September 2022; date of current version 5 December 2022. The Associate Editor for this article was L. Wang. (Corresponding author: Carolyn J. Swinney.)*
*Carolyn J. Swinney is with the Air and Space Warfare Centre, Royal Air Force Waddington, Lincolnshire LN5 9NB, U.K., and also with the Computer Science and Electronic Engineering Department, University of Essex, Colchester CO4 3SQ, U.K. (e-mail: cjswin@essex.ac.uk).*
*John C. Woods is with the Computer Science and Electronic Engineering Department, University of Essex, Colchester CO4 3SQ, U.K.*
*Digital Object Identifier 10.1109/TITS.2022.3202011*

1558-0016 © 2022 Crown Copyright

Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:02:16 UTC from IEEE Xplore. Restrictions apply.

---
24958
IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS, VOL. 23, NO. 12, DECEMBER 2022

classification accuracy. Transfer learning is where a CNN is trained for one purpose but used for a different one. In this is successfully applied to the detection of UASs. Thermal imagery has also been researched but has not been shown to be reliable with quadcopters due to their heat efficiency when operating unless used in tandem with other sensor information. Light Detection And Ranging (LiDAR) and Laser Detection And Ranging (LADAR) have also been considered for detection and classification, but can be expensive. Acoustic signals can be used to detect and classify UAVs but can be prone to misclassification with noise from environmental conditions such as weather for example.

RF signals have also been considered due to their detection range and ability to detect and classify when environmental conditions change. RF signals can also be used to triangulate the source of the signal at a low cost using software defined radios (SDR). Nguyen et al. use an SDR to show that the signal between a UAS controller and the platform could be observed passively and actively through a reflection of the signal. Only two types of UAS were considered as part of this work. Huang et al. perform detection and using a low cost SDR and multiple UAS controllers, further showing that the controller could be localised using multiple SDRs. Zhao et al. consider a Wasserstein generative adversarial network to classify 4 types of UAS using an SDR. The experiments show 95% accuracy and the authors suggest open datasets for future work so comparison can be conducted. Ezuma et al. use a multi-staged approach comprised of machine learning classifier K-nearest neighbour and Bayes decision to produce 98% classification accuracy. Al-S'ad et al. provide an open source dataset called DroneRF for the classification of UAS flight modes including switched on, hovering and flying with and without a video feed. Their experiments using a deep neural network show accuracy drops as the complexity of the classification increases. 99.7% accuracy is achieved for 2 classes (UAV detection), 84% for 4 classes (UAV type classification) and 47% for 10 classes (UAV flight mode classification). Swinney and Woods take this work further by exploring the use of a CNN for feature extraction with transfer learning and machine learning classification to increase accuracy to 87% for 10 classes.

With commercially available small UASs being able to reach speeds of 45mph it is imperative that detection systems can inform countermeasure systems on a potential malicious UAS with enough time for decision making. For this reason the research in this paper considers unsupervised clustering as a first stage approach for an early warning system. We extend the previous work on supervised learning through CNN feature extraction using transfer learning by Swinney and Woods,, to look at the unsupervised clustering algorithm k-means. The research considers initialisations k-means++, random and principle component analysis (PCA) dimensionality reduction. The experiments use the ‘DroneDetect’ Dataset which was created by Swinney and Woods to expand the dataset to 7 UAS types and including signals in the presence of Bluetooth and Wi-Fi interference.
K-means was chosen as the clustering method due to its speed and we can use the value of k to help indicate new platforms when other known systems are in use on an airfield. This is due to the fact that UASs are being used increasingly for legitimate purposes. For example on airfields small UASs are being used for security, being deployed quickly to go and check a fence perimeter if an alarm is triggered. They are used for the detection of runway debris, building inspections and controlling wildlife. Early warning systems for UASs are normally made up of numerous subsystems and potentially different sensor types. So if for example, we knew that we had 4 different UASs operating on an airfield and a radar identified a potential UAS flying near the runway, an unsupervised learning algorithm could be set to k+1, so in this instance 5, to provide a quick indication and confirmation of whether the radar was picking up something of concern, i.e. a new cluster. That could then trigger a supervised learning algorithm of higher accuracy which will however take longer to determine a result. The clustering will have provided the vital information, that it is highly likely there is a new platform operating in a restricted airspace. The value comes from how quickly the clustering can give a result as an indicator and this is the reason why k-means clustering has been chosen for these experiments.

The organisation of the paper is as follows; Section 2 considers the methodology and graphical signal representation as images, CNN feature extraction, k-means clustering and metrics for performance evaluation. Section 3 discusses results and conclusions are presented in section 4.

## II. METHODOLOGY

### A. Signal Representation

The experiments use the ‘DroneDetect’ Dataset which contains 7 UAS types and including signals in the presence of Bluetooth and Wi-Fi interference. UAS types included in the dataset and the datalinks which they operate on are as follows:

- DJI Mavic 2 Air S (OcuSync 3.0)
- Parrot Disco (Wi-Fi)
- DJI Inspire 2 Pro (Lightbridge 2.0)
- DJI Mavic Pro (OcuSync 1.0)
- DJI Mavic Pro 2 (OcuSync 2.0)
- DJI Mavic Mini (Wi-Fi)
- DJI Phantom 4 (Lightbridge 2.0)
- No UAS

Samples were collected using a SDR called the BladeRF x40 which covers a frequency range of 47MHz to 6GHz and connected to a Palm Tree Vivaldi Ultra-wideband Antenna,. The SDR was set for a sample rate of 60Mbits/s, 28MHz bandwidth and a centre frequency of 2.4375GHz. This equated to each dataset entry being comprised of 12,000,000 complex samples or the equivalent of 20ms in recording time. 400 dataset entries were collected per class and fig. 1 shows the experimental set up. Samples were collected as the UAV type was flown at an altitude of 20m and flying within a radius of 40m to the collection system. For the dataset which included the interference signals from Bluetooth and Wi-Fi these devices were placed 2m from the

---
Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:02:16 UTC from IEEE Xplore. Restrictions apply.

---
**SWINNEY AND WOODS: K-MEANS CLUSTERING APPROACH TO UAS CLASSIFICATION**
**24959**

*Fig. 1. Picture of collection setup.*

*Fig. 2. Diagram of collection setup.*

antenna at ground level. Bluetooth interference was achieved using a phone connected to a Bluetooth speaker playing music and Wi-Fi interference was created using a laptop connected to a phone hotspot playing a YouTube video. The controller was located with the pilot at a 4m range from the collection system, no ground station was present, only the controller and UAV. The BladeRF was set up to receive with 28MHz of bandwidth and a centre frequency of 2.4375GHz and would therefore capture any uplink or downlink between the controller and UAV within that frequency range. Fig. 2 shows a diagram of the set up. The pilot was given free range to fly within the radius and this was done to try and achieve and capture a range of samples of varying time and spatial features from within the 40m radius of the collection system, to mimic a UAV flying around a restricted airspace but with the pilot in a static location. Unmanned Aerial Systems (UASs) refers to the UAV and controller together as a system.

The dataset was collected in an outdoor rural environment without the presence of any external noise. The SDR was used as a spectrum analyser to check that there were no external influences in the EM spectrum frequency band before the collection took place. This was done so that we could introduce our own Bluetooth and wi-fi interference signals and then observe the effect on the clustering. The signal itself is represented in two forms for the experiments, firstly as a spectrogram which displays the signal in the time domain. Secondly, as a power spectral density (PSD) which considers the signal in the frequency domain.

Both plots were created using Python 3 Matplotlib using a 1024 FFT size and a Hanning window with a 120 overlap. Fig. 3 shows a PSD graphical signal representation when no UAS is present and also without any deliberate interference signals from bluetooth or Wi-Fi. Fig. 4 shows a spectrogram representation for no UAS present when the bluetooth and Wi-Fi intereference is present. The interference signals can be seen as bursts of yellow activity. Fig. 5 shows a spectrogram representation for the DJI Inspire flying with no additional interference. The signal can be seen in yellow bursts of activity. Fig. 6 shows the PSD signal representation for the DJI Inspire flying with no interference present. On the PSD we can see the spikes of activity present at four main frequencies in the lower half of the spectrum.

Fig. 7 shows spectrogram representation for the DJI Inspire flying with interference from Bluetooth and Wi-Fi signals. The yellow bursts of activity can still be seen in the lower half of the graph but accompanied by thinner lines which stretch a larger section of the spectrum.

If fig. 6 is compared to fig. 8 it can be seen that the spikes of activity are harder to pick out by the human eye.

*Fig. 3. No UAS clean PSD.*

*Fig. 4. No UAS interference spectrogram.*

Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:02:16 UTC from IEEE Xplore. Restrictions apply.

---
24960
IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS, VOL. 23, NO. 12, DECEMBER 2022

*Fig. 5. DJI inspire clean spectrogram.*

*Fig. 6. DJI inspire clean PSD.*

*Fig. 7. DJI inspire interference spectrogram.*

In fig. 8 there is increased activity in the spectrum. Fig. 9 and fig. 10 show the spectrogram and PSD respectively for the Phantom 4 in the presence of bluetooth and Wi-Fi interference. When fig. 9 spectrogram for the Phantom 4 in the presence of interference is compared with fig. 4 no UAS with interference, it is hard to see visually with the eye the difference between the interference signals and the Phantom 4. Images for PSD and spectrogram datasets were saved as 224 x 224 pixels with 400 images per class.

### B. CNN Feature Extraction

The first datasets were produced for spectrogram and PSD images so that the raw images could be provided to the k-means clustering as a baseline for experiment 1. In experiments 2 and 3 the datasets were fed to two CNNs for feature extraction. The experiments use two different CNNs to understand whether the depth of the CNN has had an effect on the results, the 16 layer VGG-16 (experiment 2) and the 50 layer ResNet-50 (experiment 3). Both CNNs were pre-trained on ImageNet, an object detection database containing 1000 classes and 14 million images. This is known as transfer learning, a process by which a CNN trained for one purpose is used for another purpose. Although ImageNet does not contain images of this nature, spectrogram images in particular have been successful with pre-trained CNNs on ImageNet in the medical field for identifying medical conditions,. Their applicability for sound classification has been realised in various recent research

*Fig. 8. DJI inspire interference PSD.*

*Fig. 9. DJI phantom 4 interference spectrogram.*

Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:02:16 UTC from IEEE Xplore. Restrictions apply.

---
**SWINNEY AND WOODS: K-MEANS CLUSTERING APPROACH TO UAS CLASSIFICATION**
**24961**

*Fig. 10. DJI phantom 4 interference PSD.*

*Fig. 11. VGG-16 feature extraction structure.*

whereby Palanisamy et al. show that a CNN with pre-trained weights on ImageNet can provide“ a strong baseline for audio classification, even with a significant difference between spectrograms and ImageNet samples, assumptions gained from transfer learning hold firmly.” Tsalera et al. describe it as a ‘promising paradigm’ in their research surrounding transfer learning and spectrogram images. Chen et al. use a pretrained CNN on ImageNet to extract features from short fourier transform images for accurately detecting cracks in the pavement. Using pre-trained CNNs with ImageNet and the process of transfer learning with graphical representations of the signal has also been used for classifying small UASs and for GPS jamming signal classification. The diagnosis of COVID-19 has also benefited from the use of transfer learning with CNN feature extraction following by machine learning classification using logistic regression, random forest and support vector machine. The use of transfer learning and CNN feature extraction has all been used in a supervised capacity, either using the features to fine tune the CNN or feeding the features to a machine learning classifier such as logistic regression or random forest. This research is the first application of transfer learning and CNN feature extraction as a pre-cursor to an unsupervised learning algorithm.

Fig. 11 shows the VGG-16 structure being used as a feature extractor. To extract features from the CNN, forward propagation is stopped before the last fully connected layer so that a feature set can be saved. For the VGG-16 the feature set is 25088 values and for the ResNet-50 100352 values.

### C. K-Means Clustering

Unsupervised machine learning algorithm K-means clustering does not need training data labels. Instead unsupervised learning works by letting the algorithm work out the underlying inherent patterns in the data. K-means was chosen as in comparison to other clustering models as it is fast and has been proven robust. K-means clustering uses a metric of similarity to group the data and then that group is represented using a centroid. When the model is presented with new information it will assign the data to its closest centroid and therefore group. Similarity is found in these experiments by using the Euclidean distance. Equation 1 below shows the formula for calculating the Euclidean distance dij.
$$
d_{ij} = \sqrt{\sum_{k=1}^{n} (x_{ik} - x_{jk})^2} \quad (1)
$$
In equation 1 n represents the number of vectors and $x_{ik}$ and $x_{jk}$ are the two data points being compared. One of the key factors which has been shown to determine how well K-means performs is to do with how the centroids are initialised. This paper will consider the most common initialisations which include k-means++, principle component analysis (PCA) dimensionality reduction and random. K-means++ uses the probability proportional to the squared distance after selecting the first centroid randomly. The effect of this is that the centroids are moved as far away as they can be from each other. Initialisation which is random works by choosing random points from the dataset and using an average distance between the centroid and the point. PCA is a method of reducing dimensionality but keeping the feature information which is deemed important,. Due to the dimensionality being reduced, PCA can produce very quick results compared with the other initialisations. PCA was tested by Ding and He who found that PCA reduced data contained the same information which produced the same centroids for K-means. We are using PCA to reduce the dimensionality by projecting the data into an n dimensional space (where n is equal to the number of known classes). Then we are using those components of the PCA as the initialisation method which is deterministic. The method has also been tested by Li and Li who compare PCA initialisation to k-means++ and random initialisation using the handwritten digits dataset. Li and Li showed that PCA gave the same cluster centroids, similar accuracy to k-means++ and random initialisations, but performed in a much quicker time.

### D. Performance Metrics

As stated previously the point of unsupervised learning is to allow the algorithm to find inherent patterns in the data by not giving any label information. However, if label information is available it can be used to check and understand how well the model has performed. For the DroneDetect dataset we have the label information so this can be used to check how well the model is performing against our ground truth labels by using clustering quality metrics. The first is called v-measure score and it is a harmonic of homogeneity score and completeness score. Homogeneity looks of whether the clusters only contain members of a single class and completeness looks at whether all points in a class are assigned to the same cluster. A 1 represents perfect scoring between the label and the prediction, and equation 2 shows how the v-measure

---
Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:02:16 UTC from IEEE Xplore. Restrictions apply.

---
24962
IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS, VOL. 23, NO. 12, DECEMBER 2022

**TABLE I**
KEY
| Experiment | Graph Type | Method |
| :--- | :--- | :--- |
| Exp.1 | PSD or Spectrogram (SPEC) | Raw Image |
| Exp.2 | PSD or Spectrogram (SPEC) | FE using VGG-16 |
| Exp.3 | PSD or Spectrogram (SPEC) | FE using ResNet-50 |

**TABLE II**
K-MEANS CLUSTERING RESULTS CLEAN
| Init | Time (s) | Homo | Comp | v-meas | ARI | AMI |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Exp.1 PSD k++ | 1.014 | 0.392 | 0.585 | 0.469 | 0.197 | 0.467 |
| Exp.1 PSD rdm | 0.703 | 0.586 | 0.684 | 0.631 | 0.413 | 0.630 |
| Exp.1 PSD PCA | 0.280 | 0.580 | 0.668 | 0.620 | 0.428 | 0.619 |
| Exp.1 SPEC k++ | 1.207 | 0.271 | 0.314 | 0.291 | 0.148 | 0.288 |
| Exp.1 SPEC rdm | 0.895 | 0.260 | 0.279 | 0.269 | 0.144 | 0.266 |
| Exp.1 SPEC PCA | 0.401 | 0.295 | 0.319 | 0.307 | 0.159 | 0.304 |
| Exp.2 PSD k++ | 13.914 | 0.629 | 0.731 | 0.676 | 0.488 | 0.675 |
| Exp.2 PSD rdm | 9.999 | 0.631 | 0.700 | 0.664 | 0.490 | 0.663 |
| Exp.2 PSD PCA | 4.927 | 0.651 | 0.701 | 0.675 | 0.525 | 0.674 |
| Exp.2 SPEC k++ | 17.061 | 0.694 | 0.865 | 0.770 | 0.579 | 0.770 |
| Exp.2 SPEC rdm | 13.947 | 0.684 | 0.773 | 0.725 | 0.555 | 0.724 |
| Exp.2 SPEC PCA | 6.083 | 0.807 | 0.821 | 0.814 | 0.731 | 0.814 |
| Exp.3 PSD k++ | 68.375 | 0.698 | 0.724 | 0.711 | 0.581 | 0.709 |
| Exp.3 PSD rdm | 50.335 | 0.737 | 0.776 | 0.756 | 0.628 | 0.755 |
| Exp.3 PSD PCA | 24.509 | 0.711 | 0.731 | 0.721 | 0.617 | 0.720 |
| Exp.3 SPEC k++ | 67.166 | 0.717 | 0.815 | 0.762 | 0.583 | 0.762 |
| Exp.3 SPEC rdm | 56.466 | 0.678 | 0.765 | 0.719 | 0.533 | 0.718 |
| Exp.3 SPEC PCA | 24.564 | 0.720 | 0.749 | 0.734 | 0.586 | 0.733 |

score is calculated from the two.
$$
\text{v-measure} = \frac{(1+\beta)(\text{homogeneity})(\text{completeness})}{\beta(\text{homogeneity}+\text{completeness})} \quad (2)
$$
When $\beta$ is less than 1 more weight is assigned to homogeneity, when greater than 1 more weight is assigned to completeness. A perfect score is defined as v = 1. The Adjusted Rand Index (ARI) and the Adjusted Mutual Information (AMI) consider the difference between the label and the sample cluster with AMI normalising against chance. A 0 would indicate random labels and 1 indicating perfect matches.

## III. RESULTS

Table I shows a key for understanding the experiments presented in tables II and III.
Table II highlights the performance metrics for presenting the k-means clustering on PSD and spectrogram images and Feature Extraction (FE) using VGG-16 and ResNet-50 for clean signals. PCA dimensionality reduction speeds up the clustering time compared to random and k-means++ initialisations. If we consider time only then the fastest performance comes from using the images on their own with no feature extraction and PCA dimensionality reduction. The highest v-measure score is 0.814 which is using spectrogram images with the VGG-16 FE and PCA dimensionality reduction. ARI and AMI scores mimic the v-measure in terms of the highest performing. However, the highest v-measure score comes at a cost of time taking 6.083 seconds to complete. It is also interesting to note that the v-measure scores remain similar between experiment 1, 2 and 3 for PSD images, the CNN feature extraction does not increase v-measure significantly.

**TABLE III**
K-MEANS CLUSTERING RESULTS INTERFERENCE
| Init | Time (s) | Homo | Comp | v-meas | ARI | AMI |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Exp.1 PSD k++ | 1.009 | 0.363 | 0.521 | 0.428 | 0.275 | 0.426 |
| Exp.1 PSD rdm | 0.675 | 0.473 | 0.575 | 0.519 | 0.356 | 0.517 |
| Exp.1 PSD PCA | 0.193 | 0.584 | 0.617 | 0.600 | 0.475 | 0.599 |
| Exp.1 SPEC k++ | 0.945 | 0.274 | 0.290 | 0.282 | 0.144 | 0.279 |
| Exp.1 SPEC rdm | 0.798 | 0.276 | 0.292 | 0.284 | 0.145 | 0.281 |
| Exp.1 SPEC PCA | 0.321 | 0.276 | 0.292 | 0.284 | 0.145 | 0.281 |
| Exp.2 PSD k++ | 17.207 | 0.709 | 0.821 | 0.760 | 0.620 | 0.760 |
| Exp.2 PSD rdm | 11.010 | 0.674 | 0.786 | 0.726 | 0.595 | 0.725 |
| Exp.2 PSD PCA | 5.232 | 0.678 | 0.723 | 0.699 | 0.554 | 0.698 |
| Exp.2 SPEC k++ | 19.534 | 0.440 | 0.519 | 0.476 | 0.342 | 0.474 |
| Exp.2 SPEC rdm | 14.621 | 0.450 | 0.544 | 0.493 | 0.351 | 0.491 |
| Exp.2 SPEC PCA | 7.045 | 0.496 | 0.513 | 0.505 | 0.370 | 0.503 |
| Exp.3 PSD k++ | 72.722 | 0.684 | 0.702 | 0.693 | 0.566 | 0.692 |
| Exp.3 PSD rdm | 55.690 | 0.617 | 0.668 | 0.641 | 0.525 | 0.640 |
| Exp.3 PSD PCA | 24.284 | 0.633 | 0.653 | 0.643 | 0.513 | 0.642 |
| Exp.3 SPEC k++ | 79.026 | 0.448 | 0.511 | 0.477 | 0.318 | 0.475 |
| Exp.3 SPEC rdm | 78.479 | 0.441 | 0.490 | 0.464 | 0.319 | 0.462 |
| Exp.3 SPEC PCA | 31.298 | 0.463 | 0.491 | 0.477 | 0.335 | 0.475 |

*Fig. 12. PSD image PCA initialisation clean.*

While with the spectrogram images, the v-measure score doubles. For example raw spectrogram images using k++ initialisation produce 0.291 v-measure, while after the CNN feature extraction the v-measure increases to 0.770. It can also be seen that the deeper CNN architecture in experiment 3 does increase the v-measure scores for PSD but not for spectrogram.

Table III highlights the performance metrics for presenting the k-means clustering on PSD and spectrogram images and Feature Extraction (FE) using VGG-16 and ResNet-50 for signals in the presence of interference. Again, PCA dimensionality reduction speeds up the clustering time compared to random and k-means++ initialisations. As with the clean signals, if we consider time only then the fastest performance comes from using the images on their own with no feature extraction and PCA dimensionality reduction. The highest v-measure score is 0.760 which is using PSD images with the VGG-16 FE. This indicates that PSD is more robust to noise. ARI and AMI scores mimic the v-measure in terms of the highest performing. Again the highest v-measure score comes at a cost of time at 17 seconds.

To understand which implementation is best we really must consider the desried application of the system for it to be

---
Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:02:16 UTC from IEEE Xplore. Restrictions apply.

---
**SWINNEY AND WOODS: K-MEANS CLUSTERING APPROACH TO UAS CLASSIFICATION**
**24963**

*Fig. 13. Spectrogram image VGG-16 FE PCA initialisation clean.*

*Fig. 14. Spectrogram image ResNet-50 FE PCA INITIALISATION WITH INTERFERENCE.*

used within. For example if the system needs to display as high an accuracy as possible but time is not urgent then the correct choice would be a spectrogram with VGG-16 FE and PCA initialisation which takes 6 seconds and produces a 0.814 v-measure score in a clean environment. In the presence of interference the PSD outperforms the spectrogram, still with the VGG-16 and k++ initialisation which takes 17 seconds and produces a 0.760 v-measure score. However, if the system needs to be highly time sensitive but can cope with a slightly lower accuracy then the best option would be image PSD with PCA which can complete in 0.280 seconds with a v-measure of 0.620 in a clean environment and 0.193 seconds with a v-measure of 0.60 in the presence of interference. In this instance the unsupervised learning could be used for a very timely indication that a UAS might be in the vicinity to cue another system, even a supervised algorithm with much higher accuracy. Fig. 12 shows the k-means clustering for PSD images using PCA dimensionality reduction in a clean environment with no interference from bluetooth or Wi-Fi signals. Fig. 13 shows spectrogram images which have been through FE with VGG-16 and PCA initialisation in a clean environment. Comparing fig. 12 to fig. 13 it can be seen that the centroids in fig. 13 have greater seperation corresponding to a higher v-measure score, ARI and AMI.

*Fig. 15. PSD image VGG-16 FE PCA initialisation with interference.*

Fig. 14 shows the k-means clustering for spectrogram images using PCA dimensionality reduction and ResNet-50 FE in the presence of interference whereby the centroids are marked by black crosses. We can see that the clusters boundaries are quite close together. This produces the slightly lower v-measure score of 0.495 and a longer resolution time of 30 seconds Fig. 15 shows the higher accuracy implementation of the PSD image VGG-16 FE with PCA initialisation producing a v-measure of 0.699 in the presence of interference but completing in a slower time of 5.2 seconds.

## IV. CONCLUSION

Overall this paper has shown that unsupervised learning algorithms such as K-means clustering provide a potential for identifying small UAS signals which have not been seen before. Clustering graphical representations of the signal and utilising CNN feature extraction with transfer learning produces the highest v-measure score but at a cost of time, 6 seconds in a clean environment with spectrograms and 17 seconds with PSD in the presence of interference. With small UASs being capable of traveling at speeds of 45 mph, timely detection is essential in many use cases. Utilising a PSD image with PCA dimensionality and accepting a reduction of 0.2 v-measure in a clean environment allows clustering time to complete in under 0.3 second even in environments with interference from Bluetooth and Wi-Fi in the same band. Ultimately it should be dependent on the mission of the system. If a timely result is prudent then PSD images implemented with PCA initialisation would provide effective early warning to instigate the cueing of a secondary sensor or supervised algorithm with higher classification accuracy. However, employing transfer learning with CNN FE gives a higher v-measure score of 0.814 when employed with PCA initialisation and producing the clustering in 6 seconds. Future work could consider a larger number of UAV types in the dataset. Other types of unsupervised clustering algorithms and dimensionality reduction techniques could also be compared to extend this work. Another piece of valuable future work should include the collection of datasets at set distances away from the detection system. This would allow for a thorough evaluation of the correlation between clustering results and distance of

---
Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:02:16 UTC from IEEE Xplore. Restrictions apply.

---
24964
IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS, VOL. 23, NO. 12, DECEMBER 2022

the UASs from the detection equipment. Datasets could also be collected in a city based environment to understand the effect that a highly congested EM spectrum would have on the clustering results. This work has shown the ability of unsupervised clustering for the timely early warning of malicious activity in restricted airspace which remains paramount in an era of ever-increasing dependencies on small UASs.

## ACKNOWLEDGMENT

This work was carried out through the support of the School of Computer Science and Electronic Engineering, University of Essex, U.K., and the Royal Air Force, U.K.

## REFERENCES

 J. Partidge, “Royal mail to deliver to scilly isles by drone in first U.K. trial of its kind,” The Guardian, May 2021. [Online]. Available: https://www.theguardian.com/business/2021/may/10/royal-mail-to-deliver-to-scilly-isles-by-drone-in-first-uk-trial-of-its-kind
 P. M. Tagabe, “Economy-wide impact of drones,” Infrastructure Magazine, Feb. 2021. [Online]. Available: https://infrastructuremagazine.com.au/2021/02/10/economy-wide-impact-of-drones/
 PwC. (May 2018). *Skies Without Limites*. [Online]. Available: https://www.114pwc.co.uk/intelligent-digital/drones/Drones-impact-on-the-UK-economy-FINAL.pdf
 Civil Aviation Authority. (2018). *Drone Safety Risk: An assessment CAP 1627*. [Online]. Available: https://www.caa.co.uk
 *Airport Airspace Activity Study 2018*, Dedrone, San Francisco, CA, USA, 2018.
 *Analysis of Airprox in UK Airspace*, Military Aviation Authority, Civil Aviation Authority, U.K. AirProx Board, London, U.K., 2019.
 S. Shackle, “The mystery of the Gatwick drone,” *The Gaurdian*, Dec. 2020. [Online]. Available: https://www.theguardian.com/uk-news/2020/dec/01/the-mystery-of-the-gatwick-drone
 N. Herald, “Drone spotted 30 metres from plane at Auckland airport,” *The New Zealand Herald*, Apr. 2021. [Online]. Available: https://www.nzherald.co.nz/nz/drone-spotted-30-metres-from-plane-at-auckland-airport/JLFJA4D6OLRHIAHHRZO3W3LVXY/
 US News. (Mar. 2021). *Flights Halted at North Carolina Airport After Drone Sighted North Carolina News*. [Online]. Available: https://www.usnews.com/news/best-states/north-carolina/articles/2021-03-10/flights-halted-at-north-carolina-airport-after-drone-sighted
 K. McKenzie, “U.S. Army general: Small drones biggest threat since IEDs,” *Defense Post*, Arlington, VA, USA, Feb. 2021. Accessed: May 17, 2021. [Online]. Available: https://www.thedefensepost.com/2021/02/10/small-drones-threat-us-general/
 K. Parra. (Dec. 2020). *Travis AFB Launches Small Unarmed Aircraft Initiative, First on Air Force installation*. [Online]. Available: https://www.mildenhall.af.mil/News/Article-Display/Article/2449840/travis-afb-launches-small-unarmed-aircraft-initiative-first-on-air-force-instal/
 D. Daly. (2022). *5 Major Ways Airports are Using Drones*. [Online]. Available: https://consortiq.com/uas-resources/5-major-ways-airports-are-using-drones
 J. J. De Wit, R. I. Harmanny, and G. Prémel-Cabic, “Micro-Doppler analysis of small UAVs,” in *Proc. Eur. Microw. Week, Space Microw. (EuMW), 9th Eur. Radar Conf. (EuRAD)*, 2012, pp. 210–213.
 F. Fioranelli, M. Ritchie, H. Griffiths, and H. Borrion, “Classification of loaded/unloaded micro-drones using multistatic radar,” *Electron. Lett.*, vol. 51, no. 22, pp. 1813–1815, Oct. 2015. [Online]. Available: https://onlinelibrary.wiley.com/doi/10.1049/el.2015.3038
 R. I. A. Harmanny, J. J. M. de Wit, and G. P. Cabic, “Radar micro-Doppler feature extraction using the spectrogram and the cepstrogram,” in *Proc. 11th Eur. Radar Conf. (EuRAD)*, Oct. 2014, pp. 165–168.
 B. K. Kim, H.-S. Kang, and S.-O. Park, “Drone classification using convolutional neural networks with merged Doppler images,” *IEEE Geosci. Remote Sens. Lett.*, vol. 14, no. 1, pp. 38–42, Jan. 2017.
 B. Choi and D. Oh, “Classification of drone type using deep convolutional neural networks based on micro-Doppler simulation,” in *Proc. Int. Symp. Antennas Propag. (ISAP)*, no. 1, 2019, pp. 17–18.
 C. Aker and S. Kalkan, “Using deep networks for drone detection,” in *Proc. 14th IEEE Int. Conf. Adv. Video Signal Based Surveill. (AVSS)*. Piscataway, NJ, USA: Institute of Electrical and Electronics Engineers, Aug. 2017, pp. 1–6.
 B. Taha and A. Shoufan, “Machine learning-based drone detection and classification: State-of-the-art in research,” *IEEE Access*, vol. 7, pp. 138669–138682, 2019.
 A. Schumann, L. Sommer, J. Klatte, T. Schuchert, and J. Beyerer, “Deep cross-domain flying object classification for robust UAV detection,” in *Proc. 14th IEEE Int. Conf. Adv. Video Signal Based Surveill. (AVSS)*, Aug. 2017, pp. 1–6.
 M. Saqib, S. Daud Khan, N. Sharma, and M. Blumenstein, “A study on detecting drones using deep convolutional neural networks,” in *Proc. 14th IEEE Int. Conf. Adv. Video Signal Based Surveill. (AVSS)*. Piscataway, NJ, USA: Institute of Electrical and Electronics Engineers, Aug. 2017, pp. 1–5.
 P. Andraši, T. Radišić, M. Muštra, and J. Ivošević, “Night-time detection of UAVs using thermal infrared camera,” *Transp. Res. Proc.*, vol. 28, pp. 183–190, Jan. 2017, doi: 10.1016/j.trpro.2017.12.184.
 F. Svanstrom, C. Englund, and F. Alonso-Fernandez, “Real-time drone detection and tracking with visible, thermal and acoustic sensors,” in *Proc. 25th Int. Conf. Pattern Recognit. (ICPR)*, 2021, pp. 7265–7272.
 B. Kim, D. Khan, C. Bohak, W. Choi, H. Lee, and M. Kim, “V-RBNN based small drone detection in augmented datasets for 3D LADAR system,” *Sensors*, vol. 18, no. 11, p. 3825, Nov. 2018. [Online]. Available: https://www.mdpi.com/journal/sensors
 M. Salhi and N. Boudriga, “Multi-array spherical LiDAR system for drone detection,” in *Proc. 22nd Int. Conf. Transparent Opt. Netw. (ICTON)*. Washington, DC, USA: IEEE Computer Society, Jul. 2020, pp. 1–5.
 B. H. Kim, D. Khan, W. Choi, and M. Y. Kim, “Real-time counter-UAV system for long distance small drones using double pan-tilt scan laser radar,” *Proc. SPIE*, vol. 11005, p. 8, May 2019. [Online]. Available: https://www.spiedigitallibrary.org/conference-proceedings-of-spie/11005/110050C/Real-time-counter-UAV-system-for-long-distance-small-drones/10.1117/12.2520110.full
 A. Bernardini, F. Mangiatordi, E. Pallotti, and L. Capodiferro, “Drone detection by acoustic signature identification,” in *Proc. IS T Int. Symp. Electron. Imag. Sci. Technol.*, Jan. 2017, pp. 60–64.
 E. Matson, B. Yang, A. Smith, E. Dietz, and J. Gallagher, “UAV detection system with multiple acoustic nodes using machine learning models,” in *Proc. 3rd IEEE Int. Conf. Robot. Comput. (IRC)*. Piscataway, NJ, USA: Institute of Electrical and Electronics Engineers, Mar. 2019, pp. 493–498.
 X. Yue, Y. Liu, J. Wang, H. Song, and H. Cao, “Software defined radio and wireless acoustic networking for amateur drone surveillance,” *IEEE Commun. Mag.*, vol. 56, no. 4, pp. 90–97, Apr. 2018.
 Z. Shi, X. Chang, C. Yang, Z. Wu, and J. Wu, “An acoustic-based surveillance system for amateur drones detection and localization,” *IEEE Trans. Veh. Technol.*, vol. 69, no. 3, pp. 2731–2739, Mar. 2020.
 S. Jeon, J.-W. Shin, Y.-J. Lee, W.-H. Kim, Y. Kwon, and H.-Y. Yang, “Empirical study of drone sound detection in real-life environment with deep neural networks,” in *Proc. 25th Eur. Signal Process. Conf. (EUSIPCO)*. Piscataway, NJ, USA: Institute of Electrical and Electronics Engineers, Aug. 2017, pp. 1858–1862.
 P. Nguyen, M. Ravindranatha, A. Nguyen, R. Han, and T. Vu, “Investigating cost-effective RF-based detection of drones,” in *Proc. 2nd Workshop Micro Aerial Vehicle Netw., Syst., Appl. Civilian Use*, Jun. 2016, pp. 17–22.
 X. Huang, K. Yan, H.-C. Wu, and Y. Wu, “Unmanned aerial vehicle hub detection using software-defined radio,” in *Proc. IEEE Int. Symp. Broadband Multimedia Syst. Broadcast. (BMSB)*, Jun. 2019, pp. 1–6.
 C. Zhao, C. Chen, Z. Cai, M. Shi, X. Du, and M. Guizani, “Classification of small UAVs based on auxiliary classifier Wasserstein GANs,” in *Proc. IEEE Global Commun. Conf. (GLOBECOM)*, Dec. 2018, pp. 206–212.
 M. Ezuma, F. Erden, C. K. Anjinappa, O. Ozdemir, and I. Guvenc, “Detection and classification of UAVs using RF fingerprints in the presence of Wi-Fi and Bluetooth interference,” *IEEE Open J. Commun. Soc.*, vol. 1, pp. 60–76, 2019.
 M. S. Allahham, M. F. Al-Sa’d, A. Al-Ali, A. Mohamed, T. Khattab, and A. Erbad, “DroneRF dataset: A dataset of drones for RF-based detection, classification and identification,” *Data Brief*, vol. 26, Oct. 2019, Art. no. 104313.
 M. F. Al-Sa’d, A. Al-Ali, A. Mohamed, T. Khattab, and A. Erbad, “RF-based drone detection and identification using deep learning approaches: An initiative towards a large open source drone database,” *Future Gener. Comput. Syst.*, vol. 100, pp. 86–97, Nov. 2019, doi: 10.1016/j.future.2019.05.007.

---
Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:02:16 UTC from IEEE Xplore. Restrictions apply.

---
**SWINNEY AND WOODS: K-MEANS CLUSTERING APPROACH TO UAS CLASSIFICATION**
**24965**

 C. J. Swinney and J. C. Woods, “Unmanned aerial vehicle flight mode classification using convolutional neural network and transfer learning,” in *Proc. 16th Int. Comput. Eng. Conf. (ICENCO)*, Dec. 2020, pp. 83–87.
 A. Solodov, A. Williams, S. Al Hanaei, and B. Goddard, “Analyzing the threat of unmanned aerial vehicles (UAV) to nuclear facilities,” *Secur. J.*, vol. 31, no. 1, pp. 305–324, Feb. 2018.
 C. J. Swinney and J. C. Woods, “Unmanned aerial vehicle operating mode classification using deep residual learning feature extraction,” *Aerospace*, vol. 8, no. 3, p. 79, Mar. 2021. [Online]. Available: https://www.mdpi.com/2226-4310/8/3/79
 C. J. Swinney and J. C. Woods, “The effect of real-world interference on CNN feature extraction and machine learning classification of unmanned aerial systems,” *Aerospace*, vol. 8, no. 7, p. 179, 2021.
 I. T. Nassar and T. M. Weller, “A novel method for improving antipodal Vivaldi antenna performance,” *IEEE Trans. Antenna Propag.*, vol. 63, no. 7, pp. 3321–3324, Jul. 2015.
 A. M. D. Oliveira, M. B. Perotoni, S. T. Kofuji, and J. F. Justo, “A palm tree antipodal vivaldi antenna with exponential slot edge for improved radiation pattern,” *IEEE Antennas Wireless Propag. Lett.*, vol. 14, pp. 1334–1337, 2015.
 K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” in *Proc. ICLR*, 2015, pp. 1–14. [Online]. Available: http://www.robots.ox.ac.uk/
 K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Jun. 2016, pp. 770–778.
 N. B. Thota and D. U. Reddy, “Improving the accuracy of diabetic retinopathy severity classification with transfer learning,” in *Proc. IEEE 63rd Int. Midwest Symp. Circuits Syst. (MWSCAS)*, Aug. 2020, pp. 1003–1006.
 C. Jayakumari, V. Lavanya, and E. P. Sumesh, “Automated diabetic retinopathy detection and classification using ImageNet convolution neural network using fundus images,” in *Proc. Int. Conf. Smart Electron. Commun. (ICOSEC)*, Sep. 2020, pp. 577–582. [Online]. Available: https://ieeexplore.ieee.org/document/9215270/
 K. Palanisamy, D. Singhania, and A. Yao, “Rethinking CNN models for audio classification,” 2020, arXiv:2007.11154.
 E. Tsalera, A. Papadakis, and M. Samarakou, “Comparison of pretrained CNNs for audio classification using transfer learning,” *J. Sensor Actuator Netw.*, vol. 10, no. 4, p. 72, Dec. 2021.
 C. Chen, H. Seo, and Y. Zhao, “A novel pavement transverse cracks detection model using WT-CNN and STFT-CNN for smart-phone data analysis,” *Int. J. Pavement Eng.*, pp. 1–13, Jun. 2021, doi: 10.1080/10298436.2021.1945056.
 C. J. Swinney and J. C. Woods, “GNSS jamming classification via CNN, transfer learning & the novel concatenation of signal representations,” in *Proc. Int. Conf. Cyber Situational Awareness, Data Anal. Assessment (CyberSA)*, Jun. 2019, pp. 1–9. [Online]. Available: https://ieeexplore.ieee.org/document/9478250/
 C. Brown, A. Grammenos, T. Xia, P. Cicuta, and C. Mascolo, “Exploring automatic diagnosis of COVID-19 from crowdsourced respiratory sound data,” 2020, arXiv:2006.05919.
 N. Sharma et al., “Coswara—A database of breathing, cough, and voice sounds for COVID-19 diagnosis,” *Proc. Annu. Conf. Int. Speech Commun. Assoc. (INTERSPEECH)*, Oct. 2020, pp. 4811–4815.
 J. Han et al., “An early study on intelligent analysis of speech under COVID-19: Severity, sleep quality, fatigue, and anxiety, in *Proc. Annu. Conf. Int. Speech Commun. Assoc. (Interspeech)*, Oct. 2020, pp. 4946–4950.
 D. Pfitzner, R. Leibbrandt, and D. Powers, “Characterization and evaluation of similarity measures for pairs of clusterings,” *Knowl. Inf. Syst.*, vol. 19, no. 3, pp. 361–394, Jul. 2008. [Online]. Available: https://link.springer.com/article/10.1007/s10115-008-0150-6
 R. Suwanda, Z. Syahputra, and E. M. Zamzami, “Analysis of Euclidean distance and Manhattan distance in the K-means algorithm for variations number of centroid k,” *J. Phys., Conf. Ser.*, vol. 1566, no. 1, Jun. 2020, Art. no. 012058.
 P. Fränti and S. Sieranoja, “How much can kmeans be improved by using better initialization and repeats?” *Pattern Recognit.*, vol. 93, pp. 95–112, 2019, doi: 10.1016/j.patcog.2019.04.014.
 D. Arthur and S. Vassilvitskii, “k-means++: The advantages of careful seeding,” in *Proc. 11th Annu. ACM-SIAM Symp. Discrete Algorithms*, 2006, pp. 1027–1035.
 G. Hamerly and C. Elkan, “Alternatives to the k-means algorithm that find better clusterings,” in *Proc. 11th Int. Conf. Inf. Knowl. Manag.*, 2002, pp. 600–607.
 I. T. Jolliffe and J. Cadima, “Principal component analysis: A review and recent developments,” *Phil. Trans. Roy. Soc. A, Math., Phys. Eng. Sci.*, vol. 374, no. 2065, Apr. 2016, Art. no. 20150202. [Online]. Available: https://royalsocietypublishing.org/doi/abs/10.1098/rsta.2015.0202
 C. Ding, “K-means clustering via principal component analysis,” in *Proc. 21st Int. Conf. Mach. Learn.*, 2004, p. 29.
 C. Ding and X. He, “K-means clustering via principal component analysis,” in *Proc. 21st Int. Conf. Mach. Learn. (ICML)*, 2004, pp. 225–232.
 B. Li and B. Li, “An experiment of K-means initialization strategies on handwritten digits dataset,” *Intell. Inf. Manage.*, vol. 10, no. 2, pp. 43–48, Feb. 2018. [Online]. Available: http://www.scirp.org/journal/PaperInformation.aspx?PaperID=82761http://www.scirp.org/Journal/Paperabs.aspx?paperid=82761
 D. Dua and Graff. (2017). *UCI Machine Learning Repository*. [Online]. Available: http://archive.ics.uci.edu/ml
 Scikit Learn. (2022). *2.3. Clustering—Scikit-Learn 1.0.2 Documentation*. [Online]. Available: https://scikit-learn.org/stable/modules/clustering.html#clustering-evaluation

**Carolyn J. Swinney** (Student Member, IEEE) received the B.Eng. and M.Sc. degrees (Hons.) in electronics engineering from the University of Essex, Colchester, U.K., in 2007 and 2013, respectively, where she is currently pursuing the Ph.D. degree in electronic systems engineering. She graduated as a Communications and Electronics Engineering Officer at the Royal Air Force in 2014. She currently works within the Air and Space Warfare Centre. Her main research interests are signal processing, unmanned aerial vehicles, neural networks, machine learning, and cyber security.

**John C. Woods** was born Colchester, U.K., in 1964. He received the B.Eng. (Hons.) and Ph.D. degrees from the University of Essex, Colchester, in 1996 and 1999, respectively. He has been a Lecturer with the Department of Computer Science and Electronic Systems Engineering, University of Essex, since 1999. Although, his field of expertise is image processing, he has a wide range of interests including telecommunications, autonomous vehicles, and robotics.

---
Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:02:16 UTC from IEEE Xplore. Restrictions apply.