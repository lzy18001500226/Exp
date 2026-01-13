
2023 International Conference Automatics and Informatics (ICAI) | 979-8-3503-1291-1/23/$31.00 ©2023 IEEE | DOI: 10.1109/ICA158806.2023.10339072 International Conference AUTOMATICS AND INFORMATICS 2023, October 05 - 07, 2023, Varna, Bulgaria (ICAI'23)

---

# Drone Detection Approach Based on Radio Frequency Detector

**Ivan Garvanov**
University of Library Studies and
Information Technologies
Sofia, Bulgaria
ORCID 0000-0003-3113-1751

**Vladimir Ivanov**
Institute of Information and
Communication Technologies
Bulgarian Academy of Sciences
Sofia, Bulgaria
ORCID 0000-0002-9767-0136

**Denislav Kanev**
University of Library Studies and
Information Technologies
Sofia, Bulgaria
ORCID 0000-0002-9231-2196

**Magdalena Garvanova**
University of Library Studies and
Information Technologies
Sofia, Bulgaria
ORCID 0000-0003-2387-1442

***Abstract*** — **The unauthorized widely used unmanned aerial vehicle UAV (drones) raises serious concerns about the security of infrastructure and people in the settlements. This expansion leads to the need for development of effective systems for their detection and classification. In this article, a drone detection approach based on the detection of the Radio Frequency (RF) emitted during the communication between the drone and its remote controller, is proposed.**

***Keywords***—**Drone Detection, FFT, CFAR, Radio Frequency**

## I. INTRODUCTION

The widespread use of unmanned aerial vehicles (UAV) for both military purposes and civilian use is a prerequisite for unwanted incidents. This requires the development of new systems for monitoring and surveillance of aircraft in the airspace. The existence of different kind of drones in size, made from variety of materials, moving at speeds in a wide range, rapidly changing their direction of movement, using different propulsion methods, and possessing different tactical characteristics makes them exceedingly difficult to detect, recognize and accompany. In practice, there are various technologies used to enable the detection, location and tracking of drones, including the use of cooperative and non-cooperative data processing from different sensors.

In recent years, demanding work has been done on UAV detection systems development based on the work of main physical principles of the used technologies. Detection techniques use the principles of active or passive radar detection, as detection of radio frequency communication with the drone, detection by acoustic processing, detection of images obtained from optical or thermal cameras, as well as detection by merging the data from these techniques.

To improve the efficiency of UAV detection systems, cooperative multisensory data processing techniques and combining the different detection techniques are proposed. Most drones use radio frequency communication for remote control and/or data transmission (usually at 2.4 GHz). If a radio receiver is used to intercept the radio frequency channels of communication, it is possible to detect the presence of UAV.

Drone detection algorithm based on radio frequency emission is not a new solution. In, an algorithm for the detection of radio frequency sensing of drone communication signals using characteristic feature distinction is proposed. Adaptive constant false alarm rate (CFAR) algorithms using cell averaging (CA) and order statistics (OS) are analyzed in. These algorithms are adaptive to noisy environments and show good detection probability characteristics as well as to detect multiple targets. A deep learning algorithm is studied in to recognize the type of drone.

In this article, an adaptive drone detection algorithm using the radio frequency detection technology of the communication channel between the drone and its controller, through software-defined radio (SDR) is proposed and tested. The algorithm uses a fast Fourier transform – FFT to obtain the spectrum of the received signal, then using the CFAR processor detects the frequencies of the signals available in space. If these frequencies are close to the frequencies of the radio signals used for communication by the drone, it is assumed that the system has detected a drone.

The article is organized in the following order. In section 2, the detection algorithm based on the Fourier transform and CFAR processor is considered. Section 3 describes the technical characteristics of the equipment used in the experiments and shows and analyzes the results of the conducted research. Conclusions and recommendations for future research have been made at the end.

## II. DETECTION ALGORITHM

The UAV detection algorithm is based on RF detection of communication signals. The detection of the signals is performed by a CFAR detector processing the signal spectrum in the frequency area. To obtain the signal spectrum, Fourier transform has been used, which is a powerful signal processing tool.

### A. Fourier transform

From Fourier theory it is known that every function (every signal) can be represented by a finite or infinite number of harmonic functions, each with a certain amplitude, frequency and phase, i.e. there is a one-valued relationship between the temporal and frequency form of the signal. This means that the spectrum of the signal in the frequency area of the signal

---

979-8-3503-1291-1/23/$31.00 ©2023 IEEE

Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:55 UTC from IEEE Xplore. Restrictions apply.

230

---

is an assembly of multiple harmonic signals (spectral constituents, spectral components) that combined appropriately in amplitude and phase “reproduce" the signal in the time area. If the signal is periodic with a repetition period T, the individual harmonic constituents are equivalent at a distance 1/T from each other. From this it can be concluded that if the signal is non-periodic, T tends to infinite and the distance between the constituents becomes infinitesimal and tends to zero. From an energetic point of view, the signal spectrum shows how much energy is contained in each frequency component of the signal. The signal spectrum can be obtained by Fourier transform. Spectral analysis, also called spectral density and indicates the presence of signals with certain frequencies.

The Fourier right conversion transforms the signal from time area into the frequency area of the signal by the expression:

$$
X(\omega) = \int_{-\infty}^{\infty} x(t)e^{-j\omega t}dt \quad (1)
$$

Signal recovery in the signal time area performed by means of an inverse Fourier transform, following the dependence.

$$
x(t) = \frac{1}{2\pi} \int_{-\infty}^{\infty} X(\omega)e^{j\omega t}d\omega \quad (2)
$$

For the Fourier transform to be applicable, the Dirichlet conditions must be met, and the signal must be integrable. This means that the integral of its module must be finite, or:

$$
\int_{-\infty}^{\infty} |x(t)|dt < \infty. \quad (3)
$$

The modulus of the spectral function calls the magnitude response, and its argument is phase spectrum.

The Fourier transform places in full compliance the set time signal with its spectral function. The Fourier transform preserves all the information of the signal, so that the representation of the signal in the frequency area (spectral function) contains the same amount of information as the original signal located in the time area.

### B. The spectrogram on the signal

A signal spectrogram is an assembly of the instantaneous spectra of a signal in function of time:

$$
F(\omega, t) = \int_{t-T}^{t} X(\tau)e^{-j\omega\tau}d\tau \quad (4)
$$

Obtaining the spectrogram is possible by dividing the time signal into segments and applying the Fourier transform to each time segment. The set of spectra forms the signal spectrogram. If the segments of the analyzed signal do not overlap, then the resolution by frequency is determined by dependencies $\Delta f = 1/T_1$, and a time resolution – from the value of $T_1$. If the processed segments overlap, then the time resolution is equal to $\Delta t = T_1/N_1$, where $N_1$ is the number of samples of the segment subjected to the Fourier transform (hence the number $N_1$ is called the size of the Fourier transform and is a multiple of the power two). If the degree of overlap is extremely high, then the number of calculations may become unacceptably high. Due to the long duration of the signal, generated by drone, as well as the different frequency characteristics of the signal in time, it is preferable to analyze the spectral diagram of the investigated signal.

### C. Single pulse finder with CFAR processor

The detection of a single pulse of a useful signal applying a detector maintaining a constant probability of false alarm occurs after comparing the cell under test $x_0$ with the threshold of the discoverer $H_D$.

$$
H_D = VQ \quad (5)
$$

where Q is a scalar factor maintaining a preset false alarm probability constant, and V is the estimate of the noise level, which obtained as the sum of the amplitudes of all N the item in the training window.

$$
V = \sum_{i=1}^{N} x_i \quad (6)
$$

The single pulse detection in the CFAR processor is carried out according to the decision criterion (verification of two hypotheses) for the presence of ($H_1$) or absence ($H_0$) of a useful signal.

$$
\begin{cases}
H_1, & \text{if } x_i \ge H_D \\
H_0, & \text{if } x_i < H_D
\end{cases} \quad (7)
$$

where $H_1$ is the hypothesis that the sample tested is the target sought, and $H_0$ is the hypothesis that in the sample tested it contains only noise (Figure 1).

*Fig. 1. CFAR Processor*

## III. RESULTS

### A. Methodology

During the experiments, a drone DJI PHANTOM 3 ADVANCED is used, which is radio-controlled. Communication between the drone and the remote controller is carried out with RF signals with a frequency of 2.4 GHz. The topology of the experiment is shown in Figure 2.

The drone’s detection system consists of a laptop HP EliteBook 840 G5 CPU/Intel-Corei7-8550U/16GB RAM/1TB SSD to record and process signals received from software-defined radio PlutoSDR, with independent reception and transmission channels that can be operated in full duplex. SDR can generate or acquire RF analogue signals from 325 MHz to 3800 MHz and data refresh rate to 61.44 mega samples per second (MSPS). PlutoSDR is self-contained and is fully powered by USB with the default firmware. SDR uses libiio drivers and supports various operating systems.

---

Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:55 UTC from IEEE Xplore. Restrictions apply.

231

---

*Fig. 2. Topology of the experiment*

The collected records of RF signals are processed using a laptop or other interface device equipped with software LabVIEW.

### B. Experimental results

The study of the proposed Drone detection system was held at Iskar Dam, Bulgaria, where our environment is clean from radio frequency pollution (Figure 3).

*Fig. 3. Experimental scenarios*

A series of signal recordings were made in a frequency range of 2.4 GHz, which is the drone control range. The recordings have a duration of 20 seconds and a sampling rate of 528 KHz. An example signal recording in the absence of a radio-controlled drone is shown in Figure 4.

The spectrum of this signal is represented in Figure 5. It can be seen from the figure that the signal spectrum in the range of 2.4 GHz is uniform and no signals in high amplitude are noticed in this frequency range. The signal spectrum was obtained using a specialized program Spectrum Analysis of Signals in the MATLAB environment, adapted to work with the software-defined radio ADALM-PLUTO of Analog Devices.

*Fig. 4. Recording a radio signal in the absence of a radio-controlled drone*

*Fig. 5. Spectrum of the radio signal in the absence of a drone in the surrounding environment*

In the presence of a radio-controlled drone near the signal recording system, signals with the appearance as Figure 6.

*Fig. 6. Recording a radio signal in the presence of a radio-controlled drone*

The spectrum of this signal obtained through the specialized program Spectrum Analysis of MATLAB, adapted to work with the software-defined radio ADALM-PLUTO is shown in Figure 7.

---

Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:55 UTC from IEEE Xplore. Restrictions apply.

232

---

*Fig. 7. Radio signal spectrum in the presence of a drone in the surrounding environment*

It can be seen from the figure that a new component is contained in the resulting spectrum with a frequency of 2.401GHz. This frequency component belongs to the signal for communication with the drone.

The automatic detection of this frequency component is possible to obtain using a CFAR detector. For this purpose, a program has been set up to obtain the signal spectrum based on the Fast Fourier Transform then using the CFAR detector peak values of frequencies are detected in the range around 2.4 GHz.

*Fig. 8. Spectrum of the communication radio signal of a drone and the threshold of the CFAR processor*

The signal spectrum in the presence of a drone obtained with FFT is shown in Figure 8, where the signal from the drone is a clearly expressed peak. Processing this spectrum with a CFAR processor maintaining a constant probability of a false alarm of 10-4, peak is detected at a frequency of 2.401 GHz coinciding with the frequency of the drone’s communication radio signal. Figure 8 shows the threshold of the CFAR processor, which is above the signal spectrum, and only the amplitude of the drone frequency exceeds it. The output of the CFAR processor is shown in Figure 9.

*Fig. 9. Output of the CFAR processor*

Since the communication radio signal is prolonged over the time, a spectral analysis can be made to observe the spectrum of the signal over the time. The spectrogram of the communication radio signal is shown in Figure 10.

*Fig. 10. Spectrogram of the drone’s communication radio signal*

From the figure the signal with a frequency of 2.401 GHz is visible continuously in time. With our next research, we will decode the signal to decipher the communication commands.

## IV. CONCLUSIONS

The proposed algorithm for drone detection based on RF detection of the drone's communication signal is effective for radio-controlled drones. The detection is conducted in the frequency area of the signal, which is very convenient for subsequent classification of drones. The signal spectrogram can be used to subsequently track and analyze the communication signals. In our next developments, an algorithm for the classification of drones according to the frequency used for communication will be studied.

---

Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:55 UTC from IEEE Xplore. Restrictions apply.

233

---

### ACKNOWLEDGMENT

This work was supported by the National Science Program “Security and Defense”, which has received funding from the Ministry of Education and Science of the Republic of Bulgaria under the grant agreement No D01-74 /19.05.2022.

### REFERENCES

 M. Hassanalian, and A. Abdelkefi. 2017. Classifications, applications, and design challenges of drones: A review. Progress in Aerospace Sciences, vol. 91, 2017, pp. 99-131, ISSN 0376-0421, https://doi.org/10.1016/j.paerosci.2017.04.003.

 J. Wang, Y. Liu, and H. Song. 2021. Counter-Unmanned Aircraft System(s) (C-UAS): State of the Art, Challenges, and Future Trends. IEEE Aerospace and Electronic Systems Magazine, vol. 36, no. 3, pp. 4-29, 1 March 2021, doi: 10.1109/MAES.2020.3015537.

 J. Besada, I. Campaña, D. Carramiñana, L. Bergesio, and G. de Miguel. 2022. Review and Simulation of Counter-UAS Sensors for Unmanned Traffic Management. Sensors, vol. 22 (1): 189. https://doi.org/10.3390/s22010189.

 I. Garvanov, M. Garvanova, and D. Borissova. 2023. A model of a multi-sensor system for detection and tracking of vehicles and drones. In: B. Shishkov (Ed.), Business Modeling and Software Design. BMSD 2023. Lecture Notes in Business Information Processing, vol. 483, pp. 299-307. Springer, Cham, doi: https://doi.org/10.1007/978-3-031-36757-1_21.

 İ. Güvenç, O. Ozdemir, Y. Yapici, H. Mehrpouyan, and D. Matolak. 2017. Detection, localization, and tracking of unauthorized UAS and Jammers. 2017 IEEE/AIAA 36th Digital Avionics Systems Conference (DASC), St. Petersburg, FL, USA, 2017, pp. 1-10, doi: 10.1109/DASC.2017.8102043.

 J. Paredes, F. Álvarez, M. Hansard, and K. Rajab. 2021. A Gaussian Process model for UAV localization using millimetre wave radar. Expert Systems with Applications, vol. 185, 2021, 115563, ISSN 0957-4174, https://doi.org/10.1016/j.eswa.2021.115563.

 L. Zuo, J. Wang, J. Wang, and G. Chen. 2021. UAV detection via long-time coherent integration for passive bistatic radar. Digital Signal Processing, vol. 112, 2021, 102997, ISSN 1051-2004, https://doi.org/10.1016/j.dsp.2021.102997.

 I. Garvanov, M. Garvanova, D. Borissova, B. Vasovic, and D. Kanev. 2021. Towards IoT-based transport development in smart cities: Safety and security aspects. In: B. Shishkov (Ed.), Business Modeling and Software Design. BMSD 2021. Lecture Notes in Business Information Processing, vol. 422, pp. 392-398. Springer, Cham, doi: https://doi.org/10.1007/978-3-030-79976-2_27.

 N. Ibrahim, T. Sheltami, I. Ahmad, A. Yasar, and M. Abdeen. 2021. RF-Based UAV Detection and Identification Using Hierarchical Learning Approach. Sensors, vol. 21 (6): 1947, https://doi.org/10.3390/s21061947.

 P. Flak. 2021. Drone Detection Sensor With Continuous 2.4 GHz ISM Band Coverage Based on Cost-Effective SDR Platform. IEEE Access, vol. 9, pp. 114574-114586, 2021, doi: 10.1109/ACCESS.2021.3104738.

 P. Stoica, S. Basak, C. Molder, and B. Scheers. 2020. Review of Counter-UAV Solutions Based on the Detection of Remote Control Communication. 13th International Conference on Communications (COMM), Bucharest, Romania, 2020, pp. 233-238, doi: 10.1109/COMM48946.2020.9142017.

 S. Hameed. 2022. Peaks Detector Algorithm after CFAR for Multiple Targets Detection. EAI Endorsed Transactions on AI and Robotics, 2022, http://dx.doi.org/10.4108/airo.vli.1124.

 J. Gérard, J. Tomasik, C. Morisseau, A. Rimmel, and G. Vieillard. 2021. Micro-Doppler Signal Representation for Drone Classification by Deep Learning. 28th European Signal Processing Conference (EUSIPCO), Amsterdam, Netherlands, 2021, pp. 1561-1565, doi: 10.23919/Eusipco47968.2020.9287525.

 U. Oberst. 2007. The Fast Fourier Transform. SIAM Journal on Control and Optimization, vol. 46 (2), https://doi.org/10.1137/060658242.

 W. Jakob, U. Wetzker, and V. Jain. 2022. Spectrogram Data Set for Deep-Learning-Based RF Frame Detection. Data 7, № 12: 168. https://doi.org/10.3390/data7120168.

 SDR ADALM-PLUTO (PlutoSDR). 2023. Available at: https://www.analog.com/en/design-center/evaluation-hardware-and-software/evaluation-boards-kits/adalm-pluto.html#eb-overview.

---

Authorized licensed use limited to: YANGZHOU UNIVERSITY. Downloaded on September 15,2025 at 16:01:55 UTC from IEEE Xplore. Restrictions apply.

234
