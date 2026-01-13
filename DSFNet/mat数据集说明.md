DroneRFb-Spectra: Real-world spectrograms for drone recognition
file type: .npy
data type: np.float16
size: 512x512
bandwidth: 100MHz
center frequency: 915MHz, 2.44GHz, 5.80GHz for different drones

Index: classes
0: Background, including WiFi and Bluetooth
1: DJI Phantom 3
2: DJI Phantom 4 Rro
3: DJI MATRICE 200
4: DJI MATRICE 100
5: DJI Air 2S
6: DJI Mini 3 Pro
7: DJI Inspire 2
8: DJI Mavic Pro
9: DJI Mini 2
10: DJI Mavic 3
11: DJI MATRICE 300
12: DJI Phantom 4 Pro RTK
13: DJI MATRICE 30T
14: DJI AVATA
15: DJI DIY
16: DJI MATRICE 600 Pro
17: VBar
18: FrSky X20
19: Futuba T16IZ
20: Taranis Plus
21: RadioLink AT9S
22: Futaba T14SG
23: Skydroid

Note: any questions please contact email: nnyu@zju.edu.cn



T0001  DJI Phantom 3
T0010  DJI Phantom 4 Pro
T0011  DJI MATRICE 200
T0100  DJI MATRICE 100
T0101  DJI Air 2S
T0110  DJI Mini 3 Pro
T0111  DJI Inspire 2
T1000  DJI Mavic Pro
T1001  DJI Mini 2
T1010  DJI Mavic 3
T1011  DJI MATRICE 300
T1100  DJI Phantom 4 Pro RTK
T1101  DJI MATRICE 30T
T1110  DJI AVATA
T1111  DJI通信模块自组机
T10000  DJI MATRICE 600 Pro
T10001  VBar 飞控器
T10010  FrSky X20 飞控器
T10011  Futaba T6IZ 飞控器
T10100  Taranis Plus 飞控器
T10101  RadioLink AT9S 飞控器
T10110  Futaba T14SG 飞控器
T10111  云卓 T12 飞控器
T11000  云卓 T10 飞控器

D00  20～40 m
D01  40～80 m
D10  80～150 m

S0000～0111  设置初始通信在915 MHz或2.4 GHz
S1000～1111  若设备支持则切换至2.4 GHz或5.8 GHz
