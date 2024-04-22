import h5py
import numpy as np
import matplotlib.pyplot as plt
from wavelet_transforms import transform_wavelet_time
from scipy.signal import butter, filtfilt
with h5py.File(r"D:\HuaweiMoveData\Users\HUAWEI\Desktop\TAI\LISA Data Challenge\LDC2_spritz_mbhb1_training_v1.h5", 'r') as f:
    data = f["obs/tdi"][()]
    time =np.array([x[0] for y in data for x in y])#读取时间戳
    signals = np.nan_to_num(np.array([x[2] for y in data for x in y]))#读取TDI Y 通道数据
Nf = 372
Nt = 1440
dt = time[1] - time[0]
mult = 16
fs = 1 / dt
signals_out = transform_wavelet_time(signals,Nf,Nt,dt,mult=mult)#小波变换
power=(np.abs(signals_out)**(2)).T#为了适应pcolormesh的维度，转置
time_axis = np.linspace(0,time[-1]-time[0],Nt+1)#设置时间轴
freq_axis = np.linspace(10**(-4), 10**(-1), Nf + 1)#设置频率轴
log_freq_axis = np.log10(freq_axis)
plt.figure(figsize=(10, 5))
c = plt.pcolormesh(time_axis, log_freq_axis,np.log10(power), shading='auto')
plt.colorbar(c)
plt.xlabel(' (time)')
plt.ylabel(' (log10 fre Hz)')
plt.title('Spectrogram')
plt.show()