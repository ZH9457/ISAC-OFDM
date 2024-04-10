import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import commpy as cpy
import math

L = 200
N = 14 # 一个Slot中OFDM符号数
step_range = 1
K = 512 # OFDM子载波数量
CP = K//4  #25%的循环前缀长度
# CP = 0  #25%的循环前缀长度
P = K//4  # 导频数
I = (K+CP)*N
fn = 20 # KHz
Ts = 1/(K*fn*1000)
T = (K+CP)*Ts
pilotspace = K//P # 导频间隔
# pilotValue = 3+3j  # 导频格式
Modulation_type = 'QAM16' #调制方式，可选BPSK、QPSK、8PSK、QAM16、QAM64
channel_type ='random' # 信道类型，可选awgn
SNRdb = 25  # 接收端的信噪比（dB）
allCarriers = np.arange(K)  # 子载波编号 ([0, 1, ... K-1])
# pilotCarrier = allCarriers[::K//P]  # 每间隔P个子载波一个导频
# 为了方便信道估计，将最后一个子载波也作为导频
# pilotCarriers = np.hstack([pilotCarrier, np.array([allCarriers[-1]])])
# P = P+1 # 导频的数量也需要加1
m_map = {"BPSK": 1, "QPSK": 2, "8PSK": 3, "QAM16": 4, "QAM64": 6}
mu = m_map[Modulation_type]
# dataCarriers = np.delete(allCarriers, pilotCarriers)
# payloadBits_per_OFDM = (K-P)*mu  # 每个 OFDM 符号的有效载荷位数
payloadBits_per_OFDM = K*mu  # 每个 OFDM 符号的有效载荷位数
v_bin = math.ceil(N/step_range)
r_bin = step_range*P
Fd = 1/v_bin*(1/(step_range*(K+CP)*Ts))
# USF
vir_Q = K+CP
# N_ = K+CP
N_ = 2*vir_Q
overlap_Q = 0
v_bin_U = int(np.floor((np.ceil(I/step_range)-vir_Q-overlap_Q)/(N_-overlap_Q)))

# 定义制调制方式
def Modulation(bits):
    if Modulation_type == "QPSK":
        PSK4 = cpy.PSKModem(4)
        symbol = PSK4.modulate(bits)
        return symbol
    elif Modulation_type == "QAM64":
        QAM64 = cpy.QAMModem(64)
        symbol = QAM64.modulate(bits)
        return symbol
    elif Modulation_type == "QAM16":
        QAM16 = cpy.QAMModem(16)
        symbol = QAM16.modulate(bits)
        return symbol
    elif Modulation_type == "8PSK":
        PSK8 = cpy.PSKModem(8)
        symbol = PSK8.modulate(bits)
        return symbol
    elif Modulation_type == "BPSK":
        BPSK = cpy.PSKModem(2)
        symbol = BPSK.modulate(bits)
        return symbol
# 定义解调方式
def DeModulation(symbol):
    if Modulation_type == "QPSK":
        PSK4 = cpy.PSKModem(4)
        bits = PSK4.demodulate(symbol, demod_type='hard')
        return bits
    elif Modulation_type == "QAM64":
        QAM64 = cpy.QAMModem(64)
        bits = QAM64.demodulate(symbol, demod_type='hard')
        return bits
    elif Modulation_type == "QAM16":
        QAM16 = cpy.QAMModem(16)
        bits = QAM16.demodulate(symbol, demod_type='hard')
        return bits
    elif Modulation_type == "8PSK":
        PSK8 = cpy.PSKModem(8)
        bits = PSK8.demodulate(symbol, demod_type='hard')
        return bits
    elif Modulation_type == "BPSK":
        BPSK = cpy.PSKModem(2)
        bits = BPSK.demodulate(symbol, demod_type='hard')
        return bits
# 定义信道
def add_awgn(x_s, snrDB):
    data_pwr = np.mean(abs(x_s**2))
    noise_pwr = data_pwr/(10**(snrDB/10))
    noise = 1/np.sqrt(2) * (np.random.randn(len(x_s)) + 1j *
                            np.random.randn(len(x_s))) * np.sqrt(noise_pwr)
    # print(x_s.shape, noise.shape)
    # print(np.array([noise]).T)
    return x_s + np.array([noise]).T, noise_pwr
def channel(in_signal, SNRdb, channel_type="awgn"):
    channelResponse = np.array([1, 0, 0.3+0.3j])  # 随意仿真信道冲击响应
    in_signal = np.hsplit(in_signal, N)
    in_signal = np.vstack(in_signal)
    dt = np.arange(I)*Ts
    # print(dt.T)
    # in_signal = np.multiply(in_signal, np.exp(1j * 2 * math.pi * Fd * dt))
    dt = np.transpose([dt])
    # print(dt)
    in_signal = in_signal*np.exp(1j*2*math.pi*Fd*dt)
    # print(in_signal[:(-1)*L])
    in_signal = np.vstack((np.zeros((L, 1)), in_signal[:(-1)*L]))
    if channel_type == "random":
        convolved = np.convolve(in_signal, channelResponse)
        out_signal, noise_pwr = add_awgn(convolved, SNRdb)
    elif channel_type == "awgn":
        out_signal, noise_pwr = add_awgn(in_signal, SNRdb)
    # out_signal = in_signal
    # print(out_signal.shape)
    out_signal = np.vsplit(out_signal, N)
    out_signal = np.hstack(out_signal)
    # out_signal = np.reshape(out_signal, (-1, N))
    # print(out_signal.shape)
    # return out_signal, noise_pwr
    return out_signal
# 插入导频和数据，生成OFDM符号
# def OFDM_symbol(QAM_payload):
#     symbol = np.zeros((K,N), dtype=complex)  # 子载波位置
#     step = 0
#     for i in range(N):
#         step = step % step_range
#         pilotCarriers = allCarriers[step * pilotspace::pilotspace]
#         dataCarriers = np.delete(allCarriers, pilotCarriers)
#         dataNum = len(dataCarriers)
#         symbol[pilotCarriers,i] = pilotValue  # 在导频位置插入导频
#         symbol[dataCarriers] = QAM_payload[i*dataNum : (i+1)*dataNum]  # 在数据位置插入数据
#         step += 1
#     # symbol = np.reshape(symbol,(K * N,1))
#     return symbol
# 快速傅里叶逆变换
def IDFT(OFDM_data):
    return np.fft.ifft(OFDM_data, axis=0)
# 添加循环前缀
def addCP(OFDM_time):
    cp = OFDM_time[-CP:]
    # print(cp)
    return np.vstack((cp, OFDM_time))
# 接收端，去除循环前缀
def removeCP(signal):
    return signal[CP:(CP+K)]

# 快速傅里叶变换
def DFT(OFDM_RX):
    return np.fft.fft(OFDM_RX, axis=0)
# 信道估计
def channelEstimate(OFDM_demod, pilot_resource):
    step = 0
    Hest = np.array()
    for i in range(N):
        step = step % step_range
        pilotCarriers = allCarriers[step * pilotspace::pilotspace]
        dataCarriers = np.delete(allCarriers, pilotCarriers)
        pilots = OFDM_demod[pilotCarriers]  # 取导频处的数据
        Hest_at_pilots = pilots / pilot_resource[:, i]  # LS信道估计s
        # 在导频载波之间进行插值以获得估计，然后利用插值估计得到数据下标处的信道响应
        Hest_abs = interpolate.interp1d(pilotCarriers, abs(
            Hest_at_pilots), kind='linear')(allCarriers)
        Hest_phase = interpolate.interp1d(pilotCarriers, np.angle(
            Hest_at_pilots), kind='linear')(allCarriers)
        Hest = np.hstack(Hest, (Hest_abs * np.exp(1j*Hest_phase)).T)
    return Hest
# 均衡
def equalize(OFDM_demod, Hest):
    return OFDM_demod / Hest
# 取导频
def get_pilot(in_symbol):
    pilot_value = np.zeros((r_bin, v_bin), dtype=complex)
    # print(pilot_value.shape)
    step = 0
    for i in range(math.ceil(N/step_range)):
        temp = np.zeros((step_range*P, 1), dtype=complex)
        for j in range(step_range):
            if i*step_range+j < 14:
                step = step % step_range
                pilotCarriers = allCarriers[step*(pilotspace//step_range)::pilotspace]
                # print(pilotCarriers)
                pilotValue = in_symbol[pilotCarriers, i*step_range+j]
                # pilotValue = in_symbol[pilotCarriers, i]
                index = np.arange(step_range*P)[step::step_range]
                # print(index)
                # print(temp[index].T, '\n', pilotValue)
                # temp[np.isin(np.arange(step_range*P), index)] = pilotValue.T
                pilotValue = np.mat(pilotValue)
                temp[index] = pilotValue.T
                # print(pilotValue)
                step += 1
        # print(pilot_value[:,i], '\n', temp)
        # print(temp.T)
        pilot_value[:,i] = temp.T
    # print(pilot_value.shape)
    return pilot_value
def get_USF_source(*args):
    if len(args) == 1:
        in_symbol = args[0]
        pilot_value = np.zeros((in_symbol.shape[0], v_bin), dtype=complex)
        for i in range(math.ceil(N/step_range)):
            temp = np.zeros((in_symbol.shape[0], 1), dtype=complex)
            for j in range(step_range):
                if i * step_range + j < 14:
                    pilotCarriers = allCarriers[j * (pilotspace//step_range)::pilotspace].copy()
                    # print(type(pilotCarriers))
                    dataCarriers = np.delete(allCarriers, pilotCarriers)
                    cur_col = in_symbol[:, i * step_range + j].copy()
                    cur_col[dataCarriers] = 0
                    for m in range(pilotspace//step_range-1):
                        pilotCarriers += 1
                        cur_col[pilotCarriers[pilotCarriers<K]] = cur_col[pilotCarriers[pilotCarriers<K]-1]
                    # print(temp.shape, in_symbol[:, i*step_range+j].shape)
                    temp += np.transpose([cur_col])
                    # print(in_symbol[:, i*step_range+j])
            pilot_value[:, i] = temp.T
        return pilot_value
    elif len(args) == 2:
        in_symbol = args[0]
        Fd_ = args[1]
        pilot_value = np.zeros((in_symbol.shape[0], v_bin), dtype=complex)
        for i in range(math.ceil(N/step_range)):
            temp = np.zeros((in_symbol.shape[0], 1), dtype=complex)
            for j in range(step_range):
                if i * step_range + j < 14:
                    pilotCarriers = allCarriers[j * (pilotspace//step_range)::pilotspace]
                    dataCarriers = np.delete(allCarriers, pilotCarriers)
                    cur_col = in_symbol[:, i * step_range + j].copy()
                    cur_col[dataCarriers] = 0
                    coe = np.exp(1j * 2 * np.pi * Fd_ * j * T)
                    cur_col[pilotCarriers] *= coe
                    temp += np.transpose([cur_col])
            pilot_value[:, i] = temp.T
        return pilot_value
def correct(in_symbol, Fd_):
    for i in range(math.ceil(N/step_range)):
        for j in range(step_range):
            if j != 0:
                index = np.arange(step_range * P)[j::step_range]
                coe = np.exp(-1j*2*np.pi*Fd_*j*T)
                in_symbol[index, i] *= coe
    return in_symbol
def remat(in_symbol, x, y):
    # print(in_symbol.shape)
    cols = np.floor((np.size(in_symbol)-y)/(x-y))
    # print(cols)
    out_symbol = np.zeros((x, int(cols)), dtype=complex)
    for i in range(int(cols)):
        start = (x-y)*i
        # print(out_symbol[:, i], in_symbol[start:start+x].shape)
        out_symbol[:, i] = in_symbol[start:start+x].T
    # print(out_symbol.shape)
    return out_symbol
def concate(in_symbol):
    pilot_value = np.zeros((in_symbol.shape[0], v_bin), dtype=complex)
    for i in range(math.ceil(N/step_range)):
        temp = np.zeros((in_symbol.shape[0], 1), dtype=complex)
        for j in range(step_range):
            if i * step_range + j < 14:
                temp += np.transpose([in_symbol[:, i * step_range + j]])
        pilot_value[:, i] = temp.T
    return pilot_value
def correct_USF(in_symbol, Fd_):
    coe = np.ones((K, np.shape(in_symbol)[1]), dtype=complex)
    for i in range(np.shape(in_symbol)[1]):
        for j in range(step_range):
            if j != 0:
                index = np.arange(step_range * P)[j::step_range]
                coe[index, i] *= np.exp(-1j*2*np.pi*Fd_*j*T)
    coe = np.fft.ifft(coe)
    coe = addCP(coe)
    for i in range(np.shape(in_symbol)[1]):
        # coe[:, i] = np.convolve(coe[:, i], in_symbol[:, i])
        # in_symbol[:, i] = np.fft.ifft(np.fft.fft(coe[:, i])*np.fft.fft(in_symbol[CP:, i]), n=np.shape(in_symbol)[0])
        coe[:, i] = np.fft.ifft(np.fft.fft(coe[:, i])*np.fft.fft(in_symbol[:, i]))
    return coe
def OFDM_simulation():
    # 产生比特流
    np.random.seed(20)
    bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM * N, ))
    # 比特信号调制
    QAM_s = Modulation(bits)
    # 生成OFDM符号
    # OFDM_data = OFDM_symbol(QAM_s)
    OFDM_data = np.reshape(QAM_s, (K,N))
    Tx_resource = OFDM_data.copy()
    # 获取原导频信号
    # pilot_resource = get_pilot(OFDM_data)  # COS
    pilot_resource = get_USF_source(OFDM_data)  # USF
    # print(pilot_resource-OFDM_data)
    # 快速逆傅里叶变换
    OFDM_time = IDFT(OFDM_data)
    pilot_resource = IDFT(pilot_resource)  # USF
    # 添加循环前缀
    OFDM_withCP = addCP(OFDM_time)
    pilot_resource = addCP(pilot_resource)  # USF
    # pilot_resource[:CP, :] = 0  # USF
    # pilot_resource[CP+P*step_range+1:, :] *= -1  # USF
    # pilot_resource[-CP:, :] = 0  # USF
    correct_pilot = pilot_resource.copy()  # USF
    correct_pilot[:(K+CP)//2, :] *= -1  # USF
    # correct_pilot[:CP, :] = 0  # USF
    # correct_pilot[-CP:, :] = 0  # USF
    # print(pilot_resource)
    # pilot_resource = OFDM_withCP

    # 经过信道
    OFDM_TX = OFDM_withCP
    OFDM_RX = channel(OFDM_TX, SNRdb, "awgn")

    # # 去除循环前缀
    # OFDM_RX_noCP = removeCP(OFDM_RX)
    # # print(OFDM_RX.shape)
    # # 快速傅里叶变换
    # OFDM_demod = DFT(OFDM_RX_noCP)
    # # 获取导频信号
    # pilot_after = get_pilot(OFDM_demod)
    #
    # # COS
    # Z = pilot_resource.conj()*pilot_after
    # # print(pilot_resource.shape)
    # # Z = np.fft.ifft(Z, axis=0)
    # # print(Z.shape)
    # Z = np.fft.fft(Z, axis=1)
    # _, col = divmod(np.argmax(20*np.log10(abs(Z))), v_bin)
    # # print(row, col)
    # Z = correct(Z, col/v_bin*(1/(step_range*(K+CP)*Ts)))
    # Z = np.fft.ifft(Z, axis=0)
    # row_, col_ = divmod(np.argmax(20 * np.log10(abs(Z))), v_bin)
    # print(row_, col_)
    # # print(Z.shape)
    # xx = np.arange(v_bin)
    # yy = np.arange(r_bin)
    # X, Y = np.meshgrid(xx, yy)
    # fig = plt.figure()
    # ax = plt.axes(projection = '3d')
    # # ax.scatter3D(X, Y, abs(Z))
    # ax.set_xlabel('Doppler')
    # ax.set_ylabel('Range')
    # ax.plot_surface(X, Y, 20*np.log10(abs(Z)))
    # plt.show()

    # USF

    pilot_resource = np.hsplit(pilot_resource, v_bin)
    pilot_resource = np.vstack(pilot_resource)
    correct_pilot = np.hsplit(correct_pilot, v_bin)
    correct_pilot = np.vstack(correct_pilot)
    OFDM_RX = concate(OFDM_RX)
    OFDM_RX_src = OFDM_RX.copy()
    OFDM_RX = np.hsplit(OFDM_RX, math.ceil(N/step_range))
    OFDM_RX = np.vstack(OFDM_RX)
    # print(remat([1,2,3,4,5,6,7,8,9], 4, 2))
    pilot_resource = remat(pilot_resource, N_, overlap_Q)
    correct_pilot = remat(correct_pilot, N_, overlap_Q)
    OFDM_RX = remat(OFDM_RX, N_+vir_Q, vir_Q+overlap_Q)
    OFDM_RX[:vir_Q, :] += OFDM_RX[N_:N_+vir_Q,:]
    # print(OFDM_RX)
    OFDM_RX = np.vsplit(OFDM_RX, indices_or_sections=[N_])[0]
    # print(np.vsplit(OFDM_RX, indices_or_sections=[0, N_]))
    OFDM_RX = DFT(OFDM_RX)
    pilot_resource = DFT(pilot_resource)
    correct_pilot = DFT(correct_pilot)
    # print(pilot_resource.shape)
    # print(v_bin_U)
    Z = OFDM_RX*pilot_resource[:, :v_bin_U].conj()
    Z = np.fft.ifft(Z, axis=0)
    Z = np.fft.fft(Z, axis=1)
    Z_ = OFDM_RX*correct_pilot[:, :v_bin_U].conj()
    Z_ = np.fft.ifft(Z_, axis=0)
    Z_ = np.fft.fft(Z_, axis=1)
    Z = abs(Z)-abs(Z_)
    row, col = divmod(np.argmax(20*np.log10(abs(Z))), v_bin_U)
    # row, col = divmod(np.argmax(20 * np.log10(Z)), v_bin_U)
    print(row, col, 20*np.log10(abs(Z))[row, col])
    xx = np.arange(v_bin_U)
    yy = np.arange(N_)
    X, Y = np.meshgrid(xx, yy)
    fig = plt.figure()
    ax = plt.axes(projection = '3d')
    # ax.scatter3D(X, Y, abs(Z))
    ax.set_xlabel('Doppler')
    ax.set_ylabel('Range')
    # ax.plot_surface(X, Y, 20*np.log10(abs(Z)))
    ax.plot_surface(X, Y, abs(Z))
    plt.show()

    # correct_resource = get_USF_source(Tx_resource, col/v_bin*(1/(step_range*(K+CP)*Ts)))
    # correct_resource = IDFT(correct_resource)
    # correct_resource = addCP(correct_resource)
    # correct_resource = np.hsplit(correct_resource, v_bin)
    # correct_resource = np.vstack(correct_resource)
    # correct_resource = remat(correct_resource, N_, overlap_Q)
    # correct_resource = DFT(correct_resource)
    # # print(correct_resource.shape)
    # ZZ = OFDM_RX * correct_resource[:, :v_bin_U].conj()
    # ZZ = np.fft.ifft(ZZ, axis=0)
    # ZZ = np.fft.fft(ZZ, axis=1)
    # row_, col_ = divmod(np.argmax(20*np.log10(abs(ZZ))), v_bin_U)
    # print(row_, col_)
    # # print(correct_resource-pilot_resource)
    # xx = np.arange(v_bin_U)
    # yy = np.arange(N_)
    # X, Y = np.meshgrid(xx, yy)
    # fig = plt.figure()
    # ax = plt.axes(projection = '3d')
    # ax.set_xlabel('Doppler')
    # ax.set_ylabel('Range')
    # # ax.plot_surface(X, Y, 20*np.log10(abs(ZZ)))
    # ax.plot_surface(X, Y, abs(ZZ))
    # plt.show()

    # # 信道估计
    # Hest = channelEstimate(OFDM_demod, pilot_resource)
    # # 均衡
    # equalized_Hest = equalize(OFDM_demod, Hest)
    # # 获取数据位置的数据
    # def get_payload(equalized):
    #     return equalized[dataCarriers]
    # QAM_est = get_payload(equalized_Hest)
    # # 反映射，解调
    # bits_est = DeModulation(QAM_est)
    # # print(bits_est)
    # print("误比特率BER： ", np.sum(abs(bits-bits_est))/len(bits))
if __name__ == '__main__':
    OFDM_simulation()
