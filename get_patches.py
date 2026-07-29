# import signal
import math
import scipy.io as sio
from scipy.io import savemat
from scipy.signal import convolve2d as conv2
from scipy import signal
from scipy.signal.windows import triang
from scipy.signal import convolve2d as conv2
import pywt
import numpy as np
import segyio
import torch
from matplotlib import pyplot as plt


def get_dx_from_segy(seg_file_path):
    """
    从 SEGY 文件中提取检波点（GroupX）的空间采样间隔 dx（单位：米）

    参数:
        seg_file_path (str): SEGY 文件路径

    返回:
        dx (float): 空间采样间隔（单位：米）
    """
    with segyio.open(seg_file_path, "r", ignore_geometry=True) as f:
        group_x = f.attributes(segyio.TraceField.GroupX)[:]  # 获取所有检波点X坐标
        print(group_x[:10])  # 打印前 10 个接收点位置
        dxs = np.diff(group_x)  # 相邻检波点之间的间距
        dx = np.median(dxs)  # 使用中位数避免异常值影响
        print(f"空间采样间隔 dx = {dx} 米")
    return dx


def fk_spectra(data, dt, dx, L=6):
    """
    f-k(频率-波数)频谱分析
    :param data: 二维的地震数据
    :param dt: 时间采样间隔
    :param dx: 道间距
    :param L: 平滑窗口
    :return: S(频谱结果), f(频率范围), k(波数范围)
    """
    print(data.shape)
    data = np.array(data)
    [nt, nx] = data.shape  # 获取数据维度
    # 计算nk和nf是为了加快傅里叶变换速度,等同于nextpow2
    i = 0
    while (2 ** i) <= nx:
        i = i + 1
    nk = 4 * 2 ** i
    j = 0
    while (2 ** j) <= nt:
        j = j + 1
    nf = 4 * 2 ** j
    S = np.fft.fftshift(abs(np.fft.fft2(data, (nf, nk))))  # 二维傅里叶变换
    H1 = np.hamming(L)
    # 设置汉明窗口大小，汉明窗的时域波形两端不能到零，而海宁窗时域信号两端是零。从频域响应来看，汉明窗能够减少很近的旁瓣泄露
    H = (H1.reshape(L, -1)) * (H1.reshape(1, L))
    S = signal.convolve2d(S, H, boundary='symm', mode='same')  # 汉明平滑
    S = S[nf // 2:nf, :]
    f = np.arange(0, nf / 2, 1)
    f = f / nf / dt
    k = np.arange(-nk / 2, nk / 2, 1)
    k = k / nk / dx
    return S, k, f


def plot_seismic_f_k_npy(seismic_data, save=False, save_path=None, show=False, fontsize=14):
    """
    绘制地震数据的 f-k 频谱图

    参数:
        seismic_data: 地震数据
        save: 是否保存图片
        save_path: 保存路径
        show: 是否显示图片
        fontsize: 字体大小，默认14
    """
    dx = 125
    dt = 0.008
    S, k, f = fk_spectra(seismic_data, dt, dx)
    S[S <= 0] = 1e-10  # 避免对数域中的零值或负值
    amplitude_db = 10 * np.log10(S)

    plt.figure(figsize=(5, 6))
    plt.pcolormesh(k, f, amplitude_db, shading='auto', cmap='viridis', vmin=0, vmax=62.5)
    # 添加色彩轴并设置字体样式，减少右侧空白
    cbar = plt.colorbar(pad=0.02)  # 减少colorbar与主图的间距
    # 设置色彩轴刻度标签字体：Times New Roman、加粗、竖着显示、与刻度居中
    for label in cbar.ax.get_yticklabels():
        label.set_fontfamily('Times New Roman')
        label.set_fontweight('bold')
        label.set_fontsize(fontsize)
        label.set_rotation(270)  # 数字竖着显示
        label.set_va('center')  # 与刻度居中
    plt.xlabel('k [c/m]', fontfamily='Times New Roman', fontweight='bold', fontsize=fontsize)
    plt.ylabel('f [Hz]', fontfamily='Times New Roman', fontweight='bold', fontsize=fontsize)
    # 倒转 y 轴，使得低频在底部，高频在顶部
    plt.gca().invert_yaxis()
    x_ticks = np.linspace(k.min(), k.max(), 3)  # 生成 5 个等间距的刻度
    plt.xticks(x_ticks)  # 设置 x 轴刻度
    # 设置坐标轴刻度标签字体
    for label in plt.gca().get_xticklabels():
        label.set_fontfamily('Times New Roman')
        label.set_fontweight('bold')
        label.set_fontsize(fontsize)
    for label in plt.gca().get_yticklabels():
        label.set_fontfamily('Times New Roman')
        label.set_fontweight('bold')
        label.set_fontsize(fontsize)
        label.set_rotation(90)  # 纵坐标数字竖着显示
        label.set_va('center')  # 纵坐标数字与刻度轴居中
    # plt.title('f-k Spectrum')
    # 使用tight_layout自动调整布局，去除多余空白，减少右侧边距
    plt.tight_layout(pad=0.5, rect=[0, 0, 0.98, 1])  # rect参数控制[left, bottom, right, top]
    # plt.show()
    # 你可以通过调整 k 和 f 的范围来放大图像
    # k_min, k_max = -0.01025, 0  # 设置你希望显示的波数范围
    # f_min, f_max = 0, 55  # 设置你希望显示的频率范围

    # 绘制频谱图
    # plt.figure(figsize=(6, 6))
    # plt.pcolormesh(k, f, amplitude_db, shading='auto', cmap='viridis', vmin=0, vmax=100)
    #
    # # 放大显示感兴趣的区域
    # plt.xlim(k_min, k_max)  # 限制 x 轴（波数 k）的范围
    # plt.ylim(f_min, f_max)  # 限制 y 轴（频率 f）的范围
    # plt.colorbar()
    # plt.xlabel('k [c/m]')
    # plt.ylabel('f [Hz]')
    # # 设置 x 轴刻度
    # x_ticks = np.linspace(k_min, k_max, 3)  # 生成 3 个等间距的刻度
    # plt.xticks(x_ticks)  # 设置 x 轴刻度
    # plt.subplots_adjust(left=0.18, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)

    if save and save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
        print(f"📷 Saved figure: {save_path}")
    elif show:
        plt.show()
    plt.close()


def plot_seismic_tensor(seis_tensor, extent_time, save=False, save_path=None, show=True, fontsize=14):
    """
    绘制地震数据张量图

    参数:
        seis_tensor: 地震数据张量
        extent_time: 时间范围
        save: 是否保存图片
        save_path: 保存路径
        show: 是否显示图片
        fontsize: 字体大小，默认14
    """
    plt.figure(figsize=(4.5, 6))
    plt.imshow(seis_tensor, cmap='gray', extent=extent_time, aspect='auto', vmin=-1, vmax=1)
    plt.xlabel('Trace', fontfamily='Times New Roman', fontweight='bold', fontsize=fontsize)
    plt.ylabel('Time (ms)', fontfamily='Times New Roman', fontweight='bold', fontsize=fontsize)
    plt.title('')
    # 设置坐标轴刻度标签字体
    for label in plt.gca().get_xticklabels():
        label.set_fontfamily('Times New Roman')
        label.set_fontweight('bold')
        label.set_fontsize(fontsize)
    for label in plt.gca().get_yticklabels():
        label.set_fontfamily('Times New Roman')
        label.set_fontweight('bold')
        label.set_fontsize(fontsize)
        label.set_rotation(90)  # 纵坐标数字竖着显示
        label.set_va('center')  # 纵坐标数字与刻度轴居中
    # 使用tight_layout自动调整布局，去除多余空白
    plt.tight_layout(pad=0.8)
    if save and save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
        print(f"📷 Saved figure: {save_path}")
    elif show:
        plt.show()
    plt.close()


def plot_seismic_npy(path_file, extent_time, save=False, save_path=None, show=False, fontsize=14):
    """
    从numpy数组绘制地震数据图

    参数:
        path_file: numpy数组数据
        extent_time: 时间范围
        save: 是否保存图片
        save_path: 保存路径
        show: 是否显示图片
        fontsize: 字体大小，默认14
    """
    dataset_t = path_file
    seis_tensor = torch.tensor(dataset_t)
    plot_seismic_tensor(seis_tensor, extent_time, save=save, save_path=save_path, show=show, fontsize=fontsize)


def calculate_snr(target_v, output_v):
    # 如果是Tensor，转为numpy
    if isinstance(output_v, torch.Tensor):
        output_v = output_v.detach().cpu().numpy()
    if isinstance(target_v, torch.Tensor):
        target_v = target_v.detach().cpu().numpy()

    # flatten后整体计算能量
    origSignal = target_v.flatten()
    errorSignal = (target_v - output_v).flatten()

    signal_power = np.sum(origSignal ** 2)
    noise_power = np.sum(errorSignal ** 2)

    # 避免除零错误
    if noise_power == 0:
        return float('inf')

    snr = 10 * math.log10(signal_power / noise_power)
    return snr


def calculate_rmse(origin, predicted):
    # 加载数据
    # print(origin.shape)
    # print(predicted.shape)

    # 确保数据形状一致
    if origin.shape != predicted.shape:
        raise ValueError("Origin and predicted signals must have the same shape.")

    # 计算均方根误差（RMSE）
    mse = np.mean((predicted - origin) ** 2)
    rmse = np.sqrt(mse)

    return rmse


def get_info_seg(seg_file_path):
    with segyio.open(seg_file_path, 'r', ignore_geometry=True) as f:
        # 读取所有地震道数据
        f.mmap()
        sourceX = f.attributes(segyio.TraceField.SourceX)[:]
        nTrace = f.tracecount
        nSample = f.bin[segyio.BinField.Samples]
        startT = 0
        deltaT = f.bin[segyio.BinField.Interval]
        print("     Number of Trace   = %d" % (nTrace))
        print("     Number of Samples = %d" % (nSample))
        print("     Start Samples     = %d" % (startT))
        print("     Sampling Rate     = %d" % (deltaT))
        data = np.asarray([np.copy(trace) for trace in f.trace])
    data = data.T
    time_length = (nSample * deltaT) / 1000.0
    extent_time = [0, nTrace, time_length, 0]
    return data, nSample, extent_time


def get_mat(mat_file_path):
    dataset_p = sio.loadmat(mat_file_path)
    print(dataset_p.keys())
    keys = list(dataset_p.keys())
    last_key = keys[-1]
    dataset_p = dataset_p[last_key]  # 假设文件中存有字符变量是matrix，
    return dataset_p


def predict_data_extract_paired_patches(noise_data, clean_data, patch_length=256, stride=128):
    """
    从噪声和干净数据中滑窗提取一一对应的 patch。
    输入：
        noise_data: shape = (1151, 16000)
        clean_data: shape = (1151, 16000)
    输出：
        noise_patches: shape = (N, 1, patch_length)
        clean_patches: shape = (N, 1, patch_length)
    """
    n_samples, n_traces = noise_data.shape
    noise_patches = []
    clean_patches = []

    # std_thresh = 0
    for trace in range(n_traces):
        start = 0
        while start + patch_length <= n_samples:
            n_patch = noise_data[start:start + patch_length, trace]
            c_patch = clean_data[start:start + patch_length, trace]

            noise_patches.append(n_patch[np.newaxis, :])
            clean_patches.append(c_patch[np.newaxis, :])
            start += stride

        # 若最后一段不够 patch_length，则从尾部往前截取完整 patch
        if start < n_samples:
            # end = n_samples
            # start_last = max(end - patch_length, 0)
            #
            # n_patch = noise_data[start_last:end, trace]
            # c_patch = clean_data[start_last:end, trace]
            #
            # noise_patches.append(n_patch[np.newaxis, :])
            # clean_patches.append(c_patch[np.newaxis, :])

            n_remain = noise_data[start:, trace]
            c_remain = clean_data[start:, trace]
            pad_len = patch_length - len(n_remain)

            n_padded = np.pad(n_remain, (0, pad_len), mode='constant')
            c_padded = np.pad(c_remain, (0, pad_len), mode='constant')

            noise_patches.append(n_padded[np.newaxis, :])
            clean_patches.append(c_padded[np.newaxis, :])

    return (
        np.array(noise_patches),  # shape: (N, 1, patch_length)
        np.array(clean_patches)
    )


def extract_paired_patches(noise_data, clean_data, patch_length=256, stride=128):
    """
    从噪声和干净数据中滑窗提取一一对应的 patch。
    输入：
        noise_data: shape = (1151, 16000)
        clean_data: shape = (1151, 16000)
    输出：
        noise_patches: shape = (N, 1, patch_length)
        clean_patches: shape = (N, 1, patch_length)
    """
    n_samples, n_traces = noise_data.shape
    noise_patches = []
    clean_patches = []
    std_thresh = 1e-3
    # std_thresh = 0
    for trace in range(n_traces):
        start = 0
        while start + patch_length <= n_samples:
            n_patch = noise_data[start:start + patch_length, trace]
            c_patch = clean_data[start:start + patch_length, trace]

            # 添加筛选条件
            if np.sum(c_patch) != 0 and np.std(c_patch) > std_thresh:
                noise_patches.append(n_patch[np.newaxis, :])
                clean_patches.append(c_patch[np.newaxis, :])
            start += stride

        # 补足最后一段
        if start < n_samples:
            # end = n_samples
            # start_last = max(end - patch_length, 0)
            #
            # n_patch = noise_data[start_last:end, trace]
            # c_patch = clean_data[start_last:end, trace]

            n_remain = noise_data[start:, trace]
            c_remain = clean_data[start:, trace]
            pad_len = patch_length - len(n_remain)

            n_patch = np.pad(n_remain, (0, pad_len), mode='constant')
            c_patch = np.pad(c_remain, (0, pad_len), mode='constant')

            if np.sum(c_patch) != 0 and np.std(n_patch) > std_thresh:
                noise_patches.append(n_patch[np.newaxis, :])
                clean_patches.append(c_patch[np.newaxis, :])

    return (
        np.array(noise_patches),  # shape: (N, 1, patch_length)
        np.array(clean_patches)
    )


def extract_paired_patches_3d(noise_data, clean_data, patch_length=256, stride=128):
    """
    支持通道数的版本。
    输入：
        noise_data: shape = (1151, 16000, 2)
        clean_data: shape = (1151, 16000, 1)
    输出：
        noise_patches: shape = (N, 2, patch_length)
        clean_patches: shape = (N, 1, patch_length)
    """
    n_samples, n_traces, n_channels_noise = noise_data.shape
    _, _, n_channels_clean = clean_data.shape

    noise_patches = []
    clean_patches = []
    std_thresh = 1e-3

    for trace in range(n_traces):
        start = 0
        while start + patch_length <= n_samples:
            n_patch = noise_data[start:start + patch_length, trace, :]  # (patch_len, 2)
            c_patch = clean_data[start:start + patch_length, trace, :]  # (patch_len, 1)

            n_patch = n_patch.T  # -> (2, patch_len)
            c_patch = c_patch.T  # -> (1, patch_len)

            if np.sum(c_patch) != 0 and np.std(n_patch) > std_thresh:
                noise_patches.append(n_patch)
                clean_patches.append(c_patch)

            start += stride

        # 补最后一段
        if start < n_samples:
            # n_remain = noise_data[start:, trace, :]  # (残长, 2)
            # c_remain = clean_data[start:, trace, :]  # (残长, 1)
            # pad_len = patch_length - len(n_remain)
            #
            # n_patch = np.pad(n_remain, ((0, pad_len), (0, 0)), mode='constant').T  # -> (2, patch_len)
            # c_patch = np.pad(c_remain, ((0, pad_len), (0, 0)), mode='constant').T  # -> (1, patch_len)
            start_last = n_samples - patch_length
            n_patch = noise_data[start_last:start_last + patch_length, trace, :].T
            c_patch = clean_data[start_last:start_last + patch_length, trace, :].T

            if np.sum(c_patch) != 0 and np.std(n_patch) > std_thresh:
                noise_patches.append(n_patch)
                clean_patches.append(c_patch)

    return (
        np.array(noise_patches),  # shape: (N, 2, patch_length)
        np.array(clean_patches)  # shape: (N, 1, patch_length)
    )


def extract_data_2d_extract_paired_patches(noise_data, clean_data, patch_length=64, stride=32):
    """
    从噪声和干净数据中提取 2D patch（时间 × 震道）。
    输出：
        noise_patches: shape = (N, 1, patch_length, patch_length)
        clean_patches: shape = (N, 1, patch_length, patch_length)
    """
    n_samples, n_traces = noise_data.shape
    noise_patches = []
    clean_patches = []

    std_thresh = 1e-3

    half_patch = patch_length // 2

    for center_trace in range(half_patch, n_traces - half_patch):
        start = 0
        while start + patch_length <= n_samples:
            # 时间窗口
            time_slice = slice(start, start + patch_length)
            # 空间窗口（震道）
            trace_slice = slice(center_trace - half_patch, center_trace + half_patch)

            n_patch = noise_data[time_slice, trace_slice]  # shape: (patch_length, patch_length)
            c_patch = clean_data[time_slice, trace_slice]

            # 筛选条件：避免全零和标准差过小的patch
            if np.sum(c_patch) != 0 and np.std(c_patch) > std_thresh:
                noise_patches.append(n_patch[np.newaxis, :, :])  # (1, patch_length, patch_length)
                clean_patches.append(c_patch[np.newaxis, :, :])
            # noise_patches.append(n_patch[np.newaxis, :, :])  # shape: (1, patch_length, patch_length)
            # clean_patches.append(c_patch[np.newaxis, :, :])
            start += stride

        # 处理最后一段（padding）
        if start < n_samples:
            n_remain = noise_data[start:, center_trace - half_patch:center_trace + half_patch]
            c_remain = clean_data[start:, center_trace - half_patch:center_trace + half_patch]

            pad_len = patch_length - n_remain.shape[0]
            n_padded = np.pad(n_remain, ((0, pad_len), (0, 0)), mode='constant')
            c_padded = np.pad(c_remain, ((0, pad_len), (0, 0)), mode='constant')

            if np.sum(c_patch) != 0 and np.std(c_patch) > std_thresh:
                noise_patches.append(n_patch[np.newaxis, :, :])
                clean_patches.append(c_patch[np.newaxis, :, :])
            # noise_patches.append(n_padded[np.newaxis, :, :])
            # clean_patches.append(c_padded[np.newaxis, :, :])

    noise_patches = np.array(noise_patches)  # shape: (N, 1, patch_length, patch_length)
    clean_patches = np.array(clean_patches)
    return noise_patches, clean_patches


def predict_data_extract_paired_patches_3d(noise_data, clean_data, patch_length=256, stride=128):
    """
    支持通道数的版本。
    输入：
        noise_data: shape = (1151, 16000, 2)
        clean_data: shape = (1151, 16000, 1)
    输出：
        noise_patches: shape = (N, 2, patch_length)
        clean_patches: shape = (N, 1, patch_length)
    """
    n_samples, n_traces, n_channels_noise = noise_data.shape
    # print(clean_data.shape)
    _, _, n_channels_clean = clean_data.shape

    noise_patches = []
    clean_patches = []
    std_thresh = 1e-3

    for trace in range(n_traces):
        start = 0
        while start + patch_length <= n_samples:
            n_patch = noise_data[start:start + patch_length, trace, :]  # (patch_len, 2)
            c_patch = clean_data[start:start + patch_length, trace, :]  # (patch_len, 1)

            n_patch = n_patch.T  # -> (2, patch_len)
            c_patch = c_patch.T  # -> (1, patch_len)

            noise_patches.append(n_patch)
            clean_patches.append(c_patch)

            # if np.sum(c_patch) != 0 and np.std(n_patch) > std_thresh:
            #     noise_patches.append(n_patch)
            #     clean_patches.append(c_patch)

            start += stride

        # 补最后一段
        if start < n_samples:
            # n_remain = noise_data[start:, trace, :]  # (残长, 2)
            # c_remain = clean_data[start:, trace, :]  # (残长, 1)
            # pad_len = patch_length - len(n_remain)
            #
            # n_patch = np.pad(n_remain, ((0, pad_len), (0, 0)), mode='constant').T  # -> (2, patch_len)
            # c_patch = np.pad(c_remain, ((0, pad_len), (0, 0)), mode='constant').T  # -> (1, patch_len)
            start_last = n_samples - patch_length
            n_patch = noise_data[start_last:start_last + patch_length, trace, :].T
            c_patch = clean_data[start_last:start_last + patch_length, trace, :].T

            noise_patches.append(n_patch)
            clean_patches.append(c_patch)

            # if np.sum(c_patch) != 0 and np.std(n_patch) > std_thresh:
            #     noise_patches.append(n_patch)
            #     clean_patches.append(c_patch)

    return (
        np.array(noise_patches),  # shape: (N, 2, patch_length)
        np.array(clean_patches)  # shape: (N, 1, patch_length)
    )


def cwt_real_imag_concat_2d(data, fs=125, totalscal=32, wavelet='cmor1.5-1.0'):
    """
    对形状为 (B, C, H, W) 的 torch.Tensor 数据进行 CWT，
    对最后一个维度（宽度方向）做CWT，
    将实部与虚部堆叠成 (B, C*2, H, W) 返回。

    参数：
        data: torch.Tensor，形状为 (B, C, H, W)
        fs: 采样频率
        totalscal: CWT尺度数量
        wavelet: 小波名称，默认 'cmor1.5-1.0'

    返回：
        torch.Tensor，形状为 (B, C*2, H, W)
    """
    assert data.ndim == 4, "输入必须是形状 (B, C, H, W)"
    B, C, S, T = data.shape

    Fc = pywt.central_frequency(wavelet)
    c = 2 * Fc * totalscal
    scales = c / np.arange(1, totalscal + 1)

    output = np.zeros((B, C * 2, S, T), dtype=np.float32)

    assert data.ndim == 4, "输入必须是形状为 (B, C, S, T)"
    B, C, S, T = data.shape

    # 构造小波尺度
    Fc = pywt.central_frequency(wavelet)
    c = 2 * Fc * totalscal
    scales = c / np.arange(1, totalscal + 1)

    # 初始化输出 (B, C*2, S, T)
    output = np.zeros((B, C * 2, S, T), dtype=np.float32)

    # 遍历每个样本、通道、震道
    for b in range(B):
        for c_idx in range(C):
            for t in range(T):
                sig = data[b, c_idx, :, t]  # shape: (S,) —— 每条震道的时间序列
                if isinstance(sig, torch.Tensor):
                    sig = sig.cpu().numpy()

                # 小波变换
                coefs, _ = pywt.cwt(sig, scales, wavelet, sampling_period=1 / fs)  # (scales, S)
                real_part = np.mean(np.real(coefs), axis=0)  # (S,)
                imag_part = np.mean(np.imag(coefs), axis=0)  # (S,)

                output[b, c_idx * 2, :, t] = real_part
                output[b, c_idx * 2 + 1, :, t] = imag_part

    return torch.from_numpy(output).float()


def cwt_real_imag_concat(data, fs=125):
    """
    对形状为 (B, C, 256) 的 torch.Tensor 数据进行 CWT，并将实部与虚部堆叠成 (B, 2, 256)

    参数：
        data: torch.Tensor，形状为 (B, 1, 256)
        wavelet: 小波函数名称，如 'cmor1.5-1.0'
        totalscal: 使用的尺度数量（即频率层级数）
        fs: 采样频率（Hz）

    返回：
        torch.Tensor，形状为 (B, 3, 256)
    """

    # wavelet = 'cmor3-3'
    assert data.ndim == 3, "输入必须是形状为 (B, 1, T) 的 Tensor"
    totalscal = 32
    wavelet = 'cmor1.5-1.0'
    B, C, T = data.shape

    Fc = pywt.central_frequency(wavelet)
    c = 2 * Fc * totalscal
    scales = c / np.arange(1, totalscal + 1)

    # 初始化输出实部和虚部数组
    # real_out = np.zeros((B, T), dtype=np.float32)
    # imag_out = np.zeros((B, T), dtype=np.float32)
    # features = np.zeros((B, 1, T), dtype=np.float32)
    # 初始化输出数组 (B, C*2, T)
    output = np.zeros((B, C * 2, T), dtype=np.float32)
    for b in range(B):
        for c_idx in range(C):
            sig = data[b, c_idx].cpu().numpy()  # shape: (T,)
            coefs, _ = pywt.cwt(sig, scales, wavelet, sampling_period=1 / fs)  # (scales, T)
            real_part = np.mean(np.real(coefs), axis=0)  # (T,)
            imag_part = np.mean(np.imag(coefs), axis=0)  # (T,)
            output[b, c_idx * 2] = real_part
            output[b, c_idx * 2 + 1] = imag_part
    # for b in range(B):
    #     sig = data[b, 0].cpu().numpy()  # shape: (256,)
    #     coefs, _ = pywt.cwt(sig, scales, wavelet, sampling_period=1/fs)  # (scales, T)
    #     real_out[b] = np.mean(np.real(coefs), axis=0)  # 平均投影到时间维度
    #     imag_out[b] = np.mean(np.imag(coefs), axis=0)
    # magnitude = np.abs(coefs)  # (scales, T)
    # mean_mag = np.mean(magnitude, axis=0)  # (T,)
    # features[b, 0] = mean_mag

    # print("--------------")
    # print(real_out.shape)
    # 堆叠成 (B, 2, T)
    # output = np.stack([real_out, imag_out], axis=1)
    # print(output.shape)
    return torch.from_numpy(output).float()
    # return torch.from_numpy(features).float()


def cwt_real_imag_concat_scales(data, fs=125):
    """
    对形状为 (B, C, T) 的 torch.Tensor 数据进行 CWT，保留每个尺度的实部和虚部，
    并将其堆叠为 (B, 2*C, T, S) 的输出。

    参数：
        data: torch.Tensor，形状为 (B, C, T)
        fs: 采样频率（Hz）

    返回：
        torch.Tensor，形状为 (B, 2*C, T, S)
    """
    assert data.ndim == 3, "输入必须是形状为 (B, C, T) 的 Tensor"
    totalscal = 64
    wavelet = 'cmor1.5-1.0'
    B, C, T = data.shape

    Fc = pywt.central_frequency(wavelet)
    c = 2 * Fc * totalscal
    scales = c / np.arange(1, totalscal + 1)

    # 输出 shape: (B, 2*C, T, S)
    output = np.zeros((B, 2 * C, T, totalscal), dtype=np.float32)

    for b in range(B):
        for c_idx in range(C):
            sig = data[b, c_idx].cpu().numpy()  # (T,)
            coefs, _ = pywt.cwt(sig, scales, wavelet, sampling_period=1 / fs)  # (S, T)

            real_part = np.real(coefs).T  # (T, S)
            imag_part = np.imag(coefs).T  # (T, S)

            output[b, c_idx * 2, :, :] = real_part
            output[b, c_idx * 2 + 1, :, :] = imag_part

    return torch.from_numpy(output).float()

def _smooth_1d_onset(x, win: int):
    """一维平滑，用于直达波初至包络估计。"""
    from scipy.ndimage import uniform_filter1d

    if win <= 1:
        return x.astype(np.float64, copy=False)
    return uniform_filter1d(x.astype(np.float64), size=win, mode="nearest")


def _detect_onset_ramp(
    trace,
    win: int = 8,
    pre_max_thr: float = 1.0,
    win_start_min_thr: float = 0.05,
    win_start_max_thr: float = 1.0,
    win_end_min_thr: float = 2.0,
    search_frac: float = 0.95,
    min_onset: int = 16,
    pre_check_len: int = 48,
) -> int:
    """
    单道：滑动窗口检测振幅从近零单调抬升（适合合成数据直达波前接近 0 的记录）。

    在起点 t 取 win 个采样点（滑动窗口长度，由 ramp_win 指定，默认 8），要求：
      1. 初至前 pre_check_len 内 |x| 不超过 pre_max_thr（默认 1，合成道常为 0）；
      2. 窗口内 |x| 严格单调递增；
      3. 窗口首点在 [win_start_min_thr, win_start_max_thr]，末点 >= win_end_min_thr（如 0.1→…→6）；
         首点下限用于跳过仍为纯 0 的窗口。

    返回窗口起点 t 作为初至索引；若无匹配则选得分最高的候选，避免一律退回 min_onset。
    """
    x = np.asarray(trace, dtype=np.float64).ravel()
    n = x.size
    win = max(3, int(win))
    if n < win + min_onset + 4:
        return int(min(min_onset, max(0, n // 10)))

    search_end = min(n - win + 1, int(n * search_frac))
    search_start = max(min_onset, 0)

    best_t = -1
    best_score = -1.0

    for t in range(search_start, max(search_start + 1, search_end)):
        pre0 = max(0, t - pre_check_len)
        pre = np.abs(x[pre0:t])
        if pre.size >= 4 and float(np.max(pre)) > pre_max_thr:
            continue

        w = np.abs(x[t : t + win])
        if (
            w[0] < win_start_min_thr
            or w[0] > win_start_max_thr
            or w[-1] < win_end_min_thr
        ):
            continue
        if np.all(w[1:] > w[:-1]):
            return int(t)

        # 弱约束候选：多数点非降且振幅抬升明显
        if win_start_min_thr <= w[0] <= win_start_max_thr and w[-1] > w[0]:
            score = float(w[-1] - w[0])
            if np.sum(w[1:] >= w[:-1]) >= win - 2 and score > best_score:
                best_score, best_t = score, t

    if best_t >= 0:
        return int(best_t)

    # 仍失败：在搜索区内找包络斜率最大点
    env = np.abs(x)
    search_end_env = min(n - 1, int(n * search_frac))
    if search_end_env <= search_start + win:
        return int(search_start)
    diff = np.diff(env[search_start:search_end_env])
    if diff.size == 0:
        return int(search_start)
    return int(search_start + int(np.argmax(diff)))


def _detect_onset_jump(
    trace,
    smooth_win: int = 11,
    baseline_len: int = 64,
    jump_ratio: float = 8.0,
    search_frac: float = 0.55,
    min_onset: int = 8,
) -> int:
    """单道：根据包络相对早期背景的抬升检测直达波初至。"""
    x = np.asarray(trace, dtype=np.float64).ravel()
    n = x.size
    if n < baseline_len + min_onset + 2:
        return min(min_onset, max(0, n // 10))

    env = np.abs(x)
    env = _smooth_1d_onset(env, smooth_win)

    bl = max(16, min(baseline_len, n // 8))
    baseline = float(np.median(env[:bl])) + 1e-12
    peak = float(np.max(env[: int(n * search_frac)])) + 1e-12
    thr = max(baseline * jump_ratio, 0.15 * peak)

    search_end = int(n * search_frac)
    for t in range(min_onset, max(min_onset + 1, search_end)):
        if env[t] >= thr:
            return int(t)
    return int(min_onset)


def _detect_onset_sta_lta(
    trace,
    sta: int = 5,
    lta: int = 51,
    ratio_thr: float = 3.5,
    search_frac: float = 0.55,
    min_onset: int = 8,
    smooth_win: int = 11,
    baseline_len: int = 64,
    jump_ratio: float = 8.0,
) -> int:
    """单道：STA/LTA 初至拾取；数据过短时回退到包络抬升法。"""
    x = np.asarray(trace, dtype=np.float64).ravel()
    n = x.size
    sta = max(2, int(sta))
    lta = max(sta + 2, int(lta))
    if n < lta + min_onset + 2:
        return _detect_onset_jump(
            trace,
            smooth_win=smooth_win,
            baseline_len=baseline_len,
            jump_ratio=jump_ratio,
            search_frac=search_frac,
            min_onset=min_onset,
        )

    e = x * x
    c_sta = np.cumsum(np.insert(e, 0, 0.0))
    c_lta = np.cumsum(np.insert(e, 0, 0.0))

    ratio = np.zeros(n, dtype=np.float64)
    for t in range(lta, n):
        s = (c_sta[t] - c_sta[t - sta]) / float(sta)
        l = (c_lta[t] - c_lta[t - lta]) / float(lta)
        ratio[t] = s / (l + 1e-18)

    search_end = int(n * search_frac)
    for t in range(lta + min_onset, search_end):
        if ratio[t] >= ratio_thr:
            return int(t)
    return int(min_onset)


def detect_direct_wave(
    data,
    trace_axis=-1,
    method="ramp",
    smooth_win=11,
    baseline_len=64,
    jump_ratio=8.0,
    search_frac=0.55,
    min_onset=8,
    sta=5,
    lta=51,
    ratio_thr=3.5,
    ramp_win=8,
    ramp_pre_max_thr=1.0,
    ramp_win_start_min_thr=0.05,
    ramp_win_start_max_thr=1.0,
    ramp_win_end_min_thr=2.0,
    ramp_search_frac=0.95,
    ramp_min_onset=16,
    ramp_pre_check_len=48,
):
    """
    对地震数据逐道检测直达波初至时间采样索引（每道单独计算）。

    方法:
        - method="ramp"（默认）：长度为 ramp_win 的滑动窗口，初至前近零、
          窗口内 |x| 单调增大；适合合成数据（直达波前≈0，之后振幅逐级抬升）。
        - method="jump"：包络相对早期中值的抬升（现场含噪数据）。
        - method="sta_lta"：STA/LTA 能量比。

    参数:
        data: ndarray，二维地震剖面。
        trace_axis: 0=(震道,时间)，-1=(时间,震道)。
        method: "ramp" / "jump" / "sta_lta"。
        ramp_*: ramp 法参数（仅 method="ramp" 时使用）。
        其余: jump / sta_lta 法参数。

    返回:
        onsets: ndarray，shape (n_trace,)，每道初至采样点索引（int64）。
    """
    x = np.asarray(data, dtype=np.float64)
    if trace_axis < 0:
        trace_axis = x.ndim + trace_axis
    if trace_axis != 0 and trace_axis != x.ndim - 1:
        raise ValueError("trace_axis 仅支持 0（每行一道）或 -1/最后一维（时间在最后一维）")

    method = str(method).lower()
    onsets = []

    def _pick_trace(trace):
        if method == "ramp":
            return _detect_onset_ramp(
                trace,
                win=ramp_win,
                pre_max_thr=ramp_pre_max_thr,
                win_start_min_thr=ramp_win_start_min_thr,
                win_start_max_thr=ramp_win_start_max_thr,
                win_end_min_thr=ramp_win_end_min_thr,
                search_frac=ramp_search_frac,
                min_onset=ramp_min_onset,
                pre_check_len=ramp_pre_check_len,
            )
        if method == "sta_lta":
            return _detect_onset_sta_lta(
                trace,
                sta=sta,
                lta=lta,
                ratio_thr=ratio_thr,
                search_frac=search_frac,
                min_onset=min_onset,
                smooth_win=smooth_win,
                baseline_len=baseline_len,
                jump_ratio=jump_ratio,
            )
        return _detect_onset_jump(
            trace,
            smooth_win=smooth_win,
            baseline_len=baseline_len,
            jump_ratio=jump_ratio,
            search_frac=search_frac,
            min_onset=min_onset,
        )

    if trace_axis == 0:
        for i in range(x.shape[0]):
            onsets.append(_pick_trace(x[i]))
    else:
        for j in range(x.shape[-1]):
            onsets.append(_pick_trace(x[..., j]))

    return np.asarray(onsets, dtype=np.int64)


def _fill_noise_slice(out, noise_trace, dst_start, dst_end, src_start, src_end, combine):
    """将 noise[src_start:src_end] 叠到 out[dst_start:dst_end]（长度须一致）。"""
    n = min(dst_end - dst_start, src_end - src_start)
    if n <= 0:
        return
    if combine == "add":
        out[dst_start : dst_start + n] += noise_trace[src_start : src_start + n]
    else:
        out[dst_start : dst_start + n] = noise_trace[src_start : src_start + n]


def build_syn_to_field_map(n_syn_trace: int, n_field_trace: int) -> np.ndarray:
    """
    按震道数比例，把合成道索引映射到现场道索引（确定性、可复现）。

    设 q, r = divmod(n_syn, n_field)：
      - 多数现场道各对应 q 条合成道；
      - 余数 r 均分到「最前」与「最后」的现场道：各多负担 1 条合成道（即各覆盖 q+1 条）。
      - 例：800/119 → q=6, r=86 → 前 43、后 43 条 field 各覆盖 7 条 syn，中间 33 条各覆盖 6 条。
      - 例：480/119 → q=4, r=4 → 前 2、后 2 条 field 各覆盖 5 条 syn（与你举的 4 余 4 一致）。

    返回:
        syn_to_field: shape (n_syn_trace,)，syn_to_field[j] 为合成道 j 使用的现场道号。
    """
    n_syn_trace = int(n_syn_trace)
    n_field_trace = int(n_field_trace)
    if n_field_trace <= 0:
        raise ValueError("n_field_trace 须 > 0")
    if n_syn_trace < 0:
        raise ValueError("n_syn_trace 须 >= 0")

    q, r = divmod(n_syn_trace, n_field_trace)
    n_extra_start = r // 2
    n_extra_end = r - n_extra_start

    syn_to_field = np.empty(n_syn_trace, dtype=np.int64)
    syn_j = 0
    for fi in range(n_field_trace):
        count = q + 1 if (fi < n_extra_start or fi >= n_field_trace - n_extra_end) else q
        syn_to_field[syn_j : syn_j + count] = fi
        syn_j += count

    if syn_j != n_syn_trace:
        raise RuntimeError(f"映射长度错误: 已分配 {syn_j}，期望 {n_syn_trace}")
    return syn_to_field


def describe_syn_field_mapping(n_syn_trace: int, n_field_trace: int) -> str:
    """返回 syn–field 块映射的文字说明（便于检查 800/119 等比例）。"""
    q, r = divmod(int(n_syn_trace), int(n_field_trace))
    n_extra_start = r // 2
    n_extra_end = r - n_extra_start
    n_mid = n_field_trace - n_extra_start - n_extra_end
    lines = [
        f"n_syn={n_syn_trace}, n_field={n_field_trace} → q={q}, 余数 r={r}",
        f"  前 {n_extra_start} 条 field 道：各覆盖 {q + 1} 条 syn",
        f"  中间 {n_mid} 条 field 道：各覆盖 {q} 条 syn",
        f"  后 {n_extra_end} 条 field 道：各覆盖 {q + 1} 条 syn",
        f"  合计 syn: {n_extra_start * (q + 1) + n_mid * q + n_extra_end * (q + 1)}",
    ]
    return "\n".join(lines)


def _print_syn_field_onset_info(
    syn_trace_idx: int,
    field_trace_idx: int,
    t_syn: int,
    t_field: int,
    n_syn_time: int,
    n_field_time: int,
) -> None:
    """打印 syn / field 初至及初至前后采样数，便于对比窗口是否够用。"""
    n_before_syn = int(t_syn)
    n_after_syn = int(n_syn_time) - n_before_syn
    n_before_field = int(t_field)
    n_after_field = int(n_field_time) - n_before_field
    deficit_before = max(0, n_before_syn - n_before_field)
    deficit_after = max(0, n_after_syn - n_after_field)
    if deficit_before == 0 and deficit_after == 0:
        flag = "完全对齐"
    else:
        flag = f"初至锚点对齐 + 前/后补足(deficit前{deficit_before},后{deficit_after})"
    syn_lo = max(0, n_before_syn - n_before_field)
    syn_hi = min(n_syn_time, n_before_syn + min(n_after_syn, n_after_field))
    field_lo = max(0, t_field + (syn_lo - n_before_syn))
    field_hi = min(n_field_time, t_field + min(n_after_syn, n_after_field))
    lines = [
        f"[syn道 {syn_trace_idx} → field道 {field_trace_idx}] {flag}",
        f"  syn:   初至={t_syn},  初至前={n_before_syn},  初至后={n_after_syn}",
        f"  field: 初至={t_field},  初至前={n_before_field},  初至后={n_after_field}",
        f"  主映射: syn[{syn_lo}:{syn_hi}] ↔ field[{field_lo}:{field_hi}] "
        f"(初至 syn[{n_before_syn}] ↔ field[{t_field}])",
    ]
    if deficit_before > 0:
        lines.append(
            f"  前段补足: syn[0:{deficit_before}] ← field[0:{deficit_before}]"
        )
    if deficit_after > 0:
        lines.append(
            f"  后段补足: syn[{n_before_syn + n_after_field}:{n_syn_time}] ← "
            f"field[{t_field}:{t_field + deficit_after}] "
            f"(field初至后最前 {deficit_after} 点)"
        )
    print("\n".join(lines))


def _apply_field_noise_onset_aligned(
    syn_trace: np.ndarray,
    noise_trace: np.ndarray,
    t_field: int,
    n_before: int,
    n_after: int,
    combine: str,
    noise_scale: float = 1.0,
) -> np.ndarray:
    """
    按初至锚点叠 field 噪声，并对 syn 未覆盖段用 field 初至前/后缘补足。

    1) 主段：f = t_field + (d - n_before)，syn 初至与 field 初至对齐。
    2) 前缺 deficit_before = max(0, n_before - t_field)：
       syn[0:deficit] ← field[0:deficit]（用 field 初至前最前 deficit 个点补足）。
    3) 后缺 deficit_after = max(0, n_after - (n_f - t_field))：
       syn 尾部 ← field[t_field : t_field+deficit]（field 初至后最前 deficit 个点）。
    """
    out = np.asarray(syn_trace, dtype=np.float64).copy()
    noise_trace = np.asarray(noise_trace, dtype=np.float64).ravel() * float(noise_scale)
    n_f = int(noise_trace.size)
    t_f = int(t_field)
    n_seg = out.size

    # 主映射：初至对齐
    for d in range(n_seg):
        f = t_f + (d - n_before)
        if 0 <= f < n_f:
            if combine == "add":
                out[d] = out[d] + noise_trace[f]
            else:
                out[d] = noise_trace[f]

    n_before_field = t_f
    n_after_field = n_f - t_f
    deficit_before = max(0, n_before - n_before_field)
    deficit_after = max(0, n_after - n_after_field)

    # 前段补足：field[0:deficit_before] → syn[0:deficit_before]
    if deficit_before > 0:
        src_end = min(t_f, deficit_before)
        _fill_noise_slice(out, noise_trace, 0, src_end, 0, src_end, combine)

    # 后段补足：field 初至后最前 deficit_after 点 → syn 尾部
    if deficit_after > 0:
        dst_start = n_before + n_after_field
        dst_end = min(n_seg, dst_start + deficit_after)
        src_start = t_f
        src_end = min(n_f, t_f + (dst_end - dst_start))
        _fill_noise_slice(
            out, noise_trace, dst_start, dst_end, src_start, src_end, combine
        )

    return out


def _pick_field_trace_random_scan(
    rng: np.random.Generator,
    field_onsets: np.ndarray,
    n_before: int,
    n_after: int,
    n_field_time: int,
) -> tuple[int, int]:
    """
    随机打乱后遍历全部 field 道，取第一条满足：
      初至前 t_field >= n_before，初至后 (n_field_time - t_field) >= n_after。
    """
    n_field = int(field_onsets.size)
    order = rng.permutation(n_field)

    for fi in order:
        fi = int(fi)
        t_f = int(field_onsets[fi])
        t_field_before = t_f
        t_field_after = n_field_time - t_f
        if t_field_before >= n_before and t_field_after >= n_after:
            return fi, t_f

    # 全部不满足：选可截长度最大的道，并提示
    best_fi, best_t_f = 0, int(field_onsets[0])
    best_score = -1
    for fi in range(n_field):
        t_f = int(field_onsets[fi])
        score = min(t_f, n_before) + min(n_field_time - t_f, n_after)
        if score > best_score:
            best_score, best_fi, best_t_f = score, fi, t_f

    return best_fi, best_t_f


def add_field_noise(
    noise,
    field_data,
    syn_data,
    syn_method="ramp",
    field_method="jump",
    method=None,
    ramp_win=5,
    noise_scale=1.0,
    combine="add",
    map_mode="proportional",
    rng=None,
    print_mismatch=False,
    print_mapping=False,
    **detect_kw,
):
    """
    按直达波初至对齐，将现场 VMD 残差噪声叠加到合成剖面上。

    数据约定：三者均为 shape=(时间点, 震道)，即第 0 维为时间、第 1 维为道。
    例如 field/noise (1500, 119)，syn (1151, 800)。

    对每条合成道 j：
        1. 拾取 syn、field 初至；
        2. n_before=t_syn，n_after=n_syn_time-t_syn；
        3. 确定对应的 field 道 fi（见 map_mode）；
        4. 初至锚点对齐叠加 field 噪声；syn 初至前/后若仍有未覆盖样点，
           分别用 field 初至前最后 N 点、初至后最前 N 点补足（N 为缺口长度）。

    map_mode:
        "proportional"（默认）：按 n_syn/n_field 块映射，余数均分在首尾 field 道；
            800 条 syn、119 条 field → 800÷119=6 余 86（非 4 余 4）。
        "random"：每条 syn 随机打乱遍历 field，取第一条满足窗口的道（旧逻辑）。

    参数:
        noise: 现场残差噪声（field - VMD），shape 与 field_data 相同。
        field_data: 现场数据，用于拾取 field 初至。
        syn_data: 合成剖面。
        syn_method: 合成道初至，默认 "ramp"（滑动窗口单调抬升，适合初至前≈0）。
        field_method: 现场道初至，默认 "jump"（包络突变）。
        ramp_win: ramp 法滑动窗口长度（采样点数），默认 8；仅影响 syn_method="ramp" 时。
        method: 若指定则 syn、field 共用同一方法（覆盖 syn_method/field_method）。
        combine: "add" 为 syn+噪声段；"replace" 为整条道替换为噪声段。
        noise_scale: 噪声强度系数，叠加前 noise 乘以该值（如 1.05 表示略增强 5%）。
        map_mode: "proportional" 或 "random"。
        rng: 仅 map_mode="random" 时使用。
        print_mismatch: True 时逐道打印 syn/field 初至及初至前后采样数。
        print_mapping: True 时打印块映射说明（describe_syn_field_mapping）。
        **detect_kw: 传给 detect_direct_wave 的其余参数。

    返回:
        field_noise_syn_data: 与 syn_data 同 shape。
    """
    noise = np.asarray(noise, dtype=np.float64)
    field_data = np.asarray(field_data, dtype=np.float64)
    syn_data = np.asarray(syn_data, dtype=np.float64)

    if noise.ndim != 2 or field_data.ndim != 2 or syn_data.ndim != 2:
        raise ValueError("noise、field_data、syn_data 均须为二维 (时间点, 震道)")
    if noise.shape != field_data.shape:
        raise ValueError(f"noise shape {noise.shape} 与 field_data {field_data.shape} 不一致")

    if method is not None:
        syn_method = field_method = method

    detect_kw_field = dict(detect_kw)
    detect_kw_syn = dict(detect_kw)
    detect_kw_syn["ramp_win"] = int(ramp_win)

    field_onsets = detect_direct_wave(
        field_data, trace_axis=-1, method=field_method, **detect_kw_field
    )
    syn_onsets = detect_direct_wave(
        syn_data, trace_axis=-1, method=syn_method, **detect_kw_syn
    )

    n_field_time, n_field_trace = field_data.shape
    n_syn_time, n_syn_trace = syn_data.shape

    if syn_onsets.size != n_syn_trace:
        raise ValueError(f"syn 初至数 {syn_onsets.size} 与道数 {n_syn_trace} 不符")

    map_mode = str(map_mode).lower()
    if map_mode not in ("proportional", "random"):
        raise ValueError('map_mode 仅支持 "proportional" 或 "random"')

    if print_mapping:
        print(describe_syn_field_mapping(n_syn_trace, n_field_trace))

    syn_to_field = None
    if map_mode == "proportional":
        syn_to_field = build_syn_to_field_map(n_syn_trace, n_field_trace)
    else:
        if rng is None:
            rng = np.random.default_rng(42)

    out = np.array(syn_data, dtype=np.float64, copy=True)
    combine = str(combine).lower()
    if combine not in ("add", "replace"):
        raise ValueError('combine 仅支持 "add" 或 "replace"')

    for j in range(n_syn_trace):
        t_syn = int(syn_onsets[j])
        n_before = t_syn
        n_after = n_syn_time - t_syn

        if map_mode == "proportional":
            fi = int(syn_to_field[j])
            t_field = int(field_onsets[fi])
        else:
            fi, t_field = _pick_field_trace_random_scan(
                rng, field_onsets, n_before, n_after, n_field_time
            )

        if print_mismatch:
            _print_syn_field_onset_info(
                j, fi, t_syn, t_field, n_syn_time, n_field_time
            )

        out[:, j] = _apply_field_noise_onset_aligned(
            syn_data[:, j],
            noise[:, fi],
            t_field,
            n_before,
            n_after,
            combine,
            noise_scale=noise_scale,
        )

    return out.astype(syn_data.dtype, copy=False)


# 示例用法
if __name__ == "__main__":
    noise = np.load("merged_output.npy")

    sea, _, _ = get_info_seg("data/field_data/Sea_0_1_shot.sgy")  # get_info_seg 返回 (采样点, 震道)
    data, seismic_time, time_length = get_info_seg("data/sgy_data/2007BP_part4_11shot.sgy")
    print(describe_syn_field_mapping(data.shape[1], sea.shape[1]))
    x = add_field_noise(
        noise=noise,
        field_data=sea,
        syn_data=data,
        noise_scale=0.90,
        map_mode="proportional",
        print_mapping=False,
        print_mismatch=True,
    )
    # plot_seismic_npy(x, time_length, show=True)
    # plot_seismic_npy(x, time_length, show=True)
    mat_path = 'syn_s0.90_vmd.mat'
    savemat(mat_path, {'data': x})
    np.save('syn_s0.90_vmd', x)
    # ---------- detect_direct_wave 示例 ----------
    # 方式 1：用合成数据快速验证（不依赖本地文件）
    # n_trace, n_time = 20, 512
    # rng = np.random.default_rng(42)
    # sea_syn = rng.normal(0, 0.02, size=(n_trace, n_time)).astype(np.float32)
    # onset_true = 80
    # sea_syn[:, onset_true:] += 1.5  # 模拟直达波后振幅抬升

    # 必须用 trace_axis=-1：每列一道，沿第 0 维才是时间；写 0 会把每一行当“道”，且只得到 1500 个错误结果
    # onsets_jump = detect_direct_wave(sea_syn, trace_axis=-1, method="jump")
    # onsets_sta = detect_direct_wave(sea_syn, trace_axis=-1, method="sta_lta")
    # print("数据 shape (采样点, 震道):", sea_syn.shape, "→ onsets 长度应为震道数:", sea_syn.shape[1])
    # print("jump  初至 min/mean/max:", onsets_jump.min(), onsets_jump.mean(), onsets_jump.max())
    # print("sta_lta 初至 min/mean/max:", onsets_sta.min(), onsets_sta.mean(), onsets_sta.max())
    # print("第 0 道 jump / sta_lta:", onsets_jump[0], onsets_sta[0], "（合成真值约", onset_true, "）")

    # 画一道记录，标出拾取的初至（取消注释可弹出图窗）
    # trace_id = 0
    # t_axis = np.arange(n_time)
    # plt.figure(figsize=(10, 3))
    # plt.plot(t_axis, sea_syn[trace_id], "k", lw=0.8, label="第0道")
    # plt.axvline(onsets_jump[trace_id], color="r", ls="--", label=f"jump 初至={onsets_jump[trace_id]}")
    # plt.axvline(onsets_sta[trace_id], color="b", ls=":", label=f"sta_lta 初至={onsets_sta[trace_id]}")
    # plt.xlabel("采样点"); plt.ylabel("振幅"); plt.legend(); plt.title("直达波初至拾取示例")
    # plt.tight_layout(); plt.show()

    # 方式 1b：数据为 (采样点, 震道) = (n_time, n_trace) 时（你的常见格式）
    # n_time, n_trace = 512, 20
    # sea_nt = rng.normal(0, 0.02, size=(n_time, n_trace)).astype(np.float32)
    # sea_nt[onset_true:, :] += 1.5
    # onsets = detect_direct_wave(sea_nt, trace_axis=-1, method="jump")  # -1 可省略
    # print("shape (采样点, 震道):", sea_nt.shape, "→ onsets.shape:", onsets.shape)
    # 用法：第 j 道 → sea_nt[:onsets[j], j] 为直达波前，sea_nt[onsets[j]:, j] 为之后

    # 方式 2：用真实 .mat / .npy（按你的数据路径修改）
    # sea = get_mat("data/JMD_data/Train_data/JMD_K3.mat")
    # sea = np.load("sea_data.npy")
    # 若 shape 为 (采样点, 震道)，用默认即可：
    # onsets = detect_direct_wave(sea, method="jump")
    # 若 shape 为 (震道, 采样点)，则 trace_axis=0
    # print("初至统计:", onsets.min(), onsets.mean(), onsets.max())

    # ---------- 其它工具示例（按需取消注释） ----------
    # t = get_mat("data/JMD_data/Train_data/JMD_K3.mat")
    # print(t.shape)





