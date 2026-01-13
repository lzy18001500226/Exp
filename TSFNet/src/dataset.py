"""
数据加载与增强模块 - 参考西电论文思路
支持DroneRFb-Spectra数据集 + 真实干扰混合 + 课程式调度
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from pathlib import Path
import random
from typing import Optional, Tuple, List
import os

# 新增：模块级 worker 初始化函数（可被 Windows 多进程 pickling）
def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class DroneRFDataset(Dataset):
    """
    DroneRFb-Spectra数据集加载器
    数据格式: .npy文件, shape=(512, 512), dtype=np.float16
    """
    def __init__(
        self,
        data_list_file: str,
        interference_pool_dir: Optional[str] = None,
        size: int = 512,
        augment: bool = True,
        curriculum_epoch: int = 0,
        max_epoch: int = 100,
        enable_coord: bool = True,
        enable_ss: bool = False,
        ss_freq_bins: int = 5,
        data_root: Optional[str] = None,
    ):
        """
        Args:
            data_list_file: 数据列表文件路径 (格式: path label)
            interference_pool_dir: 干扰信号池目录 (包含背景/WiFi/蓝牙等)
            size: 谱图尺寸 (默认512)
            augment: 是否启用增强
            curriculum_epoch: 当前训练轮次 (用于课程式调度)
            max_epoch: 最大训练轮次
            enable_coord: 是否添加坐标通道
            enable_ss: 是否启用频域分块
            ss_freq_bins: 频域分块数量
        """
        super().__init__()
        
        self.list_path = Path(data_list_file).resolve()
        self.data_root = Path(data_root).resolve() if data_root else self._guess_data_root(self.list_path)
        
        # 加载数据列表（兼容两种格式："path label" 或 仅 "path"）
        self.data = []
        with open(data_list_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # 去除包裹引号
                line = line.strip().strip('"').strip("'")
                parts = line.split()
                raw_path = parts[0].strip().strip('"').strip("'")
                path = self._resolve_path(raw_path)
                label: Optional[int] = None
                if len(parts) >= 2:
                    try:
                        label = int(parts[1])
                    except Exception:
                        label = None
                if label is None:
                    # 首选父目录名作为类别（如 .../Data/0/0-a.npy -> 0）
                    try:
                        parent = os.path.basename(os.path.dirname(str(path)))
                        if parent.isdigit():
                            label = int(parent)
                    except Exception:
                        label = None
                if label is None:
                    # 其次：文件名前缀的起始数字（如 3-xxx.npy -> 3）
                    base = os.path.basename(str(path))
                    num = ''
                    for ch in base:
                        if ch.isdigit():
                            num += ch
                        else:
                            break
                    if num != '':
                        label = int(num)
                if label is None:
                    label = 0
                self.data.append((str(path), label))
        
        self.size = size
        self.augment = augment
        self.curriculum_epoch = curriculum_epoch
        self.max_epoch = max_epoch
        self.enable_coord = enable_coord
        self.enable_ss = enable_ss
        self.ss_freq_bins = ss_freq_bins
        
        # 加载干扰池 (如果提供)
        self.interference_pool = []
        if interference_pool_dir and Path(interference_pool_dir).exists():
            print(f"加载干扰池: {interference_pool_dir}")
            pool_path = Path(interference_pool_dir)
            for npy_file in pool_path.glob("*.npy"):
                try:
                    spec = np.load(str(npy_file))
                    if spec.shape == (size, size):
                        self.interference_pool.append(spec.astype(np.float32))
                except:
                    continue
            print(f"干扰池样本数: {len(self.interference_pool)}")
        
        # 计算课程式调度参数
        self._update_curriculum_params()
    
    def _update_curriculum_params(self):
        """根据当前epoch更新课程式调度参数"""
        progress = self.curriculum_epoch / max(self.max_epoch, 1)
        
        # 课程式SINR范围 (参考西电论文)
        if progress < 1/3:
            # 早期: 容易场景
            self.sinr_range = (10, 20)
            self.snr_range = (5, 15)
        elif progress < 2/3:
            # 中期: 中等场景
            self.sinr_range = (0, 10)
            self.snr_range = (0, 10)
        else:
            # 后期: 困难场景
            self.sinr_range = (-3, 5)
            self.snr_range = (-3, 5)
        
        # 增强概率
        self.p_shift = 0.5
        self.p_erase = min(0.3 + 0.2 * progress, 0.5)  # 逐渐增加遮挡概率
        self.p_mix = 0.5  # 真实干扰混合概率
        
    def set_epoch(self, epoch: int):
        """更新当前epoch (用于课程式调度)"""
        self.curriculum_epoch = epoch
        self._update_curriculum_params()
    
    def _load_spectrogram(self, path: str) -> np.ndarray:
        """加载谱图，支持 .npy 与图像(.png/.jpg/.jpeg/.bmp/.tif/.tiff)
        - 对 .npy：按原逻辑 min-max 归一化到 [0,1]
        - 对图像：按位深线性归一化到 [0,1]，再按需要 resize
        """
        p = Path(path)
        suf = p.suffix.lower()
        if suf in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}:
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img is None:
                # 若图像读取失败且同名 .npy 存在，则回退到 .npy
                npy_try = p.with_suffix('.npy')
                if npy_try.exists():
                    return self._load_spectrogram(str(npy_try))
                raise FileNotFoundError(f"无法读取图像文件: {p}")
            # 转灰度
            if img.ndim == 3:
                if img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 线性归一化到 [0,1]
            if img.dtype == np.uint8:
                spec = img.astype(np.float32) / 255.0
            elif img.dtype == np.uint16:
                spec = img.astype(np.float32) / 65535.0
            else:
                spec = img.astype(np.float32)
                imin, imax = float(spec.min()), float(spec.max())
                if imax - imin > 1e-8:
                    spec = (spec - imin) / (imax - imin)
                else:
                    spec = np.zeros_like(spec, dtype=np.float32)
            # 尺寸对齐
            if spec.shape != (self.size, self.size):
                spec = cv2.resize(spec, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
            return spec.astype(np.float32, copy=False)
        
        # 默认按 .npy 处理
        spec = np.load(str(p))
        # 压缩到二维
        if getattr(spec, 'ndim', 2) > 2:
            spec = np.squeeze(spec)
        # OpenCV 对 float16 resize 支持不好，先转 float32
        if spec.dtype != np.float32:
            spec = spec.astype(np.float32, copy=False)
        # 尺寸对齐
        if spec.shape != (self.size, self.size):
            spec = cv2.resize(spec, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        # 归一化到[0, 1]
        smin, smax = float(spec.min()), float(spec.max())
        if smax - smin > 1e-8:
            spec = (spec - smin) / (smax - smin)
        else:
            spec = np.zeros_like(spec, dtype=np.float32)
        return spec
    
    def _add_awgn(self, spec: np.ndarray, target_snr_db: float) -> np.ndarray:
        """
        添加高斯白噪声 (参考西电论文公式)
        SNR_dB = 10*log10(P_signal / P_noise)
        """
        # 计算信号功率
        P_sig = np.mean(spec ** 2)
        
        # 计算噪声功率
        P_noise = P_sig / (10 ** (target_snr_db / 10))
        
        # 生成噪声
        noise = np.random.randn(*spec.shape) * np.sqrt(P_noise)
        
        # 添加噪声
        noisy_spec = spec + noise
        
        # Clip到合理范围
        noisy_spec = np.clip(noisy_spec, 0, 1)
        
        return noisy_spec.astype(np.float32)
    
    def _random_shift(self, spec: np.ndarray) -> np.ndarray:
        """
        随机时频平移 (小幅度, 保持绝对频率语义)
        时间: ±2~5%, 频率: ±1~3%
        """
        H, W = spec.shape
        
        # 时间轴平移 (±2~5%)
        shift_t = int(np.random.uniform(-0.05, 0.05) * W)
        
        # 频率轴平移 (±1~3%, 更小以保持绝对频率)
        shift_f = int(np.random.uniform(-0.03, 0.03) * H)
        
        # Roll (循环移位)
        spec_shifted = np.roll(spec, shift_t, axis=1)
        spec_shifted = np.roll(spec_shifted, shift_f, axis=0)
        
        return spec_shifted
    
    def _random_erase(self, spec: np.ndarray) -> np.ndarray:
        """
        随机遮挡 (Cutout)
        时间: 5~20%, 频率: 3~10%
        """
        H, W = spec.shape
        
        # 遮挡尺寸
        erase_h = int(np.random.uniform(0.03, 0.10) * H)
        erase_w = int(np.random.uniform(0.05, 0.20) * W)
        
        # 随机位置
        top = np.random.randint(0, H - erase_h + 1)
        left = np.random.randint(0, W - erase_w + 1)
        
        # 填充值: 局部中位数或低幅噪声
        fill_value = np.median(spec[top:top+erase_h, left:left+erase_w])
        
        # 应用遮挡
        spec_erased = spec.copy()
        spec_erased[top:top+erase_h, left:left+erase_w] = fill_value
        
        return spec_erased
    
    def _mix_interference(self, spec: np.ndarray, target_sinr_db: float) -> np.ndarray:
        """
        真实干扰混合 (核心DA, 参考西电论文)
        在线性功率域混合: S_mix = S_uav + α·B + N_awgn
        """
        if len(self.interference_pool) == 0:
            return spec
        
        # 随机采样干扰
        interference = random.choice(self.interference_pool).copy()
        
        # 随机频率平移 (±40MHz 等效到像素)
        shift_ratio = np.random.uniform(-0.4, 0.4)  # ±40%
        shift_pixels = int(shift_ratio * self.size)
        interference = np.roll(interference, shift_pixels, axis=0)
        
        # 计算功率
        P_sig = np.mean(spec ** 2) + 1e-8
        P_int = np.mean(interference ** 2) + 1e-8
        
        # 计算缩放系数 (西电公式)
        # α = P_s / (P_b · 10^{SINR/10})
        alpha = P_sig / (P_int * (10 ** (target_sinr_db / 10)))
        
        # 混合
        mixed_spec = spec + alpha * interference
        
        # 添加小量AWGN
        P_noise = P_sig / (10 ** ((target_sinr_db + 5) / 10))  # 额外5dB余量
        noise = np.random.randn(*spec.shape) * np.sqrt(P_noise)
        mixed_spec = mixed_spec + noise
        
        # 归一化到[0, 1]
        mixed_spec = (mixed_spec - mixed_spec.min()) / (mixed_spec.max() - mixed_spec.min() + 1e-8)
        
        return mixed_spec.astype(np.float32)
    
    def _extract_edge(self, spec: np.ndarray) -> np.ndarray:
        """提取边缘特征 (Sobel + NMS)"""
        # Sobel算子
        sobel_x = cv2.Sobel(spec, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(spec, cv2.CV_32F, 0, 1, ksize=3)
        
        # 梯度幅值
        edge = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # 归一化
        edge = (edge - edge.min()) / (edge.max() - edge.min() + 1e-8)
        
        return edge.astype(np.float32)
    
    def _extract_corner(self, spec: np.ndarray) -> np.ndarray:
        """提取角点特征（加速版 Harris）
        - 先下采样到 1/2 分辨率做 Harris，再上采样回原尺寸；
        - 使用 float32 输入（cornerHarris 推荐），避免不必要的 uint8 转换。
        """
        H, W = spec.shape
        # 低分辨率计算
        small = cv2.resize(spec, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        small32 = small.astype(np.float32, copy=False)
        corner_s = cv2.cornerHarris(small32, blockSize=2, ksize=3, k=0.04)
        corner_s = np.abs(corner_s, dtype=np.float32)
        # 上采样回原尺寸
        corner = cv2.resize(corner_s, (W, H), interpolation=cv2.INTER_LINEAR)
        # 归一化
        cmin, cmax = float(corner.min()), float(corner.max())
        if cmax - cmin > 1e-8:
            corner = (corner - cmin) / (cmax - cmin)
        else:
            corner = np.zeros_like(corner, dtype=np.float32)
        return corner.astype(np.float32)
    
    def _add_coord_channels(self, spec: np.ndarray) -> np.ndarray:
        """
        添加坐标通道 (绝对位置编码)
        返回: [原始, t_coord, f_coord] 3通道
        """
        H, W = spec.shape
        
        # 时间坐标 (列方向, 0~1)
        t_coord = np.linspace(0, 1, W).reshape(1, -1).repeat(H, axis=0)
        
        # 频率坐标 (行方向, 0~1)
        f_coord = np.linspace(0, 1, H).reshape(-1, 1).repeat(W, axis=1)
        
        # Stack成3通道
        spec_with_coord = np.stack([spec, t_coord, f_coord], axis=0)  # (3, H, W)
        
        return spec_with_coord.astype(np.float32)
    
    def _generate_views(self, spec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成双流输入视图
        纹理流: 伪彩色(RGB) 或 原始+2coord
        位置流: edge + corner + 原始
        """
        # === 纹理流: 直接用原始+2coord (方案A) ===
        if self.enable_coord:
            I_tex = self._add_coord_channels(spec)  # (3, H, W)
        else:
            # 单通道重复3次
            I_tex = np.stack([spec, spec, spec], axis=0)  # (3, H, W)
        
        # === 位置流: edge + corner + 原始 ===
        edge = self._extract_edge(spec)
        corner = self._extract_corner(spec)
        I_pos = np.stack([edge, corner, spec], axis=0)  # (3, H, W)
        
        return I_tex, I_pos
    
    def _apply_ss(self, spec: np.ndarray) -> np.ndarray:
        """频域分块（SS）：沿频率轴将谱图均匀切分为 ss_freq_bins 份，返回其中一份并重采样回原尺寸。
        - 训练(augment=True)：随机选择一个频段。
        - 验证/测试(augment=False)：选择中间频段（向下取整的中位索引）。
        """
        if not self.enable_ss or self.ss_freq_bins <= 1:
            return spec
        H, W = spec.shape
        bin_h = max(H // self.ss_freq_bins, 1)
        # 选择分段索引
        if self.augment:
            idx = random.randint(0, self.ss_freq_bins - 1)
        else:
            idx = (self.ss_freq_bins - 1) // 2
        start = idx * bin_h
        end = H if idx == self.ss_freq_bins - 1 else (idx + 1) * bin_h
        seg = spec[start:end, :]
        # 重采样回原尺寸（保持时间轴宽度不变）
        seg_resized = cv2.resize(seg, (W, H), interpolation=cv2.INTER_LINEAR)
        return seg_resized.astype(np.float32)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        返回:
            I_tex: 纹理流输入 (3, H, W)
            I_pos: 位置流输入 (3, H, W)
            label: 类别标签
        """
        path, label = self.data[index]
        
        # 加载谱图
        spec = self._load_spectrogram(path)

        # === 训练时增强 ===
        if self.augment:
            # 1) AWGN
            target_snr = np.random.uniform(*self.snr_range)
            spec = self._add_awgn(spec, target_snr)
            
            # 2) 时频平移
            if random.random() < self.p_shift:
                spec = self._random_shift(spec)
            
            # 3) 随机遮挡
            if random.random() < self.p_erase:
                spec = self._random_erase(spec)
            
            # 4) 真实干扰混合 (核心)
            if random.random() < self.p_mix and len(self.interference_pool) > 0:
                target_sinr = np.random.uniform(*self.sinr_range)
                spec = self._mix_interference(spec, target_sinr)

        # === 频域分块（SS） ===
        spec = self._apply_ss(spec)

        # === 生成双流输入 ===
        I_tex, I_pos = self._generate_views(spec)
        
        # 转Tensor
        I_tex = torch.from_numpy(I_tex)  # (3, H, W)
        I_pos = torch.from_numpy(I_pos)  # (3, H, W)
        
        return I_tex, I_pos, label
    
    def __len__(self) -> int:
        return len(self.data)


    def _guess_data_root(self, list_path: Path) -> Path:
        # 优先：同层或上层存在 Data 目录
        cur = list_path.parent
        for _ in range(5):
            cand = cur / 'Data'
            if cand.exists() and cand.is_dir():
                return cand
            cur = cur.parent
        # 兜底：使用列表文件上两级目录（通常是 Exp）
        return list_path.parent.parent

    def _resolve_path(self, raw: str) -> Path:
        p = Path(raw)
        # 相对路径优先基于 data_root
        if not p.is_absolute():
            s = str(p)
            if s.startswith('./Data') or s.startswith('.\\Data') or s.startswith('Data'):
                # 去掉前导的 ./Data 或 .\Data 或 Data
                if s.startswith('./Data'):
                    rest = s[len('./Data'):]
                elif s.startswith('.\\Data'):
                    rest = s[len('.\\Data'):]
                elif s.startswith('Data'):
                    rest = s[len('Data'):]
                else:
                    rest = s
                rest = rest.lstrip('\\/').replace('/', os.sep).replace('\\', os.sep)
                p = self.data_root / rest
            else:
                p = (self.list_path.parent / p).resolve()
        # 若不存在，尝试多种候选
        if not p.exists():
            parent = p.parent
            stem = p.stem
            # 1) 显式 .npy 同名
            if p.suffix.lower() != '.npy':
                cand = p.with_suffix('.npy')
                if cand.exists():
                    return cand
            # 2) 显式 "-a.npy" 命名（对齐他人实现 path+'-a.npy'）
            cand_a = parent / f"{stem}-a.npy"
            if cand_a.exists():
                return cand_a
            # 3) 模糊匹配：优先包含 "-a" 的文件
            if parent.exists():
                matches = sorted(parent.glob(f"{stem}*.npy"))
                if matches:
                    # 优先挑选包含 "-a" 的候选
                    for m in matches:
                        if f"{stem}-a.npy".lower() == m.name.lower() or '-a' in m.stem.lower():
                            return m
                    return matches[0]
        return p

def create_dataloaders(
    train_list: str,
    val_list: str,
    test_list: str,
    interference_pool_dir: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    max_epoch: int = 100,
    data_root: Optional[str] = None,
    limit_train: Optional[int] = None,
    limit_val: Optional[int] = None,
    limit_test: Optional[int] = None,
    seed: int = 42,
    deterministic: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练/验证/测试DataLoader
    
    Args:
        train_list: 训练集列表文件
        val_list: 验证集列表文件
        test_list: 测试集列表文件
        interference_pool_dir: 干扰池目录
        batch_size: 批大小
        num_workers: 数据加载线程数
        max_epoch: 最大训练轮次
        data_root: 数据根目录（用于解析相对路径）
    
    Returns:
        train_loader, val_loader, test_loader
    """
    g = torch.Generator()
    g.manual_seed(seed)

    # 训练集 (启用增强)
    train_dataset = DroneRFDataset(
        data_list_file=train_list,
        interference_pool_dir=interference_pool_dir,
        augment=True,
        max_epoch=max_epoch,
        enable_coord=True,
        data_root=data_root,
    )
    if isinstance(limit_train, int) and limit_train > 0 and len(train_dataset) > limit_train:
        idx = list(range(len(train_dataset)))[:limit_train]
        train_dataset.data = [train_dataset.data[i] for i in idx]
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=not deterministic,  # 若要求确定性，可固定顺序
        num_workers=num_workers,
        pin_memory=True,
        drop_last=not deterministic,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=(num_workers > 0),
    )

    # 验证集 (不增强)
    val_dataset = DroneRFDataset(
        data_list_file=val_list,
        interference_pool_dir=None,  # 验证时不混合干扰
        augment=False,
        enable_coord=True,
        data_root=data_root,
    )
    if isinstance(limit_val, int) and limit_val > 0 and len(val_dataset) > limit_val:
        idx = list(range(len(val_dataset)))[:limit_val]
        val_dataset.data = [val_dataset.data[i] for i in idx]
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=(num_workers > 0),
    )

    # 测试集 (不增强)
    test_dataset = DroneRFDataset(
        data_list_file=test_list,
        interference_pool_dir=None,
        augment=False,
        enable_coord=True,
        data_root=data_root,
    )
    if isinstance(limit_test, int) and limit_test > 0 and len(test_dataset) > limit_test:
        idx = list(range(len(test_dataset)))[:limit_test]
        test_dataset.data = [test_dataset.data[i] for i in idx]
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    """测试数据加载"""
    # 示例: 创建测试dataset
    dataset = DroneRFDataset(
        data_list_file="./experiment_groups/1-known_for_train",
        interference_pool_dir="./Data/interference_pool",  # 干扰池路径
        augment=True,
        curriculum_epoch=50,
        max_epoch=100,
        enable_coord=True,
    )
    
    print(f"数据集大小: {len(dataset)}")
    print(f"干扰池大小: {len(dataset.interference_pool)}")
    print(f"当前SINR范围: {dataset.sinr_range}")
    print(f"当前SNR范围: {dataset.snr_range}")
    
    # 加载一个样本
    I_tex, I_pos, label = dataset[0]
    print(f"\n纹理流输入: {I_tex.shape}, 范围: [{I_tex.min():.3f}, {I_tex.max():.3f}]")
    print(f"位置流输入: {I_pos.shape}, 范围: [{I_pos.min():.3f}, {I_pos.max():.3f}]")
    print(f"标签: {label}")
