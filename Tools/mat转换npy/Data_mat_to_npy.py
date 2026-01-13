import h5py, numpy as np, json, os
from scipy.signal import stft, get_window
from datetime import datetime
import ctypes

# 统一转换规范（首版）：fs=100e6, NFFT=1024, hop=512, window=Hann, 输出 512x512，功率->dB->[-80,0]dB->[0,1] float16
fs = 100_000_000
nfft = 1024
hop = 512
win = get_window('hann', nfft, fftbins=True)
clip_db = (-80.0, 0.0)
seconds = 0.01  # 取前 10 ms

# 待处理 MAT 文件清单（按需使用，不在主流程执行）
MAT_LIST = [
    r"D:\无人机射频信号数据集\DroneRFa\DroneRFa\T1110_S0000.mat",
    r"D:\无人机射频信号数据集\DroneRFa\DroneRFa\T1110_S0001.mat",
    r"D:\无人机射频信号数据集\DroneRFa\DroneRFa\T1110_S0010.mat",
    r"D:\无人机射频信号数据集\DroneRFa\DroneRFa\T1110_S0011.mat",
    r"D:\无人机射频信号数据集\DroneRFa\DroneRFa\T1110_S0100.mat",
    r"D:\无人机射频信号数据集\DroneRFa\DroneRFa\T1110_S0101.mat",
    r"D:\无人机射频信号数据集\DroneRFa\DroneRFa\T1110_S0110.mat",
    r"D:\无人机射频信号数据集\DroneRFa\DroneRFa\T1110_S0111.mat",
    r"D:\无人机射频信号数据集\DroneRFa\DroneRFa\T1110_S1000.mat",
    r"D:\无人机射频信号数据集\DroneRFa\DroneRFa\T1110_S1001.mat",
    r"D:\无人机射频信号数据集\DroneRFa\DroneRFa\T1110_S1010.mat",
    r"D:\无人机射频信号数据集\DroneRFa\DroneRFa\T1110_S1011.mat",
]

# 输出目录（类别14对应 T1110），本版本主流程不写入
OUT_DIR = r"C:\Users\HP\Desktop\Exp\MatData\14"

# 仅分析的参考目录（你的原始 NPY 目录）
INSPECT_DIR = r"C:\Users\HP\Desktop\Exp\Data\14"
BASE_DATA_DIR = r"C:\Users\HP\Desktop\Exp\Data"


# ===== 新增：从 MAT/HDF5 递归提取中心频率与带宽（采样率） =====
_KEYWORDS_CENTER = ("center", "centre", "fc", "centerfreq", "center_frequency")
_KEYWORDS_FREQ = ("freq", "frequency")
_KEYWORDS_BW = ("bandwidth", "bw")
_KEYWORDS_FS = ("fs", "samplerate", "sample_rate", "samplingrate", "sampling_rate")


def _to_float(x):
    try:
        if isinstance(x, (bytes, bytearray)):
            x = x.decode("utf-8", "ignore")
        if isinstance(x, str):
            s = x.strip().lower()
            # 允许字符串中带单位
            # 优先提取数值
            import re
            m = re.search(r"([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)", s)
            val = float(m.group(1)) if m else None
            if val is None:
                return None
            if "mhz" in s:
                return val * 1e6
            if "khz" in s:
                return val * 1e3
            if "hz" in s:
                return val
            # 无单位，按 Hz 处理
            return val
        if isinstance(x, (int, float, np.integer, np.floating)):
            return float(x)
        if isinstance(x, np.ndarray):
            if x.size == 0:
                return None
            y = x.reshape(-1)[0]
            return _to_float(y)
    except Exception:
        return None
    return None


def _path_of(obj):
    try:
        return obj.name
    except Exception:
        return "<root>"


def _score_center(key_lc: str) -> int:
    s = 0
    if any(k in key_lc for k in _KEYWORDS_CENTER):
        s += 2
    if any(k in key_lc for k in _KEYWORDS_FREQ):
        s += 1
    return s


def _score_bw(key_lc: str) -> int:
    return 2 if any(k in key_lc for k in _KEYWORDS_BW) else 0


def _score_fs(key_lc: str) -> int:
    # 明确带有 sample_rate 更可信
    if "sample" in key_lc and "rate" in key_lc:
        return 2
    return 1 if any(k in key_lc for k in _KEYWORDS_FS) else 0


def extract_meta_from_mat(mat_path: str):
    """扫描 MAT(v7.3/HDF5) 文件，尽力找到中心频率与带宽/采样率。
    返回 dict: {
      'center_freq_mhz': float|None,
      'bandwidth_mhz': float|None,
      'fs_hz': float|None,
      'sources': { 'center': (path, key), 'bandwidth': (path, key), 'fs': (path, key) }
    }
    不做范围校验，单位自动识别（Hz/kHz/MHz/无单位按 Hz）。
    """
    res = {
        'center_freq_mhz': None,
        'bandwidth_mhz': None,
        'fs_hz': None,
        'sources': {'center': None, 'bandwidth': None, 'fs': None},
    }
    best_center = (-1, None, None, None)  # (score, value_hz, path, key)
    best_bw = (-1, None, None, None)
    best_fs = (-1, None, None, None)
    # 打开文件并遍历 attrs 与小型 dataset 值
    mat_path_use = _win_short_path(mat_path)
    with h5py.File(mat_path_use, 'r') as f:
        # 根属性
        for k in list(getattr(f, 'attrs', {})):
            key_lc = str(k).lower()
            v = _to_float(f.attrs[k])
            if v is None:
                continue
            sc = _score_center(key_lc)
            if sc > best_center[0]:
                best_center = (sc, v, '/', k)
            sbw = _score_bw(key_lc)
            if sbw > best_bw[0]:
                best_bw = (sbw, v, '/', k)
            sfs = _score_fs(key_lc)
            if sfs > best_fs[0]:
                best_fs = (sfs, v, '/', k)
        # 递归遍历
        def _cb(name, obj):
            # 对象属性
            try:
                for k in list(getattr(obj, 'attrs', {})):
                    key_lc = str(k).lower()
                    v = _to_float(obj.attrs[k])
                    if v is None:
                        continue
                    sc = _score_center(key_lc)
                    if sc > best_center[0]:
                        best_center = (sc, v, _path_of(obj), k)
                    sbw = _score_bw(key_lc)
                    if sbw > best_bw[0]:
                        best_bw = (sbw, v, _path_of(obj), k)
                    sfs = _score_fs(key_lc)
                    if sfs > best_fs[0]:
                        best_fs = (sfs, v, _path_of(obj), k)
            except Exception:
                pass
            # 小型数据集本体（例如 Fs = 1x1 标量）
            try:
                if isinstance(obj, h5py.Dataset):
                    if obj.size <= 4 and (np.issubdtype(obj.dtype, np.number) or obj.dtype.kind in ("S", "U")):
                        val = obj[()]
                        v = _to_float(val)
                        if v is not None:
                            key_lc = name.lower()
                            sc = _score_center(key_lc)
                            if sc > best_center[0]:
                                best_center = (sc, v, _path_of(obj), '<dataset>')
                            sbw = _score_bw(key_lc)
                            if sbw > best_bw[0]:
                                best_bw = (sbw, v, _path_of(obj), '<dataset>')
                            sfs = _score_fs(key_lc)
                            if sfs > best_fs[0]:
                                best_fs = (sfs, v, _path_of(obj), '<dataset>')
            except Exception:
                pass
        f.visititems(_cb)

    # 汇总（Hz->MHz）
    if best_center[1] is not None and best_center[0] >= 0:
        res['center_freq_mhz'] = float(best_center[1]) / 1e6
        res['sources']['center'] = (best_center[2], str(best_center[3]))
    if best_bw[1] is not None and best_bw[0] >= 0:
        res['bandwidth_mhz'] = float(best_bw[1]) / 1e6
        res['sources']['bandwidth'] = (best_bw[2], str(best_bw[3]))
    if best_fs[1] is not None and best_fs[0] >= 0:
        res['fs_hz'] = float(best_fs[1])
        res['sources']['fs'] = (best_fs[2], str(best_fs[3]))

    # 若未找到 bandwidth，则用 fs 近似（复基带情况下等于采样率）
    if res['bandwidth_mhz'] is None and res['fs_hz'] is not None:
        res['bandwidth_mhz'] = res['fs_hz'] / 1e6
    return res


def iq_to_spectrogram(i, q, frames=543):
    # 按 543 段分割：每段长度 nfft，步长 hop，得到恰好 frames=543 帧
    x = i.astype(np.float32) + 1j * q.astype(np.float32)
    need = (frames - 1) * hop + nfft
    if x.shape[0] < need:
        pad = need - x.shape[0]
        x = np.pad(x, (0, pad), mode='constant')
    else:
        x = x[:need]
    # 分帧
    idx0 = np.arange(0, need - nfft + 1, hop, dtype=np.int64)  # 长度应为 frames
    # 安全起见按 frames 截断/对齐
    if idx0.shape[0] > frames:
        idx0 = idx0[:frames]
    # 组帧矩阵 (frames, nfft)
    frames_arr = np.stack([x[s:s + nfft] for s in idx0], axis=0)
    # 加窗
    frames_arr = frames_arr * win.astype(np.float32)[None, :]
    # FFT 幅度平方
    X = np.fft.fft(frames_arr, n=nfft, axis=1)
    P = (np.abs(X) ** 2).astype(np.float32)
    # 取前 512 频点，得到 (frames, 512)
    P = P[:, :512]
    # dB -> 归一化到 [0,1]
    SdB = 10.0 * np.log10(P + 1e-12)
    SdB = np.clip(SdB, clip_db[0], clip_db[1])
    S01 = (SdB - clip_db[0]) / (clip_db[1] - clip_db[0])
    return S01.astype(np.float16)


def save_sidecar_json(out_npy_path, meta):
    meta_path = os.path.splitext(out_npy_path)[0] + '.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta_path


# Windows 下将含非 ASCII 的路径转换为 8.3 短路径，失败则返回原路径
def _win_short_path(path: str) -> str:
    if os.name != 'nt':
        return path
    try:
        GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW
        GetShortPathNameW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint]
        GetShortPathNameW.restype = ctypes.c_uint
        out_buf = ctypes.create_unicode_buffer(260)
        res = GetShortPathNameW(path, out_buf, 260)
        if res > 0:
            return out_buf.value
    except Exception:
        pass
    return path


def convert_one(mat_path, out_dir=OUT_DIR, channel='RF0', hw=(543, 512), confirm_meta=False):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(mat_path))[0]
    out_npy = os.path.join(out_dir, f"{base}_{channel}_fromMAT.npy")
    mat_path_use = _win_short_path(mat_path)

    # 先探测元数据：center_freq / bandwidth / fs
    detected = extract_meta_from_mat(mat_path_use)
    # 优先使用探测到的 fs（若有）用于时间长度估计
    local_fs = float(detected.get('fs_hz') or fs)

    if confirm_meta:
        print("== meta preview ==")
        print("  file:", mat_path)
        print("  center_freq_mhz:", detected.get('center_freq_mhz'), "source:", detected['sources'].get('center'))
        print("  bandwidth_mhz:", detected.get('bandwidth_mhz'), "source:", detected['sources'].get('bandwidth'))
        print("  fs_hz:", detected.get('fs_hz'), "source:", detected['sources'].get('fs'))
        ans = input("Proceed with these meta values? [y/N]: ").strip().lower()
        if ans != 'y':
            print('skip convert by user decision.')
            return None, None

    with h5py.File(mat_path_use, 'r') as f:
        I = f[f'{channel}_I'][0]
        Q = f[f'{channel}_Q'][0]
    # 精确按 543 帧构造所需样本数
    frames = hw[0] if isinstance(hw, (tuple, list)) else 543
    need = (frames - 1) * hop + nfft
    I = I[:need]
    Q = Q[:need]
    S = iq_to_spectrogram(I, Q, frames=frames)
    np.save(out_npy, S)
    # 统计与元数据（用 float32 计算以避免溢出）
    S32 = S.astype(np.float32)
    stats = {
        'min': float(S32.min()), 'max': float(S32.max()), 'mean': float(S32.mean()), 'std': float(S32.std())
    }
    meta = {
        'source_mat': mat_path,
        'channel': channel,
        'fs': local_fs,  # 若有探测，用探测结果
        'nfft': nfft,
        'hop': hop,
        'window': 'hann',
        'clip_db': list(clip_db),
        'frames': frames,
        'samples_used': int(need),
        'seconds_used': float(need / max(1.0, local_fs)),
        'target_hw': [frames, 512],
        'saved_at': datetime.now().isoformat(timespec='seconds'),
        'stats': stats,
        # 新增：频谱相关元数据（不做范围判断）
        'center_freq_mhz': (float(detected['center_freq_mhz']) if detected['center_freq_mhz'] is not None else None),
        'bandwidth_mhz': (float(detected['bandwidth_mhz']) if detected['bandwidth_mhz'] is not None else None),
        'meta_sources': detected.get('sources', {}),
    }
    save_sidecar_json(out_npy, meta)
    print(f"saved: {out_npy} shape: {S.shape} dtype: {S.dtype}")
    return out_npy, stats


# 批量转换
def convert_many(paths, out_dir=OUT_DIR, channel='RF0', hw=(543, 512), confirm_meta=False):
    ok, fail = 0, 0
    for p in paths:
        if not os.path.isfile(p):
            print('skip (not found):', p)
            fail += 1
            continue
        try:
            convert_one(p, out_dir=out_dir, channel=channel, hw=hw, confirm_meta=confirm_meta)
            ok += 1
        except Exception as e:
            print('failed convert:', p, '|', e)
            fail += 1
    print(f'convert summary -> ok={ok}, fail={fail}, out_dir={out_dir}')


def dump_h5_tree(mat_path: str, max_items: int = 200):
    """打印 HDF5/MAT v7.3 文件的树结构（路径、shape、dtype、部分属性）。"""
    try:
        p_use = _win_short_path(mat_path)
        with h5py.File(p_use, 'r') as f:
            print(f"== H5 tree: {mat_path}")
            count = 0
            def _show_attrs(obj):
                out = {}
                try:
                    for k in obj.attrs:
                        v = obj.attrs[k]
                        try:
                            if isinstance(v, bytes):
                                v = v.decode('utf-8', 'ignore')
                            elif isinstance(v, np.ndarray) and v.size == 1:
                                v = v.reshape(-1)[0]
                            out[str(k)] = str(v)
                        except Exception:
                            out[str(k)] = '<unrepr>'
                except Exception:
                    pass
                return out
            def _cb(name, obj):
                nonlocal count
                if count >= max_items:
                    return
                try:
                    if isinstance(obj, h5py.Dataset):
                        shape = obj.shape
                        dtype = str(obj.dtype)
                        attrs = _show_attrs(obj)
                        print(f"[DS] {obj.name} shape={shape} dtype={dtype} attrs={attrs}")
                    elif isinstance(obj, h5py.Group):
                        attrs = _show_attrs(obj)
                        print(f"[GR] {obj.name} attrs={attrs}")
                except Exception as e:
                    print(f"[??] {name} err={e}")
                count += 1
            f.visititems(_cb)
    except Exception as e:
        print(f"dump_h5_tree failed: {e}")


def scan_mats(paths):
    """仅扫描 MAT 文件，提取 center_freq_mhz / bandwidth_mhz / fs_hz，并输出汇总，不做转换。
    """
    import math
    rows = []
    uniq_center, uniq_bw, uniq_fs = set(), set(), set()
    none_center = none_bw = none_fs = 0
    for p in paths:
        item = {'path': p}
        try:
            meta = extract_meta_from_mat(p)
            cf = meta.get('center_freq_mhz')
            bw = meta.get('bandwidth_mhz')
            fs_hz = meta.get('fs_hz')
            item.update({
                'center_freq_mhz': cf,
                'bandwidth_mhz': bw,
                'fs_hz': fs_hz,
                'sources': meta.get('sources', {})
            })
            if cf is None:
                none_center += 1
            else:
                uniq_center.add(round(float(cf), 6))
            if bw is None:
                none_bw += 1
            else:
                uniq_bw.add(round(float(bw), 6))
            if fs_hz is None:
                none_fs += 1
            else:
                uniq_fs.add(round(float(fs_hz), 3))
        except Exception as e:
            item['error'] = str(e)
        rows.append(item)
    # 输出逐条结果
    print('== scan results ==')
    for r in rows:
        print(json.dumps(r, ensure_ascii=False))
    # 汇总
    summary = {
        'total': len(paths),
        'unique_center_count': len(uniq_center), 'unique_center_values_mhz': sorted(list(uniq_center)), 'none_center': none_center,
        'unique_bandwidth_count': len(uniq_bw), 'unique_bandwidth_values_mhz': sorted(list(uniq_bw)), 'none_bandwidth': none_bw,
        'unique_fs_count': len(uniq_fs), 'unique_fs_values_hz': sorted(list(uniq_fs)), 'none_fs': none_fs,
        'center_consistent': (len(uniq_center) == 1 and none_center == 0),
        'bandwidth_consistent': (len(uniq_bw) == 1 and none_bw == 0),
        'fs_consistent': (len(uniq_fs) == 1 and none_fs == 0),
    }
    print('== summary ==')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _gather_mats(root_dir: str, regex: str | None = None):
    import re as _re
    mats = []
    pat = _re.compile(regex) if regex else None
    for dp, _, files in os.walk(root_dir):
        for fn in files:
            if not fn.lower().endswith('.mat'):
                continue
            if pat is not None and not pat.search(fn):
                continue
            mats.append(os.path.join(dp, fn))
    mats.sort()
    return mats


if __name__ == '__main__':
    # 若命令行传入了 MAT 文件路径，则执行“仅扫描”模式；支持 --dump 与 --scan-dir
    import sys as _sys
    args = _sys.argv[1:]
    if args:
        if args[0] == '--dump':
            mats = args[1:]
            for m in mats:
                dump_h5_tree(m)
        elif args[0] == '--scan-dir':
            root = args[1] if len(args) >= 2 else '.'
            regex = args[2] if len(args) >= 3 else None
            files = _gather_mats(root, regex)
            print(f"scan dir: {root} regex={regex} count={len(files)}")
            if not files:
                print("no .mat found")
            else:
                scan_mats(files)
        else:
            scan_mats(args)
    else:
        # 示例：如需转换可取消注释
        # convert_many(MAT_LIST, out_dir=OUT_DIR, channel='RF0', hw=(543, 512), confirm_meta=False)
        pass