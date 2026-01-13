import os, struct, zlib, mmap, sys, re
import numpy as np

# 修复：避免 bytes.join 与 int 混用导致的异常
def head_ascii(data, minlen=4):
    if not data:
        return []
    buf = bytearray()
    for b in data:
        if 32 <= b < 127:
            buf.append(b)
        else:
            buf.append(0)
    parts = bytes(buf).split(b"\x00")
    return [p.decode('ascii', 'ignore') for p in parts if len(p) >= minlen]

def entropy_bytes(data):
    if not data: return 0.0
    arr = np.frombuffer(data, dtype=np.uint8)
    hist = np.bincount(arr, minlength=256)/len(arr)
    nz = hist[hist>0]
    return float(-(nz*np.log2(nz)).sum())

def try_zip_sig(h): return h.startswith(b'PK\x03\x04')
def try_hdf5_sig(h): return h.startswith(b'\x89HDF\r\n\x1a\n')
def try_npy_sig(h): return h.startswith(b'\x93NUMPY')
def try_mat_sig(h): return b'MATLAB 5.0 MAT-file' in h[:512]
def try_riff_wav(h): return h[:4]==b'RIFF' and h[8:12]==b'WAVE'
def try_pcap(h):
    if len(h) < 4:
        return False
    return struct.unpack("<I", h[:4])[0] in (0xa1b2c3d4, 0xd4c3b2a1, 0xa1b23c4d, 0x4d3cb2a1)

def try_mat73_sig(h):
    # MATLAB 7.3 MAT-file header 是 ASCII 文本开头，包含 "MATLAB 7.3 MAT-file" 和 "HDF5"
    try:
        hdr = h[:256].decode('ascii', 'ignore')
        return 'MATLAB 7.3 MAT-file' in hdr and 'HDF5' in hdr
    except Exception:
        return False

def sniff_known(path, head):
    info = {}
    # MATLAB v7.3 (HDF5 容器)
    if try_mat73_sig(head):
        info['format'] = 'MATLAB v7.3 (HDF5)'
        # 尝试使用 h5py 读取结构
        try:
            import h5py
            summary = []
            iq_like = []
            def walk(name, obj):
                if isinstance(obj, h5py.Dataset):
                    dt = str(obj.dtype)
                    sh = tuple(obj.shape)
                    summary.append((name, dt, sh))
                    # 简单IQ变量名提示
                    lname = name.lower()
                    if lname.endswith(('/rf0_i', '/rf0_q', '/rf1_i', '/rf1_q')) or re.search(r'/(rf\d+_[iq])$', lname):
                        iq_like.append(name)
            with h5py.File(path, 'r') as f:
                f.visititems(walk)
            # 控制输出长度
            summary = summary[:50]
            info['datasets'] = summary
            if iq_like:
                info['iq_candidates'] = iq_like[:20]
        except Exception as e:
            info['h5py_error'] = str(e)
        return info
    # WAV
    if try_riff_wav(head):
        try:
            import wave
            with wave.open(path, 'rb') as w:
                info['format'] = 'WAV'
                info['nchannels'] = w.getnchannels()
                info['sampwidth'] = w.getsampwidth() * 8
                info['framerate'] = w.getframerate()
                info['nframes'] = w.getnframes()
            return info
        except Exception:
            pass
    # NPY
    if try_npy_sig(head):
        try:
            arr = np.load(path, mmap_mode='r', allow_pickle=False)
            info['format'] = 'NPY'
            # 某些NPY可能不是ndarray（极少见），做兜底
            try:
                info['dtype'] = str(arr.dtype)
                info['shape'] = tuple(arr.shape)
            except Exception:
                info['dtype'] = 'unknown'; info['shape'] = 'unknown'
            return info
        except Exception:
            pass
    # NPZ/ZIP
    if try_zip_sig(head):
        info['format'] = 'ZIP/NPZ'
        try:
            npz = np.load(path, allow_pickle=False)
            if hasattr(npz, 'files'):
                info['npz_keys'] = list(npz.files)
        except Exception:
            pass
        return info
    # MATLAB v5
    if try_mat_sig(head):
        info['format'] = 'MATLAB v5'; return info
    # HDF5（标准签名以 0x89 'HDF' 开头；MAT 7.3 前部是ASCII，不命中这里）
    if try_hdf5_sig(head):
        info['format'] = 'HDF5'; return info
    # PCAP
    if try_pcap(head):
        info['format'] = 'PCAP'; return info
    return None

def guess_iq(path, max_bytes=4*1024*1024):
    with open(path, 'rb') as f:
        data = f.read(max_bytes)
    size = len(data)
    guesses = []
    for name, dtype in [('int16', np.int16), ('int8', np.int8), ('float32', np.float32)]:
        itemsize = np.dtype(dtype).itemsize
        # 至少若干对IQ样本
        if size < 4 * itemsize:
            continue
        n_items = (size // itemsize)
        # 偶数对 (I,Q)
        n_items = (n_items // 2) * 2
        if n_items < 2:
            continue
        try:
            arr = np.frombuffer(data[:n_items * itemsize], dtype=dtype)
        except ValueError:
            continue
        I = arr[0::2].astype(np.float32)
        Q = arr[1::2].astype(np.float32)
        if name != 'float32':
            scale = (127.5 if name == 'int8' else 32767.5)
            if scale <= 0:
                continue
            I /= scale; Q /= scale
        cplx = I + 1j * Q
        # 统计量（稳健）
        muI, muQ = float(I.mean()), float(Q.mean())
        vI, vQ = float(I.var() + 1e-9), float(Q.var() + 1e-9)
        if I.size >= 2 and Q.size >= 2:
            n = min(10000, I.size)
            try:
                corr = float(np.corrcoef(I[:n], Q[:n])[0, 1])
                if not np.isfinite(corr):
                    corr = 0.0
            except Exception:
                corr = 0.0
        else:
            corr = 0.0
        if cplx.size > 0:
            energy = float((np.abs(cplx[:min(65536, cplx.size)]) ** 2).mean())
        else:
            energy = 0.0
        score = 0
        if abs(muI) < 0.1 and abs(muQ) < 0.1: score += 1
        if 0.001 < vI < 10 and 0.001 < vQ < 10: score += 1
        if abs(corr) < 0.5: score += 1
        if energy > 1e-4: score += 1
        guesses.append((name, {'mean': (muI, muQ), 'var': (vI, vQ), 'corrIQ': corr, 'energy': energy, 'score': score}))
    guesses.sort(key=lambda x: x[1]['score'], reverse=True)
    return guesses[:3]

def analyze(path):
    st = os.stat(path)
    print(f'File: {path}')
    print(f'Size: {st.st_size} bytes')
    with open(path, 'rb') as f:
        head = f.read(4096)
    if not head:
        print('Empty or unreadable file head. Aborting.')
        return
    info = sniff_known(path, head)
    ent = entropy_bytes(head)
    comp = len(zlib.compress(head)) / max(1, len(head))
    print(f'Head entropy: {ent:.2f} bits/byte | zlib ratio on head: {comp:.2f}')
    strings = head_ascii(head)[:10]
    if strings:
        print('ASCII hints:', strings)
    if info:
        print('Detected:', info)
        return
    print('No known container detected. Trying IQ guesses...')
    guesses = guess_iq(path)
    for name, g in guesses:
        print(f'As {name} IQ -> score={g["score"]} mean={g["mean"]} var={g["var"]} corrIQ={g["corrIQ"]:.3f} energy={g["energy"]:.4g}')
    if not guesses:
        print('Not enough data or not IQ-like.')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python inspect_binary.py <binary_file>')
        sys.exit(1)
    analyze(sys.argv[1])