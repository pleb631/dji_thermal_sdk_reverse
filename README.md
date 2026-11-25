# DJI Thermal SDK 纯 Python 测温实现 (SDK-Free)

本项目是对 [DJI Thermal SDK](https://www.dji.com/cn/downloads/softwares/dji-thermal-sdk) 核心测温算法与 R-JPEG 格式解析的纯 Python / NumPy 向量化实现。

无需依赖官方 C SDK 或动态链接库，直接解算 DJI 各系列无人机红外热成像图像并输出摄氏度（float32）温度矩阵，计算结果与官方 SDK 保持 **100% 位级一致（Bit-Exact）**。

---

## 📷 机型协议与识别标记

程序通过解析 JPEG 二进制流中的 **APP 私有标记段及其魔数签名（Magic Bytes）** 自动识别协议与测温算法，无需依赖外部机型元数据：

| 内部协议 | JPEG 标记段 | 识别特征 (Magic Bytes) | 涵盖机型 / 模式 |
| :--- | :--- | :--- | :--- |
| **XT S** | **APP2** (`0xFFE2`) | `b"Plug"` | Zenmuse XT S |
| **GIRP** | **APP4** (`0xFFE4`) | `b"\xff\xd2\xd1\xffgirp"` | Matrice 30T（双增益差分模式） |
| **HIRP** | **APP4** (`0xFFE4`) | `b"\xff\xd2\xd1\xffhirp"` | Matrice 3TD、Matrice 4T（微测辐射计逆响应） |
| **DIRP** | **APP4** (`0xFFE4`) | `b"\xaa\x55\x12\x06"` | Zenmuse H20T（APP4 多项式校准） |
| **IIRP** | **APP4** (`0xFFE4`) | `b"\xff\xd2\xd1\xffiirp"` 或通用兜底 | H20N, H30T, M2EA, M3T 及标准测温模式 |

---

## ⚡ 性能与分机型测温耗时

基于官方 SDK 1.8 全部 49 个官方 R-JPEG 样本库进行基准测试（纯内存解析与解算，预热后 5 轮采样）：
- **全样本平均耗时**：**~15.8 ms / 张**（约 63 FPS）
- **端到端测试耗时**：约 **0.8 秒** 完成 49 张图像的读取、解算与 SHA-256 位级校验

| 相机机型 | 对应协议 | 典型分辨率 | 单张平均耗时 | 官方样本数 |
| :--- | :--- | :--- | :--- | :--- |
| **Matrice 3TD** | HIRP / IIRP | 640 × 512 | **6.90 ms** | 10 |
| **Zenmuse XT S** | XT S | 640 × 512 | **9.72 ms** | 5 |
| **Matrice 4T** | HIRP / IIRP | 640 × 512 | **10.65 ms** | 5 |
| **Zenmuse H20N** | IIRP | 640 × 512 | **11.04 ms** | 5 |
| **Matrice 3T** | IIRP | 640 × 512 | **11.55 ms** | 5 |
| **Mavic 2 EA** | IIRP | 640 × 512 | **14.18 ms** | 5 |
| **Zenmuse H20T** | DIRP | 640 × 512 | **29.66 ms** | 4 |
| **Matrice 30T** | GIRP / IIRP | 1280 × 1024 | **29.83 ms** | 5 |
| **Zenmuse H30T** | IIRP | 1280 × 1024 | **29.95 ms** | 5 |

---

## 🚀 快速使用

### 环境要求
- Python 3.10+
- `pip install numpy`

### Python API

```python
from thermal_measurement import measure_float32

# 读取 DJI R-JPEG 二进制文件
with open("DJI_0001_R.JPG", "rb") as f:
    jpeg_bytes = f.read()

# 解算温度矩阵（单位：摄氏度）
temperatures, (height, width) = measure_float32(jpeg_bytes)

# temperatures 为 NumPy float32 数组，可直接 reshape
temp_grid = temperatures.reshape((height, width))
print(f"分辨率: {width}x{height}")
print(f"最高温: {temp_grid.max():.2f}°C, 最低温: {temp_grid.min():.2f}°C, 均温: {temp_grid.mean():.2f}°C")
```

### CLI 命令行

```bash
python -m thermal_measurement.radiometry -s input_R.jpg -o output.raw -a measure --measurefmt float32
```

---

## 🧪 运行测试

```bash
python -m unittest discover -s tests -v
```
包含官方 SDK 1.8 全部 49 个 R-JPEG 样本的位级回归测试（SHA-256 校验）。
