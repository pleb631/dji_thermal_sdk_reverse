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

## 官方 SDK 与本仓库的性能对比

在同一个 Linux Docker 容器（`python:3.12`）和 Python 进程中，交替调用官方 SDK 的 `libdirp.so` 和本仓库的 `measure_float32`。使用 DJI Thermal SDK 1.8 样本集的 49 张 R-JPEG，预热后各测 20 轮，每种实现共测 980 次。

**全样本平均耗时从 SDK 的 173.88 ms/张降到本仓库的 1.65 ms/张，实测加速约 105.1 倍。** 49 张图的 float32 输出逐字节一致，两边均匹配参考 SHA-256。

| 机型 | 样本数 | 官方 SDK（ms/张） | 本仓库（ms/张） |
| :--- | ---: | ---: | ---: |
| H20N | 5 | 148.33 | 1.36 |
| H20T | 4 | 28.51 | 1.62 |
| H30T | 5 | 878.89 | 4.32 |
| M2EA | 5 | 136.89 | 1.23 |
| M30T | 5 | 74.64 | 1.13 |
| M3T | 5 | 154.69 | 1.27 |
| M3TD | 10 | 84.33 | 1.49 |
| M4T | 5 | 100.69 | 1.64 |
| XT S | 5 | 18.38 | 0.99 |
| **全样本平均** | **49** | **173.88** | **1.65** |

输入预先读入内存。SDK 计时包含输出缓冲区分配、句柄创建、`dirp_measure_ex` 和句柄销毁。本仓库计时覆盖 `measure_float32` 的完整调用。两边都排除了文件读写、容器启动和哈希校验。加速比以舍入前的平均耗时计算。

将 SDK 和样本复制到容器本地文件系统后复测 3 轮，H30T（1280 × 1024）平均耗时为 SDK **867.12 ms/张**、本仓库 **3.88 ms/张**。

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

## 🧪 运行测试与基准测速

### 单元与位级回归测试

```bash
python -m unittest discover -s tests -v
```

包含官方 SDK 1.8 全部 49 个 R-JPEG 样本的位级回归测试（SHA-256 校验）。

### 性能基准测试

```bash
python benchmark.py --runs 9
```

此脚本测量本仓库实现，输出总体平均延迟、吞吐率（FPS）和各机型耗时。上面的对比通过 Python `ctypes` 直接调用官方动态库。
