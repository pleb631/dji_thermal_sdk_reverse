# DJI Thermal SDK 纯 Python 测温实现 (SDK-Free)

本项目是对 [DJI Thermal SDK](https://www.dji.com/cn/downloads/softwares/dji-thermal-sdk) 核心测温算法与 R-JPEG 格式解析的纯 Python / NumPy 向量化实现。

无需依赖官方 C SDK 或动态链接库，直接解算 DJI 各系列无人机红外热成像图像并输出摄氏度（float32）温度矩阵，计算结果与官方 SDK 保持 **100% 位级一致（Bit-Exact）**。

## 📷 支持机型

涵盖 DJI 常见红外机型：
- **Zenmuse 系列**: H20N, H20T, H30T, XT S
- **Matrice 系列**: M30T, M3T, M3TD, M4T
- **御 Mavic 系列**: Mavic 2 Enterprise Advanced (M2EA)

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

## 🧪 运行测试

```bash
python -m unittest discover -s tests -v
```
包含官方 SDK 1.8 全部 49 个 R-JPEG 样本的位级回归测试（SHA-256 校验）。
