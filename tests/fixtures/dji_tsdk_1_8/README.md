# DJI Thermal SDK 1.8 测温夹具

`dataset/` 保留官方 SDK 1.8 样本包中的 49 张 JPEG 及原始相对路径。
`manifest.json` 记录输入文件的 SHA-256、温度图尺寸（高、宽），以及
`dji_irp -a measure --measurefmt float32` 原始小端输出的 SHA-256。生成参考值时省略了其他测温参数。

在 `dji_thermal_sdk` 目录运行：

```sh
python -m unittest discover -s tests -v
```

测试只需要 Python 标准库、测温包及本目录的夹具。
