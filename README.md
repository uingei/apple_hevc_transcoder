<div align="center">

# 🍏 Apple HEVC 批量转码 GUI v1.7.0  
*A full-featured Windows GUI tool for Apple HEVC (H.265) batch transcoding*

[![License](https://img.shields.io/badge/License-MIT-green.svg)]()
[![Platform](https://img.shields.io/badge/Platform-Windows%2010%2B-lightgrey.svg)]()
[![FFmpeg](https://img.shields.io/badge/FFmpeg-Required-critical.svg)]()
[![PySide6](https://img.shields.io/badge/PySide6-GUI-orange.svg)]()

---

🎬 **Apple HEVC Batch Transcoder for Windows (GUI Edition)**  
支持 SDR/HDR 自动识别、NVENC 加速、Apple HEVC 参数优化与多线程批量转码。

</div>

---

## 🧩 功能特性 | Features

| 功能 | 描述 |
|------|------|
| 🖥️ 图形化操作 | 全 GUI，无需命令行 |
| ⚙️ 自动分析 | 自动探测分辨率、帧率、色彩空间 |
| 🌈 HDR10 支持 | 自动插入 HDR10 元数据 |
| 🎮 GPU 加速 | 自动检测并使用 NVIDIA NVENC |
| 🧠 智能参数 | 自动推算 Apple HEVC Level / Profile |
| 🚀 多线程 | 支持并行批量转码 |
| 🧪 Validator | 可选 Apple HEVC Validator 合规校验 |

---

## 🖼️ 界面预览 | Screenshots

> 📸 以下为示例界面（可在仓库 `docs/` 文件夹中添加截图）

<div align="center">
  <img src="docs/screenshot_main.png" alt="Main Window" width="80%">
  <br><br>
  <img src="docs/screenshot_progress.png" alt="Progress Example" width="80%">
</div>

---

## 🧪 Apple HEVC Validator

若系统检测到 Apple 官方 HEVC Validator，程序会自动调用进行视频合规性检测。  
支持路径：
- Windows: `C:/Program Files/Apple/AppleHEVCValidator.exe`
- macOS: `/Applications/Apple Video Tools/AppleHEVCValidator`

---

## ⚠️ 注意事项 | Notes

- 输出文件统一为 `.mp4`，不会覆盖原文件  
- HDR 元数据自动继承或生成  
- NVENC 失败自动回退 CPU 模式  
- 可自定义并发线程数与调试选项  

---

## 👨‍💻 开发者信息 | Author

| 项目 | 信息 |
|------|------|
|| 作者 | **uingei** |
|| 版本 | **v1.7.0** |
|| 许可证 | MIT License |
|| 年份 | © 2026 uingei. All rights reserved. |

---

## 🔄 Changelog

| Version | Changes |
|---------|---------|
| **v1.7.0** | Apple HEVC 合规增强：GOP 统一 2s、x265 `level` 参数修复、`repeat-headers=1`/`aud=1` 强制执行、SDR 路径添加 bt709 colorimetry、`brand` 改为 Apple 推荐的 `isomav`、修复 `validator_lock` 未定义 bug、修复 `info=` 缺少空格、修复测试文件导入错误、版本统一、测试视频修复为真正的 10-bit HDR |

## 🏷️ 鸣谢 | Credits

- [FFmpeg](https://ffmpeg.org/)
- [PySide6](https://doc.qt.io/qtforpython/)
- [NVIDIA NVENC SDK](https://developer.nvidia.com/nvidia-video-codec-sdk)
- Apple Video Tools

---

<div align="center">

**🍎 Apple HEVC 批量转码 — 简洁 · 高效 · 兼容性强**  
让每一个视频都完美适配 Apple 生态。

</div>
