# CabronWEC (CWC-5X) · 纯代码流式声学合成工作台 & `.wec` 极小容器套件

100% 纯手写 JavaScript 声学编码、`AudioWorklet` 实时流式合成引擎与无损 `"CWC5"` `.wec` 二进制容器套件（零第三方编解码库、零 WASM），将《下个，路口，见 · 李宇春》（`210.373s`，`44.1 kHz` 立体声）编译为三档 **纯 JS 版本** 与 **9R2 式 `.wec` 容器 + `wec-play.mjs` 独立执行程序**：

## 1. `.wec` 二进制容器 + `wec-play.mjs` 独立播放器（零画质/音质变化，逐样本 `0.000000` 误差）

独立零依赖执行程序 `cabron-wec/bundles/wec-play.mjs`：**`35.5 KB` Raw (`35,540 B`) / `10.0 KB` Gzip (`10,027 B`)**。

| 档位 | `.wec` 容器体积 | 对比单文件 `.js` 缩减 | IEEE CRC-32 | 声学与帧间熵编码架构 |
| :--- | ---: | ---: | :---: | :--- |
| **`next-intersection-500k.wec`** | **`328.2 KB`** (`320.5 KiB` / `328,207 B`) | **`-30.9%`** (vs `475.2 KB`) | `0xc82126ff` | `CWC5` 32B 头 + 2D 频带包络预测 + 48 上下文帧间谐波直编 |
| **`next-intersection-1m.wec`** | **`735.1 KB`** (`717.9 KiB` / `735,120 B`) | **`-24.6%`** (vs `974.9 KB`) | `0x2c7306dd` | `CWC5` 32B 头 + 2D 频带包络预测 + 48 上下文帧间谐波直编 |
| **`next-intersection-hifi.wec`** | **`1.98 MB`** (`1.89 MiB` / `1,978,301 B`) | **`-19.5%`** (vs `2.46 MB`) | `0xd1405459` | `CWC5` 32B 头 + 2D 频带包络预测 + 16 上下文帧间二维游程编码 |

## 2. 纯 JS 自包含代码版（零二进制资源）

| 版本 | 8 分块纯 JS 源码体积 (Raw) | Gzip 传输体积 | 含合成器的单文件独立 `.js` (Raw) | 核心声学架构 |
| :--- | ---: | ---: | ---: | :--- |
| **CWC-500K (`< 500 KB`)** | `450.0 KB` (`450,009 B`) | `370.1 KB` (`370,136 B`) | **`475.2 KB`** (`475,248 B`, `cabron-wec/bundles/next-intersection-500k.js`) | `0–4.4 kHz` 稀疏谐波 + `4.4–12 kHz` Harmonic SBR + Velvet PNS + M/S 耦合 |
| **CWC-1M (`< 1 MB`)** | `949.7 KB` (`949,698 B`) | `781.2 KB` (`781,151 B`) | **`974.9 KB`** (`974,927 B`, `cabron-wec/bundles/next-intersection-1m.js`) | `0–6.4 kHz` 稀疏谐波 + `6.4–12 kHz` Harmonic SBR + Velvet PNS + M/S 耦合 |
| **CWC-HiFi (同音质参考档)** | `2.46 MB` (`2,457,981 B`, `-65.2%` vs 4X) | `2.02 MB` (`2,021,899 B`, `-53.9%` vs 4X) | 分块流式加载 | `0–12 kHz` 全频稀疏 MDCT + 齿音包络守恒（时域 `15.76/16.16 dB`、Bark `32.39 dB` 超越 `CWC-4X`） |

## 3. CLI 与开发命令

```bash
# 查看 .wec 容器头、CRC-32 与声学参数
node cabron-wec/bundles/wec-play.mjs inspect cabron-wec/bundles/next-intersection-500k.wec

# 校验 CRC-32 与全曲帧完整性
node cabron-wec/bundles/wec-play.mjs verify cabron-wec/bundles/next-intersection-500k.wec

# 离线极速渲染 44.1 kHz 16-bit 立体声 WAV（支持 --solo=0..4 / --mute=0..4 分轨导出）
node cabron-wec/bundles/wec-play.mjs render cabron-wec/bundles/next-intersection-500k.wec out.wav

# 重新打包全部 .wec 容器与 wec-play.mjs
npm run compile:wec

# 运行数学闭环、逐样本 0 误差校验与体积门禁单元测试
npm test

# 启动本地 3000 端口实时工作台（支持一键切换 .wec / 拖拽加载任意 .wec 文件）
npm start
```
