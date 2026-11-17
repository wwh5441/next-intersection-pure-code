# 《李宇春 - 下个，路口，见》V9 · 基于数学的纯代码 10 轨合成引擎

本项目彻底摒弃 `LZMA1 / Base122 / Base64` 传统压缩二进制参数流思路，将整首《下个，路口，见》（`210.373s`，`48 kHz` 立体声，**5 轨纯数学人声 + 5 轨纯物理建模乐队伴奏 = 10 轨独立推子分轨**）全部转化为 **100% 人类可读的数学微分方程、傅里叶/波导声学模型与明文乐谱数组**。

## 1. 纯数学代码体积实测对比

| 维度 / 方案 | 包含内容 | 未压缩明文体积 | Gzip 传输体积 | 二进制 / LZMA 依赖 |
| :--- | :--- | ---: | ---: | :--- |
| **V7 全曲多轨 (MDCT 二进制)** | 人声 + 伴奏 | — | `1,394.13 KB` | LZMA1 + Base122 二进制帧 |
| **V8.5 事件音节波码 (仅人声)** | 仅人声 8 轨 (不含伴奏) | `158.58 KB` (裸二进制) | `51.96 KB` (LZMA1) / `75.50 KB` (含JS解码) | LZMA1 + Base122 二进制块 |
| **V9 纯数学代码：人声模型 + 285 音符乐谱** | 5 轨人声与空间数据 (`models/` + `data/vocal-score.js`) | **`39.87 KB`** (`40,828 B`) | **`14.93 KB`** (`15,288 B`) | **0 字节二进制 · 0 LZMA** |
| **V9 纯数学代码：伴奏物理 + 83 小节总谱** | 5 轨乐队伴奏数据 (`models/instrument-physics.js` + `data/band-score.js`) | **`8.50 KB`** (`8,704 B`) | **`2.94 KB`** (`3,007 B`) | **0 字节二进制 · 0 LZMA** |
| **V9 纯数学代码：全曲 10 轨模型+总谱合计** | **人声 + 伴奏全曲 10 轨全部声学数据** | **`48.37 KB`** (`49,532 B`) | **`17.94 KB`** (`18,374 B`) | **0 字节二进制 · 0 LZMA** |
| **V9 音频核心总计 (含 10 轨微分方程合成器)** | 模型 + 总谱 + 全部 DSP 求解器 (`src/dsp/*.js`) | **`105.86 KB`** (`108,402 B`) | **`35.09 KB`** (`35,935 B`) | **0 字节二进制 · 0 LZMA** |
| **V9 独立单文件离线全曲版 HTML** | 含 UI + 样式 + WebWorker + 合成器 + 10 轨总谱 | **`150.79 KB`** (`154,414 B`) | **`48.77 KB`** (`49,944 B`) | **双击即播 · 零外部依赖** |

---

## 2. 10 轨纯数学声学物理方程架构

### A. 人声与空间物理方程组 (`01` – `05`)
1. **`01` 人声 · 基音与胸腔共鸣 (`vocal_chest_f0`)**：Fant LF 声门波奇次/偶次色散相位 + 切比雪夫复相量递推（`< 720 Hz` 胸腔基音与低次谐波）。
2. **`02` 人声 · 声韵母二阶连音核 (`vocal_formant_body`)**：Glasberg-Moore ERB 心理声学频率轴上的 **12 阶余弦傅里叶腔体模态方程** $\sum_{m=1}^{12} q_m(t)\cos(m\pi u_{\text{ERB}}(f))$，由 **二阶临界阻尼跨字协同发音微分方程** $\ddot{\mathbf{q}} + 2\omega\dot{\mathbf{q}} + \omega^2\mathbf{q} = \omega^2\mathbf{q}_{\text{target}}$ 实时解析积分驱动，实现前字韵尾向后字声母舌位的自然 `--` 预弯与连音。
3. **`03` 人声 · 辅音瞬态与真气声 (`vocal_consonant_air`)**：24 组普通话声母双高斯湍流包络 + VOT 嗓音起始 S 型包络 + Klatt 开相气声。
4. **`04` 人声 · 副歌双声道真和声 (`vocal_chorus_harmony`)**：副歌（*DI DI DA / 下个路口再见吧*）`±6.5 cents` 真双轨微失谐与 `14ms / 23ms` Haas 立体声重唱。
5. **`05` 空间 · 早期反射与 FDN 混响 (`studio_fdn_space`)**：4 抽头早期反射 + 8 线互质素数延迟 Householder 正交矩阵反馈网络。

### B. 乐队物理建模伴奏组 (`06` – `10`，精确锁定 `98.000 BPM`)
6. **`06` 鼓组 · 膜振底鼓与军鼓拍手 (`drum_kick_snare`)**：指数瞬态扫频圆膜微分方程底鼓 + 双贝塞尔模态腔体军鼓 + 4 抽头微延迟立体声拍手。
7. **`07` 鼓组 · 非公度金属镲与沙锤 (`drum_hihat_perc`)**：6 路非公度高频方波金属环调矩阵 + 真实踩镲开闭截断（Open-to-Closed Pedal Choke）+ 16 分音符摇摆沙锤。
8. **`08` 贝斯 · 指弹电贝斯包络滤波 (`bass_groove`)**：拾音器位置空间梳状滤波 $\sin(\pi h x_p)$ + 二阶 Chamberlin 状态变量包络低通滤波器 + 底鼓侧链联动让位。
9. **`09` 吉他 · 波导拨弦放克切分 (`guitar_funk_riff`)**：4 弦独立 Extended Karplus-Strong 数字波导拨弦方程（含右手闷音切分 `Palm-Mute Chuck` 与清音电子管微饱和）。
10. **`10` 键盘 · FM 电钢与合唱铺底 (`keys_rhodes_pad`)**：双算子 DX FM 电钢琴（`14:1` 金属音条泛音比）+ 三路微失谐锯齿波立体声温暖 Pad（覆盖 `Dm9 - Bbmaj7 - Cadd9 - Am7` 循环与 `Bar 80` 尾奏 `D9` 皮卡第大和弦）。

---

## 3. 模块化工程目录结构

- `index.html` + `server.py`：零内联样式入口与本地 ES Module 零缓存服务（默认端口 `8945`）。
- `styles/`：`tokens.css`、`layout.css`、`components/{transport,visualizer,mixer,formula-panel}.css`。
- `models/`：`singer-physics.js`、`mandarin-phonetics.js`、`instrument-physics.js`（纯明文物理常数与普通话声韵母腔体向量）。
- `data/`：`vocal-score.js`（285 颗人声音符明文数组）、`band-score.js`（83 小节伴奏明文总谱）、`code-metrics.json`。
- `src/dsp/`：`fft-minphase.js`、`math-vocal-synth.js`、`math-drums-synth.js`、`math-bass-guitar-synth.js`、`math-keys-space-synth.js`、`song-renderer.js`、`math-worker.js`。
- `src/{state,modules,components}/`：响应式状态仓库、10 轨同步 WebAudio 混音器与五大解耦视图组件。
- `tools/`：`build_single_html.py`（独立单页打包与字节统计）、`verify_pure_math.py`（零二进制/零 LZMA 自动化审计）。
- `下个路口见_纯数学代码全曲版.html`：独立离线纯文本单文件全曲版。
