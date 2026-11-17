# 《下个，路口，见》V9R1 纯代码双字典物理合成引擎

> **重锚起点**：严格回退至 `V8.5`（事件音节波码）结束、初版 `V9` 刚起步时的干净原点重新设计。  
> **核心铁律**：**零 LZMA · 零 Base122 · 零 Base64 · 零盲目 MDCT 混音波形帧 · 零二进制采样块** —— 100% 人类可读 ES Module JavaScript（**可执行乐谱 + 双重物理原型字典 + 10 轨声学微分方程**）。

---

## 1. 为什么旧 V9 走偏了，而 V9R1 走通了？

- **旧版 V9 的两处结构性失误**：
  1. `V9.2` 虽然实现了纯数学 10 轨合成，但仅对人声做了 97 音节字典化，**伴奏 5 轨（鼓/贝斯/吉他/电钢）仍是 86 小节逐帧平铺**，未能兑现“纯代码结构化复用更精简”的初衷（Gzip 体积 `174.39 KB`）。
  2. `V9.3 / V9.4` 为了强追波形信噪比，抛弃了物理乐器与声学结构，退化成了“逐帧硬背混音 MDCT 频谱表”的简陋纯文本音频编码器——既丧失了可编辑乐谱与音色解耦，体积又急剧膨胀。
- **V9R1 的正统纯代码双字典解法**：
  1. **人声声韵字典 (`models/syllable-atlas.js` + `models/singer-core.js`)**：提取全曲 `97` 个普通话声韵原型（声母 `3` 锚点 + 韵母 `5` 锚点 × `40` 阶 ERB 声道模态，经无损零行裁剪 `0.00 dB` 损耗）与李宇春 `24×40` 音色特征核，配合 `--` 预备协同发音桥、Hilbert 最小相位辅音簇与梳状陷波气声。
  2. **伴奏小节物理原型字典 (`models/band-motif-atlas.js` + `models/instrument-physics.js`)**：锁定全曲零漂移小节网格（`1 Bar = 2.450000s`，`97.9592 BPM`），通过 **PAM 感知中心点（Perceptual Medoid）** 从真实母带参数中遴选 `18` 个鼓组律动原型 + `22` 个指弹滑音贝斯原型 + `22` 个吉他/电钢复调原型（拒绝均值模糊瞬态），在**连续物理参数域**完成 `86 小节原型调度 + 稀疏过门残差` 展开后再送入连续相位模态振子/刚度弦振方程积分，实现 **86 个小节边界 100% 相位连续、零咔哒声**。

---

## 2. 实测体积与真实度对比（全曲 `210.37s` · `48kHz` · `10` 轨立体声）

| 架构版本 | 编码/生成范式 | 人声体积 | 伴奏体积 | 全曲总数据量 | 相对母带压缩比 |
| :--- | :--- | ---: | ---: | ---: | ---: |
| 原始立体声母带 | `48kHz 16-bit PCM WAV` | — | — | `40.39 MB` | `1.0 : 1` |
| `V8.5 + V8伴奏字典` (旧锚点) | 事件音节波码 + 波形残差 (`LZMA`) | `94.75 KB` | `127.76 KB` | `222.51 KB` (LZMA) | `185.9 : 1` |
| 旧 `V9.2` (未字典化伴奏) | 97音节字典 + **平铺伴奏帧** (`Gzip`) | `75.50 KB` | `98.89 KB` | `174.39 KB` (Gzip) | `237.2 : 1` |
| **`V9R1 Ultra` (极简双字典档)** | **97音节字典 + 62伴奏原型 + 6小节过门残差** | **`74.60 KB`** | **`40.00 KB`** | **`114.41 KB` (Gzip)** | **`361.5 : 1`** |
| **`V9R1 Expressive` (精修双字典档)** | **97音节字典 + 24维全PCA + 全曲微变奏残差** | **`163.88 KB`** | **`112.37 KB`** | **`276.15 KB` (Gzip)** | **`149.8 : 1`** |

- **母带物理轨迹相关系数 (`r`)**：
  - `Ultra` 极简档：底鼓瞬态 `r = 0.9815`，贝斯真滑音音高 `r = 0.9686`，贝斯包络 `r = 0.9641`，复调和声 `r = 0.9136`。
  - `Expressive` 精修档：底鼓 `r = 0.9993`，军鼓/拍手 `r = 0.9998`，贝斯音高 `r = 0.9998`，复调和声 `r = 0.9836`。
- **合成速度与低内存设计**：纯 JS 单核耗时 `~4.9s`（实时播放速度的 **42 倍**）；单声道干声/贝斯（`01, 02, 03, 08`）采用原生 `1ch AudioBuffer`，立体声轨（`04, 05, 06, 07, 09, 10`）采用 `2ch AudioBuffer`，节省 `20%` WebAudio 内存。

---

## 3. 解耦工程目录结构

```text
下个路口见_V9R1_纯代码字典合成引擎/
├── index.html                          # 零内联 <style> 纯净入口
├── server.py                           # 本地零缓存 HTTP 工作台服务 (默认端口 8947)
├── styles/                             # 严格三级样式解耦 (Tokens / Layout / Components)
│   ├── tokens.css
│   ├── layout.css
│   └── components/
│       ├── transport.css
│       ├── visualizer.css
│       ├── mixer.css
│       └── inspector.css
├── models/                             # 跨档共享双重物理字典与声学常数
│   ├── singer-core.js                  # 李宇春 40阶ERB均值 + 24维PCA基底 + 8线素数FDN
│   ├── syllable-atlas.js               # 97个普通话音节 3+5 声韵协同发音字典
│   ├── instrument-physics.js           # 56音符耦合旋子频率表 + 64原型打击乐高频码本
│   └── band-motif-atlas.js             # 86小节 PAM 中心点字典 (18鼓 + 22贝斯 + 22复调)
├── data/                               # 可执行乐谱、编曲调度表与基准指标
│   ├── song-structure.js               # 97.9592 BPM 曲式段落与和声进行
│   ├── vocal-lyrics.js                 # 34句同步歌词时间轴
│   ├── vocal-score-ultra.js            # Ultra 285字可执行人声乐谱 (10维PCA)
│   ├── band-arrangement-ultra.js       # Ultra 86小节原型调度表 + 6小节关键过门残差
│   ├── vocal-score-expressive.js       # Expressive 285字精修人声乐谱 (24维PCA)
│   ├── band-arrangement-expressive.js  # Expressive 86小节原型调度表 + 稀疏微变奏残差
│   ├── tiers-registry.js               # 双档注册表
│   ├── benchmarks.js                   # 范式对比与物理微分方程字典
│   └── code-metrics.json               # 自动审计生成的精确字节与声学验证报告
├── src/
│   ├── dsp/                            # 纯数学声学合成内核与 Web Worker
│   │   ├── fft-minphase.js             # Radix-2 FFT 与 Hilbert 最小相位重建
│   │   ├── motif-expander.js           # 参数域小节原型 + 稀疏残差零漂移展开器
│   │   ├── vocal-coart-synth.js        # 4轨声学人声合成器 (`--`连音 + 梳状气声 + 双声道和声)
│   │   ├── drums-modal-synth.js        # 扫频底鼓模态ODE + 木鱼拍手 + 64原型高频镲
│   │   ├── bass-guitar-keys-synth.js   # 18阶色散电贝斯 + Funk吉他Hook/80fps扫弦 + 电钢
│   │   ├── space-fdn-synth.js          # 22ms预延迟早期反射 + 8线素数 Householder FDN
│   │   ├── v9r1-renderer.js            # 10轨母线校准与零超调平滑限幅渲染器
│   │   └── v9r1-worker.js              # 零拷贝 Transferable ArrayBuffer 后台合成线程
│   ├── state/studio-store.js           # 响应式状态仓库
│   ├── modules/audio-mixer.js          # 10轨 WebAudio 混音与双档内存秒切控制器
│   ├── components/                     # 5个职责单一视图组件
│   │   ├── header-metrics.js
│   │   ├── transport-bar.js
│   │   ├── score-motif-canvas.js
│   │   ├── stem-mixer.js
│   │   └── dictionary-inspector.js
│   └── main.js
└── tools/
    ├── compile_v9r1_dictionary_engine.py
    ├── build_single_html.py
    └── verify_v9r1_pure_math.py
```

---

## 4. 启动与审计命令

```bash
# 启动在线工作台 (http://127.0.0.1:8947/)
python3 server.py --port 8947

# 重新生成独立离线单文件 HTML 并运行架构纯洁度 + 声学包络相关性审计
/Users/weckham/Desktop/MUSIC/multitrack/.audio-env/bin/python tools/build_single_html.py
/Users/weckham/Desktop/MUSIC/multitrack/.audio-env/bin/python tools/verify_v9r1_pure_math.py
```
