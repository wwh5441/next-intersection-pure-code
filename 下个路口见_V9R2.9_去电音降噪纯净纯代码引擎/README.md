# 《下个，路口，见》V9R2.1 · 纯代码最高音质引擎

V9R2 的独立续作。从 MP3 重新走完整路线，每一步取最高精度。产物只有发声模型参数和音符/事件参数：**0 波形帧、0 变换残差、0 二进制**。不使用 V9 / V9R1 / V9.2 的任何资产。

## 流程

1. **分轨（仅作分析材料）** — `services/hq_stem_separator.py`：htdemucs 与 htdemucs_6s 各做 8 次平移集成，48kHz 浮点，各轨相加 ≡ 原曲（145 dB）。
2. **节拍** — 底鼓起音拟合得 98.001 BPM，局部偏差 ≤ 11 ms；所有事件按实际帧位置记录。
3. **人声 01–08** — V8.5 方法全精度重做：97 音节字典 + 285 事件，逐帧音高/响度/48 阶声道微表情；气声噪声按原唱谷值重新标定（+10 dB）；伴唱改为 M/S 双声部、全频带谐波；空间送混含齿音。
4. **贝斯 09** — `services/bass_harmonic_score.py`：pYIN 音高 + 40 谐波包络（12 维 PCA）→ 连续相位振荡器组。
5. **鼓 10** — `services/drum_event_score.py`：低/中/高三族击打事件 → 聚类模板 → 卷积 NMF 联合标定模板与力度 → 形状噪声合成，含侧声道宽度。
6. **吉他/电钢 11** — `services/poly_note_score.py`：CQT 上 61 音谐波约束 NMF → 复调音符（音高、起止、响度曲线、声像）+ 每音学习谐波轮廓 → 谐波振荡器合成。

## 实测（Log-Mel SNR，对比本版高质量分轨；同一参考下重测 V9.2 / V9R2 作对照）

| | 主唱 | 完整人声 | 贝斯 | 鼓 | 吉他/电钢 | 全曲 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **V9R2.1** | **20.85** | **18.02** | **32.48** | **18.95** | **19.02** | **15.20** |
| V9R2（纯代码版） | 19.53 | 16.54 | — | — | — | 13.31* |
| V9.2 精修档伴奏 | — | — | 20.75 | 17.90 | 15.88 | — |

\* V9R2 全曲值为旧参考下测得。鼓和吉他/电钢按与原曲响度一致来设定（鼓 −2.4 dB，吉他 RMS 匹配），不是单轨指标最优值，以保证混音平衡。

数据总量 **1967.7 KB Gzip**（明文 7.2 MB）。Node 全曲合成约 16 s。

## 命令

```bash
python3 server.py --port 8953
/Users/weckham/Desktop/MUSIC/multitrack/.audio-env/bin/python tools/compile_v9r21.py
/Users/weckham/Desktop/MUSIC/multitrack/.audio-env/bin/python tools/build_single_html.py
/Users/weckham/Desktop/MUSIC/multitrack/.audio-env/bin/python tools/verify_v9r21_pure_audit.py
```
