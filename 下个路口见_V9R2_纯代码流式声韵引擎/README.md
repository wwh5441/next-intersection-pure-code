# 《下个，路口，见》V9R2 · 纯代码最高音质引擎

**0 波形帧 · 0 变换残差 · 0 二进制**。原曲只用于分析和标定，产物里只有发声模型参数和乐谱参数。

## 组成

| 部分 | 文件 | Gzip | 内容 |
| :--- | :--- | ---: | :--- |
| 李宇春音色核 | `models/singer-core.js` | 4.2 KB | 48 阶 ERB 声道均值 + 36 维音色基 + 声门相位 |
| 97 音节声韵字典 | `models/syllable-atlas.js` | 36.5 KB | 每音节 3 个声母锚点 + 5 个韵母锚点 |
| 空间混响 | `models/space-fdn.js` | 1.7 KB | 早期反射 + 8 线 FDN 参数 |
| 乐器物理常数 | `models/instrument-physics.js` | 5.2 KB | 弦振、鼓膜、旋子常数（来自 V9.2） |
| 人声事件 | `data/vocal-events-v85.js` | 844 KB | 286 个音节事件：音高、力度、清浊度、声道微表情 |
| 伴奏乐谱 | `data/band-score-expressive.js` | 120 KB | V9.2 精修档参数曲线 |
| 伴奏响度 | `data/band-stem-gains.js` | 0.1 KB | 3 个标量 |
| **合计** | | **1011.8 KB** | 明文 3.27 MB |

## 实测（Log-Mel SNR，对比原曲 48kHz 分轨）

| 主唱 | 完整人声 | 贝斯 | 鼓 | 吉他/电钢 | 全曲 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 19.44 | 16.45 | 21.94 | 17.85 | 15.91 | 13.31 |

## 命令

```bash
python3 server.py --port 8952
/Users/weckham/Desktop/MUSIC/multitrack/.audio-env/bin/python tools/compile_v9r2_from_master.py
/Users/weckham/Desktop/MUSIC/multitrack/.audio-env/bin/python tools/build_single_html.py
/Users/weckham/Desktop/MUSIC/multitrack/.audio-env/bin/python tools/verify_v9r2_pure_audit.py
```
