# 下个，路口，见 · V5 AI 发音×音色解耦纯数学波码合成

本工程通过 **AI 梅尔谱正交分解 + 纯数学声波发生方程**，将 **《李宇春 - 下个，路口，见》（全长 3 分 30 秒）** 彻底去除了传统音频/MDCT 逐帧频谱切片回放，转化为**个人生理音色核常数 + 发音口型码本 + 四轨实时波函数演算代码**。

## 四轨数学波码架构

1. **主唱人声（`vocal-wave`：极简版全曲仅 `38.2 KB`）**：
   - **三维静态音色场 $\mathcal{T}_{\text{LYC}}(m, f_0, g)$**：从全曲有声帧分离出李宇春时不变 48-Mel 生理共鸣曲线、胸头声寄存器偏移向量与咬字力度倾斜向量（显式可读常数）。
   - **普通话歌唱发音码本 $\mathcal{P}(t)$**：剥离音色后在正交发音子空间构建多阶残差矢量量化（RVQ）口型码本与 8 频段清浊气声比。
   - **纯数学算波**：`AudioWorklet` 通过三角加法递推谐波振荡器 + `64` 点实倒谱最小相位（Minimum-Phase Cepstrum）+ `512` 点湍流辅音噪声整形，逐采样算出声波。
2. **鼓组与滴答节拍（`drums-wave`）**：
   - 瞬态包络驱动的底鼓指数扫频正弦模态波 + 军鼓/木鱼腔体模态波 + AI NMF 打击乐音色矩阵与 `98.00 BPM` 小节原型残差。
3. **贝斯（`bass-wave`：极简版全曲仅 `21.7 KB`）**：
   - 电贝斯拾音器静态泛音轮廓 + 琴弦非谐性刚度因子 $\sqrt{1 + B k^2}$ 连续相位波函数 + 拨弦瞬态噪声。
4. **吉他、键盘与立体声场（`harmony-wave`）**：
   - `56` 键十二平均律耦合复数旋量（Coupled-Rotor，每采样零 `Math.sin` 开销）复音和声弦阵 + 双倍采样率木吉他扫弦拨片包络 + **纯算法 `4×4` Householder 反馈延迟网络（FDN）立体声混响器**（零侧向声场数据流开销）。

## 启动与构建

1. **单文件离线版（双击直开，零依赖）**：
   - 直接打开 `下个路口见_数学波码合成_单文件版.html`
2. **双档在线对比工作台**：
   - 双击 `启动数学波码试听.command`，或运行 `python3 -B server.py --port 8926 --open`
3. **重新解耦编码与审计**：
   ```bash
   PYTHONDONTWRITEBYTECODE=1 /Users/weckham/Desktop/MUSIC/multitrack/.audio-env/bin/python -B tools/encode_project.py
   PYTHONDONTWRITEBYTECODE=1 python3 -B tools/build_single_html.py
   PYTHONDONTWRITEBYTECODE=1 python3 -B tools/verify_project.py
   ```
