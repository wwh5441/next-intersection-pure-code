# 下个，路口，见 · V4 心理声学纯代码四轨合成

本工程将 **《李宇春 - 下个，路口，见》（词曲：李宇春 / 编曲：谭伊哲，全长 3 分 30 秒）** 转化为 **100% 纯文本源码 + V4 心理声学压缩频谱参数（MDP4/LZMA + UTF-8 Base122）**，运行目录与单文件成品中**零 `.mp3` / `.wav` 录音文件、零 AI 模型权重、零外部网络请求**。

## 核心算法架构（方案一 + 方案二全集成）

1. **5 模态 Princen-Bradley 自适应长短窗切换（`2048` 长窗 / `4×512` 瞬态短窗）**：
   - 稳态乐句使用 `2048` 长窗将持续人声、贝斯、吉他谐波坍缩至高分辨率谱线，大幅减少频谱泄漏与非零系数；在咬字爆破与鼓点瞬态自动切换至 `4×512` 短窗并通过 `LONG_START` / `LONG_STOP` / `STOP_START` 桥接窗实现严格时域混叠抵消（TDAC），消除预回声。
   - 播放端采用 `N/2` 点（`1024` / `256`）复数 FFT 快速逆 MDCT 内核，实时计算开销极低。
2. **24-Bark 临界频带同时掩蔽 + 对数差分增益（`0.50 dB` 步长）**：
   - 按人耳频率分辨特性划分 24 个非均匀 Bark 临界频带，将高阶量化比特集中分配给 `150 Hz – 4.2 kHz` 人声基频与咬字共振峰，同时利用临界频带同时掩蔽（Simultaneous Masking）与绝对听阈（ATH）剔除不可闻弱噪声；频带缩放因子采用 `0.50 dB` 对数步长与跨帧一阶差分（Delta Coding）无损紧缩。
3. **高频参数立体声与 CELT 频谱折叠（PS + PNS）**：
   - 针对高频侧向混响（`harmony-side`）与高频打击擦片泛音，低中频保留精确 MDCT 谱线，高频仅存储真实子带 RMS 包络，合成端通过带确定性相位去相关的 CELT 频谱折叠还原自然立体声宽度与空气感。
4. **98.0 BPM 鼓组小节闭环原型字典 + 稀疏残差编码（Closed-Loop Drum Residual）**：
   - 自动锁定 `98.00 BPM` 录音室节拍周期，聚类提取 1 小节典型鼓组原型库并以 `63` 阶高保真编码，再用**解码后原型**做闭环预测相减提取全曲鼓组稀疏残差，兼顾高信噪比与大幅减重，且保留每一处过门与力度微差。
5. **UTF-8 Base122 无套娃单文件封装**：
   - 单文件离线版直接拼接元数据 Gzip 块与原始二进制 `.lzpar` 参数流，去除内层多余 Base64 套娃，并通过 HTML `<script>` 安全的 UTF-8 Base122 编码（膨胀率仅 `14.29%`，远低于 Base64 的 `33.33%`）。

## 启动与构建方式

1. **免服务器单文件离线版（双击直开）**：
   - 直接用 Chrome / Safari 打开 `下个路口见_纯代码合成_单文件版.html` 即可实时计算播放。
2. **双档在线对比工作台（紧凑版 / 保真版一键切换）**：
   - 双击 `启动纯代码试听.command`，或运行 `python3 server.py --port 8925 --open` 访问 `http://127.0.0.1:8925/`。
3. **重新编码、打包与完整性校验**：
   ```bash
   PYTHONDONTWRITEBYTECODE=1 /Users/weckham/Desktop/MUSIC/multitrack/.audio-env/bin/python -B tools/encode_project.py
   PYTHONDONTWRITEBYTECODE=1 python3 -B tools/build_single_html.py
   PYTHONDONTWRITEBYTECODE=1 python3 -B tools/verify_project.py
   ```
