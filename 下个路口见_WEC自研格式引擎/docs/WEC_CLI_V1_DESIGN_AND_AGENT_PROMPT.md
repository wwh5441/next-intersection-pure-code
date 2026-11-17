# `WEC-CLI v1.0` 架构设计蓝图与独立智能体交付任务书

> **关联基准规范与真值目录**：
> - 二进制单一事实源（Schema）：[`schemas/wec-spec-v2.json`](file:///Users/weckham/Desktop/MUSIC/下个路口见_WEC自研格式引擎/schemas/wec-spec-v2.json)
> - Python 参考编译器与校验器：[`tools/wec_opus_compiler.py`](file:///Users/weckham/Desktop/MUSIC/下个路口见_WEC自研格式引擎/tools/wec_opus_compiler.py)
> - 浏览器端二进制解析器：[`parsers/wec-binary-parser.js`](file:///Users/weckham/Desktop/MUSIC/下个路口见_WEC自研格式引擎/parsers/wec-binary-parser.js)
> - 7 款三档实测基准包（测试真值集）：[`wec_files/`](file:///Users/weckham/Desktop/MUSIC/下个路口见_WEC自研格式引擎/wec_files)
> - 本地 CLI 架构参考母本：[`dsh_tools`](file:///Users/weckham/Desktop/Tools/dsh_tools/README.md) · [`hermes_tools`](file:///Users/weckham/Desktop/Tools/hermes_tools/README.md) · [`hermes-agent`](file:///Users/weckham/Desktop/Tools/hermes-agent)

---

## 一、`WEC v2.0` 三档分层与 Opus 融合总纲（跨端 100% 互通契约）

1. **绝不重复造波形编码轮子**：
   - 所有音频轨道统一采用工业界标准 **`RFC 6716 Opus`（`SILK` 声道线性预测 + `Hybrid` 混合带 + `CELT` 金字塔矢量量化与区间编码）**，`48,000 Hz` 采样率，默认 `20 ms`（`960` 采样点）帧长。
   - 容器内剔除冗余的 `OggS` 27B+ 分页头，直接存储 **裸 Opus 数据包 + `u8_ext` 紧凑包长表**（`<255` 占 `1 Byte`；`>=255` 为 `0xFF + uint16_le`），同时记录精确 `preSkip`（如 `312` samples）与 `Q7.8 dB` 增益校准，确保多轨合并 **`0 sample` 零相位漂移**。
2. **`WEC` 三档严格边界**：
   - **`WEC0`（纯最小单流核，Magic `"WEC0"`，`32-Byte` 极简文件头）**：固定 `1` 条 Opus 音轨、`0` 特征层、零 JSON 元数据税，自带 `O(1)` 随机跳播索引与整包 `CRC-32`。
   - **`WEC1`（音色+伴奏双轨档，Magic `"WEC1"`，`48-Byte` 文件头）**：允许 `1..2` 条相位锁定音轨（标准配置：`Track 0 = 人声/主音色干声轨 [独立激活 SILK/Hybrid]` + `Track 1 = 纯伴奏立体声轨 [CELT]`），配套内嵌 `TIMB`（50Hz 帧对齐 $F_0$ 基频 + 清浊音 + 四段共振峰音色包络层）。
   - **`WEC2`（全分轨可扩展母版档，Magic `"WEC2"`，`48-Byte` 文件头）**：**硬性要求 `numAudioTracks >= 4` 且 `numFeatureLayers >= 3`，向上不设上限（`uint16` 最大支持 `65,535` 轨/层）**。基线含 4+ 独立分轨（如 `Vocals / Drums / Bass / Other` 或 `6` 轨/`12` 轨母版）+ 3+ 结构化 TLV 特征层（`TIMB` 声纹层 + `GRID` 小节拍号和弦层 + `LYRC` 逐字音节事件层 + 可选 `SPAT` / `MIDI` / 自定义 `FourCC` 扩展层）。

---

## 二、`WEC-CLI v1.0` 工程架构设计（融合 `Hermes` + `DSH` 架构精髓）

参考本机 [`dsh_tools`](file:///Users/weckham/Desktop/Tools/dsh_tools) 的 **Provider/Profile 配置驱动 + 严格分层解耦** 与 [`hermes-agent`](file:///Users/weckham/Desktop/Tools/hermes-agent) 的 **Headless 单次执行 + 交互式 `/slash` REPL Shell + Web Sidecar 联动**，设计如下零单文件堆砌架构：

```text
wec                             # 顶层可执行入口 (chmod +x, 软链至 ~/.local/bin/wec)
wec_cli/
├── __init__.py                 # 版本号与导出符号
├── config.py                   # 路径解析 (~/.wec 状态库、FFmpeg/libopus 探针、.audio-env 探针)
├── cli.py                      # 顶层 argparse 分发器 (支持子命令直跑 & 无参自动进入 REPL)
├── repl.py                     # Hermes 风格交互式 Shell (prompt_toolkit/readline + /slash 命令补全)
├── data/                       # 纯静态业务字典 (严禁硬编码进逻辑层)
│   ├── profiles_catalog.json   # 预设编码档位 (wec0-12k, wec0-32k, wec0-96k, wec1-ktv, wec2-4x3, wec2-pro)
│   ├── roles_registry.json     # 音轨 RoleID 与特征层 FourCC 标准字典 (同步 wec-spec-v2.json)
│   └── backends_catalog.json   # 分轨与特征后端配置 (ffmpeg-libopus, cached-npz, stem-bar-demucs)
├── models/                     # 类型化契约层
│   └── schemas.py              # WecHeaderV2, TrackSpec, FeatureSpec, TimbFrameSummary, ToolchainHealth
├── parsers/                    # 二进制容器与协议编解码层
│   ├── wec_container.py        # WEC0(32B) / WEC1 & WEC2(48B) 打包与解包、u8_ext 索引、IEEE CRC-32
│   ├── ogg_opus_bridge.py      # OggS 页面解析提取裸包 <-> 逆向无损重封装标准 .opus 文件
│   └── timb_binary.py          # TIM1 32B 头 + N×8B 50Hz 声纹帧读写器
├── services/                   # 核心业务服务层
│   ├── toolchain_doctor.py     # 对标 `dsh status`：检测 libopus、Python .audio-env、端口健康度
│   ├── opus_encoder.py         # 多核并行调用 libopus (-1.6dB 防削波余量 + Q7.8 增益恢复)
│   ├── feature_extractor.py    # 向量化 50Hz TIMB 声纹基频共振峰提取 + LRC/Score 转 LYRC/GRID
│   ├── tier_transformer.py     # 跨档折叠与提取引擎 (WEC2 ➔ WEC1 ➔ WEC0, KTV/Acapella 渲染)
│   └── studio_runner.py        # 对标 `dsh web` / `hermes web`：守护或拉起 Web 多轨混音台 (:8946)
└── tests/                      # 自动化回归测试套件
    └── test_interoperability.py# 100% 校验 wec_files/ 下全部 7 款真值包并验证重打包字节一致性
```

### 核心命令集设计（CLI UX）

| 命令形式 | 设计对标 | 核心职责与示例 |
| :--- | :--- | :--- |
| `wec status` | `dsh status` / `hermes status` | 巡检 `libopus` 编码器路径、Apple Silicon MPS 分轨环境、`~/.wec` 默认档位与 Web 工作台在线状态 |
| `wec encode` | `ffmpeg` + `dsh profile` | 编译音频/多轨生成 `.wec`：<br>`wec encode song.wav --tier wec0 --kbps 32 -o out.wec`<br>`wec encode --tier wec1 --vocal v.wav --accomp inst.wav --auto-timb -o dual.wec`<br>`wec encode --tier wec2 --track lead:voc.wav --track drums:d.wav --track bass:b.wav --track other:o.wav --grid grid.json --lyrc song.lrc -o master.wec`（支持任意多个 `--track` 与 `--layer` 无上限追加！） |
| `wec inspect <file.wec>` | `ffprobe` | **免音频解码毫秒级透视**：打印 Magic、CRC-32、各轨 `SILK / Hybrid / CELT` 帧比例、`TIMB` 平均基频/共振峰分布、`GRID` 小节数（支持 `--json` 与 `--packet 3080`） |
| `wec verify <target>` | `flac -t` | 校验单个或整个目录 `.wec` 的 CRC-32、`u8_ext` 包长闭环与 `WEC2 >= 4+3` 档位契约 |
| `wec fold <in.wec>` | 渐进降档流转 | **跨档降维折叠**：<br>`wec fold master_wec2.wec --to wec1 -o ktv_wec1.wec`（保留人声轨与 `TIMB`，其余器乐轨锁相混缩为伴奏轨）<br>`wec fold master_wec2.wec --to wec0 --kbps 24 -o mini_wec0.wec` |
| `wec extract <in.wec>` | 轨道与特征抽离 | **无损抽离单轨 `.opus`、混音导出 `.wav` 或直取特征层**：<br>`wec extract in.wec --scene ktv -o song_ktv_48k24b.wav`<br>`wec extract in.wec --track 0 --raw-opus -o vocal_lossless_repack.opus`（零重编码直接无损回封 `.opus`！）<br>`wec extract in.wec --layer TIMB -o vocal_f0_formants.json` |
| `wec web [file.wec]` | `dsh web` / `hermes web` | 启动或连接 `http://127.0.0.1:8946/` Web 多轨工作台并自动载入目标 `.wec` |
| `wec` / `wec repl` | `hermes` REPL | 进入交互式会话模式，支持 `/load`、`/info`、`/stems`、`/solo 0`、`/mute 0`、`/timb 61.5s`、`/fold wec1`、`/export` |

---

## 三、直接发给独立智能体的完整任务提示词（Copy-Paste Prompt）

你可以将下方代码块中的指令**直接全选复制发送给独立的智能体**，里面已锁死二进制字节规范、参考代码路径、本机环境路径与验收红线，确保它做出来的 `WEC-CLI v1` 与当前 Web 工作台 **100% 字节级双向互通**：

```markdown
请为我开发实现第一版模块化命令行工具链 **`WEC-CLI v1.0` (`wec`)**，放置在 `/Users/weckham/Desktop/MUSIC/wec_cli_project/`（顶层提供可执行入口脚本 `./wec` + 模块包 `wec_cli/`）。

### 1. 必读前置基准文件（动手写代码前先读取，严禁臆造二进制偏移！）
1. **WEC v2.0 二进制格式唯一事实源（Schema）**：
   `/Users/weckham/Desktop/MUSIC/下个路口见_WEC自研格式引擎/schemas/wec-spec-v2.json`
2. **WEC-CLI v1.0 详细架构设计书**：
   `/Users/weckham/Desktop/MUSIC/下个路口见_WEC自研格式引擎/docs/WEC_CLI_V1_DESIGN_AND_AGENT_PROMPT.md`
3. **Python 参考编译器与校验实现（必须保持 100% 字节级兼容）**：
   `/Users/weckham/Desktop/MUSIC/下个路口见_WEC自研格式引擎/tools/wec_opus_compiler.py`
4. **7 款已生成的 WEC0 / WEC1 / WEC2 真值测试包（用于双向互通回归测试）**：
   `/Users/weckham/Desktop/MUSIC/下个路口见_WEC自研格式引擎/wec_files/`
5. **本机 CLI 工程风格参考母本**：
   - `/Users/weckham/Desktop/Tools/dsh` 与 `/Users/weckham/Desktop/Tools/dsh_tools/`
   - `/Users/weckham/Desktop/Tools/hermes` 与 `/Users/weckham/Desktop/Tools/hermes_tools/`

### 2. 硬性工程与架构纪律
1. **严格模块化拆分（严禁单文件摊大饼）**：
   必须按 `config.py`、`data/*.json`（纯静态字典与档位配置）、`models/`（类型化 dataclass 契约）、`parsers/`（`WEC0` 32B 头 / `WEC1 & WEC2` 48B 头读写、`u8_ext` 包长表、IEEE 802.3 CRC-32、Ogg-Opus 无损拆封与回封、`TIM1` 8B 声纹帧编解码）、`services/`（工具链巡检、`libopus` 多线程编码、向量化 `TIMB` 提取、跨档降维折叠、特征与分轨导出、Web 工作台拉起）、`cli.py` 与 `repl.py`（Hermes 风格 `/slash` 交互式 REPL）分层解耦。
2. **结合 Opus，绝不重复造波形编码轮子**：
   - 本机可用带 `libopus` 的 FFmpeg 路径优先自动探测（包含 `/Users/weckham/Library/Python/3.9/lib/python/site-packages/imageio_ffmpeg/binaries/ffmpeg-macos-aarch64-v7.1` 及系统 `PATH` / `imageio_ffmpeg`）。
   - 支持**零重编码无损回封（Lossless Ogg Re-mux）**：`wec extract in.wec --track <i|role> --raw-opus -o out.opus` 必须直接把 `.wec` 中的 RFC 6716 Opus 裸包与 `preSkip` 重新打入标准 Ogg-Opus 页面，不经过二次有损重编码！
3. **严格执行 WEC 三档契约**：
   - `WEC0`（Magic `"WEC0"`，32B 头）：纯最小核，仅 1 条音轨 + 0 特征层。
   - `WEC1`（Magic `"WEC1"`，48B 头）：允许 1~2 条音轨（人声音色轨 + 伴奏轨）+ 可选特征层（默认生成 50Hz `TIMB` 声纹基频共振峰层）。
   - `WEC2`（Magic `"WEC2"`，48B 头）：**强制要求 `numAudioTracks >= 4` 且 `numFeatureLayers >= 3`，向上不设上限**（支持通过重复传参 `--track role:label:path[:kbps]` 与 `--layer FOURCC:path` 封装 4、6、12、24+ 任意数量轨道与自定义特征层）。

### 3. 必须完整跑通的子命令与交互模式
1. `./wec status`：展示版本、`libopus` 探针状态、`.audio-env` 分轨环境状态、默认 Profile 与 Web Studio (`127.0.0.1:8946`) 在线状态。
2. `./wec inspect <file.wec> [--packet N] [--json]`：免解码毫秒级输出容器头、CRC-32、音轨表（统计各轨 `SILK-only` / `Hybrid` / `CELT-only` 帧占比！）、特征层表及 `TIMB` 基频/共振峰摘要。
3. `./wec verify <file_or_dir>`：校验全部 7 款现有 `/Users/weckham/Desktop/MUSIC/下个路口见_WEC自研格式引擎/wec_files/*.wec` 100% 通过，并对篡改字节或违反 `WEC2 >= 4+3` 的文件准确报错拦截。
4. `./wec encode`：支持从 WAV/MP3/FLAC 或多轨文件编译生成 `WEC0`、`WEC1`、`WEC2` 文件，内置 `-1.6 dB` 防削波余量与 `Q7.8` 增益自动写回。
5. `./wec fold <in.wec> --to wec1|wec0 -o <out.wec>`：实现 `WEC2 -> WEC1 -> WEC0` 锁相降档折叠。
6. `./wec extract <in.wec>`：支持 `--raw-opus` 零损抽离任意单轨 `.opus`、支持 `--scene ktv|acapella|master` 或 `--solo / --mute` 渲染导出 `48kHz/24-bit WAV`、支持 `--layer TIMB|GRID|LYRC` 秒级导出 JSON。
7. `./wec web [file.wec]`：检查或拉起 `/Users/weckham/Desktop/MUSIC/下个路口见_WEC自研格式引擎/server.py` (`:8946`)。
8. `./wec`（无参或 `./wec repl`）：进入交互式 REPL Shell，支持 `/load`、`/info`、`/stems`、`/timb`、`/fold`、`/extract`、`/help`、`/quit`。

### 4. 验收要求
编写完整的自动化测试脚本 `tests/test_wec_cli.py` 并实跑通过：
1. 对 `/Users/weckham/Desktop/MUSIC/下个路口见_WEC自研格式引擎/wec_files/` 下全部 7 个 `.wec` 文件执行 `verify` 与 `inspect` 全部 0 报错通过；
2. 用 `wec fold` 将 `WEC2` 标准 4+3 包降档折叠为 `WEC1` 与 `WEC0`，再用 Web 工作台的 JS 解析器（`node` 调用 `parsers/wec-binary-parser.js`）验证生成的 `.wec` 文件 100% 兼容可读；
3. 验证 `--raw-opus` 无损抽离出的 `.opus` 可直接被 `ffmpeg` 正常解码且采样点数严格对齐。
（注意：请勿修改 `/Users/weckham/Desktop/MUSIC/下个路口见_WEC自研格式引擎` 中已有的文件，也不要启动耗时的全曲 Demucs 重分离。）
```
