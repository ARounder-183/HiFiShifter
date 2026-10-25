# HAMOOD 伴奏分析与和声 MCP

公开版本 0.25，内部更新 037。

## 接入内容

- Beat This! small0：节拍与小节首拍，约 8.1 MB 权重。
- BTC 170 类：带起止秒数的和弦标签、候选及未校准的模型分数，约 11.7 MB 权重。
- CQT 低频音级：每段的低频 C 至 B 候选及相对强度。这是本机频谱统计，不是 NNLS 模型，也不是精确贝斯转谱。

现成权重不需要用户训练。原伴奏按原音播放，分析只建立旁路数据。分析输入先用编辑器相同的音频解码器转为临时 PCM，避免 MP3 解码延迟差异影响时间对齐；分析结束清除临时 PCM。

## 运行时

配布目录 `engines/hamood/runtime.json` 的 `python` 指向 Python 解释器，支持绝对路径及相对该配置文件的路径。更新 037 完整配布使用 `../python/python.exe`，复用编辑器随包的 CPU PyTorch 与补齐的分析依赖，不再要求本机 Anaconda。旧版 036 的本机配置仍可使用，但迁移时推荐完整包。

核心依赖：torch、torchaudio、numpy、scipy、librosa、soundfile、soxr、PyYAML。扩展目录 `deps` 内提供 beat-this、einops、rotary-embedding-torch。BTC 源码保留 MIT LICENSE，仅将已废弃的 `np.float` 替换为 `float` 以适配 NumPy 2。缓存目录重定向到工程缓存，避免首次特征提取尝试写入只读 Python 安装目录。

开发可用 `HACHI_HAMOOD_RUNTIME` 指定 runtime.json；正常配布从 EXE 同目录的 engines/hamood 加载。模型只从固定本机路径加载，不接受 MCP 任意脚本或模型 URL。

## MCP 流程

1. `project_snapshot` 获取伴奏片段 ID 和源旋律轨 ID。
2. `hamood_analyse_audio {clip_id}` 返回任务 ID，按 `editor_job_status` 查询，确认 completed 且 result.isError 为 false。
3. `hamood_audio_context {clip_id}` 默认返回选中音符所在的时间范围。无选区时须显式给 from_seconds / to_seconds，或 whole_clip=true。和弦默认每页 64 条，最多 256 条，使用 next_offset 继续。拍点在所选范围内最多返回 4096 个。
4. `hamood_alignment_preview {clip_id,from_seconds,to_seconds}` 检查当前工程拍点网格与伴奏的相位差。默认每四分音符二等分；返回前、中、后三个窗口。此工具不移动音频。确认后可调用现有 move_clip，传 expected_revision。
5. `hamood_preview {track_id,whole_track:true,key_mode:"sections",section_bars:4,voices:[2],audio_clip_id}` 预览。参数遵循原 HAMOOD；不指定 whole_track 时使用当前选区。
6. 确认预览后，使用相同参数调用 hamood_generate 并带 expected_revision。源轨不修改，新声部继承音源、FLAG、时长与唱法，一次撤销。

缓存位置为 `工程.hjpx.hamood-cache`。未保存工程使用临时缓存。保存到新工程路径后可以再次调用分析，由新位置建立缓存；不会自动删除旧缓存。音频内容 / 模型权重哈希记录在结果中；快速查询校验音频大小和修改时间，换模型或手工改写音频元数据后请用 force=true 重新分析。

## 结果与限制

- `start` / `end`、beats / downbeats 返回工程绝对秒数；和弦另带原音频 source_start / source_end，及工程小节编号。移动或裁剪片段后重新查询，不能沿用旧的时间位置。
- 当前接受伴奏轨道和未拉伸的普通原音轨片段。复杂原音时间映射不在此接口范围。
- BPM 为量化拍点间隔的粗估，存在半速 / 倍速或漏拍。不会自动覆盖工程 BPM。相位对齐精度受约 20 ms 拍点分辨率限制，也无法证明整拍 / 整小节的偏移正确；速度漂移需要另外处理。
- 和弦分数不是准确率。默认仅使用 score >= 0.35 且覆盖足够时长的和弦；其余位置保留调性规则。低音候选作为 MCP 证据输出，当前不会直接决定和声音高。
- 新声部倾向于所选三度 / 六度等音程附近的和弦音，不保证每个音严格维持该音程。长音跨和弦时汇总重叠时长，不自动切分原音符。此结果是可编辑草稿，不等于经过人工逐音配器。
- 当前这组新入口位于窗口 MCP，菜单版 HAMOOD 保留原调性流程。分析失败时返回错误，不伪造和弦结果。

官方项目：https://github.com/CPJKU/beat_this 、https://github.com/jayg996/BTC-ISMIR19 。
