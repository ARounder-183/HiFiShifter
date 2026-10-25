# MCP 连接当前编辑器窗口（0.25）

先启动最新版编辑器，打开需要编辑的工程，再启动 MCP 客户端。此模式操作窗口里的真实工程、选区、撤销历史和音频引擎。连接端是一个独立的 stdio 代理进程，不会另建空白工程。

```json
{
  "mcpServers": {
    "hachishifter": {
      "command": "E:/和声合成新UI/HachiShifter-整合版-0.25-DiffSinger/HachiShifter Next.exe",
      "args": ["--mcp-live"]
    }
  }
}
```

仓库 `.mcp.json` 已改为连接窗口。原 `--mcp` 保留为无界面的独立工程模式，旧批处理脚本不用修改。

## 窗口选择

- 只打开一个窗口时自动连接；多个窗口时不会猜测目标。
- 用 `"HachiShifter Next.exe" --mcp-list-sessions` 获取窗口列表，然后使用 `--mcp-live --session=列表中的session_id`。
- GUI 子系统没有控制台，需要从客户端管道读取输出。PowerShell 可用 `& '完整EXE路径' --mcp-list-sessions | Out-String`。
- 每个代理连接后固定绑定这个窗口。窗口关闭或崩溃后返回连接错误，不会转去修改其他工程。崩溃窗口的心跳最多约 15 秒过期。
- 本地临时目录 `HachiShifter-MCP/<随机会话ID>` 用作进程间邮箱，不开放网络端口。正常关闭自动清理本窗口目录。意外退出留下的目录不参与过期会话连接。

## 推荐调用流程

1. `initialize` → `tools/list`。
2. `editor_status` 获取 `session_id`、`revision`、工程文件、是否未保存、当前选区、播放/渲染状态。上方轨道区的框选通过 `selected_clip_ids` 和 `selected_track_ids` 返回；单数的 `selected_clip_id` / `selected_track_id` 仍表示当前焦点，下方音符选区仍使用 `selected_note_ids`。
3. **优先调用 `editor_selection {}` 读取当前选区**：直接返回歌词、音名（C4 = MIDI 60）、音高、起止时间、时长、所属轨道和稳定音符 ID，以及逐音符文字摘要。只读取真实选区，不会因空选区而返回整轨。`editor_query` 仍默认仅读取选中音符。`scope="project"` 读取工程；可组合 `track_id`、`from_seconds`、`to_seconds`、`offset` 和 `limit`（最多 256）。时间范围使用工程绝对秒数，返回 `project_start_seconds`。`include_curves=true` 才返回密集曲线。
4. 写入时传 `expected_revision`，使用刚读取的修订号；可附 `session_id` 再校验窗口。修订号不符时返回 `STALE_REVISION`，不覆盖新编辑。
5. 耗时命令返回 `job_id`，用 `editor_job_status` 查询到 `completed` / `failed` / `cancelled`。还要检查 `result.isError`。接收任务不代表已完成。
6. 用 `undo` / `redo` 操作窗口的同一份撤销历史。

普通编辑在草稿上执行，成功后一次写入工程；校验失败不产生部分修改。`editor_batch` 将最多 128 个普通工程编辑合并成一个撤销步骤，任何子命令失败则整批不写入。批量命令不接受导入导出、文件操作、播放或长时间推理。

```json
{
  "name": "editor_batch",
  "arguments": {
    "expected_revision": 12,
    "commands": [
      {"name": "set_note", "arguments": {"note_id": "实际音符ID", "label": "bang"}},
      {"name": "transpose_note", "arguments": {"note_id": "实际音符ID", "semitones": 2}}
    ]
  }
}
```

## 按选区读取（029）

在钢琴卷帘中点选一个音符，或按原有方式多选/框选，然后让模型“读取现在选中的音符”。模型应调用 `editor_selection`，无需打开或解析 `.hjpx` 文件。每次调用读取当时窗口中的真实选区，包括未保存的修改。

```json
{"name":"editor_selection","arguments":{}}
```

默认最多返回 64 个音符，按工程时间排列，同起点按轨道/片段/音符顺序排列。`total` 是选区总数，`returned_count` 是当前页数量；如 `next_offset` 非 null，用相同工具带上该 offset 继续读取，直到为 null。最多可设置 `limit=256`。`range` 是所有返回范围内音符的起止包络，`tracks` 列出涉及的轨道；它们不是连续框选区域，范围内未选中的音符不会被带出。

`notes` 每行包含 `id`、`lyric`/`label`、`pitch_name`、`midi`、`duration_seconds`、`track_id`/`track_name`、`clip_id`、`project_start_seconds`、`project_end_seconds`、DS 标记与读音覆盖。`summary` 提供当前页逐音符文字摘要。`start_seconds` 是相对片段的时间，工程绝对时间请使用 `project_start_seconds` / `project_end_seconds`。需要选中音符的 pitch、参考线、FLAG 等细节时，传 `include_curves=true`；曲线横轴是音符内秒数。

资源 `hachishifter://editor/selection` 提供同样的默认第一页；超过 64 个时继续用工具分页。资源与工具列表均优先展示选区入口。工程整体读取保留为显式 `project_snapshot` 或 `editor_query {"scope":"project"}`，不改变旧接口含义。

读取不会改变选区、工程、撤销历史或播放状态；`revision` / `session_id` 可用于随后的编辑校验。选区变化不会改变工程修订号，因此用户重新选择后应再次读取，不要沿用旧的音符列表；分页期间也应保持选区不变。

## 窗口与文件

`editor_select` 更改真实选区和焦点；`note_ids: []` 清空选区。空选区查询不会变成全轨查询。旧 `utau_render_selection` 只选择渲染范围，保留旧接口含义。

`project_save` 同步收集已完成的 DS 预测，更新 Ctrl+S 的文件路径与已保存状态，并迁移工程旁 DS 缓存。打开工程、新建工程和替换工程的 Melodyne 导入会重置选区、播放位置和缓存会话。工程未保存时，先保存，或在明确放弃当前修改时传 `discard_unsaved=true`。

导入在后台处理，完成后校验修订号；如果你在导入期间继续编辑，过期草稿会被丢弃。任务取消保证不再提交草稿，但已完成的外部文件写入不自动回滚。不要自动重试结果未知的写操作，先查询工程/文件状态。

活动模态对话框存在时拒绝编辑指令，避免与对话框中的编辑冲突。普通读操作、停止播放与取消任务仍可使用。

## DS 专用接口

先用 `editor_select` 聚焦音源轨道与音符。

| 工具 | 含义 |
|---|---|
| `ds_capabilities` | 后台读取当前 DS 音源能力、参数范围和支持状态 |
| `ds_query_phonemes` | 在完整乐句上下文中预测音素，但默认只返回选中音符；`whole_track=true` 返回整轨 |
| `ds_generate_pitch` | 调用与界面相同的局部 pitch 重生成；保留偏移及未选区域 |
| `ds_generate_parameters` | 调用与界面相同的预测实参重生成；保留 FLAG 偏移 |
| `ds_set_pitch_offset` | `points=[[音符内秒数,半音偏移],…]`；空数组恢复零偏移，不改变原始虚线 |
| `ds_set_parameter` | `flag` 如 BREC；`layer=actual/offset`；`points=[[音符内秒数,值],…]` |
| `ds_set_pronunciation` | `note_id` 和 `text` 设置读音覆盖，空字符串恢复自动转换，清除旧时长上下文 |
| `ds_set_timing` | 写入音素起点覆盖；使用查询返回的 `context`、`tokens` 和对应 `starts`，null 恢复预测 |

局部重生成要求实际选中音符；只有显式 `whole_track=true` 才重生成整轨。选择必须属于当前 DS 轨道的有效片段。完成后可 Ctrl+Z 撤销。

实参支持 ENE/BREC/TENC/VOIC 中音源实际支持的项目。偏移使用音源暴露的 DS 参数；实际值和偏移单位不同，例如气声实参是 dB，气声偏移是 -100..100。不会把偏移曲线写成实参，也不会覆盖预测参考。参数超出内部有效范围时按编辑器规则限制。

音素查询每行有 `note_id`、`context`、`tokens`、`index`、`note_start`、`start`、`end` 等。写入的 `timing` 为 `{context,tokens,starts}`；`starts` 长度对应 tokens，数值是相对音符起点的秒数，某项为 null 表示保留预测。修改歌词或读音后必须重新查询上下文。此接口只能调节音源本身的音素划分，不会将 ang 强制拆成 a/ng。

## 试听与导出

`render_prepare`、`transport_play`、`export_wav` 返回后台任务 ID，窗口保持响应。渲染过程中修改工程或试听选区会使旧任务失败，避免导出错误范围。`transport_stop` 立即停止并取消待开始的 MCP 播放。

`export_wav` 支持 `channels`（1/2）、`sample_rate`（8000..192000；0 跟随设备）、`bit_depth`（16/24 PCM、32 浮点），以及原有轨道和起止范围。默认双声道、设备采样率、24 bit。

`editor_cancel_job` 取消等待或拒绝尚未提交的草稿；DS 推理使用现有可取消任务机制。已导出的文件不自动删除。

## 验证

自动化测试：`juce/tests/mcp_live_smoke.py <EXE> <测试输出目录> [DS测试工程] [测试音源目录]`。测试创建独立窗口并按 session_id 连接，不操作用户已有窗口；覆盖选区、修订号保护、批量原子提交、撤销、保存重开、多窗口隔离、播放和 WAV 格式，以及可选的真实 DS 功能。


028 验证：62 个窗口模式工具；38 项窗口与真实 ZhiBin DS 联调检查通过（另含 100 次连续邮箱请求）；旧模式 schema 检查无问题，文件范围检查 16 项通过，导入/编辑/撤销/保存重开 12 次调用通过。

旧的 `juce/tests/mcp_smoke.py` 全量脚本在 `add_note` 步骤返回 `No compose clip accepts the note`；修改前的配布 EXE 与本次 EXE 均复现。该已有失败不记为本次通过项，另以上述独立模式回归验证本次改动。

029 验证：63 个窗口工具；54 项实际窗口与 ZhiBin DS 联调检查通过（另含 100 次邮箱请求）。覆盖选区优先发现、单选、多选、空选区、实时更新、跨轨道排序、非零片段起点、范围过滤、分页、资源读取、无副作用、DS 曲线精确读取与局部重生成；同一工程的旧模式完整 JSON 与更新前配布 EXE 完全一致。

## HAMOOD（030）

新增 `hamood_preview`（只读）与 `hamood_generate`（一次撤销的生成操作）。默认当前轨道选区，整轨需显式 `whole_track=true`；支持自动/分段/手动调性、多声部和 DS 唱法保留。参数说明见 [HAMOOD 使用说明](hamood.md)。

032：手动模式支持 `manual_sections: [{start_bar:1,end_bar:8,tonic:0,minor:false}, ...]`，按标尺小节指定独立调性。结束小节包含在范围内，必须覆盖目标音符起点且不能重叠。参数类型及每行必填字段先校验，失败不修改当前工程。自动分析使用整条源轨上下文，选区仍只限制生成音符；输出新增小节范围、候选分数、`analysis_note_count` 与 `mixed_tonality`，存在歧义时应结合用户意图确认调性。


## 伴奏音频分析（036）

新增 `hamood_analyse_audio`（异步分析与缓存）、`hamood_audio_context`（按选区 / 范围读取）、`hamood_alignment_preview`（只读拍点相位对齐预览）。`hamood_preview` / `hamood_generate` 支持 `audio_clip_id` 与 `minimum_chord_score`，在当前片段位置参考伴奏和弦生成独立声部。需要本机分析运行时，详见 [伴奏分析说明](hamood-audio.md)。


## HAMOOD 工程资料（055）

`hamood_get_context` 读取当前工程保存的调性设置、手动调性分段、和弦、段落及人工确认状态；`hamood_set_context` 提交完整资料对象，需要 `expected_revision`，支持一次撤销。先读取并保留无关条目，再修改所需内容。资料随工程保存，GUI 与 MCP 共用。

`hamood_preview` / `hamood_generate` 的省略参数继承已保存设置，显式参数优先；整轨处理仍须明确 `whole_track=true`。已确认和弦不会被重新分析覆盖。详细字段与时间单位见 [HAMOOD 使用说明](hamood.md#工程资料055)。

`editor_status` 同时返回 `selected_note_count` 和 `selected_region_count`，用于检查当前多区域音符选区。


## OTO 高级包络（056）

`set_note` 的 `tail_fade` 字段可设为 `off`、`linear`、`smooth`，作用于普通 UTAU / 界·UTAU / 谋·UTAU 的 OTO 最后分区。`editor_selection`、`editor_query`、工程快照返回该设置；DS 无 OTO，不接受此效果。原手绘包络和 FLAG 保持独立，多音符修改可通过 `editor_batch` 合并为一次撤销。详见 [高级包络使用说明](advanced-envelope.md)。


## 尾段淡出细节（057）

`set_note` 增加五个可选数值字段：`tail_fade_start_percent` / `tail_fade_end_percent`（在 OTO 尾段内的起止比例，0–100，间隔至少 0.1%）、`tail_fade_start_gain_percent` / `tail_fade_end_gain_percent`（相对原包络的振幅倍率，0–200，结束倍率不得大于起始倍率）、`tail_fade_curve_power`（0.25–4，默认 1）。省略的字段保留当前值，单独设置细节不改变开关和形状；`off` 保留细节以便重启。模式和细节在同一请求中整体校验，失败不会部分修改音符，选区与工程快照返回全部字段。


### 058 · UST 编码与保真交换

`import_ust` 增加可选 `encoding`。新增 `export_ust(path, track_id, encoding?)`，保持原编码为默认行为；支持 auto、UTF-8、Shift-JIS、GBK、Big5、GB18030。当前窗口写入类接口仍需要 expected_revision，导出本身不修改工程。支持原始未知字段、空值及未启用音高数据随 HJPX 留存；完整规则见 [UST 保真说明](ust-fidelity.md)。
