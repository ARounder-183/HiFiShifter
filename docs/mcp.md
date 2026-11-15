# App MCP 使用说明

HiFiShifter 的独立 App 可作为本机 MCP 服务，供 Agent 读取和编辑**当前可视工程**。轨道、素材、音高和其他参数都写入 App 当前状态，人可以继续试听、修改、撤销和保存。

本页是 Agent 的操作契约，描述当前 App MCP 的实现、调用顺序、参数规则和失败处理。素材标注、模型下载和外部选样 CLI 不属于 App MCP；它们可以离线生成计划，但最终工程修改必须通过这里列出的 MCP 工具完成。

## 1. 启用与连接

1. 启动独立 App，打开需要编辑的工程。
2. 打开 **选项 → Agent MCP**，点击“启用”。
3. 点击“复制连接配置”，添加到支持 **Streamable HTTP** 的 MCP 客户端。
4. 让 Agent 先调用 `hfs_get_context`，再调用 `hfs_get_edit_capabilities`。

连接配置示意如下，实际地址以 App 显示为准。本机使用免鉴权，不需要账号、密码或 token：

```json
{
  "mcpServers": {
    "HiFiShifter": {
      "url": "http://127.0.0.1:39393/mcp"
    }
  }
}
```

如果客户端使用不同配置格式，只填写 HTTP URL 即可。客户端 SDK 负责 MCP 初始化/版本协商，这个 URL 不是普通 REST 接口。

- 默认关闭，每次 App 运行中单独启用；地址固定为 `http://127.0.0.1:39393/mcp`，重启不变。端口被占用时明确报错，不自动换随机端口。
- 只监听 `127.0.0.1`，不做鉴权；保留 Host/Origin 本机请求校验。
- 当前提供 HTTP 传输；没有内置 stdio 桥接器，也不提供远程 OAuth 登录。
- 使用官方 Rust SDK `rmcp`。协议回归覆盖 `2025-11-25` 初始化式调用和 `2026-07-28` 每请求元数据调用。
- MCP 入口只在独立 App 中显示；VST3/ARA/CLAP 不启用这条独立工程写入通道。

## 2. 本地素材标注与对齐模型

这些模型**不内置到 MCP 服务，也不通过 MCP 下载或运行**。Agent 使用本地外部 CLI/Skill 处理素材库，再把得到的时间、音素、音高和音色元数据整理成导入计划；App MCP 只负责把计划中的 clip、轨道路由和参数写入当前可视工程。

推荐的首版本地组合：

| 模型或工具 | 用途 | 获取与本地约定 |
| --- | --- | --- |
| **HubertFA v0.0.7 ONNX** | 中日歌声音素强制对齐；输出音素起止、元音起点和边界置信度 | 需要完整模型目录，不能只放一个 `model.onnx`。发布压缩包 SHA-256：`48bd6dbcc293e47cc6cbcc556baf575e3d91d61d0ac88ddeccd6098f8f346fa2` |
| **FCPE ONNX** | 逐帧 F0、有声掩码和中位音高，用于调音与选样 | App 已有资源：`backend/src-tauri/resources/models/fcpe/fcpe.onnx`；当前清单 SHA-256：`013B507ACD406DE122B61EE497300AC6BE082511101BC2136B886E48D9737E5D` |
| **openSMILE eGeMAPSv02** | 提取音色、唱腔和相邻音色连续性特征 | 本地 Python worker 使用；不是音素对齐模型的替代品 |
| **SOFA / MFA** | HFA 失败时的中文或日文对齐回退或交叉检查 | 可选本地回退，不默认随 App 或 MCP 打包 |
| **emotion2vec+ base** | 情感软分布或 embedding，作为低权重选样线索 | 可选；不是歌声音素边界模型 |

HubertFA 目录至少包含：

```text
model.onnx
VERSION
vocab.json
config.json
japanese_dict_full.txt
ds-zh-pinyin-lite.txt
```

加载前要检查 `VERSION` 与 adapter 约定的一致性和词典覆盖率。文件名提供读音候选；如果词典外词或文件名存在歧义，结果必须标成 `needs_review`，不能根据当前歌曲旋律猜一个读音后自动导入。

标注流程应把以下字段写入素材索引：

```json
{
  "asset_id": "asset-...",
  "content_sha256": "...",
  "relative_path": "a/ka.wav",
  "reading": "ka",
  "phones": ["k", "a"],
  "phone_spans_sec": [[0.055791, 0.069803], [0.069803, 0.418452]],
  "vowel_onset_sec": 0.069803,
  "vowel_end_sec": 0.418452,
  "stable_vowel_regions_sec": [[0.1195, 0.3505]],
  "f0_median_midi": 61.02,
  "voiced_mask_artifact": "f0/asset-....npz",
  "style_artifact": "features/asset-....npz",
  "annotation_status": "auto_ready"
}
```

导入 App 时，`source_start_sec`、`source_end_sec` 和切分点都使用**原文件时间**。重采样或模型工作副本的坐标不能直接写入工程；必须保留反向映射。App 最终仍用自身轨道设置进行拉伸和 HiFi-GAN 调音。

## 3. 工具一览

工具的 JSON Schema 是可执行契约。以下参数例子是 **`tools/call.arguments`**，不包含客户端 SDK 包装的 JSON-RPC 字段。

| 工具 | 用途 |
| --- | --- |
| `hfs_get_context` | 当前工程 revision、有效设置、模型状态、GUI 时间线同步状态 |
| `hfs_get_edit_capabilities` | 轨道属性、有效调音算法、根轨参数及其单位/范围/门禁，含主音高和子轨虚拟参数 |
| `hfs_query` | 分页读取 timeline/tracks/clips、曲线/静态值/参数块、设置、播放、渲染和历史 |
| `hfs_edit` | 原子批量修改轨道、素材、Take、参数、Tempo Map 和参数块；一批一步撤销 |
| `hfs_history` | 使用同一个 App 撤销栈执行 undo/redo，并检查预期 revision |
| `hfs_transport` | 播放、停止、定位当前 App 工程 |
| `hfs_settings` | 回读默认/有效拉伸配置，或显式设置工程拉伸 override |
| `hfs_editor_view` | 查询或操作实际 GUI 的轨道/素材选择、活动参数、参考轨道和工具模式，等待 GUI 回执 |
| `hfs_save_project` | 无对话框保存可重新打开的 `.hshp` 工程副本；拒绝覆盖已有目标 |

### 配置权威

调音与拉伸使用 App 已有配置。自动编排新轨道默认使用 HiFi-GAN；已有轨道的有效算法由根轨设置决定。MCP 不在曲线请求中另设一份声码器，也不自动覆盖人工选择。

当前代码的调音算法/Compose 属于轨道，合成通常按根轨生效；拉伸 override 属于工程并继承 App 默认。`hfs_settings` 会明确返回 `scope=project/app`，不能把它当作某个 clip 的独立拉伸算法。

## 4. 推荐流程

1. `hfs_get_context` 获取 `revision`。
2. `hfs_get_edit_capabilities` 找到目标轨道和参数、合法值范围及依赖。
3. 用 `hfs_query` 查询需要修改的对象/区间，不反复读取整份工程。
4. 发送一个带 revision、幂等键和操作列表的 `hfs_edit`。
5. 确认写入成功后，查询 `kind=render` 和 GUI/模型状态；编辑提交不等于声音已渲染。
6. 人在同一 App 时间线和参数面板中检查、试听和修改。可用 `hfs_editor_view` 选中目标。

### 原子写入与重试

`expected_revision` 是 App 返回的不透明版本，不要自己生成。后端在工程副本上执行整批校验，再在实时工程锁内核对版本并提交；任何操作失败都不会把前面几步留在当前工程里。

同一请求的网络重试使用**相同的 `idempotency_key` 和完全相同的请求内容**。请求若已完成，会返回原结果并设置 `replayed=true`。相同 key 搭配不同内容会报 `IDEMPOTENCY_CONFLICT`。缓存保留本进程最近 128 个成功编辑，不承诺跨重启去重。

收到 `REVISION_CONFLICT` 时重新读取相关状态，重新计算操作并使用新 key；不要无条件覆盖人工编辑。GUI 正在持有撤销分组时可能返回 `EDITOR_BUSY`。

## 5. 轨道、素材与批处理

下面的例子创建一条轨道并开启合成，算法沿用 App 默认：

```json
{
  "expected_revision": "<hfs_get_context 返回的 revision>",
  "idempotency_key": "create-lead-001",
  "operations": [
    { "kind": "create_track", "client_id": "lead", "name": "Lead" },
    { "kind": "patch_track", "track_id": "lead", "patch": { "compose_enabled": true } }
  ]
}
```

`client_id` 是同一批中后续操作可引用的别名。结果的 `created_ids` 返回别名→真实 ID，`created_track_ids` / `created_clip_ids` 也包含复制、切分等操作新建的对象。

轨道操作：

- `create_track`、`patch_track`、`move_track`、`duplicate_track`、`remove_track`。
- patch 支持名称、颜色、静音、独奏、线性音量、Compose 和调音算法；移轨支持顺序和父轨。
- 轨道层级不能形成环。删除轨道沿用 App 的子树/素材删除语义，只移除工程对象，不删除源文件。

素材操作：

- `import_clip`：现有绝对音频路径、轨道、工程起点、源起点、长度和播放倍率。
- `patch_clip`：裁剪、时长、倍率、增益/Item 增益、静音、倒放、循环、颜色和淡化。
- `move_clip`、`split_clip`、`remove_clip`、`duplicate_clips`、`group_clips`。
- Take 支持 `set_active_take`、`rename_take`、`duplicate_take`、`remove_take`、`pack_takes`、`explode_takes`。可先导入多个版本，再打包为 Take；换样后的几何/调音需按版本边界重新安排。

例如，在已有轨道导入一段单音：

```json
{
  "expected_revision": "<最新 revision>",
  "idempotency_key": "place-note-001",
  "operations": [
    {
      "kind": "import_clip",
      "client_id": "note",
      "track_id": "<真实轨道 ID>",
      "path": "E:/voicebank/ka.wav",
      "start_sec": 2.0,
      "source_start_sec": 0.02,
      "length_sec": 0.4,
      "playback_rate": 1.0
    }
  ]
}
```

所有位置是**工程绝对秒**，源起点是**原文件秒**，长度是**工程时长**；有效消费速率由 App 的 clip/Take 模型决定。导入时源窗口不能超出文件。单音库文件名/音素/元音边界的推断由外部处理，不由这个导入接口猜测。

### 元音伸缩与交叉淡化

Agent 可以在一批操作中导入同源的两段 clip，分别设置源窗/倍率，并设置左段淡出、右段淡入。拉伸算法仍由 App 设置决定。

中间必须有真实重叠，两侧 fade 使用同一个**工程秒重叠宽度**，同时检查 auto fade 是否覆盖 manual fade。不是各写一个源域 handle 时长，也不是让两个零间隙片段分别淡出/淡入。

若目标元音的拍线为 `A`，交叉淡化宽度为 `F`，左段源时长为 `Lsrc`，右段源时长为 `Rsrc`，目标音符时长为 `D`，当前实现推荐：

```text
左段起点 = A + F - Lsrc
左段 fade-out = F
右段起点 = A
右段工程时长 = D
右段 playback_rate = Rsrc / D
右段 fade-in = F
```

相邻音符属于同一连续乐句时，可以先缩短前一个元音，使后一个辅音有清晰的进入空间；保留的重叠只用于交叉淡化。若中间存在休止或乐句边界，不要用滑音或淡化跨过休止。

当前 MCP 提供这些可组合编辑原语，不替外部 CLI 自动检测稳定元音、选择同音版本或解析 UST。

## 6. 参数曲线、调音与参数块

查询区间曲线：

```json
{
  "kind": "curve",
  "track_id": "<真实轨道 ID>",
  "param": "pitch",
  "start_frame": 400,
  "frame_count": 100
}
```

以返回的 `frame_period_ms` 为准；当前为 5 ms，因此 frame 400 对应工程 2 秒。`orig` 是原线，`edit` 是解析后的编辑线，`edited` 表示是否有显式编辑。观察 `analysis_pending` 和模型状态，缺少音高分析时先等待 App 分析。

参数包括主音高、`volume/pan/dyn`、当前算法的共振峰/气声/张力/怒音等曲线、静态枚举，以及子轨音分差、音阶度数差、共振峰差。通过能力工具读取完整 ID 和单位，尤其不要混淆 MIDI 半音、cents、scale steps、线性增益和绝对动态电平。

编辑操作：

| operation.kind | 语义 |
| --- | --- |
| `write_curve` | 从 `start_frame` 连续写 values，区间之外保持原状 |
| `restore_curve` | 指定帧区间恢复基线，取消显式编辑 |
| `set_static` | 设置描述符中列出的静态枚举值 |
| `transform_curve` | 指定工程秒区间，应用 set/ramp/shift/shift_to_note/flatten_to_note |
| `set_param_blocks` | 以查询得到的 schema 替换根轨参数块布局；与曲线写入可放在同一事务 |

参数块使用当前 App 的 `ParamBlockEditSet` v2 契约。先查询：

```json
{
  "kind": "param_blocks",
  "track_id": "<根轨或子轨 ID>"
}
```

返回的 `layout` 是可写权威，核心结构为：

```json
{
  "schemaVersion": 2,
  "revision": 3,
  "blocks": []
}
```

`view` 是由当前时间线与 `layout` 派生出的只读显示结果，不能作为写回值。修改时提交完整 `layout`；`revision` 必须来自最近一次查询，MCP 会在事务中生成下一 revision。块中的 Take 成员使用源域时间，不能写成固定的工程绝对帧，具体字段以 `hfs_query(kind=param_blocks)` 的实时返回为准。

例如，把 2.0–2.4 秒的有声音高拉平到 C4：

```json
{
  "expected_revision": "<最新 revision>",
  "idempotency_key": "tune-note-001",
  "operations": [
    {
      "kind": "transform_curve",
      "track_id": "<真实轨道 ID>",
      "param": "pitch",
      "start_sec": 2.0,
      "end_sec": 2.4,
      "transform": { "mode": "flatten_to_note", "target_midi": 60.0 }
    }
  ]
}
```

`shift_to_note` 根据该区间原线的有声音高中位数移到目标音高，保留起伏；`shift` 在当前编辑线上加参数单位的 amount（pitch 的 1.0 为半音）。拉平/平移音高会保留无声帧；缺少可靠原线返回 `ANALYSIS_PENDING` 或 `PITCH_UNAVAILABLE`。

曲线属于根轨，不是独立 clip 效果。同根下多个 clip 的重叠会共享一份目标线；不同目标音高需要分配独立根轨，不能靠两个子轨获得两份独立主音高曲线。子轨差值参数则通过其专用 ID 操作。

Compose 和分离开关等门禁会在能力工具中给出 `disabled_reason`。例如，需要气声/张力时先显式启用 `breath_enabled`，并确认 HNSEP 依赖；MCP 不暗中改开关。参数块只是边界布局，创建块本身不等于改变声音，需要同时写对应曲线。

任意曲线形状可通过 `write_curve` 精确写入。当前命名 transform 不包含单独的平滑/颤音/量化算法工具；外部 Agent/CLI 可计算样本后批量写入，不应假设尚未提供的工具已经存在。

## 7. 视图、播放、渲染与保存

选中轨道并切到音高面板：

```json
{
  "selected_track_id": "<真实轨道 ID>",
  "active_param": "pitch",
  "tool_mode": "select"
}
```

这是 `hfs_editor_view` 的参数。省略所有字段即读取实际 GUI 状态；窗口未响应返回 `VIEW_UNAVAILABLE`，人工手势期间返回 `VIEW_BUSY`。目前支持轨道/素材选择、活动参数、参考根轨和 select/draw/line/vibrato 工具；不提供坐标式鼠标操作。

播放使用 `hfs_transport`，例如 `{"action":"play","position_sec":2.0}`，停止为 `{"action":"stop"}`，定位为 `{"action":"seek","position_sec":2.0}`。

查询 `hfs_query` 的 `kind=render` 获取当前 clip 的 Pending/Rendering/Ready/Failed 状态、错误、模型状态和自动后台渲染开关。`NotRequired` 表示不需要合成，`NotObserved` 表示尚无可用结论，不应把它们当作一次 HiFi-GAN 渲染成功。App 自动后台渲染关闭时可通过正常播放触发准备。

`hfs_get_context.gui_revision` 只确认时间线快照被 GUI 采纳；它不证明参数窗口所有异步数据或音频渲染都完成。MCP 修改会触发正常的时间线和参数刷新，人工拖动期间延后采纳快照。

`hfs_save_project` 参数：

```json
{
  "expected_revision": "<最新 revision>",
  "path": "E:/projects/agent-arrangement.hshp"
}
```

它原子写出新的可编辑工程副本，返回 `saved_revision`。**当前是 snapshot_copy**：不切换当前工程路径、不清除当前工程 dirty、不包含撤销历史、不收集/复制源音频；工程使用现有媒体路径。常规保存、另存、媒体收集/打包继续使用 App 原有功能。目标已存在时拒绝覆盖，磁盘错误不会留下伪装成完整工程的半写文件。

工程副本可用 App 正常打开，继续查看素材、轨道设置和曲线；预览音频不替代工程文件。

## 8. 拉伸设置与错误处理

读取 `hfs_settings`：`{"change":{"action":"read"}}`。

显式设置工程 override 示例：

```json
{
  "change": {
    "action": "set_project_stretch",
    "expected_revision": "<最新 revision>",
    "algorithm_override": "signalsmith",
    "hifigan_mel_stretch_override": false
  }
}
```

这表示用户明确要求改变工程设置；普通编排不调用此写操作。两个 override 字段是完整目标配置，`null` 表示继承 App 默认；选择包括 signalsmith/soundtouch/linear。此操作与 App 原有设置行为一致，不进入时间线 undo，回复会标明 `undoable=false`。

常见稳定错误码：

- `REVISION_CONFLICT`：读取新状态并重新规划。
- `IDEMPOTENCY_CONFLICT`：同一 key 的内容被改动，应使用新 key。
- `INVALID_ARGUMENT`、`TRACK_NOT_FOUND`、`CLIP_NOT_FOUND`：检查 ID、范围、单位和路径。
- `PARAM_NOT_SUPPORTED`、`PARAM_DISABLED`：查询当前算法描述符和依赖。
- `ANALYSIS_PENDING`、`PITCH_UNAVAILABLE`：等待分析或核对音频/模型，不能用零音高补齐。
- `EDITOR_BUSY`、`VIEW_BUSY`、`VIEW_UNAVAILABLE`：完成当前交互或恢复 App 窗口后重试。
- `SAVE_FAILED`：检查目录/空间/已有目标，当前工程编辑仍保留。

领域失败用 MCP `isError=true` 和结构化 `ok=false/code/message` 返回。工具参数不匹配由 SDK 返回协议/参数错误。不要仅凭 HTTP 200 或“听到原声”判断编辑和调音已经完成。

## 9. 当前实现边界

- MCP 直接操作当前 App，不托管素材分析模型，不自动解析 UST 或替 Agent 挑选版本。
- 不提供任意 shell、任意 Tauri 调用、任意全工程 JSON 覆盖或源文件删除能力。
- 本版采用 `hfs_edit.operations` 统一承载轨道/素材/参数动作；独立工具入口以 `tools/list` 为准。
- 参数读窗口最多 20,000 帧，一批最多 256 个操作，曲线读改/恢复的编辑预算为 200,000 个样本；大的工程请按区间分批。
- 目前没有持久化业务任务/跨重启幂等日志、外部 bundle 注册及专门的音乐意图规划器。Agent 可以用已暴露的原子编辑动作组合完成编排与调音。

源码入口：[`hifishifter-mcp`](../backend/hifishifter-mcp/src/lib.rs)、[类型契约](../backend/hifishifter-mcp/src/contract.rs)、[App 编辑适配](../backend/src-tauri/src/mcp/editor.rs)、[GUI 同步](../frontend/src/hooks/useMcpProjectEvents.ts)。
