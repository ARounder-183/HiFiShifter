# F-2 实测：REAPER take 包络（PITCHENV）的横轴是**播放时间**，不是媒体时间

2026-11-17，REAPER **7.85/x64**，Windows x64。判据是**渲染后的音频**，不是 chunk 里
的数字自证：素材为 6 s / 440 Hz / 44.1 kHz 正弦，按 50–100 ms 窗数过零率反推音高
（440 Hz ⇒ 0 半音，880 Hz ⇒ +12 半音，中间即斜坡位置）。探针用
`SetItemStateChunk` 直接写 `<PITCHENV>`，再 `Main_OnCommand(42230)` 离线渲染。

## 要回答的问题

`hifishifter-kernel::pitch_bridge` 的时间域约定是 `u = (t − item_start) × rate`
（`u = 0` 在 take 入点）。若宿主实际按**媒体文件时间**（即含 `SOFFS`）解读包络点，
则"源起点不为 0"的片段会整条平移 `SOFFS`，插件桥接与 `.rpp` 导出都要补一项。
两种解读只差一个常量，chunk 本身看不出来 —— 只能渲染。

## 结论一：`u = 播放时间 × |PLAYRATE|`，`SOFFS` / `STARTPOS` / `MODE` / `LOOP` 都不进坐标

| # | take 几何 | 包络点 | 渲染结果（窗内主导频率） | 说明 |
|---|---|---|---|---|
| A | `SOFFS 2`、item 1 s | `u=2.0 / 2.5` | 全程 440 Hz | **点不生效**：u 若含 SOFFS 就该在 0.5 s 处跳到 +12 |
| B | `SOFFS 2`、item 1 s | `u=0 / 0.5` | 440 → 0.5 s 起斜坡 → 880 | 同一几何、只换点的域 ⇒ 生效 |
| C | `SOFFS 0`、item 1 s | `u=0 / 0.5` | 440 → 880 | 对照：机制本身可用 |
| D | `PLAYRATE 2`、item 1 s | `u=0.5 / 1.0` | 0.25 s 起斜坡、0.5 s 到 880 | ⇒ u 含 `× rate`（媒体消耗量） |
| E | `SECTION STARTPOS 3 LENGTH 3` + `SOFFS 1`、item 2 s | `u=0.5 / 1.0` | 0.5 s 起斜坡、1.0 s 到 880 | ⇒ `STARTPOS`/`SOFFS` 都不进坐标 |
| F | 倒放（`MODE 3` + `SOFFS 1`）、item 1 s | `u=4.25 / 4.75` | 全程 440 Hz | 镜像的媒体时间**不生效** |
| G | 同上 | `u=0.25 / 0.75` | 0.25 s 起斜坡、0.75 s 到 880 | ⇒ 倒放 take 的 u 是**播放顺序**，不是媒体顺序 |
| H | 倒放 `MODE 2`（段 `[3,6)`、`SOFFS 0`）、item 2 s | `u=0.5 / 1.0` | 0.5 s 起斜坡 | 段边界同样不进坐标 |
| I | 循环源（段 1 s、item 3 s、`LOOP 1`） | `u=0.5 / 2.5` | 0.5 s 起斜坡、2.5 s 到 880 | ⇒ 循环也不按"每圈重来"，u 覆盖整个 item |

A 与 B 是同一份 take 几何下只改点的域，因此这条结论不依赖"渲染有没有偏移"这类
弱信号：不对的域是**完全不生效**，对的域立刻出现斜坡。

## 结论二：chunk 里的 `PT` 就存在这个域，宿主不会重定基线

- 写完点之后把 `SOFFS` 从 2 改成 3：`PT` 仍是 `0 / 0.5`（`SetItemStateChunk` 读回逐字一致）。
- 写完点之后执行宿主自己的 `Item properties: Toggle take reverse`（41051）：它把
  `SOFFS` 改成镜像量（本例 2 → 3 = `D − 正向终点`）并加上 `SECTION MODE 3`，
  `PT` **一字不动**。再切一次回到 `SOFFS 2`、SECTION 消失，`PT` 仍不变。

即：宿主把包络点当作"入点起算的播放时间"，改源窗口/方向都不会平移它。所以

1. 写入端**不得**加 `SOFFS`，也**不得**做倒放镜像；
2. 倒放写回只改 `SOFFS` + SECTION，**不得**镜像 `PT`（宿主自己也不镜像）；
3. 读入端照 `PT` 原值采样即可。

插件路径同理但要再叠一条：它换算的基准是**片段的起点**，而片段起点恒等于宿主
item 的 position —— `plan_host_edit` 与 `plan_host_split` 都对
`geometry.start_sec ≈ clip.start_sec` 做前置校验（不一致即拒绝写回）。
因此那里同样只有一个 `× PLAYRATE`，既没有 `SOFFS` 也没有 item 起点偏移。

## 结论三：空的 `<PITCHENV>` 块会被宿主丢弃

只写 `ACT / VIS / LANEHEIGHT / ARM / DEFSHAPE` 而没有 `PT` 的块，`SetItemStateChunk`
返回成功后读回**块已消失**（`GetTakeEnvelopeByName(take,"Pitch")` 为 `NULL`）；
带至少一个 `PT` 的块才会被保留，宿主还会补一个 `EGUID`。
所以"删除包络"只有一条可靠写法：整块不写（本仓 `rewrite_take_pitch` 已是该行为）。

## 未覆盖

- **非线性拉伸标记**（`<STRETCH>`）下的分段映射没有实测。现有写回路径对带拉伸标记的
  take 一律拒绝（`host_edit` 的前置检查），因此不影响当前代码。
- `PRESERVE_PITCH` 关闭（速率改音高）时的组合没有单独测；D 已证明 `× rate` 这一半。

## 附：take 音高**不作用**于插件（ARA）的渲染结果

顺带回答的第二个问题：把 HiFiShifter 的音高线"转换到 take 包络"以后，宿主侧那份值
作用在哪一层？判据同样是渲染：

- 素材 1 s **静音**；take FX 放 REAPER 自带的 `JS: Tone Generator`（wet 0 dB、
  dry −120 dB、基频 440 Hz）—— 这样唯一的声源在 take FX 里，与源素材无关。
- A（对照）：无包络 ⇒ 渲染 **440.0 Hz**。
- B：take 的 `<PITCHENV>` 全段 +12 半音 ⇒ 渲染 **440.0 Hz**（不是 880）。

⇒ take 音高（`D_PITCH` / `PITCHENV`）在**源级**生效、早于 take FX，不会对 take FX
产生的信号再移调。

对 ARA 的推论（**推断**，本机没有第三方 ARA 插件可做反证）：ARA 插件交给宿主的渲染
结果同样落在那层之后 —— 宿主若对它再重采样一次，插件自己做的时长映射会被二次拉伸。
与本仓的现场观察一致：转换到 take 包络后听感毫无变化，而 HiFiShifter 的音高线被清空
⇒ 用户听到的"音高没了"。

因此音高互转是**复制**：插件侧的音高线原样保留，宿主侧的 `PITCHENV` / `D_PITCH`
也原样保留，两边永远不会同时处理。（唯一例外是"转换为 take 包络"时把 `D_PITCH`
归零：它是同一个 take 上的第二条音高来源，而这条命令的语义就是让包络成为 REAPER 侧
唯一来源。）
