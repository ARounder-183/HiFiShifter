# F-3 实测：一个三 take 的 item，REAPER 给 ARA 发什么

2026-10-09，REAPER **7.82/x64**，Windows x64。原始证据：`captures/f3-multi-take-7.82.md`。
夹具：`fixtures/ara_multi_take_setup.eel`，跑法见 `README.md`「无头跑探针」。

## 夹具

一个 item，**三个 take**，各自指向**不同的音频文件**（`tone44100.wav` / `tone48000.wav` /
`embedded-editor-voice.wav`），并把 active take 设成下标 1（即 `tone48000.wav`）。
夹具自证读数（`.build-tmp/f3-fixture.log`）：`takes=3`、`curtake=1`。

三个不同文件是刻意的：插件日志按 source 的 persistentID（文件路径）去重，三个不同文件
才能把"只发 active take"与"全发"区分开；active 取中间那个，才能把"只发 active"与
"只发第 0 个"也区分开。

## 结果

**（1）三个 take 各建了一个 audio source，而且 PCM 都读得到。**

```
[ara] audio_source #0: persistentID=.../tone44100.wav sampleRate=44100 sampleCount=88200
[ara] host PCM ready source=0 frames=88200 version=0
[ara] audio_source #1: persistentID=.../tone48000.wav sampleRate=48000 sampleCount=96000
[ara] host PCM ready source=1 frames=96000 version=0
[ara] audio_source #2: persistentID=.../embedded-editor-voice.wav sampleRate=44100 sampleCount=88200
[ara] host PCM ready source=2 frames=88200 version=0
```

`host PCM ready` 这一行只在 `render::source::read_source_pcm` **成功之后**才打印
（`ara/model.rs:650-679`），所以这不只是"宿主建了 source 对象"，而是"插件真的读到了
两个非 active take 的采样"。

**（2）播放区域只有一条，而且指向 active take。**

```
[ara] playback_region #0: source=.../tone48000.wav startMod=0.000000 durationMod=2.000000 ...
[ara] host inventory: 1 track(s), 1 item(s), 1 claimed by an assigned region
```

一个 item → 一条 playback region，`source` 就是 `I_CURTAKE=1` 那个 take 的文件。

## 结论：要修正 Part 2 的推断

plan Part 2 的结论是"**多 Take 在 ARA 路径上拿不到**"，理由是"REAPER 不这么发"。
实测**推翻**了后半句：REAPER 把**每个 take 都当作独立的 audio source 发出来**，
连非 active take 的 PCM 都能读。

真正缺的只有两件事，而它们都不在 ARA 里、在 REAPER 自己的 API 里：

| 缺什么 | ARA 侧 | REAPER API 侧 |
|---|---|---|
| "这几个 source 是同一个 item 的 take" | 无此概念（source 之间没有边） | `GetMediaItemNumTakes` + `GetMediaItemTake(item, i)` |
| "哪个 take 是 active" | `ARAAudioModificationProperties` 只有 name + persistentID | `I_CURTAKE`，或"唯一那条 playback region 的 source" |

也就是说：Part 2 的 Task 2.1/2.2（绑定 take 枚举、把非 active take 做成 lane）**不只是
"可以做"，而且比原方案更有利** —— 原方案只敢承诺非 active take 走"无源占位（斜纹）"，
现在实测表明它们的 PCM 也能经 ARA 拿到，所以 lane 可以是真波形而不是占位。

## 还没验证的（不要当成已知）

- **`source_path` 的授权边界**。`ui_inventory.rs:264-267` 有一条断言要求非 active take 的
  `source_path` 保持 `None`（"显示占位不得未经 ARA 授权读取文件 PCM"）。本次实测说明
  ARA 的 PCM 读取是通的，但"**这个实例**的 ARA 授权是否覆盖非 active take"是另一回事 ——
  本次只证明"宿主把 PCM 给了"，没证明"这样做在授权模型下也成立"。动 Task 2.2 之前要先把
  这条边界想清楚，否则会把一条安全底线当成技术限制绕过。
- **多轨、多 item 的情形**。夹具只有一条轨、一个 item。REAPER 是按"ARA 轨上所有 take
  源"发 source，还是按别的口径，没有区分过。
- **take 数变化时的增量**。改 take（增删、切换 active）会发多少 modification / 是否重建
  source，没测。
- **倒放位**。plan Part 2 的 Task 2.3 说 ARA 侧拿不到方向位、要从
  `PCM_Source_GetSectionInfo` 的 `revOut` 读。本次没有构造倒放 take，未验证。
