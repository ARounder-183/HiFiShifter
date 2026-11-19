# take FX 下"完全没有声音"与"take 音高转换永远失败"的两个根因

2026-11-19，REAPER 7.85/x64 + 本仓 VST3 (`218c4eecd` 之后)。判据是**插件自己的日志**与**宿主
二进制里的接口表**，不是推断：两条故障各自都有一行日志能指到具体那一位读数上。

## 一、take FX 未旁通却**一点声音都没有**

现场日志（`plugin-vst3-reaper.log`，20:52 那次会话，HiFiShifter 挂在 take FX 上）：

```
[ara] placement: take-fx; chains: track=n/a take=1/1; regions=0
[ara] writer 1 host bypass: unknown -> bypassed
[ara] bypass gate: bypassed; writers(e/p=editor/playback)=[1p:bypassed/82ms,2e:unknown/never]; background job skipped
```

链路是直的：`take=1/1`（take 链上有一个 FX，且按类名匹配到了本插件）⇒ 旁通读数却报
`bypassed` ⇒ `ExtensionOwner::prepare()` 在第二个早退处返回 ⇒ 播放实例**一帧快照都不发布**
⇒ 宿主拿到的是静音。

### 根因：读了一个**不存在**的 take 参数

上一轮把"轨道有 `I_FXEN`，take 也该有对应物"当成了对称事实，于是 take 链的整链总开关读的是
`GetMediaItemTakeInfo_Value(take, "B_TAKEFX")`。

- `reaper.exe` 的字符串表里**没有** `B_TAKEFX` 这个名字（`I_FXEN`、`D_PITCH`、`B_PPITCH`、
  `TAKEFX_NCH` 都在；搜 `B_TAKEFX` 命中 0 次）。
- take 的字段表里也没有整链开关：`TAKEFX_NCH` 只是"per-take FX 用几个内部通道"。
- `TakeFX_*` 一族**没有** `GetChainEnabled`（整张表里只有 `GetCount` / `GetEnabled` / `GetOffline` /
  `GetFXName` / `GetChainVisible` 之类，`GetChainVisible` 是**窗口可见性**，不是开关）。

宿主对**未知**参数的标准答复是 `0.0`。于是"整链总开关 = 0 ⇒ 链被关掉 ⇒ 判旁通"——一个从没被
旁通、也从没离线的实例，被自己的读数判成了静音。这也是为什么它**只在 take FX 下**出现：
轨道侧的 `I_FXEN` 是真的。

### 修法：让"没有这个读数"成为一等取值

`fx_state` 里新增 `ChainGate { Open, Closed, Unknown }`：

- 轨道链：`I_FXEN != 0` ⇒ `Open`，否则 `Closed`（不变）；
- take 链：**恒为 `Unknown`** —— take 侧没有这个读数，就不假装有；
- 判定 `decide()`：`Unknown` 这一位**不投票**。只有"全部离线"或"每个匹配到的 FX 都
  不是启用状态"或"宿主明确说链关掉了"才报旁通。

`TakeChainApi` 因此少了一个字段（不再查 `GetMediaItemTakeInfo_Value`），少一个"宿主返回 0、
我们当真"的机会。判据的其余部分（逐 FX 的 `GetEnabled` / `GetOffline`、类名匹配、`combine()`
的"全员旁通才跳过"、5 s 新鲜度）一个字都没动。

### 顺带把"凭什么"写进日志

结论行从 `[ara] writer 1 host bypass: unknown -> bypassed` 变成

```
[ara] writer 1 host bypass: unknown -> bypassed (take-fx; take chain: found=1/1 enabled=1 offline=0 gate=unknown)
```

四位事实正好把三种成因分开：没有匹配到（`found=0/1`）、用户真的旁通了（`enabled=0`）、
或者像这次一样**读了不该读的东西**（`gate=…`）。仍然只在结论变化时写一行。

## 二、"转换为 take 音高包络"永远失败

现场日志（19:03 那次会话，同一条命令四种触发路径全挂）：

```
[cmd] export_host_take_pitch failed (actor): unbalanced item state
[pitch] auto take-pitch write failed: unbalanced item state
```

### 根因：把宿主的**前缀**当成了整份 item chunk

`GetItemStateChunk(item, buf, sz, isundo)` 能写多少由**调用方给的缓冲区**决定。旧实现第一次
只要 64 KiB，并把"读到 NUL 为止"当成"读完了" —— 而宿主对放不下的两种表现（返回 false，或把
前缀写满后照常返回 true）里，后者会让一份**半截** chunk 被当成完整数据交给解析器：块没闭合 ⇒
"unbalanced item state" ⇒ 每一次读写都在解析阶段失败。

为什么"以前能用、后来不能用"：包络点是随分析长度线性增长的。仓库里的
`tests/fixtures` 那份真实 item chunk（1346 个 `PT` 点）是 **43 KB**，刚好在 64 KiB 之下；而
19:03 那次会话的片段有 **6360 帧** 的 F0（`[pitch_clip] f0_frames=6360`），同一个格式
≈ 6000+ 个点 ≈ **190 KB**，从第一次转换起就必然超过 64 KiB。这不是"偶发"，是**尺寸阈值**。

### 修法：以**结构**判"读完了"

新增 `take_chunk::is_complete_item_state()`：能不能解析成一整份 `<ITEM>` 块（闭合、且 ITEM 之后
没有内容）。读取端（`write::read_complete_item_state`）据此扩容重试，直到 8 MiB 预算：

- 宿主返回 `false`（缓冲区不够）⇒ 扩容重试（旧实现直接报"宿主拒绝读取"）；
- 宿主返回 `true` 但只给了前缀 ⇒ 结构不完整 ⇒ 扩容重试；
- 预算用尽 ⇒ 报错，并写一行 `[ara] the host item state is unusable: <字节数/行数/剩几层没闭合/首尾行>`，
  下一次同类报障不必再从零猜。

剪贴板复制（`capture_item_state`）走同一个读取器，因此"大 item 复制"也一起修好了。

### 根因二：take 标记有四种写法，旧解析只认一种

宿主写的是 `TAKE` / `TAKE SEL` / `TAKE NULL` / `TAKE NULL SEL`（`reaper.exe` 里这四个字面量
就是宿主读取器逐字比较的那张表）。只认裸 `TAKE` 时，**活跃**那条 take 的标记会被当成普通
字段：它的字段区和前一条 take 并在一起，`GUID` 认到上一条 —— 对多 take item 上的 take FX
实例，表现是"宿主 item 里没有这条 take"。

现在的判据是**内容**：只保留"自己带 `GUID`"的字段区。于是"第一条 take 带不带标记"两种宿主
写法都成立，`<ITEM` 与第一个标记之间那段 item 级字段（`POSITION` / `LENGTH` / `IGUID`）也不会
冒充"第 0 条 take"。

另外把**自闭合行**（`<KIND …>`）按"不开不关"处理：RPPXML 允许叶子块写在一行里，把它当
"开"会让栈在文件末尾多一层，整份 chunk 被判成不完整。

## 三、take FX 下 **item chunk 根本写不进去**（真机实测，深一层）

把上面两条修完之后在真机上复测，又发现第三条 —— 它才是"take 音高转换没反应"的最后一道门：

| 场景（REAPER 7.85 实测） | `SetItemStateChunk` 的答复 |
|---|---|
| 普通 item（无 take FX） | `true` |
| item + ReaEQ（普通 VST take FX，在线） | `true` |
| item + 本插件（ARA take FX，在线） | **`false`** |
| item + 本插件（ARA take FX，**offline**） | `true` |

连"把刚读到的 chunk 原样写回"都被拒（`isundo` / undo 块 / 改一个字节，三种写法全 `false`）。
结论：**宿主不允许在 ARA 实例在线时重建这个 item 的 FX 链** —— 而 take FX 放置下，
"要改包络的那个 item"恰好就是挂着本插件的那个 item。

于是任何走 chunk 的写回（take 音高、倒放）在 take FX 下必然失败，现场就是
`[pitch] auto take-pitch write failed: REAPER rejected the item state`。

### 修法：写**包络对象**，不碰 item

真机实测这条路完全可用（本插件在线时照样成功）：

- `GetTakeEnvelopeByName(take, "Pitch")` → 包络对象；
- `GetEnvelopeStateChunk` / `SetEnvelopeStateChunk` 读/整块替换它的点；
- 只有头没有 `PT` 的块 ⇒ 宿主删掉整条包络（"清空曲线"的语义）。

于是 `write_take_pitch_points` 变成三条路径，按"最不打扰宿主"排序：

1. **包络已存在** ⇒ 写包络（两种放置都走这条：不重建 item，也不会顺手重建同一条
   item 上的别的 take FX）；
2. **没有包络** ⇒ 回到 item chunk 的逐字节路径（轨道 FX 下的历史行为）；
3. 第 2 步被拒（take FX 下的**预期**行为）⇒ 借宿主的
   `Take: Toggle take pitch envelope` 动作建一条，再写包络。

【创建为什么只能用动作】ARA 2.0 与 REAPER 的扩展 API 都没有"新建一条 take 包络"的
入口；`GetTakeEnvelopeByName` 在没有包络时返回空指针，而写 chunk 正是被拒的那条路。
动作按**名字**解析（不是硬编码 id），调用前后**记录并恢复全部 item 的选择位** —— 动作
作用于所有选中的 item，不改选择就等于给用户的每条选中片段都加一条音高包络。

## 没动的东西

- 三态 `FxBypass` 与它的编码：`Unknown` 仍然是"照改动前渲染"，仍然是 `AtomicU8` 的默认值。
- `combine()`（"只有全部实例都确定被旁通才跳过整次合成"）与 5 s 新鲜度、心跳采样。
- 逐 FX 的 enabled / offline 判据、类名匹配、"查不到就 `Unknown`、绝不跨链回退"。
- 写回的全部前置校验（GUID 定位、多 take 只改自己那条、除目标块外逐字节保留）。

## 真机复核（REAPER 7.85/x64，本机，2026-11-19 晚）

用 `reaper.exe -newinst -nosplash -new <script.lua>` 直接驱动一个临时实例（不碰用户的工程与配置；
日志用 `HIFISHIFTER_PLUGIN_LOG` 指到临时文件，避免混进用户日志）：

| 读数 | 真机结果 | 说明 |
|---|---|---|
| `TakeFX_GetEnabled(take, 0)` | `true` | 未旁通的 take FX 报告"启用"，判据可用 |
| `TakeFX_GetEnabled(take, -1)` | `false` | "-1 = 整链"这个技巧在 take 侧**不成立**（所以当初才去编了 `B_TAKEFX`） |
| `GetMediaItemTakeInfo_Value(take, "B_TAKEFX")` | `0` | 与一个**我编的**参数返回同一个值 ⇒ 未知参数一律 0，旧代码必判旁通 |
| 多 take 标记（活跃那条） | `TAKE SEL` | 非活跃那条是裸 `TAKE` —— 旧解析只认后者 |
| 插件日志（take FX，修复后） | `writer 1 host bypass: unknown -> active (take-fx; take chain: found=1/1 enabled=1 offline=0 gate=unknown)` | 同一次会话里旧构建打印的是 `unknown -> bypassed` |
| 插件日志（track FX） | `... (track-fx; track chain: found=1/1 enabled=1 offline=0 gate=open)` | 轨道侧 `I_FXEN` 照旧是真读数 |
| 离线渲染（`-renderproject`，take FX 工程） | 3 s / 1 kHz / peak 0.25 / RMS 0.1768 | 与输入逐值一致 ⇒ 不再是"没有声音" |
| 190 KB 包络写回（take FX item） | `SetItemStateChunk = false` ⇒ 见第三节 | 这是第三节那条根因的直接证据 |
| 同一份包络走 `SetEnvelopeStateChunk` | `true`，读回 3 个点 | 修法可行 |

## 仍未复核

1. **倒放**（`source_direction`）在 take FX 下同样只能走 chunk，因此会被同一条拒绝挡住。
   本次只把 take 音高换成了包络路径；倒放要么也找一条不重建 item 的写法，要么在 take FX 下
   明确报"宿主不支持"。现场还没有人报过它，先留在这里。
2. 真实工程（几分钟的素材、上千个包络点、几百 KB 的 ARA 存档）里的一次完整
   "转换 → 保存 → 重开 → 再转换"往返。
