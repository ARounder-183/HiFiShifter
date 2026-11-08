# REAPER 加载 CLAP 插件即崩溃：根因取证与处置

日期：2026-11-08（晚）  
环境：REAPER 7.85 / Windows x64（`C:\Program Files\REAPER (x64)\reaper.exe`，时间戳 `0x6aece3fb`）  
结论一句话：**REAPER 的 ARA-over-CLAP 私有草案实现不完整，插件一旦应答该扩展，REAPER
在时间线上存在音频时必崩在自己代码里。本仓已改为拒绝该 ID，CLAP 插件在 REAPER 中以
Capture 形态稳定运行。**

---

## 1. 现象

插入 `CLAP: HiFiShifter` 后约 0.1~0.3 秒，REAPER 直接退出。Windows 事件日志：

```
出错应用程序名称： reaper.exe，版本： 7.8.5.0
出错模块名称：   reaper.exe，版本： 7.8.5.0
异常代码：       0xc0000005
错误偏移：       0x000000000005a677
```

多次复现（19:42、20:18、20:43、21:05、21:16 等）**错误偏移完全一致**，说明是确定性
路径，不是内存随机损坏。

## 2. 定位：崩溃发生在 REAPER 内部，且只由 ARA 路径触发

### 2.1 崩溃指令

`reaper.exe` 基址 `0x140000000`，偏移 `0x5a677` → `0x14005a677`：

```asm
14005a63b: mov  rbx, [rbp+0x8]      ; rbx = 对象的 key 字段
14005a646: test rbx, rbx
14005a649: je   0x14005a677         ; key 为 NULL 时……
...
14005a677: mov  rcx, [rbx+0x8]      ; ……仍然解引用 rbx → 读 0x8 → AV
```

VEH 探针实测异常信息：`access=0`（读）、`target=0x8`、`rip=0x14005a677`、
`ThreadId(1)`（主线程）。也就是说：**REAPER 自己的代码在对象为 NULL 时没有做检查**，
这是宿主侧的健壮性缺陷。

崩溃函数（`0x14005a5c0`）是"把对象登记进指针数组（含去重）"的例程，栈回溯给出的
直接调用点是 `0x14005781a`，返回地址落在栈上 `stack[011] = 0x14005781f`。

### 2.2 减法实验：崩溃只与 ARA 相关

所有实验都在隔离的 REAPER 资源目录（`-newinst -nosplash -new -cfgfile …`）里跑，
用同一个 `tone.wav`（48 kHz / 3 s）作为唯一的音频 item。

| 实验 | 配置 | 结果 |
| --- | --- | --- |
| A | CLAP，**无**音频 item | `exit=0` |
| B | CLAP，无音频，强制 `HIFISHIFTER_ORT_EP=cpu` | `exit=0` |
| C | CLAP，**有**音频 item | **`-1073741819` (0xC0000005)** |
| D | CLAP，有音频，强制 CPU EP | **`-1073741819`** |
| E | CLAP，有音频，`HIFISHIFTER_LOG=error` | **`-1073741819`** |
| F | CLAP，有音频，日志全关 | **`-1073741819`** |
| G | **VST3**，有音频 | `exit=0` |
| H | CLAP，有音频，`HIFISHIFTER_LOG=warn` | **`-1073741819`** |
| J1 | CLAP，有音频，ARA 开启 | **`-1073741819`** |
| J2 | CLAP，有音频，**ARA 关闭** | `exit=0` |
| J3 | CLAP，无音频，ARA 关闭 | `exit=0` |

三条结论：

1. **与 DirectML / ONNX 无关**：强制 CPU EP（B、D）结果不变，而有无音频才是分水岭。
2. **与 `clap.log` 宿主转发、日志级别无关**：日志全关（F）照崩。
3. **与 ARA 强相关**：关掉 ARA 立即稳定（J2/J3），且 **VST3 同场景不崩**（G）。

> 上一轮把"日志里出现 `bound=true` 与 `playback_region #0`"当作验收通过，是因为
> 测试工程是空的。**空工程不会走到渲染路径**，所以缺陷被完全掩盖。

## 3. 三次插件侧修补尝试，全部无效

崩溃发生在 `createPlaybackRegion` 之后，因此先怀疑"我们给宿主的接口不全"。

| 尝试 | 假设 | 结果 |
| --- | --- | --- |
| 补齐 ARA 1 遗留字段 `plugInExtensionRef` / `plugInExtensionInterface`（ARA 2 路径下本应为 NULL） | REAPER 仍按 ARA 1 读取 | 仍崩，同一偏移 |
| 强制本实例同时承担 `playback|editor|view` | REAPER 没分配 playback 却要渲染 | 仍崩，同一偏移 |
| 关闭 `samples_access` 时的同步 PCM 读取（避免在宿主回调内重入 `readAudioSamples`） | 重入宿主导致状态破坏 | 仍崩，同一偏移 |

同时用反汇编排除了结构错位：REAPER 在 `0x1400537f0` 读的是
`+0x18/0x20`（playback Ref/Interface）、`+0x28/0x30`（editor）、`+0x38/0x40`（view），
与 ARA SDK 2.3 的 `ARAPlugInExtensionInstance` 布局**逐字段一致**，并且每组都做了
NULL 配对检查。也就是说我们交出的结构体是对的。

## 4. 根因：REAPER 的 CLAP ARA 是单实例半成品

同一个工程、同一份音频，两种格式在 REAPER 里的实例拓扑不同：

| 形态 | REAPER 创建的实例 | 角色分配 | 渲染结果 |
| --- | --- | --- | --- |
| VST3 | **2 个** | 第 1 个 `editor|view`；第 2 个 **`playback`** | `renderer assignment role=1 regions=1`，正常出声 |
| CLAP | **1 个**（`create_plugin #1`，实测无第二次） | `assigned = editor|view`（**无 playback**） | `renderer assignment role=2 regions=0`，随后宿主崩溃 |

即使插件强制把 `assigned` 扩展成 `playback|editor|view`，REAPER 仍然只建一个实例，
仍然在 `createPlaybackRegion` 之后崩在同一偏移。

`com.celemony.ara.plugin_extension.draft/1` 这个 ID **从未出现在任何一版官方
`ARACLAP.h`** 里：ARA SDK 的正式草案是 2023-10-27 的
`org.ara-audio.ara.pluginextension.draft/1`，同日修订为 `draft/2`，2024-06-13
定稿为 `…/2` 并只承诺 `draft/2 → /2` 100% 兼容。REAPER 用的是 2022 年更早的
私有编号，它的宿主实现在"需要真正渲染"这一档上没有走通。

## 5. 处置

**只应答 ARACLAP.h 定稿的 ID，明确拒绝 REAPER 的私有草案 ID。**

- `backend/hifishifter-clap/src/ara.rs`：`get_extension` 的匹配分支只保留
  `org.ara-audio.ara.pluginextension/2` 与 `…draft/2`；私有草案常量仍保留，
  但只用于断言"永不应答"。
- 单元用例 `the_reaper_legacy_id_is_never_answered` 与集成用例
  `the_reaper_legacy_id_is_a_distinct_and_rejected_constant` 把这条钉成可执行事实。
- 黑盒探针 `scripts/clap/probe.c` 与 `scripts/clap/verify.sh` 改为断言该 ID
  **必须缺席**（打印 `ara_reaper_legacy=absent`）。
- 文案：`ara.rs` / `entry.rs` / `abi.rs` / `lib.rs` / 诊断注释、
  `docs/host-extension-boundary.md`、`docs/plugin-formats.md`、
  `docs/plugin-host-survey.md` 与五语 README / 用户手册同步纠正。

### 为什么不是"降级"，而是"不把宿主的缺陷引进来"

- 同一份 ARA 能力仍由 **VST3 形态**在 REAPER 中完整提供（实测稳定，两个实例、playback
  角色正常分配）。
- CLAP 插件在 REAPER 中仍是**完整可用的效果器**：捕获、编辑、渲染输出、状态保存全部
  走与 VST3 相同的那一套共享实现，只是没有 ARA 文档绑定。
- 反过来，若保留应答，用户在 REAPER 里只要时间线上有音频就会被宿主直接崩掉进程 ——
  那是比"没有 ARA"严重得多的可用性事故。

## 6. 真机复测（修复后，隔离 REAPER 7.85）

| 场景 | 结果 |
| --- | --- |
| CLAP + 音频 item | `exit=0` |
| CLAP + 空工程 | `exit=0` |
| CLAP + 音频 item（重复） | `exit=0` |

日志只有正常的 `host log sink installed` → `identity resolved` → `instance initialized`
→ `activated` → `module shutdown`，没有任何 ARA 握手行，也没有崩溃。

## 7. 给后续维护者的提醒

1. **不要**为了"兼容 REAPER"再把 `com.celemony.ara.plugin_extension.draft/1` 加回
   应答分支。用例会失败，而那正是它存在的意义。
2. 验收 ARA 时，工程里**必须有音频 item**。空工程是这条缺陷最好的伪装。
3. 判断"宿主崩在谁身上"，看三点：崩溃模块是不是宿主自身、偏移是否恒定、以及
   把插件侧对应能力关掉后是否消失。三者齐备就应当去反汇编宿主，而不是继续改插件。
4. 需要真正的 ARA 时用 VST3 形态；ARA SDK 定稿协议的 CLAP 宿主（若将来出现）仍会
   走本仓保留的那两个 ID。
