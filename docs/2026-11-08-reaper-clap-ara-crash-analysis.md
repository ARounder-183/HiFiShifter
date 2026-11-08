# REAPER 加载 CLAP 插件即崩溃：根因取证与处置

日期：2026-11-08（晚，第二次修订 22:5x）
环境：REAPER 7.85 / Windows x64（`C:\Program Files\REAPER (x64)\reaper.exe`，时间戳 `0x6aece3fb`）

结论一句话：**REAPER 7.85 的 ARA-over-CLAP 只实现了"向实例要扩展"这一条路径，
从不创建 playback 渲染器实例；插件一旦应答该扩展，REAPER 在时间线上存在音频时
必崩在自己代码里。本仓已改为拒绝该 ID，CLAP 插件在 REAPER 中以 Capture 形态稳定运行。**

---

## 1. 现象

插入 `CLAP: HiFiShifter` 后约 0.1~0.3 秒，REAPER 直接退出。Windows 事件日志：

```
出错应用程序名称： reaper.exe，版本： 7.8.5.0
出错模块名称：   reaper.exe，版本： 7.8.5.0
异常代码：       0xc0000005
错误偏移：       0x000000000005a677
```

多次复现（19:42、20:18、20:43、21:05、21:16、22:35、22:41、22:44 等）
**错误偏移完全一致**，说明是确定性路径，不是内存随机损坏。

## 2. 定位：崩溃发生在 REAPER 内部，且只由 ARA 路径触发

### 2.1 崩溃指令

`reaper.exe` 基址 `0x140000000`，偏移 `0x5a677` → `0x14005a677`：

```asm
14005a63b: mov  rbx, [rbp+0x8]      ; rbx = 要被登记的对象
14005a646: test rbx, rbx
14005a649: je   0x14005a677         ; 为 NULL 时**跳进**登记分支
...
14005a677: mov  rcx, [rbx+0x8]      ; 仍然解引用 rbx → 读 0x8 → AV
```

`je` 的目标恰好是落空地址本身，所以"对象为 NULL"这个判据在机器码层面等于没有：
它只是把空指针原样送进登记流程。**宿主侧缺少空检查。**

崩溃函数（`0x14005a5c0`）是"把对象登记进指针数组（含去重）"的例程，直接调用点是
`0x14005781a`。详见 §4.4。

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

> 更早一轮把"日志里出现 `bound=true` 与 `playback_region #0`"当作验收通过，
> 是因为测试工程是空的。**空工程不会走到渲染路径**，所以缺陷被完全掩盖。

## 3. 四次插件侧修补尝试，全部无效

崩溃发生在 `createPlaybackRegion` 之后，因此先怀疑"我们给宿主的接口不全"。

| 尝试 | 假设 | 结果 |
| --- | --- | --- |
| 补齐 ARA 1 遗留字段 `plugInExtensionRef` / `plugInExtensionInterface`（ARA 2 路径下本应为 NULL） | REAPER 仍按 ARA 1 读取 | 仍崩，同一偏移 |
| 强制本实例同时承担 `playback|editor|view` | REAPER 没分配 playback 却要渲染 | 仍崩，同一偏移 |
| 关闭 `samples_access` 时的同步 PCM 读取（避免在宿主回调内重入 `readAudioSamples`） | 重入宿主导致状态破坏 | 仍崩，同一偏移 |
| 无条件下填遗留字段（不再受 API 代际约束） | 首次尝试被代际门拦住 | 仍崩，同一偏移 |

第二条是把"接口全给"这条路走到底：日志确认交出的结构体是
`playback=true editor=true view=true`，三个接口指针**全都非 NULL**，
宿主仍然崩在同一条指令上。这说明它根本没有从我们交出的结构体里取那个对象（见 §4.4）。

同时用反汇编排除了结构错位：REAPER 在 `0x1400537f0` 读的是
`+0x18/0x20`（playback Ref/Interface）、`+0x28/0x30`（editor）、`+0x38/0x40`（view），
与 ARA SDK 2.3 的 `ARAPlugInExtensionInstance` 布局**逐字段一致**，并且每组都做了
NULL 配对检查。也就是说我们交出的结构体是对的。

## 4. 根因：REAPER 的 CLAP ARA 从不创建 playback 渲染器实例

### 4.1 实例拓扑：VST3 两个，CLAP 一个

同一个工程、同一份音频，两种格式在 REAPER 里的实例拓扑不同：

| 形态 | REAPER 创建的实例 | 角色分配 | 渲染结果 |
| --- | --- | --- | --- |
| VST3 | **2 个** | 第 1 个 `editor|view`；第 2 个 **`playback`** | `renderer assignment role=1 regions=1`，正常出声 |
| CLAP | **1 个** | `assigned = editor|view`（**无 playback**） | `renderer assignment role=2 regions=0`，随后宿主崩溃 |

### 4.2 实测：`create_plugin` 全程只被调用一次

给 `clap_plugin_factory.create_plugin`、entry 级 `get_factory` 与实例级
`get_extension` 加临时日志后复测，一次完整会话的全部请求如下：

```
[clap] entry get_factory      id=clap.plugin-factory        （×3，扫描/加载）
[clap] create_plugin          id=com.hifishifter.clap       （×1，仅此一次）
[clap] instance get_extension id=com.celemony.ara.plugin_extension.draft/1
[clap] instance get_extension id=clap.params / clap.gui / clap.tail / …
```

两条硬事实：

1. **`com.hifishifter.clap` 只被创建过一次**，之后的整个会话里再没有第二次
   `create_plugin`。REAPER 不会为 playback 渲染器补建实例。
2. **REAPER 从未向 entry 级 `get_factory` 要过任何 ARA 工厂 ID**，它只问
   `clap.plugin-factory`。也就是说它连"这个模块提供哪些 ARA 工厂"都没看过。

### 4.3 宿主二进制里只有一个 ARA-CLAP 字符串

对 `reaper.exe` 做字符串扫描，与 ARA/CLAP 相关的只有一个扩展 ID：

```
0xc26f44  com.celemony.ara.plugin_extension.draft/1
```

没有 `org.ara-audio.ara.*`，也没有任何 entry 级工厂 ID。因此 REAPER 7.85 的
CLAP ARA 只有"向实例要一个扩展"这一条路径，没有配套的工厂发现与多实例创建。

而 `ARACLAP.h` 对 `bind_to_document_controller` 写得很明确：

> This may be called only once during the lifetime of the CLAP plug-in …
> \p assignedRoles describes the roles that **this specific instance** will fulfill.

一个 `clap_plugin` 实例只能承担一组角色；要同时具备 `playback` 与 `editor|view`，
宿主必须创建**两个**实例。REAPER 只创建一个，却仍按"需要渲染"继续往下走。

`com.celemony.ara.plugin_extension.draft/1` 这个 ID 也**从未出现在任何一版官方
`ARACLAP.h`** 里：ARA SDK 的正式草案是 `org.ara-audio.ara.pluginextension.draft/1`，
同日修订为 `draft/2`，2024-06-13 定稿为 `…/2` 并只承诺 `draft/2 → /2` 100% 兼容。
REAPER 用的是 2022 年更早的私有编号。

### 4.4 崩溃现场：REAPER 把 NULL 当有效渲染器登记

崩溃瞬间的完整寄存器现场（Win32 调试 API 抓 first-chance 异常）：

```
Rip=000000014005a677   Rbx=0000000000000000   Rax=1   Rdi=1
Rbp=0000000003749ac0
  [rbp+0x00]=0000000002edcc30
  [rbp+0x08]=0000000000000000   ← 空对象，§2.1 的 rbx
  [rbp+0x10]=0000000002ec3bc0
  [rbp+0x18]=0000000000000018
异常参数：access=0（读），target=0000000000000008
```

调用方 `0x14005781a` 处在一个"逐个渲染器分配播放区域"的循环里：

```asm
140057612: mov  edx, [r14+0x0c]     ; 渲染器数组长度
140057642: mov  r15, [rcx+rax*8]    ; 当前渲染器
...
1400576d0: mov  rax, [rbp+0x80]     ; 当前渲染器包装对象
...
140057807: mov  rdx, [rbp+0x80]
140057814: mov  r9d, 1
14005781a: call 0x14005a5c0         ; 把新区域登记到该渲染器名下 → 崩
```

CLAP 形态下这个包装对象的 `+0x08`（底层渲染器对象）**从来没被填过** ——
因为没有任何实例以 `playback` 角色绑定过。这就是 §3 第二条实验的解释：三个接口全填
也没用，那个字段不是从我们交出的 `ARAPlugInExtensionInstance` 里取的，而是宿主在
"某实例以 playback 角色绑定"时自己写进去的。

### 4.5 因果链

1. REAPER 为插入点建 **1 个** CLAP 实例，并以 `assigned = editor|view` 绑定；
2. 它同时为这份文档准备了一个 playback 渲染器槽位（`[rbp+0x80]`），但没有任何实例
   去填它的底层对象；
3. 加载音频 item → `createPlaybackRegion` → 分配区域 → 循环走到那个空槽位；
4. 登记例程对 NULL 不做检查 → `mov rcx,[0+8]` → `0xC0000005`。

## 5. 处置

**只应答 ARACLAP.h 定稿的 ID，明确拒绝 REAPER 的私有草案 ID。**

- `backend/hifishifter-clap/src/ara.rs`：`get_extension` 的匹配分支只保留
  `org.ara-audio.ara.pluginextension/2` 与 `…draft/2`；私有草案常量仍保留，
  但只用于断言"永不应答"。
- 单元用例 `the_reaper_legacy_id_is_never_answered` 与集成用例
  `the_reaper_legacy_id_is_a_distinct_and_rejected_constant` 把这条钉成可执行事实。
- 黑盒探针 `scripts/clap/probe.c` 与 `scripts/clap/verify.sh` 改为断言该 ID
  **必须缺席**（打印 `ara_reaper_legacy=absent`）。
- 诊断：`create_plugin` 每次建实例记一行，实例级 ARA 扩展查询按 ID 前缀记一行。
  "宿主从没建第二个实例"与"宿主从没问 ARA"这两种现场，靠这两行即可当场分辨。
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

## 6. 真机复测（隔离 REAPER 7.85）

| 场景 | 结果 |
| --- | --- |
| CLAP + 音频 item（修复后） | `exit=0` |
| CLAP + 空工程（修复后） | `exit=0` |
| CLAP + 音频 item（重复） | `exit=0` |
| CLAP + 音频 item + 播放 | `exit=0` |
| 临时放行私有 ID（对照） | `-1073741819`，偏移恒定 `0x5a677` |

修复后的日志只有正常的 `host log sink installed` → `identity resolved` →
`instance initialized` → `activated` → `module shutdown`，
没有任何 ARA 握手行，也没有崩溃。

## 7. 给后续维护者的提醒

1. **不要**为了"兼容 REAPER"再把 `com.celemony.ara.plugin_extension.draft/1` 加回
   应答分支。用例会失败，而那正是它存在的意义。
2. 验收 ARA 时，工程里**必须有音频 item**。空工程是这条缺陷最好的伪装。
3. 判断"宿主崩在谁身上"，看三点：崩溃模块是不是宿主自身、偏移是否恒定、以及
   把插件侧对应能力关掉后是否消失。三者齐备就应当去反汇编宿主，而不是继续改插件。
4. **判断宿主 ARA 实现是否完整的唯一外部读数，是"它为一篇文档建了几个插件实例"。**
   `ARACLAP.h` 规定一个实例只承担一组角色；只建一个实例却要求多角色的宿主，
   必然在渲染路径上缺东西。日志里 `create_plugin` 的行数就是这条读数。
5. 需要真正的 ARA 时用 VST3 形态；ARA SDK 定稿协议的 CLAP 宿主（若将来出现）仍会
   走本仓保留的那两个 ID。

### 附：复现取证的两种手段

- **崩溃现场**：用 Win32 调试 API（`CreateProcessW` + `DEBUG_ONLY_THIS_PROCESS` +
  `WaitForDebugEvent` + `GetThreadContext`）抓 first-chance 异常，能拿到完整寄存器
  与栈；插件内的 VEH 探针只能拿到 `rip/target/access` 三项。
- **反汇编**：`reaper.exe` 的 `.text` 是 RVA `0x1000` / raw `0x400`，
  `VA → 文件偏移 = VA - 0x140000000 - 0x1000 + 0x400`。用 capstone 按窗口反汇编
  即可读出判据，无需符号。
