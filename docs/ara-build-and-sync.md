# HiFiShifter App与ARA插件构建及功能同步

中文产品构建说明。当前产品宿主目标为Windows x64/REAPER；用户已将Linux/macOS插件
移出本次要求。独立App的既有跨平台代码保留，不宣称其它平台插件已可用。

## 单一构建入口

在当前checkout根目录的普通PowerShell中运行；本会话使用
`E:\code\HiFiShifter\.worktrees\ara-plugin`，不在主develop工作区执行。

```powershell
.\tools\build-hifishifter.ps1 -Target All -Configuration Release -Name hfs-release-01
```

- 先构建一次同源frontend，生成App的index/detached入口与插件plugin入口。
- App与插件分别调用Cargo，避免特性合并：App保留默认ONNX/vslib，插件只依赖不带
  vslib的共享kernel。禁止把两包放到一次cargo build/test调用来图省事。
- App使用已安装的cargo tauri build生成嵌入前端的便携程序；插件复用已实测的原生
  module_loader与规范VST3 Contents/resources打包。它仍调用probe/ara中的打包helper，
  不是两份产品代码；后续helper可提升到tools而不改入口。
- 本脚本只构建、复制模型/必要DLL，不安装、不修改REAPER扫描路径、不打开宿主。

输入要求：Rust/Cargo、Node/npm（按frontend/package-lock.json安装依赖）、cargo-tauri、
MSVC C++/Windows SDK，以及已经校验过的ARA/VST3 SDK克隆与原模型资源。脚本不自动
安装依赖、下载/编辑SDK。Cargo采用offline/jobs1；SDK的repo/commit/tree校验由桥接
build.rs执行。MSVC加载后重新设置私有TEMP/TMP，避免c1xx.dll的两个已知环境坑。

运行环境还需WebView2 Runtime及Microsoft VC++ x64运行库。当前ORT静态链接在引擎/
App里；实测dumpbin导入包括DirectML.dll与SoundTouchDLL.dll，App另有vslib_x64.dll，
插件无vslib导入。不存在onnxruntime.dll并不自动表示缺运行时，依照实际链接方式判断。

预检和开发构建：

```powershell
.\tools\build-hifishifter.ps1 -Target All -Configuration Release -Name preflight-01 -PlanOnly
.\tools\build-hifishifter.ps1 -Target All -Configuration Debug -Name debug-01
.\tools\build-hifishifter.ps1 -Target Plugin -Configuration Release -Name plugin-01
```

输出位于`.build-tmp/deliveries/<Name>/`：`app/HiFiShifter.exe`及模型/运行DLL，
`HiFiShifter.vst3/Contents/x86_64-win/HiFiShifter.vst3`（loader）、HiFiShifterEngine.dll、
运行DLL和`Contents/Resources/frontend`/`models`。Name必须全新，不能覆盖已存在交付。
插件不含vslib。现阶段是便携目录/bundle，不是签名安装器。
新构建完成时生成build-manifest.json，记录源码commit/源码与模型摘要、每个产物的
SHA256、配置与是否执行源码Verify；nativeAcceptance始终false，不伪报宿主验收。
若构建期间源码或模型变动，流程失败并要求新Name，保留已有产物供诊断，不当作交付。

`-Verify`在构建前集中运行frontend、kernel两种feature配置及所选App/插件回归。
真实模型长源诊断是显式ignored测试，不因普通绿测自动声称通过；REAPER GUI/播放/
保存冷重开/宿主几何验收仍另做一次集中验收。

**当前状态：** PowerShell语法/PlanOnly实测exit0，预检不创建交付目录；真实All Debug
构建paired-build-01、带manifest的paired-build-manifest-02均exit0，包含App、插件规范
bundle及三模型。后者45个文件摘要全部复核无差异；sourceFingerprint记录实际源码/
模型版本，nativeAcceptance与verificationRequested均false。导入DLL和frontend摘要
核对通过。Release、完整Verify及真实用户验收未完成，不把Debug构建当最终验收。

## 功能同步：不复制两套产品实现

| 功能 | 唯一权威源码 | 宿主适配 |
| --- | --- | --- |
| DSP、模型、缓存、状态与编辑核心 | backend/hifishifter-kernel | App设备线程/插件准备worker |
| 原图形界面、时间轴、参数工具、DockRoot | frontend/src | index.html与plugin.html使用同一App |
| 编辑命令/历史/参数语义 | kernel/editor与state | App命令入口与plugin/editor/commands适配 |
| 文件/设备/工程生命周期 | App src-tauri | 插件由ARA宿主供源，不伪装成本地文件 |
| 播放/几何/权限 | hostCapabilities与插件host/render | REAPER真实绑定、宿主只读几何及播放请求 |

修改音高、气声、张力、共振峰或缓存：改共享kernel，然后按两种features一起回归；
不要在src-tauri和plugin各补一份算法。修改GUI：改frontend/src，统一构建会同步产出
两种入口，不另建简化插件GUI。新增命令：共享纯业务，分别接App副作用与ARA权限/
自动应用适配；明确插件中宿主控制或不支持的能力，不能偷偷调用独立App的设备/文件API。

完成一次修改后用`-Target All -Verify`集中构建/回归；插件还需独立原生宿主验证。
改变模型/算法/输入或有效参数时更新真实缓存身份，纯位置/普通fade不重推理；不要仅
凭本地文件路径、clip名字或viewID判缓存。同一工程多个窗口仍共用唯一actor，逐轨
音频只输出其真实分配区域。

## 安装与集中验收边界

关闭REAPER后才替换任何已扫描bundle；推荐先将整个新bundle所在目录加入独立profile
的vstpath64，而不是立即装入系统公共目录。不要只复制内层单文件，它依赖engine、
frontend、模型和运行DLL。不要使用-nonewinst往已有REAPER送脚本，已有项目是用户资产。

构建流不会push、stage文件或保存用户RPP。插件验收需要无独立App、双轨原GUI编辑/
播放控制/音频不串/关闭GUI供音/保存冷重开及普通/自动fade、线性拉伸与BPM同步。
独立App验收则按原导入/编辑/播放/保存路径执行，不能以插件通过代替。
