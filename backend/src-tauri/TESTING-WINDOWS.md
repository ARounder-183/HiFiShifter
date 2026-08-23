# 后端测试运行说明（Windows 本机）

## 问题

`cargo test --lib` 的测试可执行文件启动即崩溃（`STATUS_ENTRYPOINT_NOT_FOUND`,
0xC0000139），所有单元测试无法运行。

## 根因

测试 exe 隐式导入了 `comctl32.dll` 的 v6 独有符号（如 `TaskDialogIndirect`，
来自 winit/tauri 的拖拽与对话框代码）。主程序由 tauri-build 内嵌了声明
Common-Controls v6 的 manifest，因此正常运行；而 **cargo 测试可执行文件没有
manifest**，加载器把 comctl32 解析到 System32 的 5.82 旧版（无这些导出），
进程加载阶段直接失败。

这与代码改动无关（对未修改的分支同样复现），纯属本机构建产物差异。

## 解决方案

### 单元测试（--lib）：内嵌 manifest

链接测试二进制时内嵌一份声明 Common-Controls v6 的 manifest：

```powershell
$env:RUSTFLAGS = "-C link-arg=/MANIFEST:EMBED -C link-arg=/MANIFESTINPUT:$env:TEMP\hifishifter_test.manifest -C link-arg=/MANIFESTUAC:NO"
cargo test --lib
```

manifest 内容（保存为 `%TEMP%\hifishifter_test.manifest`）：

```xml
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">
  <assemblyIdentity version="1.0.0.0" processorArchitecture="*" name="HiFiShifter.tests" type="win32"/>
  <dependency>
    <dependentAssembly>
      <assemblyIdentity type="win32" name="Microsoft.Windows.Common-Controls" version="6.0.0.0" processorArchitecture="*" publicKeyToken="6595b64144ccf1df" language="*"/>
    </dependentAssembly>
  </dependency>
</assembly>
```

注意：
- RUSTFLAGS 变化会使构建缓存整体失效，首次运行会全量重编译。
- **不要**带着这套 RUSTFLAGS 去跑其它目标（bin/集成测试）——主程序已由
  tauri-build 内嵌自己的 manifest，重复传入会触发 `LNK1123`。

### 集成测试（tests/）：使用 __test-internals feature

```powershell
cargo test --test loop_semantics --test smoke --features __test-internals
```

（见 src/lib.rs 的 `__test_internals` 模块说明。）

## 附带坑：deps 目录下的陈旧 DLL 阴影副本

`target\debug\deps\` 下若残留旧版 `SoundTouchDLL.dll` / `vslib_x64.dll` /
`DirectML.dll`，会因"exe 所在目录优先"的 DLL 搜索顺序遮蔽
`target\debug\` 下 build.rs 新拷贝的同名 DLL，同样导致加载失败。
遇到入口点错误时优先检查这些阴影副本的时间戳是否落后于
`target\debug\` 根目录。
