# 正向原 GUI 全流程：进行中的实测记录

2026-10-04；一次性隔离REAPER工程，非用户原工程。验收未完成。

## 已实测

- 独立app启用custom-protocol、嵌入现有frontend/dist后构建成功。插件另一次cargo单独构建
  成功（避免workspace feature统一带入vslib）；dumpbin依赖无vslib/Tauri/WebView。
- 设置worktree专属WEBVIEW2_USER_DATA_FOLDER后原HiFiShifter GUI持续打开，不是闪退。
  此证据只覆盖当前隔离profile，未证明用户原WebView2 profile已修复。
- 最新GUI连接到真实REAPER实例，显示0秒起2秒片段及3秒起1秒裁切片段、波形与音高编辑器。
  快照为宿主供应的1个源、2个片段；无按persistentID读源文件的回退。
- plugin60测试、IPC4测试新鲜重跑全部通过；大PCM真实管道测试全套共0.08秒。
  历史大帧一次写入超时保留；16KiB分块修复不是小消息通过的推断。

## 用户截图暴露的真实冲突

GUI r0/m4手绘后提交，显示`Conflict: host model changed; refresh`。
插件原日志成功Snapshot后只有两次samples_access=false，没有begin_editing/片段变更。
源码refresh_source_pcm无条件clear_renderers，后者无条件revision++，使单纯reader授权状态
变更错误地令GUI模型版本过期。不能直接去掉所有冲突保护；真实内容/几何版本仍必须守住。
修复尚未验收。禁止通过自动刷新覆盖手绘曲线来掩盖此故障。

用户随后自行操作出现`GUI commit ready revision=1 model=8`；只读IPC检查也返回revision1、
2clips、1source。它证明一次提交已被插件接受，不证明REAPER导出的音频真的改变或持久化。

## 未完成

集中审查：局部renderer整体替换共享编辑、unsupported本地修改被清dirty、临时轨道ID恢复、
compose关闭时手绘pitch被跳过，四项均待定向修复。完整报告在本plan的SDD目录。

真实REAPER导出音高差异、关闭GUI后保存/重开仍保持曲线与输出、同路径改源不能命中过期缓存
尚无证据。倒放不做；stretch/content fades仍不承诺；每源30秒、mono/stereo、44.1/48k限制不变。

用户用Escape停止Computer Use；保留当前GUI/隔离REAPER与用户编辑，不为部署强制终止。
