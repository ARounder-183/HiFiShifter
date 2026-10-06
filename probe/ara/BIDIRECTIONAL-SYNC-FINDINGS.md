# 移动闪回与渐变宽度：2026-10-06

中文记录，明确区分源码护栏与真实宿主结果。

## 本批修复

- 插件自动读取按交互/几何/历史请求代次废弃旧响应，不覆盖拖动预览或在途编辑。
  读取被丢弃时不消费host版本，下次轮询继续同步；独立App的force读取不变。
- Native宿主setter只执行一次，回执等确切clip的目标几何和指定fade宽度回流。
  超时明确失败，不把旧timeline仍可读当作编辑成功。
- 用户允许REAPER执行渐变声音：HFS保留弯曲示意并写回手动/自动宽度，不烘焙未
  委托端、不做未知包络反算。宽度patch携带未改shape/dir时不误触委托错误；插件
  自定义shape/curvature入口暂关闭，独立App保持原功能。
- 真实加载额外发现Undo描述越界不一定是null；改用官方Undo_GetNumEntries枚举
  精确范围和校验跳转，不在timer反复探测10000项。

## 已取得证据

- 前端乱序/宽度失败释放/外部修改/异步重试/独立App初始化：7项通过。
- host_edit：7项通过，含旧可读快照不结束回执、未委托width setter四字段、冗余
  shape不改变宿主样式、重入/身份拒绝。
- host_history：3项通过，fixture越界改为空串指针，验证真实边界不依赖null结束。
- Release02隔离REAPER PID4800冷加载两轨，各region PCMready；原GUI显示两轨和
  REAPER渐变责任提示。真实几何观察初始为0/1秒、长3秒、两端0.25秒，非合成fake。
  自建工程保存到ignored acceptance.RPP。此项证明加载，不是编辑通过。

## 未完成验收

已先鼠标点插件标题、重新激活并按工具指示刷新/重试一次，CU仍拒绝拖动：目标点
属于msedgewebview2.exe的Chrome Legacy Window，不是它可选择的REAPER主窗口。
子窗口也没有出现在可选择窗口列表中。没有绕过工具检查或注入JS冒充GUI手势。
因此真实HFS→REAPER鼠标移动/宽度、混合Undo、导入/拉伸/保存冷恢复的完整矩阵
仍未验收；不把定向合同和原GUI显示成功说成全部完成。

最终包、安装与新Undo API实测结果在本批ledger/PRODUCT-HANDOFF追加。

Release03冷重开PID24760实测`Undo_GetNumEntries`存在（Lua APIExists=true）；原GUI两轨
与PCMready恢复，日志无原先的get_history_state/entry budget错误。宿主将第一clip淡出
从0.25改0.75秒，HFS自动显示2.25至3秒的淡出区，无手工重载；第二clip仍0.25秒。
这是REAPER→HFS长度/图形回流证据，不能替代尚未做成的HFS鼠标反向操作。
