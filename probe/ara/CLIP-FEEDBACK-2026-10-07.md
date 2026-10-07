# 本轮实测反馈与修复

## 最新：REAPER粘贴成功，但当前HFS窗口缺少新clip

用户实测`feedback-undo-copy-01`：宿主创建成功，HFS需要重开UI才显示；这不是复制创建
未执行。源码发现get_timeline_state的模型版本缓存早退不检查独立GUI清单代次，后台
清单刷新通知又依赖ARA fade投影成功。媒体写入请求还持续持有Undo，直到ARA/PCM回执。

本批把清单同步提取为独立刷新，在缓存早退前执行；即便ARA暂不可投影也更新结构并通知。
只确认开始时的清单版本，期间的新变化不会被后读到的fade版本吞掉。完成原生创建/删除
后作废清单枚举缓存，并结束原生写入请求；只读回执继续等待真实GUID/几何/授权音频，
不重复创建、不强制重载、不把无音频的占位冒充完整成功。

定向证据：1项Rust用例在同一session暂停idle pump，仅改变inventory版本，新增/删除能
由get_timeline_state读取，已有张力35/45、选择、历史不变。前端竞态文件8项通过，包含
粘贴回执立即显示新clip、旧刷新不移除它、完成后重新接受宿主刷新。首轮Rust测试因旧
fake宿主缺I_GROUPID默认字段abort，补齐夹具后通过；不是实机REAPER崩溃。

`feedback-paste-refresh-02` Release构建通过，实际确认REAPER未运行后完整安装到
`D:\VST\HiFiShifter.vst3`，35文件逐项SHA256一致；旧包可恢复备份位于ignored
`.build-tmp/vst-install-backups/feedback-paste-refresh-02-5194dd09`。
sourceFingerprint=`DC4E81B6F5D5D5AF26414B3F4DF89D52B9DC2CFB75D46B507A3F9C8859890CBA`。
nativeAcceptance仍false；以上不等同真实GUI粘贴已验收，参数Undo/Ctrl拖动仍待实测。

中文记录。最新目标是参数面板Ctrl+Z、Ctrl拖动复制、普通clip复制粘贴，以及低优
父子轨拖动可行性；该目标取代旧“仅缺GUI验收”的判断。

## 已定位并修改源码

- 插件 `clipboard_kind` 只返回param，clip剪贴板被误当成null。已按同一原生协议
  同时识别param与clips，保留参数剪贴板路由，不凭快捷键别名判断媒体类型。
- `Clip`存储省略媒体投影字段。粘贴decode没有normalize_takes，源起点/倍率回到
  默认0/1，导致实际REAPER复制体与回执核对不一致。已恢复Take权威后再做预检。
- Ctrl拖动发 `duplicate_clips_bulk`，旧native入口未接。现直接捕获宿主item状态和
  参数seed，按deltaSec/明确目标轨映射规划，复用真实创建/GUID/ARA回流，不覆盖
  用户系统剪贴板；前端复制批次也加入读取代次保护。
- Dirty通知已调整到可能结束Undo块的tick之前，并新增IComponentHandler2支持情况/
  setDirty结果日志，用于闭合用户实测仍失败的宿主录入环节。参数Undo未宣告修好。

新增非零源起点2.75秒、倍率2.5的真实序列化回归，以及param/clips协议路由回归；
与已有百短clip/无效数据测试一起4项通过，TypeScript检查通过。未打包安装、未
宣称真实REAPER复制或Ctrl拖动通过。当前D:\VST仍是editor-parity-05。

## 父子轨拖动可行性（本轮不实现）

可行：锁定官方SDK提供 `I_FOLDERDEPTH`、`I_FOLDERCOMPACT`、只读 `P_PARTRACK` 和
`ReorderSelectedTracks(beforeTrackIdx,makePrevFolder)`，其中makePrevFolder=1可作为
前一轨的子轨，2可扩展现有folder。不是创建HFS私有groupId即可。

主要风险：ReorderSelectedTracks操作当前工程/全局选择，接口没有project参数，
需要核对所属工程、冻结真实轨道GUID、保存恢复选择并处理原始嵌套folder闭合深度。
同时原HFS parent/root参数组语义不能误套为REAPER音频路由；ARA sequence本身不
提供完整folder父子关系，需要同步真实轨道清单。现handleMoveTrack插件早退仍
保持，不用App的move_track只改GUI。推荐在前三项修复后作为独立低优批次实现。

## 尚未闭合

后续补充源码：键盘copy/cut/paste最终通道检查现在携带动作op（此前缺参数，即使
clipClipboard=true也被拒绝）。参数面板快捷键单独发undo_parameter_edit/redo_parameter_edit，
只恢复HFS参数与可编辑轨道控制字段，不覆盖任何当前clip几何；轨道/全局入口继续
REAPER历史。该焦点语义按最新“参数面板Ctrl+Z”目标采用，不能再声称所有历史统一
同一宿主栈。分组两段曲线一次Undo、Redo回到35/45、clip位置不变的真实actor用例
通过；尚未在实际GUI验证焦点路由。

Ctrl拖动新轨落点不再调用插件未实现的App add_track：向native复制批次传明确新轨
span/映射，由真实宿主创建，原轨序号映射修正为trackId键。cargo check和tsc通过。
同轨/跨轨/新轨真实鼠标操作、冷恢复仍未验证，新包尚未构建，D:\VST仍05。

合并交付更新：`feedback-undo-copy-01` Release构建成功，已在REAPER退出时完整安装
`D:\VST\HiFiShifter.vst3`，35文件/SHA256零差异，旧05包备份至ignored
`.build-tmp/vst-install-backups/feedback-undo-copy-01`。本轮新包包含上述三条源码修复
和参数历史命令，仍未宣称用户真实键盘/鼠标验收通过。

参数Undo需要真实宿主支持/历史录入/恢复证据；新建轨道落点的Ctrl复制、批量几何
与跨轨、冷恢复仍须最终集中验证，不用旧fixture或加载smoke作用户实测成功。
