# 原GUI修复实施批次

中文计划，2026-10-06；按同日ara-ui-file-import-fixes spec执行，不缩小原双向验收。

- [x] 官方parent轨道/item/take/GUID独立GUI清单、预算/逐调用授权/模型版本去重。
- [x] 静音clip/空轨显示，GUI-only占位不进DSP、真实源GUID跨region重建保留参数。
- [x] live空成员保留曲线/持久身份，冷恢复空/歧义身份仍拒绝；显示几何与音频提交分开。
- [x] 真实目录选择器和选定目录只读授权，目录列表/搜索/元信息；选择器错误可见。
- [x] WebView2 external drop明确开放；Undo group不再因音频未ready而失败。
- [ ] 当前最终源码定向合同、生产前端与Release构建。
- [ ] 集中REAPER原GUI静音/空轨/目录/拖入/Undo/冷重开验收，失败则修，不把代码当实机。
- [ ] 完整安装D:\VST并核对35文件摘要，备份旧包，文档/ledger/仅本地提交。

01包实机已显示两个灰色mute clip和空第三轨；同时暴露live空身份及显示总时长误入
音频校验，后续源码已修正，需在最终包复验。此前截图仍是01旧包，不假报最终通过。

用户续报三项：mute clip无波形、拖入后要重开窗口才显示、seek播放头ABAB。当前源码
保留已获ARA授权生成的显示波形（不将显示缓存用于DSP），take更换/删除失效；未曾
得到宿主样本的冷启静音源仍不能凭文件路径重读。导入操作纳入读取代次屏障，自动
crossfade的旧读取不得删除新clip；native创建后立即采样清单并通知。seek在途和结束
双向废弃旧轮询，native seek后立即采样实际光标，插件不以旧echo纠正用户光标。
前端乱序6项/tsc、mute显示波形1项及此前UI清单3项/工作区9项/目录授权1项通过。
最终waveform-import-seek-fix-01 Release已构建；真实GUI全部门仍未复验，不假称完成。
