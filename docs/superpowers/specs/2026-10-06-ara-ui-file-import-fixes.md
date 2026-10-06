# 原GUI静音、空轨、目录选择和拖入修复目标

中文方案，2026-10-06。用户明确要求设为目标持续完成，重新授权Computer Use集中测试。
本批纳入原双向编辑目标，不假报原目标complete以便新建另一个目标；目标工具拒绝了
重复创建，因为旧目标未完成。后续以本文件和双向编辑spec为当前范围。

1. 静音item仍以clip显示并标为mute，不依赖ARA播放分配；解除后真实参数身份恢复。
2. 已接入HiFiShifter的轨道没有clip时仍显示；实际删除/移除接入后才移除显示。
3. GUI结构按宿主项目变化版本即时更新，静止时不重复扫描全部item；兜底轮询250ms。
   显示不等PCM，GUI占位不进入音频renderer或未经ARA授权读源；不修改独立App路径。
4. 文件浏览器使用系统真实目录选择器（文件系统目录），选择后授权只读列出/搜索/
   元信息及目录导航；错误可见，不开放任意磁盘删除/重命名。
5. Windows Explorer文件拖入通过WebView2允许外部drop、真实File AdditionalObjects/
   File.Path进入REAPER导入，仍等真实item GUID/ARA回流，不发送base64假冒源。
6. begin/end undo group不依赖PCM；显示层总时长/排序不进入音频几何提交；参数和
   几何仍按原共享Undo与保存规则。旧v4参数记录新增可选item GUID，旧档仍读。

验收集中一次：两段同源clip静音仍显示、空HFS轨显示、解除不丢曲线/不出现祖先歧义；
实际选择测试目录列出文件、从浏览器与Explorer拖入，明确新轨/多轨导入位置；复查
混合Undo/保存冷重开与播放输出。定向源码检查只验证本批边界，不反复全量/review。
最终Release完整包在REAPER退出后安装D:\VST，旧包备份，仅本地提交、不自动push。
