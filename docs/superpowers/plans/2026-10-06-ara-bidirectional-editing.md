# 双向片段编辑实施计划

中文计划，2026-10-06。目标以同日spec为准；不因前置门难以验证缩小为只读/只显示。
本计划不重复此前已获用户验收的全量测试。只批末定向验证，最后集中REAPER验收。

## Batch A：宿主状态与mute

- [x] typed读取item有效mute，按真实region采集并缓存（源码合同通过，实际宿主待验收）。
- [x] GUI状态投影；原子输出门支持实时/离线，不清参数或重新神经推理（真实VST3回调夹具通过）。
- [ ] 初次绑定、宿主回调、GUI关闭、解除mute与item solo覆盖保持正确。
- [ ] 原生字段/实例隔离/输出定向合同及实际REAPER检查。

## Batch B：双向命令与几何

- [x] 锁定官方setter/Undo/UI刷新函数签名，写能力不等同只读能力。
- [ ] native主线程宿主命令调度、请求租约、重入安全和有界拖动合并。
- [ ] 在原GUI移动/跨已接入轨道移动、裁切、长度/正向线性倍率修改。
- [ ] ARA回流更新原timeline/source参数投影；失败回滚/撤销重做协调。

2026-10-06本批：已接typed write能力与真实region→take→item目标、原GUI命令纯规划、
原Clip×Take倍率复用和批次预检、主线程setter/所属project Undo块、native回流读取重试。
bootstrap明确写能力才开放原Kernel拖拽/裁切。五项host_edit定向合同exit0（含旧按键1项），
frontend能力门3项和tsc通过；尚未验证真实REAPER回流。shape/gain/name等未接字段明确拒绝，
不假成功；跨轨写需要目标轨已有真实region，空轨导入将在Batch C补齐。
连续操作Undo组与参数Undo统一、回流时旧参数历史保留仍待实现，不能勾完整Batch B。

## Batch C：导入

- [ ] 原GUI文件选择/拖入和显式目标已接入轨道。
- [ ] 官方item/take/source创建与准确资源所有权；新ARA图ready才显示已导入。
- [ ] Unicode文件名、失败清理与宿主Undo，多clip导入不污染其它项目。

## Batch D：渐变

- [ ] 先在隔离工程验证content-fades委托后宿主不重复fade、item长度仍保留。
- [ ] 长度双向写入；REAPER只保留默认样式，HFS shape/curvature不写回宿主样式。
- [ ] HFS原kernel实际处理自定义渐变，UI/波形/播放与导出一致，缓存层避免重跑神经。
- [ ] 自定义fade按真实region持久化；移动/裁切/拆分/冷重开和两端auto交叉渐变。

## Batch E：集中验收与交付

- [ ] 实际GUI执行导入→编辑→线性拉伸→自定义渐变→播放/导出→撤销/重做→保存冷重开。
- [ ] 2+轨输出隔离、mute/解除mute与图形一致、GUI关闭仍正确、跨工程不误写。
- [ ] Release完整包与安装文件摘要一致；旧版备份，不热替换，不自动push。
- [ ] FINDINGS/ledger记录实测与未决风险，不把定向fixture或图形证明当完整音频通过。

当前：目标已建立，官方mute/相关setter签名已核对；Batch A两项定向合同exit0，
覆盖有效mute/solo覆盖、另一region隔离、实际process实时/离线静音与解除后的原PCM，
零分配和尾哨兵。未声称真实REAPER/关闭GUI门通过；Batch B已接基础写链路，B-E整体尚未完成。
