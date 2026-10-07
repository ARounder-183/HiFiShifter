# Phase 3a：宿主 PCM 输出与倒放阻塞

中文实测记录，2026-10-04，REAPER 7.81 / Windows x64。此检查点不是修音产品完成。

## 实测

通过 ARA 授权 scope 读取共享源 PCM，按宿主 renderer 分配为各处理器准备不可变快照，
经 VST3 process 输出。官方 action 41051 已执行，宿主 section reader 验证 reversed=true。
三项分别为：2 秒正常区间放在0秒；源0.25秒起裁切1秒放在3秒；2秒真倒放放在6秒。
源为44100Hz mono float WAV，最终宿主导出8秒 stereo 24bit WAV（脚本格式明确，未归一化）。

| 比对 | 最大绝对差 | 结论 |
| --- | --- | --- |
| 正常输出 vs 正常源 | 5.960464477539063e-8 | PASS，无双倍幅度 |
| 裁切输出 vs 裁切参考 | 5.960464477539063e-8 | PASS |
| 间隙 vs 零 | 0 | PASS |
| 倒放输出 vs 倒放源 | 0.5003815367817879 | FAIL |
| 倒放输出 vs 正放源 | 5.960464477539063e-8 | 正向输出 |

**U1 的输出级缺陷成立**：在此真实 REAPER 链路中，宿主没有在插件输出之外补偿倒放。
现有 ARA 模型给同一源、正向 PCM、正时长区间，没有方向输入，插件无法可靠推断。
不能设置一个默认 Clip.reversed 或依赖路径里的文件内容来假装修好。也不能将此结论
泛化成所有 ARA 插件/宿主均不支持倒放。

## 原始证据与复现

- `phase3a-plugin.log` / `phase3a-script.log`：真实绑定、region 分配、host PCM、快照与官方反向确认。
- `phase3a-output.wav`：REAPER 实际导出，SHA256 `27E7FB2FD3CC7353478C8C2139531E592FC76222A5746D7E202704513710113C`。
- `../fixtures/phase3a-asymmetric.wav`：不对称源，SHA256 `2D1DFCDE6CB51A6C1EFF4D6945E7C93CB829E6FE844F5776D2E57AA862525567`。
- `phase3a-output.json`：验证器归一化记录；`reverse_pass=false`，不是 PASS 报告。
- `phase3a-reaper.png`：隔离实例工程截图。
- 采集 DLL SHA256 `DDF0C84B7407F435F1C3A3703A89D5D5A52E94A60684A37E124D96A7DDED24DD`。

运行 `probe/ara/build_product_plugin.ps1`，按 Phase3a plan 的隔离流程启动
`build_phase3a_probe.lua`，之后 `verify_phase3a_output.ps1`。当前 verifier **预期返回错误**
`Reverse output was forward, not reversed`。`test_phase3a_output.ps1` 有6条有效/变异回归通过，
这是验证器正确性，不意味着真实倒放通过。失败的初次未绑定采集保留在 `.build-tmp`。

初次接线误将 controllerRef 读成 instance，宿主绑定失败；对照 ARAVST3.h 与 shim
已纠正为纯不透明键，原测试夹具也修正。最终源码之后又补96k拒绝和sequence迁移回归，
它们未被这份WAV覆盖；不要把此音频证据描述为最终提交每条新行为的宿主验收。

## 已做与没做

已补实际文档销毁/两种释放顺序、editor sequence 通知与展开、scope reader释放、源授权/
撤权回归、共享512MiB预算、30秒块拼接与seek单测、手算双通道重采样oracle、process动态
分配/释放守卫。静态无锁、IO与日志路径另有审查及日志回归。

还没做完整音高推理、本体界面联动、冷启动缓存 miss 保证、持久化/打包、宿主实时seek
与30秒供音压力测试。当前只支持<=30秒、mono/stereo源、44100/48000输出、普通/裁切
区间；时间拉伸/内容淡化返回Unsupported，不对本批先导实现宣称支持。普通重采样是线性
oracle验证，不是最终音质承诺。退役快照保留直到owner销毁，硬预算耗尽后拒绝发布。

## 推进决定

最终源码构建成功，插件测试55条通过（35 lib、13 mapping、1 A5、5 renderer FFI、
1 exports），两套验证器回归各6条通过，diff检查通过。真实WAV验证重跑仍返回exit 1：
`Reverse output was forward, not reversed`。内核/app全套测试未在本批重跑。

按 Phase3a Task15 的判据，**停止完整 v1 的后续实现，不开始 Phase3b/4/5**。
需要先收敛可信方向来源（宿主专用桥或SDK正确表达路径），或明确缩小产品范围为不支持
倒放；这属于产品/宿主契约改变，当前未授权具体变更。源码与证据仅作本地有限检查点。
