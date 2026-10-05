# HiFiGAN / HNSEP 缓存与长素材批次记录

中文工程记录，2026-10-05；仅 ara-plugin worktree。本文件记录源码/真实CPU模型诊断，
不是 REAPER 最终验收报告，不修改用户工程，不部署/热替换插件。

## 用户范围

HNSEP **整段处理，不分块**，这是用户确认的预期；HiFiGAN 必须支持长 clip 分块。
只做线性拉伸；倒放、非线性 stretch markers/tempo warp 不在本轮。
减少冗余 review，只最终集中一次；验证合并到功能批末，失败只定向重测。

## 本批实现

- HNSEP 保留原整段 STFT→mask→ISTFT。缓存键改为完整实际 mono PCM、采样率、
  已加载模型完整 BLAKE3 摘要、算法版本与实际 EP/设置代次；不靠 clip/view 名或
  head/tail 指纹决定内容。相同内容可跨 owner/声道共享，不同平面由真实内容区分。
- HNSEP 在已有串行 separator 的 worker 路径合并重复请求，cache 复查在推理之前；
  默认128条和128MiB双边界，扩大条数不会扩大字节上限；超预算结果不留缓存，
  有效结果仍正常返回。移除未被任何代码调用、无法证明完整内容的旧 priming 接口。
- HiFiGAN 默认512 mel帧每块保持，最多4块一批，立即拼入输出；不再为所有未命中块
  同时建立 mel/F0 与波形副本。尾块不补到整块长度，保留原模型时间语义；坏缓存重算，
  新输出长度/非有限值异常则拒绝该批写缓存。pipeline版本8使旧磁盘PCM失效。
- HiFiGAN原无上限chunk HashMap换为128MiB字节LRU，替换/失效/清理准确回收计数。
- 插件含HiFiGAN的已编辑工作区在44.1k合成一次，48k快照由原生就绪PCM派生，
  不再次运行HNSEP/神经网络。非HiFiGAN输出路径保持旧契约；交错源副本新增先收费。
- 重叠region选区修复已在 e15a0792：GUI只换可见源投影，编辑delta不回捕覆盖不可见素材。

## 实测：Windows x64 / debug / CPU / 合成谐波输入

真实模型而非fake网络，缺模型的诊断会失败，不返回skip。CPU设置只作用于测试进程。

模型SHA256：

- HNSEP：`0B46499D71799A3B47A060997F26E94C513776461BBBF9D592A4B5AF8DC8A80C`
- HiFiGAN：`C0D5D05B6CFE12E254C82ADEED6E52A31449A2D1398E05E1D7234F7DBFD720BB`

| 诊断 | 实际结果 |
| --- | --- |
| HNSEP 1秒，并发两个不同owner请求 | 375ms，1次推理/1次命中，同一stem Arc；缓存352800 bytes |
| HNSEP 原生率 h+n 重建 | 最大残差 `9.313226e-10` |
| 相同ID/长度/旧64位指纹，仅中间样本改变 | 新增1次推理，不复用原stem |
| 宿主PCM HiFiGAN四专有参数矩阵 | 1474ms；关开关时气声/张力曲线被剥离；开后HNSEP仅1次 |
| 气声增益0→2 | `(wet-dry)-2*noise` 最大误差 `2.9802322e-8`；没有新增分离 |
| 张力75 / 共振峰600 cents | 相对零气声基线平均PCM差 `0.075894184` / `0.10104014`；HNSEP复用 |
| HiFiGAN 35秒、48k长素材 | 6块、批上限4、尾块454帧；冷7935ms、暖1078ms、暖新增推理0 |
| 长素材完整性 | 精确1680000样本，每秒非静音，全部有限；暖PCM与冷PCM逐样本相同 |
| 长素材接缝 | 接缝最大相邻跳变0.02229233，整段相邻跳变p99=0.036890112 |
| 真实插件RenderInput双输出率 | 798ms，HNSEP仅1次；22050/24000样本，48k左右平面等于原生结果派生 |

### 正常退出的批末结果

- plugin重叠/选择/原actor/源迁移/范围恢复：10 passed、exit0。
- kernel内容键/字节预算/块批次/损坏/声道缓存：7 passed、1 ignored、exit0。
- 显式kernel真实模型三个诊断：3 passed、0 ignored、exit0。
- 显式plugin真实双输出率诊断：1 passed、exit0。
- plugin普通RenderInput相关：3 passed、1 ignored、exit0。

命令遵循 MSVC→重设TEMP/TMP→SDK，cargo `--offline --jobs 1`。
真实模型命令：kernel `--lib -- real_model_ --ignored --test-threads=1 --nocapture`；
plugin `--lib -- real_model_hifigan_snapshots --ignored --test-threads=1 --nocapture`。

首个选择fixture忘设帧周期而失败，修fixture后定向通过。另一次默认EP短filter虽打印
成功摘要，进程未正常退出；不是Green。仅终止本轮自有cargo父进程（40580/44544），
未触碰旧49800或任何REAPER；释放锁后仅重跑被LNK1104挡住的plugin诊断，正常exit0。

## 仍未完成，不可外推

- **35秒诊断是kernel真实模型，不是ARA源采集。** `render/source.rs`仍有旧30秒cap。
  正常完整人声的端到端宿主读取/缓存/快照与512MiB资源方案仍须实现，不能删cap假通过。
- HiFiGAN仍提取完整mel，HNSEP整段中间频谱/网络激活也仍按长度增长。两类模型cache
  各128MiB是kernel额外保留上限，不算已被plugin512MiB reservation统一覆盖；驱逐后
  活worker的Arc仍可能持有stem。没有本批RAM/GPU峰值，不能宣称常量总内存。
- HiFiGAN现有chunk/segment键仍有clip与项目时间依赖；跨owner、移动/fade-only神经
  命中和完整model/config身份、磁盘缓存/冷重开/取消矩阵仍需收尾。HNSEP绑定的是
  **实际已加载模型**，不提供在运行时替换同路径文件的自动hot reload。
- 参数矩阵验证的是真实宿主PCM共享kernel路径，不是完整原GUI或真实人声听感验收；
  共通参数/clip formant morph/复杂轨路由/多轨长素材仍纳入最终门。
- 外部一期IPC重叠提交仍有旧capture入口；内嵌原actor选择路径已有源码证据。
- 新源码没有生成规范release bundle，也未重新在REAPER验收。旧host-stretch-01不包含
  本批及此前workspace/atlas代码。独立App完整回归和最终集中review仍未执行。

下一批：完成HiFiGAN内容/模型权威和移动复用，再落实长源宿主资源策略；不要重复本批
已完成矩阵或重新研究HNSEP分块。最后才集中构建和用户恢复Computer Use后的native验收。
