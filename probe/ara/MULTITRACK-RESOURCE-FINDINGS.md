# 多轨就绪快照与整段HNSEP内存诊断

中文一次性诊断记录，2026-10-05。没有启动REAPER，也没有进行Computer Use或中途请求
用户验收。原项目SHA256仍为4AD35908AA2D252D9171A9B6F423E4D9BFEE4DEDBE29861B7665B0FE485897A3。

## 快照改动与证据

- worker逐bit比较左右平面；完全相同才省去右侧存储，实时读取两侧均使用左侧PCM。
  真正立体声/不同signed-zero不合并，不做downmix。先释放缓冲再退额度，不在process
  比较/分配/回收。准备每个renderer后即归并，而不是将全部大结果堆齐后才处理。
- 单region的局部源参数直接消费原kernel输出，省掉另一个全长mixed副本；保留全局
  solo/父链和原加零的位形。无关轨道的HiFiGAN算法不再令当前renderer改变48k路径。
- actor已经按所有版本发布成功时，空闲准备队列清掉历史错误；运行/待办中的任务不
  伪报成功。相同PreparedVersion且两率已就绪的后台重排队直接复用，不再争用预算。
- 真actor双轨各180秒、不同授权源，修改第二轨volume后无需重载：第一轨0.25不变，
  第二轨0.5→0.25；44100/48000尾部511帧准确，越界2帧静音，两轨不串。
- ready_bytes=132,624,000，source_curve_bytes=1,152,000，accounted_peak=497,088,000，
  固定hard_limit=536,870,912。首次断言漏计atlas曲线额度，补reservation身份去重诊断
  后通过；不是降低预算断言或漏算曲线。本例为普通PCM，不外推真实HNSEP多轨峰值。
- 原批13项首轮12过/1额度断言失败；修正曲线计费后该项exit0。包含Mono/真正Stereo/
  signed-zero、读者退役回收、mailbox真实失败、跨算法输出/solo、重叠参数和线性保调。
- 短真实HiFiGAN内容诊断exit0：cold789ms，神经run1，移动额外run0且PCM精确相同，
  改pitch新增run1，HNSEP仍1，48k派生精确。单region省副本没有损坏原模型/cache链。

## 三分钟真实模型：功能成功，旧资源策略失败

Windows32GB RAM机器上，显式CPU、合成谐波源、固定真实F0/目标曲线的冻结插件输入：
HiFiGAN实际多个神经批次，HNSEP始终整段一次，三分钟头/中/尾非静音，两输出率完整，
48k精确派生。暖缓存PCM相同，HiFiGAN/HNSEP额外run均0。这是模型输入诊断，不是
真实ARA读取或GUI参数验收。

旧CPU默认memory_pattern+arena：cold79,639ms / warm3,971ms，neural_runs10，
HNSEP_runs1；accounted_peak331,560,000，但实际working_set17,459,396,608，
peak_working_set18,255,597,568，peak_commit23,231,238,144。

运行中PID45484观测到working17.4GB、private23.1GB、系统free约0.6GB，准备立即停止
这一个已确认的测试进程。停止检查时它已正常exit0，**没有实际执行Stop-Process**。
旧PID49800未触碰；系统free随后恢复约18GB。不能称旧方案长源资源验收通过。

## CPU分配策略修复与仍存在的峰值

核对ort-2.0.0-rc.13的SessionBuilder/ep::CPU绑定：仅关闭memory_pattern不够，未显式
注册CPU时默认arena仍开。对原生ORT Separator关闭pattern，并注册CPU(false)禁用
arena；其它角色保持原策略，Intel macOS的ort-tract保持旧配置。HNSEP未分块，STFT/
mask/ISTFT及图优化/线程数不变。资源策略进入HNSEP内容身份v3，不沿旧冷缓存身份。

短真实mask的旧/新策略逐bit一致，exit0。第二次三分钟使用只监控本次确认exe/PID的
watchdog：working超过10GiB或系统free低于3GiB就结束该一次性测试，不碰其它进程。
PID11532正常exit0、未触发guard；该500ms采样保护**不保证捕获瞬时高水位**。

新策略：cold66,002ms / warm3,807ms，neural_runs10，HNSEP_runs1，warm_extra_runs0；
accounted_peak331,560,000，结束时working_set1,334,362,112，
peak_working_set13,305,835,520，peak_commit13,376,684,032。

已消除推理结束后巨型arena常驻，但瞬时13.3GB仍不适合一般16GB机器无条件运行。
**真实长源资源门仍open**。下一步调查完整HNSEP图的实时中间张量/CPU线程工作区，
在保持整段处理与相同输出的前提下降低峰值，并在推理前建立明确资源保护；不能依赖
外部watchdog作为产品保护、不能宣称CPU额度512MiB包含这些额外模型工作内存。

完整fade曲线oracle、稀疏工程、真实宿主长素材/多轨HNSEP与RAM/GPU矩阵、独立App及
最后一次集中review/build/用户验收也仍开放。当前结论不是技术不可实现，不关闭goal。
