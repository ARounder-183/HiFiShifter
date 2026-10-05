# 长授权源、原线身份与准备预算

中文记录，2026-10-05。源码/合同证据，不是REAPER完整长人声或GPU验收。本批未启动
REAPER，遵守用户“全部源码做好后集中打开，由用户测”的方式。

## 实现

- 内嵌workspace_snapshot返回Rust timeline与授权SourcePcm Arc，不经旧IPC/JSON复制
  完整PCM。分析WAV只用于原GUI波形/F0，音频继续消费ARA授权源；不从本地文件供音。
- 读取源取消旧30秒常量；仍限44100/48000、mono/stereo、非空/有限PCM、合法帧数，
  checked字节计费后才分配/打开reader。host reader仍每次4096帧，异常释放不发布部分数据。
- 两率最终快照、转交错源副本、混音/region临时输出在重处理前整批计入原512MiB额度。
  Reservation拆分将额度转交实际持有者，余额退出即释放；旧64MiB快照跨度改为同一全局
  额度核对。超预算明确失败，不静默截到30秒或补零。
- 原GUI同进程materialize_with_byte_limit接受已经由源Arc计费的额度，WAV流式写出；
  旧外部materialize仍保留64MiB字节门，不扩大原IPC协议。两者均不再按30秒拒绝音频。
- 私有分析路径只由宿主源ID/完整PCM内容决定，不含model/edit代次、clip位置或名称。
  已有WAV完整逐样本核对后才复用，mtime不变；坏文件从授权PCM以临时文件原子替换。
- 每次真实载入清项目orig就绪key，actor主动从准确路径的clip cache重组。首载即建立
  实际clip的根条目，不能等get_param_frames来创建；这是新回归发现的另一条原线永远
  pending路径。缓存命中或gen0冷恢复/宿主移动也登记render_requested，成功后才清，
  不依赖新的ClipPitchReady或用户再落一笔。

## 实测

- scoped ARA reader完整45秒+17帧，逐样本/尾部相同，reader创建/释放各1次；释放后
  PCM额度回归基线。超大帧数在创建reader前拒绝。
- 真actor载入180秒宿主PCM，workspace Arc与原授权对象地址相同、没有整源复制；
  原GUI片段时长/源帧数正确，分析WAV末样本0.125，关闭后额度回归基线。
- 180秒原kernel普通PCM处理，两率全部样本0.25；48k最后511帧及后2帧静音正确，
  seek回首块正确。source_bytes=31,752,000，ready_bytes=132,624,000，
  本例accounted_peak=265,248,000，hard_limit=536,870,912。
- 1小时稀疏跨度在大分配前BudgetExceeded，额度不变化。**这项是资源拒绝合同，不是
  稀疏长工程支持成功证据。** 当前仍是密集PCM表示，有效区间/共享分块策略待继续。
- 无本地generation的宿主移动自动准备正确新位置，私有分析路径和mtime均保持。
- 同ARA源ID的220Hz→440Hz真实分析回归：orig约MIDI57→69，用户目标MIDI60保留，
  新内容使用新分析路径；没有reload或读取原线命令来推动分析。
- kernel新增WAV复用/损坏重建合同及两项原混音护栏共3项exit0。单独运行无vslib
  kernel测试先暴露历史cfg(test)导入误受vslib门控，修正仅测试import，不改变产品算法。
- 资源批首轮9项中7项通过，2项新增夹具漏name；补完整夹具后2项exit0。
  后续actor/源/fade批32项中31项通过，新增无getter换源门暴露根条目未创建，修后该项
  exit0，相关选择/自动应用/原线pending/无generation移动/fade护栏7项exit0。
- 宿主浮标不再显示过时shape图标：与画布一样报告实际c/S和未校准边界，2文件3项及
  tsc -b exit0。

## 仍开放

265MB是显式PCM额度高水位，**不是进程RSS、模型/GPU、分析STFT或神经cache总峰值**。
HNSEP整段处理仍是用户确认的预期。未实测三分钟HiFiGAN/HNSEP完整宿主链；只移动/
同源换音频的GUI及实际输出、旧v2移位/拆分、跨算法轨还需最后集中native矩阵。

多轨长源原子更新时旧ready快照与新工作域共存，当前额度可能拒绝；不能将单轨180秒
合同外推为全部正常多轨長人声通过。快照共享/稀疏区间与全局资源压力策略、真实RAM/GPU
峰值、旧IPC复制路径的资源边界仍需继续。REAPER7.81任意新淡化曲线oracle未解决，
所有最新四BUG的native门、独立App与最终统一构建/review保持open。
