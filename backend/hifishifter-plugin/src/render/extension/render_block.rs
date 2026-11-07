//! 输出仲裁：从“这一块该出什么声”到“把样本写进宿主平面”的**唯一**实现。
//!
//! 【为什么必须只有一处】这条通路上每一个分支都对应过一次真实报障：
//!
//!   * in-place 宿主上先清输出会把**输入**一起抹掉（“插件在链路里却没有一点输出”）；
//!   * 停播时不清会漏出上一块的残留（“很短、内容不对的一声”）；
//!   * 分层拷贝前清会把要拷的宿主原声删掉；
//!   * 离线流上快照没就绪必须报失败，否则冻结产物里是静音；
//!   * 未知进程形态必须直通，把它当成实时流会污染捕获与去咔状态。
//!
//! 两种插件格式各写一份，就等于让这些结论各漂移一次。所以这里只认
//! [`crate::audio_block::AudioBlock`]（样本面）与
//! [`crate::block_context::BlockContext`]（时间事实）：VST3 与 CLAP 各自把宿主数据
//! 翻译成这两样东西，之后走的是同一段代码。

use super::*;

/// 一块渲染的结局。
///
/// 【为什么不是 `bool`】失败有两种完全不同的含义：**这一块没有产出**（离线流上快照
/// 未就绪 —— 宿主应把这一帧当静音）与**宿主给的数据不可用**（参数非法 / 形状不支持）。
/// 前者是正常状态，后者需要宿主知道；合成一个布尔值会让调用方无法区分。
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RenderOutcome {
    /// 本块已按语义处理完（写满、清零或直通）。
    Done,
    /// 本块没有产出可用音频（离线流上快照未就绪，或缺工程位置）。
    NotReady,
    /// 宿主给的数据不可用（参数非法）。
    Invalid,
    /// 形状不受支持（部分重叠的输入/输出 plane 等）。
    Unsupported,
}

impl ExtensionOwner {
    /// 本块该怎么写进宿主输出。**无锁、无分配**，可从音频线程调用。
    ///
    /// 调用顺序是契约的一部分：
    ///
    /// 1. 纯编辑角色 → 直通（宿主已经处理过的 fade/gain/前级 FX 必须保留）；
    /// 2. 宿主 item 静音 → 淡出（不能直接清零，那是块首的一次阶跃）；
    /// 3. 没有工程位置 → 清空（无从选择输出内容）；
    /// 4. 门控判“不该出声” → 淡出（同样不能直接清零）；
    /// 5. 未知进程形态 → 直通（最诚实的降级）；
    /// 6. Capture 通路 → 它要么自己写完（`Handled`），要么给出“读哪个发布器、
    ///    区间外要不要直通”；
    /// 7. 分层拷贝**之前不能清**（它写满整块，而 in-place 下“输入”就是这块缓冲）；
    /// 8. 写完再做接缝去咔（需要上一块输出的尾巴与这一块的新信号）。
    ///
    /// # Safety
    /// `block` 的平面由宿主提供，非空平面至少含 `frames` 帧；本函数只在本回调内使用
    /// 它们，不保留任何指针。
    pub(crate) unsafe fn render_block(
        &self,
        block: &mut crate::audio_block::AudioBlock<'_>,
        context: Option<crate::block_context::BlockContext>,
        evidence: Option<BlockEvidence>,
    ) -> RenderOutcome {
        if self.is_editor_only() {
            // 纯编辑角色不是歌曲播放 renderer；保留宿主已经处理的 fade/gain/前级 FX，
            // 包括停播监听。
            return match unsafe { block.pass_through() } {
                Ok(()) => RenderOutcome::Done,
                Err(crate::audio_abi::BufferError::InvalidArgument) => RenderOutcome::Invalid,
                Err(crate::audio_abi::BufferError::UnsupportedFormat) => RenderOutcome::Unsupported,
            };
        }
        if self.host_item_muted() {
            // 宿主静音是一次“有声 → 静音”的源切换；直接清零会在块首留下一次阶跃。
            // 与门控清零走同一条去咔路径：从上一块尾巴淡到零，其余帧写零。
            // 该函数自己写满整块，因此**不能**再调 `clear_output`。
            let Some(stream) = ProcessStream::from_mode(block.mode) else {
                // 未知 process_mode：不能冒充实时流去写 capture/去咔状态。
                block.clear_output();
                return RenderOutcome::Done;
            };
            self.apply_gate_fade_out(stream, &mut block.output, block.frames);
            return RenderOutcome::Done;
        }
        let Some(context) = context else {
            // 没有工程位置就无从选择输出内容：静音。
            self.snapshots[0].misses.fetch_add(1, Ordering::Relaxed);
            block.clear_output();
            return if block.is_offline() {
                RenderOutcome::NotReady
            } else {
                RenderOutcome::Done
            };
        };
        // REAPER 停播也会 process 固定光标位置；只离线导出允许没有 kPlaying 的供音。
        //
        // 【为什么这一支必须排在 `capture_stage` **之前**】门控判“要静音”时若已经
        // 写过快照，清输出就清不掉那一块的内容 —— 于是每次停播/移动光标都会漏出一
        // 小块**新位置**的音频。先判、后写，“清输出”才是真的清。
        if !block.is_offline() && !self.capture_should_emit(evidence, context.playing) {
            let stream = evidence.map_or(ProcessStream::Realtime, |evidence| evidence.stream);
            self.note_gate_cleared(stream);
            self.apply_gate_fade_out(stream, &mut block.output, block.frames);
            return RenderOutcome::Done;
        }
        // 【未知 process_mode 必须拒绝】SDK 只定义 0/1/2；把它当成 Realtime 会把
        // 离线/未知流写进实时捕获与门控状态。透传是最诚实的降级。
        if ProcessStream::from_mode(block.mode).is_none() {
            let _ = unsafe { block.pass_through() };
            return RenderOutcome::Done;
        }
        // 【Capture 通路】在快照输出之前插入。它要么自己写完输出（`Handled`），要么
        // 告诉下面“读哪个发布器”以及“区间外要不要直通宿主原声”（`Snapshot`）。
        // 未录制且没有素材时它是 `PassThrough` —— 插件在录制之前是完全透明的，
        // 而不是把整条轨静音。
        let (snapshot_source, layered) = match unsafe { self.capture_stage(block, evidence) } {
            CaptureStage::Handled => return RenderOutcome::Done,
            CaptureStage::Snapshot { source, layered } => (source, layered),
        };
        // 【为什么分层拷贝前不能清】它写满整块（快照范围内取快照、范围外取**输入**），
        // 而 in-place 宿主下“输入”就是这块缓冲本身 —— 先清等于把要拷的东西删掉。
        if !layered {
            block.clear_output();
        }
        // 按 `setupProcessing` 的**权威率**选快照：非原生率（96k/192k…）取最近的
        // 已声明原生快照，`copy_block` 会把它线性重采样到宿主输出率。
        // 原始磁带只有一份（它按捕获时的工程率录制），因此不参与按率选择。
        let publisher = match snapshot_source {
            SnapshotSource::Processed => {
                &self.snapshots[self.snapshot_index_for_rate(context.sample_rate)]
            }
            SnapshotSource::Raw => &self.raw_tape,
        };
        // 输出率取宿主在 processContext 里报的率（那才是宿主缓冲的真实率）；它缺失/
        // 非法时退回 `setupProcessing` 的声明，避免 0 触发无意义的巨量重采样。
        let output_rate = if context.sample_rate.is_finite() && context.sample_rate > 0.0 {
            context.sample_rate
        } else {
            self.setup_sample_rate()
        };
        // 【B0】本块分层拷贝的覆盖读数。两段（循环切分）各自写一份，这里合并。
        let mut layered_probe = crate::render::snapshot::LayeredProbe::default();
        // 【为什么要重新取一次输入视图】Capture 要“录过的那一段被替换、其余保持宿主
        // 原声”，而“范围外”这个区别只在写之前知道。这里不分配：`planes` 只借用宿主缓冲。
        let input = unsafe { block.input.planes(block.frames) };
        let ready = unsafe {
            if layered {
                match evidence {
                    Some(evidence) => {
                        let mut ready = true;
                        for part in evidence.split.iter() {
                            let mut part_probe = crate::render::snapshot::LayeredProbe::default();
                            ready &= publisher.copy_block_layered_at_probed(
                                part.position,
                                output_rate.max(1.0) as u32,
                                &mut block.output,
                                part.input_offset,
                                part.frames,
                                &input,
                                part.input_offset,
                                &mut part_probe,
                            );
                            layered_probe.merge(part_probe);
                        }
                        ready
                    }
                    None => publisher.copy_block_layered_at_probed(
                        context.project_time_samples,
                        output_rate.max(1.0) as u32,
                        &mut block.output,
                        0,
                        block.frames,
                        &input,
                        0,
                        &mut layered_probe,
                    ),
                }
            } else {
                match evidence {
                    Some(evidence) => {
                        let mut ready = true;
                        for part in evidence.split.iter() {
                            ready &= publisher.copy_block_at(
                                part.position,
                                output_rate.max(1.0) as u32,
                                &mut block.output,
                                part.input_offset,
                                part.frames,
                            );
                        }
                        ready
                    }
                    None => publisher.copy_block(
                        context.project_time_samples,
                        output_rate.max(1.0) as u32,
                        &mut block.output,
                        block.frames,
                    ),
                }
            }
        };
        // 【为什么在快照写完以后】接缝去咔需要“上一块输出的尾巴”与“这一块的新信号”
        // 做交叉淡化；写完再混，才不会把新源的第一帧直接硬切到旧源的最后帧。
        let declick_stream =
            evidence.map_or(ProcessStream::Realtime, |evidence| evidence.stream);
        if layered {
            self.note_layered_probe(declick_stream, layered_probe);
        }
        // 【这里曾经有一条“整块直通宿主原声”的保护，现已删除】
        //
        // 它的判据是“读游标跳过了超过 50 ms”，动作是**把整块**（宿主块最大可达 6 s）
        // 换成宿主输入。而直通本身就是换了一个信号源 —— 前后各一次硬切换，用户听到
        // 的那一下“咔”其实是这个保护动作自己。
        //
        // 现在位置只有一个权威（`StreamGeometry` 的 R1–R4），跳变只可能发生在 pass
        // 边界，而 pass 边界本来就要重新开始；接缝去咔的一次交叉淡化就够。
        self.apply_output_declick(declick_stream, &mut block.output, block.frames);
        if block.is_offline() && !ready {
            return RenderOutcome::NotReady;
        }
        // 【为什么 in-place 时要把掩码压平】见 `AudioBlock::in_place`：快照写下的
        // “这一段是静音”会在下一块被我们自己的捕获读成“宿主没给输入”。掩码只是提示，
        // 压平成“非静音”永远是安全的（最坏情况是宿主少省一次算力）。
        if block.in_place {
            block.output.set_silence_flags(0);
        }
        RenderOutcome::Done
    }
}
