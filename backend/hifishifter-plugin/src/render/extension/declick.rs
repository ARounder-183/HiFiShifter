//! 输出接缝判定与去咔（`apply_output_declick` / `apply_gate_fade_out`）。
use super::*;
impl ExtensionOwner {
    /// 记录本块输出源，并做接缝去咔。
    ///
    /// 【为什么在写完输出之后调用】先用新源写满整块，再把前 `OUTPUT_DECLICK_FRAMES`
    /// 帧与"上一块输出的尾巴"交叉淡化；尾巴就是用户刚刚听到的旧源信号，与旧源连续，
    /// 不需要再持有旧快照。所有状态都是原子 + 预分配缓冲，音频线程不分配、不取锁。
    ///
    /// # Safety
    /// `output` 必须是本回调内有效的 VST3 输出总线；`frames` 不超过其平面长度。
    /// 从一个源切到另一个源时，是否应该**淡入**而不是交叉淡化。
    ///
    /// 【为什么看两侧而不只看上一侧】透传源（宿主原声）与处理结果**不同源**：前者是
    /// 宿主正在播的东西，后者是插件算出来的东西。两者交叉淡化叠出的是"两声并发"，
    /// 不是一次接缝 —— 那种场合必须淡入。但**两个处理结果之间**（Raw ↔ Processed）
    /// 同样是不同源：Raw 是原始捕获，Processed 是神经声码器的输出，内容毫不相干。
    /// 只看上一侧会把 `Processed → Raw` 判成"同源切换"，于是 64 帧里把两段不相干的
    /// 信号线性叠在一起 —— 用户听到的随机"咔"正是它。
    ///
    /// 因此判据是**异源**：只要两侧有一个是透传类，或者两侧不属于同一个"已算出的"
    /// 内容族（Raw 与 Processed 分属不同族），就淡入。
    pub(super) fn seam(previous_kind: u8, kind: u8) -> Seam {
        match (Self::family(previous_kind), Self::family(kind)) {
            // 两侧都是宿主原声：同一份信号，短交叉就够。
            (ContentFamily::Passthrough, ContentFamily::Passthrough) => Seam::Crossfade,
            // 任一侧是透传 ⇒ 异源：另一侧是我们算出来的内容，叠起来就是"两声并发"。
            (ContentFamily::Passthrough, _) | (_, ContentFamily::Passthrough) => Seam::FadeIn,
            // Raw 与 Processed 是**同一段素材的两种处理**：内容相关但相位不同，
            // 该用 20 ms 等功率交叉，而不是 1.3 ms 的淡入。
            (ContentFamily::Raw, ContentFamily::Processed)
            | (ContentFamily::Processed, ContentFamily::Raw) => Seam::WideCrossfade,
            // 同族、同源、只是版本变了：内容几乎相同，短交叉即可。
            _ => Seam::Crossfade,
        }
    }

    /// 输出源的内容家族。接缝判据只关心"这一侧是什么内容"，不关心具体是哪一档。
    ///
    /// 【为什么收敛成三档】`CaptureOutput` 有七个取值，但接缝只需要回答一个问题：
    /// 两侧的信号**是否同源**。透传类（宿主原声）与算出来的内容之间必须淡入；
    /// Raw 与 Processed 是同一段素材的两种处理，可以交叉；其余同族切换同理。
    /// 按七个取值逐个写分支，每加一档就要重审一遍全部组合 —— 那正是这里曾经出过
    /// 两次错的原因：`Processed` 与 `Raw` 曾被当成"同源切换"，于是 64 帧里把两段
    /// 不相干的信号线性叠在一起。
    pub(super) fn family(kind: u8) -> ContentFamily {
        if kind == CaptureOutput::Raw.code() {
            ContentFamily::Raw
        } else if kind == CaptureOutput::PendingSynthesis.code()
            || kind == CaptureOutput::PassThrough.code()
            // 录制监听送出去的也是**宿主原声**：`Recording` 分支把输入总线原样写出，
            // 它从哪一相位开始是宿主的事，我们不该给它加淡入（那会把录制监听的前
            // 5 ms 削掉）。
            || kind == CaptureOutput::Recording.code()
        {
            ContentFamily::Passthrough
        } else {
            ContentFamily::Processed
        }
    }

    /// 按接缝种类装填一次淡化：抓下旧信号、定宽度、复位进度。
    ///
    /// 【为什么宽度也进状态】三种接缝的宽度不同（1.3 ms / 1.3 ms / 20 ms），而淡化
    /// 可能跨块继续。把宽度存进原子量之后，淡化的**每一块**都读到同一个值。
    pub(super) fn arm_seam(&self, state: &OutputDeclickState, seam: Seam) {
        let len = match seam {
            Seam::FadeIn | Seam::Crossfade => OUTPUT_DECLICK_FRAMES,
            Seam::WideCrossfade => OUTPUT_WIDE_CROSSFADE_FRAMES,
            Seam::FirstBlock => OUTPUT_FIRST_BLOCK_FADE_FRAMES,
        };
        // 淡入 = 把旧信号掩成 0；交叉淡化保留旧信号。
        let keep = match seam {
            Seam::FadeIn | Seam::FirstBlock => 0,
            _ => u32::MAX,
        };
        // 【等功率只在"两条不同源"时用】FadeIn / FirstBlock 的旧侧恒为 0，等功率
        // 退化成一条两端平缓的斜坡（比线性更平滑）；WideCrossfade 两侧是互不相关的
        // 信号，线性会在中点掉 6 dB。Crossfade 两侧是**同一份内容**的两次渲染
        // （相关性接近 1），等功率会在中点叠出 +3 dB 的鼓包 —— 那种场合线性才对。
        state
            .fade_curve
            .store(u32::from(seam.uses_equal_power()), Ordering::Release);
        // 旧信号是"上一块输出的最后 `len` 帧"，在历史缓冲里位于末尾。
        let base = DECLICK_HISTORY_FRAMES - len;
        for channel in 0..2 {
            for index in 0..len {
                let old = state.tail[channel][base + index].load(Ordering::Relaxed);
                state.fade_from[channel][index].store(old & keep, Ordering::Relaxed);
            }
        }
        state.fade_len.store(len as u32, Ordering::Release);
        state.fade_pos.store(0, Ordering::Release);
        state.fade_active.store(true, Ordering::Release);
    }

    pub(crate) unsafe fn apply_output_declick(
        &self,
        stream: ProcessStream,
        output: &mut crate::audio_abi::AudioBusBuffers,
        frames: usize,
    ) {
        if frames == 0 || output.num_channels <= 0 || output.channel_buffers.is_null() {
            return;
        }
        let token = self.output_token[stream.index()].load(Ordering::Acquire);
        let Some(state) = self.declick.get(stream.index()) else {
            return;
        };
        // 【为什么按"种类"判首块】种类变化（Raw ↔ Processed、Processed →
        // PendingSynthesis）才是用户听得出的接缝，同种类的版本更新（曲线编辑后重渲）
        // 由下面的 `last_token` 分支处理。第一块的种类若与本流已知的最后种类不同，
        // 说明上一轮播放停在另一种源上 —— 那一跳正是"咔声位置不固定"的来源之一。
        let kind = (token & 0xFF) as u32;
        // `None` = 本流还没输出过任何一块（真首块）；`Some(prev)` = 上一块的种类。
        // 【为什么两行而不是 `then(|| …)`】`bool::then` 在条件为假时**不调用**闭包，
        // 于是首块那一轮 `primed_kind` 永远不会被写入 —— 下一块就会拿"0"当上一块的
        // 种类（而 0 恰好是 `Untouched` 的编码），把一次真实的源切换判成"同族"。
        let was_primed = state.primed.swap(true, Ordering::AcqRel);
        let previous_kind = state.primed_kind.swap(kind, Ordering::AcqRel);
        let first_kind = was_primed.then_some(previous_kind);
        // 【B0】本块装填的是哪一种接缝（`None` = 没有源切换）。账本要它来区分
        // "这一下咔来自源切换"与"这一下咔来自覆盖边界"。
        let mut armed = crate::render::click::SeamCode::None;
        if first_kind.is_none() {
            // 真首块：没有"上一块输出的尾巴"。
            //
            // 【为什么不是无条件原样输出】旧契约是"从零淡入会改变宿主直通的样本"，
            // 而那**只对透传成立**：透传的样本就是宿主正在播的东西，它自己已经连续。
            // 当第一块是**我们算出来的内容**（Raw / Processed）时，它从哪一相位开始
            // 是任意的 —— 从零直接跳到块内波形就是一次标准的起播咔。
            // 【作用域：只给 Capture 的播放通路】首块淡入要解决的是"我们自己算出来的
            // 内容从任意相位开始"—— 那是 Capture 通路（插件直接对着一条轨出声）的形状。
            // ARA 通路由宿主按 region 调度，区块起点与 item 淡化都是宿主的事；在那里
            // 再叠一层淡入会改动宿主自己的包络（且没有任何报障指向它）。
            let capture_content = !self.is_bound() && self.capture.has_material();
            if capture_content && Self::family(kind as u8) != ContentFamily::Passthrough {
                self.arm_seam(state, Seam::FirstBlock);
                armed = crate::render::click::SeamCode::FadeIn;
            }
            state.last_token.store(token, Ordering::Release);
        } else if first_kind != Some(kind) {
            // 本轮从未去过咔、但上一轮停在另一种源上：仍然要去咔。至于是交叉淡化
            // 还是淡入，取决于上一个源是不是透传（见 `seam`）。
            let previous_kind = first_kind.unwrap_or(kind) as u8;
            let seam = Self::seam(previous_kind, kind as u8);
            self.arm_seam(state, seam);
            armed = seam.code();
            state.last_token.store(token, Ordering::Release);
        } else if self.force_declick[stream.index()].swap(false, Ordering::AcqRel) {
            // 【T-A1-1：同一份源内部的读位置跳变】token 没变，但音频在块边界上确实
            // 错开了一小段（起播窗口收口、循环折返、流交接）。尾巴里存的是上一块真正
            // 输出的样本，与旧位置连续 —— 用它做一次等功率交叉，比"整块换源"轻得多。
            let previous_token = state.last_token.load(Ordering::Acquire);
            let previous_kind = (previous_token & 0xFF) as u8;
            let seam = Self::seam(previous_kind, kind as u8);
            self.arm_seam(state, seam);
            armed = seam.code();
            state.last_token.store(token, Ordering::Release);
        } else {
            // 【为什么先取旧的再交换】"要不要交叉淡化"取决于**上一个**源是什么；
            // `swap` 一执行旧值就没了，所以必须先读。
            let previous_token = state.last_token.load(Ordering::Acquire);
            if state.last_token.swap(token, Ordering::AcqRel) == token {
                // 同一份快照、同一个版本：没有切换，不做任何淡化。
            } else {
                // 【为什么"透传 → 处理结果"不交叉淡化】`PendingSynthesis` / `PassThrough`
                // 输出的是**宿主原声**，`Processed` 输出的是神经声码器结果。两者内容根本
                // 不同源，交叉淡化会把两段不同内容叠在一起（听声是"两声并发"）。
                // 那种场合改为对结果做一次淡入。
                let previous_kind = (previous_token & 0xFF) as u8;
                let current_kind = (token & 0xFF) as u8;
                let seam = Self::seam(previous_kind, current_kind);
                self.arm_seam(state, seam);
                armed = seam.code();
            }
        }
        let channels = (output.num_channels as usize).min(2);
        if state.fade_active.load(Ordering::Acquire) {
            // 宽度由 `arm_seam` 定：透传接缝与同族切换 64 帧，Raw ↔ Processed 960 帧。
            let len = match state.fade_len.load(Ordering::Acquire) as usize {
                0 => OUTPUT_DECLICK_FRAMES,
                len => len.min(DECLICK_HISTORY_FRAMES),
            };
            let position = state.fade_pos.load(Ordering::Acquire) as usize;
            let remaining = len.saturating_sub(position);
            let count = remaining.min(frames);
            for channel in 0..channels {
                // SAFETY: 调用方保证输出总线有 `channels` 个可写平面。
                let plane = unsafe { *output.channel_buffers.add(channel) };
                if plane.is_null() {
                    continue;
                }
                let equal_power = state.fade_curve.load(Ordering::Acquire) != 0;
                for index in 0..count {
                    let t = (position + index + 1) as f32 / len as f32;
                    let old = f32::from_bits(
                        state.fade_from[channel][position + index].load(Ordering::Relaxed),
                    );
                    // SAFETY: `index < frames`，调用方保证平面至少这么长。
                    let current = unsafe { *plane.add(index) };
                    // 【为什么权重不能只有一套】见 `Seam::uses_equal_power`：
                    // 两条不相关信号的线性混合在中点掉 6 dB（听感是一次凹陷/咔），
                    // 而同一份内容的两次渲染用等功率会在中点叠出 +3 dB 的鼓包。
                    let mixed = if equal_power {
                        let (out_gain, in_gain) = crate::render::crossfade::equal_power(t);
                        old * out_gain + current * in_gain
                    } else {
                        old * (1.0 - t) + current * t
                    };
                    unsafe {
                        *plane.add(index) = mixed;
                    }
                }
            }
            let next = position + count;
            state.fade_pos.store(next as u32, Ordering::Release);
            if next >= len {
                state.fade_active.store(false, Ordering::Release);
            }
        }
        // 保存本块输出的尾巴，供下一次源切换使用。
        //
        // 【为什么短块要左移历史、把新样本追加在末尾】`tail` 的索引语义是
        // "0 = 最近 64 帧里最旧的一帧、63 = 最新一帧"。块长不足 64 时，若像旧实现
        // 那样把样本放在 0 起、右边补零，淡出会从**本块第一帧**开始、然后读到零，
        // 与"边界处的最新样本"差出整整一个块长 —— 短块宿主上每次接缝都会多一次
        // 阶跃。左移历史再追加，才能保持"最后 64 帧"这一不变量。
        if frames >= DECLICK_HISTORY_FRAMES {
            let start = frames - DECLICK_HISTORY_FRAMES;
            for channel in 0..channels {
                // SAFETY: 同上。
                let plane = unsafe { *output.channel_buffers.add(channel) };
                if plane.is_null() {
                    continue;
                }
                for index in 0..DECLICK_HISTORY_FRAMES {
                    unsafe {
                        state.tail[channel][index]
                            .store((*plane.add(start + index)).to_bits(), Ordering::Relaxed);
                    }
                }
            }
        } else {
            let written = frames;
            for channel in 0..channels {
                // SAFETY: 同上。
                let plane = unsafe { *output.channel_buffers.add(channel) };
                if plane.is_null() {
                    continue;
                }
                for index in 0..(DECLICK_HISTORY_FRAMES - written) {
                    let value = state.tail[channel][index + written].load(Ordering::Relaxed);
                    state.tail[channel][index].store(value, Ordering::Relaxed);
                }
                for index in 0..written {
                    unsafe {
                        state.tail[channel][DECLICK_HISTORY_FRAMES - written + index]
                            .store((*plane.add(index)).to_bits(), Ordering::Relaxed);
                    }
                }
            }
        }

        // 【B0】把这一块的形状记进账本。音频线程：只写原子量。
        //
        // 【为什么放在最后】这里读到的就是**已经输出**的样本（去咔之后），于是
        // `max_step` 描述的正是用户听到的那一段。放在去咔之前会量到被淡化前的阶跃，
        // 那对"到底有没有响一下"没有意义。
        self.record_click_sample(
            stream,
            kind as u8,
            armed,
            first_kind.is_none(),
            output,
            frames,
        );
    }

    /// 门控清零时的去咔：把上一块的尾巴在首段内线性淡到零，其余帧写零。
    ///
    /// 【为什么必须独立于 `apply_output_declick`】后者就地做
    /// `old*(1-t) + current*t`，依赖"当前缓冲里已经写着本块的新内容"。门控分支里
    /// 缓冲里是**上一块的残留**（in-place 宿主下更是等于输入），而且紧跟着的
    /// `clear_outputs` 会把整块（含刚淡好的 64 帧）一起写零 —— 先淡后清等于白淡，
    /// 先清后淡则 `current` 恒为 0。两条都得不到"从上一块尾巴淡到零"。
    ///
    /// 本函数自己写满整块：前 `min(frames, OUTPUT_DECLICK_FRAMES)` 帧按尾巴淡出，
    /// 其余为零，并把尾巴清零。调用方因此**不要**再调 `clear_outputs`。
    ///
    /// # Safety
    /// `output` 必须是本回调内有效的 VST3 输出总线；`frames` 不超过其平面长度。
    pub(crate) unsafe fn apply_gate_fade_out(
        &self,
        stream: ProcessStream,
        output: &mut crate::audio_abi::AudioBusBuffers,
        frames: usize,
    ) {
        if frames == 0 || output.num_channels <= 0 || output.channel_buffers.is_null() {
            return;
        }
        let Some(state) = self.declick.get(stream.index()) else {
            return;
        };
        let declared = output.num_channels as usize;
        let fade = frames.min(OUTPUT_DECLICK_FRAMES);
        // `tail` 里 0 是最旧的、63 是最新的；短块淡出必须从**最新的 fade 帧**开始，
        // 否则边界处仍会从一段旧样本跳到零。
        let base = DECLICK_HISTORY_FRAMES - fade;
        let mut any_signal = false;
        for channel in 0..declared {
            // SAFETY: 宿主声明 `num_channels` 个平面指针，非空平面至少含 `frames` 帧。
            let plane = unsafe { *output.channel_buffers.add(channel) };
            if plane.is_null() {
                continue;
            }
            if channel < 2 {
                for index in 0..fade {
                    let old =
                        f32::from_bits(state.tail[channel][base + index].load(Ordering::Relaxed));
                    if old != 0.0 {
                        any_signal = true;
                    }
                    // 分母用 `fade` 而不是常量：块短于 64 帧时也必须在本块内淡到零，
                    // 否则块尾留下非零值，下一块从零起就是一次新的阶跃。
                    let gain = 1.0 - (index + 1) as f32 / fade as f32;
                    // SAFETY: index < fade ≤ frames。
                    unsafe { *plane.add(index) = old * gain };
                }
                for index in fade..frames {
                    // SAFETY: index < frames。
                    unsafe { *plane.add(index) = 0.0 };
                }
            } else {
                // SAFETY: 同上；多声道总线的后几个平面不在尾巴里，直接写零。
                unsafe { std::ptr::write_bytes(plane, 0, frames) };
            }
        }
        // 【为什么按实际输出报静音位】本块通常含一段淡出（非静音）；但尾巴本来就
        // 全零时（首次静音、或连续静音的第二块），整块确实全是零 —— 这时如实报
        // 静音，宿主才能跳过下游处理。恒报"非静音"会在每次静音开始时白跑一整条链路。
        output.silence_flags = if any_signal {
            0
        } else {
            (1_u64 << declared.min(64)) - 1
        };
        // 尾巴与输出源一起归零：下一块从静音起，不需要（也不该）再交叉淡化。
        for channel in 0..declared.min(2) {
            for index in 0..OUTPUT_DECLICK_FRAMES {
                state.tail[channel][index].store(0, Ordering::Relaxed);
            }
        }
        state.fade_active.store(false, Ordering::Release);
        // 【为什么不是 0】`primed_kind == 0` 的语义是"这个流从未输出过任何东西"，
        // 下一块会被当成真首块**原样输出** —— 而本块刚刚输出了一整段淡出到零的
        // 静音。从零直接跳到完整内容就是用户听到的"咔"。借用 PassThrough 的语义：
        // 它让 `apply_output_declick` 把 `fade_from` 视为全零，于是下一块做一次
        // 线性淡入。
        state.primed_kind.store(
            u32::from(CaptureOutput::PassThrough.code()),
            Ordering::Release,
        );
        state.last_token.store(0, Ordering::Release);
    }
}
