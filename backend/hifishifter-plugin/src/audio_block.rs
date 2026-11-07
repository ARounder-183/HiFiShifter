//! 与宿主 ABI 无关的“一个音频块的**样本面**”。
//!
//! 【为什么需要它】`crate::block_context::BlockContext` 描述的是这一块的**时间事实**，
//! 而输出通路要读写的还有**平面本身**：宿主声明了几个声道、每个声道的可写指针在哪、
//! 静音掩码写回哪里、输入与输出是不是同一份总线。这些此前以 VST3 的
//! `ProcessData` / `AudioBusBuffers` 形态直接长在渲染通路的签名里，于是任何别的
//! 格式想复用同一条渲染逻辑，都只能先伪造一个 VST3 结构 —— 那是把 ABI 当中间语言。
//!
//! 现在这一层只描述平面：VST3 侧在 `crate::audio_abi::block_from_vst3` 里读一次
//! 指针数组，CLAP 侧从 `clap_audio_buffer_t` 构造同一份结构，两边之后走的是同一段代码。
//!
//! 【为什么平面是裸指针而不是切片】宿主**可以合法地**把同一块内存同时当输入和输出
//! （in-place），此时“输入的 `&[f32]`”与“输出的 `&mut [f32]`”是重叠的借用 —— 那是
//! 未定义行为，而不是一个可以商量的边界。本文件因此只用裸指针搬运地址，读写在调用点
//! 按“本回调独占”的既有契约进行（与 ABI 层直通函数的既有写法一致）。

use crate::audio_abi::BufferError;

/// 一块音频的**输出平面**。
///
/// 【为什么静音掩码是 `&mut u64` 而不是字段】它必须写回**宿主的结构**（VST3 是
/// `AudioBusBuffers::silence_flags`，CLAP 是一个本地 `u64`），而两者的存放位置不同。
/// 把它做成一个写出口之后，写掩码这件事就只有一处实现。
pub struct OutputBus<'a> {
    /// 宿主声明的声道数。可能大于 2：多声道总线只处理前两个，其余保持静音。
    pub num_channels: usize,
    pointers: &'a mut [*mut f32],
    silence: &'a mut u64,
}

impl<'a> OutputBus<'a> {
    /// 从宿主给出的平面指针数组与静音掩码写出口构造。
    ///
    /// 【为什么构造时就把掩码置成“全部静音”】`silence_flags` 的语义是“这一块**已经
    /// 写出**的内容里哪些声道全是零”。还没写任何样本时，“全部静音”是唯一诚实的初值；
    /// 每个写入点按实际写下的非零样本逐位清除它。
    pub fn new(pointers: &'a mut [*mut f32], silence: &'a mut u64) -> Self {
        *silence = crate::audio_abi::silence_mask(pointers.len());
        Self {
            num_channels: pointers.len(),
            pointers,
            silence,
        }
    }

    /// 第 `channel` 个平面的写指针；越界或 inactive 声道返回空指针。
    pub fn pointer(&self, channel: usize) -> *mut f32 {
        self.pointers
            .get(channel)
            .copied()
            .unwrap_or(std::ptr::null_mut())
    }

    /// 本块要写的声道数：只处理前两个（内部快照恒为立体声）。
    pub fn channels(&self) -> usize {
        self.num_channels.min(2)
    }

    /// 覆盖静音掩码。
    pub fn set_silence_flags(&mut self, value: u64) {
        *self.silence = value;
    }

}

/// 一块音频的**输入平面**。
///
/// 【为什么按值持有指针数组地址而不是引用】见文件头：in-place 宿主下它与
/// [`OutputBus`] 指向同一份内存，制造一个共享引用就是制造别名。这里只存“数组在哪、
/// 有几个”，读的时候再取一次指针 —— 与直通函数复制总线头的
/// 历史做法同义。
#[derive(Clone, Copy)]
pub struct InputBus {
    pointers: *const *mut f32,
    /// 宿主声明的输入声道数。
    pub num_channels: usize,
}

impl InputBus {
    /// 从宿主的输入平面指针数组构造。
    ///
    /// # Safety
    /// `pointers` 要么为空，要么指向至少 `num_channels` 个有效指针，并在本回调期间
    /// 保持有效。
    pub unsafe fn new(pointers: *const *mut f32, num_channels: usize) -> Self {
        Self {
            pointers,
            num_channels,
        }
    }

    /// 没有输入的块（纯生成/静音场景）。
    pub fn empty() -> Self {
        Self {
            pointers: std::ptr::null(),
            num_channels: 0,
        }
    }

    /// 第 `channel` 个平面的读指针；越界或 inactive 声道返回空指针。
    ///
    /// 【为什么不看 `silence_flags`】该位只是“这些样本全是零”的可选提示，置位时
    /// 缓冲**仍必须可读**。把它当成“没有输入”会让一条本来有声的轨道被插件静音掉
    /// （见 `audio_abi::capture_view` 的同名说明）。
    pub fn pointer(&self, channel: usize) -> *mut f32 {
        if channel >= self.num_channels || self.pointers.is_null() {
            return std::ptr::null_mut();
        }
        // SAFETY: 构造契约保证数组至少有 `num_channels` 个有效指针。
        unsafe { *self.pointers.add(channel) }
    }

    /// 本块可读的声道数：只处理前两个。
    pub fn channels(&self) -> usize {
        self.num_channels.min(2)
    }

    /// 前两个声道的只读视图（分层拷贝要把“宿主原声”混进输出）。
    ///
    /// 【为什么把生命周期交给调用方】`self` 是一个按值持有的描述（它不借用宿主缓冲），
    /// 而切片必须活到“本次回调”那么久。由调用方在 unsafe 契约里承诺这一点，比在这里
    /// 假装借用一下更诚实 —— 也避免它把 `AudioBlock` 的其余字段一起借走。
    ///
    /// # Safety
    /// 非空平面至少有 `frames` 帧可读；返回的切片不得带出本次回调。
    pub unsafe fn planes<'a>(&self, frames: usize) -> [Option<&'a [f32]>; 2] {
        let mut planes: [Option<&'a [f32]>; 2] = [None, None];
        if frames == 0 {
            return planes;
        }
        for channel in 0..self.channels() {
            let plane = self.pointer(channel);
            if plane.is_null() {
                continue;
            }
            // SAFETY: 由调用方保证非空平面至少有 `frames` 帧可读。
            planes[channel] = Some(unsafe { std::slice::from_raw_parts(plane, frames) });
        }
        planes
    }
}

/// 一个音频块：时间事实之外的**样本面**。
///
/// 【为什么把 frames / mode 也放进来】“这一块多长、属于哪种进程流”是渲染通路每个分支
/// 都要读的同一个事实，而它此前分别从 `ProcessData::num_samples` 与
/// `ProcessData::process_mode` 现场读 —— 两个格式各自读一遍，就多了一处可以漂移的地方。
pub struct AudioBlock<'a> {
    /// 本块的帧数。
    pub frames: usize,
    /// 进程形态（VST3/CLAP 同值：0 = 实时，1 = 预取，2 = 离线）。
    pub mode: i32,
    /// 输入平面。
    pub input: InputBus,
    /// 输出平面。
    pub output: OutputBus<'a>,
    /// 输入与输出是否**共用同一份总线结构**（in-place 宿主）。
    ///
    /// 【为什么必须知道这件事】`silence_flags` 写在总线结构里：in-place 时给我们
    /// 自己输出写的“全静音”会在**下一块**被我们自己的捕获读成“宿主没给输入”。
    /// 掩码只是给宿主省算力的可选提示，in-place 时省掉它没有任何代价。
    pub in_place: bool,
}

impl<'a> AudioBlock<'a> {
    /// 本块是否属于离线（冻结 / 胶合 / 导出）进程流。
    pub fn is_offline(&self) -> bool {
        self.mode == 2
    }

    /// 把输出平面清零，并按声道数写“全部静音”掩码（in-place 时不写掩码）。
    ///
    /// 【为什么按 `num_channels` 而不是写死 2】总线接受任意声道数（见
    /// `audio_abi::validate_bus`）：钳到 2 会让第 3..N 个声道**留着上一次回调的残留**，
    /// 而渲染只写前两个 —— 环绕声道于是输出垃圾。
    ///
    /// 【为什么 in-place 时不写掩码】见 [`AudioBlock::in_place`]：写进去的“全静音”
    /// 会在下一块被我们自己的捕获读成“宿主没给输入”。
    pub fn clear_output(&mut self) {
        let frames = self.frames;
        if self.output.num_channels == 0 {
            self.output.set_silence_flags(0);
            return;
        }
        for channel in 0..self.output.num_channels {
            let plane = self.output.pointer(channel);
            if plane.is_null() {
                continue;
            }
            // SAFETY: 调用契约保证非空平面至少有 `frames` 个可写样本。
            unsafe { std::ptr::write_bytes(plane, 0, frames) };
        }
        if !self.in_place {
            self.output
                .set_silence_flags(crate::audio_abi::silence_mask(self.output.num_channels));
        }
    }

    /// 把输入原样搬到输出（纯 editor 角色、未知进程形态的降级）。
    ///
    /// 【为什么每帧先读两个源再写输出】宿主可以合法地把同一个缓冲同时当输入和输出，
    /// 甚至把两个完整平面互换。先读后写让这两种别名都正确。
    ///
    /// 【为什么不看 `silence_flags`】同 [`InputBus::pointer`]。
    ///
    /// # Safety
    /// 调用契约保证输入非空平面可读、输出非空平面可写 `frames` 帧，且部分重叠
    /// （既非同一地址也非完整 plane 互换）的情形会被下面的检查拒绝。
    pub unsafe fn pass_through(&mut self) -> Result<(), BufferError> {
        let frames = self.frames;
        if self.output.num_channels == 0 {
            self.output.set_silence_flags(0);
            return Ok(());
        }
        let channels = self.output.channels();
        let input_channels = self.input.channels();
        let sources: [*mut f32; 2] = std::array::from_fn(|channel| {
            if input_channels == 0 {
                std::ptr::null_mut()
            } else {
                self.input
                    .pointer(if input_channels == 1 { 0 } else { channel })
            }
        });
        let targets: [*mut f32; 2] = std::array::from_fn(|channel| {
            if channel < channels {
                self.output.pointer(channel)
            } else {
                std::ptr::null_mut()
            }
        });
        let bytes = frames * std::mem::size_of::<f32>();
        for source in sources {
            for target in targets {
                if source.is_null() || target.is_null() || source == target {
                    continue;
                }
                let a = source as usize;
                let b = target as usize;
                if a < b.saturating_add(bytes) && b < a.saturating_add(bytes) {
                    return Err(BufferError::UnsupportedFormat);
                }
            }
        }
        for frame in 0..frames {
            let values = sources.map(|source| {
                if source.is_null() {
                    0.0
                } else {
                    // SAFETY: 非空输入平面至少有 `frames` 帧可读。
                    unsafe { *source.add(frame) }
                }
            });
            if channels == 1 {
                let mono = if input_channels >= 2 {
                    (values[0] + values[1]) * 0.5
                } else {
                    values[0]
                };
                if !targets[0].is_null() {
                    // SAFETY: 非空输出平面至少有 `frames` 帧可写。
                    unsafe { targets[0].add(frame).write(mono) };
                }
            } else {
                for channel in 0..channels {
                    if !targets[channel].is_null() {
                        // SAFETY: 同上。
                        unsafe {
                            targets[channel].add(frame).write(values[channel]);
                        }
                    }
                }
            }
        }
        let silence = (0..channels).fold(0, |flags, channel| {
            if sources[channel].is_null() || targets[channel].is_null() {
                flags | (1 << channel)
            } else {
                flags
            }
        });
        self.output.set_silence_flags(silence);
        Ok(())
    }
}
