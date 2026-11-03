//! 离线 pass 的入口：cursor、准备与冻结。
use super::*;

impl ExtensionOwner {
    /// 记录 setupProcessing 的 process mode。
    pub(crate) fn set_setup_process_mode(&self, mode: i32) {
        self.setup_process_mode.store(mode, Ordering::Release);
    }

    pub(crate) fn setup_process_mode(&self) -> i32 {
        self.setup_process_mode.load(Ordering::Acquire)
    }

    /// 每个 process 块调用一次：离线时返回本块本地 cursor。
    ///
    /// 【为什么只在未绑定实例上生效】ARA 的离线渲染由文档/区域驱动，仍应使用宿主
    /// 给的 projectTimeSamples；Capture 才需要自己的 0 起点。
    pub(crate) fn note_offline_block(
        &self,
        context: &crate::audio_abi::ProcessContext,
        mode: i32,
        frames: usize,
    ) -> Option<i64> {
        if mode != 2 || self.is_bound() || self.bound_once.load(Ordering::Acquire) {
            // mode 离开 2 时也要让 OfflinePass 收尾。
            if mode != 2 {
                let was_active = self.offline_pass.is_active();
                self.offline_pass.note_block(mode, 0, 0, 0.0);
                // 离开 mode=2 后，旧的 offline cursor 不能再驱动 UI。
                if was_active {
                    if let Some(clock) = self.clock.get() {
                        clock.clear_capture();
                    }
                }
            }
            return None;
        }
        self.offline_pass.note_block(
            mode,
            context.project_time_samples,
            frames,
            context.sample_rate,
        )
    }

    /// 结束离线 pass（setProcessing(false) / setActive(false) / mode 离开 2）。
    #[allow(dead_code)]
    pub(crate) fn end_offline_pass(&self) {
        self.offline_pass.end();
    }

    /// 宿主停止处理：捕获暂停、离线 pass 结束、传输时钟冻结。
    pub(crate) fn note_processing_stopped(&self) {
        self.offline_pass.end();
        self.capture().pause();
        if let Some(clock) = self.clock.get() {
            clock.freeze_transport();
            clock.clear_capture();
        }
    }

    /// 组件失活：与停止处理同一处置，但不丢素材。
    pub(crate) fn note_component_deactivated(&self) {
        self.note_processing_stopped();
    }

    /// 离线 pass 是否活跃。
    pub(crate) fn offline_pass_active(&self) -> bool {
        self.offline_pass.is_active()
    }

    pub(crate) fn offline_pass_cursor_frames(&self) -> i64 {
        self.offline_pass.cursor_frames()
    }

    #[allow(dead_code)]
    pub(crate) fn offline_pass_id(&self) -> u64 {
        self.offline_pass.pass_id()
    }

    #[allow(dead_code)]
    pub(crate) fn offline_pass_start_raw(&self) -> i64 {
        self.offline_pass.start_raw()
    }

    #[allow(dead_code)]
    pub(crate) fn offline_pass_sample_rate(&self) -> u32 {
        self.offline_pass.sample_rate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offline_cursor_starts_at_zero_and_advances_by_block_frames() {
        let owner = ExtensionOwner::default();
        let context = crate::audio_abi::ProcessContext {
            sample_rate: 48_000.0,
            project_time_samples: 1_500_000,
            ..Default::default()
        };
        assert_eq!(owner.note_offline_block(&context, 2, 512), Some(0));
        assert_eq!(owner.note_offline_block(&context, 2, 512), Some(512));
        assert_eq!(owner.offline_pass_cursor_frames(), 1_024);
        assert_eq!(owner.note_offline_block(&context, 0, 512), None);
        assert!(!owner.offline_pass_active());
    }
}
