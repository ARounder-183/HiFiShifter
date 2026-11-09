//! 组件生命周期：VST3 与 CLAP **共用的唯一入口序列**。
//!
//! 【为什么顺序必须共享】两个格式的回调名字不同（setActive / activate、
//! setProcessing / start_processing、terminate / destroy），但每个回调对插件状态机的
//! 含义完全相同：激活时恢复挂起的预览、清掉上一遍的流几何、主动重同步；停止处理时
//! 收尾处理 pass、冻结宿主时钟；终止时停掉实例 worker。
//!
//! 这些顺序此前只写在 VST3 入口里，CLAP 外壳因此少了一整组状态迁移：同一个宿主动作
//! （失活再激活、停止处理、销毁实例）在两个格式里得到不同的实例状态，而症状只会在
//! 其中一个格式上出现 —— 第二遍播放不出声、捕获接不上、实例销毁后 actor 线程不退出。
//!
//! 【纪律】外壳只回答"哪个回调对应哪个语义"；顺序本身写在这里。新增任何一条状态迁移
//! 都必须落进这个文件，否则下一次格式扩展会再漏一遍。

use super::*;

impl ExtensionOwner {
    /// 宿主声明工程采样率（并可选声明离线意图）：两种格式唯一的 setup 序列。
    ///
    /// 【为什么采样率与离线准备同处一个入口】它们同属"宿主在 process 之前声明的处理
    /// 条件"：VST3 在 `setupProcessing` 给，CLAP 在 `activate` 与 `clap.render.set` 给。
    /// 分成两处记录必然漂移 —— 采样率决定快照选择，离线意图决定要不要在出第一块之前
    /// 把渲染准备好，两者必须由同一个调用点落定。
    ///
    /// 【为什么失败返回 Err 而不是让调用方忽略】离线准备超时只该让本次产物降级成静音，
    /// 不该让宿主把插件踢出链路；错误交给外壳记日志 —— "失败之后怎么办"由各自 ABI
    /// 决定（VST3 仍返回 kResultOk，CLAP 仍返回 true）。
    pub fn setup_processing(
        self: &Arc<Self>,
        sample_rate: f64,
        offline_declared: bool,
    ) -> Result<(), String> {
        self.set_setup_sample_rate(sample_rate);
        // 【为什么进程模式也在这里落定】setup_process_mode 回答的是"宿主声明的是
        // 实时还是离线"，而它有三个读者：离线准备要不要**要求**用户曲线
        // （capture_prepare_job）、冻结预览的资格判定（note_preview_block）、以及
        // processing_started 的离线意图。此前只有 VST3 在 setupProcessing 里写它，
        // CLAP 从不写 ⇒ 恒为 0 ⇒ CLAP 的离线导出会发布一份**没有用户曲线**的产物
        // （正是 capture_render_input 文档里那条"烘焙进产物"的失败）。
        // CLAP 没有 kPrefetch 这一档，实时一律记 0、离线记 2，与 VST3 的枚举同值。
        self.set_setup_process_mode(if offline_declared { 2 } else { 0 });
        self.arm_capture_rendering();
        if !offline_declared {
            return Ok(());
        }
        let started = std::time::Instant::now();
        self.prepare_offline_until(started + OFFLINE_PREPARE_TIMEOUT)
            .map_err(|error| {
                format!(
                    "offline preparation failed after {:?}: {error}",
                    started.elapsed()
                )
            })
    }

    /// 组件进入激活状态（VST3 `setActive(true)` / CLAP `activate`）。
    pub fn component_activated(&self) {
        self.note_component_activated();
        // 【预览跨失活保留 cursor】重新激活时若有一个挂起的预览，恢复它；否则清掉
        // 上一轮的几何（同一 raw 的第一块会被判成 Resting，输出继续读上一张 take）。
        if !self.resume_processing_preview() {
            self.reset_transport_session("component_activated");
        }
        // 后台/前台切换后宿主可能已重建几何或恢复源访问；主动枚举一次，
        // 不等下一次 UI tick 的节流窗口。
        self.resume_resync();
    }

    /// 组件离开激活状态（VST3 `setActive(false)` / CLAP `deactivate`）。
    pub fn component_deactivated(&self) {
        self.note_component_deactivated();
    }

    /// 开始处理（VST3 `setProcessing(true)` / CLAP `start_processing`）。
    ///
    /// `offline_declared` = 宿主在这一阶段声明过离线模式。VST3 的声明点是
    /// `setupProcessing` 的 process_mode，CLAP 是 `clap.render.set`。
    pub fn processing_started(self: &Arc<Self>, offline_declared: bool) {
        // 恢复挂起的预览：cursor 与 pass 原样接上。
        if self.resume_processing_preview() {
            return;
        }
        if !offline_declared {
            return;
        }
        // 【离线预览的第二个入口】有些宿主声明离线模式之后直到开始处理才真正开始预览。
        // 这里同步准备一次；失败只记日志，让宿主保留插件（产物静音但可诊断）。
        let started = std::time::Instant::now();
        if let Err(error) = self.prepare_offline_until(started + OFFLINE_PREPARE_TIMEOUT) {
            crate::log_line(&format!(
                "Offline preparation at processing start failed after {:?}: {error}",
                started.elapsed()
            ));
        }
    }

    /// 停止处理（VST3 `setProcessing(false)` / CLAP `stop_processing`）。
    pub fn processing_stopped(&self) {
        self.record_processing_stop();
        if self.processing_pass_is_preview() {
            // 【预览只挂起】非实时预览完全可能在每个 chunk 之间调用停止处理。结束 pass
            // 会把 cursor/几何清空，下一块只能重新从候选开始 —— 看起来就是"只录了 1 秒"。
            self.suspend_processing_preview("processing_stopped");
        } else {
            self.note_processing_stopped();
        }
    }

    /// 宿主要求丢弃跨块状态（CLAP `plugin.reset`）。
    ///
    /// 【为什么不能是空实现】`clap/plugin.h` 的 reset 语义是"清掉会跨 process 调用残留
    /// 的状态"：宿主在定位、循环回卷、离线分段之间调它。本插件唯一跨块残留的就是
    /// **会话边界**（流几何、每流播放位、rung latch、捕获时钟）—— 不在这里清，下一块
    /// 会被判成"上一遍的延续"，表现是定位后第一块读旧快照、或捕获接不上。
    /// 它与 `processing_stopped` 走同一个原子量序列，因此可在音频线程调用。
    pub fn processing_reset(&self) {
        self.reset_transport_session("host reset");
    }

    /// 组件终止（VST3 `terminate` / CLAP `destroy`）：停 worker、释放宿主适配器。
    ///
    /// 【为什么必须在销毁实例之前调】会话 actor 与准备 worker 都持有指向本 owner 的
    /// 强引用，不先关闭邮箱线程就不会退出 —— 每销毁一个实例就漏一条线程与它持有的全部
    /// 内存。VST3 在 `terminate` 里调的是同一个方法。
    pub fn component_terminated(&self) {
        self.stop_editor();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 组件终止必须让 owner 进入 closed。
    ///
    /// 【为什么钉这一条】CLAP 的 `destroy` 只调 `component_terminated()` 这一个方法，
    /// 而它是"实例销毁后会话 actor 与准备 worker 会退出"的唯一保证 —— 漏掉它每销毁
    /// 一个实例就漏一条线程与它持有的全部内存，且没有任何报错。
    #[test]
    fn component_termination_closes_the_owner() {
        let owner = Arc::new(ExtensionOwner::default());
        assert!(!owner.is_closed());
        owner.component_terminated();
        assert!(owner.is_closed());
    }

    /// 激活/失活与处理开始/停止在"没有文档、没有宿主"的实例上必须全部可用。
    ///
    /// 【为什么钉这一条】这四个入口是宿主**无条件**会调的：一个从未绑定 ARA 的
    /// 普通插入也要走完它们。任何一条依赖文档的实现都会让非 ARA 宿主直接崩在
    /// 生命周期回调里 —— 而那是宿主进程，不是我们的。
    #[test]
    fn the_lifecycle_entries_survive_an_instance_without_a_document() {
        let owner = Arc::new(ExtensionOwner::default());
        owner.setup_processing(48_000.0, false).unwrap();
        owner.component_activated();
        owner.processing_started(false);
        owner.processing_stopped();
        owner.component_deactivated();
        assert!(!owner.is_ara_bound());
        assert!(!owner.processing_pass_active());
        // 工程采样率经同一条入口落到捕获存储：`canCapture` 只认这一个来源。
        assert_eq!(owner.setup_sample_rate(), 48_000.0);
        assert!(owner.capture().status().host_ready);
    }
}
