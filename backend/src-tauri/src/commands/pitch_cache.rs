//! 音高分析缓存的诊断与清理。
//!
//! 【历史】这里原本操作 `AppState::clip_pitch_cache`（容量 100 的 LRU），但那条路径
//! 只服务于 `pitch_analysis::analysis` 里早已不再被调用的旧流水线 —— 真正的分析结果
//! 存在 `pitch_clip` 的进程级全局缓存里。结果是"清空音高缓存"按钮对真实驻留内存
//! 零效果，而统计面板永远显示 0 条，排查内存问题时严重误导。
//!
//! 现在只保留工程切换路径真正调用的 `clear_pitch_analysis_caches`：原先暴露给
//! 前端的 `clear_pitch_cache` / `get_pitch_cache_stats` 两个 IPC 命令没有任何调用方
//! （前端已无对应按钮），已随本次清理删除。

/// 清理所有由音高分析派生的进程级缓存。
///
/// 供 `new_project` / `open_project` 调用（工程切换必须清理，否则旧工程的曲线常驻）。
///
/// 各模块只负责清自己的状态，这里做编排 —— 新增缓存时**必须**在此登记，
/// 否则又会变成"切换工程不释放"的一处遗漏。
pub(super) fn clear_pitch_analysis_caches() {
    let before = crate::pitch_clip::pitch_cache_memory_stats();

    let generation = crate::pitch_clip::clear_pitch_analysis_state();
    // 渲染阶段的 chunk 推理缓存：key 只含 (clip_id, mel_start)，无界，随音高/合成
    // 参数变化失效。工程切换后旧 clip_id 不会再来失效请求，必须整体清空。
    crate::renderer::hifigan::clear_chunk_cache();
    // 渲染状态表按 clip_id 索引，同样是"只增不减"。
    crate::clip_rendering_state::clear_all_clip_rendering_state();
    // 共振峰重建代次表按 clip_id 索引，只增不减。
    crate::formant_cache::clear_formant_rebuild_generations();

    let after = crate::pitch_clip::pitch_cache_memory_stats();
    // 只报真实存在过的驻留：切换工程时若缓存本就为空，不该刷无意义的日志。
    if before.entries > 0 || before.inflight > 0 {
        log::warn!(
            "[pitch] cleared analysis state (generation {}): {} entries / {} bytes (largest {}), inflight {} -> {} entries / {} bytes",
            generation,
            before.entries,
            before.total_bytes,
            before.largest_entry_bytes,
            before.inflight,
            after.entries,
            after.total_bytes,
        );
    }
}
