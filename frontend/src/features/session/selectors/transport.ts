/**
 * 「宿主时间此刻是否在走」的**唯一**判据。
 *
 * 【为什么必须收敛成一个函数】此前全仓有**三处三个公式**（`TimelineTransportBridge`、
 * `usePianoRollPanelModel`、`TrackList`），各写了一遍 `isPlaying || positionAdvancing`。
 * 它们恰好在"位置冻结"这个场景互相分歧 —— 而那正是用户报的"停止后光标还要往前多走
 * 一小段"。
 *
 * 【为什么不能只看 isPlaying】后端的 `isPlaying` 曾经是一个**永不过期的闩锁**：VEGAS
 * 停播时直接停止调用 `process()`、不发 `kPlaying=0` 的块，于是它永远为真，视觉插值
 * 按 1x 无界外推。后端现在给了播放位寿命（`block_playing_ttl_ms`），前端仍要用读数
 * 年龄兜一层 —— **两个独立的地方都错**才会出错。
 *
 * 【为什么 age 缺失时按"新鲜"处理】没有年龄字段说明是旧后端或测试夹具；那时退回
 * 原来的判据（只看 isPlaying），行为与改动前一致，不会把正常播放误判成停止。
 */

/**
 * 播放位允许的最大读数年龄（毫秒）。
 *
 * 【为什么是 250】宿主音频块在 44.1/48 kHz 下通常是 21~46 ms 一块；250 ms 是 5~10 块，
 * 足以吸收调度抖动与编辑器线程被抢，又短到用户察觉不到"停止后还在走"。它与后端的
 * `block_playing_ttl_ms` 下限（200 ms）同量级，两层判据因此不会互相打架。
 */
export const TRANSPORT_ADVANCE_MAX_AGE_MS = 250;

/** 判据的输入（`state.session.runtime` 的一个子集，便于单测）。 */
export interface TransportAdvanceInput {
    isPlaying: boolean;
    positionAdvancing: boolean;
    positionAgeMs?: number | null;
    /** 最近一条采样的来源（capture / offline 时由捕获本地时钟驱动）。 */
    transportSource?: string;
    /**
     * 引擎正在**原地等待渲染**（播放中位置被有意冻结）。
     *
     * 【为什么这条否决留在共享函数里】它是独立 App 唯一合法的"在播但位置不动"，
     * 而三处调用点此前各自抄了一遍。留在函数里，三处就不会再分歧；插件形态永远不会
     * 置位它（渲染不改变宿主位置）。
     */
    playbackWaitingForRender?: boolean;
}

export const isTransportAdvancing = (runtime: TransportAdvanceInput): boolean => {
    if (runtime.playbackWaitingForRender) {
        return false;
    }
    const age = runtime.positionAgeMs;
    const fresh =
        age === undefined || age === null || !Number.isFinite(age)
            ? true
            : age <= TRANSPORT_ADVANCE_MAX_AGE_MS;
    // 捕获 / 离线 pass 的时钟由后端 CaptureBlock 发布，与宿主 kPlaying 无关。
    // 它们仍在“推进”时可以让视觉播放头以 60 Hz 前进；位置冻结时 positionAdvancing
    // 为 false，光标仍会停住（停止后不漂移的同一道闸门）。
    if (runtime.transportSource === "capture" || runtime.transportSource === "offline") {
        return runtime.isPlaying && fresh;
    }
    // 【kPlaying 不是前进判据】FL Studio 停播时仍置 kPlaying=1，位置钉住；如果
    // 这里只认 isPlaying，视觉层会按 1x 无限前漂。前进必须由两次采样证明。
    return runtime.positionAdvancing && fresh;
};
