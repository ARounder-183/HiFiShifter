import { playheadIsHostOwned } from "../../../services/hostCapabilities";
import type { SessionState } from "./types";

/**
 * 用户意图：把播放光标放到 `sec`。返回是否真的写了。
 *
 * 【为什么需要它】非 ARA 宿主里 `set_transport` 必然失败，而前端每一次跳转都是
 * "乐观写 store + 打后端"两步。乐观写让光标先跳过去，随后 30 Hz 的播放轮询用宿主
 * 真实位置把它拽回来 —— 用户看到的是"先跳过去、再跳回来"。宿主拥有光标时，镜像
 * 就不该写源，所以**所有**意图型写入都从这里过闸。
 *
 * 【为什么返回值不是 void】调用点通常紧跟着 `pendingPlayheadRevealSec = playheadSec`
 * （把视图滚到新光标处）。闸住时 `playheadSec` 没变，就不该再把视图滚过去。
 *
 * 【哪些写入**不**走这里】宿主镜像：播放轮询（`syncPlaybackState`）、时间线快照
 * （`applyTimelineState`）、以及"保留原值"式的回写（tracks-only 刷新、take 投影）。
 * 它们写的是**事实**，不是意图 —— 闸住它们会让界面与宿主分叉。
 */
export function intentPlayheadSec(state: SessionState, sec: number): boolean {
    if (playheadIsHostOwned()) return false;
    state.playheadSec = Math.max(0, sec);
    return true;
}
