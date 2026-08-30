import { useCallback, useRef, useState } from "react";
import { registerDragAbort } from "../gestureFocusGuard";
import type { AppDispatch } from "../../../../app/store";
import type { SessionState } from "../../../../features/session/sessionSlice";
import { bumpParamsEpoch } from "../../../../features/session/sessionSlice";
import { resolveRootTrackId } from "../../../../features/session/trackUtils";
import { webApi } from "../../../../services/webviewApi";
import type { ClipInfo } from "../../../../features/session/sessionTypes";
import type { Keybinding } from "../../../../features/keybindings/types";
import { isModifierActive } from "../../../../features/keybindings/keybindingsSlice";
import { advanceFineAxisDrag, type FineAxisDragState } from "../fineAxisDrag";
import { computePitchDragCents, shiftPitchFrames } from "../clipPitchDrag";

/** 后端预览下发节流间隔（与淡化拖拽的远程同步节奏一致）。 */
const PITCH_DRAG_SEND_INTERVAL_MS = 120;

/** 帧窗口探测用的默认帧周期（backend 会在响应中回传实际值）。 */
const PITCH_DRAG_PROBE_FP_MS = 5;

export interface PitchDragTooltip {
    text: string;
    position: { x: number; y: number };
}

interface ClipPitchDragState {
    pointerId: number;
    clipId: string;
    rootTrackId: string;
    startFrame: number;
    frameCount: number;
    base: number[] | null;
    startClientY: number;
    fineState: FineAxisDragState | null;
    currentCents: number;
    sentCents: number;
    lastSentAt: number;
    sendTimer: number | null;
    sendChain: Promise<unknown>;
}

/**
 * useClipPitchDrag - 音频块"按住修饰键垂直拖拽波形 = 调整 Clip 范围内音高"。
 *
 * 流程：
 * 1. pointerdown（音高修饰键激活）→ 校验根轨道具备音高编辑能力
 *    （composeEnabled 且 pitchAnalysisAlgo != none，与钢琴卷帘 shiftParam
 *    同一守卫），按默认帧周期 5ms 探测取帧窗口，异步拉取基准 pitch 帧；
 * 2. pointermove → advanceFineAxisDrag 累计垂直位移（微调修饰键激活时
 *    自动进入精细刻度），换算音分；节流下发 setParamFrames(checkpoint=false)
 *    做实时预览；同时更新悬浮 ToolTips（跟随指针，展示音分变化量）；
 * 3. pointerup → 等待在途预览完成后，以 checkpoint=true 提交最终偏移
 *    （单次撤销步，与钢琴卷帘形变拖拽的提交约定一致），并 bumpParamsEpoch
 *    通知参数编辑器重新取数。
 *
 * ToolTips 文案由调用方注入（i18n 模板 + formatPitchDragCents）。
 */
export function useClipPitchDrag(deps: {
    sessionRef: React.RefObject<SessionState>;
    dispatch: AppDispatch;
    /** 微调修饰键绑定（modifier.paramFineAdjust） */
    fineAdjustKb: Keybinding;
    /** ToolTips 文案构造：(cents) → 本地化文本 */
    formatDragTooltip: (cents: number) => string;
}) {
    const { sessionRef, dispatch, fineAdjustKb, formatDragTooltip } = deps;
    const [pitchDragTooltip, setPitchDragTooltip] = useState<PitchDragTooltip | null>(null);
    const dragRef = useRef<ClipPitchDragState | null>(null);

    const startClipPitchDrag = useCallback(
        (e: React.PointerEvent, clipId: string) => {
            if (e.button !== 0 || dragRef.current) return;
            const session = sessionRef.current;
            const clip: ClipInfo | undefined = session.clips.find((c) => c.id === clipId);
            if (!clip) return;
            const rootTrackId = resolveRootTrackId(session.tracks, clip.trackId);
            const rootTrack = session.tracks.find((tr) => tr.id === rootTrackId);
            if (
                !rootTrackId ||
                !rootTrack?.composeEnabled ||
                rootTrack.pitchAnalysisAlgo === "none"
            ) {
                return;
            }

            const pointerId = e.pointerId;
            const startClientY = e.clientY;
            e.preventDefault();
            e.stopPropagation();
            (e.currentTarget as HTMLElement).setPointerCapture?.(pointerId);

            const startFrame = Math.max(
                0,
                Math.floor((clip.startSec * 1000) / PITCH_DRAG_PROBE_FP_MS),
            );
            const frameCount = Math.max(
                1,
                Math.ceil((clip.lengthSec * 1000) / PITCH_DRAG_PROBE_FP_MS),
            );

            const state: ClipPitchDragState = {
                pointerId,
                clipId,
                rootTrackId,
                startFrame,
                frameCount,
                base: null,
                startClientY,
                fineState: null,
                currentCents: 0,
                sentCents: 0,
                lastSentAt: 0,
                sendTimer: null,
                sendChain: Promise.resolve(),
            };
            dragRef.current = state;

            let finalized = false;

            // ── undo group（惰性开启）────────────────────────────────
            // 预览写入（checkpoint:false）会**真实改动后端参数帧**；若最终
            // 提交才 checkpoint:true，压入的快照是"已预览后"的状态 → 撤销
            // 变成无操作（B2）。因此首次预览写入前必须开 undo group，让组
            // 快照 = 拖拽前状态；最终写入留在组内（checkpoint:false），
            // 收尾时关闭 —— 整个拖拽 = 单个撤销步，且零位移点击不开组、
            // 不产生死撤销步。
            let undoGroupPromise: Promise<unknown> | null = null;
            const ensureUndoGroup = (): Promise<unknown> => {
                if (!undoGroupPromise) {
                    undoGroupPromise = webApi.beginUndoGroup();
                }
                return undoGroupPromise;
            };

            const scheduleSend = () => {
                if (!state.base || state.currentCents === state.sentCents) return;
                if (state.sendTimer != null) return;
                const now = Date.now();
                const wait = Math.max(0, PITCH_DRAG_SEND_INTERVAL_MS - (now - state.lastSentAt));
                state.sendTimer = window.setTimeout(() => {
                    state.sendTimer = null;
                    if (!state.base) return;
                    const cents = state.currentCents;
                    state.sentCents = cents;
                    state.lastSentAt = Date.now();
                    const values = shiftPitchFrames(state.base, cents / 100);
                    state.sendChain = state.sendChain
                        .then(() => ensureUndoGroup())
                        .then(() =>
                            webApi.setParamFrames(
                                state.rootTrackId,
                                "pitch",
                                state.startFrame,
                                values,
                                false,
                            ),
                        )
                        .catch(() => undefined);
                }, wait);
            };

            const teardown = () => {
                window.removeEventListener("pointermove", onMove, true);
                window.removeEventListener("pointerup", onUp, true);
                window.removeEventListener("pointercancel", onUp, true);
            };

            function onMove(ev: PointerEvent) {
                const st = dragRef.current;
                if (!st || ev.pointerId !== st.pointerId) return;
                ev.preventDefault();
                if (st.fineState == null) {
                    st.fineState = {
                        raw: st.startClientY,
                        adjusted: st.startClientY,
                        fineActive: false,
                    };
                }
                const adjustedY = advanceFineAxisDrag(
                    st.fineState,
                    ev.clientY,
                    isModifierActive(fineAdjustKb, ev),
                );
                const cents = computePitchDragCents(adjustedY - st.startClientY);
                st.currentCents = cents;
                setPitchDragTooltip({
                    text: formatDragTooltip(cents),
                    position: { x: ev.clientX, y: ev.clientY },
                });
                scheduleSend();
            }

            async function finish() {
                const st = dragRef.current;
                if (!st || finalized) return;
                finalized = true;
                dragRef.current = null;
                // 收尾第一步注销失焦守卫（幂等防双触发）。
                unregisterAbort();
                teardown();
                if (st.sendTimer != null) {
                    window.clearTimeout(st.sendTimer);
                    st.sendTimer = null;
                }
                setPitchDragTooltip(null);
                // 未产生偏移或基准帧未就绪：无变更，不落盘（也未开组）。
                if (!st.base || st.currentCents === 0) return;
                try {
                    await st.sendChain;
                    // 最终提交留在 undo group 内（checkpoint:false）：
                    // 组快照（拖拽前）+ 全部预览写入 = 单个撤销步。
                    await webApi.setParamFrames(
                        st.rootTrackId,
                        "pitch",
                        st.startFrame,
                        shiftPitchFrames(st.base, st.currentCents / 100),
                        false,
                    );
                } catch {
                    // 后端写入失败时保留现状；epoch 仍需推进以让编辑器回读。
                } finally {
                    if (undoGroupPromise) {
                        try {
                            await undoGroupPromise;
                        } catch {
                            // ignore
                        }
                        try {
                            await webApi.endUndoGroup();
                        } catch {
                            // 收尾失败不影响下一次拖拽（组已尽力关闭）。
                        }
                    }
                }
                // 通知参数编辑器重新取数（与拉伸联动参数线同一刷新通道）。
                dispatch(bumpParamsEpoch());
            }

            function onUp(ev: PointerEvent) {
                const st = dragRef.current;
                if (!st || ev.pointerId !== st.pointerId || finalized) return;
                void finish();
            }

            // 失焦取消：切屏期间 pointerup/pointercancel 不送达本窗口，注册
            // 事件无关的 finish()，由 gestureFocusGuard 在窗口 blur 时统一收尾。
            const unregisterAbort = registerDragAbort(() => {
                void finish();
            });

            window.addEventListener("pointermove", onMove, true);
            window.addEventListener("pointerup", onUp, true);
            window.addEventListener("pointercancel", onUp, true);

            // 异步拉取基准帧；响应回传实际 frame_period_ms —— 与探测假设不符时
            // 以真实周期重建取帧窗口（仅多一次往返，罕见路径）。
            // 以闭包中的 state 判归属，避免快速点击/重开拖拽时的错配竞态。
            void (async () => {
                try {
                    let res = await webApi.getParamFrames(
                        rootTrackId,
                        "pitch",
                        startFrame,
                        frameCount,
                        1,
                    );
                    const realFpMs = Number(res?.frame_period_ms) || PITCH_DRAG_PROBE_FP_MS;
                    if (res?.ok && realFpMs !== PITCH_DRAG_PROBE_FP_MS) {
                        if (dragRef.current === state) {
                            state.startFrame = Math.max(
                                0,
                                Math.floor((clip.startSec * 1000) / realFpMs),
                            );
                            state.frameCount = Math.max(
                                1,
                                Math.ceil((clip.lengthSec * 1000) / realFpMs),
                            );
                            res = await webApi.getParamFrames(
                                rootTrackId,
                                "pitch",
                                state.startFrame,
                                state.frameCount,
                                1,
                            );
                        }
                    }
                    if (dragRef.current !== state) return;
                    if (!res?.ok) {
                        dragRef.current = null;
                        finalized = true;
                        unregisterAbort();
                        teardown();
                        setPitchDragTooltip(null);
                        return;
                    }
                    state.base = (res.edit ?? []).map((v) => Number(v) || 0);
                    scheduleSend();
                } catch {
                    if (dragRef.current === state) {
                        dragRef.current = null;
                        teardown();
                        setPitchDragTooltip(null);
                    }
                }
            })();
        },
        [sessionRef, dispatch, fineAdjustKb, formatDragTooltip],
    );

    return { startClipPitchDrag, pitchDragTooltip };
}
