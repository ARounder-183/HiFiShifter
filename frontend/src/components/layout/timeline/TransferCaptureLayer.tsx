/**
 * 时间轴上的**实时录制带**：录制中那条正在长出来的工程区间。
 *
 * 【为什么必须有它】录制期间界面上必须**有东西在动**。此前后端每 200 ms 就把录制带的
 * 几何（起点 / 右边缘 / 当前位置）发出来，而前端一处都没画 —— 于是用户看到的是传输栏上
 * 一个"已录 0.0 秒"的数字加一条完全静止的时间轴，没有任何依据判断"我录的东西在哪、
 * 录到哪了"。这正是"不知道自己录了什么"这个失败感的来源。
 *
 * 【为什么是内容坐标系】层内元素用 `sec × pxPerSec` 布局（与吸附高亮层同一约定），
 * 滚动由宿主整层 translate 跟随 —— 层内不重排。录制读数 5 Hz 更新、滚动 60 Hz，
 * 让这一层自己跟滚动等于把两者绑在一起。
 *
 * 【为什么画在 Transfer 轨道上】Transfer 的素材只有一条流（宿主输入总线），时间轴上
 * 也只有一条轨（见后端 `TRANSFER_TRACK_ID`）。带子画在那条轨上，于是"录到的东西会
 * 出现在哪里"这件事在录制过程中就看得见 —— 而不是录完才揭晓。
 */
import React, { useSyncExternalStore } from "react";

import { frameToSec } from "../../../features/transfer/transferApi";
import { getTransferSnapshot, subscribeTransfer } from "../../../features/transfer/transferStore";
import { CLIP_BODY_PADDING_Y } from "./constants";

/** 单行高度里留给带子的竖直内缩（与 clip 主体同一内缩，视觉上对齐）。 */
const BAND_INSET_Y = CLIP_BODY_PADDING_Y;

export const TransferCaptureLayer: React.FC<{
    pxPerSec: number;
    rowHeight: number;
    /** 轨道顺序（决定带子画在第几行）。 */
    tracks: ReadonlyArray<{ id: string }>;
    contentWidth: number;
    contentHeight: number;
}> = ({ pxPerSec, rowHeight, tracks, contentWidth, contentHeight }) => {
    const snapshot = useSyncExternalStore(subscribeTransfer, getTransferSnapshot);
    const capture = snapshot.status?.capture ?? null;
    if (capture === null) return null;
    const rate = capture.sampleRate;
    if (!(rate > 0) || !(pxPerSec > 0)) return null;

    // 带子所在的轨道：Transfer 只有一条轨，取第一条即可 —— 与后端投影一致
    // （见 `transfer::timeline::TRANSFER_TRACK_ID`）。
    const rowIndex = Math.max(
        0,
        tracks.findIndex((track) => track.id === "transfer-track"),
    );
    const safeRowHeight = Math.max(1, rowHeight);
    const startSec = frameToSec(capture.startFrame, rate);
    const endSec = frameToSec(Math.max(capture.endFrame, capture.positionFrame), rate);
    const left = startSec * pxPerSec;
    const width = Math.max(1, (endSec - startSec) * pxPerSec);
    const top = rowIndex * safeRowHeight + BAND_INSET_Y;
    const height = Math.max(2, safeRowHeight - BAND_INSET_Y * 2);

    return (
        <div
            className="pointer-events-none absolute left-0 top-0 overflow-hidden"
            style={{ width: contentWidth, height: contentHeight }}
            data-transfer-capture-layer
        >
            {/* 已录区间：实心带。它表达"这一段已经有素材了"。 */}
            <div
                data-transfer-capture-band
                style={{
                    position: "absolute",
                    left,
                    top,
                    width,
                    height,
                    background: "var(--qt-danger-border)",
                    opacity: 0.45,
                    borderRadius: "var(--qt-radius-pill)",
                }}
            />
            {/* 当前位置：右边缘的一道细线。录制中唯一"在动"的东西 —— 没有它，
                用户无法判断插件到底有没有在收音频。 */}
            <div
                data-transfer-capture-head
                style={{
                    position: "absolute",
                    left: left + width,
                    top,
                    width: 1,
                    height,
                    background: "var(--qt-danger-text)",
                }}
            />
        </div>
    );
};
