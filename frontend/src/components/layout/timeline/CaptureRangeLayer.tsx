/**
 * 时间轴上的**实时捕获带**：捕获中那些正在长出来的工程区间。
 *
 * 【为什么必须有它】捕获期间界面上必须**有东西在动**。后端每 200 ms 把捕获带的几何
 * 发出来，如果一处都不画，用户看到的是传输栏上一个"已捕获 0.0 秒"的数字加一条完全
 * 静止的时间轴 —— 这正是"不知道自己捕获了什么"这个失败感的来源。
 *
 * 【为什么是**多条**带】用户在捕获过程中可以把播放光标跳到任意位置再继续录。并集会
 * 把"录了 0..3 秒、跳到 30 秒、又录了 30..33 秒"画成一条 0..33 秒的连续带子，而中间
 * 27 秒什么都没录到 —— 那是撒谎。后端因此给"已覆盖区间列表"，这里逐段画。
 *
 * 【为什么右缘要逐帧跟光标】后端读数是 **5 Hz**（`captureStore.POLL_MS = 200`），而光标
 * 是「宿主推送 + rAF 插值」。只按读数画，带子每 200 ms 往前跳一格 —— 用户说的
 * "一顿一顿"就是它。现在右缘与位置线都读**同一份视觉播放头**（60 Hz，见
 * `visualPlayheadBus`），并以 `MAX_LEAD_SEC` 为上限，不会跑到空洞里去。
 *
 * 【为什么是内容坐标系】层内元素用 `sec × pxPerSec` 布局（与吸附高亮层同一约定），
 * 滚动由宿主整层 translate 跟随 —— 层内不重排。
 *
 * 【为什么边缘必须走 `devicePixelLine`】带子与位置线此前用 CSS `width: 1` 与
 * `borderRadius: pill`：在 dpr = 1.25 / 1.5 下 1 CSS px 是 1.25 / 1.5 物理像素，
 * 边缘落在半个物理像素上被抗锯齿 —— 用户报的"边缘线宽度在 1px~2px 之间浮动"就是它。
 */
import React from "react";

import { MAX_LEAD_SEC, useCaptureBands } from "../../../features/capture/captureRange";
import { readVisualPlayheadSec } from "../../../features/capture/visualPlayheadBus";
import { useDevicePixelRatio } from "../../../hooks/useDevicePixelRatio";
import { verticalHairlineGeometry } from "../../../utils/devicePixelLine";
import { CLIP_BODY_PADDING_Y } from "./constants";

/** 单行高度里留给带子的竖直内缩（与 clip 主体同一内缩，视觉上对齐）。 */
const BAND_INSET_Y = CLIP_BODY_PADDING_Y;

/** 位置线竖线的物理像素宽（与 `SnapHighlightLayer` 的竖线同一规格）。 */
const HEAD_PHYSICAL_PX = 2;

/**
 * 找到捕获轨在轨道列表里的行号。
 *
 * 【为什么不能比较完整 id】GUI 侧的轨道 id 已经被后端加了命名空间前缀
 * （`hfs-ui-<pid>-<n>-capture-track`，见 `session.rs` 的 `rewrite_ids`）。此前这里是
 * `tracks.findIndex(t => t.id === "capture-track")` —— 那个比较**恒为 −1**，
 * `Math.max(0, −1)` 于是把带子永远画在第 0 行。现在只有一条轨所以看不出来，
 * 用户一旦多建一条轨就会露馅。
 */
function captureRowIndex(tracks: ReadonlyArray<{ id: string }>): number {
    const index = tracks.findIndex(
        (track) => track.id === "capture-track" || track.id.endsWith("-capture-track"),
    );
    return index < 0 ? 0 : index;
}

export const CaptureRangeLayer: React.FC<{
    pxPerSec: number;
    rowHeight: number;
    /** 轨道顺序（决定带子画在第几行）。 */
    tracks: ReadonlyArray<{ id: string }>;
    contentWidth: number;
    contentHeight: number;
}> = ({ pxPerSec, rowHeight, tracks, contentWidth, contentHeight }) => {
    // dpr 参与物理像素几何：变化时必须重渲染（换屏 / 改系统缩放）。
    const dpr = useDevicePixelRatio();
    const capture = useCaptureBands();

    const bands = capture?.bands ?? [];
    const lastIndex = bands.length - 1;
    const lastBand = lastIndex >= 0 ? bands[lastIndex] : null;
    const lastBandRef = React.useRef<HTMLDivElement | null>(null);
    const headRef = React.useRef<HTMLDivElement | null>(null);

    // 【逐帧跟随】读视觉播放头（60 Hz），把最后一段的右缘与位置线拉到它。
    // 只在"在录且宿主在播"时延伸；`MAX_LEAD_SEC` 是硬上限（视觉值自带 100 ms 外推，
    // 轮询到达时它会跳回真实值 —— 有上限就不会画进空洞）。
    const lastStartSec = lastBand?.startSec ?? null;
    const lastEndSec = lastBand?.endSec ?? null;
    React.useEffect(() => {
        if (!(pxPerSec > 0)) return undefined;
        let frame = 0;
        const tick = () => {
            const visual = readVisualPlayheadSec();
            if (headRef.current) {
                const head = verticalHairlineGeometry(visual * pxPerSec, HEAD_PHYSICAL_PX, dpr);
                headRef.current.style.left = `${head.left}px`;
            }
            if (lastBandRef.current !== null && lastStartSec !== null && lastEndSec !== null) {
                const endSec = Math.min(Math.max(visual, lastEndSec), lastEndSec + MAX_LEAD_SEC);
                const left = Math.round(lastStartSec * pxPerSec * dpr) / dpr;
                const right = Math.round(endSec * pxPerSec * dpr) / dpr;
                lastBandRef.current.style.width = `${Math.max(1 / dpr, right - left)}px`;
            }
            frame = requestAnimationFrame(tick);
        };
        frame = requestAnimationFrame(tick);
        return () => cancelAnimationFrame(frame);
    }, [pxPerSec, dpr, lastStartSec, lastEndSec]);

    if (capture === null) return null;
    if (!(pxPerSec > 0)) return null;

    const rowIndex = captureRowIndex(tracks);
    const safeRowHeight = Math.max(1, rowHeight);
    const top = rowIndex * safeRowHeight + BAND_INSET_Y;
    const height = Math.max(2, safeRowHeight - BAND_INSET_Y * 2);

    // 实心带的两端都要落在**设备像素边界**上：只对齐起点、宽度仍是小数时，右缘会跨在
    // 两个物理像素之间被抗锯齿（与 `keyboardInstances` 里"白键铺满轴宽"同一条纪律）。
    const geometry = bands.map((band) => {
        const left = Math.round(band.startSec * pxPerSec * dpr) / dpr;
        const right = Math.round(band.endSec * pxPerSec * dpr) / dpr;
        return { left, width: Math.max(1 / dpr, right - left), right };
    });

    // 当前位置线：位置吸附 + 整数物理像素宽，成对做才恒为整数物理像素。
    const head = verticalHairlineGeometry(capture.positionSec * pxPerSec, HEAD_PHYSICAL_PX, dpr);

    return (
        <div
            className="pointer-events-none absolute left-0 top-0 overflow-hidden"
            style={{ width: contentWidth, height: contentHeight }}
            data-capture-range-layer
        >
            {/* 已捕获区间：逐段实心带。它表达"这一段已经有素材了"。
                圆角在分数 dpr 下会多出两条抗锯齿边缘，而它是进度指示不是按钮 —— 直角。 */}
            {geometry.map((band, index) => (
                <div
                    key={index}
                    ref={index === lastIndex ? lastBandRef : undefined}
                    data-capture-range-band
                    style={{
                        position: "absolute",
                        left: band.left,
                        top,
                        width: band.width,
                        height,
                        background: "var(--qt-danger-border)",
                        opacity: 0.45,
                    }}
                />
            ))}
            {/* 当前位置：一道细线。捕获中唯一"在动"的东西 —— 没有它，
                用户无法判断插件到底有没有在收音频。
                【为什么它不依赖 bands 非空】录制刚起、或 seek 之后还没有任何一段成立，
                此前整个层返回 null，连位置线一起消失（用户以为捕获停了）。 */}
            <div
                ref={headRef}
                data-capture-range-head
                style={{
                    position: "absolute",
                    left: head.left,
                    top,
                    width: head.width,
                    height,
                    background: "var(--qt-danger-text)",
                }}
            />
        </div>
    );
};
