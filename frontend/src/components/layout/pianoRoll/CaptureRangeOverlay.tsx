/**
 * 参数编辑器标尺上的**捕获区间指示**。
 *
 * 【为什么参数编辑器也要画】插件形态的默认视图是"时间轴 / 参数编辑器"并列标签页，
 * 且默认停在**参数编辑器**上 —— 时间轴是次要工作面（宿主自己的轨道已经承担了那部分
 * 职责）。捕获期间如果只有时间轴上有红色带子，那么"只看参数编辑器"的用户在整个
 * 捕获过程中**看不到任何东西**：他不知道捕获从哪开始、到哪结束、有没有在推进。
 *
 * 【为什么画在标尺上而不是画布中央】画布主体是参数曲线，一条横贯全高的红带会盖住
 * 用户正在看的数据。标尺区是"时间位置"的既有表达区（播放头也在那里），且高度足够
 * 看清一条 4px 的带子。
 *
 * 【为什么几何走 `devicePixelLine`】与时间轴那一层同一条纪律：只吸附位置不取整宽度、
 * 或只取整宽度不吸附位置，在 dpr = 1.25 / 1.5 下都会让边缘在 1~2 物理像素之间浮动。
 */
import React from "react";

import { useCaptureRange } from "../../../features/capture/captureRange";
import { useDevicePixelRatio } from "../../../hooks/useDevicePixelRatio";
import { verticalHairlineGeometry } from "../../../utils/devicePixelLine";

/** 带子高度（CSS px）。够看清，又不压住标尺上的时间标签。 */
const BAND_HEIGHT_PX = 4;
/** 右缘竖线的物理像素宽。 */
const HEAD_PHYSICAL_PX = 2;

export const CaptureRangeOverlay: React.FC<{
    pxPerSec: number;
    /** 当前横向滚动（工程秒 → 屏幕 x 的换算基准）。 */
    scrollLeft: number;
    /** 视口宽度：区间完全在视野外时不渲染，省掉一次无意义的布局。 */
    viewportWidth: number;
}> = ({ pxPerSec, scrollLeft, viewportWidth }) => {
    const dpr = useDevicePixelRatio();
    const range = useCaptureRange();
    if (range === null) return null;
    if (!(pxPerSec > 0) || !(viewportWidth > 0)) return null;

    // 内容坐标 → 视口坐标。两端都吸附到设备像素栅格（见文件头）。
    const toViewport = (sec: number) => Math.round((sec * pxPerSec - scrollLeft) * dpr) / dpr;
    const left = toViewport(range.startSec);
    const right = toViewport(range.endSec);
    const width = Math.max(1 / dpr, right - left);
    // 完全在视野外：不渲染（也避免为不可见内容创建 DOM）。
    if (right < 0 || left > viewportWidth) return null;

    const head = verticalHairlineGeometry(right, HEAD_PHYSICAL_PX, dpr);

    return (
        <div
            className="pointer-events-none absolute inset-x-0 bottom-0"
            style={{ height: BAND_HEIGHT_PX }}
            data-capture-range-overlay
        >
            <div
                data-capture-range-overlay-band
                style={{
                    position: "absolute",
                    left,
                    top: 0,
                    width,
                    height: BAND_HEIGHT_PX,
                    background: "var(--qt-danger-border)",
                    opacity: 0.55,
                }}
            />
            <div
                data-capture-range-overlay-head
                style={{
                    position: "absolute",
                    left: head.left,
                    top: 0,
                    width: head.width,
                    height: BAND_HEIGHT_PX,
                    background: "var(--qt-danger-text)",
                }}
            />
        </div>
    );
};
