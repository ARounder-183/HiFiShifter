/**
 * 设备像素比（DPR）的订阅入口。
 *
 * 【主要内容】提供两种形态的同一个能力：
 * - `subscribeDevicePixelRatio(cb)`：命令式订阅（供内核宿主这类非 React 渲染路径使用）；
 * - `useDevicePixelRatio()`：React 形态，返回当前 dpr 并在变化时重渲染。
 *
 * 【作用】DPR 变化（浏览器缩放 / 窗口拖到不同缩放率的显示器 / 改系统缩放）**不会**
 * 触发 `ResizeObserver` —— 后者观察的是 CSS 布局盒，而纯 DPR 变化不改变 CSS 尺寸。
 * 因此所有"按 dpr 光栅化"的画布都必须额外订阅一次，否则会停留在旧 dpr 的 backing
 * store 与几何上（画面发虚），直到下一次交互或滚动才恢复。
 *
 * 【实现要点：为什么必须换绑】`(resolution: N dppx)` 查询只在 dpr **离开** N 时
 * 触发一次；回调里若继续用旧 query，第二次缩放就永远收不到通知。故每次触发后都用
 * 新 dpr 重新订阅。
 *
 * 【与其他模块的关系】`readDevicePixelRatio` 来自 `utils/devicePixelLine`（同一份
 * 回退语义）。波形面、时间轴内核宿主、参数编辑器面板、颤音画布共用本模块，取代各自
 * 手写的 matchMedia 逻辑。
 */

import React from "react";

import { readDevicePixelRatio } from "../utils/devicePixelLine";

/**
 * 订阅设备像素比变化。
 *
 * 回调收到**新的** dpr（已由 `readDevicePixelRatio` 归一化）。返回退订函数，
 * 可安全重复调用。
 *
 * 环境不支持 `matchMedia`（SSR / 测试替身）时返回空退订函数，不抛错。
 */
export function subscribeDevicePixelRatio(onChange: (dpr: number) => void): () => void {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
        return () => undefined;
    }

    let disposed = false;
    let unbind: (() => void) | null = null;

    const bind = (): void => {
        if (disposed) return;
        const mql = window.matchMedia(`(resolution: ${readDevicePixelRatio()}dppx)`);
        const handleChange = (): void => {
            if (disposed) return;
            // 旧 query 已与新 dpr 失配，先解绑再用新 dpr 重新订阅。
            unbind?.();
            unbind = null;
            onChange(readDevicePixelRatio());
            bind();
        };
        mql.addEventListener("change", handleChange);
        unbind = () => mql.removeEventListener("change", handleChange);
    };

    bind();

    return () => {
        disposed = true;
        unbind?.();
        unbind = null;
    };
}

/**
 * React 形态：返回当前 dpr，并在其变化时触发重渲染。
 *
 * 用于把 dpr 参与**渲染输出**（内联样式里的物理像素线宽等）的组件。纯命令式绘制
 * （每帧现读 dpr）不需要它，用 `subscribeDevicePixelRatio` 只做"标脏"即可。
 */
export function useDevicePixelRatio(): number {
    const [dpr, setDpr] = React.useState(() => readDevicePixelRatio());
    React.useEffect(() => subscribeDevicePixelRatio(setDpr), []);
    return dpr;
}
