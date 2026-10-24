import { useLayoutEffect, useRef, useState, type RefObject } from "react";

import { TOOLBAR_MAX_TIER } from "./toolbarOverflow";
import { chooseTier, layoutSignature, measureTierFits } from "./toolbarGauge";

/**
 * 行宽在测量途中变化时，最多连续重试几帧。
 *
 * 行宽变化意味着"这一次量到的表横跨了多个宽度"（见 `measureTierFits`），只能作废重来。
 * 上限是为了防止"永远量不到"（例如行由内容撑开）变成每帧一次的无限重试。
 */
const MAX_WIDTH_RETRIES = 4;

/**
 * 观察工具栏行容器，返回当前应隐藏到第几级。
 *
 * # 一次决策，而不是逐帧收敛
 *
 * 每次内容或宽度变化时，在**真实行**上于一个同步任务里逐级问"放不放得下"
 * （见 `measureTierFits`），再一次性算出目标层级。浏览器只在任务结束时绘制，
 * 因此用户看不到"先撑满再收回去"的中间态，也不会看到被让位的药丸闪一下。
 *
 * # 触发时机（三条路，缺一不可）
 * - `ResizeObserver`：容器尺寸变化（拖拽分栏 / 浮窗拉伸）——不会触发 React 渲染；
 * - `MutationObserver`：行内**内容**变化（参数增删、语系切换、药丸多出 chevron）；
 * - `document.fonts`：字体替换会改掉每个标签的宽度，却**不产生任何 DOM 变更**。
 *
 * # 为什么先用签名过滤
 * 平滑度百分比在拖滑块时每帧都改文本；不过滤就等于把一次强制布局塞进拖拽热路径。
 * 签名只包含影响宽度的部分（见 `layoutSignature`），不变就直接返回。
 *
 * # 不在观察回调里同步 setState
 * `ResizeObserver` 回调里同步 setState 会形成"每帧投递"的永久循环。这里统一用
 * `requestAnimationFrame` 合帧，同帧内多次触发只测一次。
 *
 * # 为什么是 `useLayoutEffect`
 * `useEffect` 在**绘制之后**才跑：打开面板的第一帧必然先画出"全部显示"，然后再收，
 * 那是一次肉眼可见的抖动。布局副作用在提交后、绘制前执行，量完当场定层级，首帧就是
 * 正确的。
 *
 * # 两条"不打断用户"的规则
 *
 * 1. **指针按下期间不测量**。测量会临时把一排槽位 `display: none`；正被拖动的那只
 *    滑块一旦被隐藏，拖动会立刻中断（元素离开布局，指针捕获失效）。抬起（或窗口失焦）
 *    后再补一次。
 * 2. **测量前后保存/恢复行内焦点**。隐藏含焦点元素的槽位会让它 `blur`，而恢复
 *    `display` 并不会把焦点还回去 —— 键盘用户会莫名其妙掉焦点。
 *
 * @param rowRef 工具栏行容器的 ref
 * @param maxTier 最大隐藏层级
 * @returns 当前隐藏层级；`0` = 全部显示
 */
export function useToolbarOverflowTier(
    rowRef: RefObject<HTMLElement | null>,
    maxTier: number = TOOLBAR_MAX_TIER,
): number {
    const [tier, setTier] = useState(0);
    /**
     * 最近一次**落到 DOM 上**的层级。
     *
     * 用 ref 而不是直接读 `tier`：测量与状态更新发生在同一个 rAF 里，读 `tier` 拿到
     * 的可能是上一次提交的旧值；而 DOM 上实际是"最后一次测量留下的那一级"。测量失败
     * 时要回到的正是这一级 —— 回到 0 会把用户已收起的工具栏重新撑开一帧。
     */
    const tierRef = useRef(0);

    useLayoutEffect(() => {
        const row = rowRef.current;
        if (!row) return;

        let disposed = false;
        let frame = 0;
        /** 上一次**成功**测量时的"布局签名"与可用宽度，见 `layoutSignature`。 */
        let lastSignature: string | null = null;
        let lastAvailable = -1;
        /** DOM 当前所在的层级：测量结束会把行留在这一级（或新的那一级）。 */
        let applied = tierRef.current;
        let widthRetries = 0;
        /** 指针是否正按在行内（见模块头"不打断用户"第 1 条）。 */
        let pointerDown = false;

        const schedule = () => {
            if (disposed || frame !== 0) return;
            frame = requestAnimationFrame(measure);
        };

        const measure = () => {
            frame = 0;
            if (disposed) return;
            const available = row.clientWidth;
            // 0 = 面板当前不可见（停泊中的标签页 / 浮窗收起）。此时量不到真实宽度，
            // 保留上一次的层级，等真的显示出来再测 —— 不能把它当成"窄到要全藏"。
            if (available <= 0) return;
            if (pointerDown) return;

            const signature = layoutSignature(row);
            if (signature === lastSignature && available === lastAvailable) return;

            const focus = captureFocus(row);
            let fitsByTier: boolean[];
            try {
                fitsByTier = measureTierFits(row, maxTier, applied);
            } finally {
                // 测量会临时隐藏一排槽位（含可能持有焦点的那个），无论成败都要还回去。
                restoreFocus(focus);
            }

            if (fitsByTier.length === 0) {
                // 宽度在测量途中变了：这张表作废（DOM 已被恢复到 `applied`）。
                // 不记签名 —— 下一帧要按同样的输入重试。
                widthRetries += 1;
                if (widthRetries <= MAX_WIDTH_RETRIES) schedule();
                return;
            }

            widthRetries = 0;
            lastSignature = signature;
            lastAvailable = available;
            const next = chooseTier(fitsByTier);
            applied = next;
            tierRef.current = next;
            setTier((previous) => (previous === next ? previous : next));
        };

        const cleanups: Array<() => void> = [];

        const resizeObserver = new ResizeObserver(schedule);
        resizeObserver.observe(row);
        cleanups.push(() => resizeObserver.disconnect());

        const mutationObserver = new MutationObserver(schedule);
        mutationObserver.observe(row, { childList: true, subtree: true, characterData: true });
        cleanups.push(() => mutationObserver.disconnect());

        // 指针守卫：按下时挂起测量，抬起后补测一次。
        //
        // 【为什么还要监听 blur】用户在拖动途中 Alt+Tab 切走时，pointerup 未必送回来；
        // 少了这一条，"按住"的标记会一直挂着，工具栏从此再也不重新测量。
        const onPointerDown = () => {
            pointerDown = true;
        };
        const onPointerRelease = () => {
            if (!pointerDown) return;
            pointerDown = false;
            schedule();
        };
        const onWindowBlur = () => {
            if (!pointerDown) return;
            pointerDown = false;
            schedule();
        };
        row.addEventListener("pointerdown", onPointerDown, true);
        window.addEventListener("pointerup", onPointerRelease, true);
        window.addEventListener("pointercancel", onPointerRelease, true);
        window.addEventListener("blur", onWindowBlur);
        cleanups.push(() => {
            row.removeEventListener("pointerdown", onPointerDown, true);
            window.removeEventListener("pointerup", onPointerRelease, true);
            window.removeEventListener("pointercancel", onPointerRelease, true);
            window.removeEventListener("blur", onWindowBlur);
        });

        // 字体是**迟到**的宽度来源：替换字体不改 DOM，签名不会变。`ready` 覆盖
        // "首屏时还在加载"的那一批，`loadingdone` 覆盖之后按需加载的字形。
        const onFontsChanged = () => {
            lastSignature = null;
            schedule();
        };
        if (typeof document !== "undefined" && "fonts" in document) {
            const fonts = document.fonts;
            void fonts.ready.then(onFontsChanged);
            fonts.addEventListener("loadingdone", onFontsChanged);
            cleanups.push(() => fonts.removeEventListener("loadingdone", onFontsChanged));
        }

        // 首帧**同步**量一次：本 effect 在绘制前执行，量完当场 setState，React 会在同一帧
        // 提交 —— 用户看到的第一帧就是正确的层级。若这里改成 `schedule()`（下一帧的 rAF），
        // 第一帧必然先画出"全部显示"，再收回去，那就是一次肉眼可见的抖动。
        measure();

        return () => {
            disposed = true;
            if (frame !== 0) cancelAnimationFrame(frame);
            for (const cleanup of cleanups) cleanup();
        };
        // 刻意**不**依赖 `tier`：层级变化只改样式，不改变内容宽度，再测一遍是白跑；
        // 旧实现把 `tier` 放进依赖里，于是每降一级都要再等一帧，形成可见的阶梯。
    }, [rowRef, maxTier]);

    return tier;
}

/**
 * 记下"当前焦点在行内"的事实，供 `restoreFocus` 使用。
 *
 * 只关心**行内**的元素：焦点在外面时，测量不该碰它。
 */
function captureFocus(row: HTMLElement): HTMLElement | null {
    const active = document.activeElement;
    if (!(active instanceof HTMLElement)) return null;
    if (active === row || !row.contains(active)) return null;
    return active;
}

/**
 * 把测量过程中被 `display: none` 弄丢的焦点还回去。
 *
 * 只在"确实丢给了 `body`"时动手：若焦点已经落到别的元素上（用户在这期间点了
 * 别处、或某个弹层自己抢了焦点），那是有意的，不该抢回来。
 */
function restoreFocus(snapshot: HTMLElement | null): void {
    if (!snapshot || !snapshot.isConnected) return;
    const active = document.activeElement;
    if (active === snapshot) return;
    if (active !== null && active !== document.body) return;
    snapshot.focus({ preventScroll: true });
}
