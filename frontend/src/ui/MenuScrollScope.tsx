/**
 * 菜单的**滚动作用域**：两条 sticky 提示行 + 一层内容容器 + 自动滚动控制器。
 *
 * 【为什么菜单不用滚动条】菜单是瞬态表面，滚动条是常驻机械装置。它还偷宽度
 * （原生条 10px）：出现时内容右缘整体左移，右对齐的快捷键列立刻不在同一列。
 * 而它给出的语义只是"能滚"，不说"这个方向还有几项"。
 *
 * 替代品是两条**渐变 + 三角**的提示行：内容在边缘溶进菜单底色（这是"下面还有"
 * 的语法），中间一支小箭头（这是"可以滚"的语法）。悬停慢滚、点击翻页、长按快滚。
 *
 * 【为什么提示行是 `position: sticky`】它必须钉在滚动视口边缘又不随内容移动。
 * 做成"壳 + 内部视口 + 兄弟提示行"要给**每个**菜单表面加一层结构，还会多出一条
 * "内容变化 → 提示行重算"的链路；sticky 只需两个 0 高元素，实测零布局代价
 * （`scrollHeight` 574 → 574）、精确钉边、伪元素命中有效。
 *
 * 【为什么壳的 `padding` 搬到了内容层】sticky 的约束盒是父元素的**内容盒**：
 * 壳若留 4px 内边距，提示行会跟着内缩 4px，边缘露出一条没被渐变的空白。
 *
 * 【为什么可见性写在提示行自己身上】子面板是父壳的 DOM 后代。把状态写到祖先壳上
 * 会让"父菜单在滚"把"子面板的提示行"一起点亮 —— 两层的提示行不该联动。
 */
import { useCallback, useLayoutEffect, useRef } from "react";
import type { ReactNode } from "react";

import { cx } from "./cx";
import { MENU_SCROLL_EPSILON_PX } from "./menuScrollModel";
import { menuItemHeightOf, menuScrollContainerOf } from "./menuScrollDom";
import { useMenuAutoScroll } from "./useMenuAutoScroll";

/**
 * 箭头：与子菜单的右箭头**同一支笔画**（1.4px、圆头、`currentColor`）。
 *
 * 【为什么是描边而不是填充三角】填充三角是"工具提示箭头"的语汇；菜单里的方向
 * 提示属于"表面边缘"，描边更轻。
 *
 * 【基础形状朝下】两条提示行共用这一支笔画：**下方**提示行不旋转（"下面还有"），
 * **上方**提示行旋转 180°（"上面还有"）。因此基础路径必须是朝下的
 * （`M1 1L5 5L9 1`：两端在上、中间在下）。
 *
 * 写成朝上的（`M1 5L5 1L9 5`）再加旋转，两条提示行的箭头会**同时指反** ——
 * 底部那条指回上方、顶部那条指向下方，读起来恰好与"还能往哪滚"相反。这一处
 * 只有肉眼能发现：几何、命中区、可见性断言全部通过。
 */
const SCROLL_CARET = (
    <svg
        className="hs-menu__scroll-caret"
        width="10"
        height="6"
        viewBox="0 0 10 6"
        fill="none"
        aria-hidden="true"
    >
        <path
            d="M1 1L5 5L9 1"
            stroke="currentColor"
            strokeWidth="1.4"
            strokeLinecap="round"
            strokeLinejoin="round"
        />
    </svg>
);

export interface MenuScrollScopeProps {
    /**
     * 内容层的追加类。
     *
     * 表单型菜单的 `flex flex-col gap-2` 必须写在这里而不是壳上：壳已经不是内容
     * 的直接父级（内容被包了一层），把布局类留在壳上会作用到那个包装元素上。
     */
    contentClassName?: string;
    /**
     * false = 内容有界，永不滚动（不渲染提示行）。
     *
     * 【纪律】只有"内容高度可证明有界"的表面才准用 —— 它让菜单在矮窗口里直接
     * 溢出视口，条目不可达。`--no-scroll` 壳（表单、淡变曲率面板、节拍器菜单）
     * 属于这一类：它们的内容由组件自己约束。
     *
     * 即使传 true，运行期仍会再判一次"壳是不是真的可滚"（`--no-scroll` 的
     * `overflow: visible` 会让它自动关闭）—— 静态声明只是让 DOM 更干净。
     */
    scrollable?: boolean;
    children: ReactNode;
}

/**
 * 菜单的滚动作用域。
 *
 * **children 无条件渲染** —— 下面任何一步失败（找不到滚动容器、ResizeObserver
 * 不存在、令牌读不到），退化的都是"提示"，而不是"内容不见了"。
 */
export function MenuScrollScope({
    contentClassName,
    scrollable = true,
    children,
}: MenuScrollScopeProps) {
    const contentRef = useRef<HTMLDivElement | null>(null);
    const upRef = useRef<HTMLDivElement | null>(null);
    const downRef = useRef<HTMLDivElement | null>(null);
    const viewportRef = useRef<HTMLElement | null>(null);
    const pressingRef = useRef(false);
    const stateRef = useRef({ canUp: false, canDown: false });

    const getViewport = useCallback(() => viewportRef.current, []);
    const getItemHeight = useCallback(() => menuItemHeightOf(viewportRef.current), []);

    const applyVisibility = useCallback(() => {
        const { canUp, canDown } = stateRef.current;
        const up = upRef.current;
        const down = downRef.current;
        // `delete` 而不是写空串：`[data-visible="1"]` 只在真的可见时命中，
        // 残留的空属性会让"有没有提示"这件事在 DOM 里看不出来。
        if (up) {
            if (canUp) up.dataset.visible = "1";
            else delete up.dataset.visible;
        }
        if (down) {
            if (canDown) down.dataset.visible = "1";
            else delete down.dataset.visible;
        }
    }, []);

    const sync = useCallback(() => {
        const viewport = viewportRef.current;
        if (!viewport) return;
        const maxScroll = Math.max(0, viewport.scrollHeight - viewport.clientHeight);
        stateRef.current = {
            canUp:
                maxScroll > MENU_SCROLL_EPSILON_PX && viewport.scrollTop > MENU_SCROLL_EPSILON_PX,
            canDown:
                maxScroll > MENU_SCROLL_EPSILON_PX &&
                viewport.scrollTop < maxScroll - MENU_SCROLL_EPSILON_PX,
        };
        // 按下期间冻结（见 `useMenuAutoScroll` 的说明）：滚到尽头会让提示行隐藏，
        // 而隐藏 = `pointer-events: none` = 收不到抬起事件。
        if (pressingRef.current) return;
        applyVisibility();
    }, [applyVisibility]);

    const autoScroll = useMenuAutoScroll({
        getViewport,
        getItemHeight,
        pressingRef,
        enabled: scrollable,
        onStateChange: sync,
    });
    const { interrupt } = autoScroll;

    useLayoutEffect(() => {
        if (!scrollable) return;
        const content = contentRef.current;
        if (!content) return;
        const viewport = menuScrollContainerOf(content);
        viewportRef.current = viewport;
        if (!viewport) {
            // `--no-scroll` 壳：整段关闭，提示行保持隐藏。
            applyVisibility();
            return;
        }
        sync();
        viewport.addEventListener("scroll", sync, { passive: true });
        // 滚轮介入即停自动滚动：用户已经自己接管了，此时还在替他滚就是抢。
        viewport.addEventListener("wheel", interrupt, { passive: true });
        /*
         * 内容与视口都观测：
         *   - 内容：本地化文案换行、异步枚举的列表落地 → 菜单变高 → 提示行要重算；
         *   - 视口：窗口 resize 改变 `max-height` → 同上。
         * `ResizeObserver` 在 jsdom 里不存在，必须判空（否则整个组件挂不上）。
         */
        const observer =
            typeof ResizeObserver === "undefined" ? null : new ResizeObserver(() => sync());
        observer?.observe(content);
        observer?.observe(viewport);
        return () => {
            viewport.removeEventListener("scroll", sync);
            viewport.removeEventListener("wheel", interrupt);
            observer?.disconnect();
        };
    }, [applyVisibility, interrupt, scrollable, sync]);

    const bindUp = autoScroll.bind(-1);
    const bindDown = autoScroll.bind(1);

    return (
        <>
            {scrollable ? (
                <div
                    ref={upRef}
                    className="hs-menu__scroll-hint hs-menu__scroll-hint--up"
                    aria-hidden="true"
                    {...bindUp}
                >
                    {SCROLL_CARET}
                </div>
            ) : null}
            <div ref={contentRef} className={cx("hs-menu__content", contentClassName)}>
                {children}
            </div>
            {scrollable ? (
                <div
                    ref={downRef}
                    className="hs-menu__scroll-hint hs-menu__scroll-hint--down"
                    aria-hidden="true"
                    {...bindDown}
                >
                    {SCROLL_CARET}
                </div>
            ) : null}
        </>
    );
}
