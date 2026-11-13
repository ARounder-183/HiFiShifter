/**
 * 自动滚屏工具栏按钮：左键开关，右键选择模式（滚动式 / 翻页式）。
 *
 * 【为什么单独一个文件】主工具栏（`ActionBar.tsx`）已经 2000+ 行；本按钮自带两枚
 * 图标、一份菜单与一条"选模式即开启"的策略，塞回去只会让那个文件更难读。
 *
 * 【交互协议】与 `RippleModeButton` 完全同构（左键动作 + 右键菜单），只是这里左键是
 * **开关**而不是"循环" —— 开关与模式是两根正交的轴，把模式塞进左键会毁掉肌肉记忆。
 *
 * 【两枚图标为什么必须是新的】旧图标（`@radix-ui/react-icons` 的
 * `DoubleArrowRightIcon`，两个右向雪佛龙）表达的是"继续向右"这一件事，无法承载
 * "两种跟随语义"。它已随本次改动**彻底弃用**（全仓不再引用）。新的一对在构图上
 * 刻意不同：滚动式 = 一个视口窗口 + 播放头 + 窗内连续位移；翻页式 = 当前页 →
 * 跨页箭头 → 下一页。二者共享 15×15 视框、`currentColor`、圆角端点，与相邻按钮
 * 同族但**一眼可分**。
 *
 * 【为什么把图标挂在菜单项上】用户是在菜单里做"图标 ↔ 模式"这个配对的；在选项旁边
 * 再画一次，选完回看按钮时不需要重新推断。
 */
import { useState } from "react";

import { useI18n } from "../../i18n/I18nProvider";
import { AppContextMenu, AppIconButton } from "../../ui";
import { AUTO_SCROLL_MODES, type AutoScrollMode } from "../../utils/autoFollowScroll";

/** 滚动式：一个视口窗口 + 播放头 + 连续右移的箭头（"视图跟着光标走"）。 */
export function AutoScrollContinuousIcon() {
    return (
        <svg
            width="15"
            height="15"
            viewBox="0 0 15 15"
            fill="none"
            aria-hidden="true"
            data-hs-auto-scroll-icon="continuous"
        >
            {/* 视口窗口（弱化：它是"画面"，不是"动作"） */}
            <rect
                x="1.4"
                y="3.1"
                width="12.2"
                height="8.8"
                rx="1.4"
                stroke="currentColor"
                strokeWidth="1"
                opacity="0.45"
            />
            {/* 播放头 */}
            <path d="M4.6 4.9V10.1" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
            {/* 连续位移：轨道 + 箭头 */}
            <path d="M6.6 7.5H11.3" stroke="currentColor" strokeWidth="1.1" strokeLinecap="round" />
            <path
                d="M9.5 5.8L11.6 7.5L9.5 9.2"
                stroke="currentColor"
                strokeWidth="1.1"
                strokeLinecap="round"
                strokeLinejoin="round"
            />
        </svg>
    );
}

/** 翻页式：当前页 / 下一页 + 跨页箭头（"越界才翻一整页"）。 */
export function AutoScrollPageIcon() {
    return (
        <svg
            width="15"
            height="15"
            viewBox="0 0 15 15"
            fill="none"
            aria-hidden="true"
            data-hs-auto-scroll-icon="page"
        >
            {/* 当前页（弱化） */}
            <rect
                x="1"
                y="3.1"
                width="4.8"
                height="8.8"
                rx="1.2"
                stroke="currentColor"
                strokeWidth="1"
                opacity="0.45"
            />
            {/* 下一页（强调） */}
            <rect
                x="9.2"
                y="3.1"
                width="4.8"
                height="8.8"
                rx="1.2"
                stroke="currentColor"
                strokeWidth="1.2"
            />
            {/* 跨页箭头 */}
            <path
                // 顶点必须落在两页之间那道空隙的正中（5.8..9.2 的中心 = 7.5），
                // 否则 200% 缩放下能看出它偏左半像素。
                d="M6.55 6.2L8.45 7.5L6.55 8.8"
                stroke="currentColor"
                strokeWidth="1.2"
                strokeLinecap="round"
                strokeLinejoin="round"
            />
        </svg>
    );
}

/** 按模式取图标。两处（按钮 / 菜单项）必须同源，否则会出现"按钮是 A、菜单写着 B"。 */
function iconFor(mode: AutoScrollMode) {
    return mode === "page" ? <AutoScrollPageIcon /> : <AutoScrollContinuousIcon />;
}

export function AutoScrollButton({
    enabled,
    mode,
    onToggle,
    onSelectMode,
}: {
    /** 自动滚屏是否开启（按钮的实心/强调态）。 */
    enabled: boolean;
    /** 当前模式（决定图标与 tooltip）。 */
    mode: AutoScrollMode;
    /** 左键：开关。 */
    onToggle: () => void;
    /** 右键菜单选中某个模式。 */
    onSelectMode: (mode: AutoScrollMode) => void;
}) {
    const { tf } = useI18n();
    const [menu, setMenu] = useState<{ x: number; y: number } | null>(null);

    return (
        <>
            <AppIconButton
                active={enabled}
                // 激活时用主题强调色（与相邻开关同款；不带 color 时 Radix 回落强调色）
                emphasis="accent"
                // tooltip 同时是 aria-label，格式固定为「功能名：模式名」（"自动滚屏：翻页式"）。
                // 它必须自带这两件事 —— 只写模式名的话，读屏用户听到的是一枚无名按钮；
                // 但也不该更长：悬停气泡是一句**标签**，不是说明书。"右键选模式"的说明
                // 写在用户手册里，不塞进气泡。
                tooltip={tf(`auto_scroll_tooltip_${mode}`)}
                tabIndex={-1}
                // 右下角小三角：提示"右键还有模式可选"（与参数编辑器的工具族按钮同一个记号）。
                menuCaret
                onClick={onToggle}
                onContextMenu={(e) => {
                    e.preventDefault();
                    setMenu({ x: e.clientX, y: e.clientY });
                }}
                icon={iconFor(mode)}
            />
            {menu && (
                <AppContextMenu
                    x={menu.x}
                    y={menu.y}
                    ariaLabel={tf("auto_scroll")}
                    onClose={() => setMenu(null)}
                    items={AUTO_SCROLL_MODES.map((value) => ({
                        key: `auto-scroll-mode-${value}`,
                        label: tf(`auto_scroll_mode_${value}`),
                        icon: iconFor(value),
                        checked: mode === value,
                        onSelect: () => {
                            onSelectMode(value);
                            setMenu(null);
                        },
                    }))}
                />
            )}
        </>
    );
}
