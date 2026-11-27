/**
 * 参数工具栏的「⋯」：宽度不足时，把右簇的控件**搬进菜单**而不是藏掉。
 *
 * 【为什么不是隐藏】让位阶梯的 1-9 级砍的都是冗余（重复的标题文字、可以缩小的滑块）。
 * 10 级之后剩下的每一项都是**唯一入口** —— 算法下拉、静态参数、参数药丸。把它们藏掉
 * 等于"面板一窄，某个功能就消失了"，那比排版难看严重得多。搬进菜单则宽度归零、
 * 功能一个不少。
 *
 * 【为什么菜单内容直接复用那三个组件】它们的交互（药丸的下拉、眼睛、门禁置灰、
 * 分离开关）全部写在组件里。为了"窄的时候换个排布"再抄一份，就等于从此有两套
 * 参数按钮要同步 —— 这正是本仓反复吃过的亏。这里只换**排布方向**。
 *
 * 【为什么菜单内容只在打开时存在】菜单关闭即卸载。行内那份被 `ToolbarSlot` 用
 * `display: none` 留在 DOM 里（测量需要它还在），而菜单里的这一份只在用户真的打开时
 * 挂载 —— 于是同一时刻最多只有一份在跑。
 *
 * 【菜单里的副本必须"全量"渲染】这些组件都从模型里读 `toolbarTier` 来决定自己要不要
 * 让位。把**行内**的层级原样传进去，等于在菜单里再让位一次：第 11 级时张力 / 气声
 * 药丸、参考轨道组的文字、导入 MIDI 都会在菜单里一起消失 —— 而"搬进菜单"的全部意义
 * 就是功能一个不少。这里统一改传 `NEVER_HIDE_TIER`（负数 = 不参与让位）。
 *
 * 【触发器为什么常挂载】见 `ToolbarSlot.showAt`：若按层级条件渲染，测量第 0-9 级时
 * 按钮仍在 DOM 里、被算进宽度，于是同一宽度会出现两个答案（收窄到 10 级后要再拉宽
 * 28px 才恢复）。
 *
 * 【"菜单里再开菜单"】面板里住着 `ReferenceTrackGroup` 与 `ChildPitchGroup`，它们
 * 各自带一个下拉。点内层下拉的条目时外层**不能**关 —— 这条由 `ui/menuDismissStack`
 * 的层栈保证（锚定菜单在打开时登记自己）。
 */
import { useRef, useState } from "react";
import { MixerHorizontalIcon } from "@radix-ui/react-icons";

import { AppAnchoredMenu, AppIconButton } from "../../../../ui";
import { ChildPitchGroup } from "./ChildPitchGroup";
import { ParamButtons } from "./ParamButtons";
import { ParamStaticControls } from "./ParamStaticControls";
import { ReferenceTrackGroup } from "./ReferenceTrackGroup";
import { ToolbarSlot } from "./toolbarParts";
import { NEVER_HIDE_TIER, TIER_ALL_PILLS, TIER_STATIC_PARAMS } from "../toolbarOverflow";
import type { ParamToolbarModel } from "./ParamToolbar";

export function ToolbarOverflowMenu({ m }: { m: ParamToolbarModel }) {
    const { tf, toolbarTier } = m;
    const pillsCollapsed = toolbarTier >= TIER_ALL_PILLS;
    /** 菜单里的副本：不参与让位（见模块头）。 */
    const menuModel: ParamToolbarModel = { ...m, toolbarTier: NEVER_HIDE_TIER };

    const anchorRef = useRef<HTMLButtonElement | null>(null);
    const menuRef = useRef<HTMLDivElement | null>(null);
    const [open, setOpen] = useState(false);

    return (
        <>
            {/* 前 9 级只砍冗余、没有东西可收 —— 按钮从第 10 级起才出现。常挂载 +
                槽位控制显隐（而不是条件渲染），理由见模块头。 */}
            <ToolbarSlot showAt={TIER_STATIC_PARAMS} tier={toolbarTier}>
                {/* 【为什么是 ref 而不是 asChild】Radix 的 Trigger 需要 asChild 才能避免
                    `<button><button/></button>`（无效 HTML，且实测点击与键盘都打不开
                    菜单）。换成共享原语后不再有嵌套按钮的问题：触发器就是这一个按钮，
                    菜单按它的矩形定位。 */}
                <AppIconButton
                    ref={anchorRef}
                    tooltip={tf("param_toolbar_more")}
                    aria-label={tf("param_toolbar_more")}
                    tabIndex={-1}
                    icon={<MixerHorizontalIcon />}
                    aria-haspopup="menu"
                    aria-expanded={open}
                    onClick={() => setOpen((value) => !value)}
                />
            </ToolbarSlot>
            <AppAnchoredMenu
                open={open}
                anchorRef={anchorRef}
                menuRef={menuRef}
                onClose={() => setOpen(false)}
                // 触发器在工具栏最右侧：左对齐展开会横跨整个面板。
                align="end"
                ariaLabel={tf("param_toolbar_more")}
            >
                {/* 【为什么内边距写在这层 div 上而不是内容层】`.hs-menu__content` 的
                    `padding` 写在未分层的样式表里，优先级高于 Tailwind 工具类 ——
                    在内容层写 `p-1` 会被它压掉。 */}
                <div className="flex flex-col gap-2 p-1">
                    {pillsCollapsed ? (
                        <div className="flex flex-col items-start gap-2">
                            <ReferenceTrackGroup m={menuModel} />
                            <ChildPitchGroup m={menuModel} />
                            <div className="flex flex-wrap items-center gap-1">
                                <ParamButtons m={menuModel} />
                            </div>
                        </div>
                    ) : null}
                    <ParamStaticControls m={menuModel} />
                </div>
            </AppAnchoredMenu>
        </>
    );
}
