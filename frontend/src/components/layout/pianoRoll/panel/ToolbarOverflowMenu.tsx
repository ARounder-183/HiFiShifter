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
 * 【为什么菜单内容只在打开时存在】Radix 的 `Content` 关闭即卸载。行内那份被
 * `ToolbarSlot` 用 `display: none` 留在 DOM 里（测量需要它还在），而菜单里的这一份
 * 只在用户真的打开时挂载 —— 于是同一时刻最多只有一份在跑。
 *
 * 【菜单里的副本必须"全量"渲染】这些组件都从模型里读 `toolbarTier` 来决定自己要不要
 * 让位。把**行内**的层级原样传进去，等于在菜单里再让位一次：第 11 级时张力 / 气声
 * 药丸、参考轨道组的文字、导入 MIDI 都会在菜单里一起消失 —— 而"搬进菜单"的全部意义
 * 就是功能一个不少。这里统一改传 `NEVER_HIDE_TIER`（负数 = 不参与让位）。
 *
 * 【触发器为什么常挂载】见 `ToolbarSlot.showAt`：若按层级条件渲染，测量第 0-9 级时
 * 按钮仍在 DOM 里、被算进宽度，于是同一宽度会出现两个答案（收窄到 10 级后要再拉宽
 * 28px 才恢复）。
 */
import type React from "react";
import { DropdownMenu, Flex } from "@radix-ui/themes";
import { MixerHorizontalIcon } from "@radix-ui/react-icons";

import { AppIconButton } from "../../../../ui";
import { ChildPitchGroup } from "./ChildPitchGroup";
import { ParamButtons } from "./ParamButtons";
import { ParamStaticControls } from "./ParamStaticControls";
import { ReferenceTrackGroup } from "./ReferenceTrackGroup";
import { ToolbarSlot } from "./toolbarParts";
import { NEVER_HIDE_TIER, TIER_ALL_PILLS, TIER_STATIC_PARAMS } from "../toolbarOverflow";
import type { ParamToolbarModel } from "./ParamToolbar";

/** `DropdownMenu.Trigger` 加一个运行时存在、类型里没声明的 `asChild`。 */
const TriggerAsChild = DropdownMenu.Trigger as unknown as React.ComponentType<
    React.ComponentProps<typeof DropdownMenu.Trigger> & { asChild?: boolean }
>;

export function ToolbarOverflowMenu({ m }: { m: ParamToolbarModel }) {
    const { tf, toolbarTier } = m;
    const pillsCollapsed = toolbarTier >= TIER_ALL_PILLS;
    /** 菜单里的副本：不参与让位（见模块头）。 */
    const menuModel: ParamToolbarModel = { ...m, toolbarTier: NEVER_HIDE_TIER };

    return (
        <DropdownMenu.Root>
            {/* 前 9 级只砍冗余、没有东西可收 —— 按钮从第 10 级起才出现。常挂载 +
                槽位控制显隐（而不是条件渲染），理由见模块头。 */}
            <ToolbarSlot showAt={TIER_STATIC_PARAMS} tier={toolbarTier}>
                {/* 【为什么是 asChild】`DropdownMenu.Trigger` 默认渲染自己的 `<button>`，把
                    `AppIconButton`（也是 `<button>`）放进去就成了 `<button><button/></button>` ——
                    无效 HTML，而且实测点击与键盘都打不开菜单（外层按钮吃掉了事件）。
                    `asChild` 让 Radix 把 props 合并到 `AppIconButton` 的那一个按钮上。

                    【为什么要 cast】Radix Themes 的 `DropdownMenu.Trigger` 类型没有声明
                    `asChild`，而它在运行时就是 `DropdownMenuPrimitive.Trigger`（支持
                    asChild，props 原样透传）。这里只补一个类型，不改行为。 */}
                <TriggerAsChild>
                    <AppIconButton
                        tooltip={tf("param_toolbar_more")}
                        aria-label={tf("param_toolbar_more")}
                        tabIndex={-1}
                        icon={<MixerHorizontalIcon />}
                    />
                </TriggerAsChild>
            </ToolbarSlot>
            <DropdownMenu.Content variant="soft" color="gray" side="bottom" align="end">
                <Flex direction="column" gap="2" p="1">
                    {pillsCollapsed ? (
                        <Flex direction="column" gap="2" align="start">
                            <ReferenceTrackGroup m={menuModel} />
                            <ChildPitchGroup m={menuModel} />
                            <Flex gap="1" wrap="wrap" align="center">
                                <ParamButtons m={menuModel} />
                            </Flex>
                        </Flex>
                    ) : null}
                    <ParamStaticControls m={menuModel} />
                </Flex>
            </DropdownMenu.Content>
        </DropdownMenu.Root>
    );
}
