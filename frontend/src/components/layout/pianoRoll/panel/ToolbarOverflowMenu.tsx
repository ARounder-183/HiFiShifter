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
 */
import { DropdownMenu, Flex } from "@radix-ui/themes";
import { MixerHorizontalIcon } from "@radix-ui/react-icons";

import { AppIconButton } from "../../../../ui";
import { ChildPitchGroup } from "./ChildPitchGroup";
import { ParamButtons } from "./ParamButtons";
import { ParamStaticControls } from "./ParamStaticControls";
import { ReferenceTrackGroup } from "./ReferenceTrackGroup";
import { TIER_ALL_PILLS, TIER_STATIC_PARAMS } from "../toolbarOverflow";
import type { ParamToolbarModel } from "./ParamToolbar";

export function ToolbarOverflowMenu({ m }: { m: ParamToolbarModel }) {
    const { tf, toolbarTier } = m;
    // 前 9 级只砍冗余，右簇全在行内 —— 没有东西可收，也就不该多出一个按钮。
    if (toolbarTier < TIER_STATIC_PARAMS) return null;
    const pillsCollapsed = toolbarTier >= TIER_ALL_PILLS;

    return (
        <DropdownMenu.Root>
            <DropdownMenu.Trigger>
                <AppIconButton
                    tooltip={tf("param_toolbar_more")}
                    aria-label={tf("param_toolbar_more")}
                    tabIndex={-1}
                    icon={<MixerHorizontalIcon />}
                />
            </DropdownMenu.Trigger>
            <DropdownMenu.Content variant="soft" color="gray" side="bottom" align="end">
                <Flex direction="column" gap="2" p="1">
                    {pillsCollapsed ? (
                        <Flex direction="column" gap="2" align="start">
                            <ReferenceTrackGroup m={m} />
                            <ChildPitchGroup m={m} />
                            <Flex gap="1" wrap="wrap" align="center">
                                <ParamButtons m={m} />
                            </Flex>
                        </Flex>
                    ) : null}
                    <ParamStaticControls m={m} />
                </Flex>
            </DropdownMenu.Content>
        </DropdownMenu.Root>
    );
}
