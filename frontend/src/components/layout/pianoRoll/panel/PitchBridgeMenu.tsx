/**
 * 参数编辑器工具栏上的「REAPER 音高」菜单：Take 音高 ⇄ 音高线，按作用域执行。
 *
 * 【为什么它长在参数编辑器里】插件里用户真正待着的地方是参数编辑器，不是时间轴 ——
 * 时间轴的大部分职能由宿主自己承担，而这条音高线只有本插件有。此前的入口只有时间轴
 * 片段的右键菜单：那意味着"想把这根轨道的音高推给 REAPER"要先回宿主里选中片段，
 * 而用户此刻正在参数编辑器里调它。
 *
 * 【为什么作用域是三个而不是"当前片段"】参数编辑器里没有"当前片段"这个概念 ——
 * 有的是**正在编辑的轨道**、**画出来的选区**、以及整个工程。三个作用域都从这三个
 * 既有事实直接落下来，用户不需要先建立一种新的选择状态。
 *
 * 【为什么每一项都带计数】这三个入口里有两个会一次改写很多条片段（"全部片段"在真实
 * 工程里轻松过千）。菜单里写清"这一下会动多少条"，是唯一在**点击之前**就能给出的
 * 防呆；事后报告只能告诉他改错了多少。
 *
 * 【为什么是平铺 + 分组标题而不是子菜单】仓库约定（见 `AppMenuItemSpec.items`）：
 * 右键/点开就想直接点的那几项不要塞进子菜单，多一次悬停就是多一次成本。这里 6 项
 * 同质、可枚举，平铺 + 两个标题行读起来最快。
 *
 * 【为什么宿主不支持时**整项不渲染**】`canBridgeHostTakePitch()` 是能力位（宿主是
 * REAPER、后端查到了包络 API）。独立 App 里这两个方向没有意义，渲染一个永远置灰的
 * 按钮只会让"这功能坏了"变成第一印象 —— 与片段右键菜单同一条策略。
 */
import { useCallback, useRef, useState } from "react";
import { Box } from "@radix-ui/themes";

import {
    useHostPitchBridge,
    type PitchBridgeDirection,
} from "../../../../features/hostPitchBridge/useHostPitchBridge";
import { scopeClipIds, type PitchBridgeScope } from "../../../../features/hostPitchBridge/scope";
import { useI18n } from "../../../../i18n/I18nProvider";
import type { MessageKey } from "../../../../i18n/messages";
import type { AppMenuItemSpec } from "../../../../ui";
import { AppAnchoredMenu, AppIconButton } from "../../../../ui";
import type { PianoRollPanelModel } from "./usePianoRollPanelModel";

type Props = {
    m: Pick<PianoRollPanelModel, "pitchBridgeScopes">;
};

/** 作用域的显示顺序：由窄到宽（与"我改动了多少东西"的直觉一致）。 */
const SCOPE_ORDER: readonly PitchBridgeScope[] = ["track", "selection", "all"];

const SCOPE_LABEL_KEY: Record<PitchBridgeScope, MessageKey> = {
    track: "pitch_bridge_scope_track",
    selection: "pitch_bridge_scope_selection",
    all: "pitch_bridge_scope_all",
};

/**
 * 「互转」图标：两条反向的水平箭头。
 *
 * 【为什么画这个而不是用现成图标】Radix 的箭头类图标（`DoubleArrowRight` 等）读作
 * "快进 / 下一步"；这一项是**两个方向互为镜像**，只有反向双箭头能一眼读出来。
 * 尺寸 / 线宽与工具栏里手绘的几枚（音阶高亮、锁定）保持一致。
 */
function BridgeIcon() {
    return (
        <svg
            width="14"
            height="14"
            viewBox="0 0 14 14"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
        >
            <path
                d="M2 5.1h8.1M8.3 3.3l2 1.8-2 1.8"
                stroke="currentColor"
                strokeWidth="1.1"
                strokeLinecap="round"
                strokeLinejoin="round"
            />
            <path
                d="M12 8.9H3.9M5.7 7.1l-2 1.8 2 1.8"
                stroke="currentColor"
                strokeWidth="1.1"
                strokeLinecap="round"
                strokeLinejoin="round"
            />
        </svg>
    );
}

export function PitchBridgeMenu({ m }: Props) {
    const { t, tVars } = useI18n();
    const { available, busy, run } = useHostPitchBridge();
    const [open, setOpen] = useState(false);
    const anchorRef = useRef<HTMLDivElement>(null);
    const menuRef = useRef<HTMLDivElement>(null);

    const scopesFor = useCallback(
        (direction: PitchBridgeDirection): AppMenuItemSpec[] =>
            SCOPE_ORDER.filter(
                // 没有选区时"选区内"恒为空：摆一个 (0) 只会让人以为选区丢了。
                (scope) => scope !== "selection" || m.pitchBridgeScopes.selection.length > 0,
            ).map((scope) => {
                const ids = scopeClipIds(m.pitchBridgeScopes, scope);
                return {
                    key: `${direction}-${scope}`,
                    label: tVars("pitch_bridge_scope_item", {
                        scope: t(SCOPE_LABEL_KEY[scope]),
                        count: ids.length,
                    }),
                    disabled: busy || ids.length === 0,
                    onSelect: () => {
                        setOpen(false);
                        void run(direction, ids);
                    },
                };
            }),
        [busy, m.pitchBridgeScopes, run, t, tVars],
    );

    // 宿主不支持时整项不出现（不是置灰）：见模块头。
    if (!available) return null;

    const items: AppMenuItemSpec[] = [
        {
            key: "import-heading",
            heading: true,
            label: t("pitch_bridge_import_heading"),
        },
        ...scopesFor("import"),
        {
            key: "export-heading",
            heading: true,
            label: t("pitch_bridge_export_heading"),
        },
        ...scopesFor("export"),
        {
            // 这一行是**只读说明**（heading 不参与键盘导航、点不动）。"转换到底是复制
            // 还是搬走"是这个功能最容易被误解的一件事，而菜单是它唯一的现场说明位。
            key: "copy-note",
            heading: true,
            label: t("pitch_bridge_copy_note"),
        },
    ];

    return (
        <Box ref={anchorRef} data-hs-context-menu>
            <AppIconButton
                tooltip={t("pitch_bridge_menu_tooltip")}
                aria-label={t("pitch_bridge_menu_tooltip")}
                tabIndex={-1}
                menuCaret
                onClick={() => setOpen((value) => !value)}
                icon={<BridgeIcon />}
            />
            <AppAnchoredMenu
                open={open}
                anchorRef={anchorRef}
                menuRef={menuRef}
                ariaLabel={t("pitch_bridge_menu_tooltip")}
                onClose={() => setOpen(false)}
                items={items}
            />
        </Box>
    );
}
