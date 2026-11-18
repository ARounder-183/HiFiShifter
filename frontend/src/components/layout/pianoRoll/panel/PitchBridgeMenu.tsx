/**
 * 参数编辑器工具栏上的「REAPER Take 音高」split button 与作用域菜单。
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
 * 【为什么是平铺 + 分组标题而不是子菜单】仓库约定（见 AppMenuItemSpec.items）：
 * 右键/点开就想直接点的那几项不要塞进子菜单，多一次悬停就是多一次成本。这里 6 项
 * 同质、可枚举，平铺 + 两个标题行读起来最快。
 *
 * 【为什么宿主不支持时**整项不渲染**】canBridgeHostTakePitch() 是能力位（宿主是
 * REAPER、后端查到了包络 API）。独立 App 里这两个方向没有意义，渲染一个永远置灰的
 * 按钮只会让"这功能坏了"变成第一印象 —— 与片段右键菜单同一条策略。
 *
 * 【交互必须与相邻按钮同构】这一排的每一枚图标按钮都是同一条协议：**左键 = 立刻做
 * 那件事，右键 = 在那枚按钮下方展开菜单**，右下角小三角由 AppIconButton 的
 * menuCaret 画（"右键还有东西"）。此前这个入口是"左键开菜单"、却带着同一枚小三角 ——
 * 读者按邻居的规矩去点，得到的是另一个结果：菜单。同一个记号在同一排里不能有两种
 * 含义，所以这里改成左键执行、右键挑选。
 *
 * 【为什么左键给"导出"而不是"导入"】导出只是把插件里这条线誊到宿主：**源不动**，
 * 再执行一次还是同一个结果，误触的代价是一步宿主撤销。导入相反 —— 它用 take 音高
 * 盖掉用户正在编辑的这条线，而那正是本插件里唯一的成品；误触一次，成品就换了内容。
 * 可重复的那一侧留给左键，不可逆的那一侧留给菜单（用户在菜单里看清作用域与计数
 * 之后才点）。
 *
 * 【为什么左键固定"当前轨道"而不记住上次的作用域】记住上次会把"点一下"变成
 * "点一下可能改写整个工程" —— 同一枚按钮的含义随上一次选择而变，肌肉记忆就此不可靠。
 * 批量作用域只从菜单进，那里每项都写着计数。
 *
 * 【菜单最后一项是开关，不是第七个作用域】「自动写入 Take 音高」打开之后，用户在参数
 * 编辑器里每改一笔音高，插件都会在停手约 0.4 s 后自己把包络写进宿主 —— 于是这个菜单
 * 从"我现在转一遍"变成"以后都别问我"。它单独一行、排在分隔线之后：上面六项都会**立刻**
 * 改写东西，这一项只改一个开关，混在同一组里会让一次右键出现七件看起来一样的事，
 * 而其中一件点了没有任何可见变化。
 */
import { useCallback, useRef, useState } from "react";
import { Box } from "@radix-ui/themes";

import {
    persistUiSettings,
    toggleReaperAutoTakePitchWrite,
} from "../../../../features/session/sessionSlice";
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
    /** `s` / `dispatch` 只为底部那条"自动写入"开关服务（见模块头最后一段）。 */
    m: Pick<PianoRollPanelModel, "dispatch" | "pitchBridgeScopes" | "s">;
};

/** 作用域的显示顺序：由窄到宽（与"我改动了多少东西"的直觉一致）。 */
const SCOPE_ORDER: readonly PitchBridgeScope[] = ["track", "selection", "all"];

/** 左键执行的那一项：方向与作用域都是**固定**的（见模块头最后两段）。 */
const PRIMARY_DIRECTION: PitchBridgeDirection = "export";
const PRIMARY_SCOPE: PitchBridgeScope = "track";

const SCOPE_LABEL_KEY: Record<PitchBridgeScope, MessageKey> = {
    track: "pitch_bridge_scope_track",
    selection: "pitch_bridge_scope_selection",
    all: "pitch_bridge_scope_all",
};

/**
 * 「互转」图标：两条反向的水平箭头。
 *
 * 【为什么画这个而不是用现成图标】Radix 的箭头类图标（DoubleArrowRight 等）读作
 * "快进 / 下一步"；这一项是**两个方向互为镜像**，只有反向双箭头能一眼读出来。
 * 尺寸 / 线宽与工具栏里手绘的几枚（音阶高亮、锁定）保持一致。
 */
function BridgeIcon() {
    return (
        <svg
            width="15"
            height="15"
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
    const { dispatch, s } = m;
    const [open, setOpen] = useState(false);
    const anchorRef = useRef<HTMLDivElement>(null);
    const menuRef = useRef<HTMLDivElement>(null);

    /** 按作用域取片段 id 并执行；空作用域不发命令（与置灰的菜单项同一条判据）。 */
    const runScope = useCallback(
        (direction: PitchBridgeDirection, scope: PitchBridgeScope) => {
            const ids = scopeClipIds(m.pitchBridgeScopes, scope);
            setOpen(false);
            if (ids.length === 0) return;
            void run(direction, ids);
        },
        [m.pitchBridgeScopes, run],
    );

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
                    onSelect: () => runScope(direction, scope),
                };
            }),
        [busy, m.pitchBridgeScopes, runScope, t, tVars],
    );

    // 宿主不支持时整项不出现（不是置灰）：见模块头。
    if (!available) return null;

    const primaryIds = scopeClipIds(m.pitchBridgeScopes, PRIMARY_SCOPE);

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
        /**
         * 自动写入：**开关**，不是又一次"现在转一遍"。
         *
         * 【为什么放在这里而不是"选项"菜单】它是这个功能的一部分，而用户想到它的时刻
         * 正是"我不想每次都点这一下"—— 也就是站在这个菜单前面的时候。
         *
         * 【为什么单独一行、分隔线之后】上面六项都会**立刻**改写东西；这一项只是改一个
         * 开关。混在同一组里，按一次右键就会看到七件"看起来一样"的事，其中一件点了不会
         * 发生任何可见变化。
         */
        {
            key: "auto-write",
            label: t("pitch_bridge_auto_write"),
            separatorBefore: true,
            checked: s.reaperAutoTakePitchWrite,
            onSelect: () => {
                dispatch(toggleReaperAutoTakePitchWrite());
                void dispatch(persistUiSettings());
            },
        },
    ];

    return (
        <Box ref={anchorRef} data-hs-context-menu>
            <AppIconButton
                // 与相邻开关同款：强调色由原语回落（这枚按钮不表达"开着"的状态，
                // 但同一排的图标按钮都用它，视觉上才是同族）。
                emphasis="accent"
                // 固定「功能名：动作名」—— 它同时是 aria-label，所以必须自带功能名。
                tooltip={t("pitch_bridge_tooltip_export_track")}
                tabIndex={-1}
                // 右下角小三角：提示"右键还有作用域可选"（与工具栏其它按钮同一个记号）。
                menuCaret
                disabled={busy || primaryIds.length === 0}
                onClick={() => runScope(PRIMARY_DIRECTION, PRIMARY_SCOPE)}
                onContextMenu={(event) => {
                    event.preventDefault();
                    setOpen(true);
                }}
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
