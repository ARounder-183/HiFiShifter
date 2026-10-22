/**
 * 参数编辑器工具栏的展示型零件。
 *
 * 【为什么单独成文件】这几个组件只依赖 props —— 不看 store、不碰 ref、不测量 DOM。
 * 它们曾经和近万行的面板挤在同一个文件里，"工具栏上的一个药丸长什么样"要翻到
 * 面板文件的第 600 多行才找得到。
 *
 * 【`ToolbarSlot` 为什么不是条件渲染】层级判据要靠"量出每个层级的内容宽度"
 * （见 `toolbarGauge`）：被让位的项若直接从 DOM 消失，就再也量不到"少藏一级要多宽"，
 * 于是只能靠"先撑开、再一级级收回去"试探，用户看到的正是那个抽动。常挂载 +
 * 由层级控制 `display` 之后，测量可以在**看不见的克隆**上一次做完。
 */
import React from "react";
import { DropdownMenu } from "@radix-ui/themes";
import { ChevronDownIcon, EyeOpenIcon, EyeClosedIcon } from "@radix-ui/react-icons";

/**
 * “气声/气流”图标：三道向右上方倾斜流动的曲线，表示风/气流（类似 Material “Air” 图标），
 * 避免被误认为汉堡菜单；关闭（off）时气流变淡并叠加一条斜杠。
 * 用于代替冗长的“气声开启/气声关闭”文本。
 */
export const BreathAirIcon: React.FC<{ off?: boolean }> = ({ off = false }) => (
    <svg
        width="14"
        height="14"
        viewBox="0 0 14 14"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        style={{ display: "block" }}
    >
        {/* 三道右倾的流动曲线 = 风/气流 */}
        <path
            d="M2.2 3.6C4.8 1.9 8.2 2.5 11 5.2"
            stroke="currentColor"
            strokeWidth="1.3"
            strokeLinecap="round"
            opacity={off ? 0.4 : 1}
        />
        <path
            d="M1.4 6.8C4.6 5.1 8.1 5.9 11.4 8.9"
            stroke="currentColor"
            strokeWidth="1.3"
            strokeLinecap="round"
            opacity={off ? 0.4 : 1}
        />
        <path
            d="M1.9 10.1C4.7 9.3 7.5 9.5 9.9 11.7"
            stroke="currentColor"
            strokeWidth="1.3"
            strokeLinecap="round"
            opacity={off ? 0.4 : 1}
        />
        {off ? (
            <path d="M3 11L11 3" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
        ) : null}
    </svg>
);

/**
 * 工具栏里的一个**可让位槽位**。
 *
 * 【为什么它不是条件渲染】层级判据要靠"量出每个层级的内容宽度"（见
 * `toolbarGauge`）。若被让位的项直接从 DOM 里消失，就再也量不到"少藏一级要多宽"——
 * 于是只能靠"先撑开、再一级级收回去"来试探，用户看到的正是那个抽动。
 * 常挂载 + 由层级控制 `display` 之后，测量可以在**看不见的克隆**上一次做完。
 *
 * 【为什么是 `display: contents`】包装元素自己不参与 flex 布局，子元素直接成为祖父
 * 的 flex 项 —— 加了一层包装也不会改变原来的间距与对齐。
 */
export const ToolbarSlot: React.FC<{
    /** 达到这个层级时隐藏本槽位。 */
    hideAt: number;
    /** 当前层级。 */
    tier: number;
    children: React.ReactNode;
}> = ({ hideAt, tier, children }) => (
    <span data-hide-at={hideAt} style={{ display: tier >= hideAt ? "none" : "contents" }}>
        {children}
    </span>
);

export type ParamToolbarPillProps = {
    /** 参数按钮上的简短标签（如 PIT / 共振峰） */
    label: string;
    /** 完整参数名（ToolTip） */
    labelTooltip?: string;
    /** 是否为主参数（激活）：整个药丸整体高亮（统一用全局强调色） */
    active: boolean;
    /** 点击标签：选中该参数 */
    onSelect: () => void;
    /** 眼睛状态：main=主参数（仅展示“睁开”，不响应点击）；on/off=副曲线叠加可见/隐藏 */
    eyeMode: "main" | "on" | "off";
    /** 点击眼睛（非 main 时调用；main 时不响应，以保持排版稳定） */
    onToggleEye?: () => void;
    /** 眼睛 ToolTip（两行：状态 + 点击动作） */
    eyeTooltip?: string;
    /** 眼睛的无障碍标签（简短，如“显示/隐藏副参数叠加曲线”） */
    eyeLabel?: string;
    /** 可选片段：子参数下拉菜单的触发按钮（已含 param-pill__seg 样式类） */
    dropdown?: React.ReactNode;
    /**
     * 参数是否不可用（置灰、不可选中）。
     *
     * 当前用途：气声分离开关闭时，`breath_gain` 与 `hifigan_tension` 都依赖分离，
     * 因此不可编辑、也不参与合成。注意**只禁用标签的选中**，眼睛（曲线显隐）
     * 仍然可用 —— 曲线需要保持可见，见下方 `disabled` 分支的说明。
     */
    disabled?: boolean;
    /** 禁用原因（ToolTip）。未提供时沿用 `labelTooltip`。 */
    disabledTooltip?: string;
};

/**
 * 参数编辑器工具栏的“参数分组药丸”：眼睛 → 参数名 → 子参数下拉。
 * 各片段共享一块连续背景；激活参数统一铺全局强调色（--accent-9），
 * 片段间用细分隔线区分；悬停时只高亮当前片段，提示其独立可点击。
 *
 * `disabled` 只作用于**标签片段**（选中该参数），不影响眼睛片段：
 * 被门禁的参数曲线仍需保持可见（用户要能看到自己画过什么），
 * 而"能否编辑"由父组件的 `editParam` 回退负责。
 */
export const ParamToolbarPill: React.FC<ParamToolbarPillProps> = ({
    label,
    labelTooltip,
    active,
    onSelect,
    eyeMode,
    onToggleEye,
    eyeTooltip,
    eyeLabel,
    dropdown,
    disabled = false,
    disabledTooltip,
}) => {
    const eyeInert = eyeMode === "main";
    const eyeIcon = eyeMode === "off" ? <EyeClosedIcon /> : <EyeOpenIcon />;
    return (
        <div
            className="param-pill"
            data-active={active ? "true" : undefined}
            data-eye-off={eyeMode === "off" ? "true" : undefined}
        >
            <button
                type="button"
                tabIndex={-1}
                className={
                    eyeInert
                        ? "param-pill__seg param-pill__seg--eye param-pill__seg--inert"
                        : "param-pill__seg param-pill__seg--eye"
                }
                data-tooltip={eyeInert ? undefined : eyeTooltip}
                aria-label={eyeInert ? label : (eyeLabel ?? eyeTooltip)}
                onClick={(e) => {
                    e.stopPropagation();
                    if (!eyeInert) onToggleEye?.();
                }}
            >
                {eyeIcon}
            </button>
            <button
                type="button"
                tabIndex={disabled ? -1 : undefined}
                className={
                    disabled
                        ? "param-pill__seg param-pill__seg--label param-pill__seg--disabled"
                        : "param-pill__seg param-pill__seg--label"
                }
                data-tooltip={disabled ? (disabledTooltip ?? labelTooltip) : labelTooltip}
                aria-disabled={disabled || undefined}
                onClick={disabled ? undefined : onSelect}
            >
                {label}
            </button>
            {dropdown}
        </div>
    );
};

/**
 * 「一个药丸 + 下拉切换两个参数」的通用按钮。
 *
 * 两个使用场景：
 * - **共振峰**：根参数（整个轨道组）与子参数（当前子轨道的偏移）。只有在
 *   选中子轨道时才有第二个选项，否则退化为普通药丸。
 * - **音量 / 动态**：两个同量纲的混音级参数。两者**始终**可选，因此
 *   `alwaysShowDropdown = true`。
 *
 * 下拉的样式与交互（RadioGroup + ChevronDown 段）与「音高」参数组完全一致，
 * 保持工具栏里三组参数的观感统一。
 */
export type ParamGroupButtonProps = {
    rootParamId: string;
    /** 按钮上的简短标签（如 FRM / 共振峰） */
    rootLabel: string;
    /** 下拉菜单中“根参数”选项的详细说明（如 Formant Shift (Track Group)） */
    rootMenuLabel?: string;
    rootTooltip?: string;
    childParamId: string | null;
    /** 按钮上的简短子参数标签（如 共振峰差） */
    childLabel: string;
    /** 下拉菜单中“子参数”选项的详细说明（如 Formant Offset (Current Sub-track)） */
    childMenuLabel?: string;
    rootActive: boolean;
    childActive: boolean;
    secondaryVisible: boolean;
    /** 眼睛的无障碍标签（简短动作说明） */
    hideSecondaryLabel: string;
    showSecondaryLabel: string;
    /** 眼睛 ToolTip（两行：状态 + 点击动作） */
    hideSecondaryTooltip: string;
    showSecondaryTooltip: string;
    onSelectRoot: () => void;
    onSelectChild: () => void;
    onToggleSecondary: () => void;
    /**
     * `true` = 即使没有子参数也显示下拉（音量/动态这类"恒有两个选项"的组）。
     * 缺省 `false`（共振峰语义：只有子轨道才有第二个选项）。
     */
    alwaysShowDropdown?: boolean;
};

export const ParamGroupButton: React.FC<ParamGroupButtonProps> = ({
    rootParamId,
    rootLabel,
    rootMenuLabel,
    rootTooltip,
    childParamId,
    childLabel,
    childMenuLabel,
    rootActive,
    childActive,
    secondaryVisible,
    hideSecondaryLabel,
    showSecondaryLabel,
    hideSecondaryTooltip,
    showSecondaryTooltip,
    onSelectRoot,
    onSelectChild,
    onToggleSecondary,
    alwaysShowDropdown = false,
}) => {
    const eyeMode: "main" | "on" | "off" =
        rootActive || childActive ? "main" : secondaryVisible ? "on" : "off";

    if (!childParamId && !alwaysShowDropdown) {
        return (
            <ParamToolbarPill
                label={rootLabel}
                labelTooltip={rootTooltip}
                active={rootActive}
                onSelect={onSelectRoot}
                eyeMode={eyeMode}
                onToggleEye={onToggleSecondary}
                eyeTooltip={secondaryVisible ? showSecondaryTooltip : hideSecondaryTooltip}
                eyeLabel={secondaryVisible ? showSecondaryLabel : hideSecondaryLabel}
            />
        );
    }

    return (
        <DropdownMenu.Root>
            <ParamToolbarPill
                label={childActive ? childLabel : rootLabel}
                labelTooltip={childActive ? (childMenuLabel ?? childLabel) : rootTooltip}
                active={rootActive || childActive}
                onSelect={onSelectRoot}
                eyeMode={eyeMode}
                onToggleEye={onToggleSecondary}
                eyeTooltip={secondaryVisible ? showSecondaryTooltip : hideSecondaryTooltip}
                eyeLabel={secondaryVisible ? showSecondaryLabel : hideSecondaryLabel}
                dropdown={
                    <DropdownMenu.Trigger
                        className="param-pill__seg param-pill__seg--chev"
                        data-tooltip={childActive ? (childMenuLabel ?? childLabel) : rootTooltip}
                        tabIndex={-1}
                    >
                        <ChevronDownIcon width="12" height="12" />
                    </DropdownMenu.Trigger>
                }
            />
            <DropdownMenu.Content variant="soft" color="gray">
                <DropdownMenu.RadioGroup
                    value={
                        rootActive
                            ? rootParamId
                            : childActive && childParamId
                              ? childParamId
                              : undefined
                    }
                    onValueChange={(value) => {
                        if (value === rootParamId) {
                            onSelectRoot();
                        } else if (value === childParamId) {
                            onSelectChild();
                        }
                    }}
                >
                    <DropdownMenu.RadioItem value={rootParamId}>
                        {rootMenuLabel ?? rootLabel}
                    </DropdownMenu.RadioItem>
                    {childParamId ? (
                        <DropdownMenu.RadioItem value={childParamId}>
                            {childMenuLabel ?? childLabel}
                        </DropdownMenu.RadioItem>
                    ) : null}
                </DropdownMenu.RadioGroup>
            </DropdownMenu.Content>
        </DropdownMenu.Root>
    );
};
