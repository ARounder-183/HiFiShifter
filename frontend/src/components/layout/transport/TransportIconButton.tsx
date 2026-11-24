/**
 * 传输栏图标按钮 —— stop / play / record / undo / redo / capture 的**唯一**原语。
 *
 * 【为什么需要它】这一行此前混用了三套盒模型：
 *   * Stop 是 Radix `Button`（文字按钮：横向 padding + 40px 宽）；
 *   * Play 是 `IconButton variant="solid"`（方形图标按钮：28px 宽）；
 *   * Record 是**被 Box 包裹的** `IconButton`，还在 soft / solid 之间切换。
 * 用户看到的"录音键的红色填充与播放键不齐"就是这三套盒模型的直接后果 ——
 * 它不是一个颜色值的问题，只改 `--red-9` 永远修不掉。
 *
 * 【为什么 `fill` 与 `active` 是两个维度】`fill` 是**常态**该长什么样（录音键待机是
 * soft 红），`active` 是**这一刻**是否处于开态（录制中是 solid 红）。此前把录音键
 * 固定成 solid 是为了治"红色填充与播放键不齐"—— 那是误诊：不齐来自三个按钮各用
 * 一套盒模型（文字 Button / 图标 IconButton / Box 包裹的 IconButton），与 soft↔solid
 * 的填充色无关。盒模型归一化落在 `index.css` 的 `[data-hs-transport-button]` 规则里，
 * 覆盖全部变体，因此变体可以自由切换而盒子不动。
 *
 * 这两件事分开之后，"待机 soft / 录制中 solid"就是安全的：它只改填充色，
 * 宽度、高度、内边距、负 margin 都仍由那条 CSS 规则钉死。
 *
 * 【为什么用 `data-hs-transport-button` 而不是给每个实例写 className】
 * `index.css` 里有一条针对该属性的盒模型规则，覆盖 ghost / soft / solid **全部**
 * 变体。只归一化 ghost（此前只有这一条）会把"ghost ↔ solid 切换不抖"修好，却留下
 * "soft vs solid 的填充边界不一致"—— 那正是用户看到的那 1px。
 */
import type { ReactNode, Ref } from "react";

import { AppIconButton } from "../../../ui/Button";

export type TransportTone = "neutral" | "accent" | "danger";
export type TransportFill = "ghost" | "soft" | "solid";

export interface TransportIconButtonProps {
    icon: ReactNode;
    /** 同时作为 `aria-label`（图标按钮没有可读文本）。 */
    tooltip: string;
    onClick?: (event: React.MouseEvent<HTMLButtonElement>) => void;
    onContextMenu?: (event: React.MouseEvent<HTMLButtonElement>) => void;
    disabled?: boolean;
    /** 切换态的"开"：强制实心并置 `aria-pressed`，不改变盒模型。 */
    active?: boolean;
    /** 语义色：`danger` 是录音这类破坏性/捕获动作的固定红。 */
    tone?: TransportTone;
    /** 填充家族。`active` 为真时强制实心。 */
    fill?: TransportFill;
    /** 键盘 Tab 顺序覆盖（传输栏整组默认不进 Tab 序）。 */
    tabIndex?: number;
    ref?: Ref<HTMLButtonElement>;
    "data-testid"?: string;
}

export function TransportIconButton({
    icon,
    tooltip,
    onClick,
    onContextMenu,
    disabled,
    active = false,
    tone = "neutral",
    fill = "ghost",
    tabIndex,
    ref,
    "data-testid": testId,
}: TransportIconButtonProps) {
    return (
        <AppIconButton
            ref={ref}
            icon={icon}
            tooltip={tooltip}
            onClick={onClick}
            onContextMenu={onContextMenu}
            disabled={disabled}
            active={active}
            // tone → AppIconButton 的语义维度：danger 是固定红，accent 走主题强调色。
            intent={tone === "danger" ? "danger" : "default"}
            emphasis={tone === "accent" ? "accent" : "neutral"}
            surface={fill}
            tabIndex={tabIndex}
            data-hs-transport-button=""
            data-hs-toolbar-control=""
            data-testid={testId}
        />
    );
}
