/*
 * 「点选即执行」的选项列表。
 *
 * 【它解决的问题】三处手写了同一个形态：导入文件的模式选择、多音轨媒体的音轨
 * 选择、外观设置的主题选择。三处都是"一列可点的卡片、点下去直接执行"，也都各自
 * 写了一遍同样的 class（`w-full text-left px-3 py-2 rounded-lg border
 * hover:bg-qt-hover`），以及各自漏掉同样的键盘契约 —— 前两处连 Esc 都没有，
 * 更不用说方向键。
 *
 * 【与 `AppSelect` 的分工】这是本原语存在的前提，不是重复建设：
 *   - `AppSelect`：**先选后确认**。值先落到控件上，由对话框页脚的按钮提交。
 *     适合表单字段（"导出格式"）。
 *   - `AppChoiceList`：**点选即执行**。不落值，选中立刻触发回调。
 *     适合"你要怎么做"这类一次性的分岔（"按时间铺开 / 按轨道铺开 / 作为 Take"）。
 *
 * 把"点选即执行"塞进 `AppSelect` 会让用户以为还要再按一次确定；反过来把表单字段
 * 做成 `AppChoiceList` 会让用户以为选了就生效。两者不可互相替代。
 *
 * 【键盘】复用 `useMenuKeyboard`：方向键循环、Home/End 到两端、Enter/Space 由真实
 * `<button>` 原生处理、焦点在输入框时不抢键。`role="menu"` + `role="menuitem"` 是
 * 诚实的语义 —— 每一项都是一个**动作**，而且选完不会留下"选中态"（`listbox` 的
 * `aria-selected` 在这里会撒谎）。这个角色也让全局快捷键分发器自动让出方向键
 * （见 `useKeybindings` 的复合控件让路规则）。
 */
import { useRef } from "react";
import type { ReactNode } from "react";

import { cx } from "./cx";
import { useMenuKeyboard } from "./useMenuKeyboard";

export interface AppChoiceOption {
    id: string;
    label: ReactNode;
    /**
     * 副行：说明或元信息（编解码器 / 声道 / 采样率 / 时长…）。
     * 渲染为 `.hs-type-label`（12px 弱化色），比标签低一档。
     */
    description?: ReactNode;
    disabled?: boolean;
}

export interface AppChoiceListProps {
    options: AppChoiceOption[];
    /** 点选即执行。 */
    onSelect: (id: string) => void;
    /** 列表的可访问名称。省略时由外层对话框的标题承担语境。 */
    ariaLabel?: string;
    className?: string;
}

export function AppChoiceList({ options, onSelect, ariaLabel, className }: AppChoiceListProps) {
    const listRef = useRef<HTMLDivElement | null>(null);
    useMenuKeyboard(listRef);

    /*
     * roving tabindex：整列只占**一个** Tab 停留点（第一个可用项），进入之后用
     * 方向键移动。与 `DockTabBar` / 菜单同一套约定 —— 若让每个选项都进 Tab 序，
     * 一个 8 项的列表就会把对话框的 Tab 路径拉长 8 倍。
     *
     * 不用 `useMemo`：列表项数是个位数，`findIndex` 的代价远小于一次依赖比较；
     * 而 memo 会引入"依赖写漏 → 停留点停在旧索引"的隐性错误（选项数组通常是
     * 渲染期新建的，依赖它等于每帧重算，memo 本来也没省下什么）。
     */
    const firstEnabledIndex = options.findIndex((option) => !option.disabled);

    return (
        <div
            ref={listRef}
            role="menu"
            aria-label={ariaLabel}
            className={cx("flex flex-col gap-2", className)}
        >
            {options.map((option, index) => (
                <button
                    key={option.id}
                    type="button"
                    role="menuitem"
                    disabled={option.disabled}
                    tabIndex={index === firstEnabledIndex ? 0 : -1}
                    onClick={() => {
                        if (option.disabled) return;
                        onSelect(option.id);
                    }}
                    className={cx(
                        "w-full rounded-qt-md border border-qt-border bg-qt-surface px-3 py-2 text-left",
                        "text-qt-md text-qt-text transition-colors",
                        option.disabled
                            ? "cursor-default opacity-50"
                            : "cursor-pointer hover:bg-qt-hover",
                    )}
                >
                    <span className="block">{option.label}</span>
                    {option.description ? (
                        <span className="hs-type-label mt-0.5 block text-qt-text-muted">
                            {option.description}
                        </span>
                    ) : null}
                </button>
            ))}
        </div>
    );
}
