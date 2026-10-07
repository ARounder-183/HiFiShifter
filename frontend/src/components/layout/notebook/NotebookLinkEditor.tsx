/*
 * 链接地址编辑浮层。
 *
 * 【为什么不再挂在工具栏里】工具栏是可隐藏的（`showToolbar` 设置），而
 * Ctrl/⌘+K 这个快捷键不该因为工具栏被收起就失效。浮层改由面板渲染、定位在
 * 编辑区顶部 —— 工具栏可见时它正好落在工具栏下方（视觉与从前一致），
 * 收起时则落在正文上沿。
 *
 * 【为什么不是 window.prompt】Tauri / WKWebView 下脚本对话框会静默返回 null
 * （见 `ClipContextMenu` 同款理由），必须用应用内的输入框。
 */

import { useI18n } from "../../../i18n/I18nProvider";

export interface NotebookLinkEditorProps {
    value: string;
    onChange: (next: string) => void;
    /** 确认（Enter / ✓）。 */
    onApply: () => void;
    /** 取消（Escape / ✕）。 */
    onCancel: () => void;
}

export function NotebookLinkEditor({
    value,
    onChange,
    onApply,
    onCancel,
}: NotebookLinkEditorProps) {
    const { t } = useI18n();
    return (
        <div className="hs-notebook-link-popover">
            {/* 键盘可用：Enter 应用、Escape 取消；stopPropagation 挡掉编辑器
                快捷键（输入框内不应触发 Ctrl+B 之类）。 */}
            <input
                autoFocus
                value={value}
                placeholder={t("notebook_link_prompt")}
                aria-label={t("notebook_link_prompt")}
                onChange={(event) => onChange(event.target.value)}
                onKeyDown={(event) => {
                    event.stopPropagation();
                    if (event.key === "Enter") {
                        event.preventDefault();
                        onApply();
                    } else if (event.key === "Escape") {
                        onCancel();
                    }
                }}
            />
            <button
                type="button"
                className="hs-notebook-toolbar-btn"
                data-tooltip={t("ok")}
                aria-label={t("ok")}
                // 按下时不抢焦点：编辑器保持选区，命令才能作用于正确的范围。
                onPointerDown={(event) => event.preventDefault()}
                onClick={onApply}
            >
                ✓
            </button>
            <button
                type="button"
                className="hs-notebook-toolbar-btn"
                data-tooltip={t("cancel")}
                aria-label={t("cancel")}
                onPointerDown={(event) => event.preventDefault()}
                onClick={onCancel}
            >
                ✕
            </button>
        </div>
    );
}
