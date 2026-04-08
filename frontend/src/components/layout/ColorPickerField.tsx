/**
 * 颜色选择器字段组件
 *
 * 提供一个 <input type="color"> 色块 + hex 文本框的组合输入，
 * 用于外观设置对话框中各颜色 token 的编辑。
 */

import React, { useCallback } from "react";

interface ColorPickerFieldProps {
    /** 颜色值（hex 格式如 #353535） */
    value: string;
    /** 颜色变化回调 */
    onChange: (color: string) => void;
    /** 标签文本 */
    label: string;
    /** 是否禁用 */
    disabled?: boolean;
}

export const ColorPickerField: React.FC<ColorPickerFieldProps> = ({
    value,
    onChange,
    label,
    disabled = false,
}) => {
    const handleColorInput = useCallback(
        (e: React.ChangeEvent<HTMLInputElement>) => {
            onChange(e.target.value);
        },
        [onChange],
    );

    const handleTextInput = useCallback(
        (e: React.ChangeEvent<HTMLInputElement>) => {
            const val = e.target.value;
            onChange(val);
        },
        [onChange],
    );

    /** 确保传给 <input type="color"> 的是有效 hex（7 位），否则回退为黑色 */
    const safeHex = /^#[0-9a-fA-F]{6}$/.test(value) ? value : "#000000";

    return (
        <div className="flex items-center gap-2 min-w-0">
            <label
                className="text-xs text-qt-text-muted truncate shrink-0 w-[140px] text-right"
                title={label}
            >
                {label}
            </label>
            <input
                type="color"
                value={safeHex}
                onChange={handleColorInput}
                disabled={disabled}
                className="w-7 h-7 rounded border border-qt-border cursor-pointer bg-transparent shrink-0 p-0"
                style={{ WebkitAppearance: "none" }}
            />
            <input
                type="text"
                value={value}
                onChange={handleTextInput}
                disabled={disabled}
                className="w-[100px] px-2 py-1 text-xs rounded border border-qt-border bg-qt-base text-qt-text font-mono"
                placeholder="#rrggbb"
                spellCheck={false}
            />
        </div>
    );
};
