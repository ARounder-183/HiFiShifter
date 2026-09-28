/*
 * 隐藏的文件选择输入。
 *
 * 【为什么需要统一】本仓库有三处"点按钮 → 选文件"：导入主题（JSON）、导入布局
 * （JSON）、向记事本插入图片。三处各自手写了一遍 `<input type="file">`，也就各自
 * 漏掉或写错了一部分：
 *
 * - **必须清空 `value`**：浏览器只在"值发生变化"时触发 `change`。用户第一次选了
 *   `a.json`、第二次又选同一个文件时，不清空就**不会再触发** —— 表现为"点了没反应"。
 *   三处里有一处漏了这一点。
 * - 隐藏方式不统一（`className="hidden"` / `style={{display:"none"}}`），
 *   也就没有一个地方可以说明"为什么它必须隐藏但仍要常驻挂载"。
 * - `accept` 与 `multiple` 的取值散在各处，没有共同的默认。
 *
 * 【为什么是组件 + 外部 ref，而不是自带按钮】三处的触发者形态不同：两个是普通
 * 按钮，另一个（布局导入）的触发项在**菜单项**里 —— 菜单关闭会卸载菜单子树，而
 * 文件选择期间 `change` 事件必须有人接。因此输入框由**常驻宿主**渲染、触发者只
 * 通过 ref 调 `click()`。组件只负责输入框本身，触发者长什么样由调用方决定。
 */
import type { RefObject } from "react";

export interface AppFileInputProps {
    /**
     * 触发用的 ref。调用方 `inputRef.current?.click()` 打开选择框。
     *
     * 输入框必须由**常驻**组件渲染（不随触发它的菜单/弹层卸载），否则用户在系统
     * 文件对话框里点确定时，`change` 事件的接收方已经不在了。
     */
    inputRef: RefObject<HTMLInputElement | null>;
    /** 传给原生 `accept`。省略 = 接受任意文件。 */
    accept?: string;
    /** 允许多选。回调会一次收到全部文件。 */
    multiple?: boolean;
    /** 选中文件后的回调。**取消选择（空列表）不触发**。 */
    onFiles: (files: File[]) => void;
}

export function AppFileInput({ inputRef, accept, multiple, onFiles }: AppFileInputProps) {
    return (
        <input
            ref={inputRef}
            type="file"
            accept={accept}
            multiple={multiple}
            // 由触发按钮承担可访问名称；输入框本身不进 Tab 序、不被读出。
            className="hidden"
            tabIndex={-1}
            aria-hidden
            onChange={(event) => {
                const files = Array.from(event.target.files ?? []);
                /*
                 * 先清空再回调：清空 `value` 才会让"再次选同一个文件"重新触发
                 * `change`。顺序不能颠倒 —— 回调里若同步抛出，`value` 就再也没机会
                 * 被清掉，此后这个输入框会一直静默。
                 */
                event.target.value = "";
                if (files.length > 0) onFiles(files);
            }}
        />
    );
}
