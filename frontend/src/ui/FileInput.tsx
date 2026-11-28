/*
 * 隐藏的文件选择输入。
 *
 * 【为什么需要统一】本仓库有四处"点按钮 → 选文件"：导入主题（JSON）、导入布局
 * （JSON）、导入颤音预设（JSON）、向记事本插入图片。四处各自手写一遍
 * `<input type="file">`，也就各自漏掉或写错了一部分：
 *
 * - **必须清空 `value`**：浏览器只在"值发生变化"时触发 `change`。用户第一次选了
 *   `a.json`、第二次又选同一个文件时，不清空就**不会再触发** —— 表现为"点了没反应"。
 *   四处里有一处漏了这一点。
 * - 隐藏方式不统一（`className="hidden"` / `style={{display:"none"}}`），
 *   也就没有一个地方可以说明"为什么它必须隐藏但仍要常驻挂载"。
 * - `accept` 与 `multiple` 的取值散在各处，没有共同的默认。
 *
 * 【为什么是组件 + 外部 ref，而不是自带按钮】四处的触发者形态不同：三个是普通
 * 按钮，另一个（布局导入）的触发项在**菜单项**里 —— 菜单关闭会卸载菜单子树，而
 * 文件选择期间 `change` 事件必须有人接。因此输入框由**常驻宿主**渲染、触发者只
 * 通过 ref 调 `click()`。组件只负责输入框本身，触发者长什么样由调用方决定。
 *
 * 【为什么移动端要换成原生选择器】Android WebView **不会**自己处理
 * `<input type="file">`：它要求宿主实现 `WebChromeClient.onShowFileChooser`，
 * 而 wry 0.57 没有实现。于是移动端上这个输入框点了**毫无反应**（没有报错、没有
 * 日志），四处的导入入口全部失效。原生选择器（SAF）只能由后端发起，所以移动形态
 * 下拦截点击、改走 `services/mobileFilePicker`，再用同样的 `File` 对象回调 ——
 * 调用方一行都不用改。
 */
import type { RefObject } from "react";

import { frontendLog } from "../services/frontendLog";
import { pickFilesOnMobile, shouldUseNativePicker } from "../services/mobileFilePicker";

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
    /**
     * 选择器本身失败时的回调（移动端的原生选择器不可用 / 文件过大）。
     *
     * 【为什么与 `onFiles` 分开】"选完了"与"选不了"是两件事：前者交给业务处理，
     * 后者要让用户知道这个入口坏了。不给这个回调时退化成一条前端日志 ——
     * 至少不会变成"点了没反应"。
     */
    onError?: (error: Error) => void;
}

export function AppFileInput({ inputRef, accept, multiple, onFiles, onError }: AppFileInputProps) {
    const reportFailure = (error: unknown) => {
        const normalized = error instanceof Error ? error : new Error(String(error));
        frontendLog("error", "[AppFileInput] file picker failed", normalized);
        onError?.(normalized);
    };

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
            /*
             * 移动形态：拦下这次点击，改走原生选择器。
             *
             * 【为什么 `preventDefault` 是安全的】桌面形态直接返回，行为与引入本
             * 分支之前逐字节相同；移动形态上 WebView 本来就不会弹任何东西
             * （wry 没有 `onShowFileChooser`），拦下它只是让"点了没反应"变成
             * "真的选了文件"。
             */
            onClick={(event) => {
                if (!shouldUseNativePicker()) return;
                event.preventDefault();
                void pickFilesOnMobile({ accept, multiple })
                    .then((files) => {
                        if (files.length > 0) onFiles(files);
                    })
                    .catch(reportFailure);
            }}
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
