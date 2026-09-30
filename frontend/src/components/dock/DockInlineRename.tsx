/*
 * 行内重命名输入框 —— 停靠标签与浮动标题条共用。
 *
 * 【交互约定】与时间轴 Clip 名称的行内重命名同一套原则：
 * - 进入编辑时预填当前自定义标题并全选：直接输入即整体覆盖，点击可局部修改；
 * - Enter 提交，Esc 取消，点击输入框以外 = 提交。提交监听走**捕获阶段**的
 *   window pointerdown，而不是失焦 —— 失焦在 portal/卸载场景下不可靠（与
 *   `DockTabMenu` 的 RenameInput 同一款结论）；
 * - 提交空串 = 清除自定义标题：面板退回"从内容派生"的标题（`renameForm`
 *   对空串即清 title）；
 * - 输入框吃掉自己的 pointerdown / keydown / dblclick：标签的拖拽启动、双击、
 *   键盘模型不得被编辑态打断。
 */

import { useEffect, useRef, useState } from "react";

export interface DockInlineRenameProps {
    /** 进入编辑时的预填文本（当前的自定义标题；未命名时为空串）。 */
    initial: string;
    /** 未命名时的占位文本（显示当前生效的派生标题，提示"它现在叫什么"）。 */
    placeholder?: string;
    ariaLabel: string;
    onCommit: (title: string) => void;
    onCancel: () => void;
}

export function DockInlineRename({
    initial,
    placeholder,
    ariaLabel,
    onCommit,
    onCancel,
}: DockInlineRenameProps) {
    const [draft, setDraft] = useState(initial);
    const ref = useRef<HTMLInputElement | null>(null);
    // 点击外部提交时读最新草稿：在事件处理器里同步（渲染期写 ref 违反
    // React Compiler 的引用规则）。
    const draftRef = useRef(initial);
    // 提交/取消各只走一次：点击外部（捕获）与随后的失焦可能先后到达，
    // 第二次必须空转 —— 否则 Escape 取消后又被 blur "复活"成提交。
    const doneRef = useRef(false);

    function commitOnce() {
        if (doneRef.current) return;
        doneRef.current = true;
        onCommit(draftRef.current);
    }

    function cancelOnce() {
        if (doneRef.current) return;
        doneRef.current = true;
        onCancel();
    }

    // 挂载即全选：只跑一次。放进订阅 effect 会随每次按键重选全文，输入法
    // 与普通输入都会被打断。
    useEffect(() => {
        ref.current?.select();
    }, []);

    // 订阅不设依赖：每次渲染重挂（代价是一次 add/remove），换取监听器永远
    // 读到最新的 commitOnce 闭包。
    useEffect(() => {
        function onPointerDown(event: PointerEvent) {
            if (ref.current?.contains(event.target as Node)) return;
            commitOnce();
        }
        window.addEventListener("pointerdown", onPointerDown, true);
        return () => window.removeEventListener("pointerdown", onPointerDown, true);
    });

    return (
        <input
            ref={ref}
            className="hs-dock-rename-input"
            aria-label={ariaLabel}
            autoFocus
            value={draft}
            placeholder={placeholder}
            onChange={(event) => {
                setDraft(event.target.value);
                draftRef.current = event.target.value;
            }}
            onPointerDown={(event) => event.stopPropagation()}
            onDoubleClick={(event) => event.stopPropagation()}
            onKeyDown={(event) => {
                event.stopPropagation();
                if (event.key === "Enter") {
                    event.preventDefault();
                    commitOnce();
                    return;
                }
                if (event.key === "Escape") {
                    event.preventDefault();
                    cancelOnce();
                }
            }}
            onBlur={() => commitOnce()}
        />
    );
}
