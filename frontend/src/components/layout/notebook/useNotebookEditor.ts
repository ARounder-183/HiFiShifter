/*
 * 记事本编辑器内核的挂载与同步。
 *
 * ## 数据流（单向，无双份状态）
 *
 * ```
 * 用户输入 → TipTap 事务 → 序列化 Markdown
 *          ├─ dispatch(setProjectNotesMarkdown)   // Redux 立即更新（保存路径读它）
 *          └─ debounce → webApi.setProjectNotes   // 后端登记为一步可撤销操作
 *
 * 外来更新（工程加载 / 应用级撤销重做）→ 与"上次自己发出的值"不同才
 * setContent(..., { emitUpdate: false })，避免打字过程中被自己回写打断。
 * ```
 *
 * ## 撤销
 *
 * 编辑器内 Ctrl+Z 先走 ProseMirror 自己的历史（细粒度、保留光标位置）；
 * 历史耗尽才回落到应用级撤销（跨过整步"编辑记事本"）。这样两个栈的语义是
 * 明确的"先细后粗"，而不是互相打架。
 *
 * ## 与后端历史的衔接
 *
 * 后端把连续的记事本写入**结构性合并**成一步（无论打字多久）。切模式、
 * 失焦、关闭面板、编辑停顿超过 `historySplitIdleMs` 时调用
 * `seal_project_notes_history`，让下一步另起 —— 这是"这一节到此为止"的
 * 显式信号，而不是靠猜。
 */

import { Extension, type Editor } from "@tiptap/core";
import { useEditor } from "@tiptap/react";
import { useCallback, useEffect, useMemo, useRef } from "react";

import { notebookApi } from "../../../services/api/notebook";
import { documentMarkdown } from "./markdownCodec";
import { buildNotebookExtensions } from "./notebookExtensions";
import type { ResolvedNotebookSettings } from "./notebookSettings";

export interface NotebookEditorBridge {
    /** 应用级撤销/重做（编辑器内历史耗尽后的回落）。 */
    undo: () => void;
    redo: () => void;
}

/** Ctrl+Z 的"先细后粗"桥接扩展。 */
function createUndoBridge(bridge: NotebookEditorBridge) {
    return Extension.create({
        name: "notebookUndoBridge",
        addKeyboardShortcuts() {
            return {
                "Mod-z": () => {
                    if (this.editor.commands.undo()) return true;
                    bridge.undo();
                    return true;
                },
                "Mod-Shift-z": () => {
                    if (this.editor.commands.redo()) return true;
                    bridge.redo();
                    return true;
                },
                "Mod-y": () => {
                    if (this.editor.commands.redo()) return true;
                    bridge.redo();
                    return true;
                },
            };
        },
    });
}

export interface UseNotebookEditorArgs {
    /** 正文（Markdown）。 */
    markdown: string;
    settings: ResolvedNotebookSettings;
    placeholder: string;
    bridge: NotebookEditorBridge;
    /** 写入 Redux（立即）。 */
    onMarkdownChange: (markdown: string) => void;
    /** 写入后端（去抖）。 */
    persist: (markdown: string) => void;
    /** 编辑停顿到阈值时另起撤销步。 */
    onIdleSplit: () => void;
}

export interface UseNotebookEditorResult {
    editor: Editor | null;
    /** 立即把待写内容刷到后端（失焦/切模式/关面板/保存前调用）。 */
    flush: () => void;
    /** 关闭撤销合并窗口。 */
    seal: () => void;
}

export function useNotebookEditor(args: UseNotebookEditorArgs): UseNotebookEditorResult {
    const { markdown, settings, placeholder, bridge, onMarkdownChange, persist, onIdleSplit } =
        args;

    // 回调与设置在 ref 里取最新值：编辑器实例不应因为回调身份变化而重建。
    // 赋值必须放在 effect 里（而非渲染期）—— 渲染期写 ref 会破坏并发渲染的
    // 假设，也会被 react-hooks/refs 判为错误。
    const onMarkdownChangeRef = useRef(onMarkdownChange);
    const persistRef = useRef(persist);
    const onIdleSplitRef = useRef(onIdleSplit);
    useEffect(() => {
        onMarkdownChangeRef.current = onMarkdownChange;
        persistRef.current = persist;
        onIdleSplitRef.current = onIdleSplit;
    }, [onIdleSplit, onMarkdownChange, persist]);

    /** 最近一次"由本编辑器写出"的 Markdown，用于识别外来更新。 */
    const lastEmittedRef = useRef(markdown);
    /** 待写入后端的文本（null = 无待写内容）。 */
    const pendingRef = useRef<string | null>(null);
    const debounceTimerRef = useRef<number | null>(null);
    const idleTimerRef = useRef<number | null>(null);

    const extensions = useMemo(
        () =>
            buildNotebookExtensions({
                placeholder,
                markdownShortcuts: settings.markdownShortcuts,
                slashCommands: settings.slashCommands,
            }),
        // 只在"是否启用输入规则"这类结构性开关变化时重建扩展。
        [placeholder, settings.markdownShortcuts, settings.slashCommands],
    );

    const undoBridge = useMemo(() => createUndoBridge(bridge), [bridge]);

    const clearTimers = useCallback(() => {
        if (debounceTimerRef.current !== null) {
            window.clearTimeout(debounceTimerRef.current);
            debounceTimerRef.current = null;
        }
        if (idleTimerRef.current !== null) {
            window.clearTimeout(idleTimerRef.current);
            idleTimerRef.current = null;
        }
    }, []);

    const flush = useCallback(() => {
        if (debounceTimerRef.current !== null) {
            window.clearTimeout(debounceTimerRef.current);
            debounceTimerRef.current = null;
        }
        const pending = pendingRef.current;
        pendingRef.current = null;
        if (pending !== null) persistRef.current(pending);
    }, []);

    const seal = useCallback(() => {
        flush();
        void notebookApi.sealNotesHistory().catch(() => {
            // 分节失败不该影响编辑（最坏情况只是撤销粒度更粗）。
        });
    }, [flush]);

    const editor = useEditor(
        {
            extensions: [...extensions, undoBridge],
            content: markdown,
            editable: true,
            editorProps: {
                attributes: {
                    // 显式保留 TipTap/ProseMirror 的默认类名：`editorProps`
                    // 是整体替换而不是深合并，漏掉它们会让依赖这些类名的
                    // 样式与查询静默失效。
                    class: "tiptap ProseMirror hs-notebook-prose",
                    spellcheck: settings.spellCheck ? "true" : "false",
                    // 声明为"可选择表面"：应用默认整页不可选中文本（DAW 风格），
                    // 只有显式标记的区域才放行原生选择。缺了它，应用在
                    // `selectstart` / `mouseup` 上的守卫会把编辑器里的选择当成
                    // "误选"清掉 —— 表现为拖选无效、点击文字不落光标。
                    // 用应用既有的 data-hs-selectable 机制，而不是给编辑器开后门。
                    "data-hs-selectable": "true",
                },
            },
            onUpdate: ({ editor: instance }) => {
                let next: string;
                try {
                    next = documentMarkdown(instance);
                } catch {
                    return;
                }
                lastEmittedRef.current = next;
                onMarkdownChangeRef.current(next);
                pendingRef.current = next;

                const delay = settings.autosaveDebounceMs;
                if (debounceTimerRef.current !== null) {
                    window.clearTimeout(debounceTimerRef.current);
                }
                debounceTimerRef.current = window.setTimeout(() => {
                    debounceTimerRef.current = null;
                    const value = pendingRef.current;
                    pendingRef.current = null;
                    if (value !== null) persistRef.current(value);
                }, delay);

                // 停顿分节：连续打字时不触发，停手一段时间后让下一步另起。
                if (settings.historySplitIdleMs > 0) {
                    if (idleTimerRef.current !== null) window.clearTimeout(idleTimerRef.current);
                    idleTimerRef.current = window.setTimeout(() => {
                        idleTimerRef.current = null;
                        onIdleSplitRef.current();
                    }, settings.historySplitIdleMs);
                }
            },
            onBlur: () => {
                // 失焦即收尾：把待写内容落盘并关掉合并窗口，这样用户切去时间轴
                // 操作后，撤销不会再和"刚才那一段笔记"合并成同一步。
                seal();
            },
        },
        // 刻意**不**把视图模式放进依赖：切到源码视图时编辑器只是不再渲染
        // （`EditorContent` 卸载），实例本身仍要保留 —— 重建会让每次切模式
        // 都销毁并新建一个编辑器（ProseMirror 历史清零、StrictMode 下还会
        // 多泄漏一个实例），而源码视图里的改动已由下面的"外来更新"effect
        // 通过 `setContent` 同步回来。
        [extensions, undoBridge],
    );

    /**
     * 运行时设置（不重建编辑器）。
     *
     * `enableInputRules` 是每次输入时现读的选项，`setOptions` 立即生效；
     * 拼写检查则**直接写 DOM 属性** —— `editorProps.attributes` 只在视图创建
     * 时应用，改它不会回写已存在的元素。
     *
     * 两处都先判 `isDestroyed`：`useEditor` 在依赖变化时会销毁旧实例，而
     * 销毁后的 `editor.view` 是个**会抛异常的 Proxy**，`editor.commands` 的
     * commandManager 也已被置空。
     */
    useEffect(() => {
        if (!editor || editor.isDestroyed) return;
        editor.setOptions({ enableInputRules: settings.markdownShortcuts });
    }, [editor, settings.markdownShortcuts]);

    useEffect(() => {
        if (!editor || editor.isDestroyed) return;
        editor.view.dom.setAttribute("spellcheck", settings.spellCheck ? "true" : "false");
    }, [editor, settings.spellCheck]);

    /**
     * 外来正文更新。
     *
     * 只在"与上次自己写出的值不同"时才覆盖编辑器内容 —— 否则打字过程中
     * 任何一次 Redux 往返都会把光标踢回开头。
     */
    useEffect(() => {
        if (!editor || editor.isDestroyed) return;
        if (markdown === lastEmittedRef.current) return;
        lastEmittedRef.current = markdown;
        pendingRef.current = null;
        try {
            editor.commands.setContent(markdown, { emitUpdate: false });
        } catch {
            // 极少数情况下（编辑器正在重建）命令不可用：内容会在下次渲染
            // 时由同一个 effect 重新同步，这里不值得把面板带崩。
        }
    }, [editor, markdown]);

    // 卸载时收尾：清定时器并把最后的内容写出去。
    useEffect(() => {
        return () => {
            clearTimers();
            const pending = pendingRef.current;
            pendingRef.current = null;
            if (pending !== null) persistRef.current(pending);
        };
    }, [clearTimers]);

    return { editor, flush, seal };
}
