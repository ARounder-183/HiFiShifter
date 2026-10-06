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
 * replaceDocumentWithoutHistory()：一条带 `preventUpdate` + `addToHistory:
 * false` 的事务整体替换（见函数注释），避免打字过程中被自己回写打断，也避免
 * 外来更新混进编辑器的撤销栈。
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

import { Extension, createNodeFromContent, type Editor } from "@tiptap/core";
import { useEditor } from "@tiptap/react";
import { useCallback, useEffect, useMemo, useRef } from "react";

import { notebookApi } from "../../../services/api/notebook";
import { documentMarkdown, markdownStorage } from "./markdownCodec";
import { buildNotebookExtensions } from "./notebookExtensions";
import type { ResolvedNotebookSettings } from "./notebookSettings";

export interface NotebookEditorBridge {
    /** 应用级撤销/重做（编辑器内历史耗尽后的回落）。 */
    undo: () => void;
    redo: () => void;
}

/**
 * 链接浮层的入口（Ctrl/⌘+K）。
 *
 * 与 `NotebookEditorBridge` 同一约定：**对象身份必须稳定**（调用方用 `useMemo`
 * 包一层），否则 `useEditor` 的依赖会变、整个编辑器被重建（ProseMirror 历史
 * 清零）。
 */
export interface NotebookLinkBridge {
    /** 打开链接地址编辑浮层。 */
    open: () => void;
}

/**
 * 撤销一步：先编辑器内历史，耗尽再回落应用级。
 *
 * 【为什么抽成导出函数】这条"先细后粗"的规则此前只写在键盘快捷键的闭包里，
 * 于是**只有键盘能用**。右键菜单的"撤销"需要同一条规则，若在菜单里再抄一份，
 * 两份迟早会漂移（例如某天给应用级撤销加上"先 seal 再撤"的前置动作，只会改到
 * 其中一处）。这里把它变成唯一实现，键盘扩展与菜单都调它。
 */
export function runNotebookUndo(editor: Editor, bridge: NotebookEditorBridge): void {
    if (editor.commands.undo()) return;
    bridge.undo();
}

/** 重做一步。语义与 `runNotebookUndo` 对称（同样"先细后粗"）。 */
export function runNotebookRedo(editor: Editor, bridge: NotebookEditorBridge): void {
    if (editor.commands.redo()) return;
    bridge.redo();
}

/** Ctrl+Z 的"先细后粗"桥接扩展。 */
function createUndoBridge(bridge: NotebookEditorBridge) {
    return Extension.create({
        name: "notebookUndoBridge",
        addKeyboardShortcuts() {
            return {
                "Mod-z": () => {
                    runNotebookUndo(this.editor, bridge);
                    return true;
                },
                "Mod-Shift-z": () => {
                    runNotebookRedo(this.editor, bridge);
                    return true;
                },
                "Mod-y": () => {
                    runNotebookRedo(this.editor, bridge);
                    return true;
                },
            };
        },
    });
}

/**
 * Ctrl/⌘+K 打开链接编辑浮层。
 *
 * 【为什么是编辑器快捷键而不是面板上的 keydown 监听】
 *   1. `Mod-` 由 ProseMirror 的 keymap 按平台展开（macOS 是 ⌘、其余是 Ctrl），
 *      与加粗 / 斜体这些内建快捷键同一套，不必自己判平台；
 *   2. 它只在**编辑器持有焦点时**触发 —— 面板上的监听会连"焦点在查找框 / 链接
 *      输入框里"的 Ctrl+K 一起吃掉（原生监听先于 React 合成事件，输入框里的
 *      `stopPropagation` 拦不住它）。
 *
 * 【为什么不与加粗共用 `addKeyboardShortcuts` 的位置】加粗 / 斜体来自
 * StarterKit，链接是记事本自己接的动作；这里给它单独一个扩展，避免再去 extend
 * StarterKit。
 */
function createLinkShortcut(bridge: NotebookLinkBridge) {
    return Extension.create({
        name: "notebookLinkShortcut",
        addKeyboardShortcuts() {
            return {
                "Mod-k": () => {
                    bridge.open();
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
    bridge: NotebookEditorBridge;
    /** 写入 Redux（立即）。 */
    onMarkdownChange: (markdown: string) => void;
    /** 写入后端（去抖）。 */
    persist: (markdown: string) => void;
    /** 编辑停顿到阈值时另起撤销步。 */
    onIdleSplit: () => void;
    /** 链接浮层入口（Ctrl/⌘+K）。 */
    linkBridge: NotebookLinkBridge;
}

export interface UseNotebookEditorResult {
    editor: Editor | null;
    /** 立即把待写内容刷到后端（失焦/切模式/关面板/保存前调用）。 */
    flush: () => void;
    /** 关闭撤销合并窗口。 */
    seal: () => void;
}

export function useNotebookEditor(args: UseNotebookEditorArgs): UseNotebookEditorResult {
    const { markdown, settings, bridge, onMarkdownChange, persist, onIdleSplit, linkBridge } = args;

    // 回调与设置在 ref 里取最新值：编辑器实例不应因为回调身份变化而重建。
    // 赋值必须放在 effect 里（而非渲染期）—— 渲染期写 ref 会破坏并发渲染的
    // 假设，也会被 react-hooks/refs 判为错误。
    const onMarkdownChangeRef = useRef(onMarkdownChange);
    const persistRef = useRef(persist);
    const onIdleSplitRef = useRef(onIdleSplit);
    // 运行时设置也必须走 ref：`useEditor` 的依赖只有 extensions / undoBridge
    // （稳定），`onUpdate` 闭包只捕获**首次渲染**的 settings —— 直接在闭包里读
    // `settings.autosaveDebounceMs` / `historySplitIdleMs` 会让设置对话框里改的
    // 值直到应用重启才生效。
    const settingsRef = useRef(settings);
    useEffect(() => {
        onMarkdownChangeRef.current = onMarkdownChange;
        persistRef.current = persist;
        onIdleSplitRef.current = onIdleSplit;
        settingsRef.current = settings;
    }, [onIdleSplit, onMarkdownChange, persist, settings]);

    /** 最近一次"由本编辑器写出"的 Markdown，用于识别外来更新。 */
    const lastEmittedRef = useRef(markdown);
    /** 待写入后端的文本（null = 无待写内容）。 */
    const pendingRef = useRef<string | null>(null);
    const debounceTimerRef = useRef<number | null>(null);
    const idleTimerRef = useRef<number | null>(null);

    const extensions = useMemo(
        () =>
            buildNotebookExtensions({
                markdownShortcuts: settings.markdownShortcuts,
                slashCommands: settings.slashCommands,
            }),
        // 只在"是否启用输入规则"这类结构性开关变化时重建扩展。
        [settings.markdownShortcuts, settings.slashCommands],
    );

    const undoBridge = useMemo(() => createUndoBridge(bridge), [bridge]);

    /*
     * 链接快捷键扩展。回调经 `bridge` 对象转发（而不是在这里读 ref）：
     * React Compiler 的引用规则会拒绝"渲染期把读 ref 的闭包传给函数"，而
     * `bridge` 的身份由调用方用 `useMemo` 固定，因此这个扩展只建一次 ——
     * 与上面 `createUndoBridge` 同一套。
     */
    const linkShortcut = useMemo(() => createLinkShortcut(linkBridge), [linkBridge]);

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
            extensions: [...extensions, undoBridge, linkShortcut],
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

                const delay = settingsRef.current.autosaveDebounceMs;
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
                const idleDelay = settingsRef.current.historySplitIdleMs;
                if (idleDelay > 0) {
                    if (idleTimerRef.current !== null) window.clearTimeout(idleTimerRef.current);
                    idleTimerRef.current = window.setTimeout(() => {
                        idleTimerRef.current = null;
                        onIdleSplitRef.current();
                    }, idleDelay);
                }
            },
            onBlur: ({ event }) => {
                /*
                 * 焦点进了**我们自己刚打开的菜单**时不算"离开编辑器"。
                 *
                 * 【为什么看 `relatedTarget` 而不是 `document.activeElement`】
                 * 焦点切换的事件顺序是"旧元素 blur → 新元素 focus"，在 blur 处理器
                 * 里 `document.activeElement` 可能还停在旧元素上，判据会漏。而
                 * `relatedTarget` 就是即将获得焦点的那个元素，确定。
                 *
                 * 【为什么必须跳过】菜单（`AppContextMenu` 的 `autoFocus`）要收焦点
                 * 才能用方向键导航。若不跳过，每次右键都会走一遍收尾：把待写内容立刻
                 * 落盘并**关掉后端的撤销合并窗口** —— 用户只是右键复制一段文字，回到
                 * 编辑器继续打字，撤销步却已经被切成两段。
                 */
                const next = event.relatedTarget as HTMLElement | null;
                if (next?.closest?.('[data-hs-context-menu="1"]')) return;
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
        [extensions, undoBridge, linkShortcut],
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
            replaceDocumentWithoutHistory(editor, markdown);
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

/**
 * 用一条**不进撤销历史**的事务把文档整体替换为 markdown。
 *
 * 不能用 `editor.commands.setContent(...)`：它派生的事务只带 `preventUpdate`
 * 元数据，没有 `addToHistory: false`（见 @tiptap/core 的 setContent 实现）。
 * 于是应用级撤销/重做改写 `notesMarkdown` 之后，编辑器内第一次 Ctrl+Z 会先
 * 撤掉这条 setContent 事务 —— "应用级撤销"被编辑器撤销原样写回，两套栈打架。
 *
 * 流程与 tiptap-markdown 的 setContent 命令一致：markdown → HTML（它的
 * parser，含 hifi-clip 围栏等自定义块的解析规则）→ ProseMirror 文档；差别
 * 只在最后一跳由我们亲手 dispatch，并补两条元数据：
 * - `preventUpdate`：保持"外来更新不回写 onUpdate"的既有语义；
 * - `addToHistory: false`：prosemirror-history 会跳过这条事务，编辑器撤销栈
 *   不被它污染，应用级撤销与编辑器内撤销"先细后粗"的分工才成立。
 */
function replaceDocumentWithoutHistory(editor: Editor, markdown: string): void {
    const html = markdownStorage(editor).parser.parse(markdown);
    const parsed = createNodeFromContent(html, editor.schema, { slice: false });
    const tr = editor.state.tr
        .replaceWith(0, editor.state.doc.content.size, parsed)
        .setMeta("preventUpdate", true)
        .setMeta("addToHistory", false);
    editor.view.dispatch(tr);
}
