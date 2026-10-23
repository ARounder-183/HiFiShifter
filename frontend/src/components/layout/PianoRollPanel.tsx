/**
 * 参数编辑器面板（装配层）。
 *
 * 【这个文件为什么只剩三十行】面板在 2026-10 之前是约 9600 行的单文件。现在它只做三件事：
 * 起模型、铺外框、把三段视图摆进来。要找具体逻辑：
 *   - 状态 / 派生值 / 回调   → `pianoRoll/panel/usePianoRollPanelModel.tsx`
 *   - 工具栏（药丸 / 算法 / 参考轨道） → `pianoRoll/panel/ParamToolbar.tsx`
 *   - 标尺 + 卷帘画布 + 滚动条        → `pianoRoll/panel/EditorArea.tsx`
 *   - 浮层与模态                      → `pianoRoll/panel/PanelOverlays.tsx`
 *
 * ★ `React.memo` 不是"顺手加的性能优化"，而是**契约的一部分**：本组件由
 * `App.tsx` 通过 `setPanelRenderer(PANEL_PARAM_EDITOR, (form) => <PianoRollPanel .../>)`
 * 的闭包渲染，而 `DockPanelHosts` 的 `PanelMount` 是**直接调用**该闭包、没有 memo 边界
 * —— 于是 `AppInner` 每重渲染一次，这里就跟着重渲染一次（元素是新创建的，但 props
 * 只有一个稳定字符串，memo 的浅比较正好能拦住）。
 *
 * 【为什么不能给 `PanelMount` 加 memo 代替】那会连带冻结时间轴面板 —— 它的渲染闭包
 * 捕获了 `App.tsx` 的约三十个 props（见 `panelRenderer.ts` 的设计说明），冻结即意味着
 * props 不再更新。而本组件的 props 只有一个稳定 id，memo 是安全的。
 */
import React from "react";
import { Flex } from "@radix-ui/themes";

import { usePianoRollPanelModel } from "./pianoRoll/panel/usePianoRollPanelModel";
import { ParamToolbar } from "./pianoRoll/panel/ParamToolbar";
import { EditorArea } from "./pianoRoll/panel/EditorArea";
import { PanelOverlays } from "./pianoRoll/panel/PanelOverlays";

/**
 * 参数编辑器面板的 props。
 *
 * 只有一个**跨渲染稳定**的字符串 —— 这正是下面能用 `React.memo` 的前提。
 */
interface PianoRollPanelProps {
    /**
     * 本窗体在停靠布局里的 id（由 `setPanelRenderer` 注入）。
     *
     * 用于判断"我是否与时间轴上下堆叠" —— 同步偏移只在堆叠时有意义。缺省时
     * 按面板 id 回退查找，保证单独渲染（测试、独立窗口）也能工作。
     */
    dockFormId?: string;
}

const PianoRollPanelImpl: React.FC<PianoRollPanelProps> = ({ dockFormId }) => {
    const m = usePianoRollPanelModel(dockFormId);
    const { paramEditorRef } = m;

    return (
        <Flex
            ref={paramEditorRef}
            direction="column"
            className="relative h-full w-full bg-qt-graph-bg border-t border-qt-border"
            // 编辑表面声明：文档级 pointerdown/focusin 捕获据此把整个参数
            // 编辑器（工具栏/标尺/卷帘）解析为「pianoRoll」表面，作为复制/
            // 剪切/粘贴等编辑快捷键的归属依据（见 focusSurface.ts）。
            data-hs-surface="pianoRoll"
        >
            <ParamToolbar m={m} />
            <EditorArea m={m} />
            <PanelOverlays m={m} />
        </Flex>
    );
};

export const PianoRollPanel = React.memo(PianoRollPanelImpl);
