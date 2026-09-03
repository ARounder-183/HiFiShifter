import { useEffect, useRef, useState, type PropsWithChildren, type ReactNode } from "react";

/**
 * 富内容注册事件：任意模块可以把某个元素的信息浮标设为 ReactNode
 * （如内联曲线图标）。AppTooltipProvider 监听此事件维护注册表。
 */
export const HS_TOOLTIP_CONTENT_EVENT = "hs-tooltip-content";
import { createPortal } from "react-dom";

export type AppTooltipPosition = {
    x: number;
    y: number;
};

/** 气泡内容：纯文本，或含内联 SVG 图标等 ReactNode。 */
export type AppTooltipContent = string | ReactNode | null;

/** 气泡定位夹紧：右侧预留约 320px 宽度余量（长链接气泡的上限宽度），
 *  底部预留三行文本（约 56px 高）的空间。 */
function clampTooltipPosition(position: AppTooltipPosition): AppTooltipPosition {
    return {
        x: Math.min(position.x + 14, Math.max(8, window.innerWidth - 320)),
        y: Math.min(position.y + 18, Math.max(8, window.innerHeight - 88)),
    };
}

/** 气泡内容：纯文本，或含内联 SVG 图标等信息浮标的 ReactNode。 */
function isRenderable(content: AppTooltipContent): boolean {
    if (content == null) return false;
    if (typeof content === "string") return content.length > 0;
    return true;
}

let contentSeq = 0;

/** 给 ReactNode 内容一个稳定的显示键，避免气泡频繁重建。 */
function nodeDisplayKey(): string {
    contentSeq += 1;
    return `__hs_tooltip_node_${contentSeq}`;
}

export function AppTooltipBubble({
    text,
    position,
}: {
    text: AppTooltipContent;
    position: AppTooltipPosition | null;
}) {
    if (!position || !isRenderable(text)) return null;

    const clamped = clampTooltipPosition(position);
    return createPortal(
        <div className="app-tooltip" role="tooltip" style={{ left: clamped.x, top: clamped.y }}>
            {text}
        </div>,
        document.body,
    );
}

export function AppTooltipProvider({
    children,
    isSuppressedExternal,
}: PropsWithChildren & {
    /** 外部抑制查询（如淡变上下文菜单打开时隐藏信息浮标）。 */
    isSuppressedExternal?: () => boolean;
}) {
    const [tooltip, setTooltip] = useState<{
        text: AppTooltipContent;
        /** ReactNode 内容的稳定键（字符串内容恒为 undefined）。 */
        nodeKey?: string;
        position: AppTooltipPosition;
    } | null>(null);
    // 外部抑制 getter 以 ref 桥接进常驻 effect，闭包变化不重启监听。
    const suppressGetterRef = useRef(isSuppressedExternal);
    // eslint-disable-next-line react-hooks/refs -- render 期写 ref 镜像：命令式绘制/事件回调需在同一提交内读取最新值（热路径既有模式）
    suppressGetterRef.current = isSuppressedExternal;

    useEffect(() => {
        let currentElement: Element | null = null;
        let lastEvent: PointerEvent | null = null;
        // 外部抑制查询：① props getter（淡变菜单打开）；② 任意上下文菜单
        // （Clip / 轨道背景 / 时间标尺等，均带 data-hs-context-menu 标记）
        // 正挂载在 DOM 中。选择器查询仅在指针事件与 DOM 变更回调里执行。
        const hasOpenContextMenu = (): boolean => {
            if (suppressGetterRef.current?.() === true) return true;
            return document.querySelector("[data-hs-floating-menu]") != null;
        };
        // ── 手势钉住（pin）语义 ─────────────────────────────────────
        // 拖拽淡化包络等细粒度控件时，指针会在一串相邻小命中块之间移动，
        // 块与块的间隙下方是没有 data-tooltip 的元素；旧逻辑因此"隐藏→
        // 显示→再定位"反复横跳（用户看到的闪烁/乱动）。现在：在带
        // tooltip 的元素上按下即进入钉住模式——气泡位置冻结在按下点，
        // 直到 pointerup/cancel；期间忽略 pointerover/out 的切换，
        // 文本仍随源元素属性变化实时刷新（MutationObserver 路径）。
        let pinned = false;
        let pinnedPosition: AppTooltipPosition | null = null;
        // ReactNode 内容注册表：元素 → {content, key}。设置方通过全局事件
        // 写入；Provider 在悬停/钉住刷新时优先取用富内容。
        const customContentByElement = new Map<
            Element,
            { content: AppTooltipContent; key: string }
        >();

        function customContentRegister(
            element: Element,
            content: AppTooltipContent,
        ): string | null {
            if (!isRenderable(content)) {
                customContentByElement.delete(element);
                return null;
            }
            const existing = customContentByElement.get(element);
            if (existing && existing.content === content) return existing.key;
            const key = nodeDisplayKey();
            customContentByElement.set(element, { content, key });
            return key;
        }

        const onCustomContent = (event: Event) => {
            const detail = (event as CustomEvent).detail as {
                element: Element;
                content: AppTooltipContent;
            } | null;
            if (!detail || !customContentRegister(detail.element, detail.content)) return;
            if (currentElement === detail.element) refreshCurrentTooltip();
        };

        const resolveContent = (
            element: Element | null,
        ): { content: AppTooltipContent; nodeKey?: string } => {
            if (!element) return { content: "" };
            const cached = customContentByElement.get(element);
            if (cached) return { content: cached.content, nodeKey: cached.key };
            return { content: element.getAttribute("data-tooltip") || "" };
        };

        const onPointerDown = (event: PointerEvent) => {
            // 右键即将弹出（或已弹出）上下文菜单：绝不钉住浮标；
            // 若气泡正在显示，此刻立即清除 —— 用户右键的第一反馈就是浮标消失。
            if (event.button !== 0 || hasOpenMenuNow()) {
                pinned = false;
                pinnedPosition = null;
                currentElement = null;
                setTooltip(null);
                return;
            }
            const target = event.target instanceof Element ? event.target : null;
            const element = target?.closest?.("[data-tooltip]") ?? null;
            const attr = element?.getAttribute("data-tooltip") || "";
            if (element && (attr || customContentByElement.has(element))) {
                pinned = true;
                currentElement = element;
                pinnedPosition = clampTooltipPosition({
                    x: event.clientX,
                    y: event.clientY,
                });
                const cached = customContentByElement.get(element);
                setTooltip(
                    cached
                        ? { text: cached.content, nodeKey: cached.key, position: pinnedPosition }
                        : { text: attr, position: pinnedPosition },
                );
            } else {
                // 按在其他地方：结束钉住（后续 pointerover 照常工作）。
                pinned = false;
                pinnedPosition = null;
            }
        };
        const onGestureEnd = () => {
            pinned = false;
            pinnedPosition = null;
        };

        const showAt = (
            content: AppTooltipContent,
            nodeKey: string | undefined,
            position: AppTooltipPosition,
        ) => {
            setTooltip((prev) =>
                prev &&
                prev.nodeKey === nodeKey &&
                typeof content === "string" &&
                typeof prev.text === "string" &&
                prev.text === content &&
                // 位置也参与去重：文本未变但鼠标移动时必须跟随更新位置，
                // 否则气泡会停在首次悬停的位置（hover 跟随语义）。
                prev.position.x === position.x &&
                prev.position.y === position.y
                    ? prev
                    : { text: content, nodeKey, position },
            );
        };

        const updateFromEvent = (event: PointerEvent) => {
            lastEvent = event;
            if (pinned) return;
            const target = event.target instanceof Element ? event.target : null;
            // 富内容元素可能没有 data-tooltip 属性 —— 容器带 data-hs-rich-tooltip 标记。
            const element = target?.closest?.("[data-tooltip], [data-hs-rich-tooltip]") ?? null;

            if (element !== currentElement) {
                currentElement = element;
                if (!element) {
                    setTooltip(null);
                    return;
                }
            }
            if (!element) return;

            if (hasOpenContextMenu()) {
                setTooltip(null);
                currentElement = null;
                return;
            }
            const { content, nodeKey } = resolveContent(element);
            if (!isRenderable(content)) {
                setTooltip(null);
                return;
            }
            showAt(content, nodeKey, { x: event.clientX, y: event.clientY });
        };

        const refreshCurrentTooltip = () => {
            if (!currentElement) return;
            const { content, nodeKey } = resolveContent(currentElement);
            if (!isRenderable(content)) {
                setTooltip(null);
                return;
            }
            setTooltip((prev) =>
                prev && prev.nodeKey === nodeKey && prev.text === content && nodeKey != null
                    ? prev
                    : {
                          text: content,
                          nodeKey,
                          position:
                              pinned && pinnedPosition
                                  ? pinnedPosition
                                  : lastEvent
                                    ? { x: lastEvent.clientX, y: lastEvent.clientY }
                                    : { x: 0, y: 0 },
                      },
            );
        };

        const onPointerOut = (event: PointerEvent) => {
            if (pinned) return;
            const target = event.target instanceof Element ? event.target : null;
            const relatedTarget =
                event.relatedTarget instanceof Element ? event.relatedTarget : null;
            if (
                target?.closest?.("[data-tooltip], [data-hs-rich-tooltip]") &&
                !relatedTarget?.closest?.("[data-tooltip], [data-hs-rich-tooltip]")
            ) {
                currentElement = null;
                setTooltip(null);
            }
        };

        // 除 click（可能被业务 stopPropagation 拦截）外，用 MutationObserver
        // 监听 data-tooltip 属性变化：拖拽中 Redux 更新 → React 重渲染同元素
        // 的属性（长度/曲率数值实时变化）→ 气泡文本即时刷新，位置不重摆。
        const clearForMenu = () => {
            pinned = false;
            pinnedPosition = null;
            currentElement = null;
            setTooltip(null);
        };
        const hasOpenMenuNow = () =>
            suppressGetterRef.current?.() === true ||
            document.querySelector("[data-hs-floating-menu]") != null;

        const tooltipObserver = new MutationObserver((mutations) => {
            // 菜单挂载/卸载属于结构变化 —— 浮标必须【立即】消失，
            // 不能等下一次指针移动才触发 updateFromEvent 清理。
            if (currentElement && hasOpenMenuNow()) {
                clearForMenu();
                return;
            }
            for (const mutation of mutations) {
                if (
                    mutation.type === "attributes" &&
                    mutation.attributeName === "data-tooltip" &&
                    mutation.target === currentElement
                ) {
                    refreshCurrentTooltip();
                    break;
                }
            }
        });
        tooltipObserver.observe(document.body, {
            attributes: true,
            attributeFilter: ["data-tooltip"],
            childList: true,
            subtree: true,
        });

        const onPointerOver = (event: PointerEvent) => updateFromEvent(event);

        const onPointerMove = (event: PointerEvent) => {
            lastEvent = event;
            if (!currentElement) return;
            if (hasOpenContextMenu()) {
                pinned = false;
                pinnedPosition = null;
                setTooltip(null);
                currentElement = null;
                return;
            }
            const { content, nodeKey } = resolveContent(currentElement);
            if (pinned) {
                // 钉住模式：只同步内容（数值可能已随拖拽更新），位置冻结。
                if (!isRenderable(content)) return;
                setTooltip((prev) =>
                    prev && prev.nodeKey === nodeKey && prev.text === content
                        ? prev
                        : {
                              text: content,
                              nodeKey,
                              position: pinnedPosition ?? prev?.position ?? { x: 0, y: 0 },
                          },
                );
                return;
            }
            if (!isRenderable(content)) {
                setTooltip(null);
                return;
            }
            showAt(content, nodeKey, { x: event.clientX, y: event.clientY });
        };

        window.addEventListener(HS_TOOLTIP_CONTENT_EVENT, onCustomContent);
        document.addEventListener("pointerdown", onPointerDown);
        document.addEventListener("pointerup", onGestureEnd);
        document.addEventListener("pointercancel", onGestureEnd);
        document.addEventListener("pointerover", onPointerOver);
        document.addEventListener("pointermove", onPointerMove);
        document.addEventListener("pointerout", onPointerOut);
        return () => {
            tooltipObserver.disconnect();
            window.removeEventListener(HS_TOOLTIP_CONTENT_EVENT, onCustomContent);
            document.removeEventListener("pointerdown", onPointerDown);
            document.removeEventListener("pointerup", onGestureEnd);
            document.removeEventListener("pointercancel", onGestureEnd);
            document.removeEventListener("pointerover", onPointerOver);
            document.removeEventListener("pointermove", onPointerMove);
            document.removeEventListener("pointerout", onPointerOut);
        };
    }, []);

    return (
        <>
            {children}
            <AppTooltipBubble text={tooltip?.text ?? ""} position={tooltip?.position ?? null} />
        </>
    );
}
