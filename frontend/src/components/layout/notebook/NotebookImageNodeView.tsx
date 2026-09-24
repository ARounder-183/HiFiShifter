/*
 * 图片 NodeView。
 *
 * 负责四件"官方 image 节点做不了"的事：
 * 1. **src 解析**：`hifi-asset://` / 相对路径都要经后端取字节再转 blob URL；
 * 2. **缺失占位**：附件被删或文件被移走时显示明确占位，而不是一个破图标；
 * 3. **拖拽改宽**：拖右侧把手改宽度，只在松手时提交一次事务（拖动过程中
 *    用本地状态预览，避免每个 pointermove 都写进撤销历史）；
 * 4. **右键菜单**：复制 / 另存为 / 在文件管理器中显示 / 还原宽度 / 删除。
 */

import { NodeViewWrapper, type NodeViewProps } from "@tiptap/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { useAppSelector } from "../../../app/hooks";
import { useI18n } from "../../../i18n/I18nProvider";
import { notebookApi } from "../../../services/api/notebook";
import { assetIdFromSrc, isAssetRef } from "./assetRef";
import { NotebookContextMenu, type NotebookMenuItem } from "./NotebookContextMenu";
import { resolveImage, subscribeAssetInvalidation } from "./notebookImageCache";
import { dirName } from "./notebookPaths";

const MIN_WIDTH = 48;

export function NotebookImageNodeView(props: NodeViewProps) {
    const { node, updateAttributes, deleteNode, selected } = props;
    const { t } = useI18n();
    const projectPath = useAppSelector((state) => state.session.project.path);
    const allowRemoteImages = useAppSelector((state) => state.notebook.settings.allowRemoteImages);
    const src = String(node.attrs.src ?? "");
    const width = typeof node.attrs.width === "number" ? node.attrs.width : null;
    const projectDir = useMemo(() => (projectPath ? dirName(projectPath) : null), [projectPath]);

    /**
     * 解析结果按 src 打标签：src 变化时**派生**出"加载中"，而不是在 effect
     * 体里同步 setState（那会触发级联渲染）。
     */
    const [resolved, setResolved] = useState<{ src: string; url: string | null; missing: boolean }>(
        {
            src: "",
            url: null,
            missing: false,
        },
    );
    const current = resolved.src === src ? resolved : { src, url: null, missing: false };
    const [menu, setMenu] = useState<{ x: number; y: number } | null>(null);
    const [editingAlt, setEditingAlt] = useState(false);
    const [altDraft, setAltDraft] = useState(String(node.attrs.alt ?? ""));
    const [dragWidth, setDragWidth] = useState<number | null>(null);
    const imgRef = useRef<HTMLImageElement | null>(null);

    // 解析 src（缓存 + 并发去重都在 notebookImageCache 里）。
    useEffect(() => {
        let cancelled = false;
        void resolveImage(src, { projectDir, allowRemoteImages }).then((result) => {
            if (!cancelled) setResolved({ src, url: result.url, missing: result.missing });
        });
        return () => {
            cancelled = true;
        };
    }, [src, projectDir, allowRemoteImages]);

    /**
     * 缓存被作废（切工程 / 显式失效）后重新解析一次。
     *
     * 这里**只会被"作废"触发**，不会被"解析完成"触发 —— 后者会让失败的 src
     * 陷入无界递归（见 notebookImageCache 文件头注释）。
     */
    useEffect(
        () =>
            subscribeAssetInvalidation(() => {
                void resolveImage(src, { projectDir, allowRemoteImages }).then((result) =>
                    setResolved({ src, url: result.url, missing: result.missing }),
                );
            }),
        [src, projectDir, allowRemoteImages],
    );

    const commitWidth = useCallback(
        (next: number | null) => {
            updateAttributes({ width: next });
        },
        [updateAttributes],
    );

    /**
     * 拖拽改宽。
     *
     * 拖动过程只更新本地 `dragWidth`（纯视觉预览），**松手时才提交一次**
     * `updateAttributes` —— 否则每个 pointermove 都会产生一个事务，撤销历史
     * 会被几十条"改宽度"塞满。
     *
     * 监听挂在 window 而不是把手指放在把手上：把手会随 selected 状态重渲染，
     * 指针捕获在 DOM 被替换时会丢事件。
     */
    const onResizePointerDown = useCallback(
        (event: React.PointerEvent<HTMLDivElement>, side: "left" | "right") => {
            event.preventDefault();
            event.stopPropagation();
            const element = imgRef.current;
            const startWidth = element?.getBoundingClientRect().width ?? width ?? 320;
            const startX = event.clientX;
            const initial = Math.round(startWidth);
            // 用 ref 记录拖动中的宽度，而不是从 setState 的 updater 里读 ——
            // updater 必须是纯函数（StrictMode 会调用两次，那样会提交两个事务）。
            let latest = initial;
            setDragWidth(initial);

            const onMove = (moveEvent: PointerEvent) => {
                const delta =
                    side === "right" ? moveEvent.clientX - startX : startX - moveEvent.clientX;
                latest = Math.max(MIN_WIDTH, Math.round(startWidth + delta));
                setDragWidth(latest);
            };
            const onUp = () => {
                window.removeEventListener("pointermove", onMove);
                window.removeEventListener("pointerup", onUp);
                setDragWidth(null);
                updateAttributes({ width: latest });
            };
            window.addEventListener("pointermove", onMove);
            window.addEventListener("pointerup", onUp);
        },
        [updateAttributes, width],
    );

    const assetId = assetIdFromSrc(src);
    const isAsset = isAssetRef(src);

    const menuItems: NotebookMenuItem[] = useMemo(() => {
        const items: NotebookMenuItem[] = [];
        items.push({
            key: "copy-image",
            label: t("notebook_image_copy"),
            disabled: !current.url,
            onSelect: () => {
                void copyImageToClipboard(current.url);
            },
        });
        if (isAsset && assetId) {
            items.push({
                key: "save-as",
                label: t("notebook_image_save_as"),
                onSelect: () => {
                    void notebookApi.saveAssetAs(assetId).catch(() => {});
                },
            });
        }
        items.push({
            key: "width-reset",
            label: t("notebook_image_width_reset"),
            separatorBefore: true,
            disabled: width === null,
            onSelect: () => commitWidth(null),
        });
        items.push({
            key: "alt",
            label: t("notebook_image_edit_alt"),
            onSelect: () => {
                setAltDraft(String(node.attrs.alt ?? ""));
                setEditingAlt(true);
            },
        });
        items.push({
            key: "remove",
            label: t("notebook_image_remove"),
            danger: true,
            separatorBefore: true,
            onSelect: () => deleteNode(),
        });
        return items;
    }, [assetId, commitWidth, current.url, deleteNode, isAsset, node.attrs.alt, t, width]);

    const displayWidth = dragWidth ?? width ?? null;

    return (
        <NodeViewWrapper
            className="hs-notebook-image-block"
            data-selected={selected ? "true" : "false"}
            data-dragging={dragWidth !== null ? "true" : "false"}
        >
            {current.url ? (
                <img
                    ref={imgRef}
                    src={current.url}
                    alt={String(node.attrs.alt ?? "")}
                    title={String(node.attrs.title ?? "")}
                    draggable={false}
                    className="hs-notebook-image"
                    style={displayWidth ? { width: `${displayWidth}px` } : undefined}
                    onContextMenu={(event) => {
                        event.preventDefault();
                        event.stopPropagation();
                        setMenu({ x: event.clientX, y: event.clientY });
                    }}
                    onDoubleClick={() => {
                        setAltDraft(String(node.attrs.alt ?? ""));
                        setEditingAlt(true);
                    }}
                />
            ) : (
                <div className="hs-notebook-image-missing" title={src}>
                    <span className="hs-notebook-image-missing-title">
                        {current.missing
                            ? t("notebook_image_missing")
                            : t("notebook_image_loading")}
                    </span>
                    <span className="hs-notebook-image-missing-src">{src}</span>
                    {current.missing ? (
                        <span className="hs-notebook-image-missing-hint">
                            {t("notebook_image_missing_hint")}
                        </span>
                    ) : null}
                </div>
            )}

            {selected && current.url ? (
                <>
                    <div
                        className="hs-notebook-image-handle hs-notebook-image-handle-left"
                        onPointerDown={(event) => onResizePointerDown(event, "left")}
                    />
                    <div
                        className="hs-notebook-image-handle hs-notebook-image-handle-right"
                        onPointerDown={(event) => onResizePointerDown(event, "right")}
                    />
                    {displayWidth ? (
                        <div className="hs-notebook-image-width-badge">{displayWidth}px</div>
                    ) : null}
                </>
            ) : null}

            {editingAlt ? (
                <div className="hs-notebook-image-alt-editor">
                    <input
                        autoFocus
                        value={altDraft}
                        placeholder={t("notebook_image_alt_placeholder")}
                        onChange={(event) => setAltDraft(event.target.value)}
                        onKeyDown={(event) => {
                            // 编辑器的键盘快捷键（Ctrl+B 等）不能在输入框里触发。
                            event.stopPropagation();
                            if (event.key === "Enter") {
                                updateAttributes({ alt: altDraft });
                                setEditingAlt(false);
                            } else if (event.key === "Escape") {
                                setEditingAlt(false);
                            }
                        }}
                        onBlur={() => {
                            updateAttributes({ alt: altDraft });
                            setEditingAlt(false);
                        }}
                    />
                </div>
            ) : null}

            {menu ? (
                <NotebookContextMenu
                    x={menu.x}
                    y={menu.y}
                    items={menuItems}
                    onClose={() => setMenu(null)}
                />
            ) : null}
        </NodeViewWrapper>
    );
}

/**
 * 复制图片到系统剪贴板。
 *
 * 走 `navigator.clipboard.write` 的 PNG 分支：WebView2 支持它，且这是唯一
 * 能把**位图**放进剪贴板的标准途径（`writeText` 只能放文本）。
 */
async function copyImageToClipboard(url: string | null): Promise<void> {
    if (!url) return;
    try {
        const response = await fetch(url);
        const blob = await response.blob();
        const png = blob.type === "image/png" ? blob : await transcodeToPng(blob);
        if (!png) return;
        await navigator.clipboard.write([new ClipboardItem({ "image/png": png })]);
    } catch {
        // 剪贴板被占用或 WebView 拒绝写入：静默失败，用户仍可用"另存为"。
    }
}

async function transcodeToPng(blob: Blob): Promise<Blob | null> {
    try {
        const bitmap = await createImageBitmap(blob);
        const canvas = document.createElement("canvas");
        canvas.width = bitmap.width;
        canvas.height = bitmap.height;
        const ctx = canvas.getContext("2d");
        if (!ctx) return null;
        ctx.drawImage(bitmap, 0, 0);
        bitmap.close?.();
        return await new Promise<Blob | null>((resolve) =>
            canvas.toBlob((result) => resolve(result), "image/png"),
        );
    } catch {
        return null;
    }
}
