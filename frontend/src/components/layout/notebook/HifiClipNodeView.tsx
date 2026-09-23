/*
 * 剪贴板暂存块的卡片视图。
 *
 * 这是"记事本充当中转站"的实体：一块 HiFiShifter 剪贴板数据在正文里显示为
 * 一张卡片，可以**恢复到系统剪贴板**、**直接插入时间轴**，也可以改名、复制
 * 围栏文本、删除。
 *
 * 卡片上的信息全部来自两处：围栏正文（标题/来源/数量/时长）与附件元数据
 * （示意预览）。两者都不需要读载荷字节 —— 载荷可能有几百 KB，卡片渲染不该
 * 碰它。只有在真正"恢复/插入"时才 `readAsset` 取字节。
 */

import { NodeViewWrapper, type NodeViewProps } from "@tiptap/react";
import { useMemo, useState } from "react";

import { useAppDispatch, useAppSelector } from "../../../app/hooks";
import { useI18n } from "../../../i18n/I18nProvider";
import { notebookApi } from "../../../services/api/notebook";
import { refreshAssetIndex } from "./notebookAssetIndex";
import {
    defaultClipTitle,
    formatClipDuration,
    parseHifiClipFenceBody,
    serializeHifiClipFenceBody,
    serializeHifiClipFence,
    type HifiClipBlockAttrs,
} from "./hifiClipBlock";
import { insertClipPayload, restoreClipPayload } from "./notebookClipboard";
import { NotebookContextMenu, type NotebookMenuItem } from "./NotebookContextMenu";

interface ClipPreviewRow {
    trackId: string;
    trackName: string;
    name: string;
    startSec: number;
    lengthSec: number;
}

interface ParamPreview {
    param: string;
    frameCount: number;
    sparkline: number[];
}

export function HifiClipNodeView(props: NodeViewProps) {
    const { node, updateAttributes, deleteNode, selected } = props;
    const { t } = useI18n();
    const dispatch = useAppDispatch();
    const settings = useAppSelector((state) => state.notebook.settings);
    const assetIndex = useAppSelector((state) => state.notebook.assetIndex);

    const attrs = useMemo(
        () => parseHifiClipFenceBody(String(node.attrs.body ?? "")),
        [node.attrs.body],
    );
    const [menu, setMenu] = useState<{ x: number; y: number } | null>(null);
    const [renaming, setRenaming] = useState(false);
    const [titleDraft, setTitleDraft] = useState("");
    const [busy, setBusy] = useState(false);
    const [notice, setNotice] = useState<string | null>(null);

    const entry = attrs ? assetIndex[attrs.id] : undefined;
    // 内嵌模型下"条目在"就等于"内容在"；只有 v6 旧工程迁移失败的条目
    // 会带着 hasData=false。
    const missing = attrs ? entry === undefined || !entry.hasData : false;

    // `meta` 来自工程文件（`serde_json::Value`）：可能是任意 JSON —— 手改过的
    // 工程、更早/更新版本的 schema 都会到这里。只认数组/对象，其余当没有，
    // 否则 `rows.map is not a function` 会在渲染期把整个应用打掉。
    const meta = (entry?.meta ?? null) as {
        preview?: unknown;
        param?: unknown;
    } | null;
    const previewRows: ClipPreviewRow[] = Array.isArray(meta?.preview)
        ? (meta?.preview as ClipPreviewRow[])
        : [];
    const paramPreview =
        meta?.param && typeof meta.param === "object" ? (meta.param as ParamPreview) : null;
    const sparkline = Array.isArray(paramPreview?.sparkline) ? paramPreview!.sparkline : [];

    if (!attrs) {
        // 正文被改坏（缺 id）时不该吞掉内容：原样显示，用户能自己修。
        return (
            <NodeViewWrapper className="hs-notebook-clip hs-notebook-clip-broken">
                <div className="hs-notebook-clip-broken-title">{t("notebook_clip_invalid")}</div>
                <pre className="hs-notebook-clip-broken-body">{String(node.attrs.body ?? "")}</pre>
            </NodeViewWrapper>
        );
    }

    const kindLabel = kindLabelOf(attrs.kind, t as unknown as (key: string) => string);

    async function run(
        action: () => Promise<{ ok: boolean; error?: string }>,
        successText: string,
        options?: { removeOnSuccess?: boolean },
    ) {
        setBusy(true);
        try {
            const result = await action();
            setNotice(result.ok ? successText : (result.error ?? t("notebook_clip_action_failed")));
            if (result.ok && options?.removeOnSuccess) deleteNode();
        } finally {
            setBusy(false);
            void refreshAssetIndex(dispatch);
        }
    }

    /** 设置里的"插入到"目标：选中轨道 / 新建轨道。 */
    const insertMode = settings.clipInsertMode === "newTracks" ? "newTracks" : "selected";
    /** 设置里的"插入后保留暂存块"：关掉时插入成功即移除该块。 */
    const removeAfterInsert = !settings.keepClipAfterInsert;

    const menuItems: NotebookMenuItem[] = [
        {
            key: "restore",
            label: t("notebook_clip_restore"),
            disabled: missing || busy,
            onSelect: () => {
                void run(() => restoreClipPayload(attrs.id), t("notebook_clip_restored"));
            },
        },
        {
            key: "insert",
            label: t("notebook_clip_insert_timeline"),
            disabled: missing || busy,
            onSelect: () => {
                void run(
                    () => insertClipPayload(attrs.id, insertMode),
                    t("notebook_clip_inserted"),
                    { removeOnSuccess: removeAfterInsert },
                );
            },
        },
        {
            key: "insert-new-tracks",
            label: t("notebook_clip_insert_new_tracks"),
            disabled: missing || busy,
            onSelect: () => {
                void run(
                    () => insertClipPayload(attrs.id, "newTracks"),
                    t("notebook_clip_inserted"),
                    { removeOnSuccess: removeAfterInsert },
                );
            },
        },
        {
            key: "rename",
            label: t("notebook_clip_rename"),
            separatorBefore: true,
            onSelect: () => {
                setTitleDraft(attrs.title);
                setRenaming(true);
            },
        },
        {
            key: "copy-fence",
            label: t("notebook_clip_copy_markdown"),
            onSelect: () => {
                void navigator.clipboard.writeText(serializeHifiClipFence(attrs));
                setNotice(t("notebook_clip_copied"));
            },
        },
        {
            key: "save-payload",
            label: t("notebook_clip_save_payload"),
            disabled: missing,
            onSelect: () => {
                void notebookApi
                    .saveAssetAs(attrs.id, `${attrs.title || attrs.id}.${entry?.ext ?? "hsf"}`)
                    .catch(() => {});
            },
        },
        {
            key: "remove",
            label: t("notebook_clip_remove"),
            danger: true,
            separatorBefore: true,
            onSelect: () => deleteNode(),
        },
    ];

    return (
        <NodeViewWrapper
            className="hs-notebook-clip"
            data-selected={selected ? "true" : "false"}
            data-kind={attrs.kind}
            data-missing={missing ? "true" : "false"}
        >
            <div
                className="hs-notebook-clip-head"
                onContextMenu={(event) => {
                    event.preventDefault();
                    event.stopPropagation();
                    setMenu({ x: event.clientX, y: event.clientY });
                }}
            >
                <span className={`hs-notebook-clip-badge hs-notebook-clip-badge-${attrs.kind}`}>
                    {kindLabel}
                </span>
                {renaming ? (
                    <input
                        autoFocus
                        className="hs-notebook-clip-title-input"
                        value={titleDraft}
                        onChange={(event) => setTitleDraft(event.target.value)}
                        onKeyDown={(event) => {
                            event.stopPropagation();
                            if (event.key === "Enter") {
                                commitTitle();
                            } else if (event.key === "Escape") {
                                setRenaming(false);
                            }
                        }}
                        onBlur={commitTitle}
                    />
                ) : (
                    <span
                        className="hs-notebook-clip-title"
                        title={attrs.title}
                        onDoubleClick={() => {
                            setTitleDraft(attrs.title);
                            setRenaming(true);
                        }}
                    >
                        {attrs.title || attrs.id}
                    </span>
                )}
                <button
                    type="button"
                    className="hs-notebook-clip-more"
                    onClick={(event) => {
                        const rect = (event.currentTarget as HTMLElement).getBoundingClientRect();
                        setMenu({ x: rect.left, y: rect.bottom + 2 });
                    }}
                    title={t("notebook_clip_actions")}
                >
                    ⋯
                </button>
            </div>

            <div className="hs-notebook-clip-meta">
                {attrs.kind === "param" ? (
                    <>
                        {attrs.param ? <span>{attrs.param}</span> : null}
                        {typeof attrs.frameCount === "number" ? (
                            <span>
                                {attrs.frameCount} {t("notebook_clip_unit_frames")}
                            </span>
                        ) : null}
                    </>
                ) : (
                    <>
                        <span>
                            {attrs.clipCount} {t("notebook_clip_unit_clips")}
                        </span>
                        {attrs.trackCount > 1 ? (
                            <span>
                                {attrs.trackCount} {t("notebook_clip_unit_tracks")}
                            </span>
                        ) : null}
                        <span>{formatClipDuration(attrs.durationSec)}</span>
                    </>
                )}
                {attrs.source ? (
                    <span className="hs-notebook-clip-source">
                        {t("notebook_clip_from")} {attrs.source}
                    </span>
                ) : null}
            </div>

            {settings.clipShowPreview && !missing ? (
                <div className="hs-notebook-clip-preview">
                    {attrs.kind === "param" && sparkline.length > 0 ? (
                        <Sparkline values={sparkline} />
                    ) : previewRows.length > 0 ? (
                        <ClipSchematic rows={previewRows} durationSec={attrs.durationSec} />
                    ) : null}
                </div>
            ) : null}

            {missing ? (
                <div className="hs-notebook-clip-missing">{t("notebook_clip_missing_payload")}</div>
            ) : null}

            <div className="hs-notebook-clip-actions">
                <button
                    type="button"
                    disabled={missing || busy}
                    onClick={() => void run(() => restoreClipPayload(attrs.id), t("notebook_clip_restored"))}
                >
                    {t("notebook_clip_restore")}
                </button>
                <button
                    type="button"
                    disabled={missing || busy}
                    onClick={() =>
                        void run(
                            () => insertClipPayload(attrs.id, insertMode),
                            t("notebook_clip_inserted"),
                            { removeOnSuccess: removeAfterInsert },
                        )
                    }
                >
                    {t("notebook_clip_insert_timeline")}
                </button>
                {notice ? <span className="hs-notebook-clip-notice">{notice}</span> : null}
            </div>

            {menu ? (
                <NotebookContextMenu x={menu.x} y={menu.y} items={menuItems} onClose={() => setMenu(null)} />
            ) : null}
        </NodeViewWrapper>
    );

    function commitTitle() {
        const next = titleDraft.trim();
        setRenaming(false);
        if (!attrs || next === attrs.title) return;
        const nextAttrs: HifiClipBlockAttrs = { ...attrs, title: next || defaultClipTitle(attrs) };
        updateAttributes({ body: serializeHifiClipFenceBody(nextAttrs) });
    }
}

function kindLabelOf(kind: HifiClipBlockAttrs["kind"], t: (key: string) => string): string {
    switch (kind) {
        case "tracks":
            return t("notebook_clip_kind_tracks");
        case "project":
            return t("notebook_clip_kind_project");
        case "param":
            return t("notebook_clip_kind_param");
        default:
            return t("notebook_clip_kind_clips");
    }
}

/** 参数线的迷你曲线。 */
function Sparkline({ values }: { values: number[] }) {
    const path = useMemo(() => {
        if (values.length < 2) return "";
        let min = Number.POSITIVE_INFINITY;
        let max = Number.NEGATIVE_INFINITY;
        for (const value of values) {
            if (value < min) min = value;
            if (value > max) max = value;
        }
        const span = max - min || 1;
        const step = 100 / (values.length - 1);
        return values
            .map((value, index) => {
                const x = index * step;
                const y = 30 - ((value - min) / span) * 28 - 1;
                return `${index === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
            })
            .join(" ");
    }, [values]);

    if (!path) return null;
    return (
        <svg viewBox="0 0 100 30" preserveAspectRatio="none" className="hs-notebook-sparkline">
            <path d={path} fill="none" strokeWidth="1.2" vectorEffect="non-scaling-stroke" />
        </svg>
    );
}

/** 时间轴片段的示意缩略：按轨道分行、按时间铺开的小方块。 */
function ClipSchematic({ rows, durationSec }: { rows: ClipPreviewRow[]; durationSec: number }) {
    // 用循环而不是 `Math.max(...rows.map(...))`：spread 一个很长的数组会爆栈
    // （"工程片段"载荷可以带上千个 clip），而渲染期爆栈等于整窗白屏。
    let total = durationSec;
    if (!(total > 0)) {
        total = 1;
        for (const row of rows) {
            const end = row.startSec + row.lengthSec;
            if (Number.isFinite(end) && end > total) total = end;
        }
    }
    const tracks: string[] = [];
    for (const row of rows) {
        if (!tracks.includes(row.trackId)) tracks.push(row.trackId);
    }
    const rowHeight = 8;
    const height = Math.max(1, tracks.length) * (rowHeight + 2);

    return (
        <svg
            viewBox={`0 0 100 ${height}`}
            preserveAspectRatio="none"
            className="hs-notebook-schematic"
            style={{ height: `${height}px` }}
        >
            {tracks.map((trackId, index) => (
                <line
                    key={`base-${trackId}`}
                    x1="0"
                    x2="100"
                    y1={index * (rowHeight + 2) + rowHeight / 2}
                    y2={index * (rowHeight + 2) + rowHeight / 2}
                    className="hs-notebook-schematic-rail"
                    vectorEffect="non-scaling-stroke"
                />
            ))}
            {rows.map((row, index) => {
                const trackIndex = tracks.indexOf(row.trackId);
                const x = (row.startSec / total) * 100;
                const width = Math.max(0.6, (row.lengthSec / total) * 100);
                return (
                    <rect
                        key={`clip-${index}`}
                        x={x}
                        y={trackIndex * (rowHeight + 2)}
                        width={width}
                        height={rowHeight}
                        rx="0.6"
                        className="hs-notebook-schematic-clip"
                    />
                );
            })}
        </svg>
    );
}

