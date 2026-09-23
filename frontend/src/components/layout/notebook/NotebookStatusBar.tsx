/*
 * 记事本状态栏：字数 / 图片数 / 暂存块数 / 附件占用 / 保存状态。
 *
 * 字数在富文本视图取 ProseMirror 的字符统计（与文档一致），源码视图退化为
 * 正文长度 —— 两者都不去"精确统计中文字数"，那是另一个维度的复杂度。
 */

import type { Editor } from "@tiptap/core";
import { useEffect, useState } from "react";

import { useI18n } from "../../../i18n/I18nProvider";
import { formatBytes } from "./notebookInsert";

export interface NotebookStatusBarProps {
    editor: Editor | null;
    imageCount: number;
    clipCount: number;
    totalBytes: number;
    dirty: boolean;
    /** 最近一次操作提示（暂存成功/失败等）。 */
    notice: string | null;
}

export function NotebookStatusBar({
    editor,
    imageCount,
    clipCount,
    totalBytes,
    dirty,
    notice,
}: NotebookStatusBarProps) {
    const { t } = useI18n();
    const [characters, setCharacters] = useState(0);

    useEffect(() => {
        if (!editor) return;
        const update = () => {
            const storage = editor.storage.characterCount as { characters?: () => number } | undefined;
            setCharacters(storage?.characters?.() ?? editor.state.doc.textContent.length);
        };
        update();
        editor.on("update", update);
        return () => {
            editor.off("update", update);
        };
    }, [editor]);

    return (
        <div className="hs-notebook-status">
            <div className="hs-notebook-status-part">
                <span>
                    {characters} {t("notebook_status_characters")}
                </span>
                {imageCount > 0 ? (
                    <span>
                        {imageCount} {t("notebook_status_images")}
                    </span>
                ) : null}
                {clipCount > 0 ? (
                    <span>
                        {clipCount} {t("notebook_status_clips")}
                    </span>
                ) : null}
                {totalBytes > 0 ? <span>{formatBytes(totalBytes)}</span> : null}
            </div>
            <div className="hs-notebook-status-part">
                {notice ? <span>{notice}</span> : null}
                <span>{dirty ? t("notebook_status_unsaved") : t("notebook_status_saved")}</span>
            </div>
        </div>
    );
}
