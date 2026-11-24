// 工程库：移动端「打开工程」的入口。
//
// 【为什么需要一个新的表面】移动端的「保存 / 打开」此前是**不对称**的：保存
// 写死到一个固定目录，打开却要用户自己走 SAF 找一个 `.hshp`。用户看不到那个
// 固定目录，于是「我保存的工程去哪了」与「我该打开哪个文件」两个问题都无法
// 回答 —— 这不是"难用"，是**模型错了**：手机上的工程不是"一个文件"，是
// "库里的一个条目"。
//
// 【为什么用 `AppDialog` 而不是自绘底部 sheet】模态表面必须走共享壳
// （`src/ui/interactionCoverage.test.ts` 会拦手写的 `fixed inset-0`）：滚动契约、
// 焦点陷阱、Esc / 外部点击、安全区让位全部由壳负责，自绘一份必然漏掉其中几条
// —— 而漏掉的那几条正是"手机上弹窗关不掉 / 被手势条盖住"的来源。
//
// 【为什么列表不解析工程内容】20 个几 MB 的 MessagePack 工程逐个解包要 1–2 秒。
// 名字取文件名、时间取 mtime —— 三个字段全部来自目录元数据（见后端
// `project_library_entries`），20 个工程在 200ms 内出结果。
import { useCallback, useEffect, useState } from "react";

import { webApi } from "../../services/webviewApi";
import { useI18n } from "../../i18n/I18nProvider";
import { frontendLog } from "../../services/frontendLog";
import { AppDialog } from "../../ui";

interface LibraryEntry {
    path: string;
    name: string;
    modifiedAtMs: number;
    sizeBytes: number;
}

export interface ProjectLibrarySheetProps {
    open: boolean;
    onClose: () => void;
    /** 打开某个库内工程（由调用方走既有的 `openProject` 通道）。 */
    onOpenProject: (path: string) => void;
    /** 「从设备导入…」：仍然走 SAF，把外部 `.hshp` 收编进库。 */
    onImportFromDevice: () => void;
}

function formatModified(ms: number): string {
    if (!Number.isFinite(ms) || ms <= 0) return "";
    const date = new Date(ms);
    const pad = (n: number) => String(n).padStart(2, "0");
    return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

function formatSize(bytes: number): string {
    if (!Number.isFinite(bytes) || bytes <= 0) return "";
    if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function ProjectLibrarySheet({
    open,
    onClose,
    onOpenProject,
    onImportFromDevice,
}: ProjectLibrarySheetProps) {
    const { t } = useI18n();
    const [entries, setEntries] = useState<LibraryEntry[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const refresh = useCallback(async () => {
        setLoading(true);
        setError(null);
        try {
            const result = await webApi.listProjects();
            if (result?.ok === false) {
                setError(result.error ?? "list_projects failed");
                setEntries([]);
                return;
            }
            setEntries(result?.projects ?? []);
        } catch (failure) {
            frontendLog("warn", "[project-library] list failed", failure);
            setError(failure instanceof Error ? failure.message : String(failure));
            setEntries([]);
        } finally {
            setLoading(false);
        }
    }, []);

    // 每次打开都重新取：用户可能刚保存过（保存不经过本组件）。
    useEffect(() => {
        if (!open) return;
        void refresh();
    }, [open, refresh]);

    return (
        <AppDialog
            open={open}
            onOpenChange={(next) => {
                if (!next) onClose();
            }}
            title={t("mobile_project_library")}
            size="lg"
            actions={[
                {
                    id: "import",
                    label: t("mobile_project_import"),
                    onClick: () => {
                        onClose();
                        onImportFromDevice();
                    },
                },
                { id: "close", label: t("close"), align: "end", onClick: onClose },
            ]}
        >
            {loading ? (
                <div className="px-1 py-4 text-center text-qt-text/60">
                    {t("mobile_project_loading")}
                </div>
            ) : null}
            {error !== null ? (
                <div className="px-1 py-4 text-center text-qt-danger">{error}</div>
            ) : null}
            {!loading && error === null && entries.length === 0 ? (
                <div className="px-1 py-4 text-center text-qt-text/60">
                    {t("mobile_project_empty")}
                </div>
            ) : null}
            {entries.map((entry) => (
                <button
                    key={entry.path}
                    type="button"
                    className="hs-menu__item w-full"
                    onClick={() => {
                        onOpenProject(entry.path);
                        onClose();
                    }}
                >
                    <span className="flex min-w-0 flex-col items-start">
                        <span className="hs-menu__label-text truncate">{entry.name}</span>
                        <span className="hs-menu__description">
                            {formatModified(entry.modifiedAtMs)}
                            {entry.sizeBytes > 0 ? ` · ${formatSize(entry.sizeBytes)}` : ""}
                        </span>
                    </span>
                </button>
            ))}
        </AppDialog>
    );
}
