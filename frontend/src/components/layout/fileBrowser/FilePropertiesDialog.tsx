/**
 * 文件属性对话框。
 *
 * 【为什么值得有】DAW 里"这个 wav 是多少 kHz / 几声道 / 多长"是导入前的常规问题；
 * 不去系统文件管理器看属性、也不试听一遍就能读到，是最省事的路径。音频部分复用
 * 后端已有的 `get_audio_file_info`（与试听同一条探测通道），不为它新增后端能力。
 *
 * 【为什么按需拉取】属性只在打开对话框时算一次：`get_audio_file_info` 要探测文件
 * 头，放在列表渲染路径上会给每次滚动都加一次 I/O。
 */

import { useEffect, useState } from "react";

import { useI18n } from "../../../i18n/I18nProvider";
import { fileBrowserApi, type FileEntry } from "../../../services/api/fileBrowser";
import { AppDialog } from "../../../ui";
import {
    isAudioFile,
    isMidiFile,
    isProjectFile,
    isVideoFile,
} from "../../../features/fileBrowser/fileKinds";
import { formatModified, formatSize } from "./formatFile";
import type { MessageKey } from "../../../i18n/messages";

export interface FilePropertiesDialogProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    entry: FileEntry | null;
}

/** 一行"标签 / 值"。 */
function Row({ label, value }: { label: string; value: string }) {
    return (
        <div className="flex gap-2 py-qt-1">
            <span className="hs-type-label shrink-0 text-qt-text-muted" style={{ width: 96 }}>
                {label}
            </span>
            <span className="hs-type-body min-w-0 flex-1 break-all" data-hs-selectable="true">
                {value}
            </span>
        </div>
    );
}

/** 文件类型的中文/本地化名称键。 */
function kindKey(entry: FileEntry): MessageKey {
    if (entry.isDir) return "fb_prop_kind_folder";
    if (isVideoFile(entry)) return "fb_prop_kind_video";
    if (isAudioFile(entry)) return "fb_prop_kind_audio";
    if (isMidiFile(entry)) return "fb_prop_kind_midi";
    if (isProjectFile(entry)) return "fb_prop_kind_project";
    return "fb_prop_kind_file";
}

/** 取所在目录（末尾不带分隔符）。 */
function parentOf(path: string): string {
    const normalized = path.replace(/\\/g, "/");
    const cut = normalized.replace(/\/+$/, "").lastIndexOf("/");
    if (cut < 0) return path;
    const parent = normalized.slice(0, cut);
    // Windows 盘符根：`C:/a.wav` → `C:` 要还原成 `C:\`，否则显示成裸盘符。
    if (/^[A-Za-z]:$/.test(parent)) {
        return path.includes("\\") ? `${parent}\\` : `${parent}/`;
    }
    return path.includes("\\") ? parent.replace(/\//g, "\\") : parent;
}

export function FilePropertiesDialog({ open, onOpenChange, entry }: FilePropertiesDialogProps) {
    const { t } = useI18n();
    const [audioInfo, setAudioInfo] = useState<{
        sampleRate: number;
        channels: number;
        durationSec: number;
    } | null>(null);
    /** 目录内的条目数；`null` 表示还没算出来（或不是目录）。 */
    const [folderCount, setFolderCount] = useState<number | null>(null);

    useEffect(() => {
        // 关闭状态不探测。调用方按 `entry.path` 给本组件设了 key，换条目即重新挂载、
        // 状态天然是新的 —— 因此这里**不需要**在关闭时把 state 清回去（在 effect 里
        // 同步 setState 会触发级联渲染，也是 React 明确不建议的写法）。
        if (!open || !entry) return;
        let cancelled = false;
        if (entry.isDir) {
            fileBrowserApi
                .listDirectory(entry.path)
                .then((items) => {
                    if (!cancelled) setFolderCount(items.length);
                })
                .catch(() => {
                    /* 目录读不到（权限等）就不显示这一行 */
                });
        } else if (isAudioFile(entry)) {
            fileBrowserApi
                .getAudioFileInfo(entry.path)
                .then((info) => {
                    if (!cancelled) setAudioInfo(info);
                })
                .catch(() => {
                    /* 探测失败（编码不支持等）就不显示音频那几行 */
                });
        }
        return () => {
            cancelled = true;
        };
    }, [open, entry]);

    const duration = audioInfo
        ? `${Math.floor(audioInfo.durationSec / 60)}:${String(
              Math.floor(audioInfo.durationSec % 60),
          ).padStart(2, "0")}`
        : "";

    return (
        <AppDialog
            open={open}
            onOpenChange={onOpenChange}
            title={t("fb_properties_title")}
            size="sm"
            actions={[
                {
                    id: "close",
                    label: t("close"),
                    intent: "primary",
                    onClick: () => onOpenChange(false),
                },
            ]}
        >
            {entry && (
                <div className="flex flex-col">
                    <Row label={t("fb_prop_name")} value={entry.name} />
                    <Row label={t("fb_prop_type")} value={t(kindKey(entry))} />
                    <Row label={t("fb_prop_location")} value={parentOf(entry.path)} />
                    {entry.isDir ? (
                        folderCount != null && (
                            <Row label={t("fb_prop_items")} value={String(folderCount)} />
                        )
                    ) : (
                        <Row label={t("fb_prop_size")} value={formatSize(entry.size)} />
                    )}
                    {audioInfo && (
                        <>
                            <Row label={t("fb_prop_duration")} value={duration} />
                            <Row
                                label={t("fb_prop_sample_rate")}
                                value={`${(audioInfo.sampleRate / 1000).toFixed(1)} kHz`}
                            />
                            <Row label={t("fb_prop_channels")} value={String(audioInfo.channels)} />
                        </>
                    )}
                    <Row label={t("fb_prop_modified")} value={formatModified(entry.modifiedTime)} />
                    <Row label={t("fb_prop_path")} value={entry.path} />
                </div>
            )}
        </AppDialog>
    );
}
