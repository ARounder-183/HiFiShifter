/**
 * 关于对话框
 *
 * 展示项目简介、版本号与构建 Commit（点击可跳转到对应源码快照），并提供
 * 前往 GitHub 仓库的按钮。
 *
 * 数据来源（get_about_info，构建期由 build.rs 烘进二进制）：
 * - commit：非 git 构建（源码 zip 等）为 null → 不展示 Commit；
 * - repoUrl：remote.origin.url 归一化后的 GitHub 链接，上游不是 GitHub 或
 *   非 git 构建为 null → 回退到 FALLBACK_REPO_URL。
 */

import { useEffect, useState } from "react";
import { Flex } from "@radix-ui/themes";
import { coreApi } from "../../services/api/core";
import { useI18n } from "../../i18n/I18nProvider";
import { AppDialog } from "../../ui/Dialog";
import { AppForm } from "../../ui/Field";

/** 上游不是 GitHub 或读取失败时的回退仓库链接。 */
const FALLBACK_REPO_URL = "https://github.com/ARounder-183/HiFiShifter";

interface AboutDialogProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
}

interface AboutInfo {
    version: string;
    commit?: string | null;
    commitShort?: string | null;
    dirty?: boolean;
    repoUrl?: string | null;
}

export function AboutDialog({ open, onOpenChange }: AboutDialogProps) {
    const { tf } = useI18n();
    const [info, setInfo] = useState<AboutInfo | null>(null);

    useEffect(() => {
        if (!open) return;
        let cancelled = false;
        coreApi
            .getAboutInfo()
            .then((result) => {
                if (!cancelled) setInfo(result);
            })
            .catch(() => {
                if (!cancelled) setInfo(null);
            });
        return () => {
            cancelled = true;
        };
    }, [open]);

    const repoUrl = info?.repoUrl || FALLBACK_REPO_URL;
    const commitShort = info?.commitShort ?? null;
    const showCommit = Boolean(info?.commit && commitShort);

    async function openExternal(url: string) {
        if (window.__HFS_PLUGIN_BOOTSTRAP__) return;
        try {
            const { openUrl } = await import("@tauri-apps/plugin-opener");
            await openUrl(url);
        } catch {
            // 打开失败不打断对话框。
        }
    }

    return (
        <AppDialog
            open={open}
            onOpenChange={onOpenChange}
            title={tf("menu_about")}
            description={tf("about_intro")}
            size="md"
            actions={[
                {
                    id: "open-repo",
                    label: tf("about_open_repo"),
                    align: "start",
                    // 迁移时丢失：悬停显示完整仓库地址（原按钮带 data-tooltip={repoUrl}）
                    tooltip: repoUrl,
                    // 异步包装：打开仓库不关闭对话框，避免页脚表单重新提交触发默认动作。
                    onClick: () => openExternal(repoUrl),
                },
                { id: "close", label: tf("close"), onClick: () => onOpenChange(false) },
            ]}
        >
            <AppForm>
                <Flex direction="column" gap="2">
                    <Flex align="center" gap="2">
                        <span className="hs-type-muted">{tf("about_version")}</span>
                        <span className="hs-type-body font-medium">{info?.version ?? "…"}</span>
                        {info?.dirty ? (
                            <span
                                className="hs-type-caption"
                                style={{ color: "var(--qt-warning-text)" }}
                            >
                                {tf("about_dirty")}
                            </span>
                        ) : null}
                    </Flex>
                    {showCommit ? (
                        <Flex align="center" gap="2">
                            <span className="hs-type-muted">{tf("about_commit")}</span>
                            {/* 点击跳转到该 commit 的源码快照；tooltip 展示完整链接——
                                按自然边界拆两行，避免 320px 气泡内在连字符处断行、
                                哈希溢出（pre-line 保留换行）。 */}
                            <button
                                type="button"
                                onClick={() => void openExternal(`${repoUrl}/tree/${info?.commit}`)}
                                data-tooltip={`${repoUrl}\n/tree/${info?.commit}`}
                                className="text-qt-xs text-qt-accent underline underline-offset-2 hover:text-qt-text"
                            >
                                {commitShort}
                            </button>
                        </Flex>
                    ) : null}
                </Flex>
            </AppForm>
        </AppDialog>
    );
}
