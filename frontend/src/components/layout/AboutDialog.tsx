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
import { Button, Dialog, Flex, Text } from "@radix-ui/themes";
import { coreApi } from "../../services/api/core";
import { useI18n } from "../../i18n/I18nProvider";

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
    const { t } = useI18n();
    const tAny = t as (key: string) => string;
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
        try {
            const { openUrl } = await import("@tauri-apps/plugin-opener");
            await openUrl(url);
        } catch {
            // 打开失败不打断对话框。
        }
    }

    return (
        <Dialog.Root open={open} onOpenChange={onOpenChange}>
            <Dialog.Content style={{ maxWidth: 460 }} onKeyDown={(e) => e.stopPropagation()}>
                <Dialog.Title>{tAny("menu_about")}</Dialog.Title>
                <Dialog.Description size="2" style={{ lineHeight: 1.7 }}>
                    {tAny("about_intro")}
                </Dialog.Description>

                <Flex direction="column" gap="3" mt="4">
                    <Flex direction="column" gap="2">
                        <Flex align="center" gap="2">
                            <Text size="2" color="gray">
                                {tAny("about_version")}
                            </Text>
                            <Text size="2" weight="medium">
                                {info?.version ?? "…"}
                            </Text>
                            {info?.dirty ? (
                                <Text size="1" color="orange">
                                    {tAny("about_dirty")}
                                </Text>
                            ) : null}
                        </Flex>
                        {showCommit ? (
                            <Flex align="center" gap="2">
                                <Text size="2" color="gray">
                                    {tAny("about_commit")}
                                </Text>
                                {/* 点击跳转到该 commit 的源码快照；tooltip 展示完整链接——
                                    按自然边界拆两行，避免 320px 气泡内在连字符处断行、
                                    哈希溢出（pre-line 保留换行）。 */}
                                <button
                                    type="button"
                                    onClick={() => void openExternal(`${repoUrl}/tree/${info?.commit}`)}
                                    data-tooltip={`${repoUrl}\n/tree/${info?.commit}`}
                                    className="text-xs text-qt-accent underline underline-offset-2 hover:text-qt-text"
                                >
                                    {commitShort}
                                </button>
                            </Flex>
                        ) : null}
                    </Flex>
                </Flex>

                <Flex gap="3" mt="4" justify="end">
                    <Button
                        size="2"
                        variant="soft"
                        data-tooltip={repoUrl}
                        onClick={() => void openExternal(repoUrl)}
                    >
                        {tAny("about_open_repo")}
                    </Button>
                    <Dialog.Close>
                        <Button size="2" variant="surface" color="gray">
                            {tAny("cancel") || "Close"}
                        </Button>
                    </Dialog.Close>
                </Flex>
            </Dialog.Content>
        </Dialog.Root>
    );
}
