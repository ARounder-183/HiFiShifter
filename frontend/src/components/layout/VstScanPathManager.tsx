// VST 扫描路径管理对话框
//
// 展示用户自定义的 VST 插件扫描路径列表，支持通过文件夹选择器添加新路径，
// 以及删除已有路径。路径变更会自动持久化到配置文件。

import { useCallback, useEffect } from "react";
import {
    Button,
    Dialog,
    Flex,
    IconButton,
    Text,
    Tooltip,
    ScrollArea,
} from "@radix-ui/themes";
import { Cross2Icon, PlusIcon } from "@radix-ui/react-icons";
import { useI18n } from "../../i18n/I18nProvider";
import { useAppDispatch, useAppSelector } from "../../app/hooks";
import type { RootState } from "../../app/store";
import {
    vstListScanPathsRemote,
    vstAddScanPathRemote,
    vstRemoveScanPathRemote,
} from "../../features/session/sessionSlice";
import { fileBrowserApi } from "../../services/api/fileBrowser";

interface Props {
    open: boolean;
    onOpenChange: (open: boolean) => void;
}

export function VstScanPathManager({ open, onOpenChange }: Props) {
    const dispatch = useAppDispatch();
    const { t } = useI18n();
    const tAny = t as (key: string) => string;

    const scanPaths = useAppSelector(
        (state: RootState) => state.session.vstScanPaths,
    );

    // 面板打开时拉取最新路径列表
    useEffect(() => {
        if (open) {
            void dispatch(vstListScanPathsRemote());
        }
    }, [dispatch, open]);

    const handleAdd = useCallback(async () => {
        try {
            // Use the file browser API which calls the backend `pick_directory` command
            // to open the native system folder dialog (same as project import).
            const res = await fileBrowserApi.pickDirectory();
            // backend returns { ok: true, canceled: bool, path?: string }
            // eslint-disable-next-line no-console
            console.debug("VST: pickDirectory result:", res);
            if (
                res &&
                res.ok &&
                !res.canceled &&
                typeof res.path === "string"
            ) {
                await dispatch(vstAddScanPathRemote(res.path));
                void dispatch(vstListScanPathsRemote());
            }
        } catch (e: any) {
            // log diagnostic and show simple alert
            // eslint-disable-next-line no-console
            console.error("VST 扫描路径选择失败:", e);
            try {
                // eslint-disable-next-line no-alert
                alert(
                    tAny("vst_scan_path_select_failed") ||
                        "无法打开文件夹选择对话框，请在 Tauri 环境下运行。",
                );
            } catch {
                // ignore
            }
        }
    }, [dispatch, tAny]);

    const handleRemove = useCallback(
        async (path: string) => {
            await dispatch(vstRemoveScanPathRemote(path));
            void dispatch(vstListScanPathsRemote());
        },
        [dispatch],
    );

    return (
        <Dialog.Root open={open} onOpenChange={onOpenChange}>
            <Dialog.Content
                style={{ maxWidth: 520, maxHeight: "70vh" }}
                onKeyDown={(e) => e.stopPropagation()}
            >
                <Dialog.Title>
                    {tAny("vst_scan_path_manager_title")}
                </Dialog.Title>

                <Dialog.Description>
                    {tAny("vst_scan_path_manager_description") ||
                        "管理自定义 VST 插件扫描路径，添加或移除文件夹。"}
                </Dialog.Description>

                <Flex direction="column" gap="3" mt="3">
                    {/* 路径列表 */}
                    <ScrollArea
                        style={{
                            maxHeight: "40vh",
                            minHeight: 100,
                        }}
                    >
                        {scanPaths.length === 0 ? (
                            <Flex align="center" justify="center" py="5">
                                <Text size="2" color="gray">
                                    {tAny("vst_no_scan_paths")}
                                </Text>
                            </Flex>
                        ) : (
                            <Flex direction="column" gap="1">
                                {scanPaths.map((p) => (
                                    <Flex
                                        key={p}
                                        align="center"
                                        gap="2"
                                        px="2"
                                        py="2"
                                        className="rounded border border-qt-border"
                                        style={{
                                            backgroundColor:
                                                "var(--qt-button-hover)",
                                        }}
                                    >
                                        <Text
                                            size="2"
                                            className="flex-1 min-w-0 truncate"
                                            title={p}
                                        >
                                            {p}
                                        </Text>
                                        <Tooltip
                                            content={tAny(
                                                "vst_remove_scan_path",
                                            )}
                                        >
                                            <IconButton
                                                size="1"
                                                variant="ghost"
                                                color="red"
                                                onClick={() =>
                                                    void handleRemove(p)
                                                }
                                            >
                                                <Cross2Icon />
                                            </IconButton>
                                        </Tooltip>
                                    </Flex>
                                ))}
                            </Flex>
                        )}
                    </ScrollArea>
                </Flex>

                <Flex justify="between" gap="2" mt="4">
                    <Button
                        size="2"
                        variant="soft"
                        onClick={() => void handleAdd()}
                    >
                        <PlusIcon /> {tAny("vst_add_scan_path")}
                    </Button>
                    <Dialog.Close>
                        <Button variant="soft" color="gray">
                            {tAny("close")}
                        </Button>
                    </Dialog.Close>
                </Flex>
            </Dialog.Content>
        </Dialog.Root>
    );
}
