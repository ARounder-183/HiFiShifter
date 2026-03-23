// VST 插件浏览器对话框
//
// 展示已扫描的 VST 插件列表，支持按名称/厂商搜索和格式过滤，
// 用户点击插件即可添加到指定轨道的 FX 链。

import { useMemo, useState } from "react";
import {
    Button,
    Dialog,
    Flex,
    Select,
    Text,
    TextField,
    Badge,
    ScrollArea,
} from "@radix-ui/themes";
import { MagnifyingGlassIcon } from "@radix-ui/react-icons";
import { useI18n } from "../../i18n/I18nProvider";
import { useAppDispatch, useAppSelector } from "../../app/hooks";
import type { RootState } from "../../app/store";
import {
    vstScanPluginsRemote,
    vstAddToChainRemote,
    vstGetTrackChainRemote,
} from "../../features/session/sessionSlice";
import type { VstPluginInfoSlim } from "../../features/session/sessionSlice";
import { VstScanPathManager } from "./VstScanPathManager";

interface Props {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    /** 目标轨道 ID，添加的插件将追加到该轨道的 FX 链 */
    trackId: string;
}

type FormatFilter = "all" | "vst2" | "vst3";

export function VstPluginBrowser({ open, onOpenChange, trackId }: Props) {
    const dispatch = useAppDispatch();
    const { t } = useI18n();
    const tAny = t as (key: string) => string;

    const plugins = useAppSelector(
        (state: RootState) => state.session.vstPlugins,
    );
    const scanning = useAppSelector(
        (state: RootState) => state.session.vstScanning,
    );

    const [search, setSearch] = useState("");
    const [formatFilter, setFormatFilter] = useState<FormatFilter>("all");
    const [pathManagerOpen, setPathManagerOpen] = useState(false);

    const filtered = useMemo(() => {
        const lowerSearch = search.toLowerCase();
        return plugins.filter((p) => {
            if (formatFilter !== "all" && p.format !== formatFilter) {
                return false;
            }
            if (lowerSearch) {
                return (
                    p.name.toLowerCase().includes(lowerSearch) ||
                    p.vendor.toLowerCase().includes(lowerSearch) ||
                    p.category.toLowerCase().includes(lowerSearch)
                );
            }
            return true;
        });
    }, [plugins, search, formatFilter]);

    function handleScan() {
        void dispatch(vstScanPluginsRemote());
    }

    async function handleAdd(plugin: VstPluginInfoSlim) {
        await dispatch(vstAddToChainRemote({ trackId, pluginUid: plugin.uid }));
        // 刷新该轨道的 FX 链
        void dispatch(vstGetTrackChainRemote(trackId));
        onOpenChange(false);
    }

    return (
        <>
        <Dialog.Root open={open} onOpenChange={onOpenChange}>
            <Dialog.Content
                style={{ maxWidth: 600, maxHeight: "80vh" }}
                onKeyDown={(e) => e.stopPropagation()}
            >
                <Dialog.Title>{tAny("vst_plugin_browser_title")}</Dialog.Title>

                <Flex direction="column" gap="3" mt="3">
                    {/* 搜索栏和过滤器 */}
                    <Flex gap="2" align="center">
                        <TextField.Root
                            size="2"
                            placeholder={tAny("vst_search_placeholder")}
                            value={search}
                            onChange={(e) => setSearch(e.target.value)}
                            style={{ flex: 1 }}
                        >
                            <TextField.Slot>
                                <MagnifyingGlassIcon height="16" width="16" />
                            </TextField.Slot>
                        </TextField.Root>
                        <Select.Root
                            size="2"
                            value={formatFilter}
                            onValueChange={(v) =>
                                setFormatFilter(v as FormatFilter)
                            }
                        >
                            <Select.Trigger
                                style={{ minWidth: 90 }}
                            />
                            <Select.Content>
                                <Select.Item value="all">All</Select.Item>
                                <Select.Item value="vst2">VST2</Select.Item>
                                <Select.Item value="vst3">VST3</Select.Item>
                            </Select.Content>
                        </Select.Root>
                        <Button
                            size="2"
                            variant="soft"
                            onClick={handleScan}
                            disabled={scanning}
                        >
                            {scanning
                                ? tAny("vst_scanning")
                                : tAny("vst_scan_plugins")}
                        </Button>
                    </Flex>

                    {/* 插件列表 */}
                    <ScrollArea
                        style={{
                            maxHeight: "50vh",
                            minHeight: 200,
                        }}
                    >
                        {filtered.length === 0 ? (
                            <Flex
                                align="center"
                                justify="center"
                                py="6"
                            >
                                <Text size="2" color="gray">
                                    {tAny("vst_no_plugins")}
                                </Text>
                            </Flex>
                        ) : (
                            <Flex direction="column" gap="1">
                                {filtered.map((plugin) => (
                                    <button
                                        key={plugin.uid}
                                        className="w-full text-left px-3 py-2 rounded hover:bg-qt-button-hover transition-colors cursor-pointer border border-transparent hover:border-qt-border"
                                        onClick={() => void handleAdd(plugin)}
                                    >
                                        <Flex
                                            justify="between"
                                            align="center"
                                        >
                                            <Flex
                                                direction="column"
                                                gap="1"
                                                className="min-w-0 flex-1"
                                            >
                                                <Flex
                                                    gap="2"
                                                    align="center"
                                                >
                                                    <Text
                                                        size="2"
                                                        weight="medium"
                                                        className="truncate"
                                                    >
                                                        {plugin.name}
                                                    </Text>
                                                    <Badge
                                                        size="1"
                                                        variant="soft"
                                                        color="gray"
                                                    >
                                                        {plugin.format.toUpperCase()}
                                                    </Badge>
                                                </Flex>
                                                <Text
                                                    size="1"
                                                    color="gray"
                                                    className="truncate"
                                                >
                                                    {plugin.vendor}
                                                    {plugin.category
                                                        ? ` · ${plugin.category}`
                                                        : ""}
                                                </Text>
                                            </Flex>
                                        </Flex>
                                    </button>
                                ))}
                            </Flex>
                        )}
                    </ScrollArea>
                </Flex>

                <Flex justify="between" gap="2" mt="4">
                    <Button
                        size="2"
                        variant="soft"
                        onClick={() => setPathManagerOpen(true)}
                    >
                        {tAny("vst_manage_scan_paths")}
                    </Button>
                    <Dialog.Close>
                        <Button variant="soft" color="gray">
                            {tAny("cancel")}
                        </Button>
                    </Dialog.Close>
                </Flex>
            </Dialog.Content>
        </Dialog.Root>

        {/* 嵌套扫描路径管理器 */}
        <VstScanPathManager
            open={pathManagerOpen}
            onOpenChange={setPathManagerOpen}
        />
        </>
    );
}
