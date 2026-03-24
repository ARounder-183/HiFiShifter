// VST FX 链编辑器面板
//
// 以弹出对话框形式展示指定轨道的 VST 效果器链，
// 支持添加、删除、旁通、重排序和打开编辑器 GUI 等操作。

import { useCallback, useEffect, useState } from "react";
import {
    Button,
    Dialog,
    Flex,
    IconButton,
    Text,
    Badge,
    Tooltip,
} from "@radix-ui/themes";
import {
    PlusIcon,
    Cross2Icon,
    ChevronUpIcon,
    ChevronDownIcon,
    GearIcon,
} from "@radix-ui/react-icons";
import { useI18n } from "../../i18n/I18nProvider";
import { useAppDispatch, useAppSelector } from "../../app/hooks";
import type { RootState } from "../../app/store";
import {
    vstGetTrackChainRemote,
    vstRemoveFromChainRemote,
    vstSetBypassRemote,
    vstReorderChainRemote,
    vstOpenEditorRemote,
} from "../../features/session/sessionSlice";
import type { VstChainSlotSlim } from "../../features/session/sessionSlice";
import { VstPluginBrowser } from "./VstPluginBrowser";

interface Props {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    trackId: string;
    trackName: string;
}

export function VstChainEditor({
    open,
    onOpenChange,
    trackId,
    trackName,
}: Props) {
    const dispatch = useAppDispatch();
    const { t } = useI18n();
    const tAny = t as (key: string) => string;

    const chain = useAppSelector(
        (state: RootState) => state.session.vstChainByTrack[trackId] ?? [],
    );

    const [browserOpen, setBrowserOpen] = useState(false);

    // 面板打开时拉取最新 FX 链
    useEffect(() => {
        if (open && trackId) {
            void dispatch(vstGetTrackChainRemote(trackId));
        }
    }, [dispatch, open, trackId]);

    const handleRemove = useCallback(
        async (index: number) => {
            await dispatch(vstRemoveFromChainRemote({ trackId, index }));
            void dispatch(vstGetTrackChainRemote(trackId));
        },
        [dispatch, trackId],
    );

    const handleBypass = useCallback(
        async (index: number, bypassed: boolean) => {
            await dispatch(vstSetBypassRemote({ trackId, index, bypassed }));
            void dispatch(vstGetTrackChainRemote(trackId));
        },
        [dispatch, trackId],
    );

    const handleMoveUp = useCallback(
        async (index: number) => {
            if (index <= 0) return;
            await dispatch(
                vstReorderChainRemote({
                    trackId,
                    fromIndex: index,
                    toIndex: index - 1,
                }),
            );
            void dispatch(vstGetTrackChainRemote(trackId));
        },
        [dispatch, trackId],
    );

    const handleMoveDown = useCallback(
        async (index: number) => {
            if (index >= chain.length - 1) return;
            await dispatch(
                vstReorderChainRemote({
                    trackId,
                    fromIndex: index,
                    toIndex: index + 1,
                }),
            );
            void dispatch(vstGetTrackChainRemote(trackId));
        },
        [dispatch, trackId, chain.length],
    );

    const handleOpenEditor = useCallback(
        (index: number) => {
            void dispatch(vstOpenEditorRemote({ trackId, index }));
        },
        [dispatch, trackId],
    );

    return (
        <>
            <Dialog.Root open={open} onOpenChange={onOpenChange}>
                <Dialog.Content
                    style={{ maxWidth: 480, maxHeight: "70vh" }}
                    onKeyDown={(e) => e.stopPropagation()}
                >
                    <Flex justify="between" align="center">
                        <Dialog.Title style={{ margin: 0 }}>
                            {tAny("vst_chain_editor_title")} — {trackName}
                        </Dialog.Title>
                    </Flex>

                    <Flex direction="column" gap="2" mt="3">
                        {/* FX 链列表 */}
                        {chain.length === 0 ? (
                            <Flex align="center" justify="center" py="5">
                                <Text size="2" color="gray">
                                    {tAny("vst_empty_chain")}
                                </Text>
                            </Flex>
                        ) : (
                            chain.map((slot: VstChainSlotSlim) => (
                                <Flex
                                    key={slot.index}
                                    align="center"
                                    gap="2"
                                    px="2"
                                    py="2"
                                    className={`rounded border transition-colors ${
                                        slot.bypassed
                                            ? "border-qt-border opacity-50"
                                            : "border-qt-border"
                                    }`}
                                    style={{
                                        backgroundColor: slot.bypassed
                                            ? "var(--qt-base)"
                                            : "var(--qt-button-hover)",
                                    }}
                                >
                                    {/* 插件名称和格式标签 */}
                                    <Flex
                                        direction="column"
                                        gap="0"
                                        className="flex-1 min-w-0"
                                    >
                                        <Flex gap="2" align="center">
                                            <Text
                                                size="2"
                                                weight="medium"
                                                className="truncate"
                                            >
                                                {slot.pluginName ||
                                                    slot.pluginUid}
                                            </Text>
                                            <Badge
                                                size="1"
                                                variant="soft"
                                                color="gray"
                                            >
                                                {slot.format.toUpperCase()}
                                            </Badge>
                                        </Flex>
                                    </Flex>

                                    {/* 操作按钮 */}
                                    <Flex gap="1" align="center" flexShrink="0">
                                        {/* 上移 */}
                                        <Tooltip
                                            content={tAny("vst_reorder_hint")}
                                        >
                                            <IconButton
                                                size="1"
                                                variant="ghost"
                                                color="gray"
                                                disabled={slot.index <= 0}
                                                onClick={() =>
                                                    void handleMoveUp(
                                                        slot.index,
                                                    )
                                                }
                                            >
                                                <ChevronUpIcon />
                                            </IconButton>
                                        </Tooltip>

                                        {/* 下移 */}
                                        <Tooltip
                                            content={tAny("vst_reorder_hint")}
                                        >
                                            <IconButton
                                                size="1"
                                                variant="ghost"
                                                color="gray"
                                                disabled={
                                                    slot.index >=
                                                    chain.length - 1
                                                }
                                                onClick={() =>
                                                    void handleMoveDown(
                                                        slot.index,
                                                    )
                                                }
                                            >
                                                <ChevronDownIcon />
                                            </IconButton>
                                        </Tooltip>

                                        {/* 旁通开关 */}
                                        <Tooltip content={tAny("vst_bypass")}>
                                            <IconButton
                                                size="1"
                                                variant={
                                                    slot.bypassed
                                                        ? "solid"
                                                        : "ghost"
                                                }
                                                color={
                                                    slot.bypassed
                                                        ? "orange"
                                                        : "gray"
                                                }
                                                onClick={() =>
                                                    void handleBypass(
                                                        slot.index,
                                                        !slot.bypassed,
                                                    )
                                                }
                                                style={{
                                                    fontWeight: 700,
                                                    fontSize: 10,
                                                    width: 20,
                                                    height: 20,
                                                }}
                                            >
                                                B
                                            </IconButton>
                                        </Tooltip>

                                        {/* 打开编辑器 */}
                                        <Tooltip
                                            content={tAny("vst_open_editor")}
                                        >
                                            <IconButton
                                                size="1"
                                                variant="ghost"
                                                color="gray"
                                                onClick={() =>
                                                    handleOpenEditor(slot.index)
                                                }
                                            >
                                                <GearIcon />
                                            </IconButton>
                                        </Tooltip>

                                        {/* 移除 */}
                                        <Tooltip
                                            content={tAny("vst_remove_plugin")}
                                        >
                                            <IconButton
                                                size="1"
                                                variant="ghost"
                                                color="red"
                                                onClick={() =>
                                                    void handleRemove(
                                                        slot.index,
                                                    )
                                                }
                                            >
                                                <Cross2Icon />
                                            </IconButton>
                                        </Tooltip>
                                    </Flex>
                                </Flex>
                            ))
                        )}
                    </Flex>

                    <Flex justify="between" gap="2" mt="4">
                        <Button
                            size="2"
                            variant="soft"
                            onClick={() => setBrowserOpen(true)}
                        >
                            <PlusIcon /> {tAny("vst_add_plugin")}
                        </Button>
                        <Dialog.Close>
                            <Button variant="soft" color="gray">
                                {tAny("cancel")}
                            </Button>
                        </Dialog.Close>
                    </Flex>
                </Dialog.Content>
            </Dialog.Root>

            {/* 嵌套插件浏览器 */}
            <VstPluginBrowser
                open={browserOpen}
                onOpenChange={setBrowserOpen}
                trackId={trackId}
            />
        </>
    );
}
