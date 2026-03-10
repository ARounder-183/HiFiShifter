import React, { useCallback, useEffect, useState } from "react";
import {
    Dialog,
    Flex,
    Text,
    Button,
    IconButton,
    ScrollArea,
    Separator,
} from "@radix-ui/themes";
import { Cross2Icon, PlusIcon, TrashIcon } from "@radix-ui/react-icons";
import { useI18n } from "../../i18n/I18nProvider";
import {
    listResamplers,
    browseResamplerExe,
    addResampler,
    removeResampler,
    updateResampler,
    type ResamplerEntry,
    type FlagParam,
} from "../../services/api/resampler";

interface ResamplerManagerDialogProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
}

/**
 * 外部 Resampler 管理面板
 * 提供注册/删除 Resampler、编辑 Extra Flags、管理 Flag 参数等功能。
 */
export const ResamplerManagerDialog: React.FC<ResamplerManagerDialogProps> = ({
    open,
    onOpenChange,
}) => {
    const { t } = useI18n();
    const tAny = t as (key: string) => string;

    // ─── Resampler 列表 ────────────────────────────────────────
    const [resamplerList, setResamplerList] = useState<ResamplerEntry[]>([]);

    // 打开时刷新列表
    useEffect(() => {
        if (open) {
            void listResamplers().then(setResamplerList).catch(() => {});
        }
    }, [open]);

    // ─── 添加 Flag 参数弹窗状态 ────────────────────────────────
    const [addFlagForId, setAddFlagForId] = useState<string | null>(null);
    const [newFlagKey, setNewFlagKey] = useState("");
    const [newFlagName, setNewFlagName] = useState("");
    const [newFlagMin, setNewFlagMin] = useState("-100");
    const [newFlagMax, setNewFlagMax] = useState("100");
    const [newFlagDefault, setNewFlagDefault] = useState("0");

    /** 打开"添加 Flag 参数"表单 */
    const openAddFlagForm = useCallback((resamplerEntryId: string) => {
        setAddFlagForId(resamplerEntryId);
        setNewFlagKey("");
        setNewFlagName("");
        setNewFlagMin("-100");
        setNewFlagMax("100");
        setNewFlagDefault("0");
    }, []);

    /** 关闭"添加 Flag 参数"表单 */
    const closeAddFlagForm = useCallback(() => {
        setAddFlagForId(null);
    }, []);

    // ─── 处理函数 ──────────────────────────────────────────────

    /** 添加新 Resampler */
    const handleAddResampler = useCallback(async () => {
        const path = await browseResamplerExe();
        if (!path) return;
        const name =
            path
                .split(/[\\/]/)
                .pop()
                ?.replace(/\.exe$/i, "") ?? "Resampler";
        const entry = await addResampler(name, path);
        if (entry) {
            setResamplerList((prev) => [...prev, entry]);
        }
    }, []);

    /** 删除 Resampler */
    const handleRemoveResampler = useCallback(async (id: string) => {
        const ok = await removeResampler(id);
        if (ok) {
            setResamplerList((prev) => prev.filter((r) => r.id !== id));
        }
    }, []);

    /** 更新 Extra Flags */
    const handleUpdateFlags = useCallback(
        async (id: string, newFlags: string) => {
            const updated = await updateResampler(id, {
                defaultFlags: newFlags,
            });
            if (updated) {
                setResamplerList((prev) =>
                    prev.map((r) =>
                        r.id === id ? { ...r, defaultFlags: newFlags } : r,
                    ),
                );
            }
        },
        [],
    );

    /** 删除某个 Flag 参数 */
    const handleRemoveFlagParam = useCallback(
        async (resamplerEntryId: string, flagKey: string) => {
            const rs = resamplerList.find((r) => r.id === resamplerEntryId);
            if (!rs) return;
            const newParams = rs.flagParams.filter((p) => p.key !== flagKey);
            const updated = await updateResampler(resamplerEntryId, {
                flagParams: newParams,
            });
            if (updated) {
                setResamplerList((prev) =>
                    prev.map((r) =>
                        r.id === resamplerEntryId
                            ? { ...r, flagParams: newParams }
                            : r,
                    ),
                );
            }
        },
        [resamplerList],
    );

    /** 确认添加 Flag 参数 */
    const handleConfirmAddFlag = useCallback(async () => {
        if (!addFlagForId || !newFlagKey.trim()) return;
        const rs = resamplerList.find((r) => r.id === addFlagForId);
        if (!rs) return;
        const existingParams = rs.flagParams ?? [];
        // 跳过重复 key
        if (existingParams.some((p) => p.key === newFlagKey.trim())) {
            closeAddFlagForm();
            return;
        }
        const newFp: FlagParam = {
            key: newFlagKey.trim(),
            displayName: newFlagName.trim() || newFlagKey.trim(),
            minValue: parseFloat(newFlagMin) || -100,
            maxValue: parseFloat(newFlagMax) || 100,
            defaultValue: parseFloat(newFlagDefault) || 0,
        };
        const newParams = [...existingParams, newFp];
        const updated = await updateResampler(addFlagForId, {
            flagParams: newParams,
        });
        if (updated) {
            setResamplerList((prev) =>
                prev.map((r) =>
                    r.id === addFlagForId
                        ? { ...r, flagParams: newParams }
                        : r,
                ),
            );
        }
        closeAddFlagForm();
    }, [
        addFlagForId,
        newFlagKey,
        newFlagName,
        newFlagMin,
        newFlagMax,
        newFlagDefault,
        resamplerList,
        closeAddFlagForm,
    ]);

    return (
        <Dialog.Root open={open} onOpenChange={onOpenChange}>
            <Dialog.Content style={{ maxWidth: 620, maxHeight: "80vh" }}>
                <Dialog.Title>
                    {tAny("resampler_dialog_title")}
                </Dialog.Title>
                <Dialog.Description size="2" color="gray">
                    {tAny("resampler_dialog_desc")}
                </Dialog.Description>

                <ScrollArea
                    style={{ maxHeight: "calc(80vh - 160px)" }}
                    scrollbars="vertical"
                >
                    <Flex direction="column" gap="3" py="3">
                        {resamplerList.length === 0 && (
                            <Text size="2" color="gray" align="center">
                                {tAny("resampler_empty")}
                            </Text>
                        )}

                        {resamplerList.map((rs) => (
                            <Flex
                                key={rs.id}
                                direction="column"
                                gap="2"
                                p="3"
                                style={{
                                    borderRadius: 8,
                                    border: "1px solid var(--gray-6)",
                                    background: "var(--gray-2)",
                                }}
                            >
                                {/* 标题行：名称 + 路径 + 可用性 + 删除按钮 */}
                                <Flex
                                    align="center"
                                    justify="between"
                                    gap="2"
                                >
                                    <Flex align="center" gap="2">
                                        <Text size="2" weight="bold">
                                            📦 {rs.displayName}
                                        </Text>
                                        {!rs.available && (
                                            <Text size="1" color="red">
                                                ⚠️ {tAny("resampler_unavailable")}
                                            </Text>
                                        )}
                                    </Flex>
                                    <IconButton
                                        size="1"
                                        variant="ghost"
                                        color="red"
                                        title={tAny("remove_resampler")}
                                        onClick={() =>
                                            void handleRemoveResampler(rs.id)
                                        }
                                        style={{ cursor: "pointer" }}
                                    >
                                        <TrashIcon />
                                    </IconButton>
                                </Flex>

                                {/* 路径 */}
                                <Text
                                    size="1"
                                    color="gray"
                                    style={{
                                        wordBreak: "break-all",
                                        fontFamily: "monospace",
                                    }}
                                >
                                    {rs.exePath}
                                </Text>

                                <Separator size="4" />

                                {/* Extra Flags */}
                                <Flex align="center" gap="2">
                                    <Text size="1" weight="medium">
                                        {tAny("resampler_extra_flags")}
                                    </Text>
                                    <input
                                        type="text"
                                        defaultValue={rs.defaultFlags}
                                        placeholder="Mt..."
                                        onBlur={(e) =>
                                            void handleUpdateFlags(
                                                rs.id,
                                                e.target.value,
                                            )
                                        }
                                        onKeyDown={(e) => {
                                            if (e.key === "Enter")
                                                (
                                                    e.target as HTMLInputElement
                                                ).blur();
                                        }}
                                        className="bg-transparent border border-gray-600 rounded px-1.5 py-0.5 text-xs text-gray-200 w-[120px] focus:border-blue-400 focus:outline-none"
                                    />
                                </Flex>

                                {/* Flag 参数列表 */}
                                {(rs.flagParams ?? []).length > 0 && (
                                    <Flex direction="column" gap="1">
                                        <Text
                                            size="1"
                                            weight="medium"
                                            color="gray"
                                        >
                                            {tAny("resampler_flag_params")}
                                        </Text>
                                        {rs.flagParams.map((fp) => (
                                            <Flex
                                                key={fp.key}
                                                align="center"
                                                gap="2"
                                                px="2"
                                                py="1"
                                                style={{
                                                    borderRadius: 4,
                                                    background: "var(--gray-3)",
                                                }}
                                            >
                                                <Text
                                                    size="1"
                                                    weight="bold"
                                                    style={{
                                                        fontFamily: "monospace",
                                                        minWidth: 28,
                                                    }}
                                                >
                                                    {fp.key}
                                                </Text>
                                                <Text size="1" color="gray">
                                                    {fp.displayName !== fp.key
                                                        ? fp.displayName
                                                        : ""}
                                                </Text>
                                                <Text
                                                    size="1"
                                                    color="gray"
                                                    style={{ marginLeft: "auto" }}
                                                >
                                                    {fp.minValue} ~ {fp.maxValue}
                                                </Text>
                                                <Text size="1" color="gray">
                                                    {tAny("resampler_flag_default")}:{" "}
                                                    {fp.defaultValue}
                                                </Text>
                                                <IconButton
                                                    size="1"
                                                    variant="ghost"
                                                    color="red"
                                                    title={`${tAny("resampler_flag_remove")} ${fp.key}`}
                                                    onClick={() =>
                                                        void handleRemoveFlagParam(
                                                            rs.id,
                                                            fp.key,
                                                        )
                                                    }
                                                    style={{
                                                        cursor: "pointer",
                                                    }}
                                                >
                                                    <Cross2Icon />
                                                </IconButton>
                                            </Flex>
                                        ))}
                                    </Flex>
                                )}

                                {/* 添加 Flag 参数按钮 / 表单 */}
                                {addFlagForId === rs.id ? (
                                    <Flex
                                        direction="column"
                                        gap="2"
                                        p="2"
                                        style={{
                                            borderRadius: 6,
                                            border: "1px solid var(--blue-6)",
                                            background: "var(--gray-3)",
                                        }}
                                    >
                                        <Flex align="center" gap="2">
                                            <Text size="1">
                                                {tAny("resampler_flag_letter")}
                                            </Text>
                                            <input
                                                type="text"
                                                value={newFlagKey}
                                                onChange={(e) =>
                                                    setNewFlagKey(
                                                        e.target.value,
                                                    )
                                                }
                                                placeholder="B"
                                                maxLength={4}
                                                className="bg-transparent border border-gray-600 rounded px-1 py-0.5 text-xs text-gray-200 w-[40px] focus:border-blue-400 focus:outline-none"
                                            />
                                            <Text size="1">
                                                {tAny("resampler_flag_name")}
                                            </Text>
                                            <input
                                                type="text"
                                                value={newFlagName}
                                                onChange={(e) =>
                                                    setNewFlagName(
                                                        e.target.value,
                                                    )
                                                }
                                                placeholder="气声"
                                                className="bg-transparent border border-gray-600 rounded px-1 py-0.5 text-xs text-gray-200 w-[80px] focus:border-blue-400 focus:outline-none"
                                            />
                                        </Flex>
                                        <Flex align="center" gap="2">
                                            <Text size="1">
                                                {tAny("resampler_flag_range")}
                                            </Text>
                                            <input
                                                type="number"
                                                value={newFlagMin}
                                                onChange={(e) =>
                                                    setNewFlagMin(
                                                        e.target.value,
                                                    )
                                                }
                                                className="bg-transparent border border-gray-600 rounded px-1 py-0.5 text-xs text-gray-200 w-[55px] focus:border-blue-400 focus:outline-none"
                                            />
                                            <Text size="1">~</Text>
                                            <input
                                                type="number"
                                                value={newFlagMax}
                                                onChange={(e) =>
                                                    setNewFlagMax(
                                                        e.target.value,
                                                    )
                                                }
                                                className="bg-transparent border border-gray-600 rounded px-1 py-0.5 text-xs text-gray-200 w-[55px] focus:border-blue-400 focus:outline-none"
                                            />
                                            <Text size="1">
                                                {tAny("resampler_flag_default")}
                                            </Text>
                                            <input
                                                type="number"
                                                value={newFlagDefault}
                                                onChange={(e) =>
                                                    setNewFlagDefault(
                                                        e.target.value,
                                                    )
                                                }
                                                className="bg-transparent border border-gray-600 rounded px-1 py-0.5 text-xs text-gray-200 w-[55px] focus:border-blue-400 focus:outline-none"
                                            />
                                        </Flex>
                                        <Flex gap="2" justify="end">
                                            <Button
                                                size="1"
                                                variant="soft"
                                                color="gray"
                                                onClick={closeAddFlagForm}
                                                style={{ cursor: "pointer" }}
                                            >
                                                {tAny("cancel")}
                                            </Button>
                                            <Button
                                                size="1"
                                                variant="solid"
                                                color="blue"
                                                disabled={
                                                    !newFlagKey.trim()
                                                }
                                                onClick={() =>
                                                    void handleConfirmAddFlag()
                                                }
                                                style={{ cursor: "pointer" }}
                                            >
                                                {tAny("confirm")}
                                            </Button>
                                        </Flex>
                                    </Flex>
                                ) : (
                                    <Button
                                        size="1"
                                        variant="soft"
                                        color="blue"
                                        onClick={() =>
                                            openAddFlagForm(rs.id)
                                        }
                                        style={{ cursor: "pointer" }}
                                    >
                                        <PlusIcon />
                                        {tAny("add_flag_param")}
                                    </Button>
                                )}
                            </Flex>
                        ))}
                    </Flex>
                </ScrollArea>

                {/* 底部按钮 */}
                <Flex justify="between" align="center" pt="3">
                    <Button
                        variant="soft"
                        color="blue"
                        size="2"
                        onClick={() => void handleAddResampler()}
                        style={{ cursor: "pointer" }}
                    >
                        <PlusIcon />
                        {tAny("add_resampler")}
                    </Button>
                    <Dialog.Close>
                        <Button variant="soft" color="gray" size="2">
                            <Cross2Icon />
                            {tAny("resampler_close")}
                        </Button>
                    </Dialog.Close>
                </Flex>
            </Dialog.Content>
        </Dialog.Root>
    );
};
