/*
 * 渲染缓存管理对话框。
 *
 * 功能：
 * - 展示缓存占用 / 条目数 / 分类占用 / 本次会话命中率
 * - 编辑渲染缓存设置（开关、容量上限、超龄清理、片段下限、单条上限、
 *   磁盘保留空间、写入模式、缓存位置、完整性校验、命中统计）
 * - 分作用域清理（全部 / 仅当前工程 / 超期 / 其它采样率）并即时反馈释放空间
 * - 在系统文件管理器中打开缓存目录
 *
 * 设计要点：
 * - 清理只删磁盘文件，不动内存缓存 —— 正在播放的内容不受影响；
 * - 「清理全部」需要二次确认（内联确认行），避免误点导致重建成本；
 * - 设置以草稿形式编辑，点「保存」才落盘（与自动备份/录音设置对话框一致）。
 */

import { useCallback, useEffect, useRef, useState, type ChangeEvent } from "react";
import {
    Button,
    Checkbox,
    Dialog,
    Flex,
    Select,
    Separator,
    Text,
    TextField,
} from "@radix-ui/themes";
import { useAppDispatch, useAppSelector } from "../../app/hooks";
import { useI18n } from "../../i18n/I18nProvider";
import {
    coreApi,
    type RenderCacheClearScope,
    type RenderCacheStats,
} from "../../services/api/core";
import {
    normalizeRenderCacheSettings,
    type RenderCacheSettings,
    type RenderCacheWriteMode,
} from "../../services/api/settings";
import { persistUiSettings } from "../../features/session/thunks/runtimeThunks";
import { setRenderCacheSettings } from "../../features/session/sessionSlice";

interface RenderCacheDialogProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
}

/** 容量预设（MB）；0 表示不限制。 */
const SIZE_PRESETS_MB = [512, 1024, 2048, 4096, 8192, 0];
/** 超龄清理预设（天）；0 表示不按时间清理。 */
const AGE_PRESETS_DAYS = [7, 30, 90, 180, 365, 0];

function formatBytes(bytes: number): string {
    if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
    const units = ["B", "KB", "MB", "GB", "TB"];
    let value = bytes;
    let unit = 0;
    while (value >= 1024 && unit < units.length - 1) {
        value /= 1024;
        unit += 1;
    }
    const digits = unit === 0 ? 0 : value >= 100 ? 0 : value >= 10 ? 1 : 2;
    return `${value.toFixed(digits)} ${units[unit]}`;
}

export function RenderCacheDialog({ open, onOpenChange }: RenderCacheDialogProps) {
    const dispatch = useAppDispatch();
    const { t } = useI18n();
    const tAny = t as (key: string) => string;
    const saved = useAppSelector((state) => state.session.renderCache);

    const [draft, setDraft] = useState<RenderCacheSettings>(saved);
    const [stats, setStats] = useState<RenderCacheStats | null>(null);
    const [loading, setLoading] = useState(false);
    const [saving, setSaving] = useState(false);
    const [busyScope, setBusyScope] = useState<RenderCacheClearScope | null>(null);
    const [pendingClearAll, setPendingClearAll] = useState(false);
    const [notice, setNotice] = useState("");
    const [errorText, setErrorText] = useState("");

    const refreshStats = useCallback(async () => {
        setLoading(true);
        try {
            const next = await coreApi.getRenderCacheStats();
            setStats(next);
        } catch {
            setErrorText(tAny("render_cache_stats_failed"));
        } finally {
            setLoading(false);
        }
    }, [tAny]);

    // 草稿只在"打开"这一时机初始化一次：保存后 Redux 中的设置会更新，若把
    // `saved` 放进依赖，effect 会立刻重跑并把"设置已保存"的提示清掉。
    const savedRef = useRef(saved);
    savedRef.current = saved;
    useEffect(() => {
        if (!open) {
            setPendingClearAll(false);
            setNotice("");
            return;
        }
        setDraft({ ...savedRef.current });
        setNotice("");
        setErrorText("");
        void refreshStats();
    }, [open, refreshStats]);

    function patch(partial: Partial<RenderCacheSettings>) {
        setDraft((prev) => ({ ...prev, ...partial }));
    }

    async function handleSave() {
        setErrorText("");
        setSaving(true);
        try {
            const normalized = normalizeRenderCacheSettings(draft);
            dispatch(setRenderCacheSettings(normalized));
            await dispatch(persistUiSettings());
            setDraft(normalized);
            setNotice(tAny("render_cache_settings_saved"));
            await refreshStats();
        } catch {
            setErrorText(tAny("render_cache_settings_save_failed"));
        } finally {
            setSaving(false);
        }
    }

    async function handleClear(scope: RenderCacheClearScope, days?: number) {
        setErrorText("");
        setNotice("");
        setPendingClearAll(false);
        setBusyScope(scope);
        try {
            const result = await coreApi.clearRenderCache(scope, days);
            if (!result?.ok) {
                setErrorText(result?.error || tAny("render_cache_clear_failed"));
                return;
            }
            setNotice(
                tAny("render_cache_cleared")
                    .replace("{n}", String(result.removedFiles ?? 0))
                    .replace("{size}", formatBytes(result.removedBytes ?? 0)),
            );
            await refreshStats();
        } catch {
            setErrorText(tAny("render_cache_clear_failed"));
        } finally {
            setBusyScope(null);
        }
    }

    async function handleOpenDir() {
        try {
            const result = await coreApi.openRenderCacheDir();
            if (!result?.ok) {
                setErrorText(result?.error || tAny("render_cache_open_dir_failed"));
            }
        } catch {
            setErrorText(tAny("render_cache_open_dir_failed"));
        }
    }

    const sessionTotal = stats ? stats.sessionHits + stats.sessionMisses : 0;
    const summaryText = stats
        ? tAny("render_cache_summary_line")
              .replace("{size}", formatBytes(stats.totalBytes))
              .replace("{n}", String(stats.entries))
              .replace("{hits}", String(stats.sessionHits))
              .replace("{total}", String(sessionTotal))
        : "";

    const agePresetValue = AGE_PRESETS_DAYS.includes(draft.maxAgeDays)
        ? String(draft.maxAgeDays)
        : "custom";
    const sizePresetValue = SIZE_PRESETS_MB.includes(draft.maxSizeMb)
        ? String(draft.maxSizeMb)
        : "custom";

    return (
        <Dialog.Root open={open} onOpenChange={onOpenChange}>
            <Dialog.Content
                style={{ maxWidth: 720 }}
                onKeyDown={(event) => event.stopPropagation()}
            >
                <Dialog.Title>{tAny("render_cache_dialog_title")}</Dialog.Title>
                <Dialog.Description>{tAny("render_cache_dialog_desc")}</Dialog.Description>

                <Flex direction="column" gap="3" mt="3">
                    {/* ── 状态 ─────────────────────────────────────────────── */}
                    <Flex direction="column" gap="1">
                        <Text size="2">{summaryText}</Text>
                        <Text size="1" color="gray" style={{ wordBreak: "break-all" }}>
                            {tAny("render_cache_location_label")}：{stats?.dir ?? "…"}
                        </Text>
                        <Flex gap="2" mt="1">
                            <Button
                                size="1"
                                variant="soft"
                                color="gray"
                                onClick={() => void handleOpenDir()}
                            >
                                {tAny("render_cache_open_dir")}
                            </Button>
                            <Button
                                size="1"
                                variant="soft"
                                color="gray"
                                disabled={loading}
                                onClick={() => void refreshStats()}
                            >
                                {tAny("render_cache_refresh")}
                            </Button>
                        </Flex>
                        {stats && !stats.writable ? (
                            <Text size="1" color="amber">
                                {tAny("render_cache_dir_not_writable")}
                            </Text>
                        ) : null}
                        {stats && stats.sessionWriteErrors > 0 ? (
                            <Text size="1" color="amber">
                                {tAny("render_cache_write_errors").replace(
                                    "{n}",
                                    String(stats.sessionWriteErrors),
                                )}
                            </Text>
                        ) : null}
                    </Flex>

                    <Separator size="4" />

                    {/* ── 开关 ─────────────────────────────────────────────── */}
                    <Flex align="center" gap="2">
                        <Checkbox
                            checked={draft.enabled}
                            onCheckedChange={(v) => patch({ enabled: Boolean(v) })}
                        />
                        <Text size="2">{tAny("render_cache_enable")}</Text>
                    </Flex>
                    <Flex align="center" gap="2">
                        <Checkbox
                            checked={draft.showHitStats}
                            onCheckedChange={(v) => patch({ showHitStats: Boolean(v) })}
                        />
                        <Text size="2">{tAny("render_cache_show_hit_stats")}</Text>
                    </Flex>

                    {/* ── 容量 ─────────────────────────────────────────────── */}
                    <Flex align="center" gap="2">
                        <Text size="2" style={{ minWidth: 132 }}>
                            {tAny("render_cache_max_size")}
                        </Text>
                        <Select.Root
                            value={sizePresetValue}
                            size="1"
                            onValueChange={(v) => {
                                if (v === "custom") return;
                                patch({ maxSizeMb: Number(v) });
                            }}
                        >
                            <Select.Trigger />
                            <Select.Content>
                                {SIZE_PRESETS_MB.map((mb) => (
                                    <Select.Item key={mb} value={String(mb)}>
                                        {mb === 0
                                            ? tAny("render_cache_unlimited")
                                            : `${mb / 1024} GB`}
                                    </Select.Item>
                                ))}
                            </Select.Content>
                        </Select.Root>
                        <TextField.Root
                            size="1"
                            type="number"
                            min={0}
                            value={String(draft.maxSizeMb)}
                            onChange={(event: ChangeEvent<HTMLInputElement>) =>
                                patch({ maxSizeMb: Number(event.target.value) })
                            }
                            style={{ width: 110 }}
                        />
                        <Text size="1" color="gray">
                            MB
                        </Text>
                    </Flex>

                    <Flex align="center" gap="2">
                        <Text size="2" style={{ minWidth: 132 }}>
                            {tAny("render_cache_max_age")}
                        </Text>
                        <Select.Root
                            value={agePresetValue}
                            size="1"
                            onValueChange={(v) => {
                                if (v === "custom") return;
                                patch({ maxAgeDays: Number(v) });
                            }}
                        >
                            <Select.Trigger />
                            <Select.Content>
                                {AGE_PRESETS_DAYS.map((days) => (
                                    <Select.Item key={days} value={String(days)}>
                                        {days === 0
                                            ? tAny("render_cache_never")
                                            : tAny("render_cache_days").replace(
                                                  "{n}",
                                                  String(days),
                                              )}
                                    </Select.Item>
                                ))}
                            </Select.Content>
                        </Select.Root>
                        <TextField.Root
                            size="1"
                            type="number"
                            min={0}
                            value={String(draft.maxAgeDays)}
                            onChange={(event: ChangeEvent<HTMLInputElement>) =>
                                patch({ maxAgeDays: Number(event.target.value) })
                            }
                            style={{ width: 90 }}
                        />
                        <Text size="1" color="gray">
                            {tAny("render_cache_days_unit")}
                        </Text>
                    </Flex>

                    <Flex align="center" gap="2" wrap="wrap">
                        <Text size="2" style={{ minWidth: 132 }}>
                            {tAny("render_cache_min_clip")}
                        </Text>
                        <TextField.Root
                            size="1"
                            type="number"
                            min={0}
                            step={0.1}
                            value={String(draft.minClipSecs)}
                            onChange={(event: ChangeEvent<HTMLInputElement>) =>
                                patch({ minClipSecs: Number(event.target.value) })
                            }
                            style={{ width: 90 }}
                        />
                        <Text size="1" color="gray">
                            {tAny("render_cache_seconds_unit")}
                        </Text>
                        <Text size="2" style={{ minWidth: 108, marginLeft: 8 }}>
                            {tAny("render_cache_max_entry")}
                        </Text>
                        <TextField.Root
                            size="1"
                            type="number"
                            min={0}
                            value={String(draft.maxEntryMb)}
                            onChange={(event: ChangeEvent<HTMLInputElement>) =>
                                patch({ maxEntryMb: Number(event.target.value) })
                            }
                            style={{ width: 90 }}
                        />
                        <Text size="1" color="gray">
                            MB
                        </Text>
                        <Text size="2" style={{ minWidth: 108, marginLeft: 8 }}>
                            {tAny("render_cache_min_free_disk")}
                        </Text>
                        <TextField.Root
                            size="1"
                            type="number"
                            min={0}
                            value={String(draft.minFreeDiskMb)}
                            onChange={(event: ChangeEvent<HTMLInputElement>) =>
                                patch({ minFreeDiskMb: Number(event.target.value) })
                            }
                            style={{ width: 90 }}
                        />
                        <Text size="1" color="gray">
                            MB
                        </Text>
                    </Flex>

                    {/* ── 高级 ─────────────────────────────────────────────── */}
                    <Flex align="center" gap="2">
                        <Text size="2" style={{ minWidth: 132 }}>
                            {tAny("render_cache_write_mode")}
                        </Text>
                        <Select.Root
                            value={draft.writeMode}
                            size="1"
                            onValueChange={(v) => patch({ writeMode: v as RenderCacheWriteMode })}
                        >
                            <Select.Trigger />
                            <Select.Content>
                                <Select.Item value="immediate">
                                    {tAny("render_cache_write_immediate")}
                                </Select.Item>
                                <Select.Item value="onExit">
                                    {tAny("render_cache_write_on_exit")}
                                </Select.Item>
                                <Select.Item value="manual">
                                    {tAny("render_cache_write_manual")}
                                </Select.Item>
                            </Select.Content>
                        </Select.Root>
                    </Flex>

                    <Flex align="center" gap="2" wrap="wrap">
                        <Text size="2" style={{ minWidth: 132 }}>
                            {tAny("render_cache_location_mode")}
                        </Text>
                        <Select.Root
                            value={draft.location}
                            size="1"
                            onValueChange={(v) =>
                                patch({ location: v === "custom" ? "custom" : "system" })
                            }
                        >
                            <Select.Trigger />
                            <Select.Content>
                                <Select.Item value="system">
                                    {tAny("render_cache_location_system")}
                                </Select.Item>
                                <Select.Item value="custom">
                                    {tAny("render_cache_location_custom")}
                                </Select.Item>
                            </Select.Content>
                        </Select.Root>
                        {draft.location === "custom" ? (
                            <TextField.Root
                                size="1"
                                placeholder={tAny("render_cache_custom_dir_placeholder")}
                                value={draft.customDir ?? ""}
                                onChange={(event: ChangeEvent<HTMLInputElement>) =>
                                    patch({ customDir: event.target.value })
                                }
                                style={{ flex: 1, minWidth: 220 }}
                            />
                        ) : null}
                    </Flex>

                    <Flex align="center" gap="2">
                        <Checkbox
                            checked={draft.verifyChecksum}
                            onCheckedChange={(v) => patch({ verifyChecksum: Boolean(v) })}
                        />
                        <Text size="2">{tAny("render_cache_verify_checksum")}</Text>
                    </Flex>

                    <Separator size="4" />

                    {/* ── 清理 ─────────────────────────────────────────────── */}
                    <Flex gap="2" wrap="wrap" align="center">
                        {pendingClearAll ? (
                            <>
                                <Text size="2" color="red">
                                    {tAny("render_cache_confirm_clear_all")}
                                </Text>
                                <Button
                                    size="1"
                                    color="red"
                                    disabled={busyScope !== null}
                                    onClick={() => void handleClear("all")}
                                >
                                    {tAny("render_cache_confirm_yes")}
                                </Button>
                                <Button
                                    size="1"
                                    variant="soft"
                                    color="gray"
                                    onClick={() => setPendingClearAll(false)}
                                >
                                    {tAny("cancel")}
                                </Button>
                            </>
                        ) : (
                            <>
                                <Button
                                    size="1"
                                    variant="soft"
                                    color="red"
                                    disabled={busyScope !== null}
                                    onClick={() => setPendingClearAll(true)}
                                >
                                    {tAny("render_cache_clear_all")}
                                </Button>
                                <Button
                                    size="1"
                                    variant="soft"
                                    color="gray"
                                    disabled={busyScope !== null}
                                    onClick={() => void handleClear("currentProject")}
                                >
                                    {tAny("render_cache_clear_project")}
                                </Button>
                                <Button
                                    size="1"
                                    variant="soft"
                                    color="gray"
                                    disabled={busyScope !== null}
                                    onClick={() =>
                                        void handleClear(
                                            "olderThan",
                                            draft.maxAgeDays > 0 ? draft.maxAgeDays : 90,
                                        )
                                    }
                                >
                                    {tAny("render_cache_clear_old")}
                                </Button>
                                <Button
                                    size="1"
                                    variant="soft"
                                    color="gray"
                                    disabled={busyScope !== null}
                                    onClick={() => void handleClear("otherSampleRates")}
                                >
                                    {tAny("render_cache_clear_other_rates")}
                                </Button>
                            </>
                        )}
                    </Flex>

                    {notice ? (
                        <Text size="2" color="green">
                            {notice}
                        </Text>
                    ) : null}
                    {errorText ? (
                        <Text size="2" color="red">
                            {errorText}
                        </Text>
                    ) : null}
                </Flex>

                <Flex justify="end" gap="2" mt="4">
                    <Button variant="soft" color="gray" onClick={() => onOpenChange(false)}>
                        {tAny("close")}
                    </Button>
                    <Button onClick={() => void handleSave()} disabled={saving}>
                        {tAny("render_cache_save_settings")}
                    </Button>
                </Flex>
            </Dialog.Content>
        </Dialog.Root>
    );
}
