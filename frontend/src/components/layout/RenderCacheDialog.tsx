/*
 * 渲染缓存管理对话框。
 *
 * 功能：
 * - 展示缓存占用 / 条目数 / 分类占用 / 本次会话命中率与**未落盘原因**
 * - 编辑渲染缓存设置（开关、容量上限、超龄清理、片段时长下限、片段大小下限、
 *   单条上限、磁盘保留空间、写入模式、缓存位置、完整性校验、命中统计）
 * - 分作用域清理（全部 / 仅当前工程 / 超期 / 其它采样率）并即时反馈释放空间
 * - 在系统文件管理器中打开缓存目录
 *
 * 设计要点：
 * - 清理只删磁盘文件，不动内存缓存 —— 正在播放的内容不受影响；
 * - 「清理全部」需要二次确认（内联确认行），避免误点导致重建成本；
 * - 设置以草稿形式编辑，点「保存」才落盘（与自动备份/录音设置对话框一致）。
 */

import { useCallback, useEffect, useRef, useState, type ChangeEvent } from "react";
import { Checkbox, Flex, Separator, TextField } from "@radix-ui/themes";
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
import { AppDialog } from "../../ui/Dialog";
import { AppField, AppForm } from "../../ui/Field";
import { AppButton, AppNumberField, AppSelect } from "../../ui";

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
    const { tf } = useI18n();
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
            setErrorText(tf("render_cache_stats_failed"));
        } finally {
            setLoading(false);
        }
    }, [tf]);

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
            setNotice(tf("render_cache_settings_saved"));
            await refreshStats();
        } catch {
            setErrorText(tf("render_cache_settings_save_failed"));
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
                setErrorText(result?.error || tf("render_cache_clear_failed"));
                return;
            }
            setNotice(
                tf("render_cache_cleared")
                    .replace("{n}", String(result.removedFiles ?? 0))
                    .replace("{size}", formatBytes(result.removedBytes ?? 0)),
            );
            await refreshStats();
        } catch {
            setErrorText(tf("render_cache_clear_failed"));
        } finally {
            setBusyScope(null);
        }
    }

    async function handleOpenDir() {
        try {
            const result = await coreApi.openRenderCacheDir();
            if (!result?.ok) {
                setErrorText(result?.error || tf("render_cache_open_dir_failed"));
            }
        } catch {
            setErrorText(tf("render_cache_open_dir_failed"));
        }
    }

    const sessionTotal = stats ? stats.sessionHits + stats.sessionMisses : 0;
    const summaryText = stats
        ? tf("render_cache_summary_line")
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
        <AppDialog
            open={open}
            onOpenChange={onOpenChange}
            title={tf("render_cache_dialog_title")}
            description={tf("render_cache_dialog_desc")}
            size="lg"
            actions={[
                { id: "close", label: tf("close"), onClick: () => onOpenChange(false) },
                {
                    id: "save",
                    label: tf("render_cache_save_settings"),
                    intent: "primary",
                    disabled: saving,
                    onClick: handleSave,
                },
            ]}
        >
            <AppForm labelWidth="lg">
                {/* ── 状态 ─────────────────────────────────────────────── */}
                <Flex direction="column" gap="1">
                    <span className="hs-type-body">{summaryText}</span>
                    <span className="hs-type-caption" style={{ wordBreak: "break-all" }}>
                        {tf("render_cache_location_label")}：{stats?.dir ?? "…"}
                    </span>
                    <Flex gap="2" mt="1">
                        <AppButton size="sm" onClick={() => void handleOpenDir()}>
                            {tf("render_cache_open_dir")}
                        </AppButton>
                        <AppButton size="sm" disabled={loading} onClick={() => void refreshStats()}>
                            {tf("render_cache_refresh")}
                        </AppButton>
                    </Flex>
                    {stats && !stats.writable ? (
                        <span
                            className="hs-type-caption"
                            style={{ color: "var(--qt-warning-text)" }}
                        >
                            {tf("render_cache_dir_not_writable")}
                        </span>
                    ) : null}
                    {stats && stats.sessionWriteErrors > 0 ? (
                        <span
                            className="hs-type-caption"
                            style={{ color: "var(--qt-warning-text)" }}
                        >
                            {tf("render_cache_write_errors").replace(
                                "{n}",
                                String(stats.sessionWriteErrors),
                            )}
                        </span>
                    ) : null}
                </Flex>

                <Separator size="4" />

                {/* ── 开关 ─────────────────────────────────────────────── */}
                <Flex align="center" gap="2">
                    <Checkbox
                        checked={draft.enabled}
                        onCheckedChange={(v) => patch({ enabled: Boolean(v) })}
                    />
                    <span className="hs-type-label">{tf("render_cache_enable")}</span>
                </Flex>
                <Flex align="center" gap="2">
                    <Checkbox
                        checked={draft.showHitStats}
                        onCheckedChange={(v) => patch({ showHitStats: Boolean(v) })}
                    />
                    <span className="hs-type-label">{tf("render_cache_show_hit_stats")}</span>
                </Flex>
                <Flex align="center" gap="2">
                    <Checkbox
                        checked={draft.exportReuseEnabled}
                        onCheckedChange={(v) => patch({ exportReuseEnabled: Boolean(v) })}
                    />
                    <span className="hs-type-label">{tf("render_cache_export_reuse")}</span>
                </Flex>

                {/* ── 容量 ─────────────────────────────────────────────── */}
                <AppField label={tf("render_cache_max_size")}>
                    <Flex align="center" gap="2">
                        <AppSelect
                            fullWidth={false}
                            // 旧写法是 size="1"（24px）：对话框里也要紧凑
                            density="compact"
                            value={sizePresetValue}
                            /*
                             * 关掉滚轮：本预设列表**末项是 0（不限）**，滚轮滚下去会把
                             * 4096 MB 直接变成"不限"。滚轮留给右侧数字框（±1 + 精细调整）。
                             */
                            wheelAdjust={false}
                            onValueChange={(v) => {
                                if (v === "custom") return;
                                patch({ maxSizeMb: Number(v) });
                            }}
                            options={SIZE_PRESETS_MB.map((mb) => ({
                                value: String(mb),
                                label: mb === 0 ? tf("render_cache_unlimited") : `${mb / 1024} GB`,
                            }))}
                        />
                        <AppNumberField
                            value={draft.maxSizeMb}
                            unit="integer"
                            min={0}
                            width={110}
                            suffix="MB"
                            ariaLabel={tf("render_cache_max_size")}
                            onCommit={(maxSizeMb) => patch({ maxSizeMb })}
                        />
                    </Flex>
                </AppField>

                <AppField label={tf("render_cache_max_age")}>
                    <Flex align="center" gap="2">
                        <AppSelect
                            fullWidth={false}
                            // 旧写法是 size="1"（24px）：对话框里也要紧凑
                            density="compact"
                            value={agePresetValue}
                            // 同上：末项是 0（永不清理），滚轮不得把它选中。
                            wheelAdjust={false}
                            onValueChange={(v) => {
                                if (v === "custom") return;
                                patch({ maxAgeDays: Number(v) });
                            }}
                            options={AGE_PRESETS_DAYS.map((days) => ({
                                value: String(days),
                                label:
                                    days === 0
                                        ? tf("render_cache_never")
                                        : tf("render_cache_days").replace("{n}", String(days)),
                            }))}
                        />
                        <AppNumberField
                            value={draft.maxAgeDays}
                            unit="integer"
                            min={0}
                            width={90}
                            suffix={tf("render_cache_days_unit")}
                            ariaLabel={tf("render_cache_max_age")}
                            onCommit={(maxAgeDays) => patch({ maxAgeDays })}
                        />
                    </Flex>
                </AppField>

                <Flex align="center" gap="2" wrap="wrap">
                    <span className="hs-type-label shrink-0" style={{ minWidth: 132 }}>
                        {tf("render_cache_min_clip")}
                    </span>
                    <AppNumberField
                        value={draft.minClipSecs}
                        unit="seconds"
                        min={0}
                        width={90}
                        suffix={tf("render_cache_seconds_unit")}
                        ariaLabel={tf("render_cache_min_clip")}
                        onCommit={(minClipSecs) => patch({ minClipSecs })}
                    />
                    <span
                        className="hs-type-label shrink-0"
                        style={{ minWidth: 108, marginLeft: 8 }}
                    >
                        {tf("render_cache_min_entry")}
                    </span>
                    <AppNumberField
                        value={draft.minEntryKb}
                        unit="integer"
                        min={0}
                        width={90}
                        suffix="KB"
                        ariaLabel={tf("render_cache_min_entry")}
                        onCommit={(minEntryKb) => patch({ minEntryKb })}
                    />
                    <span
                        className="hs-type-label shrink-0"
                        style={{ minWidth: 108, marginLeft: 8 }}
                    >
                        {tf("render_cache_max_entry")}
                    </span>
                    <AppNumberField
                        value={draft.maxEntryMb}
                        unit="integer"
                        min={0}
                        width={90}
                        suffix="MB"
                        ariaLabel={tf("render_cache_max_entry")}
                        onCommit={(maxEntryMb) => patch({ maxEntryMb })}
                    />
                    <span
                        className="hs-type-label shrink-0"
                        style={{ minWidth: 108, marginLeft: 8 }}
                    >
                        {tf("render_cache_min_free_disk")}
                    </span>
                    <AppNumberField
                        value={draft.minFreeDiskMb}
                        unit="integer"
                        min={0}
                        width={90}
                        suffix="MB"
                        ariaLabel={tf("render_cache_min_free_disk")}
                        onCommit={(minFreeDiskMb) => patch({ minFreeDiskMb })}
                    />
                </Flex>

                {/* ── 高级 ─────────────────────────────────────────────── */}
                <AppField label={tf("render_cache_write_mode")}>
                    <AppSelect
                        // 旧写法是 size="1"（24px）：对话框里也要紧凑
                        density="compact"
                        value={draft.writeMode}
                        onValueChange={(v) => patch({ writeMode: v as RenderCacheWriteMode })}
                        options={[
                            { value: "immediate", label: tf("render_cache_write_immediate") },
                            { value: "onExit", label: tf("render_cache_write_on_exit") },
                            { value: "manual", label: tf("render_cache_write_manual") },
                        ]}
                    />
                </AppField>

                <AppField label={tf("render_cache_location_mode")}>
                    <Flex align="center" gap="2" wrap="wrap">
                        <AppSelect
                            fullWidth={false}
                            // 旧写法是 size="1"（24px）：对话框里也要紧凑
                            density="compact"
                            value={draft.location}
                            onValueChange={(v) =>
                                patch({ location: v === "custom" ? "custom" : "system" })
                            }
                            options={[
                                { value: "system", label: tf("render_cache_location_system") },
                                { value: "custom", label: tf("render_cache_location_custom") },
                            ]}
                        />
                        {draft.location === "custom" ? (
                            <TextField.Root
                                size="1"
                                placeholder={tf("render_cache_custom_dir_placeholder")}
                                value={draft.customDir ?? ""}
                                onChange={(event: ChangeEvent<HTMLInputElement>) =>
                                    patch({ customDir: event.target.value })
                                }
                                style={{ flex: 1, minWidth: 220 }}
                            />
                        ) : null}
                    </Flex>
                </AppField>

                <Flex align="center" gap="2">
                    <Checkbox
                        checked={draft.verifyChecksum}
                        onCheckedChange={(v) => patch({ verifyChecksum: Boolean(v) })}
                    />
                    <span className="hs-type-label">{tf("render_cache_verify_checksum")}</span>
                </Flex>

                <Separator size="4" />

                {/* ── 清理 ─────────────────────────────────────────────── */}
                <Flex gap="2" wrap="wrap" align="center">
                    {pendingClearAll ? (
                        <>
                            <span
                                className="hs-type-body"
                                style={{ color: "var(--qt-danger-text)" }}
                            >
                                {tf("render_cache_confirm_clear_all")}
                            </span>
                            <AppButton
                                size="sm"
                                intent="danger"
                                disabled={busyScope !== null}
                                onClick={() => void handleClear("all")}
                            >
                                {tf("render_cache_confirm_yes")}
                            </AppButton>
                            <AppButton size="sm" onClick={() => setPendingClearAll(false)}>
                                {tf("cancel")}
                            </AppButton>
                        </>
                    ) : (
                        <>
                            <AppButton
                                size="sm"
                                intent="danger"
                                disabled={busyScope !== null}
                                onClick={() => setPendingClearAll(true)}
                            >
                                {tf("render_cache_clear_all")}
                            </AppButton>
                            <AppButton
                                size="sm"
                                disabled={busyScope !== null}
                                onClick={() => void handleClear("currentProject")}
                            >
                                {tf("render_cache_clear_project")}
                            </AppButton>
                            <AppButton
                                size="sm"
                                disabled={busyScope !== null}
                                onClick={() =>
                                    void handleClear(
                                        "olderThan",
                                        draft.maxAgeDays > 0 ? draft.maxAgeDays : 90,
                                    )
                                }
                            >
                                {tf("render_cache_clear_old")}
                            </AppButton>
                            <AppButton
                                size="sm"
                                disabled={busyScope !== null}
                                onClick={() => void handleClear("otherSampleRates")}
                            >
                                {tf("render_cache_clear_other_rates")}
                            </AppButton>
                        </>
                    )}
                </Flex>

                {notice ? (
                    <span className="hs-type-body" style={{ color: "var(--qt-success-text)" }}>
                        {notice}
                    </span>
                ) : null}
                {errorText ? (
                    <span className="hs-type-body" style={{ color: "var(--qt-danger-text)" }}>
                        {errorText}
                    </span>
                ) : null}
            </AppForm>
        </AppDialog>
    );
}
