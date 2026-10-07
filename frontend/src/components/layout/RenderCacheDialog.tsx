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

import { useCallback, useEffect, useRef, useState, type ChangeEvent, type ReactNode } from "react";
import { Checkbox, Flex, Separator, TextField } from "@radix-ui/themes";
import { useAppDispatch, useAppSelector } from "../../app/hooks";
import { useI18n } from "../../i18n/I18nProvider";
import {
    coreApi,
    type RenderCacheClearScope,
    type RenderCacheStats,
} from "../../services/api/core";
import { fileBrowserApi } from "../../services/api/fileBrowser";
import {
    normalizeRenderCacheSettings,
    type RenderCacheSettings,
    type RenderCacheWriteMode,
} from "../../services/api/settings";
import { persistUiSettings } from "../../features/session/thunks/runtimeThunks";
import { setRenderCacheSettings } from "../../features/session/sessionSlice";
import { AppDialog } from "../../ui/Dialog";
import { AppForm } from "../../ui/Field";
import { AppButton, AppNumberField, AppSelect } from "../../ui";

interface RenderCacheDialogProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
}

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

/**
 * 密集数值清单里的一个字段：标签 + 控件 +（可选的）取值约定。
 *
 * 【为什么不用 `AppField`】`AppField` 是"标签列 + 控件列"的**整行**布局：每个字段
 * 独占一行、并留出一条定宽标签列。本对话框的六个数值字段是一张密集清单 ——
 * 一行能放几个就放几个才不占高度（见使用处的说明）。
 *
 * 【取值约定为什么贴在同一行】`0 = 不限制` 这类文字是对**当前值**的注解，不是需要
 * 先读的说明；挂到下一行会让字段行高翻倍（长表单因此长得没必要，实测正是这处
 * 让窗口超出可视高度）。
 *
 * 【`data-hs-field-group`】给测试一个稳定的"一行"标记：断言"这一行里没有下拉"
 * （那条哨兵缺陷的回归）需要从输入框回溯到整行，而不该依赖 Tailwind 类名的层数。
 */
const FieldGroup: React.FC<{
    label: string;
    unitHint?: string;
    children: ReactNode;
}> = ({ label, unitHint, children }) => (
    <div data-hs-field-group="1" className="flex items-center gap-2">
        <span className="hs-type-label shrink-0">{label}</span>
        {children}
        {unitHint ? <span className="hs-type-caption whitespace-nowrap">{unitHint}</span> : null}
    </div>
);

export function RenderCacheDialog({ open, onOpenChange }: RenderCacheDialogProps) {
    const dispatch = useAppDispatch();
    const { tf, tVars } = useI18n();
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

    /**
     * 选择自定义缓存目录。
     *
     * 取消（`canceled`）不是错误：保持原值不动、也不提示。成功后清掉上一次的错误 ——
     * 否则用户修好了路径、屏幕上还挂着旧的失败信息。
     */
    async function handleBrowseCustomDir() {
        try {
            const result = await fileBrowserApi.pickDirectory();
            if (!result.ok) {
                setErrorText(tf("render_cache_browse_failed"));
                return;
            }
            if (!result.canceled && result.path) {
                patch({ customDir: result.path });
                setErrorText("");
            }
        } catch {
            setErrorText(tf("render_cache_browse_failed"));
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
            {/*
             * 只用 `AppForm` 的**行距**：本对话框每一行都是"标签 + 控件"的紧凑清单，
             * 没有 `AppField` 的标签列，因此不下发 `labelWidth`（下发了也没有消费者）。
             */}
            <AppForm>
                {/* ── 状态 ───────────────────────────────────────────────
                 * 摘要与两个操作按钮**同一行**：它们都是"当前状态"的附属动作，
                 * 单独占一行会让这一块多出 30px（窗口本来就在滚动边缘）。
                 * 路径是对摘要的补充，紧跟其下一行（两层之间不留表单间距）。 */}
                <div className="flex flex-col">
                    <Flex align="center" gap="2" wrap="wrap">
                        <span className="hs-type-body min-w-0 flex-1 truncate">{summaryText}</span>
                        <AppButton size="sm" onClick={() => void handleOpenDir()}>
                            {tf("render_cache_open_dir")}
                        </AppButton>
                        <AppButton size="sm" disabled={loading} onClick={() => void refreshStats()}>
                            {tf("render_cache_refresh")}
                        </AppButton>
                    </Flex>
                    <span className="hs-type-caption" style={{ wordBreak: "break-all" }}>
                        {tVars("common_label_value", {
                            label: tf("render_cache_location_label"),
                            value: stats?.dir ?? "...",
                        })}
                    </span>
                </div>
                {stats && !stats.writable ? (
                    <span className="hs-type-caption" style={{ color: "var(--qt-warning-text)" }}>
                        {tf("render_cache_dir_not_writable")}
                    </span>
                ) : null}
                {stats && stats.sessionWriteErrors > 0 ? (
                    <span className="hs-type-caption" style={{ color: "var(--qt-warning-text)" }}>
                        {tf("render_cache_write_errors").replace(
                            "{n}",
                            String(stats.sessionWriteErrors),
                        )}
                    </span>
                ) : null}

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

                {/* ── 容量与清理策略 ─────────────────────────────────────
                 * 六个数值字段挤在**同一个可换行行**里（标签 + 输入框 + 单位 +
                 * 可选取值约定），而不是"两个 `AppField` 各占一行、另外四个挤在另一
                 * 行"。此前那种排法让同一组参数有两套行高（31 / 17.5）和四种标签宽
                 * 度（132 / 108 / 108 / 108），既难看又把窗口撑到必须滚动 —— 实测
                 * 内容 550px、可视 466px。
                 *
                 * 【为什么不用 `AppField` 的标签列】它们的标签列会为每个字段强制
                 * 一条独立行（含 132px 的空标签列），六个字段就是六行；本对话框的
                 * 字段是"一行能放几个就放几个"的密集清单，不是"一列一个"的设置表。
                 *
                 * 【为什么不配预设下拉】曾经每个字段配一个 `AppSelect`（512 MB…8 GB
                 * / 不限，7…365 天 / 永不），用 `"custom"` 这个**不在选项里**的哨兵值
                 * 表示"当前值不是预设"。而 Radix 会为表单兼容渲染一个隐藏的原生
                 * `<select>`，把受控值镜像进去；受控值一旦不在 `<option>` 里，浏览器就
                 * 把 `select.value` 归为 `""` 并派发一个冒泡的 `change`，Radix 原样转发
                 * 成 `onValueChange("")` —— 于是"滚轮把 4096 调成 4097"会顺带触发一次
                 * `Number("") === 0`，把占用上限静默改成"不限"（用户报告的"滚轮直接
                 * 跳到 0"）。因此只保留数字框；`0` 的语义由取值约定讲明。
                 */}
                <Flex align="center" gap="3" wrap="wrap">
                    <FieldGroup
                        label={tf("render_cache_max_size")}
                        unitHint={tf("render_cache_max_size_hint")}
                    >
                        <AppNumberField
                            value={draft.maxSizeMb}
                            unit="megabytes"
                            min={0}
                            width={110}
                            suffix="MB"
                            ariaLabel={tf("render_cache_max_size")}
                            onCommit={(maxSizeMb) => patch({ maxSizeMb })}
                        />
                    </FieldGroup>
                    <FieldGroup
                        label={tf("render_cache_max_age")}
                        unitHint={tf("render_cache_max_age_hint")}
                    >
                        <AppNumberField
                            value={draft.maxAgeDays}
                            unit="days"
                            min={0}
                            width={90}
                            suffix={tf("render_cache_days_unit")}
                            ariaLabel={tf("render_cache_max_age")}
                            onCommit={(maxAgeDays) => patch({ maxAgeDays })}
                        />
                    </FieldGroup>
                    <FieldGroup label={tf("render_cache_min_clip")}>
                        <AppNumberField
                            value={draft.minClipSecs}
                            unit="clipSeconds"
                            min={0}
                            width={90}
                            suffix={tf("render_cache_seconds_unit")}
                            ariaLabel={tf("render_cache_min_clip")}
                            onCommit={(minClipSecs) => patch({ minClipSecs })}
                        />
                    </FieldGroup>
                    <FieldGroup label={tf("render_cache_min_entry")}>
                        <AppNumberField
                            value={draft.minEntryKb}
                            unit="kilobytes"
                            min={0}
                            width={90}
                            suffix="KB"
                            ariaLabel={tf("render_cache_min_entry")}
                            onCommit={(minEntryKb) => patch({ minEntryKb })}
                        />
                    </FieldGroup>
                    <FieldGroup label={tf("render_cache_max_entry")}>
                        <AppNumberField
                            value={draft.maxEntryMb}
                            unit="entryMegabytes"
                            min={0}
                            width={90}
                            suffix="MB"
                            ariaLabel={tf("render_cache_max_entry")}
                            onCommit={(maxEntryMb) => patch({ maxEntryMb })}
                        />
                    </FieldGroup>
                    <FieldGroup label={tf("render_cache_min_free_disk")}>
                        <AppNumberField
                            value={draft.minFreeDiskMb}
                            unit="diskMegabytes"
                            min={0}
                            width={90}
                            suffix="MB"
                            ariaLabel={tf("render_cache_min_free_disk")}
                            onCommit={(minFreeDiskMb) => patch({ minFreeDiskMb })}
                        />
                    </FieldGroup>
                </Flex>

                {/* ── 写入与位置 ─────────────────────────────────────────
                 * "什么时候写"与"写到哪里"是同一个问题的两半，同一行放下（各占半格）。
                 * 分成两行时它们各自只占半行宽，却让窗口多出约 35px —— 而窗口的高度
                 * 恰好是用户抱怨的那一项。 */}
                <Flex align="center" gap="3" wrap="wrap">
                    <FieldGroup label={tf("render_cache_write_mode")}>
                        <AppSelect
                            // 旧写法是 size="1"（24px）：对话框里也要紧凑
                            density="compact"
                            fullWidth={false}
                            value={draft.writeMode}
                            onValueChange={(v) => patch({ writeMode: v as RenderCacheWriteMode })}
                            options={[
                                { value: "immediate", label: tf("render_cache_write_immediate") },
                                { value: "onExit", label: tf("render_cache_write_on_exit") },
                                { value: "manual", label: tf("render_cache_write_manual") },
                            ]}
                        />
                    </FieldGroup>
                    <FieldGroup label={tf("render_cache_location_mode")}>
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
                    </FieldGroup>
                </Flex>

                {/* 自定义目录：只在选了"自定义目录"时出现，因此单独一行（占满宽度，
                    路径输入框本来就需要 200px 以上）。 */}
                {draft.location === "custom" ? (
                    <Flex align="center" gap="2">
                        <span className="hs-type-label shrink-0">
                            {tf("render_cache_location_custom")}
                        </span>
                        <TextField.Root
                            size="1"
                            placeholder={tf("render_cache_custom_dir_placeholder")}
                            value={draft.customDir ?? ""}
                            onChange={(event: ChangeEvent<HTMLInputElement>) =>
                                patch({ customDir: event.target.value })
                            }
                            style={{ flex: 1, minWidth: 200 }}
                        />
                        {/*
                          浏览按钮：这个字段要的是**绝对路径**，让用户手敲或从别处复制
                          路径是最容易出错的一步（Windows 反斜杠、中文目录名）。走既有的
                          `pick_directory`（原生文件夹选择器），与导出 / 快速导出同一套 ——
                          它此前只给了输入框，等于把这一步省掉了。
                        */}
                        <AppButton size="sm" onClick={() => void handleBrowseCustomDir()}>
                            {tf("render_cache_browse")}
                        </AppButton>
                    </Flex>
                ) : null}

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
