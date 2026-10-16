/**
 * 宿主链接面板（独立 App）。
 *
 * 【为什么是面板而不是常驻横条】打开宿主会话是一次长活：打开之后要反复提交 / 重新
 * 导入 / 断开。但它在独立 App 里又是低频入口 —— 多数用户根本不连宿主。因此做成
 * **默认关闭的浮出面板**：入口在「视图」菜单，需要时打开，不用时不占任何空间
 * （与「外观设置」同一取舍）。
 *
 * 【为什么状态来自模块级 store】会话的寿命不该等于面板的挂载周期：用户关掉面板再
 * 打开，工作区其实还开着。状态因此放在 `hostLinkStore`，面板只是它的一个视图。
 *
 * 【为什么状态读数用 `AppStatusChip`】它与修订号是同一类信息，全应用的状态片已经
 * 有一套语义色（见 `ui/StatusChip.tsx`）；自绘会让它在暗色主题下与其余状态片不一致。
 */
import { useEffect, useState } from "react";
import { Flex } from "@radix-ui/themes";

import { useI18n } from "../../i18n/I18nProvider";
import { AppButton, AppConfirmDialog, AppField, AppForm, AppSelect, AppStatusChip } from "../../ui";
import {
    hostLinkErrorText,
    hostLinkStatusText,
    isDirtyProjectError,
    type HostLinkSourceKind,
} from "./hostLinkApi";
import { getHostLinkSnapshot, hostLinkActions, useHostLinkSnapshot } from "./hostLinkStore";

/** 待确认的脏工程动作：打开或重新导入。 */
type PendingAction = "open" | "refresh";

export function HostLinkPanel({
    dirty,
    onTimelineChanged,
}: {
    dirty: boolean;
    onTimelineChanged: () => Promise<unknown>;
}) {
    const { t, tVars } = useI18n();
    const { root, items, session, busy, error, status } = useHostLinkSnapshot();
    const [selected, setSelected] = useState("");
    const [confirm, setConfirm] = useState<PendingAction | null>(null);

    // 打开面板就扫一次；面板重开时 store 里的列表还在，但重扫一次能反映磁盘变化。
    useEffect(() => {
        void list();
    }, []);

    async function list() {
        await hostLinkActions.list();
        // 【为什么回落到第一项而不是清空】列表里每一项都是可选的（unreadable 项作为
        // 禁用项展示原因）；清空会让下拉在每次重扫后跳动。选中项消失时才回落。
        // 这里读**非 hook** 的 getter：`list` 可能从事件处理器里调用，不是渲染期。
        const found = getHostLinkSnapshot().items;
        setSelected((value) =>
            found.some((item) => item.directory === value) ? value : (found[0]?.directory ?? ""),
        );
    }

    const selectedItem = items.find((item) => item.directory === selected);
    const selectedReadable = selectedItem !== undefined && !selectedItem.unreadable;

    /**
     * 打开 / 重新导入。
     *
     * 脏工程要**先确认**：工作区快照会覆盖本地编辑。脏也可能只被**后端**发现
     * （前端状态过期），因此后端返回 `dirty_project:` 时同样回到确认态，而不是把
     * 错误直接摊给用户。
     */
    async function operate(kind: PendingAction, force = false) {
        if (dirty && !force) {
            setConfirm(kind);
            return;
        }
        setConfirm(null);
        const failure =
            kind === "open"
                ? await hostLinkActions.open(selected, force)
                : await hostLinkActions.refresh(force);
        if (!failure) {
            await onTimelineChanged();
            return;
        }
        if (!force && isDirtyProjectError(failure)) {
            hostLinkActions.clearError();
            setConfirm(kind);
        }
    }

    async function submit() {
        const failure = await hostLinkActions.submit();
        if (!failure) await onTimelineChanged();
    }

    const options = items.map((item) => ({
        value: item.directory,
        disabled: item.unreadable !== undefined,
        // 【为什么 unreadable 单列一条】"有东西在那里，但读不了"与"什么都没有"是
        // 两回事；把原因挂在禁用项上，用户一眼能看到该去检查什么。
        label: item.unreadable
            ? tVars("host_link_unreadable_option", {
                  directory: item.directory,
                  error: item.unreadable,
              })
            : tVars("host_link_item_option", {
                  label: item.hostLabel ?? item.directory,
                  directory: item.directory,
              }),
    }));

    // 会话级事实：宿主不接受提交 / 几何改动只留在本地 / 有条目被降级。
    const commitUnsupported = session !== null && session.acceptsCommit === false;
    // `gatesGeometry === true` 表示宿主拥有几何：本地几何改动无法提交，只能留在本地。
    const geometryLocalOnly = session !== null && session.gatesGeometry === true;
    const skipped = session?.skippedCount ?? 0;
    const losses = session?.losses ?? [];
    const submitStatus = session?.status ?? null;

    return (
        <div className="hs-scroll-gutter-flush custom-scrollbar flex h-full flex-col gap-qt-4 overflow-y-auto p-qt-5">
            <AppForm labelWidth="auto">
                <AppField
                    label={t("host_link_directory_label")}
                    hint={root ? tVars("host_link_search_root", { directory: root }) : undefined}
                >
                    {/*
                     * 没有工作区时**不渲染下拉**：Radix 的 `Select.Item` 不接受空串值，
                     * 用哨兵值去凑只会多一个要处处过滤的伪状态。直接给一句说明更诚实。
                     */}
                    {items.length ? (
                        <AppSelect
                            value={selected}
                            onValueChange={setSelected}
                            options={options}
                            disabled={busy || session !== null}
                            density="compact"
                            minWidth={220}
                            ariaLabel={t("host_link_directory_label")}
                        />
                    ) : (
                        <span className="hs-type-label" style={{ color: "var(--qt-text-muted)" }}>
                            {t("host_link_no_workspaces")}
                        </span>
                    )}
                </AppField>
            </AppForm>

            {selectedItem?.unreadable ? (
                <span className="hs-type-caption" style={{ color: "var(--qt-warning-text)" }}>
                    {tVars("host_link_unreadable_option", {
                        directory: selectedItem.directory,
                        error: selectedItem.unreadable,
                    })}
                </span>
            ) : null}

            <Flex align="center" gap="2" wrap="wrap">
                <AppButton size="sm" disabled={busy} onClick={() => void list()}>
                    {t("host_link_rescan")}
                </AppButton>
                <AppButton
                    size="sm"
                    disabled={busy || !selectedReadable || session !== null}
                    onClick={() => void operate("open")}
                >
                    {t("host_link_open")}
                </AppButton>
                <AppButton
                    size="sm"
                    intent="primary"
                    // 宿主没有声明 `commit` 能力时，提交必然失败：禁用而不是让用户
                    // 点一次再吃一个错误。
                    disabled={busy || session === null || commitUnsupported}
                    onClick={() => void submit()}
                >
                    {t("host_link_submit")}
                </AppButton>
                <AppButton
                    size="sm"
                    disabled={busy || session === null}
                    onClick={() => void operate("refresh")}
                >
                    {t("host_link_refresh")}
                </AppButton>
                <AppButton
                    size="sm"
                    disabled={busy || session === null}
                    onClick={() => void hostLinkActions.release()}
                >
                    {t("host_link_release")}
                </AppButton>
            </Flex>

            {/* 重新导入保留曲线的说明常驻在刷新动作旁，用户按之前就知道。 */}
            <span className="hs-type-caption" style={{ color: "var(--qt-text-muted)" }}>
                {t("host_link_refresh_keeps_curves")}
            </span>

            <Flex align="center" gap="2" wrap="wrap">
                <AppStatusChip
                    tone={error ? "danger" : busy ? "accent" : session ? "success" : "neutral"}
                    // 修订数是两行放不下的读数，用自定义气泡换行显示（与
                    // `PluginApplyStatus` 上的同一个取舍）。
                    tooltip={
                        session
                            ? tVars("host_link_revisions", {
                                  edits: session.edits ?? 0,
                                  model: session.model ?? 0,
                              })
                            : undefined
                    }
                >
                    {busy ? t("host_link_busy") : status ? t(status) : t("host_link_status_idle")}
                </AppStatusChip>
                {session ? (
                    <AppStatusChip tone="accent" tabular>
                        {sourceKindLabel(session.sourceKind, t)}
                    </AppStatusChip>
                ) : null}
            </Flex>

            {session ? (
                <AppForm labelWidth="auto">
                    <AppField label={t("host_link_clips_label")}>{String(session.clips)}</AppField>
                    <AppField label={t("host_link_tracks_label")}>
                        {String(session.tracks)}
                    </AppField>
                    <AppField label={t("host_link_audio_label")}>
                        {tVars("host_link_audio_summary", {
                            complete: session.audio.complete,
                            partial: session.audio.partial,
                            extern: session.audio.extern,
                        })}
                    </AppField>
                    <AppField label={t("host_link_revisions_label")}>
                        {tVars("host_link_revisions", {
                            edits: session.edits,
                            model: session.model,
                        })}
                    </AppField>
                </AppForm>
            ) : null}

            {/*
             * 能力声明要**明说**，不能让用户从"提交失败"里自己猜：宿主不接受提交、
             * 几何改动不回宿主，都是明示的事实而不是谜。
             */}
            {commitUnsupported ? (
                <span className="hs-type-caption" style={{ color: "var(--qt-text-muted)" }}>
                    {t("host_link_commit_unsupported")}
                </span>
            ) : null}
            {geometryLocalOnly ? (
                <span className="hs-type-caption" style={{ color: "var(--qt-text-muted)" }}>
                    {t("host_link_geometry_local_only")}
                </span>
            ) : null}

            {skipped > 0 ? (
                <span className="hs-type-caption" style={{ color: "var(--qt-warning-text)" }}>
                    {tVars("host_link_skipped_summary", {
                        count: skipped,
                        detail:
                            session && session.skippedDetail.length
                                ? session.skippedDetail.join("; ")
                                : (session?.skipped ?? ""),
                    })}
                </span>
            ) : null}
            {losses.length > 0 ? (
                <span className="hs-type-caption" style={{ color: "var(--qt-warning-text)" }}>
                    {tVars("host_link_losses_summary", { detail: losses.join("; ") })}
                </span>
            ) : null}
            {session?.carriedParams !== undefined && session.carriedTracks !== undefined ? (
                <span className="hs-type-caption" style={{ color: "var(--qt-text-muted)" }}>
                    {tVars("host_link_carried_summary", {
                        params: session.carriedParams,
                        tracks: session.carriedTracks,
                    })}
                </span>
            ) : null}

            {/* 提交结果：已落盘（applied）还是仍在途（pending）；被拒时给本地化的错误。 */}
            {session && session.applied === true ? (
                <span role="status" className="hs-type-caption">
                    {t("host_link_applied")}
                </span>
            ) : null}
            {session && session.pending === true ? (
                <span role="status" className="hs-type-caption">
                    {t("host_link_pending")}
                </span>
            ) : null}
            {submitStatus && !submitStatus.ok ? (
                <span
                    role="alert"
                    className="hs-type-caption"
                    style={{ color: "var(--qt-danger-text)" }}
                >
                    {hostLinkStatusText(submitStatus, t, tVars)}
                </span>
            ) : null}

            {error ? (
                <span
                    role="alert"
                    className="hs-type-caption"
                    style={{ color: "var(--qt-danger-text)" }}
                >
                    {hostLinkErrorText(error, t, tVars)}
                </span>
            ) : null}

            <AppConfirmDialog
                open={confirm !== null}
                onOpenChange={(open) => {
                    if (!open) setConfirm(null);
                }}
                title={t("host_link_replace_dirty_confirm")}
                message={t("host_link_replace_dirty_message")}
                confirmLabel={t("host_link_replace_dirty_confirm")}
                cancelLabel={t("cancel")}
                intent="danger"
                tone="danger"
                onConfirm={() => {
                    if (confirm) void operate(confirm, true);
                }}
            />
        </div>
    );
}

/** 来源词条：ARA / Transfer / 桥接。都是词表里的既有键，不拼字符串。 */
function sourceKindLabel(
    kind: HostLinkSourceKind,
    t: (
        key: "host_link_source_ara" | "host_link_source_transfer" | "host_link_source_bridge",
    ) => string,
): string {
    switch (kind) {
        case "ara":
            return t("host_link_source_ara");
        case "transfer":
            return t("host_link_source_transfer");
        case "bridge":
            return t("host_link_source_bridge");
    }
}
