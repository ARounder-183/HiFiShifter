/**
 * 「REAPER 音高互转」的一次执行：分批、串成一步撤销、汇总、反馈。
 *
 * 【为什么要有这个 hook】参数编辑器里的入口不止一处（当前轨道 / 选区内 / 全部），
 * 而"执行一次互转"的步骤在每一处都必须**逐字一致**：分批大小、撤销分组、结果汇总、
 * 失败措辞。抄三份的下场是这个仓库反复吃过的那种：三个入口对同一条片段给出三种结果。
 *
 * 【为什么撤销必须在这里收口】"全部片段"会分成多条命令（每批 128 条）。每条命令自己
 * 会压一个撤销点，于是用户按一次撤销只退回 128 条 —— 而他是**一次点击**触发的。
 * `begin_undo_group` / `end_undo_group` 是仓库既有的收口方式（目录导入也用它）：
 * 组内写入不再各自压点，组结束只留一步。
 *
 * 【为什么反馈走状态栏单槽位】这个操作没有自己的窗口，也不该有：它是一次性的转换，
 * 结果是"转了几条、跳过几条"。状态栏是仓库里"操作做完了，顺带说一声"的唯一去处
 * （与参数块写入失败共用同一个槽位，见 `paramBlockFeedback`）。
 */
import { useCallback, useMemo, useState } from "react";

import { publishStatusFeedback } from "../../components/layout/pianoRoll/paramBlockFeedback";
import { useI18n } from "../../i18n/I18nProvider";
import { frontendLog } from "../../services/frontendLog";
import { canBridgeHostTakePitch } from "../../services/hostCapabilities";
import { webApi } from "../../services/webviewApi";
import {
    emptyOutcome,
    mergeOutcome,
    planPitchBridgeBatches,
    skippedClips,
    uniqueClipIds,
    type PitchBridgeOutcome,
} from "./run";

/** 两个方向。文案与撤销标签都按它分叉。 */
export type PitchBridgeDirection = "import" | "export";

/**
 * 反馈停留时长（毫秒）。
 *
 * 成功是告知性的（用户已经知道自己在做什么），失败要留到被看见 —— 与参数块反馈
 * 同一套取值理由，但独立成常量：两者的"失败"不是一回事，将来各自调整时不该互相牵制。
 */
const HOLD_MS = { success: 6_000, partial: 8_000, failure: 15_000 } as const;

/** 撤销条目里的操作 key（与 `HistoryOp` / 词典里的 `history_op_host_pitch_*` 同名）。 */
const UNDO_LABEL: Record<PitchBridgeDirection, string> = {
    import: "host_pitch_import",
    export: "host_pitch_export",
};

/** 两个方向各自的"做完了"文案。 */
const DONE_KEY: Record<
    PitchBridgeDirection,
    "status_pitch_bridge_imported" | "status_pitch_bridge_exported"
> = {
    import: "status_pitch_bridge_imported",
    export: "status_pitch_bridge_exported",
};

export interface HostPitchBridge {
    /** 宿主是否支持互转（不支持时入口**整项不渲染**，不是置灰）。 */
    readonly available: boolean;
    readonly busy: boolean;
    /** 执行一次互转；`clipIds` 由作用域解析给出。 */
    readonly run: (direction: PitchBridgeDirection, clipIds: readonly string[]) => Promise<void>;
}

export function useHostPitchBridge(): HostPitchBridge {
    const { t, plural } = useI18n();
    const [busy, setBusy] = useState(false);
    const available = canBridgeHostTakePitch();

    /** 把累计结果说成一句状态栏文案（成功 / 部分跳过 / 一条都没转）。 */
    const publishOutcome = useCallback(
        (direction: PitchBridgeDirection, outcome: PitchBridgeOutcome) => {
            const skipped = skippedClips(outcome);
            if (outcome.converted === 0) {
                publishStatusFeedback({
                    text: t("pitch_bridge_nothing_converted"),
                    tone: "warning",
                    tooltip: skipped > 0 ? t("pitch_bridge_skipped_tooltip") : undefined,
                    holdMs: HOLD_MS.partial,
                });
                return;
            }
            const head = plural(DONE_KEY[direction], outcome.converted);
            const text =
                skipped > 0 ? `${head} · ${plural("pitch_bridge_skipped_clips", skipped)}` : head;
            publishStatusFeedback({
                text,
                tone: skipped > 0 ? "warning" : "success",
                tooltip: skipped > 0 ? t("pitch_bridge_skipped_tooltip") : undefined,
                holdMs: skipped > 0 ? HOLD_MS.partial : HOLD_MS.success,
            });
        },
        [plural, t],
    );

    const run = useCallback(
        async (direction: PitchBridgeDirection, clipIds: readonly string[]) => {
            const ids = uniqueClipIds(clipIds);
            if (ids.length === 0) {
                // 空作用域不是错误：菜单项在计数为 0 时已置灰，走到这里说明状态刚变。
                publishStatusFeedback({
                    text: t("pitch_bridge_nothing_converted"),
                    tone: "warning",
                    holdMs: HOLD_MS.partial,
                });
                return;
            }
            setBusy(true);
            const batches = planPitchBridgeBatches(ids);
            // 只有真的分成多批时才开组：单批命令自己会压一个撤销点，多开一层组
            // 等于给"什么都没有写"的情况也留一步空撤销。
            const grouped = batches.length > 1;
            let outcome: PitchBridgeOutcome = emptyOutcome(ids.length);
            let groupOpen = false;
            try {
                if (grouped) {
                    await webApi.beginUndoGroup(UNDO_LABEL[direction]);
                    groupOpen = true;
                }
                for (const batch of batches) {
                    const result =
                        direction === "import"
                            ? await webApi.importHostTakePitch(batch, { tolerant: true })
                            : await webApi.exportHostTakePitch(batch, { tolerant: true });
                    if (result?.ok === false) {
                        throw new Error(result.error ?? "pitch bridge command failed");
                    }
                    outcome = mergeOutcome(outcome, result ?? {});
                }
                publishOutcome(direction, outcome);
            } catch (error) {
                const detail = error instanceof Error ? error.message : String(error);
                frontendLog("warn", `${direction}_host_take_pitch failed`, detail);
                publishStatusFeedback({
                    text: t("status_pitch_bridge_failed"),
                    tone: "danger",
                    tooltip: detail,
                    holdMs: HOLD_MS.failure,
                });
            } finally {
                // 【必须无条件收组】`begin_undo_group` 会把会话的 `suppress_history` 置真；
                // 漏掉 `end_undo_group` 之后**所有**参数编辑都不再压撤销点 —— 那不是
                // "少一条记录"，而是撤销栈从此静默失效。
                if (groupOpen) {
                    try {
                        await webApi.endUndoGroup();
                    } catch (error) {
                        frontendLog("warn", "end_undo_group failed", String(error));
                    }
                }
                setBusy(false);
            }
        },
        [publishOutcome, t],
    );

    return useMemo(() => ({ available, busy, run }), [available, busy, run]);
}
