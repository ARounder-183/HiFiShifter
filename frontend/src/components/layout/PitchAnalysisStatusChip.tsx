/*
 * 状态栏的「音高分析中」指示片。
 *
 * 【为什么单独成一个组件】音高分析的进度事件是逐 clip 的高频流（后端每个 clip
 * 都推一条），而 AppInner 对它的唯一消费是拼这一条状态栏文案。曾经 AppInner
 * 直接 `usePitchAnalysis()`，于是分析期间每条进度事件都重渲染整个应用骨架
 * （MenuBar、全部对话框、DockRoot 与所有已挂载面板）—— 与
 * `ParamDataLoadingChip` / `AppStatusProgressChips` 拆出去的理由完全相同。
 * 抽成独立组件后，重渲染被限制在这一个 `<AppStatusChip>` 上。
 */

import { useI18n } from "../../i18n/I18nProvider";
import { AppStatusChip } from "../../ui";
import { usePitchAnalysis } from "../../contexts/PitchAnalysisContext";

export function PitchAnalysisStatusChip() {
    const { t } = useI18n();
    const pitchAnalysis = usePitchAnalysis();

    if (!pitchAnalysis.pending) return null;

    const parts: string[] = [t("status_analyzing_pitch")];
    if (pitchAnalysis.currentClip) {
        parts.push(`"${pitchAnalysis.currentClip}"`);
    }
    if (pitchAnalysis.totalClips != null && pitchAnalysis.totalClips > 0) {
        parts.push(`(${pitchAnalysis.completedClips ?? 0}/${pitchAnalysis.totalClips})`);
    }
    if (pitchAnalysis.progress != null && Number.isFinite(pitchAnalysis.progress)) {
        parts.push(`${Math.round(pitchAnalysis.progress * 100)}%`);
    }

    return <AppStatusChip tone="accent">{parts.join(" ")}</AppStatusChip>;
}
