/*
 * 状态栏的「参数曲线取数中」指示片。
 *
 * 【为什么单独成一个组件】这个状态只服务这一处显示，却曾经住在包裹整个
 * `AppInner` 的 Context 里 —— 于是它每翻转一次，整棵应用树（时间轴、参数编辑器、
 * 全部面板）就重渲染一次，把用户的绘制笔画打断。抽成独立组件并直接订阅外部
 * store（`pianoRollStatusBus`）后，重渲染被限制在这一个 `<span>` 上。
 *
 * 【为什么不在 AppInner 里 useSyncExternalStore】那样等于把订阅装回大组件，
 * 放大路径原样复现。
 */

import { useSyncExternalStore } from "react";

import { useI18n } from "../../i18n/I18nProvider";
import { getPianoRollLoading, subscribePianoRollLoading } from "../../utils/pianoRollStatusBus";

export function ParamDataLoadingChip() {
    const { t } = useI18n();
    const loading = useSyncExternalStore(
        subscribePianoRollLoading,
        getPianoRollLoading,
        getPianoRollLoading,
    );

    if (!loading) return null;

    return (
        <span
            className="shrink-0 rounded px-1 py-0 text-xs font-medium"
            style={{
                background: "var(--accent-3)",
                color: "var(--accent-11)",
                fontSize: "11px",
                lineHeight: "16px",
            }}
        >
            {t("loading")}
        </span>
    );
}
