import { useMemo } from "react";

import { Badge } from "@radix-ui/themes";

import { useI18n } from "../../i18n/I18nProvider";
import { usePitchAnalysis } from "../../contexts/PitchAnalysisContext";
import { resolveRootTrackId } from "../../features/session/trackUtils";

type TrackLite = {
    id: string;
    parentId?: string | null;
    composeEnabled?: boolean;
    pitchAnalysisAlgo?: string;
};

type PitchStatusKind = "off" | "pending" | "ready" | "no_edit" | "unavailable" | "unknown";

export type PitchStatusSnapshot = {
    analysisPending: boolean;
    analysisProgress: number | null;
    pitchEditUserModified: boolean | null;
    pitchEditBackendAvailable: boolean | null;
};

export function PitchStatusBadge(props: {
    tracks: TrackLite[];
    selectedTrackId: string | null;
    status?: PitchStatusSnapshot;
    className?: string;
}) {
    const { tracks, selectedTrackId, status, className } = props;
    const { t } = useI18n();

    // 全局 pitch 分析状态（由 PitchAnalysisProvider 统一维护）
    const pitchAnalysis = usePitchAnalysis();

    const rootTrackId = useMemo(
        () => resolveRootTrackId(tracks, selectedTrackId),
        [selectedTrackId, tracks],
    );

    const rootTrack = useMemo(() => {
        if (!rootTrackId) return null;
        return tracks.find((tr) => tr.id === rootTrackId) ?? null;
    }, [tracks, rootTrackId]);

    const hardDisableReason = useMemo(() => {
        if (!rootTrack) return null;
        if (!rootTrack.composeEnabled) return t("pitch_requires_compose");
        if (String(rootTrack.pitchAnalysisAlgo ?? "") === "none") {
            return t("pitch_requires_algo");
        }
        return null;
    }, [rootTrack, t]);

    // 计算最终展示的 kind 和 progress
    const { kind, progress } = useMemo((): {
        kind: PitchStatusKind;
        progress: number | null;
    } => {
        if (hardDisableReason) {
            return { kind: "off", progress: null };
        }

        // 优先使用外部传入的 status prop（包含 userModified / backendAvailable 等信息）
        if (status) {
            const pending = Boolean(status.analysisPending);
            const backendAvail = status.pitchEditBackendAvailable ?? null;
            const userModified = status.pitchEditUserModified ?? null;

            if (backendAvail === false) return { kind: "unavailable", progress: null };
            if (pending) {
                return {
                    kind: "pending",
                    progress:
                        typeof status.analysisProgress === "number"
                            ? status.analysisProgress
                            : null,
                };
            }
            if (userModified === false) return { kind: "no_edit", progress: null };
            if (userModified === true) return { kind: "ready", progress: null };
            return { kind: "unknown", progress: null };
        }

        // 无 status prop 时，降级到全局 PitchAnalysisContext 状态
        if (pitchAnalysis.pending) {
            return {
                kind: "pending",
                progress: pitchAnalysis.progress,
            };
        }

        return { kind: "unknown", progress: null };
    }, [hardDisableReason, status, pitchAnalysis]);

    const { color, text, title } = useMemo(() => {
        if (hardDisableReason) {
            return {
                color: "gray" as const,
                text: `${t("pitch_status_label")} ${t("pitch_status_off")}`,
                title: hardDisableReason,
            };
        }

        switch (kind) {
            case "pending": {
                const p =
                    typeof progress === "number" && Number.isFinite(progress)
                        ? Math.round(Math.max(0, Math.min(1, progress)) * 100)
                        : null;
                return {
                    color: "amber" as const,
                    text:
                        p == null
                            ? `${t("pitch_status_label")} ${t("pitch_status_pending")}`
                            : `${t("pitch_status_label")} ${t("pitch_status_pending")} ${p}%`,
                    title: `${t("pitch_status_label")}: ${t("pitch_status_pending")}`,
                };
            }
            case "unavailable":
                return {
                    color: "red" as const,
                    text: `${t("pitch_status_label")} ${t("pitch_status_unavailable")}`,
                    title: t("pitch_backend_unavailable"),
                };
            case "no_edit":
                return {
                    color: "gray" as const,
                    text: `${t("pitch_status_label")} ${t("pitch_status_no_edit")}`,
                    title: t("pitch_edit_not_modified_hint"),
                };
            case "ready":
                return {
                    color: "green" as const,
                    text: `${t("pitch_status_label")} ${t("pitch_status_ready")}`,
                    title: `${t("pitch_status_label")}: ${t("pitch_status_ready")}`,
                };
            case "off":
                return {
                    color: "gray" as const,
                    text: `${t("pitch_status_label")} ${t("pitch_status_off")}`,
                    title: t("pitch_status_off"),
                };
            case "unknown":
            default:
                return {
                    color: "gray" as const,
                    text: `${t("pitch_status_label")} ${t("status_na")}`,
                    title: `${t("pitch_status_label")}: ${t("status_na")}`,
                };
        }
    }, [hardDisableReason, kind, progress, t]);

    return (
        <Badge
            variant="soft"
            size="1"
            color={color}
            title={title}
            className={"select-none " + (className ?? "")}
        >
            {text}
        </Badge>
    );
}
