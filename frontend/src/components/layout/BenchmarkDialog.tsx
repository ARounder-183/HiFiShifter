/**
 * Inference Device Benchmark Dialog
 *
 * Runs the backend run_vocoder_benchmark command which tests CPU, GPU (WebGPU),
 * and GPU (DirectML) inference latency (median over 1024 frames / ~12 s of audio)
 * and displays the results so the user can pick the fastest provider for their system.
 */

import { useEffect, useRef, useState } from "react";
import { Flex } from "@radix-ui/themes";
import type { BenchmarkResult } from "../../types/api";
import { coreApi } from "../../services/api/core";
import { useI18n } from "../../i18n/I18nProvider";
import { AppDialog } from "../../ui/Dialog";
import { AppBusy } from "../../ui";
import { AppForm } from "../../ui/Field";

interface BenchmarkDialogProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
}

type BenchmarkPhase = "idle" | "running" | "done" | "error";

function formatMs(ms: number): string {
    return `${ms.toFixed(1)} ms`;
}

function formatRtf(rtf: number): string {
    return `${rtf.toFixed(3)}×`;
}

interface EpRow {
    label: string;
    medianMs: number;
    rtf: number;
    available: boolean;
}

function buildRows(result: BenchmarkResult): EpRow[] {
    const rows: EpRow[] = [
        {
            label: "CPU",
            medianMs: result.cpuMedianMs,
            rtf: result.cpuRtFactor,
            available: true,
        },
    ];

    // GPU (WebGPU)
    if (result.gpuMedianMs != null && result.gpuRtFactor != null) {
        rows.push({
            label: result.gpuBackendName ? `GPU (${result.gpuBackendName})` : "GPU (WebGPU)",
            medianMs: result.gpuMedianMs,
            rtf: result.gpuRtFactor,
            available: true,
        });
    } else if (result.gpuAvailable) {
        rows.push({
            label: result.gpuBackendName ? `GPU (${result.gpuBackendName})` : "GPU (WebGPU)",
            medianMs: -1,
            rtf: -1,
            available: false,
        });
    }

    // GPU (DirectML)
    if (result.dmlMedianMs != null && result.dmlRtFactor != null) {
        rows.push({
            label: "GPU (DirectML)",
            medianMs: result.dmlMedianMs,
            rtf: result.dmlRtFactor,
            available: true,
        });
    } else if (result.dmlAvailable) {
        rows.push({
            label: "GPU (DirectML)",
            medianMs: -1,
            rtf: -1,
            available: false,
        });
    }

    return rows;
}

export function BenchmarkDialog({ open, onOpenChange }: BenchmarkDialogProps) {
    const { t } = useI18n();
    const [phase, setPhase] = useState<BenchmarkPhase>("idle");
    const [result, setResult] = useState<BenchmarkResult | null>(null);
    const [errorText, setErrorText] = useState<string>("");
    const abortRef = useRef(false);

    useEffect(() => {
        if (!open) return;
        // eslint-disable-next-line react-hooks/set-state-in-effect -- 对话框打开时用 props 初始化局部 state（既有模式；重构会改变打开时序）
        setPhase("idle");
        setResult(null);
        setErrorText("");
        abortRef.current = false;
    }, [open]);

    async function handleRun() {
        abortRef.current = false;
        setPhase("running");
        setResult(null);
        setErrorText("");

        try {
            const res = await coreApi.runVocoderBenchmark();
            if (abortRef.current) return;
            setResult(res);
            setPhase("done");
        } catch (e: unknown) {
            if (abortRef.current) return;
            setErrorText(e instanceof Error ? e.message : String(e));
            setPhase("error");
        }
    }

    function handleClose() {
        abortRef.current = true;
        onOpenChange(false);
    }

    const rows = result ? buildRows(result) : [];
    const fastestRow =
        rows.length > 0
            ? rows.reduce(
                  (best, r) => (r.available && r.medianMs < best.medianMs ? r : best),
                  rows[0],
              )
            : null;

    const gpuFailed = result?.gpuAvailable && result.gpuMedianMs == null;
    const dmlFailed = result?.dmlAvailable && result.dmlMedianMs == null;

    return (
        <AppDialog
            open={open}
            onOpenChange={(o) => {
                if (!o) handleClose();
            }}
            title={t("benchmark_title")}
            description={t("benchmark_desc")}
            size="md"
            actions={[
                { id: "close", label: t("benchmark_close"), onClick: handleClose },
                {
                    id: "run",
                    label:
                        phase === "running" ? t("benchmark_running_btn") : t("benchmark_run_btn"),
                    intent: "primary",
                    disabled: phase === "running",
                    autoClose: false,
                    onClick: () => {
                        void handleRun();
                    },
                },
            ]}
        >
            <AppForm>
                {/* Running state */}
                {phase === "running" && (
                    <Flex align="center" gap="2">
                        <AppBusy size="md" label={t("benchmark_running")} />
                    </Flex>
                )}

                {/* Results table */}
                {phase === "done" && result && rows.length > 0 && (
                    <Flex direction="column" gap="2">
                        <span className="hs-type-label font-medium">
                            {t("benchmark_results").replace(
                                "{samples}",
                                String(result.benchmarkSamples),
                            )}
                        </span>
                        <div
                            style={{
                                borderRadius: 6,
                                overflow: "hidden",
                                border: "1px solid var(--qt-border)",
                            }}
                        >
                            <table
                                style={{
                                    width: "100%",
                                    borderCollapse: "collapse",
                                    fontSize: "var(--qt-fs-md)",
                                }}
                            >
                                <thead>
                                    <tr
                                        style={{
                                            background: "var(--qt-surface)",
                                            textAlign: "left",
                                        }}
                                    >
                                        <th style={{ padding: "6px 12px", fontWeight: 500 }}>
                                            {t("benchmark_device_header")}
                                        </th>
                                        <th style={{ padding: "6px 12px", fontWeight: 500 }}>
                                            {t("benchmark_latency_header")}
                                        </th>
                                        <th style={{ padding: "6px 12px", fontWeight: 500 }}>
                                            {t("benchmark_rtf_header")}
                                        </th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {rows.map((row) => {
                                        const isFastest =
                                            row.label === fastestRow?.label && row.available;
                                        const isUnavailable = !row.available;
                                        return (
                                            <tr
                                                key={row.label}
                                                style={{
                                                    background: isFastest
                                                        ? "var(--accent-3)"
                                                        : "transparent",
                                                    borderTop: "1px solid var(--qt-border)",
                                                }}
                                            >
                                                <td style={{ padding: "6px 12px" }}>
                                                    <Flex align="center" gap="1">
                                                        {isFastest && (
                                                            <span data-tooltip="Fastest">⚡</span>
                                                        )}
                                                        <span
                                                            style={{
                                                                fontWeight: isFastest ? 600 : 400,
                                                                color: isUnavailable
                                                                    ? "var(--qt-text-muted)"
                                                                    : undefined,
                                                            }}
                                                        >
                                                            {row.label}
                                                        </span>
                                                    </Flex>
                                                </td>
                                                <td style={{ padding: "6px 12px" }}>
                                                    {isUnavailable ? (
                                                        <span
                                                            className="hs-type-caption"
                                                            style={{
                                                                color: "var(--qt-danger-text)",
                                                            }}
                                                        >
                                                            {t("benchmark_failed")}
                                                        </span>
                                                    ) : (
                                                        formatMs(row.medianMs)
                                                    )}
                                                </td>
                                                <td
                                                    style={{
                                                        padding: "6px 12px",
                                                        color: isUnavailable
                                                            ? "var(--qt-text-muted)"
                                                            : row.rtf >= 1
                                                              ? "var(--qt-success-text)"
                                                              : "var(--qt-danger-text)",
                                                    }}
                                                >
                                                    {isUnavailable ? "N/A" : formatRtf(row.rtf)}
                                                </td>
                                            </tr>
                                        );
                                    })}
                                </tbody>
                            </table>
                        </div>
                        <span className="hs-type-caption">{t("benchmark_rtf_hint")}</span>
                        {fastestRow && fastestRow.available && (
                            <span className="hs-type-body">
                                {t("benchmark_recommended")} <strong>{fastestRow.label}</strong>
                            </span>
                        )}

                        {/* GPU (WebGPU) available but benchmark failed */}
                        {gpuFailed && (
                            <Flex
                                direction="column"
                                gap="2"
                                style={{
                                    padding: "8px 12px",
                                    borderRadius: 6,
                                    background: "var(--qt-danger-bg)",
                                    border: "1px solid var(--qt-danger-border)",
                                }}
                            >
                                <span
                                    className="hs-type-label font-medium"
                                    style={{ color: "var(--qt-danger-text)" }}
                                >
                                    {t("benchmark_gpu_failed_title")}
                                </span>
                                <span
                                    className="hs-type-caption"
                                    style={{ color: "var(--qt-danger-text)" }}
                                >
                                    {t("benchmark_gpu_failed_desc")}
                                </span>
                                {result.gpuError && (
                                    <span
                                        className="hs-type-mono"
                                        style={{
                                            color: "var(--qt-danger-text)",
                                            whiteSpace: "pre-wrap",
                                            wordBreak: "break-word",
                                        }}
                                    >
                                        {result.gpuError}
                                    </span>
                                )}
                            </Flex>
                        )}

                        {/* DirectML available but benchmark failed */}
                        {dmlFailed && (
                            <Flex
                                direction="column"
                                gap="2"
                                style={{
                                    padding: "8px 12px",
                                    borderRadius: 6,
                                    background: "var(--qt-danger-bg)",
                                    border: "1px solid var(--qt-danger-border)",
                                }}
                            >
                                <span
                                    className="hs-type-label font-medium"
                                    style={{ color: "var(--qt-danger-text)" }}
                                >
                                    {t("benchmark_gpu_failed_title")}
                                </span>
                                <span
                                    className="hs-type-caption"
                                    style={{ color: "var(--qt-danger-text)" }}
                                >
                                    {t("benchmark_gpu_failed_desc")}
                                </span>
                            </Flex>
                        )}

                        {/* Available providers */}
                        <span className="hs-type-caption" style={{ marginTop: 4 }}>
                            {t("benchmark_providers_label")}{" "}
                            {result.availableProviders.join(", ") || "unknown"}
                        </span>
                        <span className="hs-type-caption">
                            {t("benchmark_ort_info_label")} {result.ortBuildInfo || "unknown"}
                        </span>

                        {/* GPU enumeration */}
                        {result.gpuDevices && result.gpuDevices.length > 0 && (
                            <Flex direction="column" gap="1" style={{ marginTop: 4 }}>
                                <span className="hs-type-caption font-medium">
                                    {t("benchmark_gpu_label")}:
                                </span>
                                {result.gpuDevices.map((gpu) => (
                                    <span key={gpu.deviceId} className="hs-type-caption">
                                        · {t("benchmark_gpu_device_label")} {gpu.deviceId}:{" "}
                                        {gpu.name} ({(gpu.memoryMb / 1024).toFixed(1)} GB)
                                    </span>
                                ))}
                            </Flex>
                        )}
                    </Flex>
                )}

                {/* Error state */}
                {phase === "error" && (
                    <span className="hs-type-body" style={{ color: "var(--qt-danger-text)" }}>
                        {errorText || t("benchmark_error_default")}
                    </span>
                )}

                {/* Idle hint */}
                {phase === "idle" && (
                    <span className="hs-type-muted">{t("benchmark_idle_hint")}</span>
                )}
            </AppForm>
        </AppDialog>
    );
}
