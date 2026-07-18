/**
 * Inference Device Benchmark Dialog
 *
 * Runs the backend run_vocoder_benchmark command which tests CPU, GPU (OpenCL),
 * and GPU (DirectML) inference latency (median over 1024 frames / ~12 s of audio)
 * and displays the results so the user can pick the fastest provider for their system.
 */

import { useEffect, useRef, useState } from "react";
import { Button, Dialog, Flex, Text, Spinner } from "@radix-ui/themes";
import type { BenchmarkResult } from "../../types/api";
import { coreApi } from "../../services/api/core";
import { useI18n } from "../../i18n/I18nProvider";

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

    // GPU (OpenCL)
    if (result.gpuMedianMs != null && result.gpuRtFactor != null) {
        rows.push({
            label: "GPU (OpenCL)",
            medianMs: result.gpuMedianMs,
            rtf: result.gpuRtFactor,
            available: true,
        });
    } else if (result.gpuAvailable) {
        rows.push({
            label: "GPU (OpenCL)",
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
        <Dialog.Root
            open={open}
            onOpenChange={(o) => {
                if (!o) handleClose();
            }}
        >
            <Dialog.Content style={{ maxWidth: 520 }} onKeyDown={(e) => e.stopPropagation()}>
                <Dialog.Title>{t("benchmark_title")}</Dialog.Title>
                <Dialog.Description size="2" color="gray">
                    {t("benchmark_desc")}
                </Dialog.Description>

                <Flex direction="column" gap="3" mt="4">
                    {/* Running state */}
                    {phase === "running" && (
                        <Flex align="center" gap="2">
                            <Spinner size="2" />
                            <Text size="2" color="gray">
                                {t("benchmark_running")}
                            </Text>
                        </Flex>
                    )}

                    {/* Results table */}
                    {phase === "done" && result && rows.length > 0 && (
                        <Flex direction="column" gap="2">
                            <Text size="2" weight="medium">
                                {t("benchmark_results").replace(
                                    "{samples}",
                                    String(result.benchmarkSamples),
                                )}
                            </Text>
                            <div
                                style={{
                                    borderRadius: 6,
                                    overflow: "hidden",
                                    border: "1px solid var(--gray-5)",
                                }}
                            >
                                <table
                                    style={{
                                        width: "100%",
                                        borderCollapse: "collapse",
                                        fontSize: 13,
                                    }}
                                >
                                    <thead>
                                        <tr
                                            style={{
                                                background: "var(--gray-3)",
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
                                                        borderTop: "1px solid var(--gray-4)",
                                                    }}
                                                >
                                                    <td style={{ padding: "6px 12px" }}>
                                                        <Flex align="center" gap="1">
                                                            {isFastest && (
                                                                <span title="Fastest">⚡</span>
                                                            )}
                                                            <span
                                                                style={{
                                                                    fontWeight: isFastest
                                                                        ? 600
                                                                        : 400,
                                                                    color: isUnavailable
                                                                        ? "var(--gray-9)"
                                                                        : undefined,
                                                                }}
                                                            >
                                                                {row.label}
                                                            </span>
                                                        </Flex>
                                                    </td>
                                                    <td style={{ padding: "6px 12px" }}>
                                                        {isUnavailable ? (
                                                            <Text size="1" color="red">
                                                                {t("benchmark_failed")}
                                                            </Text>
                                                        ) : (
                                                            formatMs(row.medianMs)
                                                        )}
                                                    </td>
                                                    <td
                                                        style={{
                                                            padding: "6px 12px",
                                                            color: isUnavailable
                                                                ? "var(--gray-9)"
                                                                : row.rtf >= 1
                                                                  ? "var(--green-10)"
                                                                  : "var(--red-10)",
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
                            <Text size="1" color="gray">
                                {t("benchmark_rtf_hint")}
                            </Text>
                            {fastestRow && fastestRow.available && (
                                <Text size="2">
                                    {t("benchmark_recommended")} <strong>{fastestRow.label}</strong>
                                </Text>
                            )}

                            {/* GPU (OpenCL) available but benchmark failed */}
                            {gpuFailed && (
                                <Flex
                                    direction="column"
                                    gap="2"
                                    style={{
                                        padding: "8px 12px",
                                        borderRadius: 6,
                                        background: "var(--red-3)",
                                        border: "1px solid var(--red-5)",
                                    }}
                                >
                                    <Text
                                        size="2"
                                        weight="medium"
                                        style={{ color: "var(--red-10)" }}
                                    >
                                        {t("benchmark_gpu_failed_title")}
                                    </Text>
                                    <Text size="1" style={{ color: "var(--red-9)" }}>
                                        {t("benchmark_gpu_failed_desc")}
                                    </Text>
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
                                        background: "var(--red-3)",
                                        border: "1px solid var(--red-5)",
                                    }}
                                >
                                    <Text
                                        size="2"
                                        weight="medium"
                                        style={{ color: "var(--red-10)" }}
                                    >
                                        {t("benchmark_gpu_failed_title")}
                                    </Text>
                                    <Text size="1" style={{ color: "var(--red-9)" }}>
                                        {t("benchmark_gpu_failed_desc")}
                                    </Text>
                                </Flex>
                            )}

                            {/* Available providers */}
                            <Text size="1" style={{ color: "var(--gray-9)", marginTop: 4 }}>
                                {t("benchmark_providers_label")}{" "}
                                {result.availableProviders.join(", ") || "unknown"}
                            </Text>
                            <Text size="1" style={{ color: "var(--gray-9)" }}>
                                {t("benchmark_ort_info_label")}{" "}
                                {result.ortBuildInfo || "unknown"}
                            </Text>

                            {/* GPU enumeration */}
                            {result.gpuDevices && result.gpuDevices.length > 0 && (
                                <Flex direction="column" gap="1" style={{ marginTop: 4 }}>
                                    <Text
                                        size="1"
                                        weight="medium"
                                        style={{ color: "var(--gray-9)" }}
                                    >
                                        {t("benchmark_gpu_label")}:
                                    </Text>
                                    {result.gpuDevices.map((gpu) => (
                                        <Text
                                            key={gpu.deviceId}
                                            size="1"
                                            style={{ color: "var(--gray-9)" }}
                                        >
                                            · {t("benchmark_gpu_device_label")} {gpu.deviceId}:{" "}
                                            {gpu.name} ({(gpu.memoryMb / 1024).toFixed(1)} GB)
                                        </Text>
                                    ))}
                                </Flex>
                            )}
                        </Flex>
                    )}

                    {/* Error state */}
                    {phase === "error" && (
                        <Text size="2" color="red">
                            {errorText || t("benchmark_error_default")}
                        </Text>
                    )}

                    {/* Idle hint */}
                    {phase === "idle" && (
                        <Text size="2" color="gray">
                            {t("benchmark_idle_hint")}
                        </Text>
                    )}
                </Flex>

                <Flex justify="end" gap="2" mt="4">
                    <Button variant="soft" color="gray" onClick={handleClose}>
                        {t("benchmark_close")}
                    </Button>
                    <Button onClick={() => void handleRun()} disabled={phase === "running"}>
                        {phase === "running" ? t("benchmark_running_btn") : t("benchmark_run_btn")}
                    </Button>
                </Flex>
            </Dialog.Content>
        </Dialog.Root>
    );
}
