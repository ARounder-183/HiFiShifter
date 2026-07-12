/**
 * Inference Device Benchmark Dialog
 *
 * Runs the backend run_vocoder_benchmark command which tests CPU and GPU
 * inference latency (median over 1024 frames / ~12 s of audio) and displays
 * the results so the user can pick the fastest provider for their system.
 */

import { useEffect, useRef, useState } from "react";
import { Button, Dialog, Flex, Text, Spinner } from "@radix-ui/themes";
import type { BenchmarkResult } from "../../types/api";
import { coreApi } from "../../services/api/core";

interface BenchmarkDialogProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
}

type BenchmarkPhase = "idle" | "running" | "done" | "error";

function formatMs(ms: number): string {
    return `${ms.toFixed(1)} ms`;
}

function formatRtf(rtf: number): string {
    // RTF < 1 means faster than real-time; display as e.g. "0.12×"
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
    if (result.gpuMedianMs != null && result.gpuRtFactor != null) {
        rows.push({
            label: "GPU (CUDA)",
            medianMs: result.gpuMedianMs,
            rtf: result.gpuRtFactor,
            available: true,
        });
    }
    return rows;
}

export function BenchmarkDialog({ open, onOpenChange }: BenchmarkDialogProps) {
    const [phase, setPhase] = useState<BenchmarkPhase>("idle");
    const [result, setResult] = useState<BenchmarkResult | null>(null);
    const [errorText, setErrorText] = useState<string>("");
    const abortRef = useRef(false);

    // Reset state whenever the dialog opens
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
    const fastestRow = rows.length > 0
        ? rows.reduce((best, r) => (r.medianMs < best.medianMs ? r : best), rows[0])
        : null;

    return (
        <Dialog.Root open={open} onOpenChange={(o) => { if (!o) handleClose(); }}>
            <Dialog.Content
                style={{ maxWidth: 480 }}
                onKeyDown={(e) => e.stopPropagation()}
            >
                <Dialog.Title>Inference Device Benchmark</Dialog.Title>
                <Dialog.Description size="2" color="gray">
                    Runs ~12 s of audio through the vocoder on each available device and
                    reports median latency. Use the result to pick the fastest provider
                    in the Stretch → Inference Device menu.
                </Dialog.Description>

                <Flex direction="column" gap="3" mt="4">
                    {/* Running state */}
                    {phase === "running" && (
                        <Flex align="center" gap="2">
                            <Spinner size="2" />
                            <Text size="2" color="gray">
                                Running benchmark — this may take 20–60 seconds…
                            </Text>
                        </Flex>
                    )}

                    {/* Results table */}
                    {phase === "done" && result && rows.length > 0 && (
                        <Flex direction="column" gap="2">
                            <Text size="2" weight="medium">
                                Results ({result.benchmarkSamples} frame chunks · 44.1 kHz):
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
                                                Device
                                            </th>
                                            <th style={{ padding: "6px 12px", fontWeight: 500 }}>
                                                Median latency
                                            </th>
                                            <th style={{ padding: "6px 12px", fontWeight: 500 }}>
                                                RTF
                                            </th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {rows.map((row) => {
                                            const isFastest = row.label === fastestRow?.label;
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
                                                                    fontWeight: isFastest ? 600 : 400,
                                                                }}
                                                            >
                                                                {row.label}
                                                            </span>
                                                        </Flex>
                                                    </td>
                                                    <td style={{ padding: "6px 12px" }}>
                                                        {formatMs(row.medianMs)}
                                                    </td>
                                                    <td
                                                        style={{
                                                            padding: "6px 12px",
                                                            color:
                                                                row.rtf < 1
                                                                    ? "var(--green-10)"
                                                                    : "var(--red-10)",
                                                        }}
                                                    >
                                                        {formatRtf(row.rtf)}
                                                    </td>
                                                </tr>
                                            );
                                        })}
                                    </tbody>
                                </table>
                            </div>
                            <Text size="1" color="gray">
                                RTF &lt; 1× = faster than real-time. ⚡ = fastest available
                                device.
                            </Text>
                            {fastestRow && (
                                <Text size="2">
                                    Recommended: <strong>{fastestRow.label}</strong>
                                </Text>
                            )}
                        </Flex>
                    )}

                    {/* Error state */}
                    {phase === "error" && (
                        <Text size="2" color="red">
                            {errorText || "Benchmark failed. Ensure a model is loaded."}
                        </Text>
                    )}

                    {/* Idle hint */}
                    {phase === "idle" && (
                        <Text size="2" color="gray">
                            Click "Run Benchmark" to start. A model must be loaded first.
                        </Text>
                    )}
                </Flex>

                <Flex justify="end" gap="2" mt="4">
                    <Button variant="soft" color="gray" onClick={handleClose}>
                        Close
                    </Button>
                    <Button
                        onClick={() => void handleRun()}
                        disabled={phase === "running"}
                    >
                        {phase === "running" ? "Running…" : "Run Benchmark"}
                    </Button>
                </Flex>
            </Dialog.Content>
        </Dialog.Root>
    );
}
