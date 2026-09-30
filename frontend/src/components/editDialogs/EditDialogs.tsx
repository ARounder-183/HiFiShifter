import { useState, useMemo } from "react";
import { Flex } from "@radix-ui/themes";
import { useI18n } from "../../i18n/I18nProvider";
import type { ScaleKey } from "../../utils/musicalScales";
import { useAppSelector } from "../../app/hooks";
import { buildScaleSelectGroups } from "../../utils/scaleSelection";
import { AppNumberField, AppSelect, AppSlider, AppSliderReadout, useDialogDraft } from "../../ui";
import { AppDialog } from "../../ui/Dialog";
import { AppField, AppForm } from "../../ui/Field";

interface Props {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    defaultSmoothness?: number;
    onConfirm?: (cents: number, edgeSmoothnessPercent: number) => void;
}

export function TransposeCentsDialog({
    open,
    onOpenChange,
    defaultSmoothness = 0,
    onConfirm,
}: Props) {
    const { tf } = useI18n();
    const [cents, setCents] = useState("0");
    const [smoothness, setSmoothness] = useDialogDraft(open, () =>
        String(Math.round(defaultSmoothness)),
    );

    return (
        <AppDialog
            open={open}
            onOpenChange={onOpenChange}
            title={tf("menu_transpose_cents")}
            size="sm"
            actions={[
                { id: "cancel", label: tf("cancel"), onClick: () => onOpenChange(false) },
                {
                    id: "apply",
                    label: tf("ok"),
                    intent: "primary",
                    onClick: () => {
                        onConfirm?.(
                            Number(cents) || 0,
                            Math.max(0, Math.min(100, Number(smoothness) || 0)),
                        );
                        onOpenChange(false);
                    },
                },
            ]}
        >
            <AppForm>
                <AppField label={tf("dlg_cents")}>
                    <AppNumberField
                        value={Number(cents)}
                        unit="cents"
                        ariaLabel={tf("dlg_cents")}
                        onCommit={(next) => setCents(String(next))}
                    />
                </AppField>
                <AppField label={tf("edge_smoothness")}>
                    <Flex align="center" gap="2">
                        <AppSlider
                            value={Math.round(Number(smoothness) || 0)}
                            unit="percent"
                            min={0}
                            max={100}
                            ariaLabel={tf("edge_smoothness")}
                            onChange={(next) => setSmoothness(String(next))}
                        />
                        <AppSliderReadout>{Math.round(Number(smoothness) || 0)}%</AppSliderReadout>
                    </Flex>
                </AppField>
            </AppForm>
        </AppDialog>
    );
}

interface TransposeDegreesProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    defaultScale?: ScaleKey;
    defaultUseProjectScale?: boolean;
    projectScaleLabel?: string;
    defaultSmoothness?: number;
    onConfirm?: (degrees: number, scaleValue: string, edgeSmoothnessPercent: number) => void;
}

export function TransposeDegreesDialog({
    open,
    onOpenChange,
    defaultScale = "C",
    defaultUseProjectScale = true,
    projectScaleLabel,
    defaultSmoothness = 0,
    onConfirm,
}: TransposeDegreesProps) {
    const { tf } = useI18n();
    const [degrees, setDegrees] = useState("3");
    const [scaleValue, setScaleValue] = useDialogDraft<string>(open, () =>
        defaultUseProjectScale ? "__project__" : defaultScale,
    );
    const [smoothness, setSmoothness] = useDialogDraft(open, () =>
        String(Math.round(defaultSmoothness)),
    );
    const customScalePresets = useAppSelector((state) => state.session.customScalePresets);
    const scaleSelectGroups = useMemo(
        () =>
            buildScaleSelectGroups(
                projectScaleLabel ?? tf("project_scale_generic"),
                customScalePresets,
            ),
        [projectScaleLabel, customScalePresets, tf],
    );

    return (
        <AppDialog
            open={open}
            onOpenChange={onOpenChange}
            title={tf("menu_transpose_degrees")}
            size="sm"
            actions={[
                { id: "cancel", label: tf("cancel"), onClick: () => onOpenChange(false) },
                {
                    id: "apply",
                    label: tf("ok"),
                    intent: "primary",
                    onClick: () => {
                        onConfirm?.(
                            Number(degrees) || 0,
                            scaleValue,
                            Math.max(0, Math.min(100, Number(smoothness) || 0)),
                        );
                        onOpenChange(false);
                    },
                },
            ]}
        >
            <AppForm>
                <AppField label={tf("transpose_degrees_amount")}>
                    <AppNumberField
                        value={Number(degrees)}
                        unit="integer"
                        ariaLabel={tf("transpose_degrees_amount")}
                        onCommit={(next) => setDegrees(String(next))}
                    />
                </AppField>
                <AppField label={tf("base_scale")}>
                    <AppSelect
                        value={scaleValue}
                        onValueChange={setScaleValue}
                        options={[
                            scaleSelectGroups.projectOption,
                            { separator: true },
                            ...scaleSelectGroups.builtinOptions,
                            ...(scaleSelectGroups.customOptions.length > 0
                                ? [{ separator: true } as const]
                                : []),
                            ...scaleSelectGroups.customOptions,
                        ]}
                    />
                </AppField>
                <AppField label={tf("edge_smoothness")}>
                    <Flex align="center" gap="2">
                        <AppSlider
                            value={Math.round(Number(smoothness) || 0)}
                            unit="percent"
                            min={0}
                            max={100}
                            ariaLabel={tf("edge_smoothness")}
                            onChange={(next) => setSmoothness(String(next))}
                        />
                        <AppSliderReadout>{Math.round(Number(smoothness) || 0)}%</AppSliderReadout>
                    </Flex>
                </AppField>
            </AppForm>
        </AppDialog>
    );
}

interface SetPitchProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    titleText?: string;
    valueLabelText?: string;
    defaultValue?: number;
    defaultSmoothness?: number;
    onConfirm?: (value: number, edgeSmoothnessPercent: number) => void;
}

export function SetPitchDialog({
    open,
    onOpenChange,
    titleText,
    valueLabelText,
    defaultValue = 60,
    defaultSmoothness = 0,
    onConfirm,
}: SetPitchProps) {
    const { tf } = useI18n();
    const [note, setNote] = useDialogDraft(open, () => String(defaultValue));
    const [smoothness, setSmoothness] = useDialogDraft(open, () =>
        String(Math.round(defaultSmoothness)),
    );

    return (
        <AppDialog
            open={open}
            onOpenChange={onOpenChange}
            title={titleText ?? tf("menu_set_pitch")}
            size="sm"
            actions={[
                { id: "cancel", label: tf("cancel"), onClick: () => onOpenChange(false) },
                {
                    id: "apply",
                    label: tf("ok"),
                    intent: "primary",
                    onClick: () => {
                        const parsedNote = Number(note);
                        const nextValue = Number.isFinite(parsedNote) ? parsedNote : defaultValue;
                        onConfirm?.(nextValue, Math.max(0, Math.min(100, Number(smoothness) || 0)));
                        onOpenChange(false);
                    },
                },
            ]}
        >
            <AppForm>
                <AppField label={valueLabelText ?? tf("dlg_midi_note")}>
                    <AppNumberField
                        value={Number(note)}
                        unit="semitone"
                        ariaLabel={valueLabelText ?? tf("dlg_midi_note")}
                        onCommit={(next) => setNote(String(next))}
                    />
                </AppField>
                <AppField label={tf("edge_smoothness")}>
                    <Flex align="center" gap="2">
                        <AppSlider
                            value={Math.round(Number(smoothness) || 0)}
                            unit="percent"
                            min={0}
                            max={100}
                            ariaLabel={tf("edge_smoothness")}
                            onChange={(next) => setSmoothness(String(next))}
                        />
                        <AppSliderReadout>{Math.round(Number(smoothness) || 0)}%</AppSliderReadout>
                    </Flex>
                </AppField>
            </AppForm>
        </AppDialog>
    );
}

interface AverageProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    onConfirm?: (strength: number) => void;
}

export function AverageDialog({ open, onOpenChange, onConfirm }: AverageProps) {
    const { tf } = useI18n();
    const [strength, setStrength] = useState("100");

    return (
        <AppDialog
            open={open}
            onOpenChange={onOpenChange}
            title={tf("menu_average")}
            size="sm"
            actions={[
                { id: "cancel", label: tf("cancel"), onClick: () => onOpenChange(false) },
                {
                    id: "apply",
                    label: tf("ok"),
                    intent: "primary",
                    onClick: () => {
                        onConfirm?.(Math.max(0, Math.min(100, Math.round(Number(strength) || 0))));
                        onOpenChange(false);
                    },
                },
            ]}
        >
            <AppForm>
                <AppField label={tf("dlg_average_strength")}>
                    <Flex align="center" gap="2">
                        <AppSlider
                            value={Math.round(Number(strength) || 0)}
                            unit="percent"
                            min={0}
                            max={100}
                            ariaLabel={tf("dlg_average_strength")}
                            onChange={(next) => setStrength(String(next))}
                        />
                        <AppSliderReadout>{Math.round(Number(strength) || 0)}%</AppSliderReadout>
                    </Flex>
                </AppField>
            </AppForm>
        </AppDialog>
    );
}

interface SmoothProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    defaultSmoothness?: number;
    onConfirm?: (strength: number) => void;
}

export function SmoothDialog({
    open,
    onOpenChange,
    defaultSmoothness = 50,
    onConfirm,
}: SmoothProps) {
    const { tf } = useI18n();
    const [strength, setStrength] = useDialogDraft(open, () =>
        Math.max(0, Math.min(100, Math.round(defaultSmoothness))),
    );

    return (
        <AppDialog
            open={open}
            onOpenChange={onOpenChange}
            title={tf("menu_smooth")}
            size="sm"
            actions={[
                { id: "cancel", label: tf("cancel"), onClick: () => onOpenChange(false) },
                {
                    id: "apply",
                    label: tf("ok"),
                    intent: "primary",
                    onClick: () => {
                        onConfirm?.(Math.max(0, Math.min(100, Math.round(strength))));
                        onOpenChange(false);
                    },
                },
            ]}
        >
            <AppForm>
                <AppField label={tf("dlg_smoothness")}>
                    <Flex align="center" gap="2">
                        <AppSlider
                            value={Math.round(strength)}
                            unit="percent"
                            min={0}
                            max={100}
                            ariaLabel={tf("dlg_smoothness")}
                            onChange={(next) => setStrength(next)}
                        />
                        <AppSliderReadout>{Math.round(strength)}%</AppSliderReadout>
                    </Flex>
                </AppField>
            </AppForm>
        </AppDialog>
    );
}

interface QuantizeProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    valueMode?: boolean;
    defaultQuantizeUnit?: number;
    defaultTolerance?: number;
    defaultScale?: ScaleKey;
    defaultUseProjectScale?: boolean;
    projectScaleLabel?: string;
    defaultToleranceCents?: number;
    defaultSmoothness?: number;
    onConfirm?: (
        unit: "semitone" | "scale" | "value",
        scaleValue: string,
        toleranceCents: number,
        quantizeUnit: number | undefined,
        edgeSmoothnessPercent: number,
    ) => void;
}

export function QuantizeDialog({
    open,
    onOpenChange,
    valueMode = false,
    defaultQuantizeUnit = 1,
    defaultTolerance = 0,
    defaultScale = "C",
    defaultUseProjectScale = true,
    projectScaleLabel,
    defaultToleranceCents = 0,
    defaultSmoothness = 0,
    onConfirm,
}: QuantizeProps) {
    const { tf } = useI18n();
    const toleranceDefault = defaultTolerance ?? defaultToleranceCents;
    const [unit, setUnit] = useState<"semitone" | "scale">("semitone");
    const [scaleValue, setScaleValue] = useDialogDraft<string>(open, () =>
        defaultUseProjectScale ? "__project__" : defaultScale,
    );
    const customScalePresets = useAppSelector((state) => state.session.customScalePresets);
    const scaleSelectGroups = useMemo(
        () =>
            buildScaleSelectGroups(
                projectScaleLabel ?? tf("project_scale_generic"),
                customScalePresets,
            ),
        [projectScaleLabel, customScalePresets, tf],
    );
    const [toleranceCents, setToleranceCents] = useDialogDraft(open, () =>
        String(toleranceDefault),
    );
    const [quantizeUnit, setQuantizeUnit] = useDialogDraft(open, () => String(defaultQuantizeUnit));
    const [smoothness, setSmoothness] = useDialogDraft(open, () =>
        String(Math.round(defaultSmoothness)),
    );

    return (
        <AppDialog
            open={open}
            onOpenChange={onOpenChange}
            title={tf("menu_quantize")}
            size="sm"
            actions={[
                { id: "cancel", label: tf("cancel"), onClick: () => onOpenChange(false) },
                {
                    id: "apply",
                    label: tf("ok"),
                    intent: "primary",
                    onClick: () => {
                        const parsed = Math.abs(Math.round(Number(toleranceCents) || 0));
                        const parsedUnit = Math.abs(Number(quantizeUnit) || 0);
                        onConfirm?.(
                            valueMode ? "value" : unit,
                            scaleValue,
                            parsed,
                            valueMode ? parsedUnit : undefined,
                            Math.max(0, Math.min(100, Number(smoothness) || 0)),
                        );
                        onOpenChange(false);
                    },
                },
            ]}
        >
            <AppForm>
                {!valueMode && (
                    <AppField label={tf("quantize_unit")}>
                        <AppSelect
                            value={unit}
                            onValueChange={(v) => setUnit(v as "semitone" | "scale")}
                            options={[
                                { value: "semitone", label: tf("quantize_semitone") },
                                { value: "scale", label: tf("quantize_scale") },
                            ]}
                        />
                    </AppField>
                )}
                {!valueMode && unit === "scale" && (
                    <AppField label={tf("base_scale")}>
                        <AppSelect
                            value={scaleValue}
                            onValueChange={setScaleValue}
                            options={[
                                scaleSelectGroups.projectOption,
                                { separator: true },
                                ...scaleSelectGroups.builtinOptions,
                                ...(scaleSelectGroups.customOptions.length > 0
                                    ? [{ separator: true } as const]
                                    : []),
                                ...scaleSelectGroups.customOptions,
                            ]}
                        />
                    </AppField>
                )}
                {valueMode && (
                    <AppField label={tf("quantize_unit")}>
                        <AppNumberField
                            value={Number(quantizeUnit)}
                            unit="integer"
                            ariaLabel={tf("quantize_unit")}
                            onCommit={(next) => setQuantizeUnit(String(next))}
                        />
                    </AppField>
                )}
                <AppField label={valueMode ? tf("quantize_tolerance") : tf("pitch_snap_tolerance")}>
                    <AppNumberField
                        value={Number(toleranceCents)}
                        unit="cents"
                        ariaLabel={
                            valueMode ? tf("quantize_tolerance") : tf("pitch_snap_tolerance")
                        }
                        onCommit={(next) => setToleranceCents(String(next))}
                    />
                </AppField>
                <AppField label={tf("edge_smoothness")}>
                    <Flex align="center" gap="2">
                        <AppSlider
                            value={Math.round(Number(smoothness) || 0)}
                            unit="percent"
                            min={0}
                            max={100}
                            ariaLabel={tf("edge_smoothness")}
                            onChange={(next) => setSmoothness(String(next))}
                        />
                        <AppSliderReadout>{Math.round(Number(smoothness) || 0)}%</AppSliderReadout>
                    </Flex>
                </AppField>
            </AppForm>
        </AppDialog>
    );
}

interface MeanQuantizeProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    valueMode?: boolean;
    defaultQuantizeUnit?: number;
    defaultTolerance?: number;
    defaultScale?: ScaleKey;
    defaultUseProjectScale?: boolean;
    projectScaleLabel?: string;
    defaultToleranceCents?: number;
    defaultSmoothness?: number;
    onConfirm?: (
        unit: "semitone" | "scale" | "value",
        scaleValue: string,
        toleranceCents: number,
        quantizeUnit: number | undefined,
        edgeSmoothnessPercent: number,
    ) => void;
}

export function MeanQuantizeDialog({
    open,
    onOpenChange,
    valueMode = false,
    defaultQuantizeUnit = 1,
    defaultTolerance = 0,
    defaultScale = "C",
    defaultUseProjectScale = true,
    projectScaleLabel,
    defaultToleranceCents = 0,
    defaultSmoothness = 0,
    onConfirm,
}: MeanQuantizeProps) {
    const { tf } = useI18n();
    const toleranceDefault = defaultTolerance ?? defaultToleranceCents;
    const [unit, setUnit] = useState<"semitone" | "scale">("semitone");
    const [scaleValue, setScaleValue] = useDialogDraft<string>(open, () =>
        defaultUseProjectScale ? "__project__" : defaultScale,
    );
    const customScalePresets = useAppSelector((state) => state.session.customScalePresets);
    const scaleSelectGroups = useMemo(
        () =>
            buildScaleSelectGroups(
                projectScaleLabel ?? tf("project_scale_generic"),
                customScalePresets,
            ),
        [projectScaleLabel, customScalePresets, tf],
    );
    const [toleranceCents, setToleranceCents] = useDialogDraft(open, () =>
        String(toleranceDefault),
    );
    const [quantizeUnit, setQuantizeUnit] = useDialogDraft(open, () => String(defaultQuantizeUnit));
    const [smoothness, setSmoothness] = useDialogDraft(open, () =>
        String(Math.round(defaultSmoothness)),
    );

    return (
        <AppDialog
            open={open}
            onOpenChange={onOpenChange}
            title={tf("mean_quantize_title")}
            size="sm"
            actions={[
                { id: "cancel", label: tf("cancel"), onClick: () => onOpenChange(false) },
                {
                    id: "apply",
                    label: tf("ok"),
                    intent: "primary",
                    onClick: () => {
                        const parsed = Math.abs(Math.round(Number(toleranceCents) || 0));
                        const parsedUnit = Math.abs(Number(quantizeUnit) || 0);
                        onConfirm?.(
                            valueMode ? "value" : unit,
                            scaleValue,
                            parsed,
                            valueMode ? parsedUnit : undefined,
                            Math.max(0, Math.min(100, Number(smoothness) || 0)),
                        );
                        onOpenChange(false);
                    },
                },
            ]}
        >
            <AppForm>
                {!valueMode && (
                    <AppField label={tf("quantize_unit")}>
                        <AppSelect
                            value={unit}
                            onValueChange={(v) => setUnit(v as "semitone" | "scale")}
                            options={[
                                { value: "semitone", label: tf("quantize_semitone") },
                                { value: "scale", label: tf("quantize_scale") },
                            ]}
                        />
                    </AppField>
                )}
                {!valueMode && unit === "scale" && (
                    <AppField label={tf("base_scale")}>
                        <AppSelect
                            value={scaleValue}
                            onValueChange={setScaleValue}
                            options={[
                                scaleSelectGroups.projectOption,
                                { separator: true },
                                ...scaleSelectGroups.builtinOptions,
                                ...(scaleSelectGroups.customOptions.length > 0
                                    ? [{ separator: true } as const]
                                    : []),
                                ...scaleSelectGroups.customOptions,
                            ]}
                        />
                    </AppField>
                )}
                {valueMode && (
                    <AppField label={tf("quantize_unit")}>
                        <AppNumberField
                            value={Number(quantizeUnit)}
                            unit="integer"
                            ariaLabel={tf("quantize_unit")}
                            onCommit={(next) => setQuantizeUnit(String(next))}
                        />
                    </AppField>
                )}
                <AppField label={valueMode ? tf("quantize_tolerance") : tf("pitch_snap_tolerance")}>
                    <AppNumberField
                        value={Number(toleranceCents)}
                        unit="cents"
                        ariaLabel={
                            valueMode ? tf("quantize_tolerance") : tf("pitch_snap_tolerance")
                        }
                        onCommit={(next) => setToleranceCents(String(next))}
                    />
                </AppField>
                <AppField label={tf("edge_smoothness")}>
                    <Flex align="center" gap="2">
                        <AppSlider
                            value={Math.round(Number(smoothness) || 0)}
                            unit="percent"
                            min={0}
                            max={100}
                            ariaLabel={tf("edge_smoothness")}
                            onChange={(next) => setSmoothness(String(next))}
                        />
                        <AppSliderReadout>{Math.round(Number(smoothness) || 0)}%</AppSliderReadout>
                    </Flex>
                </AppField>
            </AppForm>
        </AppDialog>
    );
}
