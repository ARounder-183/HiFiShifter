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
    const { t } = useI18n();
    const tAny = t as (key: string) => string;
    const [cents, setCents] = useState("0");
    const [smoothness, setSmoothness] = useDialogDraft(open, () =>
        String(Math.round(defaultSmoothness)),
    );

    return (
        <AppDialog
            open={open}
            onOpenChange={onOpenChange}
            title={tAny("menu_transpose_cents")}
            size="sm"
            actions={[
                { id: "cancel", label: tAny("cancel"), onClick: () => onOpenChange(false) },
                {
                    id: "apply",
                    label: tAny("ok"),
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
                <AppField label={tAny("dlg_cents")}>
                    <AppNumberField
                        value={Number(cents)}
                        unit="cents"
                        ariaLabel={tAny("dlg_cents")}
                        onCommit={(next) => setCents(String(next))}
                    />
                </AppField>
                <AppField label={tAny("edge_smoothness")}>
                    <Flex align="center" gap="2">
                        <AppSlider
                            value={Math.round(Number(smoothness) || 0)}
                            unit="percent"
                            min={0}
                            max={100}
                            ariaLabel={tAny("edge_smoothness")}
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
    const { t } = useI18n();
    const tAny = t as (key: string) => string;
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
                projectScaleLabel ?? tAny("project_scale_generic"),
                customScalePresets,
            ),
        [projectScaleLabel, customScalePresets, tAny],
    );

    return (
        <AppDialog
            open={open}
            onOpenChange={onOpenChange}
            title={tAny("menu_transpose_degrees")}
            size="sm"
            actions={[
                { id: "cancel", label: tAny("cancel"), onClick: () => onOpenChange(false) },
                {
                    id: "apply",
                    label: tAny("ok"),
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
                <AppField label={tAny("transpose_degrees_amount")}>
                    <AppNumberField
                        value={Number(degrees)}
                        unit="integer"
                        ariaLabel={tAny("transpose_degrees_amount")}
                        onCommit={(next) => setDegrees(String(next))}
                    />
                </AppField>
                <AppField label={tAny("base_scale")}>
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
                <AppField label={tAny("edge_smoothness")}>
                    <Flex align="center" gap="2">
                        <AppSlider
                            value={Math.round(Number(smoothness) || 0)}
                            unit="percent"
                            min={0}
                            max={100}
                            ariaLabel={tAny("edge_smoothness")}
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
    const { t } = useI18n();
    const tAny = t as (key: string) => string;
    const [note, setNote] = useDialogDraft(open, () => String(defaultValue));
    const [smoothness, setSmoothness] = useDialogDraft(open, () =>
        String(Math.round(defaultSmoothness)),
    );

    return (
        <AppDialog
            open={open}
            onOpenChange={onOpenChange}
            title={titleText ?? tAny("menu_set_pitch")}
            size="sm"
            actions={[
                { id: "cancel", label: tAny("cancel"), onClick: () => onOpenChange(false) },
                {
                    id: "apply",
                    label: tAny("ok"),
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
                <AppField label={valueLabelText ?? tAny("dlg_midi_note")}>
                    <AppNumberField
                        value={Number(note)}
                        unit="semitone"
                        ariaLabel={valueLabelText ?? tAny("dlg_midi_note")}
                        onCommit={(next) => setNote(String(next))}
                    />
                </AppField>
                <AppField label={tAny("edge_smoothness")}>
                    <Flex align="center" gap="2">
                        <AppSlider
                            value={Math.round(Number(smoothness) || 0)}
                            unit="percent"
                            min={0}
                            max={100}
                            ariaLabel={tAny("edge_smoothness")}
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
    const { t } = useI18n();
    const tAny = t as (key: string) => string;
    const [strength, setStrength] = useState("100");

    return (
        <AppDialog
            open={open}
            onOpenChange={onOpenChange}
            title={tAny("menu_average")}
            size="sm"
            actions={[
                { id: "cancel", label: tAny("cancel"), onClick: () => onOpenChange(false) },
                {
                    id: "apply",
                    label: tAny("ok"),
                    intent: "primary",
                    onClick: () => {
                        onConfirm?.(Math.max(0, Math.min(100, Math.round(Number(strength) || 0))));
                        onOpenChange(false);
                    },
                },
            ]}
        >
            <AppForm>
                <AppField label={tAny("dlg_average_strength")}>
                    <Flex align="center" gap="2">
                        <AppSlider
                            value={Math.round(Number(strength) || 0)}
                            unit="percent"
                            min={0}
                            max={100}
                            ariaLabel={tAny("dlg_average_strength")}
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
    const { t } = useI18n();
    const tAny = t as (key: string) => string;
    const [strength, setStrength] = useDialogDraft(open, () =>
        Math.max(0, Math.min(100, Math.round(defaultSmoothness))),
    );

    return (
        <AppDialog
            open={open}
            onOpenChange={onOpenChange}
            title={tAny("menu_smooth")}
            size="sm"
            actions={[
                { id: "cancel", label: tAny("cancel"), onClick: () => onOpenChange(false) },
                {
                    id: "apply",
                    label: tAny("ok"),
                    intent: "primary",
                    onClick: () => {
                        onConfirm?.(Math.max(0, Math.min(100, Math.round(strength))));
                        onOpenChange(false);
                    },
                },
            ]}
        >
            <AppForm>
                <AppField label={tAny("dlg_smoothness")}>
                    <Flex align="center" gap="2">
                        <AppSlider
                            value={Math.round(strength)}
                            unit="percent"
                            min={0}
                            max={100}
                            ariaLabel={tAny("dlg_smoothness")}
                            onChange={(next) => setStrength(next)}
                        />
                        <AppSliderReadout>{Math.round(strength)}%</AppSliderReadout>
                    </Flex>
                </AppField>
            </AppForm>
        </AppDialog>
    );
}

interface VibratoProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    editParam?: string;
    /** 当前参数的值域（用于自动钳制振幅默认值） */
    paramRange?: { min: number; max: number };
    onConfirm?: (
        amplitude: number,
        rate: number,
        attack: number,
        release: number,
        phase: number,
    ) => void;
}

export function VibratoDialog({ open, onOpenChange, onConfirm, editParam }: VibratoProps) {
    const { t } = useI18n();
    const tAny = t as (key: string) => string;

    const isPitch = editParam === "pitch";
    // 气声音量（breath_gain，0..2）按原值域给小振幅；dyn 落到默认 30 ——
    // 它在 op 侧是**深度百分比**（±30% 乘性调制，静音保持静音）。
    // 【注意】不能按"值域 0..2"来识别 breath_gain：dyn 的值域也是 0..2。
    const isBreathGain = editParam === "breath_gain";
    const defaultAmplitude = isPitch ? "30" : isBreathGain ? "1" : "30";

    // 对话框常驻挂载：草稿在每次「打开」时按当前参数重新播种（useDialogDraft），
    // 否则对 breath_gain 打开时仍显示 pitch 的 30。
    const [amplitude, setAmplitude] = useDialogDraft<string>(open, () => defaultAmplitude);
    const [rate, setRate] = useDialogDraft(open, () => "5.5");
    const [attack, setAttack] = useDialogDraft(open, () => "50");
    const [release, setRelease] = useDialogDraft(open, () => "50");
    const [phase, setPhase] = useDialogDraft(open, () => "0");

    // 兜底 NaN/Infinity 而不吞掉合法的 0（`Number(x) || default` 会把 0
    // 替换成默认值——对幅度/速率/相位，0 都是合法输入）。
    const parseNumberOr = (raw: string, fallback: number): number => {
        const value = Number(raw);
        return Number.isFinite(value) ? value : fallback;
    };

    return (
        <AppDialog
            open={open}
            onOpenChange={onOpenChange}
            title={tAny("menu_add_vibrato")}
            size="sm"
            actions={[
                { id: "cancel", label: tAny("cancel"), onClick: () => onOpenChange(false) },
                {
                    id: "apply",
                    label: tAny("ok"),
                    intent: "primary",
                    onClick: () => {
                        onConfirm?.(
                            parseNumberOr(amplitude, 30),
                            parseNumberOr(rate, 5.5),
                            parseNumberOr(attack, 50),
                            parseNumberOr(release, 50),
                            parseNumberOr(phase, 0),
                        );
                        onOpenChange(false);
                    },
                },
            ]}
        >
            <AppForm>
                <AppField label={isPitch ? tAny("dlg_amplitude_cents") : tAny("dlg_amplitude")}>
                    <AppNumberField
                        value={Number(amplitude)}
                        unit={isPitch ? "cents" : "integer"}
                        ariaLabel={isPitch ? tAny("dlg_amplitude_cents") : tAny("dlg_amplitude")}
                        onCommit={(next) => setAmplitude(String(next))}
                    />
                </AppField>
                <AppField label={tAny("dlg_rate_hz")}>
                    <AppNumberField
                        value={Number(rate)}
                        unit="rate"
                        ariaLabel={tAny("dlg_rate_hz")}
                        onCommit={(next) => setRate(String(next))}
                    />
                </AppField>
                <AppField label={tAny("dlg_attack_ms")}>
                    <AppNumberField
                        value={Number(attack)}
                        unit="milliseconds"
                        ariaLabel={tAny("dlg_attack_ms")}
                        onCommit={(next) => setAttack(String(next))}
                    />
                </AppField>
                <AppField label={tAny("dlg_release_ms")}>
                    <AppNumberField
                        value={Number(release)}
                        unit="milliseconds"
                        ariaLabel={tAny("dlg_release_ms")}
                        onCommit={(next) => setRelease(String(next))}
                    />
                </AppField>
                <AppField label={tAny("dlg_phase_deg")}>
                    <AppNumberField
                        value={Number(phase)}
                        unit="integer"
                        ariaLabel={tAny("dlg_phase_deg")}
                        onCommit={(next) => setPhase(String(next))}
                    />
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
    const { t } = useI18n();
    const tAny = t as (key: string) => string;
    const toleranceDefault = defaultTolerance ?? defaultToleranceCents;
    const [unit, setUnit] = useState<"semitone" | "scale">("semitone");
    const [scaleValue, setScaleValue] = useDialogDraft<string>(open, () =>
        defaultUseProjectScale ? "__project__" : defaultScale,
    );
    const customScalePresets = useAppSelector((state) => state.session.customScalePresets);
    const scaleSelectGroups = useMemo(
        () =>
            buildScaleSelectGroups(
                projectScaleLabel ?? tAny("project_scale_generic"),
                customScalePresets,
            ),
        [projectScaleLabel, customScalePresets, tAny],
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
            title={tAny("menu_quantize")}
            size="sm"
            actions={[
                { id: "cancel", label: tAny("cancel"), onClick: () => onOpenChange(false) },
                {
                    id: "apply",
                    label: tAny("ok"),
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
                    <AppField label={tAny("quantize_unit")}>
                        <AppSelect
                            value={unit}
                            onValueChange={(v) => setUnit(v as "semitone" | "scale")}
                            options={[
                                { value: "semitone", label: tAny("quantize_semitone") },
                                { value: "scale", label: tAny("quantize_scale") },
                            ]}
                        />
                    </AppField>
                )}
                {!valueMode && unit === "scale" && (
                    <AppField label={tAny("base_scale")}>
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
                    <AppField label={tAny("quantize_unit")}>
                        <AppNumberField
                            value={Number(quantizeUnit)}
                            unit="integer"
                            ariaLabel={tAny("quantize_unit")}
                            onCommit={(next) => setQuantizeUnit(String(next))}
                        />
                    </AppField>
                )}
                <AppField
                    label={valueMode ? tAny("quantize_tolerance") : tAny("pitch_snap_tolerance")}
                >
                    <AppNumberField
                        value={Number(toleranceCents)}
                        unit="cents"
                        ariaLabel={
                            valueMode ? tAny("quantize_tolerance") : tAny("pitch_snap_tolerance")
                        }
                        onCommit={(next) => setToleranceCents(String(next))}
                    />
                </AppField>
                <AppField label={tAny("edge_smoothness")}>
                    <Flex align="center" gap="2">
                        <AppSlider
                            value={Math.round(Number(smoothness) || 0)}
                            unit="percent"
                            min={0}
                            max={100}
                            ariaLabel={tAny("edge_smoothness")}
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
    const { t } = useI18n();
    const tAny = t as (key: string) => string;
    const toleranceDefault = defaultTolerance ?? defaultToleranceCents;
    const [unit, setUnit] = useState<"semitone" | "scale">("semitone");
    const [scaleValue, setScaleValue] = useDialogDraft<string>(open, () =>
        defaultUseProjectScale ? "__project__" : defaultScale,
    );
    const customScalePresets = useAppSelector((state) => state.session.customScalePresets);
    const scaleSelectGroups = useMemo(
        () =>
            buildScaleSelectGroups(
                projectScaleLabel ?? tAny("project_scale_generic"),
                customScalePresets,
            ),
        [projectScaleLabel, customScalePresets, tAny],
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
            title={tAny("mean_quantize_title")}
            size="sm"
            actions={[
                { id: "cancel", label: tAny("cancel"), onClick: () => onOpenChange(false) },
                {
                    id: "apply",
                    label: tAny("ok"),
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
                    <AppField label={tAny("quantize_unit")}>
                        <AppSelect
                            value={unit}
                            onValueChange={(v) => setUnit(v as "semitone" | "scale")}
                            options={[
                                { value: "semitone", label: tAny("quantize_semitone") },
                                { value: "scale", label: tAny("quantize_scale") },
                            ]}
                        />
                    </AppField>
                )}
                {!valueMode && unit === "scale" && (
                    <AppField label={tAny("base_scale")}>
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
                    <AppField label={tAny("quantize_unit")}>
                        <AppNumberField
                            value={Number(quantizeUnit)}
                            unit="integer"
                            ariaLabel={tAny("quantize_unit")}
                            onCommit={(next) => setQuantizeUnit(String(next))}
                        />
                    </AppField>
                )}
                <AppField
                    label={valueMode ? tAny("quantize_tolerance") : tAny("pitch_snap_tolerance")}
                >
                    <AppNumberField
                        value={Number(toleranceCents)}
                        unit="cents"
                        ariaLabel={
                            valueMode ? tAny("quantize_tolerance") : tAny("pitch_snap_tolerance")
                        }
                        onCommit={(next) => setToleranceCents(String(next))}
                    />
                </AppField>
                <AppField label={tAny("edge_smoothness")}>
                    <Flex align="center" gap="2">
                        <AppSlider
                            value={Math.round(Number(smoothness) || 0)}
                            unit="percent"
                            min={0}
                            max={100}
                            ariaLabel={tAny("edge_smoothness")}
                            onChange={(next) => setSmoothness(String(next))}
                        />
                        <AppSliderReadout>{Math.round(Number(smoothness) || 0)}%</AppSliderReadout>
                    </Flex>
                </AppField>
            </AppForm>
        </AppDialog>
    );
}
