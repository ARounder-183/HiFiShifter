import { Dialog, Flex, Select, Text, Button, TextField } from "@radix-ui/themes";
import { useState } from "react";
import { useAppDispatch, useAppSelector } from "../../app/hooks";
import type { RootState } from "../../app/store";
import { useI18n } from "../../i18n/I18nProvider";
import {
    setSplitTransitionMode,
    setSplitTransitionDurationUnit,
    setSplitTransitionDurationSec,
    setSplitTransitionDurationPercent,
    setSplitTransitionCurve,
    setSplitTransitionOverlapCrossfade,
    persistUiSettings,
} from "../../features/session/sessionSlice";
import type { FadeCurveType } from "../../features/session/sessionTypes";
import {
    isModifierActive,
    selectKeybinding,
} from "../../features/keybindings/keybindingsSlice";
import { applySelectWheelChange } from "../../utils/selectWheel";

interface Props {
    open: boolean;
    onOpenChange: (open: boolean) => void;
}

const CURVE_OPTIONS: Array<{ value: FadeCurveType; labelKey: string }> = [
    { value: "linear", labelKey: "fade_curve_linear" },
    { value: "sine", labelKey: "fade_curve_sine" },
    { value: "exponential", labelKey: "fade_curve_exponential" },
    { value: "logarithmic", labelKey: "fade_curve_logarithmic" },
    { value: "scurve", labelKey: "fade_curve_scurve" },
];

export function SplitTransitionSettingsDialog({ open, onOpenChange }: Props) {
    const dispatch = useAppDispatch();
    const paramFineAdjustKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.paramFineAdjust"),
    );
    const {
        splitTransitionMode,
        splitTransitionDurationUnit,
        splitTransitionDurationSec,
        splitTransitionDurationPercent,
        splitTransitionCurve,
        splitTransitionOverlapCrossfade,
    } = useAppSelector((state: RootState) => state.session);
    const { t } = useI18n();
    const tAny = t as (key: string) => string;
    const [durationInput, setDurationInput] = useState(
        splitTransitionDurationUnit === "percent"
            ? String(splitTransitionDurationPercent)
            : String(splitTransitionDurationSec),
    );

    function commitDuration() {
        const parsed = Number(durationInput);
        if (splitTransitionDurationUnit === "percent") {
            const value = Number.isFinite(parsed) ? parsed : 1;
            dispatch(setSplitTransitionDurationPercent(value));
        } else {
            const value = Number.isFinite(parsed) ? parsed : 0.01;
            dispatch(setSplitTransitionDurationSec(value));
        }
        void dispatch(persistUiSettings());
    }

    return (
        <Dialog.Root
            open={open}
            onOpenChange={(nextOpen) => {
                if (nextOpen) {
                    setDurationInput(
                        splitTransitionDurationUnit === "percent"
                            ? String(splitTransitionDurationPercent)
                            : String(splitTransitionDurationSec),
                    );
                }
                onOpenChange(nextOpen);
            }}
        >
            <Dialog.Content
                style={{ maxWidth: 420 }}
                onKeyDown={(e) => e.stopPropagation()}
            >
                <Dialog.Title>{tAny("split_transition_settings_title")}</Dialog.Title>

                <Flex direction="column" gap="3" mt="3">
                    <Text size="1" color="gray">
                        {tAny("split_transition_settings_desc")}
                    </Text>

                    <Flex align="center" gap="2">
                        <Text size="2" style={{ minWidth: 110 }}>
                            {tAny("split_transition_mode")}
                        </Text>
                        <Select.Root
                            value={splitTransitionMode}
                            size="2"
                            onValueChange={(v) => {
                                dispatch(setSplitTransitionMode(v as "fade" | "overlap"));
                                void dispatch(persistUiSettings());
                            }}
                        >
                            <Select.Trigger
                                style={{ flex: 1 }}
                                onWheel={(event) => {
                                    applySelectWheelChange({
                                        event,
                                        currentValue: splitTransitionMode,
                                        options: ["fade", "overlap"],
                                        onChange: (next) => {
                                            dispatch(
                                                setSplitTransitionMode(next),
                                            );
                                            void dispatch(persistUiSettings());
                                        },
                                    });
                                }}
                            />
                            <Select.Content>
                                <Select.Item value="fade">
                                    {tAny("split_transition_mode_fade")}
                                </Select.Item>
                                <Select.Item value="overlap">
                                    {tAny("split_transition_mode_overlap")}
                                </Select.Item>
                            </Select.Content>
                        </Select.Root>
                    </Flex>

                    <Flex align="center" gap="2">
                        <Text size="2" style={{ minWidth: 110 }}>
                            {tAny("split_transition_duration_unit_label")}
                        </Text>
                        <Select.Root
                            value={splitTransitionDurationUnit}
                            size="2"
                            onValueChange={(v) => {
                                const unit = v as "seconds" | "percent";
                                dispatch(setSplitTransitionDurationUnit(unit));
                                setDurationInput(
                                    unit === "percent"
                                        ? String(splitTransitionDurationPercent)
                                        : String(splitTransitionDurationSec),
                                );
                                void dispatch(persistUiSettings());
                            }}
                        >
                            <Select.Trigger
                                style={{ flex: 1 }}
                                onWheel={(event) => {
                                    applySelectWheelChange({
                                        event,
                                        currentValue: splitTransitionDurationUnit,
                                        options: ["seconds", "percent"],
                                        onChange: (next) => {
                                            dispatch(
                                                setSplitTransitionDurationUnit(next),
                                            );
                                            setDurationInput(
                                                next === "percent"
                                                    ? String(
                                                          splitTransitionDurationPercent,
                                                      )
                                                    : String(splitTransitionDurationSec),
                                            );
                                            void dispatch(persistUiSettings());
                                        },
                                    });
                                }}
                            />
                            <Select.Content>
                                <Select.Item value="seconds">
                                    {tAny("split_transition_duration_unit_seconds")}
                                </Select.Item>
                                <Select.Item value="percent">
                                    {tAny("split_transition_duration_unit_percent")}
                                </Select.Item>
                            </Select.Content>
                        </Select.Root>
                    </Flex>

                    <Flex align="center" gap="2">
                        <Text size="2" style={{ minWidth: 110 }}>
                            {tAny("split_transition_duration")}
                        </Text>
                        <TextField.Root
                            size="2"
                            type="number"
                            step={splitTransitionDurationUnit === "percent" ? "0.1" : "0.001"}
                            min={splitTransitionDurationUnit === "percent" ? "0.01" : "0.001"}
                            max={splitTransitionDurationUnit === "percent" ? "100" : "10"}
                            value={durationInput}
                            onChange={(e) => setDurationInput(e.target.value)}
                            onBlur={commitDuration}
                            onWheel={(e) => {
                                e.preventDefault();
                                e.stopPropagation();
                                const percentUnit =
                                    splitTransitionDurationUnit === "percent";
                                const fine = isModifierActive(
                                    paramFineAdjustKb,
                                    e.nativeEvent,
                                );
                                const step = percentUnit
                                    ? fine
                                        ? 0.1
                                        : 1
                                    : fine
                                      ? 0.001
                                      : 0.01;
                                const current = Number(durationInput);
                                if (!Number.isFinite(current)) return;
                                const direction = e.deltaY < 0 ? 1 : -1;
                                const notches = Math.max(
                                    1,
                                    Math.round(Math.abs(e.deltaY) / 100),
                                );
                                const min = percentUnit ? 0.01 : 0.001;
                                const max = percentUnit ? 100 : 10;
                                const next = Math.max(
                                    min,
                                    Math.min(max, current + direction * step * notches),
                                );
                                const rounded = percentUnit
                                    ? Math.round(next * 100) / 100
                                    : Math.round(next * 1000) / 1000;
                                setDurationInput(String(rounded));
                                if (percentUnit) {
                                    dispatch(setSplitTransitionDurationPercent(rounded));
                                } else {
                                    dispatch(setSplitTransitionDurationSec(rounded));
                                }
                                void dispatch(persistUiSettings());
                            }}
                            style={{ flex: 1 }}
                        />
                        <Text size="1" color="gray">
                            {tAny(
                                splitTransitionDurationUnit === "percent"
                                    ? "split_transition_duration_percent_unit"
                                    : "split_transition_duration_unit",
                            )}
                        </Text>
                    </Flex>

                    {splitTransitionDurationUnit === "percent" && (
                        <Text size="1" color="gray">
                            {tAny("split_transition_duration_percent_hint")}
                        </Text>
                    )}

                    <Flex align="center" gap="2">
                        <Text size="2" style={{ minWidth: 110 }}>
                            {tAny("split_transition_curve")}
                        </Text>
                        <Select.Root
                            value={splitTransitionCurve}
                            size="2"
                            onValueChange={(v) => {
                                dispatch(setSplitTransitionCurve(v as FadeCurveType));
                                void dispatch(persistUiSettings());
                            }}
                        >
                            <Select.Trigger
                                style={{ flex: 1 }}
                                onWheel={(event) => {
                                    applySelectWheelChange({
                                        event,
                                        currentValue: splitTransitionCurve,
                                        options: CURVE_OPTIONS.map((opt) => opt.value),
                                        onChange: (next) => {
                                            dispatch(
                                                setSplitTransitionCurve(next),
                                            );
                                            void dispatch(persistUiSettings());
                                        },
                                    });
                                }}
                            />
                            <Select.Content>
                                {CURVE_OPTIONS.map((opt) => (
                                    <Select.Item key={opt.value} value={opt.value}>
                                        {tAny(opt.labelKey)}
                                    </Select.Item>
                                ))}
                            </Select.Content>
                        </Select.Root>
                    </Flex>

                    <Flex align="center" gap="2">
                        <Text size="2" style={{ minWidth: 110 }}>
                            {tAny("split_transition_overlap_crossfade")}
                        </Text>
                        <Select.Root
                            value={splitTransitionOverlapCrossfade}
                            size="2"
                            onValueChange={(v) => {
                                dispatch(
                                    setSplitTransitionOverlapCrossfade(v as "auto" | "always"),
                                );
                                void dispatch(persistUiSettings());
                            }}
                        >
                            <Select.Trigger
                                style={{ flex: 1 }}
                                onWheel={(event) => {
                                    applySelectWheelChange({
                                        event,
                                        currentValue: splitTransitionOverlapCrossfade,
                                        options: ["auto", "always"],
                                        onChange: (next) => {
                                            dispatch(
                                                setSplitTransitionOverlapCrossfade(next),
                                            );
                                            void dispatch(persistUiSettings());
                                        },
                                    });
                                }}
                            />
                            <Select.Content>
                                <Select.Item value="auto">
                                    {tAny("split_transition_overlap_crossfade_auto")}
                                </Select.Item>
                                <Select.Item value="always">
                                    {tAny("split_transition_overlap_crossfade_always")}
                                </Select.Item>
                            </Select.Content>
                        </Select.Root>
                    </Flex>

                    {splitTransitionMode === "overlap" && (
                        <Text size="1" color="gray">
                            {tAny("split_transition_overlap_hint")}
                        </Text>
                    )}
                </Flex>

                <Flex justify="end" mt="4">
                    <Dialog.Close>
                        <Button variant="soft" color="gray" onClick={commitDuration}>
                            {tAny("ok")}
                        </Button>
                    </Dialog.Close>
                </Flex>
            </Dialog.Content>
        </Dialog.Root>
    );
}
