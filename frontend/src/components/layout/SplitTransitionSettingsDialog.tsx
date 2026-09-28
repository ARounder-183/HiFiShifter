import { Flex, Text } from "@radix-ui/themes";
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
import type { SplitTransitionCurveType } from "../../features/session/sessionTypes";
import { FADE_PRESETS } from "./timeline/reaperFade";
import { SHAPE_LABEL_KEYS } from "./timeline/fadeTooltipText";
import { AppNumberField, AppSelect } from "../../ui";
import { AppDialog } from "../../ui/Dialog";
import { AppField, AppForm } from "../../ui/Field";

interface Props {
    open: boolean;
    onOpenChange: (open: boolean) => void;
}

// 新版“分割过渡淡化曲线”枚举：最开头是“不修改淡化曲线”（保留原 Clip
// 曲线类型，默认值），其后按 REAPER 菜单顺序列出七预设（FADE_PRESETS）。
const CURVE_OPTIONS: Array<{ value: SplitTransitionCurveType; labelKey: string }> = [
    { value: "keep", labelKey: "split_transition_curve_keep" },
    ...FADE_PRESETS.map((preset) => ({
        value: preset.id,
        labelKey: SHAPE_LABEL_KEYS[preset.shape] ?? "fade_shape_linear",
    })),
];

export function SplitTransitionSettingsDialog({ open, onOpenChange }: Props) {
    const dispatch = useAppDispatch();
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

    const isPercent = splitTransitionDurationUnit === "percent";

    return (
        <AppDialog
            open={open}
            onOpenChange={onOpenChange}
            title={tAny("split_transition_settings_title")}
            size="sm"
            actions={[
                {
                    id: "ok",
                    label: tAny("ok"),
                    intent: "primary",
                    onClick: () => {
                        onOpenChange(false);
                    },
                },
            ]}
        >
            <AppForm>
                <Text size="1" color="gray">
                    {tAny("split_transition_settings_desc")}
                </Text>

                <AppField label={tAny("split_transition_mode")}>
                    <AppSelect
                        value={splitTransitionMode}
                        onValueChange={(v) => {
                            dispatch(setSplitTransitionMode(v as "fade" | "overlap"));
                            void dispatch(persistUiSettings());
                        }}
                        options={[
                            { value: "fade", label: tAny("split_transition_mode_fade") },
                            { value: "overlap", label: tAny("split_transition_mode_overlap") },
                        ]}
                    />
                </AppField>

                <AppField label={tAny("split_transition_duration_unit_label")}>
                    <AppSelect
                        value={splitTransitionDurationUnit}
                        onValueChange={(v) => {
                            const unit = v as "seconds" | "percent";
                            dispatch(setSplitTransitionDurationUnit(unit));
                            void dispatch(persistUiSettings());
                        }}
                        options={[
                            {
                                value: "seconds",
                                label: tAny("split_transition_duration_unit_seconds"),
                            },
                            {
                                value: "percent",
                                label: tAny("split_transition_duration_unit_percent"),
                            },
                        ]}
                    />
                </AppField>

                <AppField label={tAny("split_transition_duration")}>
                    <Flex align="center" gap="2">
                        <AppNumberField
                            value={
                                isPercent
                                    ? splitTransitionDurationPercent
                                    : splitTransitionDurationSec
                            }
                            unit={isPercent ? "percentFine" : "seconds"}
                            min={isPercent ? 0.01 : 0.001}
                            max={isPercent ? 100 : 10}
                            width={120}
                            className="flex-1"
                            ariaLabel={tAny("split_transition_duration")}
                            onCommit={(next) => {
                                if (isPercent) {
                                    dispatch(setSplitTransitionDurationPercent(next));
                                } else {
                                    dispatch(setSplitTransitionDurationSec(next));
                                }
                                void dispatch(persistUiSettings());
                            }}
                        />
                        <Text size="1" color="gray">
                            {tAny(
                                isPercent
                                    ? "split_transition_duration_percent_unit"
                                    : "split_transition_duration_unit",
                            )}
                        </Text>
                    </Flex>
                </AppField>

                {isPercent && (
                    <Text size="1" color="gray">
                        {tAny("split_transition_duration_percent_hint")}
                    </Text>
                )}

                <AppField label={tAny("split_transition_curve")}>
                    <AppSelect
                        value={splitTransitionCurve}
                        onValueChange={(v) => {
                            dispatch(setSplitTransitionCurve(v as SplitTransitionCurveType));
                            void dispatch(persistUiSettings());
                        }}
                        options={CURVE_OPTIONS.map((opt) => ({
                            value: opt.value,
                            label: tAny(opt.labelKey),
                        }))}
                    />
                </AppField>

                <AppField label={tAny("split_transition_overlap_crossfade")}>
                    <AppSelect
                        value={splitTransitionOverlapCrossfade}
                        onValueChange={(v) => {
                            dispatch(setSplitTransitionOverlapCrossfade(v as "auto" | "always"));
                            void dispatch(persistUiSettings());
                        }}
                        options={[
                            {
                                value: "auto",
                                label: tAny("split_transition_overlap_crossfade_auto"),
                            },
                            {
                                value: "always",
                                label: tAny("split_transition_overlap_crossfade_always"),
                            },
                        ]}
                    />
                </AppField>

                {splitTransitionMode === "overlap" && (
                    <Text size="1" color="gray">
                        {tAny("split_transition_overlap_hint")}
                    </Text>
                )}
            </AppForm>
        </AppDialog>
    );
}
