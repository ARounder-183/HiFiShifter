import { Flex } from "@radix-ui/themes";
import { useAppDispatch, useAppSelector } from "../../app/hooks";
import { shallowEqual } from "react-redux";
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
    } = useAppSelector(
        (state: RootState) => ({
            splitTransitionEnabled: state.session.splitTransitionEnabled,
            splitTransitionMode: state.session.splitTransitionMode,
            splitTransitionDurationUnit: state.session.splitTransitionDurationUnit,
            splitTransitionDurationSec: state.session.splitTransitionDurationSec,
            splitTransitionDurationPercent: state.session.splitTransitionDurationPercent,
            splitTransitionCurve: state.session.splitTransitionCurve,
            splitTransitionOverlapCrossfade: state.session.splitTransitionOverlapCrossfade,
        }),
        shallowEqual,
    );
    const { tf } = useI18n();

    const isPercent = splitTransitionDurationUnit === "percent";

    return (
        <AppDialog
            open={open}
            onOpenChange={onOpenChange}
            title={tf("split_transition_settings_title")}
            size="sm"
            actions={[
                {
                    id: "ok",
                    label: tf("ok"),
                    intent: "primary",
                    onClick: () => {
                        onOpenChange(false);
                    },
                },
            ]}
        >
            <AppForm>
                <span className="hs-type-muted">{tf("split_transition_settings_desc")}</span>

                <AppField label={tf("split_transition_mode")}>
                    <AppSelect
                        value={splitTransitionMode}
                        onValueChange={(v) => {
                            dispatch(setSplitTransitionMode(v as "fade" | "overlap"));
                            void dispatch(persistUiSettings());
                        }}
                        options={[
                            { value: "fade", label: tf("split_transition_mode_fade") },
                            { value: "overlap", label: tf("split_transition_mode_overlap") },
                        ]}
                    />
                </AppField>

                <AppField label={tf("split_transition_duration_unit_label")}>
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
                                label: tf("split_transition_duration_unit_seconds"),
                            },
                            {
                                value: "percent",
                                label: tf("split_transition_duration_unit_percent"),
                            },
                        ]}
                    />
                </AppField>

                <AppField label={tf("split_transition_duration")}>
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
                            ariaLabel={tf("split_transition_duration")}
                            onCommit={(next) => {
                                if (isPercent) {
                                    dispatch(setSplitTransitionDurationPercent(next));
                                } else {
                                    dispatch(setSplitTransitionDurationSec(next));
                                }
                                void dispatch(persistUiSettings());
                            }}
                        />
                        <span className="hs-type-caption">
                            {tf(
                                isPercent
                                    ? "split_transition_duration_percent_unit"
                                    : "split_transition_duration_unit",
                            )}
                        </span>
                    </Flex>
                </AppField>

                {isPercent && (
                    <span className="hs-type-caption">
                        {tf("split_transition_duration_percent_hint")}
                    </span>
                )}

                <AppField label={tf("split_transition_curve")}>
                    <AppSelect
                        value={splitTransitionCurve}
                        onValueChange={(v) => {
                            dispatch(setSplitTransitionCurve(v as SplitTransitionCurveType));
                            void dispatch(persistUiSettings());
                        }}
                        options={CURVE_OPTIONS.map((opt) => ({
                            value: opt.value,
                            label: tf(opt.labelKey),
                        }))}
                    />
                </AppField>

                <AppField label={tf("split_transition_overlap_crossfade")}>
                    <AppSelect
                        value={splitTransitionOverlapCrossfade}
                        onValueChange={(v) => {
                            dispatch(setSplitTransitionOverlapCrossfade(v as "auto" | "always"));
                            void dispatch(persistUiSettings());
                        }}
                        options={[
                            {
                                value: "auto",
                                label: tf("split_transition_overlap_crossfade_auto"),
                            },
                            {
                                value: "always",
                                label: tf("split_transition_overlap_crossfade_always"),
                            },
                        ]}
                    />
                </AppField>

                {splitTransitionMode === "overlap" && (
                    <span className="hs-type-caption">{tf("split_transition_overlap_hint")}</span>
                )}
            </AppForm>
        </AppDialog>
    );
}
