import { Checkbox, Flex, Text } from "@radix-ui/themes";
import { useAppDispatch, useAppSelector } from "../../app/hooks";
import type { RootState } from "../../app/store";
import { useI18n } from "../../i18n/I18nProvider";
import {
    persistUiSettings,
    setPrimaryTimeUnit,
    setRulerLabelSpacingPx,
    setSecondaryTimeUnit,
    setShowPlayheadTimeInTrackHeader,
} from "../../features/session/sessionSlice";
import type { TimeUnit, TimeUnitChoice } from "../../features/session/sessionTypes";
import { TIME_UNITS, TIME_UNIT_CHOICES } from "./timeline/timeFormat";
import { AppSelect, AppSlider, AppSliderReadout } from "../../ui";
import { AppDialog } from "../../ui/Dialog";
import { AppField, AppForm } from "../../ui/Field";

function unitLabelKey(unit: TimeUnit): string {
    switch (unit) {
        case "barBeats":
            return "time_unit_bar_beats";
        case "barDivisions":
            return "time_unit_bar_divisions";
        case "seconds":
            return "time_unit_seconds";
        case "clock":
            return "time_unit_clock";
    }
}

interface Props {
    open: boolean;
    onOpenChange: (open: boolean) => void;
}

export function TimelineDisplaySettingsDialog({ open, onOpenChange }: Props) {
    const dispatch = useAppDispatch();
    const s = useAppSelector((state: RootState) => state.session);
    const { tf } = useI18n();

    return (
        <AppDialog
            open={open}
            onOpenChange={onOpenChange}
            title={tf("timeline_display_settings")}
            description={tf("timeline_display_settings_desc")}
            size="sm"
            actions={[{ id: "close", label: tf("close"), onClick: () => onOpenChange(false) }]}
        >
            <AppForm>
                <AppField label={tf("time_unit_primary")}>
                    <AppSelect
                        value={s.primaryTimeUnit}
                        onValueChange={(v) => {
                            dispatch(setPrimaryTimeUnit(v as TimeUnit));
                            void dispatch(persistUiSettings());
                        }}
                        options={TIME_UNITS.map((unit) => ({
                            value: unit,
                            label: tf(unitLabelKey(unit)),
                        }))}
                    />
                </AppField>

                <AppField label={tf("time_unit_secondary")}>
                    <AppSelect
                        value={s.secondaryTimeUnit}
                        onValueChange={(v) => {
                            dispatch(setSecondaryTimeUnit(v as TimeUnitChoice));
                            void dispatch(persistUiSettings());
                        }}
                        options={TIME_UNIT_CHOICES.map((unit) => ({
                            value: unit,
                            label:
                                unit === "none"
                                    ? tf("time_unit_none")
                                    : tf(unitLabelKey(unit as TimeUnit)),
                        }))}
                    />
                </AppField>

                <AppField label={tf("ruler_label_spacing")}>
                    <Flex align="center" gap="2">
                        <AppSlider
                            value={s.rulerLabelSpacingPx}
                            unit="pixels"
                            min={40}
                            max={320}
                            ariaLabel={tf("ruler_label_spacing")}
                            onChange={(next) => {
                                dispatch(setRulerLabelSpacingPx(next));
                            }}
                            onCommit={() => void dispatch(persistUiSettings())}
                        />
                        <AppSliderReadout>{s.rulerLabelSpacingPx}px</AppSliderReadout>
                    </Flex>
                </AppField>

                <label className="flex items-center gap-2 cursor-pointer">
                    <Checkbox
                        size="2"
                        checked={s.showPlayheadTimeInTrackHeader}
                        onCheckedChange={(checked) => {
                            dispatch(setShowPlayheadTimeInTrackHeader(Boolean(checked)));
                            void dispatch(persistUiSettings());
                        }}
                    />
                    <Text size="2">{tf("show_playhead_time_in_track_header")}</Text>
                </label>
            </AppForm>
        </AppDialog>
    );
}
