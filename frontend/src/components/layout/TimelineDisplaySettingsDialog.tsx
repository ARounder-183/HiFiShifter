import { Checkbox, Flex, Select, Slider, Text } from "@radix-ui/themes";
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
import { applySelectWheelChange } from "../../utils/selectWheel";
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
    const { t } = useI18n();
    const tAny = t as (key: string) => string;

    return (
        <AppDialog
            open={open}
            onOpenChange={onOpenChange}
            title={tAny("timeline_display_settings")}
            description={
                <Text size="2" color="gray">
                    {tAny("timeline_display_settings_desc")}
                </Text>
            }
            size="sm"
            actions={[{ id: "close", label: tAny("close"), onClick: () => onOpenChange(false) }]}
        >
            <AppForm>
                <AppField label={tAny("time_unit_primary")}>
                    <Select.Root
                        value={s.primaryTimeUnit}
                        size="2"
                        onValueChange={(v) => {
                            dispatch(setPrimaryTimeUnit(v as TimeUnit));
                            void dispatch(persistUiSettings());
                        }}
                    >
                        <Select.Trigger
                            onWheel={(event) => {
                                applySelectWheelChange({
                                    event,
                                    currentValue: s.primaryTimeUnit,
                                    options: TIME_UNITS as readonly string[],
                                    onChange: (next) => {
                                        dispatch(setPrimaryTimeUnit(next as TimeUnit));
                                        void dispatch(persistUiSettings());
                                    },
                                });
                            }}
                        />
                        <Select.Content>
                            {TIME_UNITS.map((unit) => (
                                <Select.Item key={unit} value={unit}>
                                    {tAny(unitLabelKey(unit))}
                                </Select.Item>
                            ))}
                        </Select.Content>
                    </Select.Root>
                </AppField>

                <AppField label={tAny("time_unit_secondary")}>
                    <Select.Root
                        value={s.secondaryTimeUnit}
                        size="2"
                        onValueChange={(v) => {
                            dispatch(setSecondaryTimeUnit(v as TimeUnitChoice));
                            void dispatch(persistUiSettings());
                        }}
                    >
                        <Select.Trigger
                            onWheel={(event) => {
                                applySelectWheelChange({
                                    event,
                                    currentValue: s.secondaryTimeUnit,
                                    options: TIME_UNIT_CHOICES as readonly string[],
                                    onChange: (next) => {
                                        dispatch(setSecondaryTimeUnit(next as TimeUnitChoice));
                                        void dispatch(persistUiSettings());
                                    },
                                });
                            }}
                        />
                        <Select.Content>
                            {TIME_UNIT_CHOICES.map((unit) => (
                                <Select.Item key={unit} value={unit}>
                                    {unit === "none"
                                        ? tAny("time_unit_none")
                                        : tAny(unitLabelKey(unit as TimeUnit))}
                                </Select.Item>
                            ))}
                        </Select.Content>
                    </Select.Root>
                </AppField>

                <AppField label={tAny("ruler_label_spacing")}>
                    <Flex align="center" gap="2">
                        <Slider
                            size="1"
                            min={40}
                            max={320}
                            step={5}
                            value={[s.rulerLabelSpacingPx]}
                            onValueChange={(values: number[]) => {
                                dispatch(setRulerLabelSpacingPx(values[0]));
                            }}
                            onValueCommit={() => void dispatch(persistUiSettings())}
                            className="flex-1"
                        />
                        <Text size="1" color="gray" className="w-[36px] text-right shrink-0">
                            {s.rulerLabelSpacingPx}px
                        </Text>
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
                    <Text size="2">{tAny("show_playhead_time_in_track_header")}</Text>
                </label>
            </AppForm>
        </AppDialog>
    );
}
