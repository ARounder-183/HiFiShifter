import { Flex } from "@radix-ui/themes";
import { useAppDispatch, useAppSelector } from "../../app/hooks";
import { shallowEqual } from "react-redux";
import type { RootState } from "../../app/store";
import { useI18n } from "../../i18n/I18nProvider";
import {
    persistUiSettings,
    setPlayheadTimePlacement,
    setPrimaryTimeUnit,
    setRulerLabelSpacingPx,
    setSecondaryTimeUnit,
} from "../../features/session/sessionSlice";
import {
    PLAYHEAD_TIME_PLACEMENTS,
    playheadTimePlacementLabelKey,
} from "../../features/session/playheadTimePlacement";
import type { TimeUnit, TimeUnitChoice } from "../../features/session/sessionTypes";
import { TIME_UNITS, TIME_UNIT_CHOICES, timeUnitLabelKey } from "./timeline/timeFormat";
import { AppSelect, AppSlider, AppSliderReadout } from "../../ui";
import { AppDialog } from "../../ui/Dialog";
import { AppField, AppForm } from "../../ui/Field";

interface Props {
    open: boolean;
    onOpenChange: (open: boolean) => void;
}

export function TimelineDisplaySettingsDialog({ open, onOpenChange }: Props) {
    const dispatch = useAppDispatch();
    // 【为什么不用整片 `state.session`】这些设置对话框是**常驻挂载**的（只靠
    // `open` 控制显隐），而 `session` 的对象引用每次 dispatch 都会变 —— 整片订阅
    // 让它们在播放期间跟着 33 ms 的播放轮询一起重渲染。只选真正消费的字段，并用
    // `shallowEqual` 比较（与 `ActionBar` / `MenuBar` 同口径）。
    const s = useAppSelector(
        (state: RootState) => ({
            primaryTimeUnit: state.session.primaryTimeUnit,
            secondaryTimeUnit: state.session.secondaryTimeUnit,
            rulerLabelSpacingPx: state.session.rulerLabelSpacingPx,
            playheadTimePlacement: state.session.playheadTimePlacement,
        }),
        shallowEqual,
    );
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
                            label: tf(timeUnitLabelKey(unit)),
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
                                    : tf(timeUnitLabelKey(unit as TimeUnit)),
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

                {/* 【为什么是下拉而不是复选框】旧复选框只能表达"轨道头里显不显示"，
                    而读数的主实例一直在工具栏 —— 用户既挪不走也关不掉。位置是一个
                    五选一的放置策略，不是一个布尔。 */}
                <AppField label={tf("playhead_time_placement")}>
                    <AppSelect
                        value={s.playheadTimePlacement}
                        onValueChange={(value) => {
                            dispatch(
                                setPlayheadTimePlacement(
                                    value as (typeof PLAYHEAD_TIME_PLACEMENTS)[number],
                                ),
                            );
                            void dispatch(persistUiSettings());
                        }}
                        options={PLAYHEAD_TIME_PLACEMENTS.map((placement) => ({
                            value: placement,
                            label: tf(playheadTimePlacementLabelKey(placement)),
                        }))}
                    />
                </AppField>
            </AppForm>
        </AppDialog>
    );
}
