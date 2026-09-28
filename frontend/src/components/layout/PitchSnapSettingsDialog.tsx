import { useAppDispatch, useAppSelector } from "../../app/hooks";
import type { RootState } from "../../app/store";
import { useI18n } from "../../i18n/I18nProvider";
import {
    setPitchSnapUnit,
    setPitchSnapToleranceCents,
    persistUiSettings,
} from "../../features/session/sessionSlice";
import type { PitchSnapUnit } from "../../features/session/sessionTypes";
import { AppDialog } from "../../ui/Dialog";
import { AppField, AppForm } from "../../ui/Field";
import { AppNumberField, AppSelect } from "../../ui";

interface Props {
    open: boolean;
    onOpenChange: (open: boolean) => void;
}

export function PitchSnapSettingsDialog({ open, onOpenChange }: Props) {
    const dispatch = useAppDispatch();
    const { pitchSnapUnit, pitchSnapToleranceCents } = useAppSelector(
        (state: RootState) => state.session,
    );
    const { tf } = useI18n();

    // 容差提交：输入后按 Enter / 点遮罩关闭也应生效（与
    // SplitTransitionSettingsDialog 的 onBlur 提交一致），不能只有点 OK
    // 才提交 —— 否则数字被静默丢弃。
    const commitTolerance = (parsed: number) => {
        dispatch(setPitchSnapToleranceCents(parsed));
        void dispatch(persistUiSettings());
    };

    return (
        <AppDialog
            open={open}
            onOpenChange={onOpenChange}
            title={tf("pitch_snap_settings")}
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
                {/* Quantize Unit */}
                <AppField label={tf("quantize_unit")}>
                    <AppSelect
                        value={pitchSnapUnit}
                        onValueChange={(v) => {
                            dispatch(setPitchSnapUnit(v as PitchSnapUnit));
                            void dispatch(persistUiSettings());
                        }}
                        options={[
                            { value: "semitone", label: tf("quantize_semitone") },
                            { value: "scale", label: tf("quantize_scale") },
                        ]}
                    />
                </AppField>

                <AppField label={tf("pitch_snap_tolerance")}>
                    <AppNumberField
                        value={pitchSnapToleranceCents}
                        unit="cents"
                        ariaLabel={tf("pitch_snap_tolerance")}
                        onCommit={commitTolerance}
                    />
                </AppField>
            </AppForm>
        </AppDialog>
    );
}
