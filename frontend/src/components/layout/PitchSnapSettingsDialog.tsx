import { Select, TextField } from "@radix-ui/themes";
import { useAppDispatch, useAppSelector } from "../../app/hooks";
import type { RootState } from "../../app/store";
import { useI18n } from "../../i18n/I18nProvider";
import {
    setPitchSnapUnit,
    setPitchSnapToleranceCents,
    persistUiSettings,
} from "../../features/session/sessionSlice";
import type { PitchSnapUnit } from "../../features/session/sessionTypes";
import { useEffect, useState } from "react";
import { applySelectWheelChange } from "../../utils/selectWheel";
import { AppDialog } from "../../ui/Dialog";
import { AppField, AppForm } from "../../ui/Field";

interface Props {
    open: boolean;
    onOpenChange: (open: boolean) => void;
}

export function PitchSnapSettingsDialog({ open, onOpenChange }: Props) {
    const dispatch = useAppDispatch();
    const { pitchSnapUnit, pitchSnapToleranceCents } = useAppSelector(
        (state: RootState) => state.session,
    );
    const { t } = useI18n();
    const tAny = t as (key: string) => string;
    const [toleranceInput, setToleranceInput] = useState(String(pitchSnapToleranceCents));

    useEffect(() => {
        if (open) {
            // eslint-disable-next-line react-hooks/set-state-in-effect -- 对话框打开时用 props 初始化局部 state（既有模式；重构会改变打开时序）
            setToleranceInput(String(pitchSnapToleranceCents));
        }
    }, [open, pitchSnapToleranceCents]);

    // 容差提交：输入后按 Enter / Esc / 点遮罩关闭也应生效（与
    // SplitTransitionSettingsDialog 的 onBlur 提交一致），不能只有点 OK
    // 才提交 —— 否则数字被静默丢弃。
    const commitTolerance = () => {
        const parsed = Math.abs(Math.round(Number(toleranceInput) || 0));
        dispatch(setPitchSnapToleranceCents(parsed));
        void dispatch(persistUiSettings());
    };

    return (
        <AppDialog
            open={open}
            onOpenChange={onOpenChange}
            title={tAny("pitch_snap_settings")}
            size="sm"
            actions={[
                {
                    id: "ok",
                    label: tAny("ok"),
                    intent: "primary",
                    onClick: () => {
                        commitTolerance();
                        onOpenChange(false);
                    },
                },
            ]}
        >
            <AppForm>
                {/* Quantize Unit */}
                <AppField label={tAny("quantize_unit")}>
                    <Select.Root
                        value={pitchSnapUnit}
                        size="2"
                        onValueChange={(v) => {
                            dispatch(setPitchSnapUnit(v as PitchSnapUnit));
                            void dispatch(persistUiSettings());
                        }}
                    >
                        <Select.Trigger
                            onWheel={(event) => {
                                applySelectWheelChange({
                                    event,
                                    currentValue: pitchSnapUnit,
                                    options: ["semitone", "scale"],
                                    onChange: (next) => {
                                        dispatch(setPitchSnapUnit(next as PitchSnapUnit));
                                        void dispatch(persistUiSettings());
                                    },
                                });
                            }}
                        />
                        <Select.Content>
                            <Select.Item value="semitone">{tAny("quantize_semitone")}</Select.Item>
                            <Select.Item value="scale">{tAny("quantize_scale")}</Select.Item>
                        </Select.Content>
                    </Select.Root>
                </AppField>

                <AppField label={tAny("pitch_snap_tolerance")}>
                    <TextField.Root
                        size="2"
                        type="number"
                        value={toleranceInput}
                        onChange={(e) => setToleranceInput(e.target.value)}
                        onBlur={commitTolerance}
                        onKeyDown={(e) => {
                            if (e.key === "Enter") {
                                e.currentTarget.blur();
                            }
                        }}
                    />
                </AppField>
            </AppForm>
        </AppDialog>
    );
}
