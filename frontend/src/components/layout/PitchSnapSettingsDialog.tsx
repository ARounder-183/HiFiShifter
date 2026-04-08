import { Dialog, Flex, Select, Text, Button, TextField } from "@radix-ui/themes";
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
            setToleranceInput(String(pitchSnapToleranceCents));
        }
    }, [open, pitchSnapToleranceCents]);

    return (
        <Dialog.Root open={open} onOpenChange={onOpenChange}>
            <Dialog.Content style={{ maxWidth: 360 }} onKeyDown={(e) => e.stopPropagation()}>
                <Dialog.Title>{tAny("pitch_snap_settings")}</Dialog.Title>

                <Flex direction="column" gap="3" mt="3">
                    {/* Quantize Unit */}
                    <Flex align="center" gap="2">
                        <Text size="2" style={{ minWidth: 80 }}>
                            {tAny("quantize_unit")}
                        </Text>
                        <Select.Root
                            value={pitchSnapUnit}
                            size="2"
                            onValueChange={(v) => {
                                dispatch(setPitchSnapUnit(v as PitchSnapUnit));
                                void dispatch(persistUiSettings());
                            }}
                        >
                            <Select.Trigger
                                style={{ flex: 1 }}
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
                                <Select.Item value="semitone">
                                    {tAny("quantize_semitone")}
                                </Select.Item>
                                <Select.Item value="scale">{tAny("quantize_scale")}</Select.Item>
                            </Select.Content>
                        </Select.Root>
                    </Flex>

                    <Flex align="center" gap="2">
                        <Text size="2" style={{ minWidth: 80 }}>
                            {tAny("pitch_snap_tolerance")}
                        </Text>
                        <TextField.Root
                            size="2"
                            type="number"
                            value={toleranceInput}
                            onChange={(e) => setToleranceInput(e.target.value)}
                            style={{ flex: 1 }}
                        />
                    </Flex>
                </Flex>

                <Flex justify="end" mt="4">
                    <Dialog.Close>
                        <Button
                            variant="soft"
                            color="gray"
                            onClick={() => {
                                const parsed = Math.abs(Math.round(Number(toleranceInput) || 0));
                                dispatch(setPitchSnapToleranceCents(parsed));
                                void dispatch(persistUiSettings());
                            }}
                        >
                            {tAny("ok")}
                        </Button>
                    </Dialog.Close>
                </Flex>
            </Dialog.Content>
        </Dialog.Root>
    );
}
