import { Button, Checkbox, Dialog, Flex, Select, Slider, Text } from "@radix-ui/themes";
import { useState } from "react";
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
import { isKernelRenderingEnabled, setKernelRenderingEnabled } from "./timeline/kernel/featureFlag";

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

    /**
     * 渲染内核总开关的当前勾选状态。
     *
     * 特殊说明：初始值是模块加载期读到的快照，不是响应式派生值——四个开关都在各
     * 面板模块加载时读取一次，运行期改 localStorage 不会重挂子树。用户在本次会话里
     * 改了它只有重启后才生效，因此下面与加载值比较后给出"需重启"提示。
     */
    const [kernelEnabled, setKernelEnabled] = useState(() => isKernelRenderingEnabled());
    /**
     * 加载时的值（判断"是否需要重启"的基准）。
     *
     * 特殊说明：用 state 而非 ref——它参与渲染（决定提示文案），而渲染期读
     * `ref.current` 违反 React 规则（`react-hooks/refs` 直接报错）。它只在首次
     * 渲染时定型，之后不再变化。
     */
    const [kernelEnabledAtLoad] = useState(kernelEnabled);
    /**
     * 当前勾选状态与**加载时**是否不同（用于显示"需重启"提示）。
     *
     * 【为什么是与加载值比较，而不是一个"改过没有"的布尔量】用布尔量时，用户点开
     * 再点回来（最终与加载值相同）仍会看到"重启后生效"——而实际上什么都不用做，
     * 提示是错的。与加载值比较后，还原即消失。
     */
    const kernelDirty = kernelEnabled !== kernelEnabledAtLoad;

    return (
        <Dialog.Root open={open} onOpenChange={onOpenChange}>
            <Dialog.Content maxWidth="420px" onKeyDown={(e) => e.stopPropagation()}>
                <Dialog.Title>{tAny("timeline_display_settings")}</Dialog.Title>
                <Dialog.Description>
                    <Text size="2" color="gray">
                        {tAny("timeline_display_settings_desc")}
                    </Text>
                </Dialog.Description>

                <Flex direction="column" gap="4" mt="4">
                    <Flex align="center" gap="2">
                        <Text size="2" style={{ minWidth: 118 }}>
                            {tAny("time_unit_primary")}
                        </Text>
                        <Select.Root
                            value={s.primaryTimeUnit}
                            size="2"
                            onValueChange={(v) => {
                                dispatch(setPrimaryTimeUnit(v as TimeUnit));
                                void dispatch(persistUiSettings());
                            }}
                        >
                            <Select.Trigger
                                style={{ flex: 1 }}
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
                    </Flex>

                    <Flex align="center" gap="2">
                        <Text size="2" style={{ minWidth: 118 }}>
                            {tAny("time_unit_secondary")}
                        </Text>
                        <Select.Root
                            value={s.secondaryTimeUnit}
                            size="2"
                            onValueChange={(v) => {
                                dispatch(setSecondaryTimeUnit(v as TimeUnitChoice));
                                void dispatch(persistUiSettings());
                            }}
                        >
                            <Select.Trigger
                                style={{ flex: 1 }}
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
                    </Flex>

                    <Flex align="center" gap="2">
                        <Text size="2" style={{ minWidth: 118 }}>
                            {tAny("ruler_label_spacing")}
                        </Text>
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

                    {/* ── 渲染内核总开关（逃生门）──────────────────────────────
                        为什么放在设置界面：内核开关原本只认 localStorage，而打包版
                        没有 devtools（`backend/src-tauri/Cargo.toml` 的 tauri 未启用
                        `devtools` feature，release 下 wry 亦默认关闭），用户**没有
                        任何入口**能关掉内核。某个驱动上出问题时只能等新版本——这使
                        "逃生门"名不副实。这个复选框就是那个入口。

                        为什么改完要重启：四个开关都在各面板**模块加载时**读取一次
                        （见 `featureFlag` 文件头），运行期改变量不会重挂子树。这里
                        只提示、不自动刷新——工程可能含未保存编辑，而本工程没有脏标记
                        或 `beforeunload` 保护，替用户丢弃是错的。 */}
                    <label className="flex items-start gap-2 cursor-pointer">
                        <Checkbox
                            size="2"
                            style={{ marginTop: 2 }}
                            checked={kernelEnabled}
                            onCheckedChange={(checked) => {
                                const next = Boolean(checked);
                                // 写四层开关（localStorage），下次启动生效。
                                setKernelRenderingEnabled(next);
                                // 只更新本地显示，不尝试运行期热切换（见上方说明）。
                                // 提示文案由「与加载值是否不同」派生，点回来即消失。
                                setKernelEnabled(next);
                            }}
                        />
                        <Flex direction="column" gap="1">
                            <Text size="2">{tAny("render_kernel_enabled")}</Text>
                            <Text size="1" color="gray">
                                {kernelDirty
                                    ? tAny("render_kernel_restart_required")
                                    : tAny("render_kernel_enabled_desc")}
                            </Text>
                        </Flex>
                    </label>
                </Flex>

                <Flex justify="end" mt="4">
                    <Dialog.Close>
                        <Button variant="soft" color="gray">
                            {tAny("close")}
                        </Button>
                    </Dialog.Close>
                </Flex>
            </Dialog.Content>
        </Dialog.Root>
    );
}
