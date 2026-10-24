/**
 * 参数工具栏里**由后端描述符驱动**的那一串药丸（含两个复合组）。
 *
 * 三条分支：
 * - `formant_shift_cents`：共振峰位移有"轨道级 / 子轨道级"两个参数，合成一个
 *   带下拉的药丸（`ParamGroupButton`）；
 * - `volume`：音量与动态同量纲、同混音级，合成一个药丸 + 下拉，眼睛对整组生效；
 * - 其余参数：普通 `ParamToolbarPill`，外加"效果参数门禁"（Compose 关闭 / 气声
 *   分离开关闭时置灰不可编辑）与"气声组首的开关"（分离开关的唯一入口）。
 *
 * 【为什么单独成文件】这段 map 里有三套互不相同的渲染规则，而且**渲染顺序**就是
 * 工具栏从左到右的顺序（`orderedProcessorParams` 已按音高 → 算法参数 → 音量/声像
 * 排好）。它夹在参考轨道组、子轨道音高组和算法下拉之间时，工具栏的可读性是断的；
 * 拆出来后"参数按钮怎么来"只剩这一个地方。
 *
 * 【门禁的两级让位】第 3/4 级工具栏宽度不够时，被门禁的药丸按 `gatedPillHideOrder`
 * 逐个移入槽位（第 0 个在第 3 级、第 1 个在第 4 级），组首的药丸**也让位**，但它
 * 左侧的分离开关保留 —— 它是重新打开气声/张力的唯一入口。
 *
 * 让位一律走 `ToolbarSlot`（只改 `display`），**绝不 `return null`**：层级是按内容
 * 宽度量出来的，内容若随层级变化就成了自反馈 —— 藏一个 → 变窄 → 不用藏 → 又显示，
 * 每帧一轮，用户看到的就是控件反复横跳（见 `toolbarOverflow` 模块头的铁律）。
 */
import React from "react";
import { isEffectParamGated, SEPARATION_PARAM_ID } from "../../../../features/pitch/separationGate";
import { setEditParam } from "../../../../features/session/sessionSlice";
import { BreathAirIcon, ParamGroupButton, ParamToolbarPill, ToolbarSlot } from "./toolbarParts";
import { gatedPillHideAt } from "../toolbarOverflow";
import type { PianoRollPanelModel } from "./usePianoRollPanelModel";

/** 本文件用到的模型字段（AST 扫描得出）；`Pick<>` 保证它不会悄悄变宽。 */
type Props = {
    m: Pick<
        PianoRollPanelModel,
        | "childFormantOffsetParam"
        | "composeEnabled"
        | "dispatch"
        | "editParam"
        | "gatedPillHideOrder"
        | "getProcessorParamLabel"
        | "getProcessorParamShortLabel"
        | "handleStaticParamChange"
        | "orderedProcessorParams"
        | "processorParamsRef"
        | "secondaryParamVisible"
        | "selectedIsChildTrack"
        | "separationEnabled"
        | "separationSwitchAnchorParamId"
        | "t"
        | "toggleSecondaryParam"
        | "toolbarTier"
    >;
};

export function ParamButtons({ m }: Props) {
    const {
        childFormantOffsetParam,
        composeEnabled,
        dispatch,
        editParam,
        gatedPillHideOrder,
        getProcessorParamLabel,
        getProcessorParamShortLabel,
        handleStaticParamChange,
        orderedProcessorParams,
        processorParamsRef,
        secondaryParamVisible,
        selectedIsChildTrack,
        separationEnabled,
        separationSwitchAnchorParamId,
        t,
        toolbarTier,
        toggleSecondaryParam,
    } = m;

    return (
        <>
            {/* 由后端 processorParams 驱动的动态参数按钮（按算法排列后的顺序） */}
            {orderedProcessorParams.map((p) => {
                // 第 3 / 4 级：被门禁的药丸逐个让位（顺序见 `gatedParamHideOrder`）。
                // 层级由 `gatedPillHideAt` 换算，槽位常挂载、只改 `display` —— 见文件头
                // 关于"绝不 return null"的说明。
                //
                // 【分离开启时不参与让位】此时这两个药丸是**活的功能入口**，而第 4–10 级
                // 之间没有别的入口（「⋯」要到第 10 级才出现）—— 让位等于让参数暂时消失。
                // 这个判据只看"分离开关"，与层级无关，因此不会重新引入自反馈。
                const gatedHideIndex = gatedPillHideOrder.indexOf(p.id);
                const gatedHideAt =
                    gatedHideIndex < 0 || separationEnabled
                        ? null
                        : gatedPillHideAt(gatedHideIndex);
                /** 把受门禁的可让位元素包进槽位；不受门禁时原样返回（Fragment 只为保住 key）。 */
                const withGatedSlot = (node: React.ReactNode) =>
                    gatedHideAt === null ? (
                        <React.Fragment key={p.id}>{node}</React.Fragment>
                    ) : (
                        <ToolbarSlot key={p.id} hideAt={gatedHideAt} tier={toolbarTier}>
                            {node}
                        </ToolbarSlot>
                    );
                if (p.id === "formant_shift_cents") {
                    return (
                        <ParamGroupButton
                            key={p.id}
                            rootParamId={p.id}
                            rootLabel={getProcessorParamShortLabel(p)}
                            rootMenuLabel={t("child_formant_root_option")}
                            rootTooltip={getProcessorParamLabel(p)}
                            childParamId={selectedIsChildTrack ? childFormantOffsetParam : null}
                            childLabel={t("child_formant_mode")}
                            childMenuLabel={t("child_formant_offset_option")}
                            rootActive={editParam === p.id}
                            childActive={editParam === childFormantOffsetParam}
                            secondaryVisible={secondaryParamVisible[p.id] ?? false}
                            hideSecondaryLabel={t("hide_secondary_param")}
                            showSecondaryLabel={t("show_secondary_param")}
                            hideSecondaryTooltip={t("secondary_overlay_tooltip_hidden")}
                            showSecondaryTooltip={t("secondary_overlay_tooltip_visible")}
                            onSelectRoot={() => dispatch(setEditParam(p.id))}
                            onSelectChild={() => {
                                if (childFormantOffsetParam) {
                                    dispatch(setEditParam(childFormantOffsetParam));
                                }
                            }}
                            onToggleSecondary={() => toggleSecondaryParam(p.id)}
                        />
                    );
                }

                // 音量 / 动态：同量纲的两个混音级参数，用一个药丸 + 下拉切换。
                // 只在遇到 "volume" 时渲染这一次（"dyn" 会被跳过），
                // 否则会得到两个内容相同的按钮。
                if (p.id === "volume") {
                    const hasDyn = orderedProcessorParams.some((q) => q.id === "dyn");
                    const volDesc = processorParamsRef.current.find((d) => d.id === "volume");
                    const dynDesc = processorParamsRef.current.find((d) => d.id === "dyn");
                    const groupSecondaryVisible =
                        (secondaryParamVisible["volume"] ?? false) ||
                        (secondaryParamVisible["dyn"] ?? false);
                    return (
                        <ParamGroupButton
                            key="volume-group"
                            rootParamId="volume"
                            rootLabel={
                                volDesc
                                    ? getProcessorParamShortLabel(volDesc)
                                    : t("param_btn_volume")
                            }
                            rootMenuLabel={t("volume_label")}
                            rootTooltip={t("volume_label")}
                            childParamId={hasDyn ? "dyn" : null}
                            childLabel={
                                dynDesc ? getProcessorParamShortLabel(dynDesc) : t("param_btn_dyn")
                            }
                            childMenuLabel={t("dyn_label")}
                            rootActive={editParam === "volume"}
                            childActive={editParam === "dyn"}
                            secondaryVisible={groupSecondaryVisible}
                            hideSecondaryLabel={t("hide_secondary_param")}
                            showSecondaryLabel={t("show_secondary_param")}
                            hideSecondaryTooltip={t("secondary_overlay_tooltip_hidden")}
                            showSecondaryTooltip={t("secondary_overlay_tooltip_visible")}
                            alwaysShowDropdown
                            onSelectRoot={() => dispatch(setEditParam("volume"))}
                            onSelectChild={() => {
                                if (hasDyn) dispatch(setEditParam("dyn"));
                            }}
                            onToggleSecondary={() => {
                                // 眼睛对整组生效：音量与动态的副参数叠加一起切换，
                                // 否则会出现"眼睛亮着但只有一条叠加线"的分裂观感。
                                if (groupSecondaryVisible) {
                                    if (secondaryParamVisible["volume"]) {
                                        toggleSecondaryParam("volume");
                                    }
                                    if (secondaryParamVisible["dyn"]) {
                                        toggleSecondaryParam("dyn");
                                    }
                                } else {
                                    if (!secondaryParamVisible["volume"]) {
                                        toggleSecondaryParam("volume");
                                    }
                                    if (hasDyn && !secondaryParamVisible["dyn"]) {
                                        toggleSecondaryParam("dyn");
                                    }
                                }
                            }}
                        />
                    );
                }

                // dyn 已并入上面的音量组药丸，不再单独渲染。
                if (p.id === "dyn") {
                    return null;
                }

                const paramActive = editParam === p.id;
                const paramEyeVisible = secondaryParamVisible[p.id] ?? false;

                // 效果参数置灰：Compose 关闭、或气声分离开关闭时都不可用。
                // 不可选中，曲线保持可见但不可编辑，且不参与合成
                // （后端在 `gate_hifigan_effect_curves` 同源剥离）。
                const paramGated = isEffectParamGated(p.id, separationEnabled, composeEnabled);
                // 提示文案要指出**真正**的原因，否则用户会去开错开关。
                // 分离门禁按参数各给一句（与 `pitch_requires_compose`
                // 同一体裁：祈使句点名"开启什么、就能编辑什么"）；
                // 只有被分离门禁的两个参数会走到这里，故二选一即可。
                const gateTooltip = !composeEnabled
                    ? t("pitch_requires_compose")
                    : p.id === "hifigan_tension"
                      ? t("hifigan_tension_requires_separation")
                      : t("breath_gain_requires_separation");

                const paramPill = (
                    <ParamToolbarPill
                        key={p.id}
                        label={getProcessorParamShortLabel(p)}
                        labelTooltip={getProcessorParamLabel(p)}
                        active={paramActive}
                        onSelect={() => dispatch(setEditParam(p.id))}
                        disabled={paramGated}
                        disabledTooltip={gateTooltip}
                        eyeMode={paramActive ? "main" : paramEyeVisible ? "on" : "off"}
                        onToggleEye={() => toggleSecondaryParam(p.id)}
                        eyeTooltip={
                            paramEyeVisible
                                ? t("secondary_overlay_tooltip_visible")
                                : t("secondary_overlay_tooltip_hidden")
                        }
                        eyeLabel={
                            paramEyeVisible ? t("hide_secondary_param") : t("show_secondary_param")
                        }
                    />
                );

                // 气声分离开关：渲染在被门禁组（气声 + 张力）的**组首**，
                // 即「气声」药丸的左侧。纯图标、无文本 —— 非中文语系的
                // `Harmonic Separation` 比整套三字母缩写（PIT/FRM/BRE…）
                // 长一个数量级，写进工具栏会把这一排撑爆；开关名改由 ToolTip
                // 与无障碍标签承载。宽度恒定（`--breath` 段 min-width 22px），
                // 切换只换图标与配色，因此不会引起整排重排。
                //
                // 【为什么不放在算法下拉之前】开关只随 nsf-hifigan 出现；
                // 若位于最右端，切换算法时它整块消失 / 出现会把音量、声像、
                // 算法一起推移。放在参数组内部后，它右侧紧邻的就是固定的
                // 「音量 → 动态 → 声像」，消失时右侧纹丝不动。
                //
                // 【为什么开关自身永不置灰】它是开启「气声 / 张力」的
                // 唯一入口；若随这两个参数一起置灰就会死锁（关了再开不回来）。
                // 置灰只作用于那两个参数药丸的标签段。
                if (p.id !== separationSwitchAnchorParamId) return withGatedSlot(paramPill);
                return (
                    <React.Fragment key={p.id}>
                        <span className="param-pill">
                            <button
                                type="button"
                                className="param-pill__seg param-pill__seg--breath"
                                data-tooltip={
                                    separationEnabled
                                        ? t("breath_tooltip_on")
                                        : t("breath_tooltip_off")
                                }
                                aria-label={`${t("breath_separation_label")}: ${
                                    separationEnabled ? t("switch_on") : t("switch_off")
                                }`}
                                onClick={() =>
                                    void handleStaticParamChange(
                                        SEPARATION_PARAM_ID,
                                        separationEnabled ? 0 : 1,
                                    )
                                }
                            >
                                <BreathAirIcon off={!separationEnabled} />
                            </button>
                        </span>
                        {/* 第 4 级：组首的药丸也让位，但**开关保留** ——
                                    它是重新开启气声 / 张力的唯一入口。 */}
                        {gatedHideAt === null ? (
                            paramPill
                        ) : (
                            <ToolbarSlot hideAt={gatedHideAt} tier={toolbarTier}>
                                {paramPill}
                            </ToolbarSlot>
                        )}
                    </React.Fragment>
                );
            })}
        </>
    );
}
