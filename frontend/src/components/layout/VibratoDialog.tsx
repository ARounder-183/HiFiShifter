// 颤音窗口：预设库 + 套用到选区。逻辑在 vibratoDialog/useVibratoDialogModel。
import { Fragment } from "react";
import { ShuffleIcon } from "@radix-ui/react-icons";
import { Box, Flex, ScrollArea, TextField } from "@radix-ui/themes";
import { persistUiSettings, setVibratoBaseline } from "../../features/session/sessionSlice";
import { shapeUsesSkew } from "../../features/vibrato/vibratoCycle";
import { VIBRATO_LIMITS } from "../../features/vibrato/vibratoPresets";
import { randomVibratoSeed } from "../../features/vibrato/vibratoSeed";
import type {
    EnvelopeCurve,
    VibratoRateMode,
    WaveShape,
} from "../../features/vibrato/vibratoTypes";
import {
    AppButton,
    AppConfirmDialog,
    AppContextMenu,
    AppDialog,
    AppField,
    AppForm,
    AppFormSection,
    AppIconButton,
    AppNumberField,
    AppSegmentedControl,
    AppSelect,
    AppSlider,
    AppSliderReadout,
    AppSwitchRow,
} from "../../ui";
import { AppFileInput } from "../../ui/FileInput";
import { VibratoPreviewPane } from "./vibrato/VibratoPreviewPane";
import {
    ENVELOPE_CURVE_KEYS,
    ENVELOPE_CURVE_ORDER,
    RATE_MODE_KEYS,
    WAVE_SHAPE_KEYS,
    WAVE_SHAPE_ORDER,
    depthToCents,
    formatNumber,
    vibratoPresetLabel,
} from "./vibrato/vibratoDialogLogic";
import { VibratoCycleEditor } from "./vibrato/VibratoCycleEditor";
import { tableFromCycle } from "./vibrato/vibratoCycleEdit";
import { CONTENT_HEIGHT, LIST_WIDTH, type VibratoDialogProps } from "./vibratoDialog/types";
import { useVibratoDialogModel } from "./vibratoDialog/useVibratoDialogModel";
import { PresetRow } from "./vibratoDialog/PresetRow";

export type { VibratoApplyTarget } from "./vibratoDialog/types";

export function VibratoDialog(props: VibratoDialogProps) {
    const { open, editParam = "pitch", paramRange, applyTarget } = props;
    const model = useVibratoDialogModel(props);
    const {
        dispatch,
        t,
        session,
        resolved,
        draft,
        isBuiltin,
        previewSamples,
        previewTab,
        handlePreviewTabChange,
        previewHalfCents,
        previewHandles,
        cycleEstimate,
        appliedPreview,
        appliedHalfCents,
        appliedHandles,
        appliedStatus,
        setAppliedRefitToken,
        audition,
        toggleAudition,
        auditionCurves,
        ioNotice,
        systemListRef,
        userListRef,
        presetDrag,
        handlePresetRowClick,
        activatePreset,
        togglePresetEnabled,
        setPresetMenu,
        beginPresetDrag,
        renameProps,
        draftIsActive,
        patch,
        selectedWaveShape,
        selectedWaveSkew,
        handDraw,
        setHandDraw,
        openHandDraw,
        paramFineAdjustKb,
        inputModifiers,
        depthValue,
        depthUnit,
        presetMenu,
        presetMenuItems,
        handleImport,
        deleteTarget,
        setDeleteTarget,
        handleDelete,
        handleExtractFromSelection,
        handleExport,
        handleCreate,
        handleDuplicate,
        handleSave,
        handleApply,
        handleOpenChange,
        fileInputRef,
        atCap,
        fitPreviewAxis,
        handlePreviewGestureStart,
        handlePreviewGestureMove,
        handlePreviewGestureEnd,
        handlePreviewPinchDepth,
    } = model;

    return (
        <>
            <AppDialog
                open={open}
                onOpenChange={handleOpenChange}
                title={t("vibrato_manager_title")}
                size="xl"
                actions={[
                    /*
                     * 【「从选区提取」排在页脚最左】它是作用于**选区**的动作，与右边
                     * 那些"改这条预设 / 改这个库"的动作不同类；而且它与「删除」同属
                     * 低频、非主动作的那一类（`align: "start"` 的既有语义）。
                     * 只在有选区宿主时出现 —— 菜单栏那条路径背后没有选区。
                     */
                    ...(applyTarget?.onExtract
                        ? [
                              {
                                  id: "extract",
                                  label: t("vibrato_extract_action"),
                                  align: "start" as const,
                                  autoClose: false,
                                  onClick: async () => {
                                      await handleExtractFromSelection();
                                  },
                              },
                          ]
                        : []),
                    /*
                     * 【为什么每个动作都显式写 `autoClose: false`】
                     * `AppDialog` 对**同步**动作默认 `autoClose: true`（异步动作默认
                     * false）。本对话框里没有"点一下就完事"的动作：导入要挑文件、
                     * 新建 / 复制 / 删除都要接着在同一个窗口里继续编辑。少了这个字段，
                     * 一个同步动作就会顺手把窗口关掉 —— 而且这条规则很容易被一次
                     * "把 async 去掉"的重构悄悄改回去，所以宁可逐个写明。
                     */
                    {
                        id: "delete",
                        label: t("vibrato_manager_delete"),
                        intent: "danger",
                        align: "start",
                        disabled: !draft || isBuiltin,
                        // 删除走二次确认，不关闭主对话框。
                        autoClose: false,
                        onClick: async () => {
                            setDeleteTarget(draft);
                        },
                    },
                    {
                        id: "import",
                        label: t("vibrato_io_import"),
                        autoClose: false,
                        onClick: () => fileInputRef.current?.click(),
                    },
                    {
                        id: "export",
                        label: t("vibrato_io_export"),
                        disabled: !draft,
                        autoClose: false,
                        onClick: async () => {
                            await handleExport();
                        },
                    },
                    {
                        id: "new",
                        label: t("vibrato_manager_new"),
                        disabled: atCap,
                        autoClose: false,
                        onClick: handleCreate,
                    },
                    {
                        id: "duplicate",
                        label: t("vibrato_manager_duplicate"),
                        disabled: !draft,
                        autoClose: false,
                        onClick: handleDuplicate,
                    },
                    {
                        id: "save",
                        label: t("vibrato_manager_save"),
                        /*
                         * 【为什么「保存」不再是 primary】有选区宿主时页脚有两个
                         * "往前走"的动作（保存写库、应用写选区），而一张对话框里
                         * 只该有一个主按钮 —— 主的是「应用」（用户开这扇窗的目的）。
                         * 没有宿主时它就是唯一的主按钮（见 defaultActionId）。
                         */
                        intent: applyTarget ? "default" : "primary",
                        disabled: !draft,
                        // 保存**不关闭**对话框：用户常要"先存一版、接着调"，
                        // 存完就把窗口收掉等于逼他重新打开。
                        autoClose: false,
                        onClick: handleSave,
                    },
                    ...(applyTarget
                        ? [
                              {
                                  id: "apply",
                                  label: t("vibrato_apply_apply"),
                                  intent: "primary" as const,
                                  disabled: !draft,
                                  /*
                                   * 关窗在 `handleApply` 里显式做（而不是靠 `autoClose`
                                   * 的默认值）：先落到选区、再关，两步的顺序是这段
                                   * 逻辑的一部分，不该藏在"同步动作默认关窗"这条规则里。
                                   */
                                  autoClose: false,
                                  onClick: handleApply,
                              },
                          ]
                        : []),
                    {
                        // 显式关闭按钮：保存不再关闭窗口之后，页脚里没有"退出"的
                        // 去处，只剩 Esc / 点外部 —— 两者都不显眼。
                        id: "close",
                        label: t("close"),
                        autoClose: false,
                        onClick: () => handleOpenChange(false),
                    },
                ]}
                /*
                 * 默认动作：有选区宿主时是「应用」（用户开这扇窗的目的就是应用它），
                 * 否则是「保存」（与合并前一致）。关闭排在它们右边（页脚最右是关闭的
                 * 常见排布），但 Enter 不该变成"关掉窗口"。
                 */
                defaultActionId={applyTarget ? "apply" : "save"}
            >
                <Flex
                    direction="column"
                    gap="3"
                    data-vibrato-content
                    className="min-h-0"
                    style={{ height: CONTENT_HEIGHT }}
                >
                    {/* ---- 预览（整行置顶，不参与任何滚动） ----
                        放在两栏之上而不是塞进参数流的头部：整行宽度读波形更清楚，
                        且它不属于任何滚动区，调参数时**永远**不会滚出视野。
                        两个页签共用这一块（见 `VibratoPreviewPane`）。 */}
                    {draft && previewSamples ? (
                        <>
                            <VibratoPreviewPane
                                tab={previewTab}
                                onTabChange={handlePreviewTabChange}
                                hasSelection={Boolean(applyTarget)}
                                presetSamples={previewSamples}
                                presetHalfCents={previewHalfCents}
                                presetHandles={previewHandles}
                                cyclesEstimate={cycleEstimate}
                                appliedSamples={appliedPreview}
                                appliedHalfCents={appliedHalfCents}
                                appliedHandles={appliedHandles}
                                appliedStatus={appliedStatus}
                                /*
                                 * 「摆放方式」在波形右上角就地可改。
                                 *
                                 * 【为什么读写的不是草稿】它是**添加颤音的参数**，不属于
                                 * 任何预设：用户先定摆放方式、再挑预设，换预设不该把它
                                 * 换掉（见 `BaselineMode` 的说明）。因此它进设置、本机
                                 * 记忆，下次打开还是这个。
                                 */
                                baseline={session.vibratoBaseline}
                                onBaselineChange={(mode) => {
                                    dispatch(setVibratoBaseline(mode));
                                    void dispatch(persistUiSettings());
                                }}
                                onGestureStart={handlePreviewGestureStart}
                                onGestureMove={handlePreviewGestureMove}
                                onGestureEnd={handlePreviewGestureEnd}
                                onPinchDepth={handlePreviewPinchDepth}
                                onFit={() => {
                                    // 「适应」作用于**当前页签**：两页各有一把标尺，
                                    // 一次点击只该动用户正看着的那把。
                                    if (previewTab === "applied" && applyTarget) {
                                        setAppliedRefitToken((token) => token + 1);
                                    } else {
                                        fitPreviewAxis(draft);
                                    }
                                }}
                                audition={audition}
                                onAudition={toggleAudition}
                                appliedAuditionDisabled={!auditionCurves}
                            />
                        </>
                    ) : null}

                    {ioNotice ? (
                        <span
                            className="hs-type-caption"
                            style={{
                                color: ioNotice.danger
                                    ? "var(--qt-danger-text)"
                                    : "var(--qt-text-muted)",
                            }}
                            role="status"
                        >
                            {ioNotice.text}
                        </span>
                    ) : null}

                    <Flex gap="4" align="stretch" className="min-h-0 flex-1">
                        {/* ---- 预设列表 ---- */}
                        <Flex
                            direction="column"
                            gap="2"
                            className="min-h-0"
                            style={{ width: LIST_WIDTH, flexShrink: 0 }}
                        >
                            <ScrollArea
                                className="hs-scroll-area"
                                style={{ height: "100%" }}
                                scrollbars="vertical"
                                type="auto"
                            >
                                <Flex direction="column" gap="1" pr="2">
                                    <span className="hs-type-muted">
                                        {t("vibrato_manager_group_system")}
                                    </span>
                                    {/* 系统预设同样可拖拽排序（顺序以 id 列表持久化）。
                                        只在本组内换位，不会与下面的自定义预设混排。 */}
                                    <Flex direction="column" gap="1" ref={systemListRef}>
                                        {resolved.system.map((preset, index) => (
                                            <Fragment key={preset.id}>
                                                {presetDrag?.group === "system" &&
                                                presetDrag.insertionIndex === index ? (
                                                    <Box
                                                        aria-hidden
                                                        className="rounded-full bg-qt-accent"
                                                        style={{ height: 2 }}
                                                    />
                                                ) : null}
                                                <Box
                                                    data-preset-row={preset.id}
                                                    style={{
                                                        opacity:
                                                            presetDrag?.group === "system" &&
                                                            presetDrag.id === preset.id
                                                                ? 0.4
                                                                : undefined,
                                                    }}
                                                >
                                                    <PresetRow
                                                        preset={preset}
                                                        selected={draft?.id === preset.id}
                                                        active={
                                                            session.activeVibratoPresetId ===
                                                            preset.id
                                                        }
                                                        enabled={
                                                            !session.disabledVibratoPresetIds.includes(
                                                                preset.id,
                                                            )
                                                        }
                                                        reorderable
                                                        onSelect={() =>
                                                            handlePresetRowClick(preset)
                                                        }
                                                        onActivate={() => activatePreset(preset)}
                                                        onToggleEnabled={() =>
                                                            togglePresetEnabled(preset)
                                                        }
                                                        onContextMenu={(x, y) =>
                                                            setPresetMenu({ x, y, preset })
                                                        }
                                                        onDragPointerDown={(event) =>
                                                            beginPresetDrag(
                                                                preset,
                                                                index,
                                                                "system",
                                                                event,
                                                            )
                                                        }
                                                        {...renameProps(preset)}
                                                    />
                                                </Box>
                                            </Fragment>
                                        ))}
                                        {presetDrag?.group === "system" &&
                                        presetDrag.insertionIndex >= resolved.system.length ? (
                                            <Box
                                                aria-hidden
                                                className="rounded-full bg-qt-accent"
                                                style={{ height: 2 }}
                                            />
                                        ) : null}
                                    </Flex>

                                    <Box pt="2">
                                        <span className="hs-type-muted">
                                            {t("vibrato_manager_group_user")}
                                        </span>
                                    </Box>
                                    {resolved.user.length === 0 ? (
                                        <span className="hs-type-caption">
                                            {t("vibrato_manager_empty")}
                                        </span>
                                    ) : (
                                        /* 用户预设可**拖拽排序**：按住行上下拖，越过别行中线
                                           即换位，松手落盘。落点用一条指示线表达；被拖的那行
                                           压暗。键盘可达的等价操作在行的右键菜单里
                                           （上移 / 下移）。 */
                                        <Flex direction="column" gap="1" ref={userListRef}>
                                            {resolved.user.map((preset, index) => (
                                                <Fragment key={preset.id}>
                                                    {presetDrag?.group === "user" &&
                                                    presetDrag.insertionIndex === index ? (
                                                        <Box
                                                            aria-hidden
                                                            className="rounded-full bg-qt-accent"
                                                            style={{ height: 2 }}
                                                        />
                                                    ) : null}
                                                    <Box
                                                        data-preset-row={preset.id}
                                                        style={{
                                                            opacity:
                                                                presetDrag?.group === "user" &&
                                                                presetDrag.id === preset.id
                                                                    ? 0.4
                                                                    : undefined,
                                                        }}
                                                    >
                                                        <PresetRow
                                                            preset={preset}
                                                            selected={draft?.id === preset.id}
                                                            active={
                                                                session.activeVibratoPresetId ===
                                                                preset.id
                                                            }
                                                            enabled={
                                                                !session.disabledVibratoPresetIds.includes(
                                                                    preset.id,
                                                                )
                                                            }
                                                            reorderable
                                                            onSelect={() =>
                                                                handlePresetRowClick(preset)
                                                            }
                                                            onActivate={() =>
                                                                activatePreset(preset)
                                                            }
                                                            onToggleEnabled={() =>
                                                                togglePresetEnabled(preset)
                                                            }
                                                            onContextMenu={(x, y) =>
                                                                setPresetMenu({ x, y, preset })
                                                            }
                                                            onDragPointerDown={(event) =>
                                                                beginPresetDrag(
                                                                    preset,
                                                                    index,
                                                                    "user",
                                                                    event,
                                                                )
                                                            }
                                                            {...renameProps(preset)}
                                                        />
                                                    </Box>
                                                </Fragment>
                                            ))}
                                            {presetDrag?.group === "user" &&
                                            presetDrag.insertionIndex >= resolved.user.length ? (
                                                <Box
                                                    aria-hidden
                                                    className="rounded-full bg-qt-accent"
                                                    style={{ height: 2 }}
                                                />
                                            ) : null}
                                        </Flex>
                                    )}
                                </Flex>
                            </ScrollArea>
                        </Flex>

                        {/* ---- 编辑器 ---- */}
                        {/* `Flex`（而非 `Box className="flex …"`）：Radix 的 `.rt-Box`
                            以未分层的 `display: block` 压过 Tailwind 4 分层后的
                            `.flex`/`.flex-col`，会把这里退化成块级布局。 */}
                        <Flex
                            direction="column"
                            className="min-h-0"
                            style={{ minWidth: 0, flex: 1 }}
                        >
                            {draft && previewSamples ? (
                                <Flex direction="column" gap="3" className="min-h-0 flex-1">
                                    {/*
                                     * 正在编辑哪一条 + 一键设为当前使用。
                                     *
                                     * 【为什么钉在滚动区之上】它要回答的正是"我现在
                                     * 这条是不是当前用的"，调参数时滚下去就找不到答案了。
                                     * 系统预设没有名字输入框，这一行也是它唯一的署名。
                                     */}
                                    <Flex align="center" gap="2" className="shrink-0">
                                        <span
                                            className="hs-type-label min-w-0 flex-1"
                                            style={{
                                                overflow: "hidden",
                                                textOverflow: "ellipsis",
                                                whiteSpace: "nowrap",
                                            }}
                                        >
                                            {vibratoPresetLabel(draft, t)}
                                        </span>
                                        <AppButton
                                            size="sm"
                                            disabled={draftIsActive}
                                            onClick={() => activatePreset(draft)}
                                        >
                                            {draftIsActive
                                                ? t("vibrato_manager_in_use")
                                                : t("vibrato_manager_set_active")}
                                        </AppButton>
                                    </Flex>
                                    <ScrollArea
                                        className="hs-scroll-area min-h-0 flex-1"
                                        scrollbars="vertical"
                                        type="auto"
                                    >
                                        <Box pr="2">
                                            <AppForm>
                                                {!isBuiltin ? (
                                                    <AppField label={t("vibrato_manager_name")}>
                                                        <TextField.Root
                                                            size="2"
                                                            value={draft.name}
                                                            aria-label={t("vibrato_manager_name")}
                                                            onChange={(event) =>
                                                                patch({ name: event.target.value })
                                                            }
                                                        />
                                                    </AppField>
                                                ) : null}
                                                <AppFormSection title={t("vibrato_section_wave")}>
                                                    <AppField label={t("vibrato_shape_label")}>
                                                        <Flex align="center" gap="2" wrap="wrap">
                                                            <AppSelect
                                                                value={selectedWaveShape}
                                                                onValueChange={(value) => {
                                                                    const shape =
                                                                        value as WaveShape;
                                                                    if (handDraw) {
                                                                        // 手绘中改形状：记下"当前形状"并即时套用，
                                                                        // **不**退出编辑器 —— 否则「复位到{形状}」
                                                                        // 永远只能指向兜底的正弦。
                                                                        setHandDraw({
                                                                            ...handDraw,
                                                                            shape,
                                                                        });
                                                                        patch({
                                                                            cycle: {
                                                                                kind: "table",
                                                                                table: tableFromCycle(
                                                                                    {
                                                                                        kind: "shape",
                                                                                        shape,
                                                                                        skew: handDraw.skew,
                                                                                    },
                                                                                ),
                                                                            },
                                                                        });
                                                                        return;
                                                                    }
                                                                    // 换成参数形状即退出"手绘表"，避免编辑器与草稿错位。
                                                                    setHandDraw(null);
                                                                    patch({
                                                                        cycle: {
                                                                            kind: "shape",
                                                                            shape,
                                                                            skew:
                                                                                draft.cycle.kind ===
                                                                                "shape"
                                                                                    ? draft.cycle
                                                                                          .skew
                                                                                    : 0.5,
                                                                        },
                                                                    });
                                                                }}
                                                                options={WAVE_SHAPE_ORDER.map(
                                                                    (shape) => ({
                                                                        value: shape,
                                                                        label: t(
                                                                            WAVE_SHAPE_KEYS[shape],
                                                                        ),
                                                                    }),
                                                                )}
                                                            />
                                                            <AppButton
                                                                size="sm"
                                                                emphasis="soft"
                                                                onClick={openHandDraw}
                                                            >
                                                                {t("vibrato_handdraw_open")}
                                                            </AppButton>
                                                        </Flex>
                                                    </AppField>
                                                    <AppField label={t("vibrato_skew")}>
                                                        <Flex align="center" gap="2" wrap="wrap">
                                                            <AppSlider
                                                                unit="percent"
                                                                min={2}
                                                                max={98}
                                                                /*
                                                                 * 偏斜只对**参数式形状**有效：手绘（以及从选区提取）
                                                                 * 出来的表波形里没有"上升段占比"这个东西 —— 表就是
                                                                 * 用户画的那条曲线本身。
                                                                 *
                                                                 * 【为什么必须禁用而不是"允许调但没效果"】参数式形状的
                                                                 * 偏斜在 `sampleCycle` 里参与取样；表的取样完全不看它。
                                                                 * 留着能拖会让用户以为拖了会变，实际毫无反应。
                                                                 */
                                                                disabled={
                                                                    draft.cycle.kind !== "shape" ||
                                                                    !shapeUsesSkew(
                                                                        selectedWaveShape,
                                                                    )
                                                                }
                                                                value={Math.round(
                                                                    selectedWaveSkew * 100,
                                                                )}
                                                                ariaLabel={t("vibrato_skew")}
                                                                onChange={(next) =>
                                                                    patch({
                                                                        cycle: {
                                                                            kind: "shape",
                                                                            shape: selectedWaveShape,
                                                                            skew: next / 100,
                                                                        },
                                                                    })
                                                                }
                                                            />
                                                            <AppSliderReadout>
                                                                {`${formatNumber(selectedWaveSkew * 100)}%`}
                                                            </AppSliderReadout>
                                                        </Flex>
                                                    </AppField>
                                                    {/* 手绘周期编辑器：波形分区下方展开（只在草稿是 table 时）。 */}
                                                    {handDraw && draft.cycle.kind === "table" ? (
                                                        <VibratoCycleEditor
                                                            table={draft.cycle.table}
                                                            // 画布的 aria-label 讲"这块画布能干什么"，
                                                            // 与展开按钮的"手绘…"（讲动作）分开两个键。
                                                            ariaLabel={t("vibrato_handdraw_canvas")}
                                                            readoutLabels={{
                                                                phase: t("vibrato_handdraw_phase"),
                                                                scale: t("vibrato_handdraw_scale"),
                                                            }}
                                                            // 「精细调整」与预览画布同一个键位。
                                                            fineAdjustKb={paramFineAdjustKb}
                                                            // 压感画笔：与预览画布共用同一份标定，
                                                            // 于是"多用力 = 多写一点"与
                                                            // "多用力 = 走得快一点"是同一个手感。
                                                            pressurePaint={{
                                                                begin: () => inputModifiers.reset(),
                                                                weightFor: (pointerEvent) =>
                                                                    inputModifiers.paintWeightFor(
                                                                        pointerEvent,
                                                                    ),
                                                            }}
                                                            // 接触读数：手指落下时恰好盖住正在改的那一格。
                                                            contactReadout={
                                                                session.penInput.contactReadout
                                                            }
                                                            smoothLabel={t(
                                                                "vibrato_handdraw_smooth",
                                                            )}
                                                            resetLabel={t("vibrato_handdraw_reset")}
                                                            resetShapeLabel={t(
                                                                "vibrato_handdraw_reset_shape",
                                                            ).replace(
                                                                "{shape}",
                                                                t(
                                                                    WAVE_SHAPE_KEYS[
                                                                        selectedWaveShape
                                                                    ],
                                                                ),
                                                            )}
                                                            onChange={(table) =>
                                                                patch({
                                                                    cycle: { kind: "table", table },
                                                                })
                                                            }
                                                            onReset={() =>
                                                                patch({
                                                                    cycle: {
                                                                        kind: "table",
                                                                        table: tableFromCycle(
                                                                            handDraw.baseline,
                                                                        ),
                                                                    },
                                                                })
                                                            }
                                                            // 复位到下拉框里当前选中的形状：
                                                            // 手绘画歪了想从头来，或把提取出的
                                                            // 波形换成规整形状，都靠它。
                                                            onResetToShape={() =>
                                                                patch({
                                                                    cycle: {
                                                                        kind: "table",
                                                                        table: tableFromCycle({
                                                                            kind: "shape",
                                                                            shape: selectedWaveShape,
                                                                            skew: selectedWaveSkew,
                                                                        }),
                                                                    },
                                                                })
                                                            }
                                                        />
                                                    ) : null}
                                                </AppFormSection>

                                                <AppFormSection title={t("vibrato_section_depth")}>
                                                    <AppField label={t("vibrato_depth_label")}>
                                                        <AppNumberField
                                                            value={depthValue}
                                                            unit={depthUnit}
                                                            min={VIBRATO_LIMITS.depthCents.min}
                                                            ariaLabel={t("vibrato_depth_label")}
                                                            onChange={(next) =>
                                                                patch({
                                                                    depthCents: depthToCents(
                                                                        next,
                                                                        editParam,
                                                                        paramRange,
                                                                    ),
                                                                })
                                                            }
                                                            onCommit={() => undefined}
                                                        />
                                                    </AppField>
                                                    <AppField label={t("vibrato_depth_ramp")}>
                                                        <Flex align="center" gap="2" wrap="wrap">
                                                            <AppNumberField
                                                                value={draft.depthRamp.start}
                                                                unit="percentFine"
                                                                min={0}
                                                                max={2}
                                                                suffix={t(
                                                                    "vibrato_depth_ramp_start",
                                                                )}
                                                                ariaLabel={`${t("vibrato_depth_ramp")} ${t("vibrato_depth_ramp_start")}`}
                                                                onCommit={(next) =>
                                                                    patch({
                                                                        depthRamp: {
                                                                            ...draft.depthRamp,
                                                                            start: next,
                                                                        },
                                                                    })
                                                                }
                                                            />
                                                            <AppNumberField
                                                                value={draft.depthRamp.end}
                                                                unit="percentFine"
                                                                min={0}
                                                                max={2}
                                                                suffix={t("vibrato_depth_ramp_end")}
                                                                ariaLabel={`${t("vibrato_depth_ramp")} ${t("vibrato_depth_ramp_end")}`}
                                                                onCommit={(next) =>
                                                                    patch({
                                                                        depthRamp: {
                                                                            ...draft.depthRamp,
                                                                            end: next,
                                                                        },
                                                                    })
                                                                }
                                                            />
                                                        </Flex>
                                                    </AppField>
                                                    <AppField label={t("vibrato_bias")}>
                                                        <AppNumberField
                                                            value={draft.biasCents}
                                                            unit="cents"
                                                            ariaLabel={t("vibrato_bias")}
                                                            onCommit={(next) =>
                                                                patch({ biasCents: next })
                                                            }
                                                        />
                                                    </AppField>
                                                    <AppField label={t("vibrato_irregularity")}>
                                                        <Flex align="center" gap="2" wrap="wrap">
                                                            <AppSlider
                                                                unit="percent"
                                                                min={0}
                                                                max={100}
                                                                value={Math.round(
                                                                    draft.irregularity,
                                                                )}
                                                                ariaLabel={t(
                                                                    "vibrato_irregularity",
                                                                )}
                                                                onChange={(next) =>
                                                                    patch({ irregularity: next })
                                                                }
                                                            />
                                                            <AppSliderReadout>
                                                                {`${formatNumber(draft.irregularity)}%`}
                                                            </AppSliderReadout>
                                                            <AppIconButton
                                                                icon={<ShuffleIcon />}
                                                                /*
                                                                 * 不规则度为 0 时**没有图案可换** —— 噪声整个被乘掉了，
                                                                 * 换种子不会改变任何一帧。此时禁用并说明原因，而不是
                                                                 * 让用户点了半天看不出变化（新建的预设默认就是 0）。
                                                                 */
                                                                tooltip={
                                                                    draft.irregularity > 0
                                                                        ? t("vibrato_seed_roll")
                                                                        : t(
                                                                              "vibrato_seed_roll_needs_irregularity",
                                                                          )
                                                                }
                                                                disabled={!(draft.irregularity > 0)}
                                                                onClick={() =>
                                                                    patch({
                                                                        seed: randomVibratoSeed(),
                                                                    })
                                                                }
                                                            />
                                                        </Flex>
                                                    </AppField>
                                                </AppFormSection>

                                                <AppFormSection title={t("vibrato_section_rate")}>
                                                    <AppField label={t("vibrato_rate_mode")}>
                                                        <AppSegmentedControl<VibratoRateMode>
                                                            value={draft.rateMode}
                                                            ariaLabel={t("vibrato_rate_mode")}
                                                            onChange={(next) =>
                                                                patch({ rateMode: next })
                                                            }
                                                            options={(
                                                                ["hz", "cycles"] as const
                                                            ).map((mode) => ({
                                                                value: mode,
                                                                label: t(RATE_MODE_KEYS[mode]),
                                                            }))}
                                                        />
                                                    </AppField>
                                                    {draft.rateMode === "hz" ? (
                                                        <AppField label={t("vibrato_rate_label")}>
                                                            <AppNumberField
                                                                value={draft.rateHz}
                                                                unit="vibratoHz"
                                                                min={0.1}
                                                                max={20}
                                                                ariaLabel={t("vibrato_rate_label")}
                                                                onCommit={(next) =>
                                                                    patch({ rateHz: next })
                                                                }
                                                            />
                                                        </AppField>
                                                    ) : (
                                                        <AppField label={t("vibrato_cycles")}>
                                                            <AppNumberField
                                                                value={draft.cycles}
                                                                unit="integer"
                                                                min={0.5}
                                                                max={128}
                                                                ariaLabel={t("vibrato_cycles")}
                                                                onCommit={(next) =>
                                                                    patch({ cycles: next })
                                                                }
                                                            />
                                                        </AppField>
                                                    )}
                                                    <AppField label={t("vibrato_rate_ramp")}>
                                                        <AppNumberField
                                                            value={draft.rateRampEnd}
                                                            unit="percentFine"
                                                            min={0.25}
                                                            max={4}
                                                            ariaLabel={t("vibrato_rate_ramp")}
                                                            onCommit={(next) =>
                                                                patch({ rateRampEnd: next })
                                                            }
                                                        />
                                                    </AppField>
                                                    <AppSwitchRow
                                                        label={t("vibrato_align_cycles")}
                                                        checked={draft.alignCycles}
                                                        onCheckedChange={(checked) =>
                                                            patch({ alignCycles: checked })
                                                        }
                                                    />
                                                </AppFormSection>

                                                <AppFormSection
                                                    title={t("vibrato_section_envelope")}
                                                >
                                                    <AppField label={t("vibrato_attack")}>
                                                        <Flex align="center" gap="2" wrap="wrap">
                                                            <AppNumberField
                                                                value={draft.attackMs}
                                                                unit="milliseconds"
                                                                min={0}
                                                                ariaLabel={t("vibrato_attack")}
                                                                onCommit={(next) =>
                                                                    patch({ attackMs: next })
                                                                }
                                                            />
                                                            <AppSelect
                                                                value={draft.attackCurve}
                                                                ariaLabel={t("vibrato_curve")}
                                                                onValueChange={(value) =>
                                                                    patch({
                                                                        attackCurve:
                                                                            value as EnvelopeCurve,
                                                                    })
                                                                }
                                                                options={ENVELOPE_CURVE_ORDER.map(
                                                                    (curve) => ({
                                                                        value: curve,
                                                                        label: t(
                                                                            ENVELOPE_CURVE_KEYS[
                                                                                curve
                                                                            ],
                                                                        ),
                                                                    }),
                                                                )}
                                                            />
                                                        </Flex>
                                                    </AppField>
                                                    <AppField label={t("vibrato_release")}>
                                                        <Flex align="center" gap="2" wrap="wrap">
                                                            <AppNumberField
                                                                value={draft.releaseMs}
                                                                unit="milliseconds"
                                                                min={0}
                                                                ariaLabel={t("vibrato_release")}
                                                                onCommit={(next) =>
                                                                    patch({ releaseMs: next })
                                                                }
                                                            />
                                                            <AppSelect
                                                                value={draft.releaseCurve}
                                                                ariaLabel={t("vibrato_curve")}
                                                                onValueChange={(value) =>
                                                                    patch({
                                                                        releaseCurve:
                                                                            value as EnvelopeCurve,
                                                                    })
                                                                }
                                                                options={ENVELOPE_CURVE_ORDER.map(
                                                                    (curve) => ({
                                                                        value: curve,
                                                                        label: t(
                                                                            ENVELOPE_CURVE_KEYS[
                                                                                curve
                                                                            ],
                                                                        ),
                                                                    }),
                                                                )}
                                                            />
                                                        </Flex>
                                                    </AppField>
                                                    <AppField label={t("vibrato_phase")}>
                                                        <AppNumberField
                                                            value={draft.startPhaseDeg}
                                                            unit="integer"
                                                            min={0}
                                                            max={360}
                                                            ariaLabel={t("vibrato_phase")}
                                                            onCommit={(next) =>
                                                                patch({ startPhaseDeg: next })
                                                            }
                                                        />
                                                    </AppField>
                                                </AppFormSection>
                                            </AppForm>
                                        </Box>
                                    </ScrollArea>
                                </Flex>
                            ) : (
                                <span className="hs-type-caption">
                                    {t("vibrato_manager_empty")}
                                </span>
                            )}
                        </Flex>
                    </Flex>

                    {/* 预设行右键菜单：把"针对这一条预设"的动作收拢到指针处。
                        **必须渲染在对话框内部** —— 手写菜单的层级（--qt-z-menu: 999）
                        低于对话框（--qt-z-dialog: 1000），挂在对话框外面会被整个盖住、
                        点不到；而且它一旦跑到对话框的 DOM 之外，Radix 还会把点击当成
                        "点到了弹窗外面"从而顺手把窗口关掉。放在对话框子树里，层级归属
                        对话框自己的 stacking context，就不会被压住。
                        `position: fixed` 仍按视口定位（对话框没有常驻 transform），
                        也不会被祖先的 `overflow: auto` 裁掉。 */}
                    {presetMenu ? (
                        <AppContextMenu
                            x={presetMenu.x}
                            y={presetMenu.y}
                            items={presetMenuItems}
                            ariaLabel={t("vibrato_manager_row_menu")}
                            onClose={() => setPresetMenu(null)}
                        />
                    ) : null}
                </Flex>
            </AppDialog>

            {/* 文件输入必须由**常驻**组件渲染：文件选择期间 change 的接收方要活着
                （用户第一次选 a.json、第二次又选同一文件时，不清空 value 就不会再
                触发 —— 这条由原语处理，这里只保证宿主不随菜单卸载）。 */}
            <AppFileInput
                inputRef={fileInputRef}
                accept=".json,application/json"
                onFiles={(files) => void handleImport(files)}
            />

            <AppConfirmDialog
                open={deleteTarget !== null}
                onOpenChange={(next) => {
                    if (!next) setDeleteTarget(null);
                }}
                title={t("vibrato_manager_delete")}
                message={t("vibrato_manager_delete_confirm").replace(
                    "{name}",
                    deleteTarget ? vibratoPresetLabel(deleteTarget, t) : "",
                )}
                confirmLabel={t("vibrato_manager_delete")}
                cancelLabel={t("cancel")}
                intent="danger"
                onConfirm={handleDelete}
            />
        </>
    );
}
