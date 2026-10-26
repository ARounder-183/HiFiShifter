/**
 * 声码器**参数描述符**，以及"当前参数"的取值域 / 默认值 / 量化单位 / 标签。
 *
 * 这些值有一个共同的真值来源：后端按算法返回的参数描述符列表
 * （`paramsApi.getProcessorParams`）。面板里几乎所有与参数有关的东西都从这里派生：
 *
 * - `orderedProcessorParams`：工具栏按钮的顺序（音高 → 算法参数 → 音量/声像）；
 * - `currentParamRange` / `currentParamDefaultValue` / `currentParamQuantizeUnit`：
 *   当前参数的上下限、默认值与拖拽量化步长（选区吸附、双击复位、滚轮步进都用它）；
 * - `getProcessorParamLabel` / `getProcessorParamShortLabel` / `getStaticOptionLabel`：
 *   三个 i18n 映射（全称给 ToolTip，短标签给按钮，静态枚举值给选项文字）；
 * - `handleStaticParamChange`：静态枚举参数的写回（fire-and-forget，失败必须兜底）。
 *
 * 【为什么连 state 一起搬】`processorParams` / `processorStaticParams` /
 * `processorStaticValues` 三份 state 只有一个写入者（下面那条按 algo 重新抓取的
 * effect），读取者全部在同一个派生链里。把 state 留在面板、只搬派生值，会让"谁负责
 * 在算法切换时清空视口"重新变得模糊——那正是这段代码以前最容易出错的地方
 * （旧算法残留的视口会让新参数画在错误的纵轴范围里）。
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { RootState } from "../../../../app/store";
import type { MessageKey } from "../../../../i18n/messages";
import { paramsApi } from "../../../../services/api/params";
import type { ProcessorParamDescriptor } from "../../../../types/api";
import {
    CHILD_FORMANT_OFFSET_CENTS_RANGE,
    CHILD_PITCH_OFFSET_CENTS_RANGE,
    CHILD_PITCH_OFFSET_DEGREES_RANGE,
    isChildFormantOffsetCentsParam,
    isChildPitchOffsetCentsParam,
    isChildPitchOffsetDegreesParam,
    isChildPitchOffsetParam,
} from "../childPitchOffsetParams";
import { isDynParam } from "../paramRanges";
import type { ParamName, ValueViewport } from "../types";
import { getParamToolbarRank } from "./panelHelpers";

/** 根轨道（决定用哪一套参数描述符）。 */
type RootTrack = RootState["session"]["tracks"][number] | null;

type ProcessorParamsInputs = {
    rootTrack: RootTrack;
    rootTrackId: string | null;
    editParam: ParamName;
    /** 各参数视口（新算法参数在这里补初值）。 */
    paramViewsRef: { current: Record<string, ValueViewport> };
    /** 标脏入口（补了视口初值后请求一帧）。 */
    invalidate: () => void;
    /** i18n 取词（全称 / 短标签 / 静态枚举选项）。 */
    t: (key: MessageKey) => string;
};

/** 参数描述符 + 当前参数元数据 + 标签映射。副作用顺序与拆分前一致。 */
export function useProcessorParams({
    rootTrack,
    rootTrackId,
    editParam,
    paramViewsRef,
    invalidate,
    t,
}: ProcessorParamsInputs) {
    // 声码器参数描述符（由 algo 动态定制面板）
    const [processorParams, setProcessorParams] = useState<ProcessorParamDescriptor[]>([]);
    const processorParamsRef = useRef<ProcessorParamDescriptor[]>([]);
    const [processorStaticParams, setProcessorStaticParams] = useState<ProcessorParamDescriptor[]>(
        [],
    );
    const [processorStaticValues, setProcessorStaticValues] = useState<Record<string, number>>({});

    // 工具栏参数按钮按“音高 → 中间参数（随算法变化）→ 音量/声像”的顺序排列
    const orderedProcessorParams = useMemo(() => {
        const algo = rootTrack?.pitchAnalysisAlgo;
        return [...processorParams].sort(
            (a, b) => getParamToolbarRank(a.id, algo) - getParamToolbarRank(b.id, algo),
        );
    }, [processorParams, rootTrack?.pitchAnalysisAlgo]);
    const currentParamRange = useMemo(() => {
        if (editParam === "pitch") {
            return { min: 24, max: 108 };
        }
        if (isChildPitchOffsetCentsParam(editParam)) {
            return {
                min: CHILD_PITCH_OFFSET_CENTS_RANGE.min,
                max: CHILD_PITCH_OFFSET_CENTS_RANGE.max,
            };
        }
        if (isChildPitchOffsetDegreesParam(editParam)) {
            return {
                min: CHILD_PITCH_OFFSET_DEGREES_RANGE.min,
                max: CHILD_PITCH_OFFSET_DEGREES_RANGE.max,
            };
        }
        if (isChildFormantOffsetCentsParam(editParam)) {
            return {
                min: CHILD_FORMANT_OFFSET_CENTS_RANGE.min,
                max: CHILD_FORMANT_OFFSET_CENTS_RANGE.max,
            };
        }
        const desc = processorParams.find((d) => d.id === editParam);
        if (desc?.kind.type === "automation_curve") {
            return {
                min: desc.kind.min_value,
                max: desc.kind.max_value,
            };
        }
        return undefined;
    }, [editParam, processorParams]);

    const currentParamDefaultValue = useMemo(() => {
        if (editParam === "pitch") return 60;
        if (
            isChildPitchOffsetCentsParam(editParam) ||
            isChildPitchOffsetDegreesParam(editParam) ||
            isChildFormantOffsetCentsParam(editParam)
        ) {
            return 0;
        }
        const desc = processorParams.find((d) => d.id === editParam);
        if (desc?.kind.type === "automation_curve") {
            return Number(desc.kind.default_value) || 0;
        }
        if (editParam === "volume" || editParam === "dyn" || editParam === "dyn_edit") {
            return 1;
        }
        return 0;
    }, [editParam, processorParams]);

    const currentParamQuantizeUnit = useMemo(() => {
        if (isChildPitchOffsetCentsParam(editParam)) return 100;
        if (isChildPitchOffsetDegreesParam(editParam)) return 0.5;
        if (isChildFormantOffsetCentsParam(editParam)) return 50;
        if (editParam === "volume" || isDynParam(editParam)) {
            return 0.05;
        }
        if (editParam === "formant_shift_cents") return 50;
        if (editParam === "breath_gain" || editParam === "hifigan_tension") {
            return 0.05;
        }
        if (editParam === "pan") return 0.1;
        if (editParam === "breathiness") return 250;
        const span = Math.abs((currentParamRange?.max ?? 1) - (currentParamRange?.min ?? 0));
        if (span <= 0) return 0.01;
        return Math.max(0.01, span / 20);
    }, [editParam, currentParamRange]);

    useEffect(() => {
        if (!isChildPitchOffsetParam(editParam)) return;
        if (paramViewsRef.current[editParam]) return;
        const range = isChildPitchOffsetCentsParam(editParam)
            ? CHILD_PITCH_OFFSET_CENTS_RANGE
            : isChildPitchOffsetDegreesParam(editParam)
              ? CHILD_PITCH_OFFSET_DEGREES_RANGE
              : CHILD_FORMANT_OFFSET_CENTS_RANGE;
        paramViewsRef.current = {
            ...paramViewsRef.current,
            [editParam]: {
                center: (range.min + range.max) / 2,
                span: range.max - range.min,
            },
        };
        invalidate();
        // eslint-disable-next-line react-hooks/exhaustive-deps -- paramViewsRef 是稳定 ref；加入依赖不会改变语义
    }, [editParam, invalidate]);

    // 当 algo 变化时，重新抓取参数描述符
    useEffect(() => {
        const algo = rootTrack?.pitchAnalysisAlgo ?? "nsf_hifigan_onnx";
        let cancelled = false;
        paramsApi
            .getProcessorParams(algo)
            .then((params) => {
                if (cancelled) return;
                // 只保留 AutomationCurve 类型（可以绘制曲线的）
                const curvable = params.filter((p) => p.kind.type === "automation_curve");
                const staticParams = params.filter((p) => p.kind.type === "static_enum");
                processorParamsRef.current = curvable;
                setProcessorParams(curvable);
                setProcessorStaticParams(staticParams);
                // 初始化还没有视口的参数 (优化，直接读写 Ref)
                const nextViews = { ...paramViewsRef.current };
                let viewsChanged = false;
                for (const p of curvable) {
                    if (!nextViews[p.id] && p.kind.type === "automation_curve") {
                        const { min_value, max_value, default_value } = p.kind;
                        const span = max_value - min_value;
                        nextViews[p.id] = {
                            center: default_value,
                            span: span > 0 ? span : 1,
                        };
                        viewsChanged = true;
                    }
                }
                if (viewsChanged) {
                    paramViewsRef.current = nextViews;
                    invalidate(); // 数据有初始化，通知画布重绘
                }

                if (!rootTrackId || staticParams.length === 0) {
                    setProcessorStaticValues({});
                    return;
                }

                Promise.all(
                    staticParams.map((param) => paramsApi.getStaticParam(rootTrackId, param.id)),
                )
                    .then((values) => {
                        if (cancelled) return;
                        const nextValues: Record<string, number> = {};
                        for (const item of values) {
                            if (item.ok) {
                                nextValues[item.param] = item.value;
                            }
                        }
                        setProcessorStaticValues(nextValues);
                    })
                    .catch(() => {
                        if (!cancelled) {
                            setProcessorStaticValues({});
                        }
                    });
            })
            .catch(() => {
                if (!cancelled) {
                    processorParamsRef.current = [];
                    setProcessorParams([]);
                    setProcessorStaticParams([]);
                    setProcessorStaticValues({});
                }
            });
        return () => {
            cancelled = true;
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [rootTrack?.pitchAnalysisAlgo, rootTrackId]);

    const handleStaticParamChange = useCallback(
        async (paramId: string, value: number) => {
            if (!rootTrackId) return;
            try {
                const result = await paramsApi.setStaticParam(rootTrackId, paramId, value, true);
                if (result.ok) {
                    setProcessorStaticValues((prev) => ({
                        ...prev,
                        [paramId]: value,
                    }));
                }
            } catch (err) {
                // 静态参数调整是滑块 onValueChange 的 fire-and-forget 调用，
                // IPC 失败必须兜底，否则 unhandledrejection 且滑块值不一致。
                console.error("[setStaticParam] failed:", err);
            }
        },
        [rootTrackId],
    );

    const getProcessorParamLabel = useCallback(
        (param: ProcessorParamDescriptor) => {
            switch (param.id) {
                case "breath_enabled":
                    // 完整参数名。语义是「是否做谐波/噪声分离」，不是「气声开关」
                    // —— 气声与张力都依赖它。工具栏上开关以**纯图标**呈现（见渲染处），
                    // 全称只在需要完整名字的场合使用。
                    return t("breath_separation_label");
                case "breath_gain":
                    return t("breath_gain_label");
                case "hifigan_tension":
                    return t("hifigan_tension_label");
                case "formant_shift_cents":
                    return t("formant_shift_label");
                case "hifigan_volume":
                case "volume":
                    return t("volume_label");
                case "dyn":
                case "dyn_edit":
                    return t("dyn_label");
                case "synth_mode":
                    return t("vslib_synth_mode_label");
                case "pan":
                    return t("pan_label");
                case "breathiness":
                    return t("vslib_breathiness_label");
                default:
                    return param.display_name;
            }
        },
        [t],
    );

    // 工具栏参数按钮的简写标签（非中文语系用三个大写字形的缩写，如 PIT/BRE/VOL）。
    // 全称仍通过 getProcessorParamLabel() 提供，用于 ToolTip。
    const getProcessorParamShortLabel = useCallback(
        (param: ProcessorParamDescriptor) => {
            switch (param.id) {
                case "breath_gain":
                    return t("param_btn_breath");
                // `breath_enabled`（谐波/噪声分离）不在这条短标签路径上 ——
                // 它以**纯图标分段**渲染在被门禁组的组首（见工具栏渲染处），
                // 不显示任何文字，因此没有"短标签"可返回。
                case "hifigan_tension":
                case "tension":
                    return t("param_btn_tension");
                case "formant_shift_cents":
                    return t("param_btn_formant");
                case "hifigan_volume":
                case "volume":
                case "vslib_volume":
                    return t("param_btn_volume");
                case "dyn":
                case "dyn_edit":
                    return t("param_btn_dyn");
                case "pan":
                    return t("param_btn_pan");
                case "breathiness":
                    return t("param_btn_breathiness");
                default:
                    return getProcessorParamLabel(param);
            }
        },
        [getProcessorParamLabel, t],
    );

    const getStaticOptionLabel = useCallback(
        (paramId: string, label: string, value: number) => {
            // `breath_enabled` 不在此列：它以纯图标分段渲染在参数组内，
            // 不再走"静态选项 → 文本"的路径（见工具栏渲染处）。
            if (paramId === "synth_mode") {
                if (value === 0) return t("vslib_synth_mode_mono");
                if (value === 1) return t("vslib_synth_mode_mono_formant");
                if (value === 2) return t("vslib_synth_mode_chorus");
            }
            return label;
        },
        [t],
    );

    return {
        processorParams,
        processorParamsRef,
        processorStaticParams,
        processorStaticValues,
        orderedProcessorParams,
        currentParamRange,
        currentParamDefaultValue,
        currentParamQuantizeUnit,
        handleStaticParamChange,
        getProcessorParamLabel,
        getProcessorParamShortLabel,
        getStaticOptionLabel,
    };
}
