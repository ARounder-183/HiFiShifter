/**
 * 宿主淡化轴的本地显示投影。
 *
 * hostFades 不是持久化字段，而是后端随 TimelineState 下发的只读显示元数据；
 * 权威快照会整份替换它。连续轴宿主（REAPER >= 7.81）的画布、波形与命中测试只读
 * hostFades，因此曲率拖拽期间若只改 fadeInDir/fadeInS，画面会停在旧值，直到松手
 * 后的回执才跳变。这里把待提交的轴镜像进 hostFades，让"本地预览"和"权威回执"
 * 使用同构的字段。
 *
 * 与 backend host_edit.rs 的优先级保持同一口径：
 * - continuous + 合法 shape：curvature 取预设值，补丁里的 dir 被忽略；
 * - 同一补丁显式给了 S：S 以显式值为准，否则取预设 S；
 * - 没有 shape 时 dir/S 各自生效；
 * - 形状不是 0..6 的整数：后端不会写任何轴，本地也不猜。
 */
import type { HostFadeMetadata } from "../../types/api";
import { hostFadePresetAxes } from "../../utils/hostFadeAxes";

export interface FadeAxesPatch {
    fadeInShape?: number;
    fadeOutShape?: number;
    fadeInDir?: number;
    fadeOutDir?: number;
    fadeInS?: number;
    fadeOutS?: number;
}

export interface HostFadeAxesPreviewResult {
    hostFades: HostFadeMetadata | undefined;
    /**
     * 需要回写到 clip 本地字段的修正（目前只有 continuous 形状切换会同时纠正
     * dir/S，避免下一次拖拽从旧轴值起步而第一帧跳变）。
     */
    correctedPatch: FadeAxesPatch;
}

function clampAxis(value: number): number {
    return Math.min(1, Math.max(-1, value));
}

function finiteAxis(value: number | undefined): number | undefined {
    if (value === undefined) return undefined;
    const numeric = Number(value);
    return Number.isFinite(numeric) ? clampAxis(numeric) : undefined;
}

/** 把本地待提交的连续轴镜像到 hostFades 的显示投影（纯函数，不修改入参）。 */
export function mirrorHostFadeAxes(
    metadata: HostFadeMetadata | undefined,
    patch: FadeAxesPatch,
): HostFadeMetadata | undefined {
    return applyHostFadeAxesPreview(metadata, patch).hostFades;
}

/**
 * 与 mirrorHostFadeAxes 同源，但额外返回本地 fadeInDir/S 的修正值。
 *
 * reducer 用它同时完成两件事：写 hostFades 投影，以及让 clip 的本地轴与
 * 后端在 shape 补丁上实际写入的轴一致。
 */
export function applyHostFadeAxesPreview(
    metadata: HostFadeMetadata | undefined,
    patch: FadeAxesPatch,
): HostFadeAxesPreviewResult {
    if (!metadata || metadata.curve_mode !== "continuous") {
        return { hostFades: metadata, correctedPatch: {} };
    }

    const correctedPatch: FadeAxesPatch = {};
    let next = metadata;
    let changed = false;

    const update = (
        curvatureKey: "in_curvature" | "out_curvature",
        sKey: "in_s" | "out_s",
        curvature: number,
        s: number,
    ) => {
        if (next[curvatureKey] === curvature && next[sKey] === s) return;
        if (!changed) next = { ...metadata };
        changed = true;
        next[curvatureKey] = curvature;
        next[sKey] = s;
    };

    for (const side of ["in", "out"] as const) {
        const shapeKey = side === "in" ? "fadeInShape" : "fadeOutShape";
        const dirKey = side === "in" ? "fadeInDir" : "fadeOutDir";
        const sKey = side === "in" ? "fadeInS" : "fadeOutS";
        const curvatureKey = side === "in" ? "in_curvature" : "out_curvature";
        const metadataSKey = side === "in" ? "in_s" : "out_s";

        const shape = patch[shapeKey];
        if (shape !== undefined) {
            const preset = hostFadePresetAxes(shape);
            // 非法形状后端整段跳过（连 dir 都不写）；本地同样不制造一个后端不会写的组合。
            if (!preset) continue;
            const explicitS = finiteAxis(patch[sKey]);
            const curvature = preset[0];
            const s = explicitS ?? preset[1];
            correctedPatch[dirKey] = curvature;
            correctedPatch[sKey] = s;
            update(curvatureKey, metadataSKey, curvature, s);
            continue;
        }

        const explicitDir = finiteAxis(patch[dirKey]);
        if (explicitDir !== undefined) {
            correctedPatch[dirKey] = explicitDir;
            update(curvatureKey, metadataSKey, explicitDir, next[metadataSKey]);
        }
        const explicitS = finiteAxis(patch[sKey]);
        if (explicitS !== undefined) {
            correctedPatch[sKey] = explicitS;
            update(curvatureKey, metadataSKey, next[curvatureKey], explicitS);
        }
    }

    return { hostFades: changed ? next : metadata, correctedPatch };
}

/**
 * 把显示投影同时落到 clip 的 hostFades 与本地连续轴字段上。
 *
 * reducer 在写完原始 patch 后调用一次即可：单侧拖拽、交叉点、菜单滑杆、形状按钮、
 * 双击重置与远端乐观 pending 都走同一条路径。
 */
export function applyHostFadeAxesPreviewToClip(
    clip: {
        hostFades?: HostFadeMetadata;
        fadeInDir: number;
        fadeOutDir: number;
        fadeInS?: number;
        fadeOutS?: number;
    },
    patch: FadeAxesPatch,
): void {
    const preview = applyHostFadeAxesPreview(clip.hostFades, patch);
    if (preview.correctedPatch.fadeInDir !== undefined) {
        clip.fadeInDir = preview.correctedPatch.fadeInDir;
    }
    if (preview.correctedPatch.fadeOutDir !== undefined) {
        clip.fadeOutDir = preview.correctedPatch.fadeOutDir;
    }
    if (preview.correctedPatch.fadeInS !== undefined) {
        clip.fadeInS = preview.correctedPatch.fadeInS;
    }
    if (preview.correctedPatch.fadeOutS !== undefined) {
        clip.fadeOutS = preview.correctedPatch.fadeOutS;
    }
    if (preview.hostFades !== clip.hostFades) {
        clip.hostFades = preview.hostFades;
    }
}

export interface FadeRestoreBase {
    fadeInShape: number;
    fadeOutShape: number;
    fadeInDir: number;
    fadeOutDir: number;
    fadeInS: number;
    fadeOutS: number;
}

/** 取消手势时的完整轴还原 patch；旧轴宿主不得携带 S 字段。 */
export function fadeRestorePatch(args: {
    base: FadeRestoreBase;
    continuous: boolean;
}): FadeAxesPatch {
    const { base, continuous } = args;
    return {
        fadeInShape: base.fadeInShape,
        fadeOutShape: base.fadeOutShape,
        fadeInDir: base.fadeInDir,
        fadeOutDir: base.fadeOutDir,
        ...(continuous ? { fadeInS: base.fadeInS, fadeOutS: base.fadeOutS } : {}),
    };
}
