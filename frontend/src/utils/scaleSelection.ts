/**
 * 音阶选择工具。
 *
 * 统一管理“基准音阶”下拉框所需的数据结构：
 * - 内置音阶
 * - 工程音阶（__project__）
 * - 用户自定义音阶（token 化）
 */

import { SCALE_KEYS, SCALE_LABELS } from "./musicalScales";
import type { CustomScalePreset } from "./customScales";

export const CUSTOM_SCALE_TOKEN_PREFIX = "__custom_scale__:";

export interface ScaleSelectOption {
    value: string;
    label: string;
}

export interface ScaleSelectGroups {
    projectOption: ScaleSelectOption;
    builtinOptions: ScaleSelectOption[];
    customOptions: ScaleSelectOption[];
    wheelOptions: string[];
}

export function toCustomScaleToken(customScaleId: string): string {
    return `${CUSTOM_SCALE_TOKEN_PREFIX}${customScaleId}`;
}

export function parseCustomScaleToken(scaleToken: string): string | null {
    if (!scaleToken.startsWith(CUSTOM_SCALE_TOKEN_PREFIX)) {
        return null;
    }
    const id = scaleToken.slice(CUSTOM_SCALE_TOKEN_PREFIX.length).trim();
    return id.length > 0 ? id : null;
}

export function buildScaleSelectGroups(
    projectScaleLabel: string,
    customScalePresets: readonly CustomScalePreset[],
): ScaleSelectGroups {
    const projectOption: ScaleSelectOption = {
        value: "__project__",
        label: projectScaleLabel,
    };

    const builtinOptions: ScaleSelectOption[] = SCALE_KEYS.map((key) => ({
        value: key,
        label: SCALE_LABELS[key],
    }));

    const customOptions: ScaleSelectOption[] = [];
    const seenIds = new Set<string>();
    for (const preset of customScalePresets) {
        const id = String(preset.id ?? "").trim();
        if (!id || seenIds.has(id)) {
            continue;
        }
        seenIds.add(id);
        customOptions.push({
            value: toCustomScaleToken(id),
            label: String(preset.name ?? "").trim() || id,
        });
    }

    const wheelOptions = [
        projectOption.value,
        ...builtinOptions.map((option) => option.value),
        ...customOptions.map((option) => option.value),
    ];

    return {
        projectOption,
        builtinOptions,
        customOptions,
        wheelOptions,
    };
}
