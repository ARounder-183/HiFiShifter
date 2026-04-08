export const BUILTIN_TEMPLATE_PREFIX = "builtin:";
export const CUSTOM_TEMPLATE_PREFIX = "custom:";

export function getSelectedCustomScaleId(
    templateValue: string,
    customPresetIds: readonly string[],
): string | null {
    if (!templateValue.startsWith(CUSTOM_TEMPLATE_PREFIX)) {
        return null;
    }

    const presetId = templateValue.slice(CUSTOM_TEMPLATE_PREFIX.length);
    return customPresetIds.includes(presetId) ? presetId : null;
}

export function canDeleteSelectedCustomScale(
    templateValue: string,
    customPresetIds: readonly string[],
): boolean {
    return getSelectedCustomScaleId(templateValue, customPresetIds) !== null;
}

export function buildCustomScaleTemplateValue(prefix: "builtin" | "custom", value: string): string {
    const normalizedPrefix =
        prefix === "builtin" ? BUILTIN_TEMPLATE_PREFIX : CUSTOM_TEMPLATE_PREFIX;
    return `${normalizedPrefix}${value}`;
}
