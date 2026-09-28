/**
 * 类名拼接助手。
 *
 * 直接复用已依赖的 `clsx`，只做一层语义命名：原语内部一律用 `cx(...)`，
 * 避免每个组件各自 `import clsx` 造成风格不一。
 */
import clsx, { type ClassValue } from "clsx";

export function cx(...parts: ClassValue[]): string {
    return clsx(parts);
}

export type { ClassValue };
