/**
 * 共享 hook：获取已注册的外部 Resampler 列表
 *
 * 供 TrackList 和 PianoRollPanel 的算法选择下拉框使用，
 * 在下拉框中动态展示 "ext:<id>" 格式的外部 Resampler 选项。
 *
 * 通过监听全局自定义事件 "resampler-list-changed" 实现实时刷新：
 * 当 ResamplerManagerDialog 中增删 Resampler 后会派发该事件，
 * 所有使用本 hook 的组件会自动重新拉取最新列表。
 */
import { useEffect, useState, useCallback } from "react";
import {
    listResamplers,
    type ResamplerEntry,
} from "../services/api/resampler";

/**
 * 全局事件名：Resampler 列表发生变更时派发。
 * ResamplerManagerDialog 在添加/删除 Resampler 后调用
 * `dispatchResamplerChanged()` 来通知所有监听者刷新。
 */
export const RESAMPLER_CHANGED_EVENT = "resampler-list-changed";

/** 派发 Resampler 列表变更事件（供 ResamplerManagerDialog 调用） */
export function dispatchResamplerChanged(): void {
    window.dispatchEvent(new CustomEvent(RESAMPLER_CHANGED_EVENT));
}

/** 内置算法 ID 列表 */
export const BUILTIN_ALGO_IDS = [
    "world_dll",
    "nsf_hifigan_onnx",
    "vslib",
    "none",
] as const;

/** 判断给定算法 ID 是否为外部 Resampler（以 "ext:" 开头） */
export function isExternalResampler(algo: string): boolean {
    return algo.startsWith("ext:");
}

/** 从 "ext:<id>" 中提取 resampler id */
export function extractResamplerId(algo: string): string {
    return algo.replace(/^ext:/, "");
}

/** 根据 ResamplerEntry 构建 pitchAnalysisAlgo 值 */
export function buildExtAlgoValue(entry: ResamplerEntry): string {
    return `ext:${entry.id}`;
}

/**
 * Hook: 从后端获取已注册的外部 Resampler 列表。
 * 返回列表以及手动刷新函数。
 *
 * - 组件挂载时自动加载一次
 * - 监听全局 "resampler-list-changed" 事件，自动刷新列表
 * - 也可手动调用 refresh() 来更新
 */
export function useResamplerList() {
    const [resamplers, setResamplers] = useState<ResamplerEntry[]>([]);

    const refresh = useCallback(() => {
        void listResamplers()
            .then(setResamplers)
            .catch(() => {});
    }, []);

    useEffect(() => {
        // 挂载时立即加载
        refresh();

        // 监听全局变更事件，实时刷新
        const handler = () => refresh();
        window.addEventListener(RESAMPLER_CHANGED_EVENT, handler);
        return () => {
            window.removeEventListener(RESAMPLER_CHANGED_EVENT, handler);
        };
    }, [refresh]);

    return { resamplers, refresh } as const;
}

/**
 * 获取当前算法值在下拉框中的规范化 value：
 * - 内置算法直接返回
 * - 外部 Resampler 返回 "ext:<id>"
 * - 未识别的值 fallback 到 "nsf_hifigan_onnx"
 */
export function normalizeAlgoValue(
    algo: string,
    resamplers: ResamplerEntry[],
): string {
    if ((BUILTIN_ALGO_IDS as readonly string[]).includes(algo)) {
        return algo;
    }
    if (isExternalResampler(algo)) {
        const id = extractResamplerId(algo);
        if (resamplers.some((r) => r.id === id)) {
            return algo;
        }
    }
    return "nsf_hifigan_onnx";
}
