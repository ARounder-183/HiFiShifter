/**
 * 外部 Resampler 注册表 API
 *
 * 封装后端的 resampler 管理命令，供前端组件使用。
 */
import { invoke } from '../invoke';

// ─── 类型定义 ─────────────────────────────────────────────────────────────────

/** 单个 Flag 参数定义 */
export interface FlagParam {
    key: string;
    displayName: string;
    minValue: number;
    maxValue: number;
    defaultValue: number;
}

/** 外部 Resampler 条目 */
export interface ResamplerEntry {
    id: string;
    displayName: string;
    exePath: string;
    defaultFlags: string;
    flagParams: FlagParam[];
    available: boolean;
}

/** 列表响应 */
interface ResamplerListPayload {
    ok: boolean;
    entries: ResamplerEntry[];
}

/** 操作响应 */
interface ResamplerOpPayload {
    ok: boolean;
    error?: string;
    entry?: ResamplerEntry;
}

// ─── API 函数 ─────────────────────────────────────────────────────────────────

/** 列出所有已注册的外部 Resampler（每次调用会刷新可用性） */
export async function listResamplers(): Promise<ResamplerEntry[]> {
    const payload = await invoke<ResamplerListPayload>('list_resamplers');
    return payload.entries ?? [];
}

/** 添加一个外部 Resampler */
export async function addResampler(
    displayName: string,
    exePath: string,
    defaultFlags?: string,
    flagParams?: FlagParam[],
): Promise<ResamplerEntry | null> {
    const payload = await invoke<ResamplerOpPayload>(
        'add_resampler',
        displayName,
        exePath,
        defaultFlags,
        flagParams,
    );
    if (!payload.ok) {
        console.error('[resampler] add failed:', payload.error);
        return null;
    }
    return payload.entry ?? null;
}

/** 移除一个已注册的外部 Resampler */
export async function removeResampler(id: string): Promise<boolean> {
    const payload = await invoke<ResamplerOpPayload>('remove_resampler', id);
    if (!payload.ok) {
        console.error('[resampler] remove failed:', payload.error);
    }
    return payload.ok;
}

/** 更新已注册 Resampler 的信息 */
export async function updateResampler(
    id: string,
    opts: {
        displayName?: string;
        exePath?: string;
        defaultFlags?: string;
        flagParams?: FlagParam[];
    },
): Promise<ResamplerEntry | null> {
    const payload = await invoke<ResamplerOpPayload>(
        'update_resampler',
        id,
        opts.displayName,
        opts.exePath,
        opts.defaultFlags,
        opts.flagParams,
    );
    if (!payload.ok) {
        console.error('[resampler] update failed:', payload.error);
        return null;
    }
    return payload.entry ?? null;
}

/** 扫描指定目录下的 Resampler 可执行文件 */
export async function scanResamplers(
    directory: string,
): Promise<ResamplerEntry[]> {
    const payload = await invoke<ResamplerListPayload>(
        'scan_resamplers',
        directory,
    );
    return payload.entries ?? [];
}

/** 打开文件选择对话框让用户选择 Resampler exe */
export async function browseResamplerExe(): Promise<string | null> {
    return invoke<string | null>('browse_resampler_exe');
}
