/**
 * 移动端的"选一个文件读进内存"。
 *
 * 【为什么需要它】`<input type="file">` 在 Android 上**点了没有任何反应**：
 * wry 0.57 的 `RustWebChromeClient` 没有实现 `onShowFileChooser`，而系统文件
 * 选择器只有宿主能发起。桌面不受影响（`<input>` 由浏览器原生接管、零 IPC），
 * 所以这里是**移动形态专属**的替代路径。
 *
 * 三处调用点（导入外观主题 / 导入布局 / 往记事本插入图片）要的都是"字节 + 文件名"，
 * 与 `<input>` 给出的 `File` 完全同形 —— 于是调用方一行都不用改，
 * `AppFileInput` 内部换成这条路即可。
 *
 * 【为什么限制大小】它是"把整个文件塞进一次 IPC 的 base64 字符串"的形状。
 * 误选一个几百 MB 的音频时，明确报错远好过让 WebView 因为一个巨大的字符串而 OOM。
 */
import { isMobileForm } from "./hostDetect";
import { invoke } from "./invoke";

/** 默认上限：8 MB。文档与图片足够，且不会让 IPC 负载失控。 */
export const MOBILE_PICK_MAX_BYTES = 8 * 1024 * 1024;

interface PickFilesBytesResult {
    ok: boolean;
    canceled?: boolean;
    error?: string;
    files?: Array<{ name: string; base64: string }>;
}

/**
 * 从 `accept` 属性里抽出扩展名。
 *
 * `accept` 是 `<input type="file">` 的既有语法：`.json,image/png,image/*`。
 * 扩展名直接可用；MIME 类型要翻译成扩展名（`image/png` → `png`），
 * 而通配形式（`image/*`）翻译不出来，一律跳过 —— 抽不出来时返回空数组
 * = 不限制，用户能选到任何文件（随后由调用方校验内容）。
 */
export function extensionsFromAccept(accept?: string): string[] {
    if (!accept) return [];
    const extensions: string[] = [];
    for (const raw of accept.split(",")) {
        const token = raw.trim().toLowerCase();
        if (!token) continue;
        if (token.startsWith(".") && token.length > 1) {
            extensions.push(token.slice(1));
            continue;
        }
        if (token.includes("/") && !token.includes("*")) {
            const subtype = token.split("/")[1]?.split(";")[0]?.trim();
            if (subtype && !subtype.includes("*")) extensions.push(subtype);
        }
    }
    return Array.from(new Set(extensions));
}

/**
 * base64 → 字节缓冲区。`atob` 给出的是"每个字符一个字节"的二进制串。
 *
 * 【为什么返回 `ArrayBuffer` 而不是 `Uint8Array`】`BlobPart` 要求
 * `ArrayBufferView<ArrayBuffer>`，而 `new Uint8Array(n)` 的静态类型是
 * `Uint8Array<ArrayBufferLike>`（可能是 `SharedArrayBuffer`），两者不兼容。
 * 直接持有缓冲区既避开了这层协变，也少一次视图包装。
 */
function decodeBase64(value: string): ArrayBuffer {
    const binary = atob(value);
    const buffer = new ArrayBuffer(binary.length);
    const bytes = new Uint8Array(buffer);
    for (let index = 0; index < binary.length; index += 1) {
        bytes[index] = binary.charCodeAt(index);
    }
    return buffer;
}

/** 是否应该走移动端的原生选择器（而不是 `<input>`）。 */
export function shouldUseNativePicker(): boolean {
    return isMobileForm();
}

/**
 * 打开系统选择器并把选中的文件读成 `File` 对象；用户取消返回空数组。
 *
 * 【为什么失败时抛而不是返回空】"取消"与"选择器不可用"必须能被调用方区分：
 * 前者什么都不该发生，后者要让用户知道这个入口坏了（静默 no-op 是本仓明令禁止的
 * 形状）。取消在这里返回空数组，其余情况抛 `Error`。
 */
export async function pickFilesOnMobile(options: {
    accept?: string;
    multiple?: boolean;
    maxBytes?: number;
}): Promise<File[]> {
    const result = await invoke<PickFilesBytesResult>(
        "pick_files_bytes",
        extensionsFromAccept(options.accept),
        options.multiple === true,
        options.maxBytes ?? MOBILE_PICK_MAX_BYTES,
    );
    if (!result.ok) {
        throw new Error(result.error ?? "pick_files_bytes failed");
    }
    if (result.canceled) return [];
    return (result.files ?? []).map((file) => {
        const buffer = decodeBase64(file.base64);
        // 第二参数是文件名 —— 与 <input> 给出的对象同形，调用方无需分辨来源。
        return new File([buffer], file.name);
    });
}
