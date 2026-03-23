// 简单的类型声明以避免在非 Tauri 或缺少类型声明时的编译错误。
// 该文件声明了 @tauri-apps/api/dialog 模块的存在，详细类型可按需补充。
declare module "@tauri-apps/api/dialog" {
    export function open(options?: any): Promise<string | string[] | null>;
    export function save(options?: any): Promise<string | null>;
    const _default: unknown;
    export default _default;
}
