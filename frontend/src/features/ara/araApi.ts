// ARA 命令通过既有调用门面传输，宿主 token 与管道信息不进入前端。
import { invoke } from "../../services/invoke";

export interface AraInstance {
    instance_id: string;
    name: string;
    pid: number;
}
export interface AraResult {
    ok: boolean;
    error?: string;
    instance_id?: string;
    revision?: number;
    model_revision?: number;
}
export const araApi = {
    list: () => invoke<AraInstance[]>("ara_list_instances"),
    connect: (instanceId: string, force: boolean) =>
        invoke<AraResult>("ara_connect", instanceId, force),
    refresh: (force: boolean) => invoke<AraResult>("ara_refresh", force),
    submit: () => invoke<AraResult>("ara_submit"),
    disconnect: () => invoke<AraResult>("ara_disconnect"),
};

/** 保留后端 Conflict 和脏工程诊断，不展示调用包装器的通用错误。 */
export function araError(error: unknown): string {
    if (error instanceof Error)
        return error.cause !== undefined ? araError(error.cause) : error.message;
    return String(error);
}
