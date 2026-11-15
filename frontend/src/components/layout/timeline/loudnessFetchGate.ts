/**
 * 兼容旧路径：真实状态已迁到 features/session/loudnessFetchGate（interaction 切片）。
 * 这层 re-export 只服务仍按 timeline 相对路径导入的调用方，避免把 1351 行预算
 * 的 useTimelineKernelTrim 因一条 import 路径变长而撑破。
 */
export * from "../../../features/session/loudnessFetchGate";
