import { useDispatch, useSelector, useStore, type TypedUseSelectorHook } from "react-redux";
import type { AppDispatch, RootState } from "./store";

export const useAppDispatch = () => useDispatch<AppDispatch>();
export const useAppSelector: TypedUseSelectorHook<RootState> = useSelector;
/** 读取 Redux store 本身（可在事件监听器里同步读取最新状态，避免闭包过期）。 */
export const useAppStore = () => useStore<RootState>();