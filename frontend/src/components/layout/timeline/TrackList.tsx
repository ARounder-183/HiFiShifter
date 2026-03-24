import React, {
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Flex, Box, Text, IconButton, Slider, Select } from "@radix-ui/themes";
import { Cross2Icon, PlusIcon } from "@radix-ui/react-icons";
import type { TrackInfo } from "../../../features/session/sessionTypes";
import type { MessageKey } from "../../../i18n/messages";
import { TRACK_ADD_ROW_HEIGHT } from "./constants";

/** 轨道颜色调色板（与后端 add_track 预设一致） */
const TRACK_COLOR_PALETTE_KEYS: { value: string; key: MessageKey }[] = [
  { value: "#4f8ef7", key: "color_blue" },
  { value: "#a78bfa", key: "color_purple" },
  { value: "#34d399", key: "color_green" },
  { value: "#fb923c", key: "color_orange" },
  { value: "#f472b6", key: "color_pink" },
  { value: "#38bdf8", key: "color_sky_blue" },
  { value: "#facc15", key: "color_yellow" },
  { value: "#f87171", key: "color_red" },
];

export const TrackList: React.FC<{
  t: (key: MessageKey) => string;
  tracks: TrackInfo[];
  selectedTrackId: string | null;
  rowHeight: number;
  trackVolumeUi: Record<string, number>;
  onSelectTrack: (trackId: string) => void;
  onRemoveTrack: (trackId: string) => void;
  onMoveTrack: (payload: {
    trackId: string;
    targetIndex: number;
    parentTrackId: string | null;
  }) => void;
  onToggleMute: (trackId: string, nextMuted: boolean) => void;
  onToggleSolo: (trackId: string, nextSolo: boolean) => void;
  onToggleCompose: (trackId: string, nextComposeEnabled: boolean) => void;
  onVolumeUiChange: (trackId: string, nextVolume: number) => void;
  onVolumeCommit: (trackId: string, nextVolume: number) => void;
  onAddTrack: () => void;
  onTrackColorChange?: (trackId: string, color: string) => void;
  onAlgoChange?: (trackId: string, algo: string) => void;
  onTrackNameChange?: (trackId: string, name: string) => void;
  onDuplicateTrack?: (trackId: string) => void;
  onScrollTopChange?: (scrollTop: number) => void;
  /** 外部持有该滚动容器的 ref，用于同步右侧轨道区的竖向滚动 */
  listScrollRef?: React.MutableRefObject<HTMLDivElement | null>;
}> = ({
  t,
  tracks,
  selectedTrackId,
  rowHeight,
  trackVolumeUi,
  onSelectTrack,
  onRemoveTrack,
  onMoveTrack,
  onToggleMute,
  onToggleSolo,
  onToggleCompose,
  onVolumeUiChange,
  onVolumeCommit,
  onAddTrack,
  onTrackColorChange,
  onAlgoChange,
  onTrackNameChange,
  onDuplicateTrack,
  onScrollTopChange,
  listScrollRef,
}) => {
  const listRef = useRef<HTMLDivElement | null>(null);
  const panRef = useRef<{
    pointerId: number | null;
    startY: number;
    scrollTop: number;
  } | null>(null);
  const dragRef = useRef<{
    pointerId: number;
    trackId: string;
    startClientX: number;
    startClientY: number;
    hasMoved: boolean;
    originalParentId: string | null;
    originalIndexSelf: number;
  } | null>(null);

  const [dragUi, setDragUi] = useState<{
    draggingTrackId: string;
    overTrackId: string | null;
    mode: "reorder" | "nest";
    indicatorY: number | null;
  } | null>(null);

  // 轨道颜色选择器弹出状态
  const [colorPickerTrackId, setColorPickerTrackId] = useState<string | null>(
    null,
  );

  // 轨道名称行内编辑状态
  const [editingTrackId, setEditingTrackId] = useState<string | null>(null);
  const [editingName, setEditingName] = useState("");
  const nameInputRef = useRef<HTMLInputElement | null>(null);

  // 轨道右键菜单状态
  const [trackCtxMenu, setTrackCtxMenu] = useState<{
    x: number;
    y: number;
    trackId: string;
  } | null>(null);
  const trackCtxMenuRef = useRef<HTMLDivElement | null>(null);

  // 自动修正菜单溢出屏幕
  useLayoutEffect(() => {
    const el = trackCtxMenuRef.current;
    if (!el || !trackCtxMenu) return;
    const rect = el.getBoundingClientRect();
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    if (rect.right > vw) {
      el.style.left = `${Math.max(0, vw - rect.width)}px`;
    }
    if (rect.bottom > vh) {
      el.style.top = `${Math.max(0, vh - rect.height)}px`;
    }
  }, [trackCtxMenu]);

  // 点击其他区域关闭右键菜单
  useEffect(() => {
    if (!trackCtxMenu) return;
    const handler = (e: PointerEvent) => {
      const target = e.target as HTMLElement | null;
      if (target?.closest?.("[data-track-ctx-menu]")) return;
      setTrackCtxMenu(null);
    };
    window.addEventListener("pointerdown", handler, true);
    return () => window.removeEventListener("pointerdown", handler, true);
  }, [trackCtxMenu]);

  function commitTrackName() {
    if (!editingTrackId) return;
    const trimmed = editingName.trim();
    if (trimmed && onTrackNameChange) {
      onTrackNameChange(editingTrackId, trimmed);
    }
    setEditingTrackId(null);
  }

  // 点击其他区域关闭颜色选择器
  useEffect(() => {
    if (!colorPickerTrackId) return;
    const handler = (e: PointerEvent) => {
      const target = e.target as HTMLElement | null;
      if (target?.closest?.("[data-track-color-picker]")) return;
      setColorPickerTrackId(null);
    };
    window.addEventListener("pointerdown", handler, true);
    return () => window.removeEventListener("pointerdown", handler, true);
  }, [colorPickerTrackId]);

  const parentById = useMemo(() => {
    const m = new Map<string, string | null>();
    for (const tr of tracks) {
      m.set(tr.id, tr.parentId ?? null);
    }
    return m;
  }, [tracks]);

  useEffect(() => {
    return () => {
      // Safety cleanup.
      dragRef.current = null;
      setDragUi(null);
    };
  }, []);

  function isEditableTarget(target: EventTarget | null) {
    const el = target as HTMLElement | null;
    if (!el) return false;
    const tag = (el.tagName ?? "").toLowerCase();
    if (tag === "input" || tag === "textarea" || tag === "select") return true;
    if (el.isContentEditable) return true;
    if (el.closest?.('input,textarea,select,[contenteditable="true"]')) return true;
    return false;
  }

  function startPanPointerLocal(e: React.PointerEvent) {
    // Intercept middle-button mouse to prevent browser native autoscroll
    if (e.pointerType === "mouse" && e.button === 1) {
      e.preventDefault();
      e.stopPropagation();
    }
    if (e.pointerType !== "mouse") return;
    if (e.button !== 1) return;
    if (isEditableTarget(e.target)) return;
    const el = listRef.current;
    if (!el) return;

    panRef.current = {
      pointerId: e.pointerId,
      startY: e.clientY,
      scrollTop: el.scrollTop,
    };

    const prevCursor = document.body.style.cursor;
    const prevSelect = document.body.style.userSelect;
    document.body.style.cursor = "grabbing";
    document.body.style.userSelect = "none";

    try {
      (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    } catch {}

    function onMove(ev: PointerEvent) {
      const pan = panRef.current;
      const cur = listRef.current;
      if (!pan || !cur) return;
      if (pan.pointerId != null && ev.pointerId !== pan.pointerId) return;
      cur.scrollTop = pan.scrollTop - (ev.clientY - pan.startY);
      onScrollTopChange?.(cur.scrollTop);
    }

    function end(ev: PointerEvent) {
      const pan = panRef.current;
      if (!pan) return;
      if (pan.pointerId != null && ev.pointerId !== pan.pointerId) return;
      panRef.current = null;
      document.body.style.cursor = prevCursor;
      document.body.style.userSelect = prevSelect;
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", end);
      window.removeEventListener("pointercancel", end);
    }

    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", end);
    window.addEventListener("pointercancel", end);
  }

  useEffect(() => {
    const el = listRef.current;
    if (!el) return;

    const handler: EventListener = (evt) => {
      const e = evt as WheelEvent;
      // Keep ctrl/meta wheel available for global zoom/system gestures.
      if (e.ctrlKey || e.metaKey) return;

      const useY = Math.abs(e.deltaY) >= Math.abs(e.deltaX);
      const delta = useY ? e.deltaY : e.deltaX;
      if (!Number.isFinite(delta) || Math.abs(delta) < 0.01) return;

      const maxScrollTop = Math.max(0, el.scrollHeight - el.clientHeight);
      if (maxScrollTop <= 0) return;

      const nextScrollTop = Math.max(
        0,
        Math.min(maxScrollTop, el.scrollTop + delta),
      );
      if (Math.abs(nextScrollTop - el.scrollTop) < 0.5) return;

      e.preventDefault();
      el.scrollTop = nextScrollTop;
      onScrollTopChange?.(nextScrollTop);
    };

    el.addEventListener("wheel", handler, {
      passive: false,
    } as AddEventListenerOptions);
    return () => {
      el.removeEventListener("wheel", handler);
    };
  }, [onScrollTopChange]);

  function wouldCreateCycle(trackId: string, parentTrackId: string | null) {
    let cur = parentTrackId;
    let guard = 0;
    while (cur && guard++ < 1000) {
      if (cur === trackId) return true;
      cur = parentById.get(cur) ?? null;
    }
    return false;
  }

  function siblingsOf(parentTrackId: string | null): string[] {
    const out: string[] = [];
    for (const tr of tracks) {
      if ((tr.parentId ?? null) === parentTrackId) out.push(tr.id);
    }
    return out;
  }

  function trackAtClientY(clientY: number): {
    track: TrackInfo | null;
    yInRow: number;
    index: number;
  } {
    const el = listRef.current;
    if (!el) return { track: null, yInRow: 0, index: -1 };
    const bounds = el.getBoundingClientRect();
    const y = clientY - bounds.top + el.scrollTop;
    const rawIdx = Math.floor(y / rowHeight);
    // 当鼠标在列表上方时，clamp 到第一个轨道（yInRow=0），使拖拽可以插入到顶部
    if (rawIdx < 0) {
      return { track: tracks[0] ?? null, yInRow: 0, index: 0 };
    }
    const yInRow = y - rawIdx * rowHeight;
    if (rawIdx >= tracks.length) return { track: null, yInRow, index: rawIdx };
    return { track: tracks[rawIdx] ?? null, yInRow, index: rawIdx };
  }

  function computeDropSpec(
    draggingTrackId: string,
    clientX: number,
    clientY: number,
  ): {
    parentTrackId: string | null;
    targetIndex: number;
    mode: "reorder" | "nest";
  } {
    const el = listRef.current;
    const bounds = el?.getBoundingClientRect();

    // 当鼠标在列表容器上方时，直接插入到顶层第一个位置
    if (bounds && clientY < bounds.top && tracks.length > 0) {
      return {
        parentTrackId: null,
        targetIndex: 0,
        mode: "reorder",
      };
    }

    const { track: over, yInRow } = trackAtClientY(clientY);

    // Dropping outside -> append as root.
    if (!over) {
      const roots = siblingsOf(null).filter((id) => id !== draggingTrackId);
      return {
        parentTrackId: null,
        targetIndex: roots.length,
        mode: "reorder",
      };
    }

    const overIndent = Math.max(0, (over.depth ?? 0) * 16);
    const localX = bounds ? clientX - bounds.left : clientX;
    const nest = over.id !== draggingTrackId && localX > 24 + overIndent + 40;

    if (nest) {
      const parentTrackId = over.id;
      if (wouldCreateCycle(draggingTrackId, parentTrackId)) {
        const roots = siblingsOf(null).filter((id) => id !== draggingTrackId);
        return {
          parentTrackId: null,
          targetIndex: roots.length,
          mode: "reorder",
        };
      }
      const children = siblingsOf(parentTrackId).filter(
        (id) => id !== draggingTrackId,
      );
      return {
        parentTrackId,
        targetIndex: children.length,
        mode: "nest",
      };
    }

    let parentTrackId = over.parentId ?? null;
    if (wouldCreateCycle(draggingTrackId, parentTrackId)) {
      parentTrackId = null;
    }

    if (over.id === draggingTrackId) {
      const siblingsIncl = siblingsOf(parentTrackId);
      const indexSelf = Math.max(0, siblingsIncl.indexOf(draggingTrackId));
      return { parentTrackId, targetIndex: indexSelf, mode: "reorder" };
    }

    const siblings = siblingsOf(parentTrackId).filter(
      (id) => id !== draggingTrackId,
    );
    const baseIndex = Math.max(0, siblings.indexOf(over.id));
    // 使用 35% 边缘区域：上 35% 插入到上方，下 35% 插入到下方，中间 30% 保持不动
    const edgeZone = rowHeight * 0.35;
    const insertAfter = yInRow > rowHeight - edgeZone;
    const insertBefore = yInRow < edgeZone;
    // 如果鼠标在中间区域，保持原位不触发重排
    if (!insertAfter && !insertBefore) {
      const siblingsIncl = siblingsOf(parentTrackId);
      const indexSelf = Math.max(0, siblingsIncl.indexOf(draggingTrackId));
      // 如果不在同一层级，则追加到末尾
      const targetIndex = indexSelf >= 0 ? indexSelf : siblings.length;
      return { parentTrackId, targetIndex, mode: "reorder" };
    }
    const targetIndex = Math.min(
      siblings.length,
      baseIndex + (insertAfter ? 1 : 0),
    );
    return { parentTrackId, targetIndex, mode: "reorder" };
  }

  return (
    <Flex
      direction="column"
      className="w-64 border-r border-qt-border bg-qt-window shrink-0"
    >
      <Box className="h-6 border-b border-qt-border px-2 flex items-center bg-qt-window shadow-sm z-10">
        <Text size="1" weight="bold" color="gray">
          {t("tracks")}
        </Text>
      </Box>
      <div
        ref={(el) => {
          (listRef as React.MutableRefObject<HTMLDivElement | null>).current =
            el;
          if (listScrollRef) listScrollRef.current = el;
        }}
        onPointerDown={(e) => startPanPointerLocal?.(e)}
        onAuxClick={(e) => {
          // Prevent native autoscroll overlay on middle click
          e.preventDefault();
          e.stopPropagation();
        }}
        onMouseDown={(e) => {
          if (e.button === 1) {
            e.preventDefault();
            e.stopPropagation();
          }
        }}
        className="flex-1 relative overflow-y-auto custom-scrollbar hide-v-scrollbar"
        onScroll={(e) => {
          onScrollTopChange?.((e.currentTarget as HTMLDivElement).scrollTop);
        }}
      >
        {dragUi?.mode === "reorder" && typeof dragUi.indicatorY === "number" ? (
          <div
            className="absolute left-1 right-1 pointer-events-none z-50"
            style={{ top: dragUi.indicatorY }}
          >
            <div className="h-px bg-qt-highlight" />
          </div>
        ) : null}
        {tracks.map((track) => {
          const selected = selectedTrackId === track.id;
          const depth = Math.max(0, Number(track.depth ?? 0) || 0);
          const indent = depth * 16;
          const dragging = dragUi?.draggingTrackId === track.id;
          const isOver = dragUi?.overTrackId === track.id;
          const muted = Boolean(track.muted);
          const solo = Boolean(track.solo);
          const isRoot = (track.parentId ?? null) == null;
          const composeEnabled = Boolean(track.composeEnabled);
          const backendVolume = Math.max(
            0,
            Math.min(1, Number(track.volume ?? 0.9)),
          );
          const uiOverride = trackVolumeUi[track.id];
          const volume = Number.isFinite(uiOverride)
            ? uiOverride
            : backendVolume;

          const guideLines = depth > 0 ? Array.from({ length: depth }) : [];

          return (
            <div
              key={track.id}
              style={{ height: rowHeight }}
              className="border-b border-qt-border relative group overflow-hidden"
              onContextMenu={(e) => {
                e.preventDefault();
                setTrackCtxMenu({
                  x: e.clientX,
                  y: e.clientY,
                  trackId: track.id,
                });
              }}
              onPointerDown={(e) => {
                if (e.button !== 0) return;

                // If the pointer down starts on an interactive control, do not start a drag.
                const target = e.target as HTMLElement | null;
                if (
                  target?.closest?.(
                    "button,[role='slider'],input,textarea,select,a",
                  )
                ) {
                  return;
                }

                const overSiblings = siblingsOf(track.parentId ?? null);
                const originalIndexSelf = Math.max(
                  0,
                  overSiblings.indexOf(track.id),
                );

                dragRef.current = {
                  pointerId: e.pointerId,
                  trackId: track.id,
                  startClientX: e.clientX,
                  startClientY: e.clientY,
                  hasMoved: false,
                  originalParentId: track.parentId ?? null,
                  originalIndexSelf,
                };

                const el = e.currentTarget as HTMLDivElement;
                el.setPointerCapture(e.pointerId);

                const prevCursor = document.body.style.cursor;
                const prevSelect = document.body.style.userSelect;

                function onMove(ev: PointerEvent) {
                  const drag = dragRef.current;
                  if (!drag || drag.pointerId !== e.pointerId) return;

                  if (!drag.hasMoved) {
                    const dx = ev.clientX - drag.startClientX;
                    const dy = ev.clientY - drag.startClientY;
                    if (dx * dx + dy * dy < 9) {
                      return;
                    }
                    drag.hasMoved = true;
                    document.body.style.cursor = "grabbing";
                    document.body.style.userSelect = "none";
                  }

                  const spec = computeDropSpec(
                    drag.trackId,
                    ev.clientX,
                    ev.clientY,
                  );
                  const overInfo = trackAtClientY(ev.clientY);
                  const over = overInfo.track;

                  let indicatorY: number | null = null;
                  if (spec.mode === "reorder") {
                    const listBounds = listRef.current?.getBoundingClientRect();
                    // 鼠标在列表上方时，指示线固定在顶部
                    if (listBounds && ev.clientY < listBounds.top) {
                      indicatorY = 0;
                    } else {
                      const idx = overInfo.index;
                      const edgeZone = rowHeight * 0.35;
                      if (!Number.isFinite(idx)) {
                        indicatorY = null;
                      } else if (!over) {
                        indicatorY = tracks.length * rowHeight;
                      } else {
                        const insertAfter =
                          overInfo.yInRow > rowHeight - edgeZone;
                        const insertBefore = overInfo.yInRow < edgeZone;
                        if (insertAfter) {
                          indicatorY = idx * rowHeight + rowHeight;
                        } else if (insertBefore) {
                          indicatorY = idx * rowHeight;
                        } else {
                          // 中间区域不显示指示线
                          indicatorY = null;
                        }
                      }
                    }
                  }

                  setDragUi({
                    draggingTrackId: drag.trackId,
                    overTrackId: over?.id ?? null,
                    mode: spec.mode,
                    indicatorY,
                  });
                }

                function end(ev: PointerEvent) {
                  const drag = dragRef.current;
                  if (!drag || drag.pointerId !== e.pointerId) return;
                  dragRef.current = null;

                  window.removeEventListener("pointermove", onMove);
                  window.removeEventListener("pointerup", end);
                  window.removeEventListener("pointercancel", end);

                  document.body.style.cursor = prevCursor;
                  document.body.style.userSelect = prevSelect;

                  const moved = drag.hasMoved;
                  setDragUi(null);

                  if (!moved) {
                    onSelectTrack(drag.trackId);
                    return;
                  }

                  const spec = computeDropSpec(
                    drag.trackId,
                    ev.clientX,
                    ev.clientY,
                  );

                  if (
                    spec.parentTrackId === drag.originalParentId &&
                    spec.targetIndex === drag.originalIndexSelf
                  ) {
                    return;
                  }

                  onMoveTrack({
                    trackId: drag.trackId,
                    targetIndex: spec.targetIndex,
                    parentTrackId: spec.parentTrackId,
                  });
                }

                window.addEventListener("pointermove", onMove);
                window.addEventListener("pointerup", end);
                window.addEventListener("pointercancel", end);
              }}
            >
              {/* Always-visible left accent bar (pinned to list edge) */}
              <div
                className={`absolute left-0 top-0 bottom-0 w-1 transition-opacity ${selected ? "opacity-100" : "opacity-80 group-hover:opacity-90"}`}
                style={{
                  backgroundColor: track.color || "var(--qt-highlight)",
                }}
              />

              {/* Left gutter: makes nesting depth visible at a glance */}
              <div
                className="absolute left-0 top-0 bottom-0 bg-qt-window pointer-events-none"
                style={{ width: indent }}
              >
                {guideLines.map((_, i) => (
                  <div
                    key={i}
                    className="absolute top-0 bottom-0 border-l border-qt-border opacity-60"
                    style={{ left: i * 16 + 8 }}
                  />
                ))}
                {depth > 0 ? (
                  <div
                    className="absolute border-t border-qt-border opacity-60"
                    style={{
                      left: (depth - 1) * 16 + 8,
                      right: 0,
                      top: "50%",
                    }}
                  />
                ) : null}
              </div>

              {/* Content block: shifted right by depth */}
              <Box
                className={`absolute top-0 bottom-0 right-0 bg-qt-base transition-colors overflow-hidden ${selected ? "bg-qt-button-hover" : "hover:bg-qt-button-hover"} ${dragging ? "opacity-60" : ""} ${isOver ? "bg-qt-button-hover" : ""}`}
                style={{ left: indent }}
              >
                {/* Keep a subtle in-row bar too, but don't rely on it */}
                <div
                  className={`absolute left-0 top-0 bottom-0 w-1 transition-opacity ${selected ? "opacity-100" : "opacity-10 group-hover:opacity-30"}`}
                  style={{
                    backgroundColor: track.color || "var(--qt-highlight)",
                  }}
                />

                {isOver && dragUi?.mode === "nest" ? (
                  <div
                    className="absolute inset-0 pointer-events-none"
                    style={{
                      backgroundColor:
                        "color-mix(in oklab, var(--qt-highlight) 14%, transparent)",
                      border: "1px dashed var(--qt-highlight)",
                    }}
                  />
                ) : null}

                <Flex
                  direction="column"
                  p="2"
                  gap="2"
                  height="100%"
                  justify="center"
                >
                  <Flex justify="between" align="center">
                    <Flex align="center" gap="1" className="min-w-0 flex-1">
                      {/* 轨道颜色小圆点，点击弹出调色板 */}
                      <div
                        className="relative shrink-0"
                        data-track-color-picker
                      >
                        <button
                          className="w-3.5 h-3.5 rounded-full border border-white/20 hover:scale-125 transition-transform cursor-pointer"
                          style={{
                            backgroundColor: track.color || "#4f8ef7",
                          }}
                          title={t("track_change_color")}
                          onPointerDown={(e) => e.stopPropagation()}
                          onClick={(e) => {
                            e.stopPropagation();
                            setColorPickerTrackId(
                              colorPickerTrackId === track.id ? null : track.id,
                            );
                          }}
                        />
                        {colorPickerTrackId === track.id && (
                          <div
                            className="absolute left-0 top-full mt-1 z-50 p-1.5 rounded border border-qt-border bg-qt-window shadow-lg flex gap-1 flex-wrap"
                            style={{ width: 120 }}
                            data-track-color-picker
                          >
                            {TRACK_COLOR_PALETTE_KEYS.map((opt) => (
                              <button
                                key={opt.value}
                                title={t(opt.key)}
                                className={`w-4 h-4 rounded-full transition-transform hover:scale-125 ${
                                  (track.color || "#4f8ef7") === opt.value
                                    ? "ring-2 ring-white/80 scale-110"
                                    : ""
                                }`}
                                style={{
                                  backgroundColor: opt.value,
                                }}
                                onPointerDown={(e) => e.stopPropagation()}
                                onClick={(e) => {
                                  e.stopPropagation();
                                  onTrackColorChange?.(track.id, opt.value);
                                  setColorPickerTrackId(null);
                                }}
                              />
                            ))}
                          </div>
                        )}
                      </div>
                      {editingTrackId === track.id ? (
                        <input
                          ref={nameInputRef}
                          value={editingName}
                          className="bg-transparent outline outline-1 outline-qt-highlight rounded px-0.5 flex-1 min-w-0 text-qt-text text-sm font-medium pr-2"
                          onChange={(e) => setEditingName(e.target.value)}
                          onBlur={commitTrackName}
                          onKeyDown={(e) => {
                            if (e.key === "Enter") {
                              commitTrackName();
                            } else if (e.key === "Escape") {
                              setEditingTrackId(null);
                            }
                          }}
                          onPointerDown={(e) => e.stopPropagation()}
                          onClick={(e) => e.stopPropagation()}
                          autoFocus
                        />
                      ) : (
                        <Text
                          size="2"
                          weight="medium"
                          className={`text-qt-text truncate pr-2 ${depth > 0 ? "opacity-90" : ""} cursor-text select-none`}
                          onPointerDown={(e) => e.stopPropagation()}
                          onDoubleClick={(e) => {
                            e.stopPropagation();
                            setEditingTrackId(track.id);
                            setEditingName(track.name);
                            setTimeout(() => {
                              nameInputRef.current?.select();
                            }, 0);
                          }}
                        >
                          {track.name}
                        </Text>
                      )}
                    </Flex>
                    <IconButton
                      size="1"
                      variant="ghost"
                      color="gray"
                      className="opacity-0 group-hover:opacity-100"
                      disabled={tracks.length <= 1}
                      onPointerDown={(e) => e.stopPropagation()}
                      onClick={(e) => {
                        e.stopPropagation();
                        onRemoveTrack(track.id);
                      }}
                    >
                      <Cross2Icon />
                    </IconButton>
                  </Flex>

                  <Flex gap="2" align="center">
                    {isRoot ? (
                      <IconButton
                        size="1"
                        variant={composeEnabled ? "solid" : "ghost"}
                        color={composeEnabled ? "blue" : "gray"}
                        title={t("compose")}
                        onPointerDown={(e) => e.stopPropagation()}
                        onClick={(e) => {
                          e.stopPropagation();
                          onToggleCompose(track.id, !composeEnabled);
                        }}
                        style={{
                          fontWeight: 700,
                          fontSize: 11,
                          width: 20,
                          height: 20,
                        }}
                      >
                        C
                      </IconButton>
                    ) : null}
                    <IconButton
                      size="1"
                      variant={muted ? "solid" : "ghost"}
                      color={muted ? "red" : "gray"}
                      title={muted ? t("clip_unmute") : t("clip_mute")}
                      onPointerDown={(e) => e.stopPropagation()}
                      onClick={(e) => {
                        e.stopPropagation();
                        onToggleMute(track.id, !muted);
                      }}
                      style={{
                        fontWeight: 700,
                        fontSize: 11,
                        width: 20,
                        height: 20,
                      }}
                    >
                      M
                    </IconButton>
                    <IconButton
                      size="1"
                      variant={solo ? "solid" : "ghost"}
                      color={solo ? "amber" : "gray"}
                      title={t("solo")}
                      onPointerDown={(e) => e.stopPropagation()}
                      onClick={(e) => {
                        e.stopPropagation();
                        onToggleSolo(track.id, !solo);
                      }}
                      style={{
                        fontWeight: 700,
                        fontSize: 11,
                        width: 20,
                        height: 20,
                      }}
                    >
                      S
                    </IconButton>
                    {isRoot && composeEnabled && onAlgoChange ? (
                      <div onPointerDown={(e) => e.stopPropagation()}>
                        <Select.Root
                          size="1"
                          value={
                            [
                              "world_dll",
                              "nsf_hifigan_onnx",
                              "vslib",
                              "none",
                            ].includes(track.pitchAnalysisAlgo)
                              ? track.pitchAnalysisAlgo
                              : "nsf_hifigan_onnx"
                          }
                          onValueChange={(v) => {
                            onAlgoChange(track.id, v);
                          }}
                        >
                          <Select.Trigger style={{ minWidth: 80 }} />
                          <Select.Content>
                            <Select.Item value="world_dll">world</Select.Item>
                            <Select.Item value="nsf_hifigan_onnx">
                              nsf-hifigan
                            </Select.Item>
                            <Select.Item value="vslib">vslib</Select.Item>
                            <Select.Item value="none">{t("none")}</Select.Item>
                          </Select.Content>
                        </Select.Root>
                      </div>
                    ) : null}
                    <Box flexGrow="1" />
                    <Text size="1" color="gray">
                      {Math.round(volume * 100)}%
                    </Text>
                  </Flex>

                  <Slider
                    value={[Math.round(volume * 100)]}
                    size="1"
                    className="w-full"
                    onValueChange={(v) => {
                      const next =
                        Math.max(0, Math.min(100, Number(v[0] ?? 0))) / 100;
                      onVolumeUiChange(track.id, next);
                    }}
                    onValueCommit={(v) => {
                      const next =
                        Math.max(0, Math.min(100, Number(v[0] ?? 0))) / 100;
                      onVolumeCommit(track.id, next);
                    }}
                    onPointerDown={(e) => e.stopPropagation()}
                  />
                </Flex>
              </Box>
            </div>
          );
        })}

        <Flex
          align="center"
          justify="center"
          className="h-8 border-b border-qt-border border-dashed text-qt-text-muted hover:text-qt-text hover:bg-qt-button-hover cursor-pointer transition-colors"
          style={{ height: TRACK_ADD_ROW_HEIGHT }}
          onClick={onAddTrack}
        >
          <PlusIcon className="mr-1" /> <Text size="1">{t("track_add")}</Text>
        </Flex>
      </div>

      {/* 轨道右键菜单 */}
      {trackCtxMenu && (
        <div
          ref={trackCtxMenuRef}
          data-track-ctx-menu
          data-hs-context-menu="1"
          className="fixed z-50 min-w-[140px] rounded border border-qt-border bg-qt-window text-qt-text shadow-lg py-1"
          style={{ left: trackCtxMenu.x, top: trackCtxMenu.y }}
          onPointerDown={(e) => e.stopPropagation()}
        >
          <button
            className="w-full text-left px-3 py-1.5 text-sm hover:bg-qt-button-hover transition-colors"
            onClick={() => {
              onDuplicateTrack?.(trackCtxMenu.trackId);
              setTrackCtxMenu(null);
            }}
          >
            {t("track_clone")}
          </button>
          <button
            className="w-full text-left px-3 py-1.5 text-sm hover:bg-qt-button-hover transition-colors text-red-400 hover:text-red-300"
            disabled={tracks.length <= 1}
            onClick={() => {
              onRemoveTrack(trackCtxMenu.trackId);
              setTrackCtxMenu(null);
            }}
          >
            {t("ctx_delete")}
          </button>
        </div>
      )}
    </Flex>
  );
};
