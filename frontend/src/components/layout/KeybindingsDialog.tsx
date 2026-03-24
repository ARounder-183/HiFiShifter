import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  Dialog,
  Flex,
  Text,
  Button,
  ScrollArea,
  Separator,
} from "@radix-ui/themes";
import { ResetIcon, Cross2Icon } from "@radix-ui/react-icons";
import { useI18n } from "../../i18n/I18nProvider";
import { useAppDispatch, useAppSelector } from "../../app/hooks";
import {
  selectMergedKeybindings,
  setKeybinding,
  resetKeybinding,
  resetAllKeybindings,
  formatKeybinding,
  findConflicts,
  // isNoneBinding, // 已删除未使用变量
} from "../../features/keybindings/keybindingsSlice";
import {
  DEFAULT_KEYBINDINGS,
  ACTION_META,
  ALL_ACTION_IDS,
  GROUP_LABEL_KEYS,
} from "../../features/keybindings/defaultKeybindings";
import type { ActionId, Keybinding } from "../../features/keybindings/types";

/** "无" 绑定常量 */
const NONE_BINDING: Keybinding = { key: "__none__" };

interface KeybindingsDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

/**
 * 快捷键设置面板
 * 打开时阻塞所有下层交互（通过 Dialog overlay）
 */
export const KeybindingsDialog: React.FC<KeybindingsDialogProps> = ({
  open,
  onOpenChange,
}) => {
  const dispatch = useAppDispatch();
  const { t } = useI18n();
  const keybindings = useAppSelector(selectMergedKeybindings);
  const overrides = useAppSelector((s) => s.keybindings.overrides);

  // 打开时在 body 上标记，阻塞全局快捷键和工程编辑
  useEffect(() => {
    if (open) {
      document.body.setAttribute("data-keybindings-dialog-open", "true");
    } else {
      document.body.removeAttribute("data-keybindings-dialog-open");
    }
    return () => {
      document.body.removeAttribute("data-keybindings-dialog-open");
    };
  }, [open]);

  // 当前处于"录入模式"的 actionId
  const [recordingId, setRecordingId] = useState<ActionId | null>(null);
  // 冲突提示
  const [conflict, setConflict] = useState<{
    actionId: ActionId;
    newBinding: Keybinding;
    conflictWith: ActionId[];
  } | null>(null);

  const recordingRef = useRef(recordingId);
  useEffect(() => {
    recordingRef.current = recordingId;
  }, [recordingId]);

  // 录入模式的键盘监听
  useEffect(() => {
    if (!recordingId) return;

    const currentIsModifierOnly = Boolean(
      DEFAULT_KEYBINDINGS[recordingId]?.modifierOnly,
    );

    function onKeyDown(e: KeyboardEvent) {
      e.preventDefault();
      e.stopPropagation();

      // Escape 取消录入
      if (e.key === "Escape") {
        setRecordingId(null);
        setConflict(null);
        return;
      }

      const modKeys = ["Control", "Shift", "Alt", "Meta"];

      if (currentIsModifierOnly) {
        // modifierOnly 模式：只接受修饰键
        if (!modKeys.includes(e.key)) return;
        const key = e.key.toLowerCase();
        const newBinding: Keybinding = {
          key,
          modifierOnly: true,
          ...(e.key === "Control" ? { ctrl: true } : {}),
          ...(e.key === "Shift" ? { shift: true } : {}),
          ...(e.key === "Alt" ? { alt: true } : {}),
        };
        const currentId = recordingRef.current;
        if (!currentId) return;
        const conflicts = findConflicts(overrides, currentId, newBinding);
        if (conflicts.length > 0) {
          setConflict({
            actionId: currentId,
            newBinding,
            conflictWith: conflicts,
          });
          return;
        }
        dispatch(setKeybinding({ actionId: currentId, binding: newBinding }));
        setRecordingId(null);
        return;
      }

      // 普通模式：忽略单独按下修饰键
      if (modKeys.includes(e.key)) return;

      const key = e.key === " " ? "space" : e.key.toLowerCase();
      const isMac = navigator.platform?.toLowerCase().includes("mac");
      const ctrl = isMac ? e.metaKey : e.ctrlKey;

      const newBinding: Keybinding = {
        key,
        ...(ctrl ? { ctrl: true } : {}),
        ...(e.shiftKey ? { shift: true } : {}),
        ...(e.altKey ? { alt: true } : {}),
      };

      const currentId = recordingRef.current;
      if (!currentId) return;

      // 检测冲突
      const conflicts = findConflicts(overrides, currentId, newBinding);
      if (conflicts.length > 0) {
        setConflict({
          actionId: currentId,
          newBinding,
          conflictWith: conflicts,
        });
        return;
      }

      dispatch(setKeybinding({ actionId: currentId, binding: newBinding }));
      setRecordingId(null);
    }

    window.addEventListener("keydown", onKeyDown, true);
    return () => window.removeEventListener("keydown", onKeyDown, true);
  }, [recordingId, dispatch, overrides]);

  const handleConfirmConflict = useCallback(() => {
    if (!conflict) return;
    // 先清除冲突的绑定
    for (const cId of conflict.conflictWith) {
      dispatch(resetKeybinding(cId));
    }
    // 再设置新绑定
    dispatch(
      setKeybinding({
        actionId: conflict.actionId,
        binding: conflict.newBinding,
      }),
    );
    setConflict(null);
    setRecordingId(null);
  }, [conflict, dispatch]);

  const handleCancelConflict = useCallback(() => {
    setConflict(null);
  }, []);

  const handleResetAll = useCallback(() => {
    dispatch(resetAllKeybindings());
  }, [dispatch]);

  // 按分组组织操作
  const groups = React.useMemo(() => {
    const groupOrder: Array<
      | "playback"
      | "edit"
      | "project"
      | "clip"
      | "pianoRoll"
      | "mode"
      | "modifier"
      | "quickSearch"
    > = [
      "playback",
      "edit",
      "project",
      "clip",
      "pianoRoll",
      "mode",
      "modifier",
      "quickSearch",
    ];
    return groupOrder.map((group) => ({
      group,
      actions: ALL_ACTION_IDS.filter((id) => ACTION_META[id].group === group),
    }));
  }, []);

  const tAny = t as (key: string) => string;

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      {/* Overlay 阻塞下层所有交互 */}
      {open && (
        <div
          style={{
            position: "fixed",
            inset: 0,
            zIndex: 9998,
            background: "rgba(0,0,0,0.4)",
          }}
          onPointerDown={(e) => e.stopPropagation()}
          onKeyDown={(e) => {
            if (!recordingId) return;
            e.stopPropagation();
          }}
        />
      )}
      <Dialog.Content
        style={{
          maxWidth: 560,
          maxHeight: "80vh",
          zIndex: 9999,
          overflowY: "hidden",
        }}
        onPointerDownOutside={(e) => {
          // 如果正在录入，阻止点击外部关闭
          if (recordingId) e.preventDefault();
        }}
        onKeyDown={(e) => {
          // 阻止按键穿透到工程
          e.stopPropagation();
        }}
      >
        <Dialog.Title>{tAny("kb_dialog_title")}</Dialog.Title>
        <Dialog.Description size="2" color="gray">
          {tAny("kb_dialog_desc")}
        </Dialog.Description>
        <Text size="1" color="gray" style={{ marginTop: 4, display: "block" }}>
          {tAny("kb_dialog_hint_click")}
        </Text>

        <ScrollArea
          style={{ maxHeight: "calc(80vh - 160px)" }}
          scrollbars="vertical"
        >
          <Flex direction="column" gap="3" py="3">
            {groups.map(({ group, actions }) => (
              <Flex direction="column" gap="1" key={group}>
                <Text
                  size="1"
                  weight="bold"
                  color="gray"
                  style={{
                    textTransform: "uppercase",
                    letterSpacing: "0.05em",
                    padding: "4px 0",
                  }}
                >
                  {tAny(GROUP_LABEL_KEYS[group])}
                </Text>
                <Separator size="4" />
                {actions.map((actionId) => {
                  const meta = ACTION_META[actionId];
                  const currentKb = keybindings[actionId];
                  const defaultKb = DEFAULT_KEYBINDINGS[actionId];
                  const isDefault =
                    currentKb.key === defaultKb.key &&
                    Boolean(currentKb.ctrl) === Boolean(defaultKb.ctrl) &&
                    Boolean(currentKb.shift) === Boolean(defaultKb.shift) &&
                    Boolean(currentKb.alt) === Boolean(defaultKb.alt) &&
                    Boolean(currentKb.modifierOnly) ===
                      Boolean(defaultKb.modifierOnly);
                  const isRecording = recordingId === actionId;

                  return (
                    <Flex
                      key={actionId}
                      align="center"
                      justify="between"
                      px="2"
                      py="1"
                      style={{
                        borderRadius: 4,
                        background: isRecording ? "var(--accent-3)" : undefined,
                        minHeight: 36,
                      }}
                    >
                      <Text size="2">{tAny(meta.labelKey)}</Text>
                      <Flex align="center" gap="2">
                        {/* 快捷键显示 / 录入按钮 */}
                        <Button
                          variant={isRecording ? "solid" : "soft"}
                          color={
                            isRecording ? "blue" : !isDefault ? "green" : "gray"
                          }
                          size="1"
                          style={{
                            minWidth: 120,
                            fontFamily: "monospace",
                          }}
                          onClick={() => {
                            if (isRecording) {
                              // 录入中左键点击 → 设为"无"
                              dispatch(
                                setKeybinding({
                                  actionId,
                                  binding: NONE_BINDING,
                                }),
                              );
                              setRecordingId(null);
                              setConflict(null);
                            } else {
                              setConflict(null);
                              setRecordingId(actionId);
                            }
                          }}
                          onContextMenu={(e) => {
                            e.preventDefault();
                            // 右键点击 → 直接重置为默认
                            dispatch(resetKeybinding(actionId));
                            setRecordingId(null);
                            setConflict(null);
                          }}
                        >
                          {isRecording
                            ? tAny(
                                defaultKb.modifierOnly
                                  ? "kb_press_modifier"
                                  : "kb_press_key",
                              )
                            : formatKeybinding(currentKb, tAny("kb_none"))}
                        </Button>
                      </Flex>
                    </Flex>
                  );
                })}
              </Flex>
            ))}
          </Flex>
        </ScrollArea>

        {/* 冲突提示 */}
        {conflict && (
          <Flex
            align="center"
            gap="2"
            py="2"
            px="3"
            style={{
              background: "var(--red-3)",
              borderRadius: 6,
              marginTop: 8,
            }}
          >
            <Text size="2" color="red" style={{ flex: 1 }}>
              {tAny("kb_conflict_msg")}{" "}
              <strong>
                {conflict.conflictWith
                  .map((id) => tAny(ACTION_META[id].labelKey))
                  .join(", ")}
              </strong>
            </Text>
            <Button
              size="1"
              color="red"
              variant="soft"
              onClick={handleConfirmConflict}
            >
              {tAny("kb_conflict_override")}
            </Button>
            <Button
              size="1"
              color="gray"
              variant="soft"
              onClick={handleCancelConflict}
            >
              {tAny("kb_conflict_cancel")}
            </Button>
          </Flex>
        )}

        {/* 底部按钮 */}
        <Flex justify="between" align="center" pt="3">
          <Button variant="soft" color="gray" size="2" onClick={handleResetAll}>
            <ResetIcon />
            {tAny("kb_reset_all")}
          </Button>
          <Dialog.Close>
            <Button variant="soft" color="gray" size="2">
              <Cross2Icon />
              {tAny("kb_close")}
            </Button>
          </Dialog.Close>
        </Flex>
      </Dialog.Content>
    </Dialog.Root>
  );
};
