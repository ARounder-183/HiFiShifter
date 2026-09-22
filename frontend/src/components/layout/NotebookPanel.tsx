import { Box, Button, Flex, Text } from "@radix-ui/themes";
import { useMemo } from "react";
import { useAppDispatch, useAppSelector } from "../../app/hooks";
import { closeNotebook, setNotebookMode } from "../../features/notebook/notebookSlice";
import { setProjectNotesMarkdown } from "../../features/session/sessionSlice";
import { useI18n } from "../../i18n/I18nProvider";
import { webApi } from "../../services/webviewApi";
import { renderMarkdownPreview } from "./notebook/markdownPreview";

export function NotebookPanel() {
    const dispatch = useAppDispatch();
    const { t } = useI18n();
    const mode = useAppSelector((state) => state.notebook.mode);
    const markdown = useAppSelector((state) => state.session.project.notesMarkdown);

    const previewHtml = useMemo(() => renderMarkdownPreview(markdown), [markdown]);

    /**
     * 记事本编辑的撤销合并由**后端按历史结构**完成：历史最前沿一步是
     * 「编辑记事本」时，后续写入全部并入该步 —— 无论用户输入多久、停顿多长、
     * 失焦多少次；其它操作介入后才自然另起一步。前端因此不需要任何开窗/
     * 收尾时序（此前基于停手超时的合并把长编辑会话切成大量记录，已移除）。
     *
     * 输入框内的 Ctrl+Z 由 textarea 自己处理（全局快捷键监听已按可编辑目标
     * 排除 textarea，见 `useKeybindings.isEditableTarget`），这里不做任何拦截。
     */
    const handleChange = (next: string) => {
        dispatch(setProjectNotesMarkdown(next));
        void webApi.setProjectNotes(next);
    };

    return (
        <Flex className="h-full min-h-0 flex-col bg-qt-window">
            <Flex
                align="center"
                justify="between"
                className="shrink-0 border-b border-qt-border px-2 py-1.5"
            >
                <Text size="2" weight="medium">
                    {t("notebook")}
                </Text>
                <Flex align="center" gap="1">
                    <Button
                        size="1"
                        variant={mode === "edit" ? "solid" : "soft"}
                        color={mode === "edit" ? "blue" : "gray"}
                        onClick={() => dispatch(setNotebookMode("edit"))}
                    >
                        {t("notebook_edit")}
                    </Button>
                    <Button
                        size="1"
                        variant={mode === "preview" ? "solid" : "soft"}
                        color={mode === "preview" ? "blue" : "gray"}
                        onClick={() => dispatch(setNotebookMode("preview"))}
                    >
                        {t("notebook_preview")}
                    </Button>
                    <Button
                        size="1"
                        variant="ghost"
                        color="gray"
                        onClick={() => dispatch(closeNotebook())}
                    >
                        {t("close")}
                    </Button>
                </Flex>
            </Flex>

            <Box className="min-h-0 flex-1">
                {mode === "edit" ? (
                    <textarea
                        value={markdown}
                        onChange={(event) => handleChange(event.target.value)}
                        placeholder={t("notebook_placeholder")}
                        className="h-full w-full resize-none border-0 bg-qt-base px-3 py-3 text-sm text-qt-text outline-none"
                        spellCheck={false}
                    />
                ) : (
                    <div className="h-full overflow-auto bg-qt-base px-4 py-3 text-sm text-qt-text">
                        <div
                            className="prose prose-sm max-w-none prose-headings:text-qt-text prose-p:text-qt-text prose-li:text-qt-text prose-strong:text-qt-text prose-code:text-qt-text prose-pre:bg-black/20"
                            dangerouslySetInnerHTML={{ __html: previewHtml }}
                        />
                    </div>
                )}
            </Box>
        </Flex>
    );
}
