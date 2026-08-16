import { useState } from "react";
import { Button, Dialog, Flex, Text } from "@radix-ui/themes";
import { useI18n } from "../../i18n/I18nProvider";

export interface ImportProjectOptions {
    placeAtPlayhead: boolean;
    importTempoMap: boolean;
}

export function ImportProjectDialog({
    open,
    projectPath,
    hasExistingTempoMap,
    onOpenChange,
    onConfirm,
}: {
    open: boolean;
    projectPath: string | null;
    /** When the current project already has a tempo map the checkbox is disabled. */
    hasExistingTempoMap: boolean;
    onOpenChange: (open: boolean) => void;
    onConfirm: (options: ImportProjectOptions) => void;
}) {
    const { t } = useI18n();
    const tAny = t as (key: string) => string;
    const [placeAtPlayhead, setPlaceAtPlayhead] = useState(false);
    const [importTempoMap, setImportTempoMap] = useState(true);

    return (
        <Dialog.Root open={open} onOpenChange={onOpenChange}>
            <Dialog.Content
                style={{ maxWidth: 620 }}
                onKeyDown={(event) => event.stopPropagation()}
            >
                <Dialog.Title>{tAny("import_project_dialog_title")}</Dialog.Title>
                <Dialog.Description>{tAny("import_project_dialog_desc")}</Dialog.Description>

                <Flex direction="column" gap="3" mt="3">
                    <Text size="2" className="text-qt-text-muted break-all">
                        {tAny("import_project_file")}: {projectPath ?? ""}
                    </Text>

                    <label className="flex items-center gap-2 text-sm text-qt-text">
                        <input
                            type="radio"
                            name="hifishifter-import-position"
                            checked={!placeAtPlayhead}
                            onChange={() => setPlaceAtPlayhead(false)}
                        />
                        {tAny("import_project_original_position")}
                    </label>
                    <label className="flex items-center gap-2 text-sm text-qt-text">
                        <input
                            type="radio"
                            name="hifishifter-import-position"
                            checked={placeAtPlayhead}
                            onChange={() => setPlaceAtPlayhead(true)}
                        />
                        {tAny("import_project_playhead_position")}
                    </label>

                    <label className="flex items-center gap-2 text-sm text-qt-text">
                        <input
                            type="checkbox"
                            checked={importTempoMap}
                            disabled={hasExistingTempoMap}
                            onChange={(event) => setImportTempoMap(event.target.checked)}
                        />
                        {tAny("import_project_tempo_map")}
                    </label>
                    {hasExistingTempoMap ? (
                        <Text size="1" className="text-qt-text-muted">
                            {tAny("import_project_tempo_map_unavailable")}
                        </Text>
                    ) : null}
                </Flex>

                <Flex gap="3" mt="4" justify="end">
                    <Button variant="soft" color="gray" onClick={() => onOpenChange(false)}>
                        {tAny("cancel")}
                    </Button>
                    <Button
                        onClick={() =>
                            onConfirm({
                                placeAtPlayhead,
                                importTempoMap: importTempoMap && !hasExistingTempoMap,
                            })
                        }
                    >
                        {tAny("import_project_import")}
                    </Button>
                </Flex>
            </Dialog.Content>
        </Dialog.Root>
    );
}
