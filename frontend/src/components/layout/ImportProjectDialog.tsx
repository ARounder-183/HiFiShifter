import { useState } from "react";
import { Text } from "@radix-ui/themes";
import { useI18n } from "../../i18n/I18nProvider";
import { AppDialog } from "../../ui/Dialog";
import { AppForm, AppSwitchRow } from "../../ui/Field";

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
    const { tf } = useI18n();
    const [placeAtPlayhead, setPlaceAtPlayhead] = useState(false);
    const [importTempoMap, setImportTempoMap] = useState(true);

    return (
        <AppDialog
            open={open}
            onOpenChange={onOpenChange}
            title={tf("import_project_dialog_title")}
            description={tf("import_project_dialog_desc")}
            size="lg"
            actions={[
                { id: "cancel", label: tf("cancel"), onClick: () => onOpenChange(false) },
                {
                    id: "import",
                    label: tf("import_project_import"),
                    intent: "primary",
                    onClick: () =>
                        onConfirm({
                            placeAtPlayhead,
                            importTempoMap: importTempoMap && !hasExistingTempoMap,
                        }),
                },
            ]}
        >
            <AppForm>
                <Text size="2" className="text-qt-text-muted break-all">
                    {tf("import_project_file")}: {projectPath ?? ""}
                </Text>

                <label className="flex items-center gap-2 text-qt-md text-qt-text">
                    <input
                        type="radio"
                        name="hifishifter-import-position"
                        checked={!placeAtPlayhead}
                        onChange={() => setPlaceAtPlayhead(false)}
                    />
                    {tf("import_project_original_position")}
                </label>
                <label className="flex items-center gap-2 text-qt-md text-qt-text">
                    <input
                        type="radio"
                        name="hifishifter-import-position"
                        checked={placeAtPlayhead}
                        onChange={() => setPlaceAtPlayhead(true)}
                    />
                    {tf("import_project_playhead_position")}
                </label>

                <AppSwitchRow
                    control="checkbox"
                    label={tf("import_project_tempo_map")}
                    checked={importTempoMap}
                    disabled={hasExistingTempoMap}
                    onCheckedChange={setImportTempoMap}
                />
                {hasExistingTempoMap ? (
                    <Text size="1" className="text-qt-text-muted">
                        {tf("import_project_tempo_map_unavailable")}
                    </Text>
                ) : null}
            </AppForm>
        </AppDialog>
    );
}
