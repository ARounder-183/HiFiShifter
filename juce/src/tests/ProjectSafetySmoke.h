#pragma once
#include "../ProjectFileIO.h"
#include "../ProjectRecovery.h"
namespace hachi
{
inline bool MainComponent::diagnosticProjectSafety(const juce::File& folder)
{
    folder.createDirectory(); stopTimer(); bool ok=true;
    const auto check=[&](const char* name,bool pass){ok=ok&&pass;std::cout<<name<<'='<<pass<<std::endl;};
    const auto bytes=[](const juce::File& f){juce::MemoryBlock b;f.loadFileAsData(b);return b;};
    const auto file=folder.getChildFile(juce::String::fromUTF8("工程.hjpx"));
    const auto backup=projectio::backupFile(file);
    const auto media=folder.getChildFile("source.wav");media.replaceWithText("fixture");
    const auto make=[&](const char* name)
    {
        ProjectData d;d.name=name;TrackData t;t.id="voice";t.name=name;t.compose=false;
        ClipData c;c.id="clip";c.durationSeconds=c.sourceDurationSeconds=2;c.sourceFile=media;
        c.showNoteHints=true;c.showNormalDisplay=true;c.gainEnvelope={{0,0},{2,-6}};
        ClipData part=c;part.id="part";part.notes.clear();c.parts.push_back(part);
        NoteData n;n.id="note";n.label="bang";n.durationSeconds=1;n.midiNote=62;
        n.utauFlags="g-20B15";n.utauFlagCurveEnabled=true;n.utauFlagCurves={{"g",{{0,-10},{1,20}}}};
        n.diffSingerPitchReference={{0,61},{.5,62.5f},{1,62}};
        n.pitchControlPoints={{0,62},{1,63}};n.diffSingerTiming="{\"b\":-0.08}";
        c.notes.push_back(n);t.clips.push_back(c);d.tracks.push_back(t);return d;
    };
    auto a=make("DOCUMENT-A"),b=make("DOCUMENT-B"),c=make("DOCUMENT-C");
    juce::String error;ProjectModel model;
    model.resetDocument(a);model.replace(b);
    check("ordinary_edit_remains_undoable",model.undo()&&model.snapshot().name==a.name);
    check("ordinary_edit_redo",model.redo()&&model.snapshot().name==b.name);
    model.undo();model.resetDocument(b);
    check("new_document_clears_undo_and_redo",!model.canUndo()&&!model.canRedo()&&!model.undo());
    model.replace(c);check("new_document_edits_undo_only_to_itself",model.undo()&&model.snapshot().name==b.name&&!model.undo());
    model.clear();check("new_blank_cannot_restore_previous_document",model.snapshot().tracks.empty()&&!model.canUndo()&&!model.canRedo());
    model.resetDocument(a);check("first_save_unicode_path",model.save(file,error)&&error.isEmpty());
    const auto aBytes=bytes(file);check("first_save_needs_no_backup",!backup.exists());
    model.replace(b);check("second_save_succeeds",model.save(file,error));const auto bBytes=bytes(file);
    check("backup_is_exact_previous_version",bytes(backup)==aBytes);
    ProjectModel loaded;check("new_project_load_succeeds",loaded.load(file,error)&&error.isEmpty());
    check("save_load_flags_refs_envelopes_and_parts",loaded.contentFingerprint()==model.contentFingerprint());
    loaded.replace(c);loaded.undo();check("history_exists_before_reload",loaded.canRedo());
    check("successful_load_discards_old_history",loaded.load(file,error)&&!loaded.canUndo()&&!loaded.canRedo());
    loaded.replace(c);const auto beforeBad=loaded.contentFingerprint();const auto revision=loaded.revisionNumber();
    const auto bad=folder.getChildFile("truncated.hjpx");bad.replaceWithData(bBytes.getData(),bBytes.getSize()/2);
    check("truncated_project_is_rejected",!loaded.load(bad,error));
    check("failed_load_keeps_content_revision_and_history",loaded.contentFingerprint()==beforeBad&&loaded.revisionNumber()==revision&&loaded.canUndo());
    check("undo_still_works_after_failed_load",loaded.undo()&&loaded.snapshot().name==b.name);
    check("backup_itself_loads",loaded.load(backup,error)&&loaded.snapshot().name==a.name);
    model.replace(c);check("third_save_rotates_backup",model.save(file,error)&&bytes(backup)==bBytes);
    const auto cBytes=bytes(file);
    file.replaceWithData(cBytes.getData(),cBytes.getSize()/2);const auto corrupt=bytes(file);
    check("refuse_overwriting_corrupt_project",!model.save(file,error)&&error.isNotEmpty());
    check("corrupt_original_and_last_good_backup_preserved",bytes(file)==corrupt&&bytes(backup)==bBytes);
    file.replaceWithData(cBytes.getData(),cBytes.getSize());
    const auto blocker=folder.getChildFile("not-a-directory");blocker.replaceWithText("keep");
    check("write_failure_is_reported",!model.save(blocker.getChildFile("project.hjpx"),error)&&error.isNotEmpty());
    check("write_failure_keeps_unrelated_file",blocker.loadFileAsString()=="keep");
    const auto failDir=folder.getChildFile("backup-failure");failDir.createDirectory();
    const auto failFile=failDir.getChildFile("project.hjpx");model.resetDocument(a);model.save(failFile,error);
    projectio::backupFile(failFile).createDirectory();const auto failOriginal=bytes(failFile);
    const auto fileCount=failDir.findChildFiles(juce::File::findFiles,false).size();model.replace(b);
    check("backup_failure_aborts_main_save",!model.save(failFile,error));
    check("backup_failure_preserves_original_and_cleans_temps",bytes(failFile)==failOriginal&&failDir.findChildFiles(juce::File::findFiles,false).size()==fileCount);
    const auto readonly=folder.getChildFile("readonly.hjpx");model.resetDocument(a);model.save(readonly,error);
    const auto roBytes=bytes(readonly);const auto roSet=readonly.setReadOnly(true);model.replace(b);
    const auto roSaved=model.save(readonly,error);readonly.setReadOnly(false);
    check("replacement_failure_reported",roSet&&!roSaved&&error.isNotEmpty());
    check("replacement_failure_preserves_original_and_backup",bytes(readonly)==roBytes&&bytes(projectio::backupFile(readonly))==roBytes);
    const auto recRoot=folder.getChildFile("Recovery");juce::File abandoned;
    {
        ProjectRecovery recovery(recRoot);model.resetDocument(b);abandoned=recovery.currentFile();
        check("recovery_capture_starts",recovery.capture(model,file));recovery.finish();
        const auto tree=projectio::validatedTree(bytes(abandoned));
        check("recovery_contains_timestamp_and_original_path",tree["recoverySavedAt"].toString().isNotEmpty()&&tree["recoveryOriginalFile"].toString()==file.getFullPathName());
        check("autosave_never_overwrites_original",bytes(file)==cBytes);
        check("autosave_keeps_all_project_data",loaded.load(abandoned,error)&&loaded.contentFingerprint()==model.contentFingerprint());
        const auto rd=loaded.snapshot();check("recovery_media_and_merged_part_paths_resolve",rd.tracks[0].clips[0].sourceFile==media&&rd.tracks[0].clips[0].parts[0].sourceFile==media);
        check("own_active_recovery_not_offered",recovery.entries().empty());
        const auto blocked=std::async(std::launch::async,[&]{ProjectRecovery other(recRoot);return other.entries().empty();}).get();
        check("another_active_session_is_not_offered",blocked);
    }
    check("unapproved_shutdown_keeps_recovery",abandoned.existsAsFile());
    ProjectRecovery scanner(recRoot);const auto entries=scanner.entries();
    check("next_session_finds_interrupted_project",entries.size()==1&&entries[0].file==abandoned&&entries[0].original==file);
    {
        ProjectRecovery timer(folder.getChildFile("TimerRecovery"));auto now=juce::Time::getMillisecondCounterHiRes()+31000;
        timer.tick(model,{},false,now);check("clean_project_does_not_autosave",!timer.busy()&&!timer.currentFile().exists());
        timer.tick(model,{},true,now+31000);timer.finish();check("dirty_unnamed_project_autosaves",timer.currentFile().existsAsFile());
        const auto saved=bytes(timer.currentFile());timer.tick(model,{},true,now+62000);
        check("unchanged_revision_not_rewritten",!timer.busy()&&bytes(timer.currentFile())==saved);
        model.replace(c);timer.tick(model,{},true,now+93000);timer.finish();
        check("new_revision_captured",loaded.load(timer.currentFile(),error)&&loaded.snapshot().name==c.name);
        model.replace(a);timer.capture(model,{});timer.discard();
        check("discard_waits_for_pending_writer_and_removes_snapshot",!timer.busy()&&!timer.currentFile().exists());
    }
    const auto unrelated=recRoot.getChildFile("other.hjpx");ProjectModel::saveRecoverySnapshot(a,unrelated,{},error);
    projectRecovery=std::make_unique<ProjectRecovery>(recRoot);project.resetDocument(a);currentProjectFile=file;savedProjectRevision=project.revisionNumber();
    check("gui_recovery_restore_succeeds",!entries.empty()&&restoreProjectRecovery(entries[0]));
    check("gui_recovery_is_unnamed_dirty_new_document",currentProjectFile==juce::File{}&&savedProjectRevision!=project.revisionNumber()&&!project.canUndo()&&project.snapshot().name==b.name);
    check("recovery_source_kept_until_explicit_save",abandoned.existsAsFile());
    project.replace(c);projectRecovery->capture(project,{});projectRecovery->finish();const auto own=projectRecovery->currentFile();
    check("restored_changes_also_get_recovery",own.existsAsFile());
    const auto failedSave=saveProjectTo(blocker.getChildFile("gui.hjpx"));
    check("gui_failed_save_preserves_dirty_path_and_recoveries",!failedSave&&currentProjectFile==juce::File{}&&savedProjectRevision!=project.revisionNumber()&&own.existsAsFile()&&abandoned.existsAsFile());
    if(auto* modal=juce::Component::getCurrentlyModalComponent())modal->exitModalState(0);
    const auto recovered=folder.getChildFile("recovered.hjpx");check("gui_successful_save",saveProjectTo(recovered));
    check("gui_save_clears_only_its_own_recovery",!own.exists()&&!abandoned.exists()&&unrelated.existsAsFile());
    check("gui_saved_revision_and_path_match",currentProjectFile==recovered&&savedProjectRevision==project.revisionNumber());
    project.replace(b);check("gui_open_project_succeeds",loadProjectFile(file));
    check("gui_open_resets_undo_and_keeps_correct_path",currentProjectFile==file&&project.snapshot().name==c.name&&!project.canUndo()&&!project.canRedo());
    project.replace(a);check("gui_edit_after_load_undo_stays_in_document",project.undo()&&project.snapshot().name==c.name&&!project.undo());
    savedProjectRevision=project.revisionNumber();newProject();
    check("gui_new_project_has_no_previous_history_or_path",!project.canUndo()&&!project.canRedo()&&currentProjectFile==juce::File{}&&project.snapshot().tracks.empty());
    project.replace(a);currentProjectFile=file;replaceImportedProject(b);
    check("replacement_import_cannot_overwrite_previous_path",currentProjectFile==juce::File{}&&savedProjectRevision!=project.revisionNumber()&&!project.canUndo()&&project.snapshot().name==b.name);
    projectRecovery->capture(project,{});projectRecovery->finish();savedProjectRevision=project.revisionNumber();bool closed=false;
    requestClose([&]{closed=true;});check("approved_close_removes_own_snapshot",closed&&!projectRecovery->currentFile().exists()&&unrelated.existsAsFile());
    projectRecovery.reset();
    project.resetDocument();setAssetManagerVisible(false);
    savedProjectRevision=project.revisionNumber();
    showAssetManager();
    check("asset_panel_toggle_is_saved_dirty_state_without_undo",assetManagerVisible
        &&project.snapshot().assetManagerVisible&&project.revisionNumber()!=savedProjectRevision&&!project.canUndo());
    const auto panelOpen=folder.getChildFile("panel-open.hjpx");
    check("asset_panel_open_save",saveProjectTo(panelOpen));
    const auto cleanRevision=project.revisionNumber();
    project.setAssetManagerVisible(true);
    check("asset_panel_unchanged_setting_keeps_clean_revision",project.revisionNumber()==cleanRevision);
    showAssetManager();
    const auto panelClosed=folder.getChildFile("panel-closed.hjpx");
    check("asset_panel_closed_save",!assetManagerVisible&&saveProjectTo(panelClosed));
    check("asset_panel_load_open_restores_component",loadProjectFile(panelOpen)&&assetManagerVisible
        &&assetManager!=nullptr&&assetManager->isVisible()&&assetManagerResizer->isVisible()
        &&project.revisionNumber()==savedProjectRevision);
    check("asset_panel_load_closed_hides_component",loadProjectFile(panelClosed)&&!assetManagerVisible
        &&!assetManager->isVisible()&&!assetManagerResizer->isVisible());
    ProjectModel fresh;
    check("asset_panel_fresh_model_reads_disk",fresh.load(panelOpen,error)&&fresh.snapshot().assetManagerVisible);
    auto legacy=projectio::validatedTree(bytes(panelOpen));legacy.removeProperty("assetManagerVisible",nullptr);
    juce::MemoryOutputStream legacyBytes;legacy.writeToStream(legacyBytes);
    const auto legacyFile=folder.getChildFile("panel-legacy.hjpx");
    legacyFile.replaceWithData(legacyBytes.getData(),legacyBytes.getDataSize());
    loadProjectFile(panelOpen);
    check("asset_panel_legacy_missing_field_defaults_closed",loadProjectFile(legacyFile)&&!assetManagerVisible);
    showAssetManager();project.addTrack("undo fixture");showAssetManager();
    check("asset_panel_music_undo_preserves_visibility",project.undo()&&!project.snapshot().assetManagerVisible);
    showAssetManager();
    check("asset_panel_toggle_preserves_music_redo",project.redo()&&project.snapshot().assetManagerVisible);
    const auto panelRecovery=folder.getChildFile("panel-recovery.hjpx");
    check("asset_panel_recovery_keeps_state",ProjectModel::saveRecoverySnapshot(project.snapshot(),panelRecovery,{},error)
        &&fresh.load(panelRecovery,error)&&fresh.snapshot().assetManagerVisible);
    savedProjectRevision=project.revisionNumber();newProject();
    check("asset_panel_new_project_defaults_closed",!assetManagerVisible&&!project.snapshot().assetManagerVisible);
    return ok;
}
}
