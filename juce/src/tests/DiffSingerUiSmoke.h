#pragma once
#include "../MainComponent.h"
#include "../DiffSingerPhonemeEditor.h"
#include "../DiffSingerPronunciationEditor.h"
#include <iostream>

namespace hachi
{
inline void MainComponent::diagnosticLoadDiffSingerBank(const juce::String& trackId, const juce::File& directory)
{
    loadDiffSingerVoicebank(trackId, directory);
}

inline bool MainComponent::diagnosticDiffSingerExpressionLane()
{
    const auto data=project.snapshot();
    const auto& track=data.tracks.front();
    std::vector<juce::String> ids;
    for (const auto& note:track.clips.front().notes) ids.push_back(note.id);
    pianoRoll.setSelectedNoteIds(ids);
    refreshSelectedNoteParameter();
    if (!flagCurveButton.isEnabled()) return false;
    flagCurveButton.onClick();
    if (!flagEnvelopeButton.isEnabled()) return false;
    // DS FLAG now opens the parameter lane; offset enabling is a separate
    // action. Do not rely on the old button toggle or dereference an empty lane.
    project.setNotesUtauFlagCurveEnabled(ids,true);
    project.setNoteUtauFlagCurve(ids.front(),"DS:DYN",{{0,-120},{.7,0}});
    project.setNoteUtauFlagCurve(ids.front(),"DS:GENC",{{0,-60},{.7,60}});
    project.dispatchPendingMessages(); pianoRoll.diagnosticRefresh();
    pianoRoll.setFlagLaneFlag("DS:GENC");
    const auto gender=pianoRoll.flagLaneFlag()=="DS:GENC";
    pianoRoll.setFlagLaneFlag("DS:BREC");
    const auto disabled=pianoRoll.flagLaneFlag()=="DS:GENC";
    pianoRoll.setFlagLaneFlag("DS:PEXP");
    const auto defaults=pianoRoll.flagLaneCurveFor(project.snapshot().tracks.front().clips.front().notes.front());
    const auto default100=!defaults.empty() && defaults.front().value==100;
    pianoRoll.setFlagLaneFlag("DS:DYN"); resized();
    return gender && disabled && default100 && pianoRoll.flagCurveActiveFor(ids.front())
        && pianoRoll.currentTool()==PianoRollComponent::Tool::flagCurve;
}

struct DiffSingerUiSmoke : juce::Timer, std::enable_shared_from_this<DiffSingerUiSmoke>
{
    MainComponent component;
    juce::File folder;
    std::function<void(bool)> done;
    std::shared_ptr<DiffSingerUiSmoke> keep;
    juce::File targetBank;
    juce::String targetTrack;
    std::uint64_t revision = 0;
    int phase = 0, ticks = 0;
    bool ok = true;
    std::vector<PitchCurveEditPoint> lockedPitch, lockedReference;
    juce::String originalLyric;
    juce::String scopeJob;
    std::vector<FlagCurve> lockedParameters;
    void start(const juce::File& projectFile, bool timingSelectionOnly=false)
    {
        keep = shared_from_this();
        juce::String error;
        auto& model = component.diagnosticProject();
        if (!model.load(projectFile, error)) { finish(false); return; }
        auto data = model.snapshot();
        if (timingSelectionOnly)
        {
            component.diagnosticSelectTrack(data.tracks.front().id);
            component.diagnosticRefreshControls(); component.setSize(1400,860);
            beginSelectedTimingCheck(); startTimer(100); return;
        }
        for (auto& track : data.tracks) for (auto& clip : track.clips) for (auto& note : clip.notes)
        { note.pitchControlPoints.clear(); note.contour.clear(); }
        targetBank = data.tracks.front().voicebankDirectory;
        targetTrack = data.tracks.front().id;
        data.tracks.front().voicebankDirectory = juce::File{};
        model.replace(data); model.dispatchPendingMessages();
        component.diagnosticSelectTrack(data.tracks.front().id);
        component.diagnosticRefreshControls(); component.setSize(1400, 860);
        phase = -1;
        component.diagnosticLoadDiffSingerBank(targetTrack, targetBank);
        ok = component.diagnosticEditMenuItemEnabled(54).value_or(false);
        startTimer(100);
    }
    void timerCallback() override
    {
        if (++ticks > 1200) { finish(false); return; }
        if (component.diagnosticEditMenuItemEnabled(54).value_or(false)) return;
        auto& model = component.diagnosticProject();
        if (phase == -1)
        {
            ok = model.snapshot().tracks.front().voicebankDirectory == targetBank
                && model.snapshot().tracks.front().utauMode == UtauMode::mou && ok;
            ok = component.diagnosticStatusText().contains(juce::String::fromUTF8("预检通过")) && ok;
            std::cout << "ui_preflight_load=" << ok << "|status=" << component.diagnosticStatusText() << std::endl;
            component.diagnosticRefreshControls();
            ok = component.diagnosticEditMenuItemEnabled(53).value_or(false) && ok;
            revision = model.revisionNumber();
            phase = 0;
            component.diagnosticChooseMenuItem(53);
        }
        else if (phase == 0)
        {
            ok = model.revisionNumber() == revision + 1 && ok;
            const auto data = model.snapshot();
            for (const auto& note : data.tracks.front().clips.front().notes)
                ok = note.pitchControlPoints.size() > 100
                    && note.diffSingerPitchReference.size()==note.pitchControlPoints.size()
                    && std::count_if(note.pitchControlPoints.begin(),note.pitchControlPoints.end(),
                        [](const auto& p){return !p.diffSingerRestoreSupport;})<=16 && ok;
            component.diagnosticRefreshControls();
            ok = component.diagnosticDiffSingerExpressionLane() && ok;
            const auto screenshot = component.createComponentSnapshot(component.getLocalBounds());
            if (auto stream = folder.getChildFile("DiffSinger-editor.png").createOutputStream())
                juce::PNGImageFormat().writeImageToStream(screenshot, *stream);
            std::cout << "ui_generate=" << ok << "|status=" << component.diagnosticStatusText() << std::endl;
            revision = model.revisionNumber(); phase = 1;
            component.diagnosticChooseMenuItem(53);
            component.diagnosticChooseMenuItem(54);
        }
        else if (phase == 1)
        {
            ok = model.revisionNumber() == revision && ok;
            std::cout << "ui_cancel_preserves_project=" << ok << std::endl;
            ok = component.diagnosticEditMenuItemEnabled(55).value_or(false) && ok;
            component.diagnosticSelectNotes({model.snapshot().tracks.front().clips.front().notes.back().id});
            phase = 2; component.diagnosticChooseMenuItem(55);
        }
        else if (phase == 2)
        {
            auto* dialog = dynamic_cast<juce::DialogWindow*>(juce::Component::getCurrentlyModalComponent());
            auto* editor = dialog != nullptr ? dynamic_cast<DiffSingerPhonemeEditor*>(dialog->getContentComponent()) : nullptr;
            if (editor == nullptr) { finish(false); return; }
            const auto rows = editor->phonemes();
            ok = rows.size() == 6 && editor->displayedPhonemes().size()==2
                && editor->timings().size()==1 && ok;
            if (rows.size() != 6) { dialog->exitModalState(0); finish(false); return; }
            ok = editor->moveBoundary(3,rows[3].start-.06) && ok;
            if (auto stream=folder.getChildFile("DS-phoneme-dialog.png").createOutputStream())
                juce::PNGImageFormat().writeImageToStream(editor->createComponentSnapshot(editor->getLocalBounds()),*stream);
            for (auto* child : editor->getChildren())
                if (auto* button=dynamic_cast<juce::TextButton*>(child); button != nullptr
                    && button->getButtonText()==juce::String::fromUTF8("应用")) { button->onClick(); break; }
            ok = model.revisionNumber()==revision+1 && ok;
            ok = model.snapshot().tracks.front().clips.front().notes.back().diffSingerTiming.isNotEmpty() && ok;
            std::cout << "ui_phoneme_dialog_apply=" << ok << std::endl;
            // Closing a JUCE modal is asynchronous. In a fast GPU worker the
            // next preview may finish before the old modal has left the stack.
            // Start the next user action only after the previous window closes.
            phase = 7;
        }
        else if (phase == 7)
        {
            if (juce::Component::getCurrentlyModalComponent() != nullptr) return;
            phase = 3;
            component.diagnosticChooseMenuItem(56);
        }
        else if (phase == 3)
        {
            auto* dialog = dynamic_cast<juce::DialogWindow*>(juce::Component::getCurrentlyModalComponent());
            auto* editor = dialog != nullptr ? dynamic_cast<DiffSingerPronunciationEditor*>(dialog->getContentComponent()) : nullptr;
            if (editor == nullptr) { finish(false); return; }
            ok = editor->readings().size()==1 && editor->previewText("1").contains("zh/w") && ok;
            originalLyric = model.snapshot().tracks.front().clips.front().notes.back().label;
            editor->setReading("1", "[zh/b zh/ang]");
            phase=4; editor->validate(false);
        }
        else if (phase == 4)
        {
            auto* dialog = dynamic_cast<juce::DialogWindow*>(juce::Component::getCurrentlyModalComponent());
            auto* editor = dialog != nullptr ? dynamic_cast<DiffSingerPronunciationEditor*>(dialog->getContentComponent()) : nullptr;
            if (editor == nullptr) { finish(false); return; }
            ok = editor->previewText("1").contains("zh/b") && ok;
            if (auto stream=folder.getChildFile("DS-pronunciation-dialog.png").createOutputStream())
                juce::PNGImageFormat().writeImageToStream(editor->createComponentSnapshot(editor->getLocalBounds()),*stream);
            revision=model.revisionNumber(); phase=5; editor->validate(true);
        }
        else if (phase == 5)
        {
            auto data=model.snapshot();
            const auto& note=data.tracks.front().clips.front().notes.back();
            ok = model.revisionNumber()==revision+1 && note.label==originalLyric
                && note.diffSingerPronunciation=="[zh/b zh/ang]" && note.diffSingerTiming.isEmpty() && ok;
            juce::String error;
            const auto saved=folder.getChildFile("pronunciation-roundtrip.hjpx");
            ok=model.save(saved,error)&&ok; ProjectModel reopened;
            ok=reopened.load(saved,error)&&ok;
            ok=reopened.snapshot().tracks.front().clips.front().notes.back().diffSingerPronunciation=="[zh/b zh/ang]"&&ok;
            model.undo();
            ok=model.snapshot().tracks.front().clips.front().notes.back().diffSingerPronunciation.isEmpty()&&ok;
            model.redo();
            ok=model.snapshot().tracks.front().clips.front().notes.back().diffSingerPronunciation=="[zh/b zh/ang]"&&ok;
            lockedPitch=model.snapshot().tracks.front().clips.front().notes.front().pitchControlPoints;
            lockedReference=model.snapshot().tracks.front().clips.front().notes.front().diffSingerPitchReference;
            std::cout<<"ui_pronunciation_preview_apply_save_undo="<<ok<<std::endl;
            revision=model.revisionNumber(); phase=6;
            component.diagnosticChooseMenuItem(53);
        }
        else if (phase == 6)
        {
            const auto data=model.snapshot();
            const auto& first=data.tracks.front().clips.front().notes.front();
            ok=model.revisionNumber()==revision+1 && first.pitchControlPoints.size()==lockedPitch.size()&&ok;
            for (size_t i=0;i<std::min(first.pitchControlPoints.size(),lockedPitch.size());++i)
                ok=first.pitchControlPoints[i].timeSeconds==lockedPitch[i].timeSeconds
                    && first.pitchControlPoints[i].targetMidi==lockedPitch[i].targetMidi&&ok;
            ok=data.tracks.front().clips.front().notes.back().pitchControlPoints.size()>100&&ok;
            ok=first.diffSingerPitchReference.size()==lockedReference.size()&&ok;
            for(size_t i=0;i<std::min(first.diffSingerPitchReference.size(),lockedReference.size());++i)
                ok=first.diffSingerPitchReference[i].timeSeconds==lockedReference[i].timeSeconds
                    &&first.diffSingerPitchReference[i].targetMidi==lockedReference[i].targetMidi&&ok;
            const auto& last=data.tracks.front().clips.front().notes.back();
            ok=last.diffSingerPitchReference.size()==last.pitchControlPoints.size()&&ok;
            for(size_t i=0;i<std::min(last.diffSingerPitchReference.size(),last.pitchControlPoints.size());++i)
                ok=last.diffSingerPitchReference[i].targetMidi==last.pitchControlPoints[i].targetMidi&&ok;
            const auto insideHandles=std::count_if(last.pitchControlPoints.begin(),last.pitchControlPoints.end(),
                [&](const auto& p){return !p.diffSingerRestoreSupport&&p.timeSeconds>=0&&p.timeSeconds<=last.durationSeconds;});
            ok=insideHandles<=16&&ok;
            std::cout<<"ui_compact_regenerated_handles="<<insideHandles<<"|passed="<<ok<<std::endl;
            std::cout<<"ui_reference_generation_and_local_retake="<<ok<<std::endl;
            std::cout<<"ui_local_pitch_preserves_other_note="<<ok<<"|status="<<component.diagnosticStatusText()<<std::endl;
            // The timing toolbar must pass its selection to the worker, not
            // merely filter the dialog after a whole-track prediction succeeds.
            beginSelectedTimingCheck();
        }
        else if (phase == 8)
        {
            auto* dialog=dynamic_cast<juce::DialogWindow*>(juce::Component::getCurrentlyModalComponent());
            auto* editor=dialog ? dynamic_cast<DiffSingerPhonemeEditor*>(dialog->getContentComponent()) : nullptr;
            if (!editor) { finish(false); return; }
            const auto rows=editor->phonemes();
            const auto shown=editor->displayedPhonemes();
            std::cout<<"timing_scope_rows="<<rows.size()<<"|shown="<<shown.size()<<"|edits="<<editor->timings().size()
                <<"|revision="<<model.revisionNumber()<<"|before="<<revision<<std::endl;
            ok=rows.size()==6 && !shown.empty() && editor->timings().size()==1
                && model.revisionNumber()==revision && ok;
            for (const auto& row:shown) ok=row.id=="0"&&ok;
            ok=std::any_of(rows.begin(),rows.end(),[](const auto& row){return row.id=="1";})&&ok;
            if (auto stream=folder.getChildFile("DS-selected-timing-with-remote-overlap.png").createOutputStream())
                juce::PNGImageFormat().writeImageToStream(editor->createComponentSnapshot(editor->getLocalBounds()),*stream);
            std::cout<<"ui_selected_timing_ignores_remote_overlap_preserves_context="<<ok<<std::endl;
            dialog->exitModalState(0); phase=9;
        }
        else if (phase == 9)
        {
            if (juce::Component::getCurrentlyModalComponent()) return;
            component.diagnosticSelectNotes({model.snapshot().tracks.front().clips.back().notes.front().id});
            phase=10; component.diagnosticChooseMenuItem(55);
        }
        else if (phase == 10)
        {
            auto* alert=dynamic_cast<juce::AlertWindow*>(juce::Component::getCurrentlyModalComponent());
            if (!alert) { finish(false); return; }
            bool overlapWarning=false;
            for (auto* child:alert->getChildren())
                if (auto* label=dynamic_cast<juce::Label*>(child))
                    overlapWarning=overlapWarning||label->getText().contains(juce::String::fromUTF8("重叠"));
            ok=overlapWarning && ok;
            std::cout<<"ui_selected_overlapping_note_still_warns="<<ok<<std::endl;
            alert->exitModalState(0);
            phase=11;
        }
        else if (phase == 11)
        {
            if (juce::Component::getCurrentlyModalComponent()) return;
            auto data=model.snapshot();
            auto remote=data.tracks.front().clips.back();
            remote.id=juce::Uuid().toString(); remote.startSeconds=20;
            remote.notes.resize(1); remote.notes.front().id=juce::Uuid().toString();
            remote.notes.front().label="[missing_phoneme_229]";
            data.tracks.front().clips.push_back(remote);
            model.replace(data); model.dispatchPendingMessages();
            component.diagnosticSelectNotes({data.tracks.front().clips.front().notes.front().id});
            lockedPitch=data.tracks.front().clips.front().notes.back().pitchControlPoints;
            revision=model.revisionNumber(); phase=12;
            component.diagnosticChooseMenuItem(53);
        }
        else if (phase == 12)
        {
            const auto data=model.snapshot();const auto& notes=data.tracks.front().clips.front().notes;
            bool same=notes.back().pitchControlPoints.size()==lockedPitch.size();
            for(size_t i=0;i<std::min(notes.back().pitchControlPoints.size(),lockedPitch.size());++i)
                same=same && notes.back().pitchControlPoints[i].timeSeconds==lockedPitch[i].timeSeconds
                    && notes.back().pitchControlPoints[i].targetMidi==lockedPitch[i].targetMidi;
            const auto passed=model.revisionNumber()==revision+1 && notes.front().pitchControlPoints.size()>100 && same;
            ok=passed&&ok;std::cout<<"ui_local_pitch_ignores_remote_errors_preserves_other_note="<<passed<<std::endl;
            if(!passed){finish(false);return;}
            revision=model.revisionNumber(); phase=13;
            component.diagnosticGenerateDiffSingerParameters();
        }
        else if (phase == 13)
        {
            const auto data=model.snapshot();const auto& notes=data.tracks.front().clips.front().notes;
            const auto hasReference=[](const auto& note){return std::any_of(note.utauFlagCurves.begin(),note.utauFlagCurves.end(),
                [](const auto& curve){return curve.flag.startsWith("DS:REF:")&&!curve.points.empty();});};
            const auto passed=model.revisionNumber()==revision+1 && hasReference(notes.front()) && !hasReference(notes.back());
            ok=passed&&ok;std::cout<<"ui_local_parameters_ignores_remote_errors_preserves_other_note="<<passed<<std::endl;
            if(!passed){finish(false);return;}
            lockedParameters=notes.front().utauFlagCurves;
            component.diagnosticSelectNotes({notes.back().id});
            revision=model.revisionNumber();phase=17;
            component.diagnosticShowDiffSingerParameterLane();
        }
        else if (phase == 17)
        {
            const auto data=model.snapshot();const auto& notes=data.tracks.front().clips.front().notes;
            const auto& current=notes.front().utauFlagCurves;
            bool same=current.size()==lockedParameters.size();
            for(size_t i=0;i<std::min(current.size(),lockedParameters.size());++i)
            {
                same=same&&current[i].flag==lockedParameters[i].flag&&current[i].points.size()==lockedParameters[i].points.size();
                for(size_t j=0;j<std::min(current[i].points.size(),lockedParameters[i].points.size());++j)
                    same=same&&current[i].points[j].timeSeconds==lockedParameters[i].points[j].timeSeconds
                        &&current[i].points[j].value==lockedParameters[i].points[j].value;
            }
            const auto passed=model.revisionNumber()==revision+1 && same
                &&std::any_of(notes.back().utauFlagCurves.begin(),notes.back().utauFlagCurves.end(),
                    [](const auto& c){return c.flag.startsWith("DS:REF:")&&!c.points.empty();});
            ok=passed&&ok;std::cout<<"ui_parameter_lane_loads_current_selection_not_other_notes="<<passed<<std::endl;
            if(!passed){finish(false);return;}
            component.diagnosticSelectNotes({notes.front().id});
            phase=14; component.diagnosticChooseMenuItem(56);
        }
        else if (phase == 14)
        {
            auto* dialog=dynamic_cast<juce::DialogWindow*>(juce::Component::getCurrentlyModalComponent());
            auto* editor=dialog ? dynamic_cast<DiffSingerPronunciationEditor*>(dialog->getContentComponent()) : nullptr;
            if(!editor){finish(false);return;}
            const auto passed=editor->readings().size()==1 && editor->previewText("0").contains("zh/zh");
            ok=passed&&ok;std::cout<<"ui_reading_preview_ignores_remote_errors="<<passed<<std::endl;
            if(!passed){dialog->exitModalState(0);finish(false);return;}
            editor->setReading("0","[zh/b zh/ang]");
            revision=model.revisionNumber(); phase=15; editor->validate(true);
        }
        else if (phase == 15)
        {
            if(juce::Component::getCurrentlyModalComponent())return;
            const auto data=model.snapshot();const auto& notes=data.tracks.front().clips.front().notes;
            const auto passed=model.revisionNumber()==revision+1 && notes.front().diffSingerPronunciation=="[zh/b zh/ang]"
                && notes.back().diffSingerPronunciation.isEmpty();
            ok=passed&&ok;std::cout<<"ui_reading_apply_only_selected_note="<<passed<<std::endl;
            juce::String error;const auto file=folder.getChildFile("scope-roundtrip.hjpx");
            ProjectModel reopened;
            ok=model.save(file,error)&&reopened.load(file,error)&&ok;
            ok=reopened.snapshot().tracks.front().clips.front().notes.front().diffSingerPronunciation=="[zh/b zh/ang]"&&ok;
            model.undo();ok=model.snapshot().tracks.front().clips.front().notes.front().diffSingerPronunciation.isEmpty()&&ok;
            std::cout<<"ui_scope_save_and_undo="<<ok<<std::endl;
            model.dispatchPendingMessages();
            component.diagnosticSelectNotes({model.snapshot().tracks.front().clips.front().notes.front().id});
            component.startLiveMcp();
            auto reply=callMcp("ds_query_phonemes",juce::JSON::parse("{}"));
            if((bool)reply["isError"]){finish(false);return;}
            scopeJob=juce::JSON::parse(reply["content"][0]["text"].toString())["job_id"].toString();
            if(scopeJob.isEmpty()){finish(false);return;}
            phase=16;
        }
        else if (phase == 16)
        {
            auto args=juce::JSON::parse("{}");args.getDynamicObject()->setProperty("job_id",scopeJob);
            const auto reply=callMcp("editor_job_status",args);
            const auto job=juce::JSON::parse(reply["content"][0]["text"].toString());
            if(job["state"].toString()=="running")return;
            const auto result=juce::JSON::parse(job["result"]["content"][0]["text"].toString());
            bool passed=job["state"].toString()=="completed" && result["phonemes"].size()==2;
            const auto id=model.snapshot().tracks.front().clips.front().notes.front().id;
            if(const auto* rows=result["phonemes"].getArray())for(const auto& row:*rows)passed=passed&&row["note_id"].toString()==id;
            ok=passed&&ok;std::cout<<"mcp_phoneme_query_scopes_before_prediction="<<passed<<std::endl;
            component.diagnosticSelectNotes({});
            const auto empty=callMcp("ds_query_phonemes",juce::JSON::parse("{}"));
            scopeJob=juce::JSON::parse(empty["content"][0]["text"].toString())["job_id"].toString();
            if(scopeJob.isEmpty()){finish(false);return;}
            phase=18;
        }
        else if (phase == 18)
        {
            auto args=juce::JSON::parse("{}");args.getDynamicObject()->setProperty("job_id",scopeJob);
            const auto reply=callMcp("editor_job_status",args);
            const auto job=juce::JSON::parse(reply["content"][0]["text"].toString());
            if(job["state"].toString()=="running")return;
            const auto passed=job["state"].toString()=="failed" && (bool)job["result"]["isError"]
                && job["result"]["content"][0]["text"].toString().contains("Select DS notes first");
            ok=passed&&ok;
            std::cout<<"mcp_empty_selection_does_not_expand_to_whole_track="<<passed<<std::endl;
            component.stopLiveMcp();
            finish(ok);
        }
    }
    juce::var callMcp(const juce::String& name,const juce::var& args)
    {
        auto request=juce::JSON::parse(R"({"jsonrpc":"2.0","id":229,"method":"tools/call","params":{}})");
        request["params"].getDynamicObject()->setProperty("name",name);
        request["params"].getDynamicObject()->setProperty("arguments",args);
        bool respond=false;return component.handleLiveMcp(request,respond)["result"];
    }
    void beginSelectedTimingCheck()
    {
        auto& model=component.diagnosticProject();
        auto overlapData=model.snapshot();
        auto remote=overlapData.tracks.front().clips.front();
        remote.id=juce::Uuid().toString(); remote.startSeconds+=8;
        auto note=remote.notes.front();
        note.id=juce::Uuid().toString(); note.startSeconds=0; note.durationSeconds=1;
        remote.notes={note};
        note.id=juce::Uuid().toString(); note.startSeconds=.1;
        remote.notes.push_back(note);
        overlapData.tracks.front().clips.push_back(remote);
        model.replace(overlapData); model.dispatchPendingMessages();
        component.diagnosticSelectNotes({overlapData.tracks.front().clips.front().notes.front().id});
        revision=model.revisionNumber(); phase=8;
        component.diagnosticChooseMenuItem(55);
    }
    void finish(bool success)
    {
        stopTimer();
        if (!success)
        {
            auto* modal = juce::Component::getCurrentlyModalComponent();
            std::cout << "ui_failure_phase=" << phase << "|modal=" << (modal ? modal->getName() : juce::String("none"))
                << "|status=" << component.diagnosticStatusText() << std::endl;
        }
        auto self = keep;
        juce::MessageManager::callAsync([self, success]
        { self->done(success); self->keep.reset(); });
    }
};
inline void runDiffSingerUiSmoke(const juce::File& project, const juce::File& folder,
                                 std::function<void(bool)> done, bool timingSelectionOnly=false)
{
    auto test = std::make_shared<DiffSingerUiSmoke>();
    test->folder = folder; folder.createDirectory(); test->done = std::move(done);
    test->start(project,timingSelectionOnly);
}
}
