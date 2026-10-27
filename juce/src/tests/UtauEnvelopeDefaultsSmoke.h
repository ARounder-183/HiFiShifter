#pragma once
namespace hachi
{
template<class Check>
void verifyUtauEnvelopeDefaults(const ProjectData& source,const TrackData& original,
    const juce::File& output,Check&& check)
{
    const auto sung=std::find_if(original.clips.front().notes.begin(),original.clips.front().notes.end(),
        [](const auto& n){return n.label.isNotEmpty()&&n.label!="-";});
    if(sung==original.clips.front().notes.end()){check("default_fixture_note",false);return;}
    for(const auto mode:{UtauMode::classic,UtauMode::jie,UtauMode::mou})
    {
        const auto tag="default_mode_"+juce::String(static_cast<int>(mode))+"_";
        auto data=source;auto track=original;track.chineseCvvc=false;track.utauMode=mode;
        track.clips.resize(1);auto& clip=track.clips.front();clip.notes={*sung};
        auto& note=clip.notes.front();const auto lyric=note.label;note.label.clear();
        note.amplitudeEnvelope.clear();note.cvvcEnvelopes.clear();note.cvvcPhonemeCache.clear();
        data.tracks={track};ProjectModel model;model.resetDocument(data);I18n strings;
        PianoRollComponent roll(model,strings);roll.setSize(1200,700);roll.setFocusedTrack(track.id);roll.setFocusedClip(clip.id);
        check(tag+"blank_note_can_enter_lyric",roll.diagnosticBeginAliasEdit(note.id));
        roll.diagnosticCommitAliasEdit(lyric);model.dispatchPendingMessages();roll.diagnosticRefresh();
        const auto initial=model.snapshot().tracks[0].clips[0].notes[0].amplitudeEnvelope;
        check(tag+"lyric_creates_standard_linear_default",initial.size()==4&&
            std::all_of(initial.begin(),initial.end(),[](const auto& p){return p.linearToNext;})
            &&std::abs(initial[1].timeSeconds-initial[0].timeSeconds-.005)<1.e-6
            &&std::abs(initial[3].timeSeconds-initial[2].timeSeconds-.035)<1.e-6);
        const auto request=AudioEngine::utauRequestNotesForClip(model.snapshot(),clip.id);
        check(tag+"renderer_receives_linear_envelope",request.size()==1&&request[0].amplitudeEnvelope.size()==4
            &&std::all_of(request[0].amplitudeEnvelope.begin(),request[0].amplitudeEnvelope.end(),[](const auto& p){return p.linearToNext;}));
        for(size_t index=0;index<PianoRollComponent::envelopePresets().size();++index)
        {
            const auto before=model.contentFingerprint();roll.setSelectedNoteIds({note.id});
            check(tag+"preset_"+juce::String(index)+"_applies",roll.applyEnvelopePreset(PianoRollComponent::envelopePresets()[index])==1);
            const auto after=model.contentFingerprint();
            // The standard preset may already equal the initial standard shape.
            if(before!=after)
                check(tag+"preset_"+juce::String(index)+"_undo_redo",model.undo()&&model.contentFingerprint()==before
                    &&model.redo()&&model.contentFingerprint()==after);
            model.dispatchPendingMessages();roll.diagnosticRefresh();
        }
        juce::String error;ProjectModel reopened;const auto saved=output.getChildFile(tag+"presets.hjpx");
        check(tag+"preset_roundtrip",model.save(saved,error)&&reopened.load(saved,error)
            &&model.contentFingerprint()==reopened.contentFingerprint());
    }
}
}
