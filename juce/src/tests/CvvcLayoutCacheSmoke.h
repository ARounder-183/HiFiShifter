#pragma once
namespace hachi
{
class CvvcLayoutCacheSmoke final : private juce::Timer, public std::enable_shared_from_this<CvvcLayoutCacheSmoke>
{
public:
    static void start(const juce::File& bank,const juce::File& folder,std::function<void(bool)> done)
    {
        auto state=std::shared_ptr<CvvcLayoutCacheSmoke>(new CvvcLayoutCacheSmoke(bank,folder,std::move(done)));
        state->self=state;state->startTimer(40);
    }
private:
    CvvcLayoutCacheSmoke(const juce::File& bank,const juce::File& output,std::function<void(bool)> callback)
        : folder(output),done(std::move(callback)),roll(model,strings)
    {
        folder.createDirectory();TrackData track;track.id="track";track.compose=true;
        track.pitchAlgorithm=PitchAlgorithm::utau;track.utauMode=UtauMode::classic;track.chineseCvvc=true;track.voicebankDirectory=bank;
        for(int c=0;c<2;++c)
        {
            ClipData clip;clip.id="clip"+juce::String(c);clip.startSeconds=c*2;clip.durationSeconds=1.5;
            for(int n=0;n<2;++n){NoteData note;note.id=juce::String(c*2+n);note.label=n?"ba":"a";
                note.startSeconds=.3+n*.5;note.durationSeconds=.5;clip.notes.push_back(note);}
            track.clips.push_back(clip);
        }
        ProjectData data;data.tracks={track};model.resetDocument(data);
        fingerprint=model.contentFingerprint();revision=model.revisionNumber();
        key=AudioEngine::diagnosticUtauRenderKey(data,"clip0");
        roll.setReadsVoicebankInBackground(true);roll.setSize(1400,750);roll.setFocusedTrack(track.id);
        roll.setShowPhonemeInfo(true);roll.setSelectedNoteIds({});roll.refreshFromModel();
    }
    void timerCallback() override
    {
        auto keep=shared_from_this();++ticks;
        bool complete=true;for(int i=0;i<4;++i)complete&=!roll.diagnosticPhonemeSpans(juce::String(i)).empty();
        if(!complete&&ticks<250)return;
        stopTimer();bool ok=true;
        const auto check=[&](const char* name,bool pass){ok&=pass;std::cout<<name<<'='<<pass<<std::endl;};
        check("background_all_clips_without_selection_or_render",complete);
        check("background_layout_not_a_user_edit",model.revisionNumber()==revision&&model.contentFingerprint()==fingerprint);
        check("background_layout_not_in_audio_cache_key",key==AudioEngine::diagnosticUtauRenderKey(model.snapshot(),"clip0"));
        juce::String error;ProjectModel reopened;
        check("background_layout_persisted",model.save(folder.getChildFile("all-notes.hjpx"),error)
            &&reopened.load(folder.getChildFile("all-notes.hjpx"),error));
        PianoRollComponent view(reopened,strings);view.setSize(1400,750);view.setFocusedTrack("track");view.setShowPhonemeInfo(true);
        view.diagnosticRefresh();bool restored=true;
        for(int i=0;i<4;++i)restored&=!view.diagnosticPhonemeSpans(juce::String(i)).empty()&&view.phonemeInformationVisible(juce::String(i));
        check("all_notes_visible_immediately_after_reopen",restored);
        view.setTool(PianoRollComponent::Tool::amplitude);
        const juce::Rectangle<int> area(0,std::max(0,(int)view.diagnosticYForMidi(60)-150),1400,520);
        juce::PNGImageFormat png;if(auto stream=folder.getChildFile("all-phonemes.png").createOutputStream())
        {stream->setPosition(0);stream->truncate();png.writeImageToStream(view.createComponentSnapshot(area),*stream);}
        done(ok);self.reset();
    }
    juce::File folder;std::function<void(bool)> done;
    ProjectModel model;I18n strings;PianoRollComponent roll;
    std::shared_ptr<CvvcLayoutCacheSmoke> self;
    juce::int64 fingerprint=0;std::uint64_t revision=0;std::string key;int ticks=0;
};
}
