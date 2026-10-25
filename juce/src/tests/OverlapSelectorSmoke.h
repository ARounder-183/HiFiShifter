#pragma once
#include "../PianoRollComponent.h"
#include "../AudioEngine.h"
#include "../Theme.h"
#include <juce_audio_formats/juce_audio_formats.h>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace hachi
{
inline bool runOverlapSelectorSmoke(const juce::File& output)
{
    const auto root=output.getChildFile("overlap-selector-"+juce::Uuid().toString());if(root.createDirectory().failed())return false;
    struct RestorePalette
    {
        std::vector<std::pair<juce::Colour*,juce::Colour>> saved;
        RestorePalette(){for(auto* p:{&Palette::background,&Palette::base,&Palette::panel,&Palette::panelRaised,&Palette::button,&Palette::buttonHover,
            &Palette::border,&Palette::graphBackground,&Palette::clipBackground,&Palette::grid,&Palette::beatGrid,&Palette::accent,&Palette::accentLight,
            &Palette::noteFill,&Palette::noteLight,&Palette::noteEdge,&Palette::pitchLine,&Palette::playhead,&Palette::text,&Palette::textMuted,
            &Palette::scrollThumb,&Palette::heading,&Palette::selection,&Palette::primaryButton,&Palette::warning})saved.emplace_back(p,*p);}
        ~RestorePalette(){for(const auto& p:saved)*p.first=p.second;}
    } restorePalette;
    const auto accent=Palette::accent,accentLight=Palette::accentLight,noteColour=Palette::noteFill;
    bool ok=true;int total=0;juce::String log;
    const auto check=[&](const juce::String& name,bool pass)
    {++total;ok&=pass;log+=name+"="+(pass?"PASS":"FAIL")+"\n";};
    const auto finish=[&]{const bool written=root.getChildFile("verification.txt").replaceWithText(log);std::cout<<"overlap_selector_passed="<<(ok?1:0)<<"|checks="<<total<<"|fixture="<<root.getFullPathName()<<std::endl;if(!ok)std::cerr<<log<<std::endl;return ok&&written;};
    const auto makeAudio=[](const juce::File& file,float midi)
    {
        juce::AudioBuffer<float> pcm(1,8000);const auto hz=440*std::exp2((midi-69)/12);
        for(int i=0;i<8000;++i)pcm.setSample(0,i,.1f*static_cast<float>(std::sin(juce::MathConstants<double>::twoPi*hz*i/8000)));
        auto out=file.createOutputStream();juce::WavAudioFormat wav;
        std::unique_ptr<juce::AudioFormatWriter> writer(out?wav.createWriterFor(out.release(),8000,1,16,{},0):nullptr);
        return writer&&writer->writeFromAudioSampleBuffer(pcm,0,8000);
    };
    const auto a=root.getChildFile("physical-a.wav"),b=root.getChildFile("physical-b.wav");
    check("own_physical_source_fixtures",makeAudio(a,60)&&makeAudio(b,64));
    juce::MemoryBlock beforeA,beforeB;check("original_source_bytes_captured",a.loadFileAsData(beforeA)&&b.loadFileAsData(beforeB));if(!ok)return finish();
    const auto note=[](const juce::String& id,double start,double duration,float midi)
    {NoteData n;n.id=id;n.label=id;n.startSeconds=start;n.durationSeconds=duration;n.midiNote=n.sourceMidiCenter=midi;n.contour={{0,0,0,true},{duration,0,0,true}};return n;};
    const auto renderKeys=[](const ProjectData& data)
    {std::vector<std::uint64_t> values;for(const auto& t:data.tracks)for(const auto& c:t.clips)for(const auto& n:c.notes)values.push_back(AudioEngine::utauNoteRenderHash(n));return values;};
    I18n strings;strings.setLanguage(I18n::Language::enUS);
    for(int engine=0;engine<2;++engine)
    {
        const juce::String tag=engine==0?"native_":"voicebank_";
        ProjectData seed;TrackData track;track.id="material-track";track.compose=true;track.allowNativeAudioOverlap=true;
        track.pitchAlgorithm=engine==0?PitchAlgorithm::world:PitchAlgorithm::utau;
        // Four disjoint short-overlap pairs reproduce the Melodyne import
        // geometry that used to paint four unsolicited orange selector cards.
        const double starts[]{0,.9769,2.4,3.3896,4.8,5.792,7.2,8.1827};
        for(int i=0;i<8;++i)
        {
            ClipData c;c.id="c"+juce::String(i);c.startSeconds=starts[i];c.durationSeconds=c.sourceDurationSeconds=1;c.sourceFile=i%2?b:a;
            c.notes={note("first"+juce::String(i),0,1,60.f+i)};
            if(i==0){c.notes[0].durationSeconds=.4;c.notes.push_back(note("late0",.8,.2,64));}
            track.clips.push_back(c);
        }
        seed.tracks={track};ProjectModel model;model.resetDocument(seed);
        HachiLookAndFeel lookAndFeel;
        struct PianoBackdrop final:juce::Component
        {void paint(juce::Graphics& g) override {g.fillAll(Palette::graphBackground);}} backdrop;
        backdrop.setLookAndFeel(&lookAndFeel);
        PianoRollComponent roll(model,strings);backdrop.addAndMakeVisible(roll);
        roll.setReadsVoicebankInBackground(false);roll.setBounds(0,0,1500,1900);
        roll.setPixelsPerSecond(110);roll.setRowHeight(25);roll.setFocusedTrack(track.id);roll.setFocusedClip("c0");
        roll.setPitchScope(PianoRollComponent::PitchScope::currentTrack);roll.diagnosticRefresh();
        const auto fingerprint=model.contentFingerprint();const auto revision=model.revisionNumber();const auto originalKeys=renderKeys(model.snapshot());
        const juce::Rectangle<int> box(0,static_cast<int>(roll.diagnosticYForMidi(75)-30),1400,570);
        const auto snapshot=[&](const juce::String& name)
        {
            backdrop.setSize(roll.getWidth(),roll.getHeight());
            const auto image=backdrop.createComponentSnapshot(box);auto out=root.getChildFile(tag+name+".png").createOutputStream();
            check(tag+name+"_actual_snapshot",out&&image.isValid()&&juce::PNGImageFormat().writeImageToStream(image,*out));
        };
        snapshot("four-overlaps-default-none");
        check(tag+"overlap_lines_default_OFF",!roll.showsOverlap()&&roll.diagnosticOverlapLines().empty());
        for(const auto theme:{juce::String("light"),juce::String("dark")})
        {
            Palette::applyTheme(theme,accent,accentLight,noteColour);lookAndFeel.refreshColours();
            roll.setShowOverlap(true);snapshot(theme+"-overlap-lines-ON-inside-notes");
            bool exact=roll.diagnosticOverlapLines().size()==8;
            for(const auto& mark:roll.diagnosticOverlapLines())
            {
                bool matched=false;
                for(const auto& c:seed.tracks[0].clips)for(const auto& n:c.notes)if(n.id==mark.first)
                {
                    const auto centre=roll.diagnosticYForMidi(n.midiNote);
                    const auto left=roll.diagnosticEdgeX(c.startSeconds+n.startSeconds),right=roll.diagnosticEdgeX(c.startSeconds+n.startSeconds+n.durationSeconds);
                    matched=std::abs(mark.second.getCentreY()-centre)<.01f&&mark.second.getHeight()<=2.01f
                        &&mark.second.getX()>=left-.01f&&mark.second.getRight()<=right+.01f&&mark.second.getWidth()>0;
                }
                exact&=matched;
            }
            check(tag+theme+"_only_eight_thin_note_clipped_intersection_lines",exact&&roll.diagnosticOverlapCardCount()==0);
            // Same untouched 23.1ms intersection, viewed at a real 900px/sec
            // zoom (~20.79px), not stretched geometry or an inflated warning.
            roll.setPixelsPerSecond(900);backdrop.setSize(roll.getWidth(),roll.getHeight());
            const juce::Rectangle<int> zoomBox(static_cast<int>(roll.diagnosticEdgeX(.45)),box.getY(),1200,box.getHeight());
            const auto zoomSnapshot=[&](const juce::String& suffix)
            {
                const auto image=backdrop.createComponentSnapshot(zoomBox);auto out=root.getChildFile(tag+theme+"-one-pair-900pxsec-"+suffix+".png").createOutputStream();
                check(tag+theme+"_one_pair_zoom_"+suffix+"_actual_snapshot",out&&image.isValid()&&juce::PNGImageFormat().writeImageToStream(image,*out));
            };
            zoomSnapshot("ON");
            bool exactZoom=false;for(const auto& mark:roll.diagnosticOverlapLines())if(mark.first=="late0")
                exactZoom=std::abs(mark.second.getWidth()-static_cast<float>(.0231*900))<.01f&&mark.second.getHeight()<=2.01f;
            check(tag+theme+"_zoom_reveals_exact_23_1ms_not_inflated_line",exactZoom);
            roll.setShowOverlap(false);zoomSnapshot("OFF");roll.setPixelsPerSecond(110);
            snapshot(theme+"-overlap-lines-OFF");
            check(tag+theme+"_off_removes_all_marks_without_data_change",roll.diagnosticOverlapLines().empty()
                &&model.contentFingerprint()==fingerprint&&model.revisionNumber()==revision&&!model.canUndo()&&renderKeys(model.snapshot())==originalKeys);
        }
        check(tag+"default_no_cards_or_source_hit_buttons",roll.diagnosticOverlapCardCount()==0
            &&roll.diagnosticOverlapSelectorTarget().isEmpty()&&roll.diagnosticOverlapSourceBounds("first0").isEmpty());
        roll.setSelectedNoteIds({"late0"});roll.setTool(PianoRollComponent::Tool::points);roll.setTool(PianoRollComponent::Tool::note);
        snapshot("selection-tools-do-not-auto-open");
        check(tag+"selection_and_tool_changes_never_auto_open",roll.diagnosticOverlapCardCount()==0);
        const auto ids=roll.diagnosticEnabledNoteMenuIds("late0");
        check(tag+"one_actual_context_menu_selector_entry",std::count(ids.begin(),ids.end(),PianoRollComponent::materialSelectorNoteMenuItem)==1
            &&roll.diagnosticNoteMenuLabel("late0",PianoRollComponent::materialSelectorNoteMenuItem)=="Material selector");
        roll.applyNoteMenuChoice("late0",PianoRollComponent::materialSelectorNoteMenuItem,0,{},.98845);
        snapshot("explicit-first-selector-one");
        check(tag+"explicit_target_opens_exactly_one_card",roll.diagnosticOverlapCardCount()==1&&roll.diagnosticOverlapSelectorTarget()=="late0"
            &&!roll.diagnosticOverlapSourceBounds("first0").isEmpty()&&!roll.diagnosticOverlapSourceBounds("first1").isEmpty()
            &&roll.diagnosticOverlapSourceBounds("first6").isEmpty());
        roll.setShowOverlap(true);(void)roll.createComponentSnapshot(box);
        check(tag+"line_toggle_does_not_close_explicit_picker",roll.diagnosticOverlapCardCount()==1&&roll.diagnosticOverlapSelectorTarget()=="late0");
        roll.setShowOverlap(false);(void)roll.createComponentSnapshot(box);
        check(tag+"picker_remains_available_with_lines_OFF",roll.diagnosticOverlapCardCount()==1&&roll.diagnosticOverlapLines().empty());
        roll.setSelectedNoteIds({"first6"});
        roll.applyNoteMenuChoice("first6",PianoRollComponent::materialSelectorNoteMenuItem,0,{},8.19135);
        snapshot("different-target-replaces-selector");
        check(tag+"different_target_replaces_not_stacks",roll.diagnosticOverlapCardCount()==1&&roll.diagnosticOverlapSelectorTarget()=="first6"
            &&!roll.diagnosticOverlapSourceBounds("first6").isEmpty()&&!roll.diagnosticOverlapSourceBounds("first7").isEmpty()
            &&roll.diagnosticOverlapSourceBounds("first0").isEmpty());
        check(tag+"Escape_cancels_view_picker",roll.keyPressed(juce::KeyPress(juce::KeyPress::escapeKey)));
        snapshot("cancelled-selector-none");
        check(tag+"cancel_clears_card_state_and_all_source_hits",roll.diagnosticOverlapCardCount()==0&&roll.diagnosticOverlapSelectorTarget().isEmpty()
            &&roll.diagnosticOverlapSourceBounds("first6").isEmpty());
        roll.setSelectedNoteIds({"late0"});roll.applyNoteMenuChoice("late0",PianoRollComponent::materialSelectorNoteMenuItem,0,{},.98845);
        (void)roll.createComponentSnapshot(box);
        const auto sourceRow=roll.diagnosticOverlapSourceBounds("first0");
        const auto mouse=[&](juce::Point<float> at)
        {return juce::MouseEvent(juce::Desktop::getInstance().getMainMouseSource(),at,juce::ModifierKeys(juce::ModifierKeys::leftButtonModifier),0,0,0,0,0,&roll,&roll,juce::Time::getCurrentTime(),at,juce::Time::getCurrentTime(),1,false);};
        check(tag+"correct_source_first_note_hit_available",!sourceRow.isEmpty());
        if(!sourceRow.isEmpty()){const auto at=sourceRow.getCentre();roll.mouseDown(mouse(at));roll.mouseUp(mouse(at));}
        check(tag+"actual_source_pick_uses_first_owned_note_not_nearest_note",roll.selectedNoteIds()==std::vector<juce::String>{"first0"}
            &&roll.diagnosticOverlapSelectorTarget().isEmpty()&&roll.diagnosticOverlapCardCount()==0);
        roll.applyNoteMenuChoice("late0",PianoRollComponent::materialSelectorNoteMenuItem,0,{},.98845);(void)roll.createComponentSnapshot(box);
        const auto close=roll.diagnosticOverlapSelectorCloseBounds();
        check(tag+"one_selector_close_control_available",!close.isEmpty());
        if(!close.isEmpty()){const auto at=close.getCentre();roll.mouseDown(mouse(at));roll.mouseUp(mouse(at));}
        check(tag+"actual_close_hit_cancels_without_other_action",roll.diagnosticOverlapSelectorTarget().isEmpty()&&roll.diagnosticOverlapCardCount()==0);
        check(tag+"picker_actions_do_not_mutate_model_undo_or_render_keys",model.contentFingerprint()==fingerprint&&model.revisionNumber()==revision
            &&!model.canUndo()&&renderKeys(model.snapshot())==originalKeys);
        roll.applyNoteMenuChoice("late0",PianoRollComponent::materialSelectorNoteMenuItem,0,{},.98845);(void)roll.createComponentSnapshot(box);
        model.resetDocument(seed);roll.diagnosticRefresh();snapshot("document-import-does-not-open-selector");
        check(tag+"document_replacement_clears_old_selector",roll.diagnosticOverlapCardCount()==0&&roll.diagnosticOverlapSelectorTarget().isEmpty());
    }
    juce::MemoryBlock afterA,afterB;check("physical_WAV_files_unchanged",a.loadFileAsData(afterA)&&b.loadFileAsData(afterB)&&afterA==beforeA&&afterB==beforeB);
    return finish();
}
}
