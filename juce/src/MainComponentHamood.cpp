#include "MainComponent.h"
#include "Hamood.h"
#include "HamoodProject.h"
#include "HamoodTimelinePanel.h"
#include <iostream>
#include <set>

namespace hachi
{
namespace
{
juce::String tr(const char* text){return juce::String::fromUTF8(text);}
class HamoodPanel final : public juce::Component, private juce::ListBoxModel
{
public:
    HamoodPanel(ProjectData snapshot, const juce::String& focused, std::vector<juce::String> selected,
                std::function<bool(const hamood::Options&,const hamood::Plan&)> apply,
                std::function<bool(const juce::String&)> save = {}, juce::File cache = {})
        : data(std::move(snapshot)), selection(std::move(selected)), commit(std::move(apply)), persist(std::move(save)), cacheFolder(cache)
    {
        state=hamoodstate::read(data);
        heading.setText(tr("HAMOOD · 自动和声"),juce::dontSendNotification);
        hint.setText(tr("有效修改随工程保存，关闭窗口后保留。生成和声支持 Ctrl+Z 一次撤销。"),juce::dontSendNotification);
        sourceLabel.setText(tr("源轨道"),juce::dontSendNotification);
        scopeLabel.setText(tr("处理范围"),juce::dontSendNotification);
        keyLabel.setText(tr("调性方式"),juce::dontSendNotification);
        volumeLabel.setText(tr("和声音量（相对原轨）"),juce::dontSendNotification);
        for(auto* label:{&heading,&hint,&sourceLabel,&scopeLabel,&keyLabel,&volumeLabel})addAndMakeVisible(*label);
        for(const auto& t:data.tracks)
            if(t.compose||t.pitchAlgorithm==PitchAlgorithm::utau)
            {
                tracks.push_back(t.id);source.addItem(t.name,(int)tracks.size());
                if(t.id==focused)source.setSelectedId((int)tracks.size(),juce::dontSendNotification);
            }
        if(source.getSelectedId()==0&&!tracks.empty())source.setSelectedId(1,juce::dontSendNotification);
        scope.addItem(tr("当前轨道的选中音符"),1);scope.addItem(tr("当前整轨（明确选择）"),2);
        scope.setSelectedId(selection.empty()?2:1,juce::dontSendNotification);
        mode.addItem(tr("自动分析 · 单一调性"),1);mode.addItem(tr("自动分析 · 分段转调"),2);mode.addItem(tr("手动指定调性"),3);
        mode.setSelectedId(1,juce::dontSendNotification);
        const char* names[]{"C","C# / Db","D","D# / Eb","E","F","F# / Gb","G","G# / Ab","A","A# / Bb","B"};
        for(int i=0;i<12;++i)tonic.addItem(names[i],i+1);tonic.setSelectedId(1,juce::dontSendNotification);
        scale.addItem(tr("大调"),1);scale.addItem(tr("自然小调"),2);scale.setSelectedId(1,juce::dontSendNotification);
        for(int n:{2,4,8,16})bars.addItem(juce::String(n)+tr(" 小节 / 段"),n);bars.setSelectedId(8,juce::dontSendNotification);
        for(auto* box:{&source,&scope,&mode,&tonic,&scale,&bars}){addAndMakeVisible(*box);box->onChange=[this]{refresh();};}
        manualHint.setText(tr("手动分段：选择上方调性，再填写小节范围；未添加分段时整段使用上方调性。"),juce::dontSendNotification);
        startLabel.setText(tr("起始小节"),juce::dontSendNotification);countLabel.setText(tr("小节数量"),juce::dontSendNotification);
        for(auto* label:{&manualHint,&startLabel,&countLabel,&rangeLabel})addAndMakeVisible(*label);
        for(auto* editor:{&startBar,&barCount}){editor->setInputRestrictions(7,"0123456789");addAndMakeVisible(*editor);editor->onTextChange=[this]{updateRange();};}
        startBar.setText("1",false);barCount.setText("8",false);
        addSection.setButtonText(tr("添加分段"));updateSection.setButtonText(tr("更新所选"));removeSection.setButtonText(tr("删除所选"));
        for(auto* button:{&addSection,&updateSection,&removeSection})addAndMakeVisible(*button);
        sectionList.setModel(this);sectionList.setRowHeight(25);addAndMakeVisible(sectionList);
        addSection.onClick=[this]{
            if(!validRange())return;
            const int start=startBar.getText().getIntValue(),count=barCount.getText().getIntValue();
            manualSections.push_back({start,start+count-1,tonic.getSelectedId()-1,scale.getSelectedId()==2,sectionConfirmed.getToggleState()});
            sectionList.deselectAllRows();sectionList.updateContent();startBar.setText(juce::String(start+count));refresh();
        };
        updateSection.onClick=[this]{
            const auto row=sectionList.getSelectedRow();if(row<0||row>=(int)manualSections.size()||!validRange())return;
            const int start=startBar.getText().getIntValue();
            manualSections[(size_t)row]={start,start+barCount.getText().getIntValue()-1,tonic.getSelectedId()-1,scale.getSelectedId()==2,sectionConfirmed.getToggleState()};
            sectionList.updateContent();refresh();
        };
        removeSection.onClick=[this]{const auto row=sectionList.getSelectedRow();if(row>=0&&row<(int)manualSections.size()){manualSections.erase(manualSections.begin()+row);sectionList.deselectAllRows();sectionList.updateContent();refresh();}};
        for(int steps:{2,-2,3,-3,4,-4,5,-5,7,-7})
        {
            auto button=std::make_unique<juce::ToggleButton>(hamood::voiceName(steps));
            button->setToggleState(steps==2,juce::dontSendNotification);button->onClick=[this]{refresh();};
            addAndMakeVisible(*button);voices.emplace_back(steps,std::move(button));
        }
        preserve.setButtonText(tr("保留原唱法：pitch / 参考线平移，DS 偏移保持"));preserve.setToggleState(true,juce::dontSendNotification);
        preserve.onClick=[this]{refresh();};addAndMakeVisible(preserve);
        gain.setRange(-36,6,.5);gain.setValue(-6,juce::dontSendNotification);gain.setTextValueSuffix(" dB");
        gain.setSliderStyle(juce::Slider::LinearHorizontal);gain.setTextBoxStyle(juce::Slider::TextBoxRight,false,80,26);
        gain.onValueChange=[this]{refresh();};addAndMakeVisible(gain);
        preview.setMultiLine(true);preview.setReadOnly(true);preview.setScrollbarsShown(true);addAndMakeVisible(preview);
        generate.setButtonText(tr("生成和声轨道"));cancel.setButtonText(tr("取消"));
        generate.onClick=[this]{auto options=readOptions();auto plan=hamood::analyse(data,options);if(plan.error.isEmpty()&&commit(options,plan))close();};
        cancel.onClick=[this]{close();};addAndMakeVisible(generate);addAndMakeVisible(cancel);
        sectionConfirmed.setButtonText(tr("已人工确认"));sectionConfirmed.setToggleState(true,juce::dontSendNotification);
        sectionConfirmed.onClick=[this]{const auto i=sectionList.getSelectedRow();if(i>=0&&i<(int)manualSections.size()){manualSections[(size_t)i].confirmed=sectionConfirmed.getToggleState();sectionList.repaint();refresh();}};
        adoptPrediction.setButtonText(tr("采用预测调性"));adoptPrediction.onClick=[this]{
            auto plan=hamood::analyse(data,readOptions());if(plan.error.isNotEmpty())return;
            manualSections.clear();for(const auto& p:plan.passages)manualSections.push_back({p.startBar,p.endBar,p.key%12,p.key>=12,false});
            mode.setSelectedId(3,juce::dontSendNotification);sectionList.updateContent();refresh();
        };
        useChords.setButtonText(tr("使用工程中的和弦资料"));useChords.onClick=[this]{auto s=state["settings"];hamoodstate::put(s,"use_chords",useChords.getToggleState());hamoodstate::put(state,"settings",s);refresh();};
        timelineButton.setButtonText(tr("和弦 / 段落资料…"));timelineButton.onClick=[this]{openTimeline();};
        for(juce::Component* c:std::initializer_list<juce::Component*>{&sectionConfirmed,&adoptPrediction,&useChords,&timelineButton})addAndMakeVisible(*c);
        loadSettings();cancel.setButtonText(tr("关闭"));setSize(880,800);refresh(false);
    }
    hamood::Options readOptions() const
    {
        hamood::Options o;
        if(source.getSelectedId()>0)o.trackId=tracks[(size_t)source.getSelectedId()-1];
        o.noteIds=selection;o.wholeTrack=scope.getSelectedId()==2;
        o.keyMode=mode.getSelectedId()==3?"manual":mode.getSelectedId()==2?"sections":"auto";
        o.tonic=tonic.getSelectedId()-1;o.minor=scale.getSelectedId()==2;o.sectionBars=bars.getSelectedId();
        if(o.keyMode=="manual")o.manualSections=manualSections;
        o.voices.clear();for(const auto& [steps,button]:voices)if(button->getToggleState())o.voices.push_back(steps);
        o.preservePitch=preserve.getToggleState();o.gainDb=(float)gain.getValue();o.minimumChordScore=(double)state["settings"].getProperty("minimum_chord_score",.35);hamoodstate::attach(data,state,o);return o;
    }
    void refresh(bool save = true)
    {
        const bool manual=mode.getSelectedId()==3;
        tonic.setEnabled(manual);scale.setEnabled(manual);bars.setEnabled(mode.getSelectedId()==2);
        for(juce::Component* c:std::initializer_list<juce::Component*>{&manualHint,&startLabel,&countLabel,&rangeLabel,&startBar,&barCount,&addSection,&updateSection,&removeSection,&sectionList})c->setVisible(manual);
        sectionConfirmed.setVisible(manual);adoptPrediction.setEnabled(!manual);
        updateRange();resized();
        const auto options=readOptions();const auto plan=hamood::analyse(data,options);preview.setText(plan.description(),false);generate.setEnabled(plan.error.isEmpty());
        if(save)
        {
            auto next=juce::JSON::parse(hamoodstate::encode(state));hamoodstate::settings(next,options,manualSections);hamoodstate::predictions(data,next,plan);
            juce::String error;
            if(!hamoodstate::validate(next,error)){preview.setText(tr("本次无效修改未保存：")+error+"\n"+plan.description(),false);return;}
            const auto encoded=hamoodstate::encode(next);
            if(persist&&!persist(encoded)){preview.setText(tr("工程已变化，请重新打开 HAMOOD。"),false);generate.setEnabled(false);return;}
            state=std::move(next);data.hamoodState=encoded;
        }
    }
    bool validRange() const
    {
        const auto start=startBar.getText(),count=barCount.getText();
        return start.isNotEmpty()&&count.isNotEmpty()&&count.getIntValue()>0
            &&start.getIntValue()>=0&&start.getIntValue()+count.getIntValue()<=1000001;
    }
    void updateRange()
    {
        const bool valid=validRange();rangeLabel.setText(valid?tr("至第 ")+juce::String(startBar.getText().getIntValue()+barCount.getText().getIntValue()-1)+tr(" 小节（含）"):tr("请输入有效小节数"),juce::dontSendNotification);
        addSection.setEnabled(valid&&manualSections.size()<2048);updateSection.setEnabled(valid&&sectionList.getSelectedRow()>=0);removeSection.setEnabled(sectionList.getSelectedRow()>=0);
    }
    int getNumRows() override{return (int)manualSections.size();}
    void paintListBoxItem(int row,juce::Graphics& g,int width,int height,bool selected) override
    {
        if(row<0||row>=(int)manualSections.size())return;
        if(selected)g.fillAll(Palette::selection);
        const auto& s=manualSections[(size_t)row];g.setColour(selected?Palette::contrastingText(Palette::selection):Palette::text);
        g.drawText(tr("第 ")+juce::String(s.startBar)+" – "+juce::String(s.endBar)+tr(" 小节   |   ")+hamood::keyName(s.tonic+(s.minor?12:0))+tr(s.confirmed?"   已确认":"   待确认"),8,0,width-16,height,juce::Justification::centredLeft);
    }
    void selectedRowsChanged(int row) override
    {
        if(row>=0&&row<(int)manualSections.size())
        {
            const auto& s=manualSections[(size_t)row];
            startBar.setText(juce::String(s.startBar),false);barCount.setText(juce::String(s.endBar-s.startBar+1),false);
            tonic.setSelectedId(s.tonic+1,juce::dontSendNotification);scale.setSelectedId(s.minor?2:1,juce::dontSendNotification);
            sectionConfirmed.setToggleState(s.confirmed,juce::dontSendNotification);
        }
        updateRange();
    }
    void paint(juce::Graphics& g) override {g.fillAll(Palette::panel);}
    void resized() override
    {
        auto area=getLocalBounds().reduced(20);
        heading.setBounds(area.removeFromTop(30));hint.setBounds(area.removeFromTop(30));area.removeFromTop(10);
        auto row=area.removeFromTop(34);sourceLabel.setBounds(row.removeFromLeft(70));source.setBounds(row.removeFromLeft(310));row.removeFromLeft(16);scopeLabel.setBounds(row.removeFromLeft(75));scope.setBounds(row);
        area.removeFromTop(10);row=area.removeFromTop(34);keyLabel.setBounds(row.removeFromLeft(70));mode.setBounds(row.removeFromLeft(235));row.removeFromLeft(10);tonic.setBounds(row.removeFromLeft(110));row.removeFromLeft(10);scale.setBounds(row.removeFromLeft(115));row.removeFromLeft(10);bars.setBounds(row);
        if(mode.getSelectedId()==3)
        {
            area.removeFromTop(8);auto manualRow=area.removeFromTop(25);sectionConfirmed.setBounds(manualRow.removeFromRight(135));manualHint.setBounds(manualRow);
            row=area.removeFromTop(32);startLabel.setBounds(row.removeFromLeft(72));startBar.setBounds(row.removeFromLeft(58));row.removeFromLeft(8);
            countLabel.setBounds(row.removeFromLeft(72));barCount.setBounds(row.removeFromLeft(58));rangeLabel.setBounds(row.removeFromLeft(146));
            addSection.setBounds(row.removeFromLeft(100).reduced(3,0));updateSection.setBounds(row.removeFromLeft(100).reduced(3,0));removeSection.setBounds(row.removeFromLeft(100).reduced(3,0));
            area.removeFromTop(6);sectionList.setBounds(area.removeFromTop(80));
        }
        area.removeFromTop(12);
        for(int i=0;i<2;++i){row=area.removeFromTop(32);for(int j=0;j<5;++j)voices[(size_t)(j*2+i)].second->setBounds(row.removeFromLeft((getWidth()-40)/5));}
        area.removeFromTop(6);row=area.removeFromTop(30);adoptPrediction.setBounds(row.removeFromRight(145));row.removeFromRight(8);timelineButton.setBounds(row.removeFromRight(185));useChords.setBounds(row);
        preserve.setBounds(area.removeFromTop(30));
        row=area.removeFromTop(32);volumeLabel.setBounds(row.removeFromLeft(195));gain.setBounds(row);
        area.removeFromTop(12);auto footer=area.removeFromBottom(36);cancel.setBounds(footer.removeFromRight(105));footer.removeFromRight(10);generate.setBounds(footer.removeFromRight(185));
        area.removeFromBottom(12);preview.setBounds(area);
    }
    void loadSettings()
    {
        const auto s=state["settings"];const auto key=s.getProperty("key_mode","auto").toString();
        mode.setSelectedId(key=="manual"?3:key=="sections"?2:1,juce::dontSendNotification);tonic.setSelectedId((int)s.getProperty("tonic",0)+1,juce::dontSendNotification);
        scale.setSelectedId((bool)s.getProperty("minor",false)?2:1,juce::dontSendNotification);
        const int count=(int)s.getProperty("section_bars",8);if(bars.indexOfItemId(count)<0)bars.addItem(juce::String(count)+tr(" 小节 / 段"),count);bars.setSelectedId(count,juce::dontSendNotification);
        gain.setValue((double)s.getProperty("gain_db",-6),juce::dontSendNotification);preserve.setToggleState((bool)s.getProperty("preserve_pitch",true),juce::dontSendNotification);
        if(auto* v=s["voices"].getArray())for(auto& [steps,button]:voices)button->setToggleState(std::any_of(v->begin(),v->end(),[&](const auto& x){return (int)x==steps;}),juce::dontSendNotification);
        manualSections.clear();if(auto* rows=s["manual_sections"].getArray())for(const auto& r:*rows)manualSections.push_back({(int)r["start_bar"],(int)r["end_bar"],(int)r["tonic"],(bool)r["minor"],(bool)r.getProperty("confirmed",true)});
        useChords.setToggleState((bool)s.getProperty("use_chords",false),juce::dontSendNotification);sectionList.updateContent();
    }
    void openTimeline()
    {
        juce::Component::SafePointer<HamoodPanel> safe(this);
        auto* content=new HamoodTimelinePanel(data,state,cacheFolder,[safe](const juce::String& json){
            if(!safe)return false;if(safe->persist&&!safe->persist(json))return false;
            safe->data.hamoodState=json;safe->state=hamoodstate::read(safe->data);safe->useChords.setToggleState((bool)safe->state["settings"].getProperty("use_chords",false),juce::dontSendNotification);safe->refresh(false);return true;
        });
        juce::DialogWindow::LaunchOptions dialog;dialog.dialogTitle=tr("HAMOOD — 工程资料");dialog.dialogBackgroundColour=Palette::panel;dialog.content.setOwned(content);dialog.componentToCentreAround=this;dialog.escapeKeyTriggersCloseButton=true;dialog.useNativeTitleBar=true;dialog.resizable=true;
        if(auto* window=dialog.launchAsync())window->setResizeLimits(820,500,1400,1000);
    }
    void close(){if(auto* window=findParentComponentOfClass<juce::DialogWindow>())window->exitModalState(0);}
    ProjectData data;
    std::vector<juce::String> selection,tracks;
    std::function<bool(const hamood::Options&,const hamood::Plan&)> commit;
    juce::Label heading,hint,sourceLabel,scopeLabel,keyLabel,volumeLabel;
    juce::ComboBox source,scope,mode,tonic,scale,bars;
    std::vector<std::pair<int,std::unique_ptr<juce::ToggleButton>>> voices;
    juce::ToggleButton preserve;
    juce::Slider gain;
    juce::TextEditor preview;
    juce::TextButton generate,cancel;
    juce::Label manualHint,startLabel,countLabel,rangeLabel;
    juce::TextEditor startBar,barCount;
    juce::TextButton addSection,updateSection,removeSection;
    std::vector<hamood::ManualSection> manualSections;
    juce::ListBox sectionList;
    juce::var state;
    std::function<bool(const juce::String&)> persist;
    juce::File cacheFolder;
    juce::ToggleButton sectionConfirmed,useChords;
    juce::TextButton adoptPrediction,timelineButton;
};
}
void MainComponent::showHamood()
{
    if(diffSingerBusy){showError(tr("请先等待当前 DS 任务完成，再生成和声。"));return;}
    const auto revision=std::make_shared<std::uint64_t>(project.revisionNumber());
    juce::Component::SafePointer<MainComponent> safe(this);
    auto* content=new HamoodPanel(project.snapshot(),selectedTrackId,pianoRoll.selectedNoteIds(),
        [safe,revision](const hamood::Options& options,const hamood::Plan& plan)
        {
            if(safe==nullptr)return false;
            if(safe->project.revisionNumber()!=*revision)
            {safe->showError(tr("工程已发生变化，请关闭 HAMOOD 后重新打开预览。"));return false;}
            auto data=safe->project.snapshot();const auto ids=hamood::generate(data,options,plan);
            if(ids.empty())return false;
            data.hamoodState=hamoodstate::remember(data,options,plan);
            safe->project.replace(std::move(data));
            safe->statusLabel.setText(tr("HAMOOD 已生成 ")+juce::String((int)ids.size())+tr(" 条和声轨道，可 Ctrl+Z 撤销。"),juce::dontSendNotification);
            return true;
        },[safe,revision](const juce::String& json)
        {
            if(!safe||safe->project.revisionNumber()!=*revision)return false;
            juce::String error;if(!safe->project.setHamoodState(json,error))return false;
            *revision=safe->project.revisionNumber();return true;
        },hamoodaudio::cacheFolder(currentProjectFile));
    juce::DialogWindow::LaunchOptions dialog;dialog.dialogTitle=tr("HAMOOD — 自动和声");dialog.dialogBackgroundColour=Palette::panel;
    dialog.content.setOwned(content);dialog.componentToCentreAround=this;dialog.escapeKeyTriggersCloseButton=true;dialog.useNativeTitleBar=true;dialog.resizable=true;
    if(auto* window=dialog.launchAsync())window->setResizeLimits(840,800,1400,1100);
}
bool MainComponent::diagnosticHamood(const juce::File& directory)
{
    directory.createDirectory();bool passed=true;int checks=0;
    auto check=[&](const juce::String& name,bool value){++checks;passed&=value;std::cout<<(value?"PASS ":"FAIL ")<<name<<std::endl;};
    ProjectData data;TrackData track;track.id="lead";track.name="Lead";track.compose=true;track.pitchAlgorithm=PitchAlgorithm::utau;
    ClipData clip;clip.id="clip";clip.startSeconds=2;clip.durationSeconds=12;
    const int pitches[]{60,62,64,65,67,69,71,72};
    for(int i=0;i<8;++i)
    {
        NoteData note;note.id="n"+juce::String(i);note.label="la";note.midiNote=(float)pitches[i];note.startSeconds=i;note.durationSeconds=.8;
        note.pitchControlPoints={{0.0,note.midiNote},{.4,note.midiNote+.5f}};
        note.diffSingerPitchReference={{0.0,note.midiNote-.5f},{.4,note.midiNote}};
        note.diffSingerPitchOffset={{0.0,-2.0f},{.4,-1.0f}};
        note.diffSingerTiming="old-phrase";note.diffSingerPronunciation="la";note.vibratoEnabled=true;
        note.utauFlagCurves={{"DS:AUTO:BREC",{}},{"DS:REF:BREC",{}},{"DS:ABS:BREC",{}},{"DS:BREC",{}}};
        note.contour={{0.0,20.0f},{.4,-15.0f}};clip.notes.push_back(note);
    }
    auto restNote=clip.notes.front();restNote.id="rest";restNote.label="SP";restNote.startSeconds=9;clip.notes.push_back(restNote);
    track.utauMode=UtauMode::mou;track.voicebankDirectory=directory.getChildFile("ds-fixture");
    track.voicebankDirectory.createDirectory();track.voicebankDirectory.getChildFile("dsconfig.yaml").replaceWithText("# identity-only test fixture\n");
    track.clips.push_back(clip);data.tracks.push_back(track);
    hamood::Options options;options.trackId="lead";options.keyMode="manual";options.wholeTrack=true;options.voices={2,-2,5,-5,7,-7};
    auto plan=hamood::analyse(data,options);
    check("C major diatonic thirds vary between 3 and 4 semitones",plan.error.isEmpty()&&plan.notes[0].targets[0]==64&&plan.notes[2].targets[0]==67&&plan.notes[6].targets[0]==74);
    check("lower thirds and upper/lower octaves",plan.notes[0].targets[1]==57&&plan.notes[0].targets[4]==72&&plan.notes[0].targets[5]==48);
    check("rest notes excluded with timing preserved",plan.notes.size()==8&&plan.notes[0].start==2&&plan.notes[7].start==9);
    check("natural minor harmony",hamood::harmonyPitch(69,21,2)==72&&hamood::harmonyPitch(71,21,2)==74);
    check("chromatic alteration preserved",hamood::harmonyPitch(61,0,2)==65);
    check("microtonal deviation preserved",std::abs(hamood::harmonyPitch(60.25f,0,2)-64.25f)<1.e-5);
    auto generated=data;auto ids=hamood::generate(generated,options,plan);
    check("one new track per voice",ids.size()==6&&generated.tracks.size()==7);
    const auto& n=generated.tracks[1].clips[0].notes[0];
    check("source track untouched",generated.tracks[0].clips[0].notes[0].midiNote==60&&generated.tracks[0].clips[0].notes[0].diffSingerTiming=="old-phrase");
    check("pitch handles and DS reference shift together",n.pitchControlPoints[1].targetMidi==64.5f&&n.diffSingerPitchReference[0].targetMidi==63.5f);
    check("additive DS offset and relative contour unchanged",n.diffSingerPitchOffset[0].targetMidi==-2&&n.contour[0].relativeCents==20);
    check("generated phrase invalidates automatic DS parameter/timing cache",n.diffSingerTiming.isEmpty()&&n.utauFlagCurves.size()==2&&n.utauFlagCurves[0].flag=="DS:ABS:BREC"&&n.utauFlagCurves[1].flag=="DS:BREC");
    check("lyrics timing voice settings and source audio identity retained",n.label=="la"&&n.diffSingerPronunciation=="la"&&generated.tracks[1].clips[0].startSeconds==2&&n.durationSeconds==.8&&generated.tracks[1].pitchAlgorithm==PitchAlgorithm::utau);
    std::set<juce::String> allIds;bool unique=true;
    for(const auto& t:generated.tracks){unique&=allIds.insert(t.id).second;for(const auto& c:t.clips){unique&=allIds.insert(c.id).second;for(const auto& note:c.notes)unique&=allIds.insert(note.id).second;}}
    check("new tracks clips and notes have independent IDs",unique);
    check("new harmony defaults quieter",std::abs(generated.tracks[1].volume-std::pow(10.0f,-6.0f/20.0f))<1.e-5);
    options.wholeTrack=false;options.noteIds={"n2","n5"};options.voices={2};
    plan=hamood::analyse(data,options);generated=data;hamood::generate(generated,options,plan);
    check("selection-only generation preserves absolute placement",plan.notes.size()==2&&generated.tracks[1].clips[0].notes.size()==2&&generated.tracks[1].clips[0].notes[0].startSeconds==2);
    options.noteIds.clear();check("empty selection rejected without whole-track fallback",hamood::analyse(data,options).error.isNotEmpty());
    options.wholeTrack=true;options.preservePitch=false;plan=hamood::analyse(data,options);generated=data;hamood::generate(generated,options,plan);
    const auto& flat=generated.tracks[1].clips[0].notes[0];
    check("flat mode clears expression pitch layers",flat.pitchControlPoints.empty()&&flat.diffSingerPitchReference.empty()&&flat.diffSingerPitchOffset.empty()&&!flat.vibratoEnabled&&flat.contour[0].relativeCents==0);
    options.voices={2,2};check("duplicate voices rejected",hamood::analyse(data,options).error.isNotEmpty());
    options.voices={7};auto high=data;high.tracks[0].clips[0].notes[0].midiNote=125;check("out of range harmony rejected rather than clamped",hamood::analyse(high,options).error.isNotEmpty());
    // Strong triadic fixtures disambiguate relative major/minor; both keys appear in distinct sections.
    auto modulation=data;auto& mc=modulation.tracks[0].clips[0];mc.startSeconds=0;mc.notes.clear();mc.durationSeconds=16;
    for(int i=0;i<16;++i){NoteData note;note.id="m"+juce::String(i);note.label="a";note.startSeconds=i;note.durationSeconds=.9;const int motif[]{60,64,67,60};note.midiNote=(float)(motif[i%4]+(i>=8?2:0));mc.notes.push_back(note);}
    options.voices={2};options.keyMode="auto";
    auto single=modulation;single.tracks[0].clips[0].notes.resize(8);
    check("duration-weighted auto key finds clear C major",hamood::analyse(single,options).passages[0].key==0);
    options.keyMode="sections";options.sectionBars=4;auto segmented=hamood::analyse(modulation,options);
    check("segmented analysis detects sustained C to D modulation",segmented.passages.size()==2&&segmented.passages[0].key==0&&segmented.passages[1].key==2);
    modulation.tempoChanges={{0,120},{16,60}};segmented=hamood::analyse(modulation,options);
    check("segments follow musical time with tempo changes",segmented.passages.size()==2&&segmented.passages[0].end==8);
    ProjectModel model;model.replace(data);const auto rev=model.revisionNumber();model.replace(generated);
    check("generation commits one revision",model.revisionNumber()==rev+1);model.undo();check("one undo restores original track count",model.snapshot().tracks.size()==1);model.redo();check("redo restores independent harmony",model.snapshot().tracks.size()==2);
    const auto file=directory.getChildFile("hamood-roundtrip.hjpx");juce::String error;
    check("save harmony project",model.save(file,error));ProjectModel reopened;check("reopen harmony project",reopened.load(file,error)&&reopened.snapshot().tracks.size()==2);
    int commits=0;
    HamoodPanel panel(data,"lead",{"n0","n1"},[&](const auto& o,const auto& p){++commits;auto copy=data;return hamood::generate(copy,o,p).size()==1;});
    panel.mode.setSelectedId(3,juce::sendNotificationSync);panel.refresh();
    check("GUI defaults to selected notes",!panel.readOptions().wholeTrack&&hamood::analyse(data,panel.readOptions()).notes.size()==2);
    check("preview does not mutate model",commits==0);
    juce::PNGImageFormat png;auto image=panel.createComponentSnapshot(panel.getLocalBounds(),true,1.0f);
    if(auto out=directory.getChildFile("hamood-panel.png").createOutputStream())check("render HAMOOD panel",png.writeImageToStream(image,*out));else check("render HAMOOD panel",false);
    panel.generate.onClick();check("GUI generation invokes previewed options",commits==1);
    panel.voices[0].second->setToggleState(false,juce::dontSendNotification);panel.refresh();check("GUI prevents empty voice generation",!panel.generate.isEnabled());
    // Manual sections use absolute ruler bars, not bars relative to selection.
    auto manual=data;auto& manualClip=manual.tracks[0].clips[0];manualClip.startSeconds=0;manualClip.notes.resize(2);
    manualClip.notes[0].startSeconds=0;manualClip.notes[0].durationSeconds=2.5;
    manualClip.notes[1].startSeconds=2;manualClip.notes[1].durationSeconds=.8;
    hamood::Options man;man.trackId="lead";man.wholeTrack=true;man.keyMode="manual";
    man.manualSections={{1,1,0,false},{2,2,7,true}};
    auto mp=hamood::analyse(manual,man);
    check("manual bar segments assign onset keys and preserve cross-boundary note",mp.error.isEmpty()&&mp.notes[0].key==0&&mp.notes[1].key==19&&mp.notes[0].duration==2.5);
    auto manualGenerated=manual;hamood::generate(manualGenerated,man,mp);
    check("generated pitches follow per-bar manual keys",manualGenerated.tracks[1].clips[0].notes[0].midiNote==64&&manualGenerated.tracks[1].clips[0].notes[1].midiNote==65);
    manual.tempoChanges={{0,120},{4,60}};mp=hamood::analyse(manual,man);
    check("manual bar endpoints follow tempo map",mp.passages[1].start==2&&mp.passages[1].end==6);
    manual.tempoChanges.clear();manual.numerator=3;manual.beatOriginSeconds=.5;manualClip.notes[0].startSeconds=.5;
    mp=hamood::analyse(manual,man);
    check("manual 3/4 bars respect beat origin",mp.error.isEmpty()&&mp.passages[0].start==.5&&mp.passages[1].start==2&&mp.notes[1].key==19);
    man.manualSections={{1,2,0,false},{2,3,7,true}};check("overlapping manual bars rejected",hamood::analyse(manual,man).error.isNotEmpty());
    man.manualSections={{1,1,0,false}};check("uncovered note onsets rejected",hamood::analyse(manual,man).error.isNotEmpty());
    man.wholeTrack=false;man.noteIds={"n0"};check("manual coverage only requires selected notes",hamood::analyse(manual,man).error.isEmpty());
    man.manualSections={{3,1,0,false}};check("reversed manual range rejected",hamood::analyse(manual,man).error.isNotEmpty());
    man.manualSections={{1,2,12,false}};check("invalid manual tonic rejected",hamood::analyse(manual,man).error.isNotEmpty());
    man.manualSections.clear();check("empty table retains single manual key compatibility",hamood::analyse(manual,man).error.isEmpty());
    for(int key=0;key<24;++key)
    {
        auto fixture=single;auto& ns=fixture.tracks[0].clips[0].notes;ns.clear();fixture.tracks[0].clips[0].startSeconds=0;
        const int motif[]={0,7,0,2,4,5,7,4,2,0,7,0};
        for(int i=0;i<48;++i)
        {
            NoteData n;n.id="key-"+juce::String(i);n.label="a";n.startSeconds=i*.5;n.durationSeconds=.45;
            int degree=motif[i%12];if(key>=12&&degree==4)degree=3;
            n.midiNote=(float)(48+key%12+degree);ns.push_back(n);
        }
        hamood::Options ao;ao.trackId="lead";ao.wholeTrack=true;auto ap=hamood::analyse(fixture,ao);
        check("profile analysis across all 24 tonal centres "+juce::String(key),ap.passages[0].key==key);
        ao.wholeTrack=false;ao.noteIds={ns[7].id};auto excerpt=hamood::analyse(fixture,ao);
        check("selection retains full-track key context "+juce::String(key),excerpt.notes.size()==1&&excerpt.passages[0].key==ap.passages[0].key);
    }
    auto thin=single;thin.tracks[0].clips[0].notes.resize(1);hamood::Options thinOptions;thinOptions.trackId="lead";thinOptions.wholeTrack=true;
    check("single-tone ambiguity is reported",hamood::analyse(thin,thinOptions).passages[0].margin==0);
    auto parallel=single;auto& pn=parallel.tracks[0].clips[0].notes;pn.clear();
    const int parallelMotif[]{0,7,0,2,4,5,7,4,2,0,7,0,0,7,4,0};
    for(int i=0;i<48;++i)
    {
        NoteData n;n.id="parallel-"+juce::String(i);n.label="a";n.startSeconds=i*.5;n.durationSeconds=.45;
        int degree=parallelMotif[i%16];if(i>=16&&i<32&&degree==4)degree=3;
        n.midiNote=(float)(60+degree);pn.push_back(n);
    }
    hamood::Options parallelOptions;parallelOptions.trackId="lead";parallelOptions.wholeTrack=true;parallelOptions.keyMode="sections";parallelOptions.sectionBars=4;
    const auto parallelPlan=hamood::analyse(parallel,parallelOptions);
    check("smoothing preserves a sustained parallel minor section between major sections",parallelPlan.passages.size()==3&&parallelPlan.passages[0].key==0&&parallelPlan.passages[1].key==12&&parallelPlan.passages[2].key==0);
    parallelOptions.wholeTrack=false;parallelOptions.noteIds={"parallel-20"};const auto selectedParallel=hamood::analyse(parallel,parallelOptions);
    check("selected section uses the same modulation as full-track analysis",selectedParallel.notes.size()==1&&selectedParallel.notes[0].key==parallelPlan.notes[20].key);
    panel.voices[0].second->setToggleState(true,juce::dontSendNotification);
    panel.startBar.setText("1",false);panel.barCount.setText("1",false);panel.tonic.setSelectedId(1,juce::dontSendNotification);panel.scale.setSelectedId(1,juce::dontSendNotification);
    panel.addSection.onClick();check("GUI adds inclusive bar segment",panel.manualSections.size()==1&&panel.manualSections[0].endBar==1&&panel.startBar.getText()=="2");
    panel.tonic.setSelectedId(3,juce::dontSendNotification);panel.addSection.onClick();panel.refresh();
    check("GUI appends next segment with independently chosen key",panel.manualSections.size()==2&&panel.manualSections[1].startBar==2&&panel.manualSections[1].tonic==2&&panel.generate.isEnabled());
    panel.sectionList.selectRow(0);panel.barCount.setText("2",false);panel.updateSection.onClick();panel.refresh();
    check("GUI overlap disables generate",!panel.generate.isEnabled());
    panel.barCount.setText("1",false);panel.updateSection.onClick();panel.refresh();
    check("GUI updating selected segment resolves overlap",panel.generate.isEnabled());
    panel.sectionList.selectRow(0);panel.removeSection.onClick();check("GUI deletes chosen segment",panel.manualSections.size()==1&&panel.manualSections[0].startBar==2);
    panel.startBar.setText("4",false);panel.barCount.setText("8",false);panel.scale.setSelectedId(2,juce::dontSendNotification);panel.tonic.setSelectedId(4,juce::dontSendNotification);panel.addSection.onClick();panel.refresh();
    image=panel.createComponentSnapshot(panel.getLocalBounds(),true,1.0f);
    if(auto out=directory.getChildFile("hamood-manual-sections.png").createOutputStream())check("render manual segment editor",png.writeImageToStream(image,*out));
    auto* report=new juce::DynamicObject();report->setProperty("ok",passed);report->setProperty("checks",checks);
    directory.getChildFile("report.json").replaceWithText(juce::JSON::toString(juce::var(report)));
    return passed;
}

}

#include "tests/HamoodPersistenceSmoke.h"
