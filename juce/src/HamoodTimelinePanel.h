#pragma once
#include "I18n.h"
#include "HamoodProject.h"
#include "HamoodAudio.h"
#include "Theme.h"
namespace hachi
{
class HamoodTimelinePanel final : public juce::Component, private juce::ListBoxModel
{
public:
    HamoodTimelinePanel(ProjectData project,juce::var stored,juce::File cache,std::function<bool(const juce::String&)> save)
        : data(std::move(project)),state(juce::JSON::parse(hamoodstate::encode(stored))),cacheFolder(cache),persist(std::move(save))
    {
        using namespace hamoodstate;
        title.setText(I18n::uiText("HAMOOD · 和弦与段落资料"),juce::dontSendNotification);
        help.setText(I18n::uiText("以工程秒数编辑，按拍点保存并跟随曲速。添加、更新和删除立即保存到工程；已确认和弦不会被分析覆盖。"),juce::dontSendNotification);
        kind.addItem(I18n::uiText("和弦"),1);kind.addItem(I18n::uiText("段落"),2);kind.setSelectedId(1,juce::dontSendNotification);
        kind.onChange=[this]{list.deselectAllRows();list.updateContent();updateHelp();};
        for(const auto& track:data.tracks)for(const auto& clip:track.clips)if(hamoodaudio::audioClip(data,clip.id))
        {audioIds.push_back(clip.id);source.addItem(track.name+" / "+clip.sourceFile.getFileName(),(int)audioIds.size());if(clip.id==state["settings"]["audio_clip_id"].toString())source.setSelectedId((int)audioIds.size(),juce::dontSendNotification);}
        if(!source.getSelectedId()&&!audioIds.empty())source.setSelectedId(1,juce::dontSendNotification);
        analyse.setButtonText(I18n::uiText("读取 / 分析伴奏"));analyse.setEnabled(!audioIds.empty());
        analyse.onClick=[this]{analyseAudio();};
        startLabel.setText(I18n::uiText("开始（秒）"),juce::dontSendNotification);endLabel.setText(I18n::uiText("结束（秒）"),juce::dontSendNotification);
        nameLabel.setText(I18n::uiText("和弦 / 段落名称"),juce::dontSendNotification);
        start.setText("0",false);end.setText("2",false);name.setText("C",false);
        confirmed.setButtonText(I18n::uiText("已人工确认"));confirmed.setToggleState(true,juce::dontSendNotification);
        add.setButtonText(I18n::uiText("添加"));update.setButtonText(I18n::uiText("更新所选"));remove.setButtonText(I18n::uiText("删除所选"));close.setButtonText(I18n::uiText("关闭"));
        add.onClick=[this]{writeRow(false);};update.onClick=[this]{writeRow(true);};remove.onClick=[this]{const auto row=list.getSelectedRow();if(row<0)return;auto next=juce::JSON::parse(hamoodstate::encode(state));next[field()].getArray()->remove(row);if(store(next)){list.deselectAllRows();list.updateContent();}};
        close.onClick=[this]{if(auto* w=findParentComponentOfClass<juce::DialogWindow>())w->exitModalState(0);};
        list.setModel(this);list.setRowHeight(28);
        for(auto* c:std::initializer_list<juce::Component*>{&title,&help,&kind,&source,&analyse,&startLabel,&endLabel,&nameLabel,&start,&end,&name,&confirmed,&add,&update,&remove,&close,&list,&message})addAndMakeVisible(*c);
        updateHelp();setSize(850,500);
    }
    ~HamoodTimelinePanel() override {cancel->store(true);}
    const char* field()const{return kind.getSelectedId()==2?"sections":"chords";}
    int getNumRows() override{return state[field()].getArray()->size();}
    void updateHelp(){message.setText(juce::String::fromUTF8(kind.getSelectedId()==1?"和弦示例：C、Am、Bbmaj7、G7/B；分数为模型指标，不是概率。":"段落示例：主歌、副歌、桥段。保存名称、范围和人工确认状态。"),juce::dontSendNotification);}
    void paintListBoxItem(int row,juce::Graphics& g,int width,int height,bool selected)override
    {
        if(row<0||row>=getNumRows())return;if(selected)g.fillAll(Palette::selection);const auto v=(*state[field()].getArray())[row];
        auto text=juce::String(data.secondsForQuarterPosition((double)v["start_quarter"]),3)+" – "+juce::String(data.secondsForQuarterPosition((double)v["end_quarter"]),3)+" s   |   "+v["label"].toString();
        if(kind.getSelectedId()==1)text+="   |   "+juce::String((double)v.getProperty("score",1.0),2);
        text+=juce::String::fromUTF8((bool)v["confirmed"]?"   已确认":"   待确认");g.setColour(selected?Palette::contrastingText(Palette::selection):Palette::text);g.drawText(text,8,0,width-16,height,juce::Justification::centredLeft);
    }
    void selectedRowsChanged(int row)override
    {
        if(row<0||row>=getNumRows())return;const auto v=(*state[field()].getArray())[row];
        start.setText(juce::String(data.secondsForQuarterPosition((double)v["start_quarter"]),6),false);end.setText(juce::String(data.secondsForQuarterPosition((double)v["end_quarter"]),6),false);
        name.setText(v["label"].toString(),false);confirmed.setToggleState((bool)v["confirmed"],juce::dontSendNotification);
    }
    bool store(juce::var next)
    {
        juce::String error;if(!hamoodstate::validate(next,error)){message.setText(error,juce::dontSendNotification);return false;}
        if(persist&&!persist(hamoodstate::encode(next))){message.setText(I18n::uiText("保存失败：工程已变化，请重新打开窗口。"),juce::dontSendNotification);return false;}
        state=std::move(next);message.setText(I18n::uiText("资料已记入工程；保存工程后重开仍可使用。"),juce::dontSendNotification);return true;
    }
    void writeRow(bool replace)
    {
        using namespace hamoodstate;const int selected=list.getSelectedRow();if(replace&&(selected<0||selected>=getNumRows()))return;
        const auto numeric=[](juce::String text,double& value){try{size_t count=0;const auto s=text.trim().toStdString();value=std::stod(s,&count);return count==s.size()&&std::isfinite(value);}catch(...){return false;}};
        double a=0,b=0;if(!numeric(start.getText(),a)||!numeric(end.getText(),b)||b<=a||name.getText().trim().isEmpty()){message.setText(I18n::uiText("请输入有效起止秒数和名称。"),juce::dontSendNotification);return;}
        auto next=juce::JSON::parse(encode(state));auto row=replace?(*next[field()].getArray())[selected]:object();
        put(row,"id",row.getProperty("id",juce::Uuid().toString()));put(row,"start_quarter",data.quarterPositionForSeconds(a));put(row,"end_quarter",data.quarterPositionForSeconds(b));put(row,"label",name.getText().trim());put(row,"confirmed",confirmed.getToggleState());
        if(kind.getSelectedId()==1)
        {
            auto pcs=chordPitches(name.getText());
            // Preserve nonstandard model labels when only changing timing or confirmation.
            if(pcs.empty()&&replace&&name.getText().trim()==(*state[field()].getArray())[selected]["label"].toString())for(const auto& p:*row["pitch_classes"].getArray())pcs.push_back((int)p);
            if(pcs.empty()){message.setText(I18n::uiText("无法识别和弦，请用 C、Am、C7、Bbmaj7、Dm7b5 等写法。"),juce::dontSendNotification);return;}
            juce::Array<juce::var> values;for(auto p:pcs)values.add(p);put(row,"pitch_classes",values);put(row,"score",1.0);auto settings=next["settings"];put(settings,"use_chords",true);put(next,"settings",settings);
        }
        auto* rows=next[field()].getArray();if(replace)rows->set(selected,row);else rows->add(row);
        std::sort(rows->begin(),rows->end(),[](const auto& x,const auto& y){return (double)x["start_quarter"]<(double)y["start_quarter"];});
        if(store(next)){list.deselectAllRows();list.updateContent();}
    }
    void analyseAudio()
    {
        const int index=source.getSelectedItemIndex();if(busy||index<0)return;busy=true;analyse.setEnabled(false);
        message.setText(I18n::uiText("正在读取或分析伴奏，已确认的资料将保留…"),juce::dontSendNotification);
        const auto clipId=audioIds[(size_t)index];const auto snapshot=data;const auto folder=cacheFolder;const auto cancellation=cancel;
        juce::Component::SafePointer<HamoodTimelinePanel> safe(this);
        juce::Thread::launch([safe,snapshot,folder,clipId,cancellation]{
            juce::var result;
            try{result=hamoodaudio::analyse(snapshot,clipId,folder,false,cancellation);}
            catch(const std::exception& e){result=hamoodaudio::failure(juce::String::fromUTF8(e.what()));}
            catch(...){result=hamoodaudio::failure("Audio analysis failed");}
            juce::MessageManager::callAsync([safe,result,clipId]{
                if(!safe)return;safe->busy=false;safe->analyse.setEnabled(true);
                if(!(bool)result["ok"]){safe->message.setText(result["error"].toString(),juce::dontSendNotification);return;}
                hamood::Options options;juce::String error;
                if(!hamoodaudio::attach(safe->data,clipId,safe->cacheFolder,options,error)){safe->message.setText(error,juce::dontSendNotification);return;}
                auto next=juce::JSON::parse(hamoodstate::encode(safe->state));hamoodstate::importChords(safe->data,next,options.chords,clipId);
                if(safe->store(next)){safe->kind.setSelectedId(1,juce::dontSendNotification);safe->list.deselectAllRows();safe->list.updateContent();}
            });
        });
    }
    void paint(juce::Graphics& g)override{g.fillAll(Palette::panel);}
    void resized()override
    {
        auto a=getLocalBounds().reduced(16);title.setBounds(a.removeFromTop(28));help.setBounds(a.removeFromTop(44));auto row=a.removeFromTop(32);
        kind.setBounds(row.removeFromLeft(100));row.removeFromLeft(10);analyse.setBounds(row.removeFromRight(160));row.removeFromRight(8);source.setBounds(row);a.removeFromTop(8);
        auto bottom=a.removeFromBottom(122);row=bottom.removeFromTop(26);startLabel.setBounds(row.removeFromLeft(115));endLabel.setBounds(row.removeFromLeft(115));nameLabel.setBounds(row);
        row=bottom.removeFromTop(30);start.setBounds(row.removeFromLeft(105));row.removeFromLeft(10);end.setBounds(row.removeFromLeft(105));row.removeFromLeft(10);name.setBounds(row.removeFromLeft(225));confirmed.setBounds(row);
        bottom.removeFromTop(8);row=bottom.removeFromTop(30);add.setBounds(row.removeFromLeft(90));update.setBounds(row.removeFromLeft(120));remove.setBounds(row.removeFromLeft(120));close.setBounds(row.removeFromRight(95));
        message.setBounds(bottom);a.removeFromBottom(8);list.setBounds(a);
    }
    ProjectData data;juce::var state;juce::File cacheFolder;std::function<bool(const juce::String&)> persist;
    juce::Label title,help,startLabel,endLabel,nameLabel,message;juce::ComboBox kind,source;juce::TextEditor start,end,name;juce::ToggleButton confirmed;juce::TextButton analyse,add,update,remove,close;juce::ListBox list;
    std::vector<juce::String> audioIds;bool busy=false;std::shared_ptr<std::atomic<bool>> cancel=std::make_shared<std::atomic<bool>>(false);
};
}