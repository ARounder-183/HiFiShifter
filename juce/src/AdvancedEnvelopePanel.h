#pragma once
#include "ProjectModel.h"
#include "Theme.h"
#include "ZoomScrollBar.h"
#include "PointGroupEdit.h"
#include "OtoRegionGuides.h"
#include "TailFadePresetStore.h"
#include "backend/AdvancedEnvelope.h"
#include "backend/UtauRenderer.h"
namespace hachi
{
// A draft editor: moving controls does not alter the document until Apply.
class AdvancedEnvelopePanel final : public juce::Component
{
public:
    struct PreviewNote
    {
        juce::String id, label;
        backend::UtauTailFadeSpan tail;
        std::vector<AmplitudeEnvelopePoint> base;
        std::vector<OtoRegionGuide> regions;
        std::optional<backend::UtauTailFadeSpan> head;
    };
    AdvancedEnvelopePanel(std::vector<PreviewNote> notes,int selectedCount,int initialMode,
                          backend::TailFadeSettings initial,bool mixed,juce::File presetFile = {},bool native = false,bool phoneme = false)
        : nativeMode(native||phoneme),phonemeMode(phoneme),presetStore(std::move(presetFile)),previews(std::move(notes)),graph(*this)
    {
        addAndMakeVisible(title);title.setText(text(phonemeMode?"音素高级包络":nativeMode?"音频包络编辑":"OTO 音头与音尾包络"),juce::dontSendNotification);title.setFont(juce::FontOptions(21.0f));
        addAndMakeVisible(summary);summary.setText(text("选中 ")+juce::String(selectedCount)+text(phonemeMode?" 个音素，可用 ":" 个音符，可用 ")+juce::String((int)previews.size())+text(" 个")
            +(mixed?text(" · 设置不同，应用时统一为当前参数"):juce::String()),juce::dontSendNotification);
        enabled.setButtonText(text("启用尾段淡出"));enabled.setToggleState(initialMode!=0,juce::dontSendNotification);addAndMakeVisible(enabled);
        shape.addItem(text("线性"),1);shape.addItem(text("曲线"),2);shape.addItem(text("混合"),3);shape.setSelectedId(initialMode==2?2:1,juce::dontSendNotification);addAndMakeVisible(shape);
        addAndMakeVisible(shapeLabel);shapeLabel.setText(text("曲线形状"),juce::dontSendNotification);
        for(size_t i=0;i<previews.size();++i)preview.addItem(previews[i].label+" · "+juce::String((int)i+1),(int)i+1);
        preview.setSelectedId(1,juce::dontSendNotification);addAndMakeVisible(preview);addAndMakeVisible(previewLabel);previewLabel.setText(text("预览音符"),juce::dontSendNotification);
        section.addItem(text("音尾 · 最后一区"),1);section.addItem(text("音头 · 第一区"),2);
        section.setSelectedId(1,juce::dontSendNotification);addAndMakeVisible(section);
        section.setVisible(!nativeMode);
        section.onChange=[this]{const auto v=values();const auto m=mode();const auto sh=tailShape();activeSection=section.getSelectedId()-1;setValues(m,v);draftTailShape=sh;if(!editingHead())shape.setSelectedId(sh,juce::dontSendNotification);refresh();};
        addAndMakeVisible(graph);
        addAndMakeVisible(segment);addAndMakeVisible(segmentType);
        segmentType.addItem(text("直线"),1);segmentType.addItem(text("贝塞尔曲线"),2);
        segment.onChange=[this]{selectedSegment=segment.getSelectedId()-1;refresh();};
        segmentType.onChange=[this]{setSegmentCurved(selectedSegment,segmentType.getSelectedId()==2);};
        const char* captions[]{"淡出起点 · 尾段 %","淡出终点 · 尾段 %","淡出前响度 · %","结束响度 · %","曲线强度"};
        for(size_t i=0;i<sliders.size();++i)
        {
            addAndMakeVisible(labels[i]);labels[i].setText(text(captions[i]),juce::dontSendNotification);
            addAndMakeVisible(sliders[i]);sliders[i].setSliderStyle(juce::Slider::LinearHorizontal);sliders[i].setTextBoxStyle(juce::Slider::TextBoxRight,false,94,27);
            sliders[i].setRange(i==4?.25:0.0,i<2?100.0:i<4?200.0:4.0,i==4?.05:.1);
            sliders[i].setTextValueSuffix(i==4?" x":" %");
            sliders[i].onValueChange=[this,i]{adjustLimits(i);refresh();};
        }
        sliders[4].setSkewFactorFromMidPoint(1.0);
        const char* curveCaptions[]{"控制点 1 · 时间 %","控制点 1 · 过渡 %","控制点 2 · 时间 %","控制点 2 · 过渡 %"};
        for(size_t i=0;i<curveControls.size();++i)
        {
            addAndMakeVisible(curveLabels[i]);curveLabels[i].setText(text(curveCaptions[i]),juce::dontSendNotification);
            addAndMakeVisible(curveControls[i]);auto& c=curveControls[i];c.setSliderStyle(juce::Slider::LinearHorizontal);
            c.setTextBoxStyle(juce::Slider::TextBoxRight,false,62,26);c.setRange(0,100);c.setNumDecimalPlacesToDisplay(1);c.setTextValueSuffix(" %");
            c.onValueChange=[this,i]{
                const auto other=i<2?i+2:i-2;
                const auto coordinates=selectedMixedCurve()?mixedCurveCoordinates():curveCoordinates;
                const auto limit=coordinates[other]*100;
                curveControls[i].setValue(i<2?std::min(limit,curveControls[i].getValue()):std::max(limit,curveControls[i].getValue()),juce::dontSendNotification);
                if(selectedMixedCurve())
                {
                    auto& b=selectedBezier();double* values[]{&b.control1Time,&b.control1Progress,&b.control2Time,&b.control2Progress};
                    *values[i]=curveControls[i].getValue()/100;
                }
                else {beginControlEditing();curveCoordinates[i]=curveControls[i].getValue()/100;}
                refresh();
            };
        }
        addAndMakeVisible(preset);preset.onChange=[this]{applyPreset(preset.getSelectedId());};
        savePresetButton.setButtonText(text("保存为预设…"));deletePresetButton.setButtonText(text("删除预设"));
        addAndMakeVisible(savePresetButton);addAndMakeVisible(deletePresetButton);
        savePresetButton.onClick=[this]{promptSavePreset();};deletePresetButton.onClick=[this]{deleteSelectedPreset();};


        addAndMakeVisible(detail);detail.setFont(juce::FontOptions(13.0f));
        addAndMakeVisible(help);help.setFont(juce::FontOptions(13.0f));help.setJustificationType(juce::Justification::topLeft);
        help.setText(text("位置按 OTO 尾段比例跟随音符；可拖动预览中的两端圆点。淡出从起点响度连续降至结束响度；原包络收尾由所选直线或曲线替代。曲线强度 < 1 先快后慢，> 1 先慢后快。"),juce::dontSendNotification);
        addAndMakeVisible(status);status.setFont(juce::FontOptions(13.0f));
        reset.setButtonText(text("恢复默认"));apply.setButtonText(text("应用"));listen.setButtonText(text("应用并试听"));close.setButtonText(text("关闭"));
        for(auto* button:{&reset,&apply,&listen,&close})addAndMakeVisible(button);
        reset.onClick=[this]{backend::TailFadeSettings v;if(phonemeMode&&!previews.empty())v=backend::cvvcEnvelopeSettings(backend::standardCvvcEnvelope(previews.front().tail.endSeconds)).shape;else if(nativeMode)v=backend::NativeEnvelopeSettings{}.shape;else if(editingHead())backend::copyEnvelopeShape(backend::HeadEnvelopeSettings{},v);setControls(nativeMode?2:1,v);status.setText(text(phonemeMode?"已恢复标准梯形草稿，点击应用生效。":nativeMode?"已恢复 100% 响度草稿，点击应用生效。":"已恢复默认草稿，点击应用生效。"),juce::dontSendNotification);};
        apply.onClick=[this]{commit(false);};listen.onClick=[this]{commit(true);};
        close.onClick=[this]{if(auto* window=findParentComponentOfClass<juce::DialogWindow>())window->exitModalState(0);};
        enabled.onClick=[this]{graph.clearSelection();refresh();};shape.onChange=[this]{graph.clearSelection();if(shape.getSelectedId()==2)beginControlEditing();adjustLimits(3);refresh();};preview.onChange=[this]{graph.clearSelection();refresh();};
        reloadPresets();setValues(initialMode,initial);enabled.setVisible(!phonemeMode);setSize(820,680);
    }
    std::function<juce::String(int,const backend::TailFadeSettings&,bool)> onApply;
    bool diagnosticPhonemeMode() const {return phonemeMode;}
    bool editingHead() const {return activeSection==1;}
    int editingMode() const {return enabled.getToggleState()?std::min(2,shape.getSelectedId()):0;}
    int mode() const {return editingHead()?draftTailMode:editingMode();}
    int tailShape() const {return editingHead()?draftTailShape:shape.getSelectedId();}
    backend::UtauTailFadeSpan editingSpan(const PreviewNote& n) const {return editingHead()?n.head.value_or(backend::UtauTailFadeSpan{}):n.tail;}
    backend::TailFadeSettings values() const
    {
        const auto controls=editingValues();auto result=editingHead()?draft:controls;
        if(editingHead()){backend::copyEnvelopeShape(controls,result.head);result.head.mode=editingMode();}
        else result.head=draft.head;
        return result;
    }
    void setValues(int mode,const backend::TailFadeSettings& v)
    {
        draft=v;draftTailMode=mode;draftTailShape=v.mixed?3:(v.customCurve||mode==2)?2:1;
        if(editingHead()){backend::TailFadeSettings h;backend::copyEnvelopeShape(v.head,h);setControls(v.head.mode,h);}
        else setControls(mode,v);
    }
    backend::TailFadeSettings editingValues() const
    {
        backend::TailFadeSettings v{sliders[0].getValue()/100,sliders[1].getValue()/100,sliders[2].getValue()/100,sliders[3].getValue()/100,sliders[4].getValue()};
        v.customCurve=shape.getSelectedId()==2&&bezierCurve;v.control1Time=curveCoordinates[0];v.control1Progress=curveCoordinates[1];
        v.control2Time=curveCoordinates[2];v.control2Progress=curveCoordinates[3];
        v.mixed=shape.getSelectedId()==3;v.firstSegmentCurved=firstSegmentCurved;v.knots=knots;v.firstBezier=firstBezier;v.allowRise=nativeMode;return v;
    }
    void setControls(int mode,const backend::TailFadeSettings& v)
    {
        graph.clearSelection();
        knots=v.knots;firstSegmentCurved=v.firstSegmentCurved;firstBezier=v.firstBezier;selectedSegment=0;
        bezierCurve=v.customCurve;enabled.setToggleState(mode!=0,juce::dontSendNotification);shape.setSelectedId(v.mixed?3:mode==1?1:(v.customCurve||mode==2)?2:1,juce::dontSendNotification);
        const double x[]{v.startFraction*100,v.endFraction*100,v.startGain*100,v.endGain*100,v.curvePower};
        for(size_t i=0;i<sliders.size();++i)sliders[i].setValue(x[i],juce::dontSendNotification);
        curveCoordinates={v.control1Time,v.control1Progress,v.control2Time,v.control2Progress};
        const auto& controls=curveCoordinates;
        for(size_t i=0;i<curveControls.size();++i)curveControls[i].setValue(controls[i]*100,juce::dontSendNotification);refresh();
    }
    void diagnosticSection(bool head) {section.setSelectedId(head?2:1,juce::sendNotificationSync);}
    int diagnosticShapeCount() const {return shape.getNumItems();}
    void diagnosticSelectShape(int id) {shape.setSelectedId(id,juce::sendNotificationSync);}
    void diagnosticSetControl(size_t i,double value) {sliders.at(i).setValue(value,juce::sendNotificationSync);}
    bool diagnosticSavePreset(const juce::String& name) {return saveCurrentPreset(name);}
    void diagnosticDeletePreset() {deleteSelectedPreset();}
    size_t diagnosticPresetCount() const {return userPresets.size();}
    void diagnosticPreset(int id) {applyPreset(id);}
    void diagnosticCurveControl(size_t i,double value) {curveControls.at(i).setValue(value,juce::sendNotificationSync);}
    void diagnosticDragCurveHandle(int index,float dx,float dy) {graph.dragControl(index,graph.controlPoint(index)+juce::Point<float>{dx,dy});}
    void diagnosticApply(bool audition=false) {commit(audition);}
    void diagnosticAddKnot(double time,double gain) {addKnot(time,gain);}
    void diagnosticMoveKnot(int index,double time,double gain) {moveKnot(index,time,gain);}
    void diagnosticDeleteKnot(int index) {deleteKnot(index);}
    void diagnosticSegment(int index,bool curved) {setSegmentCurved(index,curved);}
    void diagnosticSelectPreview(int index) {graph.clearSelection();preview.setSelectedId(index+1,juce::dontSendNotification);refresh();}
    std::vector<OtoRegionGuide> diagnosticPreviewRegions() const {return current()?current()->regions:std::vector<OtoRegionGuide>{};}
    bool diagnosticControlsFit() const
    {
        for(const auto& slider:sliders)if(slider.isVisible()&&(!getLocalBounds().contains(slider.getBounds())||slider.getWidth()<250))return false;
        for(const auto& c:curveControls)if(c.isVisible()&&(!getLocalBounds().contains(c.getBounds())||c.getWidth()<150))return false;
        return graph.getHeight()>=140&&help.getHeight()>=36&&close.getHeight()>=30&&getLocalBounds().contains(close.getBounds());
    }
    void paint(juce::Graphics& g) override {g.fillAll(Palette::panel);}
    void resized() override
    {
        const auto mixedCurve=selectedMixedCurve();
        auto area=getLocalBounds().reduced(20);
        auto heading=area.removeFromTop(30);section.setBounds(heading.removeFromRight(190));title.setBounds(heading);summary.setBounds(area.removeFromTop(26));area.removeFromTop(8);
        auto row=area.removeFromTop(30);enabled.setBounds(row.removeFromLeft(180));shapeLabel.setBounds(row.removeFromLeft(90));shape.setBounds(row.removeFromLeft(150));row.removeFromLeft(20);previewLabel.setBounds(row.removeFromLeft(82));preview.setBounds(row);
        area.removeFromTop(6);auto presetRow=area.removeFromTop(28);deletePresetButton.setBounds(presetRow.removeFromRight(90));presetRow.removeFromRight(8);savePresetButton.setBounds(presetRow.removeFromRight(116));presetRow.removeFromRight(8);preset.setBounds(presetRow);
        area.removeFromTop(4);graph.setBounds(area.removeFromTop(std::max(140,area.getHeight()-(mixedCurve?348:shape.getSelectedId()==2?(bezierCurve?344:366):310))));area.removeFromTop(6);
        detail.setBounds(area.removeFromTop(24));
        for(size_t i=0;i<(shape.getSelectedId()==3||(shape.getSelectedId()==2&&bezierCurve)?4:5);++i){auto control=area.removeFromTop(mixedCurve||(shape.getSelectedId()==2&&!bezierCurve)?30:34);labels[i].setBounds(control.removeFromLeft(175));sliders[i].setBounds(control);}
        if(shape.getSelectedId()==3){auto row=area.removeFromTop(mixedCurve?30:34);segmentType.setBounds(row.removeFromRight(150));row.removeFromRight(10);segment.setBounds(row);}
        if(shape.getSelectedId()==2||mixedCurve)for(size_t i=0;i<4;i+=2){auto row=area.removeFromTop(mixedCurve?28:bezierCurve?34:28);auto left=row.removeFromLeft(row.getWidth()/2);row.removeFromLeft(10);
            curveLabels[i].setBounds(left.removeFromLeft(130));curveControls[i].setBounds(left);
            curveLabels[i+1].setBounds(row.removeFromLeft(130));curveControls[i+1].setBounds(row);}
        area.removeFromTop(4);help.setBounds(area.removeFromTop(mixedCurve?36:40));status.setBounds(area.removeFromTop(mixedCurve?22:26));area.removeFromTop(6);
        auto buttons=area.removeFromTop(34);reset.setBounds(buttons.removeFromLeft(116));close.setBounds(buttons.removeFromRight(94));buttons.removeFromRight(8);apply.setBounds(buttons.removeFromRight(94));buttons.removeFromRight(8);listen.setBounds(buttons.removeFromRight(144));
    }
private:
    friend class MainComponent; // Existing panel smoke checks exercise real graph gestures.
    static juce::String text(const char* s){return juce::String::fromUTF8(s);}
    const PreviewNote* current() const {const auto i=preview.getSelectedId()-1;return i>=0&&i<(int)previews.size()?&previews[(size_t)i]:nullptr;}
    void adjustLimits(size_t changed)
    {
        if(changed==0&&sliders[0].getValue()>sliders[1].getValue()-.2)sliders[0].setValue(sliders[1].getValue()-.2,juce::dontSendNotification);
        if(changed==1&&sliders[1].getValue()<sliders[0].getValue()+.2)sliders[1].setValue(sliders[0].getValue()+.2,juce::dontSendNotification);
        if(!nativeMode&&!editingHead()&&shape.getSelectedId()!=3&&changed==2&&sliders[3].getValue()>sliders[2].getValue())sliders[3].setValue(sliders[2].getValue(),juce::dontSendNotification);
        if(!nativeMode&&!editingHead()&&shape.getSelectedId()!=3&&changed==3&&sliders[3].getValue()>sliders[2].getValue())sliders[3].setValue(sliders[2].getValue(),juce::dontSendNotification);
    }
    void beginControlEditing()
    {
        // Keep legacy powered S curves intact until the user edits their handles.
        // New curves and handle edits share the Bezier editor, without a second mode.
        if(!bezierCurve){bezierCurve=true;sliders[4].setValue(1,juce::dontSendNotification);}
    }
    bool selectedMixedCurve() const
    {
        return shape.getSelectedId()==3&&selectedSegment>=0&&selectedSegment<=(int)knots.size()
            &&(selectedSegment==0?firstSegmentCurved:knots[(size_t)selectedSegment-1].curvedToNext);
    }
    backend::EnvelopeBezier& selectedBezier()
    {return selectedSegment<=0?firstBezier:knots.at((size_t)selectedSegment-1).bezier;}
    std::array<double,4> mixedCurveCoordinates() const
    {
        const auto& b=selectedSegment<=0?firstBezier:knots.at((size_t)selectedSegment-1).bezier;
        return {b.control1Time,b.control1Progress,b.control2Time,b.control2Progress};
    }
    void addKnot(double time,double gain)
    {
        if(shape.getSelectedId()!=3||knots.size()>=64||time<.001||time>.999)return;
        const auto at=std::lower_bound(knots.begin(),knots.end(),time,[](const auto& p,double t){return p.time<t;});
        const auto i=(size_t)std::distance(knots.begin(),at);
        if((i>0&&time-knots[i-1].time<.002)||(i<knots.size()&&knots[i].time-time<.002))return;
        const auto curved=i==0?firstSegmentCurved:knots[i-1].curvedToNext;
        const auto bezier=i==0?firstBezier:knots[i-1].bezier;
        graph.clearSelection();knots.insert(at,{time,std::clamp(gain,0.0,2.0),curved,bezier});selectedSegment=(int)i;refresh();
    }
    void moveKnot(int index,double time,double gain)
    {
        gain=std::clamp(gain,0.0,2.0);
        if(index==0||index==(int)knots.size()+1){sliders[index==0?2:3].setValue(gain*100,juce::sendNotificationSync);return;}
        if(index<1||index>(int)knots.size())return;
        const auto i=(size_t)index-1;
        knots[i].time=std::clamp(time,i==0?.002:knots[i-1].time+.002,i+1==knots.size()?.998:knots[i+1].time-.002);
        knots[i].gain=gain;selectedSegment=index;refresh();
    }
    void deleteKnot(int index)
    {
        editKnots({index},false);
    }
    void editKnots(std::vector<int> indices,bool initialize)
    {
        std::sort(indices.begin(),indices.end(),std::greater<int>());
        indices.erase(std::unique(indices.begin(),indices.end()),indices.end());
        graph.clearSelection();
        if(initialize)for(const int index:indices)
            if(index==0||index==(int)knots.size()+1)sliders[index==0?2:3].setValue(100,juce::dontSendNotification);
        for(const auto index:indices)if(index>0&&index<=(int)knots.size())
        {
            if(initialize)knots[(size_t)index-1].gain=1;
            else knots.erase(knots.begin()+index-1);
        }
        if(initialize){graph.selectedKnots=indices;graph.selectedKnot=indices.empty()?-1:indices.front();}
        selectedSegment=std::min(selectedSegment,(int)knots.size());refresh();
    }
    void setSegmentCurved(int index,bool curved)
    {
        if(index<0||index>(int)knots.size())return;
        graph.clearSelection();
        if(index==0)firstSegmentCurved=curved;else knots[(size_t)index-1].curvedToNext=curved;
        selectedSegment=index;refresh();
    }
    void applyPreset(int id)
    {
        if(id>=100)
        {
            const auto index=(size_t)(id-100);if(index>=userPresets.size())return;const auto p=userPresets[index];
            setValues(p.enabled?(p.shape==1?1:2):0,p.settings);draftTailShape=p.shape==4?3:std::min(2,p.shape);if(!editingHead())shape.setSelectedId(draftTailShape,juce::dontSendNotification);
            preset.setSelectedId(id,juce::dontSendNotification);refresh();
            status.setText(text("已载入预设：")+p.name+text("。点击应用生效。"),juce::dontSendNotification);return;
        }
        if(id<2)return;auto v=editingValues();v.mixed=false;v.customCurve=true;v.curvePower=1;
        v.control1Time=1.0/3;v.control2Time=2.0/3;
        v.control1Progress=id==3?.75:id==5?1.0/3:0;
        v.control2Progress=id==4?.25:id==5?2.0/3:1;
        setControls(2,v);preset.setSelectedId(1,juce::dontSendNotification);refresh();
    }
    void reloadPresets(const juce::String& selected = {})
    {
        juce::String error;std::vector<TailFadePreset> loaded;
        if(!presetStore.load(loaded,error))status.setText(error,juce::dontSendNotification);
        userPresets=std::move(loaded);preset.clear(juce::dontSendNotification);
        preset.addItem(text("选择预设…"),1);preset.addSectionHeading(text("内置曲线"));
        preset.addItem(text("平滑 S 曲线"),2);preset.addItem(text("先快后慢"),3);preset.addItem(text("先慢后快"),4);preset.addItem(text("均匀淡出"),5);
        int selectedId=1;if(!userPresets.empty())preset.addSectionHeading(text("自定义预设"));
        for(size_t i=0;i<userPresets.size();++i){preset.addItem(userPresets[i].name,(int)i+100);if(userPresets[i].id==selected)selectedId=(int)i+100;}
        preset.setSelectedId(selectedId,juce::dontSendNotification);
    }
    bool saveCurrentPreset(const juce::String& name)
    {
        TailFadePreset p;p.name=name;p.shape=values().mixed?4:values().customCurve?3:tailShape();p.enabled=mode()!=0;p.settings=values();juce::String error;
        if(!presetStore.add(p,error)){status.setText(error,juce::dontSendNotification);return false;}
        reloadPresets(p.id);refresh();status.setText(text("已保存预设：")+p.name,juce::dontSendNotification);return true;
    }
    void promptSavePreset()
    {
        auto* dialog=new juce::AlertWindow(text("保存自定义预设"),text(nativeMode?"保存音频包络的分段、贝塞尔控制柄、范围、起止响度和启用状态。":"同时保存音头和音尾的曲线、范围、起止响度和启用状态。"),juce::MessageBoxIconType::NoIcon);
        dialog->addTextEditor("name","",text("预设名称"));dialog->getTextEditor("name")->setInputRestrictions(64);
        dialog->addButton(text("保存"),1,juce::KeyPress(juce::KeyPress::returnKey));dialog->addButton(text("取消"),0,juce::KeyPress(juce::KeyPress::escapeKey));
        juce::Component::SafePointer<AdvancedEnvelopePanel> safe(this);
        dialog->enterModalState(true,juce::ModalCallbackFunction::create([safe,dialog](int result){if(safe&&result==1)safe->saveCurrentPreset(dialog->getTextEditorContents("name"));delete dialog;}),false);
    }
    void deleteSelectedPreset()
    {
        const auto index=preset.getSelectedId()-100;if(index<0||index>=(int)userPresets.size())return;
        const auto p=userPresets[(size_t)index];juce::String error;
        if(!presetStore.remove(p.id,error)){status.setText(error,juce::dontSendNotification);return;}
        reloadPresets();refresh();status.setText(text("已删除预设：")+p.name+text("。当前草稿保留。"),juce::dontSendNotification);
    }
    void refresh()
    {
        const auto active=editingMode()!=0;for(auto& slider:sliders)slider.setEnabled(active);shape.setEnabled(active);sliders[4].setEnabled(active&&editingMode()==2&&!editingValues().customCurve);
        const bool custom=editingValues().customCurve;const bool curved=shape.getSelectedId()==2;const bool mixed=shape.getSelectedId()==3;
        sliders[4].setVisible(!custom&&!mixed);labels[4].setVisible(!custom&&!mixed);
        segment.setVisible(mixed);segmentType.setVisible(mixed);segment.setEnabled(active);segmentType.setEnabled(active);
        if(mixed)
        {
            selectedSegment=std::clamp(selectedSegment,0,(int)knots.size());segment.clear(juce::dontSendNotification);
            for(size_t i=0;i<=knots.size();++i)segment.addItem(text("分段 ")+juce::String((int)i+1)+"   "+juce::String((i==0?0:knots[i-1].time)*100,1)+"% – "+juce::String((i==knots.size()?1:knots[i].time)*100,1)+"%",(int)i+1);
            segment.setSelectedId(selectedSegment+1,juce::dontSendNotification);
            segmentType.setSelectedId((selectedSegment==0?firstSegmentCurved:knots[(size_t)selectedSegment-1].curvedToNext)?2:1,juce::dontSendNotification);
        }
        const auto coordinates=selectedMixedCurve()?mixedCurveCoordinates():curveCoordinates;
        for(size_t i=0;i<4;++i){curveControls[i].setVisible(curved||selectedMixedCurve());curveLabels[i].setVisible(curved||selectedMixedCurve());curveControls[i].setEnabled(active);curveControls[i].setValue(coordinates[i]*100,juce::dontSendNotification);}
        preset.setEnabled(true);savePresetButton.setEnabled(presetStore.available());deletePresetButton.setEnabled(preset.getSelectedId()>=100);
        help.setText(text(custom?"拖动方形控制柄改变曲线；时间决定横向位置，过渡决定响度变化进度。按住 Shift 精细拖动，双击控制柄恢复默认。圆点调整起止位置。控制点保持顺序，避免响度反弹。":"圆点调整起止位置。曲线模式可拖动控制柄或输入参数；已有 S 曲线保留原强度，调整控制点后改用控制点定义曲线。"),juce::dontSendNotification);
        enabled.setButtonText(text(editingHead()?"启用音头包络":"启用尾段淡出"));
        if(nativeMode)enabled.setButtonText(text(phonemeMode?"启用音素包络":"启用音频包络"));
        const char* headLabels[]{"包络起点 · 首区 %","包络终点 · 首区 %","起点响度 · %","结束响度 · %"};
        const char* tailLabels[]{"淡出起点 · 尾段 %","淡出终点 · 尾段 %","淡出前响度 · %","结束响度 · %"};
        for(size_t i=0;i<4;++i)labels[i].setText(text(editingHead()?headLabels[i]:tailLabels[i]),juce::dontSendNotification);
        if(nativeMode){const char* names[]{phonemeMode?"包络起点 · 音素 %":"包络起点 · 音频 %",phonemeMode?"包络终点 · 音素 %":"包络终点 · 音频 %","起点响度 · %","结束响度 · %"};for(size_t i=0;i<4;++i)labels[i].setText(text(names[i]),juce::dontSendNotification);}
        if(phonemeMode&&!mixed)help.setText(text("仅修改当前音素的包络。位置按音素发声范围计算；支持直线、曲线和混合分段。点击应用后生效，可撤销。"),juce::dontSendNotification);
        if(editingHead())help.setText(text("音头按 OTO 第一区定位，替代原包络起音；结束响度作为后续原包络的倍率。可拖动圆点和曲线控制柄。两侧独立启用，预设同时保存两侧设置。"),juce::dontSendNotification);
        if(mixed)help.setText(text(selectedMixedCurve()?"当前分段使用贝塞尔：拖动方形控制柄或输入时间／过渡参数，双击控制柄重置。Shift 精细拖动；双击空白加点，选中后 Del 或右键删点。各段控制柄独立保存。":"双击范围内加点，拖动节点调整时间／响度；Shift 精细拖动。选择分段后可切换直线／贝塞尔曲线；选中后 Del 或右键删点，顶部圆点调整整体范围。"),juce::dontSendNotification);
        resized();
        if(const auto* n=current())
        {
            const auto v=editingValues();const auto span=editingSpan(*n);const auto length=span.endSeconds-span.startSeconds;
            detail.setText(length<=1.e-9?text("当前音符没有有效的此区域，渲染时跳过此侧包络。"):text(phonemeMode?"音素范围：":nativeMode?"音频范围：":editingHead()?"OTO 音头：":"OTO 尾段：")+juce::String(span.startSeconds*1000,1)+" – "+juce::String(span.endSeconds*1000,1)+text(" ms  ·  包络：")
                +juce::String((span.startSeconds+length*v.startFraction)*1000,1)+" – "+juce::String((span.startSeconds+length*v.endFraction)*1000,1)+text(" ms（相对音符起点）"),juce::dontSendNotification);
        }
        graph.repaint();
    }
    void commit(bool audition)
    {
        if(!values().valid()){status.setText(text("请检查范围和响度：终点应晚于起点，结束响度不大于淡出前响度。"),juce::dontSendNotification);return;}
        if(onApply)status.setText(onApply(mode(),values(),audition),juce::dontSendNotification);
    }
    struct Graph final : juce::Component, public juce::SettableTooltipClient, private juce::ScrollBar::Listener
    {
        explicit Graph(AdvancedEnvelopePanel& p):owner(p)
        {
            setWantsKeyboardFocus(true);
            owner.addMouseListener(&selectionListener,true);
            setTooltip(text("滚轮：上下滚动；Shift + 滚轮：左右滚动；Ctrl + 滚轮：纵轴缩放；Ctrl + Shift + 滚轮：横轴缩放（以鼠标位置为中心）。"));
            for(auto* button:{&xMinus,&xPlus,&yMinus,&yPlus,&fit})addAndMakeVisible(button);
            addAndMakeVisible(xLabel);addAndMakeVisible(yLabel);
            xMinus.setButtonText(text("横轴 −"));xPlus.setButtonText("+");
            yMinus.setButtonText(text("纵轴 −"));yPlus.setButtonText("+");fit.setButtonText(text("适应全图"));
            xMinus.onClick=[this]{setZoom(zoomX/2,zoomY);};xPlus.onClick=[this]{setZoom(zoomX*2,zoomY);};
            yMinus.onClick=[this]{setZoom(zoomX,zoomY/2);};yPlus.onClick=[this]{setZoom(zoomX,zoomY*2);};
            fit.onClick=[this]{setZoom(1,1);};
            for(auto* bar:{&horizontal,&vertical}){addAndMakeVisible(bar);bar->setRangeLimits(0,1);bar->addListener(this);}
            updateView();
        }
        ~Graph() override {owner.removeMouseListener(&selectionListener);horizontal.removeListener(this);vertical.removeListener(this);}
        void clearSelection() {selectedKnot=-1;selectedKnots.clear();repaint();}
        void focusLost(FocusChangeType) override {clearSelection();}
        bool keyPressed(const juce::KeyPress& key) override
        {
            const auto initialize=key.getModifiers().isCtrlDown()&&(key.getKeyCode()=='R'||key.getKeyCode()=='r');
            if(!initialize&&!key.isKeyCode(juce::KeyPress::deleteKey)&&!key.isKeyCode(juce::KeyPress::backspaceKey))return false;
            const auto indices=selectedKnots;
            clearSelection();dragged=draggedControl=draggedKnot=-1;
            if(mixed()&&owner.editingMode()!=0)owner.editKnots(indices,initialize);
            // Consume repeats and endpoint presses inside this draft editor.
            return true;
        }
        void resized() override
        {
            auto row=getLocalBounds().removeFromTop(26).reduced(4,1);
            xMinus.setBounds(row.removeFromLeft(66));xLabel.setBounds(row.removeFromLeft(58));xPlus.setBounds(row.removeFromLeft(28));row.removeFromLeft(16);
            yMinus.setBounds(row.removeFromLeft(66));yLabel.setBounds(row.removeFromLeft(58));yPlus.setBounds(row.removeFromLeft(28));row.removeFromLeft(16);fit.setBounds(row.removeFromLeft(86));
            const auto bounds=plot().toNearestInt();horizontal.setBounds(bounds.getX(),getHeight()-11,bounds.getWidth(),11);
            vertical.setBounds(getWidth()-12,bounds.getY(),11,bounds.getHeight());
        }
        juce::Rectangle<float> plot() const {return getLocalBounds().toFloat().withTrimmedTop(34).withTrimmedBottom(33).withTrimmedLeft(52).withTrimmedRight(20);}
        double first(const PreviewNote& n) const {return std::min(n.base.empty()?0.0:n.base.front().timeSeconds,n.tail.startSeconds);}
        double last(const PreviewNote& n) const {return std::max(n.base.empty()?n.tail.endSeconds:n.base.back().timeSeconds,n.tail.endSeconds);}
        double visibleFirst(const PreviewNote& n) const {return first(n)+(last(n)-first(n))*viewX;}
        double visibleLast(const PreviewNote& n) const {return visibleFirst(n)+(last(n)-first(n))/zoomX;}
        float x(double time,const PreviewNote& n) const {return plot().getX()+(float)((time-visibleFirst(n))/std::max(1.e-9,visibleLast(n)-visibleFirst(n)))*plot().getWidth();}
        double timeAt(float px,const PreviewNote& n) const {return visibleFirst(n)+(px-plot().getX())/plot().getWidth()*(visibleLast(n)-visibleFirst(n));}
        float y(float gain) const {return plot().getY()+(1.0f-gain/ceiling()-(float)viewY)*(float)zoomY*plot().getHeight();}
        float gainAtY(float py) const {return ceiling()*(1.0f-(float)viewY-(py-plot().getY())/(plot().getHeight()*(float)zoomY));}
        void setZoom(double horizontalZoom,double verticalZoom,juce::Point<double> anchor={0.5,0.5})
        {
            const auto cx=viewX+anchor.x/zoomX,cy=viewY+anchor.y/zoomY;
            zoomX=juce::jlimit(1.0,32.0,horizontalZoom);zoomY=juce::jlimit(1.0,16.0,verticalZoom);
            viewX=cx-anchor.x/zoomX;viewY=cy-anchor.y/zoomY;updateView();
        }
        void mouseWheelMove(const juce::MouseEvent& event,const juce::MouseWheelDetails& wheel) override
        {
            if(!plot().contains(event.position)||dragged>=0||draggedControl>=0||draggedKnot>=0)return;
            const auto action=editorWheelActionFor(event.mods);
            if(action==EditorWheelAction::zoomHorizontally||action==EditorWheelAction::zoomVertically)
            {
                const auto delta=wheel.isReversed?-wheel.deltaY:wheel.deltaY;
                if(delta==0)return;
                const auto factor=std::exp(static_cast<double>(delta)*1.35);
                const juce::Point<double> anchor{(event.position.x-plot().getX())/plot().getWidth(),
                                                 (event.position.y-plot().getY())/plot().getHeight()};
                setZoom(action==EditorWheelAction::zoomHorizontally?zoomX*factor:zoomX,
                        action==EditorWheelAction::zoomVertically?zoomY*factor:zoomY,anchor);
                return;
            }
            if(action==EditorWheelAction::scrollHorizontally)
            {
                auto sideways=wheel;sideways.deltaX=wheel.deltaY!=0?wheel.deltaY:wheel.deltaX;sideways.deltaY=0;
                horizontal.mouseWheelMove(event,sideways);
            }
            else
            {
                vertical.mouseWheelMove(event,wheel);
                horizontal.mouseWheelMove(event,wheel); // Horizontal trackpad gestures.
            }
            viewX=horizontal.getCurrentRangeStart();viewY=vertical.getCurrentRangeStart();updateView();
        }
        void updateView()
        {
            viewX=juce::jlimit(0.0,1.0-1.0/zoomX,viewX);viewY=juce::jlimit(0.0,1.0-1.0/zoomY,viewY);
            horizontal.setSingleStepSize(0.1/zoomX);vertical.setSingleStepSize(0.1/zoomY);
            horizontal.setCurrentRange(viewX,1.0/zoomX,juce::dontSendNotification);
            vertical.setCurrentRange(viewY,1.0/zoomY,juce::dontSendNotification);
            xLabel.setText(juce::String(zoomX*100,0)+"%",juce::dontSendNotification);yLabel.setText(juce::String(zoomY*100,0)+"%",juce::dontSendNotification);
            xMinus.setEnabled(zoomX>1);xPlus.setEnabled(zoomX<32);yMinus.setEnabled(zoomY>1);yPlus.setEnabled(zoomY<16);repaint();
        }
        void scrollBarMoved(juce::ScrollBar* bar,double start) override
        {if(bar==&horizontal)viewX=start;else viewY=start;updateView();}
        float gainAt(const PreviewNote& n,double t) const
        {
            if(n.base.empty())return 1;const auto after=std::upper_bound(n.base.begin(),n.base.end(),t,[](double at,const auto& p){return at<p.timeSeconds;});
            if(after==n.base.begin())return backend::envelopeGainFromDb(n.base.front().gainDb);
            if(after==n.base.end())return backend::envelopeGainFromDb(n.base.back().gainDb);
            const auto& a=*std::prev(after);const auto& b=*after;
            return backend::envelopeGainFromDb(backend::envelopeDbBetween(a.gainDb,b.gainDb,(float)((t-a.timeSeconds)/std::max(1.e-9,b.timeSeconds-a.timeSeconds)),a.linearToNext));
        }
        float resultAt(double t,const PreviewNote& n) const
        {
            if(owner.nativeMode)return gainAt(n,t)*backend::tailFadeGain(owner.mode(),t,n.tail.startSeconds,n.tail.endSeconds,owner.values());
            return backend::advancedEnvelopeGain(owner.mode(),t,n.tail.startSeconds,n.tail.endSeconds,
                n.head?n.head->startSeconds:0,n.head?n.head->endSeconds:0,owner.values(),[&](double at){return gainAt(n,at);});
        }
        float anchor(const PreviewNote& n,double start,double end) const
        {
            if(owner.nativeMode)return gainAt(n,start);
            const auto v=owner.values();
            if(owner.editingHead())return gainAt(n,end)*(owner.mode()!=0?(float)v.startGain:1.0f);
            return backend::headEnvelopeGain(start,n.head?n.head->startSeconds:0,n.head?n.head->endSeconds:0,v.head,[&](double at){return gainAt(n,at);});
        }
        float ceiling() const
        {
            if(draggedKnot>=0)return dragCeiling;
            float c=1.2f;if(const auto* n=owner.current())
            {
                for(const auto& p:n->base)c=std::max(c,backend::envelopeGainFromDb(p.gainDb)*1.1f);
                for(int i=0;i<=64;++i)c=std::max(c,resultAt(first(*n)+(last(*n)-first(*n))*i/64.0,*n)*1.1f);
            }return c;
        }
        juce::Point<float> controlPoint(int index) const
        {
            if(owner.selectedMixedCurve())
            {
                const auto a=knotPoint(owner.selectedSegment),b=knotPoint(owner.selectedSegment+1);
                const auto c=owner.mixedCurveCoordinates();
                const double times[]{0,c[0],c[2],1},progress[]{0,c[1],c[3],1};
                return {a.x+(b.x-a.x)*(float)times[index],a.y+(b.y-a.y)*(float)progress[index]};
            }
            const auto* n=owner.current();if(!n)return {};const auto v=owner.editingValues();const auto len=owner.editingSpan(*n).endSeconds-owner.editingSpan(*n).startSeconds;
            const auto start=owner.editingSpan(*n).startSeconds+len*v.startFraction,end=owner.editingSpan(*n).startSeconds+len*v.endFraction;
            const double times[]{0,v.control1Time,v.control2Time,1},progress[]{0,v.control1Progress,v.control2Progress,1};
            return {x(start+(end-start)*times[index],*n),y(anchor(*n,start,end)*(float)(v.startGain+(v.endGain-v.startGain)*progress[index]))};
        }
        void dragControl(int index,juce::Point<float> at)
        {
            if(index<1||index>2||owner.editingMode()!=2)return;
            const auto a=controlPoint(0),b=controlPoint(3);const auto offset=(size_t)(index-1)*2;
            owner.curveControls[offset].setValue(100*(at.x-a.x)/std::max(1.e-6f,b.x-a.x),juce::sendNotificationSync);
            if(std::abs(b.y-a.y)>1.e-6f)owner.curveControls[offset+1].setValue(100*(at.y-a.y)/(b.y-a.y),juce::sendNotificationSync);
        }
        bool mixed() const {return owner.shape.getSelectedId()==3&&owner.editingMode()!=0;}
        double knotTime(int index) const
        {return index==0?0:index==(int)owner.knots.size()+1?1:owner.knots[(size_t)index-1].time;}
        double knotGain(int index) const
        {const auto v=owner.editingValues();return index==0?v.startGain:index==(int)owner.knots.size()+1?v.endGain:owner.knots[(size_t)index-1].gain;}
        juce::Point<float> knotPoint(int index) const
        {
            const auto* n=owner.current();if(!n)return {};
            const auto v=owner.editingValues();const auto span=owner.editingSpan(*n);const auto len=span.endSeconds-span.startSeconds;
            const auto start=span.startSeconds+len*v.startFraction,end=span.startSeconds+len*v.endFraction;
            const auto t=start+(end-start)*knotTime(index);
            // Use the same anchoring as rendering, including the opposite envelope.
            return {x(t,*n),y(resultAt(t,*n))};
        }
        int knotAt(juce::Point<float> at) const
        {
            for(int i=0;i<=(int)owner.knots.size()+1;++i)if(at.getDistanceFrom(knotPoint(i))<=9)return i;
            return -1;
        }
        std::pair<double,double> mixedCoordinates(juce::Point<float> at) const
        {
            const auto* n=owner.current();if(!n)return {};
            const auto v=owner.editingValues();const auto span=owner.editingSpan(*n);const auto len=span.endSeconds-span.startSeconds;
            const auto start=span.startSeconds+len*v.startFraction,end=span.startSeconds+len*v.endFraction;
            const auto time=(at.x-x(start,*n))/std::max(1.e-6f,x(end,*n)-x(start,*n));
            const auto level=owner.nativeMode?gainAt(*n,start+(end-start)*time):anchor(*n,start,end);
            return {time,gainAtY(at.y)/std::max(.0001f,level)};
        }
        void mixedMenu(int knot,int segmentIndex)
        {
            juce::PopupMenu menu;menu.addItem(1,text("此段：直线"));menu.addItem(2,text("此段：贝塞尔曲线"));
            const auto indices=selectedKnots.empty()?std::vector<int>{knot}:selectedKnots;
            if(std::any_of(indices.begin(),indices.end(),[&](int i){return i>0&&i<=(int)owner.knots.size();}))
            {menu.addSeparator();menu.addItem(3,text("删除选中标点"));menu.addItem(4,text("初始化选中标点（Ctrl+R）"));}
            juce::Component::SafePointer<AdvancedEnvelopePanel> safe(&owner);
            menu.showMenuAsync(juce::PopupMenu::Options().withTargetComponent(this),[safe,indices,segmentIndex](int result){
                if(!safe)return;if(result==3||result==4)safe->editKnots(indices,result==4);else if(result==1||result==2)safe->setSegmentCurved(segmentIndex,result==2);
            });
        }
        void paint(juce::Graphics& g) override
        {
            g.fillAll(Palette::graphBackground);g.setFont(juce::FontOptions(12.0f));
            g.setColour(Palette::textMuted);g.drawText(text("灰：原包络   绿：实际响度   黄：当前段"),440,2,getWidth()-448,24,juce::Justification::centredRight);
            const auto* n=owner.current();if(!n)return;const auto bounds=plot();const auto v=owner.editingValues();const auto length=owner.editingSpan(*n).endSeconds-owner.editingSpan(*n).startSeconds;
            const auto start=owner.editingSpan(*n).startSeconds+length*v.startFraction,end=owner.editingSpan(*n).startSeconds+length*v.endFraction;
            g.setColour(Palette::grid);
            for(int i=0;i<=4;++i)g.drawHorizontalLine((int)(bounds.getBottom()-bounds.getHeight()*i/4),bounds.getX(),bounds.getRight());
            g.saveState();g.reduceClipRegion(bounds.expanded(6).toNearestInt());
            g.setColour(Palette::textMuted.withAlpha(.08f));g.fillRect(juce::Rectangle<float>(x(owner.editingSpan(*n).startSeconds,*n),bounds.getY(),x(owner.editingSpan(*n).endSeconds,*n)-x(owner.editingSpan(*n).startSeconds,*n),bounds.getHeight()));
            if(owner.editingMode()!=0){g.setColour(juce::Colour(0xff72d6aa).withAlpha(.1f));g.fillRect(juce::Rectangle<float>(x(start,*n),bounds.getY(),x(end,*n)-x(start,*n),bounds.getHeight()));}
            paintOtoRegionGuides(g,bounds,n->regions,[&](double time){return x(time,*n);});
            juce::Path original,result;
            for(int i=0;i<=512;++i)
            {
                const auto t=visibleFirst(*n)+(visibleLast(*n)-visibleFirst(*n))*i/512.0;const auto base=gainAt(*n,t);
                const juce::Point<float> a{x(t,*n),y(base)};
                const juce::Point<float> b{x(t,*n),y(resultAt(t,*n))};
                if(i==0){original.startNewSubPath(a);result.startNewSubPath(b);}else{original.lineTo(a);result.lineTo(b);}
            }
            g.setColour(Palette::textMuted);g.strokePath(original,juce::PathStrokeType(1.6f));
            juce::Path dashed;const float dash[]{6,4};juce::PathStrokeType(2).createDashedStroke(dashed,result,dash,2);g.setColour(juce::Colour(0xff72d6aa));g.fillPath(dashed);
            if(owner.editingMode()!=0)for(auto t:{start,end}){const auto at=x(t,*n);g.drawVerticalLine((int)at,bounds.getY(),bounds.getBottom());g.fillEllipse(at-5,bounds.getY()-5,10,10);}
            if(owner.editingMode()==2&&(owner.shape.getSelectedId()==2||owner.selectedMixedCurve()))
            {
                g.setColour(juce::Colour(0xffffc86a));
                g.drawLine(juce::Line<float>(controlPoint(0),controlPoint(1)),1.2f);g.drawLine(juce::Line<float>(controlPoint(3),controlPoint(2)),1.2f);
                for(int i=1;i<=2;++i){const auto p=controlPoint(i);g.fillRect(juce::Rectangle<float>(p.x-5,p.y-5,10,10));
                    g.drawText(juce::String(i),(int)p.x+7,(int)p.y-20,16,18,juce::Justification::centred);}
            }
            if(mixed())
            {
                for(int i=0;i<=(int)owner.knots.size();++i)
                {
                    const auto t0=std::max(visibleFirst(*n),start+(end-start)*knotTime(i)),t1=std::min(visibleLast(*n),start+(end-start)*knotTime(i+1));
                    if(t1<t0)continue;
                    juce::Path path;
                    for(int j=0;j<=48;++j){const auto t=t0+(t1-t0)*j/48.0;const juce::Point<float> p{x(t,*n),y(resultAt(t,*n))};if(j==0)path.startNewSubPath(p);else path.lineTo(p);}
                    g.setColour(juce::Colour(i==owner.selectedSegment?0xffffc86a:0xff72d6aa));g.strokePath(path,juce::PathStrokeType(i==owner.selectedSegment?3.0f:2.0f));
                }
                for(int i=0;i<=(int)owner.knots.size()+1;++i){const auto p=knotPoint(i);g.setColour(std::find(selectedKnots.begin(),selectedKnots.end(),i)!=selectedKnots.end()?juce::Colour(0xffffc86a):Palette::graphBackground);g.fillEllipse(p.x-5,p.y-5,10,10);g.setColour(juce::Colour(0xffffc86a));g.drawEllipse(p.x-5,p.y-5,10,10,2);}
            }
            g.restoreState();
            for(int i=0;i<=4;++i)
            {
                const auto py=bounds.getBottom()-bounds.getHeight()*i/4;
                g.setColour(Palette::textMuted);g.drawText(juce::String(gainAtY(py)*100,1)+"%",0,(int)py-8,49,16,juce::Justification::centredRight);
            }
            g.setColour(Palette::textMuted);g.drawText(juce::String(visibleFirst(*n)*1000,1)+" ms",(int)bounds.getX(),(int)bounds.getBottom()+3,100,18,juce::Justification::centredLeft);
            g.drawText(juce::String(visibleLast(*n)*1000,1)+" ms",(int)bounds.getRight()-100,(int)bounds.getBottom()+3,100,18,juce::Justification::centredRight);
        }
        void mouseDown(const juce::MouseEvent& e) override
        {
            const auto previous=selectedKnots;clearSelection();grabKeyboardFocus();
            dragged=-1;draggedControl=-1;draggedKnot=-1;if(owner.editingMode()==0||!plot().expanded(6).contains(e.position))return;
            if(owner.selectedMixedCurve()&&!e.mods.isPopupMenu()&&knotAt(e.position)<0)
                for(int i=1;i<=2;++i)if(e.position.getDistanceFrom(controlPoint(i))<=10){draggedControl=i;dragOrigin=e.position;controlOrigin=controlPoint(i);return;}
            if(mixed()&&plot().expanded(10).contains(e.position)&&e.position.y>plot().getY()+9)
            {
                const auto coords=mixedCoordinates(e.position);if(coords.first>=-.02&&coords.first<=1.02)
                {
                    const auto knot=knotAt(e.position);int index=0;for(const auto& p:owner.knots)if(coords.first>p.time)++index;
                    if(knot>=0)index=std::min(knot,(int)owner.knots.size());
                    owner.selectedSegment=index;owner.refresh();
                    if(knot>=0)
                    {
                        const auto wasSelected=std::find(previous.begin(),previous.end(),knot)!=previous.end();
                        if(e.mods.isCtrlDown()||wasSelected)selectedKnots=previous;
                        if(e.mods.isCtrlDown()&&wasSelected&&!e.mods.isPopupMenu())
                        {std::erase(selectedKnots,knot);repaint();return;}
                        if(!wasSelected||selectedKnots.empty())selectedKnots.push_back(knot);
                        selectedKnot=knot;
                    }
                    if(e.mods.isPopupMenu()){mixedMenu(knot,index);return;}
                    if(knot>=0)
                    {
                        dragCeiling=ceiling();dragOrigin=e.position;controlOrigin=knotPoint(knot);draggedKnot=knot;
                        dragPoints.clear();for(int i=0;i<=(int)owner.knots.size()+1;++i)dragPoints.push_back({knotTime(i),knotGain(i)});
                        repaint();
                    }
                    return;
                }
            }
            if(owner.editingMode()==2&&owner.shape.getSelectedId()==2)for(int i=1;i<=2;++i)if(e.position.getDistanceFrom(controlPoint(i))<=12){draggedControl=i;dragOrigin=e.position;controlOrigin=controlPoint(i);return;}if(const auto* n=owner.current())
            {const auto v=owner.editingValues();const auto len=owner.editingSpan(*n).endSeconds-owner.editingSpan(*n).startSeconds;const auto a=x(owner.editingSpan(*n).startSeconds+len*v.startFraction,*n),b=x(owner.editingSpan(*n).startSeconds+len*v.endFraction,*n);if(std::min(std::abs(e.position.x-a),std::abs(e.position.x-b))<=12)dragged=std::abs(e.position.x-a)<std::abs(e.position.x-b)?0:1;}
        }
        void mouseDrag(const juce::MouseEvent& e) override
        {
            if(draggedKnot>=0)
            {
                const auto at=controlOrigin+(e.position-dragOrigin)*(e.mods.isShiftDown()?.15f:1.0f);const auto [time,gain]=mixedCoordinates(at);
                if(selectedKnots.size()<2){owner.moveKnot(draggedKnot,time,gain);return;}
                PointGroupDelta limits;limits.constrain(dragPoints,selectedKnots,0,1,0,2,.002,true);
                const auto origin=dragPoints[(size_t)draggedKnot];const auto delta=limits.clamp(time-origin.time,gain-origin.value);
                for(int i:selectedKnots)
                {
                    const auto p=dragPoints[(size_t)i];
                    if(i==0||i==(int)owner.knots.size()+1)
                        owner.sliders[i==0?2:3].setValue((p.value+delta.value)*100,juce::dontSendNotification);
                    else {owner.knots[(size_t)i-1].time=p.time+delta.time;owner.knots[(size_t)i-1].gain=p.value+delta.value;}
                }
                owner.refresh();
                return;
            }
            if(draggedControl>0){dragControl(draggedControl,controlOrigin+(e.position-dragOrigin)*(e.mods.isShiftDown()?.15f:1.0f));return;}
            if(dragged<0)return;if(const auto* n=owner.current();n&&owner.editingSpan(*n).endSeconds-owner.editingSpan(*n).startSeconds>1.e-9)
            {const auto t=timeAt(e.position.x,*n);owner.sliders[(size_t)dragged].setValue(100*(t-owner.editingSpan(*n).startSeconds)/(owner.editingSpan(*n).endSeconds-owner.editingSpan(*n).startSeconds),juce::sendNotificationSync);}
        }
        void mouseDoubleClick(const juce::MouseEvent& e) override
        {
            if(!plot().expanded(6).contains(e.position))return;
            if(mixed())
            {
                if(owner.selectedMixedCurve()&&!e.mods.isPopupMenu())for(int i=1;i<=2;++i)if(e.position.getDistanceFrom(controlPoint(i))<=10)
                {owner.curveControls[(size_t)(i-1)*2].setValue(i*100.0/3,juce::sendNotificationSync);owner.curveControls[(size_t)(i-1)*2+1].setValue(i==1?0:100,juce::sendNotificationSync);return;}
                if(!e.mods.isPopupMenu()&&plot().contains(e.position)&&knotAt(e.position)<0){const auto [time,gain]=mixedCoordinates(e.position);owner.addKnot(time,gain);}return;
            }
            if(owner.editingMode()!=2)return;
            for(int i=1;i<=2;++i)if(e.position.getDistanceFrom(controlPoint(i))<=12)
            {owner.curveControls[(size_t)(i-1)*2].setValue(i*100.0/3,juce::sendNotificationSync);owner.curveControls[(size_t)(i-1)*2+1].setValue(i==1?0:100,juce::sendNotificationSync);break;}
        }
        void mouseUp(const juce::MouseEvent&) override {dragged=-1;draggedControl=-1;draggedKnot=-1;repaint();}
        juce::TextButton xMinus,xPlus,yMinus,yPlus,fit;
        juce::Label xLabel,yLabel;
        juce::ScrollBar horizontal{false},vertical{true};
        double zoomX=1,zoomY=1,viewX=0,viewY=0;
        AdvancedEnvelopePanel& owner;int dragged=-1,draggedControl=-1,draggedKnot=-1,selectedKnot=-1;float dragCeiling=1.2f;juce::Point<float> dragOrigin,controlOrigin;
        std::vector<int> selectedKnots;
        std::vector<PointGroupCoordinate> dragPoints;
        struct SelectionListener : juce::MouseListener
        {
            explicit SelectionListener(Graph& g):graph(g){}
            void mouseDown(const juce::MouseEvent& e) override
            {if(e.eventComponent!=&graph)graph.clearSelection();}
            Graph& graph;
        } selectionListener{*this};
    };
    const bool nativeMode,phonemeMode;
    backend::TailFadeSettings draft;int draftTailMode=0,draftTailShape=1,activeSection=0;
    bool bezierCurve=false,firstSegmentCurved=false;int selectedSegment=0;
    std::vector<backend::EnvelopeKnot> knots;
    backend::EnvelopeBezier firstBezier;
    TailFadePresetStore presetStore;std::vector<TailFadePreset> userPresets;
    std::vector<PreviewNote> previews;
    juce::Label title,summary,shapeLabel,previewLabel,detail,help,status;
    juce::ToggleButton enabled;juce::ComboBox shape,preview,preset,section,segment,segmentType;
    Graph graph;std::array<juce::Label,5> labels;std::array<juce::Slider,5> sliders;
    std::array<double,4> curveCoordinates{1.0/3.0,0,2.0/3.0,1};
    std::array<juce::Label,4> curveLabels;std::array<juce::Slider,4> curveControls;
    juce::TextButton reset,apply,listen,close,savePresetButton,deletePresetButton;
};
}
