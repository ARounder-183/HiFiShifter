#pragma once
#include "I18n.h"
// Included once by PianoRollComponent.cpp, after the main implementation.
namespace hachi
{
bool PianoRollComponent::diffSingerPitchOffsetAvailable() const
{
    for (const auto& track : snapshot.tracks)
        if ((focusedTrack.isEmpty() || track.id==focusedTrack) && trackIsDiffSinger(track)) return true;
    return false;
}

void PianoRollComponent::setDiffSingerPitchOffsetMode(bool enabled)
{
    enabled=enabled && !sourceEditMode && tool==Tool::points && diffSingerPitchOffsetAvailable();
    if (enabled) setDiffSingerPitchRestoreMode(false);
    if (dsPitchOffsetMode==enabled) return;
    cancelDiffSingerPitchOffsetGesture();
    dsPitchOffsetMode=enabled;
    draggedNote.clear();pitchStroke.clear();dragMode=DragMode::none;
    if (onPitchPointModeChanged) onPitchPointModeChanged();
    repaint();
}

bool PianoRollComponent::cancelDiffSingerPitchOffsetGesture()
{
    const auto active=offsetNoteId.isNotEmpty();
    offsetNoteId.clear();offsetStroke.clear();offsetOriginal.clear();offsetInk.clear();offsetMoved=false;
    repaint();return active;
}

std::vector<PitchCurveEditPoint> PianoRollComponent::offsetCurveFor(const NoteData& note) const
{
    if (offsetNoteId==note.id && !offsetStroke.empty()) return offsetStroke;
    if (!note.diffSingerPitchOffset.empty()) return note.diffSingerPitchOffset;
    return {{0,0,PitchCurveShape::linear},{note.durationSeconds,0,PitchCurveShape::linear}};
}

float PianoRollComponent::displayedPitchOffset(const NoteData& note,double time) const
{
    if (note.diffSingerPitchOffset.empty() && offsetNoteId!=note.id) return 0;
    if (!dsPitchOffsetNoteIds.contains(note.id)) return 0;
    if (offsetNoteId==note.id && !offsetStroke.empty()) return evaluatePitchCurve(offsetStroke,time);
    return diffSingerPitchOffsetAt(note,time);
}

void PianoRollComponent::drawDiffSingerPitchOffsets(juce::Graphics& g) const
{
    const auto colour=rollRole(juce::Colour(0xffd698ff),Palette::noteEdge);
    for (const auto& hit : noteHits)
    {
        const auto* note=findNote(hit.id);
        if (!note || !dsPitchOffsetNoteIds.contains(hit.id) || backend::isRestLyric(note->label)) continue;
        const auto points=offsetCurveFor(*note);
        const auto start=hit.startSeconds+hit.clipStartSeconds;
        const auto position=[&](double t,float value){return juce::Point<float>(timeToX(start+t),midiToY(note->midiNote+value)+rowHeight*.5f);};
        juce::Path line;line.startNewSubPath(position(0,evaluatePitchCurve(points,0)));
        for (const auto& p : points) line.lineTo(position(p.timeSeconds,p.targetMidi));
        line.lineTo(position(note->durationSeconds,evaluatePitchCurve(points,note->durationSeconds)));
        const auto noteSurface=Palette::graphBackground.overlaidWith(Palette::noteFill.withAlpha(.90f));
        g.setColour(rollInk(colour.withAlpha(selectedNote==note->id?1.0f:.75f),noteSurface));g.strokePath(line,juce::PathStrokeType(2));
        for (const auto& p : points)
        {
            const auto at=position(p.timeSeconds,p.targetMidi);
            g.setColour(Palette::panelRaised);g.fillEllipse(at.x-4.5f,at.y-4.5f,9,9);
            g.setColour(colour);g.drawEllipse(at.x-4.5f,at.y-4.5f,9,9,1.5f);
        }
        if (selectedNote==note->id)
        {
            const auto value=evaluatePitchCurve(points,0);const auto at=position(0,value);
            g.setColour(rollText(colour,noteSurface));g.setFont(11.0f);
            g.drawText(I18n::uiText("偏移 ")+(value>=0?"+":"")+juce::String(value,2)+" st",
                static_cast<int>(at.x+7),static_cast<int>(at.y-19),120,16,juce::Justification::centredLeft);
        }
    }
}

void PianoRollComponent::offsetMouseDown(const juce::MouseEvent& event)
{
    grabKeyboardFocus();cancelDiffSingerPitchOffsetGesture();
    if (const auto* viewport=findParentComponentOfClass<juce::Viewport>())
        if (event.x<viewport->getViewPositionX()+58) return;
    rebuildNoteHits();
    const auto time=(event.position.x-58.0)/pixelsPerSecond;
    const NoteHit* chosen=nullptr;int anchor=-1;float best=std::numeric_limits<float>::infinity();
    for (const auto& hit : noteHits)
    {
        const auto* note=findNote(hit.id);
        if (!note || !dsPitchOffsetNoteIds.contains(hit.id) || backend::isRestLyric(note->label)) continue;
        const auto start=hit.startSeconds+hit.clipStartSeconds;
        if (time<start-8.0/pixelsPerSecond || time>start+note->durationSeconds+8.0/pixelsPerSecond) continue;
        const auto points=offsetCurveFor(*note);
        auto distance=std::abs(event.position.y-(midiToY(note->midiNote+evaluatePitchCurve(points,time-start))+rowHeight*.5f));
        int pointIndex=-1;
        for (size_t i=0;i<points.size();++i)
        {
            const auto at=juce::Point<float>(timeToX(start+points[i].timeSeconds),midiToY(note->midiNote+points[i].targetMidi)+rowHeight*.5f);
            if (at.getDistanceFrom(event.position)<=9) { pointIndex=static_cast<int>(i);distance=-1;break; }
        }
        if (distance<best) { best=distance;chosen=&hit;anchor=pointIndex; }
    }
    if (!chosen || (best>12 && !event.mods.isCtrlDown())) return;
    const auto* note=findNote(chosen->id);if (!note) return;
    const auto id=note->id;
    const auto original=offsetCurveFor(*note);
    const auto start=chosen->startSeconds+chosen->clipStartSeconds;
    selectedNote=id;selectedNotes.clear();selectedNotes.insert(id.toStdString());
    if (onNoteSelected) onNoteSelected(id);
    if (event.mods.isPopupMenu()) { showOffsetMenu(id,anchor,time-start,event.getScreenPosition());repaint();return; }
    offsetNoteId=id;offsetOriginal=original;offsetStroke=original;offsetAbsoluteStart=start;
    offsetPress=event.position;offsetAnchor=anchor;offsetWhole=anchor<0;
    offsetFreehand=event.mods.isCtrlDown();offsetMoved=false;offsetRevision=model.revisionNumber();
    offsetLastTime=juce::jlimit(0.0,note->durationSeconds,time-start);
    offsetLastValue=pitchMidiFromY(event.position.y)-note->midiNote;
    offsetInk={{offsetLastTime,offsetLastValue,PitchCurveShape::linear}};
    repaint();
}

void PianoRollComponent::offsetMouseDrag(const juce::MouseEvent& event)
{
    if (offsetNoteId.isEmpty()) return;
    const auto* note=findNote(offsetNoteId);
    if (!note || model.revisionNumber()!=offsetRevision) { cancelDiffSingerPitchOffsetGesture();return; }
    if (!offsetMoved && event.position.getDistanceFrom(offsetPress)<2) return;
    offsetMoved=true;
    const auto t=juce::jlimit(0.0,note->durationSeconds,(event.position.x-58.0)/pixelsPerSecond-offsetAbsoluteStart);
    const auto value=juce::jlimit(-127.0f,127.0f,pitchMidiFromY(event.position.y)-note->midiNote);
    if (offsetFreehand)
    {
        const auto lo=std::min(t,offsetLastTime),hi=std::max(t,offsetLastTime);
        std::erase_if(offsetInk,[&](const auto& p){return p.timeSeconds>=lo && p.timeSeconds<=hi;});
        offsetInk.push_back({offsetLastTime,offsetLastValue,PitchCurveShape::linear});
        offsetInk.push_back({t,value,PitchCurveShape::linear});
        std::stable_sort(offsetInk.begin(),offsetInk.end(),[](const auto& a,const auto& b){return a.timeSeconds<b.timeSeconds;});
        const auto from=offsetInk.front().timeSeconds,to=offsetInk.back().timeSeconds;
        offsetStroke=offsetOriginal;
        std::erase_if(offsetStroke,[&](const auto& p){return p.timeSeconds>=from-1e-5 && p.timeSeconds<=to+1e-5;});
        if (from>1e-5) offsetStroke.push_back({from-1e-5,evaluatePitchCurve(offsetOriginal,from),PitchCurveShape::linear});
        if (to<note->durationSeconds-1e-5) offsetStroke.push_back({to+1e-5,evaluatePitchCurve(offsetOriginal,to),PitchCurveShape::linear});
        offsetStroke.insert(offsetStroke.end(),offsetInk.begin(),offsetInk.end());
        std::stable_sort(offsetStroke.begin(),offsetStroke.end(),[](const auto& a,const auto& b){return a.timeSeconds<b.timeSeconds;});
        offsetLastTime=t;offsetLastValue=value;
    }
    else if (offsetWhole)
    {
        offsetStroke=offsetOriginal;
        const auto delta=(offsetPress.y-event.position.y)/rowHeight;
        for (auto& p : offsetStroke) p.targetMidi=juce::jlimit(-127.0f,127.0f,p.targetMidi+delta);
    }
    else if (offsetAnchor>=0 && offsetAnchor<static_cast<int>(offsetStroke.size()))
    {
        auto& p=offsetStroke[static_cast<size_t>(offsetAnchor)];p.targetMidi=value;
        if (offsetAnchor>0 && offsetAnchor+1<static_cast<int>(offsetStroke.size()))
        {
            const auto lo=offsetStroke[static_cast<size_t>(offsetAnchor-1)].timeSeconds+1e-5;
            const auto hi=offsetStroke[static_cast<size_t>(offsetAnchor+1)].timeSeconds-1e-5;
            if (lo<=hi) p.timeSeconds=juce::jlimit(lo,hi,t);
        }
    }
    repaint();
}

void PianoRollComponent::offsetMouseUp()
{
    const auto id=offsetNoteId;auto points=offsetStroke;
    const auto commit=offsetMoved && !id.isEmpty() && offsetRevision==model.revisionNumber();
    cancelDiffSingerPitchOffsetGesture();
    if (commit) model.setDiffSingerPitchOffset(id,std::move(points));
}

bool PianoRollComponent::addOffsetPoint(const juce::String& id,double localSeconds)
{
    const auto* note=findNote(id);
    if (!dsPitchOffsetMode || !note || !dsPitchOffsetNoteIds.contains(id)
        || !std::isfinite(localSeconds)) return false;
    auto points=offsetCurveFor(*note);
    const auto t=juce::jlimit(0.0,note->durationSeconds,localSeconds);
    for (const auto& p : points) if (std::abs(p.timeSeconds-t)*pixelsPerSecond<9) return false;
    // Insert on the existing line; adding a handle must not change the sound.
    points.push_back({t,evaluatePitchCurve(points,t),PitchCurveShape::linear});
    return model.setDiffSingerPitchOffset(id,std::move(points));
}

void PianoRollComponent::offsetDoubleClick(const juce::MouseEvent& event)
{
    if (event.mods.isPopupMenu() || event.mods.isCtrlDown() || event.mouseWasDraggedSinceMouseDown()) return;
    // JUCE sends mouseUp BEFORE mouseDoubleClick. Hit-test again because the
    // gesture has already ended; do not depend on its cleared press state.
    offsetMouseDown(event);
    const auto id=offsetNoteId;
    const auto local=(event.position.x-58.0)/pixelsPerSecond-offsetAbsoluteStart;
    cancelDiffSingerPitchOffsetGesture();
    if (id.isNotEmpty()) addOffsetPoint(id,local);
}

void PianoRollComponent::showOffsetMenu(const juce::String& id,int point,double localSeconds,juce::Point<int> screenPosition)
{
    juce::PopupMenu menu;menu.addSectionHeader(I18n::uiText("DS 音高偏移（半音）"));
    bool canAdd=point<0;
    if (const auto* note=findNote(id))
        for (const auto& p : offsetCurveFor(*note))
            if (std::abs(p.timeSeconds-localSeconds)*pixelsPerSecond<9) canAdd=false;
    menu.addItem(4,I18n::uiText("增加偏移标点"),canAdd);
    menu.addSeparator();
    menu.addItem(1,I18n::uiText("整音偏移数值…"));
    menu.addItem(2,I18n::uiText("此音偏移归零"));
    menu.addItem(3,I18n::uiText("删除此偏移点"),point>=0);
    const auto revision=model.revisionNumber();
    juce::Component::SafePointer<PianoRollComponent> safe(this);
    menu.showMenuAsync(juce::PopupMenu::Options().withTargetScreenArea({screenPosition.x,screenPosition.y,1,1}),
        [safe,id,point,localSeconds,revision](int result)
        {
            if (!safe || safe->model.revisionNumber()!=revision) return;
            const auto* note=safe->findNote(id);if (!note) return;
            if (result==4) { safe->addOffsetPoint(id,localSeconds);return; }
            if (result==2) { safe->model.setDiffSingerPitchOffset(id,{});return; }
            if (result==3)
            {
                auto points=safe->offsetCurveFor(*note);
                if (point>=0 && point<static_cast<int>(points.size())) points.erase(points.begin()+point);
                safe->model.setDiffSingerPitchOffset(id,std::move(points));return;
            }
            if (result!=1) return;
            auto* dialog=new juce::AlertWindow(I18n::uiText("整音偏移"),
                I18n::uiText("以半音为单位，负数向下。此操作将偏移线设为水平线；原始预测虚线不变。"),juce::MessageBoxIconType::NoIcon,safe.getComponent());
            dialog->addTextEditor("offset",juce::String(diffSingerPitchOffsetAt(*note,0),2),I18n::uiText("偏移（半音）"));
            dialog->getTextEditor("offset")->setInputRestrictions(0,"-0123456789.");
            dialog->addButton(I18n::uiText("应用"),1);dialog->addButton(I18n::uiText("取消"),0,juce::KeyPress(juce::KeyPress::escapeKey));
            dialog->enterModalState(true,juce::ModalCallbackFunction::create([safe,dialog,id,revision](int choice)
            {
                const auto text=dialog->getTextEditorContents("offset").trim();
                if (safe && choice==1 && safe->model.revisionNumber()==revision && text.isNotEmpty())
                    if (const auto* current=safe->findNote(id))
                        safe->model.setDiffSingerPitchOffset(id,{{0,text.getFloatValue(),PitchCurveShape::linear},
                            {current->durationSeconds,text.getFloatValue(),PitchCurveShape::linear}});
                delete dialog;
            }),false);
        });
}
}
