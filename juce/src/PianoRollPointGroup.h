#pragma once
#include "I18n.h"
#include "PointGroupEdit.h"

namespace hachi
{
void PianoRollComponent::cancelPointDrag()
{
    draggedNote.clear();draggedPitchAnchor=draggedAmplitudePoint=draggedFlagPoint=-1;
    pitchStroke.clear();amplitudeStroke.clear();flagStroke.clear();cvvcLaneDrag.reset();dragMode=DragMode::none;
    graphPointDragging=false;graphDragValid=false;clearPitchPlanPreview();
}

void PianoRollComponent::mouseDown(const juce::MouseEvent& event)
{
    auto previous=selectedPoints;
    if(previous.empty()&&selectedPoint&&selectedPoint->index>=0)previous.push_back(*selectedPoint);
    std::erase_if(previous,[&](const auto& p){return p.revision!=model.revisionNumber();});
    pointGroupGesture.reset();
    if(event.mods.isPopupMenu()&&previous.size()>1)
    {
        cancelPointDrag();
        juce::PopupMenu menu;if(previous.front().kind!=PointSelection::connectionPitch)menu.addItem(1,I18n::uiText("初始化选中标点（Ctrl+R）"));
        menu.addItem(2,I18n::uiText("删除选中标点（Del）"));
        juce::Component::SafePointer<PianoRollComponent> safe(this);
        menu.showMenuAsync(juce::PopupMenu::Options().withTargetScreenArea({event.getScreenX(),event.getScreenY(),1,1}),
            [safe,previous](int result){if(safe&&result){safe->selectedPoints=previous;safe->selectedPoint=previous.back();safe->editSelectedPoints(result==1);}});
        return;
    }
    // Existing hit testing remains authoritative. It also handles controls,
    // empty space and non-point tools, which clear the previous selection.
    {const juce::ScopedValueSetter<bool> hitTesting(groupPointHitTest,previous.size()>1);mouseDownSingle(event);}
    if(!selectedPoint||selectedPoint->index<0||dsPitchRestoreMode)return;
    auto current=*selectedPoint;
    if(current.kind==PointSelection::amplitude)
        if(const auto* member=nativeEnvelopeMember(current.noteId))
            for(const auto& p:previous)if(p.kind==current.kind)
                if(const auto* other=nativeEnvelopeMember(p.noteId);other&&other->points==member->points)
                {current.noteId=p.noteId;break;}
    const auto same=[&](const auto& p){return p.kind==current.kind&&p.noteId==current.noteId&&p.part==current.part&&p.key==current.key&&p.index==current.index;};
    const bool wasSelected=std::any_of(previous.begin(),previous.end(),same);
    if(!event.mods.isCtrlDown()&&!wasSelected)
    {if(previous.size()>1&&current.kind!=PointSelection::connectionPitch)selectedNotes={current.noteId.toStdString()};return;}
    std::erase_if(previous,[&](const auto& p){return p.kind!=current.kind||p.key!=current.key;});
    if(event.mods.isCtrlDown()&&wasSelected)
    {
        std::erase_if(previous,same);cancelPointDrag();selectedPoints=std::move(previous);
        if(selectedPoints.empty())selectedPoint->index=-1;else selectedPoint=selectedPoints.back();
        if(selectedPoints.size()==1)selectedPoints.clear();
        repaint();return;
    }
    if(!wasSelected)previous.push_back(current);
    selectedPoints=std::move(previous);selectedPoint=current;
    if(current.kind!=PointSelection::connectionPitch){selectedNotes.clear();for(const auto& p:selectedPoints)selectedNotes.insert(p.noteId.toStdString());}
    if(selectedPoints.size()>1)
    {
        auto curves=pointCurveDrafts();cancelPointDrag();
        if(!curves.empty())pointGroupGesture=PointGroupGesture{std::move(curves),event.position,model.revisionNumber(),false};
    }
    else selectedPoints.clear();
    repaint();
}

std::vector<PianoRollComponent::PointCurveDraft> PianoRollComponent::pointCurveDrafts()
{
    std::vector<PointCurveDraft> curves;
    for(const auto& p:selectedPoints)
    {
        if(p.index<0||p.revision!=model.revisionNumber())continue;
        auto at=std::find_if(curves.begin(),curves.end(),[&](const auto& d){return d.identity.kind==p.kind&&d.identity.noteId==p.noteId&&d.identity.part==p.part&&d.identity.key==p.key;});
        if(at==curves.end())
        {
            PointCurveDraft d;d.identity=p;d.edit.kind=static_cast<ProjectModel::PointCurveEdit::Kind>(p.kind);
            if(p.kind==PointSelection::connectionPitch)
            {
                bool found=false;for(const auto& curve:visibleConnectionCurves(false))if(curve.connectionId==p.noteId)
                {d.edit.connectionId=p.noteId;d.edit.pitchPoints=curve.points;d.edit.reanchorConnectionEndpoints=true;
                    d.first=0;d.last=std::max(snapshot.durationSeconds()+30,curve.toSeconds+30);found=true;break;}
                if(!found)continue;
            }
            else
            {
            const auto* note=findNote(p.noteId);if(!note)continue;
            const auto hit=std::find_if(noteHits.begin(),noteHits.end(),[&](const auto& h){return h.id==p.noteId;});
            if(hit==noteHits.end())continue;
            d.edit.noteId=p.noteId;d.edit.key=p.key;d.edit.part=p.part;
            d.start=hit->startSeconds+hit->clipStartSeconds;d.last=note->durationSeconds;
            d.base=std::max(.000001,static_cast<double>(note->amplitudeEnvelopeBasePercent)/100);
            if(p.kind==PointSelection::pitch)
            {
                d.edit.pitchPoints=pitchAnchorsFor(*note);
                d.first=std::max(-30.0,-d.start);d.last=note->durationSeconds+30;
                if(!nativeNoteTimingEnabled(p.noteId)&&!d.edit.pitchPoints.empty())
                {
                    d.first=std::min(0.0,d.edit.pitchPoints.front().timeSeconds);
                    d.last=std::max(note->durationSeconds,d.edit.pitchPoints.back().timeSeconds);
                    if(const auto span=utauSoundSpans.find(p.noteId.toStdString());span!=utauSoundSpans.end())d.first=std::min(d.first,span->second.first-d.start);
                    if(const auto prev=previousUtauNoteFor(p.noteId))d.first=std::min(d.first,lastShownPitchPoint(prev->id,prev->startSeconds).value_or(prev->startSeconds)-d.start);
                    if(const auto next=nextUtauNoteFor(p.noteId))d.last=std::max(d.last,firstShownPitchPoint(next->id,next->startSeconds).value_or(next->startSeconds)-d.start);
                }
            }
            else if(p.kind==PointSelection::amplitude)
            {
                d.edit.amplitudePoints=amplitudeEnvelopeFor(*note,d.start);d.high=2*d.base;d.separation=.005;
                if(!d.edit.amplitudePoints.empty()){d.first=d.edit.amplitudePoints.front().timeSeconds;d.last=d.edit.amplitudePoints.back().timeSeconds;}
            }
            else if(p.kind==PointSelection::flag)
            {
                d.edit.flagPoints=flagLaneCurveFor(*note);const auto span=flagLaneSpanFor(*note);
                d.first=span.first;d.last=span.second;d.low=flagCurveKindFor(p.key).minimum;d.high=flagCurveKindFor(p.key).maximum;d.separation=.005;
                if(d.edit.flagPoints.size()==1)d.first=d.last=d.edit.flagPoints.front().timeSeconds;
            }
            else
            {
                bool found=false;
                for(const auto& piece:cvvcLanePieces(p.key.isEmpty()))if(piece.noteId==p.noteId&&piece.part==p.part&&piece.key==p.key)
                {d.edit.cvvcPoints=piece.points;d.start=piece.from;d.length=piece.to-piece.from;d.last=d.length;found=true;break;}
                if(!found)continue;
                d.fixedEnds=true;d.separation=1.e-6*d.length;
                d.low=p.key.isEmpty()?0:flagCurveKindFor(p.key).minimum;d.high=p.key.isEmpty()?2*d.base:flagCurveKindFor(p.key).maximum;
            }
            }
            curves.push_back(std::move(d));at=std::prev(curves.end());
        }
        if(std::find(at->indices.begin(),at->indices.end(),p.index)==at->indices.end())at->indices.push_back(p.index);
    }
    for(auto& d:curves)
    {
        if(d.identity.kind==PointSelection::pitch)
        {
            std::vector<double> times;
            for(int i:d.indices)if(i<(int)d.edit.pitchPoints.size())times.push_back(d.edit.pitchPoints[(size_t)i].timeSeconds);
            for(double t:times){auto& ps=d.edit.pitchPoints;auto p=std::find_if(ps.begin(),ps.end(),[&](auto& a){return std::abs(a.timeSeconds-t)<1.e-8;});if(p!=ps.end()){int i=(int)(p-ps.begin());prepareSampledPitchPointEdit(ps,i);}}
            d.indices.clear();for(double t:times)for(size_t i=0;i<d.edit.pitchPoints.size();++i)if(std::abs(d.edit.pitchPoints[i].timeSeconds-t)<1.e-8)d.indices.push_back((int)i);
            for(const auto& p:d.edit.pitchPoints)d.original.emplace_back(p.timeSeconds,p.targetMidi);
        }
        else if(d.identity.kind==PointSelection::connectionPitch)for(const auto& p:d.edit.pitchPoints)d.original.emplace_back(p.timeSeconds,p.targetMidi);
        else if(d.identity.kind==PointSelection::amplitude)for(const auto& p:d.edit.amplitudePoints)d.original.emplace_back(p.timeSeconds,backend::envelopeGainFromDb(p.gainDb));
        else if(d.identity.kind==PointSelection::flag)for(const auto& p:d.edit.flagPoints)d.original.emplace_back(p.timeSeconds,p.value);
        else for(const auto& p:d.edit.cvvcPoints)d.original.emplace_back(p.position*d.length,d.identity.key.isEmpty()?backend::envelopeGainFromDb(p.value)*d.base:p.value);
        std::erase_if(d.indices,[&](int i){return i<0||i>=(int)d.original.size();});
    }
    std::erase_if(curves,[](const auto& d){return d.indices.empty();});return curves;
}

void PianoRollComponent::movePointGroup(const juce::MouseEvent& event)
{
    auto& gesture=*pointGroupGesture;
    if(gesture.revision!=model.revisionNumber()){pointGroupGesture.reset();cancelPointDrag();clearPointSelection();return;}
    PointGroupDelta limits;
    for(const auto& d:gesture.curves)
    {
        std::vector<PointGroupCoordinate> ps;for(const auto& p:d.original)ps.push_back({p.first,p.second});
        limits.constrain(ps,d.indices,d.first,d.last,d.low,d.high,d.separation,d.fixedEnds);
        if(d.identity.kind==PointSelection::pitch)
        {
            const auto* ownTrack=trackOf(d.identity.noteId);
            const auto* own=ownTrack?sharedLinesFor(*ownTrack).memberFor(d.identity.noteId):nullptr;
            if(own)for(const auto& hit:noteHits)if(hit.id!=d.identity.noteId&&trackOf(hit.id)==ownTrack)
            {
                const auto* other=sharedLinesFor(*ownTrack).memberFor(hit.id);const auto* note=findNote(hit.id);
                if(!other||!note||other->line!=own->line)continue;
                const auto draft=std::find_if(gesture.curves.begin(),gesture.curves.end(),[&](const auto& c){return c.identity.kind==PointSelection::pitch&&c.identity.noteId==hit.id;});
                const auto& anchors=pitchAnchorsFor(*note);
                for(size_t j=0;j<anchors.size();++j)
                {
                    if(draft!=gesture.curves.end()&&std::find(draft->indices.begin(),draft->indices.end(),(int)j)!=draft->indices.end())continue;
                    const auto absolute=hit.startSeconds+hit.clipStartSeconds+anchors[j].timeSeconds;
                    if(!pitchHandleOffered(*note,absolute))continue;
                    for(int i:d.indices)
                    {
                        const auto distance=absolute-(d.start+d.original[(size_t)i].first);
                        if(distance<0)limits.minTime=std::max(limits.minTime,std::min(0.0,distance+minimumAnchorSeparation));
                        if(distance>0)limits.maxTime=std::min(limits.maxTime,std::max(0.0,distance-minimumAnchorSeparation));
                    }
                }
            }
        }
    }
    const auto pixels=event.position-gesture.down;
    const bool amplitude=tool==Tool::amplitude;
    const double dy=tool==Tool::points?-pixels.y/rowHeight:amplitude
        ?-pixels.y/std::max(1.f,amplitudeLanePlotBounds().getHeight())*amplitudeLaneMaxPercent/100.0
        :flagValueFromLaneY(event.position.y)-flagValueFromLaneY(gesture.down.y);
    const auto scale=event.mods.isShiftDown()?.15:1.;
    const auto delta=limits.clamp(pixels.x/pixelsPerSecond*scale,dy*scale);
    gesture.changed=std::abs(delta.time)>1.e-8||std::abs(delta.value)>1.e-8;
    for(auto& d:gesture.curves)d.edit.authorPitchSupport=false;
    for(auto& d:gesture.curves)for(int i:d.indices)
    {
        const auto original=d.original[(size_t)i];const auto t=original.first+delta.time,v=original.second+delta.value;
        if(d.identity.kind==PointSelection::pitch)
        {
            auto& p=d.edit.pitchPoints[(size_t)i];p.timeSeconds=t;p.targetMidi=(float)v;p.diffSingerRestoreSupport=false;
            if(gesture.changed)if(const auto* note=findNote(d.identity.noteId))
                d.edit.authorPitchSupport|=std::min(original.first,t)<-1.e-9 || std::max(original.first,t)>note->durationSeconds+1.e-9;
        }
        else if(d.identity.kind==PointSelection::connectionPitch)
        {auto& p=d.edit.pitchPoints[(size_t)i];p.timeSeconds=t;if(i>0&&i+1<(int)d.edit.pitchPoints.size())p.targetMidi=(float)v;}
        else if(d.identity.kind==PointSelection::amplitude){auto& p=d.edit.amplitudePoints[(size_t)i];p.timeSeconds=t;p.gainDb=backend::envelopeDbFromGain((float)v);}
        else if(d.identity.kind==PointSelection::flag){auto& p=d.edit.flagPoints[(size_t)i];p.timeSeconds=t;p.value=(float)v;}
        else {auto& p=d.edit.cvvcPoints[(size_t)i];p.position=t/d.length;p.value=d.identity.key.isEmpty()?backend::envelopeDbFromGain((float)(v/d.base)):(float)v;}
    }
    if(std::any_of(gesture.curves.begin(),gesture.curves.end(),[](const auto& d){return d.identity.kind==PointSelection::connectionPitch;}))
    {std::vector<ProjectModel::PointCurveEdit> edits;for(const auto& d:gesture.curves)edits.push_back(d.edit);updatePitchPlanPreview(edits,gesture.revision);}
    repaint();
}

void PianoRollComponent::commitPointGroup(std::vector<PointCurveDraft> curves,std::uint64_t revision)
{
    std::vector<ProjectModel::PointCurveEdit> edits;
    for(auto& d:curves)
    {
        if(d.identity.kind==PointSelection::amplitude)
        {
            std::vector<juce::String> ids{d.identity.noteId};
            if(const auto* member=nativeEnvelopeMember(d.identity.noteId))
                for(const auto& [trackId,members]:nativeEnvelopeCache)for(const auto& [id,other]:members)
                    if(other.points==member->points&&id!=d.identity.noteId)ids.push_back(id);
            for(const auto& id:ids)
            {
                auto e=d.edit;e.noteId=id;e.amplitudePoints=mapAmplitudeEnvelopeToNote(d.edit.amplitudePoints,d.identity.noteId,id);
                if(const auto* member=nativeEnvelopeMember(id))
                {for(auto& p:e.amplitudePoints)p.timeSeconds+=member->noteStart;e.amplitudePoints=sliceNativeSharedEnvelope(e.amplitudePoints,member->noteStart,member->noteEnd);}
                if(const auto* note=findNote(id))e.amplitudePoints=unscaledAmplitudeEnvelope(e.amplitudePoints,note->amplitudeEnvelopeBasePercent);
                edits.push_back(std::move(e));
            }
        }
        else edits.push_back(std::move(d.edit));
    }
    if(model.setPointCurves(std::move(edits),revision))
    {
        if(!selectedPoints.empty())
        {
            selectedPoints.clear();
            for(const auto& d:curves)for(int i:d.indices){auto p=d.identity;p.index=i;p.revision=model.revisionNumber();selectedPoints.push_back(p);}
            if(!selectedPoints.empty())selectedPoint=selectedPoints.back();
            if(selectedPoints.size()==1)selectedPoints.clear();
        }
        if(selectedPoint)selectedPoint->revision=model.revisionNumber();
        refreshFromModel();repaint();
    }
}

void PianoRollComponent::editSelectedPoints(bool initialize)
{
    pointGroupGesture.reset();cancelPointDrag();
    auto curves=pointCurveDrafts();const auto revision=model.revisionNumber();
    for(auto& d:curves)
    {
        std::sort(d.indices.begin(),d.indices.end(),std::greater<int>());
        bool changed=false;const auto* note=findNote(d.identity.noteId);
        for(int index:d.indices)
        {
            const auto i=(size_t)index;
            const auto edit=[&](auto& ps,auto reset,bool endpoints)
            {
                if(i>=ps.size())return;
                if(initialize){reset(ps[i]);changed=true;}
                else if(ps.size()>1&&(!endpoints||(i>0&&i+1<ps.size()))){ps.erase(ps.begin()+index);changed=true;}
            };
            if(d.identity.kind==PointSelection::pitch)edit(d.edit.pitchPoints,[&](auto& p){p.targetMidi=(float)note->midiNote;p.diffSingerRestoreSupport=false;},true);
            else if(d.identity.kind==PointSelection::amplitude)edit(d.edit.amplitudePoints,[](auto& p){p.gainDb=0;},true);
            else if(d.identity.kind==PointSelection::flag)edit(d.edit.flagPoints,[&](auto& p){p.value=d.identity.key.startsWith("DS:ABS:")
                ?flagCurveValueAt(flagCurvePointsFor(*note,"DS:REF:"+d.identity.key.substring(7)),p.timeSeconds):flagCurveKindFor(d.identity.key).defaultValue;},false);
            else if(d.identity.kind==PointSelection::connectionPitch)
            {if(!initialize&&i>0&&i+1<d.edit.pitchPoints.size()){d.edit.pitchPoints.erase(d.edit.pitchPoints.begin()+index);changed=true;}}
            else edit(d.edit.cvvcPoints,[&](auto& p){p.value=d.identity.key.isEmpty()?backend::envelopeDbFromGain((float)(1/d.base)):flagCurveKindFor(d.identity.key).defaultValue;},true);
        }
        if(!changed)d.indices.clear();
    }
    std::erase_if(curves,[](const auto& d){return d.indices.empty();});
    if(!initialize){selectedPoints.clear();if(selectedPoint)selectedPoint->index=-1;}
    commitPointGroup(std::move(curves),revision);repaint();
}
}
