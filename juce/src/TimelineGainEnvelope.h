#pragma once

namespace hachi
{
inline juce::Rectangle<float> TimelineComponent::envelopeArea(std::size_t index) const
{
    const auto rowY=static_cast<float>(rulerHeight+static_cast<int>(index)*rowHeight);
    const auto top=rowY+std::min(37.0f,static_cast<float>(rowHeight)-13.0f);
    const auto bottom=rowY+static_cast<float>(rowHeight)-8.0f;
    return {0.0f,top,static_cast<float>(getWidth()),bottom-top};
}
inline const ClipData* TimelineComponent::findEnvelopeClip(const juce::String& id) const
{
    for(const auto& track:snapshot.tracks)for(const auto& clip:track.clips)if(clip.id==id)return &clip;
    return nullptr;
}
inline const std::vector<TrackGainPoint>& TimelineComponent::displayedEnvelope(const ClipData& clip) const
{ return clip.id==draggedEnvelopeClip ? envelopePreview : clip.gainEnvelope; }
inline double TimelineComponent::envelopeDisplayOrigin(const ClipData& clip) const
{
    const auto moving=dragMode==DragMode::move && draggedClip.isNotEmpty() && isClipSelected(clip.id);
    return clip.startSeconds+(moving?draggedClipPreviewStart-draggedClipStart:0.0);
}
inline std::optional<TimelineComponent::EnvelopeHit> TimelineComponent::envelopeHit(juce::Point<float> pos,bool includeLine) const
{
    const auto row=static_cast<int>(std::floor((pos.y-rulerHeight)/rowHeight));
    if(row<0 || row>=static_cast<int>(snapshot.tracks.size()))return std::nullopt;
    const auto index=static_cast<std::size_t>(row);
    // Match the topmost region. No envelope exists in empty timeline space.
    for(auto hit=clipHits.rbegin();hit!=clipHits.rend();++hit)
    {
        if(!hit->bounds.contains(pos))continue;
        const auto* clip=findEnvelopeClip(hit->id);
        if(clip==nullptr)return std::nullopt;
        const auto& points=displayedEnvelope(*clip);
        const auto area=envelopeArea(index);
        const auto yFor=[&](float db){return area.getCentreY()-trackGainLevelFromDb(db)*area.getHeight()*.5f;};
        int nearest=-1;float distance=6.0f;
        for(std::size_t i=0;i<points.size();++i)
        {
            if(points[i].timeSeconds<0 || points[i].timeSeconds>clip->durationSeconds)continue;
            const auto d=pos.getDistanceFrom({timeToX(clip->startSeconds+points[i].timeSeconds),yFor(points[i].gainDb)});
            if(d<=distance){distance=d;nearest=static_cast<int>(i);}
        }
        if(nearest>=0)return EnvelopeHit{clip->id,index,nearest};
        if(!includeLine)return std::nullopt;
        const auto grip=std::min(14.0f,hit->bounds.getWidth()*.25f);
        const auto waveTop=hit->bounds.getY()+19.0f;
        if(gainControlBounds(hit->bounds).contains(pos) || pos.y<hit->bounds.getY()+17.0f
            || pos.x<=hit->bounds.getX()+grip || pos.x>=hit->bounds.getRight()-grip
            || pos.getDistanceFrom({timeToX(hit->audioStartSeconds+hit->fadeInSeconds),waveTop})<=5.0f
            || pos.getDistanceFrom({timeToX(hit->audioStartSeconds+hit->audioDurationSeconds-hit->fadeOutSeconds),waveTop})<=5.0f)
            return std::nullopt;
        const auto db=trackGainEnvelopeDbAt(points,static_cast<double>(pos.x)/pixelsPerSecond-clip->startSeconds);
        if(std::abs(pos.y-yFor(db))<=4.0f)return EnvelopeHit{clip->id,index,-1};
        return std::nullopt;
    }
    return std::nullopt;
}
inline void TimelineComponent::paintClipEnvelope(juce::Graphics& g,const ClipData& clip,bool trackMuted,std::size_t index)
{
    const auto area=envelopeArea(index);
    const auto origin=envelopeDisplayOrigin(clip);
    const auto start=clip.id==draggedClip?draggedClipPreviewStart:origin;
    const auto duration=clip.id==draggedClip?draggedClipPreviewDuration:clip.durationSeconds;
    const auto firstX=timeToX(start),lastX=timeToX(start+duration);
    const juce::Graphics::ScopedSaveState save(g);
    g.reduceClipRegion(juce::Rectangle<float>(firstX,static_cast<float>(rulerHeight+static_cast<int>(index)*rowHeight+19),
        std::max(1.0f,lastX-firstX),static_cast<float>(rowHeight-25)).getSmallestIntegerContainer());
    const auto visible=g.getClipBounds();
    if(visible.isEmpty())return;
    const auto& points=displayedEnvelope(clip);
    const auto yFor=[&](float db){return area.getCentreY()-trackGainLevelFromDb(db)*area.getHeight()*.5f;};
    const auto left=std::max(firstX,static_cast<float>(visible.getX())),right=std::min(lastX,static_cast<float>(visible.getRight()));
    const auto first=static_cast<double>(left)/pixelsPerSecond-origin;
    const auto last=static_cast<double>(right)/pixelsPerSecond-origin;
    juce::Path path;path.startNewSubPath(left,yFor(trackGainEnvelopeDbAt(points,first)));
    for(const auto& point:points)if(point.timeSeconds>first&&point.timeSeconds<last)
        path.lineTo(timeToX(origin+point.timeSeconds),yFor(point.gainDb));
    path.lineTo(right,yFor(trackGainEnvelopeDbAt(points,last)));
    const auto light=Palette::background.getPerceivedBrightness()>.5f;
    const auto rowSurface=std::find(selectedTracks.begin(),selectedTracks.end(),snapshot.tracks[index].id)!=selectedTracks.end()
        ?Palette::base.overlaidWith(Palette::accent.withAlpha(.06f)):Palette::base;
    const auto clipSurface=rowSurface.overlaidWith(trackMuted||clip.muted
        ?Palette::clipBackground.withAlpha(.55f):Palette::clipBackground);
    auto curveInk=Palette::accent.withAlpha(trackMuted||clip.muted?.25f:points.empty()?.38f:.85f);
    if(light&&!points.empty())
    {
        curveInk=trackMuted||clip.muted?Palette::textMuted:Palette::noteEdge;
        if(Palette::contrastRatio(curveInk,clipSurface)<3.0)curveInk=Palette::contrastingText(clipSurface);
    }
    g.setColour(curveInk);
    g.strokePath(path,juce::PathStrokeType(1.15f));
    for(std::size_t i=0;i<points.size();++i)
    {
        const auto& point=points[i];
        const auto x=timeToX(origin+point.timeSeconds),y=yFor(point.gainDb);
        if(point.timeSeconds<first-1.0e-6 || point.timeSeconds>last+1.0e-6)continue;
        const auto selected=clip.id==selectedEnvelopeClip && (clip.id==draggedEnvelopeClip
            ? static_cast<int>(i)==draggedEnvelopePoint : std::abs(point.timeSeconds-selectedEnvelopeTime)<1.0e-6);
        const auto radius=selected?4.2f:3.2f;
        g.setColour(Palette::panel);g.fillEllipse(x-radius,y-radius,radius*2,radius*2);
        g.setColour(light?(selected?Palette::heading:Palette::noteEdge):(selected?Palette::text:Palette::accent));
        g.drawEllipse(x-radius,y-radius,radius*2,radius*2,1.3f);
        if(selected)
        {
            const auto label=juce::String(point.timeSeconds,3)+" s  "+gainLabel(juce::Decibels::decibelsToGain(point.gainDb,-60.0f));
            const auto width=std::min(130.0f,lastX-firstX);
            juce::Rectangle<float> box(juce::jlimit(firstX,std::max(firstX,lastX-width),x+7.0f),
                std::max(area.getY(),y-20.0f),width,16.0f);
            g.setColour(Palette::panel.withAlpha(.94f));g.fillRoundedRectangle(box,2.0f);
            const auto labelSurface=clipSurface.overlaidWith(Palette::panel.withAlpha(.94f));
            g.setColour(light?Palette::contrastingText(labelSurface):Palette::text);
            g.setFont(10.5f);g.drawText(label,box.reduced(3,0),juce::Justification::centredLeft,false);
        }
    }
}
inline void TimelineComponent::clearEnvelopeDrag()
{
    draggedEnvelopeClip.clear();draggedEnvelopePoint=-1;
    envelopeBeforeDrag.clear();envelopePreview.clear();setMouseCursor(juce::MouseCursor::NormalCursor);
}
inline void TimelineComponent::editEnvelopePoint(const juce::String& clipId,double time,int action)
{
    const auto id=clipId;clearEnvelopeDrag();const auto data=model.snapshot();
    for(const auto& track:data.tracks)for(const auto& clip:track.clips)if(clip.id==id)
    {
        auto points=clip.gainEnvelope;
        const auto point=std::find_if(points.begin(),points.end(),[&](const auto& p){return std::abs(p.timeSeconds-time)<1.0e-6;});
        if(action==3)points.clear();
        else if(point==points.end())return;
        else if(action==1)points.erase(point);
        else if(action==2)point->gainDb=0.0f;
        model.setClipGainEnvelope(id,std::move(points));
        if(action!=2)selectedEnvelopeClip.clear();repaint();return;
    }
}
inline void TimelineComponent::showEnvelopePointMenu(const EnvelopeHit& hit,juce::Point<int> screen)
{
    clearEnvelopeDrag();const auto* clip=findEnvelopeClip(hit.clipId);if(clip==nullptr)return;
    const auto time=clip->gainEnvelope[static_cast<std::size_t>(hit.pointIndex)].timeSeconds;
    juce::PopupMenu menu;
    menu.addItem(1,envelopeMenuTexts.size()>0?envelopeMenuTexts[0]:"Delete envelope point");
    menu.addItem(2,envelopeMenuTexts.size()>1?envelopeMenuTexts[1]:"Reset point to 0 dB");
    menu.addSeparator();menu.addItem(3,envelopeMenuTexts.size()>2?envelopeMenuTexts[2]:"Reset region envelope");
    juce::Component::SafePointer<TimelineComponent> safe(this);
    menu.showMenuAsync(juce::PopupMenu::Options().withTargetScreenArea({screen.x,screen.y,1,1}),
        [safe,id=hit.clipId,time](int result){if(safe!=nullptr&&result>0)safe->editEnvelopePoint(id,time,result);});
}
}