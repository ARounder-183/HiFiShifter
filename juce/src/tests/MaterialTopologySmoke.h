#pragma once
#include "../ProjectModel.h"
#include "../SampleSettings.h"
#include "../NativeNoteTiming.h"
#include "../NativePitchIdentity.h"
#include <iostream>
#include <set>
#include <array>

namespace hachi
{
inline bool runMaterialTopologySmoke(const juce::File& output)
{
    if(output.createDirectory().failed())return false;
    bool ok=true;juce::String log;
    const auto check=[&](const char* name,bool pass){ok&=pass;log+=juce::String(name)+"="+(pass?"PASS":"FAIL")+"\n";std::cout<<name<<'='<<pass<<std::endl;};
    const auto near=[](double a,double b){return std::abs(a-b)<1.e-7;};
    const auto source=output.getChildFile("topology-source.wav");
    juce::AudioBuffer<float> pcm(1,115200);pcm.clear();juce::WavAudioFormat format;
    auto stream=source.createOutputStream();if(!stream)return false;stream->setPosition(0);stream->truncate();
    std::unique_ptr<juce::AudioFormatWriter> writer(format.createWriterFor(stream.get(),48000,1,16,{},0));
    if(!writer)return false;stream.release();check("wav_fixture",writer->writeFromAudioSampleBuffer(pcm,0,pcm.getNumSamples()));writer.reset();
    juce::MemoryBlock originalBytes;check("source_bytes_captured",source.loadFileAsData(originalBytes));
    const auto row=[](const char* id,double start,double end){SampleRegionSetting r;r.materialAudioId="audio";r.materialRegionId=id;r.name=id;r.regionStartSeconds=start;r.regionEndSeconds=end;r.alignmentSeconds=start;r.role=NativeSegmentRole::vowel;r.provenance="user";return r;};
    auto root=row("root",.2,1.7),sibling=row("sibling",1.7,2.2);root.connectedNextRegionId=sibling.materialRegionId;
    const auto note=[](const SampleRegionSetting& r,const char* id,double start,double duration)
    {
        NoteData n;n.id=id;n.label=r.name;n.startSeconds=start;n.durationSeconds=duration;n.nativeRole=r.role;n.nativeProvenance=r.provenance;
        n.materialAudioId=r.materialAudioId;n.materialRegionId=r.materialRegionId;n.nativeSourceStartSeconds=n.materialSourceStartSeconds=r.regionStartSeconds;
        n.nativeSourceEndSeconds=n.materialSourceEndSeconds=r.regionEndSeconds;n.materialConnectedNextRegionId=r.connectedNextRegionId;n.nativeSegments=SampleSettings::nativeSegmentsFor(r);
        n.midiNote=63;n.sourceMidiCenter=60;n.sourcePitchMeasured=true;n.gain=.36f;
        n.pitchControlPoints={{0,63},{duration*.5,67,PitchCurveShape::customBezier,.1f,.8f,.9f,.2f},{duration*.8,62,PitchCurveShape::easeOut},{duration,64,PitchCurveShape::natural}};
        n.contour={{0,10,4,true},{duration*.4,50,7,true},{duration,-20,-3,true}};
        n.amplitudeEnvelope={{0,-12,true},{duration*.233,-2,true},{duration,-9}};
        n.vibratoEnabled=true;n.vibratoDepthCents=63;n.vibratoReferenceDurationSeconds=duration;n.vibratoTimeOffsetSeconds=.17;return n;
    };
    ClipData clip;clip.id="placement";clip.sourceFile=source;clip.startSeconds=1;clip.sourceOffsetSeconds=.2;clip.sourceDurationSeconds=2;clip.durationSeconds=4;
    clip.sourceTimeMap={{0,0},{.8,.3},{2,1},{3,1.5},{4,2}};clip.notes={note(root,"root-one",0,3),note(sibling,"sibling-one",3,1)};
    clip.notes[0].connectedToNext=true;clip.notes[1].connectedToPrevious=true;
    auto repeat=clip;repeat.id="repeat";repeat.startSeconds=7;repeat.notes[0].id="root-two";repeat.notes[1].id="sibling-two";
    TrackData track;track.id="native";track.pitchAlgorithm=PitchAlgorithm::world;track.clips={clip,repeat};ProjectData fixture;fixture.tracks={track};
    fixture.nativeConnections={{"link-one","root-one","sibling-one","pitch-and-amplitude",4,{{4,65}},{{4,-3}}},{"link-two","root-two","sibling-two","pitch-and-amplitude",10,{},{}}};
    auto left=root;left.regionEndSeconds=.7;left.connectedNextRegionId="new-middle";
    auto middle=row("new-middle",.7,1.2),right=row("new-right",1.2,1.7);middle.connectedNextRegionId=right.materialRegionId;right.connectedNextRegionId=sibling.materialRegionId;
    std::vector<SampleRegionSetting> splitRows{right,sibling,left,middle};ProjectModel model;model.resetDocument(fixture);
    const auto beforeFingerprint=model.contentFingerprint();const auto beforeRevision=model.revisionNumber();
    auto report=model.applySourceSettingsWithReport(source,splitRows,false);
    check("topology_preflight_non_mutating",!report.rejected&&report.changed()&&model.contentFingerprint()==beforeFingerprint&&model.revisionNumber()==beforeRevision&&!model.canUndo());
    report=model.applySourceSettingsWithReport(source,splitRows);auto after=model.snapshot();
    check("repeated_placements_expand_three_members_atomically",!report.rejected&&report.changed()&&after.tracks[0].clips[0].notes.size()==4&&after.tracks[0].clips[1].notes.size()==4&&model.revisionNumber()==beforeRevision+1);
    if(report.rejected||after.tracks[0].clips[0].notes.size()!=4){log+=report.diagnostics.joinIntoString("\n");output.getChildFile("material-topology-smoke.txt").replaceWithText(log);return false;}
    const auto& expanded=after.tracks[0].clips[0];
    check("stable_original_id_new_scoped_members",expanded.notes[0].id=="root-one"&&expanded.notes[0].materialRegionId=="root"&&expanded.notes[1].materialRegionId=="new-middle"&&expanded.notes[2].materialRegionId=="new-right"&&expanded.notes[3].id=="sibling-one"&&expanded.notes[1].id!=after.tracks[0].clips[1].notes[1].id);
    check("inverse_old_clock_target_seams",near(expanded.notes[1].startSeconds,nativeTargetTimeAt(clip.sourceTimeMap,.5))&&near(expanded.notes[2].startSeconds,2)&&near(expanded.notes[2].startSeconds+expanded.notes[2].durationSeconds,3));
    bool identicalClock=expanded.sourceTimeMap.size()==clip.sourceTimeMap.size();
    for(std::size_t i=0;i<clip.sourceTimeMap.size()&&identicalClock;++i)identicalClock&=expanded.sourceTimeMap[i].targetSeconds==clip.sourceTimeMap[i].targetSeconds&&expanded.sourceTimeMap[i].sourceSeconds==clip.sourceTimeMap[i].sourceSeconds;
    check("source_span_warp_timeline_unchanged",identicalClock&&expanded.sourceOffsetSeconds==clip.sourceOffsetSeconds&&expanded.sourceDurationSeconds==clip.sourceDurationSeconds&&expanded.startSeconds==clip.startSeconds&&expanded.durationSeconds==clip.durationSeconds);
    const auto compare=[&](const TrackData& oldTrack,const TrackData& newTrack,const char* label)
    {
        const auto oldLines=sharedPitchLines(oldTrack),newLines=sharedPitchLines(newTrack);const auto oldEnv=nativeSharedEnvelopes(oldTrack),newEnv=nativeSharedEnvelopes(newTrack);
        const auto find=[](const TrackData& t,double time)->std::pair<const ClipData*,const NoteData*>{for(const auto& c:t.clips)for(const auto& n:c.notes)if(time>=c.startSeconds+n.startSeconds-1.e-10&&time<c.startSeconds+n.startSeconds+n.durationSeconds-1.e-10)return {&c,&n};return {nullptr,nullptr};};
        const auto values=[&](const auto& at,const SharedPitchLines& lines,const NativeSharedEnvelopes& env,double time)
        {
            const auto& c=*at.first;const auto& n=*at.second;const auto local=time-c.startSeconds-n.startSeconds;const auto* member=lines.memberFor(n.id);
            const auto base=member&&member->renderSharedCurve?member->line->midiAt(time):evaluatePitchCurve(ownPitchPoints(n),local);const auto e=env.find(n.id);
            const auto db=e!=env.end()?nativeEnvelopeDbAt(*e->second.points,time):nativeEnvelopeDbAt(scaledAmplitudeEnvelope(n.amplitudeEnvelope,n.amplitudeEnvelopeBasePercent),local);
            bool voiced=!n.nativeUnpitched;float sourceCents=0;
            if(voiced&&!n.contour.empty())
            {
                auto r=std::lower_bound(n.contour.begin(),n.contour.end(),local,[](const auto& p,double t){return p.timeSeconds<t;});
                const auto ri=r==n.contour.end()?n.contour.size()-1:static_cast<std::size_t>(r-n.contour.begin());
                const auto li=ri>0&&n.contour[ri].timeSeconds>local?ri-1:ri;
                const auto& a=n.contour[li];const auto& b=n.contour[ri];voiced=a.voiced&&b.voiced;
                const auto u=b.timeSeconds>a.timeSeconds?static_cast<float>((local-a.timeSeconds)/(b.timeSeconds-a.timeSeconds)):0.f;
                sourceCents=a.relativeCents+(b.relativeCents-a.relativeCents)*u;
            }
            return std::array<double,6>{base+vibratoCentsAt(n,local)/100,db,vibratoCentsAt(n,local),n.gain,
                (n.sourceMidiCenter>=0?n.sourceMidiCenter:n.midiNote)+sourceCents/100.0,voiced?1.0:0.0};
        };
        double pitchError=0,envelopeError=0,vibratoError=0,gainError=0,sourceError=0;bool coverage=true,masksEqual=true;
        for(const auto& c:oldTrack.clips)for(int i=0;i<static_cast<int>(std::ceil(c.durationSeconds*1000));++i)
        {
            const auto time=c.startSeconds+i*.001+.0000001;const auto a=find(oldTrack,time),b=find(newTrack,time);if(!a.second)continue;if(!b.second){coverage=false;continue;}
            const auto x=values(a,oldLines,oldEnv,time),y=values(b,newLines,newEnv,time);
            pitchError=std::max(pitchError,std::abs(x[0]-y[0]));envelopeError=std::max(envelopeError,std::abs(x[1]-y[1]));vibratoError=std::max(vibratoError,std::abs(x[2]-y[2]));gainError=std::max(gainError,std::abs(x[3]-y[3]));
            masksEqual&=x[5]==y[5];if(x[5]&&y[5])sourceError=std::max(sourceError,std::abs(x[4]-y[4]));
        }
        log+=juce::String(label)+" maxMidiError="+juce::String(pitchError,12)+" maxDbError="+juce::String(envelopeError,12)+" maxVibratoCentsError="+juce::String(vibratoError,12)
            +" maxSourceMidiError="+juce::String(sourceError,12)+" voicedMaskEqual="+juce::String(masksEqual?1:0)+"\n";
        check(label,coverage&&masksEqual&&sourceError<.000002&&pitchError<.0002&&envelopeError<.00005&&vibratoError<.00000001&&gainError==0);
    };
    compare(fixture.tracks[0],after.tracks[0],"every_1ms_pitch_envelope_gain_vibrato_equivalent");
    bool localHandles=true,uniqueHandles=true;std::set<juce::int64> handles;const auto lines=sharedPitchLines(after.tracks[0]);
    for(const auto& c:after.tracks[0].clips)for(const auto& n:c.notes){for(const auto& p:n.pitchControlPoints)localHandles&=p.timeSeconds>=-1.e-8&&p.timeSeconds<=n.durationSeconds+1.e-8;if(const auto* member=lines.memberFor(n.id))for(const auto t:member->nativeHandleTimes)uniqueHandles&=handles.insert(static_cast<juce::int64>(std::llround(t*10000000))).second;}
    check("no_hidden_handles_or_duplicate_writable_authorities",localHandles&&uniqueHandles);
    const auto outgoing=std::find_if(after.nativeConnections.begin(),after.nativeConnections.end(),[](const auto& c){return c.id=="link-one";});
    check("outgoing_graph_migrates_preserving_curve",after.nativeConnections.size()==6&&outgoing!=after.nativeConnections.end()&&outgoing->leftNoteId==expanded.notes[2].id&&outgoing->rightNoteId=="sibling-one"&&near(outgoing->boundarySeconds,4)&&outgoing->pitchCurve.size()==1&&outgoing->amplitudeCurve.size()==1);
    check("graph_projects_all_internal_flags",!expanded.notes[0].connectedToPrevious&&expanded.notes[0].connectedToNext&&expanded.notes[1].connectedToPrevious&&expanded.notes[1].connectedToNext&&expanded.notes[2].connectedToPrevious&&expanded.notes[2].connectedToNext&&expanded.notes[3].connectedToPrevious&&!expanded.notes[3].connectedToNext);
    const auto afterFingerprint=model.contentFingerprint();report=model.applySourceSettingsWithReport(source,splitRows);
    check("repeated_refresh_noop",!report.rejected&&!report.changed()&&model.contentFingerprint()==afterFingerprint);
    check("single_undo_all_placements",model.undo()&&!model.canUndo()&&model.contentFingerprint()==beforeFingerprint);
    check("redo_topology_graph",model.redo()&&model.contentFingerprint()==afterFingerprint);
    juce::String error;ProjectModel reopened;const auto saved=output.getChildFile("topology-expanded.hachi");
    check("identity_automation_graph_save_load",model.save(saved,error)&&reopened.load(saved,error)&&reopened.snapshot().tracks[0].clips[0].notes[1].materialRegionId=="new-middle"&&reopened.snapshot().nativeConnections.size()==6);
    compare(fixture.tracks[0],reopened.snapshot().tracks[0],"every_1ms_reopen_equivalent");
    for(const float base:{50.f,200.f})
    {
        auto based=fixture;for(auto& c:based.tracks[0].clips)for(auto& n:c.notes)n.amplitudeEnvelopeBasePercent=base;
        model.resetDocument(based);report=model.applySourceSettingsWithReport(source,splitRows);
        const auto basedAfter=model.snapshot();
        check(base==50?"base50_preserved_without_saturation":"base200_preserved_without_saturation",!report.rejected
            &&basedAfter.tracks[0].clips[0].notes.size()==4&&basedAfter.tracks[0].clips[0].notes[1].amplitudeEnvelopeBasePercent==base);
        compare(based.tracks[0],basedAfter.tracks[0],base==50?"every_1ms_base50_equivalent":"every_1ms_base200_equivalent");
    }
    auto derived=fixture;
    for(auto& c:derived.tracks[0].clips)for(auto& n:c.notes)
    {
        n.pitchControlPoints.clear();n.nativePitchHandlesPlaced=false;n.drift=.7f;n.modulation=.4f;n.contour.clear();
        for(int i=0;i<=static_cast<int>(std::llround(n.durationSeconds*200));++i)
        {
            const auto t=i*.005;PitchPoint p{t,static_cast<float>(30*std::sin(t*7)),static_cast<float>(10*std::sin(t*2)),true};
            p.hasManualTarget=(i%2)==0;p.manualTargetCents=static_cast<float>(40*std::cos(t*5));n.contour.push_back(p);
        }
    }
    model.resetDocument(derived);report=model.applySourceSettingsWithReport(source,splitRows);
    const auto derivedAfter=model.snapshot();
    check("measured_automatic_partition_keeps_source_samples_read_only",!report.rejected&&derivedAfter.tracks[0].clips[0].notes.size()==4
        &&std::all_of(derivedAfter.tracks[0].clips[0].notes.begin(),derivedAfter.tracks[0].clips[0].notes.end(),[](const auto& n)
        {return n.pitchControlPoints.empty()&&!n.nativePitchHandlesPlaced&&n.sourcePitchMeasured;})
        &&derivedAfter.tracks[0].clips[0].notes[1].contour.size()<derived.tracks[0].clips[0].notes[0].contour.size());
    compare(derived.tracks[0],derivedAfter.tracks[0],"every_1ms_derived_drift_modulation_vibrato_and_source_equivalent");
    auto voiced=fixture;
    for(auto& c:voiced.tracks[0].clips)c.notes[0].contour={{0,10,4,true},{1.5,30,7,true},{2,35,8,true},{2.1,0,0,false},{2.2,45,9,true},{3,-20,-3,true}};
    model.resetDocument(voiced);report=model.applySourceSettingsWithReport(source,splitRows);
    const auto voicedAfter=model.snapshot();
    check("exact_voiced_knot_followed_by_unvoiced_preserved",!report.rejected&&voicedAfter.tracks[0].clips[0].notes.size()==4
        &&voicedAfter.tracks[0].clips[0].notes[1].contour.back().voiced&&voicedAfter.tracks[0].clips[0].notes[2].contour.front().voiced);
    compare(voiced.tracks[0],voicedAfter.tracks[0],"every_1ms_voiced_mask_and_source_f0_equivalent");
    auto subset=fixture;subset.nativeConnections.clear();subset.tracks[0].clips.resize(1);auto& only=subset.tracks[0].clips[0];only.notes.resize(1);only.notes[0].connectedToNext=false;only.durationSeconds=3;only.sourceDurationSeconds=1.5;only.sourceTimeMap.pop_back();
    model.resetDocument(subset);report=model.applySourceSettingsWithReport(source,splitRows);const auto subsetAfter=model.snapshot();
    check("single_pronunciation_only_original_interval",!report.rejected&&subsetAfter.tracks[0].clips[0].notes.size()==3&&subsetAfter.nativeConnections.size()==2&&near(subsetAfter.tracks[0].clips[0].sourceDurationSeconds,1.5)&&near(subsetAfter.tracks[0].clips[0].notes.back().nativeSourceEndSeconds,1.7));
    compare(subset.tracks[0],subsetAfter.tracks[0],"every_1ms_subset_equivalent");
    auto neutral=subset;
    auto& neutralNote=neutral.tracks[0].clips[0].notes[0];
    neutralNote.sourcePitchMeasured=false;neutralNote.sourceMidiCenter=-1;neutralNote.midiNote=60;
    neutralNote.vibratoEnabled=false;neutralNote.pitchControlPoints.clear();neutralNote.contour={{0,0,0,true},{3,0,0,true}};
    check("neutral_fixture_pcm_predicate",nativePendingPitchIsNeutral(neutralNote));
    model.resetDocument(neutral);report=model.applySourceSettingsWithReport(source,splitRows);
    const auto neutralAfter=model.snapshot();
    check("neutral_topology_does_not_invent_measured_f0_or_handles",!report.rejected&&neutralAfter.tracks[0].clips[0].notes.size()==3
        &&std::all_of(neutralAfter.tracks[0].clips[0].notes.begin(),neutralAfter.tracks[0].clips[0].notes.end(),[](const auto& n)
        {return nativePendingPitchIsNeutral(n)&&!n.sourcePitchMeasured&&n.pitchControlPoints.empty();}));
    compare(neutral.tracks[0],neutralAfter.tracks[0],"every_1ms_neutral_pitch_and_expression_equivalent");
    const auto rejectUnchanged=[&](ProjectData data,const std::vector<SampleRegionSetting>& offered,const char* name){model.resetDocument(std::move(data));const auto f=model.contentFingerprint();const auto r=model.revisionNumber();const auto p=model.applySourceSettingsWithReport(source,offered,false);const auto a=model.applySourceSettingsWithReport(source,offered);check(name,p.rejected&&a.rejected&&!a.changed()&&model.contentFingerprint()==f&&model.revisionNumber()==r&&!model.canUndo());};
    auto mixedAuthority=fixture;mixedAuthority.tracks[0].clips[0].notes[1].pitchControlPoints.clear();
    rejectUnchanged(mixedAuthority,splitRows,"mixed_authored_derived_authority_explicitly_blocked");
    auto mixedPending=fixture;mixedPending.tracks[0].clips[0].notes[0]=neutralNote;mixedPending.tracks[0].clips[0].notes[1].pitchControlPoints.clear();
    rejectUnchanged(mixedPending,splitRows,"mixed_pending_source_requires_analysis_before_partition");
    auto local=fixture;local.tracks[0].clips[1].notes[0].materialLocalOverride=true;rejectUnchanged(local,splitRows,"later_override_rejects_all_plans_atomically");
    auto cropped=subset;cropped.tracks[0].clips[0].notes[0].startSeconds=.4;cropped.tracks[0].clips[0].notes[0].durationSeconds=2;rejectUnchanged(cropped,splitRows,"cropped_topology_blocked");
    auto merged=fixture;merged.tracks[0].clips.resize(1);merged.nativeConnections.resize(1);auto& parent=merged.tracks[0].clips[0];auto child=parent;child.id="child";child.startSeconds=.5;child.notes.clear();parent.parts={child};parent.durationSeconds=4.5;for(auto& n:parent.notes){n.clipPartId="child";n.startSeconds+=.5;}rejectUnchanged(merged,splitRows,"merged_topology_blocked");
    auto unlinked=splitRows;for(auto& r:unlinked)r.connectedNextRegionId.clear();rejectUnchanged(fixture,unlinked,"unlinked_chain_not_shortened");
    auto saturated=fixture;
    for(auto& c:saturated.tracks[0].clips)
    {
        c.notes[0].amplitudeEnvelopeBasePercent=50;c.notes[0].amplitudeEnvelope={{0,12,false}};
        c.notes[1].amplitudeEnvelopeBasePercent=200;c.notes[1].amplitudeEnvelope={{0,12,false},{.2,12,false},{1,12,false}};
    }
    rejectUnchanged(saturated,splitRows,"mixed_base_saturation_rejected_without_gain_loss");
    auto advanced=fixture;advanced.tracks[0].clips[0].notes[0].nativeEnvelope.mode=1;rejectUnchanged(advanced,splitRows,"advanced_envelope_preflight_blocks");
    auto external=fixture;external.tracks[0].clips[0].notes[0].pitchControlPoints.insert(external.tracks[0].clips[0].notes[0].pitchControlPoints.begin(),{-.1,64});rejectUnchanged(external,splitRows,"outside_owner_handles_not_discarded");
    model.resetDocument(fixture);auto br=root;br.regionEndSeconds=1.6;auto bs=sibling;bs.regionStartSeconds=bs.alignmentSeconds=1.6;report=model.applySourceSettingsWithReport(source,{bs,br});check("existing_boundary_update_still_supported",!report.rejected&&report.changed()&&near(model.snapshot().tracks[0].clips[0].notes[0].nativeSourceEndSeconds,1.6));
    model.resetDocument(fixture);auto outside=row("outside",2.2,2.4);report=model.applySourceSettingsWithReport(source,{outside,sibling,root});check("outside_additions_do_not_expand_coverage",!report.rejected&&!report.changed());
    auto mr=row("merged-id",.2,2.2);report=model.applySourceSettingsWithReport(source,{mr});check("deleted_merge_retains_unresolved_snapshots",!report.rejected&&report.unresolvedNotes==4&&model.snapshot().tracks[0].clips[0].notes.size()==2&&model.snapshot().tracks[0].clips[0].notes[0].materialReferenceUnresolved);
    juce::MemoryBlock afterBytes;check("source_bytes_unchanged",source.loadFileAsData(afterBytes)&&afterBytes==originalBytes);
    check("report_written",output.getChildFile("material-topology-smoke.txt").replaceWithText(log));return ok;
}
}
