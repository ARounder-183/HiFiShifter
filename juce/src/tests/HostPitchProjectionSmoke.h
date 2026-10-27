#pragma once
#include "../AudioEngine.h"
#include "../backend/UtauRenderer.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>

namespace hachi
{
inline bool runHostPitchProjectionSmoke(const juce::File& output)
{
    if(!output.createDirectory())return false;
    bool ok=true;int checks=0;juce::Array<juce::var> results;
    const auto check=[&](const char* name,bool pass){ok&=pass;++checks;std::cout<<name<<'='<<pass<<std::endl;
        auto item=new juce::DynamicObject();item->setProperty("name",name);item->setProperty("pass",pass);results.add(juce::var(item));};
    using Plan=AudioEngine::ProjectedNotePitch;
    const auto near=[](double a,double b){return std::abs(a-b)<1.e-6;};
    const auto find=[](const std::vector<Plan>& plans,const juce::String& id)->const Plan*
    {for(const auto& p:plans)if(p.noteId==id)return &p;return nullptr;};
    const auto sample=[](const std::vector<backend::UtauPitchPoint>& points,double time)
    {
        if(points.empty())return 0.f;
        const auto next=std::lower_bound(points.begin(),points.end(),time,[](const auto& p,double t){return p.timeSeconds<t;});
        if(next==points.begin())return next->cents;if(next==points.end())return points.back().cents;
        const auto& prev=*std::prev(next);const auto u=next->timeSeconds>prev.timeSeconds?static_cast<float>((time-prev.timeSeconds)/(next->timeSeconds-prev.timeSeconds)):0.f;
        return prev.cents+(next->cents-prev.cents)*u;
    };
    const auto same=[](const std::vector<backend::UtauPitchPoint>& a,const std::vector<backend::UtauPitchPoint>& b)
    {
        if(a.size()!=b.size())return false;for(std::size_t i=0;i<a.size();++i)if(a[i].timeSeconds!=b[i].timeSeconds||a[i].cents!=b[i].cents)return false;return true;
    };
    const auto spec=[](const Plan& p)
    {backend::UtauNoteRenderSpec n;n.alias="a";n.startSeconds=p.absoluteStartSeconds;n.durationSeconds=p.durationSeconds;n.midiNote=p.midiNote;n.pitchCurve=p.sampledLocalCents;return n;};
    const auto wire=[&](const Plan& p,double pre=.1,double seconds=1.25){return backend::UtauRenderer::diagnosticPitchbend(spec(p),120,pre,seconds);};
    const auto decode=[](const juce::String& encoded)
    {
        const juce::String alphabet("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/");std::vector<int> result;
        if(encoded.length()%2)return result;
        for(int i=0;i<encoded.length();i+=2){const auto a=alphabet.indexOfChar(encoded[i]),b=alphabet.indexOfChar(encoded[i+1]);if(a<0||b<0)return std::vector<int>{};
            const auto value=a*64+b;result.push_back(value>=2048?value-4096:value);}return result;
    };
    const auto encode=[](const std::vector<int>& values)
    {
        constexpr char alphabet[]="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";juce::String result;
        for(auto value:values){if(value<0)value+=4096;result+=juce::String::charToString(alphabet[(value>>6)&63]);result+=juce::String::charToString(alphabet[value&63]);}return result;
    };
    const auto clip=[](const char* clipId,const char* noteId,const char* lyric,double start,float midi,float own)
    {
        ClipData c;c.id=clipId;c.startSeconds=start;c.durationSeconds=1;c.sourceDurationSeconds=1;
        NoteData n;n.id=noteId;n.label=lyric;n.durationSeconds=1;n.midiNote=midi;
        // Legacy PBW-style positive first/early final control: host owns a
        // nominal curve, not an implicit line through neighbouring controls.
        n.pitchControlPoints={{.1,own},{.8,own}};n.contour={{0,0,0,true},{1,0,0,true}};c.notes={n};return c;
    };
    ProjectData data;data.bpm=120;TrackData track;track.id="host";track.compose=true;track.pitchAlgorithm=PitchAlgorithm::utau;
    track.clips={clip("clip-a","A","a",0,60,60.25f),clip("clip-b","B","b",1,67,67.75f),clip("clip-c","C","c",3,72,72)};
    data.tracks={track};ProjectModel model;model.resetDocument(data);const auto fingerprint=model.contentFingerprint();const auto revision=model.revisionNumber();
    const auto legacy=AudioEngine::projectTrackNotePitch(track,{});const auto* a=find(legacy,"A");const auto* b=find(legacy,"B");
    check("public_bank_independent_projector_contains_all_note_identities",legacy.size()==3&&a&&b&&find(legacy,"C"));
    if(!a||!b)return false;
    // Update 221: authored endpoints (.8 and 1.1), not the old nominal
    // boundary +/-20ms, define the transition. Independent cubic oracle.
    bool leftWire=true,rightWire=true;
    for(const auto* p:{a,b})
    {
        const auto values=decode(wire(*p));bool accurate=true;
        for(size_t i=0;i<values.size();++i)
        {
            const auto t=p->absoluteStartSeconds+i*(60./(120*96))-.1;
            const auto u=std::clamp((t-.8)/.3,0.,1.);
            const auto desired=(60.25+7.5*u*u*(3-2*u)-p->midiNote)*100;
            accurate&=std::abs(values[i]-desired)<.65;
        }
        if(p==a)leftWire=accurate;else rightWire=accurate;
    }
    check("point_first_left_packet_uses_authored_transition",leftWire);
    check("point_first_right_packet_uses_same_transition",rightWire);
    check("authored_endpoints_are_on_the_prepared_curve",std::abs(sample(a->sampledLocalCents,.8)-25)<.001
        &&std::abs(sample(b->sampledLocalCents,.1)-75)<.001&&std::abs(sample(a->sampledLocalCents,.95)-400)<.1);
    check("point_first_support_covers_both_transition_endpoints",a->supportEndSeconds>=1.1&&b->supportStartSeconds<=.8);
    bool display=true;
    for(const auto& p:legacy)for(const auto t:{-.3,.0,.123456,.987654,1.23456})
        display&=p.continuousLocalCents&&std::abs(p.continuousLocalCents(t)-sample(p.sampledLocalCents,t))<=1.e-5f;
    check("display_and_continuous_reader_use_same_prepared_projection",display);
    const auto full=AudioEngine::utauRequestNotesForClip(data,"clip-a"),selected=AudioEngine::diagnosticUtauRequestNotes(data,"clip-a",{"A"});
    check("selection_happens_after_full_track_pitch_plan",full.size()==1&&selected.size()==1
        &&same(full[0].pitchCurve,a->sampledLocalCents)&&same(selected[0].pitchCurve,a->sampledLocalCents));
    if(full.size()==1){auto host=full[0];host.timelinePitchCents={};check("actual_request_prepared_packet_matches_public_UI_projection",
        backend::UtauRenderer::diagnosticPitchbend(host,120,.1,1.25)==wire(*a));}
    check("pure_projection_does_not_mutate_project_or_revision",model.contentFingerprint()==fingerprint&&model.revisionNumber()==revision&&!model.canUndo());
    auto bankless=track;bankless.utauMode=UtauMode::classic;bankless.voicebankDirectory=output.getChildFile("not-a-real-bank");
    bankless.clips[0].notes[0].diffSingerPitchOffset={{0,12},{1,12}};
    const auto bankIndependent=AudioEngine::projectTrackNotePitch(bankless,{});
    check("public_base_projection_is_independent_of_bank_and_DS_expression_field",find(bankIndependent,"A")&&wire(*find(bankIndependent,"A"))==wire(*a));
    auto ordinaryWithOffset=data;ordinaryWithOffset.tracks={bankless};
    const auto ordinaryRequest=AudioEngine::utauRequestNotesForClip(ordinaryWithOffset,"clip-a");
    check("DS_offset_adapter_does_not_apply_to_ordinary_UTAU",ordinaryRequest.size()==1&&same(ordinaryRequest[0].pitchCurve,a->sampledLocalCents));
    auto changedTrack=track;const auto immutable=AudioEngine::projectTrackNotePitch(changedTrack,{});
    const auto* captured=find(immutable,"A");
    constexpr std::array<double,5> captureTimes{-.1,.123456,.3,.987654,1.3};
    std::vector<float> capturedValues;
    if(captured&&captured->continuousLocalCents)for(const auto time:captureTimes)capturedValues.push_back(captured->continuousLocalCents(time));
    const auto capturedSamples=captured?captured->sampledLocalCents:std::vector<backend::UtauPitchPoint>{};
    const auto capturedPacket=captured?wire(*captured):juce::String{};
    changedTrack.clips[0].notes[0].pitchControlPoints[0].targetMidi=95;
    bool immutableValues=captured&&captured->continuousLocalCents&&capturedValues.size()==captureTimes.size()
        &&same(captured->sampledLocalCents,capturedSamples)&&wire(*captured)==capturedPacket;
    if(immutableValues)for(std::size_t i=0;i<captureTimes.size();++i)
        immutableValues&=captured->continuousLocalCents(captureTimes[i])==capturedValues[i];
    const auto replanned=AudioEngine::projectTrackNotePitch(changedTrack,{});const auto* changed=find(replanned,"A");
    const bool mutationEffective=changed&&changed->continuousLocalCents&&capturedValues.size()==captureTimes.size()
        &&changed->continuousLocalCents(.3)!=capturedValues[2];
    // Immutability compares the actual frozen float result, not an idealized
    // decimal25: legacy float Hermite may yield24.9996185303 at this time.
    check("projection_callbacks_capture_immutable_pitch_snapshot",immutableValues&&mutationEffective);
    if(capturedValues.size()==captureTimes.size()&&captured&&captured->continuousLocalCents)
        std::cout<<"immutable_snapshot_cents_at_point3_before="<<juce::String(capturedValues[2],12)
            <<"|after="<<juce::String(captured->continuousLocalCents(.3),12)<<"|fresh_mutation_effective="<<mutationEffective<<std::endl;

    auto noAuto=track;for(auto& c:noAuto.clips)for(auto& n:c.notes){n.utauAutoPitchTransition=false;n.pitchControlPoints.clear();}
    auto graphData=data;graphData.tracks={noAuto};model.resetDocument(graphData);const auto graphBaseline=model.snapshot();
    const auto base=AudioEngine::projectTrackNotePitch(noAuto,{});juce::String error;
    check("explicit_Force_graph_created_without_source_or_bank",model.setNotesPortamento({"A","B"},true,error,.1));
    const auto forced=model.snapshot();const auto curves=resolvedConnectionPitchCurves(forced.tracks[0],forced.nativeConnections);
    check("explicit_graph_resolves_for_OWN_base_projection",curves.size()==1&&curves[0].active);
    if(curves.size()==1&&curves[0].active)
    {
        const auto projected=AudioEngine::projectTrackNotePitch(forced.tracks[0],forced.nativeConnections);
        const auto* pa=find(projected,"A");const auto* pb=find(projected,"B");const auto* ba=find(base,"A");const auto* bb=find(base,"B");
        bool graphAudible=pa&&pb&&ba&&bb,roi=graphAudible;double maximumCentsError=0;
        if(graphAudible)for(const auto* p:{pa,pb})
        {
            const auto values=decode(wire(*p));const auto old=decode(wire(p->noteId=="A"?*ba:*bb));
            if(values.size()!=old.size()){graphAudible=false;roi=false;continue;}
            for(std::size_t i=0;i<values.size();++i)
            {
                const auto local=i*(60.0/(120*96))-.1,absolute=p->absoluteStartSeconds+local;
                if(absolute>=curves[0].fromSeconds&&absolute<=curves[0].toSeconds)
                {
                    const auto desired=(evaluatePitchCurve(curves[0].points,absolute)-p->midiNote)*100;
                    maximumCentsError=std::max(maximumCentsError,std::abs(values[i]-double(desired)));
                }
                else if(local>=0&&local<=p->durationSeconds)roi&=std::abs(values[i]-old[i])<=1;
            }
            graphAudible&=values!=old;
        }
        check("explicit_graph_is_sampled_final_in_both_host_packets",graphAudible&&maximumCentsError<=1.0);
        check("graph_ROI_does_not_change_original_OWN_support_outside_window",roi);
        check("graph_new_left_tail_support_holds_resolved_endpoint",pa&&pa->supportEndSeconds>=curves[0].toSeconds-1.e-9
            &&std::abs(pa->continuousLocalCents(1.2)-700)<1.e-3);
        check("nonincident_note_packet_is_byte_identical",find(projected,"C")&&find(base,"C")&&wire(*find(projected,"C"))==wire(*find(base,"C")));
        auto edited=curves[0].points;edited.insert(edited.begin()+1,{1,64.7f,PitchCurveShape::smooth});
        check("graph_interior_edit_accepted",model.setConnectionPitchCurve(curves[0].connectionId,edited,error));
        const auto changed=model.snapshot();const auto next=AudioEngine::projectTrackNotePitch(changed.tracks[0],changed.nativeConnections);
        check("graph_interior_edit_changes_packet_not_source_controls",pa&&find(next,"A")&&wire(*pa)!=wire(*find(next,"A"))
            &&changed.tracks[0].clips[0].notes[0].pitchControlPoints.empty()&&changed.tracks[0].clips[0].sourceFile==graphBaseline.tracks[0].clips[0].sourceFile);
        check("disconnect_restores_base_packet_exactly",model.setNotesPortamento({"A","B"},false,error)
            &&wire(*find(AudioEngine::projectTrackNotePitch(model.snapshot().tracks[0],model.snapshot().nativeConnections),"A"))==wire(*ba));
        // Vibrato is an independent expression, applied once after replacing the
        // base inside an explicit graph window, never summed twice.
        auto vib=forced;auto& n=vib.tracks[0].clips[0].notes[0];n.vibratoEnabled=true;n.vibratoDepthCents=35;n.vibratoLengthPercent=100;
        const auto vibCurves=resolvedConnectionPitchCurves(vib.tracks[0],vib.nativeConnections);
        const auto vibPlan=AudioEngine::projectTrackNotePitch(vib.tracks[0],vib.nativeConnections);const auto* vp=find(vibPlan,"A");
        bool once=vp&&vibCurves.size()==1&&vibCurves[0].active;
        if(once)for(int i=0;i<=40;++i)
        {
            const auto t=.95+i*.001;const auto expected=(evaluatePitchCurve(vibCurves[0].points,t)-n.midiNote)*100+static_cast<float>(vibratoCentsAt(n,t));
            once&=std::abs(vp->continuousLocalCents(t)-expected)<=.2f;
        }
        check("explicit_graph_expression_does_not_double_apply_vibrato",once);
    }
    // A global stroke is materialized into BOTH intersected note curves by
    // the existing exact composer. No implicit neighbour-reader is required.
    auto painted=data;painted.tracks[0].clips[0].notes[0].utauAutoPitchTransition=false;
    painted.tracks[0].clips[1].notes[0].utauAutoPitchTransition=false;
    const auto beforePaint=AudioEngine::projectTrackNotePitch(painted.tracks[0],{});
    std::vector<PitchCurveEditPoint> paintA,paintB;bool storeA=false,storeB=false;
    const bool composed=composeNativePitchStroke(painted.tracks[0].clips[0].notes[0],{{.8,60.25f,PitchCurveShape::linear},{1,64,PitchCurveShape::linear}},paintA,storeA,error,true,false)
        &&composeNativePitchStroke(painted.tracks[0].clips[1].notes[0],{{0,64,PitchCurveShape::linear},{.2,67.75f,PitchCurveShape::linear}},paintB,storeB,error,false,true);
    check("existing_global_composer_materializes_each_note_intersection",composed&&storeA&&storeB);
    if(composed)
    {
        painted.tracks[0].clips[0].notes[0].pitchControlPoints=paintA;painted.tracks[0].clips[1].notes[0].pitchControlPoints=paintB;
        const auto materialized=AudioEngine::projectTrackNotePitch(painted.tracks[0],{});
        const auto* ma=find(materialized,"A");const auto* mb=find(materialized,"B");bool accurate=ma&&mb,unchanged=accurate;
        if(accurate)for(const auto* p:{ma,mb})
        {
            const auto values=decode(wire(*p)),old=decode(wire(*find(beforePaint,p->noteId)));
            if(values.size()!=old.size()){accurate=false;unchanged=false;continue;}
            for(std::size_t i=0;i<values.size();++i)
            {
                const auto local=i*(60.0/(120*96))-.1,absolute=p->absoluteStartSeconds+local;
                if(local<0||local>1)continue; // declared legacy OWN body support
                if(absolute>=.802&&absolute<=1.198)
                {
                    const auto desiredMidi=60.25+18.75*(absolute-.8);
                    accurate&=std::abs(values[i]-(desiredMidi-p->midiNote)*100)<=1.0;
                }
                else if(absolute<.8||absolute>1.2)unchanged&=std::abs(values[i]-old[i])<=1;
            }
        }
        check("materialized_global_stroke_matches_both_note_body_packets",accurate);
        check("materialized_global_stroke_preserves_packets_outside_owned_paint_window",unchanged);
        check("materialized_global_seam_has_one_absolute_pitch",ma&&mb
            &&std::abs(ma->midiNote+ma->continuousLocalCents(1)/100.0-(mb->midiNote+mb->continuousLocalCents(0)/100.0))<1.e-5);
    }
    auto support=noAuto;support.clips.resize(1);support.clips[0].notes[0].pitchControlPoints={{-.2,59},{.3,61},{1.4,65}};
    const auto owned=AudioEngine::projectTrackNotePitch(support,{});const auto* op=find(owned,"A");
    check("legacy_negative_control_support_is_preserved",op&&near(op->supportStartSeconds,-.2));
    check("legacy_positive_tail_default_remains_nominal_not_implicit_shared",op&&near(op->supportEndSeconds,1)
        &&near(op->continuousLocalCents(1.3),op->continuousLocalCents(1)));
    auto tailGeometry=support;tailGeometry.clips[0].durationSeconds=1.4;
    auto& tailNote=tailGeometry.clips[0].notes[0];tailNote.durationSeconds=1.4;
    tailNote.pitchControlPoints={{-.2,59,PitchCurveShape::linear},{.3,61,PitchCurveShape::linear},
        {1.4,64,PitchCurveShape::linear},{1.401,66,PitchCurveShape::linear},{1.55,66,PitchCurveShape::linear}};
    const auto legacyTail=AudioEngine::projectTrackNotePitch(tailGeometry,{});const auto* legacyExtent=find(legacyTail,"A");
    const auto oldTailPacket=legacyExtent?wire(*legacyExtent,.1,1.9):juce::String{};
    auto authoredTail=tailGeometry;authoredTail.clips[0].notes[0].pitchSupportAuthored=true;
    const auto explicitTail=AudioEngine::projectTrackNotePitch(authoredTail,{});const auto* authoredExtent=find(explicitTail,"A");
    check("identical_legacy_tail_geometry_stays_nominal_without_author_intent",legacyExtent&&near(legacyExtent->supportEndSeconds,1.4)
        &&legacyExtent->continuousLocalCents(1.401)==legacyExtent->continuousLocalCents(1.4)
        &&wire(*find(AudioEngine::projectTrackNotePitch(tailGeometry,{}),"A"),.1,1.9)==oldTailPacket);
    check("explicit_author_intent_extends_same_curve_through_1point55",authoredExtent&&near(authoredExtent->supportEndSeconds,1.55)
        &&near(authoredExtent->midiNote+authoredExtent->continuousLocalCents(1.401)/100,66)
        &&near(authoredExtent->midiNote+authoredExtent->continuousLocalCents(1.55)/100,66));
    bool tailWire=legacyExtent&&authoredExtent,tailBody=tailWire;int changedTailSamples=0;
    if(tailWire)
    {
        const auto before=decode(oldTailPacket),after=decode(wire(*authoredExtent,.1,1.9));tailWire=before.size()==after.size();
        if(tailWire)for(std::size_t i=0;i<after.size();++i)
        {
            const auto local=i*(60.0/(120*96))-.1;
            if(local<=1.4)tailBody&=std::abs(before[i]-after[i])<=1;
            else if(local<=1.55)
            {
                const auto desired=(evaluatePitchCurve(tailNote.pitchControlPoints,local)-tailNote.midiNote)*100;
                tailWire&=std::abs(after[i]-double(desired))<=1;
                changedTailSamples+=before[i]!=after[i];
            }
        }
    }
    check("authored_tail_is_encoded_not_only_displayed",tailWire&&changedTailSamples>0);
    check("author_intent_preserves_nominal_body_packet",tailBody);
    auto legacyTailProject=data;legacyTailProject.tracks={tailGeometry};auto authoredTailProject=data;authoredTailProject.tracks={authoredTail};
    const auto legacyTailKey=AudioEngine::diagnosticUtauRenderKey(legacyTailProject,"clip-a");
    const auto authoredTailKey=AudioEngine::diagnosticUtauRenderKey(authoredTailProject,"clip-a");
    authoredTailProject.tracks[0].clips[0].notes[0].pitchSupportAuthored=false;
    check("authored_support_cache_tag_is_true_only_and_false_key_restores",legacyTailKey!=authoredTailKey
        &&legacyTailKey==AudioEngine::diagnosticUtauRenderKey(authoredTailProject,"clip-a"));
    auto denseTail=authoredTail;auto& dense=denseTail.clips[0].notes[0];dense.pitchControlPoints.clear();dense.vibratoEnabled=true;
    dense.contour={{0,0,0,true},{1.55,400,400,true}};
    const auto denseProjection=AudioEngine::projectTrackNotePitch(denseTail,{});const auto* denseExtent=find(denseProjection,"A");
    check("authored_derived_vibrato_support_also_extends_stored_extent",denseExtent&&near(denseExtent->supportEndSeconds,1.55));
    auto rest=noAuto;rest.clips[0].notes[0].label="RR";rest.clips[1].notes[0].nativeUnpitched=true;
    const auto silent=AudioEngine::projectTrackNotePitch(rest,{});
    check("reserved_rest_and_explicit_unpitched_never_get_voiced_projection",find(silent,"A")&&find(silent,"B")
        &&find(silent,"A")->sampledLocalCents.empty()&&!find(silent,"A")->continuousLocalCents
        &&find(silent,"B")->sampledLocalCents.empty()&&!find(silent,"B")->continuousLocalCents);
    auto restProject=data;restProject.tracks={rest};const auto uv=AudioEngine::utauRequestNotesForClip(restProject,"clip-b");
    check("explicit_unpitched_transport_uses_reserved_RR_not_normal_R_lyric",uv.size()==1&&uv[0].alias=="RR");
    rest.clips[0].notes[0].label="R";const auto normalR=AudioEngine::projectTrackNotePitch(rest,{});
    check("normal_single_R_lyric_remains_pitchable",find(normalR,"A")&&bool(find(normalR,"A")->continuousLocalCents));
    auto physical=noAuto;physical.pitchAlgorithm=PitchAlgorithm::nsfHifigan;physical.nativeNsfAudio=true;
    physical.clips[0].notes[0].nativeUnpitched=true;
    const auto native=AudioEngine::diagnosticNativeRequest(physical.clips[0],physical);
    check("native_FILE_UV_mask_policy_is_not_changed_by_host_projector",std::all_of(native.targetMidi.begin(),native.targetMidi.end(),[](float value){return value==0;}));
    auto report=new juce::DynamicObject();report->setProperty("ok",ok);report->setProperty("checks",checks);report->setProperty("results",juce::var(results));
    report->setProperty("wireQuantizationToleranceCents",1);report->setProperty("authoredTailIntentStatus","Explicit pitchSupportAuthored extends stored control/derived-vibrato support; false retains legacy nominal clamp. Setter/serde authoring-intent tests are model-owned.");
    const bool saved=output.getChildFile("verification.json").replaceWithText(juce::JSON::toString(juce::var(report)));
    std::cout<<"host_pitch_projection_ok="<<(ok&&saved)<<"; checks="<<checks<<std::endl;return ok&&saved;
}
}
