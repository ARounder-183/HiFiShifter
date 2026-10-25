#pragma once
#include "../AudioEngine.h"
#include "../SampleSettings.h"
#include "../OttoMaterialMigration.h"
#include "../OttoMaterialIO.h"
#include "../backend/UtauRenderer.h"
#include <cmath>
#include <iostream>
#include <type_traits>

namespace hachi
{
inline bool otoProvenanceSmoke(const juce::File& output)
{
    using backend::UtauRenderer;
    static_assert(std::is_same_v<decltype(*UtauRenderer::ResolvedSample{}.sourceOto), const VoicebankOtoEntry&>);
    bool ok=true; int checks=0; juce::Array<juce::var> failures;
    const auto check=[&](const juce::String& name,bool pass)
    {++checks;ok&=pass;if(!pass)failures.add(name);std::cout<<"oto_provenance_"<<name<<"="<<pass<<std::endl;};
    check("output_created",output.createDirectory().wasOk());
    const auto root=output.getChildFile("fixtures-"+juce::Uuid().toString());
    const auto bank=root.getChildFile("classic"),single=root.getChildFile("single"),bare=root.getChildFile("bare"),native=root.getChildFile("native");
    check("directories_created",bank.createDirectory().wasOk()&&single.createDirectory().wasOk()&&bare.createDirectory().wasOk()&&native.createDirectory().wasOk());
    const auto writeAudio=[](const juce::File& file)
    {
        juce::AudioBuffer<float> pcm(1,8820);
        for(int i=0;i<pcm.getNumSamples();++i)pcm.setSample(0,i,static_cast<float>(.1*std::sin(juce::MathConstants<double>::twoPi*440*i/44100.0)));
        juce::WavAudioFormat wav;auto stream=file.createOutputStream();
        std::unique_ptr<juce::AudioFormatWriter> writer(stream?wav.createWriterFor(stream.release(),44100,1,16,{},0):nullptr);
        return writer&&writer->writeFromAudioSampleBuffer(pcm,0,pcm.getNumSamples());
    };
    check("physical_audio_written",writeAudio(bank.getChildFile("a.wav"))&&writeAudio(bank.getChildFile("b.wav"))
        &&writeAudio(bank.getChildFile("free.wav"))&&writeAudio(single.getChildFile("a.wav"))
        &&writeAudio(bare.getChildFile("bare.wav"))&&writeAudio(native.getChildFile("a.wav")));
    const auto oto=bank.getChildFile("oto.ini");
    const juce::String original="; physical line zero\na.wav=dup,10,30,-150,20,5\na.wav=dup,20,40,-140,25,6\na.wav=a,0,30,-180,20,5\na.wav=PaS,5,30,-170,20,5\nb.wav=other,0,30,-180,20,5\n";
    check("oto_and_prefix_written",oto.replaceWithText(original)&&bank.getChildFile("prefix.map").replaceWithText("C4\tP\tS\n")
        &&single.getChildFile("oto.ini").replaceWithText("a.wav=sole,0,30,-180,20,5\n"));
    juce::String error;
    const auto duplicate=UtauRenderer::resolveVoiceSample(bank,"dup",60);
    check("duplicate_alias_exact_first_physical_line",duplicate.found&&duplicate.sourceOto&&duplicate.sourceOto->lineIndex==1
        &&duplicate.sourceOto->otoFile==oto&&duplicate.sourceOto->audioFile==bank.getChildFile("a.wav")
        &&duplicate.sourceOto->sourceName=="a.wav"&&duplicate.sourceOto->offsetMs==10&&duplicate.resolvedAlias=="dup");
    const auto repeated=UtauRenderer::resolveVoiceSample(bank,"dup",60);
    check("repeated_lookup_shares_immutable_original",duplicate.sourceOto&&duplicate.sourceOto==repeated.sourceOto);
    OttoMaterialMigration::OtoSource source;
    check("ticket_fingerprint_matches_physical_document",duplicate.sourceOto
        &&OttoMaterialMigration::readOtoSource(*duplicate.sourceOto,UtauMode::classic,source,error)
        &&source.fingerprint==duplicate.sourceOto->sourceFingerprint&&!source.fingerprint.isEmpty());
    SampleRegionSetting converted,convertedAgain;
    const bool convertedOk=duplicate.sourceOto&&OttoMaterialMigration::convertOtoEntry(*duplicate.sourceOto,UtauMode::classic,duplicate.fileSeconds,source,converted,error);
    check("existing_strict_converter_consumes_exact_ticket",convertedOk&&converted.name=="dup"&&std::abs(converted.regionStartSeconds-.01)<1.e-9
        &&OttoMaterialMigration::convertOtoEntry(*duplicate.sourceOto,UtauMode::classic,duplicate.fileSeconds,source,convertedAgain,error)
        &&converted.materialRegionId==convertedAgain.materialRegionId&&converted.materialRegionId.isNotEmpty());
    const auto prefixed=UtauRenderer::resolveVoiceSample(bank,"a",60);
    check("prefix_map_preserves_actual_selected_row",prefixed.found&&prefixed.sourceOto&&prefixed.sourceOto->lineIndex==4
        &&prefixed.sourceOto->alias=="PaS"&&prefixed.resolvedAlias=="PaS");
    const auto filename=UtauRenderer::resolveVoiceSample(bank,"b",61);
    check("filename_fallback_preserves_actual_selected_row",filename.found&&filename.sourceOto
        &&filename.sourceOto->lineIndex==5&&filename.resolvedAlias=="other");
    const auto sole=UtauRenderer::resolveVoiceSample(single,"arbitrary lyric",60);
    check("single_sample_remap_preserves_exact_origin",sole.found&&sole.sourceOto&&sole.sourceOto->lineIndex==0
        &&sole.sourceOto->alias=="sole"&&sole.resolvedAlias=="sole");
    backend::UtauOtoOverride own;own.enabled=true;own.offsetMs=30;own.consonantMs=40;own.cutoffMs=-100;own.preutteranceMs=15;own.overlapMs=2;
    const auto overridden=UtauRenderer::resolveVoiceSample(bank,"dup",60,150,false,false,.005,true,.011,true,.003,&own);
    check("note_override_STP_velocity_do_not_mutate_origin",overridden.found&&overridden.sourceOto==duplicate.sourceOto
        &&overridden.sourceOto&&overridden.sourceOto->offsetMs==10&&overridden.sourceOto->consonantMs==30
        &&overridden.sourceOto->lineIndex==1&&std::abs(overridden.offsetSeconds-.035)<1.e-9
        &&std::abs(overridden.preutteranceSeconds-.011)<1.e-9&&std::abs(overridden.overlapSeconds-.003)<1.e-9);
    const auto exact=UtauRenderer::resolveVoiceSample(bank,"dup",60,100,false,false,0,false,0,false,0,nullptr,true);
    check("already_bound_alias_uses_same_origin",exact.found&&exact.sourceOto==duplicate.sourceOto&&exact.resolvedAlias=="dup");
    const auto missing=UtauRenderer::resolveVoiceSample(bank,"not-present",60);
    check("unresolved_has_no_fake_ticket",!missing.found&&!missing.sourceOto&&missing.resolvedAlias.isEmpty());
    const auto unlisted=UtauRenderer::resolveVoiceSample(bank,"free",60);
    const auto derived=UtauRenderer::resolveVoiceSample(bare,"bare",60);
    check("unlisted_and_derived_have_no_physical_OTO_ticket",unlisted.found&&!unlisted.sourceOto&&derived.found&&!derived.sourceOto);
    const auto jie=UtauRenderer::resolveVoiceSample(bank,"dup",60,100,true,false);
    const auto mou=UtauRenderer::resolveVoiceSample(bank,"dup",60,100,true,true);
    check("Jie_Mou_unpinned_aux_sources_return_null",jie.found&&!jie.sourceOto&&mou.found&&!mou.sourceOto);
    // Native manifest authority is never bypassed to manufacture a legacy row.
    check("old_OTO_beside_native_written",native.getChildFile("oto.ini").replaceWithText("a.wav=old,0,30,-180,20,5\n"));
    SampleRegionSetting nativeRow;nativeRow.name="native";nativeRow.materialAudioId="native-audio";nativeRow.materialRegionId="native-row";
    nativeRow.regionEndSeconds=.2;nativeRow.role=NativeSegmentRole::vowel;
    OttoMaterialIO::Document document;document.audio={{nativeRow.materialAudioId,"a.wav",{nativeRow}}};
    check("native_manifest_written",OttoMaterialIO::saveDocument(native.getChildFile("otto.ini"),document,error));
    const auto nativeSample=UtauRenderer::resolveVoiceSample(native,"native",60);
    const auto oldHidden=UtauRenderer::resolveVoiceSample(native,"old",60);
    check("native_manifest_ticket_null_and_old_OTO_not_revived",nativeSample.found&&!nativeSample.sourceOto&&!oldHidden.found&&!oldHidden.sourceOto);

    backend::UtauRenderRequest request;request.voicebankDirectory=single;request.targetDurationSeconds=.2;
    backend::UtauNoteRenderSpec spec;spec.alias="sole";spec.durationSeconds=.15;spec.midiNote=69;request.notes={spec};
    const auto before=UtauRenderer::render(request);const auto metadata=UtauRenderer::resolveVoiceSample(single,"sole",69);
    const auto after=UtauRenderer::render(request);
    bool same=before.sampleRate==after.sampleRate&&before.buffer.getNumChannels()==after.buffer.getNumChannels()&&before.buffer.getNumSamples()==after.buffer.getNumSamples();
    if(same)for(int c=0;c<before.buffer.getNumChannels();++c)for(int s=0;s<before.buffer.getNumSamples();++s)
        same&=before.buffer.getSample(c,s)==after.buffer.getSample(c,s);
    check("metadata_probe_keeps_actual_PCM_and_backend_identical",metadata.sourceOto&&before.buffer.getNumSamples()>0&&same&&before.backend==after.backend&&before.warning==after.warning);
    TrackData track;track.id="ticket-track";track.trackType=TrackType::utau;track.pitchAlgorithm=PitchAlgorithm::utau;track.voicebankDirectory=single;
    ClipData clip;clip.id="ticket-clip";clip.durationSeconds=.2;NoteData note;note.id="ticket-note";note.label="sole";note.durationSeconds=.15;clip.notes={note};track.clips={clip};
    ProjectData project;project.tracks={track};const auto key=AudioEngine::diagnosticUtauRenderKey(project,clip.id);
    const auto ignored=UtauRenderer::resolveVoiceSample(single,"arbitrary lyric",69);juce::ignoreUnused(ignored);
    check("metadata_not_serialized_in_render_cache_identity",key==AudioEngine::diagnosticUtauRenderKey(project,clip.id));
    // Retain the chosen immutable ticket across an external OTO edit. Re-read
    // that exact source, not an alias search; fingerprint rejects the stale row.
    check("external_source_change_fixture",oto.appendText("; changed after selection\n"));
    OttoMaterialMigration::OtoSource staleSource;error.clear();
    check("stale_ticket_rejected_by_existing_source_fingerprint",duplicate.sourceOto
        &&!OttoMaterialMigration::readOtoSource(*duplicate.sourceOto,UtauMode::classic,staleSource,error)&&error.isNotEmpty());
    check("ticket_remains_unmodified_after_source_change",duplicate.sourceOto&&duplicate.sourceOto->lineIndex==1&&duplicate.sourceOto->offsetMs==10
        &&duplicate.sourceOto->sourceFingerprint==source.fingerprint);
    auto* report=new juce::DynamicObject;report->setProperty("ok",ok);report->setProperty("checks",checks);report->setProperty("failures",failures);
    report->setProperty("scope","exact classic OTO provenance only; metadata must not change DSP/cache; parent real-project regressions separate");
    const auto written=output.getChildFile("verification.json").replaceWithText(juce::JSON::toString(juce::var(report),true));
    return ok&&written;
}
}
