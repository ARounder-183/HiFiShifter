#pragma once
#include "../OtoWaveformEditorComponent.h"

namespace hachi
{
template<class Check>
void verifyUtauSourceEditors(const juce::File& output, Check&& check)
{
    const auto root=output.getChildFile("oto-routing-"+juce::Uuid().toString());root.createDirectory();
    const auto wav=root.getChildFile("tone.wav"),oto=root.getChildFile("oto.ini");
    {
        juce::AudioBuffer<float> pcm(1,26460);pcm.clear();
        auto stream=wav.createOutputStream();juce::WavAudioFormat format;
        std::unique_ptr<juce::AudioFormatWriter> writer(stream?format.createWriterFor(stream.release(),44100,1,16,{},0):nullptr);
        check("routing_audio_fixture",writer&&writer->writeFromAudioSampleBuffer(pcm,0,pcm.getNumSamples()));
    }
    check("routing_alias_fixture",oto.replaceWithText("tone.wav=PaS,10,40,-300,30,5\ntone.wav=other,0,20,-200,10,0\n")
        &&root.getChildFile("prefix.map").replaceWithText("C4\tP\tS\n"));
    juce::StringArray warnings;juce::String error;
    const auto entries=SampleSettings::loadAudioOto(wav,root,UtauMode::classic,warnings);
    if(entries.empty()){check("routing_entry_exists",false);return;}
    const auto match=std::find_if(entries.begin(),entries.end(),[](const auto& row){return row.alias=="PaS";});
    if(match==entries.end()){check("routing_target_alias_exists",false);return;}
    const auto original=*match;auto jie=original;
    jie.offsetMs=20;jie.consonantMs=55;jie.preutteranceMs=35;jie.hasJieOto=true;
    jie.jieOnsetMs=55;jie.jieGlideMs=90;jie.jieNucleusMs=220;
    check("routing_jie_fixture",SampleSettings::updateJieVoicebankOtoEntry(original,jie,error)
        &&SampleSettings::updateJieOtoEntry(jie,jie,error));
    auto mou=jie;mou.hasJieOto=true;mou.mouClasses="CVV";
    mou.jieOnsetMs=45;mou.jieGlideMs=210;mou.jieNucleusMs=300;
    check("routing_mou_fixture",SampleSettings::updateMouOtoEntry(jie,mou,error));
    const auto classicBytes=oto.loadFileAsString();
    for(const auto mode:{UtauMode::classic,UtauMode::jie,UtauMode::mou})
    {
        backend::UtauRenderer::invalidateVoicebankCache();
        const auto sample=backend::UtauRenderer::resolveVoiceSample(root,"a",60,100,utauModeUsesRegions(mode),mode==UtauMode::mou);
        check("routing_resolved_prefix_alias",sample.found&&sample.resolvedAlias=="PaS");
        int saves=0;
        auto editor=OtoWaveformEditorComponent::forVoicebankSample(root,mode,sample,[&]{++saves;},error);
        check("routing_opens_direct_phoneme_detail",editor&&!editor->diagnosticMaterialOverview()
            &&editor->originalEntry().alias=="PaS"&&editor->originalEntry().audioFile==wav);
        if(!editor)continue;
        check("routing_uses_track_mode_partition",mode==UtauMode::classic
            ? editor->diagnosticParameterShown(1)&&!editor->diagnosticParameterShown(5)
            : editor->diagnosticRegionCount()==(mode==UtauMode::jie?4:3));
        check("routing_uses_track_mode_timing",std::abs(editor->originalEntry().offsetMs-(mode==UtauMode::classic?10:20))<.001);
        const auto beforeClassic=oto.loadFileAsString();
        const auto beforeFour=root.getChildFile("oto4.ini").loadFileAsString();
        const auto text=editor->diagnosticParameterText(3).getDoubleValue();
        editor->diagnosticTypeParameter(3,juce::String(text+1));
        editor->diagnosticSave();
        check("routing_save_does_not_create_native_otto",saves==1&&!root.getChildFile("otto.ini").exists());
        if(mode==UtauMode::classic)
            check("routing_classic_writes_classic",oto.loadFileAsString()!=classicBytes);
        else
            check("routing_regional_save_preserves_classic",oto.loadFileAsString()==beforeClassic);
        if(mode==UtauMode::mou)
            check("routing_mou_save_preserves_four_regions",root.getChildFile("oto4.ini").loadFileAsString()==beforeFour);
        const auto name=mode==UtauMode::classic?"classic":mode==UtauMode::jie?"jie":"mou";
        auto image=output.getChildFile(juce::String("oto-detail-")+name+".png").createOutputStream();
        if(image)juce::PNGImageFormat().writeImageToStream(editor->createComponentSnapshot(editor->getLocalBounds()),*image);
    }
    check("routing_all_original_aliases_retained",SampleSettings::loadAudioOto(wav,root,UtauMode::classic,warnings).size()==2);
}
}
