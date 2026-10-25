#pragma once
#include "../AssetManagerComponent.h"
#include "../OttoMaterialIO.h"
#include "../Theme.h"
#include <juce_audio_formats/juce_audio_formats.h>
#include <iostream>
namespace hachi
{
inline bool runMaterialAssetTreeSmoke(const juce::File& output)
{
    const auto root=output.getChildFile("tree-"+juce::Uuid().toString()); root.createDirectory();
    const auto bank=root.getChildFile("bank");bank.createDirectory();
    const auto a=bank.getChildFile("recording.wav"),b=bank.getChildFile("other.wav");
    const auto makeAudio=[](const juce::File& f)
    {
        juce::AudioBuffer<float> pcm(1,48000);pcm.clear();juce::WavAudioFormat w;
        auto s=f.createOutputStream();if(!s)return false;
        std::unique_ptr<juce::AudioFormatWriter> out(w.createWriterFor(s.release(),48000,1,16,{},0));
        return out&&out->writeFromAudioSampleBuffer(pcm,0,pcm.getNumSamples());
    };
    if(!makeAudio(a)||!makeAudio(b))return false;
    std::vector<SampleRegionSetting> rows;
    for(int i=0;i<3;++i)
    {
        SampleRegionSetting row;row.materialAudioId="a1";row.materialRegionId="p"+juce::String(i+1);
        row.name=i==0?"ka":"a";row.regionStartSeconds=i/3.0;row.regionEndSeconds=(i+1)/3.0;
        row.alignmentSeconds=row.regionStartSeconds;
        if(i<2)row.connectedNextRegionId="p"+juce::String(i+2);
        row.pronunciationKind=i==2?PronunciationKind::transition:PronunciationKind::normal;rows.push_back(row);
    }
    juce::String error;
    if(!SampleSettings::save(a,rows,error)){std::cerr<<error<<std::endl;return false;}
    SampleRegionSetting other;other.name="a";other.regionEndSeconds=1;
    if(!SampleSettings::save(b,{other},error))return false;
    const auto pcmStamp=a.getLastModificationTime();
    juce::PropertiesFile::Options options;options.applicationName="otto-tree-smoke";
    options.storageFormat=juce::PropertiesFile::storeAsXML;
    juce::PropertiesFile properties(root.getChildFile("settings.xml"),options);
    I18n strings;HachiLookAndFeel look;AssetManagerComponent manager(strings,properties);
    manager.setLookAndFeel(&look);manager.setSize(400,650);
    int total=0,failures=0;juce::String report;
    const auto check=[&](const char* name,bool pass)
    {++total;if(!pass)++failures;report+=juce::String(name)+"="+(pass?"PASS":"FAIL")+"\n";};
    manager.diagnosticRegisterFolder(bank);
    manager.diagnosticToggleAudio(a);
    const auto labels=manager.diagnosticPronunciationLabels(a);
    check("three_pronunciations_under_one_audio",labels.size()==3&&labels[0].contains("ka")&&labels[2].contains("[T]"));
    const auto loads=manager.diagnosticManifestLoadCount();
    check("initial_manifest_read_once",loads==1);
    check("primary_selection",manager.diagnosticSelectPronunciation(a,"p1")&&manager.diagnosticSelectedRegionIds()==juce::StringArray{"p1"});
    check("ctrl_multi_select",manager.diagnosticSelectPronunciation(a,"p2",true)
        &&manager.diagnosticSelectedRegionIds()==juce::StringArray{"p1","p2"});
    check("ctrl_toggle",manager.diagnosticSelectPronunciation(a,"p1",true)
        &&manager.diagnosticSelectedRegionIds()==juce::StringArray{"p2"});
    check("shift_contiguous_range",manager.diagnosticSelectPronunciation(a,"p1")
        &&manager.diagnosticSelectPronunciation(a,"p3",false,true)
        &&manager.diagnosticSelectedRegionIds()==juce::StringArray{"p1","p2","p3"});
    for (const auto& theme : {juce::String("light"),juce::String("dark")})
    {
        Palette::applyTheme(theme,juce::Colour(0xff7f69ca),juce::Colour(0xffcbcbfa),juce::Colour(0xfff4c000));
        look.refreshColours();manager.sendLookAndFeelChange();
        for (const auto width : {220,320,400})
        {
            manager.setSize(width,650);
            auto png=root.getChildFile("material-tree-"+theme+"-"+juce::String(width)+".png").createOutputStream();
            check("actual_tree_snapshot",png&&juce::PNGImageFormat().writeImageToStream(manager.createComponentSnapshot(manager.getLocalBounds()),*png));
        }
    }
    const auto drag=manager.pronunciationDragDescription(a,{"p1","p2"});
    const auto dragIds=drag["regionIds"];
    check("drag_payload_stable_ids",static_cast<bool>(drag["nativeMaterial"])
        &&dragIds.isArray()&&dragIds.size()==2&&dragIds[0].toString()=="p1"&&dragIds[1].toString()=="p2"
        &&drag["manifest"].toString()==bank.getChildFile("otto.ini").getFullPathName());
    const auto whole=manager.audioDragDescription(a);const auto allIds=whole["regionIds"];
    check("whole_audio_drag_does_not_leak_selected_subset",static_cast<bool>(whole["nativeMaterial"])
        &&allIds.isArray()&&allIds.size()==0);
    juce::File requested;juce::StringArray requestedIds;
    manager.onLoadPronunciations=[&](const auto& file,const auto& ids){requested=file;requestedIds=ids;};
    check("enter_loads_selected_pronunciations",manager.keyPressed(juce::KeyPress(juce::KeyPress::returnKey))
        &&requested==a&&requestedIds==juce::StringArray{"p1","p2","p3"});
    auto bRows=SampleSettings::loadOrDerive(b,ProjectData{});
    check("selection_is_scoped_to_audio",!bRows.empty()&&manager.diagnosticSelectPronunciation(b,bRows[0].materialRegionId,true)
        &&manager.diagnosticSelectedRegionIds().size()==1);
    manager.diagnosticPronunciationLabels(a);manager.diagnosticPronunciationLabels(b);
    check("selection_and_paint_queries_reuse_manifest",manager.diagnosticManifestLoadCount()==loads);
    check("source_audio_not_touched",a.getLastModificationTime()==pcmStamp);
    root.getChildFile("verification.txt").replaceWithText(report);
    std::cout<<"tree_passed="<<total-failures<<"|total="<<total<<"|fixture="<<root.getFullPathName()<<std::endl;
    if(failures)std::cerr<<report<<std::endl;
    return failures==0;
}
}
