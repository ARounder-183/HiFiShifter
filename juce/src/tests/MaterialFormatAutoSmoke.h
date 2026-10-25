#pragma once
#include "../AssetManagerComponent.h"
#include "../MaterialAnnotationSession.h"
#include "../OttoMaterialMigration.h"
#include "../OttoMaterialIO.h"
#include "../backend/UtauRenderer.h"
#include <juce_audio_formats/juce_audio_formats.h>
#include <cmath>
#include <functional>
#include <iostream>

namespace hachi
{
// Isolated format-authority regression. No app window, popup, or audio device.
inline bool runMaterialFormatAutoSmoke(const juce::File& output)
{
    const auto root=output.getChildFile("format-auto-"+juce::Uuid().toString());
    if(!root.createDirectory().wasOk())return false;
    int total=0,failed=0;juce::String log,error;
    const auto check=[&](const juce::String& name,bool success)
    {
        ++total;if(!success)++failed;
        log+=name+"="+(success?"PASS":"FAIL")+(success||error.isEmpty()?juce::String{}:" | "+error)+"\n";
    };
    const auto finish=[&]
    {
        const bool written=root.getChildFile("verification.txt").replaceWithText(log);
        std::cout<<"material_format_auto_passed="<<total-failed<<"|total="<<total
                 <<"|fixture="<<root.getFullPathName()<<std::endl;
        if(failed)std::cerr<<log<<std::endl;
        return failed==0&&written;
    };
    // Preserve the exact process-wide lookup roots, not just this test's empty
    // PropertiesFile. Managers publish their root set while rebuilding members.
    struct RestoreRoots
    {
        juce::StringArray before=SampleSettings::materialRoots();
        ~RestoreRoots(){SampleSettings::setMaterialRoots(before);}
    } restore;
    const auto makeAudio=[](const juce::File& audio)
    {
        if(!audio.getParentDirectory().createDirectory().wasOk())return false;
        juce::AudioBuffer<float> pcm(1,48000);pcm.clear();
        auto stream=audio.createOutputStream();if(!stream)return false;
        juce::WavAudioFormat format;
        std::unique_ptr<juce::AudioFormatWriter> writer(format.createWriterFor(stream.release(),48000,1,16,{},0));
        return writer&&writer->writeFromAudioSampleBuffer(pcm,0,pcm.getNumSamples());
    };
    struct ObservedFile{juce::File file;juce::MemoryBlock bytes;};
    std::vector<ObservedFile> originals;
    const auto observe=[&](const juce::File& file)
    {ObservedFile value{file,{}};const bool ok=file.loadFileAsData(value.bytes);if(ok)originals.push_back(std::move(value));return ok;};
    const auto identities=[](const std::vector<SampleRegionSetting>& rows)
    {
        juce::StringArray ids;
        for(const auto& row:rows)ids.add(row.materialAudioId+"/"+row.materialRegionId+"/"+row.name);
        return ids;
    };
    const auto otoOnly=root.getChildFile("oto-only"),bare=root.getChildFile("bare");
    const auto native=root.getChildFile("native-authority"),corrupt=root.getChildFile("corrupt");
    const auto scoped=root.getChildFile("scoped-oto");
    for(const auto& folder:{otoOnly,bare,native,corrupt,scoped})
        check("fixture_directory_"+folder.getFileName(),folder.createDirectory().wasOk());
    const auto otoAudio=otoOnly.getChildFile("source.wav"),bareAudio=bare.getChildFile("source.wav");
    const auto nativeAudio=native.getChildFile("source.wav"),missingAudio=native.getChildFile("missing.wav");
    const auto corruptAudio=corrupt.getChildFile("source.wav"),scopedAudio=scoped.getChildFile("child/source.wav");
    for(const auto& audio:{otoAudio,bareAudio,nativeAudio,missingAudio,corruptAudio,scopedAudio})
        check("fixture_wav_"+audio.getParentDirectory().getFileName()+"_"+audio.getFileName(),makeAudio(audio)&&observe(audio));
    const auto oto=otoOnly.getChildFile("oto.ini"),nativeOto=native.getChildFile("oto.ini");
    const auto corruptOto=corrupt.getChildFile("oto.ini"),parentOto=scoped.getChildFile("oto.ini");
    const auto childOto=scoped.getChildFile("child/oto.ini");
    check("fixture_two_oto_aliases",oto.replaceWithText("source.wav=ka,100,80,-400,50,20\nsource.wav=ki,600,60,-300,30,10\n")&&observe(oto));
    check("fixture_conflicting_oto",nativeOto.replaceWithText("source.wav=obsolete,100,80,-400,50,20\nmissing.wav=must-not-fallback,0,80,-900,50,20\n")&&observe(nativeOto));
    check("fixture_corrupt_bank_oto",corruptOto.replaceWithText("source.wav=fallback-forbidden,100,80,-400,50,20\n")&&observe(corruptOto));
    check("fixture_scoped_parent_oto",parentOto.replaceWithText("child/source.wav=parent-wrong,0,80,-900,50,20\n")&&observe(parentOto));
    check("fixture_scoped_child_oto",childOto.replaceWithText("source.wav=nearest-child,200,80,-500,50,20\n")&&observe(childOto));
    std::vector<SampleRegionSetting> rows;
    check("bare_pure_read_empty_without_publication",OttoMaterialMigration::readAudioAnnotations(bareAudio,bare,UtauMode::classic,rows,error)
        &&rows.empty()&&!bare.getChildFile("otto.ini").exists());
    auto bareSession=MaterialAnnotationSession::open(bareAudio,bare.getChildFile("otto.ini"),error);
    check("bare_session_empty_without_publication",bareSession&&bareSession->rows().empty()&&!bare.getChildFile("otto.ini").exists());
    check("bare_explicit_registration_ensure",OttoMaterialMigration::ensureNativeFolder(bare,error));
    OttoMaterialIO::Document empty;
    check("bare_ensure_one_valid_empty_otto",OttoMaterialIO::loadDocument(bare.getChildFile("otto.ini"),empty,error)&&empty.audio.empty()
        &&bare.findChildFiles(juce::File::findFiles,false,"otto.ini").size()==1&&!SampleSettings::sidecarFor(bareAudio).exists());
    const auto emptyBytes=bare.getChildFile("otto.ini").loadFileAsString();
    check("bare_ensure_idempotent",OttoMaterialMigration::ensureNativeFolder(bare,error)
        &&bare.getChildFile("otto.ini").loadFileAsString()==emptyBytes);
    check("oto_only_ensure_is_read_only",OttoMaterialMigration::ensureNativeFolder(otoOnly,error)&&!otoOnly.getChildFile("otto.ini").exists());
    rows.clear();
    check("oto_only_pure_read_all_aliases",OttoMaterialMigration::readAudioAnnotations(otoAudio,otoOnly,UtauMode::classic,rows,error)
        &&rows.size()==2&&rows[0].name=="ka"&&rows[1].name=="ki"&&!otoOnly.getChildFile("otto.ini").exists());
    const auto importedIds=identities(rows);
    std::vector<SampleRegionSetting> repeat;
    check("pure_oto_import_identity_is_deterministic",OttoMaterialMigration::readAudioAnnotations(otoAudio,otoOnly,UtauMode::classic,repeat,error)
        &&identities(repeat)==importedIds&&repeat.size()==2&&!repeat[0].materialRegionId.isEmpty());
    auto imported=MaterialAnnotationSession::open(otoAudio,otoOnly.getChildFile("otto.ini"),error);
    check("session_oto_import_agrees_with_pure_read_ids",imported&&identities(imported->rows())==importedIds
        &&!otoOnly.getChildFile("otto.ini").exists());
    auto cancelled=MaterialAnnotationSession::open(otoAudio,otoOnly.getChildFile("otto.ini"),error);
    check("oto_session_draft_edit",cancelled&&cancelled->createPronunciation(.51,.59,"cancelled",error));
    cancelled.reset();
    check("oto_session_cancel_never_converts",!otoOnly.getChildFile("otto.ini").exists());
    SampleRegionSetting authored;authored.name="native-only";authored.regionStartSeconds=.25;
    authored.regionEndSeconds=.75;authored.alignmentSeconds=.3;
    authored.additionalAliases.add("native-alternate");
    check("fixture_publish_native_authority",OttoMaterialIO::saveAudio(native.getChildFile("otto.ini"),nativeAudio,{authored},error));
    rows.clear();
    check("root_otto_dominates_conflicting_oto",OttoMaterialMigration::readAudioAnnotations(nativeAudio,native,UtauMode::classic,rows,error)
        &&rows.size()==1&&rows[0].name=="native-only");
    check("root_otto_missing_audio_section_blocks_oto_fallback",OttoMaterialMigration::readAudioAnnotations(missingAudio,native,UtauMode::classic,rows,error)&&rows.empty());
    backend::UtauRenderer::invalidateVoicebankCache();
    const auto nativeResolved=backend::UtauRenderer::resolveVoiceSample(native,"native-only",60);
    const auto alternateResolved=backend::UtauRenderer::resolveVoiceSample(native,"native-alternate",60);
    const auto obsoleteResolved=backend::UtauRenderer::resolveVoiceSample(native,"obsolete",60);
    check("renderer_uses_native_ranges_and_additional_aliases",nativeResolved.found&&alternateResolved.found
        &&nativeResolved.file==nativeAudio&&std::abs(nativeResolved.offsetSeconds-.25)<1.e-8
        &&std::abs(nativeResolved.endSeconds-.75)<1.e-8&&std::abs(alternateResolved.offsetSeconds-.25)<1.e-8);
    check("renderer_never_revives_obsolete_oto_alias",!obsoleteResolved.found);
    auto absentSection=MaterialAnnotationSession::open(missingAudio,native.getChildFile("otto.ini"),error);
    check("session_obeys_root_missing_section_authority",absentSection&&absentSection->rows().empty());
    check("scoped_oto_ensure_preserves_nested_bank",OttoMaterialMigration::ensureNativeFolder(scoped,error)&&!scoped.getChildFile("otto.ini").exists());
    rows.clear();
    check("nearest_child_oto_governs",OttoMaterialMigration::readAudioAnnotations(scopedAudio,scoped,UtauMode::classic,rows,error)
        &&rows.size()==1&&rows[0].name=="nearest-child"&&std::abs(rows[0].regionStartSeconds-.2)<1.e-9);
    check("corrupt_manifest_fixture",corrupt.getChildFile("otto.ini").replaceWithText("this is not a valid OTTO document\n"));
    rows=repeat;const auto retained=identities(rows);
    check("corrupt_otto_never_falls_back_to_valid_oto",!OttoMaterialMigration::readAudioAnnotations(corruptAudio,corrupt,UtauMode::classic,rows,error)
        &&error.isNotEmpty()&&identities(rows)==retained);
    check("corrupt_otto_ensure_refused",!OttoMaterialMigration::ensureNativeFolder(corrupt,error));
    auto badSession=MaterialAnnotationSession::open(corruptAudio,corrupt.getChildFile("otto.ini"),error);
    check("corrupt_otto_session_refused",!badSession&&error.isNotEmpty());
    juce::PropertiesFile::Options options;options.applicationName="material-format-auto-smoke";
    options.storageFormat=juce::PropertiesFile::storeAsXML;
    juce::PropertiesFile properties(root.getChildFile("isolated-settings.xml"),options);
    I18n strings;strings.setLanguage(I18n::Language::enUS);
    {
        AssetManagerComponent manager(strings,properties);manager.setSize(400,650);
        check("register_oto_bank_does_not_convert",manager.diagnosticRegisterFolder(otoOnly)>=0&&!otoOnly.getChildFile("otto.ini").exists());
        check("register_bare_uses_existing_empty_manifest",manager.diagnosticRegisterFolder(bare)>=0
            &&bare.getChildFile("otto.ini").loadFileAsString()==emptyBytes);
        check("register_native_and_scoped_banks",manager.diagnosticRegisterFolder(native)>=0&&manager.diagnosticRegisterFolder(scoped)>=0);
        const auto count=manager.diagnosticFolderCount();
        const auto savedPrefs=properties.getValue("assets.materialFolders");
        check("isolated_preferences_written",properties.saveIfNeeded()&&properties.getFile().existsAsFile());
        const auto prefBytes=properties.getFile().loadFileAsString();
        check("failed_registration_does_not_add_folder_or_pref",manager.diagnosticRegisterFolder(corrupt)<0
            &&manager.diagnosticFolderCount()==count&&properties.getValue("assets.materialFolders")==savedPrefs
            &&properties.getFile().loadFileAsString()==prefBytes);
        check("auto_kind_oto_is_two",manager.otoImportKind(otoAudio)==2&&manager.otoImportKind(scopedAudio)==2);
        check("auto_kind_native_is_one",manager.otoImportKind(nativeAudio)==1&&manager.otoImportKind(missingAudio)==1&&manager.otoImportKind(bareAudio)==1);
        for(const int manual:{0,1,2,3,4}){manager.setOtoImportKind(otoOnly,manual);manager.setOtoImportKind(native,manual);}
        check("deprecated_reader_selection_is_noop",manager.otoImportKind(otoAudio)==2&&manager.otoImportKind(nativeAudio)==1
            &&properties.getValue("assets.materialFolders")==savedPrefs&&properties.getFile().loadFileAsString()==prefBytes);
        int manualSelectors=0;
        std::function<void(juce::Component&)> countSelectors=[&](juce::Component& component)
        {
            for(int i=0;i<component.getNumChildComponents();++i)
            {auto* child=component.getChildComponent(i);if(dynamic_cast<juce::ComboBox*>(child)&&child->isVisible())++manualSelectors;countSelectors(*child);}
        };
        countSelectors(manager);
        check("manager_has_no_visible_manual_reader_selector",manualSelectors==0);
        check("oto_bank_still_unconverted_after_manager_queries",!otoOnly.getChildFile("otto.ini").exists());
    }
    check("explicit_oto_session_save_publishes_native",imported&&imported->save(error)&&otoOnly.getChildFile("otto.ini").existsAsFile());
    auto reopened=MaterialAnnotationSession::open(otoAudio,otoOnly.getChildFile("otto.ini"),error);
    check("explicit_save_preserves_imported_pronunciation_ids",reopened&&identities(reopened->rows())==importedIds);
    rows.clear();
    check("native_read_after_conversion_preserves_import_ids",OttoMaterialMigration::readAudioAnnotations(otoAudio,otoOnly,UtauMode::classic,rows,error)
        &&identities(rows)==importedIds);
    bool unchanged=true;
    for(const auto& original:originals){juce::MemoryBlock after;unchanged&=original.file.loadFileAsData(after)&&after==original.bytes;}
    check("all_original_wav_and_oto_bytes_unchanged",unchanged);
    check("no_automatic_csv_created",!SampleSettings::sidecarFor(otoAudio).exists()&&!SampleSettings::sidecarFor(bareAudio).exists()
        &&!SampleSettings::sidecarFor(nativeAudio).exists()&&!SampleSettings::sidecarFor(scopedAudio).exists());
    SampleSettings::setMaterialRoots(restore.before);
    check("process_material_roots_restored",SampleSettings::materialRoots()==restore.before);
    return finish();
}
}
