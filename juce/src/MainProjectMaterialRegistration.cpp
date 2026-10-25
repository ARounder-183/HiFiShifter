#include "MainComponent.h"
#include "ProjectMaterialRegistrationFacts.h"
#include <thread>

namespace hachi
{
struct MainComponent::ProjectRegistrationOperation
{
    ProjectData snapshot;
    std::uint64_t revision=0;
    juce::String fingerprint,error;
    std::vector<ProjectMaterialRegistration::Locator> owners;
    std::atomic_bool cancel{false};
    std::thread worker;
    std::unique_ptr<ProjectMaterialRegistration::Lease> lease;
    juce::Component::SafePointer<juce::AlertWindow> prompt;
};
namespace
{
std::vector<ProjectMaterialRegistration::Locator> unregisteredProjectOwners(const ProjectData& data,const std::atomic_bool& cancel)
{
    std::vector<ProjectMaterialRegistration::Locator> result;
    struct Read{bool valid=false;std::vector<SampleRegionSetting> rows;};std::map<juce::String,Read> cache;
    for(const auto& track:data.tracks)if(trackShowsAllNativeRegions(track)&&!track.accompaniment)
        for(const auto& parent:track.clips)
        {
            const auto inspect=[&](const ClipData& owner,const juce::String& part)
            {
                if(cancel.load()||!owner.sourceFile.existsAsFile())return;
                bool hasNotes=false,bound=true;const auto manifest=OttoMaterialIO::findManifest(owner.sourceFile);
                for(const auto& note:parent.notes)if(note.clipPartId==part)
                {
                    hasNotes=true;
                    if(note.materialAudioId.isEmpty()||note.materialRegionId.isEmpty()||!manifest.existsAsFile()){bound=false;continue;}
                    const auto key=manifest.getFullPathName()+"|"+owner.sourceFile.getFullPathName()+"|"+note.materialAudioId;
                    auto found=cache.find(key);
                    if(found==cache.end())
                    {Read read;juce::String error;read.valid=OttoMaterialIO::loadAudio(manifest,owner.sourceFile,read.rows,error,nullptr,note.materialAudioId);found=cache.emplace(key,std::move(read)).first;}
                    const auto& read=found->second;
                    bound&=read.valid&&std::any_of(read.rows.begin(),read.rows.end(),[&](const auto& r){return r.materialAudioId==note.materialAudioId&&r.materialRegionId==note.materialRegionId;});
                }
                if(hasNotes&&!bound)result.push_back({track.id,parent.id,part,{}});
            };
            if(parent.parts.empty())inspect(parent,{});else for(const auto& part:parent.parts)inspect(part,part.id);
        }
    return result;
}
}

void MainComponent::cancelProjectMaterialRegistration()
{
    auto operation=std::exchange(projectRegistrationOperation,{});
    if(operation){operation->cancel=true;if(operation->prompt&&operation->prompt->isCurrentlyModal())operation->prompt->exitModalState(0);}
    projectRegistrationChooser.reset();
    // IO phases NEVER wait for a UI callback. Joining cannot form a GUI cycle.
    if(operation&&operation->worker.joinable())operation->worker.join();
    if(operation&&operation->lease){juce::String error;operation->lease->rollback(error);}
}

void MainComponent::offerMaterialFolderForImport(const ProjectData& imported)
{
    if(closingUI||!preferences)return;cancelProjectMaterialRegistration();
    juce::ignoreUnused(imported); // current live snapshot is the one adoption will guard
    auto operation=std::make_shared<ProjectRegistrationOperation>();operation->snapshot=project.snapshot();operation->revision=project.revisionNumber();
    operation->fingerprint=ProjectModel::registrationMusicSourceFingerprint(operation->snapshot);projectRegistrationOperation=operation;
    projectRegistrationTrace="offer_start revision="+juce::String(static_cast<juce::int64>(operation->revision))+" tracks="+juce::String(static_cast<int>(operation->snapshot.tracks.size()));
    juce::Component::SafePointer<MainComponent> safe(this);
    operation->worker=std::thread([safe,operation]
    {
        operation->owners=unregisteredProjectOwners(operation->snapshot,operation->cancel);
        juce::MessageManager::callAsync([safe,operation]
        {
            if(!safe||safe->closingUI||operation->cancel||safe->projectRegistrationOperation!=operation)return;
            if(operation->worker.joinable())operation->worker.join();
            safe->projectRegistrationTrace+="\nprobe owners="+juce::String(static_cast<int>(operation->owners.size()))+" expected_revision="+juce::String(static_cast<juce::int64>(operation->revision))
                +" live_revision="+juce::String(static_cast<juce::int64>(safe->project.revisionNumber()))+" music_fp_equal="+juce::String(ProjectModel::registrationMusicSourceFingerprint(safe->project.snapshot())==operation->fingerprint?1:0);
            juce::Logger::writeToLog(safe->projectRegistrationTrace);
            if(operation->owners.empty()){safe->projectRegistrationOperation.reset();return;}
            if(safe->project.revisionNumber()!=operation->revision){safe->cancelProjectMaterialRegistration();return;}
            auto* dialog=new juce::AlertWindow(safe->strings.text("Register project source materials"),
                safe->strings.text("Optionally copy the existing unregistered recordings into a NEW material folder and keep each carried clip/part annotation as an independent variant named '-'. This does not recover words or syllables. Original C/P, source clocks, pitch, envelopes and local edits remain exact. Skip keeps the loaded project unchanged; project files are saved only when you choose Save."),
                juce::MessageBoxIconType::QuestionIcon,safe.getComponent());
            operation->prompt=dialog;
            dialog->setComponentID("project-material-registration-confirm");
            dialog->addButton(safe->strings.text("Register copies..."),1);
            dialog->addButton(safe->strings.text("Skip"),0,juce::KeyPress(juce::KeyPress::escapeKey));
            dialog->enterModalState(true,juce::ModalCallbackFunction::create([safe,operation,dialog](int choice)
            {
                std::unique_ptr<juce::AlertWindow> owned(dialog);
                if(!safe||safe->closingUI||operation->cancel||safe->projectRegistrationOperation!=operation)return;
                if(choice!=1){safe->cancelProjectMaterialRegistration();return;}
                safe->chooseProjectMaterialRegistrationDestination(operation);
            }),false);
        });
    });
}

void MainComponent::chooseProjectMaterialRegistrationDestination(const std::shared_ptr<ProjectRegistrationOperation>& operation)
{
    if(projectRegistrationOperation!=operation||operation->cancel||closingUI)return;
    if(project.revisionNumber()!=operation->revision){cancelProjectMaterialRegistration();showError(strings.text("The project changed before registration. Reopen the registration action for the current project."));return;}
    juce::Component::SafePointer<MainComponent> safe(this);
    const auto chosen=[safe,operation](const juce::File& destination)
    {
        if(!safe||safe->closingUI||operation->cancel||safe->projectRegistrationOperation!=operation)return;
        if(!destination.isDirectory()){safe->cancelProjectMaterialRegistration();return;}
        safe->stageProjectMaterialRegistration(operation,destination);
    };
    if(projectRegistrationFolderPickerForTest){projectRegistrationFolderPickerForTest(chosen);return;}
    projectRegistrationChooser=std::make_unique<juce::FileChooser>(strings.text("Choose parent folder for NEW project materials"),juce::File{},juce::String{},true);
    projectRegistrationChooser->launchAsync(juce::FileBrowserComponent::openMode|juce::FileBrowserComponent::canSelectDirectories,
        [chosen](const juce::FileChooser& chooser){chosen(chooser.getResult());});
}

void MainComponent::stageProjectMaterialRegistration(const std::shared_ptr<ProjectRegistrationOperation>& operation,const juce::File& destination)
{
    if(closingUI||projectRegistrationOperation!=operation||operation->cancel)return;
    if(project.revisionNumber()!=operation->revision){cancelProjectMaterialRegistration();showError(strings.text("The project changed; registration was cancelled without changing references."));return;}
    if(operation->worker.joinable())operation->worker.join();
    const auto legal=juce::File::createLegalFileName(operation->snapshot.name);const auto child=(legal.isEmpty()?juce::String("project"):legal)+" materials "+juce::Uuid().toString().substring(0,8);
    juce::Component::SafePointer<MainComponent> safe(this);
    operation->worker=std::thread([safe,operation,destination,child]
    {
        const auto cancel=[operation]{return operation->cancel.load();};
        const ProjectMaterialRegistration::OwnerScope scope=operation->owners;
        const auto facts=ProjectMaterialRegistrationFacts::derive(operation->snapshot,ProjectMaterialRegistrationFacts::Policy::ConfirmedCarriedPlacements,scope);
        if(!facts.ready)operation->error=facts.diagnostics.joinIntoString("\n");
        else
        {
            auto plan=ProjectMaterialRegistration::prepare(operation->snapshot,operation->revision,destination,child,facts.variants,
                ProjectModel::registrationMusicSourceFingerprint,cancel,scope);
            if(!plan.ready)operation->error=plan.diagnostics.joinIntoString("\n");
            else operation->lease=ProjectMaterialRegistration::stage(plan,operation->error,cancel);
        }
        juce::MessageManager::callAsync([safe,operation]
        {
            if(!safe||safe->closingUI||operation->cancel||safe->projectRegistrationOperation!=operation)return;
            if(operation->worker.joinable())operation->worker.join();
            if(!operation->lease){const auto error=operation->error;safe->cancelProjectMaterialRegistration();safe->showError(error);return;}
            if(safe->project.revisionNumber()!=operation->revision||ProjectModel::registrationMusicSourceFingerprint(safe->project.snapshot())!=operation->fingerprint)
            {safe->cancelProjectMaterialRegistration();safe->showError(safe->strings.text("The project/source snapshot changed. Only the owned unadopted folder was rolled back."));return;}
            safe->publishProjectMaterialRegistration(operation);
        });
    });
}

void MainComponent::publishProjectMaterialRegistration(const std::shared_ptr<ProjectRegistrationOperation>& operation)
{
    if(closingUI||projectRegistrationOperation!=operation||operation->cancel||!operation->lease)return;
    if(operation->worker.joinable())operation->worker.join();juce::Component::SafePointer<MainComponent> safe(this);
    operation->worker=std::thread([safe,operation]
    {
        const bool published=operation->lease->publish(operation->error,[operation]{return operation->cancel.load();});
        juce::MessageManager::callAsync([safe,operation,published]
        {
            if(!safe||safe->closingUI||operation->cancel||safe->projectRegistrationOperation!=operation)return;
            if(operation->worker.joinable())operation->worker.join();
            if(!published){const auto error=operation->error;safe->cancelProjectMaterialRegistration();safe->showError(error);return;}
            const auto& plan=operation->lease->plan();juce::String error;
            if(!safe->project.adoptRegisteredMaterialBindings(operation->revision,operation->fingerprint,plan.owners,plan.bindings,error))
            {safe->cancelProjectMaterialRegistration();safe->showError(error);return;}
            // COMMIT BOUNDARY: no checks, IO or fallible preference write may
            // intervene before this in-memory no-fail permanent retain seam.
            operation->lease->retainAfterAdoption();
            const auto folder=plan.publishedFolder;
            const bool registered=safe->registerRetainedMaterialRoot(folder,error);
            safe->projectRegistrationOperation.reset();safe->projectRegistrationChooser.reset();
            if(!registered)
            {
                if(std::find(safe->projectRegistrationRetryRoots.begin(),safe->projectRegistrationRetryRoots.end(),folder)==safe->projectRegistrationRetryRoots.end())safe->projectRegistrationRetryRoots.push_back(folder);
                safe->showError(safe->strings.text("Project references and the copied audio/OTTO are safely retained at:\n")+folder.getFullPathName()+safe->strings.text("\nLibrary preference registration failed: ")+error+safe->strings.text("\nUse File > Retry material folder registration. No project file was automatically saved."));
            }
            else safe->statusLabel.setText(safe->strings.text("Project source materials registered. Musical edits were preserved; choose Save to persist new references."),juce::dontSendNotification);
            safe->menuItemsChanged();
        });
    });
}

bool MainComponent::registerRetainedMaterialRoot(const juce::File& root,juce::String& error)
{
    if(projectRegistrationRootWriterForTest)return projectRegistrationRootWriterForTest(root,error);
    setAssetManagerVisible(true);
    if(!assetManager){error=strings.text("Material manager is unavailable");return false;}
    if(!assetManager->registerMaterialRoot(root,error))return false;
    auto roots=SampleSettings::materialRoots();roots.addIfNotAlreadyThere(root.getFullPathName());SampleSettings::setMaterialRoots(roots);
    return true;
}
void MainComponent::retryProjectMaterialRootRegistration()
{
    std::vector<juce::File> pending;juce::StringArray errors;
    for(const auto& root:projectRegistrationRetryRoots){juce::String error;if(!registerRetainedMaterialRoot(root,error)){pending.push_back(root);errors.add(root.getFullPathName()+": "+error);}}
    projectRegistrationRetryRoots=std::move(pending);menuItemsChanged();if(!errors.isEmpty())showError(errors.joinIntoString("\n"));
}
}
