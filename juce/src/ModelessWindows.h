#pragma once
#include <juce_gui_extra/juce_gui_extra.h>
#include "Theme.h"
#include <algorithm>
#include <memory>
#include <vector>
namespace hachi
{
// LaunchOptions::launchAsync still enters application-wide modal state. Tool
// windows use create() instead, with explicit ownership and deferred deletion.
class ModelessWindows final : private juce::ComponentListener, private juce::AsyncUpdater
{
public:
    ~ModelessWindows() override { clear(); }
    bool showExisting(const juce::String& key)
    {
        for(auto& item:windows)if(item.key==key&&item.window->isVisible())
        {item.window->toFront(true);return true;}
        return false;
    }
    juce::DialogWindow* show(juce::DialogWindow::LaunchOptions& options,const juce::String& key)
    {
        options.dialogBackgroundColour = Palette::panel;
        auto window=std::unique_ptr<juce::DialogWindow>(options.create());
        auto* result=window.get();
        result->addComponentListener(this);
        windows.push_back({key,std::move(window)});
        result->setVisible(true);result->toFront(true);
        return result;
    }
    void clear()
    {
        cancelPendingUpdate();
        for(auto& item:windows)item.window->removeComponentListener(this);
        windows.clear();
    }
    void removeClosed()
    {
        for(auto& item:windows)if(!item.window->isVisible())item.window->removeComponentListener(this);
        windows.erase(std::remove_if(windows.begin(),windows.end(),[](const auto& item){return !item.window->isVisible();}),windows.end());
    }
    template<class Content> juce::DialogWindow* findVisible(const juce::String& key = {}) const
    {
        for(const auto& item:windows)if((key.isEmpty()||item.key==key)&&item.window->isVisible()&&dynamic_cast<Content*>(item.window->getContentComponent())!=nullptr)return item.window.get();
        return nullptr;
    }
    size_t size() const {return windows.size();}
private:
    void componentVisibilityChanged(juce::Component& c) override {if(!c.isVisible())triggerAsyncUpdate();}
    void handleAsyncUpdate() override {removeClosed();}
    struct Entry {juce::String key;std::unique_ptr<juce::DialogWindow> window;};
    std::vector<Entry> windows;
};
}
