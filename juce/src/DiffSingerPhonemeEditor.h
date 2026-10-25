#pragma once
#include "I18n.h"
#include <juce_gui_extra/juce_gui_extra.h>
#include "DiffSingerPhonemeDisplay.h"
#include "Theme.h"
#include <cmath>
#include <map>
#include <limits>
#include <vector>

namespace hachi
{
// Staged edits: one Apply is one project undo step. The shared boundary is
// owned by the following token, even when it precedes that token's MIDI note.
class DiffSingerPhonemeEditor final : public juce::Component
{
public:
    using Timings = std::vector<std::pair<juce::String, juce::String>>;
    struct Row
    {
        juce::String id, token, kind, context;
        juce::var tokens;
        int index = -1, phrase = 0;
        double start = 0, end = 0, predicted = 0, noteStart = 0;
        bool editable = false, edited = false;
    };
    DiffSingerPhonemeEditor(const juce::var& result, const juce::var& notes,
                           std::function<bool(const Timings&)> commit,
                           std::function<void()> audition = {}, juce::String focus = {},
                           juce::StringArray shownNotes = {})
        : noteData(notes), visibleIds(std::move(shownNotes)), applyEdits(std::move(commit)), play(std::move(audition)), canvas(*this)
    {
        dt = (double) result["frame_seconds"];
        if (!std::isfinite(dt) || dt <= 0) dt = 512.0 / 44100;
        if (auto* values = result["phonemes"].getArray())
            for (const auto& v : *values)
                rows.push_back({v["id"].toString(), v["token"].toString(), v["kind"].toString(),
                    v["context"].toString(), v["tokens"], (int) v["index"], (int) v["phrase"],
                    (double) v["start"], (double) v["end"], (double) v["predicted_start"],
                    (double) v["note_start"], (bool) v["editable"], (bool) v["edited"]});
        auto first = std::numeric_limits<double>::max(), last = -first;
        for (const auto& row : rows) if (noteVisible(row.id))
        { first = std::min(first,row.start); last = std::max(last,row.end); }
        if (const auto* values=noteData.getArray()) for (const auto& note:*values)
            if (noteVisible(note["id"].toString()))
            {
                first=std::min(first,(double)note["start"]);
                last=std::max(last,(double)note["start"]+(double)note["duration"]);
            }
        from = first <= last ? first-.08 : 0;
        until = first <= last ? last+.08 : 1;
        for (std::size_t i = 0; i < rows.size(); ++i)
            if (noteVisible(rows[i].id) && focus.isNotEmpty() && rows[i].id == focus && selected < 0) selected = (int) i;
        addAndMakeVisible(viewport); viewport.setViewedComponent(&canvas, false);
        viewport.setScrollBarsShown(false, true);
        for (auto* b : { &undo, &redo, &reset, &resetAll, &apply, &preview, &cancel }) addAndMakeVisible(*b);
        addAndMakeVisible(keepBeat); keepBeat.setToggleState(true, juce::dontSendNotification);
        keepBeat.setTooltip(u("锁定元音起点，保留 MIDI 拍点；关闭后也可拖动元音起点"));
        keepBeat.onClick = [this] { refresh(); };
        addAndMakeVisible(zoom); zoom.setRange(80, 2000, 10); zoom.setValue(300);
        zoom.setTextBoxStyle(juce::Slider::TextBoxRight, false, 75, 24); zoom.setTextValueSuffix(" px/s");
        zoom.onValueChange = [this] { resizeCanvas(); };
        addAndMakeVisible(value); value.setInputRestrictions(16, "0123456789.-");
        value.onReturnKey = [this]
        {
            if (selected < 0) return;
            const auto text = value.getText().trim();
            const auto numeric = text.isNotEmpty() && text.containsAnyOf("0123456789")
                && text.retainCharacters("0123456789.-") == text;
            if (numeric) moveBoundary(selected, rows[(std::size_t) selected].noteStart + text.getDoubleValue() / 1000);
            refresh();
        };
        addAndMakeVisible(selection); addAndMakeVisible(hint);
        hint.setText(u("拖动分界线改变相邻音素时长；双击恢复预测。数字为相对音符起点的毫秒数，允许负值。悬停音素查看完整标识。"), juce::dontSendNotification);
        if (result["ignored_timings"].size() > 0 || (bool) result["timing_clamped"])
            hint.setText(u("音源、歌词或音符长度已变化：失配的旧编辑已忽略，越界边界已限制；应用后保存当前结果。"), juce::dontSendNotification);
        undo.onClick = [this] { historyStep(false); };
        redo.onClick = [this] { historyStep(true); };
        reset.onClick = [this] { resetPrediction(false); };
        resetAll.onClick = [this] { resetPrediction(true); };
        if (!visibleIds.isEmpty()) resetAll.setButtonText(u("恢复当前预测"));
        apply.onClick = [this] { finish(false); };
        preview.onClick = [this] { finish(true); }; preview.setEnabled((bool) play);
        cancel.onClick = [this] { close(); };
        setWantsKeyboardFocus(true); setSize(1060, 350);
        if (!visibleIds.isEmpty())
            zoom.setValue(juce::jlimit(80.0,2000.0,(viewport.getWidth()-48)/std::max(.1,until-from)));
        refresh();
        if (selected >= 0) viewport.setViewPosition(std::max(0, (int) x(rows[(std::size_t) selected].start)-80), 0);
    }

    ~DiffSingerPhonemeEditor() override { viewport.setViewedComponent(nullptr, false); }

    Timings timings() const
    {
        Timings result;
        if (const auto* notes = noteData.getArray())
            for (const auto& note : *notes)
            {
                const auto id = note["id"].toString();
                if (!noteVisible(id)) continue;
                const Row* first = nullptr; bool any = false;
                juce::Array<juce::var> starts;
                for (const auto& row : rows) if (row.id == id)
                {
                    if (first == nullptr) first = &row;
                    any |= row.edited;
                    starts.add(row.edited ? juce::var(row.start-row.noteStart) : juce::var{});
                }
                if (first != nullptr && any)
                {
                    auto* object = new juce::DynamicObject();
                    object->setProperty("context", first->context);
                    object->setProperty("tokens", first->tokens);
                    object->setProperty("starts", starts);
                    result.emplace_back(id, juce::JSON::toString(juce::var(object), true));
                }
                else result.emplace_back(id, juce::String{});
            }
        return result;
    }
    bool moveBoundary(int index, double seconds, bool remember = true)
    {
        if (!canMove(index) || !std::isfinite(seconds)) return false;
        auto& row = rows[(std::size_t) index];
        const auto& previous = rows[(std::size_t) index-1];
        auto origin = row.start;
        for (const auto& item : rows) if (item.phrase == row.phrase) { origin = item.start; break; }
        const auto target = juce::jlimit(previous.start+dt, row.end-dt,
            origin+std::round((seconds-origin)/dt)*dt);
        if (std::abs(target-row.start) < 1e-9) return false;
        if (remember) rememberEdit();
        row.start = target; row.edited = true; rows[(std::size_t) index-1].end = target;
        selected = index; refresh(); return true;
    }
    const std::vector<Row>& phonemes() const { return rows; }
    bool noteVisible(const juce::String& id) const
    { return visibleIds.isEmpty() || visibleIds.contains(id); }
    std::vector<Row> displayedPhonemes() const
    {
        std::vector<Row> displayed;
        for (const auto& row:rows) if (noteVisible(row.id)) displayed.push_back(row);
        const auto continuations=continuationPhonemes();
        displayed.insert(displayed.end(),continuations.begin(),continuations.end());
        return displayed;
    }
    // A slur has no independent token or duration boundary. Show its share of
    // the sustained vowel without allowing edits to a hidden lyric note.
    std::vector<Row> continuationPhonemes() const
    {
        std::vector<Row> displayed;
        if (visibleIds.isEmpty()) return displayed;
        if (const auto* notes=noteData.getArray()) for (const auto& note:*notes)
        {
            const auto id=note["id"].toString(), lyric=note["lyric"].toString().trim();
            if (!noteVisible(id) || (lyric!="+" && lyric!="-")) continue;
            const auto start=(double)note["start"], end=start+(double)note["duration"];
            for (const auto& row:rows)
            {
                if (noteVisible(row.id) || row.kind!="V" || row.end<=start || row.start>=end) continue;
                auto slice=row; slice.id=id; slice.noteStart=start;
                slice.start=std::max(start,row.start); slice.end=std::min(end,row.end);
                slice.editable=false; slice.edited=false; displayed.push_back(std::move(slice));
            }
        }
        return displayed;
    }
    void resetPrediction(bool all)
    {
        if (!all && selected < 0) return;
        rememberEdit();
        if (all)
        {
            for (auto& row : rows) if (noteVisible(row.id))
            { row.edited = false; row.start = row.predicted; }
        }
        else
        {
            auto& row = rows[(std::size_t) selected];
            row.edited = false; row.start = row.predicted;
        }
        // Re-project manual runs between unchanged prediction anchors, exactly
        // as the renderer does when a neighbouring boundary is reset.
        for (std::size_t a = 0; a < rows.size();)
        {
            auto b = a+1;
            while (b < rows.size() && rows[b].phrase == rows[a].phrase && rows[b].edited) ++b;
            for (auto i = a+1; i < b; ++i)
            {
                const auto end = b < rows.size() && rows[b].phrase == rows[a].phrase
                    ? rows[b].start : rows[b-1].end;
                rows[i].start = juce::jlimit(rows[i-1].start+dt, end-(b-i)*dt, rows[i].start);
            }
            a = b;
        }
        for (std::size_t i = 1; i < rows.size(); ++i)
            if (rows[i-1].phrase == rows[i].phrase) rows[i-1].end = rows[i].start;
        refresh();
    }
    void resized() override
    {
        auto bounds = getLocalBounds().reduced(12);
        auto top = bounds.removeFromTop(30);
        keepBeat.setBounds(top.removeFromLeft(195)); zoom.setBounds(top.removeFromRight(290));
        undo.setBounds(top.removeFromLeft(68).reduced(2)); redo.setBounds(top.removeFromLeft(68).reduced(2));
        hint.setBounds(bounds.removeFromTop(32));
        auto bottom = bounds.removeFromBottom(34);
        cancel.setBounds(bottom.removeFromRight(75).reduced(2));
        apply.setBounds(bottom.removeFromRight(75).reduced(2));
        preview.setBounds(bottom.removeFromRight(115).reduced(2));
        resetAll.setBounds(bottom.removeFromLeft(120).reduced(2));
        reset.setBounds(bottom.removeFromLeft(120).reduced(2));
        auto info = bounds.removeFromBottom(36);
        value.setBounds(info.removeFromRight(115).reduced(2)); selection.setBounds(info);
        viewport.setBounds(bounds); resizeCanvas();
    }
    void paint(juce::Graphics& g) override { g.fillAll(Palette::panel); }
    bool keyPressed(const juce::KeyPress& key) override
    {
        if (key.getModifiers().isCommandDown() && key.getKeyCode() == 'Z')
        { historyStep(key.getModifiers().isShiftDown()); return true; }
        if (key.getModifiers().isCommandDown() && key.getKeyCode() == 'Y')
        { historyStep(true); return true; }
        return false;
    }
private:
    static juce::String u(const char* text) { return I18n::uiText(text); }
    float x(double time) const { return (float) (24+(time-from)*zoom.getValue()); }
    double time(float px) const { return from+(px-24)/zoom.getValue(); }
    bool canMove(int i) const
    {
        return i > 0 && i < (int) rows.size() && rows[(std::size_t) i].editable
            && noteVisible(rows[(std::size_t) i].id)
            && rows[(std::size_t) i-1].phrase == rows[(std::size_t) i].phrase
            && !(keepBeat.getToggleState() && rows[(std::size_t) i].kind == "V");
    }
    void rememberEdit() { undoRows.push_back(rows); if (undoRows.size()>100) undoRows.erase(undoRows.begin()); redoRows.clear(); }
    void historyStep(bool forward)
    {
        auto& source = forward ? redoRows : undoRows; auto& dest = forward ? undoRows : redoRows;
        if (source.empty()) return; dest.push_back(rows); rows=std::move(source.back()); source.pop_back(); refresh();
    }
    void refresh()
    {
        // A typed earlier consonant can leave the initially fitted viewport.
        // Expand after a drag ends so its mouse-to-time mapping stays stable.
        if (!visibleIds.isEmpty() && canvas.drag < 0)
        {
            const auto oldFrom=from, oldUntil=until;
            for (const auto& row:rows) if (noteVisible(row.id))
            { from=std::min(from,row.start-.08); until=std::max(until,row.end+.08); }
            if (from!=oldFrom || until!=oldUntil) resizeCanvas();
        }
        undo.setEnabled(!undoRows.empty()); redo.setEnabled(!redoRows.empty());
        reset.setEnabled(selected >= 0 && rows[(std::size_t) selected].editable);
        value.setEnabled(canMove(selected));
        if (selected >= 0)
        {
            const auto& row = rows[(std::size_t) selected];
            selection.setText(row.token + "  [" + row.kind + "]  " + u("时长 ")
                + juce::String((row.end-row.start)*1000, 1)+u(" ms；起点相对音符（ms）")
                + (row.edited ? u(" · 已修改") : u(" · 预测")), juce::dontSendNotification);
            value.setText(juce::String((row.start-row.noteStart)*1000, 2), false);
        }
        else selection.setText(u("C 辅音 · V 元音 · S 休止；黄色圆点为手动边界，灰线为预测位置"), juce::dontSendNotification);
        canvas.repaint();
    }
    void resizeCanvas()
    { canvas.setSize(std::max(viewport.getWidth()-2, (int) x(until)+24), std::max(140, viewport.getHeight()-18)); }
    void close() { if (auto* window = findParentComponentOfClass<juce::DialogWindow>()) window->exitModalState(0); }
    void finish(bool listen)
    {
        if (!applyEdits || !applyEdits(timings()))
        { hint.setText(u("工程已变化，未写入此次结果。请关闭后重新打开音素编辑器。"), juce::dontSendNotification); return; }
        if (listen && play) play(); close();
    }
    struct Canvas final : juce::Component, juce::SettableTooltipClient
    {
        DiffSingerPhonemeEditor& editor; int drag = -1; bool saved = false;
        explicit Canvas(DiffSingerPhonemeEditor& owner) : editor(owner) {}
        void paint(juce::Graphics& g) override
        {
            auto& e = editor; g.fillAll(Palette::graphBackground); g.setFont(13);
            const auto clip = g.getClipBounds();
            const auto step = e.zoom.getValue() < 180 ? 1.0 : .25;
            for (double t=std::floor(e.time((float) clip.getX())/step)*step; e.x(t)<clip.getRight(); t+=step)
            {
                g.setColour(juce::Colour(0xff35414a)); g.drawVerticalLine((int)e.x(t),20,(float)getHeight());
                g.setColour(juce::Colours::lightgrey); g.drawText(juce::String(t,2), (int)e.x(t)+3,0,55,20,juce::Justification::left);
            }
            if (const auto* notes=e.noteData.getArray()) for(const auto& note:*notes)
            {
                if (!e.noteVisible(note["id"].toString())) continue;
                const auto start=(double)note["start"], end=start+(double)note["duration"];
                auto rect=juce::Rectangle<float>(e.x(start),29,e.x(end)-e.x(start),27);
                if (!clip.toFloat().intersects(rect)) continue;
                g.setColour(juce::Colour(0xff435b6d)); g.fillRoundedRectangle(rect.reduced(1),3);
                g.setColour(juce::Colours::white); g.drawText(note["lyric"].toString(),rect.reduced(4).toNearestInt(),juce::Justification::centredLeft);
                g.setColour(juce::Colour(0x558fc4e5)); g.drawVerticalLine((int)e.x(start),57,(float)getHeight());
            }
            for(std::size_t i=0;i<e.rows.size();++i)
            {
                const auto& r=e.rows[i];
                if (!e.noteVisible(r.id)) continue;
                auto rect=juce::Rectangle<float>(e.x(r.start),76,e.x(r.end)-e.x(r.start),65);
                if (!clip.toFloat().intersects(rect.expanded(6))) continue;
                g.setColour(r.kind=="C"?juce::Colour(0xff467584):r.kind=="V"?juce::Colour(0xff496f57):juce::Colour(0xff41464c));
                g.fillRect(rect.reduced(1,0));
                g.setColour(juce::Colours::white); g.drawFittedText(diffSingerPhonemeLabel(r.token),rect.reduced(2,3).withHeight(25).toNearestInt(),juce::Justification::centred,1,.65f);
                g.setFont(11); g.drawText(juce::String((r.end-r.start)*1000,0)+" ms",rect.reduced(3).withTrimmedTop(29).toNearestInt(),juce::Justification::centred); g.setFont(13);
                if(r.edited) { g.setColour(juce::Colour(0xff85959c)); g.drawVerticalLine((int)e.x(r.predicted),64,148); }
                if(r.editable)
                {
                    g.setColour(r.edited?juce::Colour(0xffffce70):juce::Colours::lightgrey);
                    g.drawLine(e.x(r.start),69,e.x(r.start),146,i==(std::size_t)e.selected?3.0f:1.0f);
                    g.fillEllipse(e.x(r.start)-4,65,8,8);
                }
                if(i==(std::size_t)e.selected) {g.setColour(juce::Colour(0xffffce70));g.drawRect(rect,2);}
            }
            for (const auto& row:e.continuationPhonemes())
            {
                const auto rect=juce::Rectangle<float>(e.x(row.start),76,e.x(row.end)-e.x(row.start),65);
                if (!clip.toFloat().intersects(rect)) continue;
                g.setColour(juce::Colour(0xff496f57));g.fillRect(rect.reduced(1,0));
                g.setColour(juce::Colours::white);
                g.drawFittedText(diffSingerPhonemeLabel(row.token)+u(" · 延音"),rect.reduced(3).toNearestInt(),juce::Justification::centred,1);
            }
        }
        int hit(float px, bool handles) const
        {
            int found=-1; float best=9;
            for(std::size_t i=0;i<editor.rows.size();++i)
            {
                const auto& row=editor.rows[i];
                if (!editor.noteVisible(row.id)) continue;
                const auto distance=std::abs(editor.x(row.start)-px);
                if(handles && row.editable && distance<best) {best=distance;found=(int)i;}
                if(!handles && px>=editor.x(row.start) && px<editor.x(row.end)) found=(int)i;
            }
            return found;
        }
        void mouseDown(const juce::MouseEvent& event) override
        {
            editor.grabKeyboardFocus(); if(event.y<60) return;
            drag=hit(event.position.x,true); saved=false;
            editor.selected=drag>=0?drag:hit(event.position.x,false); editor.refresh();
        }
        void mouseDrag(const juce::MouseEvent& event) override
        { if(editor.moveBoundary(drag,editor.time(event.position.x),!saved)) saved=true; }
        void mouseUp(const juce::MouseEvent&) override { drag=-1; editor.refresh(); }
        void mouseDoubleClick(const juce::MouseEvent&) override { editor.resetPrediction(false); }
        void mouseMove(const juce::MouseEvent& event) override
        {
            setMouseCursor(editor.canMove(hit(event.position.x,true))?juce::MouseCursor::LeftRightResizeCursor:juce::MouseCursor::NormalCursor);
            const auto index = event.y >= 60 ? hit(event.position.x,false) : -1;
            auto tip=index >= 0 ? editor.rows[(std::size_t)index].token : juce::String{};
            if (tip.isEmpty() && event.y>=60) for (const auto& row:editor.continuationPhonemes())
                if (event.position.x>=editor.x(row.start) && event.position.x<editor.x(row.end))
                { tip=row.token+u(" · 延音，边界在原歌词音符上编辑"); break; }
            setTooltip(tip);
        }
        void mouseExit(const juce::MouseEvent&) override { setTooltip({}); }
    };
    juce::var noteData;
    juce::StringArray visibleIds;
    std::function<bool(const Timings&)> applyEdits;
    std::function<void()> play;
    std::vector<Row> rows;
    std::vector<std::vector<Row>> undoRows, redoRows;
    double dt=0, from=0, until=1;
    int selected=-1;
    juce::Viewport viewport;
    juce::TooltipWindow tooltip {this, 450};
    juce::ToggleButton keepBeat {u("保持元音拍点")};
    juce::Slider zoom;
    juce::TextEditor value;
    juce::Label selection, hint;
    juce::TextButton undo {u("撤销")}, redo {u("重做")}, reset {u("恢复选中预测")},
        resetAll {u("恢复全部预测")}, apply {u("应用")}, preview {u("应用并试听")}, cancel {u("取消")};
    Canvas canvas;
};
}
