--[[
  build_reverse_window_probe.lua —— F-4 的决定性证据：倒放 take **实际播放**的源窗口。

  背景：`build_reverse_probe.lua` 测出——同一个非对称裁切（源内 [0.25, 1.25]）的 item，
  倒放后 ARA region 报 `startMod=0.75`，且 take 自己的 `D_STARTOFFS` 也被 REAPER 改成了
  0.75。于是只剩一个二选一：倒放 take 到底播的是源内 **[0.25, 1.25]**（方向翻转但内容
  不变）还是 **[0.75, 1.75]**（内容也变了）？

    * 播 [0.25, 1.25] ⇒ region 的 [0.75, 1.75] 是**已镜像坐标**（内容不变，坐标被翻）
    * 播 [0.75, 1.75] ⇒ region 的 [0.75, 1.75] 是**正向坐标**（内容与坐标一致）

  做法：只放那一个倒放 item，**不挂任何 FX**，把它的播放直接渲染成 WAV。渲染结果与
  `reverse(源[0.25,1.25])` 逐样本比对，或与 `reverse(源[0.75,1.75])` 比对 —— 命中哪个
  就是哪个。这一步问的是 REAPER 自己的播放语义，与插件无关，所以必须不带 FX。

  产物：captures/reverse-window.wav（+ captures/reverse-window-probe.json 完成信号）。
  之后跑 verify_reverse_window.py 判定。
]]

local sep = package.config:sub(1, 1)
local scriptPath = debug.getinfo(1, 'S').source:sub(2)
local scriptDir = scriptPath:match('^(.*)' .. sep .. '[^' .. sep .. ']*$') or '.'
local fixturesDir = scriptDir .. sep .. 'fixtures'
local capturesDir = scriptDir .. sep .. 'captures'
local logPath = capturesDir .. sep .. 'reverse-window-script.log'
local outPath = capturesDir .. sep .. 'reverse-window-probe.json'
local wavPath = fixturesDir .. sep .. 'phase3a-asymmetric.wav'

local SOURCE_OFFSET = 0.25
local ITEM_LENGTH = 1.0
local RENDER_RATE = 44100

local function log(message)
  local line = tostring(message)
  reaper.ShowConsoleMsg('[reverse-window] ' .. line .. '\n')
  local file = io.open(logPath, 'a')
  if file then file:write(line .. '\n'); file:close() end
end

local function finish(status, detail)
  local file = io.open(outPath, 'w')
  if file then
    file:write(string.format('{"probe":"reverse-window","status":"%s","detail":"%s"}\n', status, detail))
    file:close()
  end
  log('probe status=' .. status .. ' detail=' .. detail)
end

local truncate = io.open(logPath, 'w')
if truncate then truncate:close() end
log('=== reverse window probe ' .. os.date('%Y-%m-%d %H:%M:%S') .. ' ===')

reaper.InsertTrackAtIndex(0, true)
local track = reaper.GetTrack(0, 0)
reaper.GetSetMediaTrackInfo_String(track, 'P_NAME', 'reverse-window', true)
reaper.SetOnlyTrackSelected(track)

reaper.SetEditCurPos(0.0, false, false)
reaper.InsertMedia(wavPath, 0)
local item = reaper.GetTrackMediaItem(track, reaper.CountTrackMediaItems(track) - 1)
reaper.SetMediaItemInfo_Value(item, 'D_LENGTH', ITEM_LENGTH)
reaper.SetMediaItemInfo_Value(item, 'B_LOOPSRC', 0)
reaper.SetMediaItemInfo_Value(item, 'D_FADEINLEN', 0)
reaper.SetMediaItemInfo_Value(item, 'D_FADEOUTLEN', 0)
reaper.SetMediaItemInfo_Value(item, 'D_FADEINLEN_AUTO', 0)
reaper.SetMediaItemInfo_Value(item, 'D_FADEOUTLEN_AUTO', 0)
reaper.SetMediaItemTakeInfo_Value(reaper.GetActiveTake(item), 'D_STARTOFFS', SOURCE_OFFSET)

-- 官方 action 41051 = "Item properties: Toggle take reverse"。
local reverseAction
do
  local section = reaper.SectionFromUniqueID(0)
  local index = 0
  while true do
    local command, name = reaper.kbd_enumerateActions(section, index)
    if command == 0 then break end
    if name == 'Item properties: Toggle take reverse' then reverseAction = command end
    index = index + 1
  end
end
if not reverseAction then finish('error', 'reverse action not found'); return end
reaper.SelectAllMediaItems(0, false)
reaper.SetMediaItemSelected(item, true)
reaper.Main_OnCommand(reverseAction, 0)

local take = reaper.GetActiveTake(item)
local startoffs = reaper.GetMediaItemTakeInfo_Value(take, 'D_STARTOFFS')
local ok, _, _, reversed = reaper.PCM_Source_GetSectionInfo(reaper.GetMediaItemTake_Source(take))
log(string.format('reversed item: startoffs=%.6f section=%s reversed=%s', startoffs, tostring(ok), tostring(reversed)))
if not (ok and reversed) then finish('error', 'reverse not verified'); return end

reaper.GetSetProjectInfo(0, 'RENDER_SETTINGS', 0, true)
reaper.GetSetProjectInfo(0, 'RENDER_BOUNDSFLAG', 0, true)
reaper.GetSetProjectInfo(0, 'RENDER_STARTPOS', 0, true)
reaper.GetSetProjectInfo(0, 'RENDER_ENDPOS', ITEM_LENGTH, true)
reaper.GetSetProjectInfo(0, 'RENDER_SRATE', RENDER_RATE, true)
reaper.GetSetProjectInfo(0, 'RENDER_CHANNELS', 1, true)
reaper.GetSetProjectInfo(0, 'RENDER_TAILFLAG', 0, true)
reaper.GetSetProjectInfo(0, 'RENDER_NORMALIZE', 0, true)
reaper.GetSetProjectInfo_String(0, 'RENDER_FILE', capturesDir, true)
reaper.GetSetProjectInfo_String(0, 'RENDER_PATTERN', 'reverse-window', true)
reaper.GetSetProjectInfo_String(0, 'RENDER_FORMAT', 'ZXZhdxgAAA==', true)

local renderCommand
do
  local section = reaper.SectionFromUniqueID(0)
  local index = 0
  while true do
    local command, name = reaper.kbd_enumerateActions(section, index)
    if command == 0 then break end
    if name and name:find('Render project, using the most recent render settings', 1, true) then
      if name:find('auto%-close') then renderCommand = command end
    end
    index = index + 1
  end
end
if not renderCommand then finish('error', 'render action not found'); return end

reaper.UpdateArrange()
local deadline = reaper.time_precise() + 2.0
local function render()
  if reaper.time_precise() < deadline then reaper.defer(render); return end
  reaper.Main_OnCommand(renderCommand, 0)
  finish('ok', 'render command dispatched')
end

reaper.defer(render)
