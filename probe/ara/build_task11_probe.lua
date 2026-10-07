--[[
  build_task11_probe.lua —— ARA 能力声明后的拉伸/倒放宿主级采集。

  先在插件插入前构造三个 item：正常、2 倍 playrate、倒放。插件日志记录
  playback region 的两个时间坐标与 transformation flags。本脚本是一次性探针产物。
]]

local sep = package.config:sub(1, 1)
local scriptPath = debug.getinfo(1, 'S').source:sub(2)
local scriptDir = scriptPath:match('^(.*)' .. sep .. '[^' .. sep .. ']*$') or '.'
local fixturesDir = scriptDir .. sep .. 'fixtures'
local capturesDir = scriptDir .. sep .. 'captures'
local logPath = capturesDir .. sep .. 'task11-plugin.log'
local pluginLogPath = os.getenv('HIFISHIFTER_ARA_LOG')
local wavPath = fixturesDir .. sep .. 'tone44100.wav'

local function log(message)
  local line = tostring(message)
  reaper.ShowConsoleMsg('[task11] ' .. line .. '\n')
  local file = io.open(logPath, 'a')
  if file then file:write(line .. '\n'); file:close() end
end

local function exists(path)
  local file = io.open(path, 'rb')
  if file then file:close(); return true end
  return false
end

local function readFile(path)
  if not path or path == '' then return '' end
  local file = io.open(path, 'r')
  if not file then return '' end
  local content = file:read('a') or ''
  file:close()
  return content
end

local truncate = io.open(logPath, 'w')
if truncate then truncate:close() end
if pluginLogPath then
  local pluginLog = io.open(pluginLogPath, 'w')
  if pluginLog then pluginLog:close() end
end

log('=== task11 run ' .. os.date('%Y-%m-%d %H:%M:%S') .. ' ===')
if not exists(wavPath) then log('MISSING fixture: ' .. wavPath); return end

local trackIndex = reaper.CountTracks(0)
reaper.InsertTrackAtIndex(trackIndex, true)
local track = reaper.GetTrack(0, trackIndex)
reaper.GetSetMediaTrackInfo_String(track, 'P_NAME', 'task11-transformations', true)
reaper.SetOnlyTrackSelected(track)

local function insertItem(position)
  reaper.SetEditCurPos(position, false, false)
  reaper.InsertMedia(wavPath, 0)
  return reaper.GetTrackMediaItem(track, reaper.CountTrackMediaItems(track) - 1)
end

local normal = insertItem(0.0)
local stretch = insertItem(3.0)
local reverse = insertItem(6.0)

if stretch then
  local take = reaper.GetMediaItemTake(stretch, 0)
  if take then
    reaper.SetMediaItemTakeInfo_Value(take, 'D_PLAYRATE', 2.0)
    reaper.SetMediaItemTakeInfo_Value(take, 'B_LOOPSRC', 0)
    reaper.SetMediaItemInfo_Value(stretch, 'D_LENGTH', 1.0)
    log('stretch: playrate=2.0 length=1.0')
  end
end

if reverse then
  local section = reaper.SectionFromUniqueID(0)
  local actionIndex = 0
  local reverseAction
  while true do
    local command, name = reaper.kbd_enumerateActions(section, actionIndex)
    if command == 0 then break end
    if name and name:lower():find('reverse') then
      log('reverse action: ' .. command .. ' ' .. name)
      if name == 'Item properties: Toggle take reverse' then reverseAction = command end
    end
    actionIndex = actionIndex + 1
  end
  assert(reverseAction, 'official take reverse action not found')
  reaper.SelectAllMediaItems(0, false)
  reaper.SetMediaItemSelected(reverse, true)
  reaper.Main_OnCommand(reverseAction, 0)
  local source = reaper.GetMediaItemTake_Source(reaper.GetActiveTake(reverse))
  local ok, offset, length, reversed = reaper.PCM_Source_GetSectionInfo(source)
  log('reverse verified: section=' .. tostring(ok) .. ' reversed=' .. tostring(reversed)
    .. ' offset=' .. tostring(offset) .. ' length=' .. tostring(length))
  assert(ok and reversed, 'take source must actually be reversed')
end

reaper.UpdateArrange()
local fxIndex = reaper.TrackFX_AddByName(track, 'HiFiShifter', false, -1)
log('TrackFX_AddByName -> ' .. tostring(fxIndex))

local checks = 0
local deadline = reaper.time_precise() + 20.0
local function tick()
  checks = checks + 1
  local pluginLog = readFile(pluginLogPath)
  if pluginLog:find('playback_region #1', 1, true)
      and pluginLog:find('playback_region #2', 1, true) then
    log('transformations observed at check ' .. checks)
    return
  end
  if reaper.time_precise() >= deadline then
    log('GAVE UP after ' .. checks .. ' checks')
    return
  end
  reaper.defer(tick)
end

reaper.defer(tick)
