--[[
  build_task10_probe.lua —— HiFiShifter ARA 产品模型的 REAPER 端到端采集。

  目的：在隔离实例中放入同一素材两次、插入产品插件，并等待插件日志出现
  `ara: ... playbackRegions=2 clips=2`。第二个 item 留在工程里，供人工拖动验证 A2。
  本脚本是一次性探针产物，不属于产品代码。
]]

local sep = package.config:sub(1, 1)
local scriptPath = debug.getinfo(1, 'S').source:sub(2)
local scriptDir = scriptPath:match('^(.*)' .. sep .. '[^' .. sep .. ']*$') or '.'
local fixturesDir = scriptDir .. sep .. 'fixtures'
local capturesDir = scriptDir .. sep .. 'captures'
local logPath = capturesDir .. sep .. 'task10-plugin.log'
local pluginLogPath = os.getenv('HIFISHIFTER_ARA_LOG')
local pluginName = 'HiFiShifter'
local wavPath = fixturesDir .. sep .. 'tone44100.wav'

local function log(message)
  local line = tostring(message)
  reaper.ShowConsoleMsg('[task10] ' .. line .. '\n')
  local file = io.open(logPath, 'a')
  if file then
    file:write(line .. '\n')
    file:close()
  end
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

log('=== task10 run ' .. os.date('%Y-%m-%d %H:%M:%S') .. ' ===')
log('scriptDir = ' .. scriptDir)
log('plugin log = ' .. tostring(pluginLogPath))

local stage = 0
local checks = 0
local deadline = reaper.time_precise() + 20.0
local track

local function tick()
  checks = checks + 1

  if stage == 0 then
    local fixture = io.open(wavPath, 'rb')
    if not fixture then
      log('MISSING fixture: ' .. wavPath)
      return
    end
    fixture:close()
    local index = reaper.CountTracks(0)
    reaper.InsertTrackAtIndex(index, true)
    track = reaper.GetTrack(0, index)
    reaper.GetSetMediaTrackInfo_String(track, 'P_NAME', 'task10-ara', true)
    reaper.SetOnlyTrackSelected(track)
    log('stage0: track created')
    stage = 1
    reaper.defer(tick)
    return
  end

  if stage == 1 then
    for _, position in ipairs({ 0.0, 3.0 }) do
      reaper.SetEditCurPos(position, false, false)
      reaper.InsertMedia(wavPath, 0)
    end
    reaper.UpdateArrange()
    log('stage1: items=' .. reaper.CountTrackMediaItems(track))
    stage = 2
    reaper.defer(tick)
    return
  end

  if stage == 2 then
    local fxIndex = reaper.TrackFX_AddByName(track, pluginName, false, -1)
    log('stage2: TrackFX_AddByName("' .. pluginName .. '") -> ' .. tostring(fxIndex))
    if fxIndex >= 0 then
      local _, fxName = reaper.TrackFX_GetFXName(track, fxIndex, '')
      local _, ident = reaper.TrackFX_GetNamedConfigParm(track, fxIndex, 'ident')
      log('  fx name=' .. tostring(fxName) .. ' ident=' .. tostring(ident))
    end
    stage = 3
    reaper.defer(tick)
    return
  end

  local pluginLog = readFile(pluginLogPath)
  if stage == 3 and pluginLog:find('ara: sources=1 modifications=1 regionSequences=1 playbackRegions=2 clips=2', 1, true) then
    log('stage3: A1 summary observed at check ' .. checks)
    log('stage3: items=' .. reaper.CountTrackMediaItems(track)
      .. ' fx=' .. reaper.TrackFX_GetCount(track))
    if os.getenv('HIFISHIFTER_ARA_MANUAL') == '1' then
      log('stage3: ready for UI move/split observation')
      return
    end
    stage = 4
    reaper.defer(tick)
    return
  end

  if stage == 4 then
    local second = reaper.GetTrackMediaItem(track, 1)
    if second then
      reaper.Undo_BeginBlock()
      reaper.SetMediaItemPosition(second, 5.0, true)
      reaper.UpdateItemInProject(second)
      reaper.Undo_EndBlock('Task10: move item to 5 seconds', -1)
      log('stage4: moved second item to start_sec=5.0')
      stage = 5
      reaper.defer(tick)
      return
    end
    log('stage4: second item missing')
    return
  end

  if stage == 5 and pluginLog:find('ara: clipStartsSec=[0.000000,5.000000]', 1, true) then
    log('stage5: A2 clip start update observed at check ' .. checks)
    return
  end

  if reaper.time_precise() >= deadline then
    log('stage3: GAVE UP after ' .. checks .. ' checks')
    log('stage3: items=' .. reaper.CountTrackMediaItems(track)
      .. ' fx=' .. reaper.TrackFX_GetCount(track))
    return
  end
  reaper.defer(tick)
end

reaper.defer(tick)
