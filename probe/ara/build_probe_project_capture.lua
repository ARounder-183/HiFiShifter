--[[
  build_probe_project_capture.lua —— HiFiShifter ARA 探针：隔离实例内采集

  作用：在**独立配置目录**的 REAPER 实例里搭建工程（挂载 ARA 插件 + 摆放素材），
        并分阶段延迟等待，让 ARA 文档真正建立后再检查 dump 是否落盘。

  用法（由 probe/ara/run_capture.ps1 调用，不建议手工跑）：
    reaper.exe -cfgfile <隔离目录>\REAPER.ini -new build_probe_project_capture.lua

  特殊说明：
    - 采集阶段写日志到 captures\capture.log，便于事后归因（REAPER 控制台看不见）。
    - 本脚本是一次性探针产物。
]]

local sep = package.config:sub(1, 1)
local scriptPath = debug.getinfo(1, 'S').source:sub(2)
local scriptDir = scriptPath:match('^(.*)' .. sep .. '[^' .. sep .. ']*$') or '.'
local fixturesDir = scriptDir .. sep .. 'fixtures'
local capturesDir = scriptDir .. sep .. 'captures'
local logPath = capturesDir .. sep .. 'capture.log'

local lines = {}
local function log(msg)
  lines[#lines + 1] = tostring(msg)
  reaper.ShowConsoleMsg('[probe] ' .. tostring(msg) .. '\n')
end

local function flush()
  local f = io.open(logPath, 'w')
  if f then
    f:write(table.concat(lines, '\n') .. '\n')
    f:close()
  end
end

log('=== capture run ' .. os.date('%Y-%m-%d %H:%M:%S') .. ' ===')
log('scriptDir  = ' .. scriptDir)
log('ARA_PROBE_OUT = ' .. tostring(os.getenv('ARA_PROBE_OUT')))

local function fileExistsSize(p)
  local f = io.open(p, 'rb')
  if not f then return nil end
  local size = f:seek('end')
  f:close()
  return size
end

---------------------------------------------------------------------
-- 阶段 1：搭工程
---------------------------------------------------------------------
local function buildProject()
  local trackCount = reaper.CountTracks(0)
  log('stage1: existing tracks = ' .. trackCount)

  local function addTone(wavName, label)
    local wavPath = fixturesDir .. sep .. wavName
    if not fileExistsSize(wavPath) then
      log('  MISSING fixture: ' .. wavPath)
      return
    end

    local idx = reaper.CountTracks(0)
    reaper.InsertTrackAtIndex(idx, true)
    local track = reaper.GetTrack(0, idx)
    reaper.GetSetMediaTrackInfo_String(track, 'P_NAME', label, true)

    reaper.SetOnlyTrackSelected(track)
    reaper.InsertMedia(wavPath, 0)

    local fxIndex = reaper.TrackFX_AddByName(track, 'ARATestPlugIn', false, -1)
    log('  ' .. label .. ': media=' .. wavName .. ' TrackFX_AddByName -> ' .. tostring(fxIndex))
    if fxIndex >= 0 then
      local _, fxName = reaper.TrackFX_GetFXName(track, fxIndex, '')
      log('    fx name = ' .. tostring(fxName))
      local _, ident = reaper.TrackFX_GetNamedConfigParm(track, fxIndex, 'ident')
      log('    fx ident = ' .. tostring(ident))
    else
      log('    !! 插件未插入 —— 可能未被扫描到，名字不匹配，或当前实例看不到 D:\\VST')
    end
  end

  addTone('tone44100.wav', 'probe-44k')
  reaper.UpdateArrange()
  flush()
end

---------------------------------------------------------------------
-- 阶段 2..N：分阶段等待并检查
---------------------------------------------------------------------
local checks = 0
local maxChecks = 20   -- 每次 1 秒
local buildDone = false

local function tick()
  checks = checks + 1

  if not buildDone then
    buildProject()
    buildDone = true
    reaper.defer(tick)
    return
  end

  local out = os.getenv('ARA_PROBE_OUT')
  if out then
    local size = fileExistsSize(out)
    if size and size > 0 then
      log('stage' .. (checks + 1) .. ': DUMP WRITTEN at check ' .. checks .. ', size = ' .. tostring(size))
      log('tracks now = ' .. reaper.CountTracks(0))
      -- 再等一下，让后续 dump（编辑结束回调）覆盖写入更完整的版本
      if checks < maxChecks then
        reaper.defer(tick)
        return
      end
      flush()
      return
    end
  end

  if checks >= maxChecks then
    log('stage: GAVE UP after ' .. checks .. 's — dump never appeared')
    log('tracks now = ' .. reaper.CountTracks(0))
    for i = 0, reaper.CountTracks(0) - 1 do
      local tr = reaper.GetTrack(0, i)
      local _, nm = reaper.GetSetMediaTrackInfo_String(tr, 'P_NAME', '', false)
      log('  track[' .. i .. '] name=' .. tostring(nm) .. ' fx=' .. reaper.TrackFX_GetCount(tr))
    end
    flush()
    return
  end

  reaper.defer(tick)
end

reaper.defer(tick)
