--[[
  build_task2_probe.lua —— HiFiShifter ARA 探针：Task 2 Step 3 加载验证

  作用：在**独立配置目录**的 REAPER 实例里，把 Rust 探针插件（HiFiShifter ARA Probe）
        挂到一条轨道上，并摆放两处同源素材，让 ARA 文档建立。
        判据由插件自己写：它把看到的 audioSource / playbackRegion 数量落到
        HIFISHIFTER_ARA_PROBE_LOG 指向的文件；本脚本只负责摆场景与记录 REAPER 侧结果。

  用法（隔离实例，先确保没有 REAPER 在运行）：
    reaper.exe -cfgfile <隔离目录>\REAPER.ini -new build_task2_probe.lua

  特殊说明：本脚本是一次性探针产物，不进入产品代码。
]]

local sep = package.config:sub(1, 1)
local scriptPath = debug.getinfo(1, 'S').source:sub(2)
local scriptDir = scriptPath:match('^(.*)' .. sep .. '[^' .. sep .. ']*$') or '.'
local fixturesDir = scriptDir .. sep .. 'fixtures'
local capturesDir = scriptDir .. sep .. 'captures'
local logPath = capturesDir .. sep .. 'task2-capture.log'

local PLUGIN_NAME = 'HiFiShifter ARA Probe'

local function log(msg)
  reaper.ShowConsoleMsg('[task2] ' .. tostring(msg) .. '\n')
  local f = io.open(logPath, 'a')
  if f then
    f:write(tostring(msg) .. '\n')
    f:close()
  end
end

local function flush()
  -- 逐行追加写，flush 变成空实现：崩溃/挂起时也能看到走到哪一步。
end

-- 每次运行清空日志。
local truncate = io.open(logPath, 'w')
if truncate then truncate:close() end

log('=== task2 run ' .. os.date('%Y-%m-%d %H:%M:%S') .. ' ===')
log('scriptDir = ' .. scriptDir)
log('plugin log env = ' .. tostring(os.getenv('HIFISHIFTER_ARA_PROBE_LOG')))

-- 分阶段执行：每一步都立刻落盘，这样即使 REAPER 在插入插件时挂起或崩溃，
-- 也能从 task2-capture.log 看出停在哪个阶段。
local wavPath = fixturesDir .. sep .. 'tone44100.wav'
local track
local stage = 0
local checks = 0
local maxChecks = 45

local function tick()
  checks = checks + 1

  if stage == 0 then
    if not io.open(wavPath, 'rb') then
      log('MISSING fixture: ' .. wavPath)
      return
    end
    local idx = reaper.CountTracks(0)
    reaper.InsertTrackAtIndex(idx, true)
    track = reaper.GetTrack(0, idx)
    reaper.GetSetMediaTrackInfo_String(track, 'P_NAME', 'task2-ara', true)
    reaper.SetOnlyTrackSelected(track)
    log('stage0: track created')
    stage = 1
    reaper.defer(tick)
    return
  end

  if stage == 1 then
    -- 两处同源素材：ARA 应给出 1 个 audioSource、2 个 playbackRegion。
    for _, pos in ipairs({ 0.0, 3.0 }) do
      reaper.SetEditCurPos(pos, false, false)
      reaper.InsertMedia(wavPath, 0)
    end
    reaper.UpdateArrange()
    log('stage1: items = ' .. reaper.CountTrackMediaItems(track))
    stage = 2
    reaper.defer(tick)
    return
  end

  if stage == 2 then
    log('stage2: calling TrackFX_AddByName("' .. PLUGIN_NAME .. '")')
    local fxIndex = reaper.TrackFX_AddByName(track, PLUGIN_NAME, false, -1)
    log('stage2: TrackFX_AddByName -> ' .. tostring(fxIndex))
    if fxIndex >= 0 then
      local _, fxName = reaper.TrackFX_GetFXName(track, fxIndex, '')
      log('  fx name = ' .. tostring(fxName))
      local _, ident = reaper.TrackFX_GetNamedConfigParm(track, fxIndex, 'ident')
      log('  fx ident = ' .. tostring(ident))
    else
      log('  !! 插件未插入')
    end
    stage = 3
    reaper.defer(tick)
    return
  end

  -- stage 3：等待插件日志出现 ARA 对象
  local envPath = os.getenv('HIFISHIFTER_ARA_PROBE_LOG')
  if envPath then
    local f = io.open(envPath, 'r')
    if f then
      local content = f:read('a')
      f:close()
      if content and content:find('playback_region #1') then
        log('stage3: plugin log shows playback_region at check ' .. checks)
        log('stage3: track items=' .. reaper.CountTrackMediaItems(track)
          .. ' fx=' .. reaper.TrackFX_GetCount(track))
        return
      end
    end
  end

  if checks >= maxChecks then
    log('stage3: GAVE UP after ' .. checks
      .. ' checks — plugin log never showed a playback_region')
    log('stage3: track items=' .. reaper.CountTrackMediaItems(track)
      .. ' fx=' .. reaper.TrackFX_GetCount(track))
    return
  end

  reaper.defer(tick)
end

reaper.defer(tick)
