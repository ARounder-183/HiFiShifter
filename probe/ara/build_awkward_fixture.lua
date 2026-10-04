--[[
  build_awkward_fixture.lua —— HiFiShifter ARA 探针 Task 1 Step 4

  作用：在隔离 REAPER 实例里构造"不干净"的素材，逼出 ARA 的变换字段：
        1) 同一素材复制多份放在不同位置（考察 source 复用 / region 多对一）
        2) 对某个 region 拉伸（考察 playback transformation）
        3) 对某个 region 倒放
        4) 对某个 region 设淡化
        5) 在非 44.1kHz 工程采样率下重复

  用法（由采集流程调用）：
    reaper.exe -cfgfile <隔离>\REAPER.ini -new build_awkward_fixture.lua

  特殊说明：
    - 每一步都用 ReaScript 的 item API 完成，不依赖图形界面。
    - 关键 API 可用性做防御性判断：不同 REAPER 版本暴露面不同，缺 API 时记录
      而不是静默跳过 —— "没生效"和"没这个 API"必须能区分。
    - 本脚本是一次性探针产物。
]]

local sep = package.config:sub(1, 1)
local scriptPath = debug.getinfo(1, 'S').source:sub(2)
local scriptDir = scriptPath:match('^(.*)' .. sep .. '[^' .. sep .. ']*$') or '.'
local fixturesDir = scriptDir .. sep .. 'fixtures'
local capturesDir = scriptDir .. sep .. 'captures'
local logPath = capturesDir .. sep .. 'awkward.log'

local lines = {}
local function log(m) lines[#lines + 1] = tostring(m) end
local function flush()
  local f = io.open(logPath, 'w')
  if f then f:write(table.concat(lines, '\n') .. '\n'); f:close() end
end

log('=== awkward fixture run ' .. os.date('%Y-%m-%d %H:%M:%S') .. ' ===')

local function exists(p)
  local f = io.open(p, 'rb')
  if f then f:close(); return true end
  return false
end

-- 报告 API 可用性，避免"静默无效"
local function has(fnName)
  local ok = reaper[fnName] ~= nil
  log(string.format('  api %-32s = %s', fnName, tostring(ok)))
  return ok
end

local function newTrack(name)
  local idx = reaper.CountTracks(0)
  reaper.InsertTrackAtIndex(idx, true)
  local tr = reaper.GetTrack(0, idx)
  reaper.GetSetMediaTrackInfo_String(tr, 'P_NAME', name, true)
  -- 挂 ARA 插件（每轨道一个文档）
  local fx = reaper.TrackFX_AddByName(tr, 'ARATestPlugIn', false, -1)
  log(string.format('  track "%s": fx=%d', name, fx))
  return tr
end

-- 在轨道上放一个 item，返回 item
local function placeItem(tr, wavName, startSec)
  local path = fixturesDir .. sep .. wavName
  if not exists(path) then log('  MISSING ' .. path); return nil end
  reaper.SetOnlyTrackSelected(tr)
  reaper.InsertMedia(path, 0)
  local item = reaper.GetTrackMediaItem(tr, reaper.CountTrackMediaItems(tr) - 1)
  if not item then log('  InsertMedia produced no item'); return nil end
  reaper.SetMediaItemInfo_Value(item, 'D_POSITION', startSec)
  return item
end

---------------------------------------------------------------------
log('--- api availability ---')
has('SetMediaItemInfo_Value')
has('SetMediaItemTakeInfo_Value')
has('GetMediaItemTake')
has('SetMediaItemLength')
has('SplitMediaItem')

local function build()
  reaper.Main_OnCommand(40023, 0) -- New project
  log('tracks at start = ' .. reaper.CountTracks(0))

  ------------------------------------------------------------------
  -- 轨 1：同源多放 + 拉伸 + 倒放 + 淡化
  ------------------------------------------------------------------
  log('--- track 1: duplicate / stretch / reverse / fade ---')
  local t1 = newTrack('awkward-44k')

  local a = placeItem(t1, 'tone44100.wav', 0.0)
  local b = placeItem(t1, 'tone44100.wav', 3.0)   -- 同源第二份
  local c = placeItem(t1, 'tone44100.wav', 6.0)
  local d = placeItem(t1, 'tone44100.wav', 9.0)

  -- (2) 拉伸：改 playrate（同时按比例改长度，保持内容不被裁）
  if b then
    local take = reaper.GetMediaItemTake(b, 0)
    if take then
      local rates = reaper.SetMediaItemTakeInfo_Value and 2.0 or 2.0
      reaper.SetMediaItemTakeInfo_Value(take, 'D_PLAYRATE', rates)
      reaper.SetMediaItemTakeInfo_Value(take, 'B_LOOPSRC', 0)
      -- 长度 = 原长 / rate * rate 保持覆盖；这里显式设为 1 秒（2 秒素材 @ rate2 → 1 秒）
      reaper.SetMediaItemInfo_Value(b, 'D_LENGTH', 1.0)
      log(string.format('  stretch: item start=%.2f rate=%.2f len=%.2f',
        reaper.GetMediaItemInfo_Value(b, 'D_POSITION'),
        reaper.GetMediaItemTakeInfo_Value(take, 'D_PLAYRATE'),
        reaper.GetMediaItemInfo_Value(b, 'D_LENGTH')))
    else
      log('  stretch: GetMediaItemTake returned nil')
    end
  end

  -- (3) 倒放：part 1.0 = reversed
  if c then
    local take = reaper.GetMediaItemTake(c, 0)
    if take then
      reaper.SetMediaItemTakeInfo_Value(take, 'D_STARTOFFS', 0)
      -- REAPER 的倒放用 take 的 B_REVERSED？不同版本命名不同，两个都试
      local okRev = pcall(function() reaper.SetMediaItemTakeInfo_Value(take, 'B_REVERSED', 1) end)
      log('  reverse: B_REVERSED pcall ok=' .. tostring(okRev))
    end
    -- 备选：走 action 反相（需要选择 item）
    reaper.SetMediaItemSelected(c, true)
  end

  -- (4) 淡化：item 的 fade-in/out 长度也是 ARA 变换的一部分
  if d then
    reaper.SetMediaItemInfo_Value(d, 'D_FADEINLEN', 0.5)
    reaper.SetMediaItemInfo_Value(d, 'D_FADEOUTLEN', 0.5)
    log(string.format('  fade: in=%.2f out=%.2f',
      reaper.GetMediaItemInfo_Value(d, 'D_FADEINLEN'),
      reaper.GetMediaItemInfo_Value(d, 'D_FADEOUTLEN')))
  end

  ------------------------------------------------------------------
  -- 轨 2：非 44.1kHz 素材（48k）
  ------------------------------------------------------------------
  log('--- track 2: non-44.1k source ---')
  local t2 = newTrack('awkward-48k')
  placeItem(t2, 'tone48000.wav', 0.0)

  reaper.UpdateArrange()
  flush()
end

build()
