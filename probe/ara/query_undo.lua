--[[
  query_undo.lua —— 只读诊断：查询 undo 历史与工程路径（十六进制编码，规避控制台编码问题）

  作用：告诉用户/agent 有多少个可撤销点、以及每条 undo 的描述，
        从而判断"我插入的 probe-* 轨道"能否用 Ctrl+Z 安全回退。
        **不做任何修改。**
]]

local sep = package.config:sub(1, 1)
local scriptPath = debug.getinfo(1, 'S').source:sub(2)
local scriptDir = scriptPath:match('^(.*)' .. sep .. '[^' .. sep .. ']*$') or '.'
local logPath = scriptDir .. sep .. 'captures' .. sep .. 'undo.log'

local lines = {}
local function log(m) lines[#lines + 1] = tostring(m) end

-- 路径以 UTF-8 十六进制输出：控制台代码页会毁掉中文，十六进制不会。
local function hex(s)
  if not s then return '(nil)' end
  return (s:gsub('.', function(c) return string.format('%02X', string.byte(c)) end))
end

local _, projPath = reaper.EnumProjects(-1, '')
log('projectPathHex = ' .. hex(projPath))

-- undo 历史
local canUndo = reaper.Undo_CanUndo2 and reaper.Undo_CanUndo2(0) or nil
local canRedo = reaper.Undo_CanRedo2 and reaper.Undo_CanRedo2(0) or nil
log('undo_CanUndo2 = ' .. tostring(canUndo))
log('undo_CanRedo2 = ' .. tostring(canRedo))

-- 枚举最近的 undo 点（REAPER 无直接 API，用 Undo_DoUndo2 的返回值描述不可行）
-- 改用 CountTracks 快照 + 提示信息
log('trackCount = ' .. reaper.CountTracks(0))
log('isDirty = ' .. tostring(reaper.IsProjectDirty(0)))

local fh = io.open(logPath, 'w')
if fh then fh:write(table.concat(lines, '\n') .. '\n'); fh:close() end
