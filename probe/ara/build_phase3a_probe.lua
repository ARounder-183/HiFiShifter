--[[ 一次性 Phase 3a 宿主输出探针：不对称 PCM、正常/裁切/真倒放与 float/24bit WAV 导出。 ]]
local sep = package.config:sub(1,1)
local dir = debug.getinfo(1,'S').source:sub(2):match('^(.*)' .. sep .. '[^' .. sep .. ']*$')
local captures = dir .. sep .. 'captures'
local fixture = dir .. sep .. 'fixtures' .. sep .. 'phase3a-asymmetric.wav'
local logfile = captures .. sep .. 'phase3a-script.log'
local function log(line)
  local f=assert(io.open(logfile,'a')); f:write(line .. '\n'); f:close()
end
local f=assert(io.open(logfile,'w')); f:close()
local rate=44100
local samples={}
for n=0,rate*2-1 do
  local envelope = n < rate and 0.2 or 0.07
  local value = envelope * math.sin(2*math.pi*(n<rate and 173 or 311)*n/rate)
  if n>=7000 and n<7100 then value=0.45 end
  samples[#samples+1]=string.pack('<f',value)
end
local pcm=table.concat(samples)
f=assert(io.open(fixture,'wb'))
f:write('RIFF',string.pack('<I4',36+#pcm),'WAVEfmt ',string.pack('<I4I2I2I4I4I2I2',16,3,1,rate,rate*4,4,32),'data',string.pack('<I4',#pcm),pcm); f:close()
reaper.InsertTrackAtIndex(0,true)
local track=reaper.GetTrack(0,0)
reaper.GetSetMediaTrackInfo_String(track,'P_NAME','phase3a-output',true)
reaper.SetOnlyTrackSelected(track)
local function item(position,offset,length)
  reaper.SetEditCurPos(position,false,false); reaper.InsertMedia(fixture,0)
  local object=reaper.GetTrackMediaItem(track,reaper.CountTrackMediaItems(track)-1)
  reaper.SetMediaItemInfo_Value(object,'D_LENGTH',length)
  reaper.SetMediaItemInfo_Value(object,'B_LOOPSRC',0)
  reaper.SetMediaItemInfo_Value(object,'D_FADEINLEN',0)
  reaper.SetMediaItemInfo_Value(object,'D_FADEOUTLEN',0)
  reaper.SetMediaItemInfo_Value(object,'D_FADEINLEN_AUTO',0)
  reaper.SetMediaItemInfo_Value(object,'D_FADEOUTLEN_AUTO',0)
  reaper.SetMediaItemTakeInfo_Value(reaper.GetActiveTake(object),'D_STARTOFFS',offset)
  return object
end
item(0,0,2)
item(3,0.25,1)
local reverse=item(6,0,2)
reaper.SelectAllMediaItems(0,false); reaper.SetMediaItemSelected(reverse,true)
reaper.Main_OnCommand(41051,0)
local ok,offset,length,reversed=reaper.PCM_Source_GetSectionInfo(reaper.GetMediaItemTake_Source(reaper.GetActiveTake(reverse)))
log('reverse verified: section='..tostring(ok)..' reversed='..tostring(reversed))
assert(ok and reversed)
local fx=reaper.TrackFX_AddByName(track,'HiFiShifter',false,-1)
log('fx='..fx); assert(fx>=0)
reaper.GetSetProjectInfo(0,'RENDER_SETTINGS',0,true)
reaper.GetSetProjectInfo(0,'RENDER_BOUNDSFLAG',0,true)
reaper.GetSetProjectInfo(0,'RENDER_STARTPOS',0,true)
reaper.GetSetProjectInfo(0,'RENDER_ENDPOS',8,true)
reaper.GetSetProjectInfo(0,'RENDER_SRATE',rate,true)
reaper.GetSetProjectInfo(0,'RENDER_CHANNELS',2,true)
reaper.GetSetProjectInfo(0,'RENDER_TAILFLAG',0,true)
reaper.GetSetProjectInfo(0,'RENDER_NORMALIZE',0,true)
reaper.GetSetProjectInfo_String(0,'RENDER_FILE',captures,true)
reaper.GetSetProjectInfo_String(0,'RENDER_PATTERN','phase3a-output',true)
reaper.GetSetProjectInfo_String(0,'RENDER_FORMAT','ZXZhdxgAAA==',true)
local section=reaper.SectionFromUniqueID(0)
local command
for index=0,10000 do
  local id,name=reaper.kbd_enumerateActions(section,index)
  if id==0 then break end
  if name and name:find('Render project, using the most recent render settings',1,true) then
    log('render candidate '..id..' '..name)
    if name:find('auto%-close') then command=id end
  end
end
assert(command,'auto-close render action missing')
reaper.UpdateArrange()
local deadline=reaper.time_precise()+2
local function render()
  if reaper.time_precise()<deadline then reaper.defer(render); return end
  log('render action='..command)
  reaper.Main_OnCommand(command,0)
  log('render returned')
end
reaper.defer(render)
