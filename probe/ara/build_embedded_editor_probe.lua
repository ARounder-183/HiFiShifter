--[[ 一次性内嵌原GUI验收；独立REAPER配置和工程，不向用户实例发送脚本。 ]]
local sep=package.config:sub(1,1)
local root=debug.getinfo(1,'S').source:sub(2):match('^(.*)'..sep..'[^'..sep..']*$')
local repo=root..sep..'..'..sep..'..'
local scratch=os.getenv('HIFISHIFTER_ARA_PROBE_DIR') or (repo..sep..'.build-tmp'..sep..'embedded-probe')
local captures=root..sep..'captures'
local fixture=root..sep..'fixtures'..sep..'embedded-editor-voice.wav'
local logfile=captures..sep..'embedded-editor-script.log'
local function log(line) local f=assert(io.open(logfile,'a')); f:write(line..'\n'); f:close() end
local _,project=reaper.EnumProjects(-1,'')
local rate=44100
local function write_fixture()
  local samples={}
  for n=0,rate*2-1 do
    local value=0
    for harmonic=1,16 do value=value+math.sin(2*math.pi*220*harmonic*n/rate)*(0.16/harmonic) end
    samples[#samples+1]=string.pack('<f',value)
  end
  local pcm=table.concat(samples)
  local f=assert(io.open(fixture,'wb'))
  f:write('RIFF',string.pack('<I4',36+#pcm),'WAVEfmt ',string.pack('<I4I2I2I4I4I2I2',16,3,1,rate,rate*4,4,32),'data',string.pack('<I4',#pcm),pcm); f:close()
end
if project=='' then
  write_fixture()
  reaper.InsertTrackAtIndex(0,true)
  local track=reaper.GetTrack(0,0)
  reaper.GetSetMediaTrackInfo_String(track,'P_NAME','HiFiShifter embedded acceptance',true)
  reaper.SetOnlyTrackSelected(track)
  local function item(position,offset,length)
    reaper.SetEditCurPos(position,false,false); reaper.InsertMedia(fixture,0)
    local object=reaper.GetTrackMediaItem(track,reaper.CountTrackMediaItems(track)-1)
    for key,value in pairs({D_LENGTH=length,B_LOOPSRC=0,D_FADEINLEN=0,D_FADEOUTLEN=0,D_FADEINLEN_AUTO=0,D_FADEOUTLEN_AUTO=0}) do reaper.SetMediaItemInfo_Value(object,key,value) end
    reaper.SetMediaItemTakeInfo_Value(reaper.GetActiveTake(object),'D_STARTOFFS',offset)
  end
  item(0,0,2); item(3,0.25,1)
  local fx=reaper.TrackFX_AddByName(track,'HiFiShifter',false,-1)
  log('fx='..fx); assert(fx>=0)
end
reaper.GetSetProjectInfo(0,'RENDER_SETTINGS',0,true)
reaper.GetSetProjectInfo(0,'RENDER_BOUNDSFLAG',0,true)
reaper.GetSetProjectInfo(0,'RENDER_STARTPOS',0,true)
reaper.GetSetProjectInfo(0,'RENDER_ENDPOS',5,true)
reaper.GetSetProjectInfo(0,'RENDER_SRATE',rate,true)
reaper.GetSetProjectInfo(0,'RENDER_CHANNELS',2,true)
reaper.GetSetProjectInfo(0,'RENDER_TAILFLAG',0,true)
reaper.GetSetProjectInfo(0,'RENDER_NORMALIZE',0,true)
reaper.GetSetProjectInfo_String(0,'RENDER_FILE',captures,true)
reaper.GetSetProjectInfo_String(0,'RENDER_FORMAT','ZXZhdxgAAA==',true)
local command
for index=0,10000 do
  local id,name=reaper.kbd_enumerateActions(reaper.SectionFromUniqueID(0),index)
  if id==0 then break end
  if name and name:find('Render project, using the most recent render settings',1,true) and name:find('auto%-close') then command=id end
end
assert(command)
reaper.UpdateArrange()
local track=reaper.GetTrack(0,0)
local fx=track and reaper.TrackFX_GetByName(track,'HiFiShifter',false) or -1
assert(fx>=0)
local show_at=reaper.time_precise()+3
local shown=false
local last=''
local deadline=reaper.time_precise()+2
local existing=io.open(captures..sep..'embedded-editor-baseline.wav','rb')
local baseline=project=='' and existing==nil
if existing then existing:close() end
local function render(label)
  reaper.GetSetProjectInfo_String(0,'RENDER_PATTERN','embedded-editor-'..label,true)
  reaper.Main_OnCommand(command,0)
  log('render returned '..label)
end
local function loop()
  if not shown and reaper.time_precise()>show_at then shown=true; reaper.TrackFX_Show(track,fx,3); log('native FX shown') end
  if baseline and reaper.time_precise()>deadline then baseline=false; render('baseline') end
  local f=io.open(scratch..sep..'command.txt','r')
  local request=f and f:read('*l') or ''; if f then f:close() end
  if request~='' and request~=last then
    last=request
    if request=='save' then
      reaper.Main_SaveProjectEx(0,scratch..sep..'embedded-editor.RPP',0)
      log('project saved')
    elseif request=='show' then reaper.TrackFX_Show(track,fx,3)
    elseif request=='close-view' then reaper.TrackFX_Show(track,fx,2)
    elseif request=='baseline-fresh' or request=='edited' or request=='reopened' or request=='source-changed' then render(request)
    else log('unknown command '..request) end
  end
  reaper.defer(loop)
end
reaper.defer(loop)
