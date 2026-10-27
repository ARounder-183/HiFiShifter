"""Real bank conversion, retake inputs and synthesis; no project modifications."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

sys.stdout.reconfigure(encoding='utf-8')
source, packages, voice, out = map(Path, sys.argv[1:5])
out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(packages.resolve()))
spec = importlib.util.spec_from_file_location('bridge', source.resolve())
b = importlib.util.module_from_spec(spec); sys.modules['bridge'] = b; spec.loader.exec_module(b)
np = b.np
bank = b.Bank(voice)
checks = []
def check(name, condition):
    assert condition, name
    checks.append(name); print('PASS', name, flush=True)
def notes(lyrics):
    return [dict(id=str(i), lyric=s, midi=60+i%3, start=.5+i*.7, duration=.7) for i,s in enumerate(lyrics)]
def request(operation, values, **kw):
    return dict(operation=operation, voicebank=str(voice), language='zh', notes=values,
                duration=max(n['start']+n['duration'] for n in values)+.3, **kw)
converted, info = bank.resolver.prepare(notes(['银','行','音','乐','en/hello','+2']))
check('Chinese phrase polyphones across notes', converted[1]['resolved_phones']==['zh/h','zh/ang']
      and converted[3]['resolved_phones'] != bank.phonemes('le'))
check('English multi-syllable assignment', converted[4]['resolved_phones']==['en/hh','en/ax']
      and converted[5]['resolved_phones']==['en/l','en/ow'])
separate=notes(['银','行'])
for i,n in enumerate(separate): n['context']=str(i)
prepared,_=bank.resolver.prepare(separate)
check('clip boundaries use same context in preview and playback',prepared[1]['resolved_phones']==bank.phonemes('行')
      and len(b.groups(prepared))==2)
for text in ('nǐ','ni3','ni[n i]','[zh/n zh/i]'):
    check('pinyin/hint '+text, bank.phonemes(text)==bank.phonemes('ni'))
v = notes(['张']); v[0]['pronunciation']='[zh/zh zh/a en/ng]'
prepared, info = bank.resolver.prepare(v)
check('independent override keeps lyric and coda token', prepared[0]['lyric']=='张'
      and prepared[0]['resolved_phones']==['zh/zh','zh/a','en/ng'])
_, custom = bank.resolver.prepare(notes(['行','en/read']), '行 = hang\nen/read = [r eh d]')
check('track user dictionary precedes automatic reading', custom[0]['phones']==['zh/h','zh/ang']
      and custom[1]['phones']==['en/r','en/eh','en/d'])
for values,dictionary in [(notes(['+2']), ''), (notes(['ni','+2']), ''),
    (notes(['ni']), 'x = y\ny = x'), (notes(['[zh/nonexistent]']), ''),
    (notes(['+']), ''), (notes(['RR','+']), '')]:
    try: bank.resolver.prepare(values,dictionary)
    except ValueError: pass
    else: raise AssertionError('invalid pronunciation accepted')
check('bad hints, cyclic dictionary and syllable markers rejected', True)
phones,source_name = bank.resolver.resolve('en/activationist')
check('offline English OOV model',len(phones)>4 and '预测' in source_name)
probe=notes(['ni','hao','zhang','bang'])
probe[-1]['start']+=2 # independent phrase must be skipped by local pitch
for i,n in enumerate(probe): n['pitch']=[[0,60+i*.4],[.3,60+i*.4+.2],[.7,60+i*.4]]
pitch_model=bank.model('dspitch'); real_run=pitch_model.run; seen=[]
def capture(name,values,frames=None):
    if name=='pitch': seen.append({k: np.asarray(values[k]).copy() for k in ('pitch','retake')})
    return real_run(name,values,frames)
pitch_model.run=capture
partial=b.execute(request('pitch',probe,retake_ids=['1']),out/'partial.json',bank)
pitch_model.run=real_run
check('only selected phrase and selected output',len(seen)==1 and partial['generated_phrases']==1
      and [c['id'] for c in partial['curves']]==['1'])
plan=b.timing.apply(bank,probe[:3],bank.plan(probe[:3]))[0]
time=plan['origin']+np.arange(plan['total'])*bank.main.dt
expected=(time>=probe[1]['start']) & (time<probe[1]['start']+.7)
check('real ONNX receives partial retake mask',np.array_equal(seen[0]['retake'][0],expected)
      and seen[0]['retake'].any() and not seen[0]['retake'].all())
locked=(time>=.5) & (time<1.2)
expected_pitch=np.interp(time[locked]-.5,[0,.3,.7],[60,60.2,60])
check('unedited pitch supplied as locked context', np.max(np.abs(seen[0]['pitch'][0,locked]-expected_pitch))<1e-5)
saved=bank.pitch
bank.pitch=lambda *args,**kwargs: (_ for _ in ()).throw(AssertionError('no selection must not infer'))
try:
    b.execute(request('pitch',probe,retake_ids=[]),out/'empty.json',bank)
except ValueError as e:
    check('empty retake does not become whole track', '所选音符无效' in str(e))
else:
    raise AssertionError('empty retake accepted')
bank.pitch=saved
render_notes=notes(['银','行','en/hello','+2','张'])
render_notes[-1]['pronunciation']='[zh/zh zh/a en/ng]'
pitch=b.execute(request('pitch',render_notes),out/'mixed-pitch.json',bank)
for item in pitch['curves']: render_notes[int(item['id'])]['pitch']=item['points']
render=b.execute(request('render',render_notes),out/'mixed-render.json',bank)
import soundfile as sf
wav,sr=sf.read(out/'mixed-render.wav')
check('mixed Chinese English, syllable split and coda synthesize',render['ok'] and np.isfinite(wav).all() and np.max(np.abs(wav))>.01)
rows=[r for r in render['phonemes'] if r['id']=='4']
check('coda appears as third editable phone',[r['token'] for r in rows]==['zh/zh','zh/a','en/ng'] and all(r['editable'] for r in rows))
(out/'report.json').write_text(json.dumps(dict(checks=checks,phonemes=render['phonemes'],
    pronunciations=render['pronunciations'], partial=partial,peak=float(np.max(np.abs(wav))),sample_rate=sr),ensure_ascii=False,indent=2),encoding='utf-8')
