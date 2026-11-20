from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
import subprocess, tempfile, os, re, json
from pathlib import Path

app=FastAPI(title='ZSHORTS URL Analysis API', version='0.1.0')
# Static GitHub Pages site can call this API. Restrict origins before production use.
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['POST','GET','OPTIONS'], allow_headers=['*'])
class AnalyzeRequest(BaseModel):
    url: HttpUrl
    clip_count: int=5
    clip_duration: int=45


@app.get('/health')
def health(): return {'status':'ok'}

# Small transparent transcript scoring heuristic. This is not a generative AI model.
HOOKS={"wait","what","wow","no way","unbelievable","actually","secret","mistake","crazy","insane","never","why","how","but","however","finally","imagine","look","listen","guess","truth","realize","surprise","caught","fail","win"}

def parse_vtt(text):
    entries=[]
    blocks=re.split(r'\n\s*\n', text.strip())
    for b in blocks:
        lines=b.strip().splitlines()
        time_line=next((x for x in lines if '-->' in x),None)
        if not time_line: continue
        start=time_line.split('-->')[0].strip()
        m=re.match(r'(\d{2}:)?(\d{2}):(\d{2})[.,](\d{3})',start)
        if not m: continue
        h=int(m.group(1)[:2]) if m.group(1) else 0; mins=int(m.group(2)); secs=int(m.group(3)); millis=int(m.group(4))
        ts=h*3600+mins*60+secs+millis/1000
        text=' '.join(x for x in lines if '-->' not in x and not re.match(r'^\d+$',x.strip()))
        text=re.sub(r'<[^>]*>|\{\\an\d+\}', '', text).strip()
        if text: entries.append({'start':ts,'text':text})
    return entries

def score_text(text):
    words=re.findall(r"[a-zA-Z']+",text.lower()); n=len(words)
    hook=sum(1 for phrase in HOOKS if phrase in text.lower())
    punctuation=int(any(p in text for p in ['!','?']))
    density=min(n/24,1)
    score=min(98,round(28+hook*10+punctuation*7+density*18))
    return score

@app.post('/api/analyze-url')
def analyze(req: AnalyzeRequest):
    url=str(req.url)
    if not any(host in url.lower() for host in ['youtube.com','youtu.be','youtube-nocookie.com']):
        raise HTTPException(400,'Only YouTube URLs are supported by this endpoint currently.')
    clip_count=max(1,min(req.clip_count,20)); duration=max(15,min(req.clip_duration,90))
    with tempfile.TemporaryDirectory(prefix='zshorts-') as tmp:
        out=Path(tmp)/'captions.%(ext)s'
        cmd=['yt-dlp','--skip-download','--write-subs','--write-auto-subs','--sub-langs','en.*,en','--sub-format','vtt','--no-warnings','-o',str(out),'--dump-single-json',url]
        try:
            result=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
        except subprocess.TimeoutExpired:
            raise HTTPException(504,'YouTube metadata/caption request timed out.')
        if result.returncode!=0:
            raise HTTPException(502,'yt-dlp could not retrieve this video. It may be private, age-restricted, unavailable, or captions may be blocked.')
        try: meta=json.loads(result.stdout.strip().splitlines()[-1])
        except Exception: meta={}
        vtts=list(Path(tmp).glob('captions*.vtt'))+list(Path(tmp).glob('*.vtt'))
        if not vtts:
            # Metadata query can return without downloading captions; ask yt-dlp for captions separately.
            cmd2=['yt-dlp','--skip-download','--write-subs','--write-auto-subs','--sub-langs','en.*,en','--sub-format','vtt','--no-warnings','-o',str(out),url]
            subprocess.run(cmd2,capture_output=True,text=True,timeout=120)
            vtts=list(Path(tmp).glob('*.vtt'))
        if not vtts:
            raise HTTPException(422,'No English captions found. This video may not have captions available.')
        entries=parse_vtt(vtts[0].read_text(errors='ignore'))
        if not entries: raise HTTPException(422,'Captions were downloaded but could not be parsed.')
        # Group neighboring caption cues into windows, rank, then de-overlap
        windows=[]
        step=max(10,duration//2)
        total=max(e['start'] for e in entries)
        for start in range(0,int(total)+1,step):
            chunk=[e for e in entries if start<=e['start']<start+duration]
            text=' '.join(e['text'] for e in chunk)
            if len(text.split())<8: continue
            windows.append({'start':start,'length':duration,'score':score_text(text),'text':text[:240]})
        windows.sort(key=lambda c:c['score'],reverse=True)
        chosen=[]
        for c in windows:
            if all(abs(c['start']-x['start'])>=duration*.65 for x in chosen):
                chosen.append(c)
                if len(chosen)>=clip_count: break
        chosen.sort(key=lambda c:c['start'])
        return {'title':meta.get('title','YouTube video'),'duration':meta.get('duration'),'clips':chosen,'note':'Caption-based keyword/density scoring heuristic; not a trained viral-prediction model.'}
