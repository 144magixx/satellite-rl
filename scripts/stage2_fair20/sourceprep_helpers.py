"""Read-only official asset discovery and bounded SHA-locked transfers."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
from .tunnel import atomic

DATES=('2024-02-13','2024-02-14','2024-02-15')


def metadata(day):
    if day not in DATES:raise ValueError('only preregistered dates')
    tag='v'+day.replace('-','.')+'-planes-readsb-staging-0'
    url='https://api.github.com/repos/adsblol/globe_history_2024/releases/tags/'+tag
    req=urllib.request.Request(url,headers={'User-Agent':'satellite-rl-S2-F20','Accept':'application/vnd.github+json'})
    with urllib.request.urlopen(req,timeout=30) as r:raw=r.read()
    record=json.loads(raw)
    found=[a for a in record['assets'] if a['name']==tag+'.tar']
    if len(found)!=1:raise ValueError('official date asset unavailable')
    a=found[0]
    return {'date':day,'release_tag':tag,'asset_id':a['id'],'asset_name':a['name'],'expected_bytes':a['size'],'url':a['browser_download_url'],'official_metadata_sha256':hashlib.sha256(raw).hexdigest(),'official_metadata_canonical_sha256':hashlib.sha256(json.dumps(record,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest(),'official_metadata':record}


def copy_same_source(source, destination, expected_sha256, expected_bytes):
    """Local fallback can move only identical bytes, no substituting source."""
    source=Path(source);destination=Path(destination)
    if source.stat().st_size!=expected_bytes:raise ValueError('source byte length differs')
    destination.parent.mkdir(parents=True,exist_ok=True);tmp=destination.with_suffix(destination.suffix+'.part')
    h=hashlib.sha256()
    with source.open('rb') as r,tmp.open('wb') as w:
        for chunk in iter(lambda:r.read(4*1024*1024),b''):h.update(chunk);w.write(chunk)
    if h.hexdigest()!=expected_sha256:
        tmp.unlink();raise ValueError('fallback source SHA differs')
    tmp.replace(destination)
    return {'path':str(destination.resolve()),'sha256':expected_sha256,'bytes':expected_bytes}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    atomic(a.output,{'experiment_id':'S2-F20','source_assets':[metadata(d) for d in DATES],'status':'official_metadata_only_no_science'})

if __name__=='__main__':main()
