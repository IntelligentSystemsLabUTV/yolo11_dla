#!/usr/bin/env python3
"""Extract historical commands and terminal summaries without executing any command.

Local conversations are optional and read only. Only relevant command blocks
are retained; complete conversations, credentials and unrelated content are not copied.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import ROOT, WORKSPACE

COMMAND = re.compile(r'^(?:(?:/\S*/)?trtexec\b|(?:python3?|\./tools/|bash tools/|yolo export\b).*)')
RELEVANT = re.compile(r'trtexec|yolo_e2e_tester|train_coco_dla|coco_dla_train|examples/(?:train_dla|dla_decode)|yolo export')
SECRET = re.compile(r'(?:API_KEY|api_key|access_token|Authorization:|Bearer\s)', re.I)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_commands(text):
    lines=text.splitlines()
    results=[]
    i=0
    while i<len(lines):
        stripped=lines[i].strip()
        start=i
        if '# trtexec ' in stripped and stripped.startswith('&&&&'):
            if 'RUNNING' not in stripped:
                i+=1;continue
            stripped=stripped.split('# ',1)[1]
        else:
            stripped=re.sub(r'^(?:❯|\$)\s*','',stripped)
        if COMMAND.match(stripped) and RELEVANT.search(stripped):
            parts=[stripped]
            while parts[-1].rstrip().endswith('\\') and i+1<len(lines):
                i+=1;parts.append(lines[i])
            command='\n'.join(parts)
            if not SECRET.search(command):
                results.append((start+1,command))
        i+=1
    return results


def scope(command):
    if re.search(r'yolo-dla-n|yolo26|yolo11\w*-.*seg|yolo11n-seg',command,re.I):
        return 'excluded_model_or_task'
    if re.search(r'yolo11|train_coco_dla|coco_dla_train|dla_decode',command,re.I):
        return 'yolo11_or_training_tool'
    return 'unspecified_model'


def terminal_records(path, source_label, kind):
    text=path.read_text()
    items=[]
    for line,command in extract_commands(text):
        if scope(command)=='excluded_model_or_task':continue
        item={'command':command,'scope':scope(command),'evidence':kind,
              'source':source_label,'source_sha256':digest(path),'line':line}
        if command.startswith('trtexec'):
            normalized=' '.join(command.replace('\\\n',' ').split())
            status='PASSED' if 'PASSED TensorRT.trtexec' in text else 'FAILED' if 'FAILED TensorRT.trtexec' in text else 'unknown'
            item.update(outcome=status,normalized_command=normalized)
        elif 'yolo_e2e_tester' in command:
            # Bound each summary to its own invocation, not the other engines in the same paste.
            tail='\n'.join(text.splitlines()[line-1:])
            next_prompt=re.search(r'\n❯\s',tail)
            if next_prompt:tail=tail[:next_prompt.start()]
            stages={}
            for stage,*values in re.findall(r'^\s*(\w+_ms)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$',tail,re.M):
                stages[stage]=dict(zip(['mean','median','p95','p99','min','max'],map(float,values)))
            item.update(outcome='terminal_summary' if stages else 'command_only', stages_ms=stages,
                        runtime_lines=[s.strip() for s in tail.splitlines() if any(k in s for k in
                                       ['logical=', 'strides=', 'Post-processing contract:', 'default stream in enqueueV3'])],
                        per_frame_samples_recovered=False)
        items.append(item)
    return items


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--logs',type=Path,default=WORKSPACE/'logs/fp16_matrix')
    p.add_argument('--sources',type=Path,default=ROOT/'history/sources')
    p.add_argument('--sessions',type=Path,help='Optional local conversation JSONL directory')
    p.add_argument('--before',default='2026-09-09',help='Exclude messages at/after this UTC date')
    p.add_argument('--output',type=Path,required=True,help='New JSON file; refuses overwrite')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    records=[]
    for path in sorted(args.logs.glob('yolo11*.log')):
        records.extend(terminal_records(path,str(path.relative_to(WORKSPACE)) if path.is_relative_to(WORKSPACE) else str(path),
                                        'retained_trtexec_log'))
    for path in sorted(args.sources.glob('*.txt')):
        records.extend(terminal_records(path,str(path.relative_to(ROOT)), 'recovered_user_terminal_attachment'))
    sessions_searched=[]
    if args.sessions:
        for path in sorted(args.sessions.rglob('*.jsonl')):
            selected=[]
            for line,raw in enumerate(path.open(),1):
                try:event=json.loads(raw)
                except ValueError:continue
                timestamp=event.get('timestamp','')
                if not timestamp or timestamp[:10]>=args.before:continue
                data=event.get('payload',{})
                if event.get('type')!='response_item' or data.get('type')!='message':continue
                text='\n'.join(c.get('text','') for c in data.get('content',[]) if isinstance(c,dict))
                if not RELEVANT.search(text):continue
                # Fenced code is an assistant suggestion, never proof of target execution.
                blocks=re.findall(r'```[^\n]*\n(.*?)```',text,re.S) if data.get('role')=='assistant' else [text]
                for block in blocks:
                    for command_line,command in extract_commands(block):
                        if scope(command)=='excluded_model_or_task':continue
                        selected.append({'command':command,'scope':scope(command),
                            'evidence':'assistant_suggested_command' if data.get('role')=='assistant' else 'user_message_command',
                            'source':path.name,'session_line':line,'block_line':command_line,
                            'source_sha256':digest(path),'timestamp':timestamp,'outcome':'execution_not_established'})
            if selected:
                sessions_searched.append({'file':path.name,'sha256':digest(path),'commands':len(selected)})
                records.extend(selected)
    # Deduplicate prompt/RUNNING duplicates within one source while preserving distinct sources.
    unique={}
    for item in records:
        normalized=' '.join(item['command'].replace('\\\n',' ').split())
        key=(item['source'],item['evidence'],normalized)
        if key not in unique:unique[key]=item
    records=list(unique.values())
    for index,item in enumerate(records,1):
        item['id']=f'CMD-{index:03d}'
        try:item['argv']=shlex.split(item['command'].replace('\\\n',' '))
        except ValueError:item['argv']=None
    report={'scope':'YOLO11 historical commands; no execution performed by this extractor',
            'before_utc_date':args.before,'sessions_searched':sessions_searched,'commands':records,
            'limits':['Assistant commands are suggestions, not executed experiments.',
                      'Recovered terminal summaries do not contain per-frame timing arrays.',
                      'Paths and defaults are preserved as recorded, not rewritten as new commands.']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(f'Saved {len(records)} commands to {args.output}')


if __name__=='__main__':main()
