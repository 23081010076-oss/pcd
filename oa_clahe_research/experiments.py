"""Generate or run the exact 15 main jobs; no test data evaluated."""
import argparse, json, subprocess, sys
from pathlib import Path
from core import load_config, file_hash

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');a=p.parse_args()
    c=load_config();jobs=[[sys.executable,'train.py','--scenario',s,'--seed',str(seed)] for s in c['scenarios'] for seed in c['seeds']]
    assert len(jobs)==15
    Path('reports').mkdir(exist_ok=True)
    Path('reports/experiment_plan.json').write_text(json.dumps({'jobs':jobs,'config_sha256':file_hash('config.json'),'manifest_sha256':file_hash('data/manifest.csv')},indent=2))
    for cmd in jobs:
        print(subprocess.list2cmdline(cmd),flush=True)
        if a.execute: subprocess.run(cmd,check=True)
if __name__=='__main__':main()
