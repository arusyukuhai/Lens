"""Integer-only inference. This module does not import torch or numpy."""
import argparse
import json
from pathlib import Path
from discrete_replace import discrete_replace

parser=argparse.ArgumentParser()
parser.add_argument('program',help='learned_pipeline.json or experiment results JSON')
parser.add_argument('--input',required=True,help='binary input file')
parser.add_argument('--output',required=True,help='binary output file')
args=parser.parse_args()
program=json.loads(Path(args.program).read_text())
raw=tuple(Path(args.input).read_bytes())
for r in program['rules']:
    raw=discrete_replace(raw,r['pattern'],r['replacement'],
                         min_wildcard_len=program.get('min_wildcard_len',1))
Path(args.output).write_bytes(bytes(raw))
