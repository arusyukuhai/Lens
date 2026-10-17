"""Pure integer deployment demo. No torch import or probability operations."""
import argparse
import json
from discrete_replace import discrete_replace

parser = argparse.ArgumentParser()
parser.add_argument('rule', help='learned_rule.json')
parser.add_argument('text', help='UTF-8 input; rules operate on bytes, not Unicode code points')
args = parser.parse_args()
with open(args.rule) as f:
    r = json.load(f)
y = bytes(discrete_replace(args.text.encode('utf-8'), r['pattern'], r['replacement'],
          min_wildcard_len=r['min_wildcard_len'], max_output=r['max_output']))
print(y.decode('utf-8', errors='backslashreplace'))
