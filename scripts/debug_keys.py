#!/usr/bin/env python3
"""Debug state_dict key mismatch."""
import re, sys

with open(sys.argv[1]) as f:
    txt = f.read()

# Extract first few missing and unexpected keys
for label in ['Missing key', 'Unexpected key']:
    m = re.search(rf'{label}\(s\) in state_dict:\s*(.*?)(\n\t|$)', txt, re.DOTALL)
    if m:
        raw = m.group(1)
        keys = re.findall(r'"([^"]+)"', raw)
        print(f'\n{label}s ({len(keys)} total):')
        for k in keys[:5]:
            print(f'  {k}')
        if len(keys) > 5:
            print(f'  ... and {len(keys)-5} more')
