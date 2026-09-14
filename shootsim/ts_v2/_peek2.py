# -*- coding: utf-8 -*-
"""Scan a log for distinct line shapes to understand the frame-record format."""
import io
import re
import sys
from collections import Counter

p = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\12951\Desktop\12323\logs\aim_20260906_003353.log"
shapes = Counter()
examples = {}
with io.open(p, encoding="utf-8", errors="replace") as f:
    for i, line in enumerate(f):
        t = line.rstrip()
        if not t:
            continue
        # generalize: replace numbers with #
        s = re.sub(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", "#", t)
        s = re.sub(r"#+", "#", s)
        shapes[s[:160]] += 1
        if s[:160] not in examples:
            examples[s[:160]] = (i, t[:400])
for s, c in shapes.most_common(40):
    print("count=%d  %s" % (c, s))
    i, ex = examples[s]
    print("   e.g. line %d: %s" % (i, ex))
    print()
