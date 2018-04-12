#!/usr/bin/env python
import sys

for line in sys.stdin:
    for w in line.strip():
        print "%s" % w