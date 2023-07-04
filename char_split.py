import sys

for line in sys.stdin:
    line = line.strip()
    line = line.replace(' ', '▁')
    print(' '.join(list(line)))
