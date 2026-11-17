"""sameshot scan PATH... | sameshot compare A B"""

from __future__ import annotations

import argparse
import json
import sys

from . import align, embed
from .scan import log, scan


def main(argv=None):
    parser = argparse.ArgumentParser(prog='sameshot', description='Find copies of the same photo.')
    commands = parser.add_subparsers(dest='command', required=True)
    scanning = commands.add_parser('scan', help='group copies across files and folders')
    scanning.add_argument('paths', nargs='+', help='image files or folders (searched recursively)')
    scanning.add_argument('-o', '--output', help='write the JSON report here instead of stdout')
    scanning.add_argument('--device', default='auto', help='torch device for SSCD (default: cuda if available)')
    scanning.add_argument('--threshold', type=float, default=embed.THRESHOLD,
                          help='SSCD similarity for a candidate pair (default: %(default)s)')
    scanning.add_argument('--workers', type=int, help='alignment processes (default: up to 8)')
    comparing = commands.add_parser('compare', help='check one pair of files')
    comparing.add_argument('a')
    comparing.add_argument('b')
    for command in (scanning, comparing):
        command.add_argument('--cache', help='cache directory (default: ~/.cache/sameshot)')
        command.add_argument('--min-overlap', type=float, default=align.MIN_OVERLAP,
                             help='share of the content the warp must cover (default: %(default)s)')
        command.add_argument('--min-matching-tiles', type=float, default=align.MIN_MATCHING_TILES,
                             help='share of tiles that must correlate at 0.99 (default: %(default)s)')
    args = parser.parse_args(argv)

    if args.command == 'compare':
        result = align.compare(args.a, args.b, cache_dir=args.cache, min_overlap=args.min_overlap,
                               min_matching_tiles=args.min_matching_tiles)
        print(json.dumps(result, indent=2))
        return 0 if result['same'] else 1

    report = scan(args.paths, cache_dir=args.cache, device=args.device, threshold=args.threshold,
                  min_overlap=args.min_overlap, min_matching_tiles=args.min_matching_tiles, workers=args.workers)
    text = json.dumps(report, indent=2)
    if args.output:
        with open(args.output, 'w') as file:
            file.write(text + '\n')
    else:
        print(text)
    extra = sum(len(group['files']) - 1 for group in report['groups'])
    log(f"{len(report['groups'])} groups, {extra} extra copies, {len(report['errors'])} unreadable")
    return 0


if __name__ == '__main__':
    sys.exit(main())
