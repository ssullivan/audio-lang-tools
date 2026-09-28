"""altools: command line.

    altools check [--items FILE] [--no-words]
        items (JSON list, stdin by default): [{ id, path, jyutping, text?, voice? }]
        → JSON list of results on stdout (see check.py); progress on stderr
    altools bench make --words FILE [--voice V ...]   labelled clips (bench.py) in
                                                  each voice (the first is the main one)
    altools bench train                           tone model from the train split
    altools bench score [--results FILE] [--name N] [--no-fail]
        score the checker (or another checker's results) on the test split;
        exits 1 below bench/thresholds.json
    altools bench items [--split test]            the benchmark's items as JSON,
                                                  for scoring another checker
"""
import argparse
import json
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog='altools')
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('check')
    c.add_argument('--items')
    c.add_argument('--no-words', action='store_true')
    b = sub.add_parser('bench')
    bsub = b.add_subparsers(dest='bcmd', required=True)
    m = bsub.add_parser('make')
    m.add_argument('--words', required=True)
    m.add_argument('--voice', action='append')
    bsub.add_parser('train')
    s = bsub.add_parser('score')
    s.add_argument('--results')
    s.add_argument('--name', default='new')
    s.add_argument('--no-fail', action='store_true')
    it = bsub.add_parser('items')
    it.add_argument('--split', default='test')
    a = ap.parse_args(argv)

    if a.cmd == 'check':
        from .check import Checker
        items = json.load(open(a.items) if a.items else sys.stdin)
        checker = Checker(words=not a.no_words)
        out = []
        for k, item in enumerate(items, 1):
            out.append(checker.check(item))
            sys.stderr.write(f'\r{k}/{len(items)}')
        sys.stderr.write('\n')
        json.dump(out, sys.stdout, ensure_ascii=False)
        return 0

    from . import bench
    if a.bcmd == 'make':
        bench.make(json.load(open(a.words)), **({'voices': a.voice} if a.voice else {}))
    elif a.bcmd == 'train':
        bench.train()
    elif a.bcmd == 'score':
        results = json.load(open(a.results)) if a.results else None
        return bench.score(results, a.name, fail=not a.no_fail)
    elif a.bcmd == 'items':
        json.dump([i for i in bench.load_manifest() if i['split'] == a.split], sys.stdout, ensure_ascii=False)
    return 0


if __name__ == '__main__':
    sys.exit(main())
