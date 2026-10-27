import json, glob, sys
import numpy as np
D = '/scratch/ub8/pl5189/auseflux/results/e5/parity'
for tag in sys.argv[1:]:
    for w in ['north', 'centre', 'south']:
        d = {}
        for f in glob.glob(f'{D}/{tag}_{w}_p*.json'):
            d.update(json.load(open(f)))
        rows = []
        for v, m in sorted(d.items()):
            n = sum(m['n']); eq = sum(m['n_equal']); r = sum(m['n_rel1e3']); mm = sum(m['mask_mismatch'])
            sse = sum(m['sse']); sb = sum(m['sb']); sb2 = sum(m['sb2'])
            std = np.sqrt(max(sb2 / n - (sb / n) ** 2, 1e-30)); rmse = np.sqrt(sse / n)
            if r or mm or rmse / std > 1e-4:
                rows.append(f"{v}: eq {100*eq/n:.1f}% >1e-3rel {100*r/n:.3f}% mm {mm} rmse/std {rmse/std:.1e} max {max(m['max_abs']):.3g}")
        print(f'## {tag} {w} ({len(d)} features; listed = any >1e-3 rel or mask diff or rmse/std>1e-4)')
        for x in rows:
            print('  ' + x)
