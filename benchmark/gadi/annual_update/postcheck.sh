#!/bin/bash
# postcheck.sh YEAR : for each flux product of the annual update,
#   1. check_format.py --any-year against the latest published file of that
#      flux (same layout: dims, dtypes, attributes, spatial_ref, time pattern);
#   2. a monthly summary (continental mean of the three bands, valid-pixel
#      fraction) next to the same numbers for the latest published year, so a
#      broken month stands out.  Writes $A/cmp/summary_YEAR.md; exit 1 if a
#      format check fails or a product is missing.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh" "$1"
rc=0; S=$CMP/summary_$YEAR.md
{ echo "# Annual update $YEAR: post-check ($(date -Is))"; echo; } > "$S"
for F in $FLUXES; do
    new=$PRODUCT/$F/AusEFlux_${F}_500m_quantiles_${YEAR}_v2.1.nc
    ref=$(ls "$PUB/$F"/AusEFlux_${F}_500m_quantiles_*_v2.1.nc 2>/dev/null | sort | tail -1)
    echo "## $F" >> "$S"
    if [[ ! -e $new ]]; then echo "MISSING $new" | tee -a "$S"; rc=1; continue; fi
    echo "- new: \`$new\`" >> "$S"; echo "- layout reference: \`$ref\`" >> "$S"
    echo "### format"; python "$CODE/benchmark/gadi/check_format.py" --any-year "$new" "$ref" | tee "$CMP/format_${F}_$YEAR.txt"
    grep -q "FORMAT: PASS" "$CMP/format_${F}_$YEAR.txt" || rc=1
    { echo; echo '```'; grep -E "^(FAIL|FORMAT)" "$CMP/format_${F}_$YEAR.txt"; echo '```'; } >> "$S"
    python - "$new" "$ref" "$F" >> "$S" <<'PY'
import sys, netCDF4, numpy as np
new, ref, F = sys.argv[1:4]
def table(path):
    rows = []
    with netCDF4.Dataset(path) as d:
        d.set_auto_mask(False)
        t = d['time']; stamps = netCDF4.num2date(t[:], t.units, getattr(t, 'calendar', 'standard'))
        for i, ts in enumerate(stamps):
            vals = []
            for b in ('25th_percentile', 'median', '75th_percentile'):
                a = d[f'{F}_{b}'][i]; ok = np.isfinite(a)
                vals.append((float(a[ok].mean()) if ok.any() else float('nan'), ok.mean()))
            rows.append((f'{ts.year}-{ts.month:02d}', vals))
    return rows
print(); print('| month | new q25 | new median | new q75 | valid % | ref month | ref q25 | ref median | ref q75 | valid % |')
print('|---|---|---|---|---|---|---|---|---|---|')
for (m, nv), (mr, rv) in zip(table(new), table(ref)):
    print(f'| {m} | {nv[0][0]:.3f} | {nv[1][0]:.3f} | {nv[2][0]:.3f} | {100*nv[1][1]:.2f} | {mr} | {rv[0][0]:.3f} | {rv[1][0]:.3f} | {rv[2][0]:.3f} | {100*rv[1][1]:.2f} |')
print()
PY
done
echo; cat "$S"; echo "summary: $S"
log "postcheck: $([[ $rc == 0 ]] && echo PASS || echo FAIL) ($S)"
exit $rc
