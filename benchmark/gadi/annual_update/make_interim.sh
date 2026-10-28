#!/bin/bash
# make_interim.sh YEAR : build $A/interim_hybrid = symlinks to the history
# years (first hit in INTERIM_SOURCES) + the new $YEAR files from $A/interim_500m
# (links may dangle until step1 has run).  Idempotent.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh" "$1"
yp=$((YEAR - 1)); n=0; bad=0
for v in $INTERIM_VARS; do
  mkdir -p "$HYBRID/$v"; y0=2003; [[ $v == rain ]] && y0=2002
  for yy in $(seq $y0 $yp); do
    tgt=""
    for src in $(IFS=:; for s in $INTERIM_SOURCES; do eval echo $s; done); do
      [[ -e $src/$v/${v}_500m_$yy.nc ]] && { tgt=$src/$v/${v}_500m_$yy.nc; break; }
    done
    if [[ -z $tgt ]]; then echo "MISSING $v $yy" >&2; bad=$((bad + 1)); continue; fi
    ln -sfn "$tgt" "$HYBRID/$v/${v}_500m_$yy.nc"; n=$((n + 1))
  done
  ln -sfn "$INTERIM/$v/${v}_500m_$YEAR.nc" "$HYBRID/$v/${v}_500m_$YEAR.nc"
done
rm -rf "$HYBRID"/*/.ipynb_checkpoints
echo "hybrid interim $HYBRID: $n history links + 13 links to $INTERIM ($YEAR); $bad missing"
log "make_interim: $n history links, $bad missing (sources $INTERIM_SOURCES)"
exit $(( bad > 0 ))
