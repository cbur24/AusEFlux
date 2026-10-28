# Sourced by the annual-update scripts (not a job script).  Usage:
#   source common.sh YEAR
# Defines the work tree for one annual update and a qsub wrapper that
# honours DRY_RUN=1 (prints the command, returns a fake id).
#
# Env overrides (all optional):
#   WORK     scratch work tree           default /scratch/ub8/pl5189/auseflux
#   CODE     repo checkout               default $WORK/code/AusEFlux
#   XC       reference project (read only) default /g/data/xc0/project/AusEFlux
#   PUB      published products          default /g/data/ub8/au/AusEFlux/v2.1/monthly
#   RAW      raw inputs                  default /g/data/ub8/au
#   ANNUAL   parent of the per-year trees default $WORK/annual
#   PBS_PROJECT  PBS project (not PROJECT: Gadi exports that)  default ea89
#   INTERIM_SOURCES  ':'-separated interim trees searched in order for the
#            history years (default: reference interim, then earlier annual
#            updates, then the V25 2025 interim)
#   CLIM_SRC stored climatologies to reuse (default: the V25 ones); empty = none
#   MODELS   reference joblib models      default $XC/results/models/ensemble/v2.1
#   MODELS_COMPILED compiled copies       default $WORK/results/models_compiled
#   VARIABLES_FILE  default $XC/results/variables.txt
set -euo pipefail
YEAR=${1:?usage: source common.sh YEAR}
[[ "$YEAR" =~ ^20[0-9][0-9]$ ]] || { echo "bad YEAR $YEAR" >&2; exit 2; }
WORK=${WORK:-/scratch/ub8/pl5189/auseflux}
CODE=${CODE:-$WORK/code/AusEFlux}
XC=${XC:-/g/data/xc0/project/AusEFlux}
PUB=${PUB:-/g/data/ub8/au/AusEFlux/v2.1/monthly}
RAW=${RAW:-/g/data/ub8/au}
ANNUAL=${ANNUAL:-$WORK/annual}
PBS_PROJECT=${PBS_PROJECT:-ea89}
A=$ANNUAL/$YEAR
INTERIM=$A/interim_500m            # new step1 output for $YEAR
HYBRID=$A/interim_hybrid           # history (links) + $YEAR, read by step2
FEATURES=$A/features_500m
PRODUCT=$A/product
LOGS=$A/logs
CMP=$A/cmp
INTERIM_SOURCES=${INTERIM_SOURCES:-$XC/data/interim_500m:$ANNUAL/*/interim_500m:$WORK/v25/interim_500m}
CLIM_SRC=${CLIM_SRC-$WORK/v25/features_500m/climatology}
MODELS=${MODELS:-$XC/results/models/ensemble/v2.1}
MODELS_COMPILED=${MODELS_COMPILED:-$WORK/results/models_compiled}
VARIABLES_FILE=${VARIABLES_FILE:-$XC/results/variables.txt}
STATIC=$XC/data
PBS_DIR=$CODE/benchmark/gadi/pbs
HERE=$CODE/benchmark/gadi/annual_update
STORAGE=gdata/xc0+gdata/ub8+scratch/ub8
INTERIM_VARS="kNDVI kTavg LAI LST NDVI NDWI rain SRAD Tavg Tmax Tmin VegH VPD"
FLUXES=${FLUXES:-"GPP NEE ER ET"}

qsub_() {
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        { printf 'qsub'; printf ' %q' "$@"; echo; } >&2
        echo "DRY.$RANDOM"
    else
        qsub "$@"
    fi
}
log() { echo "[$(date +%H:%M:%S)] $*"; echo "- $(date -Is) $*" >> "$A/RUNLOG.md"; }
mkdir -p "$A" "$LOGS" "$CMP"
