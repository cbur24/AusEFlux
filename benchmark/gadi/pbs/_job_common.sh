# Sourced by the PBS templates in this directory (not a job script itself).
#
# Provides: path/env defaults for Gadi, EXTRA_ARGS (from $EXTRA and
# $ARGS_FILE), run_timed (wraps /usr/bin/time -v) and job_end (prints wall
# time and an SU estimate).
#
# Variables a job may receive via `qsub -v`:
#   WORK       work tree           (default /scratch/ub8/pl5189/auseflux)
#   CODE       repo checkout       (default $WORK/code/AusEFlux)
#   EXTRA      extra CLI flags, space separated (no commas: qsub -v splits on
#              commas; use ARGS_FILE or ':' separators where noted)
#   ARGS_FILE  file with extra CLI arguments, one per line (any characters)

WORK=${WORK:-/scratch/ub8/pl5189/auseflux}
CODE=${CODE:-$WORK/code/AusEFlux}
GADI_TOOLS=$CODE/benchmark/gadi
export AUSEFLUX_INPUT=${AUSEFLUX_INPUT:-/g/data/ub8/au}
export AUSEFLUX_STATIC=${AUSEFLUX_STATIC:-/g/data/xc0/project/AusEFlux/data}
export AUSEFLUX_ROOT=${AUSEFLUX_ROOT:-$WORK}
NCPUS=${PBS_NCPUS:-$(nproc)}
export OMP_NUM_THREADS=$NCPUS OPENBLAS_NUM_THREADS=$NCPUS MKL_NUM_THREADS=$NCPUS
export NUMEXPR_MAX_THREADS=$NCPUS
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1
JOB_START=$(date +%s)

EXTRA_ARGS=()
if [[ -n "${EXTRA:-}" ]]; then
    read -r -a _x <<< "$EXTRA"
    EXTRA_ARGS+=("${_x[@]}")
fi
if [[ -n "${ARGS_FILE:-}" ]]; then
    while IFS= read -r _l || [[ -n "$_l" ]]; do
        [[ -n "$_l" ]] && EXTRA_ARGS+=("$_l")
    done < "$ARGS_FILE"
fi

# commas cannot travel through `qsub -v`; accept ':' or '+' as list separators
commas() { local s=${1//:/,}; echo "${s//+/,}"; }

job_header() {
    echo "=== ${PBS_JOBNAME:-job} ${PBS_JOBID:-local} on $(hostname) $(date -Is)"
    echo "    ncpus=$NCPUS  code=$CODE  git=$(git -C "$CODE" rev-parse --short HEAD 2>/dev/null || echo n/a)"
    echo "    AUSEFLUX_INPUT=$AUSEFLUX_INPUT AUSEFLUX_STATIC=$AUSEFLUX_STATIC AUSEFLUX_ROOT=$AUSEFLUX_ROOT"
    echo "    python=$(command -v python) $(python -c 'import sys;print(sys.version.split()[0])' 2>/dev/null)"
}

run_timed() {
    echo "+ $*"
    /usr/bin/time -v "$@"
}

job_end() {
    local rc=$1 end elapsed
    end=$(date +%s)
    elapsed=$((end - JOB_START))
    local mem_gb
    mem_gb=$(qstat -f "${PBS_JOBID:-none}" 2>/dev/null | awk -F= '/Resource_List.mem/{gsub(/ /,"",$2);print $2}')
    awk -v n="$NCPUS" -v s="$elapsed" -v rc="$rc" -v mem="$mem_gb" 'BEGIN{
        h = s / 3600.0
        printf "=== exit %d  elapsed %02d:%02d:%02d (%.3f h)\n", rc, s/3600, (s%3600)/60, s%60, h
        printf "=== SU estimate (ncpus x hours x 2): %d x %.3f x 2 = %.1f SU\n", n, h, n*h*2
        if (mem != "") {
            m = mem; gb = 0
            if (m ~ /tb$/) gb = m*1024; else if (m ~ /gb$/) gb = m+0; else if (m ~ /mb$/) gb = m/1024; else if (m ~ /kb$/) gb = m/1048576;
            eff = (gb/4 > n) ? gb/4 : n
            printf "=== Gadi normal-queue charge incl. memory (max(ncpus, mem/4GB)=%.1f): %.1f SU\n", eff, eff*h*2
        }
    }'
    exit "$rc"
}
