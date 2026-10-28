#!/bin/bash
# Submit a full step3 year as month-group jobs plus a merge job that runs only
# if all of them succeed (-W depend=afterok:...).  Default = the production
# recipe: 4 threads per job, 3 months per job (4 jobs), compiled models.
#
#   benchmark/gadi/pbs/submit_step3_year.sh FLUX YEAR [step3 flags ...]
#   e.g.  submit_step3_year.sh GPP 2025 \
#             --features_path /scratch/ub8/pl5189/auseflux/features_500m \
#             --static_path /g/data/xc0/project/AusEFlux/data \
#             --variables_file /g/data/xc0/project/AusEFlux/results/variables.txt \
#             --output_path /scratch/ub8/pl5189/auseflux/results/AusEFlux
#
# Flags are passed to every job through an args file (so commas and quoting
# survive qsub).  --output_path, if given, is also used by the merge.
# Env:
#   THREADS         cpus per job = step3 --threads (default 4); mem = 4 GB x THREADS
#   MONTHS_PER_JOB  months per job (default 3; consecutive months -> --months a-b)
#   WALLTIME        per job (default 04:00:00)
#   MODELS_PATH     --models_path (default $WORK/results/models_compiled if it
#                   holds $FLUX, else /g/data/xc0/project/AusEFlux/results/models/ensemble/v2.1);
#                   ignored if --models_path is among the flags
#   MONTHS          months to run (default "1 2 ... 12")
#   QUEUE           optional PBS queue (e.g. normalsr)
#   DEPEND          optional ':'-separated job ids; every month job waits for
#                   them (-W depend=afterok:...), e.g. the step2 jobs
#   LOG_DIR         directory for the PBS logs (default: the templates' $WORK/logs/)
#   WORK, CODE      as in _job_common.sh
#   DRY_RUN=1       print the qsub commands instead of submitting
# Old behaviour (one 24-cpu job per month): THREADS=24 MONTHS_PER_JOB=1.
set -euo pipefail
if [[ $# -lt 2 ]]; then
    sed -n '2,27p' "$0"; exit 2
fi
FLUX=$1; YEAR=$2; shift 2
WORK=${WORK:-/scratch/ub8/pl5189/auseflux}
CODE=${CODE:-$WORK/code/AusEFlux}
PBS_DIR=$CODE/benchmark/gadi/pbs
MONTHS=${MONTHS:-"1 2 3 4 5 6 7 8 9 10 11 12"}
THREADS=${THREADS:-4}
MONTHS_PER_JOB=${MONTHS_PER_JOB:-3}
WALLTIME=${WALLTIME:-04:00:00}
MEM_GB=$((4 * THREADS))
if ! [[ "$THREADS" =~ ^[1-9][0-9]*$ && "$MONTHS_PER_JOB" =~ ^[1-9][0-9]*$ ]]; then
    echo "THREADS and MONTHS_PER_JOB must be positive integers" >&2; exit 2
fi

OUTPUT=${AUSEFLUX_ROOT:-$WORK}/results/AusEFlux
args=("$@")
have_models=0
for ((i = 0; i < ${#args[@]}; i++)); do
    case "${args[$i]}" in
        --output_path) OUTPUT=${args[$((i + 1))]} ;;
        --output_path=*) OUTPUT=${args[$i]#--output_path=} ;;
        --models_path|--models_path=*) have_models=1 ;;
    esac
done
if [[ $have_models == 0 ]]; then
    if [[ -z "${MODELS_PATH:-}" ]]; then
        MC=$WORK/results/models_compiled
        if [[ -d $MC/$FLUX ]]; then
            MODELS_PATH=$MC
        else
            MODELS_PATH=/g/data/xc0/project/AusEFlux/results/models/ensemble/v2.1
            echo "note: no compiled models in $MC/$FLUX; using $MODELS_PATH" >&2
        fi
    fi
    args+=(--models_path "$MODELS_PATH")
elif [[ -n "${MODELS_PATH:-}" ]]; then
    echo "note: --models_path given in the flags; MODELS_PATH ignored" >&2
fi

mkdir -p "$WORK/logs/args"
ARGS_FILE=$WORK/logs/args/step3_${FLUX}_${YEAR}_$(date +%Y%m%dT%H%M%S)_$$.args
: > "$ARGS_FILE"
for a in "${args[@]}"; do printf '%s\n' "$a" >> "$ARGS_FILE"; done

qsub_() {
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        printf 'qsub' >&2; printf ' %q' "$@" >&2; echo >&2
        echo "DRYRUN.$RANDOM"
    else
        qsub "$@"
    fi
}
QARGS=(-l "ncpus=$THREADS,mem=${MEM_GB}GB,walltime=$WALLTIME")
[[ -n "${QUEUE:-}" ]] && QARGS+=(-q "$QUEUE")
[[ -n "${DEPEND:-}" ]] && QARGS+=(-W "depend=afterok:$DEPEND")
LOGARGS=()
if [[ -n "${LOG_DIR:-}" ]]; then
    mkdir -p "$LOG_DIR"; LOGARGS=(-o "$LOG_DIR/")
fi

# group the months: consecutive runs -> "a-b", otherwise "a:b:c"
read -r -a mlist <<< "$MONTHS"
ids=()
for ((i = 0; i < ${#mlist[@]}; i += MONTHS_PER_JOB)); do
    grp=("${mlist[@]:i:MONTHS_PER_JOB}")
    first=${grp[0]}; last=${grp[$((${#grp[@]} - 1))]}
    if [[ ${#grp[@]} == 1 ]]; then
        sel=$first; tag=$(printf %02d "$first")
    elif (( 10#$last - 10#$first + 1 == ${#grp[@]} )) && \
         [[ "${grp[*]}" == "$(seq -s ' ' $((10#$first)) $((10#$last)))" ]]; then
        sel=$((10#$first))-$((10#$last)); tag=$(printf %02d-%02d "$first" "$last")
    else
        sel=$(IFS=:; echo "${grp[*]}"); tag=$(IFS=_; echo "${grp[*]}")
    fi
    id=$(qsub_ -N "s3_${FLUX}_${YEAR}_m$tag" "${QARGS[@]}" "${LOGARGS[@]}" \
        -v "WORK=$WORK,CODE=$CODE,FLUX=$FLUX,YEAR=$YEAR,MONTH=$sel,THREADS=$THREADS,ARGS_FILE=$ARGS_FILE" \
        "$PBS_DIR/step3_month.pbs")
    echo "months $sel -> $id"
    ids+=("$id")
done
dep=$(IFS=:; echo "${ids[*]}")
MQ=(); [[ -n "${QUEUE:-}" ]] && MQ=(-q "$QUEUE")
mid=$(qsub_ -N "s3_${FLUX}_${YEAR}_merge" "${MQ[@]}" "${LOGARGS[@]}" -W "depend=afterok:$dep" \
    -v "WORK=$WORK,CODE=$CODE,FLUX=$FLUX,YEAR=$YEAR,OUTPUT=$OUTPUT,MONTHS=$(echo $MONTHS | tr ' ' ':')" \
    "$PBS_DIR/merge.pbs")
echo "merge -> $mid  (after ${#ids[@]} jobs of ${THREADS} cpus / ${MEM_GB} GB / $WALLTIME; output $OUTPUT/$FLUX/)"
echo "args file: $ARGS_FILE"
