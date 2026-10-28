#!/bin/bash
# run_year.sh YEAR [STAGES]
#
# Submit a whole AusEFlux annual update for YEAR on Gadi as one dependency
# chain: step1 (5 jobs) -> step2 (19 jobs, stored climatologies) -> compile
# models if needed -> step3 (4 fluxes x 12 months + merges) -> post-check.
# Everything lands in $WORK/annual/YEAR/ (see common.sh).  Rerunning is safe:
# step1/step2/step3 outputs are skip-if-exists, the hybrid interim is rebuilt
# from links, and STAGES limits what is submitted.
#
#   STAGES   default "inputs interim step1 step2 step3 check" (space separated)
#   DRY_RUN=1   print the qsub commands, submit nothing
#   FORCE=1     go on although check_inputs reports missing items
#   FLUXES      default "GPP NEE ER ET"
#   CLIM_SRC    stored climatologies (default V25's); CLIM_SRC= computes fresh ones
#   STEP1_DEP / STEP2_DEP  ':'-joined job ids to depend on when a stage was
#               submitted in an earlier call (printed at the end of each run)
#
# Cost (V25, 2025): step1 12 SU, step2 129 SU fresh climatologies (est. 15-25
# with stored ones), step3 72-90 SU per flux, checks < 5 SU.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh" "$1"
STAGES=${2:-"inputs interim step1 step2 step3 check"}
has() { [[ " $STAGES " == *" $1 "* ]]; }
cp -n "$HERE"/*.sh "$HERE"/*.pbs "$A/" 2>/dev/null || true   # keep the scripts used
echo "annual update $YEAR  tree $A  code $(git -C "$CODE" rev-parse --short HEAD 2>/dev/null || echo n/a)  DRY_RUN=${DRY_RUN:-0}"
log "run_year.sh $YEAR stages='$STAGES' DRY_RUN=${DRY_RUN:-0}"

if has inputs; then
    if ! "$HERE/check_inputs.sh" "$YEAR"; then
        if [[ "${FORCE:-0}" == 1 || "${DRY_RUN:-0}" == 1 ]]; then echo "continuing (FORCE/DRY_RUN)"; else
            echo "inputs missing: fix them or rerun with FORCE=1" >&2; exit 1; fi
    fi
fi
has interim && "$HERE/make_interim.sh" "$YEAR" || true

# ---- step1 ---------------------------------------------------------------
S1=()
if has step1; then
    for grp in climate:SILO:rain_prior NDWI:kNDVI NDVI:LAI LST VegH; do
        res=(-l ncpus=4,mem=16GB,walltime=03:00:00)
        [[ $grp == climate* ]] && res=(-l ncpus=12,mem=48GB,walltime=03:00:00)
        id=$(qsub_ -N "s1_${YEAR}_${grp//:/_}" -P "$PBS_PROJECT" -o "$LOGS/" "${res[@]}" \
            -v "WORK=$WORK,CODE=$CODE,YEAR=$YEAR,VARS=$grp,EXTRA=--results_path $INTERIM --orientation_check" \
            "$PBS_DIR/step1_year.pbs")
        echo "step1 $grp -> $id"; S1+=("$id")
    done
    log "step1 jobs: ${S1[*]}"
fi
S1DEP=$(IFS=:; echo "${S1[*]}"); S1DEP=${S1DEP:-${STEP1_DEP:-}}

# ---- step2 ---------------------------------------------------------------
S2=()
if has step2; then
    mkdir -p "$FEATURES/climatology"
    if [[ -n "${CLIM_SRC:-}" && -d "$CLIM_SRC" ]]; then
        for f in "$CLIM_SRC"/*_clim_500m.nc; do
            [[ "$f" -ef "$FEATURES/climatology/$(basename "$f")" ]] || ln -sfn "$f" "$FEATURES/climatology/"
        done
        echo "step2 reuses $(ls "$FEATURES/climatology" | wc -l) stored climatologies from $CLIM_SRC"
    else
        echo "step2 computes fresh climatologies (≈ +100 SU, +2 h)"
    fi
    q2() { local name=$1 dep=$2; shift 2
        local d=(); [[ -n $dep ]] && d=(-W "depend=afterok:$dep")
        qsub_ -N "$name" -P "$PBS_PROJECT" -o "$LOGS/$name.log" "${d[@]}" \
            -v "WORK=$WORK,CODE=$CODE,INTERIM=$HYBRID,OUTDIR=$FEATURES,YEARS=$YEAR-$YEAR,$*" "$HERE/step2_year.pbs"; }
    declare -A G
    for grp in NDWI:kNDVI NDVI:LAI LST:Tavg SRAD:VPD rain:VegH; do
        G[$grp]=$(q2 "s2_${YEAR}_gf_${grp//:/_}" "$S1DEP" "STAGE=gapfill,VARS=$grp"); echo "gapfill $grp -> ${G[$grp]}"; S2+=("${G[$grp]}")
    done
    ALLG=$(IFS=:; echo "${G[*]}")
    VF=$(q2 "s2_${YEAR}_vegfrac" "${G[NDVI:LAI]}" "STAGE=vegfrac"); echo "vegfrac -> $VF"; S2+=("$VF")
    CR=$(q2 "s2_${YEAR}_cmlrain" "${G[rain:VegH]}" "STAGE=cmlrain"); echo "cmlrain -> $CR"; S2+=("$CR")
    DT=$(q2 "s2_${YEAR}_deltat" "$ALLG" "STAGE=deltat"); echo "deltat -> $DT"; S2+=("$DT")
    for v in NDWI SRAD Tavg VPD kNDVI LAI rain rain_cml3 rain_cml6 rain_cml12; do
        id=$(q2 "s2_${YEAR}_an_$v" "$ALLG:$CR" "STAGE=anom,VARS=$v"); echo "anom $v -> $id"; S2+=("$id")
    done
    C4=$(q2 "s2_${YEAR}_c4" "$VF" "STAGE=c4"); echo "c4 -> $C4"; S2+=("$C4")
    log "step2 jobs: ${S2[*]}"
fi
S2DEP=$(IFS=:; echo "${S2[*]}"); S2DEP=${S2DEP:-${STEP2_DEP:-}}

# ---- step3 ---------------------------------------------------------------
MERGES=()
if has step3; then
    CDEP=()
    for F in $FLUXES; do
        if [[ ! -d $MODELS_COMPILED/$F/compiled ]]; then
            id=$(qsub_ -N "compile_$F" -P "$PBS_PROJECT" -o "$LOGS/" \
                -v "WORK=$WORK,CODE=$CODE,FLUX=$F,MODELS=$MODELS,DEST=$MODELS_COMPILED" "$HERE/compile_models.pbs")
            echo "compile $F -> $id"; CDEP+=("$id")
        fi
    done
    DEP=$S2DEP; [[ ${#CDEP[@]} -gt 0 ]] && DEP="${DEP:+$DEP:}$(IFS=:; echo "${CDEP[*]}")"
    for F in $FLUXES; do
        out=$(DEPEND=$DEP LOG_DIR=$LOGS THREADS=4 MONTHS_PER_JOB=1 WALLTIME=04:00:00 \
              WORK=$WORK CODE=$CODE MODELS_PATH=$MODELS_COMPILED DRY_RUN=${DRY_RUN:-0} \
              "$PBS_DIR/submit_step3_year.sh" "$F" "$YEAR" \
              --features_path "$FEATURES" --static_path "$STATIC" \
              --variables_file "$VARIABLES_FILE" --output_path "$PRODUCT")
        echo "$out" | sed "s/^/step3 $F: /"
        MERGES+=("$(echo "$out" | awk '/^merge ->/{print $3}')")
    done
    log "step3 merges: ${MERGES[*]}"
fi
MDEP=$(IFS=:; echo "${MERGES[*]}")

# ---- post-check ----------------------------------------------------------
if has check; then
    d=(); [[ -n $MDEP ]] && d=(-W "depend=afterok:$MDEP")
    id=$(qsub_ -N "check_$YEAR" -P "$PBS_PROJECT" -o "$LOGS/check_$YEAR.log" "${d[@]}" \
        -v "WORK=$WORK,CODE=$CODE,YEAR=$YEAR,FLUXES=${FLUXES// /:}" "$HERE/postcheck.pbs")
    echo "post-check -> $id (format vs published layout + monthly summary in $CMP)"
fi

cat <<MSG

Submitted.  Watch with:  qstat -u \$USER | grep _$YEAR
Logs: $LOGS   products: $PRODUCT/{FLUX}/AusEFlux_{FLUX}_500m_quantiles_${YEAR}_v2.1.nc
Resume later stages with STEP1_DEP=$S1DEP STEP2_DEP=$S2DEP
A month killed at walltime deletes its merge (afterok): resubmit the month with
  qsub -N s3_F_${YEAR}_mMM -l ncpus=4,mem=16GB,walltime=04:00:00 -o $LOGS/ -v WORK=$WORK,CODE=$CODE,FLUX=F,YEAR=$YEAR,MONTH=MM,THREADS=4,ARGS_FILE=<args file printed above> $PBS_DIR/step3_month.pbs
then  qsub -N s3_F_${YEAR}_merge -W depend=afterok:<id> -o $LOGS/ -v WORK=$WORK,CODE=$CODE,FLUX=F,YEAR=$YEAR,OUTPUT=$PRODUCT $PBS_DIR/merge.pbs
and   $HERE/run_year.sh $YEAR check
MSG
