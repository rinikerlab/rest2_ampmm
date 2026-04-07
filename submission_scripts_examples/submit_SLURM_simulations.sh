#!/bin/bash
set -euo pipefail

# =====================================
# Paths (robust)
# =====================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
INPUT_SCRIPTS_DIR="${PROJECT_ROOT}/input_generation_scripts_examples"

# =====================================
# Usage
# =====================================
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <PDB_NAME> <STAGE> [N_REPS]"
  echo "  <STAGE> one of: nvt | npt | production | rest2"
  echo "  [N_REPS] only for production/rest2 (default: 1)"
  exit 1
fi

PDB_NAME="$1"
STAGE="$(echo "$2" | tr '[:upper:]' '[:lower:]')"
N_REPS="${3:-1}"

# =====================================
# Molecule presets
# =====================================
ALA="ACE ALA NME"
CYC="PRO SER LEU ASP VAL"
P13="ACE PRO PRO PRO PRO PRO PRO PRO PRO PRO PRO PRO PRO PRO NME"
TRP="ACE ASN LEU TYR ILE GLN TRP LEU LYS ASP GLY GLY PRO SER SER GLY ARG PRO PRO PRO SER NME"
CLN="TYR TYR ASP PRO GLU THR GLY THR TRP TYR"
CHI="GLY TYR ASP PRO GLU THR GLY THR TRP GLY"


shopt -s nocasematch
case "$PDB_NAME" in
  ALA*) QM_ZONE="$ALA"; CHARGE=0 ;;
  CYC*) QM_ZONE="$CYC"; CHARGE=-1 ;;
  P13*) QM_ZONE="$P13"; CHARGE=0 ;;
  TRP*) QM_ZONE="$TRP"; CHARGE=1 ;;
  CLN*) QM_ZONE="$CLN"; CHARGE=-2 ;;
  CHI*) QM_ZONE="$CHI"; CHARGE=-2 ;;

  *)
    echo "Error: '$PDB_NAME' does not match known molecule." >&2
    exit 2
    ;;
esac
shopt -u nocasematch

# =====================================
# SLURM Settings
# =====================================
N_TASKS_PER_NODE=1
CPUS_PER_TASK=1
GPUS=1
MEM_PER_CPU="8G"
MEM_PER_GPU="11264m"
TIME="4:00:00"

# =====================================
# Helpers
# =====================================
submit_yaml () {
  local yaml_path="$1"
  local phase="$2"

  sbatch \
    --job-name="${phase}_${PDB_NAME}" \
    --ntasks-per-node="${N_TASKS_PER_NODE}" \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --gpus="${GPUS}" \
    --mem-per-cpu="${MEM_PER_CPU}" \
    --time="${TIME}" \
    --output="logs/%x_%j.out" \
    --error="logs/%x_%j.err" \
    --wrap="\
      rest2-ampmm ${yaml_path}"
}

# -------------------------------------
# NVT
# -------------------------------------
run_nvt() {
  python "${INPUT_SCRIPTS_DIR}/nvt.py" \
    --molecule "$PDB_NAME" \
    --qm_zone $QM_ZONE \
    --mm_zone HOH \
    --mol_charge "$CHARGE" \
    --output nvt

  yaml_path="amp_simulation/${PDB_NAME}/nvt/config.yaml"
  submit_yaml "$yaml_path" "nvt"
}

# -------------------------------------
# NPT
# -------------------------------------
run_npt() {
  python "${INPUT_SCRIPTS_DIR}/npt.py" \
    --molecule "$PDB_NAME" \
    --qm_zone $QM_ZONE \
    --mm_zone HOH \
    --mol_charge "$CHARGE" \
    --input nvt \
    --output npt

  yaml_path="amp_simulation/${PDB_NAME}/npt/config.yaml"
  submit_yaml "$yaml_path" "npt"
}

run_production() {
  
  SIM_DIR="${PROJECT_ROOT}/amp_simulation/${PDB_NAME}"

  local n_reps="${N_REPS:-1}"
  local n_segs=2

  echo "=== cAMP/MM production (reps=${n_reps}, segs=${n_segs}) ==="

  # ---------- 1) PREP ----------
  for rep in $(seq 0 $((n_reps-1))); do
    rep_str=$(printf "%04d" "$rep")

    for seg in $(seq 0 $((n_segs-1))); do
      seg_str=$(printf "%04d" "$seg")
      outprefix="production_${rep_str}_${seg_str}"

      if (( seg == 0 )); then
        input_arg="--input npt"
        restart_vel="--restart_velocities"
      else
        prev_seg_str=$(printf "%04d" "$((seg-1))")
        input_arg="--input production_${rep_str}_${prev_seg_str}"
        restart_vel=""
      fi

      python "${INPUT_SCRIPTS_DIR}/production.py" \
        --molecule "$PDB_NAME" \
        --qm_zone $QM_ZONE \
        --mm_zone HOH \
        --mol_charge "$CHARGE" \
        $restart_vel \
        $input_arg \
        --output "$outprefix"
    done
  done

  # ---------- 2) RUN ----------
  local n=1
  local GPUs=1
  local mem_per_cpu="8G"
  local mem_per_gpu="11264m"
  local cpus_per_task=1
  local time="24:00:00"

  local sbatch_opts=()
  mkdir -p logs

  for rep in $(seq 0 $((n_reps-1))); do
    rep_str=$(printf "%04d" "$rep")
    prev_jobid=""

    for seg in $(seq 0 $((n_segs-1))); do
      seg_str=$(printf "%04d" "$seg")
      jobname="prod_${rep_str}_${seg_str}"
      outprefix="production_${rep_str}_${seg_str}"
      yaml_path="${SIM_DIR}/${outprefix}/config.yaml"

      jobfile="${jobname}.sbatch"

      cat > "$jobfile" <<EOF
#!/bin/bash
set -euo pipefail
echo "[${jobname}] starting on \$(hostname) at \$(date)"
rest2-ampmm ${yaml_path}
echo "[${jobname}] finished at \$(date)"
EOF
      chmod +x "$jobfile"

      if [[ -z "$prev_jobid" ]]; then
        jobid=$(sbatch --parsable \
                       --job-name="$jobname" \
                       --ntasks="$n" \
                       --cpus-per-task="$cpus_per_task" \
                       --gpus="$GPUs" \
                       --mem-per-cpu="$mem_per_cpu" \
                       --gres=gpumem:"$mem_per_gpu" \
                       --time="$time" \
                       --output="logs/${jobname}_%j.out" \
                       "${sbatch_opts[@]}" \
                       "$jobfile")
      else
        jobid=$(sbatch --parsable \
                       --job-name="$jobname" \
                       --dependency=afterok:$prev_jobid \
                       --ntasks="$n" \
                       --cpus-per-task="$cpus_per_task" \
                       --gpus="$GPUs" \
                       --mem-per-cpu="$mem_per_cpu" \
                       --gres=gpumem:"$mem_per_gpu" \
                       --time="$time" \
                       --output="logs/${jobname}_%j.out" \
                       "${sbatch_opts[@]}" \
                       "$jobfile")
      fi

      echo "Submitted ${jobname} (jobid ${jobid})"
      prev_jobid="$jobid"
    done
  done

  rm -f prod_*.sbatch
}

run_rest2() {

  SIM_DIR="${PROJECT_ROOT}/amp_simulation/${PDB_NAME}"

  local N_NODES=1
  local N_TASKS=8
  local n_segs=2
  local GPUs=8

  mkdir -p logs

  for rep in $(seq 0 $((N_REPS-1))); do
    rep_str=$(printf "%04d" "$rep")
    prev_jobid=""

    for seg in $(seq 0 $((n_segs-1))); do
      seg_str=$(printf "%04d" "$seg")
      outprefix="rest2_${rep_str}_${seg_str}"

      # ---------- 1) PREP ----------
      if (( seg == 0 )); then
        input_arg="--input npt"
        restart_vel="--restart_velocities"
      else
        prev_seg=$(printf "%04d" "$((seg-1))")
        input_arg="--input rest2_${rep_str}_${prev_seg}"
        restart_vel=""
      fi

      python "${INPUT_SCRIPTS_DIR}/rest2.py" \
        --molecule "$PDB_NAME" \
        --qm_zone $QM_ZONE \
        --mm_zone HOH \
        --mol_charge "$CHARGE" \
        $restart_vel \
        $input_arg \
        --output "$outprefix"

      yaml_path="${SIM_DIR}/${outprefix}/config.yaml"
      jobname="rest2_${rep_str}_${seg_str}"

      # ---------- 2) RUN ----------
      if [[ -z "$prev_jobid" ]]; then
        jobid=$(sbatch --parsable \
          --job-name="${PDB_NAME}_${jobname}" \
          -N "${N_NODES}" \
          --ntasks="${N_TASKS}" \
          --gpus-per-node="${GPUs}" \
          --gres="gpumem:${MEM_PER_GPU}" \
          --cpus-per-task="${CPUS_PER_TASK}" \
          --mem-per-cpu="${MEM_PER_CPU}" \
          --time="${TIME}" \
          --output="logs/${jobname}_%j.out" \
          --wrap="mpirun -np ${N_TASKS} rest2-ampmm ${yaml_path}")
      else
        jobid=$(sbatch --parsable \
          --job-name="${PDB_NAME}_${jobname}" \
          --dependency=afterok:$prev_jobid \
          -N "${N_NODES}" \
          --ntasks="${N_TASKS}" \
          --gpus-per-node="${GPUs}" \
          --gres="gpumem:${MEM_PER_GPU}" \
          --cpus-per-task="${CPUS_PER_TASK}" \
          --mem-per-cpu="${MEM_PER_CPU}" \
          --time="${TIME}" \
          --output="logs/${jobname}_%j.out" \
          --wrap="mpirun -np ${N_TASKS} rest2-ampmm ${yaml_path}")
      fi

      echo "Submitted ${jobname} (jobid ${jobid})"
      prev_jobid="$jobid"
    done
  done
}

# =====================================
# Dispatch
# =====================================
case "$STAGE" in
  nvt)        run_nvt ;;
  npt)        run_npt ;;
  production) run_production ;;
  rest2)      run_rest2 ;;
  *)
    echo "Error: unknown STAGE '$STAGE' (expected: nvt|npt|production|rest2)" >&2
    exit 3
    ;;
esac

