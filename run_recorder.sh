#!/usr/bin/env bash
# Simple launcher that sets env and runs recorder via Isaac's python.sh
ISAAC_ROOT="${ISAAC_ROOT:-/home/rhutvik/isaacsim}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/rhutvik/isaac_sim_uav/prop_repro_project}"
SCRIPT_PATH="${PROJECT_ROOT}/scripts/isaac_replay_recorder.py"
QUAD_USD="${QUAD_USD:-/home/rhutvik/isaac_sim_uav/scripts/quad.usd}"
OUTDIR="${OUTDIR:-${PROJECT_ROOT}/isaac_dataset}"

export EXP_PATH="${EXP_PATH:-${PROJECT_ROOT}}"
export CARB_APP_PATH="${CARB_APP_PATH:-${ISAAC_ROOT}}"
export ISAAC_PATH="${ISAAC_PATH:-${ISAAC_ROOT}}"
export OMNI_KIT_CACHE="${OMNI_KIT_CACHE:-${EXP_PATH}/.omni_kit_cache}"
mkdir -p "${OMNI_KIT_CACHE}"
echo "ENV:"
echo " EXP_PATH=${EXP_PATH}"
echo " CARB_APP_PATH=${CARB_APP_PATH}"
echo " ISAAC_PATH=${ISAAC_PATH}"
echo " OMNI_KIT_CACHE=${OMNI_KIT_CACHE}"
"${ISAAC_ROOT}/python.sh" "${SCRIPT_PATH}" \
  --usd "${QUAD_USD}" --outdir "${OUTDIR}" --run-name "run_auto_$(date +%s)" \
  --robot-prim "/World/body" --duration 20 --fps 100 --fault-type unbalance --fault-motor 1 --ur 0.18 --apply-fault
