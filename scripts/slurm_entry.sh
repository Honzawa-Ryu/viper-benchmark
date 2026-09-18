#!/bin/bash
# このファイルは run_slurm.sh の末尾から `source` される前提で書かれている
# （run_slurm.sh 自体が #SBATCH ヘッダーを持つ実際の投入スクリプトであり、
# RUN_MODE/RUN_COMMAND/GRID_ARGS 等は既にその時点で定義済みのため、ここで
# run_slurm.sh を再度 source する必要はない。再度sourceすると、run_slurm.sh
# 末尾のこのファイルへのsource呼び出し自体も再実行され、循環してしまう）。

set -euo pipefail

# =====================================================
# Bootstrap validation
# =====================================================

_bootstrap_fail() {
    local reason="$1"
    local detail="$2"

    printf 'FATAL INIT [%s]: %s\n' "${reason}" "${detail}" >&2

    if [ -n "${JOB_LOG_DIR:-}" ] && [ -d "${JOB_LOG_DIR}" ]; then
        printf '%s FATAL INIT [%s]: %s\n' "$(date --iso-8601=seconds)" "${reason}" "${detail}" \
            >> "${JOB_LOG_DIR}/bootstrap_failure.log"
        printf 'status: FAILED\nfailure_phase: bootstrap\nfail_reason: %s\n' "${reason}" \
            > "${JOB_LOG_DIR}/run_metadata.yaml"
    fi

    exit 2
}

if [ -z "${PROJECT_ROOT:-}" ]; then
    _bootstrap_fail "MISSING_PROJECT_ROOT" "PROJECT_ROOT is not set. Source slurm_entry.sh from run_slurm.sh."
fi

if [ -z "${EXP_NAME:-}" ]; then
    _bootstrap_fail "MISSING_EXP_NAME" "EXP_NAME is not set. Source slurm_entry.sh from an experiment run_slurm.sh."
fi

# .env（HF_TOKEN等）をジョブ内で読む。--export=ALL は「投入したシェルの環境」を
# 引き継ぐだけなので、.env をsourceし忘れたシェルから sbatch すると、gatedリポジトリ
# のダウンロードがジョブ実行時に401で落ちる（投入時には何のエラーも出ない）。
# ここで読めば、どのシェルから投入しても同じ環境で走る。
# 既に環境にある値は上書きせず、未設定のものだけを .env から補う。
if [ -f "${PROJECT_ROOT}/.env" ]; then
    # `|| [ -n "${_env_line}" ]` は .env の最終行に改行が無い場合の取りこぼし対策。
    while IFS= read -r _env_line || [ -n "${_env_line}" ]; do
        _env_line="${_env_line#export }"
        case "${_env_line}" in
            ''|'#'*) continue ;;   # 空行・コメント
            *=*) ;;                # KEY=VALUE のみ通す
            *) continue ;;
        esac
        _env_key="${_env_line%%=*}"
        _env_val="${_env_line#*=}"
        # 値を囲むクォートがあれば剥がす
        case "${_env_val}" in
            \"*\") _env_val="${_env_val#\"}"; _env_val="${_env_val%\"}" ;;
            \'*\') _env_val="${_env_val#\'}"; _env_val="${_env_val%\'}" ;;
        esac
        if [ -z "$(eval "printf '%s' \"\${${_env_key}:-}\"")" ]; then
            export "${_env_key}=${_env_val}"
        fi
    done < "${PROJECT_ROOT}/.env"
    unset _env_line _env_key _env_val
fi

# =====================================================
# Scheduler abstraction
# =====================================================
if [ -n "${PBS_JOBID:-}" ]; then
    export SCHEDULER="pbs"
    export JOB_ID=$(echo "${PBS_JOBID}" | cut -d. -f1)
    
    if [ -n "${PBS_ARRAY_INDEX:-}" ]; then
        export ARRAY_JOB_ID=$(echo "${PBS_JOBID}" | cut -d. -f1 | cut -d'[' -f1)
        export ARRAY_TASK_ID="${PBS_ARRAY_INDEX}"
    else
        export ARRAY_JOB_ID=""
        export ARRAY_TASK_ID=""
    fi
    export JOB_PARTITION="${PBS_QUEUE:-unknown}"
    export NODENAME=$(hostname)
elif [ -n "${SLURM_JOB_ID:-}" ]; then
    export SCHEDULER="slurm"
    export JOB_ID="${SLURM_JOB_ID}"
    export ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-}"
    export ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-}"
    export JOB_PARTITION="${SLURM_JOB_PARTITION:-unknown}"
    export NODENAME="${SLURMD_NODENAME:-unknown}"
else
    export SCHEDULER="local"
    export JOB_ID="$$"
    export ARRAY_JOB_ID=""
    export ARRAY_TASK_ID=""
    export JOB_PARTITION="local"
    export NODENAME=$(hostname)
fi

# =====================================================
# Run mode
#   single : RUN_COMMAND を1つ実行（デフォルト）
#   array  : SLURM_ARRAY_TASK_ID で CONFIGS を引く
#   seq    : CONFIGS をループして順番に実行
# =====================================================

RUN_MODE="${RUN_MODE:-single}"

case "${RUN_MODE}" in
    single|array|seq)
        ;;
    *)
        _bootstrap_fail "INVALID_RUN_MODE" "RUN_MODE must be one of single, array, or seq; got '${RUN_MODE}'."
        ;;
esac

# array の場合はジョブIDにarray_idを付与
if [ "${RUN_MODE}" = "array" ]; then
    export JOB_LOG_DIR="${PROJECT_ROOT}/logs/${EXP_NAME}/${ARRAY_JOB_ID}_${ARRAY_TASK_ID}"
else
    export JOB_LOG_DIR="${PROJECT_ROOT}/logs/${EXP_NAME}/${JOB_ID}"
fi

if ! mkdir -p "${JOB_LOG_DIR}"; then
    _bootstrap_fail "LOG_DIRECTORY_CREATION_FAILED" "Could not create ${JOB_LOG_DIR}."
fi

if ! ln -sfn "${JOB_LOG_DIR}" "${PROJECT_ROOT}/logs/${EXP_NAME}/latest"; then
    _bootstrap_fail "LATEST_LOG_LINK_FAILED" "Could not update logs/${EXP_NAME}/latest."
fi

export SLURM_LOG_FILE="${JOB_LOG_DIR}/slurm.out"

if ! source "${PROJECT_ROOT}/scripts/notify_slack.sh"; then
    _bootstrap_fail "NOTIFICATION_HELPER_LOAD_FAILED" "Could not source scripts/notify_slack.sh."
fi

# run_slurm.sh の #SBATCH --time= を直接読む（run_slurm.sh は #SBATCH ヘッダーに
# しか --time を持たず、DEFAULT_TIME のようなbash変数はもう存在しないため）。
if ! JOB_TIME_LIMIT=$(
    grep -oP '(?<=^#SBATCH --time=)\S+' "${PROJECT_ROOT}/experiments/${EXP_NAME}/run_slurm.sh" \
        | head -n1
); then
    _bootstrap_fail "MISSING_TIME_LIMIT" "Could not read a #SBATCH --time directive from experiments/${EXP_NAME}/run_slurm.sh."
fi

if [ -z "${JOB_TIME_LIMIT}" ]; then
    _bootstrap_fail "MISSING_TIME_LIMIT" "The #SBATCH --time directive in experiments/${EXP_NAME}/run_slurm.sh is empty."
fi

export JOB_TIME_LIMIT

# =====================================================
# Metadata helper
# =====================================================

update_metadata() {
    local key="$1"
    local value="$2"
    local yaml_file="${JOB_LOG_DIR}/run_metadata.yaml"

    if ! python3 - "${yaml_file}" "${key}" "${value}" <<'PY'
from pathlib import Path
import sys
import yaml

path = Path(sys.argv[1])
data = yaml.safe_load(path.read_text())
if not isinstance(data, dict):
    raise ValueError(f"Metadata file is not a mapping: {path}")
data[sys.argv[2]] = sys.argv[3]
path.write_text(yaml.safe_dump(data, sort_keys=False))
PY
    then
        printf 'FATAL FINALIZATION: could not record %s=%s in run_metadata.yaml\n' \
            "${key}" "${value}" >&2
        printf '%s FATAL FINALIZATION: could not record %s=%s in run_metadata.yaml\n' \
            "$(date --iso-8601=seconds)" "${key}" "${value}" \
            >> "${JOB_LOG_DIR}/finalization_failure.log"
    fi
    return 0
}

# =====================================================
# Metadata
# =====================================================

GIT_COMMIT=$(git -C "${PROJECT_ROOT}" rev-parse HEAD 2>/dev/null || printf 'unknown')

if ! python3 - \
    "${JOB_LOG_DIR}/run_metadata.yaml" \
    "${EXP_NAME}" \
    "${JOB_ID}" \
    "${JOB_PARTITION}" \
    "${NODENAME}" \
    "${GIT_COMMIT}" \
    "${JOB_TIME_LIMIT}" \
    "${RUN_MODE}" \
    "${ARRAY_TASK_ID}" <<'PY'
from pathlib import Path
import sys
import yaml

data = {
    "exp_name":   sys.argv[2],
    "job_id":     sys.argv[3],
    "partition":  sys.argv[4],
    "node":       sys.argv[5],
    "git_commit": sys.argv[6],
    "time_limit": "unknown" if not sys.argv[7] else sys.argv[7],
    "run_mode":   sys.argv[8],
    "status":     "RUNNING",
}

if sys.argv[8] == "array":
    data["array_task_id"] = sys.argv[9]

Path(sys.argv[1]).write_text(
    yaml.safe_dump(data, sort_keys=False)
)
PY
then
    _bootstrap_fail "METADATA_INITIALIZATION_FAILED" "Could not create ${JOB_LOG_DIR}/run_metadata.yaml."
fi

# =====================================================
# Move Slurm log helper
# =====================================================

_move_slurm_log() {
    local log_file=""

    if [ "${SCHEDULER}" = "pbs" ]; then
        local pattern=""
        if [ "${RUN_MODE}" = "array" ]; then
            pattern="${PROJECT_ROOT}/logs/${EXP_NAME}/${EXP_NAME}.o${ARRAY_JOB_ID}-${ARRAY_TASK_ID}"
        else
            pattern="${PROJECT_ROOT}/logs/${EXP_NAME}/${EXP_NAME}.o${JOB_ID}"
        fi
        
        log_file=$(ls ${pattern}* 2>/dev/null | head -n1 || true)
        
        if [ -n "${log_file}" ] && [ -f "${log_file}" ]; then
            mv "${log_file}" "${JOB_LOG_DIR}/slurm.out"
        fi
        
        local err_pattern=""
        if [ "${RUN_MODE}" = "array" ]; then
            err_pattern="${PROJECT_ROOT}/logs/${EXP_NAME}/${EXP_NAME}.e${ARRAY_JOB_ID}-${ARRAY_TASK_ID}"
        else
            err_pattern="${PROJECT_ROOT}/logs/${EXP_NAME}/${EXP_NAME}.e${JOB_ID}"
        fi
        local err_log_file
        err_log_file=$(ls ${err_pattern}* 2>/dev/null | head -n1 || true)
        if [ -n "${err_log_file}" ] && [ -f "${err_log_file}" ]; then
            mv "${err_log_file}" "${JOB_LOG_DIR}/pbs.err"
        fi
    else
        if [ "${RUN_MODE}" = "array" ]; then
            # runx が --output=%A_%a_${exp_name}.out で投入するため、
            # %A（親のarray job id = SLURM_ARRAY_JOB_ID）を使う。
            # SLURM_JOB_ID は各タスク固有の値で %A とは異なるため使わない。
            log_file="${PROJECT_ROOT}/logs/${EXP_NAME}/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${EXP_NAME}.out"
        else
            log_file="${PROJECT_ROOT}/logs/${EXP_NAME}/${SLURM_JOB_ID}_${EXP_NAME}.out"
        fi

        if [ -f "${log_file}" ]; then
            mv "${log_file}" "${JOB_LOG_DIR}/slurm.out"
        fi
    fi
}

# =====================================================
# sacct helper
# =====================================================

_get_job_state() {
    if [ "${SCHEDULER}" = "local" ]; then
        echo "COMPLETED"
        return 0
    fi

    if [ "${SCHEDULER}" = "pbs" ]; then
        if command -v qstat &>/dev/null; then
            local target_id="${JOB_ID}"
            if [ "${RUN_MODE}" = "array" ]; then
                target_id="${ARRAY_JOB_ID}[${ARRAY_TASK_ID}]"
            fi
            
            local qstat_out
            qstat_out=$(qstat -f "${target_id}" 2>/dev/null || true)
            
            if [ -z "${qstat_out}" ]; then
                echo "COMPLETED"
                return
            fi
            
            local comment
            comment=$(echo "${qstat_out}" | grep -E "comment =" || true)
            
            if echo "${comment}" | grep -iq "walltime"; then
                echo "TIMEOUT"
                return
            fi
            if echo "${comment}" | grep -iq -E "oom|out of memory"; then
                echo "OUT_OF_MEMORY"
                return
            fi
            
            local exit_status
            exit_status=$(echo "${qstat_out}" | grep -E "Exit_status =" | awk '{print $3}' || echo "")
            if [ "${exit_status}" = "271" ]; then
                echo "TIMEOUT"
            elif [ "${exit_status}" = "137" ]; then
                echo "OUT_OF_MEMORY"
            elif [ "${exit_status}" = "0" ] || [ -z "${exit_status}" ]; then
                echo "COMPLETED"
            else
                echo "FAILED"
            fi
        else
            echo "UNKNOWN"
        fi
    else
        if command -v sacct &>/dev/null; then
            sacct \
                -j "${SLURM_JOB_ID}" \
                --format=JobIDRaw,State \
                --parsable2 \
                --noheader \
            | awk -F'|' -v id="${SLURM_JOB_ID}" '$1==id {print $2; exit}'
        elif command -v scontrol &>/dev/null; then
            scontrol show job "${SLURM_JOB_ID}" 2>/dev/null \
                | grep -oP 'JobState=\K\w+' || echo "UNKNOWN"
        else
            echo "UNKNOWN"
        fi
    fi
}

_handle_final_state() {
    trap - ERR
    local exit_code="${1:-0}"
    local line_no="${2:-unknown}"

    sleep 3

    local job_state
    job_state=$(_get_job_state)

    case "${job_state}" in

        TIMEOUT)
            update_metadata "status" "TIMEOUT"
            update_metadata "fail_reason" "SLURM_TIMEOUT"
            notify_fail "TIMEOUT"
            ;;

        OUT_OF_MEMORY)
            update_metadata "status" "OUT_OF_MEMORY"
            update_metadata "fail_reason" "OUT_OF_MEMORY"
            notify_fail "OUT_OF_MEMORY"
            ;;

        CANCELLED*)
            update_metadata "status" "CANCELLED"
            update_metadata "fail_reason" "CANCELLED"
            notify_fail_fast "CANCELLED"
            ;;

        NODE_FAIL)
            update_metadata "status" "NODE_FAIL"
            update_metadata "fail_reason" "NODE_FAIL"
            notify_fail "NODE_FAIL"
            ;;

        COMPLETED)
            if [ "${exit_code}" -eq 0 ]; then
                update_metadata "status" "COMPLETED"
                notify_finish
            else
                update_metadata "status" "FAILED"
                update_metadata "fail_reason" "NONZERO_EXIT_${exit_code}"
                notify_fail "NONZERO_EXIT (code ${exit_code})"
            fi
            ;;

        RUNNING|COMPLETING|CONFIGURING)
            if [ "${exit_code}" -eq 0 ]; then

                update_metadata "status" "COMPLETED"
                notify_finish

            else

                update_metadata "status" "FAILED"
                update_metadata "fail_reason" "NONZERO_EXIT_${exit_code}"

                notify_fail "NONZERO_EXIT (code ${exit_code})"

            fi
            ;;

        UNKNOWN|"")
            if [ "${exit_code}" -eq 0 ]; then
                update_metadata "status" "COMPLETED"
                update_metadata "scheduler_state" "UNKNOWN"
                notify_finish
            else
                update_metadata "status" "FAILED"
                update_metadata "scheduler_state" "UNKNOWN"
                update_metadata "fail_reason" "NONZERO_EXIT_${exit_code}_SCHEDULER_STATE_UNKNOWN"
                notify_fail "NONZERO_EXIT (code ${exit_code}; scheduler state unknown)"
            fi
            ;;

        *)
            update_metadata "status" "FAILED"
            update_metadata "fail_reason" "SCRIPT_ERROR_LINE_${line_no}_CODE_${exit_code}"
            notify_fail "SCRIPT_ERROR (line ${line_no}, code ${exit_code}, state ${job_state})"
            ;;

    esac

    _move_slurm_log
}

# =====================================================
# Signal handlers
# =====================================================

on_error() {
    trap - ERR
    local exit_code=$?
    local line_no="${1:-unknown}"

    echo ""
    echo "❌ Error on line ${line_no} (exit code: ${exit_code})"
    echo ""

    _handle_final_state "${exit_code}" "${line_no}"
}

trap 'on_error ${LINENO}' ERR

TERMINATED=0

on_terminate() {
    trap - TERM INT
    TERMINATED=1

    echo "TERM/INT RECEIVED $(date)" \
        >> "${JOB_LOG_DIR}/signal_debug.log"

    _handle_final_state 1 "signal"
}

trap on_terminate TERM INT

# =====================================================
# Time limit warning (旧 watch_job.sh の代替)
# --signal=B:USR1@<margin> により、Slurm が残り時間僅少になった時点で
# このバッチスクリプト自身に USR1 を送る。別プロセスのwatcherは使わない。
# ジョブ自体は継続する（通知のみ、_handle_final_state は呼ばない）。
# =====================================================

on_time_limit_warning() {
    echo "TIME_LIMIT_WARNING RECEIVED $(date)" \
        >> "${JOB_LOG_DIR}/signal_debug.log"
    notify_fail_fast "TIME_LIMIT_WARNING"
}

trap on_time_limit_warning USR1

# =====================================================
# GRID → CONFIGS 展開
# =====================================================

if [ "${RUN_MODE}" != "single" ] \
   && [[ -v GRID_ARGS && "${#GRID_ARGS[@]}" -gt 0 ]]; then

    # GRID_ARGS / GRID_VALUES から直積を生成して CONFIGS に展開
    _expand_grid() {
        local -n _args=$1
        local -n _values=$2
        local -a result=("")

        for i in "${!_args[@]}"; do
            local arg="${_args[$i]}"
            local -a vals
            read -ra vals <<< "${_values[$i]}"

            local -a next=()

            for prev in "${result[@]}"; do
                for val in "${vals[@]}"; do
                    if [ -z "${prev}" ]; then
                        next+=("${arg} ${val}")
                    else
                        next+=("${prev} ${arg} ${val}")
                    fi
                done
            done

            result=("${next[@]}")
        done

        printf '%s\n' "${result[@]}"
    }

    mapfile -t CONFIGS < <(_expand_grid GRID_ARGS GRID_VALUES)
fi

# =====================================================
# Resolve run command
# =====================================================

if [ "${RUN_MODE}" = "array" ]; then

    # CONFIGS[ARRAY_TASK_ID] を引く
    RESOLVED_COMMAND="${BASE_COMMAND} ${CONFIGS[${ARRAY_TASK_ID}]}"

elif [ "${RUN_MODE}" = "seq" ]; then

    # seqの場合はループで実行するのでここではダミー
    RESOLVED_COMMAND=""

else

    RESOLVED_COMMAND="${RUN_COMMAND}"

fi

# =====================================================
# Array info for notifications
# =====================================================

if [ "${RUN_MODE}" = "array" ]; then
    ARRAY_TOTAL_TASKS="${#CONFIGS[@]}"
    ARRAY_TASK_OPTIONS="${CONFIGS[${ARRAY_TASK_ID}]}"
fi

# =====================================================
# Start notify
# =====================================================

notify_start

# =====================================================
# Scratch
# =====================================================

# 一部クラスタ（ノード）には /scratch 自体が存在しない/書き込み不可のため、
# USE_LOCAL_SSD_INPUT/OUTPUT のどちらも使わない場合は作成もbindも行わない。
if [ "${USE_LOCAL_SSD_INPUT:-0}" -eq 1 ] || [ "${USE_LOCAL_SSD_OUTPUT:-0}" -eq 1 ]; then
    export SCRATCH_DIR="/scratch/${USER}/${EXP_NAME}_${JOB_ID}"
    mkdir -p "${SCRATCH_DIR}"
else
    export SCRATCH_DIR=""
fi

# =====================================================
# Input
# =====================================================

if [ "${USE_LOCAL_SSD_INPUT:-0}" -eq 1 ]; then

    mkdir -p "${SCRATCH_DIR}/data"

    rsync -a \
        "${PROJECT_ROOT}/data/" \
        "${SCRATCH_DIR}/data/"

    export DATASET_DIR="${SCRATCH_DIR}/data"

else

    export DATASET_DIR="${PROJECT_ROOT}/data"
fi

# =====================================================
# Output
# =====================================================

export OUTPUT_DIR="${PROJECT_ROOT}/outputs/${EXP_NAME}"

mkdir -p "${OUTPUT_DIR}"

if [ "${USE_LOCAL_SSD_OUTPUT:-0}" -eq 1 ]; then

    mkdir -p "${SCRATCH_DIR}/outputs"

    # lib/output_utils.get_run_dir() writes here when set; the
    # completed-guard still always checks OUTPUT_DIR (project_root),
    # never this scratch path.
    export OUTPUT_ROOT="${SCRATCH_DIR}/outputs"
fi

# =====================================================
# Save command
# =====================================================

if [ "${RUN_MODE}" = "seq" ]; then
    printf '%s\n' "${CONFIGS[@]/#/${BASE_COMMAND} }" \
        > "${JOB_LOG_DIR}/command.sh"
else
    echo "${RESOLVED_COMMAND}" \
        > "${JOB_LOG_DIR}/command.sh"
fi

# =====================================================
# Run
# =====================================================

_run_single() {
    local cmd="$1"

    set +e

    if command -v apptainer &>/dev/null && [ -n "${SIF_PATH:-}" ] && [ -f "${SIF_PATH}" ]; then
        local -a apptainer_scratch_opts=()
        if [ -n "${SCRATCH_DIR:-}" ]; then
            apptainer_scratch_opts=(
                --bind "/scratch/${USER}"
                --bind "${SCRATCH_DIR}"
                --env "UV_CACHE_DIR=${SCRATCH_DIR}/.uv_cache"
            )
        fi

        apptainer exec \
            --nv \
            "${apptainer_scratch_opts[@]}" \
            "${SIF_PATH}" \
            bash -c "
                set -euo pipefail

                source ${PROJECT_ROOT}/.venv/bin/activate
                export CUDA_HOME=/usr/local/cuda

                cd ${PROJECT_ROOT}

                ${cmd}
            " 1>&2 &
    else
        echo "⚠️ Apptainer not found or SIF_PATH not set. Running command on host system." >&2
        bash -c "
            set -euo pipefail
            if [ -f ${PROJECT_ROOT}/.venv/bin/activate ]; then
                source ${PROJECT_ROOT}/.venv/bin/activate
            fi
            cd ${PROJECT_ROOT}
            ${cmd}
        " 1>&2 &
    fi

    local pid=$!
    local code=0

    wait ${pid} || code=$?

    set -e

    echo "${code}"
}

# =====================================================
# Pre-native command (apptainer外で実行したい処理)
# ES 起動など、ネイティブ bash レイヤーで動かすコマンドを
# run_slurm.sh で PRE_NATIVE_COMMAND に設定する
# =====================================================

if [ -n "${PRE_NATIVE_COMMAND:-}" ]; then
    echo "=== PRE_NATIVE_COMMAND ==="
    eval "${PRE_NATIVE_COMMAND}"
fi

EXIT_CODE=0

if [ "${RUN_MODE}" = "seq" ]; then

    # CONFIGS をループして順番に実行
    for i in "${!CONFIGS[@]}"; do

        local_cmd="${BASE_COMMAND} ${CONFIGS[$i]}"

        echo ""
        echo "▶ [${i}/$((${#CONFIGS[@]} - 1))] ${local_cmd}"
        echo ""

        code=$(_run_single "${local_cmd}")

        if [ "${code}" -ne 0 ]; then
            echo "❌ Config [${i}] failed with exit code: ${code}"
            EXIT_CODE="${code}"
            break
        fi
    done

else

    EXIT_CODE=$(_run_single "${RESOLVED_COMMAND}")

fi

if [ "${EXIT_CODE}" -ne 0 ]; then
    echo "Child failed with exit code: ${EXIT_CODE}"
fi

# =====================================================
# Post-native command (apptainer外で実行したい後処理)
# ES データの SSD→HDD 同期など
# =====================================================

if [ -n "${POST_NATIVE_COMMAND:-}" ]; then
    echo "=== POST_NATIVE_COMMAND ==="
    eval "${POST_NATIVE_COMMAND}"
fi

# =====================================================
# Output sync-back（scratch → /workspace/outputs）
# USE_LOCAL_SSD_OUTPUT=1 の場合、experiment.py は completion.json を含む
# すべての出力を SCRATCH_DIR/outputs/ 配下に書いている。次回以降の
# completed ガードが正しく機能するよう、scratch を消す前に必ず
# OUTPUT_DIR（project_root 側）へ回収する。
# =====================================================

if [ "${USE_LOCAL_SSD_OUTPUT:-0}" -eq 1 ] \
   && [ -d "${SCRATCH_DIR}/outputs/${EXP_NAME}" ]; then

    rsync -a \
        "${SCRATCH_DIR}/outputs/${EXP_NAME}/" \
        "${OUTPUT_DIR}/"
fi

# =====================================================
# Scratch cleanup
# SCRATCH_DIR はジョブごとに固有のディレクトリなので、削除せずに残すと
# ジョブを回すたびに /scratch の使用量が際限なく積み上がっていく。
# ここに到達するのは正常終了パスのみ（Output sync-back が必要な場合は
# 既にrsync済み）なので、パスが /scratch/${USER}/ 配下であることを厳密に
# 確認した上で自動削除する。script エラーやシグナル受信（on_error/
# on_terminate 経由）で異常終了した場合はこのブロックを通らないため、
# デバッグ・手動回収のため scratch はそのまま残る。
# =====================================================

if [ -n "${SCRATCH_DIR:-}" ] \
   && [ -d "${SCRATCH_DIR}" ] \
   && [[ "${SCRATCH_DIR}" == "/scratch/${USER}/"* ]] \
   && [ "${SCRATCH_DIR}" != "/scratch/${USER}" ] \
   && [ "${SCRATCH_DIR}" != "/scratch/${USER}/" ]; then

    rm -rf "${SCRATCH_DIR}"
    echo "🧹 Removed scratch directory: ${SCRATCH_DIR}"
fi

# =====================================================
# Skip final status if already handled by signal
# =====================================================

if [ "${TERMINATED}" -eq 1 ]; then
    exit 1
fi

# =====================================================
# Final status
# =====================================================

_handle_final_state "${EXIT_CODE}" "end"
