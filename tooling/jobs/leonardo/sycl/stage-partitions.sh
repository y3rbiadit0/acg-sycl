#!/bin/bash
#
# Make the METIS partition files the SYCL campaign needs.
#
# Same method, and the same files, as cluster/leonardo/jobs/stage-partitions.sh
# in the native aCG repo. The campaign must partition ahead of time rather than
# letting the solver run METIS per run: a decomposition computed per run could
# differ between backends, folding partitioning noise into a measurement meant
# to isolate the collective.
#
# It matters that these are the *same* files the native campaign uses, not
# merely files made the same way. Comparing the SYCL numbers against the native
# ones only means something if both solved the identical decomposition, so this
# script does not carry its own partitioner: it takes the paper's artifacts, or
# drives the native repo's mtxpartition, which is the tool that produced them.
#
# Usage:
#   tooling/jobs/leonardo/sycl/stage-partitions.sh [matrix ...]   # default: both
#
# Variables:
#   ACG_ARTIFACTS_DIR   acg-artifacts-v4, or its partitions/ subdirectory
#   ACG_PARTITION_DIR   destination (default $HOME/datasets/suitesparse/partitions)
#   ACG_MATRIX_DIR      where <matrix>.mtx lives, for the mtxpartition path
#   ACG_NATIVE_ROOT     the native aCG checkout, to find or build mtxpartition
#   ACG_MTXPARTITION    path to an mtxpartition binary, skipping the search
#   ACG_PARTS           part counts (default "02 04 08 16 32")

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${ACG_PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)}

source "${SCRIPT_DIR}/matrix-lib.sh"

DEST=${ACG_PARTITION_DIR:-$HOME/datasets/suitesparse/partitions}
MATRIX_DIR=${ACG_MATRIX_DIR:-${PROJECT_ROOT}/data/matrices}
# One part per MPI rank, for the scales the campaign runs: 1n2g through 8n4g.
# A 1n1g run is single-process and needs no partition.
PARTS=${ACG_PARTS:-"02 04 08 16 32"}
MATRICES=("$@")
[ ${#MATRICES[@]} -gt 0 ] || MATRICES=(Bump_2911 Queen_4147)

# The native checkout, which supplies the partitioner. A path that was handed to
# us and does not exist is a typo, not a reason to fall through to a vague
# "nothing available" at the end.
NATIVE_ROOT=""
if [ -n "${ACG_NATIVE_ROOT:-}" ]; then
    if [ ! -d "${ACG_NATIVE_ROOT}" ]; then
        printf 'ACG_NATIVE_ROOT=%s is not a directory\n' "${ACG_NATIVE_ROOT}" >&2
        exit 1
    fi
    NATIVE_ROOT=$(cd -- "${ACG_NATIVE_ROOT}" && pwd)
else
    for cand in "${PROJECT_ROOT}/../aCG-native" "${PROJECT_ROOT}/../../aCG-native" \
                "$HOME/Projects/aCG-native" "$HOME/Projects/university/aCG-native"; do
        [ -d "${cand}" ] || continue
        NATIVE_ROOT=$(cd -- "${cand}" && pwd)
        break
    done
fi
if [ -n "${NATIVE_ROOT}" ] && [ ! -e "${NATIVE_ROOT}/mtxpartition/mtxpartition.c" ]; then
    printf 'warning: %s has no mtxpartition/mtxpartition.c; is it the aCG-native checkout?\n' \
        "${NATIVE_ROOT}" >&2
fi

# Accept either <artifacts> or <artifacts>/partitions.
SRC=""
for cand in "${ACG_ARTIFACTS_DIR:-}" "${ACG_ARTIFACTS_DIR:-}/partitions" \
            "${NATIVE_ROOT:-}/acg-artifacts-v4/partitions" "${NATIVE_ROOT:-}/acg-artifacts-v4"; do
    [ -n "${cand}" ] || continue
    if compgen -G "${cand}/*_parts.mtx*" >/dev/null 2>&1; then SRC="${cand}"; break; fi
done

mkdir -p "${DEST}"

# A partition file the campaign trusts must have one entry per row, all in
# 1..nparts, and must actually use every part. The solver rejects a bad file too
# (build_file_partition in src/matrix/distributed_partition.cpp), but catching it
# here costs nothing and does not waste an allocation to find out.
verify_partition() {
    local out=$1 want_parts=$2
    awk -v want="${want_parts}" -v f="${out}" '
        /^%/ { next }
        /^[[:space:]]*$/ { next }
        !declared { declared = $1 + 0; next }
        { n++; if ($1 < min || min == 0) min = $1; if ($1 > max) max = $1; seen[$1] = 1 }
        END {
            if (declared == 0)  { printf "%s: no row count in header\n", f > "/dev/stderr"; exit 1 }
            if (n != declared)  { printf "%s: %d entries, header declares %d\n", f, n, declared > "/dev/stderr"; exit 1 }
            if (min < 1 || max > want) { printf "%s: parts range %d..%d, expected 1..%d\n", f, min, max, want > "/dev/stderr"; exit 1 }
            used = 0; for (k in seen) used++
            if (used != want) { printf "%s: uses %d of %d parts\n", f, used, want > "/dev/stderr"; exit 1 }
            printf "    %d rows, %d parts\n", n, used
        }' "${out}"
}

stage_from_artifacts() {
    local m=$1 n=$2 out=$3
    local gz="${SRC}/${m}_${n}_parts.mtx.gz"
    local plain="${SRC}/${m}_${n}_parts.mtx"
    if [ -e "${gz}" ]; then gunzip -c "${gz}" > "${out}"; return 0; fi
    if [ -e "${plain}" ]; then cp "${plain}" "${out}"; return 0; fi
    return 1
}

generate_with_mtxpartition() {
    local m=$1 n=$2 out=$3
    local mtx
    mtx=$(acg_find_matrix "${m}") || { acg_matrix_help "${m}"; return 1; }
    [ -n "${MTXPARTITION}" ] || return 1
    printf '  %s -> %s parts (from %s)\n' "${m}" "$((10#${n}))" "${mtx}"
    # Strip leading zeros: 08 -> 8. --seed 0 is what the paper's files were made
    # with, so a regenerated file matches one staged from the artifacts.
    "${MTXPARTITION}" "${mtx}" --parts="$((10#${n}))" --seed 0 > "${out}.tmp" \
        && mv "${out}.tmp" "${out}"
}

MTXPARTITION=${ACG_MTXPARTITION:-}
if [ -z "${MTXPARTITION}" ] && [ -n "${NATIVE_ROOT}" ]; then
    for cand in "${NATIVE_ROOT}/build-mtxpartition/mtxpartition" \
                "${NATIVE_ROOT}/build-oshmpi/mtxpartition" \
                "$(command -v mtxpartition 2>/dev/null || true)"; do
        [ -n "${cand}" ] && [ -x "${cand}" ] && { MTXPARTITION="${cand}"; break; }
    done
fi

# No artifacts and no partitioner yet: build the native repo's, which is the tool
# that made the paper's files. This repo deliberately has no partitioner of its
# own -- a second implementation could drift from the one the numbers came from.
if [ -z "${SRC}" ] && [ -z "${MTXPARTITION}" ]; then
    if [ -n "${NATIVE_ROOT}" ] && [ -x "${NATIVE_ROOT}/cluster/leonardo/build/build-mtxpartition.sh" ]; then
        printf 'no paper artifacts found; building mtxpartition in %s\n' "${NATIVE_ROOT}"
        "${NATIVE_ROOT}/cluster/leonardo/build/build-mtxpartition.sh"
        MTXPARTITION="${NATIVE_ROOT}/build-mtxpartition/mtxpartition"
    fi
fi

if [ -n "${SRC}" ]; then
    printf 'source: paper artifacts at %s\n' "${SRC}"
elif [ -x "${MTXPARTITION}" ]; then
    printf 'source: %s\n' "${MTXPARTITION}"
else
    # Say what was actually looked for. A bare "nothing available" sends you
    # hunting for a missing dataset when the real answer is usually one wrong path.
    printf 'no partition source available. Searched:\n' >&2
    printf '  artifacts:\n' >&2
    for cand in "${ACG_ARTIFACTS_DIR:-(ACG_ARTIFACTS_DIR unset)}" \
                "${NATIVE_ROOT:-(no native checkout)}/acg-artifacts-v4/partitions"; do
        printf '    %s\n' "${cand}" >&2
    done
    printf '  mtxpartition binary:\n' >&2
    if [ -n "${NATIVE_ROOT}" ]; then
        printf '    %s/build-mtxpartition/mtxpartition\n' "${NATIVE_ROOT}" >&2
        printf '    %s/build-oshmpi/mtxpartition\n' "${NATIVE_ROOT}" >&2
        printf '  and could not build it: %s/cluster/leonardo/build/build-mtxpartition.sh %s\n' \
            "${NATIVE_ROOT}" \
            "$([ -x "${NATIVE_ROOT}/cluster/leonardo/build/build-mtxpartition.sh" ] && echo 'failed' || echo 'is missing or not executable')" >&2
    else
        printf '    (no native checkout found; set ACG_NATIVE_ROOT)\n' >&2
    fi
    printf '\nSet one of ACG_NATIVE_ROOT, ACG_ARTIFACTS_DIR or ACG_MTXPARTITION.\n' >&2
    exit 1
fi
printf 'destination: %s\n' "${DEST}"

status=0
for m in "${MATRICES[@]}"; do
    for n in ${PARTS}; do
        out="${DEST}/${m}_${n}_parts.mtx"
        if [ -e "${out}" ]; then
            printf '  %s_%s_parts.mtx exists\n' "${m}" "${n}"
        elif [ -n "${SRC}" ] && stage_from_artifacts "${m}" "${n}" "${out}"; then
            printf '  %s_%s_parts.mtx from artifacts\n' "${m}" "${n}"
        elif generate_with_mtxpartition "${m}" "${n}" "${out}"; then
            :
        else
            printf '  %s_%s_parts.mtx UNAVAILABLE\n' "${m}" "${n}" >&2
            rm -f "${out}.tmp"
            status=1
            continue
        fi
        verify_partition "${out}" "$((10#${n}))" || { rm -f "${out}"; status=1; }
    done
done

exit "${status}"
