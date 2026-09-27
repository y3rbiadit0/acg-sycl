#!/bin/bash
#
# Resolve a SuiteSparse matrix to a path.
#
# The campaign refers to a matrix by name (Bump_2911), not by path, so that the
# results tree, the file stem and the srun argument can never disagree about
# which matrix a run used.
#
# Layouts accepted, in order:
#   $ACG_MATRIX                        explicit full path, wins outright
#   $MATRIX_DIR/<name>/<name>.mtx      what this repo ships under data/matrices
#   $MATRIX_DIR/<name>.mtx             flat, what the paper's scripts use
#   $HOME/datasets/suitesparse/mtx/<name>{,/<name>}.mtx

# The candidate list, in one place, so that a failure can report exactly what was
# looked for. Printing a shorter list than the search actually used sends you
# hunting in a directory that was never the problem.
acg_matrix_candidates() {
    local name=$1
    local dir=${MATRIX_DIR:-${ACG_PROJECT_ROOT:-$PWD}/data/matrices}
    local root=${ACG_PROJECT_ROOT:-$PWD}
    # Same set, and the same order, as acg_find_matrix in the native aCG repo:
    # a copy that one campaign can find should not be invisible to the other.
    # dataset/ is the native repo's in-tree location; data/matrices is this
    # repo's; $HOME/datasets/suitesparse/mtx is where both stage downloads.
    printf '%s\n' \
        "${dir}/${name}/${name}.mtx" \
        "${dir}/${name}.mtx" \
        "${root}/dataset/${name}/${name}.mtx" \
        "${root}/dataset/${name}.mtx" \
        "$HOME/datasets/suitesparse/mtx/${name}.mtx" \
        "$HOME/datasets/suitesparse/mtx/${name}/${name}.mtx"
}

acg_find_matrix() {
    local name=$1
    local c

    # An explicit path wins outright, but it still has to exist. Handing back a
    # path nobody checked means a typo -- or an unsubstituted placeholder --
    # only surfaces inside the solver, once every rank has already started.
    if [ -n "${ACG_MATRIX:-}" ]; then
        if [ ! -e "${ACG_MATRIX}" ]; then
            return 1
        fi
        printf '%s' "${ACG_MATRIX}"
        return 0
    fi
    while IFS= read -r c; do
        if [ -e "${c}" ]; then printf '%s' "${c}"; return 0; fi
    done <<EOF
$(acg_matrix_candidates "${name}")
EOF
    return 1
}

# Print where a matrix could be obtained. SuiteSparse groups both of the
# matrices used here under GHS_psdef.
acg_matrix_help() {
    local name=$1
    local dir=${MATRIX_DIR:-${ACG_PROJECT_ROOT:-$PWD}/data/matrices}
    if [ -n "${ACG_MATRIX:-}" ]; then
        printf 'no matrix file for %s: ACG_MATRIX=%s does not exist\n' \
            "${name}" "${ACG_MATRIX}" >&2
        printf '\nunset ACG_MATRIX to search the usual locations instead\n' >&2
        return 0
    fi
    printf 'no matrix file found for %s\n' "${name}" >&2
    printf 'searched:\n' >&2
    acg_matrix_candidates "${name}" | sed 's/^/  /' >&2
    printf '\npoint ACG_MATRIX_DIR at an existing copy, or pass ACG_MATRIX=/path/to.mtx\n' >&2
    printf '\nor download it (SuiteSparse group GHS_psdef):\n' >&2
    printf '  mkdir -p %s && cd %s\n' "${dir}" "${dir}" >&2
    printf '  curl -LO https://suitesparse-collection-website.herokuapp.com/MM/GHS_psdef/%s.tar.gz\n' "${name}" >&2
    printf '  tar xf %s.tar.gz\n' "${name}" >&2
}
