#!/bin/bash
# Overnight runner: launches FD001..FD004 sequentially, commits results after each,
# then runs ablations, then generates RESULTS.md.
#
# Run in background:
#   nohup bash run_overnight.sh > logs/overnight.log 2>&1 &
#
# Resumes by skipping any subset whose results/<SUBSET>_results.json already exists.

cd /home/sagemaker-user/IndustrialJEPA/paper-replications/star
LOGDIR="logs"
mkdir -p "$LOGDIR"

log() {
    echo "$(date -u '+%Y-%m-%dT%H:%M:%S') $*"
}

commit_and_push() {
    local msg="$1"
    cd /home/sagemaker-user/IndustrialJEPA
    git add paper-replications/star/results/ \
            paper-replications/star/EXPERIMENT_LOG.md \
            paper-replications/star/logs/ 2>/dev/null || true
    git commit -m "$msg" 2>/dev/null || true
    git push origin main 2>/dev/null || log "git push failed (non-fatal, continuing)"
    cd /home/sagemaker-user/IndustrialJEPA/paper-replications/star
}

run_subset() {
    local subset="$1"
    if [ -f "results/${subset}_results.json" ]; then
        log "${subset} already done (results/${subset}_results.json exists), skipping"
        return 0
    fi
    log "Starting ${subset}..."
    python run_experiments.py "$subset" > "$LOGDIR/${subset}.log" 2>&1
    local rc=$?
    if [ $rc -ne 0 ]; then
        log "${subset} FAILED with exit code $rc - see $LOGDIR/${subset}.log"
        commit_and_push "STAR: ${subset} partial/failed - see logs"
        return 1
    fi
    if [ -f "results/${subset}_results.json" ]; then
        log "${subset} complete"
        commit_and_push "STAR: ${subset} results (5 seeds)"
    else
        log "WARNING: ${subset} finished but results JSON not found"
        commit_and_push "STAR: ${subset} finished with missing JSON"
    fi
    return 0
}

log "=== OVERNIGHT STAR RUNNER STARTED ==="
log "PID: $$"
log "GPU: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader 2>/dev/null || echo unknown)"

# Main subsets
for subset in FD001 FD002 FD003 FD004; do
    run_subset "$subset" || log "continuing after ${subset} failure"
done

log "=== MAIN SUBSETS COMPLETE ==="

# Ablations (optional; only run if at least one main subset succeeded)
if [ -f "results/FD001_results.json" ]; then
    mkdir -p results/ablations

    for ab in cond_norm rul_cap patch_length nheads; do
        if [ -f "results/ablations/${ab}.json" ]; then
            log "ablation ${ab} already done, skipping"
            continue
        fi
        log "Ablation: ${ab}..."
        python ablations.py "$ab" > "$LOGDIR/ablation_${ab}.log" 2>&1 || log "ablation ${ab} failed"
        commit_and_push "STAR: ablation ${ab}"
    done
else
    log "No main results - skipping ablations"
fi

# Generate final RESULTS.md
log "Generating RESULTS.md..."
python summarize_results.py > "$LOGDIR/summarize.log" 2>&1 || log "summarize failed"
commit_and_push "STAR: RESULTS.md generated, overnight run complete"

log "=== OVERNIGHT RUN COMPLETE ==="
