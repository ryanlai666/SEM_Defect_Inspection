#!/usr/bin/env bash
# All Colab L4 jobs (Dinomaly, YOLO26n-seg, detector benchmark), run detached on the VM from /content/sem:
#   nohup bash scripts/colab/run_all.sh > /content/sem/logs/run_all.log 2>&1 &
# Each step writes "<time> START|END <step> rc=<rc>" to logs/status.txt and its output to logs/<step>.log.
# A failed step does not stop the chain.  `touch logs/STOP` ends the chain at the next step boundary
# (e.g. to stay within a Colab compute budget).  logs/ALL_DONE marks the end.
set -u
cd /content/sem
export PYTHONPATH=/content/sem/src PYTHONIOENCODING=utf-8 MLFLOW_DISABLE_AGENT_HINT=1
export YOLO_CONFIG_DIR=/content/sem/.ultralytics
mkdir -p logs outputs
st() { echo "$(date '+%F %T') $*" >> logs/status.txt; }
step() {  # step <name> <cmd...>
  local n=$1; shift
  if [ -e logs/STOP ]; then st "SKIP $n (STOP)"; return 0; fi
  if grep -q "END $n rc=0" logs/status.txt 2>/dev/null; then st "SKIP $n (done)"; return 0; fi
  st "START $n"; "$@" > "logs/$n.log" 2>&1; local rc=$?; st "END $n rc=$rc"; return 0
}

{ nvidia-smi; python -c "import torch, sys; print(sys.version); print(torch.__version__, torch.version.cuda)"; \
  python -m pip list 2>/dev/null | grep -i -E "^(torch|torchvision|timm|ultralytics|numpy|pandas|scikit-learn|opencv|pycocotools|faster)"; \
  nproc; free -g; } > logs/env.txt 2>&1

step download python scripts/download_data.py
step yolo_prep python scripts/colab/yolo_prep.py --out /content/yolo_data

# ---- (a) Dinomaly x1, off-mask protocol, in the background while (b) trains (24 GB L4; training only)
if ! grep -q "END dinomaly rc=0" logs/status.txt 2>/dev/null && [ ! -e logs/STOP ]; then
  st "START dinomaly"
  ( python -m semdefect.cli run configs/anomaly/dinomaly_offmask.yaml tracking.enabled=false \
      data.num_workers=4 checkpoint_minutes=10 > logs/dinomaly.log 2>&1; st "END dinomaly rc=$?" ) &
  DPID=$!
else
  DPID=""
fi

# ---- (b) YOLO26n-seg, 35 epochs
step yolo_train python scripts/colab/yolo_train.py --epochs 35
[ -n "$DPID" ] && wait $DPID
step yolo_eval python scripts/colab/yolo_eval.py
step latency python scripts/colab/latency_l4.py   # nothing else on the GPU from here on

# ---- (c) detector benchmark, priority order, one at a time (latency measured right after each training)
step det_prep python scripts/colab/det_prep.py --src /content/yolo_data --out /content/det_data
step det_yolo26n python scripts/colab/det_ultra.py --name yolo26n
step det_sme_yolo python scripts/colab/det_ultra.py --name sme_yolo
step det_rtdetr_l python scripts/colab/det_ultra.py --name rtdetr_l
step det_faster_rcnn python scripts/colab/det_frcnn.py
step det_deimv2_l python scripts/colab/det_deimv2.py --stage all
touch logs/ALL_DONE
