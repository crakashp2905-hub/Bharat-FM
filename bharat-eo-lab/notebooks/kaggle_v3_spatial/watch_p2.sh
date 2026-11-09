#!/usr/bin/env bash
# Waits for the protocol-2 kernels, downloads their outputs, and launches the reproducibility
# kernel as soon as one of the two GPU slots frees up.
cd "$(dirname "$0")"
OUT="${P2_OUT:-../../outputs/protocol2_results}"
done_state() { kaggle kernels status "crakashpreetham/bharat-eo-$1" 2>&1 | grep -qE "COMPLETE|ERROR|CANCEL"; }
launched=0
while true; do
  if [ $launched -eq 0 ] && { done_state p2-full || done_state p2-base; }; then
    kaggle kernels push -p push_p2_repro 2>&1 | grep -v Warning | tail -1; launched=1
  fi
  if done_state p2-full && done_state p2-base && [ $launched -eq 1 ]; then break; fi
  sleep 240
done
until done_state p2-repro; do sleep 240; done
mkdir -p "$OUT"
for k in p2-full p2-base p2-repro; do
  echo "== $k"; kaggle kernels status "crakashpreetham/bharat-eo-$k" 2>&1 | tail -1
  rm -rf "$OUT/$k"; kaggle kernels output "crakashpreetham/bharat-eo-$k" -p "$OUT/$k" 2>&1 | tail -1
done
echo WATCH_DONE
