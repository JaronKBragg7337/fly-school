# Frozen protocol: TEST 9 of comm_protocol.md (2026-09-17). One process per fly. Set SEEDS to the fly list.
param([string[]]$Seeds = @(7373,7474,7575,7676,7777,7878,7979,8080), [string]$Tag = "test9", [string]$Reps = "24")
$py="C:\Users\lilli\AI-Shared\projects\fly-brain\.venv\Scripts\python.exe"
$env:COMM_GRAPH="C:/Users/lilli/Fly-Lab/versions/fly-v9/graph_v9.npz"; $env:COMM_CALIB="C:/Users/lilli/Fly-Lab/versions/fly-v10/calib.json"
$env:COMM_A="ORN_DA2"; $env:COMM_B="ORN_DL3"; $env:COMM_C="ORN_VM5d"; $env:COMM_A_HZ="40"; $env:COMM_B_HZ="40"
$env:COMM_EPOCHS="12"; $env:COMM_REPS=$Reps; $env:COMM_PULSE_MS="300"; $env:COMM_LR="0.2"
$env:COMM_ANSWER="DNa13,DNa03,MDN"; $env:COMM_READ_EXTRA="DNa13,DNa03,MDN"; $env:COMM_DECODE="valence2"; $env:COMM_BLOCK="1"
$env:COMM_REPLAY="1"; $env:COMM_NO_B="1"; $env:COMM_NO_EQUALISE="1"; $env:COMM_ANSWER_TYPES="MBON09,MBON01,MBON05,MBON03,MBON06"; $env:COMM_KEEP_STORE="1"
Remove-Item Env:COMM_TOPK,Env:COMM_ONESIDED,Env:COMM_THRESH -ErrorAction SilentlyContinue
foreach ($sd in $Seeds) { $env:COMM_SEEDS="$sd"; $env:COMM_TAG="${Tag}_s$sd"; Start-Process -FilePath $py -ArgumentList "comm_loop.py" -WorkingDirectory "C:\Users\lilli\Fly-Lab-2" -RedirectStandardOutput "results\comm_loop_${Tag}_s$sd.log" -RedirectStandardError "results\comm_loop_${Tag}_s$sd.err" -WindowStyle Hidden }
