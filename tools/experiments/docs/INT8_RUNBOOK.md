# INT8 su TensorRT 10.3 — campagna YOLO11-DLA

Aggiornamento **14 settembre 2026**. Comandi dalla root del workspace, nel
container sulla Jetson che ha generato gli engine. Nessun nuovo training,
export, split, calibrazione o build è necessario per la campagna attuale.

## Risultati già acquisiti

| Configurazione | Engine relativo a `logs/int8_matrix/` | Stato |
|---|---|---|
| Stock GPU | `debug-stock-gpu.mJXwWe/stock-gpu-int8.engine` | Build riuscita; AP held-out completa |
| Adattato GPU | `remaining.z5RBAZ/yolo11n-dla-gpu-int8.engine` | Build e smoke riusciti |
| Stock DLA + fallback GPU | `remaining.z5RBAZ/yolo11n-dla-gpu-fallback-int8.engine` | Build e smoke riusciti |
| Adattato strict DLA | `debug-strict-native-int8.gKgeWc/adapted-strict-int8.engine` | Build riuscita; validazione su immagini da eseguire |

- Quattro build `PASSED`. Lo strict contiene **un nodo DLA, zero GPU/Reformat**;
  il dettaglio interno riporta 298 operazioni INT8 e una Softmax FP16, sempre
  su DLA. Lo stock fallback contiene quattro nodi DLA e 23 Reformat, oltre a
  layer GPU, NoOp e costanti. Precisione mista e placement sono proprietà distinte.
- Le due cache risultano complete: 500 immagini identiche, seed 2027,
  EntropyCalibrator2 prima delle fusioni, TensorRT 10.3.0.
- `logs/int8-evaluation.8ZbovD/accuracy-stock-gpu`: **4.500 immagini, AP 33,175,
  AP50 47,324, AP75 36,069**. Riutilizzare questo risultato.
- Smoke su 50 ID held-out: stock GPU AP 43,531; adattato GPU 43,444;
  stock fallback 32,760. Sono diagnostici, non AP finali. Il calo del fallback
  richiede valutazione completa e analisi; la sola conclusione dello smoke
  non dimostra parità di accuratezza. I due smoke stock hanno predizioni identiche.
- Nessun tempo di build/tactic o tempo del validatore COCO sostituisce i test
  di latenza ed energia qui sotto.

## 1. Percorsi

```bash
export PAPER_TOOLS="$PWD/tools/yolo11_dla_paper"
export COCO_VAL="$PWD/logs/datasets/coco/images/val2017"
export COCO_ANN="$PWD/logs/datasets/coco/annotations/instances_val2017.json"
export INT8_SPLIT="$PWD/logs/datasets/coco/int8_split_500_seed2027"
export INT8_CAL="$PWD/logs/int8_calibration"
export INT8_OUT="$PWD/logs/int8_matrix"
export STOCK_ONNX="$PWD/logs/yolo11n.onnx"
export DLA_ONNX="$PWD/logs/yolo11n-dla-500ep.onnx"
export TRTEXEC=/usr/src/tensorrt/bin/trtexec
export INT8_READY="$PWD/logs/int8_ready"
export INT8_RUN="$PWD/logs/icra2027_int8/run_01"
```

`INT8_READY` e `INT8_RUN` sono **nuove destinazioni create dai comandi sotto**,
non cartelle storiche che dovrebbero già esistere. Nella stessa campagna
riutilizzare le variabili, senza cambiare directory a ogni test.

## 2. Registrare gli engine esistenti, senza ricompilare

```bash
python "$PAPER_TOOLS/run.py" int8-reuse \
  --matrix "$INT8_OUT" --calibration "$INT8_CAL" --split "$INT8_SPLIT" \
  --stock-onnx "$STOCK_ONNX" --adapted-onnx "$DLA_ONNX" \
  --output "$INT8_READY"
```

Questo comando CPU:

- trova esattamente un build riuscito per ciascuna configurazione;
- verifica cache/ONNX, split disgiunto, comandi e inspector;
- crea link agli engine con nomi uniformi, copia log, inspector e metadati,
  archivia gli hash e produce un nuovo manifest `complete`;
- estrae dalla cache adattata le scale float32 esatte di `images` e
  `pred_p3/p4/p5`, verificandole contro i range espliciti nel build log;
- scrive `adapted-strict.engine.quantization.json`, legato all'hash dell'engine.

Non modifica il vecchio `int8_matrix/manifest.json`, che documenta il primo
build fallito. Non esegue calibrazione né inferenza. Non ripetere la registrazione
se è già riuscita; la destinazione deve essere nuova. Non cancellare gli engine
originali, perché la cartella pronta li collega tramite symlink.

Il runtime ora carica automaticamente il sidecar per binding INT8: divide per
la scala, arrotonda ties-to-even e satura a [-128,127] prima del packing;
moltiplica le uscite per la scala dopo D2H, prima del decoder. Rifiuta un engine
INT8 senza scale valide. Queste conversioni entrano nei tempi applicativi;
gli engine con I/O FP16 conservano il percorso precedente.

La scelta delle scale non è una stima da AP: `trtexec` 10.3 assegna esplicitamente
ai binding INT8 i range della cache. I log strict confermano tali assegnazioni.
I range stampati sono arrotondati: servono al controllo, non al calcolo delle scale.
La correttezza su hardware resta da verificare con il test successivo.

## 3. Verificare lo strict su 50 immagini

```bash
python "$PAPER_TOOLS/run.py" parity \
  --engines "$INT8_READY" \
  --images "$COCO_VAL" --annotations "$INT8_READY/evaluation.json" \
  --output "$INT8_RUN/parity" --limit 50 --raw-diagnostic-only
```

Il test verifica layout e decoder dense/sparse, esegue i due engine adattati e
salva errori raw, box, score e detection post-NMS in
`parity/comparison/report.json`. Le scale usate sono incluse nel report.

`--raw-diagnostic-only` registra gli scostamenti GPU/DLA senza usare le
vecchie tolleranze FP16 come criterio di accettazione INT8. Layout errato,
valori non finiti o divergenza dense/sparse restano errori bloccanti.
Un esito completo non certifica AP equivalente: ispezionare le detection.
Se il comando fallisce, analizzare il report/log prima di lanciare i test lunghi.
Non cambiare le scale per far coincidere i risultati GPU e DLA.

## 4. Accuratezza completa: riusare lo stock, eseguire le altre valutazioni

```bash
python "$PAPER_TOOLS/run.py" accuracy \
  --engines "$INT8_READY" \
  --images "$COCO_VAL" --annotations "$INT8_READY/evaluation.json" \
  --output "$INT8_RUN/accuracy" \
  --reuse-evaluation "stock-gpu=$PWD/logs/int8-evaluation.8ZbovD/accuracy-stock-gpu"

python "$PAPER_TOOLS/run.py" summarize \
  --stage "$INT8_RUN/accuracy" --output "$INT8_RUN/accuracy-summary"
```

Nessun `--limit`: **4.500 ID held-out**, disgiunti dalle 500 immagini di
calibrazione. Non chiamare questi risultati AP sull'intero COCO val2017.

La suite verifica hash dell'engine, ID/hash immagini e soglie prima di collegare
la valutazione stock. Poi esegue cinque valutazioni: adattato GPU, stock fallback,
adattato strict, adattato GPU sparse e adattato strict sparse. Le impostazioni
restano conf 0.001, IoU 0.7, max_det 300, un thread CPU.

Se alcune di queste valutazioni complete esistono già, aggiungere altri
`--reuse-evaluation nome=/percorso/completo`. Sono accettati i nomi
`adapted-gpu`, `stock-fallback`, `adapted-strict`, `adapted-gpu-sparse`,
`adapted-strict-sparse`. Non usare uno smoke come valutazione completa.
La suite non sovrascrive una fase esistente; per riprendere dopo un fallimento
usare una destinazione nuova e collegare i risultati completati.

## 5. Confronto FP16 sullo stesso held-out, senza nuova inferenza

```bash
python "$PAPER_TOOLS/run.py" int8-rescore-fp16 \
  --source "$PWD/logs/icra2027_final/run_01/accuracy" \
  --engines "$INT8_READY" --output "$INT8_RUN/fp16-heldout"

python "$PAPER_TOOLS/run.py" summarize \
  --stage "$INT8_RUN/fp16-heldout" --output "$INT8_RUN/fp16-heldout-summary"
```

Ricalcola COCOeval dalle sei predizioni FP16 archiviate, selezionando esattamente
i 4.500 ID INT8. Usa CPU e pycocotools, non carica engine e non esegue TensorRT.
Confrontare gli AP per configurazione nei due `results.csv`; l'AP FP16 storico
sulle 5.000 immagini non è il riferimento appaiato corretto.

## 6. Latenza dell'engine e trasferimenti

Dopo aver verificato correttezza e accuratezza:

```bash
python "$PAPER_TOOLS/run.py" micro \
  --engines "$INT8_READY" --trtexec "$TRTEXEC" \
  --output "$INT8_RUN/micro" --duration 30 --repeats 1

python "$PAPER_TOOLS/run.py" summarize \
  --stage "$INT8_RUN/micro" --output "$INT8_RUN/micro-summary"
```

Quattro engine residenti: warmup 2 s, un inference stream, spin-wait,
trasferimenti abilitati, nessun profiling nel test base. `trtexec` usa input
sintetici: è un microbenchmark, non il test su immagini reali. Mantiene il
protocollo FP16, ma i binding strict INT8 trasferiscono quantità di dati diverse.
Per profili separati, aggiungere `--profile` **al primo lancio** solo se servono;
non rilanciare tutta la fase per recuperare un profilo.

## 7. Pipeline su immagini reali

```bash
python "$PAPER_TOOLS/run.py" pipeline \
  --engines "$INT8_READY" \
  --images "$COCO_VAL" --annotations "$COCO_ANN" \
  --output "$INT8_RUN/pipeline" --limit 100 --duration 60 --repeats 1

python "$PAPER_TOOLS/run.py" summarize \
  --stage "$INT8_RUN/pipeline" --output "$INT8_RUN/pipeline-summary"
```

**Qui usare `COCO_ANN`, non il held-out:** le prime 100 immagini in ordine di ID
sono le stesse del workload FP16. Warmup 30 frame, stabilizzazione 30 s, un thread
CPU, CPU decode/NMS, conf 0.25, IoU 0.45. Quantizzazione e dequantizzazione strict
sono incluse. Il test riporta throughput e mediana/P95/P99 per configurazione;
non è un replay di nodo ROS/camera né un test di concorrenza.

## 8. Energia a 10 Hz e nuovo idle

Stessa Jetson AGX Orin, L4T 36.4.4, TensorRT 10.3, modalità NVIDIA 50 W,
`jetson_clocks` disabilitato; mantenere le condizioni FP16. `tegrastats` deve
esistere ed essere eseguibile **dentro il container**. Nella campagna FP16 è
stato copiato dall'host in `/tmp/tegrastats-paper`. Se manca (ad esempio dopo
la ricreazione del container), ripetere questo comando **sull'host Jetson**:

```bash
docker cp -L /usr/bin/tegrastats \
  devcontainer-yolo_dla-jetson6-1:/tmp/tegrastats-paper
```

Poi **dentro il container**, impostare il percorso copiato e verificare che
produca campioni con i tre rail richiesti (interrompere con Ctrl+C):

```bash
export TEGRASTATS=/tmp/tegrastats-paper
"$TEGRASTATS" --interval 100
```

La copia rende disponibile il binario; lo script energy lo avvia e arresta
automaticamente per ciascun test, incluso l'idle, usando `--tegrastats`.
Il comando seguente registra quattro configurazioni e un nuovo intervallo idle
in ordine deterministico randomizzato, per 60 s ciascuno:

```bash
python "$PAPER_TOOLS/run.py" energy \
  --engines "$INT8_READY" \
  --images "$COCO_VAL" --annotations "$COCO_ANN" \
  --output "$INT8_RUN/energy-10hz" --limit 100 \
  --fps 10 --duration 60 --repeats 1 \
  --tegrastats "$TEGRASTATS" --power-profile agx-orin

python "$PAPER_TOOLS/run.py" summarize \
  --stage "$INT8_RUN/energy-10hz" --output "$INT8_RUN/energy-summary"
```

Somma dei rail `VDD_GPU_SOC + VDD_CPU_CV + VIN_SYS_5V0`: potenza del modulo,
non misura alla presa. Riporta energia totale e per rail, J/frame e incremento
rispetto al nuovo idle. Controllare FPS effettivi, drop, deadline e copertura
della telemetria; non sottrarre l'idle FP16 alle nuove misure.

## Cosa inviare e limiti della verifica locale

Inviare i nuovi log, manifest, report, frames/sequence e telemetria di
`INT8_RUN`, più `INT8_READY/manifest.json` e il sidecar delle scale. Non servono
nuove copie degli engine o delle immagini.

I test CPU verificano formule, layout, metadati e orchestration; non certificano
l'esecuzione TensorRT/DLA sulla Jetson. L'esito dei test hardware mancanti non è
stato anticipato nel paper. I risultati devono sostenere anche eventuali perdite
di accuratezza o assenza di vantaggi energetici, senza presumere benefici INT8.

## Riproduzione futura e fonti

Per una campagna nuova restano `int8-split`, `int8-calibrate` e `int8-build`.
Il wrapper di build ora usa `int8:dla_hwc4` / `int8:chw32` per lo strict;
non ha resume e non va rilanciato su questa matrice. Gli engine
`calibration-gpu.engine` sono artefatti di preparazione con binding FP32, non
configurazioni aggiuntive del paper. Per valutare uno strict nativo occorre
comunque registrarlo con `int8-reuse` e i suoi metadati.

- [TensorRT 10.3, sampleEngines.cpp](https://github.com/NVIDIA/TensorRT/blob/release/10.3/samples/common/sampleEngines.cpp): assegnazione dei range I/O dalla cache.
- [NVIDIA, quantizzazione e calibrazione](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/work-quantized-types.html): formule, precisione mista e portabilità.
- [NVIDIA, formati I/O DLA](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/work-with-dla.html#i-o-formats-on-dla): DLA_HWC4 INT8 in ingresso e CHW32 INT8 in uscita.
