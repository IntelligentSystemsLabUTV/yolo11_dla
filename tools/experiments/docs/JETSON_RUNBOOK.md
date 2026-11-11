# Test finali Jetson — solo YOLO11-DLA

Questi script producono **nuove misure**, non completano retroattivamente i log
esistenti. Leggere prima [FINAL_TESTS.md](FINAL_TESTS.md) per le priorità e i
limiti del paper. Concorrenza GPU e integrazione ROS sono rimandate per decisione
dell'autore; nessun workload concorrente viene avviato.

## 1. Riutilizzo degli engine già misurati

Le build, gli inspector, i microbenchmark e i profili in `logs/fp16_matrix/` sono
risultati acquisiti. Il percorso normale **non esegue export, build o profiling**.
Usare la Jetson originale e il suo ambiente TensorRT per le nuove inferenze.
Servono il fork Ultralytics, PyTorch CUDA, OpenCV, NumPy, pycocotools e TensorRT
Python. Nessuno script modifica clock o power mode.

Dalla root del workspace sulla Jetson. `ICRA_SOURCE` è la cartella originale
già presente; `ICRA_EXISTING` è una **nuova destinazione**, che il comando
`reuse-existing` qui sotto deve creare. Assegnare una variabile con `export`
non crea la cartella.

```bash
export PAPER_TOOLS="$PWD/tools/yolo11_dla_paper"
export ICRA_SOURCE="$PWD/logs/fp16_matrix"
export ICRA_EXISTING="$PWD/logs/icra2027_final/existing"
export ICRA_ENGINES="$ICRA_EXISTING/engines"
export ICRA_RUN="$PWD/logs/icra2027_final/run_01"
export COCO_VAL="$PWD/logs/datasets/coco/images/val2017"
export COCO_ANN="$PWD/logs/datasets/coco/annotations/instances_val2017.json"
export PYTHONPATH="$PWD/tools/ultralytics${PYTHONPATH:+:$PYTHONPATH}"
export TRTEXEC=/usr/src/tensorrt/bin/trtexec
```

La registrazione eseguita nel workspace di sviluppo **non implica che questa
cartella esista sulla Jetson**. Se non esiste, crearla con il seguente comando
(non con `mkdir`, perché servono anche i collegamenti e il manifest):

```bash
python "$PAPER_TOOLS/run.py" reuse-existing \
  --source "$ICRA_SOURCE" --output "$ICRA_EXISTING"
```

Se `$ICRA_ENGINES/manifest.json` è già presente da una precedente registrazione,
saltare questo passaggio. I riepiloghi seguenti sono opzionali; eseguirli solo
se le rispettive directory di output non esistono:

```bash
python "$PAPER_TOOLS/run.py" summarize \
  --stage "$ICRA_EXISTING/micro" --output "$ICRA_EXISTING/summary-micro"
python "$PAPER_TOOLS/run.py" summarize \
  --stage "$ICRA_ENGINES" --output "$ICRA_EXISTING/summary-build"
```

`reuse-existing` crea collegamenti relativi e manifest, senza copiare gli engine,
modificare i log o invocare TensorRT. Conservare la struttura relativa delle due
cartelle durante il trasferimento, oppure registrare di nuovo sulla Jetson in una
cartella nuova. La mappa esatta è in [EXISTING_RESULTS.md](EXISTING_RESULTS.md).

## 2. Identità e condizioni delle nuove misure

Per installare COCO validation sulla Jetson, dalla root del workspace:

```bash
bash tools/yolo11_dla_paper/datasets/download_coco_val2017.sh
export COCO_VAL="$PWD/logs/datasets/coco/images/val2017"
export COCO_ANN="$PWD/logs/datasets/coco/annotations/instances_val2017.json"
```

Il percorso predefinito è `logs/datasets/coco` nella root del workspace, ricavata
dalla posizione dello script anche quando viene invocato da un'altra directory.
Servono `wget`, `unzip` e `python3`. Gli archivi ufficiali scaricati occupano circa
1.07 GB; prevedere 3 GB liberi includendo l'estrazione. Lo script riprende i download
e verifica la presenza delle 5.000 immagini indicate dal JSON e delle 80 categorie.
Se il dataset è già completo nella destinazione, non scarica nulla. Estrae soltanto
le annotazioni detection di validation; gli archivi rimangono in `.downloads/`.
Il controllo di presenza non decodifica i JPEG né certifica hash ufficiali.

Se COCO è già sul computer di training, si possono invece trasferire via `rsync`
la cartella `images/val2017/` e `annotations/instances_val2017.json` nella stessa
struttura sulla Jetson. Il nostro evaluatore usa direttamente il JSON COCO, senza
conversione delle label in formato YOLO. Con un container, montare la cartella
dataset nel container e impostare le due variabili sui percorsi interni.

Completare `configs/hardware-notes.example.md` con module/carrier, stack e
clock/power/cooling. Recuperare le condizioni storiche dove documentate e
registrare quelle attuali separatamente: non attribuire le impostazioni di oggi
alle misure di agosto. I manifest delle nuove fasi raccolgono anche le query
`nvpmodel -q` e `jetson_clocks --show`, con gli eventuali errori.

Gli hash registrati ora identificano gli engine conservati. I log provano i
percorsi ONNX utilizzati, ma non forniscono hash storici della catena
checkpoint→ONNX→engine. Questa informazione va recuperata dai record originali;
la sua assenza non richiede una ricompilazione per misurare gli engine esistenti.

Ogni nuova fase richiede una **directory di output nuova**; non c'è resume
automatico. `--dry-run` stampa i comandi senza creare output né importare
CUDA/TensorRT. Per i nuovi benchmark mantenere condizioni hardware uniformi;
se si usa `taskset`, applicarlo all'intera suite. Non lanciare le misure insieme.

## 3. Correttezza su target prima di AP e timing

```bash
python "$PAPER_TOOLS/run.py" parity \
  --engines "$ICRA_ENGINES" \
  --images "$COCO_VAL" --annotations "$COCO_ANN" \
  --output "$ICRA_RUN/parity" --limit 50
```

`evaluation/validate.py` verifica pack e unpack separatamente contro un oracolo di offset
scalari, usando pattern che variano per canale/riga/colonna e controllando il
padding. Salva formati, strides, capacità e byte fisici. Questo test CPU non basta
a provare come il device interpreta la memoria: segue il confronto reale fra
raw maps GPU e DLA dello **stesso modello**, su immagini preprocessate identiche.

Per scala registra errore assoluto/relativo e violazioni di tolleranza; registra
anche errori di coordinate/score decodificati, liste dopo NMS e parità denso/sparso
sia a conf 0.001 multi-label sia a conf 0.25 single-label. Il gate raw predefinito
`atol=0.1, rtol=0.05` è **esplorativo**, non una soglia di equivalenza scientifica
né una tolleranza sull'AP. Dichiarare eventuali valori diversi **prima** della
raccolta con `--raw-atol/--raw-rtol`; non aumentarli solo per rendere verde il test.
La parità del decoder usa `atol=1e-3, rtol=1e-5`. Un fallimento conserva il report e
ferma la fase: verificare layout, export, precisione e casi vicino alla soglia.
Le differenze dopo NMS fra engine restano da interpretare insieme all'AP completo.

## 4. AP comparabile, incluse le due implementazioni del decoder

Prima fare uno smoke con `--limit 20 --output "$ICRA_RUN/accuracy-smoke"`.
Poi, per il paper, **omettere `--limit`**:

```bash
python "$PAPER_TOOLS/run.py" accuracy \
  --engines "$ICRA_ENGINES" \
  --images "$COCO_VAL" --annotations "$COCO_ANN" \
  --output "$ICRA_RUN/accuracy"
```

Il comando esegue **sei valutazioni sugli engine esistenti**: quattro con decode
denso e i due adattati anche con decode sparso. Non richiede checkpoint o ONNX.
Per includere anche i checkpoint FP32, aggiungere `--stock-checkpoint /percorso/yolo11n.pt`
e `--adapted-checkpoint "$PWD/logs/yolo11n-dla-500ep.pt"`: diventano otto valutazioni,
senza export. È possibile fornire anche un solo checkpoint. Recuperare l'identità
dei checkpoint originali prima di interpretare la differenza engine/checkpoint
come sola perdita di conversione. Tutti condividono i 5.000
ID/immagini val2017, letterbox 640 quadrato, conf 0.001, IoU 0.7, NMS multi-label
class-aware e max_det 300; COCOeval mantiene i suoi maxDets ufficiali. La variante
sparsa cambia solo l'ordine del filtro/DFL. Salvati predictions, hash delle immagini,
AP/AP50/AP75/APsmall/medium/large e timing per frame. Questi timing a soglia 0.001
non sono quelli del percorso applicativo a 0.25.

Il confronto distingue differenza fra modelli/checkpoint `adapted-pt − stock-pt`,
conversione GPU `adapted-gpu − adapted-pt`, DLA `adapted-strict − adapted-gpu`,
e decoder sparse/dense. La prima differenza comprende anche i diversi training;
le differenze di conversione richiedono pesi sorgente corrispondenti.
L'AP del training 37.581 non sostituisce nessuna riga.

## 5. Pipeline multi-immagine; microbenchmark già disponibile

```bash
python "$PAPER_TOOLS/run.py" pipeline \
  --engines "$ICRA_ENGINES" \
  --images "$COCO_VAL" --annotations "$COCO_ANN" \
  --output "$ICRA_RUN/pipeline" --limit 100 --duration 60 --repeats 1
```

I microbenchmark e i profili esistenti sono già riassunti in `$ICRA_EXISTING`.
Non rilanciarli per ottenere di nuovo gli stessi valori. Se si vuole aggiungere
una stima della variabilità fra processi, `micro` resta disponibile come raccolta
**aggiuntiva** con `--repeats N`, senza profiling per default. Aggiungere `--profile`
solo quando servono nuovi profili. Il singolo lancio storico si può contare insieme
ai nuovi solo se engine, protocollo e condizioni hardware sono confrontabili;
l'aggregatore non unisce automaticamente campagne diverse.

Pipeline: 100 immagini caricate in RAM prima della misura, ordinate per ID e
ripetute ciclicamente, engine residente, 30 warmup e 30 s di stabilizzazione per
condizione, decode CPU sparse a conf 0.25/IoU 0.45. RAM necessaria proporzionale
alla sequenza: aumentare `--limit` se la board lo permette e usare la stessa lista
in tutte le condizioni. I record vengono scritti dopo la misura. Non è un replay
ROS né una misura camera-to-result; è servizio su immagini già in memoria.

Ogni frame contiene preprocess, pack+H2D, engine, D2H+unpack, postprocess, totale,
candidati e detections. Il totale è un intervallo reale, non una somma di mediane.
Il rate è `frame completati / durata reale`, non il reciproco della mediana.
Per confrontare decode dense/sparse in timing usare direttamente `benchmarks/benchmark.py`
con `--dense-decode`, stessi input/threshold e nuovi output, dopo la parità.

Per un replay temporizzato aggiungere `--fps 10 --deadline-ms 100` e un output
nuovo. **10 Hz è solo un punto iniziale da verificare**, non una frequenza target
robotica già concordata. La coda simulata ha capacità uno e conserva il frame più
recente: ogni slot saltato è contato come drop. Registrati release-to-result,
attesa, deadline mancate e rate effettivo. Le release sono sintetiche, non timestamp
di acquisizione di una camera reale; una sequenza COCO non è un rosbag.

## 6. Energia del modulo AGX Orin

Nella Jetson verificata i rail disponibili sono `VDD_GPU_SOC`, `VDD_CPU_CV` e
`VIN_SYS_5V0`. Il profilo `agx-orin` integra la loro somma come **potenza del
modulo**. `VDD_CPU_CV` include DLA/PVA e `VIN_SYS_5V0` include il rail DDR:
quest'ultimo non viene sommato una seconda volta. La misura non è alla presa
e non rappresenta il consumo completo di carrier, periferiche e alimentatore.

Nel container il binario dell'host è stato copiato in `/tmp/tegrastats-paper`.
Se il container viene ricreato, renderlo nuovamente disponibile e verificare
che mostri i tre rail. Eseguire la raccolta con **una ripetizione**, come concordato:

```bash
python "$PAPER_TOOLS/run.py" energy \
  --engines "$ICRA_ENGINES" \
  --images "$COCO_VAL" --annotations "$COCO_ANN" \
  --output "$ICRA_RUN/energy-10hz-module" --fps 10 --duration 60 --repeats 1 \
  --tegrastats /tmp/tegrastats-paper --power-profile agx-orin
```

Sono quattro configurazioni e un idle, in ordine randomizzato. Non cambiare power
mode, clock o cooling fra le condizioni. Aumentare `--repeats` solo per aggiungere
repliche indipendenti; una singola ripetizione non stima la variabilità fra lanci.
La condizione a saturazione (`--fps 0`) resta una raccolta distinta, se necessaria.

Il collector conserva **tutti i rail**, le righe originali e i timestamp monotoni
alla ricezione in `telemetry.jsonl`. Prima della stabilizzazione verifica che
tegrastats sia vivo e fornisca tutti i rail richiesti; se mancano, ferma il lancio
prima della misura. Per l'integrazione usa i valori istantanei (prima della `/`),
convertiti da mW a W, e campioni comuni ai tre rail. Interpola ai bordi dello
stesso intervallo effettivo dei frame. Rifiuta rail mancanti nell'intervallo,
valori non finiti, copertura insufficiente e gap superiori a 0.5 s.

Ogni `summary.json` salva il totale del modulo e gli integrali dei tre rail,
con profilo, scope, intervallo, watt medi, joule e joule/frame. L'idle ha energia
e potenza ma non joule/frame. Il riepilogo automatico abbina l'idle dello stesso
blocco e sottrae la sua potenza media moltiplicata per la durata del lancio attivo:

```bash
python "$PAPER_TOOLS/run.py" summarize \
  --stage "$ICRA_RUN/energy-10hz-module" \
  --output "$ICRA_RUN/summary-energy-10hz-module"
```

- `results.csv`: totale del modulo, joule/frame e valori con idle sottratto,
  insieme a rate ottenuto, drop e latenze. Le colonne `power_scope` e
  `power_rails` rendono esplicito cosa è misurato.
- `energy_rails.csv`: stessa scomposizione energetica per ciascuno dei tre rail.
- `report.json`: entrambi i riepiloghi, definizione del profilo e statistiche fra
  lanci; con una ripetizione la deviazione standard fra lanci non viene stimata.

Gli eventuali valori negativi dopo sottrazione dell'idle sono conservati per
analisi. Controllare il rate effettivamente consegnato: stesso `--fps` richiesto
non implica stesso rate ottenuto. I ritardi del sensore/pipe non sono calibrati.
Nessun altro processo tegrastats viene arrestato dal collector.

### Recuperare raccolte già completate con un solo rail

Se `energy-10hz_02` è già terminata usando `--rail VDD_GPU_SOC`, non ripetere le
inferenze: i file di telemetria conservano anche gli altri rail. Ricalcolare in
una nuova directory:

```bash
python "$PAPER_TOOLS/run.py" summarize \
  --stage "$ICRA_RUN/energy-10hz_02" \
  --output "$ICRA_RUN/summary-energy-10hz_02-module" \
  --power-profile agx-orin
```

Il comando integra di nuovo i campioni originali usando `start_s`/`end_s` e il
numero di frame di ogni lancio, registra gli hash dei file di telemetria e non
modifica i risultati sorgente. Richiede una campagna completata, compreso l'idle,
e tutti i rail con copertura valida. Non recupera una campagna interrotta prima
della raccolta. Senza il flag, i vecchi risultati restano etichettati come misura
del solo rail scelto, mai automaticamente come totale del modulo.

`--rail` resta disponibile per un rail singolo (per esempio `VDD_IN` su hardware
che lo espone); è mutuamente esclusivo con `--power-profile`. Per il caso AGX Orin
verificato usare il profilo sopra. La topologia è documentata da
[NVIDIA](https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html#software-based-power-consumption-modeling)
e dal [chiarimento sui rail](https://forums.developer.nvidia.com/t/jetson-agx-orin-ina3221-power-monitor-layout/223111).

## Riepiloghi e verifica locale

```bash
python "$PAPER_TOOLS/run.py" summarize \
  --stage "$ICRA_RUN/accuracy" --output "$ICRA_RUN/summary-accuracy"
python "$PAPER_TOOLS/run.py" summarize \
  --stage "$ICRA_RUN/pipeline" --output "$ICRA_RUN/summary-pipeline"

python -m pytest "$PAPER_TOOLS/tests" -q
```

Stesso riepilogatore per gli engine registrati, parity e micro. JSON e CSV mantengono i singoli
lanci; le statistiche fra lanci non trattano migliaia di frame come repliche
indipendenti. L'aggregatore AP rifiuta confronti con immagini/hash/threshold diversi.
Nessuno script aggiorna automaticamente valori nel manoscritto.

Le verifiche già eseguite e i loro limiti sono in [LOCAL_VALIDATION.md](LOCAL_VALIDATION.md).

## Export e build soltanto quando cambia il modello

`prepare` e `build` rimangono disponibili per nuovi checkpoint, modifiche del
grafo o un cambio di piattaforma/runtime che impedisce il caricamento degli
engine. Non sono prerequisiti per la raccolta sopra. In quel caso usare directory
nuove e passare `--artifacts` anche alle fasi successive della nuova campagna.

Riferimenti implementativi: [formati TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/reference/data-format-desc.html),
[guida TensorRT 10.3](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-1030/pdf/TensorRT-Developer-Guide.pdf),
[tegrastats Jetson](https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html).
