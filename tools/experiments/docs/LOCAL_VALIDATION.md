# Verifiche degli script — 2026-09-09

Queste sono verifiche software locali, **non nuovi risultati Jetson o COCO del paper**.
Le fixture e gli export di controllo sono stati scritti in una directory temporanea
fuori da `logs/` e dal manoscritto.

- `python -m pytest docs/analysis/jetson/test_protocol.py -q`: **14 test superati**. Include due regressioni inizialmente fallite prima delle correzioni (sei candidati NMS e pareggi alla soglia), NMS end-to-end esplicita, sparse/dense multi-label e vuoto, CHW16/CHW4/DLA_HWC4/DLA_LINEAR con padding, rail/unità/integrazione, copertura/gap delle misure di potenza, scheduling/drop, strict flags/inspector e blocchi di ripetizioni randomizzati.
- `prepare.py` eseguito fino al completamento con checkpoint stock nano **casuale di controllo** e checkpoint adattato fornito. Gli export ONNX stock e adattato hanno superato i controlli sulle shape previste.
- `evaluate_coco.py` completato su CPU con checkpoint adattato e una annotation **sintetica di una sola immagine**: manifest, predictions, timing e COCOeval prodotti. L'AP della fixture non è un risultato scientifico.
- Collector energia eseguito con un processo tegrastats **simulato** a 12 W: timestamp, stop del solo figlio, copertura e integrazione producono correttamente 12 W medi. Non sono misure elettriche della macchina.
- Inspector verificato anche sui JSON originali: accetta il singolo nodo strict DLA, respinge come strict sia il fallback con quattro nodi DLA sia il puro GPU.
- Riepilogatore micro verificato sul log stock GPU esistente: conserva tutti gli 8.651 campioni e la mediana 3.74023 ms, senza scartare nuovamente il warmup.
- `compileall` e `pyflakes` superati. Flake8 nell'ambiente ha un conflitto tra plugin (`--application-import-names`); la verifica statica è stata eseguita direttamente con pyflakes.

**Ancora da verificare sul target:** comandi trtexec con la versione installata,
build dei nuovi export, significato dei formati/strides effettivi,
TensorRTEngine/COCO sui nuovi engine, errori FP16 e softmax DLA, AP completo,
prestazioni e telemetria della Jetson reale. Non è stata dichiarata superata
nessuna di queste prove hardware.

## Riordino e recupero storico — completamento 2026-09-10

- La suite in `tools/yolo11_dla_paper/tests/` comprende ora **17 test superati**:
  ai 14 originali si aggiungono isolamento dei riepiloghi di invocazioni diverse,
  estrazione dei soli comandi pertinenti e distinzione delle famiglie/task esclusi.
- Verificati i 24 spostamenti, i collegamenti dei vecchi script e tutti i link locali
  nella documentazione della raccolta. Il Makefile del paper usa i nuovi percorsi.
- Tutte le sei fasi della suite producono lo stesso piano via `run.py` e via il
  vecchio percorso, anche eseguendo da `/tmp`, senza creare output in dry-run.
- Ripetuta l'integrazione dopo lo spostamento: export ONNX stock e adattato, smoke COCO CPU
  su annotation sintetica e audit dei 20.167 campioni storici completati. Le figure
  di controllo e gli altri output sono in una cartella temporanea.
- `make -C tools/YOLO11-DLA_ICRA check` supera il controllo PDF esistente:
  otto pagine, font incorporati e nessun errore di formato rilevato; rimangono
  i 14 placeholder sperimentali/editoriali del draft. Non è stato rigenerato il PDF.
- L'archivio dei comandi contiene 11 comandi dai log TensorRT, 10 dagli allegati
  terminale e 43 suggerimenti dalle conversazioni, distinti per evidenza.
  I cinque allegati copiati conservano gli hash originali. Sono stati recuperati
  i riepiloghi E2E e i binding, non i JSON con i campioni per frame.

Durante il riordino non sono stati avviati training, sincronizzazioni W&B,
server di risultati o benchmark sulla Jetson.

## Verifica della matrice finale a due modelli — 2026-09-10

- `python -m pytest tools/yolo11_dla_paper/tests -q`: **17 test superati**.
  `compileall` e `pyflakes` sui tre script modificati superati.
- Ripetuto `prepare` su CPU con checkpoint stock nano casuale di controllo e
  checkpoint adattato fornito: esattamente due checkpoint e due export ONNX,
  shape verificate e hash delle copie identici ai checkpoint sorgente.
  Output temporanei in `/tmp/yolo11-main-models-xlvlyp7g/`; nessun nuovo dato AP.
- Verificate da `/tmp` tutte le fasi `run.py --dry-run`, senza creazione di output:
  build **5** comandi, parity **1**, accuracy **8**, micro **24** (20 misure e
  4 profili separati), pipeline **20**, energy **25**. Build e timing usano
  esclusivamente i due modelli e le quattro configurazioni principali.
- `make -C tools/YOLO11-DLA_ICRA local`: PDF rigenerato e controllo superato,
  **8 pagine**, 14 font incorporati, nessun overfull o riferimento irrisolto.
  Rimangono **13 placeholder** sperimentali/editoriali del draft.
- Nessuna prova Jetson eseguita; le verifiche TensorRT sul target restano da fare.

## Riutilizzo degli engine storici — 2026-09-10

Questo aggiornamento sostituisce il requisito operativo di preparare nuovi export
e nuove build; le sezioni precedenti descrivono verifiche software già effettuate.

- Registrati i quattro engine originali, i cinque log di build (incluso stock
  strict fallito), gli inspector, i profili disponibili e i quattro timing trace
  in `logs/icra2027_final/existing/`, usando link relativi e hash.
- Riepiloghi build e micro generati dagli originali: **20.167 campioni**, una
  sola esecuzione per configurazione, mediane coerenti con l'audit precedente.
  Nessun export, build, caricamento engine o benchmark eseguito.
- Suite: **20 test superati**, inclusi riutilizzo senza export, conservazione dei
  sorgenti, rifiuto di engine modificati, identificazione della baseline fallback
  e aggiunta facoltativa di un checkpoint alla valutazione engine. `compileall`
  e `pyflakes` sugli script modificati superati.
- Verificati hash e piani da `/tmp` sugli engine reali: parity 1, accuracy 6,
  micro 20, pipeline 20, energy 25 comandi. Nessun output in dry-run e nessuna
  invocazione di export/build. `accuracy` con un checkpoint produce 7 comandi;
  `micro --profile` ne produce 24, mentre il default non ripete i profili.
- Verificato anche il piano opzionale per nuove build: cinque configurazioni.
  I comandi correnti sono in [JETSON_RUNBOOK.md](JETSON_RUNBOOK.md).

## Energia del modulo AGX Orin — 2026-09-10

- **30 test superati**. Le nuove verifiche coprono integrazione dei tre rail con
  interpolazione, esclusione del DDR già incluso in VIN_SYS_5V0, lettura dei
  valori istantanei, rail mancanti, NaN, gap, copertura temporale e scope dell'idle.
- Verificati il piano delle quattro configurazioni più idle con una ripetizione,
  il riepilogo CLI totale/per rail e il ricalcolo dei vecchi risultati da
  `telemetry.jsonl`. Quest'ultimo conserva byte per byte i file sorgente.
- Collector e benchmark idle eseguiti con un processo tegrastats simulato:
  2.398 + 0.399 + 3.705 = **6.502 W** sul totale del modulo, tre integrali
  separati e joule/frame nullo per idle. Sono fixture software, non misure Jetson.
- Verificato il rifiuto iniziale di una sorgente simulata che espone solo
  VDD_GPU_SOC; il collector conserva il log diagnostico e termina il proprio
  processo. `compileall` e `pyflakes` sugli script modificati superati.
- Nessun benchmark hardware avviato. Le nuove misure usano
  `run.py energy --power-profile agx-orin`; `run.py summarize` riconosce lo
  scope salvato e produce anche `energy_rails.csv`.

## Preparazione INT8 — 2026-09-10

- **33 test superati**, inclusi split riproducibile e disgiunto, conservazione
  dei flag I/O FP16 e rifiuto di cache modificate o associate a un altro ONNX.
- Verificati `int8-build --dry-run`, help del calibratore, `pyflakes` e
  `compileall`. Lo split CLI è stato eseguito su 5.000 record sintetici:
  manifest completo, 500 immagini di calibrazione e 4.500 di valutazione.
  Le fixture verificano selezione e hash, non decoding o calibrazione.
- Nessun build o calibrazione INT8 eseguito. Il calibratore TensorRT, le
  precisioni effettive e la compilazione strict vanno verificati sulla Jetson
  con lo stack 10.3 originale. Procedura in [INT8_RUNBOOK.md](INT8_RUNBOOK.md).

## Manoscritto aggiornato con run_01 — 2026-09-10

- Inseriti risultati COCO, pipeline ed energia FP16 acquisiti dall'autore;
  verificate tutte le 12 righe delle tre nuove tabelle contro i CSV sorgente.
- `make -C tools/YOLO11-DLA_ICRA local`: **8 pagine**, 14 font incorporati,
  nessun Type3, link PDF, bookmark, overfull o riferimento irrisolto.
  Ispezionate visivamente tutte le pagine. Restano sei occorrenze di placeholder
  per provenienza/training, condizioni operative, repliche energetiche,
  concorrenza e revisione degli autori.
- Conservato il fallimento del gate raw; descritti limiti del singolo lancio e
  clock variabili. INT8 indicato come campagna futura. Nessuna nuova inferenza,
  build di engine o misura hardware eseguita durante l'aggiornamento del paper.

## Revisione con condizioni e inizializzazione confermate — 2026-09-11

- Inseriti 50 W, `jetson_clocks` disabilitato e ROS 2 Jazzy, dichiarati
  dall'autore; ricostruiti trasferimento dei tensori compatibili da YOLO11n
  e ricetta di training dalla revisione archiviata nel checkpoint.
- INT8 escluso temporaneamente dal testo. Tesi limitata ai risultati FP16
  misurati, senza claim di isolamento o benefici su task concorrenti.
- Verificate nuovamente le 12 righe delle tabelle rispetto ai CSV originali.
  PDF ricompilato e ispezionato su tutte le pagine: 8 pagine, 14 font
  incorporati, controlli locali superati. Resta solo il placeholder P8
  di revisione degli autori; non è stata dichiarata avvenuta tale revisione.
