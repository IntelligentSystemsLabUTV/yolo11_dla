# Test mancanti per chiudere il paper YOLO11-DLA

Rilettura di manoscritto, `logs/` e fork Ultralytics: **10 settembre 2026**.
Scope concordato: **YOLO11-DLA-n detection**, con YOLO11n come baseline. La famiglia
indipendente YOLO-DLA, YOLO26 e la segmentazione non entrano negli esperimenti.
Il workload GPU concorrente verrà definito successivamente dall'autore: la sua
integrazione è rimandata, non considerata completata.

Gli script e i comandi sono in [JETSON_RUNBOOK.md](JETSON_RUNBOOK.md).
Aggiornamento `run_01`: esperimenti eseguiti dall’autore sulla Jetson e verificati
dai log; nessuna nuova inferenza durante l’audit. Dettagli in
[RUN_01_AUDIT.md](RUN_01_AUDIT.md).

## Scope corrente concordato dopo run_01

- INT8 sarà eseguito successivamente dall'autore ed è temporaneamente escluso
  dal manoscritto; gli script restano disponibili.
- Condizioni dichiarate: AGX Orin Developer Kit, L4T 36.4.4, ROS 2 Jazzy,
  NVIDIA 50 W, `jetson_clocks` disabilitato.
- Inizializzazione da YOLO11n sui tensori compatibili, procedura e ottimizzatore
  ricostruiti: [TRAINING_RECONSTRUCTION.md](TRAINING_RECONSTRUCTION.md).
- I test FP16 acquisiti sono mantenuti senza ulteriori repliche per questa
  revisione. Percentili per frame, senza stima della variabilità fra test.
- Concorrenza e replay non sono prerequisiti per i claim attuali: il paper non
  attribuisce benefici a task concorrenti. Nessun nuovo training richiesto.
- Il manoscritto contiene solo il placeholder di revisione degli autori P8.

Le procedure e priorità qui sotto conservano il piano sperimentale più ampio;
non sono tutte attività obbligatorie per lo scope corrente.

## Cosa abbiamo realmente

| Evidenza | Stato e conseguenza |
|---|---|
| Strict compile YOLO11-DLA | Confermata nei vecchi log: un loadable DLA e nessun nodo GPU/reformat. Build e inspector già acquisiti; riutilizzare l’engine conservato. Gli hash attuali sono registrati senza inferire una catena storica mancante. |
| Stock strict | Fallisce al MatMul di C2PSA. Esito negativo già documentato: nessun nuovo tentativo necessario. |
| Microbenchmark FP16 | Già disponibili quattro configurazioni YOLO11, **un lancio ciascuna** con trace completo. Strict adapted 27.515 ms vs stock fallback 34.678 ms: −20.7% al confine engine+trasferimenti. GPU adapted 3.500 ms. |
| Accuracy | Sei valutazioni engine complete su 5.000 immagini: stock GPU **39.314 AP**, adapted GPU **37.995**, strict **37.991**; dense/sparse AP praticamente identica. Checkpoint appaiati e catena storica dei pesi ancora da completare. |
| E2E | Nuovi record multi-immagine: **56.820 ms strict vs 51.241 ms fallback**, una ripetizione per configurazione. Confermano il limite della pipeline e sostituiscono la vecchia trascrizione da singola immagine nel paper. |
| Decoder/export CPU | Già verificata parità dense/checkpoint/ONNX su un input sintetico; non equivale a correttezza TensorRT DLA o AP COCO. |
| Sistema/energia | Energia del modulo a 10 Hz acquisita: **0.874 J/frame strict**, **0.917 fallback**, **0.761 adapted GPU**, 600 frame ciascuno senza drop. Una ripetizione; condizioni/variabilità da completare. Concorrenza e replay assenti. |

Tre insidie di provenienza:

- `logs/fp16_matrix/yolo11n-dla-gpu-fallback-fp16.*` è **stock YOLO11n**: lo prova il comando ONNX nel build log.
- I soli JSON E2E conservati in quella cartella appartengono a **YOLO-DLA**, escluso. Non possono riempire le righe YOLO11-DLA.
- `logs/dla_yolo11n*.md` documentano soprattutto **segmentation 512**, non la detection 640 del paper. Il nome `best-mAP50-95_0.37395.pt` non garantisce quel valore: l'audit del contenuto rileva un altro record.

Fonti locali: [EVIDENCE_AUDIT.md](../../../docs/analysis/audits/EVIDENCE_AUDIT.md),
[MODEL_AUDIT.md](../../../docs/analysis/audits/MODEL_AUDIT.md), [engine_summary.csv](../../../docs/analysis/engine_summary.csv).
Il testo corrente è già dedicato a YOLO11-DLA; i vecchi draft e l'altro progetto
rimangono separati.

## Ordine di esecuzione e criterio di chiusura

| Priorità / placeholder | Test da fare | Script pronto | Quando il punto è chiuso |
|---|---|---|---|
| **1 — P1/P2** | Completare identità board, software, condizioni e provenienza dei checkpoint | Template hardware, manifest e record originali | Dati recuperati o limiti dichiarati; nessun nuovo export/build richiesto |
| **2 — P3** | Parità numerica e decoder sugli engine GPU/DLA esistenti | `run.py parity` | Errori documentati su immagini reali, layout e decoder verificati |
| **3 — P2/P3** | COCO val2017 sugli engine esistenti; checkpoint originali quando disponibili | `run.py accuracy`, `evaluate_coco.py` | Sei valutazioni engine, più i checkpoint forniti (otto con entrambi), stessi 5.000 ID e protocollo |
| **4 — P4** | Pipeline multi-immagine con runtime corretto, una ripetizione iniziale | `run.py pipeline` | Timing per frame, totale, mediana/P95/P99 e throughput; variabilità fra lanci solo aggiungendo repliche |
| **Aggiuntivo — P4** | Repliche del microbenchmark solo per stimare variabilità fra lanci | `run.py micro --repeats N` | Le misure storiche restano valide; nuovi lanci sono evidenza aggiuntiva, profiling disattivato per default |
| **Condizionale — P7** | Idle e rate comune sostenibile, 60 s ×1 iniziale; saturazione e repliche aggiuntive se necessarie | `run.py energy` | Somma dei tre rail del modulo AGX Orin, intervalli coperti, J/frame totali e per rail, idle sottratto e rate effettivo confrontabili |
| **Rimandato — P6** | Workload GPU affiancato e replay robotico | Da integrare dopo scelta del workload | Misure di slowdown/task latency e replay reale |
| **Estensione richiesta — INT8** | Calibrazione e build dei due ONNX, poi AP FP16/INT8 sullo stesso held-out | `int8-split`, `int8-calibrate`, `int8-build`; [protocollo](INT8_RUNBOOK.md) | Cache reali per modello, placement/precisioni verificati, calibrazione separata dalla valutazione |

**Già chiusi per gli artifact conservati:** quattro build riuscite, rifiuto strict
della baseline, inspector, microbenchmark e profili DLA/fallback. Non ricompilare
né riprofilare per riempire la nuova struttura di directory. La registrazione e i
riepiloghi sono già pronti: [EXISTING_RESULTS.md](EXISTING_RESULTS.md).

La tabella sopra conserva le procedure e i criteri di chiusura: `run_01` ha
completato le sei AP engine e la prima acquisizione pipeline/energia. Il gate
raw di parità resta fallito e documentato; layout e decoder sono verificati.
Restano provenienza/condizioni operative, checkpoint appaiati, variabilità fra
lanci e la nuova campagna INT8. I build FP16 e le AP complete non vanno ripetuti.
La concorrenza resta molto utile per motivare il contributo a ICRA, ma è fuori da
questa esecuzione per decisione dell'autore. Finché manca non si possono rivendicare
isolamento del sistema o miglioramento operativo del robot.

Una validazione identica rende comparabili le misure di AP, ma **non rende uguali
i budget di training** dello stock e dell'adattato: descriverli esplicitamente.

AP dei soli checkpoint può essere eseguita anche sulla workstation con lo stesso
protocollo; AP degli engine e misure di sistema richiedono la Jetson target. I tempi dei checkpoint su un'altra macchina non entrano nel
confronto di latenza Jetson.

## Cosa è stato preparato

- **Riutilizzo degli artifact.** `reuse-existing` registra i quattro engine e i relativi log/inspector/timing/profili tramite link e hash. Le nuove fasi accettano questi engine senza `prepare` o `build`; AP checkpoint opzionale e separata dall’export.
- **Export opzionale dei due modelli, per future modifiche.** Copia dei checkpoint YOLO11n e YOLO11-DLA, verifica di mixer/head e scala nano, export ONNX con shape statiche controllate e hash degli artifact.
- **Matrice completa con provenienza.** Comandi come liste di argomenti, nessun shell interpolation; `--dry-run`; hash verificati prima delle fasi successive; output nuovi e stop sui fallimenti inattesi. Nessuna modifica automatica a clock/power mode.
- **Audit del runtime.** Oracolo di offset per pack/unpack, pattern di canali/coordinate, errori raw per scala, errori score/box, detections dopo NMS e confronto sparse/dense a soglia applicativa e di validazione.
- **AP anche del percorso sparse.** L'evaluatore esistente ora accetta `--sparse-decode` solo per engine raw; mantiene conf 0.001 e multi-label. I riepiloghi controllano identità delle immagini e delle soglie.
- **Pipeline residente.** Sequenza precaricata, timing per frame, candidate count, pacing opzionale, coda simulata latest-only con drop e deadline espliciti. Non è un nodo ROS/camera reale.
- **Energia e aggregazione.** `--power-profile agx-orin` integra i tre rail senza doppio conteggio del DDR, conserva totale e componenti e controlla la copertura temporale. Idle appaiati per blocco; CSV/JSON per ogni lancio. `summarize --power-profile agx-orin` può ricalcolare una campagna completata dalla telemetria esistente senza nuove inferenze.

Due bug incontrati durante la preparazione sono stati riprodotti e corretti:

1. In `ultralytics/utils/nms.py`, un tensore ordinario `(1,84,6)` con esattamente sei candidati veniva interpretato come detections end-to-end `(B,N,6)`. Ora il numero esplicito di classi disambigua il caso YOLO11.
2. In `tools/yolo_e2e_tester.py`, il filtro sparse sui logits poteva divergere dal filtro dense vicino alla soglia per arrotondamento. Ora applica sigmoid al massimo logit di ogni anchor prima del confronto: mantiene la selezione prima della DFL, con lo stesso confronto floating-point del dense.

Queste correzioni richiedono **nuove misure** del percorso sparse. I numeri storici
non sono stati aggiornati né presentati come risultati del codice nuovo. I test
CPU coprono sei candidati, soglie/pareggi, multi-label, zero candidati, layout con
padding e integrazione energetica; non certificano esecuzione DLA.

## Chiusura del manoscritto — P8

Dopo la raccolta:

1. Inserire risultati misurati in abstract, tabelle AP/latency, figure e conclusione; distinguere rigorosamente engine+transfer e pipeline completa.
2. Rimuovere ogni placeholder sperimentale rimasto (P1–P4, P6–P7) oppure la relativa promessa se il test viene escluso. La sezione attuale “Validation Gaps and Robotics Experiments” non può restare un elenco di esperimenti futuri nel paper finale.
3. Limitare i claim alle configurazioni effettivamente misurate: singola board/stack, nano, detection, 640, batch 1. FP16 già disponibile; INT8 è una nuova campagna da completare prima di inserirne risultati. Altre scale, segmentation e YOLO-DLA restano fuori scope.
4. Rivedere i risultati e l'acknowledgment P8 con gli autori, quindi rigenerare e controllare il PDF secondo [GUIDELINES.md](../../../docs/paper/audits/GUIDELINES.md). Le verifiche editoriali restano distinte da quelle Jetson.

I soli dati esterni verificati per implementare gli script riguardano
[formati TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/reference/data-format-desc.html)
e [tegrastats](https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html).
Le conclusioni sperimentali sopra provengono dagli artifact locali.

Aggiornamento del recupero storico: [HISTORICAL_COMMANDS.md](HISTORICAL_COMMANDS.md)
documenta gli output terminale E2E recuperati dagli allegati della conversazione del
25 agosto. Confermano le quattro righe del riepilogo e i binding, ma non contengono
i campioni per frame dei JSON ancora mancanti.
