# Audit di run_01 — 10 settembre 2026

Sorgente: `logs/icra2027_final/run_01`. Audit dei file conservati, senza nuove
inferenze, build o modifiche ai risultati. I dati FP16 sono internamente
coerenti e utilizzabili con i limiti sotto; il manoscritto non è ancora aggiornato.

Aggiornamento successivo: il manoscritto è stato aggiornato. L'autore ha
confermato power mode 50 W, `jetson_clocks` disabilitato, ROS 2 Jazzy e
inizializzazione YOLO11n sui layer compatibili. Si veda
[TRAINING_RECONSTRUCTION.md](TRAINING_RECONSTRUCTION.md). Le osservazioni
sotto descrivono ciò che è contenuto nei log: le conferme non alterano i
manifest originali né aggiungono repliche. INT8 è escluso dalla revisione
corrente del manoscritto e la concorrenza resta fuori dai claim misurati.

## Completezza e controlli effettuati

- `accuracy`: complete, sei job riusciti, ciascuno con 5.000 ID unici identici,
  nello stesso ordine e con gli stessi hash JPEG. Hash degli engine verificati
  contro i file esistenti. Ogni detection in `predictions.json` corrisponde
  esattamente alla sequenza in `frames.jsonl`. Nessun valore non finito,
  score fuori [0,1] o larghezza/altezza negativa. Alcune box hanno area zero
  dopo clipping (70–213 per configurazione); non sono file corrotti.
- `pipeline`: complete, quattro job, 60 s e una ripetizione ciascuno.
  Conteggi, ID ciclici, hash, FPS, mediana/P95/P99 ricalcolati dai frame.
- `energy-10hz-module`: complete, quattro job attivi e idle. Integrali
  ricalcolati indipendentemente con interpolazione ai bordi e trapezi sui tre
  rail. Telemetria copre tutte le finestre; gap massimo inferiore a 0,112 s.
- I tre report automatici referenziano l'hash corretto del manifest sorgente.
- `energy-10hz`: fallito per tegrastats mancante. Escludere dai risultati;
  la campagna utilizzabile è `energy-10hz-module`.
- `parity`: fallito, da conservare come tale. Vedi sotto.

Le annotation COCO non sono presenti in questa copia locale del dataset:
l'audit verifica coerenza dei record, hash, metriche salvate e output COCOeval
nei log; non ricalcola autonomamente AP dalla ground truth.

## Accuratezza

Protocollo comune: conf 0,001, IoU NMS 0,7, multi-label, max_det 300,
letterbox centrato 640, batch 1, postprocessing CPU.

| Configurazione | AP | AP50 | AP75 |
|---|---:|---:|---:|
| Stock GPU | 39,31405 | 55,09042 | 42,75368 |
| Stock fallback | 39,24470 | 55,05974 | 42,73044 |
| Adattato GPU | 37,99540 | 53,72144 | 41,12617 |
| Adattato strict DLA | 37,99072 | 53,70009 | 41,12870 |

Differenza DLA–GPU sullo stesso adattato: −0,00468 punti AP. Differenza
adattato strict–stock GPU: −1,32333 punti AP. Quest'ultima confronta modelli
e training differenti, non isola il solo effetto di un cambiamento architetturale.
Il precedente 37,581 AP del checkpoint appartiene alla validazione del training,
non al medesimo protocollo appaiato degli engine.

Dense/sparse differiscono in AP per meno di 10^-7 punti sui due engine.
Non sono bit-identici: su GPU l'immagine 74092 contiene 251 detection dense
e 252 sparse; i totali sono 882.960 e 882.961. Sullo strict entrambi 882.695.
Descrivere equivalenza pratica delle metriche, non identità delle detection.

Parità su 50 immagini: 8 controlli layout passano; 200 controlli dense/sparse
passano. Il gate raw GPU/DLA fallisce su 50/49/49 immagini per P3/P4/P5 con
atol 0,1 e rtol 0,05. La conservazione dell'AP non rende superato quel gate
e non dimostra equivalenza numerica elemento per elemento.

## Pipeline

100 immagini identiche fra configurazioni, precaricate e ripetute per 60 s,
conf 0,25, IoU NMS 0,45, sparse per l'adattato, un thread CPU.
Esclusi disco, camera e ROS. Numero di cicli diverso perché il test è a durata
fissa: 10–12 occorrenze per immagine su DLA/fallback, 28–29 su GPU.

| Configurazione | Frame | Mediana totale ms | P95 ms | P99 ms | FPS |
|---|---:|---:|---:|---:|---:|
| Stock GPU | 2.880 | 20,030 | 24,658 | 27,702 | 47,989 |
| Adattato GPU | 2.858 | 20,307 | 24,584 | 28,591 | 47,625 |
| Stock fallback | 1.138 | 51,241 | 59,542 | 64,478 | 18,953 |
| Adattato strict DLA | 1.032 | 56,820 | 65,042 | 76,886 | 17,192 |

Strict contro fallback: engine_ms −20,95% (27,781 contro 35,143 ms), ma totale
+10,89%. Packing/H2D strict 7,747 ms contro 2,338; D2H/unpack 6,506 contro
1,518; postprocessing 8,400 contro 6,062 (mediane dei singoli stadi).
Non sommare mediane come decomposizione esatta della mediana totale.

Il campo engine_ms misura chiamata runtime e sincronizzazione lato host;
non equivale al tempo kernel puro né al confine engine+transfer di trtexec.
Il risultato storico −20,7% resta una misura distinta. Le nuove misure
confermano qualitativamente l'inversione dell'ordinamento al confine pipeline.

## Energia a 10 Hz

600 frame completati per configurazione, ogni immagine esattamente sei volte,
zero drop e zero deadline miss osservati a 100 ms. Somma VDD_GPU_SOC,
VDD_CPU_CV e VIN_SYS_5V0: energia del modulo, non consumo alla presa.

| Configurazione | W medi | J/frame totali | J/frame oltre idle |
|---|---:|---:|---:|
| Stock GPU | 7,60510 | 0,76051 | 0,06030 |
| Adattato GPU | 7,60627 | 0,76063 | 0,06042 |
| Stock fallback | 9,16571 | 0,91657 | 0,21636 |
| Adattato strict DLA | 8,73760 | 0,87376 | 0,17355 |

Idle 7,00211 W, un solo intervallo comune. Strict consuma il 4,67% in meno
del fallback, ma il 14,87% in più dell'adattato GPU sul totale J/frame.
Sono differenze osservate in un lancio, senza stima di variabilità fra lanci.
I risultati al netto dell'idle dipendono dalla stabilità di quella baseline.

## Limiti e conseguenze per il paper

- Board identificata come NVIDIA Jetson AGX Orin Developer Kit; L4T 36.4.4,
  TensorRT 10.3, CUDA runtime 12.6. `nvpmodel` e `jetson_clocks` non erano
  disponibili nel container: modalità energetica e clock policy non registrate.
  La telemetria mostra CPU variabile soprattutto fra 729 e 1497 MHz;
  non descrivere questi esperimenti come eseguiti a clock CPU bloccati.
- Nel test energetico tj osservata circa 45,9–47,4 °C; ciò non costituisce
  una verifica completa dell'assenza di throttling. Non c'è telemetria pipeline.
- Il preprocessing è circa 3,0–3,4 ms sulle GPU nella pipeline, ma circa
  5,6 ms sui percorsi DLA/fallback e a 10 Hz. Stessi dati e codice: non
  attribuire questa differenza al solo acceleratore. Governor/carico sono
  possibili spiegazioni, non cause dimostrate dai log.
- Una ripetizione consente statistiche sui frame, non intervalli di confidenza
  fra esecuzioni. Per un claim energetico centrale del 4,7%, raccogliere
  ulteriori repliche con condizioni documentate, conservando questa prova.
- Lo strict engine ha un solo nodo DLA nell'inspector storico verificato;
  la pipeline usa comunque runtime CUDA e trasferimenti. La telemetria GR3D
  strict raggiunge 7%: non rivendicare GPU globale inattiva o isolamento.
- Nessuna misura INT8 o concorrenza in questa cartella. La validazione appaiata
  dei checkpoint manca se si vuole separare training, export e deployment.
- Aggiornare abstract, sezioni accuracy/application/energy e conclusione:
  il testo dice ancora AP engine ed energia mancanti e usa la vecchia tabella
  pipeline da singola immagine. Non occorre rifare i build FP16 o le sei AP.

Fonti: `summary-accuracy/results.csv`, `summary-pipeline/results.csv`,
`summary-energy-10hz-module/{results,energy_rails}.csv`, manifest, frame,
predizioni e telemetria nelle rispettive directory di `run_01`.
