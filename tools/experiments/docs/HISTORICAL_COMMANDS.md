# Comandi storici: verifiche del 9 settembre 2026

Sono stati consultati i log originali in `logs/fp16_matrix`, le conversazioni locali
precedenti al 9 settembre e gli allegati terminale collegati alla conversazione
**“Update ICRA 2027 methodology”**, iniziata il 25 agosto. La cronologia shell
consultata non contiene comandi pertinenti. Non sono state copiate le conversazioni
integrali.

Il registro [commands.json](../history/commands.json) contiene **64 comandi**:

| Evidenza | Comandi | Significato |
|---|---:|---|
| Log TensorRT conservati | 11 | 5 build, 4 benchmark, 2 profili con comando RUNNING ed esito PASSED/FAILED |
| Allegati terminale recuperati | 10 | 2 prove iniziali trtexec e 8 invocazioni E2E con output |
| Messaggi dell'assistente | 43 | Comandi proposti in conversazione: non provano l'esecuzione |

Ogni voce riporta testo originale, sorgente, riga, hash SHA256 e tipo di evidenza.
I blocchi copiabili dei **soli comandi osservati** sono raccolti in
[OBSERVED_COMMANDS.md](../history/OBSERVED_COMMANDS.md). Il registro mantiene anche
suggerimenti su training e scale YOLO11 diverse, identificati come contesto storico;
non li trasforma in esperimenti nano effettivamente eseguiti.

## Microbenchmark FP16 effettivamente eseguiti

Il protocollo verificato è:

```text
--warmUp=2000 --duration=30 --useSpinWait --exportTimes=...
```

Nei comandi originali `--infStreams=1` **non era esplicito**: il valore 1 è riportato
nelle sezioni “Inference Options” dei log. I trasferimenti erano abilitati e
l'input generato da trtexec. Il tempo “GPU Compute Time” comprende anche i casi DLA;
il tipo di esecuzione si ricava dal build/inspector, non da quel nome.

| Configurazione effettiva | Comando benchmark | Comando build |
|---|---|---|
| Stock YOLO11n GPU | CMD-009 | CMD-010 |
| Stock YOLO11n DLA+GPU fallback | CMD-001 | CMD-002 |
| YOLO11-DLA GPU | CMD-004 | CMD-005 |
| YOLO11-DLA strict DLA | CMD-006 | CMD-007 |
| Stock YOLO11n strict DLA | nessuna misura: build fallita | CMD-011 |

Il prefisso `yolo11n-dla-gpu-fallback-fp16` è fuorviante: il build carica
**`logs/paper/yolo11n.onnx`**, quindi misura la baseline stock con fallback.
I path originali `logs/paper/fp16_matrix/...` non corrispondono alla posizione
attuale `logs/fp16_matrix/...`; non sono stati riscritti nei record storici.

Il profiling era una prova separata, **diversa dai default della nuova suite**:

```text
--warmUp=1000 --duration=0 --iterations=100 --separateProfileRun
--dumpProfile --exportProfile=...
```

I comandi CMD-003 e CMD-008 riguardano fallback e strict adapted. `iterations=100`
è il parametro richiesto, non l'affermazione che i JSON abbiano esattamente 100
campioni: l'audit esistente conta rispettivamente 129 e 137 campioni di profilo.
La nuova suite usa un profilo temporizzato a 10 s; per un confronto con i vecchi
profili dichiarare questa differenza oppure riutilizzare esplicitamente il vecchio
protocollo in una directory nuova.

## E2E: recuperate entrambe le fasi

[Output iniziale](../history/sources/2026-08-25-e2e-initial-default-stream.txt):
quattro comandi con **20 warmup / 200 iterazioni**, CPU postprocess e un thread
esplicito. Contiene il warning sullo stream CUDA predefinito; il decoder raw era
ancora denso. La conversazione propone poi il passaggio a uno stream dedicato e
la decodifica confidence-first. Queste prime misure non vanno mescolate alle finali.

[Output finale](../history/sources/2026-08-25-e2e-final-sparse.txt):
quattro comandi con **10 warmup / 100 iterazioni**, immagine
`logs/biascica_simone.jpg`, output CPU e decode sparse per gli engine adattati.
I flag non esplicitati sono conservati tali nel registro; non vengono aggiunti
retroattivamente ai comandi.

| Configurazione | Mediana totale finale (ms) | P95 (ms) | P99 (ms) | Comando |
|---|---:|---:|---:|---|
| Stock GPU | 19.203 | 19.574 | 19.880 | CMD-014 |
| Stock fallback | 52.782 | 53.098 | 53.396 | CMD-015 |
| Adapted GPU | 19.831 | 20.343 | 20.544 | CMD-013 |
| Adapted strict DLA | 56.497 | 56.864 | 57.781 | CMD-012 |

Questi numeri ora hanno una fonte terminale diretta, oltre a `FP16_RESULTS.md`.
**Non sono stati recuperati i JSON per frame**: le righe “Saved timing report”
provano che il programma ne indicò il salvataggio, non che i file siano disponibili
nel checkout. Sono una prova su una sola immagine, senza ripetizioni indipendenti.

Il terminale strict riporta inoltre i binding:

| Tensore | Formato riportato | Strides riportati | Elementi fisici |
|---|---|---|---:|
| input `(1,3,640,640)` | CHW4 | `(409600,1,640,1)` | 1.638.400 |
| P3 `(1,144,80,80)` | CHW16 | `(57600,6400,80,1)` | 921.600 |
| P4 `(1,144,40,40)` | CHW16 | `(14400,1600,40,1)` | 230.400 |
| P5 `(1,144,20,20)` | CHW16 | `(3600,400,20,1)` | 57.600 |

È una nuova evidenza storica di **questa configurazione**, non una prova di
intercambiabilità generale fra CHW4 e DLA_HWC4 né una validazione degli engine
che saranno ricostruiti. I test di layout sul nuovo target rimangono necessari.

## Training e versione del runtime

Le conversazioni conservano suggerimenti per training iniziale a 100 epoche,
resume a 300, un comando `--scratch` a 300 e una nuova proposta `--fresh` a 600.
**Nessuno di questi dimostra il comando effettivo del checkpoint a 500 epoche.**
La cronologia incorporata nel checkpoint resta la fonte per le 500 epoche;
inizializzazione e log completi del training restano da recuperare.

Il flag storico `--scratch` non è presente nel trainer attuale: il registro è
provenienza, non una garanzia di compatibilità dei vecchi argomenti. Lo script
corrente è [training/train_coco_dla.py](../training/train_coco_dla.py).

In [history/runtime/](../history/runtime/index.json) è conservata anche la versione
Git del tester precedente alle due correzioni del 9 settembre (sei candidati NMS
e soglia floating-point). Non possiamo dimostrare che quel preciso commit fosse
quello caricato sulla Jetson, poiché i terminali non riportano l'hash del sorgente.
La versione corrente non va presentata come il codice esatto dei vecchi numeri.

## Ripetere l'estrazione

Senza accedere a cronologie personali, si possono rileggere i log del workspace
e gli allegati già archiviati:

```bash
python tools/yolo11_dla_paper/run.py recover-commands \
  --output /tmp/yolo11-observed-commands.json
```

Per includere nuovamente i soli blocchi comando delle conversazioni disponibili:

```bash
python tools/yolo11_dla_paper/run.py recover-commands \
  --sessions /home/neo/.codex/sessions --before 2026-09-09 \
  --output /tmp/yolo11-historical-commands.json
```

L'estrattore non esegue i comandi, non modifica le cronologie e rifiuta di
sovrascrivere il JSON di output. I cinque allegati sono copie byte-identiche,
con hash in [sources/index.json](../history/sources/index.json).
