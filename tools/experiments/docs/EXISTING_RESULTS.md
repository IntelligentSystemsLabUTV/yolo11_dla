# Risultati esistenti: riutilizzo senza build

Verifica del **10 settembre 2026** su `logs/fp16_matrix/`.

| Configurazione suite | Engine originale | Latenza mediana engine+I/O | Campioni |
|---|---|---:|---:|
| `stock-gpu` | `yolo11n-gpu-fp16.engine` | 3.74023 ms | 8.651 |
| `adapted-gpu` | `yolo11n-dla-gpu-fp16.engine` | 3.50000 ms | 9.519 |
| `stock-fallback` | `yolo11n-dla-gpu-fallback-fp16.engine` | 34.67770 ms | 878 |
| `adapted-strict` | `yolo11n-dla-strict-dla-fp16.engine` | 27.51460 ms | 1.119 |

I quattro log di build riportano successo. Il log stock strict conserva il
fallimento al MatMul: non serve ricrearlo. L'inspector strict adattato contiene
un solo nodo DLA, senza GPU/reformat. Sono presenti i profili separati di strict
e fallback. I quattro trace comprendono **20.167 campioni**, un lancio per
configurazione. Il prefisso del fallback è ambiguo, ma il comando di build usa
`yolo11n.onnx`: è la baseline stock. I file `yolo-dla-n-*` restano esclusi.

## Registrazione nel workspace di sviluppo

`run.py reuse-existing` è stato eseguito nel workspace di sviluppo, senza
TensorRT, producendo i file seguenti. La cartella non viene creata automaticamente
sulla Jetson: usare il comando di registrazione del runbook se lì non esiste.

- [Engine, evidenze e hash](../../../logs/icra2027_final/existing/engines/manifest.json).
- [Quattro microbenchmark storici](../../../logs/icra2027_final/existing/micro/manifest.json).
- [Riepilogo timing CSV](../../../logs/icra2027_final/existing/summary-micro/results.csv).
- [Riepilogo build CSV](../../../logs/icra2027_final/existing/summary-build/results.csv).

Sono manifest e collegamenti relativi agli originali. Nessun engine è stato
ricompilato, deserializzato o modificato. La suite verifica gli hash prima delle
nuove misure e il riepilogatore verifica quelli delle evidenze archiviate.
Per trasferire il lavoro, mantenere la struttura relativa di `logs/` oppure
rieseguire la registrazione sulla Jetson in una cartella nuova.

Gli hash identificano i file **alla registrazione attuale**. I comandi storici
contengono percorsi ONNX, non hash storici di checkpoint/export/engine. Non si
attribuisce retroattivamente una provenienza completa. Il modello di training
fornito e gli ONNX già presenti in `logs/` restano disponibili; non occorre
riesportarli per inferire con gli engine conservati. L'identità del checkpoint
stock originale resta da recuperare per il confronto AP PyTorch appaiato.

## Cosa aggiungere

1. **Parità numerica e decoder sugli engine conservati:** controlla layout,
   precisione e correttezza dell'implementazione utilizzata per le nuove misure.
2. **AP COCO val2017:** manca la valutazione completa degli engine. Il comando
   `accuracy` usa sei percorsi engine; i checkpoint si aggiungono tramite flag
   espliciti, senza export. Verificare la loro provenienza prima di attribuire
   tutta la differenza di AP alla conversione.
3. **Pipeline multi-immagine:** gli E2E storici riguardano una sola immagine e
   una versione precedente del decoder. Le quattro righe restano evidenza
   storica, confermata dagli allegati; i JSON YOLO11 per frame non sono presenti.
4. **Metadati hardware e training:** recuperare quanto manca dai record originali
   e annotare separatamente le condizioni delle nuove misure.

Repliche del microbenchmark sono facoltative per rafforzare la stima di
variabilità; i profili esistenti non si ripetono. Energia resta condizionale ai
claim del paper, concorrenza GPU rimandata come concordato. Nessuna nuova build
è necessaria salvo un cambiamento del modello o un problema effettivo di
compatibilità/caricamento sulla piattaforma target.

Comandi aggiornati: [JETSON_RUNBOOK.md](JETSON_RUNBOOK.md).
