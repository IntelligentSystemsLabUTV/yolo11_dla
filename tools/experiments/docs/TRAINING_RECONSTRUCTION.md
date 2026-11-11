# Ricostruzione del training e condizioni dichiarate — 10 settembre 2026

Questo record separa conferme dell'autore, codice ispezionato e metadati del
checkpoint. Non è un nuovo training né una ricostruzione degli hash storici
degli artifact mancanti.

## Conferme dell'autore

- Jetson AGX Orin Developer Kit, L4T 36.4.4, ROS 2 Jazzy, power mode NVIDIA
  50 W, `jetson_clocks` disabilitato.
- YOLO11-DLA inizializzato da `yolo11n`, trasferendo i layer compatibili.
- Mantenere i test FP16 esistenti senza ulteriori repliche richieste per questa
  revisione. INT8 temporaneamente escluso dal manoscritto; concorrenza fuori
  dai claim sperimentali finché non misurata.

I primi due dettagli completano la descrizione sperimentale. Il power mode
non è stato retroattivamente misurato dall'audit: le query nel container
restano fallite nei manifest originali. Non modificarli.

## Percorso nel repository

1. `training/train_coco_dla.py`, `MODELS['yolo11n-dla']`: costruzione da
   `cfg/models/11/yolo11-dla.yaml`, pesi iniziali `yolo11n.pt`.
2. `model.load(...)` prima di `model.train(...)`.
3. `ultralytics/nn/tasks.py`, `BaseModel.load`: intersezione dello state dict
   e `load_state_dict(..., strict=False)`.
4. `ultralytics/utils/torch_utils.py`, `intersect_dicts`: nomi e shape uguali.
   Le parti compatibili del blocco sostituito possono essere copiate anch'esse;
   non è corretto dire che tutto C2DLA sia necessariamente inizializzato da zero.
5. Nessuna chiamata al caricamento specializzato QKV o al training a due fasi
   presente nell'esempio separato `examples/train_dla.py`.

## Configurazione conservata

Fonte: `tools/YOLO11-DLA_ICRA/analysis/checkpoint_training.json`, estratto da
`logs/yolo11n-dla-500ep.pt`, SHA256
`417967ac4a0af77f4a4e1f1f31ba4e31b38775853fa7393f0f8824f2a3511535`.

- 500 epoche, batch 32, immagini 640, seed 0, workers 8, AMP,
  `freeze=None`, `optimizer=auto`, `nbs=64`.
- Base lr 0,01, `lrf=0.01`, warmup 3 epoche, scheduler lineare,
  weight decay 0,0005; mosaic chiuso nelle ultime 10 epoche.
- HSV h/s/v 0,015/0,7/0,4; translate 0,1; scale 0,5; fliplr 0,5;
  mosaic 1; MixUp e copy-paste 0.
- 2.615.696 parametri non fusi. Storia AP finale 37,581 / AP50 53,151.

## Ottimizzatore ricostruito dalla revisione del checkpoint

Ispezionata con `git show` la revisione Ultralytics
`3f2711b10e21eedf4e2efaa907c1322d844ea2d7`, non dedotta dalla versione corrente:

- `engine/trainer.py`: iterazioni previste =
  `ceil(len(dataset) / max(batch_size, nbs)) * epochs`.
- Sopra 10.000 iterazioni, `optimizer=auto` seleziona **MuSGD**, lr 0,01 e
  momentum **0,9**, ignorando il valore nominale 0,937 negli argomenti.
- I gruppi selezionati della testa di classificazione ricevono lr ×3.
- Le otto serie `lr/pg*` registrate sono coerenti: alla quinta epoca
  0,0099208 e 0,0297624, all'ultima 0,0001198 e 0,0003594.

Questa è una ricostruzione da codice e storia salvata, non una lettura di uno
stato optimizer completo, che non è disponibile nel checkpoint stripped.
Il file storico `coco-bbox.yaml` non è presente in questa copia del workspace;
non si dichiara ricostruito il suo hash né l'esatto comando shell originale.
Questi limiti non impediscono di descrivere la procedura di inizializzazione
confermata dall'autore o le AP direttamente misurate sugli engine.
