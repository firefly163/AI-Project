#!/bin/bash
export FASTTEXT_MODEL_PATH="/home/lty/.cache/fasttext-langdetect/lid.176.bin"

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"


python -m gsv.api \
   -s $ROOT/gsv/weights/ASA_sovits_e8_s6648.pth \
  -g $ROOT/gsv/weights/ASA-gpt-e15.ckpt\
  -hb $ROOT/gsv/pretrained_models/chinese-hubert-base \
  -b  $ROOT/gsv/pretrained_models/chinese-roberta-wwm-ext-large \
  -d cuda:0 \
  --text_lang ja \
  -dr  $ROOT/gsv/extracted_ogg/v_asa1417.ogg_0000000000_0000180800.wav \
  -dt  "遠慮なさらずおっしゃってください 私は絶対に笑ったりはしません" \
  -dl ja\
  --ref_wav $ROOT/gsv/extracted_ogg/v_asa1417.ogg_0000000000_0000180800.wav \
  --ref_text "遠慮なさらずおっしゃってください 私は絶対に笑ったりはしません" \
  --ref_lang ja \
  --out_dir out_repl \
  --basename asahi \
  --port 9881

