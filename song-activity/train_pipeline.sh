#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
#  train_pipeline.sh — TTS Training Only
#  Downloads LJSpeech and trains BOTH Tacotron 2 and FastSpeech 2
# ═══════════════════════════════════════════════════════════════
set -e

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Song Activity — TTS Training Pipeline                  ║"
echo "║  Tacotron 2 vs FastSpeech 2 comparison                  ║"
echo "╚══════════════════════════════════════════════════════════╝"

# ── Config (adjust to your GPU) ──
DATA_DIR="./data"
DATASET_DIR="./dataset"
SVS_SAVE="./svs_saved"

TACOTRON_EPOCHS=500
FASTSPEECH_EPOCHS=600
BATCH_SIZE=16        # 4GB VRAM → 8 | 6GB → 16 | 8GB+ → 32
MAX_SAMPLES=""       # Empty = use all ~13,100 | Set to "500" for quick test

# ═══════════════════════════════════════════════════════════════
echo ""
echo "═══ [1/5] Checking CUDA ═══"
python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
if torch.cuda.is_available():
    print(f'  ✓ GPU: {torch.cuda.get_device_name(0)}')
    print(f'    VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
else:
    print('  ✗ CUDA not available — will be very slow')
"

# ═══════════════════════════════════════════════════════════════
echo ""
echo "═══ [2/5] Installing dependencies ═══"
pip install -r requirements.txt

# ═══════════════════════════════════════════════════════════════
echo ""
echo "═══ [3/5] Downloading LJSpeech + Preparing dataset ═══"
EXTRA_ARGS=""
if [ -n "${MAX_SAMPLES}" ]; then
    EXTRA_ARGS="--max_samples ${MAX_SAMPLES}"
fi

python3 download_tts_dataset.py \
    --output_dir "${DATA_DIR}" \
    --dataset_dir "${DATASET_DIR}" \
    ${EXTRA_ARGS}

# ═══════════════════════════════════════════════════════════════
echo ""
echo "═══ [4/5] Training Tacotron 2 (${TACOTRON_EPOCHS} epochs) ═══"
python3 main.py train-svs \
    --model tacotron2 \
    --dataset_dir "${DATASET_DIR}" \
    --epochs ${TACOTRON_EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate 0.0001 \
    --save_path "${SVS_SAVE}"

echo "  ✓ Tacotron 2 → ${SVS_SAVE}/tacotron2/"
echo ""

echo "═══ [4/5] Training FastSpeech 2 (${FASTSPEECH_EPOCHS} epochs) ═══"
python3 main.py train-svs \
    --model fastspeech2 \
    --dataset_dir "${DATASET_DIR}" \
    --epochs ${FASTSPEECH_EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate 0.0001 \
    --save_path "${SVS_SAVE}"

echo "  ✓ FastSpeech 2 → ${SVS_SAVE}/fastspeech2/"

# ═══════════════════════════════════════════════════════════════
echo ""
echo "═══ [5/5] Synthesizing test samples with BOTH models ═══"

TEST_TEXT="The quick brown fox jumps over the lazy dog."

echo "  → Tacotron 2..."
python3 main.py synthesize \
    --model_type tacotron2 \
    --model_path "${SVS_SAVE}/tacotron2/final_model.pt" \
    --input "${TEST_TEXT}" \
    --output ./output/test_tacotron2.wav

echo "  → FastSpeech 2..."
python3 main.py synthesize \
    --model_type fastspeech2 \
    --model_path "${SVS_SAVE}/fastspeech2/final_model.pt" \
    --input "${TEST_TEXT}" \
    --output ./output/test_fastspeech2.wav

# ═══════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  ✓ Training complete!                                   ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║                                                         ║"
echo "║  Tacotron 2:                                            ║"
echo "║    Report:  ${SVS_SAVE}/tacotron2/training_report.md"
echo "║    Curves:  ${SVS_SAVE}/tacotron2/learning_curves.png"
echo "║    Audio:   ./output/test_tacotron2.wav"
echo "║                                                         ║"
echo "║  FastSpeech 2:                                          ║"
echo "║    Report:  ${SVS_SAVE}/fastspeech2/training_report.md"
echo "║    Curves:  ${SVS_SAVE}/fastspeech2/learning_curves.png"
echo "║    Audio:   ./output/test_fastspeech2.wav"
echo "║                                                         ║"
echo "║  → Compare both WAVs, curves, and reports to choose!   ║"
echo "╚══════════════════════════════════════════════════════════╝"
