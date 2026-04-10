"""
T2I Text-to-Image — Full training pipeline in one file.

Includes: metrics (MSE/SSIM/LPIPS), early stopping, training loop,
validation, metric plots, and Captum XAI attention maps.

Usage:
    python train.py                          # Full training
    python train.py --generate-data-only     # Only generate dataset
    python train.py --max-epochs 5           # Smoke test
    python train.py --resume                 # Resume from checkpoint
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torchmetrics.image import StructuralSimilarityIndexMeasure
import lpips as lpips_lib
from captum.attr import LayerIntegratedGradients

from model import Text2ImageModel, MAX_TEXT_LEN, VOCAB_SIZE
from dataset import generate_dataset, create_dataloaders, tokenize_text, DATA_DIR

# ── Paths ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
PLOTS_DIR = OUTPUT_DIR / "plots"
XAI_DIR = OUTPUT_DIR / "xai"
MODELS_DIR = BASE_DIR.parent.parent / "Modelos"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Training defaults ─────────────────────────────────────────────────
MAX_EPOCHS = 200
PATIENCE = 15
MIN_DELTA = 0.001
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
LAMBDA_LPIPS = 0.1
LPIPS_WARMUP_EPOCHS = 10
TARGET_ERROR = 0.001


# ══════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════

class MetricsBundle:
    """MSE + LPIPS (for training loss), SSIM (for monitoring)."""
    def __init__(self, device):
        self.mse_fn = nn.MSELoss()
        self.ssim_fn = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
        self.lpips_fn = lpips_lib.LPIPS(net="vgg").to(device)
        for p in self.lpips_fn.parameters():
            p.requires_grad = False

    def compute(self, predicted, target):
        loss_mse = self.mse_fn(predicted, target)
        loss_lpips = self.lpips_fn(predicted.float(), target.float()).mean()
        total_loss = loss_mse + LAMBDA_LPIPS * loss_lpips
        with torch.no_grad():
            ssim_val = self.ssim_fn(predicted, target)
        return {
            "total_loss": total_loss,
            "mse": loss_mse.detach(),
            "ssim": ssim_val,
            "lpips": loss_lpips.detach(),
        }


# ══════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════

def train_model(model, train_loader, val_loader, max_epochs=MAX_EPOCHS, lr=LEARNING_RATE,
                target_error=TARGET_ERROR):
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None
    metrics = MetricsBundle(DEVICE)

    # Early stopping state
    best_loss = float("inf")
    patience_counter = 0
    stop_reason = ""

    history = {"train_loss": [], "val_loss": [], "mse": [], "ssim": [], "lpips": []}

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Training on {DEVICE} | Max epochs: {max_epochs}")
    print(f"  Patience: {PATIENCE} | Min delta: {MIN_DELTA}")
    if target_error is not None:
        print(f"  Target error: {target_error} (stop when val_loss ≤ this)")
    if DEVICE.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"{'='*60}\n")

    best_val_loss = float("inf")

    for epoch in range(max_epochs):
        # ── Train ──
        cur_lambda_lpips = LAMBDA_LPIPS if epoch >= LPIPS_WARMUP_EPOCHS else 0.0
        model.train()
        t_loss, t_n = 0.0, 0
        for text, real_images in train_loader:
            text, real_images = text.to(DEVICE), real_images.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)

            if DEVICE.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    fake = model(text)
                    res = metrics.compute(fake, real_images)
                scaler.scale(res["total_loss"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                torch.cuda.empty_cache()
            else:
                fake = model(text)
                res = metrics.compute(fake, real_images)
                res["total_loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            t_loss += res["total_loss"].item()
            t_n += 1

        avg_train = t_loss / max(t_n, 1)

        # ── Validate ──
        model.eval()
        v_totals = {"loss": 0, "mse": 0, "ssim": 0, "lpips": 0}
        v_n = 0
        with torch.no_grad():
            for text, real_images in val_loader:
                text, real_images = text.to(DEVICE), real_images.to(DEVICE)
                if DEVICE.type == "cuda":
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        fake = model(text)
                        res = metrics.compute(fake, real_images)
                else:
                    fake = model(text)
                    res = metrics.compute(fake, real_images)
                v_totals["loss"] += res["total_loss"].item()
                v_totals["mse"] += res["mse"].item()
                v_totals["ssim"] += res["ssim"].item()
                v_totals["lpips"] += res["lpips"].item()
                v_n += 1

        avg_val = {k: v / max(v_n, 1) for k, v in v_totals.items()}
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val["loss"])
        history["mse"].append(avg_val["mse"])
        history["ssim"].append(avg_val["ssim"])
        history["lpips"].append(avg_val["lpips"])

        cur_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1:03d} | Train: {avg_train:.4f} | "
            f"Val: {avg_val['loss']:.4f} | SSIM: {avg_val['ssim']:.4f} | "
            f"LPIPS: {avg_val['lpips']:.4f} | MSE: {avg_val['mse']:.4f} | "
            f"LR: {cur_lr:.2e}"
        )

        # Save best checkpoint
        val_loss = avg_val["loss"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, CHECKPOINT_DIR / "best_model.pth")

        # ── Early stopping ──
        val_loss = avg_val["loss"]
        if val_loss < best_loss - MIN_DELTA:
            best_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, CHECKPOINT_DIR / "best_model.pth")
            print(f"  ✓ Nuevo mejor modelo guardado (val_loss={best_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  [EarlyStopping] Sin mejora {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                stop_reason = f"Patience exhausted ({PATIENCE} epochs)"
                print(f"  >> {stop_reason}")
                break

        # ── Target error stop ──
        if target_error is not None and val_loss <= target_error:
            stop_reason = f"Target error reached (val_loss={val_loss:.6f} ≤ {target_error})"
            print(f"  🎯 {stop_reason}")
            break

        scheduler.step()

        if epoch + 1 >= max_epochs:
            stop_reason = f"Max epochs ({max_epochs}) reached"
            break

    # Save final model to Modelos/
    model_path = MODELS_DIR / "t2i_text2image.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved → {model_path}")

    if stop_reason:
        print(f"Stop reason: {stop_reason}")
    print(f"Training complete. {epoch+1} epochs.\n")

    plot_metrics(history)
    return history


# ══════════════════════════════════════════════════════════════════════
# Visualization
# ══════════════════════════════════════════════════════════════════════

def plot_metrics(history):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("T2I Training Metrics", fontsize=16, fontweight="bold")

    axes[0, 0].plot(epochs, history["train_loss"], label="Train", color="#e74c3c", linewidth=2)
    axes[0, 0].plot(epochs, history["val_loss"], label="Val", color="#3498db", linewidth=2)
    axes[0, 0].set_title("Loss (MSE + LPIPS)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, history["ssim"], color="#2ecc71", linewidth=2)
    axes[0, 1].set_title("SSIM (↑ better)")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, history["lpips"], color="#9b59b6", linewidth=2)
    axes[1, 0].set_title("LPIPS (↓ better)")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, history["mse"], color="#e67e22", linewidth=2)
    axes[1, 1].set_title("MSE (↓ better)")
    axes[1, 1].grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel("Epoch")

    plt.tight_layout()
    plot_path = PLOTS_DIR / "training_metrics.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Metrics plot → {plot_path}")


# ══════════════════════════════════════════════════════════════════════
# XAI — Captum Attention Maps
# ══════════════════════════════════════════════════════════════════════

def explain_text(model, text):
    """Generate per-character attribution map using Captum."""
    model.to(DEVICE).eval()
    XAI_DIR.mkdir(parents=True, exist_ok=True)

    tokens = torch.tensor([tokenize_text(text)], dtype=torch.long, device=DEVICE)

    def forward_fn(text_input):
        out = model(text_input)
        return out.sum(dim=(1, 2, 3))

    lig = LayerIntegratedGradients(forward_fn, model.text_encoder.embedding)
    attributions, _ = lig.attribute(inputs=tokens, return_convergence_delta=True, n_steps=50)

    scores = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
    chars = list(text)
    n = len(chars)
    scores = scores[:n]

    abs_scores = np.abs(scores)
    norm = abs_scores / abs_scores.max() if abs_scores.max() > 0 else abs_scores

    fig, ax = plt.subplots(figsize=(max(8, n * 0.7), 4))
    ax.bar(range(n), abs_scores, color=plt.cm.YlOrRd(norm), edgecolor="gray", alpha=0.85)
    ax.set_xticks(range(n))
    ax.set_xticklabels(chars, fontsize=12, fontweight="bold")
    ax.set_ylabel("Attribution Score")
    ax.set_title(f'Token Attribution — "{text}"', fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    safe = "".join(c if c.isalnum() else "_" for c in text[:20])
    path = XAI_DIR / f"attention_{safe}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[XAI] '{text}' → {path}")
    for i, ch in enumerate(chars):
        print(f"  '{ch}': {scores[i]:.4f}")

    return scores


# ══════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="T2I Training Pipeline")
    parser.add_argument("--generate-data-only", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--txt-dir", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument(
        "--target-error", type=float, default=None,
        metavar="THRESHOLD",
        help="Stop training when val_loss drops at or below this value (e.g. 0.001)",
    )
    args = parser.parse_args()

    print(f"Device: {DEVICE}")

    # Step 1: Dataset
    if not (DATA_DIR / "manifest.json").exists() or args.generate_data_only:
        txt_dir = Path(args.txt_dir) if args.txt_dir else None
        generate_dataset(num_samples=args.num_samples, text_source_dir=txt_dir)
        if args.generate_data_only:
            return

    # Step 2: DataLoaders
    train_loader, val_loader = create_dataloaders(batch_size=args.batch_size)

    # Step 3: Model
    model = Text2ImageModel()
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    if args.resume:
        ckpt = CHECKPOINT_DIR / "best_model.pth"
        if ckpt.exists():
            model.load_state_dict(
                torch.load(ckpt, map_location=DEVICE, weights_only=True)["model_state_dict"])
            print(f"Resumed from {ckpt}")

    # Step 4: Train
    train_model(model, train_loader, val_loader, max_epochs=args.max_epochs, lr=args.lr,
                target_error=args.target_error)

    # Step 5: XAI samples
    try:
        explain_text(model, "Santiago")
        explain_text(model, "Duvan")
        explain_text(model, "Hola Mundo")
    except Exception as e:
        print(f"[XAI] Skipped: {e}")

    print("Done!")


if __name__ == "__main__":
    main()
