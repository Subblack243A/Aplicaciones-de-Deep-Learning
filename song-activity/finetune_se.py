import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml

class SEBrain(sb.Brain):
    """
    Subclase personalizada de SpeechBrain para el entrenamiento de Speech Enhancement.
    """
    def compute_forward(self, batch, stage):
        """
        Paso Forward: Toma el audio ruidoso y predice la "máscara" o el audio limpio.
        """
        batch = batch.to(self.device)
        noisy_wavs, lens = batch.wav_noisy
        est_sources = self.modules.model(noisy_wavs)
        return est_sources[:, :, 0]

    def compute_objectives(self, predictions, batch, stage):
        """
        Calcula la pérdida (Loss) comparando la estimación con el audio limpio en el dominio del tiempo.
        Para Speech Enhancement es común usar SNR (Signal-to-Noise Ratio) Loss o siSDR.
        """
        batch = batch.to(self.device)
        clean_wavs, lens = batch.wav_clean
        loss = -sb.nnet.losses.snr_loss(clean_wavs, predictions, lens)
        return loss.mean()
    
    def on_stage_end(self, stage, stage_loss, epoch):
        """
        Se ejecuta al final de cada época o fase de validación.
        """
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        else:
            print(f"Epoch {epoch}: Train Loss: {self.train_loss:.4f} | Validation Loss: {stage_loss:.4f}")

def dataio_prep(dataset_json_path):
    """
    Prepara el DataLoader dinámico de SpeechBrain a partir del JSON generado.
    """
    import json
    if not os.path.exists(dataset_json_path):
         raise FileNotFoundError(f"No se encontró el catálogo: {dataset_json_path}")
         
    with open(dataset_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    flattened_data = {}
    for key, val in data.items():
        flattened_data[key] = val
        
    dataset = sb.dataio.dataset.DynamicItemDataset(flattened_data)

    @sb.utils.data_pipeline.takes("wav_clean")
    @sb.utils.data_pipeline.provides("wav_clean")
    def audio_pipeline_clean(wav_clean):
        sig = sb.dataio.dataio.read_audio(wav_clean) 
        return sig
    
    @sb.utils.data_pipeline.takes("wav_noisy")
    @sb.utils.data_pipeline.provides("wav_noisy")
    def audio_pipeline_noisy(wav_noisy):
        sig = sb.dataio.dataio.read_audio(wav_noisy)
        return sig

    sb.dataio.dataset.add_dynamic_item(dataset, audio_pipeline_clean)
    sb.dataio.dataset.add_dynamic_item(dataset, audio_pipeline_noisy)

    sb.dataio.dataset.set_output_keys(dataset, ["id", "wav_clean", "wav_noisy"])

    return dataset

def main():
    """
    Punto de entrada para el Fine-Tuning de la red neuronal.
    """
    save_folder = "./results/se_finetuned"
    os.makedirs(save_folder, exist_ok=True)
    
    print("\n--- Speech Enhancement Fine-Tuning pipeline ---")
    
    print("[1/3] Descargando arquitectura SepFormer pre-entrenada...")
    from speechbrain.inference.separation import SepformerSeparation
    try:
         pretrained_separator = SepformerSeparation.from_hparams(
             source="speechbrain/sepformer-wham16k",
             savedir="pretrained_models/sepformer-wham16k",
             run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
         )
    except Exception as e:
         print(f"Error cargando el modelo pre-entrenado. Asegúrate de tener conexión a internet. {e}")
         return
         
    print("[2/3] Preparando DataLoaders...")
    dataset_path = "./se_dataset/dataset.json"
    try:
         train_data = dataio_prep(dataset_path)
    except FileNotFoundError:
         print(f"Error: Debes ejecutar primero 'python synthetic_dataset.py' para crear {dataset_path}")
         return
         
    learning_rate = 1e-5
    optimizer = lambda opt_model: torch.optim.Adam(opt_model.parameters(), lr=learning_rate)
    
    modules = {"model": pretrained_separator.mods.separator}
    
    print(f"\n[3/3] Iniciando entrenamiento por 3 épocas. Learning Rate: {learning_rate}")
    se_brain = SEBrain(
        modules=modules,
        opt_class=optimizer,
        hparams={"epoch_counter": sb.utils.epoch_loop.EpochCounter(limit=3)},
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )
    
    se_brain.fit(
        epoch_counter=se_brain.hparams["epoch_counter"],
        train_set=train_data,
        valid_set=train_data,
        train_loader_kwargs={"batch_size": 2, "drop_last": False},
        valid_loader_kwargs={"batch_size": 2, "drop_last": False}
    )
    
    model_save_path = os.path.join(save_folder, "finetuned_separator.ckpt")
    torch.save(se_brain.modules.state_dict(), model_save_path)
    print(f"\n¡Entrenamiento Completado! Pesos actualizados guardados en {model_save_path}")

    # ── Visualization: waveform comparison + loss curve ──────────────────────
    print("\nGenerando visualización del proceso de entrenamiento...")
    try:
        import torchaudio

        # Load a sample from the dataset for visualization
        sample_clean = list(train_data.data.values())[0]["wav_clean"]
        sample_noisy = list(train_data.data.values())[0]["wav_noisy"]

        clean_wav, sr_c = torchaudio.load(sample_clean)
        noisy_wav, sr_n = torchaudio.load(sample_noisy)

        # Run through the fine-tuned model to get enhanced output
        se_brain.modules.model.eval()
        with torch.no_grad():
            enhanced = se_brain.modules.model(noisy_wav.unsqueeze(0))
            enhanced = enhanced[:, :, 0].squeeze(0)

        # Build figure with 3 panels
        fig, axes = plt.subplots(3, 1, figsize=(14, 9))
        fig.suptitle("Speech Enhancement: Fine-Tuning Results", fontsize=14, fontweight='bold')

        t_noisy = np.linspace(0, noisy_wav.shape[-1] / sr_n, noisy_wav.shape[-1])
        t_clean = np.linspace(0, clean_wav.shape[-1] / sr_c, clean_wav.shape[-1])
        t_enh   = np.linspace(0, enhanced.shape[-1] / sr_n, enhanced.shape[-1])

        axes[0].plot(t_noisy, noisy_wav[0].numpy(), color='#e74c3c', linewidth=0.5)
        axes[0].set_title("Input: Noisy Audio (MP4 → WAV)")
        axes[0].set_ylabel("Amplitude")
        axes[0].set_xlabel("Time (s)")

        axes[1].plot(t_enh, enhanced.numpy(), color='#27ae60', linewidth=0.5)
        axes[1].set_title("Output: Enhanced Vocals (Fine-Tuned SepFormer)")
        axes[1].set_ylabel("Amplitude")
        axes[1].set_xlabel("Time (s)")

        axes[2].plot(t_clean, clean_wav[0].numpy(), color='#2980b9', linewidth=0.5)
        axes[2].set_title("Reference: Ground Truth Clean Audio")
        axes[2].set_ylabel("Amplitude")
        axes[2].set_xlabel("Time (s)")

        plt.tight_layout()
        plot_path = os.path.join(save_folder, "training_summary.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Visualización guardada en '{plot_path}'")
    except Exception as e:
        print(f"[Visualización omitida]: {e}")

if __name__ == "__main__":
    main()
