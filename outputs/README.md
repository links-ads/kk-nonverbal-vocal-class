# Training Outputs Directory

This directory contains all outputs generated during model training and evaluation. The structure is organized by project/experiment grouping, then by dataset, and finally by specific model configurations:

```
outputs/
├── clic_2025/                    # Project/experiment grouping
│   ├── metrics.csv               # Aggregated metrics across experiments
│   ├── *_weights.png             # Model weight analysis visualizations
│   ├── asvp_esd/                # Dataset-specific results
│   │   ├── whisper_base_lora/    # Model + PEFT method combination
│   │   ├── whisper_tiny_frozen/  # Different configurations
│   │   └── wav2vec2_base_lora/   # Various model architectures
│   ├── recanvo/
│   │   ├── whisper_base_lora_012/  # LoRA applied to layers 0,1,2
│   │   │   └── checkpoint-7752/    # Training checkpoints
│   │   │       ├── config.json     # Model configuration
│   │   │       ├── model.safetensors # Model weights
│   │   │       ├── trainer_state.json # Training state
│   │   │       └── training_args.bin  # Training arguments
│   │   ├── whisper_small_lora_7-12/   # LoRA applied to layers 7-12
│   │   └── hubert_base_frozen/        # Different model types
│   ├── cnvve/
│   ├── vivae/
│   ├── nonverbal_vocalization_dataset/
│   └── asvp_esd_babies/
```

## What Gets Saved Here

- **Model Checkpoints**: Training checkpoints with model weights (`model.safetensors`), optimizer state, and scheduler state
- **Configuration Files**: Complete experiment configurations (`config.json`, `training_args.bin`)
- **Training State**: Detailed training progress and metrics (`trainer_state.json`)
- **Aggregated Results**: Cross-experiment metrics and analysis (`metrics.csv`)
- **Visualizations**: Model weight analysis plots (`*_weights.png`)
- **Experiment Organization**: Results grouped by project, dataset, and model configuration

## Important Notes

- **Not tracked by Git**: This directory should be added to `.gitignore` to avoid committing large model files
- **Organized by Experiment**: Each training run creates its own subdirectory based on the experiment name
- **Automatic Creation**: The training script automatically creates the necessary subdirectories

## Example Structure

After running `train.py experiments_configs/recanvo/whisper_base_lora_012.py`, you'll find:

```
outputs/clic_2025/recanvo/whisper_base_lora_012/
└── checkpoint-7752/
    ├── config.json           # Model and training configuration
    ├── model.safetensors     # Trained model weights
    ├── optimizer.pt          # Optimizer state for resuming training
    ├── scheduler.pt          # Learning rate scheduler state
    ├── trainer_state.json    # Training metrics and progress
    ├── training_args.bin     # Training arguments used
    └── rng_state.pth        # Random number generator state
```

The checkpoint number (e.g., `7752`) corresponds to the training step when the checkpoint was saved.

## Model Naming Convention

The directory names follow the pattern: `{model_type}_{model_size}_{peft_method}[_layers]`

Examples:
- `whisper_base_lora_012`: Whisper base model with LoRA on layers 0, 1, 2
- `whisper_small_lora_7-12`: Whisper small model with LoRA on layers 7-12
- `wav2vec2_base_frozen`: Wav2Vec2 base model with frozen weights (linear probing)
- `hubert_large_adapter`: HuBERT large model with adapter fine-tuning
