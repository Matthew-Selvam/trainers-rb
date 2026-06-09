# Changelog

## 0.1.0 (2026-06-07)

- Initial release
- Trainer class with train/evaluate/predict
- TrainingArguments configuration
- AdamW optimizer with weight decay param groups
- Linear, cosine, and constant learning rate schedulers with warmup
- LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
- LoRA adapter save/load via safetensors
- LoRA merge for inference deployment
- DataCollatorWithPadding for dynamic batch padding
- Callback system with PrinterCallback and EarlyStoppingCallback
- Model saving via safetensors
- CPU and MPS (Apple Silicon) device support
- Gradient accumulation and gradient clipping
