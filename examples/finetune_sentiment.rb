#!/usr/bin/env ruby
# frozen_string_literal: true

# Example: Fine-tune DistilBERT for sentiment analysis using trainers-rb
#
# This demonstrates the full workflow:
#   1. Load a pre-trained model + tokenizer from HuggingFace
#   2. Optionally apply LoRA for parameter-efficient fine-tuning
#   3. Train on a simple sentiment dataset
#   4. Evaluate and save the fine-tuned model
#
# Usage:
#   ruby examples/finetune_sentiment.rb
#   LORA=1 ruby examples/finetune_sentiment.rb   # with LoRA

$LOAD_PATH.unshift(File.join(__dir__, "..", "lib"))
require "trainers-rb"

USE_LORA = ENV["LORA"] == "1"

# ---------------------------------------------------------------------------
# 1. Load model and tokenizer
# ---------------------------------------------------------------------------
puts "Loading model and tokenizer..."
model, tokenizer = Trainers.from_pretrained(
  "distilbert-base-uncased",
  task: :sequence_classification,
  num_labels: 2
)

# ---------------------------------------------------------------------------
# 2. Apply LoRA (optional)
# ---------------------------------------------------------------------------
if USE_LORA
  puts "\nApplying LoRA adapters..."
  lora_config = Trainers::LoraConfig.new(
    r: 8,
    lora_alpha: 16,
    lora_dropout: 0.1,
    target_modules: ["q_lin", "v_lin"],  # DistilBERT naming
    bias: :none
  )
  Trainers::LoraModel.apply(model, lora_config)
else
  puts "\nFull fine-tuning (all parameters trainable)"
end

# ---------------------------------------------------------------------------
# 3. Prepare dataset
# ---------------------------------------------------------------------------
train_texts = [
  "This movie was absolutely fantastic!",
  "I loved every minute of this film",
  "Great acting and wonderful story",
  "Best movie I have seen this year",
  "Highly recommend this to everyone",
  "What a masterpiece of cinema",
  "Terrible movie, waste of time",
  "I hated this film so much",
  "Worst acting I have ever seen",
  "Do not waste your money on this",
  "Absolutely awful, zero stars",
  "The most boring film ever made",
]
train_labels = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]

val_texts  = [
  "A wonderful experience, loved it",
  "Dreadful and unwatchable garbage",
  "Outstanding performances throughout",
  "A complete disaster from start to end",
]
val_labels = [1, 0, 1, 0]

def tokenize_dataset(texts, labels, tokenizer)
  data = texts.zip(labels).map do |text, label|
    encoded = tokenizer.(text, padding: false, truncation: true, max_length: 128)
    {
      input_ids:      encoded["input_ids"],
      attention_mask: encoded["attention_mask"],
      labels:         label
    }
  end
  Trainers::Dataset.new(data)
end

puts "\nTokenizing datasets..."
train_dataset = tokenize_dataset(train_texts, train_labels, tokenizer)
val_dataset   = tokenize_dataset(val_texts,   val_labels,   tokenizer)
puts "Train: #{train_dataset.size} examples  |  Val: #{val_dataset.size} examples"

# ---------------------------------------------------------------------------
# 4. Configure training
# ---------------------------------------------------------------------------
args = Trainers::TrainingArguments.new(
  output_dir:                  "./output/sentiment",
  num_train_epochs:            5,
  per_device_train_batch_size: 4,
  per_device_eval_batch_size:  4,
  learning_rate:               USE_LORA ? 1e-4 : 2e-5,
  weight_decay:                0.01,
  eval_strategy:               :epoch,
  save_strategy:               :epoch,
  save_total_limit:            2,
  logging_steps:               1,
  logging_first_step:          true,
  warmup_steps:                2,
  lr_scheduler_type:           :cosine
)

# ---------------------------------------------------------------------------
# 5. Create trainer and train
# ---------------------------------------------------------------------------
trainer = Trainers::Trainer.new(
  model:          model,
  args:           args,
  train_dataset:  train_dataset,
  eval_dataset:   val_dataset,
  tokenizer:      tokenizer,
  data_collator:  Trainers::DataCollatorWithPadding.new(tokenizer: tokenizer),
  compute_metrics: ->(eval_pred) {
    preds   = eval_pred.predictions.argmax(1)
    correct = preds.eq(eval_pred.label_ids).sum.item
    { accuracy: correct.to_f / eval_pred.label_ids.size(0) }
  }
)

puts "\n--- Starting training ---\n\n"
trainer.train

# ---------------------------------------------------------------------------
# 6. Save
# ---------------------------------------------------------------------------
puts "\nSaving model..."
if USE_LORA
  Trainers::LoraModel.save_adapters(model, "./output/sentiment-lora")
  # Optionally merge and save full model:
  # Trainers::LoraModel.merge(model)
  # trainer.save_model("./output/sentiment-merged")
else
  trainer.save_model("./output/sentiment-full")
end

puts "\nDone!"
