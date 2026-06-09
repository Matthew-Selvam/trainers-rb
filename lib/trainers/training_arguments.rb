# frozen_string_literal: true

module Trainers
  class TrainingArguments
    DEFAULTS = {
      output_dir:                    "./output",
      num_train_epochs:              3,
      per_device_train_batch_size:   8,
      per_device_eval_batch_size:    8,
      learning_rate:                 5e-5,
      weight_decay:                  0.0,
      adam_beta1:                    0.9,
      adam_beta2:                    0.999,
      adam_epsilon:                  1e-8,
      max_grad_norm:                 1.0,
      gradient_accumulation_steps:   1,
      warmup_steps:                  0,
      warmup_ratio:                  0.0,
      lr_scheduler_type:             :linear,
      eval_strategy:                 :no,
      eval_steps:                    nil,
      save_strategy:                 :epoch,
      save_steps:                    500,
      save_total_limit:              nil,
      logging_steps:                 500,
      logging_first_step:            false,
      seed:                          42,
      device:                        nil,
      no_mps:                        false,
      dataloader_drop_last:          false,
      label_names:                   ["labels"]
    }.freeze

    DEFAULTS.each_key do |key|
      attr_accessor key
    end

    def initialize(**kwargs)
      DEFAULTS.each do |key, default|
        value = kwargs.fetch(key, default)
        instance_variable_set(:"@#{key}", value)
      end

      unknown = kwargs.keys - DEFAULTS.keys
      raise ArgumentError, "Unknown arguments: #{unknown.join(', ')}" unless unknown.empty?
    end

    def resolved_device
      return @device if @device

      if !@no_mps && defined?(Torch::Backends::MPS) && Torch::Backends::MPS.available?
        Torch.device("mps")
      else
        Torch.device("cpu")
      end
    end

    def to_h
      DEFAULTS.keys.each_with_object({}) do |key, hash|
        hash[key] = send(key)
      end
    end
  end
end
