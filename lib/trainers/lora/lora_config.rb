# frozen_string_literal: true

module Trainers
  class LoraConfig
    attr_accessor :r, :lora_alpha, :lora_dropout, :target_modules,
                  :bias, :task_type

    DEFAULTS = {
      r:              8,
      lora_alpha:     16,
      lora_dropout:   0.0,
      target_modules: ["query", "value"],  # or :all_linear
      bias:           :none,               # :none, :all, :lora_only
      task_type:      :sequence_classification
    }.freeze

    def initialize(**kwargs)
      DEFAULTS.each do |key, default|
        value = kwargs.fetch(key, default)
        instance_variable_set(:"@#{key}", value)
      end
    end

    def scaling
      @lora_alpha.to_f / @r
    end

    def to_h
      DEFAULTS.keys.each_with_object({}) do |key, hash|
        hash[key] = send(key)
      end
    end
  end
end
