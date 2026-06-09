# frozen_string_literal: true

module Trainers
  module Optimization
    # Linear warmup then linear decay to 0
    def self.get_linear_schedule_with_warmup(optimizer, num_warmup_steps:, num_training_steps:)
      lr_lambda = ->(current_step) {
        if current_step < num_warmup_steps
          current_step.to_f / [1, num_warmup_steps].max
        else
          remaining   = num_training_steps - current_step
          total_decay = num_training_steps - num_warmup_steps
          [0.0, remaining.to_f / [1.0, total_decay].max].max
        end
      }

      Torch::Optim::LRScheduler::LambdaLR.new(optimizer, lr_lambda)
    end

    # Linear warmup then cosine decay to 0
    def self.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps:, num_training_steps:, num_cycles: 0.5)
      lr_lambda = ->(current_step) {
        if current_step < num_warmup_steps
          current_step.to_f / [1, num_warmup_steps].max
        else
          progress = (current_step - num_warmup_steps).to_f /
                     [1, num_training_steps - num_warmup_steps].max
          [0.0, 0.5 * (1.0 + Math.cos(Math::PI * num_cycles * 2.0 * progress))].max
        end
      }

      Torch::Optim::LRScheduler::LambdaLR.new(optimizer, lr_lambda)
    end

    # Linear warmup then constant LR
    def self.get_constant_schedule_with_warmup(optimizer, num_warmup_steps:)
      lr_lambda = ->(current_step) {
        if current_step < num_warmup_steps
          current_step.to_f / [1, num_warmup_steps].max
        else
          1.0
        end
      }

      Torch::Optim::LRScheduler::LambdaLR.new(optimizer, lr_lambda)
    end

    # Dispatcher: pick scheduler by type symbol
    def self.create_scheduler(type, optimizer, num_warmup_steps:, num_training_steps:)
      case type
      when :linear
        get_linear_schedule_with_warmup(optimizer,
          num_warmup_steps: num_warmup_steps,
          num_training_steps: num_training_steps)
      when :cosine
        get_cosine_schedule_with_warmup(optimizer,
          num_warmup_steps: num_warmup_steps,
          num_training_steps: num_training_steps)
      when :constant
        get_constant_schedule_with_warmup(optimizer,
          num_warmup_steps: num_warmup_steps)
      else
        raise ArgumentError, "Unknown scheduler type: #{type}. Use :linear, :cosine, or :constant"
      end
    end
  end
end
