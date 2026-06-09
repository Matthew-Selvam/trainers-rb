# frozen_string_literal: true

module Trainers
  class Trainer
    attr_reader :model, :args, :train_dataset, :eval_dataset, :tokenizer,
                :data_collator, :optimizer, :lr_scheduler, :state, :control

    def initialize(
      model:,
      args: nil,
      train_dataset: nil,
      eval_dataset: nil,
      tokenizer: nil,
      data_collator: nil,
      compute_metrics: nil,
      callbacks: []
    )
      @model           = model
      @args            = args || TrainingArguments.new
      @train_dataset   = train_dataset
      @eval_dataset    = eval_dataset
      @tokenizer       = tokenizer
      @data_collator   = data_collator || DefaultDataCollator.new
      @compute_metrics = compute_metrics
      @state           = TrainerState.new
      @control         = TrainerControl.new

      all_callbacks = [PrinterCallback.new] + callbacks
      @callback_handler = CallbackHandler.new(all_callbacks)
    end

    def train
      device = @args.resolved_device
      @model.to(device)
      @model.train

      num_examples   = @train_dataset.size
      batch_size     = @args.per_device_train_batch_size
      steps_per_epoch = (num_examples.to_f / batch_size).ceil
      total_steps    = steps_per_epoch * @args.num_train_epochs

      @state.max_steps       = total_steps
      @state.num_train_epochs = @args.num_train_epochs

      @optimizer    = create_optimizer
      @lr_scheduler = create_scheduler(total_steps)

      @callback_handler.fire(:on_train_begin, @args, @state, @control)

      @args.num_train_epochs.times do |epoch|
        @state.epoch = epoch + 1
        @callback_handler.fire(:on_epoch_begin, @args, @state, @control)
        @model.train

        epoch_loss   = 0.0
        epoch_steps  = 0

        each_batch(@train_dataset, batch_size, shuffle: true) do |batch|
          @callback_handler.fire(:on_step_begin, @args, @state, @control)

          batch = move_to_device(batch, device)
          loss  = compute_loss(batch)

          scaled_loss = if @args.gradient_accumulation_steps > 1
                          loss / @args.gradient_accumulation_steps
                        else
                          loss
                        end

          scaled_loss.backward

          epoch_loss  += loss.item
          epoch_steps += 1
          @state.global_step += 1

          if @state.global_step % @args.gradient_accumulation_steps == 0
            clip_grad_norm!(@model.parameters, @args.max_grad_norm)
            @optimizer.step
            @lr_scheduler.step
            @optimizer.zero_grad
          end

          # Logging
          if should_log?
            logs = {
              loss:          epoch_loss / epoch_steps,
              learning_rate: current_lr,
              epoch:         @state.epoch
            }
            @state.log_history << logs.merge(step: @state.global_step)
            @callback_handler.fire(:on_log, @args, @state, @control, logs: logs)
          end

          # Step-based evaluation
          if @args.eval_strategy == :steps && @args.eval_steps &&
             @state.global_step % @args.eval_steps == 0
            metrics = evaluate
            @callback_handler.fire(:on_evaluate, @args, @state, @control, metrics: metrics)
          end

          # Step-based saving
          if @args.save_strategy == :steps && @args.save_steps &&
             @state.global_step % @args.save_steps == 0
            save_checkpoint
            @callback_handler.fire(:on_save, @args, @state, @control)
          end

          @callback_handler.fire(:on_step_end, @args, @state, @control)
          break if @control.should_training_stop || @control.should_epoch_stop
        end

        # Epoch-level logging
        epoch_avg_loss = epoch_steps > 0 ? epoch_loss / epoch_steps : 0.0
        logs = { loss: epoch_avg_loss, learning_rate: current_lr, epoch: @state.epoch }
        @state.log_history << logs.merge(step: @state.global_step)
        @callback_handler.fire(:on_log, @args, @state, @control, logs: logs)

        # Epoch-based evaluation
        if @args.eval_strategy == :epoch && @eval_dataset
          metrics = evaluate
          @callback_handler.fire(:on_evaluate, @args, @state, @control, metrics: metrics)
        end

        # Epoch-based saving
        if @args.save_strategy == :epoch
          save_checkpoint
          @callback_handler.fire(:on_save, @args, @state, @control)
        end

        @callback_handler.fire(:on_epoch_end, @args, @state, @control)
        @control.should_epoch_stop = false
        break if @control.should_training_stop
      end

      @callback_handler.fire(:on_train_end, @args, @state, @control)
      @state
    end

    def evaluate(eval_dataset: nil)
      dataset = eval_dataset || @eval_dataset
      raise ArgumentError, "No eval_dataset provided" unless dataset

      device = @args.resolved_device
      @model.eval

      all_preds  = []
      all_labels = []
      total_loss = 0.0
      total_steps = 0

      Torch.no_grad do
        each_batch(dataset, @args.per_device_eval_batch_size) do |batch|
          batch  = move_to_device(batch, device)
          labels = batch.delete(:labels) || batch.delete("labels")

          output = forward(batch)

          if labels
            logits = output.respond_to?(:logits) ? output.logits : output
            loss = Torch::NN::F.cross_entropy(logits, labels)
            total_loss += loss.item
            all_labels << labels.detach.cpu
          end
          total_steps += 1

          logits = output.respond_to?(:logits) ? output.logits : output
          all_preds << logits.detach.cpu
        end
      end

      @model.train

      metrics = {}
      metrics[:eval_loss] = total_loss / total_steps if total_steps > 0

      if @compute_metrics && all_preds.any? && all_labels.any?
        preds  = Torch.cat(all_preds)
        labels = Torch.cat(all_labels)
        eval_pred = EvalPrediction.new(predictions: preds, label_ids: labels)
        custom_metrics = @compute_metrics.call(eval_pred)
        metrics.merge!(custom_metrics)
      end

      metrics
    end

    def predict(test_dataset)
      device = @args.resolved_device
      @model.eval

      all_preds = []
      Torch.no_grad do
        each_batch(test_dataset, @args.per_device_eval_batch_size) do |batch|
          batch  = move_to_device(batch, device)
          output = forward(batch)
          logits = output.respond_to?(:logits) ? output.logits : output
          all_preds << logits.detach.cpu
        end
      end

      Torch.cat(all_preds)
    end

    def save_model(output_dir = nil)
      output_dir ||= @args.output_dir
      SaveUtils.save_pretrained(@model, @tokenizer, output_dir, training_args: @args)
    end

    private

    def compute_loss(batch)
      labels = batch.delete(:labels) || batch.delete("labels")

      # Try passing labels to the model (some models compute loss internally)
      output = begin
        forward(labels ? batch.merge(labels: labels) : batch)
      rescue => e
        # If the model doesn't support labels kwarg (e.g. transformers-rb Todo),
        # fall back to forward without labels + external loss
        if e.message.include?("Todo") || e.message.include?("not implemented")
          forward(batch)
        else
          raise
        end
      end

      # Restore labels to batch for downstream use
      batch[:labels] = labels if labels

      if output.respond_to?(:loss) && output.loss
        output.loss
      elsif labels
        logits = output.respond_to?(:logits) ? output.logits : output
        Torch::NN::F.cross_entropy(logits, labels)
      else
        raise "Model did not return a loss and no labels found in batch. " \
              "Either pass labels in your dataset or use a model that computes loss."
      end
    end

    def forward(batch)
      if batch.is_a?(Hash)
        # Filter to only keys the model accepts, using symbol keys
        @model.call(**batch)
      else
        @model.call(batch)
      end
    end

    def create_optimizer
      Optimization.create_optimizer(@model, @args)
    end

    def create_scheduler(total_steps)
      warmup_steps = if @args.warmup_steps > 0
                       @args.warmup_steps
                     elsif @args.warmup_ratio > 0
                       (total_steps * @args.warmup_ratio).to_i
                     else
                       0
                     end

      Optimization.create_scheduler(
        @args.lr_scheduler_type,
        @optimizer,
        num_warmup_steps:    warmup_steps,
        num_training_steps:  total_steps
      )
    end

    def each_batch(dataset, batch_size, shuffle: false)
      indices = (0...dataset.size).to_a
      indices.shuffle! if shuffle

      (0...dataset.size).step(batch_size) do |start|
        batch_indices = indices[start, batch_size]
        next if batch_indices.nil? || batch_indices.empty?

        features = batch_indices.map { |i| dataset[i] }
        batch = @data_collator.call(features)
        yield batch
      end
    end

    def move_to_device(batch, device)
      batch.each_with_object({}) do |(key, value), result|
        result[key] = if value.is_a?(Torch::Tensor)
                        value.to(device)
                      else
                        value
                      end
      end
    end

    def clip_grad_norm!(parameters, max_norm)
      params = parameters.select { |p| p.grad }
      return 0.0 if params.empty?

      total_norm_sq = 0.0
      params.each do |p|
        total_norm_sq += p.grad.data.norm(2).item ** 2
      end
      total_norm = Math.sqrt(total_norm_sq)

      clip_coef = max_norm / (total_norm + 1e-6)
      if clip_coef < 1.0
        params.each { |p| p.grad.data.mul!(clip_coef) }
      end

      total_norm
    end

    def current_lr
      @optimizer.param_groups.first[:lr]
    end

    def should_log?
      return true if @args.logging_first_step && @state.global_step == 1
      @state.global_step % @args.logging_steps == 0
    end

    def save_checkpoint
      dir = File.join(@args.output_dir, "checkpoint-#{@state.global_step}")
      save_model(dir)
      cleanup_checkpoints if @args.save_total_limit
    end

    def cleanup_checkpoints
      return unless @args.save_total_limit

      checkpoints = Dir.glob(File.join(@args.output_dir, "checkpoint-*"))
                       .sort_by { |d| d[/checkpoint-(\d+)/, 1].to_i }

      while checkpoints.length > @args.save_total_limit
        old = checkpoints.shift
        FileUtils.rm_rf(old)
      end
    end
  end
end
