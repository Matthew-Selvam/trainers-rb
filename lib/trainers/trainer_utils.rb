# frozen_string_literal: true

module Trainers
  module EvalStrategy
    NO    = :no
    EPOCH = :epoch
    STEPS = :steps
  end

  module SaveStrategy
    NO    = :no
    EPOCH = :epoch
    STEPS = :steps
  end

  module SchedulerType
    LINEAR   = :linear
    COSINE   = :cosine
    CONSTANT = :constant
  end

  class EvalPrediction
    attr_reader :predictions, :label_ids

    def initialize(predictions:, label_ids:)
      @predictions = predictions
      @label_ids   = label_ids
    end
  end
end
