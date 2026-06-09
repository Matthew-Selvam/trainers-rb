# frozen_string_literal: true

module Trainers
  class Dataset
    include Enumerable

    attr_reader :data

    def initialize(data)
      @data = data
    end

    def [](index)
      @data[index]
    end

    def size
      @data.size
    end
    alias_method :length, :size

    def each(&block)
      @data.each(&block)
    end
  end
end
