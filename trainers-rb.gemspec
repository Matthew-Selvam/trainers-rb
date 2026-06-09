Gem::Specification.new do |spec|
  spec.name          = "trainers-rb"
  spec.version       = "0.1.0"
  spec.authors       = ["Vishwajeetsingh Desurkar"]
  spec.email         = ["selectus2@users.noreply.rubygems.org"]
  spec.summary       = "Fine-tune transformer models in Ruby"
  spec.description   = "Training loop, LoRA, and optimization utilities for fine-tuning " \
                        "HuggingFace transformer models using torch-rb and transformers-rb. " \
                        "Supports full fine-tuning, LoRA adapters, learning rate scheduling, " \
                        "callbacks, and model serialization via safetensors."
  spec.homepage      = "https://github.com/trainers-rb/trainers-rb"
  spec.license       = "MIT"
  spec.required_ruby_version = ">= 3.1.0"

  spec.metadata = {
    "homepage_uri"    => spec.homepage,
    "source_code_uri" => spec.homepage,
    "changelog_uri"   => "#{spec.homepage}/blob/main/CHANGELOG.md"
  }

  spec.files = Dir["lib/**/*.rb", "LICENSE.txt", "README.md", "CHANGELOG.md"]
  spec.require_paths = ["lib"]

  spec.add_dependency "torch-rb",        ">= 0.17.1"
  spec.add_dependency "transformers-rb", ">= 0.2.0"
  spec.add_dependency "safetensors",     ">= 0.1.1"
  spec.add_dependency "tokenizers",      ">= 0.5.3"

  spec.add_development_dependency "rake",     "~> 13.0"
  spec.add_development_dependency "minitest", "~> 5.0"
end
