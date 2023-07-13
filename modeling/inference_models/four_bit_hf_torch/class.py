from __future__ import annotations

from modeling.inference_models.generic_hf_torch.model_backend import HFTorchModelBackend

model_backend_name = "Huggingface 4-Bit"
model_backend_type = "Huggingface" #This should be a generic name in case multiple model backends are compatible (think Hugging Face Custom and Basic Hugging Face)

class model_backend(HFTorchModelBackend):
    def __init__(self) -> None:
        super().__init__()
        self.load_in_4bit = True

    def _load(self, save_model: bool, initial_load: bool) -> None:
        self.lazy_load = False
        return super()._load(save_model, initial_load)