from __future__ import annotations

from .model_backend import HFTorchModelBackend

model_backend_name = "Huggingface"
model_backend_type = "Huggingface" #This should be a generic name in case multiple model backends are compatible (think Hugging Face Custom and Basic Hugging Face)

class model_backend(HFTorchModelBackend):
    pass