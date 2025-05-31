from transformers import MllamaForConditionalGeneration, AutoProcessor, AutoModelForVision2Seq
import torch

conv_mode = "llama_3" 

def evaluate_model_config(model, model_path, device="cuda"):
    if "cambrian" in model:
        raise NotImplementedError
    elif "llama" in model:
        processor = AutoProcessor.from_pretrained(model_path)
        model = MllamaForConditionalGeneration.from_pretrained(
            model_path,
        )
        return processor, model
    elif "SmolVLM" in model:
        processor = AutoProcessor.from_pretrained(model_path)
        try:
            model = AutoModelForVision2Seq.from_pretrained(
                model_path,
                trust_remote_code=True,  # Importante per SmolVLM
                torch_dtype=torch.bfloat16,  # Puoi adattare il tipo di dato se necessario
            )
            print(f"Successfully loaded SmolVLM model from {model_path}")
        except Exception as e:
            raise ValueError(f"Failed to load SmolVLM model: {e}")
        return processor, model
    else:
        raise ValueError(f"Unsupported model type: {model}")
