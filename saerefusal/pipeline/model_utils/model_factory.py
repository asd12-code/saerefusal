from pipeline.model_utils.model_base import ModelBase

def construct_model_base(model_path: str, device) -> ModelBase:

    if 'gemma-3-4b-it' in model_path.lower() or 'gemma-3-12b-it' in model_path.lower() or 'gemma-3-27b-it' in model_path.lower():
        from pipeline.model_utils.gemma3_model import Gemma3Model
        return Gemma3Model(model_path, device)
    elif 'gemma' in model_path.lower():
        from pipeline.model_utils.gemma_model import GemmaModel
        return GemmaModel(model_path, device) 
    else:
        raise ValueError(f"Unknown model family: {model_path}")
