import os
import torch
import transformers
from peft import LoraConfig, get_peft_model
import ast
from transformers import AutoProcessor, BitsAndBytesConfig, MllamaForConditionalGeneration, AutoModelForVision2Seq, Idefics3ForConditionalGeneration
from training.trainer import LLamaVTrainer
from training.data import make_supervised_data_module
from training.params import DataArguments, ModelArguments, TrainingArguments
from training.train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, safe_save_model_for_hf_trainer
import pathlib

local_rank = None

from transformers import TrainerCallback
from typing import Dict, List
import torch.distributed as dist

# Stampare le GPU visibili
visible_devices = torch.cuda.device_count()
print(f"Number of visible GPUs: {visible_devices}")

# Stampare informazioni su ciascuna GPU
for i in range(visible_devices):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
    print(f"Memory Reserved: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")

# Stampare la GPU attualmente in uso
if torch.cuda.is_available():
    current_device = torch.cuda.current_device()
    print(f"Current device: {current_device}")
    print(f"Device name: {torch.cuda.get_device_name(current_device)}")
else:
    print("No GPU is currently available.")

def rank0_print(*args):
    if local_rank == 0 or local_rank == '0' or local_rank is None:
        print(*args)

def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=True):
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)
    
    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        rank0_print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names

def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad
        
def configure_vision_tower(model, training_args, compute_dtype, device):
    print(f"DEBUG: Model type: {type(model).__name__}")
    
    if hasattr(model, 'model'):
        base_model = model.model
        print("DEBUG: Checking base model components:")
        for name, module in base_model.named_children():
            print(f"- {name}: {type(module).__name__}")
        
        if hasattr(base_model, 'vision_model'):
            vision_tower = base_model.vision_model
            print("Using vision_model from base model")
        elif hasattr(base_model, 'vision_encoder'):
            vision_tower = base_model.vision_encoder
            print("Using vision_encoder from base model")
        elif hasattr(base_model, 'vision_tower'):
            vision_tower = base_model.vision_tower
            print("Using vision_tower from base model")
        elif hasattr(base_model, 'visual'):
            vision_tower = base_model.visual
            print("Using visual module from base model")
        else:
            raise ValueError(f"No vision component found in base model. Available components: {list(base_model.named_children())}")
    else:
        raise ValueError(f"Model does not have a base 'model' attribute. Available attributes: {dir(model)}")

    if vision_tower is None:
        raise ValueError("Vision tower is None after initialization")
        
    vision_tower.to(dtype=compute_dtype, device=device)
    return vision_tower

def configure_llm(model, training_args):
    print("LOCALIZE: ENTERING configure_llm() in train.py")
    # Per gestire diversi tipi di modelli, incluso Idefics3ForConditionalGeneration
    if hasattr(model, 'language_model'):
        llm_params = model.language_model.parameters()
    elif hasattr(model, 'text_model'):
        llm_params = model.text_model.parameters() 
    elif hasattr(model, 'model'):
        llm_params = model.model.parameters()
    else:
        # Se non troviamo gli attributi comuni, stampa un avviso e usa il modello intero
        print(f"Avviso: Non è stato possibile trovare il modello linguistico specifico in {type(model).__name__}. Utilizzando tutti i parametri.")
        llm_params = [p for n, p in model.named_parameters() 
                    if 'vision_model' not in n and 'multi_modal_projector' not in n]
    
    set_requires_grad(llm_params, not training_args.freeze_llm)

def train():
    print("LOCALIZE: ENTERING train() in train.py")
    global local_rank
    import wandb
    os.environ["WANDB_PROJECT"] = "sft"
    # wandb.init(project=os.environ["WANDB_PROJECT"])
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    assert not (training_args.lora_enable and training_args.freeze_llm), 'When using LoRA, the LLM should not be frozen. If you want to freeze the LLM, please disable LoRA.'

    if not training_args.lora_enable:
        assert not training_args.vision_lora, \
            "Error: training_args.lora_enable is not enabled, but training_args.vision_lora is enabled."

    else:
        if training_args.lora_namespan_exclude is not None:
            training_args.lora_namespan_exclude = ast.literal_eval(training_args.lora_namespan_exclude)
        else:
            training_args.lora_namespan_exclude = ["multi_modal_projector"]

        if not training_args.vision_lora:
            training_args.lora_namespan_exclude += ["vision_model", "multi_modal_projector"]

    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4,8]:
        bnb_model_from_pretrained_args.update(dict(
            device_map={"":training_args.device},
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=training_args.bits==4,
                load_in_8bit=training_args.bits==8,
                llm_int8_skip_modules=["multi_modal_projector", "vision_model"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type,
            )
        ))

    try:
        print("Using SmolVLM model...")
        model = AutoModelForVision2Seq.from_pretrained(
            model_args.model_id,
            torch_dtype=compute_dtype,
            cache_dir=training_args.cache_dir,
            attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "eager",
            **bnb_model_from_pretrained_args
        ).to(training_args.device)
        for p in model.parameters():
            p.requires_grad = True
        print("[DEBUG] Forzato requires_grad=True su tutti i parametri del modello")
    except Exception as e:
        print(f"Failed to load as SmolVLM: {e}")
        print("Falling back to AutoModelForVision2Seq")
        print("Loading SmolVLM-250M model...")
        model = AutoModelForVision2Seq.from_pretrained(
            model_args.model_id,
            torch_dtype=compute_dtype,
            cache_dir=training_args.cache_dir,
            trust_remote_code=True,  # Important for SmolVLM
            use_flash_attention_2=not training_args.disable_flash_attn2,
            **bnb_model_from_pretrained_args
        ).to(training_args.device)
        print(f"Loaded model type: {type(model).__name__}")
        print(f"Model architecture: {model.config.model_type}")
    
    # I set a hidden size for temporary use. This is to use the deepspeed.
    # I will find a proper way later.
    model.config.hidden_size = model.config.text_config.hidden_size
    model.config.text_config.use_cache = False

    if training_args.bits in [4,8]:
        model.config.torch_dtype = (torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing, gradient_checkpointing_kwargs={"use_reentrant": False})
    
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    if training_args.lora_enable:
        lora_namespan_exclude = training_args.lora_namespan_exclude
        peft_config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_target_linear_names(model, lora_namespan_exclude=lora_namespan_exclude, num_lora_modules=training_args.num_lora_modules),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA to the model...")
        model = get_peft_model(model, peft_config)

    processor = AutoProcessor.from_pretrained(model_args.model_id)
    
    # use unk rather than eos token to prevent endless generation
    processor.tokenizer.padding_side = 'right'

    model.config.tokenizer_model_max_length = processor.tokenizer.model_max_length
    model.config.tokenizer_padding_side = processor.tokenizer.padding_side
    
    # When using LoRA, the model is rapped once more.
    if training_args.lora_enable:
        model_to_configure = model.model
    else:
        model_to_configure = model
        configure_llm(model, training_args)
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        print(f"[DEBUG] Trainable parameters: {len(trainable)}")
        print("Examples:", trainable[:5])
        assert len(trainable) > 0, "🚨 Nessun parametro addestrabile! Modello completamente frozen."
    
    
    
    if not training_args.vision_lora:
        configure_vision_tower(model_to_configure, training_args, compute_dtype, training_args.device)
        
    model.config.vision_lr = training_args.vision_lr
    model.config.projector_lr = training_args.projector_lr

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            
            if 'lm_head' in name or 'embed_token' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(processor=processor,
                                            data_args=data_args)

    trainer = LLamaVTrainer(
        model=model,
        processor=processor,
        args=training_args,
        **data_module
    )

    #QUI AVVIENE LA SCELTA TRA RIPRENDERE DAL CHECKPOINT O INIZIARE DA ZERO
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    model.config.text_config.use_cache = True
    
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )

        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters(), require_grad_only=False
        )

        if local_rank == 0 or local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_state_dict.bin"))
    else:
        safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()