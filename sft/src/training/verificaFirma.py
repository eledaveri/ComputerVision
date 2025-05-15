from transformers.trainer import TrainerState
print(dir(TrainerState))
print("Verifica Firma di init_training_references")
print(TrainerState.init_training_references.__name__)
print(TrainerState.init_training_references.__doc__)