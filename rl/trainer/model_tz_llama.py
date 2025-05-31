import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


# class VLMValue(nn.Module):
#     """
#     actually the base is also used for generation!
#     """
#     def __init__(self, base):
#         super(VLMValue, self).__init__()
#         self.base = base
        
#         # Create a value head that adapts to whatever hidden size the model actually has
#         self.check_and_create_value_head()
        
#     def check_and_create_value_head(self):
#         """Create a value head that adapts to the actual hidden size of the model."""
#         # We'll determine the hidden size dynamically when the first forward pass happens
#         self.value_head_initialized = False

#     def forward(self, inputs):
#         """
#         Differ from RL4VLM codebase
#         to adjust to cambrian
#         """
#         outputs = self.base(
#             **inputs,
#             output_hidden_states=True)
#         hidden_states = outputs.hidden_states[-1][:, -1]  # Use the last token of the last layer
        
#         # Initialize the value head if needed based on actual hidden size
#         if not self.value_head_initialized:
#             hidden_size = hidden_states.shape[-1]
#             print(f"Initializing value head with hidden size: {hidden_size}")
#             self.value_head = nn.Sequential(
#                 nn.Linear(hidden_size, 1024),
#                 nn.ReLU(),
#                 nn.Linear(1024, 512),
#                 nn.ReLU(),
#                 nn.Linear(512, 1)
#             ).to(self.base.device, dtype=torch.bfloat16)
#             self.value_head_initialized = True
            
#         values = self.value_head(hidden_states)
#         return values

class VLMValue(nn.Module):
    """
    Value model che aggiunge una testa di regressione a un modello base multimodale.
    Inizializza dinamicamente la value_head dopo il primo forward.
    """
    def __init__(self, base):
        super(VLMValue, self).__init__()
        self.base = base
        self.value_head_initialized = False
        self.value_head = None

    def forward(self, inputs):
        # Ottiene le hidden states dall'ultimo layer
        outputs = self.base(
            **inputs,
            output_hidden_states=True
        )
        hidden_states = outputs.hidden_states[-1][:, -1]  # Shape: (batch_size, hidden_size)
        hidden_states = hidden_states.to(torch.bfloat16)

        # Inizializza value_head dinamicamente
        if not self.value_head_initialized:
            hidden_size = hidden_states.shape[-1]
            print(f"[DEBUG] Dinamicamente inizializzo value_head con hidden_size={hidden_size}")
            self.value_head = nn.Sequential(
                nn.Linear(hidden_size, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 1)
            ).to(self.base.device, dtype=torch.bfloat16)
            self.value_head_initialized = True

        values = self.value_head(hidden_states)
        return values

class VLMPolicy(nn.Module):
    def __init__(self, tokenizer,
                value_model,
                generation_config,
                base_kwargs=None):
        """
        projection_f: the postprocessing function to parse text action
        """
        super(VLMPolicy, self).__init__()
        # self.args = args
        self.tokenizer = tokenizer
        self.value_model = value_model
        self.base = value_model.base
        # self.INPUT_IDS = INPUT_IDS # dumb solution
        self.temperature = generation_config.temperature
        self.max_new_tokens = generation_config.max_new_tokens
        self.thought_prob_coef = generation_config.thought_prob_coef
        self.token_cnt = 0
        self.called_inference_time = 0
        self.called_bp_time = 0


    def act_oneline(self, inputs, obs=None):
        # count the number of tokens and add to the token_cnt
        self.token_cnt += inputs['input_ids'].shape[1]
        with torch.no_grad():
            outputs = self.base.generate(
            **inputs, max_new_tokens=self.max_new_tokens, temperature=self.temperature, 
            output_scores=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
            # pad_token_id=self.tokenizer.eos_token_id
            )
            output_ids = outputs['sequences'][:, inputs['input_ids'].shape[1]:]
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        cated_io = torch.cat((inputs['input_ids'], output_ids), dim = 1)

        assert cated_io.shape == outputs['sequences'].shape
        cated_io_txt = self.tokenizer.decode(cated_io[0], skip_special_tokens=False)
        
        # llama processor will automatically add one more bos token, which sucks
        new_inputs = {"input_ids": cated_io}
        for key in inputs.keys():
            if key != 'input_ids':
                if key in ['cross_attention_mask', 'attention_mask']:
                    new_shape = list(inputs[key].shape)
                    new_shape[1] = int(output_ids.shape[1])
                    new_inputs[key] = torch.cat([inputs[key], torch.ones(new_shape, dtype=inputs[key].dtype, device=inputs[key].device)], dim=1)
                else:
                    new_inputs[key] = inputs[key]
        input_ids_register = inputs['input_ids']

        assert new_inputs['input_ids'].shape[1] == cated_io.shape[1]
        io_dict = {"io_pair": (input_ids_register, output_ids), **new_inputs}
        with torch.no_grad():
            values, sum_log_prob, action_tokens_log_prob = self.evaluate(**io_dict, inference=True)
        return values, io_dict, output_text, sum_log_prob, action_tokens_log_prob
    
    def cat_io_pair(self, io_pair):
        to_be_cated = []
        input_ids, output_ids = io_pair[0], io_pair[1]
        to_be_cated.append(input_ids.to(self.base.device))
        to_be_cated.append(output_ids.to(self.base.device))
        return torch.cat(to_be_cated, dim = 1)
    
    def evaluate(self, io_pair, inference=False, **kwargs):
        cated_io = kwargs['input_ids']
        input_ids = io_pair[0]
        output_ids = cated_io[:, input_ids.shape[1]:]

        outputs = self.base(
            output_hidden_states = True,
            **kwargs
        )
        
        if inference:
            pass
        else:
            if 'pixel_values' in kwargs.keys():
                self.token_cnt += 6 * (cated_io.shape[1]+1600) # llama 3.2 has 1601 visual tokens
                self.called_bp_time += 1
            else:
                self.token_cnt += 6 * cated_io.shape[1]
                self.called_bp_time += 1
        
        scores = outputs.logits

        input_token_len = input_ids.shape[1]
        try:
            # Get the hidden state for evaluation - using the last token
            hidden_states = outputs.hidden_states[-1][:, -1]  # Use the last token
            
            values = self.value_model.value_head(hidden_states)
            scores = scores * (1/self.temperature)
            scores = scores.to(torch.float32)
            log_probs = torch.nn.functional.log_softmax(scores, dim=-1)

            log_probs = log_probs.to(torch.bfloat16)
            output_ids_mask = (output_ids != 0)[:, 1:]

            selected_log_probs = output_ids_mask * torch.take_along_dim(log_probs[:, input_token_len:-1], output_ids[:,1:].unsqueeze(2), dim = 2).squeeze(2)
            unfolded = output_ids.unfold(dimension=-1, size=3, step=1)
        except Exception as e:
            # Better error handling
            print(f"Exception caught: {str(e)}")
            print("outputs.hidden_states:", [h.shape for h in outputs.hidden_states])
            print("input_token_len:", input_token_len)
            print("input_ids:", input_ids.shape)
            print("output_ids:", output_ids.shape)
            print("cated_io:", cated_io.shape)
            
            decoded_io = self.tokenizer.decode(cated_io[0], skip_special_tokens=False)
            decoded_input_ids = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
            print("decoded_io:", decoded_io)
            print("decoded_input_ids:", decoded_input_ids)
            
            # Return placeholder values instead of crashing
            values = torch.tensor([[-10.0]], device=self.base.device)
            sum_log_prob = torch.tensor([-10.0], device=self.base.device)
            action_tokens_log_prob = torch.tensor([-10.0], device=self.base.device)
            return values, sum_log_prob, action_tokens_log_prob

        map_dict = []
        try:
            for i in range(len(output_ids[0])-1):
                map_dict.append({self.tokenizer.batch_decode(output_ids[:,i:i+1], skip_special_tokens=True)[0]: output_ids[:,i:i+1]})
            for i, dic in enumerate(map_dict):
                if 'formula' in dic.keys():
                    target = torch.cat((list(map_dict[i-1].values())[0], list(map_dict[i].values())[0], list(map_dict[i+1].values())[0]), dim=0).flatten()
                elif 'action' in dic.keys():
                    target = torch.tensor([330, 1335, 794]).to(self.base.device)
                    break
            assert 'target' in locals()
        except:
            target = torch.tensor([330,60599,794]).to(self.base.device)

        target = target.to(self.base.device)
        
        matches = (unfolded == target).all(dim = -1)
        match_index = matches.nonzero(as_tuple=True)[-1]
        
        if match_index.shape[0] >= 1:
            match_index = match_index[-1].unsqueeze(0)
        else:
            try:
                match_index = output_ids_mask.nonzero(as_tuple=False)[-4,1]
            except:
                sum_log_prob = torch.tensor([-2]).to(self.base.device)
                action_tokens_log_prob = torch.tensor([-1]).to(self.base.device)
                return values, sum_log_prob, action_tokens_log_prob
        
        thought_log_prob = torch.sum(selected_log_probs[:,1:match_index-1], dim = 1)
        action_tokens_log_prob = torch.sum(selected_log_probs[:,match_index-1:], dim = 1)
        sum_log_prob = self.thought_prob_coef*thought_log_prob + action_tokens_log_prob
        return values, sum_log_prob, action_tokens_log_prob


    def evaluate_actions(self, io_dict, image = None):
        value, action_log_prob, _ = self.evaluate(**io_dict)
        return value, action_log_prob