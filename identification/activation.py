import argparse
from types import MethodType

import torch
from vllm import LLM, SamplingParams

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="meta-llama/Meta-Llama-3-8B")
parser.add_argument("-l", "--lang", type=str, default="zh")
parser.add_argument("-s", "--save", type=str, default="llama_3-1")
args = parser.parse_args()

model = LLM(model=args.model, tensor_parallel_size=torch.cuda.device_count(), enforce_eager=True)

max_length = model.llm_engine.model_config.max_model_len
if "gemma-3" in str(args.model).lower():
    num_layers = model.llm_engine.model_config.hf_config.text_config.num_hidden_layers
    intermediate_size = model.llm_engine.model_config.hf_config.text_config.intermediate_size
    activation_size = model.llm_engine.model_config.hf_config.text_config.intermediate_size // 2
else:
    num_layers = model.llm_engine.model_config.hf_config.num_hidden_layers
    intermediate_size = model.llm_engine.model_config.hf_config.intermediate_size
    activation_size = model.llm_engine.model_config.hf_config.intermediate_size // 2

if "llama" or "gemma" in args.model:
    over_zero = torch.zeros(num_layers, intermediate_size, dtype=torch.int32).to('cuda')
    activation_sums = torch.zeros(num_layers, intermediate_size, dtype=torch.float32).to('cuda')
    activation_counts = torch.zeros(num_layers, intermediate_size, dtype=torch.int32).to('cuda')
else:
    over_zero = torch.zeros(num_layers, activation_size, dtype=torch.int32).to('cuda')
    activation_sums = torch.zeros(num_layers, activation_size, dtype=torch.float32).to('cuda')
    activation_counts = torch.zeros(num_layers, activation_size, dtype=torch.int32).to('cuda')

def factory(idx):
    def llama_forward(self, x):
        # print(f"[DEBUG] x shape: {x.shape}")  # print shape of input to MLP

        gate_up, _ = self.gate_up_proj(x)
        # print(f"[DEBUG] gate_up shape: {gate_up.shape}")  # print shape after projection

        i = gate_up.size(-1)

        if gate_up.dim() == 3:
            # expected shape: (batch, seq_len, 2*intermediate_size)
            gate_up[:, :, : i // 2] = torch.nn.SiLU()(gate_up[:, :, : i // 2])
            activation = gate_up[:, :, : i // 2].float()
            
            # Count positive activations
            over_zero[idx, :] += (activation > 0).sum(dim=(0, 1))
            
            # Sum all activations and count total tokens for averaging
            activation_sums[idx, :] += activation.sum(dim=(0, 1))
            activation_counts[idx, :] += activation.size(0) * activation.size(1)  # batch_size * seq_len
            
            x = gate_up[:, :, : i // 2] * gate_up[:, :, i // 2 :]
        elif gate_up.dim() == 2:
            # expected shape: (batch*seq_len, 2*intermediate_size)
            gate_up[:, : i // 2] = torch.nn.SiLU()(gate_up[:, : i // 2])
            activation = gate_up[:, : i // 2].float()
            
            # Count positive activations
            over_zero[idx, :] += (activation > 0).sum(dim=0)
            
            # Sum all activations and count total tokens for averaging
            activation_sums[idx, :] += activation.sum(dim=0)
            activation_counts[idx, :] += activation.size(0)  # batch_size * seq_len
            
            x = gate_up[:, : i // 2] * gate_up[:, i // 2 :]
        else:
            raise ValueError(f"Unexpected gate_up shape: {gate_up.shape}")

        # Remove the debug print and exit for actual execution
        # print(activation)
        # sys.exit(1)
        
        x, _ = self.down_proj(x)
        return x

    return llama_forward

for i in range(num_layers):
    if "gemma-3" in str(args.model).lower():
        obj = model.llm_engine.model_executor.driver_worker.model_runner.model.language_model.model.layers[i].mlp
    else:
        obj = model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[i].mlp
    
    obj.forward = MethodType(factory(i), obj)

lang = args.lang
save = args.save.split(" ")
ids = torch.load(f'data_{save[0]}/id.{lang}.train.{save[1]}')

l = ids.size(0)
l = min(l, 99999744) // max_length * max_length
input_ids = ids[:l].reshape(-1, max_length)

output = model.generate(prompt_token_ids=input_ids.tolist(), sampling_params=SamplingParams(max_tokens=1))

# Calculate average activations
average_activations = activation_sums / activation_counts.float()

output = dict(
    n=l, 
    over_zero=over_zero.to('cpu'),
    average_activations=average_activations.to('cpu'),
    activation_counts=activation_counts.to('cpu')
)

torch.save(output, f'data_{save[0]}/activation.{lang}.train.{save[1]}')