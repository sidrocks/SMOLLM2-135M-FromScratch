import gradio as gr
import torch
from transformers import AutoTokenizer, AutoConfig
from model import SmolLM2
import os

# 1. Setup and Loading
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = "checkpoint_5000.pt"
tokenizer_path = "./custom_tokenizer"

print(f"Using device: {device}")

# Load Tokenizer
if os.path.exists(tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
else:
    # Fallback
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load Model
config = AutoConfig.from_pretrained("HuggingFaceTB/SmolLM2-135M")
config.vocab_size = len(tokenizer) # Sync vocab size
model = SmolLM2(config).to(device)

# Load Checkpoint
if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
else:
    print("Checkpoint not found! Using random weights.")

model.eval()

# 2. Generation Function
def generate(prompt, max_new_tokens, temperature, top_k, top_p, repetition_penalty):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    
    # Generation settings
    gen_kwargs = {
        "max_new_tokens": int(max_new_tokens),
        "temperature": float(temperature),
        "top_k": int(top_k),
        "top_p": float(top_p),
        "repetition_penalty": float(repetition_penalty),
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id
    }

    # Generate
    with torch.no_grad():
        generated_ids = input_ids
        for _ in range(int(max_new_tokens)):
            outputs = model(generated_ids)
            next_token_logits = outputs[:, -1, :]
            
            # Repetition Penalty
            if repetition_penalty != 1.0:
                for i in range(generated_ids.shape[0]):
                    for previous_token in set(generated_ids[i].tolist()):
                        if next_token_logits[i, previous_token] < 0:
                            next_token_logits[i, previous_token] *= repetition_penalty
                        else:
                            next_token_logits[i, previous_token] /= repetition_penalty

            # Temperature
            next_token_logits = next_token_logits / temperature
            
            # Top-K
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
                
            # Top-P (Nucleus Sampling)
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')

            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
                
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# 3. Gradio UI - Redesigned
with gr.Blocks(theme=gr.themes.Ocean()) as demo:
    gr.Markdown(
        """
        # 🌌 SmolLM2-135M Playground
        ### A custom 135M parameter model trained from scratch.
        """
    )
    
    with gr.Row():
        # Sidebar for Settings
        with gr.Column(scale=1, variant="panel"):
            gr.Markdown("### ⚙️ Generation Settings")
            gr.Markdown("Adjust these parameters to control the creativity and length of the generated text.")
            
            max_new_tokens = gr.Slider(minimum=10, maximum=1024, value=150, step=10, label="Max New Tokens", info="Maximum number of tokens to generate.")
            temperature = gr.Slider(minimum=0.1, maximum=2.0, value=0.8, step=0.1, label="Temperature", info="Higher values mean more random/creative output.")
            top_k = gr.Slider(minimum=0, maximum=100, value=40, step=1, label="Top-K", info="Limit to top K tokens.")
            top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.9, step=0.05, label="Top-P", info="Nucleus sampling probability.")
            repetition_penalty = gr.Slider(minimum=1.0, maximum=2.0, value=1.2, step=0.1, label="Repetition Penalty", info="Penalize repeated tokens.")
            
        # Main Content Area
        with gr.Column(scale=3):
            prompt = gr.Textbox(
                label="Input Prompt", 
                placeholder="Type your prompt here (e.g., 'First Citizen:')...", 
                lines=5,
                show_copy_button=True
            )
            
            gr.Examples(
                examples=[
                    ["First Citizen:"],
                    ["The meaning of life is"],
                    ["Once upon a time"],
                    ["To be or not to be"],
                    ["The quick brown fox"]
                ],
                inputs=prompt,
                label="Click on an example to load it:"
            )
            
            generate_btn = gr.Button("✨ Generate Text", variant="primary", size="lg")
            
            output = gr.Textbox(
                label="Generated Output", 
                lines=12, 
                show_copy_button=True,
                interactive=False
            )
            
    # Footer / Info
    with gr.Accordion("ℹ️ Model Information", open=False):
        gr.Markdown(
            """
            * **Architecture**: SmolLM2 (Transformer with Grouped Query Attention)
            * **Parameters**: 135M
            * **Training Data**: Wikitext / Custom Dataset
            * **Tokenizer**: Custom BPE
            """
        )

    generate_btn.click(
        fn=generate,
        inputs=[prompt, max_new_tokens, temperature, top_k, top_p, repetition_penalty],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch()
