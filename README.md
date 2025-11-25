# SmolLM2-135M: From Scratch to Production

This project demonstrates the end-to-end process of reverse-engineering, implementing, training, and deploying a custom version of the **SmolLM2-135M** language model.

## 🎯 Objectives
1.  **Reverse Engineer**: Dissect the `HuggingFaceTB/SmolLM2-135M` model to understand its architecture (Llama-style).
2.  **Build from Scratch**: Implement the model architecture purely in PyTorch (`model.py`) without relying on the `transformers` library's modeling code.
3.  **Custom Training**: Train the custom model on a local dataset (`input-1.txt`) using a custom tokenizer.
4.  **Optimize**: Implement performance optimizations (AMP, TF32, etc.) for efficient training.
5.  **Deploy**: Create a user-friendly Gradio web app for text generation.

---

## 🔍 Architecture Comparison

To ensure our custom implementation is accurate, we compared it directly against the original `HuggingFaceTB/SmolLM2-135M` model.

| Feature | Original (HuggingFace) | Custom (`model.py`) | Match? |
| :--- | :--- | :--- | :--- |
| **Architecture Base** | LlamaForCausalLM | Custom `SmolLM2` Class | ✅ |
| **Hidden Size** | 576 | 576 | ✅ |
| **Intermediate Size** | 1536 | 1536 | ✅ |
| **Layers** | 30 | 30 | ✅ |
| **Attention Heads** | 9 (Query), 3 (Key/Val) | 9 (Query), 3 (Key/Val) | ✅ |
| **Positional Embeddings** | RoPE (Rotary) | RoPE (Split-half style) | ✅ |
| **Activation** | SiLU (Swish) | SiLU (Swish) | ✅ |
| **Normalization** | RMSNorm | RMSNorm | ✅ |

**Validation:**
In `reverse_engineering.ipynb`, we successfully loaded the weights from the pre-trained HuggingFace model into our custom architecture and verified that the outputs were identical (within floating-point tolerance), confirming the structural correctness.

---

## 🧩 Key Components (`model.py`)

Our implementation breaks down the Transformer architecture into modular, understandable components:

### 1. RMSNorm (`RMSNorm`)
*   **What it is**: Root Mean Square Layer Normalization.
*   **Why**: Simpler and computationally faster than standard LayerNorm because it re-scales invariance without centering the mean.
*   **Implementation**: Calculates the root mean square of the input and divides the input by it, then scales by a learnable parameter `weight`.

### 2. Rotary Positional Embeddings (`RotaryEmbedding`)
*   **What it is**: A method to encode position information by rotating the query and key vectors in the embedding space.
*   **Why**: Allows the model to generalize to sequence lengths longer than seen during training and captures relative positions better than absolute embeddings.
*   **Implementation**: We use the "split-half" strategy characteristic of Llama models, where the embedding dimension is split into two halves that are rotated against each other.

### 3. SwiGLU MLP (`MLP`)
*   **What it is**: A Feed-Forward Network using the Swish-Gated Linear Unit activation function.
*   **Why**: SwiGLU has been shown to offer better performance than standard ReLU or GeLU activations in large language models.
*   **Structure**: It involves three linear projections: a gate projection, an up projection, and a down projection. The output is `(Swish(Gate) * Up) * Down`.

### 4. Grouped Query Attention (`GQA`)
*   **What it is**: A variant of Multi-Head Attention where multiple Query heads share a single Key/Value head.
*   **Why**: Significantly reduces the memory bandwidth required for loading Keys and Values (KV Cache) during inference, speeding up generation.
*   **Config**: Our model uses 9 Query heads and 3 KV heads, meaning every 3 Query heads share 1 KV head.

---

## 📈 Training Logs

### Phase 1: Initial Training (5000 Steps)
*   **Goal**: Train the model from scratch on the custom dataset.
*   **Observation**: Loss decreased steadily, and the model began learning the structure of the text (e.g., character names).

```text
Starting training...
Step 100: Loss 6.842
Step 200: Loss 5.912
Step 300: Loss 5.104
...
Step 2500: Loss 3.210
--- Step 2500 Generation ---
Generated: First Citizen: We are accounted poor citizens...
-----------------------------
...
Step 5000: Loss 2.150
Checkpoint saved to checkpoint_5000.pt
```

### Phase 2: Extended Training (50 Steps)
*   **Goal**: Verify checkpoint loading and resumption capability.
*   **Observation**: The model successfully loaded the state from step 5000 and continued training with a low loss, proving the checkpoint mechanism works.

```text
Resuming training...
Resume Step 10: Loss 2.145
Resume Step 20: Loss 2.138
Resume Step 30: Loss 2.120
Resume Step 40: Loss 2.115
Resume Step 50: Loss 2.108

--- Step 5050 Generation ---
Generated: SICINIUS: He has no equal.
-----------------------------
Resumed training completed.
```

---

## 📊 Parameter Calculations

### Model Parameters
The standard SmolLM2-135M has ~135 Million parameters. However, our custom version is lighter due to the reduced vocabulary size.

**Configuration:**
*   Layers ($L$): 30
*   Hidden Size ($H$): 576
*   Intermediate Size ($I$): 1536
*   Heads: 9 (Q), 3 (K/V)
*   Vocab Size ($V$): 5000 (Custom) vs 49152 (Original)

**Breakdown (Approximate):**
1.  **Embeddings**: $V \times H = 5000 \times 576 \approx \mathbf{2.88M}$
2.  **Attention Layers (x30)**:
    *   Projections (Q, K, V, O) with GQA.
    *   $\approx 2.66 \times H^2$ per layer $\approx \mathbf{0.88M}$ per layer.
3.  **MLP Layers (x30)**:
    *   Gate, Up, Down projections.
    *   $3 \times H \times I = 3 \times 576 \times 1536 \approx \mathbf{2.65M}$ per layer.
4.  **Layer Norms (x30)**: Negligible.

**Total Count:**
*   **Backbone (Layers)**: $30 \times (0.88M + 2.65M) \approx \mathbf{106M}$
*   **Embeddings (Input + Output)**: If tied weights: $\mathbf{2.88M}$. If untied: $\mathbf{5.76M}$.
*   **Total Custom Model**: $\approx \mathbf{109M - 112M}$ Parameters.
*   *Note: The original 135M count comes from the much larger 49k vocabulary ($49k \times 576 \approx 28M$ params just in embeddings).*

### Dataset
*   **Source**: `input-1.txt`
*   **Size**: ~1.1 MB
*   **Token Count**: ~300,000 tokens (estimated with vocab size 5000).

---

## 💻 How to Run

### 1. Setup
```bash
pip install -r requirements.txt
```

### 2. Train Tokenizer
Run `train_tokenizer.ipynb` to generate the `./custom_tokenizer` folder.

### 3. Train Model
Run `training.ipynb`. This will:
*   Load the custom tokenizer.
*   Initialize the model.
*   Train for 5000 steps.
*   Save `checkpoint_5000.pt`.

### 4. Run the App
Launch the Gradio interface:
```bash
python app.py
```
Open your browser to the local URL (usually `http://127.0.0.1:7860`).

---

## 📂 Project Structure

| File | Description |
| :--- | :--- |
| `model.py` | **The Core.** Contains the custom PyTorch implementation of `SmolLM2`, including `RMSNorm`, `RotaryEmbedding` (RoPE), `SwiGLU` MLP, and `GroupedQueryAttention`. |
| `reverse_engineering.ipynb` | Notebook used to inspect the original model, verify the custom implementation, and perform weight transfer validation. |
| `train_tokenizer.ipynb` | Trains a custom Byte-Pair Encoding (BPE) tokenizer on `input-1.txt` (Vocab size: 5000). |
| `training.ipynb` | The main training loop. Handles dataset chunking, mixed-precision training, checkpointing, and text generation monitoring. |
| `app.py` | A Gradio-based web application to interact with the trained model. Features a modern UI with sidebar settings. |
| `requirements.txt` | List of Python dependencies required to run the project. |
| `input-1.txt` | The local dataset used for training (Shakespearean text). |
