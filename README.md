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
Generated: The meaning of life is
The people; and therefore the very royal soldiers,
The noble duke, I'll make you both the people.

PETRUCHIO:
Well, I must:
What you have some mil'dly?

GLOUCESTER:
What
-----------------------------

Step 600: Loss 3.852552652359009
Step 700: Loss 3.67338228225708
Step 800: Loss 3.858555316925049
Step 900: Loss 3.88484263420105
Step 1000: Loss 3.3133487701416016

--- Step 1000 Generation ---
Generated: The meaning of life is
drumbling; and in the middle of love
of I am the end of the hand: I have sworn
without you have done.

LEONTES:
I pray you, sir, I'll swear
To say you
-----------------------------

Step 1100: Loss 3.7228200435638428
Step 1200: Loss 3.4301235675811768
Step 1300: Loss 3.7488114833831787
Step 1400: Loss 3.0603678226470947
Step 1500: Loss 3.1305861473083496

--- Step 1500 Generation ---
Generated: The meaning of life is it not;
And she, for all the fairest of this earth;
And, by the other of heaven she's love,
Persuitute it is not be married.

KING HENRY VI:
You bid me entreat
-----------------------------

Step 1600: Loss 2.9551541805267334
Step 1700: Loss 2.3346869945526123
Step 1800: Loss 2.3356552124023438
Step 1900: Loss 2.467982769012451
Step 2000: Loss 1.4448952674865723

--- Step 2000 Generation ---
Generated: The meaning of life is full of Naples.

KING RICHARD III:
Methoughts it is in love tooth.

DERBY:
No; but every tale hath he of a horse.

KING RICHARD III:
Think, I would have had
-----------------------------

Step 2100: Loss 1.4267183542251587
Step 2200: Loss 1.770742654800415
Step 2300: Loss 0.814985454082489
Step 2400: Loss 0.7272887229919434
Step 2500: Loss 0.8367995619773865

--- Step 2500 Generation ---
Generated: The meaning of life is ta'en out of let's
bestilence this out-virt lady and my good
fortune, within my state was not all much abused; for
you are but infanted: but then all poverse

-----------------------------

Step 2600: Loss 1.0043246746063232
Step 2700: Loss 0.259679913520813
Step 2800: Loss 0.2687181532382965
Step 2900: Loss 0.3258618414402008
Step 3000: Loss 0.12177103757858276

--- Step 3000 Generation ---
Generated: The meaning of life is procure
Than thou shalt be; and, ere I do not say.

KING RICHARD II:
This is no mortal drunibal:
This is no way to be sent for from him
More than he is, yet
-----------------------------

Step 3100: Loss 0.09513700753450394
Step 3200: Loss 0.08790317177772522
Step 3300: Loss 0.04893791675567627
Step 3400: Loss 0.0646422952413559
Step 3500: Loss 0.06223556771874428

--- Step 3500 Generation ---
Generated: The meaning of life is now a son,
To be seen in him that acts at his,
As were he, I would set you on him,
As he is proud to be his worthless.

Lord Mayor:
Hail how he
-----------------------------

Step 3600: Loss 0.054624490439891815
Step 3700: Loss 0.05200936645269394
Step 3800: Loss 0.06808382272720337
Step 3900: Loss 0.07332677394151688
Step 4000: Loss 0.08571958541870117

--- Step 4000 Generation ---
Generated: The meaning of life is full of person.

HENRY BOLINGBROKE:
You are pardon, my lord, I know my:
Not yet my grief hath praise my compare,
To fight against the fair degree and the fearful land
To this extremity and
-----------------------------

Step 4100: Loss 0.08139943331480026
Step 4200: Loss 0.1808193176984787
Step 4300: Loss 0.16262556612491608
Step 4400: Loss 0.21453623473644257
Step 4500: Loss 0.3636895418167114

--- Step 4500 Generation ---
Generated: The meaning of life is the house of you.

Nurse:
O brother, your young lady, for yourself to him.

JULIET:
Marry, and grant it should not be so yet.

ROMEO:
Peace, peace! help, peace
-----------------------------

Step 4600: Loss 0.18211005628108978
Step 4700: Loss 0.22678734362125397
Step 4800: Loss 0.23497304320335388
Step 4900: Loss 0.22399206459522247
Step 5000: Loss 0.07477153837680817

--- Step 5000 Generation ---
Generated: The meaning of life is scarce any
Than twentyh made it; who, in a kind of course,
that they shall know, as if they had been
A thing for that danger, my sovereign.

HENRY BOLINGBROKE:
Call forth Bel and set
-----------------------------

Checkpoint saved to checkpoint_5000.pt
```

### Phase 2: Extended Training (50 Steps)
*   **Goal**: Verify checkpoint loading and resumption capability.
*   **Observation**: The model successfully loaded the state from step 5000 and continued training with a low loss, proving the checkpoint mechanism works.

```text
Resume Step 10: Loss 0.14603516459465027
Resume Step 20: Loss 0.12318285554647446
Resume Step 30: Loss 0.08086450397968292
Resume Step 40: Loss 0.11517078429460526
Resume Step 50: Loss 0.13376004993915558

--- Step 5050 Generation ---
C:\Users\sidhe\AppData\Local\Temp\ipykernel_23008\2461595262.py:27: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast():
Generated: First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
Is't a verdict?

All:
No more talking on't: let it be done: away, away!

Second Citizen:
One word.

First Citizen:
We are coniled and we will.

MENENIUS:
Ay, we'll show 'em good friends,' we'll show themselves:
'Tis very strangely.

Second Citizen:
And 'twas a purposed, we'll keep no tears;
But we shall have vengeance and meet the belly;
Where, they shall stand, will we do good men, we'll
With our senators: we'll promption 'O' good,'
Their very wayward.

MENENIUS:
What we will, to do
Our Rome will in justice w: the gods do o' the city
In every mattling aparted, is as an art,
It cannot miss't: the other lady,
In peace what it has, in a little way, a kind of it is his country:
He's in a few, and his nobility, he waves mein it in his country, he'll carry it off.
What think you, say, think you?

First Citizen:
Our Rome are our garments, the very he has our voices;
He would have offices, so it is as easy next, were as an a good will, he were a
painty.

MENENIUS:
Sincts, we were a mind of fashion, they stay:
Why then we do we were as an a man could't be his
In hand, if he were not much beyond a good counsel, it is a
ity to the vivery keeps his capit.

Second Citizen:
He's brows: go to-night, sir, it doth not go: he is as another merry as those that he were a noble man, one poor master?

First Citizen:
 their tongues speak: who's heifths upon his countrymen as they were as they were a panteveter'd;' for they are very strange, I wonder will a very little grave:
Silence that we have honours that, if't please you can he pass'd,
'Farewell: I'll peize him home, that he let me desire some good belly toadovery.

Lone I can bury tauntedeem him to-morrow.
'Tis well, sir, was God's well meteads;
His very promise pass'd: his countrymen stay awhile,
'Twere an eye of his countrymen, they were as they were i' the senators are you wot'd, no queen,
How he were hanged, I did it in an answer.

First Citizen:
Gardeliused your voices: he has a pretty master?

Second Citizen:
And, sir, in what cannot think.

BRUTUS:
What is the gods will he'll show thee by you a massies them a mould manifth, I will draw not that he pass'd:
Like death to Polixenes, that it doth henew the superflow'd again, the VolscesinguriedI' the peace of his countrymen,
'Twere a perfect woman moved you:
What think he was said, is it kindly leak before his countrymen with rushireet: if it did but grief he did it, I'll pray you, in it in an actorough case lost for a painting, you have seen a bust of rathematica, he were a
meticolariarried Nicanifth us, then he's no more but He', I warrant, and then; for he were as noble, I hear say,
Not to myself, that doubt but me a-proper:
Glower as they were by rage to behold, he's a pock will not work,
Even to the senate, the very pretty fooling,
In branish thing, he was like a man well, art a little grave sir, was his son, he would have been an angry with no warrant from the one,
 a foul son, is it skill'd: he has a dear-five for the city is grown poor sir,
Scarson
-----------------------------

Checkpoint saved to checkpoint_5050.pt
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

----

## Demo

Try out the trained model interactively here 👉 

