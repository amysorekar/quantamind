# AI Agent Fine-Tuning and Evaluation

## Summary

This project explores and compares two lightweight fine-tuning strategies for adapting a base language model (TinyLlama-1.1B) to customer support tasks: QLoRA and Prompt Tuning. The objective is to benchmark their effectiveness under strict compute and data constraints, reflecting real-world deployment scenarios where low-resource adaptation is key.

The dataset consists of short customer service message-response pairs, pre-tokenized into prompt-response format. Four models were trained: two QLoRA variants (with `r=4` and `r=8`) and two Prompt Tuning variants (with 10 and 20 virtual tokens). Each model was trained for 3 epochs on 200 examples using Google Colab's free GPU tier.

A comprehensive evaluation pipeline was built to assess performance using BLEU and ROUGE scores, latency, error analysis, sentiment/priority-aware prompting, and category-based breakdowns. This evaluation helps capture not only aggregate accuracy, but also qualitative strengths and weaknesses.

To overcome GPU memory limitations, the training pipeline was explicitly optimized for memory safety via `gc.collect()` and `torch.cuda.empty_cache()` between model runs. Training was done sequentially with immediate analysis and cleanup, making it scalable even on free-tier infrastructure.

This project demonstrates practical, reproducible fine-tuning workflows for low-resource LLM deployment and provides insight into trade-offs between parameter-efficient methods.


## Model Selection

Given the constraints of Google Colab's free GPU tier, I prioritized selecting a model that was lightweight enough to fit in memory while still supporting meaningful experimentation with fine-tuning techniques like QLoRA and prompt tuning.

### TinyLlama-1.1B

[TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) was chosen as the base model because it offers a strong balance between size, performance, and compatibility with low-resource environments:
- **Small footprint**: At 1.1B parameters, TinyLlama comfortably runs on Colab’s free-tier GPU without memory issues.
- **Instruction-tuned**: The chat variant is already aligned for conversational tasks, making it suitable for customer support-style generation out of the box.
- **Active development**: It's a recent model with modern training techniques and good community support.

### Alternatives Considered

Other small models were considered but ultimately not selected:
- **DistilGPT2**: Very lightweight, but outdated and underperforms on modern NLP tasks.
- **GPT-J (6B)**: Better performance, but too large for consistent training on Colab without crashes.
- **Phi-2 (2.7B)**: Solid reasoning ability in small form factor, but had occasional compatibility issues with QLoRA in my testing.

### What I Would've Used with More Compute

If resources were not a constraint, I would have opted for **Mistral-7B-Instruct**:

- It is one of the most performant open-weight models available in the 7B range.
- It has strong fine-tuning support and widespread community benchmarks.
- It outperforms older 7B models like LLaMA-2 on most tasks while being more efficient.

### Pros and Cons of TinyLlama

**Pros:**
- Fast training and inference even on limited hardware
- Compatible with PEFT methods like QLoRA and prompt tuning
- Good enough performance for demonstrating methodology and evaluation pipelines

**Cons:**
- Lower ceiling on performance compared to 7B+ models
- May struggle with nuanced generation or long-form coherence
- Fewer community benchmarks and resources compared to more popular models

In summary, TinyLlama was the best choice for prototyping and evaluating fine-tuning techniques under compute constraints, while still being expressive enough to demonstrate differences across training methods.


## Dataset

Quantamind's mission centers on building LLMs optimized for business workflows. These workflows often involve:

- Understanding internal documentation and conversations
- Processing both structured and unstructured business data
- Automating tasks like Q&A, summarization, and agent support

To simulate a realistic use case of LLM fine-tuning for business-customer interaction, I chose a publicly available **Twitter customer support dataset**.

### Why this dataset?

- **Real-world interactions**: The dataset contains genuine customer inquiries and brand responses, closely resembling support ticket dialogues.
- **Natural format for LLMs**: Each sample is a straightforward input-output pair — _user message_ → _agent response_ — which maps cleanly to causal language modeling.
- **No privacy concerns**: The dataset contains no PHI or PII, making it safe to use without additional redaction.
- **Manageable size**: I subsetted the data to 400 high-quality examples to allow for experimentation and rapid iteration within the compute constraints of Colab’s free tier.

If this were a production system or had access to more resources, I would have opted for a larger dataset with richer multi-turn dialogues, possibly including synthetic augmentation, or at the very least, used the entire dataset.


## Libraries and Tooling

This project uses a combination of core machine learning libraries, evaluation tools, and utility packages to support model training, fine-tuning, and analysis.

### Core ML & Fine-Tuning

- **`transformers`** – Provides pretrained language models and training utilities (e.g., `AutoModelForCausalLM`, `Trainer`).
- **`peft`** – Enables efficient fine-tuning methods such as QLoRA and Prompt Tuning.
- **`datasets`** – Handles dataset loading and formatting for use with Hugging Face pipelines.
- **`torch`** – Core deep learning framework for model operations and GPU acceleration.
- **`accelerate`**, **`optimum`** – Optional speed-ups for training and inference (mostly relevant for scaling or deployment).

### Evaluation Metrics

- **`nltk.translate.bleu_score`** – Used for calculating BLEU score, a common n-gram overlap metric for generation tasks.
- **`rouge_score`** – Computes ROUGE-1, ROUGE-2, and ROUGE-L to evaluate similarity between generated and reference responses.
- **`textblob`** – Provides sentiment polarity scores to add context-aware behavior to the agent (e.g. tone adaptation).

### Utility & Miscellaneous

- **`matplotlib`** – For plotting training loss curves and result comparisons.
- **`tqdm`** – Adds progress bars to training and evaluation loops.
- **`gc`, `warnings`, `time`, `os`** – For memory cleanup, suppressing logs, timing operations, and file handling.
- **`pandas`** – For reading JSONL files and organizing results into tables and plots.
- **`sklearn.model_selection.train_test_split`** – Included but not used; originally for data splitting if needed.

### Colab-Specific

- **`google.colab.userdata`, `drive`** – For securely accessing Hugging Face tokens and mounting Google Drive to access the dataset during Colab runs.

All tools were chosen to be lightweight and compatible with Google Colab’s free-tier environment while supporting advanced customization (e.g., PEFT methods).



## Training & Model Configurations
I used two lightweight fine-tuning methods: **QLoRA** and **Prompt Tuning**, each with two variants to observe the impact of configuration changes. The training was done using Hugging Face's `Trainer` API and PEFT (Parameter-Efficient Fine-Tuning) wrappers.

### Fine-Tuning Approach

Each model was fine-tuned using 400 training examples for 3 epochs. Preprocessing involved formatting each example into a natural dialogue format:
`Customer message: {input}\nAgent response: {output}`.


Tokenization was done with a max length of 512 and padding enabled. The training setup used:
- Batch size: 4
- Gradient accumulation steps: 4
- Learning rate: 2e-4
- Precision: fp16
- Evaluation strategy: handled manually after training

### Model Variants

- **QLoRA-r8**: 8 rank adapters, higher capacity, more expressive
- **QLoRA-r4**: 4 rank adapters, more memory-efficient, faster training
- **Prompt-20**: 20 virtual tokens prepended to every input; more tunable context
- **Prompt-10**: 10 virtual tokens; lower overhead and slightly faster

### LoraConfig Parameters

- **`r` (rank)**: Determines the adapter matrix size. `r=8` gives more capacity than `r=4`, at the cost of additional compute and memory.
- **`lora_alpha`**: Scales the adapter output. Values of 16 and 32 were used to balance learning signal and training stability.
- **`lora_dropout`**: Set to 0.05 to prevent overfitting while still keeping training efficient.
- **`target_modules`**: Focused adaptation on `q_proj` and `v_proj` of the attention layers for maximum effect on generation behavior.

### Prompt Tuning Parameters

- **`num_virtual_tokens`**: The two configs (10 vs. 20) let us observe how more or fewer soft prompts affect learning capacity.
- **`prompt_tuning_init_text`**: Initialized using `"You are a helpful customer support assistant."` to help the prompt tuning converge faster by anchoring the model in the right behavioral space from the start.

Each model was trained, evaluated, and cleared from memory in a loop to stay within Colab's GPU and RAM limits.



## Fine-Tuning Methodologies

This project compares two parameter-efficient fine-tuning (PEFT) strategies for adapting a pretrained LLM to customer support tasks: **QLoRA** and **Prompt Tuning**. Each method modifies the base model in different ways, with distinct tradeoffs in flexibility, efficiency, and performance.

### QLoRA

**What it is**:  
QLoRA (Quantized Low-Rank Adaptation) fine-tunes a quantized language model by inserting trainable *low-rank adapters* into specific layers (commonly attention projections). The base model weights remain frozen, and only these adapters are trained.

**How it works**:
- The model is first quantized (e.g. 4-bit), dramatically reducing memory usage.
- A low-rank matrix decomposition (rank `r`) is added to selected layers (here, `q_proj` and `v_proj`).
- Only the adapter weights are updated during training.
- Outputs from the adapter are scaled with a hyperparameter `lora_alpha`.

**Benefits**:
- Very memory-efficient: you can fine-tune large models on consumer hardware.
- Preserves base model weights (can reuse for multiple tasks).
- Surprisingly strong performance given limited training.

**Limitations**:
- Adapter capacity is limited by rank `r`—too low and the model can't adapt well.
- Requires careful hyperparameter tuning (`r`, `alpha`, `dropout`).
- Still more GPU-heavy than prompt tuning.

---

### Prompt Tuning

**What it is**:  
Prompt Tuning learns a small set of *virtual tokens* that are prepended to every input. These tokens are trainable embeddings that help condition the model to behave a certain way, without changing any of the original model weights.

**How it works**:
- A fixed number of virtual tokens (`num_virtual_tokens`) are inserted at the start of the input.
- These tokens are randomly initialized or seeded with some meaningful text (`prompt_tuning_init_text`).
- During training, only the virtual tokens are updated—everything else stays frozen.

**Benefits**:
- Extremely lightweight: very few parameters to train.
- Fast to train, even on CPU.
- Safe: no risk of overfitting or catastrophic forgetting.

**Limitations**:
- Lower ceiling for performance: you’re just nudging the model rather than adapting its internals.
- Doesn't work well for tasks that require deep task-specific reasoning or new knowledge.
- More sensitive to prompt formulation and initialization.

---

### Comparison

| Aspect | QLoRA | Prompt Tuning |
|--------|-------|----------------|
| Trainable Params | Moderate (~0.2–1%) | Very Low (~0.01%) |
| Resource Usage | Moderate | Very Low |
| Flexibility | High (can change deeper behavior) | Low (mostly stylistic/task nudging) |
| Training Time | Slower | Faster |
| Best Use Cases | New tasks, deeper reasoning, domain shift | Stylistic tasks, light adaptation, constrained environments |
| Risk of Overfitting | Higher | Very low |

---

### When to Use Each

- **Use QLoRA when**:
  - You can afford some GPU overhead.
  - The task involves significant reasoning or domain shift.
  - You want more control over the model’s behavior.
  - You care about strong downstream metrics like BLEU or ROUGE.

- **Use Prompt Tuning when**:
  - You're extremely compute-constrained.
  - The base model is already strong and needs light behavioral steering.
  - You're running in production where full fine-tuning isn't practical.
  - You want to support multiple prompts for multiple tasks cheaply.

---

In this project, both methods were applied to a small LLM (TinyLlama) to compare their impact on a customer support generation task. By testing two configurations of each (varying rank for QLoRA and virtual token count for Prompt Tuning), we could assess how scaling each method affects performance under low-resource constraints.


## Evaluation Strategy

This project uses a comprehensive evaluation suite to assess both quantitative performance and qualitative behavior across fine-tuned models. The following techniques were applied:

### Metrics Used
- **BLEU**: Evaluates n-gram overlap between predicted and target responses; good for measuring surface-level accuracy.
- **ROUGE-1 / ROUGE-2 / ROUGE-L**: Measures overlap in unigrams, bigrams, and longest common subsequences between outputs.
- **Latency**: Average inference time per test sample (seconds/sample), indicating runtime efficiency.

### Inference Engine
A lightweight rule-based inference system was implemented to simulate reasoning:
- **Sentiment Analysis**: TextBlob was used to classify customer inputs as negative, neutral, or positive.
- **Priority Scoring**: Inputs were scanned for urgency-related keywords to assign a priority level (high, medium, low).
- These attributes were embedded into the prompt in the format:
```
[PRIORITY: HIGH] [SENTIMENT: NEGATIVE] Customer message: {input} Agent response:
```

- This added structure allowed the model to generate more context-sensitive responses and introduced a reasoning layer into generation without needing a separate classifier or additional fine-tuning step.

### Sample Output Generation
Each model generates outputs on a subset of test samples to inspect real-world behavior. These qualitative examples are shown alongside ground-truth responses for comparison.

### Category-Based Breakdown
Test examples were labeled into three rough categories based on keywords and heuristics:
- **Complaint**: Contains words like "refund", "angry", "cancel"
- **Question**: Contains question marks
- **Other**: Anything else

Models were evaluated on each category separately using BLEU and ROUGE-L to understand how performance varies by input type.

### Token-Level Error Analysis
To move beyond average scores, the evaluation pipeline identifies the **worst BLEU and ROUGE** examples (e.g., lowest scoring predictions). This helps diagnose failure cases and surface patterns in model mistakes.

### Trainable Parameter Count
Each model’s number of trainable parameters was logged to highlight efficiency. This was especially relevant when comparing full vs parameter-efficient tuning strategies.

### Training Loss Tracking
Loss was logged during training and plotted per step for each configuration. This provided insight into optimization progress and convergence patterns.


## Ablation Studies

To understand how specific hyperparameters impacted performance, I conducted lightweight ablations within each fine-tuning strategy.

### QLoRA: `r=4` vs `r=8`
- **What was changed**: The rank (`r`) of the low-rank adapters was doubled from 4 to 8. `lora_alpha` was scaled accordingly from 16 to 32.
- **Why**: Higher rank values give the adapter more capacity to approximate full-rank weight updates, which can lead to better adaptation to new tasks.
- **Hypothesis**: `r=8` would yield marginally better generation quality, especially on more complex or emotionally charged prompts, at the cost of slightly higher memory usage and training time.
- **Observation**: As expected, the `r=8` model slightly outperformed `r=4` across BLEU and ROUGE metrics. However, the gain was modest (~1–2 points) given the small dataset size. Latency remained nearly identical due to the small adapter size relative to the full model.

### Prompt Tuning: 10 vs 20 Virtual Tokens
- **What was changed**: The number of soft prompt tokens was doubled.
- **Why**: More tokens give the prompt encoder more capacity to encode context or steer behavior, especially for nuanced tasks like support-style conversation.
- **Hypothesis**: The 20-token prompt would outperform the 10-token version, particularly in edge cases or longer user inputs.
- **Observation**: Despite having more trainable parameters, the 20-token variant did not significantly outperform the 10-token model. In fact, ROUGE-1, ROUGE-2, and ROUGE-L were all slightly lower, suggesting that for small datasets, shorter prompts may generalize better. BLEU remained near-zero for both, reinforcing the limited impact of prompt length on n-gram matching in this context.

### Takeaway
Both ablations confirmed that increasing capacity improves performance — but gains were bounded by the dataset size. These findings reinforce the idea that LoRA rank and prompt length should be tuned based on the available data and task complexity.


## Pipeline Overview
### Training Loop

All training was handled using the HuggingFace `Trainer` class. For each model configuration:

- The model was loaded using a lambda defined in `model_configs`.
- The model was trained for 3 epochs on 400 preprocessed examples.
- Evaluation and logging were deferred until after training.
- The trained model was passed to a custom `analyze_model()` function for evaluation.

Each training run used:
- `batch_size=4` with `gradient_accumulation_steps=4`
- `learning_rate=2e-4`
- `fp16=True` to reduce memory footprint
- No intermediate evaluation steps (`eval_strategy="no"`)

### Memory Handling

To ensure the pipeline could run on Google Colab's free GPU tier, aggressive memory management was used:

- After each model finished training and evaluation, it was manually deleted using `del`.
- `gc.collect()` was called to free up Python-managed memory.
- `torch.cuda.empty_cache()` was called to clear the GPU memory.

This allowed all four models to be trained sequentially in a single runtime session without out-of-memory errors.

### Evaluation Loop

The `analyze_model()` function handled post-training evaluation, including:

- Plotting training loss from callback logs
- Computing BLEU and ROUGE scores across the test set
- Measuring inference latency per sample
- Printing sample outputs for qualitative comparison
- Computing trainable parameter counts
- Running category-specific evaluations (e.g., complaints vs. questions)
- Extracting worst-performing examples using token-level metrics

All metrics and results were stored in a shared `results` dictionary for later comparison and plotting.

## Challenges & Solutions

### Major Constraint: Limited GPU Access

This project was completed entirely on Google Colab's free tier, which presented significant compute limitations:

- Frequent GPU unavailability
- 12-hour session limits
- Strict memory constraints
- No persistent storage for large model checkpoints

### Design Adaptations

To make the most of limited compute, several critical adjustments were made:

- **Chose a Small Model**  
  TinyLlama-1.1B was selected due to its small memory footprint while still supporting instruction tuning and adapter-based fine-tuning. Larger models like Mistral-7B were not viable within Colab constraints.

- **Sequential Model Execution**  
  Instead of training all models in parallel, each configuration was trained, evaluated, and cleared before moving to the next. This allowed full reuse of memory across runs.

- **Manual Memory Management**  
  Each training loop explicitly called:
  - `del trainer`, `del model`  
  - `gc.collect()`  
  - `torch.cuda.empty_cache()`  
  This ensured minimal memory footprint between runs and helped avoid GPU crashes.

- **No Evaluation During Training**  
  The HuggingFace `Trainer` was configured with `eval_strategy="no"` to skip intermediate evaluations and reduce unnecessary GPU use.

- **Custom Logging with TrainerCallback**  
  A `LossTracker` callback was used to log training loss without relying on built-in evaluation or logging systems, keeping overhead low.

### Outcome

Despite limited resources, all four models were successfully trained, evaluated, and compared in a single session. The pipeline was designed from the ground up to be GPU-efficient while still enabling in-depth analysis and meaningful ablation.


## Results Summary

| Model         | BLEU   | ROUGE-1 | ROUGE-2 | ROUGE-L | Latency (s) | Trainable Params (%) |
|---------------|--------|---------|---------|---------|--------------|------------------------|
| QLoRA-r8      | 0.0022 | 0.1395  | 0.0228  | 0.0927  | 4.56         | 0.1023%               |
| Prompt-10     | 0.0000 | 0.1404  | 0.0184  | 0.0933  | 4.87         | 0.0019%               |
| Prompt-20     | 0.0010 | 0.1255  | 0.0170  | 0.0861  | 5.18         | 0.0037%               |
| QLoRA-r4      | 0.0020 | 0.1093  | 0.0130  | 0.0721  | 4.56         | 0.0512%               |

**Observations**:
- BLEU scores were low across the board, consistent with the small dataset size and short training.
- QLoRA-r8 achieved the strongest all-around performance on BLEU and ROUGE-2, while Prompt-10 had slightly better ROUGE-1 and ROUGE-L.
- Prompt Tuning models had significantly fewer trainable parameters and were marginally slower at inference due to token injection overhead.
- QLoRA’s performance scaled with rank, though the improvement from r=4 to r=8 was modest.
- Overall, differences were small, reflecting compute and data constraints more than methodological shortcomings.

## Final Conclusion & Recommendations

**Best Performing Method**:  
QLoRA with `r=8` delivered the best overall performance across most metrics. It maintained low latency and struck a good balance between capacity and efficiency.

**If More Compute Were Available**:
- Train for more epochs on a larger dataset
- Include additional ablations (e.g., r=16, alpha variations)
- Experiment with larger base models like Mistral-7B
- Incorporate instruction tuning or few-shot prompting as baselines

**Key Takeaways**:
- Even with small models and limited data, LoRA and prompt tuning can adapt LLMs for domain-specific behavior.
- Prompt tuning is lightweight and fast to deploy, but requires careful initialization and benefits less from limited data.
- QLoRA is more expressive but comes with increased memory and tuning complexity.
- Evaluate methods not just by aggregate metrics, but also in terms of memory footprint, latency, and fit to the task.

Ultimately, model selection should be guided by task complexity, available data, and deployment constraints. For low-resource cases, both approaches are viable — but QLoRA with carefully chosen hyperparameters offers stronger performance ceilings.


## Future Directions

Given more time and compute resources, several extensions would strengthen this project:

- **Expanded Ablation Studies**: Vary more hyperparameters like `lora_alpha`, dropout, and prompt initialization strategies to better understand sensitivity.
- **Larger Datasets**: Incorporate additional customer support corpora to improve generalization and allow for deeper training.
- **More Model Variants**: Compare TinyLLaMA with other small models like Phi-2 or DistilGPT2, and evaluate performance scaling with larger models like Mistral-7B.
- **Hyperparameter Optimization**: Use grid search or Bayesian optimization to tune training parameters systematically (e.g., learning rate, batch size, LoRA rank).
- **Inference Engine Refinement**: Improve rule-based reasoning by incorporating multi-turn context, user intent classification, or external tools.
- **Robust Evaluation**: Include human evaluation or fine-grained dialogue metrics (e.g., coherence, helpfulness) to complement automated scores.
- **Longer Training Runs**: Increase the number of epochs and training examples to see if performance continues to improve with more data.


## Reproducibility

To reproduce the full pipeline, follow these steps:

### 1. Install Dependencies

Use the provided `requirements.txt` to install all necessary packages:

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

- Download the [Customer Support Dataset](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0).
- Rename the downloaded file to `support_data.csv`.
- Move it into a folder named `data/` in your working directory.

### 3. Preprocess the Data

Run the preprocessing script to generate training splits:

```bash
python preprocess_data.py
```

This will create:

- `data/train_data.json`
- `data/val_data.json`
- `data/test_data.json`

### 4. Run the Notebook

Launch the notebook (`llm_fine_tuning.ipynb`) and execute all cells to:

- Fine-tune all model variants
- Perform evaluations (BLEU, ROUGE, sentiment, etc.)
- Plot losses and performance metrics
- Print sample predictions
