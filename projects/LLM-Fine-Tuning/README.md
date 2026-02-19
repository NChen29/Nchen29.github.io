# LLM Fine Tuning Projects

**Summary**

A parameter-efficient fine-tuned (PEFT) Large Language Model that adopts the dark, morbid, and emotionless persona of Wednesday Addams. This project takes a base Llama-2-7b model and fine-tunes it using QLoRA (Quantized Low-Rank Adaptation) on a custom, manually curated dataset.
Notebooks focused on fine-tuning Large Language Models.


**First two notebooks are to gain familarity with thw process. **

Fine_Tuning_Llama.ipynb

fine_tuning_GPT_2.ipynb

Final Project notebook is for Wednesday chatbot. 

**End to end process:**

Stage 1: Data Ingestion & Formatting
Manual Curation: Created a custom dataset of Wednesday Addams quotes and insults.

Hugging Face Datasets: Converted the raw CSV data into a Hugging Face Dataset object. This utilizes Apache Arrow under the hood, allowing for highly efficient, memory-mapped data processing crucial for scalable ML workflows.

Stage 2: Tokenization & Prompt Engineering
Tokenizer Initialization: Loaded the Llama 2 tokenizer and configured the padding token to the EOS (End of Sentence) token to fix native Llama 2 padding bugs.

Structural Scaffolding: Engineered a strict prompt template (### Instruction, ### User, ### Response) to teach the model conversational turn-taking.

Vectorization: Applied a .map() function to dynamically transform the formatted text into mathematically formatted tensors (input_ids and attention_mask).

Stage 3: Model Initialization & QLoRA Setup
4-Bit Quantization: Loaded the base Llama-2-7b-hf model in 4-bit precision (NormalFloat4) via BitsAndBytesConfig to reduce VRAM requirements from ~28GB to ~4GB.

Base Model Freezing: Locked all original 7 billion weights to prevent standard full-parameter updates.

LoRA Injection: Configured and attached tiny, trainable Low-Rank Adapters to the attention layers (q_proj, v_proj) using peft.get_peft_model, reducing trainable parameters to less than 0.1% of the total model.

Stage 4: Training Configuration & Execution
Hyperparameter Tuning: Configured TrainingArguments utilizing paged_adamw_8bit optimizer and gradient accumulation (gradient_accumulation_steps=4) to simulate larger batch sizes on constrained hardware.

Dynamic Padding: Implemented DataCollatorForLanguageModeling to dynamically pad batches on the fly, saving computational overhead.

Execution: Ran the Trainer loop, successfully dropping the training loss from 2.0 to 0.80, indicating optimal pattern recognition without severe overfitting.

Stage 5: Inference & Evaluation
Evaluation Mode: Switched the model to inference mode and enabled caching.

Pattern Completion: Passed the exact training scaffold to the model, leaving the ### Response: section blank to trigger the fine-tuned weights to generate persona-accurate responses.

Hallucination Mitigation: Implemented custom Python string parsing to prevent "run-on sentence" hallucinations (e.g., the model attempting to write the next user instruction).
