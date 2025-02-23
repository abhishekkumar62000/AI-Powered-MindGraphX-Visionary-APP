# Top 32 LLM Interview Questions and Answers

## Prompt-based learning
### Fine-tuning
- Explanation: Modifies the model itself, mitigating bias by adjusting prompts.

## LLM Interview Course
### Real life case studies & code solutions
### Questions & Assessments
### Regular updates
- Generative AI certification
- Real Questions from FAANG & Fortune 500

## Vector store for LLM use cases
### No
- Explanation: Some tasks (summarization, sentiment analysis, translation) do not require context augmentation.

## Aligning LLMs with human values
### Not a technique
- Data Augmentation
- Explanation: General ML technique, not specifically designed for human value alignment.

## Reward hacking in RLHF
### Exploits reward function
- Explanation: Agent finds unintended loopholes to maximize reward without following desired behavior.

## Model adaptation in fine-tuning
### Pre-trained model architecture
- Explanation: Complex architecture allows for greater adaptation to diverse tasks, regardless of dataset size.

## Self-attention mechanism in transformer architecture
### Weigh word importance
- Explanation: Illuminates relative importance of words within a sentence, enhancing context-aware analysis.

## Subword algorithms in LLMs
### Limit vocabulary size
- Explanation: Breaks words into subwords, reducing vocabulary size without losing meaning.

## Adaptive Softmax in LLMs
### Sparse word reps
- Explanation: Uses Zipf's law to group words by frequency, enabling efficient calculations for large vocabularies.

## Configuring randomness in model output layer
### Temperature
- Explanation: Lower temperature assigns higher probability to most likely word; higher temperature increases randomness.

## Masked language modeling
### Autoencoder
- Explanation: Predicts masked tokens in the input sequence to reconstruct the original sentence.

## Scaling model training across GPUs
### FSDP
- Explanation: Shards model parameters and gradients across GPUs for efficient training.

## Quantization in LLM training
### Reduce memory usage
- Explanation: Reduces precision of model weights to save memory.

## Compute-optimal model design
### Optimizing model & data size
- Explanation: Use scaling laws to estimate performance and computational cost, balancing model complexity and training cost.

## Catastrophic forgetting in fine-tuning
### Other tasks perform worse
- Explanation: Adaptation to new tasks causes performance degradation on other tasks.

## PEFT
### True
- Explanation: Updates only a subset of parameters to prevent catastrophic forgetting.

## RLHF
### You can use an algorithm other than PPO
### False
- Explanation: PPO is the most popular but not the only algorithm used in RLHF.

## Group attention in Transformer
### Pre-defined word groups
- Explanation: Uses groups of words based on specific criteria, instead of self-attention on individual words.

## LLM training
### Not directly involved
- Feature engineering
- Explanation: Feature engineering is not a typical step in LLM training.

## LLM pre-training
### General language understanding
- Explanation: Equips LLMs with foundational language understanding.

## LLM training sequence
### A -> C -> B
- Explanation: Pre-training (A), Instruction Fine-tuning (C), RLHF (B).

## Knowledge distillation
### Knowledge Distillation
- Explanation: Small model learns from a large pre-trained model.

## Tokenization method
### WordPiece
- Explanation: Places ## at the beginning of tokens.

## Mixture of Experts (MoE)
### Mixture of Experts (MoE)
- Explanation: Uses gating functions to select experts based on the input.

## Prompt leaking in LLMs
### Hijacking model's output
- Explanation: Extracting sensitive information from the model's response.

## Vector database for multi-dimensional vectors
### Vector Database
- Explanation: Optimized for storing and searching multi-dimensional vectors.

## Vector indexing technique
### Principal Component Analysis
- Explanation: Clusters similar vectors in a flat index.

## Vector index for small review dataset
### Flat Index
- Explanation: Enables exhaustive search for 100% recall.

## Inverted File Index (IVF) tuning
### nlist
- Explanation: Controls the number of clusters in the IVF index.

## Evaluation of factual language summaries
### Not used
- Perplexity
- Explanation: Perplexity measures word prediction, not summary quality.

## Random Projection Index
### Random Projection Index
- Explanation: Reduces vector size through multiplication with a random matrix.

## Pre-filtering order in vector database
### Meta-data filtering --> Top-K
- Explanation: Filters out vectors based on meta-data and then retrieves top-k results.

## Post-filtering order in vector database
### Top-K --> Meta-data filtering
- Explanation: Retrieves top-k vectors and then filters based on meta-data.