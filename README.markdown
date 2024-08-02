# Deep Learning in Natural Language Processing (NLP)

## Overview
This repository contains implementations and experiments related to parameter-efficient fine-tuning and alignment of language models, focusing on techniques like Low Rank Adaptation (LoRA) and Direct Preference Optimization (DPO).

## Table of Contents
1. Parameter Efficient Fine-Tuning with LoRA
2. Direct Preference Optimization (DPO)
3. Factuality in Language Models
4. Sentence Embeddings (SimCSE)

## Parameter Efficient Fine-Tuning with LoRA

This section implements Low Rank Adaptation (LoRA) on BERT's attention mechanism and fine-tunes a bert-base-cased model on the Yelp reviews dataset.

Key code snippet for LoRA implementation:

```python
class BertSelfAttentionLora(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # ... (initialization code)
        
        # LoRA parameters
        self.rank = 4
        self.A_q = nn.Parameter(torch.randn(self.all_head_size, self.rank))
        self.B_q = nn.Parameter(torch.randn(self.rank, config.hidden_size))
        self.A_k = nn.Parameter(torch.randn(self.all_head_size, self.rank))
        self.B_k = nn.Parameter(torch.randn(self.rank, config.hidden_size))

    def forward(self, hidden_states, ...):
        # ... (forward pass code)
        lora_query_layer = self.query(hidden_states) + self.A_q.matmul(self.B_q.matmul(hidden_states.transpose(-1, -2))).transpose(-1, -2)
        lora_key_layer = self.key(hidden_states) + self.A_k.matmul(self.B_k.matmul(hidden_states.transpose(-1, -2))).transpose(-1, -2)
```

## Direct Preference Optimization (DPO)

This section implements DPO for aligning language models with human preferences, using GPT-2 as a demonstration model.

Key code snippet for DPO loss calculation:

```python
def compute_log_probability(model, sentence):
    tokenize_input = tokenizer.encode(sentence, return_tensors="pt")
    outputs = model(tokenize_input, labels=tokenize_input)
    log_prob = outputs.loss * -1
    return log_prob.item()
```

Compute DPO loss:

β = 1

$$ \text{DPO Loss} = -\log\left(\frac{\exp(\beta \cdot (\log p_\theta(y_w|x) - \log p_\text{ref}(y_w|x)))}{\exp(\beta \cdot (\log p_\theta(y_w|x) - \log p_\text{ref}(y_w|x))) + \exp(\beta \cdot (\log p_\theta(y_l|x) - \log p_\text{ref}(y_l|x)))}\right) $$

Where:
- $p_\theta$ is the fine-tuned model
- $p_\text{ref}$ is the reference model
- $y_w$ is the winning (preferred) response
- $y_l$ is the losing (non-preferred) response
- $x$ is the input prompt

## Factuality in Language Models

This section discusses:
- Challenges in generating factual content from Large Language Models (LLMs)
- Methods to enhance factuality through fine-tuning
- Comparison of reference-based and reference-free factuality evaluations

The paper "Fine-tuning Language Models for Factuality" tackles the challenge of generating factually correct content from large language models. Factual string generation from LLMs is a challenge which arises from their training goals. Typical training approaches like maximum likelihood do not inherently prioritize fact-checking, leading to the possibility of the model favoring inaccurate responses.

To address this, the paper introduces a method for enhancing the factuality of LLMs through fine-tuning, using the Direct Preference Optimization (DPO) algorithm. This algorithm elicits preference data using two methods: reference-based and reference-free factuality evaluations.

The difference between reference-based and reference-free factuality assessments lies in their reliance on external data. Reference-based evaluations necessitate an external source for fact verification, whereas reference-free evaluations depend solely on the model's internal assessment of its responses.

## Sentence Embeddings (SimCSE)

This section explores:
- Differences between sentence and word embeddings
- Unsupervised and supervised SimCSE approaches
- Key properties for improving sentence embeddings: alignment and uniformity

Sentence embeddings represent the overall meaning of a sentence, considering the context and interactions between words, while word embeddings focus on individual words without context.

Unsupervised SimCSE predicts the input sentence itself using dropout as noise and uses other sentences in the same batch as negatives. Supervised SimCSE uses annotated sentence pairs from NLI datasets, with "entailment" as positives and "contradiction" as hard negatives.

The two key properties for improving sentence embeddings in supervised SimCSE are alignment and uniformity:

Alignment: 𝑨𝒍𝒊𝒈𝒏𝒎𝒆𝒏𝒕 = 𝔼_(𝒙,𝒙^+ )~𝒑_𝒑𝒐𝒔 [‖𝒇(𝒙)−𝒇(𝒙^+ )‖^𝟐 ]

Uniformity: 𝑼𝒏𝒊𝒇𝒐𝒓𝒎𝒊𝒕𝒚 = 𝐥𝐨𝐠 𝔼_(𝒙,𝒚~𝒑_𝒅𝒂𝒕𝒂) [𝒆^(−𝟐‖𝒇(𝒙)−𝒇(𝒚)‖^𝟐 ) ]


## Conclusion

This repository demonstrates advanced techniques in NLP, focusing on improving model efficiency, alignment with human preferences, and factual accuracy in language generation.


