# LLM Oversight Evaluation

An independent AI Safety research project investigating the reliability of LLM-based code correctness judgments and the effects of peer-model deliberation.

## Overview

Large language models are increasingly used to evaluate, review, and oversee the outputs of other AI systems. However, an important question is whether an LLM's judgment remains reliable when assessing another model's output.

This project studies this question in the context of code review. LLMs are asked to evaluate whether a given implementation is correct with respect to a specification. When two models disagree, one model is exposed to the other's judgment and supporting evidence before making a second judgment.

The goal is to examine both independent judgment accuracy and whether deliberation helps models correct mistakes or instead causes previously correct judgments to become incorrect.

## Research Questions

**RQ1.** How accurately do different LLMs independently assess code correctness against a specification?

**RQ2.** When LLMs disagree, how does deliberation affect the correctness of their judgments?

## Method

### Dataset

I constructed a 50-case evaluation set based on [DebugBench](https://huggingface.co/datasets/Rtian/DebugBench) (Tian et al., 2024).

The original cases were filtered and adapted through the following steps:

1. **Semantic bugs only**  
   Syntax-only errors, such as unclosed strings or missing colons, were removed because they can be detected directly by parsers or syntax checkers rather than requiring code-correctness reasoning.

2. **Remove duplicate problem instances**  
   Cases originating from the same underlying LeetCode problem were deduplicated using the problem `slug`.

3. **Remove malformed source cases**  
   Cases containing problems in the original dataset that prevented meaningful code evaluation were excluded.

4. **Reconstruct the expected execution environment**  
   Commonly preloaded libraries and LeetCode data structure definitions were added to the code snippets so that missing imports or standard data structures would not be mistaken for implementation bugs.

5. **Incorporate constraints into the specification**  
   Each case's problem description and input constraints were combined into a single specification. This ensures that models evaluate the implementation under the intended problem assumptions.

### Independent Evaluation

For RQ1, each model independently evaluated all 50 cases.

Each model produced a structured response containing:

- a verdict indicating whether the implementation was incorrect or correct;
- supporting evidence; and
- suspected bug location(s) when a bug was identified.

A verdict was considered accurate when the model correctly identified whether the implementation was incorrect or correct, regardless of whether the reported bug location was accurate.

For cases containing bugs, localization accuracy was evaluated separately by comparing the reported bug location with the ground-truth location.

### Deliberation

For RQ2, cases in which the models disagreed were selected for further deliberation.

Disagreements were classified into two types:

- **Verdict disagreement:** the two models reached different verdicts.
- **Localization disagreement:** the models reached the same verdict, but their localization accuracy differed.

During deliberation, a model was exposed to the other model's judgment and supporting evidence before producing a second judgment.

The second judgment was compared with the model's original judgment to examine whether deliberation resulted in:

- **correction** — an initially incorrect judgment became correct; or
- **corruption** — an initially correct judgment became incorrect.

## Repository Structure

```text
llm-oversight-eval/
├── data/
│   └── dataset.jsonl
│
├── src/
│   ├── baseline.py
│   ├── debate.py
│   └── baseline_results/
│
├── results/
│   └── debate_results/
│
└── tests/
    └── test_deliberation.py

Main Components
data/dataset.jsonl: the 50-case evaluation dataset.
src/baseline.py: runs independent model evaluations for RQ1.
src/debate.py: runs the deliberation procedure for RQ2.
src/baseline_results/: stores independent evaluation results and disagreement cases.
results/debate_results/: stores deliberation results.

Results
The experiments are currently being finalized.

The final report will compare:
independent judgment accuracy across models;
bug localization accuracy;
the types of disagreements between models; and
whether deliberation leads to correction or corruption of judgments.

Limitations
This initial study has several limitations.

First, the evaluation uses a relatively small set of 50 cases and only a limited number of models. Second, the deliberation experiment currently evaluates a single round of interaction. These design choices limit the extent to which the results can be generalized to other models, datasets, or longer deliberation processes.

Future work could evaluate a larger and more diverse set of models and cases, as well as multiple rounds of deliberation.

References
Tian et al. (2024). DebugBench.
Benton et al. (2024). Sabotage Evaluations for Frontier Models.
Ngo et al. (2023). The Alignment Problem from a Deep Learning Perspective.