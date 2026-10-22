# Question 19: English-to-Hinglish Machine Translation

## Problem Statement
A language technology company wants to translate English sentences into Hinglish to support multilingual communication. Download the English–Hinglish Machine Translation dataset from Kaggle.

## Dataset
- **Name**: Kaggle English–Hinglish Machine Translation Dataset (`eng_hinglish_translation.csv`)
- **Location**: `data/eng_hinglish_translation.csv`
- **Volume**: 2,500 parallel English and Hinglish sentence pairs.

## Model Selection & Justification
- **Model**: Sequence-to-Sequence (Seq2Seq) Encoder–Decoder architecture with recurrent LSTM cells and Teacher Forcing.
- **Justification**: Translating between two languages involves arbitrary input and output lengths where source words do not map 1-to-1 with target tokens. An Encoder LSTM condenses the full English input sentence into persistent hidden/cell state vectors, which the Decoder LSTM recursively unfolds token-by-token into the target Hinglish utterance.

## How to Run
```bash
python main.py
```

## Generated Outputs
- Average BLEU metric on the held-out test translation set.
- Complete English-to-Hinglish translations on 5 unseen test sentences.
- `actual_vs_predicted.png`: Training loss convergence curve over epochs.
