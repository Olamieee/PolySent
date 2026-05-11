# VibeSentry 🌍

**Multilingual Sentiment and Content Analysis for African Languages**

VibeSentry is an open-source NLP project focused on sentiment analysis and content moderation for low-resource African languages. It provides separate, language-specific models for Yoruba, Nigerian Pidgin, Kinyarwanda, and English — languages that are severely underrepresented in mainstream NLP research.

---

## Why VibeSentry?

Most sentiment analysis tools are built for English, Mandarin, or European languages. When applied to African languages, they fail — not because the languages are difficult, but because the models were never trained on them. Yoruba has over 50 million speakers. Nigerian Pidgin is spoken across West Africa by over 75 million people. Kinyarwanda is the primary language of Rwanda with over 12 million speakers.

VibeSentry treats each language as a first-class citizen with its own dedicated model, preprocessing pipeline, and evaluation.

---

## Languages and Tasks

| Language | Task | Model | Dataset |
|---|---|---|---|
| Yoruba | Sentiment Analysis (Positive / Neutral / Negative) | BiLSTM | AfriSenti-Yoruba (combined) |
| Nigerian Pidgin | Sentiment Analysis (Positive / Neutral / Negative) | CNN | AfriSenti-Pidgin tweets |
| Kinyarwanda | Hate Speech and Sarcasm Detection (Hate / Normal / Sarcasm) | CNN | Custom annotated dataset |
| English | Sentiment Analysis | BiLSTM | Standard English corpus |

---

## Project Structure

```
VibeSentry/
├── models/
│   ├── eng/
│   ├── kinyarwanda/
│   ├── pidgin/
│   ├── swahili/
│   └── yoruba/
├── static/
├── templates/
│   └── index.html
├── .gitignore
├── app.py
├── main.py
├── requirements.txt
├── Procfile
└── README.md
```

---

## Model Details

### Yoruba Sentiment Analysis
- **Architecture:** Bidirectional LSTM (BiLSTM)
- **Preprocessing:** spaCy Yoruba tokenizer, Yoruba and English stopword removal, URL/mention/hashtag stripping
- **Training:** Oversampled to handle class imbalance, 80/20 train-test split
- **Classes:** Positive, Neutral, Negative
- **Embedding:** Learned embedding layer (100-dim, vocab size 10,000)

### Nigerian Pidgin Sentiment Analysis
- **Architecture:** 1D Convolutional Neural Network (CNN)
- **Preprocessing:** Custom Pidgin stopword list (di, abeg, wetin, sef, abi, dey, na, o, sha, joor), URL/mention/hashtag stripping
- **Training:** Oversampled to handle class imbalance, 80/20 train-test split
- **Classes:** Positive, Neutral, Negative
- **Embedding:** Learned embedding layer (100-dim, vocab size 10,000)

### Kinyarwanda Hate Speech and Sarcasm Detection
- **Architecture:** 1D Convolutional Neural Network (CNN)
- **Preprocessing:** Lowercasing, punctuation removal, repeated character normalization
- **Training:** Oversampled to handle class imbalance, 80/20 train-test split
- **Classes:** Hate, Normal, Sarcasm
- **Note:** This model was developed in collaboration as part of a research project on content moderation for low-resource Bantu languages.

---

## Key Challenges Addressed

**Code-switching** — Nigerian Pidgin text frequently mixes Pidgin with English and Yoruba. The preprocessing pipeline handles this without stripping linguistic context.

**Morphological complexity** — Yoruba uses tonal diacritics (e.g. è, ó, ẹ) that carry meaning. The spaCy Yoruba tokenizer preserves these rather than stripping them as noise.

**Class imbalance** — All datasets had significant class imbalance. Oversampling was applied to the minority classes before training to prevent the model from defaulting to majority class predictions.

**Low-resource data scarcity** — Yoruba and Pidgin have limited labelled data. The models are designed to be lightweight and effective under data constraints.

---

## Datasets

- **AfriSenti** — A Twitter sentiment benchmark for African languages (Muhammad et al., 2023). Used for Yoruba and Pidgin. Available at: [github.com/afrisenti-nlp/afrisenti-semeval-2023](https://github.com/afrisenti-nlp/afrisenti-semeval-2023)
- **Kinyarwanda dataset** — Custom annotated dataset combining hard-text examples and realistic hate/sarcasm/normal samples.

---

## Roadmap

The following improvements are planned:

- [ ] Upgrade Yoruba and Pidgin models to transformer-based fine-tuning using AfroXLMR or AfriBERTa
- [ ] Evaluate against SemEval-2025 Task 11 baselines for Yoruba and Pidgin
- [ ] Add Streamlit demo app for live inference
- [ ] Extend to Igbo and Hausa
- [ ] Add cross-lingual transfer experiments across all four languages
- [ ] Write up findings as a short paper for AfricaNLP 2026 / IndabaX 2027

---

## About

Built by **Alonge Olamide Samson** — Machine Learning Engineer and NLP researcher based in Nigeria, with a focus on low-resource and multilingual NLP for African languages.

- GitHub: [github.com/Olamieee](https://github.com/Olamieee)
- LinkedIn: [linkedin.com/in/alonge-olamide-493237242](https://linkedin.com/in/alonge-olamide-493237242)
- Email: alongeola16@gmail.com

---

## Citation

If you use VibeSentry datasets or models in your research, please cite:

```
@misc{vibesentry2024,
  author = {Alonge, Olamide Samson},
  title = {VibeSentry: Multilingual Sentiment and Content Analysis for African Languages},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/Olamieee}
}
```

---

## References

Muhammad, S. H., et al. (2023). AfriSenti: A Twitter Sentiment Analysis Benchmark for African Languages. EMNLP 2023.

Muhammad, S. H., et al. (2025). SemEval-2025 Task 11: Bridging the Gap in Text-Based Emotion Detection. ACL 2025.

Adelani, D. I., et al. (2022). MasakhaNER 2.0: Africa-centric Transfer Learning for Named Entity Recognition. EMNLP 2022.

---

*VibeSentry is part of a broader effort to build NLP infrastructure for African languages. Contributions, feedback, and collaborations are welcome.*
