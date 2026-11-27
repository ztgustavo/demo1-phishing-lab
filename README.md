# 🛡️ Lab Interativo — Demo 1: Detector de URLs Maliciosas com ML

Interface web interativa criada como material complementar de estudo para a **Demo 1** da disciplina **Artificial Intelligence & Robotics Trends** (MBA em Cyber Security — FIAP, turma 11SEGR).

> **Créditos e Referência Original**  
> Este projeto é derivado do material aberto do **Prof. Dr. Ahirton Lopes**:  
> 🔗 [ahirtonlopes/AI-and-Robotics-Trends](https://github.com/ahirtonlopes/AI-and-Robotics-Trends)  
> Notebook de origem: [`Notebooks/Demo1_URL_Phishing_Classifier.ipynb`](https://github.com/ahirtonlopes/AI-and-Robotics-Trends/blob/main/Notebooks/Demo1_URL_Phishing_Classifier.ipynb)

---

## O que você pode experimentar no Lab

- **3 Presets prontos** que reproduzem a evolução da Demo 1 (`Base (palavras)` → `+ Balanceado (A)` → `+ N-gramas 2–4 chars (A+B)`).
- **Customização completa:** tamanho da amostra, limpeza de URL (`http/https/www`), remoção de duplicatas, estratificação, `word` vs `char` vs `char_wb`, `ngram_range`, `max_features`, `class_weight="balanced"` e regularização `C`.
- **Limiar de decisão (*Threshold*) em tempo real** (sem retreinar), mostrando o *trade-off* entre **Recall**, **Precision**, **Falsos Negativos (FN)** e **Falsos Positivos (FP)**.
- **Explicabilidade ao vivo:** digite qualquer URL para ver o risco estimado e exatamente quais pedaços de texto empurraram a decisão para *malicioso* ou *legítimo*.
- **Didática integrada:** tooltips (`?`) em todos os controles e métricas, Glossário Rápido e leitura automática dos resultados.

---

## Como rodar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy gratuito no Streamlit Community Cloud (sem cartão)

1. Acesse [share.streamlit.io](https://share.streamlit.io/) e entre com sua conta do GitHub.
2. Clique em **Create app** → selecione este repositório → arquivo principal `app.py` → **Deploy**.

---

## Dados e Licença

- **Base de dados:** Kaggle URL Dataset (`antonyj453/urldataset`), carregada automaticamente a partir de `Datasets/url_data.csv` do repositório da disciplina ([ahirtonlopes/AI-and-Robotics-Trends](https://github.com/ahirtonlopes/AI-and-Robotics-Trends)).
- **Licença do código:** MIT (veja [LICENSE](LICENSE)).
