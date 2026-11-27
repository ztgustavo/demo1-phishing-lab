"""
Laboratório Interativo — Demo 1: Detector de URLs Maliciosas com Machine Learning
MBA Cyber Security (FIAP) — AI & Robotics Trends

Pronto para deploy gratuito no Streamlit Community Cloud (share.streamlit.io).
"""

from __future__ import annotations

import re
import time
import urllib.request
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

DATA_URL = (
    "https://raw.githubusercontent.com/ahirtonlopes/"
    "AI-and-Robotics-Trends/main/Datasets/url_data.csv"
)

PRESET_URLS = [
    ("google.com", "legítima"),
    ("fiap.com.br", "legítima"),
    ("github.com/ahirtonlopes", "legítima"),
    ("paypa1-secure-login.com/verify/account", "maliciosa"),
    ("secure-itau.com.br-atualizacao.top/login.php", "maliciosa"),
    ("accounts.google.com.verify-session.ru/signin", "maliciosa"),
    ("192.168.4.7/wp-admin/login.php", "maliciosa"),
    ("empresa-2015-promocao.com/file.exe", "maliciosa"),
]


def limpar_url(u: Any) -> str:
    s = re.sub(r"\s+", "", str(u).lower())
    s = re.sub(r"^https?://", "", s)
    s = re.sub(r"^www\.", "", s)
    return s.rstrip("/")


def find_or_download_dataset() -> Path:
    here = Path(__file__).resolve().parent
    candidates = [
        here / "url_data.csv",
        here / "Datasets" / "url_data.csv",
        here.parents[2] / "Datasets" / "url_data.csv" if len(here.parents) >= 3 else here / "url_data.csv",
        Path("/tmp/url_data.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    target = Path("/tmp/url_data.csv")
    urllib.request.urlretrieve(DATA_URL, target)
    return target


@st.cache_data(show_spinner="Baixando e preparando a base de URLs (~420 mil linhas)...")
def load_dataset_cache() -> dict[str, Any]:
    csv_path = find_or_download_dataset()
    raw = pd.read_csv(csv_path)
    labels = (raw["label"] == "bad").astype(np.int8)
    raw_urls = raw["url"].astype(str)
    clean_urls = raw_urls.map(limpar_url)

    df_raw = pd.DataFrame({"url": raw_urls, "malicioso": labels})
    df_clean = pd.DataFrame({"url": clean_urls, "malicioso": labels})
    return {
        "total_raw": len(raw),
        "raw_df": df_raw,
        "raw_dedup": df_raw.drop_duplicates("url"),
        "clean_df": df_clean,
        "clean_dedup": df_clean.drop_duplicates("url"),
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "total_attacks": int(tp + fn),
        "total_legit": int(tn + fp),
    }


@st.cache_resource(max_entries=12, show_spinner="Treinando TF-IDF + Regressão Logística...")
def train_model_cached(
    sample_size: int,
    clean_url: bool,
    dedup: bool,
    test_size: float,
    stratify: bool,
    analyzer: str,
    ngram_min: int,
    ngram_max: int,
    max_features: int,
    sublinear_tf: bool,
    balanced: bool,
    c_val: float,
    seed: int,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    cache = load_dataset_cache()

    total_raw = cache["total_raw"]
    if clean_url and dedup:
        df = cache["clean_dedup"]
    elif clean_url and not dedup:
        df = cache["clean_df"]
    elif not clean_url and dedup:
        df = cache["raw_dedup"]
    else:
        df = cache["raw_df"]

    dups_removed = total_raw - len(df)
    if sample_size > 0 and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=seed)

    strat = df["malicioso"] if stratify else None
    treino, teste = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=strat
    )

    vec = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=(ngram_min, max(ngram_min, ngram_max)),
        max_features=max_features,
        sublinear_tf=sublinear_tf,
    )
    X_tr = vec.fit_transform(treino["url"])
    X_te = vec.transform(teste["url"])

    modelo = LogisticRegression(
        C=c_val,
        max_iter=1000,
        class_weight="balanced" if balanced else None,
        random_state=seed,
    )
    modelo.fit(X_tr, treino["malicioso"])

    y_test = teste["malicioso"].to_numpy()
    probs_test = modelo.predict_proba(X_te)[:, 1]
    m_dummy = compute_metrics(y_test, np.zeros_like(y_test))

    # Curva de trade-off
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    curve = []
    for t in thresholds:
        mt = compute_metrics(y_test, (probs_test >= t).astype(int))
        curve.append({"threshold": t, "recall": mt["recall"], "precision": mt["precision"]})

    # Top pesos (interpretabilidade)
    nomes = np.array(vec.get_feature_names_out())
    pesos = modelo.coef_[0]
    ordem = np.argsort(pesos)
    min_len = 4 if (analyzer != "word" and ngram_max >= 4) else (3 if analyzer != "word" and ngram_max >= 3 else 1)
    elegiveis = [int(i) for i in ordem if len(nomes[i]) >= min_len]
    if len(elegiveis) < 20:
        elegiveis = [int(i) for i in ordem]

    top_mal = pd.DataFrame(
        [{"termo": str(nomes[i]), "peso": round(float(pesos[i]), 3)} for i in elegiveis[-12:][::-1]]
    )
    top_leg = pd.DataFrame(
        [{"termo": str(nomes[i]), "peso": round(float(pesos[i]), 3)} for i in elegiveis[:12]]
    )

    return {
        "modelo": modelo,
        "vec": vec,
        "y_test": y_test,
        "probs_test": probs_test,
        "baseline": m_dummy,
        "curve": pd.DataFrame(curve),
        "top_mal": top_mal,
        "top_leg": top_leg,
        "vocab_size": int(len(nomes)),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "data_stats": {
            "total_raw": total_raw,
            "dups_removed": dups_removed,
            "used_rows": len(df),
            "train_rows": len(treino),
            "test_rows": len(teste),
            "pct_malicious": float(df["malicioso"].mean()),
        },
    }


def explain_url(
    modelo: LogisticRegression,
    vec: TfidfVectorizer,
    clean_url: bool,
    url: str,
    threshold: float,
) -> dict[str, Any]:
    cleaned = limpar_url(url) if clean_url else str(url)
    X = vec.transform([cleaned])
    prob = float(modelo.predict_proba(X)[0, 1])
    verdict = "BLOQUEAR" if prob >= threshold else "LIBERAR"

    nomes = vec.get_feature_names_out()
    pesos = modelo.coef_[0]
    row = X.tocoo()
    contribs = [
        {"termo": str(nomes[i]), "contrib": round(float(v * pesos[i]), 3)}
        for i, v in zip(row.col, row.data)
    ]
    contribs.sort(key=lambda x: x["contrib"], reverse=True)
    return {
        "cleaned": cleaned,
        "prob": prob,
        "verdict": verdict,
        "intercept": round(float(modelo.intercept_[0]), 3),
        "tokens": vec.build_analyzer()(cleaned),
        "matched": len(contribs),
        "push_mal": [c for c in contribs if c["contrib"] > 0][:8],
        "push_leg": [c for c in contribs if c["contrib"] < 0][-8:][::-1],
    }


def plot_confusion_matrix(m: dict[str, Any]):
    mat = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    labels_txt = [
        [f"{m['tn']:,}\n(OK)", f"{m['fp']:,}\n(Alarme falso)"],
        [f"{m['fn']:,}\n(Passou!)", f"{m['tp']:,}\n(Ataque pego!)"],
    ]
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    ax.imshow(mat, cmap="Blues", aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Prev: Legítimo", "Prev: Malicioso"], color="#e6edf7", fontsize=9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Real: Legítimo", "Real: Malicioso"], color="#e6edf7", fontsize=9)
    for r in range(2):
        for c in range(2):
            bg_val = mat[r, c] / max(mat.max(), 1)
            txt_color = "#f8fafc" if bg_val > 0.55 else "#0f172a"
            if r == 1 and c == 0:
                txt_color = "#ffe4e6" if bg_val > 0.55 else "#be123c"
            ax.text(c, r, labels_txt[r][c], ha="center", va="center", color=txt_color, fontweight="bold", fontsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
    fig.tight_layout()
    return fig


def plot_tradeoff_curve(curve: pd.DataFrame, current_threshold: float):
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#111827")
    ax.plot(curve["threshold"], curve["recall"] * 100, marker="o", color="#22c55e", linewidth=2.2, label="Recall (ataques pegos)")
    ax.plot(curve["threshold"], curve["precision"] * 100, marker="o", color="#38bdf8", linewidth=2.2, label="Precision (alertas reais)")
    ax.axvline(current_threshold, color="#f59e0b", linestyle="--", linewidth=1.8, label=f"Limiar atual ({current_threshold:.2f})")
    ax.set_xlim(0.05, 1.0)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Limiar de decisão (Threshold)", color="#cbd5e1", fontsize=9)
    ax.set_ylabel("%", color="#cbd5e1", fontsize=9)
    ax.tick_params(colors="#cbd5e1", labelsize=8)
    ax.grid(True, linestyle=":", alpha=0.25, color="#94a3b8")
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f1f5f9", fontsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
    fig.tight_layout()
    return fig


def init_state() -> None:
    defaults = {
        "sample_size": 50000,
        "clean_url": True,
        "dedup": True,
        "stratify": True,
        "test_size": 0.20,
        "seed": 42,
        "analyzer": "word",
        "ngram_min": 1,
        "ngram_max": 1,
        "max_features": 80000,
        "sublinear_tf": True,
        "balanced": False,
        "c_val": 1.0,
        "threshold": 0.50,
        "history": [],
        "last_sig": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def apply_preset(n: int) -> None:
    st.session_state.clean_url = True
    st.session_state.dedup = True
    st.session_state.stratify = True
    st.session_state.sublinear_tf = True
    st.session_state.c_val = 1.0
    st.session_state.threshold = 0.50
    if n == 1:
        st.session_state.analyzer = "word"
        st.session_state.ngram_min = 1
        st.session_state.ngram_max = 1
        st.session_state.balanced = False
    elif n == 2:
        st.session_state.analyzer = "word"
        st.session_state.ngram_min = 1
        st.session_state.ngram_max = 1
        st.session_state.balanced = True
    elif n == 3:
        st.session_state.analyzer = "char"
        st.session_state.ngram_min = 2
        st.session_state.ngram_max = 4
        st.session_state.balanced = True


def render_app() -> None:
    st.set_page_config(
        page_title="Demo 1 — Detector de URLs Maliciosas",
        page_icon="🛡️",
        layout="wide",
    )
    init_state()

    st.markdown(
        """
        <style>
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(6, minmax(0, 1fr));
            gap: 10px;
            margin: 8px 0 14px;
        }
        @media (max-width: 1024px) {
            .kpi-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
        }
        @media (max-width: 640px) {
            .block-container { padding-top: 1.2rem; padding-left: 0.9rem; padding-right: 0.9rem; }
            .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
        }
        .kpi-card {
            background: #161b26;
            border: 1px solid #2b354b;
            border-radius: 10px;
            padding: 10px 12px;
        }
        .kpi-lbl {
            font-size: 0.74rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.03em;
            font-weight: 600;
        }
        .kpi-val {
            font-size: 1.55rem;
            font-weight: 700;
            line-height: 1.2;
            margin: 3px 0 2px;
            font-variant-numeric: tabular-nums;
            color: #f8fafc;
        }
        .kpi-val.accent { color: #38bdf8; }
        .kpi-val.good { color: #4ade80; }
        .kpi-val.bad { color: #fb7185; }
        .kpi-val.warn { color: #fbbf24; }
        .kpi-sub {
            font-size: 0.73rem;
            color: #94a3b8;
            line-height: 1.25;
        }
        .gloss-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 8px;
            margin: 6px 0 12px;
        }
        @media (max-width: 1024px) {
            .gloss-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }
        @media (max-width: 520px) {
            .gloss-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 6px; }
        }
        .gloss-card {
            background: #131b2e;
            border: 1px solid #263554;
            border-radius: 9px;
            padding: 8px 10px;
            font-size: 0.78rem;
            line-height: 1.32;
            color: #cbd5e1;
        }
        .gloss-card strong {
            color: #38bdf8;
            display: block;
            margin-bottom: 2px;
            font-size: 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🛡️ Lab Interativo — Demo 1: Detector de URLs Maliciosas")
    st.caption(
        "Baseado na Aula 1 (MBA Cyber Security — FIAP). "
        "Escolha um Preset abaixo ou abra **⚙️ Personalizar Parâmetros** para testar combinações."
    )

    # Detecta qual preset corresponde aos parâmetros atuais
    cur_cfg = (
        st.session_state.analyzer,
        int(st.session_state.ngram_min),
        int(st.session_state.ngram_max),
        bool(st.session_state.balanced),
        bool(st.session_state.clean_url),
        bool(st.session_state.dedup),
    )
    active_preset = None
    if cur_cfg == ("word", 1, 1, False, True, True):
        active_preset = 1
    elif cur_cfg == ("word", 1, 1, True, True, True):
        active_preset = 2
    elif cur_cfg == ("char", 2, 4, True, True, True):
        active_preset = 3

    # Presets no topo + Glossário (on_click evita duplo rerun e "fantasma" dos botões!)
    p1, p2, p3, p4 = st.columns([1.2, 1.3, 1.5, 1.2])
    with p1:
        st.button(
            "Preset 1: Base (palavras)",
            key="btn_preset_1",
            use_container_width=True,
            type="primary" if active_preset == 1 else "secondary",
            on_click=apply_preset,
            args=(1,),
            help="Modelo inicial da Demo 1: palavras inteiras e sem balanceamento. Acurácia alta, mas deixa passar muitos ataques.",
        )
    with p2:
        st.button(
            "Preset 2: + Balanceado (A)",
            key="btn_preset_2",
            use_container_width=True,
            type="primary" if active_preset == 2 else "secondary",
            on_click=apply_preset,
            args=(2,),
            help="Ativa class_weight='balanced' para dar mais peso aos ataques durante o treino.",
        )
    with p3:
        st.button(
            "Preset 3: + N-gramas 2–4 chars (A+B)",
            key="btn_preset_3",
            use_container_width=True,
            type="primary" if active_preset == 3 else "secondary",
            on_click=apply_preset,
            args=(3,),
            help="Melhor modelo da Demo 1: quebra URLs em pedaços de 2 a 4 letras + pesos balanceados.",
        )
    with p4:
        show_glossary = st.toggle("Glossário Rápido", key="toggle_glossary", value=False, help="Mostra um resumo dos 8 termos principais usados nesta demo.")

    glossary_slot = st.empty()
    if show_glossary:
        glossary_slot.markdown(
            """
            <div class="gloss-grid">
              <div class="gloss-card"><strong>Acurácia</strong>% de acertos totais. Engana em base desbalanceada (dizer "legítimo" para tudo já dá ~85%).</div>
              <div class="gloss-card"><strong>Recall</strong>Dos ataques reais, quantos o alarme pegou. Métrica mais crítica em segurança.</div>
              <div class="gloss-card"><strong>Precision</strong>Dos alertas emitidos, quantos eram ataque real. Mede o ruído (falsos positivos) no SOC.</div>
              <div class="gloss-card"><strong>Limiar (Threshold)</strong>Nota de corte (0 a 1). Acima bloqueia; abaixo libera. Controla o trade-off Recall × FP.</div>
              <div class="gloss-card"><strong>Falso Negativo (FN)</strong>Ataque real liberado como legítimo (o alarme não tocou com o ladrão dentro).</div>
              <div class="gloss-card"><strong>Falso Positivo (FP)</strong>Site legítimo bloqueado por engano (alarme tocou à toa).</div>
              <div class="gloss-card"><strong>TF-IDF</strong>Converte texto da URL em números, dando peso maior a pedaços raros e distintivos.</div>
              <div class="gloss-card"><strong>Regressão Logística</strong>Modelo transparente que soma pesos de cada pedaço para calcular o risco (0–100%).</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Painel de parâmetros direto na página principal (visível e fácil de usar no celular e no desktop!)
    with st.expander("⚙️ Personalizar Parâmetros (Dados, TF-IDF e Regressão Logística)", expanded=False):
        col_d, col_v, col_m = st.columns(3)

        with col_d:
            st.markdown("**1. Dados & Pré-processamento**")
            sample_size = st.select_slider(
                "Tamanho da amostra",
                options=[20000, 50000, 100000, 150000],
                key="sample_size",
                format_func=lambda x: f"{x // 1000}k URLs",
                help="Quantas URLs da base usar no treino+teste. 50k é rápido (~2s) e excelente para experimentar.",
            )
            clean_url = st.checkbox(
                "Padronizar URL (http/https/www)",
                key="clean_url",
                help="Deixa tudo minúsculo e remove http://, https://, www. e barra final para evitar atalhos artificiais.",
            )
            dedup = st.checkbox(
                "Remover URLs duplicadas",
                key="dedup",
                help="Evita que a mesma URL caia no treino e no teste ('colar na prova') inflando a nota.",
            )
            stratify = st.checkbox(
                "Divisão estratificada",
                key="stratify",
                help="Mantém a mesma proporção (~15%) de URLs maliciosas no treino e no teste.",
            )
            test_size = st.slider(
                "Proporção de teste",
                0.10, 0.40, step=0.05, key="test_size",
                help="Fatia escondida do modelo para a 'prova final' (padrão: 20%, regra 80/20).",
            )
            seed = st.number_input(
                "Seed (semente aleatória)",
                min_value=0, max_value=9999, step=1, key="seed",
                help="Fixa o sorteio para que repetir o experimento dê sempre o mesmo resultado.",
            )

        with col_v:
            st.markdown("**2. Vetorização (TF-IDF)**")
            analyzer = st.selectbox(
                "Analisador (como quebra a URL)",
                options=["word", "char", "char_wb"],
                key="analyzer",
                help="'word' olha palavras inteiras; 'char' olha pedaços de letras (captura truques como paypa1 e amaz0n).",
            )
            c1, c2 = st.columns(2)
            ngram_min = c1.number_input(
                "N-gram mín", min_value=1, max_value=5, key="ngram_min",
                help="Menor tamanho de pedaço extraído (ex.: 2 em char gera pares de letras).",
            )
            ngram_max = c2.number_input(
                "N-gram máx", min_value=int(ngram_min), max_value=5,
                value=max(int(ngram_min), int(st.session_state.ngram_max)),
                key="ngram_max",
                help="Maior tamanho de pedaço extraído (ex.: 2 a 4 cria blocos de 2, 3 e 4 letras).",
            )
            max_features = st.slider(
                "Vocabulário máximo",
                10000, 150000, step=10000, key="max_features",
                help="Limite de pedaços/palavras diferentes memorizados pelo TF-IDF.",
            )
            sublinear_tf = st.checkbox(
                "sublinear_tf (escala log)",
                key="sublinear_tf",
                help="Usa escala logarítmica na repetição de um mesmo pedaço dentro da URL.",
            )

        with col_m:
            st.markdown("**3. Regressão Logística**")
            balanced = st.checkbox(
                'class_weight="balanced"',
                key="balanced",
                help="Faz errar um ataque (classe rara, ~15%) pesar ~5,5x mais no treino. Sobe o Recall, mas aumenta FP.",
            )
            c_val = st.slider(
                "Regularização (C)",
                0.1, 10.0, step=0.1, key="c_val",
                help="Menor C = modelo mais conservador; maior C = ajusta mais forte aos dados de treino.",
            )

    # Treina (com cache automático para parâmetros idênticos!)
    result = train_model_cached(
        sample_size=int(sample_size),
        clean_url=bool(clean_url),
        dedup=bool(dedup),
        test_size=float(test_size),
        stratify=bool(stratify),
        analyzer=str(analyzer),
        ngram_min=int(ngram_min),
        ngram_max=int(ngram_max),
        max_features=int(max_features),
        sublinear_tf=bool(sublinear_tf),
        balanced=bool(balanced),
        c_val=float(c_val),
        seed=int(seed),
    )

    ds = result["data_stats"]
    st.info(
        f"**Pipeline:** {ds['total_raw']:,} URLs brutas → **{ds['dups_removed']:,}** duplicatas removidas → "
        f"usando **{ds['used_rows']:,}** URLs (**{ds['pct_malicious']:.1%}** maliciosas) → "
        f"treino **{ds['train_rows']:,}** / teste **{ds['test_rows']:,}** · "
        f"Vocabulário: **{result['vocab_size']:,}** features · Tempo: **{result['elapsed_sec']}s**"
    )

    # Controle de limiar em tempo real (sem retreinar!)
    threshold = st.slider(
        "Limiar de Decisão (Threshold) em Tempo Real — arraste sem retreinar!",
        min_value=0.05, max_value=0.99, step=0.01, key="threshold",
        help="Nota de corte de 0 a 1: acima dela o sistema bloqueia a URL. Menor = pega mais ataques (+Recall) e gera mais alarmes falsos (+FP).",
    )

    y_test = result["y_test"]
    probs_test = result["probs_test"]
    m = compute_metrics(y_test, (probs_test >= threshold).astype(int))
    base = result["baseline"]

    # Registra no histórico da sessão quando mudar a configuração de treino
    sig = (sample_size, clean_url, dedup, stratify, test_size, seed, analyzer, ngram_min, ngram_max, max_features, sublinear_tf, balanced, c_val)
    if st.session_state.last_sig != sig:
        st.session_state.last_sig = sig
        flags = []
        if not dedup:
            flags.append("sem dedup")
        if not clean_url:
            flags.append("sem limpeza")
        flag_txt = (" · " + ", ".join(flags)) if flags else ""
        st.session_state.history.insert(0, {
            "Config": f"{analyzer} ({ngram_min}-{ngram_max}) · {'bal' if balanced else 'padrão'} · {sample_size//1000}k{flag_txt}",
            "Acur.": f"{m['accuracy']:.1%}",
            "Recall": f"{m['recall']:.1%}",
            "Prec.": f"{m['precision']:.1%}",
            "FN": m["fn"],
            "FP": m["fp"],
        })
        st.session_state.history = st.session_state.history[:12]

    # KPIs principais em grade compacta responsiva (2x3 no mobile, 3x2 no tablet, 6x1 no desktop)
    st.markdown(
        f"""
        <div class="kpi-grid">
          <div class="kpi-card" title="Total de acertos / tudo. Dizer 'legítimo' para tudo já dá ~85%!">
            <div class="kpi-lbl">Acurácia</div>
            <div class="kpi-val accent">{m['accuracy']:.1%}</div>
            <div class="kpi-sub">Baseline preguiçoso: {base['accuracy']:.1%}</div>
          </div>
          <div class="kpi-card" title="Dos ataques reais no teste, quantos % o modelo conseguiu pegar.">
            <div class="kpi-lbl">Recall</div>
            <div class="kpi-val good">{m['recall']:.1%}</div>
            <div class="kpi-sub">Ataques reais pegos</div>
          </div>
          <div class="kpi-card" title="Dos alertas disparados, quantos % eram ataque de verdade.">
            <div class="kpi-lbl">Precision</div>
            <div class="kpi-val">{m['precision']:.1%}</div>
            <div class="kpi-sub">Alertas que eram reais</div>
          </div>
          <div class="kpi-card" title="Média harmônica entre Precision e Recall.">
            <div class="kpi-lbl">F1-Score</div>
            <div class="kpi-val">{m['f1']:.1%}</div>
            <div class="kpi-sub">Equilíbrio Prec × Rec</div>
          </div>
          <div class="kpi-card" title="Ataques reais que o modelo liberou por engano (Falsos Negativos).">
            <div class="kpi-lbl">Passaram (FN)</div>
            <div class="kpi-val bad">{m['fn']:,}</div>
            <div class="kpi-sub">Ataques não detectados</div>
          </div>
          <div class="kpi-card" title="Sites legítimos bloqueados por engano (Falsos Positivos).">
            <div class="kpi-lbl">Falsos (FP)</div>
            <div class="kpi-val warn">{m['fp']:,}</div>
            <div class="kpi-sub">Alarmes falsos no SOC</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Leitura didática automática (contextual aos parâmetros ativos)
    notes = []
    warn = False
    if not dedup:
        notes.append("Você desligou a remoção de duplicatas — URLs repetidas podem cair no treino e no teste ('cola na prova').")
        warn = True
    if m["recall"] < 0.85:
        dicas = []
        if not balanced:
            dicas.append('ativar `class_weight="balanced"`')
        if analyzer == "word":
            dicas.append("mudar o analisador para `char (2–4)`")
        if threshold > 0.30:
            dicas.append("baixar o limiar")
        dica_txt = (" Tente " + ", ".join(dicas[:-1]) + (" ou " + dicas[-1] if len(dicas) > 1 else dicas[0]) + ".") if dicas else ""
        notes.append(
            f"Recall em **{m['recall']:.1%}**: **{m['fn']:,}** ataques passaram direto mesmo com acurácia de **{m['accuracy']:.1%}**.{dica_txt}"
        )
        warn = True
    elif m["precision"] < 0.70:
        notes.append(
            f"Recall alto (**{m['recall']:.1%}**), mas Precision baixa (**{m['precision']:.1%}**): **{m['fp']:,}** alarmes falsos para o SOC investigar. "
            "Subir o limiar reduz o ruído."
        )
        warn = True
    else:
        notes.append(
            f"Bom equilíbrio! O modelo pegou **{m['recall']:.1%}** dos ataques (**{m['tp']:,}** de **{m['total_attacks']:,}**) "
            f"com **{m['precision']:.1%}** de Precision. Mova o limiar acima para sentir o trade-off."
        )
    if warn:
        st.warning("**Leitura rápida:** " + " ".join(notes))
    else:
        st.success("**Leitura rápida:** " + " ".join(notes))

    # Matriz de Confusão + Curva de Trade-off
    col_cm, col_cv = st.columns(2)
    with col_cm:
        st.subheader("Matriz de Confusão (Teste)")
        st.pyplot(plot_confusion_matrix(m), use_container_width=True)
    with col_cv:
        st.subheader("Trade-off por Limiar (Recall × Precision)")
        st.pyplot(plot_tradeoff_curve(result["curve"], threshold), use_container_width=True)

    # Raio-X do modelo
    st.subheader("Olhando por dentro: o que o modelo aprendeu?")
    w1, w2 = st.columns(2)
    with w1:
        st.markdown("**Top sinais que puxam para MALICIOSO** *(pesos positivos)*")
        st.dataframe(result["top_mal"], hide_index=True, use_container_width=True)
    with w2:
        st.markdown("**Top sinais que puxam para LEGÍTIMO** *(pesos negativos)*")
        st.dataframe(result["top_leg"], hide_index=True, use_container_width=True)

    # Testador Ao Vivo com Explicabilidade
    st.subheader("Testador Ao Vivo + Por que o modelo decidiu assim?")
    user_url = st.text_input(
        "Digite qualquer URL (real ou inventada) para testar:",
        value="paypa1-secure-login.com/verify/account",
        help="Experimente adicionar '/myspace' ou '.edu' no final para ver o risco cair ao vivo!",
    )
    if user_url.strip():
        exp = explain_url(result["modelo"], result["vec"], bool(clean_url), user_url.strip(), threshold)
        if exp["verdict"] == "BLOQUEAR":
            st.error(f"**Risco estimado: {exp['prob']:.1%} — Decisão: BLOQUEAR** (limiar {threshold:.2f}) · URL limpa: `{exp['cleaned']}`")
        else:
            st.success(f"**Risco estimado: {exp['prob']:.1%} — Decisão: LIBERAR** (limiar {threshold:.2f}) · URL limpa: `{exp['cleaned']}`")
        st.caption(
            f"A URL gerou **{len(exp['tokens'])}** pedaços/palavras ({exp['matched']} conhecidos no vocabulário; intercepto = `{exp['intercept']}`)."
        )
        e1, e2 = st.columns(2)
        with e1:
            st.markdown("**Pedaços que EMPURRARAM para MALICIOSO**")
            if exp["push_mal"]:
                st.dataframe(pd.DataFrame(exp["push_mal"]), hide_index=True, use_container_width=True)
            else:
                st.caption("Nenhum sinal malicioso reconhecido.")
        with e2:
            st.markdown("**Pedaços que EMPURRARAM para LEGÍTIMO**")
            if exp["push_leg"]:
                st.dataframe(pd.DataFrame(exp["push_leg"]), hide_index=True, use_container_width=True)
            else:
                st.caption("Nenhum sinal legítimo reconhecido.")

    # Amostra rápida + Histórico
    with st.expander("Amostra rápida de URLs (atualiza com o limiar)", expanded=True):
        rows = []
        for url, real in PRESET_URLS:
            c = limpar_url(url) if clean_url else url
            p = float(result["modelo"].predict_proba(result["vec"].transform([c]))[0, 1])
            ver = "🚫 Bloquear" if p >= threshold else "✅ Liberar"
            ok = (p >= threshold) == (real == "maliciosa")
            rows.append({
                "URL": url,
                "Real": "Maliciosa" if real == "maliciosa" else "Legítima",
                "Risco": f"{p:.0%}",
                "Decisão": ver,
                "OK": "✅" if ok else "❌",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.subheader("Histórico de Experimentos (sua sessão)")
    st.dataframe(pd.DataFrame(st.session_state.history), hide_index=True, use_container_width=True)


if __name__ == "__main__":
    render_app()
