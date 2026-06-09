# -*- coding: utf-8 -*-
"""
src/model.py
============
Etapa 2 do projeto "Análise Arquitetural do Congelamento de Camadas em
Transformers Multilíngues" — carregamento do XLM-RoBERTa e congelamento C1-C4.

Fase 2.1: carregar_modelo(seed) -> (model, tokenizer).
Fase 2.2: freeze_layers(model, config) para C1-C4 (classifier sempre treinável).
"""
from __future__ import annotations

import random
import re
from typing import Dict, Tuple

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    XLMRobertaForSequenceClassification,
    set_seed,
)

NOME_MODELO = "xlm-roberta-base"
NUM_LABELS = 2  # classificação binária: 0 = Negativo, 1 = Positivo

# Configurações de congelamento (Seção 6 da metodologia / PLAN-Etapa2).
# Cada config diz se congela os embeddings e QUAIS camadas do encoder (0..11).
# A classification head (classifier.*) é SEMPRE treinável (nasce do zero).
CONFIGS: Dict[str, Dict] = {
    "C1": {"nome": "Full Fine-Tuning", "embeddings": False, "camadas": set()},
    "C2": {"nome": "Freeze Lower",     "embeddings": True,  "camadas": set(range(0, 6))},
    "C3": {"nome": "Freeze Upper",     "embeddings": False, "camadas": set(range(6, 12))},
    "C4": {"nome": "Frozen Encoder",   "embeddings": True,  "camadas": set(range(0, 12))},
}

# Captura o índice i de "roberta.encoder.layer.<i>." (evita o bug de startswith
# em que "layer.1" casaria com 1, 10 e 11).
_PADRAO_CAMADA = re.compile(r"^roberta\.encoder\.layer\.(\d+)\.")


def fixar_seed(seed: int) -> None:
    """Fixa todas as fontes de aleatoriedade (idêntico à Etapa 1).

    Chamar ANTES de instanciar: a classification head do XLM-R nasce com pesos
    aleatórios, então a seed afeta a inicialização (crítico no C4, só-head).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def carregar_modelo(
    seed: int = 42,
    nome_modelo: str = NOME_MODELO,
    num_labels: int = NUM_LABELS,
) -> Tuple[XLMRobertaForSequenceClassification, "object"]:
    """Carrega o XLM-RoBERTa-base com head de classificação binária + tokenizer.

    Head padrão do HF (RobertaClassificationHead: dropout -> dense 768->768 ->
    tanh -> dropout -> linear 768->num_labels sobre o token <s>). Tokenizer via
    AutoTokenizer(use_fast=True) para garantir o backend rápido. A seed é fixada
    ANTES da instanciação para tornar a head reprodutível.
    """
    fixar_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(nome_modelo, use_fast=True)
    model = XLMRobertaForSequenceClassification.from_pretrained(
        nome_modelo, num_labels=num_labels
    )
    return model, tokenizer


def freeze_layers(model, config: str) -> Dict[str, int]:
    """Aplica o congelamento da config (C1-C4) via requires_grad em named_parameters.

    Idempotente: define requires_grad para TODOS os parâmetros, então pode ser
    reaplicada / trocada de config sem reload do modelo. A classifier.* fica
    SEMPRE treinável. Retorna contar_parametros(model) após o congelamento.
    """
    if config not in CONFIGS:
        raise ValueError("config inválida: %r. Use uma de %s." % (config, list(CONFIGS)))
    cfg = CONFIGS[config]
    for nome, p in model.named_parameters():
        if nome.startswith("classifier"):
            p.requires_grad = True
            continue
        congelar = False
        if nome.startswith("roberta.embeddings"):
            congelar = cfg["embeddings"]
        else:
            m = _PADRAO_CAMADA.match(nome)
            if m is not None:
                congelar = int(m.group(1)) in cfg["camadas"]
        p.requires_grad = not congelar
    return contar_parametros(model)


def contar_parametros(model) -> Dict[str, int]:
    """Retorna {'total', 'treinavel', 'congelado'} em número de parâmetros."""
    total = sum(p.numel() for p in model.parameters())
    treinavel = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "treinavel": treinavel, "congelado": total - treinavel}


if __name__ == "__main__":
    # Auto-teste minimalista (CPU): carrega e exercita o congelamento C1-C4.
    model, tokenizer = carregar_modelo(seed=42)
    assert tokenizer.is_fast, "tokenizer deveria ser fast"
    head = sum(p.numel() for n, p in model.named_parameters() if n.startswith("classifier"))
    c1 = freeze_layers(model, "C1")
    assert c1["treinavel"] == c1["total"], c1
    c4 = freeze_layers(model, "C4")
    assert c4["treinavel"] == head, (c4, head)
    for cfg in CONFIGS:
        freeze_layers(model, cfg)
        assert all(p.requires_grad for n, p in model.named_parameters()
                   if n.startswith("classifier"))
    print("model OK -> freeze C1-C4 | head =", head, "| total =", c1["total"])
