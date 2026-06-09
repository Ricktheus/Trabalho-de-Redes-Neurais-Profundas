# -*- coding: utf-8 -*-
"""
tests/test_model.py
===================
Testes de verificação da Etapa 2 (Fase 2.3) — congelamento seletivo C1-C4.
Confere requires_grad por prefixo, head sempre treinável e a contagem real de
parâmetros treináveis por config. Roda standalone (python tests/test_model.py)
ou importado no notebook chamando rodar_todos(model) — sem recarregar o modelo.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import model as M

_PAD = re.compile(r"^roberta\.encoder\.layer\.(\d+)\.")

# Especificação esperada por config: (congela_embeddings, camadas congeladas).
_ESPEC = {
    "C1": (False, set()),
    "C2": (True, set(range(0, 6))),
    "C3": (False, set(range(6, 12))),
    "C4": (True, set(range(0, 12))),
}

# Contagens reais confirmadas no run da Fase 2.2 (regressão).
_ESPERADO_TREINAVEL = {
    "C1": 278_045_186,
    "C2": 43_119_362,
    "C3": 235_517_954,
    "C4": 592_130,
}


def _head(model):
    return sum(p.numel() for n, p in model.named_parameters() if n.startswith("classifier"))


def teste_c1_treina_tudo(model):
    c = M.freeze_layers(model, "C1")
    assert c["treinavel"] == c["total"], c


def teste_c4_so_head(model):
    c = M.freeze_layers(model, "C4")
    assert c["treinavel"] == _head(model), (c, _head(model))


def teste_head_sempre_treinavel(model):
    for cfg in M.CONFIGS:
        M.freeze_layers(model, cfg)
        congeladas = [n for n, p in model.named_parameters()
                      if n.startswith("classifier") and not p.requires_grad]
        assert not congeladas, (cfg, congeladas)


def teste_prefixos_por_config(model):
    for cfg, (emb_cong, cam_cong) in _ESPEC.items():
        M.freeze_layers(model, cfg)
        for n, p in model.named_parameters():
            if n.startswith("classifier"):
                assert p.requires_grad, (cfg, n)
            elif n.startswith("roberta.embeddings"):
                assert p.requires_grad == (not emb_cong), (cfg, n)
            else:
                m = _PAD.match(n)
                if m is not None:
                    i = int(m.group(1))
                    assert p.requires_grad == (i not in cam_cong), (cfg, n)


def teste_contagem_real(model):
    for cfg, val in _ESPERADO_TREINAVEL.items():
        c = M.freeze_layers(model, cfg)
        assert c["treinavel"] == val, (cfg, c["treinavel"], val)


def teste_idempotente(model):
    a = M.freeze_layers(model, "C2")["treinavel"]
    b = M.freeze_layers(model, "C2")["treinavel"]
    M.freeze_layers(model, "C3")
    c = M.freeze_layers(model, "C2")["treinavel"]
    assert a == b == c, (a, b, c)


def teste_config_invalida(model):
    try:
        M.freeze_layers(model, "C9")
    except ValueError:
        return
    raise AssertionError("freeze_layers deveria levantar ValueError para config inválida")


TESTES = [
    teste_c1_treina_tudo,
    teste_c4_so_head,
    teste_head_sempre_treinavel,
    teste_prefixos_por_config,
    teste_contagem_real,
    teste_idempotente,
    teste_config_invalida,
]


def rodar_todos(model):
    falhas = []
    for t in TESTES:
        try:
            t(model)
            print("  PASS", t.__name__)
        except AssertionError as e:
            falhas.append(t.__name__)
            print("  FAIL", t.__name__, "->", e)
    M.freeze_layers(model, "C1")  # restaura estado limpo
    if falhas:
        raise AssertionError("%d teste(s) falharam: %s" % (len(falhas), falhas))
    print("OK %d testes passaram." % len(TESTES))
    return True


if __name__ == "__main__":
    modelo, _ = M.carregar_modelo(seed=42)
    rodar_todos(modelo)
