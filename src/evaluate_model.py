
# Cygnus AI - Fase 14
# Avaliacao deterministica de modelos de linguagem

import math
from pathlib import Path

import torch
import torch.nn.functional as F


@torch.no_grad()
def avaliar_sequencias(
    modelo,
    sequencias,
    tamanho_contexto=None
):
    """
    Avalia todos os alvos possiveis de cada documento.

    Cada token, exceto o primeiro de cada documento,
    e previsto a partir dos tokens anteriores.

    O resultado nao depende de amostragem aleatoria.
    """

    if tamanho_contexto is None:
        tamanho_contexto = modelo.tamanho_contexto

    if tamanho_contexto <= 0:
        raise ValueError("Contexto deve ser positivo")

    if tamanho_contexto > modelo.tamanho_contexto:
        raise ValueError(
            "Contexto maior que o suportado pelo modelo"
        )

    if not sequencias:
        raise ValueError("Nenhuma sequencia informada")

    modo_anterior = modelo.training
    modelo.eval()

    perda_total = 0.0
    total_tokens = 0

    try:
        for sequencia in sequencias:
            if len(sequencia) < 2:
                continue

            # Avalia uma previsao por posicao.
            # Mantem o contexto maximo disponivel.
            for posicao in range(1, len(sequencia)):
                inicio = max(
                    0,
                    posicao - tamanho_contexto
                )

                entrada = sequencia[
                    inicio:posicao
                ].unsqueeze(0)

                alvo = sequencia[
                    posicao
                ].view(1)

                logits, _ = modelo(entrada)

                logits_ultimo = logits[:, -1, :]

                perda = F.cross_entropy(
                    logits_ultimo,
                    alvo,
                    reduction="sum"
                )

                perda_total += perda.item()
                total_tokens += 1

    finally:
        modelo.train(modo_anterior)

    if total_tokens == 0:
        raise ValueError(
            "Nenhum token disponivel para avaliacao"
        )

    perda_media = perda_total / total_tokens

    return {
        "perda": perda_media,
        "perplexidade": math.exp(
            min(perda_media, 80.0)
        ),
        "tokens_avaliados": total_tokens
    }


def carregar_documentos_avaliacao(
    diretorio,
    tokenizador
):
    diretorio = Path(diretorio)

    if not diretorio.is_dir():
        raise FileNotFoundError(
            f"Diretorio nao encontrado: {diretorio}"
        )

    arquivos = sorted(
        diretorio.rglob("*.txt")
    )

    sequencias = []

    for arquivo in arquivos:
        texto = arquivo.read_text(
            encoding="utf-8"
        )

        if texto.strip():
            ids = tokenizador.codificar(texto)

            sequencias.append(
                torch.tensor(
                    ids,
                    dtype=torch.long
                )
            )

    if not sequencias:
        raise ValueError(
            "Nenhum documento valido encontrado"
        )

    return sequencias
