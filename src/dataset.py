
# Cygnus AI - Fase 13
# Preparacao de dados para treinamento

from pathlib import Path

import torch

from src.byte_bpe_tokenizer import CygnusByteBPETokenizer


def carregar_documentos(diretorio):
    diretorio = Path(diretorio)

    if not diretorio.is_dir():
        raise FileNotFoundError(
            f"Diretorio nao encontrado: {diretorio}"
        )

    arquivos = sorted(diretorio.rglob("*.txt"))

    if not arquivos:
        raise ValueError(
            f"Nenhum arquivo TXT encontrado: {diretorio}"
        )

    documentos = []

    for arquivo in arquivos:
        texto = arquivo.read_text(encoding="utf-8")

        if texto.strip():
            documentos.append(texto)

    if not documentos:
        raise ValueError(
            "Todos os documentos estao vazios"
        )

    return documentos


def preparar_tokenizador(
    documentos_treino,
    numero_merges=100
):
    if not documentos_treino:
        raise ValueError(
            "Nenhum documento de treinamento"
        )

    tokenizador = CygnusByteBPETokenizer()

    # O vocabulario aprende somente com dados de treino.
    tokenizador.treinar(
        documentos_treino,
        numero_merges=numero_merges
    )

    return tokenizador


def tokenizar_documentos(
    documentos,
    tokenizador,
    tamanho_contexto=32
):
    if tamanho_contexto <= 0:
        raise ValueError(
            "O contexto deve ser positivo"
        )

    sequencias = []

    for texto in documentos:
        ids = tokenizador.codificar(texto)

        if len(ids) > tamanho_contexto:
            sequencias.append(
                torch.tensor(ids, dtype=torch.long)
            )

    if not sequencias:
        raise ValueError(
            "Nenhum documento possui tokens suficientes"
        )

    return sequencias


def criar_lote_documentos(
    sequencias,
    tamanho_contexto=32,
    tamanho_lote=8
):
    if tamanho_contexto <= 0:
        raise ValueError(
            "O contexto deve ser positivo"
        )

    if tamanho_lote <= 0:
        raise ValueError(
            "O tamanho do lote deve ser positivo"
        )

    elegiveis = [
        sequencia
        for sequencia in sequencias
        if len(sequencia) > tamanho_contexto
    ]

    if not elegiveis:
        raise ValueError(
            "Nenhuma sequencia valida para criar lotes"
        )

    # Amostragem proporcional ao numero de janelas
    # disponiveis em cada documento.
    pesos = torch.tensor(
        [
            len(sequencia) - tamanho_contexto
            for sequencia in elegiveis
        ],
        dtype=torch.float
    )

    indices = torch.multinomial(
        pesos,
        num_samples=tamanho_lote,
        replacement=True
    )

    entradas = []
    alvos = []

    for indice in indices.tolist():
        sequencia = elegiveis[indice]

        limite = len(sequencia) - tamanho_contexto

        inicio = torch.randint(
            0,
            limite,
            (1,)
        ).item()

        fim = inicio + tamanho_contexto

        entradas.append(
            sequencia[inicio:fim]
        )

        alvos.append(
            sequencia[inicio + 1:fim + 1]
        )

    return (
        torch.stack(entradas),
        torch.stack(alvos)
    )


def preparar_dataset(
    diretorio_treino="data/train",
    diretorio_validacao="data/validation",
    numero_merges=100,
    tamanho_contexto=32
):
    documentos_treino = carregar_documentos(
        diretorio_treino
    )

    documentos_validacao = carregar_documentos(
        diretorio_validacao
    )

    tokenizador = preparar_tokenizador(
        documentos_treino,
        numero_merges=numero_merges
    )

    treino = tokenizar_documentos(
        documentos_treino,
        tokenizador,
        tamanho_contexto
    )

    validacao = tokenizar_documentos(
        documentos_validacao,
        tokenizador,
        tamanho_contexto
    )

    return treino, validacao, tokenizador
