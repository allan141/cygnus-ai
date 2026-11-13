
import torch
import pytest

from src.byte_bpe_tokenizer import CygnusByteBPETokenizer
from src.language_model import CygnusLanguageModel
from src.train_language_model import (
    preparar_dados,
    criar_lote,
    avaliar,
    TAMANHO_CONTEXTO
)


def criar_dados_teste():
    texto = (
        "Cygnus aprende linguagem artificial. "
        "O Transformer aprende padrões. "
    ) * 30

    tokenizador = CygnusByteBPETokenizer()
    tokenizador.treinar([texto], numero_merges=30)

    return preparar_dados(texto, tokenizador)


def test_preparacao_dados():
    treino, validacao = criar_dados_teste()

    assert len(treino) > TAMANHO_CONTEXTO
    assert len(validacao) > TAMANHO_CONTEXTO


def test_lote_dimensoes():
    treino, _ = criar_dados_teste()

    x, y = criar_lote(treino, tamanho_lote=4)

    assert x.shape == (4, TAMANHO_CONTEXTO)
    assert y.shape == (4, TAMANHO_CONTEXTO)


def test_alvos_deslocados():
    treino, _ = criar_dados_teste()

    x, y = criar_lote(treino, tamanho_lote=4)

    assert torch.equal(x[:, 1:], y[:, :-1])


def test_lote_dados_insuficientes():
    dados = torch.arange(10)

    with pytest.raises(ValueError):
        criar_lote(dados)


def test_avaliacao():
    treino, validacao = criar_dados_teste()

    modelo = CygnusLanguageModel(
        tamanho_vocabulario=300,
        dimensao_embedding=32,
        numero_cabecas=4,
        numero_camadas=2,
        tamanho_contexto=TAMANHO_CONTEXTO
    )

    resultados = avaliar(
        modelo,
        treino,
        validacao,
        lotes=2
    )

    assert "treino" in resultados
    assert "validacao" in resultados
    assert resultados["treino"] > 0
    assert resultados["validacao"] > 0
