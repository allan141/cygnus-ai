
import math

import pytest
import torch

from src.language_model import CygnusLanguageModel
from src.byte_bpe_tokenizer import CygnusByteBPETokenizer

from src.evaluate_model import (
    avaliar_sequencias,
    carregar_documentos_avaliacao
)


def criar_modelo_teste():
    return CygnusLanguageModel(
        tamanho_vocabulario=256,
        dimensao_embedding=16,
        numero_cabecas=4,
        numero_camadas=1,
        tamanho_contexto=8
    )


def test_avaliacao_retorna_metricas():
    modelo = criar_modelo_teste()

    sequencias = [
        torch.arange(20, dtype=torch.long)
    ]

    resultado = avaliar_sequencias(
        modelo,
        sequencias
    )

    assert "perda" in resultado
    assert "perplexidade" in resultado
    assert "tokens_avaliados" in resultado


def test_quantidade_tokens_avaliados():
    modelo = criar_modelo_teste()

    sequencias = [
        torch.arange(20, dtype=torch.long),
        torch.arange(10, dtype=torch.long)
    ]

    resultado = avaliar_sequencias(
        modelo,
        sequencias
    )

    assert resultado["tokens_avaliados"] == 28


def test_avaliacao_deterministica():
    modelo = criar_modelo_teste()

    sequencias = [
        torch.arange(15, dtype=torch.long)
    ]

    resultado1 = avaliar_sequencias(
        modelo,
        sequencias
    )

    resultado2 = avaliar_sequencias(
        modelo,
        sequencias
    )

    assert resultado1["perda"] == pytest.approx(
        resultado2["perda"]
    )


def test_perplexidade_consistente():
    modelo = criar_modelo_teste()

    resultado = avaliar_sequencias(
        modelo,
        [torch.arange(12, dtype=torch.long)]
    )

    assert resultado["perplexidade"] == pytest.approx(
        math.exp(resultado["perda"])
    )


def test_preserva_modo_treinamento():
    modelo = criar_modelo_teste()
    modelo.train()

    avaliar_sequencias(
        modelo,
        [torch.arange(12, dtype=torch.long)]
    )

    assert modelo.training


def test_contexto_invalido():
    modelo = criar_modelo_teste()

    with pytest.raises(ValueError):
        avaliar_sequencias(
            modelo,
            [torch.arange(12, dtype=torch.long)],
            tamanho_contexto=0
        )


def test_contexto_maior_que_modelo():
    modelo = criar_modelo_teste()

    with pytest.raises(ValueError):
        avaliar_sequencias(
            modelo,
            [torch.arange(12, dtype=torch.long)],
            tamanho_contexto=100
        )


def test_documento_sem_alvos():
    modelo = criar_modelo_teste()

    with pytest.raises(ValueError):
        avaliar_sequencias(
            modelo,
            [torch.tensor([1])]
        )


def test_carregar_documentos_avaliacao(tmp_path):
    (tmp_path / "teste.txt").write_text(
        "Cygnus aprende linguagem.",
        encoding="utf-8"
    )

    tokenizador = CygnusByteBPETokenizer()

    tokenizador.treinar(
        ["Cygnus aprende linguagem."],
        numero_merges=10
    )

    sequencias = carregar_documentos_avaliacao(
        tmp_path,
        tokenizador
    )

    assert len(sequencias) == 1
    assert sequencias[0].dtype == torch.long


def test_diretorio_avaliacao_inexistente(tmp_path):
    tokenizador = CygnusByteBPETokenizer()

    with pytest.raises(FileNotFoundError):
        carregar_documentos_avaliacao(
            tmp_path / "inexistente",
            tokenizador
        )
