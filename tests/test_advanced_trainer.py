
import csv

import pytest
import torch

from src.advanced_trainer import (
    criar_modelo,
    avaliar,
    perplexidade,
    salvar_checkpoint,
    carregar_checkpoint,
    registrar_metricas
)


def configuracao_teste():
    return {
        "tamanho_vocabulario": 256,
        "dimensao_embedding": 16,
        "numero_cabecas": 4,
        "numero_camadas": 1,
        "tamanho_contexto": 8
    }


def test_criar_modelo():
    modelo = criar_modelo(configuracao_teste())

    assert modelo.tamanho_vocabulario == 256
    assert modelo.tamanho_contexto == 8


def test_perplexidade():
    assert perplexidade(0.0) == pytest.approx(1.0)
    assert perplexidade(1.0) == pytest.approx(
        2.718281828,
        rel=1e-6
    )


def test_avaliacao():
    modelo = criar_modelo(configuracao_teste())

    sequencias = [
        torch.arange(100, dtype=torch.long)
    ]

    perda = avaliar(
        modelo,
        sequencias,
        contexto=8,
        batch_size=2,
        lotes=2
    )

    assert perda > 0
    assert torch.isfinite(torch.tensor(perda))


def test_avaliacao_preserva_modo():
    modelo = criar_modelo(configuracao_teste())
    modelo.eval()

    avaliar(
        modelo,
        [torch.arange(100, dtype=torch.long)],
        contexto=8,
        batch_size=2,
        lotes=1
    )

    assert not modelo.training


def test_checkpoint_salvar_carregar(tmp_path):
    configuracao = configuracao_teste()
    modelo = criar_modelo(configuracao)

    otimizador = torch.optim.AdamW(
        modelo.parameters(),
        lr=0.001
    )

    caminho = tmp_path / "modelo.pt"

    pesos_originais = {
        nome: tensor.clone()
        for nome, tensor in modelo.state_dict().items()
    }

    salvar_checkpoint(
        caminho,
        modelo,
        otimizador,
        passo=25,
        configuracao=configuracao
    )

    with torch.no_grad():
        for parametro in modelo.parameters():
            parametro.add_(1.0)

    passo = carregar_checkpoint(
        caminho,
        modelo,
        otimizador
    )

    assert passo == 25

    for nome, tensor in modelo.state_dict().items():
        assert torch.equal(
            tensor,
            pesos_originais[nome]
        )


def test_registrar_metricas(tmp_path):
    caminho = tmp_path / "metricas.csv"

    registrar_metricas(
        caminho,
        passo=10,
        treino=2.5,
        validacao=3.0
    )

    registrar_metricas(
        caminho,
        passo=20,
        treino=2.0,
        validacao=2.8
    )

    with caminho.open(
        newline="",
        encoding="utf-8"
    ) as arquivo:
        linhas = list(csv.reader(arquivo))

    assert len(linhas) == 3
    assert linhas[1][0] == "10"
    assert linhas[2][0] == "20"


def test_perplexidade_alta():
    valor = perplexidade(1000.0)

    assert torch.isfinite(torch.tensor(valor))


def test_checkpoint_inexistente(tmp_path):
    modelo = criar_modelo(configuracao_teste())

    otimizador = torch.optim.AdamW(
        modelo.parameters()
    )

    with pytest.raises(FileNotFoundError):
        carregar_checkpoint(
            tmp_path / "inexistente.pt",
            modelo,
            otimizador
        )
