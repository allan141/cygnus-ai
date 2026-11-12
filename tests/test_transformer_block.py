
import pytest
import torch

from src.transformer_block import (
    CygnusFeedForward,
    CygnusTransformerBlock
)


def test_dimensoes_saida():
    modelo = CygnusTransformerBlock(32, 4)

    entrada = torch.randn(2, 5, 32)
    saida = modelo(entrada)

    assert saida.shape == (2, 5, 32)


def test_saida_finita():
    modelo = CygnusTransformerBlock(32, 4)

    entrada = torch.randn(2, 5, 32)
    saida = modelo(entrada)

    assert torch.isfinite(saida).all()


def test_feedforward_dimensoes():
    rede = CygnusFeedForward(32, 4)

    entrada = torch.randn(2, 5, 32)
    saida = rede(entrada)

    assert saida.shape == entrada.shape


def test_causalidade():
    torch.manual_seed(42)

    modelo = CygnusTransformerBlock(32, 4)
    modelo.eval()

    entrada = torch.randn(1, 5, 32)
    modificada = entrada.clone()

    # Alterar apenas o ultimo token
    modificada[:, 4, :] += 100

    with torch.no_grad():
        original = modelo(entrada)
        alterada = modelo(modificada)

    # Os quatro primeiros tokens nao podem mudar
    assert torch.allclose(
        original[:, :4, :],
        alterada[:, :4, :],
        atol=1e-6
    )


def test_gradientes():
    modelo = CygnusTransformerBlock(32, 4)

    entrada = torch.randn(2, 5, 32)
    saida = modelo(entrada)

    perda = saida.pow(2).mean()
    perda.backward()

    for parametro in modelo.parameters():
        assert parametro.grad is not None
        assert torch.isfinite(parametro.grad).all()


def test_dimensao_incorreta():
    modelo = CygnusTransformerBlock(32, 4)

    entrada = torch.randn(1, 5, 16)

    with pytest.raises(ValueError):
        modelo(entrada)


def test_entrada_sem_batch():
    modelo = CygnusTransformerBlock(32, 4)

    entrada = torch.randn(5, 32)

    with pytest.raises(ValueError):
        modelo(entrada)


def test_numero_cabecas_invalido():
    with pytest.raises(ValueError):
        CygnusTransformerBlock(32, 3)


def test_quantidade_parametros():
    modelo = CygnusTransformerBlock(32, 4)

    total = sum(
        parametro.numel()
        for parametro in modelo.parameters()
    )

    assert total == 12704


def test_conexoes_residuais():
    torch.manual_seed(42)

    modelo = CygnusTransformerBlock(32, 4)

    # Zerar as saidas das duas subcamadas
    with torch.no_grad():
        modelo.atencao.projecao_saida.weight.zero_()
        modelo.atencao.projecao_saida.bias.zero_()

        ultima_camada = modelo.feedforward.rede[-1]
        ultima_camada.weight.zero_()
        ultima_camada.bias.zero_()

    entrada = torch.randn(1, 5, 32)
    saida = modelo(entrada)

    # Sem contribuicao das subcamadas,
    # as conexoes residuais preservam a entrada
    assert torch.allclose(saida, entrada, atol=1e-6)
