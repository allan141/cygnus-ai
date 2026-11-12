
import pytest
import torch

from src.multihead_attention import CygnusMultiHeadAttention


def test_dimensoes_saida():
    modelo = CygnusMultiHeadAttention(32, 4)

    entrada = torch.randn(2, 5, 32)
    saida = modelo(entrada)

    assert saida.shape == (2, 5, 32)


def test_dimensoes_pesos():
    modelo = CygnusMultiHeadAttention(32, 4)

    entrada = torch.randn(2, 5, 32)
    _, pesos = modelo(entrada, retornar_pesos=True)

    assert pesos.shape == (2, 4, 5, 5)


def test_mascara_causal():
    modelo = CygnusMultiHeadAttention(32, 4)

    entrada = torch.randn(1, 5, 32)
    _, pesos = modelo(entrada, retornar_pesos=True)

    # Verificar todas as cabecas
    futuros = torch.triu(pesos, diagonal=1)

    assert torch.count_nonzero(futuros).item() == 0


def test_pesos_somam_um():
    modelo = CygnusMultiHeadAttention(32, 4)

    entrada = torch.randn(1, 5, 32)
    _, pesos = modelo(entrada, retornar_pesos=True)

    somas = pesos.sum(dim=-1)

    assert torch.allclose(
        somas,
        torch.ones_like(somas),
        atol=1e-6
    )


def test_futuro_nao_afeta_passado():
    torch.manual_seed(42)

    modelo = CygnusMultiHeadAttention(32, 4)
    modelo.eval()

    entrada = torch.randn(1, 5, 32)
    modificada = entrada.clone()

    modificada[:, 4, :] += 100

    with torch.no_grad():
        original = modelo(entrada)
        alterada = modelo(modificada)

    assert torch.allclose(
        original[:, :4, :],
        alterada[:, :4, :],
        atol=1e-6
    )


def test_gradientes():
    modelo = CygnusMultiHeadAttention(32, 4)

    entrada = torch.randn(2, 5, 32)
    saida = modelo(entrada)

    perda = saida.pow(2).mean()
    perda.backward()

    for parametro in modelo.parameters():
        assert parametro.grad is not None
        assert torch.isfinite(parametro.grad).all()


def test_dimensao_invalida():
    with pytest.raises(ValueError):
        CygnusMultiHeadAttention(32, 3)


def test_numero_cabecas_invalido():
    with pytest.raises(ValueError):
        CygnusMultiHeadAttention(32, 0)


def test_quantidade_parametros():
    modelo = CygnusMultiHeadAttention(32, 4)

    total = sum(
        parametro.numel()
        for parametro in modelo.parameters()
    )

    assert total == 4224
