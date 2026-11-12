
import torch
import pytest

from src.attention import CygnusCausalAttention


def test_dimensoes_saida():
    modelo = CygnusCausalAttention(32)

    entrada = torch.randn(2, 5, 32)
    saida = modelo(entrada)

    assert saida.shape == (2, 5, 32)


def test_mascara_causal():
    modelo = CygnusCausalAttention(32)

    entrada = torch.randn(1, 4, 32)
    _, pesos = modelo(entrada, retornar_pesos=True)

    # A parte superior da matriz deve ser zero
    futuros = torch.triu(pesos[0], diagonal=1)

    assert torch.count_nonzero(futuros).item() == 0


def test_pesos_somam_um():
    modelo = CygnusCausalAttention(32)

    entrada = torch.randn(1, 4, 32)
    _, pesos = modelo(entrada, retornar_pesos=True)

    soma = pesos.sum(dim=-1)

    assert torch.allclose(soma, torch.ones_like(soma))


def test_tokens_futuros_nao_afetam_passado():
    torch.manual_seed(42)

    modelo = CygnusCausalAttention(32)
    modelo.eval()

    entrada = torch.randn(1, 4, 32)
    entrada_modificada = entrada.clone()

    # Alterar somente o ultimo token
    entrada_modificada[:, 3, :] += 100

    with torch.no_grad():
        saida_original = modelo(entrada)
        saida_modificada = modelo(entrada_modificada)

    # Os tres primeiros tokens nao podem mudar
    assert torch.allclose(
        saida_original[:, :3, :],
        saida_modificada[:, :3, :],
        atol=1e-6
    )


def test_gradientes():
    modelo = CygnusCausalAttention(32)

    entrada = torch.randn(1, 4, 32)
    saida = modelo(entrada)

    perda = saida.pow(2).mean()
    perda.backward()

    for parametro in modelo.parameters():
        assert parametro.grad is not None
        assert torch.isfinite(parametro.grad).all()


def test_dimensao_incorreta():
    modelo = CygnusCausalAttention(32)

    entrada = torch.randn(1, 4, 16)

    with pytest.raises(ValueError):
        modelo(entrada)
