
import pytest
import torch

from src.language_model import CygnusLanguageModel


def criar_modelo():
    return CygnusLanguageModel(
        tamanho_vocabulario=100,
        dimensao_embedding=32,
        numero_cabecas=4,
        numero_camadas=2,
        tamanho_contexto=64
    )


def test_dimensoes_logits():
    modelo = criar_modelo()
    tokens = torch.randint(0, 100, (2, 5))

    logits, perda = modelo(tokens)

    assert logits.shape == (2, 5, 100)
    assert perda is None


def test_perda_finita():
    modelo = criar_modelo()

    tokens = torch.randint(0, 100, (2, 5))
    alvos = torch.randint(0, 100, (2, 5))

    logits, perda = modelo(tokens, alvos)

    assert logits.shape == (2, 5, 100)
    assert torch.isfinite(perda)
    assert perda.ndim == 0


def test_gradientes():
    modelo = criar_modelo()

    tokens = torch.randint(0, 100, (2, 5))
    alvos = torch.randint(0, 100, (2, 5))

    _, perda = modelo(tokens, alvos)
    perda.backward()

    for parametro in modelo.parameters():
        assert parametro.grad is not None
        assert torch.isfinite(parametro.grad).all()


def test_causalidade():
    torch.manual_seed(42)

    modelo = criar_modelo()
    modelo.eval()

    tokens = torch.tensor([[1, 2, 3, 4, 5]])
    modificados = torch.tensor([[1, 2, 3, 4, 99]])

    with torch.no_grad():
        logits_originais, _ = modelo(tokens)
        logits_modificados, _ = modelo(modificados)

    assert torch.allclose(
        logits_originais[:, :4, :],
        logits_modificados[:, :4, :],
        atol=1e-6
    )


def test_contexto_maximo():
    modelo = criar_modelo()
    tokens = torch.randint(0, 100, (1, 64))

    logits, _ = modelo(tokens)

    assert logits.shape == (1, 64, 100)


def test_contexto_excedido():
    modelo = criar_modelo()
    tokens = torch.randint(0, 100, (1, 65))

    with pytest.raises(ValueError):
        modelo(tokens)


def test_alvos_incorretos():
    modelo = criar_modelo()

    tokens = torch.randint(0, 100, (1, 5))
    alvos = torch.randint(0, 100, (1, 4))

    with pytest.raises(ValueError):
        modelo(tokens, alvos)


def test_tokens_invalidos():
    modelo = criar_modelo()

    tokens = torch.tensor([[1, 2, 100]])

    with pytest.raises(ValueError):
        modelo(tokens)


def test_probabilidades_somam_um():
    modelo = criar_modelo()

    tokens = torch.randint(0, 100, (1, 5))
    logits, _ = modelo(tokens)

    probabilidades = torch.softmax(logits, dim=-1)

    assert torch.allclose(
        probabilidades.sum(dim=-1),
        torch.ones(1, 5),
        atol=1e-6
    )


def test_quantidade_parametros():
    modelo = criar_modelo()

    total = sum(
        parametro.numel()
        for parametro in modelo.parameters()
    )

    assert total == 33920


def test_uma_etapa_treinamento():
    torch.manual_seed(42)

    modelo = criar_modelo()
    otimizador = torch.optim.AdamW(
        modelo.parameters(),
        lr=0.001
    )

    tokens = torch.randint(0, 100, (2, 5))
    alvos = torch.randint(0, 100, (2, 5))

    _, perda = modelo(tokens, alvos)

    otimizador.zero_grad()
    perda.backward()
    otimizador.step()

    _, nova_perda = modelo(tokens, alvos)

    assert torch.isfinite(nova_perda)


def test_vocabulario_pequeno():
    modelo = CygnusLanguageModel(
        tamanho_vocabulario=20,
        dimensao_embedding=16,
        numero_cabecas=4,
        numero_camadas=1,
        tamanho_contexto=8
    )

    tokens = torch.tensor([[1, 2, 3]])
    logits, _ = modelo(tokens)

    assert logits.shape == (1, 3, 20)
