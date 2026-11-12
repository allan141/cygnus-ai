
import torch

from src.embeddings import CygnusEmbeddings


def test_dimensoes_embeddings():
    modelo = CygnusEmbeddings(
        tamanho_vocabulario=50,
        dimensao_embedding=32
    )

    tokens = torch.tensor([1, 2, 3, 4])
    resultado = modelo(tokens)

    assert resultado.shape == (4, 32)


def test_mesmo_token_mesmo_embedding():
    modelo = CygnusEmbeddings(
        tamanho_vocabulario=20,
        dimensao_embedding=16
    )

    tokens = torch.tensor([3, 3])
    resultado = modelo(tokens)

    assert torch.equal(resultado[0], resultado[1])


def test_embeddings_treinaveis():
    modelo = CygnusEmbeddings(
        tamanho_vocabulario=10,
        dimensao_embedding=8
    )

    tokens = torch.tensor([1, 2, 3])
    resultado = modelo(tokens)

    perda = resultado.pow(2).mean()
    perda.backward()

    assert modelo.embeddings.weight.grad is not None
    assert modelo.embeddings.weight.grad.abs().sum().item() > 0


def test_quantidade_parametros():
    modelo = CygnusEmbeddings(
        tamanho_vocabulario=50,
        dimensao_embedding=32
    )

    parametros = sum(
        p.numel() for p in modelo.parameters()
    )

    assert parametros == 50 * 32
