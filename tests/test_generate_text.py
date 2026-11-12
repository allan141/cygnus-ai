
import pytest
import torch

from src.byte_bpe_tokenizer import CygnusByteBPETokenizer
from src.language_model import CygnusLanguageModel
from src.generate_text import (
    gerar_tokens,
    gerar_texto,
    carregar_cygnus
)


def criar_modelo_teste():
    torch.manual_seed(42)

    tokenizador = CygnusByteBPETokenizer()

    modelo = CygnusLanguageModel(
        tamanho_vocabulario=256,
        dimensao_embedding=16,
        numero_cabecas=4,
        numero_camadas=1,
        tamanho_contexto=16
    )

    return modelo, tokenizador


def test_quantidade_tokens_gerados():
    modelo, _ = criar_modelo_teste()

    entrada = [67, 121, 103]

    resultado = gerar_tokens(
        modelo,
        entrada,
        max_novos_tokens=5
    )

    assert len(resultado) == len(entrada) + 5


def test_preserva_prompt():
    modelo, _ = criar_modelo_teste()

    entrada = [65, 66, 67]

    resultado = gerar_tokens(
        modelo,
        entrada,
        max_novos_tokens=4
    )

    assert resultado[:len(entrada)] == entrada


def test_tokens_dentro_vocabulario():
    modelo, _ = criar_modelo_teste()

    resultado = gerar_tokens(
        modelo,
        [65],
        max_novos_tokens=10
    )

    assert all(
        0 <= token < modelo.tamanho_vocabulario
        for token in resultado
    )


def test_geracao_zero_tokens():
    modelo, _ = criar_modelo_teste()

    entrada = [65, 66]

    resultado = gerar_tokens(
        modelo,
        entrada,
        max_novos_tokens=0
    )

    assert resultado == entrada


def test_prompt_vazio():
    modelo, _ = criar_modelo_teste()

    with pytest.raises(ValueError):
        gerar_tokens(modelo, [])


def test_temperatura_invalida():
    modelo, _ = criar_modelo_teste()

    with pytest.raises(ValueError):
        gerar_tokens(
            modelo,
            [65],
            temperatura=0
        )


def test_top_k_invalido():
    modelo, _ = criar_modelo_teste()

    with pytest.raises(ValueError):
        gerar_tokens(
            modelo,
            [65],
            top_k=0
        )


def test_texto_gerado():
    modelo, tokenizador = criar_modelo_teste()

    resultado = gerar_texto(
        modelo,
        tokenizador,
        prompt="Cygnus",
        max_novos_tokens=5
    )

    assert isinstance(resultado, str)
    assert resultado.startswith("Cygnus")


def test_contexto_limitado():
    modelo, _ = criar_modelo_teste()

    entrada = [65] * 30

    resultado = gerar_tokens(
        modelo,
        entrada,
        max_novos_tokens=3
    )

    assert len(resultado) == 33


def test_checkpoint_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError):
        carregar_cygnus(
            caminho_modelo=tmp_path / "inexistente.pt",
            caminho_tokenizador=tmp_path / "inexistente.json"
        )


def test_carregamento_checkpoint(tmp_path):
    modelo, tokenizador = criar_modelo_teste()

    caminho_modelo = tmp_path / "modelo.pt"
    caminho_tokenizador = tmp_path / "tokenizador.json"

    configuracao = {
        "tamanho_vocabulario": 256,
        "dimensao_embedding": 16,
        "numero_cabecas": 4,
        "numero_camadas": 1,
        "tamanho_contexto": 16
    }

    torch.save(
        {
            "model_state_dict": modelo.state_dict(),
            "configuracao": configuracao
        },
        caminho_modelo
    )

    tokenizador.salvar(caminho_tokenizador)

    carregado, carregado_tokenizador = carregar_cygnus(
        caminho_modelo,
        caminho_tokenizador
    )

    assert not carregado.training
    assert carregado_tokenizador.tamanho_vocabulario == 256

    for nome, parametro in modelo.state_dict().items():
        assert torch.equal(
            parametro,
            carregado.state_dict()[nome]
        )
