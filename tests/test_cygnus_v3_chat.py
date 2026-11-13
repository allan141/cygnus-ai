
import torch
import pytest

from src.byte_bpe_tokenizer import CygnusByteBPETokenizer
from src.language_model import CygnusLanguageModel
from src.cygnus_v3_chat import (
    gerar_continuacao,
    avaliar_validacao,
)


@pytest.fixture
def modelo_teste():
    torch.manual_seed(42)

    modelo = CygnusLanguageModel(
        tamanho_vocabulario=256,
        dimensao_embedding=16,
        numero_cabecas=4,
        numero_camadas=1,
        tamanho_contexto=8,
    )

    modelo.eval()
    return modelo


@pytest.fixture
def tokenizador_teste():
    return CygnusByteBPETokenizer()


def test_geracao_retorna_texto(
    modelo_teste,
    tokenizador_teste,
):
    texto = gerar_continuacao(
        modelo_teste,
        tokenizador_teste,
        "Ola",
        max_novos_tokens=5,
    )

    assert isinstance(texto, str)
    assert texto.startswith("Ola")


def test_geracao_reproduzivel(
    modelo_teste,
    tokenizador_teste,
):
    argumentos = {
        "max_novos_tokens": 10,
        "semente": 123,
    }

    a = gerar_continuacao(
        modelo_teste,
        tokenizador_teste,
        "Teste",
        **argumentos,
    )

    b = gerar_continuacao(
        modelo_teste,
        tokenizador_teste,
        "Teste",
        **argumentos,
    )

    assert a == b


def test_prompt_vazio(
    modelo_teste,
    tokenizador_teste,
):
    with pytest.raises(ValueError):
        gerar_continuacao(
            modelo_teste,
            tokenizador_teste,
            "",
        )


def test_temperatura_invalida(
    modelo_teste,
    tokenizador_teste,
):
    with pytest.raises(ValueError):
        gerar_continuacao(
            modelo_teste,
            tokenizador_teste,
            "Teste",
            temperatura=0,
        )


def test_avaliacao_reproduzivel(
    tmp_path,
    modelo_teste,
    tokenizador_teste,
):
    (tmp_path / "teste.txt").write_text(
        "Cygnus aprende linguagem. " * 4,
        encoding="utf-8",
    )

    a = avaliar_validacao(
        modelo_teste,
        tokenizador_teste,
        tmp_path,
    )

    b = avaliar_validacao(
        modelo_teste,
        tokenizador_teste,
        tmp_path,
    )

    assert a == b
    assert a["tokens_avaliados"] > 0


def test_avaliacao_documento_curto(
    tmp_path,
    modelo_teste,
    tokenizador_teste,
):
    (tmp_path / "curto.txt").write_text(
        "A",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        avaliar_validacao(
            modelo_teste,
            tokenizador_teste,
            tmp_path,
        )
