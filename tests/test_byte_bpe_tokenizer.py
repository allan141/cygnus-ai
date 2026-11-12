
import pytest

from src.byte_bpe_tokenizer import CygnusByteBPETokenizer


def criar_tokenizador():
    tokenizador = CygnusByteBPETokenizer()

    tokenizador.treinar(
        [
            "Cygnus aprende linguagem. " * 5,
            "Inteligencia artificial e tecnologia. " * 5
        ],
        numero_merges=50
    )

    return tokenizador


def test_texto_com_acentos():
    tokenizador = criar_tokenizador()

    texto = "Olá! Inteligência artificial."

    ids = tokenizador.codificar(texto)

    assert tokenizador.decodificar(ids) == texto


def test_emojis():
    tokenizador = criar_tokenizador()

    texto = "Cygnus 🌟🚀🤖"

    ids = tokenizador.codificar(texto)

    assert tokenizador.decodificar(ids) == texto


def test_caracteres_desconhecidos():
    tokenizador = criar_tokenizador()

    texto = "こんにちは世界 Привет 🌍"

    ids = tokenizador.codificar(texto)

    assert tokenizador.decodificar(ids) == texto


def test_vocabulario_inicial():
    tokenizador = CygnusByteBPETokenizer()

    assert tokenizador.tamanho_vocabulario == 256


def test_merges_aprendidos():
    tokenizador = criar_tokenizador()

    assert len(tokenizador.merges) > 0
    assert tokenizador.tamanho_vocabulario > 256


def test_reducao_de_tokens():
    tokenizador = criar_tokenizador()

    texto = "Cygnus aprende linguagem."

    quantidade_bytes = len(texto.encode("utf-8"))
    quantidade_tokens = len(tokenizador.codificar(texto))

    assert quantidade_tokens < quantidade_bytes


def test_salvar_e_carregar(tmp_path):
    tokenizador = criar_tokenizador()

    caminho = tmp_path / "tokenizador.json"

    tokenizador.salvar(caminho)

    carregado = CygnusByteBPETokenizer.carregar(caminho)

    texto = "Olá Cygnus 🌟"

    assert carregado.codificar(texto) == tokenizador.codificar(texto)
    assert carregado.decodificar(carregado.codificar(texto)) == texto


def test_texto_vazio():
    tokenizador = criar_tokenizador()

    assert tokenizador.codificar("") == []
    assert tokenizador.decodificar([]) == ""


def test_numero_merges_invalido():
    tokenizador = CygnusByteBPETokenizer()

    with pytest.raises(ValueError):
        tokenizador.treinar(["texto"], numero_merges=-1)


def test_ids_dentro_vocabulario():
    tokenizador = criar_tokenizador()

    ids = tokenizador.codificar("Cygnus 🌟")

    assert all(
        0 <= token < tokenizador.tamanho_vocabulario
        for token in ids
    )
