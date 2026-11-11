
from src.tokenizer import CygnusTokenizer


def test_codificacao_e_decodificacao():
    tokenizer = CygnusTokenizer()
    tokenizer.treinar("Cygnus AI")

    texto = "Cygnus AI"
    tokens = tokenizer.codificar(texto)

    assert tokenizer.decodificar(tokens) == texto


def test_caractere_desconhecido():
    tokenizer = CygnusTokenizer()
    tokenizer.treinar("abc")

    tokens = tokenizer.codificar("z")

    assert tokens == [0]
    assert tokenizer.decodificar(tokens) == "<UNK>"


def test_vocabulario():
    tokenizer = CygnusTokenizer()
    tokenizer.treinar("aab")

    # Dois caracteres distintos + token desconhecido
    assert len(tokenizer.char_to_id) == 3
