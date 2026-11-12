
from src.bpe_tokenizer import CygnusBPETokenizer


def test_codificacao_e_decodificacao():
    tokenizer = CygnusBPETokenizer()
    tokenizer.treinar("casa casa casa casaco", numero_merges=10)

    texto = "casa casaco"
    tokens = tokenizer.codificar(texto)

    assert tokenizer.decodificar(tokens) == texto


def test_aprendizado_de_merges():
    tokenizer = CygnusBPETokenizer()
    tokenizer.treinar("abababab", numero_merges=10)

    assert len(tokenizer.merges) > 0
    assert ("a", "b") in tokenizer.merges


def test_reducao_de_tokens():
    tokenizer = CygnusBPETokenizer()
    tokenizer.treinar("casa casa casa casa", numero_merges=10)

    texto = "casa"
    tokens = tokenizer.codificar(texto)

    assert len(tokens) < len(texto)


def test_vocabulario():
    tokenizer = CygnusBPETokenizer()
    tokenizer.treinar("abab", numero_merges=5)

    assert "a" in tokenizer.vocabulario
    assert "b" in tokenizer.vocabulario
    assert len(tokenizer.vocabulario) >= 2


def test_texto_desconhecido():
    tokenizer = CygnusBPETokenizer()
    tokenizer.treinar("abcabc", numero_merges=5)

    tokens = tokenizer.codificar("z")

    assert tokens == [0]
    assert tokenizer.decodificar(tokens) == "<UNK>"


def test_salvar_e_carregar(tmp_path):
    tokenizer = CygnusBPETokenizer()
    tokenizer.treinar(
        "Cygnus Cygnus inteligencia inteligencia",
        numero_merges=10
    )

    caminho = tmp_path / "tokenizer.json"
    tokenizer.salvar(caminho)

    novo_tokenizer = CygnusBPETokenizer()
    novo_tokenizer.carregar(caminho)

    frase = "Cygnus inteligencia"

    assert novo_tokenizer.vocabulario == tokenizer.vocabulario
    assert novo_tokenizer.merges == tokenizer.merges

    assert novo_tokenizer.codificar(frase) == tokenizer.codificar(frase)
    assert novo_tokenizer.decodificar(
        novo_tokenizer.codificar(frase)
    ) == frase
