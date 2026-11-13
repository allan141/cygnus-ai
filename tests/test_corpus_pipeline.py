
import pytest

from src.corpus_pipeline import (
    normalizar_texto,
    hash_documento,
    carregar_corpus,
    dividir_documentos,
    estatisticas_documentos,
    preparar_corpus
)


def test_normalizacao_espacos():
    texto = "  Cygnus   aprende  "
    assert normalizar_texto(texto) == "Cygnus aprende"


def test_normalizacao_quebras():
    texto = "Linha 1\r\n\r\n\r\nLinha 2"
    assert normalizar_texto(texto) == "Linha 1\n\nLinha 2"


def test_preserva_acentos():
    texto = "Inteligência artificial é útil."
    assert normalizar_texto(texto) == texto


def test_hash_deterministico():
    assert hash_documento("Cygnus") == hash_documento("Cygnus")


def test_hash_textos_diferentes():
    assert hash_documento("Cygnus") != hash_documento("Gemini")


def test_carregar_corpus(tmp_path):
    (tmp_path / "a.txt").write_text(
        "Texto de treinamento.",
        encoding="utf-8"
    )

    documentos = carregar_corpus(tmp_path)

    assert len(documentos) == 1


def test_remove_duplicatas(tmp_path):
    (tmp_path / "a.txt").write_text(
        "Cygnus aprende.",
        encoding="utf-8"
    )

    (tmp_path / "b.txt").write_text(
        "  Cygnus  aprende.  ",
        encoding="utf-8"
    )

    documentos = carregar_corpus(tmp_path)

    assert len(documentos) == 1


def test_ignora_vazios(tmp_path):
    (tmp_path / "vazio.txt").write_text(
        "   ",
        encoding="utf-8"
    )

    assert carregar_corpus(tmp_path) == []


def test_divisao_sem_sobreposicao():
    documentos = [
        {"hash": str(i), "texto": f"Documento {i}"}
        for i in range(20)
    ]

    treino, validacao = dividir_documentos(documentos)

    hashes_treino = {doc["hash"] for doc in treino}
    hashes_validacao = {doc["hash"] for doc in validacao}

    assert hashes_treino.isdisjoint(hashes_validacao)
    assert len(treino) + len(validacao) == 20


def test_divisao_reproduzivel():
    documentos = [
        {"hash": str(i), "texto": str(i)}
        for i in range(20)
    ]

    a, b = dividir_documentos(documentos, semente=42)
    c, d = dividir_documentos(documentos, semente=42)

    assert a == c
    assert b == d


def test_divisao_poucos_documentos():
    documentos = [
        {"hash": "a", "texto": "A"},
        {"hash": "b", "texto": "B"}
    ]

    treino, validacao = dividir_documentos(documentos)

    assert len(treino) == 1
    assert len(validacao) == 1


def test_proporcao_invalida():
    with pytest.raises(ValueError):
        dividir_documentos(
            [{"hash": "a"}, {"hash": "b"}],
            proporcao_treino=1.0
        )


def test_estatisticas():
    documentos = [
        {"texto": "Cygnus aprende"},
        {"texto": "Modelo de linguagem"}
    ]

    resultado = estatisticas_documentos(documentos)

    assert resultado["documentos"] == 2
    assert resultado["palavras_aproximadas"] == 5


def test_pipeline_completo(tmp_path):
    origem = tmp_path / "raw"
    destino = tmp_path / "v3"

    origem.mkdir()

    for i in range(10):
        (origem / f"{i}.txt").write_text(
            f"Documento {i} sobre inteligência artificial.",
            encoding="utf-8"
        )

    resumo = preparar_corpus(
        origem=origem,
        destino=destino
    )

    assert resumo["total"]["documentos"] == 10
    assert resumo["treino"]["documentos"] == 9
    assert resumo["validacao"]["documentos"] == 1
    assert (destino / "resumo.json").exists()


def test_nao_sobrescreve_corpus(tmp_path):
    origem = tmp_path / "raw"
    destino = tmp_path / "v3"

    origem.mkdir()

    for i in range(3):
        (origem / f"{i}.txt").write_text(
            f"Texto diferente {i}",
            encoding="utf-8"
        )

    preparar_corpus(origem, destino)

    with pytest.raises(FileExistsError):
        preparar_corpus(origem, destino)
