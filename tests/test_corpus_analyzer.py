
# Testes - Cygnus AI, Fase 15 Parte 5

import json

import pytest

from src.byte_bpe_tokenizer import CygnusByteBPETokenizer
from src.corpus_analyzer import (
    carregar_documentos,
    analisar_documentos,
    executar_analise,
)


def test_carregar_documentos(tmp_path):
    (tmp_path / "a.txt").write_text(
        "Ola Cygnus", encoding="utf-8"
    )

    documentos = carregar_documentos(tmp_path)

    assert len(documentos) == 1
    assert documentos[0]["texto"] == "Ola Cygnus"


def test_diretorio_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError):
        carregar_documentos(tmp_path / "inexistente")


def test_diretorio_sem_txt(tmp_path):
    with pytest.raises(ValueError):
        carregar_documentos(tmp_path)


def test_documentos_vazios(tmp_path):
    (tmp_path / "vazio.txt").write_text(
        "   ", encoding="utf-8"
    )

    with pytest.raises(ValueError):
        carregar_documentos(tmp_path)


def test_contagem_tokens():
    tokenizador = CygnusByteBPETokenizer()

    documentos = [{
        "arquivo": "teste.txt",
        "texto": "Ola"
    }]

    resultado = analisar_documentos(
        documentos, tokenizador, contexto=2
    )

    assert resultado["tokens"] == 3
    assert resultado["janelas_completas"] == 1


def test_utf8():
    tokenizador = CygnusByteBPETokenizer()

    resultado = analisar_documentos(
        [{"arquivo": "teste.txt", "texto": "á"}],
        tokenizador,
        contexto=2
    )

    assert resultado["caracteres"] == 1
    assert resultado["bytes_utf8"] == 2
    assert resultado["tokens"] == 2


def test_contexto_invalido():
    tokenizador = CygnusByteBPETokenizer()

    with pytest.raises(ValueError):
        analisar_documentos(
            [], tokenizador, contexto=0
        )


def test_pipeline_completo(tmp_path):
    origem = tmp_path / "corpus"
    treino = origem / "train"
    validacao = origem / "validation"

    treino.mkdir(parents=True)
    validacao.mkdir(parents=True)

    (treino / "a.txt").write_text(
        "Cygnus aprende linguagem. " * 20,
        encoding="utf-8"
    )

    (validacao / "b.txt").write_text(
        "O modelo sera avaliado.",
        encoding="utf-8"
    )

    destino = tmp_path / "saida"

    relatorio = executar_analise(
        origem=origem,
        destino=destino,
        merges=10,
        contexto=8
    )

    assert relatorio["tamanho_vocabulario"] <= 266
    assert relatorio["treinamento"]["tokens"] > 0
    assert relatorio["validacao"]["tokens"] > 0

    assert (destino / "tokenizador.json").exists()
    assert (destino / "corpus_relatorio.json").exists()

    dados = json.loads(
        (destino / "corpus_relatorio.json").read_text(
            encoding="utf-8"
        )
    )

    assert dados["modelo"] == "Cygnus AI v3"


def test_nao_sobrescrever(tmp_path):
    origem = tmp_path / "corpus"
    (origem / "train").mkdir(parents=True)
    (origem / "validation").mkdir(parents=True)

    (origem / "train" / "a.txt").write_text(
        "Treinamento " * 20,
        encoding="utf-8"
    )
    (origem / "validation" / "b.txt").write_text(
        "Validacao " * 20,
        encoding="utf-8"
    )

    destino = tmp_path / "saida"

    executar_analise(
        origem, destino, merges=2, contexto=8
    )

    with pytest.raises(FileExistsError):
        executar_analise(
            origem, destino, merges=2, contexto=8
        )


def test_validacao_nao_influencia_vocabulario(tmp_path):
    origem = tmp_path / "corpus"
    (origem / "train").mkdir(parents=True)
    (origem / "validation").mkdir(parents=True)

    (origem / "train" / "a.txt").write_text(
        "abc abc abc abc abc",
        encoding="utf-8"
    )
    arquivo_validacao = origem / "validation" / "b.txt"
    arquivo_validacao.write_text(
        "xyz xyz", encoding="utf-8"
    )

    executar_analise(
        origem, tmp_path / "saida1",
        merges=4, contexto=4
    )

    arquivo_validacao.write_text(
        "outro texto completamente diferente",
        encoding="utf-8"
    )

    executar_analise(
        origem, tmp_path / "saida2",
        merges=4, contexto=4
    )

    primeiro = (
        tmp_path / "saida1" / "tokenizador.json"
    ).read_bytes()

    segundo = (
        tmp_path / "saida2" / "tokenizador.json"
    ).read_bytes()

    assert primeiro == segundo
