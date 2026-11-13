
import json

import pytest

from src.byte_bpe_tokenizer import CygnusByteBPETokenizer
from src.v3_trainer import (
    preparar_sequencias,
    assinatura_corpus,
)


def test_preparar_sequencias(tmp_path):
    (tmp_path / "texto.txt").write_text(
        "Cygnus aprende linguagem." * 10,
        encoding="utf-8"
    )

    tokenizador = CygnusByteBPETokenizer()

    sequencias = preparar_sequencias(
        tmp_path,
        tokenizador,
        contexto=8
    )

    assert len(sequencias) == 1
    assert len(sequencias[0]) > 8


def test_documento_curto(tmp_path):
    (tmp_path / "curto.txt").write_text(
        "Oi",
        encoding="utf-8"
    )

    tokenizador = CygnusByteBPETokenizer()

    with pytest.raises(ValueError):
        preparar_sequencias(
            tmp_path,
            tokenizador,
            contexto=128
        )


def test_assinatura_estavel(tmp_path):
    arquivo = tmp_path / "texto.txt"
    arquivo.write_text(
        "Texto de treinamento",
        encoding="utf-8"
    )

    primeira = assinatura_corpus(tmp_path)
    segunda = assinatura_corpus(tmp_path)

    assert primeira == segunda


def test_assinatura_muda_com_conteudo(tmp_path):
    arquivo = tmp_path / "texto.txt"

    arquivo.write_text("Primeiro", encoding="utf-8")
    primeira = assinatura_corpus(tmp_path)

    arquivo.write_text("Segundo", encoding="utf-8")
    segunda = assinatura_corpus(tmp_path)

    assert primeira != segunda


def test_assinatura_ignora_json(tmp_path):
    (tmp_path / "texto.txt").write_text(
        "Conteudo", encoding="utf-8"
    )

    primeira = assinatura_corpus(tmp_path)

    (tmp_path / "metadata.json").write_text(
        json.dumps({"titulo": "Teste"}),
        encoding="utf-8"
    )

    assert assinatura_corpus(tmp_path) == primeira
