
import json
import pytest

from src.corpus_collector import (
    selecionar_livros,
    pesquisar_livros,
    salvar_catalogo
)


def exemplo_livro(
    identificador=1,
    idioma="pt",
    copyright=False
):
    return {
        "id": identificador,
        "title": "Livro de teste",
        "languages": [idioma],
        "copyright": copyright,
        "authors": [
            {
                "name": "Autor Exemplo",
                "birth_year": 1800,
                "death_year": 1870
            }
        ],
        "formats": {
            "text/plain; charset=utf-8":
                "https://example.org/livro.txt"
        }
    }


def test_seleciona_portugues():
    dados = {
        "results": [exemplo_livro()]
    }

    livros = selecionar_livros(dados)

    assert len(livros) == 1
    assert livros[0]["idioma"] == "pt"


def test_ignora_outros_idiomas():
    dados = {
        "results": [
            exemplo_livro(idioma="en")
        ]
    }

    assert selecionar_livros(dados) == []


def test_ignora_copyright_restrito():
    dados = {
        "results": [
            exemplo_livro(copyright=True)
        ]
    }

    assert selecionar_livros(dados) == []


def test_direitos_pendentes():
    dados = {
        "results": [exemplo_livro()]
    }

    livro = selecionar_livros(dados)[0]

    assert livro["direitos"] == "revisao_pendente"


def test_preserva_autores():
    dados = {
        "results": [exemplo_livro()]
    }

    livro = selecionar_livros(dados)[0]

    assert livro["autores"][0]["nome"] == "Autor Exemplo"


def test_salvar_catalogo(tmp_path):
    caminho = tmp_path / "catalogo.json"

    salvar_catalogo(
        [{"id": 1, "titulo": "Teste"}],
        caminho
    )

    dados = json.loads(
        caminho.read_text(encoding="utf-8")
    )

    assert dados[0]["id"] == 1


def test_paginas_invalidas():
    with pytest.raises(ValueError):
        pesquisar_livros(paginas=0)


def test_pesquisa_sem_rede(monkeypatch):
    def consulta_falsa(
        idioma="pt",
        pagina=1,
        termo=None
    ):
        return {
            "results": [exemplo_livro()],
            "next": None
        }

    monkeypatch.setattr(
        "src.corpus_collector.consultar_catalogo",
        consulta_falsa
    )

    livros = pesquisar_livros(paginas=2)

    assert len(livros) == 1
    assert livros[0]["id"] == 1
