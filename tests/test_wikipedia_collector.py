
import json

import pytest

from src import wikipedia_collector as coletor


def test_limpar_texto():
    texto = "  Ola   mundo \r\n\r\n  Cygnus  "

    assert coletor.limpar_texto(texto) == (
        "Ola mundo\nCygnus"
    )


def test_limpar_texto_invalido():
    with pytest.raises(TypeError):
        coletor.limpar_texto(None)


def test_hash_reproduzivel():
    assert coletor.identificar_texto(
        "Ola mundo"
    ) == coletor.identificar_texto(
        "  ola   mundo "
    )


def test_buscar_artigo(monkeypatch):
    def api_simulada(parametros):
        return {
            "query": {
                "pages": [{
                    "pageid": 123,
                    "ns": 0,
                    "title": "Python",
                    "fullurl": "https://pt.wikipedia.org/wiki/Python",
                    "extract": "Python e uma linguagem.",
                }]
            }
        }

    monkeypatch.setattr(
        coletor,
        "consultar_api",
        api_simulada,
    )

    artigo = coletor.buscar_artigo("Python")

    assert artigo["id"] == 123
    assert artigo["titulo"] == "Python"


def test_artigo_inexistente(monkeypatch):
    monkeypatch.setattr(
        coletor,
        "consultar_api",
        lambda parametros: {
            "query": {
                "pages": [{"missing": True}]
            }
        },
    )

    assert coletor.buscar_artigo("Inexistente") is None


def test_coleta_e_deduplicacao(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        coletor,
        "TITULOS",
        ["Artigo teste"],
    )

    monkeypatch.setattr(
        coletor,
        "buscar_artigo",
        lambda titulo: {
            "id": 100,
            "titulo": titulo,
            "url": "https://pt.wikipedia.org/wiki/Teste",
            "texto": "Texto de exemplo. " * 40,
        },
    )

    primeiro = coletor.coletar_artigos(
        destino=tmp_path,
        limite=1,
        minimo_caracteres=100,
        pausa=0,
    )

    segundo = coletor.coletar_artigos(
        destino=tmp_path,
        limite=1,
        minimo_caracteres=100,
        pausa=0,
    )

    assert primeiro["coletados"] == 1
    assert segundo["coletados"] == 0

    assert (
        tmp_path / "artigo_100.txt"
    ).is_file()

    metadados = json.loads(
        (
            tmp_path / "artigo_100.metadata.json"
        ).read_text(encoding="utf-8")
    )

    assert metadados["pageid"] == 100


def test_limite_invalido(tmp_path):
    with pytest.raises(ValueError):
        coletor.coletar_artigos(
            destino=tmp_path,
            limite=0,
        )

