
import json

import pytest

from src import wikipedia_auto_collector as coletor


def artigo_teste():
    return {
        "id": 123,
        "titulo": "Artigo de teste",
        "url": "https://pt.wikipedia.org/wiki/Teste",
        "texto": "Texto de exemplo. " * 100,
    }


def test_descobrir_artigos(monkeypatch):
    def api_falsa(parametros):
        return {
            "query": {
                "pages": [{
                    "pageid": 123,
                    "ns": 0,
                    "title": "Teste",
                    "fullurl": "https://pt.wikipedia.org/wiki/Teste",
                    "extract": "Texto de teste",
                }]
            }
        }

    monkeypatch.setattr(
        coletor, "consultar_api", api_falsa
    )

    artigos = coletor.descobrir_artigos(1)

    assert len(artigos) == 1
    assert artigos[0]["id"] == 123


def test_quantidade_invalida():
    with pytest.raises(ValueError):
        coletor.descobrir_artigos(0)


def test_salvar_artigo(tmp_path):
    artigo = artigo_teste()
    texto = coletor.limpar_texto(artigo["texto"])

    salvo = coletor.salvar_artigo(
        artigo, texto, tmp_path, 400
    )

    assert salvo
    assert (tmp_path / "artigo_123.txt").exists()

    metadados = json.loads(
        (tmp_path / "artigo_123.metadata.json")
        .read_text(encoding="utf-8")
    )

    assert metadados["tokens_bpe_v3"] == 400


def test_nao_sobrescrever(tmp_path):
    artigo = artigo_teste()

    assert coletor.salvar_artigo(
        artigo, artigo["texto"], tmp_path, 400
    )

    assert not coletor.salvar_artigo(
        artigo, artigo["texto"], tmp_path, 400
    )


def test_carregar_existentes(tmp_path):
    artigo = artigo_teste()

    coletor.salvar_artigo(
        artigo, artigo["texto"], tmp_path, 400
    )

    hashes, ids = coletor.carregar_existentes(
        tmp_path
    )

    assert 123 in ids
    assert len(hashes) == 1


def test_filtro_artigos_sem_texto(monkeypatch):
    monkeypatch.setattr(
        coletor,
        "consultar_api",
        lambda parametros: {
            "query": {
                "pages": [{
                    "pageid": 1,
                    "ns": 0,
                    "title": "Vazio",
                    "extract": "",
                }]
            }
        },
    )

    assert coletor.descobrir_artigos(1) == []
