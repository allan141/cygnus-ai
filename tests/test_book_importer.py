
import pytest

from src.book_importer import (
    limpar_gutenberg,
    importar_livro
)


def catalogo_exemplo():
    return {
        123: {
            "id": 123,
            "titulo": "Livro de teste",
            "pagina": "https://example.org/123"
        }
    }


def test_limpeza_gutenberg():
    texto = (
        "Cabecalho\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
        "Conteudo do livro.\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
        "Rodape"
    )

    assert limpar_gutenberg(texto) == (
        "Conteudo do livro."
    )


def test_preserva_texto_sem_marcadores():
    texto = "Um texto comum em português."
    assert limpar_gutenberg(texto) == texto


def test_normaliza_quebras():
    texto = "Linha 1\r\n\r\n\r\nLinha 2"
    assert limpar_gutenberg(texto) == (
        "Linha 1\n\nLinha 2"
    )


def test_exige_autorizacao(tmp_path):
    with pytest.raises(PermissionError):
        importar_livro(
            123,
            catalogo_exemplo(),
            tmp_path,
            tmp_path / "saida"
        )


def test_arquivo_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError):
        importar_livro(
            123,
            catalogo_exemplo(),
            tmp_path,
            tmp_path / "saida",
            autorizado=True
        )


def test_livro_nao_catalogado(tmp_path):
    with pytest.raises(ValueError):
        importar_livro(
            999,
            catalogo_exemplo(),
            tmp_path,
            tmp_path / "saida",
            autorizado=True
        )


def test_importacao_completa(tmp_path):
    downloads = tmp_path / "downloads"
    destino = tmp_path / "raw"

    downloads.mkdir()

    (downloads / "123.txt").write_text(
        "Este é um documento de teste. " * 20,
        encoding="utf-8"
    )

    resultado = importar_livro(
        123,
        catalogo_exemplo(),
        downloads,
        destino,
        autorizado=True
    )

    assert resultado["id"] == 123
    assert resultado["caracteres"] > 100
    assert (destino / "livro_123.txt").exists()


def test_nao_sobrescreve(tmp_path):
    downloads = tmp_path / "downloads"
    destino = tmp_path / "raw"

    downloads.mkdir()

    (downloads / "123.txt").write_text(
        "Documento para testar sobrescrita. " * 20,
        encoding="utf-8"
    )

    importar_livro(
        123,
        catalogo_exemplo(),
        downloads,
        destino,
        autorizado=True
    )

    with pytest.raises(FileExistsError):
        importar_livro(
            123,
            catalogo_exemplo(),
            downloads,
            destino,
            autorizado=True
        )
