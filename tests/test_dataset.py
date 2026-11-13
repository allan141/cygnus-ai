
import pytest
import torch

from src.dataset import (
    carregar_documentos,
    preparar_tokenizador,
    tokenizar_documentos,
    criar_lote_documentos,
    preparar_dataset
)


def test_carregar_documentos(tmp_path):
    (tmp_path / "a.txt").write_text(
        "Cygnus aprende.",
        encoding="utf-8"
    )

    documentos = carregar_documentos(tmp_path)

    assert documentos == ["Cygnus aprende."]


def test_diretorio_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError):
        carregar_documentos(tmp_path / "inexistente")


def test_diretorio_sem_txt(tmp_path):
    with pytest.raises(ValueError):
        carregar_documentos(tmp_path)


def test_ignora_documentos_vazios(tmp_path):
    (tmp_path / "vazio.txt").write_text(
        " ",
        encoding="utf-8"
    )

    (tmp_path / "valido.txt").write_text(
        "Cygnus AI",
        encoding="utf-8"
    )

    documentos = carregar_documentos(tmp_path)

    assert documentos == ["Cygnus AI"]


def test_tokenizador_apenas_treino():
    treino = ["Cygnus aprende aprende aprende"]
    validacao = ["🌟🌟🌟"]

    tokenizador = preparar_tokenizador(
        treino,
        numero_merges=10
    )

    # Caracteres de validacao continuam representaveis,
    # mesmo sem terem participado dos merges.
    ids = tokenizador.codificar(validacao[0])

    assert tokenizador.decodificar(ids) == validacao[0]


def test_tokenizacao_documentos():
    texto = "Cygnus aprende linguagem." * 5

    tokenizador = preparar_tokenizador(
        [texto],
        numero_merges=10
    )

    sequencias = tokenizar_documentos(
        [texto],
        tokenizador,
        tamanho_contexto=8
    )

    assert len(sequencias) == 1
    assert sequencias[0].dtype == torch.long
    assert len(sequencias[0]) > 8


def test_lote_dimensoes():
    sequencias = [
        torch.arange(100, dtype=torch.long)
    ]

    x, y = criar_lote_documentos(
        sequencias,
        tamanho_contexto=16,
        tamanho_lote=4
    )

    assert x.shape == (4, 16)
    assert y.shape == (4, 16)


def test_alvos_deslocados():
    sequencias = [
        torch.arange(100, dtype=torch.long)
    ]

    x, y = criar_lote_documentos(
        sequencias,
        tamanho_contexto=16,
        tamanho_lote=4
    )

    assert torch.equal(x[:, 1:], y[:, :-1])


def test_documento_curto():
    sequencias = [
        torch.arange(5, dtype=torch.long)
    ]

    with pytest.raises(ValueError):
        criar_lote_documentos(
            sequencias,
            tamanho_contexto=16
        )


def test_dataset_completo(tmp_path):
    treino_dir = tmp_path / "train"
    validacao_dir = tmp_path / "validation"

    treino_dir.mkdir()
    validacao_dir.mkdir()

    (treino_dir / "treino.txt").write_text(
        "Cygnus aprende linguagem. " * 20,
        encoding="utf-8"
    )

    (validacao_dir / "validacao.txt").write_text(
        "Um Transformer utiliza atencao. " * 20,
        encoding="utf-8"
    )

    treino, validacao, tokenizador = preparar_dataset(
        diretorio_treino=treino_dir,
        diretorio_validacao=validacao_dir,
        numero_merges=20,
        tamanho_contexto=16
    )

    assert len(treino) == 1
    assert len(validacao) == 1
    assert tokenizador.tamanho_vocabulario >= 256
