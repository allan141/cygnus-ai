import json
from pathlib import Path

import pytest

from src.byte_bpe_tokenizer import CygnusByteBPETokenizer
from src.tokenizer_v4_benchmark import avaliar, criar_amostra, executar, ler_trechos


def test_amostra_reproduzivel():
    textos = [(Path(str(i)), "Texto de exemplo " * 10) for i in range(12)]
    assert criar_amostra(textos, 150, 42) == criar_amostra(textos, 150, 42)
    assert sum(map(len, criar_amostra(textos, 150, 42))) == 150


def test_amostra_limite_invalido():
    with pytest.raises(ValueError):
        criar_amostra([(Path("a"), "texto")], 0)


def test_avaliacao_reconstrucao():
    tok = CygnusByteBPETokenizer()
    tok.treinar(["Olá mundo! " * 5], numero_merges=8)
    resultado = avaliar(tok, [(Path("a.txt"), "Olá mundo! 🌟")])
    assert resultado["tokens"] > 0
    assert resultado["caracteres"] == len("Olá mundo! 🌟")


def test_execucao_completa(tmp_path):
    origem = tmp_path / "corpus"
    (origem / "train").mkdir(parents=True)
    (origem / "validation").mkdir()
    (origem / "train" / "a.txt").write_text("O Brasil é diverso. " * 10, encoding="utf-8")
    (origem / "validation" / "b.txt").write_text("A Terra é azul.", encoding="utf-8")
    destino = tmp_path / "saida"
    dados = executar([260], origem, destino, 100, 42)
    assert dados["resultados"][0]["vocabulario_real"] == 260
    assert (destino / "tokenizador_260.json").exists()
    assert json.loads((destino / "comparacao.json").read_text(encoding="utf-8"))["resultados"]
    # Segunda execução deve preservar a primeira comparação.
    dados2 = executar([256], origem, destino, 100, 42)
    assert len(dados2["resultados"]) == 2
