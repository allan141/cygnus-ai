import random
import torch
import pytest
from src.v4_trainer import sortear_lote, janelas_validacao, assinatura


def test_lote_nao_atravessa_documentos():
    random.seed(42)
    sequencias = [torch.arange(300), torch.arange(500, 800)]
    x, y = sortear_lote(sequencias, 16, 8)
    assert x.shape == (8, 16)
    assert torch.equal(y[:, :-1], x[:, 1:])
    assert all((linha < 300).all() or (linha >= 500).all() for linha in x)


def test_validacao_deterministica():
    sequencias = [torch.arange(100), torch.arange(200, 300)]
    a = janelas_validacao(sequencias, 16, 5)
    b = janelas_validacao(sequencias, 16, 5)
    assert len(a) == 5
    assert all(torch.equal(x[0], y[0]) for x, y in zip(a, b))


def test_arquivos_assinatura(tmp_path):
    (tmp_path / 'a.txt').write_text('abc', encoding='utf-8')
    anterior = assinatura(tmp_path)
    (tmp_path / 'a.txt').write_text('abcd', encoding='utf-8')
    assert anterior != assinatura(tmp_path)


def test_validacao_sem_janelas():
    with pytest.raises(ValueError):
        janelas_validacao([torch.arange(10)], 16)
