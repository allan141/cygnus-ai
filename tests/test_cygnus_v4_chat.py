import pytest
import torch
from src.cygnus_v4_chat import escolher_proximo_token, gerar, validar_argumentos
from src.byte_bpe_tokenizer import CygnusByteBPETokenizer


def test_argumentos_invalidos():
    for args in [(0, 0.7, 20), (10, 0, 20), (10, 0.7, -1)]:
        with pytest.raises(ValueError):
            validar_argumentos(*args)


def test_top_k_um_seleciona_maior_logit():
    logits = torch.tensor([1.0, 5.0, 2.0])
    assert escolher_proximo_token(logits, top_k=1) == 1


def test_amostragem_reproduzivel():
    logits = torch.tensor([1.0, 2.0, 3.0])
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    a = [escolher_proximo_token(logits, gerador=g1) for _ in range(12)]
    b = [escolher_proximo_token(logits, gerador=g2) for _ in range(12)]
    assert a == b


def test_gerar_preserva_prompt():
    class ModeloFalso:
        def __call__(self, x):
            logits = torch.full((1, x.shape[1], 256), -100.0)
            logits[:, -1, ord('a')] = 10.0
            return logits, None
    tok = CygnusByteBPETokenizer()
    assert gerar(ModeloFalso(), tok, 'Oi', contexto=4, max_tokens=3, top_k=1) == 'Oiaaa'


def test_prompt_vazio():
    with pytest.raises(ValueError):
        gerar(None, CygnusByteBPETokenizer(), '', contexto=4)
