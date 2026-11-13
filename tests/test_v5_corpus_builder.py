import json
import pytest
from src.v5_corpus_builder import blocos, construir, limpar, reconstruir


def criar(tmp_path, conjunto, nome, texto):
    pasta = tmp_path / 'v4' / conjunto
    pasta.mkdir(parents=True, exist_ok=True)
    (pasta / nome).write_text(texto, encoding='utf-8')


def test_limpeza_referencias():
    assert limpar('Conceito importante.\nReferências\nISBN 123') == 'Conceito importante.'


def test_preserva_texto_com_referencia_inline():
    assert 'referências úteis' in limpar('Existem referências úteis no código.')


def test_reconstrucao_sem_espaco_extra(tmp_path):
    criar(tmp_path, 'train', 'artigo_1_parte_0002.txt', 'mundo')
    criar(tmp_path, 'train', 'artigo_1_parte_0001.txt', 'Olá ')
    assert reconstruir(tmp_path / 'v4' / 'train')['1'] == 'Olá mundo'


def test_blocos_sem_cortar_cauda(tmp_path):
    partes = blocos('Palavra ' * 500, 300)
    assert len(partes) > 1
    assert all(p for p in partes)
    assert len(partes[-1]) >= 200


def test_builder_separa_conjuntos_e_preserva_origem(tmp_path):
    criar(tmp_path, 'train', 'artigo_1_parte_0001.txt', 'Aprender Python.\nReferências\nISBN 123')
    criar(tmp_path, 'validation', 'artigo_2_parte_0001.txt', 'Validar o modelo.')
    resultado = construir(tmp_path / 'v4', tmp_path / 'v5')
    assert resultado['conjuntos']['train']['artigos_originais'] == 1
    assert 'ISBN' not in (tmp_path / 'v5/train/artigo_1_parte_0001.txt').read_text()
    assert (tmp_path / 'v5/validation/artigo_2_parte_0001.txt').exists()
    assert 'ISBN' in (tmp_path / 'v4/train/artigo_1_parte_0001.txt').read_text()
    assert json.loads((tmp_path / 'v5/manifest.json').read_text())['sha256_arquivos']['train']


def test_rejeita_vazamento_artigos(tmp_path):
    criar(tmp_path, 'train', 'artigo_1_parte_0001.txt', 'A')
    criar(tmp_path, 'validation', 'artigo_1_parte_0001.txt', 'B')
    with pytest.raises(ValueError, match='ambos'):
        construir(tmp_path / 'v4', tmp_path / 'v5')


def test_rejeita_destino_preenchido(tmp_path):
    criar(tmp_path, 'train', 'artigo_1_parte_0001.txt', 'A')
    criar(tmp_path, 'validation', 'artigo_2_parte_0001.txt', 'B')
    (tmp_path / 'v5').mkdir()
    (tmp_path / 'v5' / 'antigo.txt').write_text('Não apagar')
    with pytest.raises(FileExistsError):
        construir(tmp_path / 'v4', tmp_path / 'v5')
    assert (tmp_path / 'v5' / 'antigo.txt').read_text() == 'Não apagar'
