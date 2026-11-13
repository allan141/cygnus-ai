import json
import pytest
from src.corpus_v4 import preparar, normalizar


def test_normalizar():
    assert normalizar('A  B\r\n\r\n\r\nC') == 'A B\n\nC'


def test_separacao_reproduzivel_sem_vazamento(tmp_path):
    origem = tmp_path/'origem'
    origem.mkdir()
    for i in range(12):
        (origem/f'artigo_{i}.txt').write_text((f'Artigo {i} sobre ciência. ' * 50), encoding='utf-8')
    a = preparar(origem, tmp_path/'saida1', tamanho_trecho=120)
    b = preparar(origem, tmp_path/'saida2', tamanho_trecho=120)
    m1 = json.loads((tmp_path/'saida1'/'manifest_train.json').read_text())
    v1 = json.loads((tmp_path/'saida1'/'manifest_validation.json').read_text())
    m2 = json.loads((tmp_path/'saida2'/'manifest_train.json').read_text())
    assert {x['id'] for x in m1}.isdisjoint({x['id'] for x in v1})
    assert [x['id'] for x in m1] == [x['id'] for x in m2]
    assert a['conjuntos'] == b['conjuntos']
    assert a['conjuntos']['train']['documentos'] == 11
    assert a['conjuntos']['validation']['documentos'] == 1


def test_duplicados_e_originais_preservados(tmp_path):
    origem = tmp_path/'origem'
    origem.mkdir()
    for i, texto in enumerate(['A ' * 400, 'A ' * 400, 'B ' * 400]):
        (origem/f'artigo_{i}.txt').write_text(texto, encoding='utf-8')
    resultado = preparar(origem, tmp_path/'saida', tamanho_trecho=200)
    assert len(resultado['ignorados']) == 1
    assert resultado['ignorados'][0]['motivo'] == 'duplicado_exato'
    assert len(list(origem.glob('*.txt'))) == 3


def test_nao_sobrescreve_destino(tmp_path):
    origem = tmp_path/'origem'
    origem.mkdir()
    for i in range(2):
        (origem/f'artigo_{i}.txt').write_text(str(i)*700, encoding='utf-8')
    destino = tmp_path/'saida'
    preparar(origem, destino)
    with pytest.raises(FileExistsError):
        preparar(origem, destino)
