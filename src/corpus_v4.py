"""Preparação reproduzível do corpus Cygnus v4, sem alterar os originais."""
import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path


def sha(texto):
    return hashlib.sha256(texto.encode('utf-8')).hexdigest()


def normalizar(texto):
    texto = texto.replace('\r\n', '\n').replace('\r', '\n')
    texto = re.sub(r'[ \t]+', ' ', texto)
    texto = re.sub(r'\n{3,}', '\n\n', texto)
    return texto.strip()


def preparar(origem='data/wikipedia', destino='data/v4_corpus', semente=42,
             proporcao_validacao=0.1, tamanho_trecho=3000, minimo_caracteres=500):
    origem, destino = Path(origem), Path(destino)
    if not origem.is_dir():
        raise FileNotFoundError(f'Pasta de origem inexistente: {origem}')
    if not 0 < proporcao_validacao < 1:
        raise ValueError('proporcao_validacao deve estar entre 0 e 1')
    if tamanho_trecho < 100 or minimo_caracteres < 1:
        raise ValueError('tamanho_trecho >= 100 e minimo_caracteres >= 1')
    arquivos = sorted(origem.glob('artigo_*.txt'))
    if len(arquivos) < 2:
        raise ValueError('São necessários pelo menos dois artigos')
    documentos, hashes, ignorados = [], set(), []
    for arquivo in arquivos:
        texto = normalizar(arquivo.read_text(encoding='utf-8'))
        digest = sha(texto.casefold())
        if len(texto) < minimo_caracteres:
            ignorados.append({'arquivo': arquivo.name, 'motivo': 'curto'})
            continue
        if digest in hashes:
            ignorados.append({'arquivo': arquivo.name, 'motivo': 'duplicado_exato'})
            continue
        hashes.add(digest)
        metadados = arquivo.with_suffix('.metadata.json')
        meta = json.loads(metadados.read_text(encoding='utf-8')) if metadados.exists() else {}
        documentos.append({'id': arquivo.stem, 'titulo': meta.get('titulo', arquivo.stem),
                           'url': meta.get('url', meta.get('fonte', '')),
                           'texto': texto, 'hash': digest})
    if len(documentos) < 2:
        raise ValueError('Menos de dois artigos válidos após filtragem')
    # Atribuição determinística por documento; não por trecho.
    rng = random.Random(semente)
    rng.shuffle(documentos)
    qtd_val = max(1, min(len(documentos)-1, round(len(documentos)*proporcao_validacao)))
    grupos = {'validation': documentos[:qtd_val], 'train': documentos[qtd_val:]}
    if destino.exists() and any(destino.iterdir()):
        raise FileExistsError(f'Destino não vazio: {destino}. Use uma nova pasta para não misturar versões.')
    relatorio = {'semente': semente, 'proporcao_validacao': proporcao_validacao,
                 'tamanho_trecho_caracteres': tamanho_trecho,
                 'origem': str(origem), 'ignorados': ignorados, 'conjuntos': {}}
    for grupo, docs in grupos.items():
        pasta = destino/grupo
        pasta.mkdir(parents=True, exist_ok=True)
        manifest = []
        for doc in docs:
            # Cortes em espaços próximos ao limite, preservando palavras.
            restante = doc['texto']
            trechos = []
            while restante:
                if len(restante) <= tamanho_trecho:
                    parte, restante = restante, ''
                else:
                    limite = restante.rfind(' ', 0, tamanho_trecho+1)
                    if limite <= 0:
                        limite = tamanho_trecho
                    parte, restante = restante[:limite], restante[limite:].lstrip()
                if parte.strip():
                    trechos.append(parte.strip())
            for indice, trecho in enumerate(trechos, 1):
                nome = f"{doc['id']}_parte_{indice:04d}.txt"
                (pasta/nome).write_text(trecho+'\n', encoding='utf-8')
            manifest.append({'id': doc['id'], 'titulo': doc['titulo'], 'url': doc['url'],
                             'sha256': doc['hash'], 'caracteres': len(doc['texto']),
                             'trechos': len(trechos)})
        (destino/f'manifest_{grupo}.json').write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        relatorio['conjuntos'][grupo] = {
            'documentos': len(docs), 'trechos': sum(x['trechos'] for x in manifest),
            'caracteres': sum(x['caracteres'] for x in manifest),
            'maiores_documentos': sorted(
                [{'titulo': x['titulo'], 'caracteres': x['caracteres']} for x in manifest],
                key=lambda x: x['caracteres'], reverse=True)[:10]}
    (destino/'relatorio.json').write_text(json.dumps(relatorio, ensure_ascii=False, indent=2), encoding='utf-8')
    return relatorio


def main():
    parser = argparse.ArgumentParser(description='Preparar corpus Cygnus v4')
    parser.add_argument('--origem', default='data/wikipedia')
    parser.add_argument('--destino', default='data/v4_corpus')
    parser.add_argument('--semente', type=int, default=42)
    parser.add_argument('--validacao', type=float, default=0.1)
    parser.add_argument('--trecho', type=int, default=3000)
    args = parser.parse_args()
    resultado = preparar(args.origem, args.destino, args.semente, args.validacao, args.trecho)
    print(json.dumps(resultado['conjuntos'], ensure_ascii=False, indent=2))
    print('Ignorados:', len(resultado['ignorados']))
    print('Relatório salvo em:', Path(args.destino)/'relatorio.json')


if __name__ == '__main__':
    main()
