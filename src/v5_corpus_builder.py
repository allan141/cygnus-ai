"""Reconstrói artigos do v4, limpa seções finais e produz corpus v5.

Não altera os arquivos de entrada. Requer nomes artigo_<id>_parte_<numero>.txt.
"""
import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

PADRAO = re.compile(r'^artigo_(\d+)_parte_(\d+)\.txt$')
SECOES_FINAIS = {'referências', 'referencias', 'bibliografia', 'ligações externas', 'ligacoes externas', 'notas e referências', 'notas e referencias', 'fontes'}


def identificar(nome):
    match = PADRAO.fullmatch(nome)
    if not match:
        raise ValueError(f'Nome inesperado: {nome}')
    return match.group(1), int(match.group(2))


def reconstruir(pasta):
    grupos = defaultdict(list)
    for caminho in sorted(pasta.glob('*.txt')):
        artigo, parte = identificar(caminho.name)
        grupos[artigo].append((parte, caminho))
    resultado = {}
    for artigo, partes in grupos.items():
        numeros = [n for n, _ in partes]
        if len(numeros) != len(set(numeros)):
            raise ValueError(f'Partes repetidas no artigo {artigo}')
        # O corpus v4 foi dividido por caracteres; concatenar sem inserir espaços
        # evita quebrar palavras nas fronteiras dos blocos.
        resultado[artigo] = ''.join(p.read_text(encoding='utf-8') for _, p in sorted(partes))
    return resultado


def limpar(texto):
    linhas = texto.splitlines()
    corte = len(linhas)
    for i, linha in enumerate(linhas):
        titulo = re.sub(r'\s+', ' ', linha).strip().casefold().rstrip(':')
        if titulo in SECOES_FINAIS:
            corte = i
            break
    limpo = '\n'.join(linhas[:corte]).strip()
    limpo = re.sub(r'[ \t]+', ' ', limpo)
    limpo = re.sub(r'\n{3,}', '\n\n', limpo)
    return limpo


def blocos(texto, limite=3000):
    if limite < 100:
        raise ValueError('O limite deve ser pelo menos 100 caracteres')
    # Preferir limites de parágrafo e frase, sem perder caracteres.
    saida = []
    restante = texto
    while len(restante) > limite:
        janela = restante[:limite + 1]
        candidatos = [janela.rfind('\n\n'), janela.rfind('\n'), janela.rfind('. '), janela.rfind(' ')]
        ponto = next((x for x in candidatos if x >= int(limite * .55)), limite)
        if ponto < limite and restante[ponto:ponto + 2] == '. ':
            ponto += 1
        parte = restante[:ponto].strip()
        if not parte:
            ponto = limite
            parte = restante[:ponto]
        saida.append(parte)
        restante = restante[ponto:].lstrip()
    if restante.strip():
        saida.append(restante.strip())
    # Combinar caudas muito curtas com o bloco anterior, mesmo que o último
    # bloco fique acima do limite. Isso evita perder frases de encerramento.
    if len(saida) > 1 and len(saida[-1]) < 200:
        if len(saida) >= 2:
            ultimo = saida.pop()
            saida[-1] += ' ' + ultimo
    return saida


def construir(origem, destino, limite=3000):
    origem, destino = Path(origem), Path(destino)
    conjuntos = {}
    for conjunto in ('train', 'validation'):
        pasta = origem / conjunto
        if not pasta.is_dir():
            raise FileNotFoundError(pasta)
        conjuntos[conjunto] = reconstruir(pasta)
    ids_t, ids_v = set(conjuntos['train']), set(conjuntos['validation'])
    if ids_t & ids_v:
        raise ValueError(f'Artigos em ambos os conjuntos: {sorted(ids_t & ids_v)}')
    # Planejar tudo antes de escrever: arquivos de origem permanecem intocados.
    plano = {}
    relatorio = {'origem': str(origem), 'destino': str(destino), 'limite_bloco': limite, 'conjuntos': {}}
    for conjunto, artigos in conjuntos.items():
        arquivos = {}
        originais = sum(len(t) for t in artigos.values())
        limpos = 0
        removidos = []
        for artigo, texto in sorted(artigos.items(), key=lambda x: int(x[0])):
            novo = limpar(texto)
            limpos += len(novo)
            if not novo:
                removidos.append(artigo)
                continue
            for i, parte in enumerate(blocos(novo, limite), 1):
                arquivos[f'artigo_{artigo}_parte_{i:04d}.txt'] = parte
        plano[conjunto] = arquivos
        relatorio['conjuntos'][conjunto] = {
            'artigos_originais': len(artigos), 'artigos_vazios_apos_limpeza': removidos,
            'arquivos_gerados': len(arquivos), 'caracteres_originais': originais,
            'caracteres_limpos': limpos, 'caracteres_removidos': originais - limpos,
        }
    # Impede deixar arquivos obsoletos de uma execução anterior.
    if destino.resolve() == origem.resolve() or origem.resolve() in destino.resolve().parents:
        raise ValueError('Destino não pode ser a origem nem uma subpasta dela')
    if destino.exists() and any(destino.iterdir()):
        raise FileExistsError(f'Destino não vazio: {destino}. Use uma pasta nova.')
    for conjunto, arquivos in plano.items():
        pasta = destino / conjunto
        pasta.mkdir(parents=True, exist_ok=True)
        for nome, conteudo in arquivos.items():
            (pasta / nome).write_text(conteudo, encoding='utf-8')
    manifest = {}
    for conjunto, arquivos in plano.items():
        manifest[conjunto] = {nome: hashlib.sha256(texto.encode('utf-8')).hexdigest()
                              for nome, texto in sorted(arquivos.items())}
    relatorio['sha256_arquivos'] = manifest
    (destino / 'manifest.json').write_text(json.dumps(relatorio, ensure_ascii=False, indent=2), encoding='utf-8')
    return relatorio


def main():
    parser = argparse.ArgumentParser(description='Cygnus v5: reconstrução e limpeza do corpus')
    parser.add_argument('--origem', default='data/v4_corpus')
    parser.add_argument('--destino', default='data/v5_corpus')
    parser.add_argument('--limite-bloco', type=int, default=3000)
    args = parser.parse_args()
    resultado = construir(args.origem, args.destino, args.limite_bloco)
    for conjunto, dados in resultado['conjuntos'].items():
        print(f"{conjunto}: {dados['artigos_originais']} artigos | "
              f"{dados['arquivos_gerados']} blocos | "
              f"{dados['caracteres_originais']} -> {dados['caracteres_limpos']} caracteres")
    print(f"Manifesto: {args.destino}/manifest.json")

if __name__ == '__main__':
    main()
