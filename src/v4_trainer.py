"""Treinador Cygnus v4: janelas por arquivo, validacao fixa e retomada."""
import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path

import torch

from src.byte_bpe_tokenizer import CygnusByteBPETokenizer
from src.language_model import CygnusLanguageModel
from src.model_config import ConfiguracaoCygnus


def arquivos_txt(pasta):
    arquivos = sorted(Path(pasta).glob('*.txt'))
    if not arquivos:
        raise ValueError(f'Nenhum .txt encontrado em {pasta}')
    return arquivos


def carregar_sequencias(pasta, tokenizador, contexto):
    sequencias = []
    for arquivo in arquivos_txt(pasta):
        ids = tokenizador.codificar(arquivo.read_text(encoding='utf-8'))
        if len(ids) >= contexto + 1:
            sequencias.append(torch.tensor(ids, dtype=torch.long))
    if not sequencias:
        raise ValueError(f'Nenhum trecho com pelo menos {contexto + 1} tokens em {pasta}')
    return sequencias


def assinatura(pasta):
    h = hashlib.sha256()
    for arquivo in arquivos_txt(pasta):
        h.update(arquivo.name.encode('utf-8'))
        h.update(b'\x00')
        h.update(arquivo.read_bytes())
        h.update(b'\x00')
    return h.hexdigest()


def sortear_lote(sequencias, contexto, batch_size):
    xs, ys = [], []
    for _ in range(batch_size):
        sequencia = random.choice(sequencias)
        inicio = random.randint(0, len(sequencia) - contexto - 1)
        xs.append(sequencia[inicio:inicio + contexto])
        ys.append(sequencia[inicio + 1:inicio + contexto + 1])
    return torch.stack(xs), torch.stack(ys)


def janelas_validacao(sequencias, contexto, max_janelas=32):
    """Janelas nao sobrepostas; selecao fixa e limitada para CPU."""
    janelas = []
    for sequencia in sequencias:
        for inicio in range(0, len(sequencia) - contexto, contexto):
            janelas.append((sequencia[inicio:inicio + contexto],
                            sequencia[inicio + 1:inicio + contexto + 1]))
    if not janelas:
        raise ValueError('Sem janelas de validacao')
    if len(janelas) > max_janelas:
        indices = [(i * len(janelas)) // max_janelas for i in range(max_janelas)]
        janelas = [janelas[i] for i in indices]
    return janelas


@torch.no_grad()
def avaliar_fixo(modelo, janelas):
    estava_treinando = modelo.training
    modelo.eval()
    perdas = []
    for x, y in janelas:
        _, perda = modelo(x.unsqueeze(0), y.unsqueeze(0))
        perdas.append(float(perda.item()))
    if estava_treinando:
        modelo.train()
    return sum(perdas) / len(perdas)


def salvar_atomico(caminho, dados):
    temporario = caminho.with_suffix('.tmp')
    torch.save(dados, temporario)
    temporario.replace(caminho)


def treinar(args):
    if args.passos <= 0 or args.batch_size <= 0 or args.avaliar_cada <= 0:
        raise ValueError('Passos, batch size e intervalo devem ser positivos')
    if args.lr <= 0 or args.janelas_validacao <= 0:
        raise ValueError('LR e quantidade de janelas devem ser positivos')
    random.seed(args.semente)
    torch.manual_seed(args.semente)

    corpus = Path(args.corpus)
    tokenizador_path = Path(args.tokenizador)
    destino = Path(args.experimento)
    modelo_path = destino / 'modelo.pt'
    manifesto_path = destino / 'manifesto.json'
    metricas_path = destino / 'metricas.csv'

    tok = CygnusByteBPETokenizer.carregar(tokenizador_path)
    config = ConfiguracaoCygnus(dimensao_embedding=128, numero_cabecas=4,
                                numero_camadas=4, tamanho_contexto=256)
    arquitetura = config.para_modelo(tok.tamanho_vocabulario)
    contexto = config.tamanho_contexto
    treino = carregar_sequencias(corpus / 'train', tok, contexto)
    validacao = carregar_sequencias(corpus / 'validation', tok, contexto)
    janelas = janelas_validacao(validacao, contexto, args.janelas_validacao)
    manifesto = {
        'versao': 'cygnus-v4', 'arquitetura': arquitetura,
        'tokenizador_sha256': hashlib.sha256(tokenizador_path.read_bytes()).hexdigest(),
        'treino_sha256': assinatura(corpus / 'train'),
        'validacao_sha256': assinatura(corpus / 'validation'),
        'semente': args.semente, 'lr': args.lr, 'batch_size': args.batch_size,
        'janelas_validacao': args.janelas_validacao,
    }
    if args.retomar:
        if not modelo_path.exists() or not manifesto_path.exists():
            raise FileNotFoundError('Checkpoint ou manifesto nao encontrado para retomada')
        anterior = json.loads(manifesto_path.read_text(encoding='utf-8'))
        if anterior != manifesto:
            raise ValueError('Manifesto diferente: nao e seguro retomar')
    elif any(p.exists() for p in (modelo_path, manifesto_path, metricas_path)):
        raise FileExistsError('Experimento existente. Use outro diretorio ou --retomar')

    modelo = CygnusLanguageModel(**arquitetura).to('cpu')
    otimizador = torch.optim.AdamW(modelo.parameters(), lr=args.lr)
    passo_inicial = 0
    if args.retomar:
        ckpt = torch.load(modelo_path, map_location='cpu', weights_only=True)
        if ckpt['configuracao'] != arquitetura:
            raise ValueError('Arquitetura incompatível')
        modelo.load_state_dict(ckpt['modelo'])
        otimizador.load_state_dict(ckpt['otimizador'])
        random.setstate(ckpt['estado_python'])
        torch.set_rng_state(ckpt['estado_torch'])
        passo_inicial = ckpt['passo']
    else:
        destino.mkdir(parents=True, exist_ok=True)
        manifesto_path.write_text(json.dumps(manifesto, indent=2, ensure_ascii=False), encoding='utf-8')
        with metricas_path.open('w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(['passo', 'perda_treino_lote', 'perda_validacao', 'perplexidade_validacao'])

    print('\n=== CYGNUS V4 - TREINAMENTO EXPERIMENTAL ===')
    print('Parametros:', sum(p.numel() for p in modelo.parameters()))
    print('Vocabulario:', tok.tamanho_vocabulario, '| Contexto:', contexto)
    print('Trechos utilizaveis:', len(treino), '| Validacao:', len(validacao))
    print('Janelas fixas:', len(janelas), '| Passo inicial:', passo_inicial)
    if passo_inicial >= args.passos:
        print('Nenhum passo adicional necessario.')
        return

    modelo.train()
    try:
        for passo in range(passo_inicial + 1, args.passos + 1):
            x, y = sortear_lote(treino, contexto, args.batch_size)
            _, perda = modelo(x, y)
            otimizador.zero_grad(set_to_none=True)
            perda.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            otimizador.step()
            if passo == 1 or passo % args.avaliar_cada == 0 or passo == args.passos:
                perda_validacao = avaliar_fixo(modelo, janelas)
                ppl = math.exp(min(perda_validacao, 80))
                print(f'Passo {passo:05d} | Treino(lote): {perda.item():.4f} | '
                      f'Validacao: {perda_validacao:.4f} | PPL: {ppl:.2f}', flush=True)
                with metricas_path.open('a', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerow([passo, float(perda.item()), perda_validacao, ppl])
                salvar_atomico(modelo_path, {
                    'passo': passo, 'configuracao': arquitetura,
                    'modelo': modelo.state_dict(), 'otimizador': otimizador.state_dict(),
                    'estado_python': random.getstate(), 'estado_torch': torch.get_rng_state(),
                })
    except KeyboardInterrupt:
        print('\nInterrompido. O ultimo checkpoint salvo pode ser retomado com --retomar.')
        return
    print('\nConcluido:', modelo_path)


def main():
    p = argparse.ArgumentParser(description='Treinador Cygnus v4')
    p.add_argument('--corpus', default='data/v4_corpus')
    p.add_argument('--tokenizador', default='checkpoints/cygnus_v4_tokenizers/tokenizador_2048.json')
    p.add_argument('--experimento', default='checkpoints/cygnus_v4_teste01')
    p.add_argument('--passos', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--avaliar-cada', type=int, default=25)
    p.add_argument('--janelas-validacao', type=int, default=16)
    p.add_argument('--lr', type=float, default=0.0003)
    p.add_argument('--semente', type=int, default=42)
    p.add_argument('--retomar', action='store_true')
    treinar(p.parse_args())


if __name__ == '__main__':
    main()
