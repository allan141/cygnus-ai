"""Geracao de texto com o checkpoint do Cygnus v4 (CPU)."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from src.byte_bpe_tokenizer import CygnusByteBPETokenizer
from src.language_model import CygnusLanguageModel

PROMPTS_PADRAO = (
    'Era uma vez',
    'O Brasil é',
    'A inteligência artificial',
    'O sistema solar',
)


def validar_argumentos(max_tokens, temperatura, top_k):
    if max_tokens < 1:
        raise ValueError('--max-tokens deve ser maior que zero')
    if temperatura <= 0:
        raise ValueError('--temperatura deve ser maior que zero')
    if top_k < 0:
        raise ValueError('--top-k nao pode ser negativo (0 desativa o filtro)')


def escolher_proximo_token(logits, temperatura=0.7, top_k=20, gerador=None):
    """Amostra um ID a partir dos logits do ultimo token."""
    validar_argumentos(1, temperatura, top_k)
    if logits.ndim != 1:
        raise ValueError('Esperado vetor 1D de logits')
    valores = logits.float() / temperatura
    if top_k:
        k = min(top_k, valores.numel())
        limite = torch.topk(valores, k).values[-1]
        valores = valores.masked_fill(valores < limite, float('-inf'))
    probabilidades = torch.softmax(valores, dim=-1)
    if not torch.isfinite(probabilidades).all():
        raise ValueError('Probabilidades invalidas na geracao')
    return int(torch.multinomial(probabilidades, 1, generator=gerador).item())


def carregar_modelo(experimento, caminho_tokenizador):
    experimento = Path(experimento)
    caminho_tokenizador = Path(caminho_tokenizador)
    manifesto_path = experimento / 'manifesto.json'
    checkpoint_path = experimento / 'modelo.pt'
    manifesto = json.loads(manifesto_path.read_text(encoding='utf-8'))
    if manifesto.get('versao') != 'cygnus-v4':
        raise ValueError('Manifesto nao pertence ao Cygnus v4')
    assinatura = hashlib.sha256(caminho_tokenizador.read_bytes()).hexdigest()
    if assinatura != manifesto.get('tokenizador_sha256'):
        raise ValueError('Tokenizador diferente do usado no treinamento v4')
    tokenizador = CygnusByteBPETokenizer.carregar(caminho_tokenizador)
    configuracao = manifesto['arquitetura']
    if tokenizador.tamanho_vocabulario != configuracao['tamanho_vocabulario']:
        raise ValueError('Tamanho do vocabulario incompativel com o modelo')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if checkpoint.get('configuracao') != configuracao:
        raise ValueError('Arquitetura do checkpoint diferente do manifesto')
    modelo = CygnusLanguageModel(**configuracao).to('cpu')
    modelo.load_state_dict(checkpoint['modelo'], strict=True)
    modelo.eval()
    return modelo, tokenizador, configuracao, checkpoint['passo']


@torch.inference_mode()
def gerar(modelo, tokenizador, prompt, contexto, max_tokens=80,
          temperatura=0.7, top_k=20, semente=42):
    validar_argumentos(max_tokens, temperatura, top_k)
    if not prompt:
        raise ValueError('O prompt nao pode ser vazio')
    ids = tokenizador.codificar(prompt)
    if not ids:
        raise ValueError('Prompt nao produziu tokens')
    gerador = torch.Generator(device='cpu').manual_seed(semente)
    for _ in range(max_tokens):
        entrada = torch.tensor([ids[-contexto:]], dtype=torch.long)
        logits, _ = modelo(entrada)
        novo = escolher_proximo_token(logits[0, -1], temperatura, top_k, gerador)
        ids.append(novo)
    # Byte-level BPE pode terminar no meio de um caractere UTF-8.
    # Nao alteramos o tokenizador: usamos substituicao apenas na exibicao.
    dados = b''.join(tokenizador.vocabulario[t] for t in ids)
    return dados.decode('utf-8', errors='replace')


def main():
    parser = argparse.ArgumentParser(description='Geracao de texto do Cygnus v4')
    parser.add_argument('--experimento', default='checkpoints/cygnus_v4_teste01')
    parser.add_argument('--tokenizador', default='checkpoints/cygnus_v4_tokenizers/tokenizador_2048.json')
    parser.add_argument('--prompt', default=None, help='Se omitido, testa quatro prompts padrao')
    parser.add_argument('--max-tokens', type=int, default=80)
    parser.add_argument('--temperatura', type=float, default=0.7)
    parser.add_argument('--top-k', type=int, default=20)
    parser.add_argument('--semente', type=int, default=42)
    args = parser.parse_args()
    validar_argumentos(args.max_tokens, args.temperatura, args.top_k)
    modelo, tok, config, passo = carregar_modelo(args.experimento, args.tokenizador)
    print(f'=== CYGNUS V4 | CHECKPOINT PASSO {passo} ===')
    prompts = (args.prompt,) if args.prompt is not None else PROMPTS_PADRAO
    for i, prompt in enumerate(prompts):
        print(f'\nPrompt: {prompt}')
        texto = gerar(modelo, tok, prompt, config['tamanho_contexto'],
                      args.max_tokens, args.temperatura, args.top_k, args.semente + i)
        print('Cygnus:', texto)


if __name__ == '__main__':
    main()
