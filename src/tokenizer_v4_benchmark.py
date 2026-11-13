"""Cygnus v4: compara tokenizadores BPE sem alterar o v3.

Execute: python -m src.tokenizer_v4_benchmark --vocabularios 512
"""
import argparse
import json
import random
import time
from pathlib import Path

from src.byte_bpe_tokenizer import CygnusByteBPETokenizer

FRASES = [
    "A inteligência artificial está evoluindo rapidamente.",
    "O Brasil possui uma grande diversidade cultural.",
    "A aprendizagem de máquina utiliza dados para identificar padrões.",
    "O sistema solar é composto pelo Sol e pelos corpos que o orbitam.",
]


def ler_trechos(diretorio):
    pasta = Path(diretorio)
    if not pasta.is_dir():
        raise FileNotFoundError(f"Pasta não encontrada: {pasta}")
    arquivos = sorted(pasta.rglob("*.txt"))
    if not arquivos:
        raise ValueError(f"Nenhum arquivo .txt em {pasta}")
    return [(p, p.read_text(encoding="utf-8")) for p in arquivos]


def criar_amostra(trechos, limite_caracteres, semente=42):
    """Amostra distribuída por arquivos, apenas do conjunto de treinamento."""
    if limite_caracteres <= 0:
        raise ValueError("amostra-caracteres deve ser positivo")
    itens = [(p, t) for p, t in trechos if t]
    if not itens:
        raise ValueError("Treinamento sem textos não vazios")
    rng = random.Random(semente)
    rng.shuffle(itens)
    total = 0
    amostra = []
    for _, texto in itens:
        if total >= limite_caracteres:
            break
        parte = texto[: limite_caracteres - total]
        if parte:
            amostra.append(parte)
            total += len(parte)
    return amostra


def avaliar(tokenizador, trechos):
    total_caracteres = 0
    total_tokens = 0
    for _, texto in trechos:
        ids = tokenizador.codificar(texto)
        if tokenizador.decodificar(ids) != texto:
            raise AssertionError("Tokenizador não reconstruiu o texto")
        total_caracteres += len(texto)
        total_tokens += len(ids)
    return {
        "arquivos": len(trechos),
        "caracteres": total_caracteres,
        "tokens": total_tokens,
        "caracteres_por_token": round(total_caracteres / total_tokens, 4) if total_tokens else 0,
    }


def executar(vocabularios, origem, destino, amostra_caracteres, semente):
    origem = Path(origem)
    destino = Path(destino)
    train = ler_trechos(origem / "train")
    validation = ler_trechos(origem / "validation")
    amostra = criar_amostra(train, amostra_caracteres, semente)
    destino.mkdir(parents=True, exist_ok=True)
    resultados = []
    for alvo in vocabularios:
        if alvo < 256:
            raise ValueError("Vocabulário deve ter ao menos 256 tokens")
        print(f"\n=== BPE {alvo} ===", flush=True)
        inicio = time.perf_counter()
        tok = CygnusByteBPETokenizer()
        tok.treinar(amostra, numero_merges=alvo - 256)
        duracao = time.perf_counter() - inicio
        caminho = destino / f"tokenizador_{alvo}.json"
        tok.salvar(caminho)
        treino = avaliar(tok, train)
        validacao = avaliar(tok, validation)
        frases = []
        for frase in FRASES:
            ids = tok.codificar(frase)
            if tok.decodificar(ids) != frase:
                raise AssertionError("Falha na reconstrução da frase")
            frases.append({"frase": frase, "tokens": len(ids), "caracteres_por_token": round(len(frase) / len(ids), 4)})
        resultado = {
            "vocabulario_solicitado": alvo,
            "vocabulario_real": tok.tamanho_vocabulario,
            "segundos_treinamento": round(duracao, 2),
            "amostra_caracteres": sum(map(len, amostra)),
            "arquivo": str(caminho),
            "train": treino,
            "validation": validacao,
            "frases": frases,
        }
        resultados.append(resultado)
        print(json.dumps(resultado, ensure_ascii=False, indent=2), flush=True)
    # Mescla resultados existentes para não apagar a comparação anterior.
    relatorio = destino / "comparacao.json"
    anteriores = {}
    if relatorio.exists():
        dados = json.loads(relatorio.read_text(encoding="utf-8"))
        anteriores = {item["vocabulario_solicitado"]: item for item in dados.get("resultados", [])}
    for item in resultados:
        anteriores[item["vocabulario_solicitado"]] = item
    consolidado = {
        "origem": str(origem),
        "semente": semente,
        "resultados": [anteriores[k] for k in sorted(anteriores)],
    }
    relatorio.write_text(json.dumps(consolidado, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRelatório salvo: {relatorio}")
    return consolidado


def main():
    parser = argparse.ArgumentParser(description="Benchmark BPE do Cygnus v4")
    parser.add_argument("--vocabularios", type=int, nargs="+", default=[512])
    parser.add_argument("--amostra-caracteres", type=int, default=120000)
    parser.add_argument("--origem", default="data/v4_corpus")
    parser.add_argument("--destino", default="checkpoints/cygnus_v4_tokenizers")
    parser.add_argument("--semente", type=int, default=42)
    args = parser.parse_args()
    executar(args.vocabularios, args.origem, args.destino, args.amostra_caracteres, args.semente)


if __name__ == "__main__":
    main()
