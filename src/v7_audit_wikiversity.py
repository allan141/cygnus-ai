"""Auditoria inicial do dataset Wikiversidade para Cygnus v7."""

import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from src.byte_bpe_tokenizer import CygnusByteBPETokenizer


ORIGEM = Path("data/v7_extraidos/wikiversity_20261001")
CORPUS = Path("data/v7_corpus_inicial/train")
SAIDA = Path("reports/v7_auditoria_wikiversity.json")
TOKENIZADOR = Path(
    "checkpoints/cygnus_v4_tokenizers/tokenizador_2048.json"
)


def normalizar(texto):
    texto = texto.replace("\r\n", "\n").replace("\r", "\n")
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


def assinatura(texto):
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def main():
    tokenizer = CygnusByteBPETokenizer.carregar(TOKENIZADOR)

    hashes_corpus = set()

    for arquivo in CORPUS.rglob("*.txt"):
        texto = normalizar(arquivo.read_text(encoding="utf-8"))
        hashes_corpus.add(assinatura(texto))

    registros = []
    duplicados_corpus = 0
    total_tokens = 0
    contagem_tokens = Counter()

    arquivos = sorted((ORIGEM / "documentos").glob("*.txt"))

    for indice, arquivo in enumerate(arquivos, 1):
        texto = normalizar(arquivo.read_text(encoding="utf-8"))
        digest = assinatura(texto)

        if digest in hashes_corpus:
            duplicados_corpus += 1
            continue

        tokens = len(tokenizer.codificar(texto))
        total_tokens += tokens

        registros.append({
            "arquivo": arquivo.name,
            "caracteres": len(texto),
            "tokens": tokens,
            "sha256": digest,
        })

        if tokens < 200:
            contagem_tokens["menos_200"] += 1
        elif tokens < 500:
            contagem_tokens["200_499"] += 1
        elif tokens < 2000:
            contagem_tokens["500_1999"] += 1
        else:
            contagem_tokens["2000_ou_mais"] += 1

        if indice % 1000 == 0:
            print(f"Auditados: {indice}/{len(arquivos)}", flush=True)

    random.seed(42)
    amostras = random.sample(registros, min(10, len(registros)))

    relatorio = {
        "documentos_examinados": len(arquivos),
        "documentos_nao_duplicados": len(registros),
        "duplicados_corpus_inicial": duplicados_corpus,
        "tokens_candidatos": total_tokens,
        "faixas_tokens": dict(contagem_tokens),
        "amostras": amostras,
        "documentos": registros,
        "status": "pendente_revisao",
    }

    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    SAIDA.write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== CYGNUS V7 | AUDITORIA WIKIVERSIDADE ===")
    print("Documentos examinados:", len(arquivos))
    print("Documentos não duplicados:", len(registros))
    print("Duplicados com corpus inicial:", duplicados_corpus)
    print("Tokens candidatos:", total_tokens)
    print("Distribuição:", dict(contagem_tokens))
    print("\nAMOSTRAS PARA INSPEÇÃO:")

    for registro in amostras:
        arquivo = ORIGEM / "documentos" / registro["arquivo"]
        texto = arquivo.read_text(encoding="utf-8")
        trecho = re.sub(r"\s+", " ", texto[:350])

        print("\nArquivo:", registro["arquivo"])
        print("Tokens:", registro["tokens"])
        print("Trecho:", trecho)

    print("\nRelatório:", SAIDA)
    print("AUDITORIA CONCLUÍDA")


if __name__ == "__main__":
    main()
