"""Cygnus v7 — limpeza e amostragem dos candidatos."""

import argparse
import hashlib
import html
import json
import random
import re
from collections import Counter
from pathlib import Path

from src.byte_bpe_tokenizer import CygnusByteBPETokenizer


ORIGEM = Path("data/v7_extraidos/wikiversity_20261001/documentos")
RELATORIO = Path("reports/v7_filtro_qualidade_v2.json")
DESTINO = Path("data/v7_curadoria/wikiversity_limpo")
RELATORIO_SAIDA = Path("reports/v7_limpeza_wikiversity.json")
AMOSTRAS_SAIDA = Path("reports/v7_amostras_wikiversity.txt")
TOKENIZADOR = Path(
    "checkpoints/cygnus_v4_tokenizers/tokenizador_2048.json"
)


def limpar_texto(texto):
    texto = html.unescape(texto)

    # Remove elementos HTML sem apagar o texto interno.
    texto = re.sub(
        r"<(?:script|style)\b[^>]*>.*?</(?:script|style)>",
        " ",
        texto,
        flags=re.I | re.S,
    )

    texto = re.sub(r"<[^>]+>", " ", texto)

    # Remove palavras de controle MediaWiki.
    texto = re.sub(
        r"__(?:NOTOC|TOC|NOEDITSECTION|FORCETOC)__",
        " ",
        texto,
        flags=re.I,
    )

    # Remove resíduos comuns de imagens.
    texto = re.sub(
        r"\b(?:left|right|center|thumb|thumbnail)"
        r"\s*\|\s*(?:\d+px)?",
        " ",
        texto,
        flags=re.I,
    )

    # Remove URLs isoladas, mas mantém o texto explicativo.
    texto = re.sub(r"https?://[^\s<>]+", " ", texto)
    texto = re.sub(r"www\.[^\s<>]+", " ", texto)

    # Normaliza espaços preservando parágrafos.
    texto = texto.replace("\r\n", "\n").replace("\r", "\n")
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r" *\n *", "\n", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)

    return texto.strip()


def assinatura(texto):
    return hashlib.sha256(
        texto.encode("utf-8")
    ).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--amostras", type=int, default=20)
    parser.add_argument("--min-tokens", type=int, default=200)
    args = parser.parse_args()

    if args.amostras < 1 or args.min_tokens < 1:
        raise ValueError("Parâmetros devem ser positivos.")

    if DESTINO.exists():
        raise FileExistsError(
            f"{DESTINO} já existe. "
            "Não sobrescreveremos resultados anteriores."
        )

    dados = json.loads(
        RELATORIO.read_text(encoding="utf-8")
    )

    tokenizer = CygnusByteBPETokenizer.carregar(TOKENIZADOR)

    candidatos = [
        registro
        for registro in dados["documentos"]
        if registro["categoria_v2"] == "candidato"
    ]

    DESTINO.mkdir(parents=True, exist_ok=False)

    documentos_destino = DESTINO / "documentos"
    documentos_destino.mkdir()

    registros = []
    hashes = set()
    descartados = Counter()
    tokens_antes = 0
    tokens_depois = 0

    for indice, candidato in enumerate(candidatos, 1):
        caminho = ORIGEM / candidato["arquivo"]
        texto_original = caminho.read_text(encoding="utf-8")

        texto_limpo = limpar_texto(texto_original)

        tokens_antes += candidato["tokens"]

        if not texto_limpo:
            descartados["vazio_apos_limpeza"] += 1
            continue

        quantidade_tokens = len(
            tokenizer.codificar(texto_limpo)
        )

        if quantidade_tokens < args.min_tokens:
            descartados["menos_tokens_minimos"] += 1
            continue

        digest = assinatura(texto_limpo)

        if digest in hashes:
            descartados["duplicado_apos_limpeza"] += 1
            continue

        hashes.add(digest)

        nome_arquivo = candidato["arquivo"]

        (documentos_destino / nome_arquivo).write_text(
            texto_limpo,
            encoding="utf-8",
        )

        registro = {
            "arquivo": nome_arquivo,
            "arquivo_original": candidato["arquivo"],
            "tokens_antes": candidato["tokens"],
            "tokens_depois": quantidade_tokens,
            "sha256_limpo": digest,
            "status": "pendente_revisao",
        }

        registros.append(registro)
        tokens_depois += quantidade_tokens

        if indice % 500 == 0:
            print(
                f"Processados: {indice}/{len(candidatos)}",
                flush=True,
            )

    random.seed(42)
    amostras = random.sample(
        registros,
        min(args.amostras, len(registros)),
    )

    linhas_amostras = [
        "CYGNUS V7 — AMOSTRAS ALEATÓRIAS PARA INSPEÇÃO",
        "Documentos ainda não aprovados para treinamento.",
        "",
    ]

    for numero, registro in enumerate(amostras, 1):
        caminho = documentos_destino / registro["arquivo"]
        texto = caminho.read_text(encoding="utf-8")

        linhas_amostras.extend([
            "=" * 65,
            f"AMOSTRA {numero}",
            f"Arquivo: {registro['arquivo']}",
            f"Tokens: {registro['tokens_depois']}",
            "",
            texto[:1500],
            "",
        ])

    relatorio = {
        "versao": "cygnus-v7-curadoria-21.1",
        "documentos_candidatos": len(candidatos),
        "documentos_limpos": len(registros),
        "tokens_antes": tokens_antes,
        "tokens_depois": tokens_depois,
        "descartados": dict(descartados),
        "documentos": registros,
        "status": "pendente_revisao",
    }

    RELATORIO_SAIDA.parent.mkdir(
        parents=True, exist_ok=True
    )

    RELATORIO_SAIDA.write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    AMOSTRAS_SAIDA.write_text(
        "\n".join(linhas_amostras),
        encoding="utf-8",
    )

    print("\n=== CYGNUS V7 | LIMPEZA WIKIVERSIDADE ===")
    print("Candidatos recebidos:", len(candidatos))
    print("Documentos limpos:", len(registros))
    print("Tokens antes:", tokens_antes)
    print("Tokens depois:", tokens_depois)
    print("Descartados:", dict(descartados))
    print("Relatório:", RELATORIO_SAIDA)
    print("Amostras:", AMOSTRAS_SAIDA)
    print("Status: PENDENTE DE REVISÃO")
    print("LIMPEZA CONCLUÍDA")


if __name__ == "__main__":
    main()
