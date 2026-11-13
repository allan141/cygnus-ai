
# Cygnus AI - Fase 15, Parte 5
# Analise de tokens do corpus v3

import argparse
import json
from pathlib import Path

from src.byte_bpe_tokenizer import CygnusByteBPETokenizer


def carregar_documentos(diretorio):
    pasta = Path(diretorio)

    if not pasta.is_dir():
        raise FileNotFoundError(
            f"Diretorio nao encontrado: {pasta}"
        )

    arquivos = sorted(pasta.rglob("*.txt"))

    if not arquivos:
        raise ValueError(
            f"Nenhum documento TXT encontrado: {pasta}"
        )

    documentos = []

    for arquivo in arquivos:
        texto = arquivo.read_text(encoding="utf-8")

        if texto.strip():
            documentos.append({
                "arquivo": str(arquivo),
                "texto": texto
            })

    if not documentos:
        raise ValueError("Todos os documentos estao vazios")

    return documentos


def analisar_documentos(documentos, tokenizador, contexto=128):
    if contexto < 1:
        raise ValueError("Contexto deve ser positivo")

    resultados = []
    total_tokens = 0
    total_caracteres = 0
    total_bytes = 0
    total_janelas = 0

    for documento in documentos:
        texto = documento["texto"]
        tokens = tokenizador.codificar(texto)

        caracteres = len(texto)
        bytes_utf8 = len(texto.encode("utf-8"))
        quantidade_tokens = len(tokens)

        # Janelas completas, sem atravessar documentos.
        # Cada janela precisa de contexto + 1 tokens
        # para construir entrada e alvo deslocado.
        janelas = max(
            0,
            (quantidade_tokens - 1) // contexto
        )

        resultados.append({
            "arquivo": documento["arquivo"],
            "caracteres": caracteres,
            "bytes_utf8": bytes_utf8,
            "tokens": quantidade_tokens,
            "janelas_completas": janelas
        })

        total_tokens += quantidade_tokens
        total_caracteres += caracteres
        total_bytes += bytes_utf8
        total_janelas += janelas

    return {
        "documentos": len(resultados),
        "caracteres": total_caracteres,
        "bytes_utf8": total_bytes,
        "tokens": total_tokens,
        "caracteres_por_token": round(
            total_caracteres / total_tokens, 4
        ) if total_tokens else 0,
        "bytes_por_token": round(
            total_bytes / total_tokens, 4
        ) if total_tokens else 0,
        "janelas_completas": total_janelas,
        "arquivos": resultados
    }


def executar_analise(
    origem="data/v3_livros",
    destino="checkpoints/cygnus_v3",
    merges=256,
    contexto=128
):
    if merges < 0:
        raise ValueError("Merges nao pode ser negativo")

    if contexto < 1:
        raise ValueError("Contexto deve ser positivo")

    origem = Path(origem)
    destino = Path(destino)

    documentos_treino = carregar_documentos(
        origem / "train"
    )
    documentos_validacao = carregar_documentos(
        origem / "validation"
    )

    tokenizador = CygnusByteBPETokenizer()

    tokenizador.treinar(
        [doc["texto"] for doc in documentos_treino],
        numero_merges=merges
    )

    treino = analisar_documentos(
        documentos_treino,
        tokenizador,
        contexto
    )
    validacao = analisar_documentos(
        documentos_validacao,
        tokenizador,
        contexto
    )

    relatorio = {
        "modelo": "Cygnus AI v3",
        "tokenizador": "Byte-Level BPE",
        "merges_solicitados": merges,
        "merges_aprendidos": len(tokenizador.merges),
        "tamanho_vocabulario": (
            tokenizador.tamanho_vocabulario
        ),
        "contexto": contexto,
        "treinamento": treino,
        "validacao": validacao,
        "total_tokens": (
            treino["tokens"] + validacao["tokens"]
        )
    }

    caminho_tokenizador = destino / "tokenizador.json"
    caminho_relatorio = destino / "corpus_relatorio.json"

    # Evitar sobrescrever artefatos existentes.
    if caminho_tokenizador.exists():
        raise FileExistsError(
            f"Tokenizador ja existe: {caminho_tokenizador}"
        )

    if caminho_relatorio.exists():
        raise FileExistsError(
            f"Relatorio ja existe: {caminho_relatorio}"
        )

    destino.mkdir(parents=True, exist_ok=True)

    tokenizador.salvar(caminho_tokenizador)

    caminho_relatorio.write_text(
        json.dumps(
            relatorio,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    return relatorio


def main():
    parser = argparse.ArgumentParser(
        description="Analisador de corpus Cygnus v3"
    )

    parser.add_argument(
        "--origem",
        default="data/v3_livros"
    )
    parser.add_argument(
        "--destino",
        default="checkpoints/cygnus_v3"
    )
    parser.add_argument(
        "--merges",
        type=int,
        default=256
    )
    parser.add_argument(
        "--contexto",
        type=int,
        default=128
    )

    args = parser.parse_args()

    relatorio = executar_analise(
        origem=args.origem,
        destino=args.destino,
        merges=args.merges,
        contexto=args.contexto
    )

    print("\n=== CYGNUS AI - ANALISE DO CORPUS V3 ===")
    print("Vocabulario:", relatorio["tamanho_vocabulario"])
    print("Merges aprendidos:", relatorio["merges_aprendidos"])
    print("Contexto:", relatorio["contexto"])
    print("Tokens treino:", relatorio["treinamento"]["tokens"])
    print("Tokens validacao:", relatorio["validacao"]["tokens"])
    print("Tokens totais:", relatorio["total_tokens"])
    print(
        "Janelas completas treino:",
        relatorio["treinamento"]["janelas_completas"]
    )
    print(
        "Janelas completas validacao:",
        relatorio["validacao"]["janelas_completas"]
    )
    print("Analise concluida.")


if __name__ == "__main__":
    main()
