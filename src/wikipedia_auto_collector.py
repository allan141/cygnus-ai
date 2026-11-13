
# Cygnus AI - Fase 16.2
# Expansao automatica do corpus da Wikipedia

import argparse
import hashlib
import json
import time
from pathlib import Path

from src.wikipedia_collector import (
    consultar_api,
    limpar_texto,
    identificar_texto,
)
from src.byte_bpe_tokenizer import CygnusByteBPETokenizer


def descobrir_artigos(quantidade=10):
    if not 1 <= quantidade <= 20:
        raise ValueError(
            "Quantidade deve estar entre 1 e 20"
        )

    dados = consultar_api({
        "action": "query",
        "generator": "random",
        "grnnamespace": "0",
        "grnlimit": str(quantidade),
        "prop": "extracts|info",
        "explaintext": "1",
        "exsectionformat": "plain",
        "inprop": "url",
        "format": "json",
        "formatversion": "2",
        "maxlag": "5",
    })

    artigos = []

    for pagina in dados.get("query", {}).get("pages", []):
        if (
            pagina.get("missing")
            or pagina.get("ns") != 0
            or not pagina.get("extract")
        ):
            continue

        artigos.append({
            "id": pagina["pageid"],
            "titulo": pagina["title"],
            "url": pagina.get("fullurl"),
            "texto": pagina["extract"],
        })

    return artigos


def carregar_existentes(destino):
    hashes = set()
    ids = set()

    for caminho in destino.glob("artigo_*.txt"):
        try:
            pageid = int(caminho.stem.split("_")[-1])
            texto = caminho.read_text(encoding="utf-8")
        except (ValueError, UnicodeError, OSError):
            continue

        ids.add(pageid)
        hashes.add(identificar_texto(texto))

    return hashes, ids


def contar_tokens(destino, tokenizador):
    total = 0

    for caminho in sorted(destino.glob("artigo_*.txt")):
        texto = caminho.read_text(encoding="utf-8")
        total += len(tokenizador.codificar(texto))

    return total


def salvar_artigo(artigo, texto, destino, tokens):
    pageid = artigo["id"]

    caminho = destino / f"artigo_{pageid}.txt"
    caminho_meta = destino / f"artigo_{pageid}.metadata.json"

    if caminho.exists() or caminho_meta.exists():
        return False

    conteudo = texto + "\n"

    metadados = {
        "titulo": artigo["titulo"],
        "pageid": pageid,
        "url": artigo["url"],
        "fonte": "Wikipedia em portugues",
        "licenca": (
            "Verificar licenca, historico de autores "
            "e requisitos de atribuicao na fonte"
        ),
        "sha256": hashlib.sha256(
            conteudo.encode("utf-8")
        ).hexdigest(),
        "caracteres": len(texto),
        "palavras": len(texto.split()),
        "tokens_bpe_v3": tokens,
    }

    # Se houver erro ao salvar metadados, remover
    # o texto criado nesta tentativa.
    try:
        with caminho.open("x", encoding="utf-8") as arquivo:
            arquivo.write(conteudo)

        with caminho_meta.open("x", encoding="utf-8") as arquivo:
            json.dump(
                metadados,
                arquivo,
                ensure_ascii=False,
                indent=2,
            )

    except Exception:
        if caminho.exists() and not caminho_meta.exists():
            caminho.unlink()
        raise

    return True


def expandir_corpus(
    destino="data/wikipedia",
    caminho_tokenizador=(
        "checkpoints/cygnus_v3/tokenizador.json"
    ),
    meta_tokens=1_000_000,
    max_rodadas=100,
    minimo_caracteres=1000,
    maximo_caracteres=150000,
    pausa=1.5,
):
    if meta_tokens <= 0 or max_rodadas <= 0:
        raise ValueError("Meta e rodadas devem ser positivas")

    if (
        minimo_caracteres <= 0
        or maximo_caracteres < minimo_caracteres
        or pausa < 0
    ):
        raise ValueError("Parametros de coleta invalidos")

    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)

    tokenizador = CygnusByteBPETokenizer.carregar(
        caminho_tokenizador
    )

    hashes, ids = carregar_existentes(destino)

    total_tokens = contar_tokens(destino, tokenizador)
    total_inicial = total_tokens

    salvos = 0
    ignorados = 0
    falhas = 0

    print("\n=== CYGNUS AI - EXPANSAO V4 ===")
    print("Tokens iniciais:", total_tokens)
    print("Meta:", meta_tokens)

    for rodada in range(1, max_rodadas + 1):
        if total_tokens >= meta_tokens:
            break

        print(f"\nRodada {rodada}/{max_rodadas}")

        try:
            artigos = descobrir_artigos(10)
        except Exception as erro:
            falhas += 1
            print("Erro na consulta:", erro)
            time.sleep(max(pausa, 2))
            continue

        for artigo in artigos:
            if total_tokens >= meta_tokens:
                break

            texto = limpar_texto(artigo["texto"])
            tamanho = len(texto)

            if not (
                minimo_caracteres
                <= tamanho
                <= maximo_caracteres
            ):
                ignorados += 1
                continue

            identificador = identificar_texto(texto)

            if (
                artigo["id"] in ids
                or identificador in hashes
            ):
                ignorados += 1
                continue

            tokens = len(tokenizador.codificar(texto))

            if tokens == 0:
                ignorados += 1
                continue

            if not salvar_artigo(
                artigo, texto, destino, tokens
            ):
                ignorados += 1
                continue

            ids.add(artigo["id"])
            hashes.add(identificador)

            salvos += 1
            total_tokens += tokens

            print(
                f"  {artigo['titulo'][:55]} "
                f"| +{tokens} tokens "
                f"| total: {total_tokens}"
            )

        if total_tokens < meta_tokens and pausa:
            time.sleep(pausa)

    resultado = {
        "tokens_iniciais": total_inicial,
        "tokens_finais": total_tokens,
        "tokens_adicionados": total_tokens - total_inicial,
        "meta_tokens": meta_tokens,
        "meta_atingida": total_tokens >= meta_tokens,
        "artigos_novos": salvos,
        "ignorados": ignorados,
        "falhas_consulta": falhas,
    }

    print("\n=== RESULTADO ===")
    for chave, valor in resultado.items():
        print(f"{chave}: {valor}")

    return resultado


def main():
    parser = argparse.ArgumentParser(
        description="Expansao automatica do corpus Cygnus"
    )

    parser.add_argument(
        "--destino",
        default="data/wikipedia",
    )
    parser.add_argument(
        "--tokenizador",
        default="checkpoints/cygnus_v3/tokenizador.json",
    )
    parser.add_argument(
        "--meta-tokens",
        type=int,
        default=1_000_000,
    )
    parser.add_argument(
        "--max-rodadas",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--min-caracteres",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--max-caracteres",
        type=int,
        default=150000,
    )
    parser.add_argument(
        "--pausa",
        type=float,
        default=1.5,
    )

    args = parser.parse_args()

    expandir_corpus(
        destino=args.destino,
        caminho_tokenizador=args.tokenizador,
        meta_tokens=args.meta_tokens,
        max_rodadas=args.max_rodadas,
        minimo_caracteres=args.min_caracteres,
        maximo_caracteres=args.max_caracteres,
        pausa=args.pausa,
    )


if __name__ == "__main__":
    main()
