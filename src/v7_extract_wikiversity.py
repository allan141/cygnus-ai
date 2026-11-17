"""Cygnus v7 — extrator de páginas de um dump MediaWiki."""

import argparse
import bz2
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import mwparserfromhell


def limpar_wikitexto(texto):
    codigo = mwparserfromhell.parse(texto)
    limpo = codigo.strip_code(normalize=True, collapse=True)
    limpo = re.sub(r"\n{3,}", "\n\n", limpo)
    return limpo.strip()


def extrair(dump, destino, minimo):
    if destino.exists():
        raise FileExistsError(
            f"Destino já existe: {destino}. Escolha outro nome."
        )

    destino.mkdir(parents=True)
    documentos = destino / "documentos"
    documentos.mkdir()

    total_paginas = 0
    aceitos = 0
    curtos = 0
    duplicados = 0
    hashes = set()

    manifesto = destino / "manifesto.jsonl"

    with bz2.open(dump, "rb") as arquivo, manifesto.open(
        "w", encoding="utf-8"
    ) as saida:
        contexto = ET.iterparse(
            arquivo, events=("start", "end")
        )

        _, raiz = next(contexto)

        for evento, elemento in contexto:
            if evento != "end" or not elemento.tag.endswith("}page"):
                continue

            total_paginas += 1

            namespace = elemento.findtext("./{*}ns")
            titulo = elemento.findtext("./{*}title", default="")
            pagina_id = elemento.findtext("./{*}id", default="")
            texto = elemento.findtext(
                "./{*}revision/{*}text", default=""
            )

            if namespace != "0":
                elemento.clear()
                raiz.clear()
                continue

            limpo = limpar_wikitexto(texto)

            if len(limpo) < minimo:
                curtos += 1
                elemento.clear()
                raiz.clear()
                continue

            digest = hashlib.sha256(
                limpo.encode("utf-8")
            ).hexdigest()

            if digest in hashes:
                duplicados += 1
                elemento.clear()
                raiz.clear()
                continue

            hashes.add(digest)
            aceitos += 1

            nome = f"{aceitos:06d}_{digest[:12]}.txt"

            (documentos / nome).write_text(
                limpo, encoding="utf-8"
            )

            registro = {
                "arquivo": nome,
                "titulo": titulo,
                "pagina_id": pagina_id,
                "url": (
                    "https://pt.wikiversity.org/wiki/"
                    + titulo.replace(" ", "_")
                ),
                "fonte": "Wikiversidade em português",
                "licenca_declarada": "verificar por página",
                "licenca_verificada": False,
                "status": "pendente_revisao",
                "caracteres": len(limpo),
                "sha256": digest,
            }

            saida.write(
                json.dumps(registro, ensure_ascii=False) + "\n"
            )

            elemento.clear()
            raiz.clear()

            if aceitos % 500 == 0:
                print(
                    f"Páginas: {total_paginas} | "
                    f"Documentos extraídos: {aceitos}",
                    flush=True,
                )

    print("\n=== CYGNUS V7 | EXTRAÇÃO WIKIVERSIDADE ===")
    print("Páginas examinadas:", total_paginas)
    print("Documentos extraídos:", aceitos)
    print("Textos curtos:", curtos)
    print("Duplicatas exatas:", duplicados)
    print("Destino:", destino)
    print("Status: PENDENTE DE REVISÃO")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dump", required=True)
    parser.add_argument("--saida", required=True)
    parser.add_argument("--min-caracteres", type=int, default=500)

    args = parser.parse_args()

    dump = Path(args.dump)
    destino = Path(args.saida)

    if not dump.is_file():
        raise FileNotFoundError(dump)

    if args.min_caracteres < 1:
        raise ValueError("O mínimo de caracteres deve ser positivo.")

    extrair(dump, destino, args.min_caracteres)


if __name__ == "__main__":
    main()
