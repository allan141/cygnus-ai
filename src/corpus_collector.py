
# Cygnus AI - Coletor de corpus em portugues
# Fase 15 - Parte 3

import argparse
import json
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


API_URL = "https://gutendex.com/books/"


def consultar_catalogo(
    idioma="pt",
    pagina=1,
    termo=None,
    tentativas=3,
    timeout=60
):
    parametros = {
        "languages": idioma,
        "copyright": "false",
        "page": pagina
    }

    if termo:
        parametros["search"] = termo

    url = API_URL + "?" + urllib.parse.urlencode(
        parametros
    )

    requisicao = urllib.request.Request(
        url,
        headers={
            "User-Agent": "CygnusAI-CorpusResearch/0.1",
            "Accept": "application/json"
        }
    )

    for tentativa in range(1, tentativas + 1):
        try:
            print(
                f"Tentativa {tentativa}/{tentativas} "
                "- aguardando resposta..."
            )

            with urllib.request.urlopen(
                requisicao,
                timeout=timeout
            ) as resposta:
                dados = json.load(resposta)

            print("Resposta recebida com sucesso.")
            return dados

        except (
            TimeoutError,
            socket.timeout,
            urllib.error.URLError
        ) as erro:
            print(f"Falha de conexao: {erro}")

            if tentativa == tentativas:
                raise ConnectionError(
                    "Nao foi possivel consultar o Gutendex "
                    f"apos {tentativas} tentativas."
                ) from erro

            espera = 2 ** tentativa

            print(
                f"Nova tentativa em {espera} segundos..."
            )

            time.sleep(espera)


def selecionar_livros(dados):
    livros = []

    for livro in dados.get("results", []):
        if "pt" not in livro.get("languages", []):
            continue

        if livro.get("copyright") is not False:
            continue

        autores = [
            {
                "nome": autor.get("name"),
                "nascimento": autor.get("birth_year"),
                "falecimento": autor.get("death_year")
            }
            for autor in livro.get("authors", [])
        ]

        formatos_texto = {
            formato: url
            for formato, url in livro.get(
                "formats", {}
            ).items()
            if formato.startswith("text/plain")
        }

        livros.append({
            "id": livro["id"],
            "titulo": livro["title"],
            "autores": autores,
            "idioma": "pt",
            "pagina": (
                "https://www.gutenberg.org/ebooks/"
                f"{livro['id']}"
            ),
            "formatos_texto": formatos_texto,
            "direitos": "revisao_pendente"
        })

    return livros


def pesquisar_livros(
    paginas=1,
    termo=None
):
    if paginas < 1:
        raise ValueError(
            "O numero de paginas deve ser positivo."
        )

    encontrados = {}

    for pagina in range(1, paginas + 1):
        print(f"Consultando pagina {pagina}...")

        dados = consultar_catalogo(
            pagina=pagina,
            termo=termo
        )

        for livro in selecionar_livros(dados):
            encontrados[livro["id"]] = livro

        if not dados.get("next"):
            break

    return list(encontrados.values())


def salvar_catalogo(
    livros,
    destino="data/catalogo_livros.json"
):
    caminho = Path(destino)

    caminho.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    caminho.write_text(
        json.dumps(
            livros,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    return caminho


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--paginas",
        type=int,
        default=1
    )

    parser.add_argument(
        "--termo",
        default=None
    )

    parser.add_argument(
        "--destino",
        default="data/catalogo_livros.json"
    )

    args = parser.parse_args()

    livros = pesquisar_livros(
        paginas=args.paginas,
        termo=args.termo
    )

    caminho = salvar_catalogo(
        livros,
        args.destino
    )

    print("\n=== CYGNUS AI - COLETOR ===")
    print(f"Livros encontrados: {len(livros)}")

    for livro in livros[:10]:
        print(
            f"[{livro['id']}] "
            f"{livro['titulo']}"
        )

    print(f"\nCatalogo salvo em: {caminho}")

    print(
        "Aviso: as obras ainda precisam "
        "de verificacao de direitos."
    )


if __name__ == "__main__":
    main()
