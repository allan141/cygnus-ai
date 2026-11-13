
# Cygnus AI - Fase 16.1
# Coleta controlada de textos da Wikipedia em portugues

import argparse
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


API_URL = "https://pt.wikipedia.org/w/api.php"

# Identificacao obrigatoria para uso da API Wikimedia.
# Atualize a URL caso publique o projeto em outro repositorio.
USER_AGENT = (
    "CygnusAI-CorpusCollector/0.1 "
    "(https://github.com/allan141/cygnus-ai; "
    "educational research)"
)

TITULOS = [
    "Inteligência artificial",
    "Aprendizado de máquina",
    "Rede neural artificial",
    "Processamento de linguagem natural",
    "Linguagem de programação",
    "Python",
    "Algoritmo",
    "Banco de dados",
    "Sistema operacional",
    "Internet",
    "Computador",
    "Matemática",
    "Física",
    "Química",
    "Biologia",
    "Astronomia",
    "Sistema Solar",
    "Terra",
    "História do Brasil",
    "Geografia do Brasil",
]


def consultar_api(parametros, tentativas=3):
    consulta = urllib.parse.urlencode(parametros)

    requisicao = urllib.request.Request(
        f"{API_URL}?{consulta}",
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        },
    )

    for tentativa in range(tentativas):
        try:
            with urllib.request.urlopen(
                requisicao,
                timeout=30,
            ) as resposta:
                dados = json.load(resposta)

            if "error" in dados:
                raise RuntimeError(
                    f"Erro da API: {dados['error']}"
                )

            return dados

        except urllib.error.HTTPError as erro:
            # Repetir somente erros transitorios.
            if erro.code not in (429, 500, 502, 503, 504):
                raise

            if tentativa == tentativas - 1:
                raise

        except (
            urllib.error.URLError,
            TimeoutError,
        ):
            if tentativa == tentativas - 1:
                raise

        time.sleep(2 ** tentativa)

    raise RuntimeError("Falha ao consultar a API")


def buscar_artigo(titulo):
    dados = consultar_api({
        "action": "query",
        "prop": "extracts|info",
        "titles": titulo,
        "explaintext": "1",
        "exsectionformat": "plain",
        "inprop": "url",
        "format": "json",
        "formatversion": "2",
    })

    paginas = dados.get(
        "query", {}
    ).get("pages", [])

    if not paginas:
        return None

    pagina = paginas[0]

    if pagina.get("missing"):
        return None

    if pagina.get("ns") != 0:
        return None

    return {
        "id": pagina["pageid"],
        "titulo": pagina["title"],
        "url": pagina.get("fullurl"),
        "texto": pagina.get("extract", ""),
    }


def limpar_texto(texto):
    if not isinstance(texto, str):
        raise TypeError("Texto deve ser string")

    texto = texto.replace("\r\n", "\n")
    texto = texto.replace("\r", "\n")
    texto = texto.replace("\xa0", " ")

    linhas = []

    for linha in texto.splitlines():
        linha = re.sub(r"[ \t]+", " ", linha).strip()

        if linha:
            linhas.append(linha)

    return "\n".join(linhas).strip()


def identificar_texto(texto):
    normalizado = " ".join(texto.casefold().split())

    return hashlib.sha256(
        normalizado.encode("utf-8")
    ).hexdigest()


def coletar_artigos(
    destino="data/wikipedia",
    limite=20,
    minimo_caracteres=500,
    pausa=1.0,
):
    if limite <= 0:
        raise ValueError("Limite deve ser positivo")

    if minimo_caracteres <= 0:
        raise ValueError(
            "Minimo de caracteres deve ser positivo"
        )

    if pausa < 0:
        raise ValueError("Pausa nao pode ser negativa")

    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)

    hashes = set()
    ids = set()

    # Considerar arquivos ja existentes para evitar
    # duplicacao entre execucoes.
    for arquivo in destino.glob("artigo_*.txt"):
        texto = arquivo.read_text(encoding="utf-8")
        hashes.add(identificar_texto(texto))

        try:
            ids.add(int(arquivo.stem.split("_")[-1]))
        except ValueError:
            pass

    coletados = 0
    ignorados = 0
    caracteres = 0

    for titulo in TITULOS[:limite]:
        print(f"Consultando: {titulo}")

        artigo = buscar_artigo(titulo)

        if artigo is None:
            ignorados += 1
            continue

        texto = limpar_texto(artigo["texto"])

        if len(texto) < minimo_caracteres:
            print("  Ignorado: texto muito curto")
            ignorados += 1
            continue

        identificador = identificar_texto(texto)

        if (
            identificador in hashes
            or artigo["id"] in ids
        ):
            print("  Ignorado: duplicado")
            ignorados += 1
            continue

        caminho = destino / (
            f"artigo_{artigo['id']}.txt"
        )

        caminho_meta = destino / (
            f"artigo_{artigo['id']}.metadata.json"
        )

        if caminho.exists() or caminho_meta.exists():
            print("  Ignorado: arquivo existente")
            ignorados += 1
            continue

        metadados = {
            "titulo": artigo["titulo"],
            "pageid": artigo["id"],
            "url": artigo["url"],
            "fonte": "Wikipedia em portugues",
            "licenca": (
                "Verificar licenca e atribuicao "
                "na pagina original"
            ),
            "sha256": identificador,
            "caracteres": len(texto),
            "palavras": len(texto.split()),
        }

        caminho.write_text(
            texto + "\n",
            encoding="utf-8",
        )

        caminho_meta.write_text(
            json.dumps(
                metadados,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        hashes.add(identificador)
        ids.add(artigo["id"])

        coletados += 1
        caracteres += len(texto)

        print(
            f"  Salvo: {len(texto)} caracteres"
        )

        if pausa:
            time.sleep(pausa)

    return {
        "coletados": coletados,
        "ignorados": ignorados,
        "caracteres_novos": caracteres,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Coletor Wikipedia - Cygnus AI"
    )

    parser.add_argument(
        "--destino",
        default="data/wikipedia",
    )

    parser.add_argument(
        "--limite",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--min-caracteres",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--pausa",
        type=float,
        default=1.0,
    )

    args = parser.parse_args()

    resultado = coletar_artigos(
        destino=args.destino,
        limite=args.limite,
        minimo_caracteres=args.min_caracteres,
        pausa=args.pausa,
    )

    print("\n=== CYGNUS AI - WIKIPEDIA ===")
    print("Artigos salvos:", resultado["coletados"])
    print("Ignorados:", resultado["ignorados"])
    print(
        "Caracteres novos:",
        resultado["caracteres_novos"],
    )


if __name__ == "__main__":
    main()
