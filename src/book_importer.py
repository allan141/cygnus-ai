
# Cygnus AI - Importador de livros
# Fase 15 - Parte 4

import argparse
import hashlib
import json
import re
from pathlib import Path


def limpar_gutenberg(texto):
    texto = texto.replace("\r\n", "\n")
    texto = texto.replace("\r", "\n")
    texto = texto.lstrip("\ufeff")

    inicio = re.search(
        r"^\s*\*{3}\s*START OF (?:THE|THIS) "
        r"PROJECT GUTENBERG EBOOK.*?\*{3}\s*$",
        texto,
        flags=re.IGNORECASE | re.MULTILINE
    )

    fim = re.search(
        r"^\s*\*{3}\s*END OF (?:THE|THIS) "
        r"PROJECT GUTENBERG EBOOK.*?\*{3}\s*$",
        texto,
        flags=re.IGNORECASE | re.MULTILINE
    )

    if inicio and fim and fim.start() > inicio.end():
        texto = texto[inicio.end():fim.start()]

    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)

    return texto.strip()


def carregar_catalogo(caminho):
    dados = json.loads(
        Path(caminho).read_text(encoding="utf-8")
    )

    if not isinstance(dados, list):
        raise ValueError(
            "O catalogo deve conter uma lista."
        )

    return {
        int(livro["id"]): livro
        for livro in dados
    }


def importar_livro(
    identificador,
    catalogo,
    downloads,
    destino,
    autorizado=False
):
    if not autorizado:
        raise PermissionError(
            "Confirme primeiro os direitos de uso "
            "da obra, edicao e traducao."
        )

    if identificador not in catalogo:
        raise ValueError(
            f"Livro {identificador} nao esta no catalogo."
        )

    origem = Path(downloads) / f"{identificador}.txt"

    if not origem.is_file():
        raise FileNotFoundError(
            f"Arquivo nao encontrado: {origem}"
        )

    texto_original = origem.read_text(
        encoding="utf-8-sig"
    )

    texto = limpar_gutenberg(texto_original)

    if len(texto) < 100:
        raise ValueError(
            "Texto muito curto para importacao."
        )

    pasta = Path(destino)
    pasta.mkdir(parents=True, exist_ok=True)

    caminho = pasta / f"livro_{identificador}.txt"

    if caminho.exists():
        raise FileExistsError(
            f"Livro ja importado: {caminho}"
        )

    caminho.write_text(
        texto,
        encoding="utf-8"
    )

    livro = catalogo[identificador]

    return {
        "id": identificador,
        "titulo": livro["titulo"],
        "pagina": livro["pagina"],
        "origem_local": str(origem),
        "arquivo_processado": str(caminho),
        "sha256": hashlib.sha256(
            texto.encode("utf-8")
        ).hexdigest(),
        "caracteres": len(texto),
        "palavras_aproximadas": len(texto.split()),
        "direitos": "verificados_pelo_usuario"
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--id",
        type=int,
        required=True
    )

    parser.add_argument(
        "--catalogo",
        default="data/catalogo_livros.json"
    )

    parser.add_argument(
        "--downloads",
        default="data/downloads"
    )

    parser.add_argument(
        "--destino",
        default="data/raw/livros"
    )

    parser.add_argument(
        "--direitos-verificados",
        action="store_true"
    )

    args = parser.parse_args()

    catalogo = carregar_catalogo(args.catalogo)

    resultado = importar_livro(
        identificador=args.id,
        catalogo=catalogo,
        downloads=args.downloads,
        destino=args.destino,
        autorizado=args.direitos_verificados
    )

    # Cada importacao recebe seu proprio registro.
    registro = (
        Path(args.destino)
        / f"livro_{args.id}.metadata.json"
    )

    registro.write_text(
        json.dumps(
            resultado,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print("\n=== CYGNUS AI - IMPORTADOR ===")
    print(f"Livro: {resultado['titulo']}")
    print(f"Caracteres: {resultado['caracteres']}")
    print(
        "Palavras aproximadas: "
        f"{resultado['palavras_aproximadas']}"
    )
    print(
        f"Arquivo: {resultado['arquivo_processado']}"
    )
    print("Importacao concluida.")


if __name__ == "__main__":
    main()
