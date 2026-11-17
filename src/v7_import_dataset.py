"""Cygnus v7 - importador de datasets TXT e JSONL."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from src.byte_bpe_tokenizer import CygnusByteBPETokenizer


TOKENIZADOR = Path(
    "checkpoints/cygnus_v4_tokenizers/tokenizador_2048.json"
)

DESTINO_BASE = Path("data/v7_importados")


def normalizar(texto):
    texto = texto.replace("\r\n", "\n").replace("\r", "\n")
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


def assinatura(texto):
    return hashlib.sha256(
        texto.encode("utf-8")
    ).hexdigest()


def carregar_documentos(entrada, campo_texto):
    if entrada.is_file():
        arquivos = [entrada]
    elif entrada.is_dir():
        arquivos = sorted(
            p for p in entrada.rglob("*")
            if p.is_file() and p.suffix.lower() in (".txt", ".jsonl")
        )
    else:
        raise FileNotFoundError(entrada)

    for arquivo in arquivos:
        if arquivo.suffix.lower() == ".txt":
            yield {
                "texto": arquivo.read_text(encoding="utf-8"),
                "origem": str(arquivo),
                "linha": None,
            }

        elif arquivo.suffix.lower() == ".jsonl":
            with arquivo.open(encoding="utf-8") as f:
                for numero, linha in enumerate(f, 1):
                    if not linha.strip():
                        continue

                    try:
                        objeto = json.loads(linha)
                    except json.JSONDecodeError as erro:
                        raise ValueError(
                            f"JSON inválido: {arquivo}, linha {numero}"
                        ) from erro

                    if not isinstance(objeto, dict):
                        raise ValueError(
                            f"Esperado objeto JSON: {arquivo}, linha {numero}"
                        )

                    texto = objeto.get(campo_texto)

                    if not isinstance(texto, str):
                        raise ValueError(
                            f"Campo '{campo_texto}' ausente ou inválido: "
                            f"{arquivo}, linha {numero}"
                        )

                    yield {
                        "texto": texto,
                        "origem": str(arquivo),
                        "linha": numero,
                    }


def main():
    parser = argparse.ArgumentParser(
        description="Importador de datasets do Cygnus v7"
    )

    parser.add_argument("--entrada", required=True)
    parser.add_argument("--nome", required=True)
    parser.add_argument("--fonte", required=True)
    parser.add_argument("--licenca", required=True)
    parser.add_argument("--campo-texto", default="text")
    parser.add_argument("--min-caracteres", type=int, default=500)
    parser.add_argument("--min-tokens", type=int, default=100)

    args = parser.parse_args()

    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.nome):
        raise ValueError(
            "--nome deve conter apenas letras, números, _ ou -"
        )

    if args.min_caracteres < 1 or args.min_tokens < 1:
        raise ValueError("Os limites mínimos devem ser positivos.")

    entrada = Path(args.entrada)
    destino = DESTINO_BASE / args.nome

    if destino.exists():
        raise FileExistsError(
            f"Destino já existe: {destino}. "
            "Escolha outro nome para preservar os dados anteriores."
        )

    if not TOKENIZADOR.exists():
        raise FileNotFoundError(TOKENIZADOR)

    tokenizer = CygnusByteBPETokenizer.carregar(TOKENIZADOR)

    # Evita duplicatas exatas contra o corpus v7 inicial.
    hashes_existentes = set()
    corpus_inicial = Path("data/v7_corpus_inicial/train")

    if corpus_inicial.exists():
        for arquivo in corpus_inicial.rglob("*.txt"):
            texto = normalizar(
                arquivo.read_text(encoding="utf-8")
            )
            hashes_existentes.add(assinatura(texto))

    registros = []
    rejeitados = {
        "curto": 0,
        "duplicado": 0,
    }

    # Primeiro valida e prepara os registros em memória.
    for documento in carregar_documentos(
        entrada, args.campo_texto
    ):
        texto = normalizar(documento["texto"])

        if len(texto) < args.min_caracteres:
            rejeitados["curto"] += 1
            continue

        digest = assinatura(texto)

        if digest in hashes_existentes:
            rejeitados["duplicado"] += 1
            continue

        tokens = len(tokenizer.codificar(texto))

        if tokens < args.min_tokens:
            rejeitados["curto"] += 1
            continue

        hashes_existentes.add(digest)

        registros.append({
            "texto": texto,
            "sha256": digest,
            "tokens": tokens,
            "caracteres": len(texto),
            "origem": documento["origem"],
            "linha": documento["linha"],
        })

    destino.mkdir(parents=True, exist_ok=False)
    pasta_documentos = destino / "documentos"
    pasta_documentos.mkdir()

    manifesto = []

    for indice, registro in enumerate(registros, 1):
        nome_arquivo = f"{indice:06d}_{registro['sha256'][:12]}.txt"

        (pasta_documentos / nome_arquivo).write_text(
            registro["texto"],
            encoding="utf-8",
        )

        manifesto.append({
            "arquivo": nome_arquivo,
            "sha256": registro["sha256"],
            "tokens": registro["tokens"],
            "caracteres": registro["caracteres"],
            "origem": registro["origem"],
            "linha": registro["linha"],
            "fonte": args.fonte,
            "licenca_declarada": args.licenca,
            "licenca_verificada": False,
            "status": "pendente_revisao",
        })

    relatorio = {
        "versao": "cygnus-v7-importador-20.4",
        "dataset": args.nome,
        "fonte": args.fonte,
        "licenca_declarada": args.licenca,
        "documentos_importados": len(manifesto),
        "tokens_importados": sum(x["tokens"] for x in manifesto),
        "rejeitados": rejeitados,
        "documentos": manifesto,
    }

    (destino / "manifesto.json").write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=== CYGNUS V7 | IMPORTAÇÃO DE DATASET ===")
    print("Dataset:", args.nome)
    print("Documentos importados:", len(manifesto))
    print("Tokens importados:", relatorio["tokens_importados"])
    print("Rejeitados:", rejeitados)
    print("Destino:", destino)
    print("Status: PENDENTE DE REVISÃO")
    print("IMPORTAÇÃO CONCLUÍDA")


if __name__ == "__main__":
    main()
