
# Cygnus AI - Fase 15, Parte 2
# Pipeline de preparacao de corpus

import argparse
import hashlib
import json
import random
import re
import shutil
from pathlib import Path


def normalizar_texto(texto):
    # Normaliza quebras de linha e espacos,
    # preservando acentos e pontuacao.
    texto = texto.replace("\r\n", "\n")
    texto = texto.replace("\r", "\n")
    texto = texto.replace("\u00a0", " ")

    linhas = [
        re.sub(r"[ \t]+", " ", linha).strip()
        for linha in texto.split("\n")
    ]

    texto = "\n".join(linhas)
    texto = re.sub(r"\n{3,}", "\n\n", texto)

    return texto.strip()


def hash_documento(texto):
    return hashlib.sha256(
        texto.encode("utf-8")
    ).hexdigest()


def carregar_corpus(diretorio):
    diretorio = Path(diretorio)

    if not diretorio.is_dir():
        raise FileNotFoundError(
            f"Diretorio nao encontrado: {diretorio}"
        )

    documentos = []
    hashes = set()

    arquivos = sorted(
        diretorio.rglob("*.txt")
    )

    for arquivo in arquivos:
        texto = arquivo.read_text(
            encoding="utf-8"
        )

        texto = normalizar_texto(texto)

        if not texto:
            continue

        identificador = hash_documento(texto)

        if identificador in hashes:
            continue

        hashes.add(identificador)

        documentos.append({
            "origem": str(arquivo),
            "texto": texto,
            "hash": identificador
        })

    return documentos


def dividir_documentos(
    documentos,
    proporcao_treino=0.9,
    semente=42
):
    if not 0 < proporcao_treino < 1:
        raise ValueError(
            "Proporcao de treino deve estar entre 0 e 1"
        )

    if len(documentos) < 2:
        raise ValueError(
            "Sao necessarios pelo menos dois documentos"
        )

    documentos = list(documentos)

    gerador = random.Random(semente)
    gerador.shuffle(documentos)

    quantidade_treino = round(
        len(documentos) * proporcao_treino
    )

    quantidade_treino = max(
        1,
        min(
            quantidade_treino,
            len(documentos) - 1
        )
    )

    treino = documentos[:quantidade_treino]
    validacao = documentos[quantidade_treino:]

    return treino, validacao


def salvar_documentos(documentos, diretorio):
    diretorio = Path(diretorio)
    diretorio.mkdir(parents=True, exist_ok=True)

    for indice, documento in enumerate(documentos):
        nome = f"documento_{indice:06d}.txt"

        caminho = diretorio / nome

        caminho.write_text(
            documento["texto"],
            encoding="utf-8"
        )


def estatisticas_documentos(documentos):
    return {
        "documentos": len(documentos),
        "caracteres": sum(
            len(doc["texto"])
            for doc in documentos
        ),
        "palavras_aproximadas": sum(
            len(doc["texto"].split())
            for doc in documentos
        )
    }


def preparar_corpus(
    origem="data/raw",
    destino="data/v3",
    proporcao_treino=0.9,
    semente=42
):
    documentos = carregar_corpus(origem)

    if len(documentos) < 2:
        raise ValueError(
            "Adicione pelo menos dois documentos "
            "TXT distintos em data/raw"
        )

    treino, validacao = dividir_documentos(
        documentos,
        proporcao_treino,
        semente
    )

    destino = Path(destino)

    # Evita sobrescrever silenciosamente
    # conjuntos de dados anteriores.
    if destino.exists():
        arquivos_existentes = list(
            destino.rglob("*")
        )

        if any(
            arquivo.is_file()
            for arquivo in arquivos_existentes
        ):
            raise FileExistsError(
                f"Destino ja possui arquivos: {destino}. "
                "Escolha outro destino para esta execucao."
            )

    pasta_treino = destino / "train"
    pasta_validacao = destino / "validation"

    salvar_documentos(
        treino,
        pasta_treino
    )

    salvar_documentos(
        validacao,
        pasta_validacao
    )

    resumo = {
        "origem": str(origem),
        "semente": semente,
        "proporcao_treino": proporcao_treino,
        "total": estatisticas_documentos(documentos),
        "treino": estatisticas_documentos(treino),
        "validacao": estatisticas_documentos(validacao),
        "arquivos_treino": [
            doc["origem"] for doc in treino
        ],
        "arquivos_validacao": [
            doc["origem"] for doc in validacao
        ]
    }

    caminho_resumo = destino / "resumo.json"

    caminho_resumo.write_text(
        json.dumps(
            resumo,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    return resumo


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--origem",
        default="data/raw"
    )

    parser.add_argument(
        "--destino",
        default="data/v3"
    )

    parser.add_argument(
        "--semente",
        type=int,
        default=42
    )

    args = parser.parse_args()

    resumo = preparar_corpus(
        origem=args.origem,
        destino=args.destino,
        semente=args.semente
    )

    print("\n=== CYGNUS AI - CORPUS V3 ===")

    for conjunto in ("total", "treino", "validacao"):
        dados = resumo[conjunto]

        print(
            f"{conjunto}: "
            f"{dados['documentos']} documentos | "
            f"{dados['caracteres']} caracteres | "
            f"{dados['palavras_aproximadas']} palavras"
        )

    print("\nCorpus preparado com sucesso.")


if __name__ == "__main__":
    main()
