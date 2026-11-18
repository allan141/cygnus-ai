
import hashlib
import json
import os
import pickle
import tempfile
import time
from pathlib import Path

from src.v7_dataset import preparar_documentos
from src.v7_tokenizer import CygnusV7Tokenizer


VERSAO_CACHE = 1


def assinatura_arquivo(caminho):
    hash_arquivo = hashlib.sha256()

    with open(caminho, "rb") as arquivo:
        for bloco in iter(
            lambda: arquivo.read(1024 * 1024),
            b"",
        ):
            hash_arquivo.update(bloco)

    return hash_arquivo.hexdigest()


def assinatura_dataset(
    pasta,
    caminho_tokenizador,
    proporcao_validacao,
    seed,
):
    pasta = Path(pasta)
    arquivos = sorted(pasta.glob("*.txt"))

    if not arquivos:
        raise ValueError(
            f"Nenhum documento .txt encontrado em {pasta}"
        )

    hash_dataset = hashlib.sha256()

    configuracao = {
        "versao_cache": VERSAO_CACHE,
        "proporcao_validacao": proporcao_validacao,
        "seed": seed,
        "tokenizador": assinatura_arquivo(
            caminho_tokenizador
        ),
    }

    hash_dataset.update(
        json.dumps(
            configuracao,
            sort_keys=True,
        ).encode("utf-8")
    )

    for arquivo in arquivos:
        hash_dataset.update(
            arquivo.name.encode("utf-8")
        )
        hash_dataset.update(
            assinatura_arquivo(arquivo).encode("ascii")
        )

    return hash_dataset.hexdigest()


def preparar_documentos_com_cache(
    pasta,
    caminho_tokenizador,
    proporcao_validacao=0.1,
    seed=42,
    pasta_cache="data/v7_cache",
):
    pasta_cache = Path(pasta_cache)
    pasta_cache.mkdir(
        parents=True,
        exist_ok=True,
    )

    inicio = time.perf_counter()

    print(
        "Verificando cache do dataset...",
        flush=True,
    )

    assinatura = assinatura_dataset(
        pasta,
        caminho_tokenizador,
        proporcao_validacao,
        seed,
    )

    caminho_cache = (
        pasta_cache / f"dataset_{assinatura}.pkl"
    )

    # ===== CARREGAR CACHE EXISTENTE =====

    if caminho_cache.exists():
        print(
            "Cache encontrado. Carregando...",
            flush=True,
        )

        # Carregar somente arquivos locais confiáveis.
        with open(caminho_cache, "rb") as arquivo:
            dados = pickle.load(arquivo)

        tokenizer = CygnusV7Tokenizer(
            caminho_tokenizador
        )

        print(
            f"Cache carregado em "
            f"{time.perf_counter() - inicio:.2f}s",
            flush=True,
        )

        return (
            dados["treino"],
            dados["validacao"],
            tokenizer,
        )

    # ===== CRIAR NOVO CACHE =====

    print(
        "Cache não encontrado. Tokenizando corpus...",
        flush=True,
    )

    # IMPORTANTE:
    # Chama a função ORIGINAL do v7_dataset.
    # Não chama preparar_documentos_com_cache novamente.
    treino, validacao, tokenizer = preparar_documentos(
        pasta,
        caminho_tokenizador,
        proporcao_validacao=proporcao_validacao,
        seed=seed,
    )

    dados = {
        "treino": treino,
        "validacao": validacao,
    }

    # ===== SALVAR CACHE COM SEGURANÇA =====

    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=pasta_cache,
        suffix=".tmp",
        delete=False,
    ) as arquivo:
        temporario = Path(arquivo.name)

        try:
            pickle.dump(
                dados,
                arquivo,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        except Exception:
            temporario.unlink(missing_ok=True)
            raise

    try:
        os.replace(
            temporario,
            caminho_cache,
        )
    finally:
        temporario.unlink(missing_ok=True)

    print(
        f"Cache salvo em: {caminho_cache}",
        flush=True,
    )

    print(
        f"Preparação concluída em "
        f"{time.perf_counter() - inicio:.2f}s",
        flush=True,
    )

    return treino, validacao, tokenizer
