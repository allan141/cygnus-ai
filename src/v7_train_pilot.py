
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from src.v6_language_model import CygnusLanguageModel
from src.v7_dataset_cache import preparar_documentos_com_cache
from src.v7_data_loader import CygnusV7DataLoader
from src.v7_optimization import calcular_lr, aplicar_lr


# ===== CONFIGURAÇÕES =====

SEED = 42
PASSOS = 100
BATCH_SIZE = 2
ACUMULACAO = 4
CONTEXTO = 256

LR_MAX = 3e-4
LR_MIN = 3e-5
WARMUP = 10

AVALIAR_CADA = 20
LOTES_VALIDACAO = 10

PASTA_CORPUS = "data/v7_corpus_inicial/train"
CAMINHO_TOKENIZADOR = (
    "checkpoints/cygnus_v4_tokenizers/tokenizador_2048.json"
)
PASTA_SAIDA = Path("checkpoints/cygnus_v7_piloto")


def configurar_seeds():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


@torch.no_grad()
def avaliar(modelo, loader, dispositivo):
    modelo.eval()
    perdas = []

    for _ in range(LOTES_VALIDACAO):
        x, y = loader.lote(batch_size=BATCH_SIZE)

        x = x.to(dispositivo)
        y = y.to(dispositivo)

        _, perda = modelo(x, y)
        perdas.append(perda.item())

    modelo.train()
    return sum(perdas) / len(perdas)


def main():
    configurar_seeds()

    # Evita sobrecarregar a CPU com threads excessivas.
    torch.set_num_threads(4)

    dispositivo = torch.device("cpu")
    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

    print("=== CYGNUS V7 | TREINAMENTO PILOTO ===", flush=True)
    print("Dispositivo:", dispositivo, flush=True)
    print("Preparando documentos...", flush=True)

    inicio_preparacao = time.perf_counter()

    treino, validacao, tokenizer = preparar_documentos_com_cache(
        PASTA_CORPUS,
        CAMINHO_TOKENIZADOR,
        proporcao_validacao=0.1,
        seed=SEED,
    )

    print(
        f"Preparação: {time.perf_counter() - inicio_preparacao:.1f}s",
        flush=True,
    )
    print(f"Treino: {len(treino)} documentos", flush=True)
    print(f"Validação: {len(validacao)} documentos", flush=True)

    if not treino or not validacao:
        raise RuntimeError("Treino ou validação sem documentos.")

    loader_treino = CygnusV7DataLoader(
        treino, contexto=CONTEXTO, seed=SEED
    )
    loader_validacao = CygnusV7DataLoader(
        validacao, contexto=CONTEXTO, seed=SEED + 1
    )

    modelo = CygnusLanguageModel(
        tamanho_vocabulario=tokenizer.tamanho_vocabulario,
        dimensao_embedding=192,
        numero_cabecas=6,
        numero_camadas=6,
        tamanho_contexto=CONTEXTO,
    ).to(dispositivo)

    parametros = sum(p.numel() for p in modelo.parameters())
    print(f"Parâmetros: {parametros:,}", flush=True)

    otimizador = torch.optim.AdamW(
        modelo.parameters(),
        lr=LR_MAX,
        weight_decay=0.01,
    )

    historico = []
    melhor_validacao = float("inf")
    inicio_treino = time.perf_counter()

    modelo.train()

    for passo in range(1, PASSOS + 1):
        inicio_passo = time.perf_counter()

        lr = calcular_lr(
            passo=passo,
            total_passos=PASSOS,
            warmup_passos=WARMUP,
            lr_max=LR_MAX,
            lr_min=LR_MIN,
        )
        aplicar_lr(otimizador, lr)

        otimizador.zero_grad(set_to_none=True)
        perdas_treino = []

        for _ in range(ACUMULACAO):
            x, y = loader_treino.lote(batch_size=BATCH_SIZE)
            x = x.to(dispositivo)
            y = y.to(dispositivo)

            _, perda = modelo(x, y)

            if not torch.isfinite(perda):
                raise RuntimeError(
                    f"Perda não finita no passo {passo}: {perda.item()}"
                )

            (perda / ACUMULACAO).backward()
            perdas_treino.append(perda.item())

        torch.nn.utils.clip_grad_norm_(
            modelo.parameters(), max_norm=1.0
        )

        otimizador.step()

        perda_media = sum(perdas_treino) / len(perdas_treino)
        tempo_passo = time.perf_counter() - inicio_passo

        registro = {
            "passo": passo,
            "loss_treino": perda_media,
            "lr": lr,
            "segundos": tempo_passo,
        }

        if passo == 1 or passo % AVALIAR_CADA == 0:
            perda_validacao = avaliar(
                modelo, loader_validacao, dispositivo
            )

            registro["loss_validacao"] = perda_validacao

            print(
                f"Passo {passo:03d}/{PASSOS} | "
                f"Treino: {perda_media:.4f} | "
                f"Validação: {perda_validacao:.4f} | "
                f"LR: {lr:.6f} | "
                f"Tempo: {tempo_passo:.1f}s",
                flush=True,
            )

            if perda_validacao < melhor_validacao:
                melhor_validacao = perda_validacao

                torch.save(
                    {
                        "modelo": modelo.state_dict(),
                        "passo": passo,
                        "loss_validacao": perda_validacao,
                        "configuracao": {
                            "tamanho_vocabulario": tokenizer.tamanho_vocabulario,
                            "dimensao_embedding": 192,
                            "numero_cabecas": 6,
                            "numero_camadas": 6,
                            "tamanho_contexto": CONTEXTO,
                        },
                    },
                    PASTA_SAIDA / "melhor_modelo.pt",
                )

        historico.append(registro)

    duracao = time.perf_counter() - inicio_treino

    with open(
        PASTA_SAIDA / "historico.json",
        "w",
        encoding="utf-8",
    ) as arquivo:
        json.dump(
            {
                "duracao_segundos": duracao,
                "melhor_loss_validacao": melhor_validacao,
                "historico": historico,
            },
            arquivo,
            indent=2,
            ensure_ascii=False,
        )

    print("\n=== RESULTADO FINAL ===", flush=True)
    print(f"Tempo total: {duracao / 60:.1f} minutos", flush=True)
    print(f"Melhor loss validação: {melhor_validacao:.4f}", flush=True)
    print(f"Checkpoint: {PASTA_SAIDA / 'melhor_modelo.pt'}", flush=True)
    print("Treinamento piloto concluído!", flush=True)


if __name__ == "__main__":
    main()
