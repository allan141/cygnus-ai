
# Cygnus AI - Fase 13, Parte 2
# Treinamento avancado com retomada de checkpoints

import argparse
import csv
import math
import random
from pathlib import Path

import torch

from src.dataset import (
    preparar_dataset,
    criar_lote_documentos
)
from src.byte_bpe_tokenizer import CygnusByteBPETokenizer
from src.language_model import CygnusLanguageModel


def definir_semente(semente):
    random.seed(semente)
    torch.manual_seed(semente)


def criar_modelo(configuracao):
    return CygnusLanguageModel(**configuracao)


@torch.no_grad()
def avaliar(modelo, sequencias, contexto, batch_size, lotes=5):
    if lotes <= 0:
        raise ValueError("lotes deve ser positivo")

    estava_treinando = modelo.training
    modelo.eval()

    perdas = []

    for _ in range(lotes):
        x, y = criar_lote_documentos(
            sequencias,
            tamanho_contexto=contexto,
            tamanho_lote=batch_size
        )

        _, perda = modelo(x, y)
        perdas.append(perda.item())

    modelo.train(estava_treinando)

    return sum(perdas) / len(perdas)


def perplexidade(perda):
    # Evita overflow para valores extremos.
    return math.exp(min(perda, 80.0))


def salvar_checkpoint(
    caminho,
    modelo,
    otimizador,
    passo,
    configuracao
):
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": modelo.state_dict(),
        "optimizer_state_dict": otimizador.state_dict(),
        "passo": passo,
        "configuracao": configuracao,
        "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate()
    }

    temporario = caminho.with_suffix(".tmp")
    torch.save(checkpoint, temporario)
    temporario.replace(caminho)


def carregar_checkpoint(caminho, modelo, otimizador):
    checkpoint = torch.load(
        caminho,
        map_location="cpu",
        weights_only=True
    )

    modelo.load_state_dict(
        checkpoint["model_state_dict"]
    )

    otimizador.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    torch.set_rng_state(checkpoint["torch_rng_state"])
    random.setstate(checkpoint["python_rng_state"])

    return checkpoint["passo"]


def registrar_metricas(caminho, passo, treino, validacao):
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)

    novo = not caminho.exists()

    with caminho.open(
        "a",
        newline="",
        encoding="utf-8"
    ) as arquivo:
        escritor = csv.writer(arquivo)

        if novo:
            escritor.writerow([
                "passo",
                "perda_treino",
                "perda_validacao",
                "perplexidade_treino",
                "perplexidade_validacao"
            ])

        escritor.writerow([
            passo,
            round(treino, 6),
            round(validacao, 6),
            round(perplexidade(treino), 6),
            round(perplexidade(validacao), 6)
        ])


def treinar(args):
    if args.passos <= 0:
        raise ValueError("passos deve ser positivo")
    if args.contexto <= 0 or args.batch_size <= 0:
        raise ValueError("Contexto e batch devem ser positivos")
    if args.avaliar_cada <= 0:
        raise ValueError("avaliar_cada deve ser positivo")

    definir_semente(args.semente)

    pasta = Path("checkpoints") / args.experimento
    pasta.mkdir(parents=True, exist_ok=True)

    caminho_modelo = pasta / "modelo.pt"
    caminho_tokenizador = pasta / "tokenizador.json"
    caminho_metricas = pasta / "metricas.csv"

    treino, validacao, tokenizador_novo = preparar_dataset(
        diretorio_treino="data/train",
        diretorio_validacao="data/validation",
        numero_merges=args.merges,
        tamanho_contexto=args.contexto
    )

    configuracao = {
        "tamanho_vocabulario": tokenizador_novo.tamanho_vocabulario,
        "dimensao_embedding": args.embedding,
        "numero_cabecas": args.cabecas,
        "numero_camadas": args.camadas,
        "tamanho_contexto": args.contexto
    }

    if args.retomar:
        if not caminho_modelo.exists():
            raise FileNotFoundError(
                f"Checkpoint nao encontrado: {caminho_modelo}"
            )

        if not caminho_tokenizador.exists():
            raise FileNotFoundError(
                "Tokenizador do experimento nao encontrado"
            )

        tokenizador_salvo = CygnusByteBPETokenizer.carregar(
            caminho_tokenizador
        )

        if (
            tokenizador_salvo.vocabulario != tokenizador_novo.vocabulario
            or tokenizador_salvo.merges != tokenizador_novo.merges
        ):
            raise ValueError(
                "O corpus ou o tokenizador mudou. "
                "Crie um novo experimento."
            )

        checkpoint = torch.load(
            caminho_modelo,
            map_location="cpu",
            weights_only=True
        )

        if checkpoint["configuracao"] != configuracao:
            raise ValueError(
                "Configuracao diferente da original"
            )

    else:
        if caminho_modelo.exists() or caminho_tokenizador.exists():
            raise FileExistsError(
                "Experimento ja existe. Use --retomar "
                "ou escolha outro nome."
            )

        tokenizador_novo.salvar(caminho_tokenizador)

    modelo = criar_modelo(configuracao)

    otimizador = torch.optim.AdamW(
        modelo.parameters(),
        lr=args.lr
    )

    passo_inicial = 0

    if args.retomar:
        passo_inicial = carregar_checkpoint(
            caminho_modelo,
            modelo,
            otimizador
        )

    total_parametros = sum(
        p.numel() for p in modelo.parameters()
    )

    print("\n=== CYGNUS AI - TREINAMENTO AVANCADO ===")
    print("Experimento:", args.experimento)
    print("Parametros:", total_parametros)
    print("Vocabulario:", tokenizador_novo.tamanho_vocabulario)
    print("Passo inicial:", passo_inicial)
    print("Passo final solicitado:", args.passos)

    if args.passos <= passo_inicial:
        print("Nenhum passo adicional necessario.")
        return

    modelo.train()

    try:
        for passo in range(passo_inicial + 1, args.passos + 1):
            x, y = criar_lote_documentos(
                treino,
                tamanho_contexto=args.contexto,
                tamanho_lote=args.batch_size
            )

            _, perda = modelo(x, y)

            otimizador.zero_grad()
            perda.backward()

            torch.nn.utils.clip_grad_norm_(
                modelo.parameters(),
                max_norm=1.0
            )

            otimizador.step()

            if (
                passo == 1
                or passo % args.avaliar_cada == 0
                or passo == args.passos
            ):
                perda_treino = avaliar(
                    modelo,
                    treino,
                    args.contexto,
                    args.batch_size
                )

                perda_validacao = avaliar(
                    modelo,
                    validacao,
                    args.contexto,
                    args.batch_size
                )

                print(
                    f"Passo {passo:05d} | "
                    f"Treino: {perda_treino:.4f} | "
                    f"Validacao: {perda_validacao:.4f} | "
                    f"PPL val: {perplexidade(perda_validacao):.2f}"
                )

                registrar_metricas(
                    caminho_metricas,
                    passo,
                    perda_treino,
                    perda_validacao
                )

                salvar_checkpoint(
                    caminho_modelo,
                    modelo,
                    otimizador,
                    passo,
                    configuracao
                )

    except KeyboardInterrupt:
        print("\nTreinamento interrompido pelo usuario.")
        print(
            "Retome a partir do ultimo checkpoint salvo."
        )
        return

    print("\nTreinamento concluido.")
    print("Checkpoint:", caminho_modelo)
    print("Metricas:", caminho_metricas)


def main():
    parser = argparse.ArgumentParser(
        description="Treinador avancado do Cygnus AI"
    )

    parser.add_argument(
        "--experimento",
        default="cygnus_v2"
    )
    parser.add_argument("--passos", type=int, default=500)
    parser.add_argument("--contexto", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--embedding", type=int, default=32)
    parser.add_argument("--cabecas", type=int, default=4)
    parser.add_argument("--camadas", type=int, default=2)
    parser.add_argument("--merges", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--semente", type=int, default=42)
    parser.add_argument("--avaliar-cada", type=int, default=100)
    parser.add_argument(
        "--retomar",
        action="store_true"
    )

    args = parser.parse_args()
    treinar(args)


if __name__ == "__main__":
    main()
