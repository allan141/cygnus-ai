
# Cygnus AI - Fase 14
# Interface de geracao e avaliacao

import argparse
from pathlib import Path

import torch

from src.language_model import CygnusLanguageModel
from src.byte_bpe_tokenizer import CygnusByteBPETokenizer
from src.generate_text import gerar_texto
from src.evaluate_model import (
    avaliar_sequencias,
    carregar_documentos_avaliacao
)


def carregar_modelo_v2(
    experimento="cygnus_v2"
):
    pasta = Path("checkpoints") / experimento

    caminho_modelo = pasta / "modelo.pt"
    caminho_tokenizador = pasta / "tokenizador.json"

    if not caminho_modelo.is_file():
        raise FileNotFoundError(
            f"Checkpoint nao encontrado: {caminho_modelo}"
        )

    if not caminho_tokenizador.is_file():
        raise FileNotFoundError(
            f"Tokenizador nao encontrado: {caminho_tokenizador}"
        )

    checkpoint = torch.load(
        caminho_modelo,
        map_location="cpu",
        weights_only=True
    )

    configuracao = checkpoint["configuracao"]

    tokenizador = CygnusByteBPETokenizer.carregar(
        caminho_tokenizador
    )

    if (
        configuracao["tamanho_vocabulario"]
        != tokenizador.tamanho_vocabulario
    ):
        raise ValueError(
            "Vocabulario incompativel com o modelo"
        )

    modelo = CygnusLanguageModel(
        **configuracao
    )

    modelo.load_state_dict(
        checkpoint["model_state_dict"]
    )

    modelo.eval()

    return modelo, tokenizador, checkpoint


def conversar(
    modelo,
    tokenizador,
    temperatura=0.8,
    top_k=20,
    max_tokens=80
):
    print("\n=== CYGNUS AI v2 ===")
    print("Digite 'sair' para encerrar.")
    print("Digite 'avaliar' para avaliar a validacao.\n")

    while True:
        try:
            prompt = input("Voce: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando Cygnus.")
            break

        if prompt.lower() == "sair":
            break

        if not prompt:
            continue

        if prompt.lower() == "avaliar":
            sequencias = carregar_documentos_avaliacao(
                "data/validation",
                tokenizador
            )

            resultado = avaliar_sequencias(
                modelo,
                sequencias
            )

            print(
                f"\nPerda: {resultado['perda']:.4f}"
            )
            print(
                f"Perplexidade: "
                f"{resultado['perplexidade']:.2f}"
            )
            print(
                f"Tokens avaliados: "
                f"{resultado['tokens_avaliados']}\n"
            )

            continue

        texto = gerar_texto(
            modelo,
            tokenizador,
            prompt,
            max_novos_tokens=max_tokens,
            temperatura=temperatura,
            top_k=top_k
        )

        print(f"\nCygnus: {texto}\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experimento",
        default="cygnus_v2"
    )

    parser.add_argument(
        "--temperatura",
        type=float,
        default=0.8
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=20
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=80
    )

    args = parser.parse_args()

    if args.temperatura <= 0:
        parser.error(
            "A temperatura deve ser positiva"
        )

    if args.top_k <= 0:
        parser.error(
            "top-k deve ser positivo"
        )

    if args.max_tokens < 0:
        parser.error(
            "max-tokens nao pode ser negativo"
        )

    modelo, tokenizador, checkpoint = carregar_modelo_v2(
        args.experimento
    )

    print(
        f"Modelo carregado do passo "
        f"{checkpoint['passo']}"
    )

    conversar(
        modelo,
        tokenizador,
        temperatura=args.temperatura,
        top_k=args.top_k,
        max_tokens=args.max_tokens
    )


if __name__ == "__main__":
    main()
