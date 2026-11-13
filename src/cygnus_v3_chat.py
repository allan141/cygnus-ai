
# Cygnus AI - Fase 15, Parte 7
# Geracao de texto e avaliacao deterministica do v3

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from src.byte_bpe_tokenizer import CygnusByteBPETokenizer
from src.language_model import CygnusLanguageModel
from src.corpus_analyzer import carregar_documentos


def carregar_cygnus_v3(
    checkpoint_path="checkpoints/cygnus_v3_teste01/modelo.pt",
    tokenizer_path="checkpoints/cygnus_v3/tokenizador.json",
):
    checkpoint_path = Path(checkpoint_path)
    tokenizer_path = Path(tokenizer_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    if not tokenizer_path.is_file():
        raise FileNotFoundError(tokenizer_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    configuracao = checkpoint["configuracao"]
    tokenizador = CygnusByteBPETokenizer.carregar(
        tokenizer_path
    )

    if (
        configuracao["tamanho_vocabulario"]
        != tokenizador.tamanho_vocabulario
    ):
        raise ValueError(
            "Vocabulario incompativel com o checkpoint"
        )

    # Verificar a identidade do tokenizador, quando
    # o manifesto do experimento estiver disponivel.
    manifesto_path = checkpoint_path.parent / "manifesto.json"

    if manifesto_path.is_file():
        manifesto = json.loads(
            manifesto_path.read_text(encoding="utf-8")
        )

        hash_atual = hashlib.sha256(
            tokenizer_path.read_bytes()
        ).hexdigest()

        if manifesto.get("tokenizador_sha256") != hash_atual:
            raise ValueError(
                "Tokenizador diferente do usado no treinamento"
            )

    modelo = CygnusLanguageModel(**configuracao)
    modelo.load_state_dict(
        checkpoint["model_state_dict"]
    )
    modelo.eval()

    return modelo, tokenizador, checkpoint


@torch.no_grad()
def gerar_continuacao(
    modelo,
    tokenizador,
    prompt,
    max_novos_tokens=80,
    temperatura=0.8,
    top_k=20,
    semente=42,
):
    if not prompt:
        raise ValueError("Prompt nao pode estar vazio")

    if max_novos_tokens < 0:
        raise ValueError("Quantidade de tokens invalida")

    if temperatura <= 0:
        raise ValueError("Temperatura deve ser positiva")

    if top_k <= 0:
        raise ValueError("top_k deve ser positivo")

    ids = tokenizador.codificar(prompt)

    if not ids:
        raise ValueError("Prompt sem tokens")

    tokens = torch.tensor(
        [ids],
        dtype=torch.long,
    )

    gerador = torch.Generator(device="cpu")
    gerador.manual_seed(semente)

    for _ in range(max_novos_tokens):
        entrada = tokens[
            :, -modelo.tamanho_contexto:
        ]

        logits, _ = modelo(entrada)
        logits = logits[:, -1, :] / temperatura

        k = min(top_k, logits.shape[-1])

        valores, indices = torch.topk(
            logits,
            k=k,
            dim=-1,
        )

        probabilidades = F.softmax(
            valores,
            dim=-1,
        )

        escolha = torch.multinomial(
            probabilidades,
            num_samples=1,
            generator=gerador,
        )

        proximo_token = indices.gather(
            -1,
            escolha,
        )

        tokens = torch.cat(
            [tokens, proximo_token],
            dim=1,
        )

    # Byte-Level BPE pode produzir bytes UTF-8
    # incompletos durante a geracao. Por isso,
    # decodificamos a sequencia final com reposicao.
    dados = b"".join(
        tokenizador.vocabulario[token]
        for token in tokens[0].tolist()
    )

    return dados.decode("utf-8", errors="replace")


@torch.no_grad()
def avaliar_validacao(
    modelo,
    tokenizador,
    diretorio="data/v3_livros/validation",
):
    documentos = carregar_documentos(diretorio)

    modelo.eval()
    contexto = modelo.tamanho_contexto

    perda_acumulada = 0.0
    tokens_avaliados = 0
    janelas = 0

    for documento in documentos:
        ids = tokenizador.codificar(
            documento["texto"]
        )

        # Blocos nao sobrepostos de ate contexto+1
        # tokens. Cada bloco possui entrada e alvo.
        # A avaliacao nao atravessa documentos.
        for inicio in range(0, len(ids) - 1, contexto):
            trecho = ids[
                inicio:inicio + contexto + 1
            ]

            if len(trecho) < 2:
                continue

            x = torch.tensor(
                [trecho[:-1]],
                dtype=torch.long,
            )

            y = torch.tensor(
                [trecho[1:]],
                dtype=torch.long,
            )

            _, perda = modelo(x, y)

            quantidade = y.numel()

            perda_acumulada += (
                perda.item() * quantidade
            )
            tokens_avaliados += quantidade
            janelas += 1

    if tokens_avaliados == 0:
        raise ValueError(
            "Nao ha tokens suficientes para avaliacao"
        )

    perda_media = perda_acumulada / tokens_avaliados

    return {
        "perda": round(perda_media, 6),
        "perplexidade": round(
            math.exp(min(perda_media, 80.0)),
            4,
        ),
        "tokens_avaliados": tokens_avaliados,
        "janelas": janelas,
        "documentos": len(documentos),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Cygnus AI v3 - Geracao e avaliacao"
    )

    parser.add_argument(
        "--checkpoint",
        default="checkpoints/cygnus_v3_teste01/modelo.pt",
    )
    parser.add_argument(
        "--tokenizador",
        default="checkpoints/cygnus_v3/tokenizador.json",
    )
    parser.add_argument(
        "--validacao",
        default="data/v3_livros/validation",
    )
    parser.add_argument(
        "--prompt",
        default=None,
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=80,
    )
    parser.add_argument(
        "--temperatura",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--semente",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--avaliar",
        action="store_true",
    )

    args = parser.parse_args()

    modelo, tokenizador, checkpoint = carregar_cygnus_v3(
        args.checkpoint,
        args.tokenizador,
    )

    print("\n=== CYGNUS AI v3 ===")
    print("Passo treinado:", checkpoint["passo"])
    print("Vocabulario:", tokenizador.tamanho_vocabulario)
    print("Contexto:", modelo.tamanho_contexto)

    if args.avaliar:
        resultado = avaliar_validacao(
            modelo,
            tokenizador,
            args.validacao,
        )

        print("\n=== AVALIACAO FIXA ===")
        print("Perda:", resultado["perda"])
        print("Perplexidade:", resultado["perplexidade"])
        print(
            "Tokens avaliados:",
            resultado["tokens_avaliados"],
        )
        print("Janelas:", resultado["janelas"])
        return

    if args.prompt is not None:
        texto = gerar_continuacao(
            modelo,
            tokenizador,
            args.prompt,
            max_novos_tokens=args.max_tokens,
            temperatura=args.temperatura,
            top_k=args.top_k,
            semente=args.semente,
        )

        print("\nCygnus:\n", texto)
        return

    print("\nDigite 'sair' para encerrar.")

    while True:
        try:
            prompt = input("\nVoce: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando.")
            break

        if prompt.lower() == "sair":
            break

        if not prompt:
            continue

        texto = gerar_continuacao(
            modelo,
            tokenizador,
            prompt,
            max_novos_tokens=args.max_tokens,
            temperatura=args.temperatura,
            top_k=args.top_k,
            semente=args.semente,
        )

        print("\nCygnus:\n", texto)


if __name__ == "__main__":
    main()
