
# Cygnus AI - Fase 12
# Geracao autorregressiva de texto

from pathlib import Path

import torch

from src.byte_bpe_tokenizer import CygnusByteBPETokenizer
from src.language_model import CygnusLanguageModel


CAMINHO_MODELO = Path(
    "checkpoints/cygnus_language_model.pt"
)

CAMINHO_TOKENIZADOR = Path(
    "checkpoints/cygnus_tokenizer.json"
)


def carregar_cygnus(
    caminho_modelo=CAMINHO_MODELO,
    caminho_tokenizador=CAMINHO_TOKENIZADOR
):
    caminho_modelo = Path(caminho_modelo)
    caminho_tokenizador = Path(caminho_tokenizador)

    if not caminho_modelo.exists():
        raise FileNotFoundError(
            f"Modelo nao encontrado: {caminho_modelo}"
        )

    if not caminho_tokenizador.exists():
        raise FileNotFoundError(
            f"Tokenizador nao encontrado: {caminho_tokenizador}"
        )

    # Carregar apenas checkpoints locais e confiaveis
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
            "Vocabulario incompatível com o modelo"
        )

    modelo = CygnusLanguageModel(**configuracao)

    modelo.load_state_dict(
        checkpoint["model_state_dict"]
    )

    modelo.eval()

    return modelo, tokenizador


@torch.no_grad()
def gerar_tokens(
    modelo,
    tokens_iniciais,
    max_novos_tokens=100,
    temperatura=1.0,
    top_k=20
):
    if not tokens_iniciais:
        raise ValueError("O prompt nao pode estar vazio")

    if max_novos_tokens < 0:
        raise ValueError(
            "max_novos_tokens nao pode ser negativo"
        )

    if temperatura <= 0:
        raise ValueError(
            "A temperatura deve ser maior que zero"
        )

    if top_k is not None and top_k <= 0:
        raise ValueError(
            "top_k deve ser positivo ou None"
        )

    tokens = torch.tensor(
        [tokens_iniciais],
        dtype=torch.long
    )

    modelo.eval()

    for _ in range(max_novos_tokens):
        # Respeitar o tamanho maximo do contexto
        contexto = tokens[
            :,
            -modelo.tamanho_contexto:
        ]

        logits, _ = modelo(contexto)

        # Apenas a ultima posicao preve o proximo token
        proximos_logits = logits[:, -1, :]

        # Ajustar a temperatura
        proximos_logits = (
            proximos_logits / temperatura
        )

        # Manter apenas os k tokens mais provaveis
        if top_k is not None:
            k = min(
                top_k,
                proximos_logits.size(-1)
            )

            valores, _ = torch.topk(
                proximos_logits,
                k
            )

            limite = valores[:, -1].unsqueeze(-1)

            proximos_logits = (
                proximos_logits.masked_fill(
                    proximos_logits < limite,
                    float("-inf")
                )
            )

        probabilidades = torch.softmax(
            proximos_logits,
            dim=-1
        )

        proximo_token = torch.multinomial(
            probabilidades,
            num_samples=1
        )

        tokens = torch.cat(
            [tokens, proximo_token],
            dim=1
        )

    return tokens[0].tolist()


def gerar_texto(
    modelo,
    tokenizador,
    prompt,
    max_novos_tokens=100,
    temperatura=0.8,
    top_k=20
):
    if not prompt:
        raise ValueError("O prompt nao pode estar vazio")

    tokens_iniciais = tokenizador.codificar(prompt)

    tokens_gerados = gerar_tokens(
        modelo=modelo,
        tokens_iniciais=tokens_iniciais,
        max_novos_tokens=max_novos_tokens,
        temperatura=temperatura,
        top_k=top_k
    )

    # UTF-8 pode terminar com uma sequencia incompleta
    # de bytes. Decodificamos o resultado final com
    # substituicao apenas para sequencias incompletas.
    dados = b"".join(
        tokenizador.vocabulario[token]
        for token in tokens_gerados
    )

    return dados.decode(
        "utf-8",
        errors="replace"
    )


def main():
    torch.manual_seed(42)

    print("=== CYGNUS AI - GERACAO DE TEXTO ===")

    modelo, tokenizador = carregar_cygnus()

    parametros = sum(
        p.numel()
        for p in modelo.parameters()
    )

    print("Modelo carregado com sucesso.")
    print("Parametros:", parametros)
    print(
        "Vocabulario:",
        tokenizador.tamanho_vocabulario
    )
    print(
        "Contexto:",
        modelo.tamanho_contexto
    )

    print("\nDigite 'sair' para encerrar.")

    while True:
        try:
            prompt = input("\nVoce: ")
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando Cygnus.")
            break

        if prompt.strip().lower() == "sair":
            break

        if not prompt:
            print("Digite um texto para continuar.")
            continue

        resultado = gerar_texto(
            modelo=modelo,
            tokenizador=tokenizador,
            prompt=prompt,
            max_novos_tokens=100,
            temperatura=0.3,
            top_k=5
        )

        print("\nCygnus:")
        print(resultado)


if __name__ == "__main__":
    main()
