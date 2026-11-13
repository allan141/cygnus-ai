
# Cygnus AI - Fase 11
# Primeiro treinamento do modelo de linguagem

import random
from pathlib import Path

import torch

from src.byte_bpe_tokenizer import CygnusByteBPETokenizer
from src.language_model import CygnusLanguageModel


SEMENTE = 42
TAMANHO_CONTEXTO = 32
DIMENSAO_EMBEDDING = 32
NUMERO_CABECAS = 4
NUMERO_CAMADAS = 2

BATCH_SIZE = 8
PASSOS = 300
TAXA_APRENDIZADO = 0.001
INTERVALO_AVALIACAO = 50

CAMINHO_CORPUS = Path("data/corpus.txt")
CAMINHO_CHECKPOINT = Path(
    "checkpoints/cygnus_language_model.pt"
)
CAMINHO_TOKENIZADOR = Path(
    "checkpoints/cygnus_tokenizer.json"
)


def preparar_dados(texto, tokenizador):
    ids = tokenizador.codificar(texto)

    if len(ids) < TAMANHO_CONTEXTO + 2:
        raise ValueError(
            "Corpus muito pequeno para o contexto configurado"
        )

    # Separacao temporal: os tokens finais ficam para validacao
    limite = int(len(ids) * 0.9)

    treino = torch.tensor(
        ids[:limite],
        dtype=torch.long
    )

    validacao = torch.tensor(
        ids[limite:],
        dtype=torch.long
    )

    if len(treino) <= TAMANHO_CONTEXTO:
        raise ValueError("Dados de treinamento insuficientes")

    if len(validacao) <= TAMANHO_CONTEXTO:
        raise ValueError("Dados de validacao insuficientes")

    return treino, validacao


def criar_lote(dados, tamanho_lote=BATCH_SIZE):
    limite = len(dados) - TAMANHO_CONTEXTO

    if limite <= 0:
        raise ValueError("Dados insuficientes para criar lotes")

    inicios = torch.randint(
        0,
        limite,
        (tamanho_lote,)
    )

    entradas = torch.stack([
        dados[i:i + TAMANHO_CONTEXTO]
        for i in inicios.tolist()
    ])

    alvos = torch.stack([
        dados[i + 1:i + TAMANHO_CONTEXTO + 1]
        for i in inicios.tolist()
    ])

    return entradas, alvos


@torch.no_grad()
def avaliar(modelo, treino, validacao, lotes=10):
    modelo.eval()

    resultados = {}

    for nome, dados in [
        ("treino", treino),
        ("validacao", validacao)
    ]:
        perdas = []

        for _ in range(lotes):
            x, y = criar_lote(dados)
            _, perda = modelo(x, y)
            perdas.append(perda.item())

        resultados[nome] = sum(perdas) / len(perdas)

    modelo.train()

    return resultados


def treinar():
    random.seed(SEMENTE)
    torch.manual_seed(SEMENTE)

    if not CAMINHO_CORPUS.exists():
        raise FileNotFoundError(
            f"Corpus nao encontrado: {CAMINHO_CORPUS}"
        )

    texto = CAMINHO_CORPUS.read_text(
        encoding="utf-8"
    )

    if not texto.strip():
        raise ValueError("Corpus vazio")

    print("=== CYGNUS AI - PRIMEIRO TREINAMENTO ===")
    print("Caracteres no corpus:", len(texto))

    # Treinar o nosso tokenizador
    tokenizador = CygnusByteBPETokenizer()

    tokenizador.treinar(
        [texto],
        numero_merges=100
    )

    print(
        "Tamanho do vocabulario:",
        tokenizador.tamanho_vocabulario
    )

    treino, validacao = preparar_dados(
        texto,
        tokenizador
    )

    print("Tokens de treinamento:", len(treino))
    print("Tokens de validacao:", len(validacao))

    # Criar nosso modelo do zero
    configuracao = {
        "tamanho_vocabulario": tokenizador.tamanho_vocabulario,
        "dimensao_embedding": DIMENSAO_EMBEDDING,
        "numero_cabecas": NUMERO_CABECAS,
        "numero_camadas": NUMERO_CAMADAS,
        "tamanho_contexto": TAMANHO_CONTEXTO
    }

    modelo = CygnusLanguageModel(**configuracao)

    total_parametros = sum(
        parametro.numel()
        for parametro in modelo.parameters()
    )

    print("Parametros treinaveis:", total_parametros)

    otimizador = torch.optim.AdamW(
        modelo.parameters(),
        lr=TAXA_APRENDIZADO
    )

    modelo.train()

    for passo in range(PASSOS + 1):
        if passo % INTERVALO_AVALIACAO == 0:
            resultados = avaliar(
                modelo,
                treino,
                validacao
            )

            print(
                f"Passo {passo:03d} | "
                f"Treino: {resultados['treino']:.4f} | "
                f"Validacao: {resultados['validacao']:.4f}"
            )

        if passo == PASSOS:
            break

        x, y = criar_lote(treino)

        _, perda = modelo(x, y)

        otimizador.zero_grad()
        perda.backward()

        torch.nn.utils.clip_grad_norm_(
            modelo.parameters(),
            max_norm=1.0
        )

        otimizador.step()

    # Salvar os pesos treinados
    CAMINHO_CHECKPOINT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        {
            "model_state_dict": modelo.state_dict(),
            "configuracao": configuracao,
            "passos": PASSOS,
            "seed": SEMENTE
        },
        CAMINHO_CHECKPOINT
    )

    tokenizador.salvar(CAMINHO_TOKENIZADOR)

    print("\nTreinamento finalizado.")
    print("Modelo salvo em:", CAMINHO_CHECKPOINT)
    print("Tokenizador salvo em:", CAMINHO_TOKENIZADOR)


if __name__ == "__main__":
    treinar()
