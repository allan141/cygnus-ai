
# Cygnus AI - Fase 9
# Primeiro modelo de linguagem Transformer decoder-only

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.transformer_block import CygnusTransformerBlock


class CygnusLanguageModel(nn.Module):
    def __init__(
        self,
        tamanho_vocabulario,
        dimensao_embedding=32,
        numero_cabecas=4,
        numero_camadas=2,
        tamanho_contexto=64
    ):
        super().__init__()

        if tamanho_vocabulario <= 0:
            raise ValueError("Vocabulario deve ser positivo")

        if tamanho_contexto <= 0:
            raise ValueError("Contexto deve ser positivo")

        if numero_camadas <= 0:
            raise ValueError("Numero de camadas deve ser positivo")

        self.tamanho_vocabulario = tamanho_vocabulario
        self.tamanho_contexto = tamanho_contexto

        # Representacao de cada token
        self.token_embedding = nn.Embedding(
            tamanho_vocabulario,
            dimensao_embedding
        )

        # Representacao da posicao de cada token
        self.position_embedding = nn.Embedding(
            tamanho_contexto,
            dimensao_embedding
        )

        # Blocos Transformer
        self.blocos = nn.ModuleList([
            CygnusTransformerBlock(
                dimensao_embedding=dimensao_embedding,
                numero_cabecas=numero_cabecas,
                multiplicador_ffn=4
            )
            for _ in range(numero_camadas)
        ])

        # Normalizacao final
        self.norm_final = nn.LayerNorm(dimensao_embedding)

        # Projecao para o vocabulario
        self.lm_head = nn.Linear(
            dimensao_embedding,
            tamanho_vocabulario,
            bias=False
        )

    def forward(self, tokens, alvos=None):
        if tokens.ndim != 2:
            raise ValueError(
                "Tokens devem ter formato [batch, sequencia]"
            )

        if tokens.dtype != torch.long:
            raise ValueError("Tokens devem ser inteiros torch.long")

        batch, sequencia = tokens.shape

        if sequencia == 0:
            raise ValueError("Sequencia nao pode estar vazia")

        if sequencia > self.tamanho_contexto:
            raise ValueError(
                "Sequencia maior que o contexto permitido"
            )

        if torch.any(tokens < 0) or torch.any(
            tokens >= self.tamanho_vocabulario
        ):
            raise ValueError("ID de token fora do vocabulario")

        # Posicoes: 0, 1, 2, ..., sequencia - 1
        posicoes = torch.arange(
            sequencia,
            device=tokens.device
        )

        # Embeddings de tokens e posicoes
        x = (
            self.token_embedding(tokens)
            + self.position_embedding(posicoes)
        )

        # Processamento pelos blocos Transformer
        for bloco in self.blocos:
            x = bloco(x)

        # Normalizacao e projecao final
        x = self.norm_final(x)
        logits = self.lm_head(x)

        if alvos is None:
            return logits, None

        if alvos.shape != tokens.shape:
            raise ValueError(
                "Alvos devem ter o mesmo formato dos tokens"
            )

        if alvos.dtype != torch.long:
            raise ValueError("Alvos devem ser torch.long")

        if torch.any(alvos < 0) or torch.any(
            alvos >= self.tamanho_vocabulario
        ):
            raise ValueError("ID de alvo fora do vocabulario")

        # Cross-Entropy sobre todos os tokens do lote
        perda = F.cross_entropy(
            logits.reshape(-1, self.tamanho_vocabulario),
            alvos.reshape(-1)
        )

        return logits, perda


if __name__ == "__main__":
    torch.manual_seed(42)

    modelo = CygnusLanguageModel(
        tamanho_vocabulario=100,
        dimensao_embedding=32,
        numero_cabecas=4,
        numero_camadas=2,
        tamanho_contexto=64
    )

    # IDs de tokens simulados
    tokens = torch.tensor(
        [[1, 5, 12, 7, 9]],
        dtype=torch.long
    )

    # Alvos deslocados uma posicao para a esquerda
    alvos = torch.tensor(
        [[5, 12, 7, 9, 3]],
        dtype=torch.long
    )

    logits, perda = modelo(tokens, alvos)

    print("=== CYGNUS LANGUAGE MODEL v0.1 ===")
    print("Formato dos tokens:", tokens.shape)
    print("Formato dos logits:", logits.shape)
    print("Perda inicial:", round(perda.item(), 4))

    print(
        "Parametros treinaveis:",
        sum(p.numel() for p in modelo.parameters())
    )

    # Probabilidades previstas para o proximo token
    probabilidades = F.softmax(
        logits[0, -1],
        dim=-1
    )

    print(
        "Token previsto:",
        torch.argmax(probabilidades).item()
    )
