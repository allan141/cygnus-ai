
# Cygnus AI - Fase 8
# Bloco Transformer decoder-only com Pre-LayerNorm

import torch
import torch.nn as nn

from src.multihead_attention import CygnusMultiHeadAttention


class CygnusFeedForward(nn.Module):
    def __init__(self, dimensao_embedding=32, multiplicador=4):
        super().__init__()

        if dimensao_embedding <= 0 or multiplicador <= 0:
            raise ValueError("As dimensoes devem ser positivas")

        dimensao_interna = dimensao_embedding * multiplicador

        self.rede = nn.Sequential(
            nn.Linear(dimensao_embedding, dimensao_interna),
            nn.GELU(),
            nn.Linear(dimensao_interna, dimensao_embedding)
        )

    def forward(self, x):
        return self.rede(x)


class CygnusTransformerBlock(nn.Module):
    def __init__(
        self,
        dimensao_embedding=32,
        numero_cabecas=4,
        multiplicador_ffn=4
    ):
        super().__init__()

        self.dimensao_embedding = dimensao_embedding

        # Primeira normalizacao
        self.norm1 = nn.LayerNorm(dimensao_embedding)

        # Atencao causal com multiplas cabecas
        self.atencao = CygnusMultiHeadAttention(
            dimensao_embedding=dimensao_embedding,
            numero_cabecas=numero_cabecas
        )

        # Segunda normalizacao
        self.norm2 = nn.LayerNorm(dimensao_embedding)

        # Rede feedforward
        self.feedforward = CygnusFeedForward(
            dimensao_embedding=dimensao_embedding,
            multiplicador=multiplicador_ffn
        )

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(
                "Entrada deve ter formato [batch, sequencia, dimensao]"
            )

        if x.shape[-1] != self.dimensao_embedding:
            raise ValueError("Dimensao de embedding incompatível")

        # Primeira conexao residual
        x = x + self.atencao(self.norm1(x))

        # Segunda conexao residual
        x = x + self.feedforward(self.norm2(x))

        return x


if __name__ == "__main__":
    torch.manual_seed(42)

    modelo = CygnusTransformerBlock(
        dimensao_embedding=32,
        numero_cabecas=4,
        multiplicador_ffn=4
    )

    # Simular 4 tokens com embeddings de 32 dimensoes
    entrada = torch.randn(1, 4, 32)

    saida = modelo(entrada)

    print("=== CYGNUS TRANSFORMER BLOCK ===")
    print("Formato da entrada:", entrada.shape)
    print("Formato da saida:", saida.shape)

    print(
        "\nParametros treinaveis:",
        sum(p.numel() for p in modelo.parameters())
    )

    print(
        "\nSaida contem valores finitos:",
        torch.isfinite(saida).all().item()
    )
