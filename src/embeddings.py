# Cygnus AI - Fase 5
# Primeira camada de embeddings do modelo

import torch
import torch.nn as nn

from src.bpe_tokenizer import CygnusBPETokenizer

class CygnusEmbeddings(nn.Module):
    def __init__(self, tamanho_vocabulario, dimensao_embedding=32):
        super().__init__()

        self.embeddings = nn.Embedding(
            num_embeddings=tamanho_vocabulario,
            embedding_dim=dimensao_embedding
        )

    def forward(self, tokens):
        return self.embeddings(tokens)

if __name__ == "__main__":
    torch.manual_seed(42)

    #Treinar nosso tokenizador BPE
    texto_treinamento = (
        "Cygnus AI aprende com textos. "
        "Cygnus AI sera treinado do zero. "
        "Cygnus AI utiliza redes neurais. "
    )

    tokenizer = CygnusBPETokenizer()
    tokenizer.treinar(texto_treinamento, numero_merges=20)

    # Transformar texto em IDs
    frase = "Cygnus AI"
    ids = tokenizer.codificar(frase)

    # Converter IDs em PyTorch
    tokens = torch.tensor(ids, dtype=torch.long)

    # Criar camada de embeddings
    tamanho_vocabulario = len(tokenizer.vocabulario) + 1

    modelo = CygnusEmbeddings(
        tamanho_vocabulario=tamanho_vocabulario,
        dimensao_embedding=32
    )

    # Gerar vetores
    vetores = modelo(tokens)

    print("=== CYGNUS EMBEDDINGS ===")
    print("Texto original:", frase)
    print("IDs:", ids)
    print("Quantidade de tokens:", len(ids))
    print("Dimensao dos vetores:", vetores.shape)
    print("Primeiro vetor: ", vetores[0])
    print(
        "Parametros treinaveis:",
        sum(p.numel() for p in modelo.parameters())
    )