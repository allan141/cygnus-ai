
# Cygnus AI - Fase 10
# Tokenizador Byte-Level BPE desenvolvido do zero

import json
from collections import Counter
from pathlib import Path


class CygnusByteBPETokenizer:
    def __init__(self):
        # Cada byte possui um ID proprio
        self.vocabulario = {
            i: bytes([i])
            for i in range(256)
        }

        # Lista de merges aprendidos
        self.merges = []

    @property
    def tamanho_vocabulario(self):
        return len(self.vocabulario)

    def _aplicar_merge(self, tokens, par, novo_id):
        resultado = []
        i = 0

        while i < len(tokens):
            if (
                i + 1 < len(tokens)
                and (tokens[i], tokens[i + 1]) == par
            ):
                resultado.append(novo_id)
                i += 2
            else:
                resultado.append(tokens[i])
                i += 1

        return resultado

    def treinar(self, textos, numero_merges=100):
        if numero_merges < 0:
            raise ValueError("Numero de merges nao pode ser negativo")

        # Reiniciar o treinamento do tokenizador
        self.vocabulario = {
            i: bytes([i])
            for i in range(256)
        }
        self.merges = []

        # Converter os textos para bytes
        sequencias = [
            list(texto.encode("utf-8"))
            for texto in textos
        ]

        for _ in range(numero_merges):
            frequencias = Counter()

            for sequencia in sequencias:
                for par in zip(sequencia, sequencia[1:]):
                    frequencias[par] += 1

            if not frequencias:
                break

            par, frequencia = frequencias.most_common(1)[0]

            if frequencia < 2:
                break

            novo_id = len(self.vocabulario)

            self.vocabulario[novo_id] = (
                self.vocabulario[par[0]]
                + self.vocabulario[par[1]]
            )

            self.merges.append((par, novo_id))

            sequencias = [
                self._aplicar_merge(sequencia, par, novo_id)
                for sequencia in sequencias
            ]

    def codificar(self, texto):
        tokens = list(texto.encode("utf-8"))

        for par, novo_id in self.merges:
            tokens = self._aplicar_merge(
                tokens,
                par,
                novo_id
            )

        return tokens

    def decodificar(self, tokens):
        dados = b"".join(
            self.vocabulario[token]
            for token in tokens
        )

        return dados.decode("utf-8")

    def salvar(self, caminho):
        caminho = Path(caminho)
        caminho.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        dados = {
            "versao": 1,
            "vocabulario": {
                str(k): list(v)
                for k, v in self.vocabulario.items()
            },
            "merges": [
                {
                    "par": list(par),
                    "novo_id": novo_id
                }
                for par, novo_id in self.merges
            ]
        }

        with caminho.open("w", encoding="utf-8") as arquivo:
            json.dump(
                dados,
                arquivo,
                ensure_ascii=False,
                indent=2
            )

    @classmethod
    def carregar(cls, caminho):
        with Path(caminho).open(
            "r",
            encoding="utf-8"
        ) as arquivo:
            dados = json.load(arquivo)

        if dados.get("versao") != 1:
            raise ValueError(
                "Versao de tokenizador nao suportada"
            )

        tokenizador = cls()

        tokenizador.vocabulario = {
            int(k): bytes(v)
            for k, v in dados["vocabulario"].items()
        }

        tokenizador.merges = [
            (
                tuple(item["par"]),
                item["novo_id"]
            )
            for item in dados["merges"]
        ]

        return tokenizador


if __name__ == "__main__":
    tokenizador = CygnusByteBPETokenizer()

    textos = [
        "Cygnus aprende linguagem.",
        "Cygnus aprende linguagem.",
        "A inteligencia artificial aprende.",
        "A inteligencia artificial aprende.",
        "O Transformer processa tokens.",
        "O Transformer processa tokens."
    ]

    tokenizador.treinar(
        textos,
        numero_merges=100
    )

    exemplo = "Olá, Cygnus! 🌟"

    ids = tokenizador.codificar(exemplo)
    reconstruido = tokenizador.decodificar(ids)

    print("=== CYGNUS BYTE-LEVEL BPE ===")
    print("Vocabulario:", tokenizador.tamanho_vocabulario)
    print("Texto original:", exemplo)
    print("Tokens:", ids)
    print("Texto reconstruido:", reconstruido)

    tokenizador.salvar(
        "checkpoints/cygnus_byte_bpe.json"
    )

    print("Tokenizador salvo com sucesso.")
