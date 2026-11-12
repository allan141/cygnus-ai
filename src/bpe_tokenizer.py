# Cygnus AI - Fase 4
# Tokenizador BPE desenvolvido do zero

import json
from pathlib import Path
from collections import Counter


class CygnusBPETokenizer:
    def __init__(self):
        self.vocabulario = {}
        self.merges = []

    def treinar(self, texto, numero_merges=20):
        # Começar com um token para cada caractere
        tokens = list(texto)

        # Reiniciar as combinações aprendidas
        self.merges = []

        for _ in range(numero_merges):
            # Contar pares de tokens vizinhos
            pares = Counter(zip(tokens, tokens[1:]))

            if not pares:
                break

            # Encontrar o par mais frequente
            par_mais_frequente, frequencia = pares.most_common(1)[0]

            # Parar se o par aparecer menos de duas vezes
            if frequencia < 2:
                break

            # Criar um novo token
            novo_token = "".join(par_mais_frequente)

            # Registrar a combinação
            self.merges.append(par_mais_frequente)

            # Substituir o par pelo novo token
            tokens = self._unir_par(
                tokens,
                par_mais_frequente,
                novo_token
            )

        # Construir o vocabulário
        unidades = set(texto)

        for par in self.merges:
            unidades.add("".join(par))

        self.vocabulario = {
            token: indice
            for indice, token in enumerate(
                sorted(unidades),
                start=1
            )
        }
    def _unir_par(self, tokens, par, novo_token):
        resultado = []
        i = 0

        while i < len(tokens):
            if (
                i + 1 < len(tokens)
                and tokens[i] == par[0]
                and tokens[i + 1] == par[1]
            ):
                resultado.append(novo_token)
                i += 2
            else:
                resultado.append(tokens[i])
                i += 1

        return resultado

    def codificar(self, texto):
        tokens = list(texto)

        for par in self.merges:
            tokens = self._unir_par(
                tokens, par, "".join(par)
            )

        return [
            self.vocabulario.get(token, 0)
            for token in tokens
        ]
    def decodificar(self, ids):
        inverso = {
            indice: token 
            for token, indice in self.vocabulario.items()
        }

        return "".join(
            inverso.get(indice, "<UNK>")
            for indice in ids
        )

    
    def salvar(self, caminho):
        dados = {
            "versao": 1,
            "vocabulario": self.vocabulario,
            "merges": [list(par) for par in self.merges]
        }

        caminho = Path(caminho)
        caminho.parent.mkdir(parents=True, exist_ok=True)

        with caminho.open("w", encoding="utf-8") as arquivo:
            json.dump(
                dados,
                arquivo,
                ensure_ascii=False,
                indent=4
            )

    def carregar(self, caminho):
        caminho = Path(caminho)

        with caminho.open("r", encoding="utf-8") as arquivo:
            dados = json.load(arquivo)

        if dados.get("versao") != 1:
            raise ValueError("Versão de tokenizador incompatível")

        self.vocabulario = dados["vocabulario"]
        self.merges = [tuple(par) for par in dados["merges"]]

if __name__ == "__main__":
    texto = (
        "casa casa casa casaco "
        "Cygnus Cygnus Cygnus "
        "inteligencia artificial inteligência artificial "
    )

    tokenizer = CygnusBPETokenizer()
    tokenizer.treinar(texto, numero_merges=20)

    frase = "casa Cygnus"
    ids = tokenizer.codificar(frase)

    print("=== CYGNUS BPE TOKENIZER ===")
    print("texto:", frase)
    print("Tokens:", ids)
    print("Recuperado:", tokenizer.decodificar(ids))
    print("Merges aprendidos:", tokenizer.merges)
    print("Tamanho do vocabulário:", len(tokenizer.vocabulario))