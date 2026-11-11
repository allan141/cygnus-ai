
# Cygnus AI - Fase 3
# Primeiro tokenizador proprio, baseado em caracteres

class CygnusTokenizer:
    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}

    def treinar(self, texto):
        # Descobrir os caracteres existentes
        caracteres = sorted(set(texto))

        # Reservar 0 para caracteres desconhecidos
        self.char_to_id = {"<UNK>": 0}
        self.id_to_char = {0: "<UNK>"}

        for indice, caractere in enumerate(caracteres, start=1):
            self.char_to_id[caractere] = indice
            self.id_to_char[indice] = caractere

    def codificar(self, texto):
        return [
            self.char_to_id.get(caractere, 0)
            for caractere in texto
        ]

    def decodificar(self, tokens):
        return "".join(
            self.id_to_char.get(token, "<UNK>")
            for token in tokens
        )


if __name__ == "__main__":
    texto_treinamento = (
        "Cygnus AI é uma inteligência artificial "
        "criada e treinada do zero."
    )

    tokenizer = CygnusTokenizer()
    tokenizer.treinar(texto_treinamento)

    frase = "Cygnus AI"

    tokens = tokenizer.codificar(frase)
    texto_recuperado = tokenizer.decodificar(tokens)

    print("=== CYGNUS TOKENIZER ===")
    print("Texto original:", frase)
    print("Tokens:", tokens)
    print("Texto recuperado:", texto_recuperado)
    print("Tamanho do vocabulário:", len(tokenizer.char_to_id))
