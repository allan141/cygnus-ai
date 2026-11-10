# Cygnus AI - Treinamento Básico

# Exemplos: entrada -> resposta esperada

dados = [
    (0.0, 1.0),
    (1.0, 3.0),
    (2.0, 5.0),
    (3.0, 7.0),
    (4.0, 9.0),
]

# Parametros iniciais do modelo

peso = 0.0
vies = 0.0

# Taxa de aprendizado

taxa = 0.01

# Quantidade de repeticoes

epocas = 1000

for epoca in range(epocas):
    perda = 0.0
    grad_peso = 0.0
    grad_vies = 0.0

    for x, esperado in dados:
        previsao = peso * x + vies

        erro = previsao - esperado
        perda += erro ** 2

        # Gradientes calculados manualmente

        grad_peso += 2 * erro * x
        grad_vies += 2 * erro

    n = len(dados)

    perda /= n
    grad_peso /= n
    grad_vies /= n 

    # Atualizar os parametros

    peso -= taxa * grad_peso
    vies -= taxa * grad_vies

    if epoca % 100 == 0:
        print(
            f"Epoca {epoca:04d} | "
            f"Perda: {perda:.6f} | "
            f"Peso: {peso:.4f} | "
            f"Vies: {vies:.4f}"
        )

print("\n=== TREINAMENTO CONCLUIDO ===")
print(f"Peso aprendido: {peso:.4f}")
print(f"Vies aprendido: {vies:.4f}")

# Teste com um valor utilizado no treinamento

entrada = 10.0
resultado = peso * entrada + vies

print(f"\nEntrada: {entrada}")
print(f"Previsão: {resultado:.4f}")
print("Esperado: 21.0")
