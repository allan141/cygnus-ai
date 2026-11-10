# Primeiro neuronio artificial

def neuronio(entrada, peso, vies):
    resultado = entrada * peso + vies
    return resultado


x = 2.0
w = 3.0
b = 1.0

saida = neuronio(x, w, b)

print("=== CYGNUS AI ===")
print(f"Entrada: {x}")
print(f"Peso: {w}")          
print(f"Viés: {b}")
print(f"Saída: {saida}")
