def espiral_numerica(n=7):
    """
    Imprime una matriz n x n con números del 1 a n^2
    acomodados en espiral (sentido horario).
    n debe ser impar para verse mejor (pero acepta pares).
    """
    matriz = [[0]*n for _ in range(n)]
    top, bottom, left, right = 0, n-1, 0, n-1
    num = 1

    while left <= right and top <= bottom:
        # → fila superior
        for c in range(left, right+1):
            matriz[top][c] = num; num += 1
        top += 1

        # ↓ columna derecha
        for r in range(top, bottom+1):
            matriz[r][right] = num; num += 1
        right -= 1

        if top <= bottom:
            # ← fila inferior
            for c in range(right, left-1, -1):
                matriz[bottom][c] = num; num += 1
            bottom -= 1

        if left <= right:
            # ↑ columna izquierda
            for r in range(bottom, top-1, -1):
                matriz[r][left] = num; num += 1
            left += 1

    # Mostrar bonito, con ancho fijo según el número más grande
    ancho = len(str(n*n))
    for fila in matriz:
        print(" ".join(f"{x:>{ancho}}" for x in fila))

# Ejemplo
espiral_numerica(7)
