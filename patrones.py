x1_01_str = "1110"
x2_01_str = "0001"

A_bip_str  = "-1,-1,-1,1"  

print("X1 (0/1):", x1_01_str)
print("X2 (0/1):", x2_01_str)

x1_01 = []
x2_01 = []

i = 0
while i < len(x1_01_str):
    if x1_01_str[i] == '1': x1_01.append(1)
    else:                   x1_01.append(0)
    i += 1

i = 0
while i < len(x2_01_str):
    if x2_01_str[i] == '1': x2_01.append(1)
    else:                   x2_01.append(0)
    i += 1


x1 = []
x2 = []
i = 0
while i < len(x1_01):
    if x1_01[i] == 0: x1.append(-1)
    else:             x1.append(1)
    i += 1

i = 0
while i < len(x2_01):
    if x2_01[i] == 0: x2.append(-1)
    else:             x2.append(1)
    i += 1

norm = ""
i = 0
while i < len(A_bip_str):
    ch = A_bip_str[i]
    if ch == ',':
        norm += ' '
    else:
        norm += ch
    i += 1

tokens = []
tmp = ""
i = 0
while i < len(norm):
    ch = norm[i]
    if ch == ' ' or ch == '\t' or ch == '\n':
        if len(tmp) > 0:
            tokens.append(tmp)
            tmp = ""
    else:
        tmp += ch
    i += 1
if len(tmp) > 0:
    tokens.append(tmp)

A = []
i = 0
while i < len(tokens):
    t = tokens[i]
    if t == "1":
        A.append(1)
    elif t == "-1":
        A.append(-1)
    else:
        A.append(-1)  
    i += 1

n = len(x1)
if len(A) < n:
    i = len(A)
    while i < n:
        A.append(-1)
        i += 1
elif len(A) > n:
    while len(A) > n:
        A.pop()

A_01 = []
i = 0
while i < n:
    if A[i] == 1: A_01.append(1)
    else:         A_01.append(0)
    i += 1

print("A  (0/1):", ''.join(str(d) for d in A_01))
print()
print("X1 (+/-1):", x1)
print("X2 (+/-1):", x2)
print("A  (+/-1):", A)

W1 = []
i = 0
while i < n:
    fila = []
    j = 0
    while j < n:
        fila.append(x1[i] * x1[j])
        j += 1
    W1.append(fila)
    i += 1

W2 = []
i = 0
while i < n:
    fila = []
    j = 0
    while j < n:
        fila.append(x2[i] * x2[j])
        j += 1
    W2.append(fila)
    i += 1

W = []
i = 0
while i < n:
    fila = []
    j = 0
    while j < n:
        fila.append(W1[i][j] + W2[i][j])
        j += 1
    W.append(fila)
    i += 1

i = 0
while i < n:
    W[i][i] = 0
    i += 1

print("\nMatriz W:")
i = 0
while i < n:
    print(W[i])
    i += 1

max_iters = 10
estado = []
i = 0
while i < n:
    estado.append(A[i])
    i += 1

print("\n=== Iteraciones ===\n")

it = 0
while it < max_iters:
    net = []
    i = 0
    while i < n:
        suma = 0
        j = 0
        while j < n:
            suma += W[i][j] * estado[j]
            j += 1
        net.append(suma)
        i += 1

    nuevo = []
    i = 0
    while i < n:
        if net[i] > 0:
            nuevo.append(1)
        elif net[i] < 0:
            nuevo.append(-1)
        else:
            nuevo.append(estado[i])
        i += 1

    print("Iteración", it + 1)
    print("net =", net)
    print("U  =", nuevo)

    iguales = True
    i = 0
    while i < n:
        if nuevo[i] != estado[i]:
            iguales = False
            break
        i += 1
    if iguales:
        estado = nuevo
        break

    estado = nuevo
    it += 1

print("U* (+/-1):", estado)

estado_01_out = []
i = 0
while i < n:
    if estado[i] == 1: estado_01_out.append(1)
    else:               estado_01_out.append(0)
    i += 1
#print("U* (0/1): ", estado_01_out)