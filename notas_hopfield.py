# notas_hopfield.py
# ------------------------------------------------------------
# Red de Hopfield para reconocer patrones binarios (0/1).
# ------------------------------------------------------------
import os
import re
import glob
import numpy as np

# ---------- Utilidades de IO ----------
def read_pattern(path, sep=None):
    """
    Lee patrón 0/1 desde .txt/.csv (maneja BOM).
    Acepta ; , espacios o tabs por defecto.
    Devuelve:
      vec en {-1,+1} y shape (rows, cols).
    """
    with open(path, "r", encoding="utf-8-sig") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    grid = []
    for ln in lines:
        if sep is None:
            ln = re.sub(r"[;,\t ]+", " ", ln)  # normaliza separadores
            parts = ln.split()
        else:
            parts = ln.split(sep)
        try:
            row = [int(x) for x in parts]
        except ValueError as e:
            raise ValueError(f"Error parseando '{path}'. "
                             f"¿Hay caracteres distintos de 0/1?") from e
        grid.append(row)

    arr01 = np.array(grid, dtype=np.int8)
    if not np.isin(arr01, [0, 1]).all():
        raise ValueError(f"El archivo {path} contiene valores distintos de 0/1.")
    arr = np.where(arr01 == 0, -1, 1).astype(np.int8)  # 0->-1, 1->+1
    shape = arr.shape
    vec = arr.reshape(-1)
    return vec, shape

def pretty_print(vec, shape, on="██", off="  "):
    """Imprime -1/1 como bloques (on/off) para verlo claro en consola."""
    g = vec.reshape(shape)
    for r in g:
        print("".join(on if v == 1 else off for v in r))

# ---------- Red de Hopfield ----------
class Hopfield:
    def __init__(self, n, shape=None):
        self.n = n
        self.W = np.zeros((n, n), dtype=np.float32)
        self.shape = shape  # Necesario para mostrar iteraciones

    def train_hebb(self, patterns):
        """Hebb clásico: W = sum(p p^T) / n, diag=0"""
        self.W.fill(0.0)
        for p in patterns:
            ppT = np.outer(p.astype(np.float32), p.astype(np.float32))
            self.W += ppT
        np.fill_diagonal(self.W, 0.0)
        self.W /= self.n

    def train_pseudoinverse(self, patterns):
        """Proyección (pseudoinversa): W = X^T (X X^T)^-1 X, diag=0"""
        X = np.stack(patterns, axis=0).astype(np.float64)  # (P, N)
        G = X @ X.T                                        # (P, P)
        W = X.T @ np.linalg.pinv(G) @ X                    # (N, N)
        np.fill_diagonal(W, 0.0)
        self.W = W.astype(np.float32)

    def recall(self, x, steps=20):
        """Dinámica sincrónica: x <- sign(Wx) hasta converger o agotar pasos."""
        x = x.astype(np.float32).copy()
        print("\nIteraciones para encontrar patrón:")
        for i in range(steps):
            x_new = np.sign(self.W @ x)
            x_new[x_new == 0] = 1
            print(f"\nIteración {i+1}:")
            pretty_print(x_new, self.shape, on="██", off="  ")
            if np.array_equal(x_new, x):
                print(f"\nConvergió después de {i+1} iteraciones.")
                break
            x = x_new
        return x.astype(np.int8)

# ---------- Clasificador ----------
class HopfieldClassifier:
    def __init__(self, rule="pseudoinverse"):
        self.labels = []
        self.clean = []
        self.shape = None
        self.net = None
        self.rule = rule  # "hebb" o "pseudoinverse"

    def fit_from_folder(self, folder, sep=None):
        files = [f for f in os.listdir(folder)
                 if f.lower().endswith((".txt", ".csv"))]
        if not files:
            raise RuntimeError(f"No se encontraron .txt/.csv en {folder}")
        files.sort()
        patterns, labels = [], []
        shape_ref = None
        for fname in files:
            path = os.path.join(folder, fname)
            v, shape = read_pattern(path, sep)
            if shape_ref is None:
                shape_ref = shape
            elif shape != shape_ref:
                raise ValueError(f"Formas distintas: {shape} en {fname} vs {shape_ref}")
            patterns.append(v)
            labels.append(os.path.splitext(fname)[0])
        self.labels = labels
        self.clean = patterns
        self.shape = shape_ref
        self.net = Hopfield(n=len(patterns[0]), shape=shape_ref)
        if self.rule == "hebb":
            self.net.train_hebb(self.clean)
        else:
            self.net.train_pseudoinverse(self.clean)

    def predict_vec(self, vec, steps=20):
        xr = self.net.recall(vec, steps=steps)
        dists = [int(np.sum(xr != p)) for p in self.clean]
        idx = int(np.argmin(dists))
        return self.labels[idx], dists, xr

    def predict_file(self, path, sep=None, steps=20):
        x, shape = read_pattern(path, sep)
        if shape != self.shape:
            raise ValueError(f"Forma distinta al entrenamiento: {shape} vs {self.shape}")
        return self.predict_vec(x, steps=steps), x

# ---------- Ejecución automática ----------
if __name__ == "__main__":
    folder = "dataset_notas"   # Carpeta con los patrones
    clf = HopfieldClassifier(rule="pseudoinverse")  # regla por defecto

    # Entrenamiento
    try:
        clf.fit_from_folder(folder)
        print(f"✔ Entrenado con {len(clf.labels)} clases en '{folder}' usando '{clf.rule}'.")
        print("Clases:", ", ".join(clf.labels))
    except Exception as e:
        print("✖ Error al entrenar:", e)
        exit(1)

    # Prueba con archivo
    test_file = "../test_pattern2.txt"  # archivo de prueba
    #test_file = "dataset_notas/bemol.txt"  # archivo de prueba
    try:
        (label, dists, xr), x_input = clf.predict_file(test_file)
        print("\nPatrón ingresado:")
        pretty_print(x_input, clf.shape, on="██", off="  ")

        print("\nPatrón recordado final:")
        pretty_print(xr, clf.shape, on="██", off="  ")
        print("\nDistancias Hamming:")
        for lab, d in zip(clf.labels, dists):
            print(f"  {lab}: {d}")
        print(f"\nPredicción: {label}")
    except Exception as e:
        print("✖ Error al predecir:", e)
