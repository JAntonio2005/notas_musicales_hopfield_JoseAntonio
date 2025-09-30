# notas_hopfield.py
# ------------------------------------------------------------
# Red de Hopfield para reconocer patrones binarios (0/1).
# - Lee patrones desde una carpeta (cada archivo = 1 clase).
# - Acepta separadores ; , o espacios (maneja BOM de Excel).
# - Entrena por Hebb o Pseudoinversa (menos interferencia).
# - Permite probar archivo suelto o añadir ruido a una clase.
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

def find_labeled_path(folder, label):
    """Busca label con cualquier extensión (.txt/.csv) dentro de folder."""
    candidates = glob.glob(os.path.join(folder, f"{label}.*"))
    candidates = [p for p in candidates if p.lower().endswith((".txt", ".csv"))]
    return candidates[0] if candidates else None

# ---------- Red de Hopfield ----------
class Hopfield:
    def __init__(self, n):
        self.n = n
        self.W = np.zeros((n, n), dtype=np.float32)

    def train_hebb(self, patterns):
        """Hebb clásico: W = sum(p p^T) / n, diag=0"""
        self.W.fill(0.0)
        for p in patterns:
            ppT = np.outer(p.astype(np.float32), p.astype(np.float32))
            self.W += ppT
        np.fill_diagonal(self.W, 0.0)
        self.W /= self.n

    def train_pseudoinverse(self, patterns):
        """
        Proyección (pseudoinversa): W = X^T (X X^T)^-1 X, diag=0
        Reduce solapamiento entre memorias cuando P << N.
        """
        X = np.stack(patterns, axis=0).astype(np.float64)  # (P, N)
        G = X @ X.T                                        # (P, P)
        W = X.T @ np.linalg.pinv(G) @ X                    # (N, N)
        np.fill_diagonal(W, 0.0)
        self.W = W.astype(np.float32)

    def recall(self, x, steps=20):
        """Dinámica sincrónica: x <- sign(Wx) hasta converger o agotar pasos."""
        x = x.astype(np.float32).copy()
        for _ in range(steps):
            x_new = np.sign(self.W @ x)
            x_new[x_new == 0] = 1
            if np.array_equal(x_new, x):
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
        self.net = Hopfield(n=len(patterns[0]))
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
        return self.predict_vec(x, steps=steps)

# ---------- Utilidades de prueba ----------
def add_noise_to_vec(vec_pm1, prob=0.1, rng=None):
    """
    Recibe vector en {-1,+1}, lo convierte a {0,1}, voltea bits con prob,
    y devuelve nuevamente en {-1,+1}.
    """
    rng = np.random.default_rng(rng)
    v01 = (vec_pm1 + 1) // 2  # {-1,+1} -> {0,1}
    mask = rng.random(v01.size) < float(prob)
    v01_noisy = v01.copy()
    v01_noisy[mask] = 1 - v01_noisy[mask]
    v_pm1 = np.where(v01_noisy == 0, -1, 1).astype(np.int8)
    return v_pm1

def menu():
    print("\n=== Hopfield Notas Musicales ===")
    print("1) Entrenar (carpeta)")
    print("2) Listar clases entrenadas")
    print("3) Probar archivo (.txt/.csv)")
    print("4) Probar una clase con ruido")
    # print("5) Cambiar regla de entrenamiento (hebb/pseudoinverse)")
    print("0) Salir")

if __name__ == "__main__":
    folder = "dataset_notas"   # <-- cambia si tu carpeta se llama distinto
    clf = HopfieldClassifier(rule="pseudoinverse")  # por defecto, mejor separación

    trained = False

    while True:
        menu()
        op = input("Opción: ").strip()

        if op == "1":
            folder_in = input(f"Carpeta de dataset [{folder}]: ").strip()
            if folder_in:
                folder = folder_in
            try:
                clf.fit_from_folder(folder)
                trained = True
                print(f"✔ Entrenado con {len(clf.labels)} clases en '{folder}' usando '{clf.rule}'.")
                print("Clases:", ", ".join(clf.labels))
            except Exception as e:
                print("✖ Error al entrenar:", e)

        elif op == "2":
            if not trained:
                print("Primero entrena (opción 1).")
                continue
            print("Clases entrenadas:", ", ".join(clf.labels))
            print("Forma de cada patrón:", clf.shape, "(alto x ancho)")

        elif op == "3":
            if not trained:
                print("Primero entrena (opción 1).")
                continue
            path = input("Ruta del archivo a probar: ").strip()
            if not os.path.isfile(path):
                print("✖ No existe el archivo.")
                continue
            try:
                print("Probando con:", os.path.basename(path))
                label, dists, xr = clf.predict_file(path)
                print("\nPatrón recordado:")
                pretty_print(xr, clf.shape, on="██", off="  ")
                print("\nDistancias Hamming:")
                for lab, d in zip(clf.labels, dists):
                    print(f"  {lab}: {d}")
                print(f"\nPredicción: {label}")
            except Exception as e:
                print("✖ Error al predecir:", e)

        elif op == "4":
            if not trained:
                print("Primero entrena (opción 1).")
                continue
            print("Clases disponibles:", ", ".join(clf.labels))
            lab = input("Elige clase (exacta): ").strip()
            if lab not in clf.labels:
                # intenta localizar por archivo
                p = find_labeled_path(folder, lab)
                if p:
                    lab = os.path.splitext(os.path.basename(p))[0]
                else:
                    print("✖ Clase no encontrada.")
                    continue

            # toma el vector limpio de esa clase
            idx = clf.labels.index(lab)
            clean_vec = clf.clean[idx]
            try:
                pct = float(input("Porcentaje de ruido (0-100, ej. 12): ").strip() or "10")
                prob = max(0.0, min(1.0, pct / 100.0))
            except:
                prob = 0.1

            noisy_vec = add_noise_to_vec(clean_vec, prob=prob)
            print(f"\nMostrando patrón con {int(prob*100)}% de ruido (antes de recuperación):")
            pretty_print(noisy_vec, clf.shape, on="██", off="  ")

            pred, dists, xr = clf.predict_vec(noisy_vec)
            print("\nPatrón recordado (tras recuperación):")
            pretty_print(xr, clf.shape, on="██", off="  ")

            print("\nDistancias Hamming (al estado recordado):")
            for lab2, d in zip(clf.labels, dists):
                print(f"  {lab2}: {d}")
            print(f"\nPredicción: {pred}")

        elif op == "5":
            new_rule = input("Regla (hebb/pseudoinverse) [pseudoinverse]: ").strip().lower() or "pseudoinverse"
            if new_rule not in ("hebb", "pseudoinverse"):
                print("✖ Opción no válida.")
            else:
                clf.rule = new_rule
                trained = False
                print(f"✔ Regla configurada a '{new_rule}'. Vuelve a entrenar (opción 1).")

        elif op == "0":
            break
        else:
            print("Elige una opción válida.")
