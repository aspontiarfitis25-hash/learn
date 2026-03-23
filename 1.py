import numpy as np
from collections import Counter
np.random.seed(42)
def generate_patient(diag, rng):
    if diag == "Грип":
        t = rng.uniform(38.5, 40.5)
        c = rng.choice(np.array([0,1]), p=[0.2,0.8])
        s = rng.choice(np.array([0,1]), p=[0.4,0.6])
        r = rng.choice(np.array([0,1]), p=[0.5,0.5])
        f = rng.choice(np.array([0,1]), p=[0.1,0.9])
        m = rng.choice(np.array([0,1]), p=[0.1,0.9])
        h = rng.choice(np.array([0,1]), p=[0.2,0.8])
        b = rng.choice(np.array([0,1]), p=[0.7,0.3])
        d = rng.integers(1, 6)
        a = rng.integers(5, 80)
    elif diag == "Бронхіт":
        t = rng.uniform(37.5, 38.8)
        c = rng.choice(np.array([0,1]), p=[0.1,0.9])
        s = rng.choice(np.array([0,1]), p=[0.6,0.4])
        r = rng.choice(np.array([0,1]), p=[0.5,0.5])
        f = rng.choice(np.array([0,1]), p=[0.4,0.6])
        m = rng.choice(np.array([0,1]), p=[0.6,0.4])
        h = rng.choice(np.array([0,1]), p=[0.5,0.5])
        b = rng.choice(np.array([0,1]), p=[0.3,0.7])
        d = rng.integers(5, 14)
        a = rng.integers(5, 80)
    elif diag == "Застуда":
        t = rng.uniform(36.5, 38.0)
        c = rng.choice(np.array([0,1]), p=[0.3,0.7])
        s = rng.choice(np.array([0,1]), p=[0.2,0.8])
        r = rng.choice(np.array([0,1]), p=[0.1,0.9])
        f = rng.choice(np.array([0,1]), p=[0.5,0.5])
        m = rng.choice(np.array([0,1]), p=[0.7,0.3])
        h = rng.choice(np.array([0,1]), p=[0.4,0.6])
        b = rng.choice(np.array([0,1]), p=[0.8,0.2])
        d = rng.integers(2, 8)
        a = rng.integers(5, 80)
    else:
        t = rng.uniform(36.0, 37.2)
        c = rng.choice(np.array([0,1]), p=[0.9,0.1])
        s = rng.choice(np.array([0,1]), p=[0.9,0.1])
        r = rng.choice(np.array([0,1]), p=[0.9,0.1])
        f = rng.choice(np.array([0,1]), p=[0.8,0.2])
        m = rng.choice(np.array([0,1]), p=[0.9,0.1])
        h = rng.choice(np.array([0,1]), p=[0.8,0.2])
        b = rng.choice(np.array([0,1]), p=[0.95,0.05])
        d = rng.integers(1, 3)
        a = rng.integers(5, 80)
    return [t, c, s, r, f, m, h, b, d, a]
rng = np.random.default_rng(42)
n_per_class = 200
data = []
labels = []
for diag in ["Грип", "Бронхіт", "Застуда", "Здоровий"]:
    for _ in range(n_per_class):
        data.append(generate_patient(diag, rng))
        labels.append(diag)
noise_rate = 0.07
n = len(data)
noise_idx = rng.choice(n, int(n * noise_rate), replace=False)
all_classes = ["Грип", "Бронхіт", "Застуда", "Здоровий"]
for idx in noise_idx:
    current = labels[idx]
    others = [c for c in all_classes if c != current]
    labels[idx] = rng.choice(others)
X = np.array(data)
y = np.array(labels)
classes = sorted(list(set(y)))
class_to_idx = {c: i for i, c in enumerate(classes)}
y_encoded = np.array([class_to_idx[c] for c in y])
print("ДАТАСЕТ v2 — покращена версія")
print("=" * 60)
print(f"Пацієнтів: {n} | Ознак: 10 | Класів: 4")
print(f"Ознаки: Температура, Кашель, Горло, Нежить, Втома,")
print(f"        Біль у м'язах, Головний біль, Задишка, Дні симптомів, Вік")
print(f"\nРозподіл (збалансований):")
for cls, cnt in sorted(Counter(y).items()):
    bar = "█" * (cnt // 5)
    print(f"  {cls:>12}: {cnt:>3}  {bar}")
np.random.seed(0)
indices = np.random.permutation(n)
split = int(0.8 * n)
train_idx = indices[:split]
test_idx = indices[split:]
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
def gini(y):
    if len(y) == 0:
        return 0
    counts = np.bincount(y, minlength=len(classes))
    probs = counts / len(y)
    return 1 - np.sum(probs ** 2)
def best_split(X, y):
    best_gini = float('inf')
    best_feat = None
    best_thresh = None
    for feat in range(X.shape[1]):
        thresholds = np.unique(X[:, feat])
        if len(thresholds) > 40:
            thresholds = np.percentile(X[:, feat], np.linspace(5, 95, 25))
        for thresh in thresholds:
            left = y[X[:, feat] <= thresh]
            right = y[X[:, feat] > thresh]
            if len(left) == 0 or len(right) == 0:
                continue
            g = (len(left) * gini(left) + len(right) * gini(right)) / len(y)
            if g < best_gini:
                best_gini = g
                best_feat = feat
                best_thresh = thresh
    return best_feat, best_thresh
def build_tree(X, y, depth=0, max_depth=6):
    if len(set(y)) == 1 or depth == max_depth or len(y) < 12:
        return {'leaf': True, 'prediction': np.bincount(y, minlength=len(classes)).argmax()}
    feat, thresh = best_split(X, y)
    if feat is None:
        return {'leaf': True, 'prediction': np.bincount(y, minlength=len(classes)).argmax()}
    left_mask = X[:, feat] <= thresh
    right_mask = ~left_mask
    return {
        'leaf': False, 'feature': feat, 'threshold': thresh,
        'left': build_tree(X[left_mask], y[left_mask], depth+1, max_depth),
        'right': build_tree(X[right_mask], y[right_mask], depth+1, max_depth)
    }
def predict_one(node, x):
    if node['leaf']:
        return node['prediction']
    return predict_one(node['left'], x) if x[node['feature']] <= node['threshold'] else predict_one(node['right'], x)
def predict(tree, X):
    return np.array([predict_one(tree, x) for x in X])
def build_forest(X, y, n_trees=100, max_depth=6, max_features=7):
    trees = []
    n_samples = len(X)
    for _ in range(n_trees):
        boot_idx = np.random.choice(n_samples, n_samples, replace=True)
        X_boot = X[boot_idx]
        y_boot = y[boot_idx]
        feat_idx = np.random.choice(X.shape[1], max_features, replace=False)
        tree = build_tree(X_boot[:, feat_idx], y_boot, max_depth=max_depth)
        trees.append((tree, feat_idx))
    return trees
def predict_forest(trees, X):
    all_preds = np.array([predict(tree, X[:, feat_idx]) for tree, feat_idx in trees])
    return np.array([np.bincount(all_preds[:, i], minlength=len(classes)).argmax() for i in range(X.shape[0])])
def predict_forest_proba(trees, X_row):
    votes = np.zeros(len(classes))
    for tree, feat_idx in trees:
        votes[predict_one(tree, X_row[feat_idx])] += 1
    return votes / len(trees)
print("\nНавчання Random Forest (100 дерев)...")
np.random.seed(1)
forest = build_forest(X_train, y_train, n_trees=100, max_depth=6, max_features=7)
y_pred_train = predict_forest(forest, X_train)
y_pred_test = predict_forest(forest, X_test)
acc_train = np.mean(y_pred_train == y_train)
acc_test = np.mean(y_pred_test == y_test)
print(f"\nОЦІНКА МОДЕЛІ")
print("=" * 55)
print(f"Навчальна вибірка: {len(X_train)} | Тестова: {len(X_test)}")
print(f"\nAccuracy навчальні: {acc_train:.1%}")
print(f"Accuracy тестові:   {acc_test:.1%}")
print(f"Різниця:            {abs(acc_train-acc_test):.1%}")
print(f"\nМатриця плутанини:")
print(f"{'':>12}", end="")
for c in classes:
    print(f"{c:>12}", end="")
print()
for i, actual in enumerate(classes):
    print(f"{actual:>12}", end="")
    for j in range(len(classes)):
        print(f"{np.sum((y_test==i)&(y_pred_test==j)):>12}", end="")
    print()
print(f"\nМетрики по класах:")
print(f"{'Клас':>12} {'Precision':>10} {'Recall':>10} {'F1':>8} {'Підтримка':>12}")
print("-" * 55)
f1_scores = []
for i, cls in enumerate(classes):
    tp = np.sum((y_test==i)&(y_pred_test==i))
    fp = np.sum((y_test!=i)&(y_pred_test==i))
    fn = np.sum((y_test==i)&(y_pred_test!=i))
    pr = tp/(tp+fp) if (tp+fp)>0 else 0
    rc = tp/(tp+fn) if (tp+fn)>0 else 0
    f1 = 2*pr*rc/(pr+rc) if (pr+rc)>0 else 0
    f1_scores.append(f1)
    print(f"{cls:>12} {pr:>10.2f} {rc:>10.2f} {f1:>8.2f} {np.sum(y_test==i):>12}")
print(f"{'Середнє':>12} {'':>10} {'':>10} {np.mean(f1_scores):>8.2f}")
print(f"\n{'='*55}")
print("ПРОГНОЗ ДЛЯ НОВОГО ПАЦІЄНТА")
print("="*55)
def get_float(prompt, lo, hi):
    while True:
        try:
            v = float(input(prompt))
            if lo <= v <= hi:
                return v
            print(f"  Введіть від {lo} до {hi}")
        except ValueError:
            print("  Введіть число")
def get_binary(prompt):
    while True:
        v = input(f"{prompt} (0=Ні / 1=Так): ").strip()
        if v in ("0","1"):
            return int(v)
        print("  Введіть 0 або 1")
def get_int(prompt, lo, hi):
    while True:
        try:
            v = int(input(prompt))
            if lo <= v <= hi:
                return v
            print(f"  Введіть від {lo} до {hi}")
        except ValueError:
            print("  Введіть ціле число")
t = get_float("Температура тіла (36.0-40.5): ", 36.0, 40.5)
c = get_binary("Кашель")
s = get_binary("Біль у горлі")
r = get_binary("Нежить")
f = get_binary("Втома")
m = get_binary("Біль у м'язах")
h = get_binary("Головний біль")
b = get_binary("Задишка")
d = get_int("Скільки днів симптоми (1-14): ", 1, 14)
a = get_int("Вік пацієнта (5-80): ", 5, 80)
patient = np.array([t, c, s, r, f, m, h, b, d, a])
proba = predict_forest_proba(forest, patient)
pred_idx = np.argmax(proba)
print(f"\nРЕЗУЛЬТАТ")
print(f"{'='*40}")
print(f"Діагноз:     {classes[pred_idx]}")
print(f"Впевненість: {proba[pred_idx]:.0%}")
print(f"\nІмовірності по класах:")
for i, cls in enumerate(classes):
    bar = "█" * int(proba[i] * 40)
    print(f"  {cls:>12}: {proba[i]:>5.1%}  {bar}")