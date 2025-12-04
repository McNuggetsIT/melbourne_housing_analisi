# --- 2. Configurazione del Modello ---
# Usiamo LogisticRegression (già nota)
model = LogisticRegression(solver='liblinear') # liblinear va bene per dataset piccoli

# --- 3. Stratified K-Fold ---
# Vogliamo 5 round di validazione.
# Shuffle=True è fondamentale per mescolare i dati prima di tagliare.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- 4. Esecuzione della Cross-Validation ---
# scoring='f1': Usiamo F1-score perché l'accuratezza è inutile su dati sbilanciati
scores = cross_val_score(model, X, y, cv=cv, scoring='f1')

print("\n--- Risultati Cross-Validation (5 Folds) ---")
for i, score in enumerate(scores):
print(f"Fold {i+1}: F1-Score = {score:.4f}")

print(f"\n>> Performance Media: {scores.mean():.4f}")
print(f">> Stabilità (Deviazione Std): +/- {scores.std():.4f}")