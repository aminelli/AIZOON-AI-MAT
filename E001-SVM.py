from sklearn import svm

# Dataset di add3estramento
# [Altezza, peso, Nr Scarpe]
X = [
    [170, 70, 42],
    [180, 80, 44],
    [168, 65, 38],
    [155, 50, 36],
    [190, 95, 46],
    [160, 70, 42],
    [163, 60, 40]    
]

# Classi:
# 0 = Uomo
# 1 = Donna

# Le risposte corrette ai singoli elementi del dataset di addestramento
y = [
    0,
    0,
    1,
    1,
    0,
    0,
    1    
]

clf = svm.SVC()

clf.fit(X, y)

# Previsione

p = clf.predict([[155, 60, 37]])  # Donna

print("Risultato: ",p)

print("")


