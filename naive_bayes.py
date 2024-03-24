import numpy as np
import time
import math

class NaiveBayesMultinomial:
    def fit(self, X, y):
        # Obtener las clases únicas y contar cuántas veces aparece cada una en y
        self.classes, counts = np.unique(y, return_counts=True)
        #print(self.classes)
        #print(counts)
        # Calcular las probabilidades a priori de cada clase
        self.class_probs = counts / len(y)
        #print(self.class_probs)

        # Calcular las probabilidades condicionales de cada característica dado cada clase
        self.feature_probs = {}
        for c in self.classes:
            X_c = X[y == c]  # Filtrar X para obtener solo las instancias de la clase c
            #print(X_c)
            total_count = np.sum([len(x) for x in X_c])  # Contar el total de ocurrencias de todas las características en la clase c
            self.feature_probs[c] = {}  # Inicializar un diccionario para almacenar las probabilidades condicionales de la clase c
            for feature_idx in range(X.shape[1]):
                # Contar cuántas veces aparece cada valor de la característica en la clase c y aplicar Laplace smoothing
                feature_values, feature_counts = np.unique([x[feature_idx] for x in X_c], return_counts=True)
                #print(feature_values)
                #print(feature_counts)
                self.feature_probs[c][feature_idx] = dict(zip(feature_values, (feature_counts + 1) / (total_count + len(feature_values))))  # Laplace smoothing
                #print(self.feature_probs[c])
    def predict(self, X):
        # Hacer predicciones para cada instancia en X
        predictions = []
        for x in X:
            probs = []
            for idx, c in enumerate(self.classes):
                #print(idx)
                #print(self.class_probs)
                prob = self.class_probs[idx]  # Inicializar la probabilidad con la probabilidad a priori de la clase c
                for feature_idx, feature_value in enumerate(x):
                    if feature_value in self.feature_probs[c][feature_idx]:
                        prob *= self.feature_probs[c][feature_idx][feature_value]  # Multiplicar por la probabilidad condicional de la característica dado c
                    else:
                        # Si se encuentran caracteristicas no vistas, se
                        prob *= 1 / (len(self.feature_probs[c][feature_idx]) + len(self.classes))
                        #print("No se encontro")
                probs.append(prob)  # Agregar la probabilidad calculada para c
            predictions.append(self.classes[np.argmax(probs)])  # Agregar la clase con la mayor probabilidad como predicción
        return predictions

# Ejemplo de datos de entrenamiento y etiquetas
X_train = np.array([
    ['Soleado', 'Caliente', 'Alta', 'Débil'],
    ['Soleado', 'Caliente', 'Alta', 'Fuerte'],
    ['Nublado', 'Caliente', 'Alta', 'Débil'],
    ['Lluvioso', 'Templado', 'Alta', 'Débil'],
    ['Lluvioso', 'Frío', 'Normal', 'Débil'],
    ['Lluvioso', 'Frío', 'Normal', 'Fuerte'],
    ['Nublado', 'Frío', 'Normal', 'Fuerte'],
    ['Soleado', 'Templado', 'Alta', 'Débil'],
    ['Lluvioso', 'Templado', 'Normal', 'Débil'],
    ['Lluvioso', 'Templado', 'Normal', 'Fuerte']
])
y_train = np.array(['No', 'No', 'Sí', 'Sí', 'Sí', 'No', 'Sí', 'No', 'Sí', 'Sí'])

# Ejemplo de datos de prueba
X_test = np.array([
    ['Soleado', 'Templado', 'Alta', 'Fuerte'],
    ['Nublado', 'Caliente', 'Normal', 'Débil'],
    ['Soleado', 'Templado', 'Normal', 'Débil']
])

times = []
# Inicializar y entrenar el clasificador Naive Bayes Multinomial
nb_multinomial = NaiveBayesMultinomial()
start= time.time()
nb_multinomial.fit(X_train, y_train)
end=time.time()
times.append(end-start)

# Hacer predicciones
start= time.time()
predictions = nb_multinomial.predict(X_test)
end=time.time()
times.append(end-start)
print("Predicciones:", predictions)
print(times)

def entropy(data, target_col):
    counts = data[target_col].value_counts()
    entropy = 0
    total_samples = len(data)

    for label in counts:
        probability = label / total_samples
        entropy -= probability * math.log2(probability)

    return entropy

def information_gain(data,feature_col,target_col): 
    total_entropy=entropy(data,target_col)
    feature_values=data[feature_col].unique()
    weighted_Entropy =0
    
    for value in feature_values:
        subset=data[data[feature_col]==value]
        subset_entropy=entropy(subset,target_col)
        weight=len(subset)/len(data)
        weighted_Entropy+=weight*subset_entropy
        
    return total_entropy-weighted_Entropy

def get_best_feature(data, features, target_col):
    information_gains={feature:information_gain(data,feature,target_col) for feature in features}
    return max(information_gains, key=information_gains.get)

def id3(data, original_data, features,  target_col, parent_value ):
    # Si todos los ejemplos tienen la misma clase, retornar esa clase
    if len(data[target_col].unique()) == 1:
        return data[target_col].iloc[0]

    # Si no quedan atributos, retornar la clase mayoritaria
    if len(features) == 0:
        return original_data[target_col].mode().iloc[0]

    # Calcular la entropía inicial
    entropy_init = entropy(data, target_col)

    # Inicializar mejor atributo
    best_feature = get_best_feature(data, features, target_col)
    tree={best_feature:{}}

    
    # Dividir el conjunto de datos en subconjuntos según el mejor atributo
    subsets = {}
    for value in data[best_feature].unique():
        subset = data[data[best_feature] == value]
        remaining_features = [f for f in features if f!=best_feature]
        subtree =id3(subset, original_data, remaining_features, target_col, value)
        tree[best_feature][value]=subtree
    return tree

# Ejemplo de uso
data = pd.DataFrame({
    'Outlook': ['Soleado', 'Soleado', 'Nublado', 'Lluvioso', 'Lluvioso', 'Lluvioso', 'Nublado', 'Soleado', 'Lluvioso', 'Lluvioso'],
    'Temperature': ['Caliente', 'Caliente', 'Caliente', 'Templado', 'Frío', 'Frío', 'Frío', 'Templado', 'Templado', 'Templado'],
    'Humidity': ['Alta', 'Alta', 'Alta', 'Alta', 'Normal', 'Normal', 'Normal', 'Alta', 'Normal', 'Normal'],
    'Windy': ['Débil', 'Fuerte', 'Débil', 'Débil', 'Débil', 'Fuerte', 'Fuerte', 'Débil', 'Débil', 'Fuerte'],
    'Play Golf': ['No', 'No', 'Sí', 'Sí', 'Sí', 'No', 'Sí', 'No', 'Sí', 'Sí']
})
features=[ 'Outlook', 'Temperature', 'Humidity', 'Windy' ]
target_col='Play Golf'
start= time.time()
tree = id3(data, data, features, target_col, None )
end=time.time()
times.append(end-start)
def predict(example,tree):
    feature= list(tree.keys())[0]
    value= example[feature]
    subtree= tree[feature][value]
    
    if isinstance(subtree,dict):
        return predict(example, subtree)
    else:
        return subtree
    
test_example= { 'Outlook':'Soleado','Temperature':'Templado', 'Humidity': 'Alta', 'Windy': 'Fuerte' }
test_example2= { 'Outlook':'Nublado','Temperature':'Caliente', 'Humidity': 'Normal', 'Windy': 'Débil' }
test_example3= { 'Outlook':'Soleado','Temperature':'Templado', 'Humidity': 'Normal', 'Windy': 'Débil' }
start= time.time()
print("Prediction 1: ", predict(test_example,tree.copy()))
print("Prediction 2: ", predict(test_example2,tree.copy()))
print("Prediction 3: ", predict(test_example3,tree.copy()))
end=time.time()
times.append(end-start)
print(times)
