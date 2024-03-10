Pseudocode:
Procedimiento qSort(arr, lo, hi)
    Si hi <= lo entonces
        Retornar
    lt, i, gt := lo, lo + 1, hi
    pivot := arr[lo]
    Mientras i <= gt Hacer
        Si arr[i] < pivot Entonces
            Intercambiar(arr[lt], arr[i])
            lt := lt + 1
            i := i + 1
        Sino Si arr[i] > pivot Entonces
            Intercambiar(arr[gt], arr[i])
            gt := gt - 1
        Sino
            i := i + 1
    llamar qSort(arr, lo, lt - 1)
    llamar qSort(arr, gt + 1, hi)
    Retornar arr
Fin Procedimiento

# la función recibe 3 argumentos:
# arr: arrreglo conteniendo los ementos a ordenar
# lo: el índice del primer elemento del rango a ordenar
# hi: el índice del último elemento del rango a ordenar
def qSort(arr, lo, hi):
    # Verifica si hi es menor o igual a lo, en cuyo caso, retorna, hemos terminado
    if hi <= lo:
        return 
    # lt: puntero para la sublista de elementos menores que el pivote
    # gt: puntero para la sublista de elementos mayores que el pivote
    lt, i, gt = lo, lo + 1, hi
    # se toma como pivote el primer elemento del arreglo
    pivot = arr[lo]
    while i <= gt:
    # se compara cada elemento arr[i] con el pivote 
        if arr[i] < pivot:
    # Si arr[i] es menor que pivot, se intercambian los elementos en las posiciones lt e i, y se incrementan ambos punteros lt e i
            (arr[lt], arr[i]) = (arr[i], arr[lt])
            lt += 1
            i += 1
        elif arr[i] > pivot:
    # si arr[i] es mayor que pivot, se intercambian los elementos en las posiciones gt e i, y se decrementa el puntero gt
            (arr[gt], arr[i]) = (arr[i], arr[gt])
            gt -= 1
        else:
    # Si arr[i] es igual a pivot, se incrementa i
            i += 1
    
    # Se hacce la llamada recursiva para la sublista con los numeros menores al pivote
    qSort(arr, lo, lt - 1)
    # Se hacce la llamada recursiva para la sublista con los mayores menores al pivote
    qSort(arr, gt + 1, hi)
    return arr

def bubble_sort(arr):
    n = len(arr)

    # Iterar a través de todos los elementos en el arreglo
    for i in range(n):
        # Últimos i elementos ya están ordenados, no es necesario revisarlos
        for j in range(0, n - i - 1):
            # Intercambiar si el elemento encontrado es mayor que el siguiente elemento
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]

import numpy as np
import random
import time
import pandas as pd
from matplotlib import pyplot as plt
#we are using jupyter notebooks, so, the output should be redirected to the notebook
%matplotlib inline
bubble_sort_times = []
quick_sort_times = []
array_sizes = [500, 1000,10000]

list1=list(range(500))
list2=list(range(1000))
list3=list(range(10000))
random.shuffle(list1)
random.shuffle(list2)
random.shuffle(list3)

start= time.time()
qSort(list1.copy(),0,len(list1)-1)
end=time.time()
quick_sort_times.append(end-start)
print("Tiempo de ejecución para 500 elementos, Qsort : {:.5f}".format((end-start)))
start= time.time()
bubble_sort(list1.copy())
end=time.time()
bubble_sort_times.append(end-start)
print("Tiempo de ejecución para 500 elementos, Bubble sort : {:.5f}".format((end-start)))

start= time.time()
qSort(list2.copy(),0,len(list2)-1)
end=time.time()
quick_sort_times.append(end-start)
print("Tiempo de ejecución para 1000 elementos, Qsort : {:.5f}".format((end-start)))
start= time.time()
bubble_sort(list2.copy())
end=time.time()
bubble_sort_times.append(end-start)
print("Tiempo de ejecución para 1000 elementos, Bubble sort : {:.5f}".format((end-start)))

start= time.time()
qSort(list3.copy(),0,len(list3)-1)
end=time.time()
quick_sort_times.append(end-start)
print("Tiempo de ejecución para 10000 elementos, Qsort : {:.5f}".format((end-start)))
start= time.time()
bubble_sort(list3.copy())
end=time.time()
bubble_sort_times.append(end-start)
print("Tiempo de ejecución para 10000 elementos, Bubble sort : {:.5f}".format((end-start)))


# Create a DataFrame
dFrame = pd.DataFrame({
    'bSort': bubble_sort_times,
    'qSort': quick_sort_times,
    'nElements': array_sizes,
})


plt.figure(figsize=(10, 6))
plt.plot(dFrame['nElements'], dFrame['bSort'], marker='o', label='Ordenamiento de la burbuja')
plt.plot(dFrame['nElements'], dFrame['qSort'], marker='o', label='Quick Sort')
plt.xlabel('No Elementos')
plt.ylabel('Tiempo (s)')
plt.title('Comparación tiempos  ordenamiento de la burbuja vs Quick Sort')
plt.legend()
plt.grid(True)
plt.plot()
