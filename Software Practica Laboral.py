import cv2
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
import matplotlib.pyplot as plt

def mostrar_resultados(original, umbral_simple, umbral_adaptativo):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title('Imagen Original')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(umbral_simple, cmap='gray')
    plt.title('Umbralización Simple')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(umbral_adaptativo, cmap='gray')
    plt.title('Umbralización Adaptativa')
    plt.axis('off')

    plt.show()

def segmentar_imagen(imagen_path):
    # Cargar la imagen
    imagen = cv2.imread(imagen_path)
    imagen_gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Aplicar filtro bilateral para suavizar la imagen pero mantener los bordes nítidos
    imagen_filtrada = cv2.bilateralFilter(imagen_gray, 9, 75, 75)

    # Aplicar desenfoque para eliminar ruido adicional (opcional)
    imagen_filtrada = cv2.GaussianBlur(imagen_filtrada, (5, 5), 0)

    # Umbralización simple
    _, umbral_simple = cv2.threshold(imagen_filtrada, 127, 255, cv2.THRESH_BINARY_INV)

    # Umbralización adaptativa
    umbral_adaptativo = cv2.adaptiveThreshold(imagen_filtrada, 255,
                                              cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY_INV, 11, 2)

    # Definir un kernel para las operaciones morfológicas
    kernel = np.ones((5, 5), np.uint8)

    # Aplicar apertura en umbralización simple para eliminar pequeños puntos (solo ruido)
    limpieza_simple = cv2.morphologyEx(umbral_simple, cv2.MORPH_OPEN, kernel)

    # Para la limpieza de la umbralización adaptativa, aplicamos un cierre para mantener la estructura de los eritrocitos
    kernel_adaptativo = np.ones((3, 3), np.uint8)
    limpieza_adaptativa = cv2.morphologyEx(umbral_adaptativo, cv2.MORPH_CLOSE, kernel_adaptativo)

    # Encontrar contornos para rellenar los eritrocitos en ambos métodos
    contornos_simple, _ = cv2.findContours(limpieza_simple, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
    contornos_adaptativo, _ = cv2.findContours(limpieza_adaptativa, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)

    # Dibujar y rellenar los contornos encontrados
    cv2.drawContours(umbral_simple, contornos_simple, -1, (255), thickness=cv2.FILLED)
    cv2.drawContours(umbral_adaptativo, contornos_adaptativo, -1, (255), thickness=cv2.FILLED)

    mostrar_resultados(imagen, umbral_simple, umbral_adaptativo)
    return umbral_simple, umbral_adaptativo

def evaluar_segmentacion(segmentada, referencia):
    # Redimensionar la imagen de referencia para que coincida con la segmentada
    referencia_resized = cv2.resize(referencia, (segmentada.shape[1], segmentada.shape[0]))

    # Convertir las imágenes a binario
    _, segmentada_bin = cv2.threshold(segmentada, 127, 255, cv2.THRESH_BINARY)
    _, referencia_bin = cv2.threshold(referencia_resized, 127, 255, cv2.THRESH_BINARY)

    # Convertir las imágenes de 0-255 a 0-1
    segmentada_bin = segmentada_bin // 255
    referencia_bin = referencia_bin // 255

    # Aplanar las imágenes para calcular las métricas
    segmentada_flat = segmentada_bin.flatten()
    referencia_flat = referencia_bin.flatten()

    # Calcular métricas
    accuracy = accuracy_score(referencia_flat, segmentada_flat)
    precision = precision_score(referencia_flat, segmentada_flat, zero_division=1)
    recall = recall_score(referencia_flat, segmentada_flat, zero_division=1)
    f1 = f1_score(referencia_flat, segmentada_flat, zero_division=1)
    iou = jaccard_score(referencia_flat, segmentada_flat)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"IoU: {iou:.4f}")

# Ruta a una imagen de muestra
imagen_path = "C:\\Users\\2davi\\Desktop\\Practica Laboral II\\Imagenes de prueba\\#3\\prueba 3.jpg"
muestra = cv2.imread("C:\\Users\\2davi\\Desktop\\Practica Laboral II\\Imagenes de prueba\\#3\\mask.jpg", 0)
simple, adaptativa = segmentar_imagen(imagen_path)

print("Métricas de comparación para umbralización simple: ")
evaluar_segmentacion(simple, muestra)

print("Métricas de comparación para umbralización adaptativa: ")
evaluar_segmentacion(adaptativa, muestra)