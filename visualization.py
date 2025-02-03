import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

class Visualizer:
    @staticmethod
    def plot_training_history(history):
        # Eğitim geçmişi ile doğruluk ve kayıp grafiklerini çizmek için figür oluşturuluyor
        plt.figure(figsize=(12, 4))
        
        # Doğruluk grafiği (Eğitim ve doğrulama)
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])  # Eğitim doğruluğu
        plt.plot(history.history['val_accuracy'])  # Doğrulama doğruluğu
        plt.title('Model Doğruluğu')  # Başlık
        plt.ylabel('Doğruluk')  # Y ekseni
        plt.xlabel('Epoch')  # X ekseni
        plt.legend(['Eğitim', 'Doğrulama'])  

        # Kayıp grafiği (Eğitim ve doğrulama)
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])  # Eğitim kaybı
        plt.plot(history.history['val_loss'])  # Doğrulama kaybı
        plt.title('Model Kaybı')  # Başlık
        plt.ylabel('Kayıp')  # Y ekseni
        plt.xlabel('Epoch')  # X ekseni
        plt.legend(['Eğitim', 'Doğrulama'])  

        plt.tight_layout()  # Grafikleri düzgün hizalamak
        plt.show()  # Grafikleri ekranda gösterme

    @staticmethod
    def plot_confusion_matrix(model, test_generator):
        # Test verisi üzerinde tahminler yapılır
        predictions = model.predict(test_generator)
        predicted_classes = np.argmax(predictions, axis=1)  # Tahmin edilen sınıflar
        true_classes = test_generator.classes  # Gerçek sınıflar

        # Confusion Matrix (Karmaşıklık Matrisi) hesaplanır
        cm = confusion_matrix(true_classes, predicted_classes)
        
        # Karmaşıklık matrisini çizmek için 10x8 boyutunda bir figür oluşturulur
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')  # Matris üzerinde sayıları göster
        plt.title('Karmaşıklık Matrisi')  # Başlık
        plt.ylabel('Gerçek Etiket')  # Y ekseni
        plt.xlabel('Tahmin Edilen Etiket')  # X ekseni
        plt.show()  # Matrisin görselleştirilmesini sağlar
