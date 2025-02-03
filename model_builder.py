from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2, Xception, DenseNet121
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Nadam

class ModelBuilder:
    @staticmethod
    def create_model(model_name, num_classes):
        # Seçilen model adına göre uygun temel modelin yüklenmesi
        if model_name == 'VGG16':
            base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        elif model_name == 'ResNet50':
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        elif model_name == 'MobileNetV2':
            base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        elif model_name == 'Xception':
            base_model = Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
        elif model_name == 'DenseNet121':
            base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        else:
            # Geçersiz model adı seçildiyse hata mesajı
            raise ValueError("Lütfen geçerli bir model seçin")

        # Temel modelin son 3 katmanı dışında tüm katmanları eğitilemez yapıyoruz
        for layer in base_model.layers[:-3]:
            layer.trainable = False

        # Modelin üst kısmına yeni katmanlar ekleniyor
        model = Sequential([
            base_model,  # Temel model (önceden eğitilmiş)
            Flatten(),  # Veriyi düzleştiriyoruz
            Dense(1024, activation='relu'),  # Tam bağlantılı ilk katman
            Dropout(0.2),  # Aşırı uyumdan kaçınmak için dropout
            Dense(512, activation='relu'),  # İkinci tam bağlantılı katman
            Dense(num_classes, activation='softmax')  # Son katman, sınıflandırma için
        ])

        return model

    @staticmethod
    def get_optimizer(optimizer_name, learning_rate):
        # Seçilen optimizasyon algoritmalarını bir sözlükte saklıyoruz
        optimizers = {
            'SGD': SGD,
            'Adam': Adam,
            'RMSprop': RMSprop,
            'Nadam': Nadam
        }
        
        # Seçilen optimizatörü getiriyoruz
        optimizer_class = optimizers.get(optimizer_name)
        if optimizer_class is None:
            # Geçersiz optimizatör adı seçildiyse hata mesajı
            raise ValueError("Lütfen geçerli bir optimizatör seçin")
            
        # Öğrenme oranını ayarlayarak optimizatörü döndürüyoruz
        return optimizer_class(learning_rate=learning_rate)
