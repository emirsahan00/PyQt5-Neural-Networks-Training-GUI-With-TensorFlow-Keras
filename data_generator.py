from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

class DataGenerator:
    @staticmethod
    def create_generators(data_path, model_name, train_batch_size, validation_batch_size):
        # Veri artırma işlemleri için ImageDataGenerator nesnesi oluşturuluyor
        train_datagen = ImageDataGenerator(
            rescale=1/255,  # Piksel değerlerini 0-1 aralığına ölçekleme
            rotation_range=20,  # Rastgele döndürme aralığı
            width_shift_range=0.2,  # Genişlik kaydırma
            height_shift_range=0.2,  # Yükseklik kaydırma
            horizontal_flip=True,  # Yatay çevirme
            vertical_flip=True,  # Dikey çevirme
            fill_mode='nearest',  # Boşlukları doldurma yöntemi
            validation_split=0.2  # Verinin %20'sini doğrulama için ayırma
        )

        # Test verileri için yalnızca ölçekleme işlemi uygulanıyor
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Kullanılacak modelin giriş boyutunu belirleme
        img_shape = (299, 299, 3) if model_name == 'Xception' else (224, 224, 3)
        
        # Eğitim veri kümesi oluşturuluyor
        train_generator = train_datagen.flow_from_directory(
            os.path.join(data_path, 'train'),  # Eğitim verilerinin bulunduğu dizin
            target_size=(img_shape[0], img_shape[1]),  # Resimlerin yeniden boyutlandırılacağı hedef boyut
            batch_size=train_batch_size,  # Eğitim için batch büyüklüğü
            class_mode='categorical',  # Çok sınıflı sınıflandırma
            subset='training'  # Eğitim için veri alt kümesi
        )

        # Doğrulama veri kümesi oluşturuluyor
        validation_generator = train_datagen.flow_from_directory(
            os.path.join(data_path, 'train'),  # Eğitim verilerinin bulunduğu dizin (validation_split kullanıldığı için aynı dizin)
            target_size=(img_shape[0], img_shape[1]),  # Resimlerin yeniden boyutlandırılacağı hedef boyut
            batch_size=validation_batch_size,  # Doğrulama için batch büyüklüğü
            class_mode='categorical',  # Çok sınıflı sınıflandırma
            subset='validation'  # Doğrulama için veri alt kümesi
        )

        # Test veri kümesi oluşturuluyor
        test_generator = test_datagen.flow_from_directory(
            os.path.join(data_path, 'test'),  # Test verilerinin bulunduğu dizin
            target_size=(img_shape[0], img_shape[1]),  # Resimlerin yeniden boyutlandırılacağı hedef boyut
            batch_size=validation_batch_size,  # Test için batch büyüklüğü
            class_mode='categorical'  # Çok sınıflı sınıflandırma
        )
        
        return train_generator, validation_generator, test_generator
