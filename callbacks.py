from tensorflow.keras.callbacks import Callback
import numpy as np

class CustomCallback(Callback):
    def __init__(self, progress_window, app):
        super().__init__()
        self.progress_window = progress_window  # Eğitim ilerlemesini gösterecek pencere
        self.app = app  # Arayüzü güncellemek için uygulama nesnesi
        self.train_loss_sum = 0  # Toplam eğitim kaybı
        self.train_acc_sum = 0  # Toplam eğitim doğruluğu
        self.train_batches = 0  # İşlenen batch sayısı

    def on_epoch_begin(self, epoch, logs=None):
        # Yeni epoch başladığında arayüzü güncelle
        self.progress_window.epoch_label.setText(f'Epoch: {epoch + 1}/{self.params["epochs"]}')
        epoch_progress = int(((epoch + 1) / self.params["epochs"]) * 100)
        self.progress_window.epoch_progress.setValue(epoch_progress)
        
        # Epoch başında değerleri sıfırla
        self.train_loss_sum = 0
        self.train_acc_sum = 0
        self.train_batches = 0
        self.app.processEvents()

    def on_batch_begin(self, batch, logs=None):
        # Her batch başladığında arayüzü güncelle
        total_batches = self.params['steps']  # Toplam batch sayısı
        self.progress_window.progress_label.setText(f'Batch Progress: {batch + 1}/{total_batches}')
        progress = int(((batch + 1) / total_batches) * 100)
        self.progress_window.training_progress.setValue(progress)
        self.app.processEvents()

    def on_batch_end(self, batch, logs=None):
        if logs:
            # Her batch tamamlandığında eğitim kaybı ve doğruluğu güncelle
            self.train_loss_sum += logs.get('loss', 0)
            self.train_acc_sum += logs.get('accuracy', 0)
            self.train_batches += 1
            
            # Ortalama kayıp ve doğruluğu hesapla
            avg_loss = self.train_loss_sum / max(1, self.train_batches)
            avg_acc = self.train_acc_sum / max(1, self.train_batches)
            
            # Arayüzde güncellenmiş değerleri göster
            self.progress_window.loss_label.setText(f'Loss: {avg_loss:.4f}')
            self.progress_window.accuracy_label.setText(f'Accuracy: {avg_acc:.4f}')
            self.app.processEvents()

    def on_epoch_end(self, epoch, logs=None):
        if logs:
            # Epoch sonunda doğrulama kaybı ve doğruluğunu güncelle
            val_loss = logs.get('val_loss', 0)
            val_acc = logs.get('val_accuracy', 0)
            
            # Değer NaN ise 0 olarak ata
            if np.isnan(val_loss):
                val_loss = 0
            if np.isnan(val_acc):
                val_acc = 0
            
            # Arayüzde güncellenmiş doğrulama sonuçlarını göster
            self.progress_window.val_loss_label.setText(f'Val Loss: {val_loss:.4f}')
            self.progress_window.val_accuracy_label.setText(f'Val Accuracy: {val_acc:.4f}')
            self.app.processEvents()
