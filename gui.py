import sys
from PyQt5.QtWidgets import (QWidget, QLabel, QLineEdit, QPushButton, 
                           QComboBox, QSpinBox, QHBoxLayout, QVBoxLayout, 
                           QFrame, QFileDialog, QMessageBox, QDialog, 
                           QProgressBar, QApplication)
from PyQt5.QtCore import Qt
import numpy as np
from config import Config
from data_generator import DataGenerator
from model_builder import ModelBuilder
from callbacks import CustomCallback
from visualization import Visualizer
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Eğitim ilerlemesini göstermek için bir pencere sınıfı
class TrainingProgressWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Training Progress')  # Pencere başlığını ayarla
        self.setModal(False)  # Pencereyi modalsız yap
        self.resize(400, 200)  # Pencere boyutunu ayarla
        self.setStyleSheet(Config.PROGRESS_WINDOW_STYLE)  # Stil uygulama
        self._init_ui()  # UI bileşenlerini başlat

    def _init_ui(self):
        # UI bileşenlerini başlat
        layout = QVBoxLayout()

        # Farklı metrikler için etiketler ve ilerleme çubukları
        self.epoch_label = QLabel('Epoch: 0/0')
        self.epoch_progress = QProgressBar()
        self.epoch_progress.setRange(0, 100)  # Aralık belirleme
        
        self.progress_label = QLabel('Batch Progress: 0/0')
        self.training_progress = QProgressBar()
        self.training_progress.setRange(0, 100)  # Aralık belirleme
        
        self.loss_label = QLabel('Loss: 0.0000')
        self.accuracy_label = QLabel('Accuracy: 0.0000')
        self.val_loss_label = QLabel('Val Loss: 0.0000')
        self.val_accuracy_label = QLabel('Val Accuracy: 0.0000')

        # Tüm bileşenleri layout'a ekle
        for widget in [self.epoch_label, self.epoch_progress, 
                      self.progress_label, self.training_progress,
                      self.loss_label, self.accuracy_label,
                      self.val_loss_label, self.val_accuracy_label]:
            layout.addWidget(widget)

        self.setLayout(layout)  # Layout'u pencereye uygula

    def closeEvent(self, event):
        event.ignore()  # Pencereyi kapatmaya izin verme, eğitim devam etsin

# Derin öğrenme aracının ana sınıfı
class DeepLearningToolBox(QWidget):
    def __init__(self):
        super().__init__()
        self.history = None  # Eğitim geçmişi
        self.model = None  # Model
        self.train_generator = None  # Eğitim verisi jeneratörü
        self.validation_generator = None  # Doğrulama verisi jeneratörü
        self.test_generator = None  # Test verisi jeneratörü
        self.initUI()  # UI'yi başlat

    def initUI(self):
        self.setWindowTitle('DeepLearningToolBox')  # Pencere başlığını ayarla
        self.resize(600, 500)  # Pencere boyutunu ayarla
        self.setStyleSheet(Config.WINDOW_STYLE)  # Stil uygula

        # Ana çerçeve
        main_frame = QFrame()

        # Veri yolu seçimi
        self.data_path_label = QLabel('Data Path:')  # Veri yolu etiketi
        self.data_path_input = QLineEdit()  # Veri yolu girişi
        self.data_path_button = QPushButton('Load...')  # Veri yolu yükle butonu
        self.data_path_button.clicked.connect(self.load_data_path)  # Buton tıklama olayı

        # Veri yolu yerleşimi
        data_path_layout = QHBoxLayout()
        data_path_layout.addWidget(self.data_path_label)
        data_path_layout.addWidget(self.data_path_input)
        data_path_layout.addWidget(self.data_path_button)

        # Model seçimi
        self.model_label = QLabel('Model:')  # Model etiketi
        self.model_combobox = QComboBox()  # Model seçimi combobox'ı
        self.model_combobox.addItems(['Choose Model...', 'VGG16', 'ResNet50', 'MobileNetV2', 'Xception', 'DenseNet121'])  # Modelleri ekle

        # Model yerleşimi
        model_layout = QHBoxLayout()
        model_layout.addWidget(self.model_label)
        model_layout.addWidget(self.model_combobox)

        # Öğrenme oranı
        self.learning_rate_label = QLabel('Learning Rate:')  # Öğrenme oranı etiketi
        self.learning_rate_input = QLineEdit()  # Öğrenme oranı girişi
        self.learning_rate_input.setText('0.001')  # Varsayılan değeri 0.001 olarak ayarla

        # Öğrenme oranı yerleşimi
        learning_rate_layout = QHBoxLayout()
        learning_rate_layout.addWidget(self.learning_rate_label)
        learning_rate_layout.addWidget(self.learning_rate_input)

        # Optimizatör seçimi
        self.optimizer_label = QLabel('Optimizer:')  # Optimizatör etiketi
        self.optimizer_combobox = QComboBox()  # Optimizatör combobox'ı
        self.optimizer_combobox.addItems(['Choose Optimizer...', 'SGD', 'Adam', 'RMSprop', 'Nadam'])  # Optimizatörleri ekle

        # Optimizatör yerleşimi
        optimizer_layout = QHBoxLayout()
        optimizer_layout.addWidget(self.optimizer_label)
        optimizer_layout.addWidget(self.optimizer_combobox)

        # Eğitim batch boyutu
        self.train_batch_label = QLabel('Train Batch Size:')  # Eğitim batch boyutu etiketi
        self.train_batch_input = QSpinBox()  # Eğitim batch boyutu girişi
        self.train_batch_input.setRange(1, 1024)  # Aralığı 1 ile 1024 arasında ayarla
        self.train_batch_input.setValue(32)  # Varsayılan değeri 32 olarak ayarla

        # Eğitim batch boyutu yerleşimi
        train_batch_layout = QHBoxLayout()
        train_batch_layout.addWidget(self.train_batch_label)
        train_batch_layout.addWidget(self.train_batch_input)

        # Doğrulama batch boyutu
        self.val_batch_label = QLabel('Val Batch Size:')  # Doğrulama batch boyutu etiketi
        self.val_batch_input = QSpinBox()  # Doğrulama batch boyutu girişi
        self.val_batch_input.setRange(1, 1024)  # Aralığı 1 ile 1024 arasında ayarla
        self.val_batch_input.setValue(32)  # Varsayılan değeri 32 olarak ayarla

        # Doğrulama batch boyutu yerleşimi
        val_batch_layout = QHBoxLayout()
        val_batch_layout.addWidget(self.val_batch_label)
        val_batch_layout.addWidget(self.val_batch_input)

        # Epoch sayısı
        self.epoch_label = QLabel('Epoch:')  # Epoch etiketi
        self.epoch_input = QSpinBox()  # Epoch girişi
        self.epoch_input.setRange(1, 1000)  # Aralığı 1 ile 1000 arasında ayarla
        self.epoch_input.setValue(10)  # Varsayılan değeri 10 olarak ayarla

        # Epoch yerleşimi
        epoch_layout = QHBoxLayout()
        epoch_layout.addWidget(self.epoch_label)
        epoch_layout.addWidget(self.epoch_input)

        # Eğitim başlatma butonu
        self.start_train_button = QPushButton('Start Train')  # Başlat butonu
        self.start_train_button.clicked.connect(self.start_training)  # Buton tıklama olayı

        # Ekstra butonlar
        self.graph_button = QPushButton('Train-Test Loss and Acc Graphs')  # Grafik butonu
        self.graph_button.clicked.connect(self.show_graphs)  # Buton tıklama olayı
        self.conf_matrix_button = QPushButton('Confusion Matrix')  # Konfüzyon matrisi butonu
        self.conf_matrix_button.clicked.connect(self.show_confusion_matrix)  # Buton tıklama olayı

        # Ekstra butonlar için yerleşim
        additional_buttons_layout = QHBoxLayout()
        additional_buttons_layout.addWidget(self.graph_button)
        additional_buttons_layout.addWidget(self.conf_matrix_button)

        # Ana layout
        layout = QVBoxLayout()
        layout.addLayout(data_path_layout)
        layout.addLayout(model_layout)
        layout.addLayout(learning_rate_layout)
        layout.addLayout(optimizer_layout)
        layout.addLayout(train_batch_layout)
        layout.addLayout(val_batch_layout)
        layout.addLayout(epoch_layout)
        layout.addWidget(self.start_train_button)
        layout.addLayout(additional_buttons_layout)

        main_frame.setLayout(layout)

        # Dış yerleşim
        outer_layout = QVBoxLayout()
        outer_layout.addWidget(main_frame)
        self.setLayout(outer_layout)

    def load_data_path(self):
        # Veri yolu seçme penceresi açma
        file_dialog = QFileDialog()
        file_path = file_dialog.getExistingDirectory(self, 'Choose Data Directory')  # Veri yolu seçme
        if file_path:
            self.data_path_input.setText(file_path)  # Veri yolunu girdi kutusuna yerleştir

    def start_training(self):
        try:
            # Eğitim ilerleme penceresini oluştur ve göster
            progress_window = TrainingProgressWindow(self)
            progress_window.show()

            # Ana pencereyi gizle
            self.hide()

            # Veri jeneratörlerini ayarla
            self.train_generator, self.validation_generator, self.test_generator = \
                DataGenerator.create_generators(
                    self.data_path_input.text(),
                    self.model_combobox.currentText(),
                    self.train_batch_input.value(),
                    self.val_batch_input.value()
                )
            
            # Modeli oluştur
            self.model = ModelBuilder.create_model(
                self.model_combobox.currentText(),
                len(self.train_generator.class_indices)
            )

            # Optimizatörü al
            optimizer = ModelBuilder.get_optimizer(
                self.optimizer_combobox.currentText(),
                float(self.learning_rate_input.text())
            )

            # Modeli derle
            self.model.compile(
                optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            # Callback fonksiyonlarını ayarla
            custom_callback = CustomCallback(progress_window, QApplication.instance())
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
            es1 = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=8)
            best_weights = ModelCheckpoint(
                'best_weights.keras',
                monitor='val_accuracy',
                mode='max',
                verbose=1,
                save_best_only=True
            )
            last_weights = ModelCheckpoint(
                'last_weights.keras',
                monitor='val_loss',
                mode='min',
                verbose=1,
                save_best_only=False,
                save_weights_only=False,
                save_freq='epoch'
            )
            lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
            
            # Modeli eğit
            self.history = self.model.fit(
                self.train_generator,
                validation_data=self.validation_generator,
                epochs=self.epoch_input.value(),
                callbacks=[custom_callback, es, best_weights, last_weights, lr_reducer, es1],
                verbose=0
            )

            # Ana pencereyi tekrar göster
            self.show()
            progress_window.close()
            
            QMessageBox.information(self, "Training Complete", 
                                  "Model training has finished successfully!\n"
                                  "Best weights saved as 'best_weights.keras'\n"
                                  "Last weights saved as 'last_weights.keras'")

        except Exception as e:
            self.show()
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")  # Hata mesajı göster

    def show_graphs(self):
        # Eğitim geçmişi yoksa uyarı ver
        if self.history is None:
            QMessageBox.warning(self, "Warning", "No training history available. Please train the model first.")
            return
        
        # Eğitim geçmişi grafiklerini göster
        Visualizer.plot_training_history(self.history)

    def show_confusion_matrix(self):
        # Model veya test verisi yoksa uyarı ver
        if self.model is None or self.test_generator is None:
            QMessageBox.warning(self, "Warning", "Model or test data not available. Please train the model first.")
            return
            
        # Konfüzyon matrisini göster
        Visualizer.plot_confusion_matrix(self.model, self.test_generator)
