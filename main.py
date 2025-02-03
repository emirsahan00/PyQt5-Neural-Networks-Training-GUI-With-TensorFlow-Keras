import sys
from PyQt5.QtWidgets import (QApplication)

from gui import DeepLearningToolBox  # 'DeepLearningToolBox' sınıfını 'gui' modülünden içe aktar

def main():
    app = QApplication(sys.argv)  # PyQt5 uygulamasını başlat
    window = DeepLearningToolBox()  # 'DeepLearningToolBox' penceresini oluştur
    window.show()  # Pencereyi göster
    sys.exit(app.exec_())  # Uygulama döngüsünü başlat ve sonrasında çıkış yap

if __name__ == '__main__':  # Eğer bu dosya ana program olarak çalıştırılıyorsa
    main()  # 'main' fonksiyonunu çağır
