import sys
import os
import time
import subprocess
import glob
import pandas as pd
import xml.etree.ElementTree as ET
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QTabWidget, \
    QMessageBox, QListWidget, QListWidgetItem, QSizePolicy, QHBoxLayout, QInputDialog, QTextEdit
from PyQt5.QtCore import QProcess, QSize, Qt, QThread, pyqtSignal
import cv2
from PyQt5.QtGui import QImage, QPixmap
import webbrowser


PATH_TO_LABELIMG = "../../../labelImg/labelImg.py"
PATH_TO_LABELMAP = "labelmap.pbtxt"


train_data_path = None
test_data_path = None
media_files_path = None
class_name = None


class DirectoryListWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.list_widget = QListWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.list_widget)
        self.setLayout(layout)

    def set_directory(self, directory):
        self.list_widget.clear()
        if directory:
            files = os.listdir(directory)
            self.list_widget.addItems(files)


class DirectoryTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        self.train_data_label = QLabel("Путь к тренировочной выборке: ")
        layout.addWidget(self.train_data_label)

        self.train_data_button = QPushButton("Выбрать тренировочную выборку")
        self.train_data_button.clicked.connect(self.get_train_data_path)
        layout.addWidget(self.train_data_button)

        self.train_files_widget = DirectoryListWidget()
        layout.addWidget(self.train_files_widget)

        self.test_data_label = QLabel("Путь к тестовой выборке: ")
        layout.addWidget(self.test_data_label)

        self.test_data_button = QPushButton("Выбрать тестовую выборку")
        self.test_data_button.clicked.connect(self.get_test_data_path)
        layout.addWidget(self.test_data_button)

        self.test_files_widget = DirectoryListWidget()
        layout.addWidget(self.test_files_widget)

        self.media_files_label = QLabel("Путь к медиафайлам: ")
        layout.addWidget(self.media_files_label)

        self.media_files_button = QPushButton("Выбрать медиафайлы")
        self.media_files_button.clicked.connect(self.get_media_files_path)
        layout.addWidget(self.media_files_button)

        self.media_files_widget = DirectoryListWidget()
        layout.addWidget(self.media_files_widget)

        self.setLayout(layout)

    def get_train_data_path(self):
        global train_data_path
        train_data_path = QFileDialog.getExistingDirectory(self, "Выберите папку с тренировочной выборкой")
        if train_data_path:
            self.train_data_label.setText(f"Путь к тренировочной выборке: {train_data_path}")
            self.train_files_widget.set_directory(train_data_path)

    def get_test_data_path(self):
        global test_data_path
        test_data_path = QFileDialog.getExistingDirectory(self, "Выберите папку с тестовой выборкой")
        if test_data_path:
            self.test_data_label.setText(f"Путь к тестовой выборке: {test_data_path}")
            self.test_files_widget.set_directory(test_data_path)

    def get_media_files_path(self):
        global media_files_path
        media_files_path = QFileDialog.getExistingDirectory(self, "Выберите папку с медиафайлами")
        if media_files_path:
            self.media_files_label.setText(f"Путь к медиафайлам: {media_files_path}")
            self.media_files_widget.set_directory(media_files_path)



class TrainingThread(QThread):
    training_finished = pyqtSignal()

    def __init__(self, *args, **kwargs):
        super(TrainingThread, self).__init__(*args, **kwargs)
        self.running = False
        self.process = None

    def run(self):
        self.running = True
        try:
            self.process = subprocess.Popen(["python", "model_main_tf2.py",
                                             "--pipeline_config_path=ssd_efficientdet_d0_512x512_coco17_tpu-8.config",
                                             f"--model_dir=trained_models/{class_name}",
                                             "--alsologtostderr"])
            self.process.wait()
        except Exception as e:
            print("Ошибка при выполнении model_main_tf2.py:", str(e))
        finally:
            self.running = False
            self.training_finished.emit()

    def stop(self):
        if self.running and self.process:
            print("Stopping training process...")
            self.process.terminate()
            print("Training process stopped.")



class MediaProcessingThread(QThread):
    finished = pyqtSignal()

    def __init__(self, process_command):
        super().__init__()
        self.process_command = process_command

    def run(self):
        try:
            subprocess.run(self.process_command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing subprocess: {e}")
        self.finished.emit()


class ModelTrainingTab(QWidget):
    def __init__(self):
        super().__init__()

        self.process = None
        self.training_thread = None

        layout = QVBoxLayout()

        self.mark_data_button = QPushButton("Разметить данные")
        self.mark_data_button.clicked.connect(self.start_markup)

        self.learn_model_button = QPushButton("Обучить модель")
        self.learn_model_button.clicked.connect(self.learn_model)
        layout.addWidget(self.mark_data_button)
        layout.addWidget(self.learn_model_button)

        self.stop_training_button = QPushButton("Остановить обучение")
        self.stop_training_button.clicked.connect(self.stop_training)
        self.stop_training_button.setEnabled(False)
        layout.addWidget(self.stop_training_button)

        self.monitor_training_button = QPushButton("Мониторинг обучения")
        self.monitor_training_button.setEnabled(False)
        self.monitor_training_button.clicked.connect(self.monitor_training)
        layout.addWidget(self.monitor_training_button)

        self.setLayout(layout)

    def monitor_training(self):
        global class_name
        if class_name:
            try:
                subprocess.Popen(["tensorboard", f"--logdir=trained_models/{class_name}/train"])
                webbrowser.open("http://localhost:6006/")
            except Exception as e:
                QMessageBox.warning(self, "Ошибка", f"Ошибка запуска TensorBoard: {str(e)}")
        else:
            QMessageBox.warning(self, "Внимание", "Название класса не определено.")

    def start_markup(self):
        if self.process is None or self.process.state() == QProcess.NotRunning:
            self.process = QProcess()
            self.process.start("python", [PATH_TO_LABELIMG])
        else:
            QMessageBox.warning(self, "Внимание", "Процесс уже запущен")

    def learn_model(self):
        if train_data_path is None or test_data_path is None:
            QMessageBox.warning(self, "Внимание", "Вы не указали все директории. Обучение невозможно")
        else:
            global class_name
            class_name, ok = QInputDialog.getText(self, "Название класса", "Введите название класса объекта:")
            if ok and class_name.strip():
                xml_to_csv_start(train_data_path, test_data_path, class_name)
                add_item_to_labelmap(class_name)
                subprocess.run(
                    ["python", "generate_tfrecord.py", "--csv_input=test_labels.csv", "--image_dir=" + test_data_path,
                     "--output_path=test.record"])
                subprocess.run(
                    ["python", "generate_tfrecord.py", "--csv_input=train_labels.csv", "--image_dir=" + train_data_path,
                     "--output_path=train.record"])

                self.training_thread = TrainingThread()
                self.training_thread.training_finished.connect(self.training_finished)


                self.monitor_training_button.setEnabled(True)

                self.training_thread.start()


                self.stop_training_button.setEnabled(True)
            else:
                QMessageBox.warning(self, "Внимание", "Название класса не может быть пустым")

    def stop_training(self):
        print("Stop training button clicked")
        if self.training_thread and self.training_thread.isRunning():
            self.training_thread.stop()
            self.monitor_training_button.setEnabled(False)



    def training_finished(self):
        QMessageBox.information(self, "Информация", "Обучение модели завершено.")


        self.stop_training_button.setEnabled(False)
        if class_name:
            try:
                subprocess.run(["python", "exporter_main_v2.py",
                                f"--trained_checkpoint_dir=trained_models/{class_name}",
                                "--pipeline_config_path=ssd_efficientdet_d0_512x512_coco17_tpu-8.config",
                                f"--output_directory=trained_models/{class_name}/inference_graph"])
                QMessageBox.information(self, "Информация", "Модель успешно экспортирована.")
            except Exception as e:
                QMessageBox.warning(self, "Ошибка", f"Ошибка экспорта модели: {str(e)}")

    def keyPressEvent(self, event):
        if event.modifiers() & Qt.ControlModifier and event.key() == Qt.Key_C:
            self.stop_training()





class ImageWidget(QWidget):
    def __init__(self, pixmap=None, file_name=""):
        super().__init__()
        self.layout = QVBoxLayout()
        self.image_label = QLabel()
        if pixmap:
            self.image_label.setPixmap(pixmap.scaled(200, 200, Qt.KeepAspectRatio))
        self.image_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.text_label = QLabel(file_name)
        self.text_label.setAlignment(Qt.AlignCenter)

        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.text_label)
        self.setLayout(self.layout)


class MediaListWidget(QListWidget):
    def __init__(self):
        super().__init__()
        self.setViewMode(QListWidget.IconMode)
        self.setIconSize(QSize(200, 200))
        self.setResizeMode(QListWidget.Adjust)
        self.setGridSize(QSize(220, 220))
        self.setSpacing(10)
        self.setDragEnabled(False)
        self.setAcceptDrops(False)
        self.setDropIndicatorShown(False)

    def load_media_files(self, media_files_path):
        self.clear()
        if media_files_path:
            for file_name in os.listdir(media_files_path):
                if file_name.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')):
                    file_path = os.path.join(media_files_path, file_name)
                    pixmap = QPixmap(file_path)
                    if not pixmap.isNull():
                        item = QListWidgetItem(self)
                        widget = ImageWidget(pixmap, file_name)
                        item.setSizeHint(widget.sizeHint())
                        self.addItem(item)
                        self.setItemWidget(item, widget)
                elif file_name.endswith(('.mp4', '.avi', '.mov')):
                    file_path = os.path.join(media_files_path, file_name)
                    thumbnail = self.generate_video_thumbnail(file_path)
                    if thumbnail:
                        item = QListWidgetItem(self)
                        widget = ImageWidget(thumbnail, file_name)
                        item.setSizeHint(widget.sizeHint())
                        self.addItem(item)
                        self.setItemWidget(item, widget)

    def generate_video_thumbnail(self, video_path):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame_rgb.shape
            bytes_per_line = 3 * width
            q_image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            return pixmap.scaled(200, 200, Qt.KeepAspectRatio)
        return None



class FilteredMediaWidget(QListWidget):
    def __init__(self):
        super().__init__()
        self.setViewMode(QListWidget.IconMode)
        self.setIconSize(QSize(200, 200))
        self.setResizeMode(QListWidget.Adjust)
        self.setGridSize(QSize(220, 220))
        self.setSpacing(10)
        self.setDragEnabled(False)
        self.setAcceptDrops(False)
        self.setDropIndicatorShown(False)

    def load_filtered_media_files(self, media_files_path, filter_name):

        self.clear()

        if os.path.isdir(media_files_path):
            print("Starting detect_from_image.py script...")
            print(media_files_path)
            subprocess.run(["python", "detect_from_image.py",
                            "-m", f"trained_models/{filter_name}/inference_graph/saved_model",
                            "-l", "labelmap.pbtxt",
                            "-i", media_files_path])
            print("detect_from_image.py script executed successfully.")

            subprocess.run(["python", "detect_from_video.py", "-m", f"trained_models/{filter_name}/inference_graph/saved_model", "-l", "labelmap.pbtxt", "-d", media_files_path])
            print("detect_from_video.py script executed successfully")

            image_formats = ('.png', '.jpg', '.jpeg')
            video_formats = ('.mp4', '.avi', '.mov')

            for file_name in os.listdir("outputs/"):
                if file_name.endswith(image_formats) or file_name.endswith(video_formats):
                    file_path = os.path.join("outputs/", file_name)
                    if file_name.endswith(image_formats):
                        pixmap = QPixmap(file_path)
                    else:
                        pixmap = self.generate_video_thumbnail(file_path)
                    if not pixmap.isNull():
                        item = QListWidgetItem(self)
                        widget = ImageWidget(pixmap, file_name)
                        item.setSizeHint(widget.sizeHint())
                        self.addItem(item)
                        self.setItemWidget(item, widget)
        else:
            QMessageBox.warning(self, "Ошибка", f"Директория {media_files_path} не существует.")

    def generate_video_thumbnail(self, video_path):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame_rgb.shape
            bytes_per_line = 3 * width
            q_image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            return pixmap.scaled(200, 200, Qt.KeepAspectRatio)
        return None



class TrainedModelsWidget(QWidget):
    model_selected = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.list_widget = QListWidget()
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Список фильтров:"))
        layout.addWidget(self.list_widget)
        self.setLayout(layout)
        self.list_widget.itemClicked.connect(self.item_clicked)

    def load_trained_models(self):
        self.list_widget.clear()
        trained_models_dir = "trained_models/"
        if os.path.isdir(trained_models_dir):
            trained_models_files = os.listdir(trained_models_dir)
            self.list_widget.addItems(trained_models_files)
        else:
            QMessageBox.warning(self, "Внимание", "Директория trained_models/ не существует.")

    def item_clicked(self, item: QListWidgetItem):
        self.model_selected.emit(item.text())



class FilteringTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QHBoxLayout()
        self.media_files_list_widget = MediaListWidget()
        self.filtered_media_widget = FilteredMediaWidget()
        self.trained_models_widget = TrainedModelsWidget()
        self.apply_filter_button = QPushButton("Применить фильтр")
        self.apply_filter_button.setEnabled(False)
        self.trained_models_widget.model_selected.connect(self.activate_apply_button)
        self.apply_filter_button.clicked.connect(self.apply_filter)
        layout.addWidget(self.media_files_list_widget)
        layout.addWidget(self.trained_models_widget)
        layout.addWidget(self.apply_filter_button)
        layout.addWidget(self.filtered_media_widget)
        self.setLayout(layout)

    def apply_filter(self):
        selected_model = self.trained_models_widget.list_widget.currentItem()
        if selected_model:
            selected_filter = selected_model.text()

            self.filtered_media_widget.load_filtered_media_files(media_files_path, selected_filter)
        else:
            QMessageBox.warning(self, "Предупреждение", "Не выбран фильтр")



    def activate_apply_button(self, model_name):
        self.apply_filter_button.setEnabled(True)


    def load_media_files(self):
        global media_files_path
        if media_files_path and os.path.isdir(media_files_path):
            try:
                self.media_files_list_widget.load_media_files(media_files_path)
            except Exception as e:
                QMessageBox.warning(self, "Ошибка", f"Ошибка загрузки медиафайлов: {str(e)}")
        else:
            QMessageBox.warning(self, "Предупреждение",
                                "Не указана директория медиафайлов или указанный путь недействителен.")

        self.trained_models_widget.load_trained_models()


class HelpTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        layout.addWidget(self.text_edit)

        self.setLayout(layout)

    def set_help_text(self, text):
        self.text_edit.setPlainText(text)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Менеджер обучения модели")
        self.setGeometry(100, 100, 800, 400)
        layout = QVBoxLayout()
        self.tabs = QTabWidget()
        self.train_data_tab = DirectoryTab()
        self.model_training_tab = ModelTrainingTab()
        self.filtering_tab = FilteringTab()
        self.help_tab = HelpTab()
        self.tabs.addTab(self.train_data_tab, "Выбор данных")
        self.tabs.addTab(self.model_training_tab, "Обучение модели")
        self.tabs.addTab(self.filtering_tab, "Фильтрация")
        self.tabs.addTab(self.help_tab, "Справка")
        layout.addWidget(self.tabs)
        self.setLayout(layout)
        self.tabs.currentChanged.connect(self.tab_changed)

        app.aboutToQuit.connect(self.on_exit)

    def on_exit(self):

        if self.model_training_tab.training_thread and self.model_training_tab.training_thread.isRunning():
            self.model_training_tab.training_thread.terminate()

    def tab_changed(self, index):
        if index == 2:
            self.filtering_tab.load_media_files()
        elif index == 3:

            help_text = "Добро пожаловать в справку по системе!\n\n" \
                        "Алгоритм работы с системой:\n" \
                        "1. Определяетесь с объектом, фотографии и видео с которым, хотите найти\n" \
                        "2. Скачиваете фото этого объекта с любого источник, например Kaggle\n" \
                        "3. Делаете две папки и поровну распределяете скачанные фотографии\n" \
                        "4. Во вкладке 'Обучение модели' нажимаете кнопку 'Разметить данные'\n" \
                        "5. В открытой утилите выбираете директорию с подготовленной папкой\n" \
                        "6. Размечаете каждое изображение, выделяя нужный объект. Режим выделения работает, нажав кнопку W\n" \
                        "7. После разметки изображений в двух папках, переходите во вкладку 'Выбор данных' и указываете директории тестовой и тренировочной выборки (это те папки, которые были подготовлены)\n" \
                        "8. Затем переходите во вкладку 'Обучение модели' и нажимаете кнопку 'Обучить модель'\n" \
                        "9. Ждёте обучения модели, сам процесс обучения можно посмотреть, нажав на кнопку 'Мониторинг обучеия'\n" \
                        "10. Если график потерь стабилизировался, то останавливаете обучение. После этого модель экспортируется и будет создан фильтр\n" \
                        "11. Во вкладке 'Выбор данных' указываете путь до медиафайлов, в которых нужно произвести фильтрацию\n" \
                        "12. Переходите во вкладку 'Фильтрация' и нажав на новый фильтр, применяете его. Система отфильтрует все изображения и видеозаписи по данному фильтру\n" \
                        "Хочется отметить, что все фильтры сохраняются и могут быть повторно применены в будущем\n" \
                        "Если у Вас возникли вопросы в ходе работы с системой:\n" \
                        "Моя почта: aiuvarov@edu.hse.ru\n" \
                        "Мой телефон: +79661911816\n"
            self.help_tab.set_help_text(help_text)


def add_item_to_labelmap(item_name):

    with open(PATH_TO_LABELMAP, "w") as f:

        f.write(f"item {{\n")
        f.write(f"    id: 1\n")
        f.write(f"    name: '{item_name}'\n")
        f.write(f"}}\n\n")


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(os.path.join(path, '*.xml')):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def xml_to_csv_start(train_path, test_path, class_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for folder, data_path in [('train', train_path), ('test', test_path)]:
        data_path = os.path.abspath(data_path)
        if not os.path.isdir(data_path):
            QMessageBox.warning(None, "Ошибка", f"Директория {data_path} не существует.")
            continue
        try:
            xml_df = xml_to_csv(data_path)
            csv_path = os.path.join(script_dir, f'{folder}_labels.csv')
            xml_df.to_csv(csv_path, index=None)
            print(f"Успешно конвертированы xml в csv для директории {folder}.")
        except Exception as e:
            QMessageBox.warning(None, "Ошибка", f"Ошибка конвертации xml в csv: {str(e)}")
    print('Процесс завершен.')


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
