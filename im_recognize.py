
from tensorflow.python.keras.models import Model, model_from_json,load_model
from keras.utils import load_img,img_to_array
import numpy as np
import matplotlib.pyplot as plt
import cv2

#Загружаем изображение из папки img для распознавания и создания
image_file_name = 'img/b3.jpg'
img = load_img(image_file_name, target_size=(150, 150))
plt.title('Загруженное изображение')
plt.imshow(img)

# Преобразуем картинку в массив
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.

# Загружаем данные об архитектуре сети из файла json
json_file = open("defects_normal", "r")
loaded_model_json = json_file.read()
json_file.close()

# Создаем модель на основе загруженных данных
loaded_model = model_from_json(loaded_model_json)

# Загружаем веса в модель
loaded_model.load_weights("defect_normal_concrete_cnn.h5")
prediction = loaded_model.predict(img_array)

#Делаем срез слоев признаков (1-6 слой)
activation_model = Model(inputs=loaded_model.input, outputs=loaded_model.layers[6].output)
print(activation_model.summary())
activation = activation_model.predict(img_array)
print(activation.shape)

plt.matshow(activation[0, :, :, 18], cmap='viridis')
plt.title('Результат работы модели')

#Печатаем вывод всех карт признаков
images_per_row = 16
n_filters = activation.shape[-1]
size = activation.shape[1]
n_cols = n_filters // images_per_row
display_grid = np.zeros((n_cols * size, images_per_row * size))
for col in range(n_cols):
    for row in range(images_per_row):
        channel_image = activation[0, :, :, col * images_per_row + row]
        channel_image -= channel_image.mean()
        channel_image /= channel_image.std()
        channel_image *= 64
        channel_image += 128
        channel_image = np.clip(channel_image, 0, 255).astype('uint8')
        display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image
scale = 1. / size
plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))

plt.grid(False)
plt.title('Карта признаков')
plt.xlabel("Прогноз целостности = " + str(prediction))
plt.imshow(display_grid, aspect='auto', cmap='viridis')
plt.show()

#Ввод нужной картинки для контуризации
img2 = cv2.imread(image_file_name)
#преобразование картинки в градацию серого
gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#настройка порога бинаризации
thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)[1]
contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0]
result = img2.copy()

#Поиск по контурам, рисование  и выделение нужного нам
for c in contours:
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(result, [box], -1, (0, 0, 255), 1)

# Сохранение результата с контурами
cv2.imwrite("box_recognized.png", result)

# Показать изображение с контурами
cv2.imshow("REC_IMG_with cont", result)
cv2.waitKey(0)



