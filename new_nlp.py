from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

# Каталог с данными для обучения
train_dir = 'train'
# Каталог с данными для проверки
val_dir = 'val'
# Каталог с данными для тестирования
test_dir = 'test'
# Размеры изображения
img_width, img_height = 150, 150
# Размерность тензора на основе изображения для входных данных в нейронную сеть
# backend Tensorflow, channels_last
input_shape = (img_width, img_height, 3)
# Количество эпох
epochs = 100
# Размер мини-выборки
batch_size = 4
# Количество изображений для обучения
nb_train_samples = 70
# Количество изображений для проверки
nb_validation_samples = 14
# Количество изображений для тестирования
nb_test_samples = 16

# Архитектура сети

# Слой свертки, размер ядра 3х3, количество карт признаков - 32 шт., функция активации ReLU.
# Слой подвыборки, выбор максимального значения из квадрата 2х2
# Слой свертки, размер ядра 3х3, количество карт признаков - 32 шт., функция активации ReLU.
# Слой подвыборки, выбор максимального значения из квадрата 2х2
# Слой свертки, размер ядра 3х3, количество карт признаков - 64 шт., функция активации ReLU.
# Слой подвыборки, выбор максимального значения из квадрата 2х2
# Слой преобразования из двумерного в одномерное представление
# Полносвязный слой, 64 нейрона, функция активации ReLU.
# Слой Dropout.
# Выходной слой, 1 нейрон, функция активации sigmoid

# Слои с 1 по 6 используются для выделения важных признаков в изображении, а слои с 7 по 10 - для классификации.
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# Компиляция модели
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=nb_validation_samples // batch_size)

scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)
print("Точность на тестовых данных: %.2f%%" % (scores[1]*100))

print('Saving NLP')

model_json = model.to_json()
json_file = open("defects_normal","w")
json_file.write(model_json)
json_file.close()
model.save_weights("defect_normal_concrete_cnn.h5")
print('Save Well Done!!!')