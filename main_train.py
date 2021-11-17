from model import *

import pathlib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

data_dir = pathlib.Path("garbage_images")  # 이미지 데이터 파일의 경로를 지정한다.

# 무작위 시드를 고정하고 배치 크기와 이미지 크기를 지정한다.
BATCH_SIZE = 32

IMG_SIZE = (224, 224)
IMG_SHAPE = IMG_SIZE + (3,)
seed = 123
model_name = "EfficientNet-B0"
preprocess_input = tf.keras.applications.efficientnet.preprocess_input
base_model = tf.keras.applications.EfficientNetB0(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')

train_dataset = image_dataset_from_directory(  # 훈련 데이터를 나눈다.
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=seed,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)
validation_dataset = image_dataset_from_directory(  # 검증 데이터를 나눈다.
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=seed,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_dataset.class_names
plt.figure(figsize=(9, 9))
for images, labels in train_dataset.take(1):  # 훈련용 데이터셋에서 처음 9 개의 이미지 및 레이블을 보여준다.
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.savefig('model_information/1_train_dataset.png')

# 버퍼링된 프리페치를 사용하여 I/O 차단 없이 디스크에서 이미지를 로드한다.
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

data_augmentation = tf.keras.Sequential([  # 데이터 증강을 위해 회전 및 수평 뒤집기로 훈련 이미지에 다양성을 인위적으로 도입한다.
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])
for image, _ in train_dataset.take(1):  # 증강된 데이터를 확인한다.
    plt.figure(figsize=(9, 9))
    first_image = image[0]
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
        plt.imshow(augmented_image[0] / 255)
        plt.axis('off')
plt.savefig('model_information/2_augmented_images.png')

num_classes = len(class_names)
model = load_model(  # 해당 모델이 있다면 불러오고 없다면 학습 및 저장한다.
    model_name,
    preprocess_input,
    base_model,
    train_dataset,
    validation_dataset,
    num_classes,
    IMG_SHAPE,
    data_augmentation
)

predict_test(validation_dataset, model, class_names, model_name)  # 테스트 데이터의 일부를 예측하고 출력한다.
