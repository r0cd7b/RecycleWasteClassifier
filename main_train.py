from model import *

import pathlib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

data_dir = pathlib.Path("garbage_images")  # 이미지 데이터 파일의 경로를 지정한다.

BATCH_SIZE = 32  # 배치 크기를 정한다.

# 이미지 크기과 무작위 시드 값을 정한다.
IMG_SIZE = (224, 224)
seed = 123

# 훈련과 검증 데이터를 나눈다.
train_dataset = image_dataset_from_directory(data_dir, validation_split=0.2, subset="training", seed=seed,
                                             image_size=IMG_SIZE, batch_size=BATCH_SIZE)
validation_dataset = image_dataset_from_directory(data_dir, validation_split=0.2, subset="validation", seed=seed,
                                                  image_size=IMG_SIZE, batch_size=BATCH_SIZE)
class_names = train_dataset.class_names

# 훈련용 데이터셋에서 처음 9 개의 이미지 및 레이블을 보여준다.
plt.figure(figsize=(9, 9))
for images, labels in train_dataset.take(1):
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

# 데이터 증강을 위해 회전 및 수평 뒤집기로 훈련 이미지에 다양성을 인위적으로 도입한다.
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

# 증강된 데이터를 확인한다.
for image, _ in train_dataset.take(1):
    plt.figure(figsize=(9, 9))
    first_image = image[0]
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
        plt.imshow(augmented_image[0] / 255)
        plt.axis('off')
plt.savefig('model_information/2_augmented_images.png')

# 모델에 적용할 이미지의 shape과 클래스 수를 준비한다.
IMG_SHAPE = IMG_SIZE + (3,)
num_classes = len(class_names)

# 모델을 불러오거나 학습하고 검증 데이터를 이용하여 평가 및 예측한다.
model_name = "MobileNet(alpha=0.25)"
train_model(
    tf.keras.applications.mobilenet.preprocess_input,
    tf.keras.applications.MobileNet(alpha=0.25, input_shape=IMG_SHAPE, include_top=False, weights='imagenet'),
    model_name,
    train_dataset,
    validation_dataset,
    num_classes,
    IMG_SHAPE,
    data_augmentation,
    class_names
)

model_name = "MobileNet(alpha=0.50)"
train_model(
    tf.keras.applications.mobilenet.preprocess_input,
    tf.keras.applications.MobileNet(alpha=0.50, input_shape=IMG_SHAPE, include_top=False, weights='imagenet'),
    model_name,
    train_dataset,
    validation_dataset,
    num_classes,
    IMG_SHAPE,
    data_augmentation,
    class_names
)

model_name = "MobileNetV2(alpha=0.35)"
train_model(
    tf.keras.applications.mobilenet_v2.preprocess_input,
    tf.keras.applications.MobileNetV2(alpha=0.35, input_shape=IMG_SHAPE, include_top=False, weights='imagenet'),
    model_name,
    train_dataset,
    validation_dataset,
    num_classes,
    IMG_SHAPE,
    data_augmentation,
    class_names
)

model_name = "MobileNetV2(alpha=0.50)"
train_model(
    tf.keras.applications.mobilenet_v2.preprocess_input,
    tf.keras.applications.MobileNetV2(alpha=0.50, input_shape=IMG_SHAPE, include_top=False, weights='imagenet'),
    model_name,
    train_dataset,
    validation_dataset,
    num_classes,
    IMG_SHAPE,
    data_augmentation,
    class_names
)

model_name = "MobileNetV3(small)"
train_model(
    tf.keras.applications.mobilenet_v3.preprocess_input,
    tf.keras.applications.MobileNetV3Small(input_shape=IMG_SHAPE, include_top=False, weights='imagenet'),
    model_name,
    train_dataset,
    validation_dataset,
    num_classes,
    IMG_SHAPE,
    data_augmentation,
    class_names
)

model_name = "MobileNet(alpha=0.75)"
train_model(
    tf.keras.applications.mobilenet.preprocess_input,
    tf.keras.applications.MobileNet(alpha=0.75, input_shape=IMG_SHAPE, include_top=False, weights='imagenet'),
    model_name,
    train_dataset,
    validation_dataset,
    num_classes,
    IMG_SHAPE,
    data_augmentation,
    class_names
)

model_name = "MobileNetV2(alpha=0.75)"
train_model(
    tf.keras.applications.mobilenet_v2.preprocess_input,
    tf.keras.applications.MobileNetV2(alpha=0.75, input_shape=IMG_SHAPE, include_top=False, weights='imagenet'),
    model_name,
    train_dataset,
    validation_dataset,
    num_classes,
    IMG_SHAPE,
    data_augmentation,
    class_names
)

model_name = "MobileNetV2(alpha=1.0)"
train_model(
    tf.keras.applications.mobilenet_v2.preprocess_input,
    tf.keras.applications.MobileNetV2(alpha=1.0, input_shape=IMG_SHAPE, include_top=False, weights='imagenet'),
    model_name,
    train_dataset,
    validation_dataset,
    num_classes,
    IMG_SHAPE,
    data_augmentation,
    class_names
)

model_name = "MobileNet(alpha=1.0)"
train_model(
    tf.keras.applications.mobilenet.preprocess_input,
    tf.keras.applications.MobileNet(alpha=1.0, input_shape=IMG_SHAPE, include_top=False, weights='imagenet'),
    model_name,
    train_dataset,
    validation_dataset,
    num_classes,
    IMG_SHAPE,
    data_augmentation,
    class_names
)

model_name = "EfficientNet-B0"
train_model(
    tf.keras.applications.efficientnet.preprocess_input,
    tf.keras.applications.EfficientNetB0(input_shape=IMG_SHAPE, include_top=False, weights='imagenet'),
    model_name,
    train_dataset,
    validation_dataset,
    num_classes,
    IMG_SHAPE,
    data_augmentation,
    class_names
)
