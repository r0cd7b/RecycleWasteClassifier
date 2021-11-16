import pathlib
from tensorflow.keras import preprocessing, applications
from tensorflow import data
from model import *

data_dir = pathlib.Path("garbage_images")  # 이미지 데이터 파일의 경로를 지정한다.

# 무작위 시드를 고정하고 배치 크기와 이미지 크기를 지정한다.
batch_size = 32
img_size = 224
seed = 26

# 훈련 데이터와 검증 데이터를 나눈다.
train_ds = preprocessing.image_dataset_from_directory(
    data_dir,
    batch_size=batch_size,
    image_size=(img_size, img_size),
    seed=seed,
    validation_split=0.1,
    subset="training"
)
val_ds = preprocessing.image_dataset_from_directory(
    data_dir,
    batch_size=batch_size,
    image_size=(img_size, img_size),
    seed=seed,
    validation_split=0.1,
    subset="validation"
)

# 클래스 이름을 확인한다.
class_names = train_ds.class_names
print(f"class_names = {class_names}")

# 이미지 9장을 확인한다.
# plt.figure(figsize=(5, 5))
# for images, labels in train_ds.take(1):
#     for i in range(9):
#         plt.subplot(3, 3, i + 1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(class_names[labels[i]])
#         plt.axis("off")
# plt.show()

# 데이터를 무작위로 변환하여 증강하고 확인한다.
# data_augmentation = keras.Sequential([
#     experimental.preprocessing.RandomFlip(),
#     experimental.preprocessing.RandomRotation(0.2)
# ])
# plt.figure(figsize=(5, 5))
# for images, labels in train_ds.take(1):
#     for i in range(9):
#         augmented_images = data_augmentation(images)
#         plt.subplot(3, 3, i + 1)
#         plt.imshow(augmented_images[0].numpy().astype("uint8"))
#         plt.title(class_names[labels[0]])
#         plt.axis("off")
# plt.show()

data_shape = (img_size, img_size, 3)  # 입력 데이터의 shape를 저장한다.

# on disk cache를 활용하여 성능을 향상시킨다.
auto_tune = data.experimental.AUTOTUNE
train_cache = train_ds.cache().prefetch(buffer_size=auto_tune)
val_cache = val_ds.cache().prefetch(buffer_size=auto_tune)

model = load_model(  # 모델이 있다면 불러오고 없다면 학습 및 저장한다.
    "MobileNet(alpha=1.0)",
    applications.MobileNet(include_top=False, input_shape=data_shape, pooling="avg"),  # 전이 학습할 모델을 선정한다.
    train_cache,
    val_cache,
    data_shape,
    len(class_names),
    batch_size
)
# predict_test(val_cache, model, class_names)  # 검증 데이터의 일부를 예측하고 출력한다.

# 한 개의 이미지를 예측할 때, 아래와 같이 수행한다.
# for images, labels in val_ds.take(1):
#     img = images[0]
#     img = (np.expand_dims(img, 0))
#     predictions_single = model.predict(img)
#     plt.figure(figsize=(9, 4))
#     plt.subplot(1, 2, 1)
#     plot.plot_image(0, predictions_single, [labels[0]], [images[0] * 255], class_names)
#     plt.subplot(1, 2, 2)
#     plot.plot_value_array(0, predictions_single, [labels[0]], class_names)
#     plt.show()
