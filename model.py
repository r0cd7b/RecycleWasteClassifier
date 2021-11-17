import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# 모델을 불러오거나 학습한다.
def train_model(preprocess_input, base_model, model_name, train_dataset, validation_dataset, num_classes, img_shape,
                data_augmentation, class_names):
    model_dir = f"models/{model_name}.h5"

    try:
        # 모델을 불러오고 구조를 출력한다.
        model = tf.keras.models.load_model(model_dir)
        model.summary()

    except Exception as e:
        print(e)

        # 특징 블록의 shape를 확인한다.
        image_batch, label_batch = next(iter(train_dataset))
        feature_batch = base_model(image_batch)
        print(f"Feature batch shape: {feature_batch.shape}")

        base_model.trainable = False  # 훈련 가능한 층의 가중치를 고정한다.
        base_model.summary()  # 기본 모델 아키텍처를 살펴본다.

        # 해당 레이어를 사용하여 특성을 이미지당 하나의 1280-요소 벡터로 변환하여 5x5 공간 위치에 대한 평균을 구한다.
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        feature_batch_average = global_average_layer(feature_batch)
        print(f"Feature batch average shape: {feature_batch_average.shape}")

        # 해당 레이어를 사용하여 특성을 이미지당 복수의 예측으로 변환한다.
        prediction_layer = tf.keras.layers.Dense(num_classes)
        prediction_batch = prediction_layer(feature_batch_average)
        print(f"Prediction batch shape: {prediction_batch.shape}")

        #  데이터 증강, 크기 조정, base_model 및 특성 추출기 레이어를 함께 연결하여 모델을 구축한다.
        inputs = tf.keras.Input(shape=img_shape)
        x = data_augmentation(inputs)
        x = preprocess_input(x)
        x = base_model(x, training=False)  # 모델에 BatchNormalization 레이어가 포함되어 있으므로 training=False를 사용한다.
        x = global_average_layer(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = prediction_layer(x)
        model = tf.keras.Model(inputs, outputs)

        # 복수의 클래스가 있고 모델이 선형 출력을 제공하므로 from_logits=True와 함께 Sparse Categorical Crossentropy 손실을 사용한다.
        base_learning_rate = 0.0001
        model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
        model.summary()  # 모델 아키텍처를 살펴본다.
        print(f"Length of trainable variables in the model: {len(model.trainable_variables)}")  # 훈련 가능한 객체 수를 확인한다.

        # 훈련하기 전 초기 손실과 정확도를 확인한다.
        loss0, accuracy0 = model.evaluate(validation_dataset)
        print(f"initial loss: {loss0}")
        print(f"initial accuracy: {accuracy0}")

        early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                      patience=10)  # 지정된 에포크 횟수 동안 성능 향상이 없으면 자동으로 훈련이 멈춘다.

        history = model.fit(train_dataset, epochs=1000, validation_data=validation_dataset,
                            callbacks=[early_stop])  # 모델을 훈련한다.

        base_model.trainable = True  # base_model을 고정 해제한다.
        print(f"Number of layers in the base model: {len(base_model.layers)}")

        fine_tune_at = 100  # 미세 조정할 레이어 위치를 선정한다.
        for layer in base_model.layers[:fine_tune_at]:  # "fine_tune_at" 전에 모든 레이어를 고정한다.
            layer.trainable = False

        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      optimizer=tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate / 10),
                      metrics=["accuracy"])  # 더 낮은 학습률을 사용하여 모델을 컴파일한다.
        model.summary()  # 모델 아키텍처를 살펴본다.
        print(f"Length of trainable variables in the model: {len(model.trainable_variables)}")  # 훈련 가능한 객체 수를 확인한다.

        history_fine = model.fit(train_dataset, epochs=1000, initial_epoch=history.epoch[-1],
                                 validation_data=validation_dataset, callbacks=[early_stop])  # 미세 조정된 모델로 훈련을 계속한다.
        initial_epochs = history.epoch[-1]

        model.save(model_dir)  # 학습한 모델을 저장한다.

        # 출력할 변수 값을 준비한다.
        acc = history.history["accuracy"] + history_fine.history["accuracy"]
        val_acc = history.history["val_accuracy"] + history_fine.history["val_accuracy"]
        loss = history.history["loss"] + history_fine.history["loss"]
        val_loss = history.history["val_loss"] + history_fine.history["val_loss"]

        # 전체 학습 과정을 출력한다.
        plt.figure(figsize=(9, 9))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label="Training Accuracy")
        plt.plot(val_acc, label="Validation Accuracy")
        plt.plot([initial_epochs - 1, initial_epochs - 1], plt.ylim(), label="Start Fine Tuning")
        plt.legend(loc="lower right")
        plt.ylabel("Accuracy")
        plt.xlabel("epoch")
        plt.title(f"{model_name} Training and Validation Accuracy")
        plt.subplot(2, 1, 2)
        plt.plot(loss, label="Training Loss")
        plt.plot(val_loss, label="Validation Loss")
        plt.plot([initial_epochs - 1, initial_epochs - 1], plt.ylim(), label="Start Fine Tuning")
        plt.legend(loc="upper right")
        plt.ylabel("Cross Entropy")
        plt.title(f"{model_name} Training and Validation Loss")
        plt.xlabel("epoch")
        plt.tight_layout()
        plt.savefig(f"model_information/3_{model_name}_history.png")
        plt.close()

    # 검증 데이터 셋으로 모델을 평가한다.
    loss, accuracy = model.evaluate(validation_dataset)
    print(f"Validation loss: {loss}")
    print(f"Validation accuracy: {accuracy}")

    # 모델로 검증 데이터 셋에 대한 예측을 수행한다.
    image_batch, label_batch = validation_dataset.as_numpy_iterator().next()
    predictions = model.predict_on_batch(image_batch)
    predictions = tf.nn.softmax(predictions)
    print(f"Predictions:\n{predictions.numpy()}")
    print(f"Labels:\n{label_batch}")

    plt.figure(figsize=(9, 9))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(image_batch[i].astype("uint8"))
        plt.title(
            f"{class_names[np.argmax(predictions[i])]} {100 * np.max(predictions[i]):.2f}% ({class_names[label_batch[i]]})")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"model_information/4_{model_name}_predictions.png")
    plt.close()

    return model
