from tensorflow.keras import models, layers, callbacks
from tensorflow import keras
from tensorflow.keras.layers import experimental
import matplotlib.pyplot as plt
import plot


def load_model(model_name, base_model, train_ds, val_ds, data_shape, num_classes, batch_size):
    model_path = f"models/{model_name}.h5"  # 모델 파일 이름을 정한다.

    try:
        model = models.load_model(model_path)  # 모델을 불러온다.
        model.summary()  # 모델 구조를 출력한다.

        results = model.evaluate(val_ds)  # 검증 데이터 셋으로 모델을 평가한다.
        for name, value in zip(model.metrics_names, results):
            print(f"{name}: {value}")

    except Exception as e:
        print(e)

        base_model.trainable = False  # 훈련 가능한 층의 가중치를 고정한다.
        model = keras.Sequential([
            experimental.preprocessing.Rescaling(1. / 255, input_shape=data_shape),
            experimental.preprocessing.RandomFlip(),
            experimental.preprocessing.RandomRotation(0.2),
            base_model,
            layers.Dense(num_classes, activation='softmax')  # 5개의 클래스로 분류하는 Dense 층을 쌓는다.
        ])
        model.compile(  # 모델을 컴파일한다.
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        model.summary()  # 모델 구조를 확인한다.

        history = model.fit(  # 모델을 훈련한다.
            train_ds,
            batch_size=batch_size,
            epochs=1000,
            callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=10)],
            validation_data=val_ds
        )

        model.save(model_path)  # 학습한 모델을 저장한다.

        plot.plot_history([(model_name, history)])  # 학습 과정을 출력한다.

    return model


def predict_test(val_ds, model, class_names):
    for images, labels in val_ds.take(3):  # 전체 검증 셋을 배치 단위로 예측을 수행한다.
        predictions = model.predict(images)  # 예측한 결과를 저장한다.

        # 예측한 결과를 출력한다.
        num_rows = 8
        num_cols = 4
        num_images = num_rows * num_cols
        plt.figure(figsize=(9, 9))
        for i in range(num_images):
            plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
            plot.plot_image(i, predictions, labels, images, class_names)
            plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
            plot.plot_value_array(i, predictions, labels, class_names)
        plt.show()
