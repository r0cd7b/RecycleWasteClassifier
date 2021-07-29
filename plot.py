import numpy as np
import matplotlib.pyplot as plt


def plot_history(histories):
    plt.figure(figsize=(9, 9))

    plt.subplot(2, 1, 1)
    key = 'loss'
    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_' + key], '--', label=name + ' Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(), label=name + ' Train')
    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_', ' ').title())
    plt.legend()

    plt.subplot(2, 1, 2)
    key = 'accuracy'
    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_' + key], '--', label=name + ' Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(), label=name + ' Train')
    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_', ' ').title())
    plt.legend()

    plt.show()


def plot_image(i, predictions_array, true_label, img, class_names):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img.numpy().astype("uint8"))
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel(
        f"{class_names[predicted_label]} {100 * np.max(predictions_array):2.0f}% ({class_names[true_label]})",
        color=color,
        fontsize=6
    )


def plot_value_array(i, predictions_array, true_label, class_names):
    num_classes = len(class_names)
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks(range(num_classes), class_names, rotation=45, fontsize=6)
    plt.yticks([])
    plt.ylim([0, 1])
    this_plot = plt.bar(range(num_classes), predictions_array, color="#777777")
    predicted_label = np.argmax(predictions_array)
    this_plot[predicted_label].set_color('red')
    this_plot[true_label].set_color('blue')
