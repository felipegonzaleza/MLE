import matplotlib.pyplot as plt
import tensorflow as tf

def print_loss_history(training_history, logscale=False):
    loss = training_history['loss']
    val_loss = training_history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, color='red', label='Training loss')
    plt.plot(epochs, val_loss, color='green', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    if logscale:
        plt.yscale('log')
    plt.show()
    return

def display_activation(activations, col_size, row_size, act_index): 
    activation = activations[act_index]
    activation_index=0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
    for row in range(0,row_size):
        for col in range(0,col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='plasma')
            activation_index += 1   
    return
    
def calc_metrics(y_true, y_pred):
    # calculate metrics
    y_true = tf.cast(y_true, tf.float64)
    y_pred = tf.cast(y_pred, tf.float64)
    
    TP = tf.math.count_nonzero(y_pred * y_true, axis=None)
    FP = tf.math.count_nonzero(y_pred * (y_true - 1), axis=None)
    FN = tf.math.count_nonzero((y_pred - 1) * y_true, axis=None)
    TN = tf.math.count_nonzero((y_pred - 1) * (y_true - 1), axis=None)
    
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    acc = (TP + TN) / (TP + FP + FN + TN)
    f1 = 2 * precision * recall / (precision + recall)
    
    precision = tf.reduce_mean(precision)
    recall = tf.reduce_mean(recall)
    acc = tf.reduce_mean(acc)
    f1 = tf.reduce_mean(f1)
    
    return acc.numpy(), precision.numpy(), recall.numpy(), f1.numpy()
    