from model import *


def test(hidden_size, output_size, learning_rate=0.05, epochs=200, batch_size=5000, define_batch_num=0):
    data_cache = read_mnist()
    x_train = data_cache['x_train']
    y_train = data_cache['y_train']
    x_test = data_cache['x_test']
    y_test = data_cache['y_test']

    x_train_squeezed = reshape(x_train)
    x_test_squeezed = reshape(x_test)
    y_train_onehot = one_hot(y_train)

    batch_cache = batch_partition(x_train_squeezed, y_train_onehot, batch_size=batch_size)
    model_cache = create_model(x_train_squeezed.shape[1], hidden_size, output_size, learning_rate=learning_rate)
    model = model_cache['model']

    batch_training(model_cache, batch_cache, epochs=epochs, define_batch_num=define_batch_num)
    print(testing(model, x_test_squeezed, y_test))


if __name__ == '__main__':
    test(128, 10, define_batch_num=1)
