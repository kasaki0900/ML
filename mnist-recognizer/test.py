import os
from models import model
from data_processing import data
import training


def main():
    data_cache = data.read_mnist(32)
    md = model.MnistRecognizerCNN(1, 64, 256)
    trainer = training.Trainer(md, learning_rate=0.2)

    print(trainer.test(data_cache.test_data))
    print(list(md.parameters())[0][0, 0, 0, 0].item())

    trainer.train(data_cache.train_data_loader, n_batch=16, epochs=160)
    print(trainer.test(data_cache.test_data))
    print(list(md.parameters())[0][0, 0, 0, 0].item())


if __name__ == "__main__":
    main()

# path = "model_para_pkl"
# i = 1
# file_path = os.path.join(path, "model"+str(i))
# print(file_path)
#
# if not os.path.exists(file_path):
#     with open(file_path, 'w') as f:
#         f.write("pkl file here")
# with open(os.path.join(path, file_name), w):

