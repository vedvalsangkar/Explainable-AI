# https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/conv_autoencoder.py

import time
import torch

import numpy as np

from torch import cuda, nn, optim, cosine_similarity
from torch.utils import data
from torchvision import transforms

from model_def import AndImageDataSet, AutoEncoder


def main(train_im_folder: str, test_im_folder: str, pair_file: str, tag: str, t_stmp: str):

    # -------------------------------- Hyper-parameters --------------------------------
    learning_rate = 0.0001
    lamb = 3

    epochs = 1

    batch_size = 64
    batch_print = 20

    th = 0.86

    op_dir = "models/"
    # ----------------------------------------------------------------------------------

    # --------------------------------- Fetching model ---------------------------------

    device = torch.device("cuda:0" if cuda.is_available() else "cpu")

    train_tf = transforms.Compose([transforms.RandomAffine(degrees=0,
                                                           translate=(1, 0)
                                                           ),
                                   transforms.ToTensor()])

    training_set = AndImageDataSet(transform=train_tf,
                                   img_folder=train_im_folder
                                   )

    train_loader = data.DataLoader(dataset=training_set,
                                   batch_size=batch_size,
                                   shuffle=True)

    test_set = AndImageDataSet(img_folder=test_im_folder,
                               pair_file=pair_file)

    test_loader = data.DataLoader(dataset=test_set,
                                  batch_size=batch_size,
                                  shuffle=True)

    print("Dataset and data loaders acquired.")

    model = AutoEncoder(bias=True).to(device)
    model.train(True)

    criterion = nn.MSELoss()

    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=lamb, amsgrad=False)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.4, weight_decay=lamb, nesterov=True)

    running_loss = 0.0
    past_loss = np.inf

    total_len = len(train_loader)
    # ----------------------------------------------------------------------------------

    # ------------------------------- Start of training --------------------------------
    print("\nStart of training.")
    print("Total batches in an epoch: {0}".format(total_len))

    start_time = time.time()

    for epoch in range(epochs):
        print("")
        # if (epoch + 1) % 6 == 0:
        #     learning_rate /= 2
        #     print("Learnin rate dropping to {0}".format(learning_rate))
        #     for params in optimizer.param_groups:
        #         params['lr'] = learning_rate

        for i, image in enumerate(train_loader):

            # Change variable type to match GPU requirements
            inp = image.to(device)
            # lab = labels.to(device)

            # Reset gradients before processing
            optimizer.zero_grad()

            # Get model output
            out = model(inp)

            # Calculate loss
            loss = criterion(out, inp)

            # Update weights
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % batch_print == 0:

                print(
                    "\rEpoch: {0}, step: {1}/{2} Running Loss (avg): {3:.06f}, Past: {4:.06f}    ".format(
                        epoch + 1, i + 1, total_len, (running_loss / batch_print), (past_loss / batch_print)),
                    end="")

                if running_loss < past_loss:
                    past_loss = running_loss

                running_loss = 0.0

    train_time = time.time()
    print("\nTraining completed in {0} sec\n".format(train_time - start_time))
    # ----------------------------------------------------------------------------------

    # ------------------------------ Start of evaluation -------------------------------
    print("Starting evaluation\n")

    model.eval()

    count = 0

    for i, (left_im, right_im, label, l_name, r_name) in enumerate(test_loader):

        left = left_im.to(device)
        right = right_im.to(device)
        lab = np.asarray(label)

        out1 = model(left)
        out2 = model(right)

        score = cosine_similarity(out1, out2)

        prediction = score.detach().cpu().numpy() >= th

        print(score)
        # print(prediction)
        count += (prediction == lab).sum()

    # print(count)
    acc = (count * 100)/test_set.__len__()
    print("Accuracy = {0:.06f}".format(acc))

    # ----------------------------------------------------------------------------------

    # --------------------------------- Saving model -----------------------------------
    filename = op_dir + "P2_A-{2:.03f}_{1}_T_{0}.pt".format(t_stmp, tag, acc)

    # Idea for named saved file was picked up from here:
    # https://github.com/quiltdata/pytorch-examples/blob/master/imagenet/main.py
    save_file = {"model": model.state_dict(),
                 "criterion": criterion.state_dict(),
                 "optimizer": optimizer.state_dict()
                 }

    torch.save(save_file, filename)
    # ----------------------------------------------------------------------------------


if __name__ == "__main__":

    print("Program run started at", time.asctime())

    tstmp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())

    main(train_im_folder="seen-dataset/TrainingSet/",
         test_im_folder="seen-dataset/ValidationSet/",
         pair_file="seen-dataset/dataset_seen_validation_siamese.csv",
         tag="SN",
         t_stmp=tstmp)
