# Some aspects of this program were taken from this repository:
# https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/conv_autoencoder.py
#
# Structure of the autoencoder and saving of the output while testing as an image were picked up.

import time
import torch

import numpy as np

from torch import cuda, nn, optim, cosine_similarity
from torch.utils import data
# from torchvision import transforms
from torchvision.utils import save_image

from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt

from model_def import FeatureExtractorDataSet, AutoEncoder


def main(train_im_folder: str, test_im_folder: str, pair_file: str, tag: str, t_stmp: str, train_only: bool = False):

    # -------------------------------- Hyper-parameters --------------------------------
    learning_rate = 0.005
    lamb = 0.04
    epochs = 25

    batch_size = 32
    batch_print = 20

    threshold = 0.85

    sgd_momentum = 0.3

    op_dir = "models/"
    # ----------------------------------------------------------------------------------

    # --------------------------------- Fetching model ---------------------------------

    device = torch.device("cuda:0" if cuda.is_available() else "cpu")

    training_set = FeatureExtractorDataSet(img_folder=train_im_folder)

    train_loader = data.DataLoader(dataset=training_set,
                                   batch_size=batch_size,
                                   shuffle=True)

    test_set = FeatureExtractorDataSet(img_folder=test_im_folder,
                                       pair_file=pair_file)

    test_loader = data.DataLoader(dataset=test_set,
                                  batch_size=batch_size,
                                  shuffle=True)

    print("Dataset and data loaders acquired.")

    model = AutoEncoder(bias=True).to(device)
    model.train(True)

    criterion = nn.MSELoss()

    # optimizer = optim.Adam(params=model.parameters(),
    #                        lr=learning_rate,
    #                        weight_decay=lamb,
    #                        amsgrad=False
    #                        )

    optimizer = optim.SGD(params=model.parameters(),
                          lr=learning_rate,
                          momentum=sgd_momentum,
                          weight_decay=lamb,
                          nesterov=True
                          )

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

            if i+1 == total_len and epoch == 24:
                # print(out[0])
                save_image(tensor=out[:4],
                           filename="visualization/and_Set_{0}_E{1}.png".format(tag, epoch),
                           nrow=2,
                           normalize=True)

    train_time = time.time()
    print("\nTraining completed in {0} sec\n".format(train_time - start_time))
    # ----------------------------------------------------------------------------------

    # if train_only:
    #     return model

    # ------------------------------ Start of evaluation -------------------------------
    print("Starting evaluation\n")

    model.eval()

    count = 0
    tot = 0

    total_score = np.array([])
    total_label = np.array([])

    for i, (left_im, right_im, label) in enumerate(test_loader):   # l_name, r_name

        try:

            left = left_im.to(device)
            right = right_im.to(device)
            lab = np.asarray(label)

            total_label = np.append(total_label, lab)

            out1 = model(left)
            out2 = model(right)

            score = cosine_similarity(out1, out2).detach().cpu().numpy()

            total_score = np.append(total_score, score)

            prediction = score >= threshold

            # print(score)
            # print(prediction)
            count += (prediction == lab).sum()
            tot += left_im.shape[0]

        except FileNotFoundError:
            print("File not found: {0}. Skipping iteration {1}".format(FileNotFoundError.filename, i))

    plot_roc(y_true=total_label, y_score=total_score, tag=tag, tstmp=t_stmp)

    # print(count)
    acc = (count * 100)/tot
    print("\nAccuracy = {0:.06f} %\n\n".format(acc))

    # ----------------------------------------------------------------------------------

    if train_only:
        return model

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


def plot_roc(y_true, y_score, tag, tstmp):

    sets = {"SN": "Seen", "UN": "Unseen", "SH": "Shuffled"}

    fpr, tpr, thresh = roc_curve(y_true=y_true, y_score=y_score)
    area = auc(fpr, tpr)

    print("Thresholds: ", thresh)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label="ROC curve (area = {0:.2f})".format(area))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("Receiver Operating Characteristic for {} data".format(sets[tag]))
    plt.legend(loc="lower right")
    plt.savefig(fname="visualization/ROC_{0}_{1}.jpg".format(tag, tstmp))
    # plt.show()


if __name__ == "__main__":

    print("Program run started at", time.asctime())

    tstmp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())

    main(train_im_folder="seen-dataset/TrainingSet/",
         test_im_folder="seen-dataset/ValidationSet/",
         pair_file="seen-dataset/dataset_seen_validation_siamese.csv",
         tag="SN",
         t_stmp=tstmp)

    main(train_im_folder="shuffled-dataset/TrainingSet/",
         test_im_folder="shuffled-dataset/ValidationSet/",
         pair_file="shuffled-dataset/dataset_seen_validation_siamese.csv",
         tag="SH",
         t_stmp=tstmp)

    main(train_im_folder="unseen-dataset/TrainingSet/",
         test_im_folder="unseen-dataset/ValidationSet/",
         pair_file="unseen-dataset/dataset_seen_validation_siamese.csv",
         tag="UN",
         t_stmp=tstmp)
