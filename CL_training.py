import time
import pandas as pd
import numpy as np

import torch

from torch import cuda, nn, optim, cosine_similarity
from torch.utils import data

from model_def import AutoEncoder, FeatureClassifier, FeatureClassifierDataSet, plot_roc

import AE_training as ae


def main(extractor_model: AutoEncoder, train_img_folder: str, test_img_folder: str,
         pair_file: str, tag: str, t_stmp: str):
    # -------------------------------- Hyper-parameters --------------------------------
    learning_rate = 0.0001
    lamb = 0.001
    epochs = 10

    batch_size = 32
    batch_print = 50

    threshold = 0.8

    sgd_momentum = 0.1

    op_dir = "models/"
    # ----------------------------------------------------------------------------------

    # --------------------------------- Fetching model ---------------------------------

    device = torch.device("cuda:0" if cuda.is_available() else "cpu")

    training_set = FeatureClassifierDataSet(img_folder=train_img_folder)

    train_loader = data.DataLoader(dataset=training_set,
                                   batch_size=batch_size,
                                   shuffle=True)

    test_set = FeatureClassifierDataSet(img_folder=test_img_folder,
                                        pair_file=pair_file)

    test_loader = data.DataLoader(dataset=test_set,
                                  batch_size=batch_size,
                                  shuffle=True)

    print("Dataset and data loaders acquired.")

    # for i, (left_im, right_im, label, l_name, r_name) in enumerate(test_loader):
    #     print(l_name)
    #     print(type(l_name))

    extractor_model.to(device)
    extractor_model.eval()

    # criterion = nn.CrossEntropyLoss()

    models = dict()
    optimizers = dict()
    criteria = dict()

    class_nums = get_class_nums()

    for i, val in enumerate(class_nums):
        models[i] = FeatureClassifier(num_classes=val, temprature=0.9, bias=True).to(device)
        models[i].train(True)

        optimizers[i] = optim.Adam(params=models[i].parameters(),
                                   lr=learning_rate,
                                   weight_decay=lamb,
                                   amsgrad=True
                                   )
        # optimizers[i] = optim.SGD(params=models[i].parameters(),
        #                           lr=learning_rate,
        #                           momentum=sgd_momentum,
        #                           weight_decay=lamb,
        #                           nesterov=True
        #                           )

        criteria[i] = nn.CrossEntropyLoss()

    # optimizer = optim.Adam(params=model.parameters(),
    #                        lr=learning_rate,
    #                        weight_decay=lamb,
    #                        amsgrad=False
    #                        )

    total_len = len(train_loader)
    # ----------------------------------------------------------------------------------

    # ------------------------------- Start of training --------------------------------
    print("\nStart of training.")
    print("Total batches in an epoch: {0}".format(total_len))

    start_time = time.time()

    cor = 0
    tot = 0

    for model_num in range(15):

        print("\n\nFeature {0}: ".format(model_num + 1), end="")

        for epoch in range(epochs):
            print("")

            for i, (image, feats) in enumerate(train_loader):

                inp = extractor_model(image.to(device))
                feat = feats.to(device)

                # if (i + 1) % batch_print == 0:
                #     print("\rE{0}, Step{1}/{2}".format(epoch, i, total_len), end="\t")

                # for model_num in range(15):

                # Reset gradients before processing
                optimizers[model_num].zero_grad()

                # Get model output
                out = models[model_num](inp)

                # Calculate loss
                loss = criteria[model_num](out, feat[:, model_num])

                # Accuracy
                _, predicted = torch.max(out.data, 1)

                a = predicted.cpu().numpy()
                b = feats.numpy()[:, model_num]

                cor += (a == b).sum()
                tot += predicted.size(0)

                # Update weights
                loss.backward()
                optimizers[model_num].step()

                if (i + 1) % batch_print == 0:

                    acc = (100 * cor) / tot

                    print(
                        "\rFeature: {0}, Epoch: {1}, Step: {2}/{3}, Loss: {4:.04f}, Running Accuracy: {5:.06f}".format(
                            model_num + 1, epoch + 1, i + 1, total_len, loss.item(), acc),
                        end="")
                    cor = 0
                    tot = 0

                    # if acc > 85:
                    #     break

    train_time = time.time()
    print("\n\nTraining completed in {0:.03f} sec\n".format(train_time - start_time))
    # ----------------------------------------------------------------------------------

    # ------------------------------ Start of evaluation -------------------------------
    print("Starting evaluation\n")

    for model_num in range(15):
        models[model_num].eval()

    col_names = ['pen_pressure', 'letter_spacing', 'size', 'dimension', 'is_lowercase', 'is_continuous',
                 'slantness', 'tilt', 'entry_stroke_a', 'staff_of_a', 'formation_n', 'staff_of_d',
                 'exit_stroke_d', 'word_formation', 'constancy']

    results = pd.DataFrame()

    # for i, (left_im, right_im, label, left_feat, right_feat, l_name, r_name) in enumerate(test_loader):
    for i, (left_im, right_im, label, l_name, r_name) in enumerate(test_loader):

        batch = pd.DataFrame()

        batch["left_image"] = list(l_name)
        batch["right_image"] = list(r_name)

        left_vector = extractor_model(left_im.to(device))
        right_vector = extractor_model(right_im.to(device))

        overall_score = cosine_similarity(left_vector, right_vector).detach().cpu().numpy()

        # feature_score = dict()
        pred = pd.DataFrame()

        for model_num in range(15):
            left_softmax = models[model_num](left_vector)
            right_softmax = models[model_num](right_vector)

            # feature_score[model_num] = cosine_similarity(left_softmax, right_softmax).detach().cpu().numpy()
            batch[col_names[model_num]] = cosine_similarity(left_softmax, right_softmax).detach().cpu().numpy()

            if pred.empty:
                pred = batch[col_names[model_num]] > threshold
            else:
                pred = pred & (batch[col_names[model_num]] > threshold)

        batch["overall_score"] = overall_score

        batch["dl_prediction"] = (overall_score > threshold) * 1  # for boolean to int conversion

        batch["xai_prediction"] = pred

        batch["actual"] = label.numpy()

        results = results.append(batch, ignore_index=True)

    results.to_csv(path_or_buf="results/RESULTS_{0}_{1}.csv".format(tag, t_stmp),
                   index=False)

    plot_roc(y_true=results["actual"], y_score=results["overall_score"], tag=tag, tstmp=t_stmp)

    # ----------------------------------------------------------------------------------


def get_class_nums():
    return pd.read_csv(filepath_or_buffer="15FeatureClassValueInformation.csv",
                       header=0,
                       index_col=0
                       ).count().values


if __name__ == '__main__':
    print("Program run started at", time.asctime())

    tstmp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())

    loaded = torch.load(f="models/P2_A-81.347_SN_T_20190405_163825.pt")

    feat_model = AutoEncoder(bias=True)

    feat_model.load_state_dict(loaded["model"])

    main(extractor_model=feat_model,
         train_img_folder="seen-dataset/TrainingSet/",
         test_img_folder="seen-dataset/ValidationSet/",
         pair_file="seen-dataset/dataset_seen_validation_siamese.csv",
         tag="SN",
         t_stmp=tstmp)
