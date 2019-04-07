# Main program run file
import time

import AE_training as ae
import CL_training as cl

if __name__ == '__main__':

    print("Program run started at", time.asctime())

    tstmp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())

    # ------------------------------ Start of evaluation -------------------------------

    seen_model = ae.main(train_im_folder="seen-dataset/TrainingSet/",
                         test_im_folder="seen-dataset/ValidationSet/",
                         pair_file="seen-dataset/dataset_seen_validation_siamese.csv",
                         tag="SN",
                         t_stmp=tstmp,
                         train_only=True)

    cl.main(extractor_model=seen_model,
            train_img_folder="seen-dataset/TrainingSet/",
            test_img_folder="seen-dataset/ValidationSet/",
            pair_file="seen-dataset/dataset_seen_validation_siamese.csv",
            tag="SN",
            t_stmp=tstmp)

    shuffled_model = ae.main(train_im_folder="shuffled-dataset/TrainingSet/",
                             test_im_folder="shuffled-dataset/ValidationSet/",
                             pair_file="shuffled-dataset/dataset_seen_validation_siamese.csv",
                             tag="SH",
                             t_stmp=tstmp,
                             train_only=True)

    cl.main(extractor_model=shuffled_model,
            train_img_folder="shuffled-dataset/TrainingSet/",
            test_img_folder="shuffled-dataset/ValidationSet/",
            pair_file="shuffled-dataset/dataset_seen_validation_siamese.csv",
            tag="SH",
            t_stmp=tstmp)

    unseen_model = ae.main(train_im_folder="unseen-dataset/TrainingSet/",
                           test_im_folder="unseen-dataset/ValidationSet/",
                           pair_file="unseen-dataset/dataset_seen_validation_siamese.csv",
                           tag="UN",
                           t_stmp=tstmp,
                           train_only=True)

    cl.main(extractor_model=unseen_model,
            train_img_folder="unseen-dataset/TrainingSet/",
            test_img_folder="unseen-dataset/ValidationSet/",
            pair_file="unseen-dataset/dataset_seen_validation_siamese.csv",
            tag="UN",
            t_stmp=tstmp)
