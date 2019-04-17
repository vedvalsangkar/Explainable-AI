# Main program run file
import time

import AE_training as ae
import CL_training as cl

if __name__ == '__main__':

    print("Program run started at", time.asctime())

    tstmp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())

    main_start = time.time()

    # ------------------------------ Start of evaluation -------------------------------

    seen_model = ae.main(train_im_folder="seen-dataset/TrainingSet/",
                         test_im_folder="seen-dataset/ValidationSet/",
                         pair_file="seen-dataset/dataset_seen_validation_siamese.csv",
                         tag="SN",
                         t_stmp=tstmp,
                         verbose=False)

    cl.main(extractor_model=seen_model,
            train_img_folder="seen-dataset/TrainingSet/",
            test_img_folder="seen-dataset/ValidationSet/",
            pair_file="seen-dataset/dataset_seen_validation_siamese.csv",
            tag="SN",
            t_stmp=tstmp,
            verbose=False)

    shuffled_model = ae.main(train_im_folder="shuffled-dataset/TrainingSet/",
                             test_im_folder="shuffled-dataset/ValidationSet/",
                             pair_file="shuffled-dataset/dataset_seen_validation_siamese.csv",
                             tag="SH",
                             t_stmp=tstmp,
                             verbose=False)

    cl.main(extractor_model=shuffled_model,
            train_img_folder="shuffled-dataset/TrainingSet/",
            test_img_folder="shuffled-dataset/ValidationSet/",
            pair_file="shuffled-dataset/dataset_seen_validation_siamese.csv",
            tag="SH",
            t_stmp=tstmp,
            verbose=False)

    unseen_model = ae.main(train_im_folder="unseen-dataset/TrainingSet/",
                           test_im_folder="unseen-dataset/ValidationSet/",
                           pair_file="unseen-dataset/dataset_seen_validation_siamese.csv",
                           tag="UN",
                           t_stmp=tstmp,
                           verbose=False)

    cl.main(extractor_model=unseen_model,
            train_img_folder="unseen-dataset/TrainingSet/",
            test_img_folder="unseen-dataset/ValidationSet/",
            pair_file="unseen-dataset/dataset_seen_validation_siamese.csv",
            tag="UN",
            t_stmp=tstmp,
            verbose=False)

    run_time = time.time() - main_start

    print("\n\nComplete pipeline trained and evaluated in {0:.03f} sec / {1:.03f} hrs".format(run_time,
                                                                                              run_time / 3600))
