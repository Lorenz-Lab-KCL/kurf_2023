To use each folder, copy and unzip both polymer tarball files in them

For the moe_list_encoder_decoder folder, you can run encoder_decoder_ml.py to train a model with a certain architecture. You can then import the model in the aed_visualisation.py and run it on a few molecules to see its accuracy.

For the gnn_training folder, run main training with the train_test function uncommented to train a model, or without it to load one of the preexisting models. You can then enter the filename (without .json) stored in the polymer_db_full folder to run the model on that file.
