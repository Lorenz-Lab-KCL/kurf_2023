from objects import OutputSpace


if __name__ == "__main__":
    records = OutputSpace.load("output_9.json")
    OutputSpace.plot_conv_dnn_last_loss_3d(records)
    # OutputSpace.plot_history_2d(records[0])
