import matplotlib.pyplot as plt


def visualize(y_train, y_test, pred_out, test_predicted_values):

    plt.plot(y_train, pred_out, 'ro',
             label='Correlation of Original with Predicted train data')
    # Above is for marking points in space of labels and features
    plt.plot(y_test, test_predicted_values, 'bo',
             label='Correlation of Original with Predicted test data')
    plt.title('Comparison of the predicting performance')
    plt.ylabel('Predicted data')
    plt.xlabel('Original data')
    plt.savefig('Correlation.png')
    plt.legend()    # This enable the label over the plot display
    plt.show()      # This is important for showing the plotted graph

if __name__ == '__main__':

    visualize(y_train, y_test, pred_out, test_predicted_values)
