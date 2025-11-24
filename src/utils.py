import torch
import matplotlib.pyplot as plt

def plot_result (model, distances, times):
    """
    Plots the actual data points and the models predicted line for a given dataset

    Args:
        model:the trained machine learning model to use for predictions.
        distance: The input data point for the model.
        time: the target data points for the plot
    """

    #Set the model to evaluation mode
    model.eval()

    #Disable. gradine calculation for efficient inference
    with torch.no_grad():
        #Make predictions using the trained model
        predicted_times = model(distances)

    #Create a new figure for the plot
    plt.figure(figsize=(8,6))

    #Plot the actual data points 
    plt.plot(distances.numpy(), times.numpy(), color='orange', marker='o', linestyle='None', label='Actual Delivery Times')

    #Plot the predicted line from the model
    plt.plot(distances.numpy(), predicted_times.numpy(), color='green', marker='None', label='Predicted Line')

    #Set the tile of the plot
    plt.title('Actual vs, Predicted Delievery time')

    #Set the x-axis label
    plt.xlabel('Distance (miles)')

    #Set the y-axis label
    plt.ylabel('Time(minuts)')

    #Display the legend
    plt.legend()

    #Add a grod to the plot
    plt.grid(True)
    #Show the plot
    plt.show()


def plot_nonlinear_comparision(model, new_distances, new_times):
    """
    Compares and plots the predictions of a model against new, non-liner data.

    Args:
        model: the trained model to be evaluated.
        new_distance: The new input data for generation predictions.
        new_times:The actial target values for comarison
    """

    #Set the model to evaluation mode
    model.eval()

    #Disabled the gradient disent comparion for inference
    with torch.no_grad():
        #Generate prediction using the model
        predictions = model(new_distances)

    #Create a new figure for the plot
    plt.figure(figsize=(8,6))

    #Plot the actual data point 
    plt.plot(new_distances.numpy(), new_times.numpy(), color='orange', marker='o', linestyle='None', label='Actual Data (Bikes & Cars)')

    #Plot the prediction from the model
    plt.plot(new_distances.numpy(), predictions.numpy(), color='green', marker='None', label='Linear model predictions')

    plt.title('Linear Model vs. Non-Liner Reality')
    plt.xlabel('Distance (miles)')
    plt.ylabel('Time (Minutes)')
    plt.legend()
    plt.grid(True)
    plt.show()
