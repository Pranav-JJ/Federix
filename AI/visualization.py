import matplotlib.pyplot as plt

# Data
models = ['Centralized', 'FedAvg', 'FedSGD']
accuracies = [98.8, 97.7, 97.8]

# Create bar graph
plt.bar(models, accuracies, color=['teal', 'green', 'orange'])

# Add a kink at 95% on the y-axis
plt.ylim(95, 100)

# Add labels and title
plt.xlabel('Model Strategies')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy of Different Models')

# Add labels for each bar
for i in range(len(models)):
    plt.text(i, accuracies[i], f'{accuracies[i]:.2f}%', ha='center', va='bottom')

# Show plot
plt.tight_layout()
plt.show()
