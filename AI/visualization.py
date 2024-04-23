import matplotlib.pyplot as plt

# Data
models = ['Centralized', 'FedAvg', 'FedSGD']
accuracies = [99.854, 99.712, 99.7026]

# Create bar graph
plt.bar(models, accuracies, color=['teal', 'green', 'orange'])

# Add a kink at 95% on the y-axis
plt.ylim(99, 100)

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
