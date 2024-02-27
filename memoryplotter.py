import matplotlib.pyplot as plt
import csv

# Function to read data from CSV file
def read_csv(filename):
    data = {}
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            label = row[0]
            values = [int(value) for value in row[1:]]
            data[label] = values
    return data

# Function to plot data
def plot_data(data):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    plots = [(0, 0), (0, 1), (1, 0), (1, 1)]
    labels = list(data.keys())

    for i, ax in enumerate(axs.flat):
        multi_label = labels[i * 2]
        cumul_label = labels[i * 2 + 1]
        multi_values = data[multi_label]
        cumul_values = data[cumul_label]
        x_values = list(range(1, len(multi_values) + 1))
        ax.plot(x_values, multi_values, label=multi_label)
        ax.plot(x_values, cumul_values, label=cumul_label)
        ax.set_title(f'Plot {i+1}')
        ax.legend()

    plt.tight_layout()
    plt.show()

# Main function
def main():
    filename = 'memory.csv'  # Assuming the CSV file is named 'data.csv'
    data = read_csv(filename)
    plot_data(data)

if __name__ == "__main__":
    main()