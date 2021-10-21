from matplotlib import pyplot as plt
from PIL import Image
from pathlib import Path

if __name__ == '__main__':
    csv_path = Path('train_result.csv')

    with csv_path.open() as f:
        x_times = 5
        y_times = 5

        fig = plt.figure()

        for i in range(1, x_times * y_times + 1):
            info = f.readline().strip().split(',')
            img = Image.open(info[0])

            ax = fig.add_subplot(x_times, y_times, i)
            imgplot = plt.imshow(img)
            ax.set_title(info[1])

    plt.show()

