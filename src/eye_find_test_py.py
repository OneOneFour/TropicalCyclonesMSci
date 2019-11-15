import pickle
import matplotlib.pyplot as plt
import numpy as np

with open("Data/eye_find_test", "rb") as file:
    data = pickle.load(file)
    I04data=data[0]
    I05data=data[1]

I04array = np.array(I04data)
I05array = np.array(I05data)
hot_point = np.amax(I05array)
hot_point_ind = np.unravel_index(np.argmax(I05array, axis=None), I05array.shape)

for y in range(0, hot_point_ind[0]):
    if I05array[hot_point_ind[0]-y, hot_point_ind[1]] < I05array[hot_point_ind] - 60:
        top_y = hot_point_ind[0] - y
        break
for y in range(0, len(I05array)-hot_point_ind[0]):
    if I05array[hot_point_ind[0]+y, hot_point_ind[1]] < I05array[hot_point_ind] - 60:
        bot_y = hot_point_ind[0] + y
        break
for x in range(0, hot_point_ind[1]):
    if I05array[hot_point_ind[0], hot_point_ind[1]-x] < I05array[hot_point_ind] - 60:
        left_x = hot_point_ind[1] - x
        break
for x in range(0, len(I05array[0])-hot_point_ind[1]):
    if I05array[hot_point_ind[0], hot_point_ind[1]+x] < I05array[hot_point_ind] - 60:
        right_x = hot_point_ind[1] + x
        break

eye_data = I05array[top_y:bot_y, left_x:right_x]
mid_row = int((top_y - bot_y) / 2 + bot_y)
width = 3
eye_row_I05 = np.mean(I05array[mid_row-width:mid_row+width, left_x:right_x], axis=0)
eye_row_I04 = np.mean(I04array[mid_row-width:mid_row+width, left_x:right_x], axis=0)
plt.scatter(eye_row_I04, eye_row_I05)

#fig, ax = plt.subplots()
#im = ax.imshow(eye_row, cmap="jet")
#fig.colorbar(im)

plt.show()
