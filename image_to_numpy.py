import os
import numpy as np
import cv2

# Directory containing images you wish to convert
input_dir = "C:\\gitprojects\\GE_project\\image"

directories = os.listdir(input_dir)

index = 0
index2 = 0
classes_dict = {}

for folder in directories:
    # Ignoring .DS_Store dir
    if folder == '7ce293e2-c2b9-4c2e-b728-b46e12612f46.csv':
        pass

    else:
        print(folder)

        folder2 = os.listdir(input_dir + '\\' + folder)
        index += 1
        classes_dict[index] = folder

        for image in folder2:
            if image == ".DS_Store":
                pass

            else:
                index2 += 1

                im = cv2.resize(cv2.imread(input_dir+"\\"+folder+"\\"+image), (224, 224)).astype(np.float32)
                # im[:, :, 0] -= 103.939
                # im[:, :, 1] -= 116.779
                # im[:, :, 2] -= 123.68
                im = im.transpose((2, 0, 1))
                im1 = np.expand_dims(im, axis=0)

                try:
                    # r = im[:, :, 0]  # Slicing to get R data
                    # g = im[:, :, 1]  # Slicing to get G data
                    # b = im[:, :, 2]  # Slicing to get B data

                    if index2 != 1:
                        # new_array = np.array([[r] + [g] + [b]], np.uint8)  # Creating array with shape (3, 224, 224)
                        # out = np.append(out, new_array,
                        #                 0)  # Adding new image to array shape of (x, 3, 100, 100) where x is image number
                        out1 = np.append(out1, im1, 0)

                    elif index2 == 1:
                        # out = np.array([[r] + [g] + [b]], np.uint8)  # Creating array with shape (3, 100, 100)
                        out1 = im1

                    if index == 1 and index2 == 1:
                        index_array = np.array([[index]])

                    else:
                        new_index_array = np.array([[index]], np.int8)
                        index_array = np.append(index_array, new_index_array, 0)

                except Exception as e:
                    print(e)
                    print("Removing image" + image)
                    os.remove(input_dir + "\\" + folder + "\\" + image)

print(index)
Rmean = np.mean(out1[:, :, 0])
Rstd = np.std(out1[:, :, 0])
Gmean = np.mean(out1[:, :, 1])
Gstd = np.std(out1[:, :, 1])
Bmean = np.mean(out1[:, :, 2])
Bstd = np.std(out1[:, :, 2])

np.save('X_train.npy', out1)  # Saving train image arrays
np.save('Y_train.npy', index_array)  # Saving train labels
