import numpy as np
import os
import datetime
import shutil

def main():

    # Directory containing images you wish to convert
    print(datetime.datetime.now())
    print('Start import data and save label lists')
    input_dir = "ULSOrgans"
    new_input_dir = "ULSOrgans_Split"
    directories = os.listdir(input_dir)

    # index is number of class
    index = 0

    # index2 is number of image
    index2 = 0
    directory = 'train'
    index_per_dir = 0
    classes_dict = {}
    num_of_classes = 18
    train_label_list = [0]*num_of_classes
    validation_label_list = [0] * num_of_classes
    train_label_list_index = 0
    validation_label_list_index = 0

    for folder in directories:
        # Ignoring metadata csv file
        if folder == '7ce293e2-c2b9-4c2e-b728-b46e12612f46.csv':
            pass

        else:
            print(datetime.datetime.now())
            print('Start import data from folder: %s' % folder)
            index_per_dir = 0
            folder2 = os.listdir(input_dir + '/' + folder)
            classes_dict[index] = folder
            train_label_list_index = 0
            validation_label_list_index = 0

            for image in folder2:

                index2 += 1
                index_per_dir += 1

                if index_per_dir%5 == 0:
                    directory = 'validation'
                    validation_label_list_index += 1
                else:
                    directory = 'train'
                    train_label_list_index += 1

                source_img = input_dir + '/' + folder + '/' + image
                dest_img = new_input_dir + '/' + directory + '/' + folder + '/' + image
                shutil.copy(source_img, dest_img)

            if folder == 'ABDOMEN_LIVER':
                validation_label_list_index -= 8
            train_label_list[index] = train_label_list_index
            validation_label_list[index] = validation_label_list_index
            print(datetime.datetime.now())
            total_images_in_folders = train_label_list_index + validation_label_list_index
            print('Finish folder with %d images' % total_images_in_folders)
            index += 1

    np.save('train_label_list.npy', train_label_list)  # Saving train_label_list
    np.save('validation_label.npy', validation_label_list)  # Saving validation_label_list
    print(datetime.datetime.now())
    print('Finish running with %d images' % index2)

    return

if __name__ == '__main__':
    main()
