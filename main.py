import os.path as pth
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy as np
import json

# change this to the number of samples you have from each coil
NUM_SAMPLES = 3
AREA = 'area'
stDev = 'stdev'


def main():
    file_name = 'a.txt'  # check_file_name()
    spec_with_name = run_save_concise_data(file_name)
    num_spec = spec_with_name[0]
    file_name_processed = spec_with_name[1]
    compressed_dict = group_data(file_name_processed, num_spec)
    final_fname = write_average(file_name_processed, compressed_dict, num_spec)
    prepare_plot(final_fname, num_spec)


def empty_file(fname):
    """
    deletes contents of a file so that we don't get redundant data
    :param fname: the name of the file
    :return: None
    """
    with open(fname, 'w'):
        pass


def check_file_name():
    """
    asks user for a file name, ensures it exists
    :return: the file name
    """
    file_name = get_file_name()
    while not pth.exists(file_name):
        file_name = get_file_name()
    return file_name


def get_file_name():
    """
    :return: user's file name
    """
    return input("Enter the file name, no extension ") + '.txt'


def get_info(data, str_to_find, num_sets):
    """
    reads the file and returns which line holds a certain string, along with the second
    element in that line
    :param data: the file we're using
    :param str_to_find: the string the user is looking for
    :param num_sets: the number of times we want to skip before returning the string (ie if I
    want to find the 10th ocurrence of 'Data File Name' I would set num_sets = 10)
    :return: the number of species found and the line number the species start at
    """
    counter = total_sets = 1
    with open(data, 'r') as file:
        for line in file:
            if str_to_find.lower() in line.lower():
                if total_sets == num_sets:
                    split_line = line.split("\t", 2)
                    try:
                        num_spec = split_line[1].strip("\n")
                    except IndexError:
                        num_spec = '0'
                    return [counter, num_spec]
                else:
                    total_sets += 1
            else:
                counter += 1


def remove_trash_lines(data, num_lines):
    """
    gets rid of all the lines that aren't needed
    :param data: the file we are reading
    :param num_lines: the number of lines we don't need
    :return: the first line we are interested in
    """
    with open(data, 'r') as file:
        last_line = file.readlines()[num_lines:num_lines + 1]
    return last_line


def get_area_col(area_row):
    """
    finds the column that 'Area' is in
    :param area_row: the row that 'Area' is in
    :return: the column number where 'Area' is
    """
    area_row = delete_bad_chars(area_row)
    for counter, column in enumerate(area_row, start=1):
        if column.lower() == AREA:
            return counter


def delete_bad_chars(string):
    """
    removes unnecessary characters from a string and splits it at a tab
    :param string: the string we want to do this to
    :return: the properly formatted string
    """
    string = string.strip("[]'")
    return string.split("\\t")


def save_concise_data(new_file_name, area_col, num_trash_lines, num_spec, name_of_dataset,
                      counter, data):
    """
    writes the data we want to the file
    :param new_file_name: the name of the new file we are using
    :param area_col: the column 'Area' is in
    :param num_trash_lines: the number of lines we don't need
    :param num_spec: the number of species
    :param name_of_dataset: the name of the .qgd file
    :param counter: the number of sets we have analyzed
    :param data: the file we are reading from
    :return: None
    """
    with open(new_file_name, 'a') as file:
        if counter != 1:
            file.write('\n' + name_of_dataset + '\n')
        else:
            file.write(name_of_dataset + '\n')
        write_to_file(data, counter, num_trash_lines, num_spec, file, area_col)


def write_to_file(data, counter, num_trash_lines, num_spec, file, area_col):
    """
    writes the concise info to the file (just the name of the dataset, the species, and its area)
    :param data: the old file
    :param counter: how many times we have done this
    :param num_trash_lines: the number of lines that we don't need
    :param num_spec: the number of species
    :param file: the new file
    :param area_col: the column 'Area' is in
    :return: None
    """
    for data_lines in range(num_spec):
        line = str(remove_trash_lines(data, counter * num_trash_lines +
                                      (num_spec + 1) * (counter - 1) + data_lines))
        line = delete_bad_chars(line)
        file.write(line[1] + ' ' + line[area_col - 1] + '\n')


def get_file_len(data):
    """
    finds how many lines are in the file
    :param data: the file
    :return: the number of lines
    """
    with open(data, 'r') as file:
        for i, l in enumerate(file):
            pass
    # the files can have a strange character at the end that can't be encoded so we want to
    # return a number of lines strictly less than that
    return i


def run_save_concise_data(data):
    """
    creates a new file and writes the indiv. data set name, the species' names, and their areas
    :param data: the file we are reading
    :return:
    """
    counter = 1
    # this list contains the number of trash lines as well as the number of species
    trash_line_with_num_spec = get_info(data, "# of IDs", counter)
    num_trash_lines = trash_line_with_num_spec[0]
    num_spec = int(trash_line_with_num_spec[1])
    # finds the row with 'Area'
    area_row = str(remove_trash_lines(data, num_trash_lines))
    # finds the column with 'Area'
    area_col = get_area_col(area_row)
    num_trash_lines += 1
    new_file_name = data[:-4] + '_processed.txt'
    total_lines = get_file_len(data)
    total_dataset = num_spec + num_trash_lines + 1
    empty_file(new_file_name)
    # as long as there is another set of data to analyze, we get the info  we need
    while (total_dataset * counter) <= total_lines:
        name_of_dataset = get_info(data, 'Data File Name', counter)[1].split("\\")[-1]
        save_concise_data(new_file_name, area_col, num_trash_lines, num_spec, name_of_dataset,
                          counter, data)
        counter += 1
    return [num_spec, new_file_name]


def group_data(data, num_spec):
    """
    takes the concise data and merges the 3 samples from each coil, giving us the average that is to
    be graphed
    :param data: the file the concise data is in
    :param num_spec: the number of species
    :return: a dictionary with the key the name of the species, the value the three different
    areas displayed in the file
    """
    names_to_compare = []
    compressed_dict = {}
    with open(data, 'r') as file:
        lines = file.readlines()
    total_lines = get_file_len(data)
    for counter in range(0, total_lines, num_spec + 2):
        # gets the name of the dataset without extra info
        name_of_dataset = lines[counter][:-1].strip('.gqd')
        if '_' in name_of_dataset:
            name = name_of_dataset[:name_of_dataset.index('_')]
        else:
            name = name_of_dataset[:-3]
        # fills an array with all the names
        names_to_compare.append(name)
    # finds all the coils and creates a dictionary with their respective location in the file,
    # we will use this dictionary to find the average of the area for each of the species from
    # the three samples
    for name in range(0, len(names_to_compare)):
        name_to_find = names_to_compare[name]
        if name_to_find not in compressed_dict:
            compressed_dict[name_to_find] = []
            for itr in range(0, NUM_SAMPLES):
                try:
                    current_index = \
                        [num for num, check_name in enumerate(names_to_compare) if check_name ==
                         name_to_find][itr]
                    compressed_dict[name_to_find].append(current_index)
                except IndexError:
                    print("error at ", itr, " with name ", name_to_find, ", exiting program",
                          sep='')
                    break
    return compressed_dict


def write_average(fname, names_dict, num_spec):
    """
    takes the data and averages the values from each of the shots, ie if you have three aliquots
    from each coil then it averages those three and just writes the average and the species name
    to the new file
    :param fname: the file with the processed, but not averaged, data
    :param names_dict: the dictionary containing the coil names and their indices in the file
    :param num_spec: the number of species we are working with
    :return: the name of the averaged file
    """
    final_file = fname.strip("_processed.txt") + '_averaged.txt'
    # the number of lines in a total set (it has the name of the coil, the 4 species, then a '\n'
    len_of_set = num_spec + 2
    # gets the processed data
    with open(fname, 'r') as file:
        lines = file.readlines()
    empty_file(final_file)
    errors = {}  # {Coil1: {Ar: 1, Ne: 2, ...}, ...}
    # names_dict ({'Ar': 5555555}), goes through all the datasets and takes the average of the
    # coil samples (rounding to 4 decimal places)
    for keys in names_dict:
        # collects the complete averaged data
        all_aliquots = {}
        data_indices = names_dict[keys]  # array
        all_areas = {}
        for index in range(len(data_indices)):

            # this is essentially the sample number
            location_in_file = data_indices[index]
            single_set = lines[len_of_set * location_in_file + 1: (location_in_file + 1) *
                                                                  len_of_set - 1]
            for i in range(num_spec):
                manageable_arr = single_set[i].split(' ')
                name = manageable_arr[0]
                area = int(manageable_arr[1].strip('\n'))
                if name not in all_areas:
                    all_areas[name] = [area]
                else:
                    all_areas[name].append(area)
                if name not in all_aliquots:
                    all_aliquots[name] = {}
                    all_aliquots[name][AREA] = area
                else:
                    all_aliquots[name][AREA] += area
        # averages data of each of the coil aliquots, gets standard deviation
        for name in all_aliquots:
            all_aliquots[name][AREA] = round(all_aliquots[name][AREA] / NUM_SAMPLES, 4)
            all_aliquots[name][stDev] = np.std(all_areas[name])
        errors[keys] = {}
        errors[keys] = all_aliquots

    with open(final_file, 'a') as json_file:
        json.dump(errors, json_file, indent=2)
    return final_file


def prepare_plot(fname, num_spec):
    """
    gets the data ready to graph by putting the species' names and areas into arrays
    :param fname: the file where the averaged data is
    :param num_spec: the number of species in the dataset
    :return: None
    """
    len_of_set = num_spec + 2
    # number of characters in the species name (ie 'Ar' has 2 chars)
    len_of_spec_name = 2
    with open(fname, 'r') as file:
        file_lines = json.load(file)
    for coil in file_lines:
        # TODO prepare data for charting
        print(coil)
    # file_len = get_file_len(fname)
    # grabs the indiv. dataset area
    array_of_spec_names = []
    array_of_areas = []
    entry_names = []
    for line in range(0, len(file_lines) // len_of_set):
        single_set = file_lines[line * len_of_set: (line + 1) * len_of_set - 1]
        entry_names.append(single_set[0].strip('\n'))
        for entry in range(1, len(single_set)):
            spec_name = single_set[entry][0:len_of_spec_name]
            area = single_set[entry][len_of_spec_name + 1:]
            array_of_areas.append(area)
            array_of_spec_names.append(spec_name)
    for area in range(len(array_of_areas)):
        array_of_areas[area] = float(array_of_areas[area].strip('\n'))
    create_plots(array_of_areas, array_of_spec_names, entry_names, num_spec, fname)


# TODO add typing in param names
def norm_data(name_and_area, entry_names, num_spec):
    """
    Normalizes to the user's chosen species

    :param dict name_and_area: A dictionary object with key the species-name, the value an array of
    all the areas from all the coils in the correct order
    :param list entry_names: List of the names of the coils, in the order that corresponds to the
    list of areas in the dictionary name_and_area
    :param int num_spec: the number of species in the txt file
    :return [dict, str]: array with first index a 2d dictionary with the key being the coil names,
    the value being a dictionary of the form {species: normalized area, species: normalized area, ...};
    second index the name of the species we are normalizing to
    """
    spec_to_norm_to = input(
        'Enter the name of the species that you want to normalize to ')
    # find out if the spec name ('Ar' for example) is in the list of spec names, then divide all
    # species' areas by that area
    while spec_to_norm_to not in name_and_area:
        spec_to_norm_to = input(spec_to_norm_to + ' not found, try again.\n(be sure to write the '
                                                  'case-sensitive name) ')
    coils_and_areas = {}
    errors = {}
    for name in name_and_area:
        errors[name] = []
    print('errors is', errors, 'need to use', name_and_area)
    for index, coil in enumerate(entry_names):
        coils_and_areas[coil] = {}
        for spec_name in name_and_area:
            area_to_norm_to = name_and_area[spec_to_norm_to][index]
            coils_and_areas[coil][spec_name] = name_and_area[spec_name][index] / area_to_norm_to

    return [coils_and_areas, spec_to_norm_to]


def find_bounds(upper_or_lower, max_or_min):
    bound = f"{upper_or_lower:.20f}"
    bound = str(bound)
    extreme_y_val = 1
    if max_or_min == 'max':
        bound = bound[0:bound.index('.')]
        num_zeroes = len(bound)
        for zero in range(num_zeroes):
            extreme_y_val *= 10
    else:
        bound = bound[(bound.index('.') + 1):len(bound)]
        while (len(bound) > 0) and (bound[0] == str(0)):
            extreme_y_val /= 10
            bound = bound[1:len(bound)]
        extreme_y_val /= 10
    return extreme_y_val


def create_plots(array_of_areas, array_of_spec_names, entry_names, num_spec, fname):
    """
    makes the charts
    TODO get names to norm to and normalize all the datasets, not just one
    :param array_of_areas: the areas in each coil
    :param array_of_spec_names: the species names in the coils
    :param entry_names: the names of all the coils that we are analyzing
    :param num_spec: the number of species that are being normalized
    :param fname: name of the text file, so we can get the absolute path
    :return: None
    """
    name_area_dict = {}
    for name in array_of_spec_names:
        location = [i for i, n in enumerate(array_of_spec_names) if n is name][0]
        if name_area_dict.get(name) is None:
            name_area_dict[name] = []
        name_area_dict[name].append(array_of_areas[location])
    cont = 'y'
    while cont.lower() == 'y':
        normalized_data = norm_data(name_area_dict, entry_names, num_spec)
        normalized_data_dict = normalized_data[0]
        spec_to_norm_to = normalized_data[1]

        absolute_path = pth.abspath(fname).strip(fname)
        print('saving graphs...')
        for coil in normalized_data_dict:
            normalized_data = []
            for species in normalized_data_dict[coil]:
                normalized_data.append(normalized_data_dict[coil][species])
            x_pos = np.arange(len(normalized_data))
            fig, ax = plt.subplots()
            ax.bar(x_pos, normalized_data, width=.2, alpha=0.8)
            ax.xaxis.set_major_locator(tck.MultipleLocator(10))
            ax.set_xticks(x_pos)
            ax.set_xticklabels(array_of_spec_names)
            max_y_val = find_bounds(max(normalized_data), 'max')
            min_y_val = find_bounds(min(normalized_data), 'min')
            ax.set_yscale('log')
            ax.set_ylim(min_y_val, max_y_val)
            ax.set_xlabel('Species')
            # TODO: add error bars
            ax.set_ylabel('Area normalized to ' + spec_to_norm_to)
            ax.set_title(coil)
            ax.grid()
            file_name = coil + '_normalized_to_' + spec_to_norm_to + '.svg'

            # plt.show()
            plt.savefig(pth.join(absolute_path, file_name), format='svg')
        cont = input('Press \'y\' to normalize to another species, any other key to quit ')


if __name__ == '__main__':
    main()
