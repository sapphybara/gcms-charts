import os.path as pth

import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy as np
import json

from typing import List, TextIO, Dict, Union

# change this to the number of samples you have from each coil
NUM_SAMPLES = 3

AREA = 'area'
STDEV = 'stdev'


def main():
    file_name = check_file_name()
    spec_with_name = run_save_concise_data(file_name)
    num_spec = spec_with_name[0]
    file_name_processed = spec_with_name[1]
    compressed_dict = group_data(file_name_processed, num_spec)
    final_fname = write_average(file_name_processed, compressed_dict, num_spec)
    create_plots(final_fname)


def create_plots(fname: str) -> None:
    with open(fname, 'r') as file:
        file_lines = json.load(file)
    cont = 'y'
    while cont.lower() == 'y':
        spec_to_norm_to = get_spec_to_norm_to(file_lines)
        normalized_data_dict = norm_data(file_lines, spec_to_norm_to)
        absolute_path = pth.abspath(fname).strip(fname)
        print('saving graphs...')
        for coil in normalized_data_dict:
            areas = []
            st_devs = []
            single_set = normalized_data_dict[coil]
            array_of_spec_names = []
            for species in single_set:
                single_species = single_set[species]
                areas.append(single_species[AREA])
                st_devs.append(single_species[STDEV])
                array_of_spec_names.append(species)
            x_pos = np.arange(len(areas))
            fig, ax = plt.subplots()
            ax.bar(x_pos, areas, yerr=st_devs, width=.2, alpha=0.8, capsize=4)
            ax.xaxis.set_major_locator(tck.MultipleLocator(10))
            ax.set_xticks(x_pos)
            ax.set_xticklabels(array_of_spec_names)
            max_y_val = find_bounds(max(areas), 'max')
            min_y_val = find_bounds(min(areas), 'min')
            ax.set_yscale('log')
            ax.set_ylim(min_y_val, max_y_val)
            ax.set_xlabel('Species')
            ax.set_ylabel('Area normalized to ' + spec_to_norm_to)
            ax.set_title(coil)
            ax.grid()
            file_name = coil + '_normalized_to_' + spec_to_norm_to + '.svg'

            # plt.show()
            plt.savefig(pth.join(absolute_path, file_name), format='svg')
        cont = input('Press \'y\' to normalize to another species, any other key to quit ')


def get_spec_to_norm_to(data: Dict[str, Dict[str, float]]) -> str:
    first_coil = list(data)[0]
    spec_to_norm_to = input(
        'Enter the name of the species that you want to normalize to ')
    # find out if the spec name ('Ar' for example) is in the list of spec names
    while spec_to_norm_to not in data[first_coil]:
        spec_to_norm_to = input(spec_to_norm_to + ' not found, try again.\n(be sure to write the '
                                                  'case-sensitive name) ')
    return spec_to_norm_to


def empty_file(fname: str) -> None:
    """
    deletes contents of a file so that we don't get redundant data
    :param fname: the name of the file
    :return: None
    """
    with open(fname, 'w'):
        pass


def check_file_name() -> str:
    """
    asks user for a file name, ensures it exists
    :return: the file name
    """
    while True:
        # emulate do-while
        file_name = get_file_name()
        if not pth.exists(file_name):
            print('File not found, try again (remember not to include the extension)')
        else:
            break
    return file_name


def get_file_name() -> str:
    """
    :return: user's file name
    """
    return input("Enter the file name, no extension ") + '.txt'


def get_info(data: str, str_to_find: str, num_sets: int) -> int or ValueError:
    """
    reads the file and returns which line holds a certain string, along with the second
    element in that line
    :param data: the file we're using
    :param str_to_find: the string the user is looking for
    :param num_sets: the number of times we want to skip before returning the string (ie if I
    want to find the 10th ocurrence of 'Data File Name' I would set num_sets = 10)
    :return: the number of species found and the line number the species start at
    :raises ValueError: if the string to find is not in the file
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
    raise ValueError('Cannot find ' + str_to_find + ' in your file')


def remove_trash_lines(data: str, num_lines: int) -> List[str]:
    """
    gets rid of all the lines that aren't needed
    :param data: the file we are reading
    :param num_lines: the number of lines we don't need
    :return: the first line we are interested in
    """
    with open(data, 'r') as file:
        last_line = file.readlines()[num_lines:num_lines + 1]
    return last_line


def get_area_col(area_row: str) -> int:
    """
    finds the column that 'Area' is in
    :param area_row: the row that 'Area' is in
    :return: the column number where 'Area' is
    """
    area_row = delete_bad_chars(area_row)
    for counter, column in enumerate(area_row, start=1):
        if column.lower() == AREA:
            return counter


def delete_bad_chars(string: str) -> List[str]:
    """
    removes unnecessary characters from a string and splits it at a tab
    :param string: the string we want to do this to
    :return: the properly formatted string
    """
    string = string.strip("[]'")
    return string.split("\\t")


def save_concise_data(
        new_file_name: str,
        area_col: int,
        num_trash_lines: int,
        num_spec: int,
        name_of_dataset: str,
        counter: int,
        data: str
) -> None:
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


def write_to_file(
        data: str,
        counter: int,
        num_trash_lines: int,
        num_spec: int,
        file: TextIO,
        area_col: int
) -> None:
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


def get_file_len(data: str) -> int:
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


def run_save_concise_data(data: str) -> List[Union[int, str]]:
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


def group_data(data: str, num_spec: int) -> Dict[str, list]:
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


def write_average(fname: str, names_dict: Dict[str, List[int]], num_spec: int) -> str:
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
    errors = {}  # {Coil1: {Ar: [area: 1, stdev: .4], ...}, ...}
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
            all_aliquots[name][STDEV] = np.std(all_areas[name])
        errors[keys] = {}
        errors[keys] = all_aliquots

    with open(final_file, 'a') as json_file:
        json.dump(errors, json_file, indent=2)
    return final_file


def norm_data(
        coil_data: Dict[str, Dict[str, Dict[str, float]]],
        spec_to_norm_to: str
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Normalizes to the user's chosen species
    :param dict coil_data: A dictionary object with key the species-name, the value a dictionary
    of species and their errors/areas
    :param str spec_to_norm_to: the species that we want to normalize all areas to
    :return dict: dictionary with the key being the coil names, the value being a dictionary of
    the form {species: {normalized area: 1, normalized standard deviation: .1}, ...}
    """
    normalized_data = {}
    for coil in coil_data:
        normalized_data[coil] = {}
        indiv_dataset = coil_data[coil]
        normalize = indiv_dataset[spec_to_norm_to]
        area_to_norm_to = normalize[AREA]
        stdev_to_norm_to = normalize[STDEV]
        for spec_name in indiv_dataset:
            single_spec = indiv_dataset[spec_name]
            normalized_data[coil][spec_name] = {}
            normalized_data[coil][spec_name][AREA] = single_spec[AREA] / area_to_norm_to
            normalized_data[coil][spec_name][STDEV] = single_spec[STDEV] / stdev_to_norm_to
    return normalized_data


def find_bounds(upper_or_lower: float, max_or_min: str) -> int or float:
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


if __name__ == '__main__':
    main()
