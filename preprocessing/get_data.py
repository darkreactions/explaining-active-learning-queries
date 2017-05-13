#!/usr/bin/python

import csv

"""
Takes a filename and returns the given csv as a list of dictionaries where each
dictionary maps from the name of the feature as given by the header row to its
value for that data item.
"""
def get_data_list_of_dicts(filename):
    list = []
    with open(filename) as f:
        f_csv = csv.DictReader(f)
	for row in f_csv:
	    list.append(row)
    return list

"""
Takes a filename and returns a list containing the first row (the header row) as
a list.
"""
def get_headers(filename):
    with open(filename) as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)
    return headers

"""
Takes a list of dictionaries (in the form given by get_data_list_of_dicts above),
and a header list (as given by get_headers) and writes it to the given filename.
"""
def write_data_dicts(filename, headers, rows_list_of_dicts):
    with open(filename,'w') as f:
        f_csv = csv.DictWriter(f, headers, extrasaction='ignore')
        f_csv.writeheader()
        f_csv.writerows(rows_list_of_dicts)

"""
Takes a list of lists, i.e., a list containing each row with each row represented
as a list, a list of the header row, and a filename and writes the data to the
filename as a csv.
"""
def write_data(filename, headers, rows_list_of_lists):
    with open(filename,'w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows(rows_list_of_lists)

"""
Takes a list of dictionaries (in the form given by get_data_list_of_dicts above)
and a column_name (one of the values from get_headers above) and returns the
values in that column as a list.
"""
def get_data_slice(column_name, list_of_dicts):
    list = []
    for dict in list_of_dicts:
        list.append(dict[column_name])
    return list

"""
Returns cols_list_of_dicts
"""
def get_col_list_of_dicts(headers, rows_list_of_dicts, numeric = True):
    cols = map(lambda header: get_data_slice(header, rows_list_of_dicts), headers)
    if numeric:
      cols = map(lambda a_list: map(lambda item: float(item), a_list), cols)
    else:
      cols = map(lambda a_list: map(lambda item: item, a_list), cols)
    cols_list_of_dicts = dict(zip(headers, cols))
    return cols_list_of_dicts
