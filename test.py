def read_string_file(filename):
    with open(filename, 'r') as file:
        file_contents = file.read()
    grids = file_contents.split('\n\n')
    return grids

print(read_string_file('states.txt'))