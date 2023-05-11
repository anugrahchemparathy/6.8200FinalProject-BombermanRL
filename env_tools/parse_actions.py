def parse_actions(file_name):
    actions = []
    with open(file_name, "r") as file:
        for line in file:
            actions.append(line.split()[2])
    return actions

if __name__ == '__main__':
    file_name = 'file.txt'
    print(parse_actions(file_name))