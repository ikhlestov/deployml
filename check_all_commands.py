import re
from subprocess import call


if __name__ == '__main__':
    commands = []
    with open('README.md', 'r') as f:
        for line in f:
            match = re.findall('`(python .*)`', line)
            if match:
                commands.extend(match)

    passed = []
    failed = []
    for command in commands:
        print("running command `%s`" % command)
        return_code = call(command, shell=True)
        if return_code != 0:
            failed.append(command)
        else:
            passed.append(command)
        print()

    print('-' * 10)
    print("Passed:")
    for command in passed:
        print("\t", command)
    print("Failed:")
    for command in failed:
        print("\t", command)
