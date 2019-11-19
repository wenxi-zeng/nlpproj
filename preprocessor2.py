import sys
from xml.dom import minidom
import os
from fnmatch import fnmatch

result = dict()


def parse_all_to_file(dir, labelfn):
    labelfile = open(labelfn, "w")
    f = []
    for path, subdirs, files in os.walk(dir):
        for name in files:
            if fnmatch(name, '*.xml'):
                f.append(os.path.join(path, name))

    for fn in f:
        parse(fn)

    for cor, err_list in result.items():
        if cor.strip():
            labelfile.write(cor + ': ' + ', '.join(err_list) + '\n')


def parse(file):
    xmldoc = minidom.parse(file)
    for ns in xmldoc.getElementsByTagName('NS'):
        if ns.getAttribute('type') == 'S':
            err = handle_tag_i(ns.getElementsByTagName('i'))
            cor = handle_tag_c(ns.getElementsByTagName('c'))
            if cor not in result:
                result[cor] = set()
            result[cor].add(err)


def handle_tag_c(c):
    result = ''
    for x in c:
        if x.childNodes[0].nodeValue is not None:
            result = result.strip() + x.childNodes[0].nodeValue.strip() + ' '

    return result.strip().lower()


def handle_tag_i(i):
    result = ''
    for x in i:
        if x.childNodes[0].nodeValue is not None:
            result = result.strip() + x.childNodes[0].nodeValue.strip() + ' '

    return result.strip().lower()


def main(args):
    if len(args) != 2:
        print("Invalid arguments, try:\n"
              "     preprocessor.py <src dataset directory> <output label file>")
        return

    parse_all_to_file(args[0], args[1])
    pass


if __name__ == '__main__':
    main(sys.argv[1:])
