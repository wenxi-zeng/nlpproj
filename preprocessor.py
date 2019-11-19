import sys
from xml.dom import minidom
from xml.dom.minidom import Text
import os
from fnmatch import fnmatch


def parse_all_to_file(dir, datafn):
    datafile = open(datafn, "w")
    f = []
    for path, subdirs, files in os.walk(dir):
        for name in files:
            if fnmatch(name, '*.xml'):
                f.append(os.path.join(path, name))

    for fn in f:
        data, label = parse(fn)
        datafile.write(data)


def parse(file):
    data = ''
    label = ''
    xmldoc = minidom.parse(file)
    for p in xmldoc.getElementsByTagName('p'):
        for c in p.childNodes:
            if isinstance(c, Text):
                data = data.strip(' ') + ' ' + handle_text(c)
                label = label.strip(' ') + ' ' + handle_text(c)
            else:
                data = data.strip(' ') + ' ' + handle_tag_i(c.getElementsByTagName('i'))
                label = label.strip(' ') + ' ' + handle_tag_c(c.getElementsByTagName('c'))
        data = data + '\n'
        label = label + '\n'

    return data, label


def handle_text(text):
    return text.nodeValue.strip()


def handle_tag_c(c):
    result = ''
    for x in c:
        if x.childNodes[0].nodeValue is not None:
            result = result.strip() + x.childNodes[0].nodeValue.strip() + ' '

    return result.strip()


def handle_tag_i(i):
    result = ''
    for x in i:
        if x.childNodes[0].nodeValue is not None:
            result = result.strip() + x.childNodes[0].nodeValue.strip() + ' '

    return result.strip()


def main(args):
    if len(args) != 2:
        print("Invalid arguments, try:\n"
              "     preprocessor.py <src dataset directory> <output data file>")
        return

    parse_all_to_file(args[0], args[1])
    pass


if __name__ == '__main__':
    main(sys.argv[1:])
