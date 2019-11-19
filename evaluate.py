import sys

import spellcorrector


def main(args):
    if len(args) == 1 and args[0] == '-help':
        print("-e   evaluate the spell corrector with an existing model\n"
              "-n   create a new model given corpus and n-grams\n")
        return

    if len(args) != 4:
        print("Invalid arguments, try:\n"
              "     evaluate.py -e <model file> <data file> <label file>\n"
              "     evaluate.py -n <corpus> <output model file> <n-gram>\n"
              "     evaluate.py -help")
        return

    if args[0] == '-e':
        model = spellcorrector.load_model(args[1])
        spellcorrector.run(model, args[2], args[3])
    elif args[0] == '-n':
        spellcorrector.create_and_save_model(args[1], args[2], int(args[3]))

    pass


if __name__ == '__main__':
    main(sys.argv[1:])
