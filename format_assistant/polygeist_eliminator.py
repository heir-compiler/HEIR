# Author: Zian Zhao
import sys
import getopt
import re

# replace polygeist.SubIndexOp with heir.FHEVectorLoadOp
# TODO: support polygeist dialect in heir and discard this script
def main(argv):
    input_file = ""
    opts, args = getopt.getopt(argv[1:], "hi:", ["help", "input_file=", "output_file="])

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print('poly_eliminator.py -i <input_file>')
            print('or: test_arg.py --input_file=<input_file>')
            sys.exit()
        elif opt in ("-i", "--input_file"):
            input_file = arg
    print('Input File:', input_file)

    f_in = open(input_file, 'r')
    lines = f_in.readlines()
    
    str_poly = "polygeist.subindex"

    for i, line in enumerate(lines):
        if str_poly in line:
            res = line.split("\"" + str_poly)[0]
            out_type = line.split("-> memref")[1]
            in_type = line.split(": (")[1].split(", index")[0]
            memref = line.split("subindex\"(")[1].split(") : ")[0]
            memref, index = memref.split(", ")
            full_type = line.split(" : ")[1]

            str_heir = res + "\"heir.vector_load_init\"(" + memref + ", " + index + ") : (" + \
                in_type + ", index) -> memref" + out_type
            lines[i] = str_heir
    
    f_in.close()
    
    f_out = open(input_file, 'w')

    f_out.writelines(lines)

    f_out.close()

        
if __name__ == "__main__":
    main(sys.argv)
