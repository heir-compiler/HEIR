# Author: Zian Zhao
import sys
import getopt
import re

# Transmit the generated IR code into the HALO demo C++ files
# Since HALO still has bugs in passing evaluation keys
# we cannot directly include a external function to compute the benchmarks
def main(argv):
    input_file = ""
    output_file = ""
    
    opts, args = getopt.getopt(argv[1:], "hi:o:", ["help", "input_file=", "output_file="])

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print('poly_eliminator.py -i <input_file> -o <output_file>')
            print('or: test_arg.py --input_file=<input_file> --output_file=<output_file>')
            sys.exit()
        elif opt in ("-i", "--input_file"):
            input_file = arg
        elif opt in ("-o", "--output_file"):
            output_file = arg
    print('Input File:', input_file)
    print('Output File:', output_file)

    
    f_in = open(input_file, 'r')

    f_out_read = open(output_file, 'r')
    out_lines = f_out_read.readlines()
    insert_line = 0
    for i, line in enumerate(out_lines):
        if  "AutoTimer timer(&evaluation_time);" in line:
            insert_line = i
            break
    f_out_read.close()

    f_out_write = open(output_file, "w")

    in_lines = f_in.readlines()
    for line in in_lines:
        if "{" in line: 
            continue
        if "return" in line:
            break
        if "lut" in line:
            str_index = line.find(')')
            line = line[:str_index] + ', fb_keys' + line[str_index:]
        elif "euclid" in line:
            str_index = line.find(')')
            line = line[:str_index] + ', repack_galois_keys, repack_key, relin_keys' + line[str_index:]
        elif "inner" in line:
            str_index = line.find(')')
            line = line[:str_index] + ', repack_galois_keys, repack_key, relin_keys' + line[str_index:]
        elif "lwe_multiply" in line:
            str_index = line.find(')')
            line = line[:str_index] + ', repack_galois_keys, repack_key, relin_keys' + line[str_index:]
        elif "rlwe_multily" in line:
            str_index = line.find(')')
            line = line[:str_index] + ', repack_galois_keys, repack_key, relin_keys' + line[str_index:]
        elif "encode" in line:
            str_index = line.find(', ')
            line = line[:str_index] + ', scale, lwe_parms' + line[str_index:]
        
        insert_line += 1
        out_lines.insert(insert_line, line)

    f_out_write.writelines(out_lines)

    f_in.close()
    f_out_write.close()

        
if __name__ == "__main__":
    main(sys.argv)
