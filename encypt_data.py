import pyAesCrypt
import getpass
import os
import sys


bufferSize = 64 * 1024

def encrypt_file(input_file, output_file, password):
    try:
        pyAesCrypt.encryptFile(input_file, output_file, password, bufferSize)
        print(f"File '{input_file}' encrypted successfully to '{output_file}'.")
    except Exception as e:
        print("Encryption failed:", str(e))

def main():

    files_to_encrypt = [
        "final_dataset_raw_questions.csv",
        "eval_data_release.zip",
        "eval_results_o1-2024-12-17_mentat_data_base_2025_01_26_2031.pkl",
    ]
    password = getpass.getpass("Enter encryption password: ")

    for input_file in files_to_encrypt:
        if not os.path.isfile(input_file):
            print("The input file does not exist.")
            sys.exit(1)
        output_file = input_file + ".aes"
        encrypt_file(input_file, output_file, password)

if __name__ == "__main__":
    main()