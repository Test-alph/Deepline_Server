def main():
    print("Hello from mcp-server!")


if __name__ == "__main__":
    main()

import os

if __name__ == "__main__":
    print(os.system('which mcp'))
    print(os.system('mcp --help'))

import subprocess

if __name__ == "__main__":
    subprocess.run(["pip", "install", "mcp[cli]"])
