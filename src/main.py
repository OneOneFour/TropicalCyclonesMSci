from CycloneSnapshot import CycloneSnapshot


def glob_pickle_files(directory):
    from glob import glob
    return glob(f"{directory}\*.pickle")


def pickle_file():
    fname = input("Enter file path of cyclone pickle")
    ci = CycloneSnapshot.load(fname)
    ci.draw_eye("I05")


if __name__ == "__main__":
    ci = pickle_file()
