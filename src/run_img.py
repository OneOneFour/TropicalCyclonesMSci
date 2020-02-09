from CycloneImage import CycloneImage

if __name__ == "__main__":
    path = input("Enter pickle path")
    ci = CycloneImage.load(path)
