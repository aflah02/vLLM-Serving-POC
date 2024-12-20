global x

class xz:
    def __init__(self):
        print("init")
        print(x)

if __name__ == "__main__":
    x = 1
    a = xz()
    print(x)