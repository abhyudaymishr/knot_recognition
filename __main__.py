from .model import get_resnet

def main():
    print("Package entry point loaded.")
    model = get_resnet()
    

if __name__ == "__main__":
    main()
