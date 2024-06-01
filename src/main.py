from methods.deepdta import DeedDTA

if __name__ == "__main__":
    config_file = r"C:\Users\raulc\code\projeto_ifá\configs\deepdta.yaml"
    deepdta = DeedDTA(config_file)
    deepdta.train()
