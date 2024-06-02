from methods.deepdta import DeepDTA

if __name__ == "__main__":
    config_file = r"C:\Users\raulc\code\projeto_ifá\configs\deepdta.yaml"
    deepdta = DeepDTA(config_file)
    deepdta.train()
