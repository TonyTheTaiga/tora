from tora import Tora

if __name__ == "__main__":
    client = Tora.load_experiment("609abbe8-80e4-4b7d-b628-99e584dcad11")
    print(client.hyperparams)

    for i in range(5):
        client.log("some_metric", step=i, value=i)

    client.shutdown()
