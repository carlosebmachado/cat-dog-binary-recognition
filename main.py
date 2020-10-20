from neural_network import CDRModel


# 0 cat
# 1 dog


def new_model():
    model = CDRModel()
    model.create()
    model.compile()
    model.train(16)
    model.save()
    
    print(model.recognize('dataset/manual/cat/cat01.jpg'))
    print(model.recognize('dataset/manual/cat/cat02.png'))
    print(model.recognize('dataset/manual/cat/cat03.jpg'))
    print(model.recognize('dataset/manual/cat/cat04.jpg'))


def load_model():
    model = CDRModel()
    model.load()
    model.compile()
    
    print(model.recognize('dataset/manual/cat/cat01.jpg'))
    print(model.recognize('dataset/manual/cat/cat02.png'))
    print(model.recognize('dataset/manual/cat/cat03.jpg'))
    print(model.recognize('dataset/manual/cat/cat04.jpg'))
    
    print(model.recognize('dataset/manual/dog/dog01.jpg'))
    print(model.recognize('dataset/manual/dog/dog02.jpg'))
    print(model.recognize('dataset/manual/dog/dog03.jpg'))
    print(model.recognize('dataset/manual/dog/dog04.jpg'))


def main():
    #new_model()
    load_model()


main()
