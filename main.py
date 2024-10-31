import DataDealer as dd
from MarginPerceptron import MarginPerceptron as MP

dataset = ['dataset/2d-r16-n10000.txt',
           'dataset/4d-r24-n10000.txt',
           'dataset/8d-r12-n10000.txt', ]
res_gamma = []
res_weight = []
res_margin = []
for file in dataset:
    input, label, dim, rad = dd.readData(file)
    # print(input, label, dim, rad)
    MarginPerceptron = MP(dimension=dim, radius=rad, input=input, label=label)
    while MarginPerceptron.train():
        MarginPerceptron.gamma_guess /= 2.0
        # print(MarginPerceptron.gamma_guess)
        MarginPerceptron.epochs = MarginPerceptron.max_iteration(rad, MarginPerceptron.gamma_guess)
        # print(MarginPerceptron.epochs)
        if MarginPerceptron.gamma_guess <= 1e-8:
            print("Gamma guess is too small to calculate")
            break
    print("After training, gamma_guess is ", MarginPerceptron.gamma_guess)
    res_gamma.append(MarginPerceptron.gamma_guess)
    print("After training, weight is", MarginPerceptron.get_weights())
    res_weight.append(MarginPerceptron.get_weights())
    final_margin = MarginPerceptron.calculate_margin()
    print("Final margin after training is:", final_margin)
    res_margin.append(final_margin)
    print('\n\n\n\n')

len_res = len(res_gamma)
for i in range(len(res_gamma)):
    print("The gamma_guess for dimension", 2 ** (i + 1), "is", res_gamma[i])
    print("The weight for dimension", 2 ** (i + 1), "is", res_weight[i])
    print("The margin for dimension", 2 ** (i + 1), "is", res_margin[i])
