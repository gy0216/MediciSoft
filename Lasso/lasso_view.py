import matplotlib.pyplot as plt



alpha_set = [0.0001, 0.01, 1]

max_inter_set = [10000000, 100000, 1000]



train_score = []

test_score = []

used_feature = []



for a, m in zip(alpha_set, max_inter_set):

    lasso = Lasso(alpha=a, max_iter=m).fit(x_train, y_train)

    la_tr_score = round(lasso.score(x_train, y_train), 3)

    la_te_score = round(lasso.score(x_test, y_test), 3)

    number_used = np.sum(lasso.coef_ != 0)



    train_score.append(la_tr_score)

    test_score.append(la_te_score)

    used_feature.append(number_used)



index = np.arange(len(alpha_set))

bar_width = 0.35

plt.bar(index, train_score, width=bar_width, label='train')

plt.bar(index+bar_width, test_score, width=bar_width, label='test')

plt.xticks(index+bar_width/2, alpha_set) # bar그래프 dodge를 하기 위해 기준값에 보정치를 더해줍니다.



for i, (ts, te) in enumerate(zip(train_score, test_score)):

    plt.text(i, ts+0.01, str(ts), horizontalalignment='center')

    plt.text(i+bar_width, te+0.01, str(te), horizontalalignment='center')



plt.legend(loc=1)

plt.xlabel('alpha')

plt.ylabel('score')

plt.show()
