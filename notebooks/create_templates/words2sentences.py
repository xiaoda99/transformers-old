import json


# bigger/smaller -> heavier/lighter
# 原始句 ：A is bigger than B, so A is heavier than B
# A （is/is not）（bigger/smaller）than B, so A（is/is not）（heavier/lighter）than B
# 2*2*2*2=16种
# 每种交换A和B
# 16*3=48句
# flag==1 adj -> adj 性质引发性质 is is
#                    A is bigger than B , so A is heavier than B.
# flag==2 v -> adj 行为引发性质 does is
#                    A eat more food than B , so A is fatter than B
# flag==3 adj -> v 性质引发行为（A与B同一行为的比较） is will
#                    A is faster than B , so A will arrive at the destination earlier than B.
# flag==4 v -> v 行为引发行为 （A与B同一行为的比较）does will
#                    A do more practice than B , so A will get higher marks on the exam than B.
# flag==5 adj -> v 性质引发行为（A对B的行为）后一句没有than
#                    A is poorer than B , so A will borrow money from B.
# flag==6 v -> v 行为引发行为（A对B的行为）后一句没有than
#                    A has a higher density than B , so A will sink in B.


def exchange_subject(cause, effect):
    # a_cause_position = cause.index("A")
    # b_cause_position = cause.index("B")
    a_effect_position = effect.index("A")
    b_effect_position = effect.index("B")

    # exchange_cause = (cause[:a_cause_position] + "B" + cause[a_cause_position + 1:b_cause_position] +
    #                   "A" + cause[b_cause_position + 1:])
    exchange_effect = (effect[:a_effect_position] + "B" + effect[a_effect_position + 1:b_effect_position] +
                       "A" + effect[b_effect_position + 1:])

    original = cause + effect
    # exchanged_cause = exchange_cause + effect
    exchanged_effect = cause + exchange_effect

    # return [original, exchanged_cause, exchanged_effect]
    return [original, exchanged_effect]


def generate_sentences(cause, effect, cause_antonym, effect_antonym, flag):

    output = []

    if flag == "1" or flag == "3" or flag == "5":
        cause_sentence1 = "A is " + cause + " than B , "

        negative_cause_sentence1 = cause_sentence1.replace("is", "isn't")
        antonym_cause_sentence1 = cause_sentence1.replace(cause, cause_antonym)
        negation_antonym_cause_sentence1 = negative_cause_sentence1.replace(cause, cause_antonym)

    if flag == "2" or flag == "4" or flag == "6":
        cause_sentence1 = "A " + cause + " than B , "

        a_position = cause_sentence1.index("A")
        negative_cause_sentence1 = (cause_sentence1[:a_position] + "A doesn't" + cause_sentence1[a_position + 1:])
        antonym_cause_sentence1 = cause_sentence1.replace(cause, cause_antonym)
        negation_antonym_cause_sentence1 = negative_cause_sentence1.replace(cause, cause_antonym)

    if flag == "1" or flag == "2":
        effect_sentence1 = "so A is " + effect + " than B."

        negative_effect_sentence1 = effect_sentence1.replace("is", "isn't")
        antonym_effect_sentence1 = effect_sentence1.replace(effect, effect_antonym)
        negation_antonym_effect_sentence1 = negative_effect_sentence1.replace(effect, effect_antonym)

    if flag == "3" or flag == "4":
        effect_sentence1 = "so A will " + effect + " than B."

        negative_effect_sentence1 = effect_sentence1.replace("will", "won't")
        antonym_effect_sentence1 = effect_sentence1.replace(effect, effect_antonym)
        negation_antonym_effect_sentence1 = negative_effect_sentence1.replace(effect, effect_antonym)

    if flag == "5" or flag == "6":
        effect_sentence1 = "so A will " + effect + " B."

        negative_effect_sentence1 = effect_sentence1.replace("will", "won't")
        antonym_effect_sentence1 = effect_sentence1.replace(effect, effect_antonym)
        negation_antonym_effect_sentence1 = negative_effect_sentence1.replace(effect, effect_antonym)

    cause_list = [cause_sentence1, negative_cause_sentence1, antonym_cause_sentence1, negation_antonym_cause_sentence1]
    effect_list = [effect_sentence1, negative_effect_sentence1, antonym_effect_sentence1,
                   negation_antonym_effect_sentence1]

    for i in cause_list:
        for j in effect_list:
            flag1 = 0
            flag2 = 0
            if i == negative_cause_sentence1 or i == antonym_cause_sentence1:
                flag1 = 1
            if j == negative_effect_sentence1 or j == antonym_effect_sentence1:
                flag2 = 1
            flag = (flag1 + flag2) % 2
            if flag == 1:
                j = j.replace("so", "but")
            output = output + exchange_subject(i, j)

    list = []
    for i in output:
        if i not in list:
            list.append(i)

    return list


if __name__ == '__main__':

    with open("words.json", "r") as f:
        data_json = json.load(f)

    output = {}

    for entry in data_json:
        data = data_json[entry]
        output[entry] = generate_sentences(data["cause"],
                                           data["effect"],
                                           data["cause_antonym"],
                                           data["effect_antonym"],
                                           data["flag"])

    with open('words2sentences.json', 'w') as outfile:

        json.dump(output, outfile, indent=4)
