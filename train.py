from train_model import *
from Read_utils import *

labels, text = get_label_and_data('emails/training/SMSCollection.txt')
print("成功获取标签！")
word_to_id_table = get_dict(text)
print("成功get到字典")
numerized_text = numerize_text(text, word_to_id_table)
print(numerized_text.shape)
print(labels.shape)
freq_tabel = calculate_freq(numerized_text, labels)
print("成功拿到条件概率")

# test
test_path = 'emails/test/test.txt'
with open(test_path) as f:
    labels = []
    for sentence in f.readlines():
        label = classify(sentence, freq_tabel,word_to_id_table)
        labels.append(label)
    print(labels)
