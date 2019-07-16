import seaborn as sns
import matplotlib.pyplot as plt

cm = [[467,   0,   0, 1,   2, 0,   0,  0,   0,  0],
      [0, 582,  18,  38,  93, 33,  21,   56,   43,   2],
      [0,  4, 110,   0,  0,  0,  0, 25, 10,    6],
      [146,  1,   5, 566, 1066, 220, 33,  1, 43, 0],
      [8,  32,  1, 162, 1822, 24, 15,  0,  8, 0],
      [0,  4,   0,   34, 136, 460,  0,  2,   27, 0],
      [0, 17,  77, 190, 173,  38, 1021,  20, 36, 26],
      [0, 302,   27,    6,  76, 0,  0, 242, 59,   8],
      [14,    3,   1, 42,  7,  7,  0,  0, 740,   0],
      [0, 0,   13, 0,   0,  0, 0,  8, 3,  59]]
labels = ['bass', 'brass', 'flute', 'guitar',
          'keyboard', 'mallet', 'organ', 'reed', 'string', 'vocal']
fig = plt.figure()
ax = fig.add_subplot()
ax = sns.heatmap(cm, annot=True, fmt='g')  # annot=True to annotate cells
ax.set_title('Confusion Matrix')
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.xaxis.set_ticklabels(labels)
ax.yaxis.set_ticklabels(labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
