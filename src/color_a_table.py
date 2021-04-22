import pandas as pd
from pandas.io.formats.style import Styler
import seaborn as sns


def highlight_max(s):
    '''
    highlight the maximum in a Series yellow.
    '''
    is_max = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_max]


if __name__ == '__main__':
    df = pd.DataFrame(
        [[1.0, 0.82, 0.64, 0.28, 0.18, 0.16, 0.37, 0.22, 0.22],
         [0.58, 1.0, 0.5, 0.21, 0.16, 0.11, 0.25, 0.19, 0.14],
         [0.76, 0.84, 1.0, 0.24, 0.17, 0.18, 0.3, 0.22, 0.24],
         [0.17, 0.19, 0.13, 1.0, 0.28, 0.22, 0.53, 0.24, 0.17],
         [0.26, 0.32, 0.21, 0.66, 1.0, 0.32, 0.46, 0.45, 0.28],
         [0.39, 0.39, 0.38, 0.85, 0.55, 1.0, 0.55, 0.48, 0.56],
         [0.19, 0.18, 0.13, 0.43, 0.16, 0.11, 1.0, 0.34, 0.23],
         [0.23, 0.29, 0.2, 0.4, 0.33, 0.2, 0.72, 1.0, 0.35],
         [0.43, 0.4, 0.39, 0.51, 0.37, 0.44, 0.9, 0.64, 1.0]
         ]

        , columns=['c1', "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9"]
        , index=['c1', "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9"]
    )
    print(df)

    cm = sns.light_palette("green", as_cmap=True)
    s = df.style.background_gradient()

    if isinstance(s, Styler):
        formatText = s.render()
        formatText = formatText.replace('0000</td>', '&nbsp;</td>')
        text_file = open("/Users/ducanhnguyen/Documents/mydeepconcolic/format_table/mnist_deepcheck.html", "w")
        text_file.write(formatText)
        text_file.close()
