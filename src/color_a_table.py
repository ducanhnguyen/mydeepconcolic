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
        [[1.0,0.0,0.33,0.0,0.0,0.0,0.0,0.0,0.0],
         [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
         [1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0],
         [0.0,0.0,0.0,1.0,0.33,0.27,0.0,0.0,0.02],
         [0.0,0.0,0.0,0.67,1.0,0.18,0.0,0.0,0.0],
         [0.0,0.0,0.0,1.0,0.33,1.0,0.0,0.0,0.12],
         [0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.1,0.07],
         [0.0,0.0,0.0,0.0,0.0,0.0,0.25,1.0,0.07],
         [0.0,0.0,0.0,0.33,0.0,0.45,0.75,0.3,1.0],
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
        text_file = open("/Users/ducanhnguyen/Documents/mydeepconcolic/mnist_deepcheck.html", "w")
        text_file.write(formatText)
        text_file.close()
