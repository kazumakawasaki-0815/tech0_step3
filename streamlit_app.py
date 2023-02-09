import pickle
import numpy as np
import streamlit as st
import pandas as pd
import statistics
import pickle
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression

st.cache(allow_output_mutation=True)

#xの標準化情報を読み込む（事前に別プログラムで作っているぞ）
x_statics_data = np.load("statics.npz")["X"]
x1_statics=x_statics_data[0] #[48.101753 10.071228] 平均、標準偏差
x2_statics=x_statics_data[1] # [17.20103  9.054  ] 平均、標準偏差
print(x1_statics,x2_statics)

#yの標準化情報を読み込む（事前に別プログラムで作っているぞ）
y_statics=np.load("statics.npz")["Y"][0] #平均、標準偏差
print(y_statics)

# 重回帰モデルのオープン
# with open('model.pickle', mode='rb') as f:
#     model_lr_std = pickle.load(f)

f=open('model.pickle', mode='rb')
model_lr_std = pickle.load(f)

#config
st.set_page_config(page_title="愛知県一宮市の家賃を予想してくれるアプリ", page_icon="logo-150x150.png", layout="wide", initial_sidebar_state="auto", menu_items=None)
# st.image("tech0-logo.png", width=200)
st.title("愛知県一宮市の家賃を予想してくれるアプリ")
# st.sidebar.image("tech0-logo.png", width=200)

# st.sidebar.markdown("[![Foo](https://tech--0.com/wp-content/uploads/2022/08/tech0-logo.png)](https://tech0-jp.com/)") 

#サイドバー
st.sidebar.title('条件入力')
x1=st.sidebar.number_input('専有面積：',30,100,50)
x2=st.sidebar.number_input('築年数：',1,100,50)


if st.button("れっつらごー"):

    #予測
    z=[[x1,x2]]
    #標準化（呼び出した標準化モデルを利用）
    z_nomalize=[[(z[0][0]-x1_statics[0])/x1_statics[1],(z[0][1]-x2_statics[0])/x2_statics[1]]]#それぞれ掛け算される
    #標準化された状態で予測
    predict_nomalize=model_lr_std.predict(z_nomalize)
    #標準化をさらに戻す
    result=predict_nomalize[0][0]*y_statics[1]+y_statics[0]

    st.write("計算結果＝だいたい"+str(result)+"円")
