import streamlit as st
from streamlit_sortables import sort_items
from st_draggable_list import DraggableList

import google.generativeai as genai
import os

import random

# ja_sentence_segmenter など必要なライブラリのインポート
import functools
from ja_sentence_segmenter.common.pipeline import make_pipeline
from ja_sentence_segmenter.concatenate.simple_concatenator import concatenate_matching
from ja_sentence_segmenter.normalize.neologd_normalizer import normalize
from ja_sentence_segmenter.split.simple_splitter import split_newline, split_punctuation


def main():

  
  st.markdown("""
    <style>
      .title_format {
        font-size:25px !important;
        background-color: #cc99ff;
        border-radius: 10px;
        text-align: center;
        box-shadow:
          inset 2px 2px 3px rgba(255, 255, 255, 0.6),
          inset -2px -2px 3px rgba(0, 0, 0, 0.6);
      }
    </style>
  """, unsafe_allow_html=True)
  st.markdown('<p class="title_format">お料理AI(アイ)ちゃんといっしょにお料理をつくりましょう！</p>', unsafe_allow_html=True)

  #st.title("お料理AI(アイ)ちゃん")
  st.markdown("")
  st.markdown("下の長方形にはお料理AI(アイ)ちゃんが考えたカレーの作り方が書いてあります")
  st.markdown("でも、順序がバラバラです")
  st.markdown("長方形は指で軽く抑えると上下に動かすことができます")
  st.markdown("パソコンをお使いの方はマウスでも動きます")
  st.markdown("長方形を動かして正しい順に並べてください")
  st.markdown("正しい順に物事を行うことができる実行機能という認知機能が強化できます")
  st.markdown("")
  st.markdown("なお、画面を新しくするたびに、お料理AI(アイ)ちゃんが新しい作り方を考えて表示します")
  st.markdown("ですから何度でも試すことができますよ")
  st.markdown("お料理AI(アイ)ちゃんのレシピの腕もお楽しみください")
  st.markdown("")
  st.markdown("")

  # API_KEYを環境変数に設定
  APIKEY=os.getenv('Gemini_API_KEY')
  genai.configure(api_key=APIKEY)

  safety_setting=[
      {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE"
      },
      {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE"
      },
      {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE"
      },
      {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE"
      }
  ]

  generation_config = {
      "temperature": 1,
      "top_p": 1,
      "top_k": 1,
      "max_output_tokens": 5000,
      "response_mime_type": "text/plain",  
  }

  model = genai.GenerativeModel(
      model_name="gemini-1.5-flash",
      generation_config=generation_config,
      safety_settings = safety_setting,
      # See https://ai.google.dev/gemini-api/docs/safety-settings
  )

  chat_session = model.start_chat(
      history=[
      ]
  )


  prompt_1="カレーの作り方を教えて下さい。"

  prompt_2="箇条書きで。ただし番号も中黒も振らないで。"

  prompt_3="文の最後には必ず「。」をつけて。"

  prompt_4="箇条書きは行を変えないで。"

  
  prompt=prompt_1+prompt_2+prompt_3+prompt_4

  response = chat_session.send_message(prompt)

  if 'md' not in st.session_state:
    st.session_state.md=response.text


  # segmenter の定義
  split_punc2 = functools.partial(split_punctuation, punctuations=r".。!?")
  concat_tail_no = functools.partial(concatenate_matching, former_matching_rule=r"^(?P<result>.+)(の)$", remove_former_matched=False)
  concat_tail_te = functools.partial(concatenate_matching, former_matching_rule=r"^(?P<result>.+)(て)$", remove_former_matched=False)
  concat_decimal = functools.partial(concatenate_matching, former_matching_rule=r"^(?P<result>.+)(\d.)$", latter_matching_rule=r"^(\d)(?P<result>.+)$", remove_former_matched=False, remove_latter_matched=False)
  segmenter = make_pipeline(normalize, split_newline, concat_tail_no, concat_tail_te, split_punc2, concat_decimal)

  # 句点区切りしたい文章
  content = st.session_state.md

  # 句点区切りを実行、parseのデータ型はリスト
  parse = list(segmenter(content))
  #st.write(parse)

  # parseの長さ分のランダムな数の生成
  index=list(range(0, len(parse)))
  

  if 'new_index' not in st.session_state:
    st.session_state.new_index=random.sample(index, len(index))
    st.write(st.session_state.new_index)
    st.write(st.session_state.new_index[0])


  # 辞書型のリストの初期化
  length=len(parse)
  data = [{"id":"", "order": 0, "name":""} for i in range(length)]

  # dataの'name'にGeminiの回答を与える
  for i in range(length):
    data[i]['name']=parse[st.session_state.new_index[i]]


  slist = DraggableList(data, width="80%")


  st.markdown("""
    <style>
      .write_form {
        font-size:20px !important;
        # background-color: blue;
        text-align: right;
      }
    </style>
    """, unsafe_allow_html=True)
  st.markdown('<p class="write_form">一般社団法人 コラップ</p>', unsafe_allow_html=True)

if __name__=="__main__":
  main()  

