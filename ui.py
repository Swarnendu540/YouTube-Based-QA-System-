import streamlit as st
import requests

# Set page title and layout
st.set_page_config(page_title="তিনটি ভিডিও থেকে প্রশ্ন উত্তর", layout="centered")

st.title("তিনটি ভিডিও থেকে প্রশ্ন উত্তর দেওয়ার সিস্টেম")

mode = st.radio(
    "মোড নির্বাচন করুন:",
    ("একটি ভিডিওতে প্রশ্ন করুন (টাইমস্ট্যাম্প সহ)", "তিনটি ভিডিও থেকে একসাথে প্রশ্ন করুন")
)

if mode == "একটি ভিডিওতে প্রশ্ন করুন (টাইমস্ট্যাম্প সহ)":
    video_url = st.text_input("ভিডিওর URL:")
    question = st.text_area("আপনার প্রশ্ন লিখুন:")

    if st.button("উত্তর চান"):
        if not video_url or not question:
            st.warning("ভিডিও URL এবং প্রশ্ন লিখুন।")
        else:
            payload = {
                "video_url": video_url,
                "question": question
            }
            try:
                with st.spinner("ভিডিও থেকে উত্তর খুঁজছি..."):
                    response = requests.post("http://localhost:5000/ask_individual", json=payload)
                    data = response.json()

                    if response.status_code == 200:
                        st.success("উত্তর পাওয়া গেছে!")
                        st.write("### উত্তর:")
                        st.write(data["response"])
                    else:
                        st.error("ত্রুটি: " + data.get("error", "অজানা ত্রুটি"))
            except Exception as e:
                st.error(f"সার্ভারে সংযোগ হচ্ছে না। ফ্লাস্ক চালু আছে কিনা দেখুন। {e}")

else:  # Multi-video combined mode
    video_url_1 = st.text_input("ভিডিও ১ এর URL:")
    video_url_2 = st.text_input("ভিডিও ২ এর URL:")
    video_url_3 = st.text_input("ভিডিও ৩ এর URL:")
    question = st.text_area("আপনার প্রশ্ন লিখুন:")

    if st.button("উত্তর চান"):
        if not all([video_url_1, video_url_2, video_url_3, question]):
            st.warning("অনুগ্রহ করে সবগুলো ভিডিও URL এবং প্রশ্ন লিখুন।")
        else:
            payload = {
                "video_urls": [video_url_1, video_url_2, video_url_3],
                "question": question
            }
            try:
                with st.spinner("ভিডিওগুলো থেকে উত্তর খুঁজছি..."):
                    response = requests.post("http://localhost:5000/ask_multiple", json=payload)
                    data = response.json()

                    if response.status_code == 200:
                        st.success("উত্তর পাওয়া গেছে!")
                        st.write("### উত্তর:")
                        st.write(data["response"])
                    else:
                        st.error("ত্রুটি: " + data.get("error", "অজানা ত্রুটি"))
            except Exception as e:
                st.error(f"সার্ভারে সংযোগ হচ্ছে না। ফ্লাস্ক চালু আছে কিনা দেখুন। {e}")
