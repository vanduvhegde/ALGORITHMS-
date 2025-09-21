# Required Libraries
import customtkinter as ctk
import re
import emoji
import nltk
import speech_recognition as sr
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

stop_words = set(stopwords.words('english'))

# -------------------- PREPROCESSING --------------------
def preprocess(text):
    text = emoji.demojize(text)
    text = text.lower()
    text = re.sub(r"http\S+|@\w+|#\w+|[^a-z\s\?]", "", text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words or w == '?']
    tokens = [w for w in tokens if w.isalpha() or w == '?']
    return " ".join(tokens)

# -------------------- DATA LOADING --------------------
def load_amazon_data():
    return [
        # Positive Examples
        ("I love this product! Works perfectly and arrived fast. 😊", "pos", "Happy", "Compliment",
         "Wireless Bluetooth Earbuds"),
        ("Amazing quality and excellent customer service. 👍", "pos", "Happy", "Compliment", "Portable Power Bank"),
        ("Very satisfied with my purchase, highly recommend. 😍", "pos", "Happy", "Compliment",
         "Smart Fitness Tracker watch"),
        ("This is the best thing I have bought this year. 🏆", "pos", "Happy", "Compliment",
         "Noice cancelling Headphones"),
        ("Great value for money, will buy again! 💰", "pos", "Happy", "Compliment", "Reusable Water Bottle"),
        ("Absolutely perfect, five stars! ⭐⭐⭐⭐⭐", "pos", "Happy", "Compliment", "Cordless Handheld Vacuum Cleaner"),
        ("Good product, works as described. 👌", "pos", "Happy", "Compliment", "Silicon Phone case"),
        ("Really happy with the quality. 😊", "pos", "Happy", "Compliment", "Organic Skincare Set"),
        ("I would buy this again without hesitation. 🤩", "pos", "Happy", "Compliment", "Ergonomic Office Chair"),
        ("Fast delivery and great packaging. 📦", "pos", "Happy", "Compliment", "Fast Shipping From Amazon Basics"),
        ("Excellent build quality and performance. 🛠️", "pos", "Happy", "Compliment", "Mechanical Gaming Keyboard"),
        ("Highly recommend to everyone! 🔥", "pos", "Happy", "Compliment", "Smart Home security Camera"),
        ("This phone case feels premium and fits perfectly 😊", "pos", "Happy", "Compliment", "Premium Phone Case"),
        ("Exceeded my expectations, the camera quality is amazing 📸", "pos", "Happy", "Compliment",
         "High Resolution Digital Camera"),
        ("Charging is super fast, love the battery life 🔋", "pos", "Happy", "Compliment", "Fast Charging USB-C Cable"),
        ("Comfortable to wear, looks stylish and well made 👌", "pos", "Happy", "Compliment", "Stylish Smartwatch Band"),
        ("Fantastic sound quality in the headphones, clear highs and rich bass 🎶", "pos", "Happy", "Compliment",
         "Wireless Over-Hear Headphones"),
        ("Perfect arrival, boxed well, amazing customer service 😊", "pos", "Happy", "Compliment",
         "Amazon Prime Delivery service"),
        ("Outstanding product with top-notch features! 💯", "pos", "Happy", "Compliment", "4K Ultra HD Smart TV"),
        ("This blender crushes everything perfectly! 🥤", "pos", "Happy", "Compliment", "High Power Blender"),
        # Negative Examples
        ("This product is terrible and broke after one use. 😠", "neg", "Angry", "Complaint", "Cheap Phone Charger"),
        ("Worst purchase ever, very disappointed. 😡", "neg", "Angry", "Complaint", "Budget Bluetooth Speaker"),
        ("Poor quality and awful customer support. 👎", "neg", "Angry", "Complaint", "Low Quality Headphones"),
        ("I hate it, not worth the money. 😤", "neg", "Angry", "Complaint", "Inexpensive Fitness Tracker"),
        ("Arrived broken and didn't work at all. 😞", "neg", "Angry", "Complaint",
         "Defective Smartphone Screen Protector"),
        ("Very bad experience, will not buy again. 😒", "neg", "Angry", "Complaint", "Flimsy Yoga Mat"),
        ("The product stopped working after a week. 😡", "neg", "Angry", "Complaint", "Faulty wireless Mouse"),
        ("Extremely disappointed with this item. 😔", "neg", "Sad", "Complaint", "Cheap Tablet Device"),
        ("Not as described, poor quality. 😕", "neg", "Sad", "Complaint", "Poor Quality Earbuds"),
        ("The instructions were confusing and unclear. 🤨", "neg", "Sad", "Complaint",
         "Unclear User Manual for Smartwatch"),
        ("Battery life is very poor. 🔋❌", "neg", "Angry", "Complaint", "Low Batter Life Smartphone"),
        ("I want a refund immediately! 💸😠", "neg", "Angry", "Complaint", "Defective Laptop Charger"),
        ("Package was damaged and the item inside was broken 😢", "neg", "Sad", "Complaint",
         "Damaged Package With Broken Toy"),
        ("Very disappointed, the screen flickers and the seller did not respond 😡", "neg", "Angry", "Complaint",
         "Faulty Laptop screen"),
        ("Color looks totally different than in the pictures 😕", "neg", "Sad", "Complaint", "Wrong Color Clothing Item"),
        ("Cheap material, stitching came undone after first wash 👎", "neg", "Angry", "Complaint", "Cheap Fabric Hoodie"),
        ("Product stopped working after 2 weeks, want a refund 😤", "neg", "Angry", "Complaint",
         "Broken Kitchen Appliance"),
        ("Absolute waste of money, product is broken and support doesn’t respond 😡", "neg", "Angry", "Complaint",
         "Non-Responsive Customer Support For Phone"),
        ("Poor packaging caused damage during shipping 😞", "neg", "Sad", "Complaint", "Poorly Packaged Electronics"),
        ("Terrible smell coming from the product 😷", "neg", "Angry", "Complaint", "Defective Air Purifier"),
        # Neutral Examples
        ("Product arrived on time. 📅", "neu", "Neutral", "Statement", "Standard Laptop Sleeve"),
        ("Packaging was okay, nothing special. 📦", "neu", "Neutral", "Statement", "Basic Shipping Box"),
        ("I am still testing the product. ⏳", "neu", "Neutral", "Statement", "New Smartwatch"),
        ("The color is as described. 🎨", "neu", "Neutral", "Statement", "Colored Phone Case"),
        ("Can someone help me with installation? 🤔", "neu", "Neutral", "Request", "User Manual Inquiry For TV"),
        ("Is there a warranty on this item? 🛡️", "neu", "Neutral", "Question", "Warranty Question For Blender"),
        ("It works, but nothing extraordinary. 😐", "neu", "Neutral", "Statement", "General statement About a Headphone"),
        ("Received product in good condition. ✅", "neu", "Neutral", "Statement", "Ordered Delivery Confirmation"),
        ("What is the return policy? 🔄", "neu", "Neutral", "Question", "Return Quality Question For smartwatch"),
        ("I need assistance with my order. 🆘", "neu", "Neutral", "Request", "Request For Customer Service Help"),
        ("Does this come in different colors? 🌈", "neu", "Neutral", "Question", "Color Options question For Sneakers"),
        ("Could you suggest some compatible accessories? 🧩", "neu", "Neutral", "Request",
         "Accessory Recommendation Request for Phone"),
        ("Does this case come in other colors?", "neu", "Neutral", "Question", "Phone Case Color Question"),
        ("Could you tell me the actual dimensions of the product?", "neu", "Neutral", "Request",
         "Product Dimensions Request for Desk Lamp"),
        ("It works well so far, but need to test durability over time", "neu", "Neutral", "Statement",
         "Durability Statement for Backpack"),
    ]


def load_youtube_data():
    return [
        # Positive Examples
        ("This video was amazing, really enjoyed it! 😍", "pos", "Happy", "Compliment"),
        ("Great content as always, thanks for sharing. 🙌", "pos", "Happy", "Compliment"),
        ("I love this channel, keep up the good work! 💪", "pos", "Happy", "Compliment"),
        ("Best tutorial I've seen on this topic. 🏅", "pos", "Happy", "Compliment"),
        ("Super helpful video, thank you! 🙏", "pos", "Happy", "Compliment"),
        ("This was really entertaining and informative. 🎉", "pos", "Happy", "Compliment"),
        ("Excellent explanation and nice visuals. 👌", "pos", "Happy", "Compliment"),
        ("Thanks for the valuable info! 💡", "pos", "Happy", "Compliment"),
        ("Keep making videos like this! 👍", "pos", "Happy", "Compliment"),
        ("The presenter explained everything clearly. 🧑‍🏫", "pos", "Happy", "Compliment"),
        ("Awesome editing and great pacing. 🎬", "pos", "Happy", "Compliment"),
        ("This helped me a lot, much appreciated! 😊", "pos", "Happy", "Compliment"),
        ("Loved the breakdown in this tutorial, so informative 👍", "pos", "Happy", "Compliment"),
        ("You make these complex topics seem easy – thanks 🙌", "pos", "Happy", "Compliment"),
        ("The editing style is clean and transitions smooth 🎬", "pos", "Happy", "Compliment"),
        ("Really helpful tips, though I wish you would slow down a bit.", "pos", "Happy", "Compliment"),
        ("Thanks for covering this topic in detail.", "pos", "Happy", "Compliment"),
        ("Great job with the animations!", "pos", "Happy", "Compliment"),

        # Negative Examples
        ("This video is awful and a waste of time. 😡", "neg", "Angry", "Complaint"),
        ("Terrible editing and bad audio quality. 🔇", "neg", "Angry", "Complaint"),
        ("I hate how misleading the title is. 😠", "neg", "Angry", "Complaint"),
        ("Poor explanation, very confusing. 🤯", "neg", "Angry", "Complaint"),
        ("Disappointed with the content, expected better. 😞", "neg", "Sad", "Complaint"),
        ("This channel is getting worse. 👎", "neg", "Angry", "Complaint"),
        ("The video quality is really bad. 📉", "neg", "Angry", "Complaint"),
        ("Not happy with this video. 😒", "neg", "Sad", "Complaint"),
        ("The presenter seems unprepared. 😕", "neg", "Angry", "Complaint"),
        ("Audio is so bad I couldn't hear anything. 🔇", "neg", "Angry", "Complaint"),
        ("The video kept buffering constantly. ⏳", "neg", "Angry", "Complaint"),
        ("I want a refund for the paid content! 💸😠", "neg", "Angry", "Complaint"),
        ("Audio keeps cutting, hard to follow 😤", "neg", "Angry", "Complaint"),
        ("Too many ads in this video, annoying 😒", "neg", "Sad", "Complaint"),
        ("Title is misleading, content doesn't match 😠", "neg", "Angry", "Complaint"),
        ("The background music was too loud and distracting.", "neg", "Angry", "Complaint"),
        ("Please fix the mic quality in your next video.", "neg", "Sad", "Complaint"),
        ("The video resolution is low on my device.", "neg", "Sad", "Complaint"),

        # Neutral Examples
        ("The video was uploaded yesterday. 📅", "neu", "Neutral", "Statement"),
        ("Can someone explain the topic better? 🤔", "neu", "Neutral", "Question"),
        ("I don't understand this part. 😕", "neu", "Neutral", "Request"),
        ("What software did you use for editing? 💻", "neu", "Neutral", "Question"),
        ("Thanks for the update. 🙏", "neu", "Neutral", "Compliment"),
        ("Looking forward to the next video. ⏭️", "neu", "Neutral", "Statement"),
        ("The length of the video is perfect. ⏱️", "neu", "Neutral", "Statement"),
        ("Subtitle options would be nice. 🔤", "neu", "Neutral", "Request"),
        ("Is there a transcript available? 📄", "neu", "Neutral", "Question"),
        ("Please add captions in other languages. 🌍", "neu", "Neutral", "Request"),
        ("Could you make a video about Python basics? 🐍", "neu", "Neutral", "Request"),
        ("Where can I find the source code? 💻", "neu", "Neutral", "Question"),
        ("Can you do a follow-up video on this?", "neu", "Neutral", "Request"),
        ("I’m neutral on this video, didn’t feel strongly either way.", "neu", "Neutral", "Statement")
    ]


def load_data(platform):
    if platform == "amazon":
        data = load_amazon_data()
    elif platform == "youtube":
        data = load_youtube_data()
    else:
        data = []
    texts = [preprocess(d[0]) for d in data]
    y_sentiment = [d[1] for d in data]
    y_emotion = [d[2] for d in data]
    y_intent = [d[3] for d in data]
    y_product = [d[4] if len(d) > 4 else None for d in data]  # Only amazon has product
    return texts, y_sentiment, y_emotion, y_intent, data, y_product

# -------------------- MODEL TRAINING --------------------
def train_models(X, y_sentiment, y_emotion, y_intent, y_product):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X_vect = vectorizer.fit_transform(X)

    def train_single_model(y):
        model = LogisticRegression(max_iter=1000)
        model.fit(X_vect, y)
        return model

    model_sentiment = train_single_model(y_sentiment)
    model_emotion = train_single_model(y_emotion)
    model_intent = train_single_model(y_intent)
    model_product = None
    if any(y_product):
        # Train product model only if data exists (amazon)
        model_product = train_single_model(y_product)

    return (model_sentiment, model_emotion, model_intent, model_product), vectorizer, X_vect

# -------------------- ANALYSIS --------------------
def analyze(text, models, vectorizer):
    text_preprocessed = preprocess(text)
    X_vect = vectorizer.transform([text_preprocessed])

    model_sentiment, model_emotion, model_intent, model_product = models

    sentiment = model_sentiment.predict(X_vect)[0]
    sentiment_prob = max(model_sentiment.predict_proba(X_vect)[0])

    emotion = model_emotion.predict(X_vect)[0]
    emotion_prob = max(model_emotion.predict_proba(X_vect)[0])

    intent = model_intent.predict(X_vect)[0]
    intent_prob = max(model_intent.predict_proba(X_vect)[0])

    product = None
    if model_product is not None:
        product = model_product.predict(X_vect)[0]

    return {
        "sentiment": sentiment,
        "accuracy_sentiment": sentiment_prob,
        "emotion": emotion,
        "accuracy_emotion": emotion_prob,
        "intent": intent,
        "accuracy_intent": intent_prob,
        "product": product,
    }

# -------------------- VOICE INPUT --------------------
def listen_to_voice():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source, phrase_time_limit=5)
    try:
        text = recognizer.recognize_google(audio)
        print(f"Recognized: {text}")
        return text
    except Exception as e:
        print("Voice recognition failed:", e)
        return None

# -------------------- GUI --------------------
class SentimentApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Sentiment Analyzer")
        self.geometry("500x400")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.models = None
        self.vectorizer = None
        self.platform = None
        self.init_welcome_page()

    def init_welcome_page(self):
        self.clear_frame()
        label = ctk.CTkLabel(self, text="Social Media Sentiment Analysis!", font=ctk.CTkFont(size=20, weight="bold"))
        label.pack(pady=30)

        enter_button = ctk.CTkButton(self, text="Enter", command=self.init_platform_selection_page)
        enter_button.pack(pady=10)

    def init_platform_selection_page(self):
        self.clear_frame()
        ctk.CTkLabel(self, text="Choose a Platform", font=ctk.CTkFont(size=18)).pack(pady=20)

        ctk.CTkButton(self, text="Amazon", width=200, command=lambda: self.load_platform("amazon")).pack(pady=10)
        ctk.CTkButton(self, text="YouTube", width=200, command=lambda: self.load_platform("youtube")).pack(pady=10)
        ctk.CTkButton(self, text="Back", command=self.init_welcome_page).pack(pady=10)

    def load_platform(self, platform_choice):
        self.platform = platform_choice
        X, y_sentiment, y_emotion, y_intent, raw_texts, y_product = load_data(self.platform)
        self.models, self.vectorizer, _ = train_models(X, y_sentiment, y_emotion, y_intent, y_product)
        self.init_main_page()

    def init_main_page(self):
        self.clear_frame()
        ctk.CTkLabel(self, text=f"Platform: {self.platform.capitalize()}").pack(pady=10)
        ctk.CTkButton(self, text="Text", width=120, command=self.run_text_mode).pack(pady=10)
        ctk.CTkButton(self, text="Voice", width=120, command=self.run_voice_mode).pack(pady=10)
        ctk.CTkButton(self, text="Exit", width=120, fg_color="red", command=self.exit_app).pack(pady=10)

    def run_text_mode(self):
        self.clear_frame()

        def submit_text():
            user_input = entry.get()
            if user_input:
                result = analyze(user_input, self.models, self.vectorizer)
                self.show_result(user_input, result)

        ctk.CTkLabel(self, text="Enter your comment:").pack(pady=10)
        entry = ctk.CTkEntry(self, width=400)
        entry.pack(pady=5)
        ctk.CTkButton(self, text="Submit", command=submit_text).pack(pady=10)
        ctk.CTkButton(self, text="Back", command=self.init_main_page).pack(pady=5)

    def run_voice_mode(self):
        self.clear_frame()
        ctk.CTkLabel(self, text="🎤 Listening...", font=ctk.CTkFont(size=16)).pack(pady=10)
        self.update()

        voice_text = listen_to_voice()
        if voice_text:
            result = analyze(voice_text, self.models, self.vectorizer)
            self.show_result(voice_text, result)
        else:
            ctk.CTkLabel(self, text="Could not understand voice.", text_color="red").pack(pady=10)

        ctk.CTkButton(self, text="Back", command=self.init_main_page).pack(pady=10)

    def show_result(self, text, result):
        self.clear_frame()
        ctk.CTkLabel(self, text=f"📝 Text: {text}", wraplength=400).pack(pady=10)
        ctk.CTkLabel(self, text=f"🎯 Sentiment: {result['sentiment']} (Conf: {result['accuracy_sentiment']:.2f})").pack()
        ctk.CTkLabel(self, text=f"😄 Emotion: {result['emotion']} (Conf: {result['accuracy_emotion']:.2f})").pack()
        ctk.CTkLabel(self, text=f"🧠 Intent: {result['intent']} (Conf: {result['accuracy_intent']:.2f})").pack()
        if result["product"]:
            ctk.CTkLabel(self, text=f"📦 Predicted Product: {result['product']}").pack(pady=5)
        ctk.CTkButton(self, text="Back", command=self.init_main_page).pack(pady=15)

    def exit_app(self):
        self.clear_frame()
        ctk.CTkLabel(self, text="👋 Goodbye!", font=ctk.CTkFont(size=20)).pack(pady=60)
        self.after(2000, self.destroy)

    def clear_frame(self):
        for widget in self.winfo_children():
            widget.destroy()

# -------------------- MAIN --------------------
if __name__ == "__main__":
    app = SentimentApp()
    app.mainloop()
 