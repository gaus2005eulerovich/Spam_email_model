import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import torch.nn.functional as F
import os
import imaplib
import email
from email.header import decode_header

# --- КОНФИГУРАЦИЯ ---
MODEL_PATH = "./my_spam_model"

# Настройка страницы
st.set_page_config(page_title="Spam Detective", page_icon="📧", layout="centered")


# --- ЛОГИКА МОДЕЛИ ---
@st.cache_resource
def load_model():
    current_dir = os.getcwd()
    full_path = os.path.join(current_dir, MODEL_PATH)

    if not os.path.exists(MODEL_PATH):
        st.error(f"⚠️ ОШИБКА: Папка с моделью не найдена!")
        st.warning(f"Я ищу папку тут: `{full_path}`")
        return None, None

    try:
        tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
        return tokenizer, model
    except Exception as e:
        st.error(f"Ошибка загрузки файлов модели: {e}")
        return None, None


def predict_spam(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = F.softmax(logits, dim=1)
    pred_idx = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred_idx].item()
    label_map = model.config.id2label
    pred_label = label_map.get(pred_idx, "UNKNOWN")
    return pred_label, confidence


# --- ЛОГИКА GMAIL ---
def clean_text(text):
    # Простая очистка декодированных заголовков
    if isinstance(text, bytes):
        try:
            return text.decode('utf-8')
        except:
            return text.decode('latin-1')
    return str(text)


def get_last_emails(username, password, num_emails=5):
    """
    Подключается к Gmail по IMAP и забирает последние N писем.
    """
    mail = imaplib.IMAP4_SSL("imap.gmail.com")
    try:
        mail.login(username, password)
    except imaplib.IMAP4.error:
        return None, "Ошибка авторизации! Проверьте Email и App Password."

    mail.select("inbox")

    # Ищем все письма
    status, messages = mail.search(None, "ALL")
    email_ids = messages[0].split()

    # Берем последние N
    latest_email_ids = email_ids[-num_emails:]

    emails_data = []

    for e_id in reversed(latest_email_ids):
        _, msg_data = mail.fetch(e_id, "(RFC822)")
        for response_part in msg_data:
            if isinstance(response_part, tuple):
                msg = email.message_from_bytes(response_part[1])

                # Декодируем тему
                subject, encoding = decode_header(msg["Subject"])[0]
                if isinstance(subject, bytes):
                    subject = subject.decode(encoding if encoding else "utf-8", errors="ignore")

                # Декодируем отправителя
                sender = msg.get("From")

                # Вытаскиваем текст тела (очень упрощенно, берем первый текстовый кусок)
                body = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        content_type = part.get_content_type()
                        if content_type == "text/plain":
                            try:
                                body = part.get_payload(decode=True).decode()
                            except:
                                pass
                            break
                else:
                    try:
                        body = msg.get_payload(decode=True).decode()
                    except:
                        pass

                emails_data.append({
                    "subject": subject,
                    "sender": sender,
                    "body": body
                })

    mail.close()
    mail.logout()
    return emails_data, None


# --- ИНТЕРФЕЙС (UI) ---
st.title("📧 AI Spam Filter System")
st.caption("Universal Spam Detection Interface")

tokenizer, model = load_model()

if tokenizer and model:
    # Вкладки для переключения режимов
    tab1, tab2 = st.tabs(["✍️ Ручная проверка", "📬 Gmail Connect"])

    with tab1:
        st.header("Проверка текста")
        st.info("ℹ️ Constraint: Модель работает лучше всего на английском.")
        text_input = st.text_area("Текст письма:", height=150)

        if st.button("Проверить", key="manual_btn"):
            if text_input:
                label, conf = predict_spam(text_input, tokenizer, model)

                is_spam = False
                if isinstance(label, int):
                    is_spam = (label == 1)
                elif "SPAM" in str(label).upper():
                    is_spam = True
                elif str(label) == "LABEL_1":
                    is_spam = True

                if is_spam:
                    st.error(f"🚨 SPAM DETECTED ({conf:.1%})")
                else:
                    st.success(f"✅ HAM / CLEAN ({conf:.1%})")
            else:
                st.warning("Введите текст!")

    with tab2:
        st.header("Интеграция с Gmail")
        st.markdown("""
        1. Используйте **App Password**, а не обычный пароль! ([Инструкция](https://support.google.com/accounts/answer/185833))
        2. Включите IMAP в настройках Gmail.
        """)

        col_auth1, col_auth2 = st.columns(2)
        with col_auth1:
            email_user = st.text_input("Gmail Address")
        with col_auth2:
            email_pass = st.text_input("App Password", type="password")

        limit = st.slider("Сколько писем проверить?", 1, 10, 3)

        if st.button("📥 Проверить ящик"):
            if email_user and email_pass:
                with st.spinner("Подключаемся к Gmail..."):
                    emails, error = get_last_emails(email_user, email_pass, limit)

                if error:
                    st.error(error)
                else:
                    st.success(f"Загружено {len(emails)} писем!")
                    for i, mail in enumerate(emails):
                        with st.expander(f"📩 {mail['subject']}"):
                            st.write(f"**From:** {mail['sender']}")
                            st.text_area("Body preview:", mail['body'][:200] + "...", height=80, key=f"body_{i}")

                            # АНАЛИЗ
                            if mail['body']:
                                label, conf = predict_spam(mail['body'], tokenizer, model)

                                is_spam = False
                                if isinstance(label, int):
                                    is_spam = (label == 1)
                                elif "SPAM" in str(label).upper():
                                    is_spam = True
                                elif str(label) == "LABEL_1":
                                    is_spam = True

                                if is_spam:
                                    st.error(f"🚨 **SPAM** (Вероятность: {conf:.1%})")
                                else:
                                    st.success(f"✅ **CLEAN** (Вероятность: {conf:.1%})")
                            else:
                                st.warning("Не удалось прочитать текст письма (возможно, только HTML).")
            else:
                st.warning("Введите данные для входа!")