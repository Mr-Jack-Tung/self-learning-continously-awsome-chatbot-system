# File: chat_model.py
# Author: Mr.Jack & ChatGPT-4o-mini
# Date: Sat 23 Nov 2024 11:12:23 GMT+7

import os
import random
import numpy as np
import pickle
import requests
from datasets import load_dataset

class ChatModel:
    def __init__(self, model_path=None):
        if model_path and os.path.exists(model_path):
            self.load(model_path)
        else:
            self.weights = np.random.rand(10)  # [vi] Ví dụ có 10 tham số trọng số / [en] Example: Initialize with 10 weight parameters
        self.data_handler = DataHandler()  # [vi] Không cần import từ file khác nữa / [en] No need to import from another file anymore

    def predict(self, input_text):
        # [vi] Dự đoán phản hồi từ mô hình / [en] Predict the response from the model
        response = f"Response to: {input_text}"  # [vi] Phản hồi đơn giản / [en] Simple response
        return response
    
    def finetune(self, feedback_data):
        # [vi] Cập nhật trọng số mô hình / [en] Update model weights
        learning_rate = 0.01
        self.weights -= learning_rate * np.array(feedback_data)  # [vi] Cập nhật trọng số / [en] Update weights

    def save(self, model_path):
        # [vi] Lưu trạng thái mô hình vào file / [en] Save model state to file
        with open(model_path, 'wb') as f:
            pickle.dump(self.weights, f)
        print(f"Model saved to {model_path}")
    
    def load(self, model_path):
        # [vi] Tải trọng số mô hình từ file / [en] Load model weights from file
        with open(model_path, 'rb') as f:
            self.weights = pickle.load(f)
        print(f"Model loaded from {model_path}")

    def train_with_data(self, data):
        # [vi] Huấn luyện mô hình với dữ liệu được cung cấp / [en] Train model with provided data
        if isinstance(data, list):  # [vi] Dữ liệu là danh sách câu hỏi/đáp án / [en] Data is a list of question-answer pairs
            for item in data:
                print(f"Training with data: {item}")
                feedback = np.random.rand(10)  # [vi] Giả lập phản hồi cho việc fine-tuning / [en] Simulate feedback for fine-tuning
                self.finetune(feedback)
        elif isinstance(data, dict):  # [vi] Dữ liệu từ Hugging Face / [en] Data is from Hugging Face
            for key, value in data.items():
                print(f"Training with dataset: {key} - Sample: {value[0]}")
                feedback = np.random.rand(10)  # [vi] Giả lập phản hồi cho việc fine-tuning / [en] Simulate feedback for fine-tuning
                self.finetune(feedback)

class DataHandler:
    def __init__(self):
        pass

    def generate_synthetic_data(self):
        """
        [vi] Tạo dữ liệu tổng hợp (synthetic data) dựa trên kiến thức hiện tại của mô hình.
        [en] Generate synthetic data based on the current knowledge of the model.
        """
        questions = [
            "What is your name?",
            "How does AI work?",
            "What is machine learning?",
            "Tell me about neural networks.",
            "How can I use this model?"
        ]
        
        answers = [
            "I am a model created by OpenAI.",
            "AI works by simulating human intelligence through algorithms.",
            "Machine learning is a subset of AI that focuses on learning from data.",
            "Neural networks are a type of machine learning model inspired by the human brain.",
            "You can use me by asking questions or providing inputs for me to respond to."
        ]
        
        idx = np.random.randint(0, len(questions))
        synthetic_question = questions[idx]
        synthetic_answer = answers[idx]
        
        print(f"Synthetic Data Generated: {synthetic_question} -> {synthetic_answer}")
        return synthetic_question, synthetic_answer

    def fetch_data_from_internet(self, topic):
        """
        [vi] Lấy dữ liệu từ Internet (Wikipedia API ví dụ).
        [en] Fetch data from the Internet (e.g., Wikipedia API).
        """
        url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro&titles={topic}&format=json"
        response = requests.get(url)
        data = response.json()
        
        try:
            page = next(iter(data['query']['pages'].values()))
            text = page['extract']
            print(f"Fetched Data for {topic}: {[text[:200]]}...")  # [vi] Chỉ in ra 200 ký tự đầu tiên / [en] Only print the first 200 characters
            return text
        except KeyError:
            print(f"Error fetching data for {topic}.")
            return None

    def load_data_from_file(self, file_path):
        """
        [vi] Tải dữ liệu từ file của người dùng (file .txt hoặc .csv).
        [en] Load data from user-provided file (.txt or .csv).
        """
        if not os.path.exists(file_path):
            print(f"File {file_path} not found.")
            return None
        
        with open(file_path, 'r', encoding='utf-8') as file:
            data = file.readlines()
        
        print(f"Data loaded from {file_path}")
        return data

    def load_data_from_huggingface(self, dataset_name):
        """
        [vi] Tải dữ liệu từ Hugging Face datasets.
        [en] Load data from Hugging Face datasets.
        """
        try:
            dataset = load_dataset(dataset_name)
            print(f"Loaded dataset {dataset_name} from Hugging Face.")
            return dataset
        except Exception as e:
            print(f"Error loading dataset from Hugging Face: {e}")
            return None
