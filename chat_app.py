# File: chat_app.py
# Author: Mr.Jack & ChatGPT-4o-mini
# Date: Sat 23 Nov 2024 11:12:23 GMT+7

import numpy as np
import time
from chat_model import ChatModel
from chat_model_features import SelfLearningSystem

def chat_and_finetune(model, user_input, fetch_online_data=False, file_data=None, huggingface_data=None):
    print("\n=====================> MODEL TRAINING IN PROGRESS <=====================")
    time.sleep(1)

    print(f"\nStep 1: {user_input} - Initiating Inference...")  # [vi] Bước 1: {user_input} - Khởi tạo suy luận / [en] Step 1: {user_input} - Initiating Inference...
    response = model.predict(user_input)
    print(f"Model Response: {response}")  # [vi] Phản hồi từ mô hình / [en] Model response
    
    print("\nStep 2: Generating Synthetic Data...")  # [vi] Bước 2: Tạo dữ liệu tổng hợp / [en] Step 2: Generating Synthetic Data...
    synthetic_question, synthetic_answer = model.data_handler.generate_synthetic_data()
    feedback = np.random.rand(10)  # [vi] Giả lập phản hồi cho việc fine-tuning / [en] Simulate feedback for fine-tuning
    model.finetune(feedback)
    print("Model weights updated based on synthetic data.")  # [vi] Trọng số mô hình đã được cập nhật dựa trên dữ liệu tổng hợp / [en] Model weights updated based on synthetic data.

    if fetch_online_data:
        print("\nStep 3: Fetching Data from the Internet...")  # [vi] Bước 3: Lấy dữ liệu từ Internet / [en] Step 3: Fetching data from the Internet...
        online_data = model.data_handler.fetch_data_from_internet(user_input)
        if online_data:
            model.finetune(np.random.rand(10))
            print("Model updated with online data.")  # [vi] Mô hình đã được cập nhật với dữ liệu từ internet / [en] Model updated with online data.
    
    if file_data:
        print("\nStep 4: Loading and Training with User's File Data...")  # [vi] Bước 4: Tải và huấn luyện với dữ liệu từ file của người dùng / [en] Step 4: Loading and training with user's file data...
        file_data = model.data_handler.load_data_from_file(file_data)
        if file_data:
            model.train_with_data(file_data)
    
    if huggingface_data:
        print("\nStep 5: Loading and Training with Hugging Face Data...")  # [vi] Bước 5: Tải và huấn luyện với dữ liệu Hugging Face / [en] Step 5: Loading and training with Hugging Face data...
        huggingface_data = model.data_handler.load_data_from_huggingface(huggingface_data)
        if huggingface_data:
            model.train_with_data(huggingface_data)

    print("\nStep 6: Collecting Human Feedback and Fine-tuning...")  # [vi] Bước 6: Thu thập phản hồi từ người dùng và fine-tuning / [en] Step 6: Collecting human feedback and fine-tuning...
    user_feedback = np.random.rand(10)  # [vi] Giả lập phản hồi từ người dùng / [en] Simulate user feedback
    model.finetune(user_feedback)
    print("Model weights updated based on human feedback.")  # [vi] Trọng số mô hình đã được cập nhật dựa trên phản hồi của người dùng / [en] Model weights updated based on human feedback.
    print("\n=======================================================================\n")
    
    return response

def self_learning_cycle(model, num_iterations=5, model_path='chat_model.pkl'):
    # [vi] Khởi tạo mô hình (hoặc tải từ file nếu có) / [en] Initialize the model (or load from file if available)
    model.load(model_path)

    # [vi] Biến để lưu thời gian bắt đầu / [en] Variable to track the start time
    total_start_time = time.time()

    for i in range(num_iterations):
        start_time = time.time()  # [vi] Thời gian bắt đầu của mỗi iteration / [en] Start time for each iteration
        print(f"\n=====================> Iteration {i+1} <=====================")
        
        user_input = f"Question {i + 1}"  # [vi] Giả lập câu hỏi từ người dùng / [en] Simulate user input question
        fetch_online_data = (i % 2 == 0)  # [vi] Mô phỏng việc lấy dữ liệu từ Internet mỗi lần chẵn / [en] Simulate fetching data from the Internet every even iteration
        file_data = 'user_input_data.txt' if i % 2 == 0 else None  # [vi] Tải dữ liệu từ file người dùng nếu cần / [en] Load user file data if necessary
        huggingface_data = 'ag_news' if i % 2 != 0 else None  # [vi] Dữ liệu Hugging Face nếu cần / [en] Use Hugging Face data if necessary
        
        chat_and_finetune(model, user_input, fetch_online_data, file_data, huggingface_data)

        iteration_time = time.time() - start_time
        print(f"Iteration {i+1} completed in {iteration_time:.2f} seconds.")  # [vi] Lần huấn luyện {i+1} hoàn thành trong {iteration_time:.2f} giây / [en] Iteration {i+1} completed in {iteration_time:.2f} seconds.

        model.save(model_path)

    total_time = time.time() - total_start_time
    print("\n=====================> Training Completed <=====================")
    print(f"Total training time: {total_time:.2f} seconds")  # [vi] Tổng thời gian huấn luyện: {total_time:.2f} giây / [en] Total training time: {total_time:.2f} seconds
    print(f"Final model weights after {num_iterations} iterations:", model.weights)

if __name__ == "__main__":
    # [vi] Khởi tạo hệ thống học tự động / [en] Initialize self-learning system
    self_learning_system = SelfLearningSystem()

    # [vi] Chạy quá trình mô phỏng với tự học và tính năng thú vị / [en] Run self-learning process with exciting features
    for i in range(5):
        self_learning_system.run_iteration(i + 1)

    # [vi] Chạy quá trình tự học / [en] Run self-learning cycle
    self_learning_cycle(ChatModel(), num_iterations=5)
