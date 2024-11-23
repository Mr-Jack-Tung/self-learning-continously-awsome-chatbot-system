# File: chat_model_features.py
# Author: Mr.Jack & ChatGPT-4o-mini
# Date: Sat 23 Nov 2024 11:12:23 GMT+7

import numpy as np
import time

class SelfLearningSystem:
    def __init__(self):
        self.model = None  # [vi] Mô hình học máy / [en] Machine learning model
        self.data_handler = None  # [vi] Xử lý dữ liệu / [en] Data handler
        self.learning_rate = 0.1  # [vi] Tốc độ học mặc định / [en] Default learning rate
        self.training_progress = 0  # [vi] Tiến độ huấn luyện / [en] Training progress

    def run_iteration(self, iteration_number):
        """
        Chạy một vòng huấn luyện với tính năng thú vị.
        [en] Run a training iteration with exciting features.
        """
        # Mô phỏng câu hỏi dựa trên sự tò mò
        curiosity_question = self._generate_curiosity_question(iteration_number)  # [vi] Tạo câu hỏi dựa trên sự tò mò / [en] Generate curiosity-driven question
        print(f"\n=====================> Iteration {iteration_number} <=====================")
        print(f"Curiosity-driven question: {curiosity_question}")  # [vi] Câu hỏi dựa trên sự tò mò / [en] Curiosity-driven question

        # Phát hiện cảm xúc từ câu hỏi
        detected_emotion = self._detect_emotion(curiosity_question)  # [vi] Phát hiện cảm xúc từ câu hỏi / [en] Detect emotion from the question
        print(f"Detected emotion: {detected_emotion} for text: {curiosity_question}")  # [vi] Cảm xúc phát hiện: {detected_emotion} cho văn bản: {curiosity_question} / [en] Detected emotion: {detected_emotion} for text: {curiosity_question}

        # Mô phỏng phản hồi với kiểu chuyên gia
        expert_response = self._generate_expert_style_response(curiosity_question)  # [vi] Tạo phản hồi theo kiểu chuyên gia / [en] Generate expert-style response
        print(f"Expert Style Response: {expert_response}")  # [vi] Phản hồi theo kiểu chuyên gia: {expert_response} / [en] Expert Style Response: {expert_response}

        # Mô phỏng thời gian chờ đợi trước khi phản hồi
        self._simulate_wait_time()  # [vi] Mô phỏng thời gian chờ đợi trước khi phản hồi / [en] Simulate wait time before responding

        # Điều chỉnh tốc độ học dựa trên kết quả huấn luyện
        self._adjust_learning_rate()  # [vi] Điều chỉnh tốc độ học dựa trên kết quả huấn luyện / [en] Adjust learning rate based on training performance

        # In tiến độ huấn luyện
        self.training_progress += 20  # [vi] Mỗi lần huấn luyện tăng 20% tiến độ / [en] Increase training progress by 20% each iteration
        print(f"Iteration {iteration_number} completed. Training Progress: {self.training_progress}%")  # [vi] Lần huấn luyện {iteration_number} hoàn thành. Tiến độ huấn luyện: {self.training_progress}% / [en] Iteration {iteration_number} completed. Training Progress: {self.training_progress}%

        # Cập nhật mô hình (giả lập quá trình huấn luyện)
        self._update_model()  # [vi] Cập nhật mô hình / [en] Update the model

        print("Waiting for the next iteration...\n")  # [vi] Đang chờ đợi cho vòng lặp tiếp theo / [en] Waiting for the next iteration...

    def _generate_curiosity_question(self, iteration_number):
        """
        Tạo câu hỏi dựa trên sự tò mò (curiosity-driven).
        [en] Generate curiosity-driven questions.
        """
        curiosity_questions = [
            "How does reinforcement learning work?",  # [vi] Cách thức học tăng cường hoạt động như thế nào? / [en] How does reinforcement learning work?
            "What is the future of artificial intelligence?",  # [vi] Tương lai của trí tuệ nhân tạo là gì? / [en] What is the future of artificial intelligence?
            "Why do neural networks learn from data?",  # [vi] Tại sao mạng nơ-ron học từ dữ liệu? / [en] Why do neural networks learn from data?
            "How can AI be used to solve real-world problems?",  # [vi] AI có thể được sử dụng như thế nào để giải quyết các vấn đề thực tế? / [en] How can AI be used to solve real-world problems?
            "What is the importance of data in machine learning?"  # [vi] Tầm quan trọng của dữ liệu trong học máy là gì? / [en] What is the importance of data in machine learning?
        ]
        return curiosity_questions[iteration_number % len(curiosity_questions)]  # [vi] Chọn câu hỏi theo số lần lặp / [en] Select question based on iteration number

    def _detect_emotion(self, text):
        """
        Phát hiện cảm xúc trong câu hỏi (giả lập).
        [en] Detect emotion in the question (simulated).
        """
        emotions = ['positive', 'neutral', 'negative']  # [vi] Các cảm xúc có thể phát hiện / [en] Possible emotions to detect
        # Giả lập phát hiện cảm xúc dựa trên các từ khóa
        if "why" in text.lower() or "how" in text.lower():  # [vi] Nếu có "why" hoặc "how" trong câu hỏi / [en] If the question contains "why" or "how"
            return emotions[1]  # [vi] Cảm xúc trung lập / [en] Neutral emotion
        return emotions[np.random.randint(0, 3)]  # [vi] Phản hồi cảm xúc ngẫu nhiên / [en] Return a random emotion

    def _generate_expert_style_response(self, question):
        """
        Tạo phản hồi theo kiểu chuyên gia.
        [en] Generate expert-style response.
        """
        response = f"From my experience, the best approach would be... {question}"  # [vi] Dựa trên kinh nghiệm của tôi, cách tiếp cận tốt nhất là... {question} / [en] From my experience, the best approach would be... {question}
        return response

    def _simulate_wait_time(self):
        """
        Mô phỏng thời gian chờ đợi trước khi phản hồi.
        [en] Simulate wait time before responding.
        """
        wait_time = np.random.uniform(1.0, 2.5)  # [vi] Mô phỏng thời gian chờ từ 1 đến 2.5 giây / [en] Simulate wait time between 1 to 2.5 seconds
        print(f"Waiting for {wait_time:.2f} seconds before responding...")  # [vi] Đang chờ {wait_time:.2f} giây trước khi phản hồi / [en] Waiting for {wait_time:.2f} seconds before responding...
        time.sleep(wait_time)

    def _adjust_learning_rate(self):
        """
        Điều chỉnh tốc độ học dựa trên kết quả huấn luyện.
        [en] Adjust the learning rate based on training performance.
        """
        if self.training_progress < 50:  # [vi] Nếu tiến độ huấn luyện nhỏ hơn 50% / [en] If training progress is less than 50%
            self.learning_rate = 0.1
        elif self.training_progress < 80:  # [vi] Nếu tiến độ huấn luyện nhỏ hơn 80% / [en] If training progress is less than 80%
            self.learning_rate = 0.05
        else:  # [vi] Nếu tiến độ huấn luyện đạt 80% trở lên / [en] If training progress is 80% or more
            self.learning_rate = 0.02
        print(f"Adjusting learning rate to {self.learning_rate} based on current training performance.")  # [vi] Điều chỉnh tốc độ học thành {self.learning_rate} dựa trên kết quả huấn luyện / [en] Adjusting learning rate to {self.learning_rate} based on current training performance.

    def _update_model(self):
        """
        Cập nhật mô hình sau mỗi iteration (giả lập quá trình huấn luyện).
        [en] Update the model after each iteration (simulated training process).
        """
        print(f"Updating model weights with current training data...")  # [vi] Cập nhật trọng số mô hình với dữ liệu huấn luyện hiện tại / [en] Updating model weights with current training data...
        # Giả lập quá trình cập nhật trọng số mô hình
        model_weights_update = np.random.rand(10) * self.learning_rate  # [vi] Cập nhật trọng số mô hình bằng cách giảm nhẹ / [en] Update model weights by a small amount
        print(f"Model weights updated: {model_weights_update[:5]}...")  # [vi] Trọng số mô hình đã được cập nhật: {model_weights_update[:5]}... / [en] Model weights updated: {model_weights_update[:5]}...
